"""RGT model (docs/methods/rgt.md section 4): RGTDetector of the TwiBot-22 benchmark re-implementation.

    n = Drop(LReLU(Linear(k, H/4)(num)))      b = Drop(LReLU(Linear(c, H/4)(cat)))
    t = Drop(LReLU(Linear(768, H/4)(tweet)))  d = Drop(LReLU(Linear(768, H/4)(des)))
    [j = Drop(LReLU(Linear(2, H/4)(extscore)))]                         (ExtScore "branch" only)
    x0 = Drop(LReLU(Linear(4H/4 [+H/4], H)(cat[n, b, t, d (, j)])))
    x1 = LReLU(RGTLayer(x0)); x2 = LReLU(RGTLayer(x1))             (no dropout after the layers)
    y  = Linear(64, 2)(Drop(LReLU(Linear(H [+2 late ExtScore], 64)(x2))))

RGTLayer: per relation r a TransformerConv(H, H, heads=C, concat=False, dropout=p) (root/skip term on,
heads averaged), one gate shared by all relations: a = sigmoid(W_A [u; x]), h_r = tanh(u) * a +
x * (1 - a); then HAN-style semantic attention with D heads whose per-relation weight is the mean
score over all nodes of the (sub)graph. Unlike the benchmark ``layer.py`` (early ``return`` inside
the relation loop) every relation is used for any R.

Parameter creation order follows the benchmark (num, bool, tweet, des, [extscore], linear1, RGT1, RGT2,
out1, out2; inside a layer: gate, transformers, semantic attention). Init: kaiming_uniform_ weights
and zero bias on every ``torch.nn.Linear``; the TransformerConv projections are
``torch_geometric.nn.Linear`` (not a subclass of ``nn.Linear``, checked under PyG 2.8.1) and keep
PyG's own init, as in the benchmark.

The model takes a ``bb.gnn.GraphInputs`` and an optional ``rows`` selection: the head (purely
per-node) then runs only on those rows. This is exact and saves the head on the non-loss rows.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as Fn
from torch import nn
from torch_geometric.nn import TransformerConv


class SemanticAttention(nn.Module):
    """``heads`` independent scorers Linear(H, hidden) -> tanh -> Linear(hidden, 1, no bias); the
    per-relation weight is softmax_r(mean over nodes of the score); output = mean over heads of
    sum_r beta_r z_r.  z: [N, R, H] -> [N, H]."""

    def __init__(self, in_dim: int, heads: int, hidden: int = 128):
        super().__init__()
        self.scorers = nn.ModuleList(
            nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1, bias=False))
            for _ in range(heads))

    def betas(self, z: torch.Tensor) -> torch.Tensor:
        """[heads, R] relation weights."""
        return torch.stack([torch.softmax(s(z).float().mean(0), dim=0)[:, 0] for s in self.scorers])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = None
        for s in self.scorers:
            beta = torch.softmax(s(z).float().mean(0), dim=0)              # [R, 1]
            o = (beta.to(z.dtype).unsqueeze(0) * z).sum(1)                # [N, H]
            out = o if out is None else out + o
        return out / len(self.scorers)


class EdgeCtx:
    """Per-graph edge bookkeeping shared by both layers. Keys/values are only needed for edge
    sources ``S`` and queries only for edge targets ``T`` (TwiBot-20: about 12k of 229,580 nodes
    have out-edges; TwiBot-22 subgraphs: the same few thousand core users), so the projections run
    on those rows only; the root/skip term runs on every node. Segments are (target, relation)."""

    def __init__(self, edge_index: torch.Tensor, edge_type: torch.Tensor, n: int, R: int, dense: bool = False):
        self.n, self.R = n, R
        if dense:   # every node is a potential source/target: no torch.unique (sync-free, CUDA-graph safe)
            self.S = self.T = None
            self.k_idx = edge_index[0] * R + edge_type
            self.q_idx = edge_index[1] * R + edge_type
            self.n_seg = n * R
            return
        self.S, s_loc = torch.unique(edge_index[0], return_inverse=True)
        self.T, t_loc = torch.unique(edge_index[1], return_inverse=True)
        self.k_idx = s_loc * R + edge_type          # rows of K / V   ([|S| * R])
        self.q_idx = t_loc * R + edge_type          # rows of Q = softmax/aggregation segments ([|T| * R])
        self.n_seg = len(self.T) * R

    def plan(self, C: int):
        """Index plan of ``edge_path='triton'`` (``RGTLayer._edges_triton``), built once per graph (the
        host syncs of sort / bincount / unique happen here only).

        Multi-edge segments: a (target, relation) segment with ONE in-edge has softmax weight exactly
        1 (``exp(s - s) / (1 + 1e-16)`` = 1 in float) and zero gradient w.r.t. its score, so queries
        and scores are only needed for the edges of segments with >= 2 in-edges (TwiBot-20: most of
        the 204k targets have one in-edge per relation). ``me``: those edge ids; ``segs``: their
        segments (sorted), ``m_loc``: edge -> row of ``segs``.

        Head-expanded CSR for the aggregation ``agg[q] = sum_{e in q} sum_c a'_ec V[k_e, c]`` as one
        ``seg_spmm`` with a single "head" over the rows of ``V.view(nK * C, 1, H)``: entries j = (e, c)
        sorted by segment (``ptr``, ``col`` = k_e * C + c, ``w_at`` = e * C + c) and, for the backward
        dV, sorted by V row (``ptr_v``, ``seg_v``, ``w_at_v``); ``ia``/``ib`` = (q_e, k_e * C + c) in
        natural (e, c) order for the d(a') ``sddmm``."""
        key = (C, )
        if getattr(self, "_plan_key", None) != key:
            q, k, dev = self.q_idx, self.k_idx, self.q_idx.device
            cnt = torch.bincount(q, minlength=self.n_seg)
            me = torch.nonzero(cnt[q] > 1).squeeze(1)
            segs, m_loc = torch.unique(q[me], return_inverse=True)
            n_v = len(self.S) * self.R * C
            cc = torch.arange(C, device=dev)
            ia = q.repeat_interleave(C)                                   # [E * C], j = e * C + c
            ib = (k.unsqueeze(1) * C + cc).reshape(-1)
            order = torch.argsort(ia, stable=True)
            ptr = torch.zeros(self.n_seg + 1, dtype=torch.long, device=dev)
            ptr[1:] = torch.cumsum(cnt * C, 0)
            order_v = torch.argsort(ib, stable=True)
            ptr_v = torch.zeros(n_v + 1, dtype=torch.long, device=dev)
            ptr_v[1:] = torch.cumsum(torch.bincount(ib, minlength=n_v), 0)
            self._plan = {"me": me, "segs": segs, "m_loc": m_loc, "ia": ia, "ib": ib,
                          "ptr": ptr, "col": ib[order].contiguous(), "w_at": order,
                          "ptr_v": ptr_v, "seg_v": ia[order_v].contiguous(), "w_at_v": order_v}
            self._plan_key = key
        return self._plan


class _SegAggregate(torch.autograd.Function):
    """agg[q, :] = sum_{e in segment q} sum_c w[e, c] V[k_e, c, :]   ([n_seg, H], fp32), with the CSR
    gather kernels of ``bb.methods.botmoe.edge_triton`` (review round 2): no [E, C, H] message tensor in
    forward or backward and no atomics. Backward: dV = the same CSR sum over V rows of w * dagg[q_e]
    (``seg_spmm``), dw[e, c] = <dagg[q_e], V[k_e, c]> (``sddmm``). fp32 in and out."""

    @staticmethod
    def forward(ctx, w, V, plan, n_seg: int):
        from ..botmoe.edge_triton import seg_spmm
        nV, H = V.shape[0] * V.shape[1], V.shape[2]
        Vf = V.reshape(nV, 1, H)
        wf = w.reshape(-1, 1)
        out = seg_spmm(plan["ptr"], plan["col"], wf[plan["w_at"]], Vf, n_seg)        # [n_seg, 1, H]
        ctx.save_for_backward(w, V)
        ctx.plan = plan
        return out.view(n_seg, H)

    @staticmethod
    def backward(ctx, dout):
        from ..botmoe.edge_triton import sddmm, seg_spmm
        w, V = ctx.saved_tensors
        pl = ctx.plan
        nV, H = V.shape[0] * V.shape[1], V.shape[2]
        X = dout.contiguous().view(-1, 1, H)
        Vf = V.reshape(nV, 1, H)
        dV = seg_spmm(pl["ptr_v"], pl["seg_v"], w.reshape(-1, 1)[pl["w_at_v"]], X, nV).view(V.shape)
        dw = sddmm(X, Vf, pl["ia"], pl["ib"]).view(w.shape)
        return dw, dV, None, None


def _gate_semantic(u, x, gw, gb, w1, b1, w2):
    """Shared gate + semantic attention of the fused layer as one function of (u, x) (the ops of
    ``_fused_gate`` and ``_fused_semantic``); ``torch.compile``d for full-graph training."""
    H = x.shape[1]
    a = torch.sigmoid(Fn.linear(u, gw[:, :H]) + Fn.linear(x, gw[:, H:], gb).unsqueeze(1))
    z = torch.tanh(u) * a + x.unsqueeze(1) * (1 - a)
    D = w2.shape[0]
    t = torch.tanh(Fn.linear(z, w1, b1)).view(*z.shape[:2], D, -1)
    s = torch.einsum("nrdh,dh->nrd", t, w2.to(t.dtype)).float()
    coef = torch.softmax(s.mean(0), dim=0).mean(1)
    return (z * coef.to(z.dtype).view(1, -1, 1)).sum(1)


_COMPILED = {}


def _compiled_gate_semantic():
    if "f" not in _COMPILED:
        import os
        if "TORCHINDUCTOR_COMPILE_THREADS" not in os.environ:   # no worker pool: stays inside memrun caps
            import torch._inductor.config as ic
            ic.compile_threads = 1
        _COMPILED["f"] = torch.compile(_gate_semantic, dynamic=False)
    return _COMPILED["f"]


class RGTLayer(nn.Module):
    """``fused=True`` (default) computes all relations at once: one GEMM each for every relation's
    queries (edge targets only), keys/values (edge sources only) and skip terms, one segment softmax
    over (target, relation) groups, one scatter-add, the shared gate and the D semantic scorers as
    single batched ops. It is the same
    function as the per-relation PyG path (``fused=False``, kept as the reference; tests check
    equality to 1e-5); only the RNG stream of the attention dropout differs. Parameters live in the
    PyG ``TransformerConv`` / ``nn.Linear`` modules either way (same order, init and state_dict).

    Exact fast paths of the fused layer (review round 2, used for full-graph training on CUDA):
    ``edge_path="triton"`` (``_edges_triton``: queries / scores for multi-edge segments only, CSR
    aggregation with BotMoE's gather kernels) and ``tail_path="compile"`` (``_gate_semantic`` under
    torch.compile)."""

    def __init__(self, num_relations: int, hidden: int, trans_heads: int, sem_heads: int, dropout: float,
                 sem_hidden: int = 128, fused: bool = True):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Sigmoid())   # one gate for all relations
        self.convs = nn.ModuleList(
            TransformerConv(hidden, hidden, heads=trans_heads, concat=False, beta=False, dropout=dropout,
                            root_weight=True, bias=True)
            for _ in range(num_relations))
        self.semantic = SemanticAttention(hidden, sem_heads, sem_hidden)
        self.R, self.C, self.H, self.p = num_relations, trans_heads, hidden, dropout
        self.fused = fused
        # efficiency switches of the fused path (review round 2; same function). Defaults are the
        # plain torch ops (BotMoE's ChunkedRGTLayer inherits them); RGTDetector.set_impl switches them.
        self.edge_path = "torch"      # torch | triton (CUDA, non-dense ctx: _edges_triton / _SegAggregate)
        self.tail_path = "eager"      # eager | compile (torch.compile'd gate + semantic attention)

    # -- reference path: per-relation PyG TransformerConv, as the benchmark layer.py
    def relation_outputs(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        hs = []
        for r, conv in enumerate(self.convs):
            u = conv(x, edge_index[:, edge_type == r])
            a = self.gate(torch.cat((u, x), dim=1))
            hs.append(torch.tanh(u) * a + x * (1 - a))
        return torch.stack(hs, dim=1)                                    # [N, R, H]

    # -- fused path
    def _fused_relation_outputs(self, x: torch.Tensor, ctx: EdgeCtx) -> torch.Tensor:
        return self._fused_gate(self._fused_u(x, ctx), x)

    def _fused_gate(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        H = self.H
        gw, gb = self.gate[0].weight, self.gate[0].bias                               # W_A [u ; x]
        a = torch.sigmoid(Fn.linear(u, gw[:, :H]) + Fn.linear(x, gw[:, H:], gb).unsqueeze(1))
        return torch.tanh(u) * a + x.unsqueeze(1) * (1 - a)                           # [N, R, H]

    def _fused_u(self, x: torch.Tensor, ctx: EdgeCtx) -> torch.Tensor:
        """u_r = skip_r(x) + (1/C) sum_c sum_{j -> i in r} alpha_ijc v_jc   ([N, R, H])."""
        from torch_geometric.utils import softmax as seg_softmax
        n, R, C, H = x.shape[0], self.R, self.C, self.H
        cv = self.convs
        wq = torch.cat([c.lin_query.weight for c in cv], 0)                      # [R*CH, H]
        bq = torch.cat([c.lin_query.bias for c in cv], 0)
        wkv = torch.cat([w for c in cv for w in (c.lin_key.weight, c.lin_value.weight)], 0)   # per r: [k | v]
        bkv = torch.cat([b for c in cv for b in (c.lin_key.bias, c.lin_value.bias)], 0)
        ws = torch.cat([c.lin_skip.weight for c in cv], 0)                       # [R*H, H]
        bs = torch.cat([c.lin_skip.bias for c in cv], 0)
        dense = ctx.T is None
        nS, nT = (n, n) if dense else (len(ctx.S), len(ctx.T))
        KV = Fn.linear(x if dense else x[ctx.S], wkv, bkv).view(nS * R, 2, C, H)
        K, V = KV[:, 0], KV[:, 1]
        if self.edge_path == "triton" and not dense and x.is_cuda:
            agg = self._edges_triton(x, ctx, wq, bq, K, V)
        else:
            Q = Fn.linear(x if dense else x[ctx.T], wq, bq).view(nT * R, C, H)
            alpha = (Q[ctx.q_idx] * K[ctx.k_idx]).sum(-1) / math.sqrt(H)                # [E, C]
            alpha = seg_softmax(alpha, ctx.q_idx, num_nodes=ctx.n_seg)
            alpha = Fn.dropout(alpha, p=self.p, training=self.training)
            # messages and their per-target sum in fp32 even under fp16 autocast, as PyG does (fp16
            # value x fp32 alpha -> fp32, then scatter): an fp16 running sum stalls on high in-degree
            # targets (review round 1: max error 0.6 at in-degree 50k, 2e-2 at 2k; PyG 2e-3). Heads
            # are summed per edge first, so the accumulator is [segments, H] rather than [segments, C, H].
            msg = (V.float()[ctx.k_idx] * alpha.unsqueeze(-1)).sum(1)                  # [E, H] fp32
            agg = torch.zeros((ctx.n_seg, H), dtype=msg.dtype, device=msg.device).index_add_(0, ctx.q_idx, msg)
        u = Fn.linear(x, ws, bs).view(n, R, H)                                        # root/skip, every node
        m = (agg / C).view(nT, R, H).to(u.dtype)                                      # head-averaged messages
        return u + m if dense else u.index_add(0, ctx.T, m)

    def _edges_triton(self, x, ctx: EdgeCtx, wq, bq, K, V) -> torch.Tensor:
        """The torch path's ``agg`` ([n_seg, H] fp32, sum over heads and in-edges of alpha * V) with
        queries and scores for multi-edge segments only and the CSR aggregation (``EdgeCtx.plan``).
        The attention dropout is drawn on the same full [E, C] fp32 alpha tensor as in the torch
        path, so the masks are the same RNG draws; singleton segments have alpha = 1 exactly."""
        from torch_geometric.utils import softmax as seg_softmax
        R, C, H = self.R, self.C, self.H
        pl = ctx.plan(C)
        me, segs = pl["me"], pl["segs"]
        nM = len(segs)
        Qm = Fn.linear(x[ctx.T[segs // R]], wq, bq).view(nM, R, C, H)
        Qm = Qm[torch.arange(nM, device=x.device), segs % R]                          # [nM, C, H]
        s = (Qm[pl["m_loc"]] * K[ctx.k_idx[me]]).sum(-1) / math.sqrt(H)               # [E_multi, C]
        s = seg_softmax(s, pl["m_loc"], num_nodes=nM)
        E = ctx.q_idx.shape[0]
        alpha = torch.ones((E, C), dtype=s.dtype, device=x.device).index_put((me,), s)
        alpha = Fn.dropout(alpha, p=self.p, training=self.training)
        return _SegAggregate.apply(alpha.float(), V.float(), pl, ctx.n_seg)

    def _fused_semantic(self, z: torch.Tensor, node_mask: torch.Tensor | None = None) -> torch.Tensor:
        sc = self.semantic.scorers
        D = len(sc)
        w1 = torch.cat([s[0].weight for s in sc], 0)
        b1 = torch.cat([s[0].bias for s in sc], 0)
        w2 = torch.cat([s[2].weight for s in sc], 0)                                   # [D, hid]
        t = torch.tanh(Fn.linear(z, w1, b1)).view(*z.shape[:2], D, -1)                # [N, R, D, hid]
        s = torch.einsum("nrdh,dh->nrd", t, w2.to(t.dtype))
        s = s.float()
        if node_mask is None:
            w = s.mean(0)
        else:       # padded subgraph: mean over the real nodes only
            w = (s * node_mask.view(-1, 1, 1)).sum(0) / node_mask.sum()
        beta = torch.softmax(w, dim=0)                                                 # [R, D]
        coef = beta.mean(1)                                                            # (1/D) sum_d beta_rd
        return (z * coef.to(z.dtype).view(1, -1, 1)).sum(1)

    def forward(self, x, edge_index, edge_type, ctx: EdgeCtx | None = None, node_mask=None):
        if not self.fused:
            if node_mask is not None:
                raise ValueError("node_mask needs the fused path")
            return self.semantic(self.relation_outputs(x, edge_index, edge_type))
        if ctx is None:
            ctx = EdgeCtx(edge_index, edge_type, x.shape[0], self.R)
        if self.tail_path == "compile" and node_mask is None and x.is_cuda:
            sc = self.semantic.scorers
            return _compiled_gate_semantic()(
                self._fused_u(x, ctx), x, self.gate[0].weight, self.gate[0].bias,
                torch.cat([s[0].weight for s in sc], 0), torch.cat([s[0].bias for s in sc], 0),
                torch.cat([s[2].weight for s in sc], 0))
        return self._fused_semantic(self._fused_relation_outputs(x, ctx), node_mask)


class RGTDetector(nn.Module):
    def __init__(self, *, num_dim: int = 5, cat_dim: int = 3, des_dim: int = 768, tweet_dim: int = 768,
                 hidden: int = 128, trans_heads: int = 2, sem_heads: int = 2, dropout: float = 0.5,
                 num_relations: int = 2, head_hidden: int = 64, extscore_width: int = 0, extscore_inject: str = "num",
                 slope: float = 0.01, fused: bool = True):
        super().__init__()
        if hidden % 4:
            raise ValueError("hidden must be divisible by 4")
        q = hidden // 4
        self.extscore_inject = extscore_inject if (extscore_width and extscore_inject != "num") else None
        self.in_num = nn.Linear(num_dim, q)
        self.in_cat = nn.Linear(cat_dim, q)
        self.in_tweet = nn.Linear(tweet_dim, q)
        self.in_des = nn.Linear(des_dim, q)
        self.in_extscore = nn.Linear(extscore_width, q) if self.extscore_inject == "branch" else None
        self.linear1 = nn.Linear(4 * q + (q if self.in_extscore is not None else 0), hidden)
        self.rgt1 = RGTLayer(num_relations, hidden, trans_heads, sem_heads, dropout, fused=fused)
        self.rgt2 = RGTLayer(num_relations, hidden, trans_heads, sem_heads, dropout, fused=fused)
        self.out1 = nn.Linear(hidden + (extscore_width if self.extscore_inject == "late" else 0), head_hidden)
        self.out2 = nn.Linear(head_hidden, 2)
        self.drop = nn.Dropout(dropout)
        self.slope = slope
        self.checkpoint = False     # activation checkpointing of encoder and each RGT layer (same function)

    def act(self, x):
        return Fn.leaky_relu(x, self.slope)

    def set_impl(self, edge_impl: str = "torch", tail_impl: str = "eager") -> "RGTDetector":
        """Efficiency switches of both RGT layers (same function; review round 2): ``edge_impl``
        torch | triton (fused gather kernels, CUDA, non-dense ctx), ``tail_impl`` eager | compile
        (torch.compile'd gate + semantic attention, CUDA, fixed shapes: full-graph training)."""
        if edge_impl not in ("torch", "triton") or tail_impl not in ("eager", "compile"):
            raise ValueError((edge_impl, tail_impl))
        for layer in (self.rgt1, self.rgt2):
            layer.edge_path, layer.tail_path = edge_impl, tail_impl
        return self

    def encode(self, g) -> torch.Tensor:
        d, a = self.drop, self.act
        parts = [d(a(self.in_num(g.num))), d(a(self.in_cat(g.cat))), d(a(self.in_tweet(_match(g.tweet, self.in_tweet)))),
                 d(a(self.in_des(_match(g.des, self.in_des))))]
        if self.in_extscore is not None:
            if g.extra is None:
                raise ValueError("model has a ExtScore branch but GraphInputs.extra is None")
            parts.append(d(a(self.in_extscore(g.extra))))
        return d(a(self.linear1(torch.cat(parts, dim=1))))

    def embed(self, g, node_mask=None, dense: bool = False) -> torch.Tensor:
        """``dense=True`` + ``node_mask``: padded subgraph (CUDA-graph path). Padding rows must have
        no edges to real rows; ``node_mask`` (1 = real) restricts the semantic-attention mean."""
        ck = self.checkpoint and self.training and torch.is_grad_enabled()
        # g (a dataclass) is opaque to checkpoint's device scan, which would then save and restore
        # only the CPU RNG: the recomputed encoder dropout masks would differ on CUDA. Pass a device
        # tensor (g.y) so the CUDA RNG state is preserved too.
        x = _ckpt(lambda _dev: self.encode(g), g.y) if ck else self.encode(g)
        if dense:
            ctx = EdgeCtx(g.edge_index, g.edge_type, g.n, self.rgt1.R, dense=True)
        else:
            ctx = self._ctx(g) if self.rgt1.fused else None
        for layer in (self.rgt1, self.rgt2):
            f = lambda h, layer=layer: self.act(layer(h, g.edge_index, g.edge_type, ctx, node_mask))  # noqa: E731
            x = _ckpt(f, x) if ck else f(x)
        return x

    def _ctx(self, g) -> EdgeCtx:
        """Edge bookkeeping, cached while the same edge tensors are passed (full-batch training)."""
        key = getattr(self, "_ctx_key", None)   # holds references, so identity cannot be recycled
        if key is None or key[0] is not g.edge_index or key[1] is not g.edge_type or key[2] != g.n:
            self._ctx_cache = EdgeCtx(g.edge_index, g.edge_type, g.n, self.rgt1.R)
            self._ctx_key = (g.edge_index, g.edge_type, g.n)
        return self._ctx_cache

    def forward(self, g, rows=None, node_mask=None, dense: bool = False) -> torch.Tensor:
        x = self.embed(g, node_mask, dense)
        extra = g.extra
        if rows is not None:
            x = x[rows]
            extra = None if extra is None else extra[rows]
        if self.extscore_inject == "late":
            if extra is None:
                raise ValueError("model has late ExtScore fusion but GraphInputs.extra is None")
            x = torch.cat([x, extra.to(x.dtype)], dim=1)
        return self.out2(self.drop(self.act(self.out1(x))))


def _ckpt(fn, *args):
    """torch.utils.checkpoint with the RNG state preserved, so dropout masks replay identically."""
    from torch.utils.checkpoint import checkpoint
    return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=True)


def _match(x: torch.Tensor, lin: nn.Linear) -> torch.Tensor:
    """fp16 text tables: outside autocast (CPU tests, fp32 runs) cast to the weight dtype."""
    if x.dtype != lin.weight.dtype and not torch.is_autocast_enabled(x.device.type):
        return x.to(lin.weight.dtype)
    return x


def init_weights(model: nn.Module) -> nn.Module:
    """Benchmark ``init_weight``: kaiming_uniform_ weights and zero bias on every ``torch.nn.Linear``
    (gate, semantic scorers, input projections, head); PyG Linear modules keep their own init."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    return model


def build_model(*, num_dim: int, cat_dim: int, hidden: int = 128, trans_heads: int = 2, sem_heads: int = 2,
                dropout: float = 0.5, num_relations: int = 2, extscore_width: int = 0, extscore_inject: str = "num",
                init: bool = True, fused: bool = True) -> RGTDetector:
    """``num_dim`` includes ``extscore_width`` when ``extscore_inject == "num"`` (``base + J.width``)."""
    m = RGTDetector(num_dim=num_dim, cat_dim=cat_dim, hidden=hidden, trans_heads=trans_heads, sem_heads=sem_heads,
                    dropout=dropout, num_relations=num_relations, extscore_width=extscore_width, extscore_inject=extscore_inject,
                    fused=fused)
    return init_weights(m) if init else m


def benchmark_param_count(num_dim: int = 5, cat_dim: int = 3, hidden: int = 128, heads: int = 2, sem_heads: int = 2,
                          num_relations: int = 2, sem_hidden: int = 128, head_hidden: int = 64) -> int:
    """Parameter count of the benchmark RGTDetector, written out from its module list."""
    q = hidden // 4
    enc = (num_dim + 1) * q + (cat_dim + 1) * q + 2 * (768 + 1) * q + (4 * q + 1) * hidden
    conv = 3 * (hidden * heads * hidden + heads * hidden) + (hidden * hidden + hidden)     # k, q, v + skip
    layer = (2 * hidden * hidden + hidden) + num_relations * conv + sem_heads * (hidden * sem_hidden + sem_hidden + sem_hidden)
    head = (hidden + 1) * head_hidden + (head_hidden + 1) * 2
    return enc + 2 * layer + head
