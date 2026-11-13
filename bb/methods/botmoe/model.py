"""BotMoE model, released-code architecture ``AllInOne1_rgcn_rgt_gcn`` (docs/methods/botmoe.md 4.1).

    align:   des, tweet 768->A; num k->A; cat c->A; meta m->A       (each Linear + LeakyReLU(0.01))
    graph:   three branches (GCN, RGCN, RGT), each
               4 x (Linear(A, A) + SELU) -> concat (E = 4A) -> Linear(E, E) + SELU
               GCN:  GCNConv(E,E) -> dropout 0.4 -> GCNConv(E,E)
               RGCN: ONE RGCNConv(E,E,R) applied twice (tied), dropout 0.4 between, no activation
               RGT:  RGTLayer -> dropout 0.4 -> RGTLayer (2 transformer heads, 2 semantic heads, p=0.5)
               -> MoE(E -> O), experts MLP(E -> h -> O) with LeakyReLU, h = E (GCN/RGCN) or 64 (RGT)
    text:    MoE(gate = aligned des), experts MLPclassifier: Linear(A, A/2)(tweet) | Linear(A, A/2)(des)
             -> Linear(A, 128) + LeakyReLU -> Dropout(0.3) -> Linear(128, O)
    meta:    Dropout(0.1) -> MoE(A -> 32, experts MLP(A -> 128 -> 32) with ReLU) -> Linear(32, O)
    fusion:  T = stack(gcn, rgcn, rgt, text, meta) [B, 5, O] -> BatchNorm1d(O) over the channel axis
             z = Dropout(0.3)(MHA(LN(T)))  (2 heads, attention dropout 0.3, no residual, no FFN)
             M = head-averaged attention [B, 5, 5], zero-padded to 6 x 6 (FixedPooling(6) with pool 1)
             logits = Linear(5O + 36, 2)(BatchNorm1d(5O + 36)(concat(flatten z, flatten M)))

MoE layer (``NoisyTopKMoE``): Shazeer et al. noisy top-k gating as in the released ``moe_all.py``
(David Rau's port): clean = x W_g; training adds N(0,1) * (softplus(x W_n) + 1e-2); softmax over the
top-k logits; output = log(sum_j g_j exp(E_j(x))) over the selected experts (computed as a stable
logsumexp: with k = 1 it is exactly the selected expert's output; ``dispatch="dense"`` (default)
evaluates every expert on every row and gathers the routed outputs, the same function and gradients
as the released sparse dispatch without its host syncs); balance loss = coef *
(cv^2(importance) + cv^2(load)), load from the smooth P(in top-k) with a Normal(0.1, 1) CDF (the
released buffer ``mean = 0.1``) when training with noise and k < n, else the hard counts.

ExtScore injection (``extscore_at``): ``metadata`` ([z, m] joins the metadata-expert input, so it also drives
that MoE's gate), ``num`` (generic input level: appended to the botrgcn ``num`` block, i.e. the graph
branches), ``metadata+num``, ``token`` (6th fusion token Linear(2, O); the 6 x 6 attention then fills
FixedPooling(6)), ``head`` (concatenated to the classifier input after BN), ``gate_only`` (appended
to the gate input of every MoE, not to any expert input). The width-carrying inputs (meta, num) are
built by the caller; ``extscore_width = 0`` builds exactly the ExtScore-free model.

Parameter creation order follows ``AllInOne1_rgcn_rgt_gcn.__init__``: align, classifier, fusion,
rgcn, rgt, gcn, text, meta, [extscore token], bn1, bn2.

``forward(..., plan=)`` takes host-built edge structures (``plan.SubgraphPlan``): GCN normalisation,
per-relation RGCN edge lists and the RGT segment context are then not derived on the device (same
function, bit-equal; no host syncs).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as Fn
from torch import nn
from torch_geometric.nn import GCNConv, RGCNConv

from .rgt_chunked import ChunkedRGTLayer

EXTSCORE_AT = ("metadata", "num", "metadata+num", "token", "head", "gate_only")


# ---------------------------------------------------------------------------------------------- MoE


class MLP(nn.Module):
    """moe_all.MLP (ReLU) / Botrgcn.MLP (LeakyReLU): Linear -> act -> Linear."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int, act: str = "leaky"):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.act = nn.LeakyReLU() if act == "leaky" else nn.ReLU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TextExpert(nn.Module):
    """roberta.MLPclassifier (all_mode=False)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.pre_model1 = nn.Linear(in_dim, in_dim // 2)       # tweet
        self.pre_model2 = nn.Linear(in_dim, in_dim // 2)       # des
        self.linear = nn.Linear(2 * (in_dim // 2), hidden)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, out_dim)

    def forward(self, tweet, des):
        x = torch.cat((self.pre_model1(tweet), self.pre_model2(des)), dim=1)
        return self.classifier(self.dropout(self.act(self.linear(x))))


def cv_squared(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] == 1:
        return x.new_zeros(())
    x = x.float()
    return x.var() / (x.mean() ** 2 + 1e-10)


class NoisyTopKMoE(nn.Module):
    """Sparsely gated MoE (released ``moe_all.MoE``). ``forward(gate_x, *expert_inputs)`` returns
    ``(y [B, out], balance_loss, top1 [B])``; expert ``j`` is called on the rows routed to it."""

    def __init__(self, gate_dim: int, out_dim: int, experts: list[nn.Module], k: int = 1, noisy: bool = True,
                 loss_coef: float = 1e-2, noise_eps: float = 1e-2, normal_loc: float = 0.1):
        super().__init__()
        if not 1 <= k <= len(experts):
            raise ValueError("need 1 <= k <= n_experts")
        self.experts = nn.ModuleList(experts)
        self.n, self.k, self.out_dim = len(experts), k, out_dim
        self.w_gate = nn.Parameter(torch.zeros(gate_dim, self.n))
        self.w_noise = nn.Parameter(torch.zeros(gate_dim, self.n))
        self.noisy, self.loss_coef, self.noise_eps = noisy, loss_coef, noise_eps
        self.dispatch = "dense"            # "sparse": released SparseDispatcher order (reference)
        self.register_buffer("normal_loc", torch.tensor(float(normal_loc)))

    def _phi(self, x):
        return 0.5 * (1 + torch.erf((x - self.normal_loc) / math.sqrt(2.0)))

    def gating(self, xg: torch.Tensor):
        clean = xg @ self.w_gate
        noisy_on = self.noisy and self.training
        if noisy_on:
            std = Fn.softplus(xg @ self.w_noise) + self.noise_eps
            logits = clean + torch.randn_like(clean) * std
        else:
            logits = clean
        top_v, top_i = logits.topk(min(self.k + 1, self.n), dim=1)
        tk_i = top_i[:, : self.k]
        if self.k == 1:     # softmax over one logit is exactly 1 and its gradient exactly 0: skip the ops
            tk_g = torch.ones_like(top_v[:, :1]).detach()
        else:
            tk_g = torch.softmax(top_v[:, : self.k], dim=1)
        gates = torch.zeros_like(logits).scatter(1, tk_i, tk_g)
        if noisy_on and self.k < self.n:
            thr_in = top_v[:, self.k: self.k + 1]           # (k+1)-th value: threshold if in the top k
            thr_out = top_v[:, self.k - 1: self.k]          # k-th value: threshold if out
            is_in = logits > thr_in
            prob = torch.where(is_in, self._phi((clean - thr_in) / std), self._phi((clean - thr_out) / std))
            load = prob.sum(0)
        else:
            load = (gates > 0).sum(0).float()
        return gates, load, tk_i, tk_g

    def forward(self, gate_x: torch.Tensor, *inputs: torch.Tensor):
        gates, load, tk_i, tk_g = self.gating(gate_x)
        loss = self.loss_coef * (cv_squared(gates.sum(0)) + cv_squared(load))
        B = gate_x.shape[0]
        if self.dispatch == "dense":
            # every expert on every row, then pick the routed outputs: the same function and the same
            # gradients as sparse dispatch (unrouted rows get zero gradient through the gather), but
            # no host sync (nonzero) and 3 GEMMs instead of 3 gathers + 3 GEMMs + scatters
            outs = torch.stack([ex(*inputs) for ex in self.experts], dim=1)          # [B, n, O]
            buf = outs.gather(1, tk_i.unsqueeze(-1).expand(-1, -1, outs.shape[-1]))  # [B, k, O]
        else:
            buf = gate_x.new_zeros((B, self.k, self.out_dim))
            for j, ex in enumerate(self.experts):
                rows, slots = (tk_i == j).nonzero(as_tuple=True)
                if rows.numel() == 0:
                    continue
                buf = buf.index_put((rows, slots), ex(*(t[rows] for t in inputs)).to(buf.dtype))
        if self.k == 1:
            y = buf[:, 0]                                   # log(g * exp(E)) with g == 1 (log 1 = 0 exactly)
        else:
            y = torch.logsumexp(buf + torch.log(tk_g).unsqueeze(-1), dim=1)
        return y, loss, tk_i[:, 0]


# ------------------------------------------------------------------------------------- graph branches


class GraphBranch(nn.Module):
    """BotGCN_fmoe / BotRGCN_fmoe / BotRGT_fmoe on the aligned blocks."""

    def __init__(self, kind: str, align: int, out_dim: int, n_experts: int, k: int = 1, num_relations: int = 2,
                 dropout: float = 0.4, gate_extra: int = 0):
        super().__init__()
        E, q = 4 * align, align
        self.kind = kind
        self.des = nn.Sequential(nn.Linear(align, q), nn.SELU())
        self.tweet = nn.Sequential(nn.Linear(align, q), nn.SELU())
        self.num = nn.Sequential(nn.Linear(align, q), nn.SELU())
        self.cat = nn.Sequential(nn.Linear(align, q), nn.SELU())
        self.input = nn.Sequential(nn.Linear(E, E), nn.SELU())
        if kind == "gcn":
            self.conv1, self.conv2 = GCNConv(E, E), GCNConv(E, E)
        elif kind == "rgcn":
            self.conv = RGCNConv(E, E, num_relations=num_relations)
        elif kind == "rgt":
            self.conv1 = ChunkedRGTLayer(num_relations, E, 2, 2, 0.5)
            self.conv2 = ChunkedRGTLayer(num_relations, E, 2, 2, 0.5)
        else:
            raise ValueError(kind)
        hid = 64 if kind == "rgt" else E
        self.moe = NoisyTopKMoE(E + gate_extra, out_dim, [MLP(E, out_dim, hid, "leaky") for _ in range(n_experts)], k=k)
        self.dropout = dropout
        self.checkpoint = False     # activation checkpointing of the two convolutions (training only)
        self.rgcn_segmm = False     # None = PyG heuristic (edge-level segment_matmul), False = node-level

    def convs(self, x, edge_index, edge_type, plan=None):
        # always fp32, also under autocast: pyg-lib segment_matmul (RGCN) has no autocast support, and
        # bf16 scatter-add / index_add (GCN, RGT aggregation) use slow atomics (measured 5x slower)
        with torch.autocast(x.device.type, enabled=False):
            x = x.float()
            p, tr = self.dropout, self.training
            if self.kind == "gcn":
                if plan is None:
                    return self.conv2(Fn.dropout(self.conv1(x, edge_index), p, tr), edge_index)
                # GCNConv.forward with the gcn_norm structures precomputed (plan.py): lin, propagate, bias
                ei, w = plan.gcn_ei, plan.gcn_weight()
                h = self.conv1.propagate(ei, x=self.conv1.lin(x), edge_weight=w) + self.conv1.bias
                h = Fn.dropout(h, p, tr)
                return self.conv2.propagate(ei, x=self.conv2.lin(h), edge_weight=w) + self.conv2.bias
            if self.kind == "rgcn":
                if plan is not None:
                    return self._rgcn(Fn.dropout(self._rgcn(x, plan.rgcn_ei), p, tr), plan.rgcn_ei)
                # aggregate-then-transform per relation (node-level GEMMs) instead of PyG's heuristic
                # choice of pyg-lib segment_matmul over EDGES (11x more rows on TwiBot-22 subgraphs);
                # the same linear map (mean of W x = W mean x), test test_rgcn_path_equivalence
                import torch_geometric.backend as pyg_backend
                old = pyg_backend.use_segment_matmul
                pyg_backend.use_segment_matmul = self.rgcn_segmm
                try:
                    return self.conv(Fn.dropout(self.conv(x, edge_index, edge_type), p, tr), edge_index, edge_type)
                finally:
                    pyg_backend.use_segment_matmul = old
            ctx = plan if plan is not None else self.conv1.make_ctx(edge_index, edge_type, x.shape[0])
            return self.conv2(Fn.dropout(self.conv1(x, edge_index, edge_type, ctx), p, tr), edge_index, edge_type, ctx)

    def _rgcn(self, x, rel_ei):
        """PyG RGCNConv.forward, per-relation (no segment_matmul) path, on pre-split edge lists: the
        same ops in the same order (zeros, + mean_r @ W_r per relation, + x @ root, + bias)."""
        cv = self.conv
        n = x.shape[0]
        out = torch.zeros(n, cv.out_channels, device=x.device)
        for i, ei in enumerate(rel_ei):
            h = cv.propagate(ei, x=x, edge_type_ptr=None, size=(n, n))
            out = out + (h @ cv.weight[i])
        out = out + x @ cv.root
        return out + cv.bias

    def forward(self, d, t, n, c, edge_index, edge_type, gate_extra=None, plan=None):
        x = torch.cat((self.des(d), self.tweet(t), self.num(n), self.cat(c)), dim=1)
        x = self.input(x)
        if self.checkpoint and self.kind != "rgt" and self.training and torch.is_grad_enabled():
            # recompute the convolutions in backward (RNG state restored, so dropout masks match); the
            # RGT layers bound their memory themselves (per-chunk checkpoints, rgt_chunked.py)
            from torch.utils.checkpoint import checkpoint
            x = checkpoint(self.convs, x, edge_index, edge_type, plan, use_reentrant=False)
        else:
            x = self.convs(x, edge_index, edge_type, plan)
        gx = x if gate_extra is None else torch.cat((x, gate_extra), dim=1)
        return self.moe(gx, x)


# ---------------------------------------------------------------------------------------------- model


class BotMoE(nn.Module):
    def __init__(self, *, num_dim: int, cat_dim: int, meta_dim: int, align: int = 128, out_dim: int = 4,
                 n_graph: int = 3, n_text: int = 2, n_meta: int = 3, k_graph: int = 1, k_text: int = 1,
                 k_meta: int = 1, num_relations: int = 2, extscore_width: int = 0, extscore_at: str = "metadata",
                 heads: int = 2, fusion_dropout: float = 0.3, des_dim: int = 768, tweet_dim: int = 768):
        super().__init__()
        if extscore_at not in EXTSCORE_AT:
            raise ValueError(f"extscore_at must be one of {EXTSCORE_AT}")
        jw = extscore_width
        self.extscore_at = extscore_at if jw else None
        self.extscore_width = jw
        A, O = align, out_dim
        gate_x = jw if self.extscore_at == "gate_only" else 0
        n_tok = 6 if self.extscore_at == "token" else 5
        self.n_tok, self.out_dim = n_tok, O
        # feature_align (num_dim / meta_dim already include the ExtScore width where it is injected there)
        self.align_des = nn.Linear(des_dim, A)
        self.align_tweet = nn.Linear(tweet_dim, A)
        self.align_num = nn.Linear(num_dim, A)
        self.align_cat = nn.Linear(cat_dim, A)
        self.align_meta = nn.Linear(meta_dim, A)
        self.act = nn.LeakyReLU(0.01)
        self.classifier = nn.Linear(O * n_tok + 36 + (jw if self.extscore_at == "head" else 0), 2)
        # LModel
        self.mha = nn.MultiheadAttention(O, heads, dropout=fusion_dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(O, eps=1e-5)
        self.drop1 = nn.Dropout(fusion_dropout)
        self.rgcn = GraphBranch("rgcn", A, O, n_graph, k_graph, num_relations, gate_extra=gate_x)
        self.rgt = GraphBranch("rgt", A, O, n_graph, k_graph, num_relations, gate_extra=gate_x)
        self.gcn = GraphBranch("gcn", A, O, n_graph, k_graph, num_relations, gate_extra=gate_x)
        self.text_moe = NoisyTopKMoE(A + gate_x, O, [TextExpert(A, O, 128) for _ in range(n_text)], k=k_text)
        self.meta_drop = nn.Dropout(0.1)
        self.meta_moe = NoisyTopKMoE(A + gate_x, 32, [MLP(A, 32, 128, "relu") for _ in range(n_meta)], k=k_meta)
        self.meta_out = nn.Linear(32, O)
        self.extscore_token = nn.Linear(jw, O) if self.extscore_at == "token" else None
        self.bn1 = nn.BatchNorm1d(O)
        self.bn2 = nn.BatchNorm1d(O * n_tok + 36)

    # -- gates
    def graph_moes(self):
        return [self.gcn.moe, self.rgcn.moe, self.rgt.moe]

    def moes(self) -> dict:
        return {"gcn": self.gcn.moe, "rgcn": self.rgcn.moe, "rgt": self.rgt.moe, "text": self.text_moe,
                "meta": self.meta_moe}

    def set_checkpoint(self, flag: bool) -> None:
        for b in (self.gcn, self.rgcn, self.rgt):
            b.checkpoint = flag
        self.rgt.conv1.ckpt = self.rgt.conv2.ckpt = flag

    def set_graph_gates_trainable(self, flag: bool) -> None:
        for m in self.graph_moes():
            m.w_gate.requires_grad_(flag)
            m.w_noise.requires_grad_(flag)

    @staticmethod
    def _f32(x):
        return x if x.dtype == torch.float32 else x.float()

    def forward(self, g, meta: torch.Tensor, extscore: torch.Tensor | None = None, rows: int | None = None,
                plan=None):
        """``g``: ``bb.gnn.GraphInputs`` of a (sub)graph; ``meta``: [B, meta_dim] metadata-expert
        input; ``extscore``: [B, 2] for token / head / gate_only (else ignored). ``rows``: compute the
        fusion only for the first ``rows`` nodes (the seeds; exact at eval because every per-node
        op there is row-independent). ``plan``: precomputed edge structures of ``g`` (``plan.SubgraphPlan``;
        same function, no host syncs). Returns ``(logits [rows or B, 2], aux)``."""
        a = self.act
        d = a(self.align_des(self._f32(g.des)))
        t = a(self.align_tweet(self._f32(g.tweet)))
        n = a(self.align_num(g.num))
        c = a(self.align_cat(g.cat))
        gj = extscore if self.extscore_at == "gate_only" else None
        gcn, l1, r1 = self.gcn(d, t, n, c, g.edge_index, g.edge_type, gj, plan)
        rgcn, l2, r2 = self.rgcn(d, t, n, c, g.edge_index, g.edge_type, gj, plan)
        rgt, l3, r3 = self.rgt(d, t, n, c, g.edge_index, g.edge_type, gj, plan)
        S = rows if rows is not None else d.shape[0]
        d, t = d[:S], t[:S]
        m = a(self.align_meta(meta[:S]))
        gj = None if gj is None else gj[:S]
        text, l4, r4 = self.text_moe(d if gj is None else torch.cat((d, gj), 1), t, d)
        mx = self.meta_drop(m)
        meta_h, l5, r5 = self.meta_moe(mx if gj is None else torch.cat((mx, gj), 1), mx)
        meta_o = self.meta_out(meta_h)
        toks = [gcn[:S], rgcn[:S], rgt[:S], text, meta_o]
        if self.extscore_token is not None:
            toks.append(self.extscore_token(extscore[:S]))
        T = torch.stack(toks, dim=1)                                        # [S, n_tok, O]
        T = self.bn1(T.permute(0, 2, 1)).permute(0, 2, 1)
        z, att = self.mha(*(3 * [self.norm1(T)]), need_weights=True, average_attn_weights=True)
        z = self.drop1(z)
        att = Fn.pad(att, (0, 6 - self.n_tok, 0, 6 - self.n_tok)) if self.n_tok < 6 else att
        h = self.bn2(torch.cat((z.reshape(S, -1), att.reshape(S, -1)), dim=1))
        if self.extscore_at == "head":
            h = torch.cat((h, extscore[:S].to(h.dtype)), dim=1)
        aux = {"balance": l1 + l2 + l3 + l4 + l5,
               "route": {"gcn": r1[:S], "rgcn": r2[:S], "rgt": r3[:S], "text": r4, "meta": r5}}
        return self.classifier(h), aux


def init_weights(model: BotMoE) -> BotMoE:
    """``main.py`` model_init: xavier_uniform_ on every exact ``nn.Linear`` weight (biases default; PyG
    Linear / RGCN / MHA in_proj keep their own init; MHA ``out_proj`` is a Linear subclass and is not
    matched by ``type(m) == nn.Linear``), then every ``w_gate``/``w_noise`` = 0.1."""
    for m in model.modules():
        if type(m) is nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    for moe in model.moes().values():
        nn.init.constant_(moe.w_gate, 0.1)
        nn.init.constant_(moe.w_noise, 0.1)
    return model


def build_model(*, init: bool = True, **kw) -> BotMoE:
    m = BotMoE(**kw)
    return init_weights(m) if init else m
