"""Memory-bounded RGT layer for BotMoE's RGT branch.

``bb.methods.rgt.model.RGTLayer`` (fused path) materialises several [E, C, H] edge tensors per layer
(gathered queries, keys, values, their product, the messages) and autograd keeps them for backward.
With BotMoE's width (H = 512, C = 2) a TwiBot-22 subgraph of 1024 seeds (88k edges) needs about
360 MB per such tensor and over 4 GB per training step, more than the VRAM budget once the 3 GB
text tables are resident.

``ChunkedRGTLayer`` computes the same function: the attention/aggregation stage runs over chunks
of whole (target, relation) segments (the segment softmax never crosses a chunk), each chunk under
``torch.utils.checkpoint`` (RNG state restored, so attention-dropout masks are identical in the
recompute). Saved for backward are only Q, K, V (per node, not per edge); the per-edge tensors of
one chunk exist only transiently. Parameters, init and state_dict are those of ``RGTLayer``; only
the dropout RNG stream differs from the unchunked layer. Test: ``test_chunked_rgt_equals_fused``.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as Fn

from ..rgt.model import EdgeCtx, RGTLayer


def _edge_stage(Q, K, V, q_idx, k_idx, lo: int, n_seg: int, p: float, training: bool, scale: float):
    from torch_geometric.utils import softmax as seg_softmax
    qi = q_idx - lo
    alpha = (Q[q_idx] * K[k_idx]).sum(-1) * scale                                   # [e, C]
    alpha = seg_softmax(alpha, qi, num_nodes=n_seg)
    alpha = Fn.dropout(alpha, p=p, training=training)
    msg = V[k_idx] * alpha.unsqueeze(-1).to(V.dtype)                                 # [e, C, H]
    return torch.zeros((n_seg, *msg.shape[1:]), dtype=msg.dtype, device=msg.device).index_add_(0, qi, msg)


class _EdgeAttention(torch.autograd.Function):
    """All chunks of the attention/aggregation stage with a hand-written backward (review round 2).

    Forward: the ops of ``_edge_stage`` per chunk, in the same order (same dropout RNG draws), so the
    output is bit-identical. Saved for backward: the node-level Q, K, V (alive anyway as outputs of
    the projections) and the per-edge attention before/after dropout ([E, C], small). Backward
    re-gathers the per-edge rows chunk by chunk and applies the closed-form gradients

        dV[k]  += a'_e dagg[q_e]            da'_e = <dagg[q_e], V[k_e]>      a' = dropout(a)
        da_e    = da'_e [a'_e != 0] / (1 - p)
        ds_e    = a_e (da_e - sum_{e' in seg(e)} a_e' da_e') * scale          (segment softmax)
        dQ[q]  += ds_e K[k_e]               dK[k] += ds_e Q[q_e]

    i.e. no forward recompute (checkpointing ran the forward twice) and no [E, C, H] tensors kept
    between forward and backward. The same function; gradients equal autograd's up to float
    summation order (test ``test_edge_attention_fn_equals_autograd``). The dropout mask is recovered as
    a' != 0; where a underflowed to 0 the softmax gradient is a * (...) = 0 either way."""

    @staticmethod
    def forward(ctx, Q, K, V, q_sorted, k_sorted, chunks, n_seg: int, p: float, training: bool, scale: float):
        from torch_geometric.utils import softmax as seg_softmax
        C, H = Q.shape[1], Q.shape[2]
        parts, att, att_d = [], [], []
        for a, b, s_lo, s_hi in chunks:
            q_idx, k_idx = q_sorted[a:b], k_sorted[a:b]
            qi = q_idx - s_lo
            alpha = (Q[q_idx] * K[k_idx]).sum(-1) * scale
            alpha = seg_softmax(alpha, qi, num_nodes=s_hi - s_lo)
            ad = Fn.dropout(alpha, p=p, training=training)
            msg = V[k_idx] * ad.unsqueeze(-1).to(V.dtype)
            parts.append(torch.zeros((s_hi - s_lo, C, H), dtype=msg.dtype, device=msg.device).index_add_(0, qi, msg))
            att.append(alpha)
            att_d.append(ad)
        agg = torch.cat(parts, 0) if parts else Q.new_zeros((0, C, H))
        if agg.shape[0] != n_seg:
            raise RuntimeError("chunk segments do not cover all segments")
        ctx.save_for_backward(Q, K, V, q_sorted, k_sorted, *att, *att_d)
        ctx.chunks, ctx.p, ctx.scale, ctx.training = chunks, p, scale, training
        return agg

    @staticmethod
    def backward(ctx, dagg):
        Q, K, V, q_sorted, k_sorted, *rest = ctx.saved_tensors
        nc = len(ctx.chunks)
        att, att_d = rest[:nc], rest[nc:]
        dQ, dK, dV = torch.zeros_like(Q), torch.zeros_like(K), torch.zeros_like(V)
        drop = ctx.training and ctx.p > 0
        for j, (a, b, s_lo, s_hi) in enumerate(ctx.chunks):
            q_idx, k_idx = q_sorted[a:b], k_sorted[a:b]
            qi = q_idx - s_lo
            alpha, ad = att[j], att_d[j]
            dmsg = dagg[s_lo:s_hi].index_select(0, qi)                                   # [e, C, H]
            dV.index_add_(0, k_idx, dmsg * ad.unsqueeze(-1))
            da = (dmsg * V.index_select(0, k_idx)).sum(-1)                               # [e, C]
            del dmsg
            if drop:
                da = da * (ad != 0).to(da.dtype) * (1.0 / (1.0 - ctx.p))
            seg = torch.zeros((s_hi - s_lo, da.shape[1]), dtype=da.dtype, device=da.device).index_add_(0, qi, alpha * da)
            ds = (alpha * (da - seg.index_select(0, qi)) * ctx.scale).unsqueeze(-1)       # [e, C, 1]
            dQ.index_add_(0, q_idx, ds * K.index_select(0, k_idx))
            dK.index_add_(0, k_idx, ds * Q.index_select(0, q_idx))
        return dQ, dK, dV, None, None, None, None, None, None, None


class _EdgeAttentionTriton(torch.autograd.Function):
    """``_EdgeAttention`` with fused gather kernels (``edge_triton``; CUDA): scores by ``sddmm`` and
    aggregation by a CSR ``seg_spmm`` straight from the node-level Q/K/V rows, no [E, C, H] tensor in
    forward or backward. Softmax and dropout run on the [E, C] scores with the same ops; dropout is
    drawn chunk by chunk with the chunk sizes of the other paths, so the masks are the same draws."""

    @staticmethod
    def forward(ctx, Q, K, V, q_sorted, k_sorted, ptr_q, perm_k, ptr_k, chunks, n_seg: int, p: float,
                training: bool, scale: float):
        from torch_geometric.utils import softmax as seg_softmax

        from .edge_triton import sddmm, seg_spmm
        s = sddmm(Q, K, q_sorted, k_sorted) * scale                                   # [E, C]
        alpha = seg_softmax(s, q_sorted, num_nodes=n_seg)
        if training and p > 0 and chunks:
            # clone: the CUDA dropout kernel maps random numbers to elements differently for a slice
            # that is not 16-byte aligned (vectorised vs scalar path); fresh chunk tensors, as the
            # other paths have, give the same masks
            ad = torch.cat([Fn.dropout(alpha[a:b].clone(), p=p, training=True) for a, b, _, _ in chunks], 0)
        else:
            ad = alpha
        agg = seg_spmm(ptr_q, k_sorted, ad, V, n_seg)
        ctx.save_for_backward(Q, K, V, q_sorted, k_sorted, ptr_q, perm_k, ptr_k, alpha, ad)
        ctx.p, ctx.scale, ctx.drop = p, scale, training and p > 0
        return agg

    @staticmethod
    def backward(ctx, dagg):
        from .edge_triton import sddmm, seg_spmm
        Q, K, V, q_sorted, k_sorted, ptr_q, perm_k, ptr_k, alpha, ad = ctx.saved_tensors
        dagg = dagg.contiguous()
        nK = K.shape[0]
        q_by_k = q_sorted[perm_k]
        dV = seg_spmm(ptr_k, q_by_k, ad[perm_k], dagg, nK)                            # sum a'_e dagg[q_e]
        da = sddmm(dagg, V, q_sorted, k_sorted)                                       # <dagg[q_e], V[k_e]>
        if ctx.drop:
            da = da * (ad != 0).to(da.dtype) * (1.0 / (1.0 - ctx.p))
        seg = torch.zeros((Q.shape[0], da.shape[1]), dtype=da.dtype, device=da.device).index_add_(0, q_sorted, alpha * da)
        ds = alpha * (da - seg.index_select(0, q_sorted)) * ctx.scale
        dQ = seg_spmm(ptr_q, k_sorted, ds, K, Q.shape[0])
        dK = seg_spmm(ptr_k, q_by_k, ds[perm_k], Q, nK)
        return dQ, dK, dV, None, None, None, None, None, None, None, None, None, None


class ChunkedCtx(EdgeCtx):
    """``EdgeCtx`` plus edges sorted by segment and chunk boundaries at segment edges."""

    def __init__(self, edge_index, edge_type, n: int, R: int, max_edges: int = 32768):
        super().__init__(edge_index, edge_type, n, R)
        self.max_edges = max_edges
        order = torch.argsort(self.q_idx, stable=True)
        self.q_sorted, self.k_sorted = self.q_idx[order], self.k_idx[order]
        E = int(self.q_sorted.numel())
        self.chunks = []                                   # (edge lo, edge hi, segment lo, segment hi)
        if E:
            q = self.q_sorted
            starts = torch.arange(0, E, max_edges, device=q.device)
            # move each cut to the first edge of its segment, so no segment is split
            seg_at = q[starts]
            first = torch.searchsorted(q, seg_at)
            cuts = sorted(set(first.tolist()) | {E})
            seg_cuts = q[torch.tensor(cuts[:-1], device=q.device)].tolist()
            for j, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
                s_lo = 0 if j == 0 else seg_cuts[j]
                s_hi = seg_cuts[j + 1] if j + 1 < len(seg_cuts) else self.n_seg
                self.chunks.append((a, b, s_lo, s_hi))


class ChunkedRGTLayer(RGTLayer):
    def __init__(self, *args, max_edges: int = 32768, **kw):
        super().__init__(*args, **kw)
        self.max_edges = max_edges
        self.ckpt = False          # per-chunk checkpointing (set by BotMoE.set_checkpoint; "autograd" path)
        # "auto": fused Triton kernels on CUDA, else "fn"; "fn": _EdgeAttention (hand-written backward,
        # gathers in torch); "autograd": reference (per-chunk ops under autograd / checkpoint)
        self.edge_impl = "auto"

    def make_ctx(self, edge_index, edge_type, n: int) -> ChunkedCtx:
        return ChunkedCtx(edge_index, edge_type, n, self.R, self.max_edges)

    def _fused_relation_outputs(self, x: torch.Tensor, ctx: EdgeCtx) -> torch.Tensor:
        if not hasattr(ctx, "chunks"):              # ChunkedCtx or a host-built plan.SubgraphPlan
            ctx = self.make_ctx_from(ctx, x)
        if ctx.max_edges != self.max_edges:
            raise ValueError("ctx was built for another chunk size")
        n, R, C, H = x.shape[0], self.R, self.C, self.H
        cv = self.convs
        wq = torch.cat([c.lin_query.weight for c in cv], 0)
        bq = torch.cat([c.lin_query.bias for c in cv], 0)
        wkv = torch.cat([w for c in cv for w in (c.lin_key.weight, c.lin_value.weight)], 0)
        bkv = torch.cat([b for c in cv for b in (c.lin_key.bias, c.lin_value.bias)], 0)
        ws = torch.cat([c.lin_skip.weight for c in cv], 0)
        bs = torch.cat([c.lin_skip.bias for c in cv], 0)
        nS, nT = len(ctx.S), len(ctx.T)
        Q = Fn.linear(x[ctx.T], wq, bq).view(nT * R, C, H)
        KV = Fn.linear(x[ctx.S], wkv, bkv).view(nS * R, 2, C, H)
        K, V = KV[:, 0], KV[:, 1]
        scale = 1.0 / math.sqrt(H)
        impl = self.edge_impl
        if impl == "auto":
            from .edge_triton import HAVE_TRITON
            impl = "triton" if (HAVE_TRITON and x.is_cuda and x.dtype == torch.float32) else "fn"
        if impl == "triton":
            from .edge_triton import csr_parts
            ptr_q, perm_k, ptr_k = csr_parts(ctx, nS * R)
            agg = _EdgeAttentionTriton.apply(Q, K, V, ctx.q_sorted, ctx.k_sorted, ptr_q, perm_k, ptr_k, ctx.chunks,
                                             ctx.n_seg, self.p, self.training, scale)
            return self._finish(x, agg, ctx, ws, bs)
        if impl == "fn":
            agg = _EdgeAttention.apply(Q, K, V, ctx.q_sorted, ctx.k_sorted, ctx.chunks, ctx.n_seg, self.p,
                                       self.training, scale)
            return self._finish(x, agg, ctx, ws, bs)
        grad = self.ckpt and torch.is_grad_enabled()
        parts = []
        for a, b, s_lo, s_hi in ctx.chunks:
            args = (Q, K, V, ctx.q_sorted[a:b], ctx.k_sorted[a:b], s_lo, s_hi - s_lo, self.p, self.training, scale)
            if grad:
                from torch.utils.checkpoint import checkpoint
                parts.append(checkpoint(_edge_stage, *args, use_reentrant=False))
            else:
                parts.append(_edge_stage(*args))
        agg = (torch.cat(parts, 0) if parts else Q.new_zeros((0, C, H)))
        if agg.shape[0] != ctx.n_seg:
            raise RuntimeError("chunk segments do not cover all segments")
        return self._finish(x, agg, ctx, ws, bs)

    def _finish(self, x, agg, ctx, ws, bs):
        n, R, H = x.shape[0], self.R, self.H
        nT = len(ctx.T)
        u = Fn.linear(x, ws, bs).view(n, R, H)
        u = u.index_add(0, ctx.T, agg.mean(1).view(nT, R, H).to(u.dtype))
        gw, gb = self.gate[0].weight, self.gate[0].bias
        a_ = torch.sigmoid(Fn.linear(u, gw[:, :H]) + Fn.linear(x, gw[:, H:], gb).unsqueeze(1))
        return torch.tanh(u) * a_ + x.unsqueeze(1) * (1 - a_)

    def make_ctx_from(self, ctx: EdgeCtx, x) -> ChunkedCtx:
        raise TypeError("ChunkedRGTLayer needs a ChunkedCtx (use layer.make_ctx)")

    def forward(self, x, edge_index, edge_type, ctx=None):
        if ctx is None:
            ctx = self.make_ctx(edge_index, edge_type, x.shape[0])
        return self._fused_semantic(self._fused_relation_outputs(x, ctx))
