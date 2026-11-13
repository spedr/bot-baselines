"""Fused gather kernels for the RGT edge stage (review round 2, efficiency; CUDA + Triton only).

On TwiBot-22 subgraphs (2048 seeds: ~130k edges, 2 heads, width 512) the RGT attention/aggregation
was about half of the GPU time of a training step, because every op of

    s_e = <Q[q_e], K[k_e]>,   agg[q] = sum_{e in seg q} a_e V[k_e]

materialised gathered [E, heads, 512] fp32 tensors (533 MB each) and read/wrote them several times
(and again in backward). Two Triton kernels compute the same sums straight from the node-level rows:

    sddmm(A, B, ia, ib)[e, c]          = sum_h A[ia[e], c, h] * B[ib[e], c, h]
    seg_spmm(ptr, idx, w, X)[r, c, :]  = sum_{j = ptr[r]}^{ptr[r+1]-1} w[j, c] * X[idx[j], c, :]

``seg_spmm`` walks a CSR (edges sorted by output row), so the aggregation has no atomics and is
deterministic. Backward reuses both: dV and dK are ``seg_spmm`` over the edges sorted by source row,
dQ over the segment order, d(scores) is an ``sddmm``. Only [E, heads] score tensors are stored.
fp32 accumulation; results equal the gather/index_add formulation up to summation order.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover - CPU-only installs
    HAVE_TRITON = False


if HAVE_TRITON:
    @triton.jit
    def _sddmm_kernel(A, B, IA, IB, OUT, E, sa_row, sa_c, sb_row, sb_c,
                      C: tl.constexpr, H: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_H: tl.constexpr):
        pid = tl.program_id(0)
        c = tl.program_id(1)
        offs_e = pid * BLOCK_E + tl.arange(0, BLOCK_E)
        me = offs_e < E
        ia = tl.load(IA + offs_e, mask=me, other=0).to(tl.int64)
        ib = tl.load(IB + offs_e, mask=me, other=0).to(tl.int64)
        acc = tl.zeros([BLOCK_E], dtype=tl.float32)
        for h0 in range(0, H, BLOCK_H):
            offs_h = h0 + tl.arange(0, BLOCK_H)
            m = me[:, None] & (offs_h[None, :] < H)
            a = tl.load(A + ia[:, None] * sa_row + c * sa_c + offs_h[None, :], mask=m, other=0.0)
            b = tl.load(B + ib[:, None] * sb_row + c * sb_c + offs_h[None, :], mask=m, other=0.0)
            acc += tl.sum(a * b, axis=1)
        tl.store(OUT + offs_e * C + c, acc, mask=me)

    @triton.jit
    def _spmm_kernel(PTR, IDX, W, X, OUT, sw_e, sx_row, sx_c, so_row, so_c,
                     H: tl.constexpr, BLOCK_H: tl.constexpr, UNROLL: tl.constexpr):
        r = tl.program_id(0)
        c = tl.program_id(1)
        hb = tl.program_id(2)
        offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        mh = offs_h < H
        start = tl.load(PTR + r)
        end = tl.load(PTR + r + 1)
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        # UNROLL edges per iteration: independent loads in flight (segments are short, latency-bound)
        for j0 in range(start, end, UNROLL):
            offs_j = j0 + tl.arange(0, UNROLL)
            mj = offs_j < end
            i = tl.load(IDX + offs_j, mask=mj, other=0).to(tl.int64)
            w = tl.load(W + offs_j * sw_e + c, mask=mj, other=0.0)
            x = tl.load(X + i[:, None] * sx_row + c * sx_c + offs_h[None, :], mask=mj[:, None] & mh[None, :], other=0.0)
            acc += tl.sum(w[:, None] * x, axis=0)
        tl.store(OUT + r * so_row + c * so_c + offs_h, acc, mask=mh)


def _check(X: torch.Tensor):
    if X.dim() != 3 or X.stride(2) != 1 or X.dtype != torch.float32:
        raise ValueError("expected a float32 [rows, heads, H] tensor with unit stride on H")


def sddmm(A: torch.Tensor, B: torch.Tensor, ia: torch.Tensor, ib: torch.Tensor) -> torch.Tensor:
    """[E, C]: out[e, c] = <A[ia[e], c, :], B[ib[e], c, :]>."""
    _check(A)
    _check(B)
    E, C, H = ia.numel(), A.shape[1], A.shape[2]
    out = torch.empty((E, C), dtype=torch.float32, device=A.device)
    if E:
        BLOCK_E, BLOCK_H = 32, min(128, triton.next_power_of_2(H))
        _sddmm_kernel[(triton.cdiv(E, BLOCK_E), C)](A, B, ia, ib, out, E, A.stride(0), A.stride(1), B.stride(0),
                                                    B.stride(1), C=C, H=H, BLOCK_E=BLOCK_E, BLOCK_H=BLOCK_H)
    return out


def seg_spmm(ptr: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, X: torch.Tensor, rows: int) -> torch.Tensor:
    """[rows, C, H]: out[r, c, :] = sum_{j in ptr[r]:ptr[r+1]} w[j, c] * X[idx[j], c, :]."""
    _check(X)
    C, H = X.shape[1], X.shape[2]
    if w.stride(1) != 1:
        w = w.contiguous()
    out = torch.empty((rows, C, H), dtype=torch.float32, device=X.device)
    if rows:
        BLOCK_H = min(128, triton.next_power_of_2(H))
        _spmm_kernel[(rows, C, triton.cdiv(H, BLOCK_H))](ptr, idx, w, X, out, w.stride(0), X.stride(0), X.stride(1),
                                                         out.stride(0), out.stride(1), H=H, BLOCK_H=BLOCK_H,
                                                         UNROLL=4)
    return out


def csr_parts(ctx, n_k_rows: int):
    """Segment pointers of the q-sorted edges and the source-row (k) CSR: (ptr_q, perm_k, ptr_k). Taken
    from a host-built plan when present, else computed on the device (and cached on ``ctx``)."""
    if getattr(ctx, "ptr_q", None) is not None:
        return ctx.ptr_q, ctx.perm_k, ctx.ptr_k
    q, k = ctx.q_sorted, ctx.k_sorted
    ptr_q = torch.zeros(ctx.n_seg + 1, dtype=torch.long, device=q.device)
    ptr_q[1:] = torch.cumsum(torch.bincount(q, minlength=ctx.n_seg), 0)
    perm_k = torch.argsort(k, stable=True)
    ptr_k = torch.zeros(n_k_rows + 1, dtype=torch.long, device=k.device)
    ptr_k[1:] = torch.cumsum(torch.bincount(k, minlength=n_k_rows), 0)
    ctx.ptr_q, ctx.perm_k, ctx.ptr_k = ptr_q, perm_k, ptr_k
    return ptr_q, perm_k, ptr_k
