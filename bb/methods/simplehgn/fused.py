"""Fused edge kernels (Triton, CUDA float32) for the message passing of Simple-HGN / HGT.

    gather_scale_scatter(x, src, w, dst, n)[i] = sum_{e: dst_e = i} w_e * x[src_e]      ([n, F])
    gather_rowdot(a, ia, b, ib)[e]             = <a[ia_e], b[ib_e]>                       ([E])

Eager PyTorch materialises every [E, F] intermediate (gather, scale, scatter; in backward again
gather, scale, scatter, plus the saved gathered rows for the gradient of ``w``). These kernels read
the rows in place and accumulate with atomics, as ``index_add_`` does, so results equal the eager
form up to float32 summation order. Both are autograd functions whose backward passes are again
the two kernels. CPU tensors, non-float32 inputs and machines without Triton use the eager form.
"""
from __future__ import annotations

import torch

try:  # pragma: no cover - depends on the install
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # noqa: BLE001
    HAVE_TRITON = False

ENABLED = True   # module switch (tests / A-B measurements)


def _eager_gss(x, src, w, dst, n):
    out = torch.zeros((n, x.shape[1]), dtype=x.dtype, device=x.device)
    return out.index_add_(0, dst, torch.index_select(x, 0, src) * w.unsqueeze(1))


def _eager_rowdot(a, ia, b, ib):
    return (torch.index_select(a, 0, ia) * torch.index_select(b, 0, ib)).sum(-1)


if HAVE_TRITON:
    @triton.jit
    def _gss_kernel(x_ptr, src_ptr, w_ptr, dst_ptr, out_ptr, E, F: tl.constexpr, sx, so,
                    BLOCK_E: tl.constexpr, BLOCK_F: tl.constexpr):
        pid = tl.program_id(0)
        e = pid * BLOCK_E + tl.arange(0, BLOCK_E)
        em = e < E
        s = tl.load(src_ptr + e, mask=em, other=0).to(tl.int64)
        d = tl.load(dst_ptr + e, mask=em, other=0).to(tl.int64)
        w = tl.load(w_ptr + e, mask=em, other=0.0)
        f = tl.arange(0, BLOCK_F)
        fm = f < F
        m = em[:, None] & fm[None, :]
        xv = tl.load(x_ptr + s[:, None] * sx + f[None, :], mask=m, other=0.0)
        tl.atomic_add(out_ptr + d[:, None] * so + f[None, :], xv * w[:, None], mask=m, sem="relaxed")

    @triton.jit
    def _rowdot_kernel(a_ptr, ia_ptr, b_ptr, ib_ptr, out_ptr, E, F: tl.constexpr, sa, sb,
                       BLOCK_E: tl.constexpr, BLOCK_F: tl.constexpr):
        pid = tl.program_id(0)
        e = pid * BLOCK_E + tl.arange(0, BLOCK_E)
        em = e < E
        i = tl.load(ia_ptr + e, mask=em, other=0).to(tl.int64)
        j = tl.load(ib_ptr + e, mask=em, other=0).to(tl.int64)
        f = tl.arange(0, BLOCK_F)
        m = em[:, None] & (f < F)[None, :]
        av = tl.load(a_ptr + i[:, None] * sa + f[None, :], mask=m, other=0.0)
        bv = tl.load(b_ptr + j[:, None] * sb + f[None, :], mask=m, other=0.0)
        tl.store(out_ptr + e, tl.sum(av * bv, axis=1), mask=em)


def _block(F):
    bf = triton.next_power_of_2(F)
    return max(1, min(64, 4096 // bf)), bf


def _gss_launch(x, src, w, dst, n):
    x, w = x.contiguous(), w.contiguous()
    E, F = src.shape[0], x.shape[1]
    out = torch.zeros((n, F), dtype=torch.float32, device=x.device)
    if E:
        be, bf = _block(F)
        _gss_kernel[(triton.cdiv(E, be),)](x, src, w, dst, out, E, F, x.stride(0), out.stride(0),
                                           BLOCK_E=be, BLOCK_F=bf, num_warps=4)
    return out


def _rowdot_launch(a, ia, b, ib):
    a, b = a.contiguous(), b.contiguous()
    E, F = ia.shape[0], a.shape[1]
    out = torch.empty(E, dtype=torch.float32, device=a.device)
    if E:
        be, bf = _block(F)
        _rowdot_kernel[(triton.cdiv(E, be),)](a, ia, b, ib, out, E, F, a.stride(0), b.stride(0),
                                              BLOCK_E=be, BLOCK_F=bf, num_warps=4)
    return out


class _GSS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, src, w, dst, n):
        ctx.save_for_backward(x, src, w, dst)
        return _gss_launch(x, src, w, dst, n)

    @staticmethod
    def backward(ctx, g):
        x, src, w, dst = ctx.saved_tensors
        g = g.contiguous()
        gx = _gss_launch(g, dst, w, src, x.shape[0]) if ctx.needs_input_grad[0] else None
        gw = _rowdot_launch(g, dst, x, src) if ctx.needs_input_grad[2] else None
        return gx, None, gw, None, None


class _RowDot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, ia, b, ib):
        ctx.save_for_backward(a, ia, b, ib)
        return _rowdot_launch(a, ia, b, ib)

    @staticmethod
    def backward(ctx, g):
        a, ia, b, ib = ctx.saved_tensors
        g = g.contiguous()
        ga = _gss_launch(b, ib, g, ia, a.shape[0]) if ctx.needs_input_grad[0] else None
        gb = _gss_launch(a, ia, g, ib, b.shape[0]) if ctx.needs_input_grad[2] else None
        return ga, None, gb, None


def _use(*ts) -> bool:
    return (ENABLED and HAVE_TRITON and all(t.is_cuda for t in ts)
            and all(t.dtype == torch.float32 for t in ts if t.is_floating_point()))


def gather_scale_scatter(x, src, w, dst, n: int):
    """``zeros(n, F).index_add_(0, dst, x[src] * w[:, None])`` (x [N, F] float32, w [E])."""
    if _use(x, w):
        return _GSS.apply(x, src, w, dst, n)
    return _eager_gss(x, src, w, dst, n)


def gather_rowdot(a, ia, b, ib):
    """``(a[ia] * b[ib]).sum(-1)`` (a [Na, F], b [Nb, F] float32)."""
    if _use(a, b):
        return _RowDot.apply(a, ia, b, ib)
    return _eager_rowdot(a, ia, b, ib)
