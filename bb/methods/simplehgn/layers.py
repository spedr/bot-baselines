"""Simple-HGN and HGT layers for a single node type with R edge types (docs/methods/simplehgn.md 4.2-4.3).

``SimpleHGNConv``: GAT with a learned edge-type embedding in the attention score, node residual
``W_res``, ELU, optional final L2 normalisation and the edge (attention) residual.

    e_ij  = LeakyReLU_0.2(a^T [W x_i || W x_j || W_r r_psi])          (computed as s_i + t_j + c_psi)
    alpha = softmax over all incoming edges of i (all relations jointly)
    edge residual (paper Eq. 10, HGB conv.py): alpha <- (1 - beta) alpha + beta alpha_prev
    out_i = ELU(sum_j alpha_ij W x_j + W_res x_i)  [/ ||.||_2 in the final layer]

``edge_residual="paper"`` (default) mixes the previous layer's (detached) attention into the
message, as the paper and HGB do. ``"bench"`` reproduces the TwiBot-22 benchmark ``layer.py``,
which stores the mixed alpha but multiplies the message by the unmixed one, so beta has no effect
(equivalent to beta = 0). ``naive_simplehgn`` is the benchmark's literal E x 384 concat form, kept
for the equivalence test.

``HGTConv``: PyG's current ``HGTConv`` maths written for one node type (so that it runs under fp16
autocast and without the per-edge-type copies of the source features):

    k, q, v = split(Linear(in, 3F)(x)); k_r = k A_r; v_r = v M_r    (A_r, M_r: [H, D, D], no bias)
    alpha_ij = softmax_i( (q_i . k_r,j) p_r / sqrt(D) )               (jointly over relations)
    agg = sum_j alpha_ij v_r,j;  o = Linear(F, F)(GELU(agg))
    out = s o + (1 - s) x,  s = sigmoid(skip)                          (gated skip, when in == out)

``impl="fixed"`` equals ``torch_geometric.nn.HGTConv`` (tested with copied weights);
``"per_relation"`` takes the softmax per relation and sums (PyG 2.0.3-era semantics; init still
modern); ``"bench_bug"`` emulates PyG 2.0.0-2.0.2 (``out = s * s + (1 - s) x``: the aggregated
message is discarded, the layer ignores the graph and K/Q/V/A/M receive no gradient).

Internal projections of ``HGTConv`` are raw ``Parameter``s with ``torch.nn.Linear``'s default init
(PyG ``Linear``/``HeteroLinear`` defaults), so the benchmark's kaiming re-init of ``nn.Linear``
modules leaves them alone, as it leaves PyG's modules alone.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as Fn
from torch import nn
from torch_geometric.utils import softmax as seg_softmax

from .fused import gather_rowdot, gather_scale_scatter

EDGE_RESIDUAL = ("paper", "bench")
HGT_IMPL = ("fixed", "per_relation", "bench_bug")


def _acc(t: torch.Tensor) -> torch.dtype:
    """Accumulation dtype: float32 for half inputs (autocast), else the input's own dtype."""
    return torch.promote_types(t.dtype, torch.float32)


def _gather(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``x[index]`` along dim 0 via ``index_select``: its backward is an atomic ``index_add_``,
    whereas advanced indexing backpropagates through the sort-based ``index_put`` kernel (measured
    80% of the GPU time of a TwiBot-22 training step)."""
    return torch.index_select(x, 0, index)


class SimpleHGNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_edge_types: int, rel_dim: int,
                 beta: float = 0.05, final_layer: bool = False, edge_residual: str = "paper",
                 slope: float = 0.2):
        super().__init__()
        if edge_residual not in EDGE_RESIDUAL:
            raise ValueError(f"edge_residual must be one of {EDGE_RESIDUAL}")
        # creation order of the benchmark layer.py: W, W_r, a, W_res, rel_emb
        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.W_r = nn.Linear(rel_dim, out_channels, bias=False)
        self.a = nn.Linear(3 * out_channels, 1, bias=False)
        self.W_res = nn.Linear(in_channels, out_channels, bias=False)
        self.rel_emb = nn.Embedding(num_edge_types, rel_dim)
        self.beta, self.final, self.edge_residual, self.slope = beta, final_layer, edge_residual, slope
        self.out_channels = out_channels

    def attention(self, wx: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """alpha [E] (float32): softmax over the incoming edges of each target."""
        src, dst = edge_index[0], edge_index[1]
        F = self.out_channels
        a = self.a.weight.view(-1)
        a1, a2, a3 = a[:F], a[F:2 * F], a[2 * F:]
        dt = _acc(wx)
        st = wx.to(dt) @ torch.stack([a1, a2], 1).to(dt)               # [N, 2]: target, source terms
        c = self.W_r(self.rel_emb.weight).to(dt) @ a3.to(dt)           # per type [R]
        e = Fn.leaky_relu(_gather(st[:, 0], dst) + _gather(st[:, 1], src) + _gather(c, edge_type), self.slope)
        return seg_softmax(e, dst, num_nodes=wx.shape[0])

    def forward(self, x, edge_index, edge_type, pre_alpha=None):
        wx = self.W(x)
        n = x.shape[0]
        alpha = self.attention(wx, edge_index, edge_type)
        mixed = alpha
        if pre_alpha is not None and self.beta is not None:
            mixed = alpha * (1 - self.beta) + pre_alpha * self.beta
        used = mixed if self.edge_residual == "paper" else alpha
        dt = _acc(wx)
        # sum_j alpha_ij W x_j without the [E, F] message tensor (fused kernel on CUDA float32)
        out = gather_scale_scatter(wx.to(dt), edge_index[0], used.to(dt), edge_index[1], n) + self.W_res(x).to(dt)
        out = Fn.elu(out)
        if self.final:
            out = Fn.normalize(out, dim=1)
        return out, mixed.detach()


def naive_simplehgn(layer: SimpleHGNConv, x, edge_index, edge_type, pre_alpha=None, beta_effective: bool = True):
    """Literal benchmark ``message()``: a(cat(W x_i, W x_j, W_r rel_emb)) per edge (reference)."""
    src, dst = edge_index[0], edge_index[1]
    xi, xj = x[dst], x[src]
    rel = layer.W_r(layer.rel_emb(edge_type))
    e = Fn.leaky_relu(layer.a(torch.cat([layer.W(xi), layer.W(xj), rel], 1)), layer.slope)
    alpha = seg_softmax(e, dst, num_nodes=x.shape[0])
    if pre_alpha is not None and layer.beta is not None:
        mixed = alpha * (1 - layer.beta) + pre_alpha.view(-1, 1) * layer.beta
    else:
        mixed = alpha
    used = mixed if beta_effective else alpha
    out = torch.zeros(x.shape[0], layer.out_channels, dtype=x.dtype).index_add_(0, dst, layer.W(xj) * used)
    out = Fn.elu(out + layer.W_res(x))
    if layer.final:
        out = Fn.normalize(out, dim=1)
    return out, mixed.view(-1).detach()


class HGTConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_edge_types: int, heads: int = 1,
                 impl: str = "fixed"):
        super().__init__()
        if impl not in HGT_IMPL:
            raise ValueError(f"impl must be one of {HGT_IMPL}")
        if out_channels % heads:
            raise ValueError("out_channels must be divisible by heads")
        self.in_channels, self.out_channels, self.heads, self.R = in_channels, out_channels, heads, num_edge_types
        self.impl = impl
        D = out_channels // heads
        self.kqv_weight = nn.Parameter(torch.empty(3 * out_channels, in_channels))
        self.kqv_bias = nn.Parameter(torch.empty(3 * out_channels))
        self.out_weight = nn.Parameter(torch.empty(out_channels, out_channels))
        self.out_bias = nn.Parameter(torch.empty(out_channels))
        # relation matrices, index h * R + r (PyG type_vec layout); x @ A[t]
        self.k_rel = nn.Parameter(torch.empty(heads * num_edge_types, D, D))
        self.v_rel = nn.Parameter(torch.empty(heads * num_edge_types, D, D))
        self.skip = nn.Parameter(torch.empty(1))
        self.p_rel = nn.Parameter(torch.empty(num_edge_types, heads))
        self.reset_parameters()

    def reset_parameters(self):
        def lin_(w, b, fan):  # torch.nn.Linear default (= PyG Linear / HeteroLinear default)
            bound = 1.0 / math.sqrt(fan)
            nn.init.uniform_(w, -bound, bound)   # kaiming_uniform(a=sqrt(5)) bound = 1/sqrt(fan)
            if b is not None:
                nn.init.uniform_(b, -bound, bound)
        lin_(self.kqv_weight, self.kqv_bias, self.in_channels)
        lin_(self.out_weight, self.out_bias, self.out_channels)
        D = self.out_channels // self.heads
        lin_(self.k_rel, None, D)
        lin_(self.v_rel, None, D)
        nn.init.ones_(self.skip)
        nn.init.ones_(self.p_rel)

    def forward(self, x, edge_index, edge_type):
        s = torch.sigmoid(self.skip)
        gated = self.out_channels == x.shape[-1]
        if self.impl == "bench_bug":
            if not gated:
                raise ValueError("bench_bug emulation needs in == out channels (the benchmark's case)")
            return s * s + (1 - s) * x             # PyG 2.0.2: 'alpha * alpha'; messages discarded
        n, H, R = x.shape[0], self.heads, self.R
        F = self.out_channels
        D = F // H
        kqv = Fn.linear(x, self.kqv_weight, self.kqv_bias)
        k, q, v = torch.tensor_split(kqv, 3, dim=1)
        k, q, v = k.reshape(n, H, D), q.reshape(n, H, D), v.reshape(n, H, D)
        src, dst = edge_index[0], edge_index[1]
        # per-relation transforms of the source features: [R, n, H, D]
        A = self.k_rel.view(H, R, D, D).transpose(0, 1)            # [R, H, D, D]
        M = self.v_rel.view(H, R, D, D).transpose(0, 1)
        kr = torch.einsum("nhd,rhde->rnhe", k, A.to(k.dtype))
        vr = torch.einsum("nhd,rhde->rnhe", v, M.to(v.dtype))
        flat = edge_type * n + src                                  # row of [R * n] stacks
        if H == 1:
            qi, ki = dst, flat
        else:                                                       # one row per (edge, head) of [*, D] views
            hh = torch.arange(H, device=dst.device)
            qi, ki = (dst.unsqueeze(1) * H + hh).reshape(-1), (flat.unsqueeze(1) * H + hh).reshape(-1)
        dt = _acc(q)
        # (q_i . k_r,j) and sum_j alpha_ij v_r,j without [E, H, D] gathers (fused kernels on CUDA float32)
        dot = gather_rowdot(q.reshape(n * H, D).to(dt), qi, kr.reshape(R * n * H, D).to(dt), ki).view(-1, H)
        score = dot * _gather(self.p_rel, edge_type).to(dt) / math.sqrt(D)
        index = dst if self.impl == "fixed" else dst * R + edge_type
        alpha = seg_softmax(score, index, num_nodes=n * (R if self.impl == "per_relation" else 1))
        agg = gather_scale_scatter(vr.reshape(R * n * H, D).to(dt), ki, alpha.reshape(-1), qi, n * H).view(n, F)
        o = Fn.linear(Fn.gelu(agg), self.out_weight.to(dt), self.out_bias.to(dt))
        if gated:
            o = s * o + (1 - s) * x.to(o.dtype)
        return o

    def load_from_pyg(self, conv, node_type: str = "user") -> "HGTConv":
        """Copy the weights of a ``torch_geometric.nn.HGTConv`` with one node type (for tests)."""
        with torch.no_grad():
            kqv = conv.kqv_lin.lins[node_type]
            self.kqv_weight.copy_(kqv.weight)
            self.kqv_bias.copy_(kqv.bias)
            out = conv.out_lin.lins[node_type]
            self.out_weight.copy_(out.weight)
            self.out_bias.copy_(out.bias)
            self.k_rel.copy_(conv.k_rel.weight)
            self.v_rel.copy_(conv.v_rel.weight)
            self.skip.copy_(conv.skip[node_type])
            for r, et in enumerate(conv.edge_types):
                self.p_rel[r].copy_(conv.p_rel["__".join(et)].view(-1))
        return self
