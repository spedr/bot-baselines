"""Shared PyG model skeleton for the botrgcn family: four input projections -> conv stack -> head.

    x_d = LeakyReLU(Linear(768, D/4)(des))           x_t = LeakyReLU(Linear(768, D/4)(tweet))
    x_n = LeakyReLU(Linear(k, D/4)(num))             x_c = LeakyReLU(Linear(c, D/4)(cat))
    [x_j = LeakyReLU(Linear(2, D/4)(extscore))]           (only with extscore_inject="branch")
    x   = LeakyReLU(Linear(4D/4 [+D/4], D)(cat(...)))
    x   = body(x, edge_index, edge_type)              (method-specific conv stack)
    h   = LeakyReLU(Linear(D, D)(x))
    out = Linear(D [+2 late ExtScore], n_classes)(h)

Parameter creation order is des, tweet, num, cat, [extscore branch], input, body, head.out1, head.out2,
the order of BotRGCN's official ``model.py``, so seeded runs line up and ``--extscore none`` builds
exactly the ExtScore-free model. The two 768-wide projections run their forward GEMM in float16 when the
text tables are float16 on the GPU, with a float32 backward (``text_linear``; plain fp16 autograd
without loss scaling underflowed the weight gradients on TwiBot-22); everything else is float32.

A body is any ``nn.Module`` with ``forward(x, edge_index, edge_type) -> x`` ([N, D] -> [N, D']);
``GNNClassifier`` passes ``D'`` to the head through ``body_out``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as Fn
import torch_geometric
import torch_geometric.backend
from torch import nn
from torch_geometric.nn import RGCNConv as _PygRGCNConv


class RGCNConv(nn.Module):
    """PyG ``RGCNConv`` pinned to R-GCN semantics: ``x_i' = W_root x_i + sum_r mean_{j in N_r(i)} W_r x_j
    + b`` (Schlichtkrull et al.; PyG <= 2.0 behaviour; BotRGCN paper Eq. 5).

    PyG 2.3+ switches by an input-size heuristic (``torch_geometric.backend.use_segment_matmul``,
    learned on an A100) to a ``segment_matmul`` path whose ``aggr='mean'`` averages over ALL incoming
    edges of a node across relations (source: "TODO Re-weight according to edge type degree"), i.e. a
    different function. For 2 relations and D=32 it is taken once the largest relation has more than
    about 60k edges: TwiBot-20 (117k) and TwiBot-22 (2.6M), not Cresci-2015 (7k). This wrapper holds
    the PyG layer as ``.conv`` (a subclass would break PyG's generated ``propagate``) and forces the
    per-relation loop on every call, restoring the global flag afterwards. Same parameters and init."""

    def __init__(self, in_channels: int, out_channels: int, num_relations: int, **kwargs):
        super().__init__()
        self.conv = _PygRGCNConv(in_channels, out_channels, num_relations=num_relations, **kwargs)

    def forward(self, x, edge_index, edge_type=None):
        old = torch_geometric.backend.use_segment_matmul
        torch_geometric.backend.use_segment_matmul = False
        try:
            return self.conv(x, edge_index, edge_type)
        finally:
            torch_geometric.backend.use_segment_matmul = old


_BWD_CHUNK = 65_536


class _HalfInputLinear(torch.autograd.Function):
    """``y = x W^T + b`` with ``x`` a float16/bfloat16 table: forward GEMM in half precision (output
    returned as float32), backward in float32.

    Review round 1 (bug): letting autograd differentiate ``Fn.linear(x16, W.half(), b.half()).float()``
    casts the float32 upstream gradient to float16 before the weight-gradient GEMM. With the loss
    averaged over 7e5 TwiBot-22 train rows the per-row gradients are ~1e-7..1e-9: after 100 epochs 92%
    of them flushed to zero and the des/tweet weight gradients were off by 15% / 43% (relative L2; bias
    14% / 39%). Here the gradient stays float32 and ``x`` is upcast in row chunks (transient
    ``_BWD_CHUNK x 768`` float32, nothing N-sized is kept)."""

    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        return Fn.linear(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)).float()

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        gx = gw = gb = None
        g = g.float()
        if ctx.needs_input_grad[0]:
            gx = (g @ w.float()).to(x.dtype)
        if ctx.needs_input_grad[1]:
            gw = torch.zeros(w.shape, dtype=torch.float32, device=w.device)
            for s in range(0, x.shape[0], _BWD_CHUNK):
                gw.addmm_(g[s:s + _BWD_CHUNK].t(), x[s:s + _BWD_CHUNK].float())
            gw = gw.to(w.dtype)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            gb = g.sum(0)
        return gx, gw, gb


def text_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """``x @ weight.T + bias`` for a (possibly float16) text table; float32 result. Half-precision
    CUDA inputs run the GEMM in half precision with a float32 backward (``_HalfInputLinear``)."""
    if x.dtype in (torch.float16, torch.bfloat16) and x.is_cuda:
        return _HalfInputLinear.apply(x, weight, bias)
    return Fn.linear(x.float() if x.dtype != weight.dtype else x, weight, bias)


def _text_proj(lin: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    return text_linear(x, lin.weight, lin.bias)


class InputEncoder(nn.Module):
    def __init__(self, hidden: int, *, des_dim: int = 768, tweet_dim: int = 768, num_dim: int = 5,
                 cat_dim: int = 3, extra_dim: int = 0, slope: float = 0.01, input_dropout: float = 0.0):
        super().__init__()
        if hidden % 4:
            raise ValueError("hidden must be divisible by 4")
        q = hidden // 4
        self.des = nn.Linear(des_dim, q)
        self.tweet = nn.Linear(tweet_dim, q)
        self.num = nn.Linear(num_dim, q)
        self.cat = nn.Linear(cat_dim, q)
        self.extra = nn.Linear(extra_dim, q) if extra_dim else None
        self.input = nn.Linear(4 * q + (q if extra_dim else 0), hidden)
        self.act = nn.LeakyReLU(slope)
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout else None

    def forward(self, des, tweet, num, cat, extra=None) -> torch.Tensor:
        a = self.act
        parts = [a(_text_proj(self.des, des)), a(_text_proj(self.tweet, tweet)), a(self.num(num)), a(self.cat(cat))]
        if self.extra is not None:
            if extra is None:
                raise ValueError("model has a ExtScore branch but no extra input was given")
            parts.append(a(self.extra(extra)))
        x = torch.cat(parts, dim=1)
        if self.input_dropout is not None:
            x = self.input_dropout(x)
        return a(self.input(x))


class OutputHead(nn.Module):
    def __init__(self, hidden: int, n_classes: int = 2, *, late_dim: int = 0, slope: float = 0.01, in_dim: int | None = None):
        super().__init__()
        self.out1 = nn.Linear(in_dim or hidden, hidden)
        self.out2 = nn.Linear(hidden + late_dim, n_classes)
        self.act = nn.LeakyReLU(slope)
        self.late_dim = late_dim

    def forward(self, x, late=None) -> torch.Tensor:
        h = self.act(self.out1(x))
        if self.late_dim:
            if late is None:
                raise ValueError("model has late ExtScore fusion but no extra input was given")
            h = torch.cat([h, late], dim=1)
        return self.out2(h)


class GNNClassifier(nn.Module):
    """encoder -> body -> head. ``extscore_inject`` says where ``GraphInputs.extra`` enters:
    ``"branch"`` (encoder fifth branch), ``"late"`` (head), or ``None`` (not used)."""

    def __init__(self, encoder: InputEncoder, body: nn.Module, head: OutputHead, extscore_inject: str | None = None):
        super().__init__()
        self.encoder, self.body, self.head = encoder, body, head
        self.extscore_inject = extscore_inject

    def forward(self, g) -> torch.Tensor:
        ex = g.extra if self.extscore_inject == "branch" else None
        x = self.encoder(g.des, g.tweet, g.num, g.cat, ex)
        x = self.body(x, g.edge_index, g.edge_type)
        return self.head(x, g.extra if self.extscore_inject == "late" else None)


def make_classifier(body_factory, *, hidden: int, num_dim: int, cat_dim: int, extscore_width: int = 0,
                    extscore_inject: str = "num", n_classes: int = 2, des_dim: int = 768, tweet_dim: int = 768,
                    body_out: int | None = None, input_dropout: float = 0.0, slope: float = 0.01) -> GNNClassifier:
    """Build encoder, body, head in the canonical parameter order.

    ``body_factory()`` returns the body; it is called after the encoder is built so that parameter
    creation (and RNG consumption) follows the official order. ``extscore_width`` is ``J.width`` (0 or 2).
    With ``extscore_inject="num"`` the caller already appended the columns to ``num`` (``to_device``), so
    pass ``num_dim = base + J.width``; with ``"branch"``/``"late"`` the width creates the fifth branch
    / widens the classifier, and ``num_dim`` is the base width. ``extscore_width = 0`` builds exactly the
    ExtScore-free model whatever ``extscore_inject`` says.
    """
    enc = InputEncoder(hidden, des_dim=des_dim, tweet_dim=tweet_dim, num_dim=num_dim, cat_dim=cat_dim,
                       extra_dim=extscore_width if extscore_inject == "branch" else 0, slope=slope,
                       input_dropout=input_dropout)
    body = body_factory()
    head = OutputHead(hidden, n_classes, late_dim=extscore_width if extscore_inject == "late" else 0, slope=slope,
                      in_dim=body_out)
    return GNNClassifier(enc, body, head, extscore_inject if (extscore_width and extscore_inject != "num") else None)


def kaiming_init_linears(model: nn.Module) -> nn.Module:
    """BotRGCN ``utils.init_weights``: kaiming_uniform_ (a=0, fan_in) on every ``nn.Linear`` weight;
    biases keep the PyTorch default; PyG conv parameters keep their own (glorot/zeros) init."""
    for m in model.modules():
        if type(m) is nn.Linear:
            nn.init.kaiming_uniform_(m.weight)
    return model


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
