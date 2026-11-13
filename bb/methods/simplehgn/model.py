"""Benchmark detector around Simple-HGN / HGT (docs/methods/simplehgn.md 4.1-4.3).

    n = Drop(LReLU(Linear(k, 32)(num)))   b = Drop(LReLU(Linear(c, 32)(cat)))
    t = Drop(LReLU(Linear(768, 32)(tw)))  d = Drop(LReLU(Linear(768, 32)(des)))
    x0 = Drop(LReLU(Linear(128, 128)(cat[n, b, t, d])))
    shgn: h1, a1 = SHGN1(x0); h2, _ = SHGN2(h1, pre_alpha=a1)  (ELU inside, L2-normalised output)
    hgt:  h2 = HGT2(HGT1(x0))                                     (no activation/dropout between)
    y = Linear(64, 2)([Drop](LReLU(Linear(128 [+2 late ExtScore], 64)(h2))))

Built on the shared ``bb.gnn`` skeleton: ``InputEncoder`` with ``input_dropout`` (dropout on the
concatenated activated branches = dropout on each branch) and ``GNNClassifier``. The dropout after
the input Linear opens the body. The head is local: the benchmark head is D -> 64 -> 2 with dropout
in the middle and late ExtScore fusion before ``out1`` (spec 8), which ``bb.gnn.OutputHead`` does not do.

Init (spec 4.1): every ``torch.nn.Linear`` gets ``kaiming_uniform_`` weights and zero bias (this
includes the Simple-HGN layer's W/W_r/a/W_res); the edge-type embedding and the HGT layer's own
parameters keep their default init.
"""
from __future__ import annotations

import torch
from torch import nn

from ...gnn import GNNClassifier, InputEncoder
from .layers import HGTConv, SimpleHGNConv

MODELS = ("shgn", "hgt")


class SHGNBody(nn.Module):
    def __init__(self, hidden: int = 128, num_edge_types: int = 2, rel_dim: int = 200, beta: float = 0.05,
                 dropout: float = 0.5, edge_residual: str = "paper", layers: int = 2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList([
            SimpleHGNConv(hidden, hidden, num_edge_types, rel_dim, beta=beta, final_layer=(i == layers - 1),
                          edge_residual=edge_residual) for i in range(layers)])

    def forward(self, x, edge_index, edge_type):
        x = self.dropout(x)
        alpha = None
        for conv in self.convs:
            x, alpha = conv(x, edge_index, edge_type, pre_alpha=alpha)
        return x


class HGTBody(nn.Module):
    def __init__(self, hidden: int = 128, num_edge_types: int = 2, heads: int = 1, dropout: float = 0.5,
                 impl: str = "fixed", layers: int = 2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList([HGTConv(hidden, hidden, num_edge_types, heads=heads, impl=impl)
                                    for _ in range(layers)])

    def forward(self, x, edge_index, edge_type):
        x = self.dropout(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
        return x


class BenchHead(nn.Module):
    """``out2([Drop](LReLU(out1(x [|| late]))))``; ``late`` (ExtScore [z, m]) enters before ``out1``."""

    def __init__(self, in_dim: int = 128, mid: int = 64, n_classes: int = 2, dropout: float = 0.5,
                 late_dim: int = 0, slope: float = 0.01):
        super().__init__()
        self.out1 = nn.Linear(in_dim + late_dim, mid)
        self.out2 = nn.Linear(mid, n_classes)
        self.act = nn.LeakyReLU(slope)
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.late_dim = late_dim

    def forward(self, x, late=None):
        if self.late_dim:
            if late is None:
                raise ValueError("model has late ExtScore fusion but no extra input was given")
            x = torch.cat([x, late.to(x.dtype)], dim=1)
        h = self.act(self.out1(x))
        if self.dropout is not None:
            h = self.dropout(h)
        return self.out2(h)


class Detector(GNNClassifier):
    """``GNNClassifier`` whose body and head run in float32 even under fp16 autocast: only the input
    encoder (the 768-wide text projections) uses fp16. Under full fp16 autocast an HGT seed on
    Cresci-2015 overflowed (inf/NaN outputs for some nodes from epoch 69 on); the layers are small,
    so float32 costs nothing measurable on these subgraphs."""

    def forward(self, g) -> torch.Tensor:
        ex = g.extra if self.extscore_inject == "branch" else None
        x = self.encoder(g.des, g.tweet, g.num, g.cat, ex)
        late = g.extra if self.extscore_inject == "late" else None
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = self.body(x.float(), g.edge_index, g.edge_type)
            return self.head(x, None if late is None else late.float())


def bench_init(model: nn.Module) -> nn.Module:
    """Benchmark ``init_weights``: kaiming_uniform_ on every torch.nn.Linear weight, zero bias."""
    for m in model.modules():
        if type(m) is nn.Linear:
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model


def build_model(kind: str, *, num_dim: int, cat_dim: int, hidden: int = 128, num_edge_types: int = 2,
                dropout: float = 0.5, extscore_width: int = 0, extscore_inject: str = "num", rel_dim: int = 200,
                beta: float = 0.05, edge_residual: str = "paper", hgt_impl: str = "fixed", heads: int = 1,
                head_dropout: bool = True, head_mid: int = 64, init: bool = True) -> GNNClassifier:
    """``kind`` = "shgn" | "hgt". ``num_dim`` includes ``extscore_width`` when ``extscore_inject == "num"``.
    ``extscore_width = 0`` builds exactly the ExtScore-free model whatever ``extscore_inject`` says."""
    if kind not in MODELS:
        raise ValueError(kind)
    enc = InputEncoder(hidden, num_dim=num_dim, cat_dim=cat_dim,
                       extra_dim=extscore_width if extscore_inject == "branch" else 0, input_dropout=dropout)
    if kind == "shgn":
        body = SHGNBody(hidden, num_edge_types, rel_dim, beta, dropout, edge_residual)
    else:
        body = HGTBody(hidden, num_edge_types, heads, dropout, hgt_impl)
    head = BenchHead(hidden, head_mid, dropout=dropout if head_dropout else 0.0,
                     late_dim=extscore_width if extscore_inject == "late" else 0)
    m = Detector(enc, body, head, extscore_inject if (extscore_width and extscore_inject != "num") else None)
    return bench_init(m) if init else m
