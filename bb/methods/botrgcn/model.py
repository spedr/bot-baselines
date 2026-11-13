"""BotRGCN bodies (docs/methods/botrgcn.md section 4) on the shared ``bb.gnn`` skeleton."""
from __future__ import annotations

import torch
import torch.nn.functional as Fn
from torch import nn

from ...gnn import RGCNConv, kaiming_init_linears, make_classifier   # per-relation mean (pinned path)
from ...gnn.model import text_linear


class _SpMM(torch.autograd.Function):
    """``A @ x`` for a constant sparse CSR ``A`` with a precomputed transpose ``At`` for the backward
    (``At @ g``), so neither direction needs atomics or a per-call CSR transpose."""

    @staticmethod
    def forward(ctx, x, A, At):
        ctx.At = At
        return A @ x

    @staticmethod
    def backward(ctx, g):
        return ctx.At @ g, None, None


class MeanAdjacency:
    """Per-relation in-neighbour operators of a fixed graph for ``RGCNConv``'s mean aggregation:
    ``h_r = (C_r @ x) / clamp(deg_r, 1)`` with ``C_r[i, j]`` = number of ``j -> i`` edges of type ``r``
    (CSR, plus its transpose for the backward). This is PyG's ``scatter(..., reduce='mean')`` (sum of
    the messages, then divide by the float in-degree clamped at 1) with the per-edge gather + atomic
    scatter replaced by one cuSPARSE SpMM per relation; only the summation order differs."""

    def __init__(self, edge_index: torch.Tensor, edge_type: torch.Tensor, n: int, num_relations: int,
                 dtype=torch.float32):
        import warnings
        self.edge_index, self.edge_type, self.n, self.dtype = edge_index, edge_type, n, dtype  # identity keys
        src, dst = edge_index[0], edge_index[1]
        self.ops = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*[Ss]parse (CSR tensor support|invariant checks)")
            for r in range(num_relations):
                m = edge_type == r
                s, d = src[m], dst[m]
                ones = torch.ones(s.numel(), dtype=dtype, device=s.device)
                A = torch.sparse_coo_tensor(torch.stack([d, s]), ones, (n, n)).coalesce().to_sparse_csr()
                At = torch.sparse_coo_tensor(torch.stack([s, d]), ones, (n, n)).coalesce().to_sparse_csr()
                deg = torch.bincount(d, minlength=n).to(dtype).clamp_(min=1).unsqueeze(1)
                self.ops.append((A, At, deg))

    def matches(self, edge_index, edge_type, n, dtype) -> bool:
        return (edge_index is self.edge_index and edge_type is self.edge_type and n == self.n
                and dtype == self.dtype)

    def mean(self, x: torch.Tensor, r: int) -> torch.Tensor:
        A, At, deg = self.ops[r]
        return _SpMM.apply(x, A, At) / deg


def rgcn_full_graph(conv: RGCNConv, x: torch.Tensor, adj: MeanAdjacency) -> torch.Tensor:
    """``bb.gnn.RGCNConv`` (per-relation mean, root weight, bias; PyG's non-segment path) on ``adj``,
    with PyG's operation order: ``out = sum_r h_r @ W_r``, then ``+ x @ root``, then ``+ bias``."""
    c = conv.conv
    out = None
    for r in range(c.num_relations):
        t = adj.mean(x, r) @ c.weight[r]
        out = t if out is None else out + t
    out = out + x @ c.root
    return out + c.bias


_ADJ: dict = {}   # one cached MeanAdjacency (module-level, so state_dict / deepcopy never carry it)


class TiedRGCN(nn.Module):
    """One ``RGCNConv(D, D, R)`` (mean per relation, root weight, bias) applied ``applications``
    times with dropout between and no activation (official ``model.py``: weights tied).

    ``full_graph=True`` (full-batch training; efficiency review round 2): the two applications run
    on per-relation sparse CSR operators built once for the graph tensors passed in (``MeanAdjacency``,
    cached by tensor identity) instead of PyG's per-call edge masking, per-edge message gather and
    atomic scatter. Same parameters and function (tests compare it with the PyG path). Sampled
    subgraphs change every batch and keep the PyG path (``full_graph=False``)."""

    def __init__(self, hidden: int, num_relations: int = 2, dropout: float = 0.3, applications: int = 2,
                 full_graph: bool = False):
        super().__init__()
        self.conv = RGCNConv(hidden, hidden, num_relations=num_relations)
        self.dropout = nn.Dropout(dropout)
        self.applications = applications
        self.full_graph = full_graph

    def _adjacency(self, edge_index, edge_type, x) -> MeanAdjacency:
        adj = _ADJ.get("adj")
        if adj is None or not adj.matches(edge_index, edge_type, x.shape[0], x.dtype):
            _ADJ.clear()   # free the old operators before building the new ones
            adj = _ADJ["adj"] = MeanAdjacency(edge_index, edge_type, x.shape[0], self.conv.conv.num_relations,
                                              x.dtype)
        return adj

    def forward(self, x, edge_index, edge_type):
        if self.full_graph:
            adj = self._adjacency(edge_index, edge_type, x)
            conv = lambda h: rgcn_full_graph(self.conv, h, adj)  # noqa: E731
        else:
            conv = lambda h: self.conv(h, edge_index, edge_type)  # noqa: E731
        x = conv(x)
        for _ in range(self.applications - 1):
            x = self.dropout(x)
            x = conv(x)
        return x


class LinearPair(nn.Module):
    """"w/o graph" (benchmark Table 3, spec 4.1): the two R-GCN applications become
    ``Linear(D, D)`` -> dropout -> ``Linear(D, D)``; ``tied=True`` reuses one Linear twice."""

    def __init__(self, hidden: int, dropout: float = 0.3, tied: bool = False):
        super().__init__()
        self.l1 = nn.Linear(hidden, hidden)
        self.l2 = self.l1 if tied else nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index=None, edge_type=None):
        return self.l2(self.dropout(self.l1(x)))


class MLP2(nn.Module):
    """"w/o graph" read literally (benchmark App. B.4: "we remove the R-GCN ... while using two MLP
    layers with user features"): ``cat(des, tweet, num, cat) -> Linear(1544 [+2 ExtScore], D) -> LeakyReLU
    -> Dropout -> Linear(D, 2)``. No input projections, no BotRGCN encoder/head. Review round 1: on
    TwiBot-20 this reproduces the published w/o-graph row (82.6 / 83.8), the encoder-kept ``wog`` does not.

    The first layer is one ``nn.Linear`` (kaiming fan_in = full width, as ``init_weights`` would do on
    the concatenation) applied block-wise, so the [N, 1544] concatenation is never materialised and
    the two 768-wide blocks use the fp16 text path of the skeleton."""

    def __init__(self, hidden: int, num_dim: int, cat_dim: int, dropout: float = 0.3, text_dim: int = 768,
                 slope: float = 0.01):
        super().__init__()
        self.text_dim, self.num_dim = text_dim, num_dim
        self.l1 = nn.Linear(2 * text_dim + num_dim + cat_dim, hidden)
        self.l2 = nn.Linear(hidden, 2)
        self.act = nn.LeakyReLU(slope)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g):
        w, t, k = self.l1.weight, self.text_dim, self.num_dim
        h = (text_linear(g.des, w[:, :t]) + text_linear(g.tweet, w[:, t:2 * t])
             + Fn.linear(g.num, w[:, 2 * t:2 * t + k]) + Fn.linear(g.cat, w[:, 2 * t + k:], self.l1.bias))
        return self.l2(self.dropout(self.act(h)))


def build_model(kind: str, *, hidden: int, num_dim: int, cat_dim: int, num_relations: int = 2,
                dropout: float = 0.3, extscore_width: int = 0, extscore_inject: str = "num", wog_tied: bool = False,
                init: bool = True, full_graph: bool = False):
    """``kind`` = "botrgcn" | "wog" | "mlp2". ``num_dim`` must already include ``extscore_width`` when
    ``extscore_inject == "num"`` (``base + J.width``). ``full_graph`` selects the cached sparse R-GCN
    operators (``TiedRGCN``; full-batch training on one fixed graph)."""
    if kind == "mlp2":
        if extscore_width and extscore_inject != "num":
            raise ValueError("mlp2 takes ExtScore only through num_prop (--extscore-inject num)")
        m = MLP2(hidden, num_dim, cat_dim, dropout)
        return kaiming_init_linears(m) if init else m
    if kind == "botrgcn":
        factory = lambda: TiedRGCN(hidden, num_relations, dropout, full_graph=full_graph)  # noqa: E731
    elif kind == "wog":
        factory = lambda: LinearPair(hidden, dropout, tied=wog_tied)  # noqa: E731
    else:
        raise ValueError(kind)
    m = make_classifier(factory, hidden=hidden, num_dim=num_dim, cat_dim=cat_dim, extscore_width=extscore_width,
                        extscore_inject=extscore_inject)
    return kaiming_init_linears(m) if init else m
