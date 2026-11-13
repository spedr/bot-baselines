"""GCN / GAT bodies of the TwiBot-22 benchmark (``src/GCN_GAT/model.py``) on the ``bb.gnn`` skeleton.

    x = Dropout(p)(cat(d, t, n, c))  ->  LeakyReLU(Linear(H, H))      (skeleton InputEncoder, input_dropout=p)
    x = Conv1(x) -> Dropout(p) -> Conv2(x)                            (this body; NO activation between/after)
    logits = Linear(H, 2)(LeakyReLU(Linear(H, H)(x)))                 (skeleton OutputHead)

GCN: ``GCNConv(H, H)`` x 2. GAT: ``GATConv(H, H/4, heads=4)`` (concat -> H), ``GATConv(H, H, heads=1)``.
Init: PyTorch / PyG defaults (no kaiming re-init, spec 4).

Two entry points with the same parameters:

- ``forward(x, edge_index)``: the PyG layers' own semantics on a raw graph (GCN:
  ``add_remaining_self_loops`` + symmetric normalisation on the in-degree, ``gcn_norm``; GAT:
  ``remove_self_loops`` + ``add_self_loops``). The convs are built with ``normalize=False`` /
  ``add_self_loops=False`` and this method applies exactly those PyG functions first.
- ``forward_layers(x, ei1, w1, ei2, w2)``: per-layer edge sets prepared on the host
  (``train.make_plan``): layer 1 only over edges into the nodes layer 2 reads, layer 2 only over
  edges into the seeds, GCN weights from the in-degrees of the whole sampled subgraph. The outputs at
  the seeds are identical; the work is a fraction (tested against ``forward``).
"""
from __future__ import annotations

from torch import nn
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops, remove_self_loops

from ...gnn import make_classifier

CONVS = ("gcn", "gat")


class TwoConv(nn.Module):
    def __init__(self, conv: str, hidden: int = 128, dropout: float = 0.3, heads: int = 4):
        super().__init__()
        self.kind = conv
        if conv == "gcn":
            self.conv1 = GCNConv(hidden, hidden, normalize=False)
            self.conv2 = GCNConv(hidden, hidden, normalize=False)
        elif conv == "gat":
            if hidden % heads:
                raise ValueError("hidden must be divisible by heads")
            self.conv1 = GATConv(hidden, hidden // heads, heads=heads, add_self_loops=False)
            self.conv2 = GATConv(hidden, hidden, heads=1, add_self_loops=False)
        else:
            raise ValueError(conv)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type=None):
        n = x.size(0)
        if self.kind == "gcn":
            ei, w = gcn_norm(edge_index, None, n, add_self_loops=True, dtype=x.dtype)
            return self.forward_layers(x, ei, w, ei, w)
        ei, _ = add_self_loops(remove_self_loops(edge_index)[0], num_nodes=n)
        return self.forward_layers(x, ei, None, ei, None)

    def forward_layers(self, x, ei1, w1, ei2, w2):
        if self.kind == "gcn":
            return self.conv2(self.dropout(self.conv1(x, ei1, w1)), ei2, w2)
        return self.conv2(self.dropout(self.conv1(x, ei1)), ei2)


def forward_layers(model, sub, ei1, w1, ei2, w2):
    """``GNNClassifier.forward`` with the body's per-layer entry point (same encoder/head code path)."""
    ex = sub.extra if model.extscore_inject == "branch" else None
    x = model.encoder(sub.des, sub.tweet, sub.num, sub.cat, ex)
    x = model.body.forward_layers(x, ei1, w1, ei2, w2)
    return model.head(x, sub.extra if model.extscore_inject == "late" else None)


def build_model(conv: str, *, hidden: int = 128, num_dim: int = 5, cat_dim: int = 3, dropout: float = 0.3,
                extscore_width: int = 0, extscore_inject: str = "num"):
    """``num_dim`` must already include ``extscore_width`` when ``extscore_inject == "num"`` (base + J.width)."""
    return make_classifier(lambda: TwoConv(conv, hidden, dropout), hidden=hidden, num_dim=num_dim,
                           cat_dim=cat_dim, extscore_width=extscore_width, extscore_inject=extscore_inject, input_dropout=dropout)


def reference_model(conv: str, hidden: int = 128, dropout: float = 0.3, **kw):
    """The same architecture with stock PyG layers (default self-loop/normalisation handling), for
    tests: load ``build_model``'s state_dict into it and compare."""
    class Ref(nn.Module):
        def __init__(self):
            super().__init__()
            if conv == "gcn":
                self.conv1, self.conv2 = GCNConv(hidden, hidden), GCNConv(hidden, hidden)
            else:
                self.conv1, self.conv2 = GATConv(hidden, hidden // 4, heads=4), GATConv(hidden, hidden, heads=1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, edge_index, edge_type=None):
            return self.conv2(self.dropout(self.conv1(x, edge_index)), edge_index)

    return make_classifier(Ref, hidden=hidden, input_dropout=dropout, **kw)
