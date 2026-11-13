"""BIC ``variant: v2`` (docs/methods/bic.md section 4; TwiBot20v2 ``InteractModel`` semantics, d = 768).

    graph input   x_u = [Lin(c->d/4)(cat), Lin(k->d/4)(num), Lin(768->d/4)(tweet_mean), Lin(768->d/4)(des)]
                  (no activation; order cat, num, tweet, des)
    text input    S_i = [des_i, tweet_1 .. tweet_cap]  ([B, 1+cap, 768], zero rows after the last tweet)
    layer l=1,2   text : pre-LN transformer layer (4 heads, FFN width d, LeakyReLU, dropout 0.1, no
                         key-padding mask) -> S, M_l = head-averaged attention weights [B, L, L]
                  graph: GCNConv(d, d) over the batch star {j -> i : i target}, then for each target
                         MHA_g(query h_i, keys/values [h_i, h_n1 .. h_nJ]) (4 heads, no dropout,
                         no residual; computed as a segment softmax over flat lists); neighbours
                         keep their GCN outputs
                  consistency: P_l = MaxPool(p)(zero-pad(M_l, K*p)), p = ceil(L/K), K = 16
    interaction   after layer 1 only, on t = S[:,0] and g = target graph vector (``code`` form):
                  (a,b) = softmax(<t, W_t t + b_t>, <g, t>), t' = a t + b g
                  (c,d) = softmax(<g, W_g g + b_g>, <g, t>), g' = c g + d t
    readout       logits = Lin(768 [+2 ExtScore] -> 2)([Lin(2K^2->256)(P_1, P_2), Lin(d->256)(S2[:,0]), Lin(d->256)(g2)])
                  no nonlinearity, no dropout

Unused modules of the official code (third RespectiveLayer, second InteractLayer, ``text_linear``,
the never-applied dropout 0.5) are not created.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as Fn
from torch import nn
from torch_geometric.nn import GCNConv

# Ablations. ``wo_graph`` / ``wo_text`` follow the paper (Sec. 4.3, "Modality Effectiveness Study":
# "We remove the text modality representation h_int and the graph modality representation g_int in
# equation (3)"): only the readout loses that representation; both modules and the interaction still
# run, so graph information (incl. the profile features) still reaches the text head through the
# interaction. ``wo_graph_module`` / ``wo_text_module`` remove the whole module (and the interaction).
ABLATIONS = ("none", "wo_graph", "wo_text", "wo_interaction", "wo_consistency", "wo_graph_module",
             "wo_text_module")


class TextLayer(nn.Module):
    """``nn.TransformerEncoderLayer(norm_first=True)`` that also returns the head-averaged attention
    weights (PyTorch MHA returns them after attention dropout in training mode, as the official
    layer does). Parameter creation order follows ``TransformerEncoderLayer``."""

    def __init__(self, d: int = 768, heads: int = 4, ffn: int = 768, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d, ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.norm1(s)
        a, w = self.self_attn(x, x, x, need_weights=True, average_attn_weights=True)
        s = s + self.dropout1(a)
        s = s + self.dropout2(self.linear2(self.dropout(self.act(self.linear1(self.norm2(s))))))
        return s, w


def list_attention(mha: nn.MultiheadAttention, h: torch.Tensor, n_targets: int, seg: torch.Tensor,
                   node: torch.Tensor) -> torch.Tensor:
    """``mha(query=h_i, keys=values=[h_j for j in list_i])`` for every target i at once, with the
    lists given flat (``seg`` = target of each entry, ``node`` = row of ``h``). Same parameters and
    function as ``nn.MultiheadAttention`` without dropout, computed as a segment softmax so memory
    is O(total list length), not O(B x longest list). K/V are projected once per node."""
    from torch_geometric.utils import softmax as seg_softmax
    d, H = mha.embed_dim, mha.num_heads
    dh = d // H
    w, b = mha.in_proj_weight, mha.in_proj_bias
    q = Fn.linear(h[:n_targets], w[:d], b[:d]).view(n_targets, H, dh)
    kv = Fn.linear(h, w[d:], b[d:])                                         # [V, 2d]
    k, v = kv[:, :d].view(-1, H, dh), kv[:, d:].view(-1, H, dh)
    score = (q[seg] * k[node]).sum(-1) / math.sqrt(dh)                     # [M, H]
    alpha = seg_softmax(score, seg, num_nodes=n_targets)                   # [M, H]
    out = torch.zeros((n_targets, H, dh), dtype=v.dtype, device=v.device)
    out.index_add_(0, seg, alpha.unsqueeze(-1).to(v.dtype) * v[node])
    return mha.out_proj(out.reshape(n_targets, d))


class GraphLayer(nn.Module):
    """``gnn='gcn'``: TwiBot20v2 code (GCNConv, relations merged). ``gnn='rgcn'``: the paper's text
    (Sec. 3.3, "BIC adopts relational graph convolutional networks"): RGCNConv (per-relation mean +
    root weight) over the same star edges; a diagnostic, not the reference."""

    def __init__(self, d: int = 768, heads: int = 4, gnn: str = "gcn", num_relations: int = 2):
        super().__init__()
        self.gnn = gnn
        if gnn == "gcn":
            # self-loops are added by the closure builder (PyG ``add_remaining_self_loops`` semantics)
            # so that the forward has no data-dependent host sync
            self.gcn = GCNConv(d, d, add_self_loops=False)
        elif gnn == "rgcn":
            from torch_geometric.nn import RGCNConv
            self.gcn = RGCNConv(d, d, num_relations=num_relations)
        else:
            raise ValueError(gnn)
        self.attn = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)

    def forward(self, h, edge_index, att_seg, att_node, n_targets: int, edge_type=None):
        h = self.gcn(h, edge_index) if self.gnn == "gcn" else self.gcn(h, edge_index, edge_type)
        out = list_attention(self.attn, h, n_targets, att_seg, att_node)
        return torch.cat([out, h[n_targets:]], dim=0)


class Interaction(nn.Module):
    """``code``: W_t, W_g with bias on the self terms, plain dot product on the cross terms.
    ``paper`` (Eq. 2 as the spec reads it, [U]): bias-free theta_1 (text side) and theta_2 (graph
    side) on all four scores: <t, th1 t>, <t, th2 g>, <g, th2 g>, <g, th1 t>."""

    def __init__(self, d: int = 768, form: str = "code"):
        super().__init__()
        if form not in ("code", "paper"):
            raise ValueError(form)
        self.form = form
        bias = form == "code"
        self.w_t = nn.Linear(d, d, bias=bias)
        self.w_g = nn.Linear(d, d, bias=bias)

    def forward(self, t, g):
        if self.form == "code":
            a, b = (t * self.w_t(t)).sum(-1), (g * t).sum(-1)
            c, dd = (g * self.w_g(g)).sum(-1), b
        else:
            a, b = (t * self.w_t(t)).sum(-1), (t * self.w_g(g)).sum(-1)
            c, dd = (g * self.w_g(g)).sum(-1), (g * self.w_t(t)).sum(-1)
        wt = torch.softmax(torch.stack([a, b], -1), -1)
        wg = torch.softmax(torch.stack([c, dd], -1), -1)
        t2 = wt[:, :1] * t + wt[:, 1:] * g
        g2 = wg[:, :1] * g + wg[:, 1:] * t
        return t2, g2


def consistency_pool(m: torch.Tensor, k: int = 16) -> torch.Tensor:
    """[B, L, L] -> [B, K, K]: zero-pad to K*ceil(L/K) on the right/bottom, MaxPool2d(ceil(L/K)).
    L = 201, K = 16: pad to 208, pool 13 (TwiBot20v2)."""
    L = m.shape[-1]
    p = math.ceil(L / k)
    pad = k * p - L
    if pad:
        m = Fn.pad(m, (0, pad, 0, pad))
    return Fn.max_pool2d(m.unsqueeze(1), kernel_size=p, stride=p).squeeze(1)


def _proj(lin: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    return lin(x if x.dtype == lin.weight.dtype else x.to(lin.weight.dtype))


class BIC(nn.Module):
    def __init__(self, *, num_dim: int, cat_dim: int, d: int = 768, text_dim: int = 768, heads: int = 4,
                 dropout: float = 0.1, layers: int = 2, k: int = 16, interaction: str = "code",
                 ablation: str = "none", extscore_head: int = 0, checkpoint_text: bool = False, gnn: str = "gcn",
                 num_relations: int = 2):
        super().__init__()
        self.checkpoint_text = checkpoint_text
        if ablation not in ABLATIONS:
            raise ValueError(ablation)
        if d % 4:
            raise ValueError("d must be divisible by 4")
        self.ablation, self.k, self.d, self.n_layers = ablation, k, d, layers
        self.use_graph = ablation != "wo_graph_module"
        self.use_text = ablation != "wo_text_module"
        self.use_inter = self.use_graph and self.use_text and ablation != "wo_interaction"
        self.use_cons = self.use_text and ablation != "wo_consistency"
        self.read_graph = self.use_graph and ablation != "wo_graph"     # g in the readout (Eq. 3)
        self.read_text = self.use_text and ablation != "wo_text"        # h in the readout (Eq. 3)
        # graph layers whose output is used: all, or only layer 1 (feeds the interaction) for wo_graph
        self.n_graph = layers if self.read_graph else (1 if self.use_inter else 0)
        q = d // 4
        if self.n_graph:
            self.in_cat = nn.Linear(cat_dim, q)
            self.in_num = nn.Linear(num_dim, q)
            self.in_tweet = nn.Linear(768, q)
            self.in_des = nn.Linear(768, q)
        if self.use_text and text_dim != d:
            raise ValueError("v2 has no text input projection: text_dim must equal d")
        self.text = nn.ModuleList([TextLayer(d, heads, d, dropout) for _ in range(layers)]) if self.use_text else None
        self.graph = nn.ModuleList([GraphLayer(d, heads, gnn, num_relations) for _ in range(self.n_graph)]) if self.n_graph else None
        self.inter = Interaction(d, interaction) if self.use_inter else None
        width = 0
        if self.use_cons:
            self.att = nn.Linear(layers * k * k, 256)
            width += 256
        if self.read_text:
            self.title = nn.Linear(d, 256)
            width += 256
        if self.read_graph:
            self.user = nn.Linear(d, 256)
            width += 256
        self.extscore_head = extscore_head
        self.out = nn.Linear(width + extscore_head, 2)

    def encode_graph(self, des, tweet, num, cat) -> torch.Tensor:
        return torch.cat([_proj(self.in_cat, cat), _proj(self.in_num, num), _proj(self.in_tweet, tweet),
                          _proj(self.in_des, des)], dim=1)

    def forward(self, tweets, des0, gdes, gtweet, gnum, gcat, edge_index, att_seg, att_node, n_targets=None, late=None,
                edge_type=None):
        """tweets [B, cap, 768] (fp16 ok), des0 [B, 768]: the text sequence; g*: closure node
        features (targets first, V rows); edge_index [2, E] local; att_seg/att_node [M] flat attention
        lists (self entries included)."""
        B = tweets.shape[0] if n_targets is None else n_targets
        wdt = self.out.weight.dtype
        s = torch.cat([des0.unsqueeze(1), tweets], dim=1).to(wdt) if self.use_text else None
        h = self.encode_graph(gdes, gtweet, gnum, gcat) if self.n_graph else None
        pools = []
        for l in range(self.n_layers):
            if self.use_text:
                if self.checkpoint_text and self.training and torch.is_grad_enabled():
                    # recompute the layer in backward (same function, RNG state preserved): the
                    # [B, 201, 768] activations of both layers are most of the training memory
                    from torch.utils.checkpoint import checkpoint
                    s, w = checkpoint(self.text[l], s, use_reentrant=False)
                else:
                    s, w = self.text[l](s)
                if self.use_cons:
                    pools.append(consistency_pool(w, self.k).flatten(1))
            if l < self.n_graph:
                h = self.graph[l](h, edge_index, att_seg, att_node, B, edge_type)
            if l == 0 and self.use_inter:
                t2, g2 = self.inter(s[:, 0], h[:B])
                s = torch.cat([t2.unsqueeze(1), s[:, 1:]], dim=1)
                h = torch.cat([g2, h[B:]], dim=0)
        parts = []
        if self.use_cons:
            parts.append(self.att(torch.cat(pools, dim=1).to(wdt)))
        if self.read_text:
            parts.append(self.title(s[:, 0]))
        if self.read_graph:
            parts.append(self.user(h[:B]))
        if self.extscore_head:
            if late is None:
                raise ValueError("model has a ExtScore head input but none was given")
            parts.append(late.to(parts[0].dtype))
        return self.out(torch.cat(parts, dim=1))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
