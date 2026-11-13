"""SEBot model (spec docs/methods/sebot.md section 4): three views with their own feature encoders.

- ``SEPU``  (alpha): GCN U-Net over the whole-graph coding tree (levels 0-2), SEPool with MLP+BN.
- ``SEPG``  (beta):  GCN + pure-sum tree pooling over each labelled user's ego-graph tree, mean readout.
- ``Backbone`` (gamma): two FACNConv layers (signed Gumbel/tanh edge weights + RCM) on an
  edge-dropped graph, no activation between them.
- ``SEBot``: classifier on [h_a | h_b | h_g] + InfoNCE(alpha, gamma) + InfoNCE(beta, gamma).

Written from the paper and the official code (846468230/SEBot @ 59b5407); no code copied.
"""
from __future__ import annotations

import torch
import torch.nn.functional as Fn
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GCNConv

from ...gnn.model import _text_proj


class FeatureEncoder(nn.Module):
    """des/tweet/num/cat -> Linear(., input_dim/4) + LeakyReLU each -> concat (d, t, n, c) ->
    Linear(input_dim, hidden) -> PReLU(hidden)."""

    def __init__(self, num_dim: int, cat_dim: int, input_dim: int = 128, hidden: int = 32,
                 des_dim: int = 768, tweet_dim: int = 768):
        super().__init__()
        q = input_dim // 4
        self.des = nn.Linear(des_dim, q)
        self.tweet = nn.Linear(tweet_dim, q)
        self.num = nn.Linear(num_dim, q)
        self.cat = nn.Linear(cat_dim, q)
        self.act = nn.LeakyReLU()
        self.input = nn.Linear(input_dim, hidden)
        self.prelu = nn.PReLU(hidden)

    ckpt = False      # set by SEBot(checkpoint_encoders=True): recompute in backward (same values, less VRAM)

    def forward(self, des, tweet, num, cat):
        if self.ckpt and torch.is_grad_enabled():
            return checkpoint(self._forward, des, tweet, num, cat, use_reentrant=False)
        return self._forward(des, tweet, num, cat)

    def _forward(self, des, tweet, num, cat):
        a = self.act
        x = torch.cat([a(_text_proj(self.des, des)), a(_text_proj(self.tweet, tweet)), a(self.num(num)),
                       a(self.cat(cat))], dim=1)
        return self.prelu(self.input(x))


# ------------------------------------------------------------------------------------------------
# gamma view


def _in_degree(dst, n: int, dtype):
    """Relation in-degree clamped at 1 (exact: integer counts in float)."""
    return torch.zeros(n, device=dst.device, dtype=dtype).index_add_(
        0, dst, torch.ones_like(dst, dtype=dtype)).clamp_(min=1.0)


class FACNConv(nn.Module):
    """Heterophily-aware relational conv (official ``BackBone/rgcn.py::FACNConv``).

    Per relation r (messages source -> target): x_r = x W_r; a = [x_r,i | x_r,j] att_r + logit(eps),
    eps ~ U(0,1) one scalar per relation per call (also in eval unless ``eval_noise=False``, then
    eps = 0.5); w = dropout(tanh(a / tau)); h_r,i = sum_j x_r,j w_ij / indeg_r(i).
    RCM: U = softmax_r(LeakyReLU(Linear([h_1 | ... | h_R]))) per channel; out = sum_r h_r * U_r + root(x)."""

    def __init__(self, in_dim: int, out_dim: int, num_relations: int = 2, tau: float = 0.01, rcm: bool = True,
                 att_dropout: float = 0.3, eval_noise: bool = True):
        super().__init__()
        self.R, self.out_dim, self.tau, self.rcm = num_relations, out_dim, tau, rcm
        self.att_dropout, self.eval_noise = att_dropout, eval_noise
        self.root = nn.Linear(in_dim, out_dim)
        self.weight = nn.Parameter(torch.empty(num_relations, in_dim, out_dim))
        self.relation_weight = nn.Linear(num_relations * out_dim, num_relations * out_dim)
        self.att = nn.Parameter(torch.empty(num_relations, 2 * out_dim, 1))
        self.act = nn.LeakyReLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.root.weight)
        nn.init.kaiming_uniform_(self.weight)
        nn.init.kaiming_uniform_(self.relation_weight.weight)
        nn.init.kaiming_uniform_(self.att)
        nn.init.zeros_(self.root.bias)
        nn.init.zeros_(self.relation_weight.bias)

    def forward(self, x, rel_edges: list):
        n = x.shape[0]
        root = self.root(x)
        hs = []
        for r, re in enumerate(rel_edges):
            src, dst = re[0], re[1]
            xr = x @ self.weight[r]
            s_i = xr @ self.att[r, :self.out_dim]          # target part of [x_i | x_j]
            s_j = xr @ self.att[r, self.out_dim:]          # source part
            a = s_i[dst, 0] + s_j[src, 0]
            if self.training or self.eval_noise:
                eps = torch.rand(1, device=x.device)
                a = a + torch.log(eps / (1 - eps))
            w = torch.tanh(a / self.tau)
            w = Fn.dropout(w, p=self.att_dropout, training=self.training)
            deg = re[2] if len(re) > 2 else _in_degree(dst, n, x.dtype)
            msg = xr[src] / deg[dst].unsqueeze(-1) * w.unsqueeze(-1)
            hs.append(torch.zeros(n, self.out_dim, device=x.device, dtype=msg.dtype).index_add_(0, dst, msg))
        if self.rcm:
            tot = torch.cat(hs, dim=1)
            u = self.act(self.relation_weight(tot)).view(-1, self.R, self.out_dim).softmax(dim=1)
            return (tot.view(-1, self.R, self.out_dim) * u).sum(1) + root
        return torch.stack(hs).sum(0) + root


class Backbone(nn.Module):
    """gamma view: encoder -> FACN -> dropout -> FACN -> dropout -> Linear (no activation between)."""

    def __init__(self, num_dim, cat_dim, *, hidden=32, input_dim=128, num_relations=2, dropout=0.5, pe=0.2,
                 conv: str = "facn", eval_noise=True, rcm=True, tau=0.01):
        super().__init__()
        self.encoder = FeatureEncoder(num_dim, cat_dim, input_dim, hidden)
        self.conv_kind = conv
        if conv == "facn":
            self.conv1 = FACNConv(hidden, hidden, num_relations, tau=tau, rcm=rcm, eval_noise=eval_noise)
            self.conv2 = FACNConv(hidden, hidden, num_relations, tau=tau, rcm=rcm, eval_noise=eval_noise)
        elif conv == "rgcn":     # "R-GCN encoder" ablation (paper Table 4)
            # bb.gnn.RGCNConv pins PyG's per-relation path (raw PyG RGCNConv may switch by a size
            # heuristic to segment_matmul, which averages over all relations' edges together)
            from ...gnn import RGCNConv
            self.conv1 = RGCNConv(hidden, hidden, num_relations)
            self.conv2 = RGCNConv(hidden, hidden, num_relations)
        else:
            raise ValueError(conv)
        self.classifier = nn.Linear(hidden, hidden)
        self.dropout, self.pe, self.R = dropout, pe, num_relations

    def forward(self, des, tweet, num, cat, edge_index, edge_type):
        if self.training and self.pe > 0:
            keep = torch.bernoulli(torch.full((edge_index.shape[1],), 1.0 - self.pe)).bool().to(edge_index.device)
            edge_index, edge_type = edge_index[:, keep], edge_type[keep]
        x = self.encoder(des, tweet, num, cat)
        if self.conv_kind == "facn":
            rel = []
            for r in range(self.R):
                m = edge_type == r
                dst = edge_index[1, m]
                rel.append((edge_index[0, m], dst, _in_degree(dst, x.shape[0], x.dtype)))   # shared by both layers
            x = self.conv1(x, rel)
            x = Fn.dropout(x, self.dropout, self.training)
            x = self.conv2(x, rel)
        else:
            x = self.conv1(x, edge_index, edge_type)
            x = Fn.dropout(x, self.dropout, self.training)
            x = self.conv2(x, edge_index, edge_type)
        x = Fn.dropout(x, self.dropout, self.training)
        return self.classifier(x)


# ------------------------------------------------------------------------------------------------
# alpha view


def _pool(x, parent, n_parent):
    """Sum children into parents (``SEPooling``, flow target_to_source, aggr add)."""
    return torch.zeros(n_parent, x.shape[1], device=x.device, dtype=x.dtype).index_add_(0, parent, x)


class SEPU(nn.Module):
    """alpha view (official ``SEPN/sep_u.py``, num_blocks = 2, conv = GCN, link_input off)."""

    def __init__(self, num_dim, cat_dim, *, hidden=32, input_dim=128, conv_dropout=0.5, pool_dropout=0.5):
        super().__init__()
        h = hidden
        dims = [(h, h), (h, h), (h, h), (2 * h, h), (2 * h, h)]
        self.convs = nn.ModuleList([GCNConv(i, o, cached=True) for i, o in dims])
        self.pools = nn.ModuleList([nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.BatchNorm1d(h)) for _ in range(4)])
        self.classifier = nn.Linear(h, h)
        self.encoder = FeatureEncoder(num_dim, cat_dim, input_dim, hidden)
        self.d1, self.d2 = conv_dropout, pool_dropout

    def forward(self, des, tweet, num, cat, T):
        """``T``: dict with e0 (level-0 GCN edges = the relational undirected graph), e1, e2
        (quotient edges), p1 [N] (leaf -> level-1), p2 [n1] (level-1 -> level-2), n1, n2."""
        d = lambda z, p: Fn.dropout(z, p, self.training)  # noqa: E731
        x = self.encoder(des, tweet, num, cat)
        x = d(Fn.relu(self.convs[0](x, T["e0"])), self.d1)
        xs0 = x
        x = d(Fn.relu(self.pools[0](_pool(x, T["p1"], T["n1"]))), self.d2)
        x = d(Fn.relu(self.convs[1](x, T["e1"])), self.d1)
        xs1 = x
        x = d(Fn.relu(self.pools[1](_pool(x, T["p2"], T["n2"]))), self.d2)
        x = d(Fn.relu(self.convs[2](x, T["e2"])), self.d1)
        x = d(Fn.relu(self.pools[2](x[T["p2"]])), self.d2)          # unpool level 2 -> 1
        x = torch.cat([x, xs1], dim=1)
        x = d(Fn.relu(self.convs[3](x, T["e1"])), self.d1)
        x = d(Fn.relu(self.pools[3](x[T["p1"]])), self.d2)          # unpool level 1 -> 0
        x = torch.cat([x, xs0], dim=1)
        x = self.convs[4](x, T["e0"])
        return self.classifier(x)


# ------------------------------------------------------------------------------------------------
# beta view


class SEPG(nn.Module):
    """beta view (official ``SEPG/sep_g.py``: num_convs = 3, GCN, average readout, link_input off,
    tree pooling is a pure sum; its MLP is never applied upstream and is not created here)."""

    def __init__(self, num_dim, cat_dim, *, k: int, hidden=32, input_dim=128, num_convs=3, dropout=0.5):
        super().__init__()
        if k < 3:
            raise ValueError("SEP-G needs tree depth k >= 3")
        self.k, self.num_convs = k, num_convs
        self.convs = nn.ModuleList([GCNConv(hidden, hidden, cached=True) for _ in range(num_convs)])
        self.classifier = nn.Sequential(nn.Linear(hidden * num_convs, hidden), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(hidden, hidden))
        self.encoder = FeatureEncoder(num_dim, cat_dim, input_dim, hidden)

    def forward(self, des, tweet, num, cat, S):
        """``S``: dict with uniq [U] (dataset rows to encode), gather [S0] (index into uniq of each
        leaf), edges[c] (c < num_convs), parent[d] (d >= 1), size[d], batch[d], n_graphs."""
        xe = self.encoder(des[S["uniq"]], tweet[S["uniq"]], num[S["uniq"]], cat[S["uniq"]])
        x = xe[S["gather"]]
        xs = []
        for c in range(self.num_convs):
            x = Fn.relu(self.convs[c](x, S["edges"][c]))
            if c < self.k - 1:
                x = Fn.relu(_pool(x, S["parent"][c + 1], S["size"][c + 1]))
            xs.append(x)
        pooled = []
        G = S["n_graphs"]
        for c, x in enumerate(xs):
            lvl = min(c + 1, self.k - 1)
            b = S["batch"][lvl]
            s = torch.zeros(G, x.shape[1], device=x.device, dtype=x.dtype).index_add_(0, b, x)
            pooled.append(s / S["count"][lvl].unsqueeze(1))
        return self.classifier(torch.cat(pooled, dim=1))


# ------------------------------------------------------------------------------------------------
# full model


class _NTXentLSE(torch.autograd.Function):
    """sum_i logsumexp_{k != i} (z_i . z_k / tau) over the rows of z [2n, d], by row blocks.

    Forward keeps only z / tau and the per-row lse (autograd through the blocks kept or recomputed
    several block x 2n temporaries). Backward recomputes each block once as P = exp(S - lse)
    (diagonal 0; the -lse shift is fused into the GEMM) and accumulates dz = (P z + P^T z_blk) / tau,
    the exact gradient of logsumexp through the symmetric similarity matrix."""

    @staticmethod
    def forward(ctx, z, tau: float, block: int):
        n2 = z.shape[0]
        zs, zt = z / tau, z.t()
        lse = z.new_empty(n2)
        for s in range(0, n2, block):
            e = min(n2, s + block)
            L = torch.mm(zs[s:e], zt)
            L[:, s:e].diagonal().fill_(float("-inf"))
            mx = L.amax(dim=1)                       # = torch.logsumexp, in place
            lse[s:e] = L.sub_(mx[:, None]).exp_().sum(dim=1).log_().add_(mx)
        ctx.save_for_backward(z, zs, lse)
        ctx.block = block
        ctx.tau = tau
        return lse.sum()

    @staticmethod
    def backward(ctx, g):
        z, zs, lse = ctx.saved_tensors
        n2 = z.shape[0]
        zt = z.t()
        gz = torch.zeros_like(z)
        for s in range(0, n2, ctx.block):
            e = min(n2, s + ctx.block)
            P = torch.addmm(lse[s:e, None], zs[s:e], zt, beta=-1)       # S - lse
            P[:, s:e].diagonal().fill_(float("-inf"))
            P.exp_()
            gz[s:e].addmm_(P, z)
            gz.addmm_(P.t(), z[s:e])
        return gz.mul_(g / ctx.tau), None, None


def info_nce(a: torch.Tensor, b: torch.Tensor, tau: float = 0.1, block: int = 2048) -> torch.Tensor:
    """NT-Xent over the 2n x 2n cosine-similarity matrix (official ``infonce_loss``): positive pair
    (i, i+n); denominator over all k != i (the positive included); mean over the 2n rows.
    Computed by row blocks with a hand-written backward (``_NTXentLSE``; same value and gradient as
    the dense formula, O(block * 2n) temporary memory, nothing per block kept for backward)."""
    z = Fn.normalize(torch.cat([a, b], dim=0), dim=1)
    n2 = z.shape[0]
    pos = (z * z.roll(-(n2 // 2), 0)).sum(1) / tau        # z[(i + n) mod 2n]
    return (_NTXentLSE.apply(z, tau, block) - pos.sum()) / n2


class SEBot(nn.Module):
    def __init__(self, num_dim, cat_dim, *, k: int, hidden=32, input_dim=128, proj_dim=16, num_relations=2,
                 dropout=0.5, conv_dropout=0.5, pool_dropout=0.5, pe=0.2, eval_noise=True, rcm=True,
                 backbone_conv="facn", use_u=True, use_g=True, extscore_late=0, gamma_extra=0, tau_g=0.01,
                 checkpoint_encoders=False):
        super().__init__()
        self.use_u, self.use_g = use_u, use_g
        h = hidden
        self.sep_u = SEPU(num_dim, cat_dim, hidden=h, input_dim=input_dim, conv_dropout=conv_dropout,
                          pool_dropout=pool_dropout) if use_u else None
        self.sep_g = SEPG(num_dim, cat_dim, k=k, hidden=h, input_dim=input_dim, dropout=pool_dropout) if use_g else None
        self.backbone = Backbone(num_dim + gamma_extra, cat_dim, hidden=h, input_dim=input_dim,
                                 num_relations=num_relations, dropout=dropout, pe=pe, conv=backbone_conv,
                                 eval_noise=eval_noise, rcm=rcm, tau=tau_g)
        n_views = 1 + int(use_u) + int(use_g)
        self.classifier = nn.Sequential(nn.Linear(h * n_views + extscore_late, h), nn.LeakyReLU(), nn.Linear(h, 2))
        self.proj_u = nn.Sequential(nn.Linear(h, proj_dim), nn.LeakyReLU(), nn.Linear(proj_dim, h)) if use_u else None
        self.proj_g = nn.Sequential(nn.Linear(h, proj_dim), nn.LeakyReLU(), nn.Linear(proj_dim, h)) if use_g else None
        self.extscore_late, self.gamma_extra = extscore_late, gamma_extra
        for m in self.modules():
            if isinstance(m, FeatureEncoder):
                m.ckpt = checkpoint_encoders

    def views(self, g, T, S, lab):
        """h_alpha, h_beta, h_gamma for the labelled rows ``lab`` (in order)."""
        num_c = torch.cat([g.num, g.extra], dim=1) if self.gamma_extra else g.num
        h_c = self.backbone(g.des, g.tweet, num_c, g.cat, g.edge_index, g.edge_type)[lab]
        h_u = self.sep_u(g.des, g.tweet, g.num, g.cat, T)[lab] if self.use_u else None
        h_g = self.sep_g(g.des, g.tweet, g.num, g.cat, S) if self.use_g else None
        return h_u, h_g, h_c

    def logits(self, h_u, h_g, h_c, late=None):
        parts = [x for x in (h_u, h_g, h_c) if x is not None]
        if self.extscore_late:
            parts.append(late)
        return self.classifier(torch.cat(parts, dim=1))

    def contrastive(self, h_u, h_g, h_c, tau: float, block: int):
        l1 = info_nce(self.proj_u(h_u), self.proj_u(h_c), tau, block) if self.use_u else h_c.new_zeros(())
        l2 = info_nce(self.proj_g(h_g), self.proj_g(h_c), tau, block) if self.use_g else h_c.new_zeros(())
        return l1, l2


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

