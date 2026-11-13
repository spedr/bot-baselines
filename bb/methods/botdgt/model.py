"""BotDGT model (spec section 4): feature embedding -> shared 2-layer graph transformer per snapshot
-> causal temporal self-attention with position signals -> per-snapshot FFN logits.

Written from the spec; nothing copied from the (unlicensed) official repository.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch_geometric.nn import TransformerConv


def _kaiming_normal_linear(lin, zero_bias: bool = False) -> None:
    nn.init.kaiming_normal_(lin.weight)
    if zero_bias and getattr(lin, "bias", None) is not None:
        nn.init.zeros_(lin.bias)


class FeatureEmbedding(nn.Module):
    """num, cat, des, tweet -> Linear + PReLU each (h/4), concat, Linear(h, h) + PReLU. All five
    activations are ONE shared single-parameter PReLU (spec 4). Widths follow the feature scheme
    (the official code hard-codes 5/3)."""

    def __init__(self, hidden: int, num_dim: int, cat_dim: int, text_dim: int = 768):
        super().__init__()
        q = hidden // 4
        self.num = nn.Linear(num_dim, q)
        self.cat = nn.Linear(cat_dim, q)
        self.des = nn.Linear(text_dim, q)
        self.tweet = nn.Linear(text_dim, q)
        self.out = nn.Linear(4 * q, hidden)
        self.act = nn.PReLU()
        for lin in (self.num, self.cat, self.des, self.tweet, self.out):
            _kaiming_normal_linear(lin, zero_bias=True)

    def forward(self, num, cat, des, tweet):
        a = self.act
        x = torch.cat([a(self.num(num)), a(self.cat(cat)), a(self.des(des)), a(self.tweet(tweet))], dim=1)
        return a(self.out(x))


def cumulative_transformer_conv(tc: TransformerConv, x, ei, eb, T: int, n_dst: int | None = None):
    """``tc(x, edges born <= t)[:n_dst]`` for every t = 0..T-1 at once -> [T, n_dst, heads * C].

    Same parameters and maths as PyG ``TransformerConv`` (beta off, root weight on, no edge
    features, attention dropout 0): out_i = sum_j softmax_j(q_i . k_j / sqrt(C)) v_j + W_skip x_i.
    Per (birth bucket, target) we keep the max logit and the shifted sums; a running max over t
    merges the buckets exactly (log-sum-exp), so no logit is ever exponentiated unshifted.
    ``n_dst``: only output rows 0..n_dst-1 are computed (every edge target must be < n_dst); rows
    are independent, so this is exactly the leading block of the full result."""
    H, C = tc.heads, tc.out_channels
    nd = x.shape[0] if n_dst is None else n_dst
    src, dst = ei[0], ei[1]
    xd = x[:nd]
    q = tc.lin_query(xd).view(nd, H, C)
    k = tc.lin_key(x).view(-1, H, C)
    v = tc.lin_value(x).view(-1, H, C)
    a = (q[dst] * k[src]).sum(-1) / math.sqrt(C)                          # [E, H]
    bucket = eb * nd + dst
    neg = -1e30
    with torch.no_grad():
        m = torch.full((T * nd, H), neg, device=x.device, dtype=a.dtype)
        m = m.scatter_reduce(0, bucket[:, None].expand(-1, H), a.detach(), reduce="amax", include_self=True)
    w = torch.exp(a - m[bucket])                                          # <= 1
    Sd = a.new_zeros(T * nd, H).index_add(0, bucket, w).view(T, nd, H)
    Sv = a.new_zeros(T * nd, H, C).index_add(0, bucket, w.unsqueeze(-1) * v[src]).view(T, nd, H, C)
    with torch.no_grad():
        m = m.view(T, nd, H)
        M = torch.cummax(m, dim=0).values                                # running max over t
        # W[t, b] = exp(m_b - M_t) for b <= t (<= 1; 0 above the diagonal and for empty buckets)
        # (mask BEFORE exp: above the diagonal m_b may exceed M_t, and inf * 0 would be NaN)
        low = torch.ones(T, T, device=x.device, dtype=torch.bool).tril().view(T, T, 1, 1)
        W = torch.exp(torch.where(low, m.unsqueeze(0) - M.unsqueeze(1), torch.full_like(m[:1], neg)))  # [T(t), T(b), nd, H]
    N = torch.einsum("tbnh,bnhc->tnhc", W, Sv)
    D = torch.einsum("tbnh,bnh->tnh", W, Sd)
    agg = (N / torch.where(D > 0, D, torch.ones_like(D)).unsqueeze(-1)).reshape(T, nd, H * C)
    return agg + tc.lin_skip(xd).unsqueeze(0)


class Structural(nn.Module):
    """s = PReLU(TC2(PReLU(TC1(x))) + x), TC = TransformerConv(h, h/4, heads=4) (root weight, bias,
    beta off), one shared PReLU. Relations merged, edge types ignored.

    Init: PyG's own default (kaiming_uniform a=sqrt(5) + uniform bias, = torch.nn.Linear's default).
    The official ``init_weights`` loop applies kaiming_normal only to ``torch.nn.Linear`` instances,
    and TransformerConv's ``lin_*`` are ``torch_geometric.nn.dense.linear.Linear`` (a plain
    ``torch.nn.Module`` in PyG 2.1.0 and in 2.8), so the loop never touches them (review round 1)."""

    def __init__(self, hidden: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.tc1 = TransformerConv(hidden, hidden // heads, heads=heads, concat=True, dropout=dropout)
        self.tc2 = TransformerConv(hidden, hidden // heads, heads=heads, concat=True, dropout=dropout)
        self.act = nn.PReLU()

    def forward(self, x, ei, ei_seed, seed_rows):
        """x [S, h] stacked snapshot subgraphs; returns s [T*B, h] at the seed rows."""
        h1 = self.act(self.tc1(x, ei))
        h2 = self.tc2((h1, h1[seed_rows]), ei_seed)
        return self.act(h2 + x[seed_rows])

    def forward_nested(self, x, ei, eb, T: int, e2_src, e2_dst, B: int, n1: int | None = None,
                       pad_dst: int = 0):
        """All T snapshots from ONE sample of the last snapshot (``sampler.NestedBatch``).

        Layer 1 sees static inputs, so its attention logits do not depend on t: snapshot t is the
        softmax over the in-edges born <= t, computed for every t at once by bucketing edges by birth
        and accumulating with a running max (``cumulative_transformer_conv``). Layer 2 runs on the
        (t, edge-into-seed) pairs only. Returns s [T*B, h] (row t * B + i).

        ``n1`` (``NestedBatch.n1``): layer-1 outputs are computed for rows 0..n1-1 only (seeds and
        hop-1 nodes, the only layer-2 inputs); ``e2_src`` is then ``t * n1 + src``. Default: all rows.
        ``pad_dst``: padded batch (``train.pad_nested``): layer 2 gets this many extra zero target
        rows (T * B ...) for the padded pairs, dropped from the output."""
        n = x.shape[0] if n1 is None else n1
        h1 = self.act(cumulative_transformer_conv(self.tc1, x, ei, eb, T, n))    # [T, n, h]
        flat = h1.reshape(T * n, -1)
        x_dst = h1[:, :B].reshape(T * B, -1)
        if pad_dst:
            x_dst = torch.cat([x_dst, x_dst.new_zeros(pad_dst, x_dst.shape[1])])
        h2 = self.tc2((flat, x_dst), torch.stack([e2_src, e2_dst]))
        if pad_dst:
            h2 = h2[:T * B]
        return self.act(h2 + x[:B].repeat(T, 1))

    def forward_full(self, x, ei):
        """Reference path (every node of one snapshot graph): for tests."""
        return self.act(self.tc2(self.act(self.tc1(x, ei)), ei) + x)


class Temporal(nn.Module):
    """X = LN(S + p_AT + p_LCC + p_BLR [+ p_EXTSCORE]); causal multi-head self-attention (no bias,
    kaiming_uniform, -1e9 mask, dropout on the attention weights); O = LN(attn + S) with the SAME
    LayerNorm; logits = Linear(h, 2)(PReLU(Linear(h[+2], h)(O [, extscore]))) at every step."""

    def __init__(self, hidden: int, T: int, heads: int = 4, dropout: float = 0.5, extscore_late: int = 0,
                 extscore_token: int = 0):
        super().__init__()
        self.h, self.heads = hidden, heads
        self.pos = nn.Embedding(T, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.wq = nn.Parameter(torch.empty(hidden, hidden))
        self.wk = nn.Parameter(torch.empty(hidden, hidden))
        self.wv = nn.Parameter(torch.empty(hidden, hidden))
        for w in (self.wq, self.wk, self.wv):
            nn.init.kaiming_uniform_(w)
        self.drop = nn.Dropout(dropout)
        self.ff1 = nn.Linear(hidden + extscore_late, hidden)
        self.ff_act = nn.PReLU()
        self.ff2 = nn.Linear(hidden, 2)
        self.extscore_token = nn.Linear(extscore_token, hidden) if extscore_token else None
        # official GraphTemporalLayer.init_weights: every torch.nn.Linear of the temporal layer (the
        # FFN; here also the ExtScore token) gets kaiming_normal weights and a zero bias (review round 1)
        for lin in (self.ff1, self.ff2, self.extscore_token):
            if lin is not None:
                _kaiming_normal_linear(lin, zero_bias=True)
        self.register_buffer("mask", torch.ones(T, T).tril().bool(), persistent=False)

    def forward(self, S, p_sig, extscore=None, extscore_mode: str = "num"):
        """S [B, T, h]; p_sig [B, T, h] = p_LCC + p_BLR (computed by the caller); extscore [B, 2]."""
        B, T, h = S.shape
        X = S + self.pos.weight[:T].unsqueeze(0) + p_sig
        if self.extscore_token is not None:
            X = X + self.extscore_token(extscore).unsqueeze(1)
        X = self.norm(X)
        d = h // self.heads
        q = (X @ self.wq).view(B, T, self.heads, d).transpose(1, 2)
        k = (X @ self.wk).view(B, T, self.heads, d).transpose(1, 2)
        v = (X @ self.wv).view(B, T, self.heads, d).transpose(1, 2)
        sc = q @ k.transpose(-1, -2) / math.sqrt(d)
        sc = sc.masked_fill(~self.mask[:T, :T], -1e9)
        att = self.drop(torch.softmax(sc, dim=-1))
        o = (att @ v).transpose(1, 2).reshape(B, T, h)
        O = self.norm(o + S)
        if extscore_mode == "late":
            O = torch.cat([O, extscore.unsqueeze(1).expand(B, T, extscore.shape[1])], dim=-1)
        return self.ff2(self.ff_act(self.ff1(O)))


class BotDGT(nn.Module):
    def __init__(self, hidden: int, T: int, num_dim: int, cat_dim: int, *, struct_heads: int = 4,
                 temporal_heads: int = 4, struct_dropout: float = 0.0, temporal_dropout: float = 0.5,
                 extscore_width: int = 0, extscore_inject: str = "num"):
        super().__init__()
        if hidden % 4:
            raise ValueError("hidden must be divisible by 4")
        self.extscore_inject = extscore_inject if extscore_width else "num"
        self.embed = FeatureEmbedding(hidden, num_dim, cat_dim)
        self.struct = Structural(hidden, struct_heads, struct_dropout)
        self.p_lcc = nn.Linear(1, hidden)
        nn.init.kaiming_normal_(self.p_lcc.weight)
        self.p_blr = nn.Linear(1, hidden)
        self.temporal = Temporal(hidden, T, temporal_heads, temporal_dropout,
                                 extscore_late=extscore_width if self.extscore_inject == "late" else 0,
                                 extscore_token=extscore_width if self.extscore_inject == "token" else 0)

    def param_groups(self, lr_struct: float, lr_temporal: float, weight_decay: float,
                     train_pos_signals: bool = False) -> list[dict]:
        """Official optimiser groups: embedding + structural at ``lr_struct``, temporal layer at
        ``lr_temporal``; p_LCC / p_BLR are in NO group (frozen at init) unless ``train_pos_signals``."""
        g = [{"params": list(self.embed.parameters()) + list(self.struct.parameters()), "lr": lr_struct,
              "weight_decay": weight_decay},
             {"params": list(self.temporal.parameters()), "lr": lr_temporal, "weight_decay": weight_decay}]
        if train_pos_signals:
            g.append({"params": list(self.p_lcc.parameters()) + list(self.p_blr.parameters()),
                      "lr": lr_temporal, "weight_decay": weight_decay})
        else:
            for p in list(self.p_lcc.parameters()) + list(self.p_blr.parameters()):
                p.requires_grad_(False)
        return g

    def forward(self, g, batch) -> torch.Tensor:
        """``g``: ``bb.gnn.GraphInputs`` on the device; ``batch``: dict of device tensors from
        ``train.to_device_batch``. Returns logits [B, T, 2]."""
        T = batch["lcc"].shape[1]
        B = batch["lcc"].shape[0]
        u = batch["union"]
        emb = self.embed(g.num[u], g.cat[u], g.des[u].float(), g.tweet[u].float())
        if batch["mode"] == "nested":
            s = self.struct.forward_nested(emb, batch["ei"], batch["eb"], T, batch["e2_src"], batch["e2_dst"], B,
                                           batch.get("n1"), batch.get("pad_dst", 0))
        else:
            x = emb[batch["inv"]]
            s = self.struct(x, batch["ei"], batch["ei_seed"], batch["seed_rows"])
        S = s.view(T, B, -1).transpose(0, 1)
        p_sig = self.p_lcc(batch["lcc"].unsqueeze(-1)) + self.p_blr(batch["blr"].unsqueeze(-1))
        extscore = None
        if self.extscore_inject in ("late", "token"):
            extscore = g.extra[batch["seeds"]]
        return self.temporal(S, p_sig, extscore, self.extscore_inject)


def snapshot_loss(logits: torch.Tensor, y: torch.Tensor, exist: torch.Tensor, coefficient: float = 1.1,
                  last_only: bool = False) -> torch.Tensor:
    """sum_t coefficient^t * CE_mean over seeds that exist at t (snapshots without any are skipped).
    logits [B, T, 2], y [B], exist [B, T] bool."""
    B, T, _ = logits.shape
    ce = nn.functional.cross_entropy(logits.reshape(B * T, -1), y.repeat_interleave(T), reduction="none").view(B, T)
    m = exist.to(ce.dtype)
    cnt = m.sum(0)
    per_t = (ce * m).sum(0) / cnt.clamp(min=1.0)          # 0 where no seed exists (skipped)
    w = coefficient ** torch.arange(T, device=logits.device, dtype=ce.dtype)
    if last_only:
        w = torch.zeros_like(w)
        w[-1] = 1.0
    return (w * per_t).sum()
