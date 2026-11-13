"""Neighbour-sampled training of the benchmark GCN / GAT (docs/methods/gcn_gat.md section 5).

Protocol (TwiBot-22 ``src/GCN_GAT/train.py``): Adam(lr 1e-4, wd 1e-5), unweighted CE on the seed
nodes of 128-seed batches sampled with ``NeighborLoader(num_neighbors=[256] * 4)`` (incoming edges,
without replacement, directed subgraph), validation after every epoch, "improved" = lexicographic
on (acc, F1, MCC, PR-AUC, ROC-AUC, precision, recall), stop after 50 consecutive epochs without
improvement, at most 1000 epochs.

Implemented locally rather than with ``bb.gnn.fit_sampled`` because the benchmark needs (a) the
lexicographic ``is_better`` selection (``fit_sampled`` selects on one scalar), (b) the selected AND
the final weights evaluated from the same run (the official code's ``state_dict`` reference bug
reports the final weights; we report the deep-copied best state by default and keep the other one
in ``result.json``), (c) sampled validation with fresh randomness per epoch, and (d) a step loop
without a host sync per batch.

Sampling depth (spec 5): the seeds' outputs depend on 2 sampled hops for GAT and 3 for GCN (GCN's
normalisation needs the sampled in-degree of 2-hop nodes); hop 4 never matters. ``default_fanouts``
returns those depths ("exact in distribution"); ``[256] * 4`` stays available (``--bench-hops``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ...harness import compute_metrics

LEX_KEYS = ("accuracy", "f1", "mcc", "pr_auc", "roc_auc", "precision", "recall")
DEPTH = {"gat": 2, "gcn": 3}


def default_fanouts(conv: str, fanout: int = 256, bench_hops: bool = False) -> list[int]:
    return [fanout] * (4 if bench_hops else DEPTH[conv])


@dataclass
class TrainConfig:
    conv: str = "gat"
    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 128
    fanouts: tuple[int, ...] = (256, 256)
    eval_mode: str = "sampled"        # sampled (benchmark: same fan-outs) | full (fan-out -1: exact full neighbourhood)
    eval_batch_size: int = 4096
    max_epochs: int = 1000
    patience: int = 50                # stop when this many consecutive evaluations do not improve
    eval_every: int = 1
    seed: int = 0
    fused_adam: bool | None = None    # None: fused CUDA Adam on the CUDA-graph path (review 2), default Adam otherwise
    analytic_degree: bool = True      # GCN, 3 fan-outs: degrees of 2-hop nodes without sampling hop 3 (identical)
    cuda_graph: bool = True           # CUDA: replay the training step as a captured graph (same math, ~5x faster)


@dataclass
class TrainResult:
    prob_best: np.ndarray             # [N] P(bot) at the selected (best-val, deep-copied) state; NaN outside val/test
    prob_final: np.ndarray            # [N] P(bot) at the final weights (what the official code reports)
    epochs_run: int
    best_epoch: int
    best_val: dict
    train_seconds: float
    seconds_per_epoch: float
    eval_seconds: float
    final_train_loss: float
    history: list = field(default_factory=list)
    step_info: dict = field(default_factory=dict)


def pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Official ``utils.calc_metrics`` PR-AUC: trapezoidal ``auc(recall, precision)`` over
    ``precision_recall_curve`` (NOT average precision). It is the 4th ``is_better`` key and decides
    selection/patience whenever acc, F1 and MCC tie, which is frequent on small val sets (Cresci-2015:
    ~40% of the ties-with-best epochs were decided by it; review round 1)."""
    from sklearn.metrics import auc, precision_recall_curve
    y = np.asarray(y)
    if not 0 < y.sum() < len(y):
        return float("nan")
    precision, recall, _ = precision_recall_curve(y, p)
    return float(auc(recall, precision))


def val_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    m = compute_metrics(y, p)
    m["pr_auc"] = pr_auc(y, p)
    return m


def is_better(now: dict, best: dict | None) -> bool:
    """Official ``utils.is_better``: compare the metrics in LEX_KEYS order; the first that differs
    decides (NaN never wins)."""
    if best is None:
        return True
    for k in LEX_KEYS:
        a, b = now[k], best[k]
        if a != b:
            if np.isnan(a):
                return False
            if np.isnan(b):
                return True
            return a > b
    return False


def prepare_edges(edge_index: np.ndarray, n: int) -> np.ndarray:
    """Drop self loops and add exactly one per node (what PyG's GCNConv ``add_remaining_self_loops``
    and GATConv ``remove_self_loops + add_self_loops`` do on an unweighted graph), on the host."""
    ei = np.asarray(edge_index)
    keep = ei[0] != ei[1]
    loops = np.arange(n, dtype=np.int64)
    return np.concatenate([ei[:, keep], np.stack([loops, loops])], axis=1)


@dataclass
class Plan:
    """Host-side per-layer edge sets of one sampled subgraph (2 message-passing layers).

    Only nodes up to hop 2 are kept (GCN's 3rd sampled hop only contributes in-degrees). Layer 1
    runs over the edges into seeds + hop-1 nodes (what layer 2 reads), layer 2 over the edges into
    the seeds; each set carries one self loop per target. ``w*`` are GCN's symmetric weights
    1/sqrt(deg(u) deg(v)) with deg = sampled in-degree (self loops excluded) + 1 (= ``gcn_norm`` on
    the PyG subgraph); unused (None) for GAT."""
    n_id: np.ndarray
    ei1: np.ndarray
    ei2: np.ndarray
    w1: np.ndarray | None
    w2: np.ndarray | None
    batch_size: int

    @property
    def num_nodes(self) -> int:
        return len(self.n_id)

    @property
    def sizes(self) -> tuple[int, int, int]:
        return self.num_nodes, self.ei1.shape[1], self.ei2.shape[1]


def make_plan(b, gcn: bool, deg_local: np.ndarray | None = None, layers: int = 2) -> Plan:
    """``deg_local`` [>= hop_nodes[2]]: GCN degrees of the kept nodes when they are known without
    sampling their in-edges (``_Batcher`` analytic mode); else counted on the sampled subgraph,
    which then needs a 3rd hop."""
    hn = np.asarray(b.hop_nodes)
    if len(hn) < layers + 1 or (gcn and deg_local is None and len(hn) < layers + 2):
        raise ValueError("subgraph sampled with too few hops for this model")
    src, dst = b.edge_index[0], b.edge_index[1]
    nl = src != dst
    n0, n1, n_keep = int(hn[0]), int(hn[1]), int(hn[2])
    sets = []
    deg = None
    if gcn:
        deg = deg_local if deg_local is not None else np.bincount(dst[nl], minlength=b.num_nodes) + 1.0
    for lim in (n1, n0):
        m = nl & (dst < lim)
        loops = np.arange(lim, dtype=np.int64)
        ei = np.concatenate([np.stack([src[m], dst[m]]), np.stack([loops, loops])], axis=1)
        w = (1.0 / np.sqrt(deg[ei[0]] * deg[ei[1]])).astype(np.float32) if gcn else None
        sets.append((ei, w))
    assert sets[0][0].shape[1] == 0 or sets[0][0][0].max() < n_keep
    return Plan(b.n_id[:n_keep], sets[0][0], sets[1][0], sets[0][1], sets[1][1], b.batch_size)


def in_degree_and_loops(csc) -> tuple[np.ndarray, np.ndarray]:
    ip = np.asarray(csc.indptr, dtype=np.int64)
    deg = np.diff(ip)
    rows = np.repeat(np.arange(len(deg), dtype=np.int64), deg)
    return deg, np.bincount(rows[np.asarray(csc.indices) == rows], minlength=len(deg))


def _to(a, dev):
    t = torch.from_numpy(np.ascontiguousarray(a))
    return t.pin_memory().to(dev, non_blocking=True) if dev.type == "cuda" else t



class _Batcher:
    """Sampled subgraph -> per-layer plan -> device tensors (node rows gathered on the GPU).

    GCN with exactly 3 fan-outs (the default depth) uses the *analytic* degree mode: a node first
    reached at hop h has sampled in-degree min(indeg, k_h) without its self-loop entries, which is
    deterministic whenever no node with indeg > k_h has a self loop (checked; true on all three
    datasets). The 3rd hop is then not sampled at all; hops 1-2 consume the same random stream, so
    plans are identical to 3-hop sampling (tested). Otherwise the full depth is sampled and degrees
    are counted on the subgraph."""

    def __init__(self, g, csc, fanouts, gcn: bool, *, analytic: bool = True, deg_loops=None):
        from ...harness.sampler import NeighborSampler
        self.g, self.gcn = g, gcn
        self.dev = g.device
        fanouts = list(fanouts)
        self.deg_tab = None
        if gcn and analytic and len(fanouts) == 3:
            indeg, loops = deg_loops if deg_loops is not None else in_degree_and_loops(csc)
            ok = all(k < 0 or not np.any((indeg > k) & (loops > 0)) for k in fanouts)
            if ok:
                self.deg_tab, self.k = (indeg, loops), np.asarray(fanouts, np.int64)
                fanouts = fanouts[:2]
        self.mode = "analytic-degree" if self.deg_tab is not None else "sampled"
        self.sampler = NeighborSampler(csc, fanouts)

    def _deg_local(self, b):
        indeg, loops = self.deg_tab
        hn = np.asarray(b.hop_nodes)
        nid = b.n_id[: int(hn[2])]
        hop = np.searchsorted(hn, np.arange(len(nid)), side="right")
        k = self.k[hop]
        d, lp = indeg[nid], loops[nid]
        return np.where((k < 0) | (d <= k), d - lp, k).astype(np.float64) + 1.0

    def plan(self, seeds: np.ndarray, rng_seed: int) -> Plan:
        b = self.sampler.sample(seeds, rng_seed=rng_seed)
        return make_plan(b, self.gcn, self._deg_local(b) if self.deg_tab is not None else None)

    def tensors(self, p: Plan):
        d = self.dev
        sub = self.g.nodes(_to(p.n_id, d))
        w1 = _to(p.w1, d) if p.w1 is not None else None
        w2 = _to(p.w2, d) if p.w2 is not None else None
        return sub, _to(p.ei1, d), w1, _to(p.ei2, d), w2

    def __call__(self, seeds: np.ndarray, rng_seed: int):
        p = self.plan(seeds, rng_seed)
        return self.tensors(p), p


def _forward(model, t):
    from .model import forward_layers
    return forward_layers(model, *t)


@torch.no_grad()
def predict(model: nn.Module, batcher: _Batcher, idx: np.ndarray, batch_size: int, rng_base: int) -> np.ndarray:
    was = model.training
    model.eval()
    idx = np.asarray(idx, dtype=np.int64)
    outs = []
    for k, s in enumerate(range(0, len(idx), batch_size)):
        t, p = batcher(idx[s:s + batch_size], rng_base + k)
        outs.append(torch.softmax(_forward(model, t)[: p.batch_size].float(), 1)[:, 1])
    model.train(was)
    return torch.cat(outs).cpu().numpy().astype(np.float32) if outs else np.zeros(0, np.float32)


def _copy_state(model: nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _masked_ce(logits, y, mask):
    """Mean CE over the rows with mask 1 (= ``CrossEntropyLoss()`` on those rows)."""
    ce = nn.functional.cross_entropy(logits, y.clamp(min=0), reduction="none")
    return (ce * mask).sum() / mask.sum().clamp(min=1.0)


class _Slot:
    """Static inputs of one capture bucket, packed into ONE device buffer that a single H2D copy per
    step refills from a ring of page-locked host buffers.

    Layout (int64 units): n_id [n_cap] | ei1 [2, e1] | ei2 [2, e2] | w1 f32 [e1] | w2 f32 [e2] | mask
    f32 [bs] (float32 regions are dtype views; GCN only for w). Review round 2: the previous version
    issued ~6 pinned-allocations + 6 H2D + 6 D2D copies + fills per step; on the shared GPU each extra
    transfer/dependency costs a time-slice switch (TwiBot-22 GCN: 21 ms per step in the loop against
    5 ms for back-to-back replays). Padding is written on the host: edge slots beyond the plan are
    self loops on the dummy row ``n_cap - 1`` with weight 0, node rows beyond the plan keep stale
    (valid) ids, mask rows beyond the batch are 0 -- the same values the graph saw before.
    A ring buffer is rewritten only after the copy that last read it has completed (CUDA event), so
    the host may run up to ``RING`` steps ahead of the GPU."""

    RING = 4

    def __init__(self, caps, bs, dev, gcn):
        self.caps = caps = tuple(int(c) for c in caps)
        n_cap, e1, e2 = caps
        self.bs, self.gcn = bs, gcn
        self.dummy = n_cap - 1
        half = lambda e: (e + 1) // 2  # noqa: E731  (float32 pairs per int64 slot)
        sizes = [n_cap, 2 * e1, 2 * e2, half(e1) if gcn else 0, half(e2) if gcn else 0, half(bs)]
        o = self.off = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64).tolist()
        self.dbuf = torch.zeros(o[-1], dtype=torch.long, device=dev)
        d = self.dbuf
        self.n_id = d[o[0]:o[1]]
        self.ei = [d[o[1]:o[2]].view(2, e1), d[o[2]:o[3]].view(2, e2)]
        self.w = ([d[o[3]:o[4]].view(torch.float32)[:e1], d[o[4]:o[5]].view(torch.float32)[:e2]]
                  if gcn else [None, None])
        self.mask = d[o[5]:o[6]].view(torch.float32)[:bs]
        on_cuda = torch.device(dev).type == "cuda"
        self.host = [torch.zeros(o[-1], dtype=torch.long, pin_memory=on_cuda) for _ in range(self.RING)]
        self.views = [self._views(h.numpy()) for h in self.host]
        for v in self.views:
            v[1][:] = self.dummy
            v[2][:] = self.dummy
        self.used = [[e1, e2] for _ in range(self.RING)]   # filled extent of each ring buffer's edge regions
        self.events = [torch.cuda.Event() if on_cuda else None for _ in range(self.RING)]
        self.r = 0
        self.dbuf.copy_(self.host[0])
        self.graph = self.loss = None

    def _views(self, h):
        o, (_, e1, e2) = self.off, self.caps
        f = lambda a, b, e: h[a:b].view(np.float32)[:e] if b > a else None  # noqa: E731
        return (h[o[0]:o[1]], h[o[1]:o[2]].reshape(2, e1), h[o[2]:o[3]].reshape(2, e2),
                f(o[3], o[4], e1), f(o[4], o[5], e2), h[o[5]:o[6]].view(np.float32)[: self.bs])

    def fits(self, p: Plan) -> bool:
        n, e1, e2 = p.sizes
        return n < self.caps[0] and e1 <= self.caps[1] and e2 <= self.caps[2] and p.batch_size <= self.bs

    def load(self, p: Plan) -> None:
        i = self.r
        self.r = (i + 1) % self.RING
        if self.events[i] is not None:
            self.events[i].synchronize()       # the copy that last read this host buffer is done
        nid, ea, eb, wa, wb, mask = self.views[i]
        nid[: p.num_nodes] = p.n_id
        for j, (ei, wh, reg, wreg) in enumerate(((p.ei1, p.w1, ea, wa), (p.ei2, p.w2, eb, wb))):
            e, u = ei.shape[1], self.used[i][j]
            reg[:, :e] = ei
            if e < u:
                reg[:, e:u] = self.dummy
            if wreg is not None:
                wreg[:e] = wh
                if e < u:
                    wreg[e:u] = 0.0
            self.used[i][j] = e
        mask[: p.batch_size] = 1.0
        mask[p.batch_size:] = 0.0
        self.dbuf.copy_(self.host[i], non_blocking=True)
        if self.events[i] is not None:
            self.events[i].record()


class CapturedStep:
    """One training step (gather -> forward -> masked CE -> backward -> Adam) captured as CUDA graphs
    over fixed-capacity buffers (a few size buckets), replayed once per batch.

    Why: a 128-seed step is ~180 small kernels; eager PyTorch/PyG spends ~10-14 ms per step in host
    launch overhead against ~1.6 ms of GPU work (measured on TwiBot-22), so the 5,469 steps per
    epoch were host-bound. Replaying a graph removes the per-op host work; the math is the same.

    Padding: node rows beyond the plan are gathered from stale ids and are never connected nor in
    the loss; unused edge slots are self loops on a reserved dummy node (the bucket's last row) with
    GCN weight 0, so they touch no real node. The loss is the mean CE over the real seeds (``mask``),
    equal to ``CrossEntropyLoss()`` on ``out[:batch_size]``. All buckets share the parameters, the
    gradient buffers (zeroed inside each graph) and the Adam state. The warm-up steps needed before
    capture are undone (parameters, gradients and Adam state reset), so training starts from the
    initial weights. A plan larger than every bucket falls back to an eager step on the same buffers.
    """

    def __init__(self, model, opt, g, batch_size: int, buckets, warm: Plan, gcn: bool):
        dev = g.device
        self.model, self.opt, self.g, self.bs = model, opt, g, batch_size
        self.slots = [_Slot(c, batch_size, dev, gcn) for c in sorted(set(map(tuple, buckets)))]
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.loss_sum = torch.zeros((), device=dev)   # sum over steps of batch loss x real seeds (in the graph)
        if not self.slots[-1].fits(warm):
            raise ValueError("warm-up plan exceeds the capture capacity")
        init = [p.detach().clone() for p in self.params]
        for p in self.params:
            p.grad = torch.zeros_like(p)
        warm_slot = self.slots[-1]
        warm_slot.load(warm)
        model.train()
        s = torch.cuda.Stream(dev)
        s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s):
            for _ in range(3):
                opt.zero_grad(set_to_none=False)
                self._body(warm_slot).backward()
                opt.step()
        torch.cuda.current_stream(dev).wait_stream(s)
        for sl in self.slots:
            if sl is not warm_slot:
                sl.load(warm) if sl.fits(warm) else None
            sl.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(sl.graph):
                opt.zero_grad(set_to_none=False)
                sl.loss = self._body(sl)
                sl.loss.backward()
                opt.step()
        torch.cuda.synchronize(dev)
        with torch.no_grad():                    # undo the warm-up/capture updates
            for p, p0 in zip(self.params, init):
                p.copy_(p0)
                p.grad.zero_()
            for st in opt.state.values():
                for v in st.values():
                    if torch.is_tensor(v):
                        v.zero_()
        self.replays = [0] * len(self.slots)
        self.eager_steps = 0
        self.loss_sum.zero_()

    def _body(self, sl: _Slot):
        sub = self.g.nodes(sl.n_id)
        out = _forward(self.model, (sub, sl.ei[0], sl.w[0], sl.ei[1], sl.w[1]))
        loss = _masked_ce(out[: self.bs], sub.y[: self.bs], sl.mask)
        self.loss_sum.add_(loss.detach() * sl.mask.sum())
        return loss

    def fits(self, p: Plan) -> bool:
        return self.slots[-1].fits(p)

    def step(self, p: Plan) -> None:
        """One H2D copy + one replay of the smallest bucket that holds ``p``; the batch loss x seeds
        is added to ``loss_sum`` inside the graph (no per-step kernels outside it)."""
        for j, sl in enumerate(self.slots):
            if sl.fits(p):
                sl.load(p)
                sl.graph.replay()
                self.replays[j] += 1
                return
        raise ValueError("plan exceeds every capture bucket (use eager)")

    def eager(self, t, batch_size: int) -> None:
        """Eager step for an over-capacity plan, on the shared gradient buffers and Adam state."""
        self.model.train()
        self.opt.zero_grad(set_to_none=False)
        loss = nn.functional.cross_entropy(_forward(self.model, t)[:batch_size], t[0].y[:batch_size])
        loss.backward()
        self.opt.step()
        self.eager_steps += 1
        self.loss_sum.add_(loss.detach() * batch_size)

    def info(self) -> dict:
        return {"buckets": [{"nodes": s.caps[0], "edges_layer1": s.caps[1], "edges_layer2": s.caps[2],
                             "replays": r} for s, r in zip(self.slots, self.replays)],
                "eager_steps": self.eager_steps}


def capacity(batcher: _Batcher, seeds: np.ndarray, batch_size: int, n_batches: int = 200,
             quantiles=(0.5, 0.9), margin: float = 1.1, max_margin: float = 1.5):
    """Capture buckets (nodes, layer-1 edges, layer-2 edges) from a dry run on random seed batches:
    per-dimension quantiles x ``margin`` plus the maximum x ``max_margin``; and one plan (the
    smallest) for the warm-up."""
    rng = np.random.default_rng(12345)
    sizes, warm = [], None
    for k in range(n_batches):
        p = batcher.plan(rng.choice(seeds, size=min(batch_size, len(seeds)), replace=False), 987_654 + k)
        sizes.append(p.sizes)
        if warm is None or p.num_nodes < warm.num_nodes:
            warm = p
    sz = np.asarray(sizes, np.float64)
    up = lambda x, q: int(-(-x // q) * q)  # noqa: E731
    mk = lambda v, m: (up(v[0] * m + 1, 256), up(max(v[1] * m, 1), 1024), up(max(v[2] * m, 1), 1024))  # noqa: E731
    buckets = [mk(np.quantile(sz, q, axis=0), margin) for q in quantiles] + [mk(sz.max(0), max_margin)]
    top = buckets[-1]
    return sorted({tuple(min(a, b) for a, b in zip(c, top)) for c in buckets}), warm


def fit(model: nn.Module, g, splits: dict, cfg: TrainConfig, *, csc, run=None, log_every: int = 0,
        max_train_batches: int | None = None) -> TrainResult:
    """Train with the benchmark protocol; returns val/test probabilities at the best and final states.

    ``max_train_batches`` truncates each epoch (measurement only)."""
    dev = g.device
    y = g.y.cpu().numpy()
    tr = np.asarray(splits["train"], dtype=np.int64)
    tr = tr[y[tr] >= 0]
    va = np.asarray(splits["val"], dtype=np.int64)
    te = np.asarray(splits["test"], dtype=np.int64)
    loss_fn = nn.CrossEntropyLoss()
    use_graph = cfg.cuda_graph and dev.type == "cuda"
    fused = (use_graph if cfg.fused_adam is None else cfg.fused_adam) and dev.type == "cuda"
    kw = {"fused": True} if fused else {}
    if use_graph:
        kw["capturable"] = True
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, **kw)
    gcn = cfg.conv == "gcn"
    dl = in_degree_and_loops(csc) if gcn else None
    train_b = _Batcher(g, csc, cfg.fanouts, gcn, deg_loops=dl, analytic=cfg.analytic_degree)
    ev_fan = list(cfg.fanouts) if cfg.eval_mode == "sampled" else [-1] * len(cfg.fanouts)
    eval_b = _Batcher(g, csc, ev_fan, gcn, deg_loops=dl, analytic=cfg.analytic_degree)
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    step = None
    step_info = {"mode": "cuda_graph" if use_graph else "eager", "degrees": train_b.mode,
                 "adam": ("fused" if fused else "foreach") + ("+capturable" if use_graph else "")}
    if use_graph:
        buckets, warm = capacity(train_b, tr, cfg.batch_size)
        step = CapturedStep(model, opt, g, cfg.batch_size, buckets, warm, gcn)

    best, best_state, best_ep, no_up = None, None, -1, 0
    hist, t_train, t_eval, loss_v, ep = [], 0.0, 0.0, float("nan"), -1
    for ep in range(cfg.max_epochs):
        model.train()
        sync()
        t = time.perf_counter()
        tot = step.loss_sum if step is not None else torch.zeros((), device=dev)
        tot.zero_()
        cnt = 0
        order = np.random.default_rng(cfg.seed * 100_003 + ep).permutation(len(tr))
        for k, s in enumerate(range(0, len(tr), cfg.batch_size)):
            if max_train_batches is not None and k >= max_train_batches:
                break
            seeds = tr[order[s:s + cfg.batch_size]]
            rs = (cfg.seed * 100_003 + ep) * 1_000_003 + k
            p = train_b.plan(seeds, rs)
            if step is not None and step.fits(p):
                step.step(p)                    # adds loss x seeds to tot (= step.loss_sum)
            elif step is not None:
                step.eager(train_b.tensors(p), p.batch_size)
            else:
                t_ = train_b.tensors(p)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(_forward(model, t_)[: p.batch_size], t_[0].y[: p.batch_size])
                loss.backward()
                opt.step()
                tot += loss.detach() * p.batch_size
            cnt += p.batch_size
        sync()
        t_train += time.perf_counter() - t
        loss_v = float(tot) / max(cnt, 1)
        rec = {"train_loss": loss_v}
        stop = False
        if (ep + 1) % cfg.eval_every == 0 or ep == cfg.max_epochs - 1:
            t = time.perf_counter()
            pv = predict(model, eval_b, va, cfg.eval_batch_size, rng_base=(cfg.seed * 7_919 + ep) * 1_000_003)
            t_eval += time.perf_counter() - t
            mv = val_metrics(y[va], pv)
            rec.update(val_acc=mv["accuracy"], val_f1=mv["f1"], val_mcc=mv["mcc"], val_auc=mv["roc_auc"])
            if is_better(mv, best):
                best, best_state, best_ep, no_up = mv, _copy_state(model), ep, 0   # deep copy (the fix)
                rec["improved"] = True
            else:
                no_up += 1
                stop = no_up >= cfg.patience
        hist.append({"epoch": ep, **rec})
        if run is not None:
            run.log_epoch(ep, **rec)
        if log_every and (ep % log_every == 0 or stop):
            print(f"  epoch {ep:4d} " + " ".join(f"{k}={v:.4f}" for k, v in rec.items() if not isinstance(v, bool))
                  + (" *" if rec.get("improved") else ""), flush=True)
        if stop:
            break

    def predict_eval() -> np.ndarray:   # same sampling randomness for both states
        prob = np.full(g.n, np.nan, np.float32)
        for j, ix in enumerate((va, te)):
            prob[ix] = predict(model, eval_b, ix, cfg.eval_batch_size, rng_base=(cfg.seed * 7_919 + 10_000 + j) * 1_000_003)
        return prob

    t = time.perf_counter()
    prob_final = predict_eval()
    if best_state is not None:
        model.load_state_dict(best_state)
    prob_best = predict_eval()
    t_eval += time.perf_counter() - t
    n_ep = ep + 1
    if step is not None:
        step_info.update(step.info())
    return TrainResult(prob_best, prob_final, n_ep, best_ep, best or {}, round(t_train, 3),
                       round(t_train / max(n_ep, 1), 3), round(t_eval, 3), loss_v, hist, step_info)
