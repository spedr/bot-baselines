"""BotDGT training and prediction loops (spec section 5).

- AdamW with two parameter groups (embedding + structural, temporal), CosineAnnealingLR(T_max =
  epochs, eta_min 0) stepped per epoch, loss = sum_t 1.1^t CE over existing TRAIN seeds.
- Faithful batching: the train seeds are shuffled ONCE (seeded) and every batch is sampled once
  with a fixed per-batch RNG seed; each epoch replays the identical batches in the identical order.
  The sampler is a pure function of (seeds, fan-outs, rng seed); the epoch-0 batches are packed
  once and replayed from ``BatchCache`` (int32; GPU under ``cache_gb``, then host memory under
  ``cache_host_gb``, else re-sampled), identical to re-sampling them (``cache_batches=False``). ``resample_epochs=True`` is the ablation.
- Speed (review round 2): with ``cuda_graphs`` the cached train batches are padded to bucketed
  shapes and forward + backward of each shape is replayed as one CUDA graph (``GraphedStep``);
  per-step launch overhead under WSL, not FLOPs, is what bounded the eager step.
- Model selection: last epoch (official ``test_last_model``); ``select="best_val"`` emulates the
  official ``--early_stop`` (patience 10, best val accuracy, ties broken by F1).
- Prediction: last-snapshot logits with full 2-hop neighbourhoods (fan-out -1), which is
  batch-size invariant.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ...harness import BestCheckpoint, EarlyStopper, compute_metrics
from .model import snapshot_loss
from .sampler import SnapshotSampler


@dataclass
class DGTConfig:
    epochs: int = 20
    batch_size: int = 64
    lr_struct: float = 1e-4
    lr_temporal: float = 1e-5
    weight_decay: float = 1e-2
    fanouts: tuple = (2560, 2560)
    coefficient: float = 1.1
    train_pos_signals: bool = False
    resample_epochs: bool = False
    select: str = "last_epoch"        # last_epoch | best_val
    patience: int = -1
    eval_every: int = 1               # 0: no per-epoch validation (diagnostic only in last_epoch mode)
    eval_batch_size: int = 1024
    max_union: int = 400_000          # guard: split a seed batch whose working set (rows) is larger
    sampling: str = "nested"          # nested (one sample of the last snapshot, exact uncapped) | per_snapshot
    seed: int = 0
    log_every: int = 0
    max_batches: int = 0              # measurement: stop each epoch after this many batches (0 = all)
    cache_batches: bool = True        # replay packed epoch-0 batches (False: re-sample identically each epoch)
    cache_gb: float = 0.25            # GPU budget of the batch cache (int32; the GPU is shared)
    cache_host_gb: float = 1.5        # then pinned host memory; beyond both, batches are re-sampled
    cuda_graphs: bool = False         # padded static shapes + one CUDA graph per shape for fwd/bwd (nested, CUDA)
    max_graphs: int = 128


@dataclass
class DGTResult:
    prob: np.ndarray
    epochs_run: int
    selection: dict
    train_seconds: float
    seconds_per_epoch: float
    final_train_loss: float
    history: list = field(default_factory=list)
    batch_stats: dict = field(default_factory=dict)


_INT_FIELDS = {"nested": ("seeds", "union", "ei", "eb", "e2_src", "e2_dst", "exist"),
               "stacked": ("seeds", "union", "inv", "ei", "ei_seed", "seed_rows", "exist")}


class PackedBatch:
    """A host batch packed for the device: every integer array in ONE flat tensor (int64, or int32
    when the batch is kept resident on the GPU; widened by one kernel in ``unpack``), the float
    position signals in a second one. ``pack`` runs once per batch; a resident ``PackedBatch`` is
    replayed by every later epoch without re-sampling or host-to-device copies."""
    __slots__ = ("mode", "ints", "sig", "shapes", "n1", "batch_size", "n_edges", "size", "n_capped", "n_nodes",
                 "key")

    def nbytes(self) -> int:
        return self.ints.numel() * self.ints.element_size() + self.sig.numel() * self.sig.element_size()


def bucket(x: int, ratio: float = 1.25, align: int = 64) -> int:
    """Smallest padded size >= x on a geometric grid (align * ratio^k, rounded up to ``align``)."""
    k = max(0, math.ceil(math.log(max(x, 1) / align, ratio) - 1e-9))
    return int(math.ceil(align * ratio ** k / align) * align)


PAD_DST_ROWS = 64       # dummy layer-2 target rows for padded pairs


def pad_nested(b, T: int):
    """Pad a ``NestedBatch`` to bucketed sizes so that batches share a few static shapes (CUDA
    graphs). Exact: padded union rows gather node 0; padded layer-1 edges only target rows in
    [n1, n1p), whose layer-1 outputs no layer-2 pair reads; padded layer-2 pairs only target
    ``PAD_DST_ROWS`` extra dummy output rows (dropped). The real outputs therefore get exactly zero
    from the padding, in the forward and the backward. Padded entries are spread over many sources,
    targets and birth buckets: piling them on one row serialises the atomics and the sort-based
    index backward (measured: 7x the GPU time per TwiBot-22 step). Returns (arrays, n1p, key)."""
    B, n, n1, E, P = len(b.seeds), len(b.n_id), b.n1, b.ei.shape[1], len(b.e2_src)
    n1p = bucket(n1 + 1)
    npad = max(bucket(n), n1p)
    Ep, Pp = bucket(E), bucket(P)
    union = np.zeros(npad, np.int64)
    union[:n] = b.n_id
    i = np.arange(Ep - E)
    ei = np.empty((2, Ep), np.int64)
    ei[:, :E] = b.ei
    ei[0, E:] = i % n
    ei[1, E:] = n1 + i % (n1p - n1)
    eb = np.empty(Ep, np.int64)
    eb[:E] = b.eb
    eb[E:] = (i // (n1p - n1)) % T
    j = np.arange(Pp - P)
    e2s = np.empty(Pp, np.int64)
    e2s[:P] = (b.e2_src // n1) * n1p + b.e2_src % n1
    e2s[P:] = j % (T * n1p)
    e2d = np.empty(Pp, np.int64)
    e2d[:P] = b.e2_dst
    e2d[P:] = T * B + j % PAD_DST_ROWS
    return (union, ei, eb, e2s, e2d), n1p, (B, npad, n1p, Ep, Pp)


def pack_batch(b, snap, window: slice, dev, int_dtype=torch.int64, pad: bool = False) -> PackedBatch:
    """Host batch (``NestedBatch`` or ``SnapshotBatch``) -> ``PackedBatch`` on ``dev`` (one pinned
    copy for the ints, one for the float signals; per-step launch/copy overhead is what dominates
    TwiBot-22 training). ``pad``: bucketed static shapes (``pad_nested``; nested batches only)."""
    from .sampler import NestedBatch
    seeds = b.seeds
    sig = np.stack([snap.lcc[window][:, seeds].T, snap.blr[window][:, seeds].T]).astype(np.float32)  # [2, B, T]
    exist = (snap.birth[seeds].astype(np.int64)[:, None] <= np.arange(snap.T)[window][None, :]).astype(np.int64)
    nested = isinstance(b, NestedBatch)
    key, n1 = None, (getattr(b, "n1", -1) if nested else -1)
    if nested and pad:
        (union, ei, eb, e2s, e2d), n1, key = pad_nested(b, exist.shape[1])
        arrs = (seeds, union, ei, eb, e2s, e2d, exist)
    elif nested:
        arrs = (seeds, b.n_id, b.ei, b.eb, b.e2_src, b.e2_dst, exist)
    else:
        arrs = (seeds, b.union, b.inv, b.ei, b.ei_seed, b.seed_rows, exist)
    flat = np.concatenate([np.asarray(x).reshape(-1) for x in arrs]).astype(np.int64, copy=False)
    if int_dtype == torch.int32:
        if len(flat) and int(flat.max()) >= 2**31:
            raise OverflowError("batch indices exceed int32")
        flat = flat.astype(np.int32)
    ht, hs = torch.from_numpy(flat), torch.from_numpy(sig)
    if torch.device(dev).type == "cuda":
        ht, hs = ht.pin_memory(), hs.pin_memory()
    p = PackedBatch()
    p.mode = "nested" if nested else "stacked"
    p.ints, p.sig = ht.to(dev, non_blocking=True), hs.to(dev, non_blocking=True)
    p.shapes = tuple(x.shape for x in arrs)
    p.n1, p.key = n1, key
    p.batch_size, p.n_edges, p.size = b.batch_size, b.ei.shape[1], _size(b)
    p.n_nodes = len(b.n_id) if nested else len(b.union)
    p.n_capped = getattr(b, "n_capped_rows", 0)
    return p


def unpack(p: PackedBatch) -> dict:
    """``PackedBatch`` -> dict of int64 / bool / float device views (the model's batch format)."""
    ints = p.ints if p.ints.dtype == torch.int64 else p.ints.long()
    out = {"mode": p.mode}
    o = 0
    for name, shp in zip(_INT_FIELDS[p.mode], p.shapes):
        k = int(np.prod(shp))
        out[name] = ints[o:o + k].view(*shp)
        o += k
    out["exist"] = out["exist"].bool()
    out["lcc"], out["blr"] = p.sig[0], p.sig[1]
    if p.n1 >= 0:
        out["n1"] = p.n1
    if p.key is not None:
        out["pad_dst"] = PAD_DST_ROWS
    return out


def to_device_batch(b, snap, window: slice, dev) -> dict:
    """Host batch -> dict of device tensors, plus the per-seed position signals and existence mask
    of the window."""
    return unpack(pack_batch(b, snap, window, dev))


def _size(b) -> int:
    """Rows of the per-batch structural working set (stacked nodes or T * sampled nodes)."""
    from .sampler import NestedBatch
    return b.T * len(b.n_id) if isinstance(b, NestedBatch) else len(b.inv)


def _sample(sampler, seeds, rng_seed, snaps, nested: bool):
    return sampler.nested(seeds, rng_seed, snaps[0]) if nested else sampler.sample_batch(seeds, rng_seed, snaps)


def _split_batches(sampler, seeds, rng_seed, snaps, max_union, nested: bool = True):
    """Sample a seed batch; if its working set exceeds ``max_union`` rows, halve it (micro-batches).
    Yields batches covering ``seeds`` in order."""
    b = _sample(sampler, seeds, rng_seed, snaps, nested)
    if _size(b) <= max_union or len(seeds) == 1:
        yield b
        return
    h = len(seeds) // 2
    yield from _split_batches(sampler, seeds[:h], rng_seed, snaps, max_union, nested)
    yield from _split_batches(sampler, seeds[h:], rng_seed + 7_919, snaps, max_union, nested)


class BatchCache:
    """Packed batches kept for replay (train batches are identical every epoch; eval batches of a
    fixed split too). One entry per seed batch (its micro-batches). Entries go to the GPU (int32)
    while ``gpu_bytes`` allows, then to host memory while ``host_bytes`` allows (pinned and copied
    per use); beyond both budgets an entry is not stored and the caller re-samples it (identical
    batch, the sampler is deterministic). The GPU is shared, so the GPU budget is small by default."""

    def __init__(self, dev, gpu_bytes: int, host_bytes: int, pad: bool = False):
        self.dev, self.pad = torch.device(dev), pad
        self.gpu_budget, self.host_budget = gpu_bytes, host_bytes
        self.gpu_used = self.host_used = self.n_dropped = 0
        self.entries: list = []

    @property
    def complete(self) -> bool:
        return self.n_dropped == 0

    def add(self, hb, snap, window) -> list:
        """Pack the host micro-batches of one seed batch, store them if a budget allows, return the
        device versions."""
        cuda = self.dev.type == "cuda"
        ps = [pack_batch(b, snap, window, "cpu", torch.int32 if cuda else torch.int64, pad=self.pad) for b in hb]
        nb = sum(p.nbytes() for p in ps)
        if cuda and self.gpu_used + nb <= self.gpu_budget:
            for p in ps:
                p.ints = p.ints.pin_memory().to(self.dev, non_blocking=True)
                p.sig = p.sig.pin_memory().to(self.dev, non_blocking=True)
            self.gpu_used += nb
            self.entries.append(ps)
            return ps
        # host entries stay in pageable memory and go through PyTorch's caching pinned allocator at
        # use (as every batch did before caching): keeping thousands of pinned blocks alive forces one
        # cudaHostAlloc per batch, measured at ~0.8 s each under WSL
        if self.host_used + nb <= self.host_budget:
            self.host_used += nb
            self.entries.append(ps)
        else:
            self.entries.append(None)
            self.n_dropped += 1
        return [self._dev(p) for p in ps]

    def get(self, i: int):
        """Device versions of entry ``i``, or None if it was not stored (re-sample it)."""
        e = self.entries[i]
        return None if e is None else [self._dev(p) for p in e]

    def _dev(self, p: PackedBatch) -> PackedBatch:
        if p.ints.device == self.dev:
            return p
        q = PackedBatch()
        for k in PackedBatch.__slots__:
            setattr(q, k, getattr(p, k))
        q.ints = p.ints.pin_memory().to(self.dev, non_blocking=True)
        q.sig = p.sig.pin_memory().to(self.dev, non_blocking=True)
        return q

    def info(self) -> dict:
        return {"entries": len(self.entries), "gpu_bytes": self.gpu_used, "host_bytes": self.host_used,
                "not_stored": self.n_dropped}


_SIDE: dict = {}


def _side_stream() -> torch.cuda.Stream:
    """One side stream per device for the whole process: cuBLAS keeps a workspace (~17 MB here) per
    stream it has run on, so a new stream per ``fit`` (per seed) would leak one workspace each."""
    d = torch.cuda.current_device()
    if d not in _SIDE:
        _SIDE[d] = torch.cuda.Stream()
    return _SIDE[d]


class GraphedStep:
    """Forward + loss + backward of one padded train batch as a CUDA graph per padded shape
    (``PackedBatch.key``), replayed for every later batch of that shape; the optimiser step stays
    eager (the same fused AdamW call as without graphs).

    - Gradients live in persistent ``.grad`` buffers zeroed inside the graph (0 + g == g exactly,
      as with ``set_to_none``); only parameters that received a gradient in an eager step get one.
    - The first batch of a new shape is computed eagerly on a side stream (this is its real
      training step, and the warm-up CUDA graphs require); the capture that follows records
      kernels without running them, and the CUDA RNG state is restored around it so the dropout
      stream continues exactly as eager.
    - At most ``max_graphs`` shapes are captured (one shared memory pool); later new shapes run
      eagerly on the padded batch."""

    def __init__(self, model, g, coefficient: float, max_graphs: int = 128):
        self.model, self.g, self.coef, self.max_graphs = model, g, coefficient, max_graphs
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs: dict = {}
        self.grads: list | None = None
        self.n_eager = 0
        self.side = _side_stream()

    def _fwd_bwd(self, ints, sig, p: PackedBatch):
        q = PackedBatch()
        for k in PackedBatch.__slots__:
            setattr(q, k, getattr(p, k))
        q.ints, q.sig = ints, sig
        bd = unpack(q)
        loss = snapshot_loss(self.model(self.g, bd), self.g.y[bd["seeds"]], bd["exist"], self.coef)
        loss.backward()
        return loss.detach()

    def _zero(self):
        torch._foreach_zero_(self.grads)

    def __call__(self, p: PackedBatch) -> torch.Tensor:
        """Run fwd/bwd of padded device batch ``p``; gradients land in ``param.grad``. Returns the loss."""
        if self.grads is None:                       # very first step: eager, defines the grad buffers
            for prm in self.model.parameters():
                prm.grad = None
            loss = self._fwd_bwd(p.ints, p.sig, p)
            self.grads = [prm.grad for prm in self.model.parameters() if prm.grad is not None]
            self.n_eager += 1
            return loss
        ent = self.graphs.get(p.key)
        if ent is not None:
            ent[1].copy_(p.ints)
            ent[2].copy_(p.sig)
            ent[0].replay()
            return ent[3]
        self._zero()
        if len(self.graphs) >= self.max_graphs:
            self.n_eager += 1
            return self._fwd_bwd(p.ints, p.sig, p)
        ints = torch.empty(p.ints.shape, dtype=torch.int64, device=p.ints.device)
        sig = torch.empty_like(p.sig)
        ints.copy_(p.ints)
        sig.copy_(p.sig)
        main = torch.cuda.current_stream()
        self.side.wait_stream(main)
        with torch.cuda.stream(self.side):           # real step for this batch + warm-up
            loss = self._fwd_bwd(ints, sig, p)
        main.wait_stream(self.side)
        loss.record_stream(main)
        rng = torch.cuda.get_rng_state()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            self._zero()
            static_loss = self._fwd_bwd(ints, sig, p)
        torch.cuda.set_rng_state(rng)
        self.graphs[p.key] = (graph, ints, sig, static_loss)
        self.n_eager += 1
        return loss


@torch.no_grad()
def predict(model, g, snap, idx: np.ndarray, window: slice, batch_size: int = 1024,
            max_union: int = 400_000, nested: bool = True, cache: BatchCache | None = None) -> np.ndarray:
    """P(bot) at the last snapshot. ``cache``: replay the packed batches of an earlier call on the
    same ``idx`` (an empty cache is filled; an incomplete one is ignored)."""
    was = model.training
    model.eval()
    dev = g.device
    snaps = list(range(snap.T))[window]
    idx = np.asarray(idx, dtype=np.int64)
    if cache is not None and cache.entries and cache.complete:
        packed = (p for i in range(len(cache.entries)) for p in cache.get(i))
    else:
        fill = cache is not None and not cache.entries
        sampler = SnapshotSampler(snap, [-1, -1])
        hbs = ([b for b in _split_batches(sampler, idx[s:s + batch_size], 0, snaps, max_union, nested)]
               for s in range(0, len(idx), batch_size))
        packed = (p for hb in hbs
                  for p in (cache.add(hb, snap, window) if fill else [pack_batch(b, snap, window, dev) for b in hb]))
    outs = [torch.softmax(model(g, unpack(p))[:, -1].float(), 1)[:, 1] for p in packed]
    out = torch.cat(outs).cpu().numpy() if outs else np.empty(0, np.float32)   # one device->host sync
    assert len(out) == len(idx)
    model.train(was)
    return out


def _sync(dev):
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize(dev)


def fit(model, g, snap, splits: dict, cfg: DGTConfig, *, window: slice = slice(None), run=None,
        predict_splits=("val", "test")) -> DGTResult:
    dev = g.device
    y = g.y.cpu().numpy()
    tr = np.asarray(splits["train"], dtype=np.int64)
    tr = tr[y[tr] >= 0]
    va = np.asarray(splits["val"], dtype=np.int64)
    snaps = list(range(snap.T))[window]
    # fused AdamW on CUDA: same update rule, ~300 fewer kernel launches per step (measured 42 -> 27
    # ms/step on TwiBot-22, where launch overhead dominates)
    opt = torch.optim.AdamW(model.param_groups(cfg.lr_struct, cfg.lr_temporal, cfg.weight_decay,
                                               cfg.train_pos_signals), fused=torch.device(dev).type == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=0.0)
    sampler = SnapshotSampler(snap, list(cfg.fanouts))
    nested = cfg.sampling == "nested"
    ck = BestCheckpoint(select=cfg.select, mode="max")
    # official: stop once `patience` consecutive epochs did not improve (count >= patience); the
    # harness EarlyStopper stops when bad > patience, hence patience - 1
    es = (EarlyStopper(patience=max(cfg.patience - 1, 0), mode="max")
          if (cfg.select == "best_val" and cfg.patience >= 1) else None)
    every = 1 if cfg.select == "best_val" else cfg.eval_every
    hist, t_train, loss_v, ep = [], 0.0, float("nan"), -1
    stats = {"nodes": [], "rows": [], "edges": [], "capped_rows": [], "micro_batches": 0}
    y_dev = g.y
    gpu_b, host_b = int(cfg.cache_gb * 1e9), int(cfg.cache_host_gb * 1e9)
    # faithful batching replays the epoch-0 batches every epoch: keep them packed (on the GPU while
    # under budget) instead of re-sampling and re-copying them; the val split is fixed too
    graphs = cfg.cuda_graphs and nested and torch.device(dev).type == "cuda"
    gstep = GraphedStep(model, g, cfg.coefficient, cfg.max_graphs) if graphs else None
    cache = (BatchCache(dev, gpu_b, host_b, pad=graphs) if (cfg.cache_batches and not cfg.resample_epochs)
             else None)                  # entry bi = train batch bi (its micro-batches)
    val_cache = BatchCache(dev, gpu_b // 4, host_b // 4) if (every and cfg.cache_batches) else None
    for ep in range(cfg.epochs):
        model.train()
        _sync(dev)
        t0 = time.perf_counter()
        es_seed = cfg.seed * 1_000_003 + (ep + 1 if cfg.resample_epochs else 0)
        order = np.random.default_rng(es_seed).permutation(len(tr))
        tot, cnt = torch.zeros((), device=dev), 0
        for bi, s in enumerate(range(0, len(tr), cfg.batch_size)):
            if cfg.max_batches and bi >= cfg.max_batches:
                break
            seeds = tr[order[s:s + cfg.batch_size]]
            parts = cache.get(bi) if (cache is not None and ep > 0) else None
            if parts is None:
                hb = list(_split_batches(sampler, seeds, es_seed * 65_537 + bi, snaps, cfg.max_union, nested))
                if cache is not None and ep == 0:
                    parts = cache.add(hb, snap, window)
                else:
                    parts = [pack_batch(b, snap, window, dev, pad=graphs) for b in hb]
            if gstep is not None and len(parts) == 1:
                loss = gstep(parts[0])
                tot += loss * parts[0].batch_size
            else:
                if gstep is not None and gstep.grads is not None:
                    gstep._zero()                       # persistent grad buffers (CUDA graphs)
                else:
                    opt.zero_grad(set_to_none=True)
                for p in parts:
                    bd = unpack(p)
                    logits = model(g, bd)
                    loss = snapshot_loss(logits, y_dev[bd["seeds"]], bd["exist"], cfg.coefficient)
                    (loss * (p.batch_size / len(seeds))).backward()
                    tot += loss.detach() * p.batch_size          # no per-step host sync
            for p in parts:
                if ep == 0:
                    stats["nodes"].append(p.n_nodes)
                    stats["rows"].append(p.size)
                    stats["edges"].append(p.n_edges)
                    stats["capped_rows"].append(p.n_capped)
            stats["micro_batches"] += (len(parts) - 1) if ep == 0 else 0
            opt.step()
            cnt += len(seeds)
            if cfg.log_every and bi % cfg.log_every == 0:
                print(f"    ep {ep} batch {bi} loss {float(loss):.4f} {time.perf_counter() - t0:.1f}s", flush=True)
        sched.step()
        _sync(dev)
        t_train += time.perf_counter() - t0
        loss_v = float(tot) / max(cnt, 1)
        rec = {"train_loss": loss_v, "lr_struct": opt.param_groups[0]["lr"], "train_seconds": time.perf_counter() - t0}
        stop = False
        if every and ((ep + 1) % every == 0 or ep == cfg.epochs - 1):
            pv = predict(model, g, snap, va, window, cfg.eval_batch_size, cfg.max_union, nested, val_cache)
            mv = compute_metrics(y[va], pv)
            rec.update(val_acc=mv["accuracy"], val_f1=mv["f1"], val_auc=mv["roc_auc"])
            # official early stop (``is_better``): best accuracy, ties by F1, and a full tie counts as
            # an improvement (the later epoch wins); the 1e-12 * ep term reproduces that last rule
            score = mv["accuracy"] + 1e-6 * mv["f1"] + 1e-12 * ep
            ck.update(model, score, ep)
            if es is not None:
                es.step(score)
                stop = es.should_stop
        hist.append({"epoch": ep, **rec})
        if run is not None:
            run.log_epoch(ep, **rec)
        if cfg.log_every or ep == cfg.epochs - 1:
            print(f"  epoch {ep:3d} " + " ".join(f"{k}={v:.4f}" for k, v in rec.items())
                  + f" ({time.perf_counter() - t0:.1f}s)", flush=True)
        if stop:
            break
    graph_info = None
    if gstep is not None:                    # free the graphs' memory pool before prediction
        graph_info = {"graphs": len(gstep.graphs), "eager_steps": gstep.n_eager}
        gstep.graphs.clear()
        gstep = None
    ck.restore(model)
    prob = np.full(g.n, np.nan, np.float32)
    for sp in predict_splits:
        if sp in splits and len(splits[sp]):
            ix = np.asarray(splits[sp], dtype=np.int64)
            prob[ix] = predict(model, g, snap, ix, window, cfg.eval_batch_size, cfg.max_union, nested,
                               val_cache if (sp == "val" and val_cache is not None and val_cache.entries) else None)
    n_ep = ep + 1
    bs = {k: ({"p50": float(np.percentile(v, 50)), "p99": float(np.percentile(v, 99)), "max": int(np.max(v))}
              if isinstance(v, list) and v else v) for k, v in stats.items()}
    if graph_info is not None:
        bs["cuda_graphs"] = graph_info
    if cache is not None:
        bs["cache"] = {**cache.info(), "val": val_cache.info() if val_cache else None}
    return DGTResult(prob, n_ep, ck.info(), round(t_train, 3), round(t_train / max(n_ep, 1), 3), loss_v, hist, bs)
