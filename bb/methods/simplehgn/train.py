"""Training loop for Simple-HGN / HGT (docs/methods/simplehgn.md 5).

Every step trains on a seed batch of TRAIN users and its sampled in-neighbourhood:

- TwiBot-20 / Cresci-2015 (benchmark: full-graph forward, loss on 128 shuffled train indices):
  fan-out -1 per hop gives each seed exactly its full-graph output (an L-layer body only sees the
  L-hop in-neighbourhood), so the gradient equals the benchmark's full-batch step without
  computing the other ~229k nodes.
- TwiBot-22: 2-hop sampling ([256, 256] over the union for Simple-HGN, per edge type for HGT).

Loss (default, clean): cross-entropy on the seed rows only. ``loss_nodes="all"`` emulates the
benchmark TwiBot-22 scripts (``label = batch.y``): loss over every labelled node of the sampled
subgraph, which includes val/test users (label leakage); ``val_nodes="subgraph"`` likewise scores
validation accuracy over all subgraph nodes of the val batches (mixed with train/test users), as
the benchmark's selection did. Both are off by default.

AdamW, CosineAnnealingLR(T_max=16, eta_min=0) stepped per epoch (periodic after 16 epochs), fp16
autocast with GradScaler on CUDA (benchmark ``precision=16``), validation every epoch with
fan-out -1, best val accuracy kept as a deep copy (``BestCheckpoint``), optional early stopping.

Input pipeline (review round 2): the next batches are sampled (numba, GIL released) and packed into
one pinned int64 buffer ``[n_id | src | dst | edge type]`` by a background thread; the main thread
issues one non-blocking host-to-device copy per batch. With the fused AdamW kernel, GradScaler's
inf check stays on the GPU, so a training step has no host-device synchronisation and the CPU
queues the next step while the GPU runs the current one. Same batches, same arithmetic up to the
optimiser's rounding (``fused_optimizer=False`` restores the foreach AdamW, which reads the inf
flag back every step).
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ...harness import BestCheckpoint, EarlyStopper, compute_metrics


@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 3e-5
    batch_size: int = 128
    t_max: int = 16
    amp: bool = True
    select: str = "best_val"          # best_val | last_epoch
    select_metric: str = "accuracy"
    patience: int = -1
    seed: int = 0
    eval_batch_size: int = 4096
    loss_nodes: str = "seeds"         # seeds | all (benchmark TwiBot-22 leak emulation)
    val_nodes: str = "seeds"          # seeds | subgraph (benchmark TwiBot-22 mixed-node val acc)
    max_train_batches: int = 0        # >0: stop each epoch after this many steps (measurement only)
    fused_optimizer: bool = True      # CUDA: fused AdamW (no per-step host sync with GradScaler)
    prefetch: int = 3                 # batches sampled ahead in a background thread (0: inline)


@dataclass
class TrainResult:
    prob: np.ndarray
    epochs_run: int
    selection: dict
    train_seconds: float
    seconds_per_epoch: float
    final_train_loss: float
    history: list = field(default_factory=list)
    steps: int = 0


def batch_inputs(g, b, etype: torch.Tensor):
    dev = g.device
    n_id = torch.from_numpy(b.n_id).to(dev, non_blocking=True)
    ei = torch.from_numpy(b.edge_index).to(dev, non_blocking=True)
    et = etype[torch.from_numpy(b.e_id).to(dev, non_blocking=True)]
    return g.nodes(n_id).with_edges(ei, et)


def host_etype(etype: torch.Tensor) -> np.ndarray:
    """Edge types of the CSC entries on the host (int8 when they fit) for ``pack_batch``."""
    small = etype.numel() == 0 or int(etype.max()) < 128
    return (etype.to(torch.int8) if small else etype).cpu().numpy()


def pack_batch(b, etype_host: np.ndarray, pin: bool):
    """(b, one int64 buffer ``[n_id | src | dst | edge type]``), pinned for an async copy."""
    n, E = len(b.n_id), b.edge_index.shape[1]
    buf = torch.empty(n + 3 * E, dtype=torch.int64, pin_memory=pin)
    a = buf.numpy()
    a[:n] = b.n_id
    a[n:n + 2 * E] = b.edge_index.reshape(-1)
    a[n + 2 * E:] = etype_host[b.e_id]
    return b, buf


def device_batch(g, b, buf: torch.Tensor):
    """``batch_inputs`` from a ``pack_batch`` buffer: one non-blocking copy, no host sync."""
    n, E = len(b.n_id), b.edge_index.shape[1]
    d = buf.to(g.device, non_blocking=True)
    return g.nodes(d[:n]).with_edges(d[n:n + 2 * E].view(2, E), d[n + 2 * E:])


_END = object()


class Prefetch:
    """Iterate ``gen`` in a daemon thread, at most ``depth`` items ahead (``depth`` 0: inline).
    Exceptions re-raise in the consumer; leaving the ``with`` block stops and joins the producer,
    so the sampler's scratch state is free again afterwards."""

    def __init__(self, gen, depth: int = 3):
        self.gen, self.depth = gen, depth
        if depth > 0:
            self.q: queue.Queue = queue.Queue(maxsize=depth)
            self.stop = threading.Event()
            self.t = threading.Thread(target=self._run, daemon=True)
            self.t.start()

    def _put(self, item) -> bool:
        while not self.stop.is_set():
            try:
                self.q.put(item, timeout=0.05)
                return True
            except queue.Full:
                pass
        return False

    def _run(self):
        try:
            for item in self.gen:
                if not self._put(item):
                    return
            self._put(_END)
        except BaseException as e:       # noqa: BLE001 (re-raised in the consumer)
            self._put(e)

    def __iter__(self):
        if self.depth <= 0:
            yield from self.gen
            return
        while True:
            item = self.q.get()
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.depth > 0:
            self.stop.set()
            while self.t.is_alive():
                try:
                    self.q.get(timeout=0.05)
                except queue.Empty:
                    pass
            self.t.join()
        return False


def _autocast(dev, enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled and dev.type == "cuda")


@torch.no_grad()
def predict(model: nn.Module, g, idx: np.ndarray, sampler, etype, batch_size: int = 4096, amp: bool = True,
            all_nodes: bool = False, etype_host: np.ndarray | None = None, prefetch: int = 3):
    """P(bot) for ``idx`` (seeds of each batch). ``all_nodes=True`` returns (prob, labels) over every
    labelled node of every subgraph instead (benchmark mixed-node validation). Batches are sampled
    ahead in a thread; outputs stay on the device until one copy at the end."""
    was = model.training
    model.eval()
    idx = np.asarray(idx, dtype=np.int64)
    et_h = host_etype(etype) if etype_host is None else etype_host
    pin = g.device.type == "cuda"
    gen = (pack_batch(sampler.sample(idx[s:s + batch_size], rng_seed=s), et_h, pin)
           for s in range(0, len(idx), batch_size))
    ps, ys = [], []
    with Prefetch(gen, prefetch) as it:
        for b, buf in it:
            sub = device_batch(g, b, buf)
            with _autocast(g.device, amp):
                logits = model(sub)
            p = torch.softmax(logits.float(), 1)[:, 1]
            if all_nodes:
                keep = sub.y >= 0
                ps.append(p[keep])
                ys.append(sub.y[keep])
            else:
                ps.append(p[: b.batch_size])
    model.train(was)
    if all_nodes:
        return torch.cat(ps).cpu().numpy(), torch.cat(ys).cpu().numpy()
    return torch.cat(ps).cpu().numpy() if ps else np.empty(0, np.float32)


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def fit(model: nn.Module, g, splits: dict, cfg: TrainConfig, *, train_sampler, eval_sampler, etype,
        run=None, log_every: int = 0, predict_splits=("val", "test")) -> TrainResult:
    dev = g.device
    y = g.y.cpu().numpy()
    tr = np.asarray(splits["train"])
    tr = tr[y[tr] >= 0]
    va = np.asarray(splits["val"])
    if cfg.loss_nodes not in ("seeds", "all") or cfg.val_nodes not in ("seeds", "subgraph"):
        raise ValueError("loss_nodes in {seeds, all}, val_nodes in {seeds, subgraph}")
    fused = bool(cfg.fused_optimizer and dev.type == "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                            **({"fused": True} if fused else {}))
    et_h = host_etype(etype)
    pin = dev.type == "cuda"
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.t_max, eta_min=0)
    use_amp = cfg.amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss()
    ck = BestCheckpoint(select=cfg.select, mode="max")
    es = EarlyStopper(patience=cfg.patience, mode="max") if cfg.patience >= 0 else None
    hist, t_train, loss_v, ep, steps = [], 0.0, float("nan"), -1, 0
    for ep in range(cfg.epochs):
        model.train()
        _sync(dev)
        t = time.perf_counter()
        tot = torch.zeros((), device=dev)
        cnt = 0
        lr_now = opt.param_groups[0]["lr"]
        batches = train_sampler.iter_batches(tr, cfg.batch_size, shuffle=True, epoch_seed=cfg.seed * 100_003 + ep)
        if cfg.max_train_batches:
            batches = (b for bi, b in zip(range(cfg.max_train_batches), batches))
        with Prefetch((pack_batch(b, et_h, pin) for b in batches), cfg.prefetch) as it:
            for b, buf in it:
                sub = device_batch(g, b, buf)
                opt.zero_grad(set_to_none=True)
                with _autocast(dev, use_amp):
                    logits = model(sub)
                logits = logits.float()
                if cfg.loss_nodes == "seeds":
                    lg, yy = logits[: b.batch_size], sub.y[: b.batch_size]
                    n_y = b.batch_size
                else:
                    keep = sub.y >= 0
                    lg, yy = logits[keep], sub.y[keep]
                    n_y = len(yy)
                loss = loss_fn(lg, yy)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += loss.detach() * n_y
                cnt += n_y
                steps += 1
        sched.step()
        _sync(dev)
        t_train += time.perf_counter() - t
        loss_v = float(tot) / max(cnt, 1)
        pv = predict(model, g, va, eval_sampler, etype, cfg.eval_batch_size, cfg.amp, etype_host=et_h,
                     prefetch=cfg.prefetch)
        n_nan = int(np.isnan(pv).sum())
        mv = compute_metrics(y[va], np.nan_to_num(pv, nan=0.5))
        rec = {"train_loss": loss_v, "lr": lr_now, "val_acc": mv["accuracy"], "val_f1": mv["f1"],
               "val_auc": mv["roc_auc"], "val_nan": n_nan}
        sel_v = mv[cfg.select_metric]
        if cfg.val_nodes == "subgraph":
            ps, ys = predict(model, g, va, train_sampler, etype, cfg.batch_size, cfg.amp, all_nodes=True,
                             etype_host=et_h, prefetch=cfg.prefetch)
            mm = compute_metrics(ys, ps)
            rec["val_acc_subgraph"] = mm["accuracy"]
            sel_v = mm[cfg.select_metric]
        if n_nan or not np.isfinite(loss_v):
            sel_v = -np.inf                  # never select a state with non-finite outputs
        ck.update(model, sel_v, ep)
        stop = False
        if es is not None:
            es.step(sel_v)
            stop = es.should_stop
        hist.append({"epoch": ep, **rec})
        if run is not None:
            run.log_epoch(ep, **rec)
        if log_every and (ep % log_every == 0 or ep == cfg.epochs - 1 or stop):
            print(f"  epoch {ep:4d} " + " ".join(f"{k}={v:.4f}" for k, v in rec.items()), flush=True)
        if stop:
            break
    ck.restore(model)
    prob = np.full(g.n, np.nan, np.float32)
    for s in predict_splits:
        if s in splits and len(splits[s]):
            ix = np.asarray(splits[s])
            prob[ix] = predict(model, g, ix, eval_sampler, etype, cfg.eval_batch_size, cfg.amp, etype_host=et_h,
                               prefetch=cfg.prefetch)
            if np.isnan(prob[ix]).any():
                raise FloatingPointError(f"{int(np.isnan(prob[ix]).sum())} NaN predictions on {s} at the selected state")
    n_ep = ep + 1
    return TrainResult(prob, n_ep, ck.info(), round(t_train, 3), round(t_train / max(n_ep, 1), 4), loss_v,
                       hist, steps)
