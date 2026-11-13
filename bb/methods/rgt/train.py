"""RGT training loops (docs/methods/rgt.md section 5). Plain torch, no Lightning.

``fit_full``: the benchmark TwiBot-20 protocol (``Twibot-20/Heterobot.py``): every optimiser step
runs a full-graph forward (dropout on) and takes the loss on one shuffled mini-batch of TRAIN rows
(``ceil(n_train / batch)`` steps per epoch); val is evaluated once per epoch on the full graph.

``fit_sampled``: TwiBot-22 (``Twibot-22/Heterobot_sample.py``) with ``bb.harness.sampler``
(NeighborLoader semantics: in-edges, seeds first, fan-out over the union of relations). The loss is
on the SEED rows only (clean default). ``loss_nodes="subgraph"`` reproduces the benchmark's leaky
loss over every labelled node of the sampled subgraph, val/test neighbours included (emulation only).
Val/test predictions: seed rows of fixed-seed, unshuffled batches (``eval_batch_size``), so they are
deterministic; semantic attention then averages over each eval subgraph, as in the benchmark.

Both: AdamW, CosineAnnealingLR(T_max) stepped once per epoch (periodic past T_max, PyTorch
semantics), AMP (fp16 autocast + GradScaler, the analogue of Lightning ``precision=16``; ``bf16``
and ``none`` selectable), unweighted CE, model selection on a val metric with a deep-copied
checkpoint (``BestCheckpoint``) or the last epoch, optional patience.
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ...harness import BestCheckpoint, EarlyStopper, compute_metrics


@dataclass
class RGTFit:
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 3e-5
    t_max: int = 16                   # CosineAnnealingLR period (epochs); 0 = constant lr
    batch_size: int = 128
    eval_batch_size: int = 1024       # sampled only
    fanouts: tuple[int, ...] = (256, 256)
    eval_fanouts: tuple[int, ...] | None = None    # None = same as fanouts (spec 5: "same sampler")
    loss_nodes: str = "seeds"         # seeds | subgraph (benchmark TwiBot-22 leak, emulation only)
    select: str = "best_val"          # best_val | last_epoch
    select_metric: str = "accuracy"   # benchmark monitor: val_acc
    patience: int = -1
    amp: str = "fp16"                 # fp16 | bf16 | none
    seed: int = 0
    max_steps: int = 0                # >0: stop after this many optimiser steps (measurement only)
    cuda_graph: bool = False          # sampled: replay each training step as a CUDA graph (padded subgraphs)
    graph_nodes: int = 0              # padding budget (0 = from a sampler dry run, max x 1.25)
    graph_edges: int = 0


@dataclass
class FitOut:
    prob: np.ndarray
    epochs_run: int
    selection: dict
    train_seconds: float
    seconds_per_epoch: float
    eval_seconds: float
    steps: int
    final_train_loss: float
    history: list = field(default_factory=list)


def _sync(dev):
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize(dev)


def _autocast(dev, amp: str):
    if amp == "none" or torch.device(dev).type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.float16 if amp == "fp16" else torch.bfloat16)


def _setup(model, cfg: RGTFit, dev):
    fused = torch.device(dev).type == "cuda"     # single-kernel AdamW (same update rule)
    if cfg.cuda_graph:   # capturable; lr as a device tensor so the scheduler's in-place updates reach the graph
        opt = torch.optim.AdamW(model.parameters(), lr=torch.tensor(cfg.lr, device=dev),
                                weight_decay=cfg.weight_decay, fused=True, capturable=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=fused)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.t_max, eta_min=0) if cfg.t_max > 0 else None
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp == "fp16" and torch.device(dev).type == "cuda"))
    return opt, sched, scaler


def _step(model, opt, scaler, loss_fn, logits, target) -> torch.Tensor:
    """One optimiser step; returns the detached loss on the device (no host sync per step)."""
    loss = loss_fn(logits.float(), target)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    return loss.detach()


class _Select:
    def __init__(self, cfg: RGTFit):
        self.cfg = cfg
        self.ck = BestCheckpoint(select=cfg.select, mode="max")
        self.es = EarlyStopper(patience=cfg.patience, mode="max") if cfg.patience >= 0 else None

    def step(self, model, mv: dict, ep: int) -> bool:
        v = mv[self.cfg.select_metric]
        self.ck.update(model, v, ep)
        if self.es is not None:
            self.es.step(v)
            return self.es.should_stop
        return False


# --------------------------------------------------------------------------------------- full graph
@torch.no_grad()
def predict_full(model, g, amp: str = "fp16") -> np.ndarray:
    was = model.training
    model.eval()
    with _autocast(g.device, amp):
        logits = model(g)
    model.train(was)
    return torch.softmax(logits.float(), 1)[:, 1].cpu().numpy().astype(np.float32)


def fit_full(model: nn.Module, g, splits: dict, cfg: RGTFit, *, run=None, log_every: int = 0) -> FitOut:
    dev = g.device
    y = g.y.cpu().numpy()
    tr_np, va_np = np.asarray(splits["train"]), np.asarray(splits["val"])
    tr_np = tr_np[y[tr_np] >= 0]
    opt, sched, scaler = _setup(model, cfg, dev)
    loss_fn = nn.CrossEntropyLoss()
    sel = _Select(cfg)
    hist, t_train, t_eval, steps, ep, loss_v = [], 0.0, 0.0, 0, -1, float("nan")
    for ep in range(cfg.epochs):
        model.train()
        order = np.random.default_rng(cfg.seed * 100_003 + ep).permutation(len(tr_np))
        _sync(dev)
        t = time.perf_counter()
        tot, nb = 0.0, 0
        rows_ep = torch.as_tensor(tr_np[order], device=dev)      # one H2D copy per epoch
        for s in range(0, len(order), cfg.batch_size):
            rows = rows_ep[s:s + cfg.batch_size]
            with _autocast(dev, cfg.amp):
                logits = model(g, rows=rows)
            tot += _step(model, opt, scaler, loss_fn, logits, g.y[rows]) * len(rows)
            nb += len(rows)
            steps += 1
            if cfg.max_steps and steps >= cfg.max_steps:
                break
        _sync(dev)
        t_train += time.perf_counter() - t
        lr = float(opt.param_groups[0]["lr"])
        if sched is not None:
            sched.step()
        loss_v = float(tot) / max(nb, 1)
        t = time.perf_counter()
        pv = predict_full(model, g, cfg.amp)
        mv = compute_metrics(y[va_np], pv[va_np])
        t_eval += time.perf_counter() - t
        rec = {"train_loss": loss_v, "lr": lr, "val_acc": mv["accuracy"], "val_f1": mv["f1"], "val_auc": mv["roc_auc"]}
        stop = sel.step(model, mv, ep)
        hist.append({"epoch": ep, **rec})
        if run is not None:
            run.log_epoch(ep, **rec)
        if log_every and (ep % log_every == 0 or ep == cfg.epochs - 1):
            print(f"  epoch {ep:4d} " + " ".join(f"{k}={v:.4g}" for k, v in rec.items()), flush=True)
        if stop or (cfg.max_steps and steps >= cfg.max_steps):
            break
    sel.ck.restore(model)
    prob = predict_full(model, g, cfg.amp)
    n = ep + 1
    return FitOut(prob, n, sel.ck.info(), round(t_train, 3), round(t_train / max(n, 1), 4), round(t_eval, 3),
                  steps, loss_v, hist)


# --------------------------------------------------------------------------------------- sampled
class GraphedStep:
    """One seed-loss training step on a padded subgraph, captured once as a CUDA graph and replayed.

    The sampled subgraph (n nodes, E edges, ``bs`` seeds first) is written into static buffers of
    ``n_max + 1`` nodes and ``e_max`` edges: padding nodes gather row 0; padding edges are self-loops
    on padding rows (spread round-robin, so the scatter atomics do not pile onto one row), so no real
    node sees them; the semantic-attention
    mean runs over the ``n`` real nodes only (``node_mask``); projections run densely on all rows.
    The result is the same function as the eager step (tests check forward equality). Requirements:
    fp32 or bf16 (no GradScaler), a capturable optimiser (``_setup``), seed loss only. Batches that do
    not fit (or the short last batch) run eagerly on the same parameters and optimiser."""

    def __init__(self, model, g, opt, loss_fn, etype, bs: int, n_max: int, e_max: int, amp: str):
        dev = g.device
        self.model, self.g, self.opt, self.loss_fn, self.etype, self.amp = model, g, opt, loss_fn, etype, amp
        self.bs, self.n_max, self.e_max = bs, n_max, e_max
        L = torch.long
        self.n_id = torch.zeros(n_max + 1, dtype=L, device=dev)
        self.ei = torch.full((2, e_max), n_max, dtype=L, device=dev)
        self.eid = torch.zeros(e_max, dtype=L, device=dev)
        self.n_real = torch.zeros((), dtype=torch.float32, device=dev)
        self.ar = torch.arange(n_max + 1, device=dev, dtype=torch.float32)
        # two pinned staging sets, so the host fills one while the device may still copy the other
        self.h = [(torch.zeros(n_max + 1, dtype=L).pin_memory(), torch.full((2, e_max), n_max, dtype=L).pin_memory(),
                   torch.zeros(e_max, dtype=L).pin_memory(), torch.zeros((), dtype=torch.float32).pin_memory())
                  for _ in range(2)]
        self.copied = [None, None]     # CUDA event after each set's H2D copies
        self.k = 0
        self.graph = None
        self.loss = None
        self.replays = 0
        self.eager = 0
        self.warm = 0

    def fits(self, b) -> bool:
        return b.batch_size == self.bs and b.num_nodes <= self.n_max and b.num_edges <= self.e_max

    def _load(self, b) -> None:
        """Stage batch ``b`` into the static device buffers without a device sync: wait only until
        the copies that last used this staging set are done (two steps ago)."""
        n, E = b.num_nodes, b.num_edges
        k = self.k
        self.k ^= 1
        if self.copied[k] is not None:
            self.copied[k].synchronize()
        h_nid, h_ei, h_eid, h_n = self.h[k]
        h_nid[:n] = torch.from_numpy(b.n_id)
        h_nid[n:] = 0
        h_ei[:, :E] = torch.from_numpy(b.edge_index)
        # padding edges: self-loops spread round-robin over the padding rows n..n_max (all masked,
        # unreachable from real rows); one shared dummy target would serialise the scatter atomics
        h_ei[:, E:] = torch.from_numpy(n + np.arange(self.e_max - E) % (self.n_max + 1 - n))
        h_eid[:E] = torch.from_numpy(b.e_id)
        h_eid[E:] = 0
        h_n.fill_(float(n))
        self.n_id.copy_(h_nid, non_blocking=True)
        self.ei.copy_(h_ei, non_blocking=True)
        self.eid.copy_(h_eid, non_blocking=True)
        self.n_real.copy_(h_n, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self.copied[k] = ev

    def _body(self):
        sub = self.g.nodes(self.n_id).with_edges(self.ei, self.etype[self.eid])
        mask = (self.ar < self.n_real).float()
        with _autocast(self.g.device, self.amp):
            logits = self.model(sub, rows=slice(0, self.bs), node_mask=mask, dense=True)
        loss = self.loss_fn(logits.float(), sub.y[: self.bs])
        loss.backward()
        self.opt.step()
        return loss.detach()

    def __call__(self, b) -> torch.Tensor:
        """Train on batch ``b`` (must ``fits``); returns the detached loss tensor."""
        self._load(b)
        if self.graph is None:
            if self.warm < 3:      # warm-up: the first 3 fitting batches train eagerly on a side stream
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    self.opt.zero_grad(set_to_none=True)
                    loss = self._body()
                torch.cuda.current_stream().wait_stream(s)
                self.warm += 1
                self.eager += 1
                return loss
            self.opt.zero_grad(set_to_none=True)     # capture records the step without running it
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.loss = self._body()
        self.graph.replay()
        self.replays += 1
        return self.loss.clone()

    def eager_step(self, sub, rows, target) -> torch.Tensor:
        """A step outside the graph (non-fitting / short batch). Gradients are zeroed in place so the
        graph's static grad tensors stay the ones the optimiser reads."""
        self.opt.zero_grad(set_to_none=self.graph is None)
        with _autocast(self.g.device, self.amp):
            logits = self.model(sub, rows=rows, dense=True)
        loss = self.loss_fn(logits.float(), target)
        loss.backward()
        self.opt.step()
        self.eager += 1
        return loss.detach()


def graph_budget(sampler, seeds: np.ndarray, bs: int, n_batches: int = 200) -> tuple[int, int]:
    """Padding budget from a dry run: max subgraph nodes / edges x 1.25, rounded up to 512."""
    rep = sampler.dry_run(seeds, bs, n_batches=n_batches)
    r = lambda v: int(np.ceil(v * 1.25 / 512) * 512)  # noqa: E731
    return r(rep["nodes"]["p100"]), r(rep["edges"]["p100"])


def batch_inputs(g, b, etype: torch.Tensor):
    dev = g.device
    n_id = torch.from_numpy(b.n_id).to(dev, non_blocking=True)
    ei = torch.from_numpy(b.edge_index).to(dev, non_blocking=True)
    et = etype[torch.from_numpy(b.e_id).to(dev, non_blocking=True)]
    return g.nodes(n_id).with_edges(ei, et)


@torch.no_grad()
def predict_sampled(model, g, idx: np.ndarray, sampler, etype: torch.Tensor, batch_size: int = 1024,
                    amp: str = "fp16") -> np.ndarray:
    """P(bot) for ``idx`` (in order) from unshuffled seed batches with a fixed sampler seed per batch."""
    was = model.training
    model.eval()
    idx = np.asarray(idx, dtype=np.int64)
    out = np.empty(len(idx), np.float32)
    for s in range(0, len(idx), batch_size):
        b = sampler.sample(idx[s:s + batch_size], rng_seed=s)
        with _autocast(g.device, amp):
            logits = model(batch_inputs(g, b, etype), rows=slice(0, b.batch_size), dense=True)
        out[s:s + b.batch_size] = torch.softmax(logits.float(), 1)[:, 1].cpu().numpy()
    model.train(was)
    return out


def fit_sampled(model: nn.Module, g, splits: dict, cfg: RGTFit, *, csc, run=None, log_every: int = 0,
                eval_splits=("val", "test")) -> FitOut:
    """``g``: node tensors on the device (``to_device(..., edges=False)``); ``csc`` =
    ``load_in_merged(ds, F.relations_by_type())`` (its ``etype`` is the relation id)."""
    from ...harness.sampler import NeighborSampler

    if cfg.loss_nodes not in ("seeds", "subgraph"):
        raise ValueError(cfg.loss_nodes)
    dev = g.device
    y = g.y.cpu().numpy()
    tr_np, va_np = np.asarray(splits["train"]), np.asarray(splits["val"])
    tr_np = tr_np[y[tr_np] >= 0]
    opt, sched, scaler = _setup(model, cfg, dev)
    loss_fn = nn.CrossEntropyLoss()
    etype = torch.from_numpy(np.asarray(csc.etype, dtype=np.int64)).to(dev)
    sampler = NeighborSampler(csc, list(cfg.fanouts))
    ev = NeighborSampler(csc, list(cfg.eval_fanouts or cfg.fanouts))
    gstep = None
    if cfg.cuda_graph:
        if torch.device(dev).type != "cuda" or cfg.loss_nodes != "seeds" or cfg.amp == "fp16":
            raise ValueError("cuda_graph needs CUDA, loss_nodes='seeds' and amp in {none, bf16}")
        nm, em = cfg.graph_nodes, cfg.graph_edges
        if not (nm and em):
            nm, em = graph_budget(sampler, tr_np, cfg.batch_size)
        gstep = GraphedStep(model, g, opt, loss_fn, etype, cfg.batch_size, nm, em, cfg.amp)
    sel = _Select(cfg)
    hist, t_train, t_eval, steps, ep, loss_v = [], 0.0, 0.0, 0, -1, float("nan")
    for ep in range(cfg.epochs):
        model.train()
        _sync(dev)
        t = time.perf_counter()
        tot, nb = 0.0, 0
        for b in sampler.iter_batches(tr_np, cfg.batch_size, shuffle=True, epoch_seed=cfg.seed * 100_003 + ep):
            if gstep is not None and gstep.fits(b):
                tot += gstep(b) * b.batch_size
                nb += b.batch_size
                steps += 1
                if cfg.max_steps and steps >= cfg.max_steps:
                    break
                continue
            sub = batch_inputs(g, b, etype)
            if cfg.loss_nodes == "seeds":
                rows = slice(0, b.batch_size)
                target = sub.y[: b.batch_size]
            else:   # benchmark leak: every labelled node of the subgraph, val/test neighbours included
                rows = torch.nonzero(sub.y >= 0).squeeze(1)
                target = sub.y[rows]
            if gstep is not None:
                tot += gstep.eager_step(sub, rows, target) * len(target)
            else:
                with _autocast(dev, cfg.amp):
                    logits = model(sub, rows=rows, dense=True)   # dense: no torch.unique sync
                tot += _step(model, opt, scaler, loss_fn, logits, target) * len(target)
            nb += len(target)
            steps += 1
            if cfg.max_steps and steps >= cfg.max_steps:
                break
        _sync(dev)
        t_train += time.perf_counter() - t
        lr = float(opt.param_groups[0]["lr"])
        if sched is not None:
            sched.step()
        loss_v = float(tot) / max(nb, 1)
        # val every epoch (selection with best_val; diagnostic only with last_epoch: no_grad, no RNG)
        t = time.perf_counter()
        pv = predict_sampled(model, g, va_np, ev, etype, cfg.eval_batch_size, cfg.amp)
        t_eval += time.perf_counter() - t
        mv = compute_metrics(y[va_np], pv)
        rec = {"train_loss": loss_v, "lr": lr, "val_acc": mv["accuracy"], "val_f1": mv["f1"], "val_auc": mv["roc_auc"]}
        stop = sel.step(model, mv, ep)
        hist.append({"epoch": ep, **rec})
        if run is not None:
            run.log_epoch(ep, **rec)
        if log_every and (ep % log_every == 0 or ep == cfg.epochs - 1):
            print(f"  epoch {ep:4d} " + " ".join(f"{k}={v:.4g}" for k, v in rec.items()), flush=True)
        if stop or (cfg.max_steps and steps >= cfg.max_steps):
            break
    sel.ck.restore(model)
    prob = np.full(g.n, np.nan, np.float32)
    for s in eval_splits:
        if s in splits and len(splits[s]):
            ix = np.asarray(splits[s])
            prob[ix] = predict_sampled(model, g, ix, ev, etype, cfg.eval_batch_size, cfg.amp)
    n = ep + 1
    out = FitOut(prob, n, sel.ck.info(), round(t_train, 3), round(t_train / max(n, 1), 4), round(t_eval, 3),
                 steps, loss_v, hist)
    if gstep is not None:
        out.selection = {**out.selection, "cuda_graph": {"n_max": gstep.n_max, "e_max": gstep.e_max,
                                                         "replays": gstep.replays, "eager_steps": gstep.eager}}
    return out
