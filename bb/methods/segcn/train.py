"""SEGCN training loops (spec section 5).

- ``fit_subgraph``: mini-batches of TRAIN roots (shuffled per epoch from the seed), packed so that a
  batch has at most ``batch_size`` roots and at most ``max_rows`` lifted rows (upper bound over all
  levels, ``GPUStore.cost``). Loss = unweighted cross-entropy on the batch roots (train labels only).
  Validation every epoch with the exact layer-wise eval-mode forward; model selection = best val
  accuracy (deep copy; P-code train.py:180-184), or the last epoch.
- ``fit_global``: the ``n_inner = 0`` degenerate path (plain GCN over the global graph), full batch,
  one step per epoch.

Labels of val/test rows are only read for the logged validation metrics and (``best_val``) for model
selection. Test labels are never read during training.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numba as nb
import numpy as np
import torch
from torch import nn

from ...harness import BestCheckpoint, compute_metrics


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-2
    weight_decay: float = 5e-3
    optimizer: str = "adamw"
    step_lr: bool = False             # code: StepLR(50, 0.5)
    select: str = "best_val"          # best_val (val accuracy) | last_epoch
    select_metric: str = "accuracy"
    batch_size: int = 512
    max_rows: int = 600_000
    eval_rows: int = 400_000          # lifted rows per eval chunk
    accumulate: bool = False          # one optimiser step per epoch (full-batch emulation)
    seed: int = 0
    max_batches: int = 0              # measurement: stop each epoch after this many batches (0 = all)


@dataclass
class FitResult:
    prob: np.ndarray
    epochs_run: int
    selection: dict
    train_seconds: float
    seconds_per_epoch: float
    eval_seconds: float
    final_train_loss: float
    history: list = field(default_factory=list)
    batches_per_epoch: float = 0.0
    rows_per_batch: float = 0.0


@nb.njit(cache=True)
def _pack(cost, batch_size, max_rows):
    """Greedy boundaries over the (shuffled) cost array: new batch at batch_size roots or when the
    summed cost would exceed max_rows (a batch keeps at least 2 roots). Returns batch end positions."""
    n = cost.shape[0]
    ends = np.empty(n, np.int64)
    nb_ = 0
    cnt = 0
    rows = 0
    for i in range(n):
        c = cost[i]
        if cnt >= batch_size or (cnt >= 2 and rows + c > max_rows):     # >= 2 roots (train-mode BN)
            ends[nb_] = i
            nb_ += 1
            cnt = 0
            rows = 0
        cnt += 1
        rows += c
    if cnt > 0:
        ends[nb_] = n
        nb_ += 1
    return ends[:nb_]


def pack_batches(perm: np.ndarray, cost: np.ndarray, batch_size: int, max_rows: int) -> list[np.ndarray]:
    ends = _pack(np.ascontiguousarray(cost[perm], dtype=np.int64), int(batch_size), int(max_rows))
    out, s = [], 0
    for e in ends:
        out.append(perm[s:e])
        s = int(e)
    if len(out) > 1 and len(out[-1]) < 2:          # train-mode BN needs >= 2 rows
        out[-2] = np.concatenate([out[-2], out[-1]])
        out.pop()
    return out


def make_optimizer(model: nn.Module, cfg: TrainConfig):
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(cfg.optimizer)


def _sync(dev) -> None:
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize(dev)


def _probs(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits.float(), 1)[:, 1].cpu().numpy().astype(np.float32)


@torch.no_grad()
def predict_subgraph(model, g, store, idx: np.ndarray, cfg: TrainConfig, needed=None) -> np.ndarray:
    was = model.training
    model.eval()
    r = torch.as_tensor(np.asarray(idx, np.int64), device=g.device)
    p = _probs(model.predict_layerwise(g, store, r, chunk_rows=cfg.eval_rows, needed=needed))
    model.train(was)
    return p


def fit_subgraph(model, g, store, splits: dict, cfg: TrainConfig, *, run=None, log_every: int = 0,
                 resample=None, predict_splits=("val", "test")) -> FitResult:
    """``store``: GPUStore used for training and evaluation. ``resample(ep) -> GPUStore`` (optional)
    gives a fresh training store per epoch (GNN-AK online sampling ablation); eval keeps ``store``."""
    from .model import Prefetcher, eval_sets

    dev = g.device
    L = model.n_layers
    y_np = g.y.cpu().numpy()
    tr = np.asarray(splits["train"], np.int64)
    tr = tr[y_np[tr] >= 0]
    va = np.asarray(splits["val"], np.int64)
    loss_fn = nn.CrossEntropyLoss()
    opt = make_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5) if cfg.step_lr else None
    ck = BestCheckpoint(select=cfg.select, mode="max")
    need_va = eval_sets(store, va, L)
    train_store = store
    cost = store.cost(L).cpu().numpy()
    rng = np.random.default_rng(cfg.seed * 100_003 + 17)
    hist, t_train, t_eval, loss_v, ep = [], 0.0, 0.0, float("nan"), -1
    n_batches_tot, rows_tot = 0, 0
    for ep in range(cfg.epochs):
        if resample is not None and ep > 0:
            train_store = resample(ep)
            cost = train_store.cost(L).cpu().numpy()
        model.train()
        _sync(dev)
        t = time.perf_counter()
        perm = rng.permutation(tr)
        batches = pack_batches(perm, cost, cfg.batch_size, cfg.max_rows)
        if cfg.max_batches:
            batches = batches[:cfg.max_batches]
        tot = torch.zeros((), device=dev)
        cnt = 0
        if cfg.accumulate:
            opt.zero_grad(set_to_none=True)
        for b, hp, s0, levels in Prefetcher(train_store, batches, L):     # host plans built ahead
            roots = levels[-1].roots
            rows_tot += hp.n_rows
            logits = model.forward_plan(g, s0, levels)
            loss = loss_fn(logits, g.y[roots])
            if cfg.accumulate:
                (loss * (len(b) / len(tr))).backward()
            else:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            tot += loss.detach() * len(b)
            cnt += len(b)
        if cfg.accumulate:
            opt.step()
        if sched is not None:
            sched.step()
        _sync(dev)
        t_train += time.perf_counter() - t
        n_batches_tot += len(batches)
        loss_v = float(tot) / max(cnt, 1)
        t = time.perf_counter()
        pv = predict_subgraph(model, g, store, va, cfg, needed=need_va)
        t_eval += time.perf_counter() - t
        mv = compute_metrics(y_np[va], pv)
        rec = {"train_loss": loss_v, "val_acc": mv["accuracy"], "val_f1": mv["f1"], "val_auc": mv["roc_auc"],
               "n_batches": len(batches)}
        ck.update(model, mv[cfg.select_metric], ep)
        hist.append({"epoch": ep, **rec})
        if run is not None:
            run.log_epoch(ep, **rec)
        if log_every and (ep % log_every == 0 or ep == cfg.epochs - 1):
            print(f"  epoch {ep:4d} " + " ".join(f"{k}={v:.4f}" for k, v in rec.items()), flush=True)
    ck.restore(model)
    prob = np.full(g.n, np.nan, np.float32)
    for s in predict_splits:
        if s in splits and len(splits[s]):
            ix = np.asarray(splits[s], np.int64)
            prob[ix] = predict_subgraph(model, g, store, ix, cfg)
    n_ep = ep + 1
    return FitResult(prob, n_ep, ck.info(), round(t_train, 3), round(t_train / max(n_ep, 1), 4),
                     round(t_eval, 3), loss_v, hist, n_batches_tot / max(n_ep, 1),
                     rows_tot / max(n_batches_tot, 1))


def global_adjacency(csr, device) -> torch.Tensor:
    """Normalised D^-1/2 (A+I) D^-1/2 of the simple undirected graph (sparse CSR on ``device``)."""
    from .model import normalized_adjacency
    indptr = np.asarray(csr.indptr, np.int64)
    rows = np.repeat(np.arange(len(indptr) - 1, dtype=np.int64), np.diff(indptr))
    cols = np.asarray(csr.indices, np.int64)
    keep = rows < cols                                   # each undirected edge once
    a = torch.from_numpy(rows[keep]).to(device)
    b = torch.from_numpy(cols[keep]).to(device)
    return normalized_adjacency(len(indptr) - 1, a, b)


@torch.no_grad()
def predict_global(model, g, adj) -> np.ndarray:
    was = model.training
    model.eval()
    p = _probs(model.forward_global(g, adj))
    model.train(was)
    return p


def fit_global(model, g, adj, splits: dict, cfg: TrainConfig, *, run=None, log_every: int = 0) -> FitResult:
    dev = g.device
    y_np = g.y.cpu().numpy()
    tr = torch.as_tensor(np.asarray(splits["train"], np.int64), device=dev)
    va = np.asarray(splits["val"], np.int64)
    loss_fn = nn.CrossEntropyLoss()
    opt = make_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5) if cfg.step_lr else None
    ck = BestCheckpoint(select=cfg.select, mode="max")
    hist, t_train, t_eval, loss_v, ep = [], 0.0, 0.0, float("nan"), -1
    for ep in range(cfg.epochs):
        model.train()
        _sync(dev)
        t = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        out = model.forward_global(g, adj)
        loss = loss_fn(out[tr], g.y[tr])
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        _sync(dev)
        t_train += time.perf_counter() - t
        loss_v = float(loss.detach())
        t = time.perf_counter()
        pv = predict_global(model, g, adj)
        t_eval += time.perf_counter() - t
        mv = compute_metrics(y_np[va], pv[va])
        rec = {"train_loss": loss_v, "val_acc": mv["accuracy"], "val_f1": mv["f1"], "val_auc": mv["roc_auc"]}
        ck.update(model, mv[cfg.select_metric], ep)
        hist.append({"epoch": ep, **rec})
        if run is not None:
            run.log_epoch(ep, **rec)
        if log_every and (ep % log_every == 0 or ep == cfg.epochs - 1):
            print(f"  epoch {ep:4d} " + " ".join(f"{k}={v:.4f}" for k, v in rec.items()), flush=True)
    ck.restore(model)
    prob = predict_global(model, g, adj)
    n_ep = ep + 1
    return FitResult(prob, n_ep, ck.info(), round(t_train, 3), round(t_train / max(n_ep, 1), 4),
                     round(t_eval, 3), loss_v, hist, 1.0, float(g.n))


def split_metrics_safe(prob: np.ndarray, y: np.ndarray, splits: dict, names=("train", "val", "test")) -> dict:
    """compute_metrics per split, skipping splits that were not predicted (NaN)."""
    out = {}
    for s in names:
        if s in splits and len(splits[s]):
            idx = np.asarray(splits[s])
            if np.isnan(prob[idx]).any():
                continue
            out[s] = compute_metrics(np.asarray(y)[idx], prob[idx])
    return out
