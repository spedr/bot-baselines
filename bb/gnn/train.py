"""Training loops for ``GNNClassifier`` models on ``GraphInputs`` (built on ``bb.harness``).

    cfg = FitConfig(epochs=200, lr=1e-2, weight_decay=5e-2, select="last_epoch")
    res = fit_full_batch(model, g, splits, cfg, run=run)            # one step per epoch, all nodes
    res = fit_sampled(model, g, splits, cfg, csc=load_in_merged(ds, F.relations_by_type()), run=run)
    metrics = split_metrics(res.prob, y, splits)                     # compute_metrics per split

Both loops: loss = cross-entropy (optional class weights from TRAIN labels) on train rows only;
labels of val/test rows are read only for the logged validation metrics and, with
``select="best_val"``, for model selection (``BestCheckpoint`` deep copy, optional
``EarlyStopper``). ``select="last_epoch"`` (BotRGCN's protocol) reports the final weights; val is
then purely diagnostic. Logging val metrics never changes training (no RNG use, ``no_grad``).

``fit_sampled`` trains on seed batches with ``bb.harness.sampler.NeighborSampler`` over the
in-neighbour CSC (``etype`` = relation id, so pass relations in edge-type-id order) and predicts
val/test with full fan-out (``-1`` per layer), which is exact for an L-layer message-passing body
when ``len(fanouts) == L``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ..harness import BestCheckpoint, EarlyStopper, class_weights, compute_metrics, to_torch_weights


@dataclass
class FitConfig:
    epochs: int = 200
    lr: float = 1e-2
    weight_decay: float = 5e-2
    optimizer: str = "adamw"          # adamw | adam | sgd
    select: str = "last_epoch"        # last_epoch | best_val
    select_metric: str = "f1"         # key of compute_metrics (bot-class f1, accuracy, macro_f1, ...)
    patience: int = -1                # early stopping on select_metric; -1 = off
    eval_every: int = 1               # val evaluation period (forced to 1 for best_val)
    class_weight: str = "none"        # none | balanced | sqrt | w0,w1
    seed: int = 0                     # sampled loop: epoch shuffles derive from it
    # neighbour-sampled training only
    fanouts: tuple[int, ...] = (-1, -1)
    batch_size: int = 1024
    eval_batch_size: int = 4096
    predict_train: bool = False       # sampled: also predict train rows at the end (costly on TwiBot-22)


@dataclass
class FitResult:
    prob: np.ndarray                  # float32 [N] P(bot) at the selected state (NaN where not predicted)
    epochs_run: int
    selection: dict
    train_seconds: float
    seconds_per_epoch: float
    final_train_loss: float
    history: list = field(default_factory=list)


def make_optimizer(model: nn.Module, cfg: FitConfig):
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(cfg.optimizer)


def _loss_fn(y_train: np.ndarray, cfg: FitConfig, device) -> nn.Module:
    w = class_weights(y_train, cfg.class_weight)
    return nn.CrossEntropyLoss(weight=to_torch_weights(w, device))


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def predict_full(model: nn.Module, g) -> np.ndarray:
    """P(bot) for all N nodes (one eval-mode forward)."""
    was = model.training
    model.eval()
    p = torch.softmax(model(g).float(), dim=1)[:, 1].cpu().numpy().astype(np.float32)
    model.train(was)
    return p


def split_metrics(prob: np.ndarray, y: np.ndarray, splits: dict, names=("train", "val", "test")) -> dict:
    out = {}
    for s in names:
        if s in splits and len(splits[s]):
            idx = np.asarray(splits[s])
            if np.isnan(prob[idx]).any():
                continue
            out[s] = compute_metrics(np.asarray(y)[idx], prob[idx])
    return out


class _Selector:
    def __init__(self, cfg: FitConfig):
        self.cfg = cfg
        self.ck = BestCheckpoint(select=cfg.select, mode="max")
        self.es = EarlyStopper(patience=cfg.patience, mode="max") if cfg.patience >= 0 else None

    def every(self) -> int:
        return 1 if (self.cfg.select == "best_val" or self.es is not None) else max(1, self.cfg.eval_every)

    def step(self, model, mv: dict, ep: int) -> bool:
        v = mv[self.cfg.select_metric]
        self.ck.update(model, v, ep)
        if self.es is not None:
            self.es.step(v)
            return self.es.should_stop
        return False


def fit_full_batch(model: nn.Module, g, splits: dict, cfg: FitConfig, *, run=None, log_every: int = 0) -> FitResult:
    """Full-batch training: one forward over all nodes and one optimiser step per epoch."""
    dev = g.device
    y = g.y.cpu().numpy()
    tr_np, va_np = np.asarray(splits["train"]), np.asarray(splits["val"])
    tr = torch.as_tensor(tr_np, device=dev)
    loss_fn = _loss_fn(y[tr_np], cfg, dev)
    opt = make_optimizer(model, cfg)
    sel = _Selector(cfg)
    every = sel.every()
    hist, t_train, loss_v, ep = [], 0.0, float("nan"), -1
    for ep in range(cfg.epochs):
        model.train()
        _sync(dev)
        t = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        out = model(g)
        loss = loss_fn(out[tr], g.y[tr])
        loss.backward()
        opt.step()
        _sync(dev)
        t_train += time.perf_counter() - t
        loss_v = float(loss.detach())
        rec = {"train_loss": loss_v}
        stop = False
        if (ep + 1) % every == 0 or ep == cfg.epochs - 1:
            pv = predict_full(model, g)
            mv = compute_metrics(y[va_np], pv[va_np])
            rec.update(val_acc=mv["accuracy"], val_f1=mv["f1"], val_auc=mv["roc_auc"])
            stop = sel.step(model, mv, ep)
        hist.append({"epoch": ep, **rec})
        if run is not None:
            run.log_epoch(ep, **rec)
        if log_every and (ep % log_every == 0 or ep == cfg.epochs - 1):
            print(f"  epoch {ep:4d} " + " ".join(f"{k}={v:.4f}" for k, v in rec.items()), flush=True)
        if stop:
            break
    sel.ck.restore(model)
    prob = predict_full(model, g)
    n_ep = ep + 1
    return FitResult(prob, n_ep, sel.ck.info(), round(t_train, 3), round(t_train / max(n_ep, 1), 4), loss_v, hist)


# ---------------------------------------------------------------------------------------------
# neighbour-sampled


def _batch_inputs(g, b, etype: torch.Tensor):
    dev = g.device
    n_id = torch.from_numpy(b.n_id).to(dev)
    ei = torch.from_numpy(b.edge_index).to(dev)
    et = etype[torch.from_numpy(b.e_id).to(dev)]
    return g.nodes(n_id).with_edges(ei, et)


@torch.no_grad()
def predict_sampled(model: nn.Module, g, idx: np.ndarray, sampler, etype: torch.Tensor,
                    batch_size: int = 4096) -> np.ndarray:
    """P(bot) for ``idx`` (in that order) from sampled subgraphs (use fan-out -1 for exact outputs)."""
    was = model.training
    model.eval()
    idx = np.asarray(idx, dtype=np.int64)
    out = np.empty(len(idx), np.float32)
    for s in range(0, len(idx), batch_size):
        b = sampler.sample(idx[s:s + batch_size], rng_seed=s)
        logits = model(_batch_inputs(g, b, etype))[: b.batch_size]
        out[s:s + b.batch_size] = torch.softmax(logits.float(), 1)[:, 1].cpu().numpy()
    model.train(was)
    return out


def fit_sampled(model: nn.Module, g, splits: dict, cfg: FitConfig, *, csc, run=None, log_every: int = 0,
                eval_fanouts: tuple[int, ...] | None = None) -> FitResult:
    """Mini-batch training on sampled subgraphs. ``g`` holds node tensors on the device (edges may be
    empty: ``to_device(..., edges=False)``); ``csc`` is ``load_in_merged(ds, rels_in_type_id_order)``."""
    from ..harness.sampler import NeighborSampler

    dev = g.device
    y = g.y.cpu().numpy()
    tr_np, va_np = np.asarray(splits["train"]), np.asarray(splits["val"])
    tr_np = tr_np[y[tr_np] >= 0]
    loss_fn = _loss_fn(y[tr_np], cfg, dev)
    opt = make_optimizer(model, cfg)
    etype = torch.from_numpy(np.asarray(csc.etype, dtype=np.int64)).to(dev)
    sampler = NeighborSampler(csc, list(cfg.fanouts))
    ev = NeighborSampler(csc, list(eval_fanouts or [-1] * len(cfg.fanouts)))
    sel = _Selector(cfg)
    every = sel.every()
    hist, t_train, loss_v, ep = [], 0.0, float("nan"), -1
    for ep in range(cfg.epochs):
        model.train()
        _sync(dev)
        t = time.perf_counter()
        tot, cnt = 0.0, 0
        for b in sampler.iter_batches(tr_np, cfg.batch_size, shuffle=True, epoch_seed=cfg.seed * 100_003 + ep):
            sub = _batch_inputs(g, b, etype)
            opt.zero_grad(set_to_none=True)
            logits = model(sub)[: b.batch_size]
            loss = loss_fn(logits, sub.y[: b.batch_size])
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * b.batch_size
            cnt += b.batch_size
        _sync(dev)
        t_train += time.perf_counter() - t
        loss_v = tot / max(cnt, 1)
        rec = {"train_loss": loss_v}
        stop = False
        if (ep + 1) % every == 0 or ep == cfg.epochs - 1:
            pv = predict_sampled(model, g, va_np, ev, etype, cfg.eval_batch_size)
            mv = compute_metrics(y[va_np], pv)
            rec.update(val_acc=mv["accuracy"], val_f1=mv["f1"], val_auc=mv["roc_auc"])
            stop = sel.step(model, mv, ep)
        hist.append({"epoch": ep, **rec})
        if run is not None:
            run.log_epoch(ep, **rec)
        if log_every and (ep % log_every == 0 or ep == cfg.epochs - 1):
            print(f"  epoch {ep:4d} " + " ".join(f"{k}={v:.4f}" for k, v in rec.items()), flush=True)
        if stop:
            break
    sel.ck.restore(model)
    prob = np.full(g.n, np.nan, np.float32)
    for s in (("train", "val", "test") if cfg.predict_train else ("val", "test")):
        if s in splits and len(splits[s]):
            ix = np.asarray(splits[s])
            prob[ix] = predict_sampled(model, g, ix, ev, etype, cfg.eval_batch_size)
    n_ep = ep + 1
    return FitResult(prob, n_ep, sel.ck.info(), round(t_train, 3), round(t_train / max(n_ep, 1), 4), loss_v, hist)
