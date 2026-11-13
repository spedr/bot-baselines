"""BotMoE training loop (docs/methods/botmoe.md 5): neighbour-sampled mini-batches as the released
``main.py`` (NeighborLoader, 1024 seeds, fan-out [256, 256], in-neighbours), Adam with coupled
weight decay, loss = CE(seeds) + sum of the five MoE balance losses, ``clip_grad_value_(1000)``,
MultiStepLR([200, 600], 0.8). The graph-MoE gates start frozen; they are unfrozen (and the lr set to
``lr * unfreeze_lr_factor``, 1e-5 -> 1e-6 in the code) at the end of the first epoch where the best
validation accuracy so far exceeds ``unfreeze_acc`` or ``unfreeze_epoch`` epochs have run. Model
selection: best validation accuracy (deep copy), early stopping after ``patience`` epochs without
improvement. Validation/test use the same sampler fan-outs (the released val loader samples with
[256, 256] too) with fixed sampler seeds.

During training every per-node op (the five tokens, BatchNorm statistics, balance losses) runs over
all nodes of the sampled subgraph, as in the released code, where ``[:11826]`` is a no-op for
subgraphs that small; only the seeds enter the CE loss. At eval the fusion runs on the seeds only
(exact: BN in eval mode and MoE gating are row-independent).

Efficiency (review round 2): each batch's edge structures are built on the host (``plan.py``, one
pinned non-blocking copy, no host-device syncs in the step, so the CPU queues batch i+1 while the GPU
runs batch i); Adam is the fused CUDA kernel.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ...harness import BestCheckpoint, EarlyStopper, compute_metrics


@dataclass
class TrainConfig:
    epochs: int = 400
    lr: float = 1e-5
    weight_decay: float = 1e-6
    milestones: tuple[int, ...] = (200, 600)
    gamma: float = 0.8
    clip_value: float = 1000.0
    batch_size: int = 1024
    eval_batch_size: int = 4096
    fanouts: tuple[int, ...] = (256, 256)
    eval_fanouts: tuple[int, ...] | None = None
    patience: int = 200
    unfreeze_acc: float | None = 0.865
    unfreeze_epoch: int = 200
    unfreeze_lr_factor: float = 0.1
    freeze_graph_gates: bool = True
    seed: int = 0
    amp: str = "none"            # none | bf16 (autocast for the forward pass)
    max_batches: int = 0         # >0: stop each epoch after this many batches (measurement only)
    fused_adam: bool = True      # CUDA: torch's fused Adam kernel (one launch) instead of foreach


@dataclass
class FitOut:
    prob: np.ndarray
    epochs_run: int
    selection: dict
    train_seconds: float
    seconds_per_epoch: float
    eval_seconds_per_epoch: float
    unfreeze_epoch: int | None
    routes: dict = field(default_factory=dict)
    history: list = field(default_factory=list)


class Inputs:
    """Device tensors: ``g`` (bb.gnn.GraphInputs, edges empty), ``meta`` [N, m], ``extscore`` [N, 2] or None;
    ``etype``: relation id per CSC entry. ``plan=True`` (default) builds each batch's edge structures
    on the host (``plan.SubgraphPlan``: one pinned async copy, no host syncs in the step); False
    derives them on the device as the layers do by themselves (reference path, same function)."""

    def __init__(self, g, meta: torch.Tensor, extscore: torch.Tensor | None, etype, *, num_relations: int | None = None,
                 plan: bool = True, max_edges: int = 32768):
        self.g, self.meta, self.extscore = g, meta, extscore
        self.etype_np = np.ascontiguousarray(etype.cpu().numpy() if torch.is_tensor(etype) else etype, dtype=np.int64)
        self.etype = torch.from_numpy(self.etype_np).to(g.device)
        self.R = num_relations or (int(self.etype_np.max()) + 1 if len(self.etype_np) else 1)
        self.plan, self.max_edges = plan, max_edges

    def batch(self, b):
        """(subgraph GraphInputs, meta rows, extscore rows, plan or None) of a ``SampledSubgraph``."""
        dev = self.g.device
        if self.plan:
            from .plan import SubgraphPlan
            P = SubgraphPlan.build(b.edge_index, self.etype_np[b.e_id], b.num_nodes, self.R, self.max_edges, dev,
                                   extra={"n_id": b.n_id})
            n_id = P.n_id
            sub = self.g.nodes(n_id).with_edges(P.edge_index, P.edge_type)
        else:
            P = None
            n_id = torch.from_numpy(b.n_id).to(dev)
            ei = torch.from_numpy(b.edge_index).to(dev)
            et = self.etype[torch.from_numpy(b.e_id).to(dev)]
            sub = self.g.nodes(n_id).with_edges(ei, et)
        return sub, self.meta[n_id], None if self.extscore is None else self.extscore[n_id], P


def _autocast(cfg: TrainConfig, dev):
    if cfg.amp == "bf16" and torch.device(dev).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def predict(model, X: Inputs, idx: np.ndarray, sampler, cfg: TrainConfig, rng_base: int = 0):
    """P(bot) for ``idx`` (in order) and per-MoE top-1 expert ids."""
    was = model.training
    model.eval()
    idx = np.asarray(idx, dtype=np.int64)
    out = np.empty(len(idx), np.float32)
    routes = {}
    for s in range(0, len(idx), cfg.eval_batch_size):
        b = sampler.sample(idx[s:s + cfg.eval_batch_size], rng_seed=rng_base + s)
        sub, meta, extscore, plan = X.batch(b)
        with _autocast(cfg, X.g.device):
            logits, aux = model(sub, meta, extscore, rows=b.batch_size, plan=plan)
        out[s:s + b.batch_size] = torch.softmax(logits.float(), 1)[:, 1].cpu().numpy()
        for k, r in aux["route"].items():
            routes.setdefault(k, []).append(r.cpu().numpy())
    model.train(was)
    return out, {k: np.concatenate(v) for k, v in routes.items()}


def route_hist(routes: dict, model) -> dict:
    n = {k: m.n for k, m in model.moes().items()}
    return {k: np.bincount(v, minlength=n[k]).tolist() for k, v in routes.items()}


def fit(model, X: Inputs, y: np.ndarray, splits: dict, cfg: TrainConfig, *, csc, run=None,
        log_every: int = 0) -> FitOut:
    from ...harness.sampler import NeighborSampler

    dev = X.g.device
    tr = np.asarray(splits["train"])
    tr = tr[y[tr] >= 0]
    va = np.asarray(splits["val"])
    sampler = NeighborSampler(csc, list(cfg.fanouts))
    ev = NeighborSampler(csc, list(cfg.eval_fanouts or cfg.fanouts))
    loss_fn = nn.CrossEntropyLoss()
    if cfg.freeze_graph_gates:
        model.set_graph_gates_trainable(False)
    # fused: one kernel for the whole update (foreach Adam also increments the CPU step counters one
    # tensor at a time, ~180 host ops per step); same update rule up to float rounding
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                           fused=bool(cfg.fused_adam and dev.type == "cuda"))
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(cfg.milestones), gamma=cfg.gamma)
    ck = BestCheckpoint("best_val", mode="max")
    es = EarlyStopper(patience=cfg.patience, mode="max") if cfg.patience >= 0 else None
    # the schedule trigger (lr drop + graph-gate unfreeze) fires once; with --no-gate-freeze it still
    # drops the lr (main.py ties the 1e-6 lr to best val acc, independently of the gates)
    unfrozen_at = None
    hist, t_train, t_eval, ep = [], 0.0, 0.0, -1
    params = [p for p in model.parameters()]
    for ep in range(cfg.epochs):
        model.train()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t = time.perf_counter()
        tot = torch.zeros((), device=dev)
        bal = torch.zeros((), device=dev)
        cnt = 0
        for bi, b in enumerate(sampler.iter_batches(tr, cfg.batch_size, shuffle=True,
                                                    epoch_seed=cfg.seed * 100_003 + ep)):
            if cfg.max_batches and bi >= cfg.max_batches:
                break
            sub, meta, extscore, plan = X.batch(b)
            with _autocast(cfg, dev):
                logits, aux = model(sub, meta, extscore, plan=plan)
            S = b.batch_size
            ce = loss_fn(logits[:S].float(), sub.y[:S])
            loss = ce + aux["balance"].float()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_value_(params, cfg.clip_value)
            opt.step()
            tot += ce.detach() * S                       # device accumulators: no host sync per step
            bal += aux["balance"].detach().float() * S
            cnt += S
        if unfrozen_at is None:
            sched.step()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t_train += time.perf_counter() - t
        # unfreeze check after the training epoch, on the best val acc so far (main.py order)
        if unfrozen_at is None and ((cfg.unfreeze_acc is not None and ck.best is not None and ck.best > cfg.unfreeze_acc)
                                    or ep + 1 >= cfg.unfreeze_epoch):
            if cfg.freeze_graph_gates:
                model.set_graph_gates_trainable(True)
            for gr in opt.param_groups:
                gr["lr"] = cfg.lr * cfg.unfreeze_lr_factor
            unfrozen_at = ep
        t = time.perf_counter()
        pv, _ = predict(model, X, va, ev, cfg)
        t_eval += time.perf_counter() - t
        mv = compute_metrics(y[va], pv)
        rec = {"train_ce": float(tot) / max(cnt, 1), "train_balance": float(bal) / max(cnt, 1), "val_acc": mv["accuracy"],
               "val_f1": mv["f1"], "val_auc": mv["roc_auc"], "lr": opt.param_groups[0]["lr"],
               "gates_frozen": cfg.freeze_graph_gates and unfrozen_at is None}
        ck.update(model, mv["accuracy"], ep)
        stop = False
        if es is not None:
            es.step(mv["accuracy"])
            stop = es.should_stop
        hist.append({"epoch": ep, **rec})
        if run is not None:
            run.log_epoch(ep, **rec)
        if log_every and (ep % log_every == 0 or stop or ep == cfg.epochs - 1):
            print(f"  epoch {ep:4d} ce={rec['train_ce']:.4f} bal={rec['train_balance']:.4f} "
                  f"val_acc={rec['val_acc']:.4f} val_f1={rec['val_f1']:.4f} lr={rec['lr']:.1e} "
                  f"frozen={rec['gates_frozen']} ({t_train / (ep + 1):.2f} s/ep train)", flush=True)
        if stop:
            break
    ck.restore(model)
    prob = np.full(len(y), np.nan, np.float32)
    routes = {}
    for s in ("val", "test"):
        ix = np.asarray(splits[s])
        prob[ix], r = predict(model, X, ix, ev, cfg)
        routes[s] = route_hist(r, model)
    n_ep = ep + 1
    return FitOut(prob, n_ep, ck.info(), round(t_train, 2), round(t_train / max(n_ep, 1), 3),
                  round(t_eval / max(n_ep, 1), 3), unfrozen_at, routes, hist)
