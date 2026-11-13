"""Training utilities: seeding, early stopping, checkpoint-by-value, class weights."""
from __future__ import annotations

import copy
import os
import random
from typing import Any

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (CPU + CUDA). ``deterministic`` also requests deterministic
    cuDNN/cuBLAS kernels (slower; scatter ops may still be nondeterministic on GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


class EarlyStopper:
    """Patience-based early stopping on a validation value.

    >>> es = EarlyStopper(patience=10, mode="max")
    >>> for epoch in ...: improved = es.step(val_f1); if es.should_stop: break
    """

    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 0.0):
        if mode not in ("max", "min"):
            raise ValueError(mode)
        self.patience, self.mode, self.min_delta = patience, mode, min_delta
        self.best: float | None = None
        self.best_epoch = -1
        self.bad = 0
        self.epoch = -1

    def step(self, value: float) -> bool:
        self.epoch += 1
        v = float(value)
        better = (self.best is None or
                  (v > self.best + self.min_delta if self.mode == "max" else v < self.best - self.min_delta))
        if better:
            self.best, self.best_epoch, self.bad = v, self.epoch, 0
        else:
            self.bad += 1
        return better

    @property
    def should_stop(self) -> bool:
        return self.patience >= 0 and self.bad > self.patience


class BestCheckpoint:
    """Keeps a DEEP COPY of the best model state (the official GCN/GAT/BIC code saved
    ``model.state_dict()`` by reference, so their "best" was the last epoch).

    ``select="best_val"`` restores the best state; ``select="last_epoch"`` keeps the final weights
    (what several official scripts effectively report). Either way ``restore`` is safe to call.
    """

    def __init__(self, select: str = "best_val", mode: str = "max", to_cpu: bool = True):
        if select not in ("best_val", "last_epoch"):
            raise ValueError(select)
        self.select, self.mode, self.to_cpu = select, mode, to_cpu
        self.best: float | None = None
        self.best_epoch = -1
        self.state: dict | None = None

    def _copy(self, model) -> dict:
        sd = model.state_dict()
        try:
            import torch
            return {k: (v.detach().to("cpu", copy=True) if self.to_cpu else v.detach().clone())
                    if isinstance(v, torch.Tensor) else copy.deepcopy(v) for k, v in sd.items()}
        except ImportError:  # pragma: no cover
            return copy.deepcopy(sd)

    def update(self, model, value: float, epoch: int) -> bool:
        v = float(value)
        better = self.best is None or (v > self.best if self.mode == "max" else v < self.best)
        if better:
            self.best, self.best_epoch = v, epoch
            if self.select == "best_val":
                self.state = self._copy(model)
        return better

    def restore(self, model) -> Any:
        if self.select == "best_val" and self.state is not None:
            model.load_state_dict(self.state)
        return model

    def info(self) -> dict:
        return {"select": self.select, "best_value": self.best, "best_epoch": self.best_epoch}


def class_weights(labels: np.ndarray, mode: str = "none") -> np.ndarray | None:
    """Per-class loss weights [w_human, w_bot] from TRAIN labels (pass only train labels).

    none -> None; balanced -> n / (2 n_c) (sklearn 'balanced'); sqrt -> sqrt of balanced;
    'w0,w1' -> explicit weights."""
    if mode in (None, "none"):
        return None
    y = np.asarray(labels)
    y = y[y >= 0]
    cnt = np.bincount(y, minlength=2).astype(np.float64)
    if "," in mode:
        return np.array([float(x) for x in mode.split(",")], np.float64)
    bal = len(y) / (2 * np.maximum(cnt, 1))
    if mode == "balanced":
        return bal
    if mode == "sqrt":
        return np.sqrt(bal)
    raise ValueError(f"unknown class-weight mode {mode!r}")


def to_torch_weights(w: np.ndarray | None, device=None):
    if w is None:
        return None
    import torch
    return torch.tensor(w, dtype=torch.float32, device=device)
