"""Run bookkeeping: one directory per (method, dataset, run) with config, per-epoch log,
predictions and a ``result.json`` carrying metrics, provenance, peak memory and wall time.

    runs/<method>/<dataset>/<run_name>/
        config.json      everything needed to rerun (method args, seed, extscore, selection mode)
        epochs.jsonl     one JSON line per log_epoch call
        preds_<split>.npz  idx + P(bot) (+ labels) per evaluated split
        result.json      status, metrics per split, provenance, memory, wall time, extscore report

Usage::

    with Run("botrgcn", "twibot20", config=vars(args), seed=s, data_files=[...]) as run:
        ...
        run.log_epoch(epoch, train_loss=..., val_f1=...)
        run.add_predictions("test", test_idx, prob, labels[test_idx])
        run.finish({"val": compute_metrics(...), "test": compute_metrics(...)},
                   selection=ckpt.info(), extscore=J.report())

Provenance is git-free: sha256 over (the ``bb`` package sources + the method's own source files,
the config, digests of the data artifacts the run declares).
"""
from __future__ import annotations

import hashlib
import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..io import dumps_json, file_digest, read_json, tree_source_digest, write_json
from ..mem import memory_report
from ..paths import REPO_ROOT, dataset_paths, runs_root


def data_digests(dataset: str, extra: Iterable[str | Path] = ()) -> dict[str, str]:
    """Digests of the index files every run depends on plus ``extra`` artifact paths."""
    dp = dataset_paths(dataset)
    files = [dp.users, dp.meta, *sorted(dp.splits.glob("*.npy")), *map(Path, extra)]
    out = {}
    for f in files:
        f = Path(f)
        if f.is_dir():
            for g in sorted(f.glob("*")):
                if g.is_file():
                    out[str(g.relative_to(dp.root.parent) if dp.root.parent in g.parents else g)] = file_digest(g)
        elif f.exists():
            out[str(f.relative_to(dp.root.parent) if dp.root.parent in f.parents else f)] = file_digest(f)
    return out


class Run:
    def __init__(self, method: str, dataset: str, config: dict, seed: int, *, run_name: str | None = None,
                 data_files: Iterable[str | Path] = (), source_files: Iterable[str | Path] = (),
                 root: Path | None = None):
        self.method, self.dataset, self.seed = method, dataset, seed
        self.config = {**config, "seed": seed, "method": method, "dataset": dataset}
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        extscore = config.get("extscore_tag") or config.get("with_extscore") or "none"
        self.name = run_name or f"{config.get('variant', 'default')}_extscore-{extscore}_s{seed}_{stamp}"
        base = Path(root or runs_root()) / method / dataset
        d, k = base / self.name, 1
        while d.exists():  # never overwrite an earlier run
            k += 1
            d = base / f"{self.name}_{k}"
        self.name = d.name
        self.dir = d
        self.dir.mkdir(parents=True)
        self.t0 = time.time()
        self.preds: dict[str, str] = {}
        srcs = [REPO_ROOT / "bb", *map(Path, source_files)]
        main = getattr(sys.modules.get("__main__"), "__file__", None)
        if main:
            srcs.append(Path(main))
        self.source_digest = tree_source_digest(srcs)
        self.data = data_digests(dataset, data_files)
        cfg_json = json.dumps(self.config, sort_keys=True, default=str)
        self.provenance = hashlib.sha256(
            (self.source_digest + cfg_json + json.dumps(self.data, sort_keys=True)).encode()).hexdigest()[:16]
        write_json(self.dir / "config.json", self.config)
        self._epochs = open(self.dir / "epochs.jsonl", "a")
        self.finished = False

    def log_epoch(self, epoch: int, **metrics: Any) -> None:
        rec = {"epoch": epoch, "t": round(time.time() - self.t0, 2), **metrics}
        self._epochs.write(dumps_json(rec).replace("\n", " ") + "\n")
        self._epochs.flush()

    def add_predictions(self, split: str, idx: np.ndarray, prob: np.ndarray, labels: np.ndarray | None = None) -> None:
        p = self.dir / f"preds_{split}.npz"
        arrs = {"idx": np.asarray(idx, np.int64), "prob": np.asarray(prob, np.float32)}
        if labels is not None:
            arrs["label"] = np.asarray(labels, np.int8)
        np.savez_compressed(p, **arrs)
        self.preds[split] = p.name

    def finish(self, metrics: dict[str, dict], *, status: str = "ok", **extra: Any) -> dict:
        res = {
            "status": status, "method": self.method, "dataset": self.dataset, "seed": self.seed,
            "run": self.name, "metrics": metrics, "config": self.config,
            "provenance": {"hash": self.provenance, "source_digest": self.source_digest, "data": self.data,
                           "python": sys.version.split()[0], "host": socket.gethostname(),
                           "torch": _torch_version(), "argv": sys.argv},
            "memory": memory_report(), "wall_seconds": round(time.time() - self.t0, 2),
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "predictions": self.preds, **extra,
        }
        write_json(self.dir / "result.json", res)
        self._epochs.close()
        self.finished = True
        return res

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, et, ev, tb) -> bool:
        if not self.finished:
            self.finish({}, status="failed" if et else "unfinished", error=repr(ev) if ev else None)
        return False


def _torch_version() -> str | None:
    t = sys.modules.get("torch")
    return getattr(t, "__version__", None) if t else None


def load_results(method: str, dataset: str, *, status: str = "ok", root: Path | None = None,
                 where: dict | None = None) -> list[dict]:
    """All result.json of a method/dataset, optionally filtered by config values (``where``)."""
    base = Path(root or runs_root()) / method / dataset
    out = []
    for p in sorted(base.glob("*/result.json")):
        r = read_json(p)
        if status and r.get("status") != status:
            continue
        if where and any(r["config"].get(k) != v for k, v in where.items()):
            continue
        out.append(r)
    return out


def aggregate(results: list[dict], split: str = "test") -> dict:
    """Mean/std over runs (seeds) of a split's metrics."""
    from .metrics import summarize

    return summarize([r["metrics"][split] for r in results if split in r.get("metrics", {})])
