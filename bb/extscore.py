"""The ExtScore column: an external decision model's P(bot) per account, with a missing indicator.

Storage: ``data/<dataset>/extscore/<tag>.parquet`` with one row per idx (all N rows)::

    idx int64 | p float32 (NaN = not scored) | missing uint8 | fold int8 (-1 = not out-of-fold) | q_* float32

plus ``<tag>.meta.json`` (decision-model version, prompt hash, scoring date, ``label_blind``,
``oof``, ``optimised_on`` = idx list the decision model was optimised on, coverage).

One common interface for every method::

    from bb.extscore import add_extscore_args, prepare_extscore
    add_extscore_args(parser)                          # --extscore none|p, --extscore-tag TAG, --with-extscore TAG
    J = prepare_extscore(args, dataset, splits)        # ExtScoreInputs (J.enabled False when --extscore none)
    x = J.append(x, rows, kind="neural")          # [z, m] appended; no-op when disabled

Transforms (shared_requirements.md section 8):
- neural ``[z, m]``: z = (logit(clip(p, 1e-4, 1-1e-4)) - mu) / sd, mu/sd fitted on OBSERVED TRAIN
  rows only, z = 0 where missing, m = missing indicator. Never re-standardise it with other columns.
- tree ``[p or -1, m]``; ``nan`` (p with NaN, for NaN-native learners); ``meanfill`` ``[p or
  train-mean, m]`` (Kudugunta: SMOTE interpolates).

Leakage guard (``validate_extscore``): observed TRAIN rows must be out-of-fold (fold >= 0) or come from
a label-blind decision model (meta ``label_blind: true``); the decision model must not have been
optimised on any evaluation user (meta ``optimised_on``), nor - for graph methods - on their
neighbours. ``prepare_extscore`` raises ``ExtScoreLeakageError`` unless ``--extscore-unsafe`` is given.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .io import read_json, write_json
from .paths import dataset_paths

CLIP = 1e-4


class ExtScoreLeakageError(ValueError):
    pass


@dataclass
class ExtScoreColumn:
    dataset: str
    tag: str
    p: np.ndarray        # float32 [N], NaN where missing
    missing: np.ndarray  # uint8 [N]
    fold: np.ndarray     # int8 [N]
    q: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.p)

    @property
    def observed(self) -> np.ndarray:
        return self.missing == 0


# ---------------------------------------------------------------------------------------------
# storage


def write_extscore(dataset: str, tag: str, idx: np.ndarray, p: np.ndarray, n: int | None = None, *,
              fold: np.ndarray | None = None, q: dict[str, np.ndarray] | None = None,
              meta: dict | None = None) -> Path:
    """Write a ExtScore table for ``dataset``. ``idx``/``p`` (and optional ``fold``, ``q_*``) cover the
    scored users; every other row is written as missing. ``meta`` should state ``label_blind``,
    ``oof``, ``optimised_on`` (idx list), ``model``, ``prompt_hash``, ``scored_at``."""
    from .data.users import num_users

    dp = dataset_paths(dataset)
    n = n if n is not None else num_users(dataset)
    idx = np.asarray(idx, dtype=np.int64)
    pv = np.full(n, np.nan, np.float32)
    pv[idx] = np.asarray(p, np.float32)
    miss = np.isnan(pv).astype(np.uint8)
    fv = np.full(n, -1, np.int8)
    if fold is not None:
        fv[idx] = np.asarray(fold, np.int8)
    cols = {"idx": np.arange(n, dtype=np.int64), "p": pv, "missing": miss, "fold": fv}
    for k, v in (q or {}).items():
        qq = np.full(n, np.nan, np.float32)
        qq[idx] = np.asarray(v, np.float32)
        cols[k if k.startswith("q_") else f"q_{k}"] = qq
    dp.extscore.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), dp.extscore_table(tag))
    m = dict(meta or {})
    m.setdefault("label_blind", False)
    m.setdefault("oof", fold is not None)
    m.setdefault("optimised_on", [])
    m["n"] = n
    m["n_scored"] = int((miss == 0).sum())
    write_json(dp.extscore_meta(tag), m)
    return dp.extscore_table(tag)


def load_extscore(dataset: str, tag: str) -> ExtScoreColumn:
    dp = dataset_paths(dataset)
    t = pq.read_table(dp.extscore_table(tag))
    idx = t.column("idx").to_numpy()
    if not np.array_equal(idx, np.arange(len(idx))):
        raise ValueError(f"{dp.extscore_table(tag)}: idx must be 0..N-1 in order")
    p = t.column("p").to_numpy(zero_copy_only=False).astype(np.float32)
    miss = t.column("missing").to_numpy().astype(np.uint8)
    if not np.array_equal(miss.astype(bool), np.isnan(p)):
        raise ValueError(f"{dp.extscore_table(tag)}: missing flag disagrees with NaN p")
    q = {c: t.column(c).to_numpy(zero_copy_only=False).astype(np.float32) for c in t.column_names if c.startswith("q_")}
    meta = read_json(dp.extscore_meta(tag)) if dp.extscore_meta(tag).exists() else {}
    return ExtScoreColumn(dataset, tag, p, miss, t.column("fold").to_numpy().astype(np.int8), q, meta)


# ---------------------------------------------------------------------------------------------
# transforms


def _logit(p: np.ndarray, clip: float = CLIP) -> np.ndarray:
    p = np.clip(p.astype(np.float64), clip, 1 - clip)
    return np.log(p / (1 - p))


def neural_transform(j: ExtScoreColumn, train_idx: np.ndarray, clip: float = CLIP) -> tuple[np.ndarray, dict]:
    """[z, m] float32 [N,2]; mu/sd from observed train rows only."""
    tr = np.asarray(train_idx)
    obs_tr = tr[j.observed[tr]]
    if len(obs_tr) < 2:
        raise ValueError(f"ExtScore {j.tag}: {len(obs_tr)} observed train rows; cannot fit mu/sd "
                         "(a column absent from train cannot be learned; use late fusion)")
    lt = _logit(j.p[obs_tr], clip)
    mu, sd = float(lt.mean()), float(lt.std(ddof=0))
    sd = sd if sd > 0 else 1.0
    z = np.zeros(j.n, np.float64)
    o = j.observed
    z[o] = (_logit(j.p[o], clip) - mu) / sd
    x = np.stack([z, j.missing.astype(np.float64)], 1).astype(np.float32)
    return x, {"mu": mu, "sd": sd, "n_fit": int(len(obs_tr)), "clip": clip}


def tree_transform(j: ExtScoreColumn, fill: float = -1.0) -> np.ndarray:
    """[p or fill, m] float32 [N,2]."""
    p = np.where(j.observed, j.p, fill).astype(np.float32)
    return np.stack([p, j.missing.astype(np.float32)], 1)


def nan_transform(j: ExtScoreColumn) -> np.ndarray:
    """p with NaN where missing, float32 [N,1] (NaN-native learners)."""
    return j.p.astype(np.float32)[:, None]


def meanfill_transform(j: ExtScoreColumn, train_idx: np.ndarray) -> tuple[np.ndarray, dict]:
    """[p or train-observed mean, m] float32 [N,2] (Kudugunta: SMOTE interpolates between rows)."""
    tr = np.asarray(train_idx)
    obs = tr[j.observed[tr]]
    mean = float(j.p[obs].mean()) if len(obs) else 0.5
    p = np.where(j.observed, j.p, mean).astype(np.float32)
    return np.stack([p, j.missing.astype(np.float32)], 1), {"fill": mean}


# ---------------------------------------------------------------------------------------------
# guard + reporting


def coverage(j: ExtScoreColumn, splits: dict[str, np.ndarray]) -> dict:
    out = {}
    for s, idx in splits.items():
        idx = np.asarray(idx)
        n_obs = int(j.observed[idx].sum()) if len(idx) else 0
        out[s] = {"n": int(len(idx)), "observed": n_obs, "coverage": (n_obs / len(idx)) if len(idx) else None}
    return out


def validate_extscore(j: ExtScoreColumn, splits: dict[str, np.ndarray], *, eval_splits=("val", "test"),
                 neighbours=None) -> list[str]:
    """Return a list of guard violations (empty = OK).

    1. observed train rows must be out-of-fold (fold >= 0) unless meta.label_blind is true;
    2. meta.optimised_on must not intersect any evaluation split;
    3. with ``neighbours`` (a CSR or callable idx -> neighbour idx, for graph methods), also not the
       1-hop neighbours of evaluation users.
    """
    problems = []
    tr = np.asarray(splits.get("train", []))
    if len(tr):
        obs = tr[j.observed[tr]]
        if not j.meta.get("label_blind", False):
            bad = obs[j.fold[obs] < 0]
            if len(bad):
                problems.append(f"{len(bad)} observed train rows are not out-of-fold (fold = -1) and the "
                                f"decision model is not declared label_blind")
    opt = np.asarray(j.meta.get("optimised_on") or [], dtype=np.int64)
    if len(opt):
        for s in eval_splits:
            if s in splits:
                inter = np.intersect1d(opt, np.asarray(splits[s]))
                if len(inter):
                    problems.append(f"decision model was optimised on {len(inter)} users of split {s!r}")
        if neighbours is not None:
            ev = np.concatenate([np.asarray(splits[s]) for s in eval_splits if s in splits])
            nb = _neighbours(neighbours, ev)
            inter = np.intersect1d(opt, nb)
            if len(inter):
                problems.append(f"decision model was optimised on {len(inter)} graph neighbours of evaluation users")
    need = max((int(np.max(v)) + 1 for v in splits.values() if len(v)), default=0)
    if j.n < need:
        problems.append(f"ExtScore table has {j.n} rows but the splits reference idx up to {need - 1}")
    return problems


def _neighbours(nb, idx: np.ndarray) -> np.ndarray:
    if callable(nb):
        return np.unique(nb(idx))
    indptr, indices = np.asarray(nb.indptr), nb.indices
    parts = [np.asarray(indices[indptr[i]:indptr[i + 1]]) for i in idx]
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)


def column_only_baseline(j: ExtScoreColumn, labels: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
                         objective: str = "f1") -> dict:
    """ExtScore alone: threshold on p chosen on val (maximising bot-F1 by default); missing rows are
    predicted human and scored with the val-observed mean probability for AUC."""
    from .harness.metrics import compute_metrics

    va, te = np.asarray(val_idx), np.asarray(test_idx)
    fill = float(np.nanmean(j.p[va])) if j.observed[va].any() else 0.5
    pv = np.where(j.observed, j.p, fill)
    best_t = 0.5
    cand = np.unique(np.round(pv[va][j.observed[va]], 4))
    if len(cand):
        # Vectorised sweep (one sort instead of a full compute_metrics per threshold); identical
        # to scoring each candidate with compute_metrics(prob >= t) and keeping the first maximum.
        yv = np.asarray(labels[va]).astype(np.int64)
        pe = np.where(j.observed[va], pv[va], 0.0).astype(np.float64)
        pos, neg = np.sort(pe[yv == 1]), np.sort(pe[yv == 0])
        tp = len(pos) - np.searchsorted(pos, cand, side="left")   # #(p >= t) among bots
        fp = len(neg) - np.searchsorted(neg, cand, side="left")
        fn, tn = len(pos) - tp, len(neg) - fp
        if objective == "f1":
            d = 2 * tp + fp + fn
            v = np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)
        else:
            v = (tp + tn) / max(len(yv), 1)
        best_t = float(cand[int(np.argmax(v))])
    probs = lambda ix: np.where(j.observed[ix], pv[ix], 0.0)  # noqa: E731
    return {"threshold": best_t, "objective": objective,
            "val": compute_metrics(labels[va], probs(va), threshold=best_t, prob_for_auc=pv[va]),
            "test": compute_metrics(labels[te], probs(te), threshold=best_t, prob_for_auc=pv[te])}


# ---------------------------------------------------------------------------------------------
# the common method-facing interface


def add_extscore_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("ExtScore column")
    g.add_argument("--extscore", choices=["none", "p"], default="none", help="none reproduces the baseline exactly")
    g.add_argument("--extscore-tag", default=None, help="data/<dataset>/extscore/<tag>.parquet")
    g.add_argument("--with-extscore", metavar="TAG", default=None, help="shorthand for --extscore p --extscore-tag TAG")
    g.add_argument("--extscore-unsafe", action="store_true", help="downgrade leakage-guard violations to warnings")


@dataclass
class ExtScoreInputs:
    enabled: bool
    tag: str | None = None
    column: ExtScoreColumn | None = None
    neural: np.ndarray | None = None      # [N,2] (z, m)
    tree: np.ndarray | None = None        # [N,2] (p or -1, m)
    params: dict = field(default_factory=dict)
    cov: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    baseline: dict | None = None

    @property
    def width(self) -> int:
        return 2 if self.enabled else 0

    def append(self, x: np.ndarray, rows: np.ndarray | None = None, kind: str = "neural") -> np.ndarray:
        """Concatenate the 2 ExtScore columns to ``x`` (rows = idx of x's rows; None = all N).
        Returns ``x`` unchanged (same object) when disabled."""
        if not self.enabled:
            return x
        if kind == "neural":
            cols = self.neural
        elif kind == "tree":
            cols = self.tree
        elif kind == "meanfill":
            cols = self.params["meanfill"]
        else:
            raise ValueError(kind)
        c = cols if rows is None else cols[np.asarray(rows)]
        return np.concatenate([x, c.astype(x.dtype, copy=False)], axis=1)

    def report(self) -> dict:
        if not self.enabled:
            return {"extscore": "none"}
        return {"extscore": "p", "tag": self.tag, "transform_params": {k: v for k, v in self.params.items() if k != "meanfill"},
                "coverage": self.cov, "guard_problems": self.problems, "column_only_baseline": self.baseline,
                "decision_model": {k: v for k, v in (self.column.meta if self.column else {}).items()
                                   if k != "optimised_on"}}


def prepare_extscore(args_or_tag, dataset: str, splits: dict[str, np.ndarray], labels: np.ndarray | None = None,
                *, neighbours=None, unsafe: bool = False) -> ExtScoreInputs:
    """Build the ExtScore inputs for a run. ``args_or_tag`` is the argparse namespace from
    ``add_extscore_args`` (or a tag string, or None for disabled). Computes coverage, runs the guard and
    (if ``labels`` given) the column-only baseline on val/test."""
    tag = None
    if isinstance(args_or_tag, argparse.Namespace):
        a = args_or_tag
        if getattr(a, "with_extscore", None):
            tag = a.with_extscore
        elif getattr(a, "extscore", "none") == "p":
            tag = a.extscore_tag
            if not tag:
                raise ValueError("--extscore p requires --extscore-tag")
        unsafe = unsafe or getattr(a, "extscore_unsafe", False)
    else:
        tag = args_or_tag
    if not tag:
        return ExtScoreInputs(enabled=False)
    j = load_extscore(dataset, tag)
    tr = splits["train"]
    neural, params = neural_transform(j, tr)
    mf, mfp = meanfill_transform(j, tr)
    params = {**params, "meanfill_fill": mfp["fill"], "meanfill": mf}
    problems = validate_extscore(j, splits, neighbours=neighbours)
    if problems and not unsafe:
        raise ExtScoreLeakageError("; ".join(problems))
    base = None
    if labels is not None and "val" in splits and "test" in splits:
        base = column_only_baseline(j, labels, splits["val"], splits["test"])
    return ExtScoreInputs(True, tag, j, neural, tree_transform(j), params, coverage(j, splits), problems, base)


def make_synthetic_extscore(dataset: str, tag: str = "synthetic", signal: float = 1.5, coverage: float = 0.9,
                       n_folds: int = 5, seed: int = 0, support_coverage: float = 0.0,
                       labels: np.ndarray | None = None, splits: dict[str, np.ndarray] | None = None,
                       write: bool = True):
    """Synthetic ExtScore for tests and smoke runs (NO real decision-model values exist yet).

    p = sigmoid(signal * (2y - 1) + N(0, 1)) for labelled users, observed with probability
    ``coverage`` (support users with ``support_coverage``, p = sigmoid(N(0,1)) there); train rows get
    an out-of-fold id in 0..n_folds-1 so the guard passes. Meta marks it ``synthetic: true``.
    Returns the written path (or a ExtScoreColumn when ``write=False``)."""
    from .data.users import load_split, load_users

    rng = np.random.default_rng(seed)
    if labels is None:
        labels = load_users(dataset, columns=["label"])["label"].to_numpy()
    n = len(labels)
    if splits is None:
        splits = {s: np.asarray(load_split(dataset, s)) for s in ("train", "val", "test", "support")}
    y = labels.astype(np.float64)
    noise = rng.standard_normal(n)
    logit = np.where(labels >= 0, signal * (2 * y - 1), 0.0) + noise
    p = (1 / (1 + np.exp(-logit))).astype(np.float32)
    cov_draw = rng.random(n)
    obs = np.where(labels >= 0, cov_draw < coverage, cov_draw < support_coverage)
    fold = np.full(n, -1, np.int8)
    tr = np.asarray(splits["train"])
    fold[tr] = rng.integers(0, n_folds, len(tr))
    idx = np.flatnonzero(obs)
    meta = {"synthetic": True, "label_blind": False, "oof": True, "optimised_on": [],
            "model": f"synthetic(signal={signal}, coverage={coverage}, seed={seed})",
            "note": "test fixture only; not a decision-model output"}
    if not write:
        pv = np.full(n, np.nan, np.float32)
        pv[idx] = p[idx]
        return ExtScoreColumn(dataset, tag, pv, np.isnan(pv).astype(np.uint8), fold, {}, meta)
    return write_extscore(dataset, tag, idx, p[idx], n, fold=fold[idx], meta=meta)
