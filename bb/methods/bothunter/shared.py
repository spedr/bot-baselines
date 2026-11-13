"""Helpers shared by the two profile-feature random-forest baselines (BotHunter and SGBot).

Both TwiBot-22 benchmark re-implementations (``src/BotHunter``, ``src/SGBot``) use the same
account-age parser (``calc_age``: days between Dec 31 of the dataset year and ``created_at``,
null -> 730), the same custom-profile-image flag, ``len(s.strip())`` string lengths, and the same
``RandomForestClassifier(n_estimators=100)`` with sklearn defaults. This module holds those pieces
and the seed loop (fit on train, report val/test through ``bb.harness.Run``), so that the two
method packages only define their feature vectors. Everything is vectorised over pandas/numpy
columns; no per-user Python loop.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from bb.data import load_split, load_users
from bb.harness import Run, aggregate, compute_metrics
from bb.extscore import add_extscore_args, prepare_extscore

# Dataset year of the benchmark's ``calc_age`` reference date (Dec 31 of that year).
REF_YEAR = {"twibot22": 2022, "twibot20": 2020, "cresci15": 2015}
NULL_AGE_DAYS = 730
DEFAULT_IMAGE_TOKEN = "default_profile_normal"
# sklearn's hard ``predict`` is argmax(proba): a 50/50 vote is class 0 (human). compute_metrics uses
# ``prob >= threshold``, so the smallest float above 0.5 reproduces ``predict`` exactly.
ARGMAX_THRESHOLD = float(np.nextafter(0.5, 1.0))
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------------------------
# column helpers (pandas nullable columns in, float64 numpy out)


def count_col(s: pd.Series) -> np.ndarray:
    """Integer count column; null -> 0 (benchmark: ``public_metrics.x`` or 0)."""
    return s.astype("Float64").fillna(0).to_numpy(np.float64)


def bool_col(s: pd.Series) -> np.ndarray:
    """Boolean flag; null -> 0."""
    return s.astype("Float64").fillna(0).to_numpy(np.float64)


def strip_len(s: pd.Series) -> np.ndarray:
    """``len(s.strip())`` in code points; null -> 0."""
    return s.str.strip().str.len().astype("Float64").fillna(0).to_numpy(np.float64)


def digit_count(s: pd.Series) -> np.ndarray:
    """Number of characters in '0'..'9' (ASCII digits only, as the benchmark's ``c in '0123456789'``)."""
    return s.str.count("[0-9]").astype("Float64").fillna(0).to_numpy(np.float64)


def custom_image_flag(s: pd.Series) -> np.ndarray:
    """``int('default_profile_normal' not in profile_image_url)``; null -> 0."""
    return (~s.str.contains(DEFAULT_IMAGE_TOKEN, regex=False)).astype("Float64").fillna(0).to_numpy(np.float64)


def account_age_days(created_at: pd.Series, year: int) -> tuple[np.ndarray, dict]:
    """Benchmark ``calc_age``: ``(datetime(year,12,31) - created_at_naive).days``, null -> 730.

    ``created_at`` is a UTC timestamp column, so the naive wall time equals UTC and the days are
    floored exactly like ``timedelta.days``. Returns the age (float64) and a small report."""
    ref = pd.Timestamp(year=year, month=12, day=31, tz="UTC")
    delta = ref - created_at
    days = np.floor(delta.dt.total_seconds().to_numpy(np.float64, na_value=np.nan) / 86400.0)
    null = np.isnan(days)
    days[null] = NULL_AGE_DAYS
    return days, {"reference": str(ref.date()), "null_created_at": int(null.sum()),
                  "age_le_0": int((days <= 0).sum())}


def safe_div_age(x: np.ndarray, age: np.ndarray) -> np.ndarray:
    """x / age with age floored at 1 (the benchmark would crash on age 0; never hit on our data)."""
    return x / np.maximum(age, 1.0)


def codepoints(s: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Unicode code points of every string as a zero-padded uint32 matrix [N, maxlen] plus the
    lengths (null -> length 0). Screen names are at most 16 characters, so this is ~64 MB for 1M."""
    filled = s.fillna("")
    lens = filled.str.len().to_numpy(np.int64)
    width = max(1, int(lens.max()) if len(lens) else 1)
    u = np.asarray(filled.to_numpy(dtype=object), dtype=f"U{width}")
    return u.view(np.uint32).reshape(len(u), width), lens


def shannon_entropy(s: pd.Series) -> np.ndarray:
    """Shannon entropy (natural log, as ``math.log`` in the benchmark) of the character
    distribution of each string; 0 for empty or null. Vectorised: one ``np.unique`` over
    (row, code point) keys. H = log n - (1/n) sum_c c log c."""
    cp, lens = codepoints(s)
    r, c = np.nonzero(cp)
    key = (r.astype(np.int64) << 21) | cp[r, c].astype(np.int64)
    uk, cnt = np.unique(key, return_counts=True)
    row = uk >> 21
    cnt = cnt.astype(np.float64)
    sclogc = np.bincount(row, weights=cnt * np.log(cnt), minlength=len(lens))
    n = lens.astype(np.float64)
    out = np.zeros(len(lens), np.float64)
    pos = n > 0
    out[pos] = np.log(n[pos]) - sclogc[pos] / n[pos]
    return np.maximum(out, 0.0)  # clip -1e-16 rounding for single-character names


# ---------------------------------------------------------------------------------------------
# data + RF loop


def load_index(dataset: str, split: str = "official", split_seed: int = 0
               ) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Labels [N], split idx arrays, and ``rows`` = concat(train, val, test) (the rows the feature
    matrix is built for; positions of split s are a contiguous slice).

    ``split="random"``: a 70/20/10 random re-split of the labelled users (seeded by ``split_seed``),
    for the sanity check the specs ask for on Cresci-2015 (whose official split is contiguous per
    subset in users.csv order); never used for reported benchmark numbers."""
    y = load_users(dataset, columns=["label"])["label"].to_numpy()
    splits = {s: np.asarray(load_split(dataset, s), dtype=np.int64) for s in SPLITS}
    if split == "random":
        lab = np.sort(np.concatenate([splits[s] for s in SPLITS]))
        perm = np.random.default_rng(split_seed).permutation(lab)
        a, b = int(round(0.7 * len(perm))), int(round(0.9 * len(perm)))
        splits = {"train": np.sort(perm[:a]), "val": np.sort(perm[a:b]), "test": np.sort(perm[b:])}
    rows = np.concatenate([splits[s] for s in SPLITS])
    if np.any(y[rows] < 0):
        raise ValueError(f"{dataset}: unlabelled users in train/val/test")
    return y, splits, rows


def common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--dataset", required=True, choices=sorted(REF_YEAR))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="RandomForest random_state per run (benchmark: unseeded, 5 runs)")
    ap.add_argument("--device", default="cpu", help="accepted for interface parity; RF is CPU only")
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=8, help="RF threads (<= 8 on the shared machine)")
    ap.add_argument("--reference-year", type=int, default=None,
                    help="age reference = Dec 31 of this year (default: dataset year 2022/2020/2015)")
    ap.add_argument("--drop-constant", action="store_true",
                    help="drop feature columns that are constant on the train rows (OFF = benchmark: "
                         "constant slots are kept because they consume max_features draws)")
    ap.add_argument("--split", choices=["official", "random"], default="official",
                    help="random = seeded 70/20/10 re-split of the labelled users (sanity check only)")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--perm-repeats", type=int, default=5,
                    help="permutation-importance repeats for the ExtScore columns on val (ExtScore runs only)")
    add_extscore_args(ap)


def constant_columns(X: np.ndarray, pos: slice | np.ndarray) -> list[int]:
    sub = X[pos]
    return [j for j in range(X.shape[1]) if sub.shape[0] == 0 or np.all(sub[:, j] == sub[0, j])]


def _perm_importance(rf, X: np.ndarray, y: np.ndarray, cols: list[int], names: list[str],
                     repeats: int, seed: int) -> dict:
    """Drop in val ROC-AUC / bot-F1 when one column is permuted (ExtScore columns only)."""
    rng = np.random.default_rng(seed)
    p0 = rf.predict_proba(X)[:, 1]
    m0 = compute_metrics(y, p0, threshold=ARGMAX_THRESHOLD)
    out = {}
    for j in cols:
        d_auc, d_f1 = [], []
        col = X[:, j].copy()
        for _ in range(repeats):
            X[:, j] = rng.permutation(col)
            m = compute_metrics(y, rf.predict_proba(X)[:, 1], threshold=ARGMAX_THRESHOLD)
            d_auc.append(m0["roc_auc"] - m["roc_auc"])
            d_f1.append(m0["f1"] - m["f1"])
        X[:, j] = col
        out[names[j]] = {"roc_auc_drop_mean": float(np.mean(d_auc)), "roc_auc_drop_std": float(np.std(d_auc)),
                         "f1_drop_mean": float(np.mean(d_f1)), "f1_drop_std": float(np.std(d_f1)),
                         "repeats": repeats, "split": "val"}
    return out


def release_memory() -> None:
    """Hand freed heap back to the OS between seeds: without it glibc keeps the fragments of each
    freed forest (~10M tree nodes built by 8 threads) and peak RSS creeps up by ~0.2 GiB over a
    5-seed TwiBot-22 batch. Pure memory bookkeeping; no effect on any result."""
    gc.collect()
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except Exception:  # pragma: no cover - pyarrow is a dependency; be defensive anyway
        pass
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):  # non-glibc platform
        pass


def run_rf_method(method: str, args: argparse.Namespace,
                  build: Callable[[str, np.ndarray, dict, argparse.Namespace], tuple[np.ndarray, list[str], dict]],
                  source_files: list[Path], data_files: list[Path]) -> list[dict]:
    """Build features once, then one ``Run`` per seed: fit RF on train, score val/test."""
    from sklearn.ensemble import RandomForestClassifier

    if args.n_jobs > 8:
        raise SystemExit("--n-jobs is capped at 8 on the shared machine")
    if args.device != "cpu":
        print(f"note: {method} is a CPU random forest; --device {args.device} ignored")
    ds = args.dataset
    t0 = time.time()
    y, splits, rows = load_index(ds, args.split, args.split_seed)
    X, names, info = build(ds, rows, splits, args)
    assert X.shape == (len(rows), len(names)) and X.dtype == np.float32
    n = {s: len(splits[s]) for s in SPLITS}
    # rows = concat(train, val, test): each split is a contiguous block, so slices (views, no copy)
    pos = {"train": slice(0, n["train"]), "val": slice(n["train"], n["train"] + n["val"]),
           "test": slice(n["train"] + n["val"], len(rows))}
    const = constant_columns(X, pos["train"])
    info["constant_on_train"] = [names[j] for j in const]
    info["base_width"] = X.shape[1]
    if args.drop_constant and const:
        keep = [j for j in range(X.shape[1]) if j not in set(const)]
        X, names = np.ascontiguousarray(X[:, keep]), [names[j] for j in keep]
        info["dropped_columns"] = info["constant_on_train"]
    J = prepare_extscore(args, ds, splits, labels=y)
    X = J.append(X, rows=rows, kind="tree")
    if J.enabled:
        names = names + ["extscore_p", "extscore_missing"]
    info["width"] = X.shape[1]
    info["feature_seconds"] = round(time.time() - t0, 2)
    yr = y[rows].astype(np.int64)
    print(f"{method}/{ds}: X {X.shape} ({info['feature_seconds']} s) features={names}")
    print(f"  constant on train: {info['constant_on_train']}  extscore={'on' if J.enabled else 'off'}")
    # One column-major float32 copy of the train rows, shared by every seed. sklearn's splitter
    # gathers X[samples[i], f] one feature at a time, so a contiguous column is cache-friendlier
    # than a strided row-major one (~2-5% per fit, measured); the values, hence the trees, are
    # identical (sklearn's own check_array keeps float32 F-order input without a copy).
    X_tr, y_tr = np.asfortranarray(X[pos["train"]]), yr[pos["train"]]
    release_memory()  # drop the pandas/arrow buffers of the feature build before the first forest

    results = []
    for seed in args.seeds:
        with Run(method, ds, config=vars(args), seed=seed, source_files=source_files,
                 data_files=data_files) as run:
            rf = RandomForestClassifier(n_estimators=args.n_estimators, n_jobs=args.n_jobs, random_state=seed)
            tf = time.time()
            rf.fit(X_tr, y_tr)
            fit_s = time.time() - tf
            metrics, tp = {}, time.time()
            for s in ("val", "test"):
                prob = rf.predict_proba(X[pos[s]])[:, 1]
                metrics[s] = compute_metrics(yr[pos[s]], prob, threshold=ARGMAX_THRESHOLD)
                run.add_predictions(s, splits[s], prob, yr[pos[s]])
            pred_s = time.time() - tp
            extra = {"feature_names": names, "features": info,
                     "mdi_importance": {k: float(v) for k, v in zip(names, rf.feature_importances_)},
                     "n_tree_nodes": int(sum(e.tree_.node_count for e in rf.estimators_)),
                     "fit_seconds": round(fit_s, 2), "predict_seconds": round(pred_s, 2),
                     "selection": {"select": "none", "note": "RF fit on train; val reported, not used"},
                     "threshold_rule": "argmax of predict_proba (ties -> human), = sklearn predict"}
            if J.enabled and args.perm_repeats > 0:
                Xv = np.array(X[pos["val"]])
                extra["extscore_permutation_importance"] = _perm_importance(
                    rf, Xv, yr[pos["val"]], [len(names) - 2, len(names) - 1], names, args.perm_repeats, seed)
                del Xv
            res = run.finish(metrics, extscore=J.report(), **extra)
            t = metrics["test"]
            print(f"  seed {seed}: test acc {t['accuracy']:.4f} f1 {t['f1']:.4f} P {t['precision']:.4f} "
                  f"R {t['recall']:.4f} auc {t['roc_auc']:.4f} | fit {fit_s:.1f}s nodes {extra['n_tree_nodes']} "
                  f"rss {res['memory'].get('peak_rss_bytes', 0) / 2**30:.2f} GiB")
            results.append(res)
            del rf
            release_memory()
    agg = aggregate(results)
    print({k: (round(v["mean"], 4), round(v["std"], 4)) for k, v in agg.items()})
    if J.enabled:
        b = J.baseline["test"]
        print(f"column-only baseline (val-tuned threshold {J.baseline['threshold']:.3f}): "
              f"test acc {b['accuracy']:.4f} f1 {b['f1']:.4f}")
    return results
