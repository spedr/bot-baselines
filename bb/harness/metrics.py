"""Evaluation metrics. Class 1 = bot everywhere; ``f1`` is BOT-class F1 (what the TwiBot papers
report), ``f1_human`` is given separately so the two are never mixed."""
from __future__ import annotations

import numpy as np


def compute_metrics(y_true, prob, threshold: float = 0.5, prob_for_auc=None) -> dict:
    """``prob`` = P(bot) [n] (or [n,2] class probabilities / logits-softmaxed). Hard predictions are
    ``prob >= threshold``. ROC-AUC is computed from probabilities (NaN if a class is absent)."""
    y = np.asarray(y_true).astype(np.int64)
    p = np.asarray(prob, dtype=np.float64)
    if p.ndim == 2:
        p = p[:, 1]
    if len(y) != len(p):
        raise ValueError("y_true and prob lengths differ")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("labels must be 0/1 (unlabelled rows must not be evaluated)")
    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n = len(y)

    def f1(tp_, fp_, fn_):
        d = 2 * tp_ + fp_ + fn_
        return 2 * tp_ / d if d else 0.0

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1_bot = f1(tp, fp, fn)
    f1_hum = f1(tn, fn, fp)
    mcc_d = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "n": n, "n_bot": int((y == 1).sum()), "threshold": threshold,
        "accuracy": (tp + tn) / n if n else float("nan"),
        "precision": prec, "recall": rec, "f1": f1_bot, "f1_human": f1_hum,
        "macro_f1": (f1_bot + f1_hum) / 2, "mcc": ((tp * tn - fp * fn) / mcc_d) if mcc_d else 0.0,
        "roc_auc": roc_auc(y, np.asarray(prob_for_auc if prob_for_auc is not None else p, dtype=np.float64)),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties (equals sklearn.roc_auc_score)."""
    y = np.asarray(y)
    npos = int((y == 1).sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s), np.float64)
    i = 0
    r = np.arange(1, len(s) + 1, dtype=np.float64)
    # average ranks over ties
    bounds = np.flatnonzero(np.diff(ss)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(s)]])
    avg = (r[starts] + r[ends - 1]) / 2
    ranks[order] = np.repeat(avg, ends - starts)
    del i
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def summarize(results: list[dict], keys=("accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc")) -> dict:
    """Mean and sample std (ddof=1) of metric dicts over seeds."""
    out = {}
    for k in keys:
        v = np.array([r[k] for r in results if k in r and r[k] is not None], dtype=np.float64)
        out[k] = {"mean": float(np.nanmean(v)) if len(v) else None,
                  "std": float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}
    return out
