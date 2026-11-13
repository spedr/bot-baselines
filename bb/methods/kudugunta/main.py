"""Kudugunta & Ferrara 2018, account-level model as re-implemented by the TwiBot-22 benchmark:
SMOTE -> Edited Nearest Neighbours -> AdaBoost(100 stumps) on 6/8 raw profile fields.

    uv run python -m bb.methods.kudugunta --dataset cresci15 --seeds 0 1 2 3 4
    uv run python -m bb.methods.kudugunta --dataset twibot22 --bench-misalign      # benchmark bug
    uv run python -m bb.methods.kudugunta --dataset twibot20 --with-extscore synthetic   # plumbing only

Seeds: SMOTE ``random_state = seed``, AdaBoost ``random_state = 100 * seed`` (benchmark 0..400),
ENN deterministic. Hard predictions = AdaBoost ``predict`` (decision > 0 <=> P(bot) > 0.5). Val is
never used for fitting or selection (as in the benchmark); its metrics are reported for reference.

ExtScore (kudugunta.md 8): ``J.append(X, kind="tree")`` appends ``[p or -1, m]`` BEFORE SMOTE, so the
column is resampled like any other feature; the pair (p, m) of a synthetic row is interpolated when
both parents agree on m and copied from the nearer parent when they do not, so m stays 0/1 and
p == -1 exactly where m == 1 (resample.py). ``--extscore-transform meanfill`` uses ``[p or train-mean, m]``
(the spec's option); ``--extscore-knn exclude`` computes SMOTE/ENN neighbours on the base columns only.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.special import expit

from bb.data import load_split, load_users
from bb.harness import Run, aggregate, compute_metrics
from bb.extscore import add_extscore_args, prepare_extscore
from bb.paths import dataset_paths

from .boost import AdaBoostStumps
from .features import build_features
from .resample import enn, smote

SOURCES = [Path(__file__).parent / f for f in ("main.py", "features.py", "resample.py", "boost.py")]


def variant_name(args) -> str:
    parts = ["bench"]
    if args.bench_misalign:
        parts.append("misalign")
    if args.tw20_literal_bool_strings:
        parts.append("tw20lit")
    if args.algorithm != "SAMME.R":
        parts.append(args.algorithm.lower().replace(".", ""))
    if getattr(args, "knn_tie_order", "legacy") != "legacy":
        parts.append("knninstalled")
    if getattr(args, "stumps", "fast") != "fast":
        parts.append("skstumps")
    if args.extscore != "none" or args.with_extscore:
        if args.extscore_transform != "tree":
            parts.append(f"extscore{args.extscore_transform}")
        if args.extscore_knn != "include":
            parts.append("extscoreknnexcl")
    return "-".join(parts)


def resample_and_fit(Xtr: np.ndarray, ytr: np.ndarray, seed: int, args, base_width: int, extscore_width: int):
    """SMOTE -> ENN -> AdaBoost on the training rows. Returns (model, info)."""
    t = {}
    pairs = [(base_width, base_width + 1)] if extscore_width else []
    dist_cols = list(range(base_width)) if (extscore_width and args.extscore_knn == "exclude") else None
    t0 = time.time()
    Xs, ys, si = smote(Xtr, ytr, k_neighbors=args.smote_k, random_state=seed, n_jobs=args.n_jobs,
                       dist_cols=dist_cols, binary_pairs=pairs, tie_order=args.knn_tie_order)
    t["smote_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    Xe, ye, ei = enn(Xs, ys, n_neighbors=args.enn_k, n_jobs=args.n_jobs, dist_cols=dist_cols,
                     tie_order=args.knn_tie_order)
    t["enn_s"] = round(time.time() - t0, 2)
    if pairs:  # invariant: indicator binary, sentinel/fill exactly where missing
        m, p = Xe[:, base_width + 1], Xe[:, base_width]
        assert np.isin(m, (0.0, 1.0)).all(), "ExtScore missing indicator not binary after resampling"
        fill = np.unique(Xtr[Xtr[:, base_width + 1] == 1, base_width])
        assert len(fill) <= 1 and (p[m == 1] == (fill[0] if len(fill) else -1.0)).all(), \
            "ExtScore value is not the sentinel/fill on every missing row after resampling"
    t0 = time.time()
    model = AdaBoostStumps(n_estimators=args.n_estimators, learning_rate=args.learning_rate,
                           algorithm=args.algorithm, random_state=100 * seed,
                           stumps=getattr(args, "stumps", "fast")).fit(Xe, ye)
    t["adaboost_s"] = round(time.time() - t0, 2)
    n_syn = len(Xs) - len(Xtr)
    kept = ei["kept"]
    info = {
        "n_train": int(len(Xtr)), "n_train_bot": int(ytr.sum()),
        "n_after_smote": int(len(Xs)), "n_synthetic": int(n_syn), "smote": si["per_class"],
        "n_after_enn": int(len(Xe)), "n_after_enn_bot": int(ye.sum()), "enn": ei["per_class"],
        "enn_kept_synthetic": int((kept >= len(Xtr)).sum()),
        "n_estimators_fitted": len(model.estimators_), "timing": t,
    }
    return model, info


def run_one(args, seed: int, X: np.ndarray, y: np.ndarray, splits: dict, J, base_width: int,
            data_files: list, feats: dict) -> dict:
    tr = splits["train"]
    with Run("kudugunta", args.dataset, config=vars(args), seed=seed, data_files=data_files,
             source_files=SOURCES) as run:
        model, info = resample_and_fit(X[tr], y[tr], seed, args, base_width, J.width)
        t0 = time.time()
        res, ties, flips = {}, 0, 0
        for s in ("train", "val", "test"):
            ix = splits[s]
            d = model.decision_function(X[ix])
            prob = expit(d)                   # = AdaBoost predict_proba[:, 1]
            ties += int((d == 0).sum())       # predict() says human at d == 0, prob >= 0.5 says bot
            hard = (d > 0).astype(np.float64)  # = AdaBoost predict (the benchmark's hard labels)
            flips += int(((prob >= 0.5) != (hard == 1)).sum())  # |d| < ~1e-16: expit(d) rounds to 0.5
            res[s] = compute_metrics(y[ix], hard, prob_for_auc=prob)
            if s != "train":
                run.add_predictions(s, ix, prob, y[ix])
        info["timing"]["predict_s"] = round(time.time() - t0, 2)
        info["decision_ties"] = ties
        info["prob_threshold_flips"] = flips  # rows where prob >= 0.5 would disagree with predict()
        sel = {"select": "none", "note": "benchmark protocol: no model selection; val unused"}
        return run.finish(res, selection=sel, extscore=J.report(), resampling=info,
                          features=feats)


def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=["twibot20", "twibot22", "cresci15"], default="cresci15")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cpu", help="unused (CPU-only method); accepted for a uniform CLI")
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--learning-rate", type=float, default=1.0)
    ap.add_argument("--algorithm", choices=["SAMME.R", "SAMME"], default="SAMME.R",
                    help="SAMME.R = the benchmark's 2022 sklearn default (ported; removed from sklearn 1.6)")
    ap.add_argument("--smote-k", type=int, default=5)
    ap.add_argument("--enn-k", type=int, default=3)
    ap.add_argument("--n-jobs", type=int, default=8, help="threads for the kNN queries (results do not depend on it)")
    ap.add_argument("--knn-tie-order", choices=["legacy", "installed"], default="legacy",
                    help="order of equal-distance neighbours in SMOTE/ENN: scikit-learn <= 1.5 (= imblearn 0.12.4 "
                         "on sklearn 1.5.2, the benchmark era; default) or the installed sklearn's (>= 1.6 differs; "
                         "the pre-review-1 behaviour)")
    ap.add_argument("--stumps", choices=["fast", "sklearn"], default="fast",
                    help="weak learner: presorted numba stump search (default; same splits as sklearn's "
                         "DecisionTreeClassifier(max_depth=1), review round 2) or the sklearn tree itself (verification)")
    ap.add_argument("--bench-misalign", action="store_true",
                    help="emulate the benchmark's boolean-feature/user misalignment (TwiBot-22)")
    ap.add_argument("--tw20-literal-bool-strings", action="store_true",
                    help="emulate 'False ' != 'False' on TwiBot-20 (protected/verified = 1 for every profile)")
    ap.add_argument("--extscore-transform", choices=["tree", "meanfill"], default="tree")
    ap.add_argument("--extscore-knn", choices=["include", "exclude"], default="include",
                    help="include the ExtScore columns in SMOTE/ENN distances (default) or not")
    add_extscore_args(ap)
    args = ap.parse_args(argv)
    args.variant = variant_name(args)
    ds = args.dataset
    y = load_users(ds, columns=["label"])["label"].to_numpy().astype(np.int64)
    splits = {s: np.asarray(load_split(ds, s)) for s in ("train", "val", "test")}
    X, names, finfo = build_features(ds, bench_misalign=args.bench_misalign,
                                     tw20_literal_bool_strings=args.tw20_literal_bool_strings)
    base_width = X.shape[1]
    J = prepare_extscore(args, ds, splits, labels=y)
    kind = "tree" if args.extscore_transform == "tree" else "meanfill"
    X = J.append(X, kind=kind)
    if J.enabled:
        names = names + [f"extscore_p_{kind}", "extscore_missing"]
    feats = {"names": names, "info": finfo, "width": int(X.shape[1])}
    dp = dataset_paths(ds)
    data_files = [dp.profile] + ([dp.extscore_table(J.tag)] if J.enabled else [])
    print(f"{ds}: X {X.shape} variant={args.variant} features={names} extscore={'on' if J.enabled else 'off'}", flush=True)
    out = []
    for s in args.seeds:
        r = run_one(args, s, X, y, splits, J, base_width, data_files, feats)
        m, rs = r["metrics"]["test"], r["resampling"]
        print(f"seed {s}: test acc {m['accuracy']:.4f} P {m['precision']:.4f} R {m['recall']:.4f} "
              f"F1 {m['f1']:.4f} AUC {m['roc_auc']:.4f} | train {rs['n_train']} -> smote {rs['n_after_smote']} "
              f"-> enn {rs['n_after_enn']} | {rs['timing']} | {r['wall_seconds']} s "
              f"peakRSS {r['memory'].get('peak_rss_bytes', 0) / 2**30:.2f} GB", flush=True)
        out.append(r)
    agg = aggregate(out)
    print({k: (round(v["mean"], 4), round(v["std"], 4)) for k, v in agg.items()
           if k in ("accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc")})
    if J.enabled and J.baseline:
        b = J.baseline["test"]
        print(f"column-only baseline (val-tuned threshold {J.baseline['threshold']:.3f}): "
              f"test acc {b['accuracy']:.4f} f1 {b['f1']:.4f}")
    return out


if __name__ == "__main__":
    main()
