"""Lee, Eoff & Caverlee 2011 ("Seven Months with the Devils"), TwiBot-22 benchmark version:
random forest (76 trees) on 8 profile + 6 tweet-content features (docs/methods/lee2011.md).

    tools/memrun --light -- uv run python -m bb.methods.lee2011 --dataset twibot20
    tools/memrun --light -- uv run python -m bb.methods.lee2011 --dataset cresci15 --with-extscore synthetic

Faithful defaults: display-name length as column 0, ZIP overhead 45 B, zero-tweet users dropped
from every split, TwiBot-22 first 20 stored tweets (release order), 200 elsewhere, fixed collection
dates, sklearn defaults except n_estimators=76, hard predictions (argmax: a 0.5 vote is human).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ...data import load_split, load_users, profile_meta
from ...harness import Run, aggregate, compute_metrics
from ...extscore import add_extscore_args, nan_transform, prepare_extscore
from . import features as F

HERE = Path(__file__).parent
# sklearn predict = argmax of the averaged class probabilities: ties (0.5) go to class 0 (human)
THRESHOLD = float(np.nextafter(0.5, 1.0))


def build_matrix(args, log=print) -> tuple[np.ndarray, list[str], np.ndarray, dict, dict]:
    """X [N, F] float32 (all users; unlabelled rows unused), names, has_tweets [N] bool,
    content-cache meta, info on dropped features."""
    ds = args.dataset
    users = load_users(ds, columns=["label"])
    lab = np.flatnonzero(users["label"].to_numpy() >= 0)
    del users
    # content pass first: the worker pool never overlaps with the pandas peak of the profile load
    raw, cmeta, cpath = F.content_stats(ds, lab, args.tweets_per_user, args.tweet_order,
                                        args.tweet_seed, n_workers=args.workers,
                                        tweet_budget=args.tweet_budget, recompute=args.recompute,
                                        log=log)
    Xp = F.profile_features(ds, col0=args.col0, reference_date=args.reference_date)
    Xc = F.content_columns(raw, args.zip_overhead)
    has = np.asarray(raw[:, 0]) > 0
    X = np.concatenate([Xp, Xc], 1)
    names = list(F.FEATURE_NAMES)
    # drop features whose profile fields the dataset lacks (none of the 14 in our three datasets)
    absent = set(profile_meta(ds).get("absent_columns", []))
    if args.col0 == "screen_name":
        names[0] = "screen_name_length(screen_name)"
    fields = {**F.FEATURE_FIELDS, "screen_name_length": (args.col0,)}
    drop = [i for i, n in enumerate(F.FEATURE_NAMES) if n in fields and absent & set(fields[n])]
    keep = [i for i in range(len(names)) if i not in drop]
    info = {"dropped_features": [names[i] for i in drop], "absent_profile_columns": sorted(absent),
            "content_cache": str(cpath)}
    if args.zero_tweet_policy == "fill":
        X[:, len(F.PROFILE_NAMES):] = np.where(np.isnan(X[:, len(F.PROFILE_NAMES):]), -1.0,
                                               X[:, len(F.PROFILE_NAMES):])
    return X[:, keep].astype(np.float32), [names[i] for i in keep], has, cmeta, info


def permutation_drop(rf, X, y, col: int, seed: int, repeats: int = 3) -> dict:
    """Drop in val accuracy / bot-F1 when column ``col`` is permuted (mean over repeats)."""
    base = compute_metrics(y, rf.predict_proba(X)[:, 1], threshold=THRESHOLD)
    rng = np.random.default_rng(seed)
    acc, f1 = [], []
    Xp = X.copy()
    for _ in range(repeats):
        Xp[:, col] = X[rng.permutation(len(X)), col]
        m = compute_metrics(y, rf.predict_proba(Xp)[:, 1], threshold=THRESHOLD)
        acc.append(base["accuracy"] - m["accuracy"])
        f1.append(base["f1"] - m["f1"])
    return {"accuracy_drop": float(np.mean(acc)), "f1_drop": float(np.mean(f1)), "repeats": repeats}


def run_seed(args, seed: int, X: np.ndarray, names: list[str], y: np.ndarray, sp: dict, J,
             extra: dict, data_files: list) -> dict:
    from sklearn.ensemble import RandomForestClassifier

    tr, va, te = sp["train"], sp["val"], sp["test"]
    late = J.enabled and args.extscore_fusion == "late"
    Xa, feat_names = X, list(names)
    if J.enabled and not late:
        if args.extscore_transform == "tree":
            Xa = J.append(X, kind="tree")  # X has all N rows: [p or -1, m] appended after column 13
            feat_names += ["extscore_p", "extscore_missing"]
        else:
            Xa = np.concatenate([X, nan_transform(J.column).astype(X.dtype)], 1)
            feat_names += ["extscore_p"]
    mf = int(args.max_features) if str(args.max_features).isdigit() else args.max_features
    rf = RandomForestClassifier(n_estimators=args.n_estimators, max_features=mf, random_state=seed,
                                n_jobs=args.n_jobs)
    with Run("lee2011", args.dataset, config=vars(args), seed=seed, source_files=[HERE],
             data_files=data_files) as run:
        t0 = time.time()
        rf.fit(Xa[tr], y[tr])
        t_fit = time.time() - t0
        prob = {s: rf.predict_proba(Xa[sp[s]])[:, 1] for s in ("val", "test")}
        n_nodes = int(sum(e.tree_.node_count for e in rf.estimators_))
        res = {s: compute_metrics(y[sp[s]], prob[s], threshold=THRESHOLD) for s in ("val", "test")}
        extscore_extra = {}
        if late:
            from sklearn.linear_model import LogisticRegression

            def zf(p):
                p = np.clip(p, 1e-4, 1 - 1e-4)
                return np.log(p / (1 - p))
            Zv = np.column_stack([zf(prob["val"]), J.neural[va]])
            Zt = np.column_stack([zf(prob["test"]), J.neural[te]])
            lr = LogisticRegression().fit(Zv, y[va])
            pt = lr.predict_proba(Zt)[:, 1]
            extscore_extra["late_fusion"] = {"rf_only_test": res["test"], "coef": lr.coef_.ravel().tolist(),
                                        "intercept": float(lr.intercept_[0]),
                                        "val_in_sample": compute_metrics(y[va], lr.predict_proba(Zv)[:, 1])}
            prob["test"] = pt
            res["test"] = compute_metrics(y[te], pt)  # threshold 0.5 on the LR probability
        elif J.enabled:
            jcol = feat_names.index("extscore_p")
            extscore_extra["importance"] = {
                "mdi": dict(zip(feat_names, rf.feature_importances_.round(5).tolist())),
                "permutation_val_extscore_p": permutation_drop(rf, Xa[va], y[va], jcol, seed)}
        imp = dict(zip(feat_names, rf.feature_importances_.round(5).tolist()))
        for s in ("val", "test"):
            run.add_predictions(s, sp[s], prob[s], y[sp[s]])
        out = run.finish(res, extscore={**J.report(), **extscore_extra}, features=feat_names,
                         feature_importances=imp, fit_seconds=round(t_fit, 2), n_tree_nodes=n_nodes,
                         max_features_effective=rf.estimators_[0].max_features_, **extra)
    del rf
    return out


def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=["twibot20", "twibot22", "cresci15"], default="twibot20")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cpu", help="unused (CPU-only method); kept for the common CLI")
    ap.add_argument("--variant", default=None, help="run-name prefix (default: derived from flags)")
    g = ap.add_argument_group("Lee 2011 / benchmark options (spec sections 3, 4, 8, 9)")
    g.add_argument("--tweets-per-user", type=int, default=None, help="default 20 on TwiBot-22, 200 otherwise")
    g.add_argument("--tweet-order", choices=["release", "newest", "random"], default="release")
    g.add_argument("--tweet-seed", type=int, default=0, help="seed of --tweet-order random")
    g.add_argument("--zero-tweet-policy", choices=["drop", "fill"], default="drop")
    g.add_argument("--col0", choices=["name", "screen_name"], default="name",
                   help="column 0: display name (benchmark) or screen name (paper)")
    g.add_argument("--zip-overhead", type=int, default=F.ZIP_OVERHEAD, help="45 = faithful, 0 = pure ratio")
    g.add_argument("--reference-date", default=None, help="default 2022-04-20 (TwiBot-22) / 2020-10-01")
    g.add_argument("--n-estimators", type=int, default=76)
    g.add_argument("--max-features", default="sqrt", help="sqrt (default) or an int, e.g. 3 with ExtScore")
    g.add_argument("--n-jobs", type=int, default=8)
    g.add_argument("--extscore-transform", choices=["tree", "nan"], default="tree",
                   help="tree: [p or -1, missing]; nan: p with NaN (sklearn NaN-native)")
    g.add_argument("--extscore-fusion", choices=["early", "late"], default="early",
                   help="late: LR on [logit(rf), z, m] fitted on val (RF on base features)")
    g = ap.add_argument_group("content pass")
    g.add_argument("--workers", type=int, default=6,
                   help="content-pass processes (~0.25 GB anon each + 0.41 GB shared vector pages; "
                        "6 fit the 2.5 GB light cap: 1.75 GB anon peak on TwiBot-22)")
    g.add_argument("--tweet-budget", type=int, default=20_000, help="selected tweets per worker task")
    g.add_argument("--recompute", action="store_true")
    g.add_argument("--features-only", action="store_true", help="build/cache the features and exit")
    add_extscore_args(ap)
    args = ap.parse_args(argv)
    if args.tweets_per_user is None:
        args.tweets_per_user = F.DEFAULT_CAP.get(args.dataset, F.DEFAULT_CAP_OTHER)
    if args.variant is None:
        parts = ["default" if (args.zero_tweet_policy, args.tweet_order, args.col0, args.zip_overhead)
                 == ("drop", "release", "name", F.ZIP_OVERHEAD) else "v"]
        if args.zero_tweet_policy != "drop":
            parts.append(args.zero_tweet_policy)
        if args.tweet_order != "release":
            parts.append(args.tweet_order)
        if args.col0 != "name":
            parts.append("col0sn")
        if args.zip_overhead != F.ZIP_OVERHEAD:
            parts.append(f"zip{args.zip_overhead}")
        if args.max_features != "sqrt":
            parts.append(f"mf{args.max_features}")
        if args.extscore_fusion == "late" or args.extscore_transform == "nan":
            parts.append(f"extscore-{args.extscore_fusion}-{args.extscore_transform}")
        args.variant = "-".join(parts)
    ds = args.dataset
    y = load_users(ds, columns=["label"])["label"].to_numpy()
    full = {s: np.asarray(load_split(ds, s)) for s in ("train", "val", "test")}
    t0 = time.time()
    X, names, has, cmeta, info = build_matrix(args)
    t_feat = time.time() - t0
    if args.zero_tweet_policy == "drop":
        sp = {s: v[has[v]] for s, v in full.items()}
    else:
        sp = dict(full)
    counts = {s: {"n_official": int(len(full[s])), "n_used": int(len(sp[s])),
                  "n_dropped_zero_tweets": int((~has[full[s]]).sum())} for s in full}
    print(f"{ds}: X {X.shape} features={names}; cap={args.tweets_per_user} order={args.tweet_order} "
          f"policy={args.zero_tweet_policy}; splits {counts}; dropped features {info['dropped_features']}")
    if args.features_only:
        return []
    J = prepare_extscore(args, ds, sp, labels=y)
    if J.enabled:
        print(f"extscore {J.tag}: coverage {J.cov}")
    extra = {"split_counts": counts, "feature_info": info, "feature_seconds": round(t_feat, 1),
             "content_pass": {k: cmeta.get(k) for k in ("key", "n_tweets", "wall_seconds", "tweets_per_s",
                                                         "peak_rss_parent_bytes", "peak_rss_children_bytes",
                                                         "cgroup", "n_workers")},
             "threshold_note": "hard predict: argmax of predict_proba, a 0.5 vote is human"}
    data_files = [info["content_cache"], F.cache_paths(ds, cmeta["key"])[1]]
    out = [run_seed(args, s, X, names, y, sp, J, extra, data_files) for s in args.seeds]
    agg = aggregate(out)
    print({k: (round(v["mean"], 4), round(v["std"], 4)) for k, v in agg.items()})
    if J.enabled and J.baseline:
        b = J.baseline["test"]
        print(f"column-only baseline (val-tuned threshold {J.baseline['threshold']:.3f}): "
              f"test acc {b['accuracy']:.4f} f1 {b['f1']:.4f}")
    return out
