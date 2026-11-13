"""Fast stumps vs the scikit-learn stumps on real data (review round 2; not a pytest file, loads
real artifacts):

    tools/memrun --light -- uv run python tests/methods/kudugunta_fast_check.py cresci15 0,1,2,3,4
    tools/memrun --light -- uv run python tests/methods/kudugunta_fast_check.py twibot20 0,1,2,3,4 tw20lit
    tools/memrun --light -- uv run python tests/methods/kudugunta_fast_check.py twibot22 0 [misalign] [extscore:synthetic]

For each seed: SMOTE -> ENN exactly as ``main.resample_and_fit``, then ``AdaBoostStumps`` with
``stumps='sklearn'`` (the library-backed path that produced the reported numbers) and
``stumps='fast'`` (numba emulation of the tree), for SAMME.R and SAMME. Compares, round by round, the chosen
feature and threshold (must be identical), the leaf values and estimator weights (max abs
difference), and the decision values on every user of the dataset (train, val and test; max abs
difference, values not bit-identical, flipped ``d > 0`` predictions, smallest |d|). The fast path
emulates the tree's sample order exactly, so everything must be bit-identical; for diagnosis, a round
with a different (feature, threshold) is classified by whether both stumps route every training row
and every user the same way (exact ties between features inducing the same partition). Prints
per-stage timings. Exit status 1 unless bit-identical.
"""
import argparse
import sys
import time

import numpy as np

from bb.data import load_split, load_users
from bb.extscore import prepare_extscore
from bb.methods.kudugunta.boost import AdaBoostStumps
from bb.methods.kudugunta.features import build_features
from bb.methods.kudugunta.resample import enn, smote


def main():
    ds, seeds = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
    opts = sys.argv[3:]
    kw = {"bench_misalign": "misalign" in opts, "tw20_literal_bool_strings": "tw20lit" in opts}
    extscore = next((o.split(":", 1)[1] for o in opts if o.startswith("extscore:")), None)
    algs = [a for a in ("SAMME.R", "SAMME") if f"only:{a}" in opts] or ["SAMME.R", "SAMME"]
    y = load_users(ds, columns=["label"])["label"].to_numpy().astype(np.int64)
    splits = {s: np.asarray(load_split(ds, s)) for s in ("train", "val", "test")}
    X, names, _ = build_features(ds, **kw)
    bw = X.shape[1]
    ns = argparse.Namespace(extscore="p" if extscore else "none", extscore_tag=extscore, with_extscore=extscore, extscore_unsafe=False)
    J = prepare_extscore(ns, ds, splits, labels=y)
    X = J.append(X, kind="tree")
    pairs = [(bw, bw + 1)] if J.enabled else []
    tr = splits["train"]
    ev = np.concatenate([splits[s] for s in ("train", "val", "test")])
    ok = True
    print(ds, opts, "X", X.shape, flush=True)
    for s in seeds:
        t0 = time.time()
        Xs, ys, _ = smote(X[tr], y[tr], random_state=s, binary_pairs=pairs)
        t1 = time.time()
        Xe, ye, _ = enn(Xs, ys)
        t2 = time.time()
        print(f"seed {s}: {len(tr)} -> {len(Xs)} -> {len(Xe)} rows; smote {t1 - t0:.2f} s enn {t2 - t1:.2f} s", flush=True)
        for alg in algs:
            m = {}
            tf, tp = {}, {}
            for path in ("fast", "sklearn"):
                a = time.time()
                m[path] = AdaBoostStumps(100, algorithm=alg, random_state=100 * s, stumps=path).fit(Xe, ye)
                b = time.time()
                d = m[path].decision_function(X[ev])
                tf[path], tp[path] = b - a, time.time() - b
                m[path].d = d
            from bb.methods.kudugunta.boost import stump_from_tree
            F, S = m["fast"].estimators_, [stump_from_tree(e) for e in m["sklearn"].estimators_]
            diff = [i for i, (a, b) in enumerate(zip(F, S)) if (a.feature, a.threshold) != (b.feature, b.threshold)]
            nsplit, first = len(diff), (diff[0] if diff else None)
            # a differing split is harmless iff it routes every training row AND every user the same
            # way (up to swapping left/right), with the same leaf values on each side
            Xe32, Xu32 = Xe.astype(np.float32), X[ev].astype(np.float32)
            bad_part, users_routed = 0, 0
            for i in diff:
                a, b = F[i], S[i]
                la, lb = Xe32[:, a.feature] <= a.threshold, Xe32[:, b.feature] <= b.threshold
                same, flip = np.array_equal(la, lb), np.array_equal(la, ~lb)
                va = a.value if same else a.value[::-1]
                bad_part += not ((same or flip) and np.allclose(va, b.value, rtol=0, atol=1e-12))
                ua, ub = Xu32[:, a.feature] <= a.threshold, Xu32[:, b.feature] <= b.threshold
                users_routed = max(users_routed, int((ua != (ub if same else ~ub)).sum()))
            vd = max([float(np.abs(a.value - b.value).max()) for i, (a, b) in enumerate(zip(F, S)) if i not in diff] or [0.0])
            wd = float(np.abs(m["fast"].estimator_weights_ - m["sklearn"].estimator_weights_).max())
            df, dsk = m["fast"].d, m["sklearn"].d
            flips = int(((df > 0) != (dsk > 0)).sum())
            dd = float(np.abs(df - dsk).max())
            nbit = int((df != dsk).sum())
            good = len(F) == len(S) and nsplit == 0 and vd == 0.0 and wd == 0.0 and nbit == 0
            ok &= good
            print(f"  {alg}: stumps {len(F)}/{len(S)} differing (feature, threshold) {nsplit} (first {first}; "
                  f"different train partition {bad_part}, users routed differently {users_routed}) "
                  f"max|leaf value diff| {vd:.1e} max|weight diff| {wd:.1e}; decision on {len(ev)} users: "
                  f"max|dd| {dd:.1e}, not bit-identical {nbit}, flips {flips}, min|d| {np.abs(dsk).min():.1e}; "
                  f"fit fast {tf['fast']:.2f} s sklearn {tf['sklearn']:.2f} s; decision fast {tp['fast']:.2f} s "
                  f"sklearn {tp['sklearn']:.2f} s  {'OK' if good else 'MISMATCH'}", flush=True)
    print("ALL OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
