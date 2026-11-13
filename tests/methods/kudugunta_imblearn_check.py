"""Cross-check of the Kudugunta re-implementations against the benchmark's libraries (not a pytest
file: it needs imbalanced-learn 0.12 and scikit-learn 1.5.2, which still has SAMME.R, so it runs in
a throw-away uv environment that does not touch the project's dependencies):

    uv run --isolated --no-project -p 3.12 --with imbalanced-learn==0.12.4 --with scikit-learn==1.5.2 \
        --with "numpy<2.3" python -I tests/methods/kudugunta_imblearn_check.py bb/methods/kudugunta [X.npz ...]

``X.npz`` (optional): ``Xtr`` [n,F], ``ytr`` [n], ``Xte`` [m,F] real feature matrices, e.g. dumped from
``bb.methods.kudugunta.features.build_features`` under memrun. Checks, per data set and seed:
1. ``resample.smote`` == ``imblearn SMOTE(random_state=seed).fit_resample`` (bit-identical);
2. ``resample.enn`` == ``EditedNearestNeighbours().fit_resample`` (rows and ``sample_indices_``);
3. ``boost.AdaBoostStumps`` == ``AdaBoostClassifier(n_estimators=100, random_state=100*seed,
   algorithm=...)`` for SAMME.R and SAMME (estimator weights, decision function, predictions).
Prints ALL OK or MISMATCH. Measured 2026-11-12 (after review round 1): ALL OK on synthetic data and
the Cresci-2015, TwiBot-20 (parsed, literal strings) and TwiBot-22 train features (max |decision
difference| 0.0); cross-version: 0 differing SMOTE rows / ENN rows / test predictions.

These three checks import the port INTO the isolated env, so the port runs on scikit-learn 1.5.2 there.
The reported numbers were produced in the project env (scikit-learn 1.9.x), whose kd-tree orders
equal-distance neighbours differently and whose tree leaf probabilities differ in the last ulp.
If ``X.npz`` also holds ``port_s<seed>_*`` arrays written by ``kudugunta_dump_check.py`` (project
env), a 4th check compares the real libraries with those project-env outputs ("cross-version"):
SMOTE rows, ENN kept rows, test predictions of the full library pipeline vs the port, and of
scikit-learn 1.5.2 AdaBoost refitted on the port's own resampled rows (isolates the tree from the
kNN). It reports differences instead of failing on them (review round 1: tie order is not a
semantic difference; docs/results/kudugunta.md).
"""
import importlib.util
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
SK = {}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def synth(rng, n=6000, f=8, pos=0.1):
    y = (rng.random(n) < pos).astype(np.int64)
    X = np.floor(np.exp(rng.normal(3 + y[:, None] * 0.7, 2, (n, 4))))
    B = (rng.random((n, f - 4)) < 0.2 + 0.3 * y[:, None]).astype(float)
    X = np.hstack([X, B])
    X[:50] = X[50:100]  # duplicates
    return X, y, X[:1500]


def main():
    global SK
    R = load("resample", sys.argv[1] + "/resample.py")
    B = load("boost", sys.argv[1] + "/boost.py")
    # the library-backed weak learner (review round 2 added stumps='fast', checked against it in the
    # project env by kudugunta_fast_check.py; without numba here it would run in pure Python)
    SK = {"stumps": "sklearn"} if "stumps" in B.AdaBoostStumps.__init__.__code__.co_varnames else {}
    import imblearn
    import sklearn
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import EditedNearestNeighbours
    from sklearn.ensemble import AdaBoostClassifier
    print("sklearn", sklearn.__version__, "imblearn", imblearn.__version__)
    sets = {"synthetic": synth(np.random.default_rng(0))}
    for f in sys.argv[2:]:
        d = np.load(f)
        sets[f] = (d["Xtr"], d["ytr"], d["Xte"])
    ok = True
    for name, (X, y, Xte) in sets.items():
        for seed in (0, 3):
            aX, ay = SMOTE(random_state=seed).fit_resample(X, y)
            bX, by, _ = R.smote(X, y, random_state=seed, n_jobs=4)
            s_eq = np.array_equal(aX, bX) and np.array_equal(ay, by)
            e = EditedNearestNeighbours()
            a2X, a2y = e.fit_resample(aX, ay)
            b2X, b2y, info = R.enn(bX, by, n_jobs=4)
            e_eq = np.array_equal(a2X, b2X) and np.array_equal(a2y, b2y) and np.array_equal(e.sample_indices_, info["kept"])
            res = [name, seed, "smote", s_eq, "enn", e_eq, len(aX), len(a2X)]
            for alg in ("SAMME.R", "SAMME"):
                ref = AdaBoostClassifier(n_estimators=100, random_state=100 * seed, algorithm=alg).fit(a2X, a2y)
                mine = B.AdaBoostStumps(100, algorithm=alg, random_state=100 * seed, **SK).fit(b2X, b2y)
                dd = float(np.abs(ref.decision_function(Xte) - mine.decision_function(Xte)).max())
                pe = np.array_equal(ref.predict(Xte), mine.predict(Xte))
                we = bool(np.allclose(ref.estimator_weights_, mine.estimator_weights_))
                res += [alg, "pred_eq", pe, "max|dd|", dd, "w_eq", we]
                ok &= pe and we and dd < 1e-9
            ok &= s_eq and e_eq
            print(*res)
    for f in sys.argv[2:]:
        cross_version(f)
    print("ALL OK" if ok else "MISMATCH")


def _acc_f1(y, d):
    p = d > 0
    tp, fp, fn = (p & (y == 1)).sum(), (p & (y == 0)).sum(), (~p & (y == 1)).sum()
    return round(100 * float((p == y).mean()), 3), round(100 * 2 * tp / max(2 * tp + fp + fn, 1), 3)


def cross_version(f):
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import EditedNearestNeighbours
    from sklearn.ensemble import AdaBoostClassifier
    d = np.load(f)
    seeds = sorted({int(k.split("_")[1][1:]) for k in d.files if k.startswith("port_s")})
    if not seeds or "yte" not in d.files:
        return
    X, y, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    for s in seeds:
        aX, ay = SMOTE(random_state=s).fit_resample(X, y)
        pX, py, kept = d[f"port_s{s}_smote_X"], d[f"port_s{s}_smote_y"], d[f"port_s{s}_enn_kept"]
        rows = int((aX != pX).any(1).sum()) if aX.shape == pX.shape else -1
        e = EditedNearestNeighbours()
        a2X, a2y = e.fit_resample(aX, ay)
        print(f"cross-version {f} seed {s}: SMOTE rows differing {rows}/{len(aX) - len(X)} synthetic; ENN kept "
              f"lib {len(e.sample_indices_)} port {len(kept)} sym.diff {len(np.setxor1d(e.sample_indices_, kept))}")
        for alg in ("SAMME.R", "SAMME"):
            ref = AdaBoostClassifier(n_estimators=100, random_state=100 * s, algorithm=alg).fit(a2X, a2y)
            same = AdaBoostClassifier(n_estimators=100, random_state=100 * s, algorithm=alg).fit(pX[kept], py[kept])
            dl, ds_, dp = ref.decision_function(Xte), same.decision_function(Xte), d[f"port_s{s}_{alg}_dec"]
            print(f"   {alg}: test acc/F1 lib {_acc_f1(yte, dl)} port {_acc_f1(yte, dp)}; prediction flips "
                  f"full pipeline {int(((dl > 0) != (dp > 0)).sum())}/{len(yte)}, same-input AdaBoost "
                  f"{int(((ds_ > 0) != (dp > 0)).sum())} (max|dd| {np.abs(ds_ - dp).max():.1e})", flush=True)


if __name__ == "__main__":
    main()
