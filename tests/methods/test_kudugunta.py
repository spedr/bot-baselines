"""Kudugunta (benchmark account-level model): SMOTE/ENN semantics, AdaBoost port, features and the
misalignment emulation, ExtScore plumbing, no-leak."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from bb.methods.kudugunta.boost import AdaBoostStumps
from bb.methods.kudugunta.resample import enn, smote


def _data(n=3000, pos=0.12, seed=0, f=6):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos).astype(np.int64)
    X = np.floor(np.exp(rng.normal(3 + 0.8 * y[:, None], 2, (n, 4))))
    B = (rng.random((n, f - 4)) < 0.2 + 0.4 * y[:, None]).astype(float)
    X = np.hstack([X, B])
    X[:30] = X[30:60]  # duplicates
    return X, y


# ------------------------------------------------------------------------------------- SMOTE / ENN

def test_smote_balances_and_interpolates():
    X, y = _data()
    Xs, ys, info = smote(X, y, random_state=1, n_jobs=1)
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    assert (ys == 0).sum() == (ys == 1).sum() == n0
    assert np.array_equal(Xs[:len(X)], X) and np.array_equal(ys[:len(X)], y)   # originals first, unchanged
    new = Xs[len(X):]
    par = info["parents"]
    assert len(new) == n0 - n1 and (y[par] == 1).all()
    a, b = X[par[:, 0]], X[par[:, 1]]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    assert ((new >= lo - 1e-9) & (new <= hi + 1e-9)).all()                    # on the segment
    Xs2, _, _ = smote(X, y, random_state=1, n_jobs=4)
    assert np.array_equal(Xs, Xs2)                                            # seeded, n_jobs-invariant


def test_enn_edits_only_class1_after_balanced_smote():
    X, y = _data(seed=2)
    Xs, ys, _ = smote(X, y, random_state=0, n_jobs=1)
    Xe, ye, info = enn(Xs, ys, n_jobs=1)
    assert info["per_class"][0]["edited"] is False and info["per_class"][1]["edited"] is True
    assert (ye == 0).sum() == (ys == 0).sum()                                  # imblearn tie rule: class 0 kept
    from sklearn.neighbors import NearestNeighbors

    from bb.methods.kudugunta.resample import knn_query
    nn = NearestNeighbors(n_neighbors=4).fit(Xs)
    for order in ("legacy", "installed"):
        Xe, ye, info = enn(Xs, ys, n_jobs=1, tie_order=order)
        kept = info["kept"]
        nb = knn_query(nn, Xs[ys == 1], 1, order)[:, 1:]
        if order == "installed":
            assert np.array_equal(nb, nn.kneighbors(Xs[ys == 1], return_distance=False)[:, 1:])
        expect = np.flatnonzero(ys == 1)[(ys[nb] == 1).all(1)]
        assert np.array_equal(kept[(ys[kept] == 1)], expect)
        assert np.array_equal(Xe, Xs[kept])


def test_matches_imblearn():
    imb = pytest.importorskip("imblearn")  # noqa: F841  (optional; cross-checked with 0.12.4 in docs/results)
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import EditedNearestNeighbours
    X, y = _data(seed=3)
    a, ay = SMOTE(random_state=5).fit_resample(X, y)
    b, by, _ = smote(X, y, random_state=5)
    assert np.array_equal(a, b) and np.array_equal(ay, by)
    a2, ay2 = EditedNearestNeighbours().fit_resample(a, ay)
    b2, by2, _ = enn(b, by)
    assert np.array_equal(a2, b2) and np.array_equal(ay2, by2)


def test_smote_enn_golden_imblearn_0_12_4():
    """Values produced by imblearn 0.12.4 ``SMOTE(random_state=2)`` -> ``EditedNearestNeighbours()``
    under scikit-learn 1.5.2 on this fixture. Pins the RNG call order (randint, then uniform), the
    interpolation and ENN's class selection without imblearn installed, AND the kd-tree tie order:
    the fixture has duplicate rows, and with the installed scikit-learn's own neighbour order
    (>= 1.6 sorts equal distances differently) ENN keeps 4609 rows instead of 4612."""
    X, y = _data(seed=7)
    for nj in (1, 4):
        Xs, ys, _ = smote(X, y, random_state=2, n_jobs=nj)
        assert len(Xs) == 5294 and (ys[len(X):] == 1).all()
        assert Xs[len(X):].sum() == pytest.approx(2504613.6980958283, rel=1e-14)
        Xe, ye, info = enn(Xs, ys, n_jobs=nj)
        assert len(Xe) == 4612 and int(info["kept"].sum()) == 11418041


def test_legacy_tie_order():
    """kNN rows come back sorted by distance; equal distances follow the scikit-learn <= 1.5 sort
    (checked bit for bit against 1.5.2 in kudugunta_imblearn_check.py); the pure-Python and numba
    versions of the sort agree."""
    from sklearn.neighbors import NearestNeighbors

    import bb.methods.kudugunta.resample as R
    X, _ = _data(seed=8, n=2000)
    X = np.vstack([X, X[:200], X[:50]])                                   # many exact duplicates
    nb = R.knn_indices(X, X, 6, n_jobs=1)
    ref = NearestNeighbors(n_neighbors=6).fit(X).kneighbors(X)
    d = np.sqrt(((X[:, None, :] - X[nb]) ** 2).sum(-1))
    assert np.allclose(d, ref[0]) and (np.diff(d, axis=1) >= 0).all()   # same neighbour distances, sorted
    assert all(set(a) == set(b) for a, b in zip(nb, ref[1]))             # same neighbour sets
    rd = ((X[:, None, :] - X[nb]) ** 2).sum(-1)
    rng = np.random.default_rng(0)
    perm = np.stack([rng.permutation(6) for _ in range(len(X))])
    v1, i1 = np.take_along_axis(rd, perm, 1).copy(), np.take_along_axis(nb, perm, 1).copy()
    v2, i2 = v1.copy(), i1.copy()
    R._legacy_sort_rows(v1, i1)
    getattr(R._legacy_sort_rows, "py_func", R._legacy_sort_rows)(v2, i2)
    assert np.array_equal(i1, i2) and (np.diff(v1, axis=1) >= 0).all()


def test_extscore_pair_stays_consistent_through_smote():
    X, y = _data(seed=4)
    rng = np.random.default_rng(0)
    miss = rng.random(len(y)) < 0.3
    p = np.where(miss, -1.0, rng.random(len(y)))
    Xj = np.hstack([X, p[:, None], miss[:, None].astype(float)])
    Xs, ys, info = smote(Xj, y, random_state=0, n_jobs=1, binary_pairs=[(6, 7)])
    m, pv = Xs[:, 7], Xs[:, 6]
    assert np.isin(m, (0.0, 1.0)).all()
    assert (pv[m == 1] == -1.0).all() and (pv[m == 0] >= 0).all()
    assert info["per_class"][1]["binary_pair_copies"] > 0
    # base columns are untouched by the pair rule
    Xs0, _, _ = smote(Xj, y, random_state=0, n_jobs=1)
    assert np.array_equal(Xs[:, :6], Xs0[:, :6])
    # excluding the ExtScore columns from the distances = neighbours of the base columns only
    Xb, _, ib = smote(Xj, y, random_state=0, n_jobs=1, binary_pairs=[(6, 7)], dist_cols=range(6))
    Xb0, _, ib0 = smote(X, y, random_state=0, n_jobs=1)
    assert np.array_equal(ib["parents"], ib0["parents"]) and np.array_equal(Xb[:, :6], Xb0)


# ------------------------------------------------------------------------------------- AdaBoost

def test_samme_matches_sklearn():
    from sklearn.ensemble import AdaBoostClassifier
    X, y = _data(seed=5)
    kw = {} if "algorithm" not in AdaBoostClassifier().get_params() else {"algorithm": "SAMME"}
    ref = AdaBoostClassifier(n_estimators=50, random_state=100, **kw).fit(X, y)
    mine = AdaBoostStumps(50, algorithm="SAMME", random_state=100).fit(X, y)
    assert np.allclose(ref.estimator_weights_, mine.estimator_weights_)
    assert np.allclose(ref.decision_function(X), mine.decision_function(X), atol=1e-12)
    assert np.array_equal(ref.predict(X), mine.predict(X))


def test_samme_r_is_consistent():
    X, y = _data(seed=6)
    m = AdaBoostStumps(100, random_state=0).fit(X, y)
    d, p = m.decision_function(X), m.predict_proba(X)
    assert np.array_equal(m.predict(X), (d > 0).astype(int))
    assert np.array_equal(p > 0.5, d > 0)
    assert (m.estimator_weights_[:len(m.estimators_)] == 1.0).all()
    m2 = AdaBoostStumps(100, random_state=0).fit(X, y)
    assert np.array_equal(d, m2.decision_function(X))
    assert np.array_equal(d, m.decision_function(X, chunk=777))              # chunking is exact
    assert ((m.predict(X) == y).mean()) > 0.8


def _stump_cases(n_cases=150, seed=0):
    """Adversarial inputs for the stump search: tie-heavy counts, duplicated columns (exact
    cross-feature ties: the feature visit order decides), constant columns, values closer than
    FEATURE_THRESHOLD and adjacent float32s (threshold midpoint rule), booleans, continuous; equal,
    integer (exact sums) and random weights."""
    rng = np.random.default_rng(seed)
    for t in range(n_cases):
        n, F, kind = int(rng.integers(2, 300)), int(rng.integers(1, 9)), t % 5
        if kind == 0:
            X = np.floor(np.exp(rng.normal(1, 1.5, (n, F))))
        elif kind == 1:
            X = np.repeat(np.floor(rng.normal(0, 3, (n, 1))), F, 1)
            if F > 2:
                X[:, 0] = 5.0
        elif kind == 2:
            X = (rng.integers(0, 5, (n, F)) + rng.integers(0, 3, (n, F)) * 4e-8).astype(np.float32)
            X[:, 0] = np.float32(1) + rng.integers(0, 3, n).astype(np.float32) * np.spacing(np.float32(1))
        elif kind == 3:
            X = (rng.random((n, F)) < 0.3).astype(float)
        else:
            X = rng.normal(0, 1, (n, F)) * 10.0 ** rng.integers(-3, 6)
        y = (rng.random(n) < rng.uniform(0.05, 0.95)).astype(np.int64)
        if len(np.unique(y)) < 2:
            y[0] = 1 - y[0]
        w = [np.full(n, 1.0 / n), rng.integers(1, 5, n).astype(float), rng.exponential(1, n)][t % 3]
        yield X, y, w / w.sum() if t % 3 == 2 else w, int(rng.integers(0, 2**31 - 1)), t % 3 < 2


def test_fast_stump_matches_sklearn_tree():
    """Fast stump == DecisionTreeClassifier(max_depth=1) of the installed scikit-learn, bit for bit:
    same feature (incl. exact ties across identical columns, decided by the tree's rand_r draws and
    its sort order), same threshold, same leaf for every row, identical leaf values."""
    from sklearn.tree import DecisionTreeClassifier

    import bb.methods.kudugunta.boost as B
    n_split = 0
    for X, y, w, seed, _ in _stump_cases(300):
        X32 = np.ascontiguousarray(X, np.float32)
        pre = B.Presorted(X32, y.astype(np.int8), threads=1, n_rounds=1)
        st, leaf = B._fast_stump(pre, w, 0, seed, np.random.RandomState(0))
        pre.close()
        est = DecisionTreeClassifier(max_depth=1, random_state=seed).fit(X32, y, sample_weight=w)
        ref = B.stump_from_tree(est)
        assert (st.feature, st.threshold) == (ref.feature, ref.threshold)
        assert np.array_equal(st.value, ref.value)
        if ref.feature >= 0:
            n_split += 1
            assert np.array_equal(leaf, (est.apply(X32) != est.tree_.children_left[0]).astype(np.uint8))
    assert n_split > 200


def test_sort_port_matches_sklearn():
    """``boost._sort3`` == scikit-learn's ``simultaneous_sort(..., use_three_way_partition=True)``
    (the tree's sort), including the permutation of equal values, on tie-heavy, sorted, reversed and
    small inputs. (The heapsort fallback, depth > 2 log2 n, is transliterated but not reached here;
    the real-data checks in kudugunta_fast_check.py compare whole fits bit for bit.)"""
    from sklearn.utils._sorting import _py_simultaneous_sort

    import bb.methods.kudugunta.boost as B
    rng = np.random.default_rng(3)
    cases = [rng.integers(0, k, n).astype(np.float32) for k, n in ((3, 5000), (50, 20000), (10**6, 30000))]
    cases += [np.arange(20000, dtype=np.float32), np.arange(20000, dtype=np.float32)[::-1].copy(),
              np.tile(np.array([0, 2, 1, 2, 0], np.float32), 4000), np.float32([1.0] * 40), np.float32([2, 1])]
    cases += [rng.integers(0, int(rng.integers(1, 6)), int(rng.integers(0, 200))).astype(np.float32)
              for _ in range(2000)]                                       # small, tie-heavy
    for v in cases:
        n = len(v)
        a_v, a_i = v.copy(), rng.permutation(n).astype(np.intp)
        b_v, b_i = a_v.copy(), a_i.copy()
        _py_simultaneous_sort(a_v, a_i, n, use_three_way_partition=True)
        B._sort3(b_v, b_i, n)
        assert np.array_equal(a_v, b_v) and np.array_equal(a_i, b_i)


def test_visit_order_follows_tree_seed():
    """Two identical columns: which one sklearn picks depends only on the tree's rand_r draws."""
    from sklearn.tree import DecisionTreeClassifier

    import bb.methods.kudugunta.boost as B
    X, y = _data(seed=9, n=500)
    X = np.hstack([X[:, :1], X[:, :1], np.full((500, 1), 3.0), X[:, :1]])
    picked = set()
    for seed in range(40):
        est = DecisionTreeClassifier(max_depth=1, random_state=seed).fit(X, y)
        order = B.visit_order(seed, np.array([False, False, True, False]))
        assert sorted(order) == [0, 1, 3] and est.tree_.feature[0] == order[0]
        picked.add(order[0])
    assert picked == {0, 1, 3}


@pytest.mark.parametrize("alg", ["SAMME.R", "SAMME"])
def test_fast_adaboost_matches_sklearn_path(alg):
    from bb.methods.kudugunta.boost import Stump, stump_from_tree
    X, y = _data(seed=10, n=4000)
    X[:, 0] += np.random.default_rng(1).random(4000).round(1)   # non-integer values too
    a = AdaBoostStumps(100, algorithm=alg, random_state=300, stumps="fast").fit(X, y)
    b = AdaBoostStumps(100, algorithm=alg, random_state=300, stumps="sklearn").fit(X, y)
    sb = [stump_from_tree(e) for e in b.estimators_]
    assert [(s.feature, s.threshold) for s in a.estimators_] == [(s.feature, s.threshold) for s in sb]
    assert all(np.array_equal(s.value, t.value) for s, t in zip(a.estimators_, sb))
    assert np.array_equal(a.estimator_weights_, b.estimator_weights_)
    da, db = a.decision_function(X), b.decision_function(X)
    assert np.array_equal(da, db) and np.array_equal(a.predict(X), b.predict(X))    # bit for bit
    for batch in (1, 3):                                                   # look-ahead batch size is irrelevant
        c = AdaBoostStumps(100, algorithm=alg, random_state=300, stumps="fast", n_threads=batch).fit(X, y)
        assert np.array_equal(c.decision_function(X), da)
    # the stump decision kernel is bit-identical to sklearn's predict_proba-based _decision
    b.estimators_ = sb
    assert isinstance(b.estimators_[0], Stump) and np.array_equal(b.decision_function(X), db)


# ------------------------------------------------------------------------------------- features

def _benchmark_loop(raw):
    """train.py L109-169, literally (on the fixture's raw files), returning X and y in benchmark rows."""
    data = json.loads((raw / "user.json").read_text())
    labels = list(csv.DictReader(open(raw / "label.csv")))
    split = list(csv.DictReader(open(raw / "split.csv")))
    n = len(labels)
    allk = {data[i]["id"]: 1 for i in range(n)}
    d, j, k, t = {}, 0, 0, 0
    for r in split:
        if r["id"] in allk and r["id"] not in d:
            if r["split"] == "train":
                d[r["id"]] = j; j += 1
            if r["split"] == "val":
                d[r["id"]] = k; k += 1
            if r["split"] == "test":
                d[r["id"]] = t; t += 1
    for r in split:
        if r["split"] == "val":
            d[r["id"]] += j
        if r["split"] == "test":
            d[r["id"]] += j + k
    y = np.zeros(n)
    for r in labels:
        if r["label"] == "bot":
            y[d[r["id"]]] = 1
    feat = ["followers_count", "following_count", "tweet_count", "listed_count", "protected", "verified",
            "profile_image_url", "location"]
    for u in data:
        u["protected"] = 0 if u.get("protected") in (None, "False", 0) else 1
        u["verified"] = 0 if u.get("verified") in (None, "False", 0) else 1
        u["location"] = 0 if u["location"] is None else 1
        u["profile_image_url"] = 0 if u["profile_image_url"] == "" else 1
    X = np.zeros((n, 8))
    for i in range(n):
        for c in range(4):
            X[d[data[i]["id"]]][c] = data[i]["public_metrics"][feat[c]]
        for c in range(4, 8):
            X[i][c] = data[i][feat[c]]
    return X, y, d, (j, k)


def test_tw22_features_and_misalignment(tw22, tmp_path):
    from bb.data import load_labels, load_users
    from bb.methods.kudugunta.features import build_features
    raw = tmp_path / "raw22"
    Xb, yb, d, (j, k) = _benchmark_loop(raw)
    u = load_users("twibot22", columns=["raw_id", "split"])
    pos = np.array([d[r] for r in u["raw_id"]])                          # benchmark row of each idx
    y = load_labels("twibot22")
    assert np.array_equal(yb[pos], y)
    Xf, names, _ = build_features("twibot22")
    assert names[-2:] == ["profile_image_url", "location"] and Xf.shape == (12, 8)
    Xm, _, info = build_features("twibot22", bench_misalign=True)
    assert np.array_equal(Xm, Xb[pos])                                   # exact benchmark rows, bug included
    assert np.array_equal(Xf[:, :4], Xb[pos][:, :4])                     # counts were always aligned
    assert not np.array_equal(Xf[:, 4:], Xm[:, 4:])                      # booleans are not
    raw_users = json.loads((raw / "user.json").read_text())
    assert np.array_equal(Xf[:, 7], [u["location"] is not None for u in raw_users])
    assert np.array_equal(Xf[:, 5], [bool(u["verified"]) for u in raw_users])
    assert info["misalign_same_user"] < 12
    with pytest.raises(ValueError):
        build_features("twibot22", tw20_literal_bool_strings=True)


def test_tw20_literal_bool_strings(tw20):
    from bb.data import load_profile
    from bb.methods.kudugunta.features import build_features
    X, names, _ = build_features("twibot20")
    Xl, _, _ = build_features("twibot20", tw20_literal_bool_strings=True)
    p = load_profile("twibot20", ["protected", "verified", "raw_verified"])
    has = p["raw_verified"].notna().to_numpy()
    assert X.shape[1] == 6 and np.array_equal(X[:, 5], p["verified"].fillna(False).to_numpy(dtype=bool))
    assert (Xl[has, 4:] == 1).all() and (Xl[~has, 4:] == 0).all()
    assert np.array_equal(Xl[:, :4], X[:, :4])
    with pytest.raises(ValueError):
        build_features("twibot20", bench_misalign=True)


# ------------------------------------------------------------------------------------- end to end

def _run(argv):
    from bb.methods.kudugunta.main import main
    return main(argv)


@pytest.fixture
def cr15_feats(cr15, monkeypatch):
    """The Cresci fixture's profiles are all identical (degenerate for kNN/stumps); keep its index,
    splits and labels but give the users label-correlated count features."""
    import bb.methods.kudugunta.main as M
    from bb.data import load_labels
    y = load_labels("cresci15").astype(float)
    rng = np.random.default_rng(0)
    n = len(y)
    X = np.hstack([np.floor(10 + 90 * y[:, None] + rng.normal(0, 3, (n, 4))), (rng.random((n, 2)) < 0.5)]).astype(float)
    monkeypatch.setattr(M, "build_features", lambda ds, **kw: (X.copy(), ["c0", "c1", "c2", "c3", "b0", "b1"], {}))
    return X


def test_end_to_end_with_synthetic_extscore(cr15_feats):
    from bb.extscore import make_synthetic_extscore
    make_synthetic_extscore("cresci15", "synthetic", signal=2.0, coverage=0.8)
    base = _run(["--dataset", "cresci15", "--seeds", "0", "1", "--smote-k", "2", "--enn-k", "2"])
    again = _run(["--dataset", "cresci15", "--seeds", "0", "--smote-k", "2", "--enn-k", "2", "--extscore", "none"])
    assert base[0]["metrics"] == again[0]["metrics"]                     # extscore none = baseline, deterministic
    assert base[0]["features"]["width"] == 6 and base[0]["extscore"] == {"extscore": "none"}
    for extra in ([], ["--extscore-transform", "meanfill"], ["--extscore-knn", "exclude"]):
        r = _run(["--dataset", "cresci15", "--seeds", "0", "--smote-k", "2", "--enn-k", "2",
                  "--with-extscore", "synthetic", *extra])[0]
        assert r["status"] == "ok" and r["features"]["width"] == 8
        assert r["extscore"]["tag"] == "synthetic" and r["extscore"]["coverage"]["train"]["observed"] > 0
        assert r["config"]["variant"].startswith("bench")
        for s in ("val", "test"):
            assert 0.0 <= r["metrics"][s]["accuracy"] <= 1.0


def test_no_leak_eval_labels_unused(cr15_feats, monkeypatch):
    """Flipping every val/test label must not change any prediction."""
    import bb.methods.kudugunta.main as M
    from bb.data import load_split, load_users

    r0 = _run(["--dataset", "cresci15", "--seeds", "0", "--smote-k", "2", "--enn-k", "2"])[0]
    ev = np.concatenate([np.asarray(load_split("cresci15", s)) for s in ("val", "test")])
    real = load_users

    def flipped(ds, columns=None):
        df = real(ds, columns=columns)
        if "label" in df:
            lab = df["label"].to_numpy().copy()
            lab[ev] = 1 - lab[ev]
            df["label"] = lab
        return df

    monkeypatch.setattr(M, "load_users", flipped)
    r1 = _run(["--dataset", "cresci15", "--seeds", "0", "--smote-k", "2", "--enn-k", "2"])[0]
    from pathlib import Path

    from bb.paths import runs_root
    base = Path(runs_root()) / "kudugunta" / "cresci15"
    for s in ("val", "test"):
        p0 = np.load(base / r0["run"] / f"preds_{s}.npz")["prob"]
        p1 = np.load(base / r1["run"] / f"preds_{s}.npz")["prob"]
        assert np.array_equal(p0, p1)
    assert r0["metrics"]["train"] == r1["metrics"]["train"]
    assert r0["metrics"]["test"]["accuracy"] + r1["metrics"]["test"]["accuracy"] == pytest.approx(1.0)
