"""BotHunter: feature definitions against naive re-statements of the benchmark code, newest-tweet
selection, end-to-end runs on the synthetic fixtures, ExtScore plumbing, seeded reproducibility."""
from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from bb.methods.bothunter import features as F
from bb.methods.bothunter import main as bh
from bb.methods.bothunter import shared as S


def _naive_entropy(s: str) -> float:
    s = s.strip()
    if not s:
        return 0.0
    return -sum((c / len(s)) * math.log(c / len(s)) for c in Counter(s).values())


def test_entropy_matches_naive():
    rng = np.random.default_rng(0)
    alpha = list("abcAB_01xyzé😀 ")
    names = ["".join(rng.choice(alpha, rng.integers(0, 17))) for _ in range(500)] + ["", "a", "aaaa"]
    got = S.shannon_entropy(pd.Series(names, dtype="string[pyarrow]").str.strip())
    want = np.array([_naive_entropy(s) for s in names])
    assert np.allclose(got, want, atol=1e-12)
    assert S.shannon_entropy(pd.Series([None, "ab"], dtype="string[pyarrow]"))[0] == 0.0


def _profile(rows):
    df = pd.DataFrame(rows)
    for c in ("screen_name", "description", "location", "raw_location", "profile_image_url"):
        df[c] = df[c].astype("string[pyarrow]")
    for c in ("statuses_count", "following_count", "followers_count"):
        df[c] = df[c].astype("Int64")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df


def test_profile_features_benchmark_slots():
    p = _profile([
        {"screen_name": "newsbot_1", "description": "hello", "location": "Rome", "raw_location": "Rome ",
         "profile_image_url": "https://x/default_profile_normal.png", "created_at": "2020-12-30 23:00:00",
         "statuses_count": 10, "following_count": 3, "followers_count": 7},
        {"screen_name": "Alice", "description": "I am a Bot builder", "location": None, "raw_location": "None ",
         "profile_image_url": None, "created_at": None,
         "statuses_count": None, "following_count": 0, "followers_count": 1},
        {"screen_name": "bob", "description": None, "location": "", "raw_location": " ",
         "profile_image_url": "https://x/me_normal.jpg", "created_at": "2019-12-31 12:00:00",
         "statuses_count": 365, "following_count": 1, "followers_count": 2},
    ])
    X, info = F.profile_features(p, "twibot20")
    assert X.shape == (3, 17)
    assert list(X[:, 0]) == [9, 5, 3]
    assert list(X[:, 1]) == [0, 0, 1]                                  # null url -> 0
    assert X[0, 2] == pytest.approx(_naive_entropy("newsbot_1"))
    assert list(X[:, 3]) == [1, 0, 0]
    assert list(X[:, 4]) == [10, 0, 365]
    assert list(X[:, 6]) == [3, 0, 1] and list(X[:, 7]) == [7, 1, 2]
    assert list(X[:, 14]) == [1, 0, 0]                                 # case-sensitive 'bot'
    assert list(X[:, 15]) == [0, 730, 365]                             # Dec 31 2020 ref, floor days, null -> 730
    assert X[2, 16] == pytest.approx(1.0)
    assert info["age"]["null_created_at"] == 1
    for k in F.ZERO_SLOTS:
        assert np.all(X[:, k] == 0)
    Xc, _ = F.profile_features(p, "twibot20", bot_ref_casefold=True, literal_none_location=True)
    assert list(Xc[:, 14]) == [1, 1, 0]
    assert list(Xc[:, 3]) == [1, 1, 0]                                 # raw "None " counts as a location


def _write_store(d, offsets, created, texts):
    d.mkdir(parents=True)
    np.save(d / "offsets.i64.npy", np.asarray(offsets, np.int64))
    blob = "".join(texts).encode()
    to = np.zeros(len(texts) + 1, np.int64)
    np.cumsum([len(t.encode()) for t in texts], out=to[1:])
    np.save(d / "text_offsets.i64.npy", to)
    (d / "text.bin").write_bytes(blob)
    cols = {}
    if created is not None:
        np.save(d / "created_at.npy", np.asarray(created, np.int64))
        cols["created_at"] = "int64"
    (d / "meta.json").write_text(json.dumps({"columns": cols}))


def test_newest_rows(tmp_path):
    from bb.data import TweetStore
    # users: 0 -> rows 0..2, 1 -> none, 2 -> rows 3..4 (tie: first), 3 -> row 5 (null created_at)
    _write_store(tmp_path / "a", [0, 3, 3, 5, 6], [5, 9, 7, 4, 4, -1], ["a", "b", "c", "d", "e", "f"])
    st = TweetStore(tmp_path / "a")
    for chunk in (1, 2, 100):
        assert list(F.newest_rows(st, 4, chunk_users=chunk)) == [1, -1, 3, 5]
    _write_store(tmp_path / "b", [0, 2, 2, 3], None, ["x", "y", "z"])
    assert list(F.newest_rows(TweetStore(tmp_path / "b"), 3)) == [0, -1, 2]


def test_argmax_threshold_equals_sklearn_predict():
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 3)).astype(np.float32)  # pure leaves, 2 trees -> 50/50 votes
    y = rng.integers(0, 2, 400)
    rf = RandomForestClassifier(n_estimators=2, random_state=0).fit(X, y)
    p = rf.predict_proba(X)[:, 1]
    assert np.any(p == 0.5)
    assert np.array_equal((p >= S.ARGMAX_THRESHOLD).astype(int), rf.predict(X))


def test_fortran_train_matrix_gives_identical_forest():
    """Review round 2: the seed loop fits on a column-major copy of the train rows; the forest
    (every tree array) and its probabilities must equal those of the row-major fit, and sklearn
    must not copy the float32 F-order matrix back to C order."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.utils.validation import check_array
    rng = np.random.default_rng(1)
    X = rng.integers(0, 50, (3000, 9)).astype(np.float32)  # many ties, like count features
    y = (X[:, 0] + rng.integers(0, 30, 3000) > 40).astype(np.int64)
    Xf = np.asfortranarray(X)
    assert check_array(Xf, dtype=np.float32) is Xf or np.shares_memory(check_array(Xf, dtype=np.float32), Xf)
    a = RandomForestClassifier(n_estimators=10, n_jobs=2, random_state=3).fit(X, y)
    b = RandomForestClassifier(n_estimators=10, n_jobs=2, random_state=3).fit(Xf, y)
    for ea, eb in zip(a.estimators_, b.estimators_):
        for f in ("children_left", "children_right", "feature", "threshold", "value", "n_node_samples"):
            assert np.array_equal(getattr(ea.tree_, f), getattr(eb.tree_, f))
    assert np.array_equal(a.predict_proba(X), b.predict_proba(X))
    S.release_memory()  # must be a harmless no-op on any platform


def test_end_to_end_twibot20(tw20, data_root):
    from bb.harness import load_results
    from bb.extscore import make_synthetic_extscore
    argv = ["--dataset", "twibot20", "--seeds", "0", "--n-jobs", "2"]
    bh.main(argv)
    bh.main(argv)
    make_synthetic_extscore("twibot20", tag="syn", coverage=1.0)
    bh.main(argv + ["--with-extscore", "syn", "--perm-repeats", "2"])
    bh.main(argv + ["--features", "extended"])
    base = load_results("bothunter", "twibot20", where={"with_extscore": None, "features": "benchmark"})
    extscore = load_results("bothunter", "twibot20", where={"with_extscore": "syn"})
    ext = load_results("bothunter", "twibot20", where={"features": "extended"})
    assert len(base) == 2 and len(extscore) == 1 and len(ext) == 1
    assert base[0]["metrics"]["test"] == base[1]["metrics"]["test"]       # seeded runs reproduce
    assert base[0]["features"]["width"] == 17 and extscore[0]["features"]["width"] == 19
    assert extscore[0]["feature_names"][-2:] == ["extscore_p", "extscore_missing"]
    assert base[0]["extscore"] == {"extscore": "none"} and extscore[0]["extscore"]["tag"] == "syn"
    assert set(extscore[0]["extscore_permutation_importance"]) == {"extscore_p", "extscore_missing"}
    assert "source_bin" in base[0]["features"]["constant_on_train"]
    assert "last_n_hashtags" not in ext[0]["features"]["zero_slots"]


def _capture_fits(monkeypatch):
    """Record the (X, y) every RandomForestClassifier.fit in the shared seed loop receives."""
    import sklearn.ensemble
    seen = []
    real = sklearn.ensemble.RandomForestClassifier

    class Spy(real):
        def fit(self, X, y, *a, **k):
            seen.append((np.array(X), np.array(y)))
            return super().fit(X, y, *a, **k)

    monkeypatch.setattr(sklearn.ensemble, "RandomForestClassifier", Spy)
    return seen


def test_extscore_columns_aligned_with_train_rows(tw20, data_root, monkeypatch):
    """Review round 1: the ExtScore columns the forest sees are the tree transform ([p or -1, missing]) of
    exactly the train users, in the same row order as the features and labels, and ``--extscore none``
    leaves the base matrix untouched (the ExtScore run's first 17 columns equal the baseline's)."""
    from bb.data import load_split, load_users
    from bb.extscore import load_extscore, make_synthetic_extscore
    seen = _capture_fits(monkeypatch)
    argv = ["--dataset", "twibot20", "--seeds", "0", "--n-jobs", "1"]
    bh.main(argv)
    make_synthetic_extscore("twibot20", tag="syn", coverage=0.7)
    bh.main(argv + ["--with-extscore", "syn", "--perm-repeats", "0"])
    (Xb, yb), (Xj, yj) = seen
    tr = np.asarray(load_split("twibot20", "train"))
    j = load_extscore("twibot20", "syn")
    assert Xb.shape[1] == 17 and Xj.shape[1] == 19
    assert np.array_equal(Xj[:, :17], Xb) and np.array_equal(yj, yb)
    assert np.array_equal(yb, load_users("twibot20", columns=["label"])["label"].to_numpy()[tr])
    want_p = np.where(j.observed[tr], j.p[tr], -1.0).astype(np.float32)
    assert np.array_equal(Xj[:, 17], want_p)
    assert np.array_equal(Xj[:, 18], j.missing[tr].astype(np.float32))
    assert np.all(Xj[Xj[:, 18] == 1, 17] == -1) and len(np.unique(Xj[:, 17])) > 1


def test_end_to_end_cresci_and_twibot22(cr15, tw22, data_root):
    from bb.harness import load_results
    bh.main(["--dataset", "cresci15", "--seeds", "0", "--n-jobs", "2", "--features", "extended"])
    bh.main(["--dataset", "twibot22", "--seeds", "1", "--n-jobs", "2", "--drop-constant"])
    r = load_results("bothunter", "twibot22")[0]
    assert r["features"]["width"] == 17 - len(r["features"]["constant_on_train"])
    assert r["metrics"]["test"]["n"] == 3
    assert load_results("bothunter", "cresci15")[0]["features"]["mode"] == "extended"
