"""SGBot: bigram likelihood and feature slots against naive re-statements of the benchmark code,
the train-only likelihood corpus (no val/test information), end-to-end runs and ExtScore plumbing."""
from __future__ import annotations

import string

import numpy as np
import pandas as pd
import pytest

from bb.methods.sgbot import features as F
from bb.methods.sgbot import main as sg

ALPHA = set(string.ascii_letters + string.digits + "_")


def _naive_table(names):
    c = {}
    for s in names:
        s = s.strip()
        for i in range(len(s) - 1):
            c[s[i:i + 2]] = c.get(s[i:i + 2], 0) + 1
    tot = sum(c.values())
    return {k: v / tot for k, v in c.items()}


def _naive_lik(s, tab):
    s = s.strip()
    if len(s) <= 1:
        return 1.0
    p = 1.0
    for i in range(len(s) - 1):
        p *= tab.get(s[i:i + 2], 0.0)
    return p ** (1 / (len(s) - 1))


def _s(x):
    return pd.Series(x, dtype="string[pyarrow]")


def test_likelihood_matches_naive():
    rng = np.random.default_rng(1)
    alpha = list("abcdXY019_")
    corpus = ["".join(rng.choice(alpha, rng.integers(0, 16))) for _ in range(300)]
    prob, info = F.bigram_table(_s(corpus))
    tab = _naive_table(corpus)
    for k, v in tab.items():
        assert prob[F.ALPHABET.index(k[0]) * 63 + F.ALPHABET.index(k[1])] == pytest.approx(v)
    queries = corpus[:50] + ["", "a", "zz", "abQ"]
    got, linfo = F.likelihood(_s(queries), prob)
    want = [_naive_lik(q, tab) for q in queries]
    assert np.allclose(got, want, rtol=1e-10, atol=0)
    assert got[-2] == 0 and got[-3] == 1.0                # unseen bigram -> 0; len 1 -> 1
    assert F.likelihood(_s([None]), prob)[0][0] == 0.0     # null -> 0
    assert info["invalid_bigrams_skipped"] == 0


def test_invalid_characters_are_skipped_and_counted():
    prob, info = F.bigram_table(_s(["ab", "aé", "b"]))
    assert info["invalid_bigrams_skipped"] == 1 and info["corpus_bigrams"] == 1


def _profile(rows):
    df = pd.DataFrame(rows)
    for c in ("screen_name", "name", "description", "profile_image_url"):
        df[c] = df[c].astype("string[pyarrow]")
    for c in ("statuses_count", "followers_count", "following_count", "listed_count", "favourites_count"):
        df[c] = df[c].astype("Int64")
    for c in ("verified", "default_profile", "profile_use_background_image"):
        df[c] = df[c].astype("boolean")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df


def test_profile_feature_slots():
    p = _profile([
        {"screen_name": "ab12", "name": " Bob 7 ", "description": " hi ", "profile_image_url": "x/default_profile_normal.png",
         "verified": True, "created_at": "2022-12-21 00:00:00", "statuses_count": 100, "followers_count": 50,
         "following_count": 0, "listed_count": 5, "favourites_count": 20, "default_profile": True,
         "profile_use_background_image": False},
        {"screen_name": "c", "name": None, "description": None, "profile_image_url": "x/me.png",
         "verified": None, "created_at": None, "statuses_count": None, "followers_count": 10,
         "following_count": 4, "listed_count": 0, "favourites_count": 73, "default_profile": None,
         "profile_use_background_image": True},
    ])
    prob, _ = F.bigram_table(p["screen_name"])
    X, info = F.profile_features(p, "twibot22", prob)
    assert X.shape == (2, 20)
    assert list(X[:, 5]) == [0, 1] and list(X[:, 7]) == [1, 0]
    assert np.all(X[:, [3, 6, 11]] == 0)
    assert X[0, 8] == pytest.approx(100 / 10) and X[1, 9] == pytest.approx(10 / 730)
    assert list(X[:, 13]) == [0, 2.5]                      # benchmark: ratio 0 when friends = 0
    assert list(X[:, 14]) == [4, 1] and list(X[:, 15]) == [2, 0]
    assert list(X[:, 16]) == [5, 0] and list(X[:, 17]) == [1, 0] and list(X[:, 18]) == [2, 0]
    assert X[1, 19] == 1.0
    Xp, info = F.profile_features(p, "twibot20", prob, mode="paper")
    assert list(Xp[:, 3]) == [20, 73] and list(Xp[:, 5]) == [1, 0] and list(Xp[:, 6]) == [0, 1]
    assert list(Xp[:, 13]) == [50, 2.5]                    # paper: / max(friends, 1)
    assert Xp[1, 11] == pytest.approx(73 / 730)
    Xa, info = F.profile_features(p, "twibot22", prob, mode="paper", absent={"favourites_count", "default_profile"})
    assert np.all(Xa[:, 3] == 0) and list(Xa[:, 5]) == [0, 1]   # absent fields keep benchmark slots


def test_train_users_corpus_uses_no_eval_names(tw20, data_root):
    """No-leak: with likelihood_corpus=train_users the features of train rows do not change when
    val/test/support screen names change (the corpus sees train names only)."""
    import argparse
    from bb.methods.bothunter.shared import load_index
    y, splits, rows = load_index("twibot20")
    a = argparse.Namespace(features="benchmark", likelihood_corpus="train_users", reference_year=None)
    X1, _, info1 = F.build_features("twibot20", rows, splits, a)
    a.likelihood_corpus = "all_users"
    X2, _, info2 = F.build_features("twibot20", rows, splits, a)
    tr_names = F.load_profile("twibot20", ["screen_name"])["screen_name"].iloc[splits["train"]]
    assert info1["bigram"]["corpus_names"] == int(tr_names.notna().sum())
    assert info2["bigram"]["corpus_names"] > info1["bigram"]["corpus_names"]
    assert np.array_equal(X1[:, :19], X2[:, :19])          # only the likelihood column differs
    prob, _ = F.bigram_table(tr_names)
    lik, _ = F.likelihood(F.load_profile("twibot20", ["screen_name"], rows=rows)["screen_name"], prob)
    assert np.allclose(X1[:, 19], lik)


def _bench_age(created_at: str | None, year: int, fmt: str) -> int:
    """Benchmark ``calc_age`` (src/SGBot/preprocess.py L109-130) on the release string."""
    from datetime import datetime
    if created_at is None:
        return 730
    t = datetime.strptime(created_at.strip(), fmt).replace(tzinfo=None)
    return (datetime.strptime(f"{year} Dec 31", "%Y %b %d") - t).days


@pytest.mark.parametrize("ds", ["twibot20", "twibot22"])
def test_benchmark_vector_matches_naive_port(ds, tw20, tw22, data_root):
    """Review round 1: the whole 20-column benchmark vector of every labelled user equals a naive
    per-user port of the benchmark's preprocess.py/utils.py run on the release strings (raw_*
    columns: trailing spaces, 'True '/'False ' flags), with the bigram table from ALL user nodes."""
    import argparse
    from bb.methods.bothunter.shared import load_index
    y, splits, rows = load_index(ds)
    a = argparse.Namespace(features="benchmark", likelihood_corpus="all_users", reference_year=None)
    X, _, _ = F.build_features(ds, rows, splits, a)
    # null; and the TwiBot-20 literal "None " string, which the data layer reads as null (clean
    # semantics, spec 3.5; no labelled real TwiBot-20 user has it in name/description)
    raw = lambda v: (None if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v))
                     or (isinstance(v, str) and v.strip() == "None") else v)
    tab = _naive_table([s for s in F.load_profile(ds, ["raw_screen_name"])["raw_screen_name"] if raw(s) is not None])
    year, fmt = (2020, "%a %b %d %H:%M:%S %z %Y") if ds == "twibot20" else (2022, "%Y-%m-%d %H:%M:%S%z")
    cols = ["raw_screen_name", "raw_name", "raw_description", "raw_profile_image_url", "raw_verified",
            "raw_created_at", "statuses_count", "followers_count", "following_count", "listed_count"]
    p = F.load_profile(ds, cols, rows=rows)
    L = lambda s: 0 if s is None else len(s.strip())
    D = lambda s: 0 if s is None else sum("0" <= c <= "9" for c in s)
    for i, r in enumerate(p.itertuples(index=False)):
        sn, nm, de, url, ver, ca = (raw(v) for v in (r.raw_screen_name, r.raw_name, r.raw_description,
                                                       r.raw_profile_image_url, r.raw_verified, r.raw_created_at))
        st, fo, fr, li = (0 if raw(v) is None else int(v) for v in (r.statuses_count, r.followers_count,
                                                                    r.following_count, r.listed_count))
        age = _bench_age(ca, year, fmt)
        v = 0 if ver is None else int(str(ver).strip() == "True")
        img = 0 if url is None else int(url.find("default_profile_normal") == -1)
        want = [st, fo, fr, 0, li, img, 0, v, st / age, fo / age, fr / age, 0, li / age,
                0 if fr == 0 else fo / fr, L(sn), D(sn), L(nm), D(nm), L(de),
                0 if sn is None else _naive_lik(sn, tab)]
        assert np.allclose(X[i], np.float32(want), rtol=1e-6, atol=0), (ds, i, X[i], want)


def test_end_to_end(tw20, cr15, tw22, data_root):
    from bb.harness import load_results
    from bb.extscore import make_synthetic_extscore
    argv = ["--dataset", "twibot20", "--seeds", "3", "--n-jobs", "2"]
    sg.main(argv)
    sg.main(argv)
    make_synthetic_extscore("twibot20", tag="syn", coverage=0.8)
    sg.main(argv + ["--with-extscore", "syn", "--perm-repeats", "1"])
    sg.main(argv + ["--features", "paper"])
    base = load_results("sgbot", "twibot20", where={"with_extscore": None, "features": "benchmark"})
    extscore = load_results("sgbot", "twibot20", where={"with_extscore": "syn"})
    assert len(base) == 2 and base[0]["metrics"]["test"] == base[1]["metrics"]["test"]
    assert base[0]["features"]["width"] == 20 and extscore[0]["features"]["width"] == 22
    assert extscore[0]["extscore"]["coverage"]["train"]["n"] == 6
    paper = load_results("sgbot", "twibot20", where={"features": "paper"})[0]
    assert paper["features"]["paper_fields_filled"] == ["favourites_count", "default_profile",
                                                        "profile_use_background_image"]
    sg.main(["--dataset", "cresci15", "--seeds", "0", "--n-jobs", "2", "--split", "random"])
    sg.main(["--dataset", "twibot22", "--seeds", "0", "--n-jobs", "2"])
    r = load_results("sgbot", "twibot22")[0]
    assert r["features"]["zero_slots"] == ["favourites_count", "profile_use_background_image", "favourites_growth"]
    assert load_results("sgbot", "cresci15")[0]["config"]["split"] == "random"
