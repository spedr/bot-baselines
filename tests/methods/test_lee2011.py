"""Lee 2011 (bb.methods.lee2011) on synthetic fixtures. spaCy is replaced by a fake asset set (the
blank English tokenizer + random 8-d vectors for a few words), so no model download is needed.
spaCy imports torch through thinc; run under memrun --light:

    tools/memrun --light -- uv run pytest tests/methods/test_lee2011.py

The real-model check (ours vs ``Doc.similarity``) is tests/methods/lee2011_spacy_check.py.
"""
from __future__ import annotations

import itertools
import os
import zipfile

import numpy as np
import pytest

from bb.io import save_npy, write_json
from bb.methods.lee2011 import features as F

WORDS = ["tw", "tweet", "of", "t0.0", "RT", "@x", ":", "😀", "é", "hello", "world"]


def fake_assets(dim: int = 8) -> dict:
    """Blank-English tokenizer + random vectors for WORDS, at the paths the method reads."""
    from spacy.lang.en import English

    from bb.methods.lee2011 import assets as A

    nlp = English()
    md = A.model_dir()
    (md / "vocab").mkdir(parents=True, exist_ok=True)
    nlp.tokenizer.to_disk(md / "tokenizer")
    (md / "vocab" / "vectors").write_bytes(b"")
    keys = np.array(sorted({nlp.vocab.strings.add(w) for w in WORDS}), np.uint64)
    vec = np.random.default_rng(0).standard_normal((len(keys), dim)).astype(np.float32)
    vd = A.vectors_dir()
    save_npy(vd / "keys.u64.npy", keys)
    save_npy(vd / "rows.i32.npy", np.arange(len(keys), dtype=np.int32))
    save_npy(vd / "vectors.f32.npy", vec)
    write_json(vd / "meta.json", {"model": "fake", "n_keys": len(keys), "dim": dim})
    return {"keys": keys, "vec": vec}


# ---------------------------------------------------------------------------------------------
# pure functions


def test_regexes_and_counts():
    texts = ["RT @a_b: see https://t.co/AB?x=1#f and http://x.io", "@a_b @c+d http://x.io", ""]
    n_url, n_uurl, n_men, n_umen, U, C = F.text_stats(texts)
    assert (n_url, n_uurl) == (3, 2)
    # [$-_] (0x24-0x5F) keeps ':/?=' and upper case; '#' (0x23) ends the match
    assert F.URL_RE.findall(texts[0])[0] == "https://t.co/AB?x=1"
    assert (n_men, n_umen) == (3, 2)
    assert U == len(("\n".join(texts) + "\n").encode())
    # the joined findall equals the benchmark's per-tweet findall (no match crosses '\n')
    assert n_url == sum(len(F.URL_RE.findall(t)) for t in texts)


def test_zip_overhead_matches_benchmark_zipfile(tmp_path, monkeypatch):
    """Emulate the benchmark's zip_ratio: write temp_tweets.txt, ZipFile(...).write() and read
    the archive size before it is closed. Its size must be raw-deflate length + 45."""
    monkeypatch.chdir(tmp_path)
    # the last case is > 8 KiB, so zipfile feeds the compressor in several 8 KiB chunks
    big = [f"RT @u{i % 7}: tweet {i} https://t.co/{i * 7919:x} é 😀 {'x' * (i % 40)}" for i in range(2000)]
    for texts in (["hello world"] * 30, ["a", "é 😀 b", "https://x.io/" * 50], [""], big):
        with open("./temp_tweets.txt", "w") as f:
            for t in texts:
                f.write(t + "\n")
        unzip = os.path.getsize("./temp_tweets.txt")
        z = zipfile.ZipFile("./temp_tweets.txt.zip", "w", zipfile.ZIP_DEFLATED)
        z.write("./temp_tweets.txt")
        zsize = os.path.getsize("./temp_tweets.txt.zip")
        z.close()
        *_, U, C = F.text_stats(texts)
        assert U == unzip
        assert zsize == C + F.ZIP_OVERHEAD
        raw = np.array([[len(texts), 0, 0, 0, 0, U, C, 1.0]])
        assert F.content_columns(raw)[0, 4] == pytest.approx(unzip / zsize)


def _naive_similarity(docs_tok, docs_vec):
    n = len(docs_tok)
    if n == 0:
        return np.nan
    if n == 1:
        return 1.0
    out = []
    for i, j in itertools.combinations(range(n), 2):
        if docs_tok[i] == docs_tok[j]:
            out.append(1.0)
            continue
        vi, vj = docs_vec[i], docs_vec[j]
        ni, nj = np.linalg.norm(vi), np.linalg.norm(vj)
        out.append(0.0 if ni == 0 or nj == 0 else float(vi @ vj / (ni * nj)))
    return float(np.mean(out))


def test_similarity_kernel_matches_spacy_rule():
    rng = np.random.default_rng(1)
    vec = rng.standard_normal((5, 4)).astype(np.float32)
    # users: 0 tweets, 1 tweet, identical docs + empty doc + all-OOV doc + normal docs
    users = [[], [[1, 2]], [[1, 2], [1, 2], [], [], [7, 8], [3], [1, 4, 4]]]
    orth_of = {k: np.uint64(1000 + k) for k in range(10)}
    row_of = {k: (k if k < 5 else -1) for k in range(10)}  # tokens 5..9 have no vector
    uo, to, orth, row = [0], [0], [], []
    for u in users:
        for d in u:
            orth += [orth_of[k] for k in d]
            row += [row_of[k] for k in d]
            to.append(len(orth))
        uo.append(len(to) - 1)
    out = np.empty(len(users))
    F.similarity_kernel(np.array(uo, np.int64), np.array(to, np.int64), np.array(orth, np.uint64),
                        np.array(row, np.int64), vec, out)
    for k, u in enumerate(users):
        vecs = [np.mean([vec[row_of[t]] if row_of[t] >= 0 else np.zeros(4) for t in d], 0) if d
                else np.zeros(4) for d in u]
        ref = _naive_similarity(u, vecs)
        if np.isnan(ref):
            assert np.isnan(out[k])
        else:
            assert out[k] == pytest.approx(ref, abs=1e-6)


TRICKY = ["RT @a_b: see https://t.co/AB?x=1#f and http://x.io :) :-( <3", "don't U.S.A. e.g. (e.g.)",
          "dyn-o-mite-dave 1,000.50 $5 5km 10:30pm", "😀😀 hello!!! ¿qué? naïve", "", "  \n\t two  spaces",
          "a.m . a.m. p.m.", "#tag @user_1 email@x.com www.example.com/path?q=1"]


def test_lean_tokenizer_keeps_orth(data_root):
    """Round 2: the lean tokenizer (no lexeme attribute getters) gives the same ORTH sequences and
    vector rows as the default English tokenizer on every text (also on repeats, i.e. from its
    span cache)."""
    from spacy.attrs import ORTH
    from spacy.lang.en import English

    from bb.methods.lee2011 import assets as A

    fa = fake_assets()
    keys, rows = fa["keys"], np.arange(len(fa["keys"]), dtype=np.int32)
    ref_tok = English().tokenizer
    texts = TRICKY + WORDS + TRICKY[::-1] + [w + " " + w for w in WORDS] + TRICKY
    ref = [ref_tok(t).to_array(ORTH).astype(np.uint64) for t in texts]
    lean = A.load_tokenizer()
    assert lean.vocab.lex_attr_getters == {}
    off, orth, row = F.tokenize(lean, texts, keys, rows)
    assert len(off) == len(texts) + 1
    for i, r in enumerate(ref):
        assert np.array_equal(orth[off[i]:off[i + 1]], r), texts[i]
        assert np.array_equal(row[off[i]:off[i + 1]], A.lookup_rows(r, keys, rows))
    assert np.isin(row, -1).any() and (row >= 0).any()


def test_ensure_assets_needs_only_runtime_files(data_root, monkeypatch):
    """After prune (wheel, extracted vocab and full extraction deleted) the runtime assets still
    load without the wheel; a missing tokenizer triggers a rebuild from the (fetched) wheel."""
    from bb.methods.lee2011 import assets as A

    fake_assets()
    root = A.asset_root()
    (root / "dl").mkdir(parents=True)
    (root / "dl" / A.WHEEL).write_bytes(b"x" * 10)
    full = root / f"{A.MODEL}-{A.MODEL_VERSION}-full" / "tok2vec"
    full.mkdir(parents=True)
    (full / "model").write_bytes(b"y" * 5)
    calls = []

    def fake_fetch(*a, **k):  # a downloaded file that fails the sha256 pin
        calls.append(1)
        (root / "dl").mkdir(parents=True, exist_ok=True)
        (root / "dl" / A.WHEEL).write_bytes(b"not the wheel")
        return root / "dl" / A.WHEEL

    monkeypatch.setattr(A, "fetch_wheel", fake_fetch)
    assert A.prune_assets() == 15
    assert not (root / "dl").exists() and not (A.model_dir() / "vocab").exists() and not full.exists()
    assert A.ensure_assets()["model"] == "fake" and not calls  # no rebuild needed
    assert A.load_tokenizer()("hello world")[1].text == "world"
    (A.model_dir() / "tokenizer").unlink()
    with pytest.raises(ValueError, match="sha256"):  # rebuild path: fetch, then check the wheel
        A.ensure_assets()
    assert calls


def test_plan_chunks_respects_budget():
    u = np.arange(10)
    c = F.plan_chunks(u, np.array([5, 5, 5, 20, 1, 1, 1, 1, 1, 1]), 10)
    assert np.array_equal(np.concatenate(c), u)
    assert [list(x) for x in c][:3] == [[0, 1], [2], [3]]


# ---------------------------------------------------------------------------------------------
# data-backed (synthetic fixtures)


def test_profile_features_definitions(cr15):
    from bb.data import load_profile

    X = F.profile_features("cresci15")
    p = load_profile("cresci15", columns=["name", "description", "followers_count", "following_count",
                                          "statuses_count"])
    assert X.shape == (len(p), 8)
    assert np.array_equal(X[:, 0], p["name"].str.len().to_numpy(dtype=float))
    assert np.array_equal(X[:, 1], p["description"].str.len().fillna(0).to_numpy(dtype=float))
    # created 2007-04-06, collection date 2020-10-01 for every non-TwiBot-22 dataset
    assert X[0, 2] == (np.datetime64("2020-10-01") - np.datetime64("2007-04-06")).astype(int)
    assert np.allclose(X[:, 5], p["following_count"].to_numpy(float) / p["followers_count"].to_numpy(float))
    assert np.allclose(X[:, 7], X[:, 6] / X[:, 2])
    Xs = F.profile_features("cresci15", col0="screen_name")
    assert not np.array_equal(Xs[:, 0], X[:, 0])


def test_profile_features_nulls_and_zero_branches(monkeypatch):
    """Benchmark user_features.py branches: null -> 0, followers 0 -> ratio 0, longevity 0 (null
    created_at, or created on the collection date) -> tweets_per_day = tweets_count; date-only
    arithmetic (the time of day is dropped)."""
    import pandas as pd

    import bb.data

    ts = pd.to_datetime(["2022-04-19 23:59:59", None, "2022-04-20 08:00:00", "2010-01-01 12:00:00"], utc=True)
    df = pd.DataFrame({
        "idx": range(4),
        "name": pd.array(["ab", None, "", "é😀"], dtype="string"),
        "description": pd.array([None, "x y", "", "abc"], dtype="string"),
        "created_at": ts,
        "following_count": pd.array([10, None, 3, 7], dtype="Int64"),
        "followers_count": pd.array([0, 5, None, 2], dtype="Int64"),
        "statuses_count": pd.array([30, 40, None, 365], dtype="Int64")})
    monkeypatch.setattr(bb.data, "load_profile", lambda ds, columns=None, **k: df[["idx", *columns]])
    X = F.profile_features("twibot22")
    days = [1, 0, 0, (pd.Timestamp("2022-04-20") - pd.Timestamp("2010-01-01")).days]
    exp = np.array([[2, 0, 1, 10, 0, 0.0, 30, 30.0],
                    [0, 3, 0, 0, 5, 0.0, 40, 40.0],
                    [0, 0, 0, 3, 0, 0.0, 0, 0.0],
                    [2, 3, days[3], 7, 2, 3.5, 365, 365 / days[3]]])
    assert np.allclose(X, exp, rtol=0, atol=1e-12)
    # non-TwiBot-22 datasets use 2020-10-01
    assert F.profile_features("twibot20")[3, 2] == (pd.Timestamp("2020-10-01") - pd.Timestamp("2010-01-01")).days


def test_select_rows_orders(tw22, tmp_path):
    from bb.build import twibot22
    from bb.data import TweetStore

    twibot22.build_tweets(files=tw22["files"], n_buckets=3, flush_rows=7, checkpoint_bytes=2000, cap=6)
    st = TweetStore("twibot22")
    users = np.arange(st.n_users)
    r_rel, o_rel = F.select_rows(st, users, 3, "release")
    assert np.array_equal(np.diff(o_rel), np.minimum(np.diff(np.asarray(st.offsets)), 3))
    r_new, o_new = F.select_rows(st, users, 3, "newest")
    assert np.array_equal(o_new, o_rel)
    ca = np.asarray(st.column("created_at"))
    for j in range(len(users)):
        sel = r_new[o_new[j]:o_new[j + 1]]
        allr = np.arange(st.offsets[j], st.offsets[j + 1])
        assert list(ca[sel]) == sorted(ca[allr], reverse=True)[:len(sel)]
    a = F.select_rows(st, users, 3, "random", seed=5)
    b = F.select_rows(st, users, 3, "random", seed=5)
    c = F.select_rows(st, users[::-1].copy(), 3, "random", seed=5)  # per-user seeding: order-free
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    for j in range(len(users)):
        k = len(users) - 1 - j
        assert np.array_equal(a[0][a[1][j]:a[1][j + 1]], c[0][c[1][k]:c[1][k + 1]])


def test_content_pass_pool_equals_serial_and_caches(cr15):
    fake_assets()
    from bb.data import load_split

    users = np.concatenate([np.asarray(load_split("cresci15", s)) for s in ("train", "val", "test")])
    a, meta, path = F.content_stats("cresci15", users, 200, n_workers=1, log=lambda *_: None)
    a = np.array(a)
    b, _, _ = F.content_stats("cresci15", users, 200, n_workers=2, recompute=True, tweet_budget=3,
                              log=lambda *_: None)
    assert np.array_equal(a, np.asarray(b), equal_nan=True)
    _, meta2, _ = F.content_stats("cresci15", users, 200, log=lambda *_: None)  # cache hit
    assert meta2["key"] == meta["key"] == "cap200_release"
    # a cache written by another statistics version is stale and recomputed
    msgs = []
    mpath = F.cache_paths("cresci15", meta["key"])[1]
    write_json(mpath, {**meta2, "content_version": F.CONTENT_VERSION + 1})
    F.content_stats("cresci15", users, 200, n_workers=1, log=msgs.append)
    assert any("stale" in m for m in msgs)
    n = a[users, 0]
    assert (n == 0).any() and (n > 0).any()        # fixture has zero-tweet users
    sim = a[users, 7]
    assert np.isnan(sim[n == 0]).all() and (sim[n == 1] == 1.0).all()
    X = F.content_columns(a[users])
    assert np.isnan(X[n == 0]).all() and np.isfinite(X[n > 0]).all()


def _main(args):
    from bb.methods.lee2011.main import main

    return main(args + ["--n-jobs", "1", "--workers", "1"])


def test_end_to_end_drop_fill_and_extscore(cr15):
    fake_assets()
    from bb.data import load_split
    from bb.extscore import make_synthetic_extscore

    tr = np.asarray(load_split("cresci15", "train"))
    base = _main(["--dataset", "cresci15", "--seeds", "0"])
    r = base[0]
    assert r["status"] == "ok" and len(r["features"]) == 14
    sc = r["split_counts"]
    assert sum(v["n_dropped_zero_tweets"] for v in sc.values()) > 0
    assert sc["test"]["n_used"] == r["metrics"]["test"]["n"]
    again = _main(["--dataset", "cresci15", "--seeds", "0", "--extscore", "none"])[0]
    assert again["metrics"]["test"] == r["metrics"]["test"]  # extscore none reproduces the baseline
    fill = _main(["--dataset", "cresci15", "--seeds", "0", "--zero-tweet-policy", "fill"])[0]
    assert fill["metrics"]["test"]["n"] == sc["test"]["n_official"]
    # ExtScore plumbing with a synthetic column (never a result)
    make_synthetic_extscore("cresci15", "synthetic", seed=0)
    j = _main(["--dataset", "cresci15", "--seeds", "0", "--with-extscore", "synthetic"])[0]
    assert j["features"][-2:] == ["extscore_p", "extscore_missing"] and len(j["features"]) == 16
    assert j["max_features_effective"] == 4 and r["max_features_effective"] == 3
    assert j["extscore"]["coverage"]["train"]["n"] == sc["train"]["n_used"]
    assert "permutation_val_extscore_p" in j["extscore"]["importance"]
    j3 = _main(["--dataset", "cresci15", "--seeds", "0", "--with-extscore", "synthetic", "--max-features", "3"])[0]
    assert j3["max_features_effective"] == 3
    jn = _main(["--dataset", "cresci15", "--seeds", "0", "--with-extscore", "synthetic", "--extscore-transform", "nan"])[0]
    assert jn["features"][-1] == "extscore_p" and len(jn["features"]) == 15
    jl = _main(["--dataset", "cresci15", "--seeds", "0", "--with-extscore", "synthetic", "--extscore-fusion", "late"])[0]
    assert len(jl["features"]) == 14 and "late_fusion" in jl["extscore"]
    assert len(tr) > 0


def test_no_label_leak_from_eval_splits(cr15):
    """Test predictions must not depend on val/test labels (features are the user's own data)."""
    fake_assets()
    import argparse

    from bb.data import load_split, load_users
    from bb.extscore import prepare_extscore
    from bb.methods.lee2011 import main as M

    args = argparse.Namespace(dataset="cresci15", col0="name", reference_date=None, tweets_per_user=200,
                              tweet_order="release", tweet_seed=0, workers=1, tweet_budget=20_000,
                              recompute=False, zip_overhead=45, zero_tweet_policy="drop", n_estimators=10,
                              max_features="sqrt", n_jobs=1, extscore_transform="tree", extscore_fusion="early",
                              variant="leak", extscore="none", extscore_tag=None, with_extscore=None, extscore_unsafe=False)
    X, names, has, cmeta, info = M.build_matrix(args, log=lambda *_: None)
    y = load_users("cresci15", columns=["label"])["label"].to_numpy()
    sp = {s: np.asarray(load_split("cresci15", s)) for s in ("train", "val", "test")}
    sp = {s: v[has[v]] for s, v in sp.items()}
    J = prepare_extscore(None, "cresci15", sp)
    a = M.run_seed(args, 0, X, names, y, sp, J, {}, [])
    y2 = y.copy()
    ev = np.concatenate([sp["val"], sp["test"]])
    y2[ev] = 1 - y2[ev]
    b = M.run_seed(args, 0, X, names, y2, sp, J, {}, [])
    pa = np.load(os.path.join(os.environ["BB_RUNS_ROOT"], "lee2011", "cresci15", a["run"], "preds_test.npz"))["prob"]
    pb = np.load(os.path.join(os.environ["BB_RUNS_ROOT"], "lee2011", "cresci15", b["run"], "preds_test.npz"))["prob"]
    assert np.array_equal(pa, pb)
