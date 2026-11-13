"""Builders on synthetic raw releases: index order, splits, profile typing (null vs empty, raw copies),
tweet store order/caps/flags, CSR graphs and benchmark edge order, resumability."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bb.data import (TweetStore, load_csr, load_edge_index, load_labels, load_profile, load_split,
                     load_users)
from bb.io import read_json
from bb.paths import dataset_paths


# ------------------------------------------------------------------------------------- TwiBot-20
def test_twibot20_index_and_splits(tw20):
    u = load_users("twibot20")
    assert len(u) == tw20["n"] == 19
    assert u["split"].astype(str).tolist() == ["train"] * 6 + ["val"] * 3 + ["test"] * 2 + ["support"] * 8
    assert u["source_file"].astype(str).tolist()[:6] == ["train_bak.json"] * 6  # never train.json
    assert (u["label"].to_numpy()[11:] == -1).all() and set(u["label"].to_numpy()[:11]) <= {0, 1}
    assert np.array_equal(load_split("twibot20", "val"), np.arange(6, 9))
    assert isinstance(load_split("twibot20", "train"), np.memmap)
    assert u["user_id"].tolist()[0] == "1000"


def test_twibot20_profile_typing(tw20):
    p = load_profile("twibot20")
    assert p.loc[0, "name"] == "Name 0" and p.loc[0, "raw_name"] == "Name 0 "
    assert pd.isna(p.loc[1, "location"]) and p.loc[1, "raw_location"] == "None "
    assert pd.isna(p.loc[3, "description"])            # "None " -> null
    assert p.loc[4, "description"] == ""               # " " -> '' (not null)
    assert bool(p.loc[5, "profile_missing"]) and pd.isna(p.loc[5, "followers_count"])
    assert p.loc[2, "followers_count"] == 20 and str(p["followers_count"].dtype) == "Int64"
    assert bool(p.loc[0, "verified"]) is True and bool(p.loc[1, "verified"]) is False
    assert p.loc[0, "raw_default_profile_image"] == "True "
    assert p.loc[0, "created_at"] == pd.Timestamp("2008-11-18 10:27:00", tz="UTC")
    assert pd.isna(p.loc[0, "utc_offset"])


def test_twibot20_tweets(tw20):
    t = TweetStore("twibot20")
    assert t.n_users == 19
    for i, tw in enumerate(tw20["tweets"]):
        assert t.user_texts(i) == (tw or [])
    assert t.tweet_list_null()[2] and not t.tweet_list_null()[0]
    assert np.array_equal(np.asarray(t.column("release_rank"))[t.user_rows(3)], np.arange(3))


def test_twibot20_graph(tw20):
    for rel, key in (("following", "edges_following"), ("follower", "edges_follower")):
        c = load_csr("twibot20", rel)
        rows, cols = c.coo()
        assert sorted(zip(rows.tolist(), cols.tolist())) == sorted(tw20[key])  # unknown id dropped
    ei, et = load_edge_index("twibot20")
    gm = read_json(dataset_paths("twibot20").graph / "meta.json")
    assert gm["edge_type_ids"] == {"following": 0, "follower": 1}
    # Dataset.py order: node by node, following before follower
    pairs = list(zip(ei[0].tolist(), et.tolist()))
    assert pairs == sorted(pairs, key=lambda x: (x[0], x[1]))
    assert gm["relations"]["following"]["dropped_outside_node_set"] == len(tw20["edges_following"])


# ------------------------------------------------------------------------------------- TwiBot-22
def test_twibot22_users_profile(tw22):
    u = load_users("twibot22")
    assert u["raw_id"].tolist() == tw22["ids"] and u["user_id"].tolist()[0] == "5000"
    assert u["split"].astype(str).tolist() == tw22["split"]          # mapped through split.csv ids
    assert load_labels("twibot22").tolist() == [int(x == "bot") for x in tw22["label"]]
    assert (u["split_csv_order"].to_numpy() == np.arange(12)[::-1]).all()
    p = load_profile("twibot22", ["description", "favourites_count", "following_count"])
    assert pd.isna(p.loc[1, "description"]) and p.loc[2, "description"] == ""
    assert p["favourites_count"].isna().all()                          # absent in API v2: null, not 0
    m = read_json(dataset_paths("twibot22").profile.with_suffix(".meta.json"))
    assert "favourites_count" in m["absent_columns"]


def test_twibot22_graph(tw22):
    from bb.build import twibot22
    twibot22.build_graph()
    for rel in ("following", "followers"):
        c = load_csr("twibot22", rel)
        rows, cols = c.coo()
        got = sorted(zip(rows.tolist(), cols.tolist(), np.asarray(c.edge_pos).tolist()))
        assert got == sorted(tw22["follow"][rel])
    ei, et = load_edge_index("twibot22")
    pos = np.load(dataset_paths("twibot22").graph / "edge_pos.i64.npy")
    assert (np.diff(pos) > 0).all()                                    # file order
    assert set(np.unique(et)) == {0, 1}
    gm = read_json(dataset_paths("twibot22").graph / "meta.json")
    assert gm["relations"]["following"]["dropped_non_user_endpoints"] == 1
    assert gm["all_relation_counts"]["post"] == 12


def test_twibot22_tweets_cap_order_flags(tw22, tmp_path):
    from bb.build import twibot22
    m = twibot22.build_tweets(files=tw22["files"], n_buckets=3, flush_rows=7, checkpoint_bytes=2000, cap=6)
    t = TweetStore("twibot22")
    nt = np.asarray(t.n_total())
    for i, lst in tw22["tweets"].items():
        assert nt[i] == len(lst)
        kept = lst[:6]
        assert [r for r, _ in kept] == np.asarray(t.column("release_rank"))[t.user_rows(i)].tolist()
        assert t.user_texts(i) == [x or "" for _, x in kept]
    assert m["stats"]["non_user_author"] == 2 and m["tweets_beyond_cap"] == int(np.maximum(nt - 6, 0).sum())
    rows = np.arange(t.n_tweets)
    texts = t.texts(rows)
    rt = np.asarray(t.column("is_retweet"))
    assert all((x.startswith("RT @")) == bool(f) for x, f in zip(texts, rt))
    assert (np.asarray(t.column("is_quote")) == -1).all()              # referenced_tweets null -> unknown
    assert (np.asarray(t.column("reply_count")) == -1).all()           # JSON null -> -1
    assert np.asarray(t.column("text_null")).sum() == 1


def test_twibot22_tweets_resume_equivalence(tw22, tmp_path):
    """Kill mid-pass (simulated by a corrupt second file), fix the file, rerun: identical store."""
    from bb.build import twibot22
    ref = tmp_path / "ref"
    twibot22.build_tweets(files=tw22["files"], out_dir=ref, work_dir=tmp_path / "w1", n_buckets=3,
                          flush_rows=5, checkpoint_bytes=500, cap=6)
    good = tw22["files"][1].read_bytes()
    tw22["files"][1].write_bytes(good[: len(good) // 2])               # truncated -> fails mid-file
    with pytest.raises(ValueError):
        twibot22.build_tweets(files=tw22["files"], out_dir=tmp_path / "res", work_dir=tmp_path / "w2",
                              n_buckets=3, flush_rows=5, checkpoint_bytes=500, cap=6)
    ck = json.loads((tmp_path / "w2" / "checkpoint.json").read_text())
    assert ck["file_i"] == 1 and ck["byte_off"] > 0                    # resumes inside file 1
    tw22["files"][1].write_bytes(good)
    twibot22.build_tweets(files=tw22["files"], out_dir=tmp_path / "res", work_dir=tmp_path / "w2",
                          n_buckets=3, flush_rows=5, checkpoint_bytes=500, cap=6)
    for f in sorted(p.name for p in ref.iterdir()):
        if f != "meta.json":
            assert (ref / f).read_bytes() == (tmp_path / "res" / f).read_bytes(), f


# ------------------------------------------------------------------------------------- Cresci-2015
def test_cresci15(cr15):
    from bb.build.cresci15 import SUBSET_ORDER
    u = load_users("cresci15")
    exp_ids = [i for s in SUBSET_ORDER for i in cr15["ids"][s]]
    assert u["user_id"].tolist() == exp_ids
    assert u["label"].tolist() == [0] * 14 + [1] * 14
    # per-subset contiguous 70/20/10
    e13 = u["split"].astype(str).to_numpy()[:10].tolist()
    assert e13 == ["train"] * 7 + ["val"] * 2 + ["test"]
    p = load_profile("cresci15", ["description", "default_profile", "geo_enabled", "url"])
    assert p.loc[0, "description"] == "" and pd.isna(p.loc[1, "description"])  # "" vs unquoted empty
    assert bool(p.loc[0, "default_profile"]) and pd.isna(p.loc[0, "geo_enabled"])
    t = TweetStore("cresci15")
    assert t.counts().tolist() == [i % 3 for s in SUBSET_ORDER for i in range(cr15["sizes"][s])]
    assert t.user_texts(1) == [f"tw {exp_ids[1]} 0"]
    assert np.asarray(t.column("like_count"))[0] == 2 and np.asarray(t.column("is_retweet")).sum() == 0
    c = load_csr("cresci15", "friend")
    assert c.nnz == 28                                                  # the "42" endpoint dropped
