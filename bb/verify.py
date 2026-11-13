"""Post-build verification of shared artifacts (``bb verify <dataset>``).

Checks invariants that every consumer relies on (row alignment, offsets monotone, CSR shapes,
split sizes) and, where an external reference exists, published counts:
- TwiBot-20: 8,278/2,365/1,183/217,754 users; 227,979 follow edges; 33,488,192 tweets.
- TwiBot-22: 700k/200k/100k users with 54,586/55,913/29,444 bots; 2,626,979 following and
  1,116,655 followers edges.
- Cresci-2015: 5,301 users (1,950 human / 3,351 bot); row order, split and edges cross-checked
  against the BotRGCN-format tensors of the MGTAB repo (``verify_against_mgtab``).
"""
from __future__ import annotations

import io
import zipfile

import numpy as np

from .paths import dataset_paths, raw_dir

EXPECTED = {
    "twibot20": {"splits": {"train": 8278, "val": 2365, "test": 1183, "support": 217754},
                 "edges": {"following": 117110, "follower": 110869}, "edges_total": 227979,
                 "tweets": 33488192},
    "twibot22": {"splits": {"train": 700000, "val": 200000, "test": 100000},
                 "bots": {"train": 54586, "val": 55913, "test": 29444},
                 "edges": {"following": 2626979, "followers": 1116655}},
    "cresci15": {"splits": {"train": 3708, "val": 1058, "test": 535}, "n": 5301, "bots": 3351},
}


def _check(ok: bool, msg: str, out: list) -> bool:
    out.append(("OK  " if ok else "FAIL") + " " + msg)
    return ok


def verify(dataset: str, verbose: bool = True) -> bool:
    from .data import TweetStore, load_csr, load_users, relations
    from .io import read_json

    dp = dataset_paths(dataset)
    exp = EXPECTED.get(dataset, {})
    out: list[str] = []
    ok = True
    users = load_users(dataset)
    n = len(users)
    ok &= _check(np.array_equal(users["idx"].to_numpy(), np.arange(n)), f"users.idx dense 0..{n - 1}", out)
    split = users["split"].astype(str).to_numpy()
    for s, k in exp.get("splits", {}).items():
        ok &= _check(int((split == s).sum()) == k, f"split {s}: {(split == s).sum()} (expected {k})", out)
    for s, k in exp.get("bots", {}).items() if isinstance(exp.get("bots"), dict) else []:
        b = int(((split == s) & (users["label"].to_numpy() == 1)).sum())
        ok &= _check(b == k, f"bots in {s}: {b} (expected {k})", out)
    import pyarrow.parquet as pq
    if dp.profile.exists():
        pn = pq.ParquetFile(dp.profile).metadata.num_rows
        ok &= _check(pn == n, f"profile rows {pn} == users {n}", out)
    if (dp.tweets / "meta.json").exists():
        t = TweetStore(dataset)
        off = np.asarray(t.offsets)
        ok &= _check(len(off) == n + 1 and off[0] == 0 and bool(np.all(np.diff(off) >= 0)),
                     "tweets offsets [N+1] monotone", out)
        ok &= _check(len(t.text_offsets) == t.n_tweets + 1 and int(t.text_offsets[-1]) == t.blob.shape[0],
                     f"text offsets cover text.bin ({t.n_tweets} tweets, {t.blob.shape[0] / 1e9:.2f} GB)", out)
        for c in t.columns:
            ok &= _check(t.column(c).shape[0] == t.n_tweets, f"tweet column {c} length", out)
        if "tweets" in exp:
            ok &= _check(t.n_tweets == exp["tweets"], f"tweets {t.n_tweets} (expected {exp['tweets']})", out)
    if (dp.graph / "meta.json").exists():
        tot = 0
        for r in relations(dataset):
            c = load_csr(dataset, r)
            tot += c.nnz
            good = c.n == n and len(c.indices) == c.nnz and (c.nnz == 0 or int(np.max(c.indices)) < n)
            ok &= _check(good, f"CSR {r}: n={c.n} nnz={c.nnz}", out)
            if r in exp.get("edges", {}):
                ok &= _check(c.nnz == exp["edges"][r], f"edges {r}: {c.nnz} (expected {exp['edges'][r]})", out)
        ei = np.load(dp.graph / "edge_index.i64.npy", mmap_mode="r")
        ok &= _check(ei.shape == (2, tot), f"edge_index shape {ei.shape} == (2, {tot})", out)
    if dataset == "cresci15":
        ok &= verify_against_mgtab(out)
    if verbose:
        print("\n".join(out))
    return bool(ok)


def verify_against_mgtab(out: list | None = None) -> bool:
    """Compare our Cresci-2015 index with the MGTAB-repo BotRGCN tensors: labels per row, numeric
    columns (followers, active_days@2020-09-01, len(screen_name), friends, listed; z-scored over
    all rows, ddof=1), split index sets, and the user-user edge count."""
    import pandas as pd
    import torch

    from .data import load_profile, load_split, load_users

    out = out if out is not None else []
    z = zipfile.ZipFile(raw_dir("mgtab_cresci15") / "Cresci-15.zip")
    T = {n.split("/")[-1][:-3]: torch.load(io.BytesIO(z.read(n)), weights_only=False)
         for n in z.namelist() if n.endswith(".pt")}
    users = load_users("cresci15")
    ok = True
    lab = users["label"].to_numpy()
    ok &= _check(np.array_equal(lab, T["label"].numpy()), "MGTAB labels equal row by row", out)
    p = load_profile("cresci15", ["followers_count", "following_count", "listed_count", "screen_name", "created_at"])
    ref = pd.Timestamp("2020-09-01", tz="UTC")
    days = (ref - p["created_at"]).dt.days
    cols = [p["followers_count"].astype(float), days.astype(float),
            p["screen_name"].str.len().astype(float), p["following_count"].astype(float),
            p["listed_count"].astype(float)]
    num = np.stack([((c - c.mean()) / c.std(ddof=1)).fillna(0).to_numpy() for c in cols], 1)
    mg = T["num_properties_tensor"].numpy()
    err = np.abs(num - mg).max(0)
    ok &= _check(bool((err < 1e-3).all()), f"MGTAB num_properties match row by row (max abs err per col {np.round(err, 5).tolist()})", out)
    for s in ("train", "val", "test"):
        a = np.sort(np.asarray(load_split("cresci15", s)))
        b = np.sort(T[f"{s}_idx"].numpy())
        ok &= _check(np.array_equal(a, b), f"MGTAB {s}_idx equals our {s} split ({len(a)} vs {len(b)})", out)
    from .data import load_csr
    e = sum(load_csr("cresci15", r).nnz for r in ("friend", "follower"))
    ei = T["edge_index"].numpy()
    et = T["edge_type"].numpy()
    from .data import load_edge_index
    oi, ot = load_edge_index("cresci15")
    k_m = np.sort(ei[0] * 10**8 + ei[1] * 10 + et)
    k_o = np.sort(np.asarray(oi[0]) * 10**8 + np.asarray(oi[1]) * 10 + np.asarray(ot))
    ok &= _check(np.array_equal(k_m, k_o), f"MGTAB (src, dst, type) edge multiset equals ours ({ei.shape[1]} vs {e})", out)
    out.append("INFO MGTAB edge order equals ours: " + str(bool(np.array_equal(ei, np.asarray(oi)))))
    return bool(ok)
