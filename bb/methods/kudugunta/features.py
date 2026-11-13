"""Account features of the TwiBot-22 benchmark's Kudugunta re-implementation
(``src/Kudugunta/train.py`` L81-100, L159-169; kudugunta.md 3.2).

Columns, raw and unscaled (float64, as the benchmark's ``np.zeros``):

    0-3  followers_count, following_count, tweet_count (= statuses_count), listed_count
    4-5  protected, verified: 0 if absent/None/'False'/0 (so bool False -> 0), else 1
    6    TwiBot-22 only: profile_image_url == '' -> 0, else 1 (null -> 1, as ``None != ''``)
    7    TwiBot-22 only: location is None -> 0, else 1 ('' -> 1)

Missing counts (TwiBot-20: 96 users without a profile) -> 0; missing booleans -> 0.

Emulations (off by default; kudugunta.md 9.1 and 3.2):
- ``bench_misalign`` (TwiBot-22): the benchmark writes the counts to row ``dict[id]`` (position in
  split.csv grouped train/val/test) but the boolean columns to row ``i`` (position in user.json),
  with labels at ``dict[id]``. So the user at benchmark position r gets the booleans of the user at
  raw position r. We rebuild that permutation from ``split``, ``split_csv_order`` and ``raw_order``.
- ``tw20_literal_bool_strings`` (TwiBot-20): the raw booleans are ``"False "``/``"True "``; if the
  benchmark's node.json kept them, ``'False ' != 'False'`` makes columns 4-5 equal 1 for every user
  with a profile.
"""
from __future__ import annotations

import numpy as np

from bb.data import load_profile, load_users

COUNTS = ["followers_count", "following_count", "statuses_count", "listed_count"]
NAMES = ["followers_count", "following_count", "tweet_count", "listed_count", "protected", "verified",
         "profile_image_url", "location"]
BOOL_START = 4


def n_features(dataset: str) -> int:
    return 8 if dataset == "twibot22" else 6


def bench_positions(split: np.ndarray, split_csv_order: np.ndarray) -> np.ndarray:
    """Benchmark row of each user: train users in split.csv order get 0..j-1, then val, then test
    (``train.py`` L121-140)."""
    split = np.asarray(split).astype(str)
    order = np.asarray(split_csv_order, np.int64)
    if (order < 0).any():
        raise ValueError("split_csv_order missing for some users; the misalignment emulation needs it")
    pos = np.full(len(split), -1, np.int64)
    off = 0
    for s in ("train", "val", "test"):
        ix = np.flatnonzero(split == s)
        ix = ix[np.argsort(order[ix], kind="stable")]
        pos[ix] = off + np.arange(len(ix))
        off += len(ix)
    if (pos < 0).any():
        raise ValueError("users outside train/val/test; the benchmark index assumes all users are labelled")
    return pos


def misalign_permutation(dataset: str) -> np.ndarray:
    """``src[u]`` = idx of the user whose booleans user ``u`` receives in the benchmark."""
    u = load_users(dataset, columns=["split", "raw_order", "split_csv_order"])
    pos = bench_positions(u["split"].to_numpy(), u["split_csv_order"].to_numpy())
    raw = u["raw_order"].to_numpy().astype(np.int64)
    inv_raw = np.full(len(raw), -1, np.int64)
    if raw.min() < 0 or raw.max() >= len(raw) or len(np.unique(raw)) != len(raw):
        raise ValueError("raw_order is not a permutation of 0..N-1")
    inv_raw[raw] = np.arange(len(raw))
    return inv_raw[pos]


def _bool01(p, col: str) -> np.ndarray:
    return p[col].astype("boolean").fillna(False).to_numpy().astype(np.float64)


def build_features(dataset: str, *, bench_misalign: bool = False, tw20_literal_bool_strings: bool = False,
                   profile=None) -> tuple[np.ndarray, list[str], dict]:
    """[N, 6|8] float64 features for every idx, names, and an info dict."""
    tw22 = dataset == "twibot22"
    if bench_misalign and not tw22:
        raise ValueError("--bench-misalign is defined for TwiBot-22 only (split.csv order vs user.json order)")
    if tw20_literal_bool_strings and dataset != "twibot20":
        raise ValueError("--tw20-literal-bool-strings applies to TwiBot-20 only")
    cols = COUNTS + ["protected", "verified"]
    if tw20_literal_bool_strings:
        cols += ["raw_protected", "raw_verified"]
    if tw22:
        cols += ["profile_image_url", "location"]
    p = profile if profile is not None else load_profile(dataset, cols)
    out = [p[c].astype("Float64").fillna(0).to_numpy(dtype=np.float64) for c in COUNTS]
    info = {"n_missing_counts": int(p[COUNTS[0]].isna().sum())}
    for c in ("protected", "verified"):
        if tw20_literal_bool_strings:
            raw = p[f"raw_{c}"]
            v = ~(raw.isna() | (raw == "False")).to_numpy(dtype=bool, na_value=True)
            out.append(v.astype(np.float64))
        else:
            out.append(_bool01(p, c))
    if tw22:
        img = p["profile_image_url"]
        out.append((~(img == "").fillna(False)).to_numpy(dtype=bool).astype(np.float64))
        out.append((~p["location"].isna()).to_numpy(dtype=bool).astype(np.float64))
    X = np.stack(out, 1)
    names = NAMES[:X.shape[1]]
    if bench_misalign:
        src = misalign_permutation(dataset)
        X[:, BOOL_START:] = X[src, BOOL_START:]
        info["misalign_same_user"] = int((src == np.arange(len(src))).sum())
    info["column_means"] = dict(zip(names, map(float, X.mean(0))))
    return X, names, info
