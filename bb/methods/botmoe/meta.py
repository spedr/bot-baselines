"""Metadata-expert inputs (docs/methods/botmoe.md 3.1-3.3).

- ``botrgcn8`` (default, the paper's metadata): concat(botrgcn num [k], cat [c]) (8 columns on
  TwiBot-20/22, 6 on Cresci-2015, whose scheme has one flag).
- ``hayawi9``: the 9 Hayawi/DeeProBot columns the released code loads as ``num_for_h`` (spec 3.1
  table), z-scored over labelled users (ddof 1), correctly aligned to our node index.
  ``hayawi_bugs=True`` keeps the two bugs of the TwiBot-22 benchmark Hayawi preprocessing
  ("tweet_freq" = listed_count / active_days; "entropy" = sum_c -p_c^2 log2 p_c); False uses
  statuses_count / active_days and Shannon entropy.
- ``zeros`` (emulation of the released pipeline for labelled users): a [N, 9] zero matrix. In the
  released code ``num_for_h = cat(zeros(N - 11826, 9), num_for_h)`` puts the real rows on support
  nodes, so every labelled node's metadata expert sees zeros and outputs a near-constant token.
- ``released`` (TwiBot-20 only; exact released layout): ``hayawi9`` (with bugs) rows of the 11,826
  labelled users placed at node rows N-11826..N-1 (support users), zeros everywhere else.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

META_INPUTS = ("botrgcn8", "hayawi9", "zeros", "released")
HAYAWI_COLS = ("followers_count", "tweet_freq", "screen_name_length", "following_count", "statuses_count",
               "listed_count", "name_bigram_freq", "name_entropy", "des_entropy")
REF_DATE = {"twibot20": "2020-09-05", "twibot22": "2020-09-05", "cresci15": "2020-09-01"}


def bigram_freq(s: str) -> float:
    """Hayawi ``bigrams_freq``: (#character bigrams) / (#unique bigrams), 0 for length <= 1."""
    if len(s) <= 1:
        return 0.0
    bg = [s[i:i + 2] for i in range(len(s) - 1)]
    return len(bg) / len(set(bg))


def entropy(s: str, buggy: bool = True) -> float:
    """``buggy``: sum_c -p_c^2 log2 p_c (what the per-position-sum / length code computes);
    else Shannon entropy -sum_c p_c log2 p_c. Empty string -> 0."""
    if not s:
        return 0.0
    n = len(s)
    tot = 0.0
    for cnt in Counter(s).values():
        p = cnt / n
        tot -= (p * p if buggy else p) * math.log2(p)
    return tot


def hayawi9(dataset: str, labels: np.ndarray, bugs: bool = True) -> np.ndarray:
    import pandas as pd

    from ...data import load_profile

    p = load_profile(dataset, ["followers_count", "following_count", "statuses_count", "listed_count",
                               "created_at", "screen_name", "name", "description"])
    num = {c: p[c].astype("float64").fillna(0).to_numpy() for c in
           ("followers_count", "following_count", "statuses_count", "listed_count")}
    days = (pd.Timestamp(REF_DATE[dataset], tz="UTC") - p["created_at"]).dt.days.astype("float64")
    days = np.maximum(days.fillna(1).to_numpy(), 1.0)
    txt = {c: p[c].astype("string").fillna("").tolist() for c in ("screen_name", "name", "description")}
    cols = [num["followers_count"],
            (num["listed_count"] if bugs else num["statuses_count"]) / days,
            np.array([len(s) for s in txt["screen_name"]], np.float64),
            num["following_count"], num["statuses_count"], num["listed_count"],
            np.array([bigram_freq(s) for s in txt["name"]], np.float64),
            np.array([entropy(s, bugs) for s in txt["name"]], np.float64),
            np.array([entropy(s, bugs) for s in txt["description"]], np.float64)]
    x = np.stack(cols, axis=1)
    lab = np.asarray(labels) >= 0
    mu = x[lab].mean(0)
    sd = x[lab].std(0, ddof=1)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return ((x - mu) / sd).astype(np.float32)


def meta_input(kind: str, F, *, hayawi_bugs: bool = True) -> tuple[np.ndarray, dict]:
    """[N, m] float32 metadata-expert input and a small description for the run config."""
    n = F.n
    if kind == "botrgcn8":
        x = np.concatenate([np.asarray(F.num_prop, np.float32), np.asarray(F.cat_prop, np.float32)], axis=1)
        return x, {"meta_input": kind, "columns": list(F.meta.get("num_columns", [])) + list(F.meta.get("cat_columns", []))}
    if kind == "zeros":
        return np.zeros((n, 9), np.float32), {"meta_input": kind, "note": "released behaviour for labelled nodes"}
    if kind == "hayawi9":
        return hayawi9(F.dataset, F.label, hayawi_bugs), {"meta_input": kind, "columns": list(HAYAWI_COLS),
                                                          "hayawi_bugs": hayawi_bugs}
    if kind == "released":
        if F.dataset != "twibot20":
            raise ValueError("meta_input 'released' reproduces the TwiBot-20 index misalignment only")
        h = hayawi9(F.dataset, F.label, True)
        lab = np.flatnonzero(np.asarray(F.label) >= 0)
        if not np.array_equal(lab, np.arange(len(lab))):
            raise ValueError("expected labelled TwiBot-20 users first in the index")
        x = np.zeros((n, 9), np.float32)
        x[n - len(lab):] = h[lab]
        return x, {"meta_input": kind, "note": "num_for_h = cat(zeros(N-11826, 9), hayawi9[labelled])"}
    raise ValueError(f"meta_input must be one of {META_INPUTS}")
