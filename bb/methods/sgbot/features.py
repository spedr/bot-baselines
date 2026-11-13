"""SGBot / BotometerLite feature vector (docs/methods/sgbot.md section 3).

``benchmark`` mode reproduces the 20 columns of the TwiBot-22 re-implementation
(``src/SGBot/preprocess.py`` L100-125): slots 3 (favourites), 6 (profile_use_background_image) and
11 (favourites growth) are constant 0, slot 5 is the custom-profile-image flag instead of
``default_profile``, and the follower/friend ratio is 0 when friends = 0. ``paper`` mode restores
favourites / default_profile / profile_use_background_image where the profile has them (TwiBot-20,
Cresci-2015) and uses followers / max(friends, 1) everywhere.

Screen-name likelihood (benchmark ``utils.py``): bigram counts over the 63-character alphabet
``[A-Za-z0-9_]`` from every user's ``screen_name.strip()`` (``all_users``: all nodes of the dataset
incl. test and support - label-free transduction, the benchmark default; ``train_users``: train
split only), p = count / total without smoothing, likelihood = prod p^(1/(len-1)), 1.0 when
len <= 1, 0 when any bigram is unseen, 0 for a null name. Bigrams with a character outside the
alphabet would raise KeyError in the benchmark; we skip them (no screen name in our three
datasets has one) and count them in the run info.
"""
from __future__ import annotations

import string

import numpy as np
import pandas as pd

from bb.data import load_profile, profile_meta

from ..bothunter.shared import (REF_YEAR, account_age_days, bool_col, codepoints, count_col,
                                custom_image_flag, digit_count, safe_div_age, strip_len)

ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits + "_"
A = len(ALPHABET)  # 63
_LUT = np.full(128, -1, np.int64)
_LUT[[ord(c) for c in ALPHABET]] = np.arange(A)

NAMES = ["statuses_count", "followers_count", "friends_count", "favourites_count", "listed_count",
         "default_profile", "profile_use_background_image", "verified", "tweet_freq", "followers_growth",
         "friends_growth", "favourites_growth", "listed_growth", "followers_friends_ratio",
         "screen_name_len", "screen_name_digits", "name_len", "name_digits", "description_len",
         "screen_name_likelihood"]
PROFILE_COLS = ["screen_name", "name", "description", "profile_image_url", "verified", "created_at",
                "statuses_count", "followers_count", "following_count", "listed_count"]
PAPER_COLS = ["favourites_count", "default_profile", "profile_use_background_image"]


def _bigram_codes(s: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per string: bigram codes a*63+b [N, L-1] (-1 = invalid / padding), lengths, n_invalid."""
    cp, lens = codepoints(s)
    ai = np.where(cp < 128, _LUT[np.minimum(cp, 127)], -1)
    ai[cp == 0] = -1
    a, b = ai[:, :-1], ai[:, 1:]
    ok = (a >= 0) & (b >= 0)
    code = np.where(ok, a * A + b, -1)
    in_len = np.arange(cp.shape[1] - 1)[None, :] < (lens[:, None] - 1)
    n_invalid = int((in_len & ~ok).sum())
    return code, lens, n_invalid


def bigram_table(names: pd.Series) -> tuple[np.ndarray, dict]:
    """Bigram probabilities [63*63] from the given screen names (stripped; nulls skipped)."""
    code, _, n_invalid = _bigram_codes(names.dropna().str.strip())
    counts = np.bincount(code[code >= 0], minlength=A * A).astype(np.float64)
    total = counts.sum()
    prob = counts / total if total else counts
    return prob, {"corpus_names": int(names.notna().sum()), "corpus_bigrams": int(total),
                  "distinct_bigrams_seen": int((counts > 0).sum()), "invalid_bigrams_skipped": n_invalid}


def likelihood(names: pd.Series, prob: np.ndarray) -> tuple[np.ndarray, dict]:
    """Geometric-mean bigram likelihood of each screen name (benchmark definition)."""
    null = names.isna().to_numpy()
    code, lens, n_invalid = _bigram_codes(names.str.strip())
    valid = code >= 0
    with np.errstate(divide="ignore"):
        logp = np.log(prob)
    lp = np.where(valid, logp[np.maximum(code, 0)], 0.0)
    unseen = np.any(valid & np.isneginf(lp), axis=1)
    s = np.where(np.isneginf(lp), 0.0, lp).sum(axis=1)
    out = np.ones(len(lens), np.float64)
    long = lens > 1
    out[long] = np.exp(s[long] / (lens[long] - 1))
    out[unseen] = 0.0
    out[null] = 0.0
    return out, {"unseen_bigram_names": int(unseen.sum()), "invalid_bigrams_skipped_scoring": n_invalid}


def profile_features(p: pd.DataFrame, dataset: str, prob: np.ndarray, *, mode: str = "benchmark",
                     year: int | None = None, absent: set[str] = frozenset()) -> tuple[np.ndarray, dict]:
    """[n, 20] float64 features for the profile rows ``p`` (bigram table ``prob`` precomputed)."""
    X = np.zeros((len(p), 20), np.float64)
    X[:, 0] = count_col(p["statuses_count"])
    X[:, 1] = count_col(p["followers_count"])
    X[:, 2] = count_col(p["following_count"])
    X[:, 4] = count_col(p["listed_count"])
    X[:, 5] = custom_image_flag(p["profile_image_url"])
    X[:, 7] = bool_col(p["verified"])
    filled = []
    if mode == "paper":
        if "favourites_count" not in absent:
            X[:, 3] = count_col(p["favourites_count"])
            filled.append("favourites_count")
        if "default_profile" not in absent:
            X[:, 5] = bool_col(p["default_profile"])
            filled.append("default_profile")
        if "profile_use_background_image" not in absent:
            X[:, 6] = bool_col(p["profile_use_background_image"])
            filled.append("profile_use_background_image")
    age, age_info = account_age_days(p["created_at"], year or REF_YEAR[dataset])
    for k in range(5):
        X[:, 8 + k] = safe_div_age(X[:, k], age)
    if mode == "paper":
        X[:, 13] = X[:, 1] / np.maximum(X[:, 2], 1.0)
    else:
        X[:, 13] = np.divide(X[:, 1], X[:, 2], out=np.zeros(len(p)), where=X[:, 2] != 0)
    X[:, 14] = strip_len(p["screen_name"])
    X[:, 15] = digit_count(p["screen_name"])
    X[:, 16] = strip_len(p["name"])
    X[:, 17] = digit_count(p["name"])
    X[:, 18] = strip_len(p["description"])
    X[:, 19], lik_info = likelihood(p["screen_name"], prob)
    return X, {"age": age_info, "likelihood": lik_info, "paper_fields_filled": filled}


def build_features(dataset: str, rows: np.ndarray, splits: dict, args) -> tuple[np.ndarray, list[str], dict]:
    absent = set(profile_meta(dataset).get("absent_columns", []))
    cols = PROFILE_COLS + (PAPER_COLS if args.features == "paper" else [])
    # bigram corpus: screen names of every user node (all_users) or of the train split only
    sn_all = load_profile(dataset, ["screen_name"])["screen_name"]
    corpus = sn_all if args.likelihood_corpus == "all_users" else sn_all.iloc[np.asarray(splits["train"])]
    prob, binfo = bigram_table(corpus)
    del sn_all, corpus
    p = load_profile(dataset, cols, rows=rows)
    X, info = profile_features(p, dataset, prob, mode=args.features, year=args.reference_year, absent=absent)
    info["mode"] = args.features
    info["bigram"] = {**binfo, "corpus": args.likelihood_corpus}
    zero = {3, 6, 11} if args.features == "benchmark" else {k for k, f in ((3, "favourites_count"), (6, "profile_use_background_image"), (11, "favourites_count")) if f in absent}
    info["zero_slots"] = [NAMES[k] for k in sorted(zero)]
    if args.features == "benchmark" or "default_profile" in absent:
        info["slot5"] = "custom_profile_image flag ('default_profile_normal' not in profile_image_url)"
    info["absent_profile_fields_used"] = sorted(absent & set(cols))
    return X.astype(np.float32), list(NAMES), info
