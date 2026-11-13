"""BotHunter Tier-1 feature vector (docs/methods/bothunter.md section 3).

``benchmark`` mode reproduces the 17 slots of the TwiBot-22 re-implementation
(``src/BotHunter/preprocess.py`` L110-134): 9 informative profile features and 8 constant-zero
slots, kept in place because they consume ``max_features`` draws. ``extended`` mode fills the
zero slots from data we hold (favourites from the profile; retweet / hashtags / mentions /
sensitive / source from the user's newest stored tweet); slots whose field the dataset lacks stay
0 and are listed in the run's ``features.absent`` entry.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from bb.data import TweetStore, load_profile, profile_meta

from .shared import (REF_YEAR, account_age_days, count_col, custom_image_flag, safe_div_age,
                     shannon_entropy, strip_len)

NAMES = ["screen_name_len", "custom_profile_image", "screen_name_entropy", "has_location",
         "statuses_count", "source_bin", "friends_count", "followers_count", "favourites_count",
         "last_is_retweet", "last_same_lang", "last_n_hashtags", "last_n_mentions",
         "last_sensitive", "bot_reference", "account_age_days", "tweets_per_day"]
ZERO_SLOTS = {5, 8, 9, 10, 11, 12, 13}  # constant 0 in benchmark mode (10 stays 0 everywhere)
PROFILE_COLS = ["screen_name", "description", "location", "raw_location", "profile_image_url", "created_at",
                "statuses_count", "following_count", "followers_count"]


def profile_features(p: pd.DataFrame, dataset: str, *, year: int | None = None,
                     literal_none_location: bool = False, bot_ref_casefold: bool = False
                     ) -> tuple[np.ndarray, dict]:
    """[n, 17] float64 benchmark-mode features for the profile rows ``p``."""
    n = len(p)
    X = np.zeros((n, 17), np.float64)
    sn = p["screen_name"]
    X[:, 0] = strip_len(sn)
    X[:, 1] = custom_image_flag(p["profile_image_url"])
    X[:, 2] = shannon_entropy(sn.str.strip())
    # has location: benchmark len(location.strip()) > 0 on the node file's string. Clean default
    # uses the parsed column (TwiBot-20 "None" -> null); the emulation uses the raw release spelling.
    loc = p["raw_location"] if literal_none_location else p["location"]
    X[:, 3] = (strip_len(loc) > 0).astype(np.float64)
    X[:, 4] = count_col(p["statuses_count"])
    X[:, 6] = count_col(p["following_count"])
    X[:, 7] = count_col(p["followers_count"])
    # 'bot' in screen name or description, case-sensitive str.find (option: case-folded)
    sn_b, de_b = sn, p["description"]
    if bot_ref_casefold:
        sn_b, de_b = sn_b.str.lower(), de_b.str.lower()
    ref = sn_b.str.contains("bot", regex=False).fillna(False) | de_b.str.contains("bot", regex=False).fillna(False)
    X[:, 14] = ref.to_numpy(bool).astype(np.float64)
    age, age_info = account_age_days(p["created_at"], year or REF_YEAR[dataset])
    X[:, 15] = age
    X[:, 16] = safe_div_age(X[:, 4], age)
    return X, {"age": age_info}


# ---------------------------------------------------------------------------------------------
# extended mode: newest stored tweet per user

_HASHTAG = re.compile(r"#\w+")
_MENTION = re.compile(r"@\w+")


def newest_rows(store: TweetStore, n_users: int, chunk_users: int = 100_000) -> np.ndarray:
    """Row of each user's newest stored tweet (max ``created_at``, first in release order on ties;
    stores without ``created_at`` (TwiBot-20, newest first) -> first row), -1 if none. Users
    0..n_users-1, processed in contiguous chunks so only one chunk of the column is resident."""
    off = np.asarray(store.offsets[: n_users + 1], dtype=np.int64)
    out = np.full(n_users, -1, np.int64)
    has_ct = "created_at" in store.columns
    for u0 in range(0, n_users, chunk_users):
        u1 = min(n_users, u0 + chunk_users)
        o = off[u0:u1 + 1]
        lens = np.diff(o)
        ne = np.flatnonzero(lens > 0)
        if not len(ne):
            continue
        if not has_ct:
            out[u0 + ne] = o[ne]
            continue
        a, b = int(o[0]), int(o[-1])
        ct = np.asarray(store.column("created_at")[a:b])
        starts = o[ne] - a
        mx = np.maximum.reduceat(ct, starts)
        seg = np.repeat(np.arange(len(ne)), lens[ne])
        cand = np.flatnonzero(ct == mx[seg])
        sc = seg[cand]
        first = cand[np.r_[True, sc[1:] != sc[:-1]]]
        out[u0 + ne] = first + a
        del ct
    return out


def tweet_features(dataset: str, rows: np.ndarray, splits: dict, source_topk: int = 10
                   ) -> tuple[dict[int, np.ndarray], dict]:
    """Extended slots 5, 9, 11, 12, 13 for the given user rows (dict slot -> [len(rows)])."""
    store = TweetStore(dataset)
    n_users = int(rows.max()) + 1
    nr = newest_rows(store, n_users)
    r = nr[rows]
    has = r >= 0
    rr = r[has]
    cols = store.columns
    out: dict[int, np.ndarray] = {}
    info = {"users_with_tweet": int(has.sum()), "users_without_tweet": int((~has).sum()),
            "newest_by": "max created_at" if "created_at" in cols else "first in release order (newest first)"}

    def gather(name):
        """Raw column values (-1 = null) of each user's newest tweet; -1 for users without one."""
        v = np.full(len(rows), -1, np.int64)
        v[has] = store.column(name)[rr]
        return v

    def take(name):
        v = gather(name)
        return np.where(v < 0, 0, v).astype(np.float64)  # null / no tweet -> 0

    if "is_retweet" in cols:
        out[9], out[11], out[12] = take("is_retweet"), take("n_hashtags"), take("n_mentions")
        info["source_of_9_11_12"] = "tweet store columns"
        if dataset == "cresci15":
            # The Cresci-2015 store's is_retweet is 0 for every tweet (retweeted_status_id is not
            # usable in the release) although ~20% of texts start with 'RT @'; use the same
            # 'RT @' fallback the TwiBot-22 builder applies.
            rt = np.array([t.startswith("RT @") for t in store.texts(rr)], np.float64)
            out[9][has] = np.maximum(out[9][has], rt)
            info["source_of_9_11_12"] += "; cresci15 retweet = is_retweet OR text starts with 'RT @'"
    else:  # TwiBot-20: text only
        texts = store.texts(rr)
        rt = np.array([t.startswith("RT @") for t in texts], np.float64)
        ht = np.array([len(_HASHTAG.findall(t)) for t in texts], np.float64)
        mn = np.array([len(_MENTION.findall(t)) for t in texts], np.float64)
        for k, v in ((9, rt), (11, ht), (12, mn)):
            out[k] = np.zeros(len(rows), np.float64)
            out[k][has] = v
        info["source_of_9_11_12"] = "regex on text ('RT @' prefix, #\\w+, @\\w+)"
    if "possibly_sensitive" in cols:
        out[13] = take("possibly_sensitive")
    if "source" in cols:
        # code 0 = no tweet / null source, 1..k = top-k sources (by frequency among TRAIN users'
        # newest tweets), k+1 = other
        raw = gather("source")
        tr_pos = np.arange(len(splits["train"]))  # rows = concat(train, val, test)
        tr_src = raw[tr_pos][raw[tr_pos] >= 0]
        vals, cnt = np.unique(tr_src, return_counts=True)
        top = vals[np.argsort(-cnt, kind="stable")[:source_topk]]
        lut = np.full(int(max(raw.max(), 0)) + 1, source_topk + 1, np.float64)
        lut[top] = np.arange(1, len(top) + 1)
        out[5] = np.where(raw >= 0, lut[np.maximum(raw, 0)], 0.0)
        info["source_topk"] = [int(v) for v in top]
    return out, info


def build_features(dataset: str, rows: np.ndarray, splits: dict, args) -> tuple[np.ndarray, list[str], dict]:
    mode = args.features
    absent_profile = set(profile_meta(dataset).get("absent_columns", []))
    p = load_profile(dataset, PROFILE_COLS + (["favourites_count"] if mode == "extended" else []), rows=rows)
    X, info = profile_features(p, dataset, year=args.reference_year,
                               literal_none_location=args.tw20_literal_none_location,
                               bot_ref_casefold=args.bot_ref_casefold)
    filled = set()
    if mode == "extended":
        if "favourites_count" not in absent_profile:
            X[:, 8] = count_col(p["favourites_count"])
            filled.add(8)
        slots, tinfo = tweet_features(dataset, rows, splits, args.source_topk)
        for k, v in slots.items():
            X[:, k] = v
            filled.add(k)
        info["tweets"] = tinfo
    info["mode"] = mode
    info["zero_slots"] = [NAMES[k] for k in sorted(ZERO_SLOTS - filled)]
    info["absent_profile_fields_used"] = sorted(absent_profile & set(PROFILE_COLS))
    return X.astype(np.float32), list(NAMES), info
