"""Real-data check of the Lee 2011 features against the TwiBot-22 benchmark's own code (review
round 1). Not a pytest file: it needs real artifacts, the content cache, the full en_core_web_lg
pipeline and a checkout of the benchmark's ``src/Lee`` (MIT; not vendored here):

    curl -sSfO https://raw.githubusercontent.com/LuoUndergradXJTU/TwiBot-22/54adf9a1dbb6dc238c932d3147a1875def5817d8/src/Lee/Twibot-22/content_features.py
    tools/memrun --light -- uv run python tests/methods/lee2011_benchmark_check.py \\
        --dataset twibot20 --users 25 --bench-content-features ./content_features.py

1. profile: the branches of the benchmark's ``user_features.py`` (re-implemented here on the raw
   ``created_at`` string, nulls -> 0) == ``profile_features`` on every labelled user;
2. content: the benchmark's ``count_urls``, ``count_mentions``, ``zip_ratio`` (real files, real
   ``zipfile``) and ``count_similarity`` (full pipeline, ``disable=['parser','ner']``) on a random
   sample of labelled users with tweets == the cached content columns our runs use;
3. empty tweet texts among the selected tweets (the benchmark turns them into the string "nan").
"""
from __future__ import annotations

import argparse
import calendar
import datetime
import importlib.util
import os
import sys
import tempfile

import numpy as np


def bench_longevity(ca: str | None, ds: str) -> int:
    if ca is None or ca[-1] == "L":
        return 0
    if ds == "twibot22":
        y, m, d = (int(x) for x in ca.split(" ")[0].split("-")[:3])
        return (datetime.datetime(2022, 4, 20) - datetime.datetime(y, m, d)).days
    c = [x for x in ca.split(" ") if x != ""]
    month = list(calendar.month_abbr).index(c[1])
    return (datetime.datetime(2020, 10, 1) - datetime.datetime(int(c[5]), month, int(c[2]))).days


def check_profile(ds: str) -> None:
    import pandas as pd

    from bb.data import load_profile, load_users
    from bb.methods.lee2011 import features as F

    p = load_profile(ds, columns=["name", "description", "raw_created_at", "following_count",
                                  "followers_count", "statuses_count"])
    assert np.array_equal(p["idx"].to_numpy(), np.arange(len(p))), "profile rows not idx-ordered"
    lab = np.flatnonzero(load_users(ds, columns=["label"])["label"].to_numpy() >= 0)
    X = F.profile_features(ds)

    def nz(v):
        return None if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)) else v

    cols = [p[c].tolist() for c in ("name", "description", "raw_created_at", "following_count",
                                    "followers_count", "statuses_count")]
    bad = dict.fromkeys(F.PROFILE_NAMES, 0)
    for i in lab.tolist():
        n, d, c, a, b, t = (nz(col[i]) for col in cols)
        c = c.strip() if isinstance(c, str) else c
        lon = bench_longevity(c, ds) if c else 0
        ref = [0 if n is None else len(n), 0 if d is None else len(d), lon, 0 if a is None else int(a),
               0 if b is None else int(b),
               0.0 if (a is None or b is None or int(b) == 0) else int(a) / int(b), 0 if t is None else int(t)]
        ref.append(ref[6] if lon == 0 else ref[6] / lon)
        for k, (r, o) in enumerate(zip(ref, X[i])):
            bad[F.PROFILE_NAMES[k]] += int(not np.isclose(r, o, rtol=1e-12, atol=0))
    print(f"{ds} profile: {len(lab):,} labelled users; mismatches {bad}")
    assert not any(bad.values())


def check_content(ds: str, k: int, bench_path: str) -> None:
    sys.modules["torch"] = None  # thinc imports torch when installed; not needed here
    import spacy

    from bb.data import TweetStore, load_users
    from bb.methods.lee2011 import features as F
    from bb.methods.lee2011.assets import extract_model

    spec = importlib.util.spec_from_file_location("bench_content_features", bench_path)
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)
    cap = F.DEFAULT_CAP.get(ds, F.DEFAULT_CAP_OTHER)
    nlp = spacy.load(extract_model(full=True), disable=["parser", "ner"])
    st = TweetStore(ds)
    off = np.asarray(st.offsets)
    to = np.asarray(st.text_offsets)
    lab = np.flatnonzero(load_users(ds, columns=["label"])["label"].to_numpy() >= 0)
    n = np.minimum(np.diff(off)[lab], cap)
    empty = sum(int((np.diff(to[s:s + m + 1]) == 0).sum()) for s, m in zip(off[lab].tolist(), n.tolist()))
    print(f"{ds}: {int(n.sum()):,} selected tweets of labelled users, {empty:,} empty texts")
    lab = lab[n > 0]
    users = np.random.default_rng(123).choice(lab, k, replace=False)
    raw = np.load(F.cache_paths(ds, F.cache_key(cap, "release", 0))[0], mmap_mode="r")
    ours = F.content_columns(np.asarray(raw[users]))
    ref = []
    here = os.getcwd()
    os.chdir(tempfile.mkdtemp())  # zip_ratio writes ./temp_tweets.txt(.zip)
    try:
        for u in users.tolist():
            a = int(off[u])
            texts = st.texts(np.arange(a, a + min(int(off[u + 1]) - a, cap)))
            li, me = cf.count_urls(texts), cf.count_mentions(texts)
            ref.append([li[0], li[1], me[0], me[1], cf.zip_ratio(texts), cf.count_similarity(nlp, list(texts))])
    finally:
        os.chdir(here)
    err = np.abs(np.asarray(ref, np.float64) - ours)
    for j, name in enumerate(F.CONTENT_NAMES):
        print(f"  {name:22s} max |benchmark - ours| = {err[:, j].max():.3e}")
    assert err[:, :5].max() < 1e-12 and err[:, 5].max() < 1e-5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twibot20")
    ap.add_argument("--users", type=int, default=25)
    ap.add_argument("--bench-content-features", required=True, help="path to src/Lee/Twibot-22/content_features.py")
    a = ap.parse_args()
    check_profile(a.dataset)
    check_content(a.dataset, a.users, a.bench_content_features)
    print("OK")


if __name__ == "__main__":
    main()
