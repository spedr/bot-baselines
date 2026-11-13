"""Lee 2011 features (TwiBot-22 benchmark re-implementation, 14 columns; docs/methods/lee2011.md 3).

Profile (8): screen_name_length (display name by default), description_length, account_longvity,
following_count, followers_count, following_to_followers, tweets_count, tweets_count_per_day.

Content (6), from the selected tweets of each user: links_ratio, unique_links_ratio, mention_ratio,
unique_mention_ratio, compression_ratio, similarity.

The content pass streams the tweet store (``TweetStore.selection`` + ``texts``) in tweet-budgeted
user chunks through a spawn-context worker pool. Each worker returns 8 raw per-user statistics
(``RAW``); ratios are formed in ``content_columns`` so the ZIP overhead can change without a
recompute. The raw statistics are cached under ``data/methods/lee2011/<dataset>/``.
"""
from __future__ import annotations

import math
import os
import re
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numba
import numpy as np

from ...io import read_json, save_npy, write_json
from ...paths import dataset_paths

# benchmark regexes (content_features.py L6-25), copied exactly; `[$-_]` is the range 0x24-0x5F
URL_RE = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
MENTION_RE = re.compile(r"@(?:[a-zA-Z]|[0-9]|[_+])+")

PROFILE_NAMES = ["screen_name_length", "description_length", "account_longvity", "following_count",
                 "followers_count", "following_to_followers", "tweets_count", "tweets_count_per_day"]
CONTENT_NAMES = ["links_ratio", "unique_links_ratio", "mention_ratio", "unique_mention_ratio",
                 "compression_ratio", "similarity"]
FEATURE_NAMES = PROFILE_NAMES + CONTENT_NAMES
RAW = ["n_tweets", "n_urls", "n_unique_urls", "n_mentions", "n_unique_mentions",
       "bytes_uncompressed", "bytes_deflate", "similarity"]

# benchmark collection dates (user_features.py L10-33): TwiBot-22 2022-04-20, all others 2020-10-01
REFERENCE_DATE = {"twibot22": "2022-04-20"}
DEFAULT_REFERENCE_DATE = "2020-10-01"
DEFAULT_CAP = {"twibot22": 20}
DEFAULT_CAP_OTHER = 200
ZIP_OVERHEAD = 45  # local header 30 B + 15 B arcname 'temp_tweets.txt' (spec 3.3)
# bump when the per-user statistics change, so stale content caches are recomputed (caches written
# before this field existed are version 1)
CONTENT_VERSION = 1

# profile fields each feature needs (dropped when the dataset lacks the field)
FEATURE_FIELDS = {
    "screen_name_length": ("name",), "description_length": ("description",),
    "account_longvity": ("created_at",), "following_count": ("following_count",),
    "followers_count": ("followers_count",),
    "following_to_followers": ("following_count", "followers_count"),
    "tweets_count": ("statuses_count",), "tweets_count_per_day": ("statuses_count", "created_at"),
}


# ---------------------------------------------------------------------------------------------
# profile features


def profile_features(dataset: str, col0: str = "name", reference_date: str | None = None
                     ) -> np.ndarray:
    """[N, 8] float64 profile features, benchmark definitions (nulls -> 0)."""
    import pandas as pd

    from ...data import load_profile

    cols = ["name" if col0 == "name" else "screen_name", "description", "created_at",
            "following_count", "followers_count", "statuses_count"]
    p = load_profile(dataset, columns=cols)
    ref = pd.Timestamp(reference_date or REFERENCE_DATE.get(dataset, DEFAULT_REFERENCE_DATE), tz="UTC")
    name_len = p[cols[0]].str.len().astype("float64").fillna(0).to_numpy()
    des_len = p["description"].str.len().astype("float64").fillna(0).to_numpy()
    # date-only arithmetic: (collection date - creation date).days
    days = (ref - p["created_at"].dt.floor("D")).dt.days.astype("float64").fillna(0).to_numpy()
    fing = p["following_count"].astype("float64").fillna(0).to_numpy()
    fers = p["followers_count"].astype("float64").fillna(0).to_numpy()
    tw = p["statuses_count"].astype("float64").fillna(0).to_numpy()
    ratio = np.divide(fing, fers, out=np.zeros_like(fing), where=fers != 0)
    per_day = np.divide(tw, days, out=tw.copy(), where=days != 0)  # longevity 0 -> tweets_count
    return np.stack([name_len, des_len, days, fing, fers, ratio, tw, per_day], 1)


# ---------------------------------------------------------------------------------------------
# per-user content statistics


def text_stats(texts: list[str]) -> tuple[int, int, int, int, int, int]:
    """(#urls, #distinct urls, #mentions, #distinct mentions, uncompressed bytes, deflate bytes),
    i.e. RAW[1:7]. Regexes never match '\\n', so one findall over the joined text equals the
    per-tweet findall the benchmark runs."""
    joined = "\n".join(texts) + "\n"
    urls = URL_RE.findall(joined)
    mens = MENTION_RE.findall(joined)
    b = joined.encode("utf-8", "surrogatepass")
    c = zlib.compressobj(-1, zlib.DEFLATED, -15)  # raw deflate, level 6 (= ZIP_DEFLATED default)
    clen = len(c.compress(b)) + len(c.flush())
    return len(urls), len(set(urls)), len(mens), len(set(mens)), len(b), clen


@numba.njit(cache=True, nogil=True)
def similarity_kernel(user_off, tw_off, orth, row, vec, out):
    """Mean pairwise spaCy ``Doc.similarity`` per user (spec 3.4).

    user_off [U+1] -> tweets; tw_off [T+1] -> tokens; orth uint64 [K] token ORTH; row int64 [K]
    vector row (-1 = no vector); vec float32 [R, D]. out float64 [U]: NaN without tweets, 1.0 for
    a single tweet. Pair rule: same length and identical ORTH sequence -> 1.0; else a zero doc
    vector -> 0.0; else cosine of the mean token vectors."""
    D = vec.shape[1]
    for u in range(len(user_off) - 1):
        a = user_off[u]
        n = user_off[u + 1] - a
        if n == 0:
            out[u] = np.nan
            continue
        if n == 1:
            out[u] = 1.0
            continue
        V = np.zeros((n, D))
        norms = np.zeros(n)
        for i in range(n):
            t0 = tw_off[a + i]
            t1 = tw_off[a + i + 1]
            for k in range(t0, t1):
                r = row[k]
                if r >= 0:
                    for d in range(D):
                        V[i, d] += vec[r, d]
            if t1 > t0:
                inv = 1.0 / (t1 - t0)
                s = 0.0
                for d in range(D):
                    V[i, d] *= inv
                    s += V[i, d] * V[i, d]
                norms[i] = math.sqrt(s)
        G = np.dot(V, np.ascontiguousarray(V.T))
        tot = 0.0
        for i in range(n):
            li = tw_off[a + i + 1] - tw_off[a + i]
            for j in range(i + 1, n):
                lj = tw_off[a + j + 1] - tw_off[a + j]
                same = li == lj
                if same:
                    oi = tw_off[a + i]
                    oj = tw_off[a + j]
                    for k in range(li):
                        if orth[oi + k] != orth[oj + k]:
                            same = False
                            break
                if same:
                    tot += 1.0
                elif norms[i] == 0.0 or norms[j] == 0.0:
                    pass
                else:
                    tot += G[i, j] / (norms[i] * norms[j])
        out[u] = tot / (n * (n - 1) / 2.0)


def tokenize(tokenizer, texts: list[str], keys, rows) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tweet token offsets [T+1], ORTH uint64 [K], vector rows int64 [K])."""
    from spacy.attrs import ORTH

    from .assets import lookup_rows

    lens = np.zeros(len(texts) + 1, np.int64)
    parts = []
    for i, doc in enumerate(tokenizer.pipe(texts, batch_size=1000)):
        a = doc.to_array(ORTH)
        parts.append(a)
        lens[i + 1] = len(a)
    off = np.cumsum(lens)
    orth = np.concatenate(parts).astype(np.uint64) if parts else np.zeros(0, np.uint64)
    return off, orth, lookup_rows(orth, keys, rows)


def single_thread_blas():
    """Context pinning BLAS to one thread. The kernel's Gram matrix is a BLAS dgemm (numba's
    np.dot calls scipy's OpenBLAS), whose last bit depends on the thread count (multi-threaded:
    up to 1 ulp off on users with ~200 tweets). One thread keeps the similarity column
    bit-reproducible whatever OMP_NUM_THREADS is and in the serial path; the cached columns were
    built that way (pool workers set OMP_NUM_THREADS=1)."""
    import scipy.linalg.cython_blas  # noqa: F401  (load the library numba calls, so it is limited)
    from threadpoolctl import threadpool_limits

    return threadpool_limits(1, user_api="blas")


def users_stats(texts: list[str], sel_off: np.ndarray, tokenizer, keys, rows, vec) -> np.ndarray:
    """[U, 8] raw statistics (``RAW``) for users whose texts are texts[sel_off[j]:sel_off[j+1]]."""
    U = len(sel_off) - 1
    out = np.zeros((U, len(RAW)), np.float64)
    for j in range(U):
        t = texts[sel_off[j]:sel_off[j + 1]]
        out[j, 0] = len(t)
        if t:
            out[j, 1:7] = text_stats(t)
    tw_off, orth, row = tokenize(tokenizer, texts, keys, rows)
    sim = np.empty(U, np.float64)
    with single_thread_blas():
        similarity_kernel(np.asarray(sel_off, np.int64), tw_off, orth, row, vec, sim)
    out[:, 7] = sim
    return out


# ---------------------------------------------------------------------------------------------
# tweet selection


def select_rows(store, users: np.ndarray, cap: int | None, order: str, seed: int = 0
                ) -> tuple[np.ndarray, np.ndarray]:
    """Rows of the selected tweets of ``users`` (in that order) and per-user offsets.

    release: first ``cap`` in store (release) order. newest: stored tweets sorted by created_at
    descending (ties and nulls keep release order, nulls last), then the first ``cap``; datasets
    without created_at (TwiBot-20, whose release order is newest first) fall back to release.
    random: per-user shuffle of the stored tweets seeded by (seed, idx), then the first ``cap``."""
    if order == "release" or (order == "newest" and "created_at" not in store.columns):
        return store.selection(cap, users)
    rows, so = store.selection(None, users)
    n = np.diff(so)
    owner = np.repeat(np.arange(len(n)), n)
    if order == "newest":
        ca = np.asarray(store.column("created_at")[rows]).astype(np.int64)
        ca = np.where(ca < 0, np.iinfo(np.int64).min, ca)
        perm = np.lexsort((np.arange(len(rows)), -ca.astype(np.float64), owner))
    elif order == "random":
        perm = np.empty(len(rows), np.int64)
        for j, u in enumerate(np.asarray(users).tolist()):
            a, b = so[j], so[j + 1]
            perm[a:b] = a + np.random.default_rng([seed, u]).permutation(b - a)
    else:
        raise ValueError(order)
    rows = rows[perm]
    keep_n = n if cap is None else np.minimum(n, cap)
    rank = np.arange(len(rows)) - np.repeat(so[:-1], n)
    rows = rows[rank < np.repeat(keep_n, n)]
    new_off = np.zeros(len(n) + 1, np.int64)
    np.cumsum(keep_n, out=new_off[1:])
    return rows, new_off


# ---------------------------------------------------------------------------------------------
# worker pool

_W: dict = {}


def _init_worker(store_dir: str) -> None:
    import sys

    # thinc imports torch when it is installed (+580 MB per process); the workers never need it
    if "torch" not in sys.modules:
        sys.modules["torch"] = None
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from ...data.tweets import TweetStore
    from .assets import load_tokenizer, load_vectors

    _W["store"] = TweetStore(store_dir)
    _W["tok"] = load_tokenizer()
    _W["since_reset"] = 0
    _W["keys"], _W["rows"], _W["vec"] = load_vectors()


TOKENIZER_RESET = 50_000  # the tokenizer interns every new string; reload it to bound the vocab


def _work(users: np.ndarray, cap, order: str, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    from .assets import load_tokenizer

    st = _W["store"]
    rows, so = select_rows(st, users, cap, order, seed)
    texts = st.texts(rows)
    if _W["since_reset"] > TOKENIZER_RESET:
        _W["tok"] = load_tokenizer()
        _W["since_reset"] = 0
    _W["since_reset"] += len(rows)
    return users, users_stats(texts, so, _W["tok"], _W["keys"], _W["rows"], _W["vec"]), len(rows)


def plan_chunks(users: np.ndarray, n_sel: np.ndarray, tweet_budget: int, max_users: int = 5000
                ) -> list[np.ndarray]:
    """Split ``users`` into consecutive chunks of at most ``tweet_budget`` selected tweets."""
    out, start, acc = [], 0, 0
    for i, c in enumerate(n_sel.tolist()):
        if i > start and (acc + c > tweet_budget or i - start >= max_users):
            out.append(users[start:i])
            start, acc = i, 0
        acc += c
    if start < len(users):
        out.append(users[start:])
    return out


def cache_key(cap, order: str, seed: int) -> str:
    return f"cap{cap if cap is not None else 'all'}_{order}" + (f"_s{seed}" if order == "random" else "")


def cache_paths(dataset: str, key: str) -> tuple[Path, Path]:
    d = dataset_paths(dataset).method_dir("lee2011")
    return d / f"content_{key}.f64.npy", d / f"content_{key}.meta.json"


def content_stats(dataset: str, users: np.ndarray, cap: int | None, order: str = "release",
                  seed: int = 0, n_workers: int = 6, tweet_budget: int = 20_000,
                  recompute: bool = False, log=print) -> tuple[np.ndarray, dict, Path]:
    """[N, 8] raw content statistics (NaN rows for users outside ``users``), cached."""
    from ...data.tweets import TweetStore
    from ...io import file_digest
    from ...mem import cgroup_memory, peak_rss_bytes, peak_rss_children_bytes
    from ...status import StepStatus
    from .assets import ensure_assets

    key = cache_key(cap, order, seed)
    path, mpath = cache_paths(dataset, key)
    store = TweetStore(dataset)
    digest = {"offsets": file_digest(store.dir / "offsets.i64.npy"),
              "meta": file_digest(store.dir / "meta.json")}
    users = np.asarray(users, np.int64)
    if path.exists() and mpath.exists() and not recompute:
        meta = read_json(mpath)
        ok = (meta.get("store_digest") == digest and meta.get("n_users_computed") == len(users)
              and meta.get("content_version", 1) == CONTENT_VERSION)
        if ok:
            log(f"content cache hit: {path}")
            return np.load(path, mmap_mode="r"), meta, path
        log(f"content cache stale ({path}); recomputing")
    vmeta = ensure_assets()
    n_sel = np.asarray(store.counts(cap))[users]  # min(stored, cap) for every order
    chunks = plan_chunks(users, n_sel, tweet_budget)
    total_t = int(n_sel.sum())
    log(f"content pass {dataset} {key}: {len(users):,} users, {total_t:,} tweets, {len(chunks)} chunks, "
        f"{n_workers} workers")
    out = np.full((store.n_users, len(RAW)), np.nan, np.float64)
    t0 = time.time()
    done_t = 0
    import multiprocessing as mp

    step = StepStatus(f"lee2011_{dataset}_content_{key}", total=total_t, unit="tweets")
    with step:
        step.start()
        if n_workers <= 1:
            _init_worker(str(store.dir))
            it = (_work(c, cap, order, seed) for c in chunks)
            for u, s, nt in it:
                out[u] = s
                done_t += nt
                step.update(done_t)
            pool_peak = {}
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(n_workers, mp_context=ctx, initializer=_init_worker,
                                     initargs=(str(store.dir),)) as ex:
                pending = set()
                ci = iter(chunks)
                for c in ci:  # bounded in-flight work: 2 chunks per worker
                    pending.add(ex.submit(_work, c, cap, order, seed))
                    if len(pending) >= 2 * n_workers:
                        break
                last = time.time()
                while pending:
                    f = next(as_completed(pending))
                    pending.remove(f)
                    u, s, nt = f.result()
                    out[u] = s
                    done_t += nt
                    nxt = next(ci, None)
                    if nxt is not None:
                        pending.add(ex.submit(_work, nxt, cap, order, seed))
                    step.update(done_t)
                    if time.time() - last > 60:
                        last = time.time()
                        el = last - t0
                        log(f"  {done_t:,}/{total_t:,} tweets  {done_t / el:,.0f} tweets/s  "
                            f"cgroup {cgroup_memory().get('memory.current', 0) / 1e9:.2f} GB")
            pool_peak = {"peak_rss_children_bytes": peak_rss_children_bytes()}
        el = time.time() - t0
        meta = {"dataset": dataset, "key": key, "cap": cap, "order": order, "seed": seed,
                "raw_columns": RAW, "content_version": CONTENT_VERSION,
                "n_users_computed": int(len(users)), "n_tweets": total_t,
                "wall_seconds": round(el, 1), "tweets_per_s": round(total_t / el, 1) if el else None,
                "users_per_s": round(len(users) / el, 1) if el else None, "n_workers": n_workers,
                "tweet_budget": tweet_budget, "peak_rss_parent_bytes": peak_rss_bytes(), **pool_peak,
                "cgroup": cgroup_memory(), "store_digest": digest, "vectors": vmeta}
        save_npy(path, out)
        write_json(mpath, meta)
        step.done(wall_seconds=meta["wall_seconds"], tweets_per_s=meta["tweets_per_s"])
    log(f"content pass done in {el:.0f} s ({meta['tweets_per_s']:,} tweets/s); cache {path}")
    return out, meta, path


def content_columns(raw: np.ndarray, zip_overhead: int = ZIP_OVERHEAD) -> np.ndarray:
    """[n, 6] content features from raw statistics; rows with 0 tweets are NaN."""
    raw = np.asarray(raw, np.float64)
    n = raw[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        cols = [raw[:, 1] / n, raw[:, 2] / n, raw[:, 3] / n, raw[:, 4] / n,
                raw[:, 5] / (raw[:, 6] + zip_overhead), raw[:, 7]]
    x = np.stack(cols, 1)
    x[~(n > 0)] = np.nan
    return x
