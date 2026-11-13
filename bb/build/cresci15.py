"""Cresci-2015 builder from the raw CNR release (cresci-2015.csv.tar.gz: E13, FSF, INT, TFP, TWT zips).

Index order: subsets in ``SUBSET_ORDER`` (humans E13, TFP; then bots INT, TWT, FSF), users.csv order
within each subset. This order was verified row-by-row against the BotRGCN-format tensors in the
MGTAB repo zip (raw/mgtab-cresci-15/Cresci-15.zip), whose node order and 70/20/10 per-subset
contiguous split are the TwiBot-22 benchmark's; see ``verify_against_mgtab``. The split written here
is that recovered split (``split_source`` in meta.json says so).

Tweets: tweets.csv order per subset, first ``CAP`` per user kept (``n_total`` records the rest).
Graph: ``friend`` (friends.csv: source follows target) and ``follower`` (followers.csv rows as
released), both endpoints restricted to the 5,301 labelled users.
"""
from __future__ import annotations

import io
import shutil
import tarfile
import time
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pcsv

from ..io import NpyAppender, file_digest, read_json, save_npy, write_json
from ..paths import dataset_paths, raw_dir
from ..schema import cresci_bool, cresci_str, parse_twitter_date, to_int
from ..status import StepStatus
from .common import ProfileWriter, profile_null_summary, write_index

DS = "cresci15"
SUBSET_ORDER = ["E13", "TFP", "INT", "TWT", "FSF"]
LABEL = {"E13": 0, "TFP": 0, "FSF": 1, "INT": 1, "TWT": 1}
EXPECTED = {"E13": 1481, "TFP": 469, "FSF": 1169, "INT": 1337, "TWT": 845}
CAP = 200
TWEET_COLS = ["id", "text", "user_id", "created_at", "retweet_count", "reply_count", "favorite_count",
              "num_hashtags", "num_urls", "num_mentions", "in_reply_to_status_id", "retweeted_status_id",
              "source"]


def _read_csv_bytes(b: bytes, columns: list[str] | None = None) -> pa.Table:
    # quoted "" stays '' while an unquoted empty field is null
    return pcsv.read_csv(io.BytesIO(b), read_options=pcsv.ReadOptions(block_size=64 << 20),
                         convert_options=pcsv.ConvertOptions(
                             column_types={}, strings_can_be_null=True, quoted_strings_can_be_null=False,
                             include_columns=columns, null_values=[""],
                             auto_dict_encode=False),
                         parse_options=pcsv.ParseOptions(newlines_in_values=True))


def _as_str_table(t: pa.Table) -> pa.Table:
    return pa.table({n: pc_cast_str(t.column(n)) for n in t.column_names})


def pc_cast_str(col):
    import pyarrow.compute as pc
    return col if pa.types.is_string(col.type) else pc.cast(col, pa.string())


def _profile_row(i: int, r: dict) -> dict:
    g = r.get
    return {
        "idx": i, "profile_missing": False,
        "name": cresci_str(g("name")), "screen_name": cresci_str(g("screen_name")),
        "description": cresci_str(g("description")), "url": cresci_str(g("url")),
        "location": cresci_str(g("location")), "profile_image_url": cresci_str(g("profile_image_url")),
        "created_at": parse_twitter_date(cresci_str(g("created_at"))),
        "followers_count": to_int(g("followers_count")), "following_count": to_int(g("friends_count")),
        "statuses_count": to_int(g("statuses_count")), "listed_count": to_int(g("listed_count")),
        "favourites_count": to_int(g("favourites_count")),
        "protected": cresci_bool(g("protected")), "verified": cresci_bool(g("verified")),
        "default_profile": cresci_bool(g("default_profile")),
        "default_profile_image": cresci_bool(g("default_profile_image")),
        "geo_enabled": cresci_bool(g("geo_enabled")),
        "profile_use_background_image": cresci_bool(g("profile_use_background_image")),
        "profile_background_tile": cresci_bool(g("profile_background_tile")),
        "has_extended_profile": None, "contributors_enabled": None, "is_translator": None,
        "is_translation_enabled": None, "lang": cresci_str(g("lang")),
        "utc_offset": to_int(cresci_str(g("utc_offset"))), "time_zone": cresci_str(g("time_zone")),
        "pinned_tweet_id": None, "withheld": None, "entities": None,
        "raw_name": g("name"), "raw_screen_name": g("screen_name"), "raw_description": g("description"),
        "raw_location": g("location"), "raw_profile_image_url": g("profile_image_url"),
        "raw_default_profile_image": g("default_profile_image"), "raw_url": g("url"),
        "raw_protected": g("protected"), "raw_verified": g("verified"), "raw_created_at": g("created_at"),
    }


def _i(col, null=-1) -> np.ndarray:
    import pyarrow.compute as pc
    c = pc.cast(col, pa.string())
    c = pc.if_else(pc.equal(c, "NULL"), pa.scalar(None, pa.string()), c)
    c = pc.cast(c, pa.int64())
    return c.fill_null(null).to_numpy()


def build(force: bool = False) -> dict:
    dp = dataset_paths(DS)
    step = f"{DS}_build"
    if dp.meta.exists() and (dp.tweets / "meta.json").exists() and not force:
        print(f"{step}: already built; use --force")
        return read_json(dp.meta)
    tar_path = raw_dir(DS) / "cresci-2015.csv.tar.gz"
    st = StepStatus(step, total=len(SUBSET_ORDER), unit="subsets")
    with st:
        st.start()
        t0 = time.time()
        work = dp.root / "_work"
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        # ---- pass 1: read every subset zip once (tar is gzip-streamed in member order)
        zips: dict[str, bytes] = {}
        tf = tarfile.open(tar_path, "r:gz")
        users_tbl: dict[str, pa.Table] = {}
        edges_raw: dict[str, list] = {"friend": [], "follower": []}
        tweet_parts = {}
        for m in tf:
            if not m.name.endswith(".csv.zip"):
                continue
            sub = m.name.split("/")[-1].split(".")[0]
            z = zipfile.ZipFile(io.BytesIO(tf.extractfile(m).read()))
            users_tbl[sub] = _as_str_table(_read_csv_bytes(z.read("users.csv")))
            for rel, fname in (("friend", "friends.csv"), ("follower", "followers.csv")):
                e = _read_csv_bytes(z.read(fname))
                edges_raw[rel].append((sub, pc_cast_str(e.column("source_id")).to_numpy(zero_copy_only=False),
                                       pc_cast_str(e.column("target_id")).to_numpy(zero_copy_only=False)))
            # tweets: text to a temp blob, metadata kept as small arrays
            tw = _read_csv_bytes(z.read("tweets.csv"), columns=TWEET_COLS)
            texts = pc_cast_str(tw.column("text")).to_pylist()
            enc = [b"" if t is None else t.encode("utf-8", "replace") for t in texts]
            lens = np.fromiter(map(len, enc), np.int64, len(enc))
            with open(work / f"{sub}.txt", "wb") as f:
                f.write(b"".join(enc))
            del enc, texts
            tweet_parts[sub] = {
                "user_id": pc_cast_str(tw.column("user_id")).to_numpy(zero_copy_only=False),
                "len": lens, "text_null": tw.column("text").is_null().to_numpy(zero_copy_only=False).astype(np.int8),
                "tweet_id": _i(tw.column("id")), "retweet_count": _i(tw.column("retweet_count")),
                "reply_count": _i(tw.column("reply_count")), "like_count": _i(tw.column("favorite_count")),
                "n_hashtags": _i(tw.column("num_hashtags")), "n_urls": _i(tw.column("num_urls")),
                "n_mentions": _i(tw.column("num_mentions")),
                "is_reply": (_i(tw.column("in_reply_to_status_id"), 0) > 0).astype(np.int8),
                "is_retweet": (_i(tw.column("retweeted_status_id"), 0) > 0).astype(np.int8),
                "created_at": np.array([-1 if d is None else int(d.timestamp()) for d in
                                        map(parse_twitter_date, pc_cast_str(tw.column("created_at")).to_pylist())],
                                       dtype=np.int64),
            }
            del tw, z
            st.update(len(users_tbl), subset=sub)
        tf.close()
        missing = set(SUBSET_ORDER) - set(users_tbl)
        if missing:
            raise ValueError(f"subsets missing from archive: {missing}")

        # ---- users + profile in SUBSET_ORDER
        pw = ProfileWriter(dp.profile, rows_per_group=10_000)
        rows = {"raw_id": [], "label": [], "raw_order": [], "source_file": []}
        i = 0
        for sub in SUBSET_ORDER:
            t = users_tbl[sub]
            if t.num_rows != EXPECTED[sub]:
                st.error(f"{sub}: {t.num_rows} users, expected {EXPECTED[sub]}")
            for k, r in enumerate(t.to_pylist()):
                rows["raw_id"].append(r["id"])
                rows["label"].append(LABEL[sub])
                rows["raw_order"].append(k)
                rows["source_file"].append(sub)
                pw.add(_profile_row(i, r))
                i += 1
        n = pw.close()
        users = pd.DataFrame({
            "idx": np.arange(n, dtype=np.int64), "user_id": rows["raw_id"], "raw_id": rows["raw_id"],
            "split": "train", "label": np.asarray(rows["label"], np.int8),
            "raw_order": np.asarray(rows["raw_order"], np.int64), "split_csv_order": -1,
            "label_csv_order": -1, "source_file": rows["source_file"],
        })
        split, split_source = benchmark_split(users)
        users["split"] = split
        write_json(dp.profile.with_suffix(".meta.json"), profile_null_summary(dp.profile))
        id2idx = pd.Index(users["user_id"].to_numpy(dtype=object))
        meta = write_index(dp, users, {
            "index_order": f"subsets {SUBSET_ORDER}, users.csv order within each",
            "split_source": split_source, "subset_sizes": EXPECTED,
            "sources": {"cresci-2015.csv.tar.gz": file_digest(tar_path)},
            "build_seconds": round(time.time() - t0, 1),
        })

        # ---- tweets: group by idx (stable: csv order), cap per user
        tw_dir = dp.tweets.with_name("tweets.partial")
        if tw_dir.exists():
            shutil.rmtree(tw_dir)
        tw_dir.mkdir(parents=True)
        cols = ["release_rank", "tweet_id", "created_at", "retweet_count", "reply_count", "like_count",
                "n_hashtags", "n_urls", "n_mentions", "is_retweet", "is_reply", "text_null"]
        allp = {c: [] for c in ["idx", "len", "start", "sub", *cols]}
        foreign = 0
        for si, sub in enumerate(SUBSET_ORDER):
            p = tweet_parts[sub]
            ix = id2idx.get_indexer(pd.Index(p["user_id"].astype(object)))
            foreign += int((ix < 0).sum())
            keep = ix >= 0
            start = np.concatenate([[0], np.cumsum(p["len"])[:-1]])
            allp["idx"].append(ix[keep])
            allp["len"].append(p["len"][keep])
            allp["start"].append(start[keep])
            allp["sub"].append(np.full(int(keep.sum()), si, np.int8))
            allp["release_rank"].append(np.flatnonzero(keep).astype(np.int64))
            for c in cols[1:]:
                allp[c].append(p[c][keep])
        A = {k: np.concatenate(v) for k, v in allp.items()}
        order = np.argsort(A["idx"], kind="stable")
        A = {k: v[order] for k, v in A.items()}
        n_total = np.bincount(A["idx"], minlength=n)
        # rank within user (csv order) -> keep first CAP
        first = np.concatenate([[0], np.cumsum(n_total)[:-1]])
        within = np.arange(len(A["idx"])) - np.repeat(first, n_total)
        keep = within < CAP
        A = {k: v[keep] for k, v in A.items()}
        kept = np.bincount(A["idx"], minlength=n)
        offsets = np.zeros(n + 1, np.int64)
        np.cumsum(kept, out=offsets[1:])
        blobs = {sub: np.memmap(work / f"{sub}.txt", dtype=np.uint8, mode="r") if (work / f"{sub}.txt").stat().st_size
                 else np.zeros(0, np.uint8) for sub in SUBSET_ORDER}
        with open(tw_dir / "text.bin", "wb") as f:
            for s_, a, l in zip(A["sub"].tolist(), A["start"].tolist(), A["len"].tolist()):
                f.write(blobs[SUBSET_ORDER[s_]][a:a + l].tobytes())
        save_npy(tw_dir / "text_offsets.i64.npy", np.concatenate([[0], np.cumsum(A["len"])]).astype(np.int64))
        dtypes = {"release_rank": np.int64, "tweet_id": np.int64, "created_at": np.int64,
                  "retweet_count": np.int32, "reply_count": np.int32, "like_count": np.int32,
                  "n_hashtags": np.int16, "n_urls": np.int16, "n_mentions": np.int16,
                  "is_retweet": np.int8, "is_reply": np.int8, "text_null": np.int8}
        for c in cols:
            save_npy(tw_dir / f"{c}.npy", A[c].astype(dtypes[c]))
        save_npy(tw_dir / "offsets.i64.npy", offsets)
        save_npy(tw_dir / "n_total.i32.npy", n_total.astype(np.int32))
        write_json(tw_dir / "meta.json", {
            "dataset": DS, "n_users": n, "n_tweets": int(offsets[-1]), "cap": CAP,
            "n_tweets_release": int(n_total.sum()), "tweets_with_foreign_user": foreign,
            "order": "tweets.csv order within each subset; first `cap` per user kept",
            "release_rank": "row position in the subset's tweets.csv",
            "columns": {c: np.dtype(dtypes[c]).name for c in cols},
            "flags_source": "is_retweet: retweeted_status_id > 0; is_reply: in_reply_to_status_id > 0; "
                            "counts from num_hashtags/num_urls/num_mentions/retweet/reply/favorite_count",
            "users_with_tweets": int((kept > 0).sum()), "users_beyond_cap": int((n_total > CAP).sum()),
        })
        if dp.tweets.exists():
            shutil.rmtree(dp.tweets)
        tw_dir.replace(dp.tweets)
        del blobs

        # ---- graph
        from ..data.graph import build_csr_from_coo, graph_stats
        rel_meta = {}
        pos = 0
        for rel in ("friend", "follower"):
            S, D, P = [], [], []
            listed = 0
            for sub, s, d in sorted(edges_raw[rel], key=lambda x: SUBSET_ORDER.index(x[0])):
                si = id2idx.get_indexer(pd.Index(s.astype(object)))
                di = id2idx.get_indexer(pd.Index(d.astype(object)))
                ok = (si >= 0) & (di >= 0)
                listed += len(s)
                S.append(si[ok]); D.append(di[ok])
                P.append(pos + np.flatnonzero(ok))
                pos += len(s)
            S, D, P = (np.concatenate(x).astype(np.int64) for x in (S, D, P))
            build_csr_from_coo(dp.csr(rel), S, D, P, n)
            rel_meta[rel] = {"edges": int(len(S)), "rows_in_release": listed}
        from .twibot22 import _write_edge_index
        _write_edge_index(dp, {"friend": 0, "follower": 1})
        write_json(dp.graph / "meta.json", {
            "dataset": DS, "relations": rel_meta, "edge_type_ids": {"friend": 0, "follower": 1},
            "semantics": {"friend": "friends.csv rows: source follows target",
                          "follower": "followers.csv rows as released (source = follower, target = followed)"},
            "edge_index_order": "friend rows (subset order, csv order) then follower rows",
            "stats": graph_stats(DS, ["friend", "follower"]),
        })
        shutil.rmtree(work)
        st.done(n_users=n, n_tweets=int(offsets[-1]), edges={k: v["edges"] for k, v in rel_meta.items()})
    return meta


def benchmark_split(users: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Per-subset contiguous 70/20/10 split in index order, as found in the MGTAB Cresci-15 tensors
    (train = first floor(0.7 n_s), val = next floor(0.2 n_s), test = rest of each subset block).
    ``verify_against_mgtab`` checks it against the tensors' train/val/test_idx."""
    split = np.empty(len(users), dtype=object)
    for sub in SUBSET_ORDER:
        idx = np.flatnonzero(users["source_file"].to_numpy() == sub)
        k = len(idx)
        a, b = int(k * 0.7), int(k * 0.7) + int(k * 0.2)
        split[idx[:a]] = "train"
        split[idx[a:b]] = "val"
        split[idx[b:]] = "test"
    return split, "per-subset contiguous 70/20/10 (reconstructed from MGTAB Cresci-15 tensors)"
