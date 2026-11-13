"""TwiBot-22 builders (raw release -> shared tables). Each step is idempotent, writes
``data/status/twibot22_<step>.json`` and can be run as ``bb build twibot22 --step <step>``.

Steps
- ``users``  : users.parquet (user.json order), splits, meta.json, profile.parquet (one pass over user.json)
- ``graph``  : graph/{following,followers}.csr + edge_index/edge_type (one pass over edge-002.csv)
- ``tweets`` : tweet store, all tweets up to 200 per user in release order (one pass over tweet_0..8.json,
               then an on-disk group-by); resumable at ~1 GB input granularity.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv

from ..io import NpyAppender, drop_cache, file_digest, load_npy, read_json, save_npy, write_json
from ..paths import dataset_paths, raw_dir
from ..schema import empty_profile_row, parse_twitter_date
from ..status import StepStatus
from .common import JsonArrayStream, ProfileWriter, bucket_bounds, profile_null_summary, write_index

DS = "twibot22"
N_EXPECTED = 1_000_000
DEFAULT_PROFILE_IMAGE = "https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png"


# =============================================================================================
# users + profile


def _profile_row(i: int, u: dict) -> dict:
    pm = u.get("public_metrics") or {}
    ca = u.get("created_at")
    ent = u.get("entities")
    wh = u.get("withheld")
    return {
        "idx": i, "profile_missing": False,
        "name": u.get("name"), "screen_name": u.get("username"),
        "description": u.get("description"), "url": u.get("url"), "location": u.get("location"),
        "profile_image_url": u.get("profile_image_url"), "created_at": parse_twitter_date(ca),
        "followers_count": pm.get("followers_count"), "following_count": pm.get("following_count"),
        "statuses_count": pm.get("tweet_count"), "listed_count": pm.get("listed_count"),
        "favourites_count": None,
        "protected": u.get("protected"), "verified": u.get("verified"),
        "default_profile": None, "default_profile_image": None, "geo_enabled": None,
        "profile_use_background_image": None, "profile_background_tile": None,
        "has_extended_profile": None, "contributors_enabled": None, "is_translator": None,
        "is_translation_enabled": None, "lang": None, "utc_offset": None, "time_zone": None,
        "pinned_tweet_id": None if u.get("pinned_tweet_id") is None else str(u.get("pinned_tweet_id")),
        "withheld": None if wh is None else json.dumps(wh, ensure_ascii=False),
        "entities": None if ent is None else json.dumps(ent, ensure_ascii=False),
        # API v2 values are already typed; raw copies kept for schema uniformity
        "raw_name": u.get("name"), "raw_screen_name": u.get("username"),
        "raw_description": u.get("description"), "raw_location": u.get("location"),
        "raw_profile_image_url": u.get("profile_image_url"), "raw_default_profile_image": None,
        "raw_url": u.get("url"),
        "raw_protected": None if u.get("protected") is None else str(u.get("protected")),
        "raw_verified": None if u.get("verified") is None else str(u.get("verified")),
        "raw_created_at": ca,
    }


def build_users(force: bool = False) -> dict:
    dp = dataset_paths(DS)
    step = f"{DS}_users"
    if dp.users.exists() and dp.profile.exists() and dp.meta.exists() and not force:
        print(f"{step}: already built ({dp.users}); use --force to rebuild")
        return read_json(dp.meta)
    raw = raw_dir(DS)
    st = StepStatus(step, total=N_EXPECTED, unit="users")
    with st:
        st.start()
        t0 = time.time()
        lab = pcsv.read_csv(raw / "label.csv", convert_options=pcsv.ConvertOptions(
            column_types={"id": pa.string(), "label": pa.string()})).to_pandas()
        spl = pcsv.read_csv(raw / "split.csv", convert_options=pcsv.ConvertOptions(
            column_types={"id": pa.string(), "split": pa.string()})).to_pandas()
        lab_pos = pd.Series(np.arange(len(lab)), index=lab["id"])
        spl_pos = pd.Series(np.arange(len(spl)), index=spl["id"])
        raw_ids: list[str] = []
        pw = ProfileWriter(dp.profile)
        n_null_desc = n_empty_desc = 0
        for i, u in enumerate(JsonArrayStream(raw / "user.json")):
            raw_ids.append(u["id"])
            d = u.get("description")
            n_null_desc += d is None
            n_empty_desc += d == ""
            pw.add(_profile_row(i, u))
            if i % 50_000 == 0:
                st.update(i)
        n = pw.close()
        st.update(n, force=True)
        ids = pd.Index(raw_ids)
        if not ids.is_unique:
            raise ValueError("duplicate ids in user.json")
        lp = lab_pos.reindex(ids).to_numpy()
        sp = spl_pos.reindex(ids).to_numpy()
        if np.isnan(lp.astype(float)).any() or np.isnan(sp.astype(float)).any():
            raise ValueError("user.json ids missing from label.csv/split.csv")
        lp = lp.astype(np.int64)
        sp = sp.astype(np.int64)
        labels = lab["label"].to_numpy()[lp]
        splits = spl["split"].to_numpy()[sp]
        splits = np.where(splits == "valid", "val", splits)  # readme says 'valid', file says 'val'
        if set(np.unique(splits)) - {"train", "val", "test"}:
            raise ValueError(f"unexpected split names {np.unique(splits)}")
        if set(np.unique(labels)) - {"bot", "human"}:
            raise ValueError(f"unexpected labels {np.unique(labels)}")
        users = pd.DataFrame({
            "idx": np.arange(n, dtype=np.int64),
            "user_id": [r[1:] for r in raw_ids], "raw_id": raw_ids,
            "split": splits, "label": (labels == "bot").astype(np.int8),
            "raw_order": np.arange(n, dtype=np.int64), "split_csv_order": sp, "label_csv_order": lp,
            "source_file": "user.json",
        })
        same_pos = int((users["split_csv_order"] == users["label_csv_order"]).sum())
        prof = profile_null_summary(dp.profile)
        prof.update(description_null=n_null_desc, description_empty=n_empty_desc,
                    default_profile_image_url=DEFAULT_PROFILE_IMAGE)
        write_json(dp.profile.with_suffix(".meta.json"), prof)
        meta = write_index(dp, users, {
            "index_order": "user.json order",
            "sources": {f: file_digest(raw / f) for f in ("user.json", "label.csv", "split.csv")},
            "user_json_order_equals_label_csv_order": bool((lp == np.arange(n)).all()),
            "user_json_order_equals_split_csv_order": bool((sp == np.arange(n)).all()),
            "split_csv_pos_equals_label_csv_pos": same_pos,
            "build_seconds": round(time.time() - t0, 1),
        })
        st.done(n_users=n, splits=meta["splits"])
    return meta


# =============================================================================================
# graph


def build_graph(force: bool = False, chunk_bytes: int = 64 << 20) -> dict:
    from ..data.graph import build_csr_from_coo, graph_stats

    dp = dataset_paths(DS)
    step = f"{DS}_graph"
    if (dp.graph / "meta.json").exists() and not force:
        print(f"{step}: already built; use --force")
        return read_json(dp.graph / "meta.json")
    from ..data.users import load_users

    users = load_users(DS, columns=["user_id"])
    n = len(users)
    uid = users["user_id"].astype("int64").to_numpy()
    del users
    order = np.argsort(uid)
    sorted_uid = uid[order]

    def lookup(col: pa.Array) -> np.ndarray:
        """'u123' strings -> idx (-1 if not a user), fully vectorised (no Python objects)."""
        num = pc.cast(pc.utf8_slice_codeunits(col, 1), pa.int64(), safe=False)
        ok = pc.starts_with(col, "u").to_numpy(zero_copy_only=False)
        v = num.fill_null(-1).to_numpy(zero_copy_only=False)
        pos = np.clip(np.searchsorted(sorted_uid, v), 0, n - 1)
        hit = ok & (sorted_uid[pos] == v)
        return np.where(hit, order[pos], -1).astype(np.int64)
    path = raw_dir(DS) / "edge-002.csv"
    size = path.stat().st_size
    st = StepStatus(step, total=size, unit="bytes")
    with st:
        st.start()
        t0 = time.time()
        reader = pcsv.open_csv(path, read_options=pcsv.ReadOptions(block_size=chunk_bytes),
                               convert_options=pcsv.ConvertOptions(column_types={
                                   "source_id": pa.string(), "relation": pa.string(), "target_id": pa.string()}))
        rel_counts: dict[str, int] = {}
        keep = {"following": [], "followers": []}
        row0 = 0
        dropped = {"following": 0, "followers": 0}
        rows_after_last_follow = 0
        for b in reader:
            rel = b.column(1)
            vc = pc.value_counts(rel)
            for s in vc:
                k = s["values"].as_py()
                rel_counts[k] = rel_counts.get(k, 0) + s["counts"].as_py()
            for r in ("following", "followers"):
                m = pc.equal(rel, r)
                pos = np.flatnonzero(m.to_numpy(zero_copy_only=False))
                if len(pos):
                    src = lookup(b.column(0).take(pa.array(pos)))
                    dst = lookup(b.column(2).take(pa.array(pos)))
                    ok = (src >= 0) & (dst >= 0)
                    dropped[r] += int((~ok).sum())
                    keep[r].append(np.stack([src[ok], dst[ok], pos[ok] + row0]).astype(np.int64))
            row0 += b.num_rows
            st.update(reader_bytes(reader, size, row0), rows=row0)
        meta_rel = {}
        for r, parts in keep.items():
            coo = np.concatenate(parts, axis=1) if parts else np.zeros((3, 0), np.int64)
            build_csr_from_coo(dp.csr(r), coo[0], coo[1], coo[2], n)
            meta_rel[r] = {"edges": int(coo.shape[1]), "dropped_non_user_endpoints": dropped[r]}
            del coo
        # benchmark edge_index: file order; edge_type followers=0, following=1
        _write_edge_index(dp, {"followers": 0, "following": 1})
        stats = graph_stats(DS, ["following", "followers"])
        meta = {"dataset": DS, "source": str(path), "source_digest": file_digest(path),
                "relations": meta_rel, "edge_type_ids": {"followers": 0, "following": 1},
                "edge_index_order": "edge-002.csv row order (edge_pos ascending)",
                "all_relation_counts": rel_counts, "rows_total": row0, "stats": stats,
                "build_seconds": round(time.time() - t0, 1)}
        write_json(dp.graph / "meta.json", meta)
        st.done(**{k: v["edges"] for k, v in meta_rel.items()})
    return meta


def reader_bytes(reader, size: int, rows: int) -> int:
    # pyarrow's streaming reader does not expose the byte position; estimate from rows
    # (edge-002.csv rows average ~52 bytes). Only used for progress/ETA.
    return min(size, rows * 52)


def _write_edge_index(dp, type_ids: dict[str, int]) -> None:
    from ..data.graph import load_csr

    parts_src, parts_dst, parts_pos, parts_typ = [], [], [], []
    for r, t in type_ids.items():
        c = load_csr(dp.name, r)
        src = np.repeat(np.arange(c.n, dtype=np.int64), np.diff(c.indptr))
        parts_src.append(src)
        parts_dst.append(np.asarray(c.indices, dtype=np.int64))
        parts_pos.append(np.asarray(c.edge_pos, dtype=np.int64))
        parts_typ.append(np.full(len(src), t, dtype=np.int64))
    pos = np.concatenate(parts_pos)
    order = np.argsort(pos, kind="stable")
    ei = np.stack([np.concatenate(parts_src)[order], np.concatenate(parts_dst)[order]])
    save_npy(dp.graph / "edge_index.i64.npy", ei)
    save_npy(dp.graph / "edge_type.i64.npy", np.concatenate(parts_typ)[order])
    save_npy(dp.graph / "edge_pos.i64.npy", pos[order])


# =============================================================================================
# tweets

CAP = 200
REC_DTYPE = np.dtype([
    ("idx", "<i4"), ("text_len", "<i4"), ("release_rank", "<i8"), ("tweet_id", "<i8"),
    ("created_at", "<i8"), ("retweet_count", "<i4"), ("reply_count", "<i4"), ("like_count", "<i4"),
    ("quote_count", "<i4"), ("lang", "<i2"), ("source", "<i4"), ("n_hashtags", "<i2"),
    ("n_urls", "<i2"), ("n_mentions", "<i2"), ("is_retweet", "i1"), ("is_quote", "i1"),
    ("is_reply", "i1"), ("possibly_sensitive", "i1"), ("text_null", "i1"),
])
# columns written to the final store (name -> dtype); text goes to text.bin + text_offsets
OUT_COLUMNS = [n for n in REC_DTYPE.names if n not in ("idx", "text_len")]
_EPOCH = None


def _ts(s) -> int:
    if s is None:
        return -1
    # '2022-02-27 04:59:35+00:00'
    from datetime import datetime
    try:
        return int(datetime.fromisoformat(s).timestamp())
    except (TypeError, ValueError):
        return -1


def _i(v) -> int:
    return -1 if v is None else int(v)


class _Vocab:
    def __init__(self, items: list[str] | None = None):
        self.items: list[str] = list(items or [])
        self.map = {s: i for i, s in enumerate(self.items)}

    def code(self, s) -> int:
        if s is None:
            return -1
        c = self.map.get(s)
        if c is None:
            c = self.map[s] = len(self.items)
            self.items.append(s)
        return c


def build_tweets(force: bool = False, files: list[Path] | None = None, work_dir: Path | None = None,
                 out_dir: Path | None = None, n_buckets: int = 128, flush_rows: int = 200_000,
                 checkpoint_bytes: int = 1 << 30, cap: int = CAP, author_to_idx: dict | None = None,
                 n_users: int | None = None, step: str = f"{DS}_tweets") -> dict:
    """One sequential pass over tweet_0..8.json, then an on-disk group-by idx.

    Phase 1 appends every kept tweet (first ``cap`` per user in release order) to one of
    ``n_buckets`` bucket files by idx range; checkpoints (bucket sizes + per-user counters + input
    byte offset) every ``checkpoint_bytes`` of input. Phase 2 sorts each bucket by idx (stable, so
    release order is kept) and appends it to the final store. ``files``/``author_to_idx``/
    ``n_users``/``out_dir`` exist for tests with synthetic inputs.
    """
    dp = dataset_paths(DS)
    out_dir = Path(out_dir or dp.tweets)
    work_dir = Path(work_dir or (dp.root / "_work_tweets"))
    if (out_dir / "meta.json").exists() and not force:
        print(f"{step}: already built ({out_dir}); use --force")
        return read_json(out_dir / "meta.json")
    if files is None:
        files = [raw_dir(DS) / f"tweet_{k}.json" for k in range(9)]
    if author_to_idx is None:
        from ..data.users import load_users
        u = load_users(DS, columns=["user_id"])
        author_to_idx = {int(x): i for i, x in enumerate(u["user_id"].to_numpy())}
        n_users = len(u)
        del u
    assert n_users is not None
    sizes = [Path(f).stat().st_size for f in files]
    total_bytes = int(sum(sizes))
    work_dir.mkdir(parents=True, exist_ok=True)
    ck_path = work_dir / "checkpoint.json"
    bounds = bucket_bounds(n_users, n_buckets)
    st = StepStatus(step, total=total_bytes, unit="bytes")
    with st:
        # ---------------- resume state
        if force and ck_path.exists() and not (work_dir / "phase1_done").exists():
            pass  # resume is still valid when forcing a rebuild of a half-built pass
        ck = read_json(ck_path) if ck_path.exists() else None
        if ck and (ck.get("files") != [str(f) for f in files] or ck.get("n_buckets") != n_buckets
                   or ck.get("cap") != cap):
            raise RuntimeError(f"checkpoint {ck_path} is for a different configuration; delete {work_dir}")
        if ck:
            counts = load_npy(work_dir / "counts.i32.npy", mmap=False).copy()
            lang = _Vocab(ck["lang_vocab"])
            source = _Vocab(ck["source_vocab"])
            file_i, byte_off, pos_in_file = ck["file_i"], ck["byte_off"], ck["pos_in_file"]
            bsizes = ck["bucket_sizes"]
            stats = ck["stats"]
        else:
            counts = np.zeros(n_users, dtype=np.int32)
            lang, source = _Vocab(), _Vocab()
            file_i, byte_off, pos_in_file = 0, 0, 0
            bsizes = [[0, 0] for _ in range(n_buckets)]
            stats = {"tweets_seen": 0, "tweets_kept": 0, "non_user_author": 0, "null_author": 0,
                     "null_text": 0, "bad_surrogates": 0, "per_file": []}
        # open bucket files, truncated to the checkpointed sizes
        rec_f, txt_f = [], []
        for b in range(n_buckets):
            for kind, lst, sz in (("rec", rec_f, bsizes[b][0]), ("txt", txt_f, bsizes[b][1])):
                p = work_dir / f"b{b:04d}.{kind}"
                fh = open(p, "r+b" if p.exists() else "w+b")
                fh.truncate(sz)
                fh.seek(sz)
                lst.append(fh)
        done_bytes = int(sum(sizes[:file_i]) + byte_off)
        st.start(done=done_bytes, phase="1-stream", file_i=file_i)

        def save_checkpoint(fi: int, off: int, pif: int) -> None:
            for fh in rec_f + txt_f:
                fh.flush()
                os.fdatasync(fh.fileno())
                drop_cache(fh)
            save_npy(work_dir / "counts.i32.npy", counts)
            write_json(ck_path, {
                "files": [str(f) for f in files], "n_buckets": n_buckets, "cap": cap,
                "file_i": fi, "byte_off": off, "pos_in_file": pif,
                "bucket_sizes": [[rec_f[b].tell(), txt_f[b].tell()] for b in range(n_buckets)],
                "lang_vocab": lang.items, "source_vocab": source.items, "stats": stats,
            })

        recs = np.zeros(flush_rows, dtype=REC_DTYPE)
        texts: list[bytes] = []
        nrec = 0

        def flush() -> None:
            nonlocal nrec, texts
            if nrec == 0:
                return
            r = recs[:nrec]
            bk = np.searchsorted(bounds, r["idx"], side="right") - 1
            order = np.argsort(bk, kind="stable")
            bko = bk[order]
            cut = np.flatnonzero(np.diff(bko)) + 1
            starts = np.concatenate([[0], cut])
            ends = np.concatenate([cut, [len(order)]])
            for s, e in zip(starts, ends):
                b = int(bko[s])
                sel = order[s:e]
                rec_f[b].write(r[sel].tobytes())
                txt_f[b].write(b"".join([texts[j] for j in sel.tolist()]))
            nrec = 0
            texts = []

        REF = {"retweeted": 0, "quoted": 1, "replied_to": 2}
        t0 = time.time()
        for fi in range(file_i, len(files)):
            stream = JsonArrayStream(files[fi], start_byte=byte_off if fi == file_i else 0)
            pif = pos_in_file if fi == file_i else 0
            next_ck = (byte_off if fi == file_i else 0) + checkpoint_bytes
            base = sum(sizes[:fi])
            seen_f = kept_f = 0
            for o in stream:
                rank = fi * 1_000_000_000 + pif
                pif += 1
                seen_f += 1
                a = o.get("author_id")
                if a is None:
                    stats["null_author"] += 1
                    continue
                ix = author_to_idx.get(int(a))
                if ix is None:
                    stats["non_user_author"] += 1
                    continue
                c = counts[ix]
                counts[ix] = c + 1
                if c >= cap:
                    continue
                kept_f += 1
                t = o.get("text")
                if t is None:
                    tb = b""
                    stats["null_text"] += 1
                else:
                    try:
                        tb = t.encode("utf-8")
                    except UnicodeEncodeError:
                        tb = t.encode("utf-8", "replace")
                        stats["bad_surrogates"] += 1
                pm = o.get("public_metrics") or {}
                ent = o.get("entities") or {}
                refs = o.get("referenced_tweets")
                if refs is None:
                    # field is null throughout the release sample we inspected: retweets are then
                    # recognised by the 'RT @' text prefix and quotes are unknown (-1)
                    stats["refs_null"] = stats.get("refs_null", 0) + 1
                    fl = [int(t is not None and t.startswith("RT @")), -1, 0]
                else:
                    fl = [0, 0, 0]
                    for rf in refs:
                        k = REF.get(rf.get("type"))
                        if k is not None:
                            fl[k] = 1
                if o.get("in_reply_to_user_id") is not None:
                    fl[2] = 1
                ps = o.get("possibly_sensitive")
                tid = o.get("id")
                recs[nrec] = (
                    ix, len(tb), rank,
                    int(tid[1:]) if isinstance(tid, str) else _i(tid),
                    _ts(o.get("created_at")),
                    _i(pm.get("retweet_count")), _i(pm.get("reply_count")), _i(pm.get("like_count")),
                    _i(pm.get("quote_count")), lang.code(o.get("lang")), source.code(o.get("source")),
                    len(ent.get("hashtags") or ()), len(ent.get("urls") or ()),
                    len(ent.get("user_mentions") or ()), fl[0], fl[1], fl[2],
                    -1 if ps is None else int(ps), int(t is None),
                )
                texts.append(tb)
                nrec += 1
                if nrec == flush_rows:
                    flush()
                    off = stream.offset()
                    st.update(base + off, file_i=fi, tweets_seen=stats["tweets_seen"] + seen_f,
                              phase="1-stream")
                    if off >= next_ck:
                        stats["tweets_seen"] += seen_f
                        stats["tweets_kept"] += kept_f
                        seen_f = kept_f = 0
                        save_checkpoint(fi, off, pif)
                        next_ck = off + checkpoint_bytes
            flush()
            stats["tweets_seen"] += seen_f
            stats["tweets_kept"] += kept_f
            stats["per_file"].append({"file": str(files[fi]), "tweets": pif,
                                      "seconds": round(time.time() - t0, 1)})
            save_checkpoint(fi + 1, 0, 0)
            st.update(sum(sizes[:fi + 1]), force=True, file_i=fi + 1, phase="1-stream")
        for fh in rec_f + txt_f:
            fh.close()
        (work_dir / "phase1_done").touch()
        phase1_s = time.time() - t0

        # ---------------- phase 2: group by idx (bucket by bucket)
        st.update(total_bytes, force=True, phase="2-group")
        tmp_out = out_dir.with_name(out_dir.name + ".partial")
        if tmp_out.exists():
            shutil.rmtree(tmp_out)
        tmp_out.mkdir(parents=True)
        apps = {c: NpyAppender(tmp_out / f"{c}.npy", REC_DTYPE[c]) for c in OUT_COLUMNS}
        text_off = NpyAppender(tmp_out / "text_offsets.i64.npy", np.int64)
        text_off.append(np.zeros(1, np.int64))
        tpos = 0
        kept_per_user = np.zeros(n_users, dtype=np.int64)
        with open(tmp_out / "text.bin", "wb") as tf:
            for b in range(n_buckets):
                r = np.fromfile(work_dir / f"b{b:04d}.rec", dtype=REC_DTYPE)
                blob = (work_dir / f"b{b:04d}.txt").read_bytes()
                if len(r):
                    order = np.argsort(r["idx"], kind="stable")
                    toff = np.concatenate([[0], np.cumsum(r["text_len"].astype(np.int64))])
                    if toff[-1] != len(blob):
                        raise RuntimeError(f"bucket {b}: text bytes {len(blob)} != records {toff[-1]}")
                    rs = r[order]
                    for c in OUT_COLUMNS:
                        apps[c].append(rs[c])
                    # gather texts in sorted order (slices of one bytes object; no per-byte index)
                    lens = rs["text_len"].astype(np.int64)
                    starts = toff[:-1][order]
                    mv = memoryview(blob)
                    tf.write(b"".join([mv[a:a + l] for a, l in zip(starts.tolist(), lens.tolist())]))
                    del mv
                    text_off.append(tpos + np.cumsum(lens))
                    tpos += int(lens.sum())
                    kept_per_user += np.bincount(rs["idx"], minlength=n_users)
                    del rs
                del r, blob
                st.update(total_bytes, phase="2-group", bucket=b + 1, n_buckets=n_buckets)
            tf.flush()
            os.fsync(tf.fileno())
        for a in apps.values():
            a.close()
        text_off.close()
        offsets = np.concatenate([[0], np.cumsum(kept_per_user)]).astype(np.int64)
        save_npy(tmp_out / "offsets.i64.npy", offsets)
        save_npy(tmp_out / "n_total.i32.npy", counts)
        if not np.array_equal(np.minimum(counts, cap), kept_per_user):
            raise RuntimeError("kept-per-user mismatch between phases")
        meta = {
            "dataset": DS, "n_users": int(n_users), "n_tweets": int(offsets[-1]), "cap": cap,
            "order": "release order (tweet_0..8.json, position in file); first `cap` per user kept",
            "release_rank": "file_index * 1e9 + position of the tweet in that file (all tweets counted)",
            "columns": {c: str(REC_DTYPE[c]) for c in OUT_COLUMNS},
            "nulls": {"created_at": -1, "counts": -1, "possibly_sensitive": -1, "lang": -1, "source": -1},
            "flags_source": ("referenced_tweets types (retweeted/quoted/replied_to) when present; when "
                             "referenced_tweets is null: is_retweet = text starts with 'RT @', is_quote = -1 "
                             "(unknown); is_reply also set when in_reply_to_user_id is not null"),
            "entity_counts_source": "len(entities.hashtags/urls/user_mentions); entities null -> 0",
            "text_null": "1 if the tweet's text field is JSON null (stored as empty bytes)",
            "lang_vocab": lang.items, "source_vocab_size": len(source.items),
            "users_with_tweets": int((counts > 0).sum()), "users_beyond_cap": int((counts > cap).sum()),
            "tweets_beyond_cap": int(np.maximum(counts - cap, 0).sum()),
            "stats": stats, "phase1_seconds": round(phase1_s, 1),
            "phase2_seconds": round(time.time() - t0 - phase1_s, 1),
            "sources": {str(f): file_digest(f) for f in files},
        }
        write_json(tmp_out / "source_vocab.json", source.items)
        write_json(tmp_out / "meta.json", meta)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        tmp_out.replace(out_dir)
        shutil.rmtree(work_dir)
        st.done(n_tweets=meta["n_tweets"], users_with_tweets=meta["users_with_tweets"])
    return meta
