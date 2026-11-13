"""TwiBot-20 builder: one sequential pass over train_bak.json, dev.json, test_bak.json, support.json
producing users/splits/meta, profile.parquet, the tweet store and the graph CSRs.

Index order (benchmark): train (8,278) | dev (2,365) | test (1,183) | support (217,754), each in file
order. ``train.json``/``test.json`` are embedding-sorted copies made by another project and are never
read. Peak RSS is bounded by one 100k-row profile batch + one user record + the edge list.
"""
from __future__ import annotations

import json
import shutil
import time

import numpy as np
import pandas as pd

from ..io import NpyAppender, file_digest, read_json, save_npy, write_json
from ..paths import dataset_paths, raw_dir
from ..schema import empty_profile_row, parse_twitter_date, to_int, tw20_bool, tw20_str
from ..status import StepStatus
from .common import JsonArrayStream, ProfileWriter, profile_null_summary, write_index

DS = "twibot20"
FILES = [("train_bak.json", "train"), ("dev.json", "val"), ("test_bak.json", "test"),
         ("support.json", "support")]
EXPECTED = {"train": 8278, "val": 2365, "test": 1183, "support": 217754}
BOOL_FIELDS = ["protected", "verified", "default_profile", "default_profile_image", "geo_enabled",
               "profile_use_background_image", "profile_background_tile", "has_extended_profile",
               "contributors_enabled", "is_translator", "is_translation_enabled"]


def _profile_row(i: int, p: dict | None) -> dict:
    if p is None:
        return empty_profile_row(i)
    g = p.get
    row = {
        "idx": i, "profile_missing": False,
        "name": tw20_str(g("name")), "screen_name": tw20_str(g("screen_name")),
        "description": tw20_str(g("description")), "url": tw20_str(g("url")),
        "location": tw20_str(g("location")), "profile_image_url": tw20_str(g("profile_image_url")),
        "created_at": parse_twitter_date(g("created_at")),
        "followers_count": to_int(g("followers_count")), "following_count": to_int(g("friends_count")),
        "statuses_count": to_int(g("statuses_count")), "listed_count": to_int(g("listed_count")),
        "favourites_count": to_int(g("favourites_count")),
        "lang": tw20_str(g("lang")), "utc_offset": to_int(tw20_str(g("utc_offset"))),
        "time_zone": tw20_str(g("time_zone")), "pinned_tweet_id": None, "withheld": None,
        "entities": tw20_str(g("entities")),
        "raw_name": g("name"), "raw_screen_name": g("screen_name"), "raw_description": g("description"),
        "raw_location": g("location"), "raw_profile_image_url": g("profile_image_url"),
        "raw_default_profile_image": g("default_profile_image"), "raw_url": g("url"),
        "raw_protected": g("protected"), "raw_verified": g("verified"), "raw_created_at": g("created_at"),
    }
    for f in BOOL_FIELDS:
        row[f] = tw20_bool(g(f))
    return row


def build(force: bool = False) -> dict:
    dp = dataset_paths(DS)
    step = f"{DS}_build"
    if dp.meta.exists() and (dp.tweets / "meta.json").exists() and (dp.graph / "meta.json").exists() and not force:
        print(f"{step}: already built; use --force")
        return read_json(dp.meta)
    raw = raw_dir(DS)
    total_bytes = sum((raw / f).stat().st_size for f, _ in FILES)
    st = StepStatus(step, total=total_bytes, unit="bytes")
    with st:
        st.start()
        t0 = time.time()
        tw_dir = dp.tweets.with_name("tweets.partial")
        if tw_dir.exists():
            shutil.rmtree(tw_dir)
        tw_dir.mkdir(parents=True)
        pw = ProfileWriter(dp.profile)
        rank_app = NpyAppender(tw_dir / "release_rank.npy", np.int64)
        null_app = NpyAppender(tw_dir / "text_null.npy", np.int8)
        toff_app = NpyAppender(tw_dir / "text_offsets.i64.npy", np.int64)
        toff_app.append(np.zeros(1, np.int64))
        blob = open(tw_dir / "text.bin", "wb", buffering=1 << 22)
        tpos = 0
        n_tw_user: list[int] = []
        flags: list[int] = []
        rows = {"raw_id": [], "split": [], "label": [], "raw_order": [], "source_file": []}
        nb_following: list[tuple[int, list]] = []
        nb_follower: list[tuple[int, list]] = []
        domains: dict[str, int] = {}
        i = 0
        done_bytes = 0
        for fname, split in FILES:
            path = raw / fname
            stream = JsonArrayStream(path)
            for k, u in enumerate(stream):
                rows["raw_id"].append(str(u["ID"]))
                rows["split"].append(split)
                lab = u.get("label")
                rows["label"].append(-1 if split == "support" or lab is None else int(str(lab).strip()))
                rows["raw_order"].append(k)
                rows["source_file"].append(fname)
                pw.add(_profile_row(i, u.get("profile")))
                tws = u.get("tweet")
                if tws is None:
                    flags.append(1)
                    tws = []
                else:
                    flags.append(0)
                enc = [b"" if t is None else t.encode("utf-8", "replace") for t in tws]
                lens = np.fromiter((len(b) for b in enc), dtype=np.int64, count=len(enc))
                blob.write(b"".join(enc))
                toff_app.append(tpos + np.cumsum(lens))
                tpos += int(lens.sum())
                rank_app.append(np.arange(len(enc), dtype=np.int64))
                null_app.append(np.fromiter((t is None for t in tws), dtype=np.int8, count=len(tws)))
                n_tw_user.append(len(enc))
                nb = u.get("neighbor")
                if nb is not None:
                    nb_following.append((i, nb.get("following") or []))
                    nb_follower.append((i, nb.get("follower") or []))
                for d in u.get("domain") or []:
                    domains[d] = domains.get(d, 0) + 1
                i += 1
                if k % 2000 == 0:
                    st.update(done_bytes + stream.offset(), users=i)
            done_bytes += path.stat().st_size
            st.update(done_bytes, force=True, users=i)
        n = pw.close()
        blob.close()
        for a in (rank_app, null_app, toff_app):
            a.close()
        counts = np.asarray(n_tw_user, dtype=np.int64)
        offsets = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        save_npy(tw_dir / "offsets.i64.npy", offsets)
        save_npy(tw_dir / "n_total.i32.npy", counts.astype(np.int32))
        save_npy(tw_dir / "user_flags.npy", np.asarray(flags, dtype=np.uint8))
        write_json(tw_dir / "meta.json", {
            "dataset": DS, "n_users": n, "n_tweets": int(offsets[-1]), "cap": None,
            "cap_note": "no cap applied; the release holds at most 200 per user (201 for a few users)",
            "max_per_user": int(counts.max()) if n else 0,
            "order": "order of each user's `tweet` list in the release (the recent 200, newest first)",
            "release_rank": "position in the user's tweet list",
            "columns": {"release_rank": "int64", "text_null": "int8"},
            "no_per_tweet_metadata": "TwiBot-20 tweets are plain strings; derive flags/entities from text",
            "user_flags": "bit 0: the user's `tweet` field is null",
            "users_tweet_null": int(np.sum(flags)), "users_with_tweets": int((counts > 0).sum()),
            "labelled_tweets": int(offsets[11826]) if n >= 11826 else None,
        })
        if dp.tweets.exists():
            shutil.rmtree(dp.tweets)
        tw_dir.replace(dp.tweets)

        # ---- users / splits
        raw_ids = rows["raw_id"]
        users = pd.DataFrame({
            "idx": np.arange(n, dtype=np.int64), "user_id": [r.strip() for r in raw_ids],
            "raw_id": raw_ids, "split": rows["split"], "label": np.asarray(rows["label"], np.int8),
            "raw_order": np.asarray(rows["raw_order"], np.int64), "split_csv_order": -1,
            "label_csv_order": -1, "source_file": rows["source_file"],
        })
        dup = users["user_id"].duplicated(keep="first")
        if dup.any():
            raise ValueError(f"{int(dup.sum())} duplicate user ids across TwiBot-20 files")
        pmeta = profile_null_summary(dp.profile)
        write_json(dp.profile.with_suffix(".meta.json"), pmeta)
        meta = write_index(dp, users, {
            "index_order": "train_bak.json, dev.json, test_bak.json, support.json; file order within each",
            "benchmark_ranges": {"train": [0, 8278], "val": [8278, 10643], "test": [10643, 11826]},
            "sources": {f: file_digest(raw / f) for f, _ in FILES}, "domains": domains,
            "build_seconds": round(time.time() - t0, 1),
        })

        # ---- graph (Dataset.py loop order: node by node, following list then follower list)
        from ..data.graph import build_csr_from_coo, graph_stats
        id2idx = {u: j for j, u in enumerate(users["user_id"].tolist())}
        coo = {"following": ([], [], []), "follower": ([], [], [])}
        pos = 0
        dropped = {"following": 0, "follower": 0}
        listed = {"following": 0, "follower": 0}
        for (src, fol), (_, fwr) in zip(nb_following, nb_follower):
            for rel, lst in (("following", fol), ("follower", fwr)):
                for t in lst:
                    listed[rel] += 1
                    j = id2idx.get(str(t).strip())
                    if j is None:
                        dropped[rel] += 1
                        continue
                    coo[rel][0].append(src)
                    coo[rel][1].append(j)
                    coo[rel][2].append(pos)
                    pos += 1
        rel_meta = {}
        for rel, (s, d, p) in coo.items():
            build_csr_from_coo(dp.csr(rel), np.asarray(s, np.int64), np.asarray(d, np.int64),
                               np.asarray(p, np.int64), n)
            rel_meta[rel] = {"edges": len(s), "listed_ids": listed[rel], "dropped_outside_node_set": dropped[rel]}
        from .twibot22 import _write_edge_index
        _write_edge_index(dp, {"following": 0, "follower": 1})
        users_with_lists = len(nb_following)
        lab_lists = sum(1 for s, _ in nb_following if s < 11826)
        write_json(dp.graph / "meta.json", {
            "dataset": DS, "relations": rel_meta, "edge_type_ids": {"following": 0, "follower": 1},
            "edge_index_order": "original BotRGCN Dataset.py loop: node idx order, following then follower",
            "semantics": {"following": "source = list owner i, target = j in i.neighbor.following",
                          "follower": "source = list owner i, target = j in i.neighbor.follower"},
            "users_with_neighbor_lists": users_with_lists, "labelled_with_lists": lab_lists,
            "support_with_lists": users_with_lists - lab_lists,
            "stats": graph_stats(DS, ["following", "follower"]),
        })
        for s, e in EXPECTED.items():
            if meta["splits"][s]["n"] != e:
                st.error(f"split {s}: {meta['splits'][s]['n']} users, expected {e}")
        st.done(n_users=n, n_tweets=int(offsets[-1]), edges={k: v["edges"] for k, v in rel_meta.items()})
    return meta
