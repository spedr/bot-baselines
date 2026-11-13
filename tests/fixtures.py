"""Synthetic raw releases in the exact on-disk formats of TwiBot-20, TwiBot-22 and Cresci-2015
(tiny; no dependence on the real data)."""
from __future__ import annotations

import csv
import io
import json
import tarfile
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------------------------
# TwiBot-20: four JSON arrays (pretty printed), v1.1 profiles as strings with trailing spaces


def tw20_profile(uid: str, k: int, none_desc=False, empty_desc=False) -> dict:
    return {
        "id": f"{uid} ", "id_str": f"{uid} ", "name": f"Name {k} ", "screen_name": f"user{k}x{k % 3} ",
        "location": "None " if k % 2 else "Paris ",
        "description": "None " if none_desc else (" " if empty_desc else f"desc of {k} "),
        "url": "None " if k % 2 else "https://t.co/x ", "entities": "{'description': {'urls': []}} ",
        "protected": "False ", "followers_count": f"{10 * k} ", "friends_count": f"{k + 1} ",
        "listed_count": "0 ", "created_at": f"Tue Nov 18 10:27:{k % 60:02d} +0000 2008 ",
        "favourites_count": f"{k} ", "utc_offset": "None ", "time_zone": "None ",
        "geo_enabled": "True ", "verified": "True " if k % 5 == 0 else "False ",
        "statuses_count": f"{100 + k} ", "lang": "None ", "contributors_enabled": "False ",
        "is_translator": "False ", "is_translation_enabled": "False ",
        "profile_background_tile": "False ",
        "profile_image_url": "http://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png ",
        "profile_use_background_image": "True ", "has_extended_profile": "False ",
        "default_profile": "True ", "default_profile_image": "True " if k % 4 == 0 else "False ",
    }


def make_twibot20(root: Path, sizes=(6, 3, 2, 8)) -> dict:
    """Users get ids 1000+k. Labelled users list neighbours (incl. one unknown id); support user 0
    lists too. Returns expectations."""
    root.mkdir(parents=True, exist_ok=True)
    files = ["train_bak.json", "dev.json", "test_bak.json", "support.json"]
    k = 0
    all_ids = [str(1000 + i) for i in range(sum(sizes))]
    exp = {"n": sum(sizes), "tweets": [], "edges_following": [], "edges_follower": []}
    for f, size in zip(files, sizes):
        recs = []
        for _ in range(size):
            uid = all_ids[k]
            labelled = f != "support.json"
            tweets = None if k == 2 else [f"tweet {j} of {uid} 😀\n" for j in range(k % 4)]
            nb = None
            if labelled and k % 3 != 1 or (not labelled and k == sum(sizes[:3])):
                fol = [all_ids[(k + 1) % len(all_ids)], "999999"]  # second is outside the node set
                fwr = [all_ids[(k + 2) % len(all_ids)]]
                nb = {"following": fol, "follower": fwr}
                exp["edges_following"].append((k, (k + 1) % len(all_ids)))
                exp["edges_follower"].append((k, (k + 2) % len(all_ids)))
            rec = {"ID": uid, "profile": tw20_profile(uid, k, none_desc=(k == 3), empty_desc=(k == 4)),
                   "tweet": tweets, "neighbor": nb, "domain": ["Politics"]}
            if k == 5:
                rec["profile"] = None
            if labelled:
                rec["label"] = str(k % 2)
            recs.append(rec)
            exp["tweets"].append(tweets)
            k += 1
        (root / f).write_text(json.dumps(recs, indent=4, ensure_ascii=True))
    # decoys that must never be read
    (root / "train.json").write_text("not json")
    (root / "test.json").write_text("not json")
    return exp


# ---------------------------------------------------------------------------------------------
# TwiBot-22: user.json array, label/split csv (different row orders), edge csv, tweet_i.json arrays


def make_twibot22(root: Path, n_users=12, n_files=2, tweets_per_file=60) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    users = []
    for i in range(n_users):
        users.append({
            "created_at": f"2015-01-{i % 28 + 1:02d} 02:02:55+00:00",
            "description": None if i == 1 else ("" if i == 2 else f"about {i}"),
            "entities": None, "id": f"u{5000 + i}", "location": None if i % 2 else "Rome",
            "name": f"N{i}", "pinned_tweet_id": None,
            "profile_image_url": "https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png",
            "protected": False, "public_metrics": {"followers_count": i, "following_count": 2 * i,
                                                   "tweet_count": 3 * i, "listed_count": 0},
            "url": "", "username": f"user_{i}", "verified": i % 3 == 0, "withheld": None})
    (root / "user.json").write_text(json.dumps(users))
    ids = [u["id"] for u in users]
    split = ["train"] * (n_users // 2) + ["val"] * (n_users // 4)
    split += ["test"] * (n_users - len(split))
    lab = ["bot" if i % 3 == 0 else "human" for i in range(n_users)]
    with open(root / "label.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i in range(n_users):
            w.writerow([ids[i], lab[i]])
    with open(root / "split.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "split"])
        for i in reversed(range(n_users)):  # different order than user.json
            w.writerow([ids[i], split[i]])
    edges = [("source_id", "relation", "target_id")]
    exp_follow = {"following": [], "followers": []}
    row = 0
    for i in range(n_users):
        for rel, j in (("following", (i + 1) % n_users), ("followers", (i + 3) % n_users), ("post", None)):
            if rel == "post":
                edges.append((ids[i], "post", f"t{i}"))
            else:
                edges.append((ids[i], rel, ids[j]))
                exp_follow[rel].append((i, j, row))
            row += 1
    edges.append(("u5000", "following", "u77"))  # non-user endpoint: dropped
    with open(root / "edge-002.csv", "w", newline="") as f:
        csv.writer(f).writerows(edges)
    # tweets: user u5000+ (t % n_users) authors; one null text; one non-user author
    exp_tweets = {i: [] for i in range(n_users)}
    rank = 0
    for fi in range(n_files):
        tw = []
        for p in range(tweets_per_file):
            a = (p * 7 + fi) % n_users
            text = None if (fi == 0 and p == 5) else (f"RT @x: t{fi}.{p} é" if p % 4 == 0 else f"t{fi}.{p} 😀")
            tw.append({"attachments": None, "author_id": 5000 + a, "created_at": f"2022-02-27 04:59:{p % 60:02d}+00:00",
                       "entities": {"hashtags": [{"tag": "a"}] * (p % 2), "urls": [], "user_mentions": [{"id": 1}] * (p % 3)},
                       "id": f"t{1000000 + fi * 1000 + p}", "in_reply_to_user_id": 1 if p % 5 == 0 else None,
                       "lang": "en" if p % 2 else "fr", "possibly_sensitive": False,
                       "public_metrics": {"retweet_count": p, "reply_count": None, "like_count": 1, "quote_count": None},
                       "referenced_tweets": None, "source": "web", "text": text, "withheld": None})
            exp_tweets[a].append((fi * 1_000_000_000 + p, text))
        tw.append({"author_id": 123, "id": "t1", "text": "foreign"})
        (root / f"tweet_{fi}.json").write_text(json.dumps(tw))
    return {"ids": ids, "split": split, "label": lab, "follow": exp_follow, "tweets": exp_tweets,
            "files": [root / f"tweet_{k}.json" for k in range(n_files)]}


# ---------------------------------------------------------------------------------------------
# Cresci-2015: tar.gz of <SUB>.csv.zip each holding users/tweets/friends/followers csv

CRESCI_USER_COLS = ["id", "name", "screen_name", "statuses_count", "followers_count", "friends_count",
                    "favourites_count", "listed_count", "created_at", "url", "lang", "time_zone", "location",
                    "default_profile", "default_profile_image", "geo_enabled", "profile_image_url",
                    "profile_use_background_image", "profile_background_tile", "utc_offset", "protected",
                    "verified", "description", "dataset"]
CRESCI_TWEET_COLS = ["created_at", "id", "text", "source", "user_id", "truncated", "in_reply_to_status_id",
                     "in_reply_to_user_id", "in_reply_to_screen_name", "retweeted_status_id", "geo", "place",
                     "retweet_count", "reply_count", "favorite_count", "num_hashtags", "num_urls",
                     "num_mentions", "timestamp"]


def _csv_bytes(header, rows) -> bytes:
    s = io.StringIO()
    w = csv.writer(s, quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        s.write(",".join("" if v is None else '"' + str(v).replace('"', '""') + '"' for v in r) + "\n")
    return s.getvalue().encode()


def make_cresci15(root: Path, sizes=None) -> dict:
    sizes = sizes or {"E13": 10, "TFP": 4, "INT": 5, "TWT": 3, "FSF": 6}
    root.mkdir(parents=True, exist_ok=True)
    uid = 100
    ids = {}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for sub in ["E13", "FSF", "INT", "TFP", "TWT"]:  # archive order differs from index order
            n = sizes[sub]
            ids[sub] = [str(uid + i) for i in range(n)]
            uid += 1000
            urows = [[u, f"n{u}", f"sn{u}", 5, 10, 3, 1, 2, "Fri Apr 06 10:58:22 +0000 2007", None, "en", None,
                      None, "1", None, None, "http://a0.twimg.com/x_normal.png", "1", None, None, None, None,
                      None if i % 2 else "", sub] for i, u in enumerate(ids[sub])]
            trows = []
            for i, u in enumerate(ids[sub]):
                for j in range(i % 3):
                    trows.append(["Fri Apr 06 10:58:22 +0000 2012", 10 ** 9 + int(u) * 10 + j, f"tw {u} {j}", "web",
                                  u, None, "0", "0", None, "NULL", None, None, "1", "0", "2", "0", "1", "1",
                                  "2012-04-06 12:58:22"])
            fr = [[ids[sub][i], ids[sub][(i + 1) % n]] for i in range(n)] + [[ids[sub][0], "42"]]
            fo = [[ids[sub][(i + 1) % n], ids[sub][i]] for i in range(n)]
            zb = io.BytesIO()
            with zipfile.ZipFile(zb, "w") as z:
                z.writestr("users.csv", _csv_bytes(CRESCI_USER_COLS, urows))
                z.writestr("tweets.csv", _csv_bytes(CRESCI_TWEET_COLS, trows))
                z.writestr("friends.csv", _csv_bytes(["source_id", "target_id"], fr))
                z.writestr("followers.csv", _csv_bytes(["source_id", "target_id"], fo))
            data = zb.getvalue()
            ti = tarfile.TarInfo(f"{sub}.csv.zip")
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
    (root / "cresci-2015.csv.tar.gz").write_bytes(buf.getvalue())
    return {"ids": ids, "sizes": sizes}
