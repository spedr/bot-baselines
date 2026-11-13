"""Shared feature scheme ``botrgcn`` (docs/methods/botrgcn.md 3.1): builder + loader.

Per dataset and variant, ``data/features/<dataset>/<variant>/`` (``DatasetPaths.features``) holds::

    num_prop.f32.npy  [N, k]  z-scored numeric profile properties (mean/std over ALL N rows, ddof=1)
    cat_prop.f32.npy  [N, c]  0/1 profile flags (no normalisation)
    des.f16.npy       [N,768] -> symlink to data/<ds>/emb/<encoder>/des.f16.npy
    tweets.f16.npy    [N,768] -> symlink to the tweet-mean table of the recipe
    edge_index.i64.npy [2,E], edge_type.i64.npy [E]  -> symlinks to data/<ds>/graph/ (release order,
                      directed as released, benchmark type ids; see meta["edge_type_ids"])
    label.i64.npy     [N]     1 bot, 0 human, -1 unlabelled (TwiBot-20 support)
    {train,val,test[,support]}_idx.i64.npy
    meta.json         variant, column names, z-score mean/std, fills, ref date, text sources (+ HF
                      commits), edge counts per type, input digests, ``text_ready``

Variants (``VARIANTS``):

| dataset | variant | num (5 unless noted) | cat | text |
|---|---|---|---|---|
| twibot22 | botrgcn | followers, active_days@2020-09-05 (fill 1), len(name), following, tweet_count | protected, verified, default_profile_image (URL test) | roberta des + first-20 mean |
| twibot22 | botrgcn-uname | as botrgcn but len(username) (the paper's intent; ablation) | same | same |
| twibot20 | botrgcn | followers, active_days@2020-09-01, len(screen_name), friends, statuses (fill 0) | protected, verified, default_profile_image flag | distilroberta des + roberta all-tweet mean |
| twibot20 | botrgcn-b20 | as botrgcn | same | roberta des + roberta first-20 mean |
| twibot20 | botrgcn-orig6 | 6: followers, friends, favourites, statuses, len(screen_name), active_days | 11 flags (original Dataset.py) | as botrgcn |
| cresci15 | botrgcn | followers, active_days@2020-09-01, len(screen_name), friends, listed | 1: default_profile_image (URL test) | roberta des + first-20 mean |

Numeric columns are filled BEFORE z-scoring (spec 3.1.2). The z-score uses all N rows: this is
transductive (unlabelled feature values of val/test/support rows), as in the benchmark.

The builder is idempotent: ``meta.json["input_key"]`` hashes the builder version, the variant and
digests of every input; a matching key means nothing is rebuilt. Text tables that are not built yet
are left out (``text_ready = False``) and linked on the next call once they exist. Writes go to a
temporary directory that replaces the old one with a rename, so readers never see a half-built set.

    tools/memrun --light -- uv run python -m bb.features.botrgcn --dataset cresci15
    from bb.features import load_features
    F = load_features("cresci15")          # builds if missing or stale; memmaps
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..data.emb import BOTRGCN_DEFAULT
from ..data.graph import graph_meta
from ..data.users import load_split, load_users
from ..data.profile import load_profile
from ..io import file_digest, load_npy, read_json, save_npy, write_json
from ..paths import dataset_paths

BUILDER_VERSION = 1

TW22_DEFAULT_IMG = "https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png"
CRESCI_DEFAULT_IMGS = frozenset(
    f"http://a0.twimg.com/sticky/default_profile_images/default_profile_{k}_normal.png" for k in range(7))
ORIG6_FLAGS = ("protected", "geo_enabled", "verified", "contributors_enabled", "is_translator",
               "is_translation_enabled", "profile_background_tile", "profile_use_background_image",
               "has_extended_profile", "default_profile", "default_profile_image")

# real-data invariants (only checked when N equals the real dataset size)
EXPECTED = {"twibot20": (229_580, 227_979), "twibot22": (1_000_000, 3_743_634), "cresci15": (5_301, 14_220)}


# ---------------------------------------------------------------------------------------------
# column helpers (pandas nullable columns -> float64 numpy, nulls kept as NaN until filled)


def _num(s: pd.Series) -> pd.Series:
    return s.astype("float64")


def _len(s: pd.Series) -> pd.Series:
    return s.str.len().astype("float64")


def _days(p: pd.DataFrame, ref: str) -> pd.Series:
    """Whole days (floor of the timedelta) from created_at to ``ref`` (UTC); NaN if missing."""
    return (pd.Timestamp(ref, tz="UTC") - p["created_at"]).dt.days.astype("float64")


def _flag(s: pd.Series) -> pd.Series:
    return s.astype("boolean").fillna(False).astype("float64")


def _img_default(s: pd.Series, defaults) -> pd.Series:
    """1 if the URL is null, '' or one of the default-image URLs, else 0."""
    hit = s.isna() | s.fillna("").isin(sorted(set(defaults) | {""}))
    return hit.astype("float64")


# ---------------------------------------------------------------------------------------------
# recipes


def _tw22_num(p, uname=False):
    return {"followers_count": _num(p["followers_count"]).fillna(0),
            "active_days": _days(p, "2020-09-05").fillna(1),
            ("username_length" if uname else "name_length"): _len(p["screen_name" if uname else "name"]).fillna(0),
            "following_count": _num(p["following_count"]).fillna(0),
            "tweet_count": _num(p["statuses_count"]).fillna(0)}


def _tw22_cat(p):
    return {"protected": _flag(p["protected"]), "verified": _flag(p["verified"]),
            "default_profile_image": _img_default(p["profile_image_url"], {TW22_DEFAULT_IMG})}


def _tw20_num(p):
    return {"followers_count": _num(p["followers_count"]).fillna(0),
            "active_days": _days(p, "2020-09-01").fillna(0),
            "screen_name_length": _len(p["screen_name"]).fillna(0),
            "friends_count": _num(p["following_count"]).fillna(0),
            "statuses_count": _num(p["statuses_count"]).fillna(0)}


def _tw20_cat(p):
    return {"protected": _flag(p["protected"]), "verified": _flag(p["verified"]),
            "default_profile_image": _flag(p["default_profile_image"])}


def _tw20_orig6_num(p):
    return {"followers_count": _num(p["followers_count"]).fillna(0),
            "friends_count": _num(p["following_count"]).fillna(0),
            "favourites_count": _num(p["favourites_count"]).fillna(0),
            "statuses_count": _num(p["statuses_count"]).fillna(0),
            "screen_name_length": _len(p["screen_name"]).fillna(0),
            "active_days": _days(p, "2020-09-01").fillna(0)}


def _tw20_orig6_cat(p):
    return {f: _flag(p[f]) for f in ORIG6_FLAGS}


def _cr15_num(p):
    return {"followers_count": _num(p["followers_count"]).fillna(0),
            "active_days": _days(p, "2020-09-01").fillna(0),
            "screen_name_length": _len(p["screen_name"]).fillna(0),
            "friends_count": _num(p["following_count"]).fillna(0),
            "listed_count": _num(p["listed_count"]).fillna(0)}


def _cr15_cat(p):
    return {"default_profile_image": _img_default(p["profile_image_url"], CRESCI_DEFAULT_IMGS)}


@dataclass(frozen=True)
class Variant:
    dataset: str
    name: str
    columns: tuple[str, ...]
    num: Callable[[pd.DataFrame], dict]
    cat: Callable[[pd.DataFrame], dict]
    ref_date: str
    text_key: str = "botrgcn"     # key into bb.data.emb.BOTRGCN_DEFAULT
    note: str = ""

    @property
    def text(self) -> tuple[tuple[str, str], tuple[str, str]]:
        return BOTRGCN_DEFAULT[(self.dataset, self.text_key)]


_TW22_COLS = ("followers_count", "following_count", "statuses_count", "created_at", "name", "screen_name",
              "protected", "verified", "profile_image_url")
_TW20_COLS = ("followers_count", "following_count", "statuses_count", "favourites_count", "created_at",
              "screen_name", *ORIG6_FLAGS)
_CR15_COLS = ("followers_count", "following_count", "listed_count", "created_at", "screen_name",
              "profile_image_url")

VARIANTS: dict[tuple[str, str], Variant] = {(v_.dataset, v_.name): v_ for v_ in [
    Variant("twibot22", "botrgcn", _TW22_COLS, _tw22_num, _tw22_cat, "2020-09-05",
            note="benchmark twibot_22/preprocess_1.py: col 2 is len(name) (benchmark bug kept)"),
    Variant("twibot22", "botrgcn-uname", _TW22_COLS, lambda p: _tw22_num(p, uname=True), _tw22_cat,
            "2020-09-05", note="ablation: col 2 = len(username), the paper's screen-name length"),
    Variant("twibot20", "botrgcn", _TW20_COLS, _tw20_num, _tw20_cat, "2020-09-01",
            note="benchmark readme 5+3 columns; text as original Dataset.py"),
    Variant("twibot20", "botrgcn-b20", _TW20_COLS, _tw20_num, _tw20_cat, "2020-09-01", text_key="botrgcn-b20",
            note="roberta-base description + first-20 tweet mean"),
    Variant("twibot20", "botrgcn-orig6", _TW20_COLS, _tw20_orig6_num, _tw20_orig6_cat, "2020-09-01",
            text_key="botrgcn-orig6", note="original repo Dataset.py 6 numeric + 11 flags"),
    Variant("cresci15", "botrgcn", _CR15_COLS, _cr15_num, _cr15_cat, "2020-09-01",
            note="benchmark cresci_15/preprocess_1.py: listed_count in col 4, 1 flag"),
]}


def feature_dir(dataset: str, variant: str = "botrgcn") -> Path:
    return dataset_paths(dataset).features(variant)


def _variant(dataset: str, variant: str) -> Variant:
    try:
        return VARIANTS[(dataset, variant)]
    except KeyError:
        known = sorted(v for d, v in VARIANTS if d == dataset)
        raise KeyError(f"no botrgcn variant {variant!r} for {dataset} (known: {known})") from None


def _text_paths(dataset: str, v: Variant) -> dict[str, Path]:
    (de, dn), (te, tn) = v.text
    dp = dataset_paths(dataset)
    return {"des": dp.emb(de) / f"{dn}.f16.npy", "tweets": dp.emb(te) / f"{tn}.f16.npy"}


def _ready(p: Path) -> bool:
    """An embedding table is finished when both the .npy and its .meta.json exist (bb.build.embed)."""
    return p.exists() and p.with_name(p.name[:-len(".f16.npy")] + ".meta.json").exists()


def _input_key(dataset: str, v: Variant) -> tuple[str, dict]:
    dp = dataset_paths(dataset)
    inputs = {"users": file_digest(dp.users), "profile": file_digest(dp.profile),
              "edge_index": file_digest(dp.graph / "edge_index.i64.npy"),
              "edge_type": file_digest(dp.graph / "edge_type.i64.npy")}
    for k, p in _text_paths(dataset, v).items():
        inputs[k] = file_digest(p) if _ready(p) else None
    blob = json.dumps({"builder": BUILDER_VERSION, "variant": v.name, "inputs": inputs}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16], inputs


def _zscore(cols: dict) -> tuple[np.ndarray, dict]:
    x = np.stack([np.asarray(c, dtype=np.float64) for c in cols.values()], 1)
    if np.isnan(x).any():
        raise ValueError("numeric property still has NaN after filling")
    mu = x.mean(0)
    sd = x.std(0, ddof=1) if len(x) > 1 else np.ones(x.shape[1])
    sd = np.where(sd > 0, sd, 1.0)
    return ((x - mu) / sd).astype(np.float32), {"mean": mu.tolist(), "std": sd.tolist()}


def _link(dst: Path, target: Path) -> None:
    os.symlink(os.path.relpath(target, dst.parent), dst)


def build_features(dataset: str, variant: str = "botrgcn", *, force: bool = False, verbose: bool = True) -> dict:
    """Build (or confirm up to date) the feature set; returns its meta. Peak RSS: the projected
    profile columns (TwiBot-22: about 0.3 GB) plus [N, k] float64."""
    v = _variant(dataset, variant)
    out = feature_dir(dataset, variant)
    key, inputs = _input_key(dataset, v)
    if not force and (out / "meta.json").exists():
        old = read_json(out / "meta.json")
        if old.get("input_key") == key:
            if verbose:
                print(f"[botrgcn features] {dataset}/{variant}: up to date (key {key})")
            return old
    t0 = time.time()
    dp = dataset_paths(dataset)
    users = load_users(dataset, columns=["label"])
    n = len(users)
    p = load_profile(dataset, list(v.columns))
    if len(p) != n or not np.array_equal(p["idx"].to_numpy(), np.arange(n)):
        raise ValueError(f"{dataset}: profile rows are not aligned with users.parquet")

    num_cols = v.num(p)
    num, zs = _zscore(num_cols)
    cat_cols = v.cat(p)
    cat = np.stack([np.asarray(c, dtype=np.float32) for c in cat_cols.values()], 1)
    del p

    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    save_npy(tmp / "num_prop.f32.npy", num)
    save_npy(tmp / "cat_prop.f32.npy", cat)
    label = users["label"].to_numpy().astype(np.int64)
    save_npy(tmp / "label.i64.npy", label)
    splits = {}
    for s in ("train", "val", "test", "support"):
        idx = np.asarray(load_split(dataset, s), dtype=np.int64)
        if s != "support" or len(idx):
            save_npy(tmp / f"{s}_idx.i64.npy", idx)
            splits[s] = {"n": int(len(idx)), "bots": int((label[idx] == 1).sum()) if len(idx) else 0}
    for f in ("edge_index.i64.npy", "edge_type.i64.npy"):
        _link(tmp / f, dp.graph / f)
    text = {}
    for k, path in _text_paths(dataset, v).items():
        enc, name = v.text[0] if k == "des" else v.text[1]
        rec = {"encoder": enc, "table": name, "path": str(path.relative_to(dp.root.parent)), "ready": _ready(path)}
        if rec["ready"]:
            _link(tmp / f"{k}.f16.npy", path)
            em = read_json(path.with_name(f"{name}.meta.json"))
            rec["meta"] = {kk: em.get(kk) for kk in ("hf_commit", "rows", "embedded", "pooling", "policy",
                                                    "max_tokens", "selection", "source", "cap") if kk in em}
            shape = load_npy(path).shape
            if shape[0] != n:
                raise ValueError(f"{path}: {shape[0]} rows, expected {n}")
        text[k] = rec

    et = np.asarray(load_npy(dp.graph / "edge_type.i64.npy"))
    ei_shape = load_npy(dp.graph / "edge_index.i64.npy").shape
    gm = graph_meta(dataset)
    type_ids = gm["edge_type_ids"]
    counts = np.bincount(et, minlength=len(type_ids))
    edges = {rel: int(counts[t]) for rel, t in type_ids.items()}
    if ei_shape != (2, len(et)):
        raise ValueError(f"edge_index shape {ei_shape} vs edge_type {len(et)}")
    checks = {}
    if dataset in EXPECTED and EXPECTED[dataset][0] == n:
        target = EXPECTED[dataset][1]
        rel_err = abs(len(et) - target) / target
        checks["edges_total"] = {"ours": int(len(et)), "reference": target, "rel_err": rel_err, "ok": rel_err <= 0.01}
        if rel_err > 0.01:
            raise ValueError(f"{dataset}: {len(et)} edges, reference {target} (off by {rel_err:.2%})")
    zmean, zstd = num.astype(np.float64).mean(0), num.astype(np.float64).std(0, ddof=1)
    checks["num_prop_z"] = {"mean": zmean.tolist(), "std": zstd.tolist(),
                            "ok": bool(np.all(np.abs(zmean) < 1e-3) and np.all(np.abs(zstd - 1) < 1e-3))}
    tweets_meta = {}
    nt = dp.tweets / "n_total.i32.npy"
    if nt.exists():
        ntot = np.asarray(load_npy(nt))
        for s in ("train", "val", "test", "support"):
            if s in splits:
                idx = np.asarray(load_split(dataset, s))
                tweets_meta[s] = {"users_without_tweets": int((ntot[idx] == 0).sum())}
    meta = {
        "scheme": "botrgcn", "variant": v.name, "dataset": dataset, "builder_version": BUILDER_VERSION,
        "input_key": key, "inputs": inputs, "n": n, "note": v.note, "ref_date": v.ref_date,
        "num_columns": list(num_cols), "num_zscore": zs,
        "num_zscore_rows": "all N rows (transductive, label-free; benchmark behaviour)",
        "num_fill": "missing values filled before z-scoring (twibot22 active_days: 1, else 0)",
        "cat_columns": list(cat_cols), "cat_positive": {c: int(cat[:, i].sum()) for i, c in enumerate(cat_cols)},
        "text": text, "text_ready": all(r["ready"] for r in text.values()),
        "edge_type_ids": type_ids, "edges": edges, "edges_total": int(len(et)),
        "edge_direction": "as released (source -> target); PyG messages flow source -> target",
        "splits": splits, "tweets": tweets_meta, "checks": checks,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_seconds": round(time.time() - t0, 2),
    }
    write_json(tmp / "meta.json", meta)
    old_dir = None
    if out.exists():
        old_dir = out.with_name(f".{out.name}.old-{os.getpid()}")
        out.replace(old_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp.replace(out)
    if old_dir is not None:
        shutil.rmtree(old_dir, ignore_errors=True)
    if verbose:
        print(f"[botrgcn features] {dataset}/{variant}: built in {meta['build_seconds']} s "
              f"(N={n}, num {num.shape[1]}, cat {cat.shape[1]}, E={len(et)}, text_ready={meta['text_ready']})")
    return meta


def ensure_features(dataset: str, variant: str = "botrgcn", verbose: bool = False) -> dict:
    """Build if missing or stale (cheap no-op otherwise)."""
    return build_features(dataset, variant, verbose=verbose)


@dataclass
class BotRGCNFeatures:
    """Memory-mapped view of one built feature set (nothing is loaded into RAM except small arrays)."""
    dataset: str
    variant: str
    dir: Path
    meta: dict
    num_prop: np.ndarray          # float32 [N, k]
    cat_prop: np.ndarray          # float32 [N, c]
    des: np.ndarray | None        # float16 memmap [N, 768]
    tweets: np.ndarray | None     # float16 memmap [N, 768]
    edge_index: np.ndarray        # int64 memmap [2, E]
    edge_type: np.ndarray         # int64 memmap [E]
    label: np.ndarray             # int64 [N]
    splits: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.label)

    @property
    def num_relations(self) -> int:
        return len(self.meta["edge_type_ids"])

    def relations_by_type(self) -> list[str]:
        """Relation names ordered by edge-type id (for ``load_in_merged``, whose etype = position)."""
        ids = self.meta["edge_type_ids"]
        return sorted(ids, key=ids.get)

    @property
    def files(self) -> list[Path]:
        """Artifact paths for ``Run(data_files=...)`` provenance (symlinks resolve to the tables)."""
        return sorted(p for p in self.dir.iterdir() if p.is_file())


def load_features(dataset: str, variant: str = "botrgcn", *, build: bool = True,
                  require_text: bool = True) -> BotRGCNFeatures:
    """Memory-mapped feature set. ``build=True`` builds or refreshes it first (idempotent)."""
    d = feature_dir(dataset, variant)
    meta = ensure_features(dataset, variant) if build else read_json(d / "meta.json")
    if require_text and not meta["text_ready"]:
        missing = [r["path"] for r in meta["text"].values() if not r["ready"]]
        raise FileNotFoundError(f"{dataset}/{variant}: text tables not built yet: {missing} (see `bb status`)")
    opt = lambda f: load_npy(d / f) if (d / f).exists() else None  # noqa: E731
    splits = {s: np.asarray(load_npy(d / f"{s}_idx.i64.npy")) for s in ("train", "val", "test", "support")
              if (d / f"{s}_idx.i64.npy").exists()}
    return BotRGCNFeatures(dataset, variant, d, meta, np.asarray(load_npy(d / "num_prop.f32.npy")),
                           np.asarray(load_npy(d / "cat_prop.f32.npy")), opt("des.f16.npy"), opt("tweets.f16.npy"),
                           load_npy(d / "edge_index.i64.npy"), load_npy(d / "edge_type.i64.npy"),
                           np.asarray(load_npy(d / "label.i64.npy")), splits)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build the shared botrgcn feature scheme")
    ap.add_argument("--dataset", required=True, nargs="+")
    ap.add_argument("--variant", nargs="+", default=["botrgcn"], help="or 'all' for every variant of the dataset")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    for ds in a.dataset:
        vs = sorted(v for d, v in VARIANTS if d == ds) if a.variant == ["all"] else a.variant
        for v in vs:
            m = build_features(ds, v, force=a.force)
            print(json.dumps({"dataset": ds, "variant": v, "text_ready": m["text_ready"], "edges": m["edges"],
                              "num_columns": m["num_columns"], "cat_columns": m["cat_columns"],
                              "checks": m["checks"]}, default=str))


if __name__ == "__main__":
    main()
