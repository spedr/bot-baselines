"""Filesystem layout of the shared data layer.

Every artifact lives under ``data_root()`` (default ``<repo>/data``, override with ``BB_DATA_ROOT``);
run outputs live under ``runs_root()`` (default ``<repo>/runs``, override with ``BB_RUNS_ROOT``).
Raw releases are read-only and are located through ``raw_dir(name)`` (override with
``BB_RAW_<NAME>``, e.g. ``BB_RAW_TWIBOT22``).

Layout per dataset (see docs/architecture.md)::

    data/<dataset>/users.parquet            canonical index (idx = row number of every table)
    data/<dataset>/splits/{train,val,test,support}_idx.npy
    data/<dataset>/meta.json                counts, class balance, source digests
    data/<dataset>/profile.parquet          typed profile table, one row per idx
    data/<dataset>/tweets/                  ragged tweet store
    data/<dataset>/emb/<encoder>/           fp16 embedding tables
    data/<dataset>/graph/<relation>.csr/    per-relation CSR (indptr, indices, edge_pos)
    data/<dataset>/extscore/<tag>.parquet        ExtScore columns (+ <tag>.meta.json)
    data/status/<step>.json                 machine-readable build status
    data/logs/<step>.log                    build logs
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS = ("twibot20", "twibot22", "cresci15")

# Raw releases default to <repo>/raw/<dir> (a symlink or copy); override with BB_RAW_<NAME>.
_RAW_DEFAULTS = {
    "twibot22": "twibot-22-full",
    "twibot20": "twibot-20-full",
    "cresci15": "bot-repository/cresci-2015",
    "mgtab_cresci15": "mgtab-cresci-15",
}


def data_root() -> Path:
    return Path(os.environ.get("BB_DATA_ROOT", REPO_ROOT / "data"))


def runs_root() -> Path:
    return Path(os.environ.get("BB_RUNS_ROOT", REPO_ROOT / "runs"))


def raw_dir(name: str) -> Path:
    env = os.environ.get(f"BB_RAW_{name.upper()}")
    if env:
        return Path(env)
    if name not in _RAW_DEFAULTS:
        raise KeyError(f"unknown raw dataset {name!r}")
    return Path(os.environ.get("BB_RAW_ROOT", REPO_ROOT / "raw")) / _RAW_DEFAULTS[name]


def status_dir() -> Path:
    return data_root() / "status"


def logs_dir() -> Path:
    return data_root() / "logs"


@dataclass(frozen=True)
class DatasetPaths:
    """All standard artifact paths of one dataset. Pure path arithmetic, no I/O."""

    name: str

    @property
    def root(self) -> Path:
        return data_root() / self.name

    @property
    def users(self) -> Path:
        return self.root / "users.parquet"

    @property
    def splits(self) -> Path:
        return self.root / "splits"

    def split_idx(self, split: str) -> Path:
        return self.splits / f"{split}_idx.npy"

    @property
    def meta(self) -> Path:
        return self.root / "meta.json"

    @property
    def profile(self) -> Path:
        return self.root / "profile.parquet"

    @property
    def tweets(self) -> Path:
        return self.root / "tweets"

    def emb(self, encoder: str) -> Path:
        return self.root / "emb" / encoder

    @property
    def graph(self) -> Path:
        return self.root / "graph"

    def csr(self, relation: str) -> Path:
        return self.graph / f"{relation}.csr"

    @property
    def extscore(self) -> Path:
        return self.root / "extscore"

    def extscore_table(self, tag: str) -> Path:
        return self.extscore / f"{tag}.parquet"

    def extscore_meta(self, tag: str) -> Path:
        return self.extscore / f"{tag}.meta.json"

    def features(self, scheme: str) -> Path:
        """Feature-scheme outputs (e.g. ``botrgcn``), written by the method that owns the scheme."""
        return data_root() / "features" / self.name / scheme

    def method_dir(self, method: str) -> Path:
        """Method-local structural artifacts (SEBot trees, BotDGT snapshots, ...)."""
        return data_root() / "methods" / method / self.name


def dataset_paths(name: str) -> DatasetPaths:
    return DatasetPaths(name)
