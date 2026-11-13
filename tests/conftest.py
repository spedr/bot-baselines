from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fixtures import make_cresci15, make_twibot20, make_twibot22  # noqa: E402


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("BB_DATA_ROOT", str(root))
    monkeypatch.setenv("BB_RUNS_ROOT", str(tmp_path / "runs"))
    return root


@pytest.fixture
def tw20(tmp_path, data_root, monkeypatch):
    raw = tmp_path / "raw20"
    exp = make_twibot20(raw)
    monkeypatch.setenv("BB_RAW_TWIBOT20", str(raw))
    from bb.build.twibot20 import build
    build()
    return exp


@pytest.fixture
def tw22(tmp_path, data_root, monkeypatch):
    raw = tmp_path / "raw22"
    exp = make_twibot22(raw)
    monkeypatch.setenv("BB_RAW_TWIBOT22", str(raw))
    from bb.build import twibot22
    twibot22.build_users()
    return exp


@pytest.fixture
def cr15(tmp_path, data_root, monkeypatch):
    raw = tmp_path / "rawc"
    exp = make_cresci15(raw)
    monkeypatch.setenv("BB_RAW_CRESCI15", str(raw))
    from bb.build.cresci15 import build
    build()
    return exp
