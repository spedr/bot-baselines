"""Machine-readable status files for long-running build steps: ``data/status/<step>.json``.

Schema (all keys always present)::

    step, state ("running" | "done" | "failed" | "interrupted"), pid, host, cmd,
    started_at, updated_at, finished_at, elapsed_s,
    unit, done, total, progress (0..1 or null), rate_per_s, eta_s,
    peak_rss_bytes, cgroup, errors (list of str), extra (dict), log

A reader can tell a dead job from a live one by ``state == "running"`` and ``pid`` not alive
(see ``read_status`` which adds ``alive``).
"""
from __future__ import annotations

import os
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import read_json, write_json
from .mem import cgroup_memory, peak_rss_bytes
from .paths import status_dir


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StepStatus:
    def __init__(self, step: str, total: int | None = None, unit: str = "rows",
                 log: str | None = None, min_interval_s: float = 10.0):
        self.step = step
        self.path = status_dir() / f"{step}.json"
        self.total = total
        self.unit = unit
        self.min_interval_s = min_interval_s
        self.t0 = time.time()
        self._last = 0.0
        self._done0 = 0  # rows already done at (re)start, excluded from the rate
        self.state: dict[str, Any] = {
            "step": step, "state": "running", "pid": os.getpid(), "host": socket.gethostname(),
            "cmd": " ".join(sys.argv), "started_at": _now(), "updated_at": _now(),
            "finished_at": None, "elapsed_s": 0.0, "unit": unit, "done": 0, "total": total,
            "progress": None, "rate_per_s": None, "eta_s": None, "peak_rss_bytes": 0,
            "cgroup": {}, "errors": [], "extra": {}, "log": log,
        }

    def start(self, done: int = 0, **extra) -> "StepStatus":
        self._done0 = done
        self.state["done"] = done
        self.state["extra"].update(extra)
        self._write()
        return self

    def update(self, done: int, total: int | None = None, force: bool = False, **extra) -> None:
        if total is not None:
            self.total = total
        self.state["done"] = int(done)
        self.state["extra"].update(extra)
        now = time.time()
        if force or now - self._last >= self.min_interval_s:
            self._write()

    def error(self, msg: str) -> None:
        self.state["errors"].append(msg)
        self._write()

    def done(self, **extra) -> None:
        self.state["state"] = "done"
        self.state["finished_at"] = _now()
        if self.total is not None:
            self.state["done"] = self.state["done"] or self.total
        self.state["extra"].update(extra)
        self._write()

    def fail(self, exc: BaseException | str) -> None:
        self.state["state"] = "failed"
        self.state["finished_at"] = _now()
        msg = exc if isinstance(exc, str) else "".join(traceback.format_exception(exc))[-4000:]
        self.state["errors"].append(msg)
        self._write()

    def _write(self) -> None:
        now = time.time()
        self._last = now
        el = now - self.t0
        done = self.state["done"]
        rate = (done - self._done0) / el if el > 0 else None
        self.state.update(
            updated_at=_now(), elapsed_s=round(el, 1), total=self.total,
            rate_per_s=round(rate, 2) if rate else None,
            progress=(round(done / self.total, 5) if self.total else None),
            eta_s=(round((self.total - done) / rate) if (self.total and rate) else None),
            peak_rss_bytes=peak_rss_bytes(), cgroup=cgroup_memory(),
        )
        write_json(self.path, self.state)

    # context manager: marks failure on exceptions, interrupted on KeyboardInterrupt
    def __enter__(self) -> "StepStatus":
        return self

    def __exit__(self, et, ev, tb) -> bool:
        if et is None:
            if self.state["state"] == "running":
                self.done()
        elif issubclass(et, KeyboardInterrupt):
            self.state["state"] = "interrupted"
            self.state["finished_at"] = _now()
            self._write()
        else:
            self.fail(ev)
        return False


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def read_status(step: str) -> dict | None:
    p = status_dir() / f"{step}.json"
    if not p.exists():
        return None
    s = read_json(p)
    s["alive"] = s.get("state") == "running" and _alive(int(s.get("pid", -1)))
    return s


def all_status() -> list[dict]:
    d = status_dir()
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        s = read_status(p.stem)
        if s:
            out.append(s)
    return out


def format_status(s: dict) -> str:
    st = s["state"]
    if st == "running" and not s.get("alive", True):
        st = "DEAD(running-state, pid gone)"
    prog = f"{100 * s['progress']:.1f}%" if s.get("progress") is not None else "-"
    eta = f"{s['eta_s'] / 60:.0f}min" if s.get("eta_s") else "-"
    rss = f"{s['peak_rss_bytes'] / 1e9:.2f}GB"
    return (f"{s['step']:<34} {st:<10} {prog:>7} done={s['done']} {s['unit']} "
            f"eta={eta} peakRSS={rss} elapsed={s['elapsed_s'] / 60:.1f}min errors={len(s['errors'])}")


def step_is_done(step: str) -> bool:
    s = read_status(step)
    return bool(s and s["state"] == "done")


def status_path(step: str) -> Path:
    return status_dir() / f"{step}.json"
