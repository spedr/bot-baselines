"""Memory measurement helpers (host RSS from /proc, VRAM from torch)."""
from __future__ import annotations

import os
import resource


def _status_kb(field: str, pid: str = "self") -> int:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def rss_bytes() -> int:
    """Current resident set size of this process."""
    return _status_kb("VmRSS") * 1024


def peak_rss_bytes() -> int:
    """Peak resident set size (VmHWM) of this process."""
    return _status_kb("VmHWM") * 1024


def peak_rss_children_bytes() -> int:
    """Peak RSS over waited-for child processes (e.g. DataLoader workers), from getrusage."""
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024


def cgroup_memory() -> dict:
    """memory.current / memory.peak / memory.max of the enclosing cgroup (memrun scope), if readable."""
    out: dict = {}
    try:
        with open("/proc/self/cgroup") as f:
            rel = f.read().strip().split("::")[-1]
        base = os.path.join("/sys/fs/cgroup", rel.lstrip("/"))
        for k in ("memory.current", "memory.peak", "memory.max"):
            p = os.path.join(base, k)
            if os.path.exists(p):
                with open(p) as g:
                    v = g.read().strip()
                out[k] = int(v) if v.isdigit() else v
    except OSError:
        pass
    return out


def cuda_peak_bytes() -> int:
    """torch.cuda.max_memory_allocated if torch+CUDA are already imported and initialised, else 0."""
    import sys

    torch = sys.modules.get("torch")
    if torch is None:
        return 0
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return int(torch.cuda.max_memory_allocated())
    except Exception:  # pragma: no cover
        return 0
    return 0


def memory_report() -> dict:
    return {
        "peak_rss_bytes": peak_rss_bytes(),
        "peak_rss_children_bytes": peak_rss_children_bytes(),
        "rss_bytes": rss_bytes(),
        "cuda_max_allocated_bytes": cuda_peak_bytes(),
        "cgroup": cgroup_memory(),
    }


def gb(x: int) -> float:
    return round(x / 1e9, 3)
