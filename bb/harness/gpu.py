"""Host -> GPU transfer of memory-mapped tables without materialising them in host RAM."""
from __future__ import annotations

import numpy as np


def memmap_to_gpu(arr: np.ndarray, *, rows: np.ndarray | None = None, dtype=None, device: str = "cuda",
                  chunk_rows: int = 50_000):
    """Copy a (memmapped) [N, D] array, or the given ``rows`` of it, into one preallocated device
    tensor through a single pinned staging buffer of ``chunk_rows`` rows. Host memory stays at one
    chunk (1M x 768 fp16 = 1.5 GB is moved through a ~77 MB buffer).

    ``rows`` should be sorted for sequential reads; the output keeps the order of ``rows``."""
    import torch

    n = arr.shape[0] if rows is None else len(rows)
    tail = tuple(arr.shape[1:])
    src_dt = torch.from_numpy(np.zeros(0, dtype=arr.dtype)).dtype
    dt = dtype or src_dt
    out = torch.empty((n, *tail), dtype=dt, device=device)
    use_pin = str(device).startswith("cuda")
    stage = torch.empty((min(chunk_rows, max(n, 1)), *tail), dtype=src_dt, pin_memory=use_pin)
    for s in range(0, n, chunk_rows):
        e = min(n, s + chunk_rows)
        blk = arr[s:e] if rows is None else arr[np.asarray(rows[s:e])]
        st = stage[: e - s]
        st.copy_(torch.from_numpy(np.ascontiguousarray(blk)))
        out[s:e].copy_(st, non_blocking=False)
    del stage
    return out


def gather_rows(arr: np.ndarray, idx: np.ndarray, dtype=None) -> np.ndarray:
    """Gather rows of a memmap in sorted order (sequential I/O), returned in ``idx`` order."""
    idx = np.asarray(idx)
    order = np.argsort(idx, kind="stable")
    out = np.empty((len(idx), *arr.shape[1:]), dtype=dtype or arr.dtype)
    out[order] = arr[idx[order]]
    return out
