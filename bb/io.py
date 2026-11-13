"""Low-level I/O: atomic writes, append-only .npy writers, memmap loading, digests, page-cache hints."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------------------------
# atomic writes


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def dumps_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=False, default=_json_default, allow_nan=True)


def write_json(path: str | Path, obj: Any) -> None:
    atomic_write_bytes(path, (dumps_json(obj) + "\n").encode())


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def save_npy(path: str | Path, arr: np.ndarray) -> None:
    """Atomically save an array as .npy (writes to a temp file in the same dir, then renames)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            np.save(f, np.ascontiguousarray(arr), allow_pickle=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_npy(path: str | Path, mmap: bool = True) -> np.ndarray:
    """Load a .npy file as a read-only memory map (default) or into RAM."""
    return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)


# ---------------------------------------------------------------------------------------------
# append-only .npy writer with a fixed-size header, so the array can be streamed to disk without
# knowing its length up front and without ever holding it in RAM.

_HEADER_LEN = 256  # total bytes of magic + header; fixed so it can be rewritten in place


def _npy_header(dtype: np.dtype, shape: tuple[int, ...]) -> bytes:
    d = {"descr": np.lib.format.dtype_to_descr(dtype), "fortran_order": False, "shape": shape}
    body = repr(d).encode("latin1")
    magic = b"\x93NUMPY\x01\x00"
    pad = _HEADER_LEN - len(magic) - 2 - len(body) - 1
    if pad < 0:
        raise ValueError("npy header too long")
    hdr = body + b" " * pad + b"\n"
    return magic + len(hdr).to_bytes(2, "little") + hdr


class NpyAppender:
    """Stream rows of a fixed trailing shape to a .npy file.

    >>> w = NpyAppender("x.npy", np.float16, (768,)); w.append(block); w.close()

    Data goes to ``<path>.partial`` and is renamed on ``close()``, so a crashed writer never
    leaves a valid-looking file. ``resume=True`` reopens an existing partial file, truncating it
    to ``resume_rows`` rows (for checkpointed builders).
    """

    def __init__(self, path: str | Path, dtype, row_shape: tuple[int, ...] = (), *,
                 resume_rows: int | None = None, buffering: int = 1 << 22):
        self.path = Path(path)
        self.partial = self.path.with_name(self.path.name + ".partial")
        self.dtype = np.dtype(dtype)
        self.row_shape = tuple(row_shape)
        self.row_bytes = int(self.dtype.itemsize * int(np.prod(self.row_shape, dtype=np.int64)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if resume_rows is not None and self.partial.exists():
            self.rows = int(resume_rows)
            with open(self.partial, "r+b") as f:
                f.truncate(_HEADER_LEN + self.rows * self.row_bytes)
            self._f = open(self.partial, "ab", buffering=buffering)
        else:
            self.rows = 0
            self._f = open(self.partial, "wb", buffering=buffering)
            self._f.write(_npy_header(self.dtype, (0, *self.row_shape)))

    def append(self, block) -> None:
        a = np.ascontiguousarray(block, dtype=self.dtype)
        if a.shape[1:] != self.row_shape:
            a = a.reshape((-1, *self.row_shape))
        self._f.write(a.tobytes(order="C"))
        self.rows += a.shape[0]

    def flush(self) -> None:
        self._f.flush()

    def close(self) -> Path:
        self._f.flush()
        os.fsync(self._f.fileno())
        self._f.close()
        with open(self.partial, "r+b") as f:
            f.write(_npy_header(self.dtype, (self.rows, *self.row_shape)))
        os.replace(self.partial, self.path)
        return self.path

    def abort(self) -> None:
        try:
            self._f.close()
        finally:
            if self.partial.exists():
                self.partial.unlink()


def open_memmap_w(path: str | Path, dtype, shape: tuple[int, ...], *, resume: bool = False) -> np.memmap:
    """Preallocated writable .npy memmap (``<path>.partial`` until ``finalize_memmap``)."""
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    if resume and partial.exists():
        mm = np.lib.format.open_memmap(partial, mode="r+")
        if mm.shape != tuple(shape) or mm.dtype != np.dtype(dtype):
            raise ValueError(f"cannot resume {partial}: shape/dtype mismatch {mm.shape} {mm.dtype}")
        return mm
    return np.lib.format.open_memmap(partial, mode="w+", dtype=dtype, shape=tuple(shape))


def finalize_memmap(mm: np.memmap, path: str | Path) -> Path:
    path = Path(path)
    mm.flush()
    partial = Path(mm.filename)
    del mm
    os.replace(partial, path)
    return path


# ---------------------------------------------------------------------------------------------
# digests (provenance)

_FULL_HASH_LIMIT = 256 << 20


def file_digest(path: str | Path, full_limit: int = _FULL_HASH_LIMIT) -> str:
    """Content digest. Files up to ``full_limit`` bytes are hashed fully (sha256); larger files are
    fingerprinted by size + 16 evenly spaced 1 MiB blocks (prefix ``s:``), which is cheap and
    catches any rebuild in practice."""
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(path, "rb") as f:
        if size <= full_limit:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
            return h.hexdigest()[:16]
        for k in range(16):
            f.seek(int((size - (1 << 20)) * k / 15))
            h.update(f.read(1 << 20))
    return "s:" + h.hexdigest()[:16]


def dir_digest(path: str | Path, pattern: str = "*") -> dict[str, str]:
    path = Path(path)
    return {p.name: file_digest(p) for p in sorted(path.glob(pattern)) if p.is_file()}


def tree_source_digest(paths: list[Path]) -> str:
    """sha256 over the contents of all .py files under the given paths (git-free provenance)."""
    h = hashlib.sha256()
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        files.extend(sorted(p.rglob("*.py")) if p.is_dir() else [p])
    for f in sorted(set(files)):
        h.update(str(f.name).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------------------------
# page-cache hints (page cache from large reads/writes is charged to the memrun cgroup)


def drop_cache(f, sync: bool = False) -> None:
    """posix_fadvise(DONTNEED) on an open file object or fd; optionally fdatasync first so dirty
    pages can actually be dropped. Silently ignored where unsupported (e.g. drvfs)."""
    try:
        fd = f if isinstance(f, int) else f.fileno()
        if sync:
            os.fdatasync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except (OSError, AttributeError, ValueError):
        pass


# ---------------------------------------------------------------------------------------------
# large row-addressed outputs WITHOUT a writable memmap. Dirty pages of a mapped 50 GB file are
# resident and charged to the job's cgroup (a memmap zero-fill took a job to 8.4 GB RSS); pwrite +
# fdatasync + POSIX_FADV_DONTNEED keeps the page-cache footprint at one chunk.


class NpyRowWriter:
    """Fixed-shape .npy written by row ranges with pwrite (``<path>.partial`` until ``close``).

    The file is created sparse (unwritten rows read as zeros). ``resume=True`` reopens an existing
    partial file of the same shape/dtype."""

    def __init__(self, path: str | Path, dtype, shape: tuple[int, ...], *, resume: bool = False,
                 sync_every_bytes: int = 256 << 20):
        self.path = Path(path)
        self.partial = self.path.with_name(self.path.name + ".partial")
        self.dtype = np.dtype(dtype)
        self.shape = tuple(int(s) for s in shape)
        self.row_bytes = self.dtype.itemsize * int(np.prod(self.shape[1:], dtype=np.int64))
        hdr = _npy_header(self.dtype, self.shape)
        self.hlen = len(hdr)
        size = self.hlen + self.row_bytes * self.shape[0]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if resume and self.partial.exists():
            if self.partial.stat().st_size != size:
                raise ValueError(f"cannot resume {self.partial}: size mismatch")
            with open(self.partial, "rb") as f:
                if f.read(self.hlen) != hdr:
                    raise ValueError(f"cannot resume {self.partial}: header mismatch")
        else:
            with open(self.partial, "wb") as f:
                f.write(hdr)
                f.truncate(size)
        self.fd = os.open(self.partial, os.O_RDWR)
        self.sync_every = sync_every_bytes
        self._since = 0

    def write_rows(self, start: int, block: np.ndarray) -> None:
        b = np.ascontiguousarray(block, dtype=self.dtype)
        if b.shape[1:] != self.shape[1:] or start < 0 or start + b.shape[0] > self.shape[0]:
            raise ValueError(f"bad block {b.shape} at row {start} for shape {self.shape}")
        os.pwrite(self.fd, b.tobytes(), self.hlen + start * self.row_bytes)
        self._since += b.nbytes
        if self._since >= self.sync_every:
            self.sync()

    def sync(self) -> None:
        os.fdatasync(self.fd)
        try:
            os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass
        self._since = 0

    def close(self) -> Path:
        self.sync()
        os.close(self.fd)
        os.replace(self.partial, self.path)
        return self.path


def read_rows(path: str | Path, start: int, stop: int, drop: bool = True) -> np.ndarray:
    """Read rows [start, stop) of a .npy with plain reads (not a mapping) and drop them from the
    page cache afterwards; for streaming over files much larger than RAM."""
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        rd = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = rd(f)
        hlen = f.tell()
        row_bytes = dtype.itemsize * int(np.prod(shape[1:], dtype=np.int64))
        stop = min(stop, shape[0])
        f.seek(hlen + start * row_bytes)
        a = np.fromfile(f, dtype=dtype, count=(stop - start) * int(np.prod(shape[1:], dtype=np.int64)))
        if drop:
            try:
                os.posix_fadvise(f.fileno(), hlen + start * row_bytes, (stop - start) * row_bytes,
                                 os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
    return a.reshape((stop - start, *shape[1:]))
