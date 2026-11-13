"""Helpers shared by the dataset builders."""
from __future__ import annotations

import codecs
import json
import queue
import threading
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..io import save_npy, write_json
from ..paths import DatasetPaths
from ..schema import PROFILE_COLUMNS, PROFILE_SCHEMA, SPLITS, USERS_SCHEMA


def write_index(dp: DatasetPaths, users: pd.DataFrame, meta: dict) -> dict:
    """Write users.parquet, splits/*.npy and meta.json. ``users`` must have the USERS_SCHEMA
    columns with idx == arange(N). Returns the meta dict that was written."""
    n = len(users)
    if not np.array_equal(users["idx"].to_numpy(), np.arange(n)):
        raise ValueError("users.idx must be 0..N-1 in row order")
    if users["user_id"].duplicated().any():
        raise ValueError("duplicate user_id in index")
    tbl = pa.Table.from_pandas(users[[f.name for f in USERS_SCHEMA]], schema=USERS_SCHEMA,
                               preserve_index=False)
    dp.root.mkdir(parents=True, exist_ok=True)
    tmp = dp.users.with_suffix(".parquet.tmp")
    pq.write_table(tbl, tmp)
    tmp.replace(dp.users)
    split = users["split"].astype(str).to_numpy()
    label = users["label"].to_numpy()
    counts = {}
    for s in SPLITS:
        idx = np.flatnonzero(split == s).astype(np.int64)
        if len(idx) or s != "support":
            save_npy(dp.split_idx(s), idx)
        lab = label[idx]
        counts[s] = {"n": int(len(idx)), "bots": int((lab == 1).sum()), "humans": int((lab == 0).sum()),
                     "unlabelled": int((lab == -1).sum()),
                     "bot_rate": (float((lab == 1).mean()) if len(idx) and s != "support" else None)}
    meta = {"dataset": dp.name, "n_users": n, "splits": counts, **meta}
    write_json(dp.meta, meta)
    return meta


class ProfileWriter:
    """Row-group parquet writer for profile.parquet; holds at most ``rows_per_group`` rows."""

    def __init__(self, path: Path, rows_per_group: int = 100_000):
        self.path = Path(path)
        self.tmp = self.path.with_suffix(".parquet.tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = pq.ParquetWriter(self.tmp, PROFILE_SCHEMA, compression="zstd")
        self.rows_per_group = rows_per_group
        self.buf: list[dict] = []
        self.n = 0

    def add(self, row: dict) -> None:
        if row["idx"] != self.n + len(self.buf):
            raise ValueError(f"profile rows out of order: got idx {row['idx']}, expected {self.n + len(self.buf)}")
        self.buf.append(row)
        if len(self.buf) >= self.rows_per_group:
            self._flush()

    def _flush(self) -> None:
        if not self.buf:
            return
        cols = {c: [r.get(c) for r in self.buf] for c in PROFILE_COLUMNS}
        self.writer.write_table(pa.table(cols, schema=PROFILE_SCHEMA))
        self.n += len(self.buf)
        self.buf = []

    def close(self) -> int:
        self._flush()
        self.writer.close()
        self.tmp.replace(self.path)
        return self.n


def profile_null_summary(path: Path) -> dict:
    """Per-column null counts of a written profile table (streamed by row group)."""
    pf = pq.ParquetFile(path)
    nulls: dict[str, int] = {}
    n = pf.metadata.num_rows
    for rg in range(pf.metadata.num_row_groups):
        md = pf.metadata.row_group(rg)
        for c in range(md.num_columns):
            col = md.column(c)
            st = col.statistics
            name = col.path_in_schema
            if st is not None and st.has_null_count:
                nulls[name] = nulls.get(name, 0) + st.null_count
    absent = sorted(k for k, v in nulls.items() if v == n and k != "idx")
    return {"n_rows": n, "null_counts": nulls, "absent_columns": absent}


# ---------------------------------------------------------------------------------------------
# streaming JSON array reader (one top-level array of objects; no line structure assumed)


def _reader_thread(f, q: queue.Queue, chunk: int) -> None:
    try:
        while True:
            b = f.read(chunk)
            q.put(b)
            if not b:
                return
    except BaseException as e:  # pragma: no cover
        q.put(e)


class JsonArrayStream:
    """Iterate over the elements of a top-level JSON array in a file of any size.

    Bytes are read sequentially by a background thread (I/O overlaps parsing), decoded
    incrementally as UTF-8 and parsed with the C ``json`` scanner (``raw_decode``). Memory is
    bounded by ``chunk`` x (queue depth + 2) plus the largest single element (an element larger
    than ``max_element`` bytes raises). While iterating, ``stream.offset()`` returns the byte offset
    just after the last yielded element; ``start_byte`` resumes at such an offset (or 0)."""

    def __init__(self, path: str | Path, chunk: int = 16 << 20, depth: int = 4,
                 start_byte: int = 0, drop_cache_every: int = 64 << 20,
                 max_element: int = 256 << 20):
        self.path = Path(path)
        self.chunk = chunk
        self.depth = depth
        self.start_byte = start_byte
        self.drop_cache_every = drop_cache_every
        self.max_element = max_element
        self.offset = lambda: self.start_byte

    def __iter__(self) -> Iterator:
        from ..io import drop_cache

        dec = json.JSONDecoder()
        raw_decode = dec.raw_decode
        f = open(self.path, "rb", buffering=0)
        if self.start_byte:
            f.seek(self.start_byte)
        q: queue.Queue = queue.Queue(maxsize=self.depth)
        th = threading.Thread(target=_reader_thread, args=(f, q, self.chunk), daemon=True)
        th.start()
        udec = codecs.getincrementaldecoder("utf-8")()
        buf = ""
        buf_ascii = True
        buf_byte0 = self.start_byte  # byte offset of buf[0]
        pos = 0
        eof = False
        started = self.start_byte > 0
        since_drop = 0

        def more() -> bool:
            nonlocal buf, buf_ascii, buf_byte0, pos, eof, since_drop
            if eof:
                return False
            b = q.get()
            if isinstance(b, BaseException):
                raise b
            if not b:
                eof = True
                s = udec.decode(b"", final=True)
                if not s:
                    return False
            else:
                since_drop += len(b)
                if since_drop >= self.drop_cache_every:
                    drop_cache(f)
                    since_drop = 0
                s = udec.decode(b)
            # discard consumed prefix
            consumed = buf[:pos]
            buf_byte0 += len(consumed) if buf_ascii else len(consumed.encode("utf-8"))
            buf = buf[pos:] + s
            buf_ascii = buf.isascii()
            pos = 0
            return True

        def byte_offset(p: int) -> int:
            return buf_byte0 + (p if buf_ascii else len(buf[:p].encode("utf-8")))

        self.offset = lambda: byte_offset(pos)

        try:
            while True:
                # skip whitespace / separators / the opening bracket
                while True:
                    n = len(buf)
                    while pos < n and buf[pos] in " \t\r\n,":
                        pos += 1
                    if pos < n:
                        break
                    if not more():
                        return
                c = buf[pos]
                if not started:
                    if c != "[":
                        raise ValueError(f"{self.path}: expected a JSON array, got {c!r}")
                    started = True
                    pos += 1
                    continue
                if c == "]":
                    return
                while True:
                    try:
                        obj, end = raw_decode(buf, pos)
                        # an element ending exactly at the buffer end may be a truncated number
                        if end == len(buf) and not eof:
                            raise json.JSONDecodeError("at buffer end", buf, end)
                        break
                    except json.JSONDecodeError:
                        if len(buf) - pos > self.max_element:
                            raise ValueError(f"{self.path}: unparseable element at byte {byte_offset(pos)}")
                        if not more():
                            raise ValueError(f"{self.path}: truncated JSON element at byte {byte_offset(pos)}")
                pos = end
                yield obj
        finally:
            f.close()


def bucket_bounds(n: int, n_buckets: int) -> np.ndarray:
    """Boundaries of ``n_buckets`` contiguous idx ranges covering 0..n."""
    return np.linspace(0, n, n_buckets + 1).round().astype(np.int64)
