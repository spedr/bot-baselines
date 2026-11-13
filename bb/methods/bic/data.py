"""BIC inputs: per-tweet sequences read straight into pinned buffers, neighbour lists and batch closures.

Text side (spec 3.2). User i's sequence is [des_i, tweet_1 .. tweet_cap] with the tweets taken
from the shared ragged table ``emb/roberta-base/tweets_seq_<sel>.f16.npy`` (rows
``offsets[i]:offsets[i+1]``, release order = newest first), the first ``cap`` of them, zero rows
after the user's last tweet. ``SeqReader`` reads each user's run of rows with one ``os.preadv``
directly into a pinned [B, cap, 768] float16 buffer: the [N, cap, 768] tensor never exists and the
table is never memory-mapped (mapped pages would count as resident memory of the job). The
description token comes from a device-resident table and is prepended on the GPU.

Graph side (spec 3.3). Both lists come from the merged ``edge_index`` (all relations, release
direction, duplicates kept):

- MHA neighbour list of i: endpoints of every edge with i as source or target (``nbr``), self first
  (added by the model); optional dedup and a random cap per epoch (``max_neighbors``).
- GCN edges: in-edges j -> i of the batch targets (``inc``), the star set of the official
  ``NeighborSampler(sizes=[-1])``. Every other node of the batch closure has only its self-loop.

``Closure.build`` returns the node set (targets first), the local GCN edge index and the flat
attention lists (target, key node) of one batch (no padding: one hub with 900 neighbours does not
inflate the other 63 lists). ``index_bug=True`` emulates the published TwiBot20v2 bug (spec
pitfall 1): the GCN's local edge ids (PyG ``NeighborSampler`` numbering: targets, then new sources
in scan order) are used as GLOBAL node ids, so messages flow between global nodes
``0 .. |n_id|-1``; the neighbour attention keeps the correct lists.
"""
from __future__ import annotations

import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------------------------- CSR


def build_csr(owner: np.ndarray, value: np.ndarray, n: int, dedup: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(indptr int64 [n+1], indices int32) grouping ``value`` by ``owner`` (stable: input order
    kept inside a row). ``dedup`` drops repeated (owner, value) pairs, keeping the first."""
    owner = np.asarray(owner, dtype=np.int64)
    value = np.asarray(value, dtype=np.int64)
    if dedup and len(owner):
        key = owner * n + value
        _, first = np.unique(key, return_index=True)
        keep = np.sort(first)
        owner, value = owner[keep], value[keep]
    order = np.argsort(owner, kind="stable")
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(owner, minlength=n), out=ptr[1:])
    return ptr, value[order].astype(np.int32)


def csr_gather(ptr: np.ndarray, ind: np.ndarray, rows: np.ndarray, extra: np.ndarray | None = None):
    """Concatenated rows of a CSR: (segment id per entry [M] int64, values [M] int64) and, with
    ``extra`` (an array aligned with ``ind``), its entries too."""
    rows = np.asarray(rows, dtype=np.int64)
    s, e = ptr[rows], ptr[rows + 1]
    deg = e - s
    tot = int(deg.sum())
    seg = np.repeat(np.arange(len(rows), dtype=np.int64), deg)
    if tot == 0:
        z = np.zeros(0, dtype=np.int64)
        return (seg, z) if extra is None else (seg, z, z)
    starts = np.repeat(s - np.concatenate([[0], np.cumsum(deg)[:-1]]), deg)
    pos = starts + np.arange(tot, dtype=np.int64)
    if extra is None:
        return seg, np.asarray(ind[pos], dtype=np.int64)
    return seg, np.asarray(ind[pos], dtype=np.int64), np.asarray(extra[pos], dtype=np.int64)


@dataclass
class GraphLists:
    """Neighbour lists (both directions, all relations) and in-edges, as in-memory CSRs."""
    n: int
    nbr_ptr: np.ndarray
    nbr_idx: np.ndarray
    inc_ptr: np.ndarray
    inc_src: np.ndarray
    inc_type: np.ndarray | None = None      # relation of each in-edge (``--gnn rgcn`` only)

    @classmethod
    def from_edge_index(cls, edge_index: np.ndarray, n: int, dedup_neighbors: bool = False,
                        edge_type: np.ndarray | None = None) -> "GraphLists":
        src = np.asarray(edge_index[0], dtype=np.int64)
        dst = np.asarray(edge_index[1], dtype=np.int64)
        # neighbour list: per edge, the source lists the target and the target lists the source.
        # Row order = out-neighbours then in-neighbours, each in edge order (spec 7.2 orders by
        # relation; attention is permutation invariant over its keys, so only the multiset matters).
        owner = np.concatenate([src, dst])
        value = np.concatenate([dst, src])
        nbr_ptr, nbr_idx = build_csr(owner, value, n, dedup=dedup_neighbors)
        inc_ptr, inc_src = build_csr(dst, src, n)
        inc_type = None
        if edge_type is not None:   # same stable order as build_csr(dst, src)
            inc_type = np.asarray(edge_type, dtype=np.int64)[np.argsort(dst, kind="stable")].astype(np.int8)
        return cls(n, nbr_ptr, nbr_idx, inc_ptr, inc_src, inc_type)

    def degree_stats(self, rows: np.ndarray) -> dict:
        d = (self.nbr_ptr[rows + 1] - self.nbr_ptr[rows]).astype(np.int64)
        di = (self.inc_ptr[rows + 1] - self.inc_ptr[rows]).astype(np.int64)
        q = lambda a, p: int(np.percentile(a, p)) if len(a) else 0  # noqa: E731
        return {"nbr_max": int(d.max(initial=0)), "nbr_p99": q(d, 99), "nbr_mean": float(d.mean()) if len(d) else 0.0,
                "in_max": int(di.max(initial=0)), "in_p99": q(di, 99), "isolated": int((d == 0).sum())}


@dataclass
class BatchGraph:
    n_id: np.ndarray        # [V] global ids, targets first
    edge_index: np.ndarray  # [2, E] local ids (row 0 source, row 1 target)
    att_seg: np.ndarray     # [M] target (0..B-1) of each attention key, self entries included
    att_node: np.ndarray    # [M] local id of the key/value node
    edge_type: np.ndarray | None = None   # [E] relation per GCN edge (``rgcn``: no self-loops added)


class Closure:
    def __init__(self, lists: GraphLists, max_neighbors: int = 0, index_bug: bool = False, rgcn: bool = False):
        self.g = lists
        self.max_neighbors = int(max_neighbors)
        self.index_bug = index_bug
        self.rgcn = rgcn
        if rgcn and (index_bug or lists.inc_type is None):
            raise ValueError("rgcn needs edge types and cannot emulate the index bug")

    def build(self, targets: np.ndarray, rng: np.random.Generator | None = None) -> BatchGraph:
        g = self.g
        t = np.asarray(targets, dtype=np.int64)
        B = len(t)
        seg, nb = csr_gather(g.nbr_ptr, g.nbr_idx, t)
        if self.max_neighbors and len(nb):
            cnt = np.bincount(seg, minlength=B)
            if cnt.max() > self.max_neighbors:
                r = rng if rng is not None else np.random.default_rng(0)
                key = r.random(len(nb))
                order = np.lexsort((key, seg))           # random order inside each segment
                rank = np.arange(len(nb)) - np.repeat(np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt)
                keep = np.zeros(len(nb), bool)
                keep[order[rank < self.max_neighbors]] = True
                seg, nb = seg[keep], nb[keep]
        if self.rgcn:
            iseg, isrc, ityp = csr_gather(g.inc_ptr, g.inc_src, t, g.inc_type)
        else:
            iseg, isrc = csr_gather(g.inc_ptr, g.inc_src, t)
        if not self.index_bug:
            rest = np.setdiff1d(np.concatenate([nb, isrc]), t)
            n_id = np.concatenate([t, rest])
            loc = _local_map(n_id)
            ei = np.stack([loc(isrc), iseg])
        else:
            # PyG NeighborSampler numbering of the 1-hop star: targets 0..B-1, then new sources in
            # order of first appearance; its local ids are then (wrongly) used as global ids.
            new = isrc[~np.isin(isrc, t)]
            _, first = np.unique(new, return_index=True)
            samp_nid = np.concatenate([t, new[np.sort(first)]])
            sl = _local_map(samp_nid)
            src_num, dst_num = sl(isrc), iseg               # local ids reinterpreted as global ids
            m = len(samp_nid)
            rest = np.setdiff1d(np.concatenate([nb, np.arange(m, dtype=np.int64)]), t)
            n_id = np.concatenate([t, rest])
            loc = _local_map(n_id)
            ei = np.stack([loc(src_num), loc(dst_num)])
        att_seg = np.concatenate([np.arange(B, dtype=np.int64), seg])   # self first, then the list
        att_node = np.concatenate([np.arange(B, dtype=np.int64), loc(nb)])
        if self.rgcn:   # RGCNConv: per-relation mean + root weight, no self-loop edges
            return BatchGraph(n_id, ei.astype(np.int64), att_seg, att_node, ityp)
        ei = add_remaining_self_loops(ei.astype(np.int64), len(n_id))
        return BatchGraph(n_id, ei, att_seg, att_node)


def add_remaining_self_loops(ei: np.ndarray, n: int) -> np.ndarray:
    """PyG 2.8 ``add_remaining_self_loops`` without edge weights (GCNConv's default): every existing
    self-loop edge is dropped and exactly one loop per node is appended."""
    loops = np.arange(n, dtype=np.int64)
    return np.concatenate([ei[:, ei[0] != ei[1]], np.stack([loops, loops])], axis=1)


def _local_map(n_id: np.ndarray):
    sorter = np.argsort(n_id, kind="stable")
    sv = n_id[sorter]

    def f(x):
        x = np.asarray(x, dtype=np.int64)
        p = np.searchsorted(sv, x)
        if len(x) and (p.max(initial=0) >= len(sv) or not np.array_equal(sv[np.minimum(p, len(sv) - 1)], x)):
            raise KeyError("node not in closure")
        return sorter[p]
    return f


# ---------------------------------------------------------------------------------------------- text


def _npy_data_offset(path: Path) -> tuple[int, tuple, np.dtype]:
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        rd = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = rd(f)
        if fortran:
            raise ValueError(f"{path}: Fortran order")
        return f.tell(), shape, dtype


_NATIVE = None


def _native_fill():
    """numba kernel (``nogil``): for each user k, ``pread`` its first ``count[u]`` rows into row k of
    ``buf`` (uint8 [B, cap*row_bytes]), zero the rest of the row, optional ``FADV_DONTNEED``.
    Returns the bytes read, or -1-k on a short read for user k. One call per batch slice, so the
    prefetch threads hold the GIL once per call instead of once per user (efficiency review: the
    per-user Python loop competed for the GIL with the training thread)."""
    global _NATIVE
    if _NATIVE is None:
        import ctypes

        from numba import njit
        libc = ctypes.CDLL(None, use_errno=True)
        pread = libc.pread
        pread.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int64]  # buffer address as size_t
        pread.restype = ctypes.c_ssize_t
        fadv = libc.posix_fadvise
        fadv.argtypes = [ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_int]
        fadv.restype = ctypes.c_int
        dontneed = int(os.POSIX_FADV_DONTNEED)

        @njit(nogil=True)
        def fill(fd, buf, users, start, count, data_off, row_bytes, drop):
            base = buf.ctypes.data
            stride = buf.shape[1]
            tot = 0
            for k in range(users.shape[0]):
                u = users[k]
                want = count[u] * row_bytes
                if want:
                    pos = data_off + start[u] * row_bytes
                    got = 0
                    while got < want:
                        r = pread(fd, base + k * stride + got, want - got, pos + got)
                        if r <= 0:
                            return -1 - k
                        got += r
                    if drop:
                        fadv(fd, pos, want, dontneed)
                    tot += want
                buf[k, want:] = 0
            return tot
        _NATIVE = fill
    return _NATIVE


class SeqReader:
    """Reads the first ``cap`` tweet embeddings of each user with one ``pread`` per user into
    caller-provided float16 buffers [B, cap, D] (zero rows after the last tweet).
    ``backend='native'`` (default) runs the per-user loop in a GIL-free numba kernel;
    ``'python'`` is the original ``os.preadv`` loop (reference; same bytes)."""

    def __init__(self, seq_path: str | Path, offsets: np.ndarray, cap: int, drop_cache: bool = False,
                 io_threads: int = 4, backend: str = "native"):
        self.path = Path(seq_path)
        self.data_off, shape, dtype = _npy_data_offset(self.path)
        if dtype != np.float16 or len(shape) != 2:
            raise ValueError(f"{self.path}: expected float16 [T, D], got {dtype} {shape}")
        self.T, self.D = int(shape[0]), int(shape[1])
        self.row_bytes = self.D * 2
        off = np.asarray(offsets, dtype=np.int64)
        if off[-1] != self.T:
            raise ValueError(f"offsets end at {off[-1]} but the table has {self.T} rows")
        self.start = off[:-1].copy()
        self.n_avail = np.diff(off)
        self.cap = int(cap)
        self.count = np.minimum(self.n_avail, self.cap)
        self.drop_cache = drop_cache
        self.fd = os.open(self.path, os.O_RDONLY)
        self.bytes_read = 0
        self.io_threads = int(io_threads)
        self._pool = ThreadPoolExecutor(self.io_threads, thread_name_prefix="bic-io") if self.io_threads > 1 else None
        if backend not in ("native", "python"):
            raise ValueError(backend)
        self.backend = backend
        if backend == "native":
            self._kernel = _native_fill()
            self._start64 = self.start.astype(np.int64)
            self._count64 = self.count.astype(np.int64)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def fill(self, buf: np.ndarray, users: np.ndarray) -> None:
        """buf: float16 array [>= len(users), cap, D] (C-contiguous). With ``io_threads > 1`` the
        users are split over a small thread pool (preadv releases the GIL; deeper I/O queue)."""
        users = np.asarray(users, dtype=np.int64)
        fn = self._fill_native if self.backend == "native" else self._fill
        if self._pool is not None and len(users) >= 2 * self.io_threads:
            parts = np.array_split(np.arange(len(users)), self.io_threads)
            self.bytes_read += sum(self._pool.map(lambda ix: fn(buf, users, ix), parts))
        else:
            self.bytes_read += fn(buf, users, np.arange(len(users)))

    def _fill_native(self, buf: np.ndarray, users: np.ndarray, ks: np.ndarray) -> int:
        if not len(ks):
            return 0
        lo, hi = int(ks[0]), int(ks[-1]) + 1                 # array_split parts are contiguous
        if buf.dtype != np.float16 or not buf.flags.c_contiguous or buf.shape[1:] != (self.cap, self.D):
            raise ValueError("buf must be a C-contiguous float16 [B, cap, D] array")
        b8 = buf[lo:hi].reshape(hi - lo, -1).view(np.uint8)
        r = self._kernel(self.fd, b8, users[lo:hi], self._start64, self._count64, self.data_off,
                         self.row_bytes, self.drop_cache)
        if r < 0:
            raise OSError(f"short read for user {users[lo - 1 - r]} of {self.path}")
        return int(r)

    def _fill(self, buf: np.ndarray, users: np.ndarray, ks: np.ndarray) -> int:
        nbytes = 0
        for k in ks:
            u = users[k]
            c = int(self.count[u])
            if c:
                view = memoryview(buf[k, :c]).cast("B")
                pos = self.data_off + int(self.start[u]) * self.row_bytes
                want = c * self.row_bytes
                got = 0
                while got < want:
                    r = os.preadv(self.fd, [view[got:]], pos + got)
                    if r <= 0:
                        raise OSError(f"short read at row {self.start[u]} of {self.path}")
                    got += r
                nbytes += want
                if self.drop_cache:
                    os.posix_fadvise(self.fd, pos, want, os.POSIX_FADV_DONTNEED)
            if c < self.cap:
                buf[k, c:] = 0
        return nbytes

    def user_seq(self, u: int) -> np.ndarray:
        """Reference reader (tests): [cap, D] float16 of user u."""
        out = np.zeros((self.cap, self.D), np.float16)
        self.fill(out[None], np.array([u]))
        return out


class Prefetcher:
    """Prepares batch k+1.. (text buffer fill + closure) on a worker thread while the GPU runs batch
    k. ``n_buf`` pinned buffers; a CUDA event per buffer guards reuse until its H2D copy is done."""

    def __init__(self, reader: SeqReader, closure: Closure, max_batch: int, device, n_buf: int | None = None):
        import torch
        self.reader, self.closure, self.device = reader, closure, torch.device(device)
        if n_buf is None:   # look further ahead when buffers are small (absorbs read-latency spikes)
            n_buf = int(np.clip(64e6 // (max_batch * reader.cap * reader.D * 2), 3, 8))
        pin = self.device.type == "cuda"
        self.bufs = [torch.zeros((max_batch, reader.cap, reader.D), dtype=torch.float16, pin_memory=pin)
                     for _ in range(n_buf)]
        self.events = [None] * n_buf
        self.n_buf = n_buf
        self.ex = ThreadPoolExecutor(1, thread_name_prefix="bic-prefetch")
        self.stats = {"read_s": 0.0, "closure_s": 0.0, "wait_s": 0.0, "batches": 0}

    def close(self) -> None:
        self.ex.shutdown(wait=True)
        self.bufs = []

    def _prepare(self, users: np.ndarray, k: int, rng_seed: int | None):
        ev = self.events[k]
        if ev is not None:
            ev.synchronize()
        buf = self.bufs[k]
        t0 = time.perf_counter()
        self.reader.fill(buf.numpy(), users)
        t1 = time.perf_counter()
        rng = np.random.default_rng(rng_seed) if rng_seed is not None else None
        bg = self.closure.build(users, rng)
        self.stats["read_s"] += t1 - t0
        self.stats["closure_s"] += time.perf_counter() - t1
        return users, k, bg

    def iterate(self, batches: list[np.ndarray], rng_seeds: list[int | None] | None = None):
        """Yields (users np [B], text fp16 device tensor [B, cap, D], BatchGraph)."""
        import torch
        nb = self.n_buf
        rng_seeds = rng_seeds or [None] * len(batches)
        futs = deque()

        def submit(i):
            futs.append(self.ex.submit(self._prepare, batches[i], i % nb, rng_seeds[i]))

        for i in range(min(nb - 1, len(batches))):
            submit(i)
        for i in range(len(batches)):
            t0 = time.perf_counter()
            users, k, bg = futs.popleft().result()
            self.stats["wait_s"] += time.perf_counter() - t0
            self.stats["batches"] += 1
            n = len(users)
            text = self.bufs[k][:n].to(self.device, non_blocking=True)
            if self.device.type == "cuda":
                ev = torch.cuda.Event()
                ev.record()
                self.events[k] = ev
            else:
                text = text.clone()
            if i + nb - 1 < len(batches):
                submit(i + nb - 1)
            yield users, text, bg
