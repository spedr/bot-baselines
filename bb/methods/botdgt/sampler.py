"""Per-snapshot 2-hop in-neighbour sampling over prefix views of one shared CSC.

Snapshot t of row v is ``indices[indptr[v] : indptr[v] + row_cnt[t, v]]`` (rows sorted by edge
birth, see ``snapshots.py``), so no per-snapshot graph is materialised. Semantics follow PyG
``NeighborLoader`` like ``bb.harness.sampler`` (seeds first, hop h+1 expands only nodes first
reached at hop h, without replacement, fan-out -1 = all); the only difference is the row extent.

``sample_batch`` samples every snapshot for one seed batch and stacks the T subgraphs into one
block-diagonal graph, so each TransformerConv layer runs once per batch instead of T times:

- ``union`` [U]: distinct global node ids over all snapshots (features are embedded once per batch)
- ``inv`` [S]: stacked node -> row of ``union`` (S = sum_t n_t; snapshot t occupies ``off[t]:off[t+1]``)
- ``ei`` [2, E]: all sampled edges in stacked numbering (layer 1)
- ``ei_seed`` [2, E1]: the edges into seeds, target renumbered to ``t * B + i`` (layer 2 only needs
  the seed outputs: a bipartite TransformerConv on these is exactly the seed rows of the full one)
- ``seed_rows`` [T * B]: stacked row of seed i in snapshot t
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from ...harness.sampler import _choose


@njit(cache=True)
def _sample_prefix(indptr, cnt, indices, seeds, fanouts, local, rng_seed):
    np.random.seed(rng_seed)
    ns = seeds.shape[0]
    nodes = seeds.copy()
    for i in range(ns):
        local[seeds[i]] = i
    fs, fe = 0, ns
    n_hops = fanouts.shape[0]
    src_parts = []
    dst_parts = []
    scratch = np.empty(max(1, fanouts.max()), np.int64)
    for h in range(n_hops):
        k = fanouts[h]
        tot = 0
        for i in range(fs, fe):
            d = cnt[nodes[i]]
            tot += d if (k < 0 or d <= k) else k
        esrc = np.empty(tot, np.int64)
        edst = np.empty(tot, np.int64)
        new = np.empty(tot, np.int64)
        nn = 0
        c = 0
        n_cur = nodes.shape[0]
        for i in range(fs, fe):
            v = nodes[i]
            a = indptr[v]
            d = cnt[v]
            if k < 0 or d <= k:
                for t in range(d):
                    u = indices[a + t]
                    if local[u] < 0:
                        local[u] = n_cur + nn
                        new[nn] = u
                        nn += 1
                    esrc[c] = local[u]
                    edst[c] = i
                    c += 1
            else:
                _choose(a, d, k, scratch)
                for t in range(k):
                    u = indices[scratch[t]]
                    if local[u] < 0:
                        local[u] = n_cur + nn
                        new[nn] = u
                        nn += 1
                    esrc[c] = local[u]
                    edst[c] = i
                    c += 1
        src_parts.append(esrc)
        dst_parts.append(edst)
        nodes = np.concatenate((nodes, new[:nn]))
        fs, fe = n_cur, n_cur + nn
    for i in range(nodes.shape[0]):
        local[nodes[i]] = -1
    tot = 0
    for p in src_parts:
        tot += p.shape[0]
    src = np.empty(tot, np.int64)
    dst = np.empty(tot, np.int64)
    o = 0
    for q in range(len(src_parts)):
        L = src_parts[q].shape[0]
        src[o:o + L] = src_parts[q]
        dst[o:o + L] = dst_parts[q]
        o += L
    return nodes, src, dst


@dataclass
class NestedBatch:
    """One 2-hop sample on the LAST snapshot; snapshot t is the sub-sample of edges born <= t.

    Exact for uncapped fan-outs (every snapshot's 2-hop in-neighbourhood is then a birth-filtered
    subset of the final one, and nodes reached only through later edges never influence the
    seeds at t). With a binding train cap (rows with more than ``fanout`` in-entries) the capped
    rows are subsampled once on the final row rather than independently per snapshot."""
    seeds: np.ndarray       # int64 [B]
    n_id: np.ndarray        # int64 [n] seeds first
    ei: np.ndarray          # int64 [2, E] local
    eb: np.ndarray          # int64 [E] edge birth (local snapshot index within the window)
    e2_src: np.ndarray      # int64 [P] layer-2 pairs (t, edge into a seed born <= t): t * n1 + src
    e2_dst: np.ndarray      # int64 [P] t * B + dst
    T: int
    n_capped_rows: int = 0
    n1: int = -1            # seeds + hop-1 nodes (= n_id[:n1]): the only rows whose layer-1 output is used

    @property
    def batch_size(self) -> int:
        return len(self.seeds)


def nested_batch(snap, seeds: np.ndarray, fanouts, rng_seed: int, local: np.ndarray, t0: int = 0) -> NestedBatch:
    """Sample the final snapshot once; ``t0`` = first snapshot of the window (births shift by t0)."""
    seeds = np.ascontiguousarray(np.asarray(seeds, dtype=np.int64))
    T = snap.T - t0
    fan = np.asarray(fanouts, dtype=np.int64)
    nd, s, d = _sample_prefix(snap.indptr, snap.row_cnt[-1], snap.indices, seeds, fan, local,
                              int(rng_seed) % (2**32 - 1))
    birth = snap.birth[nd].astype(np.int64)
    eb = np.maximum(np.maximum(birth[s], birth[d]) - t0, 0)
    B = len(seeds)
    m = d < B
    s1, d1, b1 = s[m], d[m], eb[m]
    reps = T - b1
    e_rep = np.repeat(np.arange(len(s1)), reps)
    start = np.repeat(np.cumsum(reps) - reps, reps)
    t_rep = b1[e_rep] + (np.arange(len(e_rep)) - start)
    # nodes are numbered seeds first, then hop-1 discoveries (all of them sources of edges into seeds),
    # then hop-2 discoveries; every sampled edge targets a seed or a hop-1 node, and layer 2 reads
    # layer-1 outputs of seeds and hop-1 nodes only, so layer 1 needs output rows n_id[:n1] only
    n1 = max(B, int(s1.max()) + 1) if len(s1) else B
    capped = 0
    if (fan >= 0).any():
        capped = int((snap.row_cnt[-1][nd] > fan.max()).sum())
    return NestedBatch(seeds, nd, np.stack([s, d]), eb, t_rep * n1 + s1[e_rep], t_rep * B + d1[e_rep], T, capped, n1)


@dataclass
class SnapshotBatch:
    seeds: np.ndarray       # int64 [B]
    union: np.ndarray       # int64 [U]
    inv: np.ndarray         # int64 [S]
    ei: np.ndarray          # int64 [2, E]
    ei_seed: np.ndarray     # int64 [2, E1]
    seed_rows: np.ndarray   # int64 [T*B]
    n_t: np.ndarray         # int64 [T] nodes per snapshot subgraph
    e_t: np.ndarray         # int64 [T] edges per snapshot subgraph

    @property
    def batch_size(self) -> int:
        return len(self.seeds)


class SnapshotSampler:
    def __init__(self, snap, fanouts):
        self.snap = snap
        self.fanouts = np.asarray(fanouts, dtype=np.int64)
        self.local = np.full(snap.n, -1, dtype=np.int64)
        self.T = snap.T

    def sample_t(self, t: int, seeds: np.ndarray, rng_seed: int):
        return _sample_prefix(self.snap.indptr, self.snap.row_cnt[t], self.snap.indices, seeds, self.fanouts,
                              self.local, int(rng_seed) % (2**32 - 1))

    def nested(self, seeds: np.ndarray, rng_seed: int = 0, t0: int = 0) -> NestedBatch:
        return nested_batch(self.snap, seeds, self.fanouts, rng_seed, self.local, t0)

    def sample_batch(self, seeds: np.ndarray, rng_seed: int = 0, snapshots=None) -> SnapshotBatch:
        seeds = np.ascontiguousarray(np.asarray(seeds, dtype=np.int64))
        B = len(seeds)
        ts = range(self.T) if snapshots is None else snapshots
        nodes, srcs, dsts, s_src, s_dst = [], [], [], [], []
        off = 0
        n_t, e_t = [], []
        for j, t in enumerate(ts):
            nd, s, d = self.sample_t(t, seeds, rng_seed * 131 + t)
            nodes.append(nd)
            srcs.append(s + off)
            dsts.append(d + off)
            m = d < B
            s_src.append(s[m] + off)
            s_dst.append(d[m] + j * B)
            n_t.append(len(nd))
            e_t.append(len(s))
            off += len(nd)
        allnodes = np.concatenate(nodes)
        union, inv = np.unique(allnodes, return_inverse=True)
        offs = np.cumsum([0] + n_t[:-1])
        seed_rows = (offs[:, None] + np.arange(B)[None, :]).reshape(-1)
        return SnapshotBatch(seeds, union, inv.astype(np.int64), np.stack([np.concatenate(srcs), np.concatenate(dsts)]),
                             np.stack([np.concatenate(s_src), np.concatenate(s_dst)]), seed_rows.astype(np.int64),
                             np.asarray(n_t), np.asarray(e_t))
