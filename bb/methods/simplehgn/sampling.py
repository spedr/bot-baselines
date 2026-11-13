"""Neighbour samplers for Simple-HGN / HGT.

- Simple-HGN samples over the union of relations (homogeneous NeighborLoader in the benchmark):
  the shared ``bb.harness.sampler.NeighborSampler`` on ``load_in_merged``.
- HGT samples ``k`` neighbours **per edge type** per hop (benchmark hetero NeighborLoader,
  ``num_neighbors=[256, 256]`` applied to every edge type). ``RelationSampler`` does that on the same
  merged in-neighbour CSC, whose entries are sorted by (target, etype, release position), so each
  relation's in-neighbours of a node are one contiguous run. Same output contract as the shared
  sampler (seeds first in ``n_id``; local ``edge_index`` source -> target; ``e_id`` = CSC entry).

With every fan-out -1 both samplers return the exact L-hop in-neighbourhood, so a model on the
subgraph gives the seed nodes exactly their full-graph outputs (used for full-batch-equivalent
training and for evaluation).

``UnionSampler`` is the shared sampler's algorithm run through the same kernel as
``RelationSampler`` with one relation (no etype scan): identical output to
``bb.harness.sampler.NeighborSampler`` for every (seeds, fanouts, rng_seed) (tested), but compiled
with ``nogil=True`` so that ``train.py`` can sample the next batches in a background thread while
the main thread drives the GPU. Each call seeds numba's (thread-local) generator itself, so a
batch does not depend on the thread that samples it.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from ...harness.sampler import NeighborSampler, SampledSubgraph, _choose


@njit(cache=True, nogil=True)
def _sample_rel(indptr, indices, etype, n_rel, seeds, fanouts, local, rng_seed):
    np.random.seed(rng_seed)
    ns = seeds.shape[0]
    nodes = seeds.copy()
    for i in range(ns):
        local[seeds[i]] = i
    fs, fe = 0, ns
    n_hops = fanouts.shape[0]
    hop_nodes = np.zeros(n_hops + 1, np.int64)
    hop_nodes[0] = ns
    src_parts = []
    dst_parts = []
    eid_parts = []
    kmax = 1
    for h in range(n_hops):
        for r in range(n_rel):
            if fanouts[h, r] > kmax:
                kmax = fanouts[h, r]
    scratch = np.empty(kmax, np.int64)
    for h in range(n_hops):
        cnt = 0
        for i in range(fs, fe):
            v = nodes[i]
            a = indptr[v]
            b = indptr[v + 1]
            p = a
            for r in range(n_rel):
                q = p
                if n_rel == 1:
                    q = b
                while q < b and etype[q] == r:
                    q += 1
                d = q - p
                k = fanouts[h, r]
                cnt += d if (k < 0 or d <= k) else k
                p = q
        esrc = np.empty(cnt, np.int64)
        edst = np.empty(cnt, np.int64)
        eid = np.empty(cnt, np.int64)
        new = np.empty(cnt, np.int64)
        nn = 0
        c = 0
        n_cur = nodes.shape[0]
        for i in range(fs, fe):
            v = nodes[i]
            a = indptr[v]
            b = indptr[v + 1]
            p = a
            for r in range(n_rel):
                q = p
                if n_rel == 1:
                    q = b
                while q < b and etype[q] == r:
                    q += 1
                d = q - p
                k = fanouts[h, r]
                take = d if (k < 0 or d <= k) else k
                if take == d:
                    for t in range(d):
                        scratch_ok = p + t
                        u = indices[scratch_ok]
                        if local[u] < 0:
                            local[u] = n_cur + nn
                            new[nn] = u
                            nn += 1
                        esrc[c] = local[u]
                        edst[c] = i
                        eid[c] = scratch_ok
                        c += 1
                else:
                    _choose(p, d, k, scratch)
                    for t in range(k):
                        j = scratch[t]
                        u = indices[j]
                        if local[u] < 0:
                            local[u] = n_cur + nn
                            new[nn] = u
                            nn += 1
                        esrc[c] = local[u]
                        edst[c] = i
                        eid[c] = j
                        c += 1
                p = q
        src_parts.append(esrc)
        dst_parts.append(edst)
        eid_parts.append(eid)
        nodes = np.concatenate((nodes, new[:nn]))
        fs, fe = n_cur, n_cur + nn
        hop_nodes[h + 1] = nodes.shape[0]
    for i in range(nodes.shape[0]):
        local[nodes[i]] = -1
    tot = 0
    for part in src_parts:
        tot += part.shape[0]
    src = np.empty(tot, np.int64)
    dst = np.empty(tot, np.int64)
    eids = np.empty(tot, np.int64)
    o = 0
    for z in range(len(src_parts)):
        L = src_parts[z].shape[0]
        src[o:o + L] = src_parts[z]
        dst[o:o + L] = dst_parts[z]
        eids[o:o + L] = eid_parts[z]
        o += L
    return nodes, src, dst, eids, hop_nodes


class RelationSampler(NeighborSampler):
    """Per-relation fan-outs: ``fanouts`` is a list over hops of an int (same for every relation)
    or a list of R ints. ``-1`` takes every neighbour of that relation."""

    def __init__(self, csc, fanouts, max_nodes: int | None = None):
        if csc.etype is None:
            raise ValueError("RelationSampler needs a merged CSC with etype (load_in_merged)")
        self.etype = np.ascontiguousarray(np.asarray(csc.etype, dtype=np.int64))
        self.n_rel = int(self.etype.max()) + 1 if len(self.etype) else 1
        rows = [[f] * self.n_rel if np.isscalar(f) else list(f) for f in fanouts]
        if any(len(r) != self.n_rel for r in rows):
            raise ValueError(f"need {self.n_rel} fan-outs per hop")
        super().__init__(csc, [max(r) if min(r) >= 0 else -1 for r in rows], max_nodes)
        self.rel_fanouts = np.asarray(rows, dtype=np.int64)
        self.fanouts = self.rel_fanouts  # reported by dry_run

    def sample(self, seeds: np.ndarray, rng_seed: int = 0) -> SampledSubgraph:
        seeds = np.ascontiguousarray(np.asarray(seeds, dtype=np.int64))
        if len(np.unique(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        nodes, src, dst, eid, hops = _sample_rel(self.indptr, self.indices, self.etype, self.n_rel, seeds,
                                                 self.rel_fanouts, self.local, int(rng_seed) % (2**32 - 1))
        if self.max_nodes is not None and len(nodes) > self.max_nodes:
            raise MemoryError(f"sampled subgraph has {len(nodes)} nodes > max_nodes={self.max_nodes}")
        return SampledSubgraph(nodes, np.stack([src, dst]), eid, len(seeds), hops)


class UnionSampler(NeighborSampler):
    """``NeighborSampler`` (fan-out per hop over the union of relations) with the GIL released while
    sampling; same output as the shared sampler."""

    _no_etype = np.zeros(1, np.int64)

    def sample(self, seeds: np.ndarray, rng_seed: int = 0) -> SampledSubgraph:
        seeds = np.ascontiguousarray(np.asarray(seeds, dtype=np.int64))
        if len(np.unique(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        nodes, src, dst, eid, hops = _sample_rel(self.indptr, self.indices, self._no_etype, 1, seeds,
                                                 self.fanouts.reshape(-1, 1), self.local,
                                                 int(rng_seed) % (2**32 - 1))
        if self.max_nodes is not None and len(nodes) > self.max_nodes:
            raise MemoryError(f"sampled subgraph has {len(nodes)} nodes > max_nodes={self.max_nodes}")
        return SampledSubgraph(nodes, np.stack([src, dst]), eid, len(seeds), hops)


def make_sampler(csc, fanouts, per_relation: bool):
    if per_relation:
        return RelationSampler(csc, fanouts)
    return UnionSampler(csc, [int(f) for f in fanouts])
