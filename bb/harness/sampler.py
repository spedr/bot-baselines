"""Bounded fan-out neighbour sampling over an on-disk in-neighbour CSR (numba).

Semantics follow PyG ``NeighborLoader`` (flow source_to_target): a node aggregates from the
sources of its incoming edges, so we sample from the in-neighbour view (``load_in_merged`` /
``load_transpose``). Hop h+1 expands only the nodes first reached at hop h. Seeds come first in
``n_id``; ``edge_index`` is local (row 0 = source/neighbour, row 1 = target), ``e_id`` indexes the
CSR entries (use ``csr.etype[e_id]`` / ``csr.edge_pos[e_id]`` for relation ids / release order).
Sampling is without replacement; ``fanout = -1`` takes every neighbour. Results are a pure function
of (seeds, fanouts, rng_seed), so an epoch can be re-sampled deterministically instead of cached.

    from bb.data import load_in_merged
    from bb.harness.sampler import NeighborSampler
    csc = load_in_merged("twibot22", ["followers", "following"])
    s = NeighborSampler(csc, [256, 256])
    print(s.dry_run(train_idx, batch_size=256, n_batches=200))   # mandatory before training
    for b in s.iter_batches(train_idx, 256, shuffle=True, epoch_seed=epoch): ...
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numba import njit


@njit(cache=True)
def _choose(a: int, d: int, k: int, out: np.ndarray) -> None:
    """k distinct offsets in [a, a+d) into out[:k] (selection sampling for small d, rejection otherwise)."""
    if d <= 4 * k:
        need = k
        j = 0
        for t in range(d):
            if np.random.random() * (d - t) < need:
                out[j] = a + t
                j += 1
                need -= 1
                if need == 0:
                    break
    else:
        j = 0
        while j < k:
            r = a + np.random.randint(0, d)
            dup = False
            for q in range(j):
                if out[q] == r:
                    dup = True
                    break
            if not dup:
                out[j] = r
                j += 1


@njit(cache=True)
def _sample(indptr, indices, seeds, fanouts, local, rng_seed):
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
    scratch = np.empty(max(1, fanouts.max()), np.int64)
    for h in range(n_hops):
        k = fanouts[h]
        cnt = 0
        for i in range(fs, fe):
            v = nodes[i]
            d = indptr[v + 1] - indptr[v]
            cnt += d if (k < 0 or d <= k) else k
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
            d = indptr[v + 1] - a
            if k < 0 or d <= k:
                m = d
                for t in range(d):
                    scratch_t = a + t
                    u = indices[scratch_t]
                    if local[u] < 0:
                        local[u] = n_cur + nn
                        new[nn] = u
                        nn += 1
                    esrc[c] = local[u]
                    edst[c] = i
                    eid[c] = scratch_t
                    c += 1
            else:
                _choose(a, d, k, scratch)
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
        src_parts.append(esrc)
        dst_parts.append(edst)
        eid_parts.append(eid)
        nodes = np.concatenate((nodes, new[:nn]))
        fs, fe = n_cur, n_cur + nn
        hop_nodes[h + 1] = nodes.shape[0]
    for i in range(nodes.shape[0]):
        local[nodes[i]] = -1
    tot = 0
    for p in src_parts:
        tot += p.shape[0]
    src = np.empty(tot, np.int64)
    dst = np.empty(tot, np.int64)
    eids = np.empty(tot, np.int64)
    o = 0
    for q in range(len(src_parts)):
        L = src_parts[q].shape[0]
        src[o:o + L] = src_parts[q]
        dst[o:o + L] = dst_parts[q]
        eids[o:o + L] = eid_parts[q]
        o += L
    return nodes, src, dst, eids, hop_nodes


@dataclass
class SampledSubgraph:
    n_id: np.ndarray        # int64 [n]  global ids, seeds first
    edge_index: np.ndarray  # int64 [2, E] local (source -> target)
    e_id: np.ndarray        # int64 [E]  CSR entry ids
    batch_size: int         # number of seeds (= first rows of n_id)
    hop_nodes: np.ndarray   # cumulative node counts after each hop

    @property
    def num_nodes(self) -> int:
        return len(self.n_id)

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    def to_torch(self, device=None) -> dict:
        import torch
        return {"n_id": torch.from_numpy(self.n_id).to(device),
                "edge_index": torch.from_numpy(self.edge_index).to(device),
                "e_id": torch.from_numpy(self.e_id).to(device), "batch_size": self.batch_size}


class NeighborSampler:
    def __init__(self, csc, fanouts: list[int], max_nodes: int | None = None):
        """``csc``: a ``bb.data.CSR`` in-neighbour view (rows = targets). ``max_nodes``: raise if a
        sampled subgraph exceeds it (a hard VRAM guard; callers can split the seed batch)."""
        self.csc = csc
        self.indptr = np.ascontiguousarray(np.asarray(csc.indptr, dtype=np.int64))
        self.indices = np.ascontiguousarray(np.asarray(csc.indices, dtype=np.int64))
        self.fanouts = np.asarray(fanouts, dtype=np.int64)
        self.local = np.full(len(self.indptr) - 1, -1, dtype=np.int64)
        self.max_nodes = max_nodes

    def sample(self, seeds: np.ndarray, rng_seed: int = 0) -> SampledSubgraph:
        seeds = np.ascontiguousarray(np.asarray(seeds, dtype=np.int64))
        if len(np.unique(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        nodes, src, dst, eid, hops = _sample(self.indptr, self.indices, seeds, self.fanouts, self.local,
                                             int(rng_seed) % (2**32 - 1))
        if self.max_nodes is not None and len(nodes) > self.max_nodes:
            raise MemoryError(f"sampled subgraph has {len(nodes)} nodes > max_nodes={self.max_nodes}")
        return SampledSubgraph(nodes, np.stack([src, dst]), eid, len(seeds), hops)

    def iter_batches(self, seeds: np.ndarray, batch_size: int, shuffle: bool = True, epoch_seed: int = 0):
        """Deterministic given (seeds, batch_size, shuffle, epoch_seed)."""
        seeds = np.asarray(seeds, dtype=np.int64)
        order = np.random.default_rng(epoch_seed).permutation(len(seeds)) if shuffle else np.arange(len(seeds))
        for b, s in enumerate(range(0, len(seeds), batch_size)):
            yield self.sample(seeds[order[s:s + batch_size]], rng_seed=epoch_seed * 1_000_003 + b)

    def dry_run(self, seeds: np.ndarray, batch_size: int, n_batches: int = 200, seed: int = 0,
                feature_bytes_per_node: int = 0) -> dict:
        """Sample ``n_batches`` random seed batches without training; report percentiles of subgraph
        sizes (nodes, edges, per hop), sampling time, and optional gathered-feature bytes."""
        rng = np.random.default_rng(seed)
        seeds = np.asarray(seeds, dtype=np.int64)
        nn, ne, tt = [], [], []
        hops = []
        self.sample(seeds[:min(len(seeds), 2)], 0)  # JIT warm-up
        for b in range(n_batches):
            sb = rng.choice(seeds, size=min(batch_size, len(seeds)), replace=False)
            t = time.perf_counter()
            g = self.sample(sb, rng_seed=b)
            tt.append(time.perf_counter() - t)
            nn.append(g.num_nodes)
            ne.append(g.num_edges)
            hops.append(np.diff(g.hop_nodes))
        q = [50, 90, 99, 100]

        def pct(v):
            return {f"p{x}": float(np.percentile(v, x)) for x in q}

        rep = {"batch_size": batch_size, "fanouts": self.fanouts.tolist(), "n_batches": n_batches,
               "nodes": pct(nn), "edges": pct(ne), "ms_per_batch": pct(np.array(tt) * 1e3),
               "new_nodes_per_hop_p99": np.percentile(np.stack(hops), 99, axis=0).tolist()}
        if feature_bytes_per_node:
            rep["gathered_feature_MB_p99"] = rep["nodes"]["p99"] * feature_bytes_per_node / 1e6
        return rep
