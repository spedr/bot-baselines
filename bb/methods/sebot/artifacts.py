"""SEBot structural artifacts, cached under ``data/methods/sebot/<dataset>/``.

- whole-graph coding tree (alpha view, spec 3.3): ``tree_k{k}_v2_{smooth|nosmooth}/`` with
  ``node_size.npy``, ``parent_{d}.npy`` (d = 1..k: int32 local parent index at level d of every
  level-(d-1) node), ``edges_{d}.npy`` (d = 0..k-1: int32 [2, E_d] quotient-graph edges) and
  ``meta.json`` (builder stats, input digest). Built on the simple undirected union of the two
  follow relations (``bb.data.load_undirected``), unweighted, isolated nodes included.
- subgraph trees (beta view, spec 3.4): ``subgraphs_hop1_top{t}_{mode}_cap{c}_seed{s}_k{k}.npz``,
  every level of every ego graph concatenated (offsets per level), see ``SubgraphTrees``.

Everything here is label-free: the trees and ego graphs use the graph only (the lone-node stars of
the faithful mode draw from the labelled-user index set, not from labels).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numba as nb
import numpy as np

from ...io import load_npy, read_json, save_npy, write_json
from ...paths import dataset_paths
from .tree import CodingTreeBuilder, TreeResult

TREE_VERSION = 1
SUB_VERSION = 1


def method_dir(ds: str) -> Path:
    return dataset_paths(ds).method_dir("sebot")


def _digest(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------------------------------
# whole-graph tree


def tree_graph(ds: str, rels: list[str]):
    """Simple undirected union of ``rels`` (spec 3.2 'encoding-tree graph')."""
    from ...data import load_undirected
    g = dataset_paths(ds).graph
    for r in (rels, rels[::-1]):
        if (g / ("undirected__" + "+".join(r) + ".csr")).exists():
            return load_undirected(ds, r)
    return load_undirected(ds, rels)


@dataclass
class WholeTree:
    k: int
    node_size: list[int]
    parent: list            # [None, p1, ..., pk]  int64 arrays
    edges: list             # [e0, ..., e_{k-1}]   int64 [2, E]
    meta: dict


def build_whole_tree(ds: str, rels: list[str], k: int, *, smooth: bool = True, rebuild: bool = False,
                     verbose: bool = True) -> WholeTree:
    csr = tree_graph(ds, rels)
    indptr = np.asarray(csr.indptr, np.int64)
    indices = np.asarray(csr.indices, np.int64)
    key = _digest(indptr, indices)
    d = method_dir(ds) / f"tree_k{k}_v2_{'smooth' if smooth else 'nosmooth'}"
    if (d / "meta.json").exists() and not rebuild:
        meta = read_json(d / "meta.json")
        if meta.get("input_key") == key and meta.get("version") == TREE_VERSION:
            ns = [int(x) for x in np.load(d / "node_size.npy")]
            return WholeTree(k, ns, [None] + [np.asarray(load_npy(d / f"parent_{i}.npy"), np.int64) for i in range(1, k + 1)],
                             [np.asarray(load_npy(d / f"edges_{i}.npy"), np.int64) for i in range(k)], meta)
    t = time.time()
    b = CodingTreeBuilder(indptr, indices, smooth=smooth, verbose=verbose)
    res: TreeResult = b.build(k)
    t_build = time.time() - t
    ld = res.layer_data(indptr, indices)
    from ...mem import peak_rss_bytes
    meta = {"version": TREE_VERSION, "input_key": key, "dataset": ds, "relations": rels, "k": k, "mode": "v2",
            "smooth": smooth, "n": res.n, "edges_undirected": int(len(indices) // 2), "node_size": ld["node_size"],
            "build_seconds": round(t_build, 2), "extract_seconds": round(time.time() - t - t_build, 2),
            "peak_rss_bytes": peak_rss_bytes(), "stats": res.stats,
            "isolated": int((np.diff(indptr) == 0).sum())}
    tmp = d.with_name(d.name + f".partial{os.getpid()}")      # per process: concurrent builders never collide
    tmp.mkdir(parents=True, exist_ok=True)
    save_npy(tmp / "node_size.npy", np.asarray(ld["node_size"], np.int64))
    for i in range(1, k + 1):
        save_npy(tmp / f"parent_{i}.npy", ld["parent_local"][i].astype(np.int32))
    for i in range(k):
        save_npy(tmp / f"edges_{i}.npy", ld["edges"][i].astype(np.int32))
    write_json(tmp / "meta.json", meta)
    if d.exists():
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    try:
        tmp.replace(d)
    except OSError:                       # another process published the same tree first
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    if verbose:
        print(f"[sebot] {ds} tree k={k}: node_size={ld['node_size']} build {t_build:.1f}s", flush=True)
    return WholeTree(k, ld["node_size"], ld["parent_local"], ld["edges"], meta)


# ------------------------------------------------------------------------------------------------
# ego subgraphs and their trees


def cap_out_edges(src: np.ndarray, dst: np.ndarray, n: int, top_n: int, rng: np.random.Generator):
    """``limit_neighbor_num``: every source with more than ``top_n`` out-entries (duplicates count)
    keeps a uniform sample of ``top_n`` of them, without replacement; then self-loops are removed."""
    order = np.argsort(src, kind="stable")
    s, t = src[order], dst[order]
    ptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(s, minlength=n), out=ptr[1:])
    deg = np.diff(ptr)
    keep = np.ones(len(s), bool)
    for u in np.flatnonzero(deg > top_n):                 # ascending source id: deterministic draws
        a, b = ptr[u], ptr[u + 1]
        sel = rng.choice(b - a, top_n, replace=False)
        m = np.zeros(b - a, bool)
        m[sel] = True
        keep[a:b] = m
    s, t = s[keep], t[keep]
    nl = s != t
    return s[nl], t[nl]


def _csr(rows, cols, n):
    order = np.lexsort((cols, rows))
    ptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(rows, minlength=n), out=ptr[1:])
    return ptr, np.asarray(cols[order], np.int64)


@nb.njit(cache=True)
def _ego(lab, in_ptr, in_ind, out_ptr, out_ind, mark):
    """For each labelled node i: subset = sorted({i} u in-neighbours(i)); induced undirected simple
    edges (local ids, u < v). ``mark`` int64 [N] filled with -1. Lone nodes get an empty subset."""
    nl = len(lab)
    node_ptr = np.zeros(nl + 1, np.int64)
    edge_ptr = np.zeros(nl + 1, np.int64)
    tot_n = 0
    for q in range(nl):
        i = lab[q]
        c = 0
        for e in range(in_ptr[i], in_ptr[i + 1]):
            if in_ind[e] != i:
                c += 1
        if c > 0:
            tot_n += c + 1
    nodes = np.empty(tot_n, np.int64)
    eu = np.empty(16, np.int64)
    ev = np.empty(16, np.int64)
    ne = 0
    pn = 0
    for q in range(nl):
        i = lab[q]
        node_ptr[q] = pn
        edge_ptr[q] = ne
        c = 0
        for e in range(in_ptr[i], in_ptr[i + 1]):
            if in_ind[e] != i:
                c += 1
        if c == 0:
            continue
        s0 = pn
        nodes[pn] = i
        pn += 1
        for e in range(in_ptr[i], in_ptr[i + 1]):
            if in_ind[e] != i:
                nodes[pn] = in_ind[e]
                pn += 1
        sub = np.sort(nodes[s0:pn])
        # in_ind rows are unique and sorted, so sub has no duplicates
        nodes[s0:pn] = sub
        for a in range(len(sub)):
            mark[sub[a]] = a
        for a in range(len(sub)):
            u = sub[a]
            for e in range(out_ptr[u], out_ptr[u + 1]):
                b = mark[out_ind[e]]
                if b >= 0 and b != a:
                    lo, hi = (a, b) if a < b else (b, a)
                    if ne == len(eu):
                        eu2 = np.empty(2 * ne, np.int64)
                        ev2 = np.empty(2 * ne, np.int64)
                        eu2[:ne] = eu[:ne]
                        ev2[:ne] = ev[:ne]
                        eu, ev = eu2, ev2
                    eu[ne] = lo
                    ev[ne] = hi
                    ne += 1
        for a in range(len(sub)):
            mark[sub[a]] = -1
        # dedupe this subgraph's edges
        m = ne - edge_ptr[q]
        if m > 0:
            key = eu[edge_ptr[q]:ne] * len(sub) + ev[edge_ptr[q]:ne]
            key = np.unique(key)
            ne = edge_ptr[q]
            for t in range(len(key)):
                eu[ne] = key[t] // len(sub)
                ev[ne] = key[t] % len(sub)
                ne += 1
    node_ptr[nl] = pn
    edge_ptr[nl] = ne
    return node_ptr, nodes[:pn].copy(), edge_ptr, eu[:ne].copy(), ev[:ne].copy()


@dataclass
class SubgraphTrees:
    """Concatenated ego-graph trees, one ego graph per labelled node (in ``lab`` order).

    Level d (0..k): ``size[d]`` nodes in total; ``batch[d]`` [size[d]] = ego-graph index of each node;
    ``parent[d]`` (d >= 1) [size[d-1]] = global level-d index of the parent of each level-(d-1)
    node; ``edges[d]`` (d < k) [2, E_d] undirected quotient edges (both directions), global level-d
    indices; ``nodelist`` [size[0]] = dataset row of each leaf."""
    k: int
    nodelist: np.ndarray
    batch: list
    parent: list
    edges: list
    meta: dict

    @property
    def n_graphs(self) -> int:
        return int(self.meta["n_graphs"])


def _local_tree(n: int, eu: np.ndarray, ev: np.ndarray, k: int):
    rows = np.concatenate([eu, ev])
    cols = np.concatenate([ev, eu])
    ptr, ind = _csr(rows, cols, n)
    res = CodingTreeBuilder(ptr, ind, smooth=False).build(k)
    ld = res.layer_data(ptr, ind)
    return ld


def build_subgraph_trees(ds: str, edge_index: np.ndarray, lab: np.ndarray, n: int, k: int, *, seed: int = 0,
                         top_n: int = 10, mode: str = "faithful", in_cap: int = 0, rebuild: bool = False,
                         verbose: bool = True) -> SubgraphTrees:
    """Spec 3.4. ``mode="faithful"``: in-neighbour ego graphs after capping out-edges at ``top_n``,
    lone nodes get a star of ``top_n`` random labelled nodes. ``mode="undirected"`` (ablation):
    undirected 1-hop neighbourhood capped at ``top_n`` random neighbours, lone nodes stay alone
    (a single-node graph with unary chains). ``in_cap`` > 0 caps the in-neighbours of each ego at
    ``in_cap`` random ones (TwiBot-22 deviation)."""
    src = np.asarray(edge_index[0], np.int64)
    dst = np.asarray(edge_index[1], np.int64)
    lab = np.asarray(lab, np.int64)
    key = _digest(src, dst, lab) + f"_n{n}"
    path = method_dir(ds) / f"subgraphs_hop1_top{top_n}_{mode}_cap{in_cap}_seed{seed}_k{k}.npz"
    if path.exists() and not rebuild:
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if meta.get("input_key") == key and meta.get("version") == SUB_VERSION:
            return SubgraphTrees(k, z["nodelist"], [z[f"batch_{d}"] for d in range(k + 1)],
                                 [None] + [z[f"parent_{d}"] for d in range(1, k + 1)],
                                 [z[f"edges_{d}"] for d in range(k)], meta)
    t0 = time.time()
    rng = np.random.default_rng(seed)
    if mode == "faithful":
        s, t = cap_out_edges(src, dst, n, top_n, rng)
    elif mode == "undirected":
        a = np.concatenate([src, dst])
        b = np.concatenate([dst, src])
        nl = a != b
        a, b = a[nl], b[nl]
        u = np.unique(a * n + b)
        s, t = u // n, u % n
    else:
        raise ValueError(mode)
    # in-neighbour lists (unique, sorted) and out lists of the capped graph
    u = np.unique(t * n + s)
    in_ptr, in_ind = _csr(u // n, u % n, n)
    if mode == "undirected":
        in_cap = in_cap or top_n
    if in_cap > 0:
        deg = np.diff(in_ptr)
        keep = np.ones(len(in_ind), bool)
        for v in np.flatnonzero(deg > in_cap):
            a_, b_ = in_ptr[v], in_ptr[v + 1]
            m = np.zeros(b_ - a_, bool)
            m[rng.choice(b_ - a_, in_cap, replace=False)] = True
            keep[a_:b_] = m
        rows = np.repeat(np.arange(n), np.diff(in_ptr))[keep]
        in_ptr, in_ind = _csr(rows, in_ind[keep], n)
    out_ptr, out_ind = _csr(s, t, n)
    mark = np.full(n, -1, np.int64)
    node_ptr, nodes, edge_ptr, eu, ev = _ego(lab, in_ptr, in_ind, out_ptr, out_ind, mark)
    lone = np.flatnonzero(np.diff(node_ptr) == 0)
    # lone nodes: star of top_n distinct random labelled nodes (faithful) or a single node
    nodelists, edgelists = [], []
    nl = len(lab)
    for q in range(nl):
        a_, b_ = node_ptr[q], node_ptr[q + 1]
        nodelists.append(nodes[a_:b_])
        edgelists.append((eu[edge_ptr[q]:edge_ptr[q + 1]], ev[edge_ptr[q]:edge_ptr[q + 1]]))
    n_star = min(top_n, nl - 1)          # upstream loops forever with fewer than top_n + 1 labelled users
    star_e = (np.zeros(n_star, np.int64), np.arange(1, n_star + 1, dtype=np.int64))
    for q in lone:
        i = lab[q]
        if mode == "faithful" and n_star > 0:
            chosen = [i]
            while len(chosen) < n_star + 1:
                r = lab[int(rng.integers(0, nl))]
                if r not in chosen:
                    chosen.append(r)
            nodelists[q] = np.asarray(chosen, np.int64)
            edgelists[q] = star_e
        else:
            nodelists[q] = np.asarray([i], np.int64)
            edgelists[q] = (np.zeros(0, np.int64), np.zeros(0, np.int64))
    t_ego = time.time() - t0
    # trees, cached per distinct local graph
    cache: dict = {}
    per = []
    t1 = time.time()
    for q in range(nl):
        nn_ = len(nodelists[q])
        e_u, e_v = edgelists[q]
        kk = (nn_, e_u.tobytes(), e_v.tobytes())
        ld = cache.get(kk)
        if ld is None:
            if nn_ == 1:
                ld = {"node_size": [1] * (k + 1), "parent_local": [None] + [np.zeros(1, np.int64)] * k,
                      "edges": [np.zeros((2, 0), np.int64)] * k}
            else:
                ld = _local_tree(nn_, e_u, e_v, k)
            cache[kk] = ld
        per.append(ld)
        if verbose and (q + 1) % 20000 == 0:
            print(f"[sebot] subgraph trees {q + 1}/{nl} ({time.time() - t1:.0f}s, {len(cache)} distinct)", flush=True)
    # concatenate per level
    sizes = np.array([[ld["node_size"][d] for d in range(k + 1)] for ld in per], np.int64)   # [G, k+1]
    offs = np.zeros((nl + 1, k + 1), np.int64)
    np.cumsum(sizes, axis=0, out=offs[1:])
    batch = [np.repeat(np.arange(nl, dtype=np.int64), sizes[:, d]) for d in range(k + 1)]
    parent = [None]
    for d in range(1, k + 1):
        parent.append(np.concatenate([ld["parent_local"][d] + offs[q, d] for q, ld in enumerate(per)]).astype(np.int64))
    edges = []
    for d in range(k):
        es = [ld["edges"][d] + offs[q, d] for q, ld in enumerate(per)]
        e = np.concatenate(es, axis=1) if es else np.zeros((2, 0), np.int64)
        edges.append(np.concatenate([e, e[::-1]], axis=1).astype(np.int64))
    nodelist = np.concatenate(nodelists).astype(np.int64)
    meta = {"version": SUB_VERSION, "input_key": key, "dataset": ds, "k": k, "mode": mode, "top_n": top_n,
            "in_cap": in_cap, "seed": seed, "n_graphs": nl, "lone": int(len(lone)),
            "lone_fraction": round(len(lone) / max(nl, 1), 4), "distinct_graphs": len(cache),
            "nodes_level": [int(x) for x in sizes.sum(0)], "max_graph_nodes": int(sizes[:, 0].max()),
            "edges_level": [int(e.shape[1]) for e in edges], "seconds_ego": round(t_ego, 2),
            "seconds_trees": round(time.time() - t1, 2)}
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs = {"nodelist": nodelist, "meta": np.asarray(json.dumps(meta))}
    for d in range(k + 1):
        arrs[f"batch_{d}"] = batch[d]
    for d in range(1, k + 1):
        arrs[f"parent_{d}"] = parent[d]
    for d in range(k):
        arrs[f"edges_{d}"] = edges[d]
    tmp = path.with_name(path.name + f".{os.getpid()}.partial.npz")
    np.savez(tmp, **arrs)
    tmp.replace(path)
    if verbose:
        print(f"[sebot] {ds} subgraphs: {meta}", flush=True)
    return SubgraphTrees(k, nodelist, batch, parent, edges, meta)
