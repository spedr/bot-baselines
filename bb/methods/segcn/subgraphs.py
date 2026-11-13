"""Sparse random-walk subgraph store for SEGCN (spec 3.4 and 7.3 step C).

For every root u: one (or ``repeat``) node2vec walk(s) of ``walk_length`` steps starting at u on the
simple undirected follow graph; the subgraph is the set of visited nodes (root first, then the other
nodes in ascending id order) with **all** induced graph edges between them (GNN-AK
``extract_subgraphs``: ``edge_mask = node_mask[:, src] & node_mask[:, dst]``), plus the BFS distance
of every subgraph node from the root inside the subgraph (GNN-AK ``hop_indicator``; used only with
``hop_dim > 0``).

Storage (CSR-style, cached under ``data/methods/segcn/<dataset>/<key>/``)::

    node_ptr.i64.npy  [N+1]       subgraph of u = node_ids[node_ptr[u]:node_ptr[u+1]] (root first)
    node_ids.i32.npy  [T]         global node ids
    hop.u8.npy        [T]         BFS distance from the root inside the subgraph (root 0)
    edge_ptr.i64.npy  [N+1]       induced edges of u = edge_pairs[edge_ptr[u]:edge_ptr[u+1]]
    edge_pairs.u8.npy [E_sub, 2]  LOCAL positions (i < j) inside u's node list, each undirected edge once
    meta.json                     parameters, graph digest, sizes, s_bar, build time / peak RSS

The released GNN-AK/SEGCN extractor builds dense [N, N] node masks and an [N, E] edge mask (about
470 GB on TwiBot-20, spec 7.1). Here each root costs O(walk + |V_u|^2 log d): memory is bounded by
the chunk buffers (``chunk_roots`` x pairs) plus the CSR, independent of N.

Random numbers are counter-based (splitmix64 of (seed, root, repeat)), so the store is a pure
function of (graph, walk_length, p, q, repeat, seed) and identical for any thread count.
Walk semantics follow torch_cluster ``random_walk`` where it matters for the default: an isolated
node stays in place and with p = q = 1 the walk is uniform. Otherwise this is EXACT node2vec rejection
sampling with the second-order bias (1/p return, 1 BFS-like, 1/q DFS-like; checked against the exact
two-step law in the tests). torch_cluster's CPU rejection loop chains ``else if`` tests, so it
accepts a return with probability max(1/p, 1/q)/M and a BFS move with max(1, 1/q)/M; that quirk is
not reproduced (it only affects p != q ablations, review round 1).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numba as nb
import numpy as np

from ...io import NpyAppender, file_digest, load_npy, read_json, save_npy, write_json
from ...paths import dataset_paths

STORE_VERSION = 1
MAX_NODES = 255          # local positions are uint8


# ------------------------------------------------------------------------------------------------
# counter-based RNG

@nb.njit(cache=True, inline="always")
def _splitmix(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = x
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return x, z ^ (z >> np.uint64(31))


@nb.njit(cache=True, inline="always")
def _uniform(state):
    state, z = _splitmix(state)
    return state, (z >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@nb.njit(cache=True, inline="always")
def _randint(state, n):
    state, u = _uniform(state)
    k = int(u * n)
    return state, (k if k < n else n - 1)


@nb.njit(cache=True, inline="always")
def _has_edge(indptr, indices, a, b):
    """Binary search for b in the (sorted) CSR row of a."""
    lo = indptr[a]
    hi = indptr[a + 1]
    while lo < hi:
        mid = (lo + hi) >> 1
        v = indices[mid]
        if v < b:
            lo = mid + 1
        elif v > b:
            hi = mid
        else:
            return True
    return False


@nb.njit(cache=True)
def _walk(indptr, indices, root, length, p, q, seed, rep, out):
    """One walk of ``length`` steps from ``root`` into out[0:length+1] (torch_cluster semantics)."""
    state = np.uint64(seed) * np.uint64(0x100000001B3) ^ (np.uint64(root) * np.uint64(0x9E3779B1)) \
        ^ (np.uint64(rep + 1) << np.uint64(48))
    state, _ = _splitmix(state)
    out[0] = root
    uniform = p == 1.0 and q == 1.0
    mx = max(1.0 / p, 1.0, 1.0 / q)
    prob_ret, prob_bfs, prob_dfs = (1.0 / p) / mx, 1.0 / mx, (1.0 / q) / mx
    prev = -1
    cur = root
    for s in range(1, length + 1):
        a = indptr[cur]
        d = indptr[cur + 1] - a
        if d == 0:
            out[s] = cur
            prev = cur
            continue
        if uniform or prev < 0:
            state, k = _randint(state, d)
            nxt = indices[a + k]
        else:
            while True:
                state, k = _randint(state, d)
                nxt = indices[a + k]
                state, r = _uniform(state)
                if nxt == prev:
                    acc = prob_ret
                elif _has_edge(indptr, indices, prev, nxt):
                    acc = prob_bfs
                else:
                    acc = prob_dfs
                if r < acc:
                    break
        out[s] = nxt
        prev = cur
        cur = nxt
    return out


@nb.njit(cache=True)
def random_walks(indptr, indices, roots, length, p, q, seed, repeat):
    """[len(roots), repeat * (length + 1)] int64 walks (for tests and the dense reference)."""
    w = length + 1
    out = np.empty((roots.shape[0], repeat * w), np.int64)
    buf = np.empty(w, np.int64)
    for i in range(roots.shape[0]):
        for r in range(repeat):
            _walk(indptr, indices, roots[i], length, p, q, seed, r, buf)
            out[i, r * w:(r + 1) * w] = buf
    return out


@nb.njit(cache=True, parallel=True)
def _chunk(indptr, indices, roots, length, p, q, seed, repeat, max_nodes,
           nodes, n_nodes, pairs, n_pairs, hops):
    """Per root of the chunk: node list (root first, rest ascending), induced pairs, BFS hops."""
    w = length + 1
    for i in nb.prange(roots.shape[0]):
        u = roots[i]
        walk = np.empty(repeat * w, np.int64)
        buf = np.empty(w, np.int64)
        for r in range(repeat):
            _walk(indptr, indices, u, length, p, q, seed, r, buf)
            walk[r * w:(r + 1) * w] = buf
        s = np.sort(walk)
        k = 1
        nodes[i, 0] = u
        last = -1
        for t in range(s.shape[0]):
            v = s[t]
            if v != last and v != u:
                nodes[i, k] = v
                k += 1
            last = v
        n_nodes[i] = k
        # induced edges: test every pair; search the row of the lower-degree endpoint
        m = 0
        for a in range(k):
            va = nodes[i, a]
            da = indptr[va + 1] - indptr[va]
            for b in range(a + 1, k):
                vb = nodes[i, b]
                db = indptr[vb + 1] - indptr[vb]
                hit = _has_edge(indptr, indices, va, vb) if da <= db else _has_edge(indptr, indices, vb, va)
                if hit:
                    pairs[i, m, 0] = a
                    pairs[i, m, 1] = b
                    m += 1
        n_pairs[i] = m
        # BFS distances from the root (position 0) over the induced edges
        for a in range(k):
            hops[i, a] = 255
        hops[i, 0] = 0
        changed = True
        while changed:
            changed = False
            for e in range(m):
                a = pairs[i, e, 0]
                b = pairs[i, e, 1]
                if hops[i, a] != 255 and hops[i, b] > hops[i, a] + 1:
                    hops[i, b] = hops[i, a] + 1
                    changed = True
                if hops[i, b] != 255 and hops[i, a] > hops[i, b] + 1:
                    hops[i, a] = hops[i, b] + 1
                    changed = True


# ------------------------------------------------------------------------------------------------
# batch planner kernels (review round 2): per-root normalised block CSR precomputed once, then every
# batch is assembled on the CPU (no per-batch argsort / degree / unique kernels on the GPU)

@nb.njit(cache=True)
def block_entries(node_ptr, edge_ptr, pairs, ent_ptr, ent_col, ent_val, rnnz):
    """Per root u: the rows of D^-1/2 (A_u + I) D^-1/2 over u's local node positions, in CSR order
    (row-major, columns ascending, self-loop included). ``ent_ptr`` [N+1] is given
    (= cumsum(k_u + 2 m_u)); fills ``ent_col`` uint8 (local column), ``ent_val`` float32 and ``rnnz``
    uint8 [T] (entries of every lifted row). Degrees are those of the induced subgraph."""
    n = node_ptr.shape[0] - 1
    deg = np.empty(MAX_NODES + 1, np.int64)
    fill = np.empty(MAX_NODES + 1, np.int64)
    dinv = np.empty(MAX_NODES + 1, np.float32)
    for u in range(n):
        a = node_ptr[u]
        k = node_ptr[u + 1] - a
        e0 = edge_ptr[u]
        m = edge_ptr[u + 1] - e0
        for i in range(k):
            deg[i] = 1
        for j in range(m):
            deg[pairs[e0 + j, 0]] += 1
            deg[pairs[e0 + j, 1]] += 1
        s = ent_ptr[u]
        for i in range(k):
            fill[i] = s
            rnnz[a + i] = deg[i]
            dinv[i] = np.float32(1.0) / np.sqrt(np.float32(deg[i]))
            s += deg[i]
        for i in range(k):
            ent_col[fill[i]] = i
            fill[i] += 1
        for j in range(m):
            x = pairs[e0 + j, 0]
            y = pairs[e0 + j, 1]
            ent_col[fill[x]] = y
            fill[x] += 1
            ent_col[fill[y]] = x
            fill[y] += 1
        r0 = ent_ptr[u]
        for i in range(k):
            r1 = r0 + deg[i]
            for x in range(r0 + 1, r1):                   # insertion sort of the (<= 255) columns
                c = ent_col[x]
                y = x - 1
                while y >= r0 and ent_col[y] > c:
                    ent_col[y + 1] = ent_col[y]
                    y -= 1
                ent_col[y + 1] = c
            for x in range(r0, r1):
                ent_val[x] = dinv[i] * dinv[ent_col[x]]
            r0 = r1


@nb.njit(cache=True, nogil=True)
def lift_kernel(roots, node_ptr, node_ids, hop, ent_ptr, ent_col, ent_val, rnnz,
                ids, seg, hops, crow, col, val, offs, cnt):
    """Block-diagonal batch of the subgraphs of ``roots`` (output arrays pre-sized): lifted rows
    (global ids, subgraph index, hop), CSR (crow/col/val over the batch rows), root row and size of
    every subgraph."""
    t = 0
    z = 0
    crow[0] = 0
    for i in range(roots.shape[0]):
        u = roots[i]
        a = node_ptr[u]
        k = node_ptr[u + 1] - a
        offs[i] = t
        cnt[i] = k
        for j in range(k):
            ids[t + j] = node_ids[a + j]
            seg[t + j] = i
            hops[t + j] = hop[a + j]
            crow[t + j + 1] = crow[t + j] + rnnz[a + j]
        e = ent_ptr[u]
        m = ent_ptr[u + 1] - e
        for j in range(m):
            col[z + j] = ent_col[e + j] + t
            val[z + j] = ent_val[e + j]
        t += k
        z += m


@nb.njit(cache=True, nogil=True)
def unique_kernel(ids, roots, mark, stamp, pos, uniq, inv, self_pos):
    """Sorted unique of ``ids`` into ``uniq`` (returns its length), ``inv`` = position of every id in
    it, ``self_pos`` = position of every root. ``mark``/``pos`` are [N] int32 scratch arrays
    (``mark`` holds stamps < ``stamp``)."""
    nu = 0
    for t in range(ids.shape[0]):
        v = ids[t]
        if mark[v] != stamp:
            mark[v] = stamp
            uniq[nu] = v
            nu += 1
    n = mark.shape[0]
    if nu * 32 > n:                                       # dense: scan the marks (already sorted)
        j = 0
        for v in range(n):
            if mark[v] == stamp:
                uniq[j] = v
                j += 1
    else:
        uniq[:nu] = np.sort(uniq[:nu])
    for j in range(nu):
        pos[uniq[j]] = j
    for t in range(ids.shape[0]):
        inv[t] = pos[ids[t]]
    for i in range(roots.shape[0]):
        self_pos[i] = pos[roots[i]]
    return nu


PLAN_INT_FIELDS = ("roots", "ids", "seg", "root_pos", "hop", "crow", "col", "inv", "self_pos")
PLAN_FLOAT_FIELDS = ("val", "cnt")


@nb.njit(cache=True, nogil=True)
def plan_kernel(roots, n_layers, dedup, node_ptr, node_ids, hop, ent_ptr, ent_col, ent_val, rnnz,
                mark, pos, stamp):
    """Whole batch plan in one GIL-free call: levels S_L = roots, S_{l-1} = sorted unique ids of the
    lifted rows of S_l (``dedup``; otherwise one level, no dedup), packed into one int32 and one
    float32 buffer. Returns (ints, floats, spans [L, 11, 2] (start, length) per field of
    PLAN_INT_FIELDS + PLAN_FLOAT_FIELDS, levels bottom-up), S_0 length, rows per level (bottom-up),
    new stamp)."""
    lv_int = []
    lv_flt = []
    nrows = np.zeros(n_layers, np.int64)
    cur = roots
    for l in range(n_layers):
        S = cur.shape[0]
        T = 0
        Z = 0
        for i in range(S):
            u = cur[i]
            T += node_ptr[u + 1] - node_ptr[u]
            Z += ent_ptr[u + 1] - ent_ptr[u]
        ids = np.empty(T, np.int64)
        seg = np.empty(T, np.int64)
        hops = np.empty(T, np.int64)
        crow = np.empty(T + 1, np.int64)
        col = np.empty(Z, np.int64)
        val = np.empty(Z, np.float32)
        offs = np.empty(S, np.int64)
        cnt = np.empty(S, np.float32)
        lift_kernel(cur, node_ptr, node_ids, hop, ent_ptr, ent_col, ent_val, rnnz, ids, seg, hops, crow, col,
                    val, offs, cnt)
        if dedup:
            stamp += 1
            uniq = np.empty(T, np.int64)
            inv = np.empty(T, np.int64)
            sp = np.empty(S, np.int64)
            nu = unique_kernel(ids, cur, mark, stamp, pos, uniq, inv, sp)
            nxt = uniq[:nu].copy()
        else:
            inv = np.empty(0, np.int64)
            sp = np.empty(0, np.int64)
            nxt = cur
        lv_int.append(cur)
        lv_int.append(ids)
        lv_int.append(seg)
        lv_int.append(offs)
        lv_int.append(hops)
        lv_int.append(crow)
        lv_int.append(col)
        lv_int.append(inv)
        lv_int.append(sp)
        lv_flt.append(val)
        lv_flt.append(cnt)
        nrows[l] = T
        cur = nxt
        if not dedup:
            break
    nl = len(lv_flt) // 2
    ti = cur.shape[0]
    for a in lv_int:
        ti += a.shape[0]
    tf = 0
    for a in lv_flt:
        tf += a.shape[0]
    ib = np.empty(ti, np.int32)
    fb = np.empty(tf, np.float32)
    spans = np.zeros((nl, 11, 2), np.int64)
    for j in range(cur.shape[0]):
        ib[j] = cur[j]
    o = cur.shape[0]
    p = 0
    for j in range(nl):
        l = nl - 1 - j
        for f in range(9):
            a = lv_int[l * 9 + f]
            spans[j, f, 0] = o
            spans[j, f, 1] = a.shape[0]
            for x in range(a.shape[0]):
                ib[o + x] = a[x]
            o += a.shape[0]
        for f in range(2):
            a = lv_flt[l * 2 + f]
            spans[j, 9 + f, 0] = p
            spans[j, 9 + f, 1] = a.shape[0]
            for x in range(a.shape[0]):
                fb[p + x] = a[x]
            p += a.shape[0]
    rows = np.empty(nl, np.int64)
    for j in range(nl):
        rows[j] = nrows[nl - 1 - j]
    return ib, fb, spans, cur.shape[0], rows, stamp


@nb.njit(cache=True, nogil=True)
def gather_ids(roots, node_ptr, node_ids):
    """Concatenated node lists of ``roots`` (int64)."""
    tot = 0
    for i in range(roots.shape[0]):
        tot += node_ptr[roots[i] + 1] - node_ptr[roots[i]]
    out = np.empty(tot, np.int64)
    t = 0
    for i in range(roots.shape[0]):
        for j in range(node_ptr[roots[i]], node_ptr[roots[i] + 1]):
            out[t] = node_ids[j]
            t += 1
    return out


@nb.njit(cache=True)
def level_cost(node_ptr, node_ids, c):
    """out[u] = sum_{v in V_u} c[v]."""
    n = node_ptr.shape[0] - 1
    out = np.zeros(n, np.int64)
    for u in range(n):
        s = 0
        for j in range(node_ptr[u], node_ptr[u + 1]):
            s += c[node_ids[j]]
        out[u] = s
    return out


@dataclass
class StoreArrays:
    """In-memory (or memory-mapped) store: the arrays of the module docstring."""
    node_ptr: np.ndarray
    node_ids: np.ndarray
    hop: np.ndarray
    edge_ptr: np.ndarray
    edge_pairs: np.ndarray
    meta: dict

    @property
    def n(self) -> int:
        return len(self.node_ptr) - 1

    def nodes(self, u: int) -> np.ndarray:
        return np.asarray(self.node_ids[self.node_ptr[u]:self.node_ptr[u + 1]])

    def pairs(self, u: int) -> np.ndarray:
        return np.asarray(self.edge_pairs[self.edge_ptr[u]:self.edge_ptr[u + 1]])


def _sizes(indptr, walk_length, repeat):
    max_nodes = min(repeat * walk_length + 1, int(len(indptr) - 1))
    if max_nodes > MAX_NODES:
        raise ValueError(f"repeat * walk_length + 1 = {repeat * walk_length + 1} > {MAX_NODES} (uint8 positions)")
    return max(max_nodes, 1), max(max_nodes * (max_nodes - 1) // 2, 1)


def iter_chunks(indptr, indices, *, walk_length=30, p=1.0, q=1.0, repeat=1, seed=0,
                chunk_roots: int | None = None, buffer_bytes: int = 64 << 20):
    """Yield (start, n_nodes, nodes_flat, hops_flat, n_pairs, pairs_flat) per chunk of roots."""
    indptr = np.asarray(indptr, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int32)
    n = len(indptr) - 1
    max_nodes, max_pairs = _sizes(indptr, walk_length, repeat)
    if chunk_roots is None:
        chunk_roots = int(max(256, min(200_000, buffer_bytes // (2 * max_pairs + 9 * max_nodes))))
    nodes = np.empty((chunk_roots, max_nodes), np.int64)
    pairs = np.empty((chunk_roots, max_pairs, 2), np.uint8)
    hops = np.empty((chunk_roots, max_nodes), np.uint8)
    nn_ = np.empty(chunk_roots, np.int64)
    npr = np.empty(chunk_roots, np.int64)
    for s in range(0, n, chunk_roots):
        roots = np.arange(s, min(n, s + chunk_roots), dtype=np.int64)
        c = len(roots)
        _chunk(indptr, indices, roots, walk_length, float(p), float(q), int(seed), int(repeat), max_nodes,
               nodes, nn_, pairs, npr, hops)
        k, m = nn_[:c], npr[:c]
        mask_n = np.arange(max_nodes)[None, :] < k[:, None]
        mask_e = np.arange(max_pairs)[None, :] < m[:, None]
        yield s, k.copy(), nodes[:c][mask_n].astype(np.int32), hops[:c][mask_n], m.copy(), pairs[:c][mask_e]


def build_in_memory(indptr, indices, *, walk_length=30, p=1.0, q=1.0, repeat=1, seed=0, **kw) -> StoreArrays:
    """Whole store in RAM (tests, small graphs, per-epoch resampling)."""
    ks, ids, hs, ms, prs = [], [], [], [], []
    for _, k, nid, h, m, pr in iter_chunks(indptr, indices, walk_length=walk_length, p=p, q=q, repeat=repeat,
                                           seed=seed, **kw):
        ks.append(k); ids.append(nid); hs.append(h); ms.append(m); prs.append(pr)  # noqa: E702
    n = len(indptr) - 1
    node_ptr = np.zeros(n + 1, np.int64)
    edge_ptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.concatenate(ks) if ks else np.zeros(0, np.int64), out=node_ptr[1:])
    np.cumsum(np.concatenate(ms) if ms else np.zeros(0, np.int64), out=edge_ptr[1:])
    cat = lambda xs, dt, sh: np.concatenate(xs).astype(dt) if xs else np.zeros(sh, dt)  # noqa: E731
    meta = {"walk_length": walk_length, "p": p, "q": q, "repeat": repeat, "seed": seed}
    st = StoreArrays(node_ptr, cat(ids, np.int32, (0,)), cat(hs, np.uint8, (0,)), edge_ptr,
                     cat(prs, np.uint8, (0, 2)).reshape(-1, 2), meta)
    st.meta.update(stats(st))
    return st


def stats(st: StoreArrays) -> dict:
    k = np.diff(st.node_ptr)
    m = np.diff(st.edge_ptr)
    q = lambda a, x: float(np.quantile(a, x)) if len(a) else 0.0  # noqa: E731
    return {"n_roots": int(st.n), "total_nodes": int(st.node_ptr[-1]), "total_pairs": int(st.edge_ptr[-1]),
            "s_bar": float(k.mean()) if len(k) else 0.0, "s_p50": q(k, .5), "s_p99": q(k, .99),
            "s_max": int(k.max()) if len(k) else 0, "pairs_mean": float(m.mean()) if len(m) else 0.0,
            "pairs_p99": q(m, .99), "singletons": int((k == 1).sum()),
            "mean_lifted_rows_2layer_per_root": float(_two_layer_rows(st).mean()) if len(k) else 0.0}


def _two_layer_rows(st: StoreArrays) -> np.ndarray:
    """Per root r: sum_{v in V_r} |V_v| (layer-1 lifted rows without dedup) + |V_r|."""
    k = np.diff(st.node_ptr).astype(np.int64)
    contrib = k[np.asarray(st.node_ids, dtype=np.int64)]
    seg = np.repeat(np.arange(st.n), k)
    return np.bincount(seg, weights=contrib, minlength=st.n) + k


# ------------------------------------------------------------------------------------------------
# on-disk cache

def store_key(walk_length, p, q, repeat, seed) -> str:
    f = lambda x: f"{float(x):g}"  # noqa: E731
    return f"rw_L{walk_length}_p{f(p)}_q{f(q)}_r{repeat}_s{seed}_v{STORE_VERSION}"


def segcn_graph(ds: str, rels: list[str]):
    """Simple undirected union of the follow relations (shared cached view, either order)."""
    from ...data import load_undirected
    g = dataset_paths(ds).graph
    for r in (rels, rels[::-1]):
        if (g / ("undirected__" + "+".join(r) + ".csr")).exists():
            return load_undirected(ds, r)
    return load_undirected(ds, rels)


def _graph_digest(csr) -> str:
    if csr.path is not None:
        return file_digest(Path(csr.path) / "indices.npy") + ":" + file_digest(Path(csr.path) / "indptr.npy")
    return "mem"


def load_or_build_store(ds: str, rels: list[str], *, walk_length=30, p=1.0, q=1.0, repeat=1, seed=0,
                        verbose=True) -> StoreArrays:
    """Memory-mapped store for (dataset, walk parameters, seed); built and cached on first use."""
    csr = segcn_graph(ds, rels)
    d = dataset_paths(ds).method_dir("segcn") / store_key(walk_length, p, q, repeat, seed)
    gd = _graph_digest(csr)
    if (d / "meta.json").exists():
        meta = read_json(d / "meta.json")
        if meta.get("graph_digest") == gd and meta.get("version") == STORE_VERSION:
            return StoreArrays(load_npy(d / "node_ptr.i64.npy"), load_npy(d / "node_ids.i32.npy"),
                               load_npy(d / "hop.u8.npy"), load_npy(d / "edge_ptr.i64.npy"),
                               load_npy(d / "edge_pairs.u8.npy"), meta)
    from ...mem import peak_rss_bytes
    t0 = time.time()
    d.mkdir(parents=True, exist_ok=True)
    indptr = np.asarray(csr.indptr, dtype=np.int64)
    indices = np.asarray(csr.indices, dtype=np.int32)
    n = len(indptr) - 1
    w_ids = NpyAppender(d / "node_ids.i32.npy", np.int32)
    w_hop = NpyAppender(d / "hop.u8.npy", np.uint8)
    w_pr = NpyAppender(d / "edge_pairs.u8.npy", np.uint8, (2,))
    node_ptr = np.zeros(n + 1, np.int64)
    edge_ptr = np.zeros(n + 1, np.int64)
    try:
        for s, k, nid, h, m, pr in iter_chunks(indptr, indices, walk_length=walk_length, p=p, q=q,
                                               repeat=repeat, seed=seed):
            node_ptr[s + 1:s + 1 + len(k)] = k
            edge_ptr[s + 1:s + 1 + len(m)] = m
            w_ids.append(nid)
            w_hop.append(h)
            w_pr.append(pr)
            if verbose and s and (s // len(k)) % 5 == 0:
                print(f"  segcn store {ds}: {s + len(k):,}/{n:,} roots ({time.time() - t0:.1f} s)", flush=True)
    except BaseException:
        for w in (w_ids, w_hop, w_pr):
            w.abort()
        raise
    w_ids.close(); w_hop.close(); w_pr.close()  # noqa: E702
    np.cumsum(node_ptr, out=node_ptr)
    np.cumsum(edge_ptr, out=edge_ptr)
    save_npy(d / "node_ptr.i64.npy", node_ptr)
    save_npy(d / "edge_ptr.i64.npy", edge_ptr)
    st = StoreArrays(node_ptr, load_npy(d / "node_ids.i32.npy"), load_npy(d / "hop.u8.npy"), edge_ptr,
                     load_npy(d / "edge_pairs.u8.npy"), {})
    meta = {"version": STORE_VERSION, "dataset": ds, "relations": list(rels), "graph_digest": gd,
            "graph_nnz": int(indptr[-1]), "walk_length": walk_length, "p": p, "q": q, "repeat": repeat,
            "seed": seed, **stats(st), "build_seconds": round(time.time() - t0, 2),
            "build_peak_rss_bytes": peak_rss_bytes()}
    write_json(d / "meta.json", meta)
    st.meta = meta
    if verbose:
        print(f"  segcn store {ds} {d.name}: s_bar={meta['s_bar']:.2f} pairs_mean={meta['pairs_mean']:.2f} "
              f"T={meta['total_nodes']:,} E_sub={meta['total_pairs']:,} in {meta['build_seconds']} s", flush=True)
    return st


# ------------------------------------------------------------------------------------------------
# naive dense reference (tests only; O(N^2) memory, exactly the GNN-AK extract_subgraphs recipe)

def dense_reference(indptr, indices, walks: np.ndarray):
    """GNN-AK-style dense extraction from given walks: returns (node_mask [N, N] bool,
    edge_mask [N, E_dir] bool, src, dst) over the directed edge list of the CSR."""
    n = len(indptr) - 1
    node_mask = np.zeros((n, n), bool)
    for u in range(n):
        node_mask[u, walks[u]] = True
    src = np.repeat(np.arange(n), np.diff(indptr))
    dst = np.asarray(indices, dtype=np.int64)
    edge_mask = node_mask[:, src] & node_mask[:, dst]
    return node_mask, edge_mask, src, dst
