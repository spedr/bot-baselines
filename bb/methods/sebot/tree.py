"""Sparse structural-entropy coding trees (SEBot / SEP ``PartitionTree``, mode ``v2``).

The official builder (``tree_ref.py``, ported from SEBot ``SEPN/codingTree.py``) needs a dense
N x N matrix and quadratic Python loops. This module runs the same algorithm over a CSR graph with
bounded memory and reproduces the reference tree exactly (same node ids, same parents) when the
reference iterates its sets in ascending order (``tree_ref.PartitionTree(ordered=True)``; see the
statement of changes there). ``tests/methods/test_sebot.py`` checks this on random graphs.

Algorithm (spec docs/methods/sebot.md 3.3), per call of ``__build_k_tree`` (numba ``_build_k_tree``):

- merge phase: lazy-deletion min-heap of (CombineDelta, id_lo, id_hi, cut) over adjacent active
  communities; ids are unique, so the tuple order is total and the pop sequence equals the
  reference's. A community keeps its neighbour list (target, weight) in a pool; targets may be
  stale and are resolved with a union-find over the merge forest, so
  cut(A u B, C) = cut(A, C) + cut(B, C) without leaf lists. Heap and pool are compacted when stale
  entries exceed twice the live ones.
- solitary root for disconnected inputs (vol = g_vol, g = 0), as the reference.
- compress phase: lazy heap of (CompressDelta, node) with version stamps (the reference rewrites
  entries in place and re-heapifies after every splice); heights are maintained with a per-node
  count of children at height h - 1, so a parent is rescanned only when its height really drops.

The v2 driver (``CodingTreeBuilder``) keeps the global tree in flat arrays (parent, vol, g) and runs
``leaf_up`` (every height-1 community with >= 3 leaves, one numba call over all communities) and
``root_down`` (the community graph of the root's children) exactly as the reference, including its
stale-result reuse ("flag" logic) and its global id counter.

``smooth=True`` is the whole-graph variant (vol + 1, log2); ``smooth=False`` the subgraph variant
(no + 1; natural log in CompressDelta and ``log(x)/log(2)`` in CombineDelta, as upstream).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numba as nb
import numpy as np

LN2 = math.log(2.0)


# ------------------------------------------------------------------------------------------------
# numba kernels


@nb.njit(cache=True, inline="always")
def _combine(v1, v2, g1, g2, cut, g_vol, smooth):
    if smooth:
        v1 = v1 + 1.0
        v2 = v2 + 1.0
        g1 = g1 + 1.0
        g2 = g2 + 1.0
        v12 = v1 + v2
        return ((v1 - g1) * math.log2(v12 / v1) + (v2 - g2) * math.log2(v12 / v2)
                - 2 * cut * math.log2(g_vol / v12)) / g_vol
    v12 = v1 + v2
    return ((v1 - g1) * (math.log(v12 / v1) / LN2) + (v2 - g2) * (math.log(v12 / v2) / LN2)
            - 2 * cut * (math.log(g_vol / v12) / LN2)) / g_vol


@nb.njit(cache=True, inline="always")
def _compress_delta(ccut, vn, vp, smooth):
    if smooth:
        return ccut * math.log2((vp + 1.0) / (vn + 1.0))
    return ccut * math.log(vp / vn)


@nb.njit(cache=True, inline="always")
def _lt4(k1, a1, b1, k2, a2, b2):
    if k1 < k2:
        return True
    if k1 > k2:
        return False
    if a1 < a2:
        return True
    if a1 > a2:
        return False
    return b1 < b2


@nb.njit(cache=True)
def _mh_up(K, A, B, C, pos):
    k, a, b, c = K[pos], A[pos], B[pos], C[pos]
    while pos > 0:
        par = (pos - 1) >> 1
        if _lt4(k, a, b, K[par], A[par], B[par]):
            K[pos], A[pos], B[pos], C[pos] = K[par], A[par], B[par], C[par]
            pos = par
        else:
            break
    K[pos], A[pos], B[pos], C[pos] = k, a, b, c


@nb.njit(cache=True)
def _mh_down(K, A, B, C, pos, n):
    k, a, b, c = K[pos], A[pos], B[pos], C[pos]
    while True:
        l = 2 * pos + 1
        if l >= n:
            break
        r = l + 1
        m = l
        if r < n and _lt4(K[r], A[r], B[r], K[l], A[l], B[l]):
            m = r
        if _lt4(K[m], A[m], B[m], k, a, b):
            K[pos], A[pos], B[pos], C[pos] = K[m], A[m], B[m], C[m]
            pos = m
        else:
            break
    K[pos], A[pos], B[pos], C[pos] = k, a, b, c


@nb.njit(cache=True)
def _find(uf, x):
    while uf[x] != x:
        uf[x] = uf[uf[x]]
        x = uf[x]
    return x


@nb.njit(cache=True)
def _depth(parent, ch, x):
    d = 0
    a = x
    while parent[a] != -1:
        a = parent[a]
        d += 1
    return d + ch[x]


@nb.njit(cache=True)
def _build_k_tree(m, vol0, g0, indptr, indices, w, g_vol, k, smooth):
    """One reference ``__build_k_tree`` call on m input nodes (local ids 0..m-1 in ascending order
    of their reference ids; all have child_h 0). Created nodes get local ids m, m+1, ... in creation
    order. Returns (n_nodes, parent, child_h, vol, g, alive, root, stats); ``alive`` is False for
    nodes removed by the compress phase. ``k <= 0`` skips the compress phase."""
    cap = 2 * m + 2
    parent = np.full(cap, -1, np.int64)
    ch = np.zeros(cap, np.int64)
    vol = np.zeros(cap, np.float64)
    g = np.zeros(cap, np.float64)
    ccut = np.zeros(cap, np.float64)
    merged = np.zeros(cap, np.bool_)
    alive = np.ones(cap, np.bool_)
    uf = np.arange(cap).astype(np.int64)
    for i in range(m):
        vol[i] = vol0[i]
        g[i] = g0[i]
    # neighbour pool
    nnz = indptr[m]
    pcap = max(nnz * 2, 16)
    pt = np.empty(pcap, np.int64)
    pw = np.empty(pcap, np.float64)
    pstart = np.zeros(cap, np.int64)
    plen = np.zeros(cap, np.int64)
    for i in range(m):
        pstart[i] = indptr[i]
        plen[i] = indptr[i + 1] - indptr[i]
    for e in range(nnz):
        pt[e] = indices[e]
        pw[e] = w[e]
    pn = nnz
    pool_live = nnz
    # merge heap
    hcap = max(nnz, 16)
    HK = np.empty(hcap, np.float64)
    HA = np.empty(hcap, np.int64)
    HB = np.empty(hcap, np.int64)
    HC = np.empty(hcap, np.float64)
    hn = 0
    for i in range(m):
        for e in range(indptr[i], indptr[i + 1]):
            j = indices[e]
            if j > i:
                HK[hn] = _combine(vol[i], vol[j], g[i], g[j], w[e], g_vol, smooth)
                HA[hn] = i
                HB[hn] = j
                HC[hn] = w[e]
                hn += 1
    for p in range((hn >> 1) - 1, -1, -1):
        _mh_down(HK, HA, HB, HC, p, hn)
    h_after_compact = hn
    # compress heap (filled during the merge phase)
    ccap = max(m, 16)
    CK = np.empty(ccap, np.float64)
    CA = np.empty(ccap, np.int64)
    CV = np.empty(ccap, np.int64)
    CD = np.zeros(ccap, np.float64)       # dummy 4th column for the shared heap helpers
    cn = 0
    ver = np.zeros(cap, np.int64)
    # scratch
    acc = np.zeros(cap, np.float64)
    stamp = np.full(cap, -1, np.int64)
    touched = np.empty(cap, np.int64)

    nn = m
    unmerged = m
    n_pops = 0
    n_push = hn
    n_compact = 0
    while unmerged > 1 and hn > 0:
        a = HA[0]
        b = HB[0]
        cut = HC[0]
        hn -= 1
        if hn > 0:
            HK[0], HA[0], HB[0], HC[0] = HK[hn], HA[hn], HB[hn], HC[hn]
            _mh_down(HK, HA, HB, HC, 0, hn)
        n_pops += 1
        if merged[a] or merged[b]:
            continue
        x = nn
        nn += 1
        merged[a] = True
        merged[b] = True
        parent[a] = x
        parent[b] = x
        vol[x] = vol[a] + vol[b]
        g[x] = g[a] + g[b] - 2 * cut
        ch[x] = max(ch[a], ch[b]) + 1
        ccut[x] = cut
        uf[a] = x
        uf[b] = x
        for c0 in (a, b):
            if ch[c0] > 0:
                if cn == ccap:
                    ccap *= 2
                    CK2 = np.empty(ccap, np.float64)
                    CA2 = np.empty(ccap, np.int64)
                    CV2 = np.empty(ccap, np.int64)
                    CK2[:cn] = CK[:cn]
                    CA2[:cn] = CA[:cn]
                    CV2[:cn] = CV[:cn]
                    CK, CA, CV = CK2, CA2, CV2
                    CD = np.zeros(ccap, np.float64)
                CK[cn] = _compress_delta(ccut[c0], vol[c0], vol[x], smooth)
                CA[cn] = c0
                CV[cn] = 0
                cn += 1
        unmerged -= 1
        # neighbour list of x: resolve both children's lists, aggregate cuts
        nt = 0
        for c0 in (a, b):
            s0 = pstart[c0]
            for e in range(s0, s0 + plen[c0]):
                t = _find(uf, pt[e])
                if t == x:
                    continue
                if stamp[t] != x:
                    stamp[t] = x
                    acc[t] = 0.0
                    touched[nt] = t
                    nt += 1
                acc[t] += pw[e]
        pool_live += nt - plen[a] - plen[b]
        if pn + nt > pcap or pn > 2 * pool_live + 1000000:
            # compact the pool: keep the lists of active communities only
            newcap = max(pcap, 2 * (pool_live + nt) + 16)
            pt2 = np.empty(newcap, np.int64)
            pw2 = np.empty(newcap, np.float64)
            q = 0
            for c1 in range(nn):
                if merged[c1] or c1 == x:
                    plen[c1] = 0 if merged[c1] else plen[c1]
                    continue
                s1 = pstart[c1]
                L = plen[c1]
                pstart[c1] = q
                for e in range(L):
                    pt2[q] = pt[s1 + e]
                    pw2[q] = pw[s1 + e]
                    q += 1
            pt, pw, pcap, pn = pt2, pw2, newcap, q
            n_compact += 1
        pstart[x] = pn
        plen[x] = nt
        if hn + nt > hcap:
            # grow (and compact first if mostly stale)
            if hn > 2 * h_after_compact + 1000000:
                q = 0
                for e in range(hn):
                    if not merged[HA[e]] and not merged[HB[e]]:
                        HK[q], HA[q], HB[q], HC[q] = HK[e], HA[e], HB[e], HC[e]
                        q += 1
                hn = q
                for p in range((hn >> 1) - 1, -1, -1):
                    _mh_down(HK, HA, HB, HC, p, hn)
                h_after_compact = hn
                n_compact += 1
            if hn + nt > hcap:
                hcap2 = max(2 * hcap, hn + nt + 16)
                HK2 = np.empty(hcap2, np.float64)
                HA2 = np.empty(hcap2, np.int64)
                HB2 = np.empty(hcap2, np.int64)
                HC2 = np.empty(hcap2, np.float64)
                HK2[:hn] = HK[:hn]
                HA2[:hn] = HA[:hn]
                HB2[:hn] = HB[:hn]
                HC2[:hn] = HC[:hn]
                HK, HA, HB, HC, hcap = HK2, HA2, HB2, HC2, hcap2
        for i in range(nt):
            t = touched[i]
            pt[pn] = t
            pw[pn] = acc[t]
            pn += 1
            HK[hn] = _combine(vol[t], vol[x], g[t], g[x], acc[t], g_vol, smooth)
            HA[hn] = t
            HB[hn] = x
            HC[hn] = acc[t]
            _mh_up(HK, HA, HB, HC, hn)
            hn += 1
            n_push += 1
    root = nn - 1
    if unmerged > 1:
        x = nn
        nn += 1
        mx = 0
        for i in range(x):
            if not merged[i]:
                if ch[i] > mx:
                    mx = ch[i]
        vol[x] = g_vol
        g[x] = 0.0
        ch[x] = mx + 1
        ccut[x] = 0.0
        for i in range(x):
            if not merged[i]:
                merged[i] = True
                parent[i] = x
                if ch[i] > 0:
                    if cn == ccap:
                        ccap *= 2
                        CK2 = np.empty(ccap, np.float64)
                        CA2 = np.empty(ccap, np.int64)
                        CV2 = np.empty(ccap, np.int64)
                        CK2[:cn] = CK[:cn]
                        CA2[:cn] = CA[:cn]
                        CV2[:cn] = CV[:cn]
                        CK, CA, CV = CK2, CA2, CV2
                        CD = np.zeros(ccap, np.float64)
                    CK[cn] = _compress_delta(ccut[i], vol[i], vol[x], smooth)
                    CA[cn] = i
                    CV[cn] = 0
                    cn += 1
        root = x
    n_cpops = 0
    if k > 0 and ch[root] > k:
        # children lists (doubly linked) and counts of children at height h - 1
        fc = np.full(nn, -1, np.int64)
        nx_ = np.full(nn, -1, np.int64)
        pv = np.full(nn, -1, np.int64)
        cnt = np.zeros(nn, np.int64)
        for c1 in range(nn):
            p = parent[c1]
            if p >= 0:
                h0 = fc[p]
                nx_[c1] = h0
                if h0 >= 0:
                    pv[h0] = c1
                fc[p] = c1
                if ch[c1] == ch[p] - 1:
                    cnt[p] += 1
        moved = np.empty(nn, np.int64)
        for p in range((cn >> 1) - 1, -1, -1):
            _mh_down(CK, CA, CV, CD, p, cn)
        while ch[root] > k:
            if cn == 0:
                raise RuntimeError("compress heap exhausted")
            node = CA[0]
            v0 = CV[0]
            cn -= 1
            if cn > 0:
                CK[0], CA[0], CV[0], CD[0] = CK[cn], CA[cn], CV[cn], CD[cn]
                _mh_down(CK, CA, CV, CD, 0, cn)
            n_cpops += 1
            if v0 != ver[node]:
                continue
            if _depth(parent, ch, node) <= k:
                continue
            p = parent[node]
            ccut[p] += ccut[node]
            # unlink node from p's children
            if pv[node] >= 0:
                nx_[pv[node]] = nx_[node]
            else:
                fc[p] = nx_[node]
            if nx_[node] >= 0:
                pv[nx_[node]] = pv[node]
            # move node's children to p
            nm = 0
            c1 = fc[node]
            while c1 >= 0:
                nxt = nx_[c1]
                parent[c1] = p
                h0 = fc[p]
                nx_[c1] = h0
                pv[c1] = -1
                if h0 >= 0:
                    pv[h0] = c1
                fc[p] = c1
                moved[nm] = c1
                nm += 1
                c1 = nxt
            fc[node] = -1
            alive[node] = False
            parent[node] = -2
            if ch[p] - ch[node] == 1:
                cnt[p] -= 1
                q = p
                while True:
                    if cnt[q] > 0:
                        break
                    mx = -1
                    c1 = fc[q]
                    while c1 >= 0:
                        if ch[c1] > mx:
                            mx = ch[c1]
                        c1 = nx_[c1]
                    if ch[q] == mx + 1:
                        break
                    old = ch[q]
                    ch[q] = mx + 1
                    cc = 0
                    c1 = fc[q]
                    while c1 >= 0:
                        if ch[c1] == mx:
                            cc += 1
                        c1 = nx_[c1]
                    cnt[q] = cc
                    pq = parent[q]
                    if pq == -1:
                        break
                    if old == ch[pq] - 1:
                        cnt[pq] -= 1
                    q = pq
            if ch[root] == k:
                break
            # refresh the entries of p and of the moved children
            npush = nm + 1
            if cn + npush > ccap:
                ccap = max(2 * ccap, cn + npush + 16)
                CK2 = np.empty(ccap, np.float64)
                CA2 = np.empty(ccap, np.int64)
                CV2 = np.empty(ccap, np.int64)
                CK2[:cn] = CK[:cn]
                CA2[:cn] = CA[:cn]
                CV2[:cn] = CV[:cn]
                CK, CA, CV = CK2, CA2, CV2
                CD = np.zeros(ccap, np.float64)
            anc = 0
            a1 = p
            while parent[a1] != -1:
                a1 = parent[a1]
                anc += 1
            if p != root and anc + ch[p] > k:
                ver[p] += 1
                CK[cn] = _compress_delta(ccut[p], vol[p], vol[parent[p]], smooth)
                CA[cn] = p
                CV[cn] = ver[p]
                _mh_up(CK, CA, CV, CD, cn)
                cn += 1
            for i in range(nm):
                c1 = moved[i]
                if ch[c1] == 0:
                    continue
                if anc + 1 + ch[c1] > k:
                    ver[c1] += 1
                    CK[cn] = _compress_delta(ccut[c1], vol[c1], vol[p], smooth)
                    CA[cn] = c1
                    CV[cn] = ver[c1]
                    _mh_up(CK, CA, CV, CD, cn)
                    cn += 1
    stats = np.array([n_pops, n_push, n_cpops, n_compact, hcap, pcap], np.int64)
    return nn, parent[:nn].copy(), ch[:nn].copy(), vol[:nn].copy(), g[:nn].copy(), alive[:nn].copy(), root, stats


@nb.njit(cache=True)
def _balance(nn, parent, ch, vol, g, root):
    """``check_balance``: every leaf child of the root gets a unary parent, in ascending id order.
    Returns extended (nn, parent, ch, vol, g)."""
    extra = 0
    for i in range(nn):
        if parent[i] == root and ch[i] == 0:
            extra += 1
    if extra == 0:
        return nn, parent, ch, vol, g
    n2 = nn + extra
    P = np.empty(n2, np.int64)
    H = np.empty(n2, np.int64)
    V = np.empty(n2, np.float64)
    G = np.empty(n2, np.float64)
    P[:nn] = parent
    H[:nn] = ch
    V[:nn] = vol
    G[:nn] = g
    u = nn
    for i in range(nn):
        if parent[i] == root and ch[i] == 0:
            P[u] = root
            H[u] = 1
            V[u] = vol[i]
            G[u] = g[i]
            P[i] = u
            u += 1
    return n2, P, H, V, G


@nb.njit(cache=True)
def _lg(num, den, smooth):
    if smooth:
        return math.log2((num + 1.0) / (den + 1.0))
    return math.log2(num / den)


@nb.njit(cache=True)
def _leaf_up_all(comm_ptr, comm_leaves, comm_id, cvol, cg, indptr, indices, w, deg, VOL, smooth, pos):
    """``leaf_up`` over every community with >= 3 leaves (communities in ascending id; leaves of
    each in ascending order). ``pos`` is an int64 scratch [N] filled with -1.

    Returns per community: delta contribution order-faithful total, and the flat result arrays:
    created-count per community (ids consumed), the sub-nodes (local id offset, vol, g) with their
    community, and for each leaf its sub-node offset (local created index)."""
    nc = len(comm_id)
    n_created = np.zeros(nc, np.int64)
    sub_ptr = np.zeros(nc + 1, np.int64)
    # upper bound on sub-nodes: number of leaves
    tot = comm_ptr[nc]
    sub_off = np.empty(tot, np.int64)      # created-local index of each sub-node (0-based after leaves)
    sub_vol = np.empty(tot, np.float64)
    sub_g = np.empty(tot, np.float64)
    leaf_sub = np.full(tot, -1, np.int64)  # per leaf (in comm_leaves order): index into sub_* arrays
    delta = 0.0
    ns = 0
    max_nodes = 0
    for c in range(nc):
        s = comm_ptr[c + 1] - comm_ptr[c]
        sub_ptr[c] = ns
        if s < 3:
            continue
        L = comm_leaves[comm_ptr[c]:comm_ptr[c + 1]]
        for i in range(s):
            pos[L[i]] = i
        # induced CSR (rows ascending, neighbours ascending as in the global CSR)
        lptr = np.zeros(s + 1, np.int64)
        for i in range(s):
            v = L[i]
            cc = 0
            for e in range(indptr[v], indptr[v + 1]):
                if pos[indices[e]] >= 0:
                    cc += 1
            lptr[i + 1] = lptr[i] + cc
        lind = np.empty(lptr[s], np.int64)
        lw = np.empty(lptr[s], np.float64)
        svol = np.zeros(s, np.float64)
        for i in range(s):
            v = L[i]
            q = lptr[i]
            acc = 0.0
            for e in range(indptr[v], indptr[v + 1]):
                j = pos[indices[e]]
                if j >= 0:
                    lind[q] = j
                    lw[q] = w[e]
                    acc += w[e]
                    q += 1
            svol[i] = acc
        pvol = cvol[c]
        ori = 0.0
        for i in range(s):
            v = L[i]
            ori += -(deg[v] / VOL) * _lg(deg[v], pvol, smooth)
        sub_gvol = cvol[c] - cg[c]
        nn, par, ch, vol, g, alive, root, st = _build_k_tree(s, svol, svol, lptr, lind, lw, sub_gvol, 2, smooth)
        nn, par, ch, vol, g = _balance(nn, par, ch, vol, g, root)
        n_created[c] = nn - s
        if nn > max_nodes:
            max_nodes = nn
        # sub-nodes = children of root, ascending id; restore global vol/g; entropy in BFS order
        new = 0.0
        first = ns
        for t in range(s, nn):
            if par[t] == root and alive_or_unary(t, alive):
                sub_off[ns] = t - s
                sub_vol[ns] = 0.0
                sub_g[ns] = 0.0
                ns += 1
        # leaf -> sub-node index (each leaf's parent is a root child)
        kmap = np.full(nn, -1, np.int64)
        for q in range(first, ns):
            kmap[sub_off[q] + s] = q
        for i in range(s):
            leaf_sub[comm_ptr[c] + i] = kmap[par[i]]
        # ori_vol of each sub-node: sum of global degrees of its leaves in ascending leaf order
        for i in range(s):
            q = kmap[par[i]]
            sub_vol[q] += deg[L[i]]
        for q in range(first, ns):
            t = sub_off[q] + s
            inner = vol[t] - g[t]
            sub_g[q] = sub_vol[q] - inner
            new += -(sub_g[q] / VOL) * _lg(sub_vol[q], pvol, smooth)
        # leaves in BFS order: grouped by sub-node (ascending), ascending inside (counting sort)
        nsub = ns - first
        bptr = np.zeros(nsub + 1, np.int64)
        for i in range(s):
            bptr[kmap[par[i]] - first + 1] += 1
        for q in range(nsub):
            bptr[q + 1] += bptr[q]
        border = np.empty(s, np.int64)
        fill = bptr[:nsub].copy()
        for i in range(s):
            q = kmap[par[i]] - first
            border[fill[q]] = i
            fill[q] += 1
        for q in range(nsub):
            for t in range(bptr[q], bptr[q + 1]):
                v = L[border[t]]
                new += -(deg[v] / VOL) * _lg(deg[v], sub_vol[q + first], smooth)
        delta += ori - new
        for i in range(s):
            pos[L[i]] = -1
    sub_ptr[nc] = ns
    return delta, n_created, sub_ptr, sub_off[:ns].copy(), sub_vol[:ns].copy(), sub_g[:ns].copy(), leaf_sub, max_nodes


@nb.njit(cache=True, inline="always")
def alive_or_unary(t, alive):
    return t >= len(alive) or alive[t]


@nb.njit(cache=True)
def _aggregate_pairs(ra, rb, wv, m):
    """Symmetric CSR over m community ids from directed entries (ra -> rb, weight), ra != rb,
    summing duplicates; rows and columns ascending."""
    n = len(ra)
    key = np.empty(n, np.int64)
    for i in range(n):
        key[i] = ra[i] * m + rb[i]
    order = np.argsort(key, kind="mergesort")
    indptr = np.zeros(m + 1, np.int64)
    ind = np.empty(n, np.int64)
    ww = np.empty(n, np.float64)
    q = -1
    last = -1
    for t in range(n):
        i = order[t]
        kk = key[i]
        if kk != last:
            q += 1
            ind[q] = rb[i]
            ww[q] = wv[i]
            indptr[ra[i] + 1] += 1
            last = kk
        else:
            ww[q] += wv[i]
    for i in range(m):
        indptr[i + 1] += indptr[i]
    return indptr, ind[:q + 1].copy(), ww[:q + 1].copy()


# ------------------------------------------------------------------------------------------------
# v2 driver


@dataclass
class TreeResult:
    """A balanced coding tree. ``parent[i]`` is the global id of node i's parent (-1 for the root,
    -2 for ids that are not in the tree: compressed or never applied). Leaves are 0..n-1."""
    n: int
    k: int
    parent: np.ndarray
    root: int
    smooth: bool
    stats: dict

    def levels(self) -> list[np.ndarray]:
        """Global ids per level (level 0 = leaves 0..n-1), ascending id within a level (the
        reference's ``update_node`` order)."""
        out = [np.arange(self.n, dtype=np.int64)]
        anc = out[0]
        for _ in range(self.k):
            anc = self.parent[anc]
            out.append(np.unique(anc))
        assert len(out[-1]) == 1 and out[-1][0] == self.root
        return out

    def layer_data(self, indptr, indices) -> dict:
        """``extract_layer_data`` equivalent: node_size [k+1]; parent_local[d] (d = 1..k) int64
        [node_size[d-1]] = local index at level d of the parent of each level-(d-1) node;
        edges[d] (d = 0..k-1) int64 [2, E_d] quotient-graph edges (symmetric, no self-loops,
        deduplicated; level 0 = the input graph)."""
        lv = self.levels()
        node_size = [len(x) for x in lv]
        loc = np.full(len(self.parent), -1, np.int64)
        for x in lv:
            loc[x] = np.arange(len(x))
        parent_local = [None]
        for d in range(1, self.k + 1):
            parent_local.append(loc[self.parent[lv[d - 1]]])
        n = self.n
        src = np.repeat(np.arange(n, dtype=np.int64), np.diff(np.asarray(indptr)))
        dst = np.asarray(indices, dtype=np.int64)
        edges = [np.stack([src, dst])]
        a, b = src, dst
        for d in range(1, self.k):
            a, b = parent_local[d][a], parent_local[d][b]
            keep = a != b
            key = np.unique(a[keep] * node_size[d] + b[keep])
            a, b = key // node_size[d], key % node_size[d]
            edges.append(np.stack([a, b]))
        return {"node_size": node_size, "parent_local": parent_local, "edges": edges}


class CodingTreeBuilder:
    """Sparse ``PartitionTree(adj).build_coding_tree(k, mode='v2')``.

    ``indptr``/``indices``/``weights``: symmetric CSR without self-loops (rows with ascending
    neighbours for bit-exact float sums with the reference on weighted graphs)."""

    def __init__(self, indptr, indices, weights=None, *, smooth: bool = True, verbose: bool = False):
        self.indptr = np.ascontiguousarray(indptr, dtype=np.int64)
        self.indices = np.ascontiguousarray(indices, dtype=np.int64)
        self.n = len(self.indptr) - 1
        self.w = (np.ones(len(self.indices), np.float64) if weights is None
                  else np.ascontiguousarray(weights, dtype=np.float64))
        self.deg = np.zeros(self.n, np.float64)
        # same summation order as the reference graph_parse (row-wise, ascending column)
        self.deg = _row_sums(self.indptr, self.w)
        self.VOL = _total(self.deg)
        self.smooth = smooth
        self.verbose = verbose
        self.stats: dict = {"rounds": []}

    # -- storage ------------------------------------------------------------------------------
    def _ensure(self, need: int) -> None:
        if need <= len(self.par):
            return
        cap = max(need, int(len(self.par) * 1.5) + 16)
        for name, fill, dt in (("par", -2, np.int64), ("vol", 0.0, np.float64), ("g", 0.0, np.float64)):
            old = getattr(self, name)
            new = np.full(cap, fill, dt)
            new[:len(old)] = old
            setattr(self, name, new)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[sebot-tree] {msg}", flush=True)

    # -- reference steps ----------------------------------------------------------------------
    def _initial(self) -> None:
        t = time.time()
        n = self.n
        nn, par, ch, vol, g, alive, root, st = _build_k_tree(n, self.deg, self.deg, self.indptr, self.indices,
                                                             self.w, self.VOL, 2, self.smooth)
        nn, par, ch, vol, g = _balance(nn, par, ch, vol, g, root)
        par = par.copy()
        par[:len(alive)][~alive] = -2
        self.par, self.vol, self.g = par, vol.copy(), g.copy()
        self.next_id = nn
        self.root = int(root)
        self.h = max(int(ch[root]), 2)
        self.stats["initial"] = {"seconds": round(time.time() - t, 3), "nodes": int(nn),
                                 "merge_pops": int(st[0]), "merge_pushes": int(st[1]), "compress_pops": int(st[2]),
                                 "compactions": int(st[3]), "heap_cap": int(st[4]), "pool_cap": int(st[5])}
        self._log(f"initial k=2 tree: {self.stats['initial']}")

    def _leaf_up(self):
        n = self.n
        H = self.par[:n]
        order = np.argsort(H, kind="stable")           # leaves grouped by community, ascending inside
        Hs = H[order]
        comm_id, start = np.unique(Hs, return_index=True)
        comm_ptr = np.append(start, n).astype(np.int64)
        pos = np.full(n, -1, np.int64)
        delta, n_created, sub_ptr, sub_off, sub_vol, sub_g, leaf_sub, max_nodes = _leaf_up_all(
            comm_ptr, order.astype(np.int64), comm_id.astype(np.int64), self.vol[comm_id], self.g[comm_id],
            self.indptr, self.indices, self.w, self.deg, self.VOL, self.smooth, pos)
        # global ids: communities in ascending order consume ids for their builds now
        base = self.next_id + np.concatenate([[0], np.cumsum(n_created)[:-1]])
        self.next_id += int(n_created.sum())
        sizes = np.diff(comm_ptr)
        return {"delta": delta / n, "comm_id": comm_id, "comm_ptr": comm_ptr, "leaves": order, "sizes": sizes,
                "base": base, "sub_ptr": sub_ptr, "sub_off": sub_off, "sub_vol": sub_vol, "sub_g": sub_g,
                "leaf_sub": leaf_sub, "max_nodes": int(max_nodes)}

    def _root_down(self):
        rc = np.flatnonzero(self.par[:self.next_id] == self.root)
        if len(rc) < 3:
            return {"delta": 0.0, "none": True}
        m = len(rc)
        loc = np.full(self.next_id, -1, np.int64)
        loc[rc] = np.arange(m)
        anc = np.arange(self.n, dtype=np.int64)
        for _ in range(self.h - 1):
            anc = self.par[anc]
        memb = loc[anc]
        assert (memb >= 0).all()
        src = np.repeat(np.arange(self.n, dtype=np.int64), np.diff(self.indptr))
        ra, rb = memb[src], memb[self.indices]
        keep = ra != rb
        ip, ind, ww = _aggregate_pairs(ra[keep], rb[keep], self.w[keep], m)
        vol0, g0 = self.vol[rc].copy(), self.g[rc].copy()
        g_vol = float(self.vol[self.root])
        ori = 0.0
        for i in range(m):
            if self.smooth:
                ori += -(g0[i] / g_vol) * math.log2((vol0[i] + 1) / g_vol)
            else:
                ori += -(g0[i] / g_vol) * math.log2(vol0[i] / g_vol)
        nn, par, ch, vol, g, alive, root, st = _build_k_tree(m, vol0, g0, ip, ind, ww, g_vol, 2, self.smooth)
        nn, par, ch, vol, g = _balance(nn, par, ch, vol, g, root)
        alive_full = np.ones(nn, bool)
        alive_full[:len(alive)] = alive
        new = _rd_entropy(nn, par, vol, g, alive_full, root, self.VOL, self.smooth)
        delta = (ori - new) / m
        gid = np.empty(nn, np.int64)
        gid[:m] = rc
        gid[m:] = self.next_id + np.arange(nn - m)
        self.next_id += nn - m
        return {"delta": delta, "none": False, "rc": rc, "m": m, "gid": gid, "par": par, "vol": vol, "g": g,
                "alive": alive_full, "root": int(root), "merge_pushes": int(st[1])}

    def _apply_leaf_up(self, LU) -> None:
        sizes, cp, leaves = LU["sizes"], LU["comm_ptr"], LU["leaves"]
        # single_up of the 1-2 leaf communities consumes ids now, ascending community then leaf
        small = sizes < 3
        n_small_leaves = int(sizes[small].sum())
        self._ensure(self.next_id + n_small_leaves + 1)
        # sub-nodes of the big communities
        sub_ptr, sub_off = LU["sub_ptr"], LU["sub_off"]
        comm_of_sub = np.repeat(np.arange(len(sizes)), np.diff(sub_ptr))
        sub_gid = LU["base"][comm_of_sub] + sub_off
        self._ensure(int(sub_gid.max()) + 1 if len(sub_gid) else 0)
        self.par[sub_gid] = LU["comm_id"][comm_of_sub]
        self.vol[sub_gid] = LU["sub_vol"]
        self.g[sub_gid] = LU["sub_g"]
        ls = LU["leaf_sub"]
        big_leaf = ls >= 0
        self.par[leaves[big_leaf]] = sub_gid[ls[big_leaf]]
        # small communities: unary parent for each leaf
        leaf_comm = np.repeat(np.arange(len(sizes)), sizes)
        sl = leaves[small[leaf_comm]]                    # already ascending community, then leaf
        u = self.next_id + np.arange(len(sl))
        self.par[u] = self.par[sl]
        self.vol[u] = self.vol[sl]
        self.g[u] = self.g[sl]
        self.par[sl] = u
        self.next_id += len(sl)
        self.h += 1

    def _apply_root_down(self, RD) -> None:
        gid, par, alive, root = RD["gid"], RD["par"], RD["alive"], RD["root"]
        nn = len(gid)
        self._ensure(int(gid.max()) + 1)
        idx = np.arange(nn)
        keep = alive & (idx != root)
        p = par[keep]
        gp = np.where(p == root, self.root, gid[np.maximum(p, 0)])
        self.par[gid[keep]] = gp
        created = keep & (idx >= RD["m"])
        self.vol[gid[created]] = RD["vol"][created]
        self.g[gid[created]] = RD["g"][created]
        self.h += 1

    def build(self, k: int) -> TreeResult:
        if k < 2:
            raise ValueError("k >= 2 (k = 1 is a no-op upstream)")
        t0 = time.time()
        self._initial()
        flag = 0
        LU = RD = None
        while self.h < k:
            t = time.time()
            if flag in (0, 1):
                LU = self._leaf_up()
            if flag in (0, 2):
                RD = self._root_down()
            lu_d, rd_d = LU["delta"], RD["delta"]
            if lu_d < rd_d and RD["none"]:
                # upstream would call root_down_update(None, None) and crash; take leaf_up instead
                self.stats.setdefault("guarded_none_root_down", 0)
                self.stats["guarded_none_root_down"] += 1
                lu_d = rd_d
            if lu_d < rd_d:
                flag = 2
                self._apply_root_down(RD)
                choice = "root_down"
            else:
                flag = 1
                self._apply_leaf_up(LU)
                choice = "leaf_up"
            rec = {"height": self.h, "choice": choice, "leaf_up_delta": lu_d, "root_down_delta": rd_d,
                   "seconds": round(time.time() - t, 3), "ids": int(self.next_id)}
            self.stats["rounds"].append(rec)
            self._log(str(rec))
        self.stats["seconds"] = round(time.time() - t0, 3)
        par = self.par[:self.next_id].copy()
        return TreeResult(self.n, k, par, self.root, self.smooth, self.stats)


@nb.njit(cache=True)
def _row_sums(indptr, w):
    n = len(indptr) - 1
    out = np.zeros(n, np.float64)
    for i in range(n):
        acc = 0.0
        for e in range(indptr[i], indptr[i + 1]):
            acc += w[e]
        out[i] = acc
    return out


@nb.njit(cache=True)
def _total(x):
    acc = 0.0
    for i in range(len(x)):
        acc += x[i]
    return acc


@nb.njit(cache=True)
def _rd_entropy(nn, par, vol, g, alive, root, VOL, smooth):
    """``entropy(subgraph_node_dict)`` of a root-down build: ascending id over the alive nodes with
    a parent (the copies first, then the created nodes)."""
    ent = 0.0
    for i in range(nn):
        if not alive[i] or i == root:
            continue
        p = par[i]
        if p < 0:
            continue
        ent += -(g[i] / VOL) * _lg(vol[i], vol[p], smooth)
    return ent


def build_tree(indptr, indices, k: int, *, weights=None, smooth: bool = True, verbose: bool = False) -> TreeResult:
    return CodingTreeBuilder(indptr, indices, weights, smooth=smooth, verbose=verbose).build(k)
