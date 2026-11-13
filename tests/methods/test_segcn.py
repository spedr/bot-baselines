"""SEGCN: sparse random-walk subgraph store vs a naive dense (GNN-AK) extractor, batched model vs a
naive per-subgraph dense model, the C6 vs 2xC3 expressiveness check (spec 11.1), batch packing,
no-leak (flipped val/test labels) and ExtScore plumbing. Synthetic, CPU."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from bb.io import save_npy, write_json
from bb.paths import dataset_paths


# ------------------------------------------------------------------------------------------------ helpers
def _csr(n, edges):
    """Simple undirected CSR (sorted rows, no self-loops, dedup) from an edge list."""
    s = {(min(u, v), max(u, v)) for u, v in edges if u != v}
    a = np.array([[u, v] for u, v in s] + [[v, u] for u, v in s], dtype=np.int64).reshape(-1, 2)
    key = np.unique(a[:, 0] * n + a[:, 1]) if len(a) else np.zeros(0, np.int64)
    u, v = key // n, key % n
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(u, minlength=n), out=indptr[1:])
    return indptr, v.astype(np.int32)


def _random_graph(n=40, m=70, seed=0):
    rng = np.random.default_rng(seed)
    e = [tuple(rng.integers(0, n - 4, 2)) for _ in range(m)]
    e += [(0, k) for k in range(5, 20)]                  # a hub
    return _csr(n, e)                                    # the last 4 nodes stay isolated


class _G:
    """Minimal GraphInputs stand-in (CPU float32)."""

    def __init__(self, n, num_dim=5, cat_dim=3, seed=0, extra=False, const=False):
        r = torch.Generator().manual_seed(seed)
        f = (lambda *s: torch.ones(*s)) if const else (lambda *s: torch.randn(*s, generator=r))
        self.des, self.tweet = f(n, 768), f(n, 768)
        self.num, self.cat = f(n, num_dim), f(n, cat_dim)
        self.extra = torch.randn(n, 2, generator=r) if extra else None
        self.y = torch.randint(0, 2, (n,), generator=r)

    @property
    def n(self):
        return int(self.y.shape[0])

    @property
    def device(self):
        return self.y.device


def _randomize_bn(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.running_mean.copy_(torch.randn(m.num_features, generator=g) * 0.3)
            m.running_var.copy_(torch.rand(m.num_features, generator=g) + 0.5)
            m.weight.data.copy_(torch.rand(m.num_features, generator=g) + 0.5)
            m.bias.data.copy_(torch.randn(m.num_features, generator=g) * 0.1)


# ------------------------------------------------------------------------------------------------ store
@pytest.mark.parametrize("p,q,repeat", [(1.0, 1.0, 1), (1.0, 0.5, 1), (2.0, 1.0, 2)])
def test_store_equals_dense_gnnak_extraction(p, q, repeat):
    from bb.methods.segcn.subgraphs import build_in_memory, dense_reference, random_walks
    indptr, indices = _random_graph()
    n = len(indptr) - 1
    st = build_in_memory(indptr, indices, walk_length=30, p=p, q=q, repeat=repeat, seed=3)
    walks = random_walks(indptr, indices, np.arange(n), 30, p, q, 3, repeat)
    # every step is an edge, or a stay on an isolated node
    for u in range(n):
        w = walks[u]
        for r in range(repeat):
            seq = w[r * 31:(r + 1) * 31]
            assert seq[0] == u
            for a, b in zip(seq[:-1], seq[1:]):
                row = indices[indptr[a]:indptr[a + 1]]
                assert (b in row) or (len(row) == 0 and a == b)
    node_mask, edge_mask, src, dst = dense_reference(indptr, indices, walks)
    for u in range(n):
        nodes = st.nodes(u)
        assert nodes[0] == u and len(set(nodes.tolist())) == len(nodes)
        assert set(nodes.tolist()) == set(np.flatnonzero(node_mask[u]).tolist())
        got = {(min(nodes[i], nodes[j]), max(nodes[i], nodes[j])) for i, j in st.pairs(u)}
        ref = {(min(a, b), max(a, b)) for a, b in zip(src[edge_mask[u]], dst[edge_mask[u]])}
        assert got == ref
        assert all(i < j for i, j in st.pairs(u))
        # BFS hops inside the induced subgraph
        hop = np.asarray(st.hop[st.node_ptr[u]:st.node_ptr[u + 1]])
        loc = {v: k for k, v in enumerate(nodes.tolist())}
        dist = {u: 0}
        frontier = [u]
        while frontier:
            nxt = []
            for a in frontier:
                for b in indices[indptr[a]:indptr[a + 1]]:
                    if b in loc and b not in dist:
                        dist[int(b)] = dist[a] + 1
                        nxt.append(int(b))
            frontier = nxt
        assert all(hop[loc[v]] == d for v, d in dist.items())
    # isolated roots: singleton subgraphs without edges
    for u in range(n - 4, n):
        assert st.nodes(u).tolist() == [u] and len(st.pairs(u)) == 0
    # deterministic and independent of chunking
    st2 = build_in_memory(indptr, indices, walk_length=30, p=p, q=q, repeat=repeat, seed=3, chunk_roots=7)
    for k in ("node_ptr", "node_ids", "hop", "edge_ptr", "edge_pairs"):
        assert np.array_equal(getattr(st, k), getattr(st2, k))
    st3 = build_in_memory(indptr, indices, walk_length=30, p=p, q=q, repeat=repeat, seed=4)
    assert not np.array_equal(st.node_ids, st3.node_ids)


@pytest.mark.parametrize("p,q", [(1.0, 1.0), (1.0, 0.5), (0.25, 4.0), (4.0, 0.25)])
def test_walk_distribution_is_node2vec(p, q):
    """Two-step walk law from a root vs the exact node2vec second-order law (review round 1: the store
    test above only checks that walks are valid, not how they are distributed)."""
    from bb.methods.segcn.subgraphs import random_walks
    # root 0: triangle 0-1-2, square 0-3-4-1, pendant 6, path 2-5-7 (return / BFS / DFS moves all occur)
    edges = [(0, 1), (0, 2), (1, 2), (0, 3), (3, 4), (4, 1), (0, 6), (2, 5), (5, 7)]
    indptr, indices = _csr(8, edges)
    adj = {u: set(indices[indptr[u]:indptr[u + 1]].tolist()) for u in range(8)}
    R = 200_000
    w = random_walks(indptr, indices, np.array([0], np.int64), 2, p, q, 7, R).reshape(R, 3)
    exact = {}
    for x in adj[0]:
        zs = sorted(adj[x])
        ws = np.array([1 / p if z == 0 else (1.0 if z in adj[0] else 1 / q) for z in zs])
        for z, wz in zip(zs, ws / ws.sum()):
            exact[(x, z)] = exact.get((x, z), 0.0) + wz / len(adj[0])
    keys, cnt = np.unique(w[:, 1] * 8 + w[:, 2], return_counts=True)
    emp = {(int(k) // 8, int(k) % 8): c / R for k, c in zip(keys, cnt)}
    assert set(emp) <= set(exact)
    for k, v in exact.items():
        assert abs(emp.get(k, 0.0) - v) < 5 * np.sqrt(v * (1 - v) / R) + 1e-4, (k, v, emp.get(k))


def test_store_cache_roundtrip(cr15):
    from bb.build import cresci15  # noqa: F401  (fixture already built the dataset)
    from bb.methods.segcn.subgraphs import load_or_build_store, segcn_graph
    rels = ["friend", "follower"]
    a = load_or_build_store("cresci15", rels, seed=0, verbose=False)
    b = load_or_build_store("cresci15", rels, seed=0, verbose=False)     # cached: memmaps
    assert isinstance(b.node_ids, np.memmap) and np.array_equal(a.node_ids, b.node_ids)
    assert a.meta["s_bar"] >= 1 and a.n == segcn_graph("cresci15", rels).n


# ------------------------------------------------------------------------------------------------ model
def _dense_reference_forward(model, g, indptr, indices, st):
    """Per-node dense recomputation of the eval-mode SEGCN forward (no batching, no sparse ops)."""
    n = len(indptr) - 1
    model.eval()
    with torch.no_grad():
        h = model.encode(g, None)
        for l in range(model.n_layers):
            ker = model.layers[l]
            new = torch.empty_like(h)
            for u in range(n):
                nodes = torch.as_tensor(st.nodes(u).astype(np.int64))
                k = len(nodes)
                A = torch.zeros(k, k)
                for v in range(k):                       # induced edges straight from the graph
                    for w in range(k):
                        if v != w and int(nodes[w]) in set(indices[indptr[nodes[v]]:indptr[nodes[v] + 1]].tolist()):
                            A[v, w] = 1.0
                A = A + torch.eye(k)
                d = A.sum(1).rsqrt()
                Ah = d[:, None] * A * d[None, :]
                x = h[nodes]
                if ker.hop_emb is not None:
                    hop = torch.as_tensor(np.asarray(st.hop[st.node_ptr[u]:st.node_ptr[u + 1]]).astype(np.int64))
                    x = torch.cat([x, ker.hop_emb((hop + 1).clamp(max=19))], 1)
                prev = x
                for conv, bn in zip(ker.gnn.convs, ker.gnn.norms):
                    x = torch.relu(bn(Ah @ conv(x))) + prev
                    prev = x
                x = ker.gnn.output_encoder(x)
                cen = x[0]
                s = torch.relu(ker.sub_bn(ker.sub_lin(x)))
                pooled = s.mean(0) if ker.pool == "mean" else s.sum(0)
                parts = [cen, pooled] + ([g.extra[u]] if model.extscore_inject == "root" else [])
                kk = ker.out_encoder(torch.cat(parts)[None])[0]
                new[u] = h[u] + torch.relu(model.norms[l](kk[None])[0])
            h = new
        return model.head(h, g, None)


@pytest.mark.parametrize("inner,pool,hop_dim,extscore", [(1, "mean", 0, "num"), (2, "sum", 16, "root"),
                                                     (1, "mean", 0, "late"), (1, "sum", 0, "branch")])
def test_batched_model_equals_dense_reference(inner, pool, hop_dim, extscore):
    from bb.methods.segcn.model import SEGCN, GPUStore
    from bb.methods.segcn.subgraphs import build_in_memory
    indptr, indices = _random_graph()
    n = len(indptr) - 1
    st = build_in_memory(indptr, indices, walk_length=30, seed=1)
    g = _G(n, extra=True)
    torch.manual_seed(0)
    jw = 0 if extscore == "num" else 2
    model = SEGCN(num_dim=5, cat_dim=3, hidden=16, n_layers=2, n_inner=inner, dropout=0.3, hop_dim=hop_dim,
                  pool=pool, extscore_width=jw, extscore_inject=extscore)
    _randomize_bn(model)
    model.eval()
    ref = _dense_reference_forward(model, g, indptr, indices, st)
    store = GPUStore(st, "cpu")
    with torch.no_grad():
        allr = torch.arange(n)
        lw = model.predict_layerwise(g, store, allr, chunk_rows=50)          # many small chunks
        roots = torch.tensor([3, 0, 37, 11, 5])
        s0, levels = store.plan(roots, 2)
        mb = model.forward_plan(g, s0, levels)
    assert torch.allclose(lw, ref, atol=1e-5), (lw - ref).abs().max()
    assert torch.allclose(mb, ref[roots], atol=1e-5)
    sub = model.predict_layerwise(g, store, roots)
    assert torch.allclose(sub, ref[roots], atol=1e-5)


def test_predict_uses_eval_mode():
    """predict_subgraph / predict_global must run in eval mode (no dropout, BN running stats) whatever
    the caller's mode, and restore it."""
    from bb.methods.segcn.model import SEGCN, GPUStore
    from bb.methods.segcn.subgraphs import build_in_memory
    from bb.methods.segcn.train import TrainConfig, predict_global, predict_subgraph, global_adjacency
    indptr, indices = _random_graph()
    n = len(indptr) - 1
    g = _G(n)
    store = GPUStore(build_in_memory(indptr, indices, seed=0), "cpu")
    torch.manual_seed(0)
    m = SEGCN(num_dim=5, cat_dim=3, hidden=16, dropout=0.5)
    _randomize_bn(m)
    m.eval()
    with torch.no_grad():
        ref = torch.softmax(m.predict_layerwise(g, store, torch.arange(n)), 1)[:, 1].numpy()
    m.train()
    p = predict_subgraph(m, g, store, np.arange(n), TrainConfig())
    assert m.training and np.allclose(p, ref, atol=1e-6)

    class C:
        pass
    c = C()
    c.indptr, c.indices = indptr, indices
    adj = global_adjacency(c, "cpu")
    m0 = SEGCN(num_dim=5, cat_dim=3, hidden=16, n_inner=0, dropout=0.5)
    _randomize_bn(m0)
    m0.eval()
    with torch.no_grad():
        ref0 = torch.softmax(m0.forward_global(g, adj), 1)[:, 1].numpy()
    m0.train()
    assert np.allclose(predict_global(m0, g, adj), ref0, atol=1e-6) and m0.training


def test_c6_vs_two_triangles():
    """Spec 11.1: plain GCN (inner 0) cannot separate C6 from 2xC3; the subgraph path (sum pool) can."""
    from bb.methods.segcn.model import SEGCN, GPUStore
    from bb.methods.segcn.subgraphs import build_in_memory
    from bb.methods.segcn.train import global_adjacency

    class C:
        pass
    edges = [(i, (i + 1) % 6) for i in range(6)] + [(6, 7), (7, 8), (8, 6), (9, 10), (10, 11), (11, 9)]
    c = C()
    c.indptr, c.indices = _csr(12, edges)
    g = _G(12, const=True)
    torch.manual_seed(0)
    gcn = SEGCN(num_dim=5, cat_dim=3, hidden=16, n_inner=0)
    _randomize_bn(gcn)
    gcn.eval()
    with torch.no_grad():
        out = gcn.forward_global(g, global_adjacency(c, "cpu"))
    assert torch.allclose(out[:6], out[6:], atol=1e-5)
    st = build_in_memory(c.indptr, c.indices, walk_length=30, seed=0)
    assert all(len(st.nodes(u)) == 3 for u in range(6, 12)) and any(len(st.nodes(u)) > 3 for u in range(6))
    for pool, hop in (("sum", 0), ("mean", 16)):
        torch.manual_seed(0)
        m = SEGCN(num_dim=5, cat_dim=3, hidden=16, n_inner=1, pool=pool, hop_dim=hop)
        _randomize_bn(m)
        m.eval()
        with torch.no_grad():
            o = m.predict_layerwise(g, GPUStore(st, "cpu"), torch.arange(12))
        assert torch.allclose(o[6:9], o[9:12], atol=1e-5)            # the two triangles agree
        assert not torch.allclose(o[:6].mean(0), o[6:].mean(0), atol=1e-4)
    # mean pool, constant features, no hops: not separated once every C6 walk covers the cycle (the
    # induced subgraph is then 2-regular like a triangle; a 30-step walk may leave a path, whose
    # endpoints get a different GCN normalisation)
    full = build_in_memory(c.indptr, c.indices, walk_length=200, seed=0)
    assert all(len(full.nodes(u)) == 6 for u in range(6))
    torch.manual_seed(0)
    m = SEGCN(num_dim=5, cat_dim=3, hidden=16, n_inner=1, pool="mean")
    _randomize_bn(m)
    m.eval()
    with torch.no_grad():
        o = m.predict_layerwise(g, GPUStore(full, "cpu"), torch.arange(12))
    assert torch.allclose(o[:6], o[6:], atol=1e-5)


def _reference_plan(st, roots, n_layers):
    """The pre-review-2 GPU batch builder (per-batch degrees, argsort to CSR, torch.unique), on CPU."""
    node_ptr = torch.as_tensor(np.asarray(st.node_ptr, np.int64))
    node_ids = torch.as_tensor(np.asarray(st.node_ids, np.int64))
    edge_ptr = torch.as_tensor(np.asarray(st.edge_ptr, np.int64))
    pairs = torch.as_tensor(np.asarray(st.edge_pairs, np.int64).reshape(-1, 2))

    def segs(ptr, r):
        start = ptr[r]
        cnt = ptr[r + 1] - start
        offs = torch.cumsum(cnt, 0) - cnt
        seg = torch.repeat_interleave(torch.arange(len(r)), cnt)
        return start[seg] + torch.arange(int(cnt.sum())) - offs[seg], seg, offs

    out, cur = [], roots
    for _ in range(n_layers):
        pos, seg, offs = segs(node_ptr, cur)
        ids = node_ids[pos]
        epos, eseg, _ = segs(edge_ptr, cur)
        a, b = pairs[epos, 0] + offs[eseg], pairs[epos, 1] + offs[eseg]
        t = len(ids)
        deg = torch.ones(t).index_add_(0, a, torch.ones(len(a))).index_add_(0, b, torch.ones(len(a)))
        dinv = deg.rsqrt()
        rows, cols = torch.cat([a, b, torch.arange(t)]), torch.cat([b, a, torch.arange(t)])
        vals = torch.cat([dinv[a] * dinv[b], dinv[a] * dinv[b], dinv * dinv])
        order = torch.argsort(rows * t + cols)
        prev, inv = torch.unique(ids, return_inverse=True)
        out.append(dict(ids=ids, seg=seg, root_pos=offs, crow=torch.cumsum(torch.bincount(rows, minlength=t), 0),
                        col=cols[order], val=vals[order], inv=inv, self_pos=torch.searchsorted(prev, cur)))
        cur = prev
    out.reverse()
    return cur, out


@pytest.mark.parametrize("repeat", [1, 2])
def test_host_planner_equals_reference(repeat):
    """Review round 2: the CPU planner (precomputed block CSR + numba dedup) builds the same batch as
    the original GPU builder; the symmetric SpMM backward gives autograd's gradient; the prefetcher
    yields the synchronous plans in order."""
    from bb.methods.segcn.model import GPUStore, Prefetcher, sym_spmm
    from bb.methods.segcn.subgraphs import build_in_memory
    indptr, indices = _random_graph()
    st = build_in_memory(indptr, indices, walk_length=30, seed=2, repeat=repeat)
    store = GPUStore(st, "cpu")
    for roots in ([3, 0, 37, 11, 5], [39], list(range(40))):
        r = torch.tensor(roots)
        s0, levels = store.plan(r, 2)
        s0_ref, ref = _reference_plan(st, r, 2)
        assert torch.equal(s0, s0_ref)
        for lv, rf in zip(levels, ref):
            assert lv.n_rows == len(rf["ids"])
            for k in ("ids", "seg", "root_pos", "inv", "self_pos"):
                assert torch.equal(getattr(lv, k), rf[k]), k
            assert torch.equal(lv.adj.crow_indices()[1:], rf["crow"])
            assert torch.equal(lv.adj.col_indices(), rf["col"])
            assert torch.allclose(lv.adj.values(), rf["val"], rtol=1e-6, atol=0)
            assert torch.equal(lv.cnt, torch.bincount(rf["seg"]).float())
            dense = lv.adj.to_dense()
            assert torch.equal(dense, dense.T)                       # exactly symmetric
            x = torch.randn(lv.n_rows, 3, requires_grad=True)
            w = torch.randn(lv.n_rows, 3)
            (gx,) = torch.autograd.grad((sym_spmm(lv.adj, x) * w).sum(), x)
            (gr,) = torch.autograd.grad(((dense @ x) * w).sum(), x)
            assert torch.allclose(gx, gr, atol=1e-6)
    bs = [np.array(b) for b in ([1, 2, 3], [4, 5], [6, 7, 8, 9], [10, 11])]
    got = list(Prefetcher(store, bs, 2, depth=2))
    assert [list(b) for b, *_ in got] == [list(b) for b in bs]
    for (_, hp, s0, levels), b in zip(got, bs):
        s0_ref, lv_ref = store.plan(torch.as_tensor(b), 2)
        assert torch.equal(s0, s0_ref) and hp.n_rows == sum(lv.n_rows for lv in lv_ref)
        assert all(torch.equal(x.ids, y.ids) and torch.equal(x.adj.values(), y.adj.values())
                   for x, y in zip(levels, lv_ref))


def test_cost_and_packing():
    from bb.methods.segcn.model import GPUStore
    from bb.methods.segcn.subgraphs import _two_layer_rows, build_in_memory
    from bb.methods.segcn.train import pack_batches
    indptr, indices = _random_graph()
    st = build_in_memory(indptr, indices, seed=0)
    store = GPUStore(st, "cpu")
    cost = store.cost(2).numpy()
    assert np.allclose(cost, _two_layer_rows(st))
    roots = torch.arange(10)
    s0, levels = store.plan(roots, 2)
    assert sum(lv.n_rows for lv in levels) <= int(cost[:10].sum())
    perm = np.random.default_rng(0).permutation(len(cost))
    bs = pack_batches(perm, cost, 8, 400)
    assert np.array_equal(np.concatenate(bs), perm)
    assert all(len(b) <= 8 or len(b) == 9 for b in bs) and all(len(b) >= 2 for b in bs)
    assert all(cost[b].sum() <= 400 or len(b) == 2 for b in bs[:-1])


# ------------------------------------------------------------------------------------------------ end to end
def _fake_text(dataset, tables, signal=1.0, seed=0):
    from bb.data import load_labels
    y = load_labels(dataset).astype(np.float32)
    rng = np.random.default_rng(seed)
    for enc, name in tables:
        x = rng.standard_normal((len(y), 768)).astype(np.float32)
        x[:, :4] += signal * (2 * y[:, None] - 1)
        d = dataset_paths(dataset).emb(enc)
        save_npy(d / f"{name}.f16.npy", x.astype(np.float16))
        write_json(d / f"{name}.meta.json", {"rows": len(y), "hf_commit": "fake"})


@pytest.fixture
def cr15_text(cr15):
    _fake_text("cresci15", [("roberta-base", "des"), ("roberta-base", "tweets_first20_mean")])
    return cr15


def test_no_label_leak(cr15_text):
    """Flipping test labels never changes the model (best-val selection); flipping val AND test labels
    never changes it with last-epoch selection (loss on train roots only)."""
    from bb.features import load_features
    from bb.gnn import to_device
    from bb.methods.segcn.model import SEGCN, GPUStore
    from bb.methods.segcn.subgraphs import load_or_build_store
    from bb.methods.segcn.train import TrainConfig, fit_subgraph
    F = load_features("cresci15")
    sp = {s: F.splits[s] for s in ("train", "val", "test")}
    store = GPUStore(load_or_build_store("cresci15", F.relations_by_type(), seed=0, verbose=False), "cpu")
    outs = {}
    for select, flip in (("best_val", None), ("best_val", "test"), ("last_epoch", None), ("last_epoch", "valtest")):
        g = to_device(F, "cpu", edges=False)
        if flip:
            ev = sp["test"] if flip == "test" else np.concatenate([sp["val"], sp["test"]])
            g.y[ev] = 1 - g.y[ev]
        torch.manual_seed(0)
        m = SEGCN(num_dim=5, cat_dim=1, hidden=16)
        r = fit_subgraph(m, g, store, sp, TrainConfig(epochs=3, batch_size=4, select=select, seed=0))
        outs[(select, flip)] = r.prob[sp["test"]]
    assert np.array_equal(outs[("best_val", None)], outs[("best_val", "test")])
    assert np.array_equal(outs[("last_epoch", None)], outs[("last_epoch", "valtest")])


def test_no_label_leak_gcn0(cr15_text):
    """Same flipped-label check for the ``--inner 0`` full-batch path (review round 1: its full-graph
    forward sees every node, so the loss mask and the selection must be the only label paths)."""
    from bb.features import load_features
    from bb.gnn import to_device
    from bb.methods.segcn.model import SEGCN
    from bb.methods.segcn.subgraphs import segcn_graph
    from bb.methods.segcn.train import TrainConfig, fit_global, global_adjacency
    F = load_features("cresci15")
    sp = {s: F.splits[s] for s in ("train", "val", "test")}
    adj = global_adjacency(segcn_graph("cresci15", F.relations_by_type()), "cpu")
    outs = {}
    for select, flip in (("best_val", None), ("best_val", "test"), ("last_epoch", None), ("last_epoch", "valtest")):
        g = to_device(F, "cpu", edges=False)
        if flip:
            ev = sp["test"] if flip == "test" else np.concatenate([sp["val"], sp["test"]])
            g.y[ev] = 1 - g.y[ev]
        torch.manual_seed(0)
        m = SEGCN(num_dim=5, cat_dim=1, hidden=16, n_inner=0)
        r = fit_global(m, g, adj, sp, TrainConfig(epochs=5, select=select, seed=0))
        outs[(select, flip)] = r.prob
    assert np.array_equal(outs[("best_val", None)], outs[("best_val", "test")])
    assert np.array_equal(outs[("last_epoch", None)], outs[("last_epoch", "valtest")])
    # and the labels do matter where they should: flipping TRAIN labels changes the model
    g = to_device(F, "cpu", edges=False)
    g.y[sp["train"]] = 1 - g.y[sp["train"]]
    torch.manual_seed(0)
    r = fit_global(SEGCN(num_dim=5, cat_dim=1, hidden=16, n_inner=0), g, adj, sp,
                   TrainConfig(epochs=5, select="last_epoch", seed=0))
    assert not np.allclose(r.prob, outs[("last_epoch", None)])


def test_end_to_end_extscore_and_gcn0(cr15_text, data_root):
    from bb.harness import load_results
    from bb.extscore import make_synthetic_extscore
    from bb.methods import segcn
    args = ["--dataset", "cresci15", "--device", "cpu", "--epochs", "2", "--seeds", "0", "--hidden", "16",
            "--batch-size", "8"]
    segcn.main(args)
    make_synthetic_extscore("cresci15", tag="syn", coverage=0.8)
    segcn.main(args + ["--with-extscore", "syn"])
    for inj in ("late", "root", "branch"):
        segcn.main(args + ["--with-extscore", "syn", "--extscore-inject", inj])
    segcn.main(args + ["--inner", "0"])
    base = load_results("segcn", "cresci15", where={"with_extscore": None, "inner": 1})
    extscore = load_results("segcn", "cresci15", where={"with_extscore": "syn", "extscore_inject": "num"})
    late = load_results("segcn", "cresci15", where={"with_extscore": "syn", "extscore_inject": "late"})
    root = load_results("segcn", "cresci15", where={"with_extscore": "syn", "extscore_inject": "root"})
    br = load_results("segcn", "cresci15", where={"with_extscore": "syn", "extscore_inject": "branch"})
    g0 = load_results("segcn", "cresci15", where={"inner": 0})
    assert len(base) == len(extscore) == len(late) == len(root) == len(br) == len(g0) == 1
    q = 4
    assert extscore[0]["n_params"] == base[0]["n_params"] + 2 * q
    assert late[0]["n_params"] == base[0]["n_params"] + 2 * 16
    assert root[0]["n_params"] == base[0]["n_params"] + 2 * 2 * 16          # both layers' out_encoder
    assert br[0]["n_params"] > base[0]["n_params"]
    assert base[0]["extscore"] == {"extscore": "none"} and extscore[0]["extscore"]["tag"] == "syn"
    assert extscore[0]["extscore"]["coverage"]["test"]["coverage"] < 1
    for r in base + extscore + g0:
        assert set(r["metrics"]) >= {"val", "test"} and r["status"] == "ok"
    assert base[0]["store"]["s_bar"] >= 1
