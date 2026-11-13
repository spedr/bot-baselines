"""SEBot: sparse coding-tree builder vs the ported official PartitionTree, ego subgraphs, model,
no-leak and end-to-end (synthetic, CPU)."""
from __future__ import annotations

import numpy as np
import pytest

from bb.methods.sebot.tree import build_tree
from bb.methods.sebot.tree_ref import PartitionTree, level_partitions, reference_tree, structural_entropy


def _graph(n, p, rng, *, iso=0.0, hubs=False, weighted=False):
    A = np.triu((rng.random((n, n)) < p).astype(float), 1)
    if hubs:
        for h in rng.choice(n, max(1, n // 15), replace=False):
            A[h, rng.choice(n, min(n - 1, 8), replace=False)] = 1
        A = np.triu((A + A.T) > 0, 1).astype(float)
    if weighted:
        A = A * rng.uniform(0.5, 3.0, A.shape)
    A = A + A.T
    np.fill_diagonal(A, 0)
    dead = rng.random(n) < iso
    A[dead] = 0
    A[:, dead] = 0
    return A


def _csr(A):
    nz = [np.flatnonzero(r) for r in A]
    indptr = np.concatenate([[0], np.cumsum([len(x) for x in nz])]).astype(np.int64)
    ind = np.concatenate(nz).astype(np.int64) if len(nz) else np.zeros(0, np.int64)
    w = np.concatenate([A[i, x] for i, x in enumerate(nz)])
    return indptr, ind, w


def _sparse_parent(tr):
    return {i: int(tr.parent[i]) for i in range(len(tr.parent)) if tr.parent[i] != -2}


@pytest.mark.parametrize("seed", range(6))
def test_sparse_tree_equals_reference_exactly(seed):
    """Same node ids, same parents as PartitionTree(ordered=True) for k = 2..8, both variants,
    graphs with isolated nodes, hubs (heap ties) and several components."""
    rng = np.random.default_rng(seed)
    checked = 0
    for it in range(6):
        n = int(rng.integers(5, 90))
        A = _graph(n, float(rng.uniform(0.02, 0.15)), rng, iso=0.15 if it % 2 else 0.0, hubs=it % 3 == 0)
        if A.sum() == 0:
            continue
        ip, ind, _ = _csr(A)
        for smooth in (True, False):
            if not smooth and (A.sum(1) == 0).any():
                continue          # the subgraph variant has no +1: vol 0 is undefined upstream too
            for k in range(2, 9):
                par, root, _ = reference_tree(A, k, smooth=smooth, ordered=True)
                tr = build_tree(ip, ind, k, smooth=smooth)
                assert _sparse_parent(tr) == par and tr.root == root, (seed, it, smooth, k)
                checked += 1
    assert checked > 20


def test_sparse_tree_large_graph_and_structure():
    """n = 2,000 sparse graph with isolated nodes and many components: identical tree; every leaf
    at depth k; quotient graphs symmetric without self-loops."""
    rng = np.random.default_rng(11)
    n = 2000
    edges = set()
    for v in range(1, n):
        if rng.random() < 0.15:
            continue
        u = int(rng.integers(0, v)) if rng.random() < 0.7 else int(rng.integers(0, max(1, v // 50)))
        edges.add((u, v))
    A = np.zeros((n, n))
    for u, v in edges:
        A[u, v] = A[v, u] = 1
    ip, ind, _ = _csr(A)
    for k in (3, 6):
        par, root, pt = reference_tree(A, k, smooth=True, ordered=True)
        tr = build_tree(ip, ind, k, smooth=True)
        assert _sparse_parent(tr) == par and tr.root == root
        lv = tr.levels()
        assert len(lv) == k + 1 and len(lv[-1]) == 1
        ld = tr.layer_data(ip, ind)
        for d in range(1, k + 1):
            assert ld["parent_local"][d].shape == (ld["node_size"][d - 1],)
            assert ld["parent_local"][d].max() == ld["node_size"][d] - 1
        for d in range(1, k):
            e = ld["edges"][d]
            assert (e[0] != e[1]).all()
            s = set(map(tuple, e.T))
            assert s == {(b, a) for a, b in s}
        H = structural_entropy(par, n, A, smooth=True)
        assert H > 0


def test_weighted_graphs_match_upstream_order():
    """On weighted graphs (no heap ties) the sparse tree has the same level partitions as the
    verbatim upstream algorithm (ordered=False), and the same structural entropy."""
    rng = np.random.default_rng(3)
    for it in range(8):
        n = int(rng.integers(6, 50))
        A = _graph(n, 0.12, rng, weighted=True, hubs=it % 2 == 0)
        if A.sum() == 0:
            continue
        ip, ind, w = _csr(A)
        for k in (2, 4, 6):
            par, _, _ = reference_tree(A, k, smooth=True, ordered=False)
            tr = build_tree(ip, ind, k, weights=w, smooth=True)
            sp = _sparse_parent(tr)
            assert level_partitions(sp, n) == level_partitions(par, n)
            assert structural_entropy(sp, n, A) == pytest.approx(structural_entropy(par, n, A), abs=1e-12)


def test_unweighted_ties_change_ids_not_entropy():
    """Review round 1: on unweighted graphs with many heap ties (hubs with degree-1 followers) the
    sparse tree may differ from upstream's hash-order tie-breaking (ordered=False) in node ids, but
    every leaf sits at depth k and the structural entropy is the same."""
    rng = np.random.default_rng(101)
    differ = 0
    for it in range(6):
        n = int(rng.integers(20, 70))
        A = np.zeros((n, n))
        hubs = rng.choice(n, 3, replace=False)
        for v in range(n):
            if v not in hubs and rng.random() < 0.9:
                h = int(rng.choice(hubs))
                A[h, v] = A[v, h] = 1
        if A.sum() == 0:
            continue
        ip, ind, _ = _csr(A)
        for smooth in (True, False):
            if not smooth and (A.sum(1) == 0).any():
                continue
            for k in (3, 6):
                up, _, _ = reference_tree(A, k, smooth=smooth, ordered=False)
                sp = _sparse_parent(build_tree(ip, ind, k, smooth=smooth))
                assert len(level_partitions(sp, n)) == k                 # levels 1..k, level k = root
                differ += sp != up
                assert structural_entropy(sp, n, A, smooth) == pytest.approx(
                    structural_entropy(up, n, A, smooth), rel=1e-12, abs=1e-12)
    assert differ > 0          # the ties really occur (otherwise this test checks nothing new)


def test_reference_port_formulas():
    """Spot checks of the two upstream variants: +1 smoothing (SEPN) and plain (SEPG)."""
    A = np.array([[0, 1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0, 1, 1], [0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 1, 0]], float)
    for smooth in (True, False):
        pt = PartitionTree(A, smooth=smooth, ordered=True)
        pt.build_coding_tree(2)
        par = {i: (-1 if v.parent is None else v.parent) for i, v in pt.tree_node.items()}
        lp = level_partitions(par, 7)
        assert lp[0] == frozenset({frozenset({0, 1, 2}), frozenset({3, 4, 5, 6})}) or len(lp[0]) == 2
        ip, ind, _ = _csr(A)
        assert _sparse_parent(build_tree(ip, ind, 2, smooth=smooth)) == par


# ------------------------------------------------------------------------------------------------
# ego subgraphs


def test_ego_subgraphs_faithful_and_undirected(data_root):
    from bb.methods.sebot.artifacts import build_subgraph_trees
    # 0 <- 1, 0 <- 2, 1 -> 2 (edge among in-neighbours), 3 has no in-edges (lone), 4 -> 3? no
    src = np.array([1, 2, 1, 5, 6, 0])
    dst = np.array([0, 0, 2, 4, 4, 4])
    ei = np.stack([src, dst])
    lab = np.array([0, 3, 4, 2] + list(range(7, 19)))
    n = 20
    S = build_subgraph_trees("cresci15", ei, lab, n, 3, seed=0, verbose=False)
    m = S.meta
    assert m["n_graphs"] == 16 and m["lone"] == 13                    # 3 and 7..18 have no in-neighbours
    b0 = S.batch[0]
    nl = [S.nodelist[b0 == q] for q in range(16)]
    assert list(nl[0]) == [0, 1, 2]                                   # sorted subset {0} u in(0)
    assert list(nl[2]) == [0, 4, 5, 6]
    assert list(nl[3]) == [1, 2]
    # lone node: itself + a star of top_n = 10 distinct random labelled users
    assert nl[1][0] == 3 and len(nl[1]) == 11 and len(set(nl[1])) == 11 and set(nl[1][1:]) <= set(lab) - {3}
    # every graph reaches a single root at level k
    assert np.array_equal(np.bincount(S.batch[3], minlength=16), np.ones(16))
    # level-0 edges of graph 0: 0-1, 0-2, 1-2 (both directions)
    e0 = S.edges[0]
    g0 = e0[:, (b0[e0[0]] == 0)]
    assert {tuple(x) for x in g0.T} == {(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)}
    # deterministic (cached file reused, and rebuild gives the same arrays)
    S2 = build_subgraph_trees("cresci15", ei, lab, n, 3, seed=0, rebuild=True, verbose=False)
    assert np.array_equal(S.nodelist, S2.nodelist)
    U = build_subgraph_trees("cresci15", ei, lab, n, 3, seed=0, mode="undirected", verbose=False)
    assert U.meta["lone"] == 13 and len(U.nodelist) < len(S.nodelist)   # undirected: 3 and 7..18 are isolated


def test_cap_out_edges():
    from bb.methods.sebot.artifacts import cap_out_edges
    src = np.array([0] * 15 + [1] * 3 + [2])
    dst = np.array(list(range(1, 16)) + [2, 3, 1] + [2])
    s, t = cap_out_edges(src, dst, 20, 10, np.random.default_rng(0))
    assert (s == 0).sum() == 10 and (s == 1).sum() == 2 and (s == 2).sum() == 0     # self-loops 1->1, 2->2 removed


# ------------------------------------------------------------------------------------------------
# model


def test_info_nce_chunked_equals_official():
    import torch
    from bb.methods.sebot.model import info_nce
    torch.manual_seed(0)
    a, b = torch.randn(37, 8, requires_grad=True), torch.randn(37, 8, requires_grad=True)

    def official(ei, ej, t=0.1):
        bs = ei.shape[0]
        neg = (~torch.eye(bs * 2, dtype=bool)).float()
        z = torch.cat([torch.nn.functional.normalize(ei, dim=1), torch.nn.functional.normalize(ej, dim=1)])
        sim = z @ z.t()
        pos = torch.cat([torch.diag(sim, bs), torch.diag(sim, -bs)])
        return torch.sum(-torch.log(torch.exp(pos / t) / torch.sum(neg * torch.exp(sim / t), dim=1))) / (2 * bs)

    ref = official(a, b)
    out = info_nce(a, b, 0.1, block=10)
    assert torch.allclose(ref, out, atol=1e-5)
    ga = torch.autograd.grad(ref, a)[0]
    gb = torch.autograd.grad(info_nce(a, b, 0.1, block=10), a)[0]
    assert torch.allclose(ga, gb, atol=1e-5)
    # hand-written backward: float64, several block sizes (incl. ragged last block, block > 2n) and
    # temperatures, both inputs; plus torch's numerical gradcheck
    for n, blk, t in [(37, 5, 0.1), (50, 33, 0.05), (40, 1000, 0.01)]:
        a64 = torch.randn(n, 8, dtype=torch.float64, requires_grad=True)
        b64 = torch.randn(n, 8, dtype=torch.float64, requires_grad=True)
        r = official(a64, b64, t)
        v = info_nce(a64, b64, t, block=blk)
        assert abs(float(r - v)) < 1e-12 * abs(float(r))
        for x, y in zip(torch.autograd.grad(r, (a64, b64)), torch.autograd.grad(v, (a64, b64))):
            assert torch.allclose(x, y, rtol=1e-10, atol=1e-12)
    a64 = torch.randn(9, 4, dtype=torch.float64, requires_grad=True)
    b64 = torch.randn(9, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda p, q: info_nce(p, q, 0.1, 4), (a64, b64))


def test_facn_eval_noise_flag():
    import torch
    from bb.methods.sebot.model import FACNConv
    torch.manual_seed(0)
    x = torch.randn(10, 32)
    rel = [(torch.tensor([0, 1, 2]), torch.tensor([3, 3, 4])), (torch.tensor([5, 6]), torch.tensor([7, 8]))]
    c = FACNConv(32, 32).eval()
    assert not torch.allclose(c(x, rel), c(x, rel))       # official: noise also at eval
    c.eval_noise = False
    assert torch.allclose(c(x, rel), c(x, rel))


# ------------------------------------------------------------------------------------------------
# end to end on the Cresci fixture


def _fake_text(dataset, signal=1.0, seed=0):
    from bb.data import load_labels
    from bb.io import save_npy, write_json
    from bb.paths import dataset_paths
    y = load_labels(dataset).astype(np.float32)
    rng = np.random.default_rng(seed)
    for name in ("des", "tweets_first20_mean"):
        x = rng.standard_normal((len(y), 768)).astype(np.float32)
        x[:, :4] += signal * (2 * y[:, None] - 1)
        d = dataset_paths(dataset).emb("roberta-base")
        save_npy(d / f"{name}.f16.npy", x.astype(np.float16))
        write_json(d / f"{name}.meta.json", {"rows": len(y), "hf_commit": "fake"})


ARGS = ["--dataset", "cresci15", "--seeds", "0", "--epochs", "5", "--device", "cpu", "--k", "3"]


def _trajectory(argv, flip=None):
    """Train with optionally flipped val/test labels; return per-epoch history and the selection."""
    import torch
    from bb.methods.sebot import parse_args
    from bb.methods.sebot.train import build_model, prepare, train_epochs
    from bb.harness import set_seed
    a = parse_args(argv)
    P = prepare(a)
    if flip:
        for s in flip:
            p = torch.as_tensor(P.pos[s])
            P.y_lab[p] = 1 - P.y_lab[p]
    set_seed(0)
    m = build_model(a, P)
    hist, best, _ = train_epochs(a, P, m, a.epochs)
    return hist, best, P


def test_no_label_leak(cr15, data_root):
    """Flipping every test label changes nothing; flipping val labels leaves the training
    trajectory (losses, per-epoch eval probabilities) unchanged and only moves the val-based
    selection."""
    _fake_text("cresci15")
    h0, b0, P = _trajectory(ARGS)
    h1, b1, _ = _trajectory(ARGS, flip=("test",))
    assert [r["loss"] for r in h0] == [r["loss"] for r in h1]
    assert b0["epoch"] == b1["epoch"] and np.array_equal(b0["prob"], b1["prob"])
    h2, b2, _ = _trajectory(ARGS, flip=("val", "test"))
    assert [r["loss"] for r in h0] == [r["loss"] for r in h2]
    assert [r["nce_u"] for r in h0] == [r["nce_u"] for r in h2]


def test_end_to_end_extscore_and_ablations(cr15, data_root):
    from bb.harness import load_results
    from bb.extscore import make_synthetic_extscore
    from bb.methods import sebot
    _fake_text("cresci15")
    base = sebot.main(ARGS)
    again = sebot.main(ARGS)
    assert base[0]["metrics"]["test"] == again[0]["metrics"]["test"]      # seeded runs reproduce
    make_synthetic_extscore("cresci15", tag="syn", coverage=0.8)
    extscore = sebot.main(ARGS + ["--with-extscore", "syn"])
    late = sebot.main(ARGS + ["--with-extscore", "syn", "--extscore-inject", "late"])
    gam = sebot.main(ARGS + ["--with-extscore", "syn", "--extscore-inject", "gamma"])
    nb = base[0]["n_params"]
    q = 32
    assert extscore[0]["n_params"] == nb + 3 * 2 * q          # num 5 -> 7 in the three encoders
    assert late[0]["n_params"] == nb + 2 * 32            # classifier 96 -> 98
    assert gam[0]["n_params"] == nb + 2 * q              # backbone encoder only
    assert base[0]["extscore"] == {"extscore": "none"} and extscore[0]["extscore"]["tag"] == "syn"
    abl = sebot.main(ARGS + ["--no-alpha", "--no-beta", "--no-rcm", "--no-eval-noise"])
    assert abl[0]["n_params"] < nb
    rg = sebot.main(ARGS + ["--backbone", "rgcn", "--subgraph-mode", "undirected"])
    assert rg[0]["status"] == "ok"
    res = load_results("sebot", "cresci15", where={"with_extscore": None})
    assert len(res) >= 2
    r = base[0]
    assert r["selection"]["select"] == "min_val_loss" and 0 <= r["selection"]["epoch"] < 5
    assert {"train", "val", "test"} <= set(r["metrics"]) and r["predictions"]["test"] == "preds_test.npz"
    assert r["artifacts"]["tree"]["k"] == 3 and r["artifacts"]["subgraphs"]["n_graphs"] == 28


def test_rgcn_ablation_uses_per_relation_mean():
    """The 'R-GCN encoder' ablation must be R-GCN (mean per relation), also at sizes where PyG's
    heuristic would pick segment_matmul: compare with a hand-written per-relation mean."""
    import torch
    import torch_geometric
    from bb.methods.sebot.model import Backbone
    torch.manual_seed(0)
    b = Backbone(5, 3, conv="rgcn", pe=0.0).eval()
    wrapper = b.conv1
    conv = getattr(wrapper, "conv", wrapper)               # bb.gnn.RGCNConv holds the PyG layer as .conv
    n, e = 3000, 40000
    x = torch.randn(n, 32)
    ei = torch.randint(0, n, (2, e))
    et = torch.randint(0, 2, (e,))
    old = torch_geometric.backend.use_segment_matmul
    try:
        torch_geometric.backend.use_segment_matmul = True      # force the heuristic's other branch
        out = wrapper(x, ei, et)
        raw = conv(x, ei, et)                                  # raw PyG under the forced flag
    finally:
        torch_geometric.backend.use_segment_matmul = old
    ref = x @ conv.root + conv.bias
    for r in range(2):
        m = et == r
        src, dst = ei[0, m], ei[1, m]
        msg = (x @ conv.weight[r])[src]
        s_ = torch.zeros(n, 32).index_add_(0, dst, msg)
        c = torch.zeros(n).index_add_(0, dst, torch.ones(len(dst))).clamp(min=1)
        ref = ref + s_ / c[:, None]
    assert torch.allclose(out, ref, atol=1e-4)
    if torch_geometric.typing.WITH_SEGMM:
        assert not torch.allclose(raw, ref, atol=1e-3)         # the pitfall the wrapper avoids


def test_encoder_checkpointing_is_exact():
    import torch
    from bb.methods.sebot.model import FeatureEncoder
    torch.manual_seed(0)
    enc = FeatureEncoder(5, 3)
    ins = [torch.randn(50, 768), torch.randn(50, 768), torch.randn(50, 5), torch.randn(50, 3)]
    outs = []
    for ck in (False, True):
        enc.zero_grad()
        enc.ckpt = ck
        y = enc(*ins)
        (y ** 2).sum().backward()
        outs.append((y.detach(), [p.grad.clone() for p in enc.parameters()]))
    assert torch.allclose(outs[0][0], outs[1][0])
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(outs[0][1], outs[1][1]))
