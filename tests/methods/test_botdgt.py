"""BotDGT: snapshot construction (dates, births, LCC/BLR vs networkx), prefix-view sampler, stacked
structural layer, loss, no-leak, ExtScore plumbing and end to end (synthetic fixtures, CPU)."""
from __future__ import annotations

import datetime as dt

import networkx as nx
import numpy as np
import pytest
import torch

from bb.methods.botdgt import snapshots as S


# ------------------------------------------------------------------------------- helpers
class _CSR:
    def __init__(self, indptr, indices):
        self.indptr, self.indices = np.asarray(indptr, np.int64), np.asarray(indices, np.int64)

    @property
    def n(self):
        return len(self.indptr) - 1

    @property
    def nnz(self):
        return int(self.indptr[-1])

    def degree(self):
        return np.diff(self.indptr)


def _csr(rows, cols, n):
    o = np.lexsort((np.arange(len(rows)), rows))
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(rows, minlength=n), out=indptr[1:])
    return _CSR(indptr, np.asarray(cols)[o])


def _random_graph(n=5000, m=20000, T=6, seed=0):
    rng = np.random.default_rng(seed)
    # a dense core so that triangles exist, plus a sparse rest; some duplicates and self-loops
    src = np.concatenate([rng.integers(0, 60, m // 4), rng.integers(0, n, m - m // 4)])
    dst = np.concatenate([rng.integers(0, 60, m // 4), rng.integers(0, n, m - m // 4)])
    rel = rng.integers(0, 2, m)
    birth = rng.integers(0, T, n).astype(np.int64)
    birth[rng.integers(0, n, 10)] = T - 1
    return src, dst, rel, birth


def _mem_snapshots(src, dst, birth, T, lcc=None, blr=None):
    """In-memory ``Snapshots`` over the in-neighbour CSC of (src -> dst)."""
    n = len(birth)
    csc = _csr(dst, src, n)
    perm, cnt, _ = S.sort_csc_by_birth(csc, birth, T)
    z = np.zeros((T, n), np.float32)
    return S.Snapshots("mem", {"T": T, "dates": [f"d{t}" for t in range(T)]}, birth.astype(np.int8), csc.indptr,
                       np.ascontiguousarray(csc.indices[perm]), cnt, z if lcc is None else lcc,
                       z if blr is None else blr, None)


# ------------------------------------------------------------------------------- snapshots
def test_snapshot_dates_and_births_follow_official_rule():
    days = lambda *ds: np.array([(d - S.EPOCH).days for d in ds], np.int32)  # noqa: E731
    d20 = days(dt.date(2006, 3, 1), dt.date(2020, 9, 5), dt.date(2008, 9, 30), dt.date(2008, 10, 1))
    dates, n_months = S.snapshot_dates(d20)
    assert n_months == 154 and len(dates) == 13
    assert dates[0] == dt.date(2008, 10, 1) and dates[-1] == dt.date(2020, 10, 1)
    b = S.node_birth(d20, dates)
    assert b.tolist() == [0, 12, 0, 1]                    # strict '<': created on D_t exists from t+1
    d22 = days(dt.date(1970, 1, 1), dt.date(2022, 3, 14))
    dates22, n22 = S.snapshot_dates(d22)
    assert n22 == 172 and len(dates22) == 15 and dates22[0] == dt.date(2008, 4, 1) and dates22[-1] == dt.date(2022, 4, 1)
    dc, _ = S.snapshot_dates(days(dt.date(2013, 4, 30)))
    assert dc[0] == dt.date(2008, 5, 1) and dc[-1] == dt.date(2013, 5, 1) and len(dc) == 6
    assert S.snapshot_dates(days(dt.date(2007, 4, 6)))[0] == [dt.date(2008, 1, 1)]


def test_lcc_and_blr_match_networkx():
    T = 5
    src, dst, rel, birth = _random_graph(n=5000, m=20000, T=T)
    n = len(birth)
    # undirected simple union of both relations (as bb.data.load_undirected)
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    keep = u != v
    key = np.unique(u[keep] * n + v[keep])
    und = _csr(key // n, key % n, n)
    lcc = S.compute_lcc(und, birth, T)
    fol = rel == 1
    blr = S.compute_blr(src[fol], dst[fol], birth, T, n)
    assert lcc.max() > 0 and blr.max() > 0
    for t in range(T):
        ex = np.flatnonzero(birth <= t)
        G = nx.Graph()
        G.add_nodes_from(ex.tolist())
        D = nx.DiGraph()
        D.add_nodes_from(ex.tolist())
        for a, b_, r in zip(src.tolist(), dst.tolist(), rel.tolist()):
            if birth[a] <= t and birth[b_] <= t:
                G.add_edge(a, b_)
                if r == 1:
                    D.add_edge(a, b_)
        cl = nx.clustering(G)
        ref = np.zeros(n)
        ref[list(cl)] = list(cl.values())
        assert np.allclose(lcc[t], ref, atol=1e-6), t
        rb = np.zeros(n)
        for x in D.nodes:
            succ = set(D.successors(x))
            if succ:
                rb[x] = len(succ & set(D.predecessors(x))) / len(succ)
        assert np.allclose(blr[t], rb, atol=1e-6), t
    assert np.all(lcc[:, birth == T - 1][:-1] == 0)       # absent nodes are zero


def test_prefix_sampler_equals_explicit_snapshot_graph():
    from bb.harness.sampler import NeighborSampler
    from bb.methods.botdgt.sampler import SnapshotSampler
    T = 4
    src, dst, _, birth = _random_graph(n=400, m=3000, T=T, seed=1)
    snap = _mem_snapshots(src, dst, birth, T)
    sm = SnapshotSampler(snap, [-1, -1])
    seeds = np.array([3, 17, 60, 5, 250, 399, 1])
    b = sm.sample_batch(seeds)
    off = np.concatenate([[0], np.cumsum(b.n_t)])
    for t in range(T):
        m = (birth[src] <= t) & (birth[dst] <= t)
        ref = NeighborSampler(_csr(dst[m], src[m], len(birth)), [-1, -1]).sample(seeds)
        nodes_t = b.union[b.inv[off[t]:off[t + 1]]]
        # same nodes (seeds first; neighbour discovery order differs because prefix rows are birth-sorted)
        assert np.array_equal(nodes_t[:len(seeds)], seeds) and sorted(nodes_t.tolist()) == sorted(ref.n_id.tolist())
        mask = (b.ei[1] >= off[t]) & (b.ei[1] < off[t + 1])
        got = sorted(zip(nodes_t[b.ei[0, mask] - off[t]].tolist(), nodes_t[b.ei[1, mask] - off[t]].tolist()))
        exp = sorted(zip(ref.n_id[ref.edge_index[0]].tolist(), ref.n_id[ref.edge_index[1]].tolist()))
        assert got == exp
    # seed rows and layer-2 edges
    assert np.array_equal(b.union[b.inv[b.seed_rows]], np.tile(seeds, T))
    assert np.all(b.ei_seed[1] < T * len(seeds))
    # deterministic given the rng seed (faithful replay), capped fan-out respected
    capped = SnapshotSampler(snap, [2, 2])
    x1, x2 = capped.sample_batch(seeds, 5), capped.sample_batch(seeds, 5)
    assert np.array_equal(x1.ei, x2.ei) and np.array_equal(x1.union, x2.union)
    assert np.bincount(x1.ei[1]).max() <= 2


def test_stacked_structural_equals_per_snapshot_full_forward():
    from bb.harness.sampler import NeighborSampler
    from bb.methods.botdgt.model import Structural
    from bb.methods.botdgt.sampler import SnapshotSampler
    T, n, h = 3, 300, 32
    src, dst, _, birth = _random_graph(n=n, m=2500, T=T, seed=2)
    snap = _mem_snapshots(src, dst, birth, T)
    torch.manual_seed(0)
    st = Structural(h).eval()
    feat = torch.randn(n, h)
    seeds = np.array([0, 7, 9, 150, 299])
    b = SnapshotSampler(snap, [-1, -1]).sample_batch(seeds)
    t_ = lambda a: torch.from_numpy(np.ascontiguousarray(a))  # noqa: E731
    with torch.no_grad():
        s = st(feat[t_(b.union)][t_(b.inv)], t_(b.ei), t_(b.ei_seed), t_(b.seed_rows)).view(T, len(seeds), h)
        for t in range(T):
            m = (birth[src] <= t) & (birth[dst] <= t)
            full = st.forward_full(feat, torch.from_numpy(np.stack([src[m], dst[m]])))
            assert torch.allclose(s[t], full[seeds], atol=1e-5), t
            # and the 2-hop sampled subgraph gives the same seed outputs as the full graph
            r = NeighborSampler(_csr(dst[m], src[m], n), [-1, -1]).sample(seeds)
            sub = st.forward_full(feat[t_(r.n_id)], t_(r.edge_index))[: len(seeds)]
            assert torch.allclose(sub, full[seeds], atol=1e-5)


def test_nested_cumulative_path_equals_per_snapshot_full_forward():
    """One sample of the last snapshot + cumulative layer 1 == T separate graph-transformer passes."""
    from bb.methods.botdgt.model import Structural
    from bb.methods.botdgt.sampler import SnapshotSampler
    T, n, h = 4, 300, 32
    src, dst, _, birth = _random_graph(n=n, m=2500, T=T, seed=3)
    snap = _mem_snapshots(src, dst, birth, T)
    torch.manual_seed(0)
    st = Structural(h).eval()
    feat = torch.randn(n, h) * 3          # large logits: exercises the running-max merge
    seeds = np.array([0, 7, 9, 150, 299, 42])
    sm = SnapshotSampler(snap, [-1, -1])
    t_ = lambda a: torch.from_numpy(np.ascontiguousarray(a))  # noqa: E731
    for t0 in (0, 2):                     # full sequence and a window of the last two snapshots
        b = sm.nested(seeds, 0, t0)
        Tw = T - t0
        with torch.no_grad():
            # layer 1 computes rows n_id[:n1] only (seeds + hop-1 nodes); every edge targets one of them
            assert len(seeds) <= b.n1 <= len(b.n_id) and b.ei[1].max() < b.n1 and b.e2_src.max() < Tw * b.n1
            s = st.forward_nested(feat[t_(b.n_id)], t_(b.ei), t_(b.eb), Tw, t_(b.e2_src), t_(b.e2_dst),
                                  len(seeds), b.n1).view(Tw, len(seeds), h)
            for j in range(Tw):
                t = t0 + j
                m = (birth[src] <= t) & (birth[dst] <= t)
                full = st.forward_full(feat, torch.from_numpy(np.stack([src[m], dst[m]])))
                assert torch.allclose(s[j], full[seeds], atol=1e-4), (t0, t)
    # gradients flow through the cumulative path
    b = sm.nested(seeds, 0)
    x = feat[t_(b.n_id)].clone().requires_grad_(True)
    st.forward_nested(x, t_(b.ei), t_(b.eb), T, t_(b.e2_src), t_(b.e2_dst), len(seeds), b.n1).sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_snapshot_loss_weights_and_skips_empty_snapshots():
    from bb.methods.botdgt.model import snapshot_loss
    torch.manual_seed(0)
    B, T = 6, 4
    logits = torch.randn(B, T, 2)
    y = torch.randint(0, 2, (B,))
    exist = torch.ones(B, T, dtype=torch.bool)
    exist[:, 0] = False
    exist[:3, 1] = False
    ref = sum(1.1 ** t * torch.nn.functional.cross_entropy(logits[exist[:, t], t], y[exist[:, t]])
              for t in range(1, T))
    assert torch.allclose(snapshot_loss(logits, y, exist), ref)
    assert torch.allclose(snapshot_loss(logits, y, exist, last_only=True),
                          torch.nn.functional.cross_entropy(logits[:, -1], y))


def test_model_structure_shared_modules_and_frozen_pos_signals():
    from bb.methods.botdgt.model import BotDGT
    m = BotDGT(64, T=5, num_dim=5, cat_dim=3)
    prelus = [x for x in m.modules() if isinstance(x, torch.nn.PReLU)]
    assert len(prelus) == 3 and all(p.weight.numel() == 1 for p in prelus)   # embed, struct, FFN
    assert len([x for x in m.modules() if isinstance(x, torch.nn.LayerNorm)]) == 1
    g = m.param_groups(5e-4, 5e-5, 5e-2)
    in_groups = {id(p) for grp in g for p in grp["params"]}
    assert id(m.p_lcc.weight) not in in_groups and not m.p_lcc.weight.requires_grad
    assert {id(p) for p in m.parameters()} - in_groups == {id(p) for p in (*m.p_lcc.parameters(), *m.p_blr.parameters())}
    m2 = BotDGT(64, T=5, num_dim=5, cat_dim=3)
    g2 = m2.param_groups(5e-4, 5e-5, 5e-2, train_pos_signals=True)
    assert {id(p) for grp in g2 for p in grp["params"]} == {id(p) for p in m2.parameters()}
    assert [grp["lr"] for grp in g] == [5e-4, 5e-5]
    # causal: changing the last step's input never changes earlier logits
    t = m.temporal.eval()
    S_ = torch.randn(2, 5, 64)
    p = torch.zeros(2, 5, 64)
    a = t(S_, p)
    S2 = S_.clone()
    S2[:, -1] += torch.randn(2, 64)          # (a constant shift would vanish in the LayerNorm)
    b = t(S2, p)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6) and not torch.allclose(a[:, -1], b[:, -1])


def test_init_follows_official_init_loops():
    """Official init loops touch only ``torch.nn.Linear``: embedding and temporal FFN get kaiming_normal +
    zero bias; TransformerConv's PyG ``Linear``s keep PyG's default (|w| <= 1/sqrt(fan_in))."""
    from torch_geometric.nn.dense.linear import Linear as PygLinear
    from bb.methods.botdgt.model import BotDGT
    assert not issubclass(PygLinear, torch.nn.Linear)
    torch.manual_seed(0)
    m = BotDGT(128, T=5, num_dim=5, cat_dim=3)
    for tc in (m.struct.tc1, m.struct.tc2):
        for lin in (tc.lin_key, tc.lin_query, tc.lin_value, tc.lin_skip):
            assert lin.weight.abs().max() <= 1 / 128 ** 0.5 + 1e-7 and lin.bias.abs().sum() > 0
    for lin in (m.temporal.ff1, m.temporal.ff2, m.embed.num, m.embed.out):
        assert torch.all(lin.bias == 0) and lin.weight.std() > 1.1 / lin.weight.shape[1] ** 0.5  # sqrt(2/fan_in) vs default 0.58/sqrt(fan_in)


def test_full_model_matches_official_style_reference():
    """Whole forward (structural per explicit snapshot graph + official-style temporal layer:
    tensordot, heads split/cat on the batch dim, torch.where mask) and loss, against both sampling
    paths, for the full sequence and a window."""
    import torch.nn.functional as Fn
    from torch_geometric.nn import TransformerConv
    from bb.methods.botdgt.model import BotDGT, snapshot_loss
    from bb.methods.botdgt.sampler import SnapshotSampler
    from bb.methods.botdgt.train import to_device_batch
    T, n, h = 5, 400, 32
    src, dst, _, birth = _random_graph(n=n, m=3000, T=T, seed=4)
    rng = np.random.default_rng(4)
    lcc = (rng.random((T, n)) * (birth[None] <= np.arange(T)[:, None])).astype(np.float32)
    blr = (rng.random((T, n)) * (birth[None] <= np.arange(T)[:, None])).astype(np.float32)
    snap = _mem_snapshots(src, dst, birth, T, lcc, blr)

    class G:
        pass
    g = G()
    torch.manual_seed(0)
    g.num, g.cat, g.des, g.tweet = torch.randn(n, 5), torch.rand(n, 3), torch.randn(n, 768), torch.randn(n, 768)
    g.y, g.extra = torch.from_numpy(rng.integers(0, 2, n)), None
    seeds = np.array([0, 3, 9, 50, 51, 120, 399])
    sd = torch.from_numpy(seeds)
    for t0 in (0, 3):
        ts = list(range(t0, T))
        m = BotDGT(h, len(ts), 5, 3).eval()
        with torch.no_grad():
            x = m.embed(g.num, g.cat, g.des, g.tweet)
            tcs = [TransformerConv(h, h // 4, heads=4) for _ in range(2)]
            tcs[0].load_state_dict(m.struct.tc1.state_dict())
            tcs[1].load_state_dict(m.struct.tc2.state_dict())
            a = m.struct.act
            outs = []
            for t in ts:
                ok = (birth[src] <= t) & (birth[dst] <= t)
                ei = torch.from_numpy(np.stack([src[ok], dst[ok]]))
                outs.append(a(tcs[1](a(tcs[0](x, ei)), ei) + x)[sd])
            Sx = torch.stack(outs, 1)
            tl = m.temporal
            pe = (tl.pos(torch.arange(len(ts))).unsqueeze(0)
                  + torch.stack([m.p_lcc(torch.from_numpy(lcc[t]).view(-1, 1))[sd] for t in ts], 1)
                  + torch.stack([m.p_blr(torch.from_numpy(blr[t]).view(-1, 1))[sd] for t in ts], 1))
            inp = tl.norm(Sx + pe)
            q, k, v = (torch.cat(torch.split(torch.tensordot(inp, w, dims=([2], [0])), h // 4, dim=2), dim=0)
                       for w in (tl.wq, tl.wk, tl.wv))
            sc = q @ k.permute(0, 2, 1) / (h // 4) ** 0.5
            sc = torch.where(torch.tril(torch.ones_like(sc[0]))[None] == 0, torch.full_like(sc, -1e9), sc)
            o = torch.softmax(sc, 2) @ v
            o = tl.norm(torch.cat(torch.split(o, o.shape[0] // 4, dim=0), dim=2) + Sx)
            ref = tl.ff2(tl.ff_act(tl.ff1(o)))
            ref_loss = sum(1.1 ** j * Fn.cross_entropy(ref[:, j][torch.from_numpy(birth[seeds] <= t)],
                                                      g.y[sd][torch.from_numpy(birth[seeds] <= t)])
                           for j, t in enumerate(ts) if (birth[seeds] <= t).any())
            sm = SnapshotSampler(snap, [-1, -1])
            for b in (sm.nested(seeds, 0, t0), sm.sample_batch(seeds, 0, ts)):
                bd = to_device_batch(b, snap, slice(t0, T), "cpu")
                got = m(g, bd)
                assert torch.allclose(got, ref, atol=1e-4), (t0, (got - ref).abs().max())
                assert torch.allclose(snapshot_loss(got, g.y[bd["seeds"]], bd["exist"]), ref_loss, atol=1e-4)


# ------------------------------------------------------------------------------- real-format fixtures
def _spread_created_at(dataset: str):
    """Fixture users are all created in 2007; spread them over 2006-2013 so that several snapshots exist."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from bb.paths import dataset_paths
    p = dataset_paths(dataset).profile
    tb = pq.read_table(p)
    n = tb.num_rows
    years = 2006 + (np.arange(n) * 7) % 8
    ts = [dt.datetime(int(y), 1 + i % 12, 1 + i % 27, tzinfo=dt.timezone.utc) for i, y in enumerate(years)]
    i = tb.schema.get_field_index("created_at")
    tb = tb.set_column(i, tb.schema.field(i), pa.array(ts, type=tb.schema.field(i).type))
    pq.write_table(tb, p)


def _fake_text(dataset):
    from bb.data import load_labels
    from bb.io import save_npy, write_json
    from bb.paths import dataset_paths
    y = load_labels(dataset).astype(np.float32)
    rng = np.random.default_rng(0)
    for name in ("des", "tweets_first20_mean"):
        x = rng.standard_normal((len(y), 768)).astype(np.float32)
        x[:, :4] += 1.0 * (2 * y[:, None] - 1)
        d = dataset_paths(dataset).emb("roberta-base")
        save_npy(d / f"{name}.f16.npy", x.astype(np.float16))
        write_json(d / f"{name}.meta.json", {"rows": len(y), "hf_commit": "fake"})


@pytest.fixture
def cr15_snap(cr15):
    _fake_text("cresci15")
    _spread_created_at("cresci15")
    return cr15


def test_build_snapshots_fixture_monotone_and_idempotent(cr15_snap):
    from bb.data import load_in_merged
    m = S.build_snapshots("cresci15")
    assert m["T"] > 1 and m["nodes_per_snapshot"][-1] == 28
    assert m["nodes_per_snapshot"] == sorted(m["nodes_per_snapshot"])
    assert m["directed_edges_per_snapshot"] == sorted(m["directed_edges_per_snapshot"])
    snap = S.load_snapshots("cresci15")
    csc = load_in_merged("cresci15", m["relations"])
    assert snap.row_cnt[-1].tolist() == csc.degree().tolist()
    # every prefix entry exists at t; rows are permutations of the shared CSC rows
    for v in range(snap.n):
        a, b = snap.indptr[v], snap.indptr[v + 1]
        assert sorted(snap.indices[a:b].tolist()) == sorted(np.asarray(csc.indices[a:b]).tolist())
        for t in range(snap.T):
            pre = snap.indices[a:a + snap.row_cnt[t, v]]
            assert np.all(np.maximum(snap.birth[pre], snap.birth[v]) <= t)
    mt = (snap.dir / "lcc.f32.npy").stat().st_mtime_ns
    assert S.build_snapshots("cresci15")["input_key"] == m["input_key"]
    assert (snap.dir / "lcc.f32.npy").stat().st_mtime_ns == mt


def test_no_label_leak_flip_val_test(cr15_snap):
    """Flipping every val/test label must leave the trained model's test predictions unchanged."""
    from bb.features import load_features
    from bb.gnn import to_device
    from bb.methods.botdgt.model import BotDGT
    from bb.methods.botdgt.train import DGTConfig, fit
    F = load_features("cresci15")
    snap = S.load_snapshots("cresci15")
    sp = {s: F.splits[s] for s in ("train", "val", "test")}
    outs = []
    for flip in (False, True):
        g = to_device(F, "cpu", edges=False)
        if flip:
            ev = np.concatenate([sp["val"], sp["test"]])
            g.y[ev] = 1 - g.y[ev]
        torch.manual_seed(0)
        m = BotDGT(32, snap.T, 5, 1, temporal_dropout=0.0)
        r = fit(m, g, snap, sp, DGTConfig(epochs=3, batch_size=4, lr_struct=1e-2, lr_temporal=1e-2, seed=0,
                                          fanouts=(2, 2)))
        outs.append(r.prob[sp["test"]])
    assert np.allclose(outs[0], outs[1], atol=1e-6)


def test_batch_cache_replay_equals_resampling(cr15_snap):
    """Replaying the packed epoch-0 batches (default) == re-sampling the identical batches every epoch,
    including the cached per-epoch validation; the n1-trimmed layer 1 == all rows; bucketed padding
    changes neither the logits nor the gradients."""
    from bb.features import load_features
    from bb.gnn import to_device
    from bb.methods.botdgt.model import BotDGT
    from bb.methods.botdgt.sampler import SnapshotSampler
    from bb.methods.botdgt.train import DGTConfig, fit, pack_batch, to_device_batch, unpack
    F = load_features("cresci15")
    snap = S.load_snapshots("cresci15")
    sp = {s: F.splits[s] for s in ("train", "val", "test")}
    g = to_device(F, "cpu", edges=False)
    outs = []
    for cache, host_gb in ((True, 1.0), (False, 1.0), (True, 3e-6)):     # last: budget overflows
        torch.manual_seed(0)
        m = BotDGT(32, snap.T, 5, 1)
        r = fit(m, g, snap, sp, DGTConfig(epochs=3, batch_size=16, lr_struct=1e-2, lr_temporal=1e-2, seed=0,
                                          cache_batches=cache, cache_host_gb=host_gb, eval_every=1))
        outs.append((r.prob, [h["val_acc"] for h in r.history], [h["train_loss"] for h in r.history]))
        if cache:
            info = r.batch_stats["cache"]
            assert (info["not_stored"] == 0) == (host_gb == 1.0) and info["entries"] > info["not_stored"]
    for o in outs[1:]:
        assert np.array_equal(outs[0][0], o[0], equal_nan=True)
        assert outs[0][1] == o[1] and outs[0][2] == o[2]
    ps = []
    for cache in (True, False):                                        # stacked (official-style) batches
        torch.manual_seed(0)
        m = BotDGT(32, snap.T, 5, 1)
        ps.append(fit(m, g, snap, sp, DGTConfig(epochs=2, batch_size=16, lr_struct=1e-2, lr_temporal=1e-2, seed=0,
                                                cache_batches=cache, sampling="per_snapshot")).prob)
    assert np.array_equal(ps[0], ps[1], equal_nan=True)
    # n1 trimming: same logits as computing layer 1 for every sampled row (old e2_src = t * n + src)
    b = SnapshotSampler(snap, [-1, -1]).nested(np.asarray(sp["test"][:50]), 0)
    assert b.n1 < len(b.n_id)
    bd = to_device_batch(b, snap, slice(None), "cpu")
    with torch.no_grad():
        m.eval()
        trimmed = m(g, bd)
        n, n1 = len(b.n_id), b.n1
        bd_full = {**bd, "e2_src": (bd["e2_src"] // n1) * n + bd["e2_src"] % n1}
        del bd_full["n1"]
        assert torch.allclose(trimmed, m(g, bd_full), atol=1e-6)
        # bucketed padding (CUDA-graph shapes): same logits, and zero gradient leaks from the padding
        pb = pack_batch(b, snap, slice(None), "cpu", pad=True)
        assert pb.key is not None and pb.shapes[2][1] > b.ei.shape[1]
        assert torch.allclose(trimmed, m(g, unpack(pb)), atol=1e-5)
    m.train()
    m.temporal.drop.p = 0.0
    grads = []
    for bd_ in (bd, unpack(pb)):
        m.zero_grad(set_to_none=True)
        m(g, bd_)[:, -1, 1].sum().backward()
        grads.append([p.grad.clone() for p in m.parameters() if p.grad is not None])
    assert all(torch.allclose(x, y, atol=1e-5, rtol=1e-4) for x, y in zip(*grads))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_padding_and_cuda_graphs_match_eager(cr15_snap):
    """Padded batches (bucketed shapes) and CUDA-graph replay give the same step as the unpadded eager
    path: per step to float rounding (same dropout masks), end to end to float noise."""
    from bb.features import load_features
    from bb.gnn import to_device
    from bb.methods.botdgt.model import BotDGT, snapshot_loss
    from bb.methods.botdgt.sampler import SnapshotSampler
    from bb.methods.botdgt.train import DGTConfig, GraphedStep, fit, pack_batch, unpack
    F = load_features("cresci15")
    snap = S.load_snapshots("cresci15")
    sp = {s: F.splits[s] for s in ("train", "val", "test")}
    g = to_device(F, "cuda", edges=False)
    torch.manual_seed(0)
    m = BotDGT(32, snap.T, 5, 1).cuda()
    m.param_groups(1e-3, 1e-3, 0.0)
    b = SnapshotSampler(snap, [2560, 2560]).nested(np.asarray(sp["train"][:40]), 0)

    def step(bd):
        for p in m.parameters():
            p.grad = None
        torch.cuda.manual_seed(7)
        loss = snapshot_loss(m(g, bd), g.y[bd["seeds"]], bd["exist"])
        loss.backward()
        return loss.detach(), [p.grad.clone() for p in m.parameters() if p.grad is not None]

    ref = step(unpack(pack_batch(b, snap, slice(None), "cuda")))
    pp = pack_batch(b, snap, slice(None), "cuda", torch.int32, pad=True)
    assert pp.key is not None and pp.ints.numel() > pack_batch(b, snap, slice(None), "cpu").ints.numel()
    pad = step(unpack(pp))
    gs = GraphedStep(m, g, 1.1)
    for i in range(3):                    # first (eager) / capture (side-stream eager) / replay
        torch.cuda.manual_seed(7)
        loss = gs(pp)
        grads = [p.grad.clone() for p in m.parameters() if p.grad is not None]
        for got in ((loss, grads), pad):
            assert torch.allclose(got[0], ref[0], rtol=1e-5, atol=1e-6)
            assert all(torch.allclose(x, y, rtol=1e-4, atol=1e-6) for x, y in zip(got[1], ref[1]))
    assert len(gs.graphs) == 1
    # end to end (3 epochs, dropout on): graphs vs eager
    outs = []
    for graphs in (False, True):
        torch.manual_seed(0)
        mm = BotDGT(32, snap.T, 5, 1).cuda()
        r = fit(mm, g, snap, sp, DGTConfig(epochs=3, batch_size=16, lr_struct=1e-3, lr_temporal=1e-3, seed=0,
                                           cuda_graphs=graphs, eval_every=0))
        outs.append(r)
    assert outs[1].batch_stats["cuda_graphs"]["graphs"] > 0
    assert np.allclose(outs[0].prob[sp["test"]], outs[1].prob[sp["test"]], atol=1e-3)


def test_predict_batch_size_invariant(cr15_snap):
    from bb.features import load_features
    from bb.gnn import to_device
    from bb.methods.botdgt.model import BotDGT
    from bb.methods.botdgt.train import predict
    F = load_features("cresci15")
    snap = S.load_snapshots("cresci15")
    g = to_device(F, "cpu", edges=False)
    torch.manual_seed(0)
    m = BotDGT(32, snap.T, 5, 1)
    idx = np.arange(F.n)
    a = predict(m, g, snap, idx, slice(None), batch_size=3)
    b = predict(m, g, snap, idx, slice(None), batch_size=1000)
    c = predict(m, g, snap, idx, slice(None), batch_size=1000, max_union=10)    # forced micro-batches
    assert np.allclose(a, b, atol=1e-5) and np.allclose(a, c, atol=1e-5)


def test_end_to_end_seeded_extscore_and_window(cr15_snap, data_root):
    from bb.harness import load_results
    from bb.extscore import make_synthetic_extscore
    from bb.methods import botdgt
    args = ["--dataset", "cresci15", "--seeds", "0", "--epochs", "2", "--device", "cpu", "--batch-size", "8",
            "--hidden", "32"]
    botdgt.main(args)
    botdgt.main(args)
    make_synthetic_extscore("cresci15", tag="syn", coverage=0.8)
    botdgt.main(args + ["--with-extscore", "syn"])
    botdgt.main(args + ["--with-extscore", "syn", "--extscore-inject", "late"])
    botdgt.main(args + ["--with-extscore", "syn", "--extscore-inject", "token"])
    botdgt.main(args + ["--window", "1"])
    base = load_results("botdgt", "cresci15", where={"with_extscore": None, "window": -1})
    extscore = load_results("botdgt", "cresci15", where={"with_extscore": "syn", "extscore_inject": "num"})
    late = load_results("botdgt", "cresci15", where={"with_extscore": "syn", "extscore_inject": "late"})
    tok = load_results("botdgt", "cresci15", where={"with_extscore": "syn", "extscore_inject": "token"})
    win = load_results("botdgt", "cresci15", where={"window": 1})
    assert len(base) == 2 and len(extscore) == len(late) == len(tok) == len(win) == 1
    assert base[0]["metrics"]["test"] == base[1]["metrics"]["test"]          # seeded runs reproduce
    q = 8
    assert extscore[0]["n_params"] == base[0]["n_params"] + 2 * q                 # num 5 -> 7
    assert late[0]["n_params"] == base[0]["n_params"] + 2 * 32               # ff1 h+2 -> h
    assert tok[0]["n_params"] == base[0]["n_params"] + 2 * 32 + 32           # Linear(2, h)
    assert base[0]["extscore"] == {"extscore": "none"} and extscore[0]["extscore"]["tag"] == "syn"
    assert base[0]["selection"]["select"] == "last_epoch" and base[0]["epochs_run"] == 2
    assert win[0]["config"]["T"] == 1 and base[0]["config"]["T"] > 1
    assert base[0]["config"]["variant"] == "botdgt.botrgcn" and win[0]["config"]["variant"].endswith("win1")


def test_extscore_none_builds_the_extscore_free_model():
    from bb.methods.botdgt.model import BotDGT
    torch.manual_seed(0)
    a = BotDGT(32, 4, 5, 3, extscore_width=0, extscore_inject="late")
    torch.manual_seed(0)
    b = BotDGT(32, 4, 5, 3)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
