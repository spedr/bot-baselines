"""Simple-HGN / HGT (bb.methods.simplehgn): layer equivalences, HGT vs PyG, bug emulations, sampler,
no-leak, ExtScore plumbing and the end-to-end CLI (synthetic fixtures, CPU)."""
from __future__ import annotations

import numpy as np
import pytest
import torch


def fake_text(dataset: str, tables, seed: int = 0, signal: float = 0.0) -> None:
    """Finished-looking fp16 embedding tables (+ meta) for the fixture datasets."""
    from bb.data import load_labels
    from bb.io import save_npy, write_json
    from bb.paths import dataset_paths
    y = load_labels(dataset).astype(np.float32)
    rng = np.random.default_rng(seed)
    for enc, name in tables:
        x = rng.standard_normal((len(y), 768)).astype(np.float32)
        x[:, :4] += signal * (2 * y[:, None] - 1)
        d = dataset_paths(dataset).emb(enc)
        save_npy(d / f"{name}.f16.npy", x.astype(np.float16))
        write_json(d / f"{name}.meta.json", {"rows": len(y), "hf_commit": "fake"})


def _toy_inputs(n=40, e=120, seed=0, num_dim=5, cat_dim=3, extra=False):
    from bb.gnn import GraphInputs
    g = torch.Generator().manual_seed(seed)
    return GraphInputs(torch.randn(n, 768, generator=g), torch.randn(n, 768, generator=g),
                       torch.randn(n, num_dim, generator=g), torch.randint(0, 2, (n, cat_dim), generator=g).float(),
                       torch.randint(0, n, (2, e), generator=g), torch.randint(0, 2, (e,), generator=g),
                       torch.randint(0, 2, (n,), generator=g), torch.randn(n, 2, generator=g) if extra else None)


def _graph(n=30, e=120, R=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 16, generator=g, dtype=torch.float64)
    ei = torch.randint(0, n, (2, e), generator=g)
    et = torch.randint(0, R, (e,), generator=g)
    return x, ei, et


# ------------------------------------------------------------------------------- Simple-HGN layer
@pytest.mark.parametrize("final", [False, True])
def test_shgn_efficient_equals_naive(final):
    from bb.methods.simplehgn.layers import SimpleHGNConv, naive_simplehgn
    torch.manual_seed(0)
    x, ei, et = _graph()
    l1 = SimpleHGNConv(16, 16, 2, rel_dim=7, beta=0.3, final_layer=False).double()
    l2 = SimpleHGNConv(16, 16, 2, rel_dim=7, beta=0.3, final_layer=final).double()
    h, a = l1(x, ei, et)
    hn, an = naive_simplehgn(l1, x, ei, et)
    assert torch.allclose(h.double(), hn, atol=1e-5) and torch.allclose(a.double(), an, atol=1e-5)
    for mode, eff in (("paper", True), ("bench", False)):
        l2.edge_residual = mode
        o, a2 = l2(h, ei, et, pre_alpha=a)
        on, a2n = naive_simplehgn(l2, h, ei, et, pre_alpha=a, beta_effective=eff)
        assert torch.allclose(o.double(), on, atol=1e-5) and torch.allclose(a2.double(), a2n, atol=1e-5)
    if final:
        assert torch.allclose(o.norm(dim=1), torch.ones(len(o), dtype=o.dtype), atol=1e-6)


def test_shgn_attention_softmax_and_isolated_nodes():
    from bb.methods.simplehgn.layers import SimpleHGNConv
    x, ei, et = _graph()
    ei = ei[:, ei[1] != 0]                                   # node 0 receives nothing
    et = et[: ei.shape[1]]
    l = SimpleHGNConv(16, 16, 2, rel_dim=4).double()
    out, a = l(x, ei, et)
    sums = torch.zeros(30, dtype=a.dtype).index_add_(0, ei[1], a)
    has_in = torch.bincount(ei[1], minlength=30) > 0
    assert torch.allclose(sums[has_in], torch.ones(int(has_in.sum()), dtype=a.dtype))   # joint over relations
    assert torch.allclose(out[0].double(), torch.nn.functional.elu(l.W_res(x[0:1])).view(-1).double())
    assert torch.isfinite(out).all()


def test_shgn_edge_residual_bench_is_inert_paper_is_not():
    from bb.methods.simplehgn.model import SHGNBody
    x, ei, et = _graph()
    x = x.float()
    outs = {}
    for mode in ("bench", "paper"):
        for beta in (0.0, 0.05, 0.9):
            torch.manual_seed(1)
            b = SHGNBody(hidden=16, rel_dim=4, beta=beta, dropout=0.0, edge_residual=mode).eval()
            outs[mode, beta] = b(x, ei, et)
    assert torch.allclose(outs["bench", 0.05], outs["bench", 0.9]) and torch.allclose(outs["bench", 0.0], outs["bench", 0.9])
    assert torch.allclose(outs["paper", 0.0], outs["bench", 0.9], atol=1e-6)       # bench == beta 0
    assert not torch.allclose(outs["paper", 0.9], outs["paper", 0.0], atol=1e-4)


# ------------------------------------------------------------------------------- HGT
def _pyg_hgt(heads, seed=0):
    from torch_geometric.nn import HGTConv as PygHGT
    torch.manual_seed(seed)
    meta = (["user"], [("user", "follower", "user"), ("user", "following", "user")])
    return PygHGT(16, 16, meta, heads=heads), meta


@pytest.mark.parametrize("heads", [1, 2])
def test_hgt_fixed_matches_pyg_hgtconv(heads):
    from bb.methods.simplehgn.layers import HGTConv
    conv, meta = _pyg_hgt(heads)
    with torch.no_grad():                                      # non-trivial p_rel / skip
        for k in conv.p_rel:
            conv.p_rel[k].uniform_(0.5, 1.5)
        conv.skip["user"].fill_(0.3)
    x, ei, et = _graph()
    x = x.float()
    ref = conv({"user": x}, {meta[1][r]: ei[:, et == r] for r in range(2)})["user"]
    ours = HGTConv(16, 16, 2, heads=heads).load_from_pyg(conv)
    out = ours(x, ei, et)
    assert out.shape == ref.shape and torch.allclose(out, ref, atol=1e-5)
    assert sum(p.numel() for p in ours.parameters()) == sum(p.numel() for p in conv.parameters())


def test_hgt_bench_bug_is_graph_free_fixed_is_not():
    from bb.methods.simplehgn.layers import HGTConv
    x, ei, et = _graph()
    x = x.float()
    torch.manual_seed(0)
    bug = HGTConv(16, 16, 2, impl="bench_bug")
    fixed = HGTConv(16, 16, 2, impl="fixed")
    empty = ei[:, :0], et[:0]
    assert torch.allclose(bug(x, ei, et), bug(x, *empty))
    s = torch.sigmoid(bug.skip)
    assert torch.allclose(bug(x, ei, et), s * s + (1 - s) * x)
    assert not torch.allclose(fixed(x, ei, et), fixed(x, *empty))
    bug(x, ei, et).sum().backward()                            # only the gate gets a gradient
    assert bug.skip.grad is not None and bug.kqv_weight.grad is None and bug.k_rel.grad is None


def test_hgt_per_relation_softmax():
    from bb.methods.simplehgn.layers import HGTConv
    x, ei, et = _graph()
    x = x.float()
    torch.manual_seed(0)
    a = HGTConv(16, 16, 2, impl="fixed")
    b = HGTConv(16, 16, 2, impl="per_relation")
    b.load_state_dict(a.state_dict())
    assert not torch.allclose(a(x, ei, et), b(x, ei, et))
    # one relation only: per-relation and joint softmax coincide
    m = et == 0
    assert torch.allclose(a(x, ei[:, m], et[m]), b(x, ei[:, m], et[m]), atol=1e-6)


# ------------------------------------------------------------------------------- model structure
def _expected_params(kind, num=5, cat=3, rel_dim=200, D=128, q=32):
    enc = (768 * q + q) * 2 + (num * q + q) + (cat * q + q) + (4 * q * D + D)
    head = (D * 64 + 64) + (64 * 2 + 2)
    if kind == "shgn":
        layer = D * D + rel_dim * D + 3 * D + D * D + 2 * rel_dim
    else:
        layer = (D * 3 * D + 3 * D) + (D * D + D) + 2 * 2 * D * D + 1 + 2
    return enc + 2 * layer + head


@pytest.mark.parametrize("kind", ["shgn", "hgt"])
def test_param_count_init_and_extscore_widths(kind):
    from bb.methods.simplehgn.model import build_model
    from bb.gnn import n_params
    m = build_model(kind, num_dim=5, cat_dim=3)
    assert n_params(m) == _expected_params(kind)               # extscore none = benchmark parameter count
    for mod in m.modules():
        if type(mod) is torch.nn.Linear and mod.bias is not None:
            assert torch.count_nonzero(mod.bias) == 0          # benchmark init: zero bias
    torch.manual_seed(0)
    a = build_model(kind, num_dim=5, cat_dim=3, extscore_width=0, extscore_inject="late")
    torch.manual_seed(0)
    b = build_model(kind, num_dim=5, cat_dim=3)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    nb = n_params(b)
    assert n_params(build_model(kind, num_dim=7, cat_dim=3, extscore_width=2)) == nb + 2 * 32
    assert n_params(build_model(kind, num_dim=5, cat_dim=3, extscore_width=2, extscore_inject="branch")) == nb + (2 * 32 + 32) + 32 * 128
    late = build_model(kind, num_dim=5, cat_dim=3, extscore_width=2, extscore_inject="late")
    assert n_params(late) == nb + 2 * 64 and late.head.out1.in_features == 130   # before out1 (after L2 norm)
    g = _toy_inputs(extra=True)
    for model in (late, build_model(kind, num_dim=5, cat_dim=3, extscore_width=2, extscore_inject="branch")):
        assert model.eval()(g).shape == (40, 2)


def test_hgt_bench_bug_model_ignores_graph():
    from bb.methods.simplehgn.model import build_model
    g = _toy_inputs()
    empty = g.with_edges(g.edge_index[:, :0], g.edge_type[:0])
    bug = build_model("hgt", num_dim=5, cat_dim=3, hgt_impl="bench_bug").eval()
    assert torch.allclose(bug(g), bug(empty))
    for kind in ("shgn", "hgt"):
        m = build_model(kind, num_dim=5, cat_dim=3).eval()
        assert not torch.allclose(m(g), m(empty))


# ------------------------------------------------------------------------------- sampler
class _CSC:
    def __init__(self, indptr, indices, etype):
        self.indptr, self.indices, self.etype = indptr, indices, etype


def _toy_csc(n=50, e=400, seed=0):
    from bb.data.graph import build_csr_from_coo  # noqa: F401  (only the layout matters)
    rng = np.random.default_rng(seed)
    src, dst, typ = rng.integers(0, n, e), rng.integers(0, n, e), rng.integers(0, 2, e)
    order = np.lexsort((np.arange(e), typ, dst))
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(dst, minlength=n), out=indptr[1:])
    return _CSC(indptr, src[order], typ[order].astype(np.int8))


def test_relation_sampler_per_type_fanout_and_full():
    from bb.harness.sampler import NeighborSampler
    from bb.methods.simplehgn.sampling import RelationSampler
    csc = _toy_csc()
    seeds = np.array([3, 7, 11])
    full_r = RelationSampler(csc, [-1, -1]).sample(seeds)
    full_h = NeighborSampler(csc, [-1, -1]).sample(seeds)
    assert set(full_r.n_id.tolist()) == set(full_h.n_id.tolist()) and full_r.num_edges == full_h.num_edges
    assert np.array_equal(np.sort(full_r.e_id), np.sort(full_h.e_id))
    s = RelationSampler(csc, [2, [1, 3]])
    b = s.sample(seeds, rng_seed=5)
    et = np.asarray(csc.etype)[b.e_id]
    dst = b.edge_index[1]
    hop1 = dst < b.batch_size
    for v in range(b.batch_size):
        g = int(seeds[v])
        for r in (0, 1):
            deg = int(((np.asarray(csc.etype)[csc.indptr[g]:csc.indptr[g + 1]]) == r).sum())
            got = int(((dst == v) & hop1 & (et == r)).sum())
            assert got == min(deg, 2)
    # sampled edges are real CSC entries of the right target
    assert np.array_equal(np.asarray(csc.indices)[b.e_id], b.n_id[b.edge_index[0]])
    tgt = np.searchsorted(csc.indptr, b.e_id, side="right") - 1
    assert np.array_equal(tgt, b.n_id[b.edge_index[1]])
    assert len(set(b.n_id.tolist())) == len(b.n_id) and np.array_equal(b.n_id[:3], seeds)
    b2 = s.sample(seeds, rng_seed=5)
    assert np.array_equal(b.n_id, b2.n_id) and np.array_equal(b.e_id, b2.e_id)   # deterministic


# ------------------------------------------------------------------------------- training
@pytest.fixture
def cr15_text(cr15):
    fake_text("cresci15", [("roberta-base", "des"), ("roberta-base", "tweets_first20_mean")], signal=1.0)
    return cr15


@pytest.mark.parametrize("kind", ["shgn", "hgt"])
def test_exact_subgraph_equals_full_graph(cr15_text, kind):
    from bb.data import load_in_merged
    from bb.features import load_features
    from bb.gnn import predict_full, to_device
    from bb.methods.simplehgn.model import build_model
    from bb.methods.simplehgn.sampling import make_sampler
    from bb.methods.simplehgn.train import predict
    F = load_features("cresci15")
    g = to_device(F, "cpu")
    torch.manual_seed(0)
    m = build_model(kind, num_dim=5, cat_dim=1, rel_dim=8)
    full = predict_full(m, g)
    csc = load_in_merged("cresci15", F.relations_by_type())
    etype = torch.from_numpy(np.asarray(csc.etype, dtype=np.int64))
    samp = predict(m, g, np.arange(F.n), make_sampler(csc, [-1, -1], kind == "hgt"), etype, batch_size=5)
    assert np.allclose(full, samp, atol=1e-5)


def _train(kind, flip, loss_nodes="seeds", select="last_epoch"):
    from bb.data import load_in_merged
    from bb.features import load_features
    from bb.gnn import to_device
    from bb.methods.simplehgn.model import build_model
    from bb.methods.simplehgn.sampling import make_sampler
    from bb.methods.simplehgn.train import TrainConfig, fit
    F = load_features("cresci15")
    sp = {s: np.asarray(F.splits[s]) for s in ("train", "val", "test")}
    g = to_device(F, "cpu")
    if flip:
        ev = np.concatenate([sp["val"], sp["test"]])
        g.y[ev] = 1 - g.y[ev]
    csc = load_in_merged("cresci15", F.relations_by_type())
    etype = torch.from_numpy(np.asarray(csc.etype, dtype=np.int64))
    torch.manual_seed(0)
    m = build_model(kind, num_dim=5, cat_dim=1, rel_dim=8)
    cfg = TrainConfig(epochs=4, batch_size=6, lr=1e-2, select=select, loss_nodes=loss_nodes)
    r = fit(m, g, sp, cfg, train_sampler=make_sampler(csc, [2, 2], kind == "hgt"),
            eval_sampler=make_sampler(csc, [-1, -1], kind == "hgt"), etype=etype)
    return r


@pytest.mark.parametrize("kind", ["shgn", "hgt"])
def test_no_label_leak_and_leak_emulation(cr15_text, kind):
    """Flipping every val/test label leaves the default (seed-only loss) model unchanged; the
    benchmark all-node loss (--loss-nodes all) does change it (val/test users are in the subgraphs)."""
    a, b = _train(kind, False), _train(kind, True)
    assert np.allclose(a.prob, b.prob, atol=1e-6, equal_nan=True)
    assert [h["train_loss"] for h in a.history] == [h["train_loss"] for h in b.history]
    c, d = _train(kind, False, "all"), _train(kind, True, "all")
    assert not np.allclose(c.prob, d.prob, atol=1e-6, equal_nan=True)
    # best_val selection: training trajectory still independent of val/test labels
    e, f = _train(kind, False, select="best_val"), _train(kind, True, select="best_val")
    assert [h["train_loss"] for h in e.history] == [h["train_loss"] for h in f.history]


def test_nan_val_epoch_is_never_selected(cr15_text, monkeypatch):
    """An epoch whose val outputs contain NaN is never selected, even when NaN -> 0.5 would make it
    the best val accuracy (review round 1). Val predictions are faked per epoch: all wrong (acc 0)
    except epoch 1, which is all NaN (acc = bot share > 0 after nan_to_num)."""
    from bb.methods.simplehgn import train as T
    real, calls = T.predict, {"n": 0}

    def fake(model, g, idx, *a, **k):
        i = calls["n"]
        calls["n"] += 1
        if i < 4 and not k.get("all_nodes"):
            y = g.y.cpu().numpy()[np.asarray(idx)].astype(np.float32)
            return np.full(len(idx), np.nan, np.float32) if i == 1 else 1.0 - y
        return real(model, g, idx, *a, **k)

    monkeypatch.setattr(T, "predict", fake)
    r = _train("shgn", False, select="best_val")
    assert r.history[1]["val_nan"] > 0 and r.history[1]["val_acc"] > 0   # would win without the guard
    assert r.selection["best_epoch"] != 1 and r.selection["best_value"] == 0.0
    assert np.isfinite(r.prob[~np.isnan(r.prob)]).all()


# ------------------------------------------------------------------------------- end to end
def test_end_to_end_cli_seeded_extscore_and_variants(cr15_text, data_root):
    from bb.harness import load_results
    from bb.extscore import make_synthetic_extscore
    from bb.methods import simplehgn
    base = ["--dataset", "cresci15", "--seeds", "0", "--epochs", "3", "--device", "cpu", "--rel-dim", "8"]
    simplehgn.main(base)
    simplehgn.main(base)
    make_synthetic_extscore("cresci15", tag="syn", coverage=0.8)
    simplehgn.main(base + ["--with-extscore", "syn"])
    simplehgn.main(base + ["--with-extscore", "syn", "--extscore-inject", "late"])
    simplehgn.main(base + ["--edge-residual", "bench"])
    simplehgn.main(base + ["--model", "hgt"])
    simplehgn.main(base + ["--model", "hgt", "--hgt-impl", "bench_bug", "--with-extscore", "syn"])
    shgn = load_results("simplehgn", "cresci15", where={"with_extscore": None, "edge_residual": "paper"})
    extscore = load_results("simplehgn", "cresci15", where={"with_extscore": "syn", "extscore_inject": "num"})
    late = load_results("simplehgn", "cresci15", where={"with_extscore": "syn", "extscore_inject": "late"})
    bench = load_results("simplehgn", "cresci15", where={"edge_residual": "bench"})
    hgt = load_results("hgt", "cresci15", where={"hgt_impl": "fixed"})
    bug = load_results("hgt", "cresci15", where={"hgt_impl": "bench_bug"})
    assert len(shgn) == 2 and len(extscore) == len(late) == len(bench) == len(hgt) == len(bug) == 1
    assert shgn[0]["metrics"]["test"] == shgn[1]["metrics"]["test"]           # seeded runs reproduce
    assert extscore[0]["n_params"] == shgn[0]["n_params"] + 2 * 32 and late[0]["n_params"] == shgn[0]["n_params"] + 2 * 64
    assert shgn[0]["extscore"] == {"extscore": "none"} and extscore[0]["extscore"]["tag"] == "syn"
    assert extscore[0]["extscore"]["coverage"]["test"]["coverage"] < 1
    assert shgn[0]["selection"]["select"] == "best_val" and shgn[0]["epochs_run"] == 3
    assert shgn[0]["config"]["variant"] == "shgn.botrgcn" and bench[0]["config"]["variant"] == "shgn.botrgcn.eres-bench"
    assert bug[0]["config"]["variant"] == "hgt.botrgcn.bench_bug" and hgt[0]["config"]["head_dropout"] is True
    for r in shgn + hgt:
        assert {"val", "test"} <= set(r["metrics"]) and r["predictions"]["test"] == "preds_test.npz"


def test_bench_compat_flags():
    from bb.methods.simplehgn import parse_args
    a = parse_args(["--dataset", "twibot22", "--model", "shgn", "--bench-compat"])
    assert (a.loss_nodes, a.val_nodes, a.fanouts) == ("all", "subgraph", [256] * 4)
    h = parse_args(["--dataset", "twibot22", "--model", "hgt"])
    assert (h.loss_nodes, h.fanouts, h.batch_size, h.head_dropout, h.wd) == ("seeds", [256, 256], 256, False, 3e-5)
    s = parse_args(["--dataset", "twibot22"])
    assert (s.wd, s.batch_size, s.epochs, s.patience, s.rel_dim) == (0.0, 512, 50, 10, 200)
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "twibot20", "--bench-compat"])


@pytest.mark.parametrize("kind", ["shgn", "hgt"])
def test_layers_run_in_float32_under_autocast(kind):
    """Only the input encoder uses reduced precision; body and head stay float32 (fp16 overflow guard)."""
    from bb.methods.simplehgn.model import build_model
    m = build_model(kind, num_dim=5, cat_dim=3).eval()
    seen = {}
    m.body.register_forward_hook(lambda mod, inp, out: seen.update(body_in=inp[0].dtype, body_out=out.dtype))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(_toy_inputs())
    assert seen == {"body_in": torch.float32, "body_out": torch.float32} and out.dtype == torch.float32


# ------------------------------------------------------------------------------- review round 2 (efficiency)
def test_union_sampler_equals_shared_sampler():
    from bb.harness.sampler import NeighborSampler
    from bb.methods.simplehgn.sampling import UnionSampler
    csc = _toy_csc(n=80, e=900, seed=3)
    seeds = np.arange(0, 80, 3)
    for fan in ([2, 2], [-1, -1], [5, 1, 3], [0, 4]):
        a, b = NeighborSampler(csc, fan), UnionSampler(csc, fan)
        for ep in range(3):
            for x, y in zip(a.iter_batches(seeds, 7, epoch_seed=ep), b.iter_batches(seeds, 7, epoch_seed=ep)):
                assert np.array_equal(x.n_id, y.n_id) and np.array_equal(x.edge_index, y.edge_index)
                assert np.array_equal(x.e_id, y.e_id) and np.array_equal(x.hop_nodes, y.hop_nodes)


def test_prefetch_thread_same_batches_errors_and_early_exit():
    from bb.methods.simplehgn.sampling import RelationSampler
    from bb.methods.simplehgn.train import Prefetch
    s = RelationSampler(_toy_csc(), [2, [1, 3]])
    seeds = np.arange(50)
    ref = list(s.iter_batches(seeds, 4, epoch_seed=1))
    with Prefetch(s.iter_batches(seeds, 4, epoch_seed=1), 2) as it:
        got = list(it)
    assert len(got) == len(ref) and all(np.array_equal(a.e_id, b.e_id) for a, b in zip(ref, got))
    with Prefetch(s.iter_batches(seeds, 4, epoch_seed=1), 2) as it:      # leave early: producer stops
        next(iter(it))
    assert not it.t.is_alive()

    def bad():
        yield 1
        raise KeyError("boom")
    with pytest.raises(KeyError), Prefetch(bad(), 2) as it:
        list(it)


def test_packed_batch_equals_batch_inputs():
    from bb.methods.simplehgn.sampling import RelationSampler
    from bb.methods.simplehgn.train import batch_inputs, device_batch, host_etype, pack_batch
    csc = _toy_csc()
    g = _toy_inputs(n=50)
    etype = torch.from_numpy(np.asarray(csc.etype, dtype=np.int64))
    b = RelationSampler(csc, [3, 3]).sample(np.array([1, 4, 9]), rng_seed=2)
    x, y = batch_inputs(g, b, etype), device_batch(g, *pack_batch(b, host_etype(etype), False))
    for f in ("des", "tweet", "num", "cat", "edge_index", "edge_type", "y"):
        assert torch.equal(getattr(x, f), getattr(y, f)), f


def test_fused_kernels_match_eager():
    """CUDA Triton kernels (forward + both gradients) against the float64 eager form; CPU: the
    public functions are the eager form."""
    from bb.methods.simplehgn import fused
    dev = "cuda" if torch.cuda.is_available() and fused.HAVE_TRITON else "cpu"
    g = torch.Generator().manual_seed(0)
    n, m, E, F = 40, 60, 700, 128
    src, dst = torch.randint(0, m, (E,), generator=g), torch.randint(0, n, (E,), generator=g)
    x, w, a = torch.randn(m, F, generator=g), torch.rand(E, generator=g), torch.randn(n, F, generator=g)
    up, ua = torch.randn(n, F, generator=g), torch.randn(E, generator=g)
    res = {}
    for kind, dt, d in (("test", torch.float32, dev), ("ref", torch.float64, "cpu")):
        xs, ws, as_ = (t.to(d, dt).clone().requires_grad_() for t in (x, w, a))
        s_, d_ = src.to(d), dst.to(d)
        o = fused.gather_scale_scatter(xs, s_, ws, d_, n)
        r = fused.gather_rowdot(as_, d_, xs, s_)
        ((o * up.to(d, dt)).sum() + (r * ua.to(d, dt)).sum()).backward()
        res[kind] = [t.detach().double().cpu() for t in (o, r, xs.grad, ws.grad, as_.grad)]
    for p, q in zip(res["test"], res["ref"]):
        assert torch.allclose(p, q, rtol=1e-5, atol=1e-5)
