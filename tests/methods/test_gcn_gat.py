"""GCN / GAT (TwiBot-22 benchmark versions): structure, self-loop handling, sampling-depth exactness,
lexicographic selection, no label leak, best/final-state reporting, ExtScore plumbing (synthetic, CPU)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from bb.io import save_npy, write_json
from bb.paths import dataset_paths


def fake_text(dataset: str, tables, seed: int = 0, signal: float = 1.0) -> None:
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
    fake_text("cresci15", [("roberta-base", "des"), ("roberta-base", "tweets_first20_mean")])
    return cr15


def _toy(n=60, e=400, seed=0, loops=5):
    """Random directed multigraph (duplicates + a few self loops) with GraphInputs and its in-CSC."""
    from bb.data.graph import CSR
    from bb.gnn import GraphInputs
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, n, (e,), generator=g)
    dst = torch.randint(0, n, (e,), generator=g)
    dst = torch.where(src == dst, (dst + 1) % n, dst)   # no accidental loops ...
    src[:loops] = dst[:loops]                            # ... only the planted ones
    src[loops:loops + 20], dst[loops:loops + 20] = src[loops + 20:loops + 40], dst[loops + 20:loops + 40]  # duplicates
    ei = torch.stack([src, dst])
    order = np.lexsort((np.arange(e), dst.numpy()))
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(dst.numpy(), minlength=n), out=indptr[1:])
    csc = CSR(indptr, src.numpy()[order].astype(np.int32), None, np.zeros(e, np.int8))
    gi = GraphInputs(torch.randn(n, 768, generator=g), torch.randn(n, 768, generator=g), torch.randn(n, 5, generator=g),
                     torch.randint(0, 2, (n, 3), generator=g).float(), ei, torch.zeros(e, dtype=torch.long),
                     torch.randint(0, 2, (n,), generator=g))
    return gi, csc


def _model(conv, seed=0, reference=False, **kw):
    """Our model, or (reference=True) the stock-PyG reference carrying the same weights."""
    from bb.methods.gcn_gat.model import build_model, reference_model
    torch.manual_seed(seed)
    nd, cd = kw.pop("num_dim", 5), kw.pop("cat_dim", 3)
    m = build_model(conv, hidden=128, num_dim=nd, cat_dim=cd, **kw).eval()
    if reference:
        r = reference_model(conv, num_dim=nd, cat_dim=cd, **kw).eval()
        r.load_state_dict(m.state_dict())
        return r
    return m


# ------------------------------------------------------------------------------- model
@pytest.mark.parametrize("conv", ["gcn", "gat"])
def test_structure_param_count_no_kaiming(conv):
    from torch_geometric.nn import GATConv, GCNConv
    m = _model(conv)
    H, q = 128, 32
    lin = (768 * q + q) * 2 + (5 * q + q) + (3 * q + q) + (H * H + H) + (H * H + H) + (H * 2 + 2)
    if conv == "gcn":
        body = 2 * (H * H + H)
        assert sum(isinstance(x, GCNConv) for x in m.modules()) == 2
    else:  # GATConv(128, 32, heads=4): lin 128x128, att_src/att_dst 4x32 each, bias 128; GATConv(128, 128, 1)
        body = (H * H + 2 * 4 * 32 + H) + (H * H + 2 * H + H)
        assert [c.heads for c in m.modules() if isinstance(c, GATConv)] == [4, 1]
    assert sum(p.numel() for p in m.parameters()) == lin + body
    # no activation inside the conv stack; dropout 0.3 on the input concat and between the convs
    assert not any(isinstance(x, (torch.nn.ReLU, torch.nn.ELU, torch.nn.LeakyReLU)) for x in m.body.modules())
    assert m.encoder.input_dropout.p == 0.3 and m.body.dropout.p == 0.3 and m.encoder.act.negative_slope == 0.01
    # PyTorch default init (not kaiming): uniform(+-1/sqrt(fan_in)) bounds hold for every Linear
    for x in m.modules():
        if type(x) is torch.nn.Linear:
            assert x.weight.abs().max() <= 1 / np.sqrt(x.in_features) + 1e-6


def test_gcn_body_is_affine():
    """GCNConv -> dropout -> GCNConv with no nonlinearity is an affine map of the node features."""
    m = _model("gcn")
    gi, _ = _toy()
    x1, x2 = torch.randn(gi.n, 128), torch.randn(gi.n, 128)
    with torch.no_grad():
        f = lambda x: m.body(x, gi.edge_index)  # noqa: E731
        assert torch.allclose(f(0.3 * x1 + 0.7 * x2), 0.3 * f(x1) + 0.7 * f(x2), atol=1e-4)


@pytest.mark.parametrize("conv", ["gcn", "gat"])
def test_matches_stock_pyg_layers(conv):
    """Our convs (normalize/self loops handled by us) = stock GCNConv/GATConv on a graph with
    duplicate edges and release self loops; host self-loop preparation is the same operation."""
    from bb.methods.gcn_gat.train import prepare_edges
    gi, _ = _toy()
    ref, ours = _model(conv, reference=True)(gi), _model(conv)(gi)
    assert torch.allclose(ref, ours, atol=1e-5)
    pre = gi.with_edges(torch.from_numpy(prepare_edges(gi.edge_index.numpy(), gi.n)), gi.edge_type)
    assert torch.allclose(_model(conv, reference=True)(pre), ref, atol=1e-5)


# ------------------------------------------------------------------------------- sampling depth
@pytest.mark.parametrize("conv,depth", [("gcn", 3), ("gat", 2)])
def test_reduced_depth_and_trimmed_layers_are_exact(conv, depth):
    """Full fan-out at the reduced depth with per-layer trimmed edge sets = the stock-PyG model on the
    full graph; finite fan-out at the reduced depth = the stock-PyG model on the literal 4-hop sampled
    subgraph for the same random stream (hop 4 only adds nodes nobody reads)."""
    from bb.gnn import predict_full
    from bb.harness.sampler import NeighborSampler
    from bb.methods.gcn_gat.train import _Batcher, make_plan, predict
    gi, csc = _toy()
    m, ref = _model(conv), _model(conv, reference=True)
    full = predict_full(ref, gi)
    idx = np.arange(gi.n)
    gcn = conv == "gcn"
    exact = predict(m, _Batcher(gi, csc, [-1] * depth, gcn), idx, 7, 0)
    assert np.allclose(exact, full, atol=1e-5)
    # stock PyG on the literal [3]*4 subgraph, seeds' rows
    lit_s = NeighborSampler(csc, [3] * 4)
    lit = []
    for k, s in enumerate(range(0, gi.n, 9)):
        b = lit_s.sample(idx[s:s + 9], rng_seed=123 + k)
        sub = gi.nodes(torch.from_numpy(b.n_id)).with_edges(torch.from_numpy(b.edge_index), torch.zeros(b.num_edges, dtype=torch.long))
        lit.append(torch.softmax(ref(sub)[: b.batch_size], 1)[:, 1].detach().numpy())
    lit = np.concatenate(lit)
    red = predict(m, _Batcher(gi, csc, [3] * depth, gcn), idx, 9, 123)
    assert np.allclose(red, lit, atol=1e-5) and not np.allclose(red, full, atol=1e-3)
    if gcn:   # two sampled hops are NOT enough for GCN: 2-hop nodes would lose their in-degree
        with pytest.raises(ValueError):
            make_plan(NeighborSampler(csc, [-1, -1]).sample(idx[:5]), gcn=True)
        b = NeighborSampler(csc, [-1, -1]).sample(idx[:9])
        sub = gi.nodes(torch.from_numpy(b.n_id)).with_edges(torch.from_numpy(b.edge_index), torch.zeros(b.num_edges, dtype=torch.long))
        two = torch.softmax(ref(sub)[:9], 1)[:, 1].detach().numpy()
        assert not np.allclose(two, full[:9], atol=1e-4)


def test_gcn_analytic_degrees_equal_third_hop_sampling():
    """GCN default depth: degrees from min(indeg, k) (no 3rd hop sampled) give the identical plans as
    sampling the 3rd hop and counting; graphs with self loops on high-degree nodes fall back."""
    from bb.methods.gcn_gat.train import _Batcher
    gi, csc = _toy(loops=0)
    a, s_ = _Batcher(gi, csc, [3, 3, 3], True), _Batcher(gi, csc, [3, 3, 3], True, analytic=False)
    assert a.mode == "analytic-degree" and s_.mode == "sampled" and len(a.sampler.fanouts) == 2
    for k in range(10):
        seeds = np.arange(k * 6, k * 6 + 6)
        pa, ps = a.plan(seeds, k), s_.plan(seeds, k)
        assert all(np.array_equal(x, y) for x, y in ((pa.n_id, ps.n_id), (pa.ei1, ps.ei1), (pa.ei2, ps.ei2)))
        assert np.allclose(pa.w1, ps.w1, atol=1e-7) and np.allclose(pa.w2, ps.w2, atol=1e-7)
    gl, cl = _toy(loops=25)
    assert _Batcher(gl, cl, [1, 1, 1], True).mode == "sampled"
    assert _Batcher(gl, cl, [-1, -1, -1], True).mode == "analytic-degree"


# ------------------------------------------------------------------------------- selection
def test_is_better_lexicographic():
    from bb.methods.gcn_gat.train import LEX_KEYS, is_better
    base = {k: 0.5 for k in LEX_KEYS}
    assert is_better(base, None) and not is_better(base, dict(base))
    assert is_better({**base, "accuracy": 0.6, "f1": 0.0}, base)       # acc decides first
    assert not is_better({**base, "accuracy": 0.4, "f1": 1.0}, base)
    assert is_better({**base, "recall": 0.51}, base)                   # ties fall through to the last key
    assert not is_better({**base, "f1": float("nan")}, base)


def test_pr_auc_is_the_official_trapezoid_not_average_precision():
    """Official utils.calc_metrics: auc(recall, precision) over precision_recall_curve. It breaks
    acc/F1/MCC ties in is_better (frequent on Cresci-2015), so the estimator matters."""
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve
    from bb.methods.gcn_gat.train import pr_auc, val_metrics
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(0.3 * y + rng.random(300) * 0.7, 0, 1)
    pr, rc, _ = precision_recall_curve(y, p)
    assert pr_auc(y, p) == auc(rc, pr) and abs(pr_auc(y, p) - average_precision_score(y, p)) > 1e-4
    assert val_metrics(y, p)["pr_auc"] == pr_auc(y, p) and np.isnan(pr_auc(np.ones(5), np.ones(5)))


def test_no_label_leak_and_both_states(cr15_text):
    """Test labels never matter; val labels only drive selection/stopping (the final weights of a
    fixed number of epochs do not depend on them)."""
    from bb.data import load_in_merged
    from bb.features import load_features
    from bb.gnn import to_device
    from bb.methods.gcn_gat.train import TrainConfig, fit
    F = load_features("cresci15")
    sp = {s: F.splits[s] for s in ("train", "val", "test")}
    csc = load_in_merged("cresci15", F.relations_by_type())
    outs = {}
    for flip in ("none", "test", "val"):
        g = to_device(F, "cpu", edges=False)
        if flip != "none":
            g.y[sp[flip]] = 1 - g.y[sp[flip]]
        m = _model("gat", cat_dim=1)
        cfg = TrainConfig(lr=1e-2, batch_size=8, fanouts=(2, 2), max_epochs=4, patience=1000, seed=0)
        outs[flip] = fit(m, g, sp, cfg, csc=csc)
    te, va = sp["test"], sp["val"]
    assert np.allclose(outs["none"].prob_best[te], outs["test"].prob_best[te], atol=1e-6)
    assert np.allclose(outs["none"].prob_final[te], outs["val"].prob_final[te], atol=1e-6)
    r = outs["none"]
    assert r.epochs_run == 4 and 0 <= r.best_epoch <= 3 and np.isnan(r.prob_best[sp["train"]]).all()
    if r.best_epoch == 3:
        assert np.allclose(r.prob_best[va], r.prob_final[va])
    # patience: stops after `patience` evaluations without improvement
    m = _model("gat", cat_dim=1)
    r2 = fit(m, to_device(F, "cpu", edges=False), sp, TrainConfig(lr=0.0, batch_size=64, fanouts=(2, 2),
             max_epochs=50, patience=3, seed=0), csc=csc)
    assert r2.best_epoch == 0 and r2.epochs_run == 4


# ------------------------------------------------------------------------------- end to end
def test_end_to_end_seeded_bug_flag_and_extscore(cr15_text, data_root):
    from bb.harness import load_results
    from bb.extscore import make_synthetic_extscore
    from bb.methods import gcn_gat
    args = ["--dataset", "cresci15", "--seeds", "0", "--max-epochs", "3", "--device", "cpu", "--batch-size", "16"]
    gcn_gat.main(["--conv", "gcn"] + args)
    gcn_gat.main(["--conv", "gcn"] + args)
    gcn_gat.main(["--conv", "gcn", "--emulate-state-bug"] + args)
    gcn_gat.main(["--conv", "gat", "--eval", "full", "--bench-hops"] + args)
    make_synthetic_extscore("cresci15", tag="syn", coverage=0.8)
    gcn_gat.main(["--conv", "gat", "--with-extscore", "syn"] + args)
    gcn_gat.main(["--conv", "gat", "--with-extscore", "syn", "--extscore-inject", "late"] + args)
    base = load_results("gcn", "cresci15", where={"emulate_state_bug": False})
    bug = load_results("gcn", "cresci15", where={"emulate_state_bug": True})
    gat = load_results("gat", "cresci15", where={"with_extscore": None})
    extscore = load_results("gat", "cresci15", where={"with_extscore": "syn", "extscore_inject": "num"})
    late = load_results("gat", "cresci15", where={"with_extscore": "syn", "extscore_inject": "late"})
    assert len(base) == 2 and len(bug) == 1 and len(gat) == 1 and len(extscore) == 1 and len(late) == 1
    assert base[0]["metrics"]["test"] == base[1]["metrics"]["test"]                 # seeded runs reproduce
    assert base[0]["metrics"] == base[0]["metrics_by_state"]["best_val"]
    assert bug[0]["metrics"] == bug[0]["metrics_by_state"]["final_weights"]
    assert bug[0]["metrics_by_state"] == base[0]["metrics_by_state"]                 # same run, other report
    assert bug[0]["selection"]["select"] == "final_weights" and bug[0]["config"]["variant"] == "gcn.botrgcn.finalw"
    assert base[0]["config"]["fanouts"] == [256] * 3 and gat[0]["config"]["fanouts"] == [256] * 4
    assert gat[0]["config"]["variant"] == "gat.botrgcn.f256x256x256x256.evalfull"
    assert extscore[0]["n_params"] == gat[0]["n_params"] + 2 * 32 and late[0]["n_params"] == gat[0]["n_params"] + 2 * 2
    assert gat[0]["extscore"] == {"extscore": "none"} and extscore[0]["extscore"]["tag"] == "syn"
    assert extscore[0]["extscore"]["coverage"]["test"]["coverage"] < 1
    for r in base + gat:
        assert {"val", "test"} <= set(r["metrics"]) and "pr_auc" in r["metrics"]["test"]
        assert r["predictions"]["test"] == "preds_test.npz" and r["epochs_run"] == 3


def test_too_few_hops_rejected():
    from bb.methods.gcn_gat import parse_args
    with pytest.raises(SystemExit):
        parse_args(["--conv", "gcn", "--fanouts", "256", "256"])
    assert parse_args(["--conv", "gat"]).fanouts == [256, 256]
    assert parse_args(["--conv", "gcn"]).fanouts == [256, 256, 256]
    assert parse_args(["--conv", "gcn", "--bench-hops"]).fanouts == [256] * 4


# ------------------------------------------------------------------------------- CUDA graph step
@pytest.mark.parametrize("conv", ["gcn", "gat"])
def test_slot_packing_holds_exactly_the_plan(conv):
    """The packed static buffer (one H2D copy per step from a ring of host buffers) holds, after every
    load, exactly the plan plus inert padding: edge slots beyond the plan are dummy self loops with
    weight 0 and mask rows beyond the batch are 0, also when a smaller plan follows a larger one in
    the same ring buffer (CPU: the copy is synchronous; the CUDA path is the same code + events)."""
    from bb.methods.gcn_gat.train import _Batcher, _Slot
    gi, csc = _toy(n=60, e=300)
    gcn = conv == "gcn"
    bt = _Batcher(gi, csc, [3] * (3 if gcn else 2), gcn)
    plans = [bt.plan(np.arange(s, s + k), i) for i, (s, k) in enumerate([(0, 8), (8, 3), (20, 8), (40, 1), (30, 8), (50, 5)] * 2)]
    caps = (max(p.num_nodes for p in plans) + 1, max(p.ei1.shape[1] for p in plans) + 5, max(p.ei2.shape[1] for p in plans) + 3)
    sl = _Slot(caps, 8, "cpu", gcn)
    for p in plans:
        sl.load(p)
        n, e1, e2 = p.sizes
        assert torch.equal(sl.n_id[:n], torch.from_numpy(p.n_id))
        for ei, w, pe, pw, e in ((sl.ei[0], sl.w[0], p.ei1, p.w1, e1), (sl.ei[1], sl.w[1], p.ei2, p.w2, e2)):
            assert torch.equal(ei[:, :e], torch.from_numpy(pe)) and bool((ei[:, e:] == sl.dummy).all())
            if gcn:
                assert torch.equal(w[:e], torch.from_numpy(pw)) and bool((w[e:] == 0).all())
            else:
                assert w is None
        assert sl.mask.tolist() == [1.0] * p.batch_size + [0.0] * (8 - p.batch_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (run under tools/memrun --light)")
@pytest.mark.parametrize("conv,opt", [("gcn", "sgd"), ("gat", "sgd"), ("gcn", "adam"), ("gat", "adam"), ("gcn", "fused"), ("gat", "fused")])
def test_cuda_graph_step_equals_eager(conv, opt):
    """Captured padded steps (incl. a partial batch and an over-capacity eager fallback) give the same
    parameters as plain eager steps (dropout 0 so both see the same randomness). With SGD the match
    is to float rounding (same gradients); Adam's normalised first steps amplify rounding on
    near-zero gradient entries (capturable vs default Adam code path), hence the looser tolerance."""
    from bb.methods.gcn_gat.train import CapturedStep, _Batcher, _forward
    gi, csc = _toy(n=60, e=300)
    gi = type(gi)(*(t.cuda() if torch.is_tensor(t) else t for t in
                    (gi.des, gi.tweet, gi.num, gi.cat, gi.edge_index, gi.edge_type, gi.y, gi.extra)))
    gcn = conv == "gcn"
    bt = _Batcher(gi, csc, [3] * (3 if gcn else 2), gcn)
    batches = [np.arange(s, min(s + 8, 60)) for s in range(0, 60, 8)]           # last batch has 4 seeds
    ref, cap = _model(conv, dropout=0.0).cuda().train(), _model(conv, dropout=0.0).cuda().train()
    cap.load_state_dict(ref.state_dict())
    if opt == "sgd":
        o1, o2 = torch.optim.SGD(ref.parameters(), lr=1e-2), torch.optim.SGD(cap.parameters(), lr=1e-2)
    else:
        o1 = torch.optim.Adam(ref.parameters(), lr=1e-3, weight_decay=1e-5)
        o2 = torch.optim.Adam(cap.parameters(), lr=1e-3, weight_decay=1e-5, capturable=True,
                              fused=opt == "fused")   # fused = the default on the CUDA-graph path (review 2)
    init = [p.detach().clone() for p in ref.parameters()]
    plans = [bt.plan(s, i) for i, s in enumerate(batches)]
    sizes = [p.num_nodes for p in plans]
    small = (min(sizes) + 1, 4096, 1024)
    caps = (max(sizes), 4096, 1024)                  # the largest plan(s) do not fit -> eager fallback
    n_big = sum(x >= max(sizes) for x in sizes)
    st = CapturedStep(cap, o2, gi, 8, [small, caps], plans[int(np.argmin(sizes))], gcn)
    assert all(torch.equal(a, b) for a, b in zip(ref.state_dict().values(), cap.state_dict().values()))  # warm-up undone
    ref_sum = 0.0
    for rep in range(2):                             # 16 loads > _Slot.RING: the host ring buffers wrap
        for i, p in enumerate(plans):
            t = bt.tensors(p)
            o1.zero_grad()
            loss = torch.nn.functional.cross_entropy(_forward(ref, t)[: p.batch_size], t[0].y[: p.batch_size])
            loss.backward()
            ref_sum += float(loss) * p.batch_size
            o1.step()
            if st.fits(p):
                st.step(p)
            else:
                st.eager(bt.tensors(p), p.batch_size)
    assert st.eager_steps == 2 * n_big and sum(st.replays) == 2 * (len(batches) - n_big) and st.replays[0] > 0
    # the training loss is summed inside the graph (loss x real seeds), same as the eager sum
    assert abs(float(st.loss_sum) - ref_sum) <= 1e-3 * max(1.0, abs(ref_sum)), (float(st.loss_sum), ref_sum)
    if opt == "sgd":
        for (k, a), b in zip(ref.state_dict().items(), cap.state_dict().values()):
            assert float((a - b).abs().max()) <= 1e-6, (k, float((a - b).abs().max()))
    else:
        # Adam normalises each entry, so float rounding of near-zero gradient entries is amplified
        # entry-wise; check the state handling (one step per batch, reset after warm-up) exactly and
        # the total update in norm.
        n = 2 * len(batches)
        assert all(int(o2.state[p]["step"]) == n for p in cap.parameters())
        assert all(int(o1.state[p]["step"]) == n for p in ref.parameters())
        for (k, a), b, p0 in zip(ref.named_parameters(), cap.parameters(), init):
            da, db = a.detach() - p0, b.detach() - p0
            assert float((da - db).norm() / da.norm()) < 0.05, (k, float((da - db).norm() / da.norm()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (run under tools/memrun --light)")
def test_cuda_graph_dropout_masks_change_per_replay():
    """Dropout inside the captured step must draw a fresh mask on every replay (graph-safe RNG)."""
    x = torch.ones(4096, device="cuda")
    drop = torch.nn.Dropout(0.3).train()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        drop(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = drop(x)
    g.replay()
    a = out.clone()
    g.replay()
    assert not torch.equal(a, out) and 0.6 < float((out > 0).float().mean()) < 0.8
