"""RGT (bb/methods/rgt): layer maths against a hand computation, parameter count vs the benchmark
model, ExtScore widths, no-label-leak (default) vs the leaky benchmark emulation, sampled == full, and
the CLI end to end on the synthetic Cresci fixture (CPU)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as Fn
from torch import nn


@pytest.fixture
def cr15_text(cr15):
    """Cresci fixture plus finished-looking fp16 text tables with a weak label signal."""
    from bb.data import load_labels
    from bb.io import save_npy, write_json
    from bb.paths import dataset_paths
    y = load_labels("cresci15").astype(np.float32)
    rng = np.random.default_rng(0)
    for name in ("des", "tweets_first20_mean"):
        x = rng.standard_normal((len(y), 768)).astype(np.float32)
        x[:, :4] += 2 * y[:, None] - 1
        d = dataset_paths("cresci15").emb("roberta-base")
        save_npy(d / f"{name}.f16.npy", x.astype(np.float16))
        write_json(d / f"{name}.meta.json", {"rows": len(y), "hf_commit": "fake"})
    return cr15


def _toy(n=50, e=200, R=2, dim=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=g)
    ei = torch.randint(0, n, (2, e), generator=g)
    et = torch.randint(0, R, (e,), generator=g)
    return x, ei, et


def _hand_layer(layer, x, ei, et):
    """RGT layer written out with dense loops: TransformerConv (per-head full-width q/k/v, softmax
    over in-edges / sqrt(H), heads averaged, + skip), shared gate, semantic attention (mean over nodes)."""
    n, H = x.shape
    hs = []
    for r, conv in enumerate(layer.convs):
        C = conv.heads
        q = (x @ conv.lin_query.weight.T + conv.lin_query.bias).view(n, C, H)
        k = (x @ conv.lin_key.weight.T + conv.lin_key.bias).view(n, C, H)
        v = (x @ conv.lin_value.weight.T + conv.lin_value.bias).view(n, C, H)
        src, dst = ei[:, et == r]
        u = torch.zeros(n, H)
        for i in range(n):
            js = src[dst == i]
            if len(js):
                agg = torch.zeros(H)
                for c in range(C):
                    s = (k[js, c] * q[i, c]).sum(-1) / math.sqrt(H)
                    a = torch.softmax(s, 0)
                    agg += (a[:, None] * v[js, c]).sum(0)
                u[i] = agg / C
        u = u + x @ conv.lin_skip.weight.T + conv.lin_skip.bias
        gl = layer.gate[0]
        a = torch.sigmoid(torch.cat([u, x], 1) @ gl.weight.T + gl.bias)
        hs.append(torch.tanh(u) * a + x * (1 - a))
    z = torch.stack(hs, 1)
    out = torch.zeros(n, H)
    for sc in layer.semantic.scorers:
        w = (torch.tanh(z @ sc[0].weight.T + sc[0].bias) @ sc[2].weight.T).mean(0)    # [R, 1]
        beta = torch.softmax(w, 0)
        out += (beta[None] * z).sum(1)
    return out / len(layer.semantic.scorers)


@pytest.mark.parametrize("R,C,D,fused", [(2, 2, 2, True), (2, 2, 2, False), (3, 3, 1, True), (3, 3, 1, False)])
def test_layer_matches_hand_computation(R, C, D, fused):
    from bb.methods.rgt.model import RGTLayer, init_weights
    torch.manual_seed(0)
    layer = init_weights(RGTLayer(R, 16, C, D, dropout=0.5, sem_hidden=8, fused=fused)).eval()
    x, ei, et = _toy(R=R)
    ours = layer(x, ei, et)
    ref = _hand_layer(layer, x, ei, et)
    assert torch.allclose(ours, ref, atol=1e-5)
    # every relation is used (the benchmark layer returns inside the loop and drops r >= 2 for R > 2)
    keep = et != R - 1
    assert not torch.allclose(ours, layer(x, ei[:, keep], et[keep]), atol=1e-6)
    # an isolated node gets the root/skip term only: u = W_skip x + b
    iso = torch.ones(50, dtype=torch.bool)
    iso[ei[1]] = False
    if iso.any():
        z = layer.relation_outputs(x, ei[:, :0], et[:0])
        assert torch.allclose(z[iso], layer.relation_outputs(x, ei, et)[iso], atol=1e-6)
    # no edges at all: both paths give the per-node (skip + gate) output
    assert torch.allclose(layer(x, ei[:, :0], et[:0]), _hand_layer(layer, x, ei[:, :0], et[:0]), atol=1e-5)


def test_param_count_order_init_and_extscore_widths():
    from torch_geometric.nn import TransformerConv
    from bb.methods.rgt.model import benchmark_param_count, build_model
    m = build_model(num_dim=5, cat_dim=3)
    n = sum(p.numel() for p in m.parameters())
    assert n == benchmark_param_count()
    assert sum(p.numel() for p in build_model(num_dim=5, cat_dim=3, trans_heads=8, sem_heads=8).parameters()) == \
        benchmark_param_count(heads=8, sem_heads=8)
    names = [k.split(".")[0] for k, _ in m.named_parameters()]
    first = list(dict.fromkeys(names))
    assert first == ["in_num", "in_cat", "in_tweet", "in_des", "linear1", "rgt1", "rgt2", "out1", "out2"]
    l1 = [k for k, _ in m.named_parameters() if k.startswith("rgt1.")]
    assert l1[0].startswith("rgt1.gate") and l1[-1].startswith("rgt1.semantic")
    convs = [x for x in m.modules() if isinstance(x, TransformerConv)]
    assert len(convs) == 4 and all(c.heads == 2 and not c.concat and c.root_weight for c in convs)
    assert convs[0].lin_key.weight.shape == (2 * 128, 128)           # per-head width 128, not 128 / C
    for mod in m.modules():                                          # zero bias on nn.Linear only
        if isinstance(mod, nn.Linear) and mod.bias is not None:
            assert torch.count_nonzero(mod.bias) == 0
    assert torch.count_nonzero(convs[0].lin_key.bias) > 0             # PyG Linear keeps its own init
    assert sum(p.numel() for p in build_model(num_dim=7, cat_dim=3, extscore_width=2).parameters()) == n + 2 * 32
    br = build_model(num_dim=5, cat_dim=3, extscore_width=2, extscore_inject="branch")
    late = build_model(num_dim=5, cat_dim=3, extscore_width=2, extscore_inject="late")
    assert sum(p.numel() for p in br.parameters()) == n + (2 * 32 + 32) + 32 * 128
    assert sum(p.numel() for p in late.parameters()) == n + 2 * 64
    torch.manual_seed(1)
    a = build_model(num_dim=5, cat_dim=3, extscore_width=0, extscore_inject="late")
    torch.manual_seed(1)
    b = build_model(num_dim=5, cat_dim=3)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_fused_equals_pyg_reference_model_and_grads():
    """Whole detector, fused vs per-relation PyG path: same outputs and same gradients (eval mode)."""
    from bb.methods.rgt.model import build_model
    g = _gi()
    torch.manual_seed(3)
    a = build_model(num_dim=5, cat_dim=3, fused=True).eval()
    b = build_model(num_dim=5, cat_dim=3, fused=False).eval()
    b.load_state_dict(a.state_dict())
    oa, ob = a(g), b(g)
    assert torch.allclose(oa, ob, atol=1e-5)
    oa.square().sum().backward()
    ob.square().sum().backward()
    for (k, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-4, rtol=1e-3), k


def _gi(n=40, e=120, seed=0, extra=False):
    from bb.gnn import GraphInputs
    g = torch.Generator().manual_seed(seed)
    return GraphInputs(torch.randn(n, 768, generator=g), torch.randn(n, 768, generator=g), torch.randn(n, 5, generator=g),
                       torch.randint(0, 2, (n, 3), generator=g).float(), torch.randint(0, n, (2, e), generator=g),
                       torch.randint(0, 2, (e,), generator=g), torch.randint(0, 2, (n,), generator=g),
                       torch.randn(n, 2, generator=g) if extra else None)


def test_forward_rows_and_extscore_paths():
    from bb.methods.rgt.model import build_model
    g = _gi(extra=True)
    m = build_model(num_dim=5, cat_dim=3).eval()
    full = m(g)
    rows = torch.tensor([3, 7, 11])
    assert full.shape == (40, 2) and torch.allclose(m(g, rows=rows), full[rows], atol=1e-6)
    assert not torch.allclose(full, m(g.with_edges(g.edge_index[:, :5], g.edge_type[:5])))
    for inj in ("branch", "late"):
        mj = build_model(num_dim=5, cat_dim=3, extscore_width=2, extscore_inject=inj).eval()
        o = mj(g)
        g2 = _gi(extra=True)
        g2.extra = g.extra + 1.0
        g2.des, g2.tweet, g2.num, g2.cat, g2.edge_index, g2.edge_type = g.des, g.tweet, g.num, g.cat, g.edge_index, g.edge_type
        assert o.shape == (40, 2) and not torch.allclose(o, mj(g2))
        assert torch.allclose(mj(g, rows=rows), o[rows], atol=1e-6)


def test_sampled_single_batch_equals_full(cr15_text):
    """All nodes as one seed batch with full fan-out: the sampled subgraph is the whole graph."""
    from bb.data import load_in_merged
    from bb.features import load_features
    from bb.gnn import to_device
    from bb.harness.sampler import NeighborSampler
    from bb.methods.rgt.model import build_model
    from bb.methods.rgt.train import predict_full, predict_sampled
    F = load_features("cresci15")
    g = to_device(F, "cpu")
    torch.manual_seed(0)
    m = build_model(num_dim=5, cat_dim=1)
    csc = load_in_merged("cresci15", F.relations_by_type())
    etype = torch.from_numpy(np.asarray(csc.etype, dtype=np.int64))
    full = predict_full(m, g, "none")
    samp = predict_sampled(m, g, np.arange(F.n), NeighborSampler(csc, [-1, -1]), etype, batch_size=F.n, amp="none")
    assert np.allclose(full, samp, atol=1e-5)


def test_sampled_seed_subset_full_fanout_equals_full_graph(cr15_text):
    """Seed subsets with full 2-hop fan-out: the seed outputs equal the full-graph outputs (edge
    direction and relation ids of the sampler agree with ``F.edge_index`` / ``F.edge_type``). The
    semantic weights are made graph-independent (last scorer layer zeroed -> beta = 1/R), because
    by design (benchmark) they average over the (sub)graph that the layer sees."""
    from bb.data import load_in_merged
    from bb.features import load_features
    from bb.gnn import to_device
    from bb.harness.sampler import NeighborSampler
    from bb.methods.rgt.model import build_model
    from bb.methods.rgt.train import predict_full, predict_sampled
    F = load_features("cresci15")
    g = to_device(F, "cpu")
    torch.manual_seed(0)
    m = build_model(num_dim=5, cat_dim=1)
    with torch.no_grad():
        for layer in (m.rgt1, m.rgt2):
            for sc in layer.semantic.scorers:
                sc[2].weight.zero_()
    csc = load_in_merged("cresci15", F.relations_by_type())
    etype = torch.from_numpy(np.asarray(csc.etype, dtype=np.int64))
    full = predict_full(m, g, "none")
    ip, ix = np.asarray(csc.indptr), np.asarray(csc.indices)
    deg = np.diff(ip)
    two_hop = np.array([v for v in np.flatnonzero(deg > 0) if (deg[ix[ip[v]:ip[v + 1]]] > 0).any()])
    assert len(two_hop) >= 8, "fixture graph has no 2-hop in-paths"
    seeds = two_hop[:300]                                # seeds whose in-neighbours have in-edges
    bs = 3                                               # small batches: subgraphs are proper subsets
    s2 = NeighborSampler(csc, [-1, -1])
    assert s2.sample(seeds[:bs]).num_nodes < F.n
    samp = predict_sampled(m, g, seeds, s2, etype, batch_size=bs, amp="none")
    assert np.allclose(full[seeds], samp, atol=1e-5)
    cut = predict_sampled(m, g, seeds, NeighborSampler(csc, [-1]), etype, batch_size=bs, amp="none")
    assert not np.allclose(full[seeds], cut, atol=1e-5)  # 1 hop is not enough for a 2-layer model
    # reversed edges give a different function (direction matters)
    rev = predict_full(m, g.with_edges(g.edge_index.flip(0), g.edge_type), "none")
    assert not np.allclose(full[seeds], rev[seeds], atol=1e-5)


def _train(F, sp, flip, mode, epochs=3):
    from bb.data import load_in_merged
    from bb.gnn import to_device
    from bb.methods.rgt.model import build_model
    from bb.methods.rgt.train import RGTFit, fit_full, fit_sampled
    g = to_device(F, "cpu")
    if flip:
        ev = np.concatenate([sp["val"], sp["test"]])
        g.y[ev] = 1 - g.y[ev]
    torch.manual_seed(0)
    m = build_model(num_dim=5, cat_dim=1, dropout=0.3)
    cfg = RGTFit(epochs=epochs, batch_size=8, amp="none", seed=0, select="last_epoch",
                 loss_nodes="subgraph" if mode == "leaky" else "seeds", fanouts=(256,) * (4 if mode == "leaky" else 2))
    if mode == "full":
        r = fit_full(m, g, sp, cfg)
    else:
        r = fit_sampled(m, g, sp, cfg, csc=load_in_merged("cresci15", F.relations_by_type()))
    return r.prob[sp["test"]], r


def test_no_label_leak_default_and_leaky_emulation(cr15_text):
    """Default modes: flipping every val/test label leaves the trained model unchanged. The
    ``loss_nodes='subgraph'`` emulation (benchmark TwiBot-22) does train on them."""
    from bb.features import load_features
    F = load_features("cresci15")
    sp = {s: F.splits[s] for s in ("train", "val", "test")}
    for mode in ("full", "sampled"):
        a, ra = _train(F, sp, False, mode)
        b, rb = _train(F, sp, True, mode)
        assert np.array_equal(a, b), mode
        assert [h["train_loss"] for h in ra.history] == [h["train_loss"] for h in rb.history]
    a, _ = _train(F, sp, False, "leaky")
    b, _ = _train(F, sp, True, "leaky")
    assert not np.allclose(a, b)


def test_cosine_schedule_is_periodic(cr15_text):
    from bb.features import load_features
    F = load_features("cresci15")
    sp = {s: F.splits[s] for s in ("train", "val", "test")}
    _, r = _train(F, sp, False, "full", epochs=34)
    lr = [h["lr"] for h in r.history]
    assert lr[0] == pytest.approx(1e-3) and lr[16] == pytest.approx(0, abs=1e-12)
    assert lr[32] == pytest.approx(1e-3) and lr[8] == pytest.approx(5e-4)


# ------------------------------------------------------------------------------- end to end
def test_end_to_end_seeded_extscore_and_bench_compat(cr15_text, data_root):
    from bb.harness import load_results
    from bb.extscore import make_synthetic_extscore
    from bb.methods import rgt
    args = ["--dataset", "cresci15", "--seeds", "0", "--epochs", "3", "--device", "cpu", "--batch-size", "8"]
    rgt.main(args)
    rgt.main(args)
    make_synthetic_extscore("cresci15", tag="syn", coverage=0.8)
    rgt.main(args + ["--with-extscore", "syn"])
    rgt.main(args + ["--with-extscore", "syn", "--extscore-inject", "late"])
    rgt.main(args + ["--with-extscore", "syn", "--extscore-inject", "branch"])
    rgt.main(args + ["--training", "sampled"])
    rgt.main(args + ["--training", "sampled", "--bench-compat"])
    base = load_results("rgt", "cresci15", where={"with_extscore": None, "training": "full"})
    extscore = load_results("rgt", "cresci15", where={"with_extscore": "syn", "extscore_inject": "num"})
    late = load_results("rgt", "cresci15", where={"with_extscore": "syn", "extscore_inject": "late"})
    br = load_results("rgt", "cresci15", where={"with_extscore": "syn", "extscore_inject": "branch"})
    samp = load_results("rgt", "cresci15", where={"training": "sampled", "bench_compat": False})
    bc = load_results("rgt", "cresci15", where={"bench_compat": True})
    assert len(base) == 2 and len(extscore) == len(late) == len(br) == len(samp) == len(bc) == 1
    assert base[0]["metrics"]["test"] == base[1]["metrics"]["test"]          # seeded runs reproduce
    nb = base[0]["n_params"]
    assert extscore[0]["n_params"] == nb + 64 and late[0]["n_params"] == nb + 128 and br[0]["n_params"] == nb + 96 + 32 * 128
    assert base[0]["extscore"] == {"extscore": "none"} and extscore[0]["extscore"]["tag"] == "syn"
    assert extscore[0]["extscore"]["coverage"]["test"]["coverage"] < 1
    assert base[0]["selection"]["select"] == "best_val" and base[0]["epochs_run"] == 3
    assert base[0]["config"]["variant"] == "rgt.botrgcn" and not base[0]["leaky"]
    assert samp[0]["config"]["variant"] == "rgt.botrgcn.sampled" and not samp[0]["leaky"]
    assert bc[0]["leaky"] and bc[0]["config"]["fanouts"] == [256] * 4 and bc[0]["selection"]["select"] == "last_epoch"
    assert bc[0]["config"]["patience"] == -1                                  # no val-based stopping either
    assert rgt.parse_args(["--dataset", "twibot22", "--bench-compat"]).patience == -1
    assert rgt.parse_args(["--dataset", "twibot22"]).patience == 10
    assert "benchcompat" in bc[0]["config"]["variant"]
    for r in base + samp:
        assert {"val", "test", "train"} <= set(r["metrics"]) or {"val", "test"} <= set(r["metrics"])
        assert r["predictions"]["test"] == "preds_test.npz"


# ------------------------------------------------------------------------------- CUDA-graph path
def _padded(g, n_max, e_max):
    """The GraphedStep layout: n_max + 1 rows (padding gathers row 0), padding edges = self-loops on padding rows."""
    n, E = g.n, g.edge_index.shape[1]
    n_id = torch.zeros(n_max + 1, dtype=torch.long)
    n_id[:n] = torch.arange(n)
    ei = (n + torch.arange(e_max) % (n_max + 1 - n)).repeat(2, 1)     # padding self-loops on padding rows
    ei[:, :E] = g.edge_index
    et = torch.zeros(e_max, dtype=torch.long)
    et[:E] = g.edge_type
    mask = (torch.arange(n_max + 1) < n).float()
    return g.nodes(n_id).with_edges(ei, et), mask


def test_padded_dense_forward_equals_regular():
    from bb.methods.rgt.model import build_model
    g = _gi(n=40, e=120)
    m = build_model(num_dim=5, cat_dim=3).eval()
    ref = m(g, rows=slice(0, 16))
    sub, mask = _padded(g, 64, 200)
    out = m(sub, rows=slice(0, 16), node_mask=mask, dense=True)
    assert torch.allclose(ref, out, atol=1e-5)
    # without the mask the padding rows would enter the semantic mean
    assert not torch.allclose(ref, m(sub, rows=slice(0, 16), dense=True), atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_cuda_graph_steps_equal_eager_steps(cr15_text):
    from bb.data import load_in_merged
    from bb.features import load_features
    from bb.gnn import to_device
    from bb.harness.sampler import NeighborSampler
    from bb.methods.rgt.model import build_model
    from bb.methods.rgt.train import GraphedStep, RGTFit, _setup, batch_inputs
    F = load_features("cresci15")
    g = to_device(F, "cuda", text_dtype=torch.float32)
    csc = load_in_merged("cresci15", F.relations_by_type())
    etype = torch.from_numpy(np.asarray(csc.etype, dtype=np.int64)).cuda()
    s = NeighborSampler(csc, [256, 256])
    batches = [b for ep in range(3) for b in s.iter_batches(np.asarray(F.splits["train"]), 4, epoch_seed=ep)
               if b.batch_size == 4][:6]
    assert len(batches) == 6
    cfg = RGTFit(amp="none", cuda_graph=True)
    states = []
    for graphed in (False, True):
        torch.manual_seed(0)
        m = build_model(num_dim=5, cat_dim=1, dropout=0.0).cuda()
        opt, _, _ = _setup(m, cfg, "cuda")
        lf = nn.CrossEntropyLoss()
        gs = GraphedStep(m, g, opt, lf, etype, 4, 64, 256, "none")
        for b in batches:
            if graphed:
                assert gs.fits(b)
                gs(b)
            else:
                sub = batch_inputs(g, b, etype)
                gs.eager_step(sub, slice(0, 4), sub.y[:4])
        torch.cuda.synchronize()
        states.append({k: v.detach().cpu().clone() for k, v in m.state_dict().items()})
        if graphed:
            assert gs.replays == 3 and gs.eager == 3      # 3 warm-up steps, then replays
    for k in states[0]:
        assert torch.allclose(states[0][k], states[1][k], atol=1e-5, rtol=1e-4), k


@pytest.mark.parametrize("dev", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU"))])
def test_checkpointing_same_function_and_grads(dev):
    """Activation checkpointing (dropout on) gives the same outputs and gradients as no checkpointing.
    On CUDA the device RNG must be restored for the encoder too (review round 1: it was not, because
    the checkpointed call only received the GraphInputs dataclass); the RNG is advanced between
    forward and backward so that a non-restored stream would draw different masks."""
    from bb.methods.rgt.model import build_model
    g = _gi()
    g = type(g)(*(None if t is None else t.to(dev) for t in (g.des, g.tweet, g.num, g.cat, g.edge_index,
                                                             g.edge_type, g.y, g.extra)))
    outs = []
    for ck in (False, True):
        torch.manual_seed(5)
        m = build_model(num_dim=5, cat_dim=3, dropout=0.5).to(dev).train()
        m.checkpoint = ck
        torch.manual_seed(9)
        o = m(g, rows=torch.arange(10, device=dev))
        torch.manual_seed(123)
        torch.rand(10, device=dev)
        o.square().sum().backward()
        outs.append((o.detach(), [p.grad.clone() for p in m.parameters()]))
    assert torch.allclose(outs[0][0], outs[1][0], atol=1e-6)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(outs[0][1], outs[1][1]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 autocast needs a GPU")
def test_fused_fp16_autocast_matches_pyg_on_high_in_degree():
    """Review round 1: the fused path summed messages in an fp16 accumulator, which stalls on a
    high in-degree target (error 0.6 at in-degree 50k vs 2e-3 for PyG, which sums in fp32)."""
    from bb.methods.rgt.model import EdgeCtx, RGTLayer, init_weights
    g = torch.Generator().manual_seed(1)
    n, hub = 4000, 50_000
    ei = torch.randint(0, n, (2, 20_000), generator=g)
    src = torch.randint(0, n, (hub,), generator=g)
    ei = torch.cat([ei, torch.stack([src, torch.full_like(src, 3)])], 1).cuda()
    et = torch.cat([torch.randint(0, 2, (20_000,), generator=g), torch.zeros(hub, dtype=torch.long)]).cuda()
    torch.manual_seed(0)
    fused = init_weights(RGTLayer(2, 128, 2, 2, dropout=0.0)).cuda().eval()
    ref = RGTLayer(2, 128, 2, 2, dropout=0.0, fused=False).cuda().eval()
    ref.load_state_dict(fused.state_dict())
    x = Fn_leaky(torch.randn(n, 128, generator=g).cuda() * 3)
    with torch.no_grad():
        exact = ref.relation_outputs(x, ei, et)                                      # fp32
        with torch.autocast("cuda", dtype=torch.float16):
            f16 = fused._fused_relation_outputs(x, EdgeCtx(ei, et, n, 2)).float()
            p16 = ref.relation_outputs(x, ei, et).float()
    err_f, err_p = (f16 - exact).abs().max().item(), (p16 - exact).abs().max().item()
    assert err_f < 2e-2 and err_f < 3 * err_p + 1e-3, (err_f, err_p)


def Fn_leaky(x):
    return torch.nn.functional.leaky_relu(x)


# ------------------------------------------------------------------------------- review round 2 (efficiency)
def _hub_graph(n=3000, e=20_000, hub=3000, dev="cuda"):
    g = torch.Generator().manual_seed(1)
    ei = torch.randint(0, n, (2, e), generator=g)
    src = torch.randint(0, n, (hub,), generator=g)
    ei = torch.cat([ei, torch.stack([src, torch.full_like(src, 7)])], 1)
    et = torch.randint(0, 2, (ei.shape[1],), generator=g)
    return ei.to(dev), et.to(dev), torch.randn(n, 128, generator=g).to(dev)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels / inductor need a GPU")
@pytest.mark.parametrize("C,D", [(2, 2), (8, 8)])
def test_triton_edges_and_compiled_tail_equal_torch_path(C, D):
    """``edge_path='triton'`` (queries/scores on multi-edge segments only, CSR aggregation kernels)
    and ``tail_path='compile'`` (torch.compile'd gate + semantic attention) compute the same function as the torch path, in train
    mode too: the attention dropout is drawn on the same [E, C] tensor, so the masks are identical.
    fp32: outputs to 1e-5, input and parameter gradients to 1e-4 relative."""
    from bb.methods.rgt.model import EdgeCtx, RGTLayer, init_weights
    ei, et, x0 = _hub_graph()
    torch.manual_seed(0)
    ref = init_weights(RGTLayer(2, 128, C, D, dropout=0.5)).cuda()
    w = torch.linspace(-1, 1, x0.numel(), device="cuda").view_as(x0)
    res = {}
    for impl in (("torch", "eager"), ("triton", "eager"), ("torch", "compile"), ("triton", "compile")):
        lay = init_weights(RGTLayer(2, 128, C, D, dropout=0.5)).cuda()
        lay.load_state_dict(ref.state_dict())
        lay.edge_path, lay.tail_path = impl
        ctx = EdgeCtx(ei, et, x0.shape[0], 2)
        for train in (False, True):
            lay.train(train)
            lay.zero_grad()
            x = x0.clone().requires_grad_(True)
            torch.manual_seed(42)
            o = lay(x, ei, et, ctx)
            (o * w).sum().backward()
            res[impl, train] = (o.detach(), [x.grad] + [p.grad if p.grad is not None else torch.zeros_like(p)
                                                        for p in lay.parameters()])
    for impl in (("triton", "eager"), ("torch", "compile"), ("triton", "compile")):
        for train in (False, True):
            (o0, g0), (o1, g1) = res[("torch", "eager"), train], res[impl, train]
            assert (o1 - o0).abs().max() < 1e-5, (impl, train)
            # lin_key.bias has true gradient 0 (a key bias adds q . b_k to every score of a softmax
            # group): both paths give float noise there, hence the absolute floor at the global scale
            scale = max(a.abs().max().item() for a in g0)
            for j, (a, b) in enumerate(zip(g0, g1)):
                assert torch.allclose(b, a, rtol=1e-4, atol=1e-6 * scale), (impl, train, j, (b - a).abs().max())
    # dropout is really on in train mode
    assert not torch.allclose(res[("torch", "eager"), True][0], res[("torch", "eager"), False][0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need a GPU")
def test_triton_edges_fp16_autocast_accuracy():
    """Under fp16 autocast the triton path (fp32 products and sums from the fp16 projections) is at
    least as close to the fp32 result as the torch path, including on a 50k in-degree hub."""
    from bb.methods.rgt.model import EdgeCtx, RGTLayer, init_weights
    ei, et, x = _hub_graph(n=4000, hub=50_000)
    torch.manual_seed(0)
    lay = init_weights(RGTLayer(2, 128, 2, 2, dropout=0.0)).cuda().eval()
    ctx = EdgeCtx(ei, et, x.shape[0], 2)
    with torch.no_grad():
        exact = lay._fused_relation_outputs(x, ctx)
        with torch.autocast("cuda", dtype=torch.float16):
            a = lay._fused_relation_outputs(x, ctx).float()
            lay.edge_path = "triton"
            b = lay._fused_relation_outputs(x, ctx).float()
    ea, eb = (a - exact).abs().max().item(), (b - exact).abs().max().item()
    assert eb < 2e-2 and eb <= 1.5 * ea + 1e-4, (ea, eb)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels / inductor need a GPU")
def test_fast_detector_training_steps_equal_default():
    """Whole detector, fp32, dropout on, full-graph SGD steps: ``set_impl('triton', 'compile')`` tracks
    the default path step for step (same dropout masks, same losses, same parameters). SGD, because
    Adam's normalisation would turn the float noise on the zero-gradient lin_key.bias into lr-sized
    steps in either path."""
    from bb.methods.rgt.model import build_model
    g = _gi(n=600, e=4000)
    g = type(g)(*(None if t is None else t.cuda() for t in (g.des, g.tweet, g.num, g.cat, g.edge_index,
                                                            g.edge_type, g.y, g.extra)))
    out = []
    for fast in (False, True):
        torch.manual_seed(0)
        m = build_model(num_dim=5, cat_dim=3, dropout=0.5).cuda().train()
        if fast:
            m.set_impl("triton", "compile")
        opt = torch.optim.SGD(m.parameters(), lr=0.05)
        losses = []
        torch.manual_seed(7)
        for i in range(5):
            rows = torch.arange(i * 64, (i + 1) * 64, device="cuda")
            loss = Fn.cross_entropy(m(g, rows=rows), g.y[rows])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        out.append((losses, [p.detach().clone() for p in m.parameters()]))
    assert np.allclose(out[0][0], out[1][0], atol=1e-5)
    assert all(torch.allclose(a, b, atol=1e-5, rtol=1e-4) for a, b in zip(out[0][1], out[1][1]))
