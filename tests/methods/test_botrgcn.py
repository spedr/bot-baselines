"""BotRGCN, its w/o-graph ablation, the shared botrgcn feature scheme and the bb.gnn skeleton
(synthetic fixtures, CPU)."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from bb.io import load_npy, save_npy, write_json
from bb.paths import dataset_paths


def fake_text(dataset: str, tables, seed: int = 0, signal: float = 0.0) -> None:
    """Write finished-looking fp16 embedding tables (+ meta) for the fixture datasets."""
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
    fake_text("cresci15", [("roberta-base", "des"), ("roberta-base", "tweets_first20_mean")], signal=1.0)
    return cr15


# ------------------------------------------------------------------------------- feature scheme
def test_features_cresci_shapes_zscore_links_idempotent(cr15_text):
    from bb.data import load_profile, load_split
    from bb.features import build_features, load_features
    m = build_features("cresci15")
    F = load_features("cresci15")
    n = F.n
    assert F.num_prop.shape == (n, 5) and F.cat_prop.shape == (n, 1) and F.des.shape == (n, 768)
    assert m["num_columns"] == ["followers_count", "active_days", "screen_name_length", "friends_count", "listed_count"]
    z = F.num_prop.astype(np.float64)
    assert np.allclose(z.mean(0), 0, atol=1e-5)
    sd = z.std(0, ddof=1)
    assert np.all((np.abs(sd - 1) < 1e-5) | (sd == 0))
    # raw values come back through the stored mean/std
    p = load_profile("cresci15", ["followers_count", "listed_count"])
    mu, s = np.array(m["num_zscore"]["mean"]), np.array(m["num_zscore"]["std"])
    raw = z * s + mu
    assert np.allclose(raw[:, 0], p["followers_count"].astype(float).to_numpy())
    assert np.allclose(raw[:, 4], p["listed_count"].astype(float).to_numpy())
    # symlinks resolve to the shared tables; graph and splits as in the shared layer
    assert os.path.islink(F.dir / "des.f16.npy") and np.array_equal(F.des, load_npy(dataset_paths("cresci15").emb("roberta-base") / "des.f16.npy"))
    ei = load_npy(dataset_paths("cresci15").graph / "edge_index.i64.npy")
    assert np.array_equal(F.edge_index, ei) and m["edges_total"] == ei.shape[1]
    assert sum(m["edges"].values()) == ei.shape[1] and m["edge_type_ids"] == {"friend": 0, "follower": 1}
    for s_ in ("train", "val", "test"):
        assert np.array_equal(F.splits[s_], load_split("cresci15", s_))
    # idempotent: second call is a no-op with the same key; a forced rebuild gives identical arrays
    mt = (F.dir / "num_prop.f32.npy").stat().st_mtime_ns
    m2 = build_features("cresci15")
    assert m2["input_key"] == m["input_key"] and (F.dir / "num_prop.f32.npy").stat().st_mtime_ns == mt
    build_features("cresci15", force=True)
    assert np.array_equal(load_features("cresci15").num_prop, F.num_prop)
    # a changed input (text table rebuilt) changes the key and triggers a rebuild
    fake_text("cresci15", [("roberta-base", "des")], seed=1)
    assert build_features("cresci15")["input_key"] != m["input_key"]


def test_features_twibot22_recipe_and_late_text(tw22):
    from bb.build import twibot22
    from bb.data import load_profile
    from bb.features import build_features, load_features
    twibot22.build_graph()
    m = build_features("twibot22")
    assert not m["text_ready"]                          # embeddings not built: features still usable
    with pytest.raises(FileNotFoundError):
        load_features("twibot22")
    F = load_features("twibot22", require_text=False)
    assert F.des is None
    p = load_profile("twibot22", ["name", "screen_name", "verified"])
    mu, sd = np.array(m["num_zscore"]["mean"]), np.array(m["num_zscore"]["std"])
    raw = F.num_prop.astype(np.float64) * sd + mu
    assert np.allclose(raw[:, 2], p["name"].str.len().to_numpy(dtype=float))           # len(name): benchmark bug kept
    assert m["num_columns"][2] == "name_length" and m["edge_type_ids"] == {"followers": 0, "following": 1}
    assert np.array_equal(F.cat_prop[:, 1], p["verified"].to_numpy(dtype=float))
    assert F.cat_prop[:, 2].sum() == F.n                  # every fixture user has the default image URL
    mu_ = build_features("twibot22", "botrgcn-uname")
    raw_u = load_features("twibot22", "botrgcn-uname", require_text=False).num_prop * mu_["num_zscore"]["std"][2] + mu_["num_zscore"]["mean"][2]
    assert np.allclose(raw_u[:, 2], p["screen_name"].str.len().to_numpy(dtype=float), atol=1e-4)
    # text lands later: the next call links it
    fake_text("twibot22", [("roberta-base", "des"), ("roberta-base", "tweets_first20_mean")])
    m3 = build_features("twibot22")
    assert m3["text_ready"] and m3["input_key"] != m["input_key"]
    assert load_features("twibot22").tweets.shape == (F.n, 768)


def test_features_twibot20_missing_profile_and_variants(tw20):
    from bb.features import build_features, load_features
    fake_text("twibot20", [("distilroberta-base", "des"), ("roberta-base", "tweets_all_mean"),
                           ("roberta-base", "des"), ("roberta-base", "tweets_first20_mean")])
    m = build_features("twibot20")
    F = load_features("twibot20")
    mu, sd = np.array(m["num_zscore"]["mean"]), np.array(m["num_zscore"]["std"])
    raw = F.num_prop.astype(np.float64) * sd + mu
    assert np.allclose(raw[5], 0, atol=1e-3)              # fixture user 5 has profile: null -> all fills 0
    assert np.allclose(raw[0, [0, 3, 4]], [0, 1, 100], atol=1e-3)    # followers 10k, friends k+1, statuses 100+k
    assert F.cat_prop.shape == (F.n, 3) and F.cat_prop[0, 2] == 1 and F.cat_prop[1, 2] == 0  # default_profile_image k%4==0
    assert (F.label[F.splits["support"]] == -1).all() and m["edge_type_ids"] == {"following": 0, "follower": 1}
    o = load_features("twibot20", "botrgcn-orig6")
    assert o.num_prop.shape == (F.n, 6) and o.cat_prop.shape == (F.n, 11)
    b = load_features("twibot20", "botrgcn-b20")
    assert os.readlink(b.dir / "tweets.f16.npy").endswith("roberta-base/tweets_first20_mean.f16.npy")
    assert os.readlink(F.dir / "des.f16.npy").endswith("distilroberta-base/des.f16.npy")


# ------------------------------------------------------------------------------- model + skeleton
def _toy_inputs(n=40, e=120, seed=0, num_dim=5, cat_dim=3, extra=False):
    from bb.gnn import GraphInputs
    g = torch.Generator().manual_seed(seed)
    return GraphInputs(torch.randn(n, 768, generator=g), torch.randn(n, 768, generator=g),
                       torch.randn(n, num_dim, generator=g), torch.randint(0, 2, (n, cat_dim), generator=g).float(),
                       torch.randint(0, n, (2, e), generator=g), torch.randint(0, 2, (e,), generator=g),
                       torch.randint(0, 2, (n,), generator=g), torch.randn(n, 2, generator=g) if extra else None)


def test_botrgcn_structure_tied_and_param_order():
    from torch_geometric.nn import RGCNConv
    from bb.methods.botrgcn.model import build_model
    m = build_model("botrgcn", hidden=32, num_dim=5, cat_dim=3)
    convs = [x for x in m.modules() if isinstance(x, RGCNConv)]
    assert len(convs) == 1                                 # one R-GCN applied twice (weights tied)
    q, D = 8, 32
    lin = (768 * q + q) * 2 + (5 * q + q) + (3 * q + q) + (D * D + D) + (D * D + D) + (D * 2 + 2)
    rgcn = 2 * D * D + D * D + D                           # 2 relation weights + root + bias
    assert sum(p.numel() for p in m.parameters()) == lin + rgcn
    names = [n for n, _ in m.named_parameters()]
    order = [n.split(".")[1] for n in names]
    assert order[:6] == ["des", "des", "tweet", "tweet", "num", "num"] and names[-1] == "head.out2.bias"
    g = _toy_inputs()
    m.eval()
    out = m(g)
    g2 = g.with_edges(g.edge_index[:, :10], g.edge_type[:10])
    assert out.shape == (40, 2) and not torch.allclose(out, m(g2))   # depends on the graph


def test_wog_ignores_graph_and_tied_variant():
    from bb.methods.botrgcn.model import build_model
    m = build_model("wog", hidden=32, num_dim=5, cat_dim=3).eval()
    g = _toy_inputs()
    empty = g.with_edges(g.edge_index[:, :0], g.edge_type[:0])
    assert torch.allclose(m(g), m(empty))
    t = build_model("wog", hidden=32, num_dim=5, cat_dim=3, wog_tied=True)
    assert sum(p.numel() for p in m.parameters()) - sum(p.numel() for p in t.parameters()) == 32 * 32 + 32


def test_mlp2_is_two_layer_mlp_on_concatenated_features():
    """Review round 1: the literal 'two MLP layers with user features' w/o-graph ablation."""
    from bb.methods.botrgcn.model import build_model
    torch.manual_seed(0)
    m = build_model("mlp2", hidden=32, num_dim=7, cat_dim=3).eval()
    g = _toy_inputs(num_dim=7)
    x = torch.cat([g.des, g.tweet, g.num, g.cat], 1)
    assert torch.allclose(m(g), m.l2(m.act(m.l1(x))), atol=1e-5)          # block-wise == one Linear(1546, D)
    assert sum(p.numel() for p in m.parameters()) == (1546 * 32 + 32) + (32 * 2 + 2)
    assert torch.allclose(m(g), m(g.with_edges(g.edge_index[:, :0], g.edge_type[:0])))
    with pytest.raises(ValueError):
        build_model("mlp2", hidden=32, num_dim=5, cat_dim=3, extscore_width=2, extscore_inject="late")


def test_half_input_linear_keeps_tiny_gradients():
    """Review round 1: the fp16 text projection must not underflow the weight gradient (TwiBot-22
    scale: loss averaged over ~7e5 rows -> per-row gradients ~1e-8)."""
    import torch.nn.functional as Fn
    from bb.gnn.model import _HalfInputLinear
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3000, 768, generator=g).half()
    w = (torch.randn(8, 768, generator=g) * 0.05).requires_grad_()
    b = torch.zeros(8, requires_grad=True)
    up = torch.randn(3000, 8, generator=g) * 1e-8                  # upstream grad, below fp16 subnormals
    y = _HalfInputLinear.apply(x, w, b)
    assert y.dtype == torch.float32
    assert torch.allclose(y, Fn.linear(x.float(), w, b), atol=2e-2)   # forward = fp16 GEMM
    y.backward(up)
    ref_w, ref_b = up.double().T @ x.double(), up.double().sum(0)
    assert torch.allclose(w.grad.double(), ref_w, rtol=1e-4, atol=1e-12)
    assert torch.allclose(b.grad.double(), ref_b, rtol=1e-4, atol=1e-12)
    w2 = w.detach().clone().requires_grad_()                          # the old path loses them
    Fn.linear(x, w2.half(), b.detach().half()).float().backward(up)
    assert (w2.grad.double() - ref_w).norm() / ref_w.norm() > 0.5


def test_to_device_takes_the_pread_path_for_whole_tables(cr15_text, monkeypatch):
    """Review round 1: np.load memmaps have offset = header length, so the old ``offset == 0`` test
    never selected ``table_to_device``. Whole tables must take it; values must equal the memmap."""
    import bb.gnn.inputs as I
    from bb.features import load_features
    F = load_features("cresci15")
    calls = []
    orig = I.table_to_device
    monkeypatch.setattr(I, "table_to_device", lambda *a, **k: calls.append(a) or orig(*a, **k))
    g = I.to_device(F, "cpu")
    assert len(calls) == 2
    assert np.array_equal(g.des.numpy(), np.asarray(F.des, dtype=np.float32))
    assert np.array_equal(g.tweet.numpy(), np.asarray(F.tweets, dtype=np.float32))


def test_extscore_widths_and_injection_points():
    from bb.methods.botrgcn.model import build_model
    base = build_model("botrgcn", hidden=32, num_dim=5, cat_dim=3)
    torch.manual_seed(0)
    a = build_model("botrgcn", hidden=32, num_dim=5, cat_dim=3, extscore_width=0, extscore_inject="late")
    torch.manual_seed(0)
    b = build_model("botrgcn", hidden=32, num_dim=5, cat_dim=3)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))  # width 0 = baseline
    nb = sum(p.numel() for p in base.parameters())
    num = build_model("botrgcn", hidden=32, num_dim=7, cat_dim=3, extscore_width=2, extscore_inject="num")
    br = build_model("botrgcn", hidden=32, num_dim=5, cat_dim=3, extscore_width=2, extscore_inject="branch")
    late = build_model("botrgcn", hidden=32, num_dim=5, cat_dim=3, extscore_width=2, extscore_inject="late")
    assert sum(p.numel() for p in num.parameters()) == nb + 2 * 8
    assert sum(p.numel() for p in br.parameters()) == nb + (2 * 8 + 8) + 8 * 32
    assert sum(p.numel() for p in late.parameters()) == nb + 2 * 2
    g = _toy_inputs(extra=True)
    assert br(g).shape == (40, 2) and late(g).shape == (40, 2)
    g7 = _toy_inputs(num_dim=7)
    assert num(g7).shape == (40, 2)


def test_sampled_full_fanout_equals_full_batch(cr15_text):
    from bb.data import load_in_merged
    from bb.features import load_features
    from bb.gnn import predict_full, predict_sampled, to_device
    from bb.harness.sampler import NeighborSampler
    from bb.methods.botrgcn.model import build_model
    F = load_features("cresci15")
    g = to_device(F, "cpu")
    torch.manual_seed(0)
    m = build_model("botrgcn", hidden=32, num_dim=5, cat_dim=1)
    full = predict_full(m, g)
    csc = load_in_merged("cresci15", F.relations_by_type())
    etype = torch.from_numpy(np.asarray(csc.etype, dtype=np.int64))
    idx = np.arange(F.n)
    samp = predict_sampled(m, g, idx, NeighborSampler(csc, [-1, -1]), etype, batch_size=7)
    assert np.allclose(full, samp, atol=1e-5)


def test_no_label_leak_last_epoch(cr15_text):
    """Changing every val/test label must not change the trained model (loss on train rows only)."""
    from bb.features import load_features
    from bb.gnn import FitConfig, fit_full_batch, fit_sampled, to_device
    from bb.data import load_in_merged
    from bb.methods.botrgcn.model import build_model
    F = load_features("cresci15")
    sp = {s: F.splits[s] for s in ("train", "val", "test")}
    outs = []
    for flip in (False, True):
        for sampled in (False, True):
            g = to_device(F, "cpu")
            if flip:
                ev = np.concatenate([sp["val"], sp["test"]])
                g.y[ev] = 1 - g.y[ev]
            torch.manual_seed(0)
            m = build_model("botrgcn", hidden=32, num_dim=5, cat_dim=1)
            cfg = FitConfig(epochs=4, batch_size=8, seed=0)
            if sampled:
                r = fit_sampled(m, g, sp, cfg, csc=load_in_merged("cresci15", F.relations_by_type()))
            else:
                r = fit_full_batch(m, g, sp, cfg)
            outs.append(r.prob[sp["test"]])
    assert np.allclose(outs[0], outs[2], atol=1e-6) and np.allclose(outs[1], outs[3], atol=1e-6)


# ------------------------------------------------------------------------------- end to end
def test_end_to_end_seeded_extscore_and_wog(cr15_text, data_root):
    from bb.harness import load_results
    from bb.extscore import make_synthetic_extscore
    from bb.methods import botrgcn
    args = ["--dataset", "cresci15", "--seeds", "0", "--epochs", "6", "--device", "cpu"]
    botrgcn.main(args)
    botrgcn.main(args)
    make_synthetic_extscore("cresci15", tag="syn", coverage=0.8)
    botrgcn.main(args + ["--with-extscore", "syn"])
    botrgcn.main(args + ["--with-extscore", "syn", "--extscore-inject", "late"])
    botrgcn.main(args + ["--model", "wog"])
    botrgcn.main(args + ["--training", "sampled", "--batch-size", "16"])
    botrgcn.main(args + ["--model", "mlp2", "--with-extscore", "syn"])
    mlp2 = load_results("botrgcn_mlp2", "cresci15")
    assert len(mlp2) == 1 and mlp2[0]["config"]["variant"].startswith("mlp2.")
    assert mlp2[0]["n_params"] == (768 * 2 + 5 + 2 + 1) * 32 + 32 + 32 * 2 + 2   # ExtScore [z, m] widens num 5 -> 7
    base = load_results("botrgcn", "cresci15", where={"with_extscore": None, "training": "full"})
    extscore = load_results("botrgcn", "cresci15", where={"with_extscore": "syn", "extscore_inject": "num"})
    late = load_results("botrgcn", "cresci15", where={"with_extscore": "syn", "extscore_inject": "late"})
    wog = load_results("botrgcn_wog", "cresci15")
    samp = load_results("botrgcn", "cresci15", where={"training": "sampled"})
    assert len(base) == 2 and len(extscore) == 1 and len(late) == 1 and len(wog) == 1 and len(samp) == 1
    assert base[0]["metrics"]["test"] == base[1]["metrics"]["test"]          # seeded runs reproduce
    assert extscore[0]["n_params"] == base[0]["n_params"] + 16 and late[0]["n_params"] == base[0]["n_params"] + 4
    assert base[0]["extscore"] == {"extscore": "none"} and extscore[0]["extscore"]["tag"] == "syn"
    assert extscore[0]["extscore"]["coverage"]["test"]["coverage"] < 1
    assert base[0]["selection"]["select"] == "last_epoch" and base[0]["epochs_run"] == 6
    assert wog[0]["n_params"] == base[0]["n_params"] - (2 * 32 * 32 + 32 * 32 + 32) + (32 * 32 + 32) * 2
    assert base[0]["config"]["variant"] == "botrgcn.botrgcn" and samp[0]["config"]["variant"].endswith("sampled-1x-1")
    for r in base + wog:
        assert {"val", "test", "train"} <= set(r["metrics"]) and r["predictions"]["test"] == "preds_test.npz"


# ------------------------------------------------------------------------------- R-GCN semantics
def _rgcn_reference(conv, x, ei, et):
    """x_i' = root x_i + sum_r mean_{(j->i) in E_r} x_j W_r + b (Schlichtkrull et al.; PyG 2.0)."""
    out = x @ conv.root + conv.bias
    for r in range(conv.num_relations):
        m = et == r
        src, dst = ei[0, m], ei[1, m]
        agg = torch.zeros_like(x).index_add_(0, dst, x[src])
        deg = torch.zeros(x.shape[0]).index_add_(0, dst, torch.ones(len(dst))).clamp(min=1)
        out = out + (agg / deg[:, None]) @ conv.weight[r]
    return out


@pytest.mark.parametrize("n_edges", [300, 70_000])        # below / above PyG's segment_matmul heuristic
def test_rgcnconv_per_relation_mean_on_both_heuristic_paths(n_edges):
    import torch_geometric.backend as B
    import torch_geometric.typing as T
    from torch_geometric.nn import RGCNConv as PygRGCNConv
    from bb.gnn import RGCNConv
    g = torch.Generator().manual_seed(0)
    n, D = 500, 32
    x = torch.randn(n, D, generator=g)
    # relation 0 dense, relation 1 sparse: a pooled mean over both differs from per-relation means
    ei = torch.randint(0, n, (2, n_edges), generator=g)
    et = (torch.rand(n_edges, generator=g) < 0.1).long()
    torch.manual_seed(0)
    ours = RGCNConv(D, D, num_relations=2)
    pyg = PygRGCNConv(D, D, num_relations=2)
    pyg.load_state_dict(ours.conv.state_dict())
    ref = _rgcn_reference(ours.conv, x, ei, et)
    with torch.no_grad():
        for flag in (None, True, False):                     # None = heuristic, True/False = forced
            B.use_segment_matmul = flag
            try:
                assert torch.allclose(ours(x, ei, et), ref, atol=1e-4), flag
                assert B.use_segment_matmul is flag              # global flag restored
                plain = pyg(x, ei, et)
            finally:
                B.use_segment_matmul = None
            takes_segmm = T.WITH_SEGMM and (flag is True or (flag is None and pyg._use_segment_matmul_heuristic_output))
            if takes_segmm:                                  # the PyG >= 2.3 path pools the mean over relations
                assert not torch.allclose(plain, ref, atol=1e-3)
            else:
                assert torch.allclose(plain, ref, atol=1e-4)
        if T.WITH_SEGMM:                                     # the heuristic does flip at this size
            assert pyg._use_segment_matmul_heuristic_output == (n_edges > 60_000)
