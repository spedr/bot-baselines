"""RoBERTa + MLP (docs/methods/roberta_mlp.md): model structure, ExtScore widths, input reading,
host-streamed vs device-resident equivalence, no label leakage from val/test, end to end on the
synthetic Cresci-2015 fixture (fake fp16 text tables, CPU)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from bb.io import save_npy, write_json
from bb.paths import dataset_paths


def fake_text(dataset: str, signal: float = 1.0, seed: int = 0) -> None:
    from bb.data import load_labels
    y = load_labels(dataset).astype(np.float32)
    rng = np.random.default_rng(seed)
    for name in ("des", "tweets_first20_mean"):
        x = rng.standard_normal((len(y), 768)).astype(np.float32)
        x[:, :4] += signal * (2 * y[:, None] - 1)
        d = dataset_paths(dataset).emb("roberta-base")
        save_npy(d / f"{name}.f16.npy", x.astype(np.float16))
        write_json(d / f"{name}.meta.json", {"rows": len(y), "hf_commit": "fake"})


@pytest.fixture
def cr15_text(cr15):
    fake_text("cresci15")
    return cr15


# ------------------------------------------------------------------------------------- model
def test_structure_widths_and_linear_projections():
    from bb.methods.roberta_mlp.model import RobertaMLP, n_params
    m = RobertaMLP(0.3)
    base = 2 * (768 * 384 + 384) + (768 * 128 + 128) + (128 * 2 + 2)
    assert n_params(m) == base
    assert [n for n, _ in m.named_parameters()] == [
        "tweet_proj.weight", "tweet_proj.bias", "des_proj.weight", "des_proj.bias",
        "linear_relu_tweet.0.weight", "linear_relu_tweet.0.bias", "classifier.weight", "classifier.bias"]
    assert isinstance(m.linear_relu_tweet[1], torch.nn.LeakyReLU) and m.linear_relu_tweet[1].negative_slope == 0.01
    # ExtScore widths (spec 8): fuse 770 -> 128, late 130 -> 2, branch Linear(2, k) and 768 + k -> 128
    assert n_params(RobertaMLP(0.3, extscore_width=2)) == base + 2 * 128
    assert n_params(RobertaMLP(0.3, extscore_width=2, extscore_inject="late")) == base + 2 * 2
    assert n_params(RobertaMLP(0.3, extscore_width=2, extscore_inject="branch", branch_dim=8)) == base + (2 * 8 + 8) + 8 * 128
    # extscore_width 0: the inject mode is irrelevant; same parameters in the same order and init
    torch.manual_seed(0)
    a = RobertaMLP(0.3, extscore_inject="late")
    torch.manual_seed(0)
    b = RobertaMLP(0.3)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    # the 768 -> 384 projections have no activation: the pre-LeakyReLU activation is affine in x
    m.eval()
    x1, x2 = torch.randn(3, 1536), torch.randn(3, 1536)
    pre = lambda x: m.linear_relu_tweet[0](torch.cat([m.tweet_proj(x[:, :768]), m.des_proj(x[:, 768:])], 1))  # noqa: E731
    z = pre(torch.zeros(1, 1536))
    assert torch.allclose(pre(x1 + x2) - z, (pre(x1) - z) + (pre(x2) - z), atol=1e-4)
    # input layout [tweet | des]: tweet_proj sees the first 768 columns
    with torch.no_grad():
        m.tweet_proj.weight.zero_()
        m.tweet_proj.bias.zero_()
    x3 = x1.clone()
    x3[:, :768] = 123.0
    assert torch.allclose(m(x1.half()), m(x3.half()))


def test_with_human_metrics():
    from bb.harness import compute_metrics
    from bb.methods.roberta_mlp import with_human
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.1, 0.7, 0.2, 0.3, 0.1])     # tp 2, fn 1, fp 1, tn 3
    m = with_human(compute_metrics(y, p))
    assert m["precision_human"] == pytest.approx(3 / 4) and m["recall_human"] == pytest.approx(3 / 4)
    assert m["f1_human"] == pytest.approx(0.75) and m["f1"] == pytest.approx(2 / 3)


# ------------------------------------------------------------------------------------ inputs
def test_read_text_rows_matches_gather(tmp_path):
    from bb.methods.roberta_mlp import read_text_rows
    rng = np.random.default_rng(0)
    a = rng.standard_normal((1000, 768)).astype(np.float16)
    b = rng.standard_normal((1000, 768)).astype(np.float16)
    save_npy(tmp_path / "a.npy", a)
    save_npy(tmp_path / "b.npy", b)
    splits = {"train": rng.permutation(1000)[:600], "val": np.array([999, 3, 500]), "test": np.array([0])}
    out = read_text_rows([tmp_path / "a.npy", tmp_path / "b.npy"], splits, chunk=97)
    for s, r in splits.items():
        assert out[s].dtype == np.float16 and out[s].shape == (len(r), 1536)
        assert np.array_equal(out[s][:, :768], a[r]) and np.array_equal(out[s][:, 768:], b[r])
    # caller-provided destinations (CPU tensors; CUDA tensors are filled block by block)
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devs:
        dst = {s: torch.full((len(r), 1536), 7.0, dtype=torch.float16, device=dev) for s, r in splits.items()}
        got = read_text_rows([tmp_path / "a.npy", tmp_path / "b.npy"], splits, chunk=97, out=dst)
        assert got is dst
        for s in splits:
            assert torch.equal(dst[s].cpu(), torch.from_numpy(out[s])), (dev, s)


def _setup(argv, force_host=False, dev="cpu"):
    from bb.data import load_labels, load_split
    from bb.extscore import prepare_extscore
    from bb.methods.roberta_mlp import build_splits, parse_args
    a = parse_args(argv)
    y = np.asarray(load_labels(a.dataset)).astype(np.int64)
    splits = {s: np.asarray(load_split(a.dataset, s), np.int64) for s in ("train", "val", "test")}
    J = prepare_extscore(a, a.dataset, splits, labels=y)
    data, _, info = build_splits(a, a.dataset, splits, y, J, dev, force_host=force_host)
    return a, y, J, data, info


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_placement_by_free_vram_is_result_neutral(cr15_text):
    """build_splits on the GPU: with the default budget every split is read straight into VRAM;
    with a reserve larger than the GPU every split stays on the host and is streamed through the
    double-buffered prefetcher (odd chunk). Same weights, bit for bit, eager and graph, +- ExtScore."""
    from bb.extscore import make_synthetic_extscore
    from bb.methods.roberta_mlp import train_one
    make_synthetic_extscore("cresci15", tag="syn", coverage=0.8)
    for extra in ([], ["--cuda-graph"], ["--with-extscore", "syn"]):
        states = []
        from bb.data import load_split
        row = 1536 * 2 + (8 if extra and extra[0] == "--with-extscore" else 0)
        evb = (len(load_split("cresci15", "test")) + len(load_split("cresci15", "val"))) * row
        for place in ([], ["--vram-reserve-gb", "1e6", "--chunk-rows", "100"], ["--vram-budget-gb", str(evb * 1.0001 / 1e9)]):
            a, y, J, data, info = _setup(["--dataset", "cresci15", "--epochs", "2"] + extra + place, dev="cuda")
            want = {"device"} if not place else {"host"} if place[0] == "--vram-reserve-gb" else None
            if want is not None:
                assert set(info["placement"].values()) == want, info
            else:                                   # budget = val + test: train is streamed
                assert info["placement"] == {"test": "device", "val": "device", "train": "host"}, info
            r = train_one(a, 0, data, y, J, "cuda")
            states.append({k: v.detach().cpu() for k, v in r["model"].state_dict().items()})
        for sd in states[1:]:
            assert all(torch.equal(states[0][k], sd[k]) for k in sd), extra


def test_host_streaming_equals_device_resident(cr15_text):
    from bb.methods.roberta_mlp import train_one
    from bb.extscore import make_synthetic_extscore
    make_synthetic_extscore("cresci15", tag="syn", coverage=0.8)
    for extra in ([], ["--with-extscore", "syn"]):
        argv = ["--dataset", "cresci15", "--device", "cpu", "--epochs", "3"] + extra
        states = []
        for force_host, chunk in ((False, 32_768), (True, 64), (True, 100)):   # 100 -> 96 (multiple of bs 32)
            a, y, J, data, info = _setup(argv + ["--chunk-rows", str(chunk)], force_host=force_host)
            assert info["placement"]["train"] == ("host" if force_host else "device")
            r = train_one(a, 0, data, y, J, "cpu")
            states.append((r["model"].state_dict(), r["history"]))
        for sd, h in states[1:]:
            assert all(torch.equal(x, z) for x, z in zip(states[0][0].values(), sd.values()))
            assert [e["val_f1"] for e in h] == [e["val_f1"] for e in states[0][1]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_graph_matches_eager_without_dropout(cr15_text):
    """--cuda-graph replays the same update rule: with dropout 0 the trained weights agree with
    eager training (capturable Adam; tiny float differences allowed). Includes the partial batch."""
    import dataclasses
    from bb.methods.roberta_mlp import train_one
    states = []
    for g in ([], ["--cuda-graph"]):
        a, y, J, data, _ = _setup(["--dataset", "cresci15", "--epochs", "3", "--dropout", "0",
                                   "--batch-size", "4"] + g)
        data = {s: dataclasses.replace(d, x=d.x.cuda(), y=d.y.cuda()) for s, d in data.items()}
        r = train_one(a, 0, data, y, J, "cuda")
        assert (r["model"].tweet_proj.weight.device.type == "cuda")
        states.append({k: v.cpu() for k, v in r["model"].state_dict().items()})
    assert data["train"].n % 4 != 0                       # the eager partial batch is exercised
    for k in states[0]:
        assert torch.allclose(states[0][k], states[1][k], atol=1e-5), k


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_graph_dropout_stream_matches_eager(cr15_text, monkeypatch):
    """With dropout on, graph replays draw the same dropout masks as eager steps (graph-safe Philox
    offsets), with and without ExtScore, host-streamed or not. Teeth: the same eager run with another
    CUDA dropout stream (same init, same shuffle) moves the weights by orders of magnitude more, so
    a frozen mask (the classic non-graph-safe RNG bug) or a shifted stream would fail."""
    import dataclasses
    from bb.extscore import make_synthetic_extscore
    from bb.methods import roberta_mlp as R
    make_synthetic_extscore("cresci15", tag="syn", coverage=0.8)

    def run(extra, other_stream=False, host=False):
        a, y, J, data, _ = _setup(["--dataset", "cresci15", "--epochs", "2", "--dropout", "0.5",
                                   "--batch-size", "4", "--chunk-rows", "40"] + extra)
        data = {s: dataclasses.replace(d, x=d.x if host and s == "train" else d.x.cuda(), y=d.y.cuda(),
                                       extscore=None if d.extscore is None else (d.extscore if host and s == "train" else d.extscore.cuda()),
                                       on_device=not (host and s == "train")) for s, d in data.items()}
        with monkeypatch.context() as mp:
            if other_stream:          # same CPU init and shuffle, different CUDA (dropout) stream
                orig = R.set_seed
                mp.setattr(R, "set_seed", lambda s, **k: (orig(s, **k), torch.cuda.manual_seed_all(s + 1000)))
            r = R.train_one(a, 0, data, y, J, "cuda")
        return {k: v.detach().cpu() for k, v in r["model"].state_dict().items()}

    diff = lambda p, q: max(float((p[k] - q[k]).abs().max()) for k in p)  # noqa: E731
    for extscore in ([], ["--with-extscore", "syn"]):
        eager = run(extscore)
        assert diff(eager, run(extscore + ["--cuda-graph"])) < 1e-5
        assert diff(eager, run(extscore + ["--cuda-graph"], host=True)) < 1e-5
        assert diff(eager, run(extscore, other_stream=True)) > 1e-3


def test_no_label_leak_from_val_test(cr15_text):
    """Flipping every val/test label leaves the trained (last-epoch) model unchanged."""
    from bb.methods.roberta_mlp import train_one
    a, y, J, data, _ = _setup(["--dataset", "cresci15", "--device", "cpu", "--epochs", "2"])
    r1 = train_one(a, 1, data, y, J, "cpu")
    y2 = y.copy()
    for s in ("val", "test"):
        y2[data[s].idx] = 1 - y2[data[s].idx]
        data[s].y = 1 - data[s].y
    r2 = train_one(a, 1, data, y2, J, "cpu")
    assert all(torch.equal(x, z) for x, z in zip(r1["model"].state_dict().values(), r2["model"].state_dict().values()))


# -------------------------------------------------------------------------------- end to end
def test_end_to_end_seeded_extscore_and_variants(cr15_text, data_root):
    from bb.data import load_split
    from bb.harness import load_results
    from bb.extscore import make_synthetic_extscore
    from bb.methods import roberta_mlp
    args = ["--dataset", "cresci15", "--device", "cpu", "--epochs", "4"]
    roberta_mlp.main(args + ["--seeds", "0", "1"])
    roberta_mlp.main(args + ["--seeds", "0"])                       # seeded: reproducible
    make_synthetic_extscore("cresci15", tag="syn", coverage=0.8)
    roberta_mlp.main(args + ["--seeds", "0", "--with-extscore", "syn"])
    roberta_mlp.main(args + ["--seeds", "0", "--with-extscore", "syn", "--extscore-inject", "branch", "--extscore-branch-dim", "384"])
    roberta_mlp.main(args + ["--seeds", "0", "--select", "best_val"])
    base = load_results("roberta_mlp", "cresci15", where={"with_extscore": None, "select": "last_epoch"})
    assert len(base) == 3
    s0 = [r for r in base if r["seed"] == 0]
    assert s0[0]["metrics"]["test"] == s0[1]["metrics"]["test"]
    r = base[0]
    assert r["config"]["lr"] == 1e-4 and r["config"]["batch_size"] == 32 and r["config"]["dropout"] == 0.3
    assert r["epochs_run"] == 4 and r["selection"] == {"select": "last_epoch", "epoch": 3}
    t = r["metrics"]["test"]
    for k in ("f1", "f1_human", "precision_human", "recall_human", "roc_auc"):
        assert k in t
    assert t["n"] == len(load_split("cresci15", "test"))
    assert r["secondary"]["selection"]["select"] == "best_val" and "test" in r["secondary"]["metrics"]
    assert r["extscore"] == {"extscore": "none"} and r["gpu"]["placement"]["train"] == "device"
    extscore = load_results("roberta_mlp", "cresci15", where={"with_extscore": "syn", "extscore_inject": "fuse"})
    br = load_results("roberta_mlp", "cresci15", where={"with_extscore": "syn", "extscore_inject": "branch"})
    assert len(extscore) == 1 and len(br) == 1
    assert extscore[0]["n_params"] == r["n_params"] + 256
    assert br[0]["n_params"] == r["n_params"] + (2 * 384 + 384) + 384 * 128
    assert extscore[0]["extscore"]["tag"] == "syn" and extscore[0]["extscore"]["coverage"]["test"]["coverage"] < 1
    bv = load_results("roberta_mlp", "cresci15", where={"select": "best_val"})
    assert len(bv) == 1 and bv[0]["selection"]["select"] == "best_val"
    assert bv[0]["secondary"]["metrics"]["test"] == s0[0]["metrics"]["test"]   # last epoch, same seed
    assert bv[0]["metrics"]["test"] == s0[0]["secondary"]["metrics"]["test"]
    pr = np.load(data_root.parent / "runs" / "roberta_mlp" / "cresci15" / r["run"] / "preds_test.npz")
    assert len(pr["idx"]) == t["n"] and np.all((pr["prob"] >= 0) & (pr["prob"] <= 1))
