"""Metrics, training utilities, run bookkeeping, sampler, ExtScore interface."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from bb.harness import BestCheckpoint, EarlyStopper, Run, aggregate, class_weights, compute_metrics, load_results
from bb.harness.sampler import NeighborSampler
from bb.extscore import (ExtScoreLeakageError, add_extscore_args, load_extscore, make_synthetic_extscore, neural_transform,
                    prepare_extscore, tree_transform, validate_extscore, write_extscore)


# ------------------------------------------------------------------------------- metrics
def test_metrics_match_sklearn():
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.round(rng.random(500), 2)                                     # many ties
    m = compute_metrics(y, p)
    pred = (p >= 0.5).astype(int)
    assert m["f1"] == pytest.approx(f1_score(y, pred))
    assert m["f1_human"] == pytest.approx(f1_score(y, pred, pos_label=0))
    assert m["macro_f1"] == pytest.approx(f1_score(y, pred, average="macro"))
    assert m["precision"] == pytest.approx(precision_score(y, pred))
    assert m["recall"] == pytest.approx(recall_score(y, pred))
    assert m["accuracy"] == pytest.approx(accuracy_score(y, pred))
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert np.isnan(compute_metrics(np.ones(3), np.ones(3))["roc_auc"])
    with pytest.raises(ValueError):
        compute_metrics(np.array([0, -1]), np.array([0.1, 0.2]))


# ------------------------------------------------------------------------------- training utils
def test_early_stopper():
    es = EarlyStopper(patience=2, mode="max")
    for v in [0.1, 0.3, 0.2, 0.25, 0.29]:
        es.step(v)
    assert es.best == 0.3 and es.best_epoch == 1 and es.should_stop


def test_best_checkpoint_is_a_copy():
    import torch
    m = torch.nn.Linear(2, 1)
    ck = BestCheckpoint("best_val")
    ck.update(m, 0.9, 0)
    saved = m.weight.detach().clone()
    with torch.no_grad():
        m.weight.add_(1.0)                                               # in-place, like an optimiser step
    ck.update(m, 0.5, 1)
    ck.restore(m)
    assert torch.equal(m.weight, saved) and ck.best_epoch == 0
    last = BestCheckpoint("last_epoch")
    last.update(m, 0.9, 0)
    with torch.no_grad():
        m.weight.add_(1.0)
    w = m.weight.detach().clone()
    last.restore(m)
    assert torch.equal(m.weight, w)


def test_class_weights():
    y = np.array([0] * 90 + [1] * 10 + [-1] * 5)
    assert class_weights(y) is None
    b = class_weights(y, "balanced")
    assert b == pytest.approx([100 / 180, 100 / 20])
    assert class_weights(y, "1,3").tolist() == [1, 3]


# ------------------------------------------------------------------------------- runs
def test_run_writes_result_and_provenance(data_root, tw20):
    with Run("toy", "twibot20", {"a": 1}, seed=0, run_name="r0") as run:
        run.log_epoch(0, loss=1.0)
        run.add_predictions("test", np.arange(3), np.array([0.1, 0.9, 0.5]), np.array([0, 1, 1]))
        res = run.finish({"test": compute_metrics([0, 1, 1], [0.1, 0.9, 0.5])})
    d = run.dir
    assert (d / "config.json").exists() and (d / "epochs.jsonl").read_text().count("\n") == 1
    r = json.loads((d / "result.json").read_text())
    assert r["status"] == "ok" and r["memory"]["peak_rss_bytes"] > 0 and r["wall_seconds"] >= 0
    assert "twibot20/users.parquet" in r["provenance"]["data"]
    with Run("toy", "twibot20", {"a": 2}, seed=0, run_name="r1") as run2:
        run2.finish({"test": res["metrics"]["test"]})
    assert run2.provenance != run.provenance                             # config enters the hash
    with pytest.raises(RuntimeError):
        with Run("toy", "twibot20", {"a": 3}, seed=1, run_name="r2"):
            raise RuntimeError("boom")
    assert json.loads((d.parent / "r2" / "result.json").read_text())["status"] == "failed"
    rs = load_results("toy", "twibot20")
    assert len(rs) == 2 and aggregate(rs)["f1"]["n"] == 2


# ------------------------------------------------------------------------------- sampler
def _toy_csc(n=300, e=3000, seed=0):
    from bb.data.graph import CSR
    rng = np.random.default_rng(seed)
    src, dst = rng.integers(0, n, e), rng.integers(0, n // 3, e)         # hubs among low ids
    order = np.lexsort((src, dst))
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(dst, minlength=n), out=indptr[1:])
    return CSR(indptr, src[order].astype(np.int32), np.arange(e), np.zeros(e, np.int8)), src, dst


def test_sampler_correct_bounded_deterministic():
    csc, _, _ = _toy_csc()
    s = NeighborSampler(csc, [5, 3])
    seeds = np.array([0, 1, 2, 250])
    g = s.sample(seeds, rng_seed=7)
    assert g.n_id[:4].tolist() == seeds.tolist() and len(np.unique(g.n_id)) == g.num_nodes
    src, dst = g.n_id[g.edge_index[0]], g.n_id[g.edge_index[1]]
    ip = np.asarray(csc.indptr)
    assert np.all(np.asarray(csc.indices)[g.e_id] == src)               # each edge is real
    assert np.all((ip[dst] <= g.e_id) & (g.e_id < ip[dst + 1]))
    assert len(np.unique(g.e_id)) == len(g.e_id)                          # no replacement
    deg_in = np.bincount(g.edge_index[1], minlength=g.num_nodes)
    assert deg_in[:4].max() <= 5 and deg_in[4:].max(initial=0) <= 3
    g2 = s.sample(seeds, rng_seed=7)
    assert np.array_equal(g.n_id, g2.n_id) and np.array_equal(g.e_id, g2.e_id)
    full = NeighborSampler(csc, [-1]).sample(seeds)
    assert full.num_edges == int(sum(ip[v + 1] - ip[v] for v in seeds))
    rep = s.dry_run(np.arange(100), batch_size=16, n_batches=5, feature_bytes_per_node=3072)
    assert rep["nodes"]["p100"] >= rep["nodes"]["p50"] >= 16 and "gathered_feature_MB_p99" in rep
    with pytest.raises(MemoryError):
        NeighborSampler(csc, [-1, -1], max_nodes=5).sample(seeds)


def test_sampler_iter_batches_cover_seeds():
    csc, _, _ = _toy_csc()
    s = NeighborSampler(csc, [2])
    seeds = np.arange(50, 90)
    got = np.concatenate([b.n_id[:b.batch_size] for b in s.iter_batches(seeds, 16, epoch_seed=3)])
    assert sorted(got.tolist()) == seeds.tolist()


# ------------------------------------------------------------------------------- ExtScore
def test_extscore_roundtrip_and_transforms(tw20):
    from bb.data import load_split
    splits = {s: np.asarray(load_split("twibot20", s)) for s in ("train", "val", "test", "support")}
    idx = np.array([0, 1, 2, 3, 6, 9])
    write_extscore("twibot20", "t", idx, [0.9, 0.1, 0.5, 0.0, 0.7, 1.0], fold=[0, 1, 0, 1, -1, -1],
              q={"a": [1, 2, 3, 4, 5, 6]}, meta={"model": "x"})
    j = load_extscore("twibot20", "t")
    assert j.n == 19 and j.observed.sum() == 6 and np.isnan(j.p[4]) and "q_a" in j.q
    x, prm = neural_transform(j, splits["train"])
    assert prm["n_fit"] == 4 and x.shape == (19, 2) and x[4].tolist() == [0.0, 1.0]
    tr_z = x[[0, 1, 2, 3], 0]
    assert tr_z.mean() == pytest.approx(0, abs=1e-6) and tr_z.std() == pytest.approx(1, abs=1e-5)
    assert np.isfinite(x).all()                                           # p = 0 / 1 clipped
    # val/test values do not affect mu/sd (fit on train only)
    write_extscore("twibot20", "t2", idx, [0.9, 0.1, 0.5, 0.0, 0.01, 0.02], fold=[0, 1, 0, 1, -1, -1])
    _, prm2 = neural_transform(load_extscore("twibot20", "t2"), splits["train"])
    assert prm2["mu"] == prm["mu"] and prm2["sd"] == prm["sd"]
    tt = tree_transform(j)
    assert tt[4].tolist() == [-1.0, 1.0] and tt[0, 0] == pytest.approx(0.9)


def test_extscore_guard(tw20):
    from bb.data import load_split
    splits = {s: np.asarray(load_split("twibot20", s)) for s in ("train", "val", "test")}
    write_extscore("twibot20", "leaky", [0, 1, 6], [0.9, 0.1, 0.3])           # train rows without folds
    assert validate_extscore(load_extscore("twibot20", "leaky"), splits)
    with pytest.raises(ExtScoreLeakageError):
        prepare_extscore("leaky", "twibot20", splits)
    write_extscore("twibot20", "blind", [0, 1, 6], [0.9, 0.1, 0.3], meta={"label_blind": True})
    assert validate_extscore(load_extscore("twibot20", "blind"), splits) == []
    write_extscore("twibot20", "opt", [0, 1, 6], [0.9, 0.1, 0.3], fold=[0, 1, -1], meta={"optimised_on": [9, 3]})
    assert any("split 'test'" in p for p in validate_extscore(load_extscore("twibot20", "opt"), splits))
    from bb.data import load_undirected
    und = load_undirected("twibot20", ["following", "follower"])        # neighbours of 10: {8, 9}
    assert sorted(und.row(10).tolist()) == [8, 9]
    probs = validate_extscore(load_extscore("twibot20", "opt"), {**splits, "test": np.array([10])}, neighbours=und)
    assert probs == ["decision model was optimised on 1 graph neighbours of evaluation users"]


def test_extscore_common_interface(tw20):
    from bb.data import load_labels, load_split
    splits = {s: np.asarray(load_split("twibot20", s)) for s in ("train", "val", "test")}
    y = load_labels("twibot20")
    ap = argparse.ArgumentParser()
    add_extscore_args(ap)
    off = prepare_extscore(ap.parse_args([]), "twibot20", splits, labels=y)
    x = np.zeros((19, 5), np.float32)
    assert not off.enabled and off.append(x) is x and off.width == 0 and off.report() == {"extscore": "none"}
    make_synthetic_extscore("twibot20", tag="syn", coverage=1.0, seed=1)
    on = prepare_extscore(ap.parse_args(["--with-extscore", "syn"]), "twibot20", splits, labels=y)
    assert on.enabled and on.append(x).shape == (19, 7) and on.append(x[:3], rows=np.arange(3)).shape == (3, 7)
    assert on.cov["train"]["coverage"] == 1.0
    rep = on.report()
    assert rep["column_only_baseline"]["test"]["n"] == len(splits["test"]) and "mu" in rep["transform_params"]
    assert prepare_extscore(ap.parse_args(["--extscore", "p", "--extscore-tag", "syn"]), "twibot20", splits).enabled
    with pytest.raises(ValueError):
        prepare_extscore(ap.parse_args(["--extscore", "p"]), "twibot20", splits)


def test_synthetic_extscore_properties(tw20):
    j = make_synthetic_extscore("twibot20", tag="s", signal=3.0, coverage=0.5, seed=0, write=False)
    lab = np.array([0, 1] * 5)
    assert j.meta["synthetic"] and np.all(np.isnan(j.p[11:]))           # support unscored by default
    assert np.all(j.fold[:6] >= 0) and np.all(j.fold[6:] == -1)
