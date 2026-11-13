"""BIC (variant v2): sequence reader, batch closure (vs dense references), index-bug emulation,
model pieces, ExtScore widths, no-leak, end to end (synthetic fixtures, CPU)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from bb.io import save_npy, write_json
from bb.paths import dataset_paths


def fake_tables(dataset: str, seed: int = 0, signal: float = 1.0, cap_sel: str = "first20", max_tweets: int = 20):
    """des + tweet mean [N,768] and a ragged per-tweet table with 0..max_tweets rows per user."""
    from bb.data import load_labels
    y = load_labels(dataset).astype(np.float32)
    n = len(y)
    rng = np.random.default_rng(seed)
    d = dataset_paths(dataset).emb("roberta-base")
    for name in ("des", "tweets_first20_mean"):
        x = rng.standard_normal((n, 768)).astype(np.float32)
        x[:, :4] += signal * (2 * y[:, None] - 1)
        save_npy(d / f"{name}.f16.npy", x.astype(np.float16))
        write_json(d / f"{name}.meta.json", {"rows": n, "hf_commit": "fake"})
    cnt = rng.integers(0, max_tweets + 1, n)
    cnt[0] = 0
    off = np.zeros(n + 1, np.int64)
    np.cumsum(cnt, out=off[1:])
    seq = rng.standard_normal((int(off[-1]), 768)).astype(np.float32)
    owner = np.repeat(np.arange(n), cnt)
    seq[:, :4] += signal * (2 * y[owner, None] - 1)
    save_npy(d / f"tweets_seq_{cap_sel}.f16.npy", seq.astype(np.float16))
    save_npy(d / f"tweets_seq_{cap_sel}_offsets.i64.npy", off)
    save_npy(d / f"tweets_seq_{cap_sel}_rows.i64.npy", np.arange(int(off[-1]), dtype=np.int64))
    return cnt


@pytest.fixture
def cr15_bic(cr15):
    fake_tables("cresci15")
    return cr15


# ------------------------------------------------------------------------------------- text reader
def test_seq_reader_matches_memmap(cr15_bic):
    from bb.data import load_tweet_seq
    from bb.methods.bic.data import SeqReader
    seq, off, _ = load_tweet_seq("cresci15", "roberta-base", "first20")
    off = np.asarray(off)
    for (cap, drop), backend in [(cd, b) for cd in ((20, False), (7, True)) for b in ("native", "python")]:
        r = SeqReader(seq.filename, off, cap, drop_cache=drop, backend=backend)
        users = np.array([0, 5, 3, 3, len(off) - 2])
        buf = np.full((8, cap, 768), 7, np.float16)           # stale contents must be overwritten
        r.fill(buf, users)
        for k, u in enumerate(users):
            c = min(int(off[u + 1] - off[u]), cap)
            assert np.array_equal(buf[k, :c], seq[off[u]:off[u] + c])
            assert not buf[k, c:].any()
        assert not buf[0].any()                                # user 0 has no tweets
        r.close()
    many = np.random.default_rng(0).integers(0, len(off) - 1, 40)
    outs = []
    for th, backend in ((1, "python"), (4, "python"), (1, "native"), (8, "native")):   # all == serial python
        r = SeqReader(seq.filename, off, 20, io_threads=th, backend=backend)
        b = np.full((48, 20, 768), 3, np.float16)
        r.fill(b, many)
        outs.append((b, r.bytes_read))
        r.close()
    for b, n in outs[1:]:
        assert np.array_equal(outs[0][0], b) and outs[0][1] == n > 0
    assert (outs[0][0][40:] == 3).all()                        # rows past the batch untouched


# ------------------------------------------------------------------------------------- closure
def _toy_graph(n=30, e=80, seed=0):
    g = torch.Generator().manual_seed(seed)
    ei = torch.randint(0, n, (2, e), generator=g)
    ei = torch.cat([ei, ei[:, :5], torch.tensor([[3, 3, 17], [3, 3, 17]])], 1)  # duplicates + self-loop edges
    return ei.numpy()


def _dense_reference(layer, x, ei, targets, lists, bug=False):
    """GCN over ALL nodes with the star edges of ``targets`` (or the buggy local-id edges), then the
    per-user unbatched attention loop of the official MultiAttn (self first, then the list)."""
    from torch_geometric.nn import GCNConv  # noqa: F401
    t = torch.as_tensor(targets)
    src, dst = torch.as_tensor(ei[0]), torch.as_tensor(ei[1])
    m = torch.isin(dst, t)
    star = torch.stack([src[m], dst[m]])
    if bug:   # PyG NeighborSampler numbering: targets first, then new sources seed by seed (edge order)
        nid = list(targets)
        for i in targets:
            for s_ in src[dst == int(i)].tolist():
                if s_ not in nid:
                    nid.append(s_)
        loc = {v: i for i, v in enumerate(nid)}
        star = torch.tensor([[loc[s_] for s_ in star[0].tolist()], [loc[d_] for d_ in star[1].tolist()]],
                            dtype=torch.long).reshape(2, -1)
    from torch_geometric.utils import add_remaining_self_loops
    star, _ = add_remaining_self_loops(star, num_nodes=x.shape[0])   # GCNConv's default (dense reference)
    h = layer.gcn(x, star)
    out = h.clone()
    for i in targets:
        nb = lists.nbr_idx[lists.nbr_ptr[i]:lists.nbr_ptr[i + 1]]
        kv = h[torch.as_tensor(np.concatenate([[i], nb]).astype(np.int64))].unsqueeze(0)
        o, _ = layer.attn(h[i].view(1, 1, -1), kv, kv, need_weights=False)
        out[i] = o[0, 0]
    return out


@pytest.mark.parametrize("bug", [False, True])
def test_closure_layer_equals_dense_reference(bug):
    from bb.methods.bic.data import Closure, GraphLists
    from bb.methods.bic.model import GraphLayer
    n = 30
    ei = _toy_graph(n)
    lists = GraphLists.from_edge_index(ei, n)
    torch.manual_seed(0)
    layer = GraphLayer(16, heads=4).eval()
    x = torch.randn(n, 16)
    targets = np.array([3, 17, 0, 25, 9])
    ref = _dense_reference(layer, x, ei, targets, lists, bug=bug)
    bg = Closure(lists, index_bug=bug).build(targets)
    assert np.array_equal(bg.n_id[:len(targets)], targets) and len(np.unique(bg.n_id)) == len(bg.n_id)
    with torch.no_grad():
        h = layer(x[torch.as_tensor(bg.n_id)], torch.as_tensor(bg.edge_index), torch.as_tensor(bg.att_seg),
                  torch.as_tensor(bg.att_node), len(targets))
    assert torch.allclose(h, ref[torch.as_tensor(bg.n_id)], atol=1e-5)
    # neighbour list = both directions of every edge, duplicates kept
    i = 3
    want = sorted(ei[1][ei[0] == i].tolist() + ei[0][ei[1] == i].tolist())
    assert sorted(lists.nbr_idx[lists.nbr_ptr[i]:lists.nbr_ptr[i + 1]].tolist()) == want


def test_rgcn_closure_equals_dense_reference():
    """--gnn rgcn: RGCNConv on the closure star == RGCNConv over all nodes with the targets' typed
    in-edges (no self-loops), then the same list attention."""
    from bb.methods.bic.data import Closure, GraphLists
    from bb.methods.bic.model import GraphLayer
    n = 30
    ei = _toy_graph(n)
    et = np.random.default_rng(0).integers(0, 2, ei.shape[1])
    lists = GraphLists.from_edge_index(ei, n, edge_type=et)
    torch.manual_seed(0)
    layer = GraphLayer(16, heads=4, gnn="rgcn").eval()
    x = torch.randn(n, 16)
    targets = np.array([3, 17, 0, 25, 9])
    m = np.isin(ei[1], targets)
    with torch.no_grad():
        h = layer.gcn(x, torch.as_tensor(ei[:, m]), torch.as_tensor(et[m]))
        ref = h.clone()
        for i in targets:
            nb = lists.nbr_idx[lists.nbr_ptr[i]:lists.nbr_ptr[i + 1]]
            kv = h[torch.as_tensor(np.concatenate([[i], nb]).astype(np.int64))].unsqueeze(0)
            ref[i] = layer.attn(h[i].view(1, 1, -1), kv, kv, need_weights=False)[0][0, 0]
        bg = Closure(lists, rgcn=True).build(targets)
        out = layer(x[torch.as_tensor(bg.n_id)], torch.as_tensor(bg.edge_index), torch.as_tensor(bg.att_seg),
                    torch.as_tensor(bg.att_node), len(targets), torch.as_tensor(bg.edge_type))
    assert torch.allclose(out, ref[torch.as_tensor(bg.n_id)], atol=1e-5)


def test_closure_neighbor_cap_and_dedup():
    from bb.methods.bic.data import Closure, GraphLists
    n = 10
    src = np.array([0] * 12 + [1, 1])
    dst = np.array(list(range(1, 10)) + [1, 1, 2] + [0, 0])
    lists = GraphLists.from_edge_index(np.stack([src, dst]), n)
    d = GraphLists.from_edge_index(np.stack([src, dst]), n, dedup_neighbors=True)
    assert lists.nbr_ptr[1] - lists.nbr_ptr[0] == 14 and d.nbr_ptr[1] - d.nbr_ptr[0] == 9
    bg = Closure(lists, max_neighbors=4).build(np.array([0, 5]), np.random.default_rng(1))
    assert np.array_equal(np.bincount(bg.att_seg), [5, 2])            # self + min(cap, list length)
    b1 = Closure(lists, max_neighbors=4).build(np.array([0]), np.random.default_rng(1))
    b2 = Closure(lists, max_neighbors=4).build(np.array([0]), np.random.default_rng(1))
    assert np.array_equal(b1.n_id, b2.n_id)
    # GCN in-edges are never capped (plus one self-loop per closure node)
    assert bg.edge_index.shape[1] == (dst == 0).sum() + (dst == 5).sum() + len(bg.n_id)


# ------------------------------------------------------------------------------------- model pieces
def test_consistency_pool_and_interaction():
    from bb.methods.bic.model import Interaction, consistency_pool
    m = torch.rand(2, 201, 201)
    p = consistency_pool(m, 16)
    pad = torch.zeros(2, 208, 208)
    pad[:, :201, :201] = m
    assert p.shape == (2, 16, 16) and torch.allclose(p, pad.view(2, 16, 13, 16, 13).amax((2, 4)))
    assert consistency_pool(torch.rand(3, 21, 21), 16).shape == (3, 16, 16)
    torch.manual_seed(0)
    it = Interaction(8, "code")
    t, g = torch.randn(4, 8), torch.randn(4, 8)
    t2, g2 = it(t, g)
    a = (t * (t @ it.w_t.weight.T + it.w_t.bias)).sum(1)
    b = (g * t).sum(1)
    w = torch.softmax(torch.stack([a, b], 1), 1)
    assert torch.allclose(t2, w[:, :1] * t + w[:, 1:] * g, atol=1e-6)
    c = (g * (g @ it.w_g.weight.T + it.w_g.bias)).sum(1)
    w2 = torch.softmax(torch.stack([c, b], 1), 1)
    assert torch.allclose(g2, w2[:, :1] * g + w2[:, 1:] * t, atol=1e-6)    # pre-update t
    assert Interaction(8, "paper").w_t.bias is None


def _toy_batch(B=4, cap=5, V=12, num_dim=5, cat_dim=3, seed=0, late=False):
    g = torch.Generator().manual_seed(seed)
    ei = torch.stack([torch.randint(0, V, (20,), generator=g), torch.randint(0, B, (20,), generator=g)])
    seg = torch.cat([torch.arange(B), torch.tensor([0, 1, 1, 2, 3, 3, 3])])
    node = torch.cat([torch.arange(B), torch.randint(0, V, (7,), generator=g)])
    return dict(tweets=torch.randn(B, cap, 768, generator=g).half(), des0=torch.randn(B, 768, generator=g),
                gdes=torch.randn(V, 768, generator=g), gtweet=torch.randn(V, 768, generator=g),
                gnum=torch.randn(V, num_dim, generator=g), gcat=torch.randint(0, 2, (V, cat_dim), generator=g).float(),
                edge_index=ei, att_seg=seg, att_node=node, late=torch.randn(B, 2, generator=g) if late else None)


def test_model_params_ablations_and_extscore_widths():
    from bb.methods.bic.model import BIC, n_params
    torch.manual_seed(0)
    m = BIC(num_dim=5, cat_dim=3)
    d = 768
    text = 2 * (4 * d * d + 4 * d + 2 * (d * d + d) + 4 * d)          # MHA in/out, FFN, 2 LayerNorms
    graph = 2 * (d * d + d + 4 * d * d + 4 * d)                       # GCN + MHA_g
    enc = (3 + 1) * 192 + (5 + 1) * 192 + 2 * (768 + 1) * 192
    inter = 2 * (d * d + d)
    head = (512 * 256 + 256) + 2 * (d * 256 + 256) + (768 * 2 + 2)
    assert n_params(m) == text + graph + enc + inter + head            # ~19 M (spec 4.5)
    out = m.eval()(**_toy_batch())
    assert out.shape == (4, 2)
    torch.manual_seed(0)
    m0 = BIC(num_dim=5, cat_dim=3, extscore_head=0)
    assert all(torch.equal(x, y) for x, y in zip(m.state_dict().values(), m0.state_dict().values()))
    assert n_params(BIC(num_dim=7, cat_dim=3)) == n_params(m) + 2 * 192
    mh = BIC(num_dim=5, cat_dim=3, extscore_head=2).eval()
    assert n_params(mh) == n_params(m) + 2 * 2 and mh(**_toy_batch(late=True)).shape == (4, 2)
    for ab in ("wo_graph", "wo_text", "wo_interaction", "wo_consistency", "wo_graph_module", "wo_text_module"):
        x = BIC(num_dim=5, cat_dim=3, ablation=ab).eval()
        assert x(**_toy_batch()).shape == (4, 2)
    # paper ablations (Eq. 3): only the readout loses g / h; wo_graph keeps graph layer 1 (it feeds the
    # interaction) and drops the unused layer 2
    g1 = 4 * d * d + 4 * d + d * d + d
    assert n_params(BIC(num_dim=5, cat_dim=3, ablation="wo_graph")) == n_params(m) - g1 - (d * 256 + 256) - 256 * 2
    assert n_params(BIC(num_dim=5, cat_dim=3, ablation="wo_text")) == n_params(m) - (d * 256 + 256) - 256 * 2
    b = _toy_batch()
    torch.manual_seed(0)
    m.train()
    out_plain = m(**b)
    out_plain.sum().backward()
    g_plain = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    m.checkpoint_text = True
    torch.manual_seed(0)
    out_ck = m(**b)
    out_ck.sum().backward()
    assert torch.allclose(out_plain, out_ck, atol=1e-5)               # checkpointing: same function,
    assert all(torch.allclose(x, p.grad, atol=1e-4) for x, p in zip(g_plain, m.parameters()))  # same grads
    m.checkpoint_text = False
    m.zero_grad()
    wg = BIC(num_dim=5, cat_dim=3, ablation="wo_graph_module").eval()
    b2 = {**b, "gdes": b["gdes"] * 0, "edge_index": b["edge_index"][:, :0]}
    assert torch.allclose(wg(**b), wg(**b2))                            # ignores the graph
    wp = BIC(num_dim=5, cat_dim=3, ablation="wo_graph").eval()          # paper: graph still reaches the
    assert not torch.allclose(wp(**b), wp(**b2))                        # text head via the interaction
    wn = BIC(num_dim=5, cat_dim=3, ablation="wo_interaction").eval()
    b4 = {**b, "gdes": b["gdes"] * 0}
    wt = BIC(num_dim=5, cat_dim=3, ablation="wo_text").eval()
    assert not torch.allclose(wn(**b), wn(**b4)) and not torch.allclose(wt(**b), wt(**b4))
    m.eval()
    b3 = {**b, "edge_index": b["edge_index"][:, :3]}
    assert not torch.allclose(m(**b), m(**b3))                          # depends on the graph


# ------------------------------------------------------------------------------------- training
def _inputs(args, flip=None):
    from bb.features import load_features
    from bb.extscore import prepare_extscore
    from bb.methods.bic import Inputs, parse_args
    a = parse_args(args)
    F = load_features("cresci15")
    if flip is not None:
        y = F.label.copy()
        ev = np.concatenate([F.splits[s] for s in flip])
        y[ev] = 1 - y[ev]
        F = dataclasses.replace(F, label=y)
    sp = {s: F.splits[s] for s in ("train", "val", "test")}
    J = prepare_extscore(a, "cresci15", sp, labels=F.label)
    return a, F, Inputs(a, F, J, "cpu"), sp


def _fit(args, flip=None):
    from bb.harness import set_seed
    from bb.methods.bic import build_model, fit, predict
    from bb.methods.bic.data import Prefetcher
    a, F, inp, sp = _inputs(args, flip)
    pf = Prefetcher(inp.reader, inp.closure, 64, "cpu")
    set_seed(0)
    m = build_model(a, inp)
    r = fit(m, inp, pf, sp, a, 0)
    p = predict(m, inp, pf, sp["test"], a)
    pf.close()
    return r, p


def test_no_label_leak(cr15_bic):
    """Flipping every val/test label leaves training bit-identical (select last, fixed epochs);
    with the default best-val selection, flipping test labels changes nothing and flipping val labels
    changes only which epoch is selected, never the training trajectory."""
    base = ["--dataset", "cresci15", "--epochs", "3", "--batch-size", "8", "--eval-batch-size", "8",
            "--device", "cpu", "--tweets-cap", "6"]
    last = base + ["--select", "last", "--patience", "0"]
    r0, p0 = _fit(last)
    r1, p1 = _fit(last, flip=("val", "test"))
    assert [h["train_loss"] for h in r0["history"]] == [h["train_loss"] for h in r1["history"]]
    assert np.array_equal(p0, p1)
    b0, q0 = _fit(base)
    b1, q1 = _fit(base, flip=("test",))
    b2, _ = _fit(base, flip=("val",))
    assert np.array_equal(q0, q1) and b0["selection"] == b1["selection"]
    assert [h["train_loss"] for h in b0["history"]] == [h["train_loss"] for h in b2["history"]]
    assert [h["val_acc"] for h in b2["history"]] == pytest.approx([1 - h["val_acc"] for h in b0["history"]])


def test_eval_mode_bug_emulation(cr15_bic):
    """--reproduce-eval-mode-bug (official train.py never calls model.train() again after the first
    evaluate()): epoch 0 is unchanged and later epochs train without dropout, so with dropout 0 the
    flag changes nothing at all."""
    base = ["--dataset", "cresci15", "--epochs", "2", "--batch-size", "8", "--eval-batch-size", "8",
            "--device", "cpu", "--tweets-cap", "6", "--patience", "0"]
    loss = lambda r: [h["train_loss"] for h in r["history"]]  # noqa: E731
    r0, _ = _fit(base + ["--dropout", "0.5"])
    r1, _ = _fit(base + ["--dropout", "0.5", "--reproduce-eval-mode-bug"])
    assert loss(r0)[0] == loss(r1)[0] and loss(r0)[1] != loss(r1)[1]
    z0, p0 = _fit(base + ["--dropout", "0"])
    z1, p1 = _fit(base + ["--dropout", "0", "--reproduce-eval-mode-bug"])
    assert loss(z0) == loss(z1) and np.array_equal(p0, p1)


def test_end_to_end_seeded_extscore_ablation_and_flags(cr15_bic, data_root):
    from bb.harness import load_results
    from bb.extscore import make_synthetic_extscore
    from bb.methods import bic
    args = ["--dataset", "cresci15", "--seeds", "0", "--epochs", "2", "--device", "cpu", "--batch-size", "8",
            "--eval-batch-size", "16"]
    bic.main(args)
    bic.main(args)
    make_synthetic_extscore("cresci15", tag="syn", coverage=0.8)
    bic.main(args + ["--with-extscore", "syn"])
    bic.main(args + ["--with-extscore", "syn", "--extscore-at", "graph_num+head"])
    bic.main(args + ["--ablation", "wo_graph"])
    bic.main(args + ["--reproduce-index-bug", "--select", "last", "--interaction", "paper"])
    bic.main(args + ["--gnn", "rgcn", "--reproduce-eval-mode-bug", "--ablation", "wo_text"])
    rs = load_results("bic", "cresci15")
    base = [r for r in rs if r["config"]["variant"] == "v2.botrgcn.seqfirst20c20.bs8.ep2" and r["config"]["with_extscore"] is None]
    extscore = [r for r in rs if r["config"]["with_extscore"] == "syn" and r["config"]["extscore_at"] == "graph_num"]
    both = [r for r in rs if r["config"]["with_extscore"] == "syn" and r["config"]["extscore_at"] == "graph_num+head"]
    wog = [r for r in rs if r["config"]["ablation"] == "wo_graph"]
    bug = [r for r in rs if r["config"]["reproduce_index_bug"]]
    rg = [r for r in rs if r["config"]["gnn"] == "rgcn"]
    assert len(rg) == 1 and rg[0]["config"]["variant"] == "v2.botrgcn.seqfirst20c20.wo_text.evalmodebug.rgcn.bs8.ep2"
    assert len(base) == 2 and len(extscore) == len(both) == len(wog) == len(bug) == 1
    assert base[0]["metrics"]["test"] == base[1]["metrics"]["test"]           # seeded runs reproduce
    assert extscore[0]["n_params"] == base[0]["n_params"] + 2 * 192
    assert both[0]["n_params"] == base[0]["n_params"] + 2 * 192 + 4
    assert base[0]["extscore"] == {"extscore": "none"} and extscore[0]["extscore"]["tag"] == "syn"
    assert bug[0]["config"]["variant"] == "v2.botrgcn.seqfirst20c20.interpaper.indexbug.selectlast.bs8.ep2"
    assert bug[0]["selection"]["select"] == "last_epoch" and base[0]["selection"]["select"] == "best_val"
    assert {"train", "val", "test"} <= set(base[0]["metrics"]) and base[0]["seq"]["cap"] == 20
    assert base[0]["seq"]["users_no_tweets"] >= 1


def test_measure_mode(cr15_bic):
    from bb.methods import bic
    rep = bic.main(["--dataset", "cresci15", "--device", "cpu", "--measure-batches", "1", "--batch-size", "4"])[0]
    assert rep["seconds_per_train_batch"] > 0 and rep["train_batches_per_epoch"] > 0


def test_val_probs_reused_equal_fresh_pass(cr15_bic):
    """fit returns the selected state's val predictions (no repeated final val pass); on CPU they are
    bit-identical to a fresh pass after restore, for both selection modes."""
    from bb.methods.bic import build_model, fit, predict
    from bb.methods.bic.data import Prefetcher
    from bb.harness import set_seed
    base = ["--dataset", "cresci15", "--epochs", "3", "--batch-size", "8", "--eval-batch-size", "8",
            "--device", "cpu", "--tweets-cap", "6", "--patience", "0"]
    for extra in ([], ["--select", "last"]):
        a, F, inp, sp = _inputs(base + extra)
        pf = Prefetcher(inp.reader, inp.closure, 64, "cpu")
        set_seed(0)
        m = build_model(a, inp)
        r = fit(m, inp, pf, sp, a, 0)
        assert not r["step"].compiled and np.array_equal(r["val_probs"], predict(m, inp, pf, sp["val"], a))
        pf.close()


def test_compile_auto_rule():
    from bb.methods.bic import parse_args, use_compile
    tw22 = parse_args(["--dataset", "twibot22"])
    assert use_compile(tw22, "cuda") and not use_compile(tw22, "cpu")
    assert not use_compile(parse_args(["--dataset", "twibot22", "--compile", "off"]), "cuda")
    assert not use_compile(parse_args(["--dataset", "twibot22", "--gnn", "rgcn"]), "cuda")
    assert not use_compile(parse_args(["--dataset", "twibot20"]), "cuda")
    assert use_compile(parse_args(["--dataset", "twibot20", "--compile", "on"]), "cuda")
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "twibot22", "--gnn", "rgcn", "--compile", "on"])
