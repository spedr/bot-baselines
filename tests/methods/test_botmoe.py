"""BotMoE (released-code architecture): MoE layer, model structure, metadata inputs, ExtScore plumbing,
no-leak and end-to-end on synthetic fixtures (CPU)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from test_botrgcn import _toy_inputs, cr15_text, fake_text  # noqa: F401  (fixture re-export)


def _model(**kw):
    from bb.methods.botmoe.model import build_model
    base = dict(num_dim=5, cat_dim=3, meta_dim=8, align=16, out_dim=4)
    base.update(kw)
    return build_model(**base)


# ------------------------------------------------------------------------------------------- MoE
def test_moe_k1_is_selected_expert_and_gate_learns_only_from_balance():
    from bb.methods.botmoe.model import MLP, NoisyTopKMoE
    torch.manual_seed(0)
    moe = NoisyTopKMoE(6, 3, [MLP(6, 3, 8) for _ in range(3)], k=1)
    torch.nn.init.normal_(moe.w_gate)
    x = torch.randn(50, 6)
    moe.eval()
    y, loss, r = moe(x, x)
    sel = (x @ moe.w_gate).argmax(1)
    assert torch.equal(r, sel)
    ref = torch.stack([moe.experts[int(j)](x[i:i + 1])[0] for i, j in enumerate(sel)])
    assert torch.allclose(y, ref, atol=1e-6)
    # training: the CE-like task loss gives no gradient to the gate (g == 1), the balance loss does
    moe.train()
    y, loss, _ = moe(x, x)
    g_task = torch.autograd.grad(y.sum(), moe.w_gate, allow_unused=True)[0]
    assert g_task is None or torch.count_nonzero(g_task) == 0
    y, loss, _ = moe(x, x)
    assert torch.count_nonzero(torch.autograd.grad(loss, moe.w_gate)[0]) > 0


@pytest.mark.parametrize("k", [1, 2])
def test_dense_dispatch_equals_sparse_outputs_and_grads(k):
    from bb.methods.botmoe.model import MLP, NoisyTopKMoE
    torch.manual_seed(3)
    moe = NoisyTopKMoE(6, 3, [MLP(6, 3, 8) for _ in range(3)], k=k)
    torch.nn.init.normal_(moe.w_gate)
    x = torch.randn(40, 6)
    res = []
    for mode in ("dense", "sparse"):
        moe.dispatch = mode
        moe.zero_grad()
        torch.manual_seed(7)                          # same gating noise
        y, loss, r = moe.train()(x, x)
        (y.pow(2).sum() + loss).backward()
        res.append((y.detach(), r, [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                                    for p in moe.parameters()]))
    (y0, r0, g0), (y1, r1, g1) = res
    assert torch.equal(r0, r1) and torch.allclose(y0, y1, atol=1e-6)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(g0, g1))


def test_moe_k2_logsumexp_combine_and_single_expert_no_loss():
    from bb.methods.botmoe.model import MLP, NoisyTopKMoE
    torch.manual_seed(1)
    moe = NoisyTopKMoE(4, 2, [MLP(4, 2, 5) for _ in range(3)], k=2).eval()
    torch.nn.init.normal_(moe.w_gate)
    x = torch.randn(20, 4)
    y, _, _ = moe(x, x)
    logits = x @ moe.w_gate
    v, i = logits.topk(2, 1)
    g = torch.softmax(v, 1)
    outs = torch.stack([e(x) for e in moe.experts], 1)                 # [B, n, O]
    sel = outs.gather(1, i.unsqueeze(-1).expand(-1, -1, 2))
    ref = torch.log((g.unsqueeze(-1) * sel.exp()).sum(1))               # released combine: log(sum g exp(E))
    assert torch.allclose(y, ref, atol=1e-5)
    one = NoisyTopKMoE(4, 2, [MLP(4, 2, 5)], k=1).train()
    assert float(one(x, x)[1]) == 0.0


# ----------------------------------------------------------------------------------------- model
def test_structure_init_and_tied_rgcn():
    from torch_geometric.nn import RGCNConv
    m = _model()
    assert len([x for x in m.rgcn.modules() if isinstance(x, RGCNConv)]) == 1     # one conv applied twice
    for moe in m.moes().values():
        assert torch.all(moe.w_gate == 0.1) and torch.all(moe.w_noise == 0.1)
    assert [moe.n for moe in m.moes().values()] == [3, 3, 3, 2, 3]
    assert m.rgt.moe.experts[0].fc1.out_features == 64 and m.gcn.moe.experts[0].fc1.out_features == 64  # E = 4A
    assert m.classifier.in_features == 5 * 4 + 36
    names = [n.split(".")[0] for n, _ in m.named_parameters()]
    first = list(dict.fromkeys(names))
    assert first[:8] == ["align_des", "align_tweet", "align_num", "align_cat", "align_meta", "classifier", "mha", "norm1"]
    assert first.index("rgcn") < first.index("rgt") < first.index("gcn") < first.index("text_moe") < first.index("meta_moe")


def test_eval_rows_exact_and_graph_dependence():
    m = _model().eval()
    g = _toy_inputs(n=40)
    meta = torch.randn(40, 8)
    full, aux = m(g, meta)
    part, _ = m(g, meta, rows=7)
    assert torch.allclose(full[:7], part, atol=1e-5) and set(aux["route"]) == {"gcn", "rgcn", "rgt", "text", "meta"}
    g2 = g.with_edges(g.edge_index[:, :5], g.edge_type[:5])
    assert not torch.allclose(full, m(g2, meta)[0])


def test_grad_checkpoint_is_exact():
    """Checkpointed convolutions (dropout RNG restored on recompute) give the same loss and gradients."""
    g = _toy_inputs(n=40, e=200)
    meta = torch.randn(40, 8)
    grads = []
    for ck in (False, True):
        torch.manual_seed(0)
        m = _model().train()
        m.set_checkpoint(ck)
        torch.manual_seed(1)
        out, aux = m(g, meta)
        (torch.nn.functional.cross_entropy(out, g.y) + aux["balance"]).backward()
        grads.append([p.grad.clone() for p in m.parameters() if p.grad is not None])
    assert len(grads[0]) == len(grads[1])
    assert all(torch.allclose(a, b, atol=1e-5, rtol=1e-4) for a, b in zip(*grads))


def test_meta_zeros_gives_constant_meta_token():
    """Released behaviour: zero metadata input -> the metadata token is the same for every node."""
    m = _model(meta_dim=9).eval()
    toks = []
    h = m.meta_out.register_forward_hook(lambda mod, i, o: toks.append(o))
    g = _toy_inputs(n=30)
    m(g, torch.zeros(30, 9))
    m(g, torch.randn(30, 9))
    h.remove()
    assert torch.allclose(toks[0], toks[0][:1].expand_as(toks[0]))         # zeros: one token for all
    assert not torch.allclose(toks[1], toks[1][:1].expand_as(toks[1]))     # real input: per-user tokens


def test_extscore_variants_widths_and_width0_is_baseline():
    torch.manual_seed(0)
    base = _model()
    nb = sum(p.numel() for p in base.parameters())
    for at in ("metadata", "num", "token", "head", "gate_only", "metadata+num"):
        torch.manual_seed(0)
        z = _model(extscore_width=0, extscore_at=at)
        assert all(torch.equal(x, y) for x, y in zip(z.state_dict().values(), base.state_dict().values()))
    A = 16
    assert sum(p.numel() for p in _model(meta_dim=10, extscore_width=2, extscore_at="metadata").parameters()) == nb + 2 * A
    assert sum(p.numel() for p in _model(num_dim=7, extscore_width=2, extscore_at="num").parameters()) == nb + 2 * A
    tok = _model(extscore_width=2, extscore_at="token")
    assert tok.classifier.in_features == 6 * 4 + 36
    assert sum(p.numel() for p in _model(extscore_width=2, extscore_at="head").parameters()) == nb + 2 * 2
    go = _model(extscore_width=2, extscore_at="gate_only")
    assert sum(p.numel() for p in go.parameters()) == nb + 5 * 2 * 2 * 3 - 2 * 2 * 1   # 2 extra rows of W_g, W_n
    g = _toy_inputs(n=30)
    extscore = torch.randn(30, 2)
    for mdl in (tok.eval(), go.eval(), _model(extscore_width=2, extscore_at="head").eval()):
        assert mdl(g, torch.randn(30, 8), extscore)[0].shape == (30, 2)


# ------------------------------------------------------------------------------------ meta inputs
def test_hayawi_helpers():
    from bb.methods.botmoe.meta import bigram_freq, entropy
    assert bigram_freq("a") == 0 and bigram_freq("abab") == 3 / 2
    assert entropy("") == 0 and abs(entropy("ab", buggy=False) - 1.0) < 1e-12
    assert abs(entropy("ab", buggy=True) - 0.5) < 1e-12        # 2 * (-(1/4) * log2(1/2))


def test_meta_inputs_twibot20_released_layout(tw20):
    from bb.features import load_features
    from bb.methods.botmoe.meta import meta_input
    fake_text("twibot20", [("distilroberta-base", "des"), ("roberta-base", "tweets_all_mean")])
    F = load_features("twibot20")
    b8, _ = meta_input("botrgcn8", F)
    assert b8.shape == (F.n, 8) and np.allclose(b8[:, :5], F.num_prop) and np.allclose(b8[:, 5:], F.cat_prop)
    h, info = meta_input("hayawi9", F)
    lab = np.flatnonzero(F.label >= 0)
    assert h.shape == (F.n, 9) and np.allclose(h[lab].mean(0), 0, atol=1e-5)
    rel, _ = meta_input("released", F)
    n_lab = len(lab)
    assert np.allclose(rel[F.n - n_lab:], h[lab]) and not rel[:F.n - n_lab].any()
    assert not rel[lab[lab < F.n - n_lab]].any()           # labelled rows before the tail see zeros
    assert not meta_input("zeros", F)[0].any()


# --------------------------------------------------------------------------------- training / leak
def _fit(flip_val: bool, flip_test: bool, epochs: int, cfg_over=None):
    from bb.data import load_in_merged
    from bb.features import load_features
    from bb.gnn import to_device
    from bb.methods.botmoe.meta import meta_input
    from bb.methods.botmoe.train import Inputs, TrainConfig, fit
    F = load_features("cresci15")
    sp = {s: F.splits[s] for s in ("train", "val", "test")}
    g = to_device(F, "cpu", edges=False)
    y = F.label.copy()
    for flip, s in ((flip_val, "val"), (flip_test, "test")):
        if flip:
            g.y[sp[s]] = 1 - g.y[sp[s]]
            y[sp[s]] = 1 - y[sp[s]]
    csc = load_in_merged("cresci15", F.relations_by_type())
    X = Inputs(g, torch.from_numpy(meta_input("botrgcn8", F)[0]), None,
               torch.from_numpy(np.asarray(csc.etype, dtype=np.int64)))
    torch.manual_seed(0)
    m = _model(num_dim=5, cat_dim=1, meta_dim=6)
    cfg = TrainConfig(epochs=epochs, lr=1e-3, batch_size=8, eval_batch_size=16, patience=-1, unfreeze_acc=None,
                      unfreeze_epoch=2, **(cfg_over or {}))
    return m, fit(m, X, y, sp, cfg, csc=csc)


def test_no_label_leak(cr15_text):
    """Training reads only train labels; model selection reads val; test labels are never read."""
    _, a = _fit(False, False, 3)
    _, b = _fit(False, True, 3)
    assert np.allclose(a.prob, b.prob, equal_nan=True, atol=1e-6) and a.selection == b.selection
    _, c = _fit(False, False, 1)
    _, d = _fit(True, True, 1)                      # one epoch: selection is trivially epoch 0
    assert np.allclose(c.prob, d.prob, equal_nan=True, atol=1e-6)


def test_graph_gates_frozen_until_unfreeze(cr15_text):
    m, r = _fit(False, False, 1)
    assert torch.all(m.gcn.moe.w_gate == 0.1) and r.unfreeze_epoch is None
    m2, r2 = _fit(False, False, 3)
    assert r2.unfreeze_epoch == 1
    assert not torch.all(m2.text_moe.w_gate == 0.1)            # text / meta gates train from the start
    lrs = [h["lr"] for h in r2.history]
    assert lrs[0] == 1e-3 and abs(lrs[-1] - 1e-4) < 1e-12
    # --no-gate-freeze: graph gates train from epoch 0, the lr still drops at the trigger
    m3, r3 = _fit(False, False, 3, {"freeze_graph_gates": False})
    assert not torch.all(m3.gcn.moe.w_gate == 0.1) and r3.history[0]["gates_frozen"] is False
    assert r3.unfreeze_epoch == 1 and abs(r3.history[-1]["lr"] - 1e-4) < 1e-12


# ------------------------------------------------------------------------------------- end to end
def test_end_to_end_seeded_and_extscore(cr15_text, data_root):
    from bb.harness import load_results
    from bb.extscore import make_synthetic_extscore
    from bb.methods import botmoe
    args = ["--dataset", "cresci15", "--seeds", "0", "--epochs", "2", "--device", "cpu", "--align", "16",
            "--batch-size", "16", "--lr", "1e-3"]
    botmoe.main(args)
    botmoe.main(args)
    make_synthetic_extscore("cresci15", tag="syn", coverage=0.8)
    botmoe.main(args + ["--with-extscore", "syn"])
    botmoe.main(args + ["--with-extscore", "syn", "--extscore-inject", "num"])
    botmoe.main(args + ["--with-extscore", "syn", "--extscore-inject", "token"])
    botmoe.main(args + ["--meta-input", "zeros"])
    base = load_results("botmoe", "cresci15", where={"with_extscore": None, "meta_input": "botrgcn8"})
    meta = load_results("botmoe", "cresci15", where={"with_extscore": "syn", "extscore_inject": "metadata"})
    num = load_results("botmoe", "cresci15", where={"with_extscore": "syn", "extscore_inject": "num"})
    tok = load_results("botmoe", "cresci15", where={"with_extscore": "syn", "extscore_inject": "token"})
    zer = load_results("botmoe", "cresci15", where={"meta_input": "zeros"})
    assert len(base) == 2 and len(meta) == len(num) == len(tok) == len(zer) == 1
    assert base[0]["metrics"]["test"] == base[1]["metrics"]["test"]           # seeded runs reproduce
    nb = base[0]["n_params"]
    assert meta[0]["n_params"] == nb + 2 * 16 and num[0]["n_params"] == nb + 2 * 16
    assert tok[0]["n_params"] == nb + (2 * 4 + 4) + 4 * 2 + 2 * 4             # token Linear, classifier, bn2
    assert zer[0]["n_params"] == nb + 3 * 16                                   # meta width 6 -> 9
    assert base[0]["extscore"] == {"extscore": "none"} and meta[0]["extscore"]["coverage"]["test"]["coverage"] < 1
    assert set(base[0]["expert_load"]["test"]) == {"gcn", "rgcn", "rgt", "text", "meta"}
    assert base[0]["config"]["variant"] == "code.botrgcn8.lr0.001.bs16" and tok[0]["config"]["variant"].endswith("extscoretoken")


@pytest.mark.parametrize("ckpt", [False, True])
def test_chunked_rgt_equals_fused(ckpt):
    """ChunkedRGTLayer (small chunks, optional per-chunk checkpoint) == bb.methods.rgt fused RGTLayer:
    outputs and parameter gradients (eval mode: no attention dropout)."""
    from bb.methods.botmoe.rgt_chunked import ChunkedRGTLayer
    from bb.methods.rgt.model import RGTLayer
    torch.manual_seed(0)
    ref = RGTLayer(2, 16, 2, 2, 0.5)
    ch = ChunkedRGTLayer(2, 16, 2, 2, 0.5, max_edges=7)
    ch.load_state_dict(ref.state_dict())
    ch.ckpt = ckpt
    g = torch.Generator().manual_seed(1)
    n, e = 30, 90
    ei = torch.randint(0, n, (2, e), generator=g)
    ei[1, :10] = 3                                         # a big segment spanning several chunks
    et = torch.randint(0, 2, (e,), generator=g)
    x = torch.randn(n, 16, generator=g)
    outs, grads = [], []
    for layer in (ref.eval(), ch.eval()):
        layer.zero_grad()
        xx = x.clone().requires_grad_(True)
        y = layer(xx, ei, et)
        y.pow(2).sum().backward()
        outs.append(y.detach())
        grads.append([xx.grad] + [p.grad for p in layer.parameters()])
    ctx = ch.make_ctx(ei, et, n)
    assert len(ctx.chunks) > 3 and ctx.chunks[0][2] == 0 and ctx.chunks[-1][3] == ctx.n_seg
    assert torch.allclose(outs[0], outs[1], atol=1e-5)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(*grads))


def test_rgcn_is_per_relation_mean():
    """The RGCN branch uses the R-GCN definition (mean per relation, then W_r; PyG <= 2.0 behaviour).
    PyG 2.8.1's segment_matmul fast path (picked by a size heuristic, e.g. on TwiBot-22 shapes) takes
    the mean over ALL incoming edges instead; the branch forces the per-relation path."""
    from bb.methods.botmoe.model import GraphBranch
    torch.manual_seed(0)
    b = GraphBranch("rgcn", 4, 4, 3).eval()                       # E = 16
    n, e = 30, 200
    x = torch.randn(n, 16)
    ei = torch.randint(0, n, (2, e))
    et = torch.randint(0, 2, (e,))
    conv = b.conv

    def ref(h):
        out = h @ conv.root + conv.bias
        for r in range(2):
            m = et == r
            agg = torch.zeros(n, 16).index_add_(0, ei[1, m], h[ei[0, m]])
            cnt = torch.zeros(n).index_add_(0, ei[1, m], torch.ones(int(m.sum()))).clamp(min=1)
            out = out + (agg / cnt[:, None]) @ conv.weight[r]
        return out
    with torch.no_grad():
        assert torch.allclose(b.convs(x, ei, et), ref(ref(x)), atol=1e-5)
        b.rgcn_segmm = True                                       # PyG's edge-level path: different function
        assert not torch.allclose(b.convs(x, ei, et), ref(ref(x)), atol=1e-3)


# ------------------------------------------------------------------------------ host-built plan (review 2)
@pytest.mark.parametrize("max_edges", [7, 32768])
def test_plan_matches_device_path(max_edges):
    """plan.SubgraphPlan (host-built edge structures) == what the layers derive on the device: the
    ChunkedCtx fields, PyG gcn_norm, the per-relation edge split; and the whole model gives bit-equal
    logits, balance loss and gradients (train mode with dropout, same seed; one CPU thread, since the
    multithreaded CPU index_add / index_put accumulate is itself not bit-reproducible)."""
    from torch_geometric.nn.conv.gcn_conv import gcn_norm

    from bb.methods.botmoe.plan import SubgraphPlan
    from bb.methods.botmoe.rgt_chunked import ChunkedCtx
    g = _toy_inputs(n=40, e=300)
    ei = g.edge_index.clone()
    ei[1, :25] = 3                                          # a big segment spanning chunks
    ei[0, 30:36] = ei[1, 30:36]                             # self-loops (GCN drops and re-adds them)
    g = g.with_edges(ei, g.edge_type)
    n, R = 40, 2
    P = SubgraphPlan.build(ei.numpy(), g.edge_type.numpy(), n, R, max_edges, extra={"n_id": np.arange(n)})
    ref = ChunkedCtx(ei, g.edge_type, n, R, max_edges)
    for k in ("S", "T", "q_sorted", "k_sorted"):
        assert torch.equal(getattr(P, k), getattr(ref, k)), k
    assert P.chunks == ref.chunks and P.n_seg == ref.n_seg and (max_edges > 300 or len(P.chunks) > 3)
    gei, gw = gcn_norm(ei, None, n, False, True, "source_to_target", torch.float32)
    assert torch.equal(P.gcn_ei, gei) and torch.equal(P.gcn_weight(), gw)
    for r in range(R):
        assert torch.equal(P.rgcn_ei[r], ei[:, g.edge_type == r])
    from bb.methods.botmoe.edge_triton import csr_parts
    for a, b in zip((P.ptr_q, P.perm_k, P.ptr_k), csr_parts(ref, len(ref.S) * R)):
        assert torch.equal(a, b)
    assert torch.equal(P.n_id, torch.arange(n)) and torch.equal(P.edge_index, ei)
    meta = torch.randn(n, 8)
    res = []
    nt = torch.get_num_threads()
    torch.set_num_threads(1)
    for use in (None, P):
        torch.manual_seed(0)
        m = _model().train()
        m.rgt.conv1.max_edges = m.rgt.conv2.max_edges = max_edges
        m.set_checkpoint(True)
        torch.manual_seed(1)
        out, aux = m(g, meta, plan=use)
        (torch.nn.functional.cross_entropy(out[:10], g.y[:10]) + aux["balance"]).backward()
        res.append((out.detach(), aux["balance"].detach(), [p.grad for p in m.parameters() if p.grad is not None]))
    torch.set_num_threads(nt)
    assert torch.equal(res[0][0], res[1][0]) and torch.equal(res[0][1], res[1][1])
    assert len(res[0][2]) == len(res[1][2]) and all(torch.equal(a, b) for a, b in zip(res[0][2], res[1][2]))


@pytest.mark.parametrize("train,max_edges", [(False, 7), (True, 7), (True, 32768)])
def test_edge_attention_fn_equals_autograd(train, max_edges):
    """ChunkedRGTLayer edge_impl="fn" (hand-written backward, no recompute) == the autograd path
    (checkpointed and not): bit-equal output (same dropout draws), gradients to float rounding."""
    from bb.methods.botmoe.rgt_chunked import ChunkedRGTLayer
    torch.manual_seed(0)
    L = ChunkedRGTLayer(2, 16, 2, 2, 0.5, max_edges=max_edges).train(train)
    g = torch.Generator().manual_seed(1)
    n, e = 30, 120
    ei = torch.randint(0, n, (2, e), generator=g)
    ei[1, :12] = 3
    et = torch.randint(0, 2, (e,), generator=g)
    x = torch.randn(n, 16, generator=g)
    ctx = L.make_ctx(ei, et, n)
    w = torch.randn(n, 16, generator=g)
    res = []
    for impl, ck in (("autograd", False), ("autograd", True), ("fn", False)):
        L.edge_impl, L.ckpt = impl, ck
        L.zero_grad()
        xx = x.clone().requires_grad_(True)
        torch.manual_seed(5)
        y = L(xx, ei, et, ctx)
        (y * w).sum().backward()
        res.append((y.detach(), [xx.grad] + [p.grad for p in L.parameters()]))
    for y, gr in res[1:]:
        assert torch.equal(res[0][0], y)
        assert all(torch.allclose(a, b, atol=1e-6, rtol=1e-5) for a, b in zip(res[0][1], gr))
    if train:                                   # dropout active: masks really differ from eval
        L.eval()
        assert not torch.equal(L(x, ei, et, ctx), res[0][0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused RGT kernels need CUDA + Triton")
@pytest.mark.parametrize("train", [False, True])
def test_edge_attention_triton_equals_fn(train):
    """edge_impl="triton" (fused sddmm / CSR spmm kernels) == "fn": the same dropout draws, outputs and
    gradients to float rounding, on a graph with hub segments, empty segments and several chunks."""
    from bb.methods.botmoe.plan import SubgraphPlan
    from bb.methods.botmoe.rgt_chunked import ChunkedRGTLayer
    dev = "cuda"
    torch.manual_seed(0)
    H = 96                                                  # not a power of two: masked tail
    L = ChunkedRGTLayer(2, H, 2, 2, 0.5, max_edges=64).to(dev).train(train)
    g = torch.Generator().manual_seed(1)
    n, e = 300, 2000
    ei = torch.randint(0, n, (2, e), generator=g)
    ei[1, :300] = 5                                         # hub segment spanning chunks
    ei[0, 300:400] = 9                                      # hub source
    et = torch.randint(0, 2, (e,), generator=g)
    x = torch.randn(n, H, generator=g).to(dev)
    w = torch.randn(n, H, generator=g).to(dev)
    P = SubgraphPlan.build(ei.numpy(), et.numpy(), n, 2, 64, device=dev)
    ctx_dev = L.make_ctx(ei.to(dev), et.to(dev), n)          # device-built ctx: csr_parts computed lazily
    res = []
    for impl, ctx in (("fn", P), ("triton", P), ("triton", ctx_dev)):
        L.edge_impl = impl
        L.zero_grad()
        xx = x.clone().requires_grad_(True)
        torch.manual_seed(5)
        y = L(xx, P.edge_index, P.edge_type, ctx)
        (y * w).sum().backward()
        res.append((y.detach(), [xx.grad] + [p.grad for p in L.parameters()]))
    for y, gr in res[1:]:
        assert torch.allclose(res[0][0], y, atol=1e-5, rtol=1e-5)
        for a, b in zip(res[0][1], gr):
            assert torch.allclose(a, b, atol=1e-5, rtol=1e-4), float((a - b).abs().max())
