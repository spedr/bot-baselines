"""GCN and GAT baselines of the TwiBot-22 benchmark (Feng et al., NeurIPS 2022 D&B, App. B.1).

Spec: docs/methods/gcn_gat.md. Inputs: the shared ``botrgcn`` features (bb/features/botrgcn.py);
both follow relations merged into one directed edge set (edge types ignored). Model: the shared
``bb.gnn`` encoder/head (H = 128, input dropout 0.3) around two GCNConv / GATConv layers with no
activation between them. Training: neighbour-sampled mini-batches (128 seeds, fan-out 256) with
Adam(1e-4, wd 1e-5), lexicographic early stopping on val (patience 50, at most 1000 epochs).

Default = the fixed protocol: test the deep-copied best-validation state. ``--emulate-state-bug``
reports the final weights instead (the official ``best_state_dict = model.state_dict()`` reference
bug); both are computed from the same run and stored in ``result.json["metrics_by_state"]``.

    tools/memrun --light -- uv run python -m bb.methods.gcn_gat --conv gcn --dataset cresci15
    tools/memrun --light -- uv run python -m bb.methods.gcn_gat --conv gat --dataset twibot20
    tools/memrun -m 7G -- uv run python -m bb.methods.gcn_gat --conv gat --dataset twibot22 --seeds 0
    ... --with-extscore synthetic           # ExtScore [z, m] appended to num_prop (5 -> 7)
    ... --bench-hops                   # literal [256] * 4 sampling (same outputs in distribution, slower)
    ... --eval full                    # val/test with full neighbourhoods (fan-out -1) instead of sampled
    ... --measure-batches 300          # time one epoch slice + VRAM and exit (no Run written)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from ...harness import Run, aggregate, compute_metrics, set_seed
from ...extscore import add_extscore_args, prepare_extscore

SRC = [Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("train.py")]


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="GCN / GAT (TwiBot-22 benchmark versions)")
    ap.add_argument("--conv", choices=["gcn", "gat"], required=True)
    ap.add_argument("--dataset", choices=["twibot20", "twibot22", "cresci15"], default="cresci15")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--features", default="botrgcn", help="feature variant (bb.features.VARIANTS)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--fanout", type=int, default=256)
    ap.add_argument("--bench-hops", action="store_true",
                    help="sample 4 hops as the official loader (default: 2 for GAT, 3 for GCN, equal in distribution)")
    ap.add_argument("--fanouts", type=int, nargs="+", default=None, help="explicit per-hop fan-outs (overrides)")
    ap.add_argument("--eval", dest="eval_mode", choices=["sampled", "full"], default="sampled",
                    help="sampled = benchmark (val/test sampled with the training fan-outs); full = fan-out -1")
    ap.add_argument("--eval-batch-size", type=int, default=4096,
                    help="seeds per eval batch (per-seed sampling distribution does not depend on it)")
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=50, help="epochs without val improvement before stopping")
    ap.add_argument("--eval-every", type=int, default=1,
                    help="validate every k epochs; patience then counts ceil(patience / k) evaluations (deviation)")
    ap.add_argument("--emulate-state-bug", action="store_true",
                    help="report the FINAL weights (official state_dict-by-reference bug) as the primary metrics")
    ap.add_argument("--extscore-inject", choices=["num", "branch", "late"], default="num",
                    help="num: [z, m] appended to num_prop (default); branch: fifth input branch; late: before the classifier")
    ap.add_argument("--text-dtype", choices=["auto", "float16", "float32"], default="auto")
    ap.add_argument("--fused-adam", action="store_true",
                    help="fused CUDA Adam also on the eager path (default: fused on the CUDA-graph path only)")
    ap.add_argument("--no-fused-adam", action="store_true",
                    help="foreach (multi-tensor) Adam on the CUDA-graph path too (the pre-review-2 default; "
                         "same algorithm, float-rounding differences only)")
    ap.add_argument("--no-analytic-degree", action="store_true",
                    help="GCN: sample the 3rd hop and count degrees instead of min(indeg, k) (identical plans)")
    ap.add_argument("--no-cuda-graph", action="store_true",
                    help="eager training steps (default on CUDA: the step is replayed as a captured CUDA graph)")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--measure-batches", type=int, default=0,
                    help="time K training batches + one val pass for one seed and exit (no Run written; "
                         "zero placeholders for text tables that are not built yet)")
    add_extscore_args(ap)
    a = ap.parse_args(argv)
    if a.fanouts is None:
        from .train import default_fanouts
        a.fanouts = default_fanouts(a.conv, a.fanout, a.bench_hops)
    if len(a.fanouts) < (2 if a.conv == "gat" else 3):
        raise SystemExit(f"{a.conv}: fewer hops than {2 if a.conv == 'gat' else 3} change the model's outputs (spec 5)")
    return a


def variant_name(a) -> str:
    v = [a.conv, a.features]
    if a.fanouts != [256] * len(a.fanouts) or a.bench_hops:
        v.append("f" + "x".join(map(str, a.fanouts)))
    if a.eval_mode != "sampled":
        v.append("eval" + a.eval_mode)
    if a.eval_every != 1:
        v.append(f"ev{a.eval_every}")
    if a.emulate_state_bug:
        v.append("finalw")
    if (a.with_extscore or a.extscore == "p") and a.extscore_inject != "num":
        v.append(f"extscore{a.extscore_inject}")
    return ".".join(v)


def _undirected(ds: str, rels: list[str]):
    from ...data import load_undirected
    from ...paths import dataset_paths
    g = dataset_paths(ds).graph
    for r in (rels, rels[::-1]):
        if (g / ("undirected__" + "+".join(r) + ".csr")).exists():
            return load_undirected(ds, r)
    return load_undirected(ds, rels)


def _inputs(a, F, J, dev):
    """Device inputs. Without built text tables (``--measure-batches`` only) the text tensors are
    zero rows broadcast to [N, 768] (no memory): timing is unaffected, the VRAM of the tables is not
    counted (add 2 x N x 768 x 2 B)."""
    import torch

    from ...gnn import GraphInputs, to_device
    tdt = None if a.text_dtype == "auto" else getattr(torch, a.text_dtype)
    if F.des is not None and F.tweets is not None:
        return to_device(F, dev, J=J, extscore_inject=a.extscore_inject, text_dtype=tdt, edges=False), False
    from dataclasses import replace
    one = np.zeros((1, 768), np.float16)
    g = to_device(replace(F, des=one, tweets=one), dev, J=J, extscore_inject=a.extscore_inject, edges=False, text_dtype=tdt)
    z = g.des.expand(F.n, 768)
    return GraphInputs(z, z, g.num, g.cat, g.edge_index, g.edge_type, g.y, g.extra), True


def _setup(a):
    import torch

    from ...data import load_in_merged
    from ...features import load_features
    t0 = time.time()
    F = load_features(a.dataset, a.features, require_text=not a.measure_batches)
    splits = {s: F.splits[s] for s in ("train", "val", "test")}
    extscore_on = bool(a.with_extscore or a.extscore == "p")
    J = prepare_extscore(a, a.dataset, splits, labels=F.label,
                    neighbours=_undirected(a.dataset, F.relations_by_type()) if extscore_on else None)
    dev = "cuda" if (a.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    g, placeholder = _inputs(a, F, J, dev)
    csc = load_in_merged(a.dataset, F.relations_by_type())
    load_s = round(time.time() - t0, 2)
    num_dim = F.num_prop.shape[1] + (J.width if a.extscore_inject == "num" else 0)
    print(f"{a.conv} {a.dataset}/{F.variant}: N={F.n} E={csc.indices.shape[0]} num={tuple(g.num.shape)} "
          f"cat={tuple(g.cat.shape)} text={g.des.dtype}{' (placeholder)' if placeholder else ''} on {dev} "
          f"fanouts={a.fanouts} eval={a.eval_mode} (load {load_s} s) extscore={'on:' + a.extscore_inject if J.enabled else 'off'}",
          flush=True)
    return F, splits, J, dev, g, csc, load_s, num_dim, placeholder


def _cfg(a, seed):
    from .train import TrainConfig
    k = max(1, a.eval_every)
    return TrainConfig(conv=a.conv, lr=a.lr, weight_decay=a.wd, batch_size=a.batch_size, fanouts=tuple(a.fanouts),
                       eval_mode=a.eval_mode, eval_batch_size=a.eval_batch_size, max_epochs=a.max_epochs,
                       patience=math.ceil(a.patience / k), eval_every=k, seed=seed,
                       fused_adam=True if a.fused_adam else (False if a.no_fused_adam else None),
                       cuda_graph=not a.no_cuda_graph, analytic_degree=not a.no_analytic_degree)


def _build(a, F, J, num_dim, dev):
    from .model import build_model
    return build_model(a.conv, hidden=a.hidden, num_dim=num_dim, cat_dim=F.cat_prop.shape[1], dropout=a.dropout,
                       extscore_width=J.width, extscore_inject=a.extscore_inject).to(dev)


def _metrics(prob, y, splits) -> dict:
    from .train import pr_auc
    out = {}
    for s in ("val", "test"):
        ix = np.asarray(splits[s])
        m = compute_metrics(np.asarray(y)[ix], prob[ix])
        m["pr_auc"] = pr_auc(np.asarray(y)[ix], prob[ix])
        out[s] = m
    return out


def main(argv=None) -> list[dict]:
    import torch

    from ...gnn import n_params
    from .train import fit

    a = parse_args(argv)
    F, splits, J, dev, g, csc, load_s, num_dim, _ = _setup(a)
    if a.measure_batches:
        return [measure(a, F, splits, J, dev, g, csc, load_s, num_dim)]
    y = F.label
    method = a.conv
    cfg_base = {**vars(a), "variant": variant_name(a), "features_variant": F.variant,
                "sampling_note": ("literal benchmark depth" if len(a.fanouts) == 4 else
                                  f"{len(a.fanouts)} hops: outputs equal in distribution to [256]*4 (spec 5)")}
    results = []
    for seed in a.seeds:
        set_seed(seed, deterministic=a.deterministic)
        model = _build(a, F, J, num_dim, dev)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with Run(method, a.dataset, config=cfg_base, seed=seed, data_files=F.files, source_files=SRC) as run:
            res = fit(model, g, splits, _cfg(a, seed), csc=csc, run=run, log_every=a.log_every)
            by_state = {"best_val": _metrics(res.prob_best, y, splits), "final_weights": _metrics(res.prob_final, y, splits)}
            primary = "final_weights" if a.emulate_state_bug else "best_val"
            prob = res.prob_final if a.emulate_state_bug else res.prob_best
            for s in ("val", "test"):
                run.add_predictions(s, splits[s], prob[splits[s]], y[splits[s]])
            out = run.finish(
                by_state[primary], metrics_by_state=by_state,
                selection={"select": primary, "rule": "lexicographic acc,f1,mcc,pr_auc,roc_auc,precision,recall",
                           "best_epoch": res.best_epoch, "best_val_during_training": res.best_val,
                           "patience_epochs": a.patience, "eval_every": a.eval_every},
                epochs_run=res.epochs_run, extscore=J.report(), n_params=n_params(model), with_extscore=a.with_extscore,
                timing={"load_seconds": load_s, "train_seconds": res.train_seconds,
                        "seconds_per_epoch": res.seconds_per_epoch, "eval_seconds": res.eval_seconds},
                train_step=res.step_info,
                gpu={"inputs_bytes": int(sum(t.numel() * t.element_size() for t in (g.des, g.tweet, g.num, g.cat, g.y))),
                     "run_max_allocated_bytes": int(torch.cuda.max_memory_allocated()) if dev == "cuda" else 0},
                features={"dir": str(F.dir), "input_key": F.meta["input_key"], "variant": F.variant})
        t, tf = by_state[primary]["test"], by_state["final_weights"]["test"]
        print(f"[{method} {a.dataset} seed {seed}] test acc {t['accuracy']:.4f} f1 {t['f1']:.4f} "
              f"prec {t['precision']:.4f} rec {t['recall']:.4f} auc {t['roc_auc']:.4f} ({primary}, epoch {res.best_epoch}) | "
              f"final weights acc {tf['accuracy']:.4f} f1 {tf['f1']:.4f} | {res.epochs_run} epochs, "
              f"{res.seconds_per_epoch:.2f} s/epoch train", flush=True)
        results.append(out)
        del model
    agg = aggregate(results)
    print({k: (round(v["mean"], 4), round(v["std"], 4)) for k, v in agg.items()
           if k in ("accuracy", "f1", "precision", "recall", "macro_f1", "mcc", "roc_auc")})
    if J.enabled and J.baseline:
        b = J.baseline["test"]
        print(f"column-only baseline (val-tuned threshold {J.baseline['threshold']:.3f}): "
              f"test acc {b['accuracy']:.4f} f1 {b['f1']:.4f}")
    return results


def measure(a, F, splits, J, dev, g, csc, load_s, num_dim) -> dict:
    """``--measure-batches K``: time K training batches and one val pass (seed ``a.seeds[0]``)."""
    import torch

    from ...harness.sampler import NeighborSampler
    from ...mem import memory_report
    from .train import _Batcher, fit, predict

    set_seed(a.seeds[0])
    model = _build(a, F, J, num_dim, dev)
    cfg = _cfg(a, a.seeds[0])
    cfg.max_epochs = 1
    res = fit(model, g, splits, cfg, csc=csc, max_train_batches=a.measure_batches)
    n_tr = int((F.label[splits["train"]] >= 0).sum())
    per_batch = res.train_seconds / a.measure_batches
    t = time.perf_counter()
    ev = [-1] * len(a.fanouts) if a.eval_mode == "full" else list(a.fanouts)
    predict(model, _Batcher(g, csc, ev, a.conv == "gcn"), splits["val"], a.eval_batch_size, 0)
    val_s = time.perf_counter() - t
    batches = math.ceil(n_tr / a.batch_size)
    sz = NeighborSampler(csc, list(a.fanouts)).dry_run(splits["train"], a.batch_size, n_batches=100)
    rep = {"dataset": a.dataset, "conv": a.conv, "fanouts": a.fanouts, "batch_size": a.batch_size,
           "placeholder_text": F.des is None or F.tweets is None, "load_seconds": load_s,
           "ms_per_train_batch": round(per_batch * 1e3, 3), "train_batches_per_epoch": batches,
           "projected_train_seconds_per_epoch": round(per_batch * batches, 1),
           "val_pass_seconds": round(val_s, 2), "subgraph": sz, "train_step": res.step_info,
           "cuda_max_allocated_GB": round(torch.cuda.max_memory_allocated() / 1e9, 3) if dev == "cuda" else 0,
           "memory": memory_report()}
    print("MEASURE " + json.dumps(rep, default=str), flush=True)
    return rep
