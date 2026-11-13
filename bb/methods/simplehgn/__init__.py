"""Simple-HGN (Lv et al., KDD 2021) and HGT (Hu et al., WWW 2020) as run in the TwiBot-22 benchmark.

Spec: docs/methods/simplehgn.md. Features: the shared ``botrgcn`` scheme. Model: RGT-style input
encoder (dropout 0.5) -> 2 Simple-HGN or HGT layers (one head) -> Linear(128,64)+LeakyReLU[+Drop]
-> Linear(64,2). AdamW lr 1e-3, CosineAnnealingLR(T_max=16), fp16 autocast, best val accuracy.

    tools/memrun --light -- uv run python -m bb.methods.simplehgn --dataset cresci15 --model shgn
    tools/memrun --light -- uv run python -m bb.methods.simplehgn --dataset twibot20 --model hgt
    tools/memrun -m 7G -- uv run python -m bb.methods.simplehgn --dataset twibot22 --model shgn --seeds 0
    ... --with-extscore synthetic                   # ExtScore [z, m] appended to num_prop (5 -> 7)
    ... --edge-residual bench                  # benchmark layer.py: beta has no effect (= beta 0)
    ... --model hgt --hgt-impl bench_bug       # PyG 2.0.2 HGTConv (graph-free 'alpha * alpha' bug)
    ... --dataset twibot22 --bench-compat      # benchmark leaky all-node loss + mixed val + its fan-outs

Defaults are the fixed behaviour: seed-only loss and seed-only validation, the paper's working
edge residual, and the correct (modern PyG) HGT. TwiBot-20 / Cresci-2015 train on exact
in-neighbourhoods of each batch (fan-out -1), which equals the benchmark's full-graph forward.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ...harness import Run, aggregate, set_seed
from ...extscore import add_extscore_args, prepare_extscore

# spec 4.4 (Cresci-2015 has no dedicated script: the TwiBot-20 full-batch settings are used)
PRESETS = {
    ("twibot20", "shgn"): dict(rel_dim=100, wd=3e-5, batch_size=128, epochs=200, fanouts=[-1, -1], patience=-1, head_dropout=True),
    ("cresci15", "shgn"): dict(rel_dim=100, wd=3e-5, batch_size=128, epochs=200, fanouts=[-1, -1], patience=-1, head_dropout=True),
    ("twibot22", "shgn"): dict(rel_dim=200, wd=0.0, batch_size=512, epochs=50, fanouts=[256, 256], patience=10, head_dropout=True),
    ("twibot20", "hgt"): dict(rel_dim=100, wd=3e-5, batch_size=128, epochs=200, fanouts=[-1, -1], patience=-1, head_dropout=True),
    ("cresci15", "hgt"): dict(rel_dim=100, wd=3e-5, batch_size=128, epochs=200, fanouts=[-1, -1], patience=-1, head_dropout=True),
    ("twibot22", "hgt"): dict(rel_dim=200, wd=3e-5, batch_size=256, epochs=50, fanouts=[256, 256], patience=10, head_dropout=False),
}
BENCH_FANOUTS = {"shgn": [256, 256, 256, 256], "hgt": [256, 256]}   # TwiBot-22 *_sample.py
SRC = [Path(__file__).with_name(f) for f in ("__init__.py", "fused.py", "layers.py", "model.py", "sampling.py", "train.py")]


def _bool(s: str) -> bool:
    if s.lower() in ("1", "true", "yes", "on"):
        return True
    if s.lower() in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(s)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Simple-HGN / HGT (TwiBot-22 benchmark versions)")
    ap.add_argument("--dataset", choices=["twibot20", "twibot22", "cresci15"], default="cresci15")
    ap.add_argument("--model", choices=["shgn", "hgt"], default="shgn")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--features", default="botrgcn", help="feature variant (bb.features.VARIANTS)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None, help="early stopping on val acc; -1 off")
    ap.add_argument("--fanouts", type=int, nargs="+", default=None,
                    help="training fan-out per hop (-1 = all); HGT: per edge type")
    ap.add_argument("--t-max", type=int, default=16, help="CosineAnnealingLR T_max (epochs)")
    ap.add_argument("--amp", type=_bool, default=True, help="fp16 autocast + GradScaler on CUDA")
    ap.add_argument("--select", choices=["best_val", "last_epoch"], default="best_val")
    ap.add_argument("--select-metric", default="accuracy")
    ap.add_argument("--eval-batch-size", type=int, default=4096)
    # Simple-HGN
    ap.add_argument("--rel-dim", type=int, default=None)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--edge-residual", choices=["paper", "bench"], default="paper",
                    help="paper: mixed attention used in the message (Eq. 10, default); bench: benchmark "
                         "layer.py, where beta has no effect (= beta 0)")
    # HGT
    ap.add_argument("--hgt-impl", choices=["fixed", "per_relation", "bench_bug"], default="fixed",
                    help="fixed: modern PyG HGTConv (default); per_relation: per-relation softmax then sum; "
                         "bench_bug: PyG 2.0.2 'alpha * alpha' bug (graph-free)")
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--head-dropout", type=_bool, default=None, help="dropout after out1 (HGT TwiBot-22: off)")
    # TwiBot-22 benchmark protocol emulation
    ap.add_argument("--loss-nodes", choices=["seeds", "all"], default="seeds",
                    help="all = benchmark TwiBot-22 loss over every sampled node (LEAKS val/test labels)")
    ap.add_argument("--val-nodes", choices=["seeds", "subgraph"], default="seeds",
                    help="subgraph = benchmark TwiBot-22 val acc over all nodes of val subgraphs (selection)")
    ap.add_argument("--bench-compat", action="store_true",
                    help="TwiBot-22: --loss-nodes all --val-nodes subgraph and the benchmark fan-outs (LEAKY)")
    ap.add_argument("--extscore-inject", choices=["num", "branch", "late"], default="num",
                    help="num: [z, m] appended to num_prop (default); branch: fifth input branch; late: before out1")
    ap.add_argument("--fused", type=_bool, default=True,
                    help="CUDA: Triton edge kernels + fused AdamW (same function; off = round-1 eager path)")
    ap.add_argument("--prefetch", type=int, default=3, help="batches sampled ahead in a thread (0: inline)")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="sampler dry run (subgraph sizes) and exit")
    ap.add_argument("--measure-batches", type=int, default=0,
                    help="measure: one epoch of K training steps + one validation pass, then exit (no Run)")
    ap.add_argument("--measure-full-step", action="store_true",
                    help="measure one true full-graph training step (benchmark full batch) and exit")
    add_extscore_args(ap)
    a = ap.parse_args(argv)
    for k, v in PRESETS[(a.dataset, a.model)].items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    if a.bench_compat:
        if a.dataset != "twibot22":
            raise SystemExit("--bench-compat is the TwiBot-22 sampled protocol")
        a.loss_nodes, a.val_nodes = "all", "subgraph"
        a.fanouts = BENCH_FANOUTS[a.model]
    if len(a.fanouts) < 2:
        raise SystemExit("two message-passing layers: give at least two fan-outs")
    return a


def variant_name(a) -> str:
    v = [a.model, a.features]
    if a.model == "shgn" and a.edge_residual != "paper":
        v.append(f"eres-{a.edge_residual}")
    if a.model == "hgt" and a.hgt_impl != "fixed":
        v.append(a.hgt_impl)
    if a.bench_compat:
        v.append("benchcompat")
    else:
        if a.loss_nodes != "seeds":
            v.append(f"loss-{a.loss_nodes}")
        if a.val_nodes != "seeds":
            v.append(f"val-{a.val_nodes}")
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


def _model(a, F, J, num_dim):
    from .model import build_model
    return build_model(a.model, num_dim=num_dim, cat_dim=F.cat_prop.shape[1], hidden=a.hidden,
                       num_edge_types=F.num_relations, dropout=a.dropout, extscore_width=J.width,
                       extscore_inject=a.extscore_inject, rel_dim=a.rel_dim, beta=a.beta, edge_residual=a.edge_residual,
                       hgt_impl=a.hgt_impl, heads=a.heads, head_dropout=a.head_dropout)


def _train_cfg(a, seed):
    from .train import TrainConfig
    return TrainConfig(epochs=a.epochs, lr=a.lr, weight_decay=a.wd, batch_size=a.batch_size, t_max=a.t_max,
                       amp=a.amp, select=a.select, select_metric=a.select_metric, patience=a.patience,
                       seed=seed, eval_batch_size=a.eval_batch_size, loss_nodes=a.loss_nodes,
                       val_nodes=a.val_nodes, max_train_batches=a.measure_batches, fused_optimizer=a.fused,
                       prefetch=a.prefetch)


def main(argv=None) -> list[dict]:
    import torch

    from ...data import load_in_merged
    from ...features import load_features
    from ...gnn import n_params, split_metrics, to_device
    from .sampling import make_sampler
    from .train import fit

    a = parse_args(argv)
    from . import fused
    fused.ENABLED = a.fused
    t0 = time.time()
    measuring = bool(a.measure_batches or a.measure_full_step or a.dry_run)
    F = load_features(a.dataset, a.features, require_text=not measuring)
    splits = {s: np.asarray(F.splits[s]) for s in ("train", "val", "test")}
    y = np.asarray(F.label)
    rels = F.relations_by_type()
    csc = load_in_merged(a.dataset, rels)
    per_rel = a.model == "hgt"
    train_sampler = make_sampler(csc, a.fanouts, per_rel)
    eval_sampler = make_sampler(csc, [-1, -1], per_rel)
    if a.dry_run:
        return [dry_run(a, train_sampler, eval_sampler, splits)]
    extscore_on = bool(a.with_extscore or a.extscore == "p")
    J = prepare_extscore(a, a.dataset, splits, labels=y, neighbours=_undirected(a.dataset, rels) if extscore_on else None)
    dev = "cuda" if (a.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    g = to_device(F, dev, J=J, extscore_inject=a.extscore_inject, edges=a.measure_full_step,
                  placeholder_text=measuring)
    etype = torch.from_numpy(np.asarray(csc.etype, dtype=np.int64)).to(dev)
    load_s = round(time.time() - t0, 2)
    gpu_inputs = g.nbytes()
    num_dim = F.num_prop.shape[1] + (J.width if a.extscore_inject == "num" else 0)
    print(f"{a.dataset}/{F.variant} {a.model}: N={F.n} E={csc.nnz} num={tuple(g.num.shape)} cat={tuple(g.cat.shape)} "
          f"text={g.des.dtype} inputs on {dev}: {gpu_inputs / 1e9:.2f} GB (load {load_s} s) "
          f"fanouts={a.fanouts} batch={a.batch_size} loss={a.loss_nodes} val={a.val_nodes} "
          f"extscore={'on:' + a.extscore_inject if J.enabled else 'off'}", flush=True)
    if a.measure_full_step:
        return [measure_full_step(a, F, g, J, num_dim, dev, splits)]
    if a.measure_batches:
        return [measure(a, F, g, J, num_dim, dev, load_s, splits, train_sampler, eval_sampler, etype)]
    method = "simplehgn" if a.model == "shgn" else "hgt"
    cfg_base = {**vars(a), "variant": variant_name(a), "features_variant": F.variant}
    results = []
    for seed in a.seeds:
        set_seed(seed, deterministic=a.deterministic)
        model = _model(a, F, J, num_dim).to(dev)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with Run(method, a.dataset, config=cfg_base, seed=seed, data_files=F.files, source_files=SRC) as run:
            res = fit(model, g, splits, _train_cfg(a, seed), train_sampler=train_sampler,
                      eval_sampler=eval_sampler, etype=etype, run=run, log_every=a.log_every)
            metrics = split_metrics(res.prob, y, splits)
            for s in ("val", "test"):
                run.add_predictions(s, splits[s], res.prob[splits[s]], y[splits[s]])
            out = run.finish(
                metrics, selection=res.selection, epochs_run=res.epochs_run, extscore=J.report(),
                n_params=n_params(model), with_extscore=a.with_extscore,
                timing={"load_seconds": load_s, "train_seconds": res.train_seconds,
                        "seconds_per_epoch": res.seconds_per_epoch, "train_steps": res.steps},
                gpu={"inputs_bytes": gpu_inputs,
                     "run_max_allocated_bytes": int(torch.cuda.max_memory_allocated()) if dev == "cuda" else 0},
                features={"dir": str(F.dir), "input_key": F.meta["input_key"], "variant": F.variant})
        t = metrics["test"]
        print(f"[{method} {a.dataset} seed {seed}] test acc {t['accuracy']:.4f} f1 {t['f1']:.4f} "
              f"prec {t['precision']:.4f} rec {t['recall']:.4f} auc {t['roc_auc']:.4f} | "
              f"{res.seconds_per_epoch:.2f} s/epoch, {res.epochs_run} epochs, best ep "
              f"{res.selection['best_epoch']}", flush=True)
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


def dry_run(a, train_sampler, eval_sampler, splits) -> dict:
    import json
    rep = {"dataset": a.dataset, "model": a.model,
           "train": train_sampler.dry_run(splits["train"], a.batch_size, n_batches=200,
                                          feature_bytes_per_node=1546 * 2),
           "eval": eval_sampler.dry_run(splits["val"], a.eval_batch_size, n_batches=50,
                                        feature_bytes_per_node=1546 * 2)}
    print("DRYRUN " + json.dumps(rep, default=str), flush=True)
    return rep


def measure(a, F, g, J, num_dim, dev, load_s, splits, train_sampler, eval_sampler, etype) -> dict:
    """K training steps + one full validation pass: ms/step, projected epoch time, VRAM, RSS."""
    import json

    import torch

    from ...mem import memory_report
    from .train import fit

    set_seed(a.seeds[0])
    model = _model(a, F, J, num_dim).to(dev)
    cfg = _train_cfg(a, a.seeds[0])
    cfg.epochs = 1
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t = time.time()
    res = fit(model, g, splits, cfg, train_sampler=train_sampler, eval_sampler=eval_sampler, etype=etype,
              predict_splits=())
    wall = time.time() - t
    n_tr = len(splits["train"])
    steps_per_epoch = -(-n_tr // a.batch_size)
    ms_step = res.train_seconds / max(res.steps, 1) * 1e3
    val_s = wall - res.train_seconds
    epoch_s = ms_step * steps_per_epoch / 1e3 + val_s
    rep = {"dataset": a.dataset, "model": a.model, "fanouts": a.fanouts, "batch_size": a.batch_size,
           "steps": res.steps, "ms_per_step": round(ms_step, 2), "steps_per_epoch": steps_per_epoch,
           "val_pass_seconds": round(val_s, 2), "projected_epoch_seconds": round(epoch_s, 1),
           "projected_run_hours_at_max_epochs": round(epoch_s * a.epochs / 3600, 3),
           "placeholder_text": F.des is None or F.tweets is None, "load_seconds": load_s,
           "gpu_inputs_GB": round(g.nbytes() / 1e9, 3),
           "cuda_max_allocated_GB": round(torch.cuda.max_memory_allocated() / 1e9, 3) if dev == "cuda" else 0,
           "cuda_max_reserved_GB": round(torch.cuda.max_memory_reserved() / 1e9, 3) if dev == "cuda" else 0,
           "memory": memory_report()}
    print("MEASURE " + json.dumps(rep, default=str), flush=True)
    return rep


def measure_full_step(a, F, g, J, num_dim, dev, splits, n_steps: int = 5) -> dict:
    """True full-graph training steps (benchmark full batch: forward over all N nodes, loss on one
    train batch): seconds per step and CUDA peak, to decide full-batch vs sampled training."""
    import json

    import torch
    from torch import nn

    from ...mem import memory_report

    set_seed(a.seeds[0])
    model = _model(a, F, J, num_dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp and dev == "cuda")
    tr = torch.as_tensor(splits["train"][: a.batch_size], device=dev)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    ts = []
    for _ in range(n_steps):
        t = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp and dev == "cuda"):
            out = model(g)
        loss = nn.functional.cross_entropy(out.float()[tr], g.y[tr])
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if dev == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    steps_per_epoch = -(-len(splits["train"]) // a.batch_size)
    rep = {"dataset": a.dataset, "model": a.model, "N": F.n, "E": int(g.edge_index.shape[1]),
           "seconds_per_full_step": round(float(np.median(ts[1:] or ts)), 4), "steps_per_epoch": steps_per_epoch,
           "projected_epoch_seconds": round(float(np.median(ts[1:] or ts)) * steps_per_epoch, 1),
           "gpu_inputs_GB": round(g.nbytes() / 1e9, 3),
           "cuda_max_allocated_GB": round(torch.cuda.max_memory_allocated() / 1e9, 3) if dev == "cuda" else 0,
           "memory": memory_report()}
    print("MEASURE_FULL " + json.dumps(rep, default=str), flush=True)
    return rep
