"""RGT: Relational Graph Transformer (Feng, Tan, Li, Luo, AAAI 2022). Spec: docs/methods/rgt.md.

Features: the shared ``botrgcn`` scheme (``bb.features.load_features``), device inputs and ExtScore
injection through ``bb.gnn.to_device``. Model and loops are local (``model.py``, ``train.py``): the
RGT encoder/head (dropout after every projection, 128 -> 64 -> 2 head, zero-bias kaiming init) and
its protocol (loss on mini-batches of train rows, AdamW + periodic cosine LR, AMP, best val acc)
differ from the BotRGCN skeleton's.

    tools/memrun --light -- uv run python -m bb.methods.rgt --dataset cresci15
    tools/memrun -m 7G  -- uv run python -m bb.methods.rgt --dataset twibot20               # benchmark preset
    tools/memrun -m 7G  -- uv run python -m bb.methods.rgt --dataset twibot20 --preset paper
    tools/memrun -m 7G  -- uv run python -m bb.methods.rgt --dataset twibot22 --seeds 0
    ... --with-extscore synthetic        # ExtScore [z, m] appended to num_prop (5 -> 7); --extscore-inject branch|late
    ... --bench-compat              # TwiBot-22 benchmark emulation: [256]*4 fan-out, LEAKY all-node loss,
                                    # last epoch (never a clean baseline)
    ... --measure-steps 50          # time / VRAM of 50 optimiser steps + one val pass, then exit
    ... --edge-impl torch --tail-impl eager   # reference ops (full-graph default: triton + compile)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ...harness import Run, aggregate, set_seed
from ...extscore import add_extscore_args, prepare_extscore

# spec sections 4-5. Cresci-2015 (not in the spec's table): benchmark `src/RGT/cresci-15/Heterobot.py`
# is the TwiBot-20 script with cat_num 1 and trans/semantic heads 8 by default; the readme command
# for "Twibot-20 & cresci-2015" is `--batch_size 128 --epochs 200`.
_FULL = {"training": "full", "trans_heads": 2, "sem_heads": 2, "dropout": 0.5, "lr": 1e-3, "wd": 3e-5,
         "batch_size": 128, "epochs": 200, "patience": -1}
PRESETS = {
    ("twibot20", "benchmark"): _FULL,
    ("cresci15", "benchmark"): {**_FULL, "trans_heads": 8, "sem_heads": 8},
    ("twibot20", "paper"): {**_FULL, "trans_heads": 8, "sem_heads": 8, "batch_size": 256, "epochs": 40},
    ("twibot22", "benchmark"): {"training": "sampled", "trans_heads": 2, "sem_heads": 2, "dropout": 0.3,
                                "lr": 5e-4, "wd": 1e-5, "batch_size": 256, "epochs": 50, "patience": 10,
                                "fanouts": [256, 256]},
}
SRC = [Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("train.py")]


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RGT (Relational Graph Transformer)")
    ap.add_argument("--dataset", choices=["twibot20", "twibot22", "cresci15"], default="cresci15")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--features", default="botrgcn", help="feature variant (bb.features.VARIANTS)")
    ap.add_argument("--preset", choices=["benchmark", "paper"], default="benchmark",
                    help="benchmark: TwiBot-22 repo scripts + readme overrides; paper: AAAI Table 2 (C=D=8)")
    ap.add_argument("--training", choices=["full", "sampled"], default=None)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--trans-heads", type=int, default=None)
    ap.add_argument("--sem-heads", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--t-max", type=int, default=16, help="CosineAnnealingLR T_max in epochs (0 = constant)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--eval-batch-size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None, help="early stopping on the val metric; -1 off")
    ap.add_argument("--fanouts", type=int, nargs="+", default=None)
    ap.add_argument("--eval-fanouts", type=int, nargs="+", default=None, help="default: same as --fanouts")
    ap.add_argument("--select", choices=["best_val", "last_epoch"], default=None)
    ap.add_argument("--select-metric", default="accuracy")
    ap.add_argument("--loss-nodes", choices=["seeds", "subgraph"], default=None,
                    help="sampled: seeds (clean default) | subgraph (benchmark TwiBot-22 label leak)")
    ap.add_argument("--bench-compat", action="store_true",
                    help="TwiBot-22 benchmark emulation: fan-out [256]*4, loss on every subgraph node "
                         "(LEAKS val/test labels), last epoch. Off by default; never a clean baseline")
    ap.add_argument("--amp", choices=["fp16", "bf16", "none"], default=None,
                    help="fp16 autocast + GradScaler (benchmark precision=16; default) | bf16 | none (fp32; "
                         "default with --cuda-graph)")
    ap.add_argument("--cuda-graph", action="store_true",
                    help="sampled training: replay each step as a CUDA graph on padded subgraphs (same "
                         "function; removes the per-step launch overhead; fp32/bf16 only)")
    ap.add_argument("--checkpoint", action="store_true",
                    help="activation checkpointing of the encoder and each RGT layer (same function, RNG "
                         "preserved; lower VRAM for full-graph TwiBot-20, ~1/3 more compute)")
    ap.add_argument("--edge-impl", choices=["auto", "torch", "triton"], default="auto",
                    help="attention/aggregation: torch gathers + index_add, or fused Triton gather kernels "
                         "(same function; auto = triton for full-graph training on CUDA)")
    ap.add_argument("--tail-impl", choices=["auto", "eager", "compile"], default="auto",
                    help="gate + semantic attention: eager, or torch.compile'd (same function, fixed shapes; "
                         "auto = compile for full-graph training on CUDA)")
    ap.add_argument("--graph-nodes", type=int, default=0, help="CUDA-graph node budget (0 = dry run max x 1.25)")
    ap.add_argument("--graph-edges", type=int, default=0, help="CUDA-graph edge budget (0 = dry run max x 1.25)")
    ap.add_argument("--extscore-inject", choices=["num", "branch", "late"], default="num",
                    help="num: [z, m] appended to num_prop (default); branch: fifth input projection; "
                         "late: concatenated to x2 before out1")
    ap.add_argument("--vram-cap-gb", type=float, default=0,
                    help="hard per-process CUDA allocator cap (torch set_per_process_memory_fraction); 0 = off")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--measure-steps", type=int, default=0,
                    help="measure K optimiser steps (+ val passes) of one seed and exit; no Run written; "
                         "zero placeholders for text tables that are not built yet")
    add_extscore_args(ap)
    a = ap.parse_args(argv)
    key = (a.dataset, a.preset)
    if key not in PRESETS:
        raise SystemExit(f"no preset {a.preset!r} for {a.dataset}")
    if a.bench_compat and a.patience is None:
        a.patience = -1     # the benchmark TwiBot-22 script has no val loop: no early stopping either
    for k, v in PRESETS[key].items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    if a.fanouts is None:
        a.fanouts = [256, 256]
    if a.bench_compat:
        if a.training != "sampled":
            raise SystemExit("--bench-compat emulates the sampled TwiBot-22 script; use --training sampled")
        a.fanouts = [256] * 4 if a.fanouts == [256, 256] else a.fanouts
        a.loss_nodes = a.loss_nodes or "subgraph"
        a.select = a.select or "last_epoch"
    a.loss_nodes = a.loss_nodes or "seeds"
    a.select = a.select or "best_val"
    a.amp = a.amp or ("none" if a.cuda_graph else "fp16")
    if a.cuda_graph and (a.training != "sampled" or a.loss_nodes != "seeds" or a.amp == "fp16"):
        raise SystemExit("--cuda-graph needs --training sampled, seed loss and --amp none|bf16")
    if a.loss_nodes == "subgraph" and a.training != "sampled":
        raise SystemExit("--loss-nodes subgraph applies to --training sampled only")
    return a


def variant_name(a) -> str:
    v = ["rgt", a.features]
    if a.preset != "benchmark":
        v.append(a.preset)
    if a.training != PRESETS[(a.dataset, a.preset)]["training"]:
        v.append(a.training)
    if a.training == "sampled" and a.fanouts != [256, 256]:
        v.append("fan" + "x".join(map(str, a.fanouts)))
    if a.bench_compat:
        v.append("benchcompat")
    elif a.loss_nodes != "seeds":
        v.append("loss-" + a.loss_nodes)
    if a.select != "best_val":
        v.append(a.select)
    if a.amp != "fp16":
        v.append(a.amp)
    if a.cuda_graph:
        v.append("cg")
    if (a.with_extscore or a.extscore == "p") and a.extscore_inject != "num":
        v.append(f"extscore{a.extscore_inject}")
    return ".".join(v)


def fit_config(a, seed: int, max_steps: int = 0):
    from .train import RGTFit
    return RGTFit(epochs=a.epochs, lr=a.lr, weight_decay=a.wd, t_max=a.t_max, batch_size=a.batch_size,
                  eval_batch_size=a.eval_batch_size, fanouts=tuple(a.fanouts),
                  eval_fanouts=tuple(a.eval_fanouts) if a.eval_fanouts else None, loss_nodes=a.loss_nodes,
                  select=a.select, select_metric=a.select_metric, patience=a.patience, amp=a.amp, seed=seed,
                  max_steps=max_steps, cuda_graph=a.cuda_graph, graph_nodes=a.graph_nodes,
                  graph_edges=a.graph_edges)


def _model(a, F, J, num_dim: int):
    from .model import build_model
    m = build_model(num_dim=num_dim, cat_dim=F.cat_prop.shape[1], hidden=a.hidden, trans_heads=a.trans_heads,
                    sem_heads=a.sem_heads, dropout=a.dropout, num_relations=F.num_relations,
                    extscore_width=J.width, extscore_inject=a.extscore_inject)
    m.checkpoint = a.checkpoint
    fast = a.training == "full" and a.device.startswith("cuda")
    edge = a.edge_impl if a.edge_impl != "auto" else ("triton" if fast else "torch")
    tail = a.tail_impl if a.tail_impl != "auto" else ("compile" if fast else "eager")
    return m.set_impl(edge, tail)


def main(argv=None) -> list[dict]:
    import torch

    from ...data import load_in_merged, load_undirected
    from ...features import load_features
    from ...gnn import n_params, split_metrics, to_device
    from .train import fit_full, fit_sampled

    a = parse_args(argv)
    t0 = time.time()
    F = load_features(a.dataset, a.features, require_text=not a.measure_steps)
    splits = {s: F.splits[s] for s in ("train", "val", "test")}
    y = F.label
    extscore_on = bool(a.with_extscore or a.extscore == "p")
    J = prepare_extscore(a, a.dataset, splits, labels=y,
                    neighbours=load_undirected(a.dataset, F.relations_by_type()) if extscore_on else None)
    dev = "cuda" if (a.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    if dev == "cuda" and a.vram_cap_gb:
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, a.vram_cap_gb * 1e9 / torch.cuda.get_device_properties(0).total_memory))
    if a.measure_steps and (F.des is None or F.tweets is None):
        g = _placeholder_inputs(F, dev, J, a.extscore_inject, edges=(a.training == "full"))
    else:
        g = to_device(F, dev, J=J, extscore_inject=a.extscore_inject, edges=(a.training == "full"),
                      placeholder_text=bool(a.measure_steps))
    csc = load_in_merged(a.dataset, F.relations_by_type()) if a.training == "sampled" else None
    load_s = round(time.time() - t0, 2)
    gpu_inputs = g.nbytes()
    num_dim = F.num_prop.shape[1] + (J.width if a.extscore_inject == "num" else 0)
    print(f"{a.dataset}/{F.variant}: N={F.n} E={F.edge_index.shape[1]} training={a.training} "
          f"C={a.trans_heads} D={a.sem_heads} num={tuple(g.num.shape)} cat={tuple(g.cat.shape)} "
          f"inputs on {dev}: {gpu_inputs / 1e9:.2f} GB (load {load_s} s) loss={a.loss_nodes} "
          f"extscore={'on:' + a.extscore_inject if J.enabled else 'off'}", flush=True)
    if a.measure_steps:
        return [measure(a, F, g, J, csc, num_dim, dev, load_s)]
    cfg_base = {**vars(a), "variant": variant_name(a), "features_variant": F.variant}
    results = []
    for seed in a.seeds:
        set_seed(seed, deterministic=a.deterministic)
        model = _model(a, F, J, num_dim).to(dev)
        fc = fit_config(a, seed)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with Run("rgt", a.dataset, config=cfg_base, seed=seed, data_files=F.files, source_files=SRC) as run:
            if a.training == "full":
                res = fit_full(model, g, splits, fc, run=run, log_every=a.log_every)
            else:
                res = fit_sampled(model, g, splits, fc, csc=csc, run=run, log_every=a.log_every)
            metrics = split_metrics(res.prob, y, splits)
            for s in ("val", "test"):
                run.add_predictions(s, splits[s], res.prob[splits[s]], y[splits[s]])
            out = run.finish(
                metrics, selection=res.selection, epochs_run=res.epochs_run, extscore=J.report(),
                n_params=n_params(model), with_extscore=a.with_extscore, leaky=(a.loss_nodes == "subgraph"),
                timing={"load_seconds": load_s, "train_seconds": res.train_seconds,
                        "seconds_per_epoch": res.seconds_per_epoch, "eval_seconds": res.eval_seconds,
                        "steps": res.steps},
                gpu={"inputs_bytes": gpu_inputs,
                     "run_max_allocated_bytes": int(torch.cuda.max_memory_allocated()) if dev == "cuda" else 0},
                features={"dir": str(F.dir), "input_key": F.meta["input_key"], "variant": F.variant})
        t = metrics["test"]
        print(f"[rgt {a.dataset} seed {seed}] test acc {t['accuracy']:.4f} f1 {t['f1']:.4f} "
              f"prec {t['precision']:.4f} rec {t['recall']:.4f} auc {t['roc_auc']:.4f} | "
              f"{res.seconds_per_epoch:.2f} s/epoch train, {res.epochs_run} epochs, best epoch "
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


def _placeholder_inputs(F, dev, J, extscore_inject: str, edges: bool):
    """Measurement only (text tables not built yet): like ``bb.gnn.to_device`` but the two text
    tables are one zero row expanded to [N, 768] (stride 0, no memory), so a TwiBot-22 timing run
    fits a light job. Gathers / projections see the same shapes; the resident table (N x 768 x 2 B
    per table) must be added to the measured VRAM by hand."""
    import torch

    from ...gnn import GraphInputs
    tdt = torch.float16 if torch.device(dev).type == "cuda" else torch.float32
    z = torch.zeros((1, 768), dtype=tdt, device=dev).expand(F.n, 768)
    t = lambda x, dt: torch.from_numpy(np.array(x, dtype=dt, copy=True)).to(dev)  # noqa: E731
    num = np.asarray(F.num_prop, dtype=np.float32)
    extra = None
    if J is not None and J.enabled:
        if extscore_inject == "num":
            num = J.append(num, kind="neural")
        else:
            extra = t(J.append(np.zeros((F.n, 0), np.float32), kind="neural"), np.float32)
    e = (t(F.edge_index, np.int64), t(F.edge_type, np.int64)) if edges else (
        torch.zeros((2, 0), dtype=torch.long, device=dev), torch.zeros(0, dtype=torch.long, device=dev))
    return GraphInputs(z, z, t(num, np.float32), t(F.cat_prop, np.float32), e[0], e[1], t(F.label, np.int64), extra)


def measure(a, F, g, J, csc, num_dim, dev, load_s) -> dict:
    """One seed, ``--measure-steps`` optimiser steps (one epoch at most) plus one val pass."""
    import torch

    from ...mem import memory_report
    from .train import fit_full, fit_sampled

    sp = {s: F.splits[s] for s in ("train", "val", "test")}
    warm = 0.0
    if a.training == "full" and _model(a, F, J, num_dim).rgt1.tail_path == "compile":
        # torch.compile happens on the first step (once per process); time it separately
        t = time.time()
        wc = fit_config(a, a.seeds[0], max_steps=2)
        wc.epochs = 1
        fit_full(_model(a, F, J, num_dim).to(dev), g, sp, wc)
        warm = round(time.time() - t, 1)
    set_seed(a.seeds[0])
    model = _model(a, F, J, num_dim).to(dev)
    fc = fit_config(a, a.seeds[0], max_steps=a.measure_steps)
    fc.epochs = 1
    rep = {"dataset": a.dataset, "training": a.training, "C": a.trans_heads, "D": a.sem_heads,
           "batch_size": a.batch_size, "fanouts": a.fanouts if a.training == "sampled" else None,
           "loss_nodes": a.loss_nodes, "amp": a.amp, "cuda_graph": a.cuda_graph,
           "placeholder_text": F.des is None or F.tweets is None,
           "load_seconds": load_s, "edge_path": model.rgt1.edge_path, "tail_path": model.rgt1.tail_path,
           "compile_warmup_seconds": warm}
    if a.training == "sampled":
        from ...harness.sampler import NeighborSampler
        rep["dry_run_train"] = NeighborSampler(csc, a.fanouts).dry_run(sp["train"], a.batch_size, n_batches=100)
        rep["dry_run_eval"] = NeighborSampler(csc, a.eval_fanouts or a.fanouts).dry_run(
            sp["val"], a.eval_batch_size, n_batches=50)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t = time.time()
    if a.training == "full":
        res = fit_full(model, g, sp, fc)
    else:
        res = fit_sampled(model, g, sp, fc, csc=csc, eval_splits=())
    steps_per_epoch = int(np.ceil(len(sp["train"]) / a.batch_size))
    ms = res.train_seconds / max(res.steps, 1) * 1e3
    rep.update({
        "steps": res.steps, "ms_per_step": round(ms, 2), "steps_per_epoch": steps_per_epoch,
        "projected_train_seconds_per_epoch": round(ms * steps_per_epoch / 1e3, 1),
        "val_pass_seconds": res.eval_seconds, "wall_seconds": round(time.time() - t, 2),
        "projected_seconds_per_epoch_incl_val": round(ms * steps_per_epoch / 1e3 + res.eval_seconds, 1),
        "gpu_inputs_GB": round(g.nbytes() / 1e9, 3),
        "cuda_max_allocated_GB": round(torch.cuda.max_memory_allocated() / 1e9, 3) if dev == "cuda" else 0,
        "cuda_max_reserved_GB": round(torch.cuda.max_memory_reserved() / 1e9, 3) if dev == "cuda" else 0,
        "memory": memory_report()})
    print("MEASURE " + json.dumps(rep, default=str), flush=True)
    return rep
