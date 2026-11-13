"""BotRGCN (Feng et al., ASONAM 2021) and its "w/o graph" MLP ablation (TwiBot-22 benchmark Table 3).

Spec: docs/methods/botrgcn.md. Features: the shared ``botrgcn`` scheme (bb/features/botrgcn.py).
Model: four input projections -> Linear+LeakyReLU -> one RGCNConv applied twice (tied, dropout
between) -> Linear+LeakyReLU -> Linear(D, 2); w/o graph replaces the convolutions by Linear(D, D) x 2.
Training: full batch, AdamW, unweighted CE on train rows, fixed epochs, report the LAST epoch
(``--select best_val`` is an off-by-default alternative).

    tools/memrun --light -- uv run python -m bb.methods.botrgcn --dataset cresci15
    tools/memrun --light -- uv run python -m bb.methods.botrgcn --dataset twibot20 --model wog
    tools/memrun -m 7G -- uv run python -m bb.methods.botrgcn --dataset twibot22 --seeds 0
    ... --with-extscore synthetic                       # ExtScore [z, m] appended to num_prop (5 -> 7)
    ... --preset original                          # TwiBot-20 original repo: D=128, lr 1e-3, wd 5e-3, 100 epochs
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ...harness import Run, aggregate, set_seed
from ...extscore import add_extscore_args, prepare_extscore

PRESETS = {  # spec section 5 (dropout 0.3 everywhere: the benchmark's 0.1 is never passed to the model)
    ("twibot22", "benchmark"): {"hidden": 32, "lr": 1e-2, "wd": 5e-2, "epochs": 200},
    ("twibot20", "benchmark"): {"hidden": 32, "lr": 1e-2, "wd": 5e-2, "epochs": 50},
    ("twibot20", "original"): {"hidden": 128, "lr": 1e-3, "wd": 5e-3, "epochs": 100},
    ("cresci15", "benchmark"): {"hidden": 32, "lr": 1e-2, "wd": 5e-2, "epochs": 200},
}
SRC = [Path(__file__), Path(__file__).with_name("model.py")]


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="BotRGCN / BotRGCN w/o graph")
    ap.add_argument("--dataset", choices=["twibot20", "twibot22", "cresci15"], default="cresci15")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", choices=["botrgcn", "wog", "mlp2"], default="botrgcn",
                    help="wog = BotRGCN with the two convolutions replaced by Linear(D,D) (encoder/head kept); "
                         "mlp2 = benchmark Table 3 'w/o graph' read literally: two-layer MLP on the concatenated "
                         "user features (reproduces the published TwiBot-20 row; review round 1)")
    ap.add_argument("--wog-tied", action="store_true", help="w/o graph: reuse one Linear(D, D) twice")
    ap.add_argument("--features", default="botrgcn", help="feature variant (bb.features.VARIANTS)")
    ap.add_argument("--preset", choices=["benchmark", "original"], default="benchmark")
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--select", choices=["last_epoch", "best_val"], default="last_epoch")
    ap.add_argument("--select-metric", default="f1")
    ap.add_argument("--patience", type=int, default=-1, help="early stopping (only with --select best_val); -1 off")
    ap.add_argument("--eval-every", type=int, default=1, help="val logging period (diagnostic only)")
    ap.add_argument("--class-weight", default="none")
    ap.add_argument("--training", choices=["full", "sampled"], default="full",
                    help="sampled = NeighborSampler mini-batches (fallback; changes the optimiser, deviation)")
    ap.add_argument("--fanouts", type=int, nargs="+", default=[-1, -1])
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--extscore-inject", choices=["num", "branch", "late"], default="num",
                    help="num: [z, m] appended to num_prop (default); branch: fifth input branch; late: before the classifier")
    ap.add_argument("--text-dtype", choices=["auto", "float16", "float32"], default="auto")
    ap.add_argument("--pyg-conv", action="store_true",
                    help="full batch: run the R-GCN through PyG's per-call masking/gather/scatter instead of the "
                         "cached sparse operators (same function; efficiency review round 2 A/B switch)")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--measure-epochs", type=int, default=0,
                    help="measure time/VRAM over K epochs of one seed and exit (no Run written; zero "
                         "placeholders for text tables that are not built yet)")
    add_extscore_args(ap)
    a = ap.parse_args(argv)
    key = (a.dataset, a.preset)
    if key not in PRESETS:
        raise SystemExit(f"no preset {a.preset!r} for {a.dataset}")
    for k, v in PRESETS[key].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    if a.model == "botrgcn" and len(a.fanouts) != 2 and a.training == "sampled":
        raise SystemExit("BotRGCN has two message-passing applications: give two fanouts")
    return a


def variant_name(a) -> str:
    v = [a.model, a.features]
    if a.preset != "benchmark":
        v.append(a.preset)
    if a.model == "wog" and a.wog_tied:
        v.append("tied")
    if a.training == "sampled":
        v.append("sampled" + "x".join(map(str, a.fanouts)))
    if a.select != "last_epoch":
        v.append(a.select)
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


def main(argv=None) -> list[dict]:
    import torch

    from ...features import load_features
    from ...gnn import FitConfig, fit_full_batch, fit_sampled, n_params, split_metrics, to_device
    from .model import build_model

    a = parse_args(argv)
    t0 = time.time()
    F = load_features(a.dataset, a.features, require_text=not a.measure_epochs)
    splits = {s: F.splits[s] for s in ("train", "val", "test")}
    y = F.label
    extscore_on = bool(a.with_extscore or a.extscore == "p")
    J = prepare_extscore(a, a.dataset, splits, labels=y,
                    neighbours=_undirected(a.dataset, F.relations_by_type()) if extscore_on else None)
    dev = "cuda" if (a.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    tdt = None if a.text_dtype == "auto" else getattr(torch, a.text_dtype)
    g = to_device(F, dev, J=J, extscore_inject=a.extscore_inject, text_dtype=tdt, edges=(a.training == "full"),
                  placeholder_text=bool(a.measure_epochs))
    csc = None
    if a.training == "sampled":
        from ...data import load_in_merged
        csc = load_in_merged(a.dataset, F.relations_by_type())
    load_s = round(time.time() - t0, 2)
    gpu_inputs = g.nbytes()
    num_dim = F.num_prop.shape[1] + (J.width if a.extscore_inject == "num" else 0)
    print(f"{a.dataset}/{F.variant}: N={F.n} E={F.edge_index.shape[1]} num={tuple(g.num.shape)} "
          f"cat={tuple(g.cat.shape)} text={g.des.dtype} inputs on {dev}: {gpu_inputs / 1e9:.2f} GB "
          f"(load {load_s} s) extscore={'on:' + a.extscore_inject if J.enabled else 'off'}", flush=True)
    if a.measure_epochs:
        return [measure(a, F, g, J, num_dim, dev, load_s)]
    method = {"botrgcn": "botrgcn", "wog": "botrgcn_wog", "mlp2": "botrgcn_mlp2"}[a.model]
    cfg_base = {**vars(a), "variant": variant_name(a), "features_variant": F.variant}
    results = []
    for seed in a.seeds:
        set_seed(seed, deterministic=a.deterministic)
        model = build_model(a.model, hidden=a.hidden, num_dim=num_dim, cat_dim=F.cat_prop.shape[1],
                            num_relations=F.num_relations, dropout=a.dropout, extscore_width=J.width,
                            extscore_inject=a.extscore_inject, wog_tied=a.wog_tied,
                            full_graph=a.training == "full" and not a.pyg_conv).to(dev)
        fc = FitConfig(epochs=a.epochs, lr=a.lr, weight_decay=a.wd, select=a.select, select_metric=a.select_metric,
                       patience=a.patience, eval_every=a.eval_every, class_weight=a.class_weight, seed=seed,
                       fanouts=tuple(a.fanouts), batch_size=a.batch_size)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with Run(method, a.dataset, config=cfg_base, seed=seed, data_files=F.files, source_files=SRC) as run:
            if a.training == "full":
                res = fit_full_batch(model, g, splits, fc, run=run, log_every=a.log_every)
            else:
                res = fit_sampled(model, g, splits, fc, csc=csc, run=run, log_every=a.log_every)
            metrics = split_metrics(res.prob, y, splits)
            for s in ("val", "test"):
                run.add_predictions(s, splits[s], res.prob[splits[s]], y[splits[s]])
            out = run.finish(
                metrics, selection=res.selection, epochs_run=res.epochs_run, extscore=J.report(),
                n_params=n_params(model), with_extscore=a.with_extscore,
                timing={"load_seconds": load_s, "train_seconds": res.train_seconds,
                        "seconds_per_epoch": res.seconds_per_epoch},
                gpu={"inputs_bytes": gpu_inputs,
                     "run_max_allocated_bytes": int(torch.cuda.max_memory_allocated()) if dev == "cuda" else 0},
                features={"dir": str(F.dir), "input_key": F.meta["input_key"], "variant": F.variant})
        t = metrics["test"]
        print(f"[{method} {a.dataset} seed {seed}] test acc {t['accuracy']:.4f} f1 {t['f1']:.4f} "
              f"prec {t['precision']:.4f} rec {t['recall']:.4f} auc {t['roc_auc']:.4f} | "
              f"{res.seconds_per_epoch * 1e3:.1f} ms/epoch, {res.epochs_run} epochs", flush=True)
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


def measure(a, F, g, J, num_dim, dev, load_s) -> dict:
    """One seed, ``--measure-epochs`` epochs: seconds per epoch, CUDA peak, host peak RSS."""
    import json

    import torch

    from ...gnn import FitConfig, fit_full_batch, fit_sampled
    from ...mem import memory_report
    from .model import build_model

    set_seed(a.seeds[0])
    model = build_model(a.model, hidden=a.hidden, num_dim=num_dim, cat_dim=F.cat_prop.shape[1],
                        num_relations=F.num_relations, dropout=a.dropout, extscore_width=J.width,
                        extscore_inject=a.extscore_inject, wog_tied=a.wog_tied,
                        full_graph=a.training == "full" and not a.pyg_conv).to(dev)
    fc = FitConfig(epochs=a.measure_epochs, lr=a.lr, weight_decay=a.wd, eval_every=a.eval_every,
                   fanouts=tuple(a.fanouts), batch_size=a.batch_size, seed=a.seeds[0])
    t = time.time()
    if a.training == "full":
        res = fit_full_batch(model, g, {s: F.splits[s] for s in ("train", "val", "test")}, fc)
    else:
        from ...data import load_in_merged
        res = fit_sampled(model, g, {s: F.splits[s] for s in ("train", "val", "test")}, fc,
                          csc=load_in_merged(a.dataset, F.relations_by_type()))
    rep = {"dataset": a.dataset, "model": a.model, "training": a.training, "epochs": res.epochs_run,
           "placeholder_text": F.des is None or F.tweets is None, "load_seconds": load_s,
           "train_seconds_per_epoch": res.seconds_per_epoch, "wall_seconds_incl_eval": round(time.time() - t, 2),
           "projected_seconds_full_run": round(res.seconds_per_epoch * a.epochs, 1),
           "gpu_inputs_GB": round(g.nbytes() / 1e9, 3),
           "cuda_max_allocated_GB": round(torch.cuda.max_memory_allocated() / 1e9, 3) if dev == "cuda" else 0,
           "cuda_max_reserved_GB": round(torch.cuda.max_memory_reserved() / 1e9, 3) if dev == "cuda" else 0,
           "memory": memory_report()}
    print("MEASURE " + json.dumps(rep, default=str), flush=True)
    return rep
