"""BotDGT (He et al., IJCAI 2024): dynamic graph transformer over yearly creation-date snapshots.

Spec: docs/methods/botdgt.md. Features: the shared ``botrgcn`` scheme (bb.features). Snapshots:
``snapshots.py`` (built once per dataset under data/methods/botdgt/<ds>/, seconds). Model:
``model.py``. Loops: ``train.py``.

    tools/memrun --light -- uv run python -m bb.methods.botdgt --dataset cresci15
    tools/memrun --light -- uv run python -m bb.methods.botdgt --dataset twibot20
    tools/memrun -m 7G -- uv run python -m bb.methods.botdgt --dataset twibot22 --seeds 0
    ... --with-extscore synthetic                 # ExtScore [z, m] appended to num_prop (5 -> 7), spec 8
    ... --extscore-inject late|token              # spec 8 ablations A / B
    ... --window 1                           # "w/o TM": last snapshot only
    ... --train-pos-signals                  # train p_LCC / p_BLR (official code leaves them frozen)
    ... --measure-batches 50                 # time / VRAM of 50 train batches + eval sample, then exit
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ...harness import Run, aggregate, set_seed
from ...extscore import add_extscore_args, prepare_extscore

# spec section 5 (README commands). Cresci-2015 has no official setting: the TwiBot-20 one is used
# (similar size; deviation documented in docs/results/botdgt.md).
PRESETS = {
    "twibot20": {"hidden": 128, "batch_size": 64, "wd": 1e-2, "lr_struct": 1e-4, "lr_temporal": 1e-5,
                 "epochs": 20, "eval_every": 1},
    "twibot22": {"hidden": 64, "batch_size": 256, "wd": 5e-2, "lr_struct": 5e-4, "lr_temporal": 5e-5,
                 "epochs": 20, "eval_every": 0},
    "cresci15": {"hidden": 128, "batch_size": 64, "wd": 1e-2, "lr_struct": 1e-4, "lr_temporal": 1e-5,
                 "epochs": 20, "eval_every": 1},
}
SRC = [Path(__file__).with_name(f) for f in ("__init__.py", "model.py", "train.py", "sampler.py", "snapshots.py")]


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="BotDGT")
    ap.add_argument("--dataset", choices=["twibot20", "twibot22", "cresci15"], default="cresci15")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--features", default="botrgcn", help="feature variant (bb.features.VARIANTS)")
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--lr-struct", type=float, default=None, help="feature embedding + structural layer")
    ap.add_argument("--lr-temporal", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--fanouts", type=int, nargs=2, default=[2560, 2560], help="train fan-outs (eval: all)")
    ap.add_argument("--interval", type=int, default=12, help="months between snapshots (12 = paper)")
    ap.add_argument("--window", type=int, default=-1, help="keep the last K snapshots (-1 all; 1 = w/o TM)")
    ap.add_argument("--coefficient", type=float, default=1.1, help="loss weight base per snapshot")
    ap.add_argument("--temporal-dropout", type=float, default=0.5)
    ap.add_argument("--train-pos-signals", action="store_true",
                    help="put p_LCC / p_BLR in the optimiser (the official code leaves them at init)")
    ap.add_argument("--sampling", choices=["nested", "per_snapshot"], default="nested",
                    help="nested: one 2-hop sample of the last snapshot, snapshot t = edges born <= t "
                         "(exact for uncapped rows; one layer-1 pass for all t); per_snapshot: T "
                         "independent samples stacked block-diagonally (official sampling)")
    ap.add_argument("--resample-epochs", action="store_true",
                    help="new shuffle + neighbour sample every epoch (official: sampled once, replayed)")
    ap.add_argument("--select", choices=["last_epoch", "best_val"], default="last_epoch")
    ap.add_argument("--patience", type=int, default=10, help="early stop patience with --select best_val")
    ap.add_argument("--eval-every", type=int, default=None, help="val diagnostics period (0 = off)")
    ap.add_argument("--eval-batch-size", type=int, default=1024)
    ap.add_argument("--max-union", type=int, default=400_000,
                    help="split a seed batch whose stacked snapshot subgraph has more nodes (VRAM guard)")
    ap.add_argument("--extscore-inject", choices=["num", "late", "token"], default="num",
                    help="num: [z, m] appended to num_prop (spec 8 primary); late: concat before the FFN "
                         "(ablation A); token: Linear(2, h) added to the temporal input (ablation B)")
    ap.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=True,
                    help="pad train batches to bucketed shapes and replay one CUDA graph per shape for "
                         "forward + backward (nested sampling on CUDA only; same maths)")
    ap.add_argument("--batch-cache", action=argparse.BooleanOptionalAction, default=True,
                    help="keep the packed epoch-0 train batches (and per-epoch val batches) resident and "
                         "replay them instead of re-sampling identical batches every epoch")
    ap.add_argument("--cache-gb", type=float, default=0.25, help="GPU budget of the batch cache (int32)")
    ap.add_argument("--cache-host-gb", type=float, default=1.5,
                    help="host budget of the batch cache after the GPU one (beyond: re-sampled)")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--measure-batches", type=int, default=0,
                    help="time K train batches and a 4k-seed eval sample of one seed, print, exit (no Run)")
    add_extscore_args(ap)
    a = ap.parse_args(argv)
    for k, v in PRESETS[a.dataset].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    return a


def variant_name(a) -> str:
    v = ["botdgt", a.features]
    if a.interval != 12:
        v.append(f"int{a.interval}")
    if a.window != -1:
        v.append(f"win{a.window}")
    if a.train_pos_signals:
        v.append("trainpos")
    if a.resample_epochs:
        v.append("resample")
    if a.sampling != "nested":
        v.append(a.sampling)
    if a.select != "last_epoch":
        v.append(a.select)
    if list(a.fanouts) != [2560, 2560]:
        v.append("fan" + "x".join(map(str, a.fanouts)))
    if (a.with_extscore or a.extscore == "p") and a.extscore_inject != "num":
        v.append(f"extscore{a.extscore_inject}")
    return ".".join(v)


def _window(T: int, window: int) -> slice:
    return slice(None) if window in (-1, 0) or window >= T else slice(T - window, T)


def build_model(a, T: int, num_dim: int, cat_dim: int, extscore_width: int):
    from .model import BotDGT
    return BotDGT(a.hidden, T, num_dim, cat_dim, temporal_dropout=a.temporal_dropout, extscore_width=extscore_width,
                  extscore_inject=a.extscore_inject)


def main(argv=None) -> list[dict]:
    import torch

    from ...features import load_features
    from ...gnn import n_params, split_metrics, to_device
    from .snapshots import _undirected, load_snapshots
    from .train import DGTConfig, fit

    a = parse_args(argv)
    t0 = time.time()
    F = load_features(a.dataset, a.features, require_text=not a.measure_batches)
    snap = load_snapshots(a.dataset, a.interval)
    win = _window(snap.T, a.window)
    T = len(range(snap.T)[win])
    splits = {s: F.splits[s] for s in ("train", "val", "test")}
    y = F.label
    extscore_on = bool(a.with_extscore or a.extscore == "p")
    J = prepare_extscore(a, a.dataset, splits, labels=y,
                    neighbours=_undirected(a.dataset, F.relations_by_type()) if extscore_on else None)
    dev = "cuda" if (a.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    placeholder = a.placeholder_text = bool(a.measure_batches) and (F.des is None or F.tweets is None)
    if placeholder:
        # measurement before the text tables exist: one zero row broadcast (stride 0) instead of
        # two [N, 768] zero tables, so a TwiBot-22 timing run fits a light slot (gathers are identical)
        import dataclasses
        z = np.zeros((1, 768), np.float16)
        F = dataclasses.replace(F, des=z, tweets=z)
    g = to_device(F, dev, J=J, extscore_inject="num" if a.extscore_inject == "num" else "late", edges=False)
    if placeholder:
        g.des, g.tweet = g.des.expand(F.n, -1), g.tweet.expand(F.n, -1)
    load_s = round(time.time() - t0, 2)
    num_dim = F.num_prop.shape[1] + (J.width if a.extscore_inject == "num" else 0)
    print(f"{a.dataset}/{F.variant}: N={F.n} T={T} ({snap.meta['dates'][win][0]}..{snap.meta['dates'][-1]}) "
          f"num={tuple(g.num.shape)} cat={tuple(g.cat.shape)} inputs on {dev}: {g.nbytes() / 1e9:.2f} GB "
          f"(load {load_s} s) extscore={'on:' + a.extscore_inject if J.enabled else 'off'}", flush=True)
    method = "botdgt"
    cfg_base = {**vars(a), "variant": variant_name(a), "features_variant": F.variant, "T": T,
                "snapshot_dates": snap.meta["dates"][win], "snapshot_input_key": snap.meta["input_key"]}

    def make_cfg(seed, **kw):
        return DGTConfig(epochs=a.epochs, batch_size=a.batch_size, lr_struct=a.lr_struct, lr_temporal=a.lr_temporal,
                         weight_decay=a.wd, fanouts=tuple(a.fanouts), coefficient=a.coefficient,
                         train_pos_signals=a.train_pos_signals, resample_epochs=a.resample_epochs,
                         select=a.select, patience=a.patience if a.select == "best_val" else -1,
                         eval_every=a.eval_every, eval_batch_size=a.eval_batch_size, max_union=a.max_union,
                         seed=seed, log_every=a.log_every, sampling=a.sampling, cuda_graphs=a.cuda_graphs,
                         cache_batches=a.batch_cache, cache_gb=a.cache_gb,
                         cache_host_gb=a.cache_host_gb, **kw)

    if a.measure_batches:
        return [measure(a, F, g, snap, win, T, num_dim, J, make_cfg, dev, load_s)]
    results = []
    for seed in a.seeds:
        set_seed(seed, deterministic=a.deterministic)
        model = build_model(a, T, num_dim, F.cat_prop.shape[1], J.width).to(dev)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with Run(method, a.dataset, config=cfg_base, seed=seed, data_files=F.files + snap.files,
                 source_files=SRC) as run:
            res = fit(model, g, snap, splits, make_cfg(seed), window=win, run=run)
            metrics = split_metrics(res.prob, y, splits)
            for s in ("val", "test"):
                run.add_predictions(s, splits[s], res.prob[splits[s]], y[splits[s]])
            out = run.finish(
                metrics, selection=res.selection, epochs_run=res.epochs_run, extscore=J.report(),
                n_params=n_params(model), with_extscore=a.with_extscore,
                timing={"load_seconds": load_s, "train_seconds": res.train_seconds,
                        "seconds_per_epoch": res.seconds_per_epoch},
                batch_stats=res.batch_stats,
                gpu={"inputs_bytes": g.nbytes(),
                     "run_max_allocated_bytes": int(torch.cuda.max_memory_allocated()) if dev == "cuda" else 0},
                features={"dir": str(F.dir), "input_key": F.meta["input_key"], "variant": F.variant},
                snapshots={"dir": str(snap.dir), "T": T, "dates": snap.meta["dates"][win],
                           "nodes_per_snapshot": snap.meta["nodes_per_snapshot"][win],
                           "directed_edges_per_snapshot": snap.meta["directed_edges_per_snapshot"][win]})
        t = metrics["test"]
        print(f"[botdgt {a.dataset} seed {seed}] test acc {t['accuracy']:.4f} f1 {t['f1']:.4f} "
              f"prec {t['precision']:.4f} rec {t['recall']:.4f} auc {t['roc_auc']:.4f} | "
              f"{res.seconds_per_epoch:.1f} s/epoch, {res.epochs_run} epochs", flush=True)
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


def measure(a, F, g, snap, win, T, num_dim, J, make_cfg, dev, load_s) -> dict:
    """``--measure-batches K``: K train batches for two epochs (sample + capture, then replay) +
    prediction of 4,096 test seeds; projections for a full run. Zero placeholders replace text
    tables that are not built."""
    import json

    import torch

    from ...mem import memory_report
    from .train import fit, predict

    set_seed(a.seeds[0])
    model = build_model(a, T, num_dim, F.cat_prop.shape[1], J.width).to(dev)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    cfg = make_cfg(a.seeds[0], max_batches=a.measure_batches)
    # two epochs over the same K batches: epoch 0 samples, packs and captures CUDA graphs; epoch 1
    # replays (what the remaining epochs of a real run cost)
    cfg.epochs, cfg.eval_every = 2, 0
    res = fit(model, g, snap, {s: F.splits[s] for s in ("train", "val", "test")}, cfg, window=win,
              predict_splits=())
    te = np.asarray(F.splits["test"])
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(te, size=min(4096, len(te)), replace=False))
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    predict(model, g, snap, sample, win, a.eval_batch_size, a.max_union, a.sampling == "nested")
    if dev == "cuda":
        torch.cuda.synchronize()
    ev_s = time.perf_counter() - t
    n_tr = int((F.label[F.splits["train"]] >= 0).sum())
    nb = -(-n_tr // a.batch_size)
    k = max(1, min(a.measure_batches, nb))
    first, per_batch = res.history[0]["train_seconds"] / k, res.history[1]["train_seconds"] / k
    n_eval = len(F.splits["test"]) + (len(F.splits["val"]) if a.eval_every else 0)
    proj_epoch = per_batch * nb
    proj_eval = ev_s / len(sample) * len(F.splits["test"])
    rep = {"dataset": a.dataset, "T": T, "batches_measured": min(a.measure_batches, nb), "batches_per_epoch": nb,
           "seconds_per_batch": round(per_batch, 4), "projected_seconds_per_epoch": round(proj_epoch, 1),
           "eval_seconds_per_1k_seeds": round(ev_s / len(sample) * 1000, 2),
           "projected_seconds_test_eval": round(proj_eval, 1),
           "seconds_per_batch_first_epoch": round(first, 4),
           "projected_seconds_per_seed": round(first * nb + proj_epoch * (a.epochs - 1) + proj_eval
                                               + (ev_s / len(sample) * n_eval * a.epochs if a.eval_every else 0), 1),
           "placeholder_text": bool(getattr(a, "placeholder_text", False)), "load_seconds": load_s,
           "batch_stats": res.batch_stats, "gpu_inputs_GB": round(g.nbytes() / 1e9, 3),
           "cuda_max_allocated_GB": round(torch.cuda.max_memory_allocated() / 1e9, 3) if dev == "cuda" else 0,
           "cuda_max_reserved_GB": round(torch.cuda.max_memory_reserved() / 1e9, 3) if dev == "cuda" else 0,
           "memory": memory_report()}
    print("MEASURE " + json.dumps(rep, default=str), flush=True)
    return rep
