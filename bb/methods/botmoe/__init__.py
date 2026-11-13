"""BotMoE (Liu et al., SIGIR 2023): community-aware mixture of modal-specific experts.

Spec: docs/methods/botmoe.md. Built: the released-code architecture ``AllInOne1_rgcn_rgt_gcn``
(``variant: code``; model.py) on the shared ``botrgcn`` feature scheme, trained as the released
``main.py`` (train.py). The metadata expert reads concat(botrgcn num, cat) (``--meta-input
botrgcn8``, the paper's metadata and the default); the released pipeline's misaligned input, which
is all zeros for every labelled node, is behind ``--meta-input zeros`` (or ``released`` for the
exact TwiBot-20 layout). The paper architecture (``variant: paper``) is not built.

    tools/memrun --light -- uv run python -m bb.methods.botmoe --dataset cresci15
    tools/memrun --light -- uv run python -m bb.methods.botmoe --dataset twibot20
    tools/memrun -m 7G -- uv run python -m bb.methods.botmoe --dataset twibot22 --seeds 0
    ... --with-extscore TAG                      # [z, m] joins the metadata-expert input (8 -> 10)
    ... --with-extscore TAG --extscore-inject num     # generic input level: appended to botrgcn num (graph branches)
    ... --measure-epochs 1                  # time / VRAM of one epoch, no Run written
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ...harness import Run, aggregate, set_seed
from ...extscore import add_extscore_args, prepare_extscore
from .meta import META_INPUTS
from .model import EXTSCORE_AT

# Budget/schedule per dataset (spec 5). TwiBot-20 = released main.py; Cresci-2015 and TwiBot-22 code
# is unreleased: their thresholds would never/always fire, so the gates unfreeze by epoch only.
PRESETS = {
    "twibot20": {"lr": 1e-5, "epochs": 400, "patience": 200, "unfreeze_acc": 0.865, "unfreeze_epoch": 200,
                 "vram_cap_gb": 2.5},
    "cresci15": {"lr": 1e-4, "epochs": 400, "patience": 200, "unfreeze_acc": None, "unfreeze_epoch": 200,
                 "vram_cap_gb": 2.0},
    # TwiBot-22 (code unreleased; spec 5/7 budget choices [U]): 2048 seeds per batch (subgraphs saturate
    # at the ~10k users with out-edges: 342 steps of 11k nodes instead of 684 of 9.6k), 40 epochs max,
    # patience 8, gates unfrozen after epoch 10. No activation checkpointing (review round 2: with the
    # fused RGT kernels the GCN/RGCN recompute bought only 0.15 GB and cost 14% of the step; measured
    # below 6 GB VRAM incl. the 3.07 GB text tables).
    "twibot22": {"lr": 1e-4, "epochs": 40, "patience": 8, "unfreeze_acc": None, "unfreeze_epoch": 10,
                 "vram_cap_gb": 6.5, "grad_checkpoint": False, "tf32": True, "batch_size": 2048},
}
SRC = [Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("train.py"),
       Path(__file__).with_name("meta.py"), Path(__file__).with_name("rgt_chunked.py"),
       Path(__file__).with_name("plan.py"), Path(__file__).parent.parent / "rgt" / "model.py"]


def _none_float(s: str):
    return None if s.lower() in ("none", "off", "-1") else float(s)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="BotMoE (released-code architecture)")
    ap.add_argument("--dataset", choices=sorted(PRESETS), default="cresci15")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--features", default="botrgcn", help="feature variant (bb.features.VARIANTS)")
    ap.add_argument("--meta-input", choices=META_INPUTS, default="botrgcn8",
                    help="metadata-expert input: botrgcn8 (fixed, default) | hayawi9 | zeros / released "
                         "(emulate the released misalignment: labelled nodes see zeros)")
    ap.add_argument("--hayawi-bugs", action=argparse.BooleanOptionalAction, default=True,
                    help="hayawi9: keep listed/active_days and the p^2 'entropy' (benchmark preprocessing)")
    ap.add_argument("--extscore-inject", choices=EXTSCORE_AT, default="metadata",
                    help="metadata (default): [z, m] joins the metadata-expert input (and its gate); "
                         "num: generic input level, appended to the botrgcn num block (graph branches); "
                         "metadata+num; token: 6th fusion token; head: classifier input; gate_only: every MoE gate")
    ap.add_argument("--align", type=int, default=128, help="align_size A (graph width E = 4A)")
    ap.add_argument("--out-dim", type=int, default=4, help="expert output / fusion token width O")
    ap.add_argument("--experts", type=int, nargs=3, default=[3, 2, 3], metavar=("GRAPH", "TEXT", "META"))
    ap.add_argument("--moe-k", type=int, default=1, help="top-k of every MoE (k=1: gates learn from the balance loss only)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None, help="epochs without val-acc improvement; -1 off")
    ap.add_argument("--unfreeze-acc", default="preset",
                    help="unfreeze graph gates once best val acc exceeds this (none = epoch only)")
    ap.add_argument("--unfreeze-epoch", type=int, default=None)
    ap.add_argument("--unfreeze-lr-factor", type=float, default=0.1, help="lr *= factor at unfreeze (code 1e-5 -> 1e-6)")
    ap.add_argument("--no-gate-freeze", action="store_true", help="train graph gates from the start")
    ap.add_argument("--batch-size", type=int, default=None, help="seeds per batch (released: 1024)")
    ap.add_argument("--eval-batch-size", type=int, default=4096)
    ap.add_argument("--fanouts", type=int, nargs=2, default=[256, 256])
    ap.add_argument("--eval-fanouts", type=int, nargs=2, default=None, help="default: same as --fanouts (released)")
    ap.add_argument("--amp", choices=["none", "bf16"], default="none")
    ap.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=None,
                    help="TF32 tensor-core matmuls (fp32 storage; TwiBot-22 preset on)")
    ap.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=None,
                    help="activation checkpointing of the GCN/RGCN convolutions (exact; off in every "
                         "preset since the fused RGT kernels)")
    ap.add_argument("--vram-cap-gb", type=float, default=None,
                    help="hard CUDA allocator cap (set_per_process_memory_fraction); 0 = off. Without it the "
                         "WSL driver lets the caching allocator spill into shared host memory (measured: 28 GB "
                         "reserved, ~4x slower steps) instead of freeing its cache")
    ap.add_argument("--plan", action=argparse.BooleanOptionalAction, default=True,
                    help="build each batch's edge structures on the host (one async copy, no host syncs per "
                         "step; same function). --no-plan: derive them on the device (reference path)")
    ap.add_argument("--text-dtype", choices=["auto", "float16", "float32"], default="auto")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--measure-epochs", type=int, default=0,
                    help="time/VRAM of K epochs of one seed, then exit (no Run; zero placeholders for text "
                         "tables that are not built yet)")
    ap.add_argument("--max-batches", type=int, default=0, help="measurement only: batches per epoch cap")
    add_extscore_args(ap)
    a = ap.parse_args(argv)
    if a.batch_size is None:
        a.batch_size = PRESETS[a.dataset].get("batch_size", 1024)
    for k in ("grad_checkpoint", "tf32"):
        if getattr(a, k) is None:
            setattr(a, k, PRESETS[a.dataset].get(k, False))
    for k, v in PRESETS[a.dataset].items():
        if k == "unfreeze_acc":
            a.unfreeze_acc = v if a.unfreeze_acc == "preset" else _none_float(a.unfreeze_acc)
        elif getattr(a, k) is None:
            setattr(a, k, v)
    if a.meta_input == "released" and a.dataset != "twibot20":
        raise SystemExit("--meta-input released exists for TwiBot-20 only (use zeros)")
    return a


def extscore_on(a) -> bool:
    return bool(a.with_extscore or a.extscore == "p")


def variant_name(a) -> str:
    v = ["code", a.meta_input + ("" if a.meta_input != "hayawi9" or a.hayawi_bugs else "-fixed")]
    if a.features != "botrgcn":
        v.append(a.features)
    pre = PRESETS[a.dataset]
    if a.lr != pre["lr"]:
        v.append(f"lr{a.lr:g}")
    if a.batch_size != pre.get("batch_size", 1024):
        v.append(f"bs{a.batch_size}")
    if a.moe_k != 1:
        v.append(f"k{a.moe_k}")
    if a.no_gate_freeze:
        v.append("nofreeze")
    if a.amp != "none":
        v.append(a.amp)
    if extscore_on(a) and a.extscore_inject != "metadata":
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


class _OneRowText:
    """Proxy of a feature set whose missing text tables become [1, 768] placeholders in ``to_device``."""

    def __init__(self, F):
        self._F = F

    def __getattr__(self, k):
        return getattr(self._F, k)

    n = property(lambda self: 1)
    des = property(lambda self: None)
    tweets = property(lambda self: None)


def setup_cuda(a) -> None:
    """Allocator settings before the first CUDA allocation: expandable segments (less fragmentation
    with variable subgraph sizes) and a hard per-process cap."""
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    torch.backends.cuda.matmul.allow_tf32 = bool(a.tf32)
    torch.backends.cudnn.allow_tf32 = bool(a.tf32)
    if a.device.startswith("cuda") and torch.cuda.is_available() and a.vram_cap_gb:
        tot = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_cap_gb * 1e9 / tot))


def prepare(a, *, placeholder: bool = False):
    """Features, ExtScore, device inputs. Returns (F, J, X, dims, csc, load_seconds)."""
    import torch

    setup_cuda(a)

    from ...data import load_in_merged
    from ...features import load_features
    from ...gnn import to_device
    from .meta import meta_input
    from .train import Inputs

    t0 = time.time()
    F = load_features(a.dataset, a.features, require_text=not placeholder)
    splits = {s: F.splits[s] for s in ("train", "val", "test")}
    J = prepare_extscore(a, a.dataset, splits, labels=F.label,
                    neighbours=_undirected(a.dataset, F.relations_by_type()) if extscore_on(a) else None)
    dev = "cuda" if (a.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    tdt = None if a.text_dtype == "auto" else getattr(torch, a.text_dtype)
    if placeholder:
        # measurement only: one zero text row expanded to N (a view: no 3 GB of placeholder VRAM)
        g = to_device(_OneRowText(F), dev, text_dtype=tdt, edges=False, placeholder_text=True)
        g.des, g.tweet = g.des.expand(F.n, -1), g.tweet.expand(F.n, -1)
    else:
        g = to_device(F, dev, text_dtype=tdt, edges=False)
    meta, meta_info = meta_input(a.meta_input, F, hayawi_bugs=a.hayawi_bugs)
    extscore = (torch.from_numpy(J.append(np.zeros((F.n, 0), np.float32), kind="neural").astype(np.float32)).to(dev)
           if J.enabled else None)                                   # [N, 2] (z, m)
    inj = a.extscore_inject if J.enabled else None
    if inj in ("metadata", "metadata+num"):
        meta = J.append(meta, kind="neural")
    if inj in ("num", "metadata+num"):
        g.num = torch.cat([g.num, extscore], dim=1)                      # == J.append(num_prop, kind="neural")
    meta_t = torch.from_numpy(np.ascontiguousarray(meta, dtype=np.float32)).to(dev)
    csc = load_in_merged(a.dataset, F.relations_by_type())
    etype = np.asarray(csc.etype, dtype=np.int64)
    X = Inputs(g, meta_t, extscore if inj in ("token", "head", "gate_only") else None, etype,
               num_relations=F.num_relations, plan=a.plan)
    dims = {"num_dim": int(g.num.shape[1]), "cat_dim": int(g.cat.shape[1]), "meta_dim": int(meta_t.shape[1]),
            "extscore_width": J.width, "extscore_at": a.extscore_inject, "num_relations": F.num_relations}
    return F, J, X, dims, csc, meta_info, round(time.time() - t0, 2)


def make_model(a, dims):
    from .model import build_model
    g, t, m = a.experts
    model = build_model(align=a.align, out_dim=a.out_dim, n_graph=g, n_text=t, n_meta=m, k_graph=a.moe_k,
                        k_text=a.moe_k, k_meta=a.moe_k, **dims)
    model.set_checkpoint(bool(a.grad_checkpoint))
    return model


def train_config(a, seed: int, **over):
    from .train import TrainConfig
    kw = dict(epochs=a.epochs, lr=a.lr, weight_decay=a.wd, batch_size=a.batch_size,
              eval_batch_size=a.eval_batch_size, fanouts=tuple(a.fanouts),
              eval_fanouts=tuple(a.eval_fanouts) if a.eval_fanouts else None, patience=a.patience,
              unfreeze_acc=a.unfreeze_acc, unfreeze_epoch=a.unfreeze_epoch,
              unfreeze_lr_factor=a.unfreeze_lr_factor, freeze_graph_gates=not a.no_gate_freeze, seed=seed,
              amp=a.amp, max_batches=a.max_batches)
    kw.update(over)
    return TrainConfig(**kw)


def main(argv=None) -> list[dict]:
    import torch

    from ...gnn import n_params, split_metrics
    from .train import fit

    a = parse_args(argv)
    if a.measure_epochs:
        return [measure(a)]
    F, J, X, dims, csc, meta_info, load_s = prepare(a)
    y = F.label
    splits = {s: F.splits[s] for s in ("train", "val", "test")}
    dev = X.g.device
    print(f"{a.dataset}/{F.variant}: N={F.n} E={F.edge_index.shape[1]} dims={dims} meta={a.meta_input} "
          f"inputs {X.g.nbytes() / 1e9:.2f} GB on {dev} (load {load_s} s) "
          f"extscore={'on:' + a.extscore_inject if J.enabled else 'off'}", flush=True)
    cfg_base = {**vars(a), "variant": variant_name(a), "features_variant": F.variant, "meta_info": meta_info}
    results = []
    for seed in a.seeds:
        set_seed(seed, deterministic=a.deterministic)
        model = make_model(a, dims).to(dev)
        cfg = train_config(a, seed)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with Run("botmoe", a.dataset, config=cfg_base, seed=seed, data_files=F.files, source_files=SRC) as run:
            res = fit(model, X, y, splits, cfg, csc=csc, run=run, log_every=a.log_every)
            metrics = split_metrics(res.prob, y, splits, names=("val", "test"))
            for s in ("val", "test"):
                run.add_predictions(s, splits[s], res.prob[splits[s]], y[splits[s]])
            out = run.finish(
                metrics, selection=res.selection, epochs_run=res.epochs_run, extscore=J.report(),
                n_params=n_params(model), with_extscore=a.with_extscore, expert_load=res.routes,
                unfreeze_epoch=res.unfreeze_epoch,
                timing={"load_seconds": load_s, "train_seconds": res.train_seconds,
                        "seconds_per_epoch": res.seconds_per_epoch,
                        "eval_seconds_per_epoch": res.eval_seconds_per_epoch},
                gpu={"inputs_bytes": X.g.nbytes(),
                     "run_max_allocated_bytes": int(torch.cuda.max_memory_allocated()) if dev.type == "cuda" else 0},
                features={"dir": str(F.dir), "input_key": F.meta["input_key"], "variant": F.variant})
        t = metrics["test"]
        print(f"[botmoe {a.dataset} seed {seed}] test acc {t['accuracy']:.4f} f1 {t['f1']:.4f} "
              f"macro {t['macro_f1']:.4f} auc {t['roc_auc']:.4f} | best ep {res.selection['best_epoch']} "
              f"of {res.epochs_run}, unfreeze {res.unfreeze_epoch}, {res.seconds_per_epoch:.2f} s/ep "
              f"| test load {json.dumps(res.routes['test'])}", flush=True)
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


def measure(a) -> dict:
    """``--measure-epochs K`` on the first seed: seconds per epoch (train, val), CUDA peak, host RSS,
    sampled subgraph sizes. Text tables that are not built yet are zero placeholders."""
    import torch

    from ...features import load_features
    from ...mem import memory_report
    from .train import fit

    Fchk = load_features(a.dataset, a.features, require_text=False)
    placeholder = Fchk.des is None or Fchk.tweets is None
    F, J, X, dims, csc, meta_info, load_s = prepare(a, placeholder=placeholder)
    splits = {s: F.splits[s] for s in ("train", "val", "test")}
    set_seed(a.seeds[0])
    dev = X.g.device
    model = make_model(a, dims).to(dev)
    from ...harness.sampler import NeighborSampler
    dry = NeighborSampler(csc, list(a.fanouts)).dry_run(np.asarray(splits["train"]), a.batch_size, n_batches=50)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    cfg = train_config(a, a.seeds[0], epochs=a.measure_epochs, patience=-1)
    t = time.time()
    res = fit(model, X, F.label, splits, cfg, csc=csc, log_every=1)
    from ...gnn import n_params
    n_tr = len(splits["train"])
    bpe = -(-n_tr // a.batch_size)
    rep = {"dataset": a.dataset, "placeholder_text": placeholder, "load_seconds": load_s, "n_params": n_params(model),
           "batches_per_epoch": bpe, "max_batches": a.max_batches,
           "train_seconds_per_epoch": res.seconds_per_epoch, "val_seconds_per_epoch": res.eval_seconds_per_epoch,
           "wall_seconds": round(time.time() - t, 1),
           "projected_train_s_per_full_epoch": round(res.seconds_per_epoch * (bpe / min(a.max_batches, bpe) if a.max_batches else 1), 1),
           "subgraph_dry_run": {k: dry[k] for k in ("nodes", "edges", "ms_per_batch")},
           "gpu_inputs_GB": round(X.g.nbytes() / 1e9, 3),
           "cuda_max_allocated_GB": round(torch.cuda.max_memory_allocated() / 1e9, 3) if dev.type == "cuda" else 0,
           "cuda_max_reserved_GB": round(torch.cuda.max_memory_reserved() / 1e9, 3) if dev.type == "cuda" else 0,
           "memory": memory_report(), "val_metrics_after": res.history[-1] if res.history else None}
    print("MEASURE " + json.dumps(rep, default=str), flush=True)
    return rep
