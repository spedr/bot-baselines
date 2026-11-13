"""SEBot (Yang et al., KDD 2024): structural-entropy multi-view contrastive bot detection.

Spec: docs/methods/sebot.md; results and deviations: docs/results/sebot.md.
Features: the shared ``botrgcn`` scheme (5 numeric + 3 categorical on TwiBot-20/22, 5 + 1 on
Cresci-2015, RoBERTa des + tweet means). Structure: a whole-graph coding tree of height k (alpha
view, sparse builder ``tree.py``, validated against the ported official ``PartitionTree``), one
ego-graph coding tree per labelled user (beta view), and an edge-dropped FACNConv backbone (gamma
view). Loss = CE(train) + l1 InfoNCE(alpha, gamma) + l2 InfoNCE(beta, gamma). Full batch, AdamW,
cosine LR (T_max 16, periodic), test metrics reported at the epoch of minimum validation loss.

    tools/memrun --light -- uv run python -m bb.methods.sebot --dataset cresci15
    tools/memrun --light -- uv run python -m bb.methods.sebot --dataset twibot20
    ... --with-extscore synthetic             # ExtScore [z, m] appended to num_prop in all three encoders
    ... --extscore-inject late|gamma          # ablations A (before the classifier) / B (gamma encoder only)
    ... --no-alpha / --no-beta / --no-rcm / --backbone rgcn     # paper Table 4 ablations
    ... --no-eval-noise                  # deterministic eval (FACNConv noise only in training)
    tools/memrun --light -- uv run python -m bb.methods.sebot --dataset twibot20 --build-only
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ...harness import Run, aggregate, compute_metrics, set_seed
from ...extscore import add_extscore_args, prepare_extscore

PRESETS = {  # spec section 5: paper Table 2 where it lists a value, the code otherwise
    "twibot20": {"k": 6, "lambda1": 0.09, "lambda2": 0.03, "epochs": 70, "in_cap": 0, "min_free_vram_gb": 2.5},
    "cresci15": {"k": 6, "lambda1": 0.09, "lambda2": 0.03, "epochs": 70, "in_cap": 0, "min_free_vram_gb": 1.0},
    "twibot22": {"k": 6, "lambda1": 0.09, "lambda2": 0.03, "epochs": 70, "in_cap": 10, "min_free_vram_gb": 6.0},
}
SRC = [Path(__file__).with_name(f) for f in ("__init__.py", "model.py", "tree.py", "artifacts.py", "train.py")]


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SEBot")
    ap.add_argument("--dataset", choices=["twibot20", "twibot22", "cresci15"], default="cresci15")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--features", default="botrgcn", help="feature variant (bb.features.VARIANTS)")
    ap.add_argument("--k", type=int, default=None, help="tree depth (paper 6, code default 4)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--input-dim", type=int, default=128)
    ap.add_argument("--proj-dim", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--pe", type=float, default=0.2, help="edge-drop probability of the gamma view")
    ap.add_argument("--lambda1", type=float, default=None, help="weight of InfoNCE(alpha, gamma)")
    ap.add_argument("--lambda2", type=float, default=None, help="weight of InfoNCE(beta, gamma)")
    ap.add_argument("--tau", type=float, default=0.1, help="InfoNCE temperature")
    ap.add_argument("--tau-g", type=float, default=0.01, help="FACNConv tanh temperature")
    ap.add_argument("--t-max", type=int, default=16, help="CosineAnnealingLR T_max (stepped per epoch)")
    ap.add_argument("--patience", type=int, default=50, help="stop after this many epochs without a lower val loss")
    ap.add_argument("--no-eval-noise", action="store_true",
                    help="FACNConv logistic noise only in training (official: also at eval)")
    ap.add_argument("--no-alpha", action="store_true", help="ablation: without the whole-graph tree view")
    ap.add_argument("--no-beta", action="store_true", help="ablation: without the subgraph-tree view")
    ap.add_argument("--no-rcm", action="store_true", help="ablation: without relational channel mixing")
    ap.add_argument("--backbone", choices=["facn", "rgcn"], default="facn", help="rgcn = 'R-GCN encoder' ablation")
    ap.add_argument("--subgraph-mode", choices=["faithful", "undirected"], default="faithful")
    ap.add_argument("--subgraph-seed", type=int, default=0, help="seed of out-edge capping and lone-node stars")
    ap.add_argument("--in-cap", type=int, default=None, help="cap in-neighbours per ego graph (0 = off)")
    ap.add_argument("--tree-nosmooth", action="store_true", help="whole-graph tree without the +1 smoothing")
    ap.add_argument("--rebuild-artifacts", action="store_true")
    ap.add_argument("--build-only", action="store_true", help="build/cache the trees and subgraphs, then exit")
    ap.add_argument("--nce-block", type=int, default=1024, help="InfoNCE row block (memory only)")
    ap.add_argument("--checkpoint-encoders", action=argparse.BooleanOptionalAction, default=False,
                    help="recompute the three feature encoders in backward on CUDA (same values; TwiBot-20: "
                         "-0.55 GB VRAM, +17%% time; off by default since the GPU runs one job at a time)")
    ap.add_argument("--extscore-inject", choices=["num", "late", "gamma"], default="num",
                    help="num: [z, m] in all three encoders (default); late: before the classifier; gamma: backbone only")
    ap.add_argument("--min-free-vram-gb", type=float, default=None,
                    help="exit with code 75 if the shared GPU has less free memory at start (0 = no check)")
    ap.add_argument("--allow-twibot22-full-batch", action="store_true", help="see docs/results/sebot.md (TwiBot-22)")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--measure-epochs", type=int, default=0, help="time K epochs of one seed and exit (no Run)")
    add_extscore_args(ap)
    a = ap.parse_args(argv)
    for k_, v in PRESETS[a.dataset].items():
        if getattr(a, k_) is None:
            setattr(a, k_, v)
    if a.k < 3:
        raise SystemExit("--k must be >= 3 (SEP-U uses tree levels 0-2, SEP-G levels 0-3)")
    return a


def variant_name(a) -> str:
    v = [f"k{a.k}"]
    if a.no_alpha:
        v.append("noalpha")
    if a.no_beta:
        v.append("nobeta")
    if a.no_rcm:
        v.append("norcm")
    if a.backbone != "facn":
        v.append(a.backbone)
    if a.no_eval_noise:
        v.append("noevalnoise")
    if a.subgraph_mode != "faithful":
        v.append("sub-" + a.subgraph_mode)
    if a.tree_nosmooth:
        v.append("treenosmooth")
    if a.features != "botrgcn":
        v.append(a.features)
    if (a.with_extscore or a.extscore == "p") and a.extscore_inject != "num":
        v.append(f"extscore{a.extscore_inject}")
    return ".".join(v)


def main(argv=None) -> list[dict]:
    from .train import prepare, run_seed, measure

    a = parse_args(argv)
    if a.dataset == "twibot22" and not a.build_only and not a.allow_twibot22_full_batch:
        raise SystemExit("SEBot trains full batch (spec 5); on TwiBot-22 that does not fit a 12 GB shared GPU "
                         "(projection in docs/results/sebot.md). Use --build-only for the trees, or "
                         "--allow-twibot22-full-batch to try anyway.")
    t0 = time.time()
    P = prepare(a)
    if a.build_only:
        print(f"[sebot] artifacts ready in {time.time() - t0:.1f}s")
        return []
    if a.measure_epochs:
        return [measure(a, P)]
    cfg = {**vars(a), "variant": variant_name(a), "features_variant": P.F.variant}
    results = []
    for seed in a.seeds:
        set_seed(seed, deterministic=a.deterministic)
        with Run("sebot", a.dataset, config=cfg, seed=seed, data_files=P.data_files, source_files=SRC) as run:
            out = run_seed(a, P, seed, run)
        t = out["metrics"]["test"]
        print(f"[sebot {a.dataset} seed {seed}] test acc {t['accuracy']:.4f} f1 {t['f1']:.4f} prec {t['precision']:.4f} "
              f"rec {t['recall']:.4f} auc {t['roc_auc']:.4f} | best epoch {out['selection']['epoch']} | "
              f"{out['timing']['seconds_per_epoch']:.2f} s/epoch", flush=True)
        results.append(out)
    agg = aggregate(results)
    print({k: (round(v["mean"], 4), round(v["std"], 4)) for k, v in agg.items()
           if k in ("accuracy", "f1", "precision", "recall", "macro_f1", "mcc", "roc_auc")})
    J = P.J
    if J.enabled and J.baseline:
        b = J.baseline["test"]
        print(f"column-only baseline (val-tuned threshold {J.baseline['threshold']:.3f}): "
              f"test acc {b['accuracy']:.4f} f1 {b['f1']:.4f}")
    return results


__all__ = ["main", "parse_args", "compute_metrics"]
