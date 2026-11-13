"""SEGCN (Liu et al., Sci. Rep. 14:4122, 2024): GNN-AK with a GCN kernel over random-walk subgraphs.

Spec: docs/methods/segcn.md (paper-first; GNN-AK operator details fill the gaps). The released code
(FengLiuii/SEGCN) is a renamed GNN-AK copy whose config never runs the subgraph path and whose
extractor needs dense N x N masks; this is our own sparse implementation:

- subgraphs.py: numba random walks (length 30, node2vec p/q, counter-based RNG) -> CSR-style store
  of node sets + induced edges + BFS hops, cached in data/methods/segcn/<dataset>/<key>/.
- model.py: typed encoders -> L GNN-AK layers (inner GCN on block-diagonal subgraph batches via
  sparse CSR matmul; centroid + pooled subgraph, concat FUSE) -> MLP head.
- train.py: mini-batches of train roots (B=512, lifted-row cap), exact layer-wise evaluation,
  best-val-accuracy selection; ``--inner 0`` = plain-GCN degenerate path (full graph).

    tools/memrun --light -- uv run python -m bb.methods.segcn --dataset cresci15
    tools/memrun --light -- uv run python -m bb.methods.segcn --dataset twibot20
    tools/memrun -m 7G -- uv run python -m bb.methods.segcn --dataset twibot22 --seeds 0 --measure-batches 200
    ... --with-extscore synthetic                     # ExtScore [z, m] appended to the numeric block (k -> k + 2)
    ... --inner 0                                # sanity path: plain GCN + BN + residual (spec 11.1)
    ... --build-store-only                       # step C only: build/cache the walk stores, print stats
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

# variable batch shapes fragment the caching allocator (TwiBot-22: 10 GB reserved for 5 GB allocated);
# expandable segments keep the reservation near the allocation (review round 2). Set before torch loads.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from ...harness import Run, aggregate, set_seed  # noqa: E402
from ...extscore import add_extscore_args, prepare_extscore  # noqa: E402

DEFAULT_FEATURES = {"twibot20": "botrgcn-orig6", "twibot22": "botrgcn", "cresci15": "botrgcn"}
SRC = [Path(__file__).with_name(n) for n in ("__init__.py", "model.py", "train.py", "subgraphs.py")]


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SEGCN (GNN-AK + GCN kernel, random-walk subgraphs)")
    ap.add_argument("--dataset", choices=["twibot20", "twibot22", "cresci15"], default="cresci15")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--features", default=None,
                    help="feature variant (default: twibot20 botrgcn-orig6 = Table 1's 6 numeric + 11 flags; "
                         "twibot22/cresci15 botrgcn)")
    ap.add_argument("--des-encoder", default="roberta-base",
                    help="description table encoder (spec 3.2: roberta-base [U]); 'variant' keeps the "
                         "feature variant's own table (TwiBot-20 orig6: distilroberta-base)")
    # architecture (spec 4, Table 4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2, help="outer GNN-AK layers L")
    ap.add_argument("--inner", type=int, default=1, help="inner GCN layers per kernel; 0 = plain GCN path")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--pool", choices=["mean", "sum"], default="mean")
    ap.add_argument("--no-residual", action="store_true",
                    help="drop the outer residual (GNN-AK res=True by default); with --inner 0 this is a "
                         "plain GCN + BN, the closest analogue of the benchmark GCN")
    ap.add_argument("--hop-dim", type=int, default=0, help="GNN-AK hop embedding (ablation 16)")
    # subgraphs (spec 3.4)
    ap.add_argument("--walk-length", type=int, default=30)
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--q", type=float, default=1.0)
    ap.add_argument("--repeat", type=int, default=1, help="walks per root (GNN-AK default 5; SEGCN yaml 1)")
    ap.add_argument("--resample", action="store_true", help="new training walks every epoch (GNN-AK online)")
    # training (spec 5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--wd", type=float, default=5e-3)
    ap.add_argument("--optimizer", choices=["adamw", "adam"], default="adamw")
    ap.add_argument("--step-lr", action="store_true", help="StepLR(50, 0.5) as in the released train.py")
    ap.add_argument("--select", choices=["best_val", "last_epoch"], default="best_val")
    ap.add_argument("--select-metric", default="accuracy")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-rows", type=int, default=600_000, help="cap on lifted rows per training batch")
    ap.add_argument("--eval-rows", type=int, default=400_000)
    ap.add_argument("--fuse", action="store_true",
                    help="CUDA: torch.compile the per-lifted-row BN/ReLU/dropout/residual and subgraph-pool chains "
                         "(same function, but eval logits differ from the eager path by up to ~8e-5 and inductor draws its own "
                         "dropout stream; opt-in, review round 2)")
    ap.add_argument("--vram-cap-gb", type=float, default=0,
                    help="hard per-process CUDA allocator cap (torch set_per_process_memory_fraction); 0 = off")
    ap.add_argument("--accumulate", action="store_true", help="one optimiser step per epoch (full-batch emulation)")
    ap.add_argument("--extscore-inject", choices=["num", "late", "branch", "root"], default="num",
                    help="num: [z,m] appended to the numeric block (default); late: before the head; "
                         "branch: fifth typed encoder (H=160); root: concat to the centroid only")
    ap.add_argument("--predict-train", action="store_true", help="also predict train rows (metrics)")
    ap.add_argument("--tag", default="", help="extra label for the run variant name")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--build-store-only", action="store_true")
    ap.add_argument("--count-rows", action="store_true",
                    help="CPU only: exact lifted rows of one training epoch (batch plans, no model) and exit")
    ap.add_argument("--measure-batches", type=int, default=0,
                    help="measure: one epoch truncated to N batches + one val eval, project; no Run written")
    ap.add_argument("--placeholder-text", action="store_true",
                    help="zero text tables if not built (measurement only, never results)")
    add_extscore_args(ap)
    a = ap.parse_args(argv)
    if a.features is None:
        a.features = DEFAULT_FEATURES[a.dataset]
    return a


def variant_name(a) -> str:
    v = ["segcn" if a.inner > 0 else "gcn0", a.features]
    if a.des_encoder != "variant":
        v.append("des-" + a.des_encoder)
    if a.inner > 0:
        v.append(f"L{a.layers}i{a.inner}")
        v.append(a.pool)
        if (a.p, a.q) != (1.0, 1.0):
            v.append(f"p{a.p:g}q{a.q:g}")
        if a.walk_length != 30:
            v.append(f"wl{a.walk_length}")
        if a.repeat != 1:
            v.append(f"r{a.repeat}")
        if a.hop_dim:
            v.append(f"hop{a.hop_dim}")
        if a.resample:
            v.append("resample")
        if a.accumulate:
            v.append("accum")
    else:
        v.append(f"L{a.layers}")
    if a.no_residual:
        v.append("nores")
    if a.select != "best_val":
        v.append(a.select)
    if a.step_lr:
        v.append("steplr")
    if a.optimizer != "adamw":
        v.append(a.optimizer)
    if (a.hidden, a.dropout, a.lr, a.wd, a.epochs) != (128, 0.3, 1e-2, 5e-3, 100):
        v.append(f"h{a.hidden}d{a.dropout:g}lr{a.lr:g}wd{a.wd:g}e{a.epochs}")
    if (a.with_extscore or a.extscore == "p") and a.extscore_inject != "num":
        v.append(f"extscore{a.extscore_inject}")
    if a.tag:
        v.append(a.tag)
    return ".".join(v)


def _store(a, rels, seed):
    from .subgraphs import load_or_build_store
    return load_or_build_store(a.dataset, rels, walk_length=a.walk_length, p=a.p, q=a.q, repeat=a.repeat,
                               seed=seed)


def _store_files(a, seed) -> list[Path]:
    from ...paths import dataset_paths
    from .subgraphs import store_key
    d = dataset_paths(a.dataset).method_dir("segcn") / store_key(a.walk_length, a.p, a.q, a.repeat, seed)
    return [d / "meta.json"]


def load_inputs(a, F, J, dev):
    """GraphInputs on ``dev`` (+ path of the description table actually used)."""
    import torch

    from ...gnn import to_device
    from ...gnn.inputs import table_to_device
    from ...paths import dataset_paths

    inj = {"num": "num", "late": "late", "branch": "branch", "root": "late"}[a.extscore_inject]
    g = to_device(F, dev, J=J, extscore_inject=inj, edges=False, placeholder_text=a.placeholder_text)
    des_path = None
    if a.des_encoder != "variant":
        des_path = dataset_paths(a.dataset).emb(a.des_encoder) / "des.f16.npy"
        cur = F.des.filename if F.des is not None else None
        if cur is None or Path(cur).resolve() != des_path.resolve():
            if des_path.exists():
                g.des = None
                torch.cuda.empty_cache() if dev == "cuda" else None
                g.des = table_to_device(des_path, dtype=torch.float16 if dev == "cuda" else torch.float32, device=dev)
            elif not a.placeholder_text:
                raise FileNotFoundError(des_path)
    return g, des_path


def main(argv=None) -> list[dict]:
    import torch

    from ...features import load_features
    from .model import SEGCN, GPUStore, n_params
    from .subgraphs import build_in_memory, segcn_graph
    from .train import TrainConfig, fit_global, fit_subgraph, global_adjacency, split_metrics_safe

    a = parse_args(argv)
    t0 = time.time()
    F = load_features(a.dataset, a.features, require_text=not (a.placeholder_text or a.build_store_only or a.count_rows))
    rels = F.relations_by_type()
    if a.build_store_only:
        out = []
        for s in a.seeds:
            st = _store(a, rels, s)
            out.append(st.meta)
            print(json.dumps({k: v for k, v in st.meta.items() if k != "graph_digest"}), flush=True)
        return out
    if a.count_rows:
        return [count_rows(a, F, rels)]
    splits = {s: F.splits[s] for s in ("train", "val", "test")}
    y = F.label
    csr = segcn_graph(a.dataset, rels)
    extscore_on = bool(a.with_extscore or a.extscore == "p")
    J = prepare_extscore(a, a.dataset, splits, labels=y, neighbours=csr if extscore_on else None)
    dev = "cuda" if (a.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    g, des_path = load_inputs(a, F, J, dev)
    load_s = round(time.time() - t0, 2)
    num_dim = F.num_prop.shape[1] + (J.width if a.extscore_inject == "num" else 0)
    print(f"{a.dataset}/{F.variant}: N={F.n} undirected nnz={csr.nnz} num={tuple(g.num.shape)} "
          f"cat={tuple(g.cat.shape)} des={des_path or 'variant'} inputs {g.nbytes() / 1e9:.2f} GB on {dev} "
          f"(load {load_s} s) extscore={'on:' + a.extscore_inject if J.enabled else 'off'}", flush=True)
    if dev == "cuda" and a.vram_cap_gb:
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, a.vram_cap_gb * 1e9 / torch.cuda.get_device_properties(0).total_memory))
    adj = global_adjacency(csr, dev) if a.inner == 0 else None
    cfg_base = {**vars(a), "variant": variant_name(a), "features_variant": F.variant,
                "des_table": str(des_path) if des_path else None}
    method = "segcn"
    results = []
    for seed in a.seeds:
        set_seed(seed, deterministic=a.deterministic)
        store = st_meta = None
        if a.inner > 0:
            st = _store(a, rels, seed)
            st_meta = {k: v for k, v in st.meta.items() if k != "graph_digest"}
            store = GPUStore(st, dev)
            del st
        set_seed(seed, deterministic=a.deterministic)
        model = SEGCN(num_dim=num_dim, cat_dim=F.cat_prop.shape[1], hidden=a.hidden, n_layers=a.layers,
                      n_inner=a.inner, dropout=a.dropout, hop_dim=a.hop_dim, pool=a.pool, extscore_width=J.width,
                      extscore_inject=a.extscore_inject, residual=not a.no_residual).to(dev)
        if a.fuse and dev == "cuda" and a.inner > 0:
            from .model import set_fuse
            set_fuse(model)
        cfg = TrainConfig(epochs=a.epochs, lr=a.lr, weight_decay=a.wd, optimizer=a.optimizer, step_lr=a.step_lr,
                          select=a.select, select_metric=a.select_metric, batch_size=a.batch_size,
                          max_rows=a.max_rows, eval_rows=a.eval_rows, accumulate=a.accumulate, seed=seed)
        resample = None
        if a.resample and store is not None:
            ip, ix = np.asarray(csr.indptr), np.asarray(csr.indices)

            def resample(ep, _seed=seed):
                return GPUStore(build_in_memory(ip, ix, walk_length=a.walk_length, p=a.p, q=a.q, repeat=a.repeat,
                                                seed=_seed * 1_000_003 + ep), dev)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        if a.measure_batches:
            results.append(measure(a, model, g, store, adj, splits, cfg, load_s, st_meta))
            break
        pred_splits = ("train", "val", "test") if a.predict_train else ("val", "test")
        data_files = [*F.files, *([des_path] if des_path else []), *(_store_files(a, seed) if store else [])]
        with Run(method, a.dataset, config=cfg_base, seed=seed, data_files=data_files, source_files=SRC) as run:
            if store is not None:
                res = fit_subgraph(model, g, store, splits, cfg, run=run, log_every=a.log_every, resample=resample,
                                   predict_splits=pred_splits)
            else:
                res = fit_global(model, g, adj, splits, cfg, run=run, log_every=a.log_every)
            metrics = split_metrics_safe(res.prob, y, splits)
            for s in ("val", "test"):
                run.add_predictions(s, splits[s], res.prob[splits[s]], y[splits[s]])
            out = run.finish(
                metrics, selection=res.selection, epochs_run=res.epochs_run, extscore=J.report(),
                n_params=n_params(model), with_extscore=a.with_extscore, store=st_meta,
                timing={"load_seconds": load_s, "train_seconds": res.train_seconds,
                        "seconds_per_epoch": res.seconds_per_epoch, "eval_seconds": res.eval_seconds,
                        "batches_per_epoch": res.batches_per_epoch, "lifted_rows_per_batch": res.rows_per_batch},
                store_host_bytes=store.nbytes() if store else 0,
                gpu={"inputs_bytes": g.nbytes(),
                     "run_max_allocated_bytes": int(torch.cuda.max_memory_allocated()) if dev == "cuda" else 0,
                     "run_max_reserved_bytes": int(torch.cuda.max_memory_reserved()) if dev == "cuda" else 0},
                features={"dir": str(F.dir), "input_key": F.meta["input_key"], "variant": F.variant})
        t = metrics["test"]
        print(f"[segcn {a.dataset} seed {seed}] test acc {t['accuracy']:.4f} f1 {t['f1']:.4f} "
              f"prec {t['precision']:.4f} rec {t['recall']:.4f} auc {t['roc_auc']:.4f} | best ep "
              f"{res.selection['best_epoch']} | {res.seconds_per_epoch:.2f} s/epoch train, "
              f"{res.eval_seconds / max(res.epochs_run, 1):.2f} s/epoch eval", flush=True)
        results.append(out)
        del model, store       # no empty_cache(): keep the reservation (WDDM may block re-allocation on a full shared GPU)
    if not a.measure_batches and results:
        agg = aggregate(results)
        print({k: (round(v["mean"], 4), round(v["std"], 4)) for k, v in agg.items()
               if k in ("accuracy", "f1", "precision", "recall", "macro_f1", "roc_auc")}, flush=True)
        if J.enabled and J.baseline:
            b = J.baseline["test"]
            print(f"column-only baseline (val-tuned threshold {J.baseline['threshold']:.3f}): "
                  f"test acc {b['accuracy']:.4f} f1 {b['f1']:.4f}", flush=True)
    return results


def measure(a, model, g, store, adj, splits, cfg, load_s, st_meta) -> dict:
    """Train ``--measure-batches`` batches of one epoch + one full val evaluation; project the cost."""
    import dataclasses

    import torch

    from ...mem import memory_report
    from .train import fit_global, fit_subgraph

    n_tr = int((np.asarray(g.y.cpu())[splits["train"]] >= 0).sum())
    t = time.time()
    if store is not None:
        fit_subgraph(model, g, store, splits, dataclasses.replace(cfg, epochs=1, max_batches=3),
                     predict_splits=())                       # warm-up (CUDA/cuBLAS init, numba cache), discarded
        c = dataclasses.replace(cfg, epochs=1, max_batches=a.measure_batches)
        res = fit_subgraph(model, g, store, splits, c, predict_splits=())
        from .train import pack_batches
        cost = store.cost(model.n_layers).cpu().numpy()
        n_batches = len(pack_batches(np.random.default_rng(0).permutation(splits["train"]), cost, cfg.batch_size,
                                     cfg.max_rows))
        per_batch = res.train_seconds / max(min(a.measure_batches, n_batches), 1)
        epoch_s = per_batch * n_batches
    else:
        res = fit_global(model, g, adj, splits, dataclasses.replace(cfg, epochs=2))
        n_batches, per_batch, epoch_s = 1, res.seconds_per_epoch, res.seconds_per_epoch
    eval_s = res.eval_seconds / max(res.epochs_run, 1)
    rep = {"dataset": a.dataset, "variant": variant_name(a), "train_roots": n_tr, "batches_per_epoch": n_batches,
           "measured_batches": min(a.measure_batches, n_batches), "seconds_per_batch": round(per_batch, 4),
           "lifted_rows_per_batch": round(res.rows_per_batch), "projected_train_seconds_per_epoch": round(epoch_s, 1),
           "val_eval_seconds": round(eval_s, 2),
           "projected_seconds_per_seed": round((epoch_s + eval_s) * a.epochs, 1),
           "projected_gpu_hours_5_seeds": round((epoch_s + eval_s) * a.epochs * 5 / 3600, 2),
           "placeholder_text": bool(a.placeholder_text), "load_seconds": load_s, "wall": round(time.time() - t, 1),
           "gpu_inputs_GB": round(g.nbytes() / 1e9, 3), "store_host_GB": round(store.nbytes() / 1e9, 3) if store else 0,
           "cuda_max_allocated_GB": round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else 0,
           "cuda_max_reserved_GB": round(torch.cuda.max_memory_reserved() / 1e9, 3) if torch.cuda.is_available() else 0,
           "store": st_meta, "memory": memory_report()}
    print("MEASURE " + json.dumps(rep, default=str), flush=True)
    return rep


def count_rows(a, F, rels) -> dict:
    """Exact lifted rows per level for one shuffled training epoch (CPU, batch plans only)."""
    import torch

    from ...mem import memory_report
    from .model import GPUStore, eval_sets
    from .train import pack_batches

    t = time.time()
    st = _store(a, rels, a.seeds[0])
    store = GPUStore(st, "cpu")
    cost = store.cost(a.layers).numpy()
    tr = np.asarray(F.splits["train"], np.int64)
    perm = np.random.default_rng(a.seeds[0] * 100_003 + 17).permutation(tr)
    batches = pack_batches(perm, cost, a.batch_size, a.max_rows)
    rows = np.zeros(a.layers, np.int64)
    s0 = 0
    per_batch = []
    for b in batches:
        r0, levels = store.plan(torch.as_tensor(b), a.layers)
        lv_rows = [lv.n_rows for lv in levels]
        rows += lv_rows
        s0 += len(r0)
        per_batch.append(sum(lv_rows))
    va = torch.as_tensor(np.asarray(F.splits["val"], np.int64))
    need = eval_sets(store, va, a.layers)
    sizes = store.sizes_np
    eval_rows = [int(sizes[need[l + 1]].sum()) for l in range(a.layers)]
    rep = {"dataset": a.dataset, "train_roots": len(tr), "batches": len(batches),
           "cost_upper_bound_rows": int(cost[tr].sum()), "lifted_rows_per_level": rows.tolist(),
           "lifted_rows_total": int(rows.sum()), "encoder_rows_S0": int(s0),
           "rows_per_batch_p50_p99_max": [float(np.quantile(per_batch, .5)), float(np.quantile(per_batch, .99)),
                                          int(max(per_batch))],
           "val_eval_rows_per_level": eval_rows, "val_eval_S0": int(len(need[0])),
           "seconds": round(time.time() - t, 1), "memory": memory_report()}
    print("COUNT " + json.dumps(rep, default=str), flush=True)
    return rep
