"""SEBot data preparation and full-batch training (spec sections 3, 5)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as Fn

from ...harness import compute_metrics
from ...extscore import prepare_extscore
from .artifacts import build_subgraph_trees, build_whole_tree, method_dir, tree_graph
from .model import SEBot, n_params


def relational_undirected(edge_index: np.ndarray, edge_type: np.ndarray, R: int):
    """Official ``relational_undirected``: per relation the unique directed edges plus their
    reverses, relations concatenated in type order (a pair present in both directions of a
    relation, or in both relations, appears more than once)."""
    es, ts = [], []
    for r in range(R):
        e = np.asarray(edge_index[:, np.asarray(edge_type) == r], np.int64)
        e1 = np.unique(e, axis=1) if e.shape[1] else e
        es.append(np.concatenate([e1, e1[::-1]], axis=1))
        ts.append(np.full(2 * e1.shape[1], r, np.int64))
    return np.concatenate(es, axis=1), np.concatenate(ts)


@dataclass
class Prepared:
    F: object
    J: object
    g: object                     # GraphInputs (edges = relational undirected)
    T: dict | None                # alpha-view tensors
    S: dict | None                # beta-view tensors
    lab: torch.Tensor             # dataset rows of the labelled users: train, val, test
    pos: dict                     # split -> positions in lab
    y_lab: torch.Tensor
    data_files: list
    info: dict = field(default_factory=dict)


def prepare(a) -> Prepared:
    from ...features import load_features
    from ...gnn import to_device

    t0 = time.time()
    F = load_features(a.dataset, a.features, require_text=not (a.build_only or a.measure_epochs))
    rels = F.relations_by_type()
    splits = {s: np.asarray(F.splits[s]) for s in ("train", "val", "test")}
    lab = np.concatenate([splits["train"], splits["val"], splits["test"]]).astype(np.int64)
    info: dict = {}
    data_files = list(F.files)
    T_np = None
    if not a.no_alpha:
        WT = build_whole_tree(a.dataset, rels, a.k, smooth=not a.tree_nosmooth, rebuild=a.rebuild_artifacts)
        T_np = WT
        info["tree"] = {k: v for k, v in WT.meta.items() if k not in ("stats",)}
        info["tree"]["rounds"] = [r["choice"] for r in WT.meta["stats"]["rounds"]]
        data_files.append(method_dir(a.dataset) / f"tree_k{a.k}_v2_{'nosmooth' if a.tree_nosmooth else 'smooth'}" / "meta.json")
    S_np = None
    if not a.no_beta:
        S_np = build_subgraph_trees(a.dataset, np.asarray(F.edge_index), lab, F.n, a.k, seed=a.subgraph_seed,
                                    mode=a.subgraph_mode, in_cap=a.in_cap, rebuild=a.rebuild_artifacts)
        info["subgraphs"] = S_np.meta
    info["artifact_seconds"] = round(time.time() - t0, 2)
    if a.build_only:
        return Prepared(F, None, None, None, None, None, {}, None, data_files, info)

    extscore_on = bool(a.with_extscore or a.extscore == "p")
    J = prepare_extscore(a, a.dataset, splits, labels=F.label, neighbours=tree_graph(a.dataset, rels) if extscore_on else None)
    dev = "cuda" if (a.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    if dev == "cuda" and a.min_free_vram_gb > 0:
        free, total = torch.cuda.mem_get_info()
        info["vram_free_at_start_GB"] = round(free / 1e9, 2)
        if free < a.min_free_vram_gb * 1e9:
            # the GPU is shared: under WSL a full card does not raise OOM but falls back to system
            # memory and runs orders of magnitude slower, so refuse to start instead
            import sys
            print(f"[sebot] only {free / 1e9:.2f} GB of {total / 1e9:.1f} GB VRAM free (< --min-free-vram-gb "
                  f"{a.min_free_vram_gb}); exiting with code 75", file=sys.stderr, flush=True)
            raise SystemExit(75)
    inj = {"num": "num", "late": "late", "gamma": "branch"}[a.extscore_inject]
    g = to_device(F, dev, J=J, extscore_inject=inj, edges=False, placeholder_text=bool(a.measure_epochs))
    ei, et = relational_undirected(np.asarray(F.edge_index), np.asarray(F.edge_type), F.num_relations)
    g = g.with_edges(torch.from_numpy(ei).to(dev), torch.from_numpy(et).to(dev))
    info["gamma_edges"] = int(ei.shape[1])
    T = None
    if T_np is not None:
        T = {"e0": g.edge_index,
             "e1": torch.from_numpy(np.asarray(T_np.edges[1], np.int64)).to(dev),
             "e2": torch.from_numpy(np.asarray(T_np.edges[2], np.int64)).to(dev),
             "p1": torch.from_numpy(np.asarray(T_np.parent[1], np.int64)).to(dev),
             "p2": torch.from_numpy(np.asarray(T_np.parent[2], np.int64)).to(dev),
             "n1": int(T_np.node_size[1]), "n2": int(T_np.node_size[2])}
    S = None
    if S_np is not None:
        uniq, gather = np.unique(S_np.nodelist, return_inverse=True)
        L = 4
        S = {"uniq": torch.from_numpy(uniq).to(dev), "gather": torch.from_numpy(gather.astype(np.int64)).to(dev),
             "edges": [torch.from_numpy(S_np.edges[c]).to(dev) for c in range(min(3, a.k))],
             "parent": [None] + [torch.from_numpy(S_np.parent[d]).to(dev) for d in range(1, min(L, a.k + 1))],
             "size": [len(S_np.batch[d]) for d in range(a.k + 1)],
             "batch": [torch.from_numpy(S_np.batch[d]).to(dev) for d in range(min(L, a.k + 1))],
             "n_graphs": S_np.n_graphs}
        S["count"] = [torch.bincount(b, minlength=S["n_graphs"]).float().clamp_(min=1) for b in S["batch"]]
        data_files.append(method_dir(a.dataset) / f"subgraphs_hop1_top10_{a.subgraph_mode}_cap{a.in_cap}_seed{a.subgraph_seed}_k{a.k}.npz")
    lab_t = torch.from_numpy(lab).to(dev)
    n = [len(splits[s]) for s in ("train", "val", "test")]
    pos = {"train": np.arange(0, n[0]), "val": np.arange(n[0], n[0] + n[1]), "test": np.arange(n[0] + n[1], sum(n))}
    y_lab = torch.from_numpy(np.asarray(F.label)[lab].astype(np.int64)).to(dev)
    info["load_seconds"] = round(time.time() - t0, 2)
    info["device"] = dev
    print(f"[sebot] {a.dataset}: N={F.n} labelled={len(lab)} gamma edges={ei.shape[1]} "
          f"tree={None if T_np is None else T_np.node_size} "
          f"subgraphs={None if S_np is None else {k: S_np.meta[k] for k in ('lone', 'nodes_level', 'max_graph_nodes')}} "
          f"extscore={'on:' + a.extscore_inject if J.enabled else 'off'} ({info['load_seconds']} s)", flush=True)
    return Prepared(F, J, g, T, S, lab_t, pos, y_lab, data_files, info)


def build_model(a, P: Prepared) -> SEBot:
    F, J = P.F, P.J
    num_dim = F.num_prop.shape[1] + (J.width if a.extscore_inject == "num" else 0)
    return SEBot(num_dim, F.cat_prop.shape[1], k=a.k, hidden=a.hidden, input_dim=a.input_dim, proj_dim=a.proj_dim,
                 num_relations=F.num_relations, dropout=a.dropout, conv_dropout=a.dropout, pool_dropout=a.dropout,
                 pe=a.pe, eval_noise=not a.no_eval_noise, rcm=not a.no_rcm, backbone_conv=a.backbone,
                 use_u=not a.no_alpha, use_g=not a.no_beta,
                 extscore_late=J.width if a.extscore_inject == "late" else 0,
                 gamma_extra=J.width if a.extscore_inject == "gamma" else 0, tau_g=a.tau_g,
                 checkpoint_encoders=a.checkpoint_encoders and P.lab.device.type == "cuda")


def _late(a, P):
    return P.g.extra[P.lab] if (a.extscore_inject == "late" and P.J.enabled) else None


def train_epochs(a, P: Prepared, model: SEBot, epochs: int, run=None, log_every: int = 0):
    """Full-batch loop. Returns per-epoch history and the eval probabilities (all labelled rows,
    lab order) of the epoch with the minimum validation loss (first on ties)."""
    dev = P.lab.device
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.t_max, eta_min=0)
    tr = torch.as_tensor(P.pos["train"], device=dev)
    va = torch.as_tensor(P.pos["val"], device=dev)
    late = _late(a, P)
    y = P.y_lab
    best = {"val_loss": float("inf"), "epoch": -1, "prob": None}
    hist, t_train, bad = [], 0.0, 0
    for ep in range(epochs):
        model.train()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        h_u, h_g, h_c = model.views(P.g, P.T, P.S, P.lab)
        logits = model.logits(h_u, h_g, h_c, late)
        ce = Fn.cross_entropy(logits[tr], y[tr])
        l1, l2 = model.contrastive(h_u, h_g, h_c, a.tau, a.nce_block)
        loss = ce + a.lambda1 * l1 + a.lambda2 * l2
        loss.backward()
        opt.step()
        sched.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_train += time.perf_counter() - t
        model.eval()
        with torch.no_grad():
            h_u, h_g, h_c = model.views(P.g, P.T, P.S, P.lab)
            lo = model.logits(h_u, h_g, h_c, late).float()
            vl = float(Fn.cross_entropy(lo[va], y[va]))
            prob = torch.softmax(lo, 1)[:, 1].cpu().numpy().astype(np.float32)
        yv = y[va].cpu().numpy()
        mv = compute_metrics(yv, prob[P.pos["val"]])
        rec = {"loss": float(loss.detach()), "ce": float(ce.detach()), "nce_u": float(l1.detach()),
               "nce_g": float(l2.detach()), "lr": sched.get_last_lr()[0], "val_loss": vl,
               "val_acc": mv["accuracy"], "val_f1": mv["f1"]}
        hist.append({"epoch": ep, **rec})
        if run is not None:
            run.log_epoch(ep, **rec)
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d} " + " ".join(f"{k}={v:.4f}" for k, v in rec.items()), flush=True)
        if vl < best["val_loss"]:
            best = {"val_loss": vl, "epoch": ep, "prob": prob}
            bad = 0
        else:
            bad += 1
            if a.patience >= 0 and bad > a.patience:
                break
    return hist, best, t_train


def run_seed(a, P: Prepared, seed: int, run) -> dict:
    dev = P.lab.device
    model = build_model(a, P).to(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    hist, best, t_train = train_epochs(a, P, model, a.epochs, run=run, log_every=a.log_every)
    prob = best["prob"]
    y = P.y_lab.cpu().numpy()
    lab = P.lab.cpu().numpy()
    metrics = {s: compute_metrics(y[P.pos[s]], prob[P.pos[s]]) for s in ("train", "val", "test")}
    for s in ("val", "test"):
        run.add_predictions(s, lab[P.pos[s]], prob[P.pos[s]], y[P.pos[s]])
    ep_run = len(hist)
    return run.finish(
        metrics, selection={"select": "min_val_loss", "epoch": best["epoch"], "val_loss": best["val_loss"]},
        epochs_run=ep_run, extscore=P.J.report(), n_params=n_params(model), with_extscore=a.with_extscore,
        timing={"load_seconds": P.info.get("load_seconds"), "artifact_seconds": P.info.get("artifact_seconds"),
                "train_seconds": round(t_train, 2), "seconds_per_epoch": round(t_train / max(ep_run, 1), 4)},
        gpu={"run_max_allocated_bytes": int(torch.cuda.max_memory_allocated()) if dev.type == "cuda" else 0},
        artifacts={k: v for k, v in P.info.items() if k in ("tree", "subgraphs", "gamma_edges")},
        features={"dir": str(P.F.dir), "input_key": P.F.meta["input_key"], "variant": P.F.variant})


def measure(a, P: Prepared) -> dict:
    from ...harness import set_seed
    from ...mem import memory_report
    set_seed(a.seeds[0])
    dev = P.lab.device
    model = build_model(a, P).to(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t = time.time()
    hist, best, t_train = train_epochs(a, P, model, a.measure_epochs)
    rep = {"dataset": a.dataset, "epochs": len(hist), "train_seconds_per_epoch": round(t_train / len(hist), 4),
           "wall_seconds_incl_eval": round(time.time() - t, 2),
           "projected_seconds_full_run": round((time.time() - t) / len(hist) * a.epochs, 1),
           "cuda_max_allocated_GB": round(torch.cuda.max_memory_allocated() / 1e9, 3) if dev.type == "cuda" else 0,
           "n_params": n_params(model), "info": P.info, "memory": memory_report()}
    print("MEASURE " + json.dumps(rep, default=str), flush=True)
    return rep
