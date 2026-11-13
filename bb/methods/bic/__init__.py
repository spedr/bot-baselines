"""BIC (Lei et al., ACL 2023): text-graph interaction + semantic consistency. Spec: docs/methods/bic.md.

Reference semantics: ``variant: v2`` (TwiBot20v2 ``InteractModel``), with the published bugs fixed by
default and reproducible behind off-by-default flags:

- graph local-index bug (spec pitfall 1): ``--reproduce-index-bug``
- best state stored by reference, so the "best" model is the last one (pitfall 2): ``--select last``
- ``model.train()`` never restored after the first validation, so epochs 1.. train in eval mode
  (dropout off): ``--reproduce-eval-mode-bug``

Inputs: graph side = shared ``botrgcn`` feature scheme (``bb.features``; num/cat/des/tweet-mean and the
merged ``edge_index``); text side = [roberta-base des, first ``cap`` per-tweet roberta-base embeddings
in release order] read per mini-batch from ``emb/roberta-base/tweets_seq_<sel>.f16.npy``
(``data.SeqReader``: one preadv per user into a pinned fp16 buffer; nothing [N, cap, 768] is built).

    tools/memrun --light -- uv run python -m bb.methods.bic --dataset cresci15
    tools/memrun -m 4G   -- uv run python -m bb.methods.bic --dataset twibot20
    tools/memrun -m 5G   -- uv run python -m bb.methods.bic --dataset twibot22 --measure-batches 300
    ... --compile auto|on|off (auto: on for TwiBot-22 + gcn on CUDA)  --io-backend native|python
    ... --io-threads 8  --prefetch-batches 0 (auto)               # speed only, same results
    ... --with-extscore synthetic [--extscore-at graph_num|head|graph_num+head]
    ... --ablation wo_graph|wo_text|wo_interaction|wo_consistency|wo_graph_module|wo_text_module
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from ...harness import BestCheckpoint, Run, aggregate, compute_metrics, set_seed
from ...extscore import add_extscore_args, prepare_extscore

SRC = [Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("data.py")]

# per-tweet sequence table per dataset (docs/data_status.md) and the cap it allows
SEQ_DEFAULT = {"twibot20": ("all", 200), "cresci15": ("first20", 20), "twibot22": ("first20", 20)}
SEL_CAP = {"all": 200, "first20": 20, "first200": 200}
LR_DEFAULT = {"twibot20": 1e-4, "cresci15": 1e-3, "twibot22": 1e-4}   # Cresci15v2/train.py: 1e-3
MAXNB_DEFAULT = {"twibot20": 0, "cresci15": 0, "twibot22": 256}       # spec 7.2
EXTSCORE_AT = ("graph_num", "head", "graph_num+head")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="BIC (variant v2)")
    ap.add_argument("--dataset", choices=["twibot20", "twibot22", "cresci15"], default="cresci15")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--features", default="botrgcn", help="graph-side botrgcn feature variant")
    ap.add_argument("--seq-selection", default=None, help="tweets_seq_<sel> table (default per dataset)")
    ap.add_argument("--tweets-cap", type=int, default=None, help="tweets per user (paper 200; default = table cap)")
    ap.add_argument("--tok0-encoder", default="roberta-base", help="encoder of the description token")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30, help="epoch cap (official: unbounded; paper: 30)")
    ap.add_argument("--patience", type=int, default=8, help="stop after this many epochs without a val gain")
    ap.add_argument("--select", choices=["best_val", "last"], default="best_val",
                    help="last = emulate the by-reference checkpoint bug (evaluated model = final epoch)")
    ap.add_argument("--select-metric", default="accuracy")
    ap.add_argument("--dropout", type=float, default=0.1, help="text-layer dropout (the only active one in v2)")
    ap.add_argument("--interaction", choices=["code", "paper"], default="code")
    ap.add_argument("--ablation", default="none",
                    choices=["none", "wo_graph", "wo_text", "wo_interaction", "wo_consistency", "wo_graph_module",
                             "wo_text_module"],
                    help="wo_graph/wo_text = the paper's ablation (drop g/h from the readout, Eq. 3); "
                         "*_module = remove the whole module and the interaction")
    ap.add_argument("--pool-k", type=int, default=16)
    ap.add_argument("--grad-clip", type=float, default=0.0, help="max grad norm (0 = off, as the official code)")
    ap.add_argument("--dedup-neighbors", action="store_true")
    ap.add_argument("--max-neighbors", type=int, default=None, help="cap of the attention list (0 = none)")
    ap.add_argument("--reproduce-index-bug", action="store_true", help="forensic: TwiBot20v2 GCN index bug")
    ap.add_argument("--reproduce-eval-mode-bug", action="store_true",
                    help="TwiBot20v2/train.py never calls model.train() after evaluate(): train epochs 1.. "
                         "run in eval mode (no dropout, incl. attention dropout on the consistency maps)")
    ap.add_argument("--gnn", choices=["gcn", "rgcn"], default="gcn",
                    help="gcn = TwiBot20v2 code (reference); rgcn = the paper's text (diagnostic)")
    ap.add_argument("--extscore-at", choices=EXTSCORE_AT, default="graph_num")
    ap.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none")
    ap.add_argument("--vram-cap-gb", type=float, default=None,
                    help="torch.cuda.set_per_process_memory_fraction cap (shared GPU; default 3, TwiBot-22 6; 0 = none)")
    ap.add_argument("--drop-cache", choices=["auto", "on", "off"], default="auto",
                    help="FADV_DONTNEED after each tweet read (auto: when the labelled users' rows exceed 1.5 GB)")
    ap.add_argument("--checkpoint-text", action="store_true",
                    help="activation checkpointing of the text layers (same results up to float noise; less VRAM)")
    ap.add_argument("--io-threads", type=int, default=8, help="parallel pread threads per batch")
    ap.add_argument("--prefetch-batches", type=int, default=0,
                    help="pinned batch buffers in the prefetch ring (0 = auto: 3-8, about 64 MB in total)")
    ap.add_argument("--io-backend", choices=["native", "python"], default="native",
                    help="per-user read loop: GIL-free numba kernel (default) or the os.preadv loop (same bytes)")
    ap.add_argument("--compile", choices=["auto", "on", "off"], default="auto",
                    help="torch.compile the model forward/backward and the RAdam step (same function, dropout "
                         "masks kept equal with inductor fallback_random; float noise only). auto = on for "
                         "TwiBot-22 with --gnn gcn on CUDA (CPU-dispatch-bound there), off elsewhere")
    ap.add_argument("--tf32", action="store_true", help="allow TF32 matmuls (faster; not used for reported numbers)")
    ap.add_argument("--predict-train", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--train-subsample", type=int, default=0, help="stratified train subsample (TwiBot-22 fallback)")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--log-every", type=int, default=0, help="print every K train batches")
    ap.add_argument("--measure-batches", type=int, default=0,
                    help="time K train batches + one val pass for one seed, print a projection, exit")
    add_extscore_args(ap)
    a = ap.parse_args(argv)
    sel, cap = SEQ_DEFAULT[a.dataset]
    a.seq_selection = a.seq_selection or sel
    tcap = SEL_CAP.get(a.seq_selection, cap)
    a.tweets_cap = tcap if a.tweets_cap is None else a.tweets_cap
    if a.tweets_cap > tcap:
        raise SystemExit(f"tweets_seq_{a.seq_selection} holds at most {tcap} tweets per user")
    a.lr = LR_DEFAULT[a.dataset] if a.lr is None else a.lr
    a.max_neighbors = MAXNB_DEFAULT[a.dataset] if a.max_neighbors is None else a.max_neighbors
    a.vram_cap_gb = (6.0 if a.dataset == "twibot22" else 3.0) if a.vram_cap_gb is None else a.vram_cap_gb
    if a.compile == "on" and a.gnn == "rgcn":
        # PyG RGCNConv graph-breaks on int(segment_count.max()) and its compiled path does not match
        # eager (loss differs by 3e-4 on the first TwiBot-22 batch; efficiency review)
        raise SystemExit("--compile on is not supported with --gnn rgcn")
    return a


def use_compile(a, dev) -> bool:
    if a.compile == "off" or str(dev) != "cuda":
        return False
    if a.compile == "on":
        return True
    return a.dataset == "twibot22" and a.gnn == "gcn" and not a.deterministic and not a.checkpoint_text


class Step:
    """Model forward and optimizer step, eager or ``torch.compile``d. The compiled forward shares the
    module's parameters (``model`` stays the plain module: state_dict, checkpoints, train/eval mode)."""

    def __init__(self, model, opt, compiled: bool, scaler=None):
        self.model, self.opt, self.compiled = model, opt, compiled
        self.fwd = model
        self.opt_step = opt.step if opt is not None else None
        if compiled:
            import torch
            import torch._inductor.config as ic
            ic.fallback_random = True        # eager RNG ops: dropout masks identical to the eager model
            ic.compile_threads = min(ic.compile_threads, 4)   # bounded compile workers (host memory)
            # one Step per seed: drop the previous seed's graphs so the recompile limit (8 per code
            # object) is never reached in a multi-seed process (inductor's on-disk cache keeps it fast)
            torch._dynamo.reset()
            self.fwd = torch.compile(model, dynamic=True)
            if opt is not None and scaler is None:
                @torch.compile(fullgraph=False)
                def _step():
                    opt.step()
                self.opt_step = _step


def variant_name(a) -> str:
    v = ["v2", a.features, f"seq{a.seq_selection}c{a.tweets_cap}"]
    if a.ablation != "none":
        v.append(a.ablation)
    if a.interaction != "code":
        v.append("interpaper")
    if a.reproduce_index_bug:
        v.append("indexbug")
    if a.reproduce_eval_mode_bug:
        v.append("evalmodebug")
    if a.gnn != "gcn":
        v.append(a.gnn)
    if a.select != "best_val":
        v.append("selectlast")
    if a.dedup_neighbors:
        v.append("dedup")
    if a.max_neighbors != MAXNB_DEFAULT[a.dataset]:
        v.append(f"nb{a.max_neighbors}")
    if a.amp != "none":
        v.append(a.amp)
    if a.tf32:
        v.append("tf32")
    if a.train_subsample:
        v.append(f"sub{a.train_subsample}")
    if a.grad_clip:
        v.append(f"clip{a.grad_clip:g}")
    if a.lr != LR_DEFAULT[a.dataset]:
        v.append(f"lr{a.lr:g}")
    # every other training knob that changes results must show in the variant too (review round 1:
    # a --dropout 0 run used to share the default variant and would be aggregated with it)
    for name, default, tag in (("dropout", 0.1, "do"), ("wd", 1e-5, "wd"), ("batch_size", 64, "bs"),
                               ("epochs", 30, "ep"), ("patience", 8, "pat"), ("pool_k", 16, "k")):
        if getattr(a, name) != default:
            v.append(f"{tag}{getattr(a, name):g}")
    if a.select_metric != "accuracy":
        v.append(f"sel{a.select_metric}")
    if (a.with_extscore or a.extscore == "p") and a.extscore_at != "graph_num":
        v.append("extscore" + a.extscore_at.replace("+", "_"))
    return ".".join(v)


# ---------------------------------------------------------------------------------------------- setup


class Inputs:
    """Everything a run needs on the device and the host (shared across seeds)."""

    def __init__(self, a, F, J, dev):
        import torch

        from ...data import load_embedding, load_tweet_seq
        from ...gnn import to_device
        from ...harness.gpu import memmap_to_gpu
        from .data import Closure, GraphLists, SeqReader

        self.F, self.J, self.dev = F, J, torch.device(dev)
        extscore_num = J.enabled and a.extscore_at in ("graph_num", "graph_num+head")
        self.extscore_head = J.enabled and a.extscore_at in ("head", "graph_num+head")
        self.g = to_device(F, dev, J=J if extscore_num else None, extscore_inject="num", edges=False)
        self.late = None
        if self.extscore_head:
            self.late = torch.from_numpy(J.append(np.zeros((F.n, 0), np.float32), kind="neural")).to(dev)
        self.labelled = np.sort(np.concatenate([F.splits[s] for s in ("train", "val", "test")]))
        # description token: same table as the graph-side des when the recipes agree
        d0 = load_embedding(F.dataset, a.tok0_encoder, "des")
        same = F.des is not None and os.path.realpath(F.dir / "des.f16.npy") == os.path.realpath(d0.filename)
        self.tok0_file = Path(d0.filename)
        if same:
            self.tok0, self.tok0_pos = self.g.des, None
        else:
            self.tok0 = memmap_to_gpu(d0, rows=self.labelled, dtype=self.g.des.dtype, device=str(dev))
            pos = np.full(F.n, -1, np.int64)
            pos[self.labelled] = np.arange(len(self.labelled))
            self.tok0_pos = torch.from_numpy(pos).to(dev)
        del d0
        seq, off, _rows = load_tweet_seq(F.dataset, "roberta-base", a.seq_selection)
        self.seq_file = Path(seq.filename)
        offsets = np.asarray(off, dtype=np.int64)
        del seq, off, _rows
        cnt = np.minimum(np.diff(offsets), a.tweets_cap)
        lab_bytes = int(cnt[self.labelled].sum()) * 768 * 2
        drop = {"on": True, "off": False}.get(a.drop_cache, lab_bytes > 1.5e9)
        self.reader = SeqReader(self.seq_file, offsets, a.tweets_cap, drop_cache=drop, io_threads=a.io_threads,
                                backend=a.io_backend)
        lab = self.labelled
        n_av = np.diff(offsets)[lab]
        self.seq_stats = {"table": self.seq_file.name, "cap": a.tweets_cap, "labelled_users": int(len(lab)),
                          "users_no_tweets": int((n_av == 0).sum()), "users_over_cap": int((n_av > a.tweets_cap).sum()),
                          "mean_tweets_used": float(cnt[lab].mean()), "labelled_bytes": lab_bytes,
                          "drop_cache": drop}
        t = time.time()
        et = np.asarray(F.edge_type) if a.gnn == "rgcn" else None
        self.num_relations = int(et.max()) + 1 if et is not None and len(et) else 1
        self.lists = GraphLists.from_edge_index(np.asarray(F.edge_index), F.n, dedup_neighbors=a.dedup_neighbors,
                                                edge_type=et)
        del et
        self.graph_stats = {**self.lists.degree_stats(lab), "build_seconds": round(time.time() - t, 2)}
        self.closure = Closure(self.lists, max_neighbors=a.max_neighbors, index_bug=a.reproduce_index_bug,
                               rgcn=a.gnn == "rgcn")

    @property
    def files(self) -> list[Path]:
        return [*self.F.files, self.seq_file, self.tok0_file]

    def batch(self, users, text, bg):
        """Device tensors of one prepared batch -> model kwargs."""
        import torch
        dev = self.dev
        # one pinned upload for every index array, non-blocking: no stream drain per batch (each
        # pageable copy waits for all queued GPU work, which costs a time slice on a shared GPU)
        parts = [np.asarray(users, np.int64), bg.n_id, bg.edge_index[0], bg.edge_index[1], bg.att_seg, bg.att_node]
        if bg.edge_type is not None:
            parts.append(bg.edge_type)
        host = torch.from_numpy(np.concatenate(parts))
        if dev.type == "cuda":
            host = host.pin_memory()
        flat = host.to(dev, non_blocking=True)
        u, n_id, e0, e1, a_seg, a_node, *et = torch.split(flat, [len(x) for x in parts])
        g = self.g
        des0 = self.tok0[u] if self.tok0_pos is None else self.tok0[self.tok0_pos[u]]
        kw = dict(tweets=text, des0=des0, gdes=g.des[n_id], gtweet=g.tweet[n_id], gnum=g.num[n_id],
                  gcat=g.cat[n_id], edge_index=torch.stack([e0, e1]), att_seg=a_seg, att_node=a_node,
                  n_targets=len(users), edge_type=et[0] if et else None,
                  late=self.late[u] if self.late is not None else None)
        return kw, g.y[u]


def build_model(a, inp: Inputs):
    from .model import BIC
    return BIC(num_dim=int(inp.g.num.shape[1]), cat_dim=int(inp.g.cat.shape[1]), dropout=a.dropout,
               k=a.pool_k, interaction=a.interaction, ablation=a.ablation,
               extscore_head=2 if inp.extscore_head else 0, checkpoint_text=a.checkpoint_text, gnn=a.gnn,
               num_relations=inp.num_relations)


def _autocast(a, dev):
    import contextlib

    import torch
    if a.amp == "none" or torch.device(dev).type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if a.amp == "bf16" else torch.float16)


def _batches(idx: np.ndarray, bs: int) -> list[np.ndarray]:
    return [idx[s:s + bs] for s in range(0, len(idx), bs)]


def predict(model, inp: Inputs, pf, idx: np.ndarray, a, fwd=None) -> np.ndarray:
    """P(bot) for ``idx`` (in order). Neighbour caps use fixed per-batch seeds (deterministic).
    ``fwd``: callable sharing ``model``'s parameters (the compiled forward), default ``model``."""
    import torch
    fwd = model if fwd is None else fwd
    was = model.training
    model.eval()
    out = torch.empty(len(idx), dtype=torch.float32, device=inp.dev)
    bl = _batches(np.asarray(idx, np.int64), a.eval_batch_size)
    pos = 0
    with torch.no_grad(), _autocast(a, inp.dev):
        for users, text, bg in pf.iterate(bl, [10_000_019 + i for i in range(len(bl))]):
            kw, _ = inp.batch(users, text, bg)
            out[pos:pos + len(users)] = torch.softmax(fwd(**kw).float(), 1)[:, 1]
            pos += len(users)
    model.train(was)
    return out.cpu().numpy()


def train_epoch(model, opt, inp: Inputs, pf, train_idx, a, seed: int, ep: int, scaler=None, max_batches: int = 0,
                timer: list | None = None, step: Step | None = None) -> float:
    import torch
    from torch import nn
    fwd = model if step is None else step.fwd
    opt_step = opt.step if step is None else step.opt_step
    # official TwiBot20v2: evaluate() sets eval mode and nothing switches back, so every epoch after
    # the first trains with dropout off (--reproduce-eval-mode-bug)
    model.train(not (a.reproduce_eval_mode_bug and ep > 0))
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng([seed, ep])
    bl = _batches(rng.permutation(train_idx), a.batch_size)
    if max_batches:
        bl = bl[:max_batches]
    seeds = [int(x) for x in rng.integers(0, 2**31 - 1, len(bl))]
    tot = torch.zeros((), device=inp.dev)
    n = 0
    for i, (users, text, bg) in enumerate(pf.iterate(bl, seeds)):
        if timer is not None and i == min(20, len(bl) - 1):
            if inp.dev.type == "cuda":
                torch.cuda.synchronize()
            timer.append((time.perf_counter(), i))
        kw, yb = inp.batch(users, text, bg)
        opt.zero_grad(set_to_none=True)
        with _autocast(a, inp.dev):
            logits = fwd(**kw)
        loss = loss_fn(logits.float(), yb)
        if scaler is not None:
            scaler.scale(loss).backward()
            if a.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if a.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt_step()
        tot += loss.detach() * len(users)
        n += len(users)
        if a.log_every and (i + 1) % a.log_every == 0:
            print(f"    batch {i + 1}/{len(bl)} loss {float(tot) / n:.4f}", flush=True)
    if timer is not None:
        if inp.dev.type == "cuda":
            torch.cuda.synchronize()
        timer.append((time.perf_counter(), len(bl)))
    return float(tot) / max(n, 1)


def _subsample(train: np.ndarray, y: np.ndarray, k: int, seed: int) -> np.ndarray:
    if not k or k >= len(train):
        return train
    rng = np.random.default_rng(seed)
    out = []
    for c in (0, 1):
        t = train[y[train] == c]
        out.append(rng.choice(t, size=int(round(k * len(t) / len(train))), replace=False))
    return np.sort(np.concatenate(out))


def fit(model, inp: Inputs, pf, splits: dict, a, seed: int, run=None) -> dict:
    """RAdam, CE on train targets, early stopping on val ``select_metric`` (stop after ``patience``
    epochs without improvement), deep-copied best state (or the last state with ``--select last``).
    Also returns ``step`` (the forward used, for the final predictions) and ``val_probs``, the
    validation predictions of the selected state (computed when it was selected; the final val
    pass is not repeated)."""
    import torch
    y = inp.F.label
    tr = _subsample(np.asarray(splits["train"]), y, a.train_subsample, seed)
    va = np.asarray(splits["val"])
    opt = torch.optim.RAdam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    scaler = torch.amp.GradScaler("cuda") if (a.amp == "fp16" and inp.dev.type == "cuda") else None
    ck = BestCheckpoint(select="best_val" if a.select == "best_val" else "last_epoch")
    step = Step(model, opt, use_compile(a, inp.dev.type), scaler)
    best, bad, hist, t_train, ep, diverged, val_probs = None, 0, [], 0.0, -1, None, None
    for ep in range(a.epochs):
        t = time.perf_counter()
        pf.stats.update(read_s=0.0, closure_s=0.0, wait_s=0.0, batches=0)
        loss = train_epoch(model, opt, inp, pf, tr, a, seed, ep, scaler, step=step)
        io = dict(pf.stats)
        if inp.dev.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t
        t_train += dt
        te = time.perf_counter()
        pv = predict(model, inp, pf, va, a, fwd=step.fwd)
        mv = compute_metrics(y[va], pv)
        v = mv[a.select_metric]
        if ck.update(model, v, ep) or ck.select == "last_epoch":
            val_probs = pv
        if best is None or v > best:  # NaN never counts as an improvement
            best, bad = v, 0
        else:
            bad += 1
        rec = {"train_loss": loss, "val_acc": mv["accuracy"], "val_f1": mv["f1"], "val_auc": mv["roc_auc"],
               "train_seconds": round(dt, 2), "val_seconds": round(time.perf_counter() - te, 2),
               "prefetch_wait_seconds": round(io["wait_s"], 2), "read_seconds": round(io["read_s"], 2)}
        hist.append({"epoch": ep, **rec})
        if run is not None:
            run.log_epoch(ep, **rec)
        print(f"  ep {ep:3d} loss {loss:.4f} val acc {mv['accuracy']:.4f} f1 {mv['f1']:.4f} "
              f"({dt:.1f}s + {rec['val_seconds']:.1f}s)", flush=True)
        if not math.isfinite(loss):
            # parameters are non-finite from here on: every later epoch would predict NaN, so under
            # best-val selection stopping now is exactly equivalent (the best state is restored)
            diverged = ep
            print(f"  training diverged (non-finite loss) at epoch {ep}; stopping", flush=True)
            break
        if a.patience > 0 and bad >= a.patience:
            break
    ck.restore(model)
    n_ep = ep + 1
    return {"epochs_run": n_ep, "diverged_at_epoch": diverged,
            "selection": {**ck.info(), "metric": a.select_metric, "patience": a.patience},
            "train_seconds": round(t_train, 2), "seconds_per_epoch": round(t_train / max(n_ep, 1), 2),
            "history": hist, "n_train": int(len(tr)), "step": step, "val_probs": val_probs}


# ---------------------------------------------------------------------------------------------- main


def main(argv=None) -> list[dict]:
    import torch

    from ...features import load_features
    from ...mem import memory_report
    from .data import Prefetcher
    from .model import n_params

    a = parse_args(argv)
    # the text layers allocate many differently-sized activation blocks; expandable segments keep
    # the reserved pool close to the allocated peak on the shared GPU (read at the first CUDA call)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    t0 = time.time()
    F = load_features(a.dataset, a.features)
    splits = {s: F.splits[s] for s in ("train", "val", "test")}
    y = F.label
    extscore_on = bool(a.with_extscore or a.extscore == "p")
    neigh = None
    if extscore_on:
        from ...data import load_undirected
        neigh = load_undirected(a.dataset, F.relations_by_type())
    J = prepare_extscore(a, a.dataset, splits, labels=y, neighbours=neigh)
    dev = "cuda" if (a.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = bool(a.tf32) and not a.deterministic
    if dev == "cuda" and a.vram_cap_gb > 0:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_cap_gb * 1e9 / total))
    inp = Inputs(a, F, J, dev)
    pf = Prefetcher(inp.reader, inp.closure, max(a.batch_size, a.eval_batch_size), dev,
                    n_buf=a.prefetch_batches or None)
    load_s = round(time.time() - t0, 2)
    print(f"bic {a.dataset}/{F.variant}: N={F.n} E={F.edge_index.shape[1]} seq={inp.seq_stats} "
          f"graph={inp.graph_stats} inputs {inp.g.nbytes() / 1e9:.2f} GB on {dev} (load {load_s} s) "
          f"extscore={'on:' + a.extscore_at if J.enabled else 'off'}", flush=True)
    try:
        if a.measure_batches:
            return [measure(a, inp, pf, splits, dev, load_s)]
        cfg = {**vars(a), "variant": variant_name(a), "features_variant": F.variant}
        results = []
        for seed in a.seeds:
            set_seed(seed, deterministic=a.deterministic)
            t_seed = time.time()
            model = build_model(a, inp).to(dev)
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            with Run("bic", a.dataset, config=cfg, seed=seed, data_files=inp.files, source_files=SRC) as run:
                fr = fit(model, inp, pf, splits, a, seed, run=run)
                pred_train = a.predict_train == "on" or (a.predict_train == "auto" and len(splits["train"]) <= 50_000)
                names = ("train", "val", "test") if pred_train else ("val", "test")
                metrics, probs = {}, {}
                for s in names:
                    if s == "val" and fr["val_probs"] is not None:
                        probs[s] = fr["val_probs"]       # the selected state's val pass (same model)
                    else:
                        probs[s] = predict(model, inp, pf, splits[s], a, fwd=fr["step"].fwd)
                    metrics[s] = compute_metrics(y[splits[s]], probs[s])
                for s in ("val", "test"):
                    run.add_predictions(s, splits[s], probs[s], y[splits[s]])
                out = run.finish(
                    metrics, selection=fr["selection"], epochs_run=fr["epochs_run"], extscore=J.report(),
                    diverged_at_epoch=fr["diverged_at_epoch"],
                    n_params=n_params(model), with_extscore=a.with_extscore, n_train=fr["n_train"],
                    timing={"load_seconds": load_s, "train_seconds": fr["train_seconds"],
                            "seconds_per_epoch": fr["seconds_per_epoch"], "compiled": fr["step"].compiled,
                            "wall_seconds": round(time.time() - t_seed, 1)},
                    gpu={"inputs_bytes": inp.g.nbytes(),
                         "run_max_allocated_bytes": int(torch.cuda.max_memory_allocated()) if dev == "cuda" else 0},
                    seq=inp.seq_stats, graph=inp.graph_stats,
                    features={"dir": str(F.dir), "input_key": F.meta["input_key"], "variant": F.variant})
            t = metrics["test"]
            print(f"[bic {a.dataset} seed {seed}] test acc {t['accuracy']:.4f} f1 {t['f1']:.4f} "
                  f"prec {t['precision']:.4f} rec {t['recall']:.4f} auc {t['roc_auc']:.4f} | "
                  f"{fr['seconds_per_epoch']:.1f} s/epoch, {fr['epochs_run']} epochs (best {fr['selection']['best_epoch']})",
                  flush=True)
            results.append(out)
            del model, fr          # fr holds the step (compiled closures over model and optimizer)
        agg = aggregate(results)
        print({k: (round(v["mean"], 4), round(v["std"], 4)) for k, v in agg.items()
               if k in ("accuracy", "f1", "precision", "recall", "macro_f1", "mcc", "roc_auc")})
        if J.enabled and J.baseline:
            b = J.baseline["test"]
            print(f"column-only baseline (val-tuned threshold {J.baseline['threshold']:.3f}): "
                  f"test acc {b['accuracy']:.4f} f1 {b['f1']:.4f}")
        print("MEMORY " + json.dumps(memory_report(), default=str), flush=True)
        return results
    finally:
        pf.close()
        inp.reader.close()


def measure(a, inp: Inputs, pf, splits, dev, load_s) -> dict:
    """``--measure-batches K``: time K train batches (after 20 warm-up batches) and one val pass."""
    import torch

    from ...mem import memory_report
    from .model import n_params

    set_seed(a.seeds[0])
    model = build_model(a, inp).to(dev)
    opt = torch.optim.RAdam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    scaler = torch.amp.GradScaler("cuda") if (a.amp == "fp16" and dev == "cuda") else None
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    step = Step(model, opt, use_compile(a, dev), scaler)
    timer: list = []
    k = a.measure_batches + 20
    rb0 = inp.reader.bytes_read
    pf.stats.update(read_s=0.0, closure_s=0.0, wait_s=0.0, batches=0)
    t_c = time.perf_counter()
    train_epoch(model, opt, inp, pf, np.asarray(splits["train"]), a, a.seeds[0], 0, scaler, max_batches=k, timer=timer,
                step=step)
    warm_s = timer[0][0] - t_c                       # the 20 warm-up batches (incl. compilation)
    (t1, i1), (t2, i2) = timer
    per_batch = (t2 - t1) / max(i2 - i1, 1)
    read_per_batch = (inp.reader.bytes_read - rb0) / k
    st = dict(pf.stats)
    prefetch = {f"{x}_ms_per_batch": round(1e3 * st[x] / max(st["batches"], 1), 2) for x in ("read_s", "closure_s", "wait_s")}
    nv = min(len(splits["val"]), a.eval_batch_size * 40)
    if step.compiled:                                # compile the no-grad graph outside the timing
        predict(model, inp, pf, np.asarray(splits["val"])[:2 * a.eval_batch_size], a, fwd=step.fwd)
    t = time.perf_counter()
    predict(model, inp, pf, np.asarray(splits["val"])[:nv], a, fwd=step.fwd)
    per_eval_user = (time.perf_counter() - t) / nv
    n_tr = len(_subsample(np.asarray(splits["train"]), inp.F.label, a.train_subsample, a.seeds[0]))
    tb = math.ceil(n_tr / a.batch_size)
    epoch_s = tb * per_batch + len(splits["val"]) * per_eval_user
    final_s = (len(splits["val"]) + len(splits["test"])) * per_eval_user
    rep = {"dataset": a.dataset, "variant": variant_name(a), "n_params": n_params(model),
           "measured_train_batches": i2 - i1, "seconds_per_train_batch": round(per_batch, 5),
           "train_batches_per_epoch": tb, "seconds_per_eval_user": round(per_eval_user, 7),
           "read_MB_per_train_batch": round(read_per_batch / 1e6, 2), "prefetch": prefetch,
           "projected_epoch_seconds": round(epoch_s, 1),
           "projected_seed_hours_15ep": round((15 * epoch_s + final_s) / 3600, 3),
           "projected_seed_hours_30ep": round((30 * epoch_s + final_s) / 3600, 3),
           "compiled": step.compiled, "warmup_seconds": round(warm_s, 1), "io_backend": a.io_backend,
           "io_threads": a.io_threads, "prefetch_batches": pf.n_buf, "load_seconds": load_s, "gpu_inputs_GB": round(inp.g.nbytes() / 1e9, 3),
           "cuda_max_allocated_GB": round(torch.cuda.max_memory_allocated() / 1e9, 3) if dev == "cuda" else 0,
           "cuda_max_reserved_GB": round(torch.cuda.max_memory_reserved() / 1e9, 3) if dev == "cuda" else 0,
           "memory": memory_report()}
    print("MEASURE " + json.dumps(rep, default=str), flush=True)
    return rep
