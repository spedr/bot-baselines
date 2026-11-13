"""RoBERTa + MLP (TwiBot-22 benchmark text baseline, Feng et al. 2022 App. B.1).

Spec: docs/methods/roberta_mlp.md. Frozen RoBERTa features of the description and of the tweets
(the shared ``botrgcn`` text tables, ``bb.data.botrgcn_text``; nothing is re-encoded) feed a small
MLP (``model.py``) trained with mini-batch Adam and unweighted cross-entropy for a fixed number of
epochs. The reported number is the LAST epoch; the val-best epoch (by ``--select-metric``) is
recorded as a secondary number (``--select best_val`` makes it primary). Bot = 1 everywhere; the
result carries bot-class and human-class precision/recall/F1 because the benchmark's TwiBot-20
"F1" for this method is human-class (spec 3.2, 6).

    tools/memrun --light -- uv run python -m bb.methods.roberta_mlp --dataset cresci15
    tools/memrun --light -- uv run python -m bb.methods.roberta_mlp --dataset twibot20
    tools/memrun -m 7G   -- uv run python -m bb.methods.roberta_mlp --dataset twibot22 --seeds 0
    ... --with-extscore synthetic                 # [z, m] concatenated to cat(p1, p2): Linear(770, 128)
    ... --extscore-inject late | branch [--extscore-branch-dim 384]

Inputs: placement is decided first: a split (test, val, train in that order) goes to the GPU
while the running total stays under ``--vram-budget-gb`` and under the free VRAM minus
``--vram-reserve-gb``; otherwise it stays on the host. The labelled rows of the [tweet | des]
tables are then read once with plain reads (``bb.io.read_rows``, 50k-row blocks) straight into
each split's fp16 destination, so a GPU-resident split never exists on the host in full
(TwiBot-22, all splits on the GPU: 3.07 GB VRAM, host peak about 1 GB). A host-resident split is
streamed in ``--chunk-rows`` blocks through two pinned staging buffers (the next block is gathered
by a worker thread while the current one trains). Batch boundaries do not depend on where a split
lives (``chunk_rows`` is a multiple of the batch size), so every placement trains the same model.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...harness import BestCheckpoint, Run, compute_metrics, set_seed
from ...harness.metrics import summarize
from ...extscore import add_extscore_args, prepare_extscore

SRC = [Path(__file__), Path(__file__).with_name("model.py")]
EMB = 768

PRESETS = {  # spec section 5 (the argparse block of Twibot-22/train.py is an RGT decoy: ignored)
    "twibot22": {"lr": 1e-4, "wd": 1e-5, "batch_size": 64, "epochs": 50, "dropout": 0.5},
    "twibot20": {"lr": 1e-5, "wd": 1e-6, "batch_size": 64, "epochs": 100, "dropout": 0.3},
    "cresci15": {"lr": 1e-4, "wd": 1e-5, "batch_size": 32, "epochs": 100, "dropout": 0.3},
}
REPORT_KEYS = ("accuracy", "precision", "recall", "f1", "precision_human", "recall_human", "f1_human",
               "macro_f1", "mcc", "roc_auc")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RoBERTa + MLP (TwiBot-22 benchmark text baseline)")
    ap.add_argument("--dataset", choices=sorted(PRESETS), default="cresci15")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--features", default="botrgcn",
                    help="text variant (bb.data.emb.BOTRGCN_DEFAULT); twibot20: botrgcn | botrgcn-b20")
    for k in ("lr", "wd", "dropout"):
        ap.add_argument(f"--{k}", type=float, default=None, help="default: dataset preset (spec 5)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--select", choices=["last_epoch", "best_val"], default="last_epoch",
                    help="primary number: last epoch (default, spec 5) or the val-best epoch")
    ap.add_argument("--select-metric", default="f1", help="val metric for the val-best epoch (bot-class f1)")
    ap.add_argument("--class-weight", default="none",
                    help="none (baseline, spec pitfall 7) | balanced | sqrt | w0,w1 (ablation only)")
    ap.add_argument("--extscore-inject", choices=["fuse", "late", "branch"], default="fuse",
                    help="fuse: [z, m] concatenated to cat(p1, p2) (default); late: after the dropout; "
                         "branch: Linear(2, k) projection concatenated at the fusion")
    ap.add_argument("--extscore-branch-dim", type=int, default=8)
    ap.add_argument("--vram-budget-gb", type=float, default=4.0,
                    help="splits are placed on the GPU while their running total stays under this and "
                         "under the free VRAM minus --vram-reserve-gb (TwiBot-22: 3.07 GB in total)")
    ap.add_argument("--vram-reserve-gb", type=float, default=1.5,
                    help="VRAM left free (activations, other jobs on the shared GPU) when placing splits")
    ap.add_argument("--chunk-rows", type=int, default=32_768, help="host->GPU streaming block (rows)")
    ap.add_argument("--eval-batch", type=int, default=16_384)
    ap.add_argument("--cuda-graph", action="store_true",
                    help="speed-up (spec 7): replay full training batches as one CUDA graph (about 5x "
                         "faster under GPU contention; same update rule and dropout stream as eager)")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--log-test-epochs", action="store_true",
                    help="also evaluate test after every epoch into epochs.jsonl (the original prints it); off "
                         "by default: test is scored once per selection rule, after training")
    ap.add_argument("--measure-epochs", type=int, default=0,
                    help="time K epochs of one seed and exit (no Run); zero placeholders for missing text")
    add_extscore_args(ap)
    a = ap.parse_args(argv)
    for k, v in PRESETS[a.dataset].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    a.chunk_rows = max(a.batch_size, a.chunk_rows // a.batch_size * a.batch_size)
    return a


def variant_name(a) -> str:
    v = [f"text-{a.features}"]
    if a.select != "last_epoch":
        v.append(a.select)
    if a.cuda_graph:
        v.append("cudagraph")
    if a.class_weight != "none":
        v.append("cw-" + a.class_weight.replace(",", "-"))
    if (a.with_extscore or a.extscore == "p") and a.extscore_inject != "fuse":
        v.append(f"extscore{a.extscore_inject}" + (str(a.extscore_branch_dim) if a.extscore_inject == "branch" else ""))
    return ".".join(v)


def with_human(m: dict) -> dict:
    """Add human-class precision/recall (``compute_metrics`` already has ``f1_human``)."""
    c = m["confusion"]
    tn, fn, fp = c["tn"], c["fn"], c["fp"]
    return {**m, "precision_human": tn / (tn + fn) if tn + fn else 0.0,
            "recall_human": tn / (tn + fp) if tn + fp else 0.0}


# ------------------------------------------------------------------------------------- inputs
def text_paths(dataset: str, variant: str) -> tuple[Path, Path]:
    """(tweets, des) .npy paths of the botrgcn text scheme; raises if not built."""
    from ...data import botrgcn_text
    des, tw = botrgcn_text(dataset, variant)
    return Path(tw.filename), Path(des.filename)


def read_text_rows(paths, splits: dict[str, np.ndarray], chunk: int = 50_000,
                   placeholder_n: int | None = None, out: dict | None = None) -> dict:
    """``{split: fp16 [n_split, 768 * len(paths)]}``: rows ``splits[s]`` of each table side by side,
    read in contiguous blocks with plain reads (no file mapping stays resident). ``out`` gives the
    destinations (numpy arrays, CPU or CUDA tensors; default: new numpy arrays); a CUDA
    destination is filled block by block, so the split is never held on the host in full."""
    from ...io import read_rows

    if out is None:
        out = {s: np.empty((len(r), EMB * len(paths)), np.float16) for s, r in splits.items()}
    if placeholder_n is not None:
        for v in out.values():
            v[:] = 0
        return out
    dst = {}                                       # numpy view, or (cuda tensor, device)
    for s, v in out.items():
        if isinstance(v, np.ndarray):
            dst[s] = v
        elif v.device.type == "cpu":
            dst[s] = v.numpy()
        else:
            dst[s] = (v, v.device)
    srt = {s: (np.argsort(r, kind="stable"), np.sort(r)) for s, r in splits.items()}
    lo = min(int(r.min()) for r in splits.values() if len(r))
    hi = max(int(r.max()) for r in splits.values() if len(r)) + 1
    for s0 in range(lo, hi, chunk):
        e0 = min(hi, s0 + chunk)
        hits = {}
        for s, (order, rs) in srt.items():
            a, b = np.searchsorted(rs, [s0, e0])
            if b > a:
                hits[s] = (order[a:b], rs[a:b] - s0)
        if not hits:
            continue
        for k, p in enumerate(paths):
            blk = read_rows(p, s0, e0)
            for s, (pos, loc) in hits.items():
                d = dst[s]
                if isinstance(d, np.ndarray):
                    d[pos, k * EMB:(k + 1) * EMB] = blk[loc]
                else:
                    import torch
                    t, dev = d
                    t[torch.from_numpy(pos).to(dev), k * EMB:(k + 1) * EMB] = torch.from_numpy(blk[loc]).to(dev)
            del blk
    return out


@dataclass
class SplitData:
    """One split's inputs: ``x`` [n, 1536] fp16 and ``extscore`` [n, 2] fp32 (or None) on the GPU or
    on the host; ``y`` [n] int64 always on the compute device."""
    name: str
    idx: np.ndarray
    x: "torch.Tensor"
    extscore: "torch.Tensor | None"
    y: "torch.Tensor"
    on_device: bool = True       # False: x/extscore live on the host and are streamed

    @property
    def n(self) -> int:
        return len(self.idx)

    def nbytes(self) -> int:
        return int(self.x.numel() * self.x.element_size() + (0 if self.extscore is None else self.extscore.numel() * 4))


class Streamer:
    """Yields ``(x, extscore, y)`` blocks of a split on the compute device, in ``order`` (a CPU int64
    permutation) or in row order. Host-resident splits go through two pinned staging buffers: a
    worker thread gathers block i+1 into one while block i (copied out of the other) trains. The
    host-to-device copy is synchronous, so a buffer is free again as soon as its block is on the
    device. Values are pure copies: every placement yields the same blocks."""

    def __init__(self, device, chunk_rows: int):
        import torch
        self.dev, self.chunk_rows = torch.device(device), chunk_rows
        self._stage = None

    def stage(self, width: int):
        import torch
        if self._stage is None or self._stage[0].shape[1] != width:
            self._stage = [torch.empty((self.chunk_rows, width), dtype=torch.float16,
                                       pin_memory=self.dev.type == "cuda") for _ in range(2)]
        return self._stage

    def blocks(self, sp: SplitData, order=None):
        n, cr = sp.n, self.chunk_rows
        order_dev = None if order is None else order.to(self.dev)
        starts = range(0, n, cr)
        if sp.on_device:
            for s in starts:
                sel = slice(s, min(n, s + cr)) if order is None else order_dev[s:s + cr]
                yield sp.x[sel], None if sp.extscore is None else sp.extscore[sel], sp.y[sel]
            return
        from concurrent.futures import ThreadPoolExecutor

        import torch
        bufs = self.stage(sp.x.shape[1])

        def gather(i):
            s = starts[i]
            st = bufs[i % 2][: min(n, s + cr) - s]
            if order is None:
                st.copy_(sp.x[s:s + cr])
            else:
                torch.index_select(sp.x, 0, order[s:s + cr], out=st)
            return st

        if not len(starts):
            return
        with ThreadPoolExecutor(1) as ex:
            fut = ex.submit(gather, 0)
            for i, s in enumerate(starts):
                st = fut.result()
                x = st.to(self.dev) if self.dev.type != "cpu" else st.clone()   # buffer free on return
                if i + 1 < len(starts):
                    fut = ex.submit(gather, i + 1)
                sel = slice(s, s + cr) if order is None else order[s:s + cr]
                sel_dev = slice(s, s + cr) if order is None else order_dev[s:s + cr]
                j = None if sp.extscore is None else sp.extscore[sel].to(self.dev)
                yield x, j, sp.y[sel_dev]


def vram_budget(a, dev: str) -> float:
    """Bytes of inputs that may be placed on ``dev``: ``--vram-budget-gb``, capped by the free VRAM
    minus ``--vram-reserve-gb`` (the GPU is shared; placement never changes the numbers)."""
    import torch
    budget = a.vram_budget_gb * 1e9
    if torch.device(dev).type == "cuda":
        free, _ = torch.cuda.mem_get_info()
        budget = min(budget, free - a.vram_reserve_gb * 1e9)
    return budget


def build_splits(a, dataset: str, splits: dict[str, np.ndarray], y: np.ndarray, J, dev: str,
                 placeholder: bool = False, force_host: bool = False) -> tuple[dict[str, SplitData], list[Path], dict]:
    import torch

    t0 = time.time()
    paths = [] if placeholder else list(text_paths(dataset, a.features))
    width = EMB * 2
    jcols = J.append(np.zeros((len(y), 0), np.float32), kind="neural") if J.enabled else None   # [N, 2]
    on_cpu = torch.device(dev).type == "cpu"
    budget = float("inf") if on_cpu else vram_budget(a, dev)
    dest, put, used, placement = {}, {}, 0, {}
    for s in ("test", "val", "train"):              # eval splits first: they are read every epoch
        n = len(splits[s])
        sz = n * width * 2 + (0 if jcols is None else n * jcols.shape[1] * 4)
        put[s] = not force_host and used + sz <= budget
        if put[s] and not on_cpu:
            try:
                dest[s] = torch.empty((n, width), dtype=torch.float16, device=dev)
            except torch.cuda.OutOfMemoryError:     # another job took the memory since the check
                put[s] = False
        if s not in dest:
            dest[s] = torch.empty((n, width), dtype=torch.float16)
        if put[s]:
            used += sz
        placement[s] = "device" if put[s] else "host"
    read_text_rows(paths or [None, None], splits, placeholder_n=len(y) if placeholder else None, out=dest)
    out = {}
    for s in ("test", "val", "train"):
        js = None if jcols is None else torch.from_numpy(np.ascontiguousarray(jcols[splits[s]]))
        if put[s] and js is not None:
            js = js.to(dev)
        ys = torch.from_numpy(np.asarray(y[splits[s]], np.int64)).to(dev)
        out[s] = SplitData(s, splits[s], dest.pop(s), js, ys, on_device=put[s])
    info = {"load_seconds": round(time.time() - t0, 2), "placement": placement, "device_input_bytes": used,
            "host_input_bytes": sum(v.nbytes() for v in out.values() if not v.on_device)}
    return out, paths, info


# ------------------------------------------------------------------------------------ training
def predict(model, sp: SplitData, streamer: Streamer, eval_batch: int):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for x, j, _ in streamer.blocks(sp):
            for k in range(0, len(x), eval_batch):
                lg = model(x[k:k + eval_batch], None if j is None else j[k:k + eval_batch])
                out.append(torch.softmax(lg.float(), 1)[:, 1])
    return torch.cat(out).cpu().numpy().astype(np.float64)


def build_model(a, J, dev):
    from .model import RobertaMLP
    return RobertaMLP(a.dropout, extscore_width=J.width, extscore_inject=a.extscore_inject, branch_dim=a.extscore_branch_dim).to(dev)


class Stepper:
    """One optimiser step per batch. Eager by default. With ``graph`` (``--cuda-graph``), full
    batches replay one captured CUDA graph (forward + backward + ``Adam(capturable=True)``, same
    update rule); the first ``warmup`` steps (real training steps, on a side stream, as the CUDA
    graph recipe requires) and the partial last batch of each epoch run eagerly on the same
    parameter, gradient and optimiser-state tensors. Dropout draws from the graph-safe CUDA
    generator, whose Philox offsets advance as in eager mode, so graph and eager draw the same
    masks (``test_cuda_graph_dropout_stream_matches_eager``); on Cresci-2015 the per-epoch train
    loss of graph and eager runs agrees to about 1e-6 for dozens of epochs and then drifts apart by
    float rounding only (capturable vs foreach Adam arithmetic)."""

    def __init__(self, model, loss_fn, opt, bs: int, dev, graph: bool = False, warmup: int = 3):
        import torch
        self.model, self.loss_fn, self.opt, self.bs, self.graph, self.warmup = model, loss_fn, opt, bs, graph, warmup
        self.loss_sum = torch.zeros((), device=dev)
        self.n_steps, self.g, self.static = 0, None, None

    def _eager(self, x, j, y, set_to_none: bool):
        loss = self.loss_fn(self.model(x, j), y)
        self.opt.zero_grad(set_to_none=set_to_none)
        loss.backward()
        self.opt.step()
        self.loss_sum.add_(loss.detach() * len(y))

    def _capture(self, x, j, y):
        import torch
        sx, sy = x.clone(), y.clone()
        sj = None if j is None else j.clone()
        self.opt.zero_grad(set_to_none=True)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            loss = self.loss_fn(self.model(sx, sj), sy)
            loss.backward()
            self.opt.step()
            self.loss_sum.add_(loss.detach() * self.bs)
        self.static = (sx, sj, sy)

    def step(self, x, j, y) -> None:
        import torch
        if not self.graph:
            self._eager(x, j, y, True)
        elif self.g is None and (self.n_steps < self.warmup or len(y) != self.bs):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._eager(x, j, y, True)
            torch.cuda.current_stream().wait_stream(s)
        else:
            if self.g is None:
                self._capture(x, j, y)
            if len(y) != self.bs:
                self._eager(x, j, y, False)          # accumulate into the graph's gradient tensors
            else:
                sx, sj, sy = self.static
                sx.copy_(x)
                sy.copy_(y)
                if sj is not None:
                    sj.copy_(j)
                self.g.replay()
        self.n_steps += 1


def train_one(a, seed: int, data: dict[str, SplitData], y: np.ndarray, J, dev: str, run=None,
              epochs: int | None = None) -> dict:
    """Train one seed. Returns the last-epoch model, the val-best checkpoint and timing. Loss uses
    train rows only; val is evaluated after every epoch (val-best selection), test only with
    ``--log-test-epochs`` (diagnostic)."""
    import torch

    from ...harness import class_weights
    set_seed(seed, deterministic=a.deterministic)
    model = build_model(a, J, dev)
    w = class_weights(y[data["train"].idx], a.class_weight)
    loss_fn = torch.nn.CrossEntropyLoss(weight=None if w is None else torch.tensor(w, dtype=torch.float32, device=dev))
    graph = bool(a.cuda_graph) and torch.device(dev).type == "cuda"
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd, **({"capturable": True} if graph else {}))
    gen = torch.Generator().manual_seed(seed)       # shuffling (train only; val/test in row order)
    streamer = Streamer(dev, a.chunk_rows)
    best = BestCheckpoint(select="best_val", mode="max")
    tr, bs = data["train"], a.batch_size
    stepper = Stepper(model, loss_fn, opt, bs, dev, graph=graph)
    epochs = a.epochs if epochs is None else epochs
    t_train = 0.0
    hist = []
    for ep in range(epochs):
        t = time.time()
        model.train()
        stepper.loss_sum.zero_()
        for x, j, yb in streamer.blocks(tr, order=torch.randperm(tr.n, generator=gen)):
            for k in range(0, len(x), bs):
                stepper.step(x[k:k + bs], None if j is None else j[k:k + bs], yb[k:k + bs])
        loss_sum = stepper.loss_sum
        if torch.device(dev).type == "cuda":
            torch.cuda.synchronize()
        t_train += time.time() - t
        mv = compute_metrics(y[data["val"].idx], predict(model, data["val"], streamer, a.eval_batch))
        rec = {"train_loss": float(loss_sum) / tr.n, "val_acc": mv["accuracy"], "val_f1": mv["f1"],
               "val_f1_human": mv["f1_human"]}
        if a.log_test_epochs:                         # diagnostic only; never used for selection
            mt = compute_metrics(y[data["test"].idx], predict(model, data["test"], streamer, a.eval_batch))
            rec.update(test_acc=mt["accuracy"], test_f1=mt["f1"], test_f1_human=mt["f1_human"])
        rec["epoch_seconds"] = round(time.time() - t, 3)
        hist.append(rec)
        if run is not None:
            run.log_epoch(ep, **rec)
        best.update(model, mv[a.select_metric], ep)
    return {"model": model, "best": best, "streamer": streamer, "epochs_run": epochs,
            "train_seconds": round(t_train, 2), "seconds_per_epoch": round(t_train / max(epochs, 1), 3),
            "history": hist}


def evaluate(model, data, y, streamer, eval_batch, splits=("train", "val", "test")):
    probs = {s: predict(model, data[s], streamer, eval_batch) for s in splits}
    return probs, {s: with_human(compute_metrics(y[data[s].idx], probs[s])) for s in splits}


def run_seed(a, seed, data, y, J, dev, paths, load_info) -> dict:
    import torch

    from .model import n_params
    if torch.device(dev).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    cfg = {**vars(a), "variant": variant_name(a)}
    with Run("roberta_mlp", a.dataset, config=cfg, seed=seed, data_files=paths, source_files=SRC) as run:
        r = train_one(a, seed, data, y, J, dev, run=run)
        model, best, st = r["model"], r["best"], r["streamer"]
        p_last, m_last = evaluate(model, data, y, st, a.eval_batch)
        best.restore(model)                         # deep copy of the val-best state
        p_best, m_best = evaluate(model, data, y, st, a.eval_batch)
        last = {"selection": {"select": "last_epoch", "epoch": r["epochs_run"] - 1}, "metrics": m_last}
        bestv = {"selection": {**best.info(), "select": "best_val", "metric": a.select_metric}, "metrics": m_best}
        prim, sec = (last, bestv) if a.select == "last_epoch" else (bestv, last)
        probs = p_last if a.select == "last_epoch" else p_best
        for s in ("val", "test"):
            run.add_predictions(s, data[s].idx, probs[s], y[data[s].idx])
        out = run.finish(
            prim["metrics"], selection=prim["selection"], secondary=sec, epochs_run=r["epochs_run"],
            extscore=J.report(), with_extscore=a.with_extscore, n_params=n_params(model),
            timing={"load_seconds": load_info["load_seconds"], "train_seconds": r["train_seconds"],
                    "seconds_per_epoch": r["seconds_per_epoch"]},
            gpu={**{k: load_info[k] for k in ("placement", "device_input_bytes", "host_input_bytes")},
                 "run_max_allocated_bytes": int(torch.cuda.max_memory_allocated()) if dev == "cuda" else 0})
    t, tb = prim["metrics"]["test"], sec["metrics"]["test"]
    print(f"[roberta_mlp {a.dataset} seed {seed}] test acc {t['accuracy']:.4f} f1(bot) {t['f1']:.4f} "
          f"prec {t['precision']:.4f} rec {t['recall']:.4f} f1(human) {t['f1_human']:.4f} auc {t['roc_auc']:.4f} "
          f"| {sec['selection']['select']} (ep {sec['selection'].get('best_epoch', sec['selection'].get('epoch'))}): "
          f"acc {tb['accuracy']:.4f} f1 {tb['f1']:.4f} | {r['seconds_per_epoch']:.2f} s/epoch", flush=True)
    del model
    return out


def measure(a, data, y, J, dev, load_info) -> dict:
    import json

    import torch

    from ...mem import memory_report
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    r = train_one(a, a.seeds[0], data, y, J, dev, epochs=a.measure_epochs)
    rep = {"dataset": a.dataset, "epochs": r["epochs_run"], "train_seconds_per_epoch": r["seconds_per_epoch"],
           "epoch_seconds_incl_eval": [h["epoch_seconds"] for h in r["history"]],
           "projected_seconds_per_seed": round(np.mean([h["epoch_seconds"] for h in r["history"]]) * a.epochs, 1),
           **load_info,
           "cuda_max_allocated_GB": round(torch.cuda.max_memory_allocated() / 1e9, 3) if dev == "cuda" else 0,
           "cuda_max_reserved_GB": round(torch.cuda.max_memory_reserved() / 1e9, 3) if dev == "cuda" else 0,
           "memory": memory_report()}
    print("MEASURE " + json.dumps(rep, default=str), flush=True)
    return rep


def main(argv=None) -> list[dict]:
    import torch

    from ...data import load_labels, load_split
    a = parse_args(argv)
    y = np.asarray(load_labels(a.dataset)).astype(np.int64)
    splits = {s: np.asarray(load_split(a.dataset, s), np.int64) for s in ("train", "val", "test")}
    for s, r in splits.items():
        if np.any((y[r] != 0) & (y[r] != 1)):
            raise ValueError(f"{a.dataset}/{s}: unlabelled rows in an evaluated split")
    J = prepare_extscore(a, a.dataset, splits, labels=y)          # no graph: no neighbour guard needed
    dev = "cuda" if (a.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    placeholder = False
    if a.measure_epochs:
        try:
            text_paths(a.dataset, a.features)
        except FileNotFoundError:
            placeholder = True
    data, paths, info = build_splits(a, a.dataset, splits, y, J, dev, placeholder=placeholder)
    print(f"{a.dataset}/{a.features}: train/val/test {[data[s].n for s in ('train', 'val', 'test')]} "
          f"inputs {info['placement']} device {info['device_input_bytes'] / 1e9:.2f} GB host "
          f"{info['host_input_bytes'] / 1e9:.2f} GB (load {info['load_seconds']} s) "
          f"lr {a.lr} wd {a.wd} bs {a.batch_size} epochs {a.epochs} dropout {a.dropout} "
          f"extscore={'on:' + a.extscore_inject if J.enabled else 'off'}{' PLACEHOLDER TEXT' if placeholder else ''}",
          flush=True)
    if a.measure_epochs:
        return [measure(a, data, y, J, dev, info)]
    results = [run_seed(a, s, data, y, J, dev, paths, info) for s in a.seeds]
    agg = summarize([r["metrics"]["test"] for r in results], keys=REPORT_KEYS)
    print("test mean/sd:", {k: (round(v["mean"], 4), round(v["std"], 4)) for k, v in agg.items()})
    if J.enabled and J.baseline:
        b = J.baseline["test"]
        print(f"column-only baseline (val-tuned threshold {J.baseline['threshold']:.3f}): "
              f"test acc {b['accuracy']:.4f} f1 {b['f1']:.4f}")
    return results
