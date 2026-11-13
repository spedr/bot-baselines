"""Reference method (smoke test of the contracts, not a published baseline): logistic regression
on a handful of profile features, trained full-batch with Adam, early stopping on val bot-F1,
best-by-value checkpoint, optional class weighting, optional ExtScore column via the common interface.

    uv run python -m bb.methods.logreg_ref --dataset twibot20 --seeds 0 1 2
    uv run python -m bb.methods.logreg_ref --dataset twibot20 --with-extscore synthetic

It shows every step a method agent takes: load the index and splits, project profile columns,
build features on train statistics, honour --extscore through prepare_extscore/append, train with
EarlyStopper/BestCheckpoint, evaluate with compute_metrics, write a Run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bb.data import load_profile, load_split, load_users
from bb.harness import BestCheckpoint, EarlyStopper, Run, aggregate, class_weights, compute_metrics, load_results, set_seed
from bb.extscore import add_extscore_args, prepare_extscore

REF_DATE = {"twibot20": "2020-09-01", "twibot22": "2022-04-20", "cresci15": "2015-12-31"}
COUNTS = ["followers_count", "following_count", "statuses_count", "listed_count"]


def features(dataset: str) -> tuple[np.ndarray, list[str]]:
    """[N, F] float32 raw features (not yet standardised); nulls -> 0 with an indicator column."""
    p = load_profile(dataset, COUNTS + ["created_at", "protected", "verified", "description", "name", "screen_name"])
    ref = pd.Timestamp(REF_DATE[dataset], tz="UTC")
    cols, names = [], []
    for c in COUNTS:
        cols.append(np.log1p(p[c].astype("float64").fillna(0).clip(lower=0).to_numpy()))
        names.append(f"log1p_{c}")
    age = (ref - p["created_at"]).dt.days.astype("float64")
    cols += [age.fillna(0).to_numpy(), age.isna().to_numpy().astype(float)]
    names += ["age_days", "created_at_missing"]
    for c in ("protected", "verified"):
        cols.append(p[c].astype("float64").fillna(0).to_numpy())
        names.append(c)
    for c in ("description", "name", "screen_name"):
        cols.append(p[c].str.len().astype("float64").fillna(0).to_numpy())
        names.append(f"len_{c}")
    cols.append(p["screen_name"].str.count(r"\d").astype("float64").fillna(0).to_numpy())
    names.append("digits_screen_name")
    return np.stack(cols, 1).astype(np.float32), names


def run_one(args, seed: int, X: np.ndarray, y: np.ndarray, splits: dict, J) -> dict:
    import torch

    set_seed(seed)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd                     # standardise base features on train rows only
    Xs = J.append(Xs, kind="neural")       # ExtScore [z, m] appended AFTER standardisation (never re-scaled)
    dev = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    xt = torch.from_numpy(Xs).to(dev)
    yt = torch.from_numpy(y.astype(np.int64)).to(dev)
    model = torch.nn.Linear(Xs.shape[1], 2).to(dev)
    w = class_weights(y[tr], args.class_weight)
    loss_fn = torch.nn.CrossEntropyLoss(weight=None if w is None else torch.tensor(w, dtype=torch.float32, device=dev))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    es = EarlyStopper(patience=args.patience, mode="max")
    ck = BestCheckpoint(select=args.select, mode="max")
    tr_t, va_t = torch.as_tensor(np.array(tr), device=dev), torch.as_tensor(np.array(va), device=dev)
    with Run("logreg_ref", args.dataset, config=vars(args), seed=seed,
             source_files=[Path(__file__)]) as run:
        for ep in range(args.epochs):
            model.train()
            opt.zero_grad()
            loss = loss_fn(model(xt[tr_t]), yt[tr_t])
            loss.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                pv = torch.softmax(model(xt[va_t]), 1)[:, 1].cpu().numpy()
            mv = compute_metrics(y[va], pv)
            run.log_epoch(ep, train_loss=loss.item(), val_f1=mv["f1"], val_acc=mv["accuracy"])
            ck.update(model, mv["f1"], ep)
            es.step(mv["f1"])
            if es.should_stop:
                break
        ck.restore(model)
        model.eval()
        with torch.no_grad():
            prob = torch.softmax(model(xt), 1)[:, 1].cpu().numpy()
        res = {s: compute_metrics(y[splits[s]], prob[splits[s]]) for s in ("train", "val", "test")}
        for s in ("val", "test"):
            run.add_predictions(s, splits[s], prob[splits[s]], y[splits[s]])
        return run.finish(res, selection=ck.info(), epochs_run=ep + 1, extscore=J.report(),
                          n_params=sum(p.numel() for p in model.parameters()))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="twibot20")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--select", choices=["best_val", "last_epoch"], default="best_val")
    ap.add_argument("--class-weight", default="none", help="none | balanced | sqrt | w0,w1")
    ap.add_argument("--device", default="cpu")
    add_extscore_args(ap)
    args = ap.parse_args(argv)
    users = load_users(args.dataset, columns=["label"])
    y = users["label"].to_numpy()
    splits = {s: np.asarray(load_split(args.dataset, s)) for s in ("train", "val", "test")}
    X, names = features(args.dataset)
    J = prepare_extscore(args, args.dataset, splits, labels=y)
    print(f"{args.dataset}: X {X.shape} features={names} extscore={'on' if J.enabled else 'off'}")
    out = [run_one(args, s, X, y, splits, J) for s in args.seeds]
    agg = aggregate(out)
    print({k: (round(v["mean"], 4), round(v["std"], 4)) for k, v in agg.items()})
    if J.enabled:
        b = J.baseline["test"]
        print(f"column-only baseline (val-tuned threshold {J.baseline['threshold']:.3f}): "
              f"test acc {b['accuracy']:.4f} f1 {b['f1']:.4f}")


if __name__ == "__main__":
    main()
