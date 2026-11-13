"""Project-env half of the Kudugunta library cross-check (not a pytest file; loads real artifacts):

    tools/memrun --light -- uv run python tests/methods/kudugunta_dump_check.py cresci15 runs/kd_cr15.npz 0,3
    tools/memrun --light -- uv run python tests/methods/kudugunta_dump_check.py twibot22 runs/kd_tw22.npz 0

Writes ``Xtr, ytr, Xte, yte`` (default features of the dataset; ``misalign`` / ``tw20lit`` as a 4th
argument selects the emulations) and, per seed, the outputs of the port AS RUN IN THE PROJECT
ENVIRONMENT (the installed scikit-learn, 1.9.x): SMOTE rows, ENN kept indices, AdaBoost SAMME.R and
SAMME decision values on Xte and estimator weights. ``kudugunta_imblearn_check.py`` (isolated env
with imbalanced-learn 0.12.4 + scikit-learn 1.5.2) then compares the real libraries against these
outputs, i.e. the code path that produced the reported numbers, not only the port re-imported
under scikit-learn 1.5.2.
"""
import sys

import numpy as np

from bb.data import load_split, load_users
from bb.methods.kudugunta.boost import AdaBoostStumps
from bb.methods.kudugunta.features import build_features
from bb.methods.kudugunta.resample import enn, smote


def main():
    ds, out = sys.argv[1], sys.argv[2]
    seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 3]
    var = sys.argv[4] if len(sys.argv) > 4 else ""
    kw = {"bench_misalign": var == "misalign", "tw20_literal_bool_strings": var == "tw20lit"}
    y = load_users(ds, columns=["label"])["label"].to_numpy().astype(np.int64)
    tr, te = (np.asarray(load_split(ds, s)) for s in ("train", "test"))
    X, _, _ = build_features(ds, **kw)
    d = {"Xtr": X[tr], "ytr": y[tr], "Xte": X[te], "yte": y[te]}
    for s in seeds:
        Xs, ys, _ = smote(X[tr], y[tr], random_state=s)
        Xe, ye, ei = enn(Xs, ys)
        d[f"port_s{s}_smote_X"], d[f"port_s{s}_smote_y"], d[f"port_s{s}_enn_kept"] = Xs, ys, ei["kept"]
        for alg in ("SAMME.R", "SAMME"):
            m = AdaBoostStumps(100, algorithm=alg, random_state=100 * s).fit(Xe, ye)
            d[f"port_s{s}_{alg}_dec"] = m.decision_function(d["Xte"])
            d[f"port_s{s}_{alg}_w"] = m.estimator_weights_
        print(ds, var or "default", "seed", s, X[tr].shape, "->", Xs.shape, "->", Xe.shape, flush=True)
    np.savez(out, **d)


if __name__ == "__main__":
    main()
