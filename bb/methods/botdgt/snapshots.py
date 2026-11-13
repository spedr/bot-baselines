"""BotDGT snapshots: a creation-date filtration of the final static graph (spec 3.3).

A user exists in snapshot t if its UTC ``created_at`` date is strictly before the snapshot date D_t;
an edge exists if both endpoints exist. Snapshot dates are the first days of months from 2008-01-01
up to the first month at which every user exists, keeping every ``interval``-th month counting back
from the last one (interval 12 = the paper's yearly setting).

Nothing is copied per snapshot. The artifacts under ``data/methods/botdgt/<dataset>/`` are:

- ``node_birth.i8.npy`` [N]       first snapshot index at which the user exists
- ``csc_perm.i64.npy`` [E]        permutation of the shared in-neighbour CSC (``load_in_merged``)
                                  that sorts every row by edge birth (stable, release order kept)
- ``row_cnt.i32.npy`` [T, N]      number of in-entries of row v that exist at snapshot t, so the
                                  snapshot-t CSC is the per-row prefix ``indptr[v] : indptr[v] + row_cnt[t, v]``
                                  of the permuted CSC
- ``lcc.f32.npy`` / ``blr.f32.npy`` [T, N]   local clustering coefficient and bidirectional-link ratio
- ``meta.json``                   dates, per-snapshot node/edge counts, input key

Edge birth = max(node_birth[src], node_birth[dst]). LCC uses the simple undirected union of both
relations (``nx.clustering`` semantics: distinct neighbours, self-loops excluded); BLR uses the
following relation only as a ``DiGraph`` (duplicates collapse, self-loops kept as in networkx).
Both are computed in one pass over the final graph: every triangle / edge is credited to the
snapshot at which it is born and a cumulative sum over t gives every snapshot.
"""
from __future__ import annotations

import datetime as dt
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit

from ...io import file_digest, load_npy, read_json, save_npy, write_json
from ...paths import dataset_paths

VERSION = 1
FILL_DATE = dt.date(2006, 3, 1)       # official TwiBot-20 fill for a missing created_at (spec 3.3.1)
START = dt.date(2008, 1, 1)           # first monthly candidate
EPOCH = dt.date(1970, 1, 1)
# relation that carries "v follows u" (BLR is computed on it only; spec 3.3.5 / pitfall 4)
FOLLOWING_REL = {"twibot20": "following", "twibot22": "following", "cresci15": "friend"}


def _days(d: dt.date) -> int:
    return (d - EPOCH).days


def created_days(dataset: str) -> tuple[np.ndarray, int]:
    """UTC creation date of every user as int32 days since 1970-01-01, missing -> FILL_DATE."""
    from ...data import load_profile
    c = load_profile(dataset, ["created_at"])["created_at"]
    miss = c.isna().to_numpy()
    ns = c.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]", na_value=np.datetime64("NaT"))
    days = ns.astype("datetime64[D]").astype(np.int64)        # floor to the UTC date
    days[miss] = _days(FILL_DATE)
    return days.astype(np.int32), int(miss.sum())


def snapshot_dates(days: np.ndarray, interval: int = 12, start: dt.date = START) -> tuple[list[dt.date], int]:
    """(selected snapshot dates, number of monthly candidates)."""
    last = int(np.max(days))
    months = []
    y, m = start.year, start.month
    while True:
        d = dt.date(y, m, 1)
        months.append(d)
        if _days(d) > last:          # every user exists: date(v) < d for all v
            break
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    sel = months[::-1][::interval][::-1]
    return sel, len(months)


def node_birth(days: np.ndarray, dates: list[dt.date]) -> np.ndarray:
    """First t with date(v) < D_t (strict)."""
    d = np.array([_days(x) for x in dates], dtype=np.int64)
    b = np.searchsorted(d, days.astype(np.int64), side="right")
    assert b.max() < len(dates)
    return b.astype(np.int8)


# ------------------------------------------------------------------------------------- numba kernels
@njit(cache=True)
def _triangles(indptr, indices, birth, T):
    """Per node and birth snapshot: undirected degree and triangle counts (simple graph, no loops).
    Forward algorithm on a degree-ordered orientation, O(m^1.5)."""
    n = indptr.shape[0] - 1
    deg = np.empty(n, np.int64)
    for v in range(n):
        deg[v] = indptr[v + 1] - indptr[v]
    # orientation u -> w iff (deg[u], u) < (deg[w], w)
    optr = np.zeros(n + 1, np.int64)
    for u in range(n):
        c = 0
        for j in range(indptr[u], indptr[u + 1]):
            w = indices[j]
            if deg[u] < deg[w] or (deg[u] == deg[w] and u < w):
                c += 1
        optr[u + 1] = optr[u] + c
    oind = np.empty(optr[n], np.int64)
    for u in range(n):
        p = optr[u]
        for j in range(indptr[u], indptr[u + 1]):
            w = indices[j]
            if deg[u] < deg[w] or (deg[u] == deg[w] and u < w):
                oind[p] = w
                p += 1
    dcnt = np.zeros((T, n), np.int32)
    tcnt = np.zeros((T, n), np.int32)
    for u in range(n):
        for j in range(indptr[u], indptr[u + 1]):
            w = indices[j]
            b = birth[u] if birth[u] > birth[w] else birth[w]
            dcnt[b, u] += 1
    mark = np.full(n, -1, np.int64)
    for u in range(n):
        for j in range(optr[u], optr[u + 1]):
            mark[oind[j]] = u
        for j in range(optr[u], optr[u + 1]):
            v = oind[j]
            buv = birth[u] if birth[u] > birth[v] else birth[v]
            for k in range(optr[v], optr[v + 1]):
                w = oind[k]
                if mark[w] == u:
                    b = buv if buv > birth[w] else birth[w]
                    tcnt[b, u] += 1
                    tcnt[b, v] += 1
                    tcnt[b, w] += 1
    return dcnt, tcnt


def lcc_from_counts(dcnt: np.ndarray, tcnt: np.ndarray) -> np.ndarray:
    d = np.cumsum(dcnt, axis=0, dtype=np.int64).astype(np.float64)
    t = np.cumsum(tcnt, axis=0, dtype=np.int64).astype(np.float64)
    den = d * (d - 1)
    out = np.zeros(d.shape, np.float32)
    ok = d >= 2
    out[ok] = (2.0 * t[ok] / den[ok]).astype(np.float32)
    return out


def compute_lcc(und, birth: np.ndarray, T: int) -> np.ndarray:
    dcnt, tcnt = _triangles(np.asarray(und.indptr, np.int64), np.asarray(und.indices, np.int64),
                            np.asarray(birth, np.int64), T)
    return lcc_from_counts(dcnt, tcnt)


def compute_blr(src: np.ndarray, dst: np.ndarray, birth: np.ndarray, T: int, n: int) -> np.ndarray:
    """BLR(v) = |{u: v->u and u->v}| / outdeg(v) on the following DiGraph of each snapshot."""
    key = np.unique(np.asarray(src, np.int64) * n + np.asarray(dst, np.int64))
    s, d = key // n, key % n
    rec = np.isin(d * n + s, key, assume_unique=False)
    b = np.maximum(birth[s], birth[d]).astype(np.int64)
    flat = b * n + s
    out_c = np.cumsum(np.bincount(flat, minlength=T * n).reshape(T, n), axis=0)
    rec_c = np.cumsum(np.bincount(flat[rec], minlength=T * n).reshape(T, n), axis=0)
    blr = np.zeros((T, n), np.float32)
    ok = out_c > 0
    blr[ok] = (rec_c[ok] / out_c[ok]).astype(np.float32)
    return blr


def sort_csc_by_birth(csc, birth: np.ndarray, T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(perm [E], row_cnt [T, N] int32, edge_birth [E] int8 in CSC order)."""
    n = csc.n
    deg = csc.degree()
    rows = np.repeat(np.arange(n, dtype=np.int64), deg)
    src = np.asarray(csc.indices, np.int64)
    eb = np.maximum(birth[rows], birth[src]).astype(np.int64)
    perm = np.lexsort((np.arange(len(src)), eb, rows))
    cnt = np.cumsum(np.bincount(eb * n + rows, minlength=T * n).reshape(T, n), axis=0).astype(np.int32)
    return perm.astype(np.int64), cnt, eb.astype(np.int8)


# ------------------------------------------------------------------------------------- build / load
def _undirected(ds: str, rels: list[str]):
    from ...data import load_undirected
    g = dataset_paths(ds).graph
    for r in (rels, rels[::-1]):
        if (g / ("undirected__" + "+".join(r) + ".csr")).exists():
            return load_undirected(ds, r)
    return load_undirected(ds, rels)


def relations_by_type(ds: str) -> list[str]:
    from ...data.graph import graph_meta
    ids = graph_meta(ds)["edge_type_ids"]
    return sorted(ids, key=ids.get)


def input_key(ds: str, interval: int) -> str:
    import hashlib
    dp = dataset_paths(ds)
    rels = relations_by_type(ds)
    parts = [str(VERSION), str(interval), FILL_DATE.isoformat(), START.isoformat(), ",".join(rels),
             file_digest(dp.profile)]
    for r in rels:
        parts += [file_digest(dp.csr(r) / "indptr.npy"), file_digest(dp.csr(r) / "indices.npy")]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def snap_dir(ds: str, interval: int = 12) -> Path:
    return dataset_paths(ds).method_dir("botdgt") / f"interval{interval}"


def build_snapshots(ds: str, interval: int = 12, force: bool = False) -> dict:
    """Build (idempotently) the snapshot artifacts of ``ds``. Returns their meta."""
    from ...data import load_csr, load_in_merged
    from ...mem import peak_rss_bytes

    out = snap_dir(ds, interval)
    key = input_key(ds, interval)
    if not force and (out / "meta.json").exists() and read_json(out / "meta.json").get("input_key") == key:
        return read_json(out / "meta.json")
    t0 = time.time()
    rels = relations_by_type(ds)
    days, n_missing = created_days(ds)
    dates, n_months = snapshot_dates(days, interval)
    T = len(dates)
    if T > 127:
        raise ValueError("more than 127 snapshots do not fit int8 births")
    birth = node_birth(days, dates)
    n = len(birth)
    csc = load_in_merged(ds, rels)
    perm, row_cnt, eb = sort_csc_by_birth(csc, birth.astype(np.int64), T)
    t1 = time.time()
    und = _undirected(ds, rels)
    lcc = compute_lcc(und, birth, T)
    t2 = time.time()
    fol = load_csr(ds, FOLLOWING_REL[ds])
    s, d = fol.coo()
    blr = compute_blr(s, d, birth.astype(np.int64), T, n)
    t3 = time.time()
    nodes_t = np.cumsum(np.bincount(birth, minlength=T)).tolist()
    edges_t = np.cumsum(np.bincount(eb.astype(np.int64), minlength=T)).tolist()
    und_deg_t = np.zeros(T, np.int64)
    rows_u = np.repeat(np.arange(n, dtype=np.int64), und.degree())
    bu = np.maximum(birth[rows_u], birth[np.asarray(und.indices)]).astype(np.int64)
    und_deg_t = (np.cumsum(np.bincount(bu, minlength=T)) // 2).tolist()
    del rows_u, bu
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    save_npy(tmp / "node_birth.i8.npy", birth)
    save_npy(tmp / "csc_perm.i64.npy", perm)
    save_npy(tmp / "row_cnt.i32.npy", row_cnt)
    save_npy(tmp / "lcc.f32.npy", lcc)
    save_npy(tmp / "blr.f32.npy", blr)
    meta = {
        "dataset": ds, "version": VERSION, "input_key": key, "interval_months": interval,
        "relations": rels, "following_relation": FOLLOWING_REL[ds], "fill_date": FILL_DATE.isoformat(),
        "n_missing_created_at": n_missing, "n_monthly_candidates": n_months, "T": T,
        "dates": [x.isoformat() for x in dates], "nodes_per_snapshot": nodes_t,
        "directed_edges_per_snapshot": edges_t, "undirected_simple_edges_per_snapshot": und_deg_t,
        "created_min": str(np.datetime64(int(days.min()), "D")), "created_max": str(np.datetime64(int(days.max()), "D")),
        "lcc_mean_last": float(lcc[-1].mean()), "blr_mean_last": float(blr[-1].mean()),
        "seconds": {"sort_csc": round(t1 - t0, 2), "lcc": round(t2 - t1, 2), "blr": round(t3 - t2, 2),
                    "total": round(time.time() - t0, 2)},
        "peak_rss_bytes": peak_rss_bytes(),
    }
    assert all(a <= b for a, b in zip(nodes_t, nodes_t[1:])) and nodes_t[-1] == n
    assert all(a <= b for a, b in zip(edges_t, edges_t[1:])) and edges_t[-1] == csc.nnz
    write_json(tmp / "meta.json", meta)
    if out.exists():
        shutil.rmtree(out)
    tmp.replace(out)
    return meta


@dataclass
class Snapshots:
    dataset: str
    meta: dict
    birth: np.ndarray       # int8 [N]
    indptr: np.ndarray      # int64 [N+1] of the in-neighbour CSC
    indices: np.ndarray     # int64 [E] sources, rows sorted by edge birth
    row_cnt: np.ndarray     # int32 [T, N]
    lcc: np.ndarray         # float32 [T, N]
    blr: np.ndarray         # float32 [T, N]
    dir: Path

    @property
    def T(self) -> int:
        return int(self.meta["T"])

    @property
    def n(self) -> int:
        return len(self.birth)

    @property
    def files(self) -> list[Path]:
        return sorted(p for p in self.dir.iterdir() if p.is_file())


def load_snapshots(ds: str, interval: int = 12, build: bool = True) -> Snapshots:
    from ...data import load_in_merged
    if build:
        build_snapshots(ds, interval)
    d = snap_dir(ds, interval)
    meta = read_json(d / "meta.json")
    csc = load_in_merged(ds, meta["relations"])
    perm = load_npy(d / "csc_perm.i64.npy", mmap=False)
    indices = np.asarray(csc.indices, np.int64)[perm]
    return Snapshots(ds, meta, load_npy(d / "node_birth.i8.npy", mmap=False),
                     np.ascontiguousarray(np.asarray(csc.indptr, np.int64)), np.ascontiguousarray(indices),
                     load_npy(d / "row_cnt.i32.npy", mmap=False), load_npy(d / "lcc.f32.npy", mmap=False),
                     load_npy(d / "blr.f32.npy", mmap=False), d)


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="build BotDGT snapshot artifacts")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--interval", type=int, default=12)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    m = build_snapshots(a.dataset, a.interval, force=a.force)
    print({k: m[k] for k in ("T", "dates", "nodes_per_snapshot", "directed_edges_per_snapshot",
                             "n_missing_created_at", "seconds", "peak_rss_bytes")})


if __name__ == "__main__":
    main()
