"""SMOTE and Edited Nearest Neighbours, re-implemented to match imbalanced-learn 0.12 defaults.

Only numpy + scikit-learn are imported here, so this file can be imported by path in an
environment that has imbalanced-learn installed and cross-checked against it (see
``tests/methods/test_kudugunta.py::test_matches_imblearn``).

What imblearn does (0.12.4, read from its source):

SMOTE(k_neighbors=5, sampling_strategy='auto', random_state=r)
    'auto' = 'not majority': every class except the majority (first maximum) is raised to the
    majority count. For each such class c: NearestNeighbors(6) fitted on X[y == c] and queried on
    the same rows; column 0 is dropped (normally the point itself; with duplicates it can be a
    duplicate, imblearn does not care and neither do we). ``RandomState(r)`` (re-created per
    class) draws ``randint(0, nns.size, n_new)`` (flat index -> base row, neighbour column) then
    ``uniform(size=n_new)`` steps; new = x_base + step * (x_nn - x_base), cast to X.dtype. The
    output is ``vstack([X, new_c1, new_c2, ...])``.

EditedNearestNeighbours(n_neighbors=3, kind_sel='all', sampling_strategy='auto')
    'auto' = 'not minority': the minority class is ``min(counts, key=counts.get)`` over the classes
    in sorted order, i.e. the FIRST class with the smallest count. After SMOTE both classes have
    equal counts, so the "minority" is class 0 and only class 1 (bots, real and synthetic) is
    edited. NearestNeighbors(4) is fitted on all rows; each row of an edited class is kept iff all
    3 neighbours after dropping column 0 share its label. Output order: classes in sorted order,
    rows in their original order within a class.

kNN tie order (review round 1): every kNN query goes through ``knn_query``, which takes the kd-tree's
unsorted k-neighbour heap and orders it with the scikit-learn <= 1.5 sort. The neighbour sets are the
same in every scikit-learn version, but 1.6 changed the sort and with it the order of EQUAL-distance
neighbours; since imblearn drops column 0, that order decides which of a set of duplicate rows is
dropped. With the legacy order the outputs here equal imblearn 0.12.4 on scikit-learn 1.5.2 bit for
bit even though the project runs scikit-learn 1.9 (``tie_order='installed'`` = 1.9's order).

Additions (ours, off unless requested):
- ``dist_cols``: compute neighbours on a subset of columns (ExtScore excluded from distances).
- ``binary_pairs``: (value_col, indicator_col) pairs (the ExtScore ``[p, m]`` columns). Where the two
  parents of a synthetic row disagree on the indicator, the whole pair is copied from the nearer
  parent (step < 0.5 -> base row, else neighbour), so the indicator stays exactly 0/1 and the value
  is the sentinel/fill exactly when the indicator says missing. Where they agree, the pair is
  interpolated like any other feature (both missing -> the sentinel is preserved exactly).
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _legacy_sort_rows(rd: np.ndarray, ind: np.ndarray) -> None:
    """In place, per row: scikit-learn <= 1.5 ``utils._sorting.simultaneous_sort`` (median-of-3
    quicksort, special cases for 2 and 3 items), which orders the k neighbours of a kd-tree query.
    scikit-learn 1.6 replaced it with an introsort (insertion sort for k <= 15) that orders
    EQUAL-distance neighbours differently. imblearn drops column 0 of the kNN result (assumed to
    be the point itself), so with duplicate rows the tie order decides which neighbour is dropped
    (review round 1, docs/results/kudugunta.md)."""
    stack = np.empty((64, 2), np.int64)
    for r in range(rd.shape[0]):
        v, ix = rd[r], ind[r]
        stack[0, 0], stack[0, 1] = 0, rd.shape[1]
        top = 1
        while top > 0:
            top -= 1
            o, n = stack[top, 0], stack[top, 1]
            if n <= 1:
                continue
            if n == 2:
                if v[o] > v[o + 1]:
                    v[o], v[o + 1] = v[o + 1], v[o]
                    ix[o], ix[o + 1] = ix[o + 1], ix[o]
                continue
            if n == 3:
                if v[o] > v[o + 1]:
                    v[o], v[o + 1] = v[o + 1], v[o]
                    ix[o], ix[o + 1] = ix[o + 1], ix[o]
                if v[o + 1] > v[o + 2]:
                    v[o + 1], v[o + 2] = v[o + 2], v[o + 1]
                    ix[o + 1], ix[o + 2] = ix[o + 2], ix[o + 1]
                    if v[o] > v[o + 1]:
                        v[o], v[o + 1] = v[o + 1], v[o]
                        ix[o], ix[o + 1] = ix[o + 1], ix[o]
                continue
            p, last = o + n // 2, o + n - 1
            if v[o] > v[last]:
                v[o], v[last] = v[last], v[o]
                ix[o], ix[last] = ix[last], ix[o]
            if v[last] > v[p]:
                v[last], v[p] = v[p], v[last]
                ix[last], ix[p] = ix[p], ix[last]
                if v[o] > v[last]:
                    v[o], v[last] = v[last], v[o]
                    ix[o], ix[last] = ix[last], ix[o]
            pv = v[last]
            s = o
            for i in range(o, last):
                if v[i] < pv:
                    v[i], v[s] = v[s], v[i]
                    ix[i], ix[s] = ix[s], ix[i]
                    s += 1
            v[s], v[last] = v[last], v[s]
            ix[s], ix[last] = ix[last], ix[s]
            pi = s - o
            if pi > 1:
                stack[top, 0], stack[top, 1] = o, pi
                top += 1
            if pi + 2 < n:
                stack[top, 0], stack[top, 1] = s + 1, n - pi - 1
                top += 1


def _rdist_rows(Xf: np.ndarray, Xq: np.ndarray, ind: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance (the kd-tree's reduced distance, summed feature by feature in
    float64 like ``EuclideanDistance.rdist``) between each query row and its neighbours."""
    out = np.empty(ind.shape, np.float64)
    for r in range(ind.shape[0]):
        for c in range(ind.shape[1]):
            d = 0.0
            for j in range(Xq.shape[1]):
                t = Xq[r, j] - Xf[ind[r, c], j]
                d += t * t
            out[r, c] = d
    return out


try:  # numba in the project env; the isolated library-check env runs the same code in pure Python
    import numba as _nb
    _legacy_sort_rows = _nb.njit(cache=True, nogil=True)(_legacy_sort_rows)
    _rdist_rows = _nb.njit(cache=True, nogil=True)(_rdist_rows)
except ImportError:  # pragma: no cover
    pass


def knn_indices(X_fit: np.ndarray, X_query: np.ndarray, k: int, n_jobs: int | None = 8,
                tie_order: str = "legacy") -> np.ndarray:
    """Indices of the k nearest rows of X_fit for each row of X_query (Euclidean), using the
    same estimator imblearn uses (``NearestNeighbors`` defaults; algorithm 'auto' -> kd-tree on
    low-dimensional dense data). ``n_jobs`` does not change the result."""
    return knn_query(NearestNeighbors(n_neighbors=k, n_jobs=n_jobs).fit(X_fit), X_query, n_jobs, tie_order)


def knn_query(nn: NearestNeighbors, X_query: np.ndarray, n_jobs: int | None = 8,
              tie_order: str = "legacy") -> np.ndarray:
    """``nn.kneighbors(X_query, return_distance=False)`` with a version-independent tie order.

    ``tie_order='legacy'`` (default): the kd-tree's unsorted k-neighbour heap is ordered with the
    scikit-learn <= 1.5 sort, so the result equals ``kneighbors`` of scikit-learn 1.5.2 (the
    version imblearn 0.12.4 is checked against; that sort is the one every scikit-learn up to 1.5
    used, so also the benchmark's 2022 version) whatever scikit-learn is installed. The neighbour
    SET is the same in every version (same tree, same heap); only the order of equal distances
    differs. ``'installed'``: plain ``kneighbors`` of the installed version (1.9 in the project)."""
    if tie_order not in ("legacy", "installed"):
        raise ValueError(tie_order)
    k = nn.n_neighbors
    if tie_order == "installed" or getattr(nn, "_fit_method", None) != "kd_tree":
        return nn.kneighbors(X_query, return_distance=False)
    Xq = np.ascontiguousarray(X_query, dtype=np.float64)
    Xf = np.ascontiguousarray(nn._fit_X, dtype=np.float64)
    n_chunks = max(1, min(n_jobs or 1, len(Xq) // 10_000))
    bounds = np.linspace(0, len(Xq), n_chunks + 1).astype(np.int64)

    def one(a, b):
        ind = nn._tree.query(Xq[a:b], k=k, return_distance=False, sort_results=False)
        ind = np.ascontiguousarray(ind, dtype=np.intp)
        _legacy_sort_rows(_rdist_rows(Xf, Xq[a:b], ind), ind)
        return ind

    if n_chunks == 1:
        return one(0, len(Xq))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(n_chunks) as ex:
        return np.concatenate(list(ex.map(lambda i: one(bounds[i], bounds[i + 1]), range(n_chunks))))


def smote(X: np.ndarray, y: np.ndarray, *, k_neighbors: int = 5, random_state: int | None = None,
          n_jobs: int | None = 8, dist_cols=None, binary_pairs=(), tie_order: str = "legacy",
          ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (X_res, y_res, info). ``info['parents']`` holds, per synthetic row, (base, neighbour)
    row indices into X (for tests and audits)."""
    X = np.asarray(X)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n_major = counts.max()
    c_major = classes[int(np.argmax(counts))]
    Xd = X if dist_cols is None else X[:, list(dist_cols)]
    outX, outy, parents, per_class = [X.copy()], [y.copy()], [], {}
    for c, n_c in zip(classes, counts):
        n_new = int(n_major - n_c)
        if c == c_major or n_new == 0:
            continue
        idx_c = np.flatnonzero(y == c)
        Xc = X[idx_c]
        nns = knn_indices(Xd[idx_c], Xd[idx_c], k_neighbors + 1, n_jobs, tie_order)[:, 1:]
        rs = np.random.RandomState(random_state) if not isinstance(random_state, np.random.RandomState) else random_state
        flat = rs.randint(low=0, high=nns.size, size=n_new)
        steps = rs.uniform(size=n_new)[:, np.newaxis]
        rows = np.floor_divide(flat, nns.shape[1])
        cols = np.mod(flat, nns.shape[1])
        nb = nns[rows, cols]
        new = (Xc[rows] + steps * (Xc[nb] - Xc[rows])).astype(X.dtype)
        n_fixed = 0
        for vcol, mcol in binary_pairs:
            mixed = Xc[rows, mcol] != Xc[nb, mcol]
            if mixed.any():
                near = np.where(steps[:, 0] < 0.5, rows, nb)[mixed]
                new[np.ix_(mixed, [vcol, mcol])] = Xc[np.ix_(near, [vcol, mcol])]
                n_fixed += int(mixed.sum())
        outX.append(new)
        outy.append(np.full(n_new, c, dtype=y.dtype))
        parents.append(np.stack([idx_c[rows], idx_c[nb]], 1))
        per_class[int(c)] = {"n_before": int(n_c), "n_synthetic": n_new, "binary_pair_copies": n_fixed}
    info = {"per_class": per_class,
            "parents": np.concatenate(parents) if parents else np.zeros((0, 2), np.int64)}
    return np.vstack(outX), np.hstack(outy), info


def enn(X: np.ndarray, y: np.ndarray, *, n_neighbors: int = 3, kind_sel: str = "all",
        n_jobs: int | None = 8, dist_cols=None, tie_order: str = "legacy") -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (X_res, y_res, info) with ``info['kept']`` = indices into X of the kept rows."""
    X = np.asarray(X)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    c_min = classes[int(np.argmin(counts))]          # first minimum, as imblearn (ties -> smallest label)
    Xd = X if dist_cols is None else X[:, list(dist_cols)]
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs).fit(Xd)
    kept, per_class = [], {}
    for c in classes:
        idx_c = np.flatnonzero(y == c)
        if c == c_min:
            kept.append(idx_c)
            per_class[int(c)] = {"n_before": int(len(idx_c)), "n_kept": int(len(idx_c)), "edited": False}
            continue
        nb = knn_query(nn, Xd[idx_c], n_jobs, tie_order)[:, 1:]
        lab = y[nb]
        if kind_sel == "all":
            ok = np.all(lab == c, axis=1)
        elif kind_sel == "mode":
            # scipy.stats.mode semantics: most frequent label, ties -> smallest label
            vals = np.unique(y)
            cnt = np.stack([(lab == v).sum(1) for v in vals], 1)
            ok = vals[np.argmax(cnt, 1)] == c
        else:
            raise ValueError(kind_sel)
        kept.append(idx_c[ok])
        per_class[int(c)] = {"n_before": int(len(idx_c)), "n_kept": int(ok.sum()), "edited": True}
    keep = np.concatenate(kept)
    return X[keep], y[keep], {"per_class": per_class, "kept": keep}
