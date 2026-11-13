"""AdaBoost with decision stumps: SAMME.R (the benchmark's 2022 scikit-learn default) and SAMME.

scikit-learn removed SAMME.R in 1.6 (we have 1.9). This is a line-by-line port of
``sklearn.ensemble.AdaBoostClassifier`` 1.5.2 (``BaseWeightBoosting.fit``, ``_boost_real``,
``_boost_discrete``, ``decision_function``, ``_compute_proba_from_decision``) for two classes, with
``DecisionTreeClassifier(max_depth=1)`` as the weak learner and the same per-estimator seeding
(``_set_random_states``: one ``randint(int32 max)`` per estimator from ``RandomState(seed)``).
Only numpy, scipy and scikit-learn are imported, so the file can be cross-checked by path against
scikit-learn 1.5.2 in a throw-away environment.

X is converted to float32 once (the tree does this on every fit/predict anyway), which gives the
same splits and predictions without a 1.2M x F copy per boosting round.

Weak learner (review round 2, efficiency): ``stumps='fast'`` (default) replaces
``DecisionTreeClassifier(max_depth=1)`` (about 0.8 s per round on TwiBot-22: it re-sorts every feature
in every round) by a numba re-implementation that is bit-identical to the scikit-learn 1.9 tree:
``BestSplitter`` / ``Gini`` / ``DepthFirstTreeBuilder`` operation by operation (candidate positions
``x[p] > x[p-1] + FEATURE_THRESHOLD`` in float32, ``Criterion.update`` forward/reverse accumulation,
proxy ``-wR*giniR - wL*giniL``, strict ``>``, midpoint threshold rule, feature draws from the tree's
``rand_r`` state, root leaf rules, leaf values ``sum_c / weighted_n``) AND the order in which the tree
adds the weights, i.e. its unstable introsort chained over the drawn features and its final partition
swaps (``_sort3``, ``_round_chain``, ``_partition``). Weighted sums are order-dependent in the last
ulps, and AdaBoost's eps-clipped weights create near-ties between splits that only the exact order
resolves the library's way (review round 2 measured 1-3 differing stumps per TwiBot-20 seed and
decision differences up to 3e-3 with a stable-sort order). The sort orders do not depend on the
weights, so a thread pool computes them for the next rounds while the current round scans. ``stumps='sklearn'`` is the
library-backed path, kept for verification (``tests/methods/kudugunta_fast_check.py``;
docs/results/kudugunta.md, "Review round 2").
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.special import expit, xlogy
from sklearn.tree import DecisionTreeClassifier

ALGORITHMS = ("SAMME.R", "SAMME")
STUMPS = ("fast", "sklearn")
FEATURE_THRESHOLD = np.float32(1e-7)    # sklearn/tree/_partitioner.pxd (float32)
RAND_R_MAX = 2147483647                 # sklearn/tree/_utils.pxd
TREE_EPSILON = np.finfo(np.float64).eps  # sklearn/tree/_tree.pyx EPSILON


# ---------------------------------------------------------------------------- fast stump kernels
# Bit-exact emulation of DecisionTreeClassifier(max_depth=1).fit (scikit-learn 1.9). The weighted sums
# of the tree depend on the ORDER of the samples, and that order comes from an unstable sort: at the
# root, ``samples = arange(n)``; for every drawn feature (constant ones too) ``simultaneous_sort``
# (introsort, 3-way partition) re-sorts ``samples`` in place, starting from the previous feature's
# output; the split search scans each feature in its sorted order; finally ``partition_samples_final``
# swaps the last order into left/right and the children's sums follow that order. None of these
# orders depends on the sample weights, only on X and the tree's seed, so they are computed for several
# future boosting rounds in a thread pool (``_round_chain``, ``Presorted``) and each round then only runs the
# weighted scans (``_scan_features``) and the partition (``_partition``), in exactly the tree's order.

def _sort3(v, s, n0):
    """sklearn/utils/_sorting.pyx ``simultaneous_sort(values, indices, n, use_three_way_partition=True)``:
    introsort with median-of-3 pivot and 3-way partition, heapsort past depth 2*int(log2(n)),
    insertion sort for n <= 15. The recursion (left part recursive, right part looped) is replaced by
    an explicit stack; the parts are disjoint, so the processing order does not change the result."""
    if n0 == 0:
        return
    stk = np.empty((192, 3), np.int64)
    stk[0, 0] = 0
    stk[0, 1] = n0
    stk[0, 2] = 2 * int(np.log2(np.float64(n0)))
    top = 1
    while top > 0:
        top -= 1
        lo = stk[top, 0]
        n = stk[top, 1]
        maxd = stk[top, 2]
        if n > 15 and maxd <= 0:                           # heapsort(values + lo, n)
            start = (n - 2) // 2
            end = n
            while True:
                root = start                               # sift_down(start, end)
                while True:
                    child = root * 2 + 1
                    mx = root
                    if child < end and v[lo + mx] < v[lo + child]:
                        mx = child
                    if child + 1 < end and v[lo + mx] < v[lo + child + 1]:
                        mx = child + 1
                    if mx == root:
                        break
                    v[lo + root], v[lo + mx] = v[lo + mx], v[lo + root]
                    s[lo + root], s[lo + mx] = s[lo + mx], s[lo + root]
                    root = mx
                if start == 0:
                    break
                start -= 1
            end = n - 1
            while end > 0:
                v[lo], v[lo + end] = v[lo + end], v[lo]
                s[lo], s[lo + end] = s[lo + end], s[lo]
                root = 0                                   # sift_down(0, end)
                while True:
                    child = root * 2 + 1
                    mx = root
                    if child < end and v[lo + mx] < v[lo + child]:
                        mx = child
                    if child + 1 < end and v[lo + mx] < v[lo + child + 1]:
                        mx = child + 1
                    if mx == root:
                        break
                    v[lo + root], v[lo + mx] = v[lo + mx], v[lo + root]
                    s[lo + root], s[lo + mx] = s[lo + mx], s[lo + root]
                    root = mx
                end -= 1
            continue
        if n <= 15:                                        # insertion_sort
            for i in range(1, n):
                tv = v[lo + i]
                ti = s[lo + i]
                j = i
                while j > 0 and v[lo + j - 1] > tv:
                    v[lo + j] = v[lo + j - 1]
                    s[lo + j] = s[lo + j - 1]
                    j -= 1
                v[lo + j] = tv
                s[lo + j] = ti
            continue
        maxd -= 1
        a = v[lo]                                          # median3
        b = v[lo + n // 2]
        c = v[lo + n - 1]
        if a < b:
            if b < c:
                pivot = b
            elif a < c:
                pivot = c
            else:
                pivot = a
        elif b < c:
            if a < c:
                pivot = a
            else:
                pivot = c
        else:
            pivot = b
        i = 0
        lft = 0
        r = n
        while i < r:                                       # [< pivot] [== pivot] [> pivot]
            if v[lo + i] < pivot:
                v[lo + i], v[lo + lft] = v[lo + lft], v[lo + i]
                s[lo + i], s[lo + lft] = s[lo + lft], s[lo + i]
                i += 1
                lft += 1
            elif v[lo + i] > pivot:
                r -= 1
                v[lo + i], v[lo + r] = v[lo + r], v[lo + i]
                s[lo + i], s[lo + r] = s[lo + r], s[lo + i]
            else:
                i += 1
        stk[top, 0] = lo
        stk[top, 1] = lft
        stk[top, 2] = maxd
        stk[top + 1, 0] = lo + r
        stk[top + 1, 1] = n - r
        stk[top + 1, 2] = maxd
        top += 2


def _first_orders(Xc, first):
    """``first[f]`` = the tree's order after sorting feature f from ``samples = arange(n)``: the
    first draw of every round, the same in every round (cached once per fit)."""
    n = Xc.shape[1]
    for f in _prange(Xc.shape[0]):
        v = Xc[f].copy()
        s = np.arange(n).astype(first.dtype)
        _sort3(v, s, n)
        first[f, :] = s


def _round_chain(Xc, first, draws, slot, orders, final):
    """One round: the tree's sample order after sorting every drawn feature (``draws``, in draw
    order, constant features included; the first one comes from ``first``). ``orders[slot[f]]`` =
    order in which the split search scans non-constant feature f; ``final`` = order after the last
    draw (input of ``partition_samples_final``). Serial and GIL-free: rounds run in a thread pool."""
    n = Xc.shape[1]
    v = np.empty(n, np.float32)
    s = first[draws[0]].copy()
    for k in range(draws.shape[0]):
        f = draws[k]
        if k > 0:
            for i in range(n):
                v[i] = Xc[f, s[i]]
            _sort3(v, s, n)
        if slot[f] >= 0:
            orders[slot[f], :] = s
    final[:] = s


def _root_sums(y, w):
    """``ClassificationCriterion.init`` at the root: samples in index order (all weights > 0)."""
    s0 = 0.0
    s1 = 0.0
    wn = 0.0
    for i in range(y.shape[0]):
        if y[i] == 0:
            s0 += w[i]
        else:
            s1 += w[i]
        wn += w[i]
    return s0, s1, wn


def _scan_features(orders, xs, y, w, feats, slot, st0, st1, wn, best_out, pos_out):
    """``node_split_best``'s inner loop for each non-constant feature ``feats[j]`` scanned in the
    tree's order ``orders[slot[f]]`` (xs[f] = the sorted values, the same sequence for any order).
    ``best_out[j]``/``pos_out[j]``: first maximum of the proxy improvement (-inf / -1 if none)."""
    n = orders.shape[1]
    ft = np.float32(1e-7)
    for j in _prange(feats.shape[0]):
        f = feats[j]
        o = orders[slot[f]]
        x = xs[f]
        sl0 = 0.0
        sl1 = 0.0
        wl = 0.0
        pos = 0
        best = -np.inf
        bp = -1
        for p in range(1, n):
            if x[p] <= x[p - 1] + ft:           # PartitionerDense.next_p (float32 sum)
                continue
            if (p - pos) <= (n - p):             # ClassificationCriterion.update, forward
                for q in range(pos, p):
                    i = o[q]
                    if y[i] == 0:
                        sl0 += w[i]
                    else:
                        sl1 += w[i]
                    wl += w[i]
            else:                                # reverse_reset + backward
                sl0 = st0
                sl1 = st1
                wl = wn
                for q in range(n - 1, p - 1, -1):
                    i = o[q]
                    if y[i] == 0:
                        sl0 -= w[i]
                    else:
                        sl1 -= w[i]
                    wl -= w[i]
            pos = p
            wr = wn - wl
            sr0 = st0 - sl0
            sr1 = st1 - sl1
            if wl < 0.0 or wr < 0.0:             # min_weight_leaf = 0
                continue
            gl = 1.0 - (sl0 * sl0 + sl1 * sl1) / (wl * wl)   # Gini.children_impurity
            gr = 1.0 - (sr0 * sr0 + sr1 * sr1) / (wr * wr)
            proxy = -wr * gr - wl * gl                       # Criterion.proxy_impurity_improvement
            if proxy > best:
                best = proxy
                bp = p
        best_out[j] = best
        pos_out[j] = bp


def _partition(Xc, f, thr, smp, y, w, pos, st0, st1, wn, leaf):
    """``partition_samples_final`` on ``smp`` (in place), the children's ``init`` sums in the
    partitioned order (s[leaf, class], wc[leaf]), the left sums of ``reset(); update(pos)`` (which
    the improvement uses: forward over the left part, or total minus the right part backwards), and
    the leaf of every row (``x <= thr`` in float64, as the tree's predict)."""
    n = smp.shape[0]
    ps = 0
    pe = n
    while ps < pe:
        if Xc[f, smp[ps]] <= thr:
            ps += 1
        else:
            pe -= 1
            smp[ps], smp[pe] = smp[pe], smp[ps]
    s = np.zeros((2, 2))
    wc = np.zeros(2)
    for q in range(n):
        lf = 0 if q < ps else 1
        i = smp[q]
        s[lf, y[i]] += w[i]
        wc[lf] += w[i]
    upd = np.empty(3)
    if pos <= n - pos:
        upd[0] = s[0, 0]
        upd[1] = s[0, 1]
        upd[2] = wc[0]
    else:
        upd[0] = st0
        upd[1] = st1
        upd[2] = wn
        for q in range(n - 1, pos - 1, -1):
            i = smp[q]
            upd[y[i]] -= w[i]
            upd[2] -= w[i]
    for i in range(n):
        leaf[i] = 0 if Xc[f, i] <= thr else 1
    return ps, s, wc, upd


def _decide(X, feat, thr, C, W, out):
    """Sum of per-stump decision contributions C[t, leaf, class] (same order as ``_decision``),
    then ``-(a / W) + b / W`` (sklearn's binary decision)."""
    T = feat.shape[0]
    for r in _prange(X.shape[0]):
        a = 0.0
        b = 0.0
        for t in range(T):
            f = feat[t]
            lf = 0
            if f >= 0 and not X[r, f] <= thr[t]:
                lf = 1
            a += C[t, lf, 0]
            b += C[t, lf, 1]
        out[r] = -(a / W) + b / W


try:  # numba in the project env; plain Python elsewhere (slow, fine for small checks)
    import numba as _nb
    _prange = _nb.prange
    _sort3 = _nb.njit(cache=True, nogil=True)(_sort3)
    _round_chain = _nb.njit(cache=True, nogil=True)(_round_chain)
    _first_orders = _nb.njit(cache=True, nogil=True, parallel=True)(_first_orders)
    _root_sums = _nb.njit(cache=True, nogil=True)(_root_sums)
    _scan_features = _nb.njit(cache=True, nogil=True, parallel=True)(_scan_features)
    _partition = _nb.njit(cache=True, nogil=True)(_partition)
    _decide = _nb.njit(cache=True, nogil=True, parallel=True)(_decide)
except ImportError:  # pragma: no cover
    _nb = None
    _prange = range


def _our_rand_r(state: int) -> tuple[int, int]:
    """sklearn/utils/_random.pxd ``our_rand_r`` (xorshift on a uint32 state)."""
    if state == 0:
        state = 1
    state ^= (state << 13) & 0xFFFFFFFF
    state ^= state >> 17
    state ^= (state << 5) & 0xFFFFFFFF
    return state, state % (RAND_R_MAX + 1)


def tree_draws(tree_seed: int, constant: np.ndarray) -> tuple[list[int], list[int]]:
    """(features in the order ``node_split_best`` draws and sorts them, constant ones included;
    the non-constant ones = the order their splits are evaluated) at the root of
    ``DecisionTreeClassifier(random_state=tree_seed)`` with max_features = all: the Splitter's
    ``rand_r_state = RandomState(tree_seed).randint(0, RAND_R_MAX)``, then the Fisher-Yates draws
    ``rand_int(n_drawn_constants, f_i - n_found_constants)`` (no known constants at the root)."""
    state = int(np.random.RandomState(tree_seed).randint(0, RAND_R_MAX))
    F = len(constant)
    features = list(range(F))
    f_i, n_visited, n_found, n_drawn, n_total = F, 0, 0, 0, 0
    draws, visit = [], []
    while f_i > n_total and (n_visited < F or n_visited <= n_found + n_drawn):
        n_visited += 1
        state, r = _our_rand_r(state)
        f_j = n_drawn + r % (f_i - n_found - n_drawn) + n_found
        feat = features[f_j]
        draws.append(feat)
        if constant[feat]:
            features[f_j], features[n_total] = features[n_total], features[f_j]
            n_found += 1
            n_total += 1
            continue
        f_i -= 1
        features[f_i], features[f_j] = features[f_j], features[f_i]
        visit.append(feat)
    return draws, visit


def visit_order(tree_seed: int, constant: np.ndarray) -> list[int]:
    """Order in which the non-constant features' splits are evaluated (see ``tree_draws``)."""
    return tree_draws(tree_seed, constant)[1]


class Stump(NamedTuple):
    """A fitted depth-1 tree: ``feature`` (-1 = the root is a leaf), ``threshold`` (go left iff
    ``x <= threshold``), ``value[leaf] = P(class)`` (left = 0, right = 1; the tree's ``value``)."""
    feature: int
    threshold: float
    value: np.ndarray


class Presorted:
    """Per-fit data of the stump search: X column-major (float32), the sorted values of every
    feature, constant flags (the tree's ``max - min <= FEATURE_THRESHOLD``), the first-draw orders,
    and a thread pool that computes the tree sample orders of the next ``threads`` boosting rounds
    while the current round runs its weighted scans (the orders do not depend on the weights)."""

    def __init__(self, X32: np.ndarray, yidx: np.ndarray, threads: int = 8, n_rounds: int | None = None):
        n, F = X32.shape
        self.n, self.F, self.threads, self.n_rounds = n, F, max(1, threads), n_rounds
        self.Xc = np.ascontiguousarray(X32.T)
        self.y = np.ascontiguousarray(yidx, np.int8)
        self.xs = np.sort(self.Xc, axis=1)
        self.constant = np.array([self.xs[f, -1] <= np.float32(self.xs[f, 0] + FEATURE_THRESHOLD)
                                  for f in range(F)], bool)
        self.feats = np.flatnonzero(~self.constant).astype(np.int64)
        self.slot = np.full(F, -1, np.int64)
        self.slot[self.feats] = np.arange(len(self.feats))
        self.idx_dtype = np.int32 if n < 2**31 else np.int64
        self.first = np.empty((F, n), self.idx_dtype)
        _first_orders(self.Xc, self.first)
        self._pool = None
        self._fut: dict[int, tuple] = {}
        self._ahead, self._next = None, 0

    def _submit(self, r: int, seed: int) -> None:
        draws, visit = tree_draws(seed, self.constant)
        orders = np.empty((len(self.feats), self.n), self.idx_dtype)
        final = np.empty(self.n, self.idx_dtype)
        fut = self._pool.submit(_round_chain, self.Xc, self.first, np.asarray(draws, np.int64), self.slot,
                                orders, final)
        self._fut[r] = (seed, fut, orders, final, visit)

    def round_data(self, it: int, tree_seed: int, rs: np.random.RandomState):
        """(orders, final, visit) of boosting round ``it`` (tree seed ``tree_seed``). Seeds of later
        rounds are drawn from a copy of ``rs`` (the AdaBoost loop's RandomState after this round's
        draw; the loop draws exactly one seed per round, checked when the round is consumed)."""
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(self.threads)
        if it not in self._fut:              # first call: start the look-ahead here
            self.close(shutdown=False)
            self._submit(it, tree_seed)
            self._ahead = np.random.RandomState()
            self._ahead.set_state(rs.get_state())
            self._next = it + 1
        last = it + self.threads if self.n_rounds is None else min(it + self.threads, self.n_rounds - 1)
        while self._next <= last:
            self._submit(self._next, int(self._ahead.randint(np.iinfo(np.int32).max)))
            self._next += 1
        seed, fut, orders, final, visit = self._fut.pop(it)
        fut.result()
        if seed != tree_seed:
            raise RuntimeError("tree seed sequence out of sync with the AdaBoost RandomState")
        return orders, final, visit

    def close(self, shutdown: bool = True) -> None:
        for _, fut, *_ in self._fut.values():
            fut.cancel()
        for _, fut, *_ in self._fut.values():
            if not fut.cancelled():
                fut.result()
        self._fut.clear()
        if shutdown and self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


def _fast_stump(pre: Presorted, w: np.ndarray, it: int, tree_seed: int, rs) -> tuple[Stump, np.ndarray]:
    """``DecisionTreeClassifier(max_depth=1, random_state=tree_seed).fit(X, y, sample_weight=w)``
    (scikit-learn 1.9, gini, best splitter, min_samples_leaf=1, all features), bit for bit.
    Returns the stump and the leaf index of every training row."""
    n = len(w)
    if not (w > 0).all():   # AdaBoost clips weights at eps, so a zero weight is never passed
        raise ValueError("fast stumps need strictly positive sample weights")
    st0, st1, wn = _root_sums(pre.y, w)
    imp = 1.0 - (st0 * st0 + st1 * st1) / (wn * wn)          # Gini.node_impurity
    root = np.array([[st0 / wn, st1 / wn]])
    leaf_only = (Stump(-1, -2.0, np.repeat(root, 2, 0)), np.zeros(n, np.uint8))
    if n < 2 or imp <= TREE_EPSILON:                         # min_samples_split, pure node
        return leaf_only
    orders, final, visit = pre.round_data(it, tree_seed, rs)
    best = np.empty(len(pre.feats))
    pos = np.empty(len(pre.feats), np.int64)
    _scan_features(orders, pre.xs, pre.y, w, pre.feats, pre.slot, st0, st1, wn, best, pos)
    bf, bv = -1, -np.inf
    for f in visit:                                          # global first maximum, strict >
        if best[pre.slot[f]] > bv:
            bv, bf = best[pre.slot[f]], f
    if bf < 0:                                               # no valid split -> leaf
        return leaf_only
    p = int(pos[pre.slot[bf]])
    xs = pre.xs[bf]
    thr = float(xs[p - 1]) / 2.0 + float(xs[p]) / 2.0
    if thr == float(xs[p]) or np.isinf(thr):
        thr = float(xs[p - 1])
    leaf = np.empty(n, np.uint8)
    ps, s, wc, upd = _partition(pre.Xc, bf, thr, final.copy(), pre.y, w, p, st0, st1, wn, leaf)
    if ps != p:
        raise RuntimeError("partition does not match the split position")
    sl0, sl1, wl = upd
    wr, sr0, sr1 = wn - wl, st0 - sl0, st1 - sl1
    gl = 1.0 - (sl0 * sl0 + sl1 * sl1) / (wl * wl)
    gr = 1.0 - (sr0 * sr0 + sr1 * sr1) / (wr * wr)
    improvement = (wn / wn) * (imp - (wr / wn * gr) - (wl / wn * gl))
    if improvement + TREE_EPSILON < 0.0:                     # min_impurity_decrease = 0
        return leaf_only
    return Stump(bf, thr, s / wc[:, None]), leaf


def stump_from_tree(est: DecisionTreeClassifier) -> Stump:
    """The ``Stump`` of a fitted ``DecisionTreeClassifier(max_depth=1)`` (for checks)."""
    t = est.tree_
    if t.node_count == 1:
        return Stump(-1, -2.0, np.repeat(t.value[0, 0][None], 2, 0).astype(np.float64))
    return Stump(int(t.feature[0]), float(t.threshold[0]),
                 np.stack([t.value[t.children_left[0], 0], t.value[t.children_right[0], 0]]).astype(np.float64))


class AdaBoostStumps:
    def __init__(self, n_estimators: int = 100, learning_rate: float = 1.0, algorithm: str = "SAMME.R",
                 random_state: int | None = None, max_depth: int = 1, stumps: str = "fast", n_threads: int = 8):
        if algorithm not in ALGORITHMS:
            raise ValueError(algorithm)
        if stumps not in STUMPS:
            raise ValueError(stumps)
        if stumps == "fast" and max_depth != 1:
            raise ValueError("stumps='fast' implements max_depth=1 only")
        self.n_estimators, self.learning_rate = n_estimators, learning_rate
        self.algorithm, self.random_state, self.max_depth = algorithm, random_state, max_depth
        self.stumps, self.n_threads = stumps, n_threads

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostStumps":
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.asarray(y)
        n = len(y)
        sw = np.ones(n, dtype=np.float64)
        sw /= sw.sum()
        self.estimators_: list = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)
        rs = np.random.RandomState(self.random_state) if not isinstance(self.random_state, np.random.RandomState) \
            else self.random_state
        eps = np.finfo(sw.dtype).eps
        zero_mask = sw == 0.0
        if self.stumps == "fast":
            classes, yidx = np.unique(y, return_inverse=True)
            if len(classes) != 2:
                raise ValueError("stumps='fast' supports binary y only")
            self.classes_, self.n_classes_ = classes, 2
            threads = None
            if _nb is not None:
                threads = _nb.get_num_threads()
                _nb.set_num_threads(max(1, min(self.n_threads, _nb.config.NUMBA_NUM_THREADS)))
            try:
                self._pre = Presorted(X, yidx.astype(np.int8), threads=self.n_threads, n_rounds=self.n_estimators)
                self._yidx = yidx
                boost = self._boost_real_fast if self.algorithm == "SAMME.R" else self._boost_discrete_fast
                self._fit_loop(boost, X, y, sw, rs, eps, zero_mask)
            finally:
                if getattr(self, "_pre", None) is not None:
                    self._pre.close()
                for a in ("_pre", "_yidx", "_sw_now", "_it"):      # per-fit caches, not part of the model
                    self.__dict__.pop(a, None)
                if threads is not None:
                    _nb.set_num_threads(threads)
            return self
        boost = self._boost_real if self.algorithm == "SAMME.R" else self._boost_discrete
        return self._fit_loop(boost, X, y, sw, rs, eps, zero_mask)

    def _fit_loop(self, boost, X, y, sw, rs, eps, zero_mask):
        for it in range(self.n_estimators):
            sw = np.clip(sw, a_min=eps, a_max=None)
            sw[zero_mask] = 0.0
            sw, w, err = boost(it, X, y, sw, rs)
            if sw is None:
                break
            self.estimator_weights_[it] = w
            self.estimator_errors_[it] = err
            if err == 0:
                break
            s = np.sum(sw)
            if not np.isfinite(s) or s <= 0:
                break
            if it < self.n_estimators - 1:
                sw /= s
        return self

    def _make(self, rs: np.random.RandomState) -> DecisionTreeClassifier:
        est = DecisionTreeClassifier(max_depth=self.max_depth, random_state=rs.randint(np.iinfo(np.int32).max))
        self.estimators_.append(est)
        return est

    def _boost_real(self, it, X, y, sw, rs):
        est = self._make(rs)
        est.fit(X, y, sample_weight=sw)
        proba = est.predict_proba(X)
        if it == 0:
            self.classes_ = est.classes_
            self.n_classes_ = len(self.classes_)
        pred = self.classes_.take(np.argmax(proba, axis=1), axis=0)
        incorrect = pred != y
        err = np.mean(np.average(incorrect, weights=sw, axis=0))
        if err <= 0:
            return sw, 1.0, 0.0
        K = self.n_classes_
        y_codes = np.array([-1.0 / (K - 1), 1.0])
        y_coding = y_codes.take(self.classes_ == y[:, np.newaxis])
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)
        ew = -1.0 * self.learning_rate * ((K - 1.0) / K) * xlogy(y_coding, proba).sum(axis=1)
        if not it == self.n_estimators - 1:
            sw *= np.exp(ew * ((sw > 0) | (ew < 0)))
        return sw, 1.0, err

    def _boost_discrete(self, it, X, y, sw, rs):
        est = self._make(rs)
        est.fit(X, y, sample_weight=sw)
        pred = est.predict(X)
        if it == 0:
            self.classes_ = est.classes_
            self.n_classes_ = len(self.classes_)
        incorrect = pred != y
        err = np.mean(np.average(incorrect, weights=sw, axis=0))
        if err <= 0:
            return sw, 1.0, 0.0
        K = self.n_classes_
        if err >= 1.0 - (1.0 / K):
            self.estimators_.pop(-1)
            if not self.estimators_:
                raise ValueError("weak learner worse than random; ensemble cannot be fit")
            return None, None, None
        w = self.learning_rate * (np.log((1.0 - err) / err) + np.log(K - 1.0))
        if not it == self.n_estimators - 1:
            sw = np.exp(np.log(sw) + w * incorrect * (sw > 0))
        return sw, w, err

    # Fast path: the same arithmetic as _boost_real/_boost_discrete, with predict_proba(X) = V[leaf].
    # Row-wise quantities depend only on (leaf, class), so they are computed on the 2 x 2 table with
    # the same elementwise operations and then gathered (bit-identical to the full-array version).
    def _fast_fit_stump(self, rs):
        st, leaf = _fast_stump(self._pre, self._sw_now, self._it, rs.randint(np.iinfo(np.int32).max), rs)
        self.estimators_.append(st)
        pred_leaf = self.classes_.take(np.argmax(st.value, axis=1), axis=0)
        return st, leaf, pred_leaf

    def _boost_real_fast(self, it, X, y, sw, rs):
        self._sw_now, self._it = sw, it
        st, leaf, pred_leaf = self._fast_fit_stump(rs)
        incorrect = pred_leaf[leaf] != y
        err = np.mean(np.average(incorrect, weights=sw, axis=0))
        if err <= 0:
            return sw, 1.0, 0.0
        K = self.n_classes_
        y_codes = np.array([-1.0 / (K - 1), 1.0])
        coding = y_codes.take(self.classes_ == self.classes_[:, np.newaxis])   # [class of y, k]
        proba = st.value.copy()                                               # [leaf, k]
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)
        tab = -1.0 * self.learning_rate * ((K - 1.0) / K) * xlogy(coding[None, :, :], proba[:, None, :]).sum(axis=2)
        ew = tab[leaf, self._yidx]
        if not it == self.n_estimators - 1:
            sw *= np.exp(ew * ((sw > 0) | (ew < 0)))
        return sw, 1.0, err

    def _boost_discrete_fast(self, it, X, y, sw, rs):
        self._sw_now, self._it = sw, it
        st, leaf, pred_leaf = self._fast_fit_stump(rs)
        incorrect = pred_leaf[leaf] != y
        err = np.mean(np.average(incorrect, weights=sw, axis=0))
        if err <= 0:
            return sw, 1.0, 0.0
        K = self.n_classes_
        if err >= 1.0 - (1.0 / K):
            self.estimators_.pop(-1)
            if not self.estimators_:
                raise ValueError("weak learner worse than random; ensemble cannot be fit")
            return None, None, None
        w = self.learning_rate * (np.log((1.0 - err) / err) + np.log(K - 1.0))
        if not it == self.n_estimators - 1:
            sw = np.exp(np.log(sw) + w * incorrect * (sw > 0))
        return sw, w, err

    # ------------------------------------------------------------------ predict
    def decision_function(self, X: np.ndarray, chunk: int = 200_000) -> np.ndarray:
        """Binary decision value (> 0 -> class 1), chunked over rows to bound memory."""
        X = np.ascontiguousarray(X, dtype=np.float32)
        out = np.empty(len(X), np.float64)
        if self.estimators_ and isinstance(self.estimators_[0], Stump):
            self._decision_stumps(X, out)
            return out
        for a in range(0, len(X), chunk):
            out[a:a + chunk] = self._decision(X[a:a + chunk])
        return out

    def stump_contributions(self) -> np.ndarray:
        """[T, leaf, class] decision contribution of each stump, computed with the formulas of
        ``_decision`` on the 2-row leaf-value table (rows of predict_proba are leaf values)."""
        K = self.n_classes_
        if K != 2:
            raise ValueError("binary only")
        classes = self.classes_[:, np.newaxis]
        out = []
        for st, w in zip(self.estimators_, self.estimator_weights_):
            if self.algorithm == "SAMME.R":
                p = st.value.copy()
                np.clip(p, np.finfo(p.dtype).eps, None, out=p)
                lp = np.log(p)
                out.append((K - 1) * (lp - (1.0 / K) * lp.sum(axis=1)[:, np.newaxis]))
            else:
                pred = self.classes_.take(np.argmax(st.value, axis=1), axis=0)
                out.append(np.where((pred == classes).T, w, -1 / (K - 1) * w))
        return np.asarray(out, np.float64).reshape(len(out), 2, 2)

    def _decision_stumps(self, X, out):
        feat = np.array([st.feature for st in self.estimators_], np.int64)
        thr = np.array([st.threshold for st in self.estimators_], np.float64)
        C = np.ascontiguousarray(self.stump_contributions())
        W = float(self.estimator_weights_.sum())
        threads = None
        if _nb is not None:
            threads = _nb.get_num_threads()
            _nb.set_num_threads(max(1, min(self.n_threads, _nb.config.NUMBA_NUM_THREADS)))
        try:
            _decide(X, feat, thr, C, W, out)
        finally:
            if threads is not None:
                _nb.set_num_threads(threads)

    def _decision(self, X):
        K = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        if self.algorithm == "SAMME.R":
            pred = 0
            for est in self.estimators_:
                p = est.predict_proba(X)
                np.clip(p, np.finfo(p.dtype).eps, None, out=p)
                lp = np.log(p)
                pred = pred + (K - 1) * (lp - (1.0 / K) * lp.sum(axis=1)[:, np.newaxis])
        else:
            pred = sum(np.where((est.predict(X) == classes).T, w, -1 / (K - 1) * w)
                       for est, w in zip(self.estimators_, self.estimator_weights_))
        pred = pred / self.estimator_weights_.sum()
        if K != 2:
            raise ValueError("binary only")
        pred[:, 0] *= -1
        return pred.sum(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """P(class 1) = softmax([-d/2, d/2])[1] = sigmoid(d) (sklearn's binary rule)."""
        return expit(self.decision_function(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_.take((self.decision_function(X) > 0).astype(int), axis=0)
