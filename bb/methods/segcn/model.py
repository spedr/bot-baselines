"""SEGCN model (spec section 4): typed input encoders -> L GNN-AK layers with a GCN kernel -> MLP head.

    h0      = [LReLU(Lin(768,q)(tweet)), LReLU(Lin(768,q)(des)), LReLU(Lin(k,q)(num)), LReLU(Lin(c,q)(cat))]
    h_{l+1} = h_l + Dropout(ReLU(BN(Kernel_l(h_l))))                       (GNN-AK outer layer, res=True)
    Kernel_l(u): x = h_l[V_u] (lifted rows), hop embedding concat (optional)
                 x = x + Dropout(ReLU(BN(GCN_sub(x))))  x L_in                (GNN-AK ``GNN``)
                 centroid = x[root]; sub = POOL_{V_u} ReLU(BN(Lin(Dropout(x))))
                 Kernel_l(u) = Lin(2H -> H, bias=False)(Dropout([centroid, sub]))
    out     = Lin(H,2)(ReLU(BN(Lin(H,H,bias=False)(Dropout(h_L)))))

GCN_sub is GCNConv (glorot weight, no bias) with symmetric normalisation and self-loops computed
INSIDE each subgraph. Subgraphs of a batch are packed block-diagonally; aggregation is one sparse
CSR matmul per inner layer (no per-edge message tensors).

``n_inner = 0`` is the released code's degenerate path: a plain GCN over the global graph with the
same BN/ReLU/dropout/residual (full graph only; sanity check, spec 11.1).

Mini-batch plan for L layers and roots R: S_L = R, S_{l-1} = unique ids of the lifted rows of S_l's
subgraphs. h_0 is computed for S_0 only; layer l maps h on S_{l-1} to h on S_l (S_l is a subset of
S_{l-1} because each root is in its own subgraph). Batch-norm statistics are taken over the rows of
each level (GNN-AK lifted-row semantics, spec pitfall 4).

Batching (review round 2): the normalised block adjacency of every root is precomputed once per
store (``subgraphs.block_entries``); a batch plan (lifted rows, block CSR, per-level dedup) is built on
the CPU by one GIL-free numba call (``subgraphs.plan_kernel``), shipped in two pinned copies and
prefetched in a worker thread (``Prefetcher``), so the GPU stream only runs the model. The SpMM
backward reuses the (exactly symmetric) adjacency (``sym_spmm``); ``--fuse`` compiles the per-row
BN/ReLU/dropout/residual chains with inductor (``set_fuse``).
"""
from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as Fn
from torch import nn

from ...gnn.model import _text_proj
from .subgraphs import PLAN_FLOAT_FIELDS, PLAN_INT_FIELDS

PLAN_FIELDS = PLAN_INT_FIELDS + PLAN_FLOAT_FIELDS        # field order of subgraphs.plan_kernel spans

HOP_VOCAB = 20
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")


# ------------------------------------------------------------------------------------------------
# GPU subgraph store and block-diagonal batches


def normalized_adjacency(n_rows: int, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Sparse CSR  D^-1/2 (A + I) D^-1/2  over ``n_rows`` rows for undirected pairs (a, b) (global
    graph of the ``n_inner = 0`` path; exactly symmetric values)."""
    dev = a.device
    deg = torch.ones(n_rows, device=dev)
    one = torch.ones(len(a), device=dev)
    deg.index_add_(0, a, one).index_add_(0, b, one)
    dinv = deg.rsqrt()
    ar = torch.arange(n_rows, device=dev)
    rows = torch.cat([a, b, ar])
    cols = torch.cat([b, a, ar])
    w = dinv[a] * dinv[b]
    vals = torch.cat([w, w, dinv * dinv])
    order = torch.argsort(rows * n_rows + cols)
    crow = torch.zeros(n_rows + 1, dtype=torch.long, device=dev)
    crow[1:] = torch.cumsum(torch.bincount(rows, minlength=n_rows), 0)
    return torch.sparse_csr_tensor(crow, cols[order], vals[order], (n_rows, n_rows))


class _SymSpMM(torch.autograd.Function):
    """``adj @ x`` for an exactly symmetric sparse CSR ``adj``: the backward is ``adj @ grad`` (the
    same non-transposed SpMM) instead of autograd's transposed cuSPARSE SpMM (slower and
    nondeterministic). Same function and gradient (adj^T = adj bitwise)."""

    @staticmethod
    def forward(ctx, adj, x):
        ctx.adj = adj
        return adj @ x

    @staticmethod
    def backward(ctx, grad):
        return None, ctx.adj @ grad


def sym_spmm(adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return _SymSpMM.apply(adj, x) if torch.is_grad_enabled() and x.requires_grad else adj @ x


@dataclass
class Lifted:
    """Block-diagonal batch of the subgraphs of ``roots``."""
    roots: torch.Tensor      # [S] global ids of the subgraph roots
    ids: torch.Tensor        # [T] global id of every lifted row
    seg: torch.Tensor        # [T] subgraph index of every row
    root_pos: torch.Tensor   # [S] row of each root
    hop: torch.Tensor        # [T] BFS distance from the root
    adj: torch.Tensor        # sparse CSR [T, T] normalised adjacency (self-loops, per-subgraph degrees)
    cnt: torch.Tensor        # [S] float subgraph sizes (mean pooling)
    n_rows: int
    inv: torch.Tensor | None = None       # [T] position of ids in the previous level's node set
    self_pos: torch.Tensor | None = None  # [S] position of roots in the previous level's node set


@dataclass
class HostPlan:
    """A batch plan assembled on the CPU: one int32 and one float32 buffer (pinned on CUDA) plus the
    (start, length) span of every field of every level (``subgraphs.plan_kernel``)."""
    ints: torch.Tensor
    floats: torch.Tensor
    spans: np.ndarray        # [levels, 11, 2], levels bottom-up
    n_s0: int                # S_0 = ints[:n_s0]
    rows: np.ndarray         # lifted rows per level (bottom-up)
    dedup: bool

    @property
    def n_rows(self) -> int:
        return int(self.rows.sum())


class GPUStore:
    """Subgraph store used for batching (review round 2: CPU-side planner).

    Holds the store arrays in host RAM plus, per root, the rows of its normalised block adjacency
    (``subgraphs.block_entries``, computed once). ``plan_host`` assembles a batch with numba
    (lifted rows, block CSR, per-level dedup) and packs it into one int32 and one float32 buffer;
    ``to_device`` ships both with two (non-blocking, pinned) copies and slices them into ``Lifted``
    levels. No per-batch index kernels or host syncs remain on the GPU stream, and the plan of the
    next batch can be built in a background thread while the GPU trains on the current one.
    """

    def __init__(self, st, device):
        from .subgraphs import block_entries

        self.node_ptr = np.ascontiguousarray(st.node_ptr, dtype=np.int64)
        self.node_ids = np.ascontiguousarray(st.node_ids, dtype=np.int32)
        self.hop = np.ascontiguousarray(st.hop, dtype=np.uint8)
        edge_ptr = np.ascontiguousarray(st.edge_ptr, dtype=np.int64)
        pairs = np.ascontiguousarray(np.asarray(st.edge_pairs).reshape(-1, 2), dtype=np.uint8)
        self.n = len(self.node_ptr) - 1
        self.sizes_np = np.diff(self.node_ptr)
        self.ent_ptr = np.zeros(self.n + 1, np.int64)
        np.cumsum(self.sizes_np + 2 * np.diff(edge_ptr), out=self.ent_ptr[1:])
        self.ent_col = np.empty(int(self.ent_ptr[-1]), np.uint8)
        self.ent_val = np.empty(int(self.ent_ptr[-1]), np.float32)
        self.rnnz = np.empty(len(self.node_ids), np.uint8)
        block_entries(self.node_ptr, edge_ptr, pairs, self.ent_ptr, self.ent_col, self.ent_val, self.rnnz)
        self.device = torch.device(device)
        self._pin = self.device.type == "cuda"
        self._tls = threading.local()

    def nbytes(self) -> int:
        """Host bytes of the batching arrays."""
        return sum(x.nbytes for x in (self.node_ptr, self.node_ids, self.hop, self.ent_ptr, self.ent_col,
                                      self.ent_val, self.rnnz))

    def sizes(self) -> torch.Tensor:
        return torch.from_numpy(self.sizes_np)

    # -- host side
    def _scratch(self):
        s = getattr(self._tls, "s", None)
        if s is None:
            s = self._tls.s = [np.zeros(self.n, np.int32), np.empty(self.n, np.int32), 0]
        return s

    def plan_host(self, roots, n_layers: int, *, lift_only: bool = False) -> HostPlan:
        """Plan for top-level ``roots``: levels S_L = roots, S_{l-1} = unique ids of S_l's lifted rows.
        ``lift_only``: one level, no dedup (layer-wise evaluation). One numba call (GIL released), so
        it overlaps with the GPU when run in a worker thread."""
        from .subgraphs import plan_kernel
        r = np.ascontiguousarray(roots.cpu().numpy() if torch.is_tensor(roots) else roots, dtype=np.int64)
        sc = self._scratch()
        ib, fb, spans, n_s0, rows, sc[2] = plan_kernel(
            r, int(n_layers), not lift_only, self.node_ptr, self.node_ids, self.hop, self.ent_ptr, self.ent_col,
            self.ent_val, self.rnnz, sc[0], sc[1], sc[2])
        ib, fb = torch.from_numpy(ib), torch.from_numpy(fb)
        if self._pin:
            ib, fb = ib.pin_memory(), fb.pin_memory()
        return HostPlan(ib, fb, spans, int(n_s0), rows, not lift_only)

    # -- device side
    def to_device(self, hp: HostPlan) -> tuple[torch.Tensor, list[Lifted]]:
        nb_ = self._pin
        I = hp.ints.to(self.device, non_blocking=nb_).long()
        Fl = hp.floats.to(self.device, non_blocking=nb_)
        out = []
        for j in range(len(hp.rows)):
            sp = hp.spans[j]
            f = {name: (I if k < 9 else Fl)[int(sp[k, 0]):int(sp[k, 0] + sp[k, 1])]
                 for k, name in enumerate(PLAN_FIELDS)}
            t = int(hp.rows[j])
            adj = torch.sparse_csr_tensor(f["crow"], f["col"], f["val"], (t, t))
            out.append(Lifted(f["roots"], f["ids"], f["seg"], f["root_pos"], f["hop"], adj, f["cnt"], t,
                              f["inv"] if hp.dedup else None, f["self_pos"] if hp.dedup else None))
        return I[:hp.n_s0], out

    def lift(self, roots) -> Lifted:
        return self.to_device(self.plan_host(roots, 1, lift_only=True))[1][0]

    def plan(self, roots, n_layers: int) -> tuple[torch.Tensor, list[Lifted]]:
        """(S_0, [Lifted for S_1, ..., S_L]) for top-level ``roots``."""
        return self.to_device(self.plan_host(roots, n_layers))

    def all_ids(self, roots: np.ndarray) -> np.ndarray:
        from .subgraphs import gather_ids
        return gather_ids(np.ascontiguousarray(roots, dtype=np.int64), self.node_ptr, self.node_ids)

    def cost(self, n_layers: int) -> torch.Tensor:
        """Upper bound on lifted rows of all levels per root (no dedup), [N] int64 (CPU)."""
        from .subgraphs import level_cost
        c = self.sizes_np.astype(np.int64)
        tot = c.copy()
        for _ in range(n_layers - 1):
            c = level_cost(self.node_ptr, self.node_ids, c)
            tot = tot + c
        return torch.from_numpy(tot)


class Prefetcher:
    """Iterate device plans of ``batches`` while a worker thread builds the next ``depth`` host plans
    (numba releases the GIL; pinned buffers are recycled by torch's caching host allocator only after
    their copies finish)."""

    def __init__(self, store: GPUStore, batches, n_layers: int, depth: int = 2):
        self.store, self.batches, self.L, self.depth = store, batches, n_layers, depth

    def __iter__(self):
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(1) as ex:
            q = deque()
            it = iter(self.batches)
            for b in it:
                q.append((b, ex.submit(self.store.plan_host, b, self.L)))
                if len(q) >= self.depth:
                    break
            while q:
                b, f = q.popleft()
                nxt = next(it, None)
                if nxt is not None:
                    q.append((nxt, ex.submit(self.store.plan_host, nxt, self.L)))
                hp = f.result()
                s0, levels = self.store.to_device(hp)
                yield b, hp, s0, levels


# ------------------------------------------------------------------------------------------------
# modules


def _post_conv(bn: nn.Module, y: torch.Tensor, prev: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Inner GCN layer after the aggregation: prev + Dropout(ReLU(BN(y)))."""
    return prev + Fn.dropout(Fn.relu(bn(y)), p, training)


def _sub_pool(lin: nn.Module, bn: nn.Module, x: torch.Tensor, seg: torch.Tensor, cnt: torch.Tensor, p: float,
              training: bool, mean: bool) -> torch.Tensor:
    """Subgraph embedding: POOL_{V_u} ReLU(BN(Lin(Dropout(x)))) (cnt = subgraph sizes >= 1)."""
    s = Fn.relu(bn(lin(Fn.dropout(x, p, training))))
    pooled = torch.zeros((cnt.shape[0], s.shape[1]), dtype=s.dtype, device=s.device).index_add_(0, seg, s)
    return pooled / cnt[:, None] if mean else pooled


_FUSED: dict = {}


def _fused(name: str):
    """torch.compile'd (inductor, dynamic shapes) versions of the two per-lifted-row chains above:
    BN statistics, BN backward, ReLU, dropout and the residual run as a few fused kernels instead of
    ~20 memory-bound passes over [T, H] (review round 2, ``--fuse``). Same function; inductor draws
    its own dropout masks (same Bernoulli(1 - p) / (1 - p) law, different random stream)."""
    if name not in _FUSED:
        _FUSED[name] = torch.compile({"post": _post_conv, "sub": _sub_pool}[name], dynamic=True)
    return _FUSED[name]


def set_fuse(model: nn.Module, on: bool = True) -> None:
    """Route the inner-GCN / subgraph-pool chains of every SubgraphKernel through ``_fused``."""
    for m in model.modules():
        if isinstance(m, (InnerGCN, SubgraphKernel)):
            m.fuse = on



def _glorot_linear(nin: int, nout: int, bias: bool = False) -> nn.Linear:
    lin = nn.Linear(nin, nout, bias=bias)
    nn.init.xavier_uniform_(lin.weight)          # PyG GCNConv weight init
    if bias:
        nn.init.zeros_(lin.bias)
    return lin


class InnerGCN(nn.Module):
    """GNN-AK ``GNN`` with GCNConv layers (bias=False because of BN), residual, then output encoder."""

    def __init__(self, nin: int, nout: int, nlayer: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList([_glorot_linear(nin, nin) for _ in range(nlayer)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(nin) for _ in range(nlayer)])
        self.output_encoder = nn.Linear(nin, nout) if nin != nout else nn.Identity()
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        post = _fused("post") if getattr(self, "fuse", False) else _post_conv
        for conv, bn in zip(self.convs, self.norms):
            x = post(bn, sym_spmm(adj, conv(x)), x, self.dropout, self.training)   # x + Drop(ReLU(BN(Â x W)))
        return self.output_encoder(x)


class SubgraphKernel(nn.Module):
    """GNN-AK ``SubgraphGNNKernel`` with embs=(centroid, subgraph), combine='concat' (spec 4)."""

    def __init__(self, hidden: int, n_inner: int, dropout: float, *, hop_dim: int = 0, pool: str = "mean",
                 root_extra: int = 0):
        super().__init__()
        self.hop_emb = nn.Embedding(HOP_VOCAB, hop_dim) if hop_dim else None
        self.gnn = InnerGCN(hidden + hop_dim, hidden, n_inner, dropout)
        self.sub_lin = nn.Linear(hidden, hidden, bias=False)
        self.sub_bn = nn.BatchNorm1d(hidden)
        self.out_encoder = nn.Linear(2 * hidden + root_extra, hidden, bias=False)
        self.dropout, self.pool = dropout, pool

    def forward(self, x: torch.Tensor, lv: Lifted, root_extra: torch.Tensor | None = None) -> torch.Tensor:
        if self.hop_emb is not None:
            x = torch.cat([x, self.hop_emb((lv.hop + 1).clamp(max=HOP_VOCAB - 1))], dim=1)
        x = self.gnn(x, lv.adj)
        centroid = x.index_select(0, lv.root_pos)
        sub = _fused("sub") if getattr(self, "fuse", False) else _sub_pool
        pooled = sub(self.sub_lin, self.sub_bn, x, lv.seg, lv.cnt, self.dropout, self.training, self.pool == "mean")
        parts = [centroid, pooled] + ([root_extra] if root_extra is not None else [])
        return self.out_encoder(Fn.dropout(torch.cat(parts, dim=1), self.dropout, self.training))


class GlobalGCN(nn.Module):
    """Degenerate ``n_inner = 0`` layer: GCNConv over the global graph (adjacency passed in)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.conv = _glorot_linear(hidden, hidden)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return sym_spmm(adj, self.conv(x))


class TypedEncoder(nn.Module):
    """Paper Eqs. 1-2: four (five with the ExtScore branch) LReLU(Linear(., q)) blocks, concatenated.
    No extra Linear(4q, 4q) (BotRGCN has one; the paper does not, spec 4 row 0)."""

    def __init__(self, q: int, *, num_dim: int, cat_dim: int, extra_dim: int = 0, des_dim: int = 768,
                 tweet_dim: int = 768, slope: float = 0.01):
        super().__init__()
        self.tweet = nn.Linear(tweet_dim, q)
        self.des = nn.Linear(des_dim, q)
        self.num = nn.Linear(num_dim, q)
        self.cat = nn.Linear(cat_dim, q)
        self.extra = nn.Linear(extra_dim, q) if extra_dim else None
        self.slope = slope

    def forward(self, des, tweet, num, cat, extra=None) -> torch.Tensor:
        a = lambda z: Fn.leaky_relu(z, self.slope)  # noqa: E731
        parts = [a(_text_proj(self.tweet, tweet)), a(_text_proj(self.des, des)), a(self.num(num)), a(self.cat(cat))]
        if self.extra is not None:
            parts.append(a(self.extra(extra)))
        return torch.cat(parts, dim=1)


class SEGCN(nn.Module):
    def __init__(self, *, num_dim: int, cat_dim: int, hidden: int = 128, n_layers: int = 2, n_inner: int = 1,
                 dropout: float = 0.3, hop_dim: int = 0, pool: str = "mean", extscore_width: int = 0,
                 extscore_inject: str = "num", residual: bool = True):
        super().__init__()
        self.residual = residual
        if hidden % 4:
            raise ValueError("hidden must be divisible by 4")
        q = hidden // 4
        branch = extscore_width if extscore_inject == "branch" else 0
        self.extscore_inject = extscore_inject if (extscore_width and extscore_inject != "num") else None
        self.encoder = TypedEncoder(q, num_dim=num_dim, cat_dim=cat_dim, extra_dim=branch)
        H = hidden + (q if branch else 0)                        # spec 8 ablation 2: all H raised to 160
        self.hidden = H
        self.n_layers, self.n_inner, self.dropout = n_layers, n_inner, dropout
        root = extscore_width if self.extscore_inject == "root" else 0
        if n_inner > 0:
            self.layers = nn.ModuleList([SubgraphKernel(H, n_inner, dropout, hop_dim=hop_dim, pool=pool,
                                                        root_extra=root) for _ in range(n_layers)])
        else:
            self.layers = nn.ModuleList([GlobalGCN(H) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(H) for _ in range(n_layers)])
        late = extscore_width if self.extscore_inject == "late" else 0
        self.head1 = nn.Linear(H + late, H, bias=False)
        self.head_bn = nn.BatchNorm1d(H)
        self.head2 = nn.Linear(H, 2)

    @property
    def subgraph(self) -> bool:
        return self.n_inner > 0

    # -- pieces
    def encode(self, g, ids: torch.Tensor | None) -> torch.Tensor:
        if ids is None:
            ex = g.extra if self.extscore_inject == "branch" else None
            return self.encoder(g.des, g.tweet, g.num, g.cat, ex)
        ex = g.extra[ids] if self.extscore_inject == "branch" else None
        return self.encoder(g.des[ids], g.tweet[ids], g.num[ids], g.cat[ids], ex)

    def outer(self, l: int, h_prev_self: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        out = Fn.dropout(Fn.relu(self.norms[l](k)), self.dropout, self.training)
        return h_prev_self + out if self.residual else out

    def kernel(self, l: int, x: torch.Tensor, lv: Lifted, g) -> torch.Tensor:
        rx = g.extra[lv.roots] if self.extscore_inject == "root" else None
        return self.layers[l](x, lv, rx)

    def head(self, h: torch.Tensor, g, ids: torch.Tensor | None) -> torch.Tensor:
        h = Fn.dropout(h, self.dropout, self.training)
        if self.extscore_inject == "late":
            h = torch.cat([h, g.extra if ids is None else g.extra[ids]], dim=1)
        return self.head2(Fn.relu(self.head_bn(self.head1(h))))

    # -- forward passes
    def forward_plan(self, g, s0: torch.Tensor, levels: list[Lifted]) -> torch.Tensor:
        """Mini-batch forward for the plan of ``GPUStore.plan`` -> logits for the top-level roots."""
        h = self.encode(g, s0)
        for l, lv in enumerate(levels):
            # index_select: its backward is an index_add_ (atomic scatter), much cheaper than the
            # sort-based accumulate of advanced indexing (review round 2)
            k = self.kernel(l, h.index_select(0, lv.inv), lv, g)
            h = self.outer(l, h.index_select(0, lv.self_pos), k)
        return self.head(h, g, levels[-1].roots)

    def forward_global(self, g, adj: torch.Tensor) -> torch.Tensor:
        """``n_inner = 0`` path: full graph."""
        h = self.encode(g, None)
        for l in range(self.n_layers):
            h = self.outer(l, h, self.layers[l](h, adj))
        return self.head(h, g, None)

    @torch.no_grad()
    def predict_layerwise(self, g, store: GPUStore, roots: torch.Tensor, *, chunk_rows: int = 2_000_000,
                          needed: list | None = None) -> torch.Tensor:
        """Exact eval-mode logits for ``roots``: layer by layer over the needed node sets, in chunks of
        roots whose lifted rows stay under ``chunk_rows``. ``needed`` = ``eval_sets(store, roots, L)``."""
        dev = roots.device
        if needed is None:
            needed = eval_sets(store, roots, self.n_layers)
        sizes = store.sizes_np
        h = None
        for l in range(self.n_layers):
            src = needed[l]
            if l == 0:
                h_prev = torch.zeros((store.n, self.hidden), device=dev)
                for c in _chunks_by(src, np.ones(len(src), np.int64), 200_000):
                    cd = torch.from_numpy(c).to(dev)
                    h_prev[cd] = self.encode(g, cd)
            else:
                h_prev = h
            tgt = needed[l + 1]
            h = torch.zeros_like(h_prev) if l + 1 < self.n_layers else None
            outs = []
            for c in _chunks_by(tgt, sizes[tgt], chunk_rows):
                lv = store.lift(c)
                k = self.kernel(l, h_prev[lv.ids], lv, g)
                o = self.outer(l, h_prev[lv.roots], k)
                if h is not None:
                    h[lv.roots] = o
                else:
                    outs.append(self.head(o, g, lv.roots))
            if h is None:
                return torch.cat(outs) if outs else torch.zeros((0, 2), device=dev)
            del h_prev
        raise AssertionError("unreachable")


def eval_sets(store: GPUStore, roots, n_layers: int) -> list[np.ndarray]:
    """[S_0, S_1, ..., S_L = roots] (host int64 arrays): node sets needed at each level for exact
    predictions of roots."""
    r = np.asarray(roots.cpu().numpy() if torch.is_tensor(roots) else roots, dtype=np.int64)
    sets = [r]
    cur = r
    for _ in range(n_layers):
        cur = np.union1d(cur, store.all_ids(cur))
        sets.append(cur)
    sets.reverse()
    sets[-1] = r
    return sets


def _chunks_by(idx: np.ndarray, cost: np.ndarray, budget: int):
    """Contiguous chunks of ``idx`` with summed ``cost`` <= budget (at least one element each)."""
    if len(idx) == 0:
        return
    cs = np.cumsum(np.asarray(cost, dtype=np.int64))
    start, base = 0, 0
    n = len(idx)
    while start < n:
        end = int(np.searchsorted(cs, base + budget, side="right"))
        end = max(end, start + 1)
        yield idx[start:end]
        base = int(cs[end - 1])
        start = end


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
