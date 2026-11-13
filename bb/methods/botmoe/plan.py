"""Host-side edge bookkeeping for one sampled subgraph (review round 2, efficiency).

Every graph convolution of BotMoE derives index structures from the subgraph's edges: GCNConv adds
self-loops (boolean mask indexing), RGCNConv splits the edges by relation (one boolean mask per
relation and call), the RGT layer needs ``unique`` sources/targets and segment-sorted edges with chunk
boundaries (``.tolist()``). On the GPU each of these is a host-device synchronisation (about 15 per
training step, recomputed again inside checkpointed backward passes), so the CPU cannot run ahead
and the GPU idles while Python launches the next kernels. ``SubgraphPlan`` computes the same integer
structures once per batch in numpy (where the sampler already produced the edges), packs them into
one pinned buffer and copies it with a single ``non_blocking`` transfer. The convolutions then run
without host syncs, so sampling and planning of batch i+1 overlap the GPU work of batch i.

All structures are exact integer re-derivations of what PyG / ``rgt_chunked`` compute on the device
(same edge order: boolean masks keep order, the stable argsort and ``unique`` are the same
permutations), so every float op sees identical inputs. Test: ``test_plan_matches_device_path``.
"""
from __future__ import annotations

import numpy as np
import torch


class SubgraphPlan:
    """Device views of the per-subgraph index structures.

    gcn_ei [2, E' + n]   PyG ``add_remaining_self_loops`` order: non-loop edges, then (i, i) for all i
    gcn_deg [n]          in-degree incl. the self-loop (float32, exact integers) -> ``gcn_norm``
    rgcn_ei[r] [2, E_r]  ``edge_index[:, edge_type == r]``
    rgt_*                ``ChunkedCtx`` fields: S, T (sorted unique sources / targets), q_sorted,
                         k_sorted (edges stably sorted by (target, relation) segment), chunks, n_seg
    ptr_q, perm_k, ptr_k CSR pointers of the segment-sorted edges and the source-row order, for the
                         fused RGT kernels (``edge_triton``)
    """

    def __init__(self, n: int, R: int):
        self.n, self.R = n, R

    # ------------------------------------------------------------------------------------------ build
    @classmethod
    def build(cls, edge_index: np.ndarray, edge_type: np.ndarray, n: int, R: int, max_edges: int = 32768,
              device="cpu", extra: dict | None = None) -> "SubgraphPlan":
        """``edge_index`` [2, E] local (source, target), ``edge_type`` [E]; ``extra``: further int64
        arrays (e.g. ``n_id``) to ship in the same transfer, exposed as attributes."""
        ei = np.ascontiguousarray(edge_index, dtype=np.int64)
        et = np.ascontiguousarray(edge_type, dtype=np.int64)
        src, dst = ei[0], ei[1]
        E = int(et.shape[0])
        # GCN: add_remaining_self_loops + degree over the target index (flow source_to_target)
        keep = src != dst
        loops = np.arange(n, dtype=np.int64)
        gcn_src = np.concatenate([src[keep], loops])
        gcn_dst = np.concatenate([dst[keep], loops])
        gcn_deg = np.bincount(gcn_dst, minlength=n).astype(np.int64)
        # RGCN: per-relation edge lists in edge order
        rgcn = []
        for r in range(R):
            m = et == r
            rgcn += [src[m], dst[m]]
        # RGT (EdgeCtx + ChunkedCtx)
        S, s_loc = np.unique(src, return_inverse=True)
        T, t_loc = np.unique(dst, return_inverse=True)
        k_idx = s_loc.reshape(-1) * R + et
        q_idx = t_loc.reshape(-1) * R + et
        order = np.argsort(q_idx, kind="stable")
        q_sorted, k_sorted = q_idx[order], k_idx[order]
        n_seg = len(T) * R
        chunks = []
        if E:
            starts = np.arange(0, E, max_edges)
            first = np.searchsorted(q_sorted, q_sorted[starts], side="left")
            cuts = sorted(set(first.tolist()) | {E})
            seg_cuts = q_sorted[np.asarray(cuts[:-1], dtype=np.int64)].tolist()
            for j, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
                s_lo = 0 if j == 0 else seg_cuts[j]
                s_hi = seg_cuts[j + 1] if j + 1 < len(seg_cuts) else n_seg
                chunks.append((a, b, s_lo, s_hi))
        # [2, *] blocks are laid out row-major so they come back as views, without device-side stacking
        # CSR views for the fused RGT kernels (edge_triton): segment pointers of the q-sorted edges,
        # and the edges re-sorted by source row k with its pointers
        nK = len(S) * R
        ptr_q = np.zeros(n_seg + 1, np.int64)
        np.cumsum(np.bincount(q_sorted, minlength=n_seg), out=ptr_q[1:])
        perm_k = np.argsort(k_sorted, kind="stable")
        ptr_k = np.zeros(nK + 1, np.int64)
        np.cumsum(np.bincount(k_sorted, minlength=nK), out=ptr_k[1:])
        parts = {"edge_index": ei.reshape(-1), "edge_type": et, "gcn_ei": np.concatenate([gcn_src, gcn_dst]),
                 "gcn_deg": gcn_deg, "S": S, "T": T, "q_sorted": q_sorted, "k_sorted": k_sorted,
                 "ptr_q": ptr_q, "perm_k": perm_k, "ptr_k": ptr_k}
        for r in range(R):
            parts[f"rgcn_ei{r}"] = np.concatenate([rgcn[2 * r], rgcn[2 * r + 1]])
        for k, v in (extra or {}).items():
            parts[k] = np.ascontiguousarray(v, dtype=np.int64).reshape(-1)
        sizes = [len(v) for v in parts.values()]
        buf = torch.from_numpy(np.concatenate(list(parts.values())) if parts else np.zeros(0, np.int64))
        dev = torch.device(device)
        if dev.type == "cuda":
            buf = buf.pin_memory().to(dev, non_blocking=True)     # one async copy for every index array
        views = dict(zip(parts, torch.split(buf, sizes)))
        p = cls(n, R)
        p.edge_index = views.pop("edge_index").view(2, E)
        p.edge_type = views.pop("edge_type")
        p.gcn_ei = views.pop("gcn_ei").view(2, -1)
        p._gcn_deg = views.pop("gcn_deg")
        p.rgcn_ei = [views.pop(f"rgcn_ei{r}").view(2, -1) for r in range(R)]
        p.S, p.T = views.pop("S"), views.pop("T")
        p.q_sorted, p.k_sorted = views.pop("q_sorted"), views.pop("k_sorted")
        p.ptr_q, p.perm_k, p.ptr_k = views.pop("ptr_q"), views.pop("perm_k"), views.pop("ptr_k")
        p.n_seg, p.chunks, p.max_edges = n_seg, chunks, max_edges
        for k, v in views.items():
            setattr(p, k, v)
        p._gcn_w = None
        return p

    # ------------------------------------------------------------------------------------------- GCN
    def gcn_weight(self) -> torch.Tensor:
        """``gcn_norm`` edge weights (same ops as PyG: deg^-1/2[row] * 1 * deg^-1/2[col]); computed
        once per subgraph and shared by both GCN layers (and the checkpoint recompute)."""
        if self._gcn_w is None:
            deg = self._gcn_deg.to(torch.float32)
            dis = deg.pow_(-0.5)
            dis.masked_fill_(dis == float("inf"), 0)
            row, col = self.gcn_ei[0], self.gcn_ei[1]
            self._gcn_w = dis[row] * dis[col]            # PyG: dis[row] * ones * dis[col] (x * 1 == x)
        return self._gcn_w
