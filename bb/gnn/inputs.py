"""Device-resident node inputs for the ``botrgcn`` feature scheme.

``to_device`` moves a ``BotRGCNFeatures`` set to the GPU (text tables stay float16 through one
pinned staging buffer, see ``bb.harness.gpu.memmap_to_gpu``) and applies the ExtScore column through the
common interface:

- ``extscore_inject="num"`` (default for the botrgcn family): ``J.append(num_prop, kind="neural")`` ->
  ``num`` gets the two columns ``[z, m]`` (width k -> k + 2) before the numeric projection.
- ``"branch"`` / ``"late"``: ``num`` stays as is and the ``[z, m]`` block goes to ``extra`` (the
  model decides where it enters: a fifth input branch, or concatenated before the classifier).

With ``--extscore none`` (``J.enabled`` False) every path returns exactly the ExtScore-free inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

EXTSCORE_INJECT = ("num", "branch", "late")


def _tensor(arr, dtype=None):
    """Owned tensor from a (possibly read-only, memory-mapped) array: ``torch.from_numpy`` on a
    read-only mapping shares it, and an in-place write would then segfault on CPU."""
    import torch
    return torch.from_numpy(np.array(arr, dtype=dtype, copy=True, order="C"))


@dataclass
class GraphInputs:
    des: "torch.Tensor"          # [N, 768] float16 (GPU) / float32 (CPU)
    tweet: "torch.Tensor"        # [N, 768]
    num: "torch.Tensor"          # [N, k (+2 with ExtScore 'num')] float32
    cat: "torch.Tensor"          # [N, c] float32
    edge_index: "torch.Tensor"   # [2, E] int64, source -> target
    edge_type: "torch.Tensor"    # [E] int64
    y: "torch.Tensor"            # [N] int64, -1 = unlabelled
    extra: "torch.Tensor | None" = None   # [N, 2] ExtScore block for 'branch'/'late' injection

    @property
    def n(self) -> int:
        return int(self.y.shape[0])

    @property
    def device(self):
        return self.y.device

    def nodes(self, n_id) -> "GraphInputs":
        """Rows ``n_id`` of every node tensor (edges left empty; set them with ``with_edges``)."""
        import torch
        e = torch.zeros((2, 0), dtype=torch.long, device=self.device)
        return GraphInputs(self.des[n_id], self.tweet[n_id], self.num[n_id], self.cat[n_id], e,
                           torch.zeros(0, dtype=torch.long, device=self.device), self.y[n_id],
                           None if self.extra is None else self.extra[n_id])

    def with_edges(self, edge_index, edge_type) -> "GraphInputs":
        return replace(self, edge_index=edge_index, edge_type=edge_type)

    def nbytes(self) -> int:
        ts = [self.des, self.tweet, self.num, self.cat, self.edge_index, self.edge_type, self.y, self.extra]
        return int(sum(t.numel() * t.element_size() for t in ts if t is not None))


def table_to_device(path, *, dtype=None, device="cuda", chunk_rows: int = 50_000):
    """Copy a whole [N, D] .npy to one device tensor with plain reads (``bb.io.read_rows``: pread +
    POSIX_FADV_DONTNEED) through one pinned staging buffer. Unlike slicing a memory map, no file
    pages stay mapped or cached, so host RSS stays at one chunk (TwiBot-22 des: 1.5 GB moved through
    77 MB)."""
    import torch

    from ..io import read_rows

    hdr = np.load(path, mmap_mode="r")
    n, tail, src = hdr.shape[0], tuple(hdr.shape[1:]), hdr.dtype
    del hdr
    src_t = torch.from_numpy(np.zeros(0, dtype=src)).dtype
    dev = torch.device(device)
    out = torch.empty((n, *tail), dtype=dtype or src_t, device=dev)
    stage = torch.empty((min(chunk_rows, max(n, 1)), *tail), dtype=src_t, pin_memory=dev.type == "cuda")
    for s in range(0, n, chunk_rows):
        blk = read_rows(path, s, min(n, s + chunk_rows))
        st = stage[: len(blk)]
        st.copy_(torch.from_numpy(blk))
        out[s:s + len(blk)].copy_(st)
    del stage
    return out


def to_device(F, device: str = "cuda", *, J=None, extscore_inject: str = "num", text_dtype=None,
              edges: bool = True, placeholder_text: bool = False) -> GraphInputs:
    """Build ``GraphInputs`` from a ``BotRGCNFeatures`` (``bb.features.load_features``).

    ``text_dtype`` defaults to float16 on CUDA (spec 7.2: halves the 3 GB TwiBot-22 tables) and
    float32 on CPU. ``edges=False`` skips the edge tensors (sampled training keeps them on the host).
    ``placeholder_text=True`` puts zero tensors of the right shape in place of text tables that are
    not built yet: for time/VRAM measurements only, never for results.
    """
    import torch

    from ..harness.gpu import memmap_to_gpu

    if extscore_inject not in EXTSCORE_INJECT:
        raise ValueError(f"extscore_inject must be one of {EXTSCORE_INJECT}")
    if (F.des is None or F.tweets is None) and not placeholder_text:
        raise FileNotFoundError(f"{F.dataset}/{F.variant}: text tables not built")
    dev = torch.device(device)
    if text_dtype is None:
        text_dtype = torch.float16 if dev.type == "cuda" else torch.float32

    def whole_file(arr) -> bool:
        """``arr`` maps a complete .npy (not a view): np.load's memmap has ``offset`` = header length,
        never 0, so compare with a fresh header read (review round 1: the old ``offset == 0`` test was
        always False and the pread path was never taken)."""
        fn = getattr(arr, "filename", None)
        if not fn or len(arr) != F.n or not arr.flags.c_contiguous:
            return False
        h = np.load(fn, mmap_mode="r")
        return h.shape == arr.shape and h.dtype == arr.dtype and h.offset == arr.offset

    def text(arr):
        if arr is None:
            return torch.zeros((F.n, 768), dtype=text_dtype, device=dev)
        if whole_file(arr):
            return table_to_device(arr.filename, dtype=text_dtype, device=dev)
        return memmap_to_gpu(arr, dtype=text_dtype, device=str(dev))

    des, tweet = text(F.des), text(F.tweets)
    num = np.asarray(F.num_prop, dtype=np.float32)
    extra = None
    if J is not None and J.enabled:
        if extscore_inject == "num":
            num = J.append(num, kind="neural")                                   # [N, k+2]
        else:
            extra = _tensor(J.append(np.zeros((F.n, 0), np.float32), kind="neural")).to(dev)  # [N, 2]
    if edges:
        ei = _tensor(F.edge_index, np.int64).to(dev)
        et = _tensor(F.edge_type, np.int64).to(dev)
    else:
        ei = torch.zeros((2, 0), dtype=torch.long, device=dev)
        et = torch.zeros(0, dtype=torch.long, device=dev)
    return GraphInputs(des, tweet, _tensor(num, np.float32).to(dev), _tensor(F.cat_prop, np.float32).to(dev),
                       ei, et, _tensor(F.label, np.int64).to(dev), extra)
