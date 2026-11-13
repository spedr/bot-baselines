"""Text-embedding pass: frozen HF encoder, fp16 autocast, length-bucketed batches, outputs written
straight into fp16 memmaps (row-aligned with idx or with the tweet selection). Resumable per chunk.

Pooling (benchmark recipe, botrgcn.md 3.1.4): last hidden layer, arithmetic mean over ALL token
positions including ``<s>`` and ``</s>``, excluding only padding. Truncation at ``max_tokens``.

Outputs in ``data/<dataset>/emb/<encoder>/``::

    des.f16.npy                         [N,768]   description embedding (null policy per dataset)
    tweets_seq_<sel>.f16.npy            [T,768]   one row per selected tweet (sel = first20 | all | first<k>)
    tweets_seq_<sel>_offsets.i64.npy    [N+1]     user idx -> rows of tweets_seq_<sel>
    tweets_seq_<sel>_rows.i64.npy       [T]       row of each entry in the tweet store
    tweets_first20_mean.f16.npy         [N,768]   mean of the first <= 20 (release order)
    tweets_all_mean.f16.npy             [N,768]   mean of all stored tweets (only when sel = all)
    e_empty.f16.npy                     [768]     embedding of the empty string
    meta.json / <file>.meta.json        encoder name + HF commit hash, pooling, policies, counts

Null / empty policies (botrgcn.md 3.1.4-3.1.5):
- des, twibot22/cresci15: JSON null -> zero vector; '' -> embedded (e_empty).
- des, twibot20 (original Dataset.py): stripped raw string embedded; null/"None " -> literal "None".
- tweet means: twibot22/cresci15 first20 -> null text contributes a zero vector (counted), no tweets
  -> zero vector; twibot20 all -> `tweet: null` user -> e_empty; twibot20 first20 -> zero if none.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..io import NpyRowWriter, load_npy, read_json, read_rows, save_npy, write_json
from ..paths import data_root, dataset_paths
from ..status import StepStatus

os.environ.setdefault("HF_HOME", str(data_root() / "hf"))  # keep model downloads inside the repo
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DIM = 768


def _hf(encoder: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(encoder)
    model = AutoModel.from_pretrained(encoder, attn_implementation="sdpa", add_pooling_layer=False)
    model.eval().cuda()
    commit = getattr(model.config, "_commit_hash", None) or _cached_commit(encoder)
    return tok, model, commit


def _cached_commit(encoder: str) -> str | None:
    """HF commit hash of the cached snapshot (``$HF_HOME/hub/models--<name>/refs/main``)."""
    p = Path(os.environ["HF_HOME"]) / "hub" / f"models--{encoder.replace('/', '--')}" / "refs" / "main"
    return p.read_text().strip() if p.exists() else None


class Encoder:
    """Frozen encoder with benchmark mean pooling. ``encode(texts)`` -> fp16 [len(texts), 768]."""

    def __init__(self, encoder: str = "roberta-base", max_tokens: int = 512, batch_tokens: int = 65536,
                 max_batch: int = 1024, *, model=None, tokenizer=None, device: str = "cuda",
                 amp: bool = True):
        """``model``/``tokenizer`` may be injected (tests use a tiny random model on CPU)."""
        import torch

        self.torch = torch
        self.name = encoder
        if model is None:
            self.tok, self.model, self.commit = _hf(encoder)
        else:
            self.tok, self.model, self.commit = tokenizer, model.eval().to(device), None
        self.device = device
        self.amp = amp and device.startswith("cuda")
        self.dim = int(self.model.config.hidden_size)
        self.max_tokens = max_tokens
        self.batch_tokens = batch_tokens
        self.max_batch = max_batch
        self.n_truncated = 0

    def tokenize(self, texts: list[str], sub: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
        """Tokenise into a flat int32 id array + int64 offsets [n+1] (sub-batches of ``sub`` texts,
        so Python list-of-list transients stay small: 400k tweets as lists cost ~0.7 GB)."""
        flats, lens = [], []
        for i in range(0, len(texts), sub):
            enc = self.tok(texts[i:i + sub], add_special_tokens=True, truncation=True,
                           max_length=self.max_tokens, return_attention_mask=False,
                           return_token_type_ids=False)["input_ids"]
            lens.append(np.fromiter(map(len, enc), np.int64, len(enc)))
            flats.append(np.fromiter((t for ids in enc for t in ids), np.int32, int(lens[-1].sum())))
            del enc
        lens_a = np.concatenate(lens) if lens else np.zeros(0, np.int64)
        offs = np.zeros(len(lens_a) + 1, np.int64)
        np.cumsum(lens_a, out=offs[1:])
        return (np.concatenate(flats) if flats else np.zeros(0, np.int32)), offs

    def encode_ids(self, tokens: tuple[np.ndarray, np.ndarray], out: np.ndarray) -> None:
        """Encode pre-tokenised texts ``(flat_ids, offsets)`` into ``out`` (fp16 [n, 768])."""
        torch = self.torch
        flat, offs = tokens
        lens = np.diff(offs)
        n = len(lens)
        self.n_truncated += int((lens >= self.max_tokens).sum())
        order = np.argsort(lens, kind="stable")
        pad = self.pad_token_id
        i = 0
        dev = self.device
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            while i < n:
                L = int(lens[order[min(n - 1, i + self.max_batch - 1)]])
                bs = max(1, min(self.max_batch, self.batch_tokens // max(L, 1)))
                sel = order[i:i + bs]
                # pad L to a multiple of 16: an unaligned mask makes SDPA fall back to the slow
                # math kernel (measured 3.8x slower)
                L = -(-int(lens[sel[-1]]) // 16) * 16
                arr = np.full((len(sel), L), pad, dtype=np.int64)
                ls = lens[sel]
                rr = np.repeat(np.arange(len(sel)), ls)
                cc = np.arange(int(ls.sum())) - np.repeat(np.cumsum(ls) - ls, ls)
                arr[rr, cc] = flat[np.repeat(offs[sel], ls) + cc]
                x = torch.from_numpy(arr).to(dev, non_blocking=True)
                m = (torch.arange(L, device=dev)[None, :] < torch.from_numpy(ls).to(dev)[:, None])
                h = self.model(input_ids=x, attention_mask=m.long()).last_hidden_state.float()
                mf = m.unsqueeze(-1).float()
                v = (h * mf).sum(1) / mf.sum(1)
                out[sel] = v.half().cpu().numpy()
                i += len(sel)

    @property
    def pad_token_id(self) -> int:
        p = getattr(self.tok, "pad_token_id", None) if self.tok is not None else None
        return int(p if p is not None else getattr(self.model.config, "pad_token_id", 1) or 1)

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), np.float16)
        if texts:
            self.encode_ids(self.tokenize(texts), out)
        return out


# ---------------------------------------------------------------------------------------------


def _des_texts(dataset: str) -> tuple[list[str | None], str]:
    from ..data.profile import load_profile

    p = load_profile(dataset, ["description"], arrow=True)
    d = p.column("description").to_pylist()
    if dataset == "twibot20":
        return [("None" if x is None else x) for x in d], "null/'None ' -> literal 'None'; stripped raw string embedded"
    return d, "null -> zero vector; '' embedded"


def _means_done(d: Path, selection: str) -> bool:
    names = ["tweets_first20_mean"] + (["tweets_all_mean"] if selection == "all" else [])
    return all((d / f"{n}.f16.npy").exists() and (d / f"{n}.meta.json").exists() for n in names)


def _parse_sel(selection: str) -> int | None:
    if selection == "all":
        return None
    if selection.startswith("first"):
        return int(selection[5:])
    raise ValueError(f"selection must be 'all' or 'first<k>', got {selection!r}")


def run_embed(dataset: str, encoder: str = "roberta-base", target: str = "des", selection: str = "first20",
              max_tokens: int = 512, batch_tokens: int = 65536, chunk_texts: int = 200_000,
              force: bool = False) -> Path:
    dp = dataset_paths(dataset)
    out_dir = dp.emb(encoder)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "des" if target == "des" else f"tweets_seq_{selection}"
    final = out_dir / f"{name}.f16.npy"
    meta_path = out_dir / f"{name}.meta.json"
    step = f"{dataset}_emb_{encoder}_{name}"
    if final.exists() and meta_path.exists() and not force:
        m = read_json(meta_path)
        if not m.get("hf_commit") and _cached_commit(encoder):  # backfill (older builds lacked it)
            m["hf_commit"] = _cached_commit(encoder)
            write_json(meta_path, m)
        if target == "tweets" and not _means_done(dp.emb(encoder), selection):
            make_means(dataset, encoder, selection)
        print(f"{step}: already built ({final})")
        return final
    cfg = {"dataset": dataset, "encoder": encoder, "target": target, "selection": selection,
           "max_tokens": max_tokens, "chunk_texts": chunk_texts}
    prog_path = out_dir / f"{name}.progress.json"
    st = StepStatus(step, unit="texts")
    with st:
        t0 = time.time()
        if target == "des":
            texts, policy = _des_texts(dataset)
            n_rows = len(texts)
            store = rows = None
        else:
            from ..data.tweets import TweetStore
            store = TweetStore(dataset)
            cap = _parse_sel(selection)
            rows, sel_off = store.selection(cap)
            save_npy(out_dir / f"{name}_offsets.i64.npy", sel_off)
            save_npy(out_dir / f"{name}_rows.i64.npy", rows)
            n_rows = len(rows)
            texts = None
            policy = "every selected tweet embedded (null text stored as '' and embedded; zeroed in means)"
        # chunks are contiguous output-row ranges; within a des chunk only non-null rows are encoded
        bounds = list(range(0, n_rows, chunk_texts)) + [n_rows]
        chunks = list(zip(bounds[:-1], bounds[1:]))
        prog = read_json(prog_path) if prog_path.exists() and not force else None
        if prog and prog.get("cfg") != cfg:
            prog = None
        done = set(prog["chunks_done"]) if prog else set()
        out = NpyRowWriter(final, np.float16, (n_rows, DIM), resume=bool(prog))
        n_texts = n_rows if texts is None else sum(t is not None for t in texts)
        st.total = n_rows
        st.start(done=sum(chunks[k][1] - chunks[k][0] for k in done), chunks=len(chunks), texts=n_texts)
        enc = Encoder(encoder, max_tokens=max_tokens, batch_tokens=batch_tokens)
        e_empty = enc.encode([""])[0]
        save_npy(out_dir / "e_empty.f16.npy", e_empty)
        pending = [k for k in range(len(chunks)) if k not in done]
        pool = ThreadPoolExecutor(1)

        def prep(k):
            a_, b_ = chunks[k]
            if texts is not None:
                local = [i for i in range(a_, b_) if texts[i] is not None]
                return local, enc.tokenize([texts[i] for i in local])
            return None, enc.tokenize(store.texts(rows[a_:b_]))

        fut = pool.submit(prep, pending[0]) if pending else None
        n_done = sum(chunks[k][1] - chunks[k][0] for k in done)
        for j, k in enumerate(pending):
            local, ids = fut.result()
            if j + 1 < len(pending):
                fut = pool.submit(prep, pending[j + 1])
            a_, b_ = chunks[k]
            buf = np.zeros((b_ - a_, DIM), np.float16)  # null descriptions stay zero
            if local is None:
                enc.encode_ids(ids, buf)
            elif local:
                tmp = np.empty((len(local), DIM), np.float16)
                enc.encode_ids(ids, tmp)
                buf[np.asarray(local) - a_] = tmp
            out.write_rows(a_, buf)
            out.sync()
            done.add(k)
            n_done += b_ - a_
            write_json(prog_path, {"cfg": cfg, "chunks_done": sorted(done)})
            st.update(n_done, force=True, chunk=k, truncated=enc.n_truncated)
        pool.shutdown()
        out.close()
        import torch
        meta = {**cfg, "hf_commit": enc.commit, "dim": DIM, "dtype": "float16", "rows": n_rows,
                "embedded": int(n_texts), "pooling": "mean over all positions incl. <s>,</s>; pads excluded",
                "autocast": "float16", "policy": policy, "truncated_at_max_tokens": enc.n_truncated,
                "seconds": round(time.time() - t0, 1),
                "texts_per_s": round(n_texts / max(1e-9, time.time() - t0), 1),
                "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated())}
        write_json(meta_path, meta)
        prog_path.unlink(missing_ok=True)
        if target == "tweets":
            st.update(n_rows, force=True, phase="means")
            make_means(dataset, encoder, selection)
        st.done(rows=n_rows, texts_per_s=meta["texts_per_s"])
    return final


def make_means(dataset: str, encoder: str, selection: str, chunk_users: int = 20_000,
               max_rows: int = 150_000) -> list[Path]:
    """Per-user means from tweets_seq_<selection>: always ``tweets_first20_mean``; also
    ``tweets_all_mean`` when selection == 'all'. Streams users in chunks with plain reads (bounded
    memory, no page-cache build-up)."""
    from ..data.tweets import TweetStore

    dp = dataset_paths(dataset)
    d = dp.emb(encoder)
    name = f"tweets_seq_{selection}"
    seq_path = d / f"{name}.f16.npy"
    off = np.asarray(load_npy(d / f"{name}_offsets.i64.npy"))
    rows = load_npy(d / f"{name}_rows.i64.npy")
    store = TweetStore(dataset)
    tnull = store.column("text_null") if "text_null" in store.columns else None
    e_empty = np.asarray(load_npy(d / "e_empty.f16.npy"), dtype=np.float32)
    list_null = store.tweet_list_null()
    n = len(off) - 1
    cap_sel = _parse_sel(selection)
    targets = []
    if cap_sel is None or cap_sel >= 20:
        targets.append(("tweets_first20_mean", 20))
    if cap_sel is None:
        targets.append(("tweets_all_mean", None))
    writers = {f: NpyRowWriter(d / f"{f}.f16.npy", np.float16, (n, DIM)) for f, _ in targets}
    s = 0
    while s < n:
        # chunk by users AND by a row budget: max_rows x 768 x (2 + 4 + 4) B ~ 1.2 GB transient
        e = min(n, s + chunk_users, max(s + 1, int(np.searchsorted(off, off[s] + max_rows, side="right")) - 1))
        a, b = int(off[s]), int(off[e])
        blk = read_rows(seq_path, a, b).astype(np.float32)
        if tnull is not None and b > a:
            z = np.asarray(tnull[np.asarray(rows[a:b])]).astype(bool)
            blk[z] = 0.0  # null text contributes a zero vector (counted in the denominator)
        cnt = np.diff(off[s:e + 1])
        for fname, cap in targets:
            if cap is not None:
                cnt_c = np.minimum(cnt, cap)
                within = np.arange(b - a) - np.repeat(off[s:e] - a, cnt)
                keep = within < np.repeat(cnt_c, cnt)
                seg = np.repeat(np.arange(e - s), cnt)[keep]
                vals = blk[keep]
            else:
                cnt_c = cnt
                seg = np.repeat(np.arange(e - s), cnt)
                vals = blk
            sums = np.zeros((e - s, DIM), np.float32)
            _segsum(sums, seg, vals)
            m = sums / np.maximum(cnt_c, 1)[:, None]
            if dataset == "twibot20" and cap is None:
                m[cnt_c == 0] = e_empty  # original Dataset.py: tweet null -> [''] -> e_empty
            writers[fname].write_rows(s, m.astype(np.float16))
        s = e
    written = []
    for fname, cap in targets:
        written.append(writers[fname].close())
        policy = ("zero vector if no tweets; null text -> zero vector counted" if not
                  (dataset == "twibot20" and cap is None) else "no tweets (`tweet: null`) -> e_empty")
        write_json(d / f"{fname}.meta.json", {"source": name, "cap": cap, "policy": policy,
                                               "users_with_tweets": int((np.diff(off) > 0).sum()),
                                               "list_null_users": int(np.asarray(list_null).sum())})
    return written


def _segsum(sums: np.ndarray, seg: np.ndarray, vals: np.ndarray) -> None:
    """sums[seg[i]] += vals[i] for sorted seg (reduceat; much faster than np.add.at)."""
    if len(seg) == 0:
        return
    starts = np.flatnonzero(np.concatenate([[True], seg[1:] != seg[:-1]]))
    sums[seg[starts]] += np.add.reduceat(vals, starts, axis=0)
