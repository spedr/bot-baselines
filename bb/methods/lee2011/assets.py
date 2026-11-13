"""spaCy ``en_core_web_lg`` assets for the Lee 2011 similarity feature (method-local).

The benchmark computes ``Doc.similarity`` with ``en_core_web_lg``. Only two parts of the pipeline
affect that number: the tokenizer (which tokens exist, their ORTH) and the static vector table
(``Vocab.get_vector`` keyed by ORTH, zeros when absent). The tagger/lemmatizer/tok2vec do not
change ORTH or the vectors. So we keep:

    data/methods/lee2011/assets/en_core_web_lg-3.8.0/   tokenizer + vocab/ + config/meta (from the wheel)
    data/methods/lee2011/assets/vectors/keys.u64.npy    sorted ORTH hashes that have a vector [K]
    data/methods/lee2011/assets/vectors/rows.i32.npy    vector row of each key [K]
    data/methods/lee2011/assets/vectors/vectors.f32.npy [R, 300] float32 (memory-mapped at use)

``load_tokenizer()`` builds a blank English pipeline and loads the model's serialized tokenizer
rules into it, so no vectors are loaded into each worker's heap (they are memory-mapped and shared
through the page cache). ORTH hashes are the murmur hash of the token text and do not depend on
the StringStore content, so blank-pipeline ORTH == model ORTH.

Runtime needs only ``<model>/tokenizer`` and ``vectors/`` (about 0.4 GB). The wheel (``dl/``), the
extracted ``<model>/vocab`` and the full extraction (``<model>-full``, only for the spaCy checks in
``tests/methods/``) are rebuilt on demand: ``extract_model`` downloads the wheel (sha256-pinned)
when it is missing. ``python -m bb.methods.lee2011.assets prune`` deletes them again.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

from ...io import read_json, save_npy, write_json
from ...paths import data_root

MODEL = "en_core_web_lg"
MODEL_VERSION = "3.8.0"
WHEEL = f"{MODEL}-{MODEL_VERSION}-py3-none-any.whl"
WHEEL_URL = (f"https://github.com/explosion/spacy-models/releases/download/"
             f"{MODEL}-{MODEL_VERSION}/{WHEEL}")
WHEEL_SHA256 = "293e9547a655b25499198ab15a525b05b9407a75f10255e405e8c3854329ab63"


def asset_root() -> Path:
    return data_root() / "methods" / "lee2011" / "assets"


def model_dir() -> Path:
    return asset_root() / f"{MODEL}-{MODEL_VERSION}"


def vectors_dir() -> Path:
    return asset_root() / "vectors"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def fetch_wheel(dst: Path | None = None) -> Path:
    """Download the model wheel from the spaCy GitHub release (383 MB) and check its sha256."""
    import urllib.request

    dst = Path(dst or asset_root() / "dl" / WHEEL)
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    h = hashlib.sha256()
    with urllib.request.urlopen(WHEEL_URL, timeout=60) as r, open(tmp, "wb") as f:
        while b := r.read(1 << 22):
            h.update(b)
            f.write(b)
    if h.hexdigest() != WHEEL_SHA256:
        tmp.unlink()
        raise ValueError(f"{WHEEL_URL}: sha256 {h.hexdigest()} != {WHEEL_SHA256}")
    tmp.replace(dst)
    return dst


def extract_model(wheel: Path | None = None, full: bool = False) -> Path:
    """Extract the model data directory (no Python code) from the wheel, downloading it when
    missing. ``full`` also extracts the trained pipes (only needed to check our similarity against
    spaCy itself)."""
    out = model_dir() if not full else asset_root() / f"{MODEL}-{MODEL_VERSION}-full"
    if (out / "tokenizer").exists() and (out / "vocab" / "vectors").exists() and (
            not full or (out / "tok2vec").exists()):
        return out
    wheel = Path(wheel) if wheel else fetch_wheel()
    if not wheel.exists():
        raise FileNotFoundError(f"{wheel} missing; download {WHEEL_URL} into {wheel.parent}/")
    got = _sha256(wheel)
    if got != WHEEL_SHA256:
        raise ValueError(f"{wheel}: sha256 {got} != {WHEEL_SHA256}")
    prefix = f"{MODEL}/{MODEL}-{MODEL_VERSION}/"
    keep = ("tokenizer", "config.cfg", "meta.json", "vocab/")
    with zipfile.ZipFile(wheel) as z:
        for info in z.infolist():
            if not info.filename.startswith(prefix) or info.is_dir():
                continue
            rel = info.filename[len(prefix):]
            if (not full and not rel.startswith(keep)) or ".." in Path(rel).parts:
                continue
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".tmp")
            with z.open(info) as src, open(tmp, "wb") as f:
                while b := src.read(1 << 22):
                    f.write(b)
            tmp.replace(dst)
    return out


def export_vectors() -> Path:
    """Export the vector table to plain .npy (sorted keys, rows, float32 matrix). Idempotent."""
    vd = vectors_dir()
    if (vd / "meta.json").exists():
        return vd
    import spacy
    from spacy.vectors import Vectors

    md = extract_model()
    v = Vectors().from_disk(md / "vocab")
    k2r = v.key2row
    keys = np.fromiter(k2r.keys(), dtype=np.uint64, count=len(k2r))
    rows = np.fromiter(k2r.values(), dtype=np.int64, count=len(k2r))
    order = np.argsort(keys, kind="stable")
    data = np.asarray(v.data, dtype=np.float32)
    save_npy(vd / "keys.u64.npy", keys[order])
    save_npy(vd / "rows.i32.npy", rows[order].astype(np.int32))
    save_npy(vd / "vectors.f32.npy", data)
    write_json(vd / "meta.json", {
        "model": MODEL, "model_version": MODEL_VERSION, "wheel_sha256": WHEEL_SHA256,
        "spacy_version": spacy.__version__, "n_keys": int(len(keys)), "n_rows": int(data.shape[0]),
        "dim": int(data.shape[1]), "mode": read_json(md / "vocab" / "vectors.cfg").get("mode"),
        "attr": "ORTH"})
    return vd


def ensure_assets() -> dict:
    """Runtime assets (tokenizer rules + exported vectors); rebuilt from the wheel when missing."""
    vd = vectors_dir()
    if not ((model_dir() / "tokenizer").exists() and (vd / "meta.json").exists()):
        extract_model()
        export_vectors()
    return read_json(vd / "meta.json")


def prune_assets() -> int:
    """Delete everything not needed at runtime (wheel, extracted vocab, full extraction); returns
    the bytes freed. ``ensure_assets`` / ``extract_model(full=True)`` rebuild them on demand."""
    ensure_assets()
    root = asset_root()
    targets = [root / "dl", model_dir() / "vocab", root / f"{MODEL}-{MODEL_VERSION}-full"]
    freed = 0
    for t in targets:
        if t.exists():
            freed += sum(f.stat().st_size for f in t.rglob("*") if f.is_file())
            shutil.rmtree(t)
    return freed


def load_tokenizer():
    """Blank English pipeline with the model's serialized tokenizer rules.

    The lexeme attribute getters (norm, shape, prefix/suffix, like_url, is_stop, ...; about 20
    Python calls per new lexeme) are switched off: the tokenizer never reads them (its splits come
    from the prefix/suffix/infix/token_match/url_match patterns and the special cases, matched on
    ORTH), and ORTH is the hash of the token text, so token boundaries and ORTH are unchanged."""
    from spacy.lang.en import English

    nlp = English()
    nlp.vocab.lex_attr_getters = {}
    nlp.tokenizer.from_disk(model_dir() / "tokenizer")
    return nlp.tokenizer


def load_vectors() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(sorted keys uint64 [K], rows int32 [K], vectors float32 memmap [R, D])."""
    vd = vectors_dir()
    return (np.load(vd / "keys.u64.npy"), np.load(vd / "rows.i32.npy"),
            np.load(vd / "vectors.f32.npy", mmap_mode="r"))


def lookup_rows(orth: np.ndarray, keys: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Vector row of each ORTH hash, -1 where the model has no vector (spaCy returns zeros)."""
    orth = np.asarray(orth, dtype=np.uint64)
    if len(keys) == 0:
        return np.full(len(orth), -1, np.int64)
    pos = np.searchsorted(keys, orth)
    pc = np.minimum(pos, len(keys) - 1)
    hit = keys[pc] == orth
    return np.where(hit, rows[pc].astype(np.int64), -1)


if __name__ == "__main__":  # python -m bb.methods.lee2011.assets {ensure,prune}
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ensure"
    if cmd == "prune":
        print(f"freed {prune_assets() / 1e9:.2f} GB under {asset_root()}")
    elif cmd == "ensure":
        print(ensure_assets())
    else:
        raise SystemExit(f"unknown command {cmd!r} (ensure | prune)")
