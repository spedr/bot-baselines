"""Real-data check of the Lee 2011 similarity against spaCy itself (not a pytest file: it loads
the full en_core_web_lg pipeline and real tweets; run under memrun --light):

    tools/memrun --light -- uv run python tests/methods/lee2011_spacy_check.py --dataset twibot20 --users 40

1. ORTH sequences from the blank tokenizer == those of the full model pipeline;
2. similarity_kernel == mean pairwise Doc.similarity with nlp = spacy.load(lg, disable=[parser, ner])
   (the benchmark's call) for the first tweets of real users;
3. tokenizer throughput (tweets/s, one core).
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twibot20")
    ap.add_argument("--users", type=int, default=40)
    ap.add_argument("--cap", type=int, default=30)
    a = ap.parse_args()
    import spacy
    from spacy.attrs import ORTH

    from bb.data import TweetStore, load_split
    from bb.methods.lee2011.assets import extract_model, load_tokenizer, load_vectors
    from bb.methods.lee2011.features import users_stats

    st = TweetStore(a.dataset)
    users = np.asarray(load_split(a.dataset, "train"))[: a.users * 3]
    users = users[np.asarray(st.counts())[users] > 1][: a.users]
    rows, so = st.selection(a.cap, users)
    texts = st.texts(rows)
    tok = load_tokenizer()
    keys, krows, vec = load_vectors()
    t0 = time.time()
    blank = [d.to_array(ORTH) for d in tok.pipe(texts, batch_size=1000)]
    rate = len(texts) / (time.time() - t0)
    nlp = spacy.load(extract_model(full=True), disable=["parser", "ner"])
    docs = list(nlp.pipe(texts, batch_size=256))
    n_tok_diff = sum(not np.array_equal(b, d.to_array(ORTH)) for b, d in zip(blank, docs))
    ref = []
    for j in range(len(users)):
        ds = docs[so[j]:so[j + 1]]
        if len(ds) == 1:
            ref.append(1.0)
            continue
        ref.append(float(np.mean([x.similarity(y) for x, y in itertools.combinations(ds, 2)])))
    ours = users_stats(texts, so, tok, keys, krows, vec)[:, 7]
    err = np.abs(np.asarray(ref) - ours)
    print(f"{a.dataset}: {len(users)} users, {len(texts)} tweets; tokenizer {rate:,.0f} tweets/s/core")
    print(f"ORTH sequences differing blank vs full pipeline: {n_tok_diff} / {len(texts)}")
    print(f"similarity max |ours - spaCy| = {err.max():.2e} (mean {err.mean():.2e}); "
          f"ref range {min(ref):.3f}..{max(ref):.3f}")
    assert n_tok_diff == 0 and err.max() < 1e-5


if __name__ == "__main__":
    main()
