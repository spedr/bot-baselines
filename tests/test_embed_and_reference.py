"""Embedding pooling/means on a tiny random encoder (CPU) and the reference method end to end."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch


def _tiny_encoder():
    from transformers import RobertaConfig, RobertaModel
    from bb.build.embed import Encoder
    torch.manual_seed(0)
    cfg = RobertaConfig(vocab_size=60, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=32, max_position_embeddings=80, pad_token_id=1)
    m = RobertaModel(cfg, add_pooling_layer=False)
    return Encoder("tiny", model=m, tokenizer=None, device="cpu", amp=False, batch_tokens=40, max_batch=4), m


def test_pooling_matches_unpadded_mean_over_all_tokens():
    enc, m = _tiny_encoder()
    rng = np.random.default_rng(0)
    seqs = [[0] + rng.integers(3, 60, rng.integers(0, 30)).tolist() + [2] for _ in range(23)]
    lens = np.array([len(s) for s in seqs])
    offs = np.concatenate([[0], np.cumsum(lens)])
    flat = np.concatenate(seqs).astype(np.int32)
    out = np.zeros((len(seqs), 16), np.float16)
    enc.encode_ids((flat, offs), out)                                    # bucketed, padded batches
    with torch.no_grad():
        for i, s in enumerate(seqs):                                     # reference: one text, no padding
            ref = m(input_ids=torch.tensor([s])).last_hidden_state[0].mean(0).numpy()  # incl. <s>, </s>
            assert np.allclose(out[i].astype(np.float32), ref, atol=2e-3), i


def test_make_means_policies(tw20):
    from bb.build.embed import make_means
    from bb.data import TweetStore
    from bb.io import save_npy
    from bb.paths import dataset_paths
    t = TweetStore("twibot20")
    rows, so = t.selection(None)
    d = dataset_paths("twibot20").emb("fake")
    rng = np.random.default_rng(0)
    seq = rng.standard_normal((len(rows), 768)).astype(np.float16)
    save_npy(d / "tweets_seq_all.f16.npy", seq)
    save_npy(d / "tweets_seq_all_offsets.i64.npy", so)
    save_npy(d / "tweets_seq_all_rows.i64.npy", rows)
    e_empty = np.full(768, 0.5, np.float16)
    save_npy(d / "e_empty.f16.npy", e_empty)
    make_means("twibot20", "fake", "all", chunk_users=3)
    allm = np.load(d / "tweets_all_mean.f16.npy").astype(np.float32)
    f20 = np.load(d / "tweets_first20_mean.f16.npy").astype(np.float32)
    for i in range(t.n_users):
        a, b = so[i], so[i + 1]
        if b > a:
            assert np.allclose(allm[i], seq[a:b].astype(np.float32).mean(0), atol=2e-3)
            assert np.allclose(f20[i], seq[a:min(b, a + 20)].astype(np.float32).mean(0), atol=2e-3)
        else:
            assert np.allclose(f20[i], 0)                                # first20: zero if none
            if t.tweet_list_null()[i]:
                assert np.allclose(allm[i], 0.5)                         # tw20 original: null -> e_empty


def test_reference_method_end_to_end(tw20, data_root):
    from bb.harness import load_results
    from bb.extscore import make_synthetic_extscore
    from bb.methods import logreg_ref
    logreg_ref.main(["--dataset", "twibot20", "--seeds", "0", "--epochs", "20", "--patience", "5"])
    logreg_ref.main(["--dataset", "twibot20", "--seeds", "0", "--epochs", "20", "--patience", "5"])
    make_synthetic_extscore("twibot20", tag="syn", coverage=1.0)
    logreg_ref.main(["--dataset", "twibot20", "--seeds", "0", "--epochs", "20", "--with-extscore", "syn"])
    base = load_results("logreg_ref", "twibot20", where={"with_extscore": None})
    extscore = load_results("logreg_ref", "twibot20", where={"with_extscore": "syn"})
    assert len(base) == 2 and len(extscore) == 1
    assert base[0]["metrics"]["test"] == base[1]["metrics"]["test"]       # seeded runs reproduce
    assert base[0]["n_params"] == 13 * 2 and extscore[0]["n_params"] == 15 * 2  # extscore none: unchanged width
    assert base[0]["extscore"] == {"extscore": "none"} and extscore[0]["extscore"]["tag"] == "syn"
    assert extscore[0]["selection"]["select"] == "best_val"
    assert (data_root.parent / "runs" / "logreg_ref" / "twibot20").exists()
