"""io, streaming reader, schema parsers, tweet-store access, graph views."""
from __future__ import annotations

import json
import tracemalloc

import numpy as np
import pytest

from bb.build.common import JsonArrayStream
from bb.io import NpyAppender, NpyRowWriter, file_digest, load_npy, read_rows, save_npy
from bb.schema import cresci_bool, parse_twitter_date, to_int, tw20_bool, tw20_str


def test_npy_appender_and_resume(tmp_path):
    p = tmp_path / "a.npy"
    w = NpyAppender(p, np.float16, (4,))
    w.append(np.ones((3, 4)))
    w.append(np.zeros((2, 4)))
    w.flush()
    w2 = NpyAppender(p, np.float16, (4,), resume_rows=3)  # crash recovery: keep 3 rows
    w2.append(np.full((1, 4), 7))
    w2.close()
    a = np.load(p)
    assert a.shape == (4, 4) and a.dtype == np.float16 and a[3, 0] == 7 and a[:3].sum() == 12
    assert not p.with_name("a.npy.partial").exists()


def test_row_writer_sparse_and_resume(tmp_path):
    p = tmp_path / "r.npy"
    w = NpyRowWriter(p, np.float16, (10, 3))
    w.write_rows(4, np.ones((2, 3)))
    w.sync()
    w2 = NpyRowWriter(p, np.float16, (10, 3), resume=True)
    w2.write_rows(9, np.full((1, 3), 2))
    w2.close()
    a = np.load(p)
    assert a[:4].sum() == 0 and a[4:6].sum() == 6 and a[9, 0] == 2
    assert np.array_equal(read_rows(p, 3, 7), a[3:7])
    with pytest.raises(ValueError):
        NpyRowWriter(tmp_path / "x.npy", np.float16, (2, 3)).write_rows(1, np.ones((2, 3)))


def test_save_load_digest(tmp_path):
    save_npy(tmp_path / "x.npy", np.arange(5))
    m = load_npy(tmp_path / "x.npy")
    assert isinstance(m, np.memmap) and m.tolist() == [0, 1, 2, 3, 4]
    d1 = file_digest(tmp_path / "x.npy")
    save_npy(tmp_path / "x.npy", np.arange(6))
    assert file_digest(tmp_path / "x.npy") != d1


@pytest.mark.parametrize("ensure_ascii", [True, False])
@pytest.mark.parametrize("indent", [None, 2])
def test_json_stream_chunk_boundaries_and_resume(tmp_path, ensure_ascii, indent):
    objs = [{"i": i, "t": "é😀\"]}," * (i % 7), "x": [1.5, None, True]} for i in range(300)]
    p = tmp_path / "a.json"
    p.write_text(json.dumps(objs, ensure_ascii=ensure_ascii, indent=indent), encoding="utf-8")
    for chunk in (3, 64, 1 << 20):
        st = JsonArrayStream(p, chunk=chunk)
        got, offs = [], {}
        for o in st:
            got.append(o)
            if len(got) in (1, 150):
                offs[len(got)] = st.offset()
        assert got == objs
        for k, off in offs.items():
            assert list(JsonArrayStream(p, chunk=chunk, start_byte=off)) == objs[k:]


def test_json_stream_errors(tmp_path):
    p = tmp_path / "t.json"
    p.write_text('[{"a": 1}, {"a": 2')
    with pytest.raises(ValueError):
        list(JsonArrayStream(p, chunk=4))
    p.write_text('{"a": 1}')
    with pytest.raises(ValueError):
        list(JsonArrayStream(p))


def test_json_stream_memory_is_bounded(tmp_path):
    """Peak traced allocation does not grow with file size (streaming, not json.load)."""
    def peak(n):
        p = tmp_path / f"m{n}.json"
        p.write_text(json.dumps([{"k": i, "s": "x" * 200} for i in range(n)]))
        tracemalloc.start()
        cnt = sum(1 for _ in JsonArrayStream(p, chunk=1 << 16, depth=2))
        _, pk = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert cnt == n
        return pk
    small, big = peak(2_000), peak(40_000)          # ~0.5 MB vs ~9 MB of JSON
    assert big < 3 * small + (1 << 20)


def test_schema_parsers():
    assert tw20_str("SHAQ ") == "SHAQ" and tw20_str("None ") is None and tw20_str(" ") == ""
    assert tw20_bool("True ") is True and tw20_bool("False ") is False and tw20_bool("None ") is None
    assert to_int("16596 ") == 16596 and to_int("None") is None and to_int(None) is None
    assert parse_twitter_date("Tue Nov 18 10:27:25 +0000 2008 ").year == 2008
    assert parse_twitter_date("2020-01-16 02:02:55+00:00").hour == 2
    assert parse_twitter_date("None ") is None
    assert cresci_bool("1") is True and cresci_bool("") is None and cresci_bool("NULL") is None


# ------------------------------------------------------------------------------- tweet store
def test_tweetstore_selection_and_chunks(tw20):
    from bb.data import TweetStore
    t = TweetStore("twibot20")
    rows, so = t.selection(cap=2)
    cnt = np.minimum(t.counts(), 2)
    assert np.array_equal(np.diff(so), cnt)
    assert all(t.texts(rows[so[i]:so[i + 1]]) == t.user_texts(i, cap=2) for i in range(t.n_users))
    seen = 0
    for u, so2, texts, cols in t.iter_users(chunk_users=4, cap=3, columns=("release_rank",)):
        assert len(u) <= 4 and len(texts) == so2[-1] and len(cols["release_rank"]) == len(texts)
        seen += len(u)
    assert seen == t.n_users
    shuffled = rows[::-1]
    assert t.texts(shuffled) == t.texts(rows)[::-1]


def test_tweetstore_reads_only_requested_bytes(tw20, monkeypatch):
    from bb.data import TweetStore
    t = TweetStore("twibot20")
    calls = []
    orig = t._pread
    monkeypatch.setattr(t, "_pread", lambda a, b: (calls.append(b - a), orig(a, b))[1])
    t.user_texts(1)
    assert sum(calls) == int(t.text_offsets[t.offsets[2]] - t.text_offsets[t.offsets[1]])


# ------------------------------------------------------------------------------- graph
def test_graph_views(tmp_path, data_root):
    from bb.data.graph import (build_csr_from_coo, graph_stats, load_in_merged, load_transpose,
                               load_undirected)
    from bb.paths import dataset_paths
    import pandas as pd
    from bb.build.common import write_index
    n = 5
    users = pd.DataFrame({"idx": np.arange(n), "user_id": [str(i) for i in range(n)],
                          "raw_id": [str(i) for i in range(n)], "split": ["train"] * 3 + ["test"] * 2,
                          "label": [0, 1, 0, 1, 0], "raw_order": np.arange(n), "split_csv_order": -1,
                          "label_csv_order": -1, "source_file": "x"})
    dp = dataset_paths("toy")
    write_index(dp, users, {})
    # relation a: 0->1 (pos 5), 0->2 (pos 1), 3->3 self loop (pos 2); relation b: 1->0 (pos 0), 2->0 dup x2
    build_csr_from_coo(dp.csr("a"), [0, 0, 3], [1, 2, 3], [5, 1, 2], n)
    build_csr_from_coo(dp.csr("b"), [1, 2, 2], [0, 0, 0], [0, 3, 4], n)
    from bb.data import load_csr
    a = load_csr("toy", "a")
    assert a.row(0).tolist() == [2, 1]                                  # release order within a row
    t = load_transpose("toy", "a")
    assert t.row(1).tolist() == [0] and t.row(3).tolist() == [3]
    m = load_in_merged("toy", ["a", "b"])
    assert m.row(0).tolist() == [1, 2, 2] and m.etype[m.indptr[0]:m.indptr[1]].tolist() == [1, 1, 1]
    u = load_undirected("toy", ["a", "b"])
    assert u.row(0).tolist() == [1, 2] and u.row(3).tolist() == []      # dedup, self loop removed
    s = graph_stats("toy", ["a", "b"])
    assert s["a"]["self_loops"] == 1 and s["b"]["unique_pairs"] == 2
    assert s["overlap__a__b"]["a_pairs_found_reversed_in_b"] == 2
    assert s["isolated_by_split"]["test"]["isolated"] == 1             # node 4


def test_loaders_return_memmaps_and_gpu_copy_is_chunked(tw20, monkeypatch):
    import torch
    from bb.data import load_embedding, load_tweet_seq
    from bb.harness import gpu
    from bb.io import save_npy
    from bb.paths import dataset_paths
    d = dataset_paths("twibot20").emb("fake")
    a = np.arange(19 * 8, dtype=np.float16).reshape(19, 8)
    save_npy(d / "des.f16.npy", a)
    m = load_embedding("twibot20", "fake", "des")
    assert isinstance(m, np.memmap) and not m.flags.writeable
    seen = []
    orig = torch.Tensor.copy_
    monkeypatch.setattr(torch.Tensor, "copy_", lambda self, src, **k: (seen.append(src.shape[0]), orig(self, src, **k))[1])
    t = gpu.memmap_to_gpu(m, rows=np.array([3, 1, 7]), device="cpu", chunk_rows=2)
    assert torch.equal(t, torch.from_numpy(a[[3, 1, 7]])) and max(seen) <= 2
    assert np.array_equal(gpu.gather_rows(m, np.array([5, 0, 5])), a[[5, 0, 5]])
