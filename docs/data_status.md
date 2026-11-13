# Data status

Live state: `uv run bb status` (or `data/status/<step>.json`). Invariant and count checks:
`uv run bb verify <dataset>` (`make verify`). All times and peaks below were measured on the
development machine (RTX 4070 Ti 12 GB, Ryzen 7 5800X, 16 GB RAM under WSL2) with `/usr/bin/time -v`
max RSS and the VmHWM in the status files.

## Readiness

| dataset | index + splits | profile | tweet store | graph CSR | des embeddings | tweet embeddings |
|---|---|---|---|---|---|---|
| TwiBot-20 | done | done | done | done | done (distilroberta-base, roberta-base) | done (`all` per-tweet sequences + means, `first20` mean) |
| Cresci-2015 | done | done | done | done | done (roberta-base) | done (`first20` + mean) |
| TwiBot-22 | done | done | done | done | done | done (`first20` + mean) |

The embedding chain is `tools/precompute.sh` (run detached with `make precompute`). Each step is a
separate `tools/memrun -m 8G` job holding the heavy-job lock. Every step is idempotent and resumable
at 200k-text chunks: if the chain dies, re-run it. Check progress with `uv run bb status`,
`tail data/logs/precompute.log` and `data/logs/<step>.log`. The chain is finished when
`data/logs/precompute.log` ends with `precompute chain finished`.

Built by the methods rather than the shared layer: the `botrgcn` feature scheme
(`data/features/<ds>/botrgcn/`, built by `bb.features.botrgcn`). Not built: TwiBot-22 per-tweet embeddings beyond
the first 20 (BIC extension: `bb embed twibot22 --target tweets --selection first200`, about 77M
tweets, about 5-6 h, about 118 GB), method-local artifacts.

## Verified counts

| check | ours | reference |
|---|---|---|
| TwiBot-20 users train/val/test/support | 8,278 / 2,365 / 1,183 / 217,754 | spec + benchmark hard-coded ranges |
| TwiBot-20 follow edges | following 117,110 + follower 110,869 = 227,979 | 227,979 (BotRGCN paper) |
| TwiBot-20 users with neighbour lists | 10,739 labelled + 1,185 support | spec (counted) |
| TwiBot-20 tweets | 33,488,192 | 33,488,192 (TwiBot-22 paper Table 1) |
| TwiBot-20 `tweet: null` users | 80 labelled, 23,457 support | spec |
| TwiBot-20 tweets of labelled users | **1,999,788** | spec says 1,786,245 (probably counted on the embedding-sorted `train.json`/`test.json`; our total matches the paper) |
| TwiBot-20 missing profiles | 96 (`created_at` null) | 96 (BotDGT code comment) |
| TwiBot-22 users / splits | 1,000,000; 700k/200k/100k | spec |
| TwiBot-22 bots train/val/test | 54,586 / 55,913 / 29,444 | spec |
| TwiBot-22 `split.csv` row == `label.csv` row | 1 user | spec (Kudugunta misalignment) |
| TwiBot-22 `user.json` order == `label.csv` order | yes (all 1M) | spec: inferred, now verified |
| TwiBot-22 following / followers edges | 2,626,979 / 1,116,655 (none dropped) | paper Table 7 |
| TwiBot-22 tweets in release | 88,217,457 (= `post` edges) | paper Table 7 |
| TwiBot-22 zero-tweet users train / val / test | 36,797 / 16,501 / 12,830 | 36,797 / 16,501 (unified manifest) |
| Cresci-2015 users | 5,301 (1,950 human / 3,351 bot) | benchmark |
| Cresci-2015 vs MGTAB BotRGCN tensors | labels, all 5 numeric columns (followers, active_days@2020-09-01, len(screen_name), friends, listed; z-scored) and train/val/test (3,708/1,058/535) **equal row by row**; (src, dst, type) edge multiset equal (14,220) | `raw/mgtab-cresci-15/Cresci-15.zip` |
| profile parsing vs unified jsonl | 0 mismatches on 1,183 TwiBot-20 test users and 3,000 TwiBot-22 users; Cresci 79 descriptions differ (double-encoded UTF-8 in the release, repaired in unified; we keep the release spelling) | a separate unified cross-dataset export |

The Cresci-2015 benchmark split was marked "unverified" in the spec. Our index order (subsets E13,
TFP, INT, TWT, FSF in users.csv order) and the per-subset contiguous 70/20/10 split reproduce the
MGTAB-repo tensors exactly. `bb verify cresci15` re-checks this. The MGTAB **text** tensors are not
roberta-base (cosine about 0 with ours, different dominant dimensions), so they cannot validate
embedding values. Their zero rows (1,216 null descriptions, 153 users without tweets) match ours
exactly.

## Facts found while building (relevant to method specs)

- TwiBot-22 `user.json` has no null description (124,230 are `''`). So "null -> zero vector"
  never triggers on TwiBot-22: every empty description gets `e_empty`.
- TwiBot-22 tweet order: within a user, 97.2% of consecutive release-order pairs have
  non-increasing `created_at` (mostly newest first). Users' tweets are split over several file
  blocks for 67% of users. 100,259 users have more than 200 tweets (11.27M tweets beyond the cap).
- TwiBot-22 `referenced_tweets` is null for 75.3M of the 76.9M stored tweets. `is_retweet` falls
  back to the `RT @` prefix there, and `is_quote` is unknown (-1). `reply_count` and `quote_count`
  are null in the sample inspected.
- TwiBot-22 follow graph: only about 10k users have out-edges (9,936 `following`, 9,653
  `followers`). Max in-degree is 2,076 per relation; the 270,344 in the paper is for the
  heterogeneous graph. 410,823 pairs appear in both relations in the same direction. 50.7% of test,
  25.1% of train and 39.9% of val users are isolated.
- TwiBot-22 sampler dry run (`load_in_merged(followers+following)`, 100 batches, train seeds):

  | fanouts, seeds | nodes p50 / p99 | edges p99 | ms per batch p50 |
  |---|---|---|---|
  | [256,256], 256 | 6,503 / 7,398 | 39,203 | 0.3 |
  | [256,256], 1,024 | 9,570 / 9,832 | 98,415 | 1.0 |
  | [256]*4, 256 (RGT/SHGN `bench_compat`) | 9,976 / 9,998 | 190,929 | 2.0 |
  | [25,10], 4,096 (SEBot plan) | 12,462 / 12,555 | 51,241 | 1.8 |

  The specs' fear that 4-hop sampling approaches the full graph does not hold: subgraphs saturate
  at about 10k nodes.
- TwiBot-20 graph: edges leave only users who have lists. In-degree p99 is 3 (`following`) and the
  max is 124. 23,838 support users and 12 labelled users are isolated.
- TwiBot-20 has 3 users with 201 tweets.

## Measured time and peak memory per step

| step | wall | peak RSS | notes |
|---|---|---|---|
| `twibot22_users` (user.json 0.78 GB + label/split) | 23.6 s | 1.16 GB | |
| `twibot22_graph` (edge-002.csv 6.6 GB, 170.2M rows) | 36 s | 1.28 GB | pyarrow CSV, vectorised id map; ran `--no-lock -m 2G` |
| `twibot22_tweets` (tweet_0..8, 104.4 GB) | 17.7 min (stream 15.0 min, group-by 2.6 min) | 1.25 GB | 116 MB/s average; ran `--no-lock -m 2G` (I/O-bound, no GPU) |
| `twibot20_build` (5.6 GB, four files, one pass) | 50 s | 0.80 GB | |
| `cresci15_build` | 30 s | 3.1 GB | whole subset zips in memory (largest E13 tweets.csv 0.6 GB) |
| `twibot20` des distilroberta-base / roberta-base | 35 s / 48 s | 3.7 GB | 6,636 / 4,744 texts/s; built with the earlier memmap writer (now NpyRowWriter: about 1.9 GB) |
| `cresci15` des / tweets first20 + mean | 8 s / 23 s | 1.9 GB / 2.6 GB | 4,079 tweets/s |
| `twibot20` tweets all (33.5M) | 3.3 h | 4.1 GB | 2,623 tweets/s average (GPU shared with other jobs) |
| `twibot22` tweets first20 (17.8M) | about 2.3 h (resumed once) | 4.5 GB | 2,175 tweets/s average (GPU shared) |
| derived views (in_merged, undirected) + 4 sampler dry runs, TwiBot-22 | 6.8 s | 0.92 GB | |
| reference method, TwiBot-20, 3 seeds | 3.4 s | 1.28 GB | CPU; torch import dominates |

VRAM for the embedding jobs is at most 2.6 GB (`cuda_max_allocated_bytes` in each `<name>.meta.json`).
Disk use: `data/twibot22` 16 GB (tweet text 10.6 GB), `data/twibot20` 8.7 GB, rising by about 51 GB for
the TwiBot-20 per-tweet embeddings and by about 30 GB for TwiBot-22 (`first20` sequence + des + mean).

## Language choice: measurements

| measurement | value |
|---|---|
| raw read speed from the Windows mount, cold (first 1 GB of tweet_0.json / 300 MB of edge CSV) | 45 / 50 MB/s |
| actual average during the full tweet pass | 116 MB/s (Windows-side caching) |
| Python `json.JSONDecoder.raw_decode` stream, objects only | 168 MB/s |
| same + full per-tweet field extraction (the builder's work) | 134 MB/s, 115k tweets/s |
| `JsonArrayStream` (reader thread + incremental UTF-8 + raw_decode) | about 180 MB/s |
| pyarrow streaming CSV on edge-002.csv | 303 MB/s |

Parsing in Python is 1.2-3x faster than the disk delivers, and phase 1 of the tweet pass was
disk-bound (15 min for 104 GB). Rust (installed: `~/.cargo/bin`) would only have saved CPU time, not
wall time, so every builder is Python.

Embedding speed: roberta-base fp16 autocast forward on the 4070 Ti reaches about 245k tokens/s at
any length (L = 16-512). The first implementation reached only 1,144 tweets/s because unaligned
padded lengths made SDPA fall back to its math kernel. Padding batch length to a multiple of 16 gave
4,392 tweets/s on a 100k TwiBot-20 sample (mean 45 tokens, p99 166, max 512).

## Incidents (all resolved)

- An early version of the embedding job zero-filled its 51 GB output through a writable memmap.
  RSS reached 8.4 GB (just under the 8 GiB cap) before the job was stopped. Since then large
  outputs use `NpyRowWriter` (pwrite + fdatasync + FADV_DONTNEED) and inputs use `read_rows`/pread.
  The same job was restarted once more to cut token-list memory (lists to flat int32 arrays:
  6.8 GB to 3.0 GB) and once to chunk the means by a row budget. It resumes from its chunk
  checkpoint each time.
- `twibot20_build` ran twice (support.json read twice) because a post-pass step failed on a step
  ordering bug, now fixed. Writes are now ordered so that a late failure never needs a re-read.

## Open items

- Embedding metadata of the TwiBot-20 tweet job may lack `hf_commit` (the fix landed after it
  started). Re-running `tools/precompute.sh` (all steps are no-ops once built) backfills it. Commits:
  roberta-base `e2da8e2f811d1448a5b465c236feacd80ffbac7b`, distilroberta-base
  `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b`.
- After the chain finishes, run `uv run bb verify twibot20` and check
  `data/twibot20/emb/roberta-base/tweets_all_mean.f16.npy` and
  `data/twibot22/emb/roberta-base/{des,tweets_first20_mean}.f16.npy`.
- Other datasets (Cresci-2017, Caverlee-2011, BotSim-24) have no builder yet. Feature/text methods
  need one before they can run there.
