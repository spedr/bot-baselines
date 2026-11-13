# Shared data layer: requirements

What the shared data layer and harness must provide so that every method in `docs/methods.md` can be
built on it without touching raw files. Method-specific details live in `docs/methods/<method>.md`;
this file collects what is common and what has to be decided once. Evidence labels: **[P]** checked in
a primary source or on our files during the spec pass, **[U]** unverified or estimated.

## 0. Hard memory and runtime rules (mandatory)

The machine has 15.9 GB RAM, 4 GB swap and other processes using 1-2 GB. Exceeding memory crashes
WSL. Therefore:

1. **Every** preprocessing, embedding, graph-building, training and evaluation job runs through
   `tools/memrun [-m 10G] [--no-lock] -- <cmd>`. It runs the command in a systemd user scope with
   `MemoryMax` (default 10G) and no swap, so an oversize job is OOM-killed on its own, and it
   serialises heavy jobs with an exclusive lock. `--no-lock` is only for light jobs (< 1 GB).
2. Design target: peak resident memory (RSS) per job **at most about 8 GB**; 10 GB is the hard kill
   line. Each step in each spec carries a host-RSS and VRAM estimate; the harness logs the measured
   peak (`/proc/self/status` VmHWM, `torch.cuda.max_memory_allocated`) into the run's `meta.json`.
3. **No step loads a full TwiBot-22 tweet file (`tweet_i.json`, about 11 GB each), the edge file
   (`edge-002.csv`, 6.6 GB), `user.json` (0.78 GB raw, several GB as Python objects), TwiBot-20
   `support.json` (5.3 GB) or a full feature table into RAM.** Stream (ijson / line readers / chunked
   CSV), chunk, memory-map (`np.load(mmap_mode='r')`, `np.memmap`), or use on-disk CSR.
4. DataLoader worker processes (each has its own RSS, typically 0.3-1 GB with torch imported) and
   pinned-memory buffers count toward the budget. Default `num_workers=0` for in-memory/memmap
   tensors; at most 2 workers where a method spec justifies it.
5. Artifacts are written to the Linux disk (`artifacts/` under the repo, ext4, about 500 GB free),
   never to the Windows-mounted raw-data drive (about 70 MB/s). Raw data is read once per pass.
6. Never build a Python dict/list of per-tweet objects for TwiBot-22 (88M tweets): use numpy arrays
   appended in chunks and flushed to disk.

## 1. Datasets and which methods they support

| Dataset | Held as | Graph | Tweet text | Per-tweet metadata | Official split | Methods supported |
|---|---|---|---|---|---|---|
| TwiBot-22 | full raw release (`user.json`, `tweet_0..8.json`, `edge-002.csv`, `label.csv`, `split.csv`) | yes (follow: 3,743,634 user-user edges [P-paper Table 7]) | yes, about 88.2M tweets [P-paper] | yes (raw tweet objects) | yes, 700k/200k/100k | all; official numbers exist for every method except SEBot and BIC |
| TwiBot-20 | full release (`train_bak.json`, `dev.json`, `test_bak.json`, `support.json`) | yes (neighbour lists, at most 20 + 20 per user; 227,979 edges [P-paper]) | yes, at most 200 per user, about 33.5M incl. support | **no** (texts only) | yes, 8,278/2,365/1,183 + 217,754 support | all; official numbers for every method except Kudugunta-CLSTM |
| Cresci-2015 | raw CSVs (users, tweets, followers, friends per subset) + BotRGCN-format tensors from the MGTAB repo (`raw/mgtab-cresci-15/Cresci-15.zip`) | yes (friends/followers) | yes | yes (CSV columns) | benchmark split not held in raw form; the MGTAB zip carries a 3,708/1,058/535 split [P: read from the tensors] whose identity with the benchmark split is [U] | feature, text and graph methods; comparable to published numbers only on the benchmark split |
| Cresci-2017 | raw CSVs | no | yes | yes | ours (hash split, see unified manifest) | feature and text methods only |
| Caverlee-2011 | raw | no | yes | partial | ours | feature and text methods (Lee et al. was designed on it) |
| BotSim-24 (Reddit) | raw + unified | small user graph via the BotSim repo [U] | yes (posts/comments) | partial | ours (the paper's 2,304/582/291 split is seed-dependent [U]) | feature and text methods; graph methods need a Reddit-specific feature scheme (out of scope for the first pass) |
| MGTAB | feature tensors only | yes (7 relations) | no raw text | no | random 7:2:1 | graph methods without ExtScore only (no text, so no ExtScore column can be computed) |

The unified per-account jsonl under `unified/` (a separate cross-dataset export) must **not** be used as the source
of official splits: it removes accounts from TwiBot-20/22 train and dev for the cross-dataset rule
(e.g. 521 TwiBot-20 train users, 1,755 TwiBot-22 train users removed) [P: manifest.json], and its
TwiBot-20 tweet order differs from the release [P: first tweet of the first test user differs].
Build from the raw releases. The unified files remain useful for cross-dataset experiments and as a
cross-check of profile parsing.

Local TwiBot-20 `train.json` and `test.json` are embedding-sorted copies made by another project; use
`train_bak.json`, `dev.json`, `test_bak.json`, `support.json` [P: botrgcn.md 3.0].

## 2. Canonical user index: `artifacts/data/<dataset>/users.parquet`

| column | type | meaning |
|---|---|---|
| `idx` | int64 | dense 0..N-1, the row index of every other table |
| `user_id` | string | decimal digits (TwiBot-22 `u` prefix removed; keep `raw_id` too) |
| `raw_id` | string | id exactly as in the release |
| `split` | category | `train`, `val`, `test`, `support` |
| `label` | int8 | 1 bot, 0 human, -1 unlabelled (support) |
| `raw_order` | int64 | position in the raw user file (`user.json`; TwiBot-20: position within its source file) |
| `split_csv_order` | int64 | row position in `split.csv` (TwiBot-22) / -1 elsewhere |
| `source_file` | category | TwiBot-20: which of the four files |

Index order (it must match the benchmark so that index-based code paths and emulated bugs line up):
- **TwiBot-22:** `user.json` order, all 1,000,000 users, all labelled. `split.csv` uses `train`,
  `val`, `test` [P]. Class balance [P: computed from label.csv + split.csv]: train 54,586 bots /
  645,414 humans (7.8%), val 55,913 / 144,087 (27.9%), test 29,444 / 70,556 (29.4%). The train prior
  differs strongly from val/test: report it, and never tune a threshold on train.
- **TwiBot-20:** train (8,278), dev (2,365), test (1,183), then support (217,754), each in file
  order; 229,580 rows [P: botrgcn.md 3.1.1]. The benchmark hard-codes `train_idx = 0..8277`,
  `val_idx = 8278..10642`, `test_idx = 10643..11825`.
- **Cresci-2015 / others:** labelled users in raw file order; record the split provenance.

Also write `splits/{train,val,test}_idx.npy` (int64) and `meta.json` (counts, class balance, source
hashes).

**Evaluation-set leakage to guard against.** BotSay-340 (the 340-account balanced TwiBot-22 test set
used by BotPercent/BotSay/BotCompletion) draws 201 users from the official train split, 103 from val
and 36 from test [P: computed]. A model trained on the official TwiBot-22 train split must not be
evaluated on BotSay-340; use the BotSay train split (2,694) for that protocol, or exclude the 340 from
training and say so. The harness should refuse an evaluation set that overlaps the training indices.

## 3. Profile table: `artifacts/data/<dataset>/profile.parquet`

One row per `idx`, typed, with nulls preserved. TwiBot-20 raw values are strings with a trailing
space (`"16596 "`, `"True "`, `"None "`) [P]; parse them (strip, `"None"` to null, `"True"/"False"`
to bool, counts to int64, dates to UTC timestamps) and **also keep a `raw_` copy of the fields whose
raw spelling some benchmark code compares against** (`raw_location`, `raw_profile_image_url`,
`raw_default_profile_image`, `raw_url`; see bothunter.md and botrgcn.md 3.1.3).

Union of fields needed by the selected methods:

| field | TwiBot-22 source (`user.json`, API v2) | TwiBot-20 source (`profile`, API v1.1) | used by |
|---|---|---|---|
| `followers_count`, `following_count` (`friends_count`), `statuses_count` (`tweet_count`), `listed_count` | `public_metrics.*` | same names (strings) | all feature methods, `botrgcn` |
| `favourites_count` | **absent** in TwiBot-22 (the processed jsonl's `favourites_count: 0` is a placeholder) [P] | present | BotHunter/SGBot extended modes, `botrgcn-orig6` |
| `created_at` | ISO string | `"Mon Oct 28 13:39:41 +0000 2019 "` | all; BotDGT snapshots |
| `name`, `screen_name` (`username`) | yes | yes | Lee, SGBot, BotHunter, `botrgcn` (TwiBot-22 uses `len(name)`, a benchmark bug) |
| `description` | string; **keep JSON null distinct from `''`** | string | `botrgcn` (null gives a zero vector, `''` is embedded), Lee, SGBot |
| `url`, `location` | yes (nullable) | yes | BotHunter, Kudugunta, SGBot |
| `protected`, `verified` | bool | string bool | `botrgcn`, Kudugunta, SGBot |
| `profile_image_url` | yes (default-image test by exact URL) | yes | `botrgcn` cat col 2, SGBot, BotHunter, Kudugunta |
| `default_profile`, `default_profile_image`, `geo_enabled`, `profile_use_background_image`, `profile_background_tile`, `has_extended_profile`, `contributors_enabled`, `is_translator`, `is_translation_enabled`, `lang`, `utc_offset`, `time_zone` | **absent** | present | `botrgcn-orig6`, SGBot paper mode, SEGCN TwiBot-20 (11 flags) |
| `pinned_tweet_id`, `withheld`, `entities` | yes | `entities` only | none of the defaults (keep for completeness) |
| `collected_at` (reference date) | constant per dataset, as each method's spec states (benchmark constants differ by method: e.g. 2020-09-05, 2022-04-20) | constant | per-method config, never a profile column |

Peak memory: stream `user.json` with ijson (installed into the venv; not present yet) or a raw-decode
loop, write parquet in row groups of 100k: under 1 GB RSS [U estimate]. `support.json` likewise
(ijson, `item` prefix): under 1 GB RSS.

## 4. Tweet store: `artifacts/data/<dataset>/tweets/`

Ragged, column-oriented, one shard set per dataset:
- `offsets.i64.npy` [N+1]: tweets of user `idx` are rows `offsets[idx]:offsets[idx+1]`.
- `text` stored as one UTF-8 byte blob (`text.bin`) + `text_offsets.i64.npy` [T+1], or Arrow
  `LargeStringArray` in a parquet with row groups of 1M tweets.
- per-tweet columns (int64/int32/int8, null-able where marked): `release_rank` (TwiBot-22: global
  position across `tweet_0..8.json`, i.e. file index * 1e9 + position; TwiBot-20: position in the
  user's `tweet` list), `tweet_id`, `created_at` (TwiBot-22 only), `is_retweet`, `is_reply`,
  `is_quote`, `retweet_count`, `reply_count`, `like_count`, `quote_count`, `n_hashtags`, `n_urls`,
  `n_mentions`, `lang`, `possibly_sensitive`, `source` (as available).
- Order within a user: **release order** (the order of appearance in the raw files), with
  `release_rank` kept so any method can re-sort. Several recipes depend on it: `botrgcn` uses the
  **first 20 in release order** on TwiBot-22 (not the newest 20), Lee uses the first 20 of an unstable
  sort (emulated from release order), BIC and Lee use the full list on TwiBot-20, where release order
  is newest first ("the recent 200 tweets", README) [P].
- Cap: keep **all** tweets for TwiBot-22 users up to 200 per user in release order and record
  `n_tweets_total` per user in `users.parquet`; nothing selected needs more than 200 [P: method
  specs]. TwiBot-20 is already capped at 200 by the release.

TwiBot-22 build (the single most expensive I/O step; do it once):
- One streaming pass over `tweet_0.json` ... `tweet_8.json` (about 100 GB on a 70 MB/s drive: at least
  24 min of pure I/O; with a C-backed streaming JSON parser expect 2-6 h in total [U]). The file
  layout (one top-level JSON array per file vs one object per line) is [U]; the reader must handle
  both and must never `json.load` a file. Peak RSS: per-user counter array (1M x int32 = 4 MB),
  output buffers of 1M tweets flushed to disk (about 0.3 GB), parser state: under 2 GB.
- Map `author_id` to `idx` with a sorted id array + `np.searchsorted` (1M x int64 = 8 MB), not a
  Python dict of 1M strings (about 150 MB, acceptable but slower).
- Writes are sharded by file; a final on-disk pass groups by `idx`: argsort of the (idx,
  release_rank) keys for at most about 88M tweets is 88M x 8 B = 0.7 GB per key array plus a 0.7 GB
  permutation, so about 2-3 GB RSS; then gather the text blob in chunks. Disk: texts about 10-15 GB,
  metadata about 3 GB [U estimate].
- Also count per-user tweets beyond the cap and users with zero tweets (36,797 train and 16,501 val
  users have none per the unified manifest [S]).

TwiBot-20 build: stream the four files with ijson; tweets are plain strings; per-tweet metadata other
than what regexes can recover (`RT @` prefix, URLs, mentions, hashtags) does not exist [P].

## 5. Text-embedding tables: `artifacts/data/<dataset>/emb/<encoder>/`

Fixed encoders (pin the HF revision hash in `meta.json`):
- `roberta-base`: tweets on all datasets, descriptions on TwiBot-22 and Cresci-2015 (benchmark).
- `distilroberta-base`: TwiBot-20 descriptions in the original BotRGCN recipe (the benchmark README
  says RoBERTa; see botrgcn.md 3.1.4, variant `botrgcn-b20` uses roberta-base throughout).

Pooling (this is not the usual masked mean): last hidden layer, **arithmetic mean over all token
positions including `<s>` and `</s>`**, excluding only padding; truncation at 512 tokens (the
benchmark's `max_length=50` was most likely ignored by the pipeline [U]; keep a `max_tokens` flag).
Run in fp16 autocast with length-bucketed batches; outputs are cast to fp16 for storage. Empty-string
input is embedded (gives a constant vector `e_empty`); a JSON-null description gives a zero vector
(TwiBot-22) or is embedded as the literal string `"None"` (TwiBot-20 original); see botrgcn.md 3.1.4.

Tables (all fp16 memmaps, row-aligned with `idx`):

| file | shape | content | consumers |
|---|---|---|---|
| `des.f16.npy` | [N, 768] | description embedding | every neural method |
| `tweets_seq.f16.npy` + `tweets_seq_offsets.i64.npy` | [T_sel, 768] | per-tweet embeddings of the selected tweets | BIC; the source for every mean below |
| `tweets_first20_mean.f16.npy` | [N, 768] | mean of the first 20 in release order (TwiBot-22, Cresci-2015 default) | `botrgcn` family, RoBERTa-MLP |
| `tweets_all_mean.f16.npy` | [N, 768] | mean over all (at most 200) tweets (TwiBot-20 original recipe) | `botrgcn` family on TwiBot-20 |

Recommended selection for `tweets_seq`: TwiBot-22 first 20 per user in release order (about 18M
tweets [U]); TwiBot-20 all tweets of all 229,580 nodes (about 33.5M). Derive the means from
`tweets_seq` so the encoder runs once. If disk is tight, keep `tweets_seq` for TwiBot-20 labelled
users only (1,786,245 tweets [P]) and stream means for support users.

Cost on one RTX 4070 Ti (assumption: 2,500 tweets/s for roberta-base fp16 at the observed tweet
lengths; measure on a 100k sample first and update this table):

| item | texts | time | disk (fp16, 1,536 B/row) |
|---|---|---|---|
| TwiBot-22 descriptions | 1.0M (about 0.78M non-empty) | about 5-7 min | 1.5 GB |
| TwiBot-22 tweets, first 20 | about 18M [U] | about 2.0 h | about 28 GB |
| TwiBot-22 tweets, all (at most 200) | up to about 88M | 8-16 h | up to 135 GB (only for the BIC extension; prefer a 50-tweet cap: about 40M, 4.5 h, 61 GB [U]) |
| TwiBot-20 descriptions (distilroberta + roberta) | 2 x 229,580 | about 5 min | 0.7 GB |
| TwiBot-20 tweets, all nodes | about 33.5M | about 3.7 h | 51 GB |
| TwiBot-20 tweets, labelled only | 1.79M | about 12 min | 2.7 GB |

Peak memory of the embedding job: host about 2.5-3 GB (torch + tokenizer + a 50k-tweet text chunk +
output written per batch to the memmap); VRAM about 2-3 GB (fp16 weights 0.25 GB + activations for a
512 x 128-token batch). Reads from the tweet store in chunks, never all texts at once.

Other encoders a method needs are method-local assets, not shared tables: spaCy `en_core_web_lg`
vectors (Lee), GloVe Twitter 200-d (Kudugunta-CLSTM variant only).

## 6. Graph: `artifacts/data/<dataset>/graph/`

- One CSR per relation, `<relation>.csr.npz` with `indptr` int64 [N+1], `indices` int32 [E_r], and
  `edge_pos` int64 [E_r] (the edge's position in the release, so the exact benchmark `edge_index`
  order can be rebuilt). Source to target exactly as in the release; no deduplication, no reverse
  edges, no self-loops at this layer. Derived views (transpose / in-neighbour CSR, undirected union,
  deduplicated simple graph) are built on demand and cached as `<relation>.T.csr.npz`,
  `follow_undirected.csr.npz`.
- `edge_index.i64.npy` [2, E] and `edge_type.i64.npy` [E] in the benchmark's concatenation order and
  type ids, for PyG code (TwiBot-22: `followers` = 0, `following` = 1; TwiBot-20 original:
  following = 0, follower = 1 [P: botrgcn.md 3.1.6]).
- TwiBot-22: stream `edge-002.csv` in 5M-row chunks (pyarrow/pandas with `usecols`), keep only the
  `following` and `followers` relations (2,626,979 + 1,116,655 rows [P-paper]); map ids with
  `searchsorted`. Interaction relations (`retweeted`, `quoted`, `replied`, `mentioned`, `like`, ...)
  are user-tweet or user-user via tweets and are not needed by any selected method; record their
  counts only. I/O about 95 s at 70 MB/s; peak RSS under 1.5 GB. Final size: about 3.74M edges x
  (4 + 8) B = 45 MB.
- TwiBot-20: from each node's `neighbor.following` / `neighbor.follower` lists (only 10,739 labelled
  and 1,185 support users have lists [P: counted]); ids outside the 229,580-node set are dropped.
- Cresci-2015: friends/followers CSVs restricted to labelled users.
- Graph statistics to write into `meta.json` (several specs need them for a sampling dry run):
  degree quantiles and maximum per relation and direction, reciprocal-edge fraction, overlap between
  `following` and reversed `followers`, number of isolated nodes per split.

## 7. Feature schemes

- `botrgcn` (and variants `botrgcn-b20`, `botrgcn-orig6`): defined exactly in botrgcn.md 3.1. Written
  to `artifacts/features/<dataset>/botrgcn[-variant]/` as `num_prop.f32.npy` [N,5], `cat_prop.f32.npy`
  [N,3], links to `des`/`tweets_*_mean`, `edge_index`, `edge_type`, `label`, split indices and
  `meta.json` (z-score mean/std, which are computed over **all N rows** in the benchmark; transductive
  but label-free; reproduce and document). Consumers: BotRGCN, RoBERTa-MLP, GCN/GAT, RGT,
  Simple-HGN/HGT, BotMoE, BotDGT, SEBot, BIC, SEGCN (TwiBot-22).
- Method-local feature builders (Lee, BotHunter, SGBot, Kudugunta, BotMoE `num_for_h`, SEGCN's
  11-flag TwiBot-20 table) read only `profile.parquet`, the tweet store and `users.parquet`.
- Method-local structural artifacts (built by the method, stored under `artifacts/<method>/`): SEBot
  encoding trees (sparse rewrite, numba), BotDGT snapshots (from `created_at`), SEGCN random-walk
  subgraphs. Each spec gives their memory plan.

## 8. ExtScore column: `artifacts/data/<dataset>/extscore/<tag>.parquet`

| column | type | meaning |
|---|---|---|
| `idx` | int64 | row in `users.parquet` (every N row present) |
| `p` | float32 | decision-model P(bot), NaN if not scored |
| `missing` | uint8 | 1 if not scored |
| `fold` | int8 | out-of-fold id for train rows (-1 for val/test/support) |
| optional `q_*` | float32 | sub-question probabilities, same missing rule |

plus `meta.json`: decision-model version, prompt hash, scoring date, coverage per split, and which
users the decision model was optimised on.

Transforms (one implementation in the shared library, `bb.extscore.transform`):
- neural: `[z, m]`, `z = (logit(clip(p, 1e-4, 1 - 1e-4)) - mu) / sd`, mu and sd fitted on observed
  TRAIN rows only, `z := 0` where missing, `m` = missing indicator. Never re-standardised together
  with the other numeric columns.
- trees / AdaBoost / linear: `[p or -1, m]` (or NaN where the library handles it); Kudugunta fills
  missing with the train mean before SMOTE (kudugunta.md 8).

Rules:
- Flag `extscore: none | p` on every method; `none` must reproduce the unmodified baseline exactly (same
  widths, same parameter count, same RNG consumption order where possible).
- Default injection for the `botrgcn` family: append `[z, m]` to `num_prop` (5 to 7) before the
  numeric input projection; each spec also lists a late-fusion ablation and method-specific variants
  (BotMoE: metadata expert input; RF methods: two extra columns).
- Graph nodes without a score (TwiBot-20 support, unscored TwiBot-22 users) get `[0, 1]`.
- Train-row values must be produced the same way as test-row values: out-of-fold or from a decision
  model that never saw those labels. The decision model must never have been optimised on any user
  in the evaluation split (and, for graph methods, on their neighbours either, since neighbours'
  values propagate). The harness reports ExtScore coverage per split and the column-only baseline
  (threshold on `p` chosen on val) next to every "+ExtScore" result.

## 9. Harness requirements (all methods)

- Metrics on val and test: accuracy, bot-class precision, recall, **bot-class F1** (what the
  literature reports on TwiBot-20/22), **macro F1**, ROC-AUC, plus the confusion matrix. Note: the
  benchmark's RoBERTa TwiBot-20 "F1" is human-class F1 (roberta_mlp.md); never mix.
- Seeds: 5 by default (benchmark convention), mean and std. Every run writes config, git hash,
  data-artifact hashes, measured peak RSS/VRAM and wall-clock to `runs/<method>/<dataset>/<run>/`.
- Model selection flags per spec: `select: last_epoch` (what several official scripts effectively do,
  including checkpoint-by-reference bugs) vs `select: best_val` (our clean default where a spec says
  so). Report which.
- Optional emulation flags for known benchmark behaviours (label leakage through sampled neighbours
  in the TwiBot-22 RGT/Simple-HGN/HGT scripts, misaligned boolean features in Kudugunta, HGT's PyG
  2.0.2 bug) default to off, with the clean behaviour reported as our number.

## 10. Build order and total cost

| step | depends on | wall-clock | peak RSS | VRAM |
|---|---|---|---|---|
| users + splits + profile (TwiBot-22, TwiBot-20, Cresci) | raw | under 30 min | under 1.5 GB | 0 |
| TwiBot-22 graph CSR | users | about 5 min | under 1.5 GB | 0 |
| TwiBot-20 graph CSR | users | under 10 min (streaming support.json) | under 1 GB | 0 |
| TwiBot-22 tweet store | users | 2-6 h [U] | under 3 GB | 0 |
| TwiBot-20 tweet store | users | about 15 min | under 1 GB | 0 |
| embeddings TwiBot-22 (des + first 20) | tweet store | about 2.2 h | under 3 GB | about 3 GB |
| embeddings TwiBot-20 (des x 2 + all tweets) | tweet store | about 3.8 h | under 3 GB | about 3 GB |
| `botrgcn` feature scheme (all datasets) | above | under 15 min | under 2 GB | 0 |
| ExtScore tables | decision-model scores (external) | minutes | under 1 GB | 0 |

Total shared precompute about 9-12 h of mostly unattended time, dominated by the TwiBot-22 tweet pass
and the RoBERTa passes. Disk about 100 GB with `tweets_seq` kept for both datasets.
