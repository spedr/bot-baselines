# Architecture: shared data layer and harness

This is the contract between the shared foundation (`bb/`) and the method implementations. Method
agents should read this file, `docs/data_status.md` (what is built), `docs/shared_requirements.md`
(why) and their own `docs/methods/<method>.md`.

## 0. Rules that apply to every job

- Run every preprocessing, embedding, training and evaluation job as
  `tools/memrun [-m 10G] -- uv run python -m bb.methods.<method> ...` (systemd scope with MemoryMax,
  no swap, one heavy job at a time). Design for **<= 8 GB peak RSS**. `--no-lock` only for jobs that
  stay under about 1 GB (the reference method's `torch` import alone costs about 1.3 GB, so it uses
  `-m 2G --no-lock` only as a smoke test).
- Never `json.load`/`torch.load` raw TwiBot-22 files. Everything you need is in `data/` (below) as
  memory maps or column-projected parquet. If something is missing, extend the shared layer (a new
  builder or view) rather than reading raw files from a method.
- Load big tables as memory maps and move them to the GPU with `bb.harness.gpu.memmap_to_gpu`
  (one pinned staging buffer, 50k rows). Never `np.array(memmap)` a full [N,768] table on TwiBot-22.
- Write large outputs with `bb.io.NpyRowWriter` (pwrite + fadvise), **not** a writable `np.memmap`:
  dirty pages of a mapped file are resident and charged to the cgroup. A memmap zero-fill of a 51 GB
  output took a job to 8.4 GB RSS before it was stopped.
- DataLoader workers: 0 by default. Our sampler and memmaps make workers unnecessary.

## 1. Package layout

```
bb/
  paths.py        data_root() (BB_DATA_ROOT, default <repo>/data), runs_root(), raw_dir(name), DatasetPaths
  schema.py       users/profile schemas, value parsers (TwiBot-20 strings, Twitter dates, Cresci flags)
  io.py           atomic writes, NpyAppender (stream rows), NpyRowWriter (pwrite rows), read_rows,
                  load_npy (memmap), file_digest, tree_source_digest, drop_cache
  status.py       StepStatus -> data/status/<step>.json; read_status/all_status
  mem.py          peak_rss_bytes (VmHWM), cgroup_memory, cuda_peak_bytes, memory_report
  data/           LOADERS (read-only views)
    users.py        load_users, load_split, load_labels, load_meta, num_users, id_lookup, assert_disjoint
    profile.py      load_profile(columns=...), iter_profile, profile_meta
    tweets.py       TweetStore
    emb.py          load_embedding, load_tweet_seq, botrgcn_text, embedding_meta
    graph.py        CSR, load_csr, load_transpose, load_in_merged, load_undirected, load_edge_index,
                    graph_meta, graph_stats, build_csr_from_coo
  build/          BUILDERS (raw -> data/), each idempotent, with a status file
    common.py       JsonArrayStream (streaming JSON array reader), ProfileWriter, write_index
    twibot20.py     one pass over the four release files
    twibot22.py     steps users | graph | tweets
    cresci15.py     from cresci-2015.csv.tar.gz (order + split verified against MGTAB tensors)
    embed.py        Encoder + run_embed + make_means
  extscore.py          ExtScore column storage, transforms, guard, common CLI interface, synthetic generator
  harness/
    metrics.py      compute_metrics, roc_auc, summarize
    train.py        set_seed, EarlyStopper, BestCheckpoint, class_weights
    run.py          Run (run directory + result.json), load_results, aggregate, data_digests
    sampler.py      NeighborSampler (numba, on-disk CSR, bounded fan-out, dry_run)
    gpu.py          memmap_to_gpu, gather_rows
  methods/
    logreg_ref.py   tiny reference method wired through everything (copy its structure)
  verify.py       `bb verify <dataset>` invariant and count checks
  cli.py          `bb build|embed|status|verify|extscore-synth`
tools/memrun        memory-capped, lock-serialised job runner
tools/precompute.sh embedding chain (idempotent; re-run after an interruption)
tests/              pytest on synthetic fixtures (no real data needed)
```

Method code goes in `bb/methods/<method>.py`, or in a `bb/methods/<method>/` package with
`__main__.py`. It runs as `python -m bb.methods.<method> --dataset <ds> [--seeds ...] [--extscore ...]`.
Method-local artifacts (SEBot trees, BotDGT snapshots, SEGCN walks, spaCy vectors) go under
`DatasetPaths(ds).method_dir("<method>")` = `data/methods/<method>/<dataset>/`. Feature schemes
shared by several methods (the `botrgcn` scheme) go under `DatasetPaths(ds).features("botrgcn")`
= `data/features/<dataset>/botrgcn/`. The BotRGCN agent owns that scheme.

## 2. File layout per dataset (`data/<dataset>/`)

| path | contents |
|---|---|
| `users.parquet` | canonical index, schema below; row i has `idx == i` |
| `splits/{train,val,test,support}_idx.npy` | int64 idx arrays (support only if non-empty) |
| `meta.json` | counts and class balance per split, index order, source digests, dataset-specific checks |
| `profile.parquet` + `profile.meta.json` | typed profile (row-aligned); meta lists `absent_columns` and null counts |
| `tweets/` | ragged tweet store (section 5) |
| `emb/<encoder>/` | fp16 embedding tables (section 6) |
| `graph/<relation>.csr/` | `indptr.npy` int64 [N+1], `indices.npy` int32 [E], `edge_pos.npy` int64 [E], `meta.json` |
| `graph/edge_index.i64.npy`, `edge_type.i64.npy`, `edge_pos.i64.npy` | benchmark-order COO of all user-user relations |
| `graph/<rel>.T.csr/`, `in_merged__<r1>+<r2>.csr/`, `undirected__<r1>+<r2>.csr/` | cached derived views (built on first use) |
| `graph/meta.json` | relation counts, edge-type ids, release semantics, degree stats |
| `extscore/<tag>.parquet` + `<tag>.meta.json` | ExtScore columns (section 7) |

Deviation from `shared_requirements.md`: the spec's `artifacts/data/<dataset>/...` is `data/<dataset>/...`,
for a shorter layout. CSRs are directories of `.npy` files rather than `.npz`, so that
they can be memory-mapped. `n_tweets_total` lives in `tweets/n_total.i32.npy` rather than
`users.parquet`, so the index stays immutable after its build step.

## 3. Canonical index (`bb.data.users`)

`users.parquet` columns: `idx` int64, `user_id` str (digits), `raw_id` str (as released; TwiBot-22
`u123`), `split` category (`train`/`val`/`test`/`support`), `label` int8 (1 bot, 0 human, -1
unlabelled), `raw_order` int64 (position in the raw user file; TwiBot-20 within its source file),
`split_csv_order`, `label_csv_order` int64 (TwiBot-22 rows in split.csv/label.csv, else -1),
`source_file` category.

Order (benchmark-compatible; index-based code paths and emulated bugs line up):
- TwiBot-22: `user.json` order (= `label.csv` order, verified; `split.csv` differs).
- TwiBot-20: train_bak (0..8277), dev (8278..10642), test_bak (10643..11825), support (11826..229579).
- Cresci-2015: subsets E13, TFP, INT, TWT, FSF, each in users.csv order. This equals the node order
  of the BotRGCN-format tensors (MGTAB repo) row for row. The split is that tensor set's
  per-subset contiguous 70/20/10 split.

```python
from bb.data import load_users, load_split, load_labels, assert_disjoint
y = load_labels("twibot22")                 # int8 [N]
tr = load_split("twibot22", "train")       # int64 memmap
assert_disjoint(tr, eval_idx)              # raises LeakageError (e.g. BotSay-340 vs official train)
```

## 4. Profile table (`bb.data.profile`)

`load_profile(ds, columns=[...], rows=None, arrow=False)` returns pandas with nullable dtypes
(`Int64`, `boolean`, `string[pyarrow]`, UTC timestamps), so null stays distinct from 0/False/''.
Always project columns on TwiBot-22. `iter_profile(ds, columns, batch_rows)` streams.

Columns (all datasets share the schema; a field the dataset lacks is all-null and listed in
`profile.meta.json["absent_columns"]`): `profile_missing`, `name`, `screen_name` (TwiBot-22
`username`), `description`, `url`, `location`, `profile_image_url`, `created_at` (UTC),
`followers_count`, `following_count` (= `friends_count`), `statuses_count` (= `tweet_count`),
`listed_count`, `favourites_count`, `protected`, `verified`, `default_profile`,
`default_profile_image`, `geo_enabled`, `profile_use_background_image`, `profile_background_tile`,
`has_extended_profile`, `contributors_enabled`, `is_translator`, `is_translation_enabled`, `lang`,
`utc_offset`, `time_zone`, `pinned_tweet_id`, `withheld`, `entities` (JSON string), plus exact raw
spellings `raw_name`, `raw_screen_name`, `raw_description`, `raw_location`,
`raw_profile_image_url`, `raw_default_profile_image`, `raw_url`, `raw_protected`, `raw_verified`,
`raw_created_at`.

Parsing: TwiBot-20 strings are stripped, `"None"` -> null, `"True"/"False"` -> bool, `" "` -> `''`.
Cresci: unquoted empty CSV field -> null, quoted `""` -> `''`, `NULL` -> null. TwiBot-22: JSON
values as is (no description is null in the release; 124,230 are `''`). Reference dates for
"age" features are per-method constants and never profile columns.

## 5. Tweet store (`bb.data.TweetStore`)

```
tweets/offsets.i64.npy [N+1]      user i -> rows offsets[i]:offsets[i+1]
tweets/text.bin + text_offsets.i64.npy [T+1]
tweets/<col>.npy [T]              per-tweet columns, -1 = null
tweets/n_total.i32.npy [N]        tweets in the release incl. beyond the cap
tweets/user_flags.npy [N]         (TwiBot-20) bit 0 = `tweet: null`
```

Rows within a user are in **release order**: TwiBot-20 = the user's `tweet` list (newest first);
TwiBot-22 = order of appearance in tweet_0..8.json. Measured over all 76.9M stored rows: 97.2% of
consecutive within-user pairs have non-increasing `created_at`, so release order is mostly newest
first. Only 33% of users have their stored tweets in one contiguous release block; the others are
split over several blocks of up to about 40. Cresci = tweets.csv order. `release_rank` re-sorts.

| dataset | cap | columns |
|---|---|---|
| TwiBot-22 | first 200 per user (release order) | release_rank (file*1e9 + position), tweet_id, created_at (unix s), retweet/reply/like/quote_count, lang (code, vocab in meta), source (code, `source_vocab.json`), n_hashtags, n_urls, n_mentions, is_retweet, is_quote, is_reply, possibly_sensitive, text_null |
| TwiBot-20 | none (release has at most 200; 201 for 3 users) | release_rank, text_null (no per-tweet metadata exists; derive from text) |
| Cresci-2015 | first 200 per user (csv order) | release_rank, tweet_id, created_at, retweet/reply/like_count, n_hashtags/urls/mentions, is_retweet, is_reply, text_null |

TwiBot-22 caveats: `referenced_tweets` is null for 97.9% of tweets. Where it is null, `is_retweet`
means the text starts with `RT @` and `is_quote` is -1 (unknown). `reply_count`/`quote_count` are
null throughout the inspected sample.

```python
t = TweetStore("twibot22")
t.user_texts(i, cap=20)                      # first 20 in release order
rows, off = t.selection(cap=20)              # [T_sel] rows + [N+1] offsets (botrgcn first-20)
for users, off, texts, cols in t.iter_users(chunk_users=20_000, cap=200, columns=("is_retweet",)):
    ...                                      # bounded memory; text read with one pread per user run
t.column("created_at"); t.n_total(); t.tweet_list_null()
```

## 6. Embeddings (`bb.data.emb`, built by `bb embed`)

Encoder: HF `roberta-base` / `distilroberta-base` (commit hash in `<name>.meta.json`), fp16 autocast,
length-bucketed batches with lengths padded to a multiple of 16. Pooling: last hidden layer,
arithmetic mean over **all** positions including `<s>`/`</s>`, pads excluded (botrgcn.md 3.1.4).
Truncation at 512 tokens (`--max-tokens`). Cosine similarity with an fp32 unbatched reference is
0.9999998.

| file | shape | policy |
|---|---|---|
| `des.f16.npy` | [N,768] | TwiBot-22/Cresci: null -> zero vector, `''` embedded (= `e_empty`). TwiBot-20: stripped raw string, null/`"None "` -> literal `"None"` |
| `tweets_seq_<sel>.f16.npy` + `_offsets.i64.npy` [N+1] + `_rows.i64.npy` [T] | [T,768] | one row per selected tweet (`first20`: first 20 in release order; `all`) |
| `tweets_first20_mean.f16.npy` | [N,768] | mean of the first <= 20; null text counts as a zero vector; no tweets -> zeros |
| `tweets_all_mean.f16.npy` | [N,768] | (TwiBot-20) mean of all; `tweet: null` -> `e_empty` (original Dataset.py) |
| `e_empty.f16.npy` | [768] | embedding of `''` |

```python
from bb.data import botrgcn_text, load_embedding, load_tweet_seq
des, tw = botrgcn_text("twibot20")          # distilroberta des + roberta all-tweet mean (benchmark default)
seq, off, rows = load_tweet_seq("twibot20", "roberta-base", "all")   # BIC per-tweet sequence
from bb.harness.gpu import memmap_to_gpu
des_gpu = memmap_to_gpu(des)                # fp16 on GPU, host RSS ~ one 50k-row chunk
```

## 7. Graph (`bb.data.graph`)

Base relations exactly as released (source -> target, no dedup, no reverse edges, no self-loops
added; rows in release order; `edge_pos` = release position):

| dataset | relations (edge_type id) | semantics |
|---|---|---|
| TwiBot-22 | `followers` (0), `following` (1) | edge-002.csv rows; `following`: source follows target |
| TwiBot-20 | `following` (0), `follower` (1) | owner i -> j in `i.neighbor.following` / `.follower`; `edge_index` order = original Dataset.py loop |
| Cresci-2015 | `friend` (0), `follower` (1) | friends.csv / followers.csv rows; both ends labelled users. Equals the MGTAB tensors' edge multiset with the same type ids |

Views: `load_transpose(ds, rel)` (in-neighbours), `load_in_merged(ds, [rels])` (CSC of the merged
graph with `etype` per entry; the sampler input), `load_undirected(ds, [rels])` (simple undirected:
SEGCN, SEBot trees, BotDGT LCC), `load_edge_index(ds)` (PyG `edge_index`/`edge_type`). Views are
cached on disk at first use.

## 8. ExtScore column (`bb.extscore`)

Table `extscore/<tag>.parquet`: `idx`, `p` float32 (NaN = unscored), `missing` uint8, `fold` int8 (-1 =
not out-of-fold), optional `q_*`; `<tag>.meta.json` holds `label_blind`, `oof`, `optimised_on` (idx
list), `model`, `prompt_hash`, `scored_at`. Write it with
`bb.extscore.write_extscore(ds, tag, idx, p, fold=..., meta=...)`. Every unscored row is stored as missing.

**One common interface for all methods:**

```python
from bb.extscore import add_extscore_args, prepare_extscore
add_extscore_args(parser)        # --extscore none|p, --extscore-tag TAG, --with-extscore TAG, --extscore-unsafe
J = prepare_extscore(args, ds, {"train": tr, "val": va, "test": te}, labels=y,
                neighbours=load_undirected(ds, rels))   # neighbours only for graph methods
x_num = J.append(x_num, rows=None, kind="neural")   # [z, m] appended; returns x unchanged if --extscore none
x_rf  = J.append(x_rf, rows=idx, kind="tree")       # [p or -1, m]
J.width                     # 0 or 2: create the model with base_width + J.width (create nothing extra when 0)
run.finish(metrics, extscore=J.report())                  # coverage, transform params, guard result,
                                                     # column-only baseline (val-tuned threshold) on val/test
```

- `neural`: z = (logit(clip(p, 1e-4, 1-1e-4)) - mu) / sd with mu, sd fitted on **observed train
  rows only**; z = 0 and m = 1 where missing. Do not z-score it again with other columns.
- `tree` `[p or -1, m]`, `meanfill` `[p or train-mean, m]` (Kudugunta), `nan_transform` (NaN-native).
- Guard (`validate_extscore`, enforced by `prepare_extscore`): observed train rows must have `fold >= 0` or
  the decision model must be `label_blind`; `optimised_on` must not touch val/test users or, if
  `neighbours` is given, their graph neighbours. Violations raise `ExtScoreLeakageError` (`--extscore-unsafe`
  downgrades them to a recorded warning).
- `extscore none` must reproduce the baseline exactly. Construct the ExtScore-free model with the same widths
  and parameter-creation order (the reference method checks this in tests).
- No real ExtScore values exist yet. `bb extscore-synth <ds> --tag synthetic` (or `make_synthetic_extscore`) writes
  a synthetic column (p = sigmoid(signal*(2y-1) + noise), coverage < 1, folds on train) for tests and
  smoke runs. Never report numbers from it.

## 9. Harness (`bb.harness`)

```python
from bb.harness import Run, compute_metrics, EarlyStopper, BestCheckpoint, class_weights, set_seed, aggregate, load_results
set_seed(seed)
es, ck = EarlyStopper(patience=..., mode="max"), BestCheckpoint(select="best_val" | "last_epoch")
with Run(method, ds, config=vars(args), seed=seed, data_files=[...emb paths...]) as run:
    for epoch ...:
        run.log_epoch(epoch, train_loss=..., val_f1=...)
        ck.update(model, val_value, epoch); es.step(val_value)
        if es.should_stop: break
    ck.restore(model)                              # deep copy; never the by-reference bug
    run.add_predictions("test", te, prob_te, y[te])
    run.finish({"val": compute_metrics(y[va], p_va), "test": compute_metrics(y[te], p_te)},
               selection=ck.info(), extscore=J.report())
agg = aggregate(load_results(method, ds, where={"with_extscore": None}))   # mean/std over seeds
```

- `compute_metrics(y, prob, threshold=0.5)`: accuracy, bot-class precision/recall/**f1**,
  `f1_human`, `macro_f1`, MCC, ROC-AUC from probabilities (tie-correct), confusion. Labels must be
  0/1.
- `runs/<method>/<dataset>/<run>/`: `config.json`, `epochs.jsonl`, `preds_<split>.npz`, `result.json`
  (status, metrics, git-free provenance hash = sha256 over `bb/` + method sources + config + data
  digests, peak RSS/VmHWM, cgroup peak, CUDA max allocated, wall time, ExtScore report, selection).
- `class_weights(y_train, "none" | "balanced" | "sqrt" | "w0,w1")`.
- Neighbour sampling: `NeighborSampler(load_in_merged(ds, rels), fanouts, max_nodes=None)`.
  `.sample(seeds, rng_seed)` returns local `edge_index` (row 0 = neighbour/source, row 1 = target),
  `n_id` (seeds first), `e_id` (index into the CSC: use `csc.etype[e_id]`, `csc.edge_pos[e_id]`).
  `.iter_batches(seeds, bs, epoch_seed)` is deterministic, so epochs can be re-sampled instead of
  cached. `.dry_run(seeds, bs, n_batches=200, feature_bytes_per_node=...)` reports subgraph-size
  percentiles; run it before training. PyG `NeighborLoader` (pyg-lib 0.9.0 is installed) is a valid
  alternative when a method wants it. Build its `edge_index` from `load_edge_index`.

## 10. Library choices

- Python 3.12, torch 2.14.0+cu130 (driver 610.88, CUDA 13.3 UMD), torch_geometric 2.8.1 + pyg-lib
  0.9.0, numpy 2.3, pandas 3.1, pyarrow 26, numba 0.68, transformers 5.22 (pinned in `uv.lock`).
- PyG over pure-torch sparse ops: 9 of the 13 methods are specified in terms of PyG modules
  (RGCNConv, GCNConv/GATConv, TransformerConv, HGTConv, NeighborLoader, k_hop_subgraph, softmax), and
  their faithful-reproduction notes (PyG 2.0.2 HGT bug, flow conventions) assume them. The shared
  layer itself stays library-neutral: plain CSR + numba sampler on memmaps.
- No Rust/C++ for raw parsing. Measured: drvfs reads at 45-120 MB/s, Python `raw_decode` with full
  field extraction at 134 MB/s (115k tweets/s), pyarrow edge CSV at 303 MB/s. The passes are
  I/O-bound, so a native parser would not shorten them (see data_status.md).
- HF models download to `data/hf` (`HF_HOME`), inside the repo.

## 11. Adding a method (checklist)

1. `bb/methods/<method>.py` with `main(argv)`. Use `add_extscore_args` and `prepare_extscore`, and create the
   model with `base_width + J.width`.
2. Read only via `bb.data` loaders. Put method-local artifacts in `DatasetPaths(ds).method_dir(...)`
   and write them with `NpyRowWriter`/`NpyAppender`, plus a `StepStatus` if they are long.
3. Train with `set_seed`, `EarlyStopper`/`BestCheckpoint` (or `select=last_epoch` where the spec says
   the official code reports the last epoch), and evaluate with `compute_metrics` on 0/1 labels only.
4. Wrap each seed in `Run(...)` and pass `data_files=[...]` for the embeddings and features used.
5. Add a test on the synthetic fixtures (`tests/fixtures.py`: `tw20`/`tw22`/`cr15` pytest fixtures).
6. Run real jobs under `tools/memrun`. Check the result's `memory.peak_rss_bytes`.

## 12. Shared `botrgcn` feature scheme and GNN skeleton (`bb.features`, `bb.gnn`)

Owned by the BotRGCN implementation; GCN/GAT, RGT, Simple-HGN/HGT, BotMoE, BotDGT, SEBot, BIC and
SEGCN read the features, and GCN/GAT/RGT/Simple-HGN can reuse the skeleton. Spec:
`docs/methods/botrgcn.md` 3.1 (features) and 4 (model).

### 12.1 Features: `bb/features/botrgcn.py`

Built per (dataset, variant) into `DatasetPaths(ds).features(variant)` =
`data/features/<dataset>/<variant>/` (the `features()` path helper of `bb.paths`; the variant is the
directory name, e.g. `data/features/twibot20/botrgcn-b20/`):

| file | content |
|---|---|
| `num_prop.f32.npy` [N,k] | numeric properties, missing filled, then z-scored over **all N rows** (ddof 1; transductive, label-free, as the benchmark) |
| `cat_prop.f32.npy` [N,c] | 0/1 flags, no normalisation |
| `des.f16.npy`, `tweets.f16.npy` [N,768] | relative symlinks to `data/<ds>/emb/<encoder>/...` (no copy) |
| `edge_index.i64.npy` [2,E], `edge_type.i64.npy` [E] | symlinks to `data/<ds>/graph/` (release order and direction; type ids in `meta.json["edge_type_ids"]`) |
| `label.i64.npy` [N] | 1 bot, 0 human, -1 unlabelled (TwiBot-20 support) |
| `{train,val,test[,support]}_idx.i64.npy` | copies of the shared splits |
| `meta.json` | column names, z-score mean/std, fills, ref date, text sources + HF commits, edge counts per type, checks, `input_key`, `text_ready` |

Variants (`bb.features.VARIANTS`): `twibot22/botrgcn` (num: followers, active_days@2020-09-05 fill 1,
len(name) [benchmark bug kept], following, tweet_count; cat: protected, verified,
default_profile_image by URL), `twibot22/botrgcn-uname` (len(username) ablation),
`twibot20/botrgcn` (followers, active_days@2020-09-01, len(screen_name), friends, statuses; protected,
verified, default_profile_image; distilroberta des + roberta all-tweet mean), `twibot20/botrgcn-b20`
(roberta des + first-20 mean), `twibot20/botrgcn-orig6` (6 numeric + 11 flags of the original
Dataset.py), `cresci15/botrgcn` (followers, active_days, len(screen_name), friends, listed; one flag;
equal to the MGTAB BotRGCN tensors row by row, num max abs err 2.4e-7, cat identical).

```python
from bb.features import load_features, build_features
F = load_features("twibot22")            # builds if missing/stale; memmaps; raises if text not built
F.num_prop, F.cat_prop, F.des, F.tweets, F.edge_index, F.edge_type, F.label, F.splits, F.meta
F.relations_by_type()                    # ['followers', 'following'] (etype order, for load_in_merged)
F.files                                  # for Run(data_files=...)
load_features("twibot22", require_text=False)    # num/cat/graph before the embeddings exist
```

Build: `tools/memrun --light -- uv run python -m bb.features.botrgcn --dataset <ds> [--variant all]`
(TwiBot-22: 0.5 s, 1.1 GB peak RSS for all five variants together). Idempotent: `meta.json["input_key"]`
hashes the builder version, the variant and digests of users/profile/graph/text inputs; an equal key
is a no-op, a changed input (e.g. text tables landing later) rebuilds. The directory is replaced with
a rename, so a reader never sees a half-written set (open memmaps stay valid). ExtScore is **not** stored
here: methods append it at load time through `J.append(..., kind="neural")` (below).

### 12.2 Skeleton: `bb/gnn/`

- `to_device(F, device, J=None, extscore_inject="num", text_dtype=None, edges=True)` -> `GraphInputs`
  (`des`, `tweet`, `num`, `cat`, `edge_index`, `edge_type`, `y`, `extra`). Whole text tables are copied with
  plain reads (`table_to_device`: pread + FADV_DONTNEED through one pinned 50k-row buffer; before
  review round 1 a wrong memmap-offset test sent every table through `memmap_to_gpu` instead), so host
  RSS does not grow with the table; they stay **float16 on CUDA** (float32 on CPU). Host arrays are
  copied (never `from_numpy` on a read-only memmap: an in-place write would segfault on CPU).
  ExtScore: `"num"` -> `num = J.append(num_prop, kind="neural")` (k -> k + 2, the default for the family);
  `"branch"`/`"late"` -> `extra = [z, m]`. With `--extscore none` nothing changes.
  `GraphInputs.nodes(n_id).with_edges(ei, et)` builds a sampled subgraph.
- `make_classifier(body_factory, hidden=D, num_dim=k + J.width, cat_dim=c, extscore_width=J.width,
  extscore_inject=...)` -> `GNNClassifier`: `InputEncoder` (Linear+LeakyReLU to D/4 for des, tweet, num,
  cat [+ ExtScore branch], concat, Linear+LeakyReLU to D) -> body -> `OutputHead` (Linear+LeakyReLU D->D,
  Linear D[+2 late ExtScore]->2). A body is any module `forward(x, edge_index, edge_type) -> x`;
  `body_factory` is called after the encoder so parameter creation follows BotRGCN's official order.
  The 768-wide projections run their forward GEMM in fp16 on fp16 inputs with a **float32 backward**
  (`bb.gnn.model.text_linear` / `_HalfInputLinear`; `_text_proj(lin, x)` is the same for an
  `nn.Linear`), the rest in fp32. Do not differentiate a plain fp16 `Fn.linear` instead: without loss
  scaling the float32 upstream gradient is cast to fp16 and, with a loss averaged over 7e5 TwiBot-22
  train rows, underflows (review round 1: 92% of per-row gradients flushed to zero after 100 epochs,
  weight-gradient error 15% des / 43% tweet; TwiBot-20 0.2%).
  `kaiming_init_linears(model)` = BotRGCN's `init_weights` (kaiming_uniform_ on `nn.Linear` weights
  only). Extra knobs: `input_dropout` (GCN_GAT-folder variants), `body_out` (body width != D).
- `FitConfig` + `fit_full_batch(model, g, splits, cfg, run=None)` (one step per epoch over all
  nodes; val metrics logged every `eval_every` epochs; `select="last_epoch"` default or `"best_val"`
  with `BestCheckpoint` deep copy and optional `patience`) and `fit_sampled(model, g, splits, cfg,
  csc=load_in_merged(ds, F.relations_by_type()))` (`NeighborSampler` mini-batches with
  `cfg.fanouts`; val/test predicted with fan-out -1, exact for an L-layer body when
  `len(fanouts) == L`). Both return `FitResult(prob [N] (NaN where not predicted), epochs_run,
  selection, train_seconds, seconds_per_epoch, final_train_loss, history)`;
  `split_metrics(prob, y, splits)` gives `compute_metrics` per split. Loss uses train rows only;
  tests assert that flipping every val/test label leaves the trained model unchanged.

- `bb.gnn.RGCNConv(in, out, num_relations)`: **use this instead of `torch_geometric.nn.RGCNConv`.**
  PyG 2.8.1 picks, by an input-size heuristic, a `segment_matmul` path whose `aggr='mean'` averages
  over all incoming edges across relations (`message()` carries "TODO Re-weight according to edge
  type degree") instead of R-GCN's per-relation mean then sum. With 2 relations and D=32 the heuristic
  takes that path once the largest relation has more than about 60k edges: TwiBot-20 (117k) and
  TwiBot-22 (2.6M) are affected, Cresci-2015 (7k) is not. The wrapper (PyG layer at `.conv`, same
  parameters and init) sets `torch_geometric.backend.use_segment_matmul = False` for each call and
  restores the flag. `tests/methods/test_botrgcn.py::test_rgcnconv_per_relation_mean_on_both_heuristic_paths`
  checks both heuristic outcomes and both forced flags against a hand-written per-relation reference
  (and that plain PyG really deviates on the segment_matmul path). Any method on PyG's `RGCNConv`
  (RGT/Simple-HGN baselines, GCN_GAT-folder BotRGCN, BotMoE) needs the same pin.
