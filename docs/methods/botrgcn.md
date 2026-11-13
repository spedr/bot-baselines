# BotRGCN (and BotRGCN "w/o graph" MLP ablation) + shared feature scheme `botrgcn`

Status: spec, written 2026-11-12. Evidence labels: **[P-paper]** read in the paper, **[P-code]** read in
official code (path given), **[S]** secondary source, **[U]** unverified / assumed / derived by us.

## 1. Summary

BotRGCN is a transductive, full-graph node classifier. Every user is a node with four feature blocks
(RoBERTa description embedding, mean RoBERTa tweet embedding, z-scored numeric profile counts, binary
profile flags). Each block is projected by Linear+LeakyReLU to D/4, the four are concatenated, passed
through Linear+LeakyReLU, then **two applications of one shared R-GCN layer** over a two-relation
user-user follow graph (following / follower), then Linear+LeakyReLU and a 2-way linear classifier.
Family: graph neural network. Graph: yes (2 relations, user-user only). Text: yes, frozen encoder
features only (no fine-tuning). Temporal: no (only account age). Encoders: `roberta-base`
(tweets; descriptions in the benchmark) and `distilroberta-base` (descriptions in the original repo).
The "w/o graph" variant replaces the R-GCN layers by plain linear layers (an MLP on the same four blocks).

The feature tensors defined here (section 3.1, "Shared feature scheme `botrgcn`") are reused by RGT,
SimpleHGN, HGT, GCN, GAT, BotMoE, SEBot, BotDGT and others. Build them once per dataset.

## 2. Citation and code

- Paper: Shangbin Feng, Herun Wan, Ningnan Wang, Minnan Luo. "BotRGCN: Twitter Bot Detection with
  Relational Graph Convolutional Networks". ASONAM 2021, pp. 236-239. DOI 10.1145/3487351.3488336,
  arXiv 2106.13092 (v4 read).
- Benchmark re-run: Feng et al., "TwiBot-22: Towards Graph-Based Twitter Bot Detection", NeurIPS 2022
  Datasets and Benchmarks, arXiv 2206.04564 (Tables 2, 3, 8; App. B.4).
- R-GCN layer: Schlichtkrull et al., ESWC 2018, DOI 10.1007/978-3-319-93417-4_38.
- Official code (original, TwiBot-20 raw json): `BunsenFeng/BotRGCN`, commit
  `eabe7fdfc73019144ce3ea50edbd78dd24857031` (2022-11-07), MIT. Files: `Dataset.py`, `model.py`,
  `main.py`, `utils.py`.
- Benchmark code: `LuoUndergradXJTU/TwiBot-22`, commit `54adf9a1dbb6dc238c932d3147a1875def5817d8`
  (2025-10-14), MIT. Files: `src/BotRGCN/{twibot_22,twibot_20,cresci_15}/{preprocess_1.py,
  preprocess_2.py,model.py,train.py,Dataset.py,utils.py,dataset_tool.py}`, `src/BotRGCN/readme.md`,
  parallel feature scripts `src/twibot22_Botrgcn_feature/{twi0..10.py,tweet0..8.py,des_embedding.py}`.
- Licence: both MIT. We may read the code and reuse ideas with attribution; we write our own code.

## 3. Inputs

### 3.0 What BotRGCN needs from the shared data layer

| Need | Source in shared layer | Notes |
|---|---|---|
| node list, order, split, label | `users.parquet` (`idx`, `split`, `label`) | TwiBot-20: 229,580 nodes = 11,826 labelled + 217,754 support; TwiBot-22: 1,000,000 all labelled |
| numeric fields | `profile.parquet`: `followers_count`, `following_count` (= `friends_count`), `statuses_count` (TwiBot-22: `tweet_count`), `listed_count` (variant only), `created_at`, `screen_name` (TwiBot-20) / `name` (TwiBot-22, see 3.1.3) | raw values, null where missing |
| flags | `profile.parquet`: `protected`, `verified`, `default_profile_image` (TwiBot-20) / `profile_image_url` (TwiBot-22) | TwiBot-22 flag is derived from the URL |
| description text | `profile.parquet.description` | must distinguish JSON null from empty string for TwiBot-22 (see 3.1.4) |
| tweet texts | ragged tweet store | **TwiBot-22 needs release-file order** (order of appearance in `tweet_0..8.json`), first 20; TwiBot-20 original recipe needs all tweets (<=200) in release order. See 3.1.5 |
| embeddings | `emb/roberta-base/des.f16.npy`, `emb/roberta-base/tweets_first20.f16.npy` (TwiBot-22), `emb/distilroberta-base/des.f16.npy` + `emb/roberta-base/tweets_all.f16.npy` (TwiBot-20 original recipe) | pooling defined in 3.1.4/3.1.5, which is **not** the usual masked mean that excludes special tokens |
| graph | `graph/following.csr.npz`, `graph/followers.csr.npz` (TwiBot-22), `graph/following.csr.npz`, `graph/follower.csr.npz` (TwiBot-20) | source -> target exactly as in the release, see 3.1.6 |
| ExtScore | `extscore/<tag>.parquet` | section 8 |

The shared layer's default tweet store is "newest first"; for TwiBot-20 the release order already is
"the recent 200 tweets" (README), so newest-first = release order **if** the store keeps the order of
`train_bak.json` / `dev.json` / `test_bak.json` / `support.json` (the local `train.json`/`test.json`
are embedding-sorted copies made by another project; never use them; the unified manifest already
documents this). For TwiBot-22 the per-tweet order in `tweet_i.json` is not known to be chronological
[U]; the store must keep a `release_rank` (file index, position) per tweet so "first 20 in release
order" can be rebuilt. If that is impossible, use newest 20 and record the deviation.

### 3.1 Shared feature scheme `botrgcn`

Three recipes exist and they differ. We define four named variants; the default for each dataset is
marked. All tensors are row-aligned with `users.parquet.idx`.

| Variant | Dataset | num_prop (dim) | cat_prop (dim) | des encoder | tweets | max tokens |
|---|---|---|---|---|---|---|
| `botrgcn` (**default TwiBot-22**) | TwiBot-22 | 5 | 3 | roberta-base | first 20, release order | 512 effective (see 3.1.7) |
| `botrgcn` (**default TwiBot-20**) | TwiBot-20 | 5 | 3 | distilroberta-base | all (<=200), release order | 512 effective |
| `botrgcn-b20` | TwiBot-20 | 5 | 3 | roberta-base | first 20, release order | 512 |
| `botrgcn-orig6` | TwiBot-20 | 6 | 11 | distilroberta-base | all | 512 |
| `botrgcn` | Cresci-2015 (benchmark split only) | 5 | 1 | roberta-base | first 20 | 512 |

Why the TwiBot-20 default mixes sources: the original repo's `main.py` builds the model with
`num_prop_size=5, cat_prop_size=3` [P-code `BotRGCN/main.py` l.16], but its `Dataset.py` writes 6
numeric and 11 categorical columns [P-code `Dataset.py` l.251, l.266]; so the shipped (Google Drive)
TwiBot-20 tensors that produced the paper numbers were **not** made by the posted `Dataset.py` as is.
The benchmark's `twibot_20/preprocess_1.py` cannot have produced its tensors either: it reshapes the
categorical tensor to `[5301,1]` (a Cresci-2015 leftover, l.183) and would crash on 229,580 users, and
it takes `listed_count` as the "statuses" column (l.95). The benchmark README states the 5 and 3
columns [P-code `src/BotRGCN/readme.md` l.48-72]; we follow it. Text recipe for TwiBot-20 follows the
original `Dataset.py` (the only TwiBot-20 text code that runs) [U: that the Drive tensors match it].
**Strongly recommended check:** obtain the authors' Drive zip
(`https://drive.google.com/file/d/14X1yg-TK_5-On1JfHop6off5_bLzrRvA`, linked in the original README)
and compare our tensors row by row (cosine of des/tweets rows; correlation of each num column with
candidate raw fields to identify the 5 columns). If the zip is available, its tensors are the most
faithful TwiBot-20 input of all.

#### 3.1.1 Node set and index order

- TwiBot-22: all 1,000,000 users of `user.json`, in `user.json` order [P-code
  `twibot_22/preprocess_1.py` l.12-19]. All are labelled (bot 139,943 / human 860,057). Splits from
  `split.csv` (`train`/`val`/`test` = 700,000/200,000/100,000; the file uses `val`, verified locally).
  Test has 29,444 bots / 70,556 humans; train 54,586 / 645,414; val 55,913 / 144,087 (computed
  locally from `label.csv` + `split.csv`). No support nodes.
  `user.json` order vs `label.csv` order: the benchmark RoBERTa script indexes BotRGCN tensors with
  `label.csv` positions and still gets precision 0.63 on a 29% base rate, which implies the orders
  agree [U, inferred]; order does not matter for us as long as all tensors use `users.parquet.idx`.
- TwiBot-20 (original): `train` (8,278) then `dev` (2,365) then `test` (1,183) then `support`
  (217,754) = 229,580 rows, each in file order [P-code `Dataset.py` l.32-33, l.329-333]. Benchmark:
  same order (it hard-codes `train_idx = 0..8277`, `val_idx = 8278..10642`, `test_idx = 10643..11825`
  [P-code `src/GCN_GAT/dataset.py` l.43-47]). The shared layer's "labelled first, support after"
  order matches if labelled users are kept as train, dev, test in file order.
- Labels: bot = 1, human = 0. TwiBot-20 support nodes get label -1 (never in the loss).

#### 3.1.2 Numeric properties (`num_properties_tensor`, float32 [N, 5])

Column order used by the benchmark (TwiBot-22, Cresci-2015, and the intended TwiBot-20) [P-code
`twibot_22/preprocess_1.py` l.158]:

| col | name in code | TwiBot-22 raw field | TwiBot-20 raw field (default) | missing value |
|---|---|---|---|---|
| 0 | followers_count | `public_metrics.followers_count` | `profile.followers_count` | 0 |
| 1 | active_days | `(ref - created_at).days`, ref = 2020-09-05 00:00 UTC | same, ref = 2020-09-01 00:00 UTC | TwiBot-22: 1 (`fillna(1)`, l.127); TwiBot-20: 0 |
| 2 | screen_name_length | **`len(name)`** (display name, l.130) | `len(screen_name)` | 0 |
| 3 | following_count | `public_metrics.following_count` | `profile.friends_count` | 0 |
| 4 | statuses | `public_metrics.tweet_count` | `profile.statuses_count` | 0 |

- Transform: **none** (no log). Raw counts are z-scored directly:
  `x' = (x - mean) / std` with pandas `std` (ddof = 1), mean/std over **all N rows** (TwiBot-22: all
  1M users incl. val and test; TwiBot-20: all 229,580 incl. support and test) [P-code l.117-156;
  `Dataset.py` l.227-249]. This is transductive (uses unlabelled feature values of test/support rows,
  no labels); reproduce it and document it. TwiBot-22 `followers_count` is z-scored twice (l.117-119
  and l.136-137); the second pass is an identity up to float error.
- The reference date is irrelevant after z-scoring (a constant shift), except through the fill value
  for missing `created_at`. Use `.days` of the timedelta (floor). Negative values are allowed (TwiBot-22
  accounts created after 2020-09-05).
- `len(name)` on TwiBot-22 is a benchmark bug relative to the paper ("screen name character count");
  the script computes `len(username)` into `num_username` and never uses it (l.107-112). Default:
  reproduce `len(name)`; ablation `len(username)`.
- Original repo 6-column variant (`botrgcn-orig6`): order [followers, friends, favourites, statuses,
  screen_name_length, active_days], same z-score, missing -> 0, ref 2020-09-01 [P-code `Dataset.py`
  l.157-251]. TwiBot-20 raw profile values are strings with a trailing space (`"136 "`) and missing
  values are the string `"None "`; the unified layer strips them and maps `"None"` to null.
- TwiBot-20 benchmark-code variant (listed_count instead of statuses_count in col 4) is a bug [U];
  not a default.
- Cresci-2015 (benchmark, `cresci_15/preprocess_1.py`): same 5 columns with ref 2020-09-01,
  `len(username)` for col 2 and **`listed_count`** for col 4 [P-code l.89-148].

#### 3.1.3 Categorical properties (`cat_properties_tensor`, float32 [N, 3])

| col | TwiBot-22 [P-code `twibot_22/preprocess_1.py` l.164-197] | TwiBot-20 default |
|---|---|---|
| 0 protected | 1 if `protected is True` else 0 | 1 if `profile.protected == "True "` else 0 |
| 1 verified | 1 if `verified is True` else 0 | same with `verified` |
| 2 default_profile_image | 1 if `profile_image_url` is null, `''`, or exactly `https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png`; else 0 | 1 if `profile.default_profile_image == "True "` [U: field choice; the benchmark TwiBot-20 code compares `profile_image_url` to `http://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png ` instead] |

No normalisation, missing -> 0. The shared layer must keep TwiBot-22 `profile_image_url` (the
processed jsonl does not have it; it must come from `user.json`).
`botrgcn-orig6` 11 flags, in this order: protected, geo_enabled, verified, contributors_enabled,
is_translator, is_translation_enabled, profile_background_tile, profile_use_background_image,
has_extended_profile, default_profile, default_profile_image; 1 iff the string equals `"True "`;
whole profile missing -> all 0 [P-code `Dataset.py` l.266-280].
Cresci-2015: 1 column, default_profile_image = 1 if `profile_image_url` is null or one of
`http://a0.twimg.com/sticky/default_profile_images/default_profile_{0..6}_normal.png` [P-code
`cresci_15/preprocess_1.py` l.153-176].

#### 3.1.4 Description embedding (`des_tensor`, [N, 768])

- Encoder: TwiBot-22 `roberta-base` [P-code `twibot_22/preprocess_2.py` l.17]; TwiBot-20 original
  **`distilroberta-base`** [P-code `Dataset.py` l.75] (paper says RoBERTa [P-paper Eq. 2]).
- Tokenisation: HF tokenizer of the same model, `add_special_tokens=True`, one description per call
  (no padding), truncation to the model maximum (512) in practice (see 3.1.7).
- Pooling: last hidden layer, **arithmetic mean over all token positions including `<s>` and `</s>`**
  [P-code l.27-34: sums `feature[0][i]` over all i and divides by `feature.shape[1]`]. When we batch
  with padding we must mask pads but keep the two special tokens.
- Empty / missing:
  - TwiBot-22: JSON null -> zero vector [P-code l.24-25]; empty string `''` is embedded (mean of the
    `<s>`,`</s>` states, a constant vector `e_empty`). user.json (API v2) normally has `''`, so most
    empty descriptions get `e_empty`, not zeros [U: null frequency]. The unified layer's null must be
    traced back to the raw value.
  - TwiBot-20 original: the raw value is always a string (empty is `" "`); it is embedded as is;
    a null profile or description becomes the literal string `'None'` [P-code `Dataset.py` l.57-58].
    Implementation: empty/whitespace -> `e_empty`; null -> embed `"None"`.
- Storage: fp16 memmap `emb/<encoder>/des.f16.npy` [N,768]; cast to fp32/autocast at use.

#### 3.1.5 Tweet embedding (`tweets_tensor`, [N, 768])

- Encoder `roberta-base` for all variants [P-code `Dataset.py` l.124; `twibot_22/preprocess_2.py`
  l.17]. Per-tweet pooling identical to 3.1.4 (mean over all positions incl. specials).
- Across tweets: arithmetic mean of the per-tweet vectors (uniform weights).
- Which tweets:
  - TwiBot-22 benchmark: tweets are grouped per author by scanning `tweet_0.json` ... `tweet_8.json`
    in order and appending `text` when `author_id` is one of the 1M users [P-code
    `twibot_22/preprocess_1.py` l.203-216]; the **first 20** in that order are used (loop breaks at
    j == 20 and divides by 20, else by the count) [P-code `preprocess_2.py` l.58-80]. A null text
    contributes a zero vector to the mean. No tweets -> zero vector.
  - TwiBot-20 original: **all** tweets in the user's `tweet` list (up to 200, "the recent 200"),
    release order irrelevant for an unweighted mean; `tweet` null -> the list `['']` -> `e_empty`
    (not zeros) [P-code `Dataset.py` l.104-105, 126-140].
  - `botrgcn-b20` / Cresci-2015: first 20 by `post` edge order, zero vector if none.
- Counts (for cost): TwiBot-20 labelled users hold 1,786,245 tweets (mean 166.3; 80 labelled users
  without tweets) (computed locally); support.json holds about 31.7M tweet strings and 23,457 support
  users have `tweet: null` (line counts, computed locally) -> about 33.5M tweets, matching the
  33,488,192 in TwiBot-22 Table 1 [P-paper]. TwiBot-22: 88,217,457 tweets [P-paper Table 7]; users
  without tweets: 36,797 of train and 16,501 of val (unified manifest) -> first-20 total about 17-19M
  tweets [U estimate].

#### 3.1.6 Graph (`edge_index` int64 [2,E], `edge_type` int64 [E])

PyG convention: an edge `[s, t]` sends a message from s to t.

- TwiBot-22 [P-code `twibot_22/preprocess_1.py` l.53-73]: only user-user relations `followers` and
  `following` of the 14 (`pinned, following, own, post, followers, membership, retweeted, like,
  followed, quoted, discuss, replied, mentioned, contain`) are used. Edges are taken exactly as in
  `edge.csv`, `[uid_index[source_id], uid_index[target_id]]`, rows whose ids are not users are
  dropped. `edge_type`: **`followers` = 0, `following` = 1.** Semantics [P-paper relation table]:
  `following`: source follows target -> message follower -> followee. `followers`: source is followed
  by target -> message followee -> follower. No reverse edges, no self loops added, no dedup.
  Counts [P-paper Table 7]: following 2,626,979; follower 1,116,655; total E = 3,743,634 (both ends
  are users, so the KeyError filter should drop none [U]). Overlap between the two relations (same
  pair stored both ways) unknown [U]. Max degree in the full heterogeneous graph 270,344 [P-paper].
- TwiBot-20 original [P-code `Dataset.py` l.291-327]: for every node i with `neighbor` not null:
  for each id in `neighbor.following`: edge `[i, j]`, type **0**; for each id in `neighbor.follower`:
  edge `[i, j]`, type **1**; ids not in the 229,580-node set are skipped. So type 0 sends
  follower -> followee, type 1 sends followee -> follower (the two types are swapped relative to
  TwiBot-22's numbering; the model is symmetric in relation index, so this is cosmetic). Only users
  with neighbour lists emit edges: 10,739 labelled users (1,087 labelled have null) and 1,185
  support users (216,569 support null) (counted locally). Labelled lists hold 105,701 following +
  103,015 follower ids; the paper reports **227,979 edges** [P-paper III.A]; the rest come from the
  1,185 support users [U, consistent]. Lists are random samples of at most 20 each [P README].
  Benchmark Format22: `friend` -> type 0, every other non-`post` relation (`follow`) -> type 1
  [P-code `twibot_20/preprocess_1.py` l.48-74].
- Shared-layer mapping: `graph/following.csr.npz` (source = the account that follows; TwiBot-20:
  source = i, target = j in `i.neighbor.following`) and `graph/followers.csr.npz` (TwiBot-22: as in
  edge.csv) / `graph/follower.csr.npz` (TwiBot-20: source = i, target = j in `i.neighbor.follower`).
  Build `edge_index` by concatenating COO of both CSRs, type ids as above.
- Cresci-2015 (benchmark): `friend` -> 0, `follow` -> 1, user-user only; 7,086,134 edges incl.
  2,827,757 `post` [P-paper Table 1], so at most 4,258,377 user-user edges before the user filter [U].
  Our unified Cresci split is not the benchmark split: Cresci numbers are not comparable.

#### 3.1.7 Truncation detail

Benchmark scripts call `pipeline('feature-extraction', ..., padding=True, truncation=True,
max_length=50)` (TwiBot-22) and original `max_length=500` (TwiBot-20 tweets). In transformers >= 4.1x
`FeatureExtractionPipeline._sanitize_parameters` only consumes `truncation` (and `tokenize_kwargs`);
`max_length`/`padding`/`add_special_tokens` passed to the constructor are dropped, so the effective
limit is the model maximum 512 [U: depends on the transformers version used in 2022]. Tweets
(<= 280 chars) and descriptions (<= 160 chars) rarely exceed 128 tokens, so the choice barely matters.
Default: truncate at 512 (equivalent to no truncation); ablation flag `max_tokens=50`.

#### 3.1.8 Files we write (per dataset, `artifacts/features/<dataset>/botrgcn[-variant]/`)

`num_prop.f32.npy` [N,5], `cat_prop.f32.npy` [N,3], `des.f16.npy` and `tweets.f16.npy` (or symlinks
to `emb/...`), `edge_index.i64.npy` [2,E], `edge_type.i64.npy` [E], `label.i64.npy` [N] (-1 support),
`train_idx/val_idx/test_idx.i64.npy`, `meta.json` (variant, encoder names and HF revisions, z-score
means/stds, edge counts per type, tweet counts, ref dates).

### 3.2 Model inputs

`des [N,768]`, `tweet [N,768]`, `num_prop [N,5]` (+2 with ExtScore), `cat_prop [N,3]`, `edge_index`,
`edge_type`. All nodes are forwarded every step (full batch).

## 4. Architecture

`D` = embedding dimension (TwiBot-22 benchmark 32; TwiBot-20 benchmark 32; original repo 128).
[P-code `BotRGCN/model.py` class `BotRGCN` l.6-55; identical in `src/BotRGCN/twibot_22/model.py`]

| step | op | shape |
|---|---|---|
| 1 | d = LeakyReLU(Linear(768, D/4)(des)) | N x D/4 |
| 2 | t = LeakyReLU(Linear(768, D/4)(tweet)) | N x D/4 |
| 3 | n = LeakyReLU(Linear(5, D/4)(num_prop)) | N x D/4 |
| 4 | c = LeakyReLU(Linear(3, D/4)(cat_prop)) | N x D/4 |
| 5 | x = cat(d, t, n, c) | N x D |
| 6 | x = LeakyReLU(Linear(D, D)(x)) | N x D |
| 7 | x = RGCN(x, edge_index, edge_type) | N x D |
| 8 | x = Dropout(p)(x) | |
| 9 | x = RGCN(x, edge_index, edge_type) (**same module as step 7, weights tied**) | N x D |
| 10 | x = LeakyReLU(Linear(D, D)(x)) | N x D |
| 11 | logits = Linear(D, 2)(x) | N x 2 |

- LeakyReLU default slope 0.01. No activation between or after the two R-GCN applications.
- RGCN = PyG `RGCNConv(D, D, num_relations=2)` defaults: `aggr='mean'` per relation, `root_weight=True`,
  `bias=True`, no basis/block decomposition:
  `x_i' = W_root x_i + sum_r mean_{j: (j->i) in E_r} W_r x_j + b` (a relation with no incoming edges
  contributes 0). Matches paper Eq. 5 [P-paper].
- Dropout p: model default **0.3**. The benchmark `train.py` sets `dropout=0.1` in a variable that is
  never passed to the model (l.19, l.28), so the effective value is 0.3 [P-code]. Dropout only at
  step 8 (no input dropout in this model).
- Init: `model.apply(init_weights)`: every `nn.Linear.weight` gets `kaiming_uniform_` (a=0,
  fan_in, gain sqrt 2); biases keep PyTorch default; RGCNConv keeps PyG glorot/zeros [P-code
  `utils.py` l.10-12].
- Loss: unweighted cross-entropy on train nodes only (softmax in loss). Paper Eq. 8 adds L2; code
  uses AdamW decoupled weight decay instead.
- Not to confuse with `src/GCN_GAT/model.py::BotRGCN` (two untied RGCN layers + input dropout),
  which is a different variant run only inside the GCN_GAT folder (see gcn_gat.md).

### 4.1 "w/o graph" ablation (benchmark Table 3)

Paper: "We remove the R-GCN and RGT while using two MLP layers with user features ... their 'w/o
graph' results are identical" [P-paper App. B.4]. No code is released (searched `src/BotRGCN`,
`src/RGT`, `src/GCN_GAT/remove_graph.py` is byte-identical to `train.py`, `src/simple_NN` is an
unrelated MLP on `simple_vectorize` features). Spec [U, our reading]: identical to section 4 except
steps 7 and 9 become two separate `nn.Linear(D, D)` layers (no activation, dropout between, as in
BotRGCN); equivalently an R-GCN on an empty edge set without weight tying. Same features,
hyperparameters and training as BotRGCN on that dataset. Variant flag `wog_tied=true` reuses one
Linear twice. Expected: section 6.

Review round 1 (2026-11-12, measured): the literal reading "two MLP layers with user features",
`cat(des, tweet, num, cat) -> Linear(1544, D) -> LeakyReLU -> Dropout(0.3) -> Linear(D, 2)` with the
BotRGCN TwiBot-20 benchmark preset, gives 82.9 +- 1.3 acc / 84.6 +- 1.3 F1 (5 seeds), matching Table 3
(82.6 / 83.8); the encoder-kept variant above gives 86.3 / 87.8 and BotRGCN itself 85.7 / 87.4. It is
implemented as `--model mlp2` (results under `botrgcn_mlp2`); `--model wog` stays as the
same-architecture graph ablation. Edge direction does not explain the gap (reversed or undirected
edges: 86.3 / 86.1 acc).

## 5. Training

| setting | TwiBot-22 benchmark | TwiBot-20 benchmark | TwiBot-20 original repo | Cresci-2015 benchmark |
|---|---|---|---|---|
| source | `twibot_22/train.py` | `twibot_20/train.py` | `BotRGCN/main.py` | `cresci_15/train.py` |
| D | 32 | 32 | 128 | 32 |
| dropout (effective) | 0.3 | 0.3 | 0.3 | 0.3 |
| optimiser | AdamW | AdamW | AdamW | AdamW |
| lr | 1e-2 | 1e-2 | 1e-3 | 1e-2 |
| weight decay | 5e-2 | 5e-2 | 5e-3 | 5e-2 |
| epochs | 200 | 50 | 100 | 200 |
| cat dim | 3 | 3 | 3 | 1 |

- Full batch: every epoch is one forward over all nodes and one optimiser step; loss on `train_idx`.
- No early stopping, no model selection: test is evaluated once after the last epoch. Validation
  accuracy is only printed. We also log val metrics per epoch (for diagnostics) but report last epoch.
- Seeds: none set in code; the benchmark reports 5 runs. We use seeds 0-4
  (`torch.manual_seed`, `cuda` deterministic scatter where possible; PyG scatter-mean on GPU is
  non-deterministic in float order, accept tiny run-to-run noise).
- Recommended defaults for our repo: TwiBot-22 = benchmark column; TwiBot-20 = **benchmark column**
  (it produced the 85.8/87.3 row; D=128 original settings as a flag `--preset original`).
- Train/val/test usage: train for loss, val only logged, test once at the end. Features of all nodes
  (incl. test and support) are visible in message passing: transductive by design.

## 6. Evaluation and published numbers

Metrics: accuracy and **bot-class** binary F1 (`sklearn.f1_score`, pos_label 1 = bot), plus precision,
recall, MCC/AUC (AUC in benchmark `train.py` is computed on hard labels).

| Dataset | Source | Acc | Bot F1 | Prec | Rec |
|---|---|---|---|---|---|
| TwiBot-20 | BotRGCN paper Table III [P-paper] (single run, MCC 0.7021) | 84.62 | 87.07 | - | - |
| TwiBot-20 | benchmark readme, mean +- std of 5 [P-code readme] | 85.75 +- 0.68 | 87.25 +- 0.73 | 84.52 | 90.19 |
| TwiBot-20 | benchmark paper Table 2/8 [P-paper] | 85.8 +- 0.7 | 87.3 +- 0.7 | | |
| TwiBot-22 | benchmark readme [P-code] | 79.66 +- 0.14 | 57.50 +- 1.42 | 74.81 | 46.80 |
| TwiBot-22 | benchmark paper [P-paper] | 79.7 +- 0.1 | 57.5 +- 1.4 | | |
| Cresci-2015 | benchmark readme [P-code] | 96.52 +- 0.71 | 97.30 +- 0.53 | | |
| TwiBot-20 w/o graph | benchmark Table 3 [P-paper] | 82.6 (Prev. 85.7) | 83.8 (Prev. 87.3) | | |
| TwiBot-22 w/o graph | benchmark Table 3 [P-paper] | 75.4 (Prev. 79.7) | 41.2 (Prev. 57.5) | | |

Other re-runs of BotRGCN on TwiBot-22 bot F1: 54.99 (BotMoE), 56.47 (BotDGT), 49.28 (CDRBot),
48.29 (BotUmc) [S: literature survey]. The TwiBot-22 test set is 29.4% bots while train is 7.8%,
so threshold/prior effects dominate F1 variance.

## 7. Scale plan for our hardware

**Hard requirement:** every step runs via `tools/memrun` (MemoryMax 10G, no swap, serialised lock);
design target peak host RSS <= 8 GB; never load a full TwiBot-22 tweet file, `edge-002.csv`,
`user.json` or a full feature table into RAM: stream / chunk / memmap / on-disk CSR. DataLoader
workers and pinned buffers count. Baseline process cost: Python + torch + CUDA context ~ 1.5 GB RSS.

### 7.1 Building the `botrgcn` scheme

| step | TwiBot-22 peak host RSS | TwiBot-20 peak host RSS | VRAM |
|---|---|---|---|
| profile fields | stream `user.json` (780 MB JSON array) with `ijson` (C backend), keep 1M x 6 numbers (48 MB f64) + 3 flags + description strings (~1M x ~100 B ~ 0.15-0.3 GB as Python str; better: write descriptions to the shared layer, not RAM) -> **< 1 GB**. `json.load` would need ~5-8 GB: forbidden | stream `support.json` (5.3 GB) with `ijson` item by item (never `json.load`); per item keep only profile fields -> **< 1 GB** (labelled files are 30-240 MB; `json.load` of `train_bak.json` ~ 1.5-2 GB is tolerable but stream anyway) | 0 |
| z-score | 1M x 5 f64 = 40 MB | 229,580 x 5 = 9 MB | 0 |
| graph | from the shared layer's CSR (TwiBot-22: indptr 8 MB, indices 3.74M x 4 B = 15 MB per relation); `edge_index` int64 = 2 x 3,743,634 x 8 B = 60 MB, `edge_type` 30 MB -> **< 0.5 GB**. If built from `edge-002.csv` directly: pandas `read_csv(chunksize=2e6, usecols=...)`, keep rows with relation in {followers, following}, map ids through a 1M-entry dict (~0.15 GB): ~1.5 GB peak | 228k edges, negligible | 0 |
| tweet text selection (TwiBot-22) | stream `tweet_0..8.json` (~100 GB, 70 MB/s disk -> >= 24 min I/O, ijson parse ~45-90 min total) keeping a uint8 count per user (1 MB) and appending `(idx, text)` of the first 20 per user to an on-disk ragged store; **never hold texts in RAM** (20M texts x ~150 B as Python objects = ~3 GB) -> < 1 GB. Done once by the shared-layer builder if it records `release_rank` | from the shared tweet store | 0 |
| RoBERTa encoding | ~2.5 GB (torch+CUDA 1.5 GB, tokenizer, one length-bucketed batch, fp16 output memmap written row by row as each user's tweets finish; no N x 768 accumulator in RAM: that would be 1M x 768 x 4 B = 3.1 GB) | ~2.5 GB | roberta-base fp16 weights 0.25 GB + batch 256 x 128 tokens activations ~1 GB -> ~2 GB |

Encoding cost (assumption: roberta-base fp16 on the 4070 Ti, length-bucketed batches, 2,500 short
texts/s; range 1,500-4,000/s):

| item | texts | time |
|---|---|---|
| TwiBot-22 descriptions (all 1M; empty ones reuse `e_empty`) | ~0.78M non-empty (77.65% share in the train sample) | 0.78M / 2,500 = 5 min (<= 9 min) |
| TwiBot-22 tweets, first 20 | ~17-19M (upper bound 20M) | 18M / 2,500 = 2.0 h (1.3-3.3 h) |
| TwiBot-22 all <=200 tweets (not needed) | ~70-88M | 8-10 h |
| TwiBot-20 descriptions (distilroberta, ~2x faster) | 229,580 | ~1 min |
| TwiBot-20 tweets, all (default) | ~33.5M | 33.5M / 2,500 = 3.7 h (2.3-6.2 h) |
| TwiBot-20 tweets, first 20 (`botrgcn-b20`) | <= 4.6M | ~0.5 h |

Outputs: TwiBot-22 2 x 1M x 768 x 2 B = 3.07 GB on disk; TwiBot-20 2 x 229,580 x 768 x 2 B = 0.71 GB.
Total precompute GPU time: TwiBot-22 ~2.2 h, TwiBot-20 ~3.8 h (plus ~1 h CPU/I-O for TwiBot-22
tweet selection).

### 7.2 Training BotRGCN on TwiBot-22 (full batch, D = 32)

The benchmark ran this full batch on 11 GB RTX 2080 Ti cards [P-paper B.6], with fp32 inputs
(6.1 GB) and D = 32. Our plan keeps it full batch with **fp16 input storage on the GPU**.

VRAM arithmetic (N = 1e6, E = 3.74e6, D = 32, fp32 activations):
- des + tweets fp16 resident: 2 x 1e6 x 768 x 2 B = **3.07 GB** (fp32 would be 6.14 GB).
- num/cat/labels/idx: ~0.06 GB; edge_index + edge_type: 0.09 GB; per-relation index copies inside
  RGCNConv: ~0.1 GB.
- activations kept for backward: four N x 8 blocks (pre and post activation) 0.26 GB; concat and
  input layer 0.38 GB; each RGCN application gathers E_r x D x 4 B = 0.34 GB (following) + 0.14 GB
  (followers) plus per-relation N x D outputs 0.26 GB -> ~0.75 GB per application, x2 = 1.5 GB;
  dropout mask + output layer 0.3 GB -> ~2.5 GB.
- backward transients ~1.5 GB; caching-allocator slack ~15%.
- **Peak ~ 3.07 + 0.25 + 2.5 + 1.5 = 7.3 GB, ~8.5 GB with slack** (< 12 GB). With fp32 inputs it
  would be ~11 GB: do not.
- Use `torch.autocast(dtype=float16)` (or bf16) only for the two 768 -> 8 linears, then `.float()`;
  the rest in fp32.

Host RSS: load `des.f16.npy`/`tweets.f16.npy` into preallocated GPU tensors in 50k-row chunks
(`np.load(mmap_mode='r')` slice -> pinned 77 MB staging buffer -> `copy_`), drop the mapping after;
peak **~2 GB** (torch+CUDA 1.5 GB, staging, small arrays). No DataLoader workers.

Time: per epoch the dominant cost is reading 3 GB of inputs twice (fwd + bwd) at ~500 GB/s plus
scatter over 3.7M edges: ~0.05-0.3 s. 200 epochs < 1-2 min; 5 seeds < 10 min. Eval: one forward.
If memory is ever short: fall back to PyG `NeighborLoader` (2 hops, fanout = full) with GPU-resident
inputs, which is exact for the 2-layer model's outputs but changes the optimiser to mini-batch
(deviation, record it).

### 7.3 TwiBot-20 (N = 229,580, E ~ 228k)

fp32 inputs 2 x 229,580 x 768 x 4 B = 1.41 GB; D=32 (or 128) activations ~0.1-0.4 GB; VRAM < 3 GB;
host RSS < 2.5 GB; 50-100 epochs in seconds. 5 seeds x 2 presets: minutes.

### 7.4 GPU-hour summary

BotRGCN + w/o graph, both datasets, 5 seeds: < 0.5 GPU-h training. Precompute: ~6 GPU-h total
(shared with RoBERTa+MLP, GCN, GAT, RGT, SimpleHGN, HGT, BotMoE, ...).

## 8. ExtScore column

- Transform: shared `[z, m]` (z = standardised logit of clipped p, mu/sd from observed TRAIN rows,
  z = 0 where missing; m = missing indicator).
- Default injection: append `[z, m]` to `num_prop`, raising width 5 -> 7, before
  `linear_relu_num_prop` (`Linear(7, D/4)`). Do **not** apply the transductive z-score of 3.1.2 to
  these two columns. Through the R-GCN, a node also sees its neighbours' ExtScore (by design).
- Missing handling: support nodes (TwiBot-20) and any account without a score get z = 0, m = 1.
  ExtScore must be computed without access to test labels; scoring val/test accounts is fine (it is an
  input feature).
- `extscore: none` must reproduce the baseline exactly: same widths (5), same init order (create the
  ExtScore-free model when `extscore=none`; seed-for-seed identity requires the parameter creation order to be
  unchanged).
- Ablations: (a) separate fifth branch `LeakyReLU(Linear(2, D/4))`, concat width 5D/4 ->
  `Linear(5D/4, D)`; (b) late fusion: concat `[z, m]` to the post-`linear_relu_output1` vector,
  `Linear(D+2, 2)` (no neighbour propagation of ExtScore); (c) both early and late; (d) `p` raw instead of
  logit (check calibration effect). Same for the w/o-graph MLP (where early and late differ only in
  depth).
- The same `num_prop` extension serves every downstream method of the `botrgcn` scheme; it is stored
  as a separate file `extscore_zm.f32.npy` [N,2] and concatenated at load time, so the base tensors stay
  untouched.

## 9. Pitfalls

1. **Weight tying**: BotRGCN applies one `RGCNConv` twice. Re-implementations with two layers differ.
2. **Dropout 0.1 vs 0.3**: benchmark `train.py` defines 0.1 but never passes it; effective 0.3.
3. **No model selection**: report the last epoch; selecting on val/test inflates numbers.
4. **Column sets**: paper 6 + 11, original code 6 + 11, original `main.py` 5 + 3, benchmark 5 + 3
   (Cresci 1). Use 5 + 3.
5. **TwiBot-22 "screen_name_length" is `len(name)`** (display name), not `len(username)`.
6. **No log transform** on heavy-tailed counts; z-score over all nodes incl. test/support
   (transductive, no labels). Do not "fix" it in the baseline; a `log1p` variant is an ablation.
7. **Token pooling includes `<s>` and `</s>`** and pads must be masked. Standard sentence-transformer
   mean pooling (pads masked, specials included) is the same only if specials are kept; CLS pooling
   is wrong.
8. **Empty text handling differs**: TwiBot-22 no tweets -> zeros, null description -> zeros, `''`
   description -> `e_empty`; TwiBot-20 original null tweets -> `e_empty`, description `None` ->
   embedding of the string "None".
9. **Tweet subset**: TwiBot-22 first 20 in release-file order, not newest 20; TwiBot-20 original all.
10. **TwiBot-20 local files**: `train.json`/`test.json` in `data/raw/twibot-20-full/` are
    embedding-sorted copies; use `train_bak.json`, `dev.json`, `test_bak.json`, `support.json`.
11. **Relation ids swap** between TwiBot-20 (following = 0) and TwiBot-22 (followers = 0); harmless
    but keep `meta.json` explicit.
12. **Direction**: edges are directed as in the release; do not add reverse edges or make the graph
    undirected (that changes R-GCN's per-relation means).
13. **TwiBot-22 class shift**: train 7.8% bots, test 29.4% bots. Do not rebalance in the baseline.
14. Benchmark AUC is computed on hard predictions; compute AUC from probabilities ourselves and
    label it.
15. Benchmark TwiBot-20 preprocessing code is broken (cat reshape 5301, `listed_count` as statuses);
    do not port it.
16. `transformers` pipeline silently ignores `max_length` (3.1.7).

## 10. Open questions / unverified

- Which 5 numeric / 3 categorical columns and which description encoder the Drive TwiBot-20 tensors
  used (check against the zip if obtainable).
- Whether `tweet_i.json` order is chronological per user (decides "first 20" vs "newest 20").
- Whether TwiBot-22 `user.json` order equals `label.csv` order (inferred yes; irrelevant to us).
- Overlap / duplicates between TwiBot-22 `following` and `followers` edges; exact E after filtering
  (compute when the shared layer builds the CSR and record in `meta.json`).
- Exact w/o-graph architecture (tied vs untied linear layers).
- Effective truncation length in the 2022 runs.
- Fraction of TwiBot-22 descriptions that are JSON null vs `''`.

## 11. Acceptance check

5 seeds, mean reported, bot-class F1:

| Dataset / variant | Target | Tolerance |
|---|---|---|
| TwiBot-22 BotRGCN | acc 79.7, F1 57.5 | acc +-1.0, F1 +-3.5 (published re-runs span 48-57.5) |
| TwiBot-20 BotRGCN (benchmark preset) | acc 85.8, F1 87.3 | +-1.5 each |
| TwiBot-20 BotRGCN (original preset) | acc 84.6, F1 87.1 | +-1.5 each |
| TwiBot-22 w/o graph | acc 75.4, F1 41.2 | acc +-1.5, F1 +-5 |
| TwiBot-20 w/o graph | acc 82.6, F1 83.8 | +-2 |

Also required: graph ablation sign (w/o graph below BotRGCN on both datasets); feature-builder
unit checks (row counts, z-score means ~0 / std ~1 over all rows, edge counts per type logged and
TwiBot-20 total within 1% of 227,979, TwiBot-22 within 1% of 3,743,634).
