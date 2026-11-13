# SEGCN: subgraph-encoding GCN (GNN-AK with a GCN kernel) for social bot detection

Status: spec written 2026-11-12. The official code exists but is a lightly renamed copy of GNN-AK.
Its shipped configuration runs a plain 2-layer GCN, not the subgraph path, and its subgraph
extractor cannot run on either Twitter dataset (dense N x N masks). This spec is therefore
**paper-first**. The GNN-AK code fills in the operator details the paper leaves out. Every
departure is tagged.

Tags: **[P-paper]** read in the paper; **[P-code]** read in the official SEGCN repo (path given);
**[P-gnnak]** read in upstream GNN-AK; **[S]** secondary source (named); **[U]** unverified or our
own default.

---

## 1. Summary

SEGCN (Liu et al., Sci. Rep. 2024) is a transductive, homogeneous-graph GNN for user-level bot
detection. Each user starts from a 128-d vector concatenating four BotRGCN-style encodings:
RoBERTa description, RoBERTa mean tweet, numeric properties and categorical properties [P-paper
Eqs. 1-2, Table 1, "processing details can be referred to [BotRGCN]"]. The GCN message passing is
then replaced by **GNN-AK ("GNN as Kernel", Zhao et al., ICLR 2022)** with a GCN base: for every
node u, a subgraph is extracted by a node2vec-style random walk of length 30 rooted at u. A GCN
runs inside that induced subgraph. u's new representation is FUSE(concat) of the root's own
embedding inside its subgraph ("centroid") and a pooled readout of the whole subgraph ("subgraph
encoding") [P-paper Eqs. 3-6]. Two such layers feed an MLP head with softmax and cross-entropy
[P-paper Eqs. 7-8].

- Family: subgraph GNN (beyond 1-WL).
- Graph: yes, a single relation (union of following and follower edges), treated as undirected.
- Text: yes, via frozen RoBERTa embeddings (same as BotRGCN).
- Temporal: no, apart from "active days".
- Encoders: roberta-base (frozen, feature extraction only).

## 2. Citation and code

| Item | Value |
|---|---|
| Paper | Feng Liu, Zhenyu Li, Chunfang Yang, Daofu Gong, Haoyu Lu, Fenlin Liu. "SEGCN: a subgraph encoding based graph convolutional network model for social bot detection." *Scientific Reports* 14:4122, 2024. DOI 10.1038/s41598-024-54809-z (OpenAlex W4391925527). CC-BY 4.0 [P-paper]. Received 22 Jul 2023, accepted 16 Feb 2024. |
| Data-availability statement | Points only to the TwiBot-20 and TwiBot-22 dataset repositories [P-paper]. |
| Official code | https://github.com/FengLiuii/SEGCN, licence **MIT**. Read at commit `413d1d6408f5bc1be3c295aa1f18d28e1ffb122a` (2025-10-11). The repo was created 2023-11-09, but its only code commit ("V1", `4740e10`) was pushed **2025-10-11, about 20 months after publication**. |
| Provenance of the code | `core/model.py`, `core/transform.py`, `core/transform_utils/*`, `core/model_utils/*` and `core/config.py` are **byte-identical** (modulo whitespace) to upstream GNN-AK `LingxiaoShawn/GNNAsKernel` @ `479dee1ab4172a02a4d236c30bdf54f3691f308f` (MIT). The only differences: `cfg.dataset='twi20'`, plus a new `core/model_segcn.py` that copies `GNNAsKernel` as class `SEGCN` with `node_embedding=True` (no graph pooling) and `data.clone()` per layer. |
| Upstream method | Lingxiao Zhao, Wei Jin, Leman Akoglu, Neil Shah. "From Stars to Subgraphs: Uplifting Any GNN with Local Structure Awareness." ICLR 2022, arXiv 2110.03753. Code: GNNAsKernel (MIT). SEGCN cites it as ref. 51. |
| May we read or reuse the code? | Yes. Both repos are MIT, so reuse with attribution is allowed. We still write our own scalable implementation, because the upstream extractor is O(N^2) memory (Section 9). |
| Third-party reimplementations | **None found.** Checked: GitHub repo search ("SEGCN", "SEGCN bot", "subgraph encoding bot detection", "social bot subgraph"), web search, and the 26 works citing W4391925527 listed by OpenAlex (full texts not available through our tool). No citing paper was found that reports SEGCN as a baseline [U for the paywalled ones]. |

## 3. Inputs

### 3.1 What the method needs from the shared layer

| Need | Shared-layer artifact | Notes |
|---|---|---|
| Users, split, label | `users.parquet` (`idx`, `split`, `label`) | Transductive: all N nodes (incl. support) are in the graph; loss on train only. |
| Description embedding | `emb/roberta-base/des.f16.npy` [N,768] | Same tensor as BotRGCN (Section 3.2). |
| Tweet embedding | `emb/roberta-base/tweets_mean.f16.npy` [N,768] | Same tensor as BotRGCN: per-tweet token-mean, then mean over tweets (Section 3.2). |
| Numeric properties | `profile.parquet` | TwiBot-20: 6; TwiBot-22: 5 (Section 3.3). |
| Categorical properties | `profile.parquet` | TwiBot-20: 11; TwiBot-22: 3 (Section 3.3). |
| Graph | `graph/following.csr.npz`, `graph/followers.csr.npz` | Union, symmetrised, self-loops removed, deduplicated. One homogeneous undirected graph [P-paper: "social graph whose edge represents the following and follower relationship"; P-code: the Planetoid loader symmetrises]. |
| **NEW: random-walk subgraph store** | `graph/segcn_rw_L30_p1_q1_seed{s}/` | Not in the shared layer. Built by this method (Section 7.3). Per root: up to 31 node ids, plus the local induced edge list. |
| ExtScore (optional) | `extscore/<tag>.parquet` | Section 8. |

No per-tweet metadata, no temporal snapshots and no 2-hop CSR are needed. Only one CSR is needed
(the undirected union).

### 3.2 Semantic features (paper Eqs. 1-2) [P-paper]

- Description: `N_d = LeakyReLU(W_d · mean_i RoBERTa(d)_i + b_d)`, where the mean is over the
  description's token embeddings, giving a 32-d vector (D = 32, see below).
- Tweets: `N_t = LeakyReLU(W_s · (1/M_u) Σ_j (1/m_j) Σ_i RoBERTa(w_ji) + b_s)`, giving 32-d.
  This is token-mean per tweet, then mean over the user's M_u tweets.
- RoBERTa variant: "RoBERTa ... in Transformer.pipeline", i.e. HF feature-extraction pipeline
  [P-paper]. The checkpoint is not named. Default `roberta-base` (768-d) [U], as in the BotRGCN
  and TwiBot-22 benchmark code.
- Which tweets: not stated in SEGCN. It says processing follows BotRGCN [P-paper]. Use exactly
  the BotRGCN pipeline of the shared layer:
  - TwiBot-20: all tweets in the release (up to 200).
  - TwiBot-22: the benchmark's `src/BotRGCN/twibot_22/preprocess_2.py` loop. It sums token-means of
    tweets j = 0..19, breaks at j == 20 and then divides. That is about 20 tweets per user [S:
    benchmark code]. Our own BotRGCN spec should have pinned the exact quirk.
- Users with no description or no tweets get a zero 768-d vector (benchmark convention) [S].
- Dimension bookkeeping [P-paper]: `N_s = [N_t, N_d] ∈ R^{2D}`, `N_p = [N_pn, N_pc] ∈ R^{2D}`,
  `N_init = [N_s, N_p] ∈ R^{4D}`. Table 4 gives "embedding size 128", hence D = 32 per block,
  exactly BotRGCN's `embedding_dimension/4` layout.

### 3.3 Property features (paper Table 1) [P-paper]

Table 1 (17 items, BotRGCN's TwiBot-20 set):

- Numeric (6): followers, followings, favourites, statuses, active days (registration to
  collection time), screen-name length.
- Categorical (11): protected, contributors_enabled, is_translator, is_translation_enabled,
  geo_enabled, profile_background_tile, profile_use_background_image, has_extended_profile,
  default_profile, default_profile_image, verified.
- Preprocessing: numeric z-score; categorical one-hot / binary [P-paper].

Per dataset:

| Dataset | Numeric (z-score, fit on all N nodes as BotRGCN does [S]) | Categorical (0/1) |
|---|---|---|
| TwiBot-20 | the 6 above → `num` [N,6] | the 11 above → `cat` [N,11] (exactly BotRGCN `num_prop_size=6, cat_prop_size=11`) |
| TwiBot-22 | followers_count, following_count, tweet_count, username length, active_days → [N,5] | protected, verified, default_profile_image → [N,3] |

TwiBot-22's v2 user objects lack most Table-1 fields. The benchmark's TwiBot-22 BotRGCN uses
`num_prop_size=5, cat_prop_size=3` [S: `src/BotRGCN/twibot_22/model.py`, `train.py:28`,
`preprocess_1.py`]. The paper does not say what it did for TwiBot-22 [U]. We use the benchmark set,
because SEGCN's TwiBot-22 BotRGCN row equals the benchmark's numbers (Section 6).

### 3.4 Graph and subgraph extraction (paper Eq. 3, Table 4)

- Relation: following ∪ follower, homogeneous, **undirected** [P-paper Section "Input"; the
  paper's GCN baseline description says "GCN which aggregates on undirected graphs"; P-code
  Planetoid symmetrises]. Remove self-loops and duplicates.
- Subgraph of root u: the node set visited by **one random walk of length W_l = 30 steps** from u,
  using node2vec transition rules [P-paper Eq. 3 and Table 4 "Walk length 30"]. Table 4 lists
  "Number of walks", return p and in-out q as "–" for SEGCN.
  - Defaults: `repeat = 1` [U; GNN-AK cfg default is 5, the SEGCN yaml uses 1], `p = q = 1` [U;
    GNN-AK default, i.e. a uniform DeepWalk walk].
  - The node set is therefore at most 31 distinct nodes including u.
- Induced subgraph: **all** graph edges between visited nodes, not only the traversed ones [P-gnnak
  `extract_subgraphs`: `edge_mask = node_mask[:, src] & node_mask[:, dst]`].
- Isolated root: the walk stays at u (torch_cluster semantics), so the subgraph is {u} with no
  edges. GCNConv's self-loop still applies.
- Hop/distance labels: GNN-AK can embed shortest-path distance to the root (`hops_dim`, default
  16 in GNN-AK). The SEGCN paper does not mention it and the SEGCN yaml sets `hops_dim: 0`.
  Default **off** [U]; ablation `hop_dim=16` (BFS distance inside the subgraph, capped at 10, 0 =
  root).
- Sampled once per seed before training. SEGCN `train.py` applies the transform once
  (`data = transform(data)`), so subgraphs are fixed for all epochs [P-code `train.py:87`].
  Ablation: resample each epoch (GNN-AK "online").

## 4. Architecture (default = paper Eqs. 1-8, with GNN-AK operator details)

N nodes, hidden H = 128, outer GNN-AK layers L = 2, inner GCN layers per kernel L_in = 1, dropout
p = 0.3.

| # | Block | Op | Shape | Source |
|---|---|---|---|---|
| 0 | Typed input encoders | `d=LReLU(Lin(768,32)(des))`, `t=LReLU(Lin(768,32)(tw))`, `n=LReLU(Lin(n_num,32)(num))`, `c=LReLU(Lin(n_cat,32)(cat))`; `h0=[t,d,n,c]` | [N,128] | P-paper Eqs. 1-2, Section "Property representation". BotRGCN applies one extra `Lin(128,128)+LReLU`; the paper does not, so it is omitted [U]. |
| 1..L | GNN-AK layer l (below) | `h_{l+1} = h_l + Dropout(ReLU(BN(Kernel_l(h_l))))` | [N,128] | P-gnnak `GNNAsKernel.forward` (outer BN, ReLU, dropout, residual `res=True`) |
| L+1 | Head | `Dropout(h_L)` → `Lin(128,128,bias=False)` → `BN` → `ReLU` → `Lin(128,2)` | [N,2] | P-paper Eqs. 7-8 (σ(W h + b), then softmax). P-gnnak `output_decoder = MLP(nhid, nout, nlayer=2)`. |
| loss | Cross-entropy on train nodes, unweighted, no label smoothing | | | P-paper; P-code `train.py:131` |

**Kernel_l (subgraph GCN encoder)** for every root u, with subgraph nodes V_u (|V_u| ≤ 31) and
induced edges E_u:

1. `x = h_l[V_u]`. Rows are lifted copies: a node in k subgraphs gets k independent rows.
2. Inner GCN, repeated L_in times: `x = x + Dropout(ReLU(BN(GCNConv_{128→128, bias=False}(x,
   E_u))))`.
   - GCNConv uses symmetric normalisation with self-loops, degrees computed **inside the
     subgraph**.
   - One BN is shared over all lifted rows of the batch.
   - Followed by `output_encoder = Identity` [P-gnnak `GNN`].
3. `centroid_u = x[root position]` [P-paper Eq. 6 "centroid encoding"].
4. `subgraph_u = POOL_{v∈V_u} ReLU(BN(Lin(128,128,bias=False)(Dropout(x_v))))`, with POOL =
   **mean** [P-paper: "typical choices ... SUM and MEAN"; P-code yaml `pool: 'mean'`; P-gnnak
   `subgraph_transform = MLP(nlayer=1, with_final_activation=True)`].
5. `Kernel_l(u) = Lin(256,128,bias=False)(Dropout([centroid_u, subgraph_u]))`. FUSE = concat
   [P-paper Eq. 6 "FUSE is concatenation"; P-gnnak `out_encoder` with `embs_combine_mode='concat'`].
   - **No context encoding.** GNN-AK+ "context" pools over all subgraphs that contain u. It is in
     GNN-AK and in the SEGCN yaml (`embs: (0,1,2)`, `'add'`), but not in the paper. Ablation:
     `embs=(0,1,2)`.

Parameters: about 0.2 M (typed encoders 2×768×32 + small; each layer about 3×128×128 + 256×128;
head about 17k). Activations dominate memory, not weights.

## 5. Training

| Item | Default | Source / alternatives |
|---|---|---|
| Optimiser | AdamW | P-paper Table 4. Code: Adam [P-code `train.py:166`]. |
| lr | 0.01 | P-paper Table 4; code yaml 0.01. |
| Weight decay | 0.005 | P-paper Table 4. Code yaml 5e-4. |
| Dropout | 0.3 | P-paper Table 4. Code yaml 0.5. |
| Hidden / embedding | 128 | P-paper Table 4. Code yaml 64. |
| Layers | "Network layers 2" → L = 2 outer, L_in = 1 inner | P-paper Table 4. Interpretation [U]; ablate L = 1, L_in = 2. Code yaml `num_layers: 2, mini_layers: 0`, which disables subgraphs (Section 9). |
| Epochs | 100 | P-paper Table 4. Code yaml 200. |
| LR schedule | none | Paper silent. Code: StepLR(step 50, γ 0.5) [P-code]; ablation. |
| Model selection | evaluate val every epoch; report test at the epoch of best val accuracy | P-code `train.py:180-184`. No early stop. |
| Batching | **mini-batch over train roots** (B = 512 roots, packed so that ≤ 600k lifted rows per batch; Section 7) | Paper silent; code is full-batch transductive. Mini-batching is mathematically the same model (BN statistics are per batch, as in any mini-batch BN). |
| Subgraph drop | **off** | P-paper: "we use the subgraph drop strategy, more details refer to." The reference is missing. GNN-AK-S trains only on sampled subgraphs and propagates to the rest, an approximation. Unnecessary with root mini-batching. Ablation: `random`, rate 0.5. |
| Seeds | 5 runs, seeds 0-4, same split; report mean ± std | P-paper ("5 iterations ... same partitioned data"). Seed also controls walk sampling. |
| Splits | official: TwiBot-20 8,278/2,365/1,183; TwiBot-22 700k/200k/100k | P-paper "70/20/10", matching both official splits. |
| Class weights | none | Paper silent. Do not add. |

The official `train.py` reads data via `Planetoid(name='twi20')` "public" split semantics. Validation
is then the 500 nodes after the training block, not 2,365. The README says only "process the data
into the PyG Cora format", so the official val split is unknowable [P-code]. Use the official
dataset split.

## 6. Evaluation and published numbers

Metrics: accuracy, F1, precision, AUC [P-paper]. F1 type is not stated. SEGCN's BotRGCN row
(57.50) equals the benchmark's **bot-class** F1, and TwiBot-20 F1 > accuracy, so F1 is assumed to
be **bot-class (binary) F1** [U]. Released code computes accuracy only.

Paper Table 5 (%; mean ± std over 5 runs):

| Dataset | Acc | F1 | Precision | AUC |
|---|---|---|---|---|
| TwiBot-20 | 87.01 ± 0.08 | 88.74 ± 0.13 | 85.83 ± 0.06 | 93.79 ± 0.32 |
| TwiBot-22 | 82.71 ± 0.16 | 59.31 ± 0.12 | 77.23 ± 0.17 | 82.31 ± 0.49 |

SEGCN is not in the TwiBot-22 benchmark. No independent re-run was found (Section 2).

**Forensics on Table 5 baseline rows.** Compared with the benchmark README / NeurIPS tables at
2-decimal precision:

| Row | SEGCN Table 5 | Benchmark | Verdict |
|---|---|---|---|
| TwiBot-22 BotRGCN | acc 79.66 ± 0.14, F1 57.50 ± 0.12, P 74.81 ± 0.21 | acc 79.66, F1 57.50 ± 1.42, P 74.81 ± 2.22 | Means **copied**. Stds do not match the benchmark (1.42 → 0.12, 2.22 → 0.21). |
| TwiBot-22 GCN | acc 72.39, F1 44.80, P 71.19 | acc 78.39, F1 54.96, P 71.19, **recall 44.80** | P copied. "F1" is the benchmark's **recall**. Acc differs by one digit (78.39 → 72.39). |
| TwiBot-22 GAT | acc 78.36, F1 55.86, P 72.23 | acc 79.48, F1 55.86, P 76.23 | F1 copied. P differs by one digit (76.23 → 72.23). |
| TwiBot-22 SATAR | 78.71 / 57.10 / 74.07 | "–" (benchmark could not run SATAR on TwiBot-22) | Not from the benchmark; provenance unknown. |
| TwiBot-20 SATAR | 84.02 / 86.07 / 81.50 | 84.0 / 86.07 / 81.50 | Copied; stds differ. |
| TwiBot-20 GAT | 83.27 / 85.25 / 81.26 | 83.3 / 85.25 / 81.39 | F1 copied. |
| TwiBot-20 BotRGCN | 84.61 / 87.07 / 83.79 | 85.75 / 87.25 / 84.52 | Not copied; lower than benchmark. |
| TwiBot-20 GCN | 74.64 / 77.03 / 73.17 | 77.53 / 80.86 / 75.23 | Not copied; lower. |

The paper also claims "the models are trained and tested for 5 iterations" for all baselines
[P-paper], which is inconsistent with copied means and non-matching stds. The BotRGCN baseline on
TwiBot-20 is about 1.1 acc below the benchmark, which inflates SEGCN's claimed "+2.4%". Other
errors: the Table 5 caption says "three benchmark datasets" (there are two); Table 2 swaps
TwiBot-20 bot/human counts (it gives bot 5,273 / human 6,589; the truth is 6,589 bots / 5,237
humans) and gives 229,573 users (the release has 229,580).

## 7. Scale plan (our hardware; mandatory memory budget)

**Requirement (common brief, HARD MEMORY CONSTRAINT):**
- Every step runs through `tools/memrun` (MemoryMax 10G, no swap, serialised by lock).
- Design target: peak RSS ≤ 8 GB.
- Never load a full TwiBot-22 tweet file, edge file or full feature table into RAM. Features stay
  memmapped and are streamed to the GPU in chunks; the graph is on-disk CSR.
- DataLoader workers: **0**. Batches are assembled on the GPU from GPU-resident indices.

### 7.1 Why the official code cannot be used for the subgraph path

`k_hop_subgraph` / `random_walk_subgraph` allocate dense `node_mask` [N,N] bool and
`hop_indicator` [N,N] int64. `extract_subgraphs` then builds `edge_mask` [N,E] bool [P-gnnak
`subgraph_extractors.py`, `transform.py:121`].

| Dataset | N | node_mask (N² B) | hop_indicator (8 N² B) | edge_mask (N·E_dir B) |
|---|---|---|---|---|
| TwiBot-20 (all nodes) | 229,580 | 52.7 GB | 421.6 GB | 229,580 × ~456k ≈ 105 GB |
| TwiBot-20 (labelled only) | 11,826 | 0.14 GB | 1.12 GB | small |
| TwiBot-22 | 1,000,000 | 1.0 TB | 8.0 TB | 1e6 × ~7.5M ≈ 7.5 TB |

The authors' server had 128 GB RAM and 2×24 GB GPUs [P-paper]. So the released code could not
have produced the TwiBot-22 numbers, and could only have run TwiBot-20 on a heavily reduced node
set. We must use a sparse extractor.

### 7.2 Sizes

- TwiBot-22: following+followers user-user edges = 3,743,634 directed [S: HW-GNN Table 3; equal to
  the benchmark BotRGCN `edge_index`]. Symmetrised undirected CSR has ≤ 7.49M entries.
- TwiBot-20: 227,979 edges [S: HW-GNN Table 3]; ≤ 456k entries.
- s̄ = mean distinct nodes per 30-step walk. Unknown; bounded by 31. Estimates use s̄ = 20
  (typical) and 31 (worst).

### 7.3 Steps, cost and peak memory

| Step | Work | Host peak RSS | VRAM | Time |
|---|---|---|---|---|
| A. RoBERTa features | Reused from the BotRGCN pipeline (shared layer). TwiBot-22: 1M descriptions + about 20M tweets at about 2,000/s → about 2.9 h. TwiBot-20: about 11.8k users × ≤200 tweets ≈ 2.3M tweets → about 20 min. **Zero marginal cost if BotRGCN already ran.** | about 3 GB (torch + tokenizer + batch of 512 texts; streamed jsonl input; output memmap written per batch) | about 2 GB (roberta-base fp16 0.25 GB + activations 512 × 128 tokens) | as given |
| B. Undirected CSR | Read the two CSRs (memmap), concatenate (src,dst) and (dst,src) int32: 7.49M × 2 × 4 B = 60 MB; lexsort (int64 keys 60 MB, plus argsort 60 MB); dedup; write `indptr` int64 (8 MB) and `indices` int32 (30 MB). | < 0.5 GB | 0 | < 1 min |
| C. RW subgraph store (**the case to watch**) | Process roots in chunks of 50k. Walk (numba or torch_cluster on CPU): 50k × 31 int32 = 6.2 MB. Per row: unique nodes (≤31); induced edges by testing the ≤465 pairs with binary search in sorted CSR rows (hubs cost log deg). Append to on-disk arrays: `nodes` int32 [N,31] padded −1 (TwiBot-22 124 MB), `n_nodes` uint8 [N], `eptr` int64 [N+1] (8 MB), `epairs` uint8 [E_sub,2] local indices. E_sub worst case 1M × 465 = 465M pairs = 0.93 GB on disk; realistic ≈ 2·s̄ per root → about 40M pairs = 80 MB. Memory is **independent of N** apart from 38 MB CSR. | < 0.6 GB (CSR 38 MB + chunk buffers ~100 MB + numba/py ~400 MB) | 0 | TwiBot-22: 1M × 465 × ~50 ns ≈ 25 s for pair tests, under 3 min total with 16 threads. TwiBot-20: seconds. Once per seed (5 stores, ≤ 1.2 GB each worst case). |
| D. GPU residency | Copy features to GPU in 64 MB chunks through one pinned buffer. TwiBot-22: des + tweets 1M × 1,536 × 2 B (fp16) = **3.07 GB** + num/cat 1M × 8 × 4 B = 32 MB. TwiBot-20: 0.71 GB. Copy the subgraph store: TwiBot-22 nodes 124 MB + eptr 8 MB + epairs 80 MB (≤ 0.93 GB worst). | ≤ 3 GB (CUDA context ~1.5 GB + pinned 64 MB + memmap pages, reclaimable) | TwiBot-22: 3.3 GB typical / 4.1 GB worst. TwiBot-20: 0.8 GB. | seconds (disk at 70 MB/s from the Windows drive: 3.1 GB ≈ 45 s; copy features to local ext4 first) |
| E. Training, per batch | B roots. Layer L needs h_{L-1} for U = ∪ V_u (\|U\| ≤ 31B). Each of those needs its own subgraph at layer 1. Lifted rows ≈ B·s̄² + B·s̄ (≤ 992B). At B = 512, s̄ = 31: 508k rows. Saved activations ≈ 9 tensors × 128 × 4 B = 4.6 KB/row → **2.3 GB**. At s̄ = 20: 0.98 GB. Inner GCN **must use sparse aggregation** (torch.sparse CSR spmm or PyG SparseTensor), not per-edge message tensors. Per-edge materialisation at ≤930 dir. edges × 16k subgraphs ≈ 15M edges × 512 B = 7.6 GB would blow VRAM. Pack roots so lifted rows ≤ 600k (≈ 2.8 GB activations). Gather h0 for the ≤ 16k nodes of U plus layer-1 subgraph nodes from the GPU feature table (no host traffic). | ≤ 3 GB (unchanged from D; no workers) | 3.3 (resident) + ≤ 2.8 (activations) + ~0.5 (index tensors, spmm workspace) + ~1 (allocator slack) ≈ **7.6 GB worst, ~5.5 GB typical** | below |
| F. Evaluation (exact, layer-wise) | No autograd, eval-mode BN. h0 for all N: 1M × 128 × 2 B = 256 MB. h1 for all N in chunks of 32k roots (≤ 1M lifted rows → ~0.5 GB transient). Store h1 [N,128] fp16 = 256 MB. Layer 2 + head on val+test roots (300k), or all N for ExtScore export. | ≤ 3 GB | 3.3 + 0.26 + 0.26 + 0.5 ≈ **4.4 GB** | TwiBot-22: about 1M × s̄ × 2 ≈ 40M rows fwd ≈ 10-20 s/eval |

**Per-epoch training cost.** Lifted rows per epoch = |train| × (s̄² + s̄).

- TwiBot-22 (700k train roots): s̄ = 20 → 294M rows; s̄ = 31 → 694M rows.
- Assumed throughput: about 5M rows/s fwd+bwd for 128-d gather + spmm + 2 linears + 2 BN on a
  4070 Ti. This is conservative and Python/index-build bound [U].
- Plus about 1,400 batches × ~30 ms overhead.
- Result: **about 1.7 min/epoch (s̄ = 20) to about 3 min/epoch (s̄ = 31)**, plus 15 s eval.
- 100 epochs → about 3-5.5 h per seed. **5 seeds ≈ 15-28 GPU h.**
- Precompute (C) about 15 min total. RoBERTa (A) is shared with BotRGCN (about 3 h if not yet
  done).
- TwiBot-20 (8,278 train roots): 3.3M-8.2M rows/epoch ≈ 1-2 s + eval about 2 s → under 10 min
  per seed. **5 seeds under 1 GPU h.**

Fallback if VRAM is short: B = 256, or keep the fp16 tweet/des table on the host as a memmap and
gather per batch. The gather is ≤ (31B + 961B) rows × 3 KB ≈ 1.5 GB per batch at B = 512, which
is too slow, so prefer reducing B.

## 8. ExtScore column

- Default injection (`extscore: p`): append `[z, m]` to the **numeric property vector before its
  typed encoder**. Width goes 6 → 8 (TwiBot-20) / 5 → 7 (TwiBot-22) in
  `n = LReLU(Lin(n_num, 32)(num))`. h0 width stays 128, and nothing else changes.
  - Transform: shared-layer neural transform `z = (logit(clip(p,1e-4,1-1e-4)) − μ)/σ`, with μ and σ
    fit on observed train rows; `z = 0` and `m = 1` where missing.
  - Support/unlabelled nodes without a ExtScore value get `z = 0`, `m = 1`. Do not z-score `[z,m]` again
    with the other numeric columns.
- Why there: SEGCN lifts every node's h0 into every subgraph containing it. So a node's ExtScore also
  reaches its walk neighbours' subgraph encodings, a "neighbour ExtScore" signal BotRGCN-style models
  get only through 1-hop aggregation.
- Ablations:
  1. **Late fusion**: concat `[z, m]` to `h_L` before the head: `Lin(130,128,bias=False)`.
  2. **Own block**: a fifth typed encoder `LReLU(Lin(2,32))`, with h0 widened to 160 and all H
     raised to 160.
  3. **Root-only**: inject `[z,m]` into `centroid_u` only (concat before `out_encoder`, 258 → 128)
     so neighbours' ExtScore is not used.
- `extscore: none` must give bit-identical inputs to the unmodified model (no extra columns).
- Leakage: the ExtScore for val/test nodes must come from a decision model that never saw their labels.
  Because of lifting, a leaky ExtScore on a support or train neighbour also leaks into test roots'
  subgraphs.

## 9. Pitfalls

1. **The released config never runs SEGCN.** `train/configs/twi20.yaml` (a header still reading
   "Cora Node Classification Configuration") sets `mini_layers: 0`. In `SEGCN.forward`
   `num_inner_layers == 0` takes the "Standard message passing" branch: a plain `GCNConv` per
   layer with BN, ReLU, dropout and residual. The subgraph kernel is never called, although the
   transform still builds subgraphs [P-code `core/model_segcn.py:261-266`]. It also uses
   `hops: 2, walk_length: 0` (ego-nets, not the paper's random walks), hidden 64, dropout 0.5,
   wd 5e-4, Adam, 200 epochs and seed 42 with 1 run. None of these match Table 4.
2. The official extractor is O(N²) memory (Section 7.1). Do not port it.
3. `train.py` applies `NormalizeFeatures()` (row-sum normalisation) to the input. That is
   meaningless for signed RoBERTa/z-scored features, and a Planetoid public split gives a 500-node
   val set. Do not reproduce either.
4. GNN-AK lifts rows: a hub node appears in many subgraphs. BN in the inner GCN sees lifted rows, so
   batch statistics are degree-weighted. This is faithful to GNN-AK; keep it.
5. Subgraph-local GCN normalisation: compute degrees inside each subgraph, **not** from the global
   graph.
6. The context embedding (GNN-AK+) is not local to u's own subgraph. Adding it breaks the
   mini-batch locality analysis in 7.3 (it needs every subgraph containing u). Keep it off unless
   ablating, and if so build the inverse index (node → subgraphs containing it) once from the
   store.
7. TwiBot-20 graph shape: only the 11,826 labelled users have neighbour lists (≤ 20 each). The
   graph is roughly bipartite labelled ↔ support and has very few support-support edges. Triangles,
   the paper's stated motivation, are rare by construction, and 30-step walks mostly bounce
   between a labelled hub and its leaves. Expect a small gain over GCN on TwiBot-20.
8. Table 1's TwiBot-20 categorical fields do not exist in TwiBot-22 (Section 3.3).
9. Transductive leakage: the official code computes the forward over the full graph including
   test nodes' features. That is standard and accepted for all TwiBot baselines. Labels of
   val/test nodes must never enter the loss, model selection uses val only, and the hop/subgraph
   construction is label-free.
10. Do not tune walk length, p, q or pooling on test. Any sweep uses val.

## 10. Open questions / unverified items

- [U] Which RoBERTa checkpoint was used, and how many tweets per user on TwiBot-22.
- [U] TwiBot-22 property feature set (we assume the benchmark's 5 + 3).
- [U] Meaning of "Network layers 2" (outer 2 × inner 1 assumed).
- [U] Values of repeat, p, q, pool, and hop labelling. Whether context encoding and subgraph drop
  were used, and with which sampling mode and rate (the reference is missing in the paper).
- [U] Whether subgraphs were resampled per epoch.
- [U] Whether F1 is bot-class (assumed) or macro.
- [U] Which code produced Table 5. The released code cannot run on TwiBot-22 and its config does
  not exercise the method. Several baseline cells are copied from the benchmark with transcription
  errors (Section 6). The 82.71 / 59.31 TwiBot-22 numbers should be treated as **unreproduced
  claims**.
- [U] s̄ (mean distinct nodes per walk) on both graphs. Measure it in step C and update 7.3.

## 11. Acceptance check

Because no runnable official configuration exists, use two levels:

1. **Implementation sanity (must pass).** With `L_in = 0` (the code's degenerate path: plain GCN
   + BN + residual + the same features), results should match the benchmark GCN within the
   benchmark's spread plus 1.5 points:
   - TwiBot-20: acc 77.5 ± 1.5, bot-F1 80.9 ± 1.5.
   - TwiBot-22: acc 78.4 ± 1.5, bot-F1 55.0 ± 2.0.

   Also run a unit test: C6 vs two disjoint triangles (2×C3), constant node features, `pool=sum`,
   walk length 30. Plain GCN (`L_in = 0`) must give identical node outputs on both. The subgraph
   path must differ, because the subgraph sizes are 6 vs 3.
   - With `pool=mean` and constant features the two are *not* separated, so test with sum or with
     `hop_dim > 0`.
   - The paper's Fig. 2 style pair, e.g. 4×4 rook's graph vs Shrikhande, is not a valid test: their
     ego-nets are 1-WL-equivalent.
2. **Paper claim (report either way).** SEGCN default, mean of 5 seeds, official splits:
   - "Faithful-compatible": acc within 1.0 of 87.01 and bot-F1 within 1.0 of 88.74 on TwiBot-20;
     acc within 1.0 of 82.71 and bot-F1 within 1.5 of 59.31 on TwiBot-22.
   - "Plausible": at least the benchmark BotRGCN level (TwiBot-20 acc ≥ 85.0; TwiBot-22 acc ≥ 79.0,
     bot-F1 ≥ 56.0).
   - Below that after the val-only sweep {pool ∈ {mean, sum}, p,q ∈ {(1,1),(1,0.5),(1,2)},
     L_in ∈ {1,2}, context on/off}: record as **not reproduced**. Do not tune further.
   - Our prior expectation: the TwiBot-22 +3 acc claim over BotRGCN will not reproduce
     (Section 6 forensics; Section 9.1-9.2).
