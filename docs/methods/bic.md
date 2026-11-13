# BIC: spec

Status: spec only, nothing implemented. Evidence labels: **[P-paper]** read in the paper (arXiv
2208.08320v2; the ACL 2023 camera-ready was checked too and has the same tables), **[P-code]** read in
the official code (path:line given), **[S]** secondary source, **[U]** unverified or assumed.

## 1. Summary

BIC is a supervised, transductive, text+graph detector. For each target user, a text module (a
transformer layer) runs over the sequence [description embedding, embeddings of up to 200 individual
tweets]. A graph module (a GNN layer followed by multi-head attention over the user's neighbours) runs
over BotRGCN-style node features. After the first layer, a text-graph interaction module mixes the
text "head" token (description position) and the user's graph vector with learned softmax similarity
weights. The text module's attention maps (tweet-to-tweet attention, [201 x 201]) are max-pooled to a
K x K grid and flattened into a "semantic consistency" vector. The classifier reads text, graph and
consistency representations. Graph: yes (follow relations; the reference code ignores relation type).
Text: yes, **per-tweet sequences** (the expensive part). Temporal: no. Encoder: frozen `roberta-base`
features, precomputed. The paper was evaluated on Cresci-15 and TwiBot-20 only, plus a private
10k-user COVID crawl (Table 3). **No TwiBot-22 result exists.** The repo has only TwiBot-22
preprocessing.

## 2. Citation and code

- Lei, Wan, Zhang, Feng, Chen, Li, Zheng, Luo. "BIC: Twitter Bot Detection with Text-Graph Interaction
  and Semantic Consistency". ACL 2023 (long), pp. 10326-10340. DOI 10.18653/v1/2023.acl-long.575,
  arXiv 2208.08320.
- Official code: https://github.com/LzyFischer/BIC, commit `b7d45921f80148798673a9022a775abd9c91eb64`
  (2024-11-11). **No LICENSE file**, so all rights are reserved by default. We may read it for
  reference only. No copying of code; write our own from this spec.
- The repo holds several divergent implementations [P-code]:
  - `TwiBot20/` (`model.py` class `BIC`, `train.py`): hidden 256, RGCN, 4-hop full NeighborLoader. The
    README says "For TwiBot-20 we recommend the code in the Twibot20 folder".
  - `TwiBot20v2/` (`model.py` class `InteractModel`, `train.py`): hidden 768, GCN, 16x16 consistency.
    **This is the implementation behind the headline number:** the shipped test predictions
    `TwiBot20v2/BIC20.json` score acc 87.91 / bot F1 89.17 (recomputed by us on 1,183 test users), and
    that matches the checkpoint name commented in `TwiBot20v2/train.py:118`
    (`87.91_89.17_86.49_92.03.model`). Paper Table 4 (hidden 768, dropout 0.5, M=2) also matches v2.
    `TwiBot20/BIC20text.json` and `BIC20graph.json` score 85.71/87.03 and 85.38/86.90, consistent
    with the paper's "w/o Graph" (85.44/86.97) and "w/o Text" (85.78/87.25) rows.
  - `Cresci15v2/`: the same model as TwiBot20v2 (lr 1e-3, cat_dim 1). The README recommends it for
    Cresci-15.
  - `TwiBot22/`: preprocessing only (embeds up to 200 tweets per user into 25k/50k-user chunks,
    `TwiBot22/main.py`, `data-process/cat.py`). No model and no results.
- Preprocessed data: Baidu Pan only (README). Not downloaded. Section 3.4 explains why we do not need it.

**Decision:** implement `variant: v2` (TwiBot20v2 semantics) as the reference, with the graph-index bug
fixed (Pitfall 1). `variant: v1` (the `BIC` class) is an optional ablation, described in 4.6.

## 3. Inputs

### 3.1 Graph-side node features: feature scheme `botrgcn` (see botrgcn.md), no deviation

v2 reads `data/num_properties_tensor.pt` [N,5], `cat_properties_tensor.pt` [N,3], `tweets_tensor.pt`
[N,768] (mean tweet embedding), `des_tensor.pt` [N,768] and `edge_index.pt` [P-code
`TwiBot20v2/train.py:44-48`; `InteractModel(num_property_dim=5, cat_property_dim=3)`]. These are the
BotRGCN processed_data file names and widths. Needed for all N graph nodes (incl. TwiBot-20 support).
`edge_type` is not read: the GCN merges both relations.

(v1 instead builds its own profile features (`TwiBot20/preprocess.py:72-103`): 6 numerical (followers,
following, tweet_count, listed_count, active days to **2022-02-01**, username length; z-scored with
population std over all users) + 3 categorical (protected, verified, default profile image). Only for
`variant: v1`.)

### 3.2 Text sequence (per target user)

- Token 0: the description embedding (the botrgcn `des` vector) [P-code `TwiBot22/data-process/cat.py`:
  `text = cat(des.unsqueeze(1), tweet)`; for TwiBot-20 the `text.pt` builder is not shipped, assumed the
  same [U]].
- Tokens 1..200: one embedding per tweet, for the **first 200 tweets** of the user in `edge.csv` `post`
  order [P-code `TwiBot20/data-process/process_id_tweet.py`, `userTweetEmbedding.py:62-67`],
  zero-padded to 200. Users with no tweets get 200 zero vectors. The paper says "We limit each user's
  tweet number to 200 ... vectors made up of all zeros" [P-paper App. C].
- Per-tweet embedding: `roberta-base` HF `feature-extraction` pipeline, one text per call, so the mean
  runs over **all tokens including `<s>` and `</s>`** and there is no padding. `max_length` = 512 in
  `userTweetEmbedding.py:55`; v1's `preprocess.py:15` uses `model_max_length=64`. Encoder frozen.
- Ours: shared-layer tweet store, newest first, cap 200 (the TwiBot-20 raw `tweet` field is the recent
  timeline; the `post` order of the benchmark conversion is assumed to be the same [U]). Encoder
  `roberta-base`, fp16, `max_len=128` (tweets are at most 280 characters; log the truncation rate; it
  should be under 0.1%). Mean over non-pad tokens **including** special tokens.
- Only **target users** (labelled train/val/test) need the sequence. Neighbours use only the graph-side
  features. TwiBot-20: 11,826 users. TwiBot-22: all 1,000,000 users (all labelled).

### 3.3 Neighbour lists (graph-module attention) and GCN edges

- MHA neighbour list for user i: `[i] + neighbours`, where neighbours = endpoints of every
  following/followers edge that has i as source **or** target, duplicates kept, self first [P-code:
  TwiBot-22 builder `TwiBot22/data-process/preprocess_3.py` (commented code) appends both directions for
  both relations; the TwiBot-20 `user_neighbor_index.json` builder is not shipped [U]]. Flag
  `dedup_neighbors` (default false).
- GCN edges: only edges **into** the target users (`NeighborSampler(edge_index, node_idx=seeds,
  sizes=[-1])` samples in-edges j -> seed, flow source_to_target) [P-code `TwiBot20v2/model.py:107-116`],
  using the botrgcn `edge_index` direction, relation types merged.

### 3.4 Can we rebuild everything without the Baidu Pan files? Yes.

| Baidu file (v2) | rebuild from |
|---|---|
| `num/cat_properties_tensor.pt`, `des_tensor.pt`, `tweets_tensor.pt`, `edge_index.pt`, `node_label.pt` | feature scheme `botrgcn` (shared layer) |
| `text.pt` [N_lab, 201, 768] | botrgcn des + per-tweet RoBERTa embeddings (3.2) |
| `user_neighbor_index.json` | CSR + transposed CSR of following/followers (3.3) |
| `split.csv`, `label.csv` | `users.parquet` |

What we cannot reproduce bit-exactly: the tweet order inside the 200 cap, the truncation length used for
the shipped embeddings (64 vs 512), duplicate handling in neighbour lists, and HF pipeline numerics.
None is essential. Sanity check: prediction agreement with `TwiBot20v2/BIC20.json` (section 11).

### 3.5 Needed from the shared layer

`users.parquet`; botrgcn blocks (`emb/roberta-base/des.f16.npy`, `emb/roberta-base/tweets_mean.f16.npy`,
num[5], cat[3]); **`emb/roberta-base/tweets_seq.f16.npy` [T,768] + `offsets` int64 [N+1], newest first,
cap 200, for target users** (support users may be skipped); `graph/following.csr.npz`,
`graph/followers.csr.npz` (TwiBot-22) or `following`/`follower` (TwiBot-20), **plus their transposes**
(in-neighbours). The transposes can be built by us; see 7. Optional `extscore/<tag>.parquet`.

## 4. Architecture (`variant: v2`, d = 768)

### 4.1 Encodings

- Graph input for every node u in the batch closure (targets + their neighbours):
  `x_u = concat(Lin(3->192)(cat), Lin(5->192)(num), Lin(768->192)(tweets_mean), Lin(768->192)(des))`,
  **no activation** [P-code `TwiBot20v2/model.py:38-42`]. The order is cat, num, tweet, des.
- Text input: S_i in R^{201 x 768} (3.2). **No input projection** (`self.text_linear` is defined and
  unused).

### 4.2 RespectiveLayer l (l = 1, 2) [P-code `model.py:76-121`]

- **Text**: pre-LN transformer layer `LModel(768, heads=4, dropout=0.1, LeakyReLU, norm_first=True)`:
  `S = S + Drop(MHA(LN1(S)))`, then `S = S + Drop(Lin2(Drop(LeakyReLU(Lin1(LN2(S))))))`, FFN width 768
  (not 4x). MHA attention dropout 0.1. **No key-padding mask**: zero-padded tweet slots take part.
  Output M_l = head-averaged attention weights [B, 201, 201] (from LN1(S)).
- **Graph**: `GCNConv(768, 768)` (PyG defaults: add self-loops, symmetric norm, bias) over the star
  edge set {j -> i : i target} on the closure nodes. Non-target nodes then get only the self-loop
  transform. Then, per target i: `h_i = MHA_g([h_i, h_n1, ..., h_nJ])[position 0]` with MHA_g of 4
  heads, no dropout, no residual, over self + the neighbour list (3.3). Vectorise: pad neighbour lists
  to J_max with a key-padding mask (equivalent to the unbatched per-user loop). Neighbours keep their
  GCN outputs.
- **Consistency**: `P_l = FixedPool16(M_l)`: zero-pad 201 -> 208 on the right/bottom, MaxPool2d(13,
  stride 13) -> [B,16,16].

### 4.3 Interaction (once, after layer 1) [P-code `model.py:124-166`]

With t = S[:,0] (text head) and g = graph vector of the target, both after layer 1:
- `a = <t, W_t t + b_t>`, `b = <g, t>`; `(a,b) = softmax(a,b)`; `t' = a t + b g`
- `c = <g, W_g g + b_g>`, `d = <g, t>`; `(c,d) = softmax(c,d)`; `g' = c g + d t` (uses the
  pre-update t)
- Write t' back to S[:,0] and g' to the target's graph vector. Layer 2's GCN uses the updated target
  vectors and the layer-1 neighbour vectors.

The paper's Eq. 2 differs: it puts a learnable theta on all four scores (`w_hg = h (theta2 g)`,
`w_gh = g (theta1 h)`). Flag `interaction: code|paper`, default `code`. In `code`, W_t and W_g have bias
and the cross terms are plain dot products.

### 4.4 Readout [P-code `model.py:44-66`]

`title = Lin(768->256)(S^(2)[:,0])`; `user = Lin(768->256)(g^(2))`; `att = Lin(512->256)(concat(flat
P_1, flat P_2))`; `logits = Lin(768->2)(concat(att, title, user))`. **No nonlinearity and no dropout** in
the head. `InteractModel.dropout` and `RespectiveLayer.dropout(0.5)` are defined but never used, so
the only active dropout is LModel's 0.1. A third `RespectiveLayer` (`Model_2`) and a second
`InteractLayer` are instantiated but unused. Do not create them; this changes only RNG/parameter count.

### 4.5 Loss

`CrossEntropyLoss()` (mean, no class weights) on targets. L2 through optimizer weight decay 1e-5.
Parameter count: about 19 M used (about 26 M allocated in the official code). The paper states 4.2 M
[P-paper App. C.2], another paper/code mismatch.

### 4.6 `variant: v1` (class `BIC`, `TwiBot20/model.py`), ablation only

d=256; text/graph inputs linearly projected (des 768->d, tweets 768->d; graph blocks to d/4 each),
then LeakyReLU+dropout. Per layer: text = plain MHA (4 heads, **with** key-padding mask, no FFN, no
residual); graph = `RGCNConv(d,d,2 relations)` + dropout + per-user MHA over self + union of in/out
neighbours (deduplicated). M=2 layers with the interaction only after layer 1 (`InterHead`: theta on all
terms, no bias; the graph update uses the *updated* text). Consistency: K=32 pooling + shared
`Lin(K^2->d)`, `semantic_mode=cat`. Residual from the input encodings, LeakyReLU+dropout. Classifier
`Lin(d->2)(LeakyReLU(Drop(Lin(3d->d)(concat(graph, sc_1, sc_2)))))`. **The text head is not in the
classifier** (commented out, `model.py:228`). Training: RAdam lr 1e-4, wd 1e-5, dropout 0.0, batch 64,
NeighborLoader `[-1]*4`, patience 8. Profile features from 3.1 (v1 note).

## 5. Training

| item | code (v2, `TwiBot20v2/train.py`) | paper Table 4 | use |
|---|---|---|---|
| optimiser | `torch.optim.RAdam`, lr 1e-4, wd 1e-5 (coupled L2) | "RAdamW", lr 1e-4, L2 1e-5 | code |
| batch | 64 target users (plain DataLoader over train idx, shuffle) | 64 | 64 |
| epochs | unbounded; stop after 8 epochs without val-acc improvement | 30 max, early stop 10 | patience 8, cap 30 |
| lr schedule | none | plateau, patience 5, factor 0.1 | none (code) |
| dropout | effectively 0.1 (4.4) | 0.5 | code |
| model selection | best val acc, **but** `best_model = model.state_dict()` stores references, so the evaluated model is the **last** one (8 epochs after best) | - | deep-copy the best state (fix); report the flag |
| runs | `while True: train()`; each run saves a file named with its **test** metrics | 5 seeds | seeds 0..4, report all |
| Cresci-15 | lr 1e-3 (`Cresci15v2/train.py:20`) | - | code |

Splits: train for gradients, val for early stopping and selection, test once at the end. Graph and text
features of val/test users are visible during training (transductive, standard). Their labels are not.

## 6. Evaluation and published numbers

Metrics: accuracy and **bot-class F1** (sklearn `f1_score` default, `TwiBot20v2/train.py:84-90`). The
paper does not say "binary", but the code is binary. 5 runs, mean (+- std) [P-paper Table 1].

| dataset | BIC acc | BIC bot F1 | source |
|---|---|---|---|
| Cresci-15 | 98.35 (+-0.24) | 98.71 (+-0.18) | [P-paper Table 1] |
| TwiBot-20 | 87.61 (+-0.21) | 89.13 (+-0.15) | [P-paper Table 1] |
| TwiBot-20, shipped predictions (one run) | 87.91 | 89.17 (macro F1 87.75) | our recomputation of `TwiBot20v2/BIC20.json` |
| Cresci-15, shipped predictions (one run, 535 test users) | 98.69 | 98.97 | our recomputation of `Cresci15/BIC15.json` |
| COVID crawl (5k/5k, random 8:2) | 67.25 | 67.78 | [P-paper Table 3], not reproducible (private) |
| TwiBot-22 | none | none | not evaluated in the paper; not in the TwiBot-22 benchmark |

TwiBot-20 ablations [P-paper Tables 1-2]: w/o Graph 85.44/86.97; w/o Text 85.78/87.25; w/o interaction
85.97/87.42. Baselines as run by BIC: BotRGCN 83.27/85.26, RGT 86.57/88.01.

## 7. Scale plan

**Memory rule (mandatory):** every job runs via `tools/memrun` (MemoryMax 10G, no swap, lock-serialised).
Target peak host RSS <= 8 GB. Never load a full TwiBot-22 tweet file, edge file or feature table. Stream,
chunk, memmap, on-disk CSR. DataLoader workers open memmaps themselves (never pickle arrays to them).
Count every worker's RSS and pinned buffers. Page cache from large writes is charged to the cgroup:
`flush()` the output memmap and `posix_fadvise(DONTNEED)` after each chunk. All artifacts must live on
the WSL ext4 disk, not on the Windows-mounted raw-data drive (about 70 MB/s).

The official v2 code holds `text.pt` fp32 in RAM: TwiBot-20 11,826 x 201 x 768 x 4 B = **7.3 GB**, plus
features on the GPU. On TwiBot-22 the same tensor would be 1M x 201 x 768 x 4 B = 617 GB. We use ragged
fp16 memmaps and assemble [B,201,768] per batch.

### 7.1 Per-tweet embedding precompute (the dominant cost)

Cost per tweet embedding (fp16, 768-d) = 1,536 B. Throughput assumption: 1,500-3,000 tweets/s (roberta-base
fp16, max_len 128, length-bucketed batches of about 512 on the 4070 Ti); central 2,000/s.

| dataset | tweets T | storage fp16 | encode time (1.5k / 2k / 3k per s) |
|---|---|---|---|
| TwiBot-20, 11,826 target users | <= 11,826 x 200 = 2.37 M (upper bound; release average about 146 per user) | <= 3.6 GB | <= 26 / 20 / 13 min |
| TwiBot-22, 1M users, cap 200 | <= about 88 M (total tweets in the release, so the cap can only reduce it) | <= 135 GB (+ 8 MB offsets) | <= 16.3 / 12.2 / 8.1 h |
| TwiBot-22, cap 50 (budget fallback, deviation) | <= 50 M, likely far fewer | <= 77 GB | <= 9.3 / 6.9 / 4.6 h |

Do this pass once in the shared layer. botrgcn `tweets_mean` can be derived from the same per-tweet
vectors if botrgcn.md accepts the same truncation and tweet selection (BotRGCN's TwiBot-22 script uses
the first 20 tweets, max_length 50, `TwiBot-22/src/BotRGCN/twibot_22/preprocess_2.py`). Otherwise it is a
separate, smaller pass.

Memory of the embedding job: host: stream the tweet store in chunks of 50k tweets (texts about 10 MB,
token ids 50k x 128 x 8 B = 51 MB, output 50k x 1,536 B = 77 MB) + torch/CUDA/model about 2.5 GB, so
**RSS about 3 GB**. VRAM: weights 0.25 GB + activations for batch 512 x 128 (FFN intermediate 512 x 128 x
3072 x 2 B = 0.4 GB) = **under 2 GB**. Write sequentially, and record offsets as we go.

### 7.2 Neighbour CSR (in + out)

The transpose of each relation CSR is built on disk by a chunked counting sort: per relation, int32
indices 4 B x E plus int64 indptr 8 MB. With E about 5 M user-user follow edges on TwiBot-22 [U, count
it]: 2 x 20 MB, RSS **under 0.5 GB**. Neighbour list for MHA = concat(out-following, in-following,
out-followers, in-followers). TwiBot-22 hubs: cap J at `max_neighbors=256` (random subset, resampled
each epoch; a deviation from the official full list, needed for memory). TwiBot-20: no cap needed
(each labelled user lists at most about 40 neighbours; in-edges come from labelled users only).

### 7.3 Training and evaluation

| | TwiBot-20 | TwiBot-22 |
|---|---|---|
| graph features | des+tweets_mean fp16 229,580 x 1,536 x 2 = 0.71 GB on GPU | 1M x 1,536 x 2 = **3.07 GB on GPU**, copied in 100k-row chunks (154 MB host each) |
| batch text tensor | 64 x 201 x 768 fp16 = 19.8 MB gathered from the memmap | same; IO per epoch 700k users x <= 88 tweets x 1,536 B <= 95 GB, about 60 s at 1.5 GB/s NVMe (23 min on the 70 MB/s drive, so do not use it) |
| host RSS | main 3 GB + 2 workers x 0.6 GB = **about 4.2 GB** | main 3 GB + 4 workers x 0.6 GB + pinned 4 x 20 MB = **about 5.5 GB** |
| VRAM | params+grad+Adam about 19 M x 16 B = 0.3 GB; text activations about 10 saved [64,201,768] fp32 tensors per layer (40 MB each) + attention 64 x 4 x 201^2 x 4 B = 41 MB (x2 with dropout masks), so about 1 GB for 2 layers; graph: closure <= 64 x 257 = 16.4k nodes x 768 x 4 B = 50 MB per tensor plus [64,257,257] x 4 heads attention = 67 MB, so about 0.5 GB. **Total about 2.5 GB** | 3.07 + 0.3 + 1 + 0.5 = **about 5 GB** (bf16 autocast optional) |
| time per epoch | 130 train batches x about 0.1 s = 15 s, + val/test 56 batches | 10,938 train batches x 60-100 ms = 11-18 min, + val 3,125 + test 1,563 forward batches (about 3 min), so **15-22 min** |
| epochs | about 15-30 (patience 8) | about 15-30, capped at 30 |
| GPU h per seed | about 0.15 | **4-10** |
| 5 seeds | under 1 GPU h (+ 0.3-0.5 h precompute) | **20-50 GPU h + 8-16 h precompute** |

Budget fallbacks for TwiBot-22 (each a recorded deviation): 3 seeds; `tweets_cap: 50`; train on a
stratified 200k subsample of the 700k training users (cuts the epoch to about 6 min). Validate the whole
pipeline on TwiBot-20 first.

## 8. ExtScore column

Transform: shared neural `[z, m]`. `extscore: none` reproduces the baseline exactly (no extra width, no extra
parameters).

**Primary (`extscore_at: graph_num`)**: append [z, m] to the botrgcn numerical block **before**
`Lin(5->192)`, so the width goes 5 -> 7. It applies to every node in the closure (targets and
neighbours). The GCN and neighbour attention can then use neighbours' ExtScore values (homophily), and the
interaction module can move the signal into the text head. Nodes without a score (TwiBot-20 support
nodes, unscored TwiBot-22 users) get z=0, m=1.

Ablations:
1. `extscore_at: head` (late fusion, recommended second): concat [z, m] to the readout vector, so
   `Lin(770->2)`. Cheap, and it shows whether BIC's representation adds information beyond ExtScore.
2. `extscore_at: graph_num+head`: both.
3. Not recommended: a ExtScore token in the text sequence. It would change the 201 x 201 attention map that
   the semantic-consistency module reads, so it no longer measures tweet-to-tweet consistency.

ExtScore values on val/test users are inputs, not labels. This is fine as long as the decision model was not
tuned on our test users.

## 9. Pitfalls

1. **Graph local-index bug (v2)**: `RespectiveLayer.forward` passes `adj[0]` from `NeighborSampler`, an
   edge index in **local** (re-numbered) ids, to `GCNConv` over the **global** feature matrix
   (`TwiBot20v2/model.py:112-115`). So messages go between wrong nodes: global nodes 0..|n_id|-1
   receive aggregates of unrelated nodes. The neighbour attention (MultiAttn) uses correct global lists.
   Our default is the intended semantics (4.2). Flag `reproduce_index_bug: true` exists only for
   forensic comparison. Reported numbers were produced with the bug, so a corrected model may differ
   by about +-1 point.
2. **`best_model = model.state_dict()` aliasing** (v1 and v2): the "best" model is the final one. Fix
   with a deep copy. Optionally report `select: last` to mimic.
3. **Cherry-picking risk**: an infinite run loop with test metrics in the file names. The shipped
   prediction file (87.91) is above the reported mean (87.61). Report all seeds.
4. Paper vs code: the paper says RGCN, code v2 uses GCN with merged relations. The paper says M=2
   interactions, the code has 2 layers but 1 interaction. The paper says dropout 0.5, effectively 0.1.
   The paper says 30 epochs / patience 10 / LR plateau, the code has no cap / patience 8 / no
   scheduler. The paper says 4.2 M parameters, the code has about 19 M. Equation 2 (theta on every
   term) differs from code (theta only on self-terms). Equation 3 concatenates h, g, d; v2 concatenates
   them after separate 768->256 projections (v1 drops h).
5. **No padding mask in v2**: the zero tweet slots take part in attention and in the consistency map.
   This is intended for faithfulness; v1 masks them. Do not "fix" it in v2.
6. Mean pooling for per-tweet embeddings must exclude padding when we batch (the official code
   encodes one text at a time) and must include `<s>`/`</s>`.
7. **Per-user Python loops** (neighbour extraction and MHA in v1/v2) are O(batch) kernel launches;
   vectorise with padded neighbour tensors plus a key-padding mask. MultiAttn's in-place sequential
   update (a later target sees an earlier target's updated vector if they are neighbours) is not
   reproduced; we update all targets in parallel (minor deviation).
8. Index order: v2 indexes users by `label.csv` order and BotRGCN tensors by `node.json` order, and
   assumes they agree. We use the shared `idx` everywhere.
9. Duplicated neighbours: TwiBot-22 lists can repeat a node across relations. Keep duplicates by
   default (`dedup_neighbors: false`), because v1 deduplicates and v2's builder appears not to.
10. The Cresci-15 numbers use the benchmark split. Our unified Cresci split is a different split, so it
    is not comparable.

## 10. Open questions / unverified

- The TwiBot-20 builders for `text.pt` and `user_neighbor_index.json` (v2) are not shipped. They are
  assumed to match the TwiBot-22 scripts [U].
- Truncation used for the shipped TwiBot-20 tweet embeddings: 512 (`userTweetEmbedding.py`) or 64
  (`preprocess.py`) [U].
- Whether the paper's 5-run mean used v2 with or without later edits. We assume the commit we read.
- TwiBot-22 follow-edge count and degree distribution (sets `max_neighbors` and the closure size) [U].
- How many TwiBot-22 users have more than 200 tweets (sets T). Count it in the shared layer before
  allocating 135 GB.

## 11. Acceptance check

5 seeds, `variant: v2`, index bug fixed, `extscore: none`:
- **TwiBot-20 (official split): mean acc >= 86.0 and bot F1 >= 87.6** (paper 87.61 / 89.13; tolerance
  about 1.5 points for the bug fix and re-embedding). Also, test-prediction agreement with
  `TwiBot20v2/BIC20.json` should be at least 90%. Our "w/o graph" ablation should land near 85.4 / 87.0.
- With `reproduce_index_bug: true`, mean acc >= 86.5 is expected if our inputs match the originals.
  This is a diagnostic only.
- **TwiBot-22**: no published reference. Plausibility floor: acc >= 76 and bot F1 >= 50 (the range of
  BotRGCN re-runs). Below that, suspect a bug. Report it as a new number, not a reproduction.
- Cresci-15 (optional, needs the benchmark split): acc >= 97.5, bot F1 >= 98.0.
