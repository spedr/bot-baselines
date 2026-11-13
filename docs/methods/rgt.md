# RGT: Relational Graph Transformer (Feng, Tan, Li, Luo 2022)

Spec status: written 2026-11-12 from the paper (arXiv v3), the official repo and the TwiBot-22 benchmark
re-implementation. Labels: **[P-paper]** read in paper, **[P-code]** read in code (path given),
**[S]** secondary, **[U]** unverified/assumed.

## 1. Summary

RGT is a graph neural network for account-level bot detection on a user-user graph with typed edges
(follower, following). Each user starts from the BotRGCN feature vector: RoBERTa description
embedding, mean RoBERTa tweet embedding, numeric profile properties and boolean properties. Four
per-block linear layers project these features and a fully connected layer fuses them. Two RGT
layers follow. Each layer runs one graph-transformer convolution per relation (multi-head dot-product
attention over that relation's in-neighbours), applies a gated residual per relation, then fuses the
relation-specific outputs with HAN-style semantic attention. A 2-layer MLP head outputs 2 classes
and the loss is cross-entropy. Family: heterogeneous (relational) GNN, transductive node
classification. Uses the graph and text (through precomputed embeddings). No temporal modelling.
Encoder: frozen `roberta-base` features only, owned by feature scheme `botrgcn` (see botrgcn.md).

## 2. Citation and code

- Feng, S., Tan, Z., Li, R., Luo, M. (2022). Heterogeneity-Aware Twitter Bot Detection with Relational
  Graph Transformers. AAAI 36(4):3977-3985. DOI 10.1609/aaai.v36i4.20314, arXiv 2109.02927.
- Official code: https://github.com/BunsenFeng/BotHeterogeneity, commit
  `0f74fc48a4849abf79ba3042272a4a24790ced3a` (2022-03-01). **No licence file**: read it for reference
  only and copy nothing verbatim. It covers TwiBot-20 only and its paths are hard-coded to Colab (`TBD.py`).
- Benchmark re-implementation (the source of the TwiBot-20/22 numbers we compare against):
  https://github.com/LuoUndergradXJTU/TwiBot-22, `src/RGT/`, commit
  `54adf9a1dbb6dc238c932d3147a1875def5817d8`, **MIT** (reuse is allowed with attribution, but we write
  our own code). Files: `Twibot-20/Heterobot.py` (full-batch), `Twibot-22/Heterobot_sample.py`
  (NeighborLoader), `Twibot-22/layer.py` (identical to `Twibot-20/layer.py`), `readme.md`.
- Pinned environment in the benchmark [P-code `src/RGT/readme.md`]: CUDA 10.2, torch 1.9.1,
  torch-geometric 2.0.2, pytorch-lightning 1.4.9, torch-sparse 0.6.12, torch-scatter 2.0.9.

## 3. Inputs

### 3.1 Node features: feature scheme `botrgcn` (see botrgcn.md)
RGT consumes exactly the four BotRGCN blocks [P-code `Heterobot_sample.py` `load_data`, `readme.md` item 1]:

| block | width (benchmark) | role |
|---|---|---|
| `cat_properties` | 3 (`--cat_num 3`) | boolean properties |
| `num_properties` | 5 (`--numeric_num 5`) | z-scored numeric properties |
| `tweets` | 768 | mean roberta-base tweet embedding |
| `des` | 768 | mean roberta-base description embedding |

How RGT loads these differently from BotRGCN (verified in code):
- **TwiBot-22 (sampled):** the four tensors are concatenated into one `x` in the order
  `[cat | num | tweets | des]` (width 1544) so that NeighborLoader can slice rows. Each training
  step slices the columns back apart [P-code `Twibot-22/Heterobot_sample.py` L21, L68-71]. This is a
  storage detail only, with no maths change. We keep separate memmaps and gather rows per block.
- **TwiBot-20 (full-batch):** the four tensors stay separate and are held on the GPU [P-code `Twibot-20/Heterobot.py` L24-27].
- No extra normalisation, no self-loops and no reverse edges are added. The input projection differs
  from BotRGCN: each block goes through `Linear(block_width, 32)` + LeakyReLU + Dropout, and then
  `Linear(128,128)` + LeakyReLU + Dropout (section 4). Use the 5/3 widths on both datasets. The
  argparse defaults are 5 and 3 for TwiBot-20 too [P-code `Twibot-20/Heterobot.py` L152-154]. If
  botrgcn.md settles on other widths for TwiBot-20, read the widths from the tensors.
- The official AAAI code used a different feature set: 7 numeric features (z-scored with one global
  scalar mean/var over the whole matrix, a bug), 11 boolean features declared
  (`IN_FEATURES_BOOL = 11`; the bool script emits 7) [P-code BotHeterogeneity `TBD.py` L259-270,
  `data process/user_features_numeric.py` L46]. We do **not** use it. The benchmark numbers come from
  the BotRGCN 5/3 scheme.

### 3.2 Graph
- Nodes: all users. TwiBot-22: 1,000,000, all labelled. TwiBot-20: 229,580, of which 11,826 are
  labelled and the rest are support users.
- Two relations, user-user only:
  - TwiBot-22: `followers` (type 0) and `following` (type 1), from `edge.csv`, **stored source->target
    exactly as in the release**, with no deduplication and no symmetrisation [P-code
    `src/BotRGCN/twibot_22/preprocess_1.py` L49-69]. Release semantics: `following` = "source follows
    target", `followers` = "source is followed by target" [P-paper TwiBot-22 Table 5]. Counts:
    following 2,626,979, follower 1,116,655, total 3,743,634 [P-paper TwiBot-22 Table 7]. Our
    `edge-002.csv` may differ slightly [U]. The graph builder should log the counts.
  - TwiBot-20: `follower` and `following` from the `neighbor` field. Edge = (list owner -> listed
    neighbour) [P-code BotHeterogeneity `data process/follower_following_edge.py` L26-45]. The paper
    reports 455,958 follow relationships in total [P-paper].
- Message direction: PyG `flow="source_to_target"`, so **the target node aggregates from its sources**.
  Replicate this exactly. Consequence on TwiBot-20: only list owners have out-edges, so a labelled user
  receives messages only from other labelled users that list it. Many labelled nodes therefore have
  in-degree 0 in each relation and rely on the gate and root/skip terms [U: inferred from code; the
  graph builder must log the in-degree-0 fraction]. Do not "fix" this in the faithful run. Reverse edges
  are an ablation (section 9).
- Relation index order (0 = followers/follower, 1 = following) does not affect the maths. Keep it
  fixed for checkpoint compatibility.

### 3.3 What we need from the shared data layer
- `users.parquet`: `idx`, `split`, `label`.
- Feature scheme `botrgcn` outputs, as built per botrgcn.md: `emb/roberta-base/des.f16.npy` [N,768],
  `emb/roberta-base/tweets_mean.f16.npy` [N,768] (the tweet subset and pooling are defined there), and
  the 5 numeric and 3 boolean property columns (derived from `profile.parquet` per botrgcn.md).
- `graph/followers.csr.npz` + `graph/following.csr.npz` (TwiBot-22), or `graph/follower.csr.npz` +
  `graph/following.csr.npz` (TwiBot-20), source -> target. The loader converts them to one
  `edge_index` [2,E] int64 plus an `edge_type` [E] uint8, or to per-relation CSC (incoming lists) for
  sampling.
- `extscore/<tag>.parquet` when `extscore: p`.
- Nothing else: no per-tweet data, no timestamps, no non-user nodes.

## 4. Architecture

Notation: H = 128 hidden, C = transformer heads, D = semantic heads, R = 2 relations, p = dropout.
LeakyReLU uses the torch default slope 0.01 [P-code `nn.LeakyReLU()`].

**Input encoder** [P-code `Heterobot_sample.py` L42-46, L77-83]
```
n = Drop(LReLU(Linear(5 [7 with ExtScore] -> 32)(num)))
b = Drop(LReLU(Linear(3 -> 32)(cat)))
t = Drop(LReLU(Linear(768 -> 32)(tweets)))
d = Drop(LReLU(Linear(768 -> 32)(des)))
x0 = Drop(LReLU(Linear(128 -> 128)(concat[n, b, t, d])))      # concat order: num, bool, tweet, des
```
Paper Eq. (1) describes a single FC layer `x0 = LReLU(W_I x + b_I)` [P-paper]. Code is authoritative.

**RGT layer l** (applied twice; in = out = 128, because the gate requires in == out)
[P-code `layer.py` L39-82; P-paper Eqs. 2-9]

For each relation r in {0,1}, with E_r the edges of type r:
1. Relational graph transformer = PyG `TransformerConv(128, 128, heads=C, concat=False, dropout=p,
   root_weight=True, beta=False, bias=True)`, one independent instance per relation. For target i and
   head c:
   - `q_c,i = W^r_{c,q} x_i + b`, `k_c,j = W^r_{c,k} x_j + b`, `v_c,j = W^r_{c,v} x_j + b`. **Each head
     has its own full 128-d projection** (the projection widths are C*128, not 128/C).
   - `alpha_c,ij = softmax_{j in N_r(i)} (q_c,i . k_c,j / sqrt(128))`, followed by dropout(p) on alpha
     during training.
   - `u^r_i = (1/C) sum_c sum_j alpha_c,ij v_c,j  +  W^r_skip x_i + b_skip` (heads averaged, then the
     root/skip term is added).
   - The `W_skip x_i` root term is **not in the paper** (paper Eq. 4 has only the attention sum). It
     comes from the TransformerConv default `root_weight=True` [P-code PyG 2.0.2 `transformer_conv.py`]
     and is kept. A node with no in-edges gets `u = W_skip x_i + b`.
2. Gated residual, **one gate shared by both relations** within the layer (`self.gate` is a single
   module) [P-code `layer.py` L42-45, L71, L78]:
   - `a^r_i = sigmoid(W_A [u^r_i ; x_i] + b_A)`, where W_A is 128 x 256 and the concat order is (u, x).
   - `h^r_i = tanh(u^r_i) * a^r_i + x_i * (1 - a^r_i)`.

   Paper Eqs. 5-6 write `W_A^r`, which is ambiguous about per-relation parameters, while the code shares
   one gate. **Use the shared gate.**
3. Semantic attention across relations, with D heads [P-code `layer.py` L7-37; P-paper Eqs. 7-9]:
   - Stack `Z = [h^0, h^1]` with shape [N, R, 128].
   - For each head d: `s_d = Linear(128->128) -> tanh -> Linear(128->1, bias=False)` applied to Z gives
     [N,R,1]. Take the **mean over all nodes in the current graph or subgraph** to get [R,1], then
     `beta_d = softmax over R`.
   - `x^(l)_i = (1/D) sum_d sum_r beta_d,r h^r_i`. The scorer's hidden width is 128 (default
     `hidden_size=128`).
4. After each RGT layer the detector applies LeakyReLU, **without dropout**:
   `x1 = LReLU(RGT1(x0))`, `x2 = LReLU(RGT2(x1))` [P-code `Heterobot_sample.py` L85-86].

Code quirk: `RGTLayer.forward` has its `return` inside the relation loop (`layer.py` L82). With R = 2
it returns after relation 1 is appended, which is correct. With R > 2 it would silently drop relations.
Our implementation must loop over all relations.

**Head and loss** [P-code L88-90]
```
y = Linear(64 -> 2)( Drop(LReLU(Linear(128 -> 64)(x2))) )
loss = CrossEntropy(y, label)          # unweighted, no label smoothing
```
Paper Eq. 11 adds an explicit L2 term λ = 3e-5, implemented as AdamW weight decay. P(bot) for export
is `softmax(y)[:,1]`.

**Init** [P-code `init_weight`]: every `torch.nn.Linear` (input encoder, gate, semantic scorers, head)
gets `kaiming_uniform_` weights and zero bias. TransformerConv's internal `torch_geometric.nn.Linear`
modules are **not** `nn.Linear` in PyG 2.0.2, so they keep the PyG default init (kaiming-uniform-like
`reset_parameters`) [P-code]. Check this again under the PyG version we pin: in modern PyG,
`torch_geometric.nn.Linear` is still not a subclass of `torch.nn.Linear` [U: verify at implementation].

**Hyperparameters**

| param | paper Table 2 [P-paper] | benchmark TwiBot-20 `Heterobot.py` [P-code] | benchmark TwiBot-22 `Heterobot_sample.py` [P-code] |
|---|---|---|---|
| layers | 2 | 2 | 2 |
| hidden | 128 | 128 | 128 |
| transformer heads C | 8 | **2** (default) | 2 |
| semantic heads D | 8 | **2** (default) | 2 |
| dropout p | 0.5 | 0.5 | 0.3 |
| optimiser | AdamW | AdamW (amsgrad False) | AdamW |
| lr | 1e-3 | 1e-3 | 5e-4 |
| weight decay | 3e-5 | 3e-5 | 1e-5 |
| batch size | 256 | default 128; readme command `--batch_size 128` | default 128; readme command `--batch_size 256` |
| max epochs | 40 | default 50; readme command `--epochs 200` | default 1000; readme command `--epochs 200` |
| scheduler | - | CosineAnnealingLR(T_max=16, eta_min=0), stepped per epoch | same |
| precision | - | PL `precision=16` (AMP) | same |

Recommendation: for faithfulness to the benchmark numbers, use the **benchmark column with the
readme command-line overrides**: TwiBot-20 C = D = 2, batch 128, 200 epochs; TwiBot-22 C = D = 2,
batch 256. Also run the paper config (C = D = 8, batch 256, 40 epochs) on TwiBot-20 as a secondary
check, because that is the config behind the AAAI number 86.64/88.21.

## 5. Training

### TwiBot-20 (full-batch, faithful to `Twibot-20/Heterobot.py`)
- Each step runs a full-graph forward over all 229,580 nodes with dropout on, then computes the loss on
  a mini-batch of 128 **training indices** (shuffled, `DataLoader(range(8278), batch_size=128)`, so 65
  steps per epoch) [P-code L54-72, `Dataset.py`].
- Validation: once per epoch on the full val set (2,365); compute acc and bot-F1 on val nodes only.
  Model selection: `ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=1)` [P-code L174-179];
  the best-val-acc checkpoint is then tested once on the test set (1,183).
- The CosineAnnealingLR(T_max=16) scheduler steps once per epoch. Past epoch 16 it does **not**
  restart: LR follows the periodic cosine, reaching 0 at epochs 16, 48, 80, ... and the maximum at 32,
  64, ... Reproduce this (PyTorch semantics) for faithfulness.
- Seeds: 5 runs, seeds 0-4 (`pl.seed_everything`). The benchmark reports mean ± std of 5 runs.

### TwiBot-22 (neighbour sampling)
Benchmark behaviour [P-code `Twibot-22/Heterobot_sample.py`]:
- `NeighborLoader(data, num_neighbors=[256]*4, input_nodes=train_idx, batch_size=args.batch_size,
  shuffle=True)`: 4 hops with fan-out 256 per hop, applied to all relations together (homogeneous
  `Data` with `edge_attr=edge_type`) (L233). **The readme claims "num_neighbors 20, 2 iterations"; the
  code uses [256]*4. Trust the code.** The `[20]` setting appears only in the unused TwiBot-20 sampled
  script.
- **The loss is computed over every node in the sampled subgraph, not just the seed nodes**:
  `loss = CE(out2(...), train_batch.y)` (L88-90). `label.pt` holds labels for all 1M users, since all
  TwiBot-22 users are labelled [P-code `BotRGCN/twibot_22/preprocess_1.py` L26-47]. Val and test users
  reached as neighbours therefore contribute their labels to training. This is **test-label leakage**.
- There is no validation loop: `trainer.fit(model, train_loader)`, and the val loader and checkpoint
  callback are commented out. PL's default checkpoint keeps the **last epoch**, which is the model
  tested (L238-246).
- Test: NeighborLoader with batch 200; only the first 200 rows (the seed nodes) are scored
  (L141, L157). 100,000 / 200 = 500 full batches, so the slice is exact. Metrics come from the
  concatenated predictions (L249-259).

**Our default ("clean") protocol for TwiBot-22.** This is a deviation; label it as such in results.
- NeighborLoader over in-edges with **2 hops**. Under PyG's directed sampling, hops 3-4 cannot affect
  seed outputs of a 2-layer model, so seed predictions match the 4-hop setting in distribution.
  Fan-out `[256, 256]` counted over the union of both relations, as in the benchmark (homogeneous
  sampler with an edge-type attribute). Seed batch 256.
- **Loss on seed nodes only** (the first `batch_size` rows of the subgraph).
- Validation every epoch on all 200k val seeds (batch 1024, same sampler, fixed sampler seed, no
  shuffle). Select on val accuracy (the benchmark's monitor); log bot-F1 as well. Max 50 epochs,
  early-stopping patience 10. The benchmark's "200 epochs, last epoch" is not affordable and uses no
  model selection.
- Semantic attention averages over **all nodes in the sampled subgraph at that layer**, as in the
  benchmark. As a result beta depends on the batch, and predictions depend slightly on batch
  composition. For deterministic evaluation use a fixed sampler seed and a fixed batch order.
- Keep the same optimiser, lr 5e-4, wd 1e-5, dropout 0.3, C = D = 2, cosine schedule and AMP.
- Optional `bench_compat: true` mode reproduces the leaky all-node loss with `[256]*4` fan-out. Run it
  only if the memory dry-run (section 7) shows it fits. Never report it as a clean baseline.

### Split usage
The official train/val/test splits on both datasets. TwiBot-20 support users are unlabelled nodes:
they are present in the graph and never in the loss.

## 6. Evaluation and published numbers

The metric set is accuracy, precision, recall and **bot-class (binary, positive = bot) F1**, plus AUC.
All benchmark numbers are mean (± std) over 5 runs.

| source | dataset | Acc | bot-F1 | notes |
|---|---|---|---|---|
| AAAI paper Table 1 [P-paper] | TwiBot-20 | 86.64 | 88.21 | single number, paper config |
| TwiBot-22 benchmark Tables 2/8 [P-paper arXiv 2206.04564] | TwiBot-20 | 86.6 ± 0.4 | 88.0 ± 0.4 | |
| same | TwiBot-22 | 76.5 ± 0.4 | 42.9 ± 0.5 | leaky all-node loss, last epoch |
| same | Cresci-2015 | 97.2 ± 0.3 | 97.8 ± 0.2 | |
| benchmark `src/RGT/readme.md` [P-code] | TwiBot-20 | 86.57 ± 0.41 | 88.01 ± 0.41 | P 85.15, R 91.06 |
| same | TwiBot-22 | 76.47 ± 0.45 | 42.94 ± 0.49 | P 75.03, R 30.10 |
| benchmark Table 3, "w/o graph" (2-layer MLP on the same features) [P-paper] | TwiBot-20 / TwiBot-22 | 82.6 / 75.4 | 83.8 / 41.2 | |

On TwiBot-22 RGT has very low recall (0.30) on a test set that is 14% bots. Bot-F1 there sits below
BotRGCN's 57.5.

## 7. Scale plan for our hardware

**Mandatory memory rule (common brief, HARD MEMORY CONSTRAINT).** Every job runs as
`tools/memrun [-m 10G] -- <cmd>`. Design target is peak host RSS <= 8 GB; 10 GB is the OOM-kill line.
Never load a full feature table, the edge file or a tweet file into RAM. Use memmaps and on-disk CSR,
and count DataLoader workers and pinned buffers in the budget. The figures below are estimates
(assumption: a torch + CUDA process baseline of about 2.5 GB RSS and about 0.5 GB VRAM context [U]).
Measure them on the first run and record them in the run log.

### 7.1 Precompute (owned by botrgcn.md; shared with BotRGCN, SimpleHGN and HGT)
RoBERTa embeddings are computed once by the `botrgcn` pipeline and are not repeated here. Order of
magnitude for TwiBot-22: <= 20 tweets/user x 1M users ≈ 20M tweets at about 2,000 tweets/s gives about
2.8 GPU-h, plus 1M descriptions at about 3,000/s, about 6 min. TwiBot-20: <= 20 x 229,580 ≈ 4.6M tweets,
about 0.6 h. RGT-specific preprocessing: building the edge arrays from the CSR files.
- TwiBot-22: E = 3.74M. `edge_index` int64 is 3.74M x 2 x 8 B = 60 MB, `edge_type` is 3.7 MB, and a
  CSC copy for the sampler (indptr 8 MB + indices 15 MB + permutation 30 MB) is about 50 MB.
  **Peak RSS ≈ 2.5 + 0.3 ≈ 3 GB**, VRAM 0.
- TwiBot-20: E ≈ 0.46M, negligible.

### 7.2 TwiBot-22 training (sampled)
Feature storage: des and tweets fp16 memmaps are 1M x 768 x 2 B = 1.54 GB each. num and cat are
1M x 8 x 4 B = 32 MB. Total ≈ 3.1 GB.
- Strategy: **copy all four blocks onto the GPU once, in 64k-row chunks** (chunk = 64k x 1536 x 2 B =
  197 MB host transient). VRAM then holds 3.1 GB resident. Per batch, only `n_id` moves to the GPU and
  the rows are gathered with `index_select` there. Host RSS stays free of the feature table. Fallback
  if VRAM is short: keep the memmaps on the host and gather `n_id` rows per batch on the CPU (page-cache
  backed; the cgroup may count up to 3.1 GB of reclaimable page cache, which stays under the limit).
- Subgraph size with 2 hops, fan-out 256, seed batch 256: hop 1 ≈ 256 x mean in-degree 3.7 ≈ 1k
  nodes; hop 2 is degree-biased, ≈ 1k x (50-100 capped at 256) ≈ 50-100k sampled edges and ≈ 40-80k
  unique nodes [U: estimate]. **Mandatory dry run:** sample 200 batches without training and log the
  percentiles of `|n_id|` and `|E_sub|`. If p99 `|n_id|` > 250k, drop the hop-2 fan-out to 64.
- VRAM per training step (fp16 AMP, C = 2, |n_id| = 80k, |E_sub| = 100k, split over 2 relations):
  - gathered inputs: 80k x 1546 x 2 B = 247 MB
  - encoder activations: about 10 tensors of 80k x 128 x 4 B (fp32 master copies), about 0.4 GB
  - per relation-layer: q, k, v gathered per edge E_r x C x 128 x 2 B x 3 ≈ 50k x 2 x 128 x 6 B =
    77 MB, plus alpha, skip, gate (80k x 256) and semantic scores ≈ 150 MB; x 4 relation-layers ≈ 0.6 GB
  - backward roughly doubles the activations, so ≈ 2.5 GB per step
  - **VRAM peak ≈ 0.5 ctx + 3.1 features + 2.5 step + model/optimizer (< 10 MB) ≈ 6-6.5 GB** (< 12 GB)
- Host RSS: 2.5 GB base + graph 0.15 GB + sampler state. NeighborLoader `num_workers=2`: each worker
  is a fork holding the CSC (shared copy-on-write, but count it as ≈ 0.4 GB RSS each) plus
  `pin_memory` buffers for `n_id` and edge tensors (< 10 MB per batch, prefetch 2). **Peak RSS ≈ 3.5-4 GB.**
- `bench_compat` ([256]*4, all-node loss): a 4-hop sample could cover a large share of the 1M nodes.
  At 500k nodes and 2M edges, VRAM is 500k x 1546 x 2 = 1.5 GB of inputs plus about 2M x 2 x 128 x
  6 B x 4 = 12 GB of per-edge tensors, **which does not fit**. Run it only if the dry run shows p99
  `|n_id|` < 150k; otherwise record "not reproducible on 12 GB".
- Time: 700k / 256 = 2,735 iterations per epoch. Estimated per iteration: sampling with pyg-lib ≈ 20
  ms + GPU forward/backward ≈ 30 ms ≈ 50 ms, so about 2.3 min per epoch. Val is 200k seeds at batch
  1024 ≈ 0.5 min. With 50 epochs max and early stopping at about 25 epochs, a run takes about 1-2 h
  [U]. **5 seeds ≈ 5-10 GPU-h.**
- Evaluation: same loader, 100k test seeds, batch 1024, `torch.no_grad`. Peak VRAM ≈ 3.1 + 1.5 GB;
  RSS as in training. Write P(bot) for all test (and val) users.

### 7.3 TwiBot-20 training (full-batch)
- Features: 229,580 x 1544 x 2 B = 0.71 GB fp16 (1.42 GB fp32), loaded to the GPU in chunks. Host
  transient < 0.2 GB. Edges ≈ 0.46M, 7 MB.
- VRAM per step, C = D = 2, AMP: encoder ≈ 10 x N x 128 x 4 B = 1.2 GB; per relation-layer, per-edge
  q/k/v ≈ 0.23M x 2 x 128 x 6 B = 0.35 GB plus node tensors (skip, gate, semantic) ≈ 0.25 GB, giving
  ≈ 0.6 GB; x 4 = 2.4 GB; backward x 2. **Peak ≈ 0.5 + 0.7 + ~5 GB ≈ 6 GB.**
- With C = D = 8 (paper config), per-edge tensors grow 4x to ≈ 1.4 GB per relation-layer, and the
  peak ≈ 0.5 + 0.7 + 2 x (1.2 + 4 x 1.7) ≈ 12-17 GB, **which may not fit**. Then switch the paper-config
  check to sampling with full neighbourhoods (TwiBot-20 in-degrees are small), using batch 256 seeds
  and 2-hop `[-1,-1]`. The math is unchanged except that semantic attention averages over subgraph
  nodes.
- Host peak RSS ≈ 3 GB.
- Time: 65 steps per epoch x 200 epochs = 13k full-graph steps at an estimated 60 ms each, ≈ 15 min
  per run. **5 seeds ≈ 1.3 GPU-h**; the paper config adds about 0.5 h.

**Total RGT: ≈ 7-12 GPU-h**, excluding the shared RoBERTa precompute (≈ 3.5 GPU-h, owned by botrgcn.md).

### 7.4 Library versions (modern equivalents)
- Use current torch (2.x) and torch_geometric (≥ 2.5), with `pyg-lib` or `torch-sparse` for sampling.
  Do not use pytorch-lightning; write a plain training loop that replicates per-epoch val and
  best-val-acc selection.
- `TransformerConv`: same semantics in modern PyG (`concat=False` averages heads; `root_weight=True`
  adds `lin_skip(x_i)`; attention dropout applies to alpha). Modern versions compute q/k/v on nodes and
  gather to edges, which is equivalent and cheaper. Verify on a toy graph by copying weights.
- `torch_geometric.loader.NeighborLoader`: in modern PyG it samples incoming edges, places the seed
  nodes first and returns `batch.n_id`, `batch.e_id` and `batch.batch_size`. Pass
  `edge_attr=edge_type` (or index `edge_type[batch.e_id]`). Keep features out of the `Data` object and
  gather them by `n_id` on the GPU (section 7.2).
- AMP: use `torch.autocast(dtype=float16)` with GradScaler, the analogue of PL `precision=16`. bf16
  also works on the 4070 Ti and avoids the scaler. Record which one is used.

## 8. ExtScore column

- Flag `extscore: none | p`. The default `none` must reproduce the baseline bit-for-bit in architecture,
  with the numeric width staying 5.
- **Default injection: append `[z, m]` to the numeric property block**, raising `numeric_num` from 5
  to 7, before `in_linear_numeric` (Linear(7 -> 32)). `z = (logit(clip(p,1e-4,1-1e-4)) - mu)/sd`, with
  mu and sd fitted on observed TRAIN rows; z = 0 where p is missing; m = missing indicator. Nothing
  else changes. ExtScore then also propagates to neighbours through message passing.
- Missing handling: TwiBot-20 support users, and any TwiBot-22 user without a ExtScore value, get
  `[0, 1]`. Every node in the graph needs a row, so build the column for all N from `extscore/<tag>.parquet`
  and fill the rest as missing.
- Ablations:
  1. **late fusion**: concatenate `[z, m]` to `x2` before `out1`, making `out1` Linear(130 -> 64).
     Here ExtScore bypasses the graph and only affects the node itself.
  2. **separate branch**: a fifth input projection `Linear(2 -> 32)`, with `linear1` widened to
     Linear(160 -> 128).
  3. both 1 and 2 together, or the default plus late fusion.
- Requirement on the ExtScore source: train-split users must have ExtScore values produced the same way as for
  test users (out-of-fold, or from a model that never saw their labels). Otherwise the GNN learns to
  over-trust an in-sample-optimistic p. Report the observed-ExtScore coverage per split.

## 9. Pitfalls

1. **Test-label leakage in the benchmark TwiBot-22 script.** The loss runs over all sampled nodes,
   including val/test neighbours (section 5). Our default uses seed-only loss. The difference from
   76.5/42.9 can go either way; report it.
2. Readme vs code fan-out: the readme says 20 x 2 hops, the code uses 256 x 4 hops. Trust the code.
3. The TwiBot-22 benchmark has no validation or model selection (last epoch of 200). The TwiBot-20
   script selects on val acc.
4. Benchmark defaults (C = D = 2) differ from the paper's Table 2 (C = D = 8, 40 epochs, batch 256).
   The benchmark number 86.6/88.0 almost certainly comes from C = D = 2 with batch 128 and 200 epochs
   [U: inferred from the readme command plus defaults].
5. Per-head width is 128 (TransformerConv `out_channels` per head), not 128/C. The root/skip term
   `W_skip x_i` is present, although the paper omits it.
6. One gate per layer is shared across relations, and the semantic-attention beta is a global
   (graph- or batch-mean) per-relation weight, not a per-node weight.
7. Cosine schedule with T_max = 16 over 200 epochs is periodic: LR hits 0 at epoch 16, 48, 80, ...
8. With R > 2, the benchmark layer's early `return` drops relations. Do not copy it.
9. Edge direction: messages go from list owner to listed neighbour (source to target). On TwiBot-20
   many labelled nodes have no in-edges. Reverse-edge (symmetrised, 4-relation) graphs are an ablation
   only.
10. Ordering: the benchmark TwiBot-20 scripts assume train/val/test are rows 0-8277 / 8278-10642 /
    10643-11825 (`Dataset.py`). RGT's own `preprocess.py` builds `id2index` from `split.csv` while the
    BotRGCN features follow `node.json` order, so a misalignment is possible in the benchmark
    [U]. We avoid it by using the shared `idx` everywhere.
11. sklearn `f1_score` default is binary F1 for label 1 (bot). Do not report macro-F1 as comparable.
    The benchmark's AUC is computed on the raw class-1 logit; we use the softmax probability (same
    ranking only approximately; minor).
12. In fp16 AMP, the TransformerConv softmax and semantic softmax are computed in fp32 by autocast.
    Keep it that way.

## 10. Open questions / unverified items

- Which hyperparameters actually produced the benchmark numbers (defaults vs readme command lines)
  [U].
- Whether the TwiBot-22 benchmark run really completed 200 epochs with `[256]*4` sampling. That
  seems implausibly expensive [U].
- Exact TwiBot-20 relation semantics of the benchmark `edge.csv` `friend`/`follow` labels vs the raw
  `neighbor.following`/`follower`. It does not matter for the maths (two symmetric relation slots),
  only for naming.
- Real subgraph sizes on our TwiBot-22 graph (dry run required), and the actual edge counts in
  `edge-002.csv`.
- Whether to also report a version with the semantic beta computed over seed nodes only (not done in
  any source).

## 11. Acceptance check

- **TwiBot-20 (primary).** Benchmark config (C = D = 2, batch 128, 200 epochs, best val acc), 5 seeds:
  mean acc within **86.6 ± 1.0** and bot-F1 within **88.0 ± 1.0**, with std ≤ 1.0. The paper config
  should land within ±1.0 of 86.64/88.21.
- **TwiBot-22 (clean protocol).** Acc in [74.5, 78.5] and bot-F1 in [38, 50]. The protocol differs
  from the benchmark (no leakage, val selection), so the tolerance is wide. If F1 falls outside, run
  `bench_compat` (if it fits) to separate implementation bugs from protocol effects.
- Unit checks: (a) on a 50-node toy graph, our layer matches PyG `TransformerConv` + gate + semantic
  attention computed by hand to 1e-5; (b) with `extscore: none`, the parameter count matches the benchmark
  model; (c) with an all-missing ExtScore column, results agree with `extscore: none` within seed noise.
