# Simple-HGN (primary) and HGT (variant), as run in the TwiBot-22 benchmark

Spec status: written 2026-11-12 from the papers, the official Simple-HGN code (THUDM/HGB), PyG source at
the pinned version, and the TwiBot-22 benchmark `src/HGT_SimpleHGN`. Labels: **[P-paper]**,
**[P-code]** (path given), **[S]**, **[U]**.

## 1. Summary

Both methods are generic heterogeneous GNNs. The TwiBot-22 benchmark applies them to the BotRGCN user
graph: users are the only node type, and there are two edge types, follower and following.
- **Simple-HGN** is GAT extended with a learnable edge-type embedding in the attention score, a node
  residual, an edge (attention) residual, and L2 normalisation of the output embedding.
- **HGT** (Heterogeneous Graph Transformer) uses type-specific K/Q/V projections, relation-specific
  attention and message matrices, a relation prior, and a gated skip connection.

The benchmark wraps each in the same detector as RGT: BotRGCN-style four-block input encoder, then 2
GNN layers, then an MLP head. Family: heterogeneous GNN, transductive node classification. Uses the
graph and text (precomputed frozen roberta-base features). Not temporal; HGT's relative temporal
encoding is not used. Encoders: feature scheme `botrgcn` (see botrgcn.md).

Benchmark settings: **one attention head** for both methods, users-only graph, follower/following
relations only [P-code `src/HGT_SimpleHGN/readme.md` items 1-2].

## 2. Citation and code

- Lv, Q., Ding, M., Liu, Q., Chen, Y., Feng, W., He, S., Zhou, C., Jiang, J., Dong, Y., Tang, J.
  (2021). Are we really making much progress? Revisiting, benchmarking, and refining heterogeneous graph
  neural networks. KDD 2021, 1150-1160. DOI 10.1145/3447548.3467350, arXiv 2112.14936.
  Official code: https://github.com/THUDM/HGB, `NC/benchmark/methods/baseline/{GNN.py,conv.py,run_new.py}`,
  commit `ca6fd5bb0c1ca32e63b132c8bfe8f11a4a6629fe`. **No licence**: read for reference only.
- Hu, Z., Dong, Y., Wang, K., Sun, Y. (2020). Heterogeneous Graph Transformer. WWW 2020, 2704-2710.
  DOI 10.1145/3366423.3380027, arXiv 2003.01332. We use PyG's `HGTConv` (MIT), not the original code.
- Benchmark: https://github.com/LuoUndergradXJTU/TwiBot-22, `src/HGT_SimpleHGN/`, commit
  `54adf9a1dbb6dc238c932d3147a1875def5817d8`, **MIT**. Files:
  - `Twibot-20/{SimpleHGN.py,HGT.py,Dataset.py,layer.py}`, full-batch
  - `Twibot-22/{SimpleHGN_sample.py,HGT_sample.py,layer.py}`, NeighborLoader
  - `layer.py` is identical across datasets.
  - `Twibot-20/preprocess/` builds an unused `graph.pt`; the training scripts read the BotRGCN tensors.
- Pinned environment [P-code readme]: torch 1.9.1, **torch-geometric 2.0.2**, pytorch-lightning 1.4.9,
  CUDA 10.2. This pin matters for HGT (section 9.1).

## 3. Inputs

### 3.1 Node features: feature scheme `botrgcn` (see botrgcn.md)
These are the same four blocks as RGT: `cat_properties` (3), `num_properties` (5), `tweets` (768),
`des` (768). The argparse defaults are `--cat_num 3 --numeric_num 5` on both datasets [P-code all
four scripts].

Deviations from BotRGCN in loading and preprocessing (verified in code):
- TwiBot-22 scripts concatenate `x = [cat | num | tweets | des]` (width 1544) for NeighborLoader and
  slice it back apart per step [P-code `SimpleHGN_sample.py` L20, L69-72; `HGT_sample.py` L20,
  L80-83]. This is a storage detail with no maths change. TwiBot-20 scripts keep the four tensors
  separate on the GPU.
- Input projection is identical to RGT (four `Linear(.,32)` + LeakyReLU + Dropout, then
  `Linear(128,128)` + LeakyReLU + Dropout), **not** BotRGCN's.
- No extra normalisation, no self-loops and no reverse edges. Official Simple-HGN symmetrises the
  graph with distinct reverse-edge types and adds self-loops with their own type (2R+1 edge types)
  [P-code HGB `run_new.py` L80-98]. **The benchmark does neither.** Use the benchmark behaviour by
  default (section 9).
- HGT on TwiBot-22 builds a `HeteroData` with node type `user` and edge types
  `('user','follower','user') = edge_type==0` and `('user','following','user') = edge_type==1`
  [P-code `HGT_sample.py` L27-31]. On TwiBot-20, `HGT.py` maps edge_type 0 to `following` and 1 to
  `follower` (L20-21). The names are arbitrary (two symmetric slots); only consistency matters.

### 3.2 Graph
The graph is identical to rgt.md section 3.2:
- user-user edges, relations followers/follower = type 0 and following = type 1, stored
  source -> target as released
- messages flow source to target and the target aggregates
- TwiBot-22 has 3,743,634 user-user edges [P-paper TwiBot-22 Table 7]; TwiBot-20 has ≈ 455,958
  [P-paper RGT]
- on TwiBot-20 only list owners have out-edges, so many labelled nodes have in-degree 0 [U: log it]

Simple-HGN's softmax runs over **all incoming edges of a node across both relations jointly** (one
`propagate` over the full edge list with an `edge_type` attribute) [P-code `layer.py` L26, L38].

### 3.3 What we need from the shared data layer
This is the same as rgt.md section 3.3:
- `users.parquet` (`idx`, `split`, `label`)
- botrgcn feature blocks: `emb/roberta-base/des.f16.npy`, `emb/roberta-base/tweets_mean.f16.npy`, and
  the 5 numeric + 3 boolean columns
- `graph/{followers|follower,following}.csr.npz`
- `extscore/<tag>.parquet`

Nothing else.

## 4. Architecture

### 4.1 Shared detector (both methods)
[P-code `SimpleHGN_sample.py` L42-56, L78-91; `HGT_sample.py` L55-68, L86-101]
```
n = Drop(LReLU(Linear(5 [7 with ExtScore] -> 32)(num)));  b = Drop(LReLU(Linear(3 -> 32)(cat)))
t = Drop(LReLU(Linear(768 -> 32)(tweets)));          d = Drop(LReLU(Linear(768 -> 32)(des)))
x0 = Drop(LReLU(Linear(128 -> 128)(concat[n, b, t, d])))
```
- LeakyReLU slope 0.01; p = dropout.
- Init: all `torch.nn.Linear` get `kaiming_uniform_` weights and zero bias, applied recursively, so
  this also covers the `nn.Linear`s inside the SimpleHGN layer. PyG's own `Linear` modules (inside
  HGTConv) and `nn.Embedding` keep their default init. The SimpleHGN layer's own `init_weight`
  (xavier) is **never called** [P-code].
- Loss: unweighted `CrossEntropyLoss` on 2 logits. P(bot) = softmax[:,1].

### 4.2 Simple-HGN layer (benchmark version)
`SimpleHGN(in=128, out=128, num_edge_type=2, rel_dim, beta=0.05, final_layer)` [P-code `layer.py`]

Parameters:
- `W`: Linear(128->128, no bias)
- `W_r`: Linear(rel_dim->128, no bias)
- `a`: Linear(384->1, no bias)
- `W_res`: Linear(128->128, no bias)
- `rel_emb`: Embedding(2, rel_dim), default N(0,1) init

Forward, for target i and incoming edge (j -> i) of type psi:
```
e_ij     = LeakyReLU_0.2( a^T [ W x_i || W x_j || W_r rel_emb[psi] ] )
alpha_ij = softmax over all incoming edges of i (both relations jointly)
out_i    = ELU( sum_j alpha_ij W x_j  +  W_res x_i )
if final_layer: out_i = out_i / ||out_i||_2            (F.normalize, eps 1e-12)
returns (out, alpha_mixed.detach())  where alpha_mixed = (1-beta) alpha + beta pre_alpha (if given)
```
- This matches paper Eq. 7 (edge-type embedding in the attention) and Eq. 9 (node residual with
  W_res) with sigma = ELU, and Eq. 15 (L2 normalisation) [P-paper].
- **Edge residual is a no-op in the benchmark.** The mixed alpha is stored and returned, but the
  message uses the **unmixed** alpha (`out = out * alpha.view(-1,1)`, `layer.py` L39-43). Layer 2
  receives layer 1's alpha and discards the mix, so beta has no effect on outputs. The paper's Eq. 10
  (and HGB `conv.py` L125-126) apply the mixed alpha in the message, with residual attention from
  layer 2 onward. Faithful reproduction therefore means **edge residual off** (equivalently beta = 0).
  Provide `edge_residual: bench(off) | paper(on)`. In `on`, pre_alpha must come from the same edge
  ordering (true when both layers run on the same subgraph); HGB detaches it, so do the same.
- No multi-head attention (1 head; the paper uses K = 8), no feature or attention dropout inside the
  layer (HGB uses `feat_drop = attn_drop = 0.5`), no bias, no self-loops.
- Layer stack:
  - `h1, a1 = SHGN1(x0)`
  - `h2, _ = SHGN2(h1, pre_alpha=a1)` with `final_layer=True` (L2-normalised)
  - **no extra LeakyReLU or dropout between layers** (ELU is inside the layer)
  - Both layers have in = 128 (`in_channels=args.linear_channels`) and out = 128.
- Head: `y = Linear(64->2)(Drop(LReLU(Linear(128->64)(h2))))` [P-code L89-90].
- Efficient but equivalent implementation (recommended):
  - split `a = [a_1; a_2; a_3]`
  - compute node scalars `s_i = a_1 . W x_i` and `t_j = a_2 . W x_j`, and per-type scalars
    `c_psi = a_3 . W_r rel_emb[psi]` (2 values)
  - `e_ij = LReLU_0.2(s_i + t_j + c_psi)`. This is exact and avoids materialising the E x 384 concat
    and the twice-computed `W x_j` of the benchmark `message()`.
  - Aggregate with `torch_geometric.utils.softmax` + scatter, or via `GATConv`-style propagate. Verify
    against the naive form to 1e-5 on a toy graph.

### 4.3 HGT variant
Benchmark: `HGTConv(in_channels=128, out_channels=128, metadata=(['user'], [follower, following]))`,
with defaults `heads=1`, `group='sum'`, two layers, applied as `x = HGT2(HGT1(x0))` with **no
activation or dropout between the layers** [P-code `HGT_sample.py` L61-62, L94-97].

Head:
- TwiBot-22: `y = out2(LReLU(out1(x)))`, **no dropout** (L99-100)
- TwiBot-20: `y = out2(Drop(LReLU(out1(x))))` (`HGT.py` L73)

`rel_dim` is parsed but unused.

PyG 2.0.2 `HGTConv` maths (single node type, per relation r) [P-code PyG 2.0.2 `hgt_conv.py`]:
```
k = K x,  q = Q x,  v = V x                       (Linear 128->128 with bias; per node type)
for each relation r:   k_r = k A_r,  v_r = v M_r   (A_r, M_r: [heads, d, d], glorot; p_r: [heads] init 1)
    alpha_ij = softmax_{j in N_r(i)} ( (q_i . k_r,j) * p_r / sqrt(d) )   # softmax PER RELATION
    out_r,i  = sum_j alpha_ij v_r,j
agg = sum_r out_r                                  # group='sum'
o   = A_lin(GELU(agg))                             # Linear 128->128
s   = sigmoid(skip)  (skip init 1)
2.0.2 (bug):  out = s * s + (1 - s) * x            # 'alpha * alpha' -- o is DISCARDED
>= 2.0.3:     out = s * o + (1 - s) * x
```
- **At the pinned PyG 2.0.2 the HGT layers ignore the graph entirely.** The skip branch is taken
  because in = out = 128, and the output is `s^2 + (1-s) x`. The K/Q/V/A/M parameters receive no
  gradient, and the benchmark "HGT" is effectively an MLP on the input features. Verified at tags
  2.0.0-2.0.2 (bug) and fixed in 2.0.3 [P-code PyG `hgt_conv.py` L172/174 per tag].
- Modern PyG `HGTConv` (current master, commit `d1f2f8642387b6e0563d6aee227c39130746c738`)
  [P-code PyG master]:
  - concatenates all edge types into **one** propagate, so the softmax is joint over all incoming
    edges across relations, as in HGT paper Eq. 3 [P-paper]
  - uses `HeteroDictLinear` for K/Q/V and `HeteroLinear` for the relation matrices (different init
    from glorot)
  - has the fixed gated skip
  - has no `group` argument

  The HGT paper's own residual is `A-Linear(sigma(H~)) + H_prev`, ungated [P-paper Eq. 5].
- Variants to provide:
  - `hgt_impl: bench_bug` emulates 2.0.2 by computing `s^2 + (1-s)x` per layer; use it for the
    acceptance check against benchmark numbers.
  - `hgt_impl: fixed` uses modern `torch_geometric.nn.HGTConv`, heads=1; this is the actual HGT
    baseline we report.
  - Optional `hgt_impl: per_relation_softmax` replicates 2.0.3-era semantics (per-relation softmax
    then sum). Low priority.

### 4.4 Hyperparameters [P-code argparse defaults; readme command lines in brackets]

| param | SHGN TwiBot-20 `SimpleHGN.py` | SHGN TwiBot-22 `SimpleHGN_sample.py` | HGT TwiBot-20 `HGT.py` | HGT TwiBot-22 `HGT_sample.py` |
|---|---|---|---|---|
| hidden / out | 128 / 128 | 128 / 128 | 128 / 128 | 128 / 128 |
| heads | 1 | 1 | 1 | 1 |
| rel_dim | 100 | 200 | (unused) | (unused) |
| beta (edge residual) | 0.05 (no effect) | 0.05 (no effect) | - | - |
| dropout | 0.5 | 0.5 | 0.5 | 0.5 |
| lr | 1e-3 | 1e-3 | 1e-3 | 1e-3 |
| weight decay (AdamW) | 3e-5 | **0** | 3e-5 | 3e-5 |
| batch size | 128 [128] | 512 [128] | 128 [128] | 256 [128] |
| epochs | 50 [200] | 50 [200] | 50 [200] | 50 [200] |
| sampler | full-batch | `[256]*4`, homogeneous | full-batch | `[256]*2` **per relation**, hetero |
| scheduler | CosineAnnealingLR(T_max=16, eta_min=0), per epoch | same | same | same |
| precision | AMP fp16 | same | same | same |

The readme TwiBot-22 commands name the non-sampled scripts (`SimpleHGN.py`, `HGT.py`), which do not
exist for TwiBot-22, so which batch and epoch values were used there is [U]. Recommendation: TwiBot-20
uses the readme commands (batch 128, 200 epochs); TwiBot-22 uses the sample-script defaults (SHGN
batch 512, HGT batch 256, max 50 epochs).

## 5. Training

### TwiBot-20 (full-batch; both methods)
- Each step runs a full-graph forward over all 229,580 nodes, with loss on 128 shuffled training
  indices (65 steps per epoch).
- Per-epoch validation on all val nodes; keep the best val-acc checkpoint (`save_top_k=1`) and test it
  once [P-code `SimpleHGN.py` L54-72, L172-196; `HGT.py` same].
- The cosine schedule with T_max = 16 is periodic over 200 epochs (see rgt.md section 9.7).
- 5 seeds (0-4).

### TwiBot-22 (sampled)
Benchmark behaviour [P-code]:
- Training loss over **all** nodes of the sampled subgraph (`label = batch.y`), the same test-label
  leakage as RGT, because all 1M users are labelled.
- Validation loop **is** on for both methods. `val_acc` is computed over all subgraph nodes (seeds plus
  neighbours, i.e. it is mixed with train/test nodes), and checkpoint selection uses it.
- Test scores the first 200 rows per batch (`test_batch_size=200`). SimpleHGN fan-out is `[256]*4`;
  HGT fan-out is `[256]*2` per edge type.

Our default clean protocol (identical to rgt.md section 5):
- 2-hop sampling over in-edges
  - SHGN: homogeneous `[256,256]`, seed batch 512
  - HGT: hetero `{follower:[256,256], following:[256,256]}`, seed batch 256
- seed-only loss
- val metrics on seed nodes only, 200k val seeds, select on val acc, max 50 epochs, patience 10
- same optimiser and schedule settings as the table
- optional `bench_compat` (all-node loss, original fan-outs), only if the memory dry run passes

## 6. Evaluation and published numbers

Metrics: accuracy and **bot-class binary F1** (plus precision, recall, AUC). Mean ± std over 5 runs.

| method | dataset | Acc | bot-F1 | source |
|---|---|---|---|---|
| SimpleHGN | TwiBot-20 | 86.7 ± 0.2 | 88.2 ± 0.2 | benchmark Tables 2/8 [P-paper arXiv 2206.04564] |
| SimpleHGN | TwiBot-22 | 76.7 ± 0.3 | 45.4 ± 0.4 | same |
| SimpleHGN | Cresci-2015 | 96.7 ± 0.5 | 97.5 ± 0.4 | same |
| SimpleHGN | TwiBot-20 | 86.74 ± 0.22 | 88.25 ± 0.18 | `src/HGT_SimpleHGN/readme.md` (P 84.76, R 92.06) [P-code] |
| SimpleHGN | TwiBot-22 | 76.72 ± 0.27 | 45.44 ± 0.42 | same (P 72.57, R 32.90) |
| HGT | TwiBot-20 | 86.9 ± 0.2 | 88.2 ± 0.2 | benchmark Tables 2/8 [P-paper] |
| HGT | TwiBot-22 | 74.9 ± 0.1 | 39.6 ± 2.1 | same |
| HGT | Cresci-2015 | 96.0 ± 0.3 | 96.9 ± 0.2 | same |
| HGT | TwiBot-20 | 86.91 ± 0.24 | 88.19 ± 0.19 | readme (P 85.55, R 91.00) [P-code] |
| HGT | TwiBot-22 | 74.91 ± 0.13 | 39.60 ± 2.11 | readme (P 68.22, R 28.03) |

The original papers report no bot-detection results; Simple-HGN reports HGB numbers (e.g. node
classification on DBLP, ACM, IMDB, Freebase), which are not relevant here.

## 7. Scale plan for our hardware

**Mandatory memory rule (common brief, HARD MEMORY CONSTRAINT).** Run every job as
`tools/memrun [-m 10G] -- <cmd>`. Target peak host RSS <= 8 GB; 10 GB is the kill line. Never load the
full feature table, the edge file or a tweet file into RAM; use memmaps and on-disk CSR, and count
workers and pinned buffers. Assumed baseline: torch + CUDA process ≈ 2.5 GB RSS and ≈ 0.5 GB VRAM
context [U]. Measure and log real peaks.

### 7.1 Precompute
- Shared with BotRGCN and RGT (botrgcn.md): RoBERTa features ≈ 2.8 GPU-h for TwiBot-22 and ≈ 0.6 h for
  TwiBot-20.
- Graph arrays (edge_index int64 60 MB plus CSC ≈ 50 MB for TwiBot-22). Building HGT's `HeteroData`
  splits `edge_index` by type: another 60 MB.
- **Peak RSS ≈ 3 GB, VRAM 0.**

### 7.2 TwiBot-22 training
- Features: as in rgt.md section 7.2. Copy the fp16 des and tweets memmaps (2 x 1.54 GB) plus num/cat
  (32 MB) to the GPU in 64k-row chunks. VRAM 3.1 GB resident; host transient ≤ 0.2 GB. Gather rows
  per batch by `n_id` on the GPU.
- Mandatory sampler dry run: log the percentiles of `|n_id|` and `|E_sub|` over 200 batches for each
  method's loader.
- Size estimates [U]:
  - SHGN (batch 512, `[256,256]` homogeneous): ≈ 80-160k nodes and ≈ 100-200k edges per batch.
  - HGT (batch 256, 256 per relation per hop, so up to 512 per node per hop): ≈ 60-150k nodes.
  - If p99 `|n_id|` > 250k, lower the hop-2 fan-out to 64.
- **SHGN VRAM per step** (fp16, 160k nodes, 200k edges, efficient form 4.2):
  - inputs 160k x 1546 x 2 B = 0.49 GB
  - encoder ≈ 10 x 160k x 128 x 4 B = 0.82 GB
  - per layer: `W x` (N x 128) plus gathered messages E x 128 x 2 B = 51 MB plus per-edge scalars,
    ≈ 0.25 GB per layer
  - backward x 2, giving ≈ 3 GB
  - **Peak ≈ 0.5 + 3.1 + 3 ≈ 6.6 GB.** The naive benchmark `message()` (E x 896 floats) would add
    ≈ 0.4 GB per layer, which still fits.
- **HGT (fixed) VRAM per step** (150k nodes, ~200k edges, heads 1):
  - K/Q/V nodes 3 x 150k x 128 x 2 B = 115 MB
  - per relation, per-edge k_r, v_r, q gathers 3 x E_r x 128 x 2 B ≈ 77 MB, plus GELU/A_lin node
    tensors, ≈ 0.4 GB per layer
  - encoder 0.8 GB; inputs 0.46 GB; backward x 2
  - **Peak ≈ 0.5 + 3.1 + 3.5 ≈ 7 GB.**
- `hgt_impl: bench_bug` needs no message passing; the subgraph sampler is still run for parity.
- Host RSS (both): base 2.5 GB + graph 0.2 GB (HGT: + 0.06 GB of per-type copies) + 2 NeighborLoader
  workers ≈ 0.4 GB each + pinned prefetch buffers < 50 MB. **Peak ≈ 3.5-4 GB.** `num_workers` ≤ 2.
- `bench_compat` with SHGN `[256]*4`: this likely samples hundreds of thousands of nodes per batch. At
  500k nodes, inputs are 1.5 GB plus encoder ≈ 2.6 GB, x 2 for backward, ≈ 8+ GB on top of the 3.1 GB
  resident features, **so it probably does not fit**. Run it only if the dry run shows p99
  `|n_id|` < 200k.
- Time [U]:
  - SHGN: 700k / 512 = 1,368 iterations x ≈ 60 ms ≈ 1.4 min per epoch, plus val ≈ 0.5 min. With ≈ 25
    epochs under early stopping, ≈ 0.8 h per run, **5 seeds ≈ 4 GPU-h**.
  - HGT fixed: 2,735 iterations x ≈ 50 ms ≈ 2.3 min per epoch, ≈ 1.2 h per run, **5 seeds ≈ 6
    GPU-h**.
  - HGT bench_bug: ≈ 0.5 h per run (≈ 2.5 GPU-h for 5 seeds; optional on TwiBot-22).
- Evaluation: same loaders, 100k test seeds, batch 1024, no grad. VRAM ≈ 3.1 + 1-2 GB; RSS as above.
  Write P(bot) for val and test users.

### 7.3 TwiBot-20 training (full-batch)
- Features 229,580 x 1544 x 2 B = 0.71 GB on the GPU (loaded in chunks). Edges ≈ 0.46M.
- SHGN VRAM:
  - encoder ≈ 10 x 229,580 x 128 x 4 B ≈ 1.2 GB
  - layers ≈ 2 x (N x 128 x 4 tensors + E x 128) ≈ 0.6 GB
  - backward x 2
  - **≈ 0.5 + 0.7 + 3.6 ≈ 5 GB**
- HGT similar, **≈ 5-6 GB**.
- Host RSS ≈ 3 GB.
- Time: 65 steps x 200 epochs x ≈ 40-60 ms ≈ 10-13 min per run. 5 seeds x 2 methods (+ HGT
  bench_bug) ≈ **3 GPU-h**.

**Total (SHGN + HGT fixed + HGT bench_bug on TwiBot-20): ≈ 13-16 GPU-h**, excluding the shared
RoBERTa precompute.

### 7.4 Library versions
- Use modern torch 2.x and PyG ≥ 2.5 with pyg-lib. No Lightning; write a plain loop with per-epoch val
  and best-val-acc checkpointing.
- Simple-HGN: implement our own `MessagePassing` layer (or the efficient scatter form), since PyG has
  no SimpleHGN conv. `torch_geometric.utils.softmax(src, index, num_nodes=...)` keeps the same
  semantics.
- HGT: use `torch_geometric.nn.HGTConv` (modern: joint softmax across relations, fixed gated skip). The
  bug emulation is a 3-line custom module.
- NeighborLoader: modern version, seeds first and `batch.batch_size` provided. The hetero variant takes
  `num_neighbors` as a dict per edge type, or a list (applied to every edge type). Note that
  `input_nodes=('user', idx)`.
- AMP: use `torch.autocast` fp16 with GradScaler (or bf16). Scatter softmax runs in fp32 under
  autocast; check for overflow in `e_ij` under fp16.

## 8. ExtScore column

These are the same conventions as rgt.md section 8.
- `extscore: none | p`. With `none`, the numeric width stays 5 and the model is exactly the baseline.
- **Default injection: append `[z, m]` to `num_properties`**, so `numeric_num` goes from 5 to 7 before
  `in_linear_numeric` (Linear(7 -> 32)). z is the logit of p, standardised with mu/sd from observed
  TRAIN rows; z := 0 when missing; m = missing indicator. This applies to both Simple-HGN and HGT.
- Missing handling: every graph node needs a value. TwiBot-20 support users, and TwiBot-22 users
  without ExtScore, get `[0,1]`.
- Ablations:
  1. late fusion: concatenate `[z,m]` to the final GNN output before `out1` (Linear(130 -> 64)). For
     SHGN this comes **after** the L2 normalisation, so z is not squashed.
  2. separate input branch Linear(2 -> 32), with `linear1` widened to 160 -> 128.
  3. (SHGN only) feed ExtScore into the attention as an extra scalar on the source term. Low priority.
- ExtScore on train users must be out-of-fold or label-blind, as in RGT; report coverage per split.

## 9. Pitfalls

1. **HGT at PyG 2.0.2 is graph-free** (`alpha * alpha` bug). The published HGT numbers (86.9/88.2 on
   TwiBot-20, 74.9/39.6 on TwiBot-22) come from an MLP-like model if the pinned version was really
   used [U]. The TwiBot-20 HGT result beats the benchmark's own "w/o graph" MLP (82.6 acc), which casts
   some doubt on whether 2.0.2 was really the runtime. Run both `bench_bug` and `fixed`.
2. **Simple-HGN edge residual is a no-op in the benchmark**: the mixed alpha is returned but never used
   in messages. Our faithful mode equals beta = 0.
3. Benchmark Simple-HGN departs from the paper and HGB in several ways: 1 head (HGB 8); no self-loops
   or reverse edge types (HGB 2R+1 types); no feature or attention dropout inside layers; L2
   normalisation applied to the 128-d hidden followed by an MLP (HGB normalises the class logits);
   kaiming instead of xavier init; LeakyReLU slope 0.2 (HGB default 0.05). Keep the benchmark
   behaviour for comparability, and offer `shgn_variant: paper` (8 heads, self-loops + reverse types,
   working edge residual) as an ablation only.
4. TwiBot-22 training labels leak (all-node loss over sampled val/test neighbours). Validation
   accuracy is computed on mixed node sets. Our clean protocol fixes both.
5. HGT TwiBot-22 samples 256 per **edge type** per hop (hetero loader), while SHGN and RGT sample 256
   over the union. Keep each method's convention.
6. Weight decay is 0 for SHGN TwiBot-22 but 3e-5 elsewhere. Dropout after `out1` is missing in HGT
   TwiBot-22 but present in HGT TwiBot-20. These are faithful quirks; keep them.
7. The cosine schedule is periodic with T_max = 16 (LR = 0 at epochs 16, 48, ...).
8. Message direction is source to target. On TwiBot-20 many labelled nodes have no in-edges; Simple-HGN
   without self-loops then outputs `ELU(W_res x)` for them, which is still well defined. Under softmax
   over an empty set, scatter returns 0; check that no NaNs appear.
9. In modern PyG `HGTConv`, the relation-matrix init and joint softmax differ from 2.0.3-era PyG.
   Numbers from "fixed" HGT are a new baseline, not a reproduction.
10. Bot-F1 is binary (label 1), not macro.

## 10. Open questions / unverified items

- The actual PyG version used for the benchmark HGT runs (2.0.2 bug or not) [U].
- The TwiBot-22 batch size and epochs actually used (readme commands vs sample-script defaults) [U].
- Real sampled subgraph sizes on our TwiBot-22 graph; dry run required.
- When the PyG `HGTConv` rewrite to a joint softmax landed (believed to be around 2.3) [U]. It only
  matters for the optional `per_relation_softmax` variant.

## 11. Acceptance check

- **SimpleHGN, TwiBot-20** (benchmark config, 5 seeds): acc **86.7 ± 1.0**, bot-F1 **88.2 ± 1.0**.
- **SimpleHGN, TwiBot-22** (clean protocol): acc in [74.5, 78.5], bot-F1 in [38, 52]. The tolerance
  is wide because the protocol removes the leakage.
- **HGT, TwiBot-20**: one of `bench_bug` / `fixed` should reach acc **86.9 ± 1.0** and F1
  **88.2 ± 1.0**. Record which one does; that answers open question 1. Report `fixed` as the HGT
  baseline either way.
- **HGT, TwiBot-22** (clean, fixed): no strict target. Sanity bounds are acc ≥ 72 and F1 in [30, 52].
- Unit checks:
  - the efficient SHGN layer equals the naive benchmark form to 1e-5
  - `hgt_impl: bench_bug` output is invariant to edge removal (graph-free), while `fixed` is not
  - `extscore: none` has the same parameter count as the benchmark
  - an all-missing ExtScore column changes results only within seed noise
