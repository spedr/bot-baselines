# BotDGT (He et al., IJCAI 2024): implementation spec

Label key: **[P-paper]** read in the paper (IJCAI proceedings PDF; its result tables match arXiv
2404.15070v2), **[P-code]** read in `Peien429/BotDGT` at commit
`095fd09313b2f942cec61fda9a1ad87a40d89454` (paths relative to the repo root), **[S]** secondary,
**[U]** unverified or assumed.

## 1. Summary

BotDGT is a "dynamic graph" bot detector built on the BotRGCN feature scheme. The final static user
graph is cut into **cumulative yearly snapshots by account creation date**: a user exists in snapshot
t if `created_at < t`, and an edge exists if both endpoints exist. There are no edge timestamps.

Per snapshot, a 2-layer graph transformer (PyG `TransformerConv`, 4 heads, relations merged) encodes
each target user from its sampled 2-hop in-neighbourhood. A causal multi-head self-attention over the
snapshot sequence then mixes the per-snapshot vectors. Its inputs are a learned snapshot-index
embedding plus two per-snapshot structural signals: the local clustering coefficient (LCC) and the
bidirectional-link ratio (BLR). A shared FFN emits logits for every snapshot. The loss is cross-entropy
summed over snapshots with weights 1.1^t, and the prediction comes from the last snapshot.

Node features are static: the same final-collection values in every snapshot. Frozen RoBERTa features
come from the botrgcn scheme; there is no text model inside BotDGT. It uses the graph (both relations,
as directed edges) and a temporal signal (only account `created_at`).

## 2. Citation and code

- Buyun He, Yingguang Yang, Qi Wu, Hao Liu, Renyu Yang, Hao Peng, Xiang Wang, Yong Liao, Pengyuan
  Zhou. "Dynamicity-aware Social Bot Detection with Dynamic Graph Transformers." IJCAI 2024,
  pp. 5844-5852. DOI 10.24963/ijcai.2024/646, arXiv 2404.15070.
- Code: https://github.com/Peien429/BotDGT, commit `095fd093…` (last push 2025-06-06).
- **No licence**: there is no LICENSE file, and the GitHub API returns `license: null` (confirms the
  survey). We may read it for reference only. Write our own code and **copy nothing verbatim**. Do not
  redistribute the shipped checkpoint `output/Twibot-20/year + 1234 + 0.87574 + 0.89071.pt` (seed
  1234: acc 87.574, F1 89.071).
- Pinned versions: torch 1.13.0, torch-geometric 2.1.0, pytorch-lightning 2.0.4 (used only for
  `seed_everything`) [P-code `requirements.txt`].
- Datasets in code: TwiBot-20 and TwiBot-22 (`data/Twibot-20/preprocess_twibot20.py`,
  `data/Twibot-22/preprocess_twibot22.py`). Both expect the TwiBot-22-benchmark BotRGCN
  `processed_data` tensors (README) plus `node.json` (TwiBot-20) or `user.json` (TwiBot-22) for
  `created_at` [P-code].

## 3. Inputs

### 3.1 Node features: feature scheme `botrgcn` (see botrgcn.md), unchanged

The model reads `des_tensor` (768), `tweets_tensor` (768), `num_properties_tensor` and
`cat_properties_tensor` (`utils/dataset.py` l.35-40). `NodeFeatureEmbeddingLayer` hard-codes
`numerical_feature_size=5`, `categorical_feature_size=3` (`models/NodeFeatureEmbeddingLayer.py` l.6;
`models/model.py` l.59 passes only `hidden_dim`).

- **Deviation note.** The benchmark's `src/BotRGCN/twibot_20/preprocess_1.py` would give one
  categorical column (a copy-paste bug, reshape to 5,301). The authors' TwiBot-20 tensor therefore had
  3 categorical columns from a fixed script **[U]**. Use the botrgcn scheme's 5 numeric and
  3 categorical columns for both datasets, and make the widths follow the scheme.
- The features are static and identical in every snapshot. In particular, active_days and the counts
  are collection-time values.

### 3.2 Graph

The structural layer uses the botrgcn `edge_index` **as directed edges, both relations merged, with
types ignored**. `edge_type` is loaded but never passed to the model (`models/model.py` l.236-237)
[P-code]. Relation codes:

- TwiBot-20: `friend` = 0 (following), `follow` = 1.
- TwiBot-22: `followers` = 0, `following` = 1.

These come from the benchmark preprocess scripts [P-code TwiBot-22 repo]. The edge direction must
match botrgcn.md exactly, because PyG message passing and `NeighborLoader` run source → target.
Shared layer: `graph/following.csr.npz` and `graph/follower(s).csr.npz`.

### 3.3 Snapshots: exactly how they are built [P-code `data/Twibot-20/preprocess_twibot20.py`]

The **timestamp** is each user's account `created_at`, for every node including TwiBot-20 support
users. It is truncated to its UTC date (`.date()`). There is no edge time and no tweet time. The paper
only says "snapshots at a fixed time interval Δt" [P-paper §3.1].

1. Missing `created_at` is filled with **2006-03-01** (TwiBot-20: 96 users per the code comment). The
   TwiBot-22 script has no fill (comment: range 1970-01-01 to 2022-03-14). Apply the same fill to both
   datasets.
2. Candidate dates are the first day of every month. TwiBot-20 starts at 2008-01-01 and stops at the
   first month where every user exists. TwiBot-22 runs over the 2008-2022 range with the same stop
   rule.
3. Snapshot at date D: nodes {v : date(v) < D}. Edges: botrgcn edges whose source and target are both
   in the node set (an induced subgraph).
4. Interval selection (`utils/dataset.py::load_graphs` l.6-20): sort the monthly files and take every
   `interval`-th file counting **back from the last** (`year` = 12, the default and the setting behind
   the paper's results).
   - TwiBot-20: last month 2020-10-01, 154 monthly files, giving **13 snapshots: 2008-10-01,
     2009-10-01, …, 2020-10-01**.
   - TwiBot-22: last month 2022-04-01, 172 files, giving **15 snapshots: 2008-04-01 … 2022-04-01**.
   - These counts are derived from the code's comments on min/max dates. Recompute them from our data
     by the same rule. Snapshots before 2008 do not exist, and users created earlier are present from
     the first snapshot.
5. Per-snapshot node signals (zero for nodes absent from the snapshot):
   - **LCC.** `nx.clustering` on an undirected simple `nx.Graph` of all snapshot edges (both
     relations), whose node set is the existing users. LCC(v) = 2T(v) / (d(d−1)), where d is the number
     of distinct undirected neighbours (self-loops excluded) and LCC = 0 when d < 2. The paper's eq.5
     says k = in-degree + out-degree; the code uses the distinct undirected degree. **Use the code.**
   - **BLR.** A `DiGraph` of **following-relation edges only** (TwiBot-20 type 0, TwiBot-22 type 1;
     duplicates collapse). BLR(v) = |{u : v→u and u→v}| / outdeg(v), set to 0 when outdeg is 0.
   - `exist_nodes` is a 0/1 vector.
6. Window: `window_size = -1` keeps all snapshots.

**Can we rebuild this from our data? Yes.**

- TwiBot-20: every user, including support users, has `profile.created_at`
  (`"Tue Jan 08 23:05:31 +0000 2019 "`, with a trailing space; parse `%a %b %d %H:%M:%S %z %Y` after
  strip). The graph comes from the `neighbor` field.
- TwiBot-22: `user.json` `created_at` reaches us through `profile.parquet`, and edges through the
  following/followers CSR.

Nothing else from `node.json`/`user.json` is used. **From the shared layer we need:**
`profile.parquet.created_at` (UTC, nullable), `users.parquet` (idx, split, label), the two CSR
relations, the botrgcn embedding and property tensors, and `extscore/<tag>.parquet`.

Our derived artifacts, in `artifacts/methods/botdgt/<dataset>/`:

- `snap_dates.json`
- `edge_birth.npy`: int8 per botrgcn edge, the first snapshot index where both endpoints exist, i.e.
  index of max(date_src, date_dst). The snapshot-t graph is the edges with birth ≤ t.
- `node_birth.npy`: int8 per node.
- `lcc.f32.npy` and `blr.f32.npy`: [T, N].

No per-snapshot edge copies are needed.

## 4. Architecture

Shown for hidden h. TwiBot-20 uses h = 128, TwiBot-22 uses h = 64. Heads: 4 structural, 4 temporal.

**NodeFeatureEmbeddingLayer** [P-code `models/NodeFeatureEmbeddingLayer.py`]:

- num 5→h/4, cat 3→h/4, des 768→h/4, tweet 768→h/4, each Linear followed by PReLU.
- Concatenate in the order (**num, cat, des, tweet**) → Linear(h,h) → PReLU.
- **All five PReLUs are one shared `nn.PReLU()` with a single parameter.**
- Init: kaiming_normal weights, zero bias. The `dropout` attribute is unused.
- Features are static, so compute this once per unique node in the batch and gather it for each
  snapshot. That is mathematically identical, with no dropout involved, and saves memory.

**Structural layer** (one module shared by all snapshots) [P-code `models/GraphStructuralLayer.py`]:

- s = PReLU(TC2(PReLU(TC1(x))) + x). Both PReLU uses are one shared single-parameter module.
- TC = PyG `TransformerConv(h, h/4, heads=4, concat=True, dropout=structural_drop=0)`, with defaults
  `beta=False`, `root_weight=True` (a skip linear is added) and `bias=True`.
- Attention softmax over incoming edges of (q_i·k_j)/sqrt(h/4). Messages are value(x_j).
- Init: PyG's default for the TransformerConv linears (kaiming_uniform a=sqrt(5), uniform bias). **Correction
  (review round 1):** the official `init_weights` loop applies kaiming_normal only to `torch.nn.Linear`
  instances, and TransformerConv's `lin_*` are `torch_geometric.nn.dense.linear.Linear`, a plain
  `torch.nn.Module` in PyG 2.1.0 (verified in the 2.1.0 source), so the loop never reaches them.
- Run it on each snapshot's sampled subgraph and keep the first `batch_size` rows (the seeds).
- Non-existent seeds have no edges, so their output is the skip and residual path only. They are
  still computed.

**Position signals:**

- p_AT = `nn.Embedding(T, h)` indexed by snapshot position 0…T−1. It is trainable.
- p_LCC = Linear(1,h)(LCC), with kaiming_normal weight and default bias init.
- p_BLR = Linear(1,h)(BLR), all default init.
- **p_LCC and p_BLR are not in any optimiser parameter group** (`main.py` l.28-33), so they stay at
  their random initialisation and get no weight decay. The paper implies they are trained. **Faithful
  mode: freeze them**; flag `train_pos_signals` for the ablation.

**Temporal layer** (`temporal_module_type='attention'`) [P-code `models/GraphTemporalLayer.py`
l.43-91]:

- Input S of shape [B,T,h] (structural outputs).
- X = LayerNorm(S + p_AT + p_LCC + p_BLR).
- Q = X·Wq, K = X·Wk, V = X·Wv, with W of shape [h,h], no bias, kaiming_uniform init.
- Split into 4 heads of size h/4. Scores = QKᵀ/sqrt(h/4), with a **causal mask**: position a attends
  to b ≤ a, and masked entries are set to −1e9.
- softmax → Dropout(temporal_drop = 0.5) → ·V → concatenate heads.
- O = **LayerNorm(O + S)**. The residual is the raw S, not X. **The same LayerNorm module (shared
  parameters) is used both times.**
- Logits[B,T,2] = Linear(h,2)(PReLU(Linear(h,h)(O))). Both FFN linears get kaiming_normal weights and zero
  bias (`GraphTemporalLayer.init_weights` loops over its `torch.nn.Linear` modules) [P-code; added in
  review round 1].
- `exist_nodes` is passed in but not used: there is no masking of time steps before a node exists.
- The `gru`/`lstm` options exist (residual RNN followed by the same FFN) but are not used for the
  paper's numbers.

**Loss** [P-code `utils/loss.py`]: L = Σ_{t=0}^{T−1} 1.1^t · CE_mean({seeds with exist_t = 1}).
Snapshots with no existing seed are skipped. Classes are unweighted, and only train seeds contribute.

- The paper's eq.11 is an unweighted BCE sum over snapshots. **Use the code** (`coefficient = 1.1`).
- The last-snapshot weight is 1.1^12 ≈ 3.14 for TwiBot-20 and 1.1^14 ≈ 3.80 for TwiBot-22.

**Prediction:** argmax of the logits at the **last snapshot** (all users exist there).

## 5. Training

| Item | TwiBot-20 | TwiBot-22 | Source |
|---|---|---|---|
| batch (seed nodes) | 64 | 256 | README commands |
| hidden h | 128 | 64 | README |
| weight decay (AdamW, all groups) | 1e-2 | 5e-2 | README |
| lr: feature embedding + structural | 1e-4 | 5e-4 | README |
| lr: temporal layer | 1e-5 | 5e-5 | README |
| epochs | 20 | 20 | `config.py` |
| schedule | CosineAnnealingLR(T_max=20, eta_min=0), stepped per epoch | same | `main.py` l.34 |
| heads (struct/temporal) | 4/4 | 4/4 | `config.py` |
| dropout (struct attention / temporal attention) | 0 / 0.5 | same | `config.py` |
| interval / window / coefficient | year / −1 / 1.1 | same | `config.py` |
| seed | 1234 (`seed_everything`) | same | `config.py` |
| early stop | **off** (README commands omit `--early_stop`) | off | README, `main.py` l.187-190 |

The paper states no hyperparameters [P-paper], so use the code and README.

- **Model selection: the last-epoch model is tested** (`test_last_model`). Validation is computed
  every epoch but is unused without `--early_stop`. If early stop is enabled: patience 10, best by
  val accuracy, then F1.
- **Batches and neighbour sampling** (`utils/dataset.py` l.95-166):
  - One PyG `NeighborLoader` per snapshot, each with `input_nodes` = split indices,
    `num_neighbors = [2560, 2560]` for train and `[-1, -1]` (all) for val and test, and
    `torch.Generator().manual_seed(seed)`. Train is shuffled; val and test are not.
  - Every loader yields the same seed order, so batch b holds the same seeds in every snapshot.
  - Sampling uses incoming edges, without replacement.
  - **All batches are sampled once and cached.** Each epoch replays the identical batches in the
    identical order. Reproduce this: sample once with seed s, then reuse. Per-epoch resampling is an
    ablation.
- Splits: official train/val/test. TwiBot-20 support nodes are graph-only, with label padding 3,
  which is never used. TwiBot-22 uses 700k/200k/100k.

## 6. Evaluation and published numbers

sklearn on the last snapshot over the test seeds: accuracy and **binary (bot-class) F1**, precision and
recall (`utils/metrics.py`). The paper reports the mean ± std of 5 runs [P-paper Table 1].

| Dataset | Acc | F1 (bot) | Precision | Recall |
|---|---|---|---|---|
| TwiBot-20 | 87.25 ± 0.51 | 88.87 ± 0.55 | 84.44 ± 0.39 | 94.24 ± 0.37 |
| TwiBot-22 | 79.33 ± 0.22 | 58.15 ± 0.74 | 72.42 ± 0.70 | 48.46 ± 0.92 |

- The paper's own BotRGCN re-runs give TwiBot-20 85.83 / 87.44 and TwiBot-22 78.95 / 56.47.
- Ablation "w/o TM" (static): TwiBot-20 85.72 / 86.99, TwiBot-22 78.64 / 55.23.
- Temporal module bolted onto BotRGCN: TwiBot-20 86.33 / 87.92, TwiBot-22 79.17 / 57.70 (Tables 2-3).
- The interval sweep in Fig. 3 peaks at 12 months.
- The TwiBot-22 benchmark has no BotDGT entry (BotDGT postdates it).

## 7. Scale plan (hard memory rule: every job runs through `tools/memrun`, MemoryMax 10G, no swap; target ≤ 8 GB host RSS)

The text-embedding precompute is inherited from botrgcn.md, so BotDGT adds no encoder cost. Never load
`edge-002.csv`, `user.json` or the tweet files: use only the shared-layer parquet, CSR and memmaps.

### Snapshot preprocessing (CPU, numba)

| Step | TwiBot-20 (N = 229,580, E ≈ 228k) | TwiBot-22 (N = 1M, E ≈ 3.74M [U]) |
|---|---|---|
| Load `created_at` → int32 days; load the CSRs | under 50 MB | days 4 MB; 2 relations: indptr 2×8 MB, indices 3.74M×4 B = 15 MB; under 0.1 GB |
| node_birth, edge_birth | trivial | 1M + 3.74M int8 = 5 MB |
| Per snapshot (T = 13/15): undirected dedup CSR of existing edges → LCC by degree-ordered triangle counting | under 50 MB | sym. edges ≤ 7.5M × 4 B = 30 MB, plus sort buffers (int64 keys 7.5M × 8 B = 60 MB) ≈ 0.2 GB; O(m^1.5) worst case, about seconds to minutes per snapshot |
| BLR: sort following edges as (min, max, dir) keys and count reciprocal pairs | small | 2.6M [U] × 8 B keys ≈ 21 MB |
| Output lcc/blr [T,N] float32 | 2 × 13 × 229,580 × 4 B = 24 MB | 2 × 15 × 1M × 4 B = 120 MB |
| **Host peak** | under 0.5 GB | about 1 GB; wall-clock under 15 min |

The official pipeline instead builds 154/172 monthly snapshots with networkx, at about 1-3 GB of
Python graph per TwiBot-22 snapshot and hours of runtime. Build only the T selected snapshots. The
monthly snapshots matter only for interval sweeps.

### Training and evaluation

**VRAM, TwiBot-22:**

- Feature tables on the GPU in fp16: 2 × 1M × 768 × 2 B = 3.07 GB. num/cat fp32: 32 MB.
  lcc/blr: 120 MB. CSC with edge_birth: about 30 MB.
- Model: under 1M parameters.
- Per batch:
  - Embed the U unique nodes once. The des/tweet Linear saves its input (1,536 × 4 B = 6 KB per node).
    With U ≈ 30k [U], that is 0.18 GB.
  - TransformerConv per snapshot saves q, k, v, attention and messages, about E_t × 4 heads × ~16 B
    plus U_t × h × 4 B × ~8.
  - With Σ_t E_t ≈ 15 × 200k [U], that is about 0.2-1 GB.
- Peak about 5-6 GB.
- **Guard:** if a batch's union exceeds 300k nodes (hubs; 2560-cap 2-hop), split its seeds into
  micro-batches with gradient accumulation.
- **Eval with `[-1, -1]` is batch-size invariant.** A seed's output depends only on its full 2-hop
  in-neighbourhood, so eval can use micro-batches of any size and give identical results.

**Host RSS:** torch with the CUDA context about 2.5 GB. Memmaps are streamed to the GPU in 50k-row
chunks, which keeps resident page cache small. Use `num_workers = 0`: sampling runs in the main
process with our own CSC sampler, or PyG `NeighborLoader` without workers. The sample cache lives on
disk as int32 memmaps: TwiBot-22 train ≈ 2,735 batches × 15 snapshots × ~200k edges × 8 B ≈ 66 GB
[U]. That is too much, so store it as a seed plus a per-batch RNG state and **resample
deterministically** each epoch. This is identical to caching, as long as the sampler is deterministic
for a given RNG state. Peak about 3-4 GB.

**TwiBot-20:** features fp16 2 × 229,580 × 768 × 2 B = 0.7 GB VRAM. Subgraphs are tiny, since in
TwiBot-20 in-edges reach labelled users only from labelled users [U]. VRAM peak under 2 GB, host under
3 GB.

**Wall-clock:**

- TwiBot-20: 130 train batches × 13 snapshots per epoch. Under 10 s per epoch, so under 5 min per
  seed, about 0.3 GPU-h for 5 seeds.
- TwiBot-22: 2,735 batches × 15 snapshots, about 0.1-0.3 s per batch [U], so 5-15 min per epoch and
  20 epochs ≈ 2-5 GPU-h per seed.
  - Skip per-epoch validation in faithful mode: it is unused without early stop.
  - Test: 100k seeds with full neighbourhoods takes about 10-20 min.
  - Total for 5 seeds about 12-25 GPU-h, or 3 seeds about 8-15 GPU-h. Snapshot preprocessing about
    0.3 CPU-h.

## 8. ExtScore column

- Flag `extscore: none | p`. `none` gives exactly §4.
- **Primary:** append [z, m] (shared neural transform, μ and σ fitted on observed TRAIN rows) to the
  numeric properties, raising `numerical_feature_size` from 5 to 7 in `NodeFeatureEmbeddingLayer`. It
  is static across snapshots, like every other feature.
- Support nodes (TwiBot-20) and unlabelled or missing rows get z = 0 and m = 1.
- Caveat: the value is a collection-time judgement that also enters "historical" snapshots. The same
  is true of every BotDGT feature, so this is no new leakage.
- **Ablation A (late fusion):** concatenate [z, m] to the temporal-layer output O before
  `feedforward_linear_1` (h → h+2), at every time step.
- **Ablation B (temporal token):** p_EXTSCORE = Linear(2,h)([z, m]), added to the temporal input X sum with
  the same value at every t.

## 9. Pitfalls

1. "Dynamic" snapshots are a creation-date filtration of the final graph. There are no edge or tweet
   timestamps. Do not use tweet times or edge times: that would be a different method.
2. Features are static across snapshots. The structural layer is shared over time. Relations are
   merged, so `edge_type` is ignored.
3. Interval selection counts back from the **last** monthly date (so 2008-10 … 2020-10 for TwiBot-20),
   not forward from 2008-01. `created_at` is compared as a UTC date against the first day of the
   month, with a strict `<`.
4. LCC uses the undirected distinct degree; the paper's formula is wrong for the code. BLR uses
   following edges only, and the relation code differs per dataset (0 on TwiBot-20, 1 on TwiBot-22).
5. p_LCC and p_BLR are **frozen at random init** (they are missing from the optimiser).
6. Shared single-parameter PReLUs. One LayerNorm reused twice. The temporal residual adds the raw
   structural output.
7. Causal mask, with no masking of pre-creation steps. Pre-creation steps are excluded from the loss
   only.
8. Loss weights 1.1^t. Separate learning rates for the temporal layer (10× smaller).
9. Train batches are sampled once with a fixed generator and replayed every epoch. The train
   neighbour cap is 2560; eval uses the full 2-hop neighbourhood.
10. The final model is the **last epoch** (no early stopping). The paper does not say this.
11. Labels for TwiBot-20 support nodes are padded with 3. Never let them enter the loss: only seeds
    are labelled.
12. The NaN prints in `model.py` hint at instability. Use fp32 for the model; only the input tables
    are fp16.

## 10. Open questions / unverified

- The authors' TwiBot-20 tensor provenance (3 categorical columns; benchmark script bug).
- The exact min/max `created_at` in our copies, and therefore T. Recompute and log.
- The 5-run seeds: only 1234 is known.
- The TwiBot-22 2-hop sample sizes, hub behaviour and runtime (pilot needed).
- The edge direction of our CSR versus botrgcn `edge_index`, which defines in-neighbour sampling.
  Inherit it from botrgcn.md.

## 11. Acceptance check

- **TwiBot-20, official split, `extscore: none`, 5 seeds, faithful mode:** mean accuracy 87.25 ± 1.0 and
  bot-F1 88.87 ± 1.0. Recall should come out high (about 94) with precision about 84.
- **TwiBot-22, ≥ 3 seeds:** accuracy 79.33 ± 1.0 and bot-F1 58.15 ± 2.0.
- **Sanity checks:**
  - "w/o TM" (T = 1, last snapshot only) should lose about 1 point of accuracy on TwiBot-20.
  - Our BotDGT should beat or match our BotRGCN on bot-F1 on both datasets.
- **Unit tests:**
  - LCC and BLR match networkx on random snapshots of 5k nodes.
  - The snapshot dates and the per-snapshot node and edge counts are logged and monotone.
