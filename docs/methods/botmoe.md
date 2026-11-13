# BotMoE: spec

Status: spec only, nothing implemented. Evidence labels: **[P-paper]** read in the paper (arXiv 2304.06280v2),
**[P-code]** read in the official code (path:line given), **[S]** secondary source, **[U]** unverified or assumed.

## 1. Summary

BotMoE is a supervised, transductive, multi-modal graph detector. It encodes each user from three
modalities: graph (GNN over the follow graph, using BotRGCN-style node features), text (description
and mean tweet embeddings from RoBERTa) and metadata (profile numbers). Each modality goes through a
sparsely-gated, noisy top-k mixture-of-experts layer (Shazeer et al. 2017) that the paper calls
"community-aware": a gate routes each user to one of n small MLP experts. The modality outputs are
stacked as tokens, fused by one multi-head self-attention block, and the attention matrix is flattened
into a "consistency" vector. A linear classifier reads the fused tokens and the consistency vector. Loss:
cross-entropy, plus a load-balancing loss per MoE, plus L2. Graph: yes (2 relations). Text: yes, only
through precomputed RoBERTa means (no per-tweet sequence). Temporal: no. Encoder: `roberta-base`,
frozen, features precomputed. The released code differs a lot from the paper (three GNN encoders, not
one; metadata expert fed by a different, misaligned feature file; no convolution on the attention map).
Section 4 gives both and says which to build.

## 2. Citation and code

- Liu, Tan, Wang, Feng, Zheng, Luo. "BotMoE: Twitter Bot Detection with Community-Aware Mixtures of
  Modal-Specific Experts". SIGIR 2023, pp. 485-495. DOI 10.1145/3539618.3591646, arXiv 2304.06280.
- Official code: https://github.com/lyh6560new/BotMoE, commit `a423930bc52844f73f49e5bdb2f3de8a1145b84e`
  (2023-09-16, the only commit read). Licence: **MIT** ("Copyright (c) 2022 lyh6560new") [P-code `LICENSE`].
  We may read it and reuse with attribution. We write our own code anyway because the release is not
  runnable as is (absolute paths such as `/data3/whr/lyh/...`, TwiBot-20 sizes hard-coded) [P-code].
- Only TwiBot-20 code is released. `code_for_22/` and `code_for_cresci_15/` are in `.gitignore` [P-code
  `.gitignore`], so the TwiBot-22 and Cresci-15 settings are known only from paper Table 1.
- Entry point used for the paper: `main.py` trains `AllInOne1_rgcn_rgt_gcn` (`ThreeInONE.py:557-603`)
  [P-code `main.py:237`]; `test.py` loads one saved checkpoint and evaluates it on test [P-code `test.py:276-277`].
- Not in the TwiBot-22 benchmark repo (published later), so there is no benchmark re-run.

## 3. Inputs

### 3.1 What the official code reads [P-code `sample_utils.py:311-344`]

From BotRGCN `processed_data/` (TwiBot-20): `des_tensor.pt` [N,768], `tweets_tensor.pt` [N,768],
`num_properties_tensor.pt` [N,5], `cat_properties_tensor.pt` [N,3], `edge_index.pt`, `edge_type.pt`
(2 relations), `label.pt`. These are feature scheme `botrgcn` (see botrgcn.md), unchanged.

**Extra input `num_for_h`** [P-code `sample_utils.py:322-323`]: `data/Twibot-20/num_properties_tensor.pt`,
a [11826, 9] tensor for labelled users only. The file is not in the BotRGCN folder. The class that
consumes it is `DeeProBot_MoE` in `Hayawi.py`, and its width (9) and file name match the TwiBot-22
benchmark's Hayawi/DeeProBot preprocessing, `TwiBot-22/src/Hayawi/twibot-20.py:261` (commit 54adf9a),
which builds exactly 9 columns. The match is strong but not proven, so treat it as [U]:

| # | feature (Hayawi twibot-20.py) | definition |
|---|---|---|
| 1 | followers_count | z-score |
| 2 | "tweet_freq" | **listed_count / active_days** (a bug: listed, not statuses), z-scored twice |
| 3 | len(username) | z-score |
| 4 | following_count | z-score |
| 5 | statuses (tweet_count) | z-score |
| 6 | listed_count | z-score |
| 7 | screen_name_freq | `bigrams_freq(name)` = (#char bigrams)/(#unique bigrams), 0 if len<=1; z-score |
| 8 | name_entropy | sum over characters c of -p_c^2 log2 p_c (the code divides a per-position sum by the length, which gives this, not Shannon entropy); z-score |
| 9 | des_entropy | same formula on the description; z-score |

active_days uses the reference date 2020-09-05. The z-scores use pandas `.std()` (ddof=1) over the
11,826 labelled users.

**Alignment bug** [P-code `sample_utils.py:323-328`]:
`num_for_h = cat(zeros(229580-11826, 9), num_for_h)` puts the real rows at indices 217,754..229,579.
Labelled users are indices 0..11,825 (labels are concatenated as `cat([labels, fill(2)])` and
`train_idx = arange(0, 8278)`). So in the released pipeline **every labelled node's metadata-expert
input is all zeros**, and the real 9-d rows sit on support nodes, which are never classified. The
metadata expert then outputs a near-constant token for labelled users. This fits the small "w/o
metadata" ablation drop (87.76 to 87.10 acc) [P-paper Table 3].

### 3.2 What the paper says [P-paper Sec. 3.1]

- Graph encoder input: numerical + categorical metadata + description + tweets, as in BotRGCN.
- Text: RoBERTa (T5 also mentioned). t_i = mean over texts of mean over tokens (Eq. 3). Description and
  tweet features are concatenated and passed through a 2-layer MLP.
- Metadata: 5 numerical (followers, followings, statuses, active days, screen-name length) and 3
  categorical (protected, verified, default profile image), z-scored, then a 2-layer MLP. These are
  exactly the botrgcn `num` and `cat` blocks.

### 3.3 Our input decision

Default `meta_input: botrgcn8`, the paper's version: metadata-expert input = concat(botrgcn num[5],
cat[3]) = 8 columns. Reason: in the released code the metadata expert sees zeros, so the reported
numbers do not depend on the Hayawi features. Option `meta_input: hayawi9` rebuilds the 9 columns above
(the two bugs can be toggled with `hayawi_bugs: true|false`). Option `meta_input: zeros` reproduces the
released behaviour exactly.

Needed from the shared layer, per dataset (TwiBot-20, TwiBot-22, optionally Cresci-15):
- Everything feature scheme `botrgcn` needs: `users.parquet`, the botrgcn des/tweets_mean embeddings
  (`emb/roberta-base/des.f16.npy`, `emb/roberta-base/tweets_mean.f16.npy`, with the tweet count and
  truncation botrgcn.md specifies), the botrgcn num[5]/cat[3] blocks, and the two follow relations with
  the botrgcn edge direction and edge-type ids (`graph/following.csr.npz`, `graph/followers.csr.npz`
  on TwiBot-22; `following`/`follower` on TwiBot-20).
- For `hayawi9` only: `profile.parquet` columns `followers_count, following_count, statuses_count,
  listed_count, created_at, screen_name (username), name, description`, plus the collection reference
  date.
- No per-tweet sequence, no timestamps, no extra relations.
- `extscore/<tag>.parquet` when `extscore: p`.

## 4. Architecture

Notation: B = number of nodes in the sampled subgraph, S = number of seed (target) nodes, A =
`align_size`, O = `output_size` (expert output width = fusion token width). Code values for TwiBot-20
(`main.py:246-254`): A=128, O=4, `num_gnn`=3, `num_text`=2, `num_cat`=3 (class default), k=1 everywhere.

### 4.1 Released architecture `AllInOne1_rgcn_rgt_gcn` (build this as `variant: code`)

1. **feature_align** (`ThreeInONE.py:12-38`), applied to all B nodes: des 768->A, tweet 768->A, num 5->A,
   cat 3->A, meta_in (9 in code) ->A; each `Linear + LeakyReLU(0.01)`.
2. **Three graph branches** (`Botrgcn.py:491-546, 914-968, 1107-1162`). Each takes the four aligned blocks:
   - 4 x `Linear(A, A) + SELU` (des, tweet, num, cat; A = E/4 with E = 4A = 512), concat to E, then
     `Linear(E,E) + SELU`.
   - RGCN branch: **one** `RGCNConv(E,E, num_relations=2)` applied **twice, same weights**, with
     `dropout(0.4)` between, no activation (`Botrgcn.py:536-539`).
   - GCN branch: `GCNConv(E,E)`, dropout 0.4, `GCNConv(E,E)` (relations ignored).
   - RGT branch: 2 x `RGTLayer(2 relations, E, E, trans_heads=2, semantic_heads=2, dropout=0.5)`,
     dropout 0.4 between. RGTLayer = per-relation `TransformerConv(heads=2, concat=False)`, sigmoid gate
     `a = sigmoid(W[u;x])`, `h_r = tanh(u)*a + x*(1-a)`, then semantic attention over relations
     (2 heads, hidden 128, tanh; weights averaged over nodes; heads averaged) (`Botrgcn.py:1029-1104`).
     The `return` sits inside the relation loop (`Botrgcn.py:1104`); with 2 relations this is harmless.
   - Each branch ends in its own MoE: input E, output O, experts `MLP(E -> h -> O)` with LeakyReLU,
     h = E for RGCN/GCN and 64 for RGT, `num_gnn`=3 experts, k=1.
3. **Text MoE** (`moe_all.py:285-301`, `roberta.py:178-217`): **gate input = aligned description only**
   (A). Experts `MLPclassifier(A, O, hidden 128)`: `Linear(A, A/2)` on tweet and `Linear(A, A/2)` on des,
   concat (A), `Linear(A,128) + LeakyReLU`, `Dropout(0.3)`, `Linear(128, O)`. 2 experts, k=1.
4. **Metadata MoE** `DeeProBot_MoE` (`Hayawi.py:41-62`): input = aligned meta_in (A). `dropout(0.1)`, then
   MoE(A -> 32, 3 experts `MLP(A -> 128 -> 32)` with ReLU, k=1), then `Linear(32, O)`. The `cat_prop`
   argument is ignored. Gate input = expert input.
5. **MoE layer** (`moe_all.py:134-283`, David Rau's port of Shazeer et al.):
   - `clean = x @ W_gate` [*, n]; during training `noisy = clean + N(0,1) * (softplus(x @ W_noise) + 1e-2)`.
   - top-(k+1) of the logits; softmax over the top-k logits; scatter into a gates matrix [*, n].
   - Combine: `y = log( sum_j g_j * exp(E_j(x)) )`, a log-sum-exp mixture (`moe_all.py:83-107`), with
     exact zeros replaced by eps before the log. With k=1, g=1 and y = E_selected(x) exactly.
   - Balance loss = `1e-2 * (cv2(importance) + cv2(load))`, importance = gates.sum(0), load =
     sum of P(expert in top-k) (smooth, needs noise and k<n) when training, else count(gates>0); cv2 =
     var/mean^2 (unbiased var), 0 if n=1.
   - **Consequence of k=1:** the gate value is always 1, so no classification gradient reaches W_gate or
     W_noise. The routing is learned **only** from the balance loss. "Communities" are therefore a
     balanced, otherwise arbitrary partition of the gate-input space. Record this in results.
   - Init (`main.py:62-111`): xavier_uniform on every `nn.Linear` weight (PyG conv weights keep their own
     init); then every `W_gate` and `W_noise` is set to the constant 0.1. Identical gate columns give tied
     clean logits; at eval the tie is broken by `topk` order (normally expert 0) until the gates
     diverge. The three graph-MoE gates start frozen (`requires_grad=False`). See 5.
6. **Fusion** (`ThreeInONE.py:589-603`, `Transformer.py:117-164`):
   - tokens T = stack([gcn, rgcn, rgt, text, meta]) -> [S, 5, O]; `BatchNorm1d(O)` over the channel
     axis (applied as [S, O, 5]).
   - `y = Dropout(out)` is computed and **discarded** (line 596), so there is no dropout here.
   - `LModel(embed_dim=O, heads=2, dropout=0.3, norm_first=True)`: `z = Dropout(0.3)(MHA(LN(T)))`.
     No residual and no FFN (commented out). MHA attention dropout 0.3. It returns the head-averaged
     attention M [S,5,5].
   - `FixedPooling(6)` on M: zero-pad 5x5 to 6x6, MaxPool with pool size 1, so it is just padding. No
     convolution, unlike the paper.
   - h = concat(flatten(z) [5*O], flatten(M_pad) [36]) -> `BatchNorm1d(5*O+36)` -> `Linear(5*O+36, 2)`.
     The 11 padded zeros become constant BN outputs; we keep 36 to match exactly (equivalent to 25 plus
     bias).
7. Every branch output is sliced `[:11826]` (TwiBot-20 labelled count) in batch-local indices. Our
   implementation slices to the S seed nodes instead (Pitfalls).

### 4.2 Paper architecture (build as `variant: paper`, ablation only)

One graph encoder (RGCN, l layers), so 3 tokens (graph, text, metadata). Experts are 2-layer MLPs. Gate
(Eq. 4) `softmax(KeepTopK(W_g x, k))`. MoE output (Eq. 5) = linear gate-weighted sum. Fusion (Eq. 6) is a
"multi-head transformer encoder" with C=2 heads. Consistency (Eq. 7): fixed pooling of the attention
matrix, a 2-D conv with filter omega, flatten. Classifier (Eq. 8) is linear on [fused tokens, consistency
vector]. Loss (Eq. 9-10): CE + lambda1 L2 + lambda2 sum BL, BL = w_imp CV(G)^2 + w_ld CV(P)^2.
Table 1 gives experts graph/text/meta = 2/2/3 (Cresci-15, TwiBot-20) and 3/2/3 (TwiBot-22), hidden
256/256/128, k not stated. The paper does not give the conv filter size, k, or the MLP widths [U].

**Decision:** build `variant: code` with `meta_input: botrgcn8` as the reference. Report numbers from it.
`variant: paper` is optional.

### 4.3 Loss, class weighting

`CrossEntropyLoss()` (mean, unweighted) on seed nodes + the sum of the 5 MoE balance losses (each
already x1e-2) [P-code `main.py:143`]. L2 = Adam `weight_decay=1e-6` (coupled). No class weights.

## 5. Training

Official TwiBot-20 values (`main.py`) vs the paper's Table 1:

| item | code | paper | use |
|---|---|---|---|
| optimiser | Adam, wd 1e-6 | Adam, L2 1e-6 | Adam, wd 1e-6 |
| lr | **1e-5**; MultiStepLR milestones [200,600] gamma 0.8; set to **1e-6** for good once best val acc > 0.865 | 1e-4 | code |
| batch | NeighborLoader, 1024 seeds, `num_neighbors=[256,256]` | - | code |
| epochs | 400 max | 400 | 400 TwiBot-20; see 7 for TwiBot-22 |
| early stop | stop when the count of epochs with 0.85 < val acc < best exceeds 200 | - | patience 200 epochs on val acc (every non-improving epoch counts) |
| grad clip | `clip_grad_value_(1000)` | - | same |
| gate freeze | graph-MoE W_gate and W_noise frozen until best val acc > 0.865, then unfrozen | - | unfreeze at the first epoch with best val acc >= `unfreeze_acc` (0.865 TwiBot-20) **or** at epoch 200 (the code comment "freeze gate for 200 epochs"); lr -> 1e-6 at that point |
| model selection | best val acc (only counted when > 0.85) | "five runs, average" | best val acc, no thresholds |
| seeds | random seed per run; any exception (e.g. shape mismatch) silently restarts with a new seed (`main.py:263-287`) | - | fixed seeds 0..4 |

The thresholds 0.85, 0.865 and 0.871 (checkpoint save) are TwiBot-20 accuracies. On TwiBot-22
(accuracy about 79%) they would never fire, so the unreleased TwiBot-22 code must have differed [U].
TwiBot-22 per paper: lr 1e-4, hidden 128, experts 3/2/3, dropout 0.3, 400 epochs. We use A=128, O=4,
num_gnn=3, num_text=2, num_cat=3 for every dataset, `unfreeze` at epoch 20, max 100 epochs, patience 20
on TwiBot-22 [U, a budget choice; record it in results].

Splits: train for gradients, val for selection, test reported once. Support nodes (TwiBot-20) are graph
nodes only, never in the loss.

## 6. Evaluation and published numbers

Metrics: accuracy and **bot-class (binary) F1**, sklearn `f1_score(y, pred)` [P-code `sample_utils.py:463`;
P-paper Table 2 caption "binary F1"]. Mean (std) over 5 runs [P-paper].

| dataset | BotMoE acc | BotMoE bot F1 | source |
|---|---|---|---|
| Cresci-15 | 98.50 (0.0) | 98.82 (0.0) | [P-paper Table 2] |
| TwiBot-20 | 87.76 (0.2) | 89.22 (0.3) | [P-paper Table 2] |
| TwiBot-22 | 79.25 (0.0) | 56.62 (0.4) | [P-paper Table 2] |
| TwiBot-20, independent re-run | 84.22 +- 0.34 | 86.89 +- 0.34 | BotUmc, Yang et al. 2025, Table 1 [S] |
| Cresci-15, independent re-run | 95.30 +- 0.16 | 96.39 +- 0.11 | same [S] |

BotUmc copies the TwiBot-22 numbers unchanged. Ablations on TwiBot-20 [P-paper Table 3]: w/o graph
80.55/82.47; w/o text 86.95/88.10; w/o metadata 87.10/88.67; MoE replaced by MLP 86.90/88.44; fusion by
mean pooling 87.37/88.92. Several stds of 0.0 suggest repeated evaluation of one checkpoint rather than
five trainings [U].

## 7. Scale plan

**Memory rule (mandatory):** every job runs through `tools/memrun` (MemoryMax 10G, no swap, serialised by
lock). Target peak host RSS <= 8 GB. Never load a full TwiBot-22 tweet file, edge file or full feature
table with `torch.load`/`json.load`. Read memmaps in chunks and use on-disk CSR. Count DataLoader worker
RSS and pinned buffers.

BotMoE adds **no precompute** beyond feature scheme `botrgcn` (and the tiny `hayawi9` table if chosen).
The RoBERTa cost is paid once in the shared layer.

### TwiBot-20 (N = 229,580; labelled 11,826)

| step | host RSS | VRAM | time |
|---|---|---|---|
| load features | des+tweets fp16: 229,580 x 768 x 2 B x 2 = 0.71 GB, copied to GPU in 50k-row chunks (77 MB each); num/cat 1 MB; CSR about 2 MB | 0.71 GB | seconds |
| `hayawi9` build (optional) | 229,580 strings; under 0.5 GB | 0 | < 1 min CPU |
| train | python+torch+CUDA about 2.5 GB, sampler 0 workers (in-process) | features 0.71 + params about 15 M x 16 B = 0.24 GB + activations below 1 GB = **about 2 GB** | 9 batches per epoch; about 3-5 s per epoch; 400 epochs about 30 min |
| eval | same | same | seconds |

Total: about 0.5 GPU h per seed, 2.5 h for 5 seeds.

Parameter estimate: align 2 x 98k; RGCN branch 4 x 16.5k + 262k + RGCN 3 x 262k + MoE 3 x (262k+2k) =
about 1.9 M; GCN branch about 1.4 M; RGT branch about 2 x (2 relations x 4 x 512 x 1024 + gate + semantic) =
about 8.9 M. So **about 12-15 M parameters**, not the "< 1 million" the paper states (another sign the
paper and code differ).

### TwiBot-22 (N = 1,000,000, all labelled; train/val/test = 700k/200k/100k)

| step | host RSS | VRAM | time |
|---|---|---|---|
| CSR of following+followers | indptr (1M+1) x 8 B = 8 MB per relation; indices int32 4 B x E. E is unknown here (a few million user-user follow edges [U]; count it while streaming edge-002.csv in the shared layer). At E = 5M: 20 MB + 5 MB of int8 edge types | 0 | from shared layer |
| features | des + tweets_mean fp16: 1M x 768 x 2 B x 2 = **3.07 GB**; copied to GPU in 100k-row chunks (154 MB host each), never fully resident on host. num/cat fp32 32 MB | 3.07 GB | about 1 min (NVMe) |
| sampler | custom CSR sampler (2 hops, fanout [256,256], in-neighbours, i.e. sources of edges into the node, as PyG NeighborLoader does); 0 workers or 2 workers x about 0.5 GB | - | - |
| train step | about 3 GB total | subgraph size n_sub: with 1024 seeds and mean in-degree around 4-5 [U], n_sub is about 20-80k (hubs capped at 256). Activations: 3 branches x about 12 saved [n_sub, 512] fp32 tensors = 36 x 80k x 512 x 4 B = 5.9 GB worst case; bf16 autocast halves it to about 3 GB. Total **about 7 GB** worst case | 684 batches; est. 0.1-0.2 s each, so about 2-3 min per epoch incl. val |
| eval | same | same | 100k test / 1024 = 98 batches, < 30 s |

Guard: if n_sub > 150k for a batch, split the seeds in half. Expected 100 epochs x 2.5 min = about 4 h per
seed (the paper reports 4 h on a V100), so **about 20 GPU h for 5 seeds**. Fallback if over budget: 3
seeds, or max 50 epochs.

## 8. ExtScore column

Transform: shared neural transform `[z, m]` (z = standardised logit of clipped p, fitted on observed
TRAIN rows; z=0 and m=1 where missing). `extscore: none` must give exactly the baseline (no extra columns, no
extra parameters).

**Primary injection (`extscore_at: meta`)**: append [z, m] to the metadata-expert input before
`feature_align.linear_num_for_h_prop`, so its input width goes 8 -> 10 (`botrgcn8`; 9 -> 11 for
`hayawi9`). The column then also drives that MoE's **gate**, because the gate input is the expert input.
So ExtScore can route users to metadata experts ("communities by prior"). This is the natural slot: the
metadata MoE is a per-user MLP and does not propagate over the graph.

Ablations worth running (each a flag):
1. `extscore_at: meta+graph`: also append [z, m] to botrgcn `num` (5 -> 7) for the graph branches only, so
   neighbours' ExtScore values are aggregated (homophily signal). Every graph node needs a value. Support or
   unscored nodes get z=0, m=1.
2. `extscore_at: token` (late fusion, recommended second): a 6th fusion token `Linear(2, O)([z, m])`. The
   attention matrix becomes 6x6, which exactly fills `FixedPooling(6)` (no padding); the classifier
   input becomes 6*O + 36. Fusion attention can then weigh ExtScore against the other modalities, and the
   consistency matrix records agreement between ExtScore and each modality.
3. `extscore_at: head`: concatenate [z, m] to the classifier input (5*O + 36 + 2) after BN. Simplest late
   fusion.
4. `extscore_at: gate_only`: add [z, m] only to the gate inputs of all MoEs (not to expert inputs). This
   tests the "community" idea directly. Note that with k=1 the gate learns only from the balance loss
   (4.1.5), so expect little effect unless k=2 (`moe_k: 2` makes the gate differentiable through the
   mixture).

Missing handling: m=1, z=0 everywhere ExtScore is absent, including support/unlabelled TwiBot-20 nodes, which
only matter for `meta+graph` because other variants are computed only for seeds. ExtScore values for
val/test users are inputs, not labels, so there is no leakage, provided the decision model was never
tuned on our test users.

## 9. Pitfalls

1. **Metadata misalignment** (3.1): code-faithful runs feed zeros to the metadata expert for labelled
   nodes. Do not copy the `cat(zeros, num_for_h)` padding. Reproducing it is available only as
   `meta_input: zeros`.
2. **Batch-local slicing `[:11826]`** (`ThreeInONE.py:582-589`): it relies on labelled seeds coming first
   in the subgraph and on subgraphs having at most 11,826 nodes. If a subgraph is larger, `torch.stack`
   fails and the bare `except` in `main.py:272` silently restarts training with a new seed. Our code
   slices the first S (= seed count) rows everywhere and fails loudly otherwise.
3. **"best_test" in `main.py` is validation**: `main.py` logs val metrics under `best_test_*`. Test
   numbers come from `test.py` on a checkpoint saved only when val acc > 0.871, picked from up to 20x5
   runs. In `test.py` the training loop selects and early-stops on **test** accuracy
   (`test.py:223-239`), which is leakage if used. Use val only.
4. **Gate learning**: with k=1 only the balance loss trains gates (4.1.5). Graph gates are frozen with
   constant 0.1 weights at the start, so during that phase routing is pure noise in training and a
   fixed tie-break at eval. Reproduce it (it is the method) but log per-expert load at eval.
5. **Log-sum-exp combine**: the MoE output is `log(sum g exp(E))`, not a linear mixture. It matters only
   for k>1.
6. **RGCN weight sharing**: the RGCN branch applies one conv twice. Do not create two convs.
7. **No dropout before fusion** (the discarded `self.dropout(out_tensor)`). Keep it that way for
   faithfulness.
8. **lr**: the paper says 1e-4, the code 1e-5 then 1e-6. Use the code value on TwiBot-20.
9. **Edge direction and support nodes (TwiBot-20)**: with BotRGCN TwiBot-20 tensors, edge sources are
   labelled users (they own the neighbour lists), so in-neighbour sampling never brings support nodes
   in as message sources for labelled seeds [U, inferred from `TwiBot-22/src/BotRGCN/twibot_20/preprocess_1.py:52-72`;
   confirm against botrgcn.md]. Use whatever direction botrgcn.md fixes, identically for every method.
10. **Transductive**: val/test node features are visible during training (standard here). Labels are not.
11. Use fixed seeds and deterministic sampler seeds. The original draws a random seed per run.

## 10. Open questions / unverified

- The source of `num_for_h` (Hayawi features) is inferred, not shown (3.1) [U].
- The TwiBot-22 and Cresci-15 code is unreleased: TwiBot-22 hyperparameters beyond Table 1, how
  thresholds were handled, and the meaning of "hidden size" 256/128 vs `align_size`=128 in code are [U].
- Paper Table 1 says 3 metadata experts. Figure 7 says the best is 2/2/2. Code uses 3/2/3 (graph/text/meta).
- Was T5 used for the reported numbers? `get_train_data_with_t5` and `AllInOne1_rgcn_rgt_gcn_t5` exist,
  but `main.py` uses the RoBERTa-only model. We assume RoBERTa [U].
- TwiBot-22 user-user follow edge count: count it in the shared layer.

## 11. Acceptance check

5 seeds, mean, `variant: code`, `meta_input: botrgcn8`, `extscore: none`:
- **TwiBot-20 (official split): acc >= 85.5 and bot F1 >= 87.0.** Paper 87.76 / 89.22; independent re-run
  84.22 / 86.89; BotRGCN re-runs 83-86 acc. We expect to land between the re-run and the paper. If
  below 85.5, check the gate schedule and lr first.
- **TwiBot-22: acc >= 78.0 and bot F1 >= 53.0** (paper 79.25 / 56.62; benchmark BotRGCN bot F1 57.5,
  other re-runs 48-57).
- Also report macro F1 and the per-expert load histogram. Our run must be at least as good as our own
  BotRGCN re-run on the same tensors minus 1 point; otherwise suspect a bug.
