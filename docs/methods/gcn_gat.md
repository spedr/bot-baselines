# GCN and GAT on the BotRGCN features (TwiBot-22 benchmark baselines)

Status: spec, written 2026-11-12. Labels: **[P-paper]**, **[P-code]** (path), **[S]**, **[U]**.

## 1. Summary

Two homogeneous-graph GNN baselines that share BotRGCN's input pipeline and head and only swap the
message-passing layers: GCN (Kipf and Welling 2017) and GAT (Velickovic et al. 2018). Inputs are the
four `botrgcn` feature blocks (botrgcn.md section 3.1); both follow relations are merged into one
directed edge set (edge types ignored). Unlike BotRGCN they are trained with PyG `NeighborLoader`
mini-batches (fanout 256, 4 hops, batch 128) and early stopping on validation accuracy.
Family: GNN. Graph: yes (user-user following + follower, homogeneous). Text: frozen RoBERTa features.
Temporal: no.

## 2. Citation and code

- Thomas N. Kipf, Max Welling. "Semi-Supervised Classification with Graph Convolutional Networks".
  ICLR 2017, arXiv 1609.02907.
- Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, Yoshua Bengio.
  "Graph Attention Networks". ICLR 2018, arXiv 1710.10903.
- As run: Feng et al., "TwiBot-22", NeurIPS 2022 D&B, arXiv 2206.04564, App. B.1: "The initial user
  features are identical with BotRGCN" [P-paper].
- Code: `LuoUndergradXJTU/TwiBot-22`, commit `54adf9a1dbb6dc238c932d3147a1875def5817d8`, MIT,
  `src/GCN_GAT/{model.py,train.py,dataset.py,utils.py,readme.md}` (`remove_graph.py` is byte-identical
  to `train.py`; `transfer.py`/`annotation_test.py` are other experiments).
- Licence MIT: reuse with attribution; we write our own code.
- Not to be confused with `BotGCN`/`BotGAT` in `BunsenFeng/BotRGCN/model.py` (BotRGCN paper Fig. 3,
  full-batch on TwiBot-20); those are a different, unpublished-number variant.

## 3. Inputs

From `dataset.py::get_train_data` [P-code l.32-62]: the BotRGCN `processed_data` of the dataset:
`des_tensor`, `tweets_tensor`, `num_properties_tensor` (5), `cat_properties_tensor` (3; Cresci 1),
`edge_index`, `edge_type` (loaded, unused by GCN/GAT), `label`, `train/val/test_idx`.

- Feature definitions: exactly the `botrgcn` scheme, default variant per dataset (botrgcn.md 3.1).
- TwiBot-20: 229,580 nodes; labels of the 217,754 support nodes are filled with 2 (never seeds);
  train = 0..8277, val = 8278..10642, test = 10643..11825 [P-code l.43-47].
- TwiBot-22: 1,000,000 nodes, all labelled; split files.
- Graph: `edge_index` of the `botrgcn` scheme, both relation types concatenated, **directed as
  stored** (TwiBot-22: following edges follower -> followee and followers edges followee -> follower;
  TwiBot-20: from each user's neighbour lists, see botrgcn.md 3.1.6). Edge types dropped. No reverse
  edges or dedup are added by the code; the conv layers add self loops.
- From the shared layer: same as BotRGCN (`users.parquet`, the `botrgcn` feature files,
  `graph/following.csr.npz` + `graph/followers.csr.npz` / `graph/follower.csr.npz`). For sampling we
  additionally need the merged graph in **CSC** (incoming neighbours per node): colptr int64 [N+1],
  row int32/int64 [E]. Build it on disk once from the two CSRs.

## 4. Architecture

`hidden_dim` H = 128 (default arg) [P-code `train.py` l.16]. Shared front and head
[P-code `model.py` l.8-55 (GAT), l.58-105 (GCN)]:

| step | op | shape |
|---|---|---|
| 1 | d = LeakyReLU(Linear(768, H/4)(des)) | n x 32 |
| 2 | t = LeakyReLU(Linear(768, H/4)(tweet)) | n x 32 |
| 3 | n_ = LeakyReLU(Linear(5, H/4)(num_prop)) | n x 32 |
| 4 | c = LeakyReLU(Linear(3, H/4)(cat_prop)) | n x 32 |
| 5 | x = Dropout(p)(cat(d, t, n_, c)) | n x 128 |
| 6 | x = LeakyReLU(Linear(H, H)(x)) | n x 128 |
| 7 | x = Conv1(x, edge_index) | n x 128 |
| 8 | x = Dropout(p)(x) | |
| 9 | x = Conv2(x, edge_index) | n x 128 |
| 10 | x = LeakyReLU(Linear(H, H)(x)) | n x 128 |
| 11 | logits = Linear(H, 2)(x) | n x 2 |

- GCN: `Conv1 = GCNConv(128, 128)`, `Conv2 = GCNConv(128, 128)`, PyG defaults: add self loops,
  symmetric normalisation `D^-1/2 (A + I) D^-1/2` with degrees = in-degree (+1) **computed on the
  sampled subgraph**, bias, no caching.
- GAT: `Conv1 = GATConv(128, 32, heads=4)` (concat -> 128), `Conv2 = GATConv(128, 128, heads=1)`;
  PyG defaults: LeakyReLU(0.2) attention, attention dropout 0, self loops added, bias.
- **No nonlinearity between or after the two conv layers** (no ELU/ReLU), unlike the original
  GCN/GAT papers. Dropout p = 0.3 at steps 5 and 8 (same module).
- LeakyReLU slope 0.01; init: PyTorch/PyG defaults (no kaiming re-init here).
- Loss: unweighted cross-entropy on the batch's seed nodes (`out[:batch_size]`).
- `model.py` also has `BotRGCN` (two **untied** RGCNConv layers + input dropout) usable via
  `--mode RGCN`; it is not the published BotRGCN row.

## 5. Training

[P-code `train.py` l.13-22, l.63-94; `utils.py`]

| setting | value |
|---|---|
| optimiser | Adam, lr 1e-4, weight decay 1e-5 |
| batch size | 128 seed nodes |
| sampler | `NeighborLoader(num_neighbors=[256]*4)`, sampling incoming edges, without replacement, directed subgraph (only sampled edges); train shuffled; val and test also sampled (not shuffled) |
| max epochs | 1000 |
| early stopping | stop when validation has not improved for `no_up` = 50 consecutive epochs |
| "improved" | lexicographic on val metrics: acc, then F1, MCC, PR-AUC, ROC-AUC, precision, recall (`is_better`) |
| model used for test | see below (bug) |
| seeds | readme: 100, 200, 300, 400, 500; the code sets no seed |
| dims | hidden 128, dropout 0.3, for all datasets (no per-dataset overrides in code) |

**Model-selection bug** [P-code `train.py` l.75-84]: `best_state_dict = model.state_dict()` stores
references to the live parameter tensors (no copy), so later optimiser steps overwrite it;
`model.load_state_dict(best_state_dict)` therefore loads the **final** weights, i.e. the weights 50
epochs after the best validation epoch (or after epoch 1000). Faithful reproduction: test the final
weights (stop rule unchanged). Flag `--fix-best-state` (deepcopy) gives the intended behaviour;
report it separately.

Sampling depth: the model has 2 message-passing layers, but sampling is 4 hops. For GAT (softmax
over incoming edges of each target) the seeds' outputs depend only on the first 2 sampled hops, so
`[256, 256]` gives the same output distribution as `[256]*4` [U, derived]. For GCN the normalisation
of an edge j -> i uses deg(j), and deg of a 2-hop node is its number of sampled in-edges from hop 3,
so `[256, 256, 256]` is needed for equivalence; hop 4 never matters [U, derived]. We may use these
reduced depths (exact in distribution, much cheaper); record it in the run metadata. Each node's
neighbours are sampled once (when it first enters the frontier) in PyG's sampler; our sampler must do
the same.

## 6. Evaluation and published numbers

Metrics computed by `calc_metrics`: acc, binary F1 / precision / recall with pos_label 1 = **bot**
(labels come from BotRGCN `label.pt`, bot = 1), MCC, ROC-AUC and PR-AUC from softmax P(bot).
Readme means +- std over 5 runs [P-code readme]; paper Tables 2 and 8 print the same means but with
"(+-0.0)" std [P-paper], which contradicts the readme stds.

| Model | Dataset | Acc | Bot F1 | Prec | Rec |
|---|---|---|---|---|---|
| GCN | Cresci-2015 | 96.37 +- 0.57 | 97.17 +- 0.43 | 95.59 | 98.81 |
| GCN | TwiBot-20 | 77.53 +- 1.73 | 80.86 +- 0.68 | 75.23 | 87.62 |
| GCN | TwiBot-22 | 78.39 +- 0.09 | 54.96 +- 0.91 | 71.19 | 44.80 |
| GAT | Cresci-2015 | 96.89 +- 0.21 | 97.58 +- 0.15 | 96.10 | 99.11 |
| GAT | TwiBot-20 | 83.27 +- 0.56 | 85.25 +- 0.38 | 81.39 | 89.53 |
| GAT | TwiBot-22 | 79.48 +- 0.09 | 55.86 +- 1.01 | 76.23 | 44.12 |

Consistency check (TwiBot-20 GCN, bot positive, 640 bots / 543 humans): TP = 0.8762 x 640 = 561,
FP = 561 / 0.7523 - 561 = 185, TN = 358, acc = 919 / 1,183 = 0.777 (published 0.7753): bot class
confirmed.

## 7. Scale plan for our hardware

**Hard requirement:** every step via `tools/memrun` (MemoryMax 10G, no swap); peak host RSS target
<= 8 GB; features memmapped / GPU-resident in fp16, graph as on-disk CSC loaded as two small arrays;
never load a full feature table into host RAM. DataLoader workers count toward RSS.

Precompute: none beyond the shared `botrgcn` tensors (botrgcn.md 7.1) and the merged CSC.

### 7.1 TwiBot-22 (N = 1e6, E = 3.74e6 directed, + self loops)

- Graph in host RAM: colptr 1e6 x 8 B = 8 MB; row 3.74e6 x 8 B = 30 MB. Building it from the two
  CSRs: < 0.3 GB.
- Features GPU-resident fp16: des + tweets 3.07 GB, num/cat 0.03 GB, loaded in 50k-row chunks
  through a pinned 77 MB buffer (host RSS peak ~2 GB: torch + CUDA 1.5 GB + sampler arrays).
- Sampling on CPU (pyg-lib / torch-sparse `neighbor_sample`) in the main process, or 2 workers
  (+~0.7 GB RSS each, fork copy-on-write). Host RSS peak **~3-4 GB** with 2 workers.
- Per batch: subgraph node ids -> GPU gather of features. Subgraph size: seeds 128; in-degree is
  heavy-tailed (max degree in the full graph 270,344 [P-paper]); followee hubs pull 256 of their
  followers each, so a 2-hop subgraph is typically 10^3-10^5 nodes and a 4-hop one can approach
  10^5-10^6 [U, estimate]. At 1e5 nodes: gathered inputs 1e5 x 1536 x 2 B = 0.31 GB, H = 128
  activations 1e5 x 128 x 4 B = 51 MB each, GAT attention on ~1e5-1e6 edges x 4 heads: < 1 GB.
  **VRAM peak ~5-6 GB** with 2-3 hop sampling; with the literal 4-hop sampling a batch can approach
  the full graph (~12 GB for GAT training): use the reduced depths of section 5.
- Time: 700,000 / 128 = 5,469 train batches + 1,563 val batches per epoch. At ~15-30 ms per batch
  (2-3 hops) -> 2-3.5 min per epoch. Epoch count = best epoch + 50; assume 80-200 epochs ->
  **3-12 h per run** [U]. GAT (2 hops) at the low end, GCN (3 hops) higher. 5 seeds x 2 models:
  ~30-100 GPU-h. Plan: 1 seed each first, measure, then 3-5 seeds. Allowed deviations (record them):
  (a) full-neighbourhood inference for val/test in one full-graph pass (no grad: inputs 3.1 GB +
  conv messages (3.74e6 + 1e6) x 128 x 4 B = 2.4 GB per layer transient -> ~7 GB VRAM, host RSS
  ~2 GB), which differs from sampled eval only at nodes with in-degree > 256; (b) evaluate val every
  k epochs with patience 50/k.
- Test eval: 782 batches, minutes.

### 7.2 TwiBot-20 (N = 229,580, E ~ 228k)

Edges leave only labelled users and 1,185 support users, so subgraphs are small (~10^3-10^4 nodes).
Inputs fp32 1.41 GB on GPU; VRAM < 3 GB; host RSS < 2.5 GB. 65 train batches + 19 val batches per
epoch, ~1-2 s per epoch; up to 1,000 epochs -> < 30 min per run; 5 seeds x 2 models ~2-4 GPU-h
worst case, usually < 1 h.

### 7.3 Total

TwiBot-20: ~1-4 GPU-h. TwiBot-22: ~30-100 GPU-h for 5 seeds of both (reduce seeds if needed).
Plus the shared `botrgcn` precompute (~6 GPU-h, shared).

## 8. ExtScore column

- Transform: shared `[z, m]` (standardised logit, train-fitted; z = 0, m = 1 where missing).
- Default injection (same as BotRGCN, so the whole `botrgcn` family is comparable): append `[z, m]`
  to `num_prop` (5 -> 7) before `linear_relu_num_prop`. Neighbours' ExtScore then flows through the convs.
- Unscored nodes (TwiBot-20 support, any missing account): z = 0, m = 1.
- Ablations: late fusion `[z, m]` concatenated after `linear_relu_output1` (`Linear(130, 2)`), which
  keeps ExtScore out of message passing; both early and late.
- `extscore: none` reproduces the baseline (width 5).

## 9. Pitfalls

1. **`state_dict` reference bug**: the published numbers are from the final weights, not the best
   validation epoch.
2. **No activation between conv layers**, dropout on the input concat, H = 128 (not BotRGCN's 32).
3. **Mini-batch NeighborLoader** training, not full batch; val/test predictions are also sampled.
4. **Edge types ignored, edges directed as stored**; do not symmetrise (that changes GCN degrees and
   GAT neighbourhoods).
5. GCN normalisation uses subgraph degrees: a full-graph GCN at inference is a different model.
6. Hop depth 4 is wasteful but its outputs equal 2-hop (GAT) / 3-hop (GCN) sampling in distribution;
   do not use fewer hops than that.
7. Paper std "(+-0.0)" vs readme std (e.g. GCN TwiBot-20 acc +-1.73): use the readme.
8. Same feature caveats as BotRGCN (`len(name)`, no log, transductive z-score, token mean incl.
   special tokens, first 20 tweets on TwiBot-22).

## 10. Open questions / unverified

- PyG version used (affects sampler internals; `NeighborLoader` exists from PyG 2.0).
- Typical TwiBot-22 subgraph sizes and the epoch count at early stop (measure on the first run).
- Whether the reported numbers came from the code as committed (bug included); the GCN TwiBot-20
  std of 1.73 points is consistent with an unselected final model [U].

## 11. Acceptance check

Mean over seeds (>= 3), bot-class F1, faithful settings (final-weights evaluation):

| Model / dataset | Target acc / F1 | Tolerance |
|---|---|---|
| GCN TwiBot-22 | 78.4 / 55.0 | acc +-1.0, F1 +-3.5 |
| GAT TwiBot-22 | 79.5 / 55.9 | acc +-1.0, F1 +-3.5 |
| GCN TwiBot-20 | 77.5 / 80.9 | acc +-3 (published std 1.7), F1 +-2 |
| GAT TwiBot-20 | 83.3 / 85.3 | +-1.5 |

Ordering sanity: on TwiBot-20, GCN < GAT < BotRGCN in accuracy.
