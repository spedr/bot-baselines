# RoBERTa + MLP (TwiBot-22 benchmark text baseline)

Status: spec, written 2026-11-12. Labels: **[P-paper]**, **[P-code]** (path), **[S]**, **[U]**.

## 1. Summary

A text-only baseline: frozen RoBERTa features of a user's description and of their tweets (the same
`des_tensor` / `tweets_tensor` built for BotRGCN, see `botrgcn.md` section 3.1) are fed to a small
MLP classifier trained with mini-batches. Family: text embedding + MLP. Graph: no. Text: yes
(frozen encoder, no fine-tuning; the benchmark readme's word "finetune" refers to the MLP). Temporal:
no. Encoders: `roberta-base` (tweets; descriptions on TwiBot-22) and, on TwiBot-20, whatever produced
the BotRGCN TwiBot-20 tensors (our default: `distilroberta-base` descriptions, see botrgcn.md). No
profile metadata, no numeric features.

## 2. Citation and code

- Encoder: Yinhan Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
  arXiv 1907.11692, 2019.
- Baseline as run: Feng et al., "TwiBot-22: Towards Graph-Based Twitter Bot Detection", NeurIPS 2022
  D&B, arXiv 2206.04564, App. B.1: "leverages pre-trained language model RoBERTa to encode user
  tweets and descriptions, then feed them to an MLP" [P-paper].
- Code: `LuoUndergradXJTU/TwiBot-22`, commit `54adf9a1dbb6dc238c932d3147a1875def5817d8`, MIT,
  `src/RoBERTa/{Twibot-22,Twibot-20,cresci-2015,cresci-2017}/{train.py,label_list.py,id_list.py,
  des_embedding.py,tweets_tensor.py,user_tweets_dict.py}`, `src/RoBERTa/readme.md`.
- Licence MIT: reuse with attribution allowed; we write our own code.

## 3. Inputs

### 3.1 Which features the reported runs actually used

The RoBERTa folder's own embedding scripts are not what the training scripts read:
- `Twibot-22/train.py` loads `data/twibot22/tweets_tensor.pt` and `des_tensor.pt` [P-code l.46-48].
  `data/twibot22/` is the output directory of `src/twibot22_Botrgcn_feature/twi*.py` (the parallel
  BotRGCN TwiBot-22 feature scripts) [P-code `twi0.py` l.15]. `Twibot-22/tweets_tensor.py` in the
  RoBERTa folder is a T5-small script (copy-paste, 11,826 users, random 512-d fallback) and
  `Twibot-22/des_embedding.py` does not exist. So **TwiBot-22 RoBERTa+MLP = MLP on the BotRGCN
  TwiBot-22 tensors** [P-code, inference from paths; U only in that the file on disk was the same].
- `Twibot-20/train.py` loads `src/BotRGCN/twibot_20/processed_data/{tweets_tensor,des_tensor}.pt`
  [P-code l.40-43]: **the BotRGCN TwiBot-20 tensors**. (The folder's `des_embedding.py` writes to
  `data/T5/...` with a 512-d zero fallback and its `tweets_tensor.py` reads Cresci-2017 data; both
  are unused leftovers.)
- `cresci-2015/train.py` loads `src/botrgcn_new/cresci_15/processed_data` [P-code l.39-40]: BotRGCN
  Cresci-2015 tensors.

Therefore:

| Dataset | des input | tweets input | which tweets |
|---|---|---|---|
| TwiBot-22 | `botrgcn` des: roberta-base, mean over all tokens incl. `<s>`/`</s>`, null -> zeros, `''` -> e_empty | `botrgcn` tweets: roberta-base per-tweet token mean, **first 20 tweets in release-file order**, uniform mean, none -> zeros | 20 per user max ("users' tweet counts could be cut to 20 for time consumption issue" [P-code readme l.42]) |
| TwiBot-20 | `botrgcn` (TwiBot-20 default) des | `botrgcn` (TwiBot-20 default) tweets: all <=200 | see botrgcn.md 3.1 for the provenance caveat |
| Cresci-2015 | `botrgcn` Cresci des | first 20 by `post` edge order | benchmark split only |

Definitions, pooling, truncation and empty handling: `botrgcn.md` sections 3.1.4, 3.1.5, 3.1.7.
From the shared layer: `emb/<encoder>/des.f16.npy`, the `botrgcn` tweets mean file, `users.parquet`
(split, label). Nothing else. No graph, no metadata.

### 3.2 Labels and index alignment

- TwiBot-22: `label_list.py` maps `human -> 1`, else 0, in `label.csv` order; `train.py` uses
  `label = 1 - label_list` so **bot = 1** [P-code l.35-38 of label_list.py, l.49 of train.py].
  Split indices are positions in `label.csv` [P-code train.py l.18-25] and are used to index the
  BotRGCN tensors (stored in `user.json` order). Correct only if both orders agree; the published
  precision (0.63 at a 29% base rate) implies they do [U, inferred]. We index everything by
  `users.parquet.idx`.
- TwiBot-20: `label_list.py` maps `human -> 1`, bot -> 0, support -> -1, in node order, and
  `Twibot-20/train.py` does **not** flip it (l.43): **class 1 = human on TwiBot-20** [P-code].
  sklearn `f1_score` default pos_label = 1, so the published TwiBot-20 F1 is **human-class F1**
  (verified numerically in section 6). Cresci-2015 `train.py` also loads unflipped labels (same
  code), so its F1 is probably human-class too [U].
- Our implementation: always bot = 1 internally; report bot-class F1 and, for the acceptance check
  on TwiBot-20, also human-class F1.

## 4. Architecture

`MLPclassifier` [P-code `Twibot-22/train.py` l.76-111; same structure in Twibot-20/cresci]:

| step | op | shape |
|---|---|---|
| 1 | p1 = Linear(768, 384)(tweet) (no activation) | B x 384 |
| 2 | p2 = Linear(768, 384)(des) (no activation) | B x 384 |
| 3 | x = cat(p1, p2) | B x 768 |
| 4 | x = LeakyReLU(Linear(768, 128)(x)) (module `linear_relu_tweet`) | B x 128 |
| 5 | x = Dropout(p)(x) | |
| 6 | logits = Linear(128, 2)(x) | B x 2 |

- LeakyReLU slope 0.01. Steps 1-4 are a linear map followed by one nonlinearity (the 768->384
  projections have no activation).
- Unused modules exist in the class (`linear_relu1`, `linear_relu2` on TwiBot-22; `linear_relu` on
  TwiBot-20; their call is commented out). They do not affect outputs; they consume RNG at init, so
  seed-for-seed identity with the original is impossible anyway. Do not instantiate them.
- Init: PyTorch defaults (no custom init). Loss: unweighted `CrossEntropyLoss`.
- Dropout p: TwiBot-22 0.5; TwiBot-20 and Cresci-2015 0.3 [P-code].

## 5. Training

| setting | TwiBot-22 | TwiBot-20 | Cresci-2015 |
|---|---|---|---|
| source | `Twibot-22/train.py` | `Twibot-20/train.py` | `cresci-2015/train.py` |
| optimiser | Adam | Adam | Adam |
| lr | 1e-4 | 1e-5 | 1e-4 |
| weight decay | 1e-5 | 1e-6 | 1e-5 |
| batch size | 64 | 64 | 32 |
| epochs | 50 | 100 | 100 |
| dropout | 0.5 | 0.3 | 0.3 |
| shuffle | train yes; val/test no | same | same |

- The argparse block in `Twibot-22/train.py` (lr 1e-3, epochs 50, batch 64, ...) is copied from RGT;
  only `--batch_size` is used (default 64); lr/wd/epochs/dropout come from the `RobertaTrianer`
  defaults above [P-code l.119-127, l.258].
- 5 runs (`for i in range(5)` creates a fresh trainer; same data loaders), no seeds set.
- No early stopping, no checkpointing. After every epoch the code prints train, val and test metrics;
  which epoch's test numbers went into the paper is not recorded [U]. Our default: **last epoch**;
  also log val-best-epoch test metrics as a secondary number.
- Splits: train for loss; val/test only evaluated. Support users (TwiBot-20) are not used.
- `Twibot-22/train00.py` is a separate 10x10 domain-transfer experiment (300 epochs, best test acc);
  ignore it.

## 6. Evaluation and published numbers

Metrics: accuracy, F1 (binary, pos_label = 1; see 3.2 for which class), precision, recall, AUC from
the class-1 logit.

| Dataset | Source | Acc | F1 (as published) | Prec | Rec | class of "F1" |
|---|---|---|---|---|---|---|
| TwiBot-22 | readme [P-code] / Table 2, 8 [P-paper] | 72.07 +- 0.16 | 20.53 +- 1.71 | 63.28 | 12.27 | bot |
| TwiBot-20 | readme / Tables 2, 8 | 75.55 +- 0.18 | 73.09 +- 0.59 | 73.88 | 72.38 | **human** |
| Cresci-2015 | readme / Tables | 97.01 +- 0.13 | 95.86 +- 0.19 | 97.58 | 94.11 | human? [U] |
| Cresci-2017 | readme / Tables | 97.19 +- 0.09 | 94.30 +- 0.18 | 92.43 | 96.27 | human? [U] |

Check of the class (TwiBot-22 test 29,444 bots / 70,556 humans; TwiBot-20 test 640 bots / 543
humans):
- TwiBot-22, bot positive: TP = 0.1227 x 29,444 = 3,613; predicted positive = 3,613 / 0.6328 =
  5,709; FP = 2,096; TN = 68,460; acc = (3,613 + 68,460) / 100,000 = 0.7207. Matches -> bot class.
- TwiBot-20, human positive: TP = 0.7238 x 543 = 393; predicted positive = 393 / 0.7388 = 532;
  FP = 139; TN = 640 - 139 = 501; acc = (393 + 501) / 1,183 = 0.7557. Matches. Bot positive would
  give TP = 463, FP = 164, acc = 0.712: does not match. So the TwiBot-20 F1 is human-class.
- Derived **bot-class** TwiBot-20 metrics (from the means; approximate): precision 501 / 651 =
  0.770, recall 501 / 640 = 0.783, **F1 = 0.776**.

The survey and every leaderboard that copies "RoBERTa 73.1 F1 on TwiBot-20" mix classes; our tables
must use 77.6 (derived) or our own bot-class number.

## 7. Scale plan for our hardware

**Hard requirement:** run every step via `tools/memrun` (MemoryMax 10G, no swap); target peak RSS
<= 8 GB; never load a full feature table into RAM: memmap and gather the needed rows.

- Precompute: none of its own; it reuses the `botrgcn` des/tweets embeddings (botrgcn.md 7.1:
  TwiBot-22 ~2.2 GPU-h, TwiBot-20 ~3.8 GPU-h, shared).
- Training TwiBot-22: 700,000 train rows x (768 + 768) fp16 = 2.15 GB; val 0.61 GB; test 0.31 GB.
  Copy all 1M rows (3.07 GB fp16) to the GPU in 50k-row chunks from the memmaps (host RSS peak
  ~2 GB: torch + CUDA 1.5 GB + 77 MB pinned staging); batches are GPU index gathers
  (`perm[k:k+64]`), no DataLoader workers. VRAM: 3.1 GB inputs + model (0.6M params) + activations
  (64 x 768 x 4 B) ~ **3.5 GB**.
  Steps: 700,000 / 64 = 10,938 per epoch x 50 epochs = 547k optimiser steps; at ~0.4 ms per step
  (tiny MLP, launch-bound) ~ 3.6 min, plus per-epoch val/test eval (300k rows, < 1 s) -> ~5 min per
  run, ~25 min for 5 runs. A per-item `Dataset.__getitem__` + DataLoader would cost ~30 s per epoch
  of Python overhead (~25 min per run); avoid.
  Optional speed-up (deviation, flag): `torch.compile` or CUDA graphs; keep batch 64 for fidelity.
- Training TwiBot-20: 8,278 train rows; 100 epochs x 130 steps = 13k steps; < 1 min per run; VRAM
  < 1 GB; host RSS < 2 GB (memmap gather of 11,826 rows = 36 MB fp16).
- Total: < 0.5 GPU-h for all datasets and seeds (excluding the shared precompute).

## 8. ExtScore column

- Transform: shared `[z, m]` (standardised logit, train-fitted; z = 0 and m = 1 where missing).
- Default injection: concatenate `[z, m]` to `x = cat(p1, p2)` at step 3, widening
  `linear_relu_tweet` to `Linear(770, 128)`. Rationale: same depth as the text features, one
  nonlinearity after fusion.
- Ablations: (a) late fusion: concatenate `[z, m]` after the dropout, `Linear(130, 2)` (a
  logistic-regression-like correction of the text model); (b) a third projection branch
  `p3 = Linear(2, k)([z, m])` with k in {8, 384} concatenated at step 3 (k = 384 mirrors the text
  branches; it lets the MLP up-weight a 2-d input that is otherwise 2 of 770 inputs).
- No graph, so no unscored neighbours; accounts without a score are handled by `m`.
- `extscore: none` must reproduce the baseline (widths 768 / 128).

## 9. Pitfalls

1. **TwiBot-20 label polarity**: class 1 = human in the benchmark TwiBot-20 (and probably Cresci)
   runs; published F1 there is human-class. Use bot = 1.
2. **Not the RoBERTa folder's embedding scripts**: they are T5/Cresci leftovers; the runs used the
   BotRGCN tensors.
3. **First 20 tweets on TwiBot-22**, release-file order, token mean incl. special tokens.
4. **The 768 -> 384 projections have no activation**; the only nonlinearity is after
   `Linear(768, 128)`.
5. Different hyperparameters per dataset (lr 1e-4 vs 1e-5, dropout 0.5 vs 0.3); the argparse
   defaults in `Twibot-22/train.py` are decoys.
6. Index alignment by `label.csv` order in the original: we use `users.parquet.idx` everywhere.
7. Strong class shift on TwiBot-22 (train 7.8% bots, test 29.4%): low bot recall (12%) is expected,
   do not "fix" with class weights in the baseline.
8. Unknown reported epoch: report last epoch; never select on test.

## 10. Open questions / unverified

- Exact epoch used for the published numbers (last vs best).
- Whether Cresci-2015/2017 F1 are human-class (needs the benchmark Cresci test composition).
- TwiBot-20 tensor provenance (shared with BotRGCN; see botrgcn.md section 10).
- Whether the on-disk `data/twibot22/*_tensor.pt` read by `train.py` were exactly the BotRGCN
  tensors (paths say yes).

## 11. Acceptance check

5 seeds, last epoch, mean:

| Dataset | Target | Tolerance |
|---|---|---|
| TwiBot-22 | acc 72.1, bot F1 20.5, bot recall ~12 | acc +-1.0; F1 +-5 (very sensitive to the class shift) |
| TwiBot-20 | acc 75.5, human-class F1 73.1, derived bot F1 ~77.6 | +-1.5 |

Sanity: TwiBot-22 accuracy must exceed the all-human rate 70.56%; TwiBot-20 accuracy must exceed
54.1% (all-bot).
