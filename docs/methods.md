# Selected methods

Fourteen specs covering thirteen method families (GCN and GAT share one spec; HGT is a variant
section of the Simple-HGN spec). They are listed in implementation order: each step reuses the
artifacts of the steps before it. Tier A is what a reviewer expects as the baseline suite. Tier B
contains the recent state-of-the-art graph models; they are harder, and their published numbers are
less certain. Specs: `docs/methods/<file>.md`. Shared inputs: `docs/shared_requirements.md`.

All numbers below were checked by the spec writers in the paper PDF or the official/benchmark code
(sources and evidence labels are in each spec). F1 is **bot-class** F1 unless marked otherwise. The
"Benchmark" numbers are the TwiBot-22 benchmark re-runs (Feng et al. 2022, Tables 2 and 8: mean over
5 runs); "Paper" numbers come from the method's own paper.

## Implementation order

Tier A: core baselines
1. **SGBot** (`sgbot.md`). Yang et al. 2020 profile features with a random forest. Profile only, CPU. It
   validates `profile.parquet` and the harness.
2. **BotHunter** (`bothunter.md`). Beskow & Carley 2018 Tier-1 features with a random forest. Profile
   only; a second classic feature baseline.
3. **Lee et al. 2011** (`lee2011.md`). Random forest on profile plus tweet-content features. It is the
   first consumer of the tweet store and is strong on Cresci.
4. **Kudugunta & Ferrara 2018** (`kudugunta.md`). The benchmark's account-level SMOTE+ENN+AdaBoost
   version. It has the highest benchmark bot-F1 among the cheap TwiBot-22 baselines (51.7). The
   paper's contextual LSTM is an optional variant.
5. **RoBERTa + MLP** (`roberta_mlp.md`). The text-only baseline. It validates the embedding tables and
   isolates what text alone gives.
6. **BotRGCN, plus the "w/o graph" MLP** (`botrgcn.md`). The reference graph baseline. It defines the
   shared `botrgcn` feature scheme that every later graph method reads.
7. **GCN / GAT** (`gcn_gat.md`). Homogeneous GNN baselines on the same tensors.
8. **RGT** (`rgt.md`). A relational graph transformer and the standard heterogeneity-aware baseline.
9. **Simple-HGN, plus the HGT variant** (`simplehgn.md`). Heterogeneous GNN baselines from the
   benchmark.

Tier B: recent state of the art (each reuses `botrgcn`)

10. **BotMoE** (`botmoe.md`). A community-aware mixture of experts with an explicit metadata expert,
    which is the most natural slot for the ExtScore column. It has official TwiBot-22 numbers.
11. **BotDGT** (`botdgt.md`). A dynamic graph transformer over yearly `created_at` snapshots. It is
    the only temporal method and has official TwiBot-22 numbers.
12. **SEBot** (`sebot.md`). Structural-entropy multi-view contrastive learning. It is the top
    TwiBot-20 graph method with Apache-2.0 code. The coding-tree builder must be rewritten sparse.
13. **BIC** (`bic.md`). A text-graph interaction model with semantic consistency over per-tweet
    sequences. It is the strongest TwiBot-20 result with code, but it is expensive on TwiBot-22.
14. **SEGCN** (`segcn.md`). A subgraph-encoding GCN (GNN-AK style). It is
    reimplemented from the paper because the released code cannot produce the paper's numbers (see
    below).

## Summary table

GPU hours are for 5 seeds on our RTX 4070 Ti, excluding the shared precompute. The shared precompute
is about 9-12 h of wall-clock, mostly the TwiBot-22 tweet pass plus RoBERTa (see
shared_requirements.md section 10). Ranges are estimates from each spec; measure on one seed first.

| # | Method | Family | Graph? (relations) | Tweet text? | Temporal? | Official numbers: acc / bot-F1 | Difficulty | GPU h (TB-20 / TB-22) |
|---|---|---|---|---|---|---|---|---|
| 1 | SGBot | feature, RF | no | no (profile only) | no | Benchmark: TB-22 75.1 / 36.6; TB-20 81.6 / 84.9; C-15 77.1 / 77.9 (artefact) | easy | 0 / 0 (CPU minutes) |
| 2 | BotHunter | feature, RF | no | no | no | Benchmark: TB-22 72.8 / 23.5; TB-20 75.2 / 79.1; C-15 96.5 / 97.2 | easy | 0 / 0 |
| 3 | Lee 2011 | feature, RF | no | yes (first 20 per user on TB-22, all on TB-20) | no | Benchmark: TB-22 76.3 / 30.4; TB-20 77.4 / 80.0; C-15 98.2 / 98.6; C-17 98.8 / 99.3 | easy-medium | 0 / 0 (about 1 CPU h) |
| 4 | Kudugunta | feature, SMOTE+ENN+AdaBoost (CLSTM variant optional) | no | no (CLSTM: yes plus per-tweet metadata) | no | Benchmark: TB-22 65.9 / 51.7; TB-20 59.6 / 47.3; C-15 acc 75.3 | easy (CLSTM medium) | 0 / 0 (CLSTM about 3) |
| 5 | RoBERTa + MLP | text | no | yes (des + mean of tweets) | no | Benchmark: TB-22 72.1 / 20.5; TB-20 75.5 / 73.1, but that F1 is **human-class** (bot-class about 77.6, derived) | easy | under 0.5 / under 0.5 |
| 6 | BotRGCN (+ w/o graph) | relational GNN | yes (following, followers) | yes | no | Paper TB-20 84.62 / 87.07; Benchmark TB-20 85.8 / 87.3, TB-22 79.7 / 57.5; w/o graph TB-20 82.6 / 83.8, TB-22 75.4 / 41.2 | medium (TB-22 medium-hard) | under 0.5 / under 0.5 |
| 7 | GCN / GAT | homogeneous GNN | yes (both merged, untyped) | yes | no | Benchmark GCN TB-20 77.5 / 80.9, TB-22 78.4 / 55.0; GAT TB-20 83.3 / 85.3, TB-22 79.5 / 55.9 | medium | 1-4 / 30-100 (sampling-bound; can be reduced, see spec) |
| 8 | RGT | relational graph transformer | yes (following, followers) | yes | no | Paper TB-20 86.64 / 88.21; Benchmark TB-20 86.6 / 88.0, TB-22 76.5 / 42.9 | medium | 7-12 in total |
| 9 | Simple-HGN / HGT | heterogeneous GNN | yes (following, followers) | yes | no | Benchmark SHGN TB-20 86.7 / 88.2, TB-22 76.7 / 45.4; HGT TB-20 86.9 / 88.2, TB-22 74.9 / 39.6 | medium | 13-16 in total |
| 10 | BotMoE | MoE over modalities + GNN experts | yes (following, followers) | yes | no | Paper TB-20 87.76 / 89.22; TB-22 79.25 / 56.62; C-15 98.50 / 98.82 | medium | about 2.5 / about 20 |
| 11 | BotDGT | dynamic graph transformer | yes (both merged) | yes | **yes** (yearly snapshots from account `created_at`) | Paper TB-20 87.25 / 88.87; TB-22 79.33 / 58.15 | medium | about 0.3 / 8-25 |
| 12 | SEBot | structural-entropy contrastive GNN | yes (following, followers, plus encoding trees) | yes | no | Paper TB-20 87.24 / 88.74 (no TB-22 numbers; TB-22 would be our extension) | hard | under 0.5 (+ up to 1 CPU h) / 15-30 (+ 2-12 CPU h; extension) |
| 13 | BIC | text-graph interaction | yes (follow, merged) | yes (**per-tweet** sequence, at most 200) | no | Paper TB-20 87.61 / 89.13; C-15 98.35 / 98.71 (no TB-22 numbers; extension) | hard | under 1 / 20-50 + 8-16 h precompute (extension) |
| 14 | SEGCN | subgraph-encoding GCN | yes (following and followers, undirected) | yes | no | Paper TB-20 87.01 / 88.74; TB-22 82.71 / 59.31 (unreproduced; see risks) | hard | under 1 / 15-28 |

Datasets beyond TwiBot-20/22: every Tier-A feature and text method also runs on Cresci-2015,
Cresci-2017, Caverlee-2011 and BotSim-24 (feature schemes adapted where fields are missing). The
graph methods also run on Cresci-2015, but numbers are comparable to published ones only on the
benchmark split. That split may be the one in the MGTAB-repo Cresci-15 tensors we hold (unverified;
see shared_requirements.md section 1).

## Notable findings from the spec pass

These change how published numbers should be read. Details and evidence are in each spec.

- **Benchmark label leakage on TwiBot-22.** The RGT, Simple-HGN and HGT scripts compute the loss on
  every node in the sampled subgraph. All 1M TwiBot-22 users are labelled, so val and test labels
  enter training. Our default is seed-only loss, with an emulation flag.
- **The HGT baseline ignored the graph.** In PyG 2.0.2 (the pinned version), `HGTConv` drops the
  aggregated messages. It is not clear which version produced the published numbers.
- **The Simple-HGN edge residual is inert** in the benchmark code (effectively beta = 0).
- **RoBERTa-MLP's TwiBot-20 "F1" of 73.09 is human-class.** The bot-class F1 is about 77.6.
- **Kudugunta (benchmark) is not the paper's deep model.** Its boolean features are also misaligned
  with users on TwiBot-22 (1 of 1M ids is at the same position in `split.csv` and `label.csv`).
- **TwiBot-20 tensors behind the published BotRGCN-family numbers** were not produced by any posted
  script. The original `Dataset.py` writes 6 + 11 columns, the model reads 5 + 3, and the benchmark's
  TwiBot-20 preprocessing crashes. The authors' Google Drive zip (if still live) is the best check.
- **BotMoE's metadata expert sees zeros** for every labelled user in the released code (index
  misalignment). The released code also differs architecturally from the paper (3 graph MoEs, no 2-D
  convolution, lr 1e-5). We build the code architecture with the metadata input fixed.
- **BIC's numbers come from the `TwiBot20v2` folder.** The GCN in that folder indexes local subgraph
  edges against global features.
- **Checkpoint-by-reference bugs** (`best = model.state_dict()` without a copy) in GCN/GAT and BIC
  mean that the "best" model is the last one.
- **TwiBot-22 class prior shift.** Bots are 7.8% of train, but 27.9% of val and 29.4% of test
  (computed from `label.csv` + `split.csv`).

## SEGCN resolution

"SEGCN" is a real, distinct bot-detection method: **Liu, Li, Yang, Gong, Lu, Liu 2024, "SEGCN: a
subgraph encoding based graph convolutional network model for social bot detection", Scientific
Reports 14:4122, DOI 10.1038/s41598-024-54809-z**. It reports TwiBot-20 87.01 / 88.74 and TwiBot-22
82.71 / 59.31. It is not a misspelling of SEBot, although SEBot's TwiBot-20 F1 is coincidentally also
88.74. No other bot detector goes by SEGCN, SE-GCN or SeGCN (scholarly and GitHub search).

Code `FengLiuii/SEGCN` (MIT) was published in October 2025. It does not help reproduce the paper:
- The files are a byte-identical copy of GNN-AK with a renamed model class.
- The shipped config never runs the subgraph path (`mini_layers: 0`) and none of the paper's
  hyperparameters appear in it.
- The extractor builds dense N x N masks, about 470 GB for TwiBot-20 and about 9 TB for TwiBot-22.

The spec is therefore paper-first: the paper's equations and Table 4, with GNN-AK operator details
filling the gaps, each gap marked. The paper's Table 5 has visible problems. Its TwiBot-22 baseline
rows copy the benchmark means with different standard deviations. Its GCN "F1" equals the benchmark's
GCN recall. It also reports SATAR on TwiBot-22, which the benchmark could not run. So we treat the
TwiBot-22 82.71 as a claim to test, not a target.

## Excluded methods and why

| Method | Reason |
|---|---|
| HW-GNN (Liu et al., Findings of EMNLP 2026; TB-22 80.73 / 61.95, TB-20 90.37 / 91.51) | No code: the URL given in the paper (`github.com/liuzida/HW-GNN`) returns 404. The paper also omits the input features, hidden size, lr, epochs, focal-loss parameters and block count. Its TwiBot-22 baseline rows give several spectral GNNs about 58-60% accuracy, below the 70.6% all-human rate of the test split. The F1 type is not stated. |
| BSG4Bot (ICDE 2025), BotBR (SIGIR 2025), EvoBot (ACL 2025) | Not assessed in this pass. Their numbers are known only from HW-GNN's tables (secondary source). `liuzida/BSG4Bot` exists (no licence), so BSG4Bot is the first candidate for a later pass. |
| BotUmc, CDRBot, LGB, LKD, BotSay, BotPercent | These need LLM inference or fine-tuning (API spend or 14B-scale training on several 80 GB GPUs), have no code (LGB, CDRBot URL truncated) or are inference-only (BotPercent). They fall outside "reimplement on one 12 GB GPU without API spend". |
| LMBot (WSDM 2024) | It has code but uses a random 1:1:8 split, so its numbers are not comparable to the official splits. It also needs LM fine-tuning. It is a reasonable later addition as a distillation baseline. |
| HOFA, MGDIL, TRACE-Bot, SAHG, TopGateGNN, RMNP | Non-standard splits or setups (HOFA 10/10/80, SAHG k-NN graphs, MGDIL cross-domain), code not found, or not read. They are not leaderboard baselines a reviewer would require. |
| Botometer / BotometerLite | API services. No public training code exists (SGBot is the faithful stand-in). |
| SATAR, Kantepe, Abreu, Kouvela, LOBO, Varol, Cresci, and the rest of the benchmark's `src/` | They are re-implemented in the benchmark repo (MIT) and could be added cheaply. They were left out to keep the suite to the families a reviewer expects. Kantepe et al. (TB-22 bot-F1 58.7, the best feature-based number in the benchmark) is the strongest candidate to add next. |

## Order rationale and dependencies

- Steps 1-4 need only `users.parquet`, `profile.parquet` and (for Lee) the tweet store. They can start
  while the TwiBot-22 tweet pass and RoBERTa embeddings run.
- Step 5 needs the embedding tables. Steps 6-14 need the `botrgcn` scheme and the graph CSR.
- Steps 6-9 share one PyG model skeleton (input projections of the four feature blocks, then a conv
  stack, then a head). Build it once in step 6.
- Steps 12-14 each need a method-local structural artifact: SEBot coding trees, BIC per-tweet
  sequences, SEGCN random-walk subgraphs. They can run in parallel only if their memory plans fit
  side by side, which by default they do not, because `tools/memrun` serialises them.
