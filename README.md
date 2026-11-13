# bot-baselines

Re-implementations of published social-bot detection methods for Twitter/X, built on one shared data
layer and one evaluation harness so that every method is trained and scored under the same splits,
inputs and metrics. The suite covers feature-based classifiers, a text-only baseline, homogeneous
and relational graph neural networks, heterogeneous GNNs, a mixture of experts, contrastive and
dynamic-graph models. It is sized to run on a single 12 GB GPU with 16 GB of system RAM, including
on the full TwiBot-22 graph (1M accounts).

This is an independent re-implementation, not the authors' code. Each method was written from a
spec built from the paper and, where it exists, the official or benchmark code
(`docs/methods/<method>.md`). Each was then reviewed twice, first for correctness and then for
efficiency. Where the paper, the official code and the TwiBot-22
benchmark re-run disagree, the spec records which one we follow and why. Several methods also have
a flag that reproduces a documented upstream bug for comparison.

## Methods

| Method | Paper | Module | Family |
|---|---|---|---|
| SGBot | Yang et al., AAAI 2020 [`Yang2020Scalable`] | `bb.methods.sgbot` | profile features, random forest |
| BotHunter | Beskow & Carley, SBP-BRiMS 2018 [`Beskow2018BotHunter`] | `bb.methods.bothunter` | profile features, random forest |
| Lee et al. | Lee, Eoff & Caverlee, ICWSM 2011 [`Lee2011Seven`] | `bb.methods.lee2011` | profile + tweet-content features, random forest |
| Kudugunta & Ferrara | Information Sciences 2018 [`Kudugunta2018Deep`] | `bb.methods.kudugunta` | SMOTE+ENN, AdaBoost (benchmark's account-level version) |
| RoBERTa + MLP | Liu et al. 2019 [`Liu2019RoBERTa`], TwiBot-22 benchmark | `bb.methods.roberta_mlp` | frozen text embeddings, MLP |
| BotRGCN (+ w/o-graph ablation) | Feng et al., ASONAM 2021 [`Feng2021BotRGCN`] | `bb.methods.botrgcn` | relational GCN |
| GCN / GAT | Kipf & Welling 2017; Veličković et al. 2018 | `bb.methods.gcn_gat` | homogeneous GNN |
| RGT | Feng et al., AAAI 2022 [`Feng2022RGT`] | `bb.methods.rgt` | relational graph transformer |
| Simple-HGN / HGT | Lv et al., KDD 2021; Hu et al., WWW 2020 | `bb.methods.simplehgn` | heterogeneous GNN |
| BotMoE | Liu et al., SIGIR 2023 [`Liu2023BotMoE`] | `bb.methods.botmoe` | community-aware mixture of experts |
| BotDGT | He et al., IJCAI 2024 [`He2024BotDGT`] | `bb.methods.botdgt` | dynamic graph transformer |
| SEBot | Yang et al., KDD 2024 [`Yang2024SEBot`] | `bb.methods.sebot` | structural-entropy multi-view contrastive GNN |
| BIC | Lei et al., ACL 2023 [`Lei2023BIC`] | `bb.methods.bic` | text-graph interaction, semantic consistency |
| SEGCN | Liu et al., Scientific Reports 2024 [`Liu2024SEGCN`] | `bb.methods.segcn` | subgraph-encoding GCN |

Datasets: TwiBot-20 [`Feng2021TwiBot20`], TwiBot-22 [`Feng2022TwiBot22`] and Cresci-2015
[`Cresci2015Fame`], on their official splits. `references.bib` has BibTeX for every method, dataset
and building block above, and for the related work the specs discuss. Please cite the original
papers when you use a re-implementation.

`docs/methods.md` explains why these methods were chosen, which were excluded and why, and lists
findings from the spec pass that change how some published numbers should be read. Examples:
val/test label leakage in several benchmark scripts on TwiBot-22, an HGT baseline that ignores the
graph in the pinned PyG version, and the class-prior shift between the TwiBot-22 splits.

## Status

This is a snapshot of work in progress (November 2026); interfaces may still change. All fourteen methods are implemented and tested. Results are not published yet; they will be added
once every method has its full set of seeds on all three datasets. SEBot's full-batch training does
not fit a 12 GB GPU on TwiBot-22.

## Installation

```
uv sync          # Python 3.12, torch 2.14.0+cu130, torch_geometric + pyg-lib, pinned in uv.lock
uv run pytest    # synthetic fixtures; no real data needed
```

## Data

Raw releases are read in place, read-only, from `raw/` (a directory or symlink holding
`twibot-20-full/`, `twibot-22-full/` and `bot-repository/cresci-2015/`). `BB_RAW_ROOT` or
`BB_RAW_<NAME>` override the location. Derived artifacts go to `data/`, which `BB_DATA_ROOT`
overrides.

```
make build-twibot20 build-cresci15 build-twibot22   # index, profile, tweet store, graph CSR
make precompute                                    # frozen text embeddings (hours, resumable)
make status verify                                 # build status, invariant and count checks
```

`docs/architecture.md` describes the on-disk layout and the contracts of the data layer and harness.
`docs/shared_requirements.md` gives the requirements behind it, and `docs/data_status.md` the
verified counts and the measured cost of each build step.

## Running a method

```
tools/memrun -m 7G -- uv run python -m bb.methods.<method> --dataset twibot20 --seeds 0 1 2 3 4
```

Results go to `runs/<method>/<dataset>/<run>/result.json`. Each file holds the metrics (accuracy,
bot-class precision, recall and F1, AUC), the config, data digests, timing and peak memory.
`tools/memrun` runs a command in a systemd scope with a hard memory cap and serialises heavy jobs, so
an oversize job is killed on its own instead of taking the machine down. `--help` on each module
lists its variants and ablations.

`tools/vast/` holds a multi-GPU job runner for running many seeds on a rented machine. See its
README.

## External score column

Every method accepts one optional extra per-account input: a probability score from an external
model, with a missing-value indicator. This lets each baseline be run unmodified and with the extra
input under identical conditions. Write `data/<dataset>/extscore/<tag>.parquet` with
`bb.extscore.write_extscore`, then pass `--with-extscore <tag>`.

A shared leakage guard requires train-row scores to be out-of-fold, or to come from a label-blind
model, and the external model must not have been tuned on any evaluation account. Per-method flags such
as `--extscore-inject` select where the score enters the model. `bb extscore-synth <dataset>` writes a synthetic column
for tests. `docs/architecture.md` section 8 has the details.

## Licence and attribution

Released under the MIT licence (`LICENSE`). The code in this repository is new, with one exception: `bb/methods/sebot/tree_ref.py` is a port
of SEBot's coding-tree builder (Apache-2.0; licence and statement of changes next to it), used as a
test oracle. Several original implementations carry no licence, so nothing was copied from them;
each spec lists the upstream repositories that were consulted.
The datasets are not redistributed here; obtain them from their authors under their terms.
