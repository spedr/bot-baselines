# Kudugunta & Ferrara 2018 (SMOTENN + AdaBoost; contextual LSTM) — implementation spec

Method id: `kudugunta` (benchmark account-level model). Optional paper variant: `kudugunta_clstm`
(tweet-level contextual LSTM; NOT what the benchmark ran).
Family: feature-based (benchmark). Graph: no. Text: no (benchmark); yes (CLSTM variant: GloVe Twitter
+ LSTM). Temporal: no. GPU: none (benchmark); small (CLSTM).

## 1. Summary

Kudugunta and Ferrara (Information Sciences 2018) study two tasks on Cresci-2017: (1) account-level
classification from 10 user-metadata fields with classical classifiers, where SMOTE oversampling
followed by Edited-Nearest-Neighbours undersampling (SMOTENN) lifts AdaBoost to 99.81% accuracy;
(2) tweet-level classification with a "contextual LSTM" that reads GloVe-embedded tweet tokens and
concatenates tweet metadata after the LSTM (96.33% accuracy, 200-d GloVe). The TwiBot-22 benchmark
re-implementation (`src/Kudugunta/train.py`) implements ONLY the account-level model: SMOTE → ENN →
`AdaBoostClassifier(n_estimators=100)` on 8 (TwiBot-22) or 6 (other datasets) raw profile fields.
No LSTM, no GloVe, no tweets. Its readme calls the classifier "random forest"; the code uses AdaBoost.
The code also misaligns the boolean features with users (section 9). We implement the benchmark model
as `kudugunta` (with the alignment bug as an opt-in emulation), and spec the CLSTM as an optional variant.

## 2. Citation and code

- Paper: Sneha Kudugunta, Emilio Ferrara. "Deep neural networks for bot detection." Information
  Sciences 467:312-322, 2018. DOI 10.1016/j.ins.2018.08.019; arXiv 1802.04289 (read) **[P-paper]**.
- Official code: none found (GitHub search for the method/authors returns nothing relevant) **[P: searched]**.
- Benchmark re-implementation: `LuoUndergradXJTU/TwiBot-22`, `src/Kudugunta/{train.py,readme.md}`,
  HEAD `54adf9a1dbb6dc238c932d3147a1875def5817d8`; code unchanged since `484f3fcc3b` (2022-06-04).
  MIT licence. Deps: scikit-learn, imbalanced-learn.
- Benchmark paper: Feng et al. 2022, arXiv 2206.04564v6, Tables 2 and 8; appendix B.1 describes the
  baseline as both tasks, but only account level was run **[P-paper vs P-code]**.

## 3. Inputs

### 3.1 Paper features **[P-paper §2]**

Account level (10): statuses count, followers count, friends count, favourites count, listed count,
default profile, geo enabled, profile uses background image, verified, protected.
Tweet level ("6", but the paper lists 5): retweet count, reply count, favorite count, # hashtags,
# URLs (the sixth is presumably # mentions **[U]**).

### 3.2 Benchmark features (`train.py` L145-164, L223-233) **[P-code]**

Order: `['followers_count','following_count','tweet_count','listed_count','protected','verified',
'profile_image_url','location']`; the first 6 on all datasets, all 8 on TwiBot-22 only.

| # | Value |
|---|---|
| 0-3 | `public_metrics.{followers_count, following_count, tweet_count, listed_count}`, raw, unscaled |
| 4 | `protected`: 0 if key absent, None, the string `'False'`, or `== 0` (so bool False → 0); else 1 |
| 5 | `verified`: same rule |
| 6 | TwiBot-22 only: 0 if `profile_image_url == ''` else 1 (≈ constant 1) |
| 7 | TwiBot-22 only: 0 if `location is None` else 1 |

Dropped vs paper: favourites count (absent from v2 user objects; readme L16), default profile,
geo enabled, background image; added: image and location flags on TwiBot-22.
TwiBot-20 caveat: the raw TwiBot-20 booleans are `"False "`/`"True "` (trailing space). If the
benchmark's converted `node.json` kept them (BotHunter/SGBot code special-cases exactly these strings
for TwiBot-20), `'False ' != 'False'` sets columns 4-5 to 1 for every TwiBot-20 user (constant)
**[inference, U]**. Option `kudugunta.tw20_literal_bool_strings: true` to emulate; default: parse properly.

### 3.3 What we need from the shared layer

- `users.parquet`: `idx, user_id, split, label`; for the misalignment emulation also `raw_order`
  (position of the user in the raw `user.json` / `node.json`) and `split_csv_order` (row position in
  `split.csv`). Request both from the data-layer agent.
- `profile.parquet`: `followers_count, following_count, statuses_count, listed_count, protected,
  verified` (+ TwiBot-22 `profile_image_url`, `location` as raw nullable strings, keeping the
  distinction between null and empty).
- CLSTM variant only (3.4): per-tweet text + `retweet_count`, `reply_count`, `like_count`,
  `n_hashtags`, `n_urls`, `n_mentions` in the tweets store; GloVe Twitter vectors.

### 3.4 CLSTM variant inputs (optional) **[P-paper §3.2]**

- Tokenisation: GloVe-Twitter preprocessing: hashtags → `<hashtag>`, URLs → `<url>`, numbers →
  `<number>`, mentions → `<user>`; common emoticons → `<smile>`, `<heart>`, `<lolface>`,
  `<neutralface>`, `<angryface>`; ALL-CAPS word → lower-cased word + `<allcaps>`; letters repeated
  >2 times → word + `<elong>` (tag name per the GloVe script **[U]**); lower-case everything.
- Embedding: pre-trained `glove.twitter.27B` 200-d (paper's best; 25/50/100 also tested), 1,193,514
  tokens; frozen **[U: frozen or fine-tuned not stated]**; max length 48 tokens **[U]**.
- Auxiliary input per tweet: retweet count, reply count, favourite (like) count, # hashtags, # URLs,
  # mentions; log1p + standardise on train **[U: paper gives no scaling]**.
- Which tweets: TwiBot-22: newest 20 per user from the store (cost cap, ours); TwiBot-20: all ≤ 200
  texts, but TwiBot-20 has no per-tweet counts (only texts) → counts of hashtags/URLs/mentions by
  regex, retweet/reply/like counts unavailable (set 0 + document) → on TwiBot-20 the CLSTM is
  effectively text + 3 entity counts.
- TwiBot-22 per-tweet metadata exists only in raw `tweet_*.json` (`public_metrics`, `entities`); the
  processed jsonl has texts only, so the shared layer must stream-extract it.

## 4. Architecture

### 4.1 Benchmark (`kudugunta`) **[P-code L117-142]**

Per run i ∈ {0..4}:
1. `X_s, y_s = SMOTE().fit_resample(X_train, y_train)`: imblearn defaults: `k_neighbors=5`,
   `sampling_strategy='auto'` (minority up to majority count), `random_state=None` (unseeded).
2. `X_e, y_e = EditedNearestNeighbours().fit_resample(X_s, y_s)`: `n_neighbors=3`,
   `kind_sel='all'`, `sampling_strategy='auto'` (clean all classes but the minority).
3. `AdaBoostClassifier(n_estimators=100, random_state=i*100)` → seeds 0,100,200,300,400 (the readme says
   100-500). Defaults of the 2022 sklearn: base estimator `DecisionTreeClassifier(max_depth=1)`,
   `learning_rate=1.0`, `algorithm='SAMME.R'`. SAMME.R was deprecated in sklearn 1.4 and removed in
   1.6; pin `scikit-learn==1.5.2`, `imbalanced-learn==0.12.*` for the faithful run, or use SAMME and
   record the difference **[U: benchmark versions]**.
4. Hard `predict` on test.
Features are raw counts: SMOTE/ENN Euclidean distances are dominated by followers/following/tweet
counts; reproduce that (no scaling) in faithful mode.

### 4.2 CLSTM variant (`kudugunta_clstm`) **[P-paper §3.2, Fig. 1-2]**

- `LSTM(input=200, hidden=32)`, single layer, state reset per tweet; output = last hidden h ∈ R^32.
- Auxiliary output: `Linear(32→1)` + sigmoid on h (regulariser).
- Main: `concat(h, aux_meta[6])` → `Linear(38→128)` + ReLU → `Linear(128→64)` + ReLU →
  `Linear(64→1)` + sigmoid.
- Loss: 0.8 × BCE(main) + 0.2 × BCE(aux). No class weights (the paper does not oversample the
  CLSTM inputs).
- Account score (ours, not in paper **[U]**): mean of tweet-level P(bot) over the account's tweets;
  threshold 0.5 (or tuned on val, reported separately).

## 5. Training

### 5.1 Benchmark

- Index construction (L184-214): train users in `split.csv` order get rows 0..j-1, then val, then test;
  `train_mask = range(j)`, `test_mask = range(j+k, N)`. Val is never used.
- SMOTE/ENN on train only; refit every run. 5 runs; mean and std with `ddof=1` (L137-142).
- Metrics computed manually (bot class): P = TP/(TP+FP), R, F1, acc, AUC on HARD labels.
- Ours: seeds: AdaBoost `random_state = 100*i`, SMOTE `random_state = i`, ENN deterministic,
  i ∈ {0..4}; `n_jobs` for ENN neighbours = 8.

### 5.2 CLSTM variant

Paper gives no optimiser/epochs/split for the LSTM **[U]**. Ours: Adam lr 1e-3, batch 1024 tweets,
≤ 5 epochs, early stopping on val account-level bot F1 (patience 1), tweets inherit their account's
label and split (no account appears in two splits), seeds 0..4.

## 6. Evaluation and published numbers

Benchmark (accuracy Table 2; bot F1 Table 8; P/R `src/Kudugunta/readme.md`; mean ± std over 5 runs):

| Dataset | Acc | Prec | Rec | F1 (bot) |
|---|---|---|---|---|
| TwiBot-22 | 65.9 ±0.0 | 44.31 ±0.00 | 61.98 ±0.00 | 51.7 ±0.0 |
| TwiBot-20 | 59.6 ±0.7 | 80.40 ±0.60 | 33.47 ±1.30 | 47.3 ±1.4 |
| Cresci-2015 | 75.3 ±0.1 | 100.0 | 60.95 | 75.3 ±0.2 (readme 75.74 ±0.16) |
| Cresci-2017 | 88.3 ±0.2 | 98.53 | 85.88 | 91.7 ±0.2 |
| G-17 / M-18 / C-S-18 / C-R-19 / B-F-19 | 70.0 / 91.0 / 77.5 / 62.9 / 74.0 | | | 49.8 / 94.5 / 50.9 / 49.2 / 49.6 |

Note: TwiBot-22 bot F1 51.7 is higher than SGBot's 36.6 despite lower accuracy: SMOTENN rebalances the
training set, so the model trades precision for recall at the 0.5 threshold.
LMBot re-run (1:1:8 random split, different from the benchmark): Cresci-2015 96.11 / 96.88,
TwiBot-20 58.08 / 48.07, Cresci-2017 91.26 / 93.98, Midterm-18 91.86 / 94.96 **[P-paper, LMBot Table 1]**.
The Cresci-2015 gap (75.3 vs 96.1) has the same signature as SGBot's (P ≈ 1, R ≈ 0.61); see
`sgbot.md` §6.1: likely a defect in the benchmark's converted Cresci-2015 file **[U]**.

Original paper (Cresci-2017 genuine + social spambots #1-3; protocol not stated) **[P-paper Tables 2-3]**:
account level: RF 98.39% acc / 0.9845 AUC (no resampling); AdaBoost + SMOTENN 99.81% / 0.9981;
tweet level: metadata-only RF 80.42%, AdaBoost + SMOTENN 92.34%, LSTM tweet-only (50-d) 95.53%,
contextual LSTM 25/50/100/200-d 95.67 / 96.18 / 96.18 / 96.33% (AUC 0.9643 at 200-d).

## 7. Scale plan

Memory rule (mandatory): every job via `tools/memrun` (MemoryMax 10G, no swap, lock); target peak RSS
≤ 8 GB; never load a full TwiBot-22 tweet/edge file or full feature table into RAM; count DataLoader
workers and pinned buffers.

### 7.1 Benchmark model

| Step | TwiBot-22 arithmetic | Peak host RSS | VRAM | Wall |
|---|---|---|---|---|
| Features | 1M × 8 × 8 B = 64 MB (column projection of `profile.parquet`) | < 0.5 GB | 0 | < 1 min |
| SMOTE | train 700k (~98k bots, ~602k humans [label.csv: 14.0% bots overall]) → +504k synthetic → 1.2M × 8 × 8 B = 77 MB; 5-NN over 98k minority | < 1 GB | 0 | < 1 min |
| ENN | 3-NN (4 incl. self) over 1.2M points in 8-D, kd-tree; neighbour index arrays 1.2M × 4 × 8 B = 38 MB; tree ~100 MB | < 1.5 GB | 0 | 2-10 min [U] |
| AdaBoost | 100 stumps on ~1.1M × 8 | < 1 GB | 0 | 2-4 min |
| Total | 5 runs | ≤ 1.5 GB | 0 | ≈ 30-60 CPU-min |

TwiBot-20: 8,278 train users; seconds.

### 7.2 CLSTM variant (optional)

| Step | Arithmetic | Peak host RSS | VRAM | Wall |
|---|---|---|---|---|
| Per-tweet metadata (shared layer) | stream ~100 GB of raw `tweet_*.json` at ~70 MB/s ≈ 25 min, line/ijson streaming, never a whole file | < 2 GB | 0 | ~30-60 min (shared cost) |
| Tokenise + map to GloVe ids | 20M tweets (20/user × 1M) × 48 int32 = 3.84 GB → write to on-disk memmap in chunks; GloVe vocab dict ~1.19M strings ≈ 150 MB; 8 forked workers | ≈ 3 GB | 0 | regex tokenizer ~20k tweets/s/core [U] → 20M / 160k/s ≈ 2-5 min; budget 15 min |
| Training | GloVe 1.19M × 200 fp16 = 477 MB on GPU; batch 1024 × 48 × 200 fp16 = 20 MB; LSTM params ~30k; 2 DataLoader workers reading memmap slices, pinned buffers 2 × 1024 × 48 × 4 B ≈ 0.4 MB | ≈ 2.5 GB (memmap pages are page cache, not RSS-charged beyond touched pages; bound by reading sequential chunks) | < 2 GB | 20M tweets / ~50k tweets/s ≈ 7 min/epoch [U]; 5 epochs × 5 seeds ≈ 3 GPU-h |
| Inference / aggregation | 100k test users × 20 tweets | < 2 GB | < 1 GB | 1 min |

## 8. ExtScore column

- `extscore: none` reproduces the benchmark feature matrix exactly.
- `extscore: p` (benchmark model): append `extscore_p` and `extscore_missing` to the account feature vector BEFORE
  SMOTE (width 8 → 10 on TwiBot-22, 6 → 8 elsewhere). SMOTE interpolates synthetic bots between real
  bots, so a -1 sentinel would produce meaningless intermediate values. Default transform here:
  `extscore_p` = p if observed else the TRAIN observed mean of p; `extscore_missing` ∈ {0,1}; after SMOTE set
  `extscore_missing := round(extscore_missing)`. Ablation: shared `[p_or_-1, m]` transform unchanged.
  The raw-count features dominate SMOTE/ENN distances, so the neighbour structure is essentially
  unchanged by the ExtScore columns (the stumps still see them). Ablation: ExtScore excluded from SMOTE/ENN
  distances (resample on the 8 base columns, carry ExtScore of the nearer parent to synthetic rows).
- Late-fusion ablation: logistic regression on `[logit(clip(ada_proba)), z, m]` fitted on val.
- CLSTM variant: append `[z, m]` (neural transform) to the auxiliary metadata vector, so the
  post-LSTM concatenation becomes `concat(h[32], meta[6], z, m)` → `Linear(40→128)`; the auxiliary
  output head stays text-only. Alternative: late fusion at account level (LR on mean tweet logit + z + m).
- ExtScore must cover train; log coverage per split. Support users unused.

## 9. Pitfalls

1. **Feature/user misalignment bug** (L228-233): count columns are written to row `dict[id]`
   (split-order position) but boolean columns to row `i` (raw file position):
   `X[dict[data[i]['id']]][j] = ...` for j < 4, `X[i][j] = ...` for j ≥ 4. Labels use `dict[id]`. On
   TwiBot-22, `split.csv` is grouped train/val/test while `label.csv` (and, judging from the processed
   jsonl, `user.json`) interleave splits: only 1 of 1M rows has the same id at the same position in
   `split.csv` and `label.csv` **[P: files inspected; user.json order U]**. So the boolean columns are
   effectively shuffled across users, i.e. noise. Default: correct alignment. `kudugunta.bench_misalign:
   true` reproduces the bug exactly from `raw_order` and `split_csv_order`: the user at split-order
   position r receives the booleans of the user at raw position r.
2. The readme says random forest and seeds 100-500; the code uses AdaBoost and seeds 0-400.
3. SMOTE is unseeded in the benchmark; the reported std (0.0000 on TwiBot-22) reflects the large
   train set, not determinism.
4. `get_accuracy` swallows exceptions and returns 0 (crashes the unpacking) on a degenerate predictor.
5. `all` is built from the first `len(label.csv)` entries of the raw file; it assumes labelled users come
   first. On TwiBot-20 (229,580 user nodes, 11,826 labelled) this must hold for the code to work **[U]**.
6. SAMME vs SAMME.R (sklearn ≥ 1.6) changes results; pin versions.
7. CLSTM: tweet-level labels inherited from accounts; never split one account's tweets across splits.
8. No transductive leakage in the benchmark model.

## 10. Open questions / unverified

- Benchmark sklearn/imblearn versions **[U]**.
- Whether `user.json` order equals `label.csv` order (needed to emulate the bug exactly) **[U]**;
  the data layer should record the raw order.
- Content of the benchmark's TwiBot-20 `node.json` booleans **[U]**.
- CLSTM: optimiser, epochs, max length, embedding fine-tuning, 6th tweet feature, tweet→account
  aggregation are all unspecified in the paper **[U]**.

## 11. Acceptance check

`kudugunta`, faithful versions pinned, `extscore: none`, 5 runs:
- TwiBot-22 official split, `bench_misalign: true`: accuracy 65.9 ± 1.5, bot F1 51.7 ± 2.0.
  With correct alignment, report separately (expected similar or slightly better; booleans were noise).
- TwiBot-20 official split: accuracy 59.6 ± 2.0, bot F1 47.3 ± 3.0 (with `tw20_literal_bool_strings`
  both on and off; whichever matches is evidence for the benchmark's conversion).
- Cresci-2017 (our hash split; sanity): accuracy ≥ 85.
`kudugunta_clstm` has no benchmark target; sanity only (TwiBot-22 bot F1 should beat 30).
