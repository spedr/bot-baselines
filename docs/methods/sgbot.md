# SGBot / BotometerLite features (Yang, Varol, Hui, Menczer 2020) — implementation spec

Method id: `sgbot`. Family: feature-based (user-object metadata + derived features), random forest.
Graph: no. Text: no (profile strings only). Temporal: no (account age only for rate features).
Encoders: none. GPU: none.

## 1. Summary

Yang et al. (AAAI 2020) propose a scalable detector (later BotometerLite) that uses only the user
object: 8 metadata fields plus 12 derived features (rates per unit age, follower/friend ratio, name
lengths/digit counts, description length, screen-name bigram likelihood), a random forest with 100
trees, and "data selection": training on the combination of labelled datasets that generalises best
to held-out datasets. The TwiBot-22 benchmark re-implementation (`src/SGBot`) keeps the 20-slot
feature vector (with 3 slots zeroed or substituted), drops data selection (each dataset trains on its
own train split) and trains `RandomForestClassifier(n_estimators=100)`. We implement the benchmark
version; a `paper` feature mode restores the original fields where the data has them (TwiBot-20).

## 2. Citation and code

- Paper: Kai-Cheng Yang, Onur Varol, Pik-Mai Hui, Filippo Menczer. "Scalable and Generalizable Social
  Bot Detection through Data Selection." AAAI 2020, 34(01):1096-1103. DOI 10.1609/aaai.v34i01.5460;
  arXiv 1911.09179 (read) **[P-paper]**.
- Official training code: none found. Checked: `osome-iu` org repos (`botometer-python` = API client,
  `Botometer101` = API-usage tutorial scripts, others unrelated), the authors' accounts `yang3kc` and
  `onurvarol`, GitHub repository search for "botometerlite" / data-selection terms (no hits)
  **[P: searched 2026-11-12]**. Agrees with the survey.
- Benchmark re-implementation: `LuoUndergradXJTU/TwiBot-22`, `src/SGBot/{utils.py,preprocess.py,
  train.py,domain_transfer.py,readme.md}`, HEAD `54adf9a1dbb6dc238c932d3147a1875def5817d8`; code
  unchanged since `484f3fcc3b` (2022-06-04), later commits edit only the readme. MIT licence.
- Benchmark paper: Feng et al. 2022, arXiv 2206.04564v6, Tables 2, 8, 12 **[P-paper]**.

## 3. Inputs

### 3.1 Paper features (Table 1 of the paper, 20 features) **[P-paper]**

Metadata (8): statuses count, followers count, friends count, favourites count, listed count,
default profile (bin), profile use background image (bin), verified (bin). (`geo_enabled` was removed
because Twitter stopped providing it.)
Derived (12): tweet freq = statuses/age; followers growth = followers/age; friends growth;
favourites growth; listed growth; followers/friends ratio with denominator `max(friends,1)`;
screen name length; # digits in screen name; name length; # digits in name; description length;
screen-name likelihood. User age = HOURS between probe time (query time, or tweet `created_at` when the
user object comes from a tweet) and account `created_at`. Age itself is not a feature.
Likelihood: geometric mean of bigram probabilities over the 3,969 = 63² bigrams of
`[A-Za-z0-9_]`, estimated from > 2M unique screen names.

### 3.2 Benchmark feature vector (20 columns, this order; `preprocess.py` L100-125) **[P-code]**

| # | Benchmark computation | Paper slot |
|---|---|---|
| 0 | `public_metrics.tweet_count` (null → 0) | statuses count |
| 1 | `followers_count` | followers count |
| 2 | `following_count` | friends count |
| 3 | constant 0 | favourites count |
| 4 | `listed_count` | listed count |
| 5 | `int('default_profile_normal' not in profile_image_url)` (null → 0) | default profile (SUBSTITUTED by a custom-image flag) |
| 6 | constant 0 | profile use background image |
| 7 | `verified` → int (TwiBot-20: `"True "` → 1, `"False "` → 0); null → 0 | verified |
| 8 | col0 / age | tweet freq |
| 9 | col1 / age | followers growth |
| 10 | col2 / age | friends growth |
| 11 | col3 / age (= 0) | favourites growth |
| 12 | col4 / age | listed growth |
| 13 | col1 / col2, but 0 if col2 == 0 | followers/friends (paper: / max(friends,1)) |
| 14 | `len(username.strip())` | screen name length |
| 15 | # chars in `'0'..'9'` in username (unstripped) | digits in screen name |
| 16 | `len(name.strip())` | name length |
| 17 | # digits in name | digits in name |
| 18 | `len(description.strip())` | description length |
| 19 | likelihood of `username.strip()` (3.3) | screen name likelihood |

Age = `(datetime(Y,12,31) - created_at_naive).days` with Y = dataset year (TwiBot-22 → 2022,
TwiBot-20 → 2020, cresci-2015 → 2015, ...), null `created_at` → 730; DAYS not hours (a constant factor
of 24; RF-invariant). Same parser as BotHunter (`calc_age`, L65-86). Null strings give 0 for lengths,
digits and likelihood.

### 3.3 Screen-name likelihood (benchmark `utils.py`) **[P-code]**

1. Alphabet A = `A-Z`, `a-z`, `0-9`, `_` (63 chars; case-sensitive) → 3,969 bigram counters at 0.
2. For EVERY user node in the dataset file (TwiBot-22: all 1M users; TwiBot-20: all 229,580 user nodes
   including unlabelled support users; train, val and test alike), take `username.strip()` and add
   each adjacent pair. No add-one smoothing. A bigram outside A raises KeyError in the benchmark
   (our code: skip it and count the event).
3. p(bigram) = count / total count.
4. likelihood(s) = Π_{i} p(s_i s_{i+1})^(1/(len(s)-1)); 1.0 if len(s) ≤ 1; 0 if any bigram unseen.

Transductive but label-free (uses test/support screen names, no labels). Option
`sgbot.likelihood_corpus: all_users|train_users` (default `all_users` for parity). The paper used an
external corpus of > 2M screen names that is not available.

### 3.4 `paper` mode (optional)

TwiBot-20 raw v1.1 profiles contain `favourites_count`, `default_profile`,
`profile_use_background_image`: fill slots 3, 5, 6 with them and slot 11 = favourites/age; ratio with
`max(friends,1)`. TwiBot-22 user objects (API v2) have none of the three (our processed jsonl stores
`favourites_count = 0`), so `paper` mode on TwiBot-22 only changes slot 13. Ablation only.

### 3.5 What we need from the shared layer

- `users.parquet` (labelled users for training/eval; ALL users incl. support for the bigram corpus).
- `profile.parquet`: `screen_name`, `name`, `description`, `profile_image_url`, `verified`,
  `created_at`, `statuses_count`, `followers_count`, `following_count`, `listed_count`; for `paper`
  mode also `favourites_count`, `default_profile`, `profile_use_background_image`. TwiBot-20 values must
  be stripped and typed (raw strings like `"45568 "`, `"True "`).
- No tweets, no graph, no embeddings.

## 4. Architecture

`RandomForestClassifier(n_estimators=100)`, sklearn defaults (gini, unlimited depth,
`max_features='sqrt'` → int(sqrt(20)) = 4, bootstrap, no class weights, no `random_state`)
(`train.py` L54) **[P-code]**. Paper: 100 trees **[P-paper]**; other hyperparameters not given.
Keep the three constant columns (they consume `max_features` draws).

## 5. Training

Train split → fit; val predicted and printed only; test reported. Seeds 0..4 (`random_state`),
`n_jobs=8`. No data selection (the paper's core contribution is a training-set selection across
Bot-Repository datasets; the benchmark does single-dataset training). `domain_transfer.py` trains on
TwiBot-22's 10 sub-communities (`datasets/Twibot-22/domain/user{i}.json`, not held) in a 10 × 10 grid;
out of scope.

## 6. Evaluation and published numbers

Accuracy (Table 2) and bot F1 (Table 8); P/R from `src/SGBot/readme.md` (mean ± std, 5 runs).

| Dataset | Acc | Prec | Rec | F1 (bot) |
|---|---|---|---|---|
| TwiBot-22 | 75.1 ±0.1 | 73.11 ±0.18 | 24.32 ±0.09 | 36.6 ±0.2 |
| TwiBot-20 | 81.6 ±0.5 | 76.40 ±0.40 | 94.91 ±0.69 | 84.9 ±0.4 |
| Cresci-2015 | 77.1 ±0.2 | 99.45 | 63.67 | 77.9 ±0.1 |
| Cresci-2017 | 92.1 ±0.3 | 98.26 | 90.86 | 94.6 ±0.2 |
| G-17 / M-18 / C-S-18 / C-R-19 / B-F-19 | 78.6 / 99.2 / 81.3 / 80.9 / 75.5 | | | 72.1 / 99.5 / 82.3 / 82.7 / 49.6 |

Table 12 (train+val of TwiBot-22, tested on 1,000 expert labels): acc 91.87, F1 47.43, P 76.16,
R 34.48 **[P-paper]**. An earlier readme revision listed TwiBot-22 0.7392 / 0.2754 (commit
`2fe54129ef` changed it to 0.7508 / 0.3659) **[P-code history]**.

LMBot (Cai et al., WSDM 2024) re-ran SGBot with a 1:1:8 random split (DIFFERENT split): Cresci-2015
97.76 / 98.23, TwiBot-20 79.58 / 83.51, Cresci-2017 95.49 / 97.04, Midterm-18 98.61 / 99.18
**[P-paper, LMBot Table 1]**. Original paper: AUC only (five-fold CV and cross-dataset), not comparable.

### 6.1 The Cresci-2015 artefact (77.1 vs ~97)

Facts: precision 0.9945 and recall 0.637 → nearly all errors are bots predicted human (≈ 36% of test
bots). Kudugunta in the same benchmark shows the same signature on Cresci-2015 (P 1.000, R 0.610,
acc 75.3), while LMBot's re-run of Kudugunta reaches 96.11; Lee (98.2) and BotHunter (96.5), which
share SGBot's follower/friend/status counts, age and name/description lengths (BotHunter also the
image flag), do not. The SGBot and Kudugunta code paths
share no preprocessing (different scripts, different index logic), so the cause is most likely in the
benchmark's converted Cresci-2015 `node.json` itself, in a field that these two methods use and Lee /
BotHunter do not: `public_metrics.listed_count` and `verified` are the only such fields **[inference, U]**.
In the raw Cresci-2015 CSVs (held), the bot users have `verified` empty and `listed_count` mostly 0
(FSF 1169/1169 zero, INT 1333/1337, TWT 750/845) **[P: raw files inspected]**, so a correct conversion
would not hurt; a conversion that, e.g., wrote the counts or flags from the wrong column/record for
one bot subset (INT is 40% of bots, close to the 36-39% missed) would. Not provable without the
benchmark's file. Action: run our implementation on Cresci-2015 built from the raw CSVs (7:2:1 random
split, 5 seeds); expect ≥ 95%. If we get that, report the benchmark number as an artefact and cite both.

## 7. Scale plan

Memory rule (mandatory): every job via `tools/memrun` (MemoryMax 10G, no swap, lock); target peak RSS
≤ 8 GB; never load full TwiBot-22 tweet/edge files or a full feature table into RAM.

| Step | TwiBot-22 | Peak host RSS | VRAM | Wall |
|---|---|---|---|---|
| Bigram table | stream `screen_name` column (1M × ~12 chars) | < 0.3 GB | 0 | < 1 min |
| Features | 10 profile columns for 1M users (column projection); X 1M × 20 × 8 B = 160 MB | < 1 GB | 0 | 1-2 min |
| RF fit per seed | 700k × 20 float32 = 56 MB; 100 trees × ~300k nodes [U] × ~80 B ≈ 2.4 GB | ≈ 3 GB | 0 | n_jobs=8: 1-3 min |
| Eval | 100k | ≈ 2.5 GB | 0 | < 1 min |

TwiBot-20: bigram corpus 229,580 names, features for 11,826 labelled users; seconds.
Total (5 seeds, both datasets): < 30 CPU-min, 0 GPU-h.

## 8. ExtScore column

- `extscore: none` = 20-column baseline, unchanged.
- `extscore: p`: append `extscore_p` (missing → -1.0) and `extscore_missing` as columns 20-21 (width 22;
  int(sqrt(22)) = 4, unchanged draw count).
- Ablations: NaN-native `extscore_p` only; late fusion (LR on `[logit(rf_proba), z, m]` fitted on val);
  ExtScore-alone reference; `paper` mode + ExtScore on TwiBot-20.
- ExtScore needed on (a representative part of) train; log coverage. Support users contribute only screen
  names to the bigram corpus; they never need a ExtScore value.
- Report MDI and permutation importance of `extscore_p`.

## 9. Pitfalls

1. Slot 5 is NOT `default_profile` in the benchmark; it is the custom-profile-image flag.
2. Follower/friend ratio = 0 when friends = 0 (paper: divide by max(friends, 1), i.e. = followers).
3. Bigram likelihood is fitted on all users of the dataset incl. test and support (label-free
   transduction). State this when reporting; offer `train_users`.
4. Age in days relative to Dec 31 of the dataset year; null `created_at` → 730 days.
5. Digits in screen name are counted on the unstripped string (L120-121); only matters with the
   TwiBot-20 trailing space (no digits there).
6. TwiBot-20 `"True "`/`"False "` strings must become ints; the benchmark handles them only for
   TwiBot-20 (L58-59).
7. Cresci-2015 benchmark number (6.1) should not be used as a target.

## 10. Open questions / unverified

- Root cause of the Cresci-2015 artefact (6.1) **[U]**.
- Node count per tree / model memory **[U]**; measure on first fit.
- Std convention of the benchmark (ddof) **[U]**.

## 11. Acceptance check

Benchmark mode, `extscore: none`, 5 seeds:
- TwiBot-22 official split: accuracy 75.1 ± 1.0, bot F1 36.6 ± 2.0.
- TwiBot-20 official split: accuracy 81.6 ± 1.5, bot F1 84.9 ± 1.5.
- Cresci-2015 from raw CSVs, 7:2:1 random split: accuracy ≥ 95 (expected ~97-98; NOT 77.1).
