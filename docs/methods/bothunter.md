# BotHunter (Beskow & Carley 2018) — implementation spec

Method id: `bothunter`. Family: feature-based (profile metadata; Tier-1 "account + last status"),
random forest. Graph: no. Text: no in the benchmark version (profile strings only; the paper's
last-status features are zeroed). Temporal: no (account age only). Encoders: none. GPU: none.

## 1. Summary

Bot-hunter is a tiered bot-detection framework; the paper builds only the Tier-1 model: 17 features
from the user object plus the account's most recent status, fed to a random forest (AUC 0.994 on the
authors' event-annotated data). The TwiBot-22 benchmark re-implementation (`src/BotHunter`) computes
the same 17-slot vector but fills 8 slots with constant 0 (source, favourites, and the six
last-status / language features), leaving 9 informative profile features, and trains
`RandomForestClassifier(n_estimators=100)`. We implement the benchmark version (the numbers come
from it) and offer an `extended` feature mode that fills the zeroed slots from data we hold.

## 2. Citation and code

- Paper: David M. Beskow, Kathleen M. Carley. "Bot-hunter: A Tiered Approach to Detecting &
  Characterizing Automated Activity on Twitter." SBP-BRiMS 2018 (late-breaking paper LB_5; no DOI).
  PDF read from the CASOS copy via the Wayback Machine
  (`casos.cs.cmu.edu/publications/papers/LB_5.pdf`, snapshot 2024-08-09) **[P-paper]**.
- Official code: none public (no repository found; CASOS tools are not released as open source) **[U]**.
- Benchmark re-implementation: `LuoUndergradXJTU/TwiBot-22`, `src/BotHunter/{preprocess.py,train.py,
  readme.md}`, repo HEAD `54adf9a1dbb6dc238c932d3147a1875def5817d8`; code unchanged since commit
  `484f3fcc3b` (2022-06-04). MIT licence: may read and reuse with attribution; we write our own code.
- Benchmark paper: Feng et al. 2022, arXiv 2206.04564v6, Tables 2 and 8 **[P-paper]**.

## 3. Inputs

### 3.1 Paper feature set (Table 2 of the paper, 17 features) **[P-paper]**

| Group | Features |
|---|---|
| User attributes | screen name length; default profile image?; screen name entropy; has location?; total tweets; source (binned) |
| Network attributes | # friends; # followers; # favorites |
| Content (last status) | is last status retweet?; same language? (user lang == status lang); hashtags in last status; mentions in last status; last status sensitive?; bot reference? ('bot' in name/description) |
| Timing | account age; avg tweets per day |

Entropy: Shannon, base 2, over normalised character counts of the screen name (paper Eq. 1).

### 3.2 Benchmark feature vector (17 columns, this order; `preprocess.py` L110-134) **[P-code]**

| # | Benchmark computation | Paper feature |
|---|---|---|
| 0 | `len(username.strip())` (username = screen name); null → 0 | screen name length |
| 1 | `int('default_profile_normal' not in profile_image_url)` (1 = custom image); null → 0 | default profile image? (inverted; RF-invariant) |
| 2 | Shannon entropy of `username.strip()` with NATURAL log (`math.log`), 0 for empty | screen name entropy (log2; constant factor, RF-invariant) |
| 3 | `int(len(location.strip()) > 0)`; null → 0 | has location? |
| 4 | `public_metrics.tweet_count`; null → 0 | total tweets |
| 5 | constant 0 | source (binned) |
| 6 | `public_metrics.following_count` | # friends |
| 7 | `public_metrics.followers_count` | # followers |
| 8 | constant 0 | # favorites |
| 9 | constant 0 | last status retweet? |
| 10 | constant 0 | same language? |
| 11 | constant 0 | hashtags last status |
| 12 | constant 0 | mentions last status |
| 13 | constant 0 | last status sensitive? |
| 14 | `int(any('bot' in s for s in (username, description) if s is not None))`, case-SENSITIVE `str.find` | bot reference (paper: name/description) |
| 15 | account age in days: `(datetime(Y,12,31) - created_at_naive).days`, Y = dataset year (TwiBot-22 → 2022, TwiBot-20 → 2020, cresci-2015 → 2015, ...); null `created_at` → 730; tz dropped | account age |
| 16 | col4 / col15 | avg tweets per day |

Parsing (`calc_age`, L80-101): TwiBot-22 `'%Y-%m-%d %H:%M:%S%z'`; TwiBot-20/Cresci
`'%a %b %d %H:%M:%S %z %Y'` after `.strip()`; strings containing `L` are epoch-ms (`int(s.replace('000L',''))`).
Division by zero would crash if age == 0 (cannot happen with Dec-31 reference and pre-collection
accounts; guard anyway: age := max(age, 1) in our code, log any hits).

Keep the 8 constant-zero columns in `benchmark` mode: with `max_features='sqrt'`, sklearn draws
int(sqrt(17)) = 4 candidate features per split, and constant features consume draws (the splitter
keeps drawing until it finds a non-constant one), so removing them changes the forest's randomness
(int(sqrt(9)) = 3) **[P-code sklearn splitter behaviour; effect size U]**.

### 3.3 `extended` mode (optional, not in benchmark)

Fill zero slots where data exist: 8 = `favourites_count` (TwiBot-20 only; TwiBot-22 v2 user objects
have no favourites; our processed jsonl stores 0); 9/11/12 from the NEWEST stored tweet: retweet =
text starts with `RT @` (TwiBot-20) or `referenced_tweets.type == retweeted` (TwiBot-22 raw), hashtag
count, mention count (TwiBot-22: entities; TwiBot-20: regex `#\w+`, `@\w+`); 10 = user `lang` ==
tweet `lang` (TwiBot-22 tweets have `lang`, users do not → leave 0; TwiBot-20 user `lang` is mostly
"None" → leave 0); 13 = `possibly_sensitive` (TwiBot-22 tweet field); 5 = tweet `source` binned
(TwiBot-22 raw tweet `source`, if present, top-k sources one-hot collapsed to an integer code)
**[U: availability of `source` in the TwiBot-22 tweet dump]**. Extended mode is an ablation only.

### 3.4 What we need from the shared layer

- `users.parquet`: `idx, user_id, split, label` (labelled users only).
- `profile.parquet`: `screen_name`, `description`, `location`, `profile_image_url`, `created_at`,
  `statuses_count`, `following_count`, `followers_count` (+ `favourites_count` for extended). TwiBot-20
  raw values are strings with trailing spaces and literal `"None "`; the layer must strip, map
  "None"/"" to null and cast counts. NOTE: the benchmark's TwiBot-20 file kept `"True "`/`"False "`
  strings (the code special-cases them, L73-74), so it may also have kept `"None "` locations, which
  would make column 3 = 1 for every user **[U]**; clean semantics (null → 0) by default,
  `bothunter.tw20_literal_none_location: true` to emulate.
- Extended mode only: newest tweet per user with `is_retweet`, hashtag/mention counts,
  `possibly_sensitive`, `lang`, `source` (TwiBot-22 from raw `tweet_*.json`, streamed).
- Reference years: constant per dataset (`bothunter.reference_date`, default Dec 31 of the dataset year).

## 4. Architecture

`RandomForestClassifier(n_estimators=100)` with sklearn defaults: gini, `max_depth=None`,
`min_samples_split=2`, `min_samples_leaf=1`, `max_features='sqrt'`, `bootstrap=True`,
`class_weight=None`, no `random_state` (`train.py` L54) **[P-code]**. No scaling. Hard `predict`.
Paper: RF "with tuning", parameters not reported **[P-paper]**.

## 5. Training

- Features are computed for every user node; train/val/test indices from `split.csv` (`train`,
  `val`/`valid`, `test`); users missing from the feature index are skipped silently (L37-38).
- Fit on train; val predictions are printed but unused for selection; report test.
- Seeds: ours `random_state ∈ {0..4}`, `n_jobs=8`. Benchmark unseeded, five runs.

## 6. Evaluation and published numbers

Accuracy and bot-class F1 from benchmark Tables 2 and 8; precision/recall from `src/BotHunter/readme.md`
(mean ± std, 5 runs) **[P-paper, P-code readme]**.

| Dataset | Acc | Prec | Rec | F1 (bot) |
|---|---|---|---|---|
| TwiBot-22 | 72.8 ±0.0 (readme 72.79 ±0.02) | 68.09 ±0.36 | 14.07 ±0.12 | 23.5 ±0.1 |
| TwiBot-20 | 75.2 ±0.4 | 72.77 ±0.25 | 86.75 ±0.46 | 79.1 ±0.4 |
| Cresci-2015 | 96.5 ±1.2 | 98.55 (earlier readme 97.90) | 91.48 (earlier 96.45) | 97.2 ±1.0 |
| Cresci-2017 | 88.1 ±0.2 | 98.65 | 85.40 | 91.6 ±0.1 |
| G-17 / M-18 / C-S-18 / C-R-19 / B-F-19 | 76.4 / 99.3 / 81.2 / 81.5 / 74.7 | | | 69.2 / 99.6 / 82.2 / 82.9 / 49.6 |

Readme inconsistency: the current Cresci-2015 P/R (98.55/91.48) imply F1 ≈ 94.9, not the listed 97.2;
the readme revision of 2022-08-06 (commit `944e7281ea`) replaced P/R/std but kept acc/F1. The earlier
P/R (97.90/96.45) are consistent with F1 97.2 **[P-code history]**. Trust acc/F1.

Benchmark Table 12 (trained on TwiBot-22 train+val, tested on 1,000 expert labels): acc 91.44,
F1 40.39, P 78.28, R 27.24 **[P-paper]**.

Original paper (own DFR-Lab event data, 70k humans, ~19k bots): RF accuracy 0.981, AUC 0.994,
precision 0.996, recall 0.966 **[P-paper]**.

## 7. Scale plan

Memory rule (mandatory): run every job via `tools/memrun` (MemoryMax 10G, no swap, lock); target peak
RSS ≤ 8 GB; never load full TwiBot-22 tweet/edge files or a full feature table; stream/chunk/memmap.

| Step | TwiBot-22 | Peak host RSS | VRAM | Wall |
|---|---|---|---|---|
| Features (benchmark mode) | read 8 profile columns for 1M users from `profile.parquet` (column projection, ~150 MB of strings) → X 1M × 17 × 8 B = 136 MB | < 1 GB | 0 | 1-2 min |
| Features (extended) | newest tweet per user from the store (1M records) | < 1.5 GB | 0 | +5 min |
| RF fit per seed | 700k × 17 float32 = 48 MB; 100 trees × ~300k nodes [U] × ~80 B = 2.4 GB | ≈ 3 GB | 0 | n_jobs=8: 1-3 min |
| Predict / eval | 100k test | ≈ 2.5 GB | 0 | < 1 min |

TwiBot-20: 11,826 labelled users, seconds. Whole method, 5 seeds × 2 datasets: < 30 CPU-min, 0 GPU-h.
Free each forest after scoring.

## 8. ExtScore column

- `extscore: none` reproduces the 17-column baseline exactly.
- `extscore: p`: append `extscore_p` (missing → -1.0) and `extscore_missing` as columns 17 and 18 (width 19;
  int(sqrt(19)) = 4, same per-split draw count as baseline, but each draw now has 2 more candidates).
- Ablations: NaN-native `extscore_p` only (sklearn ≥ 1.4); late fusion (LR on `[logit(rf_proba), z, m]`
  fitted on val); ExtScore-alone reference; `extended` + ExtScore.
- Require ExtScore coverage on train; report coverage per split. Support users unused.
- Report MDI and permutation importance for `extscore_p`.

## 9. Pitfalls

1. Keep the zero columns in benchmark mode (3.2) or the RF randomness differs.
2. Two reference-date conventions across our feature baselines: BotHunter/SGBot use Dec 31 of the
   dataset year; Lee uses 2022-04-20 / 2020-10-01. Do not "harmonise" in faithful mode.
3. `'bot'` match is case-sensitive and uses the screen name, not the display name (paper:
   name/description). `Bot`/`BOT` are not matched. Option `bothunter.bot_ref_casefold`.
4. `get_feature(..., 'profile_image_url')` reads the global loop variable `item` (L72); it is the
   current user, so no bug in practice; but a null URL returns 0 via the `None` guard (counts as
   "default image"), keep that.
5. Entropy of an all-whitespace or empty name: loop never runs → 0 (no division by zero).
6. TwiBot-20 boolean / "None" strings with trailing spaces (3.4).
7. No leakage: purely per-account features; no transductive steps.

## 10. Open questions / unverified

- Exact content of the benchmark's converted TwiBot-20 `node.json` (trailing spaces, "None" strings) **[U]**.
- Whether `source` exists in the TwiBot-22 raw tweet objects (extended mode only) **[U]**.
- Node counts per tree (memory estimate) **[U]**; measure on the first fit.

## 11. Acceptance check

Benchmark mode, `extscore: none`, 5 seeds:
- TwiBot-22 official split: accuracy 72.8 ± 1.0, bot F1 23.5 ± 2.0.
- TwiBot-20 official split: accuracy 75.2 ± 1.5, bot F1 79.1 ± 1.5.
- Cresci-2015 (our hash split, sanity only): accuracy ≥ 94.
