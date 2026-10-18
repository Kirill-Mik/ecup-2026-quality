# ecup-2026-quality

Solution code for E-CUP 2026 by Ozon Tech, task 2 "Quality control"
([competition page](https://ods.ai/tracks/e-cup-2026-competitions/competitions/e-cup-2026-quality)).
August 2026.

## Task

Marketplace moderation: for each product card in two categories, dietary
supplements and flammable goods, decide "ban" or "not ban" and write a
50-300 character comment. The answer format is
`<комментарий>{text}<вердикт>{бан|не бан}`.

- Training set: 12,971 cards (7,469 supplements, 25.5% banned; 5,502 flammable
  goods, 96.4% banned).
- Metric: F1 computed per category and averaged.
- Inference runs offline in the organizers' Docker image
  (`odsai/ecup26-quality-baseline:1.0`) with 3 / 20 / 40-minute limits for the
  check, public and private stages.

Findings from the data that shaped the model:

- A card is banned mostly when it lacks the mandatory labelling of its
  category; for supplements, mentioning the word "БАД" alone gives
  P(not ban) = 0.82 against 0.46 without it.
- For flammable goods, fire-safety warnings do not separate the labels. What
  separates them is whether the product contains the substance itself (gas,
  matches, fireworks) or is an accessory to it (a stove without gas).
- 41.6% of the training set are exact duplicates by name and description, with
  96.7% label agreement.

## Pipeline

```
rules -> duplicates -> [write submission] -> 6 TF-IDF heads -> [write]
      -> e5-small (kNN + logistic head + near-duplicate vote)
      -> blend by profile -> rate_guard -> [final write]
```

- `src/rules.py` — hand-written rules from the labelling findings above.
- `src/dupes.py` — exact and name-level duplicate lookup in the training set.
- `src/text_heads.py` — six TF-IDF + linear model variants with different
  tokenization and n-gram ranges.
- `src/emb_text.py` — multilingual-e5-small embeddings: kNN over the training
  set, a logistic head, and a vote from near-duplicates above a similarity
  threshold.
- `src/blend.py` — weighted blend and per-category thresholds.

## Making the submission reliable

The first three submissions scored 0.71, 0.67 and 0.72 on the public
leaderboard while the local out-of-fold estimate was 0.90-0.92. The work after
that went into finding out why.

Delivery problems found and fixed:

- Steps that only ran on the server (GPU heads, OCR, an LLM judge) had never
  been run end to end locally and silently broke predictions. They were
  removed; only CPU-checkable heads with all weights in the archive remain.
- The weight tuner treated an unspecified source weight as 0 and the runtime
  treated it as 1, so the model evaluated locally was not the one submitted.
  The formulas are now shared and covered by a unit test.
- Degradation profiles: weights and thresholds are tuned separately for the
  full set of heads and for a set without e5; the runtime picks the profile by
  which sources were actually computed.
- `rate_guard`: if the share of "not ban" in a category leaves the range seen
  out of fold, the threshold is replaced by a quantile threshold.
- A valid submission file is written seconds after start and overwritten after
  each stage; on failure a fallback with per-category defaults is written.
- `tools/rehearse.py` runs the real `run.py` on the full training set and
  writes a manifest with file hashes; `tools/make_submit.py` refuses to build
  the archive without a fresh manifest.
- `tools/check_image_compat.py` verifies that artifacts trained with
  scikit-learn 1.5.2 load and predict identically under 1.9.0 from the image.

After these fixes the gap remained, so the problem was in validation, not
delivery. `train/diag_neardup.py` showed that 70% of training rows have a
near-duplicate (e5 cosine >= 0.95) in another fold: the blend scored 0.945 on
those rows and 0.828 on the rest. Grouping folds by exact copies did not catch
reworded cards of the same product, and the test set contains different
products. `train/make_folds_neardup.py` rebuilds the folds from connected
components of near-duplicates, so weights and thresholds are tuned on an
estimate that matches unseen products.

## Layout

```
run.py                 inference entry point (cascade above)
metadata.json          Docker image and entry point for the platform
src/                   runtime: rules, duplicates, heads, blend, validation
train/                 training of heads, fold construction, blend tuning
tools/                 rehearsal, archive builder, compatibility check
tests/                 unit tests for format, blend, profiles and guards
```

## Running

```bash
pip install -r requirements-train.txt
python train/train_e5.py                 # first run caches e5 embeddings
python train/make_folds_neardup.py       # folds from near-duplicate components
python train/train_cheap_heads.py
python train/train_text_heads.py --variants t3 t5 t7 t9 t10 t12
python train/train_e5.py
python train/tune_blend.py
python tools/rehearse.py
python tools/make_submit.py
python -m pytest tests -q
```

The training data (`data/data.csv`) and trained artifacts are not included:
the data belongs to the organizers, and the artifacts take about 500 MB.
Code comments are in Russian.
