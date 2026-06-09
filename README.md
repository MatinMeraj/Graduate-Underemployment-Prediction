# Graduate Underemployment Prediction

Predicting whether Canadian post-secondary graduates are **overqualified** for their
current jobs, using survey data from the **National Graduates Survey (NGS), Class of
2020** (Statistics Canada). Binary classification, evaluated with **ROC-AUC**.

> This was a 4-person machine-learning challenge. This fork documents my own
> contribution and the analysis I can personally defend. **My role:** data
> preprocessing / feature engineering and the final **XGBoost** model (tuning and
> evaluation). The Random Forest and Logistic Regression baselines were teammates'
> work. Original team repo:
> [AMO-iLLi/Graduate-Underemployment-Prediction](https://github.com/AMO-iLLi/Graduate-Underemployment-Prediction).

---

## TL;DR

- Built and tuned an **XGBoost** classifier reaching **0.70 ROC-AUC** on a held-out
  validation set (consistent with ~0.69 five-fold CV).
- **Audited the target for leakage:** the label is defined as
  `overqualified = (credential level > job requirement level)`, so credential level is
  *partly endogenous* to the target. I quantified this by retraining without it — AUC
  fell only **0.007** (0.704 → 0.696), confirming performance rested on genuine signal,
  not the mechanical relationship.
- Evaluated **beyond accuracy** (precision, recall, F1, confusion matrix, ROC) because
  the classes are imbalanced (~38% positive) and accuracy is misleading here.

---

## Problem

Underemployment occurs when a graduate's credential exceeds the qualification their job
requires. Using demographic, educational, and financial variables collected three years
after graduation, the task is to predict:

- `overqualified = 1` → overqualified / underemployed
- `overqualified = 0` → appropriately matched or underqualified

The target is constructed as `CERTLEVP > JOBQLEVP` (credential level exceeds job
requirement level). **This definition matters** — see the leakage audit below.

---

## Data

National Graduates Survey (NGS) Class of 2020 PUMF, distributed for the challenge.

- **7,709** training rows, **2,509** test rows (test labels withheld — competition format)
- **23** predictors: survey-coded categoricals (credential level, field of study, prior
  education, student loans, demographics, parental education, pre-program activity)
- **Class balance:** ~38% positive (overqualified) — mild imbalance, ~1.6:1
- Missing values present throughout; high-cardinality field-of-study variable

---

## Method

### Preprocessing / feature engineering (my work)
- Treated every predictor as a categorical factor (survey codes are not continuous).
- Encoded missing values as an explicit `"Missing"` level rather than dropping rows, so
  "did not answer" is treated as potentially informative.
- One-hot encoded via sparse matrices, aligning train/test columns so both share an
  identical feature space.

### Model (my work)
- **XGBoost** (`binary:logistic`, `eval_metric = auc`).
- **5-fold cross-validation** (`xgb.cv`) with **early stopping** to choose the number of
  boosting rounds without overfitting.
- `scale_pos_weight` set to gently up-weight the positive class given the mild imbalance.
- Final model trained on all training rows for the competition submission; a **separate
  model trained on an 80% split** and evaluated on the held-out 20% for the honest
  metrics reported here.

### Why a held-out split?
The competition test set has no labels, so it can't produce a confusion matrix or
precision/recall. All metrics below come from a **stratified 80/20 split of the labeled
training data** — trained on 80%, evaluated on the 20% the model never saw. Every number
is reproducible from `final_submission.R`.

---

## Results

All figures are on the held-out validation set (20% never seen in training).

| Metric | Value |
|---|---|
| **ROC-AUC** | **0.704** |
| Accuracy | 0.688 |
| Precision | 0.606 |
| Recall | 0.538 |
| F1 | 0.570 |

5-fold cross-validation AUC was **~0.69**, so the held-out 0.704 is consistent (a single
split is slightly noisier than CV averaging). For reference, the "always predict matched"
baseline accuracy is **61.5%** — which is exactly why accuracy alone is a misleading
metric for this problem.

**ROC curve (held-out):**

![ROC curve](roc_curve.png)

**Confusion matrix @ threshold 0.5:**

|  | predicted matched | predicted overqualified |
|---|---|---|
| **actually matched** | 742 | 207 |
| **actually overqualified** | 274 | 319 |

At the default 0.5 threshold the model is reasonably balanced, catching ~54% of truly
overqualified graduates (recall) while being right ~61% of the time when it flags someone
(precision). The operating threshold is a lever: lowering it would raise recall at the
cost of precision — the right choice depends on whether a missed at-risk graduate or a
false alarm is costlier.

---

## Leakage audit (the most important part)

The target is defined as `overqualified = (CERTLEVP > JOBQLEVP)`. Because **CERTLEVP
(credential level) is one of the two ingredients used to build the label**, it is partly
endogenous — a model "predicting" with it is partly re-deriving the target's definition.

To quantify how much this inflated performance, I retrained the model **without**
CERTLEVP and re-evaluated on the same held-out split:

| Model | Held-out AUC |
|---|---|
| With CERTLEVP | 0.704 |
| Without CERTLEVP | 0.696 |
| **Difference** | **0.007** |

The drop is negligible (~0.007). **Performance did not depend on the endogenous feature**
— the genuine signal from the other variables carries it. Removing all three
education-level variables (CERTLEVP, prior education, highest schooling) dropped AUC by
~0.06, confirming education is informative but far from the whole story.

This is the headline takeaway: the model is honest, and I can show *why*.

---

## What drives the prediction (insights)

Directional rates below are **univariate** (overqualification rate within each category) —
raw associations, not effects adjusted for other variables. Overall baseline rate: 38.4%.

- **Credential level (CERTLEVP):** Master's/Doctorate grads 54.1% overqualified vs.
  Bachelor's 27.1% — a 27-point spread. *Note:* partly mechanical, since the target is
  defined using this variable. Reported for completeness, not as a discovered insight.
- **Student loans (STULOANS):** graduates who received government loans were *less* likely
  to be overqualified (34.6% vs. 42.0%). This association is **independent of the target
  definition**, making it the cleanest genuine signal — and a counterintuitive one worth
  investigating further.
- **Field of study (PGMCIPAP):** ~16-point spread in overqualification rate across fields
  (highest ~43%, lowest ~27%). [Field-name labels to be added from the CIP-2021 lookup.]
- **Gender (GENDER2):** minor — Male 41.5% vs. Female 36.1% (~5-point gap).

---

## How to run

```r
# Requires: tidyverse, Matrix, xgboost, pROC, forcats
# Place train.csv and test.csv in the working directory, then:
source("final_submission.R")
```

The script preprocesses the data, runs 5-fold CV to select boosting rounds, trains the
final model, writes `submission.csv`, and prints the held-out evaluation (AUC, confusion
matrix, precision/recall/F1), the leakage check, and saves `roc_curve.png`.

---

## Files

- `final_submission.R` — full pipeline: preprocessing, CV-tuned XGBoost, held-out
  evaluation, leakage check
- `RandomForest.qmd`, `LogisticRegression.qmd` — teammates' baseline models
- `train.csv`, `test.csv`, `sample_submission.csv` — challenge data
- `roc_curve.png` — held-out ROC curve

---

## Credits

Team challenge by Ilia Janfeshan, Georgi Kuzhel, Matin Meraj Mohammadi, and Julia
Kristensen. This fork reflects my contribution (preprocessing + XGBoost) and analysis.
Data: Statistics Canada, National Graduates Survey (NGS) Class of 2020 (Cat. no.
81M0011X), Public Use Microdata File. Computations and interpretation are my own and do
not represent Statistics Canada.
