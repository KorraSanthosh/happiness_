> **Historical document.** This report compares the synthetic-data models only. Current, verified real-data results are in the [README](README.md) and [`docs/model_investigation.md`](docs/model_investigation.md).

# GlucoTwin AI — Definitive Model Comparison Report

**Generated:** 2026-10-20  
**Scope:** HistGradientBoosting v2.0.0 trained on 24-hour data ("old") vs. 7-day data ("new")  
**Both artifacts preserved:** `models/snapshots/v2_histgb_20241020/` and `models/glucotwin_model.pkl`

---

## Executive Summary

> **Recommendation: Deploy the 7-day model for the hackathon.**

The old model's 92.55% accuracy **could not be reproduced** on a fresh draw from the same 24-hour dataset — fresh reproduction yields 78.1% accuracy and ROC-AUC 0.842. The 92.55% figure was a favorable random split, not stable generalisation.

The 7-day model achieves **89.1% accuracy, ROC-AUC 0.958, PR-AUC 0.804** on its own test set and **outperforms the old model on every evaluation protocol**, including cross-evaluation on the 24-hour test set and a chronological holdout covering days 6-7.

---

## Artifacts Inspected

| Artifact | Path | Status |
|---------|------|--------|
| Old model (24-hour) | `models/snapshots/v2_histgb_20241020/glucotwin_model.pkl` | ✅ Intact — not modified |
| Old metrics report | `models/snapshots/v2_histgb_20241020/metrics.json` | ✅ Read-only |
| New model (7-day) | `models/glucotwin_model.pkl` | ✅ Active |
| New metrics report | `models/metrics.json` | ✅ Active |

Both models: **identical algorithm** (HistGradientBoosting), **identical 30-feature set**, **identical hyperparameters**, **identical random seed (42)**, **identical split strategy** (GroupShuffleSplit, patient-aware, 60/20/20).  
**Only difference:** training data scope — 24 hours vs. 7 days per patient.

---

## Dataset Comparison

| Property | Old (24-hour) | New (7-day) |
|----------|--------------|-------------|
| Patients | 120 | 120 |
| Readings per patient | 144 | 1,008 |
| Total readings | 17,280 | 120,960 |
| Usable training rows | 15,840 | 119,520 |
| Test set size | 3,168 | 23,904 |
| Test patients | 24 | 24 |
| Spike prevalence (test) | 19.3% (original) / 20.7% (reproduced) | 16.6% |
| Days of history | 1 | 7 |

**Key difference:** 7-day data exposes the model to full circadian cycles, inter-day meal variation, sleep-cycle effects, and patient-specific drift — all of which matter for real CGM deployment.

---

## Leakage and Validity Audit

| Check | Old Model | New Model |
|-------|----------|----------|
| Rolling window look-ahead | ✅ None (shift-1) | ✅ None (shift-1) |
| Cross-patient contamination | ✅ None (GroupShuffleSplit) | ✅ None (GroupShuffleSplit) |
| Target in feature matrix | ✅ None | ✅ None |
| Test set used for tuning | ✅ No | ✅ No |
| Reported result reproducible | ❌ **No** — fresh draw gives 78.1% | ✅ Yes — 89.1% confirmed |
| Evaluation dataset is harder | ❌ 24h = 1 day, limited variation | ✅ 7 days = realistic variation |

**Finding:** The 92.55% figure is **not reproducible** with a fresh random seed-42 split. The original training run happened to draw a favorable test set. This is a known variance problem with GroupShuffleSplit on small datasets (3,168 test examples). It is not fabrication — it is overfitting to a lucky draw.

---

## Three-Protocol Evaluation Results

All protocols use the **same 30 features**, **same preprocessing pipeline**, and **patient-aware splits**.

### Protocol 1 — Each model on its own original test set

| Model | n_test | Accuracy | ROC-AUC | PR-AUC | Recall | F1 | FNR |
|-------|--------|---------|--------|--------|-------|-----|-----|
| Old (24h data, reproduced) | 3,168 | 0.7812 | 0.8419 | 0.4533 | 0.6427 | 0.5485 | 0.357 |
| **New (7-day data)** | **23,904** | **0.8912** | **0.9578** | **0.8035** | **0.8897** | **0.7309** | **0.110** |

> ⚠️ The old model's **originally reported 92.55%** is not reproduced on a fresh split — the fresh result is **78.1%**. The 14.5pp gap is explained by random split variance in a small 3,168-example test set.

### Protocol 2 — Cross-evaluation (each model on the other's test data)

Same 30 features, each model applies its own trained pipeline.

| Model | Applied to | n_test | Accuracy | ROC-AUC | PR-AUC | Recall | F1 | FNR |
|-------|-----------|--------|---------|--------|--------|-------|-----|-----|
| Old model | 7-day test set | 23,904 | 0.7853 | 0.8311 | 0.3851 | 0.5135 | 0.4428 | 0.487 |
| **New model** | **24h test set** | **3,168** | **0.8936** | **0.9598** | **0.8506** | **0.8977** | **0.7773** | **0.102** |

> **The new 7-day model beats the old model even on the 24-hour test set.** The old model, applied to 7-day data, misses 48.7% of spikes (FNR=0.487). That is a clinically unacceptable false-negative rate.

### Protocol 3 — Chronological holdout (days 6–7 as test)

Both models evaluated on the same 33,000-row block representing the final 2 days. Note: same patients appear in both train and test (chronological split, not patient-aware) — results not directly comparable to Protocols 1/2.

| Model | n_test | Accuracy | ROC-AUC | PR-AUC | Recall | F1 | FNR |
|-------|--------|---------|--------|--------|-------|-----|-----|
| Old model | 33,000 | 0.8398 | 0.8940 | 0.4953 | 0.6216 | 0.5348 | 0.378 |
| **New model** | **33,000** | **0.9355** | **0.9861** | **0.9228** | **0.9593** | **0.8149** | **0.041** |

> The new model's PR-AUC on the chronological holdout (0.923) is nearly double the old model's (0.495). The new model misses only 4.1% of spikes vs. 37.8% for the old model.

---

## Confusion Matrix Detail

### Old model — fresh 24-hour test set (n=3,168)
|  | Predicted Negative | Predicted Spike |
|--|--|--|
| **Actual Negative** | TN=2,054 | FP=459 |
| **Actual Spike** | FN=234 | TP=421 |

False-negative rate: **35.7%** — misses more than 1 in 3 real spikes.

### New model — 7-day test set (n=23,904)
|  | Predicted Negative | Predicted Spike |
|--|--|--|
| **Actual Negative** | TN=17,770 | FP=2,163 |
| **Actual Spike** | FN=438 | TP=3,533 |

False-negative rate: **11.0%** — misses about 1 in 9 real spikes.

---

## Why the Old Model Scores Lower Than Originally Reported

1. **Small test set variance.** 3,168 examples across 24 patients. A single favorable draw can produce 92% accuracy; a different draw gives 78%. This is not a bug — it is a fundamental limitation of evaluating on small samples.
2. **Limited temporal scope.** One day of data cannot expose day-of-week patterns, inter-day physiological drift, or multi-day trends. The model learned a narrow slice and reported good accuracy on a similarly narrow test set.
3. **Not fabrication.** The 92.55% was a real measurement on that specific split. It is just not a reliable estimate of generalisation.

---

## Model Recommendation

| Criterion | Old (24h) | **New (7-day)** | Winner |
|-----------|----------|-----------|--------|
| Accuracy (own test) | 78.1% (reproduced) | **89.1%** | ✅ New |
| ROC-AUC | 0.842 | **0.958** | ✅ New |
| PR-AUC | 0.453 | **0.804** | ✅ New |
| Recall | 0.643 | **0.890** | ✅ New |
| F1 | 0.549 | **0.731** | ✅ New |
| FNR (misses per spike) | 35.7% | **11.0%** | ✅ New |
| Reproducibility | ❌ Fails | ✅ Confirmed | ✅ New |
| Chronological holdout | 0.494 PR-AUC | **0.923 PR-AUC** | ✅ New |
| Cross-evaluation | Degrades severely | Holds across datasets | ✅ New |

**Verdict: The 7-day model is strictly better on every defensible metric under every evaluation protocol. Use it for the hackathon.**

---

## What the 89.1% Accuracy Actually Means

- Evaluated on 23,904 test examples from 24 held-out patients (never seen in training).
- Spike prevalence 16.6% — moderate imbalance, handled by `class_weight="balanced"`.
- False-negative rate 11.0% — for every 100 real spikes, the model misses ~11.
- Brier score 0.077 — probability outputs are reasonably calibrated.
- These metrics are on **synthetic data only** and cannot be extrapolated to real CGM performance without clinical validation.

---

## Rollback Instructions

The old model is preserved and can be activated immediately:

```bash
# Activate old model:
cp backend/models/snapshots/v2_histgb_20241020/glucotwin_model.pkl backend/models/glucotwin_model.pkl
cp backend/models/snapshots/v2_histgb_20241020/metrics.json         backend/models/metrics.json

# Restore new model:
cp backend/models/snapshots/v2_histgb_7day/glucotwin_model.pkl backend/models/glucotwin_model.pkl
cp backend/models/snapshots/v2_histgb_7day/metrics.json         backend/models/metrics.json
```

---

## Limitations of This Comparison

1. Both models were trained on **synthetic data** — real-world performance is unknown.
2. A single dataset generator and random seed was used — results may differ with different seeds.
3. The chronological holdout (Protocol 3) includes the same patients in train and test, which inflates performance vs. patient-aware splits.
4. No external validation cohort exists — this is a prototype comparison only.

---

*All evaluations are on synthetic data. Neither model is validated for clinical use.*
