> **Historical document.** This script covers only the original synthetic demo. The current 20-minute speaking script is [`docs/GlucoTwin_20_Minute_Speaking_Script.pdf`](docs/GlucoTwin_20_Minute_Speaking_Script.pdf).

# GlucoTwin AI — Hackathon Demo Script (20+ minutes)

> **Audience:** Healthcare AI hackathon judges, clinicians, and technical evaluators.
> **Goal:** Demonstrate a defensible, end-to-end AI prototype for glucose spike prediction.

---

## Before You Start (Setup Checklist)

```bash
# Terminal 1 — Backend
cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend
cd frontend && npm run dev

# Browser: http://localhost:5173
```

Confirm:
- [ ] Backend health: `curl http://localhost:8000/api/health` returns `"status": "ok"`
- [ ] Frontend loads the welcome screen
- [ ] 120 patients visible in sidebar

---

## Part 1 — Problem & Context (3 min)

**Script:**
> "Diabetes affects 537 million adults globally. Continuous Glucose Monitors generate readings every 10 minutes — that's 144 data points per day per patient. The challenge isn't collecting data; it's predicting what will happen next."
>
> "GlucoTwin AI answers one specific question: will this patient's glucose exceed 180 mg/dL — the ADA postprandial threshold — in the next two hours? If we can answer that reliably, a clinician can intervene proactively — a dietary adjustment, a medication reminder — before the spike occurs."
>
> "Every patient in this system has a Digital Twin: a live, AI-powered model of their physiology, built from their EHR history, their CGM trace, and their wearable sensor data."

**Show:** The welcome screen with 120 virtual patients, risk stratification, and the dashboard stats.

**Highlight:**
- High-risk count is live from the backend — not hardcoded
- "SYNTHETIC DATA — EXPERIMENTAL PROTOTYPE" label is prominent

---

## Part 2 — Overview Dashboard (3 min)

**Script:**
> "The overview shows population-level intelligence. 37 high-risk patients right now. Average glucose across the cohort is 136 mg/dL. These numbers come directly from the FastAPI backend — every refresh recalculates from the actual data."

**Show:**
1. Stats row: total patients, high-risk count, average glucose
2. Recent alerts section (patient IDs with high risk_level)
3. Point out the data-quality indicator

**Key point:** "None of this is hardcoded. Disconnect the backend and the dashboard shows an error, not fake data."

---

## Part 3 — Select a Virtual Patient (2 min)

**Script:**
> "Let me pick a high-risk patient. P0010 has Type 2 Diabetes, HbA1c of 9.1, and their glucose has been trending upward for the last hour."

**Action:** Click patient in sidebar → wait for digital twin to load.

**Show:**
- Patient demographics: age, BMI, diabetes type
- Diagnoses list (ICD codes)
- Lab results: HbA1c, fasting glucose, cholesterol, creatinine
- "Last updated" timestamp is live

---

## Part 4 — Digital Twin: Sensor Data (4 min)

**Script:**
> "The Digital Twin fuses static EHR data with dynamic sensor streams. We're looking at 7 days of continuous CGM and wearable data — 1,008 readings per patient at 10-minute intervals."

**Show the charts:**

### A. Glucose time series
> "The blue line is the CGM trace. You can see meal-related spikes — the patient has a higher variance than a non-diabetic. The dashed line at 180 mg/dL is the spike threshold."

**Highlight:** x-axis shows dates (not just indices), y-axis labeled in mg/dL.

### B. Heart rate trend
> "Heart rate increases during activity and meals — we use this as a proxy for metabolic demand."

### C. Activity / steps
> "Step count and activity intensity show the circadian pattern. This patient is sedentary at night, moderately active mid-morning."

### D. Sleep
> "Sleep duration per night. Short sleep is associated with impaired glucose regulation — we include it as a feature."

**Key point:** "All charts update automatically when you select a different patient. None of these are static screenshots."

---

## Part 5 — Data Quality & Feature Engineering (3 min)

**Script:**
> "Before we trained anything, we ran a formal data-quality validation pass. Let me show you what we check."

**Show:** Model Performance panel → expand it OR curl the API:
```bash
curl http://localhost:8000/api/model-card | python3 -m json.tool | grep -A5 '"checks"'
```

**Walk through:**
- Missing-value percentages per column
- Duplicate record check (0 found in clean data)
- Plausibility bounds: glucose 30–600 mg/dL, HR 30–220 bpm
- Implausible values replaced with patient-specific median (not zero)

> "We engineered 30 features from the raw time series. The most important design decision: every rolling window uses `shift(1)` before computing — this ensures the feature at timestamp T never sees any data from T+1 or later. We have automated tests that verify this."

---

## Part 6 — Run the Prediction (4 min)

**Script:**
> "Now the main event. Let me click 'Predict Next 2 Hours' for this patient."

**Action:** Click the Predict button → wait for result (~1–2s).

**Show the output:**

### Risk circle
> "The model assigns HIGH risk — a 88% probability that glucose will exceed 180 mg/dL in the next two hours. The risk category has three levels: Low (<30%), Moderate (30–60%), High (>60%)."

### Current vs forecast
> "The current CGM reading is 163 mg/dL — measured. The extrapolated forecast of 189 mg/dL is a linear projection from the rate of change — it is NOT a separate trained regression model. We label it clearly as an estimate."

### Feature importances
> "These are the top factors that drove this prediction, computed using permutation importance on the validation set. HbA1c is most influential — not because it changed just now, but because patients with high long-term glucose control have systematically different risk profiles. The 2-hour glucose trend is second — this patient's glucose rose 22 mg/dL in the last 2 hours."

**Critical framing:**
> "Important: correlation does not imply causation. HbA1c being important to the model doesn't mean it causes the spike. The model has learned a pattern — an association — not a physiological mechanism."

### Explanation text
> "The explanation text is generated from real feature values and real importance scores. It never invents numbers."

---

## Part 7 — Model Evaluation & Honesty (3 min)

**Script:**
> "Every good AI demo must be honest about its limitations. Let me show you the model evaluation."

**Show:** Model Performance panel → expand → scroll through.

**Walk through the metrics table:**

| Model | Accuracy | ROC-AUC | PR-AUC | Recall |
|-------|---------|--------|--------|--------|
| Logistic Regression | 88% | 0.980 | 0.840 | 96% |
| Random Forest | 90% | 0.989 | 0.912 | 95% |
| **HistGradientBoosting** | **89%** | **0.958** | **0.804** | **89%** |

> "We compare three models. We selected HistGradientBoosting by PR-AUC on a validation set — PR-AUC is better than accuracy for imbalanced data because it penalises missing real spikes."

> "We report 89% accuracy, not 95% or 99%. The honest result on a proper patient-aware split is more valuable than an inflated number."

**Confusion matrix:**
> "Of 3,971 actual spikes in the test set, we correctly flag 3,534. We miss 437 — those are false negatives. We also generate false alerts — important to quantify for clinical adoption."

**Leakage audit:**
> "Our automated leakage audit confirms: no future glucose values in the feature matrix, no patient boundary contamination, no test-set tuning."

---

## Part 8 — Architecture & Reproducibility (2 min)

**Script:**
> "The architecture is three components: a FastAPI backend with a trained HistGradientBoosting pipeline, a React frontend served by Vite, and a shared feature-engineering module used identically at training time and inference time."

> "Everything is reproducible. Run `python -m app.data_generator --force` to regenerate the synthetic data with seed 42. Run `python -m app.train_model` to retrain. The model card JSON records every hyperparameter, split strategy, and metric."

**Show the test suite:**
```bash
cd backend && python -m pytest tests/ -v
# 52 tests: 13 API + 39 feature/validation unit tests
```

> "52 automated tests cover API contracts, feature leakage prevention, temporal boundary correctness, and data validation integrity."

---

## Part 9 — Limitations & Honest Assessment (2 min)

**Script (read exactly):**
> "Let me be explicit about what this system cannot do."
>
> "All 120 patients and 120,960 readings are synthetic. The 89% accuracy is measured on synthetic data. We cannot claim this model would achieve 89% on real CGM traces — that requires a real clinical validation study."
>
> "The 180 mg/dL threshold is an assumption. Different clinical contexts may require different thresholds."
>
> "This system is not CE marked, not FDA cleared, and must not be used for real patient care."
>
> "What we have demonstrated is a complete, defensible end-to-end engineering prototype: data generation, quality validation, leakage-free feature engineering, multi-model comparison, honest evaluation, and a working full-stack application."

---

## Closing (1 min)

> "GlucoTwin AI shows that a small engineering team can build a rigorous, reproducible ML pipeline for healthcare prediction — with proper evaluation, honest communication of limitations, and a polished clinical-grade interface. The architecture is production-ready to be retrained on real anonymised CGM data with appropriate ethics approval."

**Q&A ready points:**
- Why HistGradientBoosting? → Best PR-AUC on validation; handles imbalance; fast inference
- Why not 92%? → Honest 7-day generalisation test is harder; we don't inflate metrics
- How would you deploy? → Docker + model registry + Kubernetes; re-train quarterly
- How would you validate clinically? → IRB approval, real CGM cohort, prospective study

---

*Generated by GlucoTwin AI — Hackathon Submission 2024*
