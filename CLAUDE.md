# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

State recorded here is as of **2026-10-23** (110/110 tests passing). `README.md` was rewritten for the Digital Twin Challenge submission (v7) and is current. `DEMO_SCRIPT.md` and `MODEL_COMPARISON.md` still describe only the synthetic-data pipeline and are stale.

## What this is

**GlucoTwin AI** (repo folder `project_happiness`): predicts the probability that a patient's CGM glucose exceeds **180 mg/dL within the next 2 hours**, and presents a per-patient "Digital Twin" (EHR + CGM/wearable streams). FastAPI backend + React/Vite frontend. The app serves two modes: the original **synthetic** demo (`/api/*`) and the **real-data v7 twin** (`/api/v7/*`, open CC BY 4.0 data); both are experimental prototypes, not for clinical use. Keep the "EXPERIMENTAL — synthetic — not for clinical use" disclaimers in API responses (`predictor.py` `experimental_status`) and UI.

## Model-improvement policy (from the user, 2026-10-23)

Do not keep changing algorithms, hyperparameters or weighting schemes to chase accuracy. Algorithm, tuning, early-stopping and weighting variants have all landed within ±0.1 points (see v6 and v8). Improve the model only by investigating **new, valid information**:
- **Data quality:** gaps, noise, sensor limits, label reliability near 180 mg/dL.
- **Additional relevant features:** must be causal, i.e. known at prediction time.
- **Patient-level generalisation:** which people fail, and why.
- **Calibration:** overall, per dataset and per patient.
- **Analysis of missed spikes:** what the false negatives have in common.

Each new feature needs a stated hypothesis before testing, the v7 folds, a paired patient-bootstrap comparison against v7, and an entry in the leakage test.

## Setup and commands

Environment on this machine: macOS, system Python 3.9.6 with deps in the *user* site-packages (no venv; `python3 -I` cannot see them), Node 26 / npm 11, `frontend/node_modules` installed. Git repo (branch `main`, remote `origin` = https://github.com/KorraSanthosh/happiness_, author identity from the global git config). No env vars are read anywhere — `.env.example` documents defaults only, no `.env` needed. `from __future__ import annotations` is used throughout the backend because Python 3.9 cannot evaluate `X | None`; keep it.

```bash
pip install -r backend/requirements.txt     # + `cd frontend && npm install`
```

Backend — run from `backend/` (modules are run with `-m app.<name>`):
```bash
python -m app.data_generator [--force]      # (re)generate backend/data/{patients.json,readings.parquet}
python -m app.train_model                   # synthetic v2 model -> backend/models/ (+ data_quality_report.json)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload   # Swagger: :8000/docs
python -m pytest tests/ -q                                    # all 89 tests, ~15-45 s
python -m pytest tests/test_api.py::test_predict_success -v   # single test
python -m pytest tests/test_features.py -k leakage -v         # by keyword
```
Frontend — run from `frontend/`: `npm run dev` (port 5173; Vite proxies `/api` to `localhost:8000`), `npm run build`, `npm run preview`. No lint/format config for Python or JS, and no frontend tests.

v5 real-data pipeline (OhioT1DM + BrisT1D, from `backend/`; ~25 min for all stages, writes only `models/combined_v5/`):
```bash
python -m app.real_data_eval --stage all     # or one of: ablation | tune | diagnostics | temporal | final
python -m pytest tests/test_real_pipeline.py -v   # stitching, binning, causality (leak) and label tests
```

v6 (87 patients) and v7 (the submission model, 187 people + Ohio external), each writing only its own `models/combined_v6|v7/`; frames are cached in `frame.parquet`, `--rebuild` to refresh:
```bash
python -m app.real_data_v6 --stage all   # or: features | data | lodo | learning | personal | final | meals
python -m app.real_data_v7 --stage all   # ~1 h: cv (3 horizons) | ablation | final; --rebuild to refresh the frame
python -m app.build_twin_demo            # dashboard demo pack + demo models (needs the v7 results)
```

v9 improvement round (writes `models/combined_v9/`; run through the isolated launcher, ~45 min for cv..report):
```bash
python3 -I scripts/run_isolated.py -m:app.error_analysis_v9                 # what the v7 errors have in common
python3 -I scripts/run_isolated.py -m:app.real_data_v9 --stage all          # build | split | cv | policy | final | labels | report
python3 -I scripts/run_isolated.py -m:app.build_twin_v9 [--examples-only]   # served v9 2-h model + demo extras
python3 -I scripts/run_isolated.py -m:app.verify_results --update-artifacts # 297 checks; rewrites cards/summary/report
```
Downloaded datasets live in `external_data/` and are untrusted (they include `.py`/`.ipynb` files that must never be run). Read them only with Python in isolated mode (`python3 -I`) using a launcher that adds `backend/` and `site.getusersitepackages()` to `sys.path`. Plain `-I` hides the user site-packages.

Legacy real-data trainers (superseded by the above; **not run** because they overwrite the repo-root `models/`):
```bash
python -m app.ohio_train_model     # v3.1, OhioT1DM only
python -m app.combined_train       # v4.0, OhioT1DM + BrisT1D (+SMOTE, +XGBoost if installed)
```
`combined_train.py` needs `xgboost` and `imbalanced-learn` (optional imports, absent from `requirements.txt`; both are installed here). `docs/generate_architecture.py` needs `matplotlib` (installed; pinned in `requirements.txt` with `openpyxl` and `xlrd`).

## Architecture: the synthetic pipeline and the real-data pipelines (v2 → v7)

**1. Synthetic v2 — the one the API serves.**
`data_generator.py` (seed 42; 120 patients x 1,008 readings @10 min; also precomputes `spike_target`) -> `data_validation.py` (QC + patient-median imputation) -> `features.py` (30 features) -> `train_model.py` (LR / RF / HistGB, `GroupShuffleSplit` 60/20/20 by patient, select by validation PR-AUC) -> `backend/models/glucotwin_model.pkl`. Artifact is a **joblib dict** `{pipeline, feature_cols, version, model_name, feature_importances}`; `predictor.py` loads and caches it. Served model: HistGradientBoosting, test ROC-AUC 0.958 / PR-AUC 0.804.

**2. Real-data experiments — not wired into the API.**
`ohio_data_loader.py` (12 T1D patients) and `brist1d_loader.py` (converts the wide 6-h-lookback Kaggle format to long) -> `ohio_features.py` (**33 different features**: carbs, bolus, IOB proxy, GSR; 5-min steps; horizon 24) -> `ohio_train_model.py` / `combined_train.py` -> **repo-root `models/`** as a plain **pickle** `{model, threshold}`. Nothing in `app/main.py`, `predictor.py` or `services.py` reads it, and `ohio_features.py` has no inference-time extractor, so serving it would need new predictor + inference code. On disk now: v3.1 Random Forest, threshold 0.4, test ROC-AUC 0.932 / PR-AUC 0.961 on Ohio weeks 7-8 (`models/metrics.json`; `combined_train` v4.0 output is not what's saved). Raw data must sit at repo root: `ohiot1dm_raw/ohiot1dm-glucose-dataset-main/Ohio Data/Ohio{2018,2020}_processed/{train,test}/*.csv` and `brist1d_raw/{train,test}.csv`. The root also holds duplicate zips and a `brist1d (1)/` folder the loaders ignore.

The two pipelines also differ in target: synthetic `spike_target` = glucose **at exactly t+12 steps** > 180; Ohio = **max over t+1..t+24 steps** > 180. Both mark the last horizon rows `-1` (excluded from training).

Caveats in the real-data scripts: **scale** — `brist1d_raw/train.csv` has 177,025 wide rows x 72 steps, so `load_brist1d` builds ~12.7M long rows, and `ohio_features.iob_proxy` is a per-row Python loop; expect very high memory and a very long `combined_train` run (inferred from the code, never timed; the saved root model is v3.1, so it may not have completed). BrisT1D `timestamp` is a fabricated index (`row*72+step`), so `hour_of_day`/`time_of_day_*` are meaningless for `brist_*` rows; validation sets are the row-order tail of the sorted frame (not patient-aware; in `combined_train` that tail is entirely BrisT1D, while test is Ohio).

**3. Combined real-data v5 — current real-data work (`real_data_pipeline.py` + `real_data_eval.py`), not wired into the API.**
- **Stitching:** BrisT1D's overlapping 6-hour windows are stitched into one true timeline per patient. There were 0 conflicts in 11.4M writes, and this replaces the old ~12.7M duplicated rows.
- **Common grid:** all sources are put on a **15-minute grid**, because BrisT1D p01/p05/p06 have 15-minute CGM. Features use the last reading per bin; the label uses the bin max over the next 8 bins.
- **No leakage shortcuts:** missing CGM stays NaN (HistGB handles it natively) and nothing is imputed from the future.
- **Features:** 29 base and 66 engineered features, all past-only. Ohio insulin is computed as bolus + basal/12.
- **Evaluation:** 7-fold patient-grouped CV over 21 patients, with nested tuning and threshold selection.
- **Results (from `models/combined_v5/stage_*.json`):**
  - **Nested accuracy:** 0.819 (95% CI 0.802–0.836), ROC-AUC 0.884, recall 0.70.
  - **Baselines:** "glucose > 180 now" scores 0.797; majority class scores 0.526.
  - **What helped:** adding BrisT1D gives +0.8 to +1.1 points and feature engineering +0.3 points (both significant under a patient bootstrap). The tuner prefers shallow trees, and held-out loss is best at about 100–200 iterations.
  - **Weak spots:** recall when glucose is currently at or below 180 is only 0.27; BrisT1D recall is lower than Ohio's (0.66 vs 0.75); the afternoon is 12 points worse than the night; the source is recoverable from the features (AUC 0.81).
  - **Ohio temporal split:** about 0.84, but patients are seen in training, so this is optimistic and comparable to v3.1's 0.846.
- **Final model:** `models/combined_v5/model.joblib`, a joblib dict `{model, calibrator, feature_cols, threshold, ...}`.

**4. v6: 87 patients (`external_loaders.py` + `real_data_v6.py`), the current best real-data model, not wired into the API.**
- **New data (all CC BY 4.0; attribute the sources):** HUPA-UCM (24 usable patients), T1D-UOM (17) and AZT1D (25), added to Ohio and BrisT1D.
- **Harmonisation rules** (listed in the `external_loaders.py` docstring):
  - **HUPA glucose and HR** were linearly interpolated (mostly 15-min Libre data stretched to 5 min). Collinear interior points are masked to remove the future leak.
  - **HUPA carbs** mix grams and 10-g portions; the unit is decided per patient by the daily total (< 40 → ×10).
  - **UOM long-acting basal** is spread over 24 h; pump basal is converted as rate/12.
  - **AZT1D basal > 10 U/h** is dropped as corrupt.
  - **Unrecorded modalities** are NaN ("unknown"), not 0.
- **New features:** `ACTIVITY_FEATURES` and `PERSONAL_FEATURES`, both causal (same-time-of-day lags of 1–7 days, 24 h stats). The v5 features stay bit-identical.
- **Folds:** the original 21 patients keep their v5 folds; the new patients are dealt round-robin into the same folds. Training weights are capped per patient (`CAP_ROWS=6000`).
- **Results (`models/combined_v6/stage_*.json`):**
  - **Nested, all 87 patients:** accuracy 0.829 (CI 0.817–0.839), AUC 0.900, recall 0.745 at the tuned threshold 0.40. At threshold 0.5, accuracy is ~0.834 with recall ~0.70.
  - **Original 21:** accuracy ≈ v5 (0.818), but AUC rose from 0.884 to 0.898 and recall from 0.70 to 0.77.
  - **Leave-one-dataset-out:** 0.813–0.850, beating the rule on every dataset.
- **What did not help:**
  - **Steps/kcal features:** +0.
  - **More patients:** the learning curve is flat after ~20 patients (0.831 → 0.833).
  - **Per-patient recalibration on the first 1–3 days:** ≤ +0.1, or worse.
  - **Even perfect knowledge of meals 2 h ahead:** only +0.24. Only 35% of spikes from normal glucose follow a logged meal.
- **Bias:** HUPA probabilities are over-predicted by +3.4 points, the afternoon is 12 points worse than the night, and per-patient accuracy ranges from 0.74 to 1.0.
- **Final model:** `models/combined_v6/model.joblib`, with config `slow` and threshold 0.40.

**5. v7: the Digital Twin Challenge submission (`real_data_v7.py`, `twin_v7.py`, `build_twin_demo.py`, `frontend/.../TwinV7.jsx`); served by the API.**
- **Training data, open licences only:** BrisT1D, HUPA-UCM, T1D-UOM, AZT1D, Shanghai T1DM and Shanghai T2DM = 187 people (87 T1D, 100 T2D). OhioT1DM is under a data-use agreement: it is **never trained on and never committed**, only an external test.
- **Shanghai** brings **real clinical EHR** per recording (`external_loaders.shanghai_ehr`, 31 `EHR_FEATURES`): labs, history, complications, drug classes. Insulin is free text (`_parse_doses`); meals are free text, so carbs are unknown (NaN). Needs `xlrd` for `.xls`.
- **Three horizons** via `y_30m`, `y_60m`, `y` (2 h); each has accuracy-mode and safety-mode thresholds (nested-CV medians).
- **Results (`models/combined_v7/stage_*.json`):**
  - **Accuracy (30 min / 1 h / 2 h):** 94.7 / 91.1 / 84.9%, against 93.7 / 89.7 / 82.5% for the "glucose > 180 now" rule. OhioT1DM external: 94.5 / 90.6 / 83.7%.
  - **Type 2 only:** 96.4 / 93.8 / 89.8%.
  - **Train-vs-test gap:** 0.3 / 0.5 / 1.0 points.
- **Fusion finding (be honest in the pitch):** EHR + CGM is not measurably better than CGM alone (89.7 vs 89.8% on Shanghai). In a patient's first 24 h the EHR adds +0.2 points (not significant). EHR alone: AUC 0.65.
- **Dashboard:** `/api/v7` runs the same feature code live on a raw stream up to a chosen 15-min bin, plus the EHR. **Since v9 the 2-h model is v9** (see section 8); 30 min / 1 h stay v7. It uses the committed demo pack `backend/data/twin_v7` (30 patients, 14 days each, CC BY 4.0 sources only) and `backend/models/twin_v7` (trained WITHOUT those patients). Regenerate with `python -m app.build_twin_demo` (needs `models/combined_v7`). The frontend has a "Real patients · v7" / "Synthetic demo" switch in `App.jsx`.
- **Submission artefacts** (team hackers_, IIT (ISM) Dhanbad, leader Korra Santhosh; other members and the video link are still pending):
  - `README.md`.
  - `docs/GlucoTwin_Presentation.pptx` + `.pdf` (19 slides, built by `docs/presentation/build_deck.js`). Charts are drawn from shapes because Keynote does not import pptxgenjs charts. Export the PDF via Keynote AppleScript and **quit Keynote first**, or it re-exports a stale open copy.
  - `docs/GlucoTwin_Architecture.pdf` + `architecture.pdf|png` (`docs/generate_architecture.py`, which reads `verified_metrics.json`).
  - `docs/GlucoTwin_20_Minute_Speaking_Script.md|pdf` (PDF via `docs/presentation/build_script_pdf.py`, which uses headless Chromium; about 2,680 spoken words ≈ 20 min).
  - `docs/DATA_SOURCES.md`, `docs/VIDEO_PENDING.md`, and `docs/screenshots/` + `docs/dashboard.png` (studio UI). Rebuild the deck with `NODE_PATH=<dir with pptxgenjs react-icons react react-dom sharp> APPLY_THEME=<skill>/scripts/apply_theme.js node docs/presentation/build_deck.js`.
- **Data download:** `backend/scripts/download_open_data.py` (BrisT1D needs a Kaggle account; this machine has no Kaggle credentials).
- **Gitignore:** raw datasets, `external_data/`, zips, the root `models/glucotwin_model.pkl` and `models/combined_v*/*.parquet|*.joblib` are excluded (they contain or derive from OhioT1DM rows).
- **Honest limits:** the Shanghai T2D cohort is Chinese, not Indian, and the demo replay's "example early warning" moments are chosen for illustration (the slider shows misses too).

**6. v8 Type 2 study (`real_data_t2d.py`, results in `models/combined_t2d/results.json`): no improvement; v7 stays the submission model.**
- **Variants tested on the 100 Type 2 people, 2 h:** person-grouped early stopping; Shanghai meal-log carbs estimated from free text (`external_loaders.estimate_meal_carbs`, `FOOD_CARB_PCT`) plus oral-drug dose features (`ORAL_MED_FEATURES`, enabled with `_load_shanghai(..., meals=True)`); Type 2 rows weighted ×3; training on Type 2 only.
- **Result:** all scored 89.7–89.8% vs v7's 89.77%; no difference is significant under a patient bootstrap. Meal and drug data raise recall 73.3 → 75.2–75.7% at lower precision, i.e. a trade-off, not a gain (per-fold thresholds, rescored). Early stopping picks 250–730 trees and the train-vs-test gap is 1.4–1.8 points, so v7 was not over-fitted.
- **Note:** scikit-learn's HGB defaults to `early_stopping="auto"` (a random 10% of rows); the v7 models still built all 500 trees. v8 sets `early_stopping=False` and does person-grouped early stopping itself.
- **Isolated launcher (survives scratchpad resets):** `python3 -I backend/scripts/run_isolated.py -m:app.<module>`.

**7. Error investigation and verification (audited 2026-10-23).**
- **Analysis:** `model_investigation.py` writes `models/investigation/analysis.json` and `per_patient.csv`. `verify_results.py --update-artifacts` independently re-derives every published number, regenerates `docs/model_investigation.md`, and writes the verified block into `models/combined_v7/model_card.json` and the dashboard `summary.json`. It writes `verified_metrics.json` (297 checks including v9) and exits non-zero on any mismatch. **Never hand-edit those numbers.** `tests/test_results_consistency.py` checks the README, the summary, the model card and the report against the committed results; it also recomputes from the OOF parquet when that exists.
- **Bug fixed in the audit:** recall/precision were computed at the median threshold while accuracy used per-fold thresholds. `metrics()` now accepts per-row thresholds. `real_data_v7 --stage rescore` and `real_data_t2d --rescore` recompute from the saved predictions with **no retraining**; model checksums are unchanged. Correct 2-h recall is **68.9%** (not 69.2%) and AUC is **0.913** (not 0.914, a double-rounding error). The v5 and v6 JSON still use the old median-threshold recall/precision (historical, not rescored).
- **Verified figures (2 h, 187 people):**
  - **Accuracy:** 84.9% overall = 90.2% on clear outcomes and 66.8% where the actual peak was within ±10% of 180 (22.8% of moments, 49.9% of errors).
  - **Type 2:** 89.8% = 94.4% / 71.1%.
  - **Caveat:** the ±10% band is an assumption, and the split uses the actual outcome, so it is an error decomposition, not a selectable subset.
- **Serious spikes** (peak ≥ 250 or ≥ 3 consecutive bins > 180):
  - **Per distinct episode:** **36.8%** (3,495 of 9,503) get no warning 30–120 min ahead (person CI 34–40%).
  - **Per overlapping moment:** 23.8%. Don't quote the moment rate as an event rate.
- **Fingerstick (Shanghai only):** 4,010 same-row CGM/CBG pairs from 98 people, timed only to the 15-min grid. MARD 13.9%, median absolute difference 11.0% (CI 9.6–12.5), median bias −7.0% (CI −10.6 to −3.8); Type 1 is worse (MARD 18.4%, 10 people). Meters have their own ±15% error.
- **Calibration** (held-out): ECE 0.15 / 0.34 / 0.38% with person-bootstrap CIs. Per-person figures require ≥ 200 rows.
- **Model beats the rule** for 91.9% of 186 people.

**8. v9: data-driven improvement round (`error_analysis_v9.py`, `real_data_v9.py`, `build_twin_v9.py`); 2-h model served.**
- **Frozen baseline:** `models/baseline_v7/manifest.json` (checksums, 111 features, thresholds, metrics). No v7 file was changed.
- **Protocol:**
  - **Split:** 187 open-data people split once into DEV 150 / TEST 37 (`split.json`, seed 2026, never overwritten).
  - **Selection on DEV only:** feature groups, thresholds and the alert policy, via 5-fold person CV with inner GroupKFold(3) thresholds.
  - **Retention rule:** the AUC CI excludes 0 and accuracy is not significantly worse.
  - **Single TEST run:** TEST, Ohio and a chronological holdout (DEV + first half of each TEST record → second half) are scored once.
  - **Promotion rule:** written in the `real_data_v9` docstring *before* TEST was scored.
  - **Identity check:** the `build` stage checks the v9 frame against v7 (same rows and labels; v7 inputs identical except Shanghai carbs, which `base_inputs()` masks).
- **Candidate feature groups** (all past-only, in the leakage test):
  - **C1** `HISTORY_FEATURES`: 14-day same-time spike rate (labels shifted ≥ 1 day), hours since high, 72-h share above, 72-h onsets/day.
  - **C2** `ORAL_MED_FEATURES` plus unmasked Shanghai meal-text carbs.
  - **C3** `DYNAMICS_FEATURES`: 2-h OLS slope, max 1-h rise, hours to 180 (capped 0–8), 72-h CV, glucose minus 72-h mean.
- **Results:**
  - **DEV:** all three groups were retained; combined: accuracy +0.15 points (CI +0.09 to +0.22), AUC +0.0022.
  - **TEST** (37 people, 147,200 moments): accuracy 85.24 → 85.33 % (diff CI −0.05 to +0.25, n.s.); AUC 0.9154 → 0.9167 (+0.0013, CI +0.0004 to +0.0021).
  - **Ohio and chronological holdout:** Ohio 83.72 → 83.98 %; chronological 85.44 → 85.45 %. Type 2 90.41 → 90.37 %.
  - **Honest reading:** a small ranking gain, not an accuracy gain.
- **Alert policy** (`alert_eval`):
  - **Firing rule:** fire only while glucose ≤ 180, at most once an hour.
  - **Threshold:** 0.51, chosen on DEV over a 0.01-step grid. It maximises serious-episode recall at ≤ the v7 policy's false alerts/day.
  - **Lead time:** measured from the *earliest* alert in the 30–120 min window (fixed before the TEST run; the "latest alert" pinned it at 30 min).
  - **TEST, v7 model:** warned 1,526 → 1,571 of 2,515 serious episodes (+1.8 points, CI +1.3 to +2.2); false alerts/day 1.46 → 1.34 (CI of the diff −0.21 to −0.001); alerts/day 8.6 → 3.6.
  - **TEST, v9 model:** +1.7 points (CI +0.1 to +3.9); false alerts −0.14/day (n.s.).
  - **Person-bootstrap CIs:** in `stage_report.json`.
- **Label sensitivity** (`stage_labels.json`): 82.1–85.9 % accuracy for peak > 170 / 190 / 198 or ≥ 2 bins; 49 % of errors are within ±10 % of 180.
- **Data search:** no openly licensed Indian Type 2 CGM dataset was available; documented as a limitation.
- **Serving** (`twin_v7.py`):
  - **Model loading:** loads `backend/models/twin_v9/model_2h.joblib` if present (trained without the 30 demo patients; thresholds from DEV); otherwise the v7 2-h model. **Rollback = delete `backend/models/twin_v9/`.**
  - **Shanghai inputs:** estimated Shanghai carbs and oral-drug doses come from `data/twin_v7/streams_v9_extra.parquet`, merged for model inputs only (the chart shows the recorded stream). `streams.parquet` is unchanged.
  - **v7 masking:** v7 bundles get Shanghai carb features masked (`mask_shanghai_carbs`, default True), because v7 never saw them.
  - **Responses:** the 2-h response has `early_warning {active, threshold, rule}` and `model_version`.
  - **Example moments:** re-picked with v9 in `twin_v9/demo_bins.json`.
  - **History window:** `HISTORY_DAYS` is now 15 (14-day feature).
- **Final model on all 187 people:** `models/combined_v9/model_final.joblib` (gitignored). Verified numbers are in `models/combined_v9/model_card.json` and the dashboard `summary.json["v9"]`, written by `verify_results --update-artifacts`.

### Feature contract (`backend/app/features.py`)
`FEATURE_COLS` is the single source of truth, with two implementations: `build_features_from_df` (training, vectorised, `shift(1)` windows grouped by patient) and `extract_inference_features` (serving; now just the last row of `build_features_from_df` on ~24 h of readings). Adding or changing a feature means editing both, `FEATURE_DESCRIPTIONS` in `predictor.py`, and retraining.

**Train/serve skew is fixed (2026-10-23).** `extract_inference_features` now runs `build_features_from_df` on the supplied readings and returns the last row, so serving uses the training code path. Callers must pass ~24 h of readings (144 at 10-min sampling; `main.predict_glucose` and `services.build_digital_twin` do). `test_inference_values_match_training_features` compares values, not just column names. The saved synthetic model was NOT retrained; its reported metrics were always measured on the training path, so served predictions now match them.

### Backend request flow
`main.py` routes -> `data_loader.py` (module-level in-memory cache of the parquet/json; re-filters the full DataFrame per call) -> `services.py` (`build_digital_twin`, `get_dashboard_stats`) -> `predictor.predict`. Endpoints: `/api/health`, `/api/stats`, `/api/patients[/{id}[/readings|/twin]]`, `POST /api/predict/{id}`, `/api/model-card`; plus the real-data twin under `/api/v7/*` (see v7 below). The FastAPI lifespan warms the data cache and **trains synchronously if the `.pkl` is missing**.
- `/api/v7` extras for the studio:
  - `POST /api/v7/predict/{pid}?bin=` takes an optional JSON body `Scenario {carbs_g, insulin_u, oral_tablets}`, applied at the "now" 5-min step. It returns `scenario {horizons, used_by_horizon, ignored, note}`; v7 models with Shanghai carbs masked and non-Shanghai tablets are reported as ignored.
  - `GET /api/v7/patients/{pid}/overview?bin=` returns completeness and 24-h stats.
  - `GET /api/v7/patients/{pid}/trends?bin=` returns per-day TIR, an hourly profile and GMI. Both use only data up to the chosen bin.
  - Stream `hours` now goes up to 336.
  - **Your own data:** `POST /api/v7/upload` takes a multipart CSV plus optional form fields (diabetes_type, age, sex, hba1c, bmi, years_with_diabetes, on_insulin, on_metformin) and requires `consent=true`.
    - **Parser:** `app/user_upload.py` auto-detects Dexcom Clarity, LibreView (metadata first line, day-first dates, mmol/L) and generic CSV, then reuses `external_loaders._assemble`. It keeps the last 30 days and the last segment after a > 6 h gap; minimum 6 h, warns below 24 h.
    - **Storage:** the upload is registered as patient `upload_xxxxxx` in the twin's **in-memory** cache only (max 20), so every `/api/v7` endpoint works on it. Nothing is written to disk; it is lost on restart.
    - **Other endpoints:** `DELETE /api/v7/upload/{pid}` removes an upload. `GET /api/v7/upload/example.csv` serves 3 days of demo patient hupa_0001 (CC BY) in the generic format.
    - **Frontend:** the studio "My data" page (`studio/pages/MyDataPage.jsx`).
- Prediction uses the last 144 readings (24 h). Risk bands are hard-coded in `predict`: >=0.60 high, >=0.30 moderate. `predicted_glucose_mgdl` is a linear extrapolation heuristic, not a model output.
- `build_digital_twin` swallows prediction exceptions, so `latest_prediction` can be `null` with no error.
- Dashboard `risk_level` and "high-risk" counts come from HbA1c/BMI rules in `data_generator.py`, **not** from the ML model; `avg_glucose` averages only the first 20 patients.
- `/api/model-card` merges `backend/models/model_card.json` with live `metrics.json` (as `live_*` keys).
- `schemas.py` is imported nowhere and no route sets `response_model`; it is out of sync with the real responses (e.g. `PatientSummary.last_glucose` is not returned).

### Data quirks
Readings are anchored to *generation* time (`end = now`) and then persisted, so the "live" data is frozen (currently ending 2026-10-20 17:14 UTC) and regenerating shifts every timestamp. Meal/exercise events are indexed from the series start rather than wall-clock, while activity and heart rate use the real clock hour, so time-of-day features are not aligned with the simulated meal schedule.

### Frontend (`frontend/src`)
**Default UI: the "studio" layout (`src/studio/`)**, a light glass clinical design (with dark theme).
- **Shell:** `Studio.jsx` (sidebar + header). Pages are in `studio/pages/`: Patient input, Prediction, Live monitor, Trends, Reports, Analytics and Settings, plus `SyntheticPage` for synthetic mode.
- **State:** `store.jsx` holds the four **modes** in localStorage key `glucotwin.modes`; URL params such as `?theme=dark&data=synthetic&layout=classic` override them. The page follows the URL hash, e.g. `#trends`.
  - `data`: `open` | `synthetic`.
  - `alert`: `standard` alerts at high risk (≥ `threshold_accuracy_mode`); `safety` also alerts at medium risk (≥ `threshold_safety_mode`).
  - `theme`: `light` | `dark`.
  - `layout`: `studio` | `classic`.
  - `bg`: a custom page-background hex or `null`. Settings has presets, a colour picker and R/G/B sliders; `?bg=%23132743` also works. `backgroundVars()` picks readable header text by WCAG luminance and makes cards solid on dark colours.
- **Shared UI:** charts and primitives are in `ui.jsx`. Range colours were validated with the dataviz palette script: CVD-safe red / teal / amber; red vs green failed for deutan readers. Bars start at zero, and there is no dual axis.
- **Daily context:** the form becomes a what-if scenario via `contextToScenario`. Only carbs, rapid insulin and oral tablets are sent; activity, sleep and notes are context only, and the UI says so.
- **Classic layout:** the original dashboard (`App.jsx` `ClassicApp`, below) is kept as `layout=classic`.
- **Screenshots:** headless Brave against `npm run dev` + uvicorn :8000. Don't pass `--user-data-dir` (it hangs), and note macOS has no `timeout`.

Classic layout:
`App.jsx` holds the selected patient; `PatientView` composes `DigitalTwin`, `PatientProfile`, `PredictionPanel`. Data goes through `hooks/useApi.js` -> `services/api.js` (axios, base `/api`, 30 s timeout). Exception: `ModelPerformance.jsx` (mounted inside `PredictionPanel`) calls `fetch('/api/model-card')` directly and picks the ACTIVE row by `card.model_name`. Styling is one vanilla-CSS file, `index.css` (CSS variables) plus heavy inline styles. `App.jsx` opens in the v7 real-data mode (`components/TwinV7.jsx`, its own sidebar and `/api/v7` calls in `services/api.js`) with a top-bar switch to the synthetic demo.

## Gotchas

- **README's "reproduce from scratch" is destructive:** `rm -f backend/models/*.json` deletes `model_card.json`, which no current code regenerates (only `ohio_train_model.py` writes a model card, to root `models/`). Without it `/api/model-card` loses `model_name` and the UI shows no ACTIVE model. Back it up first. `data_quality_report.json` and `metrics.json` are regenerated by `train_model`.
- **Swapping or rolling back a model:** copy a `.pkl` + `metrics.json` from `backend/models/snapshots/{v2_histgb_20241020 (24 h data), v2_histgb_7day}/` (commands in `MODEL_COMPARISON.md`) and **restart the server** — `predictor` caches the bundle (`reload_model()` exists but nothing calls it).
- `train_model.py` does not compute accuracy (no `accuracy` key in `metrics.json`); the UI derives the headline figure from the confusion matrix, so per-model accuracy cells in the comparison table show "—".
- **Rounding:** never round an already-rounded number. Format from full precision or from counts. The verifier has caught this three times (AUC 0.914, false alerts/day 1.47 → 1.46, deck).
- **Deck PDF export:** `osascript` with Keynote (`open` → `delay 8` → `export front document … as PDF` → `close saving no`). Render pages for QA with `fitz` (PyMuPDF; poppler is not installed). Dashboard screenshot: headless Brave against `npm run dev` + uvicorn on :8000.
- `tests/test_twin_v7_api.py` is skipped unless the demo pack exists; its v9 tests skip unless `backend/models/twin_v9` exists. Python here is 3.9: no `X | Y` unions at runtime and no backslashes inside f-string expressions.
- Tests need `backend/data/*` on disk (asserts >=100 patients) and a trained model (trains if absent). `TestClient(app)` is created without a context manager, so the lifespan does not run. `test_features.py` generates data in memory and writes nothing.
- `.gitignore` excludes raw datasets, `external_data/`, zips, the root `models/glucotwin_model.pkl` and `models/combined_*/*.parquet|*.joblib`. That covers `combined_t2d/`, whose 100 MB frame was missed by the old `combined_v*` rule. Before every commit, check `git status` that no OhioT1DM-derived file is staged.
- **Licence wording:** BrisT1D's dataset is CC BY 4.0, but the project used its Kaggle competition copy (rules not verified). Don't write "every training dataset is CC BY 4.0".

## Maintaining this file

Update it in the same change whenever the served model or `FEATURE_COLS` changes, an endpoint is added, or a Gotcha above is fixed (e.g. train/serve skew resolved, a real-data model wired into `predictor.py`). Refresh the date and test count at the top. Prefer pointing at files over copying their contents; README already holds the full feature table, API table and metrics.
