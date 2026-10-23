"""
Data Quality Validation Pipeline for GlucoTwin AI.
Runs validation checks on the synthetic dataset and produces a quality report.
Documented assumptions and explicit handling — no silent replacements.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

REPORT_PATH = Path(__file__).parent.parent / "models" / "data_quality_report.json"

# ── Plausibility bounds (medically informed) ──────────────────────────────────
GLUCOSE_MIN_MGDL  = 30.0    # Below this CGM noise / sensor error
GLUCOSE_MAX_MGDL  = 600.0   # Above this physiologically implausible
HR_MIN_BPM        = 30.0    # Extreme bradycardia
HR_MAX_BPM        = 220.0   # Maximum possible HR
SLEEP_MIN_H       = 0.0
SLEEP_MAX_H       = 24.0
BMI_MIN           = 10.0
BMI_MAX           = 80.0
HBA1C_MIN         = 3.0
HBA1C_MAX         = 20.0
AGE_MIN           = 0
AGE_MAX           = 120
STEPS_MIN         = 0
STEPS_MAX         = 10000   # per 10-minute interval
EXPECTED_INTERVAL_MIN = 10  # CGM sampling interval


def validate_readings(df: pd.DataFrame, patients: List[Dict]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run all data-quality checks on the readings DataFrame.
    Returns (cleaned_df, report_dict).
    """
    report: Dict[str, Any] = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "raw_record_count": int(len(df)),
        "raw_patient_count": int(df["patient_id"].nunique()),
        "checks": {},
        "corrections": [],
        "exclusions": [],
        "final_record_count": None,
        "final_patient_count": None,
        "target_distribution": {},
        "missing_value_pct": {},
        "assumptions": [
            "Glucose spike threshold: >180 mg/dL (ADA postprandial upper bound).",
            "Prediction horizon: 2 hours (12 × 10-min intervals).",
            "spike_target == -1 rows (no future data) are excluded from training only.",
            "Plausibility bounds applied per column; implausible values are NaN-flagged and imputed with patient median.",
            "Missing future target not treated as negative event — rows excluded instead.",
        ],
    }

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # ── 1. Missing values ─────────────────────────────────────────────────────
    mv_pct = {}
    for col in df.columns:
        pct = df[col].isna().mean() * 100
        mv_pct[col] = round(pct, 2)
    report["missing_value_pct"] = mv_pct

    # ── 2. Duplicates ─────────────────────────────────────────────────────────
    dups = int(df.duplicated(subset=["patient_id", "timestamp"]).sum())
    report["checks"]["duplicate_patient_timestamp_pairs"] = dups
    if dups > 0:
        df = df.drop_duplicates(subset=["patient_id", "timestamp"], keep="first")
        report["corrections"].append(
            f"Removed {dups} duplicate (patient_id, timestamp) rows — kept first occurrence."
        )

    # ── 3. Patient ID consistency ─────────────────────────────────────────────
    ehr_ids  = {p["patient_id"] for p in patients}
    read_ids = set(df["patient_id"].unique())
    orphan   = read_ids - ehr_ids
    missing  = ehr_ids - read_ids
    report["checks"]["orphan_reading_patients"]  = list(orphan)
    report["checks"]["ehr_patients_without_readings"] = list(missing)
    if orphan:
        df = df[~df["patient_id"].isin(orphan)]
        report["exclusions"].append(f"Excluded {len(orphan)} patient(s) with readings but no EHR record.")

    # ── 4. Timestamp validity ─────────────────────────────────────────────────
    bad_ts = int(df["timestamp"].isna().sum())
    report["checks"]["invalid_timestamps"] = bad_ts
    if bad_ts:
        df = df.dropna(subset=["timestamp"])
        report["exclusions"].append(f"Excluded {bad_ts} rows with unparseable timestamps.")

    # ── 5. Chronological ordering ─────────────────────────────────────────────
    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

    # ── 6. Sampling interval check ────────────────────────────────────────────
    intervals = (
        df.groupby("patient_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60)
        .dropna()
    )
    irregular = int((intervals != EXPECTED_INTERVAL_MIN).sum())
    report["checks"]["irregular_sampling_intervals"] = irregular
    if irregular > 0:
        report["corrections"].append(
            f"{irregular} intervals deviate from expected {EXPECTED_INTERVAL_MIN}-min cadence "
            "(start/end boundaries or timestamp rounding)."
        )

    # ── 7. Glucose plausibility ───────────────────────────────────────────────
    bad_glucose = ((df["glucose_mgdl"] < GLUCOSE_MIN_MGDL) | (df["glucose_mgdl"] > GLUCOSE_MAX_MGDL)).sum()
    report["checks"]["glucose_implausible_count"] = int(bad_glucose)
    if bad_glucose:
        df.loc[
            (df["glucose_mgdl"] < GLUCOSE_MIN_MGDL) | (df["glucose_mgdl"] > GLUCOSE_MAX_MGDL),
            "glucose_mgdl"
        ] = np.nan
        # Impute with patient-specific median (no cross-patient contamination)
        df["glucose_mgdl"] = df.groupby("patient_id")["glucose_mgdl"].transform(
            lambda s: s.fillna(s.median())
        )
        report["corrections"].append(
            f"Replaced {bad_glucose} glucose readings outside [{GLUCOSE_MIN_MGDL}, {GLUCOSE_MAX_MGDL}] mg/dL "
            "with patient-level median."
        )

    # ── 8. Heart-rate plausibility ────────────────────────────────────────────
    bad_hr = ((df["heart_rate_bpm"] < HR_MIN_BPM) | (df["heart_rate_bpm"] > HR_MAX_BPM)).sum()
    report["checks"]["heart_rate_implausible_count"] = int(bad_hr)
    if bad_hr:
        df.loc[
            (df["heart_rate_bpm"] < HR_MIN_BPM) | (df["heart_rate_bpm"] > HR_MAX_BPM),
            "heart_rate_bpm"
        ] = np.nan
        df["heart_rate_bpm"] = df.groupby("patient_id")["heart_rate_bpm"].transform(
            lambda s: s.fillna(s.median())
        )
        report["corrections"].append(f"Replaced {bad_hr} HR values outside [{HR_MIN_BPM}, {HR_MAX_BPM}] bpm.")

    # ── 9. Steps plausibility ─────────────────────────────────────────────────
    bad_steps = ((df["steps"] < STEPS_MIN) | (df["steps"] > STEPS_MAX)).sum()
    report["checks"]["steps_implausible_count"] = int(bad_steps)
    if bad_steps:
        df.loc[(df["steps"] < STEPS_MIN) | (df["steps"] > STEPS_MAX), "steps"] = np.nan
        df["steps"] = df.groupby("patient_id")["steps"].transform(
            lambda s: s.fillna(s.median())
        )
        report["corrections"].append(f"Replaced {bad_steps} step values outside [{STEPS_MIN}, {STEPS_MAX}].")

    # ── 10. BMI / HbA1c / Age static checks ──────────────────────────────────
    bad_bmi   = ((df["bmi"] < BMI_MIN) | (df["bmi"] > BMI_MAX)).sum()
    bad_hba1c = ((df["hba1c"] < HBA1C_MIN) | (df["hba1c"] > HBA1C_MAX)).sum()
    report["checks"]["bmi_implausible_count"]   = int(bad_bmi)
    report["checks"]["hba1c_implausible_count"] = int(bad_hba1c)

    # ── 11. Target label check ────────────────────────────────────────────────
    valid_targets = {-1, 0, 1}
    bad_target = (~df["spike_target"].isin(valid_targets)).sum()
    report["checks"]["invalid_target_labels"] = int(bad_target)

    # Rows with spike_target == -1 (horizon extends beyond dataset)
    unknown_horizon = int((df["spike_target"] == -1).sum())
    report["checks"]["unknown_horizon_rows_excluded_from_training"] = unknown_horizon

    # ── 12. Target class distribution (training rows only) ───────────────────
    train_df = df[df["spike_target"] != -1]
    vc = train_df["spike_target"].value_counts().to_dict()
    report["target_distribution"] = {
        "no_spike_0":  int(vc.get(0, 0)),
        "spike_1":     int(vc.get(1, 0)),
        "prevalence":  round(int(vc.get(1, 0)) / max(len(train_df), 1), 4),
    }

    # ── 13. Per-patient record count ─────────────────────────────────────────
    counts = df.groupby("patient_id").size()
    report["checks"]["patients_with_fewer_than_24_readings"] = int((counts < 24).sum())

    # ── Final tallies ─────────────────────────────────────────────────────────
    report["final_record_count"]  = int(len(df))
    report["final_patient_count"] = int(df["patient_id"].nunique())

    return df, report


def run_and_save(patients: List[Dict], readings_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    cleaned, report = validate_readings(readings_df, patients)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[validation] Data-quality report saved → {REPORT_PATH}")
    print(f"[validation] Raw: {report['raw_record_count']} rows  →  Clean: {report['final_record_count']} rows")
    print(f"[validation] Spike prevalence: {report['target_distribution'].get('prevalence', 0):.1%}")
    return cleaned, report
