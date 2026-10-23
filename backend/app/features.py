"""
Feature engineering for GlucoTwin AI — v2.
Shared between training (build_features_from_df) and inference (extract_inference_features).
Both paths produce features in IDENTICAL order as defined by FEATURE_COLS.

Leakage-prevention guarantees:
  • Rolling windows use only past observations (closed='left' / shift(1) semantics).
  • No future glucose values are included in any feature.
  • Rolling is always grouped per patient — never crosses patient boundaries.
  • Target column is never used to derive any feature.
  • StandardScaler is fitted on training data only (in the pipeline).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# ── Sampling constants ─────────────────────────────────────────────────────────
INTERVAL_MIN = 10
STEPS_1H  = 6    # 6 × 10 min = 1 hour
STEPS_2H  = 12   # 12 × 10 min = 2 hours
STEPS_3H  = 18
STEPS_4H  = 24   # 4-hour look-back for rate-of-change

SPIKE_THRESHOLD = 180.0  # mg/dL — ADA postprandial threshold

# ── Feature column list — SINGLE SOURCE OF TRUTH ─────────────────────────────
# Training and inference must use exactly these columns in this order.
FEATURE_COLS = [
    # Glucose — current and recent history
    "latest_glucose",
    "glucose_lag_1",            # one reading ago (10 min)
    "glucose_lag_3",            # 30 min ago
    "glucose_change_30min",     # Δ over last 30 min
    "glucose_change_1h",        # Δ over last 60 min
    "glucose_change_2h",        # Δ over last 2 h
    "glucose_roc_1h",           # rate of change per hour (mg/dL/h)
    "glucose_rolling_mean_1h",
    "glucose_rolling_mean_2h",
    "glucose_rolling_std_1h",
    "glucose_rolling_std_2h",
    "glucose_rolling_min_2h",
    "glucose_rolling_max_2h",
    "glucose_above_threshold",  # binary: current reading > 180
    "glucose_in_range",         # binary: 70–180 mg/dL
    # Heart rate
    "heart_rate_mean_1h",
    "heart_rate_std_1h",
    "heart_rate_max_1h",
    # Activity / wearable
    "steps_last_1h",
    "activity_score_mean_1h",
    "activity_score_max_1h",
    # Sleep
    "sleep_hours_last_night",
    # Temporal
    "time_of_day_sin",
    "time_of_day_cos",
    "hour_of_day",
    # Static patient
    "hba1c",
    "age",
    "bmi",
    "diabetes_flag",
    "gender_male",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _time_encoding(hour: int, minute: int) -> tuple[float, float]:
    frac = (hour * 60 + minute) / 1440.0
    return math.sin(2 * math.pi * frac), math.cos(2 * math.pi * frac)


def _safe_float(v, default: float = 0.0) -> float:
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


# ── Training-time feature builder ─────────────────────────────────────────────

def build_features_from_df(readings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix from the readings DataFrame.
    Used at training time.  Target column is NOT used here — no leakage.
    """
    df = readings_df.copy()
    df = df.sort_values(["patient_id", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    grp = df.groupby("patient_id", group_keys=False)

    # ── Glucose features ─────────────────────────────────────────────────────
    df["latest_glucose"] = df["glucose_mgdl"]

    # Lags (shift by k within patient — uses only past data)
    df["glucose_lag_1"] = grp["glucose_mgdl"].shift(1)
    df["glucose_lag_3"] = grp["glucose_mgdl"].shift(3)

    # Δ changes (diff from k steps back)
    df["glucose_change_30min"] = grp["glucose_mgdl"].transform(lambda s: s.diff(3))
    df["glucose_change_1h"]    = grp["glucose_mgdl"].transform(lambda s: s.diff(STEPS_1H))
    df["glucose_change_2h"]    = grp["glucose_mgdl"].transform(lambda s: s.diff(STEPS_2H))

    # Rate of change: mg/dL per hour over last 1h window
    df["glucose_roc_1h"] = df["glucose_change_1h"] / 1.0  # Δ over 1 h → units are mg/dL/h

    # Rolling stats (window = past N readings, strictly backward — no look-ahead)
    df["glucose_rolling_mean_1h"] = grp["glucose_mgdl"].transform(
        lambda s: s.shift(1).rolling(STEPS_1H, min_periods=1).mean()
    )
    df["glucose_rolling_mean_2h"] = grp["glucose_mgdl"].transform(
        lambda s: s.shift(1).rolling(STEPS_2H, min_periods=1).mean()
    )
    df["glucose_rolling_std_1h"] = grp["glucose_mgdl"].transform(
        lambda s: s.shift(1).rolling(STEPS_1H, min_periods=2).std().fillna(0)
    )
    df["glucose_rolling_std_2h"] = grp["glucose_mgdl"].transform(
        lambda s: s.shift(1).rolling(STEPS_2H, min_periods=2).std().fillna(0)
    )
    df["glucose_rolling_min_2h"] = grp["glucose_mgdl"].transform(
        lambda s: s.shift(1).rolling(STEPS_2H, min_periods=1).min()
    )
    df["glucose_rolling_max_2h"] = grp["glucose_mgdl"].transform(
        lambda s: s.shift(1).rolling(STEPS_2H, min_periods=1).max()
    )

    # Binary flags
    df["glucose_above_threshold"] = (df["glucose_mgdl"] > SPIKE_THRESHOLD).astype(float)
    df["glucose_in_range"]        = ((df["glucose_mgdl"] >= 70) & (df["glucose_mgdl"] <= 180)).astype(float)

    # ── Heart rate ────────────────────────────────────────────────────────────
    df["heart_rate_mean_1h"] = grp["heart_rate_bpm"].transform(
        lambda s: s.shift(1).rolling(STEPS_1H, min_periods=1).mean()
    )
    df["heart_rate_std_1h"] = grp["heart_rate_bpm"].transform(
        lambda s: s.shift(1).rolling(STEPS_1H, min_periods=2).std().fillna(0)
    )
    df["heart_rate_max_1h"] = grp["heart_rate_bpm"].transform(
        lambda s: s.shift(1).rolling(STEPS_1H, min_periods=1).max()
    )

    # ── Activity ──────────────────────────────────────────────────────────────
    df["steps_last_1h"] = grp["steps"].transform(
        lambda s: s.shift(1).rolling(STEPS_1H, min_periods=1).sum().fillna(0)
    )
    df["activity_score_mean_1h"] = grp["activity_score"].transform(
        lambda s: s.shift(1).rolling(STEPS_1H, min_periods=1).mean().fillna(0)
    )
    df["activity_score_max_1h"] = grp["activity_score"].transform(
        lambda s: s.shift(1).rolling(STEPS_1H, min_periods=1).max().fillna(0)
    )

    # ── Sleep ─────────────────────────────────────────────────────────────────
    df["sleep_hours_last_night"] = grp["sleep_hours"].transform(
        lambda s: s.ffill().fillna(7.0)
    )

    # ── Temporal ──────────────────────────────────────────────────────────────
    df["hour_of_day"] = df["timestamp"].dt.hour.astype(float)
    df["time_of_day_sin"] = df["hour_of_day"].apply(
        lambda h: math.sin(2 * math.pi * h / 24)
    )
    df["time_of_day_cos"] = df["hour_of_day"].apply(
        lambda h: math.cos(2 * math.pi * h / 24)
    )

    # ── Fill remaining NaN from lags at series starts ──────────────────────
    lag_fill = {
        "glucose_lag_1": df["glucose_mgdl"],
        "glucose_lag_3": df["glucose_mgdl"],
        "glucose_change_30min": 0.0,
        "glucose_change_1h": 0.0,
        "glucose_change_2h": 0.0,
        "glucose_roc_1h": 0.0,
        "glucose_rolling_mean_1h": df["glucose_mgdl"],
        "glucose_rolling_mean_2h": df["glucose_mgdl"],
        "glucose_rolling_std_1h": 0.0,
        "glucose_rolling_std_2h": 0.0,
        "glucose_rolling_min_2h": df["glucose_mgdl"],
        "glucose_rolling_max_2h": df["glucose_mgdl"],
        "heart_rate_mean_1h": df.get("heart_rate_bpm", 75.0),
        "heart_rate_std_1h": 0.0,
        "heart_rate_max_1h": df.get("heart_rate_bpm", 75.0),
        "steps_last_1h": 0.0,
        "activity_score_mean_1h": 0.0,
        "activity_score_max_1h": 0.0,
    }
    for col, filler in lag_fill.items():
        if col in df.columns:
            if isinstance(filler, (int, float)):
                df[col] = df[col].fillna(filler)
            else:
                df[col] = df[col].fillna(filler)

    return df


# ── Inference-time feature extractor ─────────────────────────────────────────

def extract_inference_features(
    recent_readings: List[Dict],
    patient_static: Dict[str, Any],
) -> Dict[str, float]:
    """
    Inference features for the latest reading = the last row of build_features_from_df run on the recent
    readings, i.e. literally the training code path (no train/serve skew). Pass >= 24 h of readings
    (144 at 10-min sampling) so the nightly sleep value and every rolling window are populated as in training.
    Only data available at prediction time is used.
    """
    if not recent_readings:
        raise ValueError("No recent readings provided for inference.")

    df = pd.DataFrame(recent_readings)
    for col in ("heart_rate_bpm", "sleep_hours", "steps", "activity_score"):
        if col not in df.columns:
            df[col] = np.nan
    df["patient_id"] = "inference"
    for col in ("heart_rate_bpm", "sleep_hours"):                       # API rows carry None, not NaN
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["steps"] = pd.to_numeric(df["steps"], errors="coerce").fillna(0)
    df["activity_score"] = pd.to_numeric(df["activity_score"], errors="coerce").fillna(0)
    df["hba1c"] = _safe_float(patient_static.get("hba1c", 5.5))
    df["age"] = _safe_float(patient_static.get("age", 45))
    df["bmi"] = _safe_float(patient_static.get("bmi", 25.0))
    df["diabetes_flag"] = float(patient_static.get("diabetes_type") is not None)
    df["gender_male"] = float(patient_static.get("gender", "Male") == "Male")
    feats = build_features_from_df(df)
    last = feats.sort_values("timestamp").iloc[-1]
    return {c: _safe_float(last[c], 0.0) for c in FEATURE_COLS}


def glucose_trend(readings: List[Dict]) -> str:
    """Return 'rising', 'falling', or 'stable' based on last 3–6 readings."""
    if len(readings) < 2:
        return "stable"
    glucose_vals = [r["glucose_mgdl"] for r in readings[-6:]]
    delta = glucose_vals[-1] - glucose_vals[0]
    if delta > 15:
        return "rising"
    if delta < -15:
        return "falling"
    return "stable"
