"""
Feature Engineering for OhioT1DM Real Data
==========================================
Builds 40 features from real CGM + meal + insulin + HR + GSR signals.
Uses the same shift(1)+groupby pattern to prevent data leakage.

Key NEW features vs synthetic pipeline:
  - carbs_grams, carbs_last_30min, carbs_last_1h, carbs_last_2h
  - bolus_units, bolus_last_1h, insulin_on_board (IOB proxy)
  - basal_rate
  - gsr (galvanic skin response — stress proxy)
  - glucose_acceleration (2nd derivative)
"""

import pandas as pd
import numpy as np
from typing import List

# ── Feature column contract ────────────────────────────────────────────────
FEATURE_COLS: List[str] = [
    # CGM history
    "latest_glucose",
    "glucose_lag_1",
    "glucose_lag_3",
    "glucose_lag_6",
    # Rate of change
    "glucose_change_30min",
    "glucose_change_1h",
    "glucose_change_2h",
    "glucose_roc_1h",
    "glucose_acceleration",
    # Rolling statistics
    "glucose_rolling_mean_1h",
    "glucose_rolling_mean_2h",
    "glucose_rolling_std_1h",
    "glucose_rolling_std_2h",
    "glucose_rolling_min_2h",
    "glucose_rolling_max_2h",
    # Flags
    "glucose_above_threshold",
    "glucose_in_range",
    # Meal / carbohydrates (KEY new features)
    "carbs_now",
    "carbs_last_30min",
    "carbs_last_1h",
    "carbs_last_2h",
    # Insulin (KEY new features)
    "bolus_now",
    "bolus_last_1h",
    "bolus_last_2h",
    "basal_rate",
    "insulin_on_board",      # proxy: weighted bolus decay over 2h
    # Heart rate
    "heart_rate_mean_1h",
    "heart_rate_std_1h",
    "heart_rate_max_1h",
    # GSR — stress proxy
    "gsr_mean_1h",
    # Temporal
    "time_of_day_sin",
    "time_of_day_cos",
    "hour_of_day",
]

TARGET_COL = "spike_target"
SPIKE_THRESHOLD = 180.0       # mg/dL — ADA postprandial upper bound
HORIZON_STEPS = 24            # 24 × 5 min = 2 hours ahead


def _roll(series: pd.Series, window: int, func: str = "mean") -> pd.Series:
    """Shift-then-roll to prevent look-ahead leakage."""
    shifted = series.shift(1)
    r = shifted.rolling(window, min_periods=1)
    return getattr(r, func)()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the combined OhioT1DM DataFrame, add feature columns and target.
    Input must have columns: patient_id, timestamp, glucose_mgdl,
                             carbs_grams, bolus_units, basal_rate,
                             heart_rate_bpm, gsr
    """
    df = df.copy().sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

    g = df.groupby("patient_id")

    # ── CGM history ────────────────────────────────────────────────────────
    df["latest_glucose"]    = df["glucose_mgdl"]
    df["glucose_lag_1"]     = g["glucose_mgdl"].shift(1)
    df["glucose_lag_3"]     = g["glucose_mgdl"].shift(3)
    df["glucose_lag_6"]     = g["glucose_mgdl"].shift(6)

    # ── Rate of change ─────────────────────────────────────────────────────
    df["glucose_change_30min"] = df["latest_glucose"] - g["glucose_mgdl"].shift(6)
    df["glucose_change_1h"]    = df["latest_glucose"] - g["glucose_mgdl"].shift(12)
    df["glucose_change_2h"]    = df["latest_glucose"] - g["glucose_mgdl"].shift(24)
    df["glucose_roc_1h"]       = df["glucose_change_1h"] / 12.0
    # Acceleration = change in velocity (2nd derivative)
    roc_prev = (g["glucose_mgdl"].shift(1) - g["glucose_mgdl"].shift(13)) / 12.0
    df["glucose_acceleration"] = df["glucose_roc_1h"] - roc_prev

    # ── Rolling glucose stats ──────────────────────────────────────────────
    df["glucose_rolling_mean_1h"]  = g["glucose_mgdl"].transform(lambda s: _roll(s, 12))
    df["glucose_rolling_mean_2h"]  = g["glucose_mgdl"].transform(lambda s: _roll(s, 24))
    df["glucose_rolling_std_1h"]   = g["glucose_mgdl"].transform(lambda s: _roll(s, 12, "std")).fillna(0)
    df["glucose_rolling_std_2h"]   = g["glucose_mgdl"].transform(lambda s: _roll(s, 24, "std")).fillna(0)
    df["glucose_rolling_min_2h"]   = g["glucose_mgdl"].transform(lambda s: _roll(s, 24, "min"))
    df["glucose_rolling_max_2h"]   = g["glucose_mgdl"].transform(lambda s: _roll(s, 24, "max"))

    # ── Glucose flags ──────────────────────────────────────────────────────
    df["glucose_above_threshold"] = (df["latest_glucose"] > SPIKE_THRESHOLD).astype(float)
    df["glucose_in_range"]        = ((df["latest_glucose"] >= 70) & (df["latest_glucose"] <= 180)).astype(float)

    # ── Meal / carbohydrates ───────────────────────────────────────────────
    df["carbs_now"]       = df["carbs_grams"]
    df["carbs_last_30min"]= g["carbs_grams"].transform(lambda s: _roll(s, 6,  "sum"))
    df["carbs_last_1h"]   = g["carbs_grams"].transform(lambda s: _roll(s, 12, "sum"))
    df["carbs_last_2h"]   = g["carbs_grams"].transform(lambda s: _roll(s, 24, "sum"))

    # ── Insulin ────────────────────────────────────────────────────────────
    df["bolus_now"]      = df["bolus_units"]
    df["bolus_last_1h"]  = g["bolus_units"].transform(lambda s: _roll(s, 12, "sum"))
    df["bolus_last_2h"]  = g["bolus_units"].transform(lambda s: _roll(s, 24, "sum"))
    df["basal_rate"]     = df["basal_rate"]

    # Insulin-on-Board proxy: exponential decay of boluses over last 2h
    # Each bolus decays to ~0 over 4h; simplified linear proxy over 24 steps
    def iob_proxy(s: pd.Series) -> pd.Series:
        weights = np.linspace(1.0, 0.0, 24)
        result = pd.Series(index=s.index, dtype=float)
        arr = s.values
        for i in range(len(arr)):
            window = arr[max(0, i-23): i+1]
            w = weights[-len(window):]
            result.iloc[i] = float(np.dot(window, w))
        return result

    df["insulin_on_board"] = g["bolus_units"].transform(iob_proxy)

    # ── Heart rate ─────────────────────────────────────────────────────────
    df["heart_rate_mean_1h"] = g["heart_rate_bpm"].transform(lambda s: _roll(s, 12))
    df["heart_rate_std_1h"]  = g["heart_rate_bpm"].transform(lambda s: _roll(s, 12, "std")).fillna(0)
    df["heart_rate_max_1h"]  = g["heart_rate_bpm"].transform(lambda s: _roll(s, 12, "max"))

    # ── GSR (stress proxy) ─────────────────────────────────────────────────
    df["gsr_mean_1h"] = g["gsr"].transform(lambda s: _roll(s, 12))

    # ── Temporal ───────────────────────────────────────────────────────────
    # timestamp is fractional minutes since some epoch; convert to hour-of-day
    # The timestamps are 5-minute intervals counted from a baseline
    minutes_in_day = 24 * 60
    df["_minute_of_day"] = (df["timestamp"] * 5) % minutes_in_day
    df["hour_of_day"]    = (df["_minute_of_day"] / 60).astype(int)
    df["time_of_day_sin"] = np.sin(2 * np.pi * df["_minute_of_day"] / minutes_in_day)
    df["time_of_day_cos"] = np.cos(2 * np.pi * df["_minute_of_day"] / minutes_in_day)
    df = df.drop(columns=["_minute_of_day"])

    # ── Target: will glucose exceed 180 within next 2 hours? ──────────────
    def make_target(glucose: pd.Series) -> pd.Series:
        future_max = glucose.shift(-HORIZON_STEPS).rolling(HORIZON_STEPS, min_periods=1).max()
        target = (future_max > SPIKE_THRESHOLD).astype(float)
        # Mark rows where future window goes beyond data as -1 (exclude later)
        target.iloc[-HORIZON_STEPS:] = -1
        return target

    df[TARGET_COL] = g["glucose_mgdl"].transform(make_target)

    return df


def get_training_data(df: pd.DataFrame):
    """
    Returns X (features), y (target), groups (patient_id) for sklearn.
    Excludes rows with unknown future (target == -1) and NaN features.
    """
    featured = build_features(df)
    # Exclude unknown-horizon rows
    featured = featured[featured[TARGET_COL] != -1].copy()
    # Drop rows with any NaN in features
    featured = featured.dropna(subset=FEATURE_COLS + [TARGET_COL])

    X = featured[FEATURE_COLS].values
    y = featured[TARGET_COL].values.astype(int)
    groups = featured["patient_id"].values
    return X, y, groups, featured


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from ohio_data_loader import load_ohio_data

    print("Loading OhioT1DM data...")
    df = load_ohio_data()
    print(f"Raw rows: {len(df):,}")

    X, y, groups, featured = get_training_data(df)
    print(f"Training rows: {len(X):,}")
    print(f"Features: {len(FEATURE_COLS)}")
    print(f"Spike prevalence: {y.mean()*100:.1f}%")
    print(f"Patients: {len(set(groups))}")
