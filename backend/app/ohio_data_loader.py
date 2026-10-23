"""
OhioT1DM Real Data Loader
Loads and merges all 12 real T1D patients from Ohio 2018 & 2020 datasets.

Columns in raw CSV:
  5minute_intervals_timestamp, missing_cbg, cbg, finger,
  basal, hr, gsr, carbInput, bolus
"""

import os
import glob
import pandas as pd
import numpy as np

# Absolute path to the extracted dataset
OHIO_BASE = os.path.join(
    os.path.dirname(__file__),
    "../../ohiot1dm_raw/ohiot1dm-glucose-dataset-main/Ohio Data"
)


def load_ohio_data(years=("2018", "2020"), splits=("train", "test")) -> pd.DataFrame:
    """
    Load all OhioT1DM CSV files and combine into a single DataFrame.
    Returns a cleaned DataFrame with standardised column names.
    """
    base = os.path.abspath(OHIO_BASE)
    records = []

    for year in years:
        for split in splits:
            pattern = os.path.join(base, f"Ohio{year}_processed", split, "*.csv")
            files = glob.glob(pattern)
            for fpath in sorted(files):
                patient_id = os.path.basename(fpath).split("-")[0]
                df = pd.read_csv(fpath)
                df["patient_id"] = patient_id
                df["year"] = year
                df["split"] = split
                records.append(df)

    if not records:
        raise FileNotFoundError(
            f"No OhioT1DM CSV files found under: {base}\n"
            "Make sure ohiot1dm_raw/ folder is at project root."
        )

    combined = pd.concat(records, ignore_index=True)

    # ── Rename columns to standard names ──────────────────────────────────
    combined = combined.rename(columns={
        "5minute_intervals_timestamp": "timestamp",
        "cbg":        "glucose_mgdl",   # Continuous blood glucose (CGM)
        "finger":     "finger_glucose",  # Finger-stick (reference)
        "basal":      "basal_rate",      # Basal insulin rate
        "hr":         "heart_rate_bpm",
        "gsr":        "gsr",             # Galvanic skin response (stress proxy)
        "carbInput":  "carbs_grams",     # Meal carbohydrate input
        "bolus":      "bolus_units",     # Bolus insulin dose
        "missing_cbg":"missing_cbg",
    })

    # ── Sort by patient + time ─────────────────────────────────────────────
    combined = combined.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

    # ── Forward-fill basal (it's a continuous setting) ────────────────────
    combined["basal_rate"] = (
        combined.groupby("patient_id")["basal_rate"]
        .transform(lambda s: s.ffill().fillna(0))
    )

    # ── Fill carbs & bolus NaN → 0  (no meal / no dose at that interval) ──
    combined["carbs_grams"] = combined["carbs_grams"].fillna(0.0)
    combined["bolus_units"] = combined["bolus_units"].fillna(0.0)

    # ── Fill heart rate & GSR with patient median ─────────────────────────
    for col in ["heart_rate_bpm", "gsr"]:
        combined[col] = (
            combined.groupby("patient_id")[col]
            .transform(lambda s: s.fillna(s.median()))
        )

    # ── Drop rows where CGM is missing (cannot train/predict without it) ──
    combined = combined[combined["missing_cbg"] == 0].copy()
    combined = combined.dropna(subset=["glucose_mgdl"]).reset_index(drop=True)

    return combined


def get_patient_list(df: pd.DataFrame) -> list[dict]:
    """Return a list of patient summary dicts."""
    patients = []
    for pid, grp in df.groupby("patient_id"):
        patients.append({
            "patient_id": pid,
            "n_readings": len(grp),
            "mean_glucose": round(grp["glucose_mgdl"].mean(), 1),
            "spike_prevalence": round((grp["glucose_mgdl"] > 180).mean() * 100, 1),
            "years": grp["year"].unique().tolist(),
        })
    return patients


if __name__ == "__main__":
    df = load_ohio_data()
    print(f"Loaded {len(df):,} rows for {df['patient_id'].nunique()} patients")
    print(df.head())
    print("\nPatient summary:")
    for p in get_patient_list(df):
        print(p)
