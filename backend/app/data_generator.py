"""
Synthetic data generator for GlucoTwin AI — v2.
Generates realistic but entirely SYNTHETIC EHR + CGM/wearable records.
All data is clearly artificial — no real patient information.

Changes from v1:
  • 7-day histories (1008 readings per patient @ 10-min intervals)
  • Richer inter-patient variation: different meal timing, exercise patterns, stress
  • Realistic sleep: separate hours for each night, not one value per patient
  • Glucose patterns influenced by HbA1c, BMI, and diabetes type (no leakage)
  • activity_score stored per row for better feature engineering
  • Spike target integrity: -1 only when future window extends beyond dataset

Deterministic: seeded at the module level — run with the same seed → identical output.

Usage:
  python -m app.data_generator            # generate and save
  from app.data_generator import load_or_generate
"""
from __future__ import annotations

import json
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ── Reproducible seed ─────────────────────────────────────────────────────────
SEED = 42
RNG = np.random.default_rng(SEED)
random.seed(SEED)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Dataset constants ─────────────────────────────────────────────────────────
NUM_PATIENTS           = 120
DAYS_OF_HISTORY        = 7
INTERVAL_MIN           = 10
READINGS_PER_DAY       = 24 * 60 // INTERVAL_MIN          # 144
READINGS_PER_PATIENT   = READINGS_PER_DAY * DAYS_OF_HISTORY  # 1008
SPIKE_THRESHOLD_MGDL   = 180.0    # ADA postprandial upper bound
HORIZON_STEPS          = 12       # 12 × 10 min = 2 hours ahead

# ── Name pools ────────────────────────────────────────────────────────────────
FIRST_NAMES = [
    "Aarav","Priya","Rahul","Sneha","Vikram","Ananya","Rohan","Divya",
    "Karthik","Meera","Arjun","Pooja","Siddharth","Kavya","Nikhil",
    "Riya","Aditya","Shreya","Manish","Nisha","James","Emily","Robert",
    "Sarah","Michael","Jessica","David","Ashley","Daniel","Amanda",
    "Chen","Wei","Fatima","Omar","Amara","Kofi","Sofia","Mateo",
]
LAST_NAMES = [
    "Sharma","Patel","Kumar","Singh","Reddy","Gupta","Nair","Iyer",
    "Mehta","Joshi","Smith","Johnson","Williams","Brown","Jones",
    "Garcia","Miller","Davis","Martinez","Wilson","Chen","Wang","Ali",
]
GENDERS        = ["Male","Female"]
DIABETES_TYPES = ["Type 1","Type 2",None,None,None]   # ~40% diabetic

DIAGNOSES_POOL = [
    ("E11","Type 2 Diabetes Mellitus"),
    ("E10","Type 1 Diabetes Mellitus"),
    ("I10","Essential Hypertension"),
    ("E78.5","Hyperlipidemia"),
    ("E66","Obesity"),
    ("J45","Asthma"),
    ("K21","GERD"),
    ("M54.5","Low Back Pain"),
    ("F32.9","Major Depressive Disorder"),
    ("N18.3","Chronic Kidney Disease Stage 3"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _random_name() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def _glucose_series(
    base: float,
    n: int,
    diabetic: bool,
    hba1c: float,
    bmi: float,
    meal_offset_h: float = 0.0,
    stress_factor: float = 1.0,
) -> np.ndarray:
    """
    Simulate a 7-day CGM trace.
    Meals recur every 24 h at patient-specific times (meal_offset_h).
    Stress_factor raises post-meal spikes for high-stress patients.
    HbA1c and BMI modulate the baseline and spike magnitude.
    """
    series = np.zeros(n)
    series[0] = base
    noise_sd = 2.5 if diabetic else 1.5

    # Meal indices: breakfast, lunch, dinner, snack — each recurs daily
    meal_offsets_min = [
        int((7 + meal_offset_h) * 60),   # breakfast
        int((13 + meal_offset_h * 0.5) * 60),  # lunch
        int((19 + meal_offset_h * 0.3) * 60),  # dinner
        int((22 + meal_offset_h * 0.1) * 60),  # snack
    ]
    meal_indices = set()
    for day in range(DAYS_OF_HISTORY):
        for offset in meal_offsets_min:
            idx = day * READINGS_PER_DAY + offset // INTERVAL_MIN
            if idx < n:
                meal_indices.add(idx)

    # Exercise indices: once or twice per day
    ex_offsets_min = [int(10 * 60), int(17 * 60)]   # 10am, 5pm
    exercise_indices = set()
    for day in range(DAYS_OF_HISTORY):
        for offset in ex_offsets_min:
            idx = day * READINGS_PER_DAY + offset // INTERVAL_MIN
            if idx < n:
                exercise_indices.add(idx)

    # HbA1c and BMI influence spike magnitude
    hba1c_factor = max(0.5, (hba1c - 4.0) / 5.0)   # ~0.5 at normal, ~2 at HbA1c 14
    bmi_factor   = max(0.8, bmi / 30.0)

    for i in range(1, n):
        delta = RNG.normal(0, noise_sd)

        # Meal spike
        if i in meal_indices:
            base_spike = RNG.uniform(30, 80) if diabetic else RNG.uniform(15, 40)
            delta += base_spike * hba1c_factor * stress_factor

        # Exercise dip
        if i in exercise_indices:
            delta -= RNG.uniform(8, 25) * bmi_factor

        # Mean-reversion to patient baseline
        delta += 0.06 * (base - series[i - 1])

        series[i] = np.clip(series[i - 1] + delta, 40.0, 400.0)

    return series


def _sleep_series(n_days: int, base_hours: float) -> List[float]:
    """Generate a realistic nightly sleep duration for each day."""
    hours = []
    for _ in range(n_days):
        h = float(np.clip(RNG.normal(base_hours, 0.8), 3.5, 10.0))
        hours.append(round(h, 1))
    return hours


# ── Patient generator ─────────────────────────────────────────────────────────

def generate_patients() -> List[Dict]:
    patients = []
    for idx in range(NUM_PATIENTS):
        pid    = f"P{idx + 1:04d}"
        gender = random.choice(GENDERS)
        age    = int(RNG.integers(18, 85))
        weight = float(RNG.normal(78 if gender == "Male" else 67, 14))
        height = float(RNG.normal(173 if gender == "Male" else 161, 8))
        bmi    = round(weight / ((height / 100) ** 2), 1)
        bmi    = float(np.clip(bmi, 14.0, 60.0))

        diabetes_type = random.choice(DIABETES_TYPES)
        diabetic      = diabetes_type is not None

        hba1c = float(RNG.normal(8.5 if diabetic else 5.4, 1.0 if diabetic else 0.4))
        hba1c = round(float(np.clip(hba1c, 4.0, 14.0)), 1)

        # Diagnoses
        n_dx = int(RNG.integers(1, 5))
        dx_pool = list(DIAGNOSES_POOL)
        if diabetic:
            dx_main = (
                ("E11","Type 2 Diabetes Mellitus")
                if diabetes_type == "Type 2"
                else ("E10","Type 1 Diabetes Mellitus")
            )
            dx_pool = [d for d in dx_pool if d[0] not in ("E11","E10")]
            selected_dx = [dx_main] + random.sample(dx_pool, min(n_dx - 1, len(dx_pool)))
        else:
            selected_dx = random.sample(dx_pool, min(n_dx, len(dx_pool)))

        base_date = datetime(2024, 1, 1)
        diagnoses = [
            {
                "code":         code,
                "description":  desc,
                "diagnosed_date": (base_date - timedelta(days=int(RNG.integers(30, 1825)))).strftime("%Y-%m-%d"),
            }
            for code, desc in selected_dx
        ]

        labs = [
            {
                "test_name":       "HbA1c",
                "value":           hba1c,
                "unit":            "%",
                "reference_range": "4.0–5.6",
                "date":            (base_date - timedelta(days=int(RNG.integers(1, 90)))).strftime("%Y-%m-%d"),
                "abnormal":        hba1c > 5.7,
            },
            {
                "test_name":       "Fasting Glucose",
                "value":           round(float(RNG.normal(135 if diabetic else 88, 20)), 1),
                "unit":            "mg/dL",
                "reference_range": "70–99",
                "date":            (base_date - timedelta(days=int(RNG.integers(1, 90)))).strftime("%Y-%m-%d"),
                "abnormal":        diabetic,
            },
            {
                "test_name":       "Total Cholesterol",
                "value":           round(float(RNG.normal(205, 38)), 1),
                "unit":            "mg/dL",
                "reference_range": "<200",
                "date":            (base_date - timedelta(days=int(RNG.integers(1, 180)))).strftime("%Y-%m-%d"),
                "abnormal":        False,
            },
            {
                "test_name":       "Creatinine",
                "value":           round(float(np.clip(RNG.normal(1.0, 0.25), 0.4, 3.0)), 2),
                "unit":            "mg/dL",
                "reference_range": "0.6–1.2",
                "date":            (base_date - timedelta(days=int(RNG.integers(1, 180)))).strftime("%Y-%m-%d"),
                "abnormal":        False,
            },
        ]

        risk_level = (
            "high"     if (hba1c > 7.5 or (diabetic and bmi > 32))
            else "moderate" if (hba1c > 6.0 or diabetic)
            else "low"
        )

        patients.append({
            "patient_id":   pid,
            "name":         _random_name(),
            "age":          age,
            "gender":       gender,
            "weight_kg":    round(weight, 1),
            "height_cm":    round(height, 1),
            "bmi":          bmi,
            "diabetes_type": diabetes_type,
            "diagnoses":    diagnoses,
            "lab_results":  labs,
            "hba1c":        hba1c,
            "risk_level":   risk_level,
            "last_updated": datetime.utcnow().isoformat(),
        })
    return patients


# ── Readings generator ────────────────────────────────────────────────────────

def generate_readings(patients: List[Dict]) -> pd.DataFrame:
    all_rows: List[Dict] = []
    # Anchor: readings end "now" so inference always has fresh context
    end_time   = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start_time = end_time - timedelta(minutes=INTERVAL_MIN * READINGS_PER_PATIENT)

    for p in patients:
        pid      = p["patient_id"]
        diabetic = p["diabetes_type"] is not None
        hba1c    = p["hba1c"]
        bmi      = p["bmi"]

        # Per-patient variation parameters
        base_glucose    = float(RNG.normal(155 if diabetic else 93, 22))
        base_glucose    = float(np.clip(base_glucose, 55, 310))
        hr_base         = float(RNG.normal(76, 10))
        hr_base         = float(np.clip(hr_base, 45, 120))
        meal_offset     = float(RNG.uniform(-1.0, 1.0))   # ±1 hour variation in meal timing
        stress_factor   = float(RNG.uniform(0.7, 1.5))    # inter-patient stress variation
        base_sleep      = float(np.clip(RNG.normal(6.8, 1.0), 4.0, 9.5))

        glucose_series = _glucose_series(
            base=base_glucose, n=READINGS_PER_PATIENT,
            diabetic=diabetic, hba1c=hba1c, bmi=bmi,
            meal_offset_h=meal_offset, stress_factor=stress_factor,
        )
        sleep_by_night = _sleep_series(DAYS_OF_HISTORY, base_sleep)

        for i in range(READINGS_PER_PATIENT):
            ts   = start_time + timedelta(minutes=INTERVAL_MIN * i)
            hour = ts.hour
            day  = i // READINGS_PER_DAY

            # Circadian activity pattern: low at night, active mid-morning and evening
            hour_frac    = (hour - 6) / 12.0
            activity_score = float(np.clip(
                1.5 * np.sin(np.pi * hour_frac) + RNG.normal(0, 0.25),
                0.0, 3.0
            ))
            steps_10min = int(np.clip(RNG.normal(activity_score * 180, 40), 0, 1200))
            hr          = float(np.clip(RNG.normal(hr_base + activity_score * 6, 5), 40, 160))

            activity_level = (
                "sedentary" if activity_score < 0.5
                else "light"    if activity_score < 1.2
                else "moderate" if activity_score < 2.0
                else "vigorous"
            )

            # Sleep: use that night's value at midnight, else None
            # (first reading of each day carries the previous night's sleep)
            sleep_hours = sleep_by_night[day] if (i % READINGS_PER_DAY == 0) else None

            # Spike target: will glucose exceed threshold in 2 hours?
            future_idx = i + HORIZON_STEPS
            if future_idx < READINGS_PER_PATIENT:
                future_glucose = glucose_series[future_idx]
                spike_target   = int(future_glucose > SPIKE_THRESHOLD_MGDL)
            else:
                spike_target = -1   # unknown — exclude from training

            all_rows.append({
                "patient_id":     pid,
                "timestamp":      ts.isoformat(),
                "glucose_mgdl":   round(float(glucose_series[i]), 1),
                "heart_rate_bpm": round(hr, 1),
                "sleep_hours":    sleep_hours,
                "steps":          steps_10min,
                "activity_level": activity_level,
                "activity_score": round(activity_score, 3),
                "spike_target":   spike_target,
                # Static EHR features (joined for convenience — no leakage)
                "age":            p["age"],
                "bmi":            bmi,
                "hba1c":          hba1c,
                "diabetes_flag":  int(diabetic),
                "gender_male":    int(p["gender"] == "Male"),
            })

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    return df


# ── Persistence ───────────────────────────────────────────────────────────────

def save_data(patients: List[Dict], readings_df: pd.DataFrame) -> None:
    with open(DATA_DIR / "patients.json", "w") as f:
        json.dump(patients, f, indent=2, default=str)
    readings_df.to_parquet(DATA_DIR / "readings.parquet", index=False)
    print(
        f"[data_generator] Saved {len(patients)} patients "
        f"and {len(readings_df):,} readings "
        f"({DAYS_OF_HISTORY}-day histories)."
    )


def load_or_generate(force: bool = False) -> tuple[List[Dict], pd.DataFrame]:
    patients_path = DATA_DIR / "patients.json"
    readings_path = DATA_DIR / "readings.parquet"

    if not force and patients_path.exists() and readings_path.exists():
        with open(patients_path) as f:
            patients = json.load(f)
        readings = pd.read_parquet(readings_path)
        print(
            f"[data_generator] Loaded {len(patients)} patients "
            f"and {len(readings):,} readings from disk."
        )
        return patients, readings

    print(f"[data_generator] Generating {DAYS_OF_HISTORY}-day synthetic data …")
    patients = generate_patients()
    readings = generate_readings(patients)
    save_data(patients, readings)
    return patients, readings


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv or "-f" in sys.argv
    pts, rdf = load_or_generate(force=force)
    print(f"\n  Patients : {len(pts)}")
    print(f"  Readings : {len(rdf):,}")
    print(f"  Per pt   : {len(rdf) // len(pts)}")
    print(f"  Span     : {DAYS_OF_HISTORY} days × 144 readings/day")
    print(f"  Spike %  : {rdf[rdf.spike_target != -1]['spike_target'].mean():.1%}")
    print("\nSample rows:")
    print(rdf.head(3).to_string(index=False))
