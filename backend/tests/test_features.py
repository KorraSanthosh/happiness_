"""
Unit tests for feature engineering, data validation, temporal boundaries, and leakage prevention.
These complement test_api.py which covers HTTP contracts.

Run: cd backend && python -m pytest tests/test_features.py -v
"""
from __future__ import annotations

import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from app.features import (
    FEATURE_COLS,
    build_features_from_df,
    extract_inference_features,
    glucose_trend,
    _time_encoding,
    SPIKE_THRESHOLD,
)
from app.data_validation import validate_readings
from app.data_generator import generate_patients, generate_readings


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_data():
    """Small but realistic dataset for unit testing."""
    patients = generate_patients()[:10]   # 10 patients only
    readings = generate_readings(patients)
    return patients, readings


@pytest.fixture(scope="module")
def feature_df(synthetic_data):
    _, readings = synthetic_data
    return build_features_from_df(readings)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE COLUMN CONTRACT
# ═══════════════════════════════════════════════════════════════════════════════

def test_feature_cols_not_empty():
    assert len(FEATURE_COLS) >= 20, "Expected at least 20 engineered features"


def test_feature_cols_no_duplicates():
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS)), "Duplicate column names in FEATURE_COLS"


def test_build_features_produces_all_cols(feature_df):
    missing = [c for c in FEATURE_COLS if c not in feature_df.columns]
    assert missing == [], f"Missing feature columns: {missing}"


def test_build_features_no_extra_nans(feature_df):
    """After build_features_from_df all FEATURE_COLS should have no NaN."""
    nan_cols = [c for c in FEATURE_COLS if feature_df[c].isna().any()]
    assert nan_cols == [], f"NaN values remain in feature columns: {nan_cols}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LEAKAGE PREVENTION — rolling windows never use future data
# ═══════════════════════════════════════════════════════════════════════════════

def test_no_future_leakage_in_rolling_mean():
    """
    For each row, glucose_rolling_mean_2h must equal the mean of up to
    STEPS_2H PREVIOUS readings (shift=1), not including the current row.
    """
    patients = generate_patients()[:2]
    df_raw = generate_readings(patients)
    df = build_features_from_df(df_raw)

    for pid, grp in df.groupby("patient_id"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        for i in range(1, min(len(grp), 15)):
            # The feature at row i should only see rows 0..i-1
            past_glucose = grp["glucose_mgdl"].iloc[max(0, i - 12):i].values
            expected_mean = float(np.mean(past_glucose))
            actual_mean   = float(grp["glucose_rolling_mean_2h"].iloc[i])
            assert abs(actual_mean - expected_mean) < 0.5, (
                f"Patient {pid} row {i}: rolling_mean_2h={actual_mean:.2f} "
                f"but expected {expected_mean:.2f} (past-only)"
            )


def test_no_future_leakage_in_rolling_std():
    """glucose_rolling_std_2h must only use past observations."""
    patients = generate_patients()[:2]
    df_raw = generate_readings(patients)
    df = build_features_from_df(df_raw)

    for pid, grp in df.groupby("patient_id"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        for i in range(2, min(len(grp), 10)):
            past = grp["glucose_mgdl"].iloc[max(0, i - 12):i].values
            expected_std = float(np.std(past, ddof=1)) if len(past) >= 2 else 0.0
            actual_std   = float(grp["glucose_rolling_std_2h"].iloc[i])
            # Allow small float tolerance
            assert abs(actual_std - expected_std) < 1.0, (
                f"Std leakage? Patient {pid} row {i}: "
                f"actual={actual_std:.3f}, expected={expected_std:.3f}"
            )


def test_no_patient_boundary_leakage(feature_df):
    """
    Rolling features at the FIRST reading of a patient must not reflect
    the last reading of the previous patient (i.e. no cross-patient contamination).
    glucose_lag_1 at position 0 of a patient should equal latest_glucose (self-fill).
    """
    for pid, grp in feature_df.groupby("patient_id"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        row0 = grp.iloc[0]
        # At first reading, lag-1 must equal the current reading (no prior data)
        assert abs(row0["glucose_lag_1"] - row0["latest_glucose"]) < 0.01, (
            f"Patient {pid}: lag_1 at row 0 ({row0['glucose_lag_1']:.1f}) "
            f"!= latest_glucose ({row0['latest_glucose']:.1f}) — cross-patient leak?"
        )


def test_target_not_in_feature_cols():
    """spike_target must never appear in FEATURE_COLS."""
    assert "spike_target" not in FEATURE_COLS
    assert "future_glucose" not in FEATURE_COLS


# ═══════════════════════════════════════════════════════════════════════════════
# 3. INDIVIDUAL FEATURE CORRECTNESS
# ═══════════════════════════════════════════════════════════════════════════════

def test_glucose_above_threshold_binary(feature_df):
    """glucose_above_threshold must be 0 or 1, consistent with actual glucose."""
    sample = feature_df.head(500)
    for _, row in sample.iterrows():
        expected = 1.0 if row["glucose_mgdl"] > SPIKE_THRESHOLD else 0.0
        assert row["glucose_above_threshold"] == expected, (
            f"glucose={row['glucose_mgdl']:.1f}, "
            f"above_threshold={row['glucose_above_threshold']}, expected={expected}"
        )


def test_glucose_in_range_binary(feature_df):
    """glucose_in_range: 1 iff 70 <= glucose <= 180."""
    sample = feature_df.head(500)
    for _, row in sample.iterrows():
        expected = 1.0 if 70.0 <= row["glucose_mgdl"] <= 180.0 else 0.0
        assert row["glucose_in_range"] == expected


def test_time_encoding_range():
    """Sine/cosine time encoding must be in [-1, 1]."""
    for h in range(24):
        for m in [0, 15, 30, 45]:
            s, c = _time_encoding(h, m)
            assert -1.0 <= s <= 1.0, f"sin out of range at {h}:{m:02d}"
            assert -1.0 <= c <= 1.0, f"cos out of range at {h}:{m:02d}"


def test_time_encoding_periodicity():
    """00:00 and 24:00 (= 00:00 next day) must give same encoding."""
    s0, c0 = _time_encoding(0, 0)
    # After full 24h cycle: frac = 1.0 → sin(2π) = 0, cos(2π) = 1 = sin(0), cos(0)
    assert abs(s0) < 1e-10
    assert abs(c0 - 1.0) < 1e-10


def test_hour_of_day_range(feature_df):
    """hour_of_day must be 0–23."""
    assert feature_df["hour_of_day"].between(0, 23).all(), \
        "hour_of_day outside [0, 23]"


def test_steps_last_1h_non_negative(feature_df):
    assert (feature_df["steps_last_1h"] >= 0).all(), "Negative step sum"


def test_latest_glucose_equals_raw(feature_df):
    """latest_glucose must mirror glucose_mgdl exactly."""
    assert (feature_df["latest_glucose"] == feature_df["glucose_mgdl"]).all()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. INFERENCE FEATURES — identical output to training
# ═══════════════════════════════════════════════════════════════════════════════

def _make_readings(n=20, base=120.0):
    """Generate a list of synthetic reading dicts for inference testing."""
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    readings = []
    for i in range(n):
        ts = (now - timedelta(minutes=10 * (n - i))).isoformat()
        readings.append({
            "timestamp":      ts,
            "glucose_mgdl":   base + i * 2.0,
            "heart_rate_bpm": 75.0 + i * 0.5,
            "sleep_hours":    7.0 if i == 0 else None,
            "steps":          100 + i * 10,
            "activity_score": 0.5,
        })
    return readings


def _make_static():
    return {
        "hba1c": 7.2, "age": 52, "bmi": 28.5,
        "diabetes_type": "Type 2", "gender": "Female",
    }


def test_inference_returns_all_feature_cols():
    feats = extract_inference_features(_make_readings(), _make_static())
    missing = [c for c in FEATURE_COLS if c not in feats]
    assert missing == [], f"Inference missing features: {missing}"


def test_inference_feature_order_matches_training():
    """dict keys must cover exactly FEATURE_COLS in the same order."""
    feats = extract_inference_features(_make_readings(), _make_static())
    keys = list(feats.keys())
    assert keys == FEATURE_COLS, (
        f"Feature order mismatch.\nExpected: {FEATURE_COLS}\nGot:      {keys}"
    )


def test_inference_no_nan_values():
    feats = extract_inference_features(_make_readings(), _make_static())
    nans = [k for k, v in feats.items() if math.isnan(float(v))]
    assert nans == [], f"NaN in inference features: {nans}"


def test_inference_latest_glucose_correct():
    rdgs = _make_readings(n=15, base=100.0)
    feats = extract_inference_features(rdgs, _make_static())
    expected = rdgs[-1]["glucose_mgdl"]
    assert abs(feats["latest_glucose"] - expected) < 0.01


def test_inference_diabetes_flag():
    static_diabetic   = {**_make_static(), "diabetes_type": "Type 2"}
    static_nodiabetes = {**_make_static(), "diabetes_type": None}
    f1 = extract_inference_features(_make_readings(), static_diabetic)
    f2 = extract_inference_features(_make_readings(), static_nodiabetes)
    assert f1["diabetes_flag"] == 1.0
    assert f2["diabetes_flag"] == 0.0


def test_inference_gender_encoding():
    sm = {**_make_static(), "gender": "Male"}
    sf = {**_make_static(), "gender": "Female"}
    fm = extract_inference_features(_make_readings(), sm)
    ff = extract_inference_features(_make_readings(), sf)
    assert fm["gender_male"] == 1.0
    assert ff["gender_male"] == 0.0


def test_inference_raises_on_empty_readings():
    with pytest.raises(ValueError, match="No recent readings"):
        extract_inference_features([], _make_static())


def test_inference_values_match_training_features(synthetic_data):
    """No train/serve skew: features from a 24 h window equal the training row for the same moment."""
    patients, readings = synthetic_data
    p = patients[0]
    sub = readings[readings.patient_id == p["patient_id"]].sort_values("timestamp").reset_index(drop=True)
    train = build_features_from_df(sub).sort_values("timestamp").reset_index(drop=True)
    cols = ["timestamp", "glucose_mgdl", "heart_rate_bpm", "sleep_hours", "steps", "activity_level", "activity_score"]
    for i in (300, 500, 700):
        window = sub.iloc[i - 143:i + 1][cols].copy()
        window["timestamp"] = window["timestamp"].astype(str)
        inf = extract_inference_features(window.to_dict("records"), p)
        for c in FEATURE_COLS:
            assert abs(inf[c] - float(train.loc[i, c])) < 1e-6, (c, i, inf[c], train.loc[i, c])


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GLUCOSE TREND
# ═══════════════════════════════════════════════════════════════════════════════

def _rdg(glucose):
    return {"glucose_mgdl": glucose}


def test_glucose_trend_rising():
    rdgs = [_rdg(100), _rdg(110), _rdg(120), _rdg(130), _rdg(145)]
    assert glucose_trend(rdgs) == "rising"


def test_glucose_trend_falling():
    rdgs = [_rdg(150), _rdg(140), _rdg(130), _rdg(120), _rdg(110)]
    assert glucose_trend(rdgs) == "falling"


def test_glucose_trend_stable():
    rdgs = [_rdg(110), _rdg(112), _rdg(109), _rdg(111)]
    assert glucose_trend(rdgs) == "stable"


def test_glucose_trend_single_reading():
    assert glucose_trend([_rdg(120)]) == "stable"


def test_glucose_trend_empty():
    assert glucose_trend([]) == "stable"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DATA VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_validation_removes_duplicates(synthetic_data):
    patients, readings = synthetic_data
    # Inject 5 duplicates
    duped = pd.concat([readings, readings.head(5)], ignore_index=True)
    cleaned, report = validate_readings(duped, patients)
    assert report["checks"]["duplicate_patient_timestamp_pairs"] == 5
    assert len(cleaned) == len(readings)  # duplicates removed


def test_validation_flags_implausible_glucose(synthetic_data):
    patients, readings = synthetic_data
    corrupted = readings.copy()
    corrupted.loc[0, "glucose_mgdl"] = 9999.0   # impossible value
    corrupted.loc[1, "glucose_mgdl"] = -10.0     # negative
    cleaned, report = validate_readings(corrupted, patients)
    assert report["checks"]["glucose_implausible_count"] >= 2
    # Values must have been corrected (not zero-filled)
    assert (cleaned["glucose_mgdl"] >= 30).all()
    assert (cleaned["glucose_mgdl"] <= 600).all()


def test_validation_reports_target_distribution(synthetic_data):
    patients, readings = synthetic_data
    _, report = validate_readings(readings, patients)
    dist = report["target_distribution"]
    assert "no_spike_0" in dist
    assert "spike_1"    in dist
    assert "prevalence" in dist
    assert 0.0 <= dist["prevalence"] <= 1.0


def test_validation_raw_count_preserved(synthetic_data):
    patients, readings = synthetic_data
    _, report = validate_readings(readings, patients)
    assert report["raw_record_count"] == len(readings)


def test_validation_orphan_patient_excluded(synthetic_data):
    patients, readings = synthetic_data
    # Add readings for a patient not in EHR
    ghost_rows = readings.head(5).copy()
    ghost_rows["patient_id"] = "GHOST001"
    corrupted = pd.concat([readings, ghost_rows], ignore_index=True)
    cleaned, report = validate_readings(corrupted, patients)
    assert "GHOST001" in report["checks"]["orphan_reading_patients"]
    assert "GHOST001" not in cleaned["patient_id"].values


def test_validation_missing_value_pct_keys(synthetic_data):
    patients, readings = synthetic_data
    _, report = validate_readings(readings, patients)
    assert "glucose_mgdl" in report["missing_value_pct"]
    assert "heart_rate_bpm" in report["missing_value_pct"]


def test_validation_assumptions_documented(synthetic_data):
    patients, readings = synthetic_data
    _, report = validate_readings(readings, patients)
    assert len(report["assumptions"]) >= 3


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TARGET LABEL INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════════

def test_spike_target_valid_values(synthetic_data):
    _, readings = synthetic_data
    valid = {-1, 0, 1}
    bad = set(readings["spike_target"].unique()) - valid
    assert bad == set(), f"Invalid spike_target values: {bad}"


def test_spike_target_last_rows_are_minus1(synthetic_data):
    """Last HORIZON_STEPS rows per patient should have spike_target == -1."""
    from app.data_generator import HORIZON_STEPS
    _, readings = synthetic_data
    for pid, grp in readings.groupby("patient_id"):
        grp = grp.sort_values("timestamp")
        tail = grp.tail(HORIZON_STEPS)
        assert (tail["spike_target"] == -1).all(), \
            f"Patient {pid}: last {HORIZON_STEPS} rows should all have spike_target=-1"


def test_spike_target_consistent_with_future_glucose(synthetic_data):
    """
    For each row with spike_target=1, glucose at HORIZON_STEPS steps ahead
    must exceed SPIKE_THRESHOLD. For spike_target=0 it must not.
    """
    from app.data_generator import HORIZON_STEPS, SPIKE_THRESHOLD_MGDL
    _, readings = synthetic_data
    for pid, grp in readings.groupby("patient_id"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        for i, row in grp.iterrows():
            fut_i = i + HORIZON_STEPS
            if row["spike_target"] == -1:
                continue
            if fut_i >= len(grp):
                continue
            fut_glucose = grp.loc[fut_i, "glucose_mgdl"]
            if row["spike_target"] == 1:
                assert fut_glucose > SPIKE_THRESHOLD_MGDL, (
                    f"P{pid} row {i}: spike_target=1 but future glucose={fut_glucose:.1f}"
                )
            else:
                assert fut_glucose <= SPIKE_THRESHOLD_MGDL, (
                    f"P{pid} row {i}: spike_target=0 but future glucose={fut_glucose:.1f}"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. PATIENT-AWARE SPLIT — no cross-patient contamination
# ═══════════════════════════════════════════════════════════════════════════════

def test_patient_aware_split_no_overlap():
    """Train and test patient sets must be disjoint."""
    from sklearn.model_selection import GroupShuffleSplit
    patients = generate_patients()[:20]
    readings = generate_readings(patients)
    from app.features import build_features_from_df
    df = build_features_from_df(readings)
    df = df[df["spike_target"] != -1]

    groups = df["patient_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, df["spike_target"], groups))

    train_patients = set(groups.iloc[train_idx])
    test_patients  = set(groups.iloc[test_idx])
    overlap = train_patients & test_patients
    assert overlap == set(), f"Patient leakage: {overlap}"


def test_no_temporal_leakage_in_features():
    """
    At any timestep t, no feature value should depend on t+1 or later.
    Proxy test: for a patient with monotonically increasing glucose,
    rolling_mean at step i must be less than glucose[i] (since it uses past data).
    """
    from datetime import datetime, timedelta, timezone
    # Strictly increasing glucose
    n = 20
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = [{
        "patient_id":     "TESTONLY",
        "timestamp":      (base_time + timedelta(minutes=10 * i)).isoformat(),
        "glucose_mgdl":   float(100 + i * 5),  # strictly increasing
        "heart_rate_bpm": 75.0,
        "sleep_hours":    7.0 if i == 0 else None,
        "steps":          100,
        "activity_level": "light",
        "activity_score": 0.5,
        "spike_target":   0,
        "age": 40, "bmi": 25.0, "hba1c": 5.5,
        "diabetes_flag": 0, "gender_male": 1,
    } for i in range(n)]

    df_raw = pd.DataFrame(rows)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True)
    df_feat = build_features_from_df(df_raw).sort_values("timestamp").reset_index(drop=True)

    # For a strictly increasing series, rolling_mean_2h at row i (using shift=1)
    # must be LESS than glucose[i] (since past readings are all lower)
    for i in range(2, n):
        mean_2h  = df_feat.loc[i, "glucose_rolling_mean_2h"]
        current  = df_feat.loc[i, "glucose_mgdl"]
        assert mean_2h < current, (
            f"Row {i}: rolling_mean_2h ({mean_2h:.1f}) >= current glucose ({current:.1f}) — "
            "suggests future data was included"
        )
