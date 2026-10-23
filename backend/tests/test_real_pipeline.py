"""
Tests for the OhioT1DM + BrisT1D pipeline (app/real_data_pipeline.py): stitching, binning,
causality of features and label definition. Synthetic inputs only, except the last integration test.

Run: cd backend && python -m pytest tests/test_real_pipeline.py -v
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from app import real_data_pipeline as P


# ── helpers ───────────────────────────────────────────────────────────────────

def _grid5(n=300, seed=0, source="ohio", activity=False) -> pd.DataFrame:
    """One synthetic 5-min segment with glucose, carbs, insulin, HR (+ steps/kcal if activity)."""
    rng = np.random.default_rng(seed)
    glucose = np.clip(150 + np.cumsum(rng.normal(0, 6, n)), 50, 350)
    carbs = np.where(rng.random(n) < 0.03, rng.integers(10, 80, n), 0).astype(float)
    insulin = np.where(rng.random(n) < 0.05, rng.uniform(0.5, 6, n), 0.08)
    df = pd.DataFrame({
        "patient_id": "t_1", "source": source, "segment": "t_1_0", "step": np.arange(n),
        "clock_min": (360 + 5 * np.arange(n)) % 1440,
        "glucose": glucose, "carbs": carbs, "insulin": insulin, "hr": 70 + rng.normal(0, 4, n),
    })
    if activity:
        df["steps"] = rng.integers(0, 300, n).astype(float)
        df["kcal"] = rng.uniform(4, 15, n)
    return df


def _lag_name(prefix: str, k: int) -> str:
    m = 5 * k
    return f"{prefix}-{m // 60}:{m % 60:02d}"


def _wide_from_series(series_mmol: np.ndarray, first_row: int, last_row: int, p="p99") -> pd.DataFrame:
    rows = []
    for i in range(first_row, last_row + 1):
        r = {"p_num": p, "time": f"{(6 * 60 + 5 * i) // 60 % 24:02d}:{(6 * 60 + 5 * i) % 60:02d}:00"}
        for k in range(72):
            r[_lag_name("bg", k)] = series_mmol[i - k]
            r[_lag_name("insulin", k)] = 0.1
            r[_lag_name("carbs", k)] = np.nan
            r[_lag_name("hr", k)] = 70.0
        r["bg+1:00"] = series_mmol[i + 12]
        rows.append(r)
    return pd.DataFrame(rows)


# ── stitching ─────────────────────────────────────────────────────────────────

def test_brist_stitch_reconstructs_true_series_without_conflicts():
    rng = np.random.default_rng(1)
    series = np.clip(8 + np.cumsum(rng.normal(0, 0.2, 260)), 3, 20)
    wide = _wide_from_series(series, first_row=71, last_row=200)
    grid, stats = P.stitch_brist_frame(wide)
    assert stats["conflicts"] == 0
    assert grid["segment"].nunique() == 1
    # grid glucose must equal the original series (mg/dL) over the whole reconstructed span
    rec = grid["glucose"].to_numpy()
    expect = series[0:213] * P.MMOL_TO_MGDL                       # rows 71..200 cover indices 0..212
    assert np.allclose(rec[: len(expect)], expect[: len(rec)], atol=1e-6)


def test_brist_large_gap_starts_new_segment():
    rng = np.random.default_rng(2)
    series = 8 + np.cumsum(rng.normal(0, 0.2, 400))
    a = _wide_from_series(series, 71, 120)
    b = _wide_from_series(series, 250, 300)
    b["time"] = pd.to_timedelta(pd.to_timedelta(b["time"]) + pd.Timedelta(minutes=700)).astype(str).str[-8:]
    grid, _ = P.stitch_brist_frame(pd.concat([a, b], ignore_index=True))
    assert grid["segment"].nunique() == 2


# ── 15-minute binning ─────────────────────────────────────────────────────────

def test_to_15min_bin_semantics():
    g = _grid5(n=9)
    g["glucose"] = [100, np.nan, 110, 120, 130, np.nan, np.nan, np.nan, np.nan]
    g["carbs"] = [0, 5, 0, 0, 0, 7, 0, 0, 0]
    g["insulin"] = [0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 1.0, 0, 0]
    b = P.to_15min(g)
    assert len(b) == 3
    assert b["glucose"].iloc[0] == 110 and b["glucose_max"].iloc[0] == 110      # last / max real reading
    assert b["glucose"].iloc[1] == 130 and b["glucose_max"].iloc[1] == 130
    assert np.isnan(b["glucose"].iloc[2])
    assert b["carbs"].tolist() == [5, 7, 0]
    assert b["insulin"].iloc[0] == pytest.approx(0.6) and b["insulin_peak"].iloc[2] == 1.0


def test_to_15min_drops_partial_trailing_bin():
    assert len(P.to_15min(_grid5(n=10))) == 3


# ── causality ─────────────────────────────────────────────────────────────────

def test_features_never_use_future_data():
    g = _grid5(n=360, seed=3)
    base = P.build_features(g)
    cut_raw = 180                                    # mutate everything from raw step 180 on
    g2 = g.copy()
    rng = np.random.default_rng(99)
    for col, lo, hi in (("glucose", 40, 400), ("carbs", 0, 90), ("insulin", 0, 8), ("hr", 40, 150)):
        g2.loc[g2.step >= cut_raw, col] = rng.uniform(lo, hi, (g2.step >= cut_raw).sum())
    mut = P.build_features(g2)
    last_clean_bin = cut_raw // P.BIN - 1            # bins fully before the mutation
    a = base[base.step <= last_clean_bin][P.ENGINEERED_FEATURES]
    b = mut[mut.step <= last_clean_bin][P.ENGINEERED_FEATURES]
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_new_features_never_use_future_data_over_multiple_days():
    """Activity + personal-history features (same-time-yesterday, 24 h stats) must be causal too."""
    g = _grid5(n=3 * 96 * 3, seed=7, activity=True)                       # 3 days of 5-min data
    base = P.build_features(g)
    cut_raw = 3 * 96 * 2 + 30                                             # mutate from day 2 + 2.5 h on
    g2 = g.copy()
    rng = np.random.default_rng(5)
    mk = g2.step >= cut_raw
    for col, lo, hi in (("glucose", 40, 400), ("carbs", 0, 90), ("insulin", 0, 8), ("hr", 40, 150),
                        ("steps", 0, 2000), ("kcal", 0, 40)):
        g2.loc[mk, col] = rng.uniform(lo, hi, mk.sum())
    mut = P.build_features(g2)
    last_clean_bin = cut_raw // P.BIN - 1
    cols = P.V6_FEATURES + P.HISTORY_FEATURES + P.DYNAMICS_FEATURES     # v9 features must be causal too
    a = base[base.step <= last_clean_bin][cols].reset_index(drop=True)
    b = mut[mut.step <= last_clean_bin][cols].reset_index(drop=True)
    assert a["p_same_time_mean_7d"].notna().sum() > 50                   # the lagged features were exercised
    assert a["p_frac_above_72h"].notna().sum() > 50 and a["g_slope_2h"].notna().sum() > 50
    pd.testing.assert_frame_equal(a, b)


def test_unrecorded_carbs_and_insulin_become_unknown_not_zero():
    g = _grid5(n=300)
    g["carbs"] = np.nan
    g["insulin"] = np.nan
    feat = P.build_features(g)
    for c in ("carbs_2h", "cob", "since_meal", "last_meal_size", "iob", "since_bolus", "cob_per_iob"):
        assert feat[c].isna().all(), c
    assert feat["glucose"].notna().any()


def test_missing_activity_columns_give_nan_activity_features():
    feat = P.build_features(_grid5(n=300))                                # no steps/kcal columns (like Ohio)
    assert feat[P.ACTIVITY_FEATURES].isna().all().all()


def test_features_are_independent_across_segments():
    a, b = _grid5(200, seed=4), _grid5(200, seed=5)
    b["segment"], b["patient_id"] = "t_2_0", "t_2"
    both = P.build_features(pd.concat([a, b], ignore_index=True))
    alone = P.build_features(b)
    x = both[both.segment == "t_2_0"][P.ENGINEERED_FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(x, alone[P.ENGINEERED_FEATURES].reset_index(drop=True))


# ── label ─────────────────────────────────────────────────────────────────────

def test_label_is_max_over_next_8_bins_and_excludes_current():
    n = 3 * 40
    g = _grid5(n=n)
    g["glucose"] = 120.0
    g.loc[g.step == 3 * 10 + 1, "glucose"] = 200.0        # one spike inside bin 10
    feat = P.build_features(g).set_index("step")
    # bin 10 is spiking: rows 2..9 see it in their next-8-bin window, row 10 itself and row 1 do not
    for s in range(2, 10):
        assert feat.loc[s, "y"] == 1.0, s
    assert feat.loc[1, "y"] == 0.0 and feat.loc[10, "y"] == 0.0 and feat.loc[11, "y"] == 0.0


def test_short_horizon_labels_look_exactly_30_and_60_minutes_ahead():
    n = 3 * 40
    g = _grid5(n=n)
    g["glucose"] = 120.0
    g.loc[g.step == 3 * 10 + 1, "glucose"] = 200.0        # spike inside bin 10
    feat = P.build_features(g).set_index("step")
    assert feat.loc[[8, 9], "y_30m"].tolist() == [1.0, 1.0] and feat.loc[[7, 10], "y_30m"].tolist() == [0.0, 0.0]
    assert feat.loc[[6, 7, 8, 9], "y_60m"].tolist() == [1.0] * 4 and feat.loc[[5, 10], "y_60m"].tolist() == [0.0, 0.0]
    assert (feat["y_30m"].notna().sum() > feat["y"].notna().sum())   # shorter horizon -> more labelled rows


def test_last_horizon_rows_are_unlabelled_and_unusable():
    feat = P.build_features(_grid5(n=300))
    tail = feat.tail(P.HORIZON_STEPS)
    assert tail["y"].isna().all() and not tail["usable"].any()


def test_sparse_future_coverage_means_no_label():
    g = _grid5(n=300)
    g.loc[(g.step >= 60) & (g.step < 100), "glucose"] = np.nan        # CGM outage
    feat = P.build_features(g).set_index("step")
    assert np.isnan(feat.loc[20, "y"])                                 # window mostly inside the outage


# ── integration with the real data (skipped when raw data is absent) ──────────

@pytest.mark.skipif(not (P.BRIST_DIR / "train.csv").exists() or not P.OHIO_BASE.exists(),
                    reason="raw OhioT1DM/BrisT1D data not present")
def test_real_data_has_all_patients_and_no_stitch_conflicts():
    grid, stats = P.load_all_grids()
    assert stats["conflicts"] == 0
    m = P.modelling_frame(P.build_features(grid))
    assert m.groupby("source")["patient_id"].nunique().to_dict() == {"brist": 9, "ohio": 12}
    assert set(m["y"].unique()) == {0, 1}
    assert not m[P.ENGINEERED_FEATURES].isin([np.inf, -np.inf]).any().any()
