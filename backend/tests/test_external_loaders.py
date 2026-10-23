"""
Tests for app/external_loaders.py harmonisation helpers (synthetic inputs only; no downloaded data needed).

Run: cd backend && python -m pytest tests/test_external_loaders.py -v
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from app import external_loaders as E


def _series(values, start="2024-01-01 00:00", freq="5min"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq=freq), dtype=float)


def test_mask_collinear_removes_linear_interpolation_but_keeps_real_readings():
    # real readings every 15 min, linearly interpolated to 5 min (HUPA-style)
    real = np.array([100.0, 130.0, 115.0, 160.0])
    interp = np.interp(np.arange(10), np.arange(0, 10, 3), real)
    out = E._mask_collinear(interp)
    assert np.allclose(out[::3], real)                      # the true readings survive
    assert np.isnan(out[[1, 2, 4, 5, 7, 8]]).all()          # interpolated points are removed


def test_mask_collinear_leaves_noisy_real_data_mostly_intact():
    rng = np.random.default_rng(0)
    g = np.round(150 + np.cumsum(rng.normal(0, 4, 2000)))
    assert np.isnan(E._mask_collinear(g)).mean() < 0.15


def test_assemble_splits_segments_at_long_cgm_gaps():
    a = _series(np.full(50, 120.0))
    b = _series(np.full(50, 130.0), start="2024-01-02 00:00")              # ~20 h later
    grid = E._assemble("x", "test", pd.concat([a, b]), None, None)
    assert grid["segment"].nunique() == 2
    assert grid.groupby("segment")["glucose"].apply(lambda s: s.notna().all()).all()


def test_assemble_marks_unrecorded_modalities_unknown_and_recorded_events_zero_filled():
    g = _series(np.full(30, 140.0))
    carbs = _series([0.0] * 10 + [45.0] + [0.0] * 19)
    grid = E._assemble("x", "test", g, carbs, None)
    assert grid["insulin"].isna().all()                     # never recorded -> unknown
    assert grid["carbs"].sum() == 45.0 and grid["carbs"].notna().all()
    assert grid[["steps", "kcal", "hr"]].isna().all().all()


def test_assemble_drops_implausible_glucose():
    g = _series([120.0, 5.0, 900.0, 125.0])
    grid = E._assemble("x", "test", g, None, None)
    assert grid["glucose"].dropna().tolist() == [120.0, 125.0]


def test_assemble_zero_ok_keeps_genuine_zero_insulin():
    g = _series(np.full(30, 140.0))
    none = _series(np.zeros(30))
    assert E._assemble("x", "t", g, None, none)["insulin"].isna().all()                       # unrecorded
    assert (E._assemble("x", "t", g, None, none, zero_ok=("insulin",))["insulin"] == 0).all()  # truly none


def test_parse_shanghai_doses():
    assert E._parse_doses("Insulin glargine, 12 IU, Humulin 70/30, 8 IU") == [("insulin glargine", 12.0), ("humulin 70/30", 8.0)]
    assert E._parse_doses("500ml 0.9% sodium chloride, 6 IU Novolin R, 10 ml 10% potassium chloride")[0][1] == 6.0
    assert E._parse_doses(np.nan) == []
