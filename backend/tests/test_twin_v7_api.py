"""
API tests for the real-data Digital Twin (/api/v7). Skipped when the demo pack has not been built
(python -m app.build_twin_demo).

Run: cd backend && python -m pytest tests/test_twin_v7_api.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import twin_v7
from app.main import app

pytestmark = pytest.mark.skipif(
    not (twin_v7.PACK_DIR / "streams.parquet").exists() or not (twin_v7.MODEL_DIR / "model_2h.joblib").exists(),
    reason="v7 demo pack not built",
)
client = TestClient(app)


@pytest.fixture(scope="module")
def first_patient():
    r = client.get("/api/v7/patients")
    assert r.status_code == 200
    pats = r.json()["patients"]
    assert len(pats) > 0
    return pats[0]["patient_id"]


def test_patient_list_has_both_diabetes_types():
    pats = client.get("/api/v7/patients").json()["patients"]
    assert {p["diabetes_type"] for p in pats} == {"T1D", "T2D"}
    assert all(p["dataset"] in {"Shanghai T2DM", "Shanghai T1DM", "AZT1D", "T1D-UOM", "HUPA-UCM"} for p in pats)


def test_patient_detail_has_ehr_and_timeline(first_patient):
    d = client.get(f"/api/v7/patients/{first_patient}").json()
    assert d["n_bins"] > d["default_bin"] >= d["min_bin"]
    assert "labs" in d["ehr"]


def test_stream_returns_past_and_future_points(first_patient):
    d = client.get(f"/api/v7/patients/{first_patient}").json()
    s = client.get(f"/api/v7/patients/{first_patient}/stream", params={"bin": d["default_bin"]}).json()
    assert any(p["future"] for p in s["points"]) and any(not p["future"] for p in s["points"])
    assert s["now_bin"] == d["default_bin"]


def test_live_prediction_three_horizons(first_patient):
    d = client.get(f"/api/v7/patients/{first_patient}").json()
    r = client.post(f"/api/v7/predict/{first_patient}", params={"bin": d["default_bin"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["horizons"]) == {"30m", "60m", "2h"}
    for h in body["horizons"].values():
        assert 0.0 <= h["probability"] <= 1.0
        assert h["level"] in {"spike likely", "watch", "unlikely"}
    assert any(s["name"] == "Current glucose" and s["value"] is not None for s in body["signals"])


def test_live_features_match_offline_pipeline():
    """Live features (history truncated to HISTORY_DAYS) equal the offline pipeline run on the full record.
    The demo streams are <= 14 days and HISTORY_DAYS is 15, so today the live window is the whole record."""
    import numpy as np
    from app import real_data_pipeline as P
    twin_v7._ensure()
    b = 10 * P.DAY + 5
    pid = next((p for p, s in twin_v7._cache["streams"].items() if len(s) // 3 > b + 1), None)
    if pid is None:
        pytest.skip("no demo patient longer than 10 days")
    stream = twin_v7._cache["streams"][pid]
    full = P.build_features(stream).reset_index(drop=True)
    end = 3 * b + 2
    start = max(0, ((end + 1 - twin_v7.HISTORY_DAYS * 288) // 3) * 3)
    raw = stream[(stream.step >= start) & (stream.step <= end)].copy()
    raw["step"] -= start
    live = P.build_features(raw).iloc[-1]
    # if a window is ever truncated: EWMs have (vanishing) infinite memory and last_meal_size reaches back far
    cols = [c for c in P.V6_FEATURES + P.HISTORY_FEATURES + P.DYNAMICS_FEATURES
            if c not in ("g_ewm_short", "g_ewm_long", "g_ewm_gap", "last_meal_size")]
    np.testing.assert_allclose(live[cols].astype(float).to_numpy(), full.loc[b, cols].astype(float).to_numpy(),
                               rtol=1e-6, atol=1e-6, equal_nan=True)


def test_unknown_patient_404():
    assert client.get("/api/v7/patients/nope").status_code == 404
    assert client.post("/api/v7/predict/nope", params={"bin": 10}).status_code == 404


def test_model_card_summary():
    body = client.get("/api/v7/model-card").json()
    assert set(body["models"]) == {"30m", "60m", "2h"}


v9 = pytest.mark.skipif(not (twin_v7.MODEL_DIR_V9 / "model_2h.joblib").exists(), reason="v9 2 h model not built")


@v9
def test_v9_two_hour_model_and_early_warning_rule():
    twin_v7._ensure()
    b = twin_v7._cache["models"]["2h"]
    assert b["version"].startswith("v9") and b["mask_shanghai_carbs"] is False
    assert twin_v7._cache["models"]["30m"].get("mask_shanghai_carbs", True)          # 30 min / 1 h stay v7
    pid = next(p for p in twin_v7._cache["people"])
    bins = twin_v7._cache["bins"][pid]
    seen = set()
    for k in range(4 * 96, len(bins) - 9, 7):
        if np.isnan(bins.glucose.iloc[k]):
            continue
        h = client.post(f"/api/v7/predict/{pid}", params={"bin": k}).json()["horizons"]["2h"]
        ew = h["early_warning"]
        assert ew["active"] == (h["probability"] >= ew["threshold"] and bins.glucose.iloc[k] <= 180)
        seen.add(bins.glucose.iloc[k] > 180)
        if seen == {True, False}:
            break


@v9
def test_v9_live_features_match_training_frame_for_shanghai():
    """The served v9 inputs (incl. estimated meal carbs and oral-drug doses) equal the v9 training frame."""
    from app import real_data_pipeline as P
    from app import real_data_v9 as V9
    if not (V9.OUT / "frame.parquet").exists():
        pytest.skip("v9 frame not built")
    twin_v7._ensure()
    pid = next(p for p, q in twin_v7._cache["people"].items() if q["source"] == "sh_t2")
    stream = twin_v7._cache["streams"][pid]
    seg = stream.segment.iloc[0]
    cols = [c for c in twin_v7._cache["models"]["2h"]["feature_cols"] if not c.startswith("ehr_")]
    frame = pd.read_parquet(V9.OUT / "frame.parquet", columns=["segment", "step"] + cols)
    frame = frame[frame.segment == seg].set_index("step")
    feat = P.build_features(stream).set_index("step")
    steps = [s for s in frame.index if s >= 4 * P.DAY and s < len(stream) // 3][::50]
    assert steps and stream.oral_med.notna().any()
    np.testing.assert_allclose(feat.loc[steps, cols].astype(float).to_numpy(), frame.loc[steps, cols].astype(float).to_numpy(),
                               rtol=1e-6, atol=1e-6, equal_nan=True)


def test_scenario_what_if_changes_only_models_that_see_the_input(first_patient):
    d = client.get(f"/api/v7/patients/{first_patient}").json()
    base = client.post(f"/api/v7/predict/{first_patient}", params={"bin": d["default_bin"]}).json()
    assert "scenario" not in base
    r = client.post(f"/api/v7/predict/{first_patient}", params={"bin": d["default_bin"]}, json={"carbs_g": 60}).json()
    assert r["horizons"] == base["horizons"]                              # the baseline is untouched
    for h, used in r["scenario"]["used_by_horizon"].items():
        same = r["scenario"]["horizons"][h]["probability"] == base["horizons"][h]["probability"]
        assert same or "carbs" in used                                    # nothing changes through an input the model cannot see
    bad = client.post(f"/api/v7/predict/{first_patient}", params={"bin": d["default_bin"]}, json={"carbs_g": -5})
    assert bad.status_code == 422


def test_overview_and_trends_use_only_the_past(first_patient):
    d = client.get(f"/api/v7/patients/{first_patient}").json()
    b = d["default_bin"]
    o = client.get(f"/api/v7/patients/{first_patient}/overview", params={"bin": b}).json()
    assert o["completeness"]["total"] == len(o["completeness"]["items"]) >= 5
    assert 0 <= o["last_24h"]["tir_pct"] + o["last_24h"]["tar_pct"] + o["last_24h"]["tbr_pct"] <= 100.1
    t = client.get(f"/api/v7/patients/{first_patient}/trends", params={"bin": b}).json()
    t2 = client.get(f"/api/v7/patients/{first_patient}/trends", params={"bin": b + 8}).json()
    assert t["days"][:-1] == t2["days"][:len(t["days"]) - 1]              # earlier complete days do not change
    assert sum(x["readings"] for x in t["days"]) == t["overall"]["readings"] <= b + 1
