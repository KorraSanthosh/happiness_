"""
Real-data Digital Twin service (v7) behind /api/v7 - used by the dashboard.

For a chosen moment in a demo patient's recording, the service runs the REAL pipeline live:
    raw 5-min stream (CGM, insulin, carbs, HR, steps) up to that moment  +  static EHR
        -> real_data_pipeline.build_features  (past-only features, same code as training)
        -> 30 min / 60 min / 2 h models (trained WITHOUT the demo patients)
        -> spike probabilities, alert levels, key signals, and (replay) what actually happened next.

Data: backend/data/twin_v7 (built by `python -m app.build_twin_demo`, CC BY 4.0 sources only).
Models: 30 min / 1 h are v7. The 2 h model is v9 when backend/models/twin_v9 exists (built by
`app.build_twin_v9`; it also reads estimated Shanghai meal carbs and oral-drug doses from
streams_v9_extra.parquet), otherwise v7. Delete backend/models/twin_v9 to roll back to v7.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app import real_data_pipeline as P
from app import user_upload as U

router = APIRouter(prefix="/api/v7", tags=["Digital Twin v7 (real data)"])

BACKEND = Path(__file__).resolve().parents[1]
PACK_DIR = BACKEND / "data" / "twin_v7"
MODEL_DIR = BACKEND / "models" / "twin_v7"
MODEL_DIR_V9 = BACKEND / "models" / "twin_v9"     # optional: v9 2 h model (rollback = delete this folder)
EXTRA_PATH = PACK_DIR / "streams_v9_extra.parquet"
SHANGHAI = ("sh_t1", "sh_t2")
CARB_FEATURES = ["carbs_now", "carbs_30m", "carbs_1h", "carbs_2h", "carbs_4h", "cob", "since_meal", "last_meal_size", "cob_per_iob"]
HORIZONS = ("30m", "60m", "2h")
HORIZON_BINS = {"30m": 2, "60m": 4, "2h": 8}
HISTORY_DAYS = 15                     # >= 14 days (v9 same-time-of-day spike rate) + 24 h stats + margin

EHR_DISPLAY = {   # feature -> (label, unit, decimals)
    "ehr_age": ("Age", "years", 0), "ehr_bmi": ("BMI", "kg/m²", 1), "ehr_duration_y": ("Diabetes duration", "years", 1),
    "ehr_hba1c_pct": ("HbA1c", "%", 1), "ehr_glycated_albumin": ("Glycated albumin", "%", 1),
    "ehr_fpg": ("Fasting plasma glucose", "mg/dL", 0), "ehr_ppg_2h": ("2-h post-meal glucose", "mg/dL", 0),
    "ehr_cpep_fasting": ("Fasting C-peptide", "nmol/L", 2), "ehr_cpep_2h": ("2-h C-peptide", "nmol/L", 2),
    "ehr_tchol": ("Total cholesterol", "mmol/L", 2), "ehr_tg": ("Triglycerides", "mmol/L", 2),
    "ehr_hdl": ("HDL cholesterol", "mmol/L", 2), "ehr_ldl": ("LDL cholesterol", "mmol/L", 2),
    "ehr_creatinine": ("Creatinine", "µmol/L", 0), "ehr_egfr": ("eGFR", "mL/min/1.73m²", 0),
}
EHR_FLAGS = {
    "ehr_hypertension": "Hypertension", "ehr_hyperlipidemia": "Hyperlipidemia", "ehr_smoker": "Smoker",
    "ehr_drinker": "Alcohol use", "ehr_on_insulin": "Insulin", "ehr_on_long_acting": "Long-acting insulin",
    "ehr_on_metformin": "Metformin", "ehr_on_sulfonylurea": "Sulfonylurea", "ehr_on_alpha_glucosidase": "α-glucosidase inhibitor",
    "ehr_on_dpp4": "DPP-4 inhibitor", "ehr_on_sglt2": "SGLT2 inhibitor", "ehr_on_glp1": "GLP-1 agonist", "ehr_on_tzd": "Thiazolidinedione",
}
SOURCE_SHORT = {"sh_t2": "Shanghai T2DM", "sh_t1": "Shanghai T1DM", "azt1d": "AZT1D", "uom": "T1D-UOM", "hupa": "HUPA-UCM",
                "upload": "Your upload"}
MAX_UPLOADS = 20                      # uploads live in memory only; the oldest is dropped beyond this

_cache: Dict[str, Any] = {}


# ── loading ───────────────────────────────────────────────────────────────────

def _ensure() -> None:
    if _cache:
        return
    if not (PACK_DIR / "streams.parquet").exists() or not (MODEL_DIR / "model_2h.joblib").exists():
        raise HTTPException(503, "Digital Twin v7 demo pack not built. Run: cd backend && python -m app.build_twin_demo")
    streams = pd.read_parquet(PACK_DIR / "streams.parquet")
    people = {p["patient_id"]: p for p in json.load(open(PACK_DIR / "patients.json"))}
    shown = {pid: g.sort_values("step").reset_index(drop=True) for pid, g in streams.groupby("patient_id")}
    _cache["people"] = people
    _cache["models"] = {h: joblib.load(MODEL_DIR / f"model_{h}.joblib") for h in HORIZONS}
    if (MODEL_DIR_V9 / "model_2h.joblib").exists():
        _cache["models"]["2h"] = joblib.load(MODEL_DIR_V9 / "model_2h.joblib")
        if (MODEL_DIR_V9 / "demo_bins.json").exists():   # example moments re-picked with the v9 model
            for pid, b in json.load(open(MODEL_DIR_V9 / "demo_bins.json")).items():
                if pid in people:
                    people[pid]["demo_bin"] = b
    if EXTRA_PATH.exists():                    # model inputs only; the chart keeps showing the recorded stream
        x = pd.read_parquet(EXTRA_PATH)
        streams = streams.merge(x[["segment", "step", "carbs", "oral_med"]], on=["segment", "step"], how="left", suffixes=("", "_x"))
        streams["carbs"] = streams["carbs"].fillna(streams.pop("carbs_x"))
    _cache["streams"] = {pid: g.sort_values("step").reset_index(drop=True) for pid, g in streams.groupby("patient_id")}
    _cache["bins"] = {pid: P.to_15min(s) for pid, s in shown.items()}
    for name in ("demo_evaluation", "summary"):
        f = MODEL_DIR / f"{name}.json"
        _cache[name] = json.load(open(f)) if f.exists() else {}


def _person(pid: str):
    _ensure()
    if pid not in _cache["people"]:
        raise HTTPException(404, f"Patient '{pid}' not in the demo pack.")
    return _cache["people"][pid], _cache["streams"][pid], _cache["bins"][pid]


def _label(clock_min: float, day: int) -> str:
    c = int(clock_min) % 1440
    return f"Day {day + 1} · {c // 60:02d}:{c % 60:02d}"


def _days(bins: pd.DataFrame) -> np.ndarray:
    """Day counter that increments at each midnight crossing of the 15-min clock."""
    wrap = np.r_[0, (np.diff(bins["clock_min"].to_numpy()) < 0).astype(int)]
    return np.cumsum(wrap)


def _interesting_bin(bins: pd.DataFrame) -> int:
    """Default 'now': first moment after day 3 with normal glucose and a real spike within 2 h (a good demo case)."""
    g, gmax = bins["glucose"].to_numpy(), bins["glucose_max"].to_numpy()
    for b in range(min(3 * P.DAY, len(bins) - 9), len(bins) - 9):
        if not np.isnan(g[b]) and g[b] <= 160 and np.nanmax(gmax[b + 1:b + 9]) > 180:
            return int(b)
    return int(min(3 * P.DAY, len(bins) - 9))


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.get("/patients")
def list_patients():
    _ensure()
    out = []
    for pid, p in _cache["people"].items():
        e = p.get("ehr", {})
        out.append({"patient_id": pid, "display_id": pid.split("_")[-1], "source": p["source"],
                    "dataset": SOURCE_SHORT.get(p["source"], p["source"]), "diabetes_type": p["diabetes_type"],
                    "age": e.get("ehr_age"), "sex": None if e.get("ehr_male") is None else ("M" if e["ehr_male"] == 1 else "F"),
                    "hba1c": e.get("ehr_hba1c_pct"), "days": p["days"], "has_ehr": "ehr_age" in e})
    out.sort(key=lambda r: (r["dataset"] != "Your upload", r["dataset"] != "Shanghai T2DM", r["dataset"], r["display_id"]))
    return {"patients": out, "total": len(out)}


@router.get("/patients/{pid}")
def patient_detail(pid: str):
    p, stream, bins = _person(pid)
    e = p.get("ehr", {})
    labs = [{"name": lab, "value": round(e[k], dec), "unit": unit} for k, (lab, unit, dec) in EHR_DISPLAY.items()
            if e.get(k) is not None]
    flags = [name for k, name in EHR_FLAGS.items() if e.get(k) == 1]
    return {
        "patient_id": pid, "display_id": pid.split("_")[-1], "dataset": SOURCE_SHORT.get(p["source"], p["source"]),
        "dataset_citation": p["dataset"], "source": p["source"],
        "diabetes_type": p["diabetes_type"], "days": p["days"], "n_bins": int(len(bins)),
        "default_bin": _last_reading_bin(bins) if p.get("uploaded") else
                       p.get("demo_bin") if p.get("demo_bin") is not None else _interesting_bin(bins),
        "uploaded": bool(p.get("uploaded")), "upload_summary": p.get("upload_summary"),
        "default_is_example_warning": p.get("demo_bin") is not None, "min_bin": int(P.MIN_HISTORY_STEPS),
        "ehr": {"available": "ehr_age" in e,
                "sex": None if e.get("ehr_male") is None else ("Male" if e["ehr_male"] == 1 else "Female"),
                "complications": None if e.get("ehr_n_complications") is None else int(e["ehr_n_complications"]),
                "labs": labs, "conditions_and_drugs": flags},
    }


@router.get("/patients/{pid}/stream")
def patient_stream(pid: str, bin: int = Query(..., ge=0), hours: int = Query(12, ge=1, le=336),
                   future: bool = Query(True)):
    _, _, bins = _person(pid)
    if bin >= len(bins):
        raise HTTPException(422, f"bin must be < {len(bins)}")
    days = _days(bins)
    lo, hi = max(0, bin - hours * 4 + 1), min(len(bins) - 1, bin + (8 if future else 0))
    rows = []
    for b in range(lo, hi + 1):
        r = bins.iloc[b]
        nz = lambda v: None if pd.isna(v) else round(float(v), 2)
        rows.append({"bin": b, "time": _label(r.clock_min, int(days[b])), "future": b > bin,
                     "glucose": nz(r.glucose), "glucose_max": nz(r.glucose_max), "insulin": nz(r.insulin),
                     "carbs": nz(r.carbs), "steps": nz(r.steps), "hr": nz(r.hr)})
    return {"patient_id": pid, "now_bin": bin, "now": _label(bins.clock_min.iloc[bin], int(days[bin])), "points": rows}


class Scenario(BaseModel):
    """Optional 'daily context' entered by the clinician, applied at the chosen moment ("what if, right now").
    Only inputs the models were trained on are used; anything else is reported back as ignored."""
    carbs_g: Optional[float] = Field(None, ge=0, le=300, description="meal eaten now, grams of carbohydrate")
    insulin_u: Optional[float] = Field(None, ge=0, le=50, description="rapid-acting insulin taken now, units")
    oral_tablets: Optional[int] = Field(None, ge=0, le=5, description="number of oral glucose-lowering drugs taken now")


def _feature_row(p: Dict[str, Any], stream: pd.DataFrame, bin: int, sc: Optional[Scenario] = None) -> pd.DataFrame:
    end = 3 * bin + 2                                                   # last raw 5-min step of the 'now' bin
    start = max(0, ((end + 1 - HISTORY_DAYS * 288) // 3) * 3)          # keep 15-min bins aligned
    raw = stream[(stream.step >= start) & (stream.step <= end)].copy()
    if sc is not None:                                                  # the scenario happens in the 'now' step
        last = raw.index[raw.step == end]
        for col, v in (("carbs", sc.carbs_g), ("insulin", sc.insulin_u), ("oral_med", sc.oral_tablets)):
            if v:
                if col not in raw.columns:
                    raw[col] = np.nan
                raw.loc[last, col] = raw.loc[last, col].fillna(0) + float(v)
    raw["step"] = raw["step"] - start
    row = P.build_features(raw).iloc[[-1]].copy()
    if pd.isna(row["glucose_raw"].iloc[0]):
        raise HTTPException(422, "No CGM reading at this moment (sensor gap) - pick another time.")
    for k, v in p.get("ehr", {}).items():                              # static EHR stream
        row[k] = np.nan if v is None else v
    return row


def _score(p: Dict[str, Any], row: pd.DataFrame, bins: pd.DataFrame, bin: int) -> Dict[str, Any]:
    out = {}
    for h in HORIZONS:
        b = _cache["models"][h]
        x = row.reindex(columns=b["feature_cols"]).astype(float)
        if b.get("mask_shanghai_carbs", True) and p["source"] in SHANGHAI:   # v7 never saw Shanghai meal carbs
            x.loc[:, x.columns.intersection(CARB_FEATURES)] = np.nan
        prob = float(b["model"].predict_proba(x)[0, 1])
        ta, ts = b["threshold_accuracy_mode"], b["threshold_safety_mode"]
        level = "spike likely" if prob >= ta else ("watch" if prob >= ts else "unlikely")
        nb = HORIZON_BINS[h]
        fut = bins["glucose_max"].iloc[bin + 1: bin + 1 + nb]
        actual = None if fut.notna().sum() < max(1, int(np.ceil(0.75 * nb))) else {
            "max_glucose": round(float(fut.max()), 1), "spiked": bool(fut.max() > P.SPIKE_MGDL)}
        out[h] = {"probability": round(prob, 4), "level": level, "model_version": b.get("version"),
                  "threshold_accuracy_mode": ta, "threshold_safety_mode": ts, "actual": actual}
        if "threshold_early_warning" in b:      # v9 alert policy: only warn while glucose is still <= 180
            tw = b["threshold_early_warning"]
            out[h]["early_warning"] = {
                "active": bool(prob >= tw and row["glucose"].iloc[0] <= P.SPIKE_MGDL), "threshold": tw,
                "rule": b.get("early_warning_rule")}
    return out


def _scenario_use(p: Dict[str, Any], sc: Scenario) -> Dict[str, Any]:
    """Which scenario inputs each horizon's model can actually see for this patient."""
    used, ignored = {}, []
    for h in HORIZONS:
        b = _cache["models"][h]
        cols, mask = set(b["feature_cols"]), b.get("mask_shanghai_carbs", True) and p["source"] in SHANGHAI
        u = []
        if sc.carbs_g and not mask and cols & set(CARB_FEATURES):
            u.append("carbs")
        if sc.insulin_u:
            u.append("insulin")
        if sc.oral_tablets and "oral_med_2h" in cols and p["source"] in SHANGHAI:
            u.append("oral tablets")
        used[h] = u
    if sc.oral_tablets and p["source"] not in SHANGHAI:
        ignored.append("oral tablets: the models only learned tablet timing from the Shanghai records")
    if sc.carbs_g and not any("carbs" in u for u in used.values()):
        ignored.append("carbs: no model learned meals for this dataset")
    return {"used_by_horizon": used, "ignored": ignored}


@router.post("/predict/{pid}")
def predict(pid: str, bin: int = Query(..., ge=0), scenario: Optional[Scenario] = Body(None)):
    p, stream, bins = _person(pid)
    if bin < P.MIN_HISTORY_STEPS or bin >= len(bins):
        raise HTTPException(422, f"bin must be in [{P.MIN_HISTORY_STEPS}, {len(bins) - 1}]")
    t0 = time.perf_counter()
    row = _feature_row(p, stream, bin)
    out: Dict[str, Any] = {"patient_id": pid, "bin": bin, "horizons": _score(p, row, bins, bin)}
    if scenario is not None and any((scenario.carbs_g, scenario.insulin_u, scenario.oral_tablets)):
        out["scenario"] = {"inputs": scenario.dict(), "horizons": _score(p, _feature_row(p, stream, bin, scenario), bins, bin),
                           **_scenario_use(p, scenario),
                           "note": "What-if estimate: the models learned from logged meals and doses; scenario answers "
                                   "are not separately validated. The replay outcome is what really happened."}
    f = row.iloc[0]
    val = lambda k, d=1: None if k not in f or pd.isna(f[k]) else round(float(f[k]), d)
    out["signals"] = [
        {"name": "Current glucose", "value": val("glucose", 0), "unit": "mg/dL"},
        {"name": "Trend (last 30 min)", "value": val("g_slope_30m", 0), "unit": "mg/dL per hour"},
        {"name": "Projected in 30 min", "value": val("g_proj_30m", 0), "unit": "mg/dL"},
        {"name": "Spikes at this time, last 7 days", "value": None if val("p_same_time_spike_7d", 2) is None else round(100 * val("p_same_time_spike_7d", 2)), "unit": "%"},
        {"name": "Mean glucose, last 24 h", "value": val("p_mean_24h", 0), "unit": "mg/dL"},
        {"name": "Insulin on board", "value": val("iob", 1), "unit": "U"},
        {"name": "Carbs on board" + (" (estimated from meal log)" if p["source"] in SHANGHAI and EXTRA_PATH.exists() else ""),
         "value": val("cob", 0), "unit": "g"},
        {"name": "HbA1c (EHR)", "value": val("ehr_hba1c_pct", 1), "unit": "%"},
    ]
    out["inference_ms"] = round(1000 * (time.perf_counter() - t0), 1)
    out["note"] = "Models never saw this patient during training. Research prototype - not for clinical use."
    return out


def _window_stats(g: np.ndarray) -> Dict[str, Any]:
    v = g[~np.isnan(g)]
    if len(v) == 0:
        return {"readings": 0}
    r = lambda x: round(float(x), 1)
    return {"readings": int(len(v)), "coverage_pct": r(100 * len(v) / len(g)), "mean": r(v.mean()), "sd": r(v.std()),
            "cv_pct": r(100 * v.std() / v.mean()), "gmi_pct": r(3.31 + 0.02392 * v.mean()),
            "tir_pct": r(100 * ((v >= 70) & (v <= 180)).mean()), "tar_pct": r(100 * (v > 180).mean()),
            "tbr_pct": r(100 * (v < 70).mean()), "max": r(v.max()), "min": r(v.min())}


@router.get("/patients/{pid}/overview")
def patient_overview(pid: str, bin: int = Query(..., ge=0)):
    """Data completeness and glucose summary up to the chosen moment (nothing after 'now' is used)."""
    p, stream, bins = _person(pid)
    if bin >= len(bins):
        raise HTTPException(422, f"bin must be < {len(bins)}")
    g = bins["glucose"].to_numpy(float)
    day = slice(max(0, bin - P.DAY + 1), bin + 1)
    raw = stream[(stream.step >= 3 * (bin - P.DAY + 1)) & (stream.step <= 3 * bin + 2)]
    has = lambda c: bool(c in raw.columns and raw[c].notna().any())
    sh = p["source"] in SHANGHAI
    ehr = p.get("ehr", {})
    items = [
        {"name": "Glucose (CGM)", "available": bool(np.isfinite(g[day]).any()),
         "detail": f"{_window_stats(g[day]).get('coverage_pct', 0):.0f} % of the last 24 h"},
        {"name": "Insulin", "available": has("insulin"), "detail": "recorded" if has("insulin") else "not recorded"},
        {"name": "Meals (carbs)", "available": has("carbs"),
         "detail": ("estimated from the meal log" if sh else "logged") if has("carbs") else "not recorded"},
        {"name": "Oral tablets", "available": sh and has("oral_med"),
         "detail": "from the medication log" if sh and has("oral_med") else "not recorded for this dataset"},
        {"name": "Heart rate", "available": has("hr"), "detail": "wearable" if has("hr") else "not recorded"},
        {"name": "Activity (steps)", "available": has("steps"), "detail": "wearable" if has("steps") else "not recorded"},
        {"name": "Clinical record (EHR)", "available": "ehr_age" in ehr,
         "detail": "labs, history, medicines" if "ehr_age" in ehr else "diabetes type only"},
    ]
    gl = g[bin]
    prev = g[bin - 2] if bin >= 2 else np.nan
    return {"patient_id": pid, "bin": bin, "now": _label(bins.clock_min.iloc[bin], int(_days(bins)[bin])),
            "glucose_now": None if np.isnan(gl) else round(float(gl), 1),
            "trend_mgdl_per_min": None if np.isnan(gl) or np.isnan(prev) else round(float((gl - prev) / 30), 2),
            "last_24h": _window_stats(g[day]), "so_far": _window_stats(g[: bin + 1]),
            "completeness": {"items": items, "available": sum(i["available"] for i in items), "total": len(items)},
            "not_used_by_model": ["sleep (no training data had it)", "free-text notes"]}


@router.get("/patients/{pid}/trends")
def patient_trends(pid: str, bin: int = Query(..., ge=0)):
    """Per-day summaries and a time-of-day profile from the start of the record up to the chosen moment."""
    _, _, bins = _person(pid)
    if bin >= len(bins):
        raise HTTPException(422, f"bin must be < {len(bins)}")
    b = bins.iloc[: bin + 1].assign(day=_days(bins)[: bin + 1])
    g = b["glucose"].to_numpy(float)
    above = g > P.SPIKE_MGDL
    onset = np.r_[False, above[1:] & (g[:-1] <= P.SPIKE_MGDL)]          # needs an observed in-range bin before
    days = []
    for d, idx in b.groupby("day").indices.items():
        s = _window_stats(g[idx])
        days.append({"day": int(d) + 1, "spikes": int(onset[idx].sum()), **s})
    hour = (b["clock_min"].to_numpy(float) % 1440) // 60
    prof = []
    for h in range(24):
        v = g[(hour == h) & ~np.isnan(g)]
        prof.append({"hour": h, "n": int(len(v)), **({} if len(v) < 4 else
                     {k: round(float(np.percentile(v, q)), 1) for k, q in (("p10", 10), ("p25", 25), ("p50", 50), ("p75", 75), ("p90", 90))})})
    return {"patient_id": pid, "bin": bin, "days": days, "hourly_profile": prof, "overall": _window_stats(g)}


@router.get("/model-card")
def model_card():
    _ensure()
    return {"summary": _cache.get("summary", {}), "demo_evaluation": _cache.get("demo_evaluation", {}),
            "models": {h: {k: v for k, v in b.items() if k not in ("model", "feature_cols", "excluded_patients")}
                       for h, b in _cache["models"].items()}}


# ── your own data: upload a CGM export (kept in memory only) ─────────────────

def _last_reading_bin(bins: pd.DataFrame) -> int:
    ok = np.flatnonzero(bins["glucose"].notna().to_numpy())
    return int(ok[-1]) if len(ok) else len(bins) - 1


@router.post("/upload")
async def upload_cgm(file: UploadFile = File(...), diabetes_type: str = Form("T2D"), age: Optional[float] = Form(None),
                     sex: Optional[str] = Form(None), hba1c: Optional[float] = Form(None), bmi: Optional[float] = Form(None),
                     years_with_diabetes: Optional[float] = Form(None), on_insulin: bool = Form(False),
                     on_metformin: bool = Form(False), consent: bool = Form(False)):
    """Create a twin from the user's own CGM export. Nothing is written to disk; it disappears on restart or DELETE."""
    _ensure()
    if not consent:
        raise HTTPException(422, "Please confirm you understand this is a research prototype, not medical advice.")
    if diabetes_type not in ("T1D", "T2D"):
        raise HTTPException(422, "diabetes_type must be T1D or T2D")
    for name, v, lo, hi in (("age", age, 1, 110), ("hba1c", hba1c, 3, 20), ("bmi", bmi, 10, 80), ("years_with_diabetes", years_with_diabetes, 0, 90)):
        if v is not None and not lo <= v <= hi:
            raise HTTPException(422, f"{name} should be between {lo} and {hi}")
    try:
        grid, summary = U.parse_cgm_csv(await file.read())
    except U.UploadError as e:
        raise HTTPException(422, str(e))
    pid = f"upload_{uuid.uuid4().hex[:6]}"
    grid = grid.assign(patient_id=pid, segment=f"{pid}_0")
    ehr = {k: None for k in ("ehr_age", "ehr_male", "ehr_bmi", "ehr_duration_y", "ehr_hba1c_pct")}
    ehr.update({"ehr_t2d": 1.0 if diabetes_type == "T2D" else 0.0, "ehr_age": age, "ehr_bmi": bmi,
                "ehr_duration_y": years_with_diabetes, "ehr_hba1c_pct": hba1c,
                "ehr_male": None if sex not in ("M", "F") else float(sex == "M"),
                "ehr_on_insulin": float(on_insulin), "ehr_on_metformin": float(on_metformin)})
    if age is None:                                      # no clinical summary: like the non-Shanghai datasets
        ehr = {"ehr_t2d": ehr["ehr_t2d"]}
    uploads = [k for k, v in _cache["people"].items() if v.get("uploaded")]
    for old in uploads[: max(0, len(uploads) - MAX_UPLOADS + 1)]:
        _forget(old)
    _cache["people"][pid] = {"patient_id": pid, "source": "upload", "segment": f"{pid}_0", "uploaded": True,
                             "dataset": "Uploaded by you; kept in this server's memory only (deleted on restart or when you delete it)",
                             "diabetes_type": diabetes_type, "days": summary["days"], "ehr": ehr, "upload_summary": summary}
    _cache["streams"][pid] = grid
    _cache["bins"][pid] = P.to_15min(grid)
    return {"patient_id": pid, "summary": summary}


def _forget(pid: str) -> None:
    for k in ("people", "streams", "bins"):
        _cache[k].pop(pid, None)


@router.delete("/upload/{pid}")
def delete_upload(pid: str):
    _ensure()
    if not _cache["people"].get(pid, {}).get("uploaded"):
        raise HTTPException(404, "No uploaded data with this id.")
    _forget(pid)
    return {"deleted": pid}


@router.get("/upload/example.csv", response_class=PlainTextResponse)
def example_csv():
    """An example file in the generic format: 3 days of a demo patient (HUPA-UCM, CC BY 4.0), with made-up dates."""
    _ensure()
    s = _cache["streams"]["hupa_0001"].iloc[: 3 * 288]
    t0 = pd.Timestamp("2024-03-01") + pd.Timedelta(minutes=float(s.clock_min.iloc[0]))
    out = pd.DataFrame({"timestamp": (t0 + pd.to_timedelta(5 * np.arange(len(s)), unit="min")).strftime("%Y-%m-%d %H:%M"),
                        "glucose_mg_dl": s.glucose.round(0), "carbs_g": s.carbs.where(s.carbs > 0).round(0),
                        "insulin_units": s.insulin.where(s.insulin > 0).round(2)})
    out = out[out.glucose_mg_dl.notna() | out.carbs_g.notna() | out.insulin_units.notna()]
    return PlainTextResponse(out.to_csv(index=False), headers={"Content-Disposition": "attachment; filename=glucotwin_example.csv"})
