"""
Data loader — in-memory store with lazy loading.
Serves as the data access layer for the FastAPI endpoints.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

import pandas as pd

from app.data_generator import load_or_generate

_patients: List[Dict] = []
_patient_map: Dict[str, Dict] = {}
_readings_df: Optional[pd.DataFrame] = None


def _ensure_loaded() -> None:
    global _patients, _patient_map, _readings_df
    if not _patients:
        _patients, _readings_df = load_or_generate()
        _patient_map = {p["patient_id"]: p for p in _patients}


def get_all_patients() -> List[Dict]:
    _ensure_loaded()
    return _patients


def get_patient(patient_id: str) -> Optional[Dict]:
    _ensure_loaded()
    return _patient_map.get(patient_id)


def get_readings(
    patient_id: str,
    limit: int = 144,
) -> List[Dict]:
    _ensure_loaded()
    df = _readings_df
    pf = df[df["patient_id"] == patient_id].tail(limit)
    cols = [
        "timestamp", "glucose_mgdl", "heart_rate_bpm",
        "sleep_hours", "steps", "activity_level", "activity_score",
    ]
    rows = pf[cols].copy()
    rows["timestamp"] = rows["timestamp"].astype(str)
    rows["steps"] = rows["steps"].astype(int)
    records = rows.to_dict(orient="records")
    # Fill None sleep with None (parquet may produce NaN)
    for r in records:
        if pd.isna(r.get("sleep_hours")):
            r["sleep_hours"] = None
        if pd.isna(r.get("heart_rate_bpm")):
            r["heart_rate_bpm"] = None
    return records


def get_patient_count() -> int:
    _ensure_loaded()
    return len(_patients)
