"""
GlucoTwin AI — FastAPI main application.
All data is SYNTHETIC and EXPERIMENTAL. Not for clinical use.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app import data_loader, predictor, services
from app.train_model import train
from app.twin_v7 import router as twin_v7_router

# ── Startup ───────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data and train model on startup if not already done."""
    print("[startup] Loading patient data …")
    data_loader.get_all_patients()          # warm up cache

    if not predictor.model_ready():
        print("[startup] Model not found — training now …")
        train()
    else:
        print("[startup] Model already trained — skipping.")

    yield                                   # app is running

    print("[shutdown] GlucoTwin AI shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="GlucoTwin AI",
    description="Predictive Healthcare Digital Twin API — EXPERIMENTAL, synthetic data only",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the Vite dev server and any local origin
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(twin_v7_router)          # /api/v7 - real-data Digital Twin (see app/twin_v7.py)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _patient_or_404(patient_id: str) -> Dict:
    patient = data_loader.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")
    return patient


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/api/health", tags=["Meta"])
def health_check():
    return {
        "status": "ok",
        "model_ready": predictor.model_ready(),
        "patient_count": data_loader.get_patient_count(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
    }


@app.get("/api/stats", tags=["Meta"])
def dashboard_stats():
    return services.get_dashboard_stats()


@app.get("/api/patients", tags=["Patients"])
def list_patients(
    search: Optional[str] = Query(None, description="Search by name or ID"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    patients = data_loader.get_all_patients()

    if search:
        q = search.lower()
        patients = [
            p for p in patients
            if q in p["name"].lower() or q in p["patient_id"].lower()
        ]

    total = len(patients)
    page = patients[offset: offset + limit]

    summaries = [
        {
            "patient_id": p["patient_id"],
            "name": p["name"],
            "age": p["age"],
            "gender": p["gender"],
            "diabetes_type": p["diabetes_type"],
            "risk_level": p["risk_level"],
            "hba1c": p["hba1c"],
            "last_updated": p["last_updated"],
        }
        for p in page
    ]
    return {"patients": summaries, "total": total, "limit": limit, "offset": offset}


@app.get("/api/patients/{patient_id}", tags=["Patients"])
def get_patient(patient_id: str):
    return _patient_or_404(patient_id)


@app.get("/api/patients/{patient_id}/readings", tags=["Patients"])
def get_readings(
    patient_id: str,
    limit: int = Query(144, ge=1, le=500),
):
    _patient_or_404(patient_id)
    readings = data_loader.get_readings(patient_id, limit=limit)
    return {
        "patient_id": patient_id,
        "readings": readings,
        "total": len(readings),
    }


@app.get("/api/patients/{patient_id}/twin", tags=["DigitalTwin"])
def get_digital_twin(patient_id: str):
    _patient_or_404(patient_id)
    twin = services.build_digital_twin(patient_id)
    if twin is None:
        raise HTTPException(status_code=500, detail="Failed to build Digital Twin.")
    return twin


@app.post("/api/predict/{patient_id}", tags=["Prediction"])
def predict_glucose(patient_id: str):
    patient = _patient_or_404(patient_id)

    if not predictor.model_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not ready. It is being trained. Please retry in a moment.",
        )

    readings = data_loader.get_readings(patient_id, limit=144)     # 24 h: same windows as training
    if not readings:
        raise HTTPException(
            status_code=422,
            detail="No sensor readings available for this patient.",
        )

    try:
        result = predictor.predict(patient_id, readings, patient)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return result


@app.get("/api/model-card", tags=["Meta"])
def get_model_card():
    """
    Return the saved model card — version, metrics, leakage audit, comparison table.
    All metrics are from synthetic data. NOT for clinical use.
    """
    import json
    from pathlib import Path
    card_path = Path(__file__).parent.parent / "models" / "model_card.json"
    metrics_path = Path(__file__).parent.parent / "models" / "metrics.json"

    card = {}
    if card_path.exists():
        with open(card_path) as f:
            card = json.load(f)

    # Merge live metrics.json for freshness
    if metrics_path.exists():
        with open(metrics_path) as f:
            live = json.load(f)
        card["live_test_metrics"] = live.get("test_metrics", {})
        card["live_validation_comparison"] = live.get("validation_comparison", {})
        card["live_feature_importances"] = live.get("feature_importances", {})

    return card

