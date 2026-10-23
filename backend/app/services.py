"""
Business logic / service layer for GlucoTwin AI.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app import data_loader, predictor
from app.features import extract_inference_features


def build_digital_twin(patient_id: str) -> Dict[str, Any]:
    """Assemble the full Digital Twin for a patient."""
    patient = data_loader.get_patient(patient_id)
    if not patient:
        return None

    readings = data_loader.get_readings(patient_id, limit=144)

    # Derive features from the last 24 h (same code path and windows as training)
    recent = readings
    features = extract_inference_features(recent, patient)

    # Run prediction
    try:
        prediction = predictor.predict(patient_id, recent, patient)
    except Exception as e:
        prediction = None

    return {
        "patient_id": patient_id,
        "patient_info": patient,
        "recent_readings": readings[-48:],
        "derived_features": features,
        "latest_prediction": prediction,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def get_dashboard_stats() -> Dict[str, Any]:
    patients = data_loader.get_all_patients()
    high = sum(1 for p in patients if p["risk_level"] == "high")
    moderate = sum(1 for p in patients if p["risk_level"] == "moderate")
    low = sum(1 for p in patients if p["risk_level"] == "low")

    # Recent alerts — top 5 high-risk patients
    alerts = [
        {
            "patient_id": p["patient_id"],
            "name": p["name"],
            "risk_level": p["risk_level"],
            "last_updated": p["last_updated"],
        }
        for p in patients
        if p["risk_level"] == "high"
    ][:5]

    # Average last glucose (from data store — compute quickly)
    all_readings = [data_loader.get_readings(p["patient_id"], limit=1) for p in patients[:20]]
    last_glucoses = [r[0]["glucose_mgdl"] for r in all_readings if r]
    avg_glucose = round(sum(last_glucoses) / len(last_glucoses), 1) if last_glucoses else 0.0

    return {
        "total_patients": len(patients),
        "high_risk_count": high,
        "moderate_risk_count": moderate,
        "low_risk_count": low,
        "latest_prediction_time": datetime.now(timezone.utc).isoformat(),
        "avg_glucose": avg_glucose,
        "recent_alerts": alerts,
    }
