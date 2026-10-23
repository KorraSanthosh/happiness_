"""
Automated backend tests for GlucoTwin AI.
Run: cd backend && python -m pytest tests/ -v
"""
import sys
import os

# Ensure the backend package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app import data_loader, predictor
from app.train_model import train

client = TestClient(app)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def ensure_model():
    """Train the model once for the whole test session."""
    if not predictor.model_ready():
        train()
    # Warm up data
    data_loader.get_all_patients()
    yield


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_endpoint():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model_ready" in body
    assert "patient_count" in body
    assert body["patient_count"] >= 100


# ── Patient listing ───────────────────────────────────────────────────────────

def test_list_patients():
    resp = client.get("/api/patients")
    assert resp.status_code == 200
    body = resp.json()
    assert "patients" in body
    assert body["total"] >= 100
    assert len(body["patients"]) > 0


def test_list_patients_search():
    resp = client.get("/api/patients?search=P0001")
    assert resp.status_code == 200
    patients = resp.json()["patients"]
    assert any(p["patient_id"] == "P0001" for p in patients)


def test_list_patients_pagination():
    resp = client.get("/api/patients?limit=10&offset=0")
    assert resp.status_code == 200
    assert len(resp.json()["patients"]) == 10


# ── Patient detail ────────────────────────────────────────────────────────────

def test_get_patient_detail():
    resp = client.get("/api/patients/P0001")
    assert resp.status_code == 200
    body = resp.json()
    assert body["patient_id"] == "P0001"
    assert "age" in body
    assert "diagnoses" in body
    assert "lab_results" in body
    assert "hba1c" in body


def test_get_patient_not_found():
    resp = client.get("/api/patients/P9999")
    assert resp.status_code == 404


# ── Readings ──────────────────────────────────────────────────────────────────

def test_get_readings():
    resp = client.get("/api/patients/P0001/readings")
    assert resp.status_code == 200
    body = resp.json()
    assert "readings" in body
    assert body["total"] > 0
    reading = body["readings"][0]
    assert "timestamp" in reading
    assert "glucose_mgdl" in reading


def test_get_readings_invalid_patient():
    resp = client.get("/api/patients/INVALID/readings")
    assert resp.status_code == 404


# ── Digital Twin ──────────────────────────────────────────────────────────────

def test_digital_twin():
    resp = client.get("/api/patients/P0001/twin")
    assert resp.status_code == 200
    body = resp.json()
    assert body["patient_id"] == "P0001"
    assert "patient_info" in body
    assert "recent_readings" in body
    assert "derived_features" in body
    assert "generated_at" in body


# ── Prediction ────────────────────────────────────────────────────────────────

def test_predict_success():
    resp = client.post("/api/predict/P0001")
    assert resp.status_code == 200
    body = resp.json()
    assert body["patient_id"] == "P0001"
    assert "spike_probability" in body
    assert 0.0 <= body["spike_probability"] <= 1.0
    assert body["risk_category"] in ("low", "moderate", "high")
    assert "feature_importances" in body
    assert len(body["feature_importances"]) > 0
    assert "experimental_status" in body


def test_predict_invalid_patient():
    resp = client.post("/api/predict/P9999")
    assert resp.status_code == 404


# ── Stats ─────────────────────────────────────────────────────────────────────

def test_dashboard_stats():
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "total_patients" in body
    assert "high_risk_count" in body
    assert body["total_patients"] >= 100


# ── Model loading ─────────────────────────────────────────────────────────────

def test_model_inference_directly():
    """Test model predictor module directly."""
    patients = data_loader.get_all_patients()
    patient = patients[0]
    readings = data_loader.get_readings(patient["patient_id"], limit=24)
    result = predictor.predict(patient["patient_id"], readings, patient)
    assert "spike_probability" in result
    assert "risk_category" in result
    assert result["risk_category"] in ("low", "moderate", "high")
