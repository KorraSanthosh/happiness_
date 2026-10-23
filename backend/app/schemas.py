"""
Pydantic schemas for GlucoTwin AI API.
All data is synthetic and for experimental purposes only.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# ── Patient models ────────────────────────────────────────────────────────────

class Diagnosis(BaseModel):
    code: str
    description: str
    diagnosed_date: str


class LabResult(BaseModel):
    test_name: str
    value: float
    unit: str
    reference_range: str
    date: str
    abnormal: bool


class PatientSummary(BaseModel):
    patient_id: str
    name: str
    age: int
    gender: str
    diabetes_type: Optional[str]
    risk_level: str          # low | moderate | high
    last_glucose: float
    last_updated: str


class PatientDetail(BaseModel):
    patient_id: str
    name: str
    age: int
    gender: str
    weight_kg: float
    height_cm: float
    bmi: float
    diabetes_type: Optional[str]
    diagnoses: List[Diagnosis]
    lab_results: List[LabResult]
    hba1c: float
    risk_level: str
    last_updated: str


# ── Sensor / reading models ───────────────────────────────────────────────────

class SensorReading(BaseModel):
    timestamp: str
    glucose_mgdl: float
    heart_rate_bpm: Optional[float]
    sleep_hours: Optional[float]
    steps: Optional[int]
    activity_level: Optional[str]   # sedentary | light | moderate | vigorous


class ReadingsResponse(BaseModel):
    patient_id: str
    readings: List[SensorReading]
    total: int


# ── Feature / Twin models ─────────────────────────────────────────────────────

class DerivedFeatures(BaseModel):
    latest_glucose: float
    glucose_change_1h: float
    glucose_rolling_mean_2h: float
    glucose_rolling_std_2h: float
    heart_rate_mean_1h: float
    sleep_hours_last_night: float
    steps_last_1h: int
    activity_score: float
    time_of_day_sin: float
    time_of_day_cos: float
    hba1c: float
    age: int
    bmi: float
    diabetes_flag: int


class DigitalTwin(BaseModel):
    patient_id: str
    patient_info: PatientDetail
    recent_readings: List[SensorReading]
    derived_features: DerivedFeatures
    latest_prediction: Optional["PredictionResponse"]
    generated_at: str


# ── Prediction models ─────────────────────────────────────────────────────────

class FeatureImportance(BaseModel):
    feature: str
    importance: float
    description: str


class PredictionResponse(BaseModel):
    patient_id: str
    prediction_timestamp: str
    prediction_horizon_hours: int = 2
    spike_probability: float = Field(..., ge=0.0, le=1.0)
    predicted_glucose_mgdl: float
    risk_category: str           # low | moderate | high
    recent_glucose_trend: str    # rising | falling | stable
    feature_importances: List[FeatureImportance]
    model_version: str
    experimental_status: str = "EXPERIMENTAL — Not for clinical use"
    explanation: str


# ── Health check ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    patient_count: int
    timestamp: str
    version: str


# ── Dashboard stats ───────────────────────────────────────────────────────────

class DashboardStats(BaseModel):
    total_patients: int
    high_risk_count: int
    moderate_risk_count: int
    low_risk_count: int
    latest_prediction_time: Optional[str]
    avg_glucose: float
    recent_alerts: List[Dict[str, Any]]
