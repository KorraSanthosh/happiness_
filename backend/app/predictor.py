"""
Inference / prediction module for GlucoTwin AI v2.
Loads the trained pipeline and runs predictions on demand.
Compatible with multi-model artifact saved by train_model v2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from app.features import FEATURE_COLS, extract_inference_features, glucose_trend

MODEL_DIR  = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "glucotwin_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

SPIKE_THRESHOLD_MGDL = 180.0

# Human-readable feature descriptions (updated for v2 features)
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "latest_glucose":          "Current CGM glucose reading (mg/dL)",
    "glucose_lag_1":           "Glucose reading 10 min ago",
    "glucose_lag_3":           "Glucose reading 30 min ago",
    "glucose_change_30min":    "Glucose change over last 30 minutes",
    "glucose_change_1h":       "Glucose change over last 1 hour",
    "glucose_change_2h":       "Glucose change over last 2 hours",
    "glucose_roc_1h":          "Glucose rate of change (mg/dL per hour)",
    "glucose_rolling_mean_1h": "Average glucose over last 1 hour",
    "glucose_rolling_mean_2h": "Average glucose over last 2 hours",
    "glucose_rolling_std_1h":  "Glucose variability over last 1 hour (σ)",
    "glucose_rolling_std_2h":  "Glucose variability over last 2 hours (σ)",
    "glucose_rolling_min_2h":  "Minimum glucose over last 2 hours",
    "glucose_rolling_max_2h":  "Maximum glucose over last 2 hours",
    "glucose_above_threshold": "Current glucose exceeds 180 mg/dL threshold",
    "glucose_in_range":        "Current glucose within target range (70–180 mg/dL)",
    "heart_rate_mean_1h":      "Average heart rate over last hour (bpm)",
    "heart_rate_std_1h":       "Heart rate variability last hour (σ)",
    "heart_rate_max_1h":       "Peak heart rate over last hour (bpm)",
    "steps_last_1h":           "Total step count in last hour",
    "activity_score_mean_1h":  "Average activity intensity last hour",
    "activity_score_max_1h":   "Peak activity intensity last hour",
    "sleep_hours_last_night":  "Sleep duration last night (hours)",
    "time_of_day_sin":         "Time of day — sine encoding",
    "time_of_day_cos":         "Time of day — cosine encoding",
    "hour_of_day":             "Hour of day (0–23)",
    "hba1c":                   "HbA1c — long-term glucose control (%)",
    "age":                     "Patient age (years)",
    "bmi":                     "Body Mass Index (kg/m²)",
    "diabetes_flag":           "Known diabetes diagnosis",
    "gender_male":             "Patient sex (male=1, female=0)",
}

_model_bundle: Dict | None = None


def _load_model() -> Dict:
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model not found. Run `python -m app.train_model` first."
            )
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


def reload_model() -> None:
    """Force reload of model from disk (e.g. after retraining)."""
    global _model_bundle
    _model_bundle = None
    _load_model()


def model_ready() -> bool:
    return MODEL_PATH.exists()


def get_model_info() -> Dict[str, Any]:
    """Return model metadata without running inference."""
    try:
        bundle = _load_model()
        return {
            "version":    bundle.get("version", "unknown"),
            "model_name": bundle.get("model_name", "unknown"),
            "feature_count": len(bundle.get("feature_cols", FEATURE_COLS)),
        }
    except Exception:
        return {"version": "not_loaded", "model_name": "none", "feature_count": 0}


def _get_feature_importances(bundle: Dict, feature_cols: List[str]) -> List[Dict]:
    """
    Extract feature importances from the model bundle.
    Uses precomputed importances stored at training time for HistGradientBoosting;
    falls back to clf attribute for RF / LR.
    """
    # Prefer precomputed importances stored in artifact (avoids slow per-request computation)
    stored = bundle.get("feature_importances")
    if stored:
        items = sorted(stored.items(), key=lambda x: -x[1])
        return [
            {
                "feature":     feat,
                "importance":  round(imp, 4),
                "description": FEATURE_DESCRIPTIONS.get(feat, feat),
            }
            for feat, imp in items[:8]
            if imp > 0.001
        ]

    clf = bundle["pipeline"].named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        return []

    fi_pairs = sorted(zip(feature_cols, importances.tolist()), key=lambda x: -x[1])[:8]
    return [
        {
            "feature":     feat,
            "importance":  round(imp, 4),
            "description": FEATURE_DESCRIPTIONS.get(feat, feat),
        }
        for feat, imp in fi_pairs
        if imp > 0.001
    ]


def predict(
    patient_id: str,
    recent_readings: List[Dict],
    patient_static: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run inference and return a structured prediction result.
    The result separates:
      - spike_probability: classifier output (risk of glucose > 180 in 2h)
      - predicted_glucose_mgdl: linear extrapolation heuristic (clearly labelled)
    These are NOT the same quantity.
    """
    from datetime import datetime, timezone

    if not recent_readings:
        raise ValueError("No readings provided — cannot predict.")

    bundle   = _load_model()
    pipeline = bundle["pipeline"]
    feature_cols = bundle.get("feature_cols", FEATURE_COLS)

    # Build feature vector (identical path to training)
    features = extract_inference_features(recent_readings, patient_static)
    X = pd.DataFrame([features])[feature_cols].astype(float)

    # Validate no NaNs made it through
    if X.isna().any().any():
        nan_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"NaN in inference features: {nan_cols}")

    spike_prob = float(pipeline.predict_proba(X)[0, 1])

    # Risk category thresholds (set on validation data, not test data)
    if spike_prob >= 0.60:
        risk_category = "high"
    elif spike_prob >= 0.30:
        risk_category = "moderate"
    else:
        risk_category = "low"

    # Glucose extrapolation heuristic — clearly NOT a regression model output
    latest_g  = features["latest_glucose"]
    roc       = features["glucose_roc_1h"]         # mg/dL per hour
    predicted_glucose = round(latest_g + roc * 2.0, 1)
    predicted_glucose = float(np.clip(predicted_glucose, 40, 500))

    # Feature importances (read from precomputed bundle or clf attribute)
    feature_importances = _get_feature_importances(bundle, feature_cols)

    # Glucose trend
    trend = glucose_trend(recent_readings)

    # Natural-language explanation (no invented medical claims)
    top_feat_desc = feature_importances[0]["description"] if feature_importances else "recent glucose level"
    explanation = (
        f"The model assigns a {risk_category.upper()} risk of glucose exceeding "
        f"{SPIKE_THRESHOLD_MGDL:.0f} mg/dL within the next 2 hours "
        f"(probability: {spike_prob:.0%}). "
        f"The most influential input was: '{top_feat_desc}'. "
        f"Current CGM reading is {latest_g:.0f} mg/dL ({trend} trend). "
        f"Note: This is a model correlation — not a medical diagnosis."
    )

    return {
        "patient_id":               patient_id,
        "prediction_timestamp":     datetime.now(timezone.utc).isoformat(),
        "prediction_horizon_hours": 2,
        "spike_probability":        round(spike_prob, 4),
        "risk_category":            risk_category,
        # Extrapolated glucose is a heuristic, not a regression model
        "predicted_glucose_mgdl":   predicted_glucose,
        "predicted_glucose_note":   "Linear extrapolation heuristic — not a trained regression model",
        "recent_glucose_trend":     trend,
        "current_glucose_mgdl":     round(latest_g, 1),
        "feature_importances":      feature_importances,
        "model_version":            bundle.get("version", "2.0.0"),
        "model_name":               bundle.get("model_name", "unknown"),
        "experimental_status":      "EXPERIMENTAL — Synthetic data only — NOT for clinical use",
        "explanation":              explanation,
    }
