"""
GlucoTwin AI — Multi-Model Training & Evaluation Pipeline v2.

Compares three models with patient-aware GroupShuffleSplit:
  1. Logistic Regression (baseline)
  2. Random Forest
  3. HistGradientBoosting (best default for tabular data)

Selects the best model by PR-AUC on validation set.
Evaluates once on a held-out test set.
Saves: model pipeline, feature list, comprehensive metrics JSON.

Run: python -m app.train_model
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.data_generator import load_or_generate
from app.data_validation import run_and_save
from app.features import FEATURE_COLS, build_features_from_df

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH   = MODEL_DIR / "glucotwin_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
MODEL_VERSION = "2.0.0"

RANDOM_SEED = 42


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe(fn, *args, **kwargs) -> Any:
    try:
        v = fn(*args, **kwargs)
        return float(v) if not isinstance(v, (list, dict, np.ndarray)) else v
    except Exception:
        return None


def _eval_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label: str,
) -> Dict[str, Any]:
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    specificity = tn / (tn + fp) if (tn + fp) > 0 else None

    m: Dict[str, Any] = {
        "model": label,
        "precision":      _safe(precision_score, y_test, y_pred, zero_division=0),
        "recall":         _safe(recall_score,    y_test, y_pred, zero_division=0),
        "specificity":    round(specificity, 4) if specificity is not None else None,
        "f1":             _safe(f1_score,         y_test, y_pred, zero_division=0),
        "roc_auc":        _safe(roc_auc_score,    y_test, y_proba),
        "pr_auc":         _safe(average_precision_score, y_test, y_proba),
        "brier_score":    _safe(brier_score_loss, y_test, y_proba),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
        "positive_count": int(y_test.sum()),
        "negative_count": int((y_test == 0).sum()),
        "support_total":  int(len(y_test)),
    }
    print(f"\n  [{label}] PR-AUC={m['pr_auc']:.3f}  ROC-AUC={m['roc_auc']:.3f}  "
          f"F1={m['f1']:.3f}  Recall={m['recall']:.3f}  Precision={m['precision']:.3f}")
    return m


# ── Main training function ─────────────────────────────────────────────────────

def train(force_retrain: bool = True) -> dict:
    """Train, evaluate, and save the best model."""

    print("[train] Loading / generating synthetic data …")
    patients, raw_readings = load_or_generate()

    print("[train] Running data-quality validation …")
    readings_df, dq_report = run_and_save(patients, raw_readings)

    print("[train] Building features …")
    df = build_features_from_df(readings_df)

    # Exclude rows with unknown future horizon
    df = df[df["spike_target"] != -1].copy()
    print(f"[train] Training rows available: {len(df)}")

    X = df[FEATURE_COLS].astype(float)
    y = df["spike_target"].astype(int)
    groups = df["patient_id"]

    # ── Patient-aware three-way split: train / val / test ─────────────────────
    # Split 1: hold out 20% of patients as test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
    dev_idx, test_idx = next(gss_test.split(X, y, groups))

    X_dev, X_test = X.iloc[dev_idx], X.iloc[test_idx]
    y_dev, y_test = y.iloc[dev_idx], y.iloc[test_idx]
    g_dev         = groups.iloc[dev_idx]

    # Split 2: hold out 20% of remaining patients as validation
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED)  # 0.25 × 0.80 = 20% overall
    train_idx, val_idx = next(gss_val.split(X_dev, y_dev, g_dev))

    X_train, X_val = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
    y_train, y_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]

    print(f"[train] Split — Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"[train] Spike prevalence — train: {y_train.mean():.2%}  val: {y_val.mean():.2%}  test: {y_test.mean():.2%}")

    # ── Define candidate models ────────────────────────────────────────────────
    # class_weight='balanced' handles imbalance without oversampling
    candidates = {
        "Logistic Regression (baseline)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_SEED,
                C=0.1,
            )),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=6,
                learning_rate=0.05,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            )),
        ]),
    }

    # ── Train and evaluate on validation set ──────────────────────────────────
    print("\n[train] Comparing models on validation set …")
    val_results = {}
    fitted_pipelines = {}

    for name, pipe in candidates.items():
        print(f"\n  Training: {name}")
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        val_results[name] = _eval_model(pipe, X_val, y_val, name)

    # ── Select best model by PR-AUC (best for imbalanced clinical screening) ──
    best_name = max(val_results, key=lambda k: val_results[k].get("pr_auc") or 0)
    best_pipeline = fitted_pipelines[best_name]
    print(f"\n[train] ✓ Best model selected: {best_name} "
          f"(val PR-AUC={val_results[best_name]['pr_auc']:.3f})")

    # ── Final evaluation on held-out test set ─────────────────────────────────
    print("\n[train] Evaluating best model on held-out test set …")
    test_metrics = _eval_model(best_pipeline, X_test, y_test, best_name)
    print("\n" + classification_report(
        y_test, best_pipeline.predict(X_test), zero_division=0
    ))

    # ── Feature importances ───────────────────────────────────────────────────
    clf = best_pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = dict(zip(FEATURE_COLS, clf.feature_importances_.tolist()))
    elif hasattr(clf, "coef_"):
        importances = dict(zip(FEATURE_COLS, np.abs(clf.coef_[0]).tolist()))
    else:
        # HistGradientBoosting: use permutation importance on validation set
        # (never computed on test set — val set is safe here)
        from sklearn.inspection import permutation_importance
        print("[train] Computing permutation importance (HistGB) on validation set …")
        perm = permutation_importance(
            best_pipeline, X_val, y_val,
            n_repeats=5, random_state=RANDOM_SEED, n_jobs=-1,
            scoring="average_precision",
        )
        importances = dict(zip(FEATURE_COLS, perm.importances_mean.tolist()))

    # ── Compile full metrics report ───────────────────────────────────────────
    metrics = {
        "model_version":          MODEL_VERSION,
        "selected_model":         best_name,
        "selection_criterion":    "PR-AUC on validation set (robust for imbalanced classification)",
        "train_size":             int(len(X_train)),
        "val_size":               int(len(X_val)),
        "test_size":              int(len(X_test)),
        "spike_prevalence_train": float(y_train.mean()),
        "spike_prevalence_val":   float(y_val.mean()),
        "spike_prevalence_test":  float(y_test.mean()),
        "split_strategy":         "Patient-aware GroupShuffleSplit — same patient never in train+test",
        "test_metrics":           test_metrics,
        "validation_comparison":  {
            name: {k: v for k, v in r.items() if k not in ("confusion_matrix",)}
            for name, r in val_results.items()
        },
        "feature_importances":    importances,
        "feature_cols":           FEATURE_COLS,
        "spike_threshold_mgdl":   180.0,
        "prediction_horizon_h":   2,
        "data_quality_report":    dq_report,
        "limitations": [
            "All data is SYNTHETIC — no real patient data was used.",
            "Metrics reflect performance on synthetic data only.",
            "Not validated for clinical use.",
            "Model calibration not formally verified on independent data.",
        ],
        "note": "EXPERIMENTAL — Synthetic data only — NOT FOR CLINICAL USE",
    }

    # ── Save model and metrics ────────────────────────────────────────────────
    artifact = {
        "pipeline":            best_pipeline,
        "feature_cols":        FEATURE_COLS,
        "version":             MODEL_VERSION,
        "model_name":          best_name,
        "feature_importances": importances,   # precomputed — used by predictor at inference
    }
    joblib.dump(artifact, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\n[train] ✓ Model saved → {MODEL_PATH}")
    print(f"[train] ✓ Metrics saved → {METRICS_PATH}")

    return metrics


if __name__ == "__main__":
    train(force_retrain=True)
