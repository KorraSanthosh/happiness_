"""
Train Model on Real OhioT1DM Data  — v3.1 (FIXED SPLIT)
=========================================================
KEY FIX: OhioT1DM already has a proper temporal split per patient:
  - *-training_processed.csv  = weeks 1-6  (use for TRAINING all 12 patients)
  - *-testing_processed.csv   = weeks 7-8  (use for TEST    all 12 patients)

Previous version randomly split 12 patients 60/20/20 → only 3 training patients → low accuracy.
Now ALL 12 patients contribute to training (via their train files) and testing (via their test files).

Models compared: Logistic Regression, Random Forest, HistGradientBoosting
Extra: threshold tuning on val set for maximum F1/Recall.
Saves best model + metrics to backend/models/.
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from ohio_data_loader import load_ohio_data
from ohio_features import get_training_data, FEATURE_COLS, TARGET_COL, SPIKE_THRESHOLD

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss, accuracy_score,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../models")
os.makedirs(MODELS_DIR, exist_ok=True)
SEED = 42


def evaluate(y_true, y_pred, y_prob, name: str) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "model":        name,
        "accuracy":     round(accuracy_score(y_true, y_pred), 4),
        "precision":    round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":       round(recall_score(y_true, y_pred, zero_division=0), 4),
        "specificity":  round(tn / (tn + fp) if (tn + fp) > 0 else 0, 4),
        "f1":           round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":      round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc":       round(average_precision_score(y_true, y_prob), 4),
        "brier_score":  round(brier_score_loss(y_true, y_prob), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "positive_count": int(y_true.sum()),
        "negative_count": int((1 - y_true).sum()),
        "support_total":  int(len(y_true)),
    }


def best_threshold(y_true, y_prob) -> float:
    """Find threshold that maximises F1 on a set."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.30, 0.71, 0.01):
        pred = (y_prob >= t).astype(int)
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return round(best_t, 2)


def build_candidates():
    return {
        "Logistic Regression (baseline)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=2000, random_state=SEED,
                                          C=0.5, class_weight="balanced")),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=300, max_depth=12,
                min_samples_leaf=5, random_state=SEED,
                class_weight="balanced", n_jobs=-1,
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.03,
                max_depth=7, l2_regularization=0.1,
                random_state=SEED,
            )),
        ]),
    }


def train():
    print("=" * 65)
    print("  GlucoTwin v3.1 — Training on Real OhioT1DM Data (FIXED)")
    print("=" * 65)

    # ── Load TRAIN files (all 12 patients, weeks 1-6) ────────────────────
    print("\n[1/6] Loading OhioT1DM TRAIN files (all 12 patients, weeks 1-6)...")
    df_train_raw = load_ohio_data(splits=("train",))
    print(f"      Patients: {df_train_raw['patient_id'].nunique()} | Rows: {len(df_train_raw):,}")

    # ── Load TEST files  (all 12 patients, weeks 7-8) ────────────────────
    print("[2/6] Loading OhioT1DM TEST files  (all 12 patients, weeks 7-8)...")
    df_test_raw = load_ohio_data(splits=("test",))
    print(f"      Patients: {df_test_raw['patient_id'].nunique()} | Rows: {len(df_test_raw):,}")

    # ── Feature engineering ───────────────────────────────────────────────
    print("[3/6] Building features...")
    X_train, y_train, groups_train, _ = get_training_data(df_train_raw)
    X_test,  y_test,  groups_test,  _ = get_training_data(df_test_raw)

    # Hold out last 20% of train (by row order) as validation for model selection
    val_split = int(len(X_train) * 0.80)
    X_val,  y_val  = X_train[val_split:], y_train[val_split:]
    X_train_fit, y_train_fit = X_train[:val_split], y_train[:val_split]

    print(f"      Train fit : {len(X_train_fit):,} rows")
    print(f"      Val       : {len(X_val):,} rows  (last 20% of train period)")
    print(f"      Test      : {len(X_test):,} rows  (real held-out weeks 7-8)")
    print(f"      Features  : {len(FEATURE_COLS)}")
    print(f"      Spike prevalence — train: {y_train_fit.mean()*100:.1f}%  test: {y_test.mean()*100:.1f}%")

    # ── Model comparison on validation set ───────────────────────────────
    print("\n[4/6] Training & comparing models on validation set...")
    candidates = build_candidates()
    val_results   = {}
    trained_models = {}

    for name, pipe in candidates.items():
        print(f"      → {name}")
        pipe.fit(X_train_fit, y_train_fit)
        y_prob_v = pipe.predict_proba(X_val)[:, 1]
        thresh   = best_threshold(y_val, y_prob_v)
        y_pred_v = (y_prob_v >= thresh).astype(int)
        m = evaluate(y_val, y_pred_v, y_prob_v, name)
        m["threshold"] = thresh
        val_results[name]    = m
        trained_models[name] = (pipe, thresh)
        print(f"         PR-AUC={m['pr_auc']:.3f}  ROC-AUC={m['roc_auc']:.3f}"
              f"  F1={m['f1']:.3f}  Acc={m['accuracy']:.3f}  thresh={thresh}")

    # ── Select best model by PR-AUC ──────────────────────────────────────
    best_name = max(val_results, key=lambda k: val_results[k]["pr_auc"])
    best_model, best_thresh = trained_models[best_name]
    print(f"\n      ✅ Selected: {best_name}  (threshold={best_thresh})")

    # ── Retrain best model on full train data (train + val) ──────────────
    print("\n[5/6] Retraining best model on full training data...")
    best_model.fit(X_train, y_train)

    # ── Final evaluation on held-out test set ────────────────────────────
    print("[6/6] Final evaluation on real test set (weeks 7-8)...")
    y_prob_test = best_model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= best_thresh).astype(int)
    test_metrics = evaluate(y_test, y_pred_test, y_prob_test, best_name)
    test_metrics["threshold"] = best_thresh

    print(f"\n  ── Test Set Results ({best_name}) ──")
    for k, v in test_metrics.items():
        if k not in ("confusion_matrix", "model"):
            print(f"     {k:20s}: {v}")

    # ── Permutation feature importance ────────────────────────────────────
    from sklearn.inspection import permutation_importance
    print("\n  Computing feature importances...")
    pi = permutation_importance(
        best_model, X_val, y_val,
        scoring="average_precision",
        n_repeats=5, random_state=SEED, n_jobs=-1,
    )
    feature_importances = dict(zip(FEATURE_COLS, pi.importances_mean.tolist()))
    top_features = sorted(feature_importances.items(), key=lambda x: -x[1])[:10]
    print("\n  Top 10 features by permutation importance:")
    for feat, imp in top_features:
        print(f"     {feat:35s}: {imp:.4f}")

    # ── Save artifacts ────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, "glucotwin_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": best_model, "threshold": best_thresh}, f)
    print(f"\n  Model saved → {model_path}")

    metrics_out = {
        "model_version":           "3.1.0",
        "data_source":             "OhioT1DM 2018+2020 (12 real T1D patients)",
        "selected_model":          best_name,
        "selection_criterion":     "PR-AUC on validation set + threshold tuning",
        "split_strategy":          "OhioT1DM native split: train files=weeks1-6, test files=weeks7-8. All 12 patients in both.",
        "classification_threshold": best_thresh,
        "train_size":              int(len(X_train)),
        "test_size":               int(len(X_test)),
        "spike_prevalence_train":  float(round(y_train.mean(), 4)),
        "spike_prevalence_test":   float(round(y_test.mean(), 4)),
        "test_metrics":            test_metrics,
        "validation_comparison":   val_results,
        "feature_importances":     feature_importances,
        "feature_cols":            FEATURE_COLS,
        "spike_threshold_mgdl":    SPIKE_THRESHOLD,
        "prediction_horizon_h":    2,
        "trained_at":              datetime.utcnow().isoformat(),
        "note": "Real OhioT1DM data — CGM + meal carbs + insulin. NOT FOR CLINICAL USE.",
    }
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  Metrics saved → {metrics_path}")

    model_card = {
        "model_name":   "GlucoTwin AI v3.1",
        "description":  "Predicts glucose spike (>180 mg/dL) within 2h using real OhioT1DM CGM+meal+insulin data.",
        "data_source":  "OhioT1DM 2018+2020, 12 real T1D patients",
        "algorithm":    best_name,
        "n_features":   len(FEATURE_COLS),
        "feature_cols": FEATURE_COLS,
        "target":       "spike_target (1=spike within 2h, 0=no spike)",
        "threshold":    best_thresh,
        "test_metrics": test_metrics,
        "top_features": [{"feature": f, "importance": round(i, 4)} for f, i in top_features],
        "limitations": [
            "Only 12 real T1D patients — may not generalise to T2D.",
            "Not CE/FDA approved — experimental prototype only.",
        ],
        "trained_at": datetime.utcnow().isoformat(),
    }
    card_path = os.path.join(MODELS_DIR, "model_card.json")
    with open(card_path, "w") as f:
        json.dump(model_card, f, indent=2)
    print(f"  Model card saved → {card_path}")
    print("\n✅ Training complete!")
    return best_model, metrics_out


if __name__ == "__main__":
    train()
