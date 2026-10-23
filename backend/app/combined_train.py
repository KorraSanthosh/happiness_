"""
Combined Training — OhioT1DM + BrisT1D → 95%+ Accuracy Target
==============================================================
Strategy:
  - OhioT1DM: 12 real T1D patients (weeks 1-6 = train, weeks 7-8 = test)
  - BrisT1D:  15+ real T1D patients (train.csv → training pool)
  - Combined: ~27 patients → much better generalisation
  - Model: XGBoost + SMOTE + threshold tuning → target 95%+ accuracy

Run this AFTER placing brist1d_raw/train.csv and brist1d_raw/test.csv
in the project root.
"""

import os, sys, json, pickle, warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from ohio_data_loader import load_ohio_data
from brist1d_loader   import load_brist1d
from ohio_features    import get_training_data, FEATURE_COLS, TARGET_COL, SPIKE_THRESHOLD

from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import StandardScaler
from sklearn.ensemble       import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model   import LogisticRegression
from sklearn.metrics        import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss,
)

# Try importing XGBoost — install if missing
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  ⚠️  XGBoost not installed. Run: pip install xgboost")

# Try SMOTE
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("  ⚠️  imbalanced-learn not installed. Run: pip install imbalanced-learn")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../models")
os.makedirs(MODELS_DIR, exist_ok=True)
SEED = 42


def evaluate(y_true, y_pred, y_prob, name):
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
        "brier":        round(brier_score_loss(y_true, y_prob), 4),
        "confusion":    {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_positive":   int(y_true.sum()),
        "n_negative":   int((1 - y_true).sum()),
        "n_total":      int(len(y_true)),
    }


def best_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.25, 0.76, 0.01):
        pred = (y_prob >= t).astype(int)
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return round(best_t, 2)


def build_candidates():
    candidates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, C=1.0,
                class_weight="balanced", random_state=SEED,
            )),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=400, max_depth=15,
                min_samples_leaf=3, class_weight="balanced",
                random_state=SEED, n_jobs=-1,
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(
                max_iter=500, learning_rate=0.02,
                max_depth=8, l2_regularization=0.05,
                random_state=SEED,
            )),
        ]),
    }
    if HAS_XGB:
        candidates["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=500, learning_rate=0.02,
                max_depth=7, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=1.0,
                scale_pos_weight=1,
                random_state=SEED, n_jobs=-1,
                eval_metric="logloss", verbosity=0,
            )),
        ])
    return candidates


def train():
    print("=" * 65)
    print("  GlucoTwin v4.0 — OhioT1DM + BrisT1D Combined Training")
    print("  Target: 95%+ Accuracy")
    print("=" * 65)

    # ── Load OhioT1DM ──────────────────────────────────────────────────────
    print("\n[1/7] Loading OhioT1DM (12 patients)...")
    ohio_train = load_ohio_data(splits=("train",))
    ohio_test  = load_ohio_data(splits=("test",))
    print(f"      Ohio train: {len(ohio_train):,} rows | test: {len(ohio_test):,} rows")

    # ── Load BrisT1D ───────────────────────────────────────────────────────
    print("\n[2/7] Loading BrisT1D...")
    brist_train = load_brist1d("train")
    print(f"      BrisT1D train: {len(brist_train):,} rows")

    # ── Combine ────────────────────────────────────────────────────────────
    print("\n[3/7] Combining datasets...")
    combined_train = pd.concat([ohio_train, brist_train], ignore_index=True)
    combined_test  = ohio_test   # keep OhioT1DM test as gold standard
    print(f"      Combined train: {len(combined_train):,} rows | "
          f"Patients: {combined_train['patient_id'].nunique()}")
    print(f"      Test (Ohio):    {len(combined_test):,} rows")

    # ── Feature engineering ────────────────────────────────────────────────
    print("\n[4/7] Building features...")
    X_train, y_train, _, _ = get_training_data(combined_train)
    X_test,  y_test,  _, _ = get_training_data(combined_test)

    val_idx = int(len(X_train) * 0.85)
    X_val, y_val = X_train[val_idx:], y_train[val_idx:]
    X_tr,  y_tr  = X_train[:val_idx], y_train[:val_idx]

    print(f"      Train: {len(X_tr):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"      Features: {len(FEATURE_COLS)}")
    print(f"      Spike prev — train: {y_tr.mean()*100:.1f}%  test: {y_test.mean()*100:.1f}%")

    # ── SMOTE on training data ─────────────────────────────────────────────
    if HAS_SMOTE:
        print("\n[5/7] Applying SMOTE to balance classes...")
        sm = SMOTE(random_state=SEED, k_neighbors=5)
        X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)
        print(f"      After SMOTE: {len(X_tr_sm):,} rows "
              f"(spike: {y_tr_sm.mean()*100:.1f}%)")
    else:
        X_tr_sm, y_tr_sm = X_tr, y_tr
        print("\n[5/7] Skipping SMOTE (not installed)")

    # ── Model comparison ───────────────────────────────────────────────────
    print("\n[6/7] Training & comparing models...")
    candidates    = build_candidates()
    val_results   = {}
    trained_models = {}

    for name, pipe in candidates.items():
        print(f"      → {name}")
        pipe.fit(X_tr_sm, y_tr_sm)
        y_prob_v = pipe.predict_proba(X_val)[:, 1]
        thresh   = best_threshold(y_val, y_prob_v)
        y_pred_v = (y_prob_v >= thresh).astype(int)
        m = evaluate(y_val, y_pred_v, y_prob_v, name)
        m["threshold"] = thresh
        val_results[name]    = m
        trained_models[name] = (pipe, thresh)
        print(f"         Acc={m['accuracy']:.3f}  F1={m['f1']:.3f}  "
              f"PR-AUC={m['pr_auc']:.3f}  thresh={thresh}")

    best_name = max(val_results, key=lambda k: val_results[k]["pr_auc"])
    best_model, best_thresh = trained_models[best_name]
    print(f"\n      ✅ Best: {best_name} (threshold={best_thresh})")

    # ── Retrain on full data ───────────────────────────────────────────────
    print("\n[7/7] Retraining on full combined data + final test evaluation...")
    if HAS_SMOTE:
        sm2 = SMOTE(random_state=SEED, k_neighbors=5)
        X_full_sm, y_full_sm = sm2.fit_resample(X_train, y_train)
        best_model.fit(X_full_sm, y_full_sm)
    else:
        best_model.fit(X_train, y_train)

    y_prob_test = best_model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= best_thresh).astype(int)
    test_m = evaluate(y_test, y_pred_test, y_prob_test, best_name)
    test_m["threshold"] = best_thresh

    print(f"\n  ── FINAL TEST RESULTS ({best_name}) ──")
    for k, v in test_m.items():
        if k not in ("confusion", "model"):
            print(f"     {k:20s}: {v}")

    # ── Save ───────────────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, "glucotwin_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": best_model, "threshold": best_thresh}, f)

    metrics_out = {
        "model_version":    "4.0.0",
        "data_source":      "OhioT1DM 2018+2020 + BrisT1D Kaggle",
        "selected_model":   best_name,
        "threshold":        best_thresh,
        "smote_applied":    HAS_SMOTE,
        "train_size":       int(len(X_train)),
        "test_size":        int(len(X_test)),
        "spike_prev_train": float(round(y_train.mean(), 4)),
        "spike_prev_test":  float(round(y_test.mean(), 4)),
        "test_metrics":     test_m,
        "val_comparison":   val_results,
        "feature_cols":     FEATURE_COLS,
        "trained_at":       datetime.utcnow().isoformat(),
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"\n  ✅ Model saved → {model_path}")
    print(f"  ✅ Metrics saved → models/metrics.json")
    print("\n🏆 Training complete!")
    return best_model, metrics_out


if __name__ == "__main__":
    train()
