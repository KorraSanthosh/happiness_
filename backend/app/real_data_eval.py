"""
Evaluation, bias / over- / under-fitting diagnostics and final training for the combined
OhioT1DM + BrisT1D model (features from app/real_data_pipeline.py).

Protocol (everything is PATIENT-grouped: a patient is never in train and test at once)
  A. 7-fold patient-grouped CV over all 21 patients (3 per fold).
  B. Nested tuning inside A: config + decision threshold are chosen on an inner 3-fold patient-grouped
     CV of the outer-train patients only; outer-test patients are touched once.
  C. Ohio native temporal split (train files = weeks 1-6, test files = weeks 7-8; patients seen in
     training) - the protocol the old v3.1 numbers used; optimistic, kept for comparability only.

Stages:  python -m app.real_data_eval --stage {ablation,tune,diagnostics,temporal,final,all}
Outputs: models/combined_v5/   (the served model in backend/models/ is never touched)
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score,
    log_loss, matthews_corrcoef, roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app import real_data_pipeline as P

warnings.filterwarnings("ignore")

OUT_DIR = P.ROOT / "models" / "combined_v5"
SEED = 42
N_OUTER = 7
N_INNER = 3
MODEL_VERSION = "5.0.0"

GLUCOSE_ONLY = ["glucose", "g_lag_15m", "g_lag_30m", "g_lag_1h", "g_d30m", "g_d1h", "g_d2h", "g_roc_1h"]

HGB_DEFAULT = dict(max_iter=300, learning_rate=0.05, max_depth=6, min_samples_leaf=100, l2_regularization=1.0)
TUNE_GRID: Dict[str, Dict[str, Any]] = {
    "default":      dict(),
    "shallow":      dict(max_depth=3, min_samples_leaf=200, l2_regularization=5.0, max_iter=400),
    "deep":         dict(max_depth=10, min_samples_leaf=50, l2_regularization=0.5),
    "slow":         dict(learning_rate=0.03, max_iter=500, max_depth=5, min_samples_leaf=150, l2_regularization=2.0),
    "leafy-reg":    dict(max_leaf_nodes=15, max_depth=None, min_samples_leaf=300, l2_regularization=10.0, max_iter=400),
}


# ═════════════════════════════════════════════════════════════════════════════
# data, folds, models, metrics
# ═════════════════════════════════════════════════════════════════════════════

def load_frame(verbose: bool = True) -> pd.DataFrame:
    grid, stats = P.load_all_grids()
    m = P.modelling_frame(P.build_features(grid))
    obs = grid.groupby("patient_id")["glucose"].apply(lambda s: s.notna().mean())
    m["cgm_native"] = m["patient_id"].map(lambda p: "15min" if obs[p] < 0.6 else "5min")
    if verbose:
        print(f"[data] {len(m):,} rows | {m.patient_id.nunique()} patients | "
              f"stitch conflicts={stats['conflicts']} | prevalence={m.y.mean():.3f}", flush=True)
    return m


def assign_folds(m: pd.DataFrame, n: int = N_OUTER, seed: int = SEED) -> Dict[str, int]:
    rng = np.random.default_rng(seed)
    order: List[str] = []
    for src in ("ohio", "brist"):
        ps = sorted(m.loc[m.source == src, "patient_id"].unique())
        rng.shuffle(ps)
        order += list(ps)
    return {p: i % n for i, p in enumerate(order)}


def make_hgb(**kw) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**{**HGB_DEFAULT, **kw}, random_state=SEED)


def make_lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.5, max_iter=3000))


def make_rf():
    return RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=20,
                                  max_samples=0.3, n_jobs=-1, random_state=SEED)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        mk = idx == b
        if mk.any():
            tot += mk.mean() * abs(y[mk].mean() - p[mk].mean())
    return float(tot)


def metrics(y, p, thr: Any = 0.5) -> Dict[str, Any]:
    """thr is a scalar or a per-row array (e.g. each row's fold threshold); every threshold-dependent
    metric then uses the same per-row thresholds. "threshold" reports the median."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    thr_arr = np.asarray(thr, dtype=float)
    pred = (p >= thr_arr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    both = len(np.unique(y)) == 2
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)), "prevalence": round(float(y.mean()), 4), "threshold": round(float(np.median(thr_arr)), 3),
        "accuracy": round((tp + tn) / len(y), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4) if both else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "specificity": round(tn / (tn + fp), 4) if tn + fp else None,
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y, pred)), 4) if both else None,
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if both else None,
        "pr_auc": round(float(average_precision_score(y, p)), 4) if both else None,
        "brier": round(float(brier_score_loss(y, pc)), 4),
        "log_loss": round(float(log_loss(y, pc, labels=[0, 1])), 4),
        "ece": round(ece(y, p), 4),
        "mean_pred": round(float(p.mean()), 4),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def best_threshold(y, p, lo: float = 0.2, hi: float = 0.8) -> float:
    ts = np.arange(lo, hi + 1e-9, 0.01)
    scores = [balanced_accuracy_score(y, (p >= t).astype(int)) for t in ts]
    return round(float(ts[int(np.argmax(scores))]), 2)


def subset_metrics(m: pd.DataFrame, p: np.ndarray, thr: Any = 0.5) -> Dict[str, Dict[str, Any]]:
    out = {}
    thr_arr = np.full(len(m), thr) if np.isscalar(thr) else np.asarray(thr)
    for name, mask in (("all", np.ones(len(m), bool)),
                       ("ohio", (m.source == "ohio").to_numpy()),
                       ("brist", (m.source == "brist").to_numpy())):
        if mask.any() and not np.isnan(p[mask]).any():
            t = float(np.median(thr_arr[mask]))
            pred_acc = ((p[mask] >= thr_arr[mask]).astype(int) == m.y.to_numpy()[mask]).mean()
            r = metrics(m.y.to_numpy()[mask], p[mask], t)
            r["accuracy"] = round(float(pred_acc), 4)            # exact per-row thresholds
            out[name] = r
    return out


def patient_level(m: pd.DataFrame, p: np.ndarray, thr: Any = 0.5) -> Dict[str, float]:
    thr_arr = np.full(len(m), thr) if np.isscalar(thr) else np.asarray(thr)
    ok = ((p >= thr_arr).astype(int) == m.y.to_numpy())
    per = pd.Series(ok).groupby(m.patient_id.to_numpy()).mean()
    return {"mean": round(float(per.mean()), 4), "std": round(float(per.std()), 4),
            "min": round(float(per.min()), 4), "max": round(float(per.max()), 4)}


def cluster_bootstrap(m: pd.DataFrame, probs: Dict[str, np.ndarray], thr: Dict[str, Any],
                      n: int = 500, seed: int = SEED) -> Dict[str, Any]:
    """Resample PATIENTS with replacement; 95% CIs for accuracy / ROC-AUC (and paired differences vs first key)."""
    rng = np.random.default_rng(seed)
    groups = {pid: np.where(m.patient_id.to_numpy() == pid)[0] for pid in m.patient_id.unique()}
    pids = list(groups)
    y = m.y.to_numpy()
    names = list(probs)
    acc = {k: [] for k in names}; auc = {k: [] for k in names}
    for _ in range(n):
        idx = np.concatenate([groups[pids[i]] for i in rng.integers(0, len(pids), len(pids))])
        for k in names:
            t = np.full(len(m), thr[k]) if np.isscalar(thr[k]) else np.asarray(thr[k])
            acc[k].append(float((((probs[k][idx] >= t[idx]).astype(int)) == y[idx]).mean()))
            auc[k].append(float(roc_auc_score(y[idx], probs[k][idx])))
    ci = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    res: Dict[str, Any] = {k: {"accuracy_ci95": ci(acc[k]), "roc_auc_ci95": ci(auc[k])} for k in names}
    base = names[0]
    for k in names[1:]:
        da = np.array(acc[k]) - np.array(acc[base]); dr = np.array(auc[k]) - np.array(auc[base])
        res[k]["vs_" + base] = {
            "accuracy_diff_mean": round(float(da.mean()), 4), "accuracy_diff_ci95": ci(da),
            "roc_auc_diff_mean": round(float(dr.mean()), 4), "roc_auc_diff_ci95": ci(dr),
            "accuracy_diff_excludes_0": bool(np.percentile(da, 2.5) > 0 or np.percentile(da, 97.5) < 0),
        }
    return res


def _fit_predict(factory: Callable, cols: List[str], tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    clf = factory().fit(tr[cols], tr["y"])
    return clf.predict_proba(te[cols])[:, 1]


def _save(name: str, obj: Any) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[save] {OUT_DIR / (name + '.json')}", flush=True)


def _load(name: str) -> Any:
    with open(OUT_DIR / f"{name}.json") as f:
        return json.load(f)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1: baselines + ablations (features vs data)
# ═════════════════════════════════════════════════════════════════════════════

def stage_ablation(m: pd.DataFrame) -> Dict[str, Any]:
    fold_of = assign_folds(m)
    fold = m.patient_id.map(fold_of).to_numpy()
    variants: Dict[str, Tuple[Callable, List[str], Optional[str]]] = {
        "LR glucose-only | combined":  (make_lr, GLUCOSE_ONLY, None),
        "LR engineered | combined":    (make_lr, P.ENGINEERED_FEATURES, None),
        "RF engineered | combined":    (make_rf, P.ENGINEERED_FEATURES, None),
        "HGB base | ohio-only":        (make_hgb, P.BASE_FEATURES, "ohio"),
        "HGB base | brist-only":       (make_hgb, P.BASE_FEATURES, "brist"),
        "HGB base | combined":         (make_hgb, P.BASE_FEATURES, None),
        "HGB engineered | ohio-only":  (make_hgb, P.ENGINEERED_FEATURES, "ohio"),
        "HGB engineered | brist-only": (make_hgb, P.ENGINEERED_FEATURES, "brist"),
        "HGB engineered | combined":   (make_hgb, P.ENGINEERED_FEATURES, None),
    }
    oof = {k: np.full(len(m), np.nan) for k in variants}
    t0 = time.time()
    for f in range(N_OUTER):
        te = fold == f
        for name, (factory, cols, src) in variants.items():
            trm = ~te if src is None else (~te) & (m.source == src).to_numpy()
            oof[name][te] = _fit_predict(factory, cols, m[trm], m[te])
        print(f"[ablation] outer fold {f + 1}/{N_OUTER} done ({time.time() - t0:.0f}s)", flush=True)

    # rule baselines (no training involved)
    rules = {
        "rule: majority class": np.full(len(m), float(m.y.mean() >= 0.5)),
        "rule: glucose > 180 now": (m.glucose > 180).astype(float).to_numpy(),
        "rule: glucose or 2h projection > 180": ((m.glucose > 180) | (m.g_proj_2h > 180)).astype(float).to_numpy(),
    }
    res: Dict[str, Any] = {"rules": {}, "models": {}, "n_rows": len(m)}
    for k, p in rules.items():
        res["rules"][k] = subset_metrics(m, p, 0.5)
    for k, p in oof.items():
        res["models"][k] = {"subsets": subset_metrics(m, p, 0.5), "per_patient_accuracy": patient_level(m, p, 0.5)}

    # paired patient-cluster bootstrap: does each step actually help?
    pairs = {
        "feature engineering (HGB combined)": ("HGB base | combined", "HGB engineered | combined"),
        "adding BrisT1D to Ohio training (base)": ("HGB base | ohio-only", "HGB base | combined"),
        "adding BrisT1D to Ohio training (engineered)": ("HGB engineered | ohio-only", "HGB engineered | combined"),
        "HGB vs logistic regression (engineered, combined)": ("LR engineered | combined", "HGB engineered | combined"),
        "HGB vs rule 'glucose>180' (engineered, combined)": ("rule", "HGB engineered | combined"),
    }
    boots = {}
    for label, (a, b) in pairs.items():
        pa = rules["rule: glucose > 180 now"] if a == "rule" else oof[a]
        boots[label] = cluster_bootstrap(m, {"A": pa, "B": oof[b]}, {"A": 0.5, "B": 0.5})["B"]["vs_A"]
    res["paired_bootstrap"] = boots
    pd.DataFrame({"patient_id": m.patient_id, "segment": m.segment, "step": m.step, "y": m.y,
                  **{k: v for k, v in oof.items()}}).to_parquet(OUT_DIR / "oof_ablation.parquet")
    _save("stage_ablation", res)
    return res


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2: nested tuning + calibration + threshold (headline model)
# ═════════════════════════════════════════════════════════════════════════════

def _inner_cv(tr: pd.DataFrame, cols: List[str], cfg: Dict[str, Any]) -> np.ndarray:
    oof = np.full(len(tr), np.nan)
    for a, b in GroupKFold(N_INNER).split(tr, tr.y, tr.patient_id):
        clf = make_hgb(**cfg).fit(tr.iloc[a][cols], tr.iloc[a].y)
        oof[b] = clf.predict_proba(tr.iloc[b][cols])[:, 1]
    return oof


def stage_tune(m: pd.DataFrame, cols: Optional[List[str]] = None) -> Dict[str, Any]:
    cols = cols or P.ENGINEERED_FEATURES
    fold_of = assign_folds(m)
    fold = m.patient_id.map(fold_of).to_numpy()
    nested = pd.DataFrame({"patient_id": m.patient_id.to_numpy(), "segment": m.segment.to_numpy(),
                           "step": m.step.to_numpy(), "y": m.y.to_numpy(), "fold": fold,
                           "p": np.nan, "p_cal": np.nan, "thr": np.nan})
    folds_out = []
    models = []
    t0 = time.time()
    for f in range(N_OUTER):
        te = fold == f
        tr_df, te_df = m[~te].reset_index(drop=True), m[te]
        scores = {}
        inner_oof = {}
        for cname, cfg in TUNE_GRID.items():
            o = _inner_cv(tr_df, cols, cfg)
            inner_oof[cname] = o
            scores[cname] = float(log_loss(tr_df.y, np.clip(o, 1e-6, 1 - 1e-6)))
        best = min(scores, key=scores.get)
        o = inner_oof[best]
        thr = best_threshold(tr_df.y.to_numpy(), o)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(o, tr_df.y)
        clf = make_hgb(**TUNE_GRID[best]).fit(tr_df[cols], tr_df.y)
        p_te = clf.predict_proba(te_df[cols])[:, 1]
        p_tr = clf.predict_proba(tr_df[cols])[:, 1]
        nested.loc[te, "p"] = p_te
        nested.loc[te, "p_cal"] = iso.predict(p_te)
        nested.loc[te, "thr"] = thr
        models.append((clf, iso, thr, best))
        folds_out.append({
            "fold": f, "test_patients": sorted(te_df.patient_id.unique()), "selected_config": best,
            "inner_cv_logloss": {k: round(v, 4) for k, v in scores.items()}, "threshold": thr,
            "train": metrics(tr_df.y, p_tr, thr), "test": metrics(te_df.y, p_te, thr),
            "test_calibrated": metrics(te_df.y, iso.predict(p_te), thr),
        })
        print(f"[tune] fold {f + 1}/{N_OUTER}: config={best} thr={thr} "
              f"train_acc={folds_out[-1]['train']['accuracy']:.3f} test_acc={folds_out[-1]['test']['accuracy']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    p = nested.p.to_numpy()
    thr_row = nested.thr.to_numpy()
    res = {
        "features": "engineered", "n_features": len(cols),
        "config_grid": TUNE_GRID, "selection_metric": "inner patient-grouped CV log-loss",
        "threshold_rule": "balanced-accuracy optimum on inner out-of-fold predictions",
        "pooled_nested": subset_metrics(m, p, thr_row),
        "pooled_nested_at_0.5": subset_metrics(m, p, 0.5),
        "pooled_nested_calibrated": subset_metrics(m, nested.p_cal.to_numpy(), thr_row),
        "per_patient_accuracy": patient_level(m, p, thr_row),
        "fold_accuracy": [f["test"]["accuracy"] for f in folds_out],
        "fold_train_accuracy": [f["train"]["accuracy"] for f in folds_out],
        "folds": folds_out,
        "selected_configs": pd.Series([f["selected_config"] for f in folds_out]).value_counts().to_dict(),
        "bootstrap": cluster_bootstrap(m, {"model": p}, {"model": thr_row}),
    }
    nested.to_parquet(OUT_DIR / "oof_nested.parquet")
    joblib.dump(models, OUT_DIR / "_fold_models.joblib")
    _save("stage_tune", res)
    return res


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3: diagnostics - over/under-fitting and bias
# ═════════════════════════════════════════════════════════════════════════════

def _group_table(df: pd.DataFrame, by: str) -> List[Dict[str, Any]]:
    rows = []
    for key, g in df.groupby(by, observed=True):
        y = g.y.to_numpy(); p = g.p.to_numpy(); pred = (p >= g.thr.to_numpy()).astype(int)
        pos, neg = (y == 1).sum(), (y == 0).sum()
        rows.append({
            by: str(key), "n": int(len(g)), "prevalence": round(float(y.mean()), 3),
            "mean_pred": round(float(p.mean()), 3), "calibration_gap": round(float(p.mean() - y.mean()), 3),
            "accuracy": round(float((pred == y).mean()), 3),
            "recall": round(float(((pred == 1) & (y == 1)).sum() / pos), 3) if pos else None,
            "false_pos_rate": round(float(((pred == 1) & (y == 0)).sum() / neg), 3) if neg else None,
            "roc_auc": round(float(roc_auc_score(y, p)), 3) if len(np.unique(y)) == 2 else None,
        })
    return rows


def stage_diagnostics(m: pd.DataFrame, cols: Optional[List[str]] = None) -> Dict[str, Any]:
    cols = cols or P.ENGINEERED_FEATURES
    fold_of = assign_folds(m)
    fold = m.patient_id.map(fold_of).to_numpy()
    nested = pd.read_parquet(OUT_DIR / "oof_nested.parquet")
    tuned = _load("stage_tune")
    models = joblib.load(OUT_DIR / "_fold_models.joblib")
    res: Dict[str, Any] = {}
    t0 = time.time()

    # ── 1. over/under-fitting: train vs held-out gap per fold + baselines (underfit reference)
    gaps = []
    for f in tuned["folds"]:
        gaps.append({"fold": f["fold"],
                     "train_acc": f["train"]["accuracy"], "test_acc": f["test"]["accuracy"],
                     "acc_gap": round(f["train"]["accuracy"] - f["test"]["accuracy"], 4),
                     "train_logloss": f["train"]["log_loss"], "test_logloss": f["test"]["log_loss"],
                     "train_pr_auc": f["train"]["pr_auc"], "test_pr_auc": f["test"]["pr_auc"]})
    res["train_vs_test_by_fold"] = gaps
    res["mean_accuracy_gap"] = round(float(np.mean([g["acc_gap"] for g in gaps])), 4)
    res["mean_logloss_gap"] = round(float(np.mean([g["test_logloss"] - g["train_logloss"] for g in gaps])), 4)

    # ── 2. complexity curve (boosting iterations), staged predictions
    stages = [10, 25, 50, 100, 200, 300, 500, 750, 1000]
    curve = {s: {"train_ll": [], "test_ll": [], "train_acc": [], "test_acc": []} for s in stages}
    for f in range(N_OUTER):
        te = fold == f
        tr_df, te_df = m[~te], m[te]
        clf = make_hgb(max_iter=1000).fit(tr_df[cols], tr_df.y)
        tr_stage = {i + 1: p[:, 1] for i, p in enumerate(clf.staged_predict_proba(tr_df[cols])) if (i + 1) in stages}
        te_stage = {i + 1: p[:, 1] for i, p in enumerate(clf.staged_predict_proba(te_df[cols])) if (i + 1) in stages}
        for s in stages:
            for key, (pp, d) in {"train": (tr_stage[s], tr_df), "test": (te_stage[s], te_df)}.items():
                curve[s][f"{key}_ll"].append(log_loss(d.y, np.clip(pp, 1e-6, 1 - 1e-6)))
                curve[s][f"{key}_acc"].append(((pp >= 0.5).astype(int) == d.y.to_numpy()).mean())
    res["complexity_curve_iterations"] = [
        {"iterations": s, **{k: round(float(np.mean(v)), 4) for k, v in curve[s].items()}} for s in stages]
    best_it = min(res["complexity_curve_iterations"], key=lambda r: r["test_ll"])["iterations"]
    res["best_iterations_by_test_logloss"] = best_it
    print(f"[diag] complexity curve done ({time.time() - t0:.0f}s) best_iter={best_it}", flush=True)

    # ── 3. capacity sweep: tree depth
    cap = []
    for depth in (2, 3, 4, 6, 8, 12):
        tl, el, ta, ea = [], [], [], []
        for f in range(N_OUTER):
            te = fold == f
            tr_df, te_df = m[~te], m[te]
            clf = make_hgb(max_depth=depth).fit(tr_df[cols], tr_df.y)
            ptr, pte = clf.predict_proba(tr_df[cols])[:, 1], clf.predict_proba(te_df[cols])[:, 1]
            tl.append(log_loss(tr_df.y, ptr)); el.append(log_loss(te_df.y, pte))
            ta.append(((ptr >= .5) == tr_df.y).mean()); ea.append(((pte >= .5) == te_df.y).mean())
        cap.append({"max_depth": depth, "train_logloss": round(float(np.mean(tl)), 4), "test_logloss": round(float(np.mean(el)), 4),
                    "train_acc": round(float(np.mean(ta)), 4), "test_acc": round(float(np.mean(ea)), 4),
                    "test_acc_std_across_folds": round(float(np.std(ea)), 4)})
    res["capacity_sweep_depth"] = cap
    print(f"[diag] depth sweep done ({time.time() - t0:.0f}s)", flush=True)

    # ── 4. learning curve over NUMBER OF TRAINING PATIENTS (what matters for new patients)
    rng = np.random.default_rng(SEED)
    lc: Dict[int, Dict[str, List[float]]] = {}
    for f in range(N_OUTER):
        te = fold == f
        tr_all, te_df = m[~te], m[te]
        pats = np.array(sorted(tr_all.patient_id.unique()))
        for k in (3, 6, 9, 12, 15, len(pats)):
            for rep in range(1 if k == len(pats) else 2):
                sel = pats if k == len(pats) else rng.choice(pats, size=k, replace=False)
                tr_df = tr_all[tr_all.patient_id.isin(sel)]
                clf = make_hgb().fit(tr_df[cols], tr_df.y)
                ptr, pte = clf.predict_proba(tr_df[cols])[:, 1], clf.predict_proba(te_df[cols])[:, 1]
                d = lc.setdefault(int(k), {"train_acc": [], "test_acc": [], "train_ll": [], "test_ll": [], "test_auc": []})
                d["train_acc"].append(((ptr >= .5) == tr_df.y).mean()); d["test_acc"].append(((pte >= .5) == te_df.y).mean())
                d["train_ll"].append(log_loss(tr_df.y, ptr)); d["test_ll"].append(log_loss(te_df.y, pte))
                d["test_auc"].append(roc_auc_score(te_df.y, pte))
    res["learning_curve_patients"] = [
        {"train_patients": k, **{kk: round(float(np.mean(v)), 4) for kk, v in d.items()},
         "test_acc_std": round(float(np.std(d["test_acc"])), 4)} for k, d in sorted(lc.items())]
    print(f"[diag] learning curve done ({time.time() - t0:.0f}s)", flush=True)

    # ── 5. bias: subgroup performance & calibration of the nested out-of-fold predictions
    df = nested.merge(
        m[["patient_id", "segment", "step", "source", "glucose", "clock_min", "carbs_2h", "hr_now", "cgm_native"]],
        on=["patient_id", "segment", "step"], how="left")
    df["glucose_range"] = pd.cut(df.glucose, [0, 70, 180, 1000], labels=["<70", "70-180", ">180"]).astype(str)
    df["time_of_day"] = pd.cut(df.clock_min // 60, [-1, 5, 11, 17, 23],
                               labels=["night 00-06", "morning 06-12", "afternoon 12-18", "evening 18-24"]).astype(str)
    df["meal_last_2h"] = np.where(df.carbs_2h > 0, "yes", "no")
    df["hr_available"] = np.where(df.hr_now.notna(), "yes", "no")
    sub = {by: _group_table(df, by) for by in
           ("source", "cgm_native", "glucose_range", "time_of_day", "meal_last_2h", "hr_available")}
    per_patient = _group_table(df, "patient_id")
    res["subgroups"] = sub
    res["per_patient"] = per_patient
    accs = [r["accuracy"] for r in per_patient]
    res["per_patient_accuracy_range"] = [min(accs), max(accs)]
    res["max_abs_calibration_gap_patient"] = round(max(abs(r["calibration_gap"]) for r in per_patient), 3)
    res["disparity"] = {by: {"accuracy_gap_max_min": round(max(r["accuracy"] for r in rows) - min(r["accuracy"] for r in rows), 3),
                             "recall_gap_max_min": round(max(r["recall"] or 0 for r in rows) - min(r["recall"] or 0 for r in rows), 3),
                             "fpr_gap_max_min": round(max(r["false_pos_rate"] or 0 for r in rows) - min(r["false_pos_rate"] or 0 for r in rows), 3)}
                        for by, rows in sub.items()}
    hard = df[df.glucose <= 180]
    res["hard_subset_glucose_le_180"] = metrics(hard.y, hard.p, float(np.median(hard.thr)))
    res["hard_subset_glucose_le_180"]["accuracy"] = round(float(((hard.p >= hard.thr).astype(int) == hard.y).mean()), 4)
    res["hard_subset_rule_baseline_accuracy"] = round(float((hard.y == 0).mean()), 4)   # rule predicts 0 here
    # reliability diagram data
    bins = np.linspace(0, 1, 11)
    df["bin"] = np.clip(np.digitize(df.p, bins[1:-1]), 0, 9)
    res["reliability"] = [{"bin": f"{bins[b]:.1f}-{bins[b + 1]:.1f}", "n": int((df.bin == b).sum()),
                           "mean_pred": round(float(df.p[df.bin == b].mean()), 3), "observed": round(float(df.y[df.bin == b].mean()), 3)}
                          for b in range(10) if (df.bin == b).any()]
    print(f"[diag] subgroup analysis done ({time.time() - t0:.0f}s)", flush=True)

    # ── 6. permutation importance on held-out folds (shortcut / source-proxy check)
    imp = pd.DataFrame(index=cols)
    for f, (clf, iso, thr, best) in enumerate(models):
        te_df = m[fold == f]
        pi = permutation_importance(clf, te_df[cols], te_df.y, scoring="average_precision",
                                    n_repeats=3, random_state=SEED, n_jobs=1)
        imp[f] = pi.importances_mean
    res["permutation_importance_pr_auc_drop"] = (
        imp.mean(axis=1).sort_values(ascending=False).head(15).round(4).to_dict())

    # ── 7. can the source be recovered from the features? (dataset-shift / bias risk)
    src_y = (m.source == "brist").astype(int).to_numpy()
    src_oof = np.full(len(m), np.nan)
    for f in range(N_OUTER):
        te = fold == f
        clf = make_hgb(max_iter=150).fit(m[~te][cols], src_y[~te])
        src_oof[te] = clf.predict_proba(m[te][cols])[:, 1]
    res["source_predictability_auc"] = round(float(roc_auc_score(src_y, src_oof)), 4)
    print(f"[diag] importance + source check done ({time.time() - t0:.0f}s)", flush=True)
    _save("stage_diagnostics", res)
    return res


# ═════════════════════════════════════════════════════════════════════════════
# Stage 4: Ohio native temporal split (comparability with v3.1; patients seen in training)
# ═════════════════════════════════════════════════════════════════════════════

def stage_temporal(m: pd.DataFrame) -> Dict[str, Any]:
    is_ohio_test = m.segment.str.endswith("_test").to_numpy() & (m.source == "ohio").to_numpy()
    is_ohio_train = (~m.segment.str.endswith("_test")).to_numpy() & (m.source == "ohio").to_numpy()
    te = m[is_ohio_test]
    out = {"test_rows": int(len(te)), "test_patients": int(te.patient_id.nunique()), "test_prevalence": round(float(te.y.mean()), 4),
           "rule_glucose_gt_180_accuracy": round(float(((te.glucose > 180).astype(int) == te.y).mean()), 4),
           "note": "Patients also appear in training (earlier weeks): optimistic vs patient-held-out protocol. "
                   "Previous v3.1 reported accuracy 0.846 / PR-AUC 0.961 (Random Forest) on its own row selection."}
    for fname, cols in (("base", P.BASE_FEATURES), ("engineered", P.ENGINEERED_FEATURES)):
        for dname, trm in (("ohio-only", is_ohio_train), ("combined", is_ohio_train | (m.source == "brist").to_numpy())):
            tr = m[trm].reset_index(drop=True)
            inner = _inner_cv(tr, cols, {})
            thr = best_threshold(tr.y.to_numpy(), inner)
            clf = make_hgb().fit(tr[cols], tr.y)
            p = clf.predict_proba(te[cols])[:, 1]
            r = metrics(te.y, p, thr)
            r["accuracy_at_0.5"] = metrics(te.y, p, 0.5)["accuracy"]
            out[f"HGB {fname} | {dname}"] = r
            print(f"[temporal] {fname}/{dname}: acc={r['accuracy']:.3f} pr_auc={r['pr_auc']:.3f}", flush=True)
    _save("stage_temporal", out)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Stage 5: final model on all 21 patients
# ═════════════════════════════════════════════════════════════════════════════

def stage_final(m: pd.DataFrame, cols: Optional[List[str]] = None) -> Dict[str, Any]:
    cols = cols or P.ENGINEERED_FEATURES
    tuned = _load("stage_tune")
    nested = pd.read_parquet(OUT_DIR / "oof_nested.parquet")
    cfg_name = max(tuned["selected_configs"], key=tuned["selected_configs"].get)
    cfg = TUNE_GRID[cfg_name]
    thr = best_threshold(nested.y.to_numpy(), nested.p.to_numpy())       # from out-of-fold predictions only
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(nested.p, nested.y)
    clf = make_hgb(**cfg).fit(m[cols], m.y)
    use_cal = (tuned["pooled_nested_calibrated"]["all"]["brier"] < tuned["pooled_nested"]["all"]["brier"]
               and tuned["pooled_nested_calibrated"]["all"]["ece"] < tuned["pooled_nested"]["all"]["ece"])
    artifact = {
        "model": clf, "calibrator": iso if use_cal else None, "feature_cols": cols, "threshold": thr,
        "version": MODEL_VERSION, "config_name": cfg_name, "config": {**HGB_DEFAULT, **cfg},
        "grid": "15-minute bins; label = max glucose over next 8 bins > 180 mg/dL",
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(artifact, OUT_DIR / "model.joblib")
    card = {
        "model_name": f"GlucoTwin real-data v{MODEL_VERSION}", "algorithm": "HistGradientBoostingClassifier",
        "config": artifact["config"], "config_name": cfg_name, "n_features": len(cols), "feature_cols": cols,
        "threshold": thr, "calibrated": use_cal,
        "data": {"sources": ["OhioT1DM 2018+2020 (12 patients)", "BrisT1D Kaggle train.csv (9 patients, windows stitched)"],
                 "rows": int(len(m)), "patients": int(m.patient_id.nunique()), "prevalence": round(float(m.y.mean()), 4)},
        "validation": "7-fold patient-grouped nested CV (see stage_tune.json); nothing from the held-out patients influenced "
                      "config, threshold or calibration of their predictions",
        "expected_performance": {"pooled_nested": tuned["pooled_nested"]["all"], "bootstrap_95ci": tuned["bootstrap"]["model"]},
        "not_wired_into_api": "The FastAPI app serves the synthetic v2 model. This model needs carbs/insulin/CGM history "
                              "that the synthetic patients do not have.",
        "limitations": ["21 T1D patients only; T2D / non-diabetic not represented.",
                        "Label resolution differs by CGM device (5-min vs 15-min).",
                        "Not clinically validated; experimental."],
    }
    _save("model_card", card)
    print(f"[final] saved model.joblib (config={cfg_name}, thr={thr}, calibrated={use_cal})", flush=True)
    return card


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["ablation", "tune", "diagnostics", "temporal", "final", "all"])
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m = load_frame()
    stages = ["ablation", "tune", "diagnostics", "temporal", "final"] if a.stage == "all" else [a.stage]
    for s in stages:
        t = time.time()
        print(f"\n===== stage: {s} =====", flush=True)
        {"ablation": stage_ablation, "tune": stage_tune, "diagnostics": stage_diagnostics,
         "temporal": stage_temporal, "final": stage_final}[s](m)
        print(f"===== {s} finished in {time.time() - t:.0f}s =====", flush=True)


if __name__ == "__main__":
    main()
