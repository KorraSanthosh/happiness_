"""
v7 - the Digital Twin Challenge submission model.

  * Trained ONLY on openly licensed (CC BY 4.0) data: BrisT1D, HUPA-UCM, T1D-UOM, AZT1D, Shanghai T1DM and
    Shanghai T2DM -> 6 datasets, Type 1 AND Type 2 diabetes.
  * Fuses two streams, as the challenge requires: static EHR (real clinical summaries for the Shanghai
    patients - demographics, diabetes history, complications, drugs, labs; diabetes type for everyone) and
    the dynamic CGM / insulin / meal / wearable stream (features from real_data_pipeline).
  * Three horizons: 30 min, 60 min and 2 h (2 h = the challenge's example task).
  * OhioT1DM is under a data-use agreement and cannot be published, so it is NEVER used for training; it is
    only a local external-validation dataset.
  * Two operating points per horizon, both chosen inside the cross-validation (no test data involved):
      accuracy mode - threshold maximising accuracy;  safety mode - threshold maximising balanced accuracy.

Stages: python -m app.real_data_v7 --stage {cv, ablation, final, all}  [--rebuild]
Outputs: models/combined_v7/   (v5/v6 outputs and the served API model are untouched)
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

from app import external_loaders as E
from app import real_data_pipeline as P
from app.real_data_eval import (
    HGB_DEFAULT, N_INNER, N_OUTER, SEED, TUNE_GRID, _group_table, best_threshold, cluster_bootstrap,
    make_hgb, metrics, patient_level,
)

warnings.filterwarnings("ignore")

OUT_DIR = P.ROOT / "models" / "combined_v7"
OPEN = ("brist", "hupa", "uom", "azt1d", "sh_t1", "sh_t2")
EXTERNAL = ("ohio",)
HORIZONS = {"30m": "y_30m", "60m": "y_60m", "2h": "y"}
CFG = TUNE_GRID["slow"]               # chosen by nested tuning in 7/7 v6 folds
CAP_ROWS = 6000
MODEL_VERSION = "7.0.0"
FEATURES: List[str] = P.V6_FEATURES + E.EHR_FEATURES
DATASET_INFO = {
    "brist": "BrisT1D (Univ. of Bristol, CC BY 4.0) - Type 1",
    "hupa": "HUPA-UCM (Hidalgo et al. 2024, CC BY 4.0) - Type 1",
    "uom": "T1D-UOM (Univ. of Manchester 2025, CC BY 4.0) - Type 1",
    "azt1d": "AZT1D (Mayo Clinic Arizona 2025, CC BY 4.0) - Type 1",
    "sh_t1": "Shanghai T1DM (Zhao et al. 2023, CC BY 4.0) - Type 1, with clinical EHR",
    "sh_t2": "Shanghai T2DM (Zhao et al. 2023, CC BY 4.0) - Type 2, with clinical EHR",
    "ohio": "OhioT1DM (data-use agreement) - external validation only, never trained on",
}


# ═════════════════════════════════════════════════════════════════════════════
# data
# ═════════════════════════════════════════════════════════════════════════════

def load_frame(rebuild: bool = False) -> pd.DataFrame:
    path = OUT_DIR / "frame.parquet"
    if path.exists() and not rebuild:
        m = pd.read_parquet(path)
        print(f"[data] cached frame: {len(m):,} rows | {m.patient_id.nunique()} people", flush=True)
        return m
    grids = [P.load_brist_grid("train")[0], E.load_external_grids(("hupa", "uom", "azt1d", "sh_t1", "sh_t2"))[0],
             P.load_ohio_grid()]
    grid = pd.concat(grids, ignore_index=True)
    m = P.modelling_frame_multi(P.build_features(grid))
    obs = grid.groupby("patient_id")["glucose"].apply(lambda s: s.notna().mean())
    m["cgm_native"] = m["patient_id"].map(lambda p: "15min" if obs[p] < 0.6 else "5min")
    m["recording"] = E.recording_of_segment(m["segment"])
    m = m.merge(E.shanghai_ehr(), on="recording", how="left")
    m.loc[~m.source.isin(("sh_t1", "sh_t2")), "ehr_t2d"] = 0.0           # every other dataset is Type 1
    m["diabetes_type"] = np.where(m["ehr_t2d"] == 1, "T2D", "T1D")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m.to_parquet(path)
    print(f"[data] built frame: {len(m):,} rows | {m.patient_id.nunique()} people | "
          f"{m.groupby('source').patient_id.nunique().to_dict()}", flush=True)
    return m


def assign_folds(m: pd.DataFrame) -> Dict[str, int]:
    rng = np.random.default_rng(SEED)
    fold_of, i = {}, 0
    for src in OPEN:
        ps = sorted(m.loc[m.source == src, "patient_id"].unique())
        rng.shuffle(ps)
        for p in ps:
            fold_of[p] = i % N_OUTER
            i += 1
    return fold_of


def weights(df: pd.DataFrame) -> np.ndarray:
    n = df.groupby("patient_id")["patient_id"].transform("size").to_numpy()
    return np.minimum(1.0, CAP_ROWS / n)


def fit(df: pd.DataFrame, label: str, cols: List[str] = FEATURES, cfg: Dict[str, Any] = None):
    return make_hgb(**(CFG if cfg is None else cfg)).fit(df[cols], df[label].astype(int), sample_weight=weights(df))


def acc_threshold(y: np.ndarray, p: np.ndarray) -> float:
    ts = np.arange(0.2, 0.801, 0.01)
    return round(float(ts[int(np.argmax([((p >= t).astype(int) == y).mean() for t in ts]))]), 2)


def _subsets(m: pd.DataFrame) -> Dict[str, np.ndarray]:
    out = {"all_open": np.ones(len(m), bool), "type1": (m.diabetes_type == "T1D").to_numpy(),
           "type2": (m.diabetes_type == "T2D").to_numpy()}
    for s in OPEN:
        out[s] = (m.source == s).to_numpy()
    return out


def _block(d: pd.DataFrame, p: np.ndarray, thr: np.ndarray, label: str) -> Dict[str, Any]:
    out = {}
    y = d[label].astype(int).to_numpy()
    rule = (d.glucose > 180).astype(int).to_numpy()
    for name, mk in _subsets(d).items():
        if mk.sum() == 0:
            continue
        r = metrics(y[mk], p[mk], thr[mk])                 # per-row (fold) thresholds for every metric
        r["rule_accuracy"] = round(float((rule[mk] == y[mk]).mean()), 4)
        r["people"] = int(d.patient_id[mk].nunique())
        r["per_person_accuracy"] = patient_level(d[mk].assign(y=y[mk]), p[mk], thr[mk])
        out[name] = r
    return out


def _save(name: str, obj: Any) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[save] {OUT_DIR / (name + '.json')}", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# stage cv: nested, person-grouped cross-validation for each horizon
# ═════════════════════════════════════════════════════════════════════════════

def stage_cv(m: pd.DataFrame) -> Dict[str, Any]:
    fold_of = assign_folds(m)
    res: Dict[str, Any] = {"features": len(FEATURES), "config": {**HGB_DEFAULT, **CFG}, "horizons": {}}
    t0 = time.time()
    for hname, label in HORIZONS.items():
        d = m[m.source.isin(OPEN) & m[label].notna()].reset_index(drop=True)
        fold = d.patient_id.map(fold_of).to_numpy()
        p = np.full(len(d), np.nan)
        thr_acc, thr_bal = np.full(len(d), np.nan), np.full(len(d), np.nan)
        folds = []
        for f in range(N_OUTER):
            te = fold == f
            tr = d[~te].reset_index(drop=True)
            inner = np.full(len(tr), np.nan)
            for a, b in GroupKFold(N_INNER).split(tr, tr[label], tr.patient_id):
                inner[b] = fit(tr.iloc[a], label).predict_proba(tr.iloc[b][FEATURES])[:, 1]
            ta, tb = acc_threshold(tr[label].astype(int).to_numpy(), inner), best_threshold(tr[label].astype(int).to_numpy(), inner)
            clf = fit(tr, label)
            p[te] = clf.predict_proba(d.loc[te, FEATURES])[:, 1]
            p_tr = clf.predict_proba(tr[FEATURES])[:, 1]
            thr_acc[te], thr_bal[te] = ta, tb
            folds.append({"fold": f, "threshold_accuracy_mode": ta, "threshold_safety_mode": tb,
                          "train_acc": round(float(((p_tr >= ta).astype(int) == tr[label]).mean()), 4),
                          "test_acc": round(float(((p[te] >= ta).astype(int) == d.loc[te, label]).mean()), 4)})
            print(f"[cv {hname}] fold {f + 1}/{N_OUTER} thr={ta}/{tb} train={folds[-1]['train_acc']:.3f} "
                  f"test={folds[-1]['test_acc']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        yy = d[label].astype(int).to_numpy()
        res["horizons"][hname] = {
            "rows": int(len(d)), "people": int(d.patient_id.nunique()), "prevalence": round(float(yy.mean()), 4),
            "accuracy_mode": _block(d, p, thr_acc, label),
            "safety_mode": _block(d, p, thr_bal, label),
            "folds": folds,
            "overfit_gap_train_minus_test_acc": round(float(np.mean([x["train_acc"] - x["test_acc"] for x in folds])), 4),
            "bootstrap_accuracy_mode": cluster_bootstrap(d.assign(y=yy), {"m": p}, {"m": thr_acc}, n=300)["m"],
            "thresholds": {"accuracy_mode": float(np.median([x["threshold_accuracy_mode"] for x in folds])),
                           "safety_mode": float(np.median([x["threshold_safety_mode"] for x in folds]))},
        }
        pd.DataFrame({"patient_id": d.patient_id, "segment": d.segment, "step": d.step, "y": yy, "p": p,
                      "thr_acc": thr_acc, "thr_bal": thr_bal, "fold": fold}).to_parquet(OUT_DIR / f"oof_{hname}.parquet")
    _save("stage_cv", res)
    return res


# ═════════════════════════════════════════════════════════════════════════════
# stage ablation: is the EHR + CGM FUSION better than either stream alone?
# ═════════════════════════════════════════════════════════════════════════════

def stage_ablation(m: pd.DataFrame) -> Dict[str, Any]:
    label = "y"
    d = m[m.source.isin(OPEN) & m[label].notna()].reset_index(drop=True)
    fold = d.patient_id.map(assign_folds(m)).to_numpy()
    variants = {"EHR only (static record)": E.EHR_FEATURES,
                "CGM/wearable only (dynamic)": P.V6_FEATURES,
                "Fusion (EHR + CGM)": FEATURES}
    oof = {k: np.full(len(d), np.nan) for k in variants}
    for f in range(N_OUTER):
        te = fold == f
        for k, cols in variants.items():
            oof[k][te] = fit(d[~te], label, cols).predict_proba(d.loc[te, cols])[:, 1]
        print(f"[ablation] fold {f + 1}/{N_OUTER}", flush=True)
    sh = d.source.isin(("sh_t1", "sh_t2")).to_numpy()
    t2 = (d.diabetes_type == "T2D").to_numpy()
    res: Dict[str, Any] = {"evaluated_on": "people with a real clinical EHR (Shanghai T1D+T2D), person-held-out, 2 h horizon",
                           "variants": {}}
    for k, p in oof.items():
        res["variants"][k] = {"shanghai_with_ehr": metrics(d.y[sh], p[sh], 0.5), "type2_only": metrics(d.y[t2], p[t2], 0.5),
                              "all_open": metrics(d.y, p, 0.5)}
    sub = d[sh].reset_index(drop=True)
    bs = cluster_bootstrap(sub, {"cgm": oof["CGM/wearable only (dynamic)"][sh], "fusion": oof["Fusion (EHR + CGM)"][sh]},
                           {"cgm": 0.5, "fusion": 0.5}, n=300)
    res["fusion_vs_cgm_only_on_shanghai"] = bs["fusion"]["vs_cgm"]
    # cold start: does the static record help while the twin has little CGM history of this person?
    res["cold_start_shanghai"] = {}
    for name, mk in (("first 24 h of a recording", sh & (d.p_hist_days < 1).to_numpy()),
                     ("after 24 h", sh & (d.p_hist_days >= 1).to_numpy())):
        part = d[mk].reset_index(drop=True)
        b2 = cluster_bootstrap(part, {"cgm": oof["CGM/wearable only (dynamic)"][mk], "fusion": oof["Fusion (EHR + CGM)"][mk]},
                               {"cgm": 0.5, "fusion": 0.5}, n=300)
        res["cold_start_shanghai"][name] = {
            "rows": int(mk.sum()),
            "cgm_only": metrics(d.y[mk], oof["CGM/wearable only (dynamic)"][mk], 0.5),
            "fusion": metrics(d.y[mk], oof["Fusion (EHR + CGM)"][mk], 0.5),
            "fusion_minus_cgm": b2["fusion"]["vs_cgm"]}
    _save("stage_ablation", res)
    return res


# ═════════════════════════════════════════════════════════════════════════════
# stage final: models on all open data, Ohio external validation, bias, importance, model card
# ═════════════════════════════════════════════════════════════════════════════

def stage_final(m: pd.DataFrame) -> Dict[str, Any]:
    cv = json.load(open(OUT_DIR / "stage_cv.json"))
    res: Dict[str, Any] = {"models": {}, "ohio_external_validation": {}}
    for hname, label in HORIZONS.items():
        d = m[m.source.isin(OPEN) & m[label].notna()]
        clf = fit(d, label)
        thr = cv["horizons"][hname]["thresholds"]
        path = OUT_DIR / f"model_{hname}.joblib"
        joblib.dump({"model": clf, "feature_cols": FEATURES, "label": label, "horizon": hname,
                     "threshold_accuracy_mode": thr["accuracy_mode"], "threshold_safety_mode": thr["safety_mode"],
                     "version": MODEL_VERSION, "config": {**HGB_DEFAULT, **CFG},
                     "trained_on": {s: DATASET_INFO[s] for s in OPEN}, "people": int(d.patient_id.nunique()),
                     "trained_at": datetime.now(timezone.utc).isoformat()}, path)
        res["models"][hname] = {"path": str(path), "thresholds": thr, "rows": int(len(d)), "people": int(d.patient_id.nunique())}
        o = m[m.source.isin(EXTERNAL) & m[label].notna()]
        po = clf.predict_proba(o[FEATURES])[:, 1]
        yo = o[label].astype(int).to_numpy()
        res["ohio_external_validation"][hname] = {
            "accuracy_mode": metrics(yo, po, thr["accuracy_mode"]), "safety_mode": metrics(yo, po, thr["safety_mode"]),
            "rule_accuracy": round(float(((o.glucose > 180).astype(int) == yo).mean()), 4), "people": int(o.patient_id.nunique())}
        print(f"[final] {hname}: saved; Ohio external acc={res['ohio_external_validation'][hname]['accuracy_mode']['accuracy']:.3f}", flush=True)
        if hname == "2h":
            te = d.sample(n=min(20000, len(d)), random_state=SEED)
            pi = permutation_importance(clf, te[FEATURES], te[label].astype(int), scoring="average_precision",
                                        n_repeats=3, random_state=SEED)
            res["permutation_importance_2h_top20"] = (pd.Series(pi.importances_mean, index=FEATURES)
                                                      .sort_values(ascending=False).head(20).round(4).to_dict())
    # bias / subgroups from the cross-validated 2 h predictions (accuracy mode)
    oof = pd.read_parquet(OUT_DIR / "oof_2h.parquet")
    d = m[m.source.isin(OPEN) & m["y"].notna()].reset_index(drop=True)
    df = pd.DataFrame({"y": oof.y, "p": oof.p, "thr": oof.thr_acc, "source": d.source, "diabetes_type": d.diabetes_type,
                       "glucose": d.glucose, "clock_min": d.clock_min, "cgm_native": d.cgm_native,
                       "sex": np.where(d.ehr_male == 1, "male", np.where(d.ehr_male == 0, "female", "unknown")),
                       "age_group": pd.cut(d.ehr_age, [0, 40, 60, 120], labels=["<40", "40-60", "60+"]).astype(str),
                       "bmi_group": pd.cut(d.ehr_bmi, [0, 23, 27.5, 100], labels=["<23", "23-27.5", "27.5+"]).astype(str)})
    df["time_of_day"] = pd.cut(df.clock_min // 60, [-1, 5, 11, 17, 23],
                               labels=["night 00-06", "morning 06-12", "afternoon 12-18", "evening 18-24"]).astype(str)
    df["glucose_range"] = pd.cut(df.glucose, [0, 70, 180, 1000], labels=["<70", "70-180", ">180"]).astype(str)
    ehr_rows = df.sex != "unknown"
    res["bias"] = {by: _group_table(df, by) for by in ("source", "diabetes_type", "time_of_day", "glucose_range", "cgm_native")}
    res["bias"].update({by + " (Shanghai, real EHR)": _group_table(df[ehr_rows], by) for by in ("sex", "age_group", "bmi_group")})
    _save("stage_final", res)
    _save("model_card", {
        "model_name": f"GlucoTwin Digital Twin v{MODEL_VERSION}", "algorithm": "HistGradientBoostingClassifier (scikit-learn)",
        "task": "Probability that CGM glucose exceeds 180 mg/dL within the next 30 min / 60 min / 2 h",
        "inputs": {"static_ehr": E.EHR_FEATURES, "dynamic_stream_features": P.V6_FEATURES},
        "training_data": {s: DATASET_INFO[s] for s in OPEN}, "external_validation": DATASET_INFO["ohio"],
        "validation": "7-fold person-grouped nested cross-validation (thresholds chosen on inner folds only)",
        "operating_points": "accuracy mode (max accuracy) and safety mode (max balanced accuracy -> higher recall)",
        "performance": {h: {"accuracy_mode": cv["horizons"][h]["accuracy_mode"]["all_open"],
                            "safety_mode": cv["horizons"][h]["safety_mode"]["all_open"],
                            "ohio_external": res["ohio_external_validation"][h]["accuracy_mode"]} for h in HORIZONS},
        "limitations": ["Research prototype, not a medical device; not clinically validated.",
                        "Real EHR exists only for the Shanghai cohorts; other datasets carry diabetes type only.",
                        "Spikes without a recorded cause (stress, illness, unlogged food) are not predictable from these inputs."],
        "attribution": "Dataset authors listed in training_data (CC BY 4.0).",
    })
    return res


def stage_rescore(m: pd.DataFrame) -> Dict[str, Any]:
    """Recompute the stage_cv metric blocks (and the model card's performance) from the SAVED held-out
    predictions - no training. Used after fixing the per-row-threshold reporting (2026-10-23)."""
    cv = json.load(open(OUT_DIR / "stage_cv.json"))
    cols = ["segment", "step", "source", "diabetes_type", "glucose"]
    for hname in HORIZONS:
        o = pd.read_parquet(OUT_DIR / f"oof_{hname}.parquet").merge(m[cols], on=["segment", "step"], how="left")
        cv["horizons"][hname]["accuracy_mode"] = _block(o, o.p.to_numpy(), o.thr_acc.to_numpy(), "y")
        cv["horizons"][hname]["safety_mode"] = _block(o, o.p.to_numpy(), o.thr_bal.to_numpy(), "y")
    cv["rescored"] = "metrics recomputed from saved OOF predictions with per-row fold thresholds (no retraining)"
    _save("stage_cv", cv)
    card_path = OUT_DIR / "model_card.json"
    if card_path.exists():
        card = json.load(open(card_path))
        for h in HORIZONS:
            card["performance"][h]["accuracy_mode"] = cv["horizons"][h]["accuracy_mode"]["all_open"]
            card["performance"][h]["safety_mode"] = cv["horizons"][h]["safety_mode"]["all_open"]
        _save("model_card", card)
    return cv


STAGES = {"cv": stage_cv, "ablation": stage_ablation, "final": stage_final, "rescore": stage_rescore}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=list(STAGES) + ["all"])
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    m = load_frame(a.rebuild)
    for s in (list(STAGES) if a.stage == "all" else [a.stage]):
        t = time.time()
        print(f"\n===== stage: {s} =====", flush=True)
        STAGES[s](m)
        print(f"===== {s} finished in {time.time() - t:.0f}s =====", flush=True)


if __name__ == "__main__":
    main()
