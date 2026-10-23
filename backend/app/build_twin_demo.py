"""
Build the Digital Twin dashboard demo pack (run once, offline):

  python -m app.build_twin_demo

  1. Picks demo patients from the CC BY 4.0 datasets only (Shanghai T2DM/T1DM, AZT1D, T1D-UOM, HUPA-UCM).
     BrisT1D (Kaggle terms unverified) and OhioT1DM (data-use agreement) are never shipped.
  2. Trains the 30 min / 60 min / 2 h demo models on all open data EXCEPT the demo patients, so every
     prediction the dashboard shows is for a person the model has never seen.
  3. Writes backend/data/twin_v7/ : raw 5-min streams + EHR of the demo patients, and
     backend/models/twin_v7/ : the three demo models + their held-out evaluation on the demo patients.

The thresholds come from the v7 nested cross-validation (models/combined_v7/stage_cv.json).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app import external_loaders as E
from app import real_data_v7 as V7
from app.real_data_eval import metrics

BACKEND = Path(__file__).resolve().parents[1]
PACK_DIR = BACKEND / "data" / "twin_v7"
MODEL_DIR = BACKEND / "models" / "twin_v7"
DEMO_PER_SOURCE = {"sh_t2": 12, "sh_t1": 5, "azt1d": 5, "uom": 4, "hupa": 4}
MAX_DAYS = 14                          # keep the shipped streams small: up to 14 days per patient


def _pick_demo_patients(m: pd.DataFrame) -> list:
    """Deterministic: per source, the people with the most labelled 2 h rows that have both spikes and non-spikes."""
    picks = []
    for src, k in DEMO_PER_SOURCE.items():
        d = m[(m.source == src) & m.y.notna()]
        stats = d.groupby("patient_id").agg(n=("y", "size"), prev=("y", "mean"))
        stats = stats[(stats.prev > 0.1) & (stats.prev < 0.9)].sort_values("n", ascending=False)
        picks += stats.index[:k].tolist()
    return picks


def write_summary() -> None:
    """Dashboard model page summary from the v7 result files (no training, no patient-level data)."""
    cv = json.load(open(V7.OUT_DIR / "stage_cv.json"))
    fin = json.load(open(V7.OUT_DIR / "stage_final.json"))
    abl = json.load(open(V7.OUT_DIR / "stage_ablation.json"))
    pick = lambda r: {k: r.get(k) for k in ("accuracy", "balanced_accuracy", "recall", "precision", "roc_auc", "rule_accuracy", "people", "n")}
    summary = {
        "version": V7.MODEL_VERSION, "datasets": {s: V7.DATASET_INFO[s] for s in V7.OPEN + V7.EXTERNAL},
        "horizons": {h: {"accuracy_mode": {s: pick(r) for s, r in cv["horizons"][h]["accuracy_mode"].items()},
                         "safety_mode": pick(cv["horizons"][h]["safety_mode"]["all_open"]),
                         "accuracy_ci95": cv["horizons"][h]["bootstrap_accuracy_mode"]["accuracy_ci95"],
                         "overfit_gap": cv["horizons"][h]["overfit_gap_train_minus_test_acc"],
                         "ohio_external": pick(fin["ohio_external_validation"][h]["accuracy_mode"]) |
                                          {"rule_accuracy": fin["ohio_external_validation"][h]["rule_accuracy"]}}
                     for h in V7.HORIZONS},
        "fusion_ablation": {k: pick(v["shanghai_with_ehr"]) for k, v in abl["variants"].items()},
        "fusion_vs_cgm_only": abl["fusion_vs_cgm_only_on_shanghai"],
        "bias": {k: [{"group": r.get(k.split(" (")[0]), **{kk: r.get(kk) for kk in ("n", "accuracy", "recall", "calibration_gap")}}
                     for r in rows] for k, rows in fin["bias"].items()},
        "top_features_2h": fin.get("permutation_importance_2h_top20", {}),
    }
    with open(MODEL_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[demo] wrote {MODEL_DIR / 'summary.json'}")


def main() -> None:
    m = V7.load_frame()
    demo = _pick_demo_patients(m)
    print(f"[demo] {len(demo)} demo patients: {demo}")
    cv = json.load(open(V7.OUT_DIR / "stage_cv.json"))

    # 1) models trained without the demo patients
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    held = m[m.patient_id.isin(demo)]
    evaluation = {}
    for hname, label in V7.HORIZONS.items():
        train = m[m.source.isin(V7.OPEN) & ~m.patient_id.isin(demo) & m[label].notna()]
        clf = V7.fit(train, label)
        thr = cv["horizons"][hname]["thresholds"]
        joblib.dump({"model": clf, "feature_cols": V7.FEATURES, "label": label, "horizon": hname,
                     "threshold_accuracy_mode": thr["accuracy_mode"], "threshold_safety_mode": thr["safety_mode"],
                     "version": V7.MODEL_VERSION, "excluded_patients": demo,
                     "trained_people": int(train.patient_id.nunique())}, MODEL_DIR / f"model_{hname}.joblib")
        h = held[held[label].notna()]
        p = clf.predict_proba(h[V7.FEATURES])[:, 1]
        evaluation[hname] = {"accuracy_mode": metrics(h[label].astype(int), p, thr["accuracy_mode"]),
                             "safety_mode": metrics(h[label].astype(int), p, thr["safety_mode"]),
                             "rule_accuracy": round(float(((h.glucose > 180).astype(int) == h[label]).mean()), 4)}
        print(f"[demo] {hname}: trained on {train.patient_id.nunique()} people; held-out demo accuracy "
              f"{evaluation[hname]['accuracy_mode']['accuracy']:.3f}")

    # 2) raw 5-min streams of the demo patients (only CC BY 4.0 sources)
    grid = E.load_external_grids(("sh_t2", "sh_t1", "azt1d", "uom", "hupa"))[0]
    grid = grid[grid.patient_id.isin(demo)].copy()
    keep = []
    for pid, g in grid.groupby("patient_id", sort=False):
        segs = g.groupby("segment").size().sort_values(ascending=False)
        seg = segs.index[0]                                      # the longest recording of each person
        s = g[g.segment == seg].sort_values("step")
        keep.append(s.iloc[: MAX_DAYS * 288])
    stream = pd.concat(keep, ignore_index=True)
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    stream.to_parquet(PACK_DIR / "streams.parquet", index=False)

    # 3) static EHR per demo patient (Shanghai: real clinical summary of that recording; others: diabetes type)
    #    + an "example early warning" moment: glucose still <= 165 now, the held-out 2 h model says spike, a spike
    #    really follows, but not within 30 min (so it is a genuine early warning). Chosen for illustration only;
    #    the dashboard slider shows every moment, including misses.
    ehr = E.shanghai_ehr().set_index("recording")
    m2h = joblib.load(MODEL_DIR / "model_2h.joblib")
    people = []
    for pid, s in stream.groupby("patient_id", sort=False):
        seg = s.segment.iloc[0]
        rows = m[(m.segment == seg) & (m.step < len(s) // 3)].sort_values("step")
        p2h = m2h["model"].predict_proba(rows[V7.FEATURES])[:, 1] if len(rows) else np.array([])
        ok = (rows.step >= 2 * 96) & (rows.glucose <= 165) & (rows.y == 1) & (rows.y_30m == 0)
        hit = rows.step[ok.to_numpy() & (p2h >= m2h["threshold_accuracy_mode"])]
        demo_bin = int(hit.iloc[0]) if len(hit) else None
        rec = E.recording_of_segment(pd.Series([seg])).iloc[0]
        row = ehr.loc[rec].to_dict() if isinstance(rec, str) and rec in ehr.index else {}
        src = s.source.iloc[0]
        people.append({"patient_id": pid, "source": src, "segment": seg, "dataset": V7.DATASET_INFO[src],
                       "demo_bin": demo_bin,
                       "diabetes_type": "T2D" if src == "sh_t2" else "T1D", "days": round(len(s) / 288, 1),
                       "ehr": {k: (None if pd.isna(row.get(k, np.nan)) else float(row[k])) for k in E.EHR_FEATURES}
                       if row else {"ehr_t2d": 0.0}})
    with open(PACK_DIR / "patients.json", "w") as f:
        json.dump(people, f, indent=2)
    # 4) compact results summary for the dashboard's model page
    write_summary()

    with open(MODEL_DIR / "demo_evaluation.json", "w") as f:
        json.dump({"demo_patients": demo, "note": "demo patients were excluded from training of these models",
                   "evaluation": evaluation}, f, indent=2)
    with open(PACK_DIR / "ATTRIBUTION.md", "w") as f:
        f.write("# Demo data attribution\n\nDerived (resampled, harmonised) from datasets licensed CC BY 4.0:\n\n"
                + "\n".join(f"- {V7.DATASET_INFO[s]}" for s in DEMO_PER_SOURCE) +
                "\n\nShanghai: Zhao et al., *Chinese diabetes datasets for data-driven machine learning*, Sci Data 2023, "
                "doi:10.6084/m9.figshare.20444397.\nHUPA-UCM: Hidalgo et al., Data in Brief 2024, doi:10.17632/3hbcscwz44.1.\n"
                "T1D-UOM: University of Manchester, doi:10.5281/zenodo.15806142.\nAZT1D: Mayo Clinic Arizona, doi:10.17632/gk9m674wcx.1.\n")
    print(f"[demo] wrote {PACK_DIR} ({stream.memory_usage(deep=True).sum() / 1e6:.1f} MB in memory) and {MODEL_DIR}")


if __name__ == "__main__":
    import sys
    write_summary() if "--summary-only" in sys.argv else main()
