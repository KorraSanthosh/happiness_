"""
v9 serving build - run only after `real_data_v9 --stage final` passed the pre-registered promotion rule.

  python3 -I scripts/run_isolated.py -m:app.build_twin_v9

  1. final 2 h model: the selected v9 feature set, trained on all 187 open-data people
       -> models/combined_v9/model_final.joblib            (git-ignored, like every combined_v*/*.joblib)
  2. demo 2 h model: the same, trained WITHOUT the 30 dashboard demo patients
       -> backend/models/twin_v9/model_2h.joblib (+ demo_evaluation.json)
  3. the extra inputs the v9 model reads for the Shanghai demo patients (estimated meal carbs, oral-drug doses)
       -> backend/data/twin_v7/streams_v9_extra.parquet   (streams.parquet itself is left bit-identical)
  4. "example early warning" moments re-picked with the v9 demo model, by the same rule as the v7 pack
       -> backend/models/twin_v9/demo_bins.json            (--examples-only redoes just this step)

Only the 2 h horizon changes: v9 validated nothing else, so the 30 min / 1 h dashboard models stay v7.
Thresholds all come from DEV validation (stage_policy.json, oof_dev.parquet); none is tuned on TEST.
Rollback to pure v7 serving: delete backend/models/twin_v9/.
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from app import build_twin_demo as D
from app import external_loaders as E
from app import real_data_v7 as V7
from app import real_data_v9 as V9
from app.real_data_eval import metrics

MODEL_DIR = D.BACKEND / "models" / "twin_v9"
EXTRA_PATH = D.PACK_DIR / "streams_v9_extra.parquet"
MODEL_VERSION = "v9.0-open-data"


def thresholds() -> dict:
    cv, pol = V9._load("stage_cv"), V9._load("stage_policy")
    name = V9.selected(cv)[0]
    o = pd.read_parquet(V9.OUT / "oof_dev.parquet")
    return {"candidate": name, "accuracy_mode": pol["models"][name]["accuracy_threshold"],
            "safety_mode": float(np.median(o[f"thrbal::{name}"])),
            "early_warning": pol["models"][name]["chosen"]["threshold"]}


def pick_examples(m: pd.DataFrame) -> None:
    """Same rule as build_twin_demo: glucose <= 165 now, a real spike follows but not within 30 min, model says spike."""
    b = joblib.load(MODEL_DIR / "model_2h.joblib")
    stream = pd.read_parquet(D.PACK_DIR / "streams.parquet", columns=["patient_id", "segment", "step"])
    out = {}
    for pid, s in stream.groupby("patient_id", sort=False):
        rows = m[(m.segment == s.segment.iloc[0]) & (m.step < len(s) // 3)].sort_values("step")
        rows = V9.base_inputs(rows) if b["mask_shanghai_carbs"] else rows
        p = b["model"].predict_proba(rows[b["feature_cols"]])[:, 1] if len(rows) else np.array([])
        ok = (rows.step >= 2 * 96) & (rows.glucose <= 165) & (rows.y == 1) & (rows.y_30m == 0)
        hit = rows.step[ok.to_numpy() & (p >= b["threshold_accuracy_mode"])]
        out[pid] = int(hit.iloc[0]) if len(hit) else None
    with open(MODEL_DIR / "demo_bins.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"[v9] example warning moments: {sum(v is not None for v in out.values())} of {len(out)} people")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples-only", action="store_true")
    if ap.parse_args().examples_only:
        return pick_examples(V9.load_frame())
    fin = V9._load("stage_final")
    assert fin["selected"] != "base (v7 inputs)", "v9 selected the baseline - nothing to serve"
    name, cols, mask = V9.selected(V9._load("stage_cv"))
    thr = thresholds()
    m = V9.load_frame()
    lab = m[m.source.isin(V7.OPEN) & m[V9.LABEL].notna()]
    lab = V9.base_inputs(lab) if mask else lab
    common = {"feature_cols": cols, "label": V9.LABEL, "horizon": "2h", "version": MODEL_VERSION, "candidate": name,
              "threshold_accuracy_mode": thr["accuracy_mode"], "threshold_safety_mode": thr["safety_mode"],
              "threshold_early_warning": thr["early_warning"], "mask_shanghai_carbs": mask,
              "early_warning_rule": "probability >= threshold_early_warning while glucose <= 180; re-alert at most once per hour"}

    # 1) final model on every open-data person
    clf = V9._fit(lab, cols)
    joblib.dump({"model": clf, **common, "trained_people": int(lab.patient_id.nunique())}, V9.OUT / "model_final.joblib")
    print(f"[v9] final model: {lab.patient_id.nunique()} people, {len(cols)} features")

    # 2) demo model without the demo patients (same picks as the v7 pack)
    pack = json.load(open(D.PACK_DIR / "patients.json"))
    demo = [p["patient_id"] for p in pack]
    tr = lab[~lab.patient_id.isin(demo)]
    clf = V9._fit(tr, cols)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, **common, "excluded_patients": demo, "trained_people": int(tr.patient_id.nunique())},
                MODEL_DIR / "model_2h.joblib")
    held = lab[lab.patient_id.isin(demo)]
    p = clf.predict_proba(held[cols])[:, 1]
    y = held[V9.LABEL].astype(int)
    v7_2h = joblib.load(D.MODEL_DIR / "model_2h.joblib")
    p7 = v7_2h["model"].predict_proba(V9.base_inputs(held)[V7.FEATURES])[:, 1]
    with open(MODEL_DIR / "demo_evaluation.json", "w") as f:
        json.dump({"demo_patients": demo, "note": "demo patients were excluded from training of this model",
                   "thresholds_from_dev_validation": thr,
                   "evaluation": {"2h": {"accuracy_mode": metrics(y, p, thr["accuracy_mode"]),
                                         "safety_mode": metrics(y, p, thr["safety_mode"]),
                                         "v7_demo_model_same_rows": metrics(y, p7, v7_2h["threshold_accuracy_mode"])}}},
                  f, indent=2)
    print(f"[v9] demo model: {tr.patient_id.nunique()} people; held-out demo accuracy "
          f"{metrics(y, p, thr['accuracy_mode'])['accuracy']:.4f} (v7 demo model on the same rows "
          f"{metrics(y, p7, v7_2h['threshold_accuracy_mode'])['accuracy']:.4f})")

    # 3) extra Shanghai inputs for the demo streams (only the rows already in the pack)
    stream = pd.read_parquet(D.PACK_DIR / "streams.parquet", columns=["patient_id", "segment", "step"])
    sh = pd.concat([E.load_shanghai_t1_meals_grid()[0], E.load_shanghai_t2_meals_grid()[0]], ignore_index=True)
    extra = stream.merge(sh[["segment", "step", "carbs", "oral_med"]], on=["segment", "step"], how="inner")
    extra = extra[extra.carbs.notna() | extra.oral_med.notna()]
    extra.to_parquet(EXTRA_PATH, index=False)
    print(f"[v9] wrote {EXTRA_PATH} ({len(extra)} rows, {extra.patient_id.nunique()} people)")
    pick_examples(m)


if __name__ == "__main__":
    main()
