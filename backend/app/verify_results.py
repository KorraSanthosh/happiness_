"""
Independent re-derivation and consistency checks for every published v7 number.

  python3 -I scripts/run_isolated.py -m:app.verify_results [--update-artifacts]

1. Recomputes the headline metrics directly from the held-out predictions (oof_*.parquet) and checks them
   against models/combined_v7/stage_cv.json (all people, per diabetes type, per dataset, safety mode).
2. Re-scores OhioT1DM with the saved final models and checks stage_final.json.
3. Recomputes the CGM-vs-fingerstick statistics from the raw Shanghai files with a separate implementation.
4. Checks the internal arithmetic of models/investigation/analysis.json (shares, weighted averages, counts).
5. Checks that the dashboard summary, model card and README quote the verified values.
6. v9: split integrity, TEST metrics from the saved TEST predictions, thresholds traced to DEV, a separate
   implementation of the alert policy, the pre-registered promotion rule, and the served v9 model.
--update-artifacts writes the verified error analysis into the model card and the dashboard summary, and
regenerates docs/model_investigation.md from the verified numbers. The models are never changed.
Writes models/investigation/verified_metrics.json; exits with status 1 if any check fails.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from app import model_investigation as MI
from app import real_data_pipeline as P
from app import real_data_v7 as V7
from app import real_data_v9 as V9

ROOT = P.ROOT
INV = ROOT / "models" / "investigation"
TWIN_SUMMARY = ROOT / "backend" / "models" / "twin_v7" / "summary.json"
TOL = 1e-4


class Checks:
    def __init__(self) -> None:
        self.items: List[Dict[str, Any]] = []

    def eq(self, name: str, expected, actual, tol: float = TOL) -> None:
        ok = (expected is None and actual is None) or (
            expected is not None and actual is not None and abs(float(expected) - float(actual)) <= tol)
        self.items.append({"check": name, "expected": expected, "actual": actual, "pass": bool(ok)})

    def true(self, name: str, cond: bool, detail: str = "") -> None:
        self.items.append({"check": name, "pass": bool(cond), "detail": detail})

    @property
    def failed(self) -> List[Dict[str, Any]]:
        return [c for c in self.items if not c["pass"]]


def _load(path) -> Any:
    with open(path) as f:
        return json.load(f)


def _acc(y, p, t) -> float:
    return float(((p >= t).astype(int) == y).mean())


def _rec(y, p, t) -> float:
    return float(((p >= t) & (y == 1)).sum() / max((y == 1).sum(), 1))


def _prec(y, p, t) -> float:
    pred = p >= t
    return float(((y == 1) & pred).sum() / max(pred.sum(), 1))


# ── 1. headline metrics from held-out predictions ─────────────────────────────

def verify_cv(ck: Checks) -> Dict[str, Any]:
    cv = _load(V7.OUT_DIR / "stage_cv.json")["horizons"]
    frame = pd.read_parquet(V7.OUT_DIR / "frame.parquet", columns=["segment", "step", "source", "diabetes_type", "glucose"])
    out: Dict[str, Any] = {}
    for h in V7.HORIZONS:
        o = pd.read_parquet(V7.OUT_DIR / f"oof_{h}.parquet").merge(frame, on=["segment", "step"], how="left")
        ck.true(f"{h}: every held-out row joined to its features", o.source.notna().all())
        ck.true(f"{h}: probabilities in [0,1] and thresholds present", bool(o.p.between(0, 1).all() and o.thr_acc.notna().all()))
        y, p, ta, tb = o.y.to_numpy(), o.p.to_numpy(), o.thr_acc.to_numpy(), o.thr_bal.to_numpy()
        subsets = {"all_open": np.ones(len(o), bool), "type1": (o.diabetes_type == "T1D").to_numpy(),
                   "type2": (o.diabetes_type == "T2D").to_numpy(), **{s: (o.source == s).to_numpy() for s in V7.OPEN}}
        hres = {"rows": int(len(o)), "people": int(o.patient_id.nunique())}
        for name, mk in subsets.items():
            r = {"accuracy": _acc(y[mk], p[mk], ta[mk]), "recall": _rec(y[mk], p[mk], ta[mk]),
                 "precision": _prec(y[mk], p[mk], ta[mk]), "roc_auc": float(roc_auc_score(y[mk], p[mk])),
                 "rule_accuracy": float(((o.glucose[mk] > 180).astype(int) == y[mk]).mean()), "rows": int(mk.sum())}
            ref = cv[h]["accuracy_mode"][name]
            for k in ("accuracy", "recall", "precision", "roc_auc", "rule_accuracy"):
                ck.eq(f"{h} {name} {k} = stage_cv", ref[k], round(r[k], 4), tol=6e-5)
            hres[name] = r
        s = cv[h]["safety_mode"]["all_open"]
        ck.eq(f"{h} safety-mode accuracy = stage_cv", s["accuracy"], round(_acc(y, p, tb), 4), tol=6e-5)
        ck.eq(f"{h} safety-mode recall = stage_cv", s["recall"], round(_rec(y, p, tb), 4), tol=6e-5)
        hres["safety_mode"] = {"accuracy": _acc(y, p, tb), "recall": _rec(y, p, tb)}
        hres["accuracy_ci95"] = cv[h]["bootstrap_accuracy_mode"]["accuracy_ci95"]
        out[h] = hres
    return out


# ── 2. OhioT1DM external validation ───────────────────────────────────────────

def verify_ohio(ck: Checks) -> Dict[str, Any]:
    fin = _load(V7.OUT_DIR / "stage_final.json")["ohio_external_validation"]
    cv = _load(V7.OUT_DIR / "stage_cv.json")["horizons"]
    m = pd.read_parquet(V7.OUT_DIR / "frame.parquet")
    out = {}
    for h, label in V7.HORIZONS.items():
        b = joblib.load(V7.OUT_DIR / f"model_{h}.joblib")
        o = m[m.source.isin(V7.EXTERNAL) & m[label].notna()]
        p = b["model"].predict_proba(o[b["feature_cols"]])[:, 1]
        y = o[label].astype(int).to_numpy()
        t = cv[h]["thresholds"]["accuracy_mode"]
        r = {"accuracy": _acc(y, p, t), "roc_auc": float(roc_auc_score(y, p)), "recall": _rec(y, p, t), "rows": int(len(o)),
             "people": int(o.patient_id.nunique())}
        ck.eq(f"Ohio {h} accuracy = stage_final", fin[h]["accuracy_mode"]["accuracy"], round(r["accuracy"], 4), tol=6e-5)
        ck.eq(f"Ohio {h} ROC-AUC = stage_final", fin[h]["accuracy_mode"]["roc_auc"], round(r["roc_auc"], 4), tol=6e-5)
        ck.true(f"Ohio {h}: no Ohio person in the training folds", not any(pid.startswith("ohio") for pid in V7.assign_folds(m)))
        out[h] = r
    return out


# ── 3. CGM vs fingerstick, independent implementation ─────────────────────────

def verify_fingerstick(ck: Checks, inv: Dict[str, Any]) -> Dict[str, Any]:
    root = glob.glob(str(ROOT / "external_data" / "shanghai" / "extracted" / "*"))
    if not root:
        ck.true("fingerstick: raw Shanghai files present (re-download to verify)", False)
        return {}
    rel, people = [], set()
    for f in glob.glob(f"{root[0]}/Shanghai_*/*"):
        if os.path.basename(f).startswith("~$") or "Summary" in f:
            continue
        x = pd.read_excel(f)
        cols = {re.sub(r"\s+", " ", str(c)).strip().lower(): c for c in x.columns}
        cgm = pd.to_numeric(x[cols[next(k for k in cols if k.startswith("cgm"))]], errors="coerce")
        kb = next((k for k in cols if k.startswith("cbg")), None)
        if kb is None:
            continue
        cbg = pd.to_numeric(x[cols[kb]], errors="coerce")
        pairs = pd.DataFrame({"t": pd.to_datetime(x["Date"], errors="coerce"), "cgm": cgm, "cbg": cbg}).dropna()
        pairs = pairs[~pairs.t.duplicated()]                      # same definition: pairs first, then duplicates,
        pairs = pairs[pairs.cbg.between(40, 600)]                 # then the plausible reference range
        if len(pairs):
            people.add(("T1" if "T1DM" in f else "T2") + os.path.basename(f).split("_")[0])
        rel += list((pairs.cgm - pairs.cbg) / pairs.cbg)
    rel = np.array(rel)
    r = {"pairs": int(len(rel)), "people": len(people), "mard_pct": round(100 * float(np.abs(rel).mean()), 1),
         "median_ard_pct": round(100 * float(np.median(np.abs(rel))), 1), "median_bias_pct": round(100 * float(np.median(rel)), 1)}
    e = inv["E_cgm_vs_fingerstick"]["overall"]
    for k in r:
        ck.eq(f"fingerstick {k} (independent recompute) = analysis.json", e[k], r[k], tol=0.051)
    return r


# ── 4. internal arithmetic of the investigation ───────────────────────────────

def verify_investigation(ck: Checks, inv: Dict[str, Any], cvres: Dict[str, Any]) -> None:
    A = inv["A_label_reliability"]
    for grp, ref in (("all_open", "all_open"), ("type2", "type2"), ("type1", "type1")):
        a = A[grp]
        w = a["uncertain_share_of_rows"] * a["accuracy_uncertain"] + (1 - a["uncertain_share_of_rows"]) * a["accuracy_clear"]
        ck.eq(f"label reliability {grp}: weighted clear/uncertain = overall", a["accuracy_all"], w, tol=2e-4)
        ck.eq(f"label reliability {grp}: overall = held-out accuracy", round(cvres["2h"][ref]["accuracy"], 4), a["accuracy_all"], tol=6e-5)
        ck.eq(f"label reliability {grp}: uncertain share = rows ratio", a["uncertain_share_of_rows"], a["uncertain_rows"] / a["rows"], tol=6e-5)
    ck.eq("label reliability: rows = held-out 2-h rows", cvres["2h"]["rows"], A["all_open"]["rows"], tol=0)
    B = inv["B_missed_moments"]
    ck.eq("moments: miss rate = missed / positives", B["miss_rate"], B["missed"] / B["positives"], tol=6e-5)
    ck.eq("moments: serious miss rate = missed / positives", B["serious_miss_rate"], B["serious_missed"] / B["serious_positives"], tol=6e-5)
    ck.eq("moments: 1 - miss rate = held-out recall", 1 - B["miss_rate"], cvres["2h"]["all_open"]["recall"], tol=6e-5)
    Ep = inv["B_missed_episodes"]
    for k in ("all", "serious", "non_serious"):
        x = Ep[k]
        ck.eq(f"episodes {k}: warned + missed = episodes", x["episodes"], x["warned"] + x["missed"], tol=0)
        ck.eq(f"episodes {k}: miss rate = missed / episodes", x["miss_rate"], x["missed"] / x["episodes"], tol=6e-5)
    ck.eq("episodes: serious + non-serious = all", Ep["all"]["episodes"], Ep["serious"]["episodes"] + Ep["non_serious"]["episodes"], tol=0)
    for h in V7.HORIZONS:
        c = inv["C_calibration"][h]["all"]
        ck.eq(f"calibration {h}: rows = held-out rows", cvres[h]["rows"], c["rows"], tol=0)
        lo, hi = inv["C_calibration"][h]["all"]["ece_ci95_by_person"]
        ck.true(f"calibration {h}: ECE inside its person-bootstrap CI", lo <= c["ece"] <= hi, f"{lo} <= {c['ece']} <= {hi}")
    per = pd.read_csv(INV / "per_patient.csv")
    big = per[per.rows >= 200]
    ck.eq("patients: beats-rule share recomputed from per_patient.csv", inv["D_patients"]["beats_rule_share"],
          round(float((big.gain_vs_rule > 0).mean()), 3), tol=6e-4)
    E = inv["E_cgm_vs_fingerstick"]
    ck.eq("fingerstick: pairs by type add up", E["overall"]["pairs"], sum(v["pairs"] for v in E["by_type"].values()), tol=0)


# ── 5. published artefacts quote the verified values ──────────────────────────

# ── 6. v9 improvement round (frozen DEV/TEST split) ───────────────────────────

V9_DIR = ROOT / "models" / "combined_v9"
TWIN_V9 = ROOT / "backend" / "models" / "twin_v9" / "model_2h.joblib"


def _v9_alerts(d: pd.DataFrame, thr: float, suppress: bool, onsets: pd.DataFrame) -> Dict[str, Any]:
    """Separate implementation of the v9 alert policy (refractory 1 h; warning window 30-120 min before onset)."""
    d = d.sort_values(["segment", "step"])
    active = (d.p.to_numpy() >= thr) & ((d.glucose.to_numpy() <= 180) if suppress else True)
    fired = np.zeros(len(d), bool)
    seg, step = d.segment.to_numpy(), d.step.to_numpy()
    last_seg, last_step = None, -10 ** 9
    for i in range(len(d)):
        if seg[i] != last_seg:
            last_seg, last_step = seg[i], -10 ** 9
        if active[i] and step[i] - last_step >= 4:
            fired[i], last_step = True, step[i]
    lookup = {(a, b): (bool(c), g) for a, b, c, g in zip(seg, step, active, d.glucose.to_numpy())}
    warned, lead, n_eval = 0, [], 0
    for e in onsets.itertuples():
        win = [(k, lookup.get((e.segment, e.onset_step - k))) for k in range(2, 9)]
        win = [(k, v) for k, v in win if v is not None and v[1] <= 180]
        if not win:
            continue
        n_eval += 1
        hits = [k for k, v in win if v[0]]
        if hits:
            warned, lead = warned + 1, lead + [15 * max(hits)]
    days = len(d) / 96
    return {"false_alerts_per_patient_day": float((fired & (d.y.to_numpy() == 0)).sum() / days),
            "serious_episodes": n_eval, "serious_warned": warned, "serious_recall": warned / max(n_eval, 1),
            "serious_lead_min_median": float(np.median(lead)) if lead else None}


def verify_v9(ck: Checks) -> Dict[str, Any]:
    sp, cv = _load(V9_DIR / "split.json"), _load(V9_DIR / "stage_cv.json")
    pol, fin = _load(V9_DIR / "stage_policy.json"), _load(V9_DIR / "stage_final.json")
    dev, test = set(sp["dev_people"]), set(sp["test_people"])
    v7 = pd.read_parquet(V7.OUT_DIR / "oof_2h.parquet", columns=["patient_id"])
    ck.true("v9 split: no person in both DEV and TEST", not dev & test)
    ck.true("v9 split: DEV + TEST = the 187 open-data people", dev | test == set(v7.patient_id) and len(dev | test) == 187)
    ck.true("v9 split: OhioT1DM in neither", not any(p.startswith("ohio") for p in dev | test))
    o = pd.read_parquet(V9_DIR / "oof_dev.parquet")
    ck.true("v9 validation predictions cover DEV people only", set(o.patient_id) == dev)
    sel = fin["selected"]
    for name in ("base (v7 inputs)", sel):
        ck.eq(f"v9 accuracy threshold of {name[:20]} = median of DEV inner-fold thresholds",
              float(np.median(o[f"thr::{name}"])), pol["models"][name]["accuracy_threshold"])
        dv = o[f"p::{name}"].to_numpy()
        ck.eq(f"v9 DEV accuracy of {name[:20]} = stage_cv", round(_acc(o.y.to_numpy(), dv, o[f"thr::{name}"].to_numpy()), 4),
              cv["candidates"][name]["metrics"]["accuracy"], tol=6e-5)
    frame = pd.read_parquet(V9_DIR / "frame.parquet", columns=["segment", "step", "glucose", "diabetes_type"])
    b15 = MI.load_bins()
    eps = V9.episodes(b15)
    serious = eps[eps.serious.to_numpy(bool)]
    out: Dict[str, Any] = {}
    roles = {"baseline (v7 config, DEV-trained)": "baseline", f"candidate: {sel}": "candidate"}
    for mname, role in roles.items():
        t = pd.read_parquet(V9_DIR / f"test_pred_{role}.parquet").merge(frame, on=["segment", "step"], how="left")
        ck.true(f"v9 TEST predictions ({role}) are TEST people only", set(t.patient_id) == test)
        thr = fin["thresholds_from_dev"][mname]
        y, p = t.y.astype(int).to_numpy(), t.p.to_numpy()
        r = {"accuracy": _acc(y, p, thr["accuracy"]), "roc_auc": float(roc_auc_score(y, p)),
             "recall": _rec(y, p, thr["accuracy"]), "precision": _prec(y, p, thr["accuracy"]), "rows": int(len(t))}
        t2 = (t.diabetes_type == "T2D").to_numpy()
        r["type2_accuracy"] = _acc(y[t2], p[t2], thr["accuracy"])
        ref = fin["results"][mname]["test"]
        for k in ("accuracy", "roc_auc", "recall", "precision"):
            ck.eq(f"v9 TEST {role} {k} = stage_final", ref[k], round(r[k], 4), tol=6e-5)
        ck.eq(f"v9 TEST {role} Type 2 accuracy = stage_final", ref["type2"]["accuracy"], round(r["type2_accuracy"], 4), tol=6e-5)
        tt = t[["segment", "step", "p", "y", "glucose"]]
        on = serious[serious.segment.isin(set(t.segment))]
        for pname, th, sup in (("alerts_v7_policy", thr["accuracy"], False), ("alerts_chosen_policy", thr["policy"], True)):
            a, refa = _v9_alerts(tt, th, sup, on), ref[pname]
            for k in ("serious_episodes", "serious_warned"):
                ck.eq(f"v9 TEST {role} {pname} {k} (independent)", refa[k], a[k], tol=0)
            ck.eq(f"v9 TEST {role} {pname} false alerts/day (independent)", refa["false_alerts_per_patient_day"],
                  round(a["false_alerts_per_patient_day"], 3), tol=1.5e-3)
            ck.eq(f"v9 TEST {role} {pname} median lead (independent)", refa["serious_lead_min_median"], a["serious_lead_min_median"], tol=0)
            r[pname] = a
        out[role] = r
    vs = fin["test_candidate_vs_baseline"]
    rb, rc = fin["results"]["baseline (v7 config, DEV-trained)"], fin["results"][f"candidate: {sel}"]
    ch = fin["chronological_holdout"]
    rule = {"1 AUC gain > 0 and CI low > -0.002": vs["roc_auc_diff_mean"] > 0 and vs["roc_auc_diff_ci95"][0] > -0.002,
            "2 accuracy CI low >= -0.005": vs["accuracy_diff_ci95"][0] >= -0.005,
            "3 Ohio accuracy and AUC not down > 0.002": rc["ohio_external"]["accuracy"] - rb["ohio_external"]["accuracy"] >= -0.002
                                                      and rc["ohio_external"]["roc_auc"] - rb["ohio_external"]["roc_auc"] >= -0.002,
            "4 chronological accuracy not down": ch[f"candidate: {sel}"]["accuracy"] >= ch["baseline (v7 config, DEV-trained)"]["accuracy"],
            "policy: more serious warned, no more false alerts": (
                rb["test"]["alerts_chosen_policy"]["serious_warned"] > rb["test"]["alerts_v7_policy"]["serious_warned"]
                and rb["test"]["alerts_chosen_policy"]["false_alerts_per_patient_day"] <= rb["test"]["alerts_v7_policy"]["false_alerts_per_patient_day"])}
    for k, v in rule.items():
        ck.true(f"v9 promotion rule {k}", v)
    out["promotion_rule"] = rule
    rep = _load(V9_DIR / "stage_report.json")["arms"]
    for role in ("baseline", "candidate"):
        for pol_name, key in (("v7 policy", "alerts_v7_policy"), ("v9 policy", "alerts_chosen_policy")):
            a = rep[f"{role} + {pol_name}"]
            ck.eq(f"v9 report {role} + {pol_name}: warned episodes = stage_final", out[role][key]["serious_warned"], a["serious_warned"], tol=0)
            ck.eq(f"v9 report {role} + {pol_name}: false alerts/day = stage_final", round(out[role][key]["false_alerts_per_patient_day"], 3),
                  a["false_alerts_per_patient_day"], tol=1.5e-3)
    if TWIN_V9.exists():
        b = joblib.load(TWIN_V9)
        ck.eq("served v9 2 h accuracy threshold = DEV", fin["thresholds_from_dev"][f"candidate: {sel}"]["accuracy"], b["threshold_accuracy_mode"])
        ck.eq("served v9 early-warning threshold = DEV policy", fin["thresholds_from_dev"][f"candidate: {sel}"]["policy"], b["threshold_early_warning"])
        ck.true("served v9 features = the selected feature set", b["feature_cols"] == V9.selected(cv)[1])
        pack = [p["patient_id"] for p in _load(ROOT / "backend" / "data" / "twin_v7" / "patients.json")]
        ck.true("served v9 demo model excluded exactly the demo patients", set(b["excluded_patients"]) == set(pack))
    return out


def verify_published_v9(ck: Checks, v9: Dict[str, Any]) -> None:
    readme = (ROOT / "README.md").read_text()
    fin = _load(V9_DIR / "stage_final.json")
    rb, rc = (fin["results"][k] for k in fin["results"])
    pc2 = lambda x: f"{100 * x:.2f} %"
    for label, val in (("TEST accuracy v7 config", v9["baseline"]["accuracy"]), ("TEST accuracy v9", v9["candidate"]["accuracy"]),
                       ("Ohio accuracy v7 config", rb["ohio_external"]["accuracy"]), ("Ohio accuracy v9", rc["ohio_external"]["accuracy"])):
        ck.true(f"README v9 {label} ({pc2(val)}) quoted", pc2(val) in readme)
    for label, val in (("TEST AUC v7 config", v9["baseline"]["roc_auc"]), ("TEST AUC v9", v9["candidate"]["roc_auc"])):
        ck.true(f"README v9 {label} ({val:.4f}) quoted", f"{val:.4f}" in readme)
    for role in ("baseline", "candidate"):
        for key in ("alerts_v7_policy", "alerts_chosen_policy"):
            a = v9[role][key]
            if role == "candidate" and key == "alerts_v7_policy":
                continue
            ck.true(f"README v9 {role} {key} warned count ({a['serious_warned']:,}) quoted", f"{a['serious_warned']:,} (" in readme)
            ck.true(f"README v9 {role} {key} false alerts/day ({a['false_alerts_per_patient_day']:.2f}) quoted",
                    f"{a['false_alerts_per_patient_day']:.2f}" in readme)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def verify_published(ck: Checks, cvres: Dict[str, Any], ohio: Dict[str, Any]) -> None:
    summ = _load(TWIN_SUMMARY)
    for h in V7.HORIZONS:
        for name in ("all_open", "type2"):
            ck.eq(f"dashboard summary {h} {name} accuracy", round(cvres[h][name]["accuracy"], 4),
                  summ["horizons"][h]["accuracy_mode"][name]["accuracy"], tol=6e-5)
        ck.eq(f"dashboard summary {h} Ohio accuracy", round(ohio[h]["accuracy"], 4), summ["horizons"][h]["ohio_external"]["accuracy"], tol=6e-5)
    card = _load(V7.OUT_DIR / "model_card.json")
    for h in V7.HORIZONS:
        ck.eq(f"model card {h} accuracy", round(cvres[h]["all_open"]["accuracy"], 4), card["performance"][h]["accuracy_mode"]["accuracy"], tol=6e-5)
    readme = (ROOT / "README.md").read_text()
    for h in V7.HORIZONS:
        r = cvres[h]
        for label, val in ((f"README {h} accuracy", r["all_open"]["accuracy"]), (f"README {h} Type 2 accuracy", r["type2"]["accuracy"]),
                           (f"README {h} Ohio accuracy", ohio[h]["accuracy"]), (f"README {h} recall", r["all_open"]["recall"])):
            ck.true(label + f" ({_pct(val)} %) quoted", f"{_pct(val)} %" in readme or f"{_pct(val)}%" in readme)
        for label, val in ((f"README {h} precision", r["all_open"]["precision"]), (f"README {h} safety recall", r["safety_mode"]["recall"])):
            ck.true(label + f" ({_pct(val)} %) quoted", f"{_pct(val)} %" in readme)
        auc = f"{r['all_open']['roc_auc']:.3f}"                 # from full precision: no double rounding
        ck.true(f"README {h} ROC-AUC ({auc}) quoted", f"| {auc} |" in readme)
        oauc = f"AUC {ohio[h]['roc_auc']:.3f}"
        ck.true(f"README {h} Ohio {oauc} quoted", oauc in readme)
    inv = _load(INV / "analysis.json")
    for label, val in (("README errors near 180", inv["A_label_reliability"]["all_open"]["uncertain_share_of_errors"]),
                       ("README accuracy on clear outcomes", inv["A_label_reliability"]["all_open"]["accuracy_clear"]),
                       ("README serious-episode miss rate", inv["B_missed_episodes"]["serious"]["miss_rate"])):
        ck.true(label + f" ({_pct(val)} %) quoted", f"{_pct(val)} %" in readme)
    ck.true("README fingerstick MARD quoted", f"MARD {inv['E_cgm_vs_fingerstick']['overall']['mard_pct']} %" in readme)
    ck.true("README no longer quotes the unverifiable '65 %' meal claim", "65 % of spikes" not in readme)
    # qualified v6 claims are backed by the saved v6 results
    v6 = ROOT / "models" / "combined_v6"
    meals = _load(v6 / "stage_meals.json")["paired_vs_v6"]["+ meals announced 2 h ahead (what-if)"]["accuracy_diff_mean"]
    ck.true("v6 claim: future meals add about +0.2 points", 0.0015 <= meals <= 0.0035, f"{meals}")
    curve = {str(r["train_patients"]): r["test_acc_all"] for r in _load(v6 / "stage_learning.json")["curve"]}
    ck.true("v6 claim: learning curve flat after ~20 patients", abs(curve["20"] - curve["all (~75)"]) < 0.005, f"{curve}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-artifacts", action="store_true")
    a = ap.parse_args()
    ck = Checks()
    inv = _load(INV / "analysis.json")
    cvres = verify_cv(ck)
    ohio = verify_ohio(ck)
    fs = verify_fingerstick(ck, inv)
    verify_investigation(ck, inv, cvres)
    v9 = verify_v9(ck) if (V9_DIR / "stage_final.json").exists() else None
    if a.update_artifacts and not ck.failed:
        update_artifacts(inv, cvres, v9)
    verify_published(ck, cvres, ohio)
    if v9:
        verify_published_v9(ck, v9)
    res = {"checks_total": len(ck.items), "checks_failed": len(ck.failed), "failed": ck.failed,
           "verified": {"cv": cvres, "ohio": ohio, "fingerstick": fs, "v9": v9}, "checks": ck.items}
    with open(INV / "verified_metrics.json", "w") as f:
        json.dump(res, f, indent=2, default=str)       # verified values are finite; NaN would be a bug
    print(f"[verify] {len(ck.items)} checks, {len(ck.failed)} failed -> {INV / 'verified_metrics.json'}")
    for c in ck.failed:
        print("  FAIL", c)
    sys.exit(1 if ck.failed else 0)


# ── writing verified results into the published artefacts ─────────────────────

def error_analysis_block(inv: Dict[str, Any]) -> Dict[str, Any]:
    A, B, Ep, C, E = (inv["A_label_reliability"], inv["B_missed_moments"], inv["B_missed_episodes"],
                      inv["C_calibration"], inv["E_cgm_vs_fingerstick"])
    return {
        "source": "models/investigation/analysis.json (verified by app/verify_results.py)",
        "protocol": inv["protocol"],
        "label_reliability_2h": {k: A[k] for k in ("band_mg_dl", "assumption", "stratified_by", "all_open", "type2")},
        "serious_spikes_2h": {"definition": Ep["serious_definition"],
                              "episode_level": {"rule": Ep["definition"], **Ep["serious"],
                                                "miss_rate_ci95_by_person": Ep["serious_miss_rate_ci95_by_person"]},
                              "moment_level": {k: B[k] for k in ("serious_positives", "serious_missed", "serious_miss_rate")}},
        "calibration": {h: {**{k: C[h]["all"][k] for k in ("rows", "ece", "ece_ci95_by_person", "slope", "intercept")},
                            "per_person": C[h]["per_person"]} for h in V7.HORIZONS} | {"method": C["method"], "limitations": C["limitations"]},
        "cgm_vs_fingerstick": {k: E[k] for k in ("source", "pairing", "method", "overall", "median_ard_ci95_pct",
                                                  "median_bias_ci95_pct", "by_type", "caveats")},
    }


def v9_block() -> Dict[str, Any]:
    """Verified v9 summary for the model cards and the dashboard (from the v9 result files)."""
    cv, pol = _load(V9_DIR / "stage_cv.json"), _load(V9_DIR / "stage_policy.json")
    fin, lab, rep = _load(V9_DIR / "stage_final.json"), _load(V9_DIR / "stage_labels.json"), _load(V9_DIR / "stage_report.json")
    keys = ("accuracy", "roc_auc", "pr_auc", "recall", "precision", "specificity", "f1", "brier", "ece")
    names = {"baseline (v7 config, DEV-trained)": "v7_config", f"candidate: {fin['selected']}": "v9"}
    return {
        "source": "models/combined_v9 (verified by app/verify_results.py)",
        "protocol": "187 open-data people split once into DEV 150 / TEST 37 (seed 2026). Feature groups, thresholds and "
                    "the alert policy were chosen on DEV only (5-fold person CV); TEST, OhioT1DM and a chronological "
                    "holdout were scored once. Label unchanged (max CGM in next 2 h > 180).",
        "selected": fin["selected"], "served_2h_model": "v9 (30 min / 1 h stay v7)",
        "dev_cv": {k: {"accuracy": v["metrics"]["accuracy"], "roc_auc": v["metrics"]["roc_auc"], "retained": v.get("retained"),
                       "accuracy_diff_ci95": v.get("vs_base", {}).get("accuracy_diff_ci95"),
                       "roc_auc_diff_ci95": v.get("vs_base", {}).get("roc_auc_diff_ci95")} for k, v in cv["candidates"].items()},
        "test": {names[k]: {**{m: r["test"][m] for m in keys}, "accuracy_ci95_by_person": r["test"]["accuracy_ci95_by_person"],
                            "type2_accuracy": r["test"]["type2"]["accuracy"], "ohio_accuracy": r["ohio_external"]["accuracy"],
                            "ohio_roc_auc": r["ohio_external"]["roc_auc"]} for k, r in fin["results"].items()},
        "test_v9_vs_v7_config": fin["test_candidate_vs_baseline"],
        "chronological_holdout": {names[k]: {m: v[m] for m in ("accuracy", "roc_auc")} for k, v in fin["chronological_holdout"].items() if k in names},
        "test_people": fin["test_people"], "test_rows": fin["test_rows"],
        "alert_policy": {"rule": pol["rule"], "thresholds": fin["thresholds_from_dev"], **rep,
                         "alerts_per_patient_day": {f"{'baseline' if names[k] == 'v7_config' else 'candidate'} + {pn}": r["test"][key]["alerts_per_patient_day"]
                                                    for k, r in fin["results"].items()
                                                    for pn, key in (("v7 policy", "alerts_v7_policy"), ("v9 policy", "alerts_chosen_policy"))}},
        "label_sensitivity": lab["alternative_labels"], "uncertainty_bands": lab["uncertainty_bands"],
    }


def update_artifacts(inv: Dict[str, Any], cvres: Dict[str, Any], v9: Any = None) -> None:
    block = error_analysis_block(inv)
    v9b = v9_block() if v9 else None
    card_path = V7.OUT_DIR / "model_card.json"
    card = _load(card_path)
    card["verified_error_analysis"] = block
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2, default=str)
    summ = _load(TWIN_SUMMARY)
    summ["error_analysis"] = block
    if v9b:
        summ["v9"] = v9b
        with open(V9_DIR / "model_card.json", "w") as f:
            json.dump({"model": "GlucoTwin v9 2-h spike model", "version": "v9.0-open-data", "intended_use":
                       "Research prototype - not for clinical use.", "verified": v9b}, f, indent=2, default=str)
    with open(TWIN_SUMMARY, "w") as f:
        json.dump(summ, f, indent=2, default=str)
    (ROOT / "docs" / "model_investigation.md").write_text(render_report(inv, cvres) + (render_v9(v9b) if v9b else ""))
    print("[verify] updated model cards, dashboard summary.json, docs/model_investigation.md")


def render_v9(b: Dict[str, Any]) -> str:
    pc2 = lambda x: f"{100 * x:.2f} %"
    ci = lambda c, f=lambda x: f"{100 * x:+.2f}": f"{f(c[0])} to {f(c[1])}"
    dev = "\n".join(f"| {k} | {pc2(v['accuracy'])} | {v['roc_auc']:.4f} | "
                     f"{'—' if not v['accuracy_diff_ci95'] else ci(v['accuracy_diff_ci95'])} | "
                     f"{'—' if v['retained'] is None else ('kept' if v['retained'] else 'rejected')} |" for k, v in b["dev_cv"].items())
    t7, t9 = b["test"]["v7_config"], b["test"]["v9"]
    test = "\n".join(f"| {m} | {t7[m]:.4f} | {t9[m]:.4f} |" for m in ("accuracy", "roc_auc", "pr_auc", "recall", "precision",
                                                                    "specificity", "f1", "brier", "ece", "type2_accuracy",
                                                                    "ohio_accuracy", "ohio_roc_auc"))
    vs = b["test_v9_vs_v7_config"]
    arms = b["alert_policy"]["arms"]
    alerts = "\n".join(f"| {k} | {a['serious_warned']:,} of {a['serious_episodes']:,} ({pc2(a['serious_recall'])}, CI {pc2(a['serious_recall_ci95'][0])}–{pc2(a['serious_recall_ci95'][1])}) | "
                        f"{a['false_alerts_per_patient_day']:.3f} (CI {a['false_alerts_per_patient_day_ci95'][0]:.2f}–{a['false_alerts_per_patient_day_ci95'][1]:.2f}) |"
                        for k, a in arms.items())
    paired = "\n".join(f"| {k} | {100 * v['serious_recall_diff']:+.2f} ({ci(v['serious_recall_diff_ci95'])}) | "
                        f"{v['false_alerts_per_day_diff']:+.3f} ({v['false_alerts_per_day_diff_ci95'][0]:+.3f} to {v['false_alerts_per_day_diff_ci95'][1]:+.3f}) |"
                        for k, v in b["alert_policy"]["paired_vs_baseline_v7_policy"].items())
    labels = "\n".join(f"| {k} | {v['prevalence']:.3f} | {pc2(v['accuracy'])} | {pc2(v['recall'])} | {v['roc_auc']:.4f} |"
                        for k, v in b["label_sensitivity"].items())
    return f"""
## 7. v9 improvement round (locked TEST set)

{b['protocol']}

### Feature groups on DEV (validation)

| Candidate | Accuracy | ROC-AUC | Accuracy gain vs base, points (95 % person CI) | Rule |
|---|---|---|---|---|
{dev}

### Single TEST evaluation ({b['test_people']} people, {b['test_rows']:,} moments)

| Metric | v7 config (DEV-trained) | v9 |
|---|---|---|
{test}

Paired person bootstrap, v9 minus v7 config: accuracy {ci(vs['accuracy_diff_ci95'])} points, ROC-AUC {vs['roc_auc_diff_mean']:+.4f} ({vs['roc_auc_diff_ci95'][0]:+.4f} to {vs['roc_auc_diff_ci95'][1]:+.4f}).

### Alert policy on TEST (serious episodes warned 30–120 min ahead)

Rule chosen on DEV: {b['alert_policy']['rule']}.

| Arm | Serious episodes warned | False alerts per patient-day |
|---|---|---|
{alerts}

| Paired vs v7 model + v7 policy | Serious recall, points (95 % CI) | False alerts/day (95 % CI) |
|---|---|---|
{paired}

### Label sensitivity (same TEST predictions, v7 config)

| Label | Prevalence | Accuracy | Recall | ROC-AUC |
|---|---|---|---|---|
{labels}
"""


def render_report(inv: Dict[str, Any], cvres: Dict[str, Any]) -> str:
    A, B, Ep, C, D, E = (inv["A_label_reliability"], inv["B_missed_moments"], inv["B_missed_episodes"],
                         inv["C_calibration"], inv["D_patients"], inv["E_cgm_vs_fingerstick"])
    pc = lambda x: f"{100 * x:.1f} %"
    pc2 = lambda x: f"{100 * x:.2f} %"
    a, t2 = A["all_open"], A["type2"]
    s, sm = Ep["serious"], B
    eo, ci_a, ci_b = E["overall"], E["median_ard_ci95_pct"], E["median_bias_ci95_pct"]
    t1e, t2e = E["by_type"]["T1DM"], E["by_type"]["T2DM"]
    cal_rows = "\n".join(
        f"| {h} | {C[h]['all']['rows']:,} | {pc2(C[h]['all']['ece'])} ({pc2(C[h]['all']['ece_ci95_by_person'][0])} to {pc2(C[h]['all']['ece_ci95_by_person'][1])}) "
        f"| {C[h]['all']['slope']:.3f} | {C[h]['per_person']['people']} | {pc(C[h]['per_person']['median_abs_gap'])} | {pc(C[h]['per_person']['share_over_5pts'])} |"
        for h in V7.HORIZONS)
    return f"""# Where does the remaining error come from? — Verified investigation of the GlucoTwin v7 model

> Generated by `backend/app/verify_results.py --update-artifacts` from `models/investigation/analysis.json`
> after all automated consistency checks passed (`models/investigation/verified_metrics.json`). Do not edit numbers by hand.

## Evaluation protocol (applies to every figure unless stated)

- **Predictions:** v7 held-out predictions (`models/combined_v7/oof_*.parquet`), 7-fold **person-grouped** nested
  cross-validation over **{inv['protocol']['people']} people**. Every prediction comes from a model that never saw that person.
- **Unit:** one 15-minute *prediction moment* with a real CGM reading and a known label. The 2-hour horizon has **{inv['protocol']['rows_2h']:,}** moments.
- **Label:** 1 if the maximum CGM reading in the next 2 h is > 180 mg/dL. Re-deriving every label from the raw glucose reproduced all of them exactly.
- **Decision threshold:** accuracy mode, chosen per fold on inner folds only (median 0.51).
- **No model was changed for this report.** No new algorithm was tried.

## 1. Accuracy and label reliability (2-hour horizon)

| Population | Moments | Accuracy | Moments near 180* | Accuracy, clear outcome | Accuracy, outcome near 180 | Share of errors near 180 |
|---|---|---|---|---|---|---|
| All {a['people']} people | {a['rows']:,} | **{pc(a['accuracy_all'])}** | {pc(a['uncertain_share_of_rows'])} | {pc(a['accuracy_clear'])} | {pc(a['accuracy_uncertain'])} | {pc(a['uncertain_share_of_errors'])} |
| Type 2 ({t2['people']} people) | {t2['rows']:,} | **{pc(t2['accuracy_all'])}** | {pc(t2['uncertain_share_of_rows'])} | {pc(t2['accuracy_clear'])} | {pc(t2['accuracy_uncertain'])} | {pc(t2['uncertain_share_of_errors'])} |

\\* "Near 180" means the **actual** 2-h peak was {A['band_mg_dl'][0]}–{A['band_mg_dl'][1]} mg/dL ({A['assumption']}). The split
is made using the true outcome, so it **decomposes the error**; it is not a subset the model could pick in advance.
Overall accuracy is the weighted average of the two columns, for example {pc(a['uncertain_share_of_rows'])} × {pc(a['accuracy_uncertain'])} + {pc(1 - a['uncertain_share_of_rows'])} × {pc(a['accuracy_clear'])} = {pc(a['accuracy_all'])}.

## 2. How noisy is the reference itself? CGM vs fingerstick (Shanghai only)

- **Data:** {E['source']}. **{eo['pairs']:,} pairs from {eo['people']} people** (median {E['pairs_per_person_median']} pairs per person).
- **Pairing:** {E['pairing']}
- **Method:** {E['method']}.

| | Pairs | People | MARD | Median abs. difference (95 % CI) | Median bias (95 % CI) | Within ±15 % |
|---|---|---|---|---|---|---|
| All | {eo['pairs']:,} | {eo['people']} | {eo['mard_pct']} % | {eo['median_ard_pct']} % ({ci_a[0]} to {ci_a[1]}) | {eo['median_bias_pct']:+} % ({ci_b[0]} to {ci_b[1]}) | {pc(eo['within_15pct'])} |
| Type 2 | {t2e['pairs']:,} | {t2e['people']} | {t2e['mard_pct']} % | {t2e['median_ard_pct']} % | {t2e['median_bias_pct']:+} % | {pc(t2e['within_15pct'])} |
| Type 1 | {t1e['pairs']:,} | {t1e['people']} | {t1e['mard_pct']} % | {t1e['median_ard_pct']} % | {t1e['median_bias_pct']:+} % | {pc(t1e['within_15pct'])} |

A negative bias means the CGM reads lower than the fingerstick. **Caveats:** {' '.join(E['caveats'])} The ±10 % band in section 1
is a typical-CGM assumption; it is not derived from these pairs.

## 3. Missed spikes (2-hour horizon)

**Serious spike:** {Ep['serious_definition']}.

| Counting unit | Serious spikes | Missed | Miss rate |
|---|---|---|---|
| Distinct episodes, warning required 30–120 min before onset (clinically meaningful) | {s['episodes']:,} | {s['missed']:,} | **{pc(s['miss_rate'])}** (95 % CI {pc(Ep['serious_miss_rate_ci95_by_person'][0])} to {pc(Ep['serious_miss_rate_ci95_by_person'][1])}, by person) |
| Prediction moments (overlapping windows; one spike counted many times) | {sm['serious_positives']:,} | {sm['serious_missed']:,} | {pc(sm['serious_miss_rate'])} |

- **Episode rule:** {Ep['definition']} {Ep['episodes_found'] - Ep['episodes_evaluable']} of {Ep['episodes_found']:,} episodes had no held-out prediction in the warning window and are excluded.
- **All spike episodes:** {pc(Ep['all']['miss_rate'])} missed ({Ep['all']['missed']:,} of {Ep['all']['episodes']:,}). For warned serious episodes, the latest warning came a median of {Ep['serious_warned_lead_min_median']:.0f} min before onset.
- **What the misses look like (moment level):** glucose at prediction time had a median of {B['false_negatives']['glucose_now_median']} mg/dL for misses vs {B['true_positives']['glucose_now_median']} for spikes caught.
  {pc(B['false_negatives']['spike_starts_after_60min'])} of misses started more than 60 min later (vs {pc(B['true_positives']['spike_starts_after_60min'])}).
  Where meals are logged, {pc(B['false_negatives']['meal_logged_in_next_2h'])} of missed moments had a meal logged **after** the prediction (vs {pc(B['true_positives']['meal_logged_in_next_2h'])}).
- **Limitation:** {Ep['note']}

## 4. Calibration (held-out predictions)

Method: {C['method']}.

| Horizon | Moments | ECE (95 % CI) | Calibration slope | People with ≥ 200 moments | Median per-person gap | People off by > 5 points |
|---|---|---|---|---|---|---|
{cal_rows}

**Limitations:** {' '.join(C['limitations'])}

## 5. Patient-level generalisation (2-hour horizon, people with ≥ 200 moments)

- **Beats the "already above 180" rule:** {pc(D['beats_rule_share'])} of {D['people_with_min_rows']} people.
- **Per-person accuracy** (5th / 50th / 95th percentile): {' / '.join(pc(x) for x in D['accuracy_p05_p50_p95'])}.
- **Per-person AUC** (same percentiles): {' / '.join(f'{x:.3f}' for x in D['auc_p05_p50_p95'])}.
- **Within each dataset**, people with more frequent spikes have lower accuracy but not lower AUC (Spearman correlation of AUC with spike prevalence is positive in all six datasets, significant in three: `within_dataset_spearman` in `analysis.json`). Their outcomes are closer to 50/50 rather than ranked worse.

## 6. Conclusions

1. **Calibration** is good on held-out people (ECE ≤ {pc2(max(C[h]['all']['ece'] for h in V7.HORIZONS))}).
2. **About half of the 2-h errors** ({pc(a['uncertain_share_of_errors'])}) occur where the actual peak was within ±10 % of 180 mg/dL.
   Where a reference exists, CGM and fingerstick differ by a median of {eo['median_ard_pct']} % (MARD {eo['mard_pct']} %).
3. **Distinct serious episodes:** {pc(s['miss_rate'])} get no warning 30–120 min ahead. Missed spikes typically start from normal glucose and later in the window,
   often after a meal logged *after* the prediction time.
4. **Next steps** are new information known at prediction time: meal announcements, reference checks, and an Indian cohort. A different algorithm is not the next step.
"""


if __name__ == "__main__":
    main()
