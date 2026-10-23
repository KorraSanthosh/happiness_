"""
v9 - data-driven improvement round, evaluated against the FROZEN v7 baseline (models/baseline_v7/manifest.json).

Protocol (fixed before any result was seen):
  * split   20 % of the 187 open-data people are locked as the FINAL TEST set (seed 2026, stratified by
            dataset); the other 80 % are DEV. The split is saved once and never overwritten.
  * select  every choice (feature groups, thresholds, alert policy) uses only DEV, through 5-fold
            person-grouped CV with thresholds chosen on inner folds (= "validation").
  * test    the chosen candidate and the baseline (same v7 config, retrained on DEV) are evaluated ONCE on the
            TEST people, on OhioT1DM (external) and on a chronological holdout.
  * label   unchanged: max CGM in the next 2 h > 180 mg/dL.  Serious event: unchanged (peak >= 250 or >= 3
            consecutive 15-min bins > 180).
Candidate feature groups (hypotheses from models/combined_v9/error_analysis.json):
  C1 longer personal history   C2 recorded meals + oral drugs (Shanghai)   C3 multi-window dynamics
Retention rule (pre-registered): keep a group only if, on DEV validation, ROC-AUC improves with a 95 %
person-bootstrap CI excluding 0 AND accuracy is not significantly worse.
Promotion rule (written 2026-10-23 after the DEV results, before any TEST result existed): the candidate
replaces v7 in the app only if, on TEST, (1) the ROC-AUC difference vs the DEV-trained baseline is > 0 with a
95 % person-CI lower bound > -0.002, (2) the accuracy-difference CI lower bound is >= -0.005, (3) OhioT1DM
accuracy and AUC do not drop by more than 0.002, and (4) chronological-holdout accuracy does not drop.
The suppressed alert policy is adopted only if, on TEST, it warns of more serious episodes than the v7 policy
with no more false alerts per patient-day. Otherwise v7 stays and the findings are documented.

python3 -I scripts/run_isolated.py -m:app.real_data_v9 --stage {build,split,cv,policy,final,labels,report,all}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from app import external_loaders as E
from app import model_investigation as MI
from app import real_data_pipeline as P
from app import real_data_v7 as V7
from app.real_data_eval import best_threshold, cluster_bootstrap, ece, metrics

warnings.filterwarnings("ignore")

OUT = P.ROOT / "models" / "combined_v9"
SPLIT_SEED = 2026
TEST_SHARE = 0.20
N_FOLDS = 5
LABEL = "y"
REFRACTORY_BINS = 4                       # an alert re-fires at most once per hour
CARB_FEATURES = ["carbs_now", "carbs_30m", "carbs_1h", "carbs_2h", "carbs_4h", "cob", "since_meal", "last_meal_size", "cob_per_iob"]
CANDIDATES: Dict[str, List[str]] = {
    "base (v7 inputs)": V7.FEATURES,
    "+C1 personal history": V7.FEATURES + P.HISTORY_FEATURES,
    "+C2 recorded meals & oral drugs": V7.FEATURES + P.ORAL_MED_FEATURES,
    "+C3 multi-window dynamics": V7.FEATURES + P.DYNAMICS_FEATURES,
}


def _save(name: str, obj: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.json", "w") as f:
        json.dump(MI._no_nan(obj), f, indent=2, default=str, allow_nan=False)
    print(f"[save] {OUT / (name + '.json')}", flush=True)


def _load(name: str) -> Any:
    with open(OUT / f"{name}.json") as f:
        return json.load(f)


# ── build ─────────────────────────────────────────────────────────────────────

def stage_build() -> pd.DataFrame:
    grids = [P.load_brist_grid("train")[0],
             E.load_external_grids(("hupa", "uom", "azt1d"))[0],
             E.load_shanghai_t1_meals_grid()[0], E.load_shanghai_t2_meals_grid()[0], P.load_ohio_grid()]
    grid = pd.concat(grids, ignore_index=True)
    m = P.modelling_frame_multi(P.build_features(grid))
    obs = grid.groupby("patient_id")["glucose"].apply(lambda s: s.notna().mean())
    m["cgm_native"] = m["patient_id"].map(lambda p: "15min" if obs[p] < 0.6 else "5min")
    m["recording"] = E.recording_of_segment(m["segment"])
    m = m.merge(E.shanghai_ehr(), on="recording", how="left")
    m.loc[~m.source.isin(("sh_t1", "sh_t2")), "ehr_t2d"] = 0.0
    m["diabetes_type"] = np.where(m["ehr_t2d"] == 1, "T2D", "T1D")
    # identity with the frozen v7 frame: same rows, same labels, same v7 inputs (Shanghai carbs excepted)
    v7 = pd.read_parquet(V7.OUT_DIR / "frame.parquet")
    j = v7.merge(m, on=["segment", "step"], how="outer", suffixes=("_v7", "_v9"), indicator=True)
    sh = j.source_v7.isin(("sh_t1", "sh_t2"))
    report = {"rows_v7": int(len(v7)), "rows_v9": int(len(m)), "rows_in_both": int((j._merge == "both").sum()),
              "labels_identical": bool(all(np.allclose(j[f"{c}_v7"].astype(float), j[f"{c}_v9"].astype(float), equal_nan=True)
                                           for c in ("y", "y_30m", "y_60m")))}
    diff = {}
    for c in V7.FEATURES:
        rows = ~sh if c in CARB_FEATURES else np.ones(len(j), bool)
        same = np.allclose(j.loc[rows, f"{c}_v7"].astype(float), j.loc[rows, f"{c}_v9"].astype(float), equal_nan=True)
        if not same:
            diff[c] = int((~np.isclose(j.loc[rows, f"{c}_v7"].astype(float), j.loc[rows, f"{c}_v9"].astype(float), equal_nan=True)).sum())
    report["v7_inputs_identical_except_shanghai_carbs"] = not diff
    report["differences"] = diff
    assert report["rows_in_both"] == len(v7) == len(m) and report["labels_identical"] and not diff, report
    OUT.mkdir(parents=True, exist_ok=True)
    m.to_parquet(OUT / "frame.parquet")
    _save("build_identity", report)
    return m


def load_frame() -> pd.DataFrame:
    return pd.read_parquet(OUT / "frame.parquet")


def base_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """v7 saw no Shanghai meal carbs: mask them so the baseline gets exactly v7's inputs."""
    d = df.copy()
    d.loc[d.source.isin(("sh_t1", "sh_t2")), CARB_FEATURES] = np.nan
    return d


# ── split ─────────────────────────────────────────────────────────────────────

def stage_split(m: pd.DataFrame, force: bool = False) -> Dict[str, Any]:
    path = OUT / "split.json"
    if path.exists() and not force:
        print("[split] frozen split exists - not overwriting", flush=True)
        return _load("split")
    rng = np.random.default_rng(SPLIT_SEED)
    test = []
    for src in V7.OPEN:
        ps = sorted(m.loc[m.source == src, "patient_id"].unique())
        rng.shuffle(ps)
        test += list(ps[: max(1, int(round(TEST_SHARE * len(ps))))])
    open_people = sorted(m.loc[m.source.isin(V7.OPEN), "patient_id"].unique())
    dev = [p for p in open_people if p not in set(test)]
    # leakage checks
    keys_dup = int(m.duplicated(["segment", "step"]).sum())
    seg_hash = m[m.source.isin(V7.OPEN)].groupby("segment").glucose_raw.apply(
        lambda s: hashlib.md5(np.round(s.to_numpy(float), 1).tobytes()).hexdigest())
    seg_owner = m.drop_duplicates("segment").set_index("segment").patient_id
    dup_groups = seg_hash.groupby(seg_hash).apply(lambda s: sorted({seg_owner[x] for x in s.index}))
    cross_person_dups = [g for g in dup_groups if len(g) > 1]
    split = {"seed": SPLIT_SEED, "test_share": TEST_SHARE, "dev_people": dev, "test_people": sorted(test),
             "n_dev": len(dev), "n_test": len(test),
             "test_by_source": pd.Series([p.rsplit("_", 1)[0] for p in test]).value_counts().to_dict(),
             "checks": {"no_person_in_both": not set(dev) & set(test), "ohio_excluded": not any(p.startswith("ohio") for p in dev + test),
                        "duplicate_segment_step_rows": keys_dup, "identical_glucose_segments_across_people": cross_person_dups}}
    assert split["checks"]["no_person_in_both"] and split["checks"]["ohio_excluded"] and keys_dup == 0, split["checks"]
    _save("split", split)
    return split


# ── evaluation helpers ────────────────────────────────────────────────────────

def episodes(b15: pd.DataFrame) -> pd.DataFrame:
    """Distinct > 180 episodes (as in model_investigation), with the serious flag."""
    eps = []
    for seg, s in b15.groupby("segment", sort=False):
        s = s.sort_values("step")
        gm, steps = s["glucose_max"].to_numpy(float), s["step"].to_numpy()
        above = np.nan_to_num(gm, nan=-1) > P.SPIKE_MGDL
        b = 1
        while b < len(gm):
            if above[b] and not np.isnan(gm[b - 1]) and gm[b - 1] <= P.SPIKE_MGDL:
                e = b
                while e + 1 < len(gm) and above[e + 1]:
                    e += 1
                eps.append({"segment": seg, "onset_step": int(steps[b]), "bins": e - b + 1, "peak": float(np.nanmax(gm[b:e + 1]))})
                b = e + 1
            else:
                b += 1
    ep = pd.DataFrame(eps)
    ep["serious"] = (ep.peak >= MI.SERIOUS_PEAK) | (ep.bins >= MI.SERIOUS_BINS)
    return ep


def alert_eval(d: pd.DataFrame, p: np.ndarray, thr: Any, ep: pd.DataFrame, suppress_when_high: bool) -> Dict[str, Any]:
    """
    Event-level evaluation of an alert policy.
      alert active at a moment : p >= thr (and glucose <= 180 if suppress_when_high)
      alert FIRES              : active and no alert fired in the previous hour (refractory)
      false alert              : a fired alert with no CGM reading > 180 in the next 2 h (y == 0)
      serious episode warned   : an active alert 30-120 min before onset while glucose <= 180;
                                 lead time = minutes from the EARLIEST such alert to onset
    """
    thr_arr = np.full(len(d), thr) if np.isscalar(thr) else np.asarray(thr)
    active = (p >= thr_arr) & ((d.glucose.to_numpy() <= P.SPIKE_MGDL) if suppress_when_high else True)
    dd = pd.DataFrame({"segment": d.segment.to_numpy(), "step": d.step.to_numpy(), "patient_id": d.patient_id.to_numpy(),
                       "y": d[LABEL].to_numpy(), "glucose": d.glucose.to_numpy(), "active": active})
    fired = np.zeros(len(dd), bool)
    for _, idx in dd.groupby("segment", sort=False).indices.items():
        idx = idx[np.argsort(dd.step.to_numpy()[idx])]
        last = -10 ** 9
        st = dd.step.to_numpy()[idx]
        ac = dd.active.to_numpy()[idx]
        for k, i in enumerate(idx):
            if ac[k] and st[k] - last >= REFRACTORY_BINS:
                fired[i], last = True, st[k]
    dd["fired"] = fired
    days = len(dd) / P.DAY
    false_alerts = int((dd.fired & (dd.y == 0)).sum())
    act = dd.set_index(["segment", "step"])[["active", "glucose"]]
    warned, lead, evaluable = [], [], []
    segs = set(dd.segment)
    for r in ep[ep.segment.isin(segs)].itertuples():
        w = act.reindex([(r.segment, r.onset_step - k) for k in range(2, 9)]).dropna()
        w = w[w.glucose <= P.SPIKE_MGDL]
        evaluable.append(len(w) > 0)
        hit = w[w.active.astype(bool)]
        warned.append(len(hit) > 0)
        lead.append(15 * (r.onset_step - min(k[1] for k in hit.index)) if len(hit) else np.nan)   # first warning
    e = ep[ep.segment.isin(segs)].assign(evaluable=np.asarray(evaluable, bool), warned=np.asarray(warned, bool),
                                         lead=np.asarray(lead, float))
    e = e[e.evaluable.to_numpy(bool)]                    # explicit bool: an empty column must not act as a selector
    ser = e[e.serious.to_numpy(bool)]
    return {"patient_days": round(days, 1), "people": int(dd.patient_id.nunique()),
            "alerts_fired": int(dd.fired.sum()), "false_alerts": false_alerts,
            "false_alerts_per_patient_day": round(false_alerts / days, 3),
            "alerts_per_patient_day": round(float(dd.fired.sum()) / days, 3),
            "serious_episodes": int(len(ser)), "serious_warned": int(ser.warned.sum()),
            "serious_recall": round(float(ser.warned.mean()), 4) if len(ser) else None,
            "all_episodes": int(len(e)), "all_recall": round(float(e.warned.mean()), 4) if len(e) else None,
            "serious_lead_min_median": float(ser.lead.median()) if ser.warned.any() else None}


def summarise(d: pd.DataFrame, p: np.ndarray, thr: Any) -> Dict[str, Any]:
    y = d[LABEL].astype(int).to_numpy()
    r = metrics(y, p, thr)
    t2 = (d.diabetes_type == "T2D").to_numpy()
    r["type2"] = {k: metrics(y[t2], p[t2], np.asarray(thr)[t2] if not np.isscalar(thr) else thr)[k]
                  for k in ("accuracy", "recall", "precision", "roc_auc")} if t2.any() else None
    r["people"] = int(d.patient_id.nunique())
    return r


def _fit(df: pd.DataFrame, cols: List[str]):
    return V7.fit(df, LABEL, cols)                       # identical config / weights / seed to v7


def cv_candidate(dev: pd.DataFrame, cols: List[str], tag: str) -> Dict[str, np.ndarray]:
    fold_of = {p: i % N_FOLDS for i, p in enumerate(sorted(dev.patient_id.unique(),
                                                         key=lambda x: hashlib.md5(f"{SPLIT_SEED}{x}".encode()).hexdigest()))}
    fold = dev.patient_id.map(fold_of).to_numpy()
    p, ta, tb = (np.full(len(dev), np.nan) for _ in range(3))
    t0 = time.time()
    for f in range(N_FOLDS):
        te = fold == f
        tr = dev[~te].reset_index(drop=True)
        inner = np.full(len(tr), np.nan)
        for a, b in GroupKFold(V7.N_INNER).split(tr, tr[LABEL], tr.patient_id):
            inner[b] = _fit(tr.iloc[a], cols).predict_proba(tr.iloc[b][cols])[:, 1]
        y_tr = tr[LABEL].astype(int).to_numpy()
        ta[te], tb[te] = V7.acc_threshold(y_tr, inner), best_threshold(y_tr, inner)
        p[te] = _fit(tr, cols).predict_proba(dev.loc[te, cols])[:, 1]
        print(f"[cv {tag}] fold {f + 1}/{N_FOLDS} ({time.time() - t0:.0f}s)", flush=True)
    return {"p": p, "thr_acc": ta, "thr_bal": tb, "fold": fold}


# ── stages ────────────────────────────────────────────────────────────────────

def _dev_test(m: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sp = _load("split")
    lab = m[m[LABEL].notna()]
    dev = lab[lab.patient_id.isin(sp["dev_people"])].reset_index(drop=True)
    test = lab[lab.patient_id.isin(sp["test_people"])].reset_index(drop=True)
    ohio = lab[lab.source.isin(V7.EXTERNAL)].reset_index(drop=True)
    return dev, test, ohio


def stage_cv(m: pd.DataFrame) -> Dict[str, Any]:
    dev, _, _ = _dev_test(m)
    b15 = MI.load_bins()
    ep = episodes(b15)
    oof, res = {}, {"population": "DEV people only (validation)", "people": int(dev.patient_id.nunique()), "rows": int(len(dev)),
                    "candidates": {}}
    for name, cols in CANDIDATES.items():
        frame = base_inputs(dev) if name.startswith("base") or "C2" not in name else dev
        out = cv_candidate(frame, cols, name)
        oof[name] = out
        res["candidates"][name] = {"features": len(cols), "metrics": summarise(dev, out["p"], out["thr_acc"]),
                                   "alerts_baseline_policy": alert_eval(dev, out["p"], out["thr_acc"], ep, False)}
    base = "base (v7 inputs)"
    for name in CANDIDATES:
        if name == base:
            continue
        bs = cluster_bootstrap(dev.assign(y=dev[LABEL].astype(int)), {"base": oof[base]["p"], "cand": oof[name]["p"]},
                               {"base": oof[base]["thr_acc"], "cand": oof[name]["thr_acc"]}, n=300)["cand"]["vs_base"]
        keep = bs["roc_auc_diff_ci95"][0] > 0 and bs["accuracy_diff_ci95"][1] >= 0
        res["candidates"][name]["vs_base"] = bs
        res["candidates"][name]["retained"] = bool(keep)
    kept = [n for n, c in res["candidates"].items() if c.get("retained")]
    if len(kept) >= 2:                                   # validate the combination itself before promoting it
        cols = list(V7.FEATURES)
        for n in kept:
            cols += [c for c in CANDIDATES[n] if c not in cols]
        name = "combined: " + " + ".join(kept)
        frame = dev if any("C2" in n for n in kept) else base_inputs(dev)
        out = cv_candidate(frame, cols, name)
        oof[name] = out
        res["candidates"][name] = {"features": len(cols), "metrics": summarise(dev, out["p"], out["thr_acc"]),
                                   "alerts_baseline_policy": alert_eval(dev, out["p"], out["thr_acc"], ep, False)}
        bs = cluster_bootstrap(dev.assign(y=dev[LABEL].astype(int)), {"base": oof[base]["p"], "cand": out["p"]},
                               {"base": oof[base]["thr_acc"], "cand": out["thr_acc"]}, n=300)["cand"]["vs_base"]
        res["candidates"][name]["vs_base"] = bs
        res["candidates"][name]["retained"] = bool(bs["roc_auc_diff_ci95"][0] > 0 and bs["accuracy_diff_ci95"][1] >= 0)
    pd.DataFrame({"segment": dev.segment, "step": dev.step, "patient_id": dev.patient_id, "y": dev[LABEL],
                  **{f"p::{k}": v["p"] for k, v in oof.items()}, **{f"thr::{k}": v["thr_acc"] for k, v in oof.items()},
                  **{f"thrbal::{k}": v["thr_bal"] for k, v in oof.items()}}).to_parquet(OUT / "oof_dev.parquet")
    _save("stage_cv", res)
    return res


def selected(cv: Dict[str, Any]) -> Tuple[str, List[str], bool]:
    """(candidate name in stage_cv, feature list, mask Shanghai carbs?) - the base if nothing was retained."""
    kept = [n for n, c in cv["candidates"].items() if c.get("retained") and not n.startswith("combined")]
    combined = [n for n in cv["candidates"] if n.startswith("combined") and cv["candidates"][n].get("retained")]
    if len(kept) >= 2 and not combined:                 # the combination failed validation -> keep the best single
        kept = [max(kept, key=lambda n: cv["candidates"][n]["vs_base"]["roc_auc_diff_mean"])]
    if not kept:
        return "base (v7 inputs)", list(V7.FEATURES), True
    name = combined[0] if len(kept) >= 2 else kept[0]
    cols = list(V7.FEATURES)
    for n in kept:
        cols += [c for c in CANDIDATES[n] if c not in cols]
    return name, cols, not any("C2" in n for n in kept)


def stage_policy(m: pd.DataFrame) -> Dict[str, Any]:
    """Choose the alert policy on DEV validation predictions only."""
    dev, _, _ = _dev_test(m)
    cv = _load("stage_cv")
    o = pd.read_parquet(OUT / "oof_dev.parquet")
    ep = episodes(MI.load_bins())
    res = {"population": "DEV validation predictions", "rule": "maximise serious-episode recall subject to false alerts "
           "per patient-day <= the v7 policy's on the same validation data; alerts suppressed while glucose > 180",
           "models": {}}
    names = ["base (v7 inputs)"] + ([selected(cv)[0]] if selected(cv)[0] != "base (v7 inputs)" else [])
    ref = alert_eval(dev, o["p::base (v7 inputs)"].to_numpy(), o["thr::base (v7 inputs)"].to_numpy(), ep, False)
    res["v7_policy_on_base"] = ref
    for name in names:
        p = o[f"p::{name}"].to_numpy()
        grid = [{"threshold": float(t), **alert_eval(dev, p, float(t), ep, True)} for t in np.round(np.arange(0.15, 0.705, 0.01), 2)]
        ok = [g for g in grid if g["false_alerts_per_patient_day"] <= ref["false_alerts_per_patient_day"]]
        res["models"][name] = {"grid_with_suppression": grid,
                               "chosen": max(ok, key=lambda g: g["serious_recall"]) if ok else None,
                               "accuracy_threshold": float(np.median(o[f"thr::{name}"]))}
    _save("stage_policy", res)
    return res


def stage_final(m: pd.DataFrame) -> Dict[str, Any]:
    dev, test, ohio = _dev_test(m)
    cv, pol = _load("stage_cv"), _load("stage_policy")
    o = pd.read_parquet(OUT / "oof_dev.parquet")
    sel_name, sel_cols, sel_mask = selected(cv)
    ep = episodes(MI.load_bins())
    models = {"baseline (v7 config, DEV-trained)": ("base (v7 inputs)", V7.FEATURES, True)}
    if sel_name != "base (v7 inputs)":
        models[f"candidate: {sel_name}"] = (sel_name, sel_cols, sel_mask)
    res: Dict[str, Any] = {"selected": sel_name, "test_people": int(test.patient_id.nunique()), "test_rows": int(len(test)),
                           "thresholds_from_dev": {k: {"accuracy": pol["models"][v[0]]["accuracy_threshold"],
                                                       "policy": (pol["models"][v[0]]["chosen"] or {}).get("threshold")}
                                                   for k, v in models.items()}, "results": {}}
    for mname, (cvname, cols, mask) in models.items():
        thr_acc = res["thresholds_from_dev"][mname]["accuracy"]
        pol_thr = res["thresholds_from_dev"][mname]["policy"]
        role = "baseline" if mname.startswith("baseline") else "candidate"
        fit_dev = base_inputs(dev) if mask else dev
        clf = _fit(fit_dev, cols)
        joblib.dump({"model": clf, "feature_cols": cols, "threshold_accuracy_mode": thr_acc, "mask_shanghai_carbs": mask,
                     "policy_threshold": pol_thr, "trained_on": "DEV people"}, OUT / f"model_{role}_dev.joblib")
        blk = {}
        for tag, df in (("test", test), ("ohio_external", ohio)):
            x = base_inputs(df) if mask else df
            p = clf.predict_proba(x[cols])[:, 1]
            r = summarise(df, p, thr_acc)
            r["accuracy_ci95_by_person"] = cluster_bootstrap(df.assign(y=df[LABEL].astype(int)), {"m": p}, {"m": thr_acc}, n=300)["m"]["accuracy_ci95"]
            r["alerts_v7_policy"] = alert_eval(df, p, thr_acc, ep, False) if tag == "test" else None
            if tag == "test" and pol_thr is not None:
                r["alerts_chosen_policy"] = alert_eval(df, p, pol_thr, ep, True)
            r["by_source"] = {s: metrics(df[LABEL][df.source == s].astype(int), p[(df.source == s).to_numpy()], thr_acc)["accuracy"]
                              for s in sorted(df.source.unique())}
            blk[tag] = r
            if tag == "test":
                pd.DataFrame({"segment": df.segment, "step": df.step, "patient_id": df.patient_id, "y": df[LABEL], "p": p}
                             ).to_parquet(OUT / f"test_pred_{role}.parquet")
        res["results"][mname] = blk
    # chronological holdout: DEV + first half of each TEST person's record -> second half
    # frame rows are in recording order within each person (loaders emit segments chronologically)
    half = test.groupby("patient_id").cumcount() < test.groupby("patient_id").patient_id.transform("size") / 2
    early, late = test[half.to_numpy()], test[~half.to_numpy()]
    res["chronological_holdout"] = {"train": "DEV + first half of each TEST person's record", "test": "second half of TEST records"}
    for mname, (cvname, cols, mask) in models.items():
        thr_acc = res["thresholds_from_dev"][mname]["accuracy"]
        tr = pd.concat([dev, early], ignore_index=True)
        clf = _fit(base_inputs(tr) if mask else tr, cols)
        x = base_inputs(late) if mask else late
        p = clf.predict_proba(x[cols])[:, 1]
        res["chronological_holdout"][mname] = summarise(late, p, thr_acc)
    if len(models) == 2:
        tb, tc = pd.read_parquet(OUT / "test_pred_baseline.parquet"), pd.read_parquet(OUT / "test_pred_candidate.parquet")
        th = [res["thresholds_from_dev"][k]["accuracy"] for k in models]
        bs = cluster_bootstrap(test.assign(y=test[LABEL].astype(int)), {"base": tb.p.to_numpy(), "cand": tc.p.to_numpy()},
                               {"base": th[0], "cand": th[1]}, n=400)["cand"]["vs_base"]
        res["test_candidate_vs_baseline"] = bs
    _save("stage_final", res)
    return res


def stage_labels(m: pd.DataFrame) -> Dict[str, Any]:
    """Pre-defined label-uncertainty sensitivity on the TEST predictions of the DEV-trained baseline."""
    _, test, _ = _dev_test(m)
    tp = pd.read_parquet(OUT / "test_pred_baseline.parquet")
    thr = _load("stage_policy")["models"]["base (v7 inputs)"]["accuracy_threshold"]
    fut = MI.future_table(MI.load_bins())
    d = test.merge(fut, on=["segment", "step"], how="left").assign(p=tp.p.to_numpy())
    pred = (d.p >= thr).astype(int)
    res: Dict[str, Any] = {"population": "TEST people, DEV-trained baseline, accuracy threshold from DEV",
                           "primary_label": "peak > 180 (unchanged)",
                           "note": "Alternative labels re-score the SAME predictions; they are sensitivity analyses, not new targets.",
                           "alternative_labels": {}, "uncertainty_bands": {}}
    for name, lab in (("peak > 170", d.fut_max > 170), ("peak > 180 (primary)", d.fut_max > 180),
                      ("peak > 190", d.fut_max > 190), ("peak > 198 (180 + 10 %)", d.fut_max > 198),
                      ("> 180 for >= 2 consecutive bins", d.fut_consec_bins_above >= 2)):
        y = lab.astype(int).to_numpy()
        res["alternative_labels"][name] = {"prevalence": round(float(y.mean()), 4), "accuracy": round(float((pred == y).mean()), 4),
                                           "recall": round(float(((pred == 1) & (y == 1)).sum() / max(y.sum(), 1)), 4),
                                           "roc_auc": round(float(roc_auc_score(y, d.p)), 4)}
    for band in (0.05, 0.10, 0.15):
        lo, hi = 180 * (1 - band), 180 * (1 + band)
        unc = d.fut_max.between(lo, hi).to_numpy()
        ok = (pred == d[LABEL].astype(int)).to_numpy()
        res["uncertainty_bands"][f"±{int(band * 100)} % ({lo:.0f}-{hi:.0f})"] = {
            "coverage_clear": round(float((~unc).mean()), 4), "accuracy_clear": round(float(ok[~unc].mean()), 4),
            "accuracy_near": round(float(ok[unc].mean()), 4), "share_of_errors_near": round(float(unc[~ok].mean()), 4)}
    res["reference_disagreement"] = "Shanghai CGM vs fingerstick: see models/investigation/analysis.json E (MARD 13.9 %); that is sensor-meter disagreement, not true sensor error."
    _save("stage_labels", res)
    return res


def stage_report(m: pd.DataFrame) -> Dict[str, Any]:
    """Person-bootstrap CIs for the TEST alert results (saved TEST predictions only - nothing is refitted)."""
    _, test, _ = _dev_test(m)
    fin = _load("stage_final")
    ep = episodes(MI.load_bins())
    arms = {}
    for mname, role in (("baseline (v7 config, DEV-trained)", "baseline"), (f"candidate: {fin['selected']}", "candidate")):
        tp = pd.read_parquet(OUT / f"test_pred_{role}.parquet")
        thr = fin["thresholds_from_dev"][mname]
        arms[f"{role} + v7 policy"] = (tp.p.to_numpy(), thr["accuracy"], False)
        arms[f"{role} + v9 policy"] = (tp.p.to_numpy(), thr["policy"], True)
    per = {a: {} for a in arms}                          # person -> (serious episodes, warned, false alerts, rows)
    for pid, idx in test.groupby("patient_id").indices.items():
        d = test.iloc[idx]
        for a, (p, thr, sup) in arms.items():
            r = alert_eval(d, p[idx], thr, ep, sup)
            per[a][pid] = (r["serious_episodes"], r["serious_warned"], r["false_alerts"], len(d))
    people = sorted(per[next(iter(arms))])
    arr = {a: np.array([per[a][q] for q in people], float) for a in arms}
    stat = lambda x: (x[:, 1].sum() / max(x[:, 0].sum(), 1), x[:, 2].sum() / (x[:, 3].sum() / P.DAY))
    rng = np.random.default_rng(SPLIT_SEED)
    boots = {a: [] for a in arms}
    for _ in range(2000):
        ix = rng.integers(0, len(people), len(people))
        for a in arms:
            boots[a].append(stat(arr[a][ix]))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    res: Dict[str, Any] = {"population": f"TEST people ({len(people)}), saved TEST predictions", "bootstrap": "2000 person resamples",
                           "arms": {}, "paired_vs_baseline_v7_policy": {}}
    ref = "baseline + v7 policy"
    for a in arms:
        b = np.array(boots[a])
        rec, fa = stat(arr[a])
        res["arms"][a] = {"serious_episodes": int(arr[a][:, 0].sum()), "serious_warned": int(arr[a][:, 1].sum()),
                          "serious_recall": round(rec, 4), "serious_recall_ci95": ci(b[:, 0]),
                          "false_alerts": int(arr[a][:, 2].sum()), "false_alerts_per_patient_day": round(fa, 3),
                          "false_alerts_per_patient_day_ci95": ci(b[:, 1])}
        if a != ref:
            dif = b - np.array(boots[ref])
            res["paired_vs_baseline_v7_policy"][a] = {"serious_recall_diff": round(rec - stat(arr[ref])[0], 4),
                                                      "serious_recall_diff_ci95": ci(dif[:, 0]),
                                                      "false_alerts_per_day_diff": round(fa - stat(arr[ref])[1], 3),
                                                      "false_alerts_per_day_diff_ci95": ci(dif[:, 1])}
    _save("stage_report", res)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["build", "split", "cv", "policy", "final", "labels", "report", "all"])
    ap.add_argument("--force-split", action="store_true")
    a = ap.parse_args()
    stages = ["build", "split", "cv", "policy", "final", "labels", "report"] if a.stage == "all" else [a.stage]
    m = None
    for s in stages:
        t = time.time()
        print(f"\n===== stage: {s} =====", flush=True)
        if s == "build":
            m = stage_build()
            continue
        m = m if m is not None else load_frame()
        {"split": lambda: stage_split(m, a.force_split), "cv": lambda: stage_cv(m), "policy": lambda: stage_policy(m),
         "final": lambda: stage_final(m), "labels": lambda: stage_labels(m), "report": lambda: stage_report(m)}[s]()
        print(f"===== {s} finished in {time.time() - t:.0f}s =====", flush=True)


if __name__ == "__main__":
    main()
