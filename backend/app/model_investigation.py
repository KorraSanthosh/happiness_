"""
Investigation of the v7 model: where is the remaining error, and could NEW, VALID information reduce it?
(No new algorithms - see the model-improvement policy in CLAUDE.md.) Every number in
docs/model_investigation.md is produced here from the v7 held-out predictions; app/verify_results.py
re-derives the headline figures independently and checks them.

Evaluation protocol for everything below unless stated otherwise:
  predictions  models/combined_v7/oof_{30m,60m,2h}.parquet - 7-fold PERSON-grouped nested CV over the 187
               open-data people; each row is predicted by a model that never saw that person
  threshold    "accuracy mode": per-fold threshold chosen on inner folds (thr_acc column; median 0.51)
  unit         a prediction moment = one 15-min bin with a real CGM reading and a known label
  label        y = 1 if the max CGM reading in the next 8 bins (2 h) is > 180 mg/dL

  A. data quality & label reliability
  B. missed spikes (moment level and distinct-episode level)
  C. calibration (held-out predictions)
  D. patient-level generalisation
  E. CGM vs fingerstick (Shanghai capillary blood glucose)

python3 -I scripts/run_isolated.py -m:app.model_investigation   ->  models/investigation/analysis.json
"""
from __future__ import annotations

import glob
import json
import os
import re
import time
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from app import external_loaders as E
from app import real_data_pipeline as P
from app import real_data_v7 as V7
from app.real_data_eval import ece

warnings.filterwarnings("ignore")

OUT_DIR = P.ROOT / "models" / "investigation"
ASSUMED_CGM_ERROR = 0.10        # +-10 % band around 180 used to flag "uncertain" labels (typical CGM MARD 9-10 %)
H8 = P.HORIZON_STEPS            # 2 h in 15-min bins
MIN_ROWS_PER_PERSON = 200       # per-person statistics need >= 200 prediction moments (~2 days)
SERIOUS_PEAK = 250.0            # serious spike: peak >= 250 mg/dL ...
SERIOUS_BINS = 3                # ... or >= 3 consecutive 15-min bins with a reading > 180
N_BOOT = 300
RNG = np.random.default_rng(42)


# ── shared data ───────────────────────────────────────────────────────────────

def load_bins() -> pd.DataFrame:
    """15-min bins of the six open datasets, built exactly as for v7 (Shanghai meals off)."""
    g = pd.concat([P.load_brist_grid("train")[0],
                   E.load_external_grids(("hupa", "uom", "azt1d", "sh_t1", "sh_t2"))[0]], ignore_index=True)
    return P.to_15min(g)


def future_table(b15: pd.DataFrame) -> pd.DataFrame:
    """Per (segment, step): the next-2-h window as the label sees it, plus meals logged in it."""
    rows = []
    for seg, s in b15.groupby("segment", sort=False):
        s = s.sort_values("step")
        gm, carbs, n = s["glucose_max"].to_numpy(float), s["carbs"].to_numpy(float), len(s)
        win = np.lib.stride_tricks.sliding_window_view(np.r_[gm, np.full(H8, np.nan)][1:], H8)[:n]
        cwin = np.lib.stride_tricks.sliding_window_view(np.r_[carbs, np.full(H8, np.nan)][1:], H8)[:n]
        above = win > P.SPIKE_MGDL
        run, best = np.zeros(n, int), np.zeros(n, int)            # longest consecutive run of bins above 180
        for j in range(H8):
            run = np.where(above[:, j], run + 1, 0)
            best = np.maximum(best, run)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows.append(pd.DataFrame({
                "segment": seg, "step": s["step"].to_numpy(),
                "fut_max": np.nanmax(np.where(np.isnan(win), -np.inf, win), axis=1),
                "fut_bins_above": above.sum(axis=1), "fut_consec_bins_above": best,
                "fut_first_above_min": np.where(above.any(axis=1), above.argmax(axis=1) + 1, 0) * P.STEP_MIN,
                "fut_carbs_2h": np.where(np.isnan(cwin).all(axis=1), np.nan, np.nansum(cwin, axis=1))}))
    return pd.concat(rows, ignore_index=True)


def load_eval(horizon: str = "2h") -> pd.DataFrame:
    """v7 held-out predictions joined with features of the moment and (for 2 h) the future window."""
    m = pd.read_parquet(V7.OUT_DIR / "frame.parquet",
                        columns=["segment", "step", "source", "diabetes_type", "glucose", "g_slope_30m",
                                 "clock_min", "carbs_2h", "ehr_hba1c_pct"])
    d = pd.read_parquet(V7.OUT_DIR / f"oof_{horizon}.parquet").merge(m, on=["segment", "step"], how="left")
    d["pred"] = (d.p >= d.thr_acc).astype(int)
    return d


def _ci(values: np.ndarray) -> List[float]:
    return [round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)]


def _person_boot(df: pd.DataFrame, stat, n: int = N_BOOT) -> List[float]:
    """95 % CI of stat(df) resampling PEOPLE (rows of one person are not independent)."""
    groups = [g.index.to_numpy() for _, g in df.groupby("patient_id")]
    vals = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in RNG.integers(0, len(groups), len(groups))])
        vals.append(stat(df.loc[idx]))
    return _ci(np.array(vals))


# ── A. data quality & label reliability ───────────────────────────────────────

def patient_quality(b15: pd.DataFrame) -> pd.DataFrame:
    out = []
    for pid, g in b15.groupby("patient_id", sort=False):
        g = g.sort_values(["segment", "step"])
        obs = g["glucose"].notna().to_numpy()
        v = g["glucose"].dropna().to_numpy()
        d = np.abs(np.diff(v)) if len(v) > 1 else np.array([0.0])
        days = len(g) / P.DAY
        miss_runs, run = [], 0
        for o in obs:
            if o:
                if run:
                    miss_runs.append(run)
                run = 0
            else:
                run += 1
        flat, cur = 0, 1                                            # >= 4 identical consecutive readings (1 h)
        for a, b in zip(v[:-1], v[1:]):
            cur = cur + 1 if a == b else 1
            flat += cur == 4
        out.append({"patient_id": pid, "source": g.source.iloc[0], "days": round(days, 1),
                    "cgm_coverage": round(float(obs.mean()), 3),
                    "gaps_over_1h_per_day": round(sum(r > 4 for r in miss_runs) / days, 3),
                    "jumps_over_45_per_day": round(float((d > 45).sum() / days), 3),
                    "flatlines_per_day": round(flat / days, 3),
                    "mean_glucose": round(float(v.mean()), 1) if len(v) else np.nan,
                    "cv_pct": round(float(100 * v.std() / v.mean()), 1) if len(v) else np.nan,
                    "time_above_180_pct": round(float(100 * np.mean(v > 180)), 1) if len(v) else np.nan,
                    "carb_logs_per_day": round(float((g["carbs"] > 0).sum() / days), 2) if g["carbs"].notna().any() else np.nan})
    return pd.DataFrame(out)


def label_reliability(d: pd.DataFrame) -> Dict[str, Any]:
    lo, hi = P.SPIKE_MGDL * (1 - ASSUMED_CGM_ERROR), P.SPIKE_MGDL * (1 + ASSUMED_CGM_ERROR)
    unc = d.fut_max.between(lo, hi).to_numpy()
    ok = (d.pred == d.y).to_numpy()
    err = ~ok

    def block(mask) -> Dict[str, Any]:
        u, o = unc[mask], ok[mask]
        return {"rows": int(mask.sum()), "people": int(d.patient_id[mask].nunique()),
                "accuracy_all": round(float(o.mean()), 4),
                "uncertain_rows": int(u.sum()), "uncertain_share_of_rows": round(float(u.mean()), 4),
                "accuracy_clear": round(float(o[~u].mean()), 4), "accuracy_uncertain": round(float(o[u].mean()), 4),
                "errors": int((~o).sum()), "uncertain_share_of_errors": round(float(u[~o].mean()), 4)}
    t2 = (d.diabetes_type == "T2D").to_numpy()
    return {"band_mg_dl": [round(lo), round(hi)],
            "assumption": f"180 ± {int(100 * ASSUMED_CGM_ERROR)} %, a typical-CGM-error assumption, not measured on these sensors",
            "stratified_by": "the ACTUAL future peak - an error decomposition, not a subset a model could pick in advance",
            "all_open": block(np.ones(len(d), bool)), "type2": block(t2), "type1": block(~t2),
            "check_weighted_average": round(float(unc.mean() * ok[unc].mean() + (1 - unc.mean()) * ok[~unc].mean()), 4),
            "errors_total": int(err.sum())}


# ── B. missed spikes ──────────────────────────────────────────────────────────

def moment_misses(d: pd.DataFrame) -> Dict[str, Any]:
    pos = (d.y == 1).to_numpy()
    fn, tp = pos & (d.pred == 0).to_numpy(), pos & (d.pred == 1).to_numpy()
    serious = ((d.fut_max >= SERIOUS_PEAK) | (d.fut_consec_bins_above >= SERIOUS_BINS)).to_numpy()

    def profile(mask) -> Dict[str, Any]:
        x = d[mask]
        meals = x.fut_carbs_2h.notna()
        return {"rows": int(mask.sum()),
                "peak_180_200": round(float((x.fut_max < 200).mean()), 3),
                "peak_250_plus": round(float((x.fut_max >= 250).mean()), 3),
                "spike_starts_after_60min": round(float((x.fut_first_above_min > 60).mean()), 3),
                "glucose_now_median": round(float(x.glucose.median()), 1),
                "rows_with_meal_logging": int(meals.sum()),
                "meal_logged_in_next_2h": round(float((x.fut_carbs_2h[meals] > 0).mean()), 3) if meals.any() else None}
    return {"unit": "prediction moment (overlapping 2-h windows: one spike is counted at several moments)",
            "positives": int(pos.sum()), "missed": int(fn.sum()), "miss_rate": round(float(fn.sum() / pos.sum()), 4),
            "serious_definition": f"actual 2-h peak >= {SERIOUS_PEAK:.0f} mg/dL OR >= {SERIOUS_BINS} consecutive 15-min bins with a reading > 180",
            "serious_positives": int((pos & serious).sum()), "serious_missed": int((fn & serious).sum()),
            "serious_miss_rate": round(float((fn & serious).sum() / (pos & serious).sum()), 4),
            "false_negatives": profile(fn), "true_positives": profile(tp)}


def episode_misses(b15: pd.DataFrame, d: pd.DataFrame) -> Dict[str, Any]:
    """
    Distinct hyperglycaemia episodes: a run of consecutive 15-min bins with a reading > 180, preceded by an
    observed bin <= 180. An episode is WARNED if any held-out 2-h prediction made 30-120 min before onset
    (bins onset-8 .. onset-2, while glucose was <= 180) was at or above its fold threshold. Episodes with no
    held-out prediction in that window are not evaluable and are counted separately.
    """
    pr = d.set_index(["segment", "step"])[["p", "thr_acc", "glucose"]]
    eps = []
    for seg, s in b15.groupby("segment", sort=False):
        s = s.sort_values("step")
        gm = s["glucose_max"].to_numpy(float)
        steps = s["step"].to_numpy()
        above = np.nan_to_num(gm, nan=-1) > P.SPIKE_MGDL
        b = 1
        while b < len(gm):
            if above[b] and not np.isnan(gm[b - 1]) and gm[b - 1] <= P.SPIKE_MGDL:
                e = b
                while e + 1 < len(gm) and above[e + 1]:
                    e += 1
                eps.append({"segment": seg, "onset_step": int(steps[b]), "bins": e - b + 1,
                            "peak": float(np.nanmax(gm[b:e + 1]))})
                b = e + 1
            else:
                b += 1
    ep = pd.DataFrame(eps)
    ep["serious"] = (ep.peak >= SERIOUS_PEAK) | (ep.bins >= SERIOUS_BINS)
    warned, evaluable, lead = [], [], []
    for r in ep.itertuples():
        keys = [(r.segment, r.onset_step - k) for k in range(2, 9)]
        sub = pr.reindex(keys).dropna(subset=["p"])
        sub = sub[sub.glucose <= P.SPIKE_MGDL]
        evaluable.append(len(sub) > 0)
        hit = sub[sub.p >= sub.thr_acc]
        warned.append(len(hit) > 0)
        lead.append(15 * (r.onset_step - max(k[1] for k in hit.index)) if len(hit) else np.nan)
    ep["evaluable"], ep["warned"], ep["lead_min_latest_warning"] = evaluable, warned, lead
    ev = ep[ep.evaluable]

    def rate(x):
        return {"episodes": int(len(x)), "warned": int(x.warned.sum()), "missed": int((~x.warned).sum()),
                "miss_rate": round(float((~x.warned).mean()), 4) if len(x) else None}
    return {"definition": episode_misses.__doc__.strip().replace("\n    ", " "),
            "serious_definition": f"peak >= {SERIOUS_PEAK:.0f} mg/dL OR >= {SERIOUS_BINS} consecutive bins above 180",
            "episodes_found": int(len(ep)), "episodes_evaluable": int(len(ev)),
            "all": rate(ev), "serious": rate(ev[ev.serious]), "non_serious": rate(ev[~ev.serious]),
            "serious_warned_lead_min_median": float(ev[ev.serious & ev.warned].lead_min_latest_warning.median()),
            "serious_miss_rate_ci95_by_person": _person_boot(
                ev[ev.serious].assign(patient_id=ev[ev.serious].segment.map(
                    d.drop_duplicates("segment").set_index("segment").patient_id)).reset_index(drop=True),
                lambda x: float((~x.warned).mean())),
            "note": "Alarm burden is not captured here: a warning counts even if many other alarms were false."}


# ── C. calibration ────────────────────────────────────────────────────────────

def _cal(y: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
    z = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    lr = LogisticRegression(C=1e6).fit(z[:, None], y)
    return {"rows": int(len(y)), "ece": round(ece(y, p), 4), "mean_pred": round(float(p.mean()), 4),
            "observed": round(float(y.mean()), 4), "slope": round(float(lr.coef_[0, 0]), 3),
            "intercept": round(float(lr.intercept_[0]), 3)}


def calibration() -> Dict[str, Any]:
    out: Dict[str, Any] = {"method": "Held-out (out-of-fold) probabilities; ECE = 10 equal-width bins weighted by "
                                     "row count; slope/intercept from logistic regression of y on logit(p); "
                                     "95 % CI by resampling people",
                           "limitations": ["Rows within a person are autocorrelated, so row counts overstate the "
                                           "effective sample size; CIs resample people to account for this.",
                                           "These are the 7 fold models' predictions pooled, not the final model "
                                           "trained on all people.",
                                           f"Per-person gaps use people with >= {MIN_ROWS_PER_PERSON} rows."]}
    for h in V7.HORIZONS:
        o = load_eval(h)
        blk = {"all": _cal(o.y.to_numpy(), o.p.to_numpy())}
        blk["all"]["ece_ci95_by_person"] = _person_boot(o, lambda x: ece(x.y.to_numpy(), x.p.to_numpy()), n=150)
        for col in ("source", "diabetes_type"):
            for k, g in o.groupby(col):
                blk[f"{col}={k}"] = _cal(g.y.to_numpy(), g.p.to_numpy())
        n = o.groupby("patient_id").size()
        gap = o.groupby("patient_id").apply(lambda g: g.p.mean() - g.y.mean())[n >= MIN_ROWS_PER_PERSON]
        blk["per_person"] = {"people": int(len(gap)), "excluded_small": int((n < MIN_ROWS_PER_PERSON).sum()),
                             "median_abs_gap": round(float(gap.abs().median()), 4),
                             "share_over_5pts": round(float((gap.abs() > 0.05).mean()), 3),
                             "share_over_10pts": round(float((gap.abs() > 0.10).mean()), 3),
                             "largest": gap.reindex(gap.abs().sort_values(ascending=False).index).head(5).round(3).to_dict()}
        out[h] = blk
    return out


# ── D. patients ───────────────────────────────────────────────────────────────

def patients(d: pd.DataFrame, q: pd.DataFrame) -> Dict[str, Any]:
    lo, hi = P.SPIKE_MGDL * (1 - ASSUMED_CGM_ERROR), P.SPIKE_MGDL * (1 + ASSUMED_CGM_ERROR)
    per = []
    for pid, g in d.groupby("patient_id"):
        rule = float(((g.glucose > 180).astype(int) == g.y).mean())
        per.append({"patient_id": pid, "source": g.source.iloc[0], "type": g.diabetes_type.iloc[0], "rows": len(g),
                    "accuracy": float((g.pred == g.y).mean()), "rule_accuracy": rule,
                    "gain_vs_rule": float((g.pred == g.y).mean()) - rule,
                    "auc": roc_auc_score(g.y, g.p) if g.y.nunique() == 2 else np.nan,
                    "prevalence": float(g.y.mean()), "uncertain_share": float(g.fut_max.between(lo, hi).mean()),
                    "hba1c": g.ehr_hba1c_pct.iloc[0]})
    per = pd.DataFrame(per).merge(q, on="patient_id", how="left", suffixes=("", "_q"))
    per.to_csv(OUT_DIR / "per_patient.csv", index=False)
    big = per[per.rows >= MIN_ROWS_PER_PERSON]
    within = {}
    for c in ("prevalence", "cv_pct", "jumps_over_45_per_day", "cgm_coverage"):
        within[c] = {}
        for src, g in big.groupby("source"):
            if len(g) >= 8:
                r_acc, p_acc = spearmanr(g.accuracy, g[c])
                r_auc, p_auc = spearmanr(g.auc, g[c], nan_policy="omit")
                within[c][src] = {"acc_r": round(float(r_acc), 2), "acc_p": round(float(p_acc), 3),
                                  "auc_r": round(float(r_auc), 2), "auc_p": round(float(p_auc), 3), "people": int(len(g))}
    return {"people_total": int(len(per)), "people_with_min_rows": int(len(big)),
            "beats_rule_share": round(float((big.gain_vs_rule > 0).mean()), 3),
            "accuracy_p05_p50_p95": big.accuracy.quantile([0.05, 0.5, 0.95]).round(3).tolist(),
            "auc_p05_p50_p95": big.auc.quantile([0.05, 0.5, 0.95]).round(3).tolist(),
            "within_dataset_spearman": within}


# ── E. CGM vs fingerstick (Shanghai) ──────────────────────────────────────────

def cgm_vs_fingerstick() -> Dict[str, Any]:
    root = glob.glob(str(E.EXT_DIR / "shanghai" / "extracted" / "*"))[0]
    rows = []
    for kind in ("T1DM", "T2DM"):
        for f in sorted(glob.glob(f"{root}/Shanghai_{kind}/*")):
            if os.path.basename(f).startswith("~$"):
                continue
            x = pd.read_excel(f)
            x.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in x.columns]
            cg = next(c for c in x.columns if c.lower().startswith("cgm"))
            cb = next((c for c in x.columns if c.lower().startswith("cbg")), None)
            if cb is None:
                continue
            t = pd.to_datetime(x["Date"], errors="coerce")
            y = pd.DataFrame({"t": t, "cgm": pd.to_numeric(x[cg], errors="coerce"), "cbg": pd.to_numeric(x[cb], errors="coerce")})
            y = y.dropna().drop_duplicates(subset=["t"])              # 91 duplicated timestamps in the T2D files
            stem = os.path.splitext(os.path.basename(f))[0]
            rows.append(y.assign(kind=kind, recording=stem, patient_id=f"{kind}_{stem.split('_')[0]}"))
    p = pd.concat(rows, ignore_index=True)
    p = p[(p.cbg >= 40) & (p.cbg <= 600)].reset_index(drop=True)
    p["rel"] = (p.cgm - p.cbg) / p.cbg
    p["ard"] = p.rel.abs()
    per_person_n = p.groupby("patient_id").size()

    def stats(x: pd.DataFrame) -> Dict[str, Any]:
        return {"pairs": int(len(x)), "people": int(x.patient_id.nunique()),
                "mard_pct": round(100 * float(x.ard.mean()), 1), "median_ard_pct": round(100 * float(x.ard.median()), 1),
                "median_bias_pct": round(100 * float(x.rel.median()), 1), "mean_bias_pct": round(100 * float(x.rel.mean()), 1),
                "within_15pct": round(float((x.ard <= 0.15).mean()), 3), "within_20pct": round(float((x.ard <= 0.20).mean()), 3)}
    out = {"source": "ShanghaiT1DM/T2DM files: 'CGM (mg / dl)' and 'CBG (mg / dl)' (capillary blood glucose) columns",
           "pairing": "A CBG value is recorded on a row of the 15-min CGM grid; that row's CGM value is its pair. The "
                      "true fingerstick time is not given, so time alignment is only to within the grid step.",
           "method": "relative difference (CGM - CBG) / CBG per pair; MARD = mean |diff|; bias = median diff; duplicate "
                     "timestamps dropped; CBG outside 40-600 mg/dL dropped; 95 % CI by resampling people",
           "overall": stats(p),
           "median_ard_ci95_pct": [round(100 * v, 1) for v in _person_boot(p, lambda x: float(x.ard.median()))],
           "median_bias_ci95_pct": [round(100 * v, 1) for v in _person_boot(p, lambda x: float(x.rel.median()))],
           "by_type": {k: stats(g) for k, g in p.groupby("kind")},
           "by_cbg_range": {k: stats(g) for k, g in p.groupby(pd.cut(p.cbg, [0, 70, 180, 1000], labels=["<70", "70-180", ">180"]))},
           "pairs_per_person_median": int(per_person_n.median()),
           "caveats": ["Fingersticks were taken at clinical times (often around meals), not at random moments.",
                       "Glucose meters have their own error (ISO 15197 allows ±15 %), so this is CGM-meter "
                       "disagreement, not the CGM's true error.",
                       "Applies to the Shanghai sensors only; the other datasets use different CGM devices and "
                       "have no reference readings."]}
    return out


def _no_nan(x):
    if isinstance(x, dict):
        return {k: _no_nan(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_no_nan(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    b15 = load_bins()
    d = load_eval("2h").merge(future_table(b15), on=["segment", "step"], how="left")
    assert d.fut_max.notna().all(), "every labelled row must have a future window"
    assert ((d.fut_max > P.SPIKE_MGDL).astype(int) == d.y).all(), "label must equal (future 2-h max > 180)"
    print(f"[data] {len(d):,} held-out 2-h rows, labels re-derived and matched ({time.time() - t0:.0f}s)", flush=True)
    q = patient_quality(b15)
    res: Dict[str, Any] = {
        "protocol": {"predictions": "models/combined_v7/oof_*.parquet (7-fold person-grouped nested CV, 187 people)",
                     "threshold": "accuracy mode, per fold (thr_acc)", "unit": "15-min prediction moment",
                     "label": "max CGM over next 2 h > 180 mg/dL", "rows_2h": int(len(d)), "people": int(d.patient_id.nunique())},
        "A_label_reliability": label_reliability(d),
        "A_data_quality_by_dataset": q.drop(columns=["patient_id"]).groupby("source").median(numeric_only=True).round(3).to_dict(orient="index"),
        "B_missed_moments": moment_misses(d),
    }
    print(f"[A,B moments] ({time.time() - t0:.0f}s)", flush=True)
    res["B_missed_episodes"] = episode_misses(b15, d)
    print(f"[B episodes] ({time.time() - t0:.0f}s)", flush=True)
    res["C_calibration"] = calibration()
    print(f"[C] ({time.time() - t0:.0f}s)", flush=True)
    res["D_patients"] = patients(d, q)
    res["E_cgm_vs_fingerstick"] = cgm_vs_fingerstick()
    with open(OUT_DIR / "analysis.json", "w") as f:
        json.dump(_no_nan(res), f, indent=2, default=str, allow_nan=False)      # strict JSON: NaN -> null
    print(f"[done] {OUT_DIR / 'analysis.json'} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
