"""
v9 step 2 - error analysis of the FROZEN v7 baseline before any change (2-hour horizon).

For false negatives (missed spikes), false positives (false alerts) and serious missed EPISODES, measure how
the error RATE varies with each candidate explanation, using only information available at prediction time
(plus the actual outcome, to describe the error). Rates within strata show where the model fails;
a factor is only "supported" if the error rate clearly differs across its strata.

python3 -I scripts/run_isolated.py -m:app.error_analysis_v9   ->  models/combined_v9/error_analysis.json
"""
from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np
import pandas as pd

from app import model_investigation as MI
from app import real_data_pipeline as P
from app import real_data_v7 as V7

OUT = P.ROOT / "models" / "combined_v9"
FEATS = ["glucose", "g_slope_30m", "g_slope_1h", "g_std_2h", "p_std_24h", "p_frac_above_24h", "p_same_time_spike_7d",
         "g_obs_frac_2h", "carbs_2h", "since_meal", "cob", "insulin_2h", "iob", "since_bolus", "steps_1h", "hr_now",
         "is_night", "hour", "cgm_native"]


def _rates(d: pd.DataFrame, by: pd.Series, name: str) -> Dict[str, Any]:
    """Miss rate among actual spikes and false-alert rate among non-spikes, per stratum."""
    out = {}
    for k, g in d.groupby(by, observed=True):
        pos, neg = g[g.y == 1], g[g.y == 0]
        out[str(k)] = {"moments": int(len(g)), "people": int(g.patient_id.nunique()),
                       "spikes": int(len(pos)), "miss_rate": round(float((pos.pred == 0).mean()), 3) if len(pos) else None,
                       "non_spikes": int(len(neg)), "false_alert_rate": round(float((neg.pred == 1).mean()), 3) if len(neg) else None}
    return {"factor": name, "strata": out}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(V7.OUT_DIR / "frame.parquet", columns=["segment", "step", "source", "diabetes_type"] + FEATS)
    b15 = MI.load_bins()
    d = pd.read_parquet(V7.OUT_DIR / "oof_2h.parquet").merge(m, on=["segment", "step"], how="left") \
        .merge(MI.future_table(b15), on=["segment", "step"], how="left")
    d["pred"] = (d.p >= d.thr_acc).astype(int)
    known_meals = d.carbs_2h.notna()
    known_ins = d.insulin_2h.notna()
    res: Dict[str, Any] = {"baseline": "v7 held-out 2-h predictions (accuracy mode)", "moments": int(len(d)),
                           "people": int(d.patient_id.nunique()),
                           "overall": {"miss_rate": round(float((d[d.y == 1].pred == 0).mean()), 4),
                                       "false_alert_rate": round(float((d[d.y == 0].pred == 1).mean()), 4)}}
    factors = [
        ("glucose now (mg/dL)", pd.cut(d.glucose, [0, 100, 140, 160, 180, 250, 1000])),
        ("trend last 30 min (mg/dL per h)", pd.cut(d.g_slope_30m, [-1e9, -30, -10, 10, 30, 60, 1e9])),
        ("variability last 2 h (SD, mg/dL)", pd.qcut(d.g_std_2h, 4, duplicates="drop")),
        ("person's spike rate at this time, past 7 days", pd.cut(d.p_same_time_spike_7d, [-0.01, 0.0, 0.25, 0.5, 0.75, 1.0])),
        ("CGM coverage last 2 h", pd.cut(d.g_obs_frac_2h, [0, 0.75, 0.99, 1.0])),
        ("CGM native interval", d.cgm_native),
        ("meal logged in last 2 h (where meals recorded)", np.where(~known_meals, "not recorded", np.where(d.carbs_2h > 0, "yes", "no"))),
        ("insulin in last 2 h (where recorded)", np.where(~known_ins, "not recorded", np.where(d.insulin_2h > 0.5, "> 0.5 U", "<= 0.5 U"))),
        ("steps last hour (where recorded)", np.where(d.steps_1h.isna(), "not recorded", np.where(d.steps_1h > 500, "> 500", "<= 500"))),
        ("night 22-06", np.where(d.is_night == 1, "night", "day")),
        ("diabetes type", d.diabetes_type), ("dataset", d.source),
    ]
    res["by_factor"] = [_rates(d, pd.Series(np.asarray(f), index=d.index), name) for name, f in factors]

    fn = d[(d.y == 1) & (d.pred == 0)]
    tp = d[(d.y == 1) & (d.pred == 1)]
    res["missed_spike_types"] = {
        "time_to_onset_min": {"missed": fn.fut_first_above_min.describe()[["25%", "50%", "75%"]].round(0).to_dict(),
                              "caught": tp.fut_first_above_min.describe()[["25%", "50%", "75%"]].round(0).to_dict()},
        "miss_rate_by_onset_time": {f"{lo}-{hi} min": round(float((d[(d.y == 1) & d.fut_first_above_min.between(lo, hi)].pred == 0).mean()), 3)
                                    for lo, hi in ((15, 30), (45, 60), (75, 90), (105, 120))},
        "miss_rate_by_peak": {str(k): round(float((g.pred == 0).mean()), 3) for k, g in
                              d[d.y == 1].groupby(pd.cut(d[d.y == 1].fut_max, [180, 200, 250, 1000]), observed=True)},
        "miss_rate_when_meal_logged_after_prediction": {
            k: round(float((g.pred == 0).mean()), 3) for k, g in d[(d.y == 1) & d.fut_carbs_2h.notna()].groupby(
                np.where(d[(d.y == 1) & d.fut_carbs_2h.notna()].fut_carbs_2h > 0, "meal after prediction", "no meal after"))},
    }
    fp = d[(d.y == 0) & (d.pred == 1)]
    res["false_alert_types"] = {
        "future_peak_of_false_alerts": fp.fut_max.describe()[["25%", "50%", "75%"]].round(1).to_dict(),
        "share_within_10pct_below_180": round(float(fp.fut_max.between(162, 180).mean()), 3),
        "glucose_now_of_false_alerts": fp.glucose.describe()[["25%", "50%", "75%"]].round(1).to_dict(),
        "share_already_above_180_now": round(float((fp.glucose > 180).mean()), 3),
    }
    per = d.groupby("patient_id").apply(lambda g: pd.Series({
        "miss_rate": (g[g.y == 1].pred == 0).mean() if (g.y == 1).any() else np.nan,
        "false_alert_rate": (g[g.y == 0].pred == 1).mean() if (g.y == 0).any() else np.nan, "n": len(g)}))
    per = per[per.n >= 200]
    res["patient_concentration"] = {
        "people": int(len(per)), "miss_rate_p10_p50_p90": per.miss_rate.quantile([0.1, 0.5, 0.9]).round(3).tolist(),
        "false_alert_rate_p10_p50_p90": per.false_alert_rate.quantile([0.1, 0.5, 0.9]).round(3).tolist(),
        "share_of_misses_from_worst_10pct_people": round(float(
            fn.groupby("patient_id").size().sort_values(ascending=False).head(max(1, len(per) // 10)).sum() / len(fn)), 3)}
    with open(OUT / "error_analysis.json", "w") as f:
        json.dump(MI._no_nan(res), f, indent=2, default=str)
    print("[done]", OUT / "error_analysis.json")


if __name__ == "__main__":
    main()
