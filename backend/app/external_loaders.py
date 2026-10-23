"""
Loaders for openly licensed (CC BY 4.0) T1D datasets, producing the same regular 5-minute grid as
real_data_pipeline (GRID_COLS + steps/kcal). Raw files live under <repo>/external_data/<name>/ and are
treated as untrusted data: only read with pandas, never executed.

  hupa   HUPA-UCM (Hidalgo et al., Data in Brief 2024; Mendeley 10.17632/3hbcscwz44.1)   25 patients
         FreeStyle Libre 2 + Fitbit (steps, kcal, HR) + insulin + carbs.
  uom    T1D-UOM (Univ. of Manchester; Zenodo 10.5281/zenodo.15806142)                  17 patients
         CGM + bolus/basal (pump or long-acting) + carbs + all-day HR/steps (Garmin).
  azt1d  AZT1D (Mayo Clinic Arizona; Mendeley 10.17632/gk9m674wcx.1)                     25 patients
         Dexcom G6 + Tandem Control-IQ (basal rate, bolus) + carbs.

Harmonisation rules (each one is a judgement call, documented here and in the model card):
  * Glucose mg/dL (UOM mmol/L x 18.018). Readings outside 39-450 mg/dL dropped.
  * HUPA's preprocessed glucose/HR were gap-filled by LINEAR INTERPOLATION (up to 11 days in one
    patient). Interpolated points use later readings -> future leakage. Every point that is the middle
    of an exactly collinear 3-point run is set to NaN (costs a few genuine points; removes the leak).
  * HUPA carb_input mixes units across patients: patients averaging < 40 per day log 10-g portions
    ("raciones") and are multiplied by 10; the rest log grams. (Observed gap: 33 vs 57 per day.)
  * Insulin = units delivered per 5 min. Basal rates (U/h) -> rate/12 per 5 min, forward-filled for at
    most 24 h. Long-acting injections (UOM 'L') are spread evenly over the next 24 h (causal).
  * A patient with no carb (or no insulin) records at all gets NaN = "not recorded", not zero.
  * A gap of > 6 h without CGM starts a new segment.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.real_data_pipeline import GRID_COLS, MMOL_TO_MGDL, OPTIONAL_GRID_COLS, ROOT

EXT_DIR = ROOT / "external_data"
FREQ = "5min"
SEGMENT_GAP = pd.Timedelta(hours=6)
G_MIN, G_MAX = 39.0, 450.0
HUPA_PORTION_G = 10.0
HUPA_PORTION_DAILY_CUTOFF = 40.0


# ── shared helpers ────────────────────────────────────────────────────────────

def _mask_collinear(x: np.ndarray) -> np.ndarray:
    """NaN out points lying exactly on a straight line with both neighbours (interpolation signature)."""
    x = x.astype(float).copy()
    mid = np.zeros(len(x), bool)
    if len(x) >= 3:
        d2 = np.abs(x[2:] - 2 * x[1:-1] + x[:-2])
        mid[1:-1] = d2 < 1e-6
    x[mid] = np.nan
    return x


def _bin_events(ts: pd.Series, values: pd.Series, how: str = "sum") -> pd.Series:
    s = pd.Series(pd.to_numeric(values, errors="coerce").to_numpy(), index=ts.dt.floor(FREQ)).dropna()
    return getattr(s.groupby(level=0), how)()


def _assemble(pid: str, source: str, glucose: pd.Series, carbs: Optional[pd.Series], insulin: Optional[pd.Series],
              hr: Optional[pd.Series] = None, steps: Optional[pd.Series] = None, kcal: Optional[pd.Series] = None,
              zero_ok: Tuple[str, ...] = (), oral_med: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Regular 5-min grid from per-bin series (index = bin start), split into segments at CGM gaps > 6 h.
    An event modality that sums to 0 is treated as "not recorded" (NaN) unless listed in `zero_ok`
    (e.g. insulin for a patient whose record says they take no insulin).
    """
    glucose = glucose[(glucose >= G_MIN) & (glucose <= G_MAX)].dropna()
    if glucose.empty:
        return pd.DataFrame(columns=GRID_COLS + OPTIONAL_GRID_COLS)
    idx = pd.date_range(glucose.index.min(), glucose.index.max(), freq=FREQ)
    df = pd.DataFrame(index=idx)
    df["glucose"] = glucose.reindex(idx)
    for name, s, event in (("carbs", carbs, True), ("insulin", insulin, True), ("hr", hr, False),
                           ("steps", steps, False), ("kcal", kcal, False), ("oral_med", oral_med, True)):
        if s is None or s.dropna().empty or (event and float(s.fillna(0).sum()) == 0.0 and name not in zero_ok):
            df[name] = np.nan                                       # modality not recorded for this patient
        else:
            s = s.groupby(level=0).sum() if event else s.groupby(level=0).mean()
            v = s.reindex(idx)
            df[name] = v.fillna(0.0) if event else v                # events: no record in a bin = none
    obs_t = df.index[df["glucose"].notna()]
    gap_start = pd.Series(obs_t).diff() > SEGMENT_GAP
    seg_of_obs = gap_start.cumsum().to_numpy()
    frames = []
    for k in np.unique(seg_of_obs):
        t_in = obs_t[seg_of_obs == k]
        part = df.loc[t_in.min():t_in.max()]
        frames.append(pd.DataFrame({
            "patient_id": f"{source}_{pid}", "source": source, "segment": f"{source}_{pid}_{k}",
            "step": np.arange(len(part)),
            "clock_min": (part.index.hour * 60 + part.index.minute).to_numpy(float),
            **{c: part[c].to_numpy(float) for c in ("glucose", "carbs", "insulin", "hr", "steps", "kcal", "oral_med")},
        }))
    return pd.concat(frames, ignore_index=True)


# ── HUPA-UCM ──────────────────────────────────────────────────────────────────

def load_hupa_grid() -> Tuple[pd.DataFrame, List[Dict]]:
    frames, notes = [], []
    for f in sorted(glob.glob(str(EXT_DIR / "hupa_ucm" / "HUPA*P.csv"))):
        pid = os.path.basename(f)[4:8]
        d = pd.read_csv(f, sep=";")
        t = pd.to_datetime(d["time"]).dt.floor(FREQ)
        g = pd.Series(_mask_collinear(d["glucose"].to_numpy()), index=t)
        hr = pd.Series(_mask_collinear(d["heart_rate"].to_numpy()), index=t)
        days = max((t.max() - t.min()).total_seconds() / 86400, 1e-9)
        raw_carb_day = d["carb_input"].sum() / days
        scale = HUPA_PORTION_G if 0 < raw_carb_day < HUPA_PORTION_DAILY_CUTOFF else 1.0
        carbs = pd.Series(d["carb_input"].to_numpy() * scale, index=t)
        insulin = pd.Series((d["basal_rate"].fillna(0) + d["bolus_volume_delivered"].fillna(0)).to_numpy(), index=t)
        grid = _assemble(pid, "hupa", g.groupby(level=0).mean(), carbs, insulin, hr,
                         pd.Series(d["steps"].to_numpy(float), index=t), pd.Series(d["calories"].to_numpy(float), index=t))
        frames.append(grid)
        notes.append({"patient": f"hupa_{pid}", "carb_unit": "10g-portions" if scale > 1 else "grams",
                      "interp_masked_frac": round(float(np.isnan(g).mean()), 3)})
    return pd.concat(frames, ignore_index=True), notes


# ── T1D-UOM ───────────────────────────────────────────────────────────────────

def _uom_time(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


def load_uom_grid() -> Tuple[pd.DataFrame, List[Dict]]:
    root = glob.glob(str(EXT_DIR / "t1d_uom" / "extracted" / "*"))[0]
    frames, notes = [], []
    for f in sorted(glob.glob(f"{root}/Glucose Data/UoMGlucose*.csv")):
        pid = re.search(r"(\d{4})\.csv$", f).group(1)
        gd = pd.read_csv(f)
        gt = _uom_time(gd["bg_ts"])
        glucose = _bin_events(gt, gd["value"] * MMOL_TO_MGDL, "mean")
        idx = pd.date_range(glucose.index.min(), glucose.index.max(), freq=FREQ)

        insulin = None
        bol_f = f"{root}/Insulin Data/Bolus Data/UoMBolus{pid}.csv"
        bas_f = f"{root}/Insulin Data/Basal Data/UoMBasal{pid}.csv"
        basal_kind = "none"
        if os.path.exists(bol_f) or os.path.exists(bas_f):
            ins = pd.Series(0.0, index=idx)
            if os.path.exists(bol_f):
                b = pd.read_csv(bol_f)
                ins = ins.add(_bin_events(_uom_time(b["bolus_ts"]), b["bolus_dose"]).reindex(idx).fillna(0), fill_value=0)
            if os.path.exists(bas_f):
                b = pd.read_csv(bas_f)
                bt = _uom_time(b["basal_ts"])
                rapid = b["insulin_kind"].astype(str).str.upper().str.startswith("R")
                if rapid.any():                                    # pump: rate in U/h, step function
                    basal_kind = "pump-rate"
                    rate = _bin_events(bt[rapid], b.loc[rapid, "basal_dose"], "last")
                    rate = rate[(rate >= 0) & (rate <= 10)]
                    per5 = rate.reindex(idx).ffill(limit=288) / 12.0
                    ins = ins.add(per5.fillna(0), fill_value=0)
                if (~rapid).any():                                 # long-acting: spread dose over next 24 h
                    basal_kind = "long-acting" if basal_kind == "none" else "mixed"
                    dose = _bin_events(bt[~rapid], b.loc[~rapid, "basal_dose"]).reindex(idx).fillna(0).to_numpy()
                    spread = np.convolve(dose, np.full(288, 1 / 288.0))[: len(idx)]
                    ins = ins.add(pd.Series(spread, index=idx), fill_value=0)
            insulin = ins

        carbs = None
        nut_f = f"{root}/Nutrition Data/UoMNutrition{pid}.csv"
        if os.path.exists(nut_f):
            n = pd.read_csv(nut_f)
            carbs = _bin_events(_uom_time(n["meal_ts"]), n["carbs_g"])

        hr = steps = None
        act_f = f"{root}/Sleep Data/UoMsleep{pid}.csv"                # all-day 5-min HR / steps
        if os.path.exists(act_f):
            a = pd.read_csv(act_f)
            at = _uom_time(a["sleep_ts"])
            hr = _bin_events(at, a["heart_rate"].where(a["heart_rate"] > 0), "mean")
            steps = _bin_events(at, a["step_count"], "sum")
        frames.append(_assemble(pid, "uom", glucose, carbs, insulin, hr, steps, None))
        notes.append({"patient": f"uom_{pid}", "basal": basal_kind, "has_bolus": os.path.exists(bol_f),
                      "has_carbs": carbs is not None, "has_hr_steps": hr is not None})
    return pd.concat(frames, ignore_index=True), notes


# ── AZT1D ─────────────────────────────────────────────────────────────────────

def load_azt1d_grid() -> Tuple[pd.DataFrame, List[Dict]]:
    root = EXT_DIR / "azt1d" / "extracted" / "AZT1D 2025" / "CGM Records"
    frames, notes = [], []
    subject_no = lambda p: int(re.search(r"Subject (\d+)\.csv$", p).group(1))
    files = sorted(glob.glob(str(root / "Subject *" / "Subject *.csv")), key=subject_no)
    for f in files:
        pid = f"{subject_no(f):02d}"
        d = pd.read_csv(f).rename(columns={"Readings (CGM / BGM)": "CGM"})    # Subject 14's header variant
        t = pd.to_datetime(d["EventDateTime"], errors="coerce")
        glucose = _bin_events(t, d["CGM"], "mean")
        idx = pd.date_range(glucose.index.min(), glucose.index.max(), freq=FREQ)
        rate = pd.to_numeric(d["Basal"], errors="coerce")
        bad_basal = int(((rate < 0) | (rate > 10)).sum())
        rate = _bin_events(t, rate.where((rate >= 0) & (rate <= 10)), "last")
        basal = (rate.reindex(idx).ffill(limit=288) / 12.0).fillna(0)
        bolus = _bin_events(t, d["TotalBolusInsulinDelivered"]).reindex(idx).fillna(0)
        carbs = _bin_events(t, d["CarbSize"])
        frames.append(_assemble(pid, "azt1d", glucose, carbs, basal + bolus))
        notes.append({"patient": f"azt1d_{pid}", "dropped_implausible_basal_rows": bad_basal})
    return pd.concat(frames, ignore_index=True), notes


LOADERS = {"hupa": load_hupa_grid, "uom": load_uom_grid, "azt1d": load_azt1d_grid}


def load_external_grids(names=("hupa", "uom", "azt1d")) -> Tuple[pd.DataFrame, Dict[str, List[Dict]]]:
    grids, notes = [], {}
    for n in names:
        g, nt = LOADERS[n]()
        grids.append(g)
        notes[n] = nt
    return pd.concat(grids, ignore_index=True), notes


# ── Shanghai T1DM / T2DM (Zhao et al., Sci Data 2023; figshare 10.6084/m9.figshare.20444397, CC BY 4.0) ──
#   12 T1D + 100 T2D Chinese patients, 15-min CGM, insulin as text ("Novolin R, 6 IU"), and a per-recording
#   clinical summary = REAL static EHR (demographics, diabetes history, complications, drugs, labs).
#   Some people have several recordings (e.g. 1002_0 / 1002_1): patient_id is the PERSON so they never span
#   train and test. 'Hypoglycemia (yes/no)' is not used: it may describe the recording period (outcome leak).

SHANGHAI_LONG_ACTING = ("degludec", "detemir", "glargine", "glarigine")

EHR_FEATURES: List[str] = [
    "ehr_t2d", "ehr_age", "ehr_male", "ehr_bmi", "ehr_duration_y", "ehr_hba1c_pct", "ehr_glycated_albumin",
    "ehr_fpg", "ehr_ppg_2h", "ehr_cpep_fasting", "ehr_cpep_2h", "ehr_tchol", "ehr_tg", "ehr_hdl", "ehr_ldl",
    "ehr_creatinine", "ehr_egfr", "ehr_smoker", "ehr_drinker", "ehr_n_complications", "ehr_hypertension",
    "ehr_hyperlipidemia", "ehr_on_insulin", "ehr_on_long_acting", "ehr_on_metformin", "ehr_on_sulfonylurea",
    "ehr_on_alpha_glucosidase", "ehr_on_dpp4", "ehr_on_sglt2", "ehr_on_glp1", "ehr_on_tzd",
]

_DRUGS = {
    "ehr_on_metformin": ("metformin",), "ehr_on_sulfonylurea": ("gliclazide", "glimepiride", "glipizide", "gliquidone", "glibenclamide"),
    "ehr_on_alpha_glucosidase": ("acarbose", "voglibose", "miglitol"), "ehr_on_dpp4": ("gliptin",),
    "ehr_on_sglt2": ("gliflozin",), "ehr_on_glp1": ("glutide", "exenatide", "lixisenatide"), "ehr_on_tzd": ("glitazone",),
    "ehr_on_insulin": ("insulin", "novolin", "humulin", "gansulin", "scilin"), "ehr_on_long_acting": SHANGHAI_LONG_ACTING,
}


def _shanghai_root() -> str:
    return glob.glob(str(EXT_DIR / "shanghai" / "extracted" / "*"))[0]


def _num(v) -> float:
    return float(pd.to_numeric(v, errors="coerce"))


def shanghai_ehr() -> pd.DataFrame:
    """One row per recording (file stem, e.g. '1002_1_20210521') with EHR_FEATURES."""
    rows = []
    for kind in ("T1DM", "T2DM"):
        s = pd.read_excel(f"{_shanghai_root()}/Shanghai_{kind}_Summary.xlsx")
        s.columns = [re.sub(r"\s+", " ", c).strip().lower() for c in s.columns]
        col = lambda key: next(c for c in s.columns if c.startswith(key))
        for _, r in s.iterrows():
            text = lambda key: str(r[col(key)]).lower() if pd.notna(r[col(key)]) else ""
            agents = text("hypoglycemic agents")
            comp = [text("acute diabetic"), text("diabetic macrovascular"), text("diabetic microvascular")]
            hba1c = _num(r[col("hba1c")])
            rec = {
                "recording": str(r[col("patient number")]).strip(),
                "ehr_t2d": float(kind == "T2DM"), "ehr_age": _num(r[col("age")]),
                "ehr_male": float(_num(r[col("gender")]) == 2), "ehr_bmi": _num(r[col("bmi")]),
                "ehr_duration_y": _num(r[col("duration of diabetes")]),
                "ehr_hba1c_pct": hba1c / 10.929 + 2.15 if not np.isnan(hba1c) else np.nan,   # IFCC mmol/mol -> NGSP %
                "ehr_glycated_albumin": _num(r[col("glycated albumin")]),
                "ehr_fpg": _num(r[col("fasting plasma glucose")]), "ehr_ppg_2h": _num(r[col("2-hour postprandial plasma")]),
                "ehr_cpep_fasting": _num(r[col("fasting c-peptide")]), "ehr_cpep_2h": _num(r[col("2-hour postprandial c-peptide")]),
                "ehr_tchol": _num(r[col("total cholesterol")]), "ehr_tg": _num(r[col("triglyceride")]),
                "ehr_hdl": _num(r[col("high-density")]), "ehr_ldl": _num(r[col("low-density")]),
                "ehr_creatinine": _num(r[col("creatinine")]), "ehr_egfr": _num(r[col("estimated glomerular")]),
                "ehr_smoker": float((_num(r[col("smoking")]) or 0) > 0),
                "ehr_drinker": float(text("alcohol").strip() == "drinker"),
                "ehr_n_complications": float(sum(len([t for t in re.split(r"[,;\n]", c) if t.strip() and t.strip() != "none"]) for c in comp)),
                "ehr_hypertension": float("hypertension" in text("comorbidities")),
                "ehr_hyperlipidemia": float("hyperlipidemia" in text("comorbidities")),
            }
            for feat, keys in _DRUGS.items():
                rec[feat] = float(any(k in agents for k in keys))
            rows.append(rec)
    return pd.DataFrame(rows)


def _parse_doses(cell) -> List[Tuple[str, float]]:
    """'insulin glargine, 12 IU, Humulin 70/30, 8 IU' -> [('insulin glargine', 12.0), ('humulin 70/30', 8.0)]."""
    if pd.isna(cell):
        return []
    s = str(cell).lower().replace("\xa0", " ")
    out, prev = [], 0
    for mt in re.finditer(r"(\d+(?:\.\d+)?)\s*iu", s):
        name = s[prev:mt.start()].strip(" ,;")
        out.append((name, float(mt.group(1))))
        prev = mt.end()
    return out


# Approximate carbohydrate content (g per 100 g as eaten) for the foods in the Shanghai meal logs.
# Standard food-composition values, rounded; the longest matching keyword wins. An APPROXIMATION used only
# to turn "Rice 150 g" into an estimated carb load - documented as such in the model card.
FOOD_CARB_PCT = {
    "coarse grain steamed bread": 45, "steamed bread": 47, "steamed vegetable bun": 40, "steamed pork bun": 35,
    "steamed bun": 40, "bun": 40, "buckwheat bread": 45, "bread": 50, "biscuit": 70, "cake": 50, "cereal": 70,
    "oat": 60, "congee": 10, "porridge": 10, "fried rice": 30, "rice": 28, "coarse grain": 25, "millet": 25,
    "buckwheat noodles": 25, "noodle": 25, "rice noodle": 25, "vermicelli": 25, "wonton": 25, "dumpling": 28,
    "pancake": 45, "sweet potato": 20, "potato": 17, "taro": 18, "corn": 20, "yam": 15, "pumpkin": 7,
    "lotus root": 15, "chestnut": 40, "bean curd roll": 5, "tofu": 3, "bean curd": 3,
    "apple": 14, "orange": 12, "banana": 23, "pear": 13, "grape": 17, "watermelon": 8, "kiwi": 15, "strawberry": 8,
    "peach": 11, "mango": 15, "pineapple": 13, "dragon fruit": 13, "persimmon": 18, "cherry tomato": 6,
    "yogurt": 12, "soybean milk": 3, "milk": 5, "beer": 4, "juice": 11, "braised pork": 5, "sugar": 100,
}
DEFAULT_CARB_PCT = 4                    # vegetables, eggs, meat, fish and anything unmatched
_FOOD_KEYS = sorted(FOOD_CARB_PCT, key=len, reverse=True)


def estimate_meal_carbs(text) -> Optional[float]:
    """'Rice 150 g\nSpinach 150 g' -> 48.0 g carbs (approximate). None if no meal / not available."""
    if pd.isna(text) or "not available" in str(text).lower():
        return None
    total, any_item = 0.0, False
    for line in re.split(r"[\n;]", str(text).lower()):
        m = re.search(r"(.*?)(\d+(?:\.\d+)?)\s*(g|ml)\b", line)
        if not m:
            continue
        any_item = True
        name, grams = m.group(1), float(m.group(2))
        pct = next((FOOD_CARB_PCT[k] for k in _FOOD_KEYS if k in name), DEFAULT_CARB_PCT)
        total += grams * pct / 100.0
    return round(total, 1) if any_item else None


def _count_drugs(cell) -> float:
    """'metformin 0.5 g, acarbose 50 mg' -> 2 oral-drug doses at that time."""
    if pd.isna(cell) or not str(cell).strip():
        return 0.0
    return float(len([x for x in re.split(r"[,;\n]", str(cell)) if re.search(r"[a-z]", x.lower())]))


def _load_shanghai(kind: str, source: str, meals: bool = False) -> Tuple[pd.DataFrame, List[Dict]]:
    """meals=True (v8+) also turns the free-text meal log into estimated carbs and adds oral-drug dose events."""
    frames, notes = [], []
    ehr_all = shanghai_ehr().set_index("recording")
    on_insulin = ehr_all["ehr_on_insulin"]
    oral_cols = ["ehr_on_metformin", "ehr_on_sulfonylurea", "ehr_on_alpha_glucosidase", "ehr_on_dpp4",
                 "ehr_on_sglt2", "ehr_on_tzd", "ehr_on_glp1"]
    on_oral = ehr_all[oral_cols].max(axis=1)
    files = sorted(f for f in glob.glob(f"{_shanghai_root()}/Shanghai_{kind}/*")
                   if not os.path.basename(f).startswith("~$"))
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        person = stem.split("_")[0]
        d = pd.read_excel(f)
        d.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in d.columns]
        for want in ("Insulin dose - s.c.", "Insulin dose - i.v.", "CSII - bolus insulin (Novolin R, IU)",
                     "CSII - basal insulin (Novolin R, IU / H)"):                 # not every file has every column
            hit = [c for c in d.columns if c.lower().startswith(want.lower()[:22])]
            d[want] = d[hit[0]] if hit else np.nan
        t = pd.to_datetime(d["Date"], errors="coerce")
        glucose = _bin_events(t, d[next(c for c in d.columns if c.lower().startswith("cgm"))], "mean")
        if glucose.empty:
            continue
        idx = pd.date_range(glucose.index.min(), glucose.index.max(), freq=FREQ)
        bolus = pd.Series(0.0, index=idx)
        long_dose = pd.Series(0.0, index=idx)
        n_doses = 0
        for col in ("Insulin dose - s.c.", "Insulin dose - i.v."):
            for ti, cell in zip(t.dt.floor(FREQ), d[col]):
                for name, dose in _parse_doses(cell):
                    if ti not in bolus.index:
                        continue
                    n_doses += 1
                    if any(k in name for k in SHANGHAI_LONG_ACTING):
                        long_dose[ti] += dose
                    else:
                        bolus[ti] += dose
        csii_b = _bin_events(t, d["CSII - bolus insulin (Novolin R, IU)"]).reindex(idx).fillna(0)
        rate = _bin_events(t, d["CSII - basal insulin (Novolin R, IU / H)"], "last")
        rate = rate[(rate >= 0) & (rate <= 10)]
        basal = (rate.reindex(idx).ffill(limit=288) / 12.0).fillna(0)
        spread = np.convolve(long_dose.to_numpy(), np.full(288, 1 / 288.0))[: len(idx)]
        insulin = bolus + csii_b + basal + pd.Series(spread, index=idx)
        no_insulin = on_insulin.get(stem, np.nan) == 0                   # oral-drug-only patient: 0 is real
        carbs = oral = None                                                # v7: meals are free text -> unknown
        if meals:
            diet = next((c for c in d.columns if c.lower().startswith("dietary intake")), None)
            est = d[diet].map(estimate_meal_carbs) if diet else pd.Series(dtype=float)
            carbs = _bin_events(t, est)
            agents = next((c for c in d.columns if c.lower().startswith("non-insulin")), None)
            oral = _bin_events(t, d[agents].map(_count_drugs)) if agents else None
        zero_ok = (("insulin",) if no_insulin else ()) + (("oral_med",) if on_oral.get(stem, np.nan) == 0 else ())
        grid = _assemble(person, source, glucose, carbs, insulin, zero_ok=zero_ok, oral_med=oral)
        grid["segment"] = grid["segment"].str.replace(f"{source}_{person}_", f"{source}_{stem}_", regex=False)
        frames.append(grid)
        notes.append({"recording": stem, "person": f"{source}_{person}", "insulin_events": n_doses,
                      "days": round(len(grid) / 288, 1)})
    return pd.concat(frames, ignore_index=True), notes


def load_shanghai_t1_grid() -> Tuple[pd.DataFrame, List[Dict]]:
    return _load_shanghai("T1DM", "sh_t1")


def load_shanghai_t2_grid() -> Tuple[pd.DataFrame, List[Dict]]:
    return _load_shanghai("T2DM", "sh_t2")


def load_shanghai_t1_meals_grid() -> Tuple[pd.DataFrame, List[Dict]]:
    return _load_shanghai("T1DM", "sh_t1", meals=True)


def load_shanghai_t2_meals_grid() -> Tuple[pd.DataFrame, List[Dict]]:
    return _load_shanghai("T2DM", "sh_t2", meals=True)


def recording_of_segment(segment: pd.Series) -> pd.Series:
    """'sh_t2_2001_1_20201117_0' -> '2001_1_20201117' (NaN for non-Shanghai segments)."""
    m = segment.str.extract(r"^sh_t[12]_(\d+_\d+_\d+)_\d+$")[0]
    return m


LOADERS.update({"sh_t1": load_shanghai_t1_grid, "sh_t2": load_shanghai_t2_grid})
