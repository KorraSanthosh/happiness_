"""
Real-data pipeline: OhioT1DM + BrisT1D on one regular 5-minute grid.

Why this exists (replaces the ohio_* / brist1d_loader path for new work):
  * BrisT1D rows are 6-hour lookback windows that overlap heavily (rows are 5-15 min apart).
    They are stitched back into ONE real glucose timeline per patient instead of being treated
    as independent series (the old loader produced ~12.7M duplicated rows with fake timestamps).
  * Missing CGM stays NaN on a regular grid, so "24 steps" always means 2 hours (the old loader
    dropped missing rows, which silently stretched the horizon across gaps).
  * No imputation from future data (the old loaders used per-patient medians over the full series).
  * Every feature at step t uses only data at steps <= t; labels use steps t+1..t+24 only.

Harmonised signals (both sources): glucose mg/dL, carbs g, insulin U, heart rate bpm, clock minute-of-day.
Ohio insulin = bolus + basal_rate/12 per 5 min (BrisT1D "insulin" is total dose per 5 min).
Source-specific signals (Ohio GSR/basal, BrisT1D steps/cals/activity) are deliberately not used.

Resolution: BrisT1D patients p01/p05/p06 have CGM only every 15 min; everyone else every 5 min.
All sources are therefore modelled on a common 15-MINUTE grid (`to_15min`): features use the last
reading in each bin, the label uses the bin maximum.

Target: 1 if max(glucose over the next 8 bins = t+15 .. t+120 min) > 180 mg/dL
(v3.1's definition, on 15-min bins instead of 5-min steps).
"""
from __future__ import annotations

import glob
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OHIO_BASE = ROOT / "ohiot1dm_raw" / "ohiot1dm-glucose-dataset-main" / "Ohio Data"
BRIST_DIR = ROOT / "brist1d_raw"

RAW_STEP_MIN = 5              # grid produced by the loaders
BIN = 3                       # raw steps per modelling step
STEP_MIN = 15                 # modelling step (minutes)
HORIZON_STEPS = 8             # 2 h in 15-min steps
SPIKE_MGDL = 180.0
MMOL_TO_MGDL = 18.018
MIN_FUTURE_OBS = 6            # of the 8 future bins, at least 6 must hold a real CGM reading
MIN_HISTORY_STEPS = 4         # 1 h of history inside the segment before a row is usable
FILL_LIMIT = 2                # causal forward-fill of CGM/HR up to 30 min
BRIST_MAX_GAP_MIN = 30        # larger jump between BrisT1D rows starts a new segment
BRIST_LOOKBACK = 72           # steps per BrisT1D window

IOB_HALF_LIFE_MIN = 90.0
COB_HALF_LIFE_MIN = 60.0
BIG_DOSE_U = 0.5              # a single 5-min insulin dose >= this counts as a bolus

GRID_COLS = ["patient_id", "source", "segment", "step", "clock_min",
             "glucose", "carbs", "insulin", "hr"]
OPTIONAL_GRID_COLS = ["steps", "kcal", "oral_med"]   # wearable activity / oral-drug doses; NaN (= unknown) if absent
# A whole-segment NaN in `carbs` / `insulin` means "not recorded" (features become NaN), not "zero".


# ═════════════════════════════════════════════════════════════════════════════
# Loading -> regular 5-minute grid
# ═════════════════════════════════════════════════════════════════════════════

def load_ohio_grid(years=("2018", "2020")) -> pd.DataFrame:
    """One segment per Ohio CSV (train/test file of a patient); files are already regular 5-min grids."""
    frames: List[pd.DataFrame] = []
    for year in years:
        for split in ("train", "test"):
            pattern = str(OHIO_BASE / f"Ohio{year}_processed" / split / "*.csv")
            for f in sorted(glob.glob(pattern)):
                pid = os.path.basename(f).split("-")[0]
                raw = pd.read_csv(f)
                n = len(raw)
                step = np.arange(n)
                clock0 = round(float(raw["5minute_intervals_timestamp"].iloc[0]) * 5 / RAW_STEP_MIN) * RAW_STEP_MIN
                basal = raw["basal"].ffill().fillna(0.0)            # causal; U/hour
                frames.append(pd.DataFrame({
                    "patient_id": f"ohio_{pid}",
                    "source": "ohio",
                    "segment": f"ohio_{pid}_{year}_{split}",
                    "step": step,
                    "clock_min": (clock0 + RAW_STEP_MIN * step) % 1440,
                    "glucose": raw["cbg"].where(raw["missing_cbg"] == 0).to_numpy(float),
                    "carbs": raw["carbInput"].fillna(0.0).to_numpy(float),
                    "insulin": (raw["bolus"].fillna(0.0) + basal / 12.0).to_numpy(float),
                    "hr": raw["hr"].to_numpy(float),
                }))
    if not frames:
        raise FileNotFoundError(f"No OhioT1DM CSVs under {OHIO_BASE}")
    return pd.concat(frames, ignore_index=True)[GRID_COLS]


def _lag_steps(col: str) -> int:
    h, m = re.search(r"(\d+):(\d+)$", col).groups()
    return (int(h) * 60 + int(m)) // RAW_STEP_MIN


def _sorted_lag_cols(columns, prefix: str) -> Tuple[List[str], np.ndarray]:
    cols = [c for c in columns if re.fullmatch(rf"{prefix}-\d+:\d+", c)]
    cols.sort(key=_lag_steps)                       # lag 0 (now) ... lag 71 (oldest)
    return cols, np.array([_lag_steps(c) for c in cols])


def stitch_brist_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Convert BrisT1D wide windows into regular grids. Returns (grid, stats).
    Each row's window is written onto an absolute grid (row time + lag); overlapping windows agree
    on the real data (checked via stats['conflicts']). bg+1:00 of a row is also written (it is the
    glucose 12 steps after that row) but only ever used as history/labels for LATER steps.
    """
    bg_cols, offs = _sorted_lag_cols(df.columns, "bg")
    sig_cols = {"glucose": bg_cols}
    for name, prefix in (("insulin", "insulin"), ("carbs", "carbs"), ("hr", "hr"), ("steps", "steps"), ("kcal", "cals")):
        cols, o = _sorted_lag_cols(df.columns, prefix)
        if name in OPTIONAL_GRID_COLS and not cols:
            continue
        assert np.array_equal(o, offs), f"lag layout mismatch for {prefix}"
        sig_cols[name] = cols

    stats = {"conflicts": 0, "writes": 0, "segments": 0}
    frames: List[pd.DataFrame] = []
    for pid, grp in df.groupby("p_num", sort=True):
        grp = grp.reset_index(drop=True)
        t = pd.to_timedelta(grp["time"]).dt.total_seconds().to_numpy() / 60.0
        delta = np.diff(t, prepend=t[0])
        delta = np.where(delta < 0, delta + 1440, delta)
        new_seg = (delta > BRIST_MAX_GAP_MIN) | (delta == 0)
        new_seg[0] = True
        seg_no = np.cumsum(new_seg) - 1

        for k in np.unique(seg_no):
            rows = np.where(seg_no == k)[0]
            d = delta[rows].copy()
            d[0] = 0.0
            g_i = np.round(np.cumsum(d) / RAW_STEP_MIN).astype(int)          # absolute grid index of each row
            lo = g_i.min() - (BRIST_LOOKBACK - 1)
            hi = g_i.max() + 12
            L = hi - lo + 1
            sub = grp.iloc[rows]
            grids = {}
            for name, cols in sig_cols.items():
                arr = np.full(L, np.nan)
                vals = sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)   # (n, 72)
                if name == "glucose":
                    vals = vals * MMOL_TO_MGDL
                idx = (g_i[:, None] - offs[None, :]) - lo
                m = ~np.isnan(vals)
                arr[idx[m]] = vals[m]
                if name == "glucose":
                    stats["writes"] += int(m.sum())
                    stats["conflicts"] += int((~np.isclose(arr[idx[m]], vals[m])).sum())
                    tgt = pd.to_numeric(sub["bg+1:00"], errors="coerce").to_numpy(float) * MMOL_TO_MGDL \
                        if "bg+1:00" in sub.columns else np.full(len(sub), np.nan)
                    tm = ~np.isnan(tgt)
                    arr[(g_i + 12 - lo)[tm]] = np.where(np.isnan(arr[(g_i + 12 - lo)[tm]]), tgt[tm],
                                                        arr[(g_i + 12 - lo)[tm]])
                grids[name] = arr

            ok = np.where(~np.isnan(grids["glucose"]))[0]
            if len(ok) == 0:
                continue
            a, b = ok[0], ok[-1] + 1                                    # trim empty edges
            n = b - a
            t0 = int(round(t[rows[0]] / RAW_STEP_MIN)) * RAW_STEP_MIN
            step = np.arange(n)
            stats["segments"] += 1
            frames.append(pd.DataFrame({
                "patient_id": f"brist_{pid}",
                "source": "brist",
                "segment": f"brist_{pid}_{k}",
                "step": step,
                "clock_min": (t0 + RAW_STEP_MIN * (lo + a + step)) % 1440,
                "glucose": grids["glucose"][a:b],
                "carbs": np.nan_to_num(grids["carbs"][a:b], nan=0.0),
                "insulin": np.nan_to_num(grids["insulin"][a:b], nan=0.0),
                "hr": grids["hr"][a:b],
                **{c: grids[c][a:b] for c in OPTIONAL_GRID_COLS if c in grids},   # NaN = watch not worn
            }))
    cols = GRID_COLS + [c for c in OPTIONAL_GRID_COLS if c in sig_cols]
    return pd.concat(frames, ignore_index=True)[cols], stats


def load_brist_grid(split: str = "train") -> Tuple[pd.DataFrame, Dict[str, float]]:
    path = BRIST_DIR / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (Kaggle BrisT1D data goes in brist1d_raw/)")
    return stitch_brist_frame(pd.read_csv(path, low_memory=False))


def load_all_grids() -> Tuple[pd.DataFrame, Dict[str, float]]:
    ohio = load_ohio_grid()
    brist, stats = load_brist_grid("train")
    return pd.concat([ohio, brist], ignore_index=True), stats


# ═════════════════════════════════════════════════════════════════════════════
# 5-min grid -> common 15-min modelling grid
# ═════════════════════════════════════════════════════════════════════════════

def _bin_last(a: np.ndarray) -> np.ndarray:
    """Last non-NaN value of each row of a (n_bins, BIN) array; NaN if the whole bin is NaN."""
    out = np.full(a.shape[0], np.nan)
    for j in range(a.shape[1]):                       # later columns overwrite earlier ones
        col = a[:, j]
        out = np.where(np.isnan(col), out, col)
    return out


def to_15min(grid: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate each segment into 15-minute bins (3 raw steps; a trailing partial bin is dropped).
    glucose      last real reading in the bin   (feature input, point-in-time)
    glucose_max  max real reading in the bin    (label only: catches brief excursions in 5-min data)
    carbs/insulin summed, insulin_peak = largest single raw dose, hr = mean, clock_min = bin end.
    """
    parts = []
    for seg_id, seg in grid.groupby("segment", sort=False):
        seg = seg.sort_values("step")
        nb = len(seg) // BIN
        if nb == 0:
            continue
        cut = nb * BIN
        shape = (nb, BIN)
        g = seg["glucose"].to_numpy(float)[:cut].reshape(shape)
        hr = seg["hr"].to_numpy(float)[:cut].reshape(shape)
        ins = seg["insulin"].to_numpy(float)[:cut].reshape(shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN bins are expected
            gmax = np.where(np.isnan(g).all(axis=1), np.nan, np.nanmax(np.where(np.isnan(g), -np.inf, g), axis=1))
            hrm = np.where(np.isnan(hr).all(axis=1), np.nan, np.nanmean(hr, axis=1))
        activity = {}
        for c in OPTIONAL_GRID_COLS:                            # sums; an all-NaN bin stays unknown
            a = seg[c].to_numpy(float)[:cut].reshape(shape) if c in seg.columns else np.full(shape, np.nan)
            activity[c] = np.where(np.isnan(a).all(axis=1), np.nan, np.nansum(a, axis=1))
        parts.append(pd.DataFrame({
            "patient_id": seg["patient_id"].iloc[0],
            "source": seg["source"].iloc[0],
            "segment": seg_id,
            "step": np.arange(nb),
            "clock_min": seg["clock_min"].to_numpy(float)[:cut].reshape(shape)[:, -1],
            "glucose": _bin_last(g),
            "glucose_max": gmax,
            "carbs": seg["carbs"].to_numpy(float)[:cut].reshape(shape).sum(axis=1),
            "insulin": ins.sum(axis=1),
            "insulin_peak": ins.max(axis=1),
            "hr": hrm,
            **activity,
        }))
    return pd.concat(parts, ignore_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# Feature engineering (past-only; one modelling step = 15 min)
# ═════════════════════════════════════════════════════════════════════════════

BASE_FEATURES: List[str] = [        # comparable in spirit to the old Ohio v3.1 set (minus GSR/basal)
    "glucose", "g_lag_15m", "g_lag_30m", "g_lag_1h",
    "g_d30m", "g_d1h", "g_d2h", "g_roc_1h",
    "g_mean_1h", "g_mean_2h", "g_std_1h", "g_std_2h", "g_min_2h", "g_max_2h",
    "g_above_180", "g_in_range",
    "carbs_now", "carbs_30m", "carbs_1h", "carbs_2h",
    "insulin_now", "insulin_1h", "insulin_2h",
    "hr_mean_1h", "hr_std_1h", "hr_max_1h",
    "tod_sin", "tod_cos", "hour",
]

ENGINEERED_FEATURES: List[str] = BASE_FEATURES + [
    # glucose dynamics
    "g_d15m", "g_slope_30m", "g_slope_1h", "g_accel", "g_ewm_short", "g_ewm_long",
    "g_ewm_gap", "g_mean_30m", "g_mean_4h", "g_std_4h", "g_max_4h", "g_min_4h",
    "g_range_2h", "g_cv_2h", "g_minus_mean_4h",
    "g_frac_above_2h", "g_frac_below_70_2h", "g_frac_in_range_2h",
    "g_since_above_180", "g_dist_180",
    "g_proj_30m", "g_proj_1h", "g_proj_2h", "g_proj_2h_gap",
    "g_obs_frac_2h",
    # meals / insulin
    "carbs_4h", "cob", "since_meal", "last_meal_size",
    "insulin_4h", "iob", "insulin_max_1h", "since_bolus", "cob_per_iob",
    # heart rate
    "hr_now", "hr_vs_4h",
    # time
    "is_night",
]

ACTIVITY_FEATURES: List[str] = [     # wearable steps / energy; NaN for sources without a watch (Ohio)
    "steps_30m", "steps_1h", "steps_2h", "kcal_1h", "kcal_2h", "since_active",
]

PERSONAL_FEATURES: List[str] = [     # this patient's own recent history ("usual pattern"), past-only
    "p_mean_24h", "p_std_24h", "p_frac_above_24h", "g_minus_p_mean_24h",
    "p_same_time_mean_7d", "g_minus_same_time", "p_same_time_spike_7d", "p_hist_days",
]

V6_FEATURES: List[str] = ENGINEERED_FEATURES + ACTIVITY_FEATURES + PERSONAL_FEATURES

HISTORY_FEATURES: List[str] = [      # v9 C1: longer personal history (past-only; addresses patient heterogeneity)
    "p_same_time_spike_14d", "p_hours_since_high", "p_frac_above_72h", "p_high_onsets_per_day_72h",
]

DYNAMICS_FEATURES: List[str] = [     # v9 C3: multi-window dynamics (past-only)
    "g_slope_2h", "g_max_rise_1h", "g_hours_to_180", "p_cv_72h", "g_minus_p_mean_72h",
]

ORAL_MED_FEATURES: List[str] = [     # Type 2 tablets (metformin, acarbose, ...); NaN where not recorded
    "oral_med_2h", "oral_med_4h", "since_oral_med",
]

EXTRA_HORIZONS = {"y_30m": 2, "y_60m": 4}  # label columns for 30 / 60 min (y = 2 h, unchanged)

W30, W1H, W2H, W4H = 2, 4, 8, 16          # window lengths in 15-min steps
DAY = 96                                  # 15-min steps per day
ACTIVE_STEPS_15M = 500                    # >= 500 steps in 15 min counts as an active bout
STEPS_PER_HOUR = 4


def _causal_conv(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.convolve(x, kernel)[: len(x)]            # y[t] = sum_k kernel[k] * x[t-k]


def _since_event(flag: np.ndarray, cap: int) -> np.ndarray:
    n = len(flag)
    idx = np.where(flag, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    return np.where(last >= 0, np.minimum(np.arange(n) - last, cap), cap).astype(float)


def _slope_kernel(w: int) -> np.ndarray:
    k = np.arange(w)                                   # lag k = 0 is the newest point
    wts = (w - 1) / 2.0 - k
    return wts / np.sum(wts ** 2)                      # OLS slope per step


def _segment_features(seg: pd.DataFrame) -> pd.DataFrame:
    """Features + label for one contiguous 15-min segment. Row order == time order."""
    n = len(seg)
    glucose = pd.Series(seg["glucose"].to_numpy(float))
    carbs = seg["carbs"].to_numpy(float)
    insulin = seg["insulin"].to_numpy(float)
    peak = seg["insulin_peak"].to_numpy(float)
    g = glucose.ffill(limit=FILL_LIMIT)                # causal fill, features only
    hr = pd.Series(seg["hr"].to_numpy(float)).ffill(limit=FILL_LIMIT)

    f: Dict[str, np.ndarray] = {}

    def roll(s: pd.Series, w: int, fn: str, minp: int | None = None) -> np.ndarray:
        return getattr(s.rolling(w, min_periods=minp or max(2, w // 2)), fn)().to_numpy()

    # ── glucose level / lags / deltas
    f["glucose"] = g.to_numpy()
    f["g_lag_15m"], f["g_lag_30m"], f["g_lag_1h"] = (g.shift(k).to_numpy() for k in (1, 2, 4))
    f["g_d15m"] = (g - g.shift(1)).to_numpy()
    f["g_d30m"] = (g - g.shift(2)).to_numpy()
    f["g_d1h"] = (g - g.shift(4)).to_numpy()
    f["g_d2h"] = (g - g.shift(8)).to_numpy()
    f["g_roc_1h"] = f["g_d1h"]                          # mg/dL per hour

    # ── rolling stats (windows end at t, inclusive)
    f["g_mean_30m"] = roll(g, W30, "mean", 1)
    f["g_mean_1h"], f["g_mean_2h"], f["g_mean_4h"] = roll(g, W1H, "mean"), roll(g, W2H, "mean"), roll(g, W4H, "mean")
    f["g_std_1h"], f["g_std_2h"], f["g_std_4h"] = roll(g, W1H, "std"), roll(g, W2H, "std"), roll(g, W4H, "std")
    f["g_min_2h"], f["g_max_2h"] = roll(g, W2H, "min"), roll(g, W2H, "max")
    f["g_min_4h"], f["g_max_4h"] = roll(g, W4H, "min"), roll(g, W4H, "max")
    f["g_range_2h"] = f["g_max_2h"] - f["g_min_2h"]
    f["g_cv_2h"] = f["g_std_2h"] / np.where(f["g_mean_2h"] > 0, f["g_mean_2h"], np.nan)
    f["g_minus_mean_4h"] = f["glucose"] - f["g_mean_4h"]
    f["g_ewm_short"] = g.ewm(span=2, adjust=False, ignore_na=True).mean().to_numpy()
    f["g_ewm_long"] = g.ewm(span=8, adjust=False, ignore_na=True).mean().to_numpy()
    f["g_ewm_gap"] = f["g_ewm_short"] - f["g_ewm_long"]

    # ── trend (OLS slope, mg/dL per hour) and acceleration
    gv = g.to_numpy()
    for name, w in (("g_slope_30m", 3), ("g_slope_1h", 5)):          # 3 / 5 points = 30 / 60 min span
        s = _causal_conv(gv, _slope_kernel(w)) * STEPS_PER_HOUR
        s[: w - 1] = np.nan
        f[name] = s
    f["g_accel"] = f["g_slope_30m"] - pd.Series(f["g_slope_30m"]).shift(2).to_numpy()
    f["g_proj_30m"] = f["glucose"] + 0.5 * f["g_slope_30m"]
    f["g_proj_1h"] = f["glucose"] + 1.0 * f["g_slope_1h"]
    f["g_proj_2h"] = f["glucose"] + 2.0 * f["g_slope_1h"]
    f["g_proj_2h_gap"] = f["g_proj_2h"] - SPIKE_MGDL

    # ── threshold exposure
    known = g.notna()
    above = (g > SPIKE_MGDL).astype(float).where(known)
    below = (g < 70).astype(float).where(known)
    inr = ((g >= 70) & (g <= SPIKE_MGDL)).astype(float).where(known)
    f["g_above_180"], f["g_in_range"] = above.to_numpy(), inr.to_numpy()
    f["g_frac_above_2h"] = roll(above, W2H, "mean")
    f["g_frac_below_70_2h"] = roll(below, W2H, "mean")
    f["g_frac_in_range_2h"] = roll(inr, W2H, "mean")
    f["g_since_above_180"] = _since_event(above.fillna(0).to_numpy() > 0, W4H)
    f["g_dist_180"] = SPIKE_MGDL - f["glucose"]
    f["g_obs_frac_2h"] = roll(glucose.notna().astype(float), W2H, "mean", 1)   # CGM coverage

    # ── carbohydrates
    cs = pd.Series(carbs)
    f["carbs_now"] = carbs
    for name, w in (("carbs_30m", W30), ("carbs_1h", W1H), ("carbs_2h", W2H), ("carbs_4h", W4H)):
        f[name] = cs.rolling(w, min_periods=1).sum().to_numpy()
    f["cob"] = _causal_conv(carbs, 0.5 ** (np.arange(24) * STEP_MIN / COB_HALF_LIFE_MIN))   # 6 h kernel
    meal = carbs > 0
    f["since_meal"] = _since_event(meal, 24)
    f["last_meal_size"] = pd.Series(np.where(meal, carbs, np.nan)).ffill().fillna(0.0).to_numpy()

    # ── insulin
    ins = pd.Series(insulin)
    f["insulin_now"] = insulin
    for name, w in (("insulin_1h", W1H), ("insulin_2h", W2H), ("insulin_4h", W4H)):
        f[name] = ins.rolling(w, min_periods=1).sum().to_numpy()
    f["iob"] = _causal_conv(insulin, 0.5 ** (np.arange(16) * STEP_MIN / IOB_HALF_LIFE_MIN))  # 4 h kernel
    f["insulin_max_1h"] = pd.Series(peak).rolling(W1H, min_periods=1).max().to_numpy()
    f["since_bolus"] = _since_event(peak >= BIG_DOSE_U, 24)
    f["cob_per_iob"] = f["cob"] / (f["iob"] + 0.1)
    # a segment with no carb / insulin records at all is "unknown", not "nothing eaten / injected"
    if np.isnan(carbs).all():
        for k in ("carbs_now", "carbs_30m", "carbs_1h", "carbs_2h", "carbs_4h", "cob", "since_meal", "last_meal_size", "cob_per_iob"):
            f[k] = np.full(n, np.nan)
    if np.isnan(insulin).all():
        for k in ("insulin_now", "insulin_1h", "insulin_2h", "insulin_4h", "iob", "insulin_max_1h", "since_bolus", "cob_per_iob"):
            f[k] = np.full(n, np.nan)

    # ── heart rate (NaN-native: HistGradientBoosting handles missing values)
    f["hr_now"] = hr.to_numpy()
    f["hr_mean_1h"], f["hr_std_1h"], f["hr_max_1h"] = roll(hr, W1H, "mean", 2), roll(hr, W1H, "std", 2), roll(hr, W1H, "max", 2)
    f["hr_vs_4h"] = f["hr_now"] - roll(hr, W4H, "mean", 4)

    # ── time of day (real clock minute)
    clock = seg["clock_min"].to_numpy(float)
    f["tod_sin"], f["tod_cos"] = np.sin(2 * np.pi * clock / 1440), np.cos(2 * np.pi * clock / 1440)
    f["hour"] = np.floor(clock / 60)
    f["is_night"] = ((f["hour"] >= 22) | (f["hour"] < 6)).astype(float)

    # ── wearable activity (sums ignore unknown bins; all-unknown window -> NaN)
    for col, prefix in (("steps", "steps"), ("kcal", "kcal")):
        s = pd.Series(seg[col].to_numpy(float) if col in seg.columns else np.full(n, np.nan))
        for name, w in ((f"{prefix}_30m", W30), (f"{prefix}_1h", W1H), (f"{prefix}_2h", W2H)):
            if name in ACTIVITY_FEATURES:
                f[name] = s.rolling(w, min_periods=1).sum().to_numpy()
        if col == "steps":
            f["since_active"] = (np.full(n, np.nan) if s.isna().all()
                                 else _since_event((s >= ACTIVE_STEPS_15M).to_numpy(), 24))

    # ── oral glucose-lowering drugs (dose events)
    om = pd.Series(seg["oral_med"].to_numpy(float) if "oral_med" in seg.columns else np.full(n, np.nan))
    if om.isna().all():
        f["oral_med_2h"] = f["oral_med_4h"] = f["since_oral_med"] = np.full(n, np.nan)
    else:
        f["oral_med_2h"] = om.rolling(W2H, min_periods=1).sum().to_numpy()
        f["oral_med_4h"] = om.rolling(W4H, min_periods=1).sum().to_numpy()
        f["since_oral_med"] = _since_event((om.fillna(0) > 0).to_numpy(), 24)

    out = pd.DataFrame(f)

    # ── label: any real CGM reading > 180 in the next 8 bins; needs the full horizon inside the
    #    segment and >= MIN_FUTURE_OBS observed bins (coverage rule is applied to positives AND negatives)
    gmax = pd.Series(seg["glucose_max"].to_numpy(float))
    nxt = gmax.shift(-1).iloc[::-1]
    fut_max = nxt.rolling(HORIZON_STEPS, min_periods=1).max().iloc[::-1].to_numpy()
    fut_cnt = nxt.rolling(HORIZON_STEPS, min_periods=1).count().iloc[::-1].to_numpy()
    full = (np.arange(n) + HORIZON_STEPS) <= (n - 1)
    label_ok = full & (fut_cnt >= MIN_FUTURE_OBS)
    out["y"] = np.where(label_ok, (fut_max > SPIKE_MGDL).astype(float), np.nan)
    out["usable"] = glucose.notna().to_numpy() & label_ok & (np.arange(n) >= MIN_HISTORY_STEPS)
    # shorter horizons (same rule: full window inside the segment, >= 75 % of its bins observed)
    for col, h in EXTRA_HORIZONS.items():
        fm = nxt.rolling(h, min_periods=1).max().iloc[::-1].to_numpy()
        fc = nxt.rolling(h, min_periods=1).count().iloc[::-1].to_numpy()
        ok = ((np.arange(n) + h) <= (n - 1)) & (fc >= int(np.ceil(0.75 * h)))
        out[col] = np.where(ok, (fm > SPIKE_MGDL).astype(float), np.nan)

    # ── personal history ("this patient's usual pattern"), strictly from earlier data:
    #    same-time-of-day values come from exactly k days earlier (k = 1..7); a label k>=1 days back
    #    only looked HORIZON_STEPS (2 h) ahead, so it was fully known before t.
    out["p_mean_24h"] = roll(g, DAY, "mean", DAY // 2)
    out["p_std_24h"] = roll(g, DAY, "std", DAY // 2)
    out["p_frac_above_24h"] = roll(above, DAY, "mean", DAY // 2)
    out["g_minus_p_mean_24h"] = out["glucose"] - out["p_mean_24h"]
    y_series = pd.Series(out["y"].to_numpy())
    same_g = np.column_stack([g.shift(DAY * k).to_numpy() for k in range(1, 8)])
    same_y = np.column_stack([y_series.shift(DAY * k).to_numpy() for k in range(1, 8)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["p_same_time_mean_7d"] = np.nanmean(same_g, axis=1)
        out["p_same_time_spike_7d"] = np.nanmean(same_y, axis=1)
    out["g_minus_same_time"] = out["glucose"] - out["p_same_time_mean_7d"]
    out["p_hist_days"] = np.minimum(np.arange(n), 7 * DAY) / DAY

    # ── v9 C1: longer personal history. Labels are only used >= 1 day back (known 2 h after they start).
    same_y14 = np.column_stack([y_series.shift(DAY * k).to_numpy() for k in range(1, 15)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["p_same_time_spike_14d"] = np.nanmean(same_y14, axis=1)
    high = (g > SPIKE_MGDL).to_numpy()
    out["p_hours_since_high"] = _since_event(high, DAY) / STEPS_PER_HOUR          # capped at 24 h
    out["p_frac_above_72h"] = roll(above, 3 * DAY, "mean", DAY)
    onset = pd.Series((high & ~np.r_[False, high[:-1]]).astype(float)).where(known)
    out["p_high_onsets_per_day_72h"] = (onset.rolling(3 * DAY, min_periods=DAY).sum()
                                        / (onset.rolling(3 * DAY, min_periods=DAY).count() / DAY)).to_numpy()

    # ── v9 C3: multi-window dynamics
    s2 = _causal_conv(gv, _slope_kernel(9)) * STEPS_PER_HOUR                     # OLS slope over 2 h (9 points)
    s2[:8] = np.nan
    out["g_slope_2h"] = s2
    out["g_max_rise_1h"] = pd.Series(f["g_d15m"]).rolling(W1H, min_periods=2).max().to_numpy() * STEPS_PER_HOUR
    slope = np.asarray(f["g_slope_30m"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        hrs = np.where(f["glucose"] > SPIKE_MGDL, 0.0, np.where(slope > 1, (SPIKE_MGDL - f["glucose"]) / slope, 8.0))
    out["g_hours_to_180"] = np.where(np.isnan(f["glucose"]) | np.isnan(slope), np.nan, np.clip(hrs, 0, 8))
    m72 = roll(g, 3 * DAY, "mean", DAY)
    out["p_cv_72h"] = roll(g, 3 * DAY, "std", DAY) / m72
    out["g_minus_p_mean_72h"] = out["glucose"] - m72
    for c in ("patient_id", "source", "segment", "step", "clock_min"):
        out[c] = seg[c].to_numpy()
    out["glucose_raw"] = glucose.to_numpy()
    return out


def build_features(grid5: pd.DataFrame) -> pd.DataFrame:
    """5-min grid -> 15-min grid -> features/label for every step (`usable` marks modelling rows)."""
    grid15 = to_15min(grid5)
    parts = [_segment_features(seg.sort_values("step")) for _, seg in grid15.groupby("segment", sort=False)]
    return pd.concat(parts, ignore_index=True)


def modelling_frame_multi(feat: pd.DataFrame) -> pd.DataFrame:
    """Rows with real CGM now and >= 1 h history where ANY horizon's label is known (y* stay NaN where unknown)."""
    known = feat[["y"] + list(EXTRA_HORIZONS)].notna().any(axis=1)
    m = feat[feat["glucose_raw"].notna() & (feat["step"] >= MIN_HISTORY_STEPS) & known].copy()
    return m.reset_index(drop=True)


def modelling_frame(feat: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for modelling (real CGM now, known label, >=1 h history)."""
    m = feat[feat["usable"]].copy()
    m["y"] = m["y"].astype(int)
    return m.reset_index(drop=True)
