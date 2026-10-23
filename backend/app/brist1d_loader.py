"""
BrisT1D Data Loader — Real structure (508 columns, 9 patients, 177k rows)
==========================================================================
Wide format: each row = one prediction point with 6-hour lookback (72×5min)
  bg-5:55 … bg-0:00  (72 BG cols, oldest→newest, mmol/L)
  insulin-5:55 … insulin-0:00
  carbs-5:55   … carbs-0:00
  hr-5:55      … hr-0:00

Converts to OhioT1DM-compatible LONG format for combined training.
"""

import os, re
import numpy as np
import pandas as pd

BRIST1D_DIR = os.path.join(os.path.dirname(__file__), "../../brist1d_raw")
MMOL_TO_MGDL = 18.018   # BrisT1D uses mmol/L


def _offset_minutes(col: str) -> int:
    """'bg-5:55' → 355 minutes."""
    m = re.search(r"(\d+):(\d+)$", col)
    return int(m.group(1)) * 60 + int(m.group(2)) if m else 0


def load_brist1d(split: str = "train") -> pd.DataFrame:
    fpath = os.path.join(os.path.abspath(BRIST1D_DIR), f"{split}.csv")
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"BrisT1D {split}.csv not found at: {fpath}\n"
            "Download from https://www.kaggle.com/competitions/brist1d/data\n"
            "and place in: project_happiness/brist1d_raw/"
        )

    print(f"  Reading BrisT1D {split}.csv ...")
    df = pd.read_csv(fpath, low_memory=False)
    print(f"  {len(df):,} rows | {df['p_num'].nunique()} patients | {len(df.columns)} columns")

    # Sort columns oldest→newest (highest offset = oldest)
    def sorted_cols(prefix):
        cols = [c for c in df.columns if c.startswith(f"{prefix}-")]
        return sorted(cols, key=_offset_minutes, reverse=True)

    bg_cols    = sorted_cols("bg")
    ins_cols   = sorted_cols("insulin")
    carbs_cols = sorted_cols("carbs")
    hr_cols    = sorted_cols("hr")
    n_steps    = len(bg_cols)

    records = []
    for pid, grp in df.groupby("p_num"):
        grp = grp.reset_index(drop=True)
        n   = len(grp)

        for i, bgc in enumerate(bg_cols):
            bg    = pd.to_numeric(grp[bgc], errors="coerce").values
            ins   = pd.to_numeric(grp[ins_cols[i]],   errors="coerce").fillna(0).values if i < len(ins_cols)   else np.zeros(n)
            carbs = pd.to_numeric(grp[carbs_cols[i]], errors="coerce").fillna(0).values if i < len(carbs_cols) else np.zeros(n)
            hr    = pd.to_numeric(grp[hr_cols[i]],    errors="coerce").values            if i < len(hr_cols)    else np.full(n, np.nan)

            records.append(pd.DataFrame({
                "patient_id":     f"brist_{pid}",
                "timestamp":      grp.index.values * n_steps + i,
                "glucose_mgdl":   bg * MMOL_TO_MGDL,
                "carbs_grams":    carbs,
                "bolus_units":    ins,
                "basal_rate":     0.0,
                "heart_rate_bpm": hr,
                "gsr":            0.0,
                "missing_cbg":    np.where(np.isnan(bg), 1, 0),
                "year":           "2024",
                "split":          split,
            }))

    result = pd.concat(records, ignore_index=True)
    result = result[result["missing_cbg"] == 0].dropna(subset=["glucose_mgdl"])
    result = result.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    result["heart_rate_bpm"] = (
        result.groupby("patient_id")["heart_rate_bpm"]
        .transform(lambda s: s.fillna(s.median()))
    )
    print(f"  → {len(result):,} rows | {result['patient_id'].nunique()} patients | "
          f"Spike prev: {(result['glucose_mgdl']>180).mean()*100:.1f}%")
    return result


if __name__ == "__main__":
    df = load_brist1d("train")
    print(df.head())
