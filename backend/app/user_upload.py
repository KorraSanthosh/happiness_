"""
Parse a CGM export uploaded by a user into the same 5-minute grid the models were trained on.

Accepted (auto-detected):
  * Dexcom Clarity CSV export      ("Timestamp (YYYY-MM-DDThh:mm:ss)", "Event Type", "Glucose Value (mg/dL)", ...)
  * FreeStyle LibreView CSV export (metadata first line, then "Device Timestamp", "Historic Glucose mg/dL", ...)
  * any CSV with a time column and a glucose column (optional carbs / insulin columns)
Units: mg/dL or mmol/L (from the column name, else from the values). The grid is built by
external_loaders._assemble - the same harmonisation as every training dataset (range 39-450 mg/dL,
"not recorded" kept as NaN, a > 6 h gap starts a new segment). Only the most recent segment is kept.

The file is never written to disk; the caller keeps the result in memory only.
"""
from __future__ import annotations

import io
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app import external_loaders as E
from app.real_data_pipeline import MMOL_TO_MGDL

MAX_BYTES = 5 * 1024 * 1024
MAX_DAYS = 30
MIN_HOURS = 6


class UploadError(ValueError):
    """A problem the user can fix; the message is shown to them."""


def _norm(c: str) -> str:
    return re.sub(r"\s+", " ", str(c)).strip().lower()


def _find(cols: List[str], *keys: str, exclude: Tuple[str, ...] = ()) -> Optional[str]:
    for k in keys:
        for c in cols:
            n = _norm(c)
            if k in n and not any(x in n for x in exclude):
                return c
    return None


def _read_table(text: str) -> pd.DataFrame:
    """Find the header row (LibreView puts a metadata line first) and read the table."""
    lines = text.splitlines()
    for skip in range(min(12, len(lines))):
        try:
            d = pd.read_csv(io.StringIO(text), skiprows=skip, sep=None, engine="python", dtype=str)
        except Exception:                                   # noqa: BLE001 - try the next candidate header row
            continue
        cols = list(d.columns)
        if _find(cols, "timestamp", "date", "time") and _find(cols, "glucose", "sgv", "cgm", "bg", exclude=("event",)):
            return d
    raise UploadError("Could not find a time column and a glucose column. Expected a Dexcom Clarity or LibreView "
                      "export, or a CSV with columns such as 'timestamp' and 'glucose'.")


def _parse_times(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    cands = []
    for dayfirst in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)        # both orders are tried on purpose
            t = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, utc=False)
        if hasattr(t.dt, "tz") and t.dt.tz is not None:
            t = t.dt.tz_localize(None)                      # keep the wall-clock time the device recorded
        cands.append(t)
    ok = [t.notna().sum() for t in cands]
    if ok[0] != ok[1]:
        return cands[int(np.argmax(ok))]
    spans = [(t.max() - t.min()) if t.notna().any() else pd.Timedelta(0) for t in cands]
    return cands[int(np.argmin(spans))]                     # ambiguous day/month: the shorter span is the real one


def _num(s: Optional[pd.Series]) -> Optional[pd.Series]:
    if s is None:
        return None
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False).str.strip(), errors="coerce")


def parse_cgm_csv(raw: bytes) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if len(raw) > MAX_BYTES:
        raise UploadError(f"File is larger than {MAX_BYTES // (1024 * 1024)} MB. Export a shorter period (2-4 weeks is plenty).")
    text = raw.decode("utf-8-sig", errors="replace")
    d = _read_table(text)
    cols = list(d.columns)
    tcol = _find(cols, "timestamp", "date time", "datetime", "date", "time")
    notes: List[str] = []
    fmt = "generic CSV"
    etype = _find(cols, "event type", "record type")
    t = _parse_times(d[tcol])

    gcols = [c for c in (_find(cols, "historic glucose"), _find(cols, "scan glucose")) if c] or \
            [_find(cols, "glucose value", "sensor glucose", "glucose", "sgv", "cgm", "bg", exclude=("event", "insulin", "carb"))]
    if any("historic glucose" in _norm(c) for c in gcols):
        fmt = "FreeStyle LibreView export"
    elif etype and "egv" in d[etype].astype(str).str.lower().unique():
        fmt = "Dexcom Clarity export"
    g = None
    for c in gcols:                                          # Libre: historic readings first, scans fill the gaps
        v = _num(d[c])
        g = v if g is None else g.fillna(v)
    if etype and fmt == "Dexcom Clarity export":
        g = g.where(d[etype].astype(str).str.upper() == "EGV")
    high_low = int(d[gcols[0]].astype(str).str.strip().str.lower().isin(("high", "low", "hi", "lo")).sum())
    if high_low:
        notes.append(f"{high_low} 'High'/'Low' sensor readings had no number and were skipped.")
    mmol = any("mmol" in _norm(c) for c in gcols) or (g.dropna().median() < 30 if g.notna().any() else False)
    if mmol:
        g = g * MMOL_TO_MGDL
        notes.append("Glucose was in mmol/L and was converted to mg/dL.")

    carbs_col = _find(cols, "carb")
    rapid_col = _find(cols, "rapid-acting insulin", "fast-acting insulin", "bolus", "insulin value", "insulin", exclude=("long", "basal"))
    long_col = _find(cols, "long-acting insulin", "basal insulin")
    keep = t.notna()
    t, g = t[keep], g[keep]
    if g.notna().sum() == 0:
        raise UploadError("No glucose numbers found in the glucose column.")
    end = t[g.notna()].max()
    start = end - pd.Timedelta(days=MAX_DAYS)
    window = (t >= start) & (t <= end)
    if (t[g.notna()] < start).any():
        notes.append(f"Only the most recent {MAX_DAYS} days were kept.")
    tw = t[window]
    glucose = E._bin_events(tw, g[window], "mean")
    carbs = E._bin_events(tw, _num(d.loc[keep, carbs_col])[window]) if carbs_col else None
    insulin = E._bin_events(tw, _num(d.loc[keep, rapid_col])[window]) if rapid_col else None
    if long_col:                                             # long-acting insulin: spread over 24 h (causal), as in training
        ld = E._bin_events(tw, _num(d.loc[keep, long_col])[window])
        if not ld.empty:
            idx = pd.date_range(ld.index.min(), end.floor(E.FREQ), freq=E.FREQ)
            spread = pd.Series(np.convolve(ld.reindex(idx).fillna(0).to_numpy(), np.full(288, 1 / 288.0))[: len(idx)], index=idx)
            insulin = spread if insulin is None else insulin.add(spread, fill_value=0)
    grid = E._assemble("me", "upload", glucose, carbs, insulin)
    if grid.empty:
        raise UploadError("No glucose readings in the usable range (39-450 mg/dL).")
    last = grid.segment.iloc[-1]                             # segments come out in time order
    if grid.segment.nunique() > 1:
        notes.append("The data has gaps longer than 6 hours; only the period after the last gap is used.")
    grid = grid[grid.segment == last].reset_index(drop=True)
    obs = grid.glucose.notna()
    hours = len(grid) * 5 / 60
    if hours < MIN_HOURS:
        raise UploadError(f"Only {hours:.1f} hours of continuous data after the last gap; at least {MIN_HOURS} hours are needed (24 hours or more works best).")
    if hours < 24:
        notes.append("Less than 24 hours of data: predictions will be less reliable (24 h minimum, 7-14 days best).")
    step = np.median(np.diff(np.flatnonzero(obs.to_numpy()))) * 5 if obs.sum() > 1 else 5
    summary = {"format": fmt, "units": "mmol/L (converted)" if mmol else "mg/dL", "readings": int(obs.sum()),
               "hours": round(hours, 1), "days": round(hours / 24, 1), "reading_interval_min": int(step),
               "first": str(grid_start(glucose, grid)), "last": str(end),
               "has_carbs": bool(grid.carbs.notna().any()), "has_insulin": bool(grid.insulin.notna().any()),
               "notes": notes}
    return grid, summary


def grid_start(glucose: pd.Series, grid: pd.DataFrame) -> pd.Timestamp:
    """Wall-clock start of the kept segment (the grid only stores steps and time of day)."""
    end = glucose.dropna().index.max()
    return end - pd.Timedelta(minutes=5 * (len(grid) - 1))
