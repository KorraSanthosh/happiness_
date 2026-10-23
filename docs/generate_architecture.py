#!/usr/bin/env python3
"""
GlucoTwin AI architecture diagram -> docs/GlucoTwin_Architecture.pdf (submission) and docs/architecture.pdf|png
(used by the README and the slide deck).

Flow: data sources -> validation & preprocessing -> EHR + sensor integration -> feature engineering ->
trained models (artifacts) -> FastAPI (API boundary) -> React dashboard -> clinician-facing prediction.
Headline numbers are read from models/investigation/verified_metrics.json (written by app/verify_results.py).

Run from the repo root:  python3 docs/generate_architecture.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs"
VERIFIED = ROOT / "models" / "investigation" / "verified_metrics.json"

INK, MUTED = "#0f2a35", "#46616d"
COLS = {"data": ("#e0f2fe", "#0284c7"), "synth": ("#fef3c7", "#b45309"), "prep": ("#ede9fe", "#7c3aed"),
        "feat": ("#e0f7f4", "#0f8f8a"), "model": ("#dcfce7", "#15803d"), "api": ("#fee2e2", "#b91c1c"),
        "ui": ("#e6f1f4", "#0c4c58"), "out": ("#0c4c58", "#0c4c58")}


def box(ax, x, y, w, h, title, lines, kind, step=None):
    fill, edge = COLS[kind]
    txt = "#ffffff" if kind == "out" else INK
    sub = "#d6eef1" if kind == "out" else MUTED
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015", fc=fill, ec=edge, lw=1.6))
    head = f"{step}. {title}" if step else title
    ax.text(x + 0.01, y + h - 0.022, head, ha="left", va="top", fontsize=9.8, fontweight="bold", color=txt)
    ax.text(x + 0.01, y + h - 0.068, "\n".join(lines), ha="left", va="top", fontsize=7.7, color=sub, linespacing=1.42)


def arrow(ax, x1, y1, x2, y2, label=None, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15, lw=1.7, color="#334155",
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.018, label, ha="center", va="bottom", fontsize=7, color=MUTED, style="italic")


def headline() -> str:
    try:
        v = json.load(open(VERIFIED))["verified"]
        cv = v["cv"]
        parts = [f"{h}: {cv[h]['all_open']['accuracy']:.1%}" for h in ("30m", "60m", "2h")]
        t2 = cv["2h"]["type2"]["accuracy"]
        return ("Verified held-out accuracy, 187 unseen people (7-fold person-grouped CV)  ·  " + "  ·  ".join(parts)
                + f"  ·  Type 2 only, 2 h: {t2:.1%}  ·  calibration (ECE) is reported separately")
    except (FileNotFoundError, KeyError):
        return "Verified metrics: run `python3 -I scripts/run_isolated.py -m:app.verify_results` in backend/"


def main() -> None:
    fig, ax = plt.subplots(figsize=(16.5, 10))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(0.5, 0.985, "GlucoTwin AI — architecture of the predictive diabetes digital twin",
            ha="center", va="top", fontsize=16, fontweight="bold", color=INK)
    ax.text(0.5, 0.952, "Team hackers_ · IIT (ISM) Dhanbad · research prototype, not a medical device · "
            "trained on open research data (not an Indian cohort)", ha="center", va="top", fontsize=9.5, color=MUTED)

    # ── row 1: data -> validation -> integration -> features ──
    top, h1 = 0.515, 0.405
    box(ax, 0.01, top + 0.215, 0.205, 0.19, "Data sources", [
        "Open research data, 187 people:", "  Shanghai T2DM / T1DM (with EHR),",
        "  HUPA-UCM, T1D-UOM, AZT1D, BrisT1D", "External test only: OhioT1DM (DUA,", "  never trained on, never published)"],
        "data", 1)
    box(ax, 0.01, top, 0.205, 0.2, "Other inputs (labelled)", [
        "Synthetic demo: 120 simulated patients", "  (separate model, marked SYNTHETIC)",
        "User upload: own CGM CSV (Dexcom /", "  LibreView / generic) — kept in memory", "  only, never saved to disk",
        "Clinician 'daily context': meal, insulin,", "  tablets (what-if, not validated)"], "synth")
    box(ax, 0.255, top, 0.225, h1, "Validation & preprocessing", [
        "external_loaders.py · real_data_pipeline.py", "user_upload.py (same rules for uploads)", "",
        "• units: mmol/L → mg/dL, carb portions,", "  basal U/h → U per 5 min",
        "• glucose range 39–450 mg/dL", "• 'not recorded' kept as NaN, not zero",
        "• interpolated (future-leaking) points removed", "• BrisT1D 6-h windows stitched (0 conflicts)",
        "• > 6 h CGM gap starts a new segment", "• QC report for the synthetic data"], "prep", 2)
    box(ax, 0.52, top, 0.215, h1, "EHR + sensor integration", [
        "Common 15-minute grid per person", "(5-min and 15-min CGMs)", "",
        "Dynamic stream: CGM, insulin, meals,", "  heart rate, steps, oral-drug doses",
        "Static stream: EHR joined per recording", "  (age, sex, BMI, duration, HbA1c,",
        "  labs, complications, drug classes)", "", "Label: max CGM in the next 30 min /",
        "  1 h / 2 h > 180 mg/dL"], "prep", 3)
    box(ax, 0.775, top, 0.215, h1, "Feature engineering", [
        "Past-only (causal) features:", "• glucose level, slopes, projections,",
        "  rolling stats, time above / below range", "• insulin & carbs on board",
        "• personal history: same time of day", "  over 7 / 14 days, 72-h variability",
        "• wearable HR / steps, time of day", "• 31 EHR features", "",
        "One code path for training and serving;", "leak tests: changing future data never", "changes a past feature"],
        "feat", 4)

    # ── row 2 (right to left): models -> API -> UI -> clinician ──
    bot, h2 = 0.115, 0.335
    box(ax, 0.775, bot, 0.215, h2, "Trained ML models", [
        "HistGradientBoosting (scikit-learn)", "30 min & 1 h: v7 · 2 h: v9", "",
        "Artifacts (loaded once, cached):", "  backend/models/twin_v7/*.joblib", "  backend/models/twin_v9/model_2h.joblib",
        "Thresholds from validation only", "", "Validation: person-grouped nested CV,",
        "  locked test set (v9), OhioT1DM external,", "  calibration, bias, verifier (297 checks)"], "model", 5)
    box(ax, 0.52, bot, 0.215, h2, "FastAPI endpoints", [
        "/api/v7/patients[/{id}]  /stream", "/api/v7/predict/{id}?bin=", "  (+ optional what-if scenario)",
        "/api/v7/patients/{id}/overview | trends", "/api/v7/upload  (multipart CSV)", "/api/v7/model-card",
        "/api/*  synthetic demo + /api/health", "", "Per request: raw stream up to 'now' →", "same feature code → 3 models (~50–100 ms)"],
        "api", 6)
    box(ax, 0.255, bot, 0.225, h2, "React dashboard", [
        "Vite + React + Recharts (light / dark)", "Patient · My data · Prediction ·",
        "  Live monitor · Trends · Reports ·", "  Analytics · Settings", "",
        "Observed data (past, solid) is drawn", "apart from outcomes (replay, dashed)", "and from predictions (probabilities)",
        "Modes: data · alert · theme · layout"], "ui", 7)
    box(ax, 0.01, bot, 0.205, h2, "Clinician-facing output", [
        "Spike probability for 30 min, 1 h, 2 h", "Risk level from validated thresholds",
        "Early warning: only while glucose", "  is still ≤ 180 (max once per hour)",
        "Signals the model read (explanation)", "Data completeness & last reading time",
        "Replay: what really happened next", "Printable report", "",
        "EXPERIMENTAL — not for clinical use"], "out", 8)

    arrow(ax, 0.215, top + 0.31, 0.255, top + 0.25)
    arrow(ax, 0.215, top + 0.1, 0.255, top + 0.15)
    arrow(ax, 0.48, top + 0.2, 0.52, top + 0.2)
    arrow(ax, 0.735, top + 0.2, 0.775, top + 0.2)
    arrow(ax, 0.8825, top, 0.8825, bot + h2)
    ax.text(0.89, (top + bot + h2) / 2, "training", ha="left", va="center", fontsize=7.5, color=MUTED, style="italic")
    arrow(ax, 0.775, bot + 0.17, 0.735, bot + 0.17, "joblib")
    arrow(ax, 0.52, bot + 0.17, 0.48, bot + 0.17)
    arrow(ax, 0.255, bot + 0.17, 0.215, bot + 0.17)
    # API boundary
    ax.plot([0.5, 0.5], [bot - 0.02, bot + h2 + 0.03], ls="--", lw=1.3, color="#b91c1c")
    ax.text(0.5, bot + h2 + 0.034, "API boundary (JSON over HTTP)", ha="center", va="bottom", fontsize=8, color="#b91c1c", fontweight="bold")

    ax.add_patch(FancyBboxPatch((0.01, 0.012), 0.98, 0.06, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc="#f8fafc", ec="#94a3b8", lw=1.1))
    ax.text(0.5, 0.042, headline(), ha="center", va="center", fontsize=8.6, fontweight="bold", color=INK)

    OUT.mkdir(exist_ok=True)
    for name in ("GlucoTwin_Architecture.pdf", "architecture.pdf"):
        fig.savefig(OUT / name, bbox_inches="tight")
    fig.savefig(OUT / "architecture.png", dpi=160, bbox_inches="tight")
    print(f"wrote {OUT / 'GlucoTwin_Architecture.pdf'}, architecture.pdf and architecture.png")


if __name__ == "__main__":
    main()
