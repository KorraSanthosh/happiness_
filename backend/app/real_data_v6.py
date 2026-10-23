"""
v6 experiments on 87 T1D patients: OhioT1DM + BrisT1D (the original 21) + HUPA-UCM + T1D-UOM + AZT1D (66 new).

Every comparison uses the SAME patient-held-out folds: the original 21 patients keep exactly the v5 folds
(app.real_data_eval.assign_folds); the 66 new patients are dealt round-robin into the same 7 folds.

Stages (python -m app.real_data_v6 --stage ...):
  features  A. activity / personal-history features vs v5 features (train + test on the original 21)
  data      B. original-only vs original+external training, evaluated on the same held-out patients
  lodo      C. leave-one-DATASET-out: train on 4 datasets, test on a 5th never seen
  learning  D. held-out accuracy vs number of training patients (does more data still help?)
  personal  E. adapt a fold-held-out model to each new patient using their first 1-3 days
  final     F. nested tuning + diagnostics (over/under-fitting, bias) + final model
Outputs: models/combined_v6/  (v5 outputs and the served API model are untouched)
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from app import external_loaders as E
from app import real_data_pipeline as P
from app.real_data_eval import (
    HGB_DEFAULT, N_INNER, N_OUTER, SEED, TUNE_GRID, _group_table, assign_folds, best_threshold,
    cluster_bootstrap, make_hgb, metrics, patient_level,
)

warnings.filterwarnings("ignore")

OUT_DIR = P.ROOT / "models" / "combined_v6"
ORIG = ("ohio", "brist")
EXT = ("hupa", "uom", "azt1d")
CFG = TUNE_GRID["shallow"]            # v5 nested tuning picked this in 6/7 folds
CAP_ROWS = 6000                       # per-patient training weight cap (~62 days) so long records don't dominate
MODEL_VERSION = "6.0.0"
DAY = P.DAY

FEATURE_SETS: Dict[str, List[str]] = {
    "v5_engineered": P.ENGINEERED_FEATURES,
    "+activity": P.ENGINEERED_FEATURES + P.ACTIVITY_FEATURES,
    "+personal": P.ENGINEERED_FEATURES + P.PERSONAL_FEATURES,
    "v6_all": P.V6_FEATURES,
}


# ═════════════════════════════════════════════════════════════════════════════
# data / folds / helpers
# ═════════════════════════════════════════════════════════════════════════════

def load_frame(rebuild: bool = False) -> pd.DataFrame:
    path = OUT_DIR / "frame.parquet"
    if path.exists() and not rebuild:
        m = pd.read_parquet(path)
        print(f"[data] cached frame: {len(m):,} rows | {m.patient_id.nunique()} patients", flush=True)
        return m
    grid_o, stats = P.load_all_grids()
    grid_e, notes = E.load_external_grids(EXT)
    grid = pd.concat([grid_o, grid_e], ignore_index=True)
    m = P.modelling_frame(P.build_features(grid))
    obs = grid.groupby("patient_id")["glucose"].apply(lambda s: s.notna().mean())
    m["cgm_native"] = m["patient_id"].map(lambda p: "15min" if obs[p] < 0.6 else "5min")
    tail = m["segment"].str.rsplit("_", n=1).str[-1]
    m["seg_order"] = np.where(tail == "train", 0, np.where(tail == "test", 1, pd.to_numeric(tail, errors="coerce"))).astype(float)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m.to_parquet(path)
    with open(OUT_DIR / "data_notes.json", "w") as f:
        json.dump({"stitch_conflicts": stats["conflicts"], "external": notes,
                   "rows_by_source": m.source.value_counts().to_dict(),
                   "patients_by_source": m.groupby("source").patient_id.nunique().to_dict()}, f, indent=2, default=str)
    print(f"[data] built frame: {len(m):,} rows | {m.patient_id.nunique()} patients | "
          f"{m.groupby('source').patient_id.nunique().to_dict()}", flush=True)
    return m


def assign_folds_v6(m: pd.DataFrame) -> Dict[str, int]:
    fold_of = assign_folds(m[m.source.isin(ORIG)])          # identical to the v5 folds
    rng = np.random.default_rng(SEED + 1)
    i = 0
    for src in EXT:
        ps = sorted(m.loc[m.source == src, "patient_id"].unique())
        rng.shuffle(ps)
        for p in ps:
            fold_of[p] = i % N_OUTER
            i += 1
    return fold_of


def patient_weights(df: pd.DataFrame) -> np.ndarray:
    n = df.groupby("patient_id")["patient_id"].transform("size").to_numpy()
    return np.minimum(1.0, CAP_ROWS / n)


def fit_hgb(df: pd.DataFrame, cols: List[str], cfg: Optional[Dict[str, Any]] = None):
    return make_hgb(**(CFG if cfg is None else cfg)).fit(df[cols], df["y"], sample_weight=patient_weights(df))


def masks(m: pd.DataFrame) -> Dict[str, np.ndarray]:
    out = {"original_21": m.source.isin(ORIG).to_numpy(), "external_66": m.source.isin(EXT).to_numpy(),
           "all_87": np.ones(len(m), bool)}
    for s in ORIG + EXT:
        out[s] = (m.source == s).to_numpy()
    return out


def eval_block(m: pd.DataFrame, p: np.ndarray, thr: Any = 0.5) -> Dict[str, Any]:
    thr_arr = np.full(len(m), thr) if np.isscalar(thr) else np.asarray(thr)
    out = {}
    for name, mk in masks(m).items():
        mk = mk & ~np.isnan(p)
        if mk.sum() == 0:
            continue
        r = metrics(m.y.to_numpy()[mk], p[mk], float(np.median(thr_arr[mk])))
        r["accuracy"] = round(float(((p[mk] >= thr_arr[mk]).astype(int) == m.y.to_numpy()[mk]).mean()), 4)
        r["per_patient_accuracy"] = patient_level(m[mk], p[mk], thr_arr[mk])
        r["patients"] = int(m.patient_id[mk].nunique())
        out[name] = r
    return out


def rule_block(m: pd.DataFrame) -> Dict[str, float]:
    ok = ((m.glucose > 180).astype(int) == m.y).to_numpy()
    return {k: round(float(ok[v].mean()), 4) for k, v in masks(m).items() if v.any()}


def boot(m: pd.DataFrame, mk: np.ndarray, base: np.ndarray, other: Dict[str, np.ndarray], n: int = 300) -> Dict[str, Any]:
    sub = m[mk].reset_index(drop=True)
    probs = {"base": base[mk], **{k: v[mk] for k, v in other.items()}}
    r = cluster_bootstrap(sub, probs, {k: 0.5 for k in probs}, n=n)
    return {k: r[k]["vs_base"] for k in other}


def _save(name: str, obj: Any) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[save] {OUT_DIR / (name + '.json')}", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# A. features (original 21 only - isolates the feature effect)
# ═════════════════════════════════════════════════════════════════════════════

def stage_features(m: pd.DataFrame) -> Dict[str, Any]:
    fold = m.patient_id.map(assign_folds_v6(m)).to_numpy()
    orig = m.source.isin(ORIG).to_numpy()
    oof = {k: np.full(len(m), np.nan) for k in FEATURE_SETS}
    t0 = time.time()
    for f in range(N_OUTER):
        tr, te = orig & (fold != f), orig & (fold == f)
        for k, cols in FEATURE_SETS.items():
            oof[k][te] = fit_hgb(m[tr], cols).predict_proba(m.loc[te, cols])[:, 1]
        print(f"[features] fold {f + 1}/{N_OUTER} ({time.time() - t0:.0f}s)", flush=True)
    res = {"train_and_test": "original 21 patients only", "config": CFG,
           "variants": {k: eval_block(m, v)["original_21"] for k, v in oof.items()},
           "per_source": {k: {s: eval_block(m, v)[s]["accuracy"] for s in ORIG} for k, v in oof.items()},
           "paired_vs_v5_engineered": boot(m, orig, oof["v5_engineered"], {k: v for k, v in oof.items() if k != "v5_engineered"})}
    _save("stage_features", res)
    return res


# ═════════════════════════════════════════════════════════════════════════════
# B. more data: original-only vs original+external training
# ═════════════════════════════════════════════════════════════════════════════

def stage_data(m: pd.DataFrame) -> Dict[str, Any]:
    fold = m.patient_id.map(assign_folds_v6(m)).to_numpy()
    orig = m.source.isin(ORIG).to_numpy()
    variants = [("v5_engineered", "orig"), ("v5_engineered", "orig+ext"), ("v6_all", "orig"), ("v6_all", "orig+ext")]
    oof = {f"{a}|{b}": np.full(len(m), np.nan) for a, b in variants}
    t0 = time.time()
    for f in range(N_OUTER):
        te = fold == f
        for fs, data in variants:
            tr = (fold != f) & (orig if data == "orig" else True)
            cols = FEATURE_SETS[fs]
            oof[f"{fs}|{data}"][te] = fit_hgb(m[tr], cols).predict_proba(m.loc[te, cols])[:, 1]
        print(f"[data] fold {f + 1}/{N_OUTER} ({time.time() - t0:.0f}s)", flush=True)
    ext = m.source.isin(EXT).to_numpy()
    res = {"rule_glucose_gt_180": rule_block(m),
           "variants": {k: eval_block(m, v) for k, v in oof.items()},
           "paired_on_original_21_vs_v5_baseline": boot(m, orig, oof["v5_engineered|orig"],
                                                          {k: v for k, v in oof.items() if k != "v5_engineered|orig"}),
           "paired_on_external_66_vs_orig_only_v6": boot(m, ext, oof["v6_all|orig"], {"v6_all|orig+ext": oof["v6_all|orig+ext"]})}
    pd.DataFrame({"patient_id": m.patient_id, "segment": m.segment, "step": m.step, "y": m.y, **oof}
                 ).to_parquet(OUT_DIR / "oof_data.parquet")
    _save("stage_data", res)
    return res


# ═════════════════════════════════════════════════════════════════════════════
# C. leave-one-dataset-out (the "brand-new hospital / device" test)
# ═════════════════════════════════════════════════════════════════════════════

def stage_lodo(m: pd.DataFrame) -> Dict[str, Any]:
    cols = FEATURE_SETS["v6_all"]
    res: Dict[str, Any] = {}
    for d in ORIG + EXT:
        te = (m.source == d).to_numpy()
        p = fit_hgb(m[~te], cols).predict_proba(m.loc[te, cols])[:, 1]
        r = {"trained_on": [s for s in ORIG + EXT if s != d], "test": metrics(m.y[te], p, 0.5),
             "rule_glucose_gt_180_accuracy": round(float(((m.glucose[te] > 180).astype(int) == m.y[te]).mean()), 4),
             "per_patient_accuracy": patient_level(m[te], p, 0.5)}
        if d in EXT:
            tr = m.source.isin(ORIG).to_numpy()
            r["test_if_trained_on_original_21_only"] = metrics(m.y[te], fit_hgb(m[tr], cols).predict_proba(m.loc[te, cols])[:, 1], 0.5)
        res[d] = r
        print(f"[lodo] {d}: acc={r['test']['accuracy']:.3f} auc={r['test']['roc_auc']:.3f} "
              f"(rule {r['rule_glucose_gt_180_accuracy']:.3f})", flush=True)
    _save("stage_lodo", res)
    return res


# ═════════════════════════════════════════════════════════════════════════════
# D. learning curve over number of training patients
# ═════════════════════════════════════════════════════════════════════════════

def stage_learning(m: pd.DataFrame) -> Dict[str, Any]:
    cols = FEATURE_SETS["v6_all"]
    fold = m.patient_id.map(assign_folds_v6(m)).to_numpy()
    rng = np.random.default_rng(SEED)
    curve: Dict[int, Dict[str, List[float]]] = {}
    t0 = time.time()
    for f in range(N_OUTER):
        pool = np.array(sorted(m.patient_id[fold != f].unique()))
        te = fold == f
        for k in (5, 10, 20, 40, 60, len(pool)):
            k = min(k, len(pool))
            sel = pool if k == len(pool) else rng.choice(pool, k, replace=False)
            tr = m[m.patient_id.isin(sel) & (fold != f)]
            clf = fit_hgb(tr, cols)
            pte, ptr = clf.predict_proba(m.loc[te, cols])[:, 1], clf.predict_proba(tr[cols])[:, 1]
            key = 999 if k == len(pool) else k
            d = curve.setdefault(key, {"test_acc_all": [], "test_acc_original": [], "test_auc_all": [], "train_acc": [], "test_logloss": []})
            yte = m.y[te].to_numpy()
            o = m.source[te].isin(ORIG).to_numpy()
            d["test_acc_all"].append(((pte >= .5) == yte).mean())
            d["test_acc_original"].append(((pte[o] >= .5) == yte[o]).mean())
            d["test_auc_all"].append(roc_auc_score(yte, pte))
            d["train_acc"].append(((ptr >= .5) == tr.y.to_numpy()).mean())
            d["test_logloss"].append(log_loss(yte, np.clip(pte, 1e-6, 1 - 1e-6)))
        print(f"[learning] fold {f + 1}/{N_OUTER} ({time.time() - t0:.0f}s)", flush=True)
    res = [{"train_patients": ("all (~75)" if k == 999 else k), **{kk: round(float(np.mean(v)), 4) for kk, v in d.items()},
            "test_acc_all_std": round(float(np.std(d["test_acc_all"])), 4)} for k, d in sorted(curve.items())]
    _save("stage_learning", {"curve": res})
    return {"curve": res}


# ═════════════════════════════════════════════════════════════════════════════
# E. personalisation: adapt to each held-out patient with their first K days
# ═════════════════════════════════════════════════════════════════════════════

def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def stage_personal(m: pd.DataFrame, key: str = "v6_all|orig+ext") -> Dict[str, Any]:
    oof = pd.read_parquet(OUT_DIR / "oof_data.parquet")
    df = m[["patient_id", "source", "seg_order", "step", "y"]].copy()
    df["p"] = oof[key].to_numpy()
    df = df.sort_values(["patient_id", "seg_order", "step"]).reset_index(drop=True)
    df["k"] = df.groupby("patient_id").cumcount()                  # n-th usable 15-min row of this patient
    eval_start = 3 * DAY
    keep = df[df.k >= eval_start].groupby("patient_id").size()
    pats = keep[keep >= DAY].index                                 # need >= 1 day left to evaluate on
    ev_all = df[df.patient_id.isin(pats) & (df.k >= eval_start)].copy()
    out_p: Dict[str, np.ndarray] = {"global": ev_all.p.to_numpy()}
    out_t: Dict[str, np.ndarray] = {"global": np.full(len(ev_all), 0.5)}
    for K in (1, 2, 3):
        pl, sh, th = [], [], []
        for pid, d in df[df.patient_id.isin(pats)].groupby("patient_id", sort=False):
            ad, ev = d[d.k < K * DAY], d[d.k >= eval_start]
            z = _logit(ev.p.to_numpy())
            # (a) Platt recalibration of the global score on this patient's first K days
            p_pl = ev.p.to_numpy()
            if ad.y.nunique() == 2:
                lr = LogisticRegression(C=1.0).fit(_logit(ad.p.to_numpy())[:, None], ad.y)
                if lr.coef_[0, 0] > 0:
                    p_pl = lr.predict_proba(z[:, None])[:, 1]
            # (b) shrunken intercept shift: observed vs predicted spike rate in the first K days
            b = _logit(np.array([ad.y.mean()]))[0] - _logit(np.array([ad.p.mean()]))[0]
            w = len(ad) / (len(ad) + DAY)
            p_sh = 1 / (1 + np.exp(-(z + w * b)))
            # (c) personal decision threshold (probabilities untouched)
            t = best_threshold(ad.y.to_numpy(), ad.p.to_numpy(), 0.3, 0.7) if ad.y.nunique() == 2 else 0.5
            pl.append(p_pl); sh.append(p_sh); th.append(np.full(len(ev), t))
        out_p[f"platt_{K}d"] = np.concatenate(pl); out_t[f"platt_{K}d"] = np.full(len(ev_all), 0.5)
        out_p[f"shift_{K}d"] = np.concatenate(sh); out_t[f"shift_{K}d"] = np.full(len(ev_all), 0.5)
        out_p[f"threshold_{K}d"] = ev_all.p.to_numpy(); out_t[f"threshold_{K}d"] = np.concatenate(th)
    ev_all = ev_all.reset_index(drop=True)
    res: Dict[str, Any] = {"model": key, "eval_rows": "each patient's data after their first 3 days (same rows for every method)",
                           "patients": int(len(pats)), "rows": int(len(ev_all)), "methods": {}}
    for name in out_p:
        r = metrics(ev_all.y, out_p[name], 0.5)
        r["accuracy"] = round(float(((out_p[name] >= out_t[name]).astype(int) == ev_all.y).mean()), 4)
        r["per_patient_accuracy"] = patient_level(ev_all, out_p[name], out_t[name])
        r["accuracy_original_21"] = round(float((((out_p[name] >= out_t[name]).astype(int) == ev_all.y)[ev_all.source.isin(ORIG)]).mean()), 4)
        res["methods"][name] = r
    bs = cluster_bootstrap(ev_all, {k: out_p[k] for k in out_p}, {k: out_t[k] for k in out_t}, n=300)
    res["paired_vs_global"] = {k: bs[k]["vs_global"] for k in out_p if k != "global"}
    _save("stage_personal", res)
    return res


# ═════════════════════════════════════════════════════════════════════════════
# F. final: nested tuning + diagnostics + saved model
# ═════════════════════════════════════════════════════════════════════════════

FINAL_GRID = {k: TUNE_GRID[k] for k in ("shallow", "default", "slow")}


def stage_final(m: pd.DataFrame) -> Dict[str, Any]:
    cols = FEATURE_SETS["v6_all"]
    fold = m.patient_id.map(assign_folds_v6(m)).to_numpy()
    nested = pd.DataFrame({"patient_id": m.patient_id, "source": m.source, "y": m.y, "fold": fold,
                           "p": np.nan, "thr": np.nan})
    folds, models = [], []
    t0 = time.time()
    for f in range(N_OUTER):
        te = fold == f
        tr = m[~te].reset_index(drop=True)
        inner, scores = {}, {}
        for cname, cfg in FINAL_GRID.items():
            o = np.full(len(tr), np.nan)
            for a, b in GroupKFold(N_INNER).split(tr, tr.y, tr.patient_id):
                o[b] = fit_hgb(tr.iloc[a], cols, cfg).predict_proba(tr.iloc[b][cols])[:, 1]
            inner[cname] = o
            scores[cname] = float(log_loss(tr.y, np.clip(o, 1e-6, 1 - 1e-6), sample_weight=patient_weights(tr)))
        best = min(scores, key=scores.get)
        thr = best_threshold(tr.y.to_numpy(), inner[best])
        clf = fit_hgb(tr, cols, FINAL_GRID[best])
        p_te, p_tr = clf.predict_proba(m.loc[te, cols])[:, 1], clf.predict_proba(tr[cols])[:, 1]
        nested.loc[te, "p"], nested.loc[te, "thr"] = p_te, thr
        models.append(clf)
        folds.append({"fold": f, "config": best, "threshold": thr, "inner_logloss": {k: round(v, 4) for k, v in scores.items()},
                      "train_acc": round(float(((p_tr >= thr) == tr.y).mean()), 4),
                      "test_acc": round(float(((p_te >= thr) == m.y[te]).mean()), 4),
                      "train_logloss": round(float(log_loss(tr.y, np.clip(p_tr, 1e-6, 1 - 1e-6))), 4),
                      "test_logloss": round(float(log_loss(m.y[te], np.clip(p_te, 1e-6, 1 - 1e-6))), 4)})
        print(f"[final] fold {f + 1}/{N_OUTER}: {best} thr={thr} train={folds[-1]['train_acc']:.3f} "
              f"test={folds[-1]['test_acc']:.3f} ({time.time() - t0:.0f}s)", flush=True)

    p, thr = nested.p.to_numpy(), nested.thr.to_numpy()
    orig = m.source.isin(ORIG).to_numpy()
    sub_o = m[orig].reset_index(drop=True)
    res: Dict[str, Any] = {
        "features": len(cols), "grid": FINAL_GRID, "folds": folds,
        "selected_configs": pd.Series([x["config"] for x in folds]).value_counts().to_dict(),
        "nested": eval_block(m, p, thr),
        "bootstrap_original_21": cluster_bootstrap(sub_o, {"m": p[orig]}, {"m": thr[orig]}, n=300)["m"],
        "bootstrap_all_87": cluster_bootstrap(m.reset_index(drop=True), {"m": p}, {"m": thr}, n=200)["m"],
        "overfit_gap": {"mean_train_minus_test_acc": round(float(np.mean([x["train_acc"] - x["test_acc"] for x in folds])), 4),
                        "mean_test_minus_train_logloss": round(float(np.mean([x["test_logloss"] - x["train_logloss"] for x in folds])), 4)},
    }
    # bias / subgroup analysis on nested predictions
    df = nested.assign(glucose=m.glucose.to_numpy(), clock_min=m.clock_min.to_numpy(), cgm_native=m.cgm_native.to_numpy())
    df["glucose_range"] = pd.cut(df.glucose, [0, 70, 180, 1000], labels=["<70", "70-180", ">180"]).astype(str)
    df["time_of_day"] = pd.cut(df.clock_min // 60, [-1, 5, 11, 17, 23],
                               labels=["night 00-06", "morning 06-12", "afternoon 12-18", "evening 18-24"]).astype(str)
    res["subgroups"] = {by: _group_table(df, by) for by in ("source", "cgm_native", "glucose_range", "time_of_day")}
    per_pat = _group_table(df, "patient_id")
    res["per_patient_accuracy_range"] = [min(r["accuracy"] for r in per_pat), max(r["accuracy"] for r in per_pat)]
    hard = df.glucose <= 180
    res["hard_subset_glucose_le_180"] = {"accuracy": round(float(((df.p[hard] >= df.thr[hard]) == df.y[hard]).mean()), 4),
                                         "recall": round(float(((df.p[hard] >= df.thr[hard]) & (df.y[hard] == 1)).sum() / max((df.y[hard] == 1).sum(), 1)), 4),
                                         "rule_accuracy": round(float((df.y[hard] == 0).mean()), 4)}
    # permutation importance on two held-out folds (subsampled for speed)
    imp = []
    for f in (0, 1):
        te = m[fold == f].sample(n=min(15000, int((fold == f).sum())), random_state=SEED)
        pi = permutation_importance(models[f], te[cols], te.y, scoring="average_precision", n_repeats=3, random_state=SEED)
        imp.append(pd.Series(pi.importances_mean, index=cols))
    res["permutation_importance_top15"] = pd.concat(imp, axis=1).mean(axis=1).sort_values(ascending=False).head(15).round(4).to_dict()

    # final model on all 87 patients
    cfg_name = res["selected_configs"] and max(res["selected_configs"], key=res["selected_configs"].get)
    final_thr = best_threshold(m.y.to_numpy(), p)
    clf = fit_hgb(m, cols, FINAL_GRID[cfg_name])
    joblib.dump({"model": clf, "feature_cols": cols, "threshold": final_thr, "version": MODEL_VERSION,
                 "config_name": cfg_name, "config": {**HGB_DEFAULT, **FINAL_GRID[cfg_name]},
                 "grid": "15-minute bins; label = max glucose over next 8 bins > 180 mg/dL",
                 "training_patients": int(m.patient_id.nunique()), "trained_at": datetime.now(timezone.utc).isoformat()},
                OUT_DIR / "model.joblib")
    res["final_model"] = {"config": cfg_name, "threshold": final_thr, "path": str(OUT_DIR / "model.joblib")}
    nested.to_parquet(OUT_DIR / "oof_nested.parquet")
    _save("stage_final", res)
    _save("model_card", {
        "model_name": f"GlucoTwin real-data v{MODEL_VERSION}", "algorithm": "HistGradientBoostingClassifier",
        "config": {**HGB_DEFAULT, **FINAL_GRID[cfg_name]}, "n_features": len(cols), "feature_cols": cols, "threshold": final_thr,
        "data": {"datasets": {"ohio": "OhioT1DM (DUA)", "brist": "BrisT1D Kaggle (stitched)", "hupa": "HUPA-UCM CC BY 4.0",
                              "uom": "T1D-UOM CC BY 4.0", "azt1d": "AZT1D CC BY 4.0"},
                 "patients": int(m.patient_id.nunique()), "rows": int(len(m))},
        "validation": "7-fold patient-grouped nested CV; original 21 patients keep the v5 folds",
        "expected_performance": {"original_21": res["nested"]["original_21"], "all_87": res["nested"]["all_87"]},
        "not_wired_into_api": "The FastAPI app serves the synthetic v2 model.",
        "attribution": "HUPA-UCM (Hidalgo et al. 2024), T1D-UOM (Univ. of Manchester 2025), AZT1D (Mayo Clinic AZ 2025) - CC BY 4.0",
    })
    return res


# ═════════════════════════════════════════════════════════════════════════════
# G. what would meal information be worth?  (what-if experiment, not a deployable model)
# ═════════════════════════════════════════════════════════════════════════════

def _meal_features(grid15: pd.DataFrame) -> pd.DataFrame:
    """Per 15-min step: FUTURE carbs (oracle = perfectly announced meals) and a causal meal-HABIT feature."""
    parts = []
    for seg_id, s in grid15.groupby("segment", sort=False):
        s = s.sort_values("step")
        c = pd.Series(s["carbs"].to_numpy(float))
        known = not c.isna().all()
        c0 = c.fillna(0.0)
        nxt = c0.shift(-1).iloc[::-1]
        fut = {w: nxt.rolling(w, min_periods=1).sum().iloc[::-1].to_numpy() for w in (2, 4, 8)}   # t+1 .. t+w
        meal_next2h = (pd.Series(fut[8]) > 0).astype(float)
        habit = np.nanmean(np.column_stack([meal_next2h.shift(DAY * k).to_numpy() for k in range(1, 8)]), axis=1) \
            if len(s) > DAY else np.full(len(s), np.nan)                                          # k >= 1 day back: known before t
        nan = np.full(len(s), np.nan)
        parts.append(pd.DataFrame({
            "segment": seg_id, "step": s["step"].to_numpy(),
            "oracle_carbs_next_30m": fut[2] if known else nan, "oracle_carbs_next_1h": fut[4] if known else nan,
            "oracle_carbs_next_2h": fut[8] if known else nan, "habit_meal_next_2h_7d": habit if known else nan}))
    return pd.concat(parts, ignore_index=True)


def stage_meals(m: pd.DataFrame) -> Dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = pd.concat([P.load_all_grids()[0], E.load_external_grids(EXT)[0]], ignore_index=True)
        mf = _meal_features(P.to_15min(grid))
    d = m.merge(mf, on=["segment", "step"], how="left")
    d = d[d.oracle_carbs_next_2h.notna()].reset_index(drop=True)          # patients whose meals are recorded
    fold = d.patient_id.map(assign_folds_v6(m)).to_numpy()
    base = FEATURE_SETS["v6_all"]
    variants = {
        "v6 (no future info)": base,
        "+ meal-time habit (deployable)": base + ["habit_meal_next_2h_7d"],
        "+ meals announced 30 min ahead (what-if)": base + ["oracle_carbs_next_30m"],
        "+ meals announced 2 h ahead (what-if)": base + ["oracle_carbs_next_30m", "oracle_carbs_next_1h", "oracle_carbs_next_2h"],
    }
    oof = {k: np.full(len(d), np.nan) for k in variants}
    t0 = time.time()
    for f in range(N_OUTER):
        te = fold == f
        for k, cols in variants.items():
            oof[k][te] = fit_hgb(d[~te], cols).predict_proba(d.loc[te, cols])[:, 1]
        print(f"[meals] fold {f + 1}/{N_OUTER} ({time.time() - t0:.0f}s)", flush=True)
    hard = (d.glucose <= 180).to_numpy()
    res: Dict[str, Any] = {"rows": int(len(d)), "patients": int(d.patient_id.nunique()),
                           "patients_by_source": d.groupby("source").patient_id.nunique().to_dict(),
                           "rule_accuracy": round(float(((d.glucose > 180).astype(int) == d.y).mean()), 4), "variants": {}}
    for k, p in oof.items():
        r = metrics(d.y, p, 0.5)
        h = metrics(d.y[hard], p[hard], 0.5)
        res["variants"][k] = {"accuracy": r["accuracy"], "roc_auc": r["roc_auc"], "recall": r["recall"],
                              "precision": r["precision"], "per_patient_accuracy": patient_level(d, p, 0.5),
                              "hard_rows_glucose_le_180": {"accuracy": h["accuracy"], "recall": h["recall"], "roc_auc": h["roc_auc"]}}
    first = list(oof)[0]
    bs = cluster_bootstrap(d, {"base": oof[first], **{k: v for k, v in oof.items() if k != first}},
                           {k: 0.5 for k in ["base"] + [k for k in oof if k != first]}, n=300)
    res["paired_vs_v6"] = {k: bs[k]["vs_base"] for k in oof if k != first}
    _save("stage_meals", res)
    return res


STAGES = {"features": stage_features, "data": stage_data, "lodo": stage_lodo, "learning": stage_learning,
          "personal": stage_personal, "final": stage_final, "meals": stage_meals}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=list(STAGES) + ["all"])
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    m = load_frame(a.rebuild)
    for s in (list(STAGES) if a.stage == "all" else [a.stage]):
        t = time.time()
        print(f"\n===== stage: {s} =====", flush=True)
        STAGES[s](m)
        print(f"===== {s} finished in {time.time() - t:.0f}s =====", flush=True)


if __name__ == "__main__":
    main()
