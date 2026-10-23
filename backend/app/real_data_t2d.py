"""
v8 - can we raise 2-hour accuracy for TYPE 2 diabetes without over-fitting?

Every variant is scored on the SAME 100 Type 2 people (ShanghaiT2DM), each predicted by a model that never
saw them (the v7 person-grouped folds), so the comparison with v7 is paired and fair.

  V0  v7 as shipped                       (models/combined_v7/oof_2h.parquet)
  V1  v7 features + person-grouped early stopping + Type-2-specific threshold
  V2  V1 + Type 2 meal carbs (estimated from the meal log) + oral-drug features
  V3  V2 with Type 2 rows weighted x3 in training
  V4  V2 trained on Type 2 people only

Early stopping (no leakage): inside each outer fold, a 3-fold PERSON-grouped inner CV records held-out
log-loss every 10 trees; the tree count with the lowest inner Type-2 log-loss is used to refit on the whole
outer-training set. scikit-learn's own early stopping (random 10 % of ROWS) is switched off. The decision
threshold is the accuracy optimum on the same inner out-of-fold Type 2 predictions.

python -m app.real_data_t2d [--rebuild]   ->  models/combined_t2d/
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from app import external_loaders as E
from app import real_data_pipeline as P
from app import real_data_v7 as V7
from app.real_data_eval import SEED, cluster_bootstrap, metrics, patient_level

warnings.filterwarnings("ignore")

OUT_DIR = P.ROOT / "models" / "combined_t2d"
LABEL = "y"                                         # 2-hour horizon
ES_MAX_ITER, ES_EVERY = 900, 10
T2D_WEIGHT = 3.0
FEATS_V7 = V7.FEATURES
FEATS_V8 = V7.FEATURES + P.ORAL_MED_FEATURES        # carb features already in V7.FEATURES; now filled for Shanghai
CARB_FEATURES = ["carbs_now", "carbs_30m", "carbs_1h", "carbs_2h", "carbs_4h", "cob", "since_meal", "last_meal_size", "cob_per_iob"]


def load_frame(rebuild: bool = False) -> pd.DataFrame:
    """v7 frame with the Shanghai rows rebuilt using meal-log carbs and oral-drug events."""
    path = OUT_DIR / "frame.parquet"
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    v7 = pd.read_parquet(V7.OUT_DIR / "frame.parquet")
    keep = v7[~v7.source.isin(("sh_t1", "sh_t2"))].copy()
    for c in P.ORAL_MED_FEATURES:
        keep[c] = np.nan
    grids = [E.load_shanghai_t1_meals_grid()[0], E.load_shanghai_t2_meals_grid()[0]]
    sh = P.modelling_frame_multi(P.build_features(pd.concat(grids, ignore_index=True)))
    sh["cgm_native"] = "15min"
    sh["recording"] = E.recording_of_segment(sh["segment"])
    sh = sh.merge(E.shanghai_ehr(), on="recording", how="left")
    sh["diabetes_type"] = np.where(sh["ehr_t2d"] == 1, "T2D", "T1D")
    m = pd.concat([keep, sh.reindex(columns=keep.columns)], ignore_index=True)   # same columns, same order
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m.to_parquet(path)
    return m


def _hgb(n_iter: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**{**V7.HGB_DEFAULT, **V7.CFG, "max_iter": n_iter},
                                          early_stopping=False, random_state=SEED)


def _weights(df: pd.DataFrame, t2d_weight: float) -> np.ndarray:
    w = V7.weights(df)
    return w * np.where(df.diabetes_type.to_numpy() == "T2D", t2d_weight, 1.0)


def nested_es(d: pd.DataFrame, fold: np.ndarray, cols: List[str], train_mask: Callable[[pd.DataFrame], np.ndarray],
              t2d_weight: float, tag: str) -> Dict[str, Any]:
    """Outer person-held-out CV with inner person-grouped early stopping. Returns OOF p/thr and fold info."""
    p = np.full(len(d), np.nan)
    thr = np.full(len(d), np.nan)
    folds, t0 = [], time.time()
    checkpoints = list(range(ES_EVERY, ES_MAX_ITER + 1, ES_EVERY))
    for f in range(V7.N_OUTER):
        te = fold == f
        tr = d[(~te) & train_mask(d)].reset_index(drop=True)
        y_tr = tr[LABEL].astype(int).to_numpy()
        staged = {c: np.full(len(tr), np.nan) for c in checkpoints}
        for a, b in GroupKFold(V7.N_INNER).split(tr, y_tr, tr.patient_id):
            clf = _hgb(ES_MAX_ITER).fit(tr.iloc[a][cols], y_tr[a], sample_weight=_weights(tr.iloc[a], t2d_weight))
            for k, pr in enumerate(clf.staged_predict_proba(tr.iloc[b][cols]), start=1):
                if k in staged:
                    staged[k][b] = pr[:, 1]
        t2 = (tr.diabetes_type == "T2D").to_numpy()
        losses = {c: log_loss(y_tr[t2], np.clip(staged[c][t2], 1e-6, 1 - 1e-6)) for c in checkpoints}
        best = min(losses, key=losses.get)
        th = V7.acc_threshold(y_tr[t2], staged[best][t2])
        clf = _hgb(best).fit(tr[cols], y_tr, sample_weight=_weights(tr, t2d_weight))
        p[te] = clf.predict_proba(d.loc[te, cols])[:, 1]
        thr[te] = th
        ptr = clf.predict_proba(tr[cols][t2])[:, 1]
        te2 = te & (d.diabetes_type == "T2D").to_numpy()
        folds.append({"fold": f, "trees": best, "threshold": th,
                      "train_acc_t2d": round(float(((ptr >= th).astype(int) == y_tr[t2]).mean()), 4),
                      "test_acc_t2d": round(float(((p[te2] >= th).astype(int) == d.loc[te2, LABEL]).mean()), 4)})
        print(f"[{tag}] fold {f + 1}/{V7.N_OUTER}: trees={best} thr={th} train={folds[-1]['train_acc_t2d']:.3f} "
              f"test={folds[-1]['test_acc_t2d']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return {"p": p, "thr": thr, "folds": folds}


def _summary(d: pd.DataFrame, p: np.ndarray, thr: np.ndarray, folds=None) -> Dict[str, Any]:
    t2 = (d.diabetes_type == "T2D").to_numpy()
    y = d[LABEL].astype(int).to_numpy()
    r = metrics(y[t2], p[t2], thr[t2])                     # per-row (fold) thresholds for every metric
    r["per_person_accuracy"] = patient_level(d[t2].assign(y=y[t2]), p[t2], thr[t2])
    if folds:
        r["trees_per_fold"] = [x["trees"] for x in folds]
        r["overfit_gap_t2d"] = round(float(np.mean([x["train_acc_t2d"] - x["test_acc_t2d"] for x in folds])), 4)
    return r


def rescore() -> None:
    """Recompute results.json metrics from the saved predictions (no training)."""
    res = json.load(open(OUT_DIR / "results.json"))
    o = pd.read_parquet(OUT_DIR / "oof_2h_t2d.parquet")
    o = o.merge(load_frame()[["segment", "step", "patient_id"]], on=["segment", "step"], how="left")
    names = {"V0": "V0 v7 as shipped", "V1": "V1 + person-grouped early stopping, T2D threshold",
             "V2": "V2 + Type 2 meal carbs & oral-drug features", "V3": "V3 = V2 with Type 2 rows weighted x3",
             "V4": "V4 = V2 trained on Type 2 people only"}
    for k, name in names.items():
        old = res["variants"][name]
        new = _summary(o, o[f"p_{k}"].to_numpy(), o[f"thr_{k}"].to_numpy())
        for keep in ("trees_per_fold", "overfit_gap_t2d", "folds"):
            if keep in old:
                new[keep] = old[keep]
        res["variants"][name] = new
    res["rescored"] = "metrics recomputed from saved predictions with per-row fold thresholds (no retraining)"
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    print("[rescore]", OUT_DIR / "results.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--rescore", action="store_true", help="recompute metrics from saved predictions only")
    a = ap.parse_args()
    if a.rescore:
        return rescore()
    m = load_frame(a.rebuild)
    d = m[m.source.isin(V7.OPEN) & m[LABEL].notna()].reset_index(drop=True)
    fold = d.patient_id.map(V7.assign_folds(m)).to_numpy()
    print(f"[data] {len(d):,} rows | {d.patient_id.nunique()} people | T2D rows {int((d.diabetes_type == 'T2D').sum()):,} | "
          f"T2D carbs known {d.loc[d.diabetes_type == 'T2D', 'carbs_2h'].notna().mean():.2f}", flush=True)

    # V0: the shipped v7 predictions, aligned to these rows
    v0 = pd.read_parquet(V7.OUT_DIR / "oof_2h.parquet")[["segment", "step", "p", "thr_acc"]]
    al = d[["segment", "step"]].merge(v0, on=["segment", "step"], how="left")
    res: Dict[str, Any] = {"evaluated_on": "100 Type 2 people (ShanghaiT2DM), 2-hour horizon, person-held-out (v7 folds)",
                           "variants": {"V0 v7 as shipped": _summary(d, al.p.to_numpy(), al.thr_acc.to_numpy())}}
    probs = {"V0": al.p.to_numpy()}
    thrs = {"V0": al.thr_acc.to_numpy()}
    everyone = lambda df: np.ones(len(df), bool)
    only_t2d = lambda df: (df.diabetes_type == "T2D").to_numpy()
    # V1 must see exactly v7's inputs: Shanghai meal carbs were unknown (NaN) in v7
    d_v7 = d.copy()
    d_v7.loc[d_v7.source.isin(("sh_t1", "sh_t2")), CARB_FEATURES] = np.nan
    runs = [("V1", "V1 + person-grouped early stopping, T2D threshold", d_v7, FEATS_V7, everyone, 1.0),
            ("V2", "V2 + Type 2 meal carbs & oral-drug features", d, FEATS_V8, everyone, 1.0),
            ("V3", "V3 = V2 with Type 2 rows weighted x3", d, FEATS_V8, everyone, T2D_WEIGHT),
            ("V4", "V4 = V2 trained on Type 2 people only", d, FEATS_V8, only_t2d, 1.0)]
    for key, name, frame, cols, mask, w in runs:
        out = nested_es(frame, fold, cols, mask, w, key)
        probs[key], thrs[key] = out["p"], out["thr"]
        res["variants"][name] = _summary(d, out["p"], out["thr"], out["folds"])
        res["variants"][name]["folds"] = out["folds"]
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / "results.json", "w") as f:
            json.dump(res, f, indent=2, default=str)
    t2 = (d.diabetes_type == "T2D").to_numpy()
    sub = d[t2].reset_index(drop=True).assign(y=d.loc[t2, LABEL].astype(int).to_numpy())
    bs = cluster_bootstrap(sub, {k: v[t2] for k, v in probs.items()}, {k: v[t2] for k, v in thrs.items()}, n=400)
    res["paired_vs_V0_on_type2"] = {k: bs[k]["vs_V0"] for k in probs if k != "V0"}
    res["rule_accuracy_type2"] = round(float(((d.glucose[t2] > 180).astype(int) == d.loc[t2, LABEL]).mean()), 4)
    pd.DataFrame({"segment": d.segment, "step": d.step, "y": d[LABEL], "diabetes_type": d.diabetes_type, "fold": fold,
                  **{f"p_{k}": v for k, v in probs.items()}, **{f"thr_{k}": v for k, v in thrs.items()}}
                 ).to_parquet(OUT_DIR / "oof_2h_t2d.parquet")
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    print("[done]", OUT_DIR / "results.json", flush=True)


if __name__ == "__main__":
    main()
