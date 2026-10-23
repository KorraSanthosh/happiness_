"""
Tests for the v9 evaluation logic: episode detection, alert policy (refractory, suppression, false alerts,
warning window) and the pre-registered feature-retention rule. Synthetic inputs only.

Run: cd backend && python -m pytest tests/test_v9_evaluation.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from app import real_data_v9 as V9


def _bins(gmax):
    return pd.DataFrame({"segment": "s", "step": np.arange(len(gmax)), "glucose_max": np.asarray(gmax, float)})


def _frame(glucose, y, p):
    n = len(glucose)
    return pd.DataFrame({"segment": "s", "step": np.arange(n), "patient_id": "x", "glucose": np.asarray(glucose, float),
                         "y": np.asarray(y, float), "p": np.asarray(p, float)})


def test_episodes_need_an_observed_normal_bin_before_onset_and_flag_serious():
    g = [150] * 5 + [190, 200] + [150] * 3 + [190, 260, 150] + [150] * 3 + [np.nan, 190, 150]
    ep = V9.episodes(_bins(g))
    assert ep.onset_step.tolist() == [5, 10]                 # the onset after a gap (NaN) is not counted
    assert ep.bins.tolist() == [2, 2] and ep.serious.tolist() == [False, True]


def test_three_consecutive_high_bins_make_an_episode_serious():
    ep = V9.episodes(_bins([150] * 3 + [185, 185, 185] + [150]))
    assert ep.serious.tolist() == [True]


def test_refractory_fires_once_per_hour_and_counts_false_alerts():
    n = 20
    d = _frame([150] * n, [0] * n, [0.9] * n)                # always alerting, never a spike
    ep = pd.DataFrame(columns=["segment", "onset_step", "bins", "peak", "serious"])
    r = V9.alert_eval(d, d.p.to_numpy(), 0.5, ep, False)
    assert r["alerts_fired"] == 5                             # steps 0, 4, 8, 12, 16
    assert r["false_alerts"] == 5
    assert abs(r["false_alerts_per_patient_day"] - 5 / (n / 96)) < 1e-3


def test_suppression_ignores_moments_already_above_180():
    d = _frame([200] * 8 + [150] * 8, [0] * 16, [0.9] * 16)
    ep = pd.DataFrame(columns=["segment", "onset_step", "bins", "peak", "serious"])
    off = V9.alert_eval(d, d.p.to_numpy(), 0.5, ep, False)
    on = V9.alert_eval(d, d.p.to_numpy(), 0.5, ep, True)
    assert on["alerts_fired"] < off["alerts_fired"] and on["alerts_fired"] == 2      # only steps 8 and 12


def test_episode_warned_only_by_alerts_30_to_120_min_before_onset():
    ep = pd.DataFrame({"segment": ["s"], "onset_step": [10], "bins": [3], "peak": [200.0], "serious": [True]})
    glucose = [150] * 10 + [200, 200, 200]
    late = np.zeros(13); late[9] = 0.9                       # 15 min before onset: too late to count
    early = np.zeros(13); early[5] = 0.9                     # 75 min before onset: counts
    assert V9.alert_eval(_frame(glucose, [1] * 13, late), late, 0.5, ep, True)["serious_warned"] == 0
    r = V9.alert_eval(_frame(glucose, [1] * 13, early), early, 0.5, ep, True)
    assert r["serious_warned"] == 1 and r["serious_lead_min_median"] == 75


def _cv(**retained):
    cands = {"base (v7 inputs)": {}}
    for name, (keep, auc) in retained.items():
        cands[name] = {"retained": keep, "vs_base": {"roc_auc_diff_mean": auc}}
    return {"candidates": cands}


def test_selection_keeps_base_when_nothing_is_retained():
    name, cols, mask = V9.selected(_cv(**{"+C1 personal history": (False, 0.001)}))
    assert name == "base (v7 inputs)" and mask is True and cols == list(V9.V7.FEATURES)


def test_selection_single_group_and_carb_masking():
    name, cols, mask = V9.selected(_cv(**{"+C3 multi-window dynamics": (True, 0.002)}))
    assert name == "+C3 multi-window dynamics" and mask is True and "g_slope_2h" in cols
    name, cols, mask = V9.selected(_cv(**{"+C2 recorded meals & oral drugs": (True, 0.002)}))
    assert mask is False and "oral_med_4h" in cols


def test_selection_requires_a_validated_combination():
    cv = _cv(**{"+C1 personal history": (True, 0.001), "+C3 multi-window dynamics": (True, 0.003)})
    name, _, _ = V9.selected(cv)
    assert name == "+C3 multi-window dynamics"                # combination not validated -> best single group
    cv["candidates"]["combined: +C1 personal history + +C3 multi-window dynamics"] = {"retained": True}
    name, cols, _ = V9.selected(cv)
    assert name.startswith("combined") and "g_slope_2h" in cols and "p_hours_since_high" in cols


def test_lead_time_is_measured_from_the_earliest_warning():
    ep = pd.DataFrame({"segment": ["s"], "onset_step": [10], "bins": [3], "peak": [200.0], "serious": [True]})
    p = np.zeros(13); p[3] = p[7] = 0.9                      # alerts 105 and 45 min before onset
    r = V9.alert_eval(_frame([150] * 10 + [200] * 3, [1] * 13, p), p, 0.5, ep, True)
    assert r["serious_warned"] == 1 and r["serious_lead_min_median"] == 105
