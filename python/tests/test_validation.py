import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score

from credit_validation import config, models, stability
from credit_validation import columns as C
from credit_validation import validation as V
from credit_validation import benchmark as B


@pytest.fixture(autouse=True)
def isolate_outputs(tmp_path, monkeypatch):
    # tests must never write into the pipeline's real output or model folders
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "EXCEL_INPUT_DIR", tmp_path / "out" / "excel_inputs")


def test_auc_hand_and_ties_and_sklearn():
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])
    assert V.auc(y, p) == pytest.approx(0.75)
    assert V.gini(y, p) == pytest.approx(0.5)
    assert V.auc(y, np.full(4, 0.3)) == pytest.approx(0.5)
    rng = np.random.default_rng(1)
    yy, pp = (rng.random(500) < 0.2).astype(int), np.round(rng.random(500), 2)
    assert V.auc(yy, pp) == pytest.approx(roc_auc_score(yy, pp))


def test_ks_hand():
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])
    assert V.ks(y, p) == pytest.approx(0.5)
    assert V.ks(y, np.full(4, 0.3)) == pytest.approx(0.0)


def test_decile_counts_and_tie_break_by_input_order():
    N = 25
    y = np.zeros(N)
    y[:3] = 1
    t = V.decile_table(y, np.full(N, 0.2))              # all tied: first three input rows land in decile 1
    assert t["n"].tolist() == [3, 2, 3, 2, 3, 2, 3, 2, 3, 2]
    assert t["bads"].iloc[0] == 3 and t["bads"].iloc[1:].sum() == 0
    assert t["cum_bad_share"].iloc[-1] == pytest.approx(1.0) and t["cum_good_share"].iloc[-1] == pytest.approx(1.0)


def test_decile_table_ks_and_order():
    rng = np.random.default_rng(2)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    t = V.decile_table(y, p)
    assert t["n"].sum() == 1000 and t["bads"].sum() == y.sum()
    assert t["ks_at_decile"].max() <= V.ks(y, p) + 1e-12
    assert t["mean_pd"].is_monotonic_decreasing


def test_rank_order_inversions():
    assert V.rank_order_inversions([0.1, 0.08, 0.09, 0.05]) == 1
    assert V.rank_order_inversions([0.05, 0.06, 0.055, 0.07], increasing=True) == 1
    assert V.rank_order_inversions([0.1, 0.09, 0.08]) == 0


def test_cap_table_endpoints():
    rng = np.random.default_rng(4)
    p = rng.random(400)
    y = (rng.random(400) < p * 0.3).astype(int)
    c = V.cap_table(y, p)
    assert c["pop_share"].iloc[0] == 0 and c["bad_share"].iloc[0] == 0
    assert c["pop_share"].iloc[-1] == pytest.approx(1) and c["bad_share"].iloc[-1] == pytest.approx(1)
    assert (c["bad_share"] <= c["perfect"] + 1e-9).all() and c["bad_share"].is_monotonic_increasing


def test_psi_hand_and_floor():
    assert stability.psi_from_shares([0.5, 0.5], [0.4, 0.6]) == pytest.approx(-0.1 * np.log(0.8) + 0.1 * np.log(1.2))
    got = stability.psi_from_shares([0.5, 0.5], [0.0, 1.0])       # zero share is floored at 1e-4
    assert got == pytest.approx((1e-4 - 0.5) * np.log(1e-4 / 0.5) + (1 - 0.5) * np.log(1 / 0.5))
    x = np.arange(1000.0)
    assert stability.psi(x, x) == pytest.approx(0.0, abs=1e-12)
    assert stability.psi(x, x + 300) > 0.25


def test_psi_edges_use_dev_only_and_left_open_bins():
    dev = np.arange(1, 101, dtype=float)
    edges = stability.dev_edges(dev)
    assert len(edges) == 9
    sh = stability.shares(np.array([edges[0]]), edges)           # value equal to an edge falls in the lower bin
    assert sh[0] == 1.0


def test_hosmer_lemeshow_hand():
    p = np.r_[np.full(10, 0.1), np.full(10, 0.5)]
    y = np.r_[np.ones(2), np.zeros(8), np.ones(7), np.zeros(3)]
    h = V.hosmer_lemeshow(y, p, g=2)
    stat = 1 / 0.9 + 4 / 2.5
    assert h["stat"] == pytest.approx(stat) and h["df"] == 2
    assert h["p_value"] == pytest.approx(stats.chi2.sf(stat, 2))
    hi = V.hosmer_lemeshow(y, p, g=4, in_sample=True)
    assert hi["df"] == 2                                          # g - 2 in sample
    assert V.hosmer_lemeshow(np.zeros(120_000), np.full(120_000, 0.01))["note"] != ""


def test_binomial_and_jeffreys_vs_scipy():
    rng = np.random.default_rng(5)
    p = np.r_[np.full(400, 0.05), np.full(300, 0.2)]
    y = (rng.random(700) < p * 1.3).astype(int)
    g = np.r_[np.full(400, "A"), np.full(300, "B")]
    t = V.binomial_test_by_group(y, p, g).set_index("group")
    for grp, pm in (("A", 0.05), ("B", 0.2)):
        n, d = t.loc[grp, "n"], t.loc[grp, "bads"]
        assert t.loc[grp, "p_binom"] == pytest.approx(stats.binom.sf(d - 1, n, pm))
        assert t.loc[grp, "p_jeffreys"] == pytest.approx(stats.beta.cdf(pm, d + 0.5, n - d + 0.5))
        assert t.loc[grp, "p_z"] == pytest.approx(stats.norm.sf((d - n * pm) / np.sqrt(n * pm * (1 - pm))))


def test_delong_symmetry_and_identity():
    rng = np.random.default_rng(6)
    y = (rng.random(800) < 0.25).astype(float)
    p1 = rng.random(800) + 0.5 * y
    p2 = rng.random(800) + 0.2 * y
    d12, z12, pv12 = V.delong_test(y, p1, p2)
    d21, z21, pv21 = V.delong_test(y, p2, p1)
    assert d12 == pytest.approx(-d21) and z12 == pytest.approx(-z21) and pv12 == pytest.approx(pv21)
    assert d12 == pytest.approx(V.auc(y, p1) - V.auc(y, p2))
    assert V.delong_test(y, p1, p1)[0] == 0.0


def test_bootstrap_ci_brackets_estimate_and_cap():
    rng = np.random.default_rng(7)
    y = (rng.random(2000) < 0.2).astype(float)
    p = rng.random(2000) + 0.4 * y
    lo, hi, n_used = V.bootstrap_ci(V.auc, y, p, n_boot=200)
    assert lo < V.auc(y, p) < hi and n_used == 2000
    lo2, hi2, n2 = V.bootstrap_ci(V.auc, y, p, n_boot=50, cap=500)
    assert n2 == 500 and lo2 < hi2
    dlo, dhi, _ = V.bootstrap_diff_ci(V.auc, y, p, rng.random(2000), n_boot=100)
    assert dlo < dhi


def test_bootstrap_is_seeded():
    rng = np.random.default_rng(8)
    y = (rng.random(500) < 0.3).astype(float)
    p = rng.random(500)
    assert V.bootstrap_ci(V.auc, y, p, n_boot=30) == V.bootstrap_ci(V.auc, y, p, n_boot=30)


def test_calibration_slope_near_one_when_calibrated():
    rng = np.random.default_rng(9)
    p = rng.uniform(0.02, 0.4, 150_000)
    y = (rng.random(150_000) < p).astype(int)
    r = V.calibration_slope_intercept(y, p)
    assert r["slope"] == pytest.approx(1.0, abs=0.03) and abs(r["intercept"]) < 0.05
    assert r["slope_lo"] < r["slope"] < r["slope_hi"]
    assert V.central_tendency(y, p)["rel_dev"] < 0.02


def test_brier_and_skill():
    y, p = np.array([1, 0, 0, 0]), np.array([0.5, 0.1, 0.1, 0.1])
    assert V.brier(y, p) == pytest.approx((0.25 + 3 * 0.01) / 4)
    assert V.brier_skill(y, np.full(4, 0.25)) == pytest.approx(0.0)


@pytest.mark.parametrize("fn,key,cases", [
    (V.light_high, "gini", [(0.40, "Green"), (0.3999, "Amber"), (0.30, "Amber"), (0.2999, "Red")]),
    (V.light_high, "ks", [(0.30, "Green"), (0.25, "Amber"), (0.199, "Red")]),
    (V.light_high, "hl_p", [(0.05, "Green"), (0.0499, "Amber"), (0.01, "Amber"), (0.0099, "Red")]),
    (V.light_high, "binom_p", [(0.05, "Green"), (1e-4, "Amber"), (9e-5, "Red")]),
    (V.light_low, "psi", [(0.10, "Green"), (0.1001, "Amber"), (0.25, "Amber"), (0.2501, "Red")]),
    (V.light_low, "gini_decay", [(0.10, "Green"), (0.15, "Amber"), (0.2001, "Red")]),
    (V.light_low, "central_tendency", [(0.10, "Green"), (0.25, "Amber"), (0.26, "Red")]),
    (V.light_low, "rank_inversions", [(0, "Green"), (1, "Amber"), (2, "Amber"), (3, "Red")]),
    (V.light_band, "slope", [(0.9, "Green"), (1.1, "Green"), (0.89, "Amber"), (1.2, "Amber"), (0.79, "Red"), (1.21, "Red")]),
])
def test_traffic_light_boundaries(fn, key, cases):
    for v, expected in cases:
        assert fn(v, key) == expected, (key, v)
    assert fn(float("nan"), key) == V.LIGHT_NA


def test_result_row_keys():
    r = V.result("gini", "m", "oot1", 0.5, "gini")
    assert set(r) == {"test", "model", "split", "value", "threshold", "light"} and r["light"] == "Green"
    assert V.result("auc", "m", "oot1", 0.7)["light"] == "Info"


@pytest.fixture(scope="module")
def champs(fixture_splits):
    return models.fit_champions(fixture_splits)


def test_run_battery_on_fixture(champs, fixture_splits, tmp_path):
    out = tmp_path / "battery"
    res = V.run_battery(champs, fixture_splits, out_dir=out, n_boot=20)
    assert list(res.columns) == V.RESULT_COLUMNS
    assert set(res["light"]) <= {"Green", "Amber", "Red", "Info", "n/a"}
    for f in ("validation_results.csv", "benchmark.csv", "psi_csi.csv", "psi_by_quarter.csv", "by_vintage.csv", "scorecard_points.csv",
              "binning_report.csv", "model_meta.json", "grade_binomial_oot1.csv", "deciles_sc_full_oot1.csv", "calibration_sc_full_oot1.csv"):
        assert (out / f).exists(), f
    meta = json.loads((out / "model_meta.json").read_text())
    assert meta["data_source"] == config.DATA_SOURCE
    assert not (config.OUT_DIR / "validation_results.csv").exists()


def test_benchmark_ordinal_is_stable_across_splits():
    a = B.ordinal(pd.Series(["A1", "B2"]))
    b = B.ordinal(pd.Series(["B2", "G5", "A1"]))
    assert a[1] == b[0] and a[0] == b[2]
    assert B.ordinal(pd.Series(["A", "C"])).tolist() == [0.0, 2.0]
    assert B.beats_light(0.01, 0.01) == "Green" and B.beats_light(0.01, 0.4) == "Amber" and B.beats_light(-0.01, 0.0) == "Red"


@pytest.mark.slow
def test_real_data_gini_is_in_a_sane_range():
    from credit_validation import data, features, targets, splits
    loans = features.build_features(targets.add_targets(data.load_loans()))
    sp = splits.time_splits(loans)
    b = models.fit_champions(sp)
    g = V.gini(sp[C.OOT1][C.TARGET_BAD12], b["sc_full"].predict_pd(sp[C.OOT1]))
    assert 0.2 < g < 0.6
