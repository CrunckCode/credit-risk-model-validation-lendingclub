import numpy as np
import pandas as pd
import pytest

from credit_validation import charts, config, explain

lgb = pytest.importorskip("lightgbm")
pytest.importorskip("shap")

# ===== CONFIG (user inputs) =====
N_ROWS, SEED = 600, 3
FEATURE_NAMES = ("x_num_a", "x_num_b", "x_cat")
TARGET = "y_test"
MODELS = ("sc_full", "sc_indep", "xgb", "lgbm")
SPLITS = ("dev_holdout", "oot1", "oot2")
# ===== END CONFIG =====


class _Bundle:
    """Minimal stand-in for models.ModelBundle (features, estimator, predict_margin)."""

    def __init__(self, est, features):
        self.estimator, self.features = est, list(features)

    def predict_margin(self, X):
        return self.estimator.predict(X[self.features], raw_score=True)


@pytest.fixture(scope="module")
def small():
    rng = np.random.default_rng(SEED)
    a, b, c = FEATURE_NAMES
    X = pd.DataFrame({a: rng.normal(size=N_ROWS), b: rng.normal(size=N_ROWS),
                      c: pd.Categorical(rng.choice(list("abc"), N_ROWS))})
    y = (rng.uniform(size=N_ROWS) < 1 / (1 + np.exp(-(1.5 * X[a] - 0.5)))).astype(int)
    est = lgb.LGBMClassifier(n_estimators=30, num_leaves=7, verbose=-1, random_state=SEED).fit(X, y)
    return _Bundle(est, X.columns), X.assign(**{TARGET: y})


def test_shap_additivity(small):
    b, X = small
    assert explain.additivity_gap(b, X) < 1e-4
    assert explain.assert_additive(b, X) < 1e-4


def test_run_explain_writes_only_to_tmp(small, tmp_path, monkeypatch):
    b, X = small
    monkeypatch.setattr(config, "SHAP_SAMPLE", 200)
    pd.DataFrame({"variable": [FEATURE_NAMES[0], FEATURE_NAMES[1]], "points": [10, 50]}).to_csv(tmp_path / "scorecard_points.csv", index=False)
    pd.DataFrame({"variable": [FEATURE_NAMES[0], FEATURE_NAMES[1]], "iv_contrib": [0.5, 0.01]}).to_csv(tmp_path / "binning_report.csv", index=False)
    res = explain.run_explain(b, X, out_dir=tmp_path, chart_dir=tmp_path / "ch")
    assert res["additivity_gap"] < 1e-4
    imp = pd.read_csv(tmp_path / "shap_importance.csv")
    assert imp["feature"].iloc[0] == FEATURE_NAMES[0]
    cmp = pd.read_csv(tmp_path / "shap_vs_scorecard.csv")
    assert {"shap_rank", "iv_rank", "points_rank"} <= set(cmp.columns)
    assert len(list((tmp_path / "ch").glob("shap_dependence_*.png"))) == 3


def _synthetic_outputs(d):
    dec = pd.DataFrame({"decile": range(1, 11), "n": 100, "bads": np.arange(1, 11) * 2,
                        "bad_rate": np.arange(1, 11) * 0.02, "mean_pd": np.arange(1, 11) * 0.019})
    for m in MODELS:
        dec.to_csv(d / f"deciles_{m}_oot1.csv", index=False)
    for s in SPLITS:
        dec.rename(columns={"bad_rate": "observed"}).to_csv(d / f"calibration_sc_full_{s}.csv", index=False)
    pd.DataFrame({"quarter": ["2013Q1", "2013Q2", "2013Q3"], "psi": [0.02, 0.06, 0.12]}).to_csv(d / "psi_by_quarter.csv", index=False)
    vint = pd.DataFrame({"model": ["sc_full"] * 3 + ["sc_indep"] * 3, "vintage": [2010, 2011, 2012] * 2, "n": 1000,
                         "bad_rate": [.05, .04, .05] * 2, "mean_pd": .05, "auc": .7, "gini": [.4, .38, .35, .3, .28, .25]})
    vint.to_csv(d / "by_vintage.csv", index=False)
    pd.DataFrame({"variable": list("AABB"), "bin_id": [1, 2, 1, 2], "points": [10, 30, 5, 8], "woe": [-.3, .4, -.1, .1],
                  "iv_contrib": [.04, .05, .01, .01], "model": "sc_full"}).to_csv(d / "scorecard_points.csv", index=False)
    pd.read_csv(d / "scorecard_points.csv").to_csv(d / "binning_report.csv", index=False)
    rows = [dict(test="gini", model=m, split=s, value=v, threshold=0.4, light="Green")
            for m, v in (("sc_full", .45), ("sc_indep", .38)) for s in SPLITS]
    pd.DataFrame(rows).to_csv(d / "validation_results.csv", index=False)
    pd.DataFrame({"split": "oot", "component": ["lgd_expected"] * 3 + ["ccf"] * 3, "decile": [1, 2, 3] * 2, "n": 50,
                  "mean_pred": [.8, .85, .9] * 2, "mean_actual": [.82, .84, .91] * 2}).to_csv(d / "lgd_ead_deciles.csv", index=False)
    pd.DataFrame({"vintage": 2013, "grade": ["A", "B", "C", "ALL"], "el": [1e6, 2e6, 3e6, 6e6], "realized_loss": [1.1e6, 1.9e6, 3.5e6, 6.5e6],
                  "sample": "out_of_time", "status": ""}).to_csv(d / "el_backtest_12m.csv", index=False)
    pd.DataFrame({"scenario": ["dti +5", "int_rate +2", "pd x1.25"], "model": "sc_full", "delta_el_pct": [4.0, 12.0, 25.0]}
                 ).to_csv(d / "sensitivity.csv", index=False)
    pd.DataFrame({"kind": "csi", "model": "all", "split": "oot1", "name": list("AB"), "psi": [0.02, 0.2], "light": "Green"}).to_csv(d / "psi_csi.csv", index=False)
    pd.DataFrame({"feature": ["a", "b"], "mean_abs_shap": [0.5, 0.2]}).to_csv(d / "shap_importance.csv", index=False)


def test_charts_smoke_on_synthetic_outputs(tmp_path):
    _synthetic_outputs(tmp_path)
    paths = charts.make_all_charts(out_dir=tmp_path, chart_dir=tmp_path / "charts")
    assert len(paths) >= 14
    assert all(p.exists() and p.stat().st_size > 1000 and p.parent == tmp_path / "charts" for p in paths)


def test_charts_skip_missing_inputs_gracefully(tmp_path):
    assert charts.make_all_charts(out_dir=tmp_path, chart_dir=tmp_path / "charts") == []
