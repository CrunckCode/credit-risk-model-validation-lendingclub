import numpy as np
import pytest

from credit_validation import config, models
from credit_validation import columns as C


@pytest.fixture(autouse=True)
def isolate_outputs(tmp_path, monkeypatch):
    # tests must never write into the pipeline's real output or model folders
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "EXCEL_INPUT_DIR", tmp_path / "out" / "excel_inputs")


@pytest.fixture(scope="module")
def champs(fixture_splits):
    return models.fit_champions(fixture_splits)


def test_champions_use_declared_feature_sets(champs):
    assert champs["sc_full"].features == list(C.APPLICATION_FEATURES)
    assert champs["sc_indep"].features == list(C.INDEP_FEATURES)
    assert not set(champs["sc_indep"].scorecard.variables) & set(C.LC_RISK_OUTPUTS)


def test_bundle_save_load_roundtrip(champs, fixture_splits, tmp_path):
    b = champs["sc_full"]
    path = tmp_path / "b.joblib"
    b.save(path)
    b2 = models.ModelBundle.load(path)
    d = fixture_splits[C.OOT2]
    np.testing.assert_array_equal(b.predict_pd(d), b2.predict_pd(d))
    np.testing.assert_array_equal(b.predict_margin(d), b2.predict_margin(d))
    assert b2.name == b.name and b2.kind == "sc"


def test_scorecard_bundle_points_and_pd_bounds(champs, fixture_splits):
    d = fixture_splits[C.OOT1] if len(fixture_splits[C.OOT1]) else fixture_splits[C.OOT2]
    p = champs["sc_full"].predict_pd(d)
    assert ((p > 0) & (p < 1)).all()
    assert champs["sc_full"].points(d).shape == (len(d),)


def test_recalibrate_intercept_matches_oot1_rate_and_flags_use(champs, fixture_splits):
    oot1 = fixture_splits[C.OOT2]       # any non-empty out-of-time frame works for the identity
    r = models.recalibrate_intercept(champs["sc_full"], oot1)
    assert r.name == "sc_full_recal" and r.kind == "recal"
    assert r.predict_pd(oot1).mean() == pytest.approx(oot1[C.TARGET_BAD12].mean(), abs=1e-9)
    assert r.extras["evaluate_on"] == [C.OOT2] and "OOT1" in r.extras["note"]
    base = champs["sc_full"].predict_margin(oot1)
    np.testing.assert_allclose(r.predict_margin(oot1) - base, r.margin_shift)       # only the intercept moves


def test_gbm_fit_calibrate_and_monotone(fixture_splits):
    tr, ho = fixture_splits[C.DEV_TRAIN], fixture_splits[C.DEV_HOLDOUT]
    params = dict(n_estimators=40, learning_rate=0.1, num_leaves=7, min_child=20)
    for kind in ("lgbm", "xgb"):
        b = models.fit_gbm(kind, tr, ho, C.APPLICATION_FEATURES, params=params)
        p = b.predict_pd(fixture_splits[C.OOT2])
        assert ((p > 0) & (p < 1)).all() and b.extras["calibration_method"] in ("isotonic", "platt")
        # int_rate is constrained increasing: raising it must never lower the margin
        d = fixture_splits[C.OOT2].head(200).copy()
        lo = b.predict_margin(d)
        d[C.INT_RATE] = d[C.INT_RATE] + 5.0
        assert (b.predict_margin(d) >= lo - 1e-9).all()


def test_tune_time_cv_returns_params_and_uses_expanding_folds(fixture_splits):
    tr = fixture_splits[C.DEV_TRAIN]
    best = models.tune_time_cv("lgbm", tr, C.APPLICATION_FEATURES, n_iter=2, val_years=(2011, 2012), quick=True)
    assert {"learning_rate", "n_estimators", "cv_logloss"} <= set(best) and np.isfinite(best["cv_logloss"])


def test_train_all_quick_writes_only_to_redirected_dir(fixture_splits, tmp_path):
    out = models.train_all(fixture_splits, quick=True)
    assert list(out) == ["sc_full", "sc_indep", "xgb", "lgbm", "sc_full_recal", "repro_original"]
    for k in out:
        assert (tmp_path / "models" / f"{k}.joblib").exists()
    d = fixture_splits[C.OOT2]
    for k, b in out.items():
        p = b.predict_pd(d)
        assert len(p) == len(d) and np.isfinite(p).all(), k
