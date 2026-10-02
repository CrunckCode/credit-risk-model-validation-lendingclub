import logging

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from credit_validation import stats_models as SM


@pytest.fixture(scope="module")
def sim():
    rng = np.random.default_rng(3)
    n = 3000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n) * 2 + 1, "c": rng.integers(0, 2, n).astype(float)})
    z = -0.5 + 0.8 * X["a"] - 0.3 * X["b"] + 0.6 * X["c"]
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    yc = 1.0 + 2.0 * X["a"] - 1.0 * X["b"] + rng.normal(size=n)
    return X, y, yc


def test_logit_matches_statsmodels(sim):
    X, y, _ = sim
    m = SM.LogisticRegressionWithPValues().fit(X, y)
    ref = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    np.testing.assert_allclose(m.beta_, ref.params.to_numpy(), atol=1e-5)
    np.testing.assert_allclose(m.se_, ref.bse.to_numpy(), atol=1e-5)
    np.testing.assert_allclose(m.z_, ref.tvalues.to_numpy(), atol=1e-4)
    np.testing.assert_allclose(m.p_values_, ref.pvalues.to_numpy(), atol=1e-5)
    assert list(m.summary()["term"]) == ["const", "a", "b", "c"]


def test_logit_intercept_in_information_matrix(sim):
    X, y, _ = sim
    m = SM.LogisticRegressionWithPValues().fit(X, y)
    assert len(m.se_) == X.shape[1] + 1 and np.isfinite(m.se_[0])


def test_linear_matches_statsmodels(sim):
    X, _, yc = sim
    m = SM.LinearRegressionWithPValues().fit(X, yc)
    ref = sm.OLS(yc, sm.add_constant(X)).fit()
    np.testing.assert_allclose(m.beta_, ref.params.to_numpy(), atol=1e-8)
    np.testing.assert_allclose(m.se_, ref.bse.to_numpy(), atol=1e-8)
    np.testing.assert_allclose(m.t_, ref.tvalues.to_numpy(), atol=1e-6)
    np.testing.assert_allclose(m.p_values_, ref.pvalues.to_numpy(), atol=1e-8)
    assert m.dof_ == len(yc) - X.shape[1] - 1
    np.testing.assert_allclose(m.predict(X), ref.predict(sm.add_constant(X)), atol=1e-8)


def test_align_columns_adds_zero_and_warns(caplog):
    df = pd.DataFrame({"x": [1.0, 2.0]})
    with caplog.at_level(logging.WARNING):
        out = SM.align_columns(df, ["x", "y", "z"])
    assert list(out.columns) == ["x", "y", "z"] and (out[["y", "z"]] == 0).all().all()
    assert "2 column(s)" in caplog.text


def test_align_columns_silent_when_complete(caplog):
    with caplog.at_level(logging.WARNING):
        SM.align_columns(pd.DataFrame({"x": [1.0], "y": [2.0]}), ["y", "x"])
    assert caplog.text == ""
