"""Regression wrappers that expose Wald/t p-values (unpenalized, intercept included in the information matrix)."""
import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression

# ===== CONFIG (user inputs) =====
LOGIT_TOL, LOGIT_MAX_ITER = 1e-12, 5000
INTERCEPT_LABEL = "const"
# ===== END CONFIG =====

log = logging.getLogger(__name__)


def _names(X) -> list:
    return list(X.columns) if isinstance(X, pd.DataFrame) else [f"x{i}" for i in range(np.asarray(X).shape[1])]


def _with_const(X) -> np.ndarray:
    A = np.asarray(X, dtype=np.float64)
    return np.column_stack([np.ones(len(A)), A])


class LogisticRegressionWithPValues:
    """Unpenalized logit; Wald SE from inv(X'VX) over [1, X] so the intercept is part of the Fisher matrix."""

    def __init__(self):
        self.model = LogisticRegression(C=np.inf, solver="lbfgs", tol=LOGIT_TOL, max_iter=LOGIT_MAX_ITER)

    def fit(self, X, y):
        self.names_ = [INTERCEPT_LABEL] + _names(X)
        A = np.asarray(X, dtype=np.float64)
        self.model.fit(A, np.asarray(y))
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        beta = np.r_[self.intercept_, self.coef_.ravel()]
        D = _with_const(A)
        p = 1.0 / (1.0 + np.exp(-(D @ beta)))
        cov = np.linalg.pinv((D * (p * (1 - p))[:, None]).T @ D)
        self.beta_ = beta
        self.se_ = np.sqrt(np.diag(cov))
        self.z_ = beta / self.se_
        self.p_values_ = 2 * stats.norm.sf(np.abs(self.z_))
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(np.asarray(X, dtype=np.float64))

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({"term": self.names_, "coef": self.beta_, "se": self.se_, "z": self.z_, "p_value": self.p_values_})


class LinearRegressionWithPValues(LinearRegression):
    """OLS with classical t-tests; dof is n-p-1 and SE covers the intercept."""

    def fit(self, X, y, sample_weight=None):
        super().fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), sample_weight)
        A, yv = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
        self.names_ = [INTERCEPT_LABEL] + _names(X)
        n, p = A.shape
        D = _with_const(A)
        beta = np.r_[self.intercept_, self.coef_.ravel()]
        resid = yv - D @ beta
        self.dof_ = n - p - 1
        self.sigma2_ = float(resid @ resid) / self.dof_
        cov = self.sigma2_ * np.linalg.pinv(D.T @ D)
        self.beta_ = beta
        self.se_ = np.sqrt(np.diag(cov))
        self.t_ = beta / self.se_
        self.p_values_ = 2 * stats.t.sf(np.abs(self.t_), self.dof_)
        self.resid_ = resid
        self.design_ = D
        return self

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({"term": self.names_, "coef": self.beta_, "se": self.se_, "t": self.t_, "p_value": self.p_values_})


def align_columns(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Reindex to the training design; missing dummies become 0 but are logged so they never go unnoticed."""
    columns = list(columns)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        log.warning("align_columns: %d column(s) missing and filled with 0: %s", len(missing), missing[:10])
    return df.reindex(columns=columns, fill_value=0)
