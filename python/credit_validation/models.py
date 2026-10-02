"""Model bundles (scorecards, XGBoost, LightGBM, reproduced original), time-aware tuning, calibration, training driver."""
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from . import config
from . import columns as C
from . import repro_original
from . import scorecard as sc_mod
from .features import assert_no_leakage
from .stats_models import align_columns

# ===== CONFIG (user inputs) =====
KIND_SC, KIND_XGB, KIND_LGBM, KIND_RECAL, KIND_REPRO = "sc", "xgb", "lgbm", "recal", "repro"
NAME_SC_FULL, NAME_SC_INDEP, NAME_XGB, NAME_LGBM = "sc_full", "sc_indep", "xgb", "lgbm"
NAME_RECAL, NAME_REPRO = "sc_full_recal", "repro_original"
TARGET = C.TARGET_BAD12
CALIB_METHODS = ("isotonic", "platt")
PROB_CLIP = 1e-6
EARLY_STOP = 30
TUNE_MAX_ROWS = 80_000
QUICK_TUNE_ITER, QUICK_MAX_TREES, FULL_MAX_TREES = 3, 100, 500
DEFAULT_PARAMS = dict(learning_rate=0.08, max_depth=4, num_leaves=15, min_child=50, subsample=0.8,
                      colsample=0.8, reg_lambda=5.0, n_estimators=250)
SEARCH = dict(learning_rate=(0.03, 0.15), max_depth=(3, 6), num_leaves=(8, 48), min_child=(30, 300),
              subsample=(0.6, 1.0), colsample=(0.6, 1.0), reg_lambda=(1.0, 30.0))
N_JOBS = max((os.cpu_count() or 2) - 1, 1)
REPRO_MAX_ITER = 1000
# ===== END CONFIG =====

warnings.filterwarnings("ignore", message=".*eval_set.*")


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


@dataclass
class Calibrator:
    method: str
    model: object

    def transform(self, margin):
        if self.method == "isotonic":
            p = self.model.predict(_sigmoid(margin))
        else:
            p = self.model.predict_proba(np.asarray(margin).reshape(-1, 1))[:, 1]
        return np.clip(p, PROB_CLIP, 1 - PROB_CLIP)


def _fit_calibrator(method, margin, y):
    if method == "isotonic":
        m = IsotonicRegression(out_of_bounds="clip", increasing=True).fit(_sigmoid(margin), y)
    else:
        m = LogisticRegression(C=1e6, max_iter=500).fit(np.asarray(margin).reshape(-1, 1), y)
    return Calibrator(method, m)


@dataclass
class Encoder:
    """Categories learned on TRAIN so every later frame is encoded identically."""
    features: list
    categories: dict

    @classmethod
    def fit(cls, df, features):
        cats = {f: sorted(pd.Series(df[f]).dropna().astype(str).unique()) for f in features if f in C.CATEGORICAL_FEATURES}
        return cls(list(features), cats)

    def category_frame(self, df):
        out = {}
        for f in self.features:
            if f in self.categories:
                s = df[f].astype(object)
                out[f] = pd.Categorical(s.where(s.isna(), s.astype(str)), categories=self.categories[f])
            else:
                out[f] = df[f].to_numpy(dtype=np.float32)
        return pd.DataFrame(out, index=df.index)

    def onehot_frame(self, df):
        out = {}
        for f in self.features:
            if f in self.categories:
                s = df[f].astype(str).to_numpy()
                for c in self.categories[f]:
                    out[f"{f}={c}"] = (s == c).astype(np.float32)
            else:
                out[f] = df[f].to_numpy(dtype=np.float32)
        return pd.DataFrame(out, index=df.index)


@dataclass
class ReproEstimator:
    """Original-notebook dummy-design logit (L2, C=1, includes the loan-age proxy) refit on the random split."""
    model: object
    columns: list

    def margin(self, df):
        X = align_columns(repro_original.build_dummy_design(df, True), self.columns)
        return np.asarray(self.model.decision_function(X), dtype=np.float64)


@dataclass
class ModelBundle:
    name: str
    kind: str
    features: list
    estimator: object
    calibrator: object = None
    encoder: object = None
    margin_shift: float = 0.0           # intercept-only remediation (recal bundle)
    extras: dict = field(default_factory=dict)

    @property
    def scorecard(self):
        return self.estimator if isinstance(self.estimator, sc_mod.Scorecard) else None

    def design(self, df):
        """Matrix exactly as the estimator sees it (use for SHAP)."""
        if self.kind == KIND_LGBM:
            return self.encoder.category_frame(df)
        if self.kind == KIND_XGB:
            return self.encoder.onehot_frame(df)
        return df

    def predict_margin(self, df) -> np.ndarray:
        if self.kind == KIND_LGBM:
            m = np.asarray(self.estimator.predict(self.design(df), raw_score=True), dtype=np.float64)
        elif self.kind == KIND_XGB:
            m = np.asarray(self.estimator.predict(self.design(df), output_margin=True), dtype=np.float64)
        else:
            m = np.asarray(self.estimator.margin(df), dtype=np.float64)
        return m + self.margin_shift

    def predict_pd(self, df) -> np.ndarray:
        m = self.predict_margin(df)
        return _sigmoid(m) if self.calibrator is None else self.calibrator.transform(m)

    def points(self, df):
        """Scorecard points when the estimator is a scorecard, else None."""
        return None if self.scorecard is None else self.scorecard.points(df)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)


# ---------- champions ----------
def fit_champions(splits, train=None) -> dict:
    tr = splits[C.DEV_TRAIN] if train is None else train
    out = {}
    for name, feats in ((NAME_SC_FULL, C.APPLICATION_FEATURES), (NAME_SC_INDEP, C.INDEP_FEATURES)):
        assert_no_leakage(feats)
        sc = sc_mod.fit_scorecard(tr, TARGET, list(feats))
        out[name] = ModelBundle(name, KIND_SC, list(feats), sc)
    return out


# ---------- calibration ----------
def calibrate(bundle: ModelBundle, holdout, method: str = "auto") -> ModelBundle:
    """Platt vs isotonic chosen by cross-fitted Brier on the holdout, then refit on the full holdout."""
    margin = bundle.predict_margin(holdout)
    y = holdout[TARGET].to_numpy(dtype=np.float64)
    if method == "auto":
        idx = np.random.default_rng(config.SEED).permutation(len(y))
        halves = np.array_split(idx, 2)
        score = {}
        for m in CALIB_METHODS:
            errs = []
            for a, b in ((0, 1), (1, 0)):
                c = _fit_calibrator(m, margin[halves[a]], y[halves[a]])
                errs.append(np.mean((c.transform(margin[halves[b]]) - y[halves[b]]) ** 2))
            score[m] = float(np.mean(errs))
        method = min(score, key=score.get)
        bundle.extras["calibration_brier"] = score
    bundle.calibrator = _fit_calibrator(method, margin, y)
    bundle.extras["calibration_method"] = method
    return bundle


def recalibrate_intercept(sc: ModelBundle, oot1) -> ModelBundle:
    """Remediation challenger: shift only the logit intercept so mean PD equals the OOT1 bad rate.

    Uses OOT1 outcomes, so it must be evaluated on OOT2 only (OOT1 would be in-sample).
    """
    m = sc.predict_margin(oot1)
    y = oot1[TARGET].to_numpy(dtype=np.float64)
    delta = brentq(lambda d: _sigmoid(m + d).mean() - y.mean(), -10.0, 10.0)
    b = ModelBundle(NAME_RECAL, KIND_RECAL, list(sc.features), sc.estimator, margin_shift=float(delta))
    b.extras.update(intercept_shift=float(delta), fitted_on=C.OOT1, evaluate_on=[C.OOT2],
                    note="intercept fitted with OOT1 outcomes; OOT1 metrics are in-sample, use OOT2 only")
    return b


# ---------- GBMs ----------
def _constraints(cols, monotone):
    return [int(monotone.get(c, 0)) for c in cols]


def _lib_params(kind, p, n_est):
    if kind == KIND_XGB:
        return dict(learning_rate=p["learning_rate"], max_depth=int(p["max_depth"]), min_child_weight=p["min_child"] / 10.0,
                    subsample=p["subsample"], colsample_bytree=p["colsample"], reg_lambda=p["reg_lambda"],
                    n_estimators=int(n_est), tree_method="hist", n_jobs=N_JOBS, eval_metric="logloss", random_state=config.SEED)
    return dict(learning_rate=p["learning_rate"], num_leaves=int(p["num_leaves"]), min_child_samples=int(p["min_child"]),
                subsample=p["subsample"], subsample_freq=1, colsample_bytree=p["colsample"], reg_lambda=p["reg_lambda"],
                n_estimators=int(n_est), n_jobs=N_JOBS, random_state=config.SEED, verbose=-1,
                monotone_constraints_method="intermediate")


def _make_estimator(kind, p, cols, monotone, n_est, early_stop=None):
    mc = _constraints(cols, monotone)
    lp = _lib_params(kind, p, n_est)
    if kind == KIND_XGB:
        import xgboost as xgb
        if early_stop:
            lp["early_stopping_rounds"] = early_stop
        return xgb.XGBClassifier(monotone_constraints=tuple(mc), **lp)
    import lightgbm as lgb
    return lgb.LGBMClassifier(monotone_constraints=mc, **lp)


def _design(kind, enc, df):
    return enc.category_frame(df) if kind == KIND_LGBM else enc.onehot_frame(df)


def _fit_est(kind, est, X, y, Xv=None, yv=None):
    if Xv is None:
        return est.fit(X, y)
    if kind == KIND_XGB:
        return est.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    import lightgbm as lgb
    return est.fit(X, y, eval_set=[(Xv, yv)], eval_metric="binary_logloss",
                   callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])


def _best_iter(kind, est):
    return int(est.best_iteration) + 1 if kind == KIND_XGB else int(est.best_iteration_ or est.n_estimators)


def fit_gbm(kind, train, holdout, features, monotone=None, params=None, name=None) -> ModelBundle:
    """Fit on train with the tuned tree count; calibrate on the holdout (never on OOT)."""
    assert_no_leakage(features)
    p = {**DEFAULT_PARAMS, **(params or {})}
    mono = C.MONOTONE_SIGNS if monotone is None else monotone
    enc = Encoder.fit(train, features)
    X = _design(kind, enc, train)
    est = _make_estimator(kind, p, list(X.columns), mono, p["n_estimators"])
    _fit_est(kind, est, X, train[TARGET].to_numpy())
    b = ModelBundle(name or kind, kind, list(features), est, None, enc)
    b.extras["params"] = {k: p[k] for k in DEFAULT_PARAMS}
    return calibrate(b, holdout) if holdout is not None else b


def tune_time_cv(kind, train, features, n_iter=None, val_years=None, quick=False, seed=None, monotone=None) -> dict:
    """Random search; expanding folds train on issue years before each validation year, early stopping on the fold."""
    n_iter = config.TUNE_ITER if n_iter is None else n_iter
    val_years = config.TUNE_VAL_YEARS if val_years is None else val_years
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    mono = C.MONOTONE_SIGNS if monotone is None else monotone
    enc = Encoder.fit(train, features)
    X = _design(kind, enc, train)
    y = train[TARGET].to_numpy()
    year = train[C.ISSUE_D].dt.year.to_numpy()
    folds = []
    for vy in val_years:
        tr, va = np.where(year < vy)[0], np.where(year == vy)[0]
        if len(tr) > TUNE_MAX_ROWS:
            tr = np.sort(rng.choice(tr, TUNE_MAX_ROWS, replace=False))
        if len(tr) and len(va) and y[tr].sum() and y[va].sum():
            folds.append((tr, va))
    max_trees = QUICK_MAX_TREES if quick else FULL_MAX_TREES
    best, best_loss = dict(DEFAULT_PARAMS), np.inf
    for _ in range(n_iter):
        cand = dict(learning_rate=float(np.exp(rng.uniform(np.log(SEARCH["learning_rate"][0]), np.log(SEARCH["learning_rate"][1])))),
                    max_depth=int(rng.integers(SEARCH["max_depth"][0], SEARCH["max_depth"][1] + 1)),
                    num_leaves=int(rng.integers(SEARCH["num_leaves"][0], SEARCH["num_leaves"][1] + 1)),
                    min_child=int(rng.integers(SEARCH["min_child"][0], SEARCH["min_child"][1] + 1)),
                    subsample=float(rng.uniform(*SEARCH["subsample"])), colsample=float(rng.uniform(*SEARCH["colsample"])),
                    reg_lambda=float(np.exp(rng.uniform(np.log(SEARCH["reg_lambda"][0]), np.log(SEARCH["reg_lambda"][1])))))
        if quick:
            cand["max_depth"], cand["num_leaves"] = min(cand["max_depth"], 3), min(cand["num_leaves"], 8)
            cand["learning_rate"] = max(cand["learning_rate"], 0.1)
        losses, iters = [], []
        for tr, va in folds:
            est = _make_estimator(kind, cand, list(X.columns), mono, max_trees, EARLY_STOP if kind == KIND_XGB else None)
            _fit_est(kind, est, X.iloc[tr], y[tr], X.iloc[va], y[va])
            pv = np.clip(est.predict_proba(X.iloc[va])[:, 1], PROB_CLIP, 1 - PROB_CLIP)
            yv = y[va]
            losses.append(-np.mean(yv * np.log(pv) + (1 - yv) * np.log(1 - pv)))
            iters.append(_best_iter(kind, est))
        loss = float(np.mean(losses))
        if loss < best_loss:
            best_loss, best = loss, {**cand, "n_estimators": int(max(np.median(iters) * 1.15, 20))}
    best["cv_logloss"] = best_loss
    return best


# ---------- reproduction of the original model ----------
def fit_repro(splits, train=None) -> ModelBundle:
    tr = splits[C.REPRO_TRAIN] if train is None else train
    X = repro_original.build_dummy_design(tr, True)
    mdl = LogisticRegression(C=repro_original.L2_C, max_iter=REPRO_MAX_ITER).fit(X, tr[C.TARGET_LIFETIME])
    b = ModelBundle(NAME_REPRO, KIND_REPRO, sorted(repro_original.ORIGINAL_COARSE_CLASSES), ReproEstimator(mdl, list(X.columns)))
    b.extras["target"] = C.TARGET_LIFETIME
    b.extras["note"] = "uses loan-age proxy mths_since_issue_d; trained on random 80/20 lifetime target"
    return b


# ---------- driver ----------
def train_all(splits, quick: bool = False) -> dict:
    tr = splits[C.DEV_TRAIN]
    rtr = splits[C.REPRO_TRAIN]
    if quick:
        tr = tr.sample(frac=config.quick_share(True), random_state=config.SEED)
        rtr = rtr.sample(frac=config.quick_share(True), random_state=config.SEED)
    ho = splits[C.DEV_HOLDOUT]
    out = fit_champions(splits, train=tr)
    n_iter = QUICK_TUNE_ITER if quick else config.TUNE_ITER
    for kind in (KIND_XGB, KIND_LGBM):
        tuned = tune_time_cv(kind, tr, C.APPLICATION_FEATURES, n_iter=n_iter, quick=quick)
        params = {k: v for k, v in tuned.items() if k in DEFAULT_PARAMS}
        if quick:
            params["n_estimators"] = min(params["n_estimators"], QUICK_MAX_TREES)
        b = fit_gbm(kind, tr, ho, C.APPLICATION_FEATURES, params=params, name=kind)
        b.extras["cv_logloss"] = tuned["cv_logloss"]
        out[kind] = b
    out[NAME_RECAL] = recalibrate_intercept(out[NAME_SC_FULL], splits[C.OOT1])
    out[NAME_REPRO] = fit_repro(splits, train=rtr)
    order = [NAME_SC_FULL, NAME_SC_INDEP, NAME_XGB, NAME_LGBM, NAME_RECAL, NAME_REPRO]
    out = {k: out[k] for k in order}
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for k, b in out.items():
        b.save(config.MODEL_DIR / f"{k}.joblib")
    return out
