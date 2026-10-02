"""SHAP explanations for the LightGBM challenger and a comparison with the scorecard's own variable ranking.

SHAP here explains the raw GBM margin (log-odds), not the calibrated probability, so base + sum(SHAP) equals
predict_margin and not predict_pd.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import charts, config

log = logging.getLogger(__name__)

# ===== CONFIG (user inputs) =====
FEATURE_COL, IMPORTANCE_COL, RANK_COL = "feature", "mean_abs_shap", "rank"
VARIABLE_COL, SHAP_RANK_COL, IV_COL, IV_RANK_COL = "variable", "shap_rank", "iv", "iv_rank"
POINTS_RANGE_COL, POINTS_RANK_COL, GAP_COL = "points_range", "points_rank", "rank_gap_shap_vs_iv"
POINTS_FILE, BINNING_FILE = "scorecard_points.csv", "binning_report.csv"
IMPORTANCE_FILE, COMPARE_FILE = "shap_importance.csv", "shap_vs_scorecard.csv"
POINTS_VAR_COL, POINTS_COL, IV_CONTRIB_COL = "variable", "points", "iv_contrib"
MODEL_COL, COMPARE_MODEL = "model", "sc_full"
DEPENDENCE_TOP_K = 6
ADDITIVITY_TOL = 1e-4
# ===== END CONFIG =====


def prepare(bundle, X: pd.DataFrame) -> pd.DataFrame:
    """Model input view of X (bundle's own encoder if it exposes one, else the feature columns)."""
    if hasattr(bundle, "design"):
        return bundle.design(X)
    return X[list(bundle.features)]


def _tree_model(bundle):
    est = bundle.estimator
    return getattr(est, "booster_", est)


def shap_matrix(bundle, X: pd.DataFrame):
    """(values, base) of TreeExplainer on the raw margin; values shape (n, p)."""
    import shap
    M = prepare(bundle, X)
    ex = shap.TreeExplainer(_tree_model(bundle))
    raw = ex.shap_values(M, check_additivity=False)
    if isinstance(raw, list):                       # older shap returns one array per class
        raw = raw[-1]
    vals = np.asarray(raw, dtype=np.float64)
    if vals.ndim == 3:
        vals = vals[:, :, -1]
    base = np.asarray(ex.expected_value, dtype=np.float64).ravel()
    return vals, float(base[-1]), list(M.columns)


def additivity_gap(bundle, X: pd.DataFrame, vals=None, base=None) -> float:
    """Max |base + sum(SHAP) - predict_margin| over rows."""
    if vals is None:
        vals, base, _ = shap_matrix(bundle, X)
    recon = base + vals.sum(axis=1)
    return float(np.max(np.abs(recon - np.asarray(bundle.predict_margin(X), dtype=np.float64))))


def assert_additive(bundle, X, tol=ADDITIVITY_TOL):
    gap = additivity_gap(bundle, X)
    assert gap <= tol, f"SHAP additivity gap {gap:.2e} exceeds {tol}"
    return gap


def global_importance(vals: np.ndarray, names) -> pd.DataFrame:
    imp = pd.DataFrame({FEATURE_COL: list(names), IMPORTANCE_COL: np.abs(vals).mean(axis=0)})
    imp = imp.sort_values(IMPORTANCE_COL, ascending=False).reset_index(drop=True)
    imp[RANK_COL] = np.arange(1, len(imp) + 1)
    return imp


def compare_with_scorecard(importance: pd.DataFrame, points: pd.DataFrame | None, binning: pd.DataFrame | None) -> pd.DataFrame:
    """Rank of every variable by mean |SHAP| versus by scorecard IV and by points range."""
    if points is not None and MODEL_COL in points.columns:
        points = points[points[MODEL_COL] == COMPARE_MODEL]
    if binning is not None and MODEL_COL in binning.columns:
        binning = binning[binning[MODEL_COL] == COMPARE_MODEL]
    cmp = importance[[FEATURE_COL, IMPORTANCE_COL, RANK_COL]].rename(
        columns={FEATURE_COL: VARIABLE_COL, IMPORTANCE_COL: "mean_abs_shap", RANK_COL: SHAP_RANK_COL})
    if binning is not None and {VARIABLE_COL, IV_CONTRIB_COL} <= set(binning.columns):
        iv = binning.groupby(VARIABLE_COL)[IV_CONTRIB_COL].sum().rename(IV_COL).reset_index()
        cmp = cmp.merge(iv, on=VARIABLE_COL, how="outer")
    else:
        cmp[IV_COL] = np.nan
    if points is not None and {POINTS_VAR_COL, POINTS_COL} <= set(points.columns):
        pr = points.groupby(POINTS_VAR_COL)[POINTS_COL].agg(lambda s: s.max() - s.min()).rename(POINTS_RANGE_COL).reset_index()
        pr = pr.rename(columns={POINTS_VAR_COL: VARIABLE_COL})
        cmp = cmp.merge(pr, on=VARIABLE_COL, how="outer")
    else:
        cmp[POINTS_RANGE_COL] = np.nan
    cmp[SHAP_RANK_COL] = cmp["mean_abs_shap"].rank(ascending=False, method="min")
    cmp[IV_RANK_COL] = cmp[IV_COL].rank(ascending=False, method="min")
    cmp[POINTS_RANK_COL] = cmp[POINTS_RANGE_COL].rank(ascending=False, method="min")
    cmp[GAP_COL] = cmp[IV_RANK_COL] - cmp[SHAP_RANK_COL]
    cmp["in_scorecard"] = cmp[POINTS_RANGE_COL].notna()
    return cmp.sort_values(SHAP_RANK_COL, na_position="last").reset_index(drop=True)


def rank_agreement(cmp: pd.DataFrame) -> float:
    """Spearman correlation of SHAP rank and IV rank over variables present in both."""
    both = cmp.dropna(subset=[SHAP_RANK_COL, IV_RANK_COL])
    if len(both) < 3:
        return float("nan")
    return float(both[SHAP_RANK_COL].corr(both[IV_RANK_COL], method="spearman"))


def dependence_plots(vals, X: pd.DataFrame, names, importance: pd.DataFrame, top_k=DEPENDENCE_TOP_K, chart_dir=config.CHART_DIR) -> list:
    chart_dir = Path(chart_dir)
    idx = {n: i for i, n in enumerate(names)}
    paths = []
    for f in importance[FEATURE_COL].head(top_k):
        if f not in idx or f not in X.columns:
            continue
        p = chart_dir / f"shap_dependence_{f}.png"
        charts.shap_dependence(vals[:, idx[f]], X[f].reset_index(drop=True), f, p)
        paths.append(p)
    return paths


def _read_optional(out_dir: Path, name: str):
    p = out_dir / name
    return pd.read_csv(p) if p.exists() else None


def run_explain(bundle_lgbm, oot1_df: pd.DataFrame, out_dir=config.OUT_DIR, chart_dir=config.CHART_DIR, seed: int = config.SEED) -> dict:
    """Sample OOT1 rows, compute SHAP, write importance and SHAP-vs-scorecard csvs and dependence plots."""
    out_dir, chart_dir = Path(out_dir), Path(chart_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)
    sample = oot1_df.sample(n=min(config.SHAP_SAMPLE, len(oot1_df)), random_state=seed).reset_index(drop=True)
    vals, base, names = shap_matrix(bundle_lgbm, sample)
    gap = additivity_gap(bundle_lgbm, sample, vals, base)
    imp = global_importance(vals, names)
    imp.to_csv(out_dir / IMPORTANCE_FILE, index=False)
    cmp = compare_with_scorecard(imp, _read_optional(out_dir, POINTS_FILE), _read_optional(out_dir, BINNING_FILE))
    cmp.to_csv(out_dir / COMPARE_FILE, index=False)
    paths = dependence_plots(vals, prepare(bundle_lgbm, sample), names, imp, chart_dir=chart_dir)
    charts.shap_importance_chart(imp, chart_dir / "shap_importance.png")
    return dict(importance=imp, comparison=cmp, additivity_gap=gap, rank_agreement=rank_agreement(cmp), dependence_paths=paths)
