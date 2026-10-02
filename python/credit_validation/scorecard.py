"""WoE logistic scorecard: binning, iterative variable elimination, points scaling."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config
from . import columns as C
from .binning import fit_numeric_bins, fit_categorical_bins

# ===== CONFIG (user inputs) =====
MAX_ITERATIONS = 200
CONST = "const"
# ===== END CONFIG =====


def _is_categorical(s: pd.Series, name: str) -> bool:
    return name in C.CATEGORICAL_FEATURES or not pd.api.types.is_numeric_dtype(s)


def woe_matrix(df: pd.DataFrame, specs: dict) -> pd.DataFrame:
    return pd.DataFrame({v: sp.transform(df[v]) for v, sp in specs.items()}, index=df.index)


def vif_table(W: pd.DataFrame) -> dict:
    """VIF from the diagonal of the inverse correlation matrix, equal to 1/(1-R2) per variable."""
    if W.shape[1] < 2:
        return {c: 1.0 for c in W.columns}
    corr = np.corrcoef(W.to_numpy(dtype=np.float64), rowvar=False)
    return dict(zip(W.columns, np.diag(np.linalg.pinv(corr)).tolist()))


@dataclass
class Scorecard:
    specs: dict
    variables: list
    coefs: dict
    intercept: float
    factor: float
    offset: float
    iv: dict
    vif: dict
    pvalues: dict
    intercept_pvalue: float = float("nan")
    dropped: list = field(default_factory=list)      # (variable, reason) in elimination order
    n_train: int = 0

    @property
    def coef(self) -> dict:                           # alias for sibling-style callers
        return self.coefs

    def margin(self, df: pd.DataFrame) -> np.ndarray:
        z = np.full(len(df), self.intercept, dtype=np.float64)
        for v in self.variables:
            z += self.coefs[v] * self.specs[v].transform(df[v])
        return z

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.margin(df)))

    def points(self, df: pd.DataFrame) -> np.ndarray:
        """Total score: Offset - Factor * margin, equal to the sum of the per-variable points."""
        return self.offset - self.factor * self.margin(df)

    def points_by_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        k = len(self.variables)
        cols = {v: -(self.coefs[v] * self.specs[v].transform(df[v]) + self.intercept / k) * self.factor + self.offset / k
                for v in self.variables}
        return pd.DataFrame(cols, index=df.index)

    def export_points_table(self) -> pd.DataFrame:
        k = len(self.variables)
        rows = []
        for v in self.variables:
            t = self.specs[v].table()
            t["points"] = -(self.coefs[v] * t["woe"] + self.intercept / k) * self.factor + self.offset / k
            t["coef"] = self.coefs[v]
            t.insert(0, "variable", v)
            rows.append(t)
        out = pd.concat(rows, ignore_index=True)
        return out[["variable", "bin_id", "lo", "hi", "categories", "woe", "coef", "points"]]


def scaling(pdo=None, base_score=None, base_odds=None):
    pdo = config.PDO if pdo is None else pdo
    base_score = config.BASE_SCORE if base_score is None else base_score
    base_odds = config.BASE_ODDS if base_odds is None else base_odds
    factor = pdo / np.log(2.0)
    return float(factor), float(base_score - factor * np.log(base_odds))


def fit_bins_for(df, y, features, min_share=None, max_bins=None, cat_min_share=None, max_groups=None) -> dict:
    min_share = config.BIN_MIN_SHARE if min_share is None else min_share
    max_bins = config.BIN_MAX_BINS if max_bins is None else max_bins
    cat_min_share = config.CAT_MIN_SHARE if cat_min_share is None else cat_min_share
    max_groups = config.CAT_MAX_GROUPS if max_groups is None else max_groups
    y = np.asarray(y, dtype=np.float64)
    specs = {}
    for f in features:
        if _is_categorical(df[f], f):
            specs[f] = fit_categorical_bins(df[f], y, min_share=cat_min_share, max_groups=max_groups, variable=f)
        else:
            specs[f] = fit_numeric_bins(df[f].to_numpy(), y, prebins=config.BIN_PREBINS, min_share=min_share,
                                        max_bins=max_bins, variable=f)
    return specs


def _drop_correlated(W, ivs, corr_max, dropped):
    keep = list(W.columns)
    corr = W.corr().abs()
    while len(keep) > 1:
        sub = corr.loc[keep, keep].to_numpy().copy()
        np.fill_diagonal(sub, 0.0)
        i, j = np.unravel_index(np.argmax(sub), sub.shape)
        if sub[i, j] <= corr_max:
            break
        weaker, stronger = (keep[i], keep[j]) if ivs[keep[i]] < ivs[keep[j]] else (keep[j], keep[i])
        dropped.append((weaker, f"corr {sub[i, j]:.2f} with {stronger}"))
        keep.remove(weaker)
    return keep


def fit_scorecard(df, y_col, features, iv_min=None, p_max=None, vif_max=None, corr_max=None,
                  min_share=None, max_bins=None, specs=None) -> Scorecard:
    iv_min = config.IV_MIN if iv_min is None else iv_min
    p_max = config.P_MAX if p_max is None else p_max
    vif_max = config.VIF_MAX if vif_max is None else vif_max
    corr_max = config.CORR_MAX if corr_max is None else corr_max
    y = df[y_col].to_numpy(dtype=np.float64)
    features = list(features)
    if specs is None:
        specs = fit_bins_for(df, y, features, min_share=min_share, max_bins=max_bins)
    ivs = {f: specs[f].iv for f in features}
    dropped = [(f, f"IV {ivs[f]:.4f} < {iv_min}") for f in features if ivs[f] < iv_min]
    keep = [f for f in features if ivs[f] >= iv_min]
    W = woe_matrix(df, {f: specs[f] for f in keep})
    keep = _drop_correlated(W, ivs, corr_max, dropped)

    fit = None
    for _ in range(MAX_ITERATIONS):
        fit = sm.Logit(y, sm.add_constant(W[keep], has_constant="add")).fit(disp=0, maxiter=100)
        pos = [f for f in keep if fit.params[f] >= 0]
        if pos:
            f = min(pos, key=lambda v: ivs[v])
            dropped.append((f, "wrong sign"))
            keep.remove(f)
            continue
        hi = [f for f in keep if fit.pvalues[f] > p_max]
        if hi:
            f = max(hi, key=lambda v: fit.pvalues[v])
            dropped.append((f, f"p {fit.pvalues[f]:.3f} > {p_max}"))
            keep.remove(f)
            continue
        vif = vif_table(W[keep])
        big = [f for f in keep if vif[f] > vif_max]
        if big:
            f = min(big, key=lambda v: ivs[v])
            dropped.append((f, f"VIF {vif[f]:.1f} > {vif_max}"))
            keep.remove(f)
            continue
        break
    else:
        raise RuntimeError("scorecard elimination did not converge")
    factor, offset = scaling()
    return Scorecard(
        specs={f: specs[f] for f in keep}, variables=list(keep), coefs={f: float(fit.params[f]) for f in keep},
        intercept=float(fit.params[CONST]), factor=factor, offset=offset, iv={f: float(ivs[f]) for f in keep},
        vif=vif_table(W[keep]), pvalues={f: float(fit.pvalues[f]) for f in keep},
        intercept_pvalue=float(fit.pvalues[CONST]), dropped=dropped, n_train=len(df),
    )
