"""Population stability: score PSI, characteristic stability index (CSI) and PSI by issue quarter.

Bins: decile edges taken from the development sample, searchsorted side="left" (lo exclusive, hi inclusive),
shares floored at PSI_FLOOR before the log. Numeric features get their own missing bin.
"""
import numpy as np
import pandas as pd

from . import config
from . import columns as C
from . import validation as V

# ===== CONFIG (user inputs) =====
PSI_FLOOR = 1e-4
PSI_BINS = 10
MISSING_KEY = "(missing)"
QUARTER_FIRST, QUARTER_LAST = "2010Q1", "2014Q4"
REF_SPLIT = C.DEV_TRAIN
QUARTER_SPLITS = (C.DEV_TRAIN, C.DEV_HOLDOUT, C.OOT1, C.OOT2)
OOT_FIRST_QUARTER = "2013Q1"
CSI_SPLITS = (C.OOT1, C.OOT2)
SKIP_CSI = (C.GRADE, C.SUB_GRADE)        # kept in CSI output; only the max over all features drives the light
# ===== END CONFIG =====


def psi_from_shares(e, a) -> float:
    e, a = np.maximum(np.asarray(e, float), PSI_FLOOR), np.maximum(np.asarray(a, float), PSI_FLOOR)
    return float(np.sum((a - e) * np.log(a / e)))


def dev_edges(expected, bins: int = PSI_BINS) -> np.ndarray:
    """Interior cut points from the development sample's quantiles (duplicates collapsed)."""
    x = np.asarray(expected, dtype=np.float64)
    x = x[~np.isnan(x)]
    return np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1]))


def shares(x, edges) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    return np.bincount(np.searchsorted(edges, x, side="left"), minlength=len(edges) + 1) / max(len(x), 1)


def psi(expected, actual, bins: int = PSI_BINS) -> float:
    edges = dev_edges(expected, bins)
    return psi_from_shares(shares(expected, edges), shares(actual, edges))


def psi_table(expected, actual, bins: int = PSI_BINS) -> pd.DataFrame:
    edges = dev_edges(expected, bins)
    e, a = shares(expected, edges), shares(actual, edges)
    lo, hi = np.r_[-np.inf, edges], np.r_[edges, np.inf]
    ef, af = np.maximum(e, PSI_FLOOR), np.maximum(a, PSI_FLOOR)
    return pd.DataFrame({"bin": np.arange(1, len(e) + 1), "lo": lo, "hi": hi, "dev_share": e, "act_share": a, "contrib": (af - ef) * np.log(af / ef)})


def _cat_shares(s: pd.Series, cats) -> np.ndarray:
    v = s.astype(object).where(s.notna(), MISSING_KEY).astype(str)
    return v.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()


def feature_psi(dev: pd.Series, cur: pd.Series) -> float:
    if pd.api.types.is_numeric_dtype(dev) and dev.name not in C.CATEGORICAL_FEATURES:
        edges = dev_edges(dev.to_numpy())
        e = np.r_[shares(dev, edges) * dev.notna().mean(), dev.isna().mean()]
        a = np.r_[shares(cur, edges) * cur.notna().mean(), cur.isna().mean()]
        return psi_from_shares(e, a)
    cats = sorted(set(dev.astype(object).where(dev.notna(), MISSING_KEY).astype(str)) | set(cur.astype(object).where(cur.notna(), MISSING_KEY).astype(str)))
    return psi_from_shares(_cat_shares(dev, cats), _cat_shares(cur, cats))


def csi(dev_df: pd.DataFrame, cur_df: pd.DataFrame, features) -> pd.DataFrame:
    rows = [dict(feature=f, psi=feature_psi(dev_df[f], cur_df[f])) for f in features]
    out = pd.DataFrame(rows)
    out["light"] = [V.light_low(v, "psi") for v in out["psi"]]
    return out


def quarter_label(issue: pd.Series) -> pd.Series:
    return issue.dt.to_period("Q").astype(str)


def psi_by_quarter(bundles: dict, splits: dict, scores: dict = None) -> pd.DataFrame:
    scores = scores or V.score_all(bundles, splits)
    rows = []
    for name, b in bundles.items():
        if (name, REF_SPLIT) not in scores:
            continue                                        # recal challenger has no development scores
        ref = scores[(name, REF_SPLIT)]
        parts = [pd.DataFrame({"q": quarter_label(splits[s][C.ISSUE_D]).to_numpy(), "p": scores[(name, s)], "split": s})
                 for s in QUARTER_SPLITS if (name, s) in scores]
        df = pd.concat(parts)
        edges = dev_edges(ref)
        e = shares(ref, edges)
        for q, g in df.groupby("q"):
            if QUARTER_FIRST <= q <= QUARTER_LAST:
                v = psi_from_shares(e, shares(g["p"], edges))
                rows.append(dict(model=name, quarter=q, n=len(g), mean_pd=float(g["p"].mean()), psi=v, light=V.light_low(v, "psi"),
                                 period="development" if q <= "2012Q4" else "out_of_time"))
    return pd.DataFrame(rows)


def battery_rows(bundles: dict, splits: dict, scores: dict, out) -> list:
    """Score PSI, CSI and quarterly PSI rows for the battery; also writes psi_csi.csv and psi_by_quarter.csv."""
    rows, tables = [], []
    for name in bundles:
        if (name, REF_SPLIT) not in scores:
            continue
        for s in (C.DEV_HOLDOUT, C.OOT1, C.OOT2):
            v = psi(scores[(name, REF_SPLIT)], scores[(name, s)])
            rows.append(V.result("psi_score", name, s, v, "psi"))
            tables.append(dict(kind="score", model=name, split=s, name="pd", psi=v, light=V.light_low(v, "psi")))
    for s in CSI_SPLITS:
        c = csi(splits[REF_SPLIT], splits[s], C.APPLICATION_FEATURES)
        for r in c.itertuples():
            tables.append(dict(kind="csi", model="all", split=s, name=r.feature, psi=r.psi, light=r.light))
        rows.append(V.result("csi_max", "all", s, float(c["psi"].max()), "psi"))
        rows.append(V.result("csi_n_red", "all", s, int((c["light"] == "Red").sum())))
    pd.DataFrame(tables).to_csv(out / "psi_csi.csv", index=False)
    q = psi_by_quarter(bundles, splits, scores)
    q.to_csv(out / "psi_by_quarter.csv", index=False)
    oq = q[q["quarter"] >= OOT_FIRST_QUARTER]
    for name, g in oq.groupby("model"):
        rows.append(V.result("psi_quarter_max_oot", name, "oot", float(g["psi"].max()), "psi"))
    return rows
