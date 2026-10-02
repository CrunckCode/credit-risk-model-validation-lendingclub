"""Fine/coarse classing: monotone numeric bins, grouped categorical bins, WoE and IV.

Convention: WoE = ln(%good / %bad), so higher WoE is safer and logit coefficients on WoE are negative.
Bin rule: np.searchsorted(edges, x, side="left"), i.e. bin_lo exclusive and bin_hi inclusive.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ===== CONFIG (user inputs) =====
SMOOTH = 0.5
IV_BANDS = ((0.02, "Not useful"), (0.1, "Weak"), (0.3, "Medium"), (0.5, "Strong"))
IV_SUSPICIOUS = "Suspicious"
MISSING_LABEL = "Missing"
CAT_JOIN = "|"
# ===== END CONFIG =====


def woe_iv(bad, good, smooth: float = SMOOTH):
    """Return (woe, iv_contrib) per bin with additive smoothing; total IV is iv_contrib.sum()."""
    bad = np.asarray(bad, dtype=np.float64)
    good = np.asarray(good, dtype=np.float64)
    k = len(bad)
    db = (bad + smooth) / (bad.sum() + smooth * k)
    dg = (good + smooth) / (good.sum() + smooth * k)
    woe = np.log(dg / db)
    return woe, (dg - db) * woe


def iv_band(iv: float) -> str:
    for limit, label in IV_BANDS:
        if iv < limit:
            return label
    return IV_SUSPICIOUS


def _rate(bad, good, smooth=SMOOTH):
    return (bad + smooth) / (bad + good + 2 * smooth)


def _merge(bad, good, edges, i):
    """Merge bin i with bin i+1 (edges has one fewer element than bins)."""
    bad = np.r_[bad[:i], bad[i] + bad[i + 1], bad[i + 2:]]
    good = np.r_[good[:i], good[i] + good[i + 1], good[i + 2:]]
    return bad, good, np.delete(edges, i)


def _merge_min_share(bad, good, edges, min_share, merge_fn):
    n_all = (bad + good).sum()
    while len(bad) > 1:
        share = (bad + good) / n_all
        j = int(np.argmin(share))
        if share[j] >= min_share:
            break
        r = _rate(bad, good)
        if j == 0:
            k = 0
        elif j == len(bad) - 1:
            k = j - 1
        else:
            k = j - 1 if abs(r[j] - r[j - 1]) <= abs(r[j] - r[j + 1]) else j
        bad, good, edges = merge_fn(bad, good, edges, k)
    return bad, good, edges


def _cap_bins(bad, good, edges, cap, merge_fn):
    while len(bad) > cap:
        k = int(np.argmin(np.abs(np.diff(_rate(bad, good)))))
        bad, good, edges = merge_fn(bad, good, edges, k)
    return bad, good, edges


@dataclass
class BinSpec:
    """Numeric binning: regular bins, optional special-value bins, then a missing bin (always last)."""
    variable: str
    edges: np.ndarray
    woe: np.ndarray                      # one per regular bin
    missing_woe: float
    iv: float
    n_bad: np.ndarray
    n_good: np.ndarray
    special_values: tuple = ()
    special_woe: np.ndarray = field(default_factory=lambda: np.array([]))
    special_bad: np.ndarray = field(default_factory=lambda: np.array([]))
    special_good: np.ndarray = field(default_factory=lambda: np.array([]))
    missing_bad: float = 0.0
    missing_good: float = 0.0
    kind: str = "numeric"

    @property
    def n_bins(self) -> int:
        return len(self.woe) + len(self.special_values) + 1

    @property
    def woe_all(self) -> np.ndarray:
        return np.r_[self.woe, self.special_woe, self.missing_woe]

    def bin_index(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        idx = np.searchsorted(self.edges, x, side="left")
        nb = len(self.woe)
        for j, v in enumerate(self.special_values):
            idx = np.where(x == v, nb + j, idx)
        return np.where(np.isnan(x), self.n_bins - 1, idx)

    def transform(self, x) -> np.ndarray:
        return self.woe_all[self.bin_index(x)]

    def table(self) -> pd.DataFrame:
        nb = len(self.woe)
        lo = np.r_[-np.inf, self.edges].tolist() + [np.nan] * (len(self.special_values) + 1)
        hi = np.r_[self.edges, np.inf].tolist() + [np.nan] * (len(self.special_values) + 1)
        labels = [_interval_label(l, h) for l, h in zip(lo[:nb], hi[:nb])]
        labels += [f"= {v:g}" for v in self.special_values] + [MISSING_LABEL]
        bad = np.r_[self.n_bad, self.special_bad, self.missing_bad]
        good = np.r_[self.n_good, self.special_good, self.missing_good]
        ivc = self._iv_contrib(bad, good)
        return pd.DataFrame({
            "bin_id": np.arange(self.n_bins), "bin": labels, "lo": lo, "hi": hi, "categories": "",
            "n": bad + good, "bads": bad, "woe": self.woe_all, "iv_contrib": ivc,
        })

    def _iv_contrib(self, bad, good):
        present = (bad + good) > 0
        ivc = np.zeros(len(bad))
        if present.any():
            ivc[present] = woe_iv(bad[present], good[present])[1]
        return ivc


@dataclass
class CatSpec:
    """Categorical binning: ordered groups of categories plus a missing bin (last). Unseen categories use the largest group."""
    variable: str
    groups: list                         # list of tuples of category labels
    woe: np.ndarray
    missing_woe: float
    iv: float
    n_bad: np.ndarray
    n_good: np.ndarray
    missing_bad: float = 0.0
    missing_good: float = 0.0
    kind: str = "categorical"

    @property
    def n_bins(self) -> int:
        return len(self.groups) + 1

    @property
    def woe_all(self) -> np.ndarray:
        return np.r_[self.woe, self.missing_woe]

    @property
    def default_group(self) -> int:
        return int(np.argmax(self.n_bad + self.n_good))

    def _lookup(self) -> dict:
        return {c: i for i, g in enumerate(self.groups) for c in g}

    def bin_index(self, x) -> np.ndarray:
        s = x if isinstance(x, pd.Series) else pd.Series(x)
        mapped = s.map(self._lookup())
        mapped = pd.to_numeric(mapped.astype("object"), errors="coerce")   # category-dtype map would stay categorical
        idx = np.where(s.isna().to_numpy(), self.n_bins - 1, mapped.fillna(self.default_group).to_numpy())
        return idx.astype(np.int64)

    def transform(self, x) -> np.ndarray:
        return self.woe_all[self.bin_index(x)]

    def table(self) -> pd.DataFrame:
        labels = [CAT_JOIN.join(str(c) for c in g) for g in self.groups]
        bad = np.r_[self.n_bad, self.missing_bad]
        good = np.r_[self.n_good, self.missing_good]
        present = (bad + good) > 0
        ivc = np.zeros(len(bad))
        ivc[present] = woe_iv(bad[present], good[present])[1]
        return pd.DataFrame({
            "bin_id": np.arange(self.n_bins), "bin": labels + [MISSING_LABEL], "lo": np.nan, "hi": np.nan,
            "categories": labels + [""], "n": bad + good, "bads": bad, "woe": self.woe_all, "iv_contrib": ivc,
        })


def _interval_label(lo, hi) -> str:
    if np.isinf(lo):
        return f"<= {hi:g}"
    if np.isinf(hi):
        return f"> {lo:g}"
    return f"({lo:g}, {hi:g}]"


def _finalize(bad, good, miss_bad, miss_good, sp_bad, sp_good):
    """WoE over all populated bins jointly so shares sum to one; empty special/missing bins stay neutral."""
    allb = np.r_[bad, sp_bad, miss_bad]
    allg = np.r_[good, sp_good, miss_good]
    present = (allb + allg) > 0
    woe_all = np.zeros(len(allb))
    ivc = np.zeros(len(allb))
    woe_all[present], ivc[present] = woe_iv(allb[present], allg[present])
    nb, ns = len(bad), len(sp_bad)
    return woe_all[:nb], woe_all[nb:nb + ns], float(woe_all[-1]), float(ivc.sum())


def fit_numeric_bins(x, y, prebins: int = 50, min_share: float = 0.05, max_bins: int = 8,
                     monotone: bool = True, special_values=(), variable: str = "") -> BinSpec:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    miss = np.isnan(x)
    special = np.zeros(len(x), dtype=bool)
    for v in special_values:
        special |= (x == v)
    reg = ~miss & ~special
    xv, yv = x[reg], y[reg]
    sp_bad = np.array([y[x == v].sum() for v in special_values], dtype=np.float64)
    sp_good = np.array([(1 - y[x == v]).sum() for v in special_values], dtype=np.float64)
    miss_bad, miss_good = float(y[miss].sum()), float((1 - y[miss]).sum())
    if len(xv) == 0:
        raise ValueError("no regular values to bin")

    cuts = np.unique(np.quantile(xv, np.linspace(0, 1, prebins + 1)[1:-1], method="lower"))
    cuts = cuts[cuts < xv.max()]
    idx = np.searchsorted(cuts, xv, side="left")
    nb = len(cuts) + 1
    bad = np.bincount(idx, weights=yv, minlength=nb)
    good = np.bincount(idx, weights=1 - yv, minlength=nb)
    edges = cuts.copy()

    bad, good, edges = _merge_min_share(bad, good, edges, min_share, _merge)
    if monotone and len(bad) > 2:
        r = _rate(bad, good)
        trend = spearmanr(np.arange(len(r)), r)[0]
        sign = 1.0 if (np.isnan(trend) or trend >= 0) else -1.0
        while len(bad) > 2:
            viol = np.flatnonzero(np.diff(_rate(bad, good) * sign) < 0)
            if len(viol) == 0:
                break
            bad, good, edges = _merge(bad, good, edges, int(viol[0]))
    bad, good, edges = _cap_bins(bad, good, edges, max_bins, _merge)

    woe, sp_woe, miss_woe, iv = _finalize(bad, good, miss_bad, miss_good, sp_bad, sp_good)
    return BinSpec(variable=variable, edges=np.asarray(edges, dtype=np.float64), woe=woe, missing_woe=miss_woe, iv=iv,
                   n_bad=bad, n_good=good, special_values=tuple(special_values), special_woe=sp_woe,
                   special_bad=sp_bad, special_good=sp_good, missing_bad=miss_bad, missing_good=miss_good)


def _merge_groups(bad, good, groups, i):
    bad = np.r_[bad[:i], bad[i] + bad[i + 1], bad[i + 2:]]
    good = np.r_[good[:i], good[i] + good[i + 1], good[i + 2:]]
    groups = groups[:i] + [groups[i] + groups[i + 1]] + groups[i + 2:]
    return bad, good, groups


def fit_categorical_bins(x, y, min_share: float = 0.02, max_groups: int = 10, variable: str = "") -> CatSpec:
    s = x if isinstance(x, pd.Series) else pd.Series(x)
    y = np.asarray(y, dtype=np.float64)
    miss = s.isna().to_numpy()
    obs = pd.DataFrame({"c": s[~miss].astype(object).to_numpy(), "bad": y[~miss], "good": 1 - y[~miss]})
    agg = obs.groupby("c", sort=False).agg(bad=("bad", "sum"), good=("good", "sum"))
    agg = agg.assign(rate=_rate(agg["bad"], agg["good"])).sort_values("rate", kind="stable")
    groups = [(c,) for c in agg.index]
    bad, good = agg["bad"].to_numpy(), agg["good"].to_numpy()

    # categories are ordered by bad rate, so adjacent merges group similar-risk levels
    def merge_fn(b, g, gr, i):
        return _merge_groups(b, g, gr, i)

    bad, good, groups = _merge_min_share(bad, good, groups, min_share, merge_fn)
    bad, good, groups = _cap_bins(bad, good, groups, max_groups, merge_fn)
    groups = [tuple(sorted(g, key=str)) for g in groups]

    miss_bad, miss_good = float(y[miss].sum()), float((1 - y[miss]).sum())
    woe, _, miss_woe, iv = _finalize(bad, good, miss_bad, miss_good, np.array([]), np.array([]))
    return CatSpec(variable=variable, groups=groups, woe=woe, missing_woe=miss_woe, iv=iv, n_bad=bad, n_good=good,
                   missing_bad=miss_bad, missing_good=miss_good)


def binning_report(specs) -> pd.DataFrame:
    """Long table for all specs; `specs` is a dict {variable: spec} or an iterable of specs."""
    items = specs.items() if isinstance(specs, dict) else [(s.variable, s) for s in specs]
    parts = []
    for name, sp in items:
        t = sp.table()
        t.insert(0, "variable", name)
        parts.append(t)
    out = pd.concat(parts, ignore_index=True)
    out["bad_rate"] = np.where(out["n"] > 0, out["bads"] / out["n"].where(out["n"] > 0, 1), np.nan)
    cols = ["variable", "bin", "lo", "hi", "categories", "n", "bads", "bad_rate", "woe", "iv_contrib", "bin_id"]
    return out[cols]
