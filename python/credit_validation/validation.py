"""Validation statistics and the traffic-light test battery.

Traffic-light rule (thresholds from config.THRESHOLDS, tuple = (green, amber)):
  higher is better (gini, ks, hl_p, binom_p): Green if v >= green, Amber if v >= amber, else Red.
  lower is better (gini_decay, psi, central_tendency, rank_inversions): Green if v <= green, Amber if v <= amber, else Red.
  slope: Green if inside the green band, Amber if inside the amber band, else Red.
Deciles: rank = position after sorting descending by score (ties broken by input order); decile = ((rank - 1) * 10) // n + 1.
"""
import json
import os
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy import stats

from . import config
from . import columns as C

# ===== CONFIG (user inputs) =====
N_BINS = 10
PROB_FLOOR = 1e-9
HL_POWER_N = 100_000
EVAL_SPLITS = (C.DEV_TRAIN, C.DEV_HOLDOUT, C.OOT1, C.OOT2)
LIFETIME_SPLITS = (C.COMPLETE36, C.REPRO_TEST)
CI_SPLITS = (C.OOT1, C.OOT2)
GRADE_BINOMIAL_SPLITS = (C.DEV_HOLDOUT, C.OOT1, C.OOT2)
MIN_SUBGRADE_N = 100
LIGHT_INFO, LIGHT_NA = "Info", "n/a"
RESULT_COLUMNS = ["test", "model", "split", "value", "threshold", "light"]
CAP_POINTS = 101
PARALLEL_MIN_WORK = 5_000_000
N_JOBS = max((os.cpu_count() or 2) - 1, 1)
# ===== END CONFIG =====


def _arr(a):
    return np.asarray(a, dtype=np.float64)


# ---------- traffic lights ----------
def _fmt(x):
    return f"{x:g}"


def light_high(v, key) -> str:
    g, a = config.THRESHOLDS[key]
    if v is None or np.isnan(v):
        return LIGHT_NA
    return "Green" if v >= g else ("Amber" if v >= a else "Red")


def light_low(v, key) -> str:
    g, a = config.THRESHOLDS[key]
    if v is None or np.isnan(v):
        return LIGHT_NA
    return "Green" if v <= g else ("Amber" if v <= a else "Red")


def light_band(v, key) -> str:
    (glo, ghi), (alo, ahi) = config.THRESHOLDS[key]
    if v is None or np.isnan(v):
        return LIGHT_NA
    return "Green" if glo <= v <= ghi else ("Amber" if alo <= v <= ahi else "Red")


def threshold_text(key) -> str:
    g, a = config.THRESHOLDS[key]
    if key == "slope":
        return f"Green in {g}; Amber in {a}"
    if key in ("gini", "ks", "hl_p", "binom_p"):
        return f"Green >= {_fmt(g)}; Amber >= {_fmt(a)}"
    return f"Green <= {_fmt(g)}; Amber <= {_fmt(a)}"


def result(test, model, split, value, key=None, light=None) -> dict:
    """One battery row; `key` selects the threshold family (None gives an informational row)."""
    if key is None:
        return dict(test=test, model=model, split=split, value=value, threshold="", light=LIGHT_INFO)
    fn = light_band if key == "slope" else (light_high if key in ("gini", "ks", "hl_p", "binom_p") else light_low)
    return dict(test=test, model=model, split=split, value=value, threshold=threshold_text(key),
                light=light if light is not None else fn(value, key))


# ---------- discrimination ----------
def auc(y, p) -> float:
    y, p = _arr(y), _arr(p)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def gini(y, p) -> float:
    return 2.0 * auc(y, p) - 1.0


def ks(y, p) -> float:
    y, p = _arr(y), _arr(p)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ps, ys = p[order], y[order]
    last = np.r_[ps[1:] != ps[:-1], True]       # evaluate only after the last of each tied score
    return float(np.max(np.abs(np.cumsum(ys)[last] / n1 - np.cumsum(1 - ys)[last] / n0)))


def _desc_order(p):
    return np.argsort(-_arr(p), kind="stable")


def cap_table(y, p, points: int = CAP_POINTS) -> pd.DataFrame:
    """Cumulative accuracy profile: population share vs bad share, riskiest first, plus perfect and random curves."""
    y, p = _arr(y), _arr(p)
    ys = y[_desc_order(p)]
    n, bads = len(ys), ys.sum()
    cum = np.r_[0.0, np.cumsum(ys)] / bads
    idx = np.unique(np.round(np.linspace(0, n, points)).astype(int))
    pop = idx / n
    return pd.DataFrame({"pop_share": pop, "bad_share": cum[idx], "perfect": np.minimum(pop * n / bads, 1.0), "random": pop})


def decile_table(y, p, n: int = N_BINS) -> pd.DataFrame:
    y, p = _arr(y), _arr(p)
    order = _desc_order(p)
    ys, ps = y[order], p[order]
    N = len(ys)
    d = (np.arange(N) * n) // N                  # 0-based decile of each ranked row
    cnt = np.bincount(d, minlength=n).astype(float)
    bads = np.bincount(d, weights=ys, minlength=n)
    mean_pd = np.bincount(d, weights=ps, minlength=n) / cnt
    tot_b, tot_g = ys.sum(), N - ys.sum()
    cum_b, cum_g = np.cumsum(bads), np.cumsum(cnt - bads)
    base = ys.mean()
    lo = np.array([ps[d == k].min() for k in range(n)])
    hi = np.array([ps[d == k].max() for k in range(n)])
    return pd.DataFrame({
        "decile": np.arange(1, n + 1), "n": cnt.astype(int), "bads": bads.astype(int), "bad_rate": bads / cnt, "mean_pd": mean_pd,
        "score_min": lo, "score_max": hi, "cum_bads": cum_b.astype(int), "cum_bad_share": cum_b / tot_b,
        "cum_good_share": cum_g / tot_g, "ks_at_decile": np.abs(cum_b / tot_b - cum_g / tot_g),
        "lift": (bads / cnt) / base, "cum_lift": (cum_b / np.cumsum(cnt)) / base,
    })


def rank_order_inversions(rates, increasing: bool = False) -> int:
    """Adjacent pairs that break the expected order (default: bad rate should fall as rank goes down the table)."""
    r = _arr(rates)
    d = np.diff(r)
    return int((d < 0).sum() if increasing else (d > 0).sum())


def subgrade_inversions(y, p, subgrade) -> dict:
    """Order sub-grades by mean predicted PD and count adjacent observed-rate inversions (groups under MIN_SUBGRADE_N skipped)."""
    g = pd.DataFrame({"y": _arr(y), "p": _arr(p), "g": np.asarray(subgrade).astype(str)}).groupby("g").agg(n=("y", "size"), obs=("y", "mean"), pdm=("p", "mean"))
    g = g[g["n"] >= MIN_SUBGRADE_N].sort_values("pdm")
    return dict(inversions=rank_order_inversions(g["obs"].to_numpy(), increasing=True), n_groups=int(len(g)))


# ---------- calibration ----------
def calibration_table(y, p, n: int = N_BINS) -> pd.DataFrame:
    y, p = _arr(y), _arr(p)
    order = np.argsort(p, kind="stable")
    ys, ps = y[order], p[order]
    N = len(ys)
    d = (np.arange(N) * n) // N
    cnt = np.bincount(d, minlength=n).astype(float)
    bads = np.bincount(d, weights=ys, minlength=n)
    mp = np.bincount(d, weights=ps, minlength=n) / cnt
    return pd.DataFrame({"bin": np.arange(1, n + 1), "n": cnt.astype(int), "mean_pd": mp, "obs_rate": bads / cnt,
                         "expected_bads": mp * cnt, "bads": bads.astype(int), "diff": bads / cnt - mp})


def hosmer_lemeshow(y, p, g: int = N_BINS, in_sample: bool = False) -> dict:
    """df = g on out-of-time data and g-2 in sample; with very large n the test rejects trivial misfit (power caveat)."""
    t = calibration_table(y, p, g)
    n, mp, e, o = t["n"].to_numpy(float), t["mean_pd"].to_numpy(), t["expected_bads"].to_numpy(), t["bads"].to_numpy(float)
    den = n * mp * (1 - mp)
    ok = den > 0
    stat = float(np.sum((o[ok] - e[ok]) ** 2 / den[ok]))
    dof = g - 2 if in_sample else g
    nn = int(n.sum())
    return dict(stat=stat, df=dof, p_value=float(stats.chi2.sf(stat, dof)), n=nn,
                note="n > 100k: HL has very high power and flags economically small misfit; read with the calibration table"
                if nn > HL_POWER_N else "")


def binomial_test_by_group(y, p, groups) -> pd.DataFrame:
    """One-sided tests that PD is understated, using the group mean PD: normal z, exact binomial and Jeffreys Beta posterior."""
    df = pd.DataFrame({"y": _arr(y), "p": _arr(p), "g": np.asarray(groups).astype(str)})
    rows = []
    for grp, s in df.groupby("g", sort=True):
        n, d, pm = len(s), int(s["y"].sum()), float(s["p"].mean())
        z = (d - n * pm) / np.sqrt(n * pm * (1 - pm)) if 0 < pm < 1 else np.nan
        rows.append(dict(group=grp, n=n, bads=d, obs_rate=d / n, mean_pd=pm, z=z, p_z=float(stats.norm.sf(z)),
                         p_binom=float(stats.binom.sf(d - 1, n, pm)), p_jeffreys=float(stats.beta.cdf(pm, d + 0.5, n - d + 0.5))))
    return pd.DataFrame(rows)


def central_tendency(y, p) -> dict:
    obs, mp = float(_arr(y).mean()), float(_arr(p).mean())
    return dict(mean_pd=mp, obs_rate=obs, ratio=mp / obs if obs else np.nan, rel_dev=abs(mp - obs) / obs if obs else np.nan)


def calibration_slope_intercept(y, p) -> dict:
    """Logistic recalibration y ~ a + b * logit(p); ideal is a = 0, b = 1."""
    pc = np.clip(_arr(p), PROB_FLOOR, 1 - PROB_FLOOR)
    X = sm.add_constant(np.log(pc / (1 - pc)))
    fit = sm.Logit(_arr(y), X).fit(disp=0, maxiter=100)
    ci = np.asarray(fit.conf_int())
    return dict(intercept=float(fit.params[0]), slope=float(fit.params[1]), intercept_lo=float(ci[0, 0]), intercept_hi=float(ci[0, 1]),
                slope_lo=float(ci[1, 0]), slope_hi=float(ci[1, 1]))


def brier(y, p) -> float:
    return float(np.mean((_arr(p) - _arr(y)) ** 2))


def brier_skill(y, p) -> float:
    base = _arr(y).mean()
    return 1.0 - brier(y, p) / (base * (1 - base))


# ---------- inference ----------
def _delong_components(y, p):
    pos, neg = p[y == 1], p[y == 0]
    m, n = len(pos), len(neg)
    r_all, r_pos, r_neg = stats.rankdata(np.r_[pos, neg]), stats.rankdata(pos), stats.rankdata(neg)
    v10 = (r_all[:m] - r_pos) / n
    v01 = 1.0 - (r_all[m:] - r_neg) / m
    return float(v10.mean()), v10, v01


def delong_test(y, p1, p2):
    """Paired DeLong test of AUC1 - AUC2; returns (dAUC, z, two-sided p)."""
    y, p1, p2 = _arr(y), _arr(p1), _arr(p2)
    a1, v10_1, v01_1 = _delong_components(y, p1)
    a2, v10_2, v01_2 = _delong_components(y, p2)
    m, n = len(v10_1), len(v01_1)
    s = np.cov(np.vstack([v10_1, v10_2])) / m + np.cov(np.vstack([v01_1, v01_2])) / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    d = a1 - a2
    if var <= 0:
        return float(d), float("nan"), float("nan")
    z = d / np.sqrt(var)
    return float(d), float(z), float(2 * stats.norm.sf(abs(z)))


def _cap_rows(n, cap, seed):
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))


def _diff_metric(metric, y, p1, p2):
    return metric(y, p1) - metric(y, p2)


def _boot_chunk(fn, arrays, draws, seed_seq):
    rng = np.random.default_rng(seed_seq)
    n = len(arrays[0])
    return [fn(*[a[i] for a in arrays]) for i in (rng.integers(0, n, n) for _ in range(draws))]


def _boot_values(fn, arrays, n_boot, seed) -> np.ndarray:
    """Resample loans with replacement; large jobs are split over processes with independent, seeded streams."""
    n_jobs = 1 if len(arrays[0]) * n_boot < PARALLEL_MIN_WORK else min(N_JOBS, n_boot)
    seqs = np.random.SeedSequence(seed).spawn(n_jobs)
    sizes = [len(c) for c in np.array_split(np.arange(n_boot), n_jobs)]
    if n_jobs == 1:
        parts = [_boot_chunk(fn, arrays, sizes[0], seqs[0])]
    else:
        parts = Parallel(n_jobs=n_jobs)(delayed(_boot_chunk)(fn, arrays, k, sq) for k, sq in zip(sizes, seqs))
    vals = np.array([v for part in parts for v in part], dtype=np.float64)
    return vals[~np.isnan(vals)]


def bootstrap_ci(metric, y, p, n_boot=None, seed=None, cap=None, level=0.95):
    """Loan-level percentile bootstrap; rows above the cap are subsampled once first. Returns (lo, hi, n_used)."""
    n_boot = config.BOOTSTRAP_N if n_boot is None else n_boot
    seed = config.SEED if seed is None else seed
    cap = config.BOOT_ROW_CAP if cap is None else cap
    y, p = _arr(y), _arr(p)
    sel = _cap_rows(len(y), cap, seed)
    vals = _boot_values(metric, (y[sel], p[sel]), n_boot, seed)
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a)), int(len(sel))


def bootstrap_diff_ci(metric, y, p1, p2, n_boot=None, seed=None, cap=None, level=0.95):
    """Paired bootstrap CI of metric(y, p1) - metric(y, p2) resampling the same loans for both models."""
    n_boot = config.BOOTSTRAP_N if n_boot is None else n_boot
    seed = config.SEED if seed is None else seed
    cap = config.BOOT_ROW_CAP if cap is None else cap
    y, p1, p2 = _arr(y), _arr(p1), _arr(p2)
    sel = _cap_rows(len(y), cap, seed)
    vals = _boot_values(partial(_diff_metric, metric), (y[sel], p1[sel], p2[sel]), n_boot, seed)
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a)), int(len(sel))


# ---------- battery ----------
def target_for(split: str) -> str:
    return C.TARGET_LIFETIME if split in LIFETIME_SPLITS else C.TARGET_BAD12


def model_splits(model: str, bundle, splits: dict) -> list:
    """Which splits a model is evaluated on (recal challenger: OOT2 only; repro: also its own random test)."""
    if model == "sc_full_recal":
        return [C.OOT2]
    out = [s for s in EVAL_SPLITS if s in splits]
    if C.COMPLETE36 in splits:
        out.append(C.COMPLETE36)
    if model == "repro_original" and C.REPRO_TEST in splits:
        out.append(C.REPRO_TEST)
    return out


def score_all(bundles: dict, splits: dict) -> dict:
    """PD predictions per (model, split), computed once and reused by every test."""
    return {(m, s): np.asarray(b.predict_pd(splits[s]), dtype=np.float64) for m, b in bundles.items() for s in model_splits(m, b, splits)}


def _calibration_applies(model: str, split: str) -> bool:
    if split == C.COMPLETE36:
        return False                               # 12-month PD vs lifetime outcome: not a like-for-like calibration
    if model == "repro_original":
        return split == C.REPRO_TEST               # the original only claims calibration on its own lifetime target
    return True


def model_meta(bundles: dict, splits: dict) -> dict:
    meta = {"data_source": config.DATA_SOURCE, "models": {}, "splits": {}}
    for name, b in bundles.items():
        m = dict(kind=b.kind, n_features=len(b.features), features=list(b.features), margin_shift=b.margin_shift)
        m.update({k: v for k, v in b.extras.items() if isinstance(v, (int, float, str, list, dict))})
        sc = b.scorecard
        if sc is not None:
            m.update(scorecard_variables=sc.variables, iv=sc.iv, vif=sc.vif, pvalues=sc.pvalues, intercept=sc.intercept,
                     factor=sc.factor, offset=sc.offset, n_train=sc.n_train, dropped=[list(d) for d in sc.dropped])
        meta["models"][name] = m
    for s, d in splits.items():
        t = target_for(s)
        meta["splits"][s] = dict(n=int(len(d)), target=t, bad_rate=float(d[t].mean()),
                                 issue_min=str(d[C.ISSUE_D].min().date()), issue_max=str(d[C.ISSUE_D].max().date()))
    meta["thresholds"] = {k: v for k, v in config.THRESHOLDS.items()}
    return meta


def _vintage_table(bundles, splits, scores) -> pd.DataFrame:
    rows = []
    for name in bundles:
        parts = [pd.DataFrame({C.VINTAGE: splits[s][C.VINTAGE].to_numpy(), "y": splits[s][C.TARGET_BAD12].to_numpy(), "p": scores[(name, s)]})
                 for s in model_splits(name, bundles[name], splits) if s in EVAL_SPLITS]
        if not parts:
            continue
        df = pd.concat(parts)
        for v, g in df.groupby(C.VINTAGE):
            rows.append(dict(model=name, vintage=int(v), n=len(g), bad_rate=g["y"].mean(), mean_pd=g["p"].mean(),
                             auc=auc(g["y"], g["p"]) if g["y"].nunique() > 1 else np.nan))
    out = pd.DataFrame(rows)
    out["gini"] = 2 * out["auc"] - 1
    return out


def run_battery(bundles: dict, splits: dict, out_dir=None, n_boot=None) -> pd.DataFrame:
    from . import benchmark, stability
    out = Path(out_dir or config.OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    scores = score_all(bundles, splits)
    rows, gini_by = [], {}
    grade_parts = {s: [] for s in GRADE_BINOMIAL_SPLITS}
    for name, b in bundles.items():
        for s in model_splits(name, b, splits):
            d, p = splits[s], scores[(name, s)]
            y = d[target_for(s)].to_numpy(dtype=np.float64)
            a = auc(y, p)
            gini_by[(name, s)] = 2 * a - 1
            rows += [result("auc", name, s, a), result("gini", name, s, 2 * a - 1, "gini"), result("ks", name, s, ks(y, p), "ks"),
                     result("brier", name, s, brier(y, p)), result("brier_skill", name, s, brier_skill(y, p))]
            dec = decile_table(y, p)
            dec.to_csv(out / f"deciles_{name}_{s}.csv", index=False)
            rows.append(result("decile_rank_inversions", name, s, rank_order_inversions(dec["bad_rate"].to_numpy()), "rank_inversions"))
            rows.append(result("ks_decile", name, s, float(dec["ks_at_decile"].max())))
            if s in CI_SPLITS:
                lo, hi, _ = bootstrap_ci(gini, y, p, n_boot=n_boot)
                rows += [result("gini_ci_lo", name, s, lo), result("gini_ci_hi", name, s, hi)]
            if not _calibration_applies(name, s):
                continue
            cal = calibration_table(y, p)
            cal.to_csv(out / f"calibration_{name}_{s}.csv", index=False)
            ct = central_tendency(y, p)
            rows += [result("mean_pd", name, s, ct["mean_pd"]), result("obs_rate", name, s, ct["obs_rate"]),
                     result("central_tendency", name, s, ct["rel_dev"], "central_tendency"),
                     result("central_tendency_signed", name, s, ct["ratio"] - 1.0)]
            hl = hosmer_lemeshow(y, p, in_sample=(s == C.DEV_TRAIN))
            rows += [result("hl_stat", name, s, hl["stat"]), result("hl_p", name, s, hl["p_value"], "hl_p")]
            sl = calibration_slope_intercept(y, p)
            rows += [result("cal_slope", name, s, sl["slope"], "slope"), result("cal_intercept", name, s, sl["intercept"]),
                     result("cal_slope_ci_lo", name, s, sl["slope_lo"]), result("cal_slope_ci_hi", name, s, sl["slope_hi"])]
            if s in GRADE_BINOMIAL_SPLITS:
                bt = binomial_test_by_group(y, p, d[C.GRADE].to_numpy())
                bt.insert(0, "model", name)
                grade_parts[s].append(bt)
                rows.append(result("binomial_grade_min_p", name, s, float(bt["p_jeffreys"].min()), "binom_p"))
                rows.append(result("binomial_grade_n_red", name, s, int((bt["p_jeffreys"] < config.THRESHOLDS["binom_p"][1]).sum())))
            if s in (C.OOT1, C.OOT2):
                sg = subgrade_inversions(y, p, d[C.SUB_GRADE].to_numpy())
                rows += [result("subgrade_rank_inversions", name, s, sg["inversions"]), result("subgrade_groups", name, s, sg["n_groups"])]
    for s, parts in grade_parts.items():
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(out / f"grade_binomial_{s}.csv", index=False)
    for name in bundles:
        if (name, C.DEV_HOLDOUT) in gini_by:
            for s in CI_SPLITS:
                if (name, s) in gini_by and gini_by[(name, C.DEV_HOLDOUT)] > 0:
                    dec = (gini_by[(name, C.DEV_HOLDOUT)] - gini_by[(name, s)]) / gini_by[(name, C.DEV_HOLDOUT)]
                    rows.append(result("gini_decay", name, s, dec, "gini_decay"))
    rows += stability.battery_rows(bundles, splits, scores, out)
    bench_df, bench_rows = benchmark.run_benchmark(bundles, splits, scores, n_boot=n_boot)
    bench_df.to_csv(out / "benchmark.csv", index=False)
    rows += bench_rows
    _vintage_table(bundles, splits, scores).to_csv(out / "by_vintage.csv", index=False)
    pts, binr = [], []
    from .binning import binning_report
    for name in ("sc_full", "sc_indep"):
        sc = bundles[name].scorecard if name in bundles else None
        if sc is None:
            continue
        t = sc.export_points_table()
        t.insert(0, "model", name)
        pts.append(t)
        r = binning_report(sc.specs)
        r.insert(0, "model", name)
        binr.append(r)
    if pts:
        pd.concat(pts, ignore_index=True).to_csv(out / "scorecard_points.csv", index=False)
        pd.concat(binr, ignore_index=True).to_csv(out / "binning_report.csv", index=False)
    res = pd.DataFrame(rows)[RESULT_COLUMNS]
    res.to_csv(out / "validation_results.csv", index=False)
    meta = model_meta(bundles, splits)
    meta["light_counts"] = res["light"].value_counts().to_dict()
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return res
