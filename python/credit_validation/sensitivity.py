"""Sensitivity analysis: feature shocks, PD multipliers, LGD add-on, target-definition and binning variants (all on OOT1)."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from . import columns as C
from . import features, splits as splits_mod
from . import scorecard as sc_mod
from . import stability
from .validation import gini

# ===== CONFIG (user inputs) =====
SHOCK_COLUMNS = {"dti_pts": (C.DTI, "add"), "int_rate_pts": (C.INT_RATE, "add"), "income_pct": (C.ANNUAL_INC, "pct"),
                 "revol_util_pts": (C.REVOL_UTIL, "add"), "inq_add": (C.INQ_LAST_6MTHS, "add")}
COMBINED_LABEL = "shock_combined"
SHOCK_MODELS = ("sc_full", "sc_indep", "xgb", "lgbm")
TARGET_MODEL = "sc_full"
MIN_SHARES, MAX_BINS_GRID = (0.03, 0.05, 0.10), (5, 8, 12)
IN_GRACE_STATUS = "In Grace Period"
FALLBACK_LGD, FALLBACK_CCF = 0.94, 0.74           # only when no LGD/EAD model is supplied
OUT_FILE = "sensitivity.csv"
COLUMNS = ["scenario", "model", "mean_pd", "delta_pd_pct", "el", "delta_el_pct", "spearman_rank", "share_band_moved", "group",
           "gini_oot1", "iv_sum", "n_variables"]
# ===== END CONFIG =====


def apply_shocks(df: pd.DataFrame, names) -> pd.DataFrame:
    out = df.copy()
    for n in names:
        col, how = SHOCK_COLUMNS[n]
        amount = config.SENSITIVITY_SHOCKS[n]
        out[col] = (out[col] + amount) if how == "add" else (out[col] * (1.0 + amount / 100.0))
    return features.build_features(out)            # derived loan-to-income and PTI follow the shocked income


def total_el(df, pd_arr, lgd_model, lgd_add: float = 0.0) -> float:
    if lgd_model is None:
        return float(np.sum(pd_arr * FALLBACK_LGD * FALLBACK_CCF * df[C.FUNDED_AMNT].to_numpy(float)))
    r = lgd_model.predict(df)
    lgd = np.clip(r[C.PRED_LGD].to_numpy() + lgd_add, 0.0, 1.0)
    return float(np.sum(pd_arr * lgd * r[C.PRED_EAD].to_numpy()))


def _row(scenario, model, group, pd1, el1, base_pd, base_el, edges, extra=None) -> dict:
    band0, band1 = np.searchsorted(edges, base_pd, side="left"), np.searchsorted(edges, pd1, side="left")
    sp = float(stats.spearmanr(base_pd, pd1)[0]) if np.std(pd1) > 0 else np.nan
    row = dict(scenario=scenario, model=model, group=group, mean_pd=float(np.mean(pd1)), delta_pd_pct=100.0 * (np.mean(pd1) / np.mean(base_pd) - 1.0),
               el=el1, delta_el_pct=100.0 * (el1 / base_el - 1.0), spearman_rank=sp, share_band_moved=float((band0 != band1).mean()))
    row.update(extra or {})
    return row


def _variant_eval(name, tr, oot, target, base_oot_pd_by_id, lgd_model, edges, base_mean_pd, base_el_ref, **fit_kw):
    sc = sc_mod.fit_scorecard(tr, target, list(C.APPLICATION_FEATURES), **fit_kw)
    pdv = sc.predict_pd(oot)
    ids = oot[C.LOAN_ID].to_numpy()
    common = pd.Series(pdv, index=ids)
    common = common[~common.index.duplicated()]
    base = base_oot_pd_by_id.reindex(common.index)
    ok = base.notna().to_numpy()
    row = _row(name, TARGET_MODEL, "", common.to_numpy()[ok], total_el(oot, pdv, lgd_model), base.to_numpy()[ok], base_el_ref, edges)
    row["delta_pd_pct"] = 100.0 * (pdv.mean() / base_mean_pd - 1.0)
    row.update(gini_oot1=gini(oot[target].to_numpy(), pdv), iv_sum=float(sum(sc.iv.values())), n_variables=len(sc.variables))
    return row


def run_sensitivity(bundles: dict, splits: dict, loans: pd.DataFrame = None, lgd_model=None, out_dir=None, models=SHOCK_MODELS) -> pd.DataFrame:
    oot = splits[C.OOT1]
    dev = splits[C.DEV_TRAIN]
    rows = []
    for m in [k for k in models if k in bundles]:
        b = bundles[m]
        base_pd = b.predict_pd(oot)
        base_el = total_el(oot, base_pd, lgd_model)
        edges = stability.dev_edges(b.predict_pd(dev))
        combos = [(n, [n]) for n in SHOCK_COLUMNS] + [(COMBINED_LABEL, list(SHOCK_COLUMNS))]
        for label, names in combos:
            sh = apply_shocks(oot, names)
            p1 = b.predict_pd(sh)
            rows.append(_row(f"shock_{label}" if label != COMBINED_LABEL else label, m, "feature_shock", p1, total_el(sh, p1, lgd_model), base_pd, base_el, edges))
        for mult in config.PD_MULTIPLIERS:
            p1 = np.clip(base_pd * mult, 0.0, 1.0)
            rows.append(_row(f"pd_x{mult:g}", m, "pd_multiplier", p1, total_el(oot, p1, lgd_model), base_pd, base_el, edges))
        rows.append(_row(f"lgd_add_{config.LGD_ADD:g}", m, "lgd_add", base_pd, total_el(oot, base_pd, lgd_model, config.LGD_ADD), base_pd, base_el, edges))
    b = bundles[TARGET_MODEL]
    base_pd = b.predict_pd(oot)
    base_el = total_el(oot, base_pd, lgd_model)
    edges = stability.dev_edges(b.predict_pd(dev))
    base_by_id = pd.Series(base_pd, index=oot[C.LOAN_ID].to_numpy())
    base_mean = float(base_pd.mean())
    base_gini = gini(oot[C.TARGET_BAD12].to_numpy(), base_pd)
    base_extra = dict(gini_oot1=base_gini, iv_sum=float(sum(b.scorecard.iv.values())), n_variables=len(b.scorecard.variables))
    rows.append(_row("baseline", TARGET_MODEL, "baseline", base_pd, base_el, base_pd, base_el, edges, base_extra))
    # binning variants (the grid points equal to the config defaults reuse the champion fit)
    for ms in MIN_SHARES:
        if ms == config.BIN_MIN_SHARE:
            rows.append(_row(f"binning_min_share_{ms:.0%}", TARGET_MODEL, "binning", base_pd, base_el, base_pd, base_el, edges, base_extra))
        else:
            rows.append({**_variant_eval(f"binning_min_share_{ms:.0%}", dev, oot, C.TARGET_BAD12, base_by_id, lgd_model, edges, base_mean, base_el, min_share=ms), "group": "binning"})
    for mb in MAX_BINS_GRID:
        if mb == config.BIN_MAX_BINS:
            rows.append(_row(f"binning_max_bins_{mb}", TARGET_MODEL, "binning", base_pd, base_el, base_pd, base_el, edges, base_extra))
        else:
            rows.append({**_variant_eval(f"binning_max_bins_{mb}", dev, oot, C.TARGET_BAD12, base_by_id, lgd_model, edges, base_mean, base_el, max_bins=mb), "group": "binning"})
    if loans is not None:
        rows += target_variants(loans, base_by_id, lgd_model, edges, base_mean, base_el)
    out = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in out:
            out[c] = np.nan
    out = out[COLUMNS]
    d = Path(out_dir or config.OUT_DIR)
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / OUT_FILE, index=False)
    return out


def target_variants(loans, base_by_id, lgd_model, edges, base_mean, base_el) -> list:
    rows = []
    # In Grace Period coded bad (the original coded it good)
    alt = loans.copy()
    grace = alt[C.LOAN_STATUS].astype(str).eq(IN_GRACE_STATUS) & (alt[C.MOB_LAST_PAY] < config.PD_WINDOW_MONTHS)
    alt[C.TARGET_BAD12] = np.maximum(alt[C.TARGET_BAD12].to_numpy(), grace.to_numpy().astype("int8"))
    s = splits_mod.time_splits(alt)
    rows.append({**_variant_eval("target_in_grace_bad", s[C.DEV_TRAIN], s[C.OOT1], C.TARGET_BAD12, base_by_id, lgd_model, edges, base_mean, base_el), "group": "target_definition"})
    # 24-month window
    s = splits_mod.time_splits(loans, target=C.TARGET_BAD24)
    rows.append({**_variant_eval("target_24m_window", s[C.DEV_TRAIN], s[C.OOT1], C.TARGET_BAD24, base_by_id, lgd_model, edges, base_mean, base_el), "group": "target_definition"})
    # DNMCP loans kept in the population
    keep = config.EXCLUDE_DNMCP
    config.EXCLUDE_DNMCP = False
    try:
        s = splits_mod.time_splits(loans)
    finally:
        config.EXCLUDE_DNMCP = keep
    rows.append({**_variant_eval("population_dnmcp_included", s[C.DEV_TRAIN], s[C.OOT1], C.TARGET_BAD12, base_by_id, lgd_model, edges, base_mean, base_el), "group": "target_definition"})
    return rows
