"""Plain matplotlib charts (Agg, 150 dpi) built from the validation output CSVs; a chart is skipped if its input is missing."""
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .columns import OOT1, OOT2, DEV_HOLDOUT, TARGET_BAD12, TARGET_LIFETIME, VINTAGE, REPRO_TRAIN, REPRO_TEST

log = logging.getLogger(__name__)

# ===== CONFIG (user inputs) =====
DPI = 150
FIG_WIDE, FIG_SQUARE, FIG_TALL = (10, 4.5), (6.5, 5.5), (8, 7)
LINE_STYLES = ("-", "--", "-.", ":")
GREY_LEVELS = ("black", "dimgray", "gray", "darkgray")
MODEL_ORDER = ("sc_full", "sc_indep", "xgb", "lgbm", "sc_full_recal")
MODEL_LABELS = {"sc_full": "Scorecard full", "sc_indep": "Scorecard independent", "xgb": "XGBoost",
                "lgbm": "LightGBM", "sc_full_recal": "Scorecard full, recalibrated"}
SPLIT_LABELS = {"dev_holdout": "Development holdout", "oot1": "OOT 2013", "oot2": "OOT 2014"}
ROC_MODELS = ("sc_full", "sc_indep", "xgb", "lgbm")
DECILE_MODELS = ("sc_full", "lgbm")
CALIB_MODEL = "sc_full"
CALIB_SPLITS = (DEV_HOLDOUT, OOT1, OOT2)
TOP_WOE_VARS = 6
TOP_POINTS_VARS = 20
TOP_SHAP = 15
GINI_TEST = "gini"
VINTAGE_COL, GRADE_COL, QUARTER_COL, MODEL_KEY = VINTAGE, "grade", "quarter", "model"
CHAMPION = "sc_full"
ALL_LABEL = "ALL"
SAMPLE_OOT = "out_of_time"
# candidate column names (first match wins) because csv writers differ slightly
CAND = dict(
    n=("n", "count", "total", "loans"), bads=("bads", "n_bad", "bad", "defaults", "events"),
    rate=("bad_rate", "observed", "obs_rate", "actual_rate", "default_rate", "observed_rate"),
    pred=("mean_pd", "avg_pd", "pred_pd", "predicted", "mean_pred", "expected_rate", "pred_rate", "avg_pred"),
    actual=("actual", "mean_actual", "observed", "obs", "actual_mean", "realized", "bad_rate"),
    pred_el=("pred_el", "predicted_el", "el_pred", "expected_loss", "el_predicted", "el", "pred_loss"),
    real_el=("realized_loss", "realised_loss", "actual_loss", "net_loss", "loss_realized", "realized", "actual_el"),
)
# ===== END CONFIG =====

plt.rcParams.update({"figure.dpi": DPI, "savefig.dpi": DPI, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _col(df, key_or_names):
    """First existing column among the candidates (case-insensitive), else None."""
    names = CAND.get(key_or_names, key_or_names) if isinstance(key_or_names, str) else key_or_names
    low = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def _read(out_dir, name):
    p = Path(out_dir) / name
    if not p.exists():
        log.warning("chart input missing: %s", p)
        return None
    try:
        df = pd.read_csv(p)
    except Exception as exc:  # unreadable csv should not stop the other charts
        log.warning("cannot read %s: %s", p, exc)
        return None
    return df if len(df) else None


def _champion_rows(d):
    """Rows of the champion scorecard when the csv has a model column."""
    if MODEL_KEY in d.columns and (d[MODEL_KEY] == CHAMPION).any():
        return d[d[MODEL_KEY] == CHAMPION]
    return d


def _label(m):
    return MODEL_LABELS.get(m, m)


# ---------- curve helpers ----------
def _curves_from_scores(y, p):
    y = np.asarray(y, dtype=float)
    order = np.argsort(-np.asarray(p, dtype=float), kind="stable")
    ys = y[order]
    tpr = np.concatenate([[0], np.cumsum(ys) / max(ys.sum(), 1)])
    fpr = np.concatenate([[0], np.cumsum(1 - ys) / max((1 - ys).sum(), 1)])
    pop = np.arange(len(ys) + 1) / max(len(ys), 1)
    return fpr, tpr, pop


def _curves_from_deciles(dec):
    """Piecewise-linear curves from a decile table (riskiest bucket first)."""
    n_c, b_c, p_c, r_c = _col(dec, "n"), _col(dec, "bads"), _col(dec, "pred"), _col(dec, "rate")
    if n_c is None or (b_c is None and r_c is None):
        return None
    d = dec.copy()
    d["_n"] = d[n_c].astype(float)
    d["_b"] = d[b_c].astype(float) if b_c else d["_n"] * d[r_c].astype(float)
    d = d.sort_values(p_c if p_c else r_c, ascending=False)
    bad, good, n = d["_b"].to_numpy(), (d["_n"] - d["_b"]).to_numpy(), d["_n"].to_numpy()
    tpr = np.concatenate([[0], np.cumsum(bad) / bad.sum()])
    fpr = np.concatenate([[0], np.cumsum(good) / good.sum()])
    pop = np.concatenate([[0], np.cumsum(n) / n.sum()])
    return fpr, tpr, pop


def _model_curves(model, bundles, splits, out_dir):
    """Exact curves from scores when bundles exist, else the decile approximation (flag returned)."""
    if bundles and splits and model in bundles and OOT1 in splits:
        try:
            df = splits[OOT1]
            return _curves_from_scores(df[TARGET_BAD12], bundles[model].predict_pd(df)), False
        except Exception as exc:
            log.warning("score-based curves failed for %s: %s", model, exc)
    dec = _read(out_dir, f"deciles_{model}_{OOT1}.csv")
    if dec is None:
        return None, False
    c = _curves_from_deciles(dec)
    return (c, True) if c is not None else (None, False)


# ---------- individual charts ----------
def roc_chart(bundles, splits, out_dir, path):
    fig, ax = plt.subplots(figsize=FIG_SQUARE)
    drawn, approx = 0, False
    for i, m in enumerate(ROC_MODELS):
        c, ap = _model_curves(m, bundles, splits, out_dir)
        if c is None:
            continue
        fpr, tpr, _ = c
        ax.plot(fpr, tpr, LINE_STYLES[i % 4], color=GREY_LEVELS[i % 4], label=f"{_label(m)} (AUC {np.trapezoid(tpr, fpr):.3f})")
        drawn, approx = drawn + 1, approx or ap
    if not drawn:
        plt.close(fig)
        return None
    ax.plot([0, 1], [0, 1], color="lightgray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="ROC curves, out-of-time 2013" + (" (from decile tables)" if approx else ""))
    ax.legend(loc="lower right")
    return _save(fig, path)


def cap_chart(bundles, splits, out_dir, path):
    fig, ax = plt.subplots(figsize=FIG_SQUARE)
    drawn = 0
    for i, m in enumerate(ROC_MODELS):
        c, _ = _model_curves(m, bundles, splits, out_dir)
        if c is None:
            continue
        _, tpr, pop = c
        ax.plot(pop, tpr, LINE_STYLES[i % 4], color=GREY_LEVELS[i % 4], label=_label(m))
        drawn += 1
    if not drawn:
        plt.close(fig)
        return None
    ax.plot([0, 1], [0, 1], color="lightgray", label="Random")
    ax.set(xlabel="Share of loans, riskiest first", ylabel="Share of bad loans captured",
           title="Cumulative accuracy profile, out-of-time 2013")
    ax.legend(loc="lower right")
    return _save(fig, path)


def ks_chart(bundles, splits, out_dir, path, model="sc_full"):
    c, _ = _model_curves(model, bundles, splits, out_dir)
    if c is None:
        return None
    fpr, tpr, pop = c
    gap = tpr - fpr
    k = int(np.argmax(gap))
    fig, ax = plt.subplots(figsize=FIG_SQUARE)
    ax.plot(pop, tpr, color="black", label="Cumulative bad share")
    ax.plot(pop, fpr, "--", color="dimgray", label="Cumulative good share")
    ax.vlines(pop[k], fpr[k], tpr[k], color="gray")
    ax.annotate(f"KS = {gap[k]:.3f}", (pop[k], (tpr[k] + fpr[k]) / 2), xytext=(8, 0), textcoords="offset points")
    ax.set(xlabel="Share of loans, riskiest first", ylabel="Cumulative share",
           title=f"KS plot, {_label(model)}, out-of-time 2013")
    ax.legend(loc="lower right")
    return _save(fig, path)


def decile_chart(out_dir, path):
    fig, axes = plt.subplots(1, len(DECILE_MODELS), figsize=FIG_WIDE, sharey=True)
    drawn = 0
    for ax, m in zip(np.atleast_1d(axes), DECILE_MODELS):
        dec = _read(out_dir, f"deciles_{m}_{OOT1}.csv")
        if dec is None or _col(dec, "pred") is None or _col(dec, "rate") is None:
            ax.set_visible(False)
            continue
        dec = dec.sort_values(_col(dec, "pred"))
        x = np.arange(1, len(dec) + 1)
        ax.bar(x, dec[_col(dec, "rate")], color="lightgray", edgecolor="gray", label="Observed bad rate")
        ax.plot(x, dec[_col(dec, "pred")], "o-", color="black", label="Mean predicted PD")
        ax.set(xlabel="Risk decile (1 = lowest predicted PD)", title=_label(m))
        drawn += 1
    if not drawn:
        plt.close(fig)
        return None
    np.atleast_1d(axes)[0].set_ylabel("12-month bad rate")
    np.atleast_1d(axes)[0].legend(loc="upper left")
    fig.suptitle("Decile bad rate versus predicted PD, out-of-time 2013")
    return _save(fig, path)


def calibration_chart(out_dir, path, model=CALIB_MODEL):
    fig, ax = plt.subplots(figsize=FIG_SQUARE)
    drawn, top = 0, 0.0
    for i, s in enumerate(CALIB_SPLITS):
        cal = _read(out_dir, f"calibration_{model}_{s}.csv")
        if cal is None:
            continue
        p_c, a_c = _col(cal, "pred"), _col(cal, ("observed", "bad_rate", "obs_rate", "actual_rate", "actual"))
        if p_c is None or a_c is None:
            continue
        cal = cal.sort_values(p_c)
        ax.plot(cal[p_c], cal[a_c], "o" + LINE_STYLES[i % 4], color=GREY_LEVELS[i % 4], label=SPLIT_LABELS.get(s, s))
        top = max(top, float(cal[[p_c, a_c]].max().max()))
        drawn += 1
    if not drawn:
        plt.close(fig)
        return None
    ax.plot([0, top], [0, top], color="lightgray", label="Perfect calibration")
    ax.set(xlabel="Mean predicted PD", ylabel="Observed 12-month bad rate",
           title=f"Reliability curve, {_label(model)}")
    ax.legend(loc="upper left")
    return _save(fig, path)


def psi_quarter_chart(out_dir, path):
    d = _read(out_dir, "psi_by_quarter.csv")
    if d is None:
        return None
    q_c = _col(d, (QUARTER_COL, "issue_quarter", "period"))
    v_c = _col(d, ("psi", "value"))
    if q_c is None or v_c is None:
        return None
    m_c = _col(d, ("model", "variable", "feature"))
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    groups = d.groupby(m_c) if m_c else [("", d)]
    for i, (name, g) in enumerate(groups):
        ax.plot(g[q_c].astype(str), g[v_c], "o" + LINE_STYLES[i % 4], color=GREY_LEVELS[i % 4], label=str(name) or None, ms=3)
    for lvl, ls in ((config.THRESHOLDS["psi"][0], "--"), (config.THRESHOLDS["psi"][1], ":")):
        ax.axhline(lvl, color="gray", linestyle=ls)
    ax.set(xlabel="Issue quarter", ylabel="PSI versus development score distribution", title="Population stability by issue quarter")
    ticks = ax.get_xticks()
    ax.set_xticks(ticks[:: max(1, len(ticks) // 16)])
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    if m_c:
        ax.legend()
    return _save(fig, path)


def _vintage_frame(out_dir):
    d = _read(out_dir, "by_vintage.csv")
    if d is None or VINTAGE_COL not in d.columns:
        return None
    return d


def _lifetime_by_vintage(splits):
    """Lifetime bad rate per vintage from the random reproduction split (it covers every loan)."""
    parts = [splits[k] for k in (REPRO_TRAIN, REPRO_TEST) if splits and k in splits]
    if not parts:
        return None
    df = pd.concat([p[[VINTAGE, TARGET_LIFETIME]] for p in parts])
    return df.groupby(VINTAGE)[TARGET_LIFETIME].mean()


def vintage_rate_chart(out_dir, path, splits=None):
    d = _vintage_frame(out_dir)
    if d is None or "bad_rate" not in d.columns:
        return None
    d = _champion_rows(d).sort_values(VINTAGE_COL)
    life = _lifetime_by_vintage(splits)
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    if life is not None:
        ax.plot(life.index.astype(int), life.values, "o--", color="dimgray", label="Lifetime bad rate (right-censored for recent vintages)")
    ax.plot(d[VINTAGE_COL].astype(int), d["bad_rate"], "s-", color="black", label="Fixed 12-month bad rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set(xlabel="Issue vintage (year)", ylabel="Bad rate", title="Bad rate by vintage: lifetime flag versus fixed 12-month window")
    ax.legend()
    return _save(fig, path)


def gini_vintage_chart(out_dir, path):
    d = _vintage_frame(out_dir)
    if d is None or "gini" not in d.columns:
        return None
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    for i, (m, g) in enumerate(d.groupby(MODEL_KEY) if MODEL_KEY in d.columns else [("", d)]):
        g = g.sort_values(VINTAGE_COL)
        ax.plot(g[VINTAGE_COL].astype(int), g["gini"], "o" + LINE_STYLES[i % 4], color=GREY_LEVELS[i % 4], label=_label(m) or None)
    ax.axhline(config.THRESHOLDS["gini"][0], color="gray", linestyle=":", label="Gini 0.40 green line")
    ax.set(xlabel="Issue vintage (year)", ylabel="Gini", title="Discrimination by issue vintage")
    ax.legend()
    return _save(fig, path)


def points_chart(out_dir, path):
    d = _read(out_dir, "scorecard_points.csv")
    if d is None or _col(d, ("variable",)) is None or _col(d, ("points",)) is None:
        return None
    d = _champion_rows(d)
    rng = d.groupby("variable")["points"].agg(lambda s: s.max() - s.min()).sort_values().tail(TOP_POINTS_VARS)
    fig, ax = plt.subplots(figsize=FIG_TALL)
    ax.barh(rng.index, rng.values, color="lightgray", edgecolor="gray")
    ax.set(xlabel="Points range (max minus min across bins)", title="Scorecard: points range by variable")
    return _save(fig, path)


def woe_chart(out_dir, path):
    d = _read(out_dir, "binning_report.csv")
    if d is None or not {"variable", "woe", "iv_contrib"} <= set(d.columns):
        return None
    d = _champion_rows(d)
    top = d.groupby("variable")["iv_contrib"].sum().sort_values(ascending=False).head(TOP_WOE_VARS).index.tolist()
    ncol = 3
    nrow = int(np.ceil(len(top) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.2 * nrow), squeeze=False)
    for ax, var in zip(axes.ravel(), top):
        g = d[d["variable"] == var].reset_index(drop=True)
        ax.bar(np.arange(len(g)), g["woe"], color="lightgray", edgecolor="gray")
        ax.axhline(0, color="black", lw=0.8)
        ax.set(title=f"{var} (IV {g['iv_contrib'].sum():.3f})", xlabel="Bin index", ylabel="WoE")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    for ax in axes.ravel()[len(top):]:
        ax.set_visible(False)
    fig.suptitle("Weight of evidence by bin, highest-IV variables")
    return _save(fig, path)


def sc_gini_chart(out_dir, path):
    d = _read(out_dir, "validation_results.csv")
    if d is None or not {"test", "model", "split", "value"} <= set(d.columns):
        return None
    g = d[d["test"].astype(str).str.lower().isin({GINI_TEST})]
    g = g[g["model"].isin(["sc_full", "sc_indep"])]
    if g.empty:
        return None
    piv = g.pivot_table(index="split", columns="model", values="value", aggfunc="first")
    order = [s for s in (DEV_HOLDOUT, OOT1, OOT2) if s in piv.index]
    piv = piv.loc[order]
    fig, ax = plt.subplots(figsize=FIG_SQUARE)
    w = 0.38
    for i, (m, hatch, col) in enumerate((("sc_full", "", "dimgray"), ("sc_indep", "//", "lightgray"))):
        if m in piv.columns:
            bars = ax.bar(np.arange(len(piv)) + (i - 0.5) * w, piv[m], w, label=_label(m), color=col, edgecolor="black", hatch=hatch)
            ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_xticks(np.arange(len(piv)))
    ax.set_xticklabels([SPLIT_LABELS.get(s, s) for s in piv.index])
    ax.set(ylabel="Gini", title="Gini: scorecard with versus without LendingClub risk outputs")
    ax.legend()
    return _save(fig, path)


def lgd_ead_chart(out_dir, path):
    d = _read(out_dir, "lgd_ead_deciles.csv")
    if d is None or not {"mean_pred", "mean_actual", "component"} <= set(d.columns):
        return None
    if "split" in d.columns:
        oot = [x for x in d["split"].unique() if "oot" in str(x).lower() or "out" in str(x).lower()]
        d = d[d["split"] == (oot[0] if oot else d["split"].unique()[-1])]
    parts = [(n, g) for n, g in d.groupby("component") if n in ("lgd_expected", "ccf")]
    if not parts:
        return None
    titles = {"lgd_expected": "LGD (expected value)", "ccf": "Credit conversion factor"}
    fig, axes = plt.subplots(1, len(parts), figsize=(5.2 * len(parts), 4.5), squeeze=False)
    for ax, (name, g) in zip(axes.ravel(), parts):
        g = g.sort_values("decile")
        ax.plot(g["decile"], g["mean_pred"], "o-", color="black", label="Predicted")
        ax.plot(g["decile"], g["mean_actual"], "s--", color="dimgray", label="Actual")
        ax.set(xlabel="Decile of predicted value (1 = highest)", ylabel="Mean value", title=titles[name])
        ax.legend()
    fig.suptitle("LGD and CCF: predicted versus actual by decile, out-of-time defaults")
    return _save(fig, path)


def el_grade_chart(out_dir, path):
    d = _read(out_dir, "el_backtest_12m.csv")
    if d is None or not {GRADE_COL, "el", "realized_loss"} <= set(d.columns):
        return None
    d = d[d[GRADE_COL] != ALL_LABEL]
    if "sample" in d.columns:
        d = d[d["sample"] == SAMPLE_OOT]
    if "status" in d.columns:
        d = d[d["status"].fillna("") == ""]
    if d.empty:
        return None
    agg = d.groupby(GRADE_COL)[["el", "realized_loss"]].sum().sort_index()
    fig, ax = plt.subplots(figsize=FIG_SQUARE)
    x = np.arange(len(agg))
    ax.bar(x - 0.2, agg["el"] / 1e6, 0.4, color="lightgray", edgecolor="black", label="Predicted expected loss")
    ax.bar(x + 0.2, agg["realized_loss"] / 1e6, 0.4, color="dimgray", edgecolor="black", label="Realized loss")
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index)
    ax.set(xlabel="Grade", ylabel="Dollars (millions)", title="12-month expected versus realized loss by grade, 2013 vintage")
    ax.legend()
    return _save(fig, path)


def tornado_chart(out_dir, path):
    d = _read(out_dir, "sensitivity.csv")
    if d is None or not {"scenario", "delta_el_pct"} <= set(d.columns):
        return None
    d = d[d["delta_el_pct"].notna() & (d["scenario"] != "baseline")]
    if MODEL_KEY in d.columns:
        d = d.assign(_pref=(d[MODEL_KEY] != CHAMPION).astype(int)).sort_values("_pref")
    d = d.drop_duplicates("scenario")
    d = d.reindex(d["delta_el_pct"].abs().sort_values().index).tail(15)
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=FIG_TALL)
    ax.barh(d["scenario"], d["delta_el_pct"], color="lightgray", edgecolor="gray")
    ax.axvline(0, color="black", lw=0.8)
    ax.set(xlabel="Change in expected loss (%)", title="Sensitivity of expected loss to shocks and definition changes")
    return _save(fig, path)


def csi_chart(out_dir, path):
    d = _read(out_dir, "psi_csi.csv")
    if d is None or not {"kind", "name", "psi"} <= set(d.columns):
        return None
    d = d[d["kind"] == "csi"]
    if "split" in d.columns and (d["split"] == OOT1).any():
        d = d[d["split"] == OOT1]
    top = d.groupby("name")["psi"].max().sort_values().tail(15)
    if top.empty:
        return None
    fig, ax = plt.subplots(figsize=FIG_TALL)
    ax.barh(top.index.astype(str), top.values, color="lightgray", edgecolor="gray")
    ax.axvline(config.THRESHOLDS["psi"][0], color="gray", linestyle="--")
    ax.axvline(config.THRESHOLDS["psi"][1], color="gray", linestyle=":")
    ax.set(xlabel="Characteristic stability index (dashed 0.10, dotted 0.25)", title="Largest characteristic shifts, development versus 2013")
    return _save(fig, path)


def shap_importance_chart(importance: pd.DataFrame, path, top_k=TOP_SHAP):
    fcol, vcol = _col(importance, ("feature",)), _col(importance, ("mean_abs_shap",))
    top = importance.sort_values(vcol, ascending=False).head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=FIG_TALL)
    ax.barh(top[fcol].astype(str), top[vcol], color="lightgray", edgecolor="gray")
    ax.set(xlabel="Mean absolute SHAP value (log-odds of the raw margin)", title="LightGBM global feature importance")
    return _save(fig, path)


def shap_importance_from_csv(out_dir, path):
    d = _read(out_dir, "shap_importance.csv")
    return None if d is None else shap_importance_chart(d, path)


def shap_dependence(values: np.ndarray, x: pd.Series, name: str, path, max_cats=12):
    """Scatter of feature value versus SHAP (numeric) or mean SHAP by category (categorical)."""
    fig, ax = plt.subplots(figsize=FIG_SQUARE)
    if pd.api.types.is_numeric_dtype(x) and not isinstance(x.dtype, pd.CategoricalDtype):
        ax.scatter(x.astype(float), values, s=5, alpha=0.4, color="dimgray")
        ax.set_xlabel(name)
    else:
        m = pd.Series(values, index=x.astype(str).to_numpy()).groupby(level=0).agg(["mean", "size"])
        m = m.sort_values("size", ascending=False).head(max_cats).sort_values("mean")
        ax.barh(m.index, m["mean"], color="lightgray", edgecolor="gray")
        ax.set_xlabel("Mean SHAP value")
    ax.axhline(0, color="black", lw=0.6) if pd.api.types.is_numeric_dtype(x) else None
    ax.set(ylabel="SHAP value (log-odds)" if pd.api.types.is_numeric_dtype(x) else "", title=f"SHAP dependence: {name}")
    return _save(fig, path)


# ---------- driver ----------
def make_all_charts(bundles=None, splits=None, out_dir=None, chart_dir=None) -> list:
    """Build every chart that has its inputs; returns the list of written paths."""
    out_dir = Path(out_dir) if out_dir is not None else config.OUT_DIR
    chart_dir = Path(chart_dir) if chart_dir is not None else config.CHART_DIR
    chart_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("roc_oot1.png", lambda p: roc_chart(bundles, splits, out_dir, p)),
        ("cap_oot1.png", lambda p: cap_chart(bundles, splits, out_dir, p)),
        ("ks_oot1.png", lambda p: ks_chart(bundles, splits, out_dir, p)),
        ("decile_bad_rate_oot1.png", lambda p: decile_chart(out_dir, p)),
        ("calibration_dev_oot1_oot2.png", lambda p: calibration_chart(out_dir, p)),
        ("psi_by_quarter.png", lambda p: psi_quarter_chart(out_dir, p)),
        ("bad_rate_by_vintage.png", lambda p: vintage_rate_chart(out_dir, p, splits)),
        ("gini_by_vintage.png", lambda p: gini_vintage_chart(out_dir, p)),
        ("scorecard_points_by_variable.png", lambda p: points_chart(out_dir, p)),
        ("woe_top_variables.png", lambda p: woe_chart(out_dir, p)),
        ("gini_full_vs_indep.png", lambda p: sc_gini_chart(out_dir, p)),
        ("lgd_ccf_deciles.png", lambda p: lgd_ead_chart(out_dir, p)),
        ("el_backtest_by_grade.png", lambda p: el_grade_chart(out_dir, p)),
        ("sensitivity_tornado.png", lambda p: tornado_chart(out_dir, p)),
        ("csi_top_variables.png", lambda p: csi_chart(out_dir, p)),
        ("shap_importance.png", lambda p: shap_importance_from_csv(out_dir, p)),
    ]
    written = []
    for fname, job in jobs:
        try:
            res = job(chart_dir / fname)
        except Exception as exc:  # one broken chart must not stop the rest
            log.warning("chart %s failed: %s", fname, exc)
            continue
        if res is not None:
            written.append(Path(res))
        else:
            log.warning("chart skipped: %s", fname)
    return written
