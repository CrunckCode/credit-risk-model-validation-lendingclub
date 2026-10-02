"""Expected-loss backtests (12-month and lifetime) and the Excel input files that depend on PD, LGD and EAD."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from . import columns as C
from . import lgd_ead as L
from . import scorecard as sc_mod
from . import stability

# ===== CONFIG (user inputs) =====
BACKTEST_FIRST_VINTAGE, BACKTEST_LAST_VINTAGE = 2010, 2014
PARTIAL_VINTAGE = 2014
LIFE_TRAIN_LAST_ISSUE = pd.Timestamp("2011-12-01")
LIFE_TEST_YEAR = 2012
ORIGINAL_EL_SHARE, ORIGINAL_EL_USD = 0.0798, 531.9e6        # original notebook: lifetime PD over all loans
ALL_LABEL = "ALL"
MIN_PER_GRADE = 100
QUICK_SAMPLE_LOANS = 1_000
CHAMPION = "sc_full"
OUT_12M, OUT_LIFE = "el_backtest_12m.csv", "el_backtest_lifetime.csv"
OUT_SAMPLE, OUT_PSI_BINS, OUT_REALIZED = "loan_sample.csv", "dev_psi_bins.csv", "realized_loss_by_grade.csv"
FLOAT_FORMAT = "%.17g"
# ===== END CONFIG =====


def score_el(df: pd.DataFrame, pd_hat, lgd_model: L.LgdEadModel) -> pd.DataFrame:
    """PD, LGD, EAD and EL = PD * LGD * EAD per loan."""
    r = lgd_model.predict(df)
    out = pd.DataFrame({C.PRED_PD: np.asarray(pd_hat, float), C.PRED_LGD: r[C.PRED_LGD].to_numpy(), C.PRED_EAD: r[C.PRED_EAD].to_numpy()}, index=df.index)
    out[C.PRED_EL] = out[C.PRED_PD] * out[C.PRED_LGD] * out[C.PRED_EAD]
    return out


def _summarize(g: pd.DataFrame, realized_col: str, bad_col: str) -> pd.Series:
    el, real = g[C.PRED_EL].sum(), g[realized_col].sum()
    funded = g[C.FUNDED_AMNT].sum()
    return pd.Series(dict(n=len(g), funded=funded, mean_pd=g[C.PRED_PD].mean(), bad_rate=g[bad_col].mean(), el=el, realized_loss=real,
                          el_pct_funded=el / funded, realized_pct_funded=real / funded, el_to_realized=el / real if real else np.nan))


def el_backtest_12m(splits: dict, pd_bundle, lgd_model: L.LgdEadModel) -> pd.DataFrame:
    """EL12 = PD12 * LGD * EAD against realized net loss of loans that went bad inside 12 months, by grade x vintage."""
    parts = []
    for s in (C.DEV_TRAIN, C.DEV_HOLDOUT, C.OOT1, C.OOT2):
        d = splits[s]
        d = d[(d[C.VINTAGE] >= BACKTEST_FIRST_VINTAGE) & (d[C.VINTAGE] <= BACKTEST_LAST_VINTAGE)]
        sc = score_el(d, pd_bundle.predict_pd(d), lgd_model)
        x = pd.concat([d[[C.GRADE, C.VINTAGE, C.FUNDED_AMNT, C.TARGET_BAD12, C.NET_LOSS, C.LOAN_STATUS]], sc], axis=1)
        parts.append(x)
    df = pd.concat(parts)
    df[C.GRADE] = df[C.GRADE].astype(str)
    df["realized"] = df[C.NET_LOSS] * df[C.TARGET_BAD12]
    rows = []
    for (v, g), x in df.groupby([C.VINTAGE, C.GRADE]):
        rows.append({C.VINTAGE: v, C.GRADE: g, **_summarize(x, "realized", C.TARGET_BAD12)})
    for v, x in df.groupby(C.VINTAGE):
        rows.append({C.VINTAGE: v, C.GRADE: ALL_LABEL, **_summarize(x, "realized", C.TARGET_BAD12)})
    out = pd.DataFrame(rows).sort_values([C.VINTAGE, C.GRADE]).reset_index(drop=True)
    out["sample"] = np.where(out[C.VINTAGE] <= L.DEV_LAST_ISSUE.year, "development (in-sample PD, LGD, EAD)", "out_of_time")
    out["status"] = np.where(out[C.VINTAGE] == PARTIAL_VINTAGE, "partially unresolved (bad-but-not-yet-charged-off loans carry zero realized loss)", "")
    out.attrs["total_realized"] = float(df["realized"].sum())
    return out


def el_backtest_lifetime(complete36: pd.DataFrame, defaults: pd.DataFrame) -> pd.DataFrame:
    """Scorecard refit on 36-month loans issued 2007-2011, tested on 2012, against realized lifetime net loss."""
    feats = [f for f in C.APPLICATION_FEATURES if f != C.TERM_M]
    train = complete36[complete36[C.ISSUE_D] <= LIFE_TRAIN_LAST_ISSUE]
    test = complete36[complete36[C.ISSUE_D].dt.year == LIFE_TEST_YEAR]
    sc = sc_mod.fit_scorecard(train, C.TARGET_LIFETIME, feats)
    lgd_model = L.fit_lgd_ead(defaults, last_issue=LIFE_TRAIN_LAST_ISSUE)      # no 2012 defaults in the LGD/EAD fit
    rows = []
    for label, d in (("train_2007_2011 (in-sample)", train), ("test_2012", test)):
        x = pd.concat([d[[C.GRADE, C.FUNDED_AMNT, C.TARGET_LIFETIME, C.NET_LOSS]], score_el(d, sc.predict_pd(d), lgd_model)], axis=1)
        x[C.GRADE] = x[C.GRADE].astype(str)
        for g, y in list(x.groupby(C.GRADE)) + [(ALL_LABEL, x)]:
            rows.append({"sample": label, C.GRADE: g, **_summarize(y, C.NET_LOSS, C.TARGET_LIFETIME)})
    out = pd.DataFrame(rows)
    out = pd.concat([out, pd.DataFrame([{"sample": "original_notebook (all loans, lifetime PD)", C.GRADE: ALL_LABEL, "el_pct_funded": ORIGINAL_EL_SHARE,
                                          "el": ORIGINAL_EL_USD}])], ignore_index=True)
    out.attrs["n_train"], out.attrs["n_test"], out.attrs["variables"] = len(train), len(test), sc.variables
    return out


def run_el(splits: dict, bundles: dict, lgd_result: dict, out_dir=None) -> dict:
    out = Path(out_dir or config.OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    b12 = el_backtest_12m(splits, bundles[CHAMPION], lgd_result["model"])
    b12.to_csv(out / OUT_12M, index=False)
    life = el_backtest_lifetime(splits[C.COMPLETE36], lgd_result["parts"][L.SPLIT_DEV])
    life.to_csv(out / OUT_LIFE, index=False)
    return dict(el12=b12, lifetime=life)


# ---------- Excel inputs ----------
def stratified_sample(df: pd.DataFrame, n_total: int, seed: int = None) -> pd.DataFrame:
    """Proportional by grade with a floor so thin grades still appear; deterministic."""
    seed = config.SEED if seed is None else seed
    g = df[C.GRADE].astype(str)
    share = g.value_counts(normalize=True)
    alloc = {k: min(int((g == k).sum()), max(MIN_PER_GRADE if n_total >= 2000 else MIN_PER_GRADE // 5, int(round(share[k] * n_total)))) for k in share.index}
    parts = [df[g == k].sample(n=m, random_state=seed) for k, m in alloc.items()]
    return pd.concat(parts).sort_index()


def write_excel_inputs(bundles: dict, splits: dict, lgd_model: L.LgdEadModel, out_dir=None, quick: bool = False) -> dict:
    out = Path(out_dir or config.OUT_DIR) / "excel_inputs"
    out.mkdir(parents=True, exist_ok=True)
    b = bundles[CHAMPION]
    sc = b.scorecard
    oot1 = splits[C.OOT1]
    samp = stratified_sample(oot1, QUICK_SAMPLE_LOANS if quick else config.EXCEL_SAMPLE_LOANS)
    sel = score_el(samp, b.predict_pd(samp), lgd_model)
    pts = sc.points_by_variable(samp).add_prefix("pts_")
    cols = [C.LOAN_ID] + list(dict.fromkeys(list(sc.variables) + [C.TARGET_BAD12, C.FUNDED_AMNT, C.NET_LOSS, C.GRADE, C.SUB_GRADE]))
    ls = samp[cols].copy()
    for c in sc.variables:
        if pd.api.types.is_numeric_dtype(ls[c]):
            ls[c] = ls[c].astype("float64")
        else:
            ls[c] = ls[c].astype(object).where(ls[c].notna(), np.nan)
    ls = pd.concat([ls, sel[[C.PRED_PD]], pts], axis=1)
    ls[C.SCORE] = sc.points(samp)
    ls = pd.concat([ls, sel[[C.PRED_LGD, C.PRED_EAD, C.PRED_EL]]], axis=1)
    ls["realized_loss_12m"] = ls[C.NET_LOSS] * ls[C.TARGET_BAD12]
    ls.to_csv(out / OUT_SAMPLE, index=False, float_format=FLOAT_FORMAT)

    dev_score = sc.points(splits[C.DEV_TRAIN])
    t = stability.psi_table(dev_score, dev_score)
    t = t.assign(dev_count=np.round(t["dev_share"] * len(dev_score)).astype(int))[["bin", "lo", "hi", "dev_share", "dev_count"]]
    t.to_csv(out / OUT_PSI_BINS, index=False, float_format=FLOAT_FORMAT)

    full = pd.DataFrame({C.GRADE: oot1[C.GRADE].astype(str).to_numpy(), "n": 1, C.FUNDED_AMNT: oot1[C.FUNDED_AMNT].to_numpy(),
                         "realized": (oot1[C.NET_LOSS] * oot1[C.TARGET_BAD12]).to_numpy(), C.TARGET_BAD12: oot1[C.TARGET_BAD12].to_numpy()})
    pop = full.groupby(C.GRADE).sum().add_prefix("oot1_")
    sm = ls.assign(**{C.GRADE: ls[C.GRADE].astype(str)}).groupby(C.GRADE).agg(
        sample_n=(C.LOAN_ID, "size"), sample_funded=(C.FUNDED_AMNT, "sum"), sample_bad12=(C.TARGET_BAD12, "sum"), sample_realized_loss=("realized_loss_12m", "sum"))
    rl = pop.join(sm).reset_index()
    rl.to_csv(out / OUT_REALIZED, index=False, float_format=FLOAT_FORMAT)
    return dict(loan_sample=ls, psi_bins=t, realized=rl)
