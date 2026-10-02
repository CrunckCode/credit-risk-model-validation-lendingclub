"""Builds docs/Credit_Risk_Model_Validation.md and .docx plus README.md from python/outputs; rerunnable.

Every result number is read from outputs/*.csv|json (or recomputed from the cached parquet and stored in
docs/_facts_cache.json); none is typed by hand. Run from anywhere: py -3 docs/build_docs.py
Placeholders: @@key@@ (value), @@tbl:name@@ (table), @@r:test:model:split:fmt@@ (validation_results.csv),
@@l:test:model:split@@ (its traffic light), @@bm:kind:split:model_a:col:fmt@@ (benchmark.csv).
"""
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# ===== CONFIG (user inputs) =====
ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR, PY_DIR = ROOT / "docs", ROOT / "python"
OUT_DIR = PY_DIR / "outputs"
MD_PATH = DOCS_DIR / "Credit_Risk_Model_Validation.md"
DOCX_PATH = DOCS_DIR / "Credit_Risk_Model_Validation.docx"
README_PATH = ROOT / "README.md"
REFERENCE_DOCX = DOCS_DIR / "reference.docx"
FACTS_CACHE = DOCS_DIR / "_facts_cache.json"
EXCEL_RECON_JSON = OUT_DIR / "excel_reconciliation.json"
CHART_REL = "../python/outputs/charts"
AUTHOR = "Deepak Chaudhary"
DOC_TITLE = "Independent Validation of a LendingClub Credit Risk Model: PD, LGD, EAD and Expected Loss"
TOC_DEPTH = 2
FACTS_VERSION = "v1"
CHAMP, INDEP = "sc_full", "sc_indep"
LOAN_SAMPLE_CSV = OUT_DIR / "excel_inputs" / "loan_sample.csv"
# ===== END CONFIG =====

sys.path.insert(0, str(PY_DIR))
from credit_validation import config  # noqa: E402
from credit_validation import columns as C  # noqa: E402


# ---------------------------------------------------------------- formatting helpers
def nz(x):
    return x is None or (isinstance(x, float) and np.isnan(x))


def fx(x, d=3):
    return "n/a" if nz(x) else f"{x:,.{d}f}"


def pc(x, d=2):
    return "n/a" if nz(x) else f"{100 * x:.{d}f}%"


def ni(x):
    return "n/a" if nz(x) else f"{int(round(x)):,}"


def pv(x):
    if nz(x):
        return "n/a"
    return f"{x:.1e}" if x < 1e-3 else f"{x:.3f}"


def usd_m(x, d=1):
    return "n/a" if nz(x) else f"USD {x / 1e6:,.{d}f} million"


def fmt(x, code):
    if code == "pv":
        return pv(x)
    if code.startswith("f"):
        return fx(x, int(code[1:]))
    if code.startswith("p"):
        return pc(x, int(code[1:]))
    if code.startswith("s"):
        return f"{x:+.{int(code[1:])}f}"
    return {"pv": pv, "i": ni, "e": lambda v: f"{v:.1e}", "m": usd_m}[code](x)


def md_table(df, fmts=None, headers=None, first_left=True):
    fmts, headers = fmts or {}, headers or {}
    cols = list(df.columns)
    def cell(c, v):
        if c in fmts and fmts[c] is not str:
            return fmts[c](v)
        if isinstance(v, (float, np.floating)) and not np.isnan(v) and float(v).is_integer():
            return str(int(v))
        return str(v)
    cells = [[cell(c, r[c]).replace("|", "/") for c in cols] for _, r in df.iterrows()]
    head = [str(headers.get(c, c)) for c in cols]
    widths = [min(max(3, len(head[j]), *(len(row[j]) for row in cells)), 40) for j in range(len(cols))]
    sep = [(":" + "-" * (widths[j] - 1)) if (first_left and j == 0) else ("-" * (widths[j] - 1) + ":") for j in range(len(cols))]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(sep) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in cells]
    return "\n".join(lines)


_CSV = {}


def rcsv(name):
    if name not in _CSV:
        _CSV[name] = pd.read_csv(OUT_DIR / name)
    return _CSV[name]


def rjson(name):
    return json.loads((OUT_DIR / name).read_text())


def vr(test, model, split, col="value"):
    v = rcsv("validation_results.csv")
    s = v[(v["test"] == test) & (v["model"] == model) & (v["split"] == split)]
    return np.nan if s.empty else s[col].iloc[0]


def bm(kind, split, model_a, col):
    b = rcsv("benchmark.csv")
    s = b[(b["kind"] == kind) & (b["split"] == split) & (b["model_a"] == model_a)]
    return np.nan if s.empty else s[col].iloc[0]


# ---------------------------------------------------------------- facts recomputed from the cached loan parquet
def panel_facts():
    ps = sorted(config.INTERIM_DIR.glob("loans_*.parquet"))
    if not ps:
        return json.loads(FACTS_CACHE.read_text()) if FACTS_CACHE.exists() else {}
    p = ps[-1]
    stamp = p.stat().st_mtime
    if FACTS_CACHE.exists():
        cached = json.loads(FACTS_CACHE.read_text())
        if cached.get("stamp") == stamp and cached.get("version") == FACTS_VERSION:
            return cached
    from sklearn.linear_model import LogisticRegression
    from scipy import stats
    from credit_validation import features, targets, splits as spl, lgd_ead, scorecard as scm, stability
    from credit_validation.stats_models import LogisticRegressionWithPValues
    raw = pd.read_parquet(p)
    df = features.build_features(targets.add_targets(raw))
    f = {"stamp": stamp, "version": FACTS_VERSION, "n_loans": int(len(df))}
    st = df[C.LOAN_STATUS].astype(str)
    f["status_counts"] = {k: int(v) for k, v in st.value_counts().items()}
    f["issue_min"], f["issue_max"] = str(df[C.ISSUE_D].min().date()), str(df[C.ISSUE_D].max().date())
    f["last_pymnt_max"] = str(df[C.LAST_PYMNT_D].max().date())
    bad = st.isin(config.BAD_STATUSES)
    f["bad_orig_n"], f["bad_orig_rate"] = int(bad.sum()), float(bad.mean())
    grace = st.eq("In Grace Period")
    f["grace_n"] = int(grace.sum())
    f["grace_in_window_n"] = int((grace & (df[C.MOB_LAST_PAY] < 12) & df[C.OBS12]).sum())
    f["grace_max_issue"] = str(df.loc[grace, C.ISSUE_D].max().date())
    f["grace_min_issue"] = str(df.loc[grace, C.ISSUE_D].min().date())
    f["dnmcp_n"] = int(st.str.startswith(config.DNMCP_PREFIX).sum())
    yr = df[C.ISSUE_D].dt.year
    f["w_share_by_year"] = {int(k): float(v) for k, v in df[C.INITIAL_LIST_STATUS].astype(str).eq("w").groupby(yr).mean().items()}
    o = df[df[C.OBS12]]
    ob = o[C.LOAN_STATUS].isin(config.BAD_STATUSES)
    g = pd.DataFrame({"vint": o[C.VINTAGE], "bad_status": ob, "bad12": o[C.TARGET_BAD12].astype(bool)}).groupby("vint").sum()
    f["bad_status_vs_bad12"] = {int(k): [int(r["bad_status"]), int(r["bad12"])] for k, r in g.iterrows()}
    sp = spl.time_splits(df)
    dev, oot2 = sp[C.DEV_TRAIN], sp[C.OOT2]
    cats = ["f", "w"]
    es = dev[C.INITIAL_LIST_STATUS].astype(str).value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
    as_ = oot2[C.INITIAL_LIST_STATUS].astype(str).value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
    f["psi_ex"] = {"cats": cats, "dev": es.tolist(), "oot2": as_.tolist(), "psi": stability.psi_from_shares(es, as_)}
    specs = scm.fit_bins_for(dev, dev[C.TARGET_BAD12].to_numpy(), list(C.APPLICATION_FEATURES))
    f["iv_all"] = {k: float(v.iv) for k, v in specs.items()}
    # as-built Wald p-values (no intercept in the Fisher matrix, penalized C=1 fit) versus the corrected class, LGD stage 1
    defaults = lgd_ead.default_population(df)
    dd = lgd_ead.split_defaults(defaults)[lgd_ead.SPLIT_DEV]
    spec = lgd_ead.DesignSpec().fit(dd)
    X = spec.transform(dd)
    y = dd[C.RECOVERED_ANY].to_numpy()
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(X.to_numpy(), y)
    pr = lr.predict_proba(X.to_numpy())[:, 1]
    F = (X.to_numpy() * (pr * (1 - pr))[:, None]).T @ X.to_numpy()
    se_o = np.sqrt(np.diag(np.linalg.inv(F)))
    p_o = 2 * stats.norm.sf(np.abs(lr.coef_[0] / se_o))
    sm = LogisticRegressionWithPValues().fit(X, y).summary().iloc[1:]
    p_c = sm["p_value"].to_numpy()
    f["wald"] = {"n_terms": int(len(p_c)), "sig_corrected": int((p_c < 0.05).sum()), "sig_asbuilt": int((p_o < 0.05).sum()),
                 "verdict_differs": int(((p_c < 0.05) != (p_o < 0.05)).sum()), "median_se_ratio": float(np.median(se_o / sm["se"].to_numpy())),
                 "max_coef_shrink": float(np.max(np.abs(lr.coef_[0]) / np.maximum(np.abs(sm["coef"].to_numpy()), 1e-12)))}
    f = json.loads(json.dumps(f))
    FACTS_CACHE.write_text(json.dumps(f, indent=1))
    return f


# ---------------------------------------------------------------- values and tables
def light_cell(test, model, split, code):
    v, l = vr(test, model, split), vr(test, model, split, "light")
    return "n/a" if nz(v) else f"{fmt(v, code)} ({l})"


def el_rule(ratio):
    gap = abs(1 - ratio)
    return "Green" if gap <= 0.10 else ("Amber" if gap <= 0.25 else "Red")


def _tl_champion(V, T):
    rows = []
    sp = (C.DEV_HOLDOUT, C.OOT1, C.OOT2)
    spec = [("Gini", "gini", "f3"), ("KS", "ks", "f3"), ("Decile rank-order inversions", "decile_rank_inversions", "i"),
            ("Central tendency, relative deviation", "central_tendency", "p1"), ("Calibration slope", "cal_slope", "f3"),
            ("Hosmer-Lemeshow p-value", "hl_p", "pv"), ("Grade binomial, smallest p (Jeffreys)", "binomial_grade_min_p", "pv"),
            ("Score PSI vs development", "psi_score", "f4")]
    for lab, t, code in spec:
        rows.append([lab] + [light_cell(t, CHAMP, s, code) for s in sp])
    rows.append(["Gini decay vs holdout", "n/a"] + [light_cell("gini_decay", CHAMP, s, "p1") for s in (C.OOT1, C.OOT2)])
    rows.append(["Largest feature CSI", "n/a"] + [light_cell("csi_max", "all", s, "f3") for s in (C.OOT1, C.OOT2)])
    rows.append(["AUC minus sub_grade benchmark", "n/a"] + [f"{bm('vs_sub_grade', s, CHAMP, 'd_auc'):+.4f} ({bm('vs_sub_grade', s, CHAMP, 'light')})" for s in (C.OOT1, C.OOT2)])
    e = rcsv("el_backtest_12m.csv")
    cells = []
    for yr in (2013, 2014):
        r = e[(e["vintage"] == yr) & (e["grade"] == "ALL")].iloc[0]
        cells.append(f"{r['el_to_realized']:.3f} ({el_rule(r['el_to_realized'])})")
    rows.append(["12-month EL / realized loss", "n/a"] + cells)
    T["tl_champion"] = md_table(pd.DataFrame(rows, columns=["t", "h", "o1", "o2"]),
                                headers=dict(t="Test", h="Dev holdout", o1="OOT 2013", o2="OOT 2014"))


def _core(V, T, X):
    ds, meta, pf = X["ds"], X["meta"], X["pf"]
    sr = ds["split_rows"]
    for k, v in sr.items():
        V[f"n_{k}"] = ni(v)
    V["n_loans"] = ni(ds["n_loans"])
    sm = meta["splits"]
    for k, v in sm.items():
        V[f"br_{k}"] = pc(v["bad_rate"], 2)
    V["source"] = ds["data_source"]
    V["today"] = date.today().isoformat()
    V["doc_title"], V["author"] = DOC_TITLE, AUTHOR
    V["seed"] = str(config.SEED)
    V["pdo"], V["base_score"], V["base_odds"] = str(config.PDO), str(config.BASE_SCORE), str(config.BASE_ODDS)
    s = meta["models"][CHAMP]
    V["factor"], V["offset"], V["intercept"] = fx(s["factor"], 4), fx(s["offset"], 4), fx(s["intercept"], 4)
    V["k_vars"] = str(len(s["scorecard_variables"]))
    V["sc_vars"] = ", ".join(f"`{v}`" for v in s["scorecard_variables"])
    V["ind_vars"] = ", ".join(f"`{v}`" for v in meta["models"][INDEP]["scorecard_variables"])
    V["n_dropped_iv"] = str(sum(1 for d in s["dropped"] if d[1].startswith("IV")))
    V["n_dropped_corr"] = str(sum(1 for d in s["dropped"] if d[1].startswith("corr")))
    V["n_dropped_p"] = str(sum(1 for d in s["dropped"] if d[1].startswith("p ")))
    V["light_counts"] = ", ".join(f"{k} {v}" for k, v in meta["light_counts"].items() if k in ("Green", "Amber", "Red"))
    V["recal_shift"] = fx(meta["models"]["sc_full_recal"]["intercept_shift"], 4)
    for m in ("xgb", "lgbm"):
        V[f"{m}_cal"] = meta["models"][m].get("calibration_method", "n/a")
        V[f"{m}_trees"] = str(meta["models"][m].get("params", {}).get("n_estimators", "n/a"))
    # target audit
    ta = rcsv("target_audit.csv")
    T["target_audit"] = md_table(ta[["vintage", "n_loans", "n_obs12", "lifetime_bad_rate", "bad12_rate", "share_current", "share_60m"]],
                                 dict(vintage=lambda x: str(int(x)), n_loans=ni, n_obs12=ni, lifetime_bad_rate=pc, bad12_rate=pc, share_current=pc, share_60m=pc),
                                 dict(vintage="Issue year", n_loans="Loans", n_obs12="Observed 12m", lifetime_bad_rate="Lifetime bad rate",
                                      bad12_rate="12-month bad rate", share_current="Still Current", share_60m="60-month share"))
    r = ta.set_index("vintage")
    V["life07"], V["life14"] = pc(r.loc[2007, "lifetime_bad_rate"], 1), pc(r.loc[2014, "lifetime_bad_rate"], 1)
    sub = r.loc[2010:2014, "bad12_rate"]
    V["b12_lo"], V["b12_hi"] = pc(sub.min(), 2), pc(sub.max(), 2)
    V["cur14"] = pc(r.loc[2014, "share_current"], 1)
    # splits table
    rows = [("Development train", "dev_train", "Issued 2007-06 to 2012-12, 80% stratified"),
            ("Development holdout", "dev_holdout", "Same window, 20% stratified"),
            ("OOT 1", "oot1", "Issued 2013"), ("OOT 2", "oot2", "Issued 2014-01 to 2014-11"),
            ("Complete 36-month cohort", "complete36", "36-month loans issued to 2012-12"),
            ("Repro train", "repro_train", "Random 80% of all loans"), ("Repro test", "repro_test", "Random 20% of all loans")]
    T["splits"] = md_table(pd.DataFrame([dict(n=a, rows=sr[k], tg=sm[k]["target"], br=sm[k]["bad_rate"], w=w) for a, k, w in rows]),
                           dict(rows=ni, br=lambda x: pc(x, 2)), dict(n="Sample", rows="Loans", tg="Target", br="Bad rate", w="Window"))
    if pf:
        V["issue_min"], V["issue_max"], V["status_date"] = pf["issue_min"], pf["issue_max"], pf["last_pymnt_max"]
        V["bad_orig_n"], V["bad_orig_rate"] = ni(pf["bad_orig_n"]), pc(pf["bad_orig_rate"], 2)
        V["grace_n"], V["grace_win"] = ni(pf["grace_n"]), ni(pf["grace_in_window_n"])
        V["grace_min"], V["grace_max"] = pf["grace_min_issue"], pf["grace_max_issue"]
        V["dnmcp_n"] = ni(pf["dnmcp_n"])
        sc = pd.DataFrame(sorted(pf["status_counts"].items(), key=lambda kv: -kv[1]), columns=["status", "n"])
        sc["coded"] = ["Bad" if s_ in config.BAD_STATUSES else "Good" for s_ in sc["status"]]
        T["status_counts"] = md_table(sc[["status", "n", "coded"]], dict(n=ni), dict(status="Loan status", n="Loans", coded="Original coding"))
        w = pf["w_share_by_year"]
        V["w_2011"], V["w_2012"], V["w_2013"], V["w_2014"] = (pc(w[str(y)], 1) for y in (2011, 2012, 2013, 2014))
        b = pd.DataFrame([dict(v=int(k), bs=a, b12=c) for k, (a, c) in pf["bad_status_vs_bad12"].items()])
        b["sh"] = b["b12"] / b["bs"]
        T["bad_vs_bad12"] = md_table(b, dict(v=lambda x: str(int(x)), bs=ni, b12=ni, sh=pc),
                                     dict(v="Issue year", bs="Loans with a bad status", b12="Of which bad within 12 months", sh="Share"))
        pe = pf["psi_ex"]
        t = pd.DataFrame(dict(cat=pe["cats"], e=pe["dev"], a=pe["oot2"]))
        t["d"] = t["a"] - t["e"]
        t["l"] = np.log(np.maximum(t["a"], 1e-4) / np.maximum(t["e"], 1e-4))
        t["c"] = t["d"] * t["l"]
        T["psi_ex"] = md_table(t, dict(e=pc, a=pc, d=lambda x: f"{100 * x:+.2f} pp", l=lambda x: fx(x, 4), c=lambda x: fx(x, 4)),
                               dict(cat="initial_list_status", e="Development share", a="OOT 2014 share", d="Difference", l="ln(actual / expected)", c="Contribution"))
        V["psi_ex_total"] = fx(pe["psi"], 4)
        iv = pf["iv_all"]
        kept = set(s["scorecard_variables"])
        why = {d[0]: d[1] for d in s["dropped"]}
        band = lambda v: "Not useful" if v < 0.02 else "Weak" if v < 0.1 else "Medium" if v < 0.3 else "Strong" if v < 0.5 else "Suspicious"
        ivt = pd.DataFrame([dict(v=k, iv=x, b=band(x), st="Kept" if k in kept else "Dropped", why=("" if k in kept else why.get(k, ""))) for k, x in
                            sorted(iv.items(), key=lambda kv: -kv[1])])
        T["iv_all"] = md_table(ivt, dict(iv=lambda x: fx(x, 4)), dict(v="Variable", iv="IV (dev train)", b="Band", st="SC_FULL", why="Reason dropped"))
        V["iv_top"], V["iv_top_v"] = fx(max(iv.values()), 3), max(iv, key=iv.get)
        wd = pf["wald"]
        V["wald_n"], V["wald_sig_c"], V["wald_sig_a"], V["wald_diff"] = (str(wd[k]) for k in ("n_terms", "sig_corrected", "sig_asbuilt", "verdict_differs"))
        V["wald_se_ratio"], V["wald_shrink"] = fx(wd["median_se_ratio"], 2), fx(wd["max_coef_shrink"], 2)
    # woe example from binning report
    br = rcsv("binning_report.csv")
    b = br[(br["model"] == CHAMP) & (br["variable"] == C.INQ_LAST_6MTHS)].copy()
    pres = b[b["n"] > 0]
    k = len(pres)
    B_, G_ = pres["bads"].sum(), (pres["n"] - pres["bads"]).sum()
    r0 = pres.iloc[0]
    db, dg = (r0["bads"] + 0.5) / (B_ + 0.5 * k), (r0["n"] - r0["bads"] + 0.5) / (G_ + 0.5 * k)
    V["woe_B"], V["woe_G"], V["woe_k"] = ni(B_), ni(G_), str(k)
    V["woe_bad0"], V["woe_good0"] = ni(r0["bads"]), ni(r0["n"] - r0["bads"])
    V["woe_db"], V["woe_dg"] = fx(db, 5), fx(dg, 5)
    V["woe_val"], V["woe_rep"] = fx(np.log(dg / db), 4), fx(r0["woe"], 4)
    V["woe_iv0"] = fx((dg - db) * np.log(dg / db), 4)
    V["woe_iv_var"] = fx(b["iv_contrib"].sum(), 4)
    T["woe_ex"] = md_table(pres[["bin", "n", "bads", "bad_rate", "woe", "iv_contrib"]],
                           dict(n=ni, bads=ni, bad_rate=pc, woe=lambda x: fx(x, 4), iv_contrib=lambda x: fx(x, 4)),
                           dict(bin="inq_last_6mths bin", n="Loans", bads="Bad", bad_rate="Bad rate", woe="WoE", iv_contrib="IV contribution"))
    # binning of annual income as second example
    b2 = br[(br["model"] == CHAMP) & (br["variable"] == C.ANNUAL_INC)]
    T["woe_inc"] = md_table(b2[b2["n"] > 0][["bin", "n", "bads", "bad_rate", "woe", "iv_contrib"]],
                            dict(n=ni, bads=ni, bad_rate=pc, woe=lambda x: fx(x, 4), iv_contrib=lambda x: fx(x, 4)),
                            dict(bin="annual_inc bin", n="Loans", bads="Bad", bad_rate="Bad rate", woe="WoE", iv_contrib="IV contribution"))
    # scorecard summary
    pts = rcsv("scorecard_points.csv")
    pts = pts[pts["model"] == CHAMP]
    rows = []
    for v in s["scorecard_variables"]:
        g = pts[pts["variable"] == v]
        nb = int((br[(br["model"] == CHAMP) & (br["variable"] == v)]["n"] > 0).sum())
        rows.append(dict(v=v, iv=s["iv"][v], nb=nb, coef=g["coef"].iloc[0], lo=g["points"].min(), hi=g["points"].max(), p=s["pvalues"][v], vif=s["vif"][v]))
    sm_ = pd.DataFrame(rows)
    sm_["rg"] = sm_["hi"] - sm_["lo"]
    T["sc_summary"] = md_table(sm_[["v", "iv", "nb", "coef", "p", "vif", "lo", "hi", "rg"]],
                               dict(iv=lambda x: fx(x, 4), nb=str, coef=lambda x: fx(x, 4), p=pv, vif=lambda x: fx(x, 2), lo=lambda x: fx(x, 1),
                                    hi=lambda x: fx(x, 1), rg=lambda x: fx(x, 1)),
                               dict(v="Variable", iv="IV", nb="Bins", coef="Coefficient", p="Wald p", vif="VIF", lo="Min points", hi="Max points", rg="Range"))
    V["rg_top"], V["rg_top_v"] = fx(sm_["rg"].max(), 1), sm_.loc[sm_["rg"].idxmax(), "v"]
    V["rg_low"], V["rg_low_v"] = fx(sm_["rg"].min(), 1), sm_.loc[sm_["rg"].idxmin(), "v"]
    # scaling map
    fac, off = s["factor"], s["offset"]
    rows = []
    for sc_ in (500, 550, 573.6, 600, 650, 700):
        odds = config.BASE_ODDS * 2 ** ((sc_ - config.BASE_SCORE) / config.PDO)
        rows.append(dict(score=sc_, odds=odds, pd=1 / (1 + odds)))
    T["score_map"] = md_table(pd.DataFrame(rows), dict(score=lambda x: fx(x, 1), odds=lambda x: fx(x, 1) + " : 1", pd=lambda x: pc(x, 3)),
                              dict(score="Score", odds="Good : bad odds", pd="Implied PD"))
    V["pd_at_600"] = pc(1 / (1 + config.BASE_ODDS), 2)
    V["avg_score"] = fx(config.BASE_SCORE + config.PDO * np.log2(0.95226 / 0.04774 / config.BASE_ODDS), 1)
    _worked_loan(V, T, s, pts)


def _worked_loan(V, T, s, pts):
    ls = pd.read_csv(LOAN_SAMPLE_CSV)
    L = ls.iloc[0]
    rows = []
    tot_cw = 0.0
    for v in s["scorecard_variables"]:
        t = pts[pts["variable"] == v]
        x = L[v]
        if t["categories"].notna().any():
            hit = t[t["categories"].fillna("").apply(lambda c: str(x) in c.split("|"))].iloc[0]
            lab, valtxt = hit["categories"], str(x)
        else:
            if nz(x):
                hit = t[t["lo"].isna()].iloc[0]
                lab, valtxt = "Missing", "missing"
            else:
                hit = t[t["lo"].notna() & (t["lo"] < x) & (x <= t["hi"])].iloc[0]
                lo = "-inf" if np.isinf(hit["lo"]) else f"{hit['lo']:g}"
                hi = "inf" if np.isinf(hit["hi"]) else f"{hit['hi']:g}"
                lab, valtxt = f"({lo}, {hi}]", f"{x:.6g}"
        cw = hit["coef"] * hit["woe"]
        tot_cw += cw
        rows.append(dict(v=v, x=valtxt, b=lab[:34], w=hit["woe"], c=hit["coef"], cw=cw, p=hit["points"]))
    tb = pd.DataFrame(rows)
    T["wl_table"] = md_table(tb, dict(w=lambda x: fx(x, 4), c=lambda x: fx(x, 4), cw=lambda x: fx(x, 4), p=lambda x: fx(x, 2)),
                             dict(v="Variable", x="Value", b="Bin", w="WoE", c="Coefficient", cw="Coef x WoE", p="Points"))
    margin = s["intercept"] + tot_cw
    V["wl_id"], V["wl_grade"] = str(int(L["id"])), str(L["sub_grade"])
    V["wl_sum_pts"], V["wl_score"] = fx(tb["p"].sum(), 3), fx(L["score"], 3)
    V["wl_margin"], V["wl_cw"] = fx(margin, 4), fx(tot_cw, 4)
    V["wl_pd"], V["wl_pd_file"] = pc(1 / (1 + np.exp(-margin)), 3), pc(L["pred_pd"], 3)
    V["wl_odds"] = fx(np.exp(-margin), 1)
    V["wl_k"] = str(len(tb))
    V["wl_off_k"] = fx(s["offset"] / len(tb), 3)


def _results(V, T, X):
    rows = []
    for m, lab in (("sc_full", "SC_FULL scorecard"), ("sc_indep", "SC_INDEP scorecard"), ("xgb", "XGBoost"), ("lgbm", "LightGBM"),
                   ("repro_original", "Reproduced original")):
        for s in (C.DEV_HOLDOUT, C.OOT1, C.OOT2):
            if nz(vr("auc", m, s)):
                continue
            rows.append(dict(m=lab, s=s, auc=vr("auc", m, s), g=vr("gini", m, s), k=vr("ks", m, s), br=vr("brier", m, s), sk=vr("brier_skill", m, s)))
    T["disc"] = md_table(pd.DataFrame(rows), dict(auc=lambda x: fx(x, 4), g=lambda x: fx(x, 4), k=lambda x: fx(x, 4), br=lambda x: fx(x, 5), sk=lambda x: fx(x, 4)),
                         dict(m="Model", s="Sample", auc="AUC", g="Gini", k="KS", br="Brier", sk="Brier skill"))
    rows = []
    for m in ("sc_full", "sc_indep", "xgb", "lgbm", "sc_full_recal"):
        for s in (C.OOT1, C.OOT2):
            if nz(vr("mean_pd", m, s)):
                continue
            rows.append(dict(m=m, s=s, mp=vr("mean_pd", m, s), ob=vr("obs_rate", m, s), sl=vr("cal_slope", m, s), ic=vr("cal_intercept", m, s),
                             hl=vr("hl_p", m, s), hs=vr("hl_stat", m, s)))
    T["calsum"] = md_table(pd.DataFrame(rows), dict(mp=pc, ob=pc, sl=lambda x: fx(x, 3), ic=lambda x: fx(x, 3), hl=pv, hs=lambda x: fx(x, 1)),
                           dict(m="Model", s="Sample", mp="Mean PD", ob="Observed", sl="Slope", ic="Intercept", hl="HL p", hs="HL statistic"))
    d = rcsv("deciles_sc_full_oot1.csv")
    T["dec_oot1"] = md_table(d[["decile", "n", "bads", "bad_rate", "mean_pd", "cum_bad_share", "cum_good_share", "ks_at_decile", "lift"]],
                             dict(decile=str, n=ni, bads=ni, bad_rate=pc, mean_pd=pc, cum_bad_share=pc, cum_good_share=pc, ks_at_decile=lambda x: fx(x, 4), lift=lambda x: fx(x, 2)),
                             dict(decile="Decile (1 = riskiest)", n="Loans", bads="Bad", bad_rate="Bad rate", mean_pd="Mean PD", cum_bad_share="Cum. bads",
                                  cum_good_share="Cum. goods", ks_at_decile="Gap", lift="Lift"))
    V["dec1_rate"], V["dec1_lift"], V["dec1_share"] = pc(d.loc[0, "bad_rate"], 2), fx(d.loc[0, "lift"], 2), pc(d.loc[0, "cum_bad_share"], 1)
    V["dec10_rate"] = pc(d.loc[9, "bad_rate"], 2)
    V["ks_dec_at"] = str(int(d.loc[d["ks_at_decile"].idxmax(), "decile"]))
    # hosmer lemeshow worked table
    c = rcsv("calibration_sc_full_oot1.csv").copy()
    c["den"] = c["n"] * c["mean_pd"] * (1 - c["mean_pd"])
    c["contrib"] = (c["bads"] - c["expected_bads"]) ** 2 / c["den"]
    T["hl_oot1"] = md_table(c[["bin", "n", "mean_pd", "obs_rate", "expected_bads", "bads", "contrib"]],
                            dict(bin=str, n=ni, mean_pd=pc, obs_rate=pc, expected_bads=lambda x: fx(x, 1), bads=ni, contrib=lambda x: fx(x, 2)),
                            dict(bin="Bin (low to high PD)", n="Loans", mean_pd="Mean PD", obs_rate="Observed", expected_bads="Expected bads", bads="Actual bads",
                                 contrib="(O-E)^2 / (n p (1-p))"))
    V["hl_sum"], V["hl_p_calc"] = fx(c["contrib"].sum(), 1), pv(__import__("scipy.stats").stats.chi2.sf(c["contrib"].sum(), 10))
    V["cal_b10_pred"], V["cal_b10_obs"] = pc(c.loc[9, "mean_pd"], 2), pc(c.loc[9, "obs_rate"], 2)
    V["cal_b10_n"], V["cal_b10_e"], V["cal_b10_o"] = ni(c.loc[9, "n"]), fx(c.loc[9, "expected_bads"], 1), ni(c.loc[9, "bads"])
    V["cal_b10_c"] = fx(c.loc[9, "contrib"], 2)
    V["cal_b10_over"] = pc(c.loc[9, "mean_pd"] / c.loc[9, "obs_rate"] - 1, 0)
    V["hl_share10"] = pc(c.loc[9, "contrib"] / c["contrib"].sum(), 0)
    # binomial worked
    g = rcsv("grade_binomial_oot2.csv")
    e = g[(g["model"] == CHAMP) & (g["group"] == "E")].iloc[0]
    V["bn_n"], V["bn_d"], V["bn_p"] = ni(e["n"]), ni(e["bads"]), pc(e["mean_pd"], 3)
    V["bn_exp"], V["bn_sd"] = fx(e["n"] * e["mean_pd"], 1), fx(np.sqrt(e["n"] * e["mean_pd"] * (1 - e["mean_pd"])), 2)
    V["bn_z"], V["bn_obs"] = fx(e["z"], 2), pc(e["obs_rate"], 2)
    gg = g[g["model"] == CHAMP][["group", "n", "bads", "obs_rate", "mean_pd", "z", "p_jeffreys"]]
    T["grade_bin"] = md_table(gg, dict(n=ni, bads=ni, obs_rate=pc, mean_pd=pc, z=lambda x: fx(x, 2), p_jeffreys=pv),
                              dict(group="Grade", n="Loans", bads="Bad", obs_rate="Observed", mean_pd="Mean PD", z="z", p_jeffreys="Jeffreys p"))
    g1 = rcsv("grade_binomial_oot1.csv")
    gg1 = g1[g1["model"] == CHAMP][["group", "n", "bads", "obs_rate", "mean_pd", "z", "p_jeffreys"]]
    T["grade_bin1"] = md_table(gg1, dict(n=ni, bads=ni, obs_rate=pc, mean_pd=pc, z=lambda x: fx(x, 2), p_jeffreys=pv),
                               dict(group="Grade", n="Loans", bads="Bad", obs_rate="Observed", mean_pd="Mean PD", z="z", p_jeffreys="Jeffreys p"))
    # benchmark table
    b = rcsv("benchmark.csv")
    st = b[b["kind"] == "standalone"].pivot_table(index="model_a", columns="split", values="auc_a")
    T["bench_stand"] = md_table(st.reset_index()[["model_a", "dev_holdout", "oot1", "oot2"]], dict(dev_holdout=lambda x: fx(x, 4), oot1=lambda x: fx(x, 4), oot2=lambda x: fx(x, 4)),
                                dict(model_a="Score used alone", dev_holdout="AUC dev holdout", oot1="AUC OOT 2013", oot2="AUC OOT 2014"))
    cv = b[b["kind"].isin(["vs_sub_grade", "incremental_over_sub_grade", "challenger_vs_champion"])][["kind", "split", "model_a", "model_b", "auc_a", "auc_b", "d_auc", "delong_p", "ci_lo", "ci_hi", "light"]]
    cv = cv[cv["split"].isin(["oot1", "oot2"])]
    T["bench_cmp"] = md_table(cv, dict(auc_a=lambda x: fx(x, 4), auc_b=lambda x: fx(x, 4), d_auc=lambda x: f"{x:+.4f}", delong_p=pv,
                                       ci_lo=lambda x: fx(x, 4), ci_hi=lambda x: fx(x, 4)),
                              dict(kind="Comparison", split="Sample", model_a="Model A", model_b="Model B", auc_a="AUC A", auc_b="AUC B", d_auc="AUC A minus B",
                                   delong_p="DeLong p", ci_lo="Boot CI low", ci_hi="Boot CI high", light="Light"))
    # gini by vintage
    bv = rcsv("by_vintage.csv")
    gv = bv[bv["model"].isin(["sc_full", "sc_indep", "lgbm"])].pivot_table(index="vintage", columns="model", values="gini").reset_index()
    nv = bv[bv["model"] == "sc_full"].set_index("vintage")["n"]
    gv["n"] = gv["vintage"].map(nv)
    T["gini_vintage"] = md_table(gv[["vintage", "n", "sc_full", "sc_indep", "lgbm"]], dict(vintage=lambda x: str(int(x)), n=ni, sc_full=lambda x: fx(x, 3), sc_indep=lambda x: fx(x, 3), lgbm=lambda x: fx(x, 3)),
                                 dict(vintage="Issue year", n="Loans", sc_full="Gini SC_FULL", sc_indep="Gini SC_INDEP", lgbm="Gini LightGBM"))
    # psi csi
    pc_ = rcsv("psi_csi.csv")
    cs = pc_[pc_["kind"] == "csi"].pivot_table(index="name", columns="split", values="psi").sort_values("oot2", ascending=False).head(8).reset_index()
    T["csi_top"] = md_table(cs, dict(oot1=lambda x: fx(x, 4), oot2=lambda x: fx(x, 4)), dict(name="Feature", oot1="CSI OOT 2013", oot2="CSI OOT 2014"))
    cc = pc_[pc_["kind"] == "csi"].pivot_table(index="name", columns="split", values="psi")
    V["csi_ils1"], V["csi_ils2"] = fx(cc.loc["initial_list_status", "oot1"], 3), fx(cc.loc["initial_list_status", "oot2"], 3)
    V["csi_rec1"], V["csi_rec2"] = fx(cc.loc["mths_since_last_record", "oot1"], 3), fx(cc.loc["mths_since_last_record", "oot2"], 3)
    sc_ = pc_[pc_["kind"] == "score"].pivot_table(index="model", columns="split", values="psi").reset_index()
    T["psi_score"] = md_table(sc_[["model", "dev_holdout", "oot1", "oot2"]], dict(dev_holdout=lambda x: fx(x, 4), oot1=lambda x: fx(x, 4), oot2=lambda x: fx(x, 4)),
                              dict(model="Model", dev_holdout="Dev holdout", oot1="OOT 2013", oot2="OOT 2014"))
    q = rcsv("psi_by_quarter.csv")
    rq = q[q["model"] == "repro_original"]
    V["rq_first"], V["rq_last"] = fx(rq.iloc[0]["mean_pd"], 3), fx(rq.iloc[-1]["mean_pd"], 3)
    V["rq_psi_last"], V["q_max_champ"] = fx(rq.iloc[-1]["psi"], 3), fx(q[q["model"] == CHAMP]["psi"].max(), 4)
    # repro
    rp = rjson("repro_original.json")
    w, wo = rp["with_mths_since_issue_d"], rp["without_mths_since_issue_d"]
    T["repro"] = md_table(pd.DataFrame([dict(a="With mths_since_issue_d", n=w["n_features"], auc=w["auc"], g=w["gini"], k=w["ks"]),
                                        dict(a="Without it", n=wo["n_features"], auc=wo["auc"], g=wo["gini"], k=wo["ks"])]),
                          dict(n=str, auc=lambda x: fx(x, 4), g=lambda x: fx(x, 4), k=lambda x: fx(x, 4)),
                          dict(a="Specification", n="Dummy columns", auc="Test AUC", g="Test Gini", k="Test KS"))
    V["rp_gdrop"], V["rp_ntrain"], V["rp_ntest"] = fx(rp["gini_drop"], 4), ni(w["n_train"]), ni(w["n_test"])
    V["rp_bad"] = pc(rp["bad_rate_train"], 2)
    # lgd ead
    ls = rjson("lgd_ead_summary.json")
    rc = ls["reconciliation"]
    V["lg_n"], V["lg_match"], V["lg_diff"] = ni(rc["n_recomputed"]), ni(rc["n_matched"]), f"{max(rc['max_abs_diff_recovery_rate'], rc['max_abs_diff_ccf']):.1e}"
    V["lg_rr"], V["lg_ccf"] = fx(rc["mean_recovery_rate"], 4), fx(rc["mean_ccf"], 4)
    for k, v in ls["n_defaults"].items():
        V[f"nd_{k}"] = ni(v)
    for k, v in ls["n_fit"].items():
        V[f"nfit_{k}"] = ni(v)
    d = rcsv("lgd_ead_diagnostics.csv")
    def dg(comp, met, split):
        s_ = d[(d["component"] == comp) & (d["metric"] == met) & (d["split"] == split)]
        return np.nan if s_.empty else s_["value"].iloc[0]
    rows = []
    for lab, comp, met, code in (("Stage 1 AUC (any recovery)", "stage1_any_recovery", "auc", "f3"), ("LGD R-squared, expected value", "lgd_expected", "r2", "f3"),
                                 ("LGD R-squared, hard classes", "lgd_hard_class", "r2", "f3"), ("LGD actual vs predicted correlation", "lgd_expected", "corr", "f3"),
                                 ("Mean actual LGD", "lgd_expected", "mean_actual", "f4"), ("Mean predicted LGD", "lgd_expected", "mean_pred", "f4"),
                                 ("CCF R-squared", "ccf", "r2", "f3"), ("CCF correlation", "ccf", "corr", "f3"), ("Mean actual CCF", "ccf", "mean_actual", "f4"),
                                 ("Mean predicted CCF", "ccf", "mean_pred", "f4"), ("EAD in USD, R-squared", "ead_usd", "r2", "f3")):
        rows.append([lab] + [fmt(dg(comp, met, s_), code) for s_ in ("dev_le_2012", "oot_2013", "report_2014")])
    T["lgdead"] = md_table(pd.DataFrame(rows, columns=["a", "b", "c", "d"]), headers=dict(a="Metric", b="Dev (defaults to 2012)", c="OOT (2013 loans)", d="Report only (2014 loans)"))
    for key, comp, met in (("s1auc", "stage1_any_recovery", "auc"), ("lgr2", "lgd_expected", "r2"), ("lgr2h", "lgd_hard_class", "r2"), ("lgcorr", "lgd_expected", "corr"),
                           ("ccfr2", "ccf", "r2"), ("ccfcorr", "ccf", "corr"), ("eadr2", "ead_usd", "r2"), ("lgma", "lgd_expected", "mean_actual"), ("lgmp", "lgd_expected", "mean_pred"),
                           ("ccfma", "ccf", "mean_actual"), ("ccfmp", "ccf", "mean_pred")):
        for s_, tag in (("dev_le_2012", "d"), ("oot_2013", "o"), ("report_2014", "r")):
            V[f"{key}_{tag}"] = fx(dg(comp, met, s_), 3)
    V["bp_stage2"], V["bp_ead"] = pv(ls["breusch_pagan"]["stage2"]["lm_p"]), pv(ls["breusch_pagan"]["ead"]["lm_p"])
    rv = rcsv("recovery_by_default_vintage.csv")
    T["recov"] = md_table(rv, dict(default_year=str, n=ni, any_recovery_share=pc, mean_recovery_rate=lambda x: fx(x, 4), mean_ccf=lambda x: fx(x, 3)),
                          dict(default_year="Year of last payment", n="Charged-off loans", any_recovery_share="Any recovery", mean_recovery_rate="Mean recovery rate", mean_ccf="Mean CCF"))
    for y in (2012, 2013, 2014, 2015):
        V[f"rec{y}"] = pc(rv[rv["default_year"] == y]["any_recovery_share"].iloc[0], 0)
    s1 = rcsv("lgd_stage1_coefs.csv")
    T["stage1_top"] = md_table(s1.reindex(s1["z"].abs().sort_values(ascending=False).index).head(8), dict(coef=lambda x: fx(x, 4), se=lambda x: fx(x, 4), z=lambda x: fx(x, 2), p_value=pv),
                               dict(term="Term", coef="Coefficient", se="Std. error", z="z", p_value="p-value"))
    V["s1_n_sig"] = str(int((s1.iloc[1:]["p_value"] < 0.05).sum()))
    V["s1_n"] = str(len(s1) - 1)
    ea = rcsv("ead_coefs.csv")
    T["ead_top"] = md_table(ea.reindex(ea["t"].abs().sort_values(ascending=False).index).head(8), dict(coef=lambda x: fx(x, 4), se=lambda x: fx(x, 4), t=lambda x: fx(x, 2), p_value=pv),
                            dict(term="Term", coef="Coefficient", se="Std. error", t="t", p_value="p-value"))
    # EL
    e = rcsv("el_backtest_12m.csv")
    ea_ = e[e["grade"] == "ALL"][["vintage", "n", "mean_pd", "bad_rate", "el", "realized_loss", "el_to_realized", "sample"]]
    T["el12"] = md_table(ea_, dict(vintage=str, n=ni, mean_pd=pc, bad_rate=pc, el=lambda x: fx(x / 1e6, 1), realized_loss=lambda x: fx(x / 1e6, 1), el_to_realized=lambda x: fx(x, 3)),
                         dict(vintage="Vintage", n="Loans", mean_pd="Mean PD", bad_rate="Bad rate", el="EL (USD m)", realized_loss="Realized (USD m)", el_to_realized="EL / realized", sample="Sample"))
    for y in (2013, 2014):
        r = e[(e["vintage"] == y) & (e["grade"] == "ALL")].iloc[0]
        V[f"el{y}"], V[f"re{y}"], V[f"ratio{y}"] = usd_m(r["el"]), usd_m(r["realized_loss"]), fx(r["el_to_realized"], 3)
        V[f"elp{y}"], V[f"rep{y}"] = pc(r["el_pct_funded"], 2), pc(r["realized_pct_funded"], 2)
    g13 = e[(e["vintage"] == 2013) & (e["grade"] != "ALL")][["grade", "n", "mean_pd", "bad_rate", "el", "realized_loss", "el_to_realized"]]
    T["el13"] = md_table(g13, dict(n=ni, mean_pd=pc, bad_rate=pc, el=lambda x: fx(x / 1e6, 2), realized_loss=lambda x: fx(x / 1e6, 2), el_to_realized=lambda x: fx(x, 3)),
                         dict(grade="Grade", n="Loans", mean_pd="Mean PD", bad_rate="Bad rate", el="EL (USD m)", realized_loss="Realized (USD m)", el_to_realized="EL / realized"))
    g14 = e[(e["vintage"] == 2014) & (e["grade"].isin(["E", "F", "G"]))]
    V["el14_g_ratio"] = ", ".join(f"{r['grade']} {r['el_to_realized']:.2f}" for _, r in g14.iterrows())
    lf = rcsv("el_backtest_lifetime.csv")
    lt = lf[lf["sample"].str.startswith(("train", "test"))]
    lt = lt[lt["grade"] == "ALL"]
    t2 = lf[(lf["sample"] == "test_2012")][["grade", "n", "mean_pd", "bad_rate", "el_pct_funded", "realized_pct_funded", "el_to_realized"]]
    T["elife"] = md_table(t2, dict(n=ni, mean_pd=pc, bad_rate=pc, el_pct_funded=pc, realized_pct_funded=pc, el_to_realized=lambda x: fx(x, 3)),
                          dict(grade="Grade", n="Loans", mean_pd="Mean lifetime PD", bad_rate="Observed", el_pct_funded="EL % funded", realized_pct_funded="Realized % funded", el_to_realized="EL / realized"))
    tt = lf[(lf["sample"] == "test_2012") & (lf["grade"] == "ALL")].iloc[0]
    V["lf_el"], V["lf_re"], V["lf_ratio"] = pc(tt["el_pct_funded"], 2), pc(tt["realized_pct_funded"], 2), fx(tt["el_to_realized"], 3)
    tr = lf[(lf["sample"].str.startswith("train")) & (lf["grade"] == "ALL")].iloc[0]
    V["lf_tr_ratio"] = fx(tr["el_to_realized"], 3)
    og = lf[lf["sample"].str.startswith("original")].iloc[0]
    V["orig_el_pct"], V["orig_el_usd"] = pc(og["el_pct_funded"], 2), usd_m(og["el"], 1)
    # sensitivity
    se = rcsv("sensitivity.csv")
    s1_ = se[(se["model"] == CHAMP) & (se["group"] != "baseline")][["scenario", "mean_pd", "delta_pd_pct", "delta_el_pct", "spearman_rank", "share_band_moved", "gini_oot1"]]
    T["sens"] = md_table(s1_, dict(mean_pd=pc, delta_pd_pct=lambda x: f"{x:+.1f}%", delta_el_pct=lambda x: f"{x:+.1f}%", spearman_rank=lambda x: fx(x, 4),
                                   share_band_moved=lambda x: pc(x, 1), gini_oot1=lambda x: fx(x, 4)),
                         dict(scenario="Scenario", mean_pd="Mean PD", delta_pd_pct="Change in PD", delta_el_pct="Change in EL", spearman_rank="Rank correlation", share_band_moved="Loans changing PD band", gini_oot1="Gini (refit variants)"))
    sx = se.set_index(["scenario", "model"])
    for k in ("shock_inq_add", "shock_combined", "shock_int_rate_pts", "shock_income_pct"):
        V[f"{k}_pd"] = f"{sx.loc[(k, CHAMP), 'delta_pd_pct']:+.1f}%"
        V[f"{k}_el"] = f"{sx.loc[(k, CHAMP), 'delta_el_pct']:+.1f}%"
    V["shock_int_rate_pts_pd_xgb"] = f"{sx.loc[('shock_int_rate_pts', 'xgb'), 'delta_pd_pct']:+.1f}%"
    V["t24_pd"], V["t24_el"] = f"{sx.loc[('target_24m_window', CHAMP), 'delta_pd_pct']:+.1f}%", f"{sx.loc[('target_24m_window', CHAMP), 'delta_el_pct']:+.1f}%"
    V["t24_mean"], V["t24_gini"] = pc(sx.loc[("target_24m_window", CHAMP), "mean_pd"], 2), fx(sx.loc[("target_24m_window", CHAMP), "gini_oot1"], 4)
    V["grace_pd"] = f"{sx.loc[('target_in_grace_bad', CHAMP), 'delta_pd_pct']:+.1f}%"
    V["dn_pd"] = f"{sx.loc[('population_dnmcp_included', CHAMP), 'delta_pd_pct']:+.2f}%"
    bn = se[se["group"] == "binning"]
    V["bin_g_lo"], V["bin_g_hi"] = fx(bn["gini_oot1"].min(), 4), fx(bn["gini_oot1"].max(), 4)
    # shap
    sv = rcsv("shap_vs_scorecard.csv")
    T["shap_cmp"] = md_table(sv[["variable", "mean_abs_shap", "shap_rank", "iv", "iv_rank", "points_range", "in_scorecard"]].head(12),
                             dict(mean_abs_shap=lambda x: fx(x, 4), shap_rank=lambda x: fx(x, 0), iv=lambda x: fx(x, 4), iv_rank=lambda x: fx(x, 0),
                                  points_range=lambda x: fx(x, 1), in_scorecard=lambda x: "yes" if x else "no"),
                             dict(variable="Variable", mean_abs_shap="Mean |SHAP|", shap_rank="SHAP rank", iv="IV", iv_rank="IV rank", points_range="Points range", in_scorecard="In scorecard"))
    sh = rcsv("shap_importance.csv")
    V["shap1"], V["shap1v"], V["shap2v"] = fx(sh.iloc[0]["mean_abs_shap"], 3), sh.iloc[0]["feature"], sh.iloc[1]["feature"]
    V["shap_grade"] = fx(sh[sh["feature"] == "grade"]["mean_abs_shap"].iloc[0], 4)
    # findings
    fl = rcsv("findings_log.csv")
    V["f_n"] = str(len(fl))
    cnt = fl["severity"].value_counts()
    V["f_hi"], V["f_med"], V["f_low"] = (str(int(cnt.get(k, 0))) for k in ("High", "Medium", "Low"))
    hi = fl[fl["severity"] == "High"]
    parts = []
    for _, r in hi.iterrows():
        parts.append(f"**{r['finding_id']}: {r['title']}** (area: {r['area']}; source: {r['source']}; evidence file: `{r['evidence_file']}`).\n\n"
                     f"Description: {r['description']}\n\nEvidence: {r['evidence']}\n\nRecommendation: {r['recommendation']}")
    T["findings_high"] = "\n\n".join(parts)
    rest = fl[fl["severity"] != "High"][["finding_id", "area", "title", "severity", "traffic_light", "source"]]
    T["findings_rest"] = md_table(rest, dict(), dict(finding_id="ID", area="Area", title="Title", severity="Severity", traffic_light="Light", source="Source"))
    bys = fl.groupby(["severity", "source"]).size().reset_index(name="n")
    T["findings_counts"] = md_table(bys, dict(n=str), dict(severity="Severity", source="Source", n="Findings"))
    _tl_champion(V, T)
    ch = lambda m, s, t: vr(t, m, s)
    V["g_ho"], V["g_o1"], V["g_o2"] = (fx(ch(CHAMP, s, "gini"), 3) for s in (C.DEV_HOLDOUT, C.OOT1, C.OOT2))
    V["k_o1"], V["k_o2"] = fx(ch(CHAMP, C.OOT1, "ks"), 3), fx(ch(CHAMP, C.OOT2, "ks"), 3)
    V["gci_lo1"], V["gci_hi1"] = fx(ch(CHAMP, C.OOT1, "gini_ci_lo"), 3), fx(ch(CHAMP, C.OOT1, "gini_ci_hi"), 3)
    V["gdec1"], V["gdec2"] = pc(ch(CHAMP, C.OOT1, "gini_decay"), 1), pc(ch(CHAMP, C.OOT2, "gini_decay"), 1)
    V["ind_g1"], V["ind_g2"] = fx(ch(INDEP, C.OOT1, "gini"), 3), fx(ch(INDEP, C.OOT2, "gini"), 3)
    V["psi_champ1"], V["psi_champ2"] = fx(ch(CHAMP, C.OOT1, "psi_score"), 4), fx(ch(CHAMP, C.OOT2, "psi_score"), 4)
    V["rep_g_o1"], V["rep_g_o2"] = fx(ch("repro_original", C.OOT1, "gini"), 3), fx(ch("repro_original", C.OOT2, "gini"), 3)
    V["rep_psi2"] = fx(ch("repro_original", C.OOT2, "psi_score"), 3)
    V["rep_mean2"] = pc(ch("repro_original", C.OOT2, "mean_pd"), 2) if not nz(ch("repro_original", C.OOT2, "mean_pd")) else "not computed"


def _toys(V, T):
    from credit_validation import validation as val
    from scipy import stats
    y = np.array([1, 1, 0, 1, 0, 0])
    p = np.array([0.95, 0.85, 0.70, 0.55, 0.40, 0.20])
    V["toy_auc"], V["toy_ks"], V["toy_gini"] = fx(val.auc(y, p), 3), fx(val.ks(y, p), 3), fx(val.gini(y, p), 3)
    pos, neg = p[y == 1], p[y == 0]
    V["toy_conc"], V["toy_pairs"] = str(int(sum(a > b for a in pos for b in neg))), str(len(pos) * len(neg))
    order = np.argsort(-p)
    cp, cn = np.cumsum(y[order]) / y.sum(), np.cumsum(1 - y[order]) / (1 - y).sum()
    toy = pd.DataFrame(dict(rank=np.arange(1, 7), score=p[order], bad=y[order], cb=cp, cg=cn, gap=np.abs(cp - cn)))
    T["toy_ks"] = md_table(toy, dict(rank=str, score=lambda x: fx(x, 2), bad=str, cb=lambda x: fx(x, 3), cg=lambda x: fx(x, 3), gap=lambda x: fx(x, 3)),
                           dict(rank="Rank (riskiest first)", score="Predicted PD", bad="Bad", cb="Cum. share of bads", cg="Cum. share of goods", gap="Gap"))
    # hosmer lemeshow toy
    n, pp, o = np.array([1000, 1000, 1000]), np.array([0.02, 0.05, 0.10]), np.array([15, 60, 125])
    ex = n * pp
    ct = (o - ex) ** 2 / (n * pp * (1 - pp))
    V["hlt_stat"], V["hlt_p"] = fx(ct.sum(), 3), fx(stats.chi2.sf(ct.sum(), 3), 4)
    T["toy_hl"] = md_table(pd.DataFrame(dict(g=[1, 2, 3], n=n, p=pp, e=ex, o=o, c=ct)), dict(g=str, n=ni, p=lambda x: pc(x, 0), e=lambda x: fx(x, 0), o=str, c=lambda x: fx(x, 3)),
                           dict(g="Group", n="Loans", p="Mean PD", e="Expected bads", o="Actual bads", c="Contribution"))
    # toy logit for Wald
    from credit_validation.stats_models import LogisticRegressionWithPValues
    x = np.arange(8, dtype=float)
    yy = np.array([0, 0, 1, 0, 1, 0, 1, 1])
    m = LogisticRegressionWithPValues().fit(pd.DataFrame({"x": x}), yy)
    D = np.column_stack([np.ones(8), x])
    pr = 1 / (1 + np.exp(-(D @ m.beta_)))
    Fm = (D * (pr * (1 - pr))[:, None]).T @ D
    cov = np.linalg.inv(Fm)
    se_wrong = 1 / np.sqrt(Fm[1, 1])
    V["wt_b0"], V["wt_b1"] = fx(m.beta_[0], 4), fx(m.beta_[1], 4)
    V["wt_F00"], V["wt_F01"], V["wt_F11"] = fx(Fm[0, 0], 4), fx(Fm[0, 1], 4), fx(Fm[1, 1], 4)
    V["wt_det"] = fx(Fm[0, 0] * Fm[1, 1] - Fm[0, 1] ** 2, 4)
    V["wt_se1"], V["wt_se0"] = fx(np.sqrt(cov[1, 1]), 4), fx(np.sqrt(cov[0, 0]), 4)
    V["wt_z"], V["wt_p"] = fx(m.z_[1], 3), fx(m.p_values_[1], 4)
    V["wt_se_wrong"] = fx(se_wrong, 4)
    V["wt_p_wrong"] = fx(2 * stats.norm.sf(abs(m.beta_[1] / se_wrong)), 4)
    try:
        import statsmodels.api as sm
        f = sm.Logit(yy, D).fit(disp=0)
        V["wt_p_sm"] = fx(f.pvalues[1], 4)
    except Exception:  # noqa: BLE001
        V["wt_p_sm"] = "n/a"
    # DeLong toy
    c1 = np.array([0.9, 0.8, 0.6, 0.4, 0.3, 0.1])
    c2 = np.array([0.7, 0.9, 0.5, 0.6, 0.2, 0.3])
    d, z, pval = val.delong_test(y, c1, c2)
    V["dl_a1"], V["dl_a2"], V["dl_d"] = fx(val.auc(y, c1), 4), fx(val.auc(y, c2), 4), f"{d:+.4f}"


EXCEL_DESC = [("Sheets", "README, Inputs, Scorecard, Score_Calc, Loan_Sample, Deciles, Calibration, PSI, EL, Checks, Conclusions, Python_Ref")]


def _excel(V, T):
    d = None
    if EXCEL_RECON_JSON.exists():
        try:
            d = json.loads(EXCEL_RECON_JSON.read_text())
        except Exception:  # noqa: BLE001
            d = None
    V["excel_present"] = "yes" if d else "no"
    if not d:
        V["excel_block"] = ("The reconciliation report `python/outputs/excel_reconciliation.json` is not present in this checkout, so the workbook is "
                            "still being finalized and no reconciliation numbers are quoted here. Rerun `py -3 docs/build_docs.py` after the workbook "
                            "is built and reconciled to include the measured Excel versus Python differences.")
        V["excel_line"] = "The workbook and its reconciliation report are being finalized."
        return
    scal = {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool)) and k != "data_source"}
    parts = []
    passed = d.get("all_passed", d.get("passed"))
    if passed is not None:
        parts.append(f"Overall result: **{'all comparisons passed' if passed else 'at least one comparison FAILED'}**.")
    if "n_cases" in d:
        parts.append(f"The reconciliation ran {d['n_cases']} scenario cases.")
    if d.get("n_formulas"):
        parts.append(f"The workbook holds {int(d['n_formulas']):,} formulas.")
    w = d.get("worst_diffs")
    if isinstance(w, dict) and w:
        wk = max(w, key=lambda k: abs(w[k]))
        parts.append(f"The largest absolute difference across all quantities is {abs(w[wk]):.1e} (`{wk}`).")
        top = sorted(w.items(), key=lambda kv: -abs(kv[1]))[:10]
        parts.append(f"The table lists the 10 largest of {len(w)} compared quantities; all others are smaller. Differences of order 1e-13 to 1e-16 are floating-point rounding.")
        T["excel_diffs"] = md_table(pd.DataFrame([dict(q=k, d=v) for k, v in top]), dict(d=lambda x: f"{x:.1e}"), dict(q="Quantity", d="Worst absolute difference"))
        V["excel_line"] = " ".join(parts[:2])
    else:
        T["excel_diffs"] = md_table(pd.DataFrame([dict(k=k, v=str(v)) for k, v in scal.items()]), headers=dict(k="Report field", v="Value"))
        V["excel_line"] = " ".join(parts) or "A reconciliation report is present."
    V["excel_block"] = " ".join(parts) + "\n\n" + T["excel_diffs"] + "\n\nTable: Excel reconciliation, from `outputs/excel_reconciliation.json`."


def build_values():
    V, T = {}, {}
    X = dict(ds=rjson("data_summary.json"), meta=rjson("model_meta.json"), pf=panel_facts())
    _core(V, T, X)
    _results(V, T, X)
    _toys(V, T)
    _readme_table(T)
    _excel(V, T)
    V["chart_rel"] = CHART_REL
    V["n_app_features"] = str(len(C.APPLICATION_FEATURES))
    from credit_validation import data as data_mod
    V["n_use_cols"] = str(len(data_mod.USE_COLUMNS))
    V["repro_dummies"] = str(rjson("repro_original.json")["with_mths_since_issue_d"]["n_features"])
    return V, T, X


# ---------------------------------------------------------------- rendering and conversion
PLACEHOLDER = re.compile(r"@@([A-Za-z0-9_:\.+]+)@@")


def render(text, V, T):
    def sub(m):
        key = m.group(1)
        if key.startswith("tbl:"):
            return T[key[4:]]
        if key.startswith("r:"):
            _, t, mo, s, code = key.split(":")
            return fmt(vr(t, mo, s), code)
        if key.startswith("l:"):
            _, t, mo, s = key.split(":")
            return str(vr(t, mo, s, "light"))
        if key.startswith("bm:"):
            _, kind, s, ma, col, code = key.split(":")
            return fmt(bm(kind, s, ma, col), code)
        return str(V[key])
    for _ in range(6):
        new = PLACEHOLDER.sub(sub, text)
        if new == text:
            break
        text = new
    if PLACEHOLDER.search(text):
        raise ValueError("unresolved placeholders: " + ", ".join(sorted(set(PLACEHOLDER.findall(text)))))
    return text


def md_stats(text):
    body = re.sub(r"```.*?```", "", text, flags=re.S)
    body = re.sub(r"^---\n.*?\n---\n", "", body, count=1, flags=re.S)
    headings = len(re.findall(r"^#{1,6} ", body, flags=re.M))
    figures = len(re.findall(r"^!\[", body, flags=re.M))
    display = len(re.findall(r"\$\$.+?\$\$", body, flags=re.S))
    rest = re.sub(r"\$\$.+?\$\$", "", body, flags=re.S)
    inline = len(re.findall(r"(?<![\\$])\$(?!\s)[^$\n]+?(?<!\s)\$(?!\d)", rest))
    tables, in_tbl = 0, False
    for line in body.splitlines():
        is_row = line.startswith("|")
        if is_row and not in_tbl:
            tables += 1
        in_tbl = is_row
    return dict(headings=headings, tables=tables, equations=display + inline, figures=figures)


def docx_stats(path):
    import zipfile
    from docx import Document
    doc = Document(str(path))
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")
    return dict(headings=sum(1 for p in doc.paragraphs if p.style.name.startswith("Heading")), tables=len(doc.tables),
                equations=xml.count("<m:oMath>") + xml.count("<m:oMath "), figures=xml.count("<w:drawing>"))


def style_tables(path):
    from docx import Document
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls
    from docx.shared import Pt
    doc = Document(str(path))
    borders = ('<w:tblBorders %s>' % nsdecls("w") + "".join(
        f'<w:{e} w:val="single" w:sz="4" w:space="0" w:color="808080"/>' for e in ("top", "left", "bottom", "right", "insideH", "insideV")) + "</w:tblBorders>")
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    for t in doc.tables:
        pr = t._tbl.tblPr
        for old in pr.findall(ns + "tblBorders"):
            pr.remove(old)
        pr.append(parse_xml(borders))
        for row in t.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.font.size = Pt(8.5)
    doc.save(str(path))


def set_author(path):
    from docx import Document
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    doc = Document(str(path))
    for st in doc.styles:
        if st.name.startswith("Heading") or st.name.startswith("TOC Heading"):
            rpr = st.element.rPr
            if rpr is not None:
                for c in rpr.findall(ns + "color"):
                    rpr.remove(c)
    cp = doc.core_properties
    cp.author, cp.last_modified_by, cp.title = AUTHOR, AUTHOR, DOC_TITLE
    doc.save(str(path))


def convert():
    import pypandoc
    pypandoc.convert_file(str(MD_PATH), "docx", format="markdown-smart", outputfile=str(DOCX_PATH),
                          extra_args=[f"--reference-doc={REFERENCE_DOCX}", "--toc", f"--toc-depth={TOC_DEPTH}", "--number-sections",
                                      f"--resource-path={DOCS_DIR}", "--standalone"])
    style_tables(DOCX_PATH)
    set_author(DOCX_PATH)


def verify(md_text):
    import pypandoc
    a, b = md_stats(md_text), docx_stats(DOCX_PATH)
    rt = pypandoc.convert_file(str(DOCX_PATH), "markdown", format="docx", extra_args=["--wrap=none"])
    bad = [("replacement character", chr(0xFFFD) in rt), ("em dash", chr(0x2014) in rt)]
    print("markdown counts:", a)
    print("docx counts    :", b)
    for name, hit in bad:
        print(f"round trip {name}: {'FOUND' if hit else 'none'}")
    if a != b or any(h for _, h in bad):
        raise SystemExit("round-trip verification failed")


def main():
    V, T, X = build_values()
    text = render(S0 + SECTIONS_A + SECTIONS_B + SECTIONS_C + SECTIONS_D, V, T)
    MD_PATH.write_text(text, encoding="utf-8")
    README_PATH.write_text(render(README_TEMPLATE, V, T), encoding="utf-8")
    convert()
    verify(text)
    print("wrote", MD_PATH.name, DOCX_PATH.name, README_PATH.name, "| words:", len(text.split()))


S0 = r"""---
title: "@@doc_title@@"
author: "@@author@@"
date: "@@today@@"
---

"""

SECTIONS_A = r"""
# Executive summary and validation opinion

## What this project is

This project is an independent validation exercise. It takes a probability-of-default (PD), loss-given-default (LGD), exposure-at-default (EAD) and expected-loss (EL) engine that was originally built as a Master's course project (the "original project", MQF OOPs II) and treats it the way a bank's model validation team would treat a model submitted for approval. The work has three parts. First, the as-built model is reviewed and its defects are logged. Second, the PD model is rebuilt on a defensible footing (a fixed 12-month default window, time-ordered samples, an application-time feature set, a points-scaled scorecard) and challenged with gradient-boosted trees, a benchmark and stress tests. Third, LGD, EAD and expected loss are re-estimated and back-tested against realized losses. An Excel workbook with live formulas lets a reviewer trace the scorecard, decile, calibration, stability and expected-loss logic cell by cell.

## Read this first: what this is and is not

The data are the public LendingClub loan files for loans issued from @@issue_min@@ to @@issue_max@@ (@@n_loans@@ loans), with outcomes observed up to about @@status_date@@. LendingClub public loan data is the only data source. This is **not a bank-approved or production model**, no regulator or bank model risk committee has reviewed it, and nothing here is evidence about any real lender's book today. It is a portfolio project that demonstrates validation method on a real, messy dataset. The author of the original model is also the author of this validation, so the independence is procedural (the validation re-derives every result from raw data and was written to challenge the original), not organizational. A real validation would be performed by a separate team.

## Validation opinion

The overall opinion is that the rebuilt scorecard (SC_FULL, the champion) is **fit as a rank-ordering tool but not fit as a stand-alone production PD model without remediation**. The reasons, each backed by a number in this document, are:

- **Discrimination is modest and stable.** Gini is @@g_ho@@ on the development holdout, @@g_o1@@ on 2013 loans (95% bootstrap interval @@gci_lo1@@ to @@gci_hi1@@) and @@g_o2@@ on 2014 loans, all Amber against the Green threshold of 0.40. The decay from holdout to out-of-time is @@gdec1@@ and @@gdec2@@, which is Green.
- **The scorecard does not beat LendingClub's own sub-grade.** Using sub_grade as a one-variable score gives an out-of-time 2013 AUC of @@bm:standalone:oot1:sub_grade_rank:auc_a:f4@@. SC_FULL is ahead by only @@bm:vs_sub_grade:oot1:sc_full:d_auc:s4@@ AUC (not significant) in 2013 and behind by @@bm:vs_sub_grade:oot2:sc_full:d_auc:s4@@ in 2014 (significant). The independent scorecard SC_INDEP, which excludes LendingClub's own risk outputs (grade, sub_grade, int_rate, installment), has Gini @@ind_g1@@ and @@ind_g2@@ out of time: most of the ranking power in SC_FULL is borrowed from the lender's own underwriting score. That is the grade and interest-rate circularity.
- **Calibration is acceptable on average, weak in the tails and by grade.** The calibration slope is Green (@@r:cal_slope:sc_full:oot1:f3@@ and @@r:cal_slope:sc_full:oot2:f3@@), and mean PD is within about @@r:central_tendency:sc_full:oot1:p0@@ of the observed 2013 rate. But the Hosmer-Lemeshow test is Red on both out-of-time samples and the grade-level binomial test is Red for 2014 (grades E, F and G are under-predicted). Section 11 explains why the first is partly a large-sample power effect and why the second is not.
- **Population stability is good for the score and poor for some inputs.** Score PSI is @@psi_champ1@@ and @@psi_champ2@@ (Green) but the characteristic stability index of initial_list_status is @@csi_ils1@@ and @@csi_ils2@@ (Red), a structural break in how LendingClub populated that field.
- **LGD and EAD are weak.** The first LGD stage (any recovery) has AUC @@s1auc_o@@ out of time, LGD R-squared is @@lgr2_o@@ and the CCF R-squared is @@ccfr2_o@@. LGD is almost a constant near @@lgma_o@@, so expected loss is driven by PD and exposure. The 12-month expected loss for the 2013 vintage is @@el2013@@ against realized @@re2013@@ (ratio @@ratio2013@@).
- **The original model has design defects**, now logged as findings (section 18): a loan-age feature that encodes censoring, random-only splits, a lifetime default flag that is right-censored, hard 0/1 labels for LGD stage 1, invalid Wald p-values and others. The rebuilt pipeline remediates each of them. @@f_n@@ findings are logged: @@f_hi@@ High, @@f_med@@ Medium and @@f_low@@ Low.

## Validation scorecard for the champion

The table summarizes the champion's traffic lights. Thresholds are the project's own conventions, defined in section 10, and are not regulatory limits.

@@tbl:tl_champion@@

Table: Traffic-light summary for SC_FULL. Each cell shows the value and its rating. Source: `validation_results.csv`, `benchmark.csv` and `el_backtest_12m.csv`.

Across the full battery of tests (all models and samples) the lights are: @@light_counts@@; the remaining rows are informational.

# Business context and SR 11-7 scope

## Why a lender needs these models

A lender who funds an unsecured consumer loan faces credit loss. Three quantities describe it. The **probability of default** (PD) is the chance the borrower fails to pay within a stated horizon. The **loss given default** (LGD) is the fraction of the exposure that is not recovered once default happens. The **exposure at default** (EAD) is how much is outstanding at that moment. Expected loss is their product for each loan:

$$EL = PD \times LGD \times EAD.$$

PD decides who is approved and at what price; PD, LGD and EAD together feed provisioning and capital. Because decisions and reserves depend on the numbers, supervisors expect a lender to prove that the models are sound. In the United States the supervisory guidance on model risk management is SR 11-7 (Federal Reserve and OCC, 2011).

## How SR 11-7 shapes this document

SR 11-7 organizes model risk work around three activities: development and implementation, validation, and governance. Validation is described as having three core elements, which this project mirrors:

- **Evaluation of conceptual soundness**: are the target definition, the data, the variables and the method appropriate? Sections 3 to 9 cover this.
- **Ongoing monitoring**: are inputs and outputs stable and is the model still performing? Sections 14 and 16 cover this, with population stability indices and sensitivity analysis.
- **Outcomes analysis**: do predictions match realized outcomes? Sections 10 to 13 and 15 cover this, with back-testing of PD and of expected loss.

A central idea in the guidance is *effective challenge*: critical analysis by informed people who are incentivized to find defects, including by building alternative models. In this project the challengers are the independent scorecard, XGBoost, LightGBM and the benchmark built from LendingClub's own sub-grade.

## Scope of this validation

In scope: the PD model, the LGD and EAD models, the expected-loss calculation, the data and target definitions, and the original project's as-built code and notebooks (reviewed from their documented behavior and from a faithful re-fit). Out of scope: any production implementation, vendor systems, macroeconomic overlays and capital models. The sample is a single lender's platform lending between 2007 and 2014, so conclusions do not transfer to other lenders or to later periods.

# Data, provenance, censoring and structural breaks

## Provenance and what was loaded

The input is the LendingClub public loan file `loan_data_2007_2014.csv` (@@n_loans@@ loans, 75 columns of which @@n_use_cols@@ are read) and the companion file `loan_data_defaults.csv` used by the original project for its LGD and EAD work. Neither file is committed to this repository; the path is set by the environment variable `CV_RAW_DIR`. The loader (`data.load_loans`) reads only the needed columns, parses percentages and month-year dates, maps the home ownership levels ANY, NONE and OTHER to a single OTHER, corrects two-digit credit-line years, stores the frame with compact dtypes and caches it as parquet.

Two-digit years deserve a note. A date such as "Jan-62" is parsed by default as 2062. Where the parsed earliest credit line falls after the issue date, `fix_two_digit_years` subtracts 100 years. The correction is tested in the unit tests and recorded as a Low finding.

## Loan status

@@tbl:status_counts@@

Table: Loan status counts in the raw file and the coding used by the original notebook (bad = Charged Off, Default, Late 31-120 days and the Does-not-meet-credit-policy charged-off status).

Under the original coding there are @@bad_orig_n@@ bad loans, a bad rate of @@bad_orig_rate@@. The status "In Grace Period" (@@grace_n@@ loans) is coded **good** by the original code even though the original project notes describe it as bad. That documentation mismatch is finding F-09. Its practical impact depends on the target, and section 4 shows that for the fixed 12-month target used here it changes nothing.

Loans with the "Does not meet the credit policy" prefix (@@dnmcp_n@@ loans) were issued under an earlier policy and are excluded from the development and validation samples (switch `EXCLUDE_DNMCP`). Including them is run as a sensitivity (section 16).

## Right-censoring

Every loan in the file is observed up to a status date of about @@status_date@@. A loan issued in 2007 has been observed for most of its term; a loan issued in late 2014 has been observed for about a year. If "bad" means "ever bad before the status date", older vintages mechanically look worse because they had longer to default. This is right-censoring. It is the single most important feature of the data and is visible in the table below.

@@tbl:target_audit@@

Table: Bad rate by issue year, lifetime flag versus the fixed 12-month flag (`target_audit.csv`). Observed 12m counts loans seen for at least 14 months.

The lifetime bad rate falls from @@life07@@ in 2007 to @@life14@@ in 2014, while the share of 2014 loans still Current is @@cur14@@. The fixed 12-month rate is flat between @@b12_lo@@ and @@b12_hi@@ across 2010 to 2014. The 2007 to 2009 vintages are small (see the loan counts) and cover the financial crisis, so they are kept in development but not used to judge stability.

![Lifetime bad rate (dashed) is right-censored and falls with vintage; the fixed 12-month rate is flat.](@@chart_rel@@/bad_rate_by_vintage.png){width=90%}

## Structural breaks in the inputs

Some fields were not populated the same way throughout. The share of loans with initial_list_status equal to "w" is 0% before 2012, @@w_2012@@ in 2012, @@w_2013@@ in 2013 and @@w_2014@@ in 2014; total_rev_hi_lim, tot_cur_bal and tot_coll_amt are missing before 2012 (finding F-12). A model developed on 2007 to 2012 data therefore sees a different data regime from the one it is later applied to. This is why initial_list_status is dropped from the scorecard (its information value is tiny in development) and why it dominates the characteristic stability report in section 14. The other three fields are not used as features.

# Target definitions and observation windows

## The fixed 12-month default flag

The target for the PD model is a **fixed-window** flag. A loan is flagged bad (1) if its final status is bad *and* the time from issue to the last payment date is under 12 months. Months to default are not in the file, so the last payment date is the best available proxy for the time of default: a loan that charged off after paying for 8 months has a last payment 8 months after issue, whereas a loan that paid for 30 months and then charged off does not count as a 12-month default. Loans that never paid are assigned zero months on book. Formally, with $m_i$ the months from issue to last payment:

$$Y_i = \mathbb{1}\{\text{status}_i \in \text{Bad}\} \cdot \mathbb{1}\{m_i < 12\}.$$

A loan is kept in the sample only if it has been observable for at least $12 + 2$ months, the 2 being a buffer for late reporting (`OBS_BUFFER_MONTHS`). Without the filter, a recent loan that is simply too young to have defaulted would be counted as a good, which is the censoring problem again. The functions are `targets.add_targets` (adds `target_bad12`, `obs12`, and the 24-month analogues) and `targets.target_audit`.

## Why a fixed window rather than lifetime

A lifetime flag answers "did this loan ever go bad" and its meaning depends on how long the loan was observed. A fixed window answers one question with one meaning for every loan: "did the loan go bad in its first year". That is also what a 12-month PD, used for provisioning and pricing, is supposed to measure. The cost is that the default timing proxy is imperfect (finding F-08) and the 12-month window ignores later defaults; the 24-month window is run as a sensitivity.

## How much the window removes

@@tbl:bad_vs_bad12@@

Table: Loans with a bad final status and the subset that went bad within 12 months, among loans observed long enough for the 12-month window.

Roughly a third of loans with a bad final status went bad inside the 12-month window; the share is larger for 2014 because those loans have not had time to go bad later. Using the fixed window shrinks the bad rate to about 4 to 5% from the lifetime rate of @@bad_orig_rate@@ and keeps it comparable across vintages.

## In Grace Period and the 12-month window

Loans in the "In Grace Period" status were issued between @@grace_min@@ and @@grace_max@@ and have last payments near the status date. For the 12-month flag the second condition (last payment within 12 months of issue) is not met by any observable loan in that status: the count of In Grace Period loans that would be flagged under the 12-month rule is @@grace_win@@. Coding the status as bad therefore changes the mean PD by @@grace_pd@@ for the 12-month target, as the sensitivity run confirms. The coding choice matters only for the lifetime flag used to reproduce the original model.

## Samples

@@tbl:splits@@

Table: Samples. Development (train and holdout) uses issues up to December 2012. OOT 1 and OOT 2 are later cohorts never used in fitting or calibration. Seed @@seed@@.

Development is split into a stratified random 80% training sample and a 20% holdout. The holdout provides the in-time metrics and is the data on which the gradient-boosted models are probability-calibrated; the two out-of-time samples are never touched during fitting, tuning or calibration. The "complete 36-month" cohort (36-month loans issued up to 2012, almost all fully resolved) is used with the lifetime flag for the lifetime expected-loss back-test. The repro samples reproduce the original model's random 80/20 design and are used only for the reproduction in section 9.

# Feature set and leakage controls

## Application-time features only

A PD model for decisions must use only information available when the loan is approved. The feature list (`columns.APPLICATION_FEATURES`) has @@n_app_features@@ variables: loan amount, term, interest rate, installment, grade and sub-grade, employment length, home ownership, annual income, verification status, purpose, state, debt-to-income, delinquencies in the past two years, inquiries in the past six months, months since last delinquency and public record, open accounts, public records, revolving balance and utilization, total accounts, initial list status, and four derived variables: credit history length in months, loan-to-income, payment-to-income (installment times twelve over income) and the term in months.

## What is forbidden

All fields generated after origination are excluded by name: payments received, outstanding principal, recoveries, the last payment date, next payment date, last credit pull date, loan status itself and similar. `features.assert_no_leakage` raises an error if any feature, any column in `POST_ORIGINATION`, any vintage proxy (`mths_since_issue_d`) or any target column appears in a model's feature list, and every model-fitting entry point calls it. The vintage proxy is forbidden because loan age to a fixed reference date identifies the issue month, and issue month correlates with how long the loan was observed (section 9).

## Two champions to expose circularity

LendingClub assigns each loan a grade, a sub-grade and an interest rate using its own underwriting model. These are outputs of a risk model, so a PD model that uses them partly re-learns the lender's model. To measure how much, two scorecards are fitted: **SC_FULL** uses all application features including the four LendingClub risk outputs (grade, sub_grade, int_rate, installment), and **SC_INDEP** excludes them. The difference in performance is the value of the lender's own score. This is finding F-07 and is quantified in section 13.
"""

SECTIONS_B = r"""
# Weight of evidence, information value and the binning algorithm

## Intuition

A logistic regression needs numbers, but credit variables are messy: income is skewed, state is a label with 50 levels, and some fields are missing for a reason. Scorecard practice replaces each variable by a small number of **bins** and replaces each bin by a single number that says how much safer or riskier that bin is than average. That number is the **weight of evidence** (WoE). The variable then enters the regression as one column of WoE values, which is monotone in risk by construction, handles missing values in a bin of their own, and makes every variable's contribution to the final score visible as points.

## The math

Let $B$ be the total number of bad loans and $G$ the total number of good loans. For bin $i$ with $b_i$ bads and $g_i$ goods, let $d^B_i = b_i / B$ and $d^G_i = g_i / G$ be the shares of all bads and of all goods that fall in the bin. Then

$$\mathrm{WoE}_i = \ln\frac{d^G_i}{d^B_i}, \qquad \mathrm{IV} = \sum_i \left(d^G_i - d^B_i\right)\mathrm{WoE}_i .$$

A positive WoE means the bin holds a larger share of goods than of bads, so it is safer than average; a negative WoE means riskier. Each term of the IV is non-negative (the difference and the log have the same sign), so IV measures total separation and is zero only if every bin looks like the population. A common rule of thumb reads IV below 0.02 as not useful, 0.02 to 0.1 weak, 0.1 to 0.3 medium, 0.3 to 0.5 strong and above 0.5 suspiciously strong (Siddiqi, 2006). IV is the symmetrized Kullback-Leibler divergence between the good and bad distributions over the bins.

The sign convention matters in this project: WoE is $\ln(\%\text{good}/\%\text{bad})$, so higher WoE is safer and the fitted logit coefficient on WoE is **negative** (the logit models the chance of bad). A positive coefficient therefore signals a "wrong sign" and the variable is removed.

Empty cells make $\ln$ undefined, so shares are smoothed by adding 0.5 to every populated bin: $d^B_i = (b_i + 0.5)/(B + 0.5k)$ and similarly for goods, with $k$ the number of populated bins (`binning.woe_iv`, constant `SMOOTH`).

## Worked example: inquiries in the last six months

The champion's bins for `inq_last_6mths` (fitted on the development training sample) are:

@@tbl:woe_ex@@

Table: Bins, bad rates, WoE and IV contributions for inq_last_6mths in SC_FULL (`binning_report.csv`). The Missing bin is empty and omitted.

Take the first bin (no inquiries). The sample has $B =$ @@woe_B@@ bads and $G =$ @@woe_G@@ goods in $k =$ @@woe_k@@ populated bins; the bin holds @@woe_bad0@@ bads and @@woe_good0@@ goods. With smoothing, $d^B =$ @@woe_db@@ and $d^G =$ @@woe_dg@@, so $\mathrm{WoE} = \ln(d^G/d^B) =$ @@woe_val@@, which matches the pipeline's value @@woe_rep@@. Its IV contribution is $(d^G - d^B) \times \mathrm{WoE} =$ @@woe_iv0@@. Summing the four bins gives a variable IV of @@woe_iv_var@@. Applicants with no recent inquiries are safer than average and each additional inquiry lowers WoE, as expected.

## The binning algorithm

`binning.fit_numeric_bins` and `binning.fit_categorical_bins` implement the fine-to-coarse classing that practitioners do by hand:

1. **Fine classing.** A numeric variable is cut at its 50 quantiles (`BIN_PREBINS`), with duplicate cut points collapsed. Missing values get their own bin that is always last.
2. **Minimum share.** Any bin holding less than 5% of loans (`BIN_MIN_SHARE`, 2% for categories) is merged into the neighbouring bin whose bad rate is closest.
3. **Monotonicity.** The sign of the Spearman correlation between bin order and bad rate sets the expected direction, and adjacent bins that violate it are merged until the bad rate is monotone. This keeps the scorecard interpretable and prevents noisy zig-zags.
4. **Cap.** If more than 8 bins (`BIN_MAX_BINS`) remain, the adjacent pair with the closest bad rates is merged until 8 remain.
5. **WoE and IV** are computed over the final bins.

Categorical variables are first sorted by their (smoothed) bad rate, so merging adjacent categories groups levels of similar risk (purpose and state). Unseen categories at scoring time go to the largest group. A bin includes values greater than its lower edge and up to and including its upper edge (`searchsorted` with `side="left"`), the same rule used in Excel.

@@tbl:woe_inc@@

Table: Annual income bins in SC_FULL. Lower incomes carry negative WoE (riskier) and the top two bins are nearly identical.

Two observations are worth stating plainly. First, the algorithm merges by bad rate alone, so sparse categories can be grouped with unrelated ones. For example, the sub-grade bin "C2|G4" in the points table merges a mid-grade with a very thin high-risk level; a human reviewer would place G4 with the other G grades. This is a limitation of automated categorical grouping and is the reason the points table should always be reviewed. Second, the top two income bins have almost identical WoE, which suggests the cap of eight bins is slightly too fine there; the binning sensitivity in section 16 shows the Gini is insensitive to this.

## Which variables survive

Information value on the development training sample for all 26 candidate features, and the elimination outcome for SC_FULL, are below. The largest IV belongs to @@iv_top_v@@ (@@iv_top@@).

@@tbl:iv_all@@

Table: Information value of each candidate feature and its fate in SC_FULL (development training sample, bins fitted by the pipeline).

# Scorecard scaling and a worked loan

## From logit to points

Customers and credit officers read scores, not log-odds. A scorecard rescales the model's log-odds to points using three conventions: a **base score** at **base odds** (good to bad) and **points to double the odds** (PDO). Here the base score is @@base_score@@ at odds of @@base_odds@@ to 1 and PDO is @@pdo@@: every @@pdo@@ extra points doubles the good:bad odds. The logit model gives the log-odds of *bad*, $z = \ln\frac{p}{1-p}$, so

$$\text{Score} = \text{Offset} - \text{Factor}\cdot z, \qquad \text{Factor} = \frac{\text{PDO}}{\ln 2}, \qquad \text{Offset} = \text{BaseScore} - \text{Factor}\cdot\ln(\text{BaseOdds}).$$

With the project settings, Factor is @@factor@@ and Offset is @@offset@@ (`scorecard.scaling`). Because $z = \alpha + \sum_j \beta_j\,\mathrm{WoE}_{ij}$ for loan $i$, with $\alpha$ the intercept and $k$ the number of variables, the score is a sum of one term per variable, each of which is a points value for the bin the loan falls in:

$$\text{Points}_{ij} = -\left(\beta_j\,\mathrm{WoE}_{ij} + \frac{\alpha}{k}\right)\text{Factor} + \frac{\text{Offset}}{k}.$$

The intercept and the offset are spread equally over the $k$ variables so that every bin has a self-contained points value (`Scorecard.export_points_table`, `points_by_variable`). The probability of bad is recovered from the score by inverting the formula.

@@tbl:score_map@@

Table: Score to odds to PD map implied by the scaling (before any recalibration). A score of 600 corresponds to a PD of @@pd_at_600@@ and the development average PD of about 4.8% corresponds to a score near @@avg_score@@.

## The champion scorecard

SC_FULL retains @@k_vars@@ variables: @@sc_vars@@. Elimination (`scorecard.fit_scorecard`) removed @@n_dropped_iv@@ variables with IV below 0.02, @@n_dropped_corr@@ for WoE correlation above 0.7 with a stronger variable (grade and int_rate lose to sub_grade, pti to loan_to_inc) and @@n_dropped_p@@ for a Wald p-value above 0.05 (term). It also removes any variable with a wrong-signed coefficient and any with a variance inflation factor above 5, though none was removed for those reasons. SC_INDEP retains: @@ind_vars@@.

@@tbl:sc_summary@@

Table: SC_FULL variables. IV is the variable's information value, coefficient is on WoE (negative as expected), points range is the spread between the best and worst bin.

The widest points range belongs to @@rg_top_v@@ (@@rg_top@@ points) and the narrowest to @@rg_low_v@@ (@@rg_low@@ points). Sub-grade and purpose carry most of the score, consistent with their information values.

![Points per bin for each SC_FULL variable.](@@chart_rel@@/scorecard_points_by_variable.png){width=90%}

![Weight of evidence by bin for the highest-IV variables.](@@chart_rel@@/woe_top_variables.png){width=90%}

## A worked loan

The first loan of the 2013 Excel sample (id @@wl_id@@, sub-grade @@wl_grade@@) is scored below. For each variable the loan's value is located in a bin, the bin's WoE is read from the points table, and the points follow from the formula.

@@tbl:wl_table@@

Table: Score walk-through for loan @@wl_id@@. Points are rounded to two decimals in the table.

The nine points values sum to @@wl_sum_pts@@, equal to the score recorded in the sample (@@wl_score@@). Going the other way, the sum of coefficient times WoE is @@wl_cw@@; adding the intercept @@intercept@@ gives a log-odds of bad of @@wl_margin@@. The probability is $1/(1+e^{-z})$ = @@wl_pd@@ (the pipeline stores @@wl_pd_file@@), equivalent to good:bad odds of @@wl_odds@@ to 1. As a check on the points formula, the offset per variable is $\text{Offset}/k =$ @@wl_off_k@@ with $k =$ @@wl_k@@ variables. This walk-through is also the first calculator in the Excel workbook (section 20), where a reviewer can change an attribute and watch the points move.

# Logistic regression and Wald p-values

## Intuition and the model

Given WoE-transformed inputs $x_i$, logistic regression models the probability of bad as $p_i = 1/(1+e^{-x_i^\top\beta})$ and estimates $\beta$ by maximum likelihood. Reviewers also ask whether each coefficient is statistically different from zero. The standard answer is the **Wald test**: divide the coefficient by its standard error and compare with a normal distribution.

## Fisher information and the Wald p-value

The log-likelihood is $\ell(\beta) = \sum_i [\,y_i \ln p_i + (1-y_i)\ln(1-p_i)\,]$. Its curvature at the maximum, the **Fisher information**, is

$$I(\hat\beta) = X^\top V X, \qquad V = \mathrm{diag}\big(\hat p_i(1-\hat p_i)\big),$$

where $X$ has a leading column of ones for the intercept. The large-sample covariance of the estimator is $I(\hat\beta)^{-1}$, so

$$\mathrm{SE}(\hat\beta_j) = \sqrt{\big[(X^\top V X)^{-1}\big]_{jj}}, \qquad z_j = \frac{\hat\beta_j}{\mathrm{SE}(\hat\beta_j)}, \qquad p_j = 2\,\Phi(-|z_j|).$$

The intercept must be a row and column of $X^\top V X$. Dropping it is not harmless. For a model with an intercept and one slope, write the matrix entries $F_{00} = \sum v_i$, $F_{01} = \sum v_i x_i$, $F_{11} = \sum v_i x_i^2$ with $v_i = \hat p_i(1-\hat p_i)$. The correct slope variance is $[F^{-1}]_{11} = F_{00}/(F_{00}F_{11} - F_{01}^2)$, which equals $1/(F_{11} - F_{01}^2/F_{00})$ and is never smaller than $1/F_{11}$, the value obtained when the intercept row and column are omitted. Omitting the intercept therefore always understates the standard error and overstates significance.

## Worked toy example

Take eight points with $x = 0, 1, \dots, 7$ and outcomes $y = 0,0,1,0,1,0,1,1$. The unpenalized maximum-likelihood fit is $\hat\beta_0 =$ @@wt_b0@@ and $\hat\beta_1 =$ @@wt_b1@@. At these estimates the information matrix has $F_{00} =$ @@wt_F00@@, $F_{01} =$ @@wt_F01@@ and $F_{11} =$ @@wt_F11@@, with determinant @@wt_det@@. Inverting gives $\mathrm{SE}(\hat\beta_1) =$ @@wt_se1@@, so $z =$ @@wt_z@@ and the two-sided p-value is @@wt_p@@; statsmodels returns @@wt_p_sm@@. If the intercept is omitted from the matrix the slope standard error becomes $1/\sqrt{F_{11}} =$ @@wt_se_wrong@@ and the p-value falls to @@wt_p_wrong@@, turning a non-significant slope into a "significant" one.

## The as-built defect and the corrected class

The original `LogisticRegression_with_p_values` built its Fisher matrix without an intercept column and wrapped scikit-learn's `LogisticRegression` with its default L2 penalty (`C=1`), so the coefficients were not the maximum-likelihood estimates at which the formula above is valid. Both effects invalidate the p-values (finding F-06). The corrected `stats_models.LogisticRegressionWithPValues` fits with `C=inf` (no penalty), carries the intercept in $X^\top V X$ and agrees with statsmodels to $10^{-5}$ in the unit tests; `LinearRegressionWithPValues` does the same for OLS with $n-p-1$ degrees of freedom. The scorecard itself uses statsmodels directly.

To see how much the defect matters on real data, the as-built logic was re-created on the LGD stage 1 design (a penalized fit, no intercept in the Fisher matrix) and compared with the corrected class. Over @@wald_n@@ terms, @@wald_sig_c@@ are significant at 5% with the corrected class and @@wald_sig_a@@ with the as-built logic; the verdict differs for @@wald_diff@@ term(s), and the median ratio of as-built to corrected standard errors is @@wald_se_ratio@@. The practical impact on this design is small because the inputs are standardized (nearly orthogonal to the constant) and the penalty is mild with 14 thousand observations. The defect is still a real correctness problem, because it would matter more for unstandardized inputs such as the dummy design of the PD model, and variable selection built on it is unsupported. It is mentioned honestly as a low-impact, high-principle finding.

# Reproduction of the original model

## What was reproduced

The original PD model is a logistic regression on a dummy design of coarse classes (@@repro_dummies@@ columns including a loan-age ladder). The cut points were transcribed from the original notebooks (`repro_original.ORIGINAL_COARSE_CLASSES`), reference categories dropped as in the original, the lifetime default flag used as target and a random 80/20 split on all loans applied. The original pickled model (`pd_model.sav`) is not available and the Final notebook contains no fitting cell for it (finding F-10), so the model was re-fit here with the same specification. The goal is not to match a number but to check whether the *design* behaves as claimed.

## Result on its own random test

@@tbl:repro@@

Table: Reproduction on the random 20% test (@@rp_ntest@@ loans; training @@rp_ntrain@@ loans, lifetime bad rate @@rp_bad@@). The loan-age proxy mths_since_issue_d is the variable of concern.

The reproduced model has Gini @@r:gini:repro_original:repro_test:f3@@ on its own random test, in line with a model of this kind. Removing the loan-age ladder lowers Gini by @@rp_gdrop@@, so the proxy contributes some ranking power on that test. That is the problem: loan age to a fixed reference date identifies the issue month, issue month determines how long a loan was observed, and a lifetime flag depends on how long a loan was observed. The feature can therefore learn the censoring pattern rather than borrower quality, and its gain on a random split does not carry to genuinely later loans.

## Out-of-time behaviour

Scored on the 12-month target, the reproduced model has Gini @@rep_g_o1@@ on 2013 and @@rep_g_o2@@ on 2014 loans, comparable to or higher than SC_FULL, so ranking alone does not expose the problem. What exposes it is stability. The score distribution drifts because loan age falls steadily for later issues: score PSI is @@r:psi_score:repro_original:oot1:f3@@ for 2013 and @@rep_psi2@@ for 2014 (Red), the mean predicted PD slides from @@rq_first@@ in early 2010 to @@rq_last@@ in the last quarter of 2014, and the quarterly PSI reaches @@rq_psi_last@@. Its calibration against the 12-month target is meaningless by construction because it predicts a lifetime outcome, which is why only its random-test calibration is reported. No amount of recalibration fixes a drift that comes from a mechanical feature.

![Quarterly population stability index of the score. The reproduced original model (dashed) breaks the 0.25 threshold from late 2013.](@@chart_rel@@/psi_by_quarter.png){width=90%}

This is the validation conclusion on the loan-age feature: it adds measurable apparent power in a random test, it drifts mechanically out of time, and it should be removed (finding F-01). The rebuilt scorecards do not contain it.
"""

SECTIONS_C = r"""
# Validation framework and thresholds

## Structure

Every test is run through one function layer (`validation.py`, `stability.py`, `benchmark.py`) that returns a row with the test name, model, sample, value, threshold text and a traffic light. `validation.run_battery` runs the full battery for every model on every applicable sample and writes `validation_results.csv`, so the tables in this document are views of that file. The tests fall into five families: discrimination (Gini, KS, deciles), calibration (central tendency, slope and intercept, Hosmer-Lemeshow, binomial by grade), stability (PSI, CSI), benchmarking (against sub_grade and between models) and outcomes (expected loss versus realized loss).

## Thresholds

| Test | Green | Amber | Red |
|:-----|------:|------:|----:|
| Gini | at least 0.40 | at least 0.30 | below 0.30 |
| KS | at least 0.30 | at least 0.20 | below 0.20 |
| Gini decay, holdout to out-of-time | at most 10% | at most 20% | above 20% |
| Score or feature PSI | at most 0.10 | at most 0.25 | above 0.25 |
| Hosmer-Lemeshow p-value | at least 0.05 | at least 0.01 | below 0.01 |
| Grade binomial p-value (Jeffreys) | at least 0.05 | at least 0.0001 | below 0.0001 |
| Central tendency, relative deviation | at most 10% | at most 25% | above 25% |
| Calibration slope | 0.9 to 1.1 | 0.8 to 1.2 | outside |
| Decile rank-order inversions | 0 | at most 2 | more than 2 |

Table: Traffic-light thresholds (`config.THRESHOLDS`).

These thresholds are common practitioner conventions chosen for this project. The PSI bands of 0.10 and 0.25 are the usual industry rule of thumb. The Gini and KS levels are generic and not tuned to unsecured consumer lending, where a Gini of 0.35 may be reasonable for a platform whose applicants were already screened, so Amber on discrimination should be read as "below a generic target" rather than "defective". The binomial traffic light is used by analogy with the Basel backtesting traffic light, which assigns green, yellow and red zones to the number of exceptions of a value-at-risk model (Basel Committee, 1996); the zone cut-offs here are this project's own p-value limits, not Basel's.

Benchmark rule: a model is Green against sub_grade if its AUC is higher and the DeLong test is significant at 5%, Amber if higher but not significant, and Red if it does not beat the benchmark. Expected-loss back-tests are rated by the ratio of expected to realized loss: Green within 10%, Amber within 25%, Red beyond.

# Tests: formulas, worked examples and results

## Discrimination: AUC, Gini and KS

**Intuition.** A good PD model gives bad loans higher PDs than good loans. The **AUC** is the probability that a randomly chosen bad loan has a higher predicted PD than a randomly chosen good loan (ties count half). The **Gini** (accuracy ratio) rescales it so that 0 is a random model and 1 a perfect one, and the **KS** statistic is the largest gap between the cumulative share of bads and the cumulative share of goods when loans are sorted from riskiest to safest.

$$\mathrm{AUC} = \frac{1}{n_1 n_0}\sum_{i:\,y_i=1}\ \sum_{j:\,y_j=0}\left[\mathbb{1}(p_i>p_j) + \tfrac12\,\mathbb{1}(p_i=p_j)\right], \qquad \mathrm{Gini} = 2\,\mathrm{AUC}-1, \qquad \mathrm{KS} = \max_t\big|F_1(t)-F_0(t)\big|.$$

**Worked toy example.** Six loans have predicted PDs 0.95, 0.85, 0.70, 0.55, 0.40, 0.20 and the first, second and fourth defaulted. There are $n_1 \times n_0 = 3 \times 3 =$ @@toy_pairs@@ bad-good pairs and @@toy_conc@@ of them rank the bad loan higher, so AUC = @@toy_auc@@ and Gini = @@toy_gini@@. Sorting riskiest first and accumulating:

@@tbl:toy_ks@@

Table: KS on the six-loan toy. The largest gap is the KS, @@toy_ks@@.

**Implementation.** `validation.auc` computes AUC from average ranks (the Mann-Whitney identity, exact with ties), `validation.ks` evaluates the gap only after the last tied score, and `validation.cap_table` builds the cumulative accuracy profile.

**Result.**

@@tbl:disc@@

Table: Discrimination by model and sample (all models, development holdout and out of time).

For SC_FULL the Gini is @@g_ho@@, @@g_o1@@ and @@g_o2@@ and KS is @@r:ks:sc_full:dev_holdout:f3@@, @@k_o1@@ and @@k_o2@@, all Amber. SC_INDEP is Red out of time (Gini @@ind_g1@@ and @@ind_g2@@). The 95% bootstrap interval for the SC_FULL Gini on 2013 loans is @@gci_lo1@@ to @@gci_hi1@@ (loan-level percentile bootstrap, section 12). On the 2013 sample the largest KS gap occurs at decile @@ks_dec_at@@.

![ROC curves on 2013 loans.](@@chart_rel@@/roc_oot1.png){width=70%}

![KS plot for SC_FULL on 2013 loans.](@@chart_rel@@/ks_oot1.png){width=70%}

![Cumulative accuracy profile on 2013 loans.](@@chart_rel@@/cap_oot1.png){width=70%}

## Rank ordering: deciles

Loans are ranked by predicted PD, riskiest first, and cut into ten equal groups with decile = $\lfloor 10(\text{rank}-1)/n\rfloor + 1$. A sound model has bad rates that fall monotonically down the table; any adjacent pair where the bad rate rises is an **inversion**. Lift is the decile bad rate divided by the portfolio bad rate.

@@tbl:dec_oot1@@

Table: SC_FULL deciles on 2013 loans (`deciles_sc_full_oot1.csv`).

The riskiest decile has an observed bad rate of @@dec1_rate@@ (lift @@dec1_lift@@) and holds @@dec1_share@@ of all bads; the safest decile has a bad rate of @@dec10_rate@@. There are @@r:decile_rank_inversions:sc_full:oot1:i@@ inversions on 2013 loans (Green). The model is also run at sub-grade level (33 to 35 groups with at least 100 loans): SC_FULL has @@r:subgrade_rank_inversions:sc_full:oot1:i@@ adjacent inversions on 2013 loans when sub-grades are ordered by their mean predicted PD, which is expected for small groups with noisy observed rates.

![Observed bad rate against predicted PD by decile, 2013 loans.](@@chart_rel@@/decile_bad_rate_oot1.png){width=80%}

## Calibration: central tendency, slope and intercept

**Central tendency** compares the average predicted PD with the observed bad rate, as a relative deviation $|\bar p - \bar y|/\bar y$. **Calibration slope and intercept** come from a logistic recalibration $\mathrm{logit}(P(y=1)) = a + b\,\mathrm{logit}(p)$ on the out-of-time data. A perfectly calibrated model has $a = 0$ and $b = 1$. A slope below 1 means predictions are too spread out (high PDs too high, low PDs too low); an intercept below zero means PDs are too high on average.

The **Brier score** $\frac1n\sum(p_i-y_i)^2$ rewards both discrimination and calibration, and the Brier skill score $1 - \mathrm{Brier}/(\bar y(1-\bar y))$ compares it with always predicting the portfolio rate.

@@tbl:calsum@@

Table: Calibration summary. Slope and intercept from the logistic recalibration; HL is the Hosmer-Lemeshow test (next section).

SC_FULL over-predicts on 2013 loans (mean PD @@r:mean_pd:sc_full:oot1:p2@@ against an observed @@r:obs_rate:sc_full:oot1:p2@@, relative deviation @@r:central_tendency:sc_full:oot1:p1@@, Amber) and is within @@r:central_tendency:sc_full:oot2:p1@@ on 2014 loans (Green). The slope is below one on both samples (@@r:cal_slope:sc_full:oot1:f3@@ and @@r:cal_slope:sc_full:oot2:f3@@) and the intercept is negative (@@r:cal_intercept:sc_full:oot1:f3@@, @@r:cal_intercept:sc_full:oot2:f3@@): the model is somewhat over-confident and too high on average, with the problem concentrated in the top deciles. The Brier skill is low (@@r:brier_skill:sc_full:oot1:f4@@ on 2013 loans) because defaults are rare events (about 4 to 5%) and the achievable improvement over a constant is small.

![Reliability curve for SC_FULL: development holdout and the two out-of-time samples.](@@chart_rel@@/calibration_dev_oot1_oot2.png){width=70%}

## Calibration: Hosmer-Lemeshow

**Intuition.** Sort loans by predicted PD, cut into $g$ equal-sized groups, and compare the expected number of bads in each group with the number observed. **Math.** For group $k$ with $n_k$ loans, mean predicted PD $\bar p_k$ and $O_k$ observed bads,

$$H = \sum_{k=1}^{g} \frac{(O_k - n_k\bar p_k)^2}{n_k\,\bar p_k(1-\bar p_k)} \;\sim\; \chi^2_{g-2}\ \text{in sample},\qquad \chi^2_{g}\ \text{out of sample}$$

(Hosmer and Lemeshow, 1980). The pipeline uses $g=10$ with $g-2$ degrees of freedom on the development training sample and $g$ on holdout and out-of-time data (`validation.hosmer_lemeshow`).

**Worked toy example.** Three groups of 1,000 loans with mean PDs 2%, 5%, 10% and observed bads 15, 60, 125:

@@tbl:toy_hl@@

Table: Hosmer-Lemeshow toy. The statistic is @@hlt_stat@@ on 3 degrees of freedom, p-value @@hlt_p@@.

**Result on real data.** The ten groups for SC_FULL on 2013 loans:

@@tbl:hl_oot1@@

Table: Hosmer-Lemeshow groups for SC_FULL on 2013 loans. The contributions sum to @@hl_sum@@ on 10 degrees of freedom, p-value @@hl_p_calc@@.

The statistic is @@r:hl_stat:sc_full:oot1:f1@@ on 2013 loans (Red) and @@r:hl_stat:sc_full:oot2:f1@@ on 2014 loans (Red), against @@r:hl_stat:sc_full:dev_holdout:f1@@ on the holdout (p = @@r:hl_p:sc_full:dev_holdout:pv@@, Green). The tenth group alone (@@cal_b10_n@@ loans, predicted @@cal_b10_pred@@, observed @@cal_b10_obs@@, expected @@cal_b10_e@@ bads against @@cal_b10_o@@ actual) contributes @@cal_b10_c@@, or @@hl_share10@@ of the total.

**Honest reading, including the power caveat.** With over 130,000 loans in a sample the test has so much power that it rejects deviations of a fraction of a percentage point. A 1-point gap between predicted and observed in a group of 13,000 loans is statistically overwhelming but economically small. The statistic is therefore not dismissed here, and it is not obeyed blindly either. The Red result is real in the sense that the model is mis-calibrated out of time: the gap is concentrated in the riskiest decile, predicted @@cal_b10_pred@@ against observed @@cal_b10_obs@@, an over-prediction of about @@cal_b10_over@@ in the group that matters most for pricing and cut-offs. What the test cannot say is whether that gap is large enough to matter commercially. The calibration slope (Green) and the decile table are the better guide to magnitude, and the fix is a recalibration (below), not a rebuild. The pipeline flags every Hosmer-Lemeshow result on samples above 100,000 loans with this caveat.

A recalibrated challenger shifts only the intercept by @@recal_shift@@ so that the mean PD equals the 2013 bad rate. It is evaluated on 2014 loans only (using 2013 outcomes to set the shift would make 2013 results in-sample). It does not improve the 2014 level: mean PD becomes @@r:mean_pd:sc_full_recal:oot2:p2@@ against @@r:obs_rate:sc_full_recal:oot2:p2@@ observed (relative deviation @@r:central_tendency:sc_full_recal:oot2:p1@@ against @@r:central_tendency:sc_full:oot2:p1@@ unadjusted) because the default rate rose in 2014. It also remains Red on Hosmer-Lemeshow (p = @@r:hl_p:sc_full_recal:oot2:pv@@) and on the grade binomial test, because the gaps are by grade and by decile rather than in the overall level, which a single shift cannot fix.

## Calibration: binomial test by grade

**Intuition.** For one grade, if the predicted PD is right, the number of defaults should look like a binomial draw. **Math.** For a group with $n$ loans, mean PD $\bar p$ and $D$ observed bads, test that PD is *understated* (the dangerous direction) with

$$z = \frac{D - n\bar p}{\sqrt{n\,\bar p\,(1-\bar p)}}, \qquad p_{\text{binom}} = P\big(X \ge D\big),\ X\sim\mathrm{Bin}(n,\bar p), \qquad p_{\text{Jeffreys}} = P\big(\pi < \bar p\big),\ \pi\sim\mathrm{Beta}\big(D+\tfrac12,\ n-D+\tfrac12\big).$$

The Jeffreys form is the posterior probability, under a non-informative prior, that the true PD is below the prediction. It behaves better than the normal approximation for small counts (`validation.binomial_test_by_group`).

**Worked example.** Grade E in 2014 for SC_FULL: $n =$ @@bn_n@@ loans, mean PD @@bn_p@@, so the expected number of bads is @@bn_exp@@ and the standard deviation is @@bn_sd@@. The observed count is @@bn_d@@ (rate @@bn_obs@@), so $z =$ @@bn_z@@ and the upper-tail p-value is essentially zero.

@@tbl:grade_bin1@@

Table: SC_FULL by grade on 2013 loans. Negative z means PD is over-predicted.

@@tbl:grade_bin@@

Table: SC_FULL by grade on 2014 loans. Grades E, F and G are significantly under-predicted.

In 2013 no grade is flagged (smallest Jeffreys p @@r:binomial_grade_min_p:sc_full:oot1:pv@@, Green) because the model over-predicts A to D and is close for E to G. In 2014 the picture reverses: the observed rates for grades E, F and G (@@bn_obs@@ for E) exceed the predictions, giving a smallest p of @@r:binomial_grade_min_p:sc_full:oot2:pv@@ (Red). The independent scorecard is worse: it under-predicts grades D to G in both years, which is the circularity again, because it cannot see the lender's grade. Defaults within a grade are correlated through common conditions, so the binomial test (which assumes independence) overstates how surprising the gaps are. The Red result is nonetheless consistent with the other evidence that the 2014 cohort deteriorated in the riskier grades.

# Challenger models, DeLong test and bootstrap

## Challengers

Two gradient-boosted tree models challenge the scorecard on the same 26 features. XGBoost (Chen and Guestrin, 2016) and LightGBM both grow an ensemble of shallow decision trees, each new tree fitted to the errors of the ensemble so far. Monotone constraints force the predicted PD to move in the economically expected direction for interest rate, DTI, inquiries, revolving utilization, loan-to-income, payment-to-income (increasing) and annual income and credit age (decreasing). Hyper-parameters (learning rate, depth, leaves, minimum child weight, subsampling, regularization, tree count) are chosen by random search with **expanding-window time folds**: models train on issue years before each validation year (2010, 2011, 2012) and are scored on that year's log-loss with early stopping. This avoids tuning on information from the future. The tuned models use @@xgb_trees@@ (XGBoost) and @@lgbm_trees@@ (LightGBM) trees. Probabilities are calibrated on the development holdout (never on out-of-time data) by Platt scaling or isotonic regression, chosen by cross-fitted Brier score; the methods chosen were @@xgb_cal@@ for XGBoost and @@lgbm_cal@@ for LightGBM.

## DeLong test for correlated AUCs

**Intuition.** Two models scored on the same loans have correlated AUCs, so comparing them with an unpaired test throws away information. DeLong, DeLong and Clarke-Pearson (1988) show that the AUC is a U-statistic and give a variance for the difference that accounts for the correlation.

**Math.** For $m$ bads with scores $X_i$ and $n$ goods with scores $Y_j$, define the structural components $V_{10}(X_i) = \frac1n\sum_j \psi(X_i,Y_j)$ and $V_{01}(Y_j) = \frac1m\sum_i \psi(X_i,Y_j)$, where $\psi = 1$ if $X>Y$, $\tfrac12$ if equal and $0$ otherwise. The AUC is the mean of the $V_{10}$ (or of the $V_{01}$). With $S_{10}$ and $S_{01}$ the covariance matrices of the two models' components, the covariance of the two AUCs is $S = S_{10}/m + S_{01}/n$ and, for the difference,

$$z = \frac{\widehat{\mathrm{AUC}}_1 - \widehat{\mathrm{AUC}}_2}{\sqrt{S_{11} + S_{22} - 2S_{12}}}.$$

Two sets of six scores for the toy outcomes give AUCs of @@dl_a1@@ and @@dl_a2@@, a difference of @@dl_d@@, which is far too small a sample to be significant; the same code applied to 100,000 loans is below. The implementation is `validation.delong_test`, computed from average ranks.

## Bootstrap confidence intervals

For Gini and for AUC differences the pipeline resamples *loans* with replacement 500 times and takes the 2.5th and 97.5th percentiles (`validation.bootstrap_ci`, `bootstrap_diff_ci`). The difference is paired: the same resampled loans are scored by both models. To keep run time bounded the bootstrap uses a random subsample of up to 50,000 loans when a sample is larger, which makes the interval somewhat wider than a full-sample one. Loans, not time periods, are resampled, so the interval captures sampling noise only and not variation across economic conditions.

## Results

@@tbl:bench_cmp@@

Table: Model comparisons on the out-of-time samples. "vs_sub_grade" compares a scorecard with sub_grade used as a score; "challenger_vs_champion" compares each model with SC_FULL. Light shows the benchmark rule for the first and is informational for the others.

XGBoost improves on SC_FULL by @@bm:challenger_vs_champion:oot1:xgb:d_auc:s4@@ AUC on 2013 and @@bm:challenger_vs_champion:oot2:xgb:d_auc:s4@@ on 2014 and LightGBM by @@bm:challenger_vs_champion:oot1:lgbm:d_auc:s4@@ and @@bm:challenger_vs_champion:oot2:lgbm:d_auc:s4@@. With this many loans the DeLong p-values are far below 0.001, so the improvement is real, and the bootstrap interval for XGBoost on 2013 (@@bm:challenger_vs_champion:oot1:xgb:ci_lo:f4@@ to @@bm:challenger_vs_champion:oot1:xgb:ci_hi:f4@@) excludes zero. But the gain is about one AUC point, the trees remain Amber on Gini (@@r:gini:xgb:oot1:f3@@ and @@r:gini:lgbm:oot1:f3@@ on 2013) and they remain Red on Hosmer-Lemeshow. The trees also show strong overfitting in development (training Gini @@r:gini:lgbm:dev_train:f3@@ for LightGBM against @@r:gini:lgbm:oot1:f3@@ out of time) and a training calibration slope far from one, which is why calibration is always done on the holdout. The pragmatic conclusion is that nonlinear structure adds little over the scorecard on these features, so the scorecard's transparency is not bought at a large accuracy cost.

# Benchmark against the LendingClub grade and the circularity

## The benchmark

Any new model should beat something simple. The obvious simple model on this data is the lender's own sub-grade (35 levels, A1 to G5) treated as an ordinal risk score, with the interest rate and the 7-level grade as alternatives. Each is evaluated alone, with no fitting.

@@tbl:bench_stand@@

Table: AUC of the lender's own ratings used as scores.

On 2013 loans sub_grade alone has AUC @@bm:standalone:oot1:sub_grade_rank:auc_a:f4@@ and interest rate alone @@bm:standalone:oot1:int_rate:auc_a:f4@@; on 2014 loans the figures are @@bm:standalone:oot2:sub_grade_rank:auc_a:f4@@ and @@bm:standalone:oot2:int_rate:auc_a:f4@@. A one-variable score that costs nothing is the benchmark every candidate has to beat.

## Does the scorecard add value?

The comparison table in section 12 gives the answer. SC_FULL differs from sub_grade by @@bm:vs_sub_grade:oot1:sc_full:d_auc:s4@@ AUC on 2013 loans (DeLong p = @@bm:vs_sub_grade:oot1:sc_full:delong_p:pv@@, not significant, Amber) and by @@bm:vs_sub_grade:oot2:sc_full:d_auc:s4@@ on 2014 loans (p = @@bm:vs_sub_grade:oot2:sc_full:delong_p:pv@@, significant and negative, Red). In other words, the champion does not add significant discrimination to the lender's own score, and in 2014 it is slightly worse. This is the headline negative result of the validation.

## The circularity

The independent scorecard SC_INDEP, built only from borrower and loan characteristics other than LendingClub's outputs, is behind the benchmark by @@bm:vs_sub_grade:oot1:sc_indep:d_auc:s4@@ and @@bm:vs_sub_grade:oot2:sc_indep:d_auc:s4@@ AUC (both Red), and adding it to sub_grade in a combined logit does not help either: the combined model is below sub_grade alone on 2013 (@@bm:incremental_over_sub_grade:oot1:sc_indep+sub_grade:d_auc:s4@@ AUC, p = @@bm:incremental_over_sub_grade:oot1:sc_indep+sub_grade:delong_p:pv@@) and on 2014 (@@bm:incremental_over_sub_grade:oot2:sc_indep+sub_grade:d_auc:s4@@, p = @@bm:incremental_over_sub_grade:oot2:sc_indep+sub_grade:delong_p:pv@@). The cost of excluding the lender's own score is a Gini drop from @@g_o1@@ to @@ind_g1@@ on 2013 and from @@g_o2@@ to @@ind_g2@@ on 2014, and the independent model also decays faster over time.

![Gini with and without LendingClub's risk outputs.](@@chart_rel@@/gini_full_vs_indep.png){width=70%}

How to interpret this. LendingClub's grade summarises information the lender had at origination, much of it not in the public file (credit bureau attributes such as a FICO band). A model without it is missing real signal, so the lower SC_INDEP performance is not "bad modelling" alone. At the same time, a model that leans on the grade inherits the lender's own score, its drift and any errors in it, and cannot be called independent evidence of credit quality. The recommended handling is to report both models, document the dependency, and treat the independent model as the measure of what the borrower characteristics add on their own.
"""

SECTIONS_D = r"""
# Stability by vintage: PSI and CSI

## Intuition and formula

A model is only valid for the population it was built on. The **population stability index** (PSI) measures how much a distribution has moved. Bin the development scores into deciles and let $e_i$ be the share of development loans in bin $i$ and $a_i$ the share of a later sample in the same bins:

$$\mathrm{PSI} = \sum_i (a_i - e_i)\ln\frac{a_i}{e_i}.$$

Every term is non-negative, the index is zero when the distributions match, and it equals the sum of the two Kullback-Leibler divergences between them. Shares are floored at $10^{-4}$ before the logarithm so an empty bin does not produce infinity. Rules of thumb: below 0.10 stable, 0.10 to 0.25 some shift, above 0.25 material shift. Applied to a model input it is called the **characteristic stability index** (CSI); numeric inputs use development deciles plus a separate missing bin and categorical inputs use their categories (`stability.psi`, `csi`, `psi_by_quarter`).

## Worked example: initial_list_status

The field takes two values. In the development training sample and in 2014 loans the shares are:

@@tbl:psi_ex@@

Table: PSI by hand for initial_list_status, development training sample versus 2014 loans. The total is @@psi_ex_total@@, matching the CSI in `psi_csi.csv`.

Each row contributes (actual minus expected) times the log ratio; the "w" row dominates because its share moved from a few percent to about half of the loans. That is the structural break from section 3, not borrower behaviour.

## Results

@@tbl:psi_score@@

Table: Score PSI against the development training sample.

Score PSI is tiny for the scorecard and the trees (SC_FULL @@psi_champ1@@ in 2013 and @@psi_champ2@@ in 2014), and the largest quarterly PSI over the out-of-time quarters is @@q_max_champ@@ for SC_FULL across all quarters. The score distribution is stable because the models rank on a handful of characteristics whose joint distribution did not move much. The reproduced original model is the exception (section 9).

@@tbl:csi_top@@

Table: Largest characteristic stability indices, ranked by 2014. The development reference is the training sample of the champion.

initial_list_status is Red in both years (@@csi_ils1@@ and @@csi_ils2@@); mths_since_last_record rises from @@csi_rec1@@ to @@csi_rec2@@ (Red in 2014); purpose and state also drift into the Amber range. The maximum CSI is rated Red on both samples (@@r:csi_max:all:oot1:f3@@ and @@r:csi_max:all:oot2:f3@@) with @@r:csi_n_red:all:oot1:i@@ and @@r:csi_n_red:all:oot2:i@@ features in the Red zone. This does not invalidate the score (it is stable) but confirms that the development period mixes data regimes, and it argues for monitoring the inputs as well as the output.

## Performance by vintage

@@tbl:gini_vintage@@

Table: Gini by issue year on the development and out-of-time samples (champion, independent scorecard and LightGBM). Early years are small samples.

Discrimination is similar across 2012 to 2014 for the scorecard. The high values for 2007 to 2011 are partly in-sample (these loans are in the development sample) and partly small-sample noise (251 loans in 2007), so they are not evidence of better performance in those years.

![Gini by vintage.](@@chart_rel@@/gini_by_vintage.png){width=80%}

![Largest characteristic stability indices.](@@chart_rel@@/csi_top_variables.png){width=80%}

# LGD, EAD and expected-loss back-tests

## Population and the two-stage LGD design

LGD and EAD are modeled on loans that actually defaulted. The charged-off population is recomputed from the raw file (@@lg_n@@ loans, including the charged-off status under the old credit policy) and reconciled to the original project's `loan_data_defaults.csv`: @@lg_match@@ loans match and the recovery-rate and credit-conversion-factor columns agree to within @@lg_diff@@. The mean recovery rate (recoveries divided by funded amount) is @@lg_rr@@ and the mean credit conversion factor is @@lg_ccf@@.

LGD follows the original two-stage design. **Stage 1** is a logistic regression for whether any recovery occurs. **Stage 2** is a linear regression for the recovery rate among loans that do recover. The expected LGD is

$$\mathrm{LGD} = 1 - P(\text{recovery})\times E[\text{recovery rate}\mid\text{recovery}],$$

clipped to $[0,1]$, so the stage 1 probability is used as a probability and not as a 0/1 label (the original applied a hard cut, finding F-05; the hard-class version is kept as a comparison). **EAD** is the credit conversion factor $\mathrm{CCF} = (\text{funded} - \text{principal repaid})/\text{funded}$ predicted by a linear regression and multiplied by the funded amount. All three are fitted on defaults of loans issued to 2012 (@@nd_dev_le_2012@@ loans for stage 1; @@nfit_stage2@@ with a recovery for stage 2), evaluated on 2013 loans (@@nd_oot_2013@@ defaults) and reported but not fitted on 2014 loans (@@nd_report_2014@@). The design uses winsorized, standardized application variables, with grade and state left out (grade is nearly collinear with interest rate and state is sparse) and uses the corrected p-value classes of section 8.

## Results

@@tbl:lgdead@@

Table: LGD and EAD diagnostics by sample (`lgd_ead_diagnostics.csv`).

- **Stage 1 has little power.** AUC is @@s1auc_d@@ on development defaults, @@s1auc_o@@ on 2013 loans and @@s1auc_r@@ on 2014 loans. Only @@s1_n_sig@@ of @@s1_n@@ terms are significant at 5%.
- **LGD is nearly constant.** Mean actual LGD is @@lgma_d@@ in development and @@lgma_o@@ out of time; the model predicts @@lgmp_o@@ for 2013 loans. R-squared is @@lgr2_d@@ in development and @@lgr2_o@@ out of time and the correlation between predicted and actual LGD is @@lgcorr_o@@. The expected-value combination is slightly better than the hard-class version (R-squared @@lgr2h_o@@ out of time), confirming the remediation, but both are weak.
- **The CCF model has modest power and shifts over time.** R-squared is @@ccfr2_d@@ in development, @@ccfr2_o@@ on 2013 loans and @@ccfr2_r@@ on 2014 loans. The 2014 value is strongly negative because loans issued in 2014 that charged off did so very early, so little principal had been repaid: actual mean CCF is @@ccfma_r@@ against a prediction of @@ccfmp_r@@. The model captures the ranking (correlation @@ccfcorr_o@@ on 2013 loans) but not the level shift.
- **Dollar EAD looks good only because of the funded amount.** R-squared of predicted EAD in dollars is @@eadr2_o@@ out of time, but it is dominated by the loan size multiplied into both the actual and the predicted value; the CCF R-squared above is the honest measure.
- **Heteroskedasticity.** The Breusch-Pagan test rejects constant variance for stage 2 (p = @@bp_stage2@@) and EAD (p = @@bp_ead@@), so classical standard errors for those two regressions are unreliable and heteroskedasticity-robust errors should be used in a production version.

![Predicted versus actual CCF and LGD by decile on 2013 defaults.](@@chart_rel@@/lgd_ccf_deciles.png){width=90%}

Coefficients with the largest absolute z for the two main regressions are below.

@@tbl:stage1_top@@

Table: LGD stage 1 (any recovery), largest |z| terms.

@@tbl:ead_top@@

Table: EAD (CCF) regression, largest |t| terms.

## Immature recoveries

Recoveries arrive months or years after charge-off. Loans that defaulted late in the observation window have had little time to recover, so recovery rates and the share with any recovery are understated for them.

@@tbl:recov@@

Table: Charged-off loans by year of last payment (a proxy for default year, because the file has no default date).

The share of defaults with any recovery is @@rec2012@@ for defaults with a last payment in 2012 and @@rec2013@@ in 2013, falls to @@rec2014@@ in 2014 and to @@rec2015@@ in 2015. The 2014 and 2015 figures are immature, not a change in recovery behaviour, so 2014 loans are reported but not fitted and the stage 1 model, which predicts a recovery probability near 74% for every sample, over-predicts for recent cohorts mechanically (finding F-11). A production LGD should be built on mature default cohorts or use a development-pattern adjustment.

## Expected-loss back-test: 12-month

For each loan the 12-month expected loss is $EL = PD_{12}\times LGD\times EAD$, with the champion scorecard's PD. The back-test sums EL and compares it with the realized net loss of loans that actually went bad inside the 12-month window (funded minus principal repaid minus recoveries plus collection fees).

@@tbl:el12@@

Table: 12-month expected versus realized loss by vintage. Vintages to 2012 are development (in-sample PD, LGD and EAD) and 2013 and 2014 are out of time. For 2014 the realized loss is incomplete because loans that are bad but not yet charged off carry no realized loss.

For the 2013 vintage the model expects @@el2013@@ (@@elp2013@@ of funded) against a realized @@re2013@@ (@@rep2013@@), a ratio of @@ratio2013@@. For 2014 the figures are @@el2014@@ against @@re2014@@ (ratio @@ratio2014@@); since 2014 realized loss is a lower bound, the true shortfall is larger. In development the ratio is also below one in every vintage, so the shortfall is not purely out-of-time. By grade for 2013:

@@tbl:el13@@

Table: 12-month expected versus realized loss by grade, 2013 vintage.

Grades A and C are close (ratios near one), B is slightly under-estimated; D, E, F and G are under-estimated, in line with their PDs and with the fall in LGD predictions for riskier loans. For 2014 the ratios for the riskiest grades are @@el14_g_ratio@@.

![12-month expected versus realized loss by grade, 2013 vintage.](@@chart_rel@@/el_backtest_by_grade.png){width=70%}

## Expected-loss back-test: lifetime

The original project computed a lifetime expected loss of @@orig_el_pct@@ of funded amount (@@orig_el_usd@@) by applying a lifetime PD to every loan, including loans already repaid, charged-off or seasoned (finding F-04). That number cannot be compared with a realized loss. The lifetime back-test here uses only the complete 36-month cohort: a scorecard is refit with the lifetime flag on 36-month loans issued from 2007 to 2011 and tested on 2012 loans against realized lifetime net loss.

@@tbl:elife@@

Table: Lifetime expected versus realized loss by grade for 2012 36-month loans (`el_backtest_lifetime.csv`).

On 2012 loans the expected loss is @@lf_el@@ of funded against a realized @@lf_re@@ (ratio @@lf_ratio@@), whereas in the training years the ratio is @@lf_tr_ratio@@, so the in-sample match deteriorates out of time. The predicted lifetime EL is lower than the original project's headline figure, as it should be: the original applied lifetime PD to every loan regardless of status.

# Sensitivity and stress analysis

## Design

Sensitivity analysis asks how the answer moves when assumptions move. All runs use 2013 loans and the champion unless noted. Feature shocks add 5 points to DTI, 2 points to the interest rate, cut income by 10%, add 10 points to revolving utilization and add one inquiry, singly and together. PD multipliers scale every PD by 1.25 and 1.5, and an LGD add-on of 0.10 is applied. Definition variants refit the scorecard under a different target (In Grace Period as bad, a 24-month window, DNMCP loans included) or different binning settings (minimum share 3, 5 and 10%, maximum bins 5, 8 and 12) and report the new Gini on 2013 loans.

@@tbl:sens@@

Table: Sensitivity of SC_FULL on 2013 loans. Refit variants report Gini; others report rank correlation with the baseline PDs and the share of loans that change PD band.

## Reading the results

- **Inquiries dominate the feature shocks.** One extra inquiry moves mean PD by @@shock_inq_add_pd@@ and expected loss by @@shock_inq_add_el@@, because most applicants have zero or one inquiry and the first bin carries a large positive WoE, so one more inquiry moves a large share of loans into a worse points bin. The combined shock moves PD by @@shock_combined_pd@@ and EL by @@shock_combined_el@@.
- **The interest-rate shock does nothing to the scorecard PD** (@@shock_int_rate_pts_pd@@) because SC_FULL uses sub_grade rather than the interest rate, but it moves EL by @@shock_int_rate_pts_el@@ through the LGD and EAD models, and it moves XGBoost PD by @@shock_int_rate_pts_pd_xgb@@. The sensitivity to a variable depends on which model is asked, which is the point of running several.
- **Target definition matters most.** A 24-month window raises mean PD to @@t24_mean@@ (@@t24_pd@@) and EL by @@t24_el@@, and the refit Gini is @@t24_gini@@. Because a lender's PD horizon is a policy choice, the horizon should always be stated next to the number.
- **In Grace Period coding and DNMCP inclusion are immaterial** for the 12-month target (mean PD changes @@grace_pd@@ and @@dn_pd@@).
- **Binning choices are not a source of fragility.** Refit Gini on 2013 loans ranges from @@bin_g_lo@@ to @@bin_g_hi@@ over the six binning settings.

![Change in expected loss under each scenario.](@@chart_rel@@/sensitivity_tornado.png){width=80%}

# Explainability: SHAP versus the scorecard

## SHAP in one page

For a tree model there is no coefficient table. **SHAP** values (Lundberg and Lee, 2017) assign to every feature a contribution to each prediction, derived from Shapley values in cooperative game theory: the contribution of feature $j$ to loan $i$ is the average change in the prediction when $j$ is added to every possible subset of the other features,

$$\phi_j = \sum_{S\subseteq F\setminus\{j\}}\frac{|S|!\,(|F|-|S|-1)!}{|F|!}\Big[f_{S\cup\{j\}}(x_S\cup x_j) - f_S(x_S)\Big],$$

and for a given loan the SHAP values add up to the difference between the prediction and the average prediction. For tree ensembles the sum can be computed exactly and quickly (`explain.shap_matrix`, using 5,000 loans from the 2013 sample). Here SHAP explains the model's raw log-odds, not the calibrated probability. Importance is the mean absolute SHAP value per feature.

## Comparison with the scorecard's own ranking

@@tbl:shap_cmp@@

Table: LightGBM importance (mean absolute SHAP) against the scorecard's IV and points range.

The leading drivers are the same: @@shap1v@@ is the top feature of the tree model (mean |SHAP| @@shap1@@) and @@shap2v@@ is second, while the scorecard places sub_grade, purpose and inquiries at the top by IV. Two differences are worth noting. The tree uses both int_rate and sub_grade (each carries lender information, and the scorecard keeps only sub_grade after the correlation filter), and the grade variable has SHAP importance of @@shap_grade@@ because sub_grade already contains it. And mths_since_last_delinq and payment-to-income appear in the tree's top ten but not in the scorecard: the first because the scorecard's IV filter dropped it for low IV (a missing-value pattern the tree can exploit) and the second because it is correlated with loan-to-income. The agreement in the main drivers is reassuring; the disagreement in the secondary ones is where the scorecard might be missing information.

![Mean absolute SHAP value by feature.](@@chart_rel@@/shap_importance.png){width=80%}

![SHAP dependence for sub-grade.](@@chart_rel@@/shap_dependence_sub_grade.png){width=60%}

# Findings log

@@f_n@@ findings are logged in `findings_log.csv`: @@f_hi@@ High, @@f_med@@ Medium and @@f_low@@ Low. Findings from the as-built review of the original project are marked "review"; findings from the automated test battery (any Red maps to High, any Amber to Medium) are marked "auto". The original developer (the author) reads the review findings as constructive: each has a remediation implemented in this project.

@@tbl:findings_counts@@

Table: Findings by severity and source.

## High-severity findings in full

@@tbl:findings_high@@

## Other findings

@@tbl:findings_rest@@

Table: Medium and Low findings (titles only). Descriptions, evidence and recommendations are in `findings_log.csv`.

The medium review findings cover invalid Wald p-values, the grade and interest-rate circularity, the last-payment-date default proxy, the In Grace Period coding mismatch, the missing fitting code with silent zero-filling of dummies, immature recoveries and structural breaks in inputs. The Low items are the EAD summary that prints the LGD stage 2 p-values (the EAD coefficients are re-estimated in `ead_coefs.csv`) and the two-digit year parse.

# Limitations

- **One lender, one era, and public data only.** The file contains 2007 to 2014 loans from one platform; the credit environment, underwriting rules and data fields changed during that time. Nothing here supports conclusions about other lenders or later years.
- **No true default date.** Timing is proxied by the last payment date; a loan that kept paying partially after missing payments is dated late. The 12-month window is therefore approximate.
- **Censoring and immature recoveries.** Even the fixed window has the 2014 cohort partly unresolved, and LGD for recent defaults is understated in recovery. The 2014 expected-loss comparison is a lower bound on realized loss.
- **No credit bureau score.** The public file has no FICO score, so the independent model is deprived of the most predictive underwriting input, and the benchmark (the lender's sub-grade) embeds it.
- **Generic thresholds.** The traffic-light cut-offs are conventions, not policy. Amber on Gini would not be a defect for every portfolio.
- **Power of large-sample tests.** Hosmer-Lemeshow and binomial tests reject small gaps; they are reported with the effect size alongside.
- **Same author.** The validator is the model's original developer, so independence is procedural only.
- **No macroeconomic dimension.** There is no stress calibrated to economic scenarios; the sensitivity analysis is mechanical shocks.
- **Reproduction without the original artifact.** The original pickled PD model is not available, so the reproduction re-fits a specification transcribed from the notebooks.

# Excel workbook walkthrough and reconciliation

## Purpose

The Excel workbook (`excel/credit_validation_workbook.xlsx`) re-implements the key validation calculations with live formulas on a stratified sample of 2013 loans (5,000 loans, sampled proportionally by grade with a floor so thin grades appear). A reviewer who does not trust code can change an input and watch every number move. The workbook is plain in format on purpose.

## Sheets

- **README** describes the workbook and the colour-free conventions.
- **Inputs** holds the scaling parameters (PDO, base score, base odds), the PSI floor, the traffic-light thresholds and the attributes for the calculator loan.
- **Scorecard** is the typed points table (bin edges, WoE, coefficients, points) exported from Python.
- **Score_Calc** is a live one-loan calculator: bin lookup, WoE, points, total score, log-odds and PD. The bin rule is the same as in Python (value greater than the lower edge and up to the upper edge, implemented as a count of edges below the value plus one), and categorical lookups use INDEX and MATCH.
- **Loan_Sample** holds the typed attributes and outcomes plus live bin index, WoE, points, score, PD and expected loss for each loan, alongside the Python PD and points.
- **Deciles** builds the decile table with COUNTIFS and SUMIFS, with a rank tie-break (RANK.EQ plus a running COUNTIF), the KS statistic and lift.
- **Calibration** computes decile and grade calibration, the Hosmer-Lemeshow statistic with CHISQ.DIST.RT, the binomial z with NORM.S.DIST and the Jeffreys p-value with BETA.DIST.
- **PSI** holds typed development bins and live out-of-time shares and the PSI.
- **EL** computes expected loss by grade against typed realized loss.
- **Checks** compares each Excel result with its Python value.
- **Conclusions** holds the static written conclusions.
- **Python_Ref** holds the Python reference values for reconciliation.

## Reconciliation

`reconcile_excel.py` recalculates the workbook headlessly, reads the values back, and compares them with independent Python computations at tight tolerances (PD 1e-9, points and score 1e-6, counts exact, KS, PSI and bad rates 1e-9, Hosmer-Lemeshow statistic 1e-6, p-values 1e-8, expected loss to the cent), on two input cases (the base loan and a loan with DTI raised by 5 points).

@@excel_block@@

Reconciliation shows that the formulas implement the same mathematics as the code, which lets a reviewer trace any number to a cell. It says nothing about whether the model is right for real loans.

# How to run

The raw data are not included. Place `loan_data_2007_2014.csv` and `loan_data_defaults.csv` in a folder and point `CV_RAW_DIR` at it.

```
cd python
py -3 -m pip install -r requirements.txt
set CV_RAW_DIR=C:\path\to\folder\with\loan_data_2007_2014.csv
py -3 -m credit_validation.cli            # full run (about 9 minutes)
py -3 -m credit_validation.cli --quick    # small run (about 2 minutes)
py -3 -m pytest -q
cd ..
py -3 docs/build_docs.py                  # rebuild this document and the README
```

The command-line run loads the data, builds the targets and samples, reproduces the original model, trains the scorecards and challengers, runs the validation battery, fits LGD and EAD and the expected-loss back-tests, runs the sensitivity analysis and SHAP, writes the findings log and charts, and builds and reconciles the Excel workbook (skip with `--skip-excel`). Every result file lands in `python/outputs/`. `docs/build_docs.py` reads those files and the cached loan parquet, templates every result into the text and rebuilds the Word document, so the document always matches the last run.

# Glossary

- **AUC**: probability that a random bad loan is ranked riskier than a random good loan.
- **Brier score**: mean squared difference between predicted probability and outcome.
- **CAP**: cumulative accuracy profile, share of bads captured against share of loans, riskiest first.
- **CCF**: credit conversion factor, the share of the funded amount still outstanding at default.
- **Censoring**: loans not observed long enough for the outcome to have occurred.
- **CSI**: characteristic stability index, PSI applied to one input variable.
- **DNMCP**: "does not meet the credit policy", a LendingClub status for loans under an earlier policy.
- **EAD**: exposure at default. **EL**: expected loss, PD times LGD times EAD.
- **Gini**: 2 AUC minus 1. **KS**: maximum separation between the cumulative bad and good distributions.
- **Hosmer-Lemeshow**: chi-squared test of predicted against observed bads in score groups.
- **IV**: information value, total separation of a binned variable.
- **LGD**: loss given default. **MRM**: model risk management.
- **OOT**: out of time, a sample from a period after development.
- **PD**: probability of default. **PDO**: points to double the odds.
- **PSI**: population stability index.
- **SHAP**: Shapley-value-based feature contributions.
- **SR 11-7**: the Federal Reserve and OCC supervisory guidance on model risk management.
- **WoE**: weight of evidence, the log ratio of the shares of goods and bads in a bin.

# References

Board of Governors of the Federal Reserve System and Office of the Comptroller of the Currency (2011). Supervisory Guidance on Model Risk Management, SR Letter 11-7, April 2011.

Basel Committee on Banking Supervision (1996). Supervisory Framework for the Use of "Backtesting" in Conjunction with the Internal Models Approach to Market Risk Capital Requirements. Bank for International Settlements.

Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring. Wiley.

DeLong, E. R., DeLong, D. M. and Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. Biometrics, 44(3), 837-845.

Hosmer, D. W. and Lemeshow, S. (1980). Goodness of fit tests for the multiple logistic regression model. Communications in Statistics, Theory and Methods, A10(10), 1043-1069.

Chen, T. and Guestrin, C. (2016). XGBoost: a scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems 30.

LendingClub. Public loan data, loans issued 2007 to 2014. Data source for all results in this document.
"""

README_TEMPLATE = r"""# Credit Risk Model Validation on LendingClub Data

**Data provenance and status.** All results use the public LendingClub loan data for loans issued @@issue_min@@ to @@issue_max@@ (@@n_loans@@ loans); LendingClub is the data source. This is an **independent validation exercise on public data and not a bank-approved or production model**. The validator is also the author of the original course project, so independence is procedural only. Raw data are not included in the repository (see the quick start).

## What it does

It validates and rebuilds the PD, LGD, EAD and expected-loss models of an earlier Master's course project in the style of SR 11-7: a fixed 12-month default definition on time-ordered samples, WoE scorecards scaled to points, gradient-boosted challengers, a full validation battery with traffic lights, a benchmark against LendingClub's own sub-grade, PSI and CSI stability, LGD, EAD and expected-loss back-tests, sensitivity analysis, SHAP, a findings log and an Excel workbook with live formulas. The full write-up, with theory and worked examples, is `docs/Credit_Risk_Model_Validation.docx` (source `docs/Credit_Risk_Model_Validation.md`).

## Quick start

```
cd python
py -3 -m pip install -r requirements.txt
set CV_RAW_DIR=C:\path\to\folder\with\loan_data_2007_2014.csv
py -3 -m credit_validation.cli --quick     # small run
py -3 -m credit_validation.cli             # full run, writes python/outputs/
py -3 -m pytest -q
cd ..
py -3 docs/build_docs.py                   # rebuild the document and this README
```

The folder in `CV_RAW_DIR` must contain `loan_data_2007_2014.csv` and `loan_data_defaults.csv` (not redistributed; obtain the LendingClub public data and place the files there).

## Headline results

@@tbl:readme_headline@@

Table: Out-of-time discrimination (2013 and 2014 issues). Development holdout Gini for SC_FULL is @@g_ho@@.

- The scorecard discriminates modestly (Gini @@g_o1@@ and @@g_o2@@, Amber) and is stable (score PSI @@psi_champ1@@ and @@psi_champ2@@).
- It does **not** beat LendingClub's own sub-grade as a score (AUC difference @@bm:vs_sub_grade:oot1:sc_full:d_auc:s4@@ in 2013, @@bm:vs_sub_grade:oot2:sc_full:d_auc:s4@@ in 2014), and the independent scorecard without grade and interest rate falls to Gini @@ind_g1@@ and @@ind_g2@@: the grade and interest-rate circularity.
- Hosmer-Lemeshow is Red out of time (a large-sample power effect, but the riskiest decile is over-predicted) and the 2014 grade binomial test is Red for grades E, F and G.
- Gradient boosting adds about one AUC point and does not fix calibration.
- LGD stage 1 and EAD are weak (stage 1 AUC @@s1auc_o@@, CCF R-squared @@ccfr2_o@@ out of time); 2014 recoveries are immature. The 12-month expected loss for 2013 is @@el2013@@ against realized @@re2013@@ (ratio @@ratio2013@@).
- The reproduced original model includes a loan-age proxy that drifts mechanically (score PSI @@rep_psi2@@ in 2014).
- @@f_n@@ findings are logged (@@f_hi@@ High, @@f_med@@ Medium, @@f_low@@ Low), including defects found in the original notebooks: In Grace Period coded good against the project notes, hard 0/1 labels in LGD stage 1, EAD summary printing LGD p-values, invalid Wald p-values, a vintage proxy feature, silent zero-fill of missing dummies, random-only splits and a lifetime-PD expected loss over all loans.

## Key findings

See section 18 of the document for the findings log (all High findings in full). Remediation of each defect is implemented in this project.

## Excel workbook

`excel/credit_validation_workbook.xlsx` re-implements the scorecard, deciles, calibration, PSI and expected-loss calculations with live formulas on 5,000 stratified 2013 loans. @@excel_line@@

## Repository layout

```
data/raw, data/interim     raw files (not committed) and parquet cache
docs/                      document (md, docx), build_docs.py, reference.docx
excel/                     workbook builder and workbook
python/credit_validation/  package: data, targets, splits, features, binning, scorecard,
                           repro_original, stats_models, models, validation, stability,
                           benchmark, lgd_ead, expected_loss, sensitivity, explain,
                           charts, findings, reconcile_excel, cli
python/tests/              pytest suites
python/outputs/            results (csv, json), charts/, models/, excel_inputs/
```

## Limitations

One lender, 2007 to 2014 only; default timing proxied by the last payment date; 2014 outcomes and recoveries partly unresolved; no bureau score; generic thresholds; large-sample tests reject small gaps; same-author validation. See section 19.

## Next steps

Recalibrate by grade on recent vintages; obtain true default dates and bureau scores; build LGD on mature cohorts with a development-pattern adjustment; use heteroskedasticity-robust errors in LGD and EAD; add macroeconomic scenarios; have the model independently reviewed by a different person.
"""


def _readme_table(T):
    rows = []
    for m, lab in (("sc_full", "SC_FULL scorecard"), ("sc_indep", "SC_INDEP scorecard"), ("xgb", "XGBoost"), ("lgbm", "LightGBM")):
        rows.append(dict(m=lab, a1=vr("auc", m, "oot1"), g1=vr("gini", m, "oot1"), k1=vr("ks", m, "oot1"), a2=vr("auc", m, "oot2"), g2=vr("gini", m, "oot2"), k2=vr("ks", m, "oot2")))
    T["readme_headline"] = md_table(pd.DataFrame(rows), {c: (lambda x: fx(x, 3)) for c in ("a1", "g1", "k1", "a2", "g2", "k2")},
                                    dict(m="Model", a1="AUC 2013", g1="Gini 2013", k1="KS 2013", a2="AUC 2014", g2="Gini 2014", k2="KS 2014"))


if __name__ == "__main__":
    main()
