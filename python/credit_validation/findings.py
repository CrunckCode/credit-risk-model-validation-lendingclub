"""Findings log: static as-built review findings on the original project plus findings derived from the validation results.

Severity rules: High = material error in model output, or Red on calibration/discrimination out-of-time;
Medium = Amber or a design weakness with bounded impact; Low = documentation or hygiene.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

# ===== CONFIG (user inputs) =====
LOG_FILE = "findings_log.csv"
LOG_COLUMNS = ("finding_id", "area", "title", "description", "evidence", "evidence_file", "severity", "traffic_light",
               "recommendation", "owner", "status", "source")
SEVERITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}
LIGHT_TO_SEVERITY = {"Red": "High", "Amber": "Medium"}
LIGHT_RANK = {"Green": 0, "Amber": 1, "Red": 2}
OWNERS = ("Model Developer", "Model Owner", "Validation")
RESULTS_FILE = "validation_results.csv"
TEST_COL, MODEL_COL, SPLIT_COL, VALUE_COL, THRESHOLD_COL, LIGHT_COL, EVIDENCE_FILE_COL = (
    "test", "model", "split", "value", "threshold", "light", "evidence_file")
CHAMPION = "sc_full"
AREA_KEYWORDS = (
    (("gini", "auc", "ks", "decile", "inversions", "cap", "benchmark", "incremental", "challenger"), "Discrimination"),
    (("hl", "hosmer", "binomial", "central", "cal", "brier", "calibration"), "Calibration"),
    (("psi", "csi", "stability"), "Stability"),
    (("lgd", "ead", "el", "backtest"), "Loss estimation"),
)
# which output file holds the detail behind a test; {model}/{split} are filled from the row
EVIDENCE_FILE_BY_PREFIX = (
    ("psi_quarter", "psi_by_quarter.csv"), ("psi", "psi_csi.csv"), ("csi", "psi_csi.csv"), ("benchmark", "benchmark.csv"),
    ("incremental", "benchmark.csv"), ("challenger", "benchmark.csv"), ("binomial_grade", "grade_binomial_{split}.csv"),
    ("hl_", "calibration_{model}_{split}.csv"), ("cal_", "calibration_{model}_{split}.csv"),
    ("central_tendency", "calibration_{model}_{split}.csv"), ("decile", "deciles_{model}_{split}.csv"),
)
# expected loss back-test: relative gap |EL / realized - 1| bands reuse the central-tendency thresholds
EL_GAP_BANDS = config.THRESHOLDS["central_tendency"]
# automated findings cover the validated scorecards and benchmark rows; challengers and the reproduction are comparators
AUTO_SCOPE_MODELS = ("sc_full", "sc_indep", "sc_full_recal", "all", "benchmark")
AUTO_SKIP_SPLITS = ("dev_train",)            # in-sample rows are not validation evidence
HL_POWER_NOTE = " Hosmer-Lemeshow has very high power at this sample size, so read it with the calibration slope and table."
EL_12M_FILE, EL_LIFE_FILE = "el_backtest_12m.csv", "el_backtest_lifetime.csv"
EL_ALL_LABEL, EL_OOT_SAMPLE, EL_LIFE_TEST_SAMPLE = "ALL", "out_of_time", "test_2012"
EXTRA_KEY_BY_TITLE_START = {"PD model uses mths_since_issue_d": "mths", "Only random 80/20": "random", "Lifetime default flag": "cens",
                            "Expected loss uses lifetime": "life", "LGD stage 1 applied": "lgd", "Grade and interest rate": "circ", "In Grace Period coded": "grace"}
RECOMMENDATION_BY_AREA = {
    "Discrimination": "Refit or redesign the model on the failing population and re-run the out-of-time discrimination tests.",
    "Calibration": "Recalibrate the PD (intercept or slope) on recent vintages and monitor calibration quarterly.",
    "Stability": "Investigate the drifting characteristics, document the cause, and set a re-development trigger.",
    "Loss estimation": "Re-estimate the loss component, document the bias, and apply a conservative overlay until fixed.",
    "Performance": "Investigate the failing test and agree a remediation date with the model owner.",
}
# ===== END CONFIG =====


def _f(area, title, description, evidence, evidence_file, severity, light, recommendation, owner):
    return dict(area=area, title=title, description=description, evidence=evidence, evidence_file=evidence_file,
                severity=severity, traffic_light=light, recommendation=recommendation, owner=owner)


# evidence numbers below are from the verified data facts in INTERFACES.md or from the original notebook cells
REVIEW_FINDINGS = [
    _f("Model design", "PD model uses mths_since_issue_d, a vintage and censoring proxy",
       "The original PD model includes loan age measured from issue date to a fixed reference date. It encodes vintage, and recent vintages "
       "have shorter observation windows, so the feature partly learns the censoring pattern instead of borrower risk.",
       "Original notebook feature list contains mths_since_issue_d; lifetime bad rate by vintage falls from 26% (2007) to 9% (2014) "
       "while the fixed 12-month flag is flat at about 4-5% from 2010 to 2014.",
       "validation_results.csv", "High", "Red",
       "Remove the feature. Use application-time characteristics only and validate on time-ordered out-of-time samples.", "Model Developer"),
    _f("Data and sampling", "Only random 80/20 splits; no out-of-time test",
       "Development and test sets are random draws over 2007-2014, so the test set shares vintages and economic conditions with training. "
       "Performance on it cannot show how the model behaves on later cohorts.",
       "Original notebook splits all 466,285 loans at random 80/20. This validation uses development 2007-06 to 2012-12, OOT1 2013 and OOT2 2014-01 to 2014-11.",
       "validation_results.csv", "High", "Red",
       "Adopt time-ordered development, out-of-time and recent-vintage monitoring samples as the standard test design.", "Model Developer"),
    _f("Target definition", "Lifetime default flag is right-censored",
       "A loan issued in 2014 has had at most about two years to default while a 2007 loan has had the full term, so the lifetime bad rate "
       "falls mechanically with vintage and is not comparable across cohorts.",
       "Lifetime bad rate by vintage falls from 26% (2007) to 9% (2014); the fixed 12-month window with an observation buffer is flat at about 4-5% from 2010 to 2014.",
       "by_vintage.csv", "High", "Red",
       "Define default on a fixed performance window and keep only loans observed for the full window plus a buffer.", "Model Developer"),
    _f("Loss estimation", "Expected loss uses lifetime PD over all loans",
       "Applying a lifetime PD to every loan, including already-resolved and seasoned loans, mixes bases and cannot be compared with realized loss. "
       "The headline expected loss is therefore not directly comparable with a realized loss.",
       "Original expected loss is 7.98% of funded amount, $531.9M. The back-test on the complete 36-month cohort is in el_backtest_lifetime.csv.",
       "el_backtest_lifetime.csv", "High", "Red",
       "Compute expected loss on a stated horizon (12-month) for the performing book and back-test against realized loss by grade and vintage.", "Model Developer"),
    _f("LGD model", "LGD stage 1 applied as hard 0/1 labels",
       "The stage 1 recovery model outputs a probability of any recovery, but the original notebook converts it into hard classes (Final notebook cell 132) "
       "before combining with the stage 2 recovery rate. This discards the probability information.",
       "Original: stage 1 thresholded to 0/1 in cell 132. Overall mean recovery rate is 0.061 and 56% of charged-off loans have any recovery, so a hard 0/1 cut differs from the expected value. "
       "Expected-value and hard-class LGD are compared in lgd_ead_summary.json.",
       "lgd_ead_summary.json", "High", "Red",
       "Combine as LGD = 1 - P(recovery) * E[recovery rate | recovery]; report the hard-class version only as a comparison.", "Model Developer"),
    _f("Statistics", "Wald p-values from the custom logistic class are invalid",
       "The original LogisticRegression_with_p_values omits the intercept from the Fisher information matrix and wraps a penalized sklearn fit (C=1). "
       "Standard errors and p-values are therefore not those of the fitted model, and variable selection built on them is unsupported.",
       "Original class: Fisher matrix built without an intercept column; sklearn LogisticRegression with default C=1. The corrected class matches statsmodels to 1e-5 in the unit tests.",
       "", "Medium", "Amber",
       "Use an unpenalized fit with the intercept in X'VX, or statsmodels, and re-run variable selection.", "Model Developer"),
    _f("Model design", "Grade and interest rate (LendingClub risk outputs) used as PD features",
       "grade, sub_grade, int_rate and installment are LendingClub's own underwriting outputs, so the model partly re-learns the lender's score and "
       "inherits its drift. Independent value added has to be shown against a model that excludes them.",
       "Two champions are fitted: SC_FULL with these inputs and SC_INDEP without. Their Gini on each sample and the benchmark against sub_grade as a score are in benchmark.csv.",
       "benchmark.csv", "Medium", "Amber",
       "Report independent-feature performance next to the full model and document why risk outputs are used or excluded.", "Model Owner"),
    _f("Target definition", "Default timing proxied by last payment date",
       "The date of default is not in the data, so months to default is approximated by months from issue to last_pymnt_d. For loans that stopped paying early this is reasonable; "
       "for loans that kept paying a partial amount it overstates time to default.",
       "The 12-month bad flag requires a bad status AND months from issue to last_pymnt_d below 12, with never-paid loans counted as 0.",
       "sensitivity.csv", "Medium", "Amber",
       "Obtain true default dates or document the proxy as a limitation and run the 24-month window variant.", "Model Developer"),
    _f("Target definition", "In Grace Period coded good in the code but described as bad in the project notes",
       "The code treats In Grace Period as good while Project_Notes describes it as bad, so the documented and implemented definitions disagree.",
       "3,146 loans carry In Grace Period status (0.7% of 466,285). The In Grace bad variant is in sensitivity.csv.",
       "sensitivity.csv", "Medium", "Amber",
       "Fix one definition, document it in the model document, and keep the alternative as a sensitivity.", "Model Developer"),
    _f("Reproducibility", "pd_model.sav loaded without fitting code; missing dummies silently zero-filled",
       "The Final notebook loads a pickled PD model with no cell that fits it, so results cannot be regenerated. Dummy columns absent from the scoring frame "
       "are filled with zero without a warning, for example mths_since_last_record:>=86 versus >86 label drift.",
       "No fitting code for pd_model.sav in the Final notebook. align_columns in this project logs the count of missing columns as a warning.",
       "", "Medium", "Amber",
       "Keep fitting code under version control, fail or warn on missing design columns, and store the training sample hash.", "Model Developer"),
    _f("Data quality", "Immature recoveries on recent defaults",
       "Loans that charged off in 2013-2014 have had little time to recover, so recovery rates and LGD fitted on them are biased upward in loss.",
       "Share of defaults with any recovery is 74-77% for 2010-2012 defaults, 57% for 2013 and 37% for 2014. 2014 is reported but not fitted.",
       "recovery_by_default_vintage.csv", "Medium", "Amber",
       "Fit LGD on mature default cohorts only and apply a development-pattern adjustment for recent cohorts.", "Model Developer"),
    _f("Data quality", "Structural breaks in input features",
       "Some inputs did not exist or were not populated before 2012, so the development period mixes two data regimes.",
       "initial_list_status 'w' is 0% before 2012; total_rev_hi_lim, tot_cur_bal and tot_coll_amt are missing before 2012.",
       "psi_csi.csv", "Medium", "Amber",
       "Exclude or restrict features with regime changes, or fit on the post-break period, and monitor CSI on them.", "Model Developer"),
    _f("Reporting", "EAD summary prints the LGD stage 2 p-values",
       "The EAD regression summary cell prints the coefficient table of the LGD stage 2 model, so the reported EAD statistics are not those of the EAD model.",
       "Original notebook cell 113 prints the LGD stage 2 p-values. The EAD coefficients are re-estimated in ead_coefs.csv.",
       "ead_coefs.csv", "Low", "Amber",
       "Correct the cell reference and re-issue the EAD statistics.", "Model Developer"),
    _f("Data quality", "Two-digit year in earliest_cr_line",
       "Dates such as Jan-62 parse as 2062, producing negative credit age unless corrected.",
       "1,169 earliest_cr_line dates parse into the future and are corrected by subtracting 100 years when after the issue date.",
       "", "Low", "Green",
       "Keep the correction in the data layer and add a unit test.", "Model Developer"),
]


def _worst(group: pd.DataFrame) -> pd.Series:
    rank = group[LIGHT_COL].map(LIGHT_RANK)
    return group.loc[rank.idxmax()]


def _area_for(test: str) -> str:
    tokens = "".join(ch if ch.isalnum() else " " for ch in test.lower()).split()
    for words, area in AREA_KEYWORDS:
        if any(t == w or (len(w) > 3 and t.startswith(w)) for t in tokens for w in words):
            return area
    return "Performance"


def _fmt(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    try:
        return f"{float(x):.4g}"
    except (TypeError, ValueError):
        return str(x) or "n/a"


def _evidence_file(test: str, model, split, out_dir) -> str:
    for prefix, pattern in EVIDENCE_FILE_BY_PREFIX:
        if str(test).startswith(prefix):
            name = pattern.format(model=model, split=split)
            if out_dir is None or (Path(out_dir) / name).exists():
                return name
    return RESULTS_FILE


def auto_findings(results: pd.DataFrame, out_dir=None) -> pd.DataFrame:
    """One finding per (test, model, split) whose worst light is Red (High) or Amber (Medium)."""
    cols = [c for c in LOG_COLUMNS if c != "finding_id"]
    if results is None or results.empty or LIGHT_COL not in results.columns:
        return pd.DataFrame(columns=cols)
    r = results.copy()
    for c in (MODEL_COL, SPLIT_COL, THRESHOLD_COL, VALUE_COL):
        if c not in r.columns:
            r[c] = np.nan
    if EVIDENCE_FILE_COL not in r.columns:
        r[EVIDENCE_FILE_COL] = ""
    r[LIGHT_COL] = r[LIGHT_COL].astype(str)
    r = r[r[LIGHT_COL].isin(LIGHT_RANK)]
    rows = []
    for _, g in r.groupby([TEST_COL, MODEL_COL], dropna=False, sort=False):
        g = g[g[LIGHT_COL].isin(LIGHT_TO_SEVERITY)]
        if g.empty:
            continue
        w = _worst(g)
        who = " ".join(str(x) for x in (w[MODEL_COL],) if pd.notna(x) and str(x))
        splits_txt = ", ".join(f"{s}: {_fmt(v)} ({li})" for s, v, li in zip(g[SPLIT_COL], g[VALUE_COL], g[LIGHT_COL]))
        title_split = f" {w[SPLIT_COL]}" if len(g) == 1 and pd.notna(w[SPLIT_COL]) else (f" ({len(g)} samples)" if len(g) > 1 else "")
        area = _area_for(str(w[TEST_COL]))
        given = str(w[EVIDENCE_FILE_COL])
        ef = given if given not in ("", "nan") else _evidence_file(w[TEST_COL], w[MODEL_COL], w[SPLIT_COL], out_dir)
        note = HL_POWER_NOTE if str(w[TEST_COL]).startswith("hl_") else ""
        rows.append(dict(
            area=area, title=f"{w[TEST_COL]} {w[LIGHT_COL]} for {who or 'all data'}{title_split}",
            description=f"Automated validation test {w[TEST_COL]} returned {w[LIGHT_COL]}.",
            evidence=f"{w[TEST_COL]} against threshold {_fmt(w[THRESHOLD_COL])} for {who or 'all data'}; {splits_txt}.{note}",
            evidence_file=ef, severity=LIGHT_TO_SEVERITY[w[LIGHT_COL]], traffic_light=w[LIGHT_COL],
            recommendation=RECOMMENDATION_BY_AREA.get(area, RECOMMENDATION_BY_AREA["Performance"]),
            owner="Model Developer", status="Open", source="auto"))
    return pd.DataFrame(rows, columns=cols)


def _el_pick_12m(d: pd.DataFrame) -> pd.DataFrame:
    d = d[(d["grade"] == EL_ALL_LABEL) & (d["sample"] == EL_OOT_SAMPLE) & (d["status"].fillna("") == "")]
    return d[d["vintage"] == d["vintage"].min()]


def _el_pick_life(d: pd.DataFrame) -> pd.DataFrame:
    return d[(d["grade"] == EL_ALL_LABEL) & (d["sample"] == EL_LIFE_TEST_SAMPLE)]


def el_findings(out_dir) -> pd.DataFrame:
    """Auto findings from the expected-loss back-tests: gap between predicted EL and realized loss."""
    out_dir = Path(out_dir)
    cols = [c for c in LOG_COLUMNS if c != "finding_id"]
    rows = []
    for fname, label, pick in ((EL_12M_FILE, "12-month EL, first fully resolved out-of-time vintage", _el_pick_12m),
                               (EL_LIFE_FILE, "lifetime EL, 2012 test cohort", _el_pick_life)):
        p = out_dir / fname
        if not p.exists():
            continue
        try:
            row = pick(pd.read_csv(p))
            el, real = float(row["el"].iloc[0]), float(row["realized_loss"].iloc[0])
        except Exception as exc:
            log.warning("cannot evaluate %s: %s", fname, exc)
            continue
        if not real:
            continue
        gap = abs(el / real - 1.0)
        light = "Green" if gap <= EL_GAP_BANDS[0] else ("Amber" if gap <= EL_GAP_BANDS[1] else "Red")
        if light == "Green":
            continue
        rows.append(dict(
            area="Loss estimation", title=f"Expected loss versus realized loss gap: {label}",
            description=f"Predicted {label} differs from realized net loss by more than {EL_GAP_BANDS[0]:.0%}.",
            evidence=(f"Predicted EL ${el:,.0f} against realized ${real:,.0f}, ratio {el / real:.3f}, gap {gap:.1%} "
                      f"(Amber above {EL_GAP_BANDS[0]:.0%}, Red above {EL_GAP_BANDS[1]:.0%}): {light}."),
            evidence_file=fname, severity=LIGHT_TO_SEVERITY[light], traffic_light=light,
            recommendation=RECOMMENDATION_BY_AREA["Loss estimation"], owner="Model Developer", status="Open", source="auto"))
    return pd.DataFrame(rows, columns=cols)


def load_results(out_dir=config.OUT_DIR) -> pd.DataFrame:
    """validation_results.csv; it already carries the benchmark, PSI/CSI and calibration rows with their lights."""
    p = Path(out_dir) / RESULTS_FILE
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def scope_results(res: pd.DataFrame) -> pd.DataFrame:
    """Keep rows for the validated scorecards and benchmark, excluding in-sample splits."""
    if res.empty:
        return res
    return res[res[MODEL_COL].isin(AUTO_SCOPE_MODELS) & ~res[SPLIT_COL].isin(AUTO_SKIP_SPLITS)]


def _val(res, test, model, split):
    if res.empty:
        return None
    m = res[(res[TEST_COL] == test) & (res[MODEL_COL] == model) & (res[SPLIT_COL] == split)]
    return float(m[VALUE_COL].iloc[0]) if len(m) else None


def _extras(out_dir: Path, res: pd.DataFrame) -> dict:
    """Extra evidence sentences keyed by finding, built only from numbers that exist in the outputs."""
    ex = {}
    g_repro, g_oot, g_dev = (_val(res, "gini", "repro_original", "repro_test"), _val(res, "gini", "sc_full", "oot1"),
                             _val(res, "gini", "sc_full", "dev_holdout"))
    if g_repro is not None:
        ex["mths"] = f" Computed here: reproduced original model (including mths_since_issue_d) has Gini {g_repro:.3f} on its random test split."
    if g_repro is not None and g_oot is not None:
        ex["random"] = (f" Computed here: random-split Gini of the reproduced model {g_repro:.3f}; time-ordered OOT 2013 Gini of the scorecard {g_oot:.3f}"
                        + (f" (development holdout {g_dev:.3f})." if g_dev is not None else "."))
    g_indep = _val(res, "gini", "sc_indep", "oot1")
    if g_indep is not None and g_oot is not None:
        ex["circ"] = f" Computed here: OOT 2013 Gini {g_oot:.3f} with LendingClub risk outputs versus {g_indep:.3f} without."
    p = out_dir / "by_vintage.csv"
    if p.exists():
        v = pd.read_csv(p)
        v = v[v["model"] == CHAMPION] if "model" in v.columns else v
        v = v[(v["vintage"] >= 2010) & (v["vintage"] <= 2014)]
        if len(v):
            ex["cens"] = f" Computed here: fixed 12-month bad rate by vintage 2010-2014 ranges from {v['bad_rate'].min():.2%} to {v['bad_rate'].max():.2%}."
    p = out_dir / EL_LIFE_FILE
    if p.exists():
        t = _el_pick_life(pd.read_csv(p))
        if len(t) and "realized_pct_funded" in t.columns:
            ex["life"] = (f" Computed here on the complete 36-month cohort (2012 test): predicted EL {float(t['el_pct_funded'].iloc[0]):.2%} of funded "
                          f"against realized {float(t['realized_pct_funded'].iloc[0]):.2%}.")
    p = out_dir / "lgd_ead_summary.json"
    if p.exists():
        sm = json.loads(p.read_text())
        exp, hard = sm.get("lgd_expected_r2", {}), sm.get("lgd_hard_r2", {})
        k = next((x for x in exp if "oot" in x.lower() and x in hard), None)
        if k is not None:
            ex["lgd"] = f" Computed here on {k} defaults: LGD R-squared {exp[k]:.3f} with the expected-value combination versus {hard[k]:.3f} with hard 0/1 classes."
    p = out_dir / "sensitivity.csv"
    if p.exists():
        s = pd.read_csv(p)
        gr = s[s["scenario"].astype(str).str.contains("grace", case=False)]
        if len(gr) and gr["delta_pd_pct"].notna().any():
            ex["grace"] = f" Computed here: coding In Grace Period as bad changes mean PD by {float(gr['delta_pd_pct'].dropna().iloc[0]):+.1f}%."
    return ex


def _review_frame(out_dir: Path, res: pd.DataFrame) -> pd.DataFrame:
    ex = _extras(out_dir, res)
    rows = []
    for f in REVIEW_FINDINGS:
        f = dict(f)
        key = next((k for t, k in EXTRA_KEY_BY_TITLE_START.items() if f["title"].startswith(t)), None)
        if key in ex:
            f["evidence"] += ex[key]
        rows.append(f)
    df = pd.DataFrame(rows)
    df["status"], df["source"] = "Open", "review"
    return df


def build_log(out_dir=config.OUT_DIR) -> pd.DataFrame:
    """Merge review and auto findings, assign F-nn ids, sort by severity then id, write findings_log.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = load_results(out_dir)
    auto = pd.concat([auto_findings(scope_results(res), out_dir), el_findings(out_dir)], ignore_index=True)
    log_df = pd.concat([_review_frame(out_dir, res), auto], ignore_index=True)
    width = max(2, len(str(len(log_df))))
    log_df.insert(0, "finding_id", [f"F-{i + 1:0{width}d}" for i in range(len(log_df))])
    log_df["_s"] = log_df["severity"].map(SEVERITY_ORDER)
    log_df = log_df.sort_values(["_s", "finding_id"], kind="stable").drop(columns="_s").reset_index(drop=True)
    log_df = log_df[list(LOG_COLUMNS)]
    log_df.to_csv(out_dir / LOG_FILE, index=False)
    return log_df
