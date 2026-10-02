"""Benchmarks: LendingClub's own sub_grade and int_rate as scores, incremental Gini, challengers vs champion."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config
from . import columns as C
from . import validation as V

# ===== CONFIG (user inputs) =====
CHAMPION = "sc_full"
SCORECARD_FOR_CIRCULARITY, FULL_SCORECARD = "sc_indep", "sc_full"
BENCH_SPLITS = (C.DEV_HOLDOUT, C.OOT1, C.OOT2)
CI_SPLITS = (C.OOT1, C.OOT2)
SIGNIFICANCE = 0.05
BENCH_SUBGRADE, BENCH_GRADE, BENCH_INT_RATE, BENCH_COMBINED = "sub_grade_rank", "grade_rank", "int_rate", "sc_indep+sub_grade"
GRADE_LADDER = tuple("ABCDEFG")
SUBGRADE_LADDER = tuple(f"{g}{k}" for g in GRADE_LADDER for k in range(1, 6))
BENCH_COLUMNS = ["kind", "split", "model_a", "model_b", "auc_a", "auc_b", "gini_a", "gini_b", "d_auc", "delong_z", "delong_p", "ci_lo", "ci_hi", "light"]
# ===== END CONFIG =====


def ordinal(s: pd.Series) -> np.ndarray:
    """Rank of a grade label on a fixed ladder (A1 < ... < G5; A < ... < G) so every split uses the same scale."""
    ladder = SUBGRADE_LADDER if s.astype(str).str.len().max() > 1 else GRADE_LADDER
    return s.astype(str).map({c: i for i, c in enumerate(ladder)}).to_numpy(dtype=np.float64)


def beats_light(d_auc, p) -> str:
    if np.isnan(d_auc):
        return V.LIGHT_NA
    if d_auc <= 0:
        return "Red"
    return "Green" if (not np.isnan(p) and p < SIGNIFICANCE) else "Amber"


def _bench_scores(df: pd.DataFrame) -> dict:
    return {BENCH_SUBGRADE: ordinal(df[C.SUB_GRADE]), BENCH_GRADE: ordinal(df[C.GRADE]), BENCH_INT_RATE: df[C.INT_RATE].to_numpy(dtype=np.float64)}


def _row(kind, split, a, b, y, pa, pb, n_boot, ci=False):
    d, z, p = V.delong_test(y, pa, pb)
    lo = hi = np.nan
    if ci:
        lo, hi, _ = V.bootstrap_diff_ci(V.auc, y, pa, pb, n_boot=n_boot)
    return dict(kind=kind, split=split, model_a=a, model_b=b, auc_a=V.auc(y, pa), auc_b=V.auc(y, pb), gini_a=V.gini(y, pa),
                gini_b=V.gini(y, pb), d_auc=d, delong_z=z, delong_p=p, ci_lo=lo, ci_hi=hi, light="")


def combined_model(train: pd.DataFrame, sc_train_margin: np.ndarray):
    """Logit of the bad flag on sub_grade rank and the SC_INDEP margin, fit on development data only."""
    X = sm.add_constant(np.column_stack([ordinal(train[C.SUB_GRADE]), sc_train_margin]))
    return sm.Logit(train[C.TARGET_BAD12].to_numpy(dtype=np.float64), X).fit(disp=0, maxiter=100)


def run_benchmark(bundles: dict, splits: dict, scores: dict = None, n_boot=None):
    scores = scores or V.score_all(bundles, splits)
    rows, tests = [], []
    for s in BENCH_SPLITS:
        d = splits[s]
        y = d[C.TARGET_BAD12].to_numpy(dtype=np.float64)
        for nm, v in _bench_scores(d).items():
            rows.append(dict(kind="standalone", split=s, model_a=nm, model_b="", auc_a=V.auc(y, v), auc_b=np.nan, gini_a=V.gini(y, v),
                             gini_b=np.nan, d_auc=np.nan, delong_z=np.nan, delong_p=np.nan, ci_lo=np.nan, ci_hi=np.nan, light=""))
            tests.append(V.result(f"benchmark_auc_{nm}", "benchmark", s, V.auc(y, v)))
    combined = None
    if SCORECARD_FOR_CIRCULARITY in bundles:
        combined = combined_model(splits[C.DEV_TRAIN], bundles[SCORECARD_FOR_CIRCULARITY].predict_margin(splits[C.DEV_TRAIN]))
    for s in (C.OOT1, C.OOT2):
        d = splits[s]
        y = d[C.TARGET_BAD12].to_numpy(dtype=np.float64)
        sg = ordinal(d[C.SUB_GRADE])
        for nm in (SCORECARD_FOR_CIRCULARITY, FULL_SCORECARD):
            if (nm, s) not in scores:
                continue
            r = _row("vs_sub_grade", s, nm, BENCH_SUBGRADE, y, scores[(nm, s)], sg, n_boot, ci=True)
            r["light"] = beats_light(r["d_auc"], r["delong_p"])
            rows.append(r)
            tests.append(V.result(f"benchmark_{nm}_vs_sub_grade_dauc", nm, s, r["d_auc"], light=r["light"], key="gini"))
            tests[-1]["threshold"] = "Green: beats sub_grade with DeLong p < 0.05; Amber: ahead but not significant; Red: does not beat"
            tests.append(V.result(f"benchmark_{nm}_vs_sub_grade_delong_p", nm, s, r["delong_p"]))
        if combined is not None:
            m = bundles[SCORECARD_FOR_CIRCULARITY].predict_margin(d)
            pc = combined.predict(sm.add_constant(np.column_stack([sg, m])))
            r = _row("incremental_over_sub_grade", s, BENCH_COMBINED, BENCH_SUBGRADE, y, pc, sg, n_boot, ci=True)
            r["light"] = V.LIGHT_INFO
            rows.append(r)
            tests += [V.result("incremental_gini_sc_indep_over_sub_grade", SCORECARD_FOR_CIRCULARITY, s, r["gini_a"] - r["gini_b"]),
                      V.result("incremental_gini_delong_p", SCORECARD_FOR_CIRCULARITY, s, r["delong_p"])]
        base = (CHAMPION, s)
        if base not in scores:
            continue
        for nm in bundles:
            if nm == CHAMPION or (nm, s) not in scores:
                continue
            r = _row("challenger_vs_champion", s, nm, CHAMPION, y, scores[(nm, s)], scores[base], n_boot, ci=True)
            r["light"] = V.LIGHT_INFO
            rows.append(r)
            tests += [V.result(f"challenger_dauc_vs_{CHAMPION}", nm, s, r["d_auc"]), V.result(f"challenger_delong_p_vs_{CHAMPION}", nm, s, r["delong_p"])]
    return pd.DataFrame(rows)[BENCH_COLUMNS], tests
