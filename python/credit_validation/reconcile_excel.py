"""Independent Python recomputation of the Excel workbook, LibreOffice recalculation and comparison.

Run: py -3 -m credit_validation.reconcile_excel [--quick]   (needs LibreOffice for the recalculation step)
"""
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import openpyxl
import pandas as pd
from scipy import stats

from . import config
from . import columns as C
from . import stability
from . import validation as V

# ===== CONFIG (user inputs) =====
EXCEL_DIR = config.ROOT / "excel"
BUILDER = EXCEL_DIR / "build_workbook.py"
XLSX = EXCEL_DIR / "credit_validation_workbook.xlsx"
REPORT = config.OUT_DIR / "excel_reconciliation.json"
SOFFICE_CANDIDATES = [r"C:\Program Files\LibreOffice\program\soffice.exe",
                      r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"]
AUTHOR = "Deepak Chaudhary"
MODEL_NAME = "sc_full"
TOL_PD, TOL_LOGIT, TOL_POINTS, TOL_COUNT = 1e-9, 1e-9, 1e-6, 0
TOL_KS = TOL_PSI = TOL_RATE = 1e-9
TOL_HL_STAT, TOL_PVALUE, TOL_EL_USD = 1e-6, 1e-8, 0.01
QUICK_LOANS = 1_000
LO_TIMEOUT_S = 1800
N_DECILES = V.N_BINS
SAMPLE_PATH = config.EXCEL_INPUT_DIR / "loan_sample.csv"
PSI_BINS_PATH = config.EXCEL_INPUT_DIR / "dev_psi_bins.csv"
REALIZED_PATH = config.EXCEL_INPUT_DIR / "realized_loss_by_grade.csv"
POINTS_PATH = config.OUT_DIR / "scorecard_points.csv"
RESULTS_PATH = config.OUT_DIR / "validation_results.csv"
MODEL_PATH = config.MODEL_DIR / f"{MODEL_NAME}.joblib"
REALIZED_COL, PTS_PREFIX = "realized_loss_12m", "pts_"          # names written by expected_loss.write_excel_inputs
MODEL_COL, VAR_COL, BIN_COL, LO_COL, HI_COL = "model", "variable", "bin_id", "lo", "hi"
CAT_COL, WOE_COL, COEF_COL, POINTS_COL = "categories", "woe", "coef", "points"
PSI_LO_COL, PSI_HI_COL, PSI_SHARE_COL, PSI_COUNT_COL = "lo", "hi", "dev_share", "dev_count"
CAT_JOIN = "|"
CALC_DTI_SHOCK = config.SENSITIVITY_SHOCKS["dti_pts"]
# ===== END CONFIG =====


# ---------------------------------------------------------------- inputs
def load_sample(quick: bool = False) -> pd.DataFrame:
    s = pd.read_csv(SAMPLE_PATH, float_precision="round_trip", keep_default_na=False, na_values=[""])
    if quick and len(s) > QUICK_LOANS:
        s = s.sample(n=QUICK_LOANS, random_state=config.SEED).sort_index().reset_index(drop=True)
    return s


def load_tables(quick: bool = False):
    sample = load_sample(quick)
    pts = pd.read_csv(POINTS_PATH)
    pts = pts[pts[MODEL_COL] == MODEL_NAME].reset_index(drop=True)
    psi_bins = pd.read_csv(PSI_BINS_PATH)
    realized = pd.read_csv(REALIZED_PATH)
    return sample, pts, psi_bins, realized


def load_scorecard():
    return joblib.load(MODEL_PATH).scorecard


# ---------------------------------------------------------------- python reference
def desc_rank(p) -> np.ndarray:
    order = np.argsort(-np.asarray(p, float), kind="stable")
    r = np.empty(len(p), dtype=np.int64)
    r[order] = np.arange(1, len(p) + 1)
    return r


def asc_rank(p) -> np.ndarray:
    order = np.argsort(np.asarray(p, float), kind="stable")
    r = np.empty(len(p), dtype=np.int64)
    r[order] = np.arange(1, len(p) + 1)
    return r


def decile_of(rank, n_dec=N_DECILES) -> np.ndarray:
    return ((rank - 1) * n_dec) // len(rank) + 1


def score_frame(frame: pd.DataFrame, sc) -> dict:
    """Bin position (1-based), WoE, per-variable points, margin, PD and score from the project scorecard."""
    k = len(sc.variables)
    pos, woe, pts = {}, {}, {}
    for v in sc.variables:
        idx = sc.specs[v].bin_index(frame[v])
        pos[v] = idx + 1
        woe[v] = sc.specs[v].woe_all[idx]
        pts[v] = -(sc.coefs[v] * woe[v] + sc.intercept / k) * sc.factor + sc.offset / k
    margin = sc.margin(frame)
    return dict(pos=pos, woe=woe, pts=pts, margin=margin, pd=1.0 / (1.0 + np.exp(-margin)), score=sc.points(frame))


def sample_reference(sample: pd.DataFrame, sc, psi_bins: pd.DataFrame) -> dict:
    """Independent metrics on the sample via the project validation and stability modules."""
    s = score_frame(sample, sc)
    y = sample[C.TARGET_BAD12].to_numpy(dtype=np.float64)
    pdv = s["pd"]
    n = len(y)
    rank = desc_rank(pdv)
    dec_id = decile_of(rank)
    cal_id = decile_of(asc_rank(pdv))
    dec = V.decile_table(y, pdv)
    cum_b, cum_g = dec["cum_bad_share"].to_numpy(), dec["cum_good_share"].to_numpy()
    xs, ys = np.r_[0.0, cum_g], np.r_[0.0, cum_b]
    auc_approx = float(np.sum(np.diff(xs) * (ys[1:] + ys[:-1]) / 2.0))
    hl = V.hosmer_lemeshow(y, pdv, g=N_DECILES, in_sample=False)
    grade = V.binomial_test_by_group(y, pdv, sample[C.GRADE].astype(str))
    ct = V.central_tendency(y, pdv)
    edges = psi_bins[PSI_HI_COL].to_numpy()[:-1]
    sh = stability.shares(s["score"], edges)
    dev_share = psi_bins[PSI_SHARE_COL].to_numpy()
    floor = stability.PSI_FLOOR
    e, a = np.maximum(dev_share, floor), np.maximum(sh, floor)
    psi_contrib = (a - e) * np.log(a / e)
    slope = V.calibration_slope_intercept(y, pdv)
    el = pdv * sample[C.PRED_LGD].to_numpy(float) * sample[C.PRED_EAD].to_numpy(float)
    return dict(
        n=n, bads=float(y.sum()), bad_rate=float(y.mean()), mean_pd=float(pdv.mean()), brier=V.brier(y, pdv),
        brier_skill=V.brier_skill(y, pdv), auc=V.auc(y, pdv), gini=V.gini(y, pdv), ks=V.ks(y, pdv), dec=dec,
        ks_decile=float(dec["ks_at_decile"].max()), auc_approx=auc_approx, gini_approx=2 * auc_approx - 1,
        rank=rank, dec_id=dec_id, cal_id=cal_id, hl=hl, grade=grade, central=ct, psi_shares=sh, psi_dev=dev_share,
        psi_contrib=psi_contrib, psi_total=float(psi_contrib.sum()), slope=slope, scored=s, el=el, y=y)


# ---------------------------------------------------------------- LibreOffice
def find_soffice():
    for c in SOFFICE_CANDIDATES:
        if Path(c).exists():
            return c
    return None


def load_builder():
    spec = importlib.util.spec_from_file_location("credit_build_workbook", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def recalc(src: Path, outdir: Path, soffice: str) -> Path:
    shutil.rmtree(outdir, ignore_errors=True)
    subprocess.run([soffice, "--headless", "--calc", "--convert-to", "xlsx", "--outdir", str(outdir), str(src)],
                   check=True, capture_output=True, timeout=LO_TIMEOUT_S)
    return outdir / src.name


def patch_author(path: Path, author: str = AUTHOR):
    tmp = path.with_suffix(".tmp.xlsx")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "docProps/core.xml":
                x = data.decode("utf-8")
                for tag in ("dc:creator", "cp:lastModifiedBy"):
                    if re.search(rf"<{tag}>.*?</{tag}>", x, flags=re.S):
                        x = re.sub(rf"<{tag}>.*?</{tag}>", f"<{tag}>{author}</{tag}>", x, flags=re.S)
                    elif re.search(rf"<{tag}\s*/>", x):
                        x = re.sub(rf"<{tag}\s*/>", f"<{tag}>{author}</{tag}>", x)
                    else:
                        x = x.replace("</cp:coreProperties>", f"<{tag}>{author}</{tag}></cp:coreProperties>")
                data = x.encode("utf-8")
            zout.writestr(item, data)
    tmp.replace(path)


# ---------------------------------------------------------------- comparison
def _col(ws, letter, first, last):
    return np.array([ws[f"{letter}{r}"].value for r in range(first, last + 1)], dtype=object)


def _num(a):
    return np.array([np.nan if v is None or isinstance(v, str) else float(v) for v in a], dtype=float)


def calc_reference(sc, inputs: dict) -> dict:
    frame = pd.DataFrame({v: [inputs.get(v)] for v in sc.variables})
    for v in sc.variables:
        if sc.specs[v].kind == "numeric":
            frame[v] = pd.to_numeric(frame[v]).astype(np.float64)
    s = score_frame(frame, sc)
    return dict(pd=float(s["pd"][0]), score=float(s["score"][0]), margin=float(s["margin"][0]),
                pos={v: int(s["pos"][v][0]) for v in sc.variables}, woe={v: float(s["woe"][v][0]) for v in sc.variables},
                pts={v: float(s["pts"][v][0]) for v in sc.variables})


def compare(label, wbv, meta, ref, sample, sc, inputs, realized) -> dict:
    a = meta["addr"]
    ls = wbv["Loan_Sample"]
    first, last = meta["first"], meta["last"]
    cols = meta["ls_cols"]
    issues, worst = [], {}

    def track(key, diff, tol):
        diff = float(diff)
        worst[key] = max(worst.get(key, 0.0), diff if np.isfinite(diff) else np.inf)
        if not diff <= tol:
            issues.append(f"{key}: diff {diff:.3e} > {tol:.0e}")

    def cell(name):
        sheet, ref_ = a[name].split("!")
        return wbv[sheet][ref_.replace("$", "")].value

    def lscol(h):
        return _num(_col(ls, cols[h], first, last))

    sc_ = ref["scored"]
    track("loan_pd", np.nanmax(np.abs(lscol("xl_pd") - sc_["pd"])), TOL_PD)
    track("loan_logit", np.nanmax(np.abs(lscol("xl_margin") - sc_["margin"])), TOL_LOGIT)
    track("loan_score", np.nanmax(np.abs(lscol("xl_score") - sc_["score"])), TOL_POINTS)
    for v in sc.variables:
        track("loan_bin_index", np.nanmax(np.abs(lscol(f"xl_bin_{v}") - sc_["pos"][v])), TOL_COUNT)
        track("loan_woe", np.nanmax(np.abs(lscol(f"xl_woe_{v}") - sc_["woe"][v])), TOL_LOGIT)
        track("loan_points", np.nanmax(np.abs(lscol(f"xl_pts_{v}") - sc_["pts"][v])), TOL_POINTS)
    track("vs_exported_python_pd", np.nanmax(np.abs(lscol("xl_pd") - sample[C.PRED_PD].to_numpy(float))), TOL_PD)
    track("vs_exported_python_score", np.nanmax(np.abs(lscol("xl_score") - sample[C.SCORE].to_numpy(float))), TOL_POINTS)
    for v in sc.variables:
        track("vs_exported_python_points", np.nanmax(np.abs(lscol(f"xl_pts_{v}") - sample[PTS_PREFIX + v].to_numpy(float))), TOL_POINTS)
    track("loan_rank", np.nanmax(np.abs(lscol("xl_rank_desc") - ref["rank"])), TOL_COUNT)
    track("loan_decile", np.nanmax(np.abs(lscol("xl_decile") - ref["dec_id"])), TOL_COUNT)
    track("loan_hl_bin", np.nanmax(np.abs(lscol("xl_cal_bin") - ref["cal_id"])), TOL_COUNT)
    el_live = lscol("xl_el")
    track("loan_el_usd", np.nanmax(np.abs(el_live - ref["el"])), TOL_EL_USD)
    track("total_el_usd", abs(np.nansum(el_live) - ref["el"].sum()), TOL_EL_USD)

    d = wbv["Deciles"]
    dec = ref["dec"]
    d0 = meta["deciles"]["first"]
    rows = range(d0, d0 + N_DECILES)

    def dcol(letter):
        return _num([d[f"{letter}{r}"].value for r in rows])

    track("decile_counts", np.abs(dcol("B") - dec["n"].to_numpy(float)).max(), TOL_COUNT)
    track("decile_bads", np.abs(dcol("C") - dec["bads"].to_numpy(float)).max(), TOL_COUNT)
    track("decile_bad_rate", np.abs(dcol("D") - dec["bad_rate"].to_numpy()).max(), TOL_RATE)
    track("decile_mean_pd", np.abs(dcol("E") - dec["mean_pd"].to_numpy()).max(), TOL_PD)
    track("decile_cum_bad_share", np.abs(dcol("G") - dec["cum_bad_share"].to_numpy()).max(), TOL_RATE)
    track("decile_cum_good_share", np.abs(dcol("H") - dec["cum_good_share"].to_numpy()).max(), TOL_RATE)
    track("decile_ks", np.abs(dcol("I") - dec["ks_at_decile"].to_numpy()).max(), TOL_KS)
    track("decile_lift", np.abs(dcol("J") - dec["lift"].to_numpy()).max(), TOL_RATE)
    track("decile_cum_lift", np.abs(dcol("K") - dec["cum_lift"].to_numpy()).max(), TOL_RATE)
    track("ks_max", abs(cell("ks_max") - ref["ks_decile"]), TOL_KS)
    track("gini_approx", abs(cell("gini_approx") - ref["gini_approx"]), TOL_RATE)

    ca = wbv["Calibration"]
    h0 = meta["hl"]["first"]
    hrows = range(h0, h0 + N_DECILES)
    ct = V.calibration_table(ref["y"], ref["scored"]["pd"])
    track("hl_counts", np.abs(_num([ca[f"B{r}"].value for r in hrows]) - ct["n"].to_numpy(float)).max(), TOL_COUNT)
    track("hl_bads", np.abs(_num([ca[f"E{r}"].value for r in hrows]) - ct["bads"].to_numpy(float)).max(), TOL_COUNT)
    track("hl_mean_pd", np.abs(_num([ca[f"C{r}"].value for r in hrows]) - ct["mean_pd"].to_numpy()).max(), TOL_PD)
    track("hl_statistic", abs(cell("hl_stat") - ref["hl"]["stat"]), TOL_HL_STAT)
    track("hl_p_value", abs(cell("hl_p") - ref["hl"]["p_value"]), TOL_PVALUE)
    track("hl_df", abs(cell("hl_df") - ref["hl"]["df"]), TOL_COUNT)
    track("brier", abs(cell("brier") - ref["brier"]), TOL_RATE)
    track("brier_skill", abs(cell("brier_skill") - ref["brier_skill"]), TOL_RATE)
    track("mean_pd", abs(cell("mean_pd") - ref["central"]["mean_pd"]), TOL_PD)
    track("obs_rate", abs(cell("obs_rate") - ref["central"]["obs_rate"]), TOL_RATE)
    track("central_tendency_ratio", abs(cell("ct_ratio") - ref["central"]["ratio"]), TOL_RATE)
    track("central_tendency_reldev", abs(cell("ct_rel_dev") - ref["central"]["rel_dev"]), TOL_RATE)
    g0 = meta["grades"]["first"]
    gr = ref["grade"].set_index("group")
    for i, g in enumerate(meta["grades"]["labels"]):
        r = g0 + i
        row = gr.loc[g]
        track("grade_n", abs(ca[f"B{r}"].value - row["n"]), TOL_COUNT)
        track("grade_bads", abs(ca[f"C{r}"].value - row["bads"]), TOL_COUNT)
        track("grade_mean_pd", abs(ca[f"E{r}"].value - row["mean_pd"]), TOL_PD)
        track("grade_binomial_z", abs(ca[f"F{r}"].value - row["z"]), TOL_RATE)
        track("grade_p_z", abs(ca[f"G{r}"].value - row["p_z"]), TOL_PVALUE)
        track("grade_p_jeffreys", abs(ca[f"H{r}"].value - row["p_jeffreys"]), TOL_PVALUE)

    ps = wbv["PSI"]
    p0 = meta["psi"]["first"]
    prow = range(p0, p0 + len(ref["psi_shares"]))
    track("psi_oot_counts", np.abs(_num([ps[f"H{r}"].value for r in prow]) - ref["psi_shares"] * ref["n"]).max(), TOL_COUNT + 1e-6)
    track("psi_oot_share", np.abs(_num([ps[f"I{r}"].value for r in prow]) - ref["psi_shares"]).max(), TOL_PSI)
    track("psi_contrib", np.abs(_num([ps[f"L{r}"].value for r in prow]) - ref["psi_contrib"]).max(), TOL_PSI)
    track("psi_total", abs(cell("psi_total") - ref["psi_total"]), TOL_PSI)

    el = wbv["EL"]
    e0 = meta["el"]["first"]
    grp = pd.DataFrame({C.GRADE: sample[C.GRADE].astype(str), "el": ref["el"], "real": sample[REALIZED_COL].to_numpy(float),
                        "n": 1, "funded": sample[C.FUNDED_AMNT].to_numpy(float)}).groupby(C.GRADE).sum()
    for i, g in enumerate(meta["el"]["labels"]):
        r = e0 + i
        track("el_grade_n", abs(el[f"B{r}"].value - grp.loc[g, "n"]), TOL_COUNT)
        track("el_grade_usd", abs(el[f"E{r}"].value - grp.loc[g, "el"]), TOL_EL_USD)
        track("el_grade_realized_usd", abs(el[f"F{r}"].value - grp.loc[g, "real"]), TOL_EL_USD)
    track("el_total_usd", abs(cell("el_total") - grp["el"].sum()), TOL_EL_USD)

    cr = calc_reference(sc, inputs)
    track("calc_pd", abs(cell("calc_pd") - cr["pd"]), TOL_PD)
    track("calc_logit", abs(cell("calc_margin") - cr["margin"]), TOL_LOGIT)
    track("calc_score", abs(cell("calc_score") - cr["score"]), TOL_POINTS)
    track("calc_points_sum", abs(cell("calc_points_sum") - cr["score"]), TOL_POINTS)
    sc_calc = wbv["Score_Calc"]
    c0 = meta["calc"]["first"]
    for i, v in enumerate(sc.variables):
        r = c0 + i
        track("calc_bin_index", abs(sc_calc[f"C{r}"].value - cr["pos"][v]), TOL_COUNT)
        track("calc_woe", abs(sc_calc[f"D{r}"].value - cr["woe"][v]), TOL_LOGIT)
        track("calc_points", abs(sc_calc[f"G{r}"].value - cr["pts"][v]), TOL_POINTS)
    if cell("checks_all") != 1:
        issues.append("workbook Checks sheet does not report all pass")
    return dict(case=label, passed=not issues, worst_diffs=worst, issues=issues, calc_inputs=inputs,
                calc_pd=cr["pd"], calc_score=cr["score"])


# ---------------------------------------------------------------- driver
def build_and_reconcile(quick: bool = False, xlsx_out=None, report_out=None) -> dict:
    t0 = time.time()
    soffice = find_soffice()
    if soffice is None:
        msg = "LibreOffice not found; install it (winget install TheDocumentFoundation.LibreOffice). Skipping recalculation."
        print(msg)
        return dict(all_passed=None, skipped=True, reason=msg, data_source=config.DATA_SOURCE)
    xlsx_out = Path(xlsx_out or XLSX)
    report_out = Path(report_out or REPORT)
    builder = load_builder()
    sample, pts, psi_bins, realized = load_tables(quick)
    sc = load_scorecard()
    ref = sample_reference(sample, sc, psi_bins)
    cases = builder.calc_cases()
    work = Path(tempfile.mkdtemp(prefix="credit_xl_"))
    results, shipped, meta0 = [], None, None
    for i, (label, inputs) in enumerate(cases.items()):
        src = work / f"case{i}.xlsx"
        meta = builder.build(src, calc_inputs=inputs, quick=quick)
        out = recalc(src, work / f"out{i}", soffice)
        wbv = openpyxl.load_workbook(out, data_only=True)
        res = compare(label, wbv, meta, ref, sample, sc, inputs, realized)
        results.append(res)
        print(("PASS" if res["passed"] else "FAIL"), label, res["issues"][:3])
        if i == 0:
            shipped, meta0 = out, meta
    xlsx_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(shipped, xlsx_out)
    patch_author(xlsx_out)
    xlsx_out.with_suffix(".meta.json").write_text(json.dumps(meta0, indent=1, default=str), encoding="utf-8")
    worst = {}
    for r in results:
        for k, v in r["worst_diffs"].items():
            worst[k] = max(worst.get(k, 0.0), v)
    report = dict(data_source=config.DATA_SOURCE, all_passed=all(r["passed"] for r in results), quick=quick,
                  n_loans=int(len(sample)), n_cases=len(results), worst_diffs=worst, cases=results,
                  runtime_s=round(time.time() - t0, 1), n_formulas=meta0.get("n_formulas"))
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    print("all passed:", report["all_passed"], "runtime", report["runtime_s"], "s")
    return report


def main():
    build_and_reconcile(quick="--quick" in sys.argv)


if __name__ == "__main__":
    main()
