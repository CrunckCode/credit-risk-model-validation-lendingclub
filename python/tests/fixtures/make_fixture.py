"""Synthetic LendingClub-shaped raw frame (same raw column names and string formats) for tests that avoid the real data."""
import numpy as np
import pandas as pd

from credit_validation import columns as C

# ===== CONFIG (user inputs) =====
N_ROWS = 3000
FIXTURE_SEED = 7
ISSUE_START, ISSUE_END = pd.Timestamp("2007-06-01"), pd.Timestamp("2014-12-01")
STATUS_END = pd.Timestamp("2016-01-01")
GRADES = list("ABCDEFG")
STATES = ["CA", "NY", "TX", "FL", "IL", "NJ", "GA", "PA", "OH", "VA", "WA", "MA"]
PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "other", "small_business", "car"]
OWNERSHIP = ["RENT", "MORTGAGE", "OWN", "OTHER", "NONE", "ANY"]
VERIFS = ["Verified", "Source Verified", "Not Verified"]
EMP = ["< 1 year", "1 year", "2 years", "5 years", "10+ years", "n/a"]
# ===== END CONFIG =====


def _fmt(d: pd.Series) -> pd.Series:
    return d.dt.strftime("%b-%y")


def make_raw_frame(n: int = N_ROWS, seed: int = FIXTURE_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range(ISSUE_START, ISSUE_END, freq="MS")
    issue = pd.Series(rng.choice(months, n)).astype("datetime64[ns]")
    grade = rng.choice(GRADES, n, p=[0.17, 0.3, 0.27, 0.15, 0.07, 0.03, 0.01])
    gidx = pd.Series(grade).map({g: i for i, g in enumerate(GRADES)}).to_numpy()
    sub = np.array([f"{g}{k}" for g, k in zip(grade, rng.integers(1, 6, n))])
    int_rate = np.round(6 + 2.6 * gidx + rng.normal(0, 0.8, n), 2)
    term = rng.choice([" 36 months", " 60 months"], n, p=[0.75, 0.25])
    funded = np.round(rng.uniform(1000, 35000, n) / 25) * 25
    inc = np.round(np.exp(rng.normal(11.0, 0.55, n)), 0)
    dti = np.round(np.clip(rng.normal(16, 7, n), 0, 40), 2)
    inq = rng.poisson(0.9, n).astype(float)
    revol_util = np.round(np.clip(rng.normal(55, 25, n), 0, 120), 1)
    z = -3.4 + 0.45 * gidx + 0.04 * (dti - 16) + 0.25 * inq + 0.012 * (revol_util - 55)
    p_bad = 1 / (1 + np.exp(-z))
    bad = rng.random(n) < p_bad
    inst = np.round(funded / np.where(term == " 36 months", 33, 52), 2)

    # earliest_cr_line with two-digit years, including old lines that %y maps to the future
    cr_year = rng.integers(1955, 2008, n)
    cr_month = rng.integers(1, 13, n)
    cr = pd.to_datetime(dict(year=cr_year, month=cr_month, day=1))
    cr = pd.Series(np.minimum(cr.to_numpy(), (issue - pd.DateOffset(months=6)).to_numpy())).astype("datetime64[ns]")

    # outcomes
    mob_obs = ((STATUS_END.year - issue.dt.year) * 12 + (STATUS_END.month - issue.dt.month)).to_numpy()
    pay_months = np.where(bad, rng.integers(0, 30, n), rng.integers(6, 60, n))
    pay_months = np.minimum(pay_months, mob_obs)
    last_pay = (issue + pd.to_timedelta(pay_months * 30.4, unit="D")).dt.to_period("M").dt.to_timestamp()
    never_paid = bad & (rng.random(n) < 0.03)
    term_m = np.where(term == " 36 months", 36, 60)
    done = ~bad & (mob_obs > term_m + 1) & (rng.random(n) < 0.95)
    status = np.where(bad, rng.choice(["Charged Off", "Default", "Late (31-120 days)"], n, p=[0.88, 0.05, 0.07]),
                      np.where(done, "Fully Paid", np.where(rng.random(n) < 0.05, "In Grace Period", "Current")))
    dnmcp = (issue < pd.Timestamp("2010-06-01")) & (rng.random(n) < 0.12)
    status = np.where(dnmcp & (status == "Fully Paid"), "Does not meet the credit policy. Status:Fully Paid", status)
    status = np.where(dnmcp & (status == "Charged Off"), "Does not meet the credit policy. Status:Charged Off", status)
    co = np.isin(status, list(("Charged Off", "Does not meet the credit policy. Status:Charged Off")))
    rec_prncp = np.where(co, np.round(funded * rng.uniform(0.05, 0.8, n), 2), np.where(done, funded, np.round(funded * rng.uniform(0, 0.6, n), 2)))
    recoveries = np.where(co & (rng.random(n) < 0.55), np.round(funded * rng.uniform(0, 0.15, n), 2), 0.0)

    emp_len = rng.choice(EMP, n)
    raw = pd.DataFrame({
        C.LOAN_ID: np.arange(1_000_000, 1_000_000 + n),
        C.LOAN_AMNT: funded, C.FUNDED_AMNT: funded, C.TERM: term, C.INT_RATE: int_rate, C.INSTALLMENT: inst,
        C.GRADE: grade, C.SUB_GRADE: sub, C.EMP_LENGTH: emp_len, C.HOME_OWNERSHIP: rng.choice(OWNERSHIP, n, p=[0.45, 0.4, 0.1, 0.02, 0.02, 0.01]),
        C.ANNUAL_INC: inc, C.VERIFICATION_STATUS: rng.choice(VERIFS, n), C.ISSUE_D: _fmt(issue), C.LOAN_STATUS: status,
        C.PURPOSE: rng.choice(PURPOSES, n), C.ADDR_STATE: rng.choice(STATES, n), C.DTI: dti,
        C.DELINQ_2YRS: rng.poisson(0.2, n).astype(float), C.EARLIEST_CR_LINE: _fmt(cr), C.INQ_LAST_6MTHS: inq,
        C.MTHS_SINCE_LAST_DELINQ: np.where(rng.random(n) < 0.55, np.nan, rng.integers(0, 90, n).astype(float)),
        C.MTHS_SINCE_LAST_RECORD: np.where(rng.random(n) < 0.85, np.nan, rng.integers(0, 100, n).astype(float)),
        C.OPEN_ACC: rng.integers(2, 30, n).astype(float), C.PUB_REC: rng.poisson(0.1, n).astype(float),
        C.REVOL_BAL: rng.integers(0, 60000, n), C.REVOL_UTIL: revol_util, C.TOTAL_ACC: rng.integers(3, 60, n).astype(float),
        C.INITIAL_LIST_STATUS: np.where((issue >= pd.Timestamp("2012-01-01")) & (rng.random(n) < 0.5), "w", "f"),
        C.OUT_PRNCP: np.where(status == "Current", np.round(funded * 0.5, 2), 0.0),
        C.TOTAL_PYMNT: np.round(rec_prncp * 1.1, 2), C.TOTAL_REC_PRNCP: rec_prncp,
        C.TOTAL_REC_INT: np.round(rec_prncp * 0.1, 2), C.TOTAL_REC_LATE_FEE: 0.0, C.RECOVERIES: recoveries,
        C.COLLECTION_RECOVERY_FEE: np.round(recoveries * 0.1, 2), C.LAST_PYMNT_D: _fmt(last_pay),
        C.LAST_PYMNT_AMNT: inst, C.ACC_NOW_DELINQ: 0.0,
    })
    raw.loc[never_paid, C.LAST_PYMNT_D] = np.nan
    # mimic the raw file's whitespace and a few missing numerics
    raw.loc[rng.random(n) < 0.002, C.ANNUAL_INC] = np.nan
    return raw


def write_fixture_csv(path, n: int = N_ROWS) -> None:
    raw = make_raw_frame(n)
    raw.insert(0, "", np.arange(len(raw)))       # the real file's unnamed leading index column
    raw.to_csv(path, index=False)


def make_defaults_csv(raw: pd.DataFrame, path) -> None:
    """Tiny stand-in for loan_data_defaults.csv built from the same fixture."""
    co = raw[raw[C.LOAN_STATUS].isin(["Charged Off", "Does not meet the credit policy. Status:Charged Off"])]
    rr = (co[C.RECOVERIES] / co[C.FUNDED_AMNT]).clip(0, 1)
    ccf = ((co[C.FUNDED_AMNT] - co[C.TOTAL_REC_PRNCP]) / co[C.FUNDED_AMNT]).clip(0, 1)
    pd.DataFrame({C.LOAN_ID: co[C.LOAN_ID], "recovery_rate": rr, "CCF": ccf}).to_csv(path, index=False)
