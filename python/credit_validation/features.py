"""Application-time feature construction and the leakage guard."""
import numpy as np
import pandas as pd

from . import columns as C

# ===== CONFIG (user inputs) =====
EXTRA_FORBIDDEN = (C.MOB_LAST_PAY, C.MOB_OBSERVED, C.SPLIT)
MONTHS_PER_YEAR = 12
# ===== END CONFIG =====

FORBIDDEN = frozenset(C.POST_ORIGINATION) | frozenset(C.VINTAGE_PROXIES) | frozenset(C.TARGET_COLUMNS) | frozenset(EXTRA_FORBIDDEN)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[C.TERM_M] = out[C.TERM].astype(str).str.extract(r"(\d+)", expand=False).astype("float32")
    emp = out[C.EMP_LENGTH].astype(str).str.extract(r"(\d+)", expand=False).astype("float32")
    # "< 1 year" carries no digit but means zero years; true n/a stays missing
    lt1 = out[C.EMP_LENGTH].astype(str).str.startswith("<")
    out[C.EMP_YEARS] = emp.where(~lt1, 0.0).astype("float32")
    cr = out[C.EARLIEST_CR_LINE]
    out[C.CREDIT_AGE_M] = ((out[C.ISSUE_D].dt.year - cr.dt.year) * MONTHS_PER_YEAR + (out[C.ISSUE_D].dt.month - cr.dt.month)).astype("float32")
    inc = out[C.ANNUAL_INC].astype("float64").where(out[C.ANNUAL_INC] > 0)
    out[C.LOAN_TO_INC] = (out[C.LOAN_AMNT] / inc).astype("float32")
    out[C.PTI] = (out[C.INSTALLMENT] * MONTHS_PER_YEAR / inc).astype("float32")
    return out


def assert_no_leakage(feature_cols) -> None:
    bad = sorted(set(feature_cols) & FORBIDDEN)
    if bad:
        raise ValueError(f"leakage: forbidden columns in feature list: {bad}")
