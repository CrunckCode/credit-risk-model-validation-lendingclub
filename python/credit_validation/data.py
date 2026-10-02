"""Raw LendingClub loader: usecols read, cleaning, typed frame, parquet cache."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from . import columns as C

# ===== CONFIG (user inputs) =====
ENCODINGS = ("utf-8", "latin-1")
DATE_FORMAT = "%b-%y"
OWNERSHIP_TO_OTHER = ("ANY", "NONE", "OTHER")
USE_COLUMNS = (
    C.LOAN_ID, C.LOAN_AMNT, C.FUNDED_AMNT, C.TERM, C.INT_RATE, C.INSTALLMENT, C.GRADE, C.SUB_GRADE, C.EMP_LENGTH,
    C.HOME_OWNERSHIP, C.ANNUAL_INC, C.VERIFICATION_STATUS, C.ISSUE_D, C.LOAN_STATUS, C.PURPOSE, C.ADDR_STATE, C.DTI,
    C.DELINQ_2YRS, C.EARLIEST_CR_LINE, C.INQ_LAST_6MTHS, C.MTHS_SINCE_LAST_DELINQ, C.MTHS_SINCE_LAST_RECORD,
    C.OPEN_ACC, C.PUB_REC, C.REVOL_BAL, C.REVOL_UTIL, C.TOTAL_ACC, C.INITIAL_LIST_STATUS, C.OUT_PRNCP, C.TOTAL_PYMNT,
    C.TOTAL_REC_PRNCP, C.TOTAL_REC_INT, C.TOTAL_REC_LATE_FEE, C.RECOVERIES, C.COLLECTION_RECOVERY_FEE, C.LAST_PYMNT_D,
    C.LAST_PYMNT_AMNT, C.ACC_NOW_DELINQ,
)
CATEGORY_COLUMNS = (
    C.TERM, C.GRADE, C.SUB_GRADE, C.EMP_LENGTH, C.HOME_OWNERSHIP, C.VERIFICATION_STATUS, C.LOAN_STATUS, C.PURPOSE,
    C.ADDR_STATE, C.INITIAL_LIST_STATUS,
)
PERCENT_COLUMNS = (C.INT_RATE, C.REVOL_UTIL)
DATE_COLUMNS = (C.ISSUE_D, C.LAST_PYMNT_D)
# money and outcome fields stay float64 so net-loss sums reconcile to the cent
FLOAT64_COLUMNS = (
    C.FUNDED_AMNT, C.OUT_PRNCP, C.TOTAL_PYMNT, C.TOTAL_REC_PRNCP, C.TOTAL_REC_INT, C.TOTAL_REC_LATE_FEE,
    C.RECOVERIES, C.COLLECTION_RECOVERY_FEE, C.LAST_PYMNT_AMNT,
)
DEFAULTS_RENAME = {C.LOAN_ID: C.LOAN_ID, "recovery_rate": C.RECOVERY_RATE, "CCF": C.CCF}
CACHE_PREFIX = "loans_"
# ===== END CONFIG =====


def _read_raw(path: Path, nrows) -> pd.DataFrame:
    err = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, usecols=list(USE_COLUMNS), encoding=enc, low_memory=False, nrows=nrows)
        except UnicodeDecodeError as e:  # latin-1 never fails, so the loop always ends
            err = e
    raise err


def parse_month_year(s: pd.Series) -> pd.Series:
    """Parse 'Dec-11' style strings once per unique value (fast, month-start timestamps)."""
    uniq = s.dropna().unique()
    mapping = dict(zip(uniq, pd.to_datetime(pd.Series(uniq), format=DATE_FORMAT, errors="coerce")))
    return s.map(mapping).astype("datetime64[ns]")


def fix_two_digit_years(earliest: pd.Series, issue: pd.Series) -> pd.Series:
    """A credit line cannot open after issue; %y mapped 1960s to 2060s, so pull those back 100 years."""
    months = earliest.values.astype("datetime64[M]")
    future = (earliest > issue).to_numpy()
    months = np.where(future, months - np.timedelta64(1200, "M"), months)
    return pd.Series(months.astype("datetime64[ns]"), index=earliest.index)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    for c in PERCENT_COLUMNS:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.rstrip("%").replace({"nan": np.nan}).astype(float)
    df[C.TERM] = df[C.TERM].str.strip()
    df[C.EMP_LENGTH] = df[C.EMP_LENGTH].where(df[C.EMP_LENGTH] != "n/a")
    df[C.HOME_OWNERSHIP] = df[C.HOME_OWNERSHIP].where(~df[C.HOME_OWNERSHIP].isin(OWNERSHIP_TO_OTHER), "OTHER")
    df[C.ISSUE_D] = parse_month_year(df[C.ISSUE_D])
    df[C.LAST_PYMNT_D] = parse_month_year(df[C.LAST_PYMNT_D])
    df[C.EARLIEST_CR_LINE] = fix_two_digit_years(parse_month_year(df[C.EARLIEST_CR_LINE]), df[C.ISSUE_D])
    for c in CATEGORY_COLUMNS:
        df[c] = df[c].astype("category")
    for c in df.columns:
        if c in CATEGORY_COLUMNS or c in DATE_COLUMNS or c == C.EARLIEST_CR_LINE:
            continue
        if c == C.LOAN_ID:
            df[c] = df[c].astype("int64")
        elif c in FLOAT64_COLUMNS:
            df[c] = df[c].astype("float64")
        else:
            df[c] = df[c].astype("float32")
    return df.reset_index(drop=True)


def _path_tag(raw: Path) -> str:
    # per-path tag so loading a different raw folder (tests) never evicts the real cache
    return hashlib.md5(str(raw.resolve()).encode()).hexdigest()[:8]


def _cache_path(raw: Path) -> Path:
    st = raw.stat()
    return config.INTERIM_DIR / f"{CACHE_PREFIX}{_path_tag(raw)}_{st.st_size}_{st.st_mtime_ns}.parquet"


def load_loans(raw_dir=None, refresh: bool = False, nrows=None) -> pd.DataFrame:
    raw = Path(raw_dir or config.RAW_DIR) / config.RAW_FILE
    if nrows is not None:
        return _clean(_read_raw(raw, nrows))
    cache = _cache_path(raw)
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    df = _clean(_read_raw(raw, None))
    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    for old in config.INTERIM_DIR.glob(f"{CACHE_PREFIX}{_path_tag(raw)}_*.parquet"):
        old.unlink()
    df.to_parquet(cache, index=False)
    return df


def load_defaults_reference(raw_dir=None) -> pd.DataFrame:
    """Original notebook's charged-off table (recovery rate and CCF) for reconciliation."""
    path = Path(raw_dir or config.RAW_DIR) / config.DEFAULTS_FILE
    err = None
    for enc in ENCODINGS:
        try:
            d = pd.read_csv(path, usecols=list(DEFAULTS_RENAME), encoding=enc, low_memory=False)
            return d.rename(columns=DEFAULTS_RENAME)[[C.LOAN_ID, C.RECOVERY_RATE, C.CCF]]
        except UnicodeDecodeError as e:
            err = e
    raise err
