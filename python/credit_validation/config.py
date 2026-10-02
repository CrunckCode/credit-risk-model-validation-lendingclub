"""Global configuration shared by all modules. Change values here, not inside modules."""
import os
from pathlib import Path
import pandas as pd

# ===== CONFIG (user inputs) =====
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = Path(os.environ.get("CV_RAW_DIR", r"D:\Job Arsenal\resume-main\OOPS_Project_Credit_Risk_Modeling"))
RAW_FILE, DEFAULTS_FILE = "loan_data_2007_2014.csv", "loan_data_defaults.csv"
INTERIM_DIR = ROOT / "data" / "interim"
OUT_DIR = ROOT / "python" / "outputs"
CHART_DIR, MODEL_DIR, EXCEL_INPUT_DIR = OUT_DIR / "charts", OUT_DIR / "models", OUT_DIR / "excel_inputs"
DATA_SOURCE = "LendingClub public 2007-2014"
SEED = 20261001

# censoring and windows
STATUS_DATE = pd.Timestamp("2016-01-31")           # latest last_pymnt_d in the file is Jan-2016
PD_WINDOW_MONTHS, OBS_BUFFER_MONTHS = 12, 2
DEV = (pd.Timestamp("2007-06-01"), pd.Timestamp("2012-12-01"))
OOT1_WINDOW = (pd.Timestamp("2013-01-01"), pd.Timestamp("2013-12-01"))
OOT2_WINDOW = (pd.Timestamp("2014-01-01"), pd.Timestamp("2014-11-01"))
HOLDOUT_SHARE = 0.2
REPRO_TEST_SHARE, REPRO_SEED = 0.2, 42
COMPLETE36_LAST_ISSUE = pd.Timestamp("2012-12-01")
EXCLUDE_DNMCP = True

# status definitions (original notebook: In Grace Period is coded GOOD)
BAD_STATUSES = ("Charged Off", "Default", "Late (31-120 days)", "Does not meet the credit policy. Status:Charged Off")
CHARGEOFF_STATUSES = ("Charged Off", "Does not meet the credit policy. Status:Charged Off")
DNMCP_PREFIX = "Does not meet the credit policy"

# scorecard
PDO, BASE_SCORE, BASE_ODDS = 20, 600, 50
BIN_MIN_SHARE, BIN_MAX_BINS, BIN_PREBINS = 0.05, 8, 50
CAT_MIN_SHARE, CAT_MAX_GROUPS = 0.02, 10
IV_MIN, P_MAX, VIF_MAX, CORR_MAX = 0.02, 0.05, 5.0, 0.7

# validation thresholds: (green_if_at_least_or_at_most, amber_limit) documented per test in validation.py
THRESHOLDS = dict(
    gini=(0.40, 0.30), gini_decay=(0.10, 0.20), ks=(0.30, 0.20), psi=(0.10, 0.25),
    hl_p=(0.05, 0.01), binom_p=(0.05, 0.0001), central_tendency=(0.10, 0.25),
    slope=((0.9, 1.1), (0.8, 1.2)), rank_inversions=(0, 2),
)
BOOTSTRAP_N, BOOT_ROW_CAP = 500, 50_000
TUNE_ITER, TUNE_VAL_YEARS = 8, (2010, 2011, 2012)
EXCEL_SAMPLE_LOANS = 5_000
QUICK_SHARE = 0.2
SHAP_SAMPLE = 5_000
SENSITIVITY_SHOCKS = dict(dti_pts=5.0, int_rate_pts=2.0, income_pct=-10.0, revol_util_pts=10.0, inq_add=1.0)
PD_MULTIPLIERS, LGD_ADD = (1.25, 1.5), 0.10
# ===== END CONFIG =====


def quick_share(quick: bool) -> float:
    return QUICK_SHARE if quick else 1.0
