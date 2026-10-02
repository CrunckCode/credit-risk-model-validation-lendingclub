"""Target and outcome construction: fixed-window bad flags, lifetime bad, loss fields, vintage audit."""
import numpy as np
import pandas as pd

from . import config
from . import columns as C

# ===== CONFIG (user inputs) =====
WINDOWS = ((12, C.TARGET_BAD12, C.OBS12), (24, C.TARGET_BAD24, C.OBS24))
AUDIT_COLUMNS = ("n_loans", "n_obs12", "lifetime_bad_rate", "bad12_rate", "share_current", "share_60m")
CURRENT_STATUS = "Current"
TERM_60 = "60 months"
# ===== END CONFIG =====


def _month_index(s: pd.Series) -> pd.Series:
    return s.dt.year * 12 + s.dt.month


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    issue = _month_index(out[C.ISSUE_D])
    status_idx = config.STATUS_DATE.year * 12 + config.STATUS_DATE.month
    # never-paid loans count as zero months on book for the default-timing proxy
    out[C.MOB_LAST_PAY] = (_month_index(out[C.LAST_PYMNT_D]) - issue).fillna(0).astype("int16")
    out[C.MOB_OBSERVED] = (status_idx - issue).astype("int16")
    is_bad = out[C.LOAN_STATUS].isin(config.BAD_STATUSES)
    for window, tgt, obs in WINDOWS:
        out[tgt] = (is_bad & (out[C.MOB_LAST_PAY] < window)).astype("int8")
        out[obs] = out[C.MOB_OBSERVED] >= window + config.OBS_BUFFER_MONTHS
    out[C.TARGET_LIFETIME] = is_bad.astype("int8")

    co = out[C.LOAN_STATUS].isin(config.CHARGEOFF_STATUSES)
    funded = out[C.FUNDED_AMNT]
    rec = (out[C.RECOVERIES] / funded).clip(0, 1)
    out[C.RECOVERY_RATE] = rec.where(co)
    out[C.RECOVERED_ANY] = (out[C.RECOVERIES] > 0).astype("float32").where(co)
    ccf = ((funded - out[C.TOTAL_REC_PRNCP]) / funded).clip(0, 1)
    out[C.CCF] = ccf.where(co)
    loss = funded - out[C.TOTAL_REC_PRNCP] - out[C.RECOVERIES] + out[C.COLLECTION_RECOVERY_FEE]
    out[C.NET_LOSS] = loss.where(co, 0.0)
    out[C.ISSUE_MONTH] = out[C.ISSUE_D]
    out[C.VINTAGE] = out[C.ISSUE_D].dt.year.astype("int16")
    return out


def target_audit(df: pd.DataFrame) -> pd.DataFrame:
    """By-vintage table showing why lifetime bad rate is censored while the 12-month flag is stable."""
    obs = df[C.OBS12].astype(bool)
    g = df.assign(
        _current=(df[C.LOAN_STATUS] == CURRENT_STATUS).astype(float),
        _m60=(df[C.TERM] == TERM_60).astype(float),
        _bad12=df[C.TARGET_BAD12].where(obs),
    ).groupby(C.VINTAGE, observed=True)
    res = pd.DataFrame({
        AUDIT_COLUMNS[0]: g.size(),
        AUDIT_COLUMNS[1]: g[C.OBS12].sum(),
        AUDIT_COLUMNS[2]: g[C.TARGET_LIFETIME].mean(),
        AUDIT_COLUMNS[3]: g["_bad12"].mean(),
        AUDIT_COLUMNS[4]: g["_current"].mean(),
        AUDIT_COLUMNS[5]: g["_m60"].mean(),
    })
    return res.reset_index()
