"""Time-aware splits for validation, plus the random split used only to reproduce the original notebook."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from . import columns as C

# ===== CONFIG (user inputs) =====
OBS_BY_TARGET = {C.TARGET_BAD12: C.OBS12, C.TARGET_BAD24: C.OBS24}
TERM_36 = "36 months"
# ===== END CONFIG =====


def _tag(d: pd.DataFrame, name: str) -> pd.DataFrame:
    d = d.copy()
    d[C.SPLIT] = name
    return d


def time_splits(df: pd.DataFrame, target: str = C.TARGET_BAD12) -> dict:
    obs_col = OBS_BY_TARGET[target]
    keep = ~df[C.LOAN_STATUS].astype(str).str.startswith(config.DNMCP_PREFIX) if config.EXCLUDE_DNMCP else pd.Series(True, index=df.index)
    issue = df[C.ISSUE_D]
    obs = df[obs_col].astype(bool)

    dev = df[keep & obs & (issue >= config.DEV[0]) & (issue <= config.DEV[1])]
    tr, ho = train_test_split(dev, test_size=config.HOLDOUT_SHARE, random_state=config.SEED, stratify=dev[target])
    oot1 = df[keep & obs & (issue >= config.OOT1_WINDOW[0]) & (issue <= config.OOT1_WINDOW[1])]
    oot2 = df[keep & obs & (issue >= config.OOT2_WINDOW[0]) & (issue <= config.OOT2_WINDOW[1])]
    c36 = df[keep & (df[C.TERM] == TERM_36) & (issue <= config.COMPLETE36_LAST_ISSUE)]
    rtr, rte = train_test_split(df, test_size=config.REPRO_TEST_SHARE, random_state=config.REPRO_SEED)
    return {
        C.DEV_TRAIN: _tag(tr.sort_index(), C.DEV_TRAIN), C.DEV_HOLDOUT: _tag(ho.sort_index(), C.DEV_HOLDOUT),
        C.OOT1: _tag(oot1, C.OOT1), C.OOT2: _tag(oot2, C.OOT2), C.COMPLETE36: _tag(c36, C.COMPLETE36),
        C.REPRO_TRAIN: _tag(rtr.sort_index(), C.REPRO_TRAIN), C.REPRO_TEST: _tag(rte.sort_index(), C.REPRO_TEST),
    }
