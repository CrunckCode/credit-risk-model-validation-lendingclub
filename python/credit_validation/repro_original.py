"""Reproduce the original notebook's PD model (dummy design, lifetime target, random 80/20) and test the vintage-proxy effect.

Coarse classes are transcribed from the Preparation/Final notebooks (cells 150/151 name the dummies, the
Preparation cells 106-198 hold the cut points). Models are refit here; pd_model.sav is not available.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from . import config
from . import columns as C

# ===== CONFIG (user inputs) =====
REF_DATE = pd.Timestamp("2017-12-01")                     # the original computed "months since" against this date
MAX_ITER, L2_C = 1000, 1.0                                # original wrapped sklearn LogisticRegression with default C=1
STATE_GROUPS = (
    "ND_NE_IA_NV_FL_HI_AL", "NM_VA", "NY", "OK_TN_MO_LA_MD_NC", "CA", "UT_KY_AZ_NJ", "AR_MI_PA_OH_MN",
    "RI_MA_DE_SD_IN", "GA_WA_OR", "WI_MT", "TX", "IL_CT", "KS_SC_CO_VT_AK_MS", "WV_NH_WY_DC_ME_ID",
)
PURPOSE_GROUPS = {
    "educ__sm_b__wedd__ren_en__mov__house": ("educational", "small_business", "wedding", "renewable_energy", "moving", "house"),
    "credit_card": ("credit_card",),
    "debt_consolidation": ("debt_consolidation",),
    "oth__med__vacation": ("other", "medical", "vacation"),
    "major_purch__car__home_impr": ("major_purchase", "car", "home_improvement"),
}
HOME_GROUPS = {"RENT_OTHER_NONE_ANY": ("RENT", "OTHER", "NONE", "ANY"), "OWN": ("OWN",), "MORTGAGE": ("MORTGAGE",)}
# numeric ladders: (label, lo, hi) with lo exclusive and hi inclusive on the original's integer or decimal scale
INT_RATE_CLASSES = (("<9.548", -np.inf, 9.548), ("9.548-12.025", 9.548, 12.025), ("12.025-15.74", 12.025, 15.74),
                    ("15.74-20.281", 15.74, 20.281), (">20.281", 20.281, np.inf))
ANNUAL_INC_CLASSES = (("<20K", -np.inf, 20e3), ("20K-30K", 20e3, 30e3), ("30K-40K", 30e3, 40e3), ("40K-50K", 40e3, 50e3),
                      ("50K-60K", 50e3, 60e3), ("60K-70K", 60e3, 70e3), ("70K-80K", 70e3, 80e3), ("80K-90K", 80e3, 90e3),
                      ("90K-100K", 90e3, 100e3), ("100K-120K", 100e3, 120e3), ("120K-140K", 120e3, 140e3), (">140K", 140e3, np.inf))
DTI_CLASSES = (("<=1.4", -np.inf, 1.4), ("1.4-3.5", 1.4, 3.5), ("3.5-7.7", 3.5, 7.7), ("7.7-10.5", 7.7, 10.5),
               ("10.5-16.1", 10.5, 16.1), ("16.1-20.3", 16.1, 20.3), ("20.3-21.7", 20.3, 21.7), ("21.7-22.4", 21.7, 22.4),
               ("22.4-35", 22.4, 35), (">35", 35, np.inf))
ISSUE_AGE_CLASSES = (("<38", -np.inf, 37), ("38-39", 37, 39), ("40-41", 39, 41), ("42-48", 41, 48), ("49-52", 48, 52),
                     ("53-64", 52, 64), ("65-84", 64, 84), (">84", 84, np.inf))
CR_AGE_CLASSES = (("<140", -np.inf, 139), ("141-164", 139, 164), ("165-247", 164, 247), ("248-270", 247, 270),
                  ("271-352", 270, 352), (">352", 352, np.inf))
EMP_CLASSES = (("0", -np.inf, 0), ("1", 0, 1), ("2-4", 1, 4), ("5-6", 4, 6), ("7-9", 6, 9), ("10", 9, np.inf))
INQ_CLASSES = (("0", -np.inf, 0), ("1-2", 0, 2), ("3-6", 2, 6), (">6", 6, np.inf))
ACC_NOW_DELINQ_CLASSES = (("0", -np.inf, 0), (">=1", 0, np.inf))
DELINQ_CLASSES = (("0-3", -np.inf, 3), ("4-30", 3, 30), ("31-56", 30, 56), (">=57", 56, np.inf))       # plus Missing
RECORD_CLASSES = (("0-2", -np.inf, 2), ("3-20", 2, 20), ("21-31", 20, 31), ("32-80", 31, 80), ("81-86", 80, 86), (">86", 86, np.inf))
TERM_CLASSES = (("36", -np.inf, 36), ("60", 36, np.inf))
LISTING_STATUSES = ("f", "w")
GRADES = tuple("ABCDEFG")
VERIFICATIONS = ("Not Verified", "Source Verified", "Verified")
# reference (dropped) categories, from cell 151
REF = {
    C.GRADE: "G", C.HOME_OWNERSHIP: "RENT_OTHER_NONE_ANY", C.ADDR_STATE: STATE_GROUPS[0], C.VERIFICATION_STATUS: "Verified",
    C.PURPOSE: "educ__sm_b__wedd__ren_en__mov__house", C.INITIAL_LIST_STATUS: "f", C.TERM: "60", C.EMP_LENGTH: "0",
    C.MTHS_SINCE_ISSUE_D: ">84", C.INT_RATE: ">20.281", C.EARLIEST_CR_LINE: "<140", C.INQ_LAST_6MTHS: ">6",
    C.ACC_NOW_DELINQ: "0", C.ANNUAL_INC: "<20K", C.DTI: ">35", C.MTHS_SINCE_LAST_DELINQ: "0-3", C.MTHS_SINCE_LAST_RECORD: "0-2",
}
ORIGINAL_COARSE_CLASSES = {
    C.INT_RATE: INT_RATE_CLASSES, C.ANNUAL_INC: ANNUAL_INC_CLASSES, C.DTI: DTI_CLASSES,
    C.MTHS_SINCE_ISSUE_D: ISSUE_AGE_CLASSES, C.EARLIEST_CR_LINE: CR_AGE_CLASSES, C.EMP_LENGTH: EMP_CLASSES,
    C.INQ_LAST_6MTHS: INQ_CLASSES, C.ACC_NOW_DELINQ: ACC_NOW_DELINQ_CLASSES, C.MTHS_SINCE_LAST_DELINQ: DELINQ_CLASSES,
    C.MTHS_SINCE_LAST_RECORD: RECORD_CLASSES, C.TERM: TERM_CLASSES, C.HOME_OWNERSHIP: HOME_GROUPS,
    C.PURPOSE: PURPOSE_GROUPS, C.ADDR_STATE: STATE_GROUPS, C.GRADE: GRADES, C.VERIFICATION_STATUS: VERIFICATIONS,
    C.INITIAL_LIST_STATUS: LISTING_STATUSES,
}
# ===== END CONFIG =====


def _months_since(dates: pd.Series) -> pd.Series:
    return ((REF_DATE.year - dates.dt.year) * 12 + (REF_DATE.month - dates.dt.month)).astype("float32")


def _ladder(x: np.ndarray, classes, var: str, out: dict, miss_label=None) -> None:
    for label, lo, hi in classes:
        out[f"{var}:{label}"] = ((x > lo) & (x <= hi)).astype("float32")
    if miss_label:
        out[f"{var}:{miss_label}"] = np.isnan(x).astype("float32")


def _group_dummies(s: pd.Series, groups, var: str, out: dict) -> None:
    sv = s.astype(object)
    if isinstance(groups, dict):
        for name, members in groups.items():
            out[f"{var}:{name}"] = sv.isin(members).to_numpy(dtype="float32")
    elif var == C.ADDR_STATE:
        for name in groups:
            out[f"{var}:{name}"] = sv.isin(name.split("_")).to_numpy(dtype="float32")
    else:
        for name in groups:
            out[f"{var}:{name}"] = (sv == name).to_numpy(dtype="float32")


def build_dummy_design(df: pd.DataFrame, with_issue_age: bool = True) -> pd.DataFrame:
    """Dummy table of the original model with reference categories dropped (float32 to keep memory low)."""
    out = {}
    num = lambda c: df[c].to_numpy(dtype=np.float64)   # noqa: E731
    cr_age = _months_since(df[C.EARLIEST_CR_LINE]).to_numpy(dtype=np.float64)
    emp = df[C.EMP_LENGTH].astype(str).str.extract(r"(\d+)", expand=False).astype("float64")
    emp = emp.where(~df[C.EMP_LENGTH].astype(str).str.startswith("<"), 0.0).fillna(0.0).to_numpy()
    for c in (C.GRADE, C.VERIFICATION_STATUS, C.INITIAL_LIST_STATUS):
        _group_dummies(df[c], ORIGINAL_COARSE_CLASSES[c], c, out)
    _group_dummies(df[C.HOME_OWNERSHIP], HOME_GROUPS, C.HOME_OWNERSHIP, out)
    _group_dummies(df[C.ADDR_STATE], STATE_GROUPS, C.ADDR_STATE, out)
    _group_dummies(df[C.PURPOSE], PURPOSE_GROUPS, C.PURPOSE, out)
    _ladder(df[C.TERM].astype(str).str.extract(r"(\d+)", expand=False).astype("float64").to_numpy(), TERM_CLASSES, C.TERM, out)
    _ladder(emp, EMP_CLASSES, C.EMP_LENGTH, out)
    if with_issue_age:
        _ladder(_months_since(df[C.ISSUE_D]).to_numpy(dtype=np.float64), ISSUE_AGE_CLASSES, C.MTHS_SINCE_ISSUE_D, out)
    _ladder(num(C.INT_RATE), INT_RATE_CLASSES, C.INT_RATE, out)
    _ladder(np.nan_to_num(cr_age, nan=0.0), CR_AGE_CLASSES, C.EARLIEST_CR_LINE, out)
    _ladder(np.nan_to_num(num(C.INQ_LAST_6MTHS), nan=0.0), INQ_CLASSES, C.INQ_LAST_6MTHS, out)
    _ladder(np.nan_to_num(num(C.ACC_NOW_DELINQ), nan=0.0), ACC_NOW_DELINQ_CLASSES, C.ACC_NOW_DELINQ, out)
    _ladder(num(C.ANNUAL_INC), ANNUAL_INC_CLASSES, C.ANNUAL_INC, out)
    _ladder(num(C.DTI), DTI_CLASSES, C.DTI, out)
    _ladder(num(C.MTHS_SINCE_LAST_DELINQ), DELINQ_CLASSES, C.MTHS_SINCE_LAST_DELINQ, out, miss_label="Missing")
    _ladder(num(C.MTHS_SINCE_LAST_RECORD), RECORD_CLASSES, C.MTHS_SINCE_LAST_RECORD, out, miss_label="Missing")
    X = pd.DataFrame(out, index=df.index)
    drop = [f"{var}:{label}" for var, label in REF.items() if not (var == C.MTHS_SINCE_ISSUE_D and not with_issue_age)]
    return X.drop(columns=[c for c in drop if c in X.columns])


def _ks(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def _metrics(y, p) -> dict:
    auc = float(roc_auc_score(y, p))
    return dict(auc=auc, gini=2 * auc - 1, ks=_ks(y, p))


def _fit_eval(train, test, with_issue_age) -> dict:
    Xtr = build_dummy_design(train, with_issue_age)
    Xte = build_dummy_design(test, with_issue_age)[Xtr.columns]
    mdl = LogisticRegression(C=L2_C, max_iter=MAX_ITER).fit(Xtr, train[C.TARGET_LIFETIME])
    res = _metrics(test[C.TARGET_LIFETIME], mdl.predict_proba(Xte)[:, 1])
    res.update(n_features=int(Xtr.shape[1]), n_train=int(len(train)), n_test=int(len(test)))
    res["coefs"] = dict(zip(Xtr.columns, mdl.coef_[0].tolist()))
    return res


def reproduce_original_pd(splits=None) -> dict:
    """Fit on REPRO_TRAIN with the lifetime target; test metrics with and without MTHS_SINCE_ISSUE_D."""
    if splits is None:
        from . import data, targets, features, splits as sp
        splits = sp.time_splits(features.build_features(targets.add_targets(data.load_loans())))
    train, test = splits[C.REPRO_TRAIN], splits[C.REPRO_TEST]
    with_age = _fit_eval(train, test, True)
    without = _fit_eval(train, test, False)
    return {
        "data_source": config.DATA_SOURCE,
        "target": C.TARGET_LIFETIME,
        "with_mths_since_issue_d": with_age,
        "without_mths_since_issue_d": without,
        "auc_drop": with_age["auc"] - without["auc"],
        "gini_drop": with_age["gini"] - without["gini"],
        "ks_drop": with_age["ks"] - without["ks"],
        "bad_rate_train": float(train[C.TARGET_LIFETIME].mean()),
    }
