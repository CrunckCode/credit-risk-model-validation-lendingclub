"""Column-name constants. All other modules import names from here; no string-literal column names elsewhere."""

# raw columns used
LOAN_ID, LOAN_AMNT, FUNDED_AMNT, TERM, INT_RATE, INSTALLMENT = "id", "loan_amnt", "funded_amnt", "term", "int_rate", "installment"
GRADE, SUB_GRADE, EMP_LENGTH, HOME_OWNERSHIP = "grade", "sub_grade", "emp_length", "home_ownership"
ANNUAL_INC, VERIFICATION_STATUS, ISSUE_D, LOAN_STATUS = "annual_inc", "verification_status", "issue_d", "loan_status"
PURPOSE, ADDR_STATE, DTI, DELINQ_2YRS = "purpose", "addr_state", "dti", "delinq_2yrs"
EARLIEST_CR_LINE, INQ_LAST_6MTHS = "earliest_cr_line", "inq_last_6mths"
MTHS_SINCE_LAST_DELINQ, MTHS_SINCE_LAST_RECORD = "mths_since_last_delinq", "mths_since_last_record"
OPEN_ACC, PUB_REC, REVOL_BAL, REVOL_UTIL, TOTAL_ACC = "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc"
INITIAL_LIST_STATUS = "initial_list_status"
# post-origination raw columns (never PD features)
OUT_PRNCP, OUT_PRNCP_INV, TOTAL_PYMNT, TOTAL_PYMNT_INV = "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv"
TOTAL_REC_PRNCP, TOTAL_REC_INT, TOTAL_REC_LATE_FEE = "total_rec_prncp", "total_rec_int", "total_rec_late_fee"
RECOVERIES, COLLECTION_RECOVERY_FEE = "recoveries", "collection_recovery_fee"
LAST_PYMNT_D, LAST_PYMNT_AMNT, NEXT_PYMNT_D, LAST_CREDIT_PULL_D = "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d"
PYMNT_PLAN, FUNDED_AMNT_INV = "pymnt_plan", "funded_amnt_inv"
COLLECTIONS_12M, ACC_NOW_DELINQ, TOT_COLL_AMT = "collections_12_mths_ex_med", "acc_now_delinq", "tot_coll_amt"
TOT_CUR_BAL, TOTAL_REV_HI_LIM, MTHS_SINCE_LAST_MAJOR_DEROG = "tot_cur_bal", "total_rev_hi_lim", "mths_since_last_major_derog"
MTHS_SINCE_ISSUE_D = "mths_since_issue_d"            # vintage proxy created by the original notebook; never a feature here

# derived columns
ISSUE_MONTH, VINTAGE, TERM_M, EMP_YEARS = "issue_month", "vintage", "term_m", "emp_years"
CREDIT_AGE_M, LOAN_TO_INC, PTI = "credit_age_m", "loan_to_inc", "pti"
MOB_LAST_PAY, MOB_OBSERVED = "mob_last_pay", "mob_observed"

# targets (bad = 1) and outcome fields
TARGET_BAD12, TARGET_BAD24, TARGET_LIFETIME = "target_bad12", "target_bad24", "target_lifetime"
OBS12, OBS24 = "obs12", "obs24"
SPLIT = "split"
RECOVERY_RATE, RECOVERED_ANY, CCF, NET_LOSS = "recovery_rate", "recovered_any", "ccf", "net_loss"

# scored outputs
PRED_PD, PRED_LGD, PRED_EAD, PRED_EL, SCORE = "pred_pd", "pred_lgd", "pred_ead", "pred_el", "score"

# split names
DEV_TRAIN, DEV_HOLDOUT, OOT1, OOT2, COMPLETE36 = "dev_train", "dev_holdout", "oot1", "oot2", "complete36"
REPRO_TRAIN, REPRO_TEST = "repro_train", "repro_test"

APPLICATION_FEATURES = (
    LOAN_AMNT, TERM_M, INT_RATE, INSTALLMENT, GRADE, SUB_GRADE, EMP_YEARS, HOME_OWNERSHIP, ANNUAL_INC,
    VERIFICATION_STATUS, PURPOSE, ADDR_STATE, DTI, DELINQ_2YRS, INQ_LAST_6MTHS, MTHS_SINCE_LAST_DELINQ,
    MTHS_SINCE_LAST_RECORD, OPEN_ACC, PUB_REC, REVOL_BAL, REVOL_UTIL, TOTAL_ACC, INITIAL_LIST_STATUS,
    CREDIT_AGE_M, LOAN_TO_INC, PTI,
)
LC_RISK_OUTPUTS = (GRADE, SUB_GRADE, INT_RATE, INSTALLMENT)       # LendingClub's own risk output: circularity flag
INDEP_FEATURES = tuple(c for c in APPLICATION_FEATURES if c not in LC_RISK_OUTPUTS)
CATEGORICAL_FEATURES = (GRADE, SUB_GRADE, HOME_OWNERSHIP, VERIFICATION_STATUS, PURPOSE, ADDR_STATE, INITIAL_LIST_STATUS)
POST_ORIGINATION = (
    OUT_PRNCP, OUT_PRNCP_INV, TOTAL_PYMNT, TOTAL_PYMNT_INV, TOTAL_REC_PRNCP, TOTAL_REC_INT, TOTAL_REC_LATE_FEE,
    RECOVERIES, COLLECTION_RECOVERY_FEE, LAST_PYMNT_D, LAST_PYMNT_AMNT, NEXT_PYMNT_D, LAST_CREDIT_PULL_D,
    LOAN_STATUS, PYMNT_PLAN, FUNDED_AMNT_INV, COLLECTIONS_12M, ACC_NOW_DELINQ, TOT_COLL_AMT, TOT_CUR_BAL,
    TOTAL_REV_HI_LIM, MTHS_SINCE_LAST_MAJOR_DEROG,
)
VINTAGE_PROXIES = (MTHS_SINCE_ISSUE_D,)
TARGET_COLUMNS = (TARGET_BAD12, TARGET_BAD24, TARGET_LIFETIME, OBS12, OBS24, RECOVERY_RATE, RECOVERED_ANY, CCF, NET_LOSS)
MONOTONE_SIGNS = {INT_RATE: 1, DTI: 1, INQ_LAST_6MTHS: 1, REVOL_UTIL: 1, LOAN_TO_INC: 1, PTI: 1, ANNUAL_INC: -1, CREDIT_AGE_M: -1}
