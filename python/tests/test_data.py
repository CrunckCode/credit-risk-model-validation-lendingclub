import numpy as np
import pandas as pd
import pytest

from credit_validation import config, data, targets, splits, features
from credit_validation import columns as C

REAL_OK = (config.RAW_DIR / config.RAW_FILE).exists()


def _mini(rows):
    """rows: (status, issue, last_pay or None, funded, rec_prncp, recoveries, fee)"""
    df = pd.DataFrame(rows, columns=[C.LOAN_STATUS, C.ISSUE_D, C.LAST_PYMNT_D, C.FUNDED_AMNT, C.TOTAL_REC_PRNCP,
                                     C.RECOVERIES, C.COLLECTION_RECOVERY_FEE])
    df[C.ISSUE_D] = pd.to_datetime(df[C.ISSUE_D])
    df[C.LAST_PYMNT_D] = pd.to_datetime(df[C.LAST_PYMNT_D])
    df[C.TERM] = "36 months"
    return targets.add_targets(df)


# ---------- fixture-based logic tests ----------
def test_bad12_hand_examples():
    t = _mini([
        ("Charged Off", "2011-01-01", "2011-06-01", 10000, 4000, 500, 50),      # bad inside 12m
        ("Charged Off", "2011-01-01", "2012-01-01", 10000, 4000, 0, 0),         # last pay at month 12: not bad12
        ("Fully Paid", "2011-01-01", "2012-06-01", 10000, 10000, 0, 0),
        ("Default", "2011-01-01", None, 10000, 0, 0, 0),                        # never paid counts as month 0
        ("In Grace Period", "2011-01-01", "2011-03-01", 10000, 500, 0, 0),      # coded good
        ("Late (31-120 days)", "2013-01-01", "2013-03-01", 5000, 100, 0, 0),
    ])
    assert t[C.TARGET_BAD12].tolist() == [1, 0, 0, 1, 0, 1]
    assert t[C.TARGET_LIFETIME].tolist() == [1, 1, 0, 1, 0, 1]
    assert t[C.MOB_LAST_PAY].tolist()[:4] == [5, 12, 17, 0]


def test_obs_window_buffer():
    t = _mini([("Current", d, "2015-12-01", 1000, 0, 0, 0) for d in ("2014-11-01", "2014-12-01", "2015-06-01")])
    # status date Jan-2016: Nov-14 is 14 months observed (12 + buffer 2), Dec-14 only 13
    assert t[C.MOB_OBSERVED].tolist() == [14, 13, 7]
    assert t[C.OBS12].tolist() == [True, False, False]


def test_loss_fields_hand_example():
    t = _mini([("Charged Off", "2011-01-01", "2011-06-01", 10000, 4000, 500, 50),
               ("Fully Paid", "2011-01-01", "2012-06-01", 10000, 10000, 0, 0)])
    assert t.loc[0, C.NET_LOSS] == pytest.approx(10000 - 4000 - 500 + 50)
    assert t.loc[0, C.CCF] == pytest.approx(0.6)
    assert t.loc[0, C.RECOVERY_RATE] == pytest.approx(0.05)
    assert t.loc[0, C.RECOVERED_ANY] == 1
    assert t.loc[1, C.NET_LOSS] == 0 and np.isnan(t.loc[1, C.CCF]) and np.isnan(t.loc[1, C.RECOVERY_RATE])


def test_clean_formats(loans):
    assert pd.api.types.is_datetime64_any_dtype(loans[C.ISSUE_D])
    assert loans[C.ISSUE_D].dt.day.eq(1).all()
    assert set(loans[C.HOME_OWNERSHIP].astype(str)) <= {"RENT", "MORTGAGE", "OWN", "OTHER"}
    assert set(loans[C.TERM].astype(str)) == {"36 months", "60 months"}
    assert loans[C.INT_RATE].dtype == np.float32
    assert loans[C.GRADE].dtype.name == "category"
    assert loans[C.EMP_LENGTH].isna().any() and "n/a" not in set(loans[C.EMP_LENGTH].dropna().astype(str))


def test_two_digit_year_fix():
    e = pd.Series(pd.to_datetime(["2062-03-01", "1985-01-01", "2011-05-01"]))
    i = pd.Series(pd.to_datetime(["2011-12-01", "2011-12-01", "2011-12-01"]))
    fixed = data.fix_two_digit_years(e, i)
    assert fixed.tolist() == list(pd.to_datetime(["1962-03-01", "1985-01-01", "2011-05-01"]))
    # pivot year: 85 -> 1985 but 62 -> 2062, which is why the fix is needed
    assert data.parse_month_year(pd.Series(["Dec-11", "Jan-85", None])).iloc[:2].tolist() == list(pd.to_datetime(["2011-12-01", "1985-01-01"]))


def test_no_future_credit_lines_fixture(loans):
    assert (loans[C.EARLIEST_CR_LINE] <= loans[C.ISSUE_D]).all()
    assert (loans[C.CREDIT_AGE_M] >= 0).all()


def test_load_loans_cache_and_nrows(raw_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INTERIM_DIR", tmp_path / "interim")
    small = data.load_loans(raw_dir=raw_dir, nrows=100)
    assert len(small) == 100 and not (tmp_path / "interim").exists()
    a = data.load_loans(raw_dir=raw_dir)
    files = list((tmp_path / "interim").glob("*.parquet"))
    assert len(files) == 1
    b = data.load_loans(raw_dir=raw_dir)
    pd.testing.assert_frame_equal(a, b)
    data.load_loans(raw_dir=raw_dir, refresh=True)
    assert len(list((tmp_path / "interim").glob("*.parquet"))) == 1


def test_defaults_reference_columns(raw_dir):
    d = data.load_defaults_reference(raw_dir=raw_dir)
    assert list(d.columns) == [C.LOAN_ID, C.RECOVERY_RATE, C.CCF]


def test_fixture_reconciles_to_defaults(raw_dir, loans):
    d = data.load_defaults_reference(raw_dir=raw_dir)
    co = loans[loans[C.RECOVERY_RATE].notna()]
    m = co.merge(d, on=C.LOAN_ID, suffixes=("", "_ref"))
    assert len(m) == len(d) == len(co)
    assert np.allclose(m[C.CCF], m[C.CCF + "_ref"], atol=1e-6)
    assert np.allclose(m[C.RECOVERY_RATE], m[C.RECOVERY_RATE + "_ref"], atol=1e-6)


@pytest.mark.parametrize("col", sorted(set(C.POST_ORIGINATION) | set(C.VINTAGE_PROXIES) | set(C.TARGET_COLUMNS)))
def test_assert_no_leakage_raises_for_each_forbidden(col):
    with pytest.raises(ValueError):
        features.assert_no_leakage([C.DTI, col])


def test_assert_no_leakage_passes_clean_lists():
    features.assert_no_leakage(C.APPLICATION_FEATURES)
    features.assert_no_leakage(C.INDEP_FEATURES)


def test_build_features(loans):
    assert set(loans[C.TERM_M].dropna()) == {36.0, 60.0}
    assert loans[C.EMP_YEARS].max() == 10 and loans[C.EMP_YEARS].min() == 0
    assert (loans[C.LOAN_TO_INC].dropna() > 0).all()
    one = loans.iloc[0]
    assert one[C.PTI] == pytest.approx(one[C.INSTALLMENT] * 12 / one[C.ANNUAL_INC], rel=1e-4)


def test_splits_disjoint_and_filters(fixture_splits):
    s = fixture_splits
    assert set(s) == {C.DEV_TRAIN, C.DEV_HOLDOUT, C.OOT1, C.OOT2, C.COMPLETE36, C.REPRO_TRAIN, C.REPRO_TEST}
    tr, ho = s[C.DEV_TRAIN], s[C.DEV_HOLDOUT]
    assert not set(tr[C.LOAN_ID]) & set(ho[C.LOAN_ID])
    assert not set(tr[C.LOAN_ID]) & set(s[C.OOT1][C.LOAN_ID])
    assert tr[C.ISSUE_D].max() <= config.DEV[1] and s[C.OOT1][C.ISSUE_D].min() >= config.OOT1_WINDOW[0]
    assert s[C.OOT1][C.ISSUE_D].max() <= config.OOT1_WINDOW[1]
    assert s[C.OOT2][C.ISSUE_D].min() >= config.OOT2_WINDOW[0]
    for k in (C.DEV_TRAIN, C.DEV_HOLDOUT, C.OOT1, C.OOT2):
        assert s[k][C.OBS12].all()
        assert not s[k][C.LOAN_STATUS].astype(str).str.startswith(config.DNMCP_PREFIX).any()
    assert not set(s[C.REPRO_TRAIN][C.LOAN_ID]) & set(s[C.REPRO_TEST][C.LOAN_ID])
    assert len(s[C.REPRO_TRAIN]) + len(s[C.REPRO_TEST]) > len(s[C.DEV_TRAIN])
    c36 = s[C.COMPLETE36]
    assert (c36[C.TERM] == "36 months").all() and (c36[C.ISSUE_D] <= config.COMPLETE36_LAST_ISSUE).all()
    assert (s[C.DEV_TRAIN][C.SPLIT] == C.DEV_TRAIN).all()


def test_holdout_is_stratified_and_deterministic(loans):
    a, b = splits.time_splits(loans), splits.time_splits(loans)
    assert a[C.DEV_HOLDOUT][C.LOAN_ID].tolist() == b[C.DEV_HOLDOUT][C.LOAN_ID].tolist()
    share = len(a[C.DEV_HOLDOUT]) / (len(a[C.DEV_HOLDOUT]) + len(a[C.DEV_TRAIN]))
    assert share == pytest.approx(config.HOLDOUT_SHARE, abs=0.01)
    assert a[C.DEV_HOLDOUT][C.TARGET_BAD12].mean() == pytest.approx(a[C.DEV_TRAIN][C.TARGET_BAD12].mean(), abs=0.01)


def test_target_audit_shape(loans):
    a = targets.target_audit(loans)
    assert a[C.VINTAGE].tolist() == sorted(loans[C.VINTAGE].unique())
    assert a["n_loans"].sum() == len(loans)
    assert a["lifetime_bad_rate"].between(0, 1).all() and a["share_60m"].between(0, 1).all()


# ---------- real data (slow) ----------
@pytest.fixture(scope="module")
def real():
    return features.build_features(targets.add_targets(data.load_loans()))


@pytest.mark.slow
def test_real_counts(real):
    assert len(real) == 466_285
    vc = real[C.LOAN_STATUS].value_counts()
    assert vc["Current"] == 224_226 and vc["Fully Paid"] == 184_739 and vc["Charged Off"] == 42_475
    assert vc["Late (31-120 days)"] == 6_900 and vc["In Grace Period"] == 3_146 and vc["Default"] == 832
    assert real[C.TARGET_LIFETIME].sum() == 50_968
    assert real[C.ISSUE_D].min() == pd.Timestamp("2007-06-01") and real[C.ISSUE_D].max() == pd.Timestamp("2014-12-01")


@pytest.mark.slow
def test_real_dates_and_status_date(real):
    assert real[C.LAST_PYMNT_D].max() <= config.STATUS_DATE
    assert (real[C.EARLIEST_CR_LINE].dropna() <= real.loc[real[C.EARLIEST_CR_LINE].notna(), C.ISSUE_D]).all()


@pytest.mark.slow
def test_real_vintage_pattern(real):
    a = targets.target_audit(real).set_index(C.VINTAGE)
    assert a.loc[2007, "lifetime_bad_rate"] > 0.2 > 0.1 > a.loc[2014, "lifetime_bad_rate"]
    assert a.loc[2010:2014, "bad12_rate"].between(0.04, 0.05).all()


@pytest.mark.slow
def test_real_defaults_reconcile(real):
    d = data.load_defaults_reference()
    assert len(d) == 43_236
    co = real[real[C.RECOVERY_RATE].notna()]
    assert len(co) == 43_236
    assert d[C.RECOVERY_RATE].mean() == pytest.approx(0.0608, abs=5e-4)
    assert d[C.CCF].mean() == pytest.approx(0.736, abs=1e-3)
    m = co.merge(d, on=C.LOAN_ID, suffixes=("", "_ref"))
    assert len(m) == 43_236
    assert np.abs(m[C.CCF] - m[C.CCF + "_ref"]).max() < 1e-6
    assert np.abs(m[C.RECOVERY_RATE] - m[C.RECOVERY_RATE + "_ref"]).max() < 1e-6


@pytest.mark.slow
def test_real_splits(real):
    s = splits.time_splits(real)
    months = {k: set(s[k][C.ISSUE_D].unique()) for k in (C.DEV_TRAIN, C.OOT1, C.OOT2)}
    assert not months[C.DEV_TRAIN] & months[C.OOT1] and not months[C.OOT1] & months[C.OOT2]
    for k in (C.DEV_TRAIN, C.DEV_HOLDOUT, C.OOT1, C.OOT2):
        assert s[k][C.OBS12].all()
    assert s[C.OOT2][C.ISSUE_D].max() == pd.Timestamp("2014-11-01")
    assert len(s[C.COMPLETE36]) > 70_000 and s[C.COMPLETE36][C.LOAN_STATUS].eq("Current").mean() < 0.001
