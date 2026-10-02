import numpy as np
import pandas as pd
import pytest

from credit_validation import config, binning, scorecard, repro_original
from credit_validation import columns as C


def _synth(n=20000, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-3 + 0.9 * x)))).astype(float)
    return x, y


def test_iv_hand_example():
    bad, good = np.array([10.0, 30.0]), np.array([90.0, 70.0])
    woe, ivc = binning.woe_iv(bad, good, smooth=0)
    assert woe == pytest.approx([np.log((90 / 160) / (10 / 40)),
                                 np.log((70 / 160) / (30 / 40))])
    assert ivc.sum() == pytest.approx(sum((g / 160 - b / 40) * np.log((g / 160) / (b / 40)) for b, g in zip(bad, good)))
    # WoE sign: the safer bin (more good) has the higher WoE
    assert woe[0] > woe[1]


def test_iv_band():
    assert [binning.iv_band(v) for v in (0.01, 0.05, 0.2, 0.4, 0.8)] == ["Not useful", "Weak", "Medium", "Strong", "Suspicious"]


def test_numeric_woe_monotone_and_min_share():
    x, y = _synth()
    sp = binning.fit_numeric_bins(x, y, min_share=0.05, max_bins=8)
    # higher x means more bad, so WoE must fall monotonically across bins
    assert np.all(np.diff(sp.woe) <= 0)
    shares = (sp.n_bad + sp.n_good) / len(x)
    assert shares.min() >= 0.05 and len(sp.woe) <= 8
    assert sp.iv > 0.1


def test_monotone_off_keeps_more_bins():
    rng = np.random.default_rng(3)
    x = rng.uniform(-3, 3, 30000)
    y = (rng.random(len(x)) < 1 / (1 + np.exp(-(-2 + 0.6 * x ** 2)))).astype(float)       # U shape
    mono = binning.fit_numeric_bins(x, y, monotone=True)
    free = binning.fit_numeric_bins(x, y, monotone=False)
    assert len(free.woe) > len(mono.woe) and free.iv > mono.iv


def test_bin_rule_left_open_right_closed_and_missing_bin():
    x = np.r_[np.arange(0, 100.0), np.full(20, np.nan)]
    y = np.r_[(np.arange(100) > 60).astype(float), np.zeros(20)]
    sp = binning.fit_numeric_bins(x, y, prebins=10, min_share=0.05, monotone=True)
    e = sp.edges[0]
    idx = sp.bin_index(np.array([e, e + 1e-9, np.nan, -5.0, 1e9]))
    assert idx[0] == 0 and idx[1] == 1               # a value equal to the edge belongs to the lower bin
    assert idx[2] == sp.n_bins - 1                   # missing has its own last bin
    assert idx[3] == 0 and idx[4] == len(sp.woe) - 1
    assert sp.transform(np.array([np.nan]))[0] == sp.missing_woe


def test_missing_bin_neutral_when_absent_in_training():
    x, y = _synth(5000)
    sp = binning.fit_numeric_bins(x, y)
    assert sp.missing_woe == 0.0 and sp.table().iloc[-1]["n"] == 0


def test_special_values_get_own_bin():
    x, y = _synth(8000)
    x = x.copy()
    x[:800] = -999.0
    y[:800] = 1.0
    sp = binning.fit_numeric_bins(x, y, special_values=(-999.0,))
    assert sp.n_bins == len(sp.woe) + 2
    assert sp.bin_index(np.array([-999.0]))[0] == len(sp.woe)
    assert sp.special_woe[0] < sp.woe.min()
    assert sp.edges.min() > -999.0


def test_categorical_grouping():
    rng = np.random.default_rng(5)
    cats = np.array(list("ABCDEFGHIJKLMNOP"))
    risk = np.linspace(0.02, 0.30, len(cats))
    c = rng.choice(cats, 30000).astype(object)
    y = (rng.random(len(c)) < risk[pd.Series(c).map({k: i for i, k in enumerate(cats)}).to_numpy()]).astype(float)
    c[:50] = "ZZ"                                    # tiny category must merge into a neighbour
    sp = binning.fit_categorical_bins(pd.Series(c).astype("category"), y, min_share=0.02, max_groups=6)
    assert len(sp.groups) <= 6
    members = [m for g in sp.groups for m in g]
    assert len(members) == len(set(members)) and "ZZ" in members
    assert ((sp.n_bad + sp.n_good) / len(c)).min() >= 0.02
    # unseen category falls in the largest group; NaN goes to the missing bin
    out = sp.bin_index(pd.Series(["A", "unseen", None]))
    assert out[1] == sp.default_group and out[2] == sp.n_bins - 1
    # groups are contiguous in risk order: group WoE must fall as risk rises
    assert np.all(np.diff(sp.woe) <= 1e-12)


def test_binning_report_columns():
    x, y = _synth(5000)
    cat = pd.Series(np.where(x > 0, "hi", "lo"))
    rep = binning.binning_report({"x": binning.fit_numeric_bins(x, y), "c": binning.fit_categorical_bins(cat, y)})
    for col in ("variable", "bin", "lo", "hi", "categories", "n", "bads", "bad_rate", "woe", "iv_contrib"):
        assert col in rep.columns
    assert rep.groupby("variable")["n"].sum().tolist() == [5000, 5000]


def test_pdo_identity_and_scaling():
    factor, offset = scorecard.scaling()
    assert factor == pytest.approx(config.PDO / np.log(2))
    score = lambda odds: offset + factor * np.log(odds)          # noqa: E731  odds = good:bad
    assert score(config.BASE_ODDS) == pytest.approx(config.BASE_SCORE)
    assert score(2 * config.BASE_ODDS) - score(config.BASE_ODDS) == pytest.approx(config.PDO)


@pytest.fixture(scope="module")
def fitted(loans, fixture_splits):
    tr = fixture_splits[C.DEV_TRAIN]
    feats = [C.INT_RATE, C.DTI, C.INQ_LAST_6MTHS, C.REVOL_UTIL, C.GRADE, C.PURPOSE, C.ANNUAL_INC, C.HOME_OWNERSHIP, C.TERM_M]
    return scorecard.fit_scorecard(tr, C.TARGET_BAD12, feats, iv_min=0.0), tr


def test_scorecard_signs_pvalues_and_vif(fitted):
    sc, _ = fitted
    assert sc.variables and all(b < 0 for b in sc.coefs.values())
    assert all(p <= config.P_MAX for p in sc.pvalues.values())
    assert set(sc.vif) == set(sc.variables)
    assert np.isfinite(sc.intercept)


def test_points_table_sums_to_score(fitted, fixture_splits):
    sc, _ = fitted
    d = fixture_splits[C.OOT1] if len(fixture_splits[C.OOT1]) else fixture_splits[C.DEV_HOLDOUT]
    table = sc.export_points_table()
    total = np.zeros(len(d))
    for v in sc.variables:
        t = table[table["variable"] == v].set_index("bin_id")["points"]
        total += t.reindex(sc.specs[v].bin_index(d[v])).to_numpy()
    assert total == pytest.approx(sc.points(d), abs=1e-6)
    assert sc.points_by_variable(d).sum(axis=1).to_numpy() == pytest.approx(sc.points(d), abs=1e-6)
    pd_ = sc.predict_pd(d)
    assert pd_ == pytest.approx(1 / (1 + np.exp(-sc.margin(d))))
    # score is the offset-factor transform of the PD odds
    odds_good = (1 - pd_) / pd_
    assert sc.points(d) == pytest.approx(sc.offset + sc.factor * np.log(odds_good), abs=1e-6)


def test_score_at_base_odds_equals_base_score(fitted):
    sc, _ = fitted
    # a margin of -ln(BASE_ODDS) is odds good:bad of BASE_ODDS
    assert sc.offset - sc.factor * (-np.log(config.BASE_ODDS)) == pytest.approx(config.BASE_SCORE)


def test_elimination_drops_correlated_weaker_and_leaky_not_allowed(loans, fixture_splits):
    tr = fixture_splits[C.DEV_TRAIN]
    sc = scorecard.fit_scorecard(tr, C.TARGET_BAD12, [C.INT_RATE, C.GRADE, C.DTI, C.REVOL_UTIL], iv_min=0.0)
    kept_pair = {C.INT_RATE, C.GRADE} & set(sc.variables)
    assert len(kept_pair) <= 1
    assert any(v in (C.INT_RATE, C.GRADE) and "corr" in r for v, r in sc.dropped)


def test_scorecard_ranks_on_holdout(fitted, fixture_splits):
    from sklearn.metrics import roc_auc_score
    sc, _ = fitted
    d = fixture_splits[C.OOT2]       # largest out-of-sample fixture split
    assert roc_auc_score(d[C.TARGET_BAD12], sc.predict_pd(d)) > 0.55


def test_repro_dummy_design_shape(loans):
    X = repro_original.build_dummy_design(loans)
    assert not any(c.endswith(":G") or c == "term:60" for c in X.columns if c.startswith(("grade", "term")))
    assert X.shape[0] == len(loans) and X.dtypes.eq(np.float32).all()
    assert "mths_since_issue_d:<38" in X.columns
    assert not any(c.startswith("mths_since_issue_d") for c in repro_original.build_dummy_design(loans, with_issue_age=False).columns)
    # each ladder is one-hot: the dti dummies plus the dropped reference cover every row with a dti
    dti_cols = [c for c in X.columns if c.startswith("dti:")]
    ok = loans[C.DTI].notna().to_numpy()
    assert (X.loc[ok, dti_cols].sum(axis=1) <= 1).all()


@pytest.mark.slow
def test_repro_runs_on_real_data():
    res = repro_original.reproduce_original_pd()
    assert 0.6 < res["without_mths_since_issue_d"]["auc"] < res["with_mths_since_issue_d"]["auc"] < 0.8
