import json

import numpy as np
import pandas as pd
import pytest

from credit_validation import config, data, expected_loss as EL, lgd_ead as L, models, sensitivity
from credit_validation import columns as C


@pytest.fixture(autouse=True)
def isolate_outputs(tmp_path, monkeypatch):
    # tests must never write into the pipeline's real output or model folders
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "EXCEL_INPUT_DIR", tmp_path / "out" / "excel_inputs")


@pytest.fixture(scope="module")
def defaults(loans):
    return L.default_population(loans)


@pytest.fixture(scope="module")
def champs(fixture_splits):
    return models.fit_champions(fixture_splits)


@pytest.fixture(scope="module")
def lgd_model(defaults):
    return L.fit_lgd_ead(L.split_defaults(defaults)[L.SPLIT_DEV])


def test_default_population_and_reconciliation(loans, defaults, raw_dir):
    assert defaults[C.LOAN_STATUS].isin(config.CHARGEOFF_STATUSES).all()
    assert len(defaults) == loans[C.LOAN_STATUS].isin(config.CHARGEOFF_STATUSES).sum()
    rec = L.reconcile_defaults(defaults, data.load_defaults_reference(raw_dir))
    assert rec["n_matched"] == rec["n_recomputed"] == rec["n_reference"]
    assert rec["max_abs_diff_recovery_rate"] < 1e-6 and rec["max_abs_diff_ccf"] < 1e-6


def test_dev_oot_split_by_issue_year(defaults):
    parts = L.split_defaults(defaults)
    assert (parts[L.SPLIT_DEV][C.ISSUE_D] <= L.DEV_LAST_ISSUE).all()
    assert (parts[L.SPLIT_OOT][C.ISSUE_D].dt.year == 2013).all() and (parts[L.SPLIT_2014][C.ISSUE_D].dt.year == 2014).all()


def test_lgd_in_unit_interval_and_combination_identity(lgd_model, defaults):
    r = lgd_model.predict(defaults)
    for c in (C.PRED_LGD, "lgd_hard", "p_recover", "rr_if_recover", "ccf_hat"):
        assert r[c].between(0, 1).all(), c
    np.testing.assert_allclose(r[C.PRED_LGD], np.clip(1 - r["p_recover"] * r["rr_if_recover"], 0, 1))
    assert (r[C.PRED_EAD] <= defaults[C.FUNDED_AMNT] + 1e-9).all() and (r[C.PRED_EAD] >= 0).all()
    # the hard-class version can only be as good as always predicting the stage-2 amount or zero
    assert set(np.unique(np.round((1 - r["lgd_hard"]) / r["rr_if_recover"].replace(0, np.nan), 9).dropna())) <= {0.0, 1.0}


def test_stage_summaries_have_pvalues(lgd_model):
    s1, s2, e = lgd_model.stage1.summary(), lgd_model.stage2.summary(), lgd_model.ead.summary()
    for t in (s1, s2, e):
        assert t["term"].iloc[0] == "const" and t["p_value"].between(0, 1).all()
    assert lgd_model.stage2.dof_ == lgd_model.n_fit["stage2"] - (len(s2) - 1) - 1


def test_el_identity(lgd_model, champs, fixture_splits):
    d = fixture_splits[C.OOT2]
    s = EL.score_el(d, champs["sc_full"].predict_pd(d), lgd_model)
    np.testing.assert_allclose(s[C.PRED_EL], s[C.PRED_PD] * s[C.PRED_LGD] * s[C.PRED_EAD])
    assert (s[C.PRED_EL] >= 0).all()


def test_backtest_totals_equal_sum_net_loss(lgd_model, champs, fixture_splits):
    bt = EL.el_backtest_12m(fixture_splits, champs["sc_full"], lgd_model)
    allrows = bt[bt[C.GRADE] == EL.ALL_LABEL]
    frames = [fixture_splits[s] for s in (C.DEV_TRAIN, C.DEV_HOLDOUT, C.OOT1, C.OOT2)]
    df = pd.concat(frames)
    df = df[(df[C.VINTAGE] >= 2010) & (df[C.VINTAGE] <= 2014)]
    expected = float((df[C.NET_LOSS] * df[C.TARGET_BAD12]).sum())
    assert allrows["realized_loss"].sum() == pytest.approx(expected)
    assert expected == pytest.approx(bt.attrs["total_realized"])
    by_grade = bt[bt[C.GRADE] != EL.ALL_LABEL]
    assert by_grade["realized_loss"].sum() == pytest.approx(expected)
    assert by_grade["el"].sum() == pytest.approx(allrows["el"].sum())
    assert by_grade["n"].sum() == len(df)
    assert (bt.loc[bt[C.VINTAGE] == 2014, "status"] != "").all()


def test_lifetime_backtest_structure(loans, fixture_splits, defaults):
    life = EL.el_backtest_lifetime(fixture_splits[C.COMPLETE36], L.split_defaults(defaults)[L.SPLIT_DEV])
    assert {"train_2007_2011 (in-sample)", "test_2012"} <= set(life["sample"])
    t = life[(life["sample"] == "test_2012") & (life[C.GRADE] == EL.ALL_LABEL)].iloc[0]
    assert t["realized_loss"] == pytest.approx(fixture_splits[C.COMPLETE36].query("issue_d.dt.year == 2012")[C.NET_LOSS].sum())
    assert life.iloc[-1]["el_pct_funded"] == EL.ORIGINAL_EL_SHARE


def test_run_lgd_ead_writes_outputs_to_redirected_dirs(loans, raw_dir, tmp_path):
    res = L.run_lgd_ead(loans, out_dir=tmp_path / "o", raw_dir=raw_dir)
    for f in L.OUT_FILES.values():
        assert (tmp_path / "o" / f).exists(), f
    summ = json.loads((tmp_path / "o" / L.OUT_FILES["summary"]).read_text())
    assert summ["data_source"] == config.DATA_SOURCE and summ["reconciliation"]["n_matched"] == len(res["defaults"])
    assert (tmp_path / "models" / "lgd_ead.joblib").exists()
    assert not (config.OUT_DIR / L.OUT_FILES["summary"]).exists()


def test_excel_inputs(lgd_model, champs, fixture_splits, tmp_path):
    out = EL.write_excel_inputs(champs, fixture_splits, lgd_model, out_dir=tmp_path / "x", quick=True)
    ls = pd.read_csv(tmp_path / "x" / "excel_inputs" / EL.OUT_SAMPLE, float_precision="round_trip")
    sc = champs["sc_full"].scorecard
    oot1 = fixture_splits[C.OOT1]
    assert len(ls) <= len(oot1) and ls[C.LOAN_ID].isin(oot1[C.LOAN_ID]).all()
    assert set(sc.variables) <= set(ls.columns)
    for c in (C.TARGET_BAD12, C.FUNDED_AMNT, C.NET_LOSS, C.GRADE, C.SUB_GRADE, C.PRED_PD, C.SCORE, C.PRED_LGD, C.PRED_EAD, C.PRED_EL):
        assert c in ls.columns
    np.testing.assert_allclose(ls[C.PRED_EL], ls[C.PRED_PD] * ls[C.PRED_LGD] * ls[C.PRED_EAD], rtol=1e-12)
    # full precision: numeric attributes survive the csv round trip exactly (float32 values widened to float64)
    src = oot1.set_index(C.LOAN_ID)
    num = [c for c in sc.variables if pd.api.types.is_numeric_dtype(oot1[c])][0]
    exp = src.loc[ls[C.LOAN_ID], num].astype("float64").to_numpy()
    np.testing.assert_array_equal(ls[num].to_numpy(), exp)
    # scores in the file reproduce the model
    np.testing.assert_allclose(ls[C.PRED_PD], champs["sc_full"].predict_pd(src.loc[ls[C.LOAN_ID]].reset_index()), rtol=1e-12)
    pts = [c for c in ls.columns if c.startswith("pts_")]
    np.testing.assert_allclose(ls[pts].sum(axis=1), ls[C.SCORE], atol=1e-9)
    bins = pd.read_csv(tmp_path / "x" / "excel_inputs" / EL.OUT_PSI_BINS)
    assert bins["dev_share"].sum() == pytest.approx(1.0) and bins["lo"].iloc[0] == -np.inf
    rl = pd.read_csv(tmp_path / "x" / "excel_inputs" / EL.OUT_REALIZED)
    assert rl["oot1_realized"].sum() == pytest.approx((oot1[C.NET_LOSS] * oot1[C.TARGET_BAD12]).sum())


def test_sensitivity_structure(champs, fixture_splits, loans, lgd_model, tmp_path):
    out = sensitivity.run_sensitivity(champs, fixture_splits, loans=loans, lgd_model=lgd_model, out_dir=tmp_path / "s", models=("sc_full",))
    assert list(out.columns) == sensitivity.COLUMNS
    assert {"shock_combined", "pd_x1.25", "pd_x1.5", "lgd_add_0.1", "target_24m_window", "target_in_grace_bad", "population_dnmcp_included",
            "binning_min_share_3%", "binning_max_bins_12"} <= set(out["scenario"])
    m = out.set_index("scenario")
    assert m.loc["pd_x1.25", "delta_el_pct"] == pytest.approx(25.0) and m.loc["pd_x1.5", "delta_pd_pct"] == pytest.approx(50.0, abs=0.5)
    assert m.loc["lgd_add_0.1", "delta_pd_pct"] == 0.0 and m.loc["lgd_add_0.1", "delta_el_pct"] > 0
    assert (tmp_path / "s" / "sensitivity.csv").exists()
    assert config.EXCLUDE_DNMCP is True


@pytest.mark.slow
def test_real_data_reconciles_to_original_defaults():
    from credit_validation import features, targets
    loans = features.build_features(targets.add_targets(data.load_loans()))
    rec = L.reconcile_defaults(L.default_population(loans), data.load_defaults_reference())
    assert rec["n_recomputed"] == rec["n_reference"] == rec["n_matched"] == 43236
    assert rec["max_abs_diff_recovery_rate"] < 1e-9
