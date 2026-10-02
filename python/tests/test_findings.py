import pandas as pd
import pytest

from credit_validation import config, findings

# ===== CONFIG (user inputs) =====
ALLOWED_SEVERITY = {"High", "Medium", "Low"}
ALLOWED_OWNER = {"Model Developer", "Model Owner", "Validation"}
EM_DASH = chr(0x2014)
# ===== END CONFIG =====


def _hand_results():
    return pd.DataFrame([
        dict(test="gini", model="sc_full", split="oot1", value=0.21, threshold=0.30, light="Red"),
        dict(test="ks", model="sc_full", split="oot1", value=0.25, threshold=0.30, light="Amber"),
        dict(test="psi", model="sc_full", split="oot2", value=0.02, threshold=0.10, light="Green"),
    ])


def test_auto_findings_maps_red_high_amber_medium():
    a = findings.auto_findings(_hand_results())
    assert len(a) == 2
    sev = dict(zip(a["title"], a["severity"]))
    assert sev["gini Red for sc_full oot1"] == "High"
    assert sev["ks Amber for sc_full oot1"] == "Medium"
    assert "0.21" in a.loc[a["severity"] == "High", "evidence"].iloc[0]
    assert set(a["source"]) == {"auto"}


def test_auto_findings_empty_inputs():
    assert findings.auto_findings(pd.DataFrame()).empty
    assert findings.auto_findings(_hand_results().assign(light="Green")).empty


def test_auto_findings_groups_rows_of_same_test():
    r = pd.DataFrame([dict(test="binom", model="m", split="oot1", value=0.2, threshold=0.05, light="Amber"),
                      dict(test="binom", model="m", split="oot1", value=1e-6, threshold=0.0001, light="Red")])
    a = findings.auto_findings(r)
    assert len(a) == 1 and a["severity"].iloc[0] == "High" and "1e-06" in a["evidence"].iloc[0]


def test_build_log_schema_and_ordering(tmp_path):
    _hand_results().to_csv(tmp_path / "validation_results.csv", index=False)
    log = findings.build_log(tmp_path)
    assert list(log.columns) == list(findings.LOG_COLUMNS)
    assert (tmp_path / findings.LOG_FILE).exists()
    assert log["finding_id"].is_unique and log["finding_id"].str.match(r"^F-\d{2,}$").all()
    assert set(log["severity"]) <= ALLOWED_SEVERITY and set(log["owner"]) <= ALLOWED_OWNER
    assert set(log["status"]) == {"Open"} and set(log["source"]) == {"auto", "review"}
    assert log["severity"].map(findings.SEVERITY_ORDER).is_monotonic_increasing
    assert (log["evidence"].str.len() > 0).all()
    assert not log.astype(str).apply(lambda c: c.str.contains(EM_DASH)).any().any()


def test_review_findings_static_fields():
    assert len(findings.REVIEW_FINDINGS) >= 12
    for f in findings.REVIEW_FINDINGS:
        assert f["severity"] in ALLOWED_SEVERITY and f["owner"] in ALLOWED_OWNER and f["evidence"]
        assert EM_DASH not in " ".join(str(v) for v in f.values())


def test_evidence_files_exist_when_outputs_exist():
    out = config.OUT_DIR
    if not (out / "validation_results.csv").exists():
        pytest.skip("pipeline outputs not generated yet")
    missing = [f["evidence_file"] for f in findings.REVIEW_FINDINGS if f["evidence_file"] and not (out / f["evidence_file"]).exists()]
    assert not missing


def test_auto_findings_one_finding_per_test_model_across_splits():
    r = pd.DataFrame([dict(test="cal_slope", model="sc_full", split=s, value=v, threshold="x", light="Red")
                      for s, v in (("dev_holdout", 0.77), ("oot1", 0.76), ("oot2", 0.74))])
    a = findings.auto_findings(r)
    assert len(a) == 1 and "oot2: 0.74" in a["evidence"].iloc[0] and a["area"].iloc[0] == "Calibration"


def test_scope_results_drops_challengers_and_in_sample():
    r = pd.DataFrame([dict(test="gini", model=m, split=s, value=0.2, threshold="", light="Red")
                      for m in ("sc_full", "xgb") for s in ("dev_train", "oot1")])
    kept = findings.scope_results(r)
    assert list(zip(kept["model"], kept["split"])) == [("sc_full", "oot1")]


def test_el_findings_flags_large_gap(tmp_path):
    pd.DataFrame({"vintage": [2013], "grade": ["ALL"], "el": [1.5e6], "realized_loss": [1.0e6], "sample": ["out_of_time"], "status": [""]}
                 ).to_csv(tmp_path / "el_backtest_12m.csv", index=False)
    a = findings.el_findings(tmp_path)
    assert len(a) == 1 and a["severity"].iloc[0] == "High" and "1.500" in a["evidence"].iloc[0]
