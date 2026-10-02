"""End-to-end pipeline: py -3 -m credit_validation.cli [--quick] [--skip-excel]"""
import json
import logging
import sys
import time

from . import (charts, config, data, expected_loss, explain, features, findings, lgd_ead, models, repro_original,
               sensitivity, splits as split_mod, targets, validation)
from . import columns as C

# ===== CONFIG (user inputs) =====
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
# ===== END CONFIG =====


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    quick = "--quick" in argv
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    t0 = time.time()
    step = lambda msg: logging.info("[%5.0fs] %s", time.time() - t0, msg)
    out = config.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    step("loading and preparing data")
    loans = features.build_features(targets.add_targets(data.load_loans(refresh=False)))
    targets.target_audit(loans).to_csv(out / "target_audit.csv", index=False)
    splits = split_mod.time_splits(loans)
    (out / "data_summary.json").write_text(json.dumps(dict(
        data_source=config.DATA_SOURCE, n_loans=len(loans), quick=quick,
        split_rows={k: len(v) for k, v in splits.items()}), indent=1))

    step("reproducing the original model")
    repro = repro_original.reproduce_original_pd(splits)
    (out / "repro_original.json").write_text(json.dumps({"data_source": config.DATA_SOURCE, **repro}, indent=1, default=float))

    step("training models")
    bundles = models.train_all(splits, quick=quick)
    step("validation battery")
    validation.run_battery(bundles, splits)
    step("LGD, EAD and expected loss")
    lgd = lgd_ead.run_lgd_ead(loans)
    expected_loss.run_el(splits, bundles, lgd)
    expected_loss.write_excel_inputs(bundles, splits, lgd["model"], quick=quick)
    step("sensitivity")
    sensitivity.run_sensitivity(bundles, splits, loans=loans, lgd_model=lgd["model"])
    step("explainability")
    explain.run_explain(bundles["lgbm"], splits[C.OOT1])
    step("findings log and charts")
    findings.build_log()
    charts.make_all_charts(bundles=bundles, splits=splits)
    if "--skip-excel" not in argv:
        step("excel workbook and reconciliation")
        from . import reconcile_excel
        reconcile_excel.build_and_reconcile(quick=quick)
    step("done")


if __name__ == "__main__":
    main()
