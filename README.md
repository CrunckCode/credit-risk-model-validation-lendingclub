# Credit Risk Model Validation on LendingClub Data

**Data provenance and status.** All results use the public LendingClub loan data for loans issued 2007-06-01 to 2014-12-01 (466,285 loans); LendingClub is the data source. This is an **independent validation exercise on public data and not a bank-approved or production model**. The validator is also the author of the original course project, so independence is procedural only. Raw data are not included in the repository (see the quick start).

## What it does

It validates and rebuilds the PD, LGD, EAD and expected-loss models of an earlier Master's course project in the style of SR 11-7: a fixed 12-month default definition on time-ordered samples, WoE scorecards scaled to points, gradient-boosted challengers, a full validation battery with traffic lights, a benchmark against LendingClub's own sub-grade, PSI and CSI stability, LGD, EAD and expected-loss back-tests, sensitivity analysis, SHAP, a findings log and an Excel workbook with live formulas. The full write-up, with theory and worked examples, is `docs/Credit_Risk_Model_Validation.docx` (source `docs/Credit_Risk_Model_Validation.md`).

## Quick start

```
cd python
py -3 -m pip install -r requirements.txt
set CV_RAW_DIR=C:\path\to\folder\with\loan_data_2007_2014.csv
py -3 -m credit_validation.cli --quick     # small run
py -3 -m credit_validation.cli             # full run, writes python/outputs/
py -3 -m pytest -q
cd ..
py -3 docs/build_docs.py                   # rebuild the document and this README
```

The folder in `CV_RAW_DIR` must contain `loan_data_2007_2014.csv` and `loan_data_defaults.csv` (not redistributed; obtain the LendingClub public data and place the files there).

## Headline results

| Model | AUC 2013 | Gini 2013 | KS 2013 | AUC 2014 | Gini 2014 | KS 2014 |
|:-----------------|-------:|--------:|------:|-------:|--------:|------:|
| SC_FULL scorecard | 0.676 | 0.353 | 0.254 | 0.673 | 0.346 | 0.249 |
| SC_INDEP scorecard | 0.647 | 0.295 | 0.208 | 0.633 | 0.266 | 0.192 |
| XGBoost | 0.688 | 0.375 | 0.280 | 0.687 | 0.374 | 0.274 |
| LightGBM | 0.683 | 0.366 | 0.266 | 0.683 | 0.367 | 0.269 |

Table: Out-of-time discrimination (2013 and 2014 issues). Development holdout Gini for SC_FULL is 0.372.

- The scorecard discriminates modestly (Gini 0.353 and 0.346, Amber) and is stable (score PSI 0.0026 and 0.0067).
- It does **not** beat LendingClub's own sub-grade as a score (AUC difference +0.0029 in 2013, -0.0102 in 2014), and the independent scorecard without grade and interest rate falls to Gini 0.295 and 0.266: the grade and interest-rate circularity.
- Hosmer-Lemeshow is Red out of time (a large-sample power effect, but the riskiest decile is over-predicted) and the 2014 grade binomial test is Red for grades E, F and G.
- Gradient boosting adds about one AUC point and does not fix calibration.
- LGD stage 1 and EAD are weak (stage 1 AUC 0.572, CCF R-squared 0.091 out of time); 2014 recoveries are immature. The 12-month expected loss for 2013 is USD 59.9 million against realized USD 67.2 million (ratio 0.892).
- The reproduced original model includes a loan-age proxy that drifts mechanically (score PSI 0.912 in 2014).
- 34 findings are logged (16 High, 16 Medium, 2 Low), including defects found in the original notebooks: In Grace Period coded good against the project notes, hard 0/1 labels in LGD stage 1, EAD summary printing LGD p-values, invalid Wald p-values, a vintage proxy feature, silent zero-fill of missing dummies, random-only splits and a lifetime-PD expected loss over all loans.

## Key findings

See section 18 of the document for the findings log (all High findings in full). Remediation of each defect is implemented in this project.

## Excel workbook

`excel/credit_validation_workbook.xlsx` re-implements the scorecard, deciles, calibration, PSI and expected-loss calculations with live formulas on 5,000 stratified 2013 loans. Overall result: **all comparisons passed**. The reconciliation ran 2 scenario cases.

## Repository layout

```
data/raw, data/interim     raw files (not committed) and parquet cache
docs/                      document (md, docx), build_docs.py, reference.docx
excel/                     workbook builder and workbook
python/credit_validation/  package: data, targets, splits, features, binning, scorecard,
                           repro_original, stats_models, models, validation, stability,
                           benchmark, lgd_ead, expected_loss, sensitivity, explain,
                           charts, findings, reconcile_excel, cli
python/tests/              pytest suites
python/outputs/            results (csv, json), charts/, models/, excel_inputs/
```

## Limitations

One lender, 2007 to 2014 only; default timing proxied by the last payment date; 2014 outcomes and recoveries partly unresolved; no bureau score; generic thresholds; large-sample tests reject small gaps; same-author validation. See section 19.

## Next steps

Recalibrate by grade on recent vintages; obtain true default dates and bureau scores; build LGD on mature cohorts with a development-pattern adjustment; use heteroskedasticity-robust errors in LGD and EAD; add macroeconomic scenarios; have the model independently reviewed by a different person.
