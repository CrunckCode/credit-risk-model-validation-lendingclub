"""Two-stage LGD and CCF-based EAD models fitted on the charged-off population, with diagnostics.

Design matrix (documented choice): application-time numerics winsorized at the development 1st/99th percentile,
log1p for income and revolving balance, median-imputed and z-scored with development statistics; months-since
variables enter as "is reported" flags only (too sparse to use as levels); categoricals are dummied with the most
frequent development level as reference and levels under RARE_SHARE pooled. Grade is left out because int_rate
already carries it (near-collinear), addr_state is left out (50 sparse levels) and initial_list_status is left out
(zero share before 2012, a structural break). All fit on development defaults only.
Recovery rate is recoveries / funded (the original definition); LGD = 1 - recovery rate.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan

from . import config
from . import columns as C
from . import data
from .stats_models import LogisticRegressionWithPValues, LinearRegressionWithPValues, align_columns

# ===== CONFIG (user inputs) =====
DEV_LAST_ISSUE = pd.Timestamp("2012-12-01")
OOT_YEAR, REPORT_YEAR = 2013, 2014
SPLIT_DEV, SPLIT_OOT, SPLIT_2014 = "dev_le_2012", "oot_2013", "report_2014"
NUMERIC = (C.LOAN_AMNT, C.TERM_M, C.INT_RATE, C.EMP_YEARS, C.ANNUAL_INC, C.DTI, C.DELINQ_2YRS, C.INQ_LAST_6MTHS, C.OPEN_ACC,
           C.PUB_REC, C.REVOL_BAL, C.REVOL_UTIL, C.TOTAL_ACC, C.CREDIT_AGE_M, C.LOAN_TO_INC)
LOG1P = (C.ANNUAL_INC, C.REVOL_BAL)
REPORTED_FLAGS = (C.MTHS_SINCE_LAST_DELINQ, C.MTHS_SINCE_LAST_RECORD)
DUMMIES = (C.HOME_OWNERSHIP, C.VERIFICATION_STATUS, C.PURPOSE)
WINSOR = (0.01, 0.99)
RARE_SHARE, RARE_LABEL = 0.01, "rare"
HARD_CLASS_CUTOFF = 0.5
RESIDUAL_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
N_DECILES = 10
RECON_ID_SAMPLE = 5
LGD_ACTUAL = "lgd_actual"
OUT_FILES = dict(stage1="lgd_stage1_coefs.csv", stage2="lgd_stage2_coefs.csv", ead="ead_coefs.csv", deciles="lgd_ead_deciles.csv",
                 diag="lgd_ead_diagnostics.csv", grade="lgd_ead_by_grade.csv", resid="lgd_ead_residuals.csv",
                 vintage="recovery_by_default_vintage.csv", summary="lgd_ead_summary.json")
# ===== END CONFIG =====


def default_population(loans: pd.DataFrame) -> pd.DataFrame:
    """Charged-off loans (incl. the DNMCP charged-off status), recomputed from the raw statuses."""
    d = loans[loans[C.LOAN_STATUS].isin(config.CHARGEOFF_STATUSES)].copy()
    d[LGD_ACTUAL] = 1.0 - d[C.RECOVERY_RATE]
    return d


def reconcile_defaults(defaults: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """Compare the recomputed population with the original notebook's loan_data_defaults.csv."""
    m = defaults[[C.LOAN_ID, C.RECOVERY_RATE, C.CCF]].merge(ref, on=C.LOAN_ID, how="inner", suffixes=("", "_ref"))
    return dict(n_recomputed=int(len(defaults)), n_reference=int(len(ref)), n_matched=int(len(m)),
                n_only_recomputed=int(len(set(defaults[C.LOAN_ID]) - set(ref[C.LOAN_ID]))),
                n_only_reference=int(len(set(ref[C.LOAN_ID]) - set(defaults[C.LOAN_ID]))),
                max_abs_diff_recovery_rate=float((m[C.RECOVERY_RATE] - m[C.RECOVERY_RATE + "_ref"]).abs().max()),
                max_abs_diff_ccf=float((m[C.CCF] - m[C.CCF + "_ref"]).abs().max()),
                mean_recovery_rate=float(defaults[C.RECOVERY_RATE].mean()), mean_recovery_rate_ref=float(ref[C.RECOVERY_RATE].mean()),
                mean_ccf=float(defaults[C.CCF].mean()), mean_ccf_ref=float(ref[C.CCF].mean()))


def split_defaults(defaults: pd.DataFrame) -> dict:
    issue = defaults[C.ISSUE_D]
    return {SPLIT_DEV: defaults[issue <= DEV_LAST_ISSUE], SPLIT_OOT: defaults[issue.dt.year == OOT_YEAR],
            SPLIT_2014: defaults[issue.dt.year == REPORT_YEAR]}


@dataclass
class DesignSpec:
    """Fitted on development defaults; transform() reproduces the identical design for any frame."""
    medians: dict = field(default_factory=dict)
    bounds: dict = field(default_factory=dict)
    mean_std: dict = field(default_factory=dict)
    levels: dict = field(default_factory=dict)
    reference: dict = field(default_factory=dict)
    columns: list = field(default_factory=list)

    @staticmethod
    def _numeric(df, v):
        x = df[v].astype("float64")
        return np.log1p(x.clip(lower=0)) if v in LOG1P else x

    def fit(self, df):
        for v in NUMERIC:
            x = self._numeric(df, v)
            self.medians[v] = float(x.median())
            lo, hi = x.quantile(WINSOR[0]), x.quantile(WINSOR[1])
            self.bounds[v] = (float(lo), float(hi))
            z = x.fillna(self.medians[v]).clip(lo, hi)
            self.mean_std[v] = (float(z.mean()), float(z.std(ddof=0)) or 1.0)
        for v in DUMMIES:
            share = df[v].astype(str).value_counts(normalize=True)
            keep = [k for k in share.index if share[k] >= RARE_SHARE]
            self.levels[v] = keep
            self.reference[v] = share.index[0]
        self.columns = list(self.transform(df).columns)
        return self

    def transform(self, df):
        out = {}
        for v in NUMERIC:
            lo, hi = self.bounds[v]
            m, s = self.mean_std[v]
            out[v] = ((self._numeric(df, v).fillna(self.medians[v]).clip(lo, hi) - m) / s).to_numpy()
        for v in REPORTED_FLAGS:
            out[f"{v}_reported"] = df[v].notna().astype("float64").to_numpy()
        for v in DUMMIES:
            s = df[v].astype(str).where(df[v].astype(str).isin(self.levels[v]), RARE_LABEL)
            for k in self.levels[v]:
                if k != self.reference[v]:
                    out[f"{v}={k}"] = (s == k).astype("float64").to_numpy()
        return pd.DataFrame(out, index=df.index)


@dataclass
class LgdEadModel:
    spec: DesignSpec
    stage1: object
    stage2: object
    ead: object
    n_fit: dict

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = align_columns(self.spec.transform(df), self.spec.columns).to_numpy()
        p1 = self.stage1.predict_proba(X)[:, 1]
        rr = np.clip(self.stage2.predict(X), 0.0, 1.0)
        ccf = np.clip(self.ead.predict(X), 0.0, 1.0)
        out = pd.DataFrame({"p_recover": p1, "rr_if_recover": rr, C.PRED_LGD: np.clip(1.0 - p1 * rr, 0.0, 1.0),
                            "lgd_hard": 1.0 - (p1 > HARD_CLASS_CUTOFF) * rr, "ccf_hat": ccf}, index=df.index)
        out[C.PRED_EAD] = ccf * df[C.FUNDED_AMNT].to_numpy(dtype=np.float64)
        return out


def fit_lgd_ead(defaults: pd.DataFrame, last_issue=None) -> LgdEadModel:
    """Stage 1 logit on any-recovery, stage 2 OLS on recovery rate given recovery, OLS on CCF; development defaults only."""
    d = defaults if last_issue is None else defaults[defaults[C.ISSUE_D] <= last_issue]
    spec = DesignSpec().fit(d)
    X = spec.transform(d)
    s1 = LogisticRegressionWithPValues().fit(X, d[C.RECOVERED_ANY].to_numpy())
    pos = (d[C.RECOVERY_RATE] > 0).to_numpy()
    s2 = LinearRegressionWithPValues().fit(X[pos], d.loc[pos, C.RECOVERY_RATE].to_numpy())
    ead = LinearRegressionWithPValues().fit(X, d[C.CCF].to_numpy())
    return LgdEadModel(spec, s1, s2, ead, dict(stage1=len(d), stage2=int(pos.sum()), ead=len(d)))


# ---------- diagnostics ----------
def reg_metrics(actual, pred) -> dict:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    res = a - p
    sst = float(((a - a.mean()) ** 2).sum())
    return dict(n=len(a), r2=1.0 - float((res ** 2).sum()) / sst if sst > 0 else np.nan, rmse=float(np.sqrt((res ** 2).mean())),
                mae=float(np.abs(res).mean()), corr=float(np.corrcoef(a, p)[0, 1]) if a.std() > 0 and p.std() > 0 else np.nan,
                spearman=float(stats.spearmanr(a, p)[0]), mean_actual=float(a.mean()), mean_pred=float(p.mean()))


def breusch_pagan(model) -> dict:
    lm, lm_p, f, f_p = het_breuschpagan(model.resid_, model.design_)
    return dict(lm=float(lm), lm_p=float(lm_p), f=float(f), f_p=float(f_p))


def _decile_rows(actual, pred, comp, split, n=N_DECILES):
    order = np.argsort(-np.asarray(pred, float), kind="stable")
    a, p = np.asarray(actual, float)[order], np.asarray(pred, float)[order]
    d = (np.arange(len(a)) * n) // len(a)
    return [dict(component=comp, split=split, decile=k + 1, n=int((d == k).sum()), mean_pred=float(p[d == k].mean()),
                 mean_actual=float(a[d == k].mean())) for k in range(n) if (d == k).any()]


def build_diagnostics(model: LgdEadModel, parts: dict) -> dict:
    diag, decs, grades, resid = [], [], [], []
    from sklearn.metrics import roc_auc_score
    for split, d in parts.items():
        if d.empty:
            continue
        pr = model.predict(d)
        actual_lgd = d[LGD_ACTUAL].to_numpy()
        comps = {"lgd_expected": (actual_lgd, pr[C.PRED_LGD].to_numpy()), "lgd_hard_class": (actual_lgd, pr["lgd_hard"].to_numpy()),
                 "ccf": (d[C.CCF].to_numpy(), pr["ccf_hat"].to_numpy()), "ead_usd": (d[C.CCF].to_numpy() * d[C.FUNDED_AMNT].to_numpy(), pr[C.PRED_EAD].to_numpy())}
        pos = (d[C.RECOVERY_RATE] > 0).to_numpy()
        if pos.sum() > 2:
            comps["recovery_rate_given_recovery"] = (d.loc[pos, C.RECOVERY_RATE].to_numpy(), pr.loc[pos, "rr_if_recover"].to_numpy())
        for comp, (a, p) in comps.items():
            for k, v in reg_metrics(a, p).items():
                diag.append(dict(split=split, component=comp, metric=k, value=v))
            if comp != "ead_usd":
                for q in RESIDUAL_QUANTILES:
                    resid.append(dict(split=split, component=comp, quantile=q, residual=float(np.quantile(a - p, q))))
        y1 = d[C.RECOVERED_ANY].to_numpy()
        diag += [dict(split=split, component="stage1_any_recovery", metric="auc", value=float(roc_auc_score(y1, pr["p_recover"]))
                      if 0 < y1.sum() < len(y1) else np.nan),
                 dict(split=split, component="stage1_any_recovery", metric="hit_rate_at_0.5", value=float(((pr["p_recover"] > HARD_CLASS_CUTOFF) == (y1 == 1)).mean())),
                 dict(split=split, component="stage1_any_recovery", metric="mean_pred", value=float(pr["p_recover"].mean())),
                 dict(split=split, component="stage1_any_recovery", metric="mean_actual", value=float(y1.mean()))]
        decs += _decile_rows(actual_lgd, pr[C.PRED_LGD], "lgd_expected", split) + _decile_rows(d[C.CCF], pr["ccf_hat"], "ccf", split)
        g = pd.DataFrame({C.GRADE: d[C.GRADE].astype(str).to_numpy(), "lgd_pred": pr[C.PRED_LGD].to_numpy(), "lgd_actual": actual_lgd,
                          "ccf_pred": pr["ccf_hat"].to_numpy(), "ccf_actual": d[C.CCF].to_numpy()}).groupby(C.GRADE).agg(
            n=("lgd_pred", "size"), lgd_pred=("lgd_pred", "mean"), lgd_actual=("lgd_actual", "mean"),
            ccf_pred=("ccf_pred", "mean"), ccf_actual=("ccf_actual", "mean")).reset_index()
        g.insert(0, "split", split)
        grades.append(g)
    return dict(diag=pd.DataFrame(diag), deciles=pd.DataFrame(decs), grade=pd.concat(grades, ignore_index=True), resid=pd.DataFrame(resid))


def recovery_by_default_vintage(defaults: pd.DataFrame) -> pd.DataFrame:
    """Default vintage proxied by the year of the last payment (no default date in the file); shows recovery immaturity."""
    yr = defaults[C.LAST_PYMNT_D].dt.year.fillna(defaults[C.ISSUE_D].dt.year)
    g = defaults.assign(default_year=yr.astype(int)).groupby("default_year")
    return g.agg(n=(C.LOAN_ID, "size"), any_recovery_share=(C.RECOVERED_ANY, "mean"), mean_recovery_rate=(C.RECOVERY_RATE, "mean"),
                 mean_ccf=(C.CCF, "mean")).reset_index()


def run_lgd_ead(loans: pd.DataFrame, out_dir=None, raw_dir=None, ref: pd.DataFrame = None, save_model: bool = True) -> dict:
    out = Path(out_dir or config.OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    defaults = default_population(loans)
    ref = data.load_defaults_reference(raw_dir) if ref is None else ref
    recon = reconcile_defaults(defaults, ref)
    parts = split_defaults(defaults)
    model = fit_lgd_ead(parts[SPLIT_DEV])
    model.stage1.summary().to_csv(out / OUT_FILES["stage1"], index=False)
    model.stage2.summary().to_csv(out / OUT_FILES["stage2"], index=False)
    model.ead.summary().to_csv(out / OUT_FILES["ead"], index=False)
    dg = build_diagnostics(model, parts)
    dg["diag"].to_csv(out / OUT_FILES["diag"], index=False)
    dg["deciles"].to_csv(out / OUT_FILES["deciles"], index=False)
    dg["grade"].to_csv(out / OUT_FILES["grade"], index=False)
    dg["resid"].to_csv(out / OUT_FILES["resid"], index=False)
    rv = recovery_by_default_vintage(defaults)
    rv.to_csv(out / OUT_FILES["vintage"], index=False)
    pick = lambda comp, metric, split: float(dg["diag"].query("component == @comp and metric == @metric and split == @split")["value"].iloc[0])  # noqa: E731
    summary = dict(
        data_source=config.DATA_SOURCE, reconciliation=recon, n_defaults={k: int(len(v)) for k, v in parts.items()}, n_fit=model.n_fit,
        breusch_pagan=dict(stage2=breusch_pagan(model.stage2), ead=breusch_pagan(model.ead)),
        stage1_auc={s: pick("stage1_any_recovery", "auc", s) for s in parts}, lgd_expected_r2={s: pick("lgd_expected", "r2", s) for s in parts},
        lgd_hard_r2={s: pick("lgd_hard_class", "r2", s) for s in parts}, lgd_expected_corr={s: pick("lgd_expected", "corr", s) for s in parts},
        ccf_r2={s: pick("ccf", "r2", s) for s in parts}, ccf_corr={s: pick("ccf", "corr", s) for s in parts},
        design_note="grade, addr_state and initial_list_status excluded; numerics winsorized and z-scored on development defaults",
        loss_basis_note="recovery rate is recoveries/funded and EAD is CCF*funded, so PD*LGD*EAD slightly overstates loss/EAD basis (original definition kept)")
    (out / OUT_FILES["summary"]).write_text(json.dumps(summary, indent=2, default=float))
    if save_model:
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, config.MODEL_DIR / "lgd_ead.joblib")
    return dict(model=model, defaults=defaults, parts=parts, summary=summary, diagnostics=dg, vintage=rv)
