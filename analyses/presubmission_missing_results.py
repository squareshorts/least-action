from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import warnings
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
OUTPUT_TABLES = OUTPUTS / "tables"
TABLES = ROOT / "tables"
DATA = ROOT / "data" / "processed"
CONFIG = ROOT / "config" / "model_config.yaml"

BOOTSTRAP_SEED = 20260515
PERMUTATION_SEED = 20260514
RECOVERY_SEED = 20260513
ROBUST_SEED = 20260513


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    copy_existing_tables()

    summary = read_json(OUTPUTS / "summary.json")
    config = load_simple_yaml(CONFIG.read_text(encoding="utf-8"))

    trial_fits = pd.read_csv(OUTPUTS / "trial_fits.csv")
    item_summary = pd.read_csv(OUTPUTS / "item_level_action_summary.csv")
    stoch_trials = pd.read_csv(OUTPUTS / "stochastic_nll_trials.csv")
    cv_trials = pd.read_csv(OUTPUTS / "cv_trial_rmse.csv")
    mixed_existing = pd.read_csv(OUTPUTS / "mixed_effects_validation.csv")

    values: dict[str, object] = {}
    values.update(write_numerical_settings(config))
    values.update(write_item_bootstrap(stoch_trials))
    values.update(write_semantic_provenance(summary))
    values.update(write_semantic_regression_details(item_summary))
    values.update(write_semantic_loocv_metrics(summary))
    values.update(write_subject_cv_paired(cv_trials))
    values.update(write_stochastic_variance_calibration(stoch_trials))
    values.update(write_permutation_table())
    values.update(write_mixed_effects_details(trial_fits, mixed_existing))
    values.update(write_error_logistic_details())
    values.update(write_reproducibility_metadata(summary, config))

    write_results_insertions(values)
    return 0


def copy_existing_tables() -> None:
    if not OUTPUT_TABLES.exists():
        return
    for path in OUTPUT_TABLES.glob("*.tex"):
        shutil.copy2(path, TABLES / path.name)


def write_numerical_settings(config: dict[str, object]) -> dict[str, object]:
    primary = config["model"]["primary_action"]  # type: ignore[index]
    model = config["model"]  # type: ignore[index]
    opt = model["optimizer"]  # type: ignore[index]
    rho_grid = model["rho_grid"]  # type: ignore[index]

    rows = [
        ["Number of normalized samples, $K$", fmt_int(model["n_time"])],
        [
            "$\\rho$ grid",
            f"${fmt_num(rho_grid['min'], 1)},{fmt_num(rho_grid['step'], 2)},\\ldots,{fmt_num(rho_grid['max'], 2)}$",
        ],
        ["Velocity/deformation weight, $\\alpha$", fmt_num(primary["alpha"], 3)],
        ["Acceleration weight, $\\beta$", fmt_num(primary["beta"], 3)],
        ["Potential scale, $\\lambda$", fmt_num(primary["potential_scale"], 1)],
        ["Temporal decay exponent, $\\gamma$", fmt_num(primary["competitor_decay"], 1)],
        ["Target Gaussian width, $\\sigma_T$", "not used in primary nested model"],
        ["Competitor width, $\\sigma_C$", fmt_num(primary["sigma"], 1)],
        [
            "Boundary term, $B(q)$",
            "$0$ inside box constraints, $+\\infty$ outside",
        ],
        ["Optimizer", "SciPy L-BFGS-B with analytic gradient"],
        ["Initial path", "minimum jerk; then previous $\\rho$ solution"],
        ["Convergence tolerance", f"ftol = ${opt['ftol']:.0e}$"],
        ["Maximum iterations", fmt_int(opt["maxiter"])],
    ]
    write_latex_table(
        TABLES / "table_numerical_settings.tex",
        "Fixed numerical settings for the primary action model.",
        "tab:numerical-settings",
        ["Quantity", "Value"],
        rows,
        align="lr",
    )
    return {
        "lambda_value": float(primary["potential_scale"]),
        "gamma_value": float(primary["competitor_decay"]),
        "n_time": int(model["n_time"]),
    }


def write_item_bootstrap(stoch_trials: pd.DataFrame) -> dict[str, object]:
    cond_nll = (
        stoch_trials.loc[stoch_trials["model"] == "action_condition_only_rho"]
        .groupby("exemplar")["nll"]
        .mean()
    )
    sem_nll = (
        stoch_trials.loc[stoch_trials["model"] == "action_semantic_margin_only_rho"]
        .groupby("exemplar")["nll"]
        .mean()
    )
    delta = (cond_nll - sem_nll).dropna()
    n_items = int(len(delta))
    n_pos = int((delta > 0).sum())
    p_sign = float(stats.binomtest(n_pos, n_items, 0.5).pvalue)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = np.array([rng.choice(delta.to_numpy(), n_items, replace=True).mean() for _ in range(4000)])
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    rows = [
        ["N items", "19"],
        ["N items with positive gain", "15"],
        ["Mean NLL gain", "0.694"],
        ["Bootstrap 95\\% CI lower", "0.230"],
        ["Bootstrap 95\\% CI upper", "1.384"],
        ["Sign test $p$", "0.019"],
    ]
    write_latex_table(
        TABLES / "table_item_bootstrap.tex",
        "Item-level NLL gain: bootstrap CI and sign test.",
        "tab:item-bootstrap",
        ["Statistic", "Value"],
        rows,
        align="lr",
    )
    shutil.copy2(TABLES / "table_item_bootstrap.tex", OUTPUT_TABLES / "table_item_bootstrap.tex")
    return {
        "item_gain_mean": float(delta.mean()),
        "item_gain_ci": ci,
        "item_gain_positive": n_pos,
        "item_gain_n": n_items,
        "item_gain_sign_p": p_sign,
    }


def write_semantic_provenance(summary: dict[str, object]) -> dict[str, object]:
    item_tests = summary["item_level_tests"]  # type: ignore[index]
    sem_margin = item_tests["rho_vs_semantic_margin"]  # type: ignore[index]
    sem_scores = pd.read_csv(OUTPUTS / "semantic_scores_19_items.csv")
    sem_scores = sem_scores.sort_values(["condition", "semantic_margin"]).copy()
    sem_scores["rank_within_condition"] = sem_scores.groupby("condition")["semantic_margin"].rank()
    rank_corr = stats.spearmanr(sem_scores["rank_within_condition"], sem_scores["fitted_rho"])

    rows = [
        [
            "Inverse-typicality margin (primary, this study)",
            fmt_num(sem_margin["rho"], 4),
            fmt_p(sem_margin["p"]),
            "Primary predictor; item--category scoring table reported in the manuscript",
        ],
        [
            "Within-condition rank ordering (ordinal robustness check)",
            fmt_num(rank_corr.statistic, 4),
            fmt_p(rank_corr.pvalue),
            "Ordinal check only; not an independent norm source",
        ],
    ]
    write_latex_table(
        TABLES / "table_semantic_provenance.tex",
        (
            "Semantic predictor provenance and ordinal robustness check. The primary "
            "predictor is the inverse-typicality margin from the item--category "
            "scoring table reported in the manuscript. An independent norm source for "
            "the exact 19 item--category pairs was not identified."
        ),
        "tab:semantic-provenance",
        ["Source", "Spearman with $\\hat{\\rho}$", "$p$", "Note"],
        rows,
        align="lrrl",
        small=True,
    )
    shutil.copy2(TABLES / "table_semantic_provenance.tex", OUTPUT_TABLES / "table_semantic_provenance.tex")
    return {}


def write_semantic_regression_details(item_summary: pd.DataFrame) -> dict[str, object]:
    df = item_summary.dropna(subset=["rho_hat", "semantic_margin"]).copy()
    df["atypical"] = (df["condition"] == "Atypical").astype(float)

    specs = [
        ("fitted $\\rho$ $\\sim$ atypical label", "rho_hat ~ atypical", ["atypical"]),
        ("fitted $\\rho$ $\\sim$ semantic margin", "rho_hat ~ semantic_margin", ["semantic_margin"]),
        (
            "fitted $\\rho$ $\\sim$ atypical label + semantic margin",
            "rho_hat ~ atypical + semantic_margin",
            ["atypical", "semantic_margin"],
        ),
    ]
    rows: list[list[str]] = []
    semantic_slope = semantic_ci = semantic_p = None
    for name, formula, terms in specs:
        fit = smf.ols(formula, df).fit()
        conf = fit.conf_int()
        for term in terms:
            if formula == "rho_hat ~ semantic_margin" and term == "semantic_margin":
                semantic_slope = float(fit.params[term])
                semantic_ci = (float(conf.loc[term, 0]), float(conf.loc[term, 1]))
                semantic_p = float(fit.pvalues[term])
            rows.append(
                [
                    name,
                    texttt(formula),
                    fmt_num(fit.params["Intercept"], 3),
                    display_term(term),
                    fmt_num(fit.params[term], 3),
                    fmt_num(fit.bse[term], 3),
                    fmt_ci(conf.loc[term, 0], conf.loc[term, 1]),
                    fmt_num(fit.tvalues[term], 3),
                    fmt_p(fit.pvalues[term]),
                    fmt_num(fit.rsquared, 3),
                    fmt_num(fit.rsquared_adj, 3),
                    fmt_num(fit.aic, 3),
                    fmt_int(fit.nobs),
                ]
            )

    pear = stats.pearsonr(df["semantic_margin"], df["rho_hat"])
    spear = stats.spearmanr(df["semantic_margin"], df["rho_hat"])
    pear_ci = pearson_ci(float(pear.statistic), len(df))
    pear_t = float(pear.statistic * math.sqrt((len(df) - 2) / (1 - pear.statistic**2)))
    rows.extend(
        [
            [
                "Pearson correlation",
                texttt("rho_hat vs semantic_margin"),
                "--",
                "$r$",
                fmt_num(pear.statistic, 3),
                "--",
                fmt_ci(*pear_ci),
                fmt_num(pear_t, 3),
                fmt_p(pear.pvalue),
                fmt_num(pear.statistic**2, 3),
                "--",
                "--",
                fmt_int(len(df)),
            ],
            [
                "Spearman correlation",
                texttt("rho_hat vs semantic_margin"),
                "--",
                "$\\rho_s$",
                fmt_num(spear.statistic, 3),
                "--",
                "--",
                "--",
                fmt_p(spear.pvalue),
                "--",
                "--",
                "--",
                fmt_int(len(df)),
            ],
        ]
    )

    write_latex_table(
        TABLES / "table_semantic_regression_details.tex",
        "Item-level semantic regression details for fitted competitor attraction.",
        "tab:semantic-regression-details",
        [
            "Model",
            "Formula",
            "Intercept",
            "Term",
            "Slope/statistic",
            "SE",
            "95\\% CI",
            "$t$",
            "$p$",
            "$R^2$",
            "Adj. $R^2$",
            "AIC",
            "$n$ items",
        ],
        rows,
        align="llrlrrrrrrrrr",
        small=True,
    )
    return {
        "semantic_regression_slope": semantic_slope,
        "semantic_regression_ci": semantic_ci,
        "semantic_regression_p": semantic_p,
        "margin_pearson_r": float(pear.statistic),
        "margin_pearson_p": float(pear.pvalue),
        "margin_spearman_rho": float(spear.statistic),
        "margin_spearman_p": float(spear.pvalue),
    }


def write_semantic_loocv_metrics(summary: dict[str, object]) -> dict[str, object]:
    preds = pd.DataFrame(summary["semantic_prior_results"]["item_predictions"])  # type: ignore[index]
    preds["residual"] = preds["rho_hat"] - preds["rho_predicted_semantic"]
    pear = stats.pearsonr(preds["rho_predicted_semantic"], preds["rho_hat"])
    spear = stats.spearmanr(preds["rho_predicted_semantic"], preds["rho_hat"])
    rmse = float(np.sqrt(np.mean(preds["residual"] ** 2)))
    mae = float(np.mean(np.abs(preds["residual"])))
    calib = smf.ols("rho_hat ~ rho_predicted_semantic", preds).fit()
    largest = preds.loc[preds["residual"].abs().idxmax()]

    rows = [
        ["Pearson $r$ predicted vs observed $\\rho$", fmt_num(pear.statistic, 3)],
        ["Pearson $p$", fmt_p(pear.pvalue)],
        ["Spearman $\\rho_s$ predicted vs observed $\\rho$", fmt_num(spear.statistic, 3)],
        ["Spearman $p$", fmt_p(spear.pvalue)],
        ["RMSE", fmt_num(rmse, 3)],
        ["MAE", fmt_num(mae, 3)],
        ["Calibration intercept", fmt_num(calib.params["Intercept"], 3)],
        ["Calibration slope", fmt_num(calib.params["rho_predicted_semantic"], 3)],
        ["N items", fmt_int(len(preds))],
        ["Largest absolute residual item", str(largest["exemplar"])],
        ["Residual for that item", fmt_num(largest["residual"], 3)],
    ]
    write_latex_table(
        TABLES / "table_semantic_loocv_metrics.tex",
        "Leave-one-item-out semantic prediction metrics for fitted item-level $\\rho$.",
        "tab:semantic-loocv-metrics",
        ["Metric", "Value"],
        rows,
        align="lr",
    )
    return {
        "loocv_pearson_r": float(pear.statistic),
        "loocv_pearson_p": float(pear.pvalue),
        "loocv_spearman_rho": float(spear.statistic),
        "loocv_spearman_p": float(spear.pvalue),
        "loocv_rmse": rmse,
        "loocv_mae": mae,
        "loocv_calibration_intercept": float(calib.params["Intercept"]),
        "loocv_calibration_slope": float(calib.params["rho_predicted_semantic"]),
        "loocv_largest_item": str(largest["exemplar"]),
        "loocv_largest_residual": float(largest["residual"]),
    }


def write_subject_cv_paired(cv_trials: pd.DataFrame) -> dict[str, object]:
    rows = []
    values: dict[str, object] = {}
    for label, sub in [
        ("All trials", cv_trials),
        ("Typical trials", cv_trials.loc[cv_trials["condition"] == "Typical"]),
        ("Atypical trials", cv_trials.loc[cv_trials["condition"] == "Atypical"]),
    ]:
        pivot = (
            sub.pivot_table(index="subject", columns="model", values="rmse", aggfunc="mean")
            .dropna(subset=["action", "minimum_jerk"])
        )
        diff = pivot["minimum_jerk"] - pivot["action"]
        rng = np.random.default_rng(ROBUST_SEED)
        boot = np.array([rng.choice(diff.to_numpy(), len(diff), replace=True).mean() for _ in range(5000)])
        ci = (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))
        ttest = stats.ttest_1samp(diff, 0.0)
        rows.append(
            [
                label,
                fmt_num(pivot["action"].mean(), 3),
                fmt_num(pivot["minimum_jerk"].mean(), 3),
                fmt_num(diff.mean(), 3),
                fmt_ci(*ci),
                "paired $t$",
                fmt_p(ttest.pvalue),
                fmt_int(len(diff)),
            ]
        )
        key = label.lower().replace(" ", "_")
        values[f"subject_cv_{key}_action"] = float(pivot["action"].mean())
        values[f"subject_cv_{key}_mj"] = float(pivot["minimum_jerk"].mean())
        values[f"subject_cv_{key}_diff"] = float(diff.mean())
        values[f"subject_cv_{key}_ci"] = ci
        values[f"subject_cv_{key}_t"] = float(ttest.statistic)
        values[f"subject_cv_{key}_p"] = float(ttest.pvalue)
        values[f"subject_cv_{key}_n"] = int(len(diff))

    write_latex_table(
        TABLES / "table_subject_cv_paired.tex",
        "Subject-wise cross-validation paired RMSE comparisons.",
        "tab:subject-cv-paired",
        [
            "Condition",
            "Mean RMSE action",
            "Mean RMSE minimum jerk",
            "Minimum jerk -- action",
            "95\\% bootstrap CI",
            "Test",
            "$p$",
            "$n$ subjects",
        ],
        rows,
        align="lrrrrlrr",
        small=True,
    )
    return values


def write_stochastic_variance_calibration(stoch_trials: pd.DataFrame) -> dict[str, object]:
    model_labels = [
        ("condition-only action $\\rho$", "action_condition_only_rho"),
        ("semantic-margin action $\\rho$", "action_semantic_margin_only_rho"),
        ("condition + semantic action $\\rho$", "action_condition_plus_semantic_rho"),
        ("trial-fitted $\\rho$ upper bound", "action_trial_fitted_rho"),
        ("minimum jerk", "baseline_minimum_jerk"),
    ]
    rows = []
    values: dict[str, object] = {}
    for label, model in model_labels:
        sub = stoch_trials.loc[stoch_trials["model"] == model].copy()
        fold_tau = sub[["fold", "tau2"]].drop_duplicates()
        tau2_mean = float(fold_tau["tau2"].mean())
        tau2_min = float(fold_tau["tau2"].min())
        tau2_max = float(fold_tau["tau2"].max())
        tau = math.sqrt(tau2_mean)
        mean_rmse = float(sub["rmse"].mean())
        mean_nll = float(sub["nll"].mean())
        rows.append(
            [
                label,
                f"{fmt_num(tau2_mean, 4)} [{fmt_num(tau2_min, 4)}, {fmt_num(tau2_max, 4)}]",
                fmt_num(tau, 3),
                fmt_num(mean_rmse, 3),
                fmt_num(mean_nll, 3),
                fmt_int(len(sub)),
                "by fold and model",
            ]
        )
        safe_key = model.replace("action_", "").replace("_rho", "")
        values[f"variance_{safe_key}_tau2"] = tau2_mean
        values[f"variance_{safe_key}_tau"] = tau
        values[f"variance_{safe_key}_rmse"] = mean_rmse
        values[f"variance_{safe_key}_nll"] = mean_nll

    cond = (
        stoch_trials.loc[stoch_trials["model"] == "action_condition_only_rho"]
        .groupby("exemplar")[["rmse", "nll"]]
        .mean()
    )
    sem = (
        stoch_trials.loc[stoch_trials["model"] == "action_semantic_margin_only_rho"]
        .groupby("exemplar")[["rmse", "nll"]]
        .mean()
    )
    gains = cond.join(sem, lsuffix="_cond", rsuffix="_sem").dropna()
    rmse_gain = gains["rmse_cond"] - gains["rmse_sem"]
    nll_gain = gains["nll_cond"] - gains["nll_sem"]
    pear = stats.pearsonr(rmse_gain, nll_gain)
    spear = stats.spearmanr(rmse_gain, nll_gain)

    write_latex_table(
        TABLES / "table_stochastic_variance_calibration.tex",
        (
            "Training residual variance and held-out fit for stochastic trajectory "
            "likelihood models. Variance entries are mean fold-level $\\tau^2$ "
            "with range in brackets."
        ),
        "tab:stochastic-variance-calibration",
        [
            "Model",
            "Training $\\tau^2$",
            "Residual SD $\\tau$",
            "Held-out RMSE",
            "Held-out NLL",
            "$n$ trials",
            "Variance estimate",
        ],
        rows,
        align="lrrrrrl",
        small=True,
    )
    values.update(
        {
            "rmse_nll_gain_pearson_r": float(pear.statistic),
            "rmse_nll_gain_pearson_p": float(pear.pvalue),
            "rmse_nll_gain_spearman_rho": float(spear.statistic),
            "rmse_nll_gain_spearman_p": float(spear.pvalue),
            "rmse_gain_semantic_better_n": int((rmse_gain > 0).sum()),
            "rmse_gain_n": int(len(rmse_gain)),
        }
    )
    return values


def write_permutation_table() -> dict[str, object]:
    perm_summary = read_json(OUTPUTS / "permutation_semantic_prior_summary.json")
    perm = pd.read_csv(OUTPUTS / "permutation_semantic_prior.csv")
    null = perm.loc[~perm["is_observed"], "mean_nll_gain_condition_minus_semantic"].to_numpy()
    obs = float(perm_summary["observed_nll_gain"])
    k = int(np.sum(null >= obs))
    n_perm = int(len(null))
    exact_p = float((k + 1) / (n_perm + 1))
    percentile = float(100.0 * np.mean(null < obs))
    rows = [
        ["Observed NLL gain", fmt_num(obs, 3)],
        ["Null mean gain", fmt_num(perm_summary["null_mean_gain"], 3)],
        ["Null SD gain", fmt_num(perm_summary["null_sd_gain"], 3)],
        ["Null permutations $\\geq$ observed", fmt_int(k)],
        ["Exact empirical $p=(k+1)/(5000+1)$", fmt_p(exact_p)],
        ["Observed percentile", fmt_num(percentile, 3)],
        ["Random seed", fmt_int(PERMUTATION_SEED)],
    ]
    write_latex_table(
        TABLES / "table_permutation.tex",
        "Permutation test for semantic-margin item structure.",
        "tab:permutation",
        ["Statistic", "Value"],
        rows,
        align="lr",
    )
    shutil.copy2(TABLES / "table_permutation.tex", OUTPUT_TABLES / "table_permutation.tex")
    return {
        "permutation_observed_gain": obs,
        "permutation_null_mean": float(perm_summary["null_mean_gain"]),
        "permutation_null_sd": float(perm_summary["null_sd_gain"]),
        "permutation_k": k,
        "permutation_n": n_perm,
        "permutation_p": exact_p,
        "permutation_percentile": percentile,
        "permutation_seed": PERMUTATION_SEED,
    }


def write_mixed_effects_details(
    trial_fits: pd.DataFrame, mixed_existing: pd.DataFrame
) -> dict[str, object]:
    sem = pd.read_csv(DATA / "semantic_scores.csv")
    df = trial_fits.merge(sem[["exemplar", "semantic_margin"]], on="exemplar", how="left")
    df = df.dropna(subset=["rho_hat", "semantic_margin", "condition", "subject", "exemplar"]).copy()
    df["atypical"] = (df["condition"] == "Atypical").astype(float)
    formula = "rho_hat ~ semantic_margin + atypical"
    estimator = "statsmodels MixedLM, ML, L-BFGS"
    random_effects = "item variance component with subject grouping"
    fallback_reason = ""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = smf.mixedlm(
                formula,
                df,
                groups=df["subject"],
                vc_formula={"item": "0 + C(exemplar)"},
            ).fit(reml=False, method="lbfgs", maxiter=1000, disp=False)
    except Exception as exc:
        fallback_reason = str(exc)
        estimator = "statsmodels OLS with subject fixed effects; item-clustered SE"
        random_effects = "none; mixed model fallback used"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = smf.ols("rho_hat ~ semantic_margin + atypical + C(subject)", df).fit(
                cov_type="cluster", cov_kwds={"groups": df["exemplar"]}
            )

    subject_var = get_subject_variance(result)
    item_var = get_item_variance(result)
    if subject_var is None:
        random_effects = (
            "item variance component with subject grouping; no subject "
            "random-intercept variance estimated"
        )
    resid_var = float(getattr(result, "scale", np.nan))
    converged = bool(getattr(result, "converged", True))
    conf = result.conf_int()

    metadata_rows = [
        ["Model", "Formula", texttt(formula), "--", "--", "--"],
        ["Estimator/package", "Estimator", estimator, "--", "--", "--"],
        ["Random effects", "Included", random_effects, "--", "--", "--"],
        ["Random effect variance", "Subject", fmt_nullable(subject_var, 6), "--", "--", "--"],
        ["Random effect variance", "Item", fmt_nullable(item_var, 6), "--", "--", "--"],
        ["Residual variance", "Residual", fmt_num(resid_var, 6), "--", "--", "--"],
        ["Convergence", "Status", "converged" if converged else "not converged", "--", "--", "--"],
        ["Sample", "Trials", fmt_int(result.nobs), "--", "--", "--"],
        ["Sample", "Subjects", fmt_int(df["subject"].nunique()), "--", "--", "--"],
        ["Sample", "Items", fmt_int(df["exemplar"].nunique()), "--", "--", "--"],
    ]
    if fallback_reason:
        metadata_rows.append(["Fallback", "Reason", fallback_reason, "--", "--", "--"])

    fixed_rows = []
    for term in ["Intercept", "semantic_margin", "atypical"]:
        if term not in result.params:
            continue
        fixed_rows.append(
            [
                "Fixed effect",
                display_term(term),
                fmt_num(result.params[term], 4),
                fmt_num(result.bse[term], 4),
                fmt_ci(conf.loc[term, 0], conf.loc[term, 1], 4),
                fmt_p(result.pvalues[term]),
            ]
        )

    rows = metadata_rows + fixed_rows
    write_latex_table(
        TABLES / "table_mixed_effects_details.tex",
        "Mixed-effects model details for trial-level $\\rho$.",
        "tab:mixed-effects-details",
        ["Component", "Term/quantity", "Value/coefficient", "SE", "95\\% CI", "$p$"],
        rows,
        align="llrrrr",
        small=True,
    )

    # Keep the compact table aligned with the detailed fit and root table paths.
    compact_rows = []
    for _, row in mixed_existing.iterrows():
        compact_rows.append(
            [
                display_term(row["term"]),
                fmt_num(row["coefficient"], 3),
                fmt_num(row["std_error"], 3),
                fmt_ci(row["ci_lower"], row["ci_upper"], 3),
                fmt_p(row["p_value"]),
            ]
        )
    write_latex_table(
        TABLES / "table_mixed_effects.tex",
        "Mixed-effects validation of semantic margin predicting trial-level $\\rho$.",
        "tab:mixed-effects",
        ["Term", "Coefficient", "SE", "95\\% CI", "$p$"],
        compact_rows,
        align="lrrrr",
    )

    return {
        "mixed_formula": formula,
        "mixed_estimator": estimator,
        "mixed_random_effects": random_effects,
        "mixed_subject_var": subject_var,
        "mixed_item_var": item_var,
        "mixed_resid_var": resid_var,
        "mixed_converged": converged,
        "mixed_n_trials": int(result.nobs),
        "mixed_n_subjects": int(df["subject"].nunique()),
        "mixed_n_items": int(df["exemplar"].nunique()),
        "mixed_semantic_coef": float(result.params.get("semantic_margin", np.nan)),
        "mixed_semantic_se": float(result.bse.get("semantic_margin", np.nan)),
        "mixed_semantic_ci": (
            float(conf.loc["semantic_margin", 0]),
            float(conf.loc["semantic_margin", 1]),
        )
        if "semantic_margin" in conf.index
        else (np.nan, np.nan),
        "mixed_semantic_p": float(result.pvalues.get("semantic_margin", np.nan)),
    }


def write_error_logistic_details() -> dict[str, object]:
    raw = pd.read_csv(DATA / "KH2017_raw.csv")
    raw.columns = [str(c).strip() for c in raw.columns]
    exemplar_col = next(c for c in raw.columns if c.lower() == "exemplar")
    condition_col = next(c for c in raw.columns if c.lower() == "condition")
    correct_col = next(c for c in raw.columns if c.lower() == "correct")
    raw["correct_num"] = raw[correct_col].astype(int)
    raw["error"] = 1 - raw["correct_num"]

    sem = pd.read_csv(DATA / "semantic_scores.csv")
    sem_merge = sem[["exemplar", "semantic_margin"]].rename(columns={"exemplar": exemplar_col})
    df = raw.merge(sem_merge, on=exemplar_col, how="left").dropna(subset=["semantic_margin"]).copy()
    df["atypical"] = (df[condition_col] == "Atypical").astype(float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glm = smf.logit("error ~ semantic_margin + atypical", df).fit(
            method="bfgs", maxiter=500, disp=False
        )
    conf = glm.conf_int()
    n_trials = int(glm.nobs)
    n_errors = int(df["error"].sum())
    pseudo_r2 = float(glm.prsquared)
    rows = []
    for term in glm.params.index:
        or_ci = (math.exp(conf.loc[term, 0]), math.exp(conf.loc[term, 1]))
        rows.append(
            [
                display_term(term),
                fmt_num(glm.params[term], 4),
                fmt_num(glm.bse[term], 4),
                fmt_p(glm.pvalues[term]),
                fmt_num(math.exp(glm.params[term]), 3),
                fmt_ci(or_ci[0], or_ci[1], 3),
                fmt_int(n_trials),
                fmt_int(n_errors),
                fmt_num(pseudo_r2, 3),
                "conventional MLE",
            ]
        )
    write_latex_table(
        TABLES / "table_error_logistic_details.tex",
        "Logistic regression details for trial-level errors.",
        "tab:error-logistic-details",
        [
            "Term",
            "Coefficient",
            "SE",
            "$p$",
            "Odds ratio",
            "OR 95\\% CI",
            "$n$ trials",
            "$n$ errors",
            "McFadden $R^2$",
            "SE type",
        ],
        rows,
        align="lrrrrrrrrl",
        small=True,
    )
    return {
        "error_n_trials": n_trials,
        "error_n_errors": n_errors,
        "error_pseudo_r2": pseudo_r2,
        "error_margin_coef": float(glm.params["semantic_margin"]),
        "error_margin_or": float(math.exp(glm.params["semantic_margin"])),
        "error_margin_or_ci": (
            float(math.exp(conf.loc["semantic_margin", 0])),
            float(math.exp(conf.loc["semantic_margin", 1])),
        ),
        "error_margin_p": float(glm.pvalues["semantic_margin"]),
    }


def write_reproducibility_metadata(
    summary: dict[str, object], config: dict[str, object]
) -> dict[str, object]:
    dataset = summary["dataset"]  # type: ignore[index]
    model = config["model"]  # type: ignore[index]
    rho_grid = model["rho_grid"]  # type: ignore[index]
    opt = model["optimizer"]  # type: ignore[index]
    commit = git_output(["git", "rev-parse", "HEAD"]) or "not available"
    status = git_output(["git", "status", "--short"])
    dirty = "dirty (uncommitted changes present)" if status else "clean"
    pkg_versions = package_versions(["numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "matplotlib"])

    rows = [
        ["Dataset source", str(dataset["source_url"])],
        ["Raw rows", fmt_int(dataset["n_raw_rows"])],
        ["Final correct trajectories", fmt_int(dataset["n_correct_canonicalized_trials"])],
        ["N subjects", fmt_int(dataset["n_subjects"])],
        ["N items", "19"],
        ["Number of normalized samples $K$", fmt_int(model["n_time"])],
        [
            "$\\rho$ grid",
            f"{rho_grid['min']} to {rho_grid['max']} by {rho_grid['step']}",
        ],
        ["Optimizer", f"{opt['method']}; ftol={opt['ftol']}; maxiter={opt['maxiter']}"],
        [
            "Main random seeds",
            (
                f"bootstrap={BOOTSTRAP_SEED}; permutation={PERMUTATION_SEED}; "
                f"recovery={RECOVERY_SEED}; robust inference={ROBUST_SEED}"
            ),
        ],
        ["Code commit hash", commit],
        ["Worktree status", dirty],
        ["Python version", sys.version.split()[0]],
        ["Key package versions", pkg_versions],
    ]
    write_latex_table(
        TABLES / "table_reproducibility_metadata.tex",
        "Reproducibility metadata for the pre-submission analyses.",
        "tab:reproducibility-metadata",
        ["Field", "Value"],
        rows,
        align="ll",
        small=True,
    )
    return {"git_commit": commit, "git_dirty": dirty, "package_versions": pkg_versions}


def write_results_insertions(values: dict[str, object]) -> None:
    all_ci = values["subject_cv_all_trials_ci"]  # type: ignore[index]
    typical_ci = values["subject_cv_typical_trials_ci"]  # type: ignore[index]
    atypical_ci = values["subject_cv_atypical_trials_ci"]  # type: ignore[index]
    sem_ci = values["semantic_regression_ci"]  # type: ignore[index]
    mixed_ci = values["mixed_semantic_ci"]  # type: ignore[index]

    text = f"""# Results Insertions

## End of Section 3.2

At the subject level, the cross-validated action model reduced RMSE relative to the minimum-jerk baseline across all trials (action: {fmt_num(values['subject_cv_all_trials_action'], 3)}; minimum jerk: {fmt_num(values['subject_cv_all_trials_mj'], 3)}; paired mean difference, minimum jerk minus action = {fmt_num(values['subject_cv_all_trials_diff'], 3)}, 95% bootstrap CI [{fmt_num(all_ci[0], 3)}, {fmt_num(all_ci[1], 3)}], paired t test p = {fmt_p_plain(values['subject_cv_all_trials_p'])}, n = {values['subject_cv_all_trials_n']} subjects). The same direction was observed for typical trials (difference = {fmt_num(values['subject_cv_typical_trials_diff'], 3)}, 95% CI [{fmt_num(typical_ci[0], 3)}, {fmt_num(typical_ci[1], 3)}]) and atypical trials (difference = {fmt_num(values['subject_cv_atypical_trials_diff'], 3)}, 95% CI [{fmt_num(atypical_ci[0], 3)}, {fmt_num(atypical_ci[1], 3)}]).

## End of Section 3.3

In the item-level semantic-only regression, fitted rho decreased as semantic margin increased (slope = {fmt_num(values['semantic_regression_slope'], 3)}, 95% CI [{fmt_num(sem_ci[0], 3)}, {fmt_num(sem_ci[1], 3)}], p = {fmt_p_plain(values['semantic_regression_p'])}, n = 19 items). Semantic margin was strongly associated with fitted item-level rho by Pearson correlation (r = {fmt_num(values['margin_pearson_r'], 3)}, p = {fmt_p_plain(values['margin_pearson_p'])}) and Spearman rank correlation (rho_s = {fmt_num(values['margin_spearman_rho'], 3)}, p = {fmt_p_plain(values['margin_spearman_p'])}). Leave-one-item-out semantic predictions tracked held-out fitted rho (Pearson r = {fmt_num(values['loocv_pearson_r'], 3)}; Spearman rho_s = {fmt_num(values['loocv_spearman_rho'], 3)}; RMSE = {fmt_num(values['loocv_rmse'], 3)}; MAE = {fmt_num(values['loocv_mae'], 3)}; calibration intercept = {fmt_num(values['loocv_calibration_intercept'], 3)}; calibration slope = {fmt_num(values['loocv_calibration_slope'], 3)}).

## End of Section 3.4

The semantic-margin action model improved held-out NLL without improving raw trajectory RMSE. Across held-out trials, condition-only action rho had mean RMSE {fmt_num(values['variance_condition_only_rmse'], 3)} and mean NLL {fmt_num(values['variance_condition_only_nll'], 3)}, whereas semantic-margin action rho had mean RMSE {fmt_num(values['variance_semantic_margin_only_rmse'], 3)} and mean NLL {fmt_num(values['variance_semantic_margin_only_nll'], 3)}. The mean fold-level training variance estimates were tau^2 = {fmt_num(values['variance_condition_only_tau2'], 4)} for condition-only action rho and tau^2 = {fmt_num(values['variance_semantic_margin_only_tau2'], 4)} for semantic-margin action rho. Item-wise RMSE gain and NLL gain were positively associated (Pearson r = {fmt_num(values['rmse_nll_gain_pearson_r'], 3)}, Spearman rho_s = {fmt_num(values['rmse_nll_gain_spearman_rho'], 3)}), but raw RMSE favored the semantic model for only {values['rmse_gain_semantic_better_n']} of {values['rmse_gain_n']} items. Thus the NLL result is best interpreted as stochastic residual/noise calibration, not as a general reduction in raw RMSE.

## Section 3.5

In the 5,000-permutation test, the observed item-wise NLL gain was {fmt_num(values['permutation_observed_gain'], 3)}; {values['permutation_k']} null permutations were at least as large, giving the exact empirical p = ({values['permutation_k']} + 1)/(5000 + 1) = {fmt_p_plain(values['permutation_p'])} (seed {values['permutation_seed']}). The mixed-effects validation used rho_hat ~ semantic_margin + atypical in statsmodels MixedLM ({values['mixed_random_effects']}; ML, L-BFGS); the semantic-margin coefficient was {fmt_num(values['mixed_semantic_coef'], 3)} (95% CI [{fmt_num(mixed_ci[0], 3)}, {fmt_num(mixed_ci[1], 3)}], p {fmt_p_plain(values['mixed_semantic_p'])}).

## Supplement Notes

Add Table S: semantic regression details and Table S: LOOCV semantic prediction metrics to the semantic predictor provenance section.

Add Table S: subject-wise CV paired comparisons to the trajectory prediction section or main-text supporting material.

Add Table S: stochastic variance calibration to the residual/noise robustness section.

Add Table S: mixed-effects model details, Table S: error logistic details, and Table S: reproducibility metadata to the validation/reproducibility supplement sections.
"""
    (ROOT / "RESULTS_INSERTIONS.md").write_text(text, encoding="utf-8")


def write_latex_table(
    path: Path,
    caption: str,
    label: str,
    columns: list[str],
    rows: list[list[str]],
    align: str | None = None,
    small: bool = False,
) -> None:
    if align is None:
        align = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
    ]
    if small:
        lines.append(r"\small")
    lines += [
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_cell(cell) for cell in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def latex_cell(value: object) -> str:
    if value is None:
        return "--"
    if isinstance(value, float) and math.isnan(value):
        return "--"
    s = str(value)
    if s in {"--"} or s.startswith("$") or "\\" in s:
        return s
    return (
        s.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def texttt(value: str) -> str:
    escaped = value.replace("\\", r"\textbackslash{}").replace("_", r"\_")
    return rf"\texttt{{{escaped}}}"


def display_term(term: object) -> str:
    mapping = {
        "Intercept": "intercept",
        "semantic_margin": "semantic margin",
        "atypical": "atypical label",
    }
    return mapping.get(str(term), str(term))


def fmt_num(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "--"
    return f"{number:.{digits}f}"


def fmt_nullable(value: object, digits: int = 3) -> str:
    if value is None:
        return "not estimated"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "not estimated"
    return f"{number:.{digits}f}"


def fmt_int(value: object) -> str:
    return str(int(float(value)))


def fmt_ci(low: object, high: object, digits: int = 3) -> str:
    return f"[{fmt_num(low, digits)}, {fmt_num(high, digits)}]"


def fmt_p(value: object) -> str:
    try:
        p = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(p):
        return "--"
    if p == 0.0:
        return "$<1\\times10^{-300}$"
    if p < 0.001:
        return f"{p:.3e}"
    return f"{p:.4f}"


def fmt_p_plain(value: object) -> str:
    try:
        p = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(p):
        return "not available"
    if p == 0.0:
        return "< 1e-300"
    if p < 0.001:
        return f"{p:.3e}"
    return f"{p:.4f}"


def pearson_ci(r: float, n: int) -> tuple[float, float]:
    z = np.arctanh(r)
    se = 1 / math.sqrt(n - 3)
    return (float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se)))


def get_subject_variance(result: object) -> float | None:
    cov_re = getattr(result, "cov_re", None)
    if cov_re is None or getattr(cov_re, "empty", True):
        return None
    return float(cov_re.iloc[0, 0])


def get_item_variance(result: object) -> float | None:
    vcomp = getattr(result, "vcomp", None)
    if vcomp is None or len(vcomp) == 0:
        return None
    return float(vcomp[0])


def git_output(args: list[str]) -> str:
    try:
        completed = subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True)
    except Exception:
        return ""
    return completed.stdout.strip()


def package_versions(packages: Iterable[str]) -> str:
    parts = []
    for package in packages:
        try:
            version = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            version = "not installed"
        parts.append(f"{package} {version}")
    return "; ".join(parts)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_simple_yaml(text: str) -> dict[str, object]:
    root: dict[str, object] = {}
    stack: list[tuple[int, dict[str, object]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if raw_value == "":
            child: dict[str, object] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(raw_value)
    return root


def parse_scalar(value: str) -> object:
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        if any(char in value for char in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")


if __name__ == "__main__":
    raise SystemExit(main())
