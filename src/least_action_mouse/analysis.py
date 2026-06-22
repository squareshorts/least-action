from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.model_selection import GroupKFold

from .action_model import ActionParams, precompute_action_grid
from .baselines import (
    bezier_attraction_paths,
    condition_mean_cv,
    cross_validated_grid_model,
    gaussian_path_loglik,
    spline_attraction_paths,
)
from .config import config_value, load_model_config
from .data import KH2017_RDA_URL, ensure_kh2017_csv
from .physical_action import PhysicalActionParams, precompute_physical_action_grid
from .preprocess import TrajectoryData, minimum_jerk_path, preprocess_kh2017
from .robust_stats import paired_subject_inference, residualized_slope_inference
from .simulation import parameter_recovery
from .semantic_prior import semantic_prior_rho, semantic_prior_plot
from .stochastic_action import stochastic_likelihood_comparison


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = ensure_kh2017_csv(args.data_dir)
    raw = pd.read_csv(csv_path)
    data = preprocess_kh2017(raw, n_time=args.n_time)

    rhos = np.round(np.arange(args.rho_min, args.rho_max + 0.5 * args.rho_step, args.rho_step), 10)
    action_grid = precompute_action_grid(rhos, args.n_time, ActionParams(maxiter=args.maxiter))
    physical_grid = precompute_physical_action_grid(
        rhos,
        args.n_time,
        PhysicalActionParams(maxiter=args.maxiter),
    )

    rmse_matrix = trajectory_rmse_matrix(data.trajectories, action_grid.target_paths)
    cv_trials, cv_summary, selected_rhos = cross_validated_rmse(data, rmse_matrix, action_grid.rhos)
    trial_fits = fit_trial_conflict(data, rmse_matrix, action_grid)
    model_trials, model_summary, model_selection = stronger_model_comparison(
        data=data,
        action_cv_trials=cv_trials,
        action_grid=action_grid,
        physical_grid=physical_grid,
        rhos=rhos,
    )

    descriptive = descriptive_tests(trial_fits)
    rho_tests = rho_condition_tests(trial_fits)
    rho_augmented = rho_augmented_auc_test(trial_fits)
    robust = robust_inference(trial_fits)
    counterfactual = counterfactual_tests(trial_fits)
    item_summary, item_tests = item_level_analysis(trial_fits, raw, args.semantic_scores)
    recovery_trials, recovery_summary = parameter_recovery(action_grid.target_paths, action_grid.rhos)

    semantic_scores_df = None
    if args.semantic_scores:
        semantic_scores_df = pd.read_csv(args.semantic_scores)

    semantic_prior_results = semantic_prior_rho(item_summary)
    semantic_prior_plot(item_summary, results_dir)

    stochastic_nll_trials = stochastic_likelihood_comparison(
        data=data,
        action_grid=action_grid,
        semantic_scores=semantic_scores_df,
        item_summary=item_summary,
        n_splits=19,
        grouping_col="exemplar",
    )
    
    stochastic_nll_summary = summarize_model_trials(stochastic_nll_trials)
    
    item_wise_nll = {}
    if "exemplar" in stochastic_nll_trials.columns:
        from scipy.stats import ttest_rel, wilcoxon
        cond_nll = stochastic_nll_trials[stochastic_nll_trials["model"] == "action_condition_only_rho"].groupby("exemplar")["nll"].mean()
        sem_nll = stochastic_nll_trials[stochastic_nll_trials["model"] == "action_semantic_margin_only_rho"].groupby("exemplar")["nll"].mean()
        full_nll = stochastic_nll_trials[stochastic_nll_trials["model"] == "action_condition_plus_semantic_rho"].groupby("exemplar")["nll"].mean()
        
        delta_nll = cond_nll - sem_nll
        t_stat, p_t = ttest_rel(cond_nll, sem_nll)
        w_stat, p_w = wilcoxon(cond_nll, sem_nll)
        item_wise_nll = {
            "mean_delta_nll_per_item": float(delta_nll.mean()),
            "total_delta_nll": float(stochastic_nll_trials[stochastic_nll_trials["model"] == "action_condition_only_rho"]["nll"].sum() - stochastic_nll_trials[stochastic_nll_trials["model"] == "action_semantic_margin_only_rho"]["nll"].sum()),
            "n_items_positive_gain": int((delta_nll > 0).sum()),
            "n_items_total": len(delta_nll),
            "paired_t": float(t_stat),
            "p_t": float(p_t),
            "wilcoxon_w": float(w_stat),
            "p_w": float(p_w),
            "mean_nll_condition": float(cond_nll.mean()),
            "mean_nll_semantic": float(sem_nll.mean()),
            "mean_nll_full": float(full_nll.mean()),
        }
    
    plot_decisive_four_panel(item_summary, semantic_prior_results, stochastic_nll_summary, results_dir / "decisive_four_panel.png")

    best_full = fit_condition_rhos(
        np.arange(len(data.metadata)),
        data.metadata,
        rmse_matrix,
        action_grid.rhos,
    )
    plot_trajectory_fit(data, action_grid, best_full, results_dir / "trajectory_fit.png")
    plot_rho_by_condition(trial_fits, results_dir / "rho_by_condition.png")
    plot_subject_paired_rho(trial_fits, results_dir / "rho_subject_paired.png")

    cv_trials.to_csv(results_dir / "cv_trial_rmse.csv", index=False)
    cv_summary.to_csv(results_dir / "cv_rmse_by_condition.csv", index=False)
    model_trials.to_csv(results_dir / "model_comparison_trials.csv", index=False)
    model_summary.to_csv(results_dir / "model_comparison_summary.csv", index=False)
    model_selection.to_csv(results_dir / "model_selection_by_fold.csv", index=False)
    trial_fits.to_csv(results_dir / "trial_fits.csv", index=False)
    item_summary.to_csv(results_dir / "item_level_action_summary.csv", index=False)
    recovery_trials.to_csv(results_dir / "parameter_recovery.csv", index=False)
    pd.DataFrame(selected_rhos).to_csv(results_dir / "selected_rhos_by_fold.csv", index=False)
    stochastic_nll_trials.to_csv(results_dir / "stochastic_nll_trials.csv", index=False)
    stochastic_nll_summary.to_csv(results_dir / "stochastic_nll_summary.csv", index=False)

    summary = build_summary(
        data=data,
        cv_summary=cv_summary,
        model_summary=model_summary,
        trial_fits=trial_fits,
        descriptive=descriptive,
        rho_tests=rho_tests,
        rho_augmented=rho_augmented,
        robust=robust,
        counterfactual=counterfactual,
        item_tests=item_tests,
        recovery_summary=recovery_summary,
        best_full=best_full,
        selected_rhos=selected_rhos,
        action_grid=action_grid,
        physical_grid=physical_grid,
        semantic_prior_results=semantic_prior_results,
        stochastic_nll_summary=stochastic_nll_summary,
    )
    summary["item_wise_nll"] = item_wise_nll
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_summary(summary, results_dir)
    return 0


def trajectory_rmse_matrix(trajectories: np.ndarray, paths: np.ndarray) -> np.ndarray:
    delta = trajectories[:, None, :, :] - paths[None, :, :, :]
    return np.sqrt(np.mean(np.sum(delta * delta, axis=-1), axis=-1))


def path_rmse_vector(trajectories: np.ndarray, path: np.ndarray) -> np.ndarray:
    delta = trajectories - path[None, :, :]
    return np.sqrt(np.mean(np.sum(delta * delta, axis=-1), axis=-1))


def stronger_model_comparison(
    data: TrajectoryData,
    action_cv_trials: pd.DataFrame,
    action_grid,
    physical_grid,
    rhos: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare action models against ablations and non-Lagrangian baselines."""

    model_rows: list[pd.DataFrame] = []
    selection_rows: list[pd.DataFrame] = []

    renamed = action_cv_trials.copy()
    renamed["selection"] = np.where(
        renamed["model"] == "action",
        "condition_specific_rho",
        "none",
    )
    renamed["model"] = renamed["model"].replace(
        {
            "minimum_jerk": "baseline_minimum_jerk",
            "action": "nested_action_condition_rho",
        }
    )
    model_rows.append(renamed)

    strength_grid = np.round(np.arange(0.0, 1.5001, 0.05), 10)
    for family_name, paths in [
        ("spline_attraction", spline_attraction_paths(strength_grid, data.trajectories.shape[1])),
        ("bezier_attraction", bezier_attraction_paths(strength_grid, data.trajectories.shape[1])),
    ]:
        rmse = trajectory_rmse_matrix(data.trajectories, paths)
        for selection in ["shared", "condition"]:
            rows, selected = cross_validated_grid_model(
                data,
                rmse,
                strength_grid,
                model_name=f"{family_name}_{selection}",
                selection=selection,
            )
            model_rows.append(rows)
            selected["model"] = f"{family_name}_{selection}"
            selection_rows.append(selected)

    model_rows.append(condition_mean_cv(data))

    model_rows.append(fixed_path_cv(data, physical_grid.motor_only_path, "physical_A_motor_only"))
    model_rows.append(fixed_path_cv(data, physical_grid.target_only_path, "physical_B_target_only"))
    physical_rmse = trajectory_rmse_matrix(data.trajectories, physical_grid.target_paths)
    for selection, model_name in [
        ("shared", "physical_C_target_competitor_shared_rho"),
        ("condition", "physical_D_target_competitor_condition_rho"),
    ]:
        rows, selected = cross_validated_grid_model(
            data,
            physical_rmse,
            rhos,
            model_name=model_name,
            selection=selection,
        )
        model_rows.append(rows)
        selected["model"] = model_name
        selection_rows.append(selected)

    model_rows.append(
        trial_level_grid_fit(
            data,
            physical_rmse,
            rhos,
            model_name="physical_E_trial_level_rho_upper_bound",
        )
    )
    model_rows.append(
        trial_level_grid_fit(
            data,
            trajectory_rmse_matrix(data.trajectories, action_grid.target_paths),
            action_grid.rhos,
            model_name="nested_action_trial_level_rho_upper_bound",
        )
    )

    trials = pd.concat(model_rows, ignore_index=True, sort=False)
    summary = summarize_model_trials(trials)
    selected = pd.concat(selection_rows, ignore_index=True, sort=False) if selection_rows else pd.DataFrame()
    return trials, summary, selected


def fixed_path_cv(data: TrajectoryData, path: np.ndarray, model_name: str, n_splits: int = 5) -> pd.DataFrame:
    metadata = data.metadata.reset_index(drop=True)
    rmse = path_rmse_vector(data.trajectories, path)
    splitter = GroupKFold(n_splits=min(n_splits, metadata["subject"].nunique()))
    rows: list[dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(data.trajectories, groups=metadata["subject"]),
        start=1,
    ):
        sigma2 = max(float(np.mean(rmse[train_idx] ** 2)) / 2.0, 1e-8)
        for idx in test_idx:
            rows.append(
                {
                    "fold": fold,
                    "source_row": metadata.loc[idx, "source_row"],
                    "subject": metadata.loc[idx, "subject"],
                    "condition": metadata.loc[idx, "condition"],
                    "model": model_name,
                    "selection": "none",
                    "rmse": float(rmse[idx]),
                    "sigma2": sigma2,
                    "loglik": gaussian_path_loglik(float(rmse[idx]), sigma2, data.trajectories.shape[1]),
                }
            )
    return pd.DataFrame(rows)


def trial_level_grid_fit(
    data: TrajectoryData,
    rmse_matrix: np.ndarray,
    parameters: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    metadata = data.metadata.reset_index(drop=True)
    best = np.argmin(rmse_matrix, axis=1)
    best_rmse = rmse_matrix[np.arange(len(best)), best]
    sigma2 = max(float(np.mean(best_rmse**2)) / 2.0, 1e-8)
    return pd.DataFrame(
        {
            "fold": "not_cv",
            "source_row": metadata["source_row"],
            "subject": metadata["subject"],
            "condition": metadata["condition"],
            "model": model_name,
            "selection": "trial_level_fit_not_predictive",
            "parameter": parameters[best],
            "rmse": best_rmse,
            "sigma2": sigma2,
            "loglik": [
                gaussian_path_loglik(float(value), sigma2, data.trajectories.shape[1])
                for value in best_rmse
            ],
        }
    )


def summarize_model_trials(trials: pd.DataFrame) -> pd.DataFrame:
    by_condition = (
        trials.groupby(["model", "condition"], as_index=False)
        .agg(
            mean_rmse=("rmse", "mean"),
            sd_rmse=("rmse", "std"),
            mean_loglik=("loglik", "mean"),
            n=("rmse", "size"),
        )
        .sort_values(["condition", "model"])
    )
    overall = (
        trials.groupby(["model"], as_index=False)
        .agg(
            mean_rmse=("rmse", "mean"),
            sd_rmse=("rmse", "std"),
            mean_loglik=("loglik", "mean"),
            n=("rmse", "size"),
        )
        .assign(condition="All")
    )
    return pd.concat([by_condition, overall], ignore_index=True)


def cross_validated_rmse(
    data: TrajectoryData,
    rmse_matrix: np.ndarray,
    rhos: np.ndarray,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    metadata = data.metadata.reset_index(drop=True)
    motor_path = minimum_jerk_path(data.trajectories.shape[1])
    motor_rmse = path_rmse_vector(data.trajectories, motor_path)

    n_subjects = metadata["subject"].nunique()
    splitter = GroupKFold(n_splits=min(n_splits, n_subjects))
    rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(data.trajectories, groups=metadata["subject"]),
        start=1,
    ):
        best_by_condition = fit_condition_rhos(train_idx, metadata, rmse_matrix, rhos)
        for condition, fit in best_by_condition.items():
            selected.append(
                {
                    "fold": fold,
                    "condition": condition,
                    "rho": fit["rho"],
                    "train_rmse": fit["train_rmse"],
                    "train_mse": fit["train_mse"],
                }
            )

        motor_train_mse = float(np.mean(motor_rmse[train_idx] ** 2))
        motor_sigma2 = max(motor_train_mse / 2.0, 1e-8)
        for idx in test_idx:
            condition = metadata.loc[idx, "condition"]
            action_fit = best_by_condition[condition]
            action_sigma2 = max(action_fit["train_mse"] / 2.0, 1e-8)
            rows.append(
                {
                    "fold": fold,
                    "source_row": metadata.loc[idx, "source_row"],
                    "subject": metadata.loc[idx, "subject"],
                    "condition": condition,
                    "model": "minimum_jerk",
                    "rmse": motor_rmse[idx],
                    "sigma2": motor_sigma2,
                    "loglik": gaussian_path_loglik(motor_rmse[idx], motor_sigma2, data.trajectories.shape[1]),
                }
            )
            rows.append(
                {
                    "fold": fold,
                    "source_row": metadata.loc[idx, "source_row"],
                    "subject": metadata.loc[idx, "subject"],
                    "condition": condition,
                    "model": "action",
                    "rho": action_fit["rho"],
                    "rmse": rmse_matrix[idx, action_fit["rho_index"]],
                    "sigma2": action_sigma2,
                    "loglik": gaussian_path_loglik(
                        rmse_matrix[idx, action_fit["rho_index"]],
                        action_sigma2,
                        data.trajectories.shape[1],
                    ),
                }
            )

    trial_results = pd.DataFrame(rows)
    summary = (
        trial_results.groupby(["model", "condition"], as_index=False)
        .agg(mean_rmse=("rmse", "mean"), sd_rmse=("rmse", "std"), n=("rmse", "size"))
        .sort_values(["condition", "model"])
    )
    overall = (
        trial_results.groupby(["model"], as_index=False)
        .agg(mean_rmse=("rmse", "mean"), sd_rmse=("rmse", "std"), n=("rmse", "size"))
        .assign(condition="All")
    )
    summary = pd.concat([summary, overall], ignore_index=True)
    return trial_results, summary, selected


def fit_condition_rhos(
    train_idx: np.ndarray,
    metadata: pd.DataFrame,
    rmse_matrix: np.ndarray,
    rhos: np.ndarray,
) -> dict[str, dict[str, Any]]:
    fits: dict[str, dict[str, Any]] = {}
    for condition in sorted(metadata.loc[train_idx, "condition"].unique()):
        condition_idx = train_idx[metadata.loc[train_idx, "condition"].to_numpy() == condition]
        mean_rmse = rmse_matrix[condition_idx].mean(axis=0)
        best = int(np.argmin(mean_rmse))
        fits[str(condition)] = {
            "rho_index": best,
            "rho": float(rhos[best]),
            "train_rmse": float(mean_rmse[best]),
            "train_mse": float(np.mean(rmse_matrix[condition_idx, best] ** 2)),
        }
    return fits


def fit_trial_conflict(
    data: TrajectoryData,
    rmse_matrix: np.ndarray,
    action_grid,
) -> pd.DataFrame:
    best_index = np.argmin(rmse_matrix, axis=1)
    trial_fits = data.metadata.copy()
    trial_fits["rho_hat"] = action_grid.rhos[best_index]
    trial_fits["action_rmse"] = rmse_matrix[np.arange(len(best_index)), best_index]
    trial_fits["target_action"] = action_grid.target_actions[best_index]
    trial_fits["competitor_action"] = action_grid.competitor_actions[best_index]
    trial_fits["action_gap"] = trial_fits["competitor_action"] - trial_fits["target_action"]
    return trial_fits


def descriptive_tests(trial_fits: pd.DataFrame) -> dict[str, Any]:
    return {
        "auc_typicality": fit_regression("auc ~ atypical", trial_fits),
        "rt_typicality": fit_regression("rt_s ~ atypical", trial_fits),
    }


def rho_augmented_auc_test(trial_fits: pd.DataFrame) -> dict[str, Any]:
    df = trial_fits.copy()
    df["rho_hat_z"] = zscore(df["rho_hat"])
    df["action_gap_z"] = zscore(df["action_gap"])
    return {
        "auc_condition_plus_rho": fit_regression("auc ~ atypical + rho_hat_z", df),
        "auc_condition_plus_gap": fit_regression("auc ~ atypical + action_gap_z", df),
    }


def rho_condition_tests(trial_fits: pd.DataFrame) -> dict[str, Any]:
    condition_summary = (
        trial_fits.groupby("condition")
        .agg(mean_rho=("rho_hat", "mean"), sd_rho=("rho_hat", "std"), n=("rho_hat", "size"))
        .to_dict(orient="index")
    )
    by_subject = trial_fits.pivot_table(
        index="subject",
        columns="condition",
        values="rho_hat",
        aggfunc="mean",
    ).dropna()
    paired = stats.ttest_rel(by_subject["Atypical"], by_subject["Typical"])
    return {
        "condition_summary": _jsonify(condition_summary),
        "subject_paired_t": {
            "n_subjects": int(len(by_subject)),
            "mean_subject_diff_atypical_minus_typical": float(
                (by_subject["Atypical"] - by_subject["Typical"]).mean()
            ),
            "t": float(paired.statistic),
            "p": float(paired.pvalue),
        },
    }


def robust_inference(trial_fits: pd.DataFrame) -> dict[str, Any]:
    df = trial_fits.copy()
    df["rho_hat_z"] = zscore(df["rho_hat"])
    df["action_gap_z"] = zscore(df["action_gap"])
    return {
        "paired_subject": {
            "rho_hat": paired_subject_inference(df, "rho_hat"),
            "auc": paired_subject_inference(df, "auc"),
            "max_deviation": paired_subject_inference(df, "max_deviation"),
            "rt_s": paired_subject_inference(df, "rt_s"),
            "action_gap": paired_subject_inference(df, "action_gap"),
        },
        "cluster_regression": {
            "auc_beyond_condition_subject_from_rho": residualized_slope_inference(
                df,
                y_col="auc",
                x_col="rho_hat_z",
            ),
            "rt_beyond_condition_subject_from_action_gap": residualized_slope_inference(
                df,
                y_col="rt_s",
                x_col="action_gap_z",
            ),
            "auc_beyond_condition_subject_from_action_gap": residualized_slope_inference(
                df,
                y_col="auc",
                x_col="action_gap_z",
            ),
        },
    }


def counterfactual_tests(trial_fits: pd.DataFrame) -> dict[str, Any]:
    df = trial_fits.copy()
    df["abs_action_gap"] = df["action_gap"].abs()
    df["inverse_abs_action_gap"] = 1.0 / np.maximum(df["abs_action_gap"], 1e-4)
    df["rho_hat_z"] = zscore(df["rho_hat"])
    df["action_gap_z"] = zscore(df["action_gap"])
    df["inverse_abs_action_gap_z"] = zscore(df["inverse_abs_action_gap"])
    return {
        "rt_from_action_gap": fit_regression(
            "rt_s ~ atypical + action_gap_z + rho_hat_z",
            df,
        ),
        "rt_from_inverse_gap": fit_regression(
            "rt_s ~ atypical + inverse_abs_action_gap_z + rho_hat_z",
            df,
        ),
        "auc_from_action_gap": fit_regression(
            "auc ~ atypical + action_gap_z + rho_hat_z",
            df,
        ),
        "robust_rt_inverse_gap": residualized_slope_inference(
            df,
            y_col="rt_s",
            x_col="inverse_abs_action_gap_z",
        ),
    }


def item_level_analysis(
    trial_fits: pd.DataFrame,
    raw: pd.DataFrame,
    semantic_scores_path: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = trial_fits.copy()
    work["competitor_category"] = np.where(
        work["category_correct"] == work["category_left"],
        work["category_right"],
        work["category_left"],
    )
    item_summary = (
        work.groupby(["exemplar", "condition", "category_correct", "competitor_category"], as_index=False)
        .agg(
            n_correct=("rho_hat", "size"),
            rho_hat=("rho_hat", "mean"),
            action_gap=("action_gap", "mean"),
            auc=("auc", "mean"),
            max_deviation=("max_deviation", "mean"),
            rt_s=("rt_s", "mean"),
        )
    )

    raw_item = (
        raw.assign(correct_numeric=raw["correct"].astype(float))
        .groupby("Exemplar", as_index=False)
        .agg(
            n_raw=("correct_numeric", "size"),
            error_rate=("correct_numeric", lambda values: float(1.0 - values.mean())),
            raw_rt_s=("response_time", lambda values: float(np.mean(values) / 1000.0)),
        )
        .rename(columns={"Exemplar": "exemplar"})
    )
    item_summary = item_summary.merge(raw_item, on="exemplar", how="left")
    semantic_columns = ["semantic_similarity_target", "semantic_similarity_competitor", "semantic_margin"]
    if semantic_scores_path:
        semantic_scores = pd.read_csv(semantic_scores_path)
        item_summary = item_summary.merge(semantic_scores, on="exemplar", how="left")
    for column in semantic_columns:
        if column not in item_summary:
            item_summary[column] = np.nan

    tests = {
        "rho_vs_error_rate_spearman": spearman_dict(item_summary, "rho_hat", "error_rate"),
        "rho_vs_item_rt_spearman": spearman_dict(item_summary, "rho_hat", "raw_rt_s"),
        "action_gap_vs_error_rate_spearman": spearman_dict(item_summary, "action_gap", "error_rate"),
        "rho_vs_semantic_competitor_similarity": spearman_dict(
            item_summary,
            "rho_hat",
            "semantic_similarity_competitor",
        ),
        "rho_vs_semantic_margin": spearman_dict(item_summary, "rho_hat", "semantic_margin"),
        "semantic_scores_path": semantic_scores_path,
        "semantic_columns": semantic_columns,
        "semantic_note": (
            "External embedding or lexical semantic scores can be merged into "
            "results/item_level_action_summary.csv using these columns or passed "
            "directly with --semantic-scores."
        ),
    }
    return item_summary, tests


def spearman_dict(df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, Any]:
    clean = df[[x_col, y_col]].dropna()
    if len(clean) < 3:
        return {"n_items": int(len(clean)), "rho": None, "p": None}
    result = stats.spearmanr(clean[x_col], clean[y_col])
    return {
        "n_items": int(len(clean)),
        "rho": float(result.statistic),
        "p": float(result.pvalue),
    }


def fit_regression(formula: str, df: pd.DataFrame) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = smf.mixedlm(formula, df, groups=df["subject"]).fit(
                reml=False,
                method="lbfgs",
                maxiter=1000,
                disp=False,
            )
            return {
                "estimator": "mixedlm_random_subject_intercept",
                "formula": formula,
                "aic": float(result.aic),
                "params": _jsonify(result.params.to_dict()),
                "pvalues": _jsonify(result.pvalues.to_dict()),
            }
        except Exception as exc:
            result = smf.ols(f"{formula} + C(subject)", df).fit()
            return {
                "estimator": "ols_subject_fixed_effect_fallback",
                "formula": formula,
                "fallback_reason": str(exc),
                "aic": float(result.aic),
                "params": _jsonify(result.params.to_dict()),
                "pvalues": _jsonify(result.pvalues.to_dict()),
            }


def build_summary(
    data: TrajectoryData,
    cv_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    trial_fits: pd.DataFrame,
    descriptive: dict[str, Any],
    rho_tests: dict[str, Any],
    rho_augmented: dict[str, Any],
    robust: dict[str, Any],
    counterfactual: dict[str, Any],
    item_tests: dict[str, Any],
    recovery_summary: dict[str, Any],
    best_full: dict[str, dict[str, Any]],
    selected_rhos: list[dict[str, Any]],
    action_grid,
    physical_grid,
    semantic_prior_results: dict[str, Any] | None = None,
    stochastic_nll_summary: pd.DataFrame | None = None,
) -> dict[str, Any]:
    cv_pivot = {
        f"{row['model']}:{row['condition']}": float(row["mean_rmse"])
        for _, row in cv_summary.iterrows()
    }
    improvement_all = cv_pivot["minimum_jerk:All"] - cv_pivot["action:All"]
    improvement_atypical = cv_pivot["minimum_jerk:Atypical"] - cv_pivot["action:Atypical"]
    improvement_typical = cv_pivot["minimum_jerk:Typical"] - cv_pivot["action:Typical"]
    return {
        "dataset": {
            "source_url": KH2017_RDA_URL,
            "n_raw_rows": 1140,
            "n_correct_canonicalized_trials": int(len(data.metadata)),
            "n_subjects": int(data.metadata["subject"].nunique()),
            "conditions": _jsonify(data.metadata["condition"].value_counts().to_dict()),
        },
        "action_grid": {
            "rho_min": float(action_grid.rhos.min()),
            "rho_max": float(action_grid.rhos.max()),
            "rho_step": float(np.diff(action_grid.rhos).min()) if len(action_grid.rhos) > 1 else None,
            "all_paths_finite": bool(
                np.isfinite(action_grid.target_actions).all()
                and np.isfinite(action_grid.competitor_actions).all()
            ),
            "optimizer_reported_success_n": int(action_grid.converged.sum()),
            "n_grid": int(len(action_grid.rhos)),
        },
        "physical_action_ablation_grid": {
            "rho_min": float(physical_grid.rhos.min()),
            "rho_max": float(physical_grid.rhos.max()),
            "rho_step": float(np.diff(physical_grid.rhos).min()) if len(physical_grid.rhos) > 1 else None,
            "all_paths_finite": bool(np.isfinite(physical_grid.target_actions).all()),
            "optimizer_reported_success_n": int(physical_grid.converged.sum()),
            "n_grid": int(len(physical_grid.rhos)),
        },
        "rho_bookkeeping": {
            "full_data_condition_rhos": "One rho per condition, optimized on the full dataset for visualization only.",
            "fold_selected_rhos": "One rho per condition per training fold, used for predictive subject-wise CV.",
            "trial_level_rho_hat": "One rho per trial, fit to that observed trajectory; used for latent-variable interpretation, not as out-of-sample prediction.",
        },
        "full_data_condition_rhos": _jsonify(best_full),
        "cross_validated_rmse": _jsonify(cv_summary.to_dict(orient="records")),
        "strong_model_comparison": _jsonify(model_summary.to_dict(orient="records")),
        "rmse_improvement": {
            "all": float(improvement_all),
            "atypical": float(improvement_atypical),
            "typical": float(improvement_typical),
        },
        "selected_rhos_by_fold": _jsonify(selected_rhos),
        "rho_condition_tests": rho_tests,
        "robust_inference": robust,
        "descriptive_tests": descriptive,
        "rho_augmented_tests": rho_augmented,
        "counterfactual_tests": counterfactual,
        "item_level_tests": item_tests,
        "parameter_recovery": recovery_summary,
        "semantic_prior_results": semantic_prior_results,
        "stochastic_nll_summary": _jsonify(stochastic_nll_summary.to_dict(orient="records")) if stochastic_nll_summary is not None else None,
        "trial_metric_means": _jsonify(
            trial_fits.groupby("condition")
            .agg(
                auc=("auc", "mean"),
                max_deviation=("max_deviation", "mean"),
                rt_s=("rt_s", "mean"),
                rho_hat=("rho_hat", "mean"),
                action_gap=("action_gap", "mean"),
            )
            .to_dict(orient="index")
        ),
    }


def print_summary(summary: dict[str, Any], results_dir: Path) -> None:
    print("\nA Stochastic Least-Action Model Links Semantic Typicality Norms to Dynamic Response Competition")
    print("===============================================================================================")
    
    print("\n--- Manuscript-grade abstract ---")
    print("To reduce circularity in the estimation of the cognitive potential landscape, we derived item-level")
    print("semantic margins from a transparent repository semantic table and used these margins to parameterize")
    print("the competitor-attraction term rho. The resulting semantic-prior action model was evaluated under held-out")
    print("stochastic path likelihood, rather than only trajectory RMSE. This model links external semantic structure")
    print("to the geometry of the action landscape and then to behavioral difficulty, providing evidence that response")
    print("competition can be formalized as semantic deformation of a cognitive potential field.\n")

    print(
        "Trials:",
        summary["dataset"]["n_correct_canonicalized_trials"],
        "correct trajectories from",
        summary["dataset"]["n_subjects"],
        "subjects",
    )

    if summary.get("stochastic_nll_summary"):
        print("\nStochastic NLL Model Hierarchy (LOO-Item CV):")
        # Parse it from the jsonify list of dicts
        nll_sum = summary["stochastic_nll_summary"]
        overall = [r for r in nll_sum if r["condition"] == "All"]
        overall.sort(key=lambda item: item["mean_loglik"], reverse=True)
        for row in overall[:10]:
            print(f"  {row['model']:<42s} mean_nll={-row['mean_loglik']:.3f} n={row['n']}")
            
        iw = summary.get("item_wise_nll", {})
        if iw:
            print("\nItem-wise NLL test (Condition-only vs Semantic-only):")
            print(f"  Mean NLL Action(Condition) = {iw['mean_nll_condition']:.3f}")
            print(f"  Mean NLL Action(Semantic)  = {iw['mean_nll_semantic']:.3f}")
            print(f"  Mean NLL Action(Cond+Sem)  = {iw['mean_nll_full']:.3f}")
            print(f"  Delta NLL (Cond - Sem)     = {iw['mean_delta_nll_per_item']:.3f} per item (Total Delta NLL = {iw['total_delta_nll']:.3f} across trials)")
            print(f"  Positive gain items        : {iw['n_items_positive_gain']} out of {iw['n_items_total']}")
            print(f"  Paired t-test              : t={iw['paired_t']:.3f}, p={iw['p_t']:.4f}")
            print(f"  Wilcoxon sign-rank         : W={iw['wilcoxon_w']:.3f}, p={iw['p_w']:.4f}")

    print("\nFitted rho by condition:")
    rho_summary = summary["rho_condition_tests"]["condition_summary"]
    for condition, values in rho_summary.items():
        print(f"  {condition:<8s}: mean={values['mean_rho']:.3f} sd={values['sd_rho']:.3f}")

    if summary.get("semantic_prior_results"):
        sem_res = summary["semantic_prior_results"]
        if "margin_vs_rho_spearman" in sem_res:
            mvr = sem_res["margin_vs_rho_spearman"]
            print(f"\nItem-level margin vs fitted rho (Spearman): r={mvr['spearman_rho']:.3f}, p={mvr['p']:.4f}")
            print(f"  Expected constraint (lower margin -> higher rho) satisfied: {mvr['expected_sign_satisfied']}")
        if "downstream" in sem_res:
            rt_res = sem_res["downstream"].get("rt")
            err_res = sem_res["downstream"].get("error_rate")
            if rt_res:
                print(f"LOOCV semantic-predicted rho vs RT: R2={rt_res['r2']:.3f}, expected sign={rt_res['expected_sign_satisfied']}")
            if err_res:
                print(f"LOOCV semantic-predicted rho vs Error Rate: R2={err_res['r2']:.3f}, expected sign={err_res['expected_sign_satisfied']}")

    recovery = summary["parameter_recovery"]
    print(
        "\nParameter recovery:",
        f"corr={recovery['overall_correlation']:.3f}",
        f"MAE={recovery['overall_mae']:.3f}",
    )
    print(f"\nWrote outputs to {results_dir.resolve()}")


def plot_trajectory_fit(
    data: TrajectoryData,
    action_grid,
    best_full: dict[str, dict[str, Any]],
    destination: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    colors = {"Typical": "#2a9d8f", "Atypical": "#c44536"}
    for condition, color in colors.items():
        idx = data.metadata["condition"].to_numpy() == condition
        mean_path = data.trajectories[idx].mean(axis=0)
        ax.plot(mean_path[:, 0], mean_path[:, 1], color=color, lw=2.5, label=f"{condition} observed")
        action_path = action_grid.target_paths[best_full[condition]["rho_index"]]
        ax.plot(action_path[:, 0], action_path[:, 1], color=color, lw=2.0, ls="--", label=f"{condition} action")

    motor = minimum_jerk_path(data.trajectories.shape[1])
    ax.plot(motor[:, 0], motor[:, 1], color="#555555", lw=1.8, ls=":", label="minimum jerk")
    ax.scatter([1.0, -1.0, 0.0], [1.0, 1.0, 0.0], c=["#222222", "#777777", "#222222"], s=[55, 45, 35])
    ax.text(1.03, 1.0, "target", va="center")
    ax.text(-0.96, 1.0, "competitor", ha="left", va="center")
    ax.set_xlabel("canonical x")
    ax.set_ylabel("canonical y")
    ax.set_xlim(-1.15, 1.12)
    ax.set_ylim(-0.03, 1.08)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_rho_by_condition(trial_fits: pd.DataFrame, destination: Path) -> None:
    colors = {"Typical": "#2a9d8f", "Atypical": "#c65a4a"}
    fig, ax = plt.subplots(figsize=(4.7, 3.9))
    order = ["Typical", "Atypical"]
    values = [trial_fits.loc[trial_fits["condition"] == condition, "rho_hat"] for condition in order]
    box = ax.boxplot(
        values,
        tick_labels=order,
        showfliers=False,
        patch_artist=True,
        widths=0.58,
        medianprops={"color": "#222222", "linewidth": 1.4},
        whiskerprops={"color": "#595959", "linewidth": 1.0},
        capprops={"color": "#595959", "linewidth": 1.0},
    )
    for patch, condition in zip(box["boxes"], order):
        patch.set_facecolor(colors[condition])
        patch.set_alpha(0.55)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(1.0)
    means = [float(v.mean()) for v in values]
    ax.scatter([1, 2], means, marker="D", s=38, color="#222222", edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_ylabel(r"Fitted competitor attraction $\rho$", fontsize=11)
    ax.tick_params(axis="both", labelsize=9.5)
    ax.margins(x=0.08)
    ax.grid(axis="y", color="#e7e7e7", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(destination, dpi=300)
    plt.close(fig)


def plot_subject_paired_rho(trial_fits: pd.DataFrame, destination: Path) -> None:
    by_subject = trial_fits.pivot_table(
        index="subject",
        columns="condition",
        values="rho_hat",
        aggfunc="mean",
    ).dropna()
    paired = paired_subject_inference(trial_fits, "rho_hat")
    diffs = (by_subject["Atypical"] - by_subject["Typical"]).to_numpy(float)

    fig, (ax, ax_delta) = plt.subplots(
        1,
        2,
        figsize=(7.0, 4.4),
        gridspec_kw={"width_ratios": [2.4, 1.0], "wspace": 0.34},
    )
    x = np.array([0.0, 1.0])
    for _, row in by_subject.iterrows():
        ax.plot(x, [row["Typical"], row["Atypical"]], color="#9aa0a6", lw=0.65, alpha=0.34)

    means = np.array([by_subject["Typical"].mean(), by_subject["Atypical"].mean()])
    values = by_subject[["Typical", "Atypical"]].to_numpy(float)
    rng = np.random.default_rng(20260513)
    boot_idx = rng.integers(0, len(by_subject), size=(5000, len(by_subject)))
    boot_means = values[boot_idx].mean(axis=1)
    mean_ci = np.quantile(boot_means, [0.025, 0.975], axis=0)
    mean_yerr = np.vstack((means - mean_ci[0], mean_ci[1] - means))
    ax.errorbar(
        x,
        means,
        yerr=mean_yerr,
        color="#c44536",
        marker="o",
        markersize=5.5,
        lw=2.6,
        capsize=4,
        label="mean across subjects +/- bootstrap 95% CI",
    )
    ax.set_xticks(x, ["Typical", "Atypical"])
    ax.set_ylabel(r"Mean fitted $\rho$")
    ax.legend(frameon=False, fontsize=8)

    delta_mean = paired["mean_diff_atypical_minus_typical"]
    delta_ci_low, delta_ci_high = paired["bootstrap_ci_95"]
    jitter = rng.normal(0.0, 0.018, size=len(diffs))
    ax_delta.axhline(0.0, color="#5f6368", lw=0.9, linestyle="--", zorder=1)
    ax_delta.scatter(jitter, diffs, color="#9aa0a6", s=11, alpha=0.34, linewidth=0, zorder=2)
    ax_delta.errorbar(
        [0.0],
        [delta_mean],
        yerr=[[delta_mean - delta_ci_low], [delta_ci_high - delta_mean]],
        color="#c44536",
        marker="o",
        markersize=6,
        lw=2.6,
        capsize=4,
        zorder=3,
    )
    ax_delta.set_xlim(-0.18, 0.18)
    ax_delta.set_xticks([0.0], [r"$\Delta\rho$"])
    ax_delta.set_ylabel(r"$\Delta\rho$ (Atypical - Typical)")

    for axis in (ax, ax_delta):
        axis.grid(axis="y", color="#e7e7e7", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="both", labelsize=9.5)

    fig.tight_layout()
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0:
        return series * 0.0
    return (series - series.mean()) / std


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    config = load_model_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--n-time", type=int, default=int(config_value(config, "model.n_time", 51)))
    parser.add_argument("--rho-min", type=float, default=float(config_value(config, "model.rho_grid.min", 0.0)))
    parser.add_argument("--rho-max", type=float, default=float(config_value(config, "model.rho_grid.max", 2.0)))
    parser.add_argument("--rho-step", type=float, default=float(config_value(config, "model.rho_grid.step", 0.05)))
    parser.add_argument("--maxiter", type=int, default=int(config_value(config, "model.optimizer.maxiter", 500)))
    parser.add_argument(
        "--semantic-scores",
        default=None,
        help=(
            "Optional CSV with exemplar, semantic_similarity_target, "
            "semantic_similarity_competitor, and semantic_margin columns."
        ),
    )
    return parser.parse_args(argv)

def plot_decisive_four_panel(
    item_summary: pd.DataFrame,
    semantic_prior_results: dict[str, Any],
    stochastic_nll_summary: pd.DataFrame,
    destination: Path,
) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(11.2, 8.6))
    colors = {"Typical": "#2a9d8f", "Atypical": "#c65a4a"}
    label_items = {
        "Wal",
        "Fledermaus",
        "Pinguin",
        "Schmetterling",
        "Aal",
        "Klapperschlange",
        "Hund",
        "Loewe",
    }
    label_offsets = {
        "Wal": (-24, -12),
        "Fledermaus": (6, 8),
        "Pinguin": (8, -18),
        "Schmetterling": (-54, -4),
        "Aal": (6, 8),
        "Klapperschlange": (-72, -2),
        "Hund": (6, 7),
        "Loewe": (6, -11),
    }
    
    df = item_summary.copy()
    if "rho_predicted_semantic" not in df.columns and "item_predictions" in semantic_prior_results:
        preds = pd.DataFrame(semantic_prior_results["item_predictions"])
        df = df.merge(preds[["exemplar", "rho_predicted_semantic"]], on="exemplar", how="left")
    df = df.dropna(subset=["semantic_margin", "rho_hat"])

    # A: Semantic margin vs fitted rho
    ax = axs[0, 0]
    for cond, cdf in df.groupby("condition"):
        ax.scatter(
            cdf["semantic_margin"],
            cdf["rho_hat"],
            label=cond,
            color=colors.get(str(cond), "grey"),
            alpha=0.9,
            s=56,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        for _, row in cdf.iterrows():
            if row["exemplar"] in label_items:
                dx, dy = label_offsets.get(row["exemplar"], (5, 4))
                ax.annotate(
                    row["exemplar"],
                    (row["semantic_margin"], row["rho_hat"]),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8,
                    color="#222222",
                    alpha=0.9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.5),
                )
    x, y = df["semantic_margin"].to_numpy(), df["rho_hat"].to_numpy()
    if len(np.unique(x)) > 1:
        m, b = np.polyfit(x, y, 1)
        xl = np.linspace(x.min(), x.max(), 10)
        ax.plot(xl, m * xl + b, color="#555555", lw=1.3, ls="--", alpha=0.75, zorder=2)
    ax.axvline(0, color="#c7c7c7", ls=":", lw=0.9, zorder=1)
    ax.set_xlim(x.min() - 0.045, x.max() + 0.06)
    ax.set_ylim(y.min() - 0.035, y.max() + 0.055)
    ax.text(-0.10, 1.04, "A", transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom")
    ax.set_xlabel("Semantic margin", fontsize=11)
    ax.set_ylabel(r"Mean fitted $\rho$", fontsize=11)
    ax.tick_params(axis="both", labelsize=9.5)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")

    # B: LOOCV Predicted rho vs Fitted rho
    ax = axs[0, 1]
    for cond, cdf in df.groupby("condition"):
        if "rho_predicted_semantic" in cdf.columns:
            ax.scatter(
                cdf["rho_predicted_semantic"],
                cdf["rho_hat"],
                label=cond,
                color=colors.get(str(cond), "grey"),
                alpha=0.9,
                s=56,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
    if "rho_predicted_semantic" in df.columns:
        lo = min(df["rho_predicted_semantic"].min(), df["rho_hat"].min()) - 0.05
        hi = max(df["rho_predicted_semantic"].max(), df["rho_hat"].max()) + 0.05
        ax.plot([lo, hi], [lo, hi], color="#9d9d9d", lw=1.0, ls="--", zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.text(-0.10, 1.04, "B", transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom")
    ax.set_xlabel(r"Predicted $\hat{\rho}$ (LOO)", fontsize=11)
    ax.set_ylabel(r"Observed item-level $\hat{\rho}$", fontsize=11)
    ax.tick_params(axis="both", labelsize=9.5)
    ax.set_aspect("equal", adjustable="box")

    # C: Stochastic NLL Comparison
    ax = axs[1, 0]
    overall = stochastic_nll_summary[stochastic_nll_summary["condition"] == "All"].copy()
    overall["mean_nll"] = -overall["mean_loglik"]
    
    model_order = [
        "baseline_condition_mean",
        "baseline_minimum_jerk",
        "bezier_condition",
        "bezier_item",
        "spline_condition",
        "spline_item",
        "action_condition_only_rho",
        "action_semantic_margin_only_rho",
        "action_condition_plus_semantic_rho",
        "action_trial_fitted_rho"
    ]
    overall = overall[overall["model"].isin(model_order)]
    overall["model"] = pd.Categorical(overall["model"], categories=model_order, ordered=True)
    overall = overall.sort_values("model").dropna()
    
    labels = {
        "baseline_condition_mean": "Condition mean",
        "baseline_minimum_jerk": "Minimum jerk",
        "bezier_condition": "Bezier condition",
        "bezier_item": "Bezier item",
        "spline_condition": "Spline condition",
        "spline_item": "Spline item",
        "action_condition_only_rho": "Action: condition",
        "action_semantic_margin_only_rho": "Action: semantic",
        "action_condition_plus_semantic_rho": "Action: cond+semantic",
        "action_trial_fitted_rho": "Action: trial fitted"
    }
    y_pos = np.arange(len(overall))
    bar_colors = []
    for model in overall["model"].astype(str):
        if model == "action_semantic_margin_only_rho":
            bar_colors.append("#2a9d8f")
        elif model == "action_condition_only_rho":
            bar_colors.append("#c65a4a")
        elif model.startswith("action_"):
            bar_colors.append("#7f9aa3")
        else:
            bar_colors.append("#c6c6c6")
    ax.barh(y_pos, overall["mean_nll"], align="center", color=bar_colors, alpha=0.9)
    ax.set_yticks(y_pos, labels=[labels.get(m, m) for m in overall["model"]])
    ax.invert_yaxis()
    ax.set_xlabel("Held-out NLL (lower is better)", fontsize=11)
    ax.text(-0.28, 1.04, "C", transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom")
    ax.tick_params(axis="x", labelsize=9.5)
    ax.tick_params(axis="y", labelsize=8.5)
    for idx, model in enumerate(overall["model"].astype(str)):
        if model in {"action_condition_only_rho", "action_semantic_margin_only_rho"}:
            ax.text(
                overall.iloc[idx]["mean_nll"] + 0.02,
                idx,
                f"{overall.iloc[idx]['mean_nll']:.2f}",
                va="center",
                fontsize=8.5,
                color="#222222",
            )
    
    min_nll = overall["mean_nll"].min()
    max_nll = overall["mean_nll"].max()
    padding = (max_nll - min_nll) * 0.1 if (max_nll - min_nll) > 0 else 1
    ax.set_xlim(min_nll - padding*2, max_nll + padding)

    # D: Predicted Rho vs one behavioral metric
    ax = axs[1, 1]
    if "rho_predicted_semantic" in df.columns:
        rt_col = "rt_s" if "rt_s" in df.columns else "raw_rt_s"
        for cond, cdf in df.groupby("condition"):
            ax.scatter(
                cdf["rho_predicted_semantic"],
                cdf[rt_col],
                color=colors.get(str(cond), "grey"),
                alpha=0.9,
                s=56,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        x = df["rho_predicted_semantic"].to_numpy()
        y = df[rt_col].to_numpy()
        m, b = np.polyfit(x, y, 1)
        xl = np.linspace(x.min(), x.max(), 10)
        ax.plot(xl, m * xl + b, color="#555555", ls="--", lw=1.3, alpha=0.75, zorder=2)
        ax.text(-0.10, 1.04, "D", transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom")
        ax.set_xlabel(r"Predicted $\hat{\rho}$ (LOO)", fontsize=11)
        ax.set_ylabel("Mean response time (s)", fontsize=11)
        ax.tick_params(axis="both", labelsize=9.5)

    for ax in axs.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#ececec", linewidth=0.6)
        ax.set_axisbelow(True)

    fig.tight_layout(pad=1.3, w_pad=2.1, h_pad=2.2)
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    if destination.suffix.lower() == ".png":
        fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    import sys
    sys.exit(main())
