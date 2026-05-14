"""stochastic_action.py
======================
Stochastic action likelihood model comparison.

Replaces the RMSE-based model comparison with held-out **negative log-likelihood**
(NLL) under a Gaussian path deviation model::

    q_obs(t) = q*_action(t) + ε(t),   ε(t) ~ N(0, τ²I)

The path noise τ is estimated strictly from training data per model per fold.

Model hierarchy
---------------
  1. ``baseline_minimum_jerk``           — no free parameters
  2. ``baseline_condition_mean``         — condition-mean trajectory
  3. ``bezier_condition``                — attraction strength, condition-specific
  4. ``spline_condition``                — attraction strength, condition-specific
  5. ``action_condition_only_rho``       — ρ, condition-specific
  6. ``action_semantic_margin_only_rho`` — ρ predicted from semantic margin
  7. ``action_condition_plus_semantic_rho`` — ρ predicted from condition + margin
  8. ``action_trial_fitted_rho``         — ρ fitted to test trial (Upper Bound)
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
import statsmodels.formula.api as smf

from .baselines import (
    bezier_attraction_paths,
    spline_attraction_paths,
    gaussian_path_loglik,
)
from .preprocess import TrajectoryData, minimum_jerk_path


def _fit_tau_from_rmse(rmse_vals: np.ndarray) -> float:
    """Estimate τ² from training RMSE values (MLE: τ² = mean(RMSE²)/2)."""
    return float(max(np.mean(rmse_vals**2) / 2.0, 1e-8))


def stochastic_likelihood_comparison(
    data: TrajectoryData,
    action_grid: Any,
    semantic_scores: pd.DataFrame | None,
    item_summary: pd.DataFrame | None = None,
    n_splits: int = 19,
    grouping_col: str = "exemplar",
) -> pd.DataFrame:
    """Compare models using held-out NLL under the Gaussian path model."""
    from .analysis import (
        trajectory_rmse_matrix,
        path_rmse_vector,
        fit_condition_rhos,
    )

    metadata = data.metadata.reset_index(drop=True)
    if semantic_scores is not None and "semantic_margin" not in metadata.columns:
        metadata = metadata.merge(
            semantic_scores[["exemplar", "semantic_margin"]],
            on="exemplar",
            how="left"
        )
    
    # Handle Atypical/Typical to numeric for OLS
    if "condition" in metadata.columns:
        metadata["atypical"] = (metadata["condition"] == "Atypical").astype(float)

    if grouping_col == "exemplar" and "exemplar" in metadata.columns:
        splitter = LeaveOneGroupOut()
    else:
        grouping_col = "subject"
        splitter = GroupKFold(n_splits=min(n_splits, metadata[grouping_col].nunique()))

    n_time = data.trajectories.shape[1]

    # Pre-compute RMSE matrices
    action_rmse_matrix = trajectory_rmse_matrix(data.trajectories, action_grid.target_paths)
    motor_path = minimum_jerk_path(n_time)
    motor_rmse = path_rmse_vector(data.trajectories, motor_path)

    strength_grid = np.round(np.arange(0.0, 1.5001, 0.05), 10)
    bezier_paths = bezier_attraction_paths(strength_grid, n_time)
    bezier_rmse_matrix = trajectory_rmse_matrix(data.trajectories, bezier_paths)
    
    spline_paths = spline_attraction_paths(strength_grid, n_time)
    spline_rmse_matrix = trajectory_rmse_matrix(data.trajectories, spline_paths)

    rows: list[dict[str, object]] = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(data.trajectories, groups=metadata[grouping_col]),
        start=1,
    ):
        # ── Model 1: minimum jerk ───────────────────────────────────────────
        tau2_mj = _fit_tau_from_rmse(motor_rmse[train_idx])
        for idx in test_idx:
            rows.append(_make_row("baseline_minimum_jerk", fold, metadata, idx, motor_rmse[idx], tau2_mj, n_time))

        # ── Model 2: condition mean ──────────────────────────────────────────
        cond_means = {}
        for cond in metadata.loc[train_idx, "condition"].unique():
            idx_cond = train_idx[metadata.loc[train_idx, "condition"].to_numpy() == cond]
            cond_means[cond] = np.mean(data.trajectories[idx_cond], axis=0)
        
        train_rmses_cm = []
        for tidx in train_idx:
            cond = metadata.loc[tidx, "condition"]
            train_rmses_cm.append(path_rmse_vector(data.trajectories[[tidx]], cond_means[cond])[0])
        tau2_cm = _fit_tau_from_rmse(np.array(train_rmses_cm))
        
        for idx in test_idx:
            cond = metadata.loc[idx, "condition"]
            if cond in cond_means:
                rmse = path_rmse_vector(data.trajectories[[idx]], cond_means[cond])[0]
                rows.append(_make_row("baseline_condition_mean", fold, metadata, idx, rmse, tau2_cm, n_time))

        # ── Model 3, 4, 3b, 4b: Bezier & Spline (condition and item) ──────────
        for name, rmse_mat, group_key in [
            ("bezier_condition", bezier_rmse_matrix, "condition"), 
            ("spline_condition", spline_rmse_matrix, "condition"),
            ("bezier_item", bezier_rmse_matrix, "exemplar"),
            ("spline_item", spline_rmse_matrix, "exemplar"),
        ]:
            if group_key not in metadata.columns:
                continue
                
            best_by_group: dict[str, int] = {}
            for grp in metadata.loc[train_idx, group_key].unique():
                c_idx = train_idx[metadata.loc[train_idx, group_key].to_numpy() == grp]
                best_by_group[grp] = int(np.argmin(rmse_mat[c_idx].mean(axis=0)))
            
            train_rmses_bz = []
            for tidx in train_idx:
                grp = metadata.loc[tidx, group_key]
                train_rmses_bz.append(rmse_mat[tidx, best_by_group[grp]])
            tau2_bz = _fit_tau_from_rmse(np.array(train_rmses_bz))
            
            for idx in test_idx:
                grp = metadata.loc[idx, group_key]
                if grp in best_by_group:
                    rmse = rmse_mat[idx, best_by_group[grp]]
                    rows.append(_make_row(name, fold, metadata, idx, rmse, tau2_bz, n_time))


        # ── Model 5: Action, condition-only ρ ──────────────────────────────
        best_action = fit_condition_rhos(train_idx, metadata, action_rmse_matrix, action_grid.rhos)
        train_rmses_ac = []
        for tidx in train_idx:
            cond = metadata.loc[tidx, "condition"]
            train_rmses_ac.append(action_rmse_matrix[tidx, best_action[cond]["rho_index"]])
        tau2_ac = _fit_tau_from_rmse(np.array(train_rmses_ac))
        
        for idx in test_idx:
            cond = metadata.loc[idx, "condition"]
            rmse = action_rmse_matrix[idx, best_action[cond]["rho_index"]]
            rows.append(_make_row("action_condition_only_rho", fold, metadata, idx, rmse, tau2_ac, n_time))

        # ── Semantic Models ──────────────────────────────────────────────────
        if "semantic_margin" in metadata.columns and metadata["semantic_margin"].notna().any():
            # Get best fit rho for each training trial to serve as target for OLS
            train_best_rhos = action_grid.rhos[np.argmin(action_rmse_matrix[train_idx], axis=1)]
            train_df = metadata.loc[train_idx].copy()
            train_df["trial_rho"] = train_best_rhos
            train_df = train_df.dropna(subset=["semantic_margin"])
            
            # Model 6: Semantic Margin Only
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_sem = smf.ols("trial_rho ~ semantic_margin", train_df).fit()
            
            # Predict and evaluate
            train_rmses_sem = []
            for tidx in train_idx:
                if pd.isna(metadata.loc[tidx, "semantic_margin"]):
                    train_rmses_sem.append(np.nan)
                    continue
                pred = fit_sem.predict(metadata.loc[[tidx]]).values[0]
                pred_idx = _nearest_rho_index(pred, action_grid.rhos)
                train_rmses_sem.append(action_rmse_matrix[tidx, pred_idx])
            tau2_sem = _fit_tau_from_rmse(np.array([r for r in train_rmses_sem if not np.isnan(r)]))
            
            for idx in test_idx:
                if pd.isna(metadata.loc[idx, "semantic_margin"]):
                    continue
                pred = fit_sem.predict(metadata.loc[[idx]]).values[0]
                pred_idx = _nearest_rho_index(pred, action_grid.rhos)
                rmse = action_rmse_matrix[idx, pred_idx]
                rows.append(_make_row("action_semantic_margin_only_rho", fold, metadata, idx, rmse, tau2_sem, n_time, {"semantic_prior_rho": pred}))
                
            # Model 7: Condition + Semantic Margin
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_full = smf.ols("trial_rho ~ atypical + semantic_margin", train_df).fit()
            
            train_rmses_full = []
            for tidx in train_idx:
                if pd.isna(metadata.loc[tidx, "semantic_margin"]):
                    train_rmses_full.append(np.nan)
                    continue
                pred = fit_full.predict(metadata.loc[[tidx]]).values[0]
                pred_idx = _nearest_rho_index(pred, action_grid.rhos)
                train_rmses_full.append(action_rmse_matrix[tidx, pred_idx])
            tau2_full = _fit_tau_from_rmse(np.array([r for r in train_rmses_full if not np.isnan(r)]))
            
            for idx in test_idx:
                if pd.isna(metadata.loc[idx, "semantic_margin"]):
                    continue
                pred = fit_full.predict(metadata.loc[[idx]]).values[0]
                pred_idx = _nearest_rho_index(pred, action_grid.rhos)
                rmse = action_rmse_matrix[idx, pred_idx]
                rows.append(_make_row("action_condition_plus_semantic_rho", fold, metadata, idx, rmse, tau2_full, n_time, {"semantic_prior_rho": pred}))

        # ── Model 8: Upper Bound (fitted per trial) ────────────────────────
        # tau2 is estimated by fitting each training trial optimally
        train_rmses_ub = np.min(action_rmse_matrix[train_idx], axis=1)
        tau2_ub = _fit_tau_from_rmse(train_rmses_ub)
        for idx in test_idx:
            best_idx = int(np.argmin(action_rmse_matrix[idx]))
            rmse = float(action_rmse_matrix[idx, best_idx])
            rows.append(_make_row("action_trial_fitted_rho", fold, metadata, idx, rmse, tau2_ub, n_time))

    return pd.DataFrame(rows)


def _make_row(
    model: str,
    fold: int,
    metadata: pd.DataFrame,
    idx: int,
    rmse: float,
    tau2: float,
    n_time: int,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "model":     model,
        "fold":      fold,
        "source_row": metadata.loc[idx, "source_row"],
        "subject":   metadata.loc[idx, "subject"],
        "condition": metadata.loc[idx, "condition"],
        "rmse":      rmse,
        "tau2":      tau2,
        "nll":       -gaussian_path_loglik(rmse, tau2, n_time),
        "loglik":    gaussian_path_loglik(rmse, tau2, n_time),
    }
    if "exemplar" in metadata.columns:
        row["exemplar"] = metadata.loc[idx, "exemplar"]
    if extra:
        row.update(extra)
    return row


def _nearest_rho_index(rho: float, rhos: np.ndarray) -> int:
    return int(np.argmin(np.abs(rhos - rho)))
