from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .preprocess import TrajectoryData, minimum_jerk_path


def spline_attraction_paths(
    strengths: np.ndarray,
    n_time: int,
    peak_power: float = 2.0,
) -> np.ndarray:
    """Non-Lagrangian one-parameter attraction baseline.

    The path is a target-directed minimum-jerk trajectory plus a smooth,
    endpoint-preserving leftward bump toward the competitor.
    """

    t = np.linspace(0.0, 1.0, n_time)
    motor = minimum_jerk_path(n_time)
    bump = t * (1.0 - t) ** peak_power
    bump = bump / bump.max()
    paths = []
    for strength in strengths:
        path = motor.copy()
        path[:, 0] -= float(strength) * bump
        paths.append(path)
    return np.stack(paths)


def bezier_attraction_paths(strengths: np.ndarray, n_time: int) -> np.ndarray:
    """Cubic Bezier competitor-attraction baseline.

    This is intentionally non-variational. It gives the trajectory model a
    competitor-directed control point without an action functional.
    """

    t = np.linspace(0.0, 1.0, n_time)[:, None]
    p0 = np.array([0.0, 0.0])
    p3 = np.array([1.0, 1.0])
    paths = []
    for strength in strengths:
        p1 = np.array([-float(strength), 0.12 + 0.18 * float(strength)])
        p2 = np.array([0.45 - 0.25 * float(strength), 0.86])
        path = (
            (1.0 - t) ** 3 * p0
            + 3.0 * (1.0 - t) ** 2 * t * p1
            + 3.0 * (1.0 - t) * t**2 * p2
            + t**3 * p3
        )
        paths.append(path)
    return np.stack(paths)


def condition_mean_cv(
    data: TrajectoryData,
    n_splits: int = 5,
    model_name: str = "condition_mean_trajectory",
) -> pd.DataFrame:
    """Predict held-out trials with the training-set mean trajectory by condition."""

    metadata = data.metadata.reset_index(drop=True)
    splitter = GroupKFold(n_splits=min(n_splits, metadata["subject"].nunique()))
    rows: list[dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(data.trajectories, groups=metadata["subject"]),
        start=1,
    ):
        means = {
            condition: data.trajectories[condition_idx].mean(axis=0)
            for condition in sorted(metadata.loc[train_idx, "condition"].unique())
            for condition_idx in [
                train_idx[metadata.loc[train_idx, "condition"].to_numpy() == condition]
            ]
        }
        for idx in test_idx:
            condition = metadata.loc[idx, "condition"]
            train_condition_idx = train_idx[metadata.loc[train_idx, "condition"].to_numpy() == condition]
            train_mse = np.mean(
                [
                    path_rmse(data.trajectories[j], means[condition]) ** 2
                    for j in train_condition_idx
                ]
            )
            sigma2 = max(train_mse / 2.0, 1e-8)
            rmse = path_rmse(data.trajectories[idx], means[condition])
            rows.append(
                {
                    "fold": fold,
                    "source_row": metadata.loc[idx, "source_row"],
                    "subject": metadata.loc[idx, "subject"],
                    "condition": condition,
                    "model": model_name,
                    "rmse": rmse,
                    "sigma2": sigma2,
                    "loglik": gaussian_path_loglik(rmse, sigma2, data.trajectories.shape[1]),
                }
            )
    return pd.DataFrame(rows)


def cross_validated_grid_model(
    data: TrajectoryData,
    rmse_matrix: np.ndarray,
    parameters: np.ndarray,
    model_name: str,
    selection: str,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a precomputed path grid using shared or condition-specific tuning."""

    if selection not in {"shared", "condition"}:
        raise ValueError("selection must be 'shared' or 'condition'.")

    metadata = data.metadata.reset_index(drop=True)
    splitter = GroupKFold(n_splits=min(n_splits, metadata["subject"].nunique()))
    rows: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(data.trajectories, groups=metadata["subject"]),
        start=1,
    ):
        if selection == "shared":
            mean_rmse = rmse_matrix[train_idx].mean(axis=0)
            best = int(np.argmin(mean_rmse))
            best_by_condition = {
                condition: best for condition in sorted(metadata.loc[train_idx, "condition"].unique())
            }
            selected.append(
                {
                    "fold": fold,
                    "selection": selection,
                    "condition": "All",
                    "parameter": float(parameters[best]),
                    "train_rmse": float(mean_rmse[best]),
                    "train_mse": float(np.mean(rmse_matrix[train_idx, best] ** 2)),
                }
            )
        else:
            best_by_condition = {}
            for condition in sorted(metadata.loc[train_idx, "condition"].unique()):
                condition_idx = train_idx[metadata.loc[train_idx, "condition"].to_numpy() == condition]
                mean_rmse = rmse_matrix[condition_idx].mean(axis=0)
                best = int(np.argmin(mean_rmse))
                best_by_condition[condition] = best
                selected.append(
                    {
                        "fold": fold,
                        "selection": selection,
                        "condition": condition,
                        "parameter": float(parameters[best]),
                        "train_rmse": float(mean_rmse[best]),
                        "train_mse": float(np.mean(rmse_matrix[condition_idx, best] ** 2)),
                    }
                )

        for idx in test_idx:
            condition = metadata.loc[idx, "condition"]
            best = best_by_condition[condition]
            if selection == "shared":
                train_mse = float(np.mean(rmse_matrix[train_idx, best] ** 2))
            else:
                condition_idx = train_idx[metadata.loc[train_idx, "condition"].to_numpy() == condition]
                train_mse = float(np.mean(rmse_matrix[condition_idx, best] ** 2))
            sigma2 = max(train_mse / 2.0, 1e-8)
            rmse = float(rmse_matrix[idx, best])
            rows.append(
                {
                    "fold": fold,
                    "source_row": metadata.loc[idx, "source_row"],
                    "subject": metadata.loc[idx, "subject"],
                    "condition": condition,
                    "model": model_name,
                    "selection": selection,
                    "parameter": float(parameters[best]),
                    "rmse": rmse,
                    "sigma2": sigma2,
                    "loglik": gaussian_path_loglik(rmse, sigma2, data.trajectories.shape[1]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(selected)


def summarize_rmse(rows: pd.DataFrame) -> pd.DataFrame:
    by_condition = (
        rows.groupby(["model", "condition"], as_index=False)
        .agg(mean_rmse=("rmse", "mean"), sd_rmse=("rmse", "std"), n=("rmse", "size"))
        .sort_values(["condition", "model"])
    )
    overall = (
        rows.groupby(["model"], as_index=False)
        .agg(mean_rmse=("rmse", "mean"), sd_rmse=("rmse", "std"), n=("rmse", "size"))
        .assign(condition="All")
    )
    return pd.concat([by_condition, overall], ignore_index=True)


def path_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((observed - predicted) ** 2, axis=1))))


def gaussian_path_loglik(rmse: float, sigma2: float, n_time: int) -> float:
    """Gaussian path log likelihood around a predicted trajectory."""

    n_observations = 2 * n_time
    sse = n_time * rmse**2
    return float(-0.5 * n_observations * np.log(2.0 * np.pi * sigma2) - sse / (2.0 * sigma2))
