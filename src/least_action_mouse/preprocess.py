from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class TrajectoryData:
    trajectories: np.ndarray
    metadata: pd.DataFrame
    time: np.ndarray


def parse_numeric_list(value: object) -> np.ndarray:
    """Parse a string like "[1, 2, 3.5]" into a numeric array."""

    return np.array([float(match) for match in NUMBER_RE.findall(str(value))], dtype=float)


def minimum_jerk_path(n_time: int = 51, endpoint: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """Minimum-jerk path from the canonical start to an endpoint."""

    t = np.linspace(0.0, 1.0, n_time)
    s = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
    return np.column_stack((s * endpoint[0], s * endpoint[1]))


def preprocess_kh2017(
    raw: pd.DataFrame,
    n_time: int = 51,
    correct_only: bool = True,
    min_points: int = 5,
    min_horizontal_px: float = 100.0,
    min_vertical_px: float = 100.0,
) -> TrajectoryData:
    """Canonicalize KH2017 trajectories for two-choice response competition."""

    grid = np.linspace(0.0, 1.0, n_time)
    trajectories: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    motor_path = minimum_jerk_path(n_time)

    for row_index, row in raw.iterrows():
        if correct_only and int(row["correct"]) != 1:
            continue

        x = parse_numeric_list(row["xpos_get_response"])
        y = parse_numeric_list(row["ypos_get_response"])
        timestamps = parse_numeric_list(row["timestamps_get_response"])
        n = min(len(x), len(y), len(timestamps))
        if n < min_points:
            continue

        x = x[:n]
        y = y[:n]
        timestamps = timestamps[:n]
        if timestamps[-1] <= timestamps[0]:
            continue

        target_side = 1.0 if row["CategoryCorrect"] == row["CategoryRight"] else -1.0
        dx = x - x[0]
        dy_up = y[0] - y
        horizontal_scale = abs(dx[-1])
        vertical_scale = dy_up[-1]
        if horizontal_scale < min_horizontal_px or vertical_scale < min_vertical_px:
            continue

        canonical_x = target_side * dx / horizontal_scale
        canonical_y = dy_up / vertical_scale
        t_norm = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])

        unique_t, unique_indices = np.unique(t_norm, return_index=True)
        if len(unique_t) < min_points:
            continue
        unique_indices = np.sort(unique_indices)
        unique_t = t_norm[unique_indices]
        canonical_x = canonical_x[unique_indices]
        canonical_y = canonical_y[unique_indices]

        q = np.column_stack(
            (
                np.interp(grid, unique_t, canonical_x),
                np.interp(grid, unique_t, canonical_y),
            )
        )

        deviation = motor_path[:, 0] - q[:, 0]
        trajectories.append(q)
        records.append(
            {
                "source_row": int(row_index),
                "subject": int(row["subject_nr"]),
                "trial": int(row["count_trial"]),
                "condition": str(row["Condition"]),
                "atypical": int(str(row["Condition"]) == "Atypical"),
                "exemplar": str(row["Exemplar"]),
                "category_left": str(row["CategoryLeft"]),
                "category_right": str(row["CategoryRight"]),
                "category_correct": str(row["CategoryCorrect"]),
                "response": str(row["response"]),
                "response_time_ms": float(row["response_time"]),
                "rt_s": float(row["response_time"]) / 1000.0,
                "target_side": int(target_side),
                "raw_points": int(n),
                "horizontal_scale_px": float(horizontal_scale),
                "vertical_scale_px": float(vertical_scale),
                "auc": float(np.mean(deviation)),
                "max_deviation": float(np.max(deviation)),
                "motor_rmse": float(_path_rmse(q, motor_path)),
            }
        )

    if not trajectories:
        raise ValueError("No usable trajectories were found after preprocessing.")

    return TrajectoryData(
        trajectories=np.stack(trajectories),
        metadata=pd.DataFrame.from_records(records),
        time=grid,
    )


def _path_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((observed - predicted) ** 2, axis=1))))

