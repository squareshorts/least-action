from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def parameter_recovery(
    action_paths: np.ndarray,
    rhos: np.ndarray,
    true_rhos: tuple[float, ...] = (0.1, 0.3, 0.6, 1.0),
    tau_values: tuple[float, ...] = (0.03, 0.06, 0.10),
    n_per_cell: int = 80,
    seed: int = 20260513,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Simulate noisy trajectories and recover rho by grid search."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for tau in tau_values:
        for true_rho in true_rhos:
            true_index = int(np.argmin(np.abs(rhos - true_rho)))
            mean_path = action_paths[true_index]
            for _ in range(n_per_cell):
                noisy = mean_path + rng.normal(0.0, tau, size=mean_path.shape)
                noisy[0] = mean_path[0]
                noisy[-1] = mean_path[-1]
                rmse = np.sqrt(np.mean(np.sum((action_paths - noisy[None, :, :]) ** 2, axis=-1), axis=-1))
                recovered_index = int(np.argmin(rmse))
                rows.append(
                    {
                        "tau": float(tau),
                        "rho_true": float(rhos[true_index]),
                        "rho_recovered": float(rhos[recovered_index]),
                        "abs_error": float(abs(rhos[recovered_index] - rhos[true_index])),
                    }
                )
    results = pd.DataFrame(rows)
    summary = {
        "n_simulated": int(len(results)),
        "overall_correlation": float(results["rho_true"].corr(results["rho_recovered"])),
        "overall_mae": float(results["abs_error"].mean()),
        "by_tau": results.groupby("tau")
        .agg(
            correlation=("rho_recovered", lambda values: float(np.corrcoef(results.loc[values.index, "rho_true"], values)[0, 1])),
            mae=("abs_error", "mean"),
            n=("abs_error", "size"),
        )
        .reset_index()
        .to_dict(orient="records"),
    }
    return results, summary
