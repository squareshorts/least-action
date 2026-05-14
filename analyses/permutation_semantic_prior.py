from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from least_action_mouse.action_model import ActionParams, precompute_action_grid
from least_action_mouse.analysis import fit_condition_rhos, trajectory_rmse_matrix
from least_action_mouse.baselines import gaussian_path_loglik
from least_action_mouse.config import config_value, load_model_config
from least_action_mouse.data import ensure_kh2017_csv
from least_action_mouse.preprocess import preprocess_kh2017


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_csv = Path(args.out)
    out_json = Path(args.summary_out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    config = load_model_config(args.config)
    n_time = int(config_value(config, "model.n_time", 51))
    rho_min = float(config_value(config, "model.rho_grid.min", 0.0))
    rho_max = float(config_value(config, "model.rho_grid.max", 2.0))
    rho_step = float(config_value(config, "model.rho_grid.step", 0.05))
    maxiter = int(config_value(config, "model.optimizer.maxiter", 500))
    rhos = np.round(np.arange(rho_min, rho_max + 0.5 * rho_step, rho_step), 10)

    raw = pd.read_csv(ensure_kh2017_csv(args.data_dir))
    data = preprocess_kh2017(raw, n_time=n_time)
    metadata = data.metadata.reset_index(drop=True).copy()
    semantic_scores = pd.read_csv(args.semantic_scores)
    item_margins = semantic_scores.set_index("exemplar")["semantic_margin"].reindex(sorted(metadata["exemplar"].unique()))
    if item_margins.isna().any():
        missing = item_margins[item_margins.isna()].index.tolist()
        raise ValueError(f"Missing semantic margins for items: {missing}")

    action_grid = precompute_action_grid(rhos, n_time, ActionParams(maxiter=maxiter))
    rmse_matrix = trajectory_rmse_matrix(data.trajectories, action_grid.target_paths)
    best_rhos_by_trial = action_grid.rhos[np.argmin(rmse_matrix, axis=1)]
    items = item_margins.index.to_numpy()
    item_to_index = {item: np.flatnonzero(metadata["exemplar"].to_numpy() == item) for item in items}
    folds = [
        (
            item,
            np.flatnonzero(metadata["exemplar"].to_numpy() != item),
            item_to_index[item],
        )
        for item in items
    ]

    condition_nll = _condition_only_item_nll(metadata, rmse_matrix, action_grid.rhos, folds, n_time)
    observed = _semantic_gain_for_margins(
        metadata=metadata,
        rmse_matrix=rmse_matrix,
        rhos=action_grid.rhos,
        best_rhos_by_trial=best_rhos_by_trial,
        folds=folds,
        item_margins=item_margins,
        condition_nll=condition_nll,
        n_time=n_time,
    )

    rng = np.random.default_rng(args.seed)
    rows = [{"permutation": 0, "is_observed": True, **observed}]
    null_gains = []
    margin_values = item_margins.to_numpy()
    for permutation in range(1, args.n_permutations + 1):
        shuffled = pd.Series(rng.permutation(margin_values), index=item_margins.index)
        result = _semantic_gain_for_margins(
            metadata=metadata,
            rmse_matrix=rmse_matrix,
            rhos=action_grid.rhos,
            best_rhos_by_trial=best_rhos_by_trial,
            folds=folds,
            item_margins=shuffled,
            condition_nll=condition_nll,
            n_time=n_time,
        )
        null_gains.append(result["mean_nll_gain_condition_minus_semantic"])
        rows.append({"permutation": permutation, "is_observed": False, **result})

    null = np.asarray(null_gains, dtype=float)
    observed_gain = float(observed["mean_nll_gain_condition_minus_semantic"])
    summary = {
        "n_permutations": int(args.n_permutations),
        "observed_nll_gain": observed_gain,
        "null_mean_gain": float(null.mean()),
        "null_sd_gain": float(null.std(ddof=1)),
        "permutation_p": float((1 + np.sum(null >= observed_gain)) / (len(null) + 1)),
        "observed_percentile": float(100.0 * np.mean(null < observed_gain)),
        "n_items_improved_observed": int(observed["n_items_improved"]),
    }

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


def _condition_only_item_nll(
    metadata: pd.DataFrame,
    rmse_matrix: np.ndarray,
    rhos: np.ndarray,
    folds: list[tuple[str, np.ndarray, np.ndarray]],
    n_time: int,
) -> pd.Series:
    rows: list[tuple[str, float]] = []
    for item, train_idx, test_idx in folds:
        best = fit_condition_rhos(train_idx, metadata, rmse_matrix, rhos)
        train_rmses = [
            rmse_matrix[tidx, best[metadata.loc[tidx, "condition"]]["rho_index"]]
            for tidx in train_idx
        ]
        tau2 = max(float(np.mean(np.asarray(train_rmses) ** 2)) / 2.0, 1e-8)
        nll_values = []
        for idx in test_idx:
            condition = metadata.loc[idx, "condition"]
            rho_index = best[condition]["rho_index"]
            nll_values.append(-gaussian_path_loglik(float(rmse_matrix[idx, rho_index]), tau2, n_time))
        rows.append((item, float(np.mean(nll_values))))
    return pd.Series(dict(rows))


def _semantic_gain_for_margins(
    metadata: pd.DataFrame,
    rmse_matrix: np.ndarray,
    rhos: np.ndarray,
    best_rhos_by_trial: np.ndarray,
    folds: list[tuple[str, np.ndarray, np.ndarray]],
    item_margins: pd.Series,
    condition_nll: pd.Series,
    n_time: int,
) -> dict[str, float | int]:
    margin_by_trial = metadata["exemplar"].map(item_margins).to_numpy(dtype=float)
    semantic_item_nll: dict[str, float] = {}

    for item, train_idx, test_idx in folds:
        x_train = margin_by_trial[train_idx]
        y_train = best_rhos_by_trial[train_idx]
        design = np.column_stack([np.ones_like(x_train), x_train])
        intercept, slope = np.linalg.lstsq(design, y_train, rcond=None)[0]

        train_pred = intercept + slope * x_train
        train_rho_idx = _nearest_rho_indices(train_pred, rhos)
        train_rmse = rmse_matrix[train_idx, train_rho_idx]
        tau2 = max(float(np.mean(train_rmse**2)) / 2.0, 1e-8)

        test_pred = intercept + slope * margin_by_trial[test_idx]
        test_rho_idx = _nearest_rho_indices(test_pred, rhos)
        nll_values = [
            -gaussian_path_loglik(float(rmse_matrix[idx, rho_idx]), tau2, n_time)
            for idx, rho_idx in zip(test_idx, test_rho_idx)
        ]
        semantic_item_nll[item] = float(np.mean(nll_values))

    semantic_nll = pd.Series(semantic_item_nll).reindex(condition_nll.index)
    gains = condition_nll - semantic_nll
    return {
        "mean_nll_condition_only": float(condition_nll.mean()),
        "mean_nll_semantic": float(semantic_nll.mean()),
        "mean_nll_gain_condition_minus_semantic": float(gains.mean()),
        "n_items_improved": int((gains > 0).sum()),
    }


def _nearest_rho_indices(values: np.ndarray, rhos: np.ndarray) -> np.ndarray:
    return np.abs(values[:, None] - rhos[None, :]).argmin(axis=1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--semantic-scores", default="data/processed/semantic_scores.csv")
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--out", default="outputs/permutation_semantic_prior.csv")
    parser.add_argument("--summary-out", default="outputs/permutation_semantic_prior_summary.json")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
