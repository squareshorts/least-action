from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from least_action_mouse.action_model import ActionParams, precompute_action_grid
from least_action_mouse.analysis import (
    fit_trial_conflict,
    item_level_analysis,
    summarize_model_trials,
    trajectory_rmse_matrix,
)
from least_action_mouse.config import config_value, load_model_config
from least_action_mouse.data import ensure_kh2017_csv
from least_action_mouse.preprocess import preprocess_kh2017
from least_action_mouse.semantic_prior import semantic_prior_rho
from least_action_mouse.stochastic_action import stochastic_likelihood_comparison


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_model_config(args.config)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_k = int(config_value(config, "model.n_time", 51))
    rho_min = float(config_value(config, "model.rho_grid.min", 0.0))
    rho_max = float(config_value(config, "model.rho_grid.max", 2.0))
    rho_step = float(config_value(config, "model.rho_grid.step", 0.05))
    rhos = np.round(np.arange(rho_min, rho_max + 0.5 * rho_step, rho_step), 10)

    base_alpha = float(config_value(config, "model.primary_action.alpha", 1.0))
    base_beta = float(config_value(config, "model.primary_action.beta", 0.003))
    base_sigma = float(config_value(config, "model.primary_action.sigma", 0.9))
    maxiter = int(config_value(config, "model.optimizer.maxiter", 500))

    scenarios = [
        ("baseline", base_k, base_alpha, base_beta, base_sigma),
        ("alpha_0.5x", base_k, 0.5 * base_alpha, base_beta, base_sigma),
        ("alpha_2x", base_k, 2.0 * base_alpha, base_beta, base_sigma),
        ("beta_0.5x", base_k, base_alpha, 0.5 * base_beta, base_sigma),
        ("beta_2x", base_k, base_alpha, 2.0 * base_beta, base_sigma),
        ("sigma_0.75x", base_k, base_alpha, base_beta, 0.75 * base_sigma),
        ("sigma_1.25x", base_k, base_alpha, base_beta, 1.25 * base_sigma),
        (f"K_{max(31, base_k - 10)}", max(31, base_k - 10), base_alpha, base_beta, base_sigma),
        (f"K_{base_k + 10}", base_k + 10, base_alpha, base_beta, base_sigma),
    ]

    raw = pd.read_csv(ensure_kh2017_csv(args.data_dir))
    semantic_scores = pd.read_csv(args.semantic_scores)
    rows: list[dict[str, object]] = []

    for label, k, alpha, beta, sigma in scenarios:
        data = preprocess_kh2017(raw, n_time=k)
        params = ActionParams(alpha=alpha, beta=beta, sigma=sigma, maxiter=maxiter)
        action_grid = precompute_action_grid(rhos, k, params)
        rmse_matrix = trajectory_rmse_matrix(data.trajectories, action_grid.target_paths)
        trial_fits = fit_trial_conflict(data, rmse_matrix, action_grid)
        item_summary, _ = item_level_analysis(trial_fits, raw, args.semantic_scores)
        semantic_result = semantic_prior_rho(item_summary)
        nll_trials = stochastic_likelihood_comparison(
            data=data,
            action_grid=action_grid,
            semantic_scores=semantic_scores,
            item_summary=item_summary,
            n_splits=19,
            grouping_col="exemplar",
        )
        nll_summary = summarize_model_trials(nll_trials)

        cond_nll = _item_mean_nll(nll_trials, "action_condition_only_rho")
        sem_nll = _item_mean_nll(nll_trials, "action_semantic_margin_only_rho")
        gains = cond_nll.align(sem_nll, join="inner")
        gain = gains[0] - gains[1]

        rows.append(
            {
                "analysis_set": label,
                "K": int(k),
                "alpha": float(alpha),
                "beta": float(beta),
                "sigma_T": float(sigma),
                "sigma_C": float(sigma),
                "spearman_margin_rho": _nested(semantic_result, "margin_vs_rho_spearman", "spearman_rho"),
                "spearman_loocv_predrho_fittedrho": _nested(semantic_result, "loocv_correlation", "spearman_rho"),
                "mean_nll_condition_only": float(cond_nll.mean()),
                "mean_nll_semantic": float(sem_nll.mean()),
                "mean_nll_gain_condition_minus_semantic": float(gain.mean()),
                "n_items_improved": int((gain > 0).sum()),
                "trial_weighted_nll_condition_only": _overall_nll(nll_summary, "action_condition_only_rho"),
                "trial_weighted_nll_semantic": _overall_nll(nll_summary, "action_semantic_margin_only_rho"),
            }
        )

    pd.DataFrame(rows).to_csv(out_path, index=False)
    return 0


def _item_mean_nll(trials: pd.DataFrame, model: str) -> pd.Series:
    return trials.loc[trials["model"] == model].groupby("exemplar")["nll"].mean()


def _overall_nll(summary: pd.DataFrame, model: str) -> float:
    row = summary[(summary["model"] == model) & (summary["condition"] == "All")]
    return float(-row.iloc[0]["mean_loglik"]) if not row.empty else float("nan")


def _nested(data: dict[str, object], *keys: str) -> float:
    value: object = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return float("nan")
        value = value[key]
    return float(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--semantic-scores", default="data/processed/semantic_scores.csv")
    parser.add_argument("--out", default="outputs/sensitivity_action_parameters.csv")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
