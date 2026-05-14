from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def paired_subject_inference(
    df: pd.DataFrame,
    value_col: str,
    n_boot: int = 5000,
    n_perm: int = 5000,
    seed: int = 20260513,
) -> dict[str, Any]:
    """Subject-level paired test, cluster bootstrap CI, and sign-flip permutation."""

    table = df.pivot_table(
        index="subject",
        columns="condition",
        values=value_col,
        aggfunc="mean",
    ).dropna()
    diffs = (table["Atypical"] - table["Typical"]).to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = np.array(
        [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n_boot)]
    )
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diffs)))
    perm = (signs * diffs).mean(axis=1)
    ttest = stats.ttest_1samp(diffs, 0.0)
    wilcoxon = stats.wilcoxon(diffs)
    observed = diffs.mean()
    return {
        "n_subjects": int(len(diffs)),
        "typical_mean": float(table["Typical"].mean()),
        "atypical_mean": float(table["Atypical"].mean()),
        "mean_diff_atypical_minus_typical": float(observed),
        "paired_t": float(ttest.statistic),
        "paired_t_p": bounded_pvalue(ttest.pvalue, n_perm),
        "wilcoxon_stat": float(wilcoxon.statistic),
        "wilcoxon_p": bounded_pvalue(wilcoxon.pvalue, n_perm),
        "bootstrap_ci_95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "signflip_p": bounded_pvalue((np.sum(np.abs(perm) >= abs(observed)) + 1) / (n_perm + 1), n_perm),
    }


def residualized_slope_inference(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    n_boot: int = 1000,
    n_perm: int = 1000,
    seed: int = 20260513,
) -> dict[str, Any]:
    """Robust check for x predicting y beyond condition and subject effects."""

    work = df[["subject", "atypical", y_col, x_col]].dropna().copy()
    observed = fixed_effect_slope(work, y_col, x_col)
    subjects = work["subject"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)

    boot = []
    for sample_idx in range(n_boot):
        sampled_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        pieces = []
        for boot_subject, subject in enumerate(sampled_subjects):
            piece = work.loc[work["subject"] == subject].copy()
            piece["subject"] = boot_subject
            pieces.append(piece)
        boot_df = pd.concat(pieces, ignore_index=True)
        boot.append(fixed_effect_slope(boot_df, y_col, x_col))
    boot = np.asarray(boot)

    perm = []
    for _ in range(n_perm):
        perm_df = work.copy()
        perm_df[x_col] = perm_df.groupby("subject")[x_col].transform(
            lambda values: rng.permutation(values.to_numpy())
        )
        perm.append(fixed_effect_slope(perm_df, y_col, x_col))
    perm = np.asarray(perm)

    return {
        "y": y_col,
        "x": x_col,
        "covariates": ["atypical", "subject_fixed_effects"],
        "slope": float(observed),
        "cluster_bootstrap_ci_95": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "cluster_permutation_p": bounded_pvalue(
            (np.sum(np.abs(perm) >= abs(observed)) + 1) / (n_perm + 1),
            n_perm,
        ),
    }


def fixed_effect_slope(df: pd.DataFrame, y_col: str, x_col: str) -> float:
    y = df[y_col].to_numpy(float)
    x = df[x_col].to_numpy(float)
    nuisance = design_matrix(df)
    y_resid = residualize(y, nuisance)
    x_resid = residualize(x, nuisance)
    denom = float(np.dot(x_resid, x_resid))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(x_resid, y_resid) / denom)


def design_matrix(df: pd.DataFrame) -> np.ndarray:
    subjects = pd.get_dummies(df["subject"].astype(str), drop_first=True, dtype=float)
    return np.column_stack(
        (
            np.ones(len(df)),
            df["atypical"].to_numpy(float),
            subjects.to_numpy(float),
        )
    )


def residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coef


def bounded_pvalue(pvalue: float, n_resamples: int) -> float:
    if np.isnan(pvalue):
        return float("nan")
    return float(max(pvalue, 1.0 / (n_resamples + 1)))

