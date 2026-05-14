"""semantic_prior.py
===================
Semantic-prior ρ model for the least-action mouse-tracking project.

Given item-level summary statistics (``item_level_action_summary.csv``) that
have been augmented with external semantic similarity scores
(``data/processed/semantic_scores.csv``), this module fits three nested OLS
models:

  ρ̂ ~ condition                            [rho_condition]
  ρ̂ ~ semantic_margin                      [rho_semantic]
  ρ̂ ~ condition + semantic_margin          [rho_full]

and evaluates:
  - Model AICs (model comparison).
  - Item-level ``rho_predicted_semantic`` from the semantic-only model.
  - Leave-one-out cross-validated (LOOCV) Spearman correlation of
    ``rho_predicted_semantic`` vs observed ``rho_hat`` across the 19 items.
  - Downstream regressions: AUC, RT, error_rate ~ rho_predicted_semantic.

Public API
----------
``semantic_prior_rho(item_summary) -> dict``
    Takes the merged item-summary DataFrame and returns a dict with all
    model-comparison results and item-level predictions.

``semantic_prior_plot(item_summary, results_dir)``
    Saves two figures:
      1. ``semantic_prior_rho.png``  — scatter: rho_hat ~ semantic_margin
      2. ``semantic_predicted_vs_fitted.png`` — LOOCV predicted vs fitted rho
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# Core function
# ─────────────────────────────────────────────────────────────────────────────

def semantic_prior_rho(item_summary: pd.DataFrame) -> dict[str, Any]:
    """Fit semantic-prior ρ models and return full comparison results.

    Parameters
    ----------
    item_summary:
        Item-level summary DataFrame, expected columns:
        ``exemplar``, ``condition``, ``rho_hat``, ``semantic_margin``,
        ``semantic_similarity_target``, ``semantic_similarity_competitor``,
        ``auc``, ``rt_s`` or ``raw_rt_s``, ``error_rate``.

    Returns
    -------
    dict with keys:
        ``models``              – AIC comparison for the three nested models.
        ``rho_predicted_semantic`` – item-level predictions (LOOCV).
        ``loocv_correlation``   – Spearman r, LOOCV predicted vs observed.
        ``downstream``          – dict of downstream regressions.
        ``item_predictions``    – list of per-item prediction records.
    """
    df = item_summary.copy()

    # Require semantic_margin
    if "semantic_margin" not in df.columns or df["semantic_margin"].isna().all():
        return {
            "error": "semantic_margin column missing or all-NaN; run compute_semantic_scores.py first.",
        }

    # Drop items without semantic scores
    df = df.dropna(subset=["semantic_margin", "rho_hat"]).copy()
    n = len(df)
    if n < 4:
        return {"error": f"Only {n} items with valid semantic scores; need ≥ 4."}

    # Encode condition as 0/1 (Atypical = 1 for consistency with trial-level)
    df["atypical"] = (df["condition"] == "Atypical").astype(float)

    # ── Three nested models ─────────────────────────────────────────────────
    models: dict[str, dict[str, Any]] = {}
    for formula, label in [
        ("rho_hat ~ atypical",                        "rho_condition"),
        ("rho_hat ~ semantic_margin",                 "rho_semantic"),
        ("rho_hat ~ atypical + semantic_margin",      "rho_full"),
    ]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.ols(formula, df).fit()
        models[label] = {
            "formula":       formula,
            "aic":           float(fit.aic),
            "r2":            float(fit.rsquared),
            "r2_adj":        float(fit.rsquared_adj),
            "n_items":       int(n),
            "params":        {k: float(v) for k, v in fit.params.items()},
            "pvalues":       {k: float(v) for k, v in fit.pvalues.items()},
        }

    # ── LOOCV: semantic-only model ──────────────────────────────────────────
    loocv_records: list[dict[str, Any]] = []
    loocv_predicted: list[float] = []
    loocv_observed:  list[float] = []

    for i in range(n):
        train = df.iloc[np.delete(np.arange(n), i)]
        test  = df.iloc[[i]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_loocv = smf.ols("rho_hat ~ semantic_margin", train).fit()
        pred = float(fit_loocv.predict(test).values[0])
        obs  = float(test["rho_hat"].values[0])
        loocv_predicted.append(pred)
        loocv_observed.append(obs)
        loocv_records.append({
            "exemplar":               str(test["exemplar"].values[0]),
            "condition":              str(test["condition"].values[0]),
            "rho_hat":                obs,
            "rho_predicted_semantic": pred,
            "semantic_margin":        float(test["semantic_margin"].values[0]),
        })

    loocv_corr = stats.spearmanr(loocv_observed, loocv_predicted)
    df["rho_predicted_semantic"] = loocv_predicted

    # ── Downstream regressions (item-level, n≈19) ──────────────────────────
    rt_col = "raw_rt_s" if "raw_rt_s" in df.columns else "rt_s"
    downstream: dict[str, Any] = {}

    for outcome, formula in [
        ("auc",        f"auc ~ rho_predicted_semantic"),
        ("rt",         f"{rt_col} ~ rho_predicted_semantic"),
        ("error_rate", f"error_rate ~ rho_predicted_semantic"),
    ]:
        if formula.split(" ~ ")[0] not in df.columns:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_ds = smf.ols(formula, df).fit()
        # Check sign constraint: rho up -> AUC/RT/error up implies positive coefficient
        expected_sign_satisfied = bool(fit_ds.params.get("rho_predicted_semantic", 0) > 0)
        
        downstream[outcome] = {
            "formula":   formula,
            "n_items":   int(n),
            "r2":        float(fit_ds.rsquared),
            "aic":       float(fit_ds.aic),
            "params":    {k: float(v) for k, v in fit_ds.params.items()},
            "pvalues":   {k: float(v) for k, v in fit_ds.pvalues.items()},
            "expected_sign_satisfied": expected_sign_satisfied,
        }

    # Item-level Spearman between semantic_margin and rho_hat
    spearman_margin_rho = stats.spearmanr(df["semantic_margin"], df["rho_hat"])

    return {
        "n_items":           n,
        "models":            models,
        "loocv_correlation": {
            "spearman_rho": float(loocv_corr.statistic),
            "p":            float(loocv_corr.pvalue),
            "n_items":      n,
            "direction":    "positive means high semantic conflict → high rho_hat (correct direction)",
        },
        "margin_vs_rho_spearman": {
            "spearman_rho": float(spearman_margin_rho.statistic),
            "p":            float(spearman_margin_rho.pvalue),
            "note": "Negative = higher target margin → lower rho (more typical → less conflict)",
            "expected_sign_satisfied": bool(float(spearman_margin_rho.statistic) < 0),
        },
        "downstream":       downstream,
        "item_predictions": loocv_records,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def semantic_prior_plot(item_summary: pd.DataFrame, results_dir: Path) -> None:
    """Save two diagnostic plots for the semantic-prior model."""
    df = item_summary.dropna(subset=["semantic_margin", "rho_hat"]).copy()
    if len(df) < 4:
        return

    colors = {"Atypical": "#c44536", "Typical": "#2a9d8f"}

    # ── Plot 1: rho_hat ~ semantic_margin ────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    for cond, cdf in df.groupby("condition"):
        ax.scatter(
            cdf["semantic_margin"], cdf["rho_hat"],
            label=cond, color=colors.get(str(cond), "grey"),
            s=70, alpha=0.85, zorder=3,
        )
        for _, row in cdf.iterrows():
            ax.annotate(
                row["exemplar"],
                (row["semantic_margin"], row["rho_hat"]),
                fontsize=7, xytext=(4, 2), textcoords="offset points",
                color=colors.get(str(cond), "grey"), alpha=0.75,
            )
    # Regression line
    x = df["semantic_margin"].to_numpy()
    y = df["rho_hat"].to_numpy()
    if len(np.unique(x)) > 1:
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, m * x_line + b, color="#555555", lw=1.5, ls="--", zorder=2)

    ax.axvline(0, color="#aaaaaa", lw=0.8, ls=":")
    ax.set_xlabel("Semantic margin (sim_target - sim_competitor)")
    ax.set_ylabel("Mean fitted ρ̂ (item level)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    dest1 = results_dir / "semantic_prior_rho.png"
    fig.savefig(dest1, dpi=180)
    plt.close(fig)
    print(f"  -> {dest1}")

    # ── Plot 2: LOOCV predicted vs observed rho ──────────────────────────
    if "rho_predicted_semantic" not in df.columns:
        # Run LOOCV to add column
        result = semantic_prior_rho(df)
        if "item_predictions" not in result:
            return
        preds = pd.DataFrame(result["item_predictions"])
        df = df.merge(preds[["exemplar", "rho_predicted_semantic"]], on="exemplar", how="left")

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    for cond, cdf in df.groupby("condition"):
        if "rho_predicted_semantic" not in cdf.columns:
            continue
        ax.scatter(
            cdf["rho_predicted_semantic"], cdf["rho_hat"],
            label=cond, color=colors.get(str(cond), "grey"),
            s=70, alpha=0.85, zorder=3,
        )
    lo = min(df["rho_predicted_semantic"].min(), df["rho_hat"].min()) - 0.02
    hi = max(df["rho_predicted_semantic"].max(), df["rho_hat"].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], color="#aaaaaa", lw=1.0, ls="--", zorder=1)  # identity
    ax.set_xlabel("ρ̂ predicted (LOOCV, semantic prior)")
    ax.set_ylabel("ρ̂ observed (trial-level fit)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    dest2 = results_dir / "semantic_predicted_vs_fitted.png"
    fig.savefig(dest2, dpi=180)
    plt.close(fig)
    print(f"  -> {dest2}")
