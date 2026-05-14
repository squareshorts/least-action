import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

def plot_decisive_four_panel(
    item_summary: pd.DataFrame,
    semantic_prior_results: dict,
    stochastic_nll_summary: pd.DataFrame,
    destination: Path,
):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    colors = {"Atypical": "#c44536", "Typical": "#2a9d8f"}
    
    df = item_summary.copy()
    if "rho_predicted_semantic" not in df.columns:
        preds = pd.DataFrame(semantic_prior_results["item_predictions"])
        df = df.merge(preds[["exemplar", "rho_predicted_semantic"]], on="exemplar", how="left")
    df = df.dropna(subset=["semantic_margin", "rho_hat"])

    # A: Semantic margin vs fitted rho
    ax = axs[0, 0]
    for cond, cdf in df.groupby("condition"):
        ax.scatter(cdf["semantic_margin"], cdf["rho_hat"], label=cond, color=colors.get(str(cond), "grey"), alpha=0.8)
        for _, row in cdf.iterrows():
            ax.annotate(row["exemplar"], (row["semantic_margin"], row["rho_hat"]), fontsize=6, alpha=0.6)
    x, y = df["semantic_margin"].to_numpy(), df["rho_hat"].to_numpy()
    if len(np.unique(x)) > 1:
        m, b = np.polyfit(x, y, 1)
        xl = np.linspace(x.min(), x.max(), 10)
        ax.plot(xl, m*xl + b, 'k--', alpha=0.5)
    ax.axvline(0, color="grey", ls=":", alpha=0.5)
    ax.text(0.02, 0.96, "A", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    ax.set_xlabel("Semantic Margin (Typicality Difference)")
    ax.set_ylabel("Trial-Fitted Competitor Attraction ρ")

    # B: LOOCV Predicted rho vs Fitted rho
    ax = axs[0, 1]
    for cond, cdf in df.groupby("condition"):
        ax.scatter(cdf["rho_predicted_semantic"], cdf["rho_hat"], color=colors.get(str(cond), "grey"), alpha=0.8)
    lo = min(df["rho_predicted_semantic"].min(), df["rho_hat"].min()) - 0.05
    hi = max(df["rho_predicted_semantic"].max(), df["rho_hat"].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5)
    ax.text(0.02, 0.96, "B", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    ax.set_xlabel("LOOCV Predicted ρ (from Semantic Prior)")
    ax.set_ylabel("Trial-Fitted Competitor Attraction ρ")

    # C: Stochastic NLL Comparison
    ax = axs[1, 0]
    overall = stochastic_nll_summary[stochastic_nll_summary["condition"] == "All"].copy()
    overall["mean_nll"] = -overall["mean_loglik"]
    
    model_order = [
        "baseline_condition_mean",
        "baseline_minimum_jerk",
        "bezier_condition",
        "spline_condition",
        "action_condition_only_rho",
        "action_semantic_margin_only_rho",
        "action_condition_plus_semantic_rho",
        "action_trial_fitted_rho"
    ]
    # Filter to what we actually have
    overall = overall[overall["model"].isin(model_order)]
    overall["model"] = pd.Categorical(overall["model"], categories=model_order, ordered=True)
    overall = overall.sort_values("model").dropna()
    
    labels = {
        "baseline_condition_mean": "Condition Mean",
        "baseline_minimum_jerk": "Minimum Jerk",
        "bezier_condition": "Bezier Baseline",
        "spline_condition": "Spline Baseline",
        "action_condition_only_rho": "Action (Condition)",
        "action_semantic_margin_only_rho": "Action (Semantic Prior)",
        "action_condition_plus_semantic_rho": "Action (Cond+Semantic)",
        "action_trial_fitted_rho": "Trial Fitted (Upper Bound)"
    }
    y_pos = np.arange(len(overall))
    
    ax.barh(y_pos, overall["mean_nll"], align='center', color="#2b6777", alpha=0.8)
    ax.set_yticks(y_pos, labels=[labels.get(m, m) for m in overall["model"]])
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("Held-out NLL (lower is better)")
    ax.text(0.02, 0.96, "C", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    
    # Adjust xlim to show differences better
    min_nll = overall["mean_nll"].min()
    max_nll = overall["mean_nll"].max()
    padding = (max_nll - min_nll) * 0.1
    if padding == 0: padding = 1
    ax.set_xlim(min_nll - padding*2, max_nll + padding)

    # D: Predicted Rho vs Behavioral Metric
    ax = axs[1, 1]
    ax.scatter(df["rho_predicted_semantic"], df["error_rate"], color="#d96c06", alpha=0.8, label="Error Rate")
    ax2 = ax.twinx()
    ax2.scatter(df["rho_predicted_semantic"], df["rt_s"] if "rt_s" in df.columns else df["raw_rt_s"], color="#333333", marker="^", alpha=0.6, label="Response Time (s)")
    
    x, y1 = df["rho_predicted_semantic"].to_numpy(), df["error_rate"].to_numpy()
    m1, b1 = np.polyfit(x, y1, 1)
    xl = np.linspace(x.min(), x.max(), 10)
    ax.plot(xl, m1*xl + b1, color="#d96c06", ls="--", alpha=0.5)
    
    y2 = df["rt_s"].to_numpy() if "rt_s" in df.columns else df["raw_rt_s"].to_numpy()
    m2, b2 = np.polyfit(x, y2, 1)
    ax2.plot(xl, m2*xl + b2, color="#333333", ls="--", alpha=0.5)
    
    ax.text(0.02, 0.96, "D", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    ax.set_xlabel("LOOCV Predicted ρ (from Semantic Prior)")
    ax.set_ylabel("Error Rate", color="#d96c06")
    ax2.set_ylabel("Response Time (s)", color="#333333")

    fig.tight_layout()
    fig.savefig(destination, dpi=300)
    plt.close(fig)
