from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
TABLES = OUTPUTS / "tables"
ROOT_TABLES = ROOT / "tables"
DEPRECATED_MAIN_TABLES = [
    "table_condition_summary.tex",
    "table_paired.tex",
    "table_semantic_models.tex",
    "table_multisource_semantic_validation.tex",
    "table_stochastic.tex",
    "table_item_nll.tex",
]


def main() -> int:
    OUTPUTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    run([sys.executable, "scripts/compute_semantic_scores.py", "--out", "data/processed/semantic_scores.csv"])
    run(
        [
            sys.executable,
            "scripts/run_analysis.py",
            "--results-dir",
            "outputs",
            "--semantic-scores",
            "data/processed/semantic_scores.csv",
        ]
    )
    run([sys.executable, "analyses/sensitivity_action_parameters.py"])
    run([sys.executable, "analyses/permutation_semantic_prior.py", "--n-permutations", "5000"])
    run([sys.executable, "analyses/mixed_effects_validation.py"])
    run([sys.executable, "analyses/secondary_semantic_predictor.py"])
    run([sys.executable, "analyses/frozen_multisource_semantic_validation.py"])

    summary = read_json(OUTPUTS / "summary.json")
    semantic_scores = write_semantic_scores_19(summary)
    reproducibility_summary = write_reproducibility_summary(summary, semantic_scores)
    copy_figures()
    write_table_files(summary, reproducibility_summary, semantic_scores)
    remove_deprecated_main_tables()
    run([sys.executable, "analyses/presubmission_missing_results.py"])
    run([sys.executable, "analyses/central_chain_validation.py"])
    remove_deprecated_main_tables()
    return 0


def run(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_semantic_scores_19(summary: dict[str, Any]) -> pd.DataFrame:
    item = pd.read_csv(OUTPUTS / "item_level_action_summary.csv")
    predictions = pd.DataFrame(summary["semantic_prior_results"]["item_predictions"])
    df = item.merge(predictions[["exemplar", "rho_predicted_semantic"]], on="exemplar", how="left")
    target_col = "category_correct_x" if "category_correct_x" in df.columns else "category_correct"
    competitor_col = "competitor_category_x" if "competitor_category_x" in df.columns else "competitor_category"
    out = pd.DataFrame(
        {
            "item": df["exemplar"],
            "target_category": df[target_col],
            "competitor_category": df[competitor_col],
            "condition": df["condition"],
            "sim_target": df["semantic_similarity_target"],
            "sim_competitor": df["semantic_similarity_competitor"],
            "semantic_margin": df["semantic_margin"],
            "fitted_rho": df["rho_hat"],
            "loocv_predicted_rho": df["rho_predicted_semantic"],
            "mean_auc": df["auc"],
            "mean_rt": df["raw_rt_s"] if "raw_rt_s" in df.columns else df["rt_s"],
            "error_rate": df["error_rate"],
        }
    ).sort_values("semantic_margin")
    out.to_csv(OUTPUTS / "semantic_scores_19_items.csv", index=False)
    return out


def write_reproducibility_summary(summary: dict[str, Any], semantic_scores: pd.DataFrame) -> dict[str, Any]:
    cv = pd.read_csv(OUTPUTS / "cv_rmse_by_condition.csv")
    nll = pd.read_csv(OUTPUTS / "stochastic_nll_summary.csv")
    trial_means = summary["trial_metric_means"]
    item_wise = summary["item_wise_nll"]
    semantic = summary["semantic_prior_results"]
    recovery = summary["parameter_recovery"]

    out = {
        "n_trials_raw": int(summary["dataset"]["n_raw_rows"]),
        "n_trials_final": int(summary["dataset"]["n_correct_canonicalized_trials"]),
        "n_subjects": int(summary["dataset"]["n_subjects"]),
        "n_items": int(len(semantic_scores)),
        "typical_trials": int(summary["dataset"]["conditions"]["Typical"]),
        "atypical_trials": int(summary["dataset"]["conditions"]["Atypical"]),
        "mean_auc_typical": float(trial_means["Typical"]["auc"]),
        "mean_auc_atypical": float(trial_means["Atypical"]["auc"]),
        "mean_rt_typical": float(trial_means["Typical"]["rt_s"]),
        "mean_rt_atypical": float(trial_means["Atypical"]["rt_s"]),
        "mean_rho_typical": float(trial_means["Typical"]["rho_hat"]),
        "mean_rho_atypical": float(trial_means["Atypical"]["rho_hat"]),
        "subject_cv_rmse_action": _cv_value(cv, "action"),
        "subject_cv_rmse_minimum_jerk": _cv_value(cv, "minimum_jerk"),
        "loo_nll_condition_only": _nll_value(nll, "action_condition_only_rho"),
        "loo_nll_semantic": _nll_value(nll, "action_semantic_margin_only_rho"),
        "loo_nll_condition_semantic": _nll_value(nll, "action_condition_plus_semantic_rho"),
        "itemwise_nll_gain": float(item_wise["mean_delta_nll_per_item"]),
        "n_items_semantic_better": int(item_wise["n_items_positive_gain"]),
        "spearman_margin_rho": float(semantic["margin_vs_rho_spearman"]["spearman_rho"]),
        "spearman_predictedrho_fittedrho": float(semantic["loocv_correlation"]["spearman_rho"]),
        "parameter_recovery_r": float(recovery["overall_correlation"]),
        "parameter_recovery_mae": float(recovery["overall_mae"]),
    }
    (OUTPUTS / "reproducibility_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _cv_value(cv: pd.DataFrame, model: str) -> float:
    row = cv[(cv["model"] == model) & (cv["condition"] == "All")]
    return float(row.iloc[0]["mean_rmse"])


def _nll_value(nll: pd.DataFrame, model: str) -> float:
    row = nll[(nll["model"] == model) & (nll["condition"] == "All")]
    return float(-row.iloc[0]["mean_loglik"])


def copy_figures() -> None:
    figure_names = [
        "trajectory_fit.png",
        "rho_by_condition.png",
        "rho_subject_paired.png",
        "semantic_prior_rho.png",
        "semantic_predicted_vs_fitted.png",
        "decisive_four_panel.png",
        "decisive_four_panel.pdf",
    ]
    for name in figure_names:
        src = OUTPUTS / name
        if src.exists():
            shutil.copy2(src, FIGURES / name)


def write_table_files(summary: dict[str, Any], repro: dict[str, Any], semantic_scores: pd.DataFrame) -> None:
    write_dataset_table(repro)
    write_numerical_table()
    write_condition_effects_table(summary)
    write_rmse_table()
    write_cluster_table(summary)
    write_stochastic_likelihood_table(summary)
    write_semantic_validation_table(summary)
    write_semantic_scores_table(semantic_scores)
    write_sensitivity_table()
    write_permutation_table()
    write_mixed_effects_table()
    write_item_recovery_table(summary)


def remove_deprecated_main_tables() -> None:
    for directory in [TABLES, ROOT_TABLES]:
        for name in DEPRECATED_MAIN_TABLES:
            path = directory / name
            if path.exists():
                path.unlink()


def write_dataset_table(repro: dict[str, Any]) -> None:
    rows = [
        ("Raw rows", repro["n_trials_raw"]),
        ("Correct canonicalized trajectories", repro["n_trials_final"]),
        ("Participants", repro["n_subjects"]),
        ("Animal exemplars", repro["n_items"]),
        ("Typical trials", repro["typical_trials"]),
        ("Atypical trials", repro["atypical_trials"]),
        ("Canonical start coordinate", "$(0,0)$"),
        ("Canonical target coordinate", "$(1,1)$"),
        ("Canonical competitor coordinate", "$(-1,1)$"),
    ]
    table("Dataset and analysis sample.", "tab:dataset", ["Quantity", "Value"], rows, "table_dataset.tex")


def write_numerical_table() -> None:
    rows = [
        ("Number of normalized samples, $K$", "51"),
        ("$\\rho$ grid", "$0,0.05,\\ldots,2.00$"),
        ("Velocity/deformation weight, $\\alpha$", "1.0"),
        ("Acceleration weight, $\\beta$", "0.003"),
        ("Target Gaussian width, $\\sigma_T$", "not used in primary nested model"),
        ("Competitor width, $\\sigma_C$", "0.9"),
        ("Boundary term, $B(q)$", "0 inside box constraints, $+\\infty$ outside"),
        ("Optimizer", "SciPy L-BFGS-B with analytic gradient"),
        ("Initial path", "minimum jerk; then previous $\\rho$ solution"),
        ("Convergence tolerance", "ftol = $10^{-9}$"),
        ("Maximum iterations", "500"),
    ]
    table("Fixed numerical settings for the primary action model.", "tab:numerical-settings", ["Quantity", "Value"], rows, "table_numerical_settings.tex")


def write_condition_table(summary: dict[str, Any]) -> None:
    m = summary["trial_metric_means"]
    rows = [
        ("Trials", "744", "320"),
        ("AUC", fmt(m["Typical"]["auc"]), fmt(m["Atypical"]["auc"])),
        ("Maximum deviation", fmt(m["Typical"]["max_deviation"]), fmt(m["Atypical"]["max_deviation"])),
        ("Response time (s)", fmt(m["Typical"]["rt_s"]), fmt(m["Atypical"]["rt_s"])),
        ("Trial-level $\\rho$", fmt(m["Typical"]["rho_hat"]), fmt(m["Atypical"]["rho_hat"])),
        ("Action gap", fmt(m["Typical"]["action_gap"]), fmt(m["Atypical"]["action_gap"])),
    ]
    table("Condition-level behavioral and action-landscape summaries.", "tab:condition-summary", ["Metric", "Typical", "Atypical"], rows, "table_condition_summary.tex")


def write_condition_effects_table(summary: dict[str, Any]) -> None:
    means = summary["trial_metric_means"]
    paired = summary["robust_inference"]["paired_subject"]
    condition_rows = [
        ("Trials", "744", "320", "--", "--"),
        ("AUC", fmt(means["Typical"]["auc"]), fmt(means["Atypical"]["auc"]), "--", "--"),
        (
            "Maximum deviation",
            fmt(means["Typical"]["max_deviation"]),
            fmt(means["Atypical"]["max_deviation"]),
            "--",
            "--",
        ),
        (
            "Response time (s)",
            fmt(means["Typical"]["rt_s"]),
            fmt(means["Atypical"]["rt_s"]),
            "--",
            "--",
        ),
        (
            "Trial-level $\\rho$",
            fmt(means["Typical"]["rho_hat"]),
            fmt(means["Atypical"]["rho_hat"]),
            "--",
            "--",
        ),
        ("Action gap", fmt(means["Typical"]["action_gap"]), fmt(means["Atypical"]["action_gap"]), "--", "--"),
    ]
    paired_labels = [
        ("rho_hat", "$\\rho$"),
        ("auc", "AUC"),
        ("max_deviation", "Maximum deviation"),
        ("rt_s", "Response time (s)"),
        ("action_gap", "Action gap"),
    ]
    paired_rows = []
    for key, label in paired_labels:
        val = paired[key]
        paired_rows.append(
            (
                label,
                fmt(val["typical_mean"]),
                fmt(val["atypical_mean"]),
                fmt(val["mean_diff_atypical_minus_typical"]),
                f"[{fmt(val['bootstrap_ci_95'][0])}, {fmt(val['bootstrap_ci_95'][1])}]",
            )
        )

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Condition-level behavioral/action summaries and paired subject-level condition effects.}",
        "\\label{tab:condition-effects}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Metric & Typical & Atypical & Difference & 95\\% bootstrap CI \\\\",
        "\\midrule",
        "\\multicolumn{5}{l}{\\textit{Panel A. Condition-level summaries}} \\\\",
    ]
    lines.extend(" & ".join(escape(cell) for cell in row) + " \\\\" for row in condition_rows)
    lines.extend(
        [
            "\\midrule",
            "\\multicolumn{5}{l}{\\textit{Panel B. Paired subject-level inference}} \\\\",
        ]
    )
    lines.extend(" & ".join(escape(cell) for cell in row) + " \\\\" for row in paired_rows)
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (TABLES / "table_condition_effects.tex").write_text("\n".join(lines), encoding="utf-8")


def write_rmse_table() -> None:
    df = pd.read_csv(OUTPUTS / "cv_rmse_by_condition.csv")
    order = [("Atypical", "action"), ("Atypical", "minimum_jerk"), ("Typical", "action"), ("Typical", "minimum_jerk"), ("All", "action"), ("All", "minimum_jerk")]
    rows = []
    for condition, model in order:
        row = df[(df["condition"] == condition) & (df["model"] == model)].iloc[0]
        rows.append((condition, model.replace("_", " "), fmt(row["mean_rmse"]), fmt(row["sd_rmse"]), int(row["n"])))
    table("Subject-wise cross-validated RMSE.", "tab:rmse", ["Condition", "Model", "Mean RMSE", "SD RMSE", "$n$"], rows, "table_rmse.tex")


def write_paired_table(summary: dict[str, Any]) -> None:
    paired = summary["robust_inference"]["paired_subject"]
    labels = [
        ("rho_hat", "$\\rho$"),
        ("auc", "AUC"),
        ("max_deviation", "Maximum deviation"),
        ("rt_s", "Response time (s)"),
        ("action_gap", "Action gap"),
    ]
    rows = []
    for key, label in labels:
        val = paired[key]
        rows.append(
            (
                label,
                fmt(val["typical_mean"]),
                fmt(val["atypical_mean"]),
                fmt(val["mean_diff_atypical_minus_typical"]),
                f"[{fmt(val['bootstrap_ci_95'][0])}, {fmt(val['bootstrap_ci_95'][1])}]",
            )
        )
    table("Robust paired subject-level inference for condition effects.", "tab:paired", ["Metric", "Typical mean", "Atypical mean", "Difference", "95\\% bootstrap CI"], rows, "table_paired.tex")


def write_cluster_table(summary: dict[str, Any]) -> None:
    cluster = summary["robust_inference"]["cluster_regression"]
    rows = [
        ("AUC", "$\\rho_z$", cluster["auc_beyond_condition_subject_from_rho"]),
        ("AUC", "Action gap$_z$", cluster["auc_beyond_condition_subject_from_action_gap"]),
        ("Response time", "Action gap$_z$", cluster["rt_beyond_condition_subject_from_action_gap"]),
        ("Response time", "Inverse $|$action gap$|_z$", summary["counterfactual_tests"]["robust_rt_inverse_gap"]),
    ]
    formatted = [
        (
            outcome,
            predictor,
            fmt(val["slope"]),
            fmtp(val["cluster_permutation_p"]),
            f"[{fmt(val['cluster_bootstrap_ci_95'][0])}, {fmt(val['cluster_bootstrap_ci_95'][1])}]",
        )
        for outcome, predictor, val in rows
    ]
    table("Cluster-resampling tests beyond condition and subject effects.", "tab:cluster-regression", ["Outcome", "Predictor", "Slope", "$p$", "95\\% bootstrap CI"], formatted, "table_cluster_regression.tex")


def write_stochastic_table() -> None:
    df = pd.read_csv(OUTPUTS / "stochastic_nll_summary.csv")
    models = [
        ("baseline_condition_mean", "Condition mean"),
        ("bezier_condition", "Bezier condition"),
        ("spline_condition", "Spline condition"),
        ("action_trial_fitted_rho", "Action: trial-fitted $\\rho$"),
        ("action_semantic_margin_only_rho", "Action: semantic margin $\\rho$"),
        ("action_condition_plus_semantic_rho", "Action: condition + semantic $\\rho$"),
        ("action_condition_only_rho", "Action: condition-only $\\rho$"),
        ("baseline_minimum_jerk", "Minimum jerk"),
    ]
    rows = []
    for model, label in models:
        row = df[(df["condition"] == "All") & (df["model"] == model)].iloc[0]
        rows.append((label, fmt(row["mean_rmse"]), fmt(-row["mean_loglik"]), int(row["n"])))
    table("Held-out stochastic model comparison across all trials.", "tab:stochastic", ["Model", "Mean RMSE", "Mean NLL", "$n$"], rows, "table_stochastic.tex")


def write_stochastic_likelihood_table(summary: dict[str, Any]) -> None:
    df = pd.read_csv(OUTPUTS / "stochastic_nll_summary.csv")
    models = [
        ("baseline_condition_mean", "Condition mean"),
        ("bezier_condition", "Bezier condition"),
        ("spline_condition", "Spline condition"),
        ("action_trial_fitted_rho", "Action: trial-fitted $\\rho$"),
        ("action_semantic_margin_only_rho", "Action: semantic margin $\\rho$"),
        ("action_condition_plus_semantic_rho", "Action: condition + semantic $\\rho$"),
        ("action_condition_only_rho", "Action: condition-only $\\rho$"),
        ("baseline_minimum_jerk", "Minimum jerk"),
    ]
    model_rows = []
    for model, label in models:
        row = df[(df["condition"] == "All") & (df["model"] == model)].iloc[0]
        model_rows.append((label, fmt(row["mean_rmse"]), fmt(-row["mean_loglik"]), int(row["n"])))

    iw = summary["item_wise_nll"]
    ci_low, ci_high, sign_p = item_gain_bootstrap_summary()
    summary_rows = [
        ("Mean NLL: condition-only action", fmt(iw["mean_nll_condition"])),
        ("Mean NLL: semantic-margin action", fmt(iw["mean_nll_semantic"])),
        ("Mean item-wise NLL gain, condition minus semantic", fmt(iw["mean_delta_nll_per_item"])),
        ("Bootstrap 95\\% CI for mean gain", f"[{fmt(ci_low)}, {fmt(ci_high)}]"),
        ("Items with positive gain", f"{iw['n_items_positive_gain']} / {iw['n_items_total']}"),
        ("Exact sign-test $p$", fmtp(sign_p)),
    ]

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Held-out stochastic likelihood summary and item-level semantic-prior gain.}",
        "\\label{tab:stochastic-likelihood}",
        "\\small",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Model & Mean RMSE & Mean NLL & $n$ \\\\",
        "\\midrule",
        "\\multicolumn{4}{l}{\\textit{Panel A. Model-level held-out stochastic NLL}} \\\\",
    ]
    lines.extend(" & ".join(escape(cell) for cell in row) + " \\\\" for row in model_rows)
    lines.extend(
        [
            "\\midrule",
            "\\multicolumn{4}{l}{\\textit{Panel B. Item-level condition-only versus semantic-margin action comparison}} \\\\",
            "Statistic & \\multicolumn{3}{r}{Value} \\\\",
        ]
    )
    lines.extend(f"{escape(stat)} & \\multicolumn{{3}}{{r}}{{{escape(value)}}} \\\\" for stat, value in summary_rows)
    lines.extend(["\\bottomrule", "\\end{tabular}", "}%", "\\end{table}", ""])
    (TABLES / "table_stochastic_likelihood.tex").write_text("\n".join(lines), encoding="utf-8")


def item_gain_bootstrap_summary() -> tuple[float, float, float]:
    stoch_trials = pd.read_csv(OUTPUTS / "stochastic_nll_trials.csv")
    cond_nll = (
        stoch_trials.loc[stoch_trials["model"] == "action_condition_only_rho"]
        .groupby("exemplar")["nll"]
        .mean()
    )
    sem_nll = (
        stoch_trials.loc[stoch_trials["model"] == "action_semantic_margin_only_rho"]
        .groupby("exemplar")["nll"]
        .mean()
    )
    delta = (cond_nll - sem_nll).dropna().to_numpy()
    n_items = len(delta)
    n_pos = int(np.sum(delta > 0))
    rng = np.random.default_rng(20260515)
    boot = np.array([rng.choice(delta, n_items, replace=True).mean() for _ in range(4000)])
    sign_p = two_tailed_binom_p(n_pos, n_items)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), sign_p


def two_tailed_binom_p(k: int, n: int) -> float:
    observed = math.comb(n, k) / (2**n)
    return min(1.0, sum(math.comb(n, i) / (2**n) for i in range(n + 1) if math.comb(n, i) / (2**n) <= observed))


def write_semantic_models_table(summary: dict[str, Any]) -> None:
    models = summary["semantic_prior_results"]["models"]
    rows = [
        ("Condition-only", "Atypical label", models["rho_condition"], "atypical"),
        ("Semantic-margin", "Semantic margin", models["rho_semantic"], "semantic_margin"),
        ("Condition + semantic", "Atypical label + semantic margin", models["rho_full"], "semantic_margin"),
    ]
    formatted = [(name, pred, fmt(val["r2"]), fmt(val["aic"]), fmt(val["params"][term])) for name, pred, val, term in rows]
    table("Item-level semantic-prior models predicting fitted $\\rho$.", "tab:semantic-models", ["Model", "Predictor set", "$R^2$", "AIC", "Key slope"], formatted, "table_semantic_models.tex")


def write_semantic_validation_table(summary: dict[str, Any]) -> None:
    models = summary["semantic_prior_results"]["models"]
    model_specs = [
        ("Condition-only", "Atypical label", models["rho_condition"], "atypical"),
        ("Semantic-margin", "Semantic margin", models["rho_semantic"], "semantic_margin"),
        ("Condition + semantic", "Atypical label + semantic margin", models["rho_full"], "semantic_margin"),
    ]
    model_rows = [
        (name, pred, fmt(val["r2"]), fmt(val["aic"]), fmt(val["params"][term]))
        for name, pred, val, term in model_specs
    ]

    validation_path = OUTPUTS / "semantic_sources" / "multisource_semantic_validation_results.csv"
    validation = pd.read_csv(validation_path)
    validation_rows = []
    for _, row in validation.iterrows():
        source = str(row["source"]).replace("_", "\\_")
        source_type = str(row["source_type"])
        validation_rows.append(
            (
                source,
                source_type,
                str(row["coverage"]),
                fmt(row["dir_spearman_r"]),
                fmtp(row["dir_spearman_p"]),
                fmt(row["loo_spearman_r"]),
                fmtp(row["loo_spearman_p"]),
                fmt(row["rmse"]),
                fmt(row["mae"]),
            )
        )

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Semantic-prior model comparison and multisource semantic validation. LOOCV: leave-one-item-out cross-validation.}",
        "\\label{tab:semantic-validation}",
        "\\small",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Model & Predictor set & $R^2$ & AIC & Key slope \\\\",
        "\\midrule",
        "\\multicolumn{5}{l}{\\textit{Panel A. Primary reported semantic margin}} \\\\",
    ]
    lines.extend(" & ".join(escape(cell) for cell in row) + " \\\\" for row in model_rows)
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\vspace{0.5em}",
            "\\resizebox{\\linewidth}{!}{%",
            "\\begin{tabular}{llccccccc}",
            "\\toprule",
            "Source & Type & Cov. & $r_s$ & $p$ & LOOCV $r_s$ & LOOCV $p$ & RMSE & MAE \\\\",
            "\\midrule",
            "\\multicolumn{9}{l}{\\textit{Panel B. Primary and embedding-based semantic sources}} \\\\",
        ]
    )
    lines.extend(" & ".join(escape(cell) for cell in row) + " \\\\" for row in validation_rows)
    lines.extend(["\\bottomrule", "\\end{tabular}", "}%", "\\end{table}", ""])
    (TABLES / "table_semantic_validation.tex").write_text("\n".join(lines), encoding="utf-8")


def write_semantic_scores_table(semantic_scores: pd.DataFrame) -> None:
    rows = [
        (
            row["item"],
            row["condition"],
            row["target_category"],
            row["competitor_category"],
            fmt(row["sim_target"]),
            fmt(row["sim_competitor"]),
            fmt(row["semantic_margin"]),
            fmt(row["fitted_rho"]),
            fmt(row["loocv_predicted_rho"]),
        )
        for _, row in semantic_scores.iterrows()
    ]
    table("Semantic scores and item-level action estimates.", "tab:semantic-scores", ["Item", "Condition", "Target", "Competitor", "sim target", "sim competitor", "Margin", "Fitted $\\rho$", "LOOCV $\\rho$"], rows, "table_semantic_scores.tex", small=True)


def write_item_nll_table(summary: dict[str, Any]) -> None:
    iw = summary["item_wise_nll"]
    rows = [
        ("Mean NLL: condition-only action", fmt(iw["mean_nll_condition"])),
        ("Mean NLL: condition + semantic action", fmt(iw["mean_nll_full"])),
        ("Mean NLL: semantic-margin action", fmt(iw["mean_nll_semantic"])),
        ("Mean item-wise NLL gain, condition minus semantic", fmt(iw["mean_delta_nll_per_item"])),
        ("Total NLL gain across all trials", fmt(iw["total_delta_nll"])),
        ("Items with positive gain", f"{iw['n_items_positive_gain']} / {iw['n_items_total']}"),
        ("Paired item-wise $t$", fmt(iw["paired_t"])),
        ("Paired item-wise $p$", fmtp(iw["p_t"])),
        ("Wilcoxon $W$", fmt(iw["wilcoxon_w"])),
        ("Wilcoxon $p$", fmtp(iw["p_w"])),
    ]
    table("Leave-one-item-out NLL comparison among action models.", "tab:item-nll", ["Model or statistic", "Value"], rows, "table_item_nll.tex")


def write_sensitivity_table() -> None:
    df = pd.read_csv(OUTPUTS / "sensitivity_action_parameters.csv")
    rows = [
        (
            row["analysis_set"],
            int(row["K"]),
            fmt(row["alpha"]),
            fmt(row["beta"]),
            fmt(row["sigma_C"]),
            fmt(row["spearman_margin_rho"]),
            fmt(row["mean_nll_gain_condition_minus_semantic"]),
            int(row["n_items_improved"]),
        )
        for _, row in df.iterrows()
    ]
    table("Sensitivity of semantic-prior results to fixed action parameters.", "tab:sensitivity", ["Analysis set", "$K$", "$\\alpha$", "$\\beta$", "$\\sigma_C$", "Spearman", "NLL gain", "Improved items"], rows, "table_sensitivity.tex", small=True)


def write_permutation_table() -> None:
    summary = read_json(OUTPUTS / "permutation_semantic_prior_summary.json")
    rows = [
        ("Observed NLL gain", fmt(summary["observed_nll_gain"])),
        ("Null mean gain", fmt(summary["null_mean_gain"])),
        ("Null SD gain", fmt(summary["null_sd_gain"])),
        ("Permutation $p$", fmtp(summary["permutation_p"])),
        ("Observed percentile", fmt(summary["observed_percentile"])),
    ]
    table("Permutation test for semantic-prior item structure.", "tab:permutation", ["Statistic", "Value"], rows, "table_permutation.tex")


def write_mixed_effects_table() -> None:
    df = pd.read_csv(OUTPUTS / "mixed_effects_validation.csv")
    rows = [
        (
            row["term"].replace("_", " "),
            fmt(row["coefficient"]),
            fmt(row["std_error"]),
            f"[{fmt(row['ci_lower'])}, {fmt(row['ci_upper'])}]",
            fmtp(row["p_value"]),
        )
        for _, row in df.iterrows()
    ]
    table("Mixed-effects validation of semantic margin predicting trial-level $\\rho$.", "tab:mixed-effects", ["Term", "Coefficient", "SE", "95\\% CI", "$p$"], rows, "table_mixed_effects.tex")


def write_item_recovery_table(summary: dict[str, Any]) -> None:
    item_tests = summary["item_level_tests"]
    recovery = summary["parameter_recovery"]
    rows = [
        ("$\\rho$ vs error rate", "Spearman $\\rho$", fmt(item_tests["rho_vs_error_rate_spearman"]["rho"])),
        ("$\\rho$ vs item RT", "Spearman $\\rho$", fmt(item_tests["rho_vs_item_rt_spearman"]["rho"])),
        ("Action gap vs error rate", "Spearman $\\rho$", fmt(item_tests["action_gap_vs_error_rate_spearman"]["rho"])),
        ("Parameter recovery", "Correlation", fmt(recovery["overall_correlation"])),
        ("Parameter recovery", "MAE", fmt(recovery["overall_mae"])),
        ("Simulated trajectories", "$n$", recovery["n_simulated"]),
    ]
    table("Item-level difficulty correlations and parameter recovery.", "tab:item-recovery", ["Analysis", "Statistic", "Value"], rows, "table_item_recovery.tex")


def table(caption: str, label: str, headers: list[str], rows: list[tuple[Any, ...]], filename: str, small: bool = False) -> None:
    spec = " ".join(["l"] + ["r" for _ in headers[1:]])
    lines = ["\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    if small:
        lines.append("\\small")
        lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.extend(["\\begin{tabular}{" + spec + "}", "\\toprule", " & ".join(headers) + " \\\\", "\\midrule"])
    for row in rows:
        lines.append(" & ".join(escape(cell) for cell in row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    if small:
        lines.append("}%")
    lines.extend(["\\end{table}", ""])
    (TABLES / filename).write_text("\n".join(lines), encoding="utf-8")


def fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmtp(value: Any) -> str:
    try:
        p = float(value)
    except (TypeError, ValueError):
        return str(value)
    if p < 0.001:
        return "$<.001$"
    return f"{p:.4f}"


def escape(value: Any) -> str:
    text = str(value)
    if "$" in text or "\\" in text:
        return text
    return (
        text.replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


if __name__ == "__main__":
    raise SystemExit(main())
