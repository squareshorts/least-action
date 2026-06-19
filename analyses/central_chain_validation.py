from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


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


BOOTSTRAP_SEED = 20260517
NEGATIVE_CONTROL_SEED = 20260518


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = ROOT / "outputs"
    root_tables = ROOT / "tables"
    output_tables = outputs / "tables"
    figures = ROOT / "figures"
    supplement = outputs / "supplement"
    for path in [root_tables, output_tables, figures, supplement]:
        path.mkdir(parents=True, exist_ok=True)

    trial_fits = pd.read_csv(outputs / "trial_fits.csv")
    item_summary = pd.read_csv(outputs / "item_level_action_summary.csv")
    semantic_scores = pd.read_csv(args.semantic_scores)
    stoch_trials = pd.read_csv(outputs / "stochastic_nll_trials.csv")

    trial_df = prepare_trial_dataframe(trial_fits, semantic_scores)
    item_df = prepare_item_dataframe(item_summary)

    trial_path, trial_rows = trial_path_models(trial_df)
    item_path, item_boot = item_mediation_bootstrap(item_df, args.bootstrap)
    negative, negative_details = negative_control_semantics(
        item_df=item_df,
        semantic_scores=semantic_scores,
        stoch_trials=stoch_trials,
        n_permutations=args.negative_permutations,
    )
    influence, influence_summary = item_slope_influence(item_df, args.bootstrap)

    write_trial_path_table(trial_path, root_tables / "table_path_mediation_trial.tex")
    write_item_path_table(item_path, root_tables / "table_path_mediation_item.tex")
    write_negative_control_table(negative, root_tables / "table_negative_control_semantics.tex")
    write_influence_summary_table(
        influence_summary, root_tables / "table_item_slope_influence_summary.tex"
    )
    write_influence_table(influence, root_tables / "table_item_slope_influence.tex")
    plot_jackknife_slope(influence, influence_summary, figures / "semantic_slope_jackknife.png")

    for name in [
        "table_path_mediation_trial.tex",
        "table_path_mediation_item.tex",
        "table_negative_control_semantics.tex",
        "table_item_slope_influence_summary.tex",
        "table_item_slope_influence.tex",
    ]:
        shutil.copy2(root_tables / name, output_tables / name)

    # Also keep CSVs for auditability.
    trial_path.to_csv(supplement / "path_mediation_trial.csv", index=False)
    item_path.to_csv(supplement / "path_mediation_item.csv", index=False)
    item_boot.to_csv(supplement / "path_mediation_item_bootstrap.csv", index=False)
    negative.to_csv(supplement / "negative_control_semantics.csv", index=False)
    pd.DataFrame(negative_details).to_csv(
        supplement / "negative_control_random_competitor_null.csv", index=False
    )
    influence.to_csv(supplement / "item_slope_influence.csv", index=False)
    pd.DataFrame([influence_summary]).to_csv(
        supplement / "item_slope_influence_summary.csv", index=False
    )

    summary = {
        "trial_path": trial_rows,
        "item_path": item_path.to_dict(orient="records"),
        "negative_controls": negative.to_dict(orient="records"),
        "item_influence": influence_summary,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "negative_control_seed": NEGATIVE_CONTROL_SEED,
    }
    (outputs / "central_chain_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


def prepare_trial_dataframe(
    trial_fits: pd.DataFrame, semantic_scores: pd.DataFrame
) -> pd.DataFrame:
    df = trial_fits.merge(
        semantic_scores[["exemplar", "semantic_margin"]],
        on="exemplar",
        how="left",
        suffixes=("", "_sem"),
    )
    if "semantic_margin_sem" in df.columns:
        df["semantic_margin"] = df["semantic_margin"].fillna(df["semantic_margin_sem"])
        df = df.drop(columns=["semantic_margin_sem"])
    df = df.dropna(
        subset=[
            "subject",
            "exemplar",
            "semantic_margin",
            "rho_hat",
            "auc",
            "max_deviation",
            "rt_s",
        ]
    ).copy()
    df["subject"] = df["subject"].astype(str)
    df["exemplar"] = df["exemplar"].astype(str)
    return df


def prepare_item_dataframe(item_summary: pd.DataFrame) -> pd.DataFrame:
    df = item_summary.dropna(subset=["semantic_margin", "rho_hat"]).copy()
    rt_col = "raw_rt_s" if "raw_rt_s" in df.columns else "rt_s"
    df["rt_item_s"] = df[rt_col]
    df["word_length"] = df["exemplar"].astype(str).str.len()
    df["atypical"] = (df["condition"] == "Atypical").astype(float)
    df["reversed_margin"] = -df["semantic_margin"]
    df["target_similarity"] = df["semantic_similarity_target"]
    df["competitor_similarity"] = df["semantic_similarity_competitor"]
    return df


def trial_path_models(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    outcomes = [
        ("auc", "AUC"),
        ("max_deviation", "Maximum deviation"),
        ("rt_s", "Response time"),
    ]

    a_fit = fit_mixed_or_clustered_ols("rho_hat ~ semantic_margin", df)
    a = coef(a_fit, "semantic_margin")
    rows: list[dict[str, Any]] = []
    display_rows: list[dict[str, Any]] = []

    for outcome, label in outcomes:
        total_fit = fit_mixed_or_clustered_ols(f"{outcome} ~ semantic_margin", df)
        out_fit = fit_mixed_or_clustered_ols(f"{outcome} ~ rho_hat + semantic_margin", df)
        b = coef(out_fit, "rho_hat")
        c_total = coef(total_fit, "semantic_margin")
        c_direct = coef(out_fit, "semantic_margin")
        indirect = a * b
        prop = indirect / c_total if c_total != 0 else math.nan
        row = {
            "outcome": outcome,
            "outcome_label": label,
            "a_margin_to_rho": a,
            "a_p": pvalue(a_fit, "semantic_margin"),
            "b_rho_to_outcome": b,
            "b_p": pvalue(out_fit, "rho_hat"),
            "indirect_a_times_b": indirect,
            "total_margin_effect": c_total,
            "total_p": pvalue(total_fit, "semantic_margin"),
            "direct_margin_effect": c_direct,
            "direct_p": pvalue(out_fit, "semantic_margin"),
            "proportion_indirect": prop,
            "n_observations": int(out_fit.nobs),
            "n_subjects": int(df["subject"].nunique()),
            "n_items": int(df["exemplar"].nunique()),
            "estimator": getattr(out_fit, "_least_action_estimator", "mixedlm"),
        }
        rows.append(row)
        display_rows.append(
            {
                "Outcome": label,
                "$a$: margin $\\rightarrow \\rho$": a,
                "$b$: $\\rho \\rightarrow$ outcome": b,
                "Indirect $a b$": indirect,
                "Total margin effect": c_total,
                "Direct margin effect": c_direct,
                "\\% indirect": 100.0 * prop if math.isfinite(prop) else math.nan,
            }
        )
    return pd.DataFrame(display_rows), rows


def fit_mixed_or_clustered_ols(formula: str, df: pd.DataFrame) -> Any:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = smf.mixedlm(
                formula,
                df,
                groups=df["subject"],
                vc_formula={"item": "0 + C(exemplar)"},
            ).fit(reml=False, method="lbfgs", maxiter=1000, disp=False)
        result._least_action_estimator = "mixedlm_subject_random_intercept_item_vc"
        return result
    except Exception:
        # If MixedLM hits a singular fit, preserve the same fixed-effect question
        # with subject fixed effects and item-clustered standard errors.
        fallback = formula + " + C(subject)"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = smf.ols(fallback, df).fit(
                cov_type="cluster", cov_kwds={"groups": df["exemplar"]}
            )
        result._least_action_estimator = "ols_subject_fixed_effect_item_clustered"
        return result


def coef(result: Any, term: str) -> float:
    return float(result.params.get(term, math.nan))


def pvalue(result: Any, term: str) -> float:
    return float(result.pvalues.get(term, math.nan))


def item_mediation_bootstrap(
    df: pd.DataFrame, n_bootstrap: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = [
        ("auc", "AUC"),
        ("max_deviation", "Maximum deviation"),
        ("rt_item_s", "Response time"),
        ("error_rate", "Error rate"),
    ]

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    records: list[dict[str, Any]] = []
    boot_records: list[dict[str, Any]] = []

    for outcome, label in outcomes:
        point = item_path_coefficients(df, outcome)
        boot_rows = []
        for _ in range(n_bootstrap):
            sample = df.iloc[rng.integers(0, len(df), size=len(df))].copy()
            if sample["semantic_margin"].nunique() < 2 or sample["rho_hat"].nunique() < 2:
                continue
            try:
                boot_rows.append(item_path_coefficients(sample, outcome))
            except Exception:
                continue
        boot = pd.DataFrame(boot_rows)
        for _, row in boot.iterrows():
            rec = row.to_dict()
            rec["outcome"] = outcome
            boot_records.append(rec)
        indirect_ci = percentile_ci(boot["indirect"])
        direct_ci = percentile_ci(boot["direct"])
        total_ci = percentile_ci(boot["total"])
        prop = point["indirect"] / point["total"] if point["total"] != 0 else math.nan
        records.append(
            {
                "Outcome": label,
                "Total margin effect": point["total"],
                "Indirect via $\\rho$": point["indirect"],
                "Indirect 95\\% CI": format_ci(indirect_ci),
                "Direct margin effect": point["direct"],
                "Direct 95\\% CI": format_ci(direct_ci),
                "\\% indirect": 100.0 * prop if math.isfinite(prop) else math.nan,
                "Total 95\\% CI": format_ci(total_ci),
            }
        )
    return pd.DataFrame(records), pd.DataFrame(boot_records)


def item_path_coefficients(df: pd.DataFrame, outcome: str) -> dict[str, float]:
    med = smf.ols("rho_hat ~ semantic_margin", df).fit()
    total = smf.ols(f"{outcome} ~ semantic_margin", df).fit()
    out = smf.ols(f"{outcome} ~ rho_hat + semantic_margin", df).fit()
    a = float(med.params["semantic_margin"])
    b = float(out.params["rho_hat"])
    return {
        "a": a,
        "b": b,
        "indirect": a * b,
        "total": float(total.params["semantic_margin"]),
        "direct": float(out.params["semantic_margin"]),
    }


def percentile_ci(values: pd.Series | np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (math.nan, math.nan)
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))


def negative_control_semantics(
    item_df: pd.DataFrame,
    semantic_scores: pd.DataFrame,
    stoch_trials: pd.DataFrame,
    n_permutations: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    resources = build_action_resources(semantic_scores)
    condition_nll = resources["condition_nll"]
    observed_condition_mean = float(condition_nll.mean())
    observed_gain = float(
        observed_condition_mean
        - stoch_trials.loc[
            stoch_trials["model"] == "action_semantic_margin_only_rho", "nll"
        ]
        .groupby(stoch_trials.loc[stoch_trials["model"] == "action_semantic_margin_only_rho", "exemplar"])
        .mean()
        .mean()
    )

    predictors = [
        (
            "Real target--competitor margin",
            "semantic_margin",
            "negative",
            "Primary semantic contrast",
        ),
        (
            "Reversed competitor--target margin",
            "reversed_margin",
            "negative",
            "Orientation control",
        ),
        ("Target similarity alone", "target_similarity", "negative", "Component control"),
        (
            "Competitor similarity alone",
            "competitor_similarity",
            "positive",
            "Component control",
        ),
        ("Word length", "word_length", "none", "Lexical control"),
        ("Atypical label", "atypical", "positive", "Condition label"),
    ]

    rows: list[dict[str, Any]] = []
    for label, column, expected, role in predictors:
        item_predictor = item_df.set_index("exemplar")[column]
        rho_stats = item_rho_predictor_stats(item_df, column, expected)
        if column == "atypical":
            mean_nll = observed_condition_mean
            gain = 0.0
            improved = 0
        else:
            pred_res = predictor_nll(item_predictor, resources)
            mean_nll = pred_res["mean_nll"]
            gain = observed_condition_mean - mean_nll
            improved = pred_res["n_items_improved"]
        rows.append(
            {
                "Predictor": label,
                "Role": role,
                "Slope": rho_stats["slope"],
                "$R^2$": rho_stats["r2"],
                "Direction ok": rho_stats["direction_ok"],
                "Mean NLL": mean_nll,
                "NLL gain": gain,
                "Improved items": improved,
                "Null/notes": "",
            }
        )
        if column == "semantic_margin":
            rows[-1]["Null/notes"] = "primary"
        elif column == "reversed_margin":
            rows[-1]["Null/notes"] = "algebraic mirror; sign fails"
        elif column in {"target_similarity", "competitor_similarity"}:
            rows[-1]["Null/notes"] = "single component"
        elif column == "word_length":
            rows[-1]["Null/notes"] = "nonsemantic"
        elif column == "atypical":
            rows[-1]["Null/notes"] = "condition reference"

    rng = np.random.default_rng(NEGATIVE_CONTROL_SEED)
    margin_values = item_df["semantic_margin"].to_numpy()
    comp_values = item_df["competitor_similarity"].to_numpy()
    target_values = item_df["target_similarity"].to_numpy()
    random_rows: list[dict[str, Any]] = []
    perm_gains = []
    random_comp_gains = []
    index = item_df["exemplar"].to_numpy()
    for permutation in range(n_permutations):
        shuffled_margin = pd.Series(rng.permutation(margin_values), index=index)
        perm_res = predictor_nll(shuffled_margin, resources)
        perm_gains.append(perm_res["gain"])

        shuffled_comp = rng.permutation(comp_values)
        fake_margin = pd.Series(target_values - shuffled_comp, index=index)
        comp_res = predictor_nll(fake_margin, resources)
        random_comp_gains.append(comp_res["gain"])
        random_rows.append(
            {
                "permutation": permutation + 1,
                "permuted_margin_gain": perm_res["gain"],
                "random_competitor_score_gain": comp_res["gain"],
            }
        )

    rows.append(
        null_row(
            "Permuted margins",
            "Permutation control",
            np.asarray(perm_gains),
            observed_gain,
            n_permutations,
        )
    )
    rows.append(
        null_row(
            "Random competitor-score assignment",
            "Competitor assignment control",
            np.asarray(random_comp_gains),
            observed_gain,
            n_permutations,
        )
    )
    return pd.DataFrame(rows), random_rows


def build_action_resources(semantic_scores: pd.DataFrame) -> dict[str, Any]:
    config = load_model_config(ROOT / "config" / "model_config.yaml")
    n_time = int(config_value(config, "model.n_time", 51))
    rho_min = float(config_value(config, "model.rho_grid.min", 0.0))
    rho_max = float(config_value(config, "model.rho_grid.max", 2.0))
    rho_step = float(config_value(config, "model.rho_grid.step", 0.05))
    maxiter = int(config_value(config, "model.optimizer.maxiter", 500))
    rhos = np.round(np.arange(rho_min, rho_max + 0.5 * rho_step, rho_step), 10)

    raw = pd.read_csv(ensure_kh2017_csv(str(ROOT / "data")))
    data = preprocess_kh2017(raw, n_time=n_time)
    metadata = data.metadata.reset_index(drop=True).copy()
    action_grid = precompute_action_grid(rhos, n_time, ActionParams(maxiter=maxiter))
    rmse_matrix = trajectory_rmse_matrix(data.trajectories, action_grid.target_paths)
    best_rhos_by_trial = action_grid.rhos[np.argmin(rmse_matrix, axis=1)]

    items = sorted(metadata["exemplar"].unique())
    folds = [
        (
            item,
            np.flatnonzero(metadata["exemplar"].to_numpy() != item),
            np.flatnonzero(metadata["exemplar"].to_numpy() == item),
        )
        for item in items
    ]
    condition_nll = condition_only_item_nll(metadata, rmse_matrix, action_grid.rhos, folds, n_time)
    return {
        "metadata": metadata,
        "rmse_matrix": rmse_matrix,
        "rhos": action_grid.rhos,
        "best_rhos_by_trial": best_rhos_by_trial,
        "folds": folds,
        "condition_nll": condition_nll,
        "n_time": n_time,
    }


def condition_only_item_nll(
    metadata: pd.DataFrame,
    rmse_matrix: np.ndarray,
    rhos: np.ndarray,
    folds: list[tuple[str, np.ndarray, np.ndarray]],
    n_time: int,
) -> pd.Series:
    rows: list[tuple[str, float]] = []
    for item, train_idx, test_idx in folds:
        best = fit_condition_rhos(train_idx, metadata, rmse_matrix, rhos)
        train_rmses = np.asarray(
            [
                rmse_matrix[tidx, best[metadata.loc[tidx, "condition"]]["rho_index"]]
                for tidx in train_idx
            ],
            dtype=float,
        )
        tau2 = max(float(np.mean(train_rmses**2)) / 2.0, 1e-8)
        nll_values = []
        for idx in test_idx:
            condition = metadata.loc[idx, "condition"]
            rho_index = best[condition]["rho_index"]
            nll_values.append(-gaussian_path_loglik(float(rmse_matrix[idx, rho_index]), tau2, n_time))
        rows.append((item, float(np.mean(nll_values))))
    return pd.Series(dict(rows))


def predictor_nll(item_predictor: pd.Series, resources: dict[str, Any]) -> dict[str, Any]:
    metadata: pd.DataFrame = resources["metadata"]
    rmse_matrix: np.ndarray = resources["rmse_matrix"]
    rhos: np.ndarray = resources["rhos"]
    best_rhos_by_trial: np.ndarray = resources["best_rhos_by_trial"]
    folds = resources["folds"]
    condition_nll: pd.Series = resources["condition_nll"]
    n_time: int = resources["n_time"]

    item_predictor = item_predictor.astype(float)
    predictor_by_trial = metadata["exemplar"].map(item_predictor).to_numpy(dtype=float)
    item_nll: dict[str, float] = {}

    for item, train_idx, test_idx in folds:
        x_train = predictor_by_trial[train_idx]
        y_train = best_rhos_by_trial[train_idx]
        ok = np.isfinite(x_train) & np.isfinite(y_train)
        x_train = x_train[ok]
        y_train = y_train[ok]
        if len(np.unique(x_train)) < 2:
            intercept = float(np.mean(y_train))
            slope = 0.0
        else:
            design = np.column_stack([np.ones_like(x_train), x_train])
            intercept, slope = np.linalg.lstsq(design, y_train, rcond=None)[0]

        train_pred = intercept + slope * predictor_by_trial[train_idx]
        train_ok = np.isfinite(train_pred)
        train_rho_idx = nearest_rho_indices(train_pred[train_ok], rhos)
        train_rmse = rmse_matrix[train_idx[train_ok], train_rho_idx]
        tau2 = max(float(np.mean(train_rmse**2)) / 2.0, 1e-8)

        test_pred = intercept + slope * predictor_by_trial[test_idx]
        test_ok = np.isfinite(test_pred)
        test_rho_idx = nearest_rho_indices(test_pred[test_ok], rhos)
        nll_values = [
            -gaussian_path_loglik(float(rmse_matrix[idx, rho_idx]), tau2, n_time)
            for idx, rho_idx in zip(test_idx[test_ok], test_rho_idx)
        ]
        item_nll[item] = float(np.mean(nll_values))

    predictor_nlls = pd.Series(item_nll).reindex(condition_nll.index)
    gains = condition_nll - predictor_nlls
    return {
        "mean_nll": float(predictor_nlls.mean()),
        "gain": float(gains.mean()),
        "n_items_improved": int((gains > 0).sum()),
    }


def nearest_rho_indices(values: np.ndarray, rhos: np.ndarray) -> np.ndarray:
    return np.abs(values[:, None] - rhos[None, :]).argmin(axis=1)


def item_rho_predictor_stats(df: pd.DataFrame, column: str, expected: str) -> dict[str, Any]:
    fit = smf.ols(f"rho_hat ~ {column}", df).fit()
    slope = float(fit.params[column])
    if expected == "negative":
        direction_ok: bool | str = slope < 0
    elif expected == "positive":
        direction_ok = slope > 0
    else:
        direction_ok = "n/a"
    return {"slope": slope, "r2": float(fit.rsquared), "direction_ok": direction_ok}


def null_row(
    label: str, role: str, gains: np.ndarray, observed_gain: float, n_permutations: int
) -> dict[str, Any]:
    p = float((1 + np.sum(gains >= observed_gain)) / (len(gains) + 1))
    return {
        "Predictor": label,
        "Role": role,
        "Slope": math.nan,
        "$R^2$": math.nan,
        "Direction ok": "n/a",
        "Mean NLL": math.nan,
        "NLL gain": float(gains.mean()),
        "Improved items": math.nan,
        "Null/notes": f"null p={format_p(p)}; {n_permutations} draws",
    }


def item_slope_influence(
    df: pd.DataFrame, n_bootstrap: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    full = smf.ols("rho_hat ~ semantic_margin", df).fit()
    full_slope = float(full.params["semantic_margin"])
    rows: list[dict[str, Any]] = []
    for excluded in sorted(df["exemplar"]):
        sub = df.loc[df["exemplar"] != excluded]
        fit = smf.ols("rho_hat ~ semantic_margin", sub).fit()
        spear = stats.spearmanr(sub["semantic_margin"], sub["rho_hat"])
        rows.append(
            {
                "Excluded item": excluded,
                "Slope without item": float(fit.params["semantic_margin"]),
                "Slope p": float(fit.pvalues["semantic_margin"]),
                "Spearman": float(spear.statistic),
                "Spearman p": float(spear.pvalue),
            }
        )
    influence = pd.DataFrame(rows).sort_values("Slope without item")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    slope_boot = []
    spearman_boot = []
    for _ in range(n_bootstrap):
        sample = df.iloc[rng.integers(0, len(df), size=len(df))]
        if sample["semantic_margin"].nunique() < 2 or sample["rho_hat"].nunique() < 2:
            continue
        try:
            slope_boot.append(float(smf.ols("rho_hat ~ semantic_margin", sample).fit().params["semantic_margin"]))
            spearman_boot.append(float(stats.spearmanr(sample["semantic_margin"], sample["rho_hat"]).statistic))
        except Exception:
            continue

    theil = stats.theilslopes(df["rho_hat"], df["semantic_margin"], alpha=0.95)
    summary = {
        "Full OLS slope": full_slope,
        "Full OLS p": float(full.pvalues["semantic_margin"]),
        "Jackknife slope min": float(influence["Slope without item"].min()),
        "Jackknife slope max": float(influence["Slope without item"].max()),
        "All jackknife slopes negative": bool((influence["Slope without item"] < 0).all()),
        "Bootstrap slope CI lower": percentile_ci(np.asarray(slope_boot))[0],
        "Bootstrap slope CI upper": percentile_ci(np.asarray(slope_boot))[1],
        "Full Spearman": float(stats.spearmanr(df["semantic_margin"], df["rho_hat"]).statistic),
        "Bootstrap Spearman CI lower": percentile_ci(np.asarray(spearman_boot))[0],
        "Bootstrap Spearman CI upper": percentile_ci(np.asarray(spearman_boot))[1],
        "Theil-Sen slope": float(theil.slope),
        "Theil-Sen CI lower": float(theil.low_slope),
        "Theil-Sen CI upper": float(theil.high_slope),
    }
    return influence, summary


def write_trial_path_table(df: pd.DataFrame, path: Path) -> None:
    write_latex_table(
        path,
        (
            "Trial-level path estimates for the semantic margin $\\rightarrow \\rho "
            "\\rightarrow$ behavior chain. Fixed effects come from mixed models with "
            "subject grouping and an item variance component."
        ),
        "tab:path-mediation-trial",
        df,
        align="lrrrrrr",
        small=True,
    )


def write_item_path_table(df: pd.DataFrame, path: Path) -> None:
    write_latex_table(
        path,
        (
            "Item-level mediation/path estimates with item-resampling bootstrap "
            f"confidence intervals ({BOOTSTRAP_SEED} seed; 4,000 resamples)."
        ),
        "tab:path-mediation-item",
        df,
        align="lrrrrrrr",
        small=True,
    )


def write_negative_control_table(df: pd.DataFrame, path: Path) -> None:
    display = df.copy()
    display["Improved items"] = display["Improved items"].map(
        lambda value: "--" if pd.isna(value) else f"{int(value)}"
    )
    write_latex_table(
        path,
        (
            "Targeted semantic negative controls. NLL gain is item-weighted "
            "condition-only action NLL minus predictor-based action NLL; positive "
            "values favor the predictor."
        ),
        "tab:negative-control-semantics",
        display,
        align="llrrlrrrl",
        small=True,
    )


def write_influence_summary_table(summary: dict[str, Any], path: Path) -> None:
    rows_list = []
    for key, value in summary.items():
        if " p" in key.lower() or key.lower().endswith("p"):
            display_value = format_p(value)
        elif isinstance(value, bool):
            display_value = "Yes" if value else "No"
        elif isinstance(value, (float, np.floating)):
            display_value = f"{float(value):.3f}"
        else:
            display_value = value
        rows_list.append({"Statistic": key, "Value": display_value})
    rows = pd.DataFrame(
        rows_list
    )
    write_latex_table(
        path,
        "Item-level semantic slope influence summary.",
        "tab:item-slope-influence-summary",
        rows,
        align="lr",
    )


def write_influence_table(df: pd.DataFrame, path: Path) -> None:
    display = df.copy()
    for column in ["Slope p", "Spearman p"]:
        display[column] = display[column].map(format_p)
    write_latex_table(
        path,
        "Leave-one-item-out influence on the item-level semantic margin slope.",
        "tab:item-slope-influence",
        display,
        align="lrrrr",
        small=True,
    )


def plot_jackknife_slope(
    influence: pd.DataFrame, summary: dict[str, Any], path: Path
) -> None:
    plot_df = influence.sort_values("Slope without item", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    y = np.arange(len(plot_df))
    point_colors = np.where(plot_df["Excluded item"].eq("Wal"), "#c65a4a", "#2a9d8f")
    point_sizes = np.where(plot_df["Excluded item"].eq("Wal"), 58, 38)
    ax.scatter(
        plot_df["Slope without item"],
        y,
        color=point_colors,
        s=point_sizes,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    ax.axvline(0.0, color="#9a9a9a", linewidth=1.0, linestyle=":", zorder=1)
    ax.axvline(summary["Full OLS slope"], color="#222222", linewidth=1.2, linestyle="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Excluded item"], fontsize=9)
    ax.set_xlabel(r"Jackknife slope predicting fitted $\rho$", fontsize=11)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.set_xlim(min(plot_df["Slope without item"].min() - 0.04, -0.62), 0.04)
    ax.set_ylim(-0.7, len(plot_df) - 0.3)
    ax.grid(axis="x", color="#e8e8e8", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.annotate(
        "Wal",
        xy=(
            float(plot_df.loc[plot_df["Excluded item"].eq("Wal"), "Slope without item"].iloc[0]),
            int(plot_df.index[plot_df["Excluded item"].eq("Wal")][0]),
        ),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color="#7a2e28",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_latex_table(
    path: Path,
    caption: str,
    label: str,
    df: pd.DataFrame,
    align: str | None = None,
    small: bool = False,
) -> None:
    if align is None:
        align = "l" + "r" * (len(df.columns) - 1)
    lines = [r"\begin{table}[htbp]", r"\centering"]
    if small:
        lines.append(r"\small")
    lines += [
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
    ]
    if small:
        lines.append(r"\resizebox{\linewidth}{!}{%")
    lines += [
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(str(c) for c in df.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(latex_cell(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if small:
        lines.append("}%")
    lines += [r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def latex_cell(value: object) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        if math.isnan(value):
            return "--"
        return f"{value:.3f}"
    if isinstance(value, (np.floating,)):
        number = float(value)
        if math.isnan(number):
            return "--"
        return f"{number:.3f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    text = str(value)
    if text == "":
        return "--"
    if text in {"Yes", "No", "n/a"}:
        return text
    if text.startswith("$") or "\\" in text:
        return text
    return (
        text.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def format_ci(ci: tuple[float, float]) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def format_p(value: object) -> str:
    try:
        p = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(p):
        return "--"
    if p < 0.001:
        return "$<.001$"
    return f"{p:.3f}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-scores", default="data/processed/semantic_scores.csv")
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--negative-permutations", type=int, default=5000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
