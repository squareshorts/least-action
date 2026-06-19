from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from least_action_mouse.action_model import ActionParams, precompute_action_grid
from least_action_mouse.analysis import path_rmse_vector, trajectory_rmse_matrix
from least_action_mouse.baselines import (
    bezier_attraction_paths,
    gaussian_path_loglik,
    spline_attraction_paths,
)
from least_action_mouse.config import config_value, load_model_config
from least_action_mouse.preprocess import TrajectoryData, minimum_jerk_path


RAW = ROOT / "data" / "external" / "koenig_robert_2024" / "raw"
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"

DOWNLOAD_DATE = "2026-06-18"
OSF_PROJECT_URL = "https://osf.io/q3hbp/"
OSF_API_URL = "https://api.osf.io/v2/nodes/q3hbp/files/osfstorage/"
ARTICLE_URL = "https://www.nature.com/articles/s41598-024-62135-7"
N_BOOT = 1000
N_PERM = 2000
N_NULL = 50
RNG_SEED = 20260618

FILES = {
    "data_animal.zip": {
        "url": "https://osf.io/download/47anw/",
        "sha256": "ea174a0097abf9bc898e9184e65001e549c9ecdf1d45327072834656c93c4fda",
    },
    "data_face.zip": {
        "url": "https://osf.io/download/6fyzm/",
        "sha256": "581314b38a939d00b52f858f39bee36850aaba624db214e8f02361d672dfeb13",
    },
    "preprocess_trajectories.m": {
        "url": "https://osf.io/download/cf58z/",
        "sha256": "b1a55565def3d97fc94b62c07b6c9984ab06b0f9d3f52f9af50a05c8bef92a17",
    },
    "analyses_animals_final.m": {
        "url": "https://osf.io/download/4fqhd/",
        "sha256": "5b877eac0583ae8aa7528679bb731c8d8cdca9b411b3d87ac8d46cb295cc917c",
    },
    "analyses_faces_final.m": {
        "url": "https://osf.io/download/mjcgd/",
        "sha256": "47b867b28883182b0354deee9648f83049cf00bf6ea870ba6a411b238a3d6148",
    },
}


@dataclass(frozen=True)
class StudySpec:
    study: str
    zip_name: str
    zip_root: str
    expected_trials: int
    block_size: int
    primary_pair: tuple[str, str]
    true_category: str
    excluded_condition: str
    positive_label_after_alignment: str


STUDIES = [
    StudySpec(
        study="face_object",
        zip_name="data_face.zip",
        zip_root="data_face",
        expected_trials=512,
        block_size=128,
        primary_pair=("ordinary_object", "face_like_object"),
        true_category="face",
        excluded_condition="extra_face",
        positive_label_after_alignment="FACE",
    ),
    StudySpec(
        study="animal_object",
        zip_name="data_animal.zip",
        zip_root="data_animal",
        expected_trials=144,
        block_size=36,
        primary_pair=("ordinary_object", "animal_like_object"),
        true_category="animal",
        excluded_condition="extra_animal",
        positive_label_after_alignment="OBJECT",
    ),
]


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    config = load_model_config()
    n_time = int(config_value(config, "model.n_time", 51))
    rho_min = float(config_value(config, "model.rho_grid.min", 0.0))
    rho_max = float(config_value(config, "model.rho_grid.max", 2.0))
    rho_step = float(config_value(config, "model.rho_grid.step", 0.05))
    maxiter = int(config_value(config, "model.optimizer.maxiter", 500))
    rhos = np.round(np.arange(rho_min, rho_max + 0.5 * rho_step, rho_step), 10)
    action_grid = precompute_action_grid(rhos, n_time, ActionParams(maxiter=maxiter))

    study_frames: list[pd.DataFrame] = []
    trial_frames: list[pd.DataFrame] = []
    preprocessing: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}

    for spec in STUDIES:
        print(f"[{spec.study}] loading raw CSV archive", flush=True)
        raw_trials, raw_report = load_raw_study(spec)
        print(f"[{spec.study}] canonicalizing trajectories", flush=True)
        data, trial_df, prep_report = canonicalize_study(raw_trials, spec, n_time)
        print(f"[{spec.study}] fitting action-grid validation models", flush=True)
        fit_df, stimulus_df, condition_df, nll_df, controls = analyze_study(data, action_grid, spec)
        trial_df = trial_df.merge(
            fit_df[["source_row", "rho_hat", "action_gap", "rmse"]],
            on="source_row",
            how="left",
        )
        study_frames.append(stimulus_df)
        trial_frames.append(trial_df)
        preprocessing.append({**raw_report, **prep_report})
        diagnostics[spec.study] = {
            "condition_tests": condition_df.to_dict(orient="records"),
            "heldout_nll": nll_df.to_dict(orient="records"),
            "controls": controls,
        }

    stimulus_all = pd.concat(study_frames, ignore_index=True)
    trials_all = pd.concat(trial_frames, ignore_index=True)
    validation = build_validation_table(stimulus_all, diagnostics)

    validation.to_csv(RESULTS / "external_koenig_robert_validation.csv", index=False)
    # Save fit-level audit outputs before rendering so a plotting failure cannot
    # discard the expensive preprocessing/model-fitting products.
    trials_all.to_csv(RESULTS / "external_koenig_robert_trial_fits.csv", index=False)
    stimulus_all.to_csv(RESULTS / "external_koenig_robert_stimulus_rho.csv", index=False)
    write_preprocessing_report(preprocessing, diagnostics)
    write_latex_table(validation, TABLES / "table_external_mouse_tracking_validation.tex")
    plot_rho_by_condition(stimulus_all, FIGURES / "external_rho_by_condition.png")
    plot_ambiguity_rho(stimulus_all, FIGURES / "external_ambiguity_rho.png")
    print(f"Wrote {RESULTS / 'external_koenig_robert_validation.csv'}")
    return 0


def load_raw_study(spec: StudySpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    zip_path = RAW / spec.zip_name
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing {zip_path}. Download it from {FILES[spec.zip_name]['url']}.")

    records: list[dict[str, Any]] = []
    raw_mouse_rows = 0
    complete_files = 0
    incomplete_files = 0
    csv_files = 0

    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(
            n
            for n in zf.namelist()
            if n.startswith(f"{spec.zip_root}/data/")
            and n.endswith(".csv")
            and not n.startswith("__MACOSX/")
        )
        csv_files = len(names)
        subject = 0
        for name in names:
            data = zf.read(name).decode("utf-8-sig", errors="replace")
            try:
                df = pd.read_csv(
                    io.StringIO(data),
                    dtype=str,
                    keep_default_na=False,
                    quoting=csv.QUOTE_MINIMAL,
                )
            except pd.errors.ParserError:
                try:
                    df = pd.read_csv(
                        io.StringIO(data),
                        dtype=str,
                        keep_default_na=False,
                        quoting=csv.QUOTE_MINIMAL,
                        engine="python",
                        on_bad_lines="skip",
                    )
                except pd.errors.ParserError:
                    incomplete_files += 1
                    continue
            mouse = df[df["trial_type"] == "mousetracking"].copy()
            raw_mouse_rows += len(mouse)
            demo = df[df["trial_type"] == "survey-html-form"]
            if demo.empty or len(mouse) != spec.expected_trials:
                incomplete_files += 1
                continue
            subject += 1
            complete_files += 1
            ppinfo = parse_participant_info(str(demo.iloc[0].get("responses", "")), spec.study)
            mouse = mouse.reset_index(drop=True)
            for i, row in mouse.iterrows():
                record = {
                    "study": spec.study,
                    "source_file": name,
                    "subject": subject,
                    "trialnr": i + 1,
                    "blocknr": int(math.ceil((i + 1) / spec.block_size)),
                    "stimulus": str(row["stimulus"]),
                    "x_position": str(row.get("x-position", "")),
                    "y_position": str(row.get("y-position", "")),
                    "mice_times": str(row.get("mice-times", "")),
                    "nRecordings": as_float(row.get("nRecordings", "")),
                }
                record.update(ppinfo)
                records.append(record)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError(f"No complete participant files found in {zip_path}.")
    df = add_stimulus_metadata(df, spec)
    df = add_author_validity_flags(df, spec)
    report = {
        "study": spec.study,
        "zip_file": spec.zip_name,
        "raw_csv_files": csv_files,
        "complete_participant_files": complete_files,
        "incomplete_or_missing_participant_files": incomplete_files,
        "raw_mouse_rows_in_all_csvs": int(raw_mouse_rows),
        "full_dataset_rows": int(len(df)),
        "subjects_before_quality_filter": int(df["subject"].nunique()),
        "participants_after_quality_filter": int(df.loc[df["include"], "subject"].nunique()),
    }
    return df, report


def parse_participant_info(text: str, study: str) -> dict[str, Any]:
    swaporder = ("cursor1" in text) if study == "animal_object" else ("cursor" in text)
    cleaned = text.replace("pointer", "cursor").replace("cursor1", "cursor")
    try:
        info = json.loads(cleaned)
    except json.JSONDecodeError:
        info = {}
    return {
        "age": as_float(info.get("age")),
        "gender": info.get("gender", ""),
        "handedness": info.get("handedness", ""),
        "native": info.get("native", ""),
        "pointer": info.get("cursor", ""),
        "swaporder": bool(swaporder),
    }


def add_stimulus_metadata(df: pd.DataFrame, spec: StudySpec) -> pd.DataFrame:
    df = df.copy()
    unique_stimuli = sorted(df["stimulus"].unique())
    stimulusnr = {stim: i + 1 for i, stim in enumerate(unique_stimuli)}
    if spec.study == "face_object":
        remap = []
        for a, b in zip(range(33, 65), range(65, 97)):
            remap.extend([a, b])
        remap.extend(range(97, 129))
        remap.extend(range(1, 33))
    else:
        remap = list(range(10, 19)) + list(range(1, 10)) + list(range(19, 37))
    if len(remap) < len(unique_stimuli):
        raise ValueError(f"Unexpected stimulus count for {spec.study}: {len(unique_stimuli)}")
    df["stimulusnr"] = df["stimulus"].map(stimulusnr).astype(int)
    df["sortedstimulusnr"] = df["stimulusnr"].map(lambda i: remap[i - 1]).astype(int)
    df["stimulus_id"] = df["sortedstimulusnr"].map(lambda x: f"{spec.study}_{x:03d}")

    lower = df["stimulus"].str.lower()
    if spec.study == "face_object":
        is_extra = lower.str.contains("extra_face", regex=False)
        is_face = lower.str.contains("face", regex=False)
        is_object = lower.str.contains("match", regex=False)
        is_lookalike = ~(is_face | is_object)
        df["condition"] = np.select(
            [is_extra, is_face, is_object, is_lookalike],
            ["extra_face", "face", "ordinary_object", "face_like_object"],
            default="unknown",
        )
        df["correct_response"] = np.where(is_face, "FACE", "OBJECT")
        df["competitor_response"] = np.where(is_face, "OBJECT", "FACE")
    else:
        is_extra = lower.str.contains("_ax", regex=False)
        is_animal = lower.str.contains("_a", regex=False)
        is_lookalike = lower.str.contains("_l", regex=False)
        is_object = lower.str.contains("_o", regex=False)
        df["condition"] = np.select(
            [is_extra, is_animal, is_object, is_lookalike],
            ["extra_animal", "animal", "ordinary_object", "animal_like_object"],
            default="unknown",
        )
        df["correct_response"] = np.where(is_animal, "ANIMAL", "OBJECT")
        df["competitor_response"] = np.where(is_animal, "OBJECT", "ANIMAL")
    df["ambiguity_proxy"] = df["condition"].isin(
        ["face_like_object", "animal_like_object"]
    ).astype(float)
    return df


def add_author_validity_flags(df: pd.DataFrame, spec: StudySpec) -> pd.DataFrame:
    df = df.copy()
    final_x: list[float] = []
    final_y: list[float] = []
    valid_trial: list[bool] = []
    selected: list[str] = []
    correct: list[bool] = []
    n_samples: list[int] = []

    for row in df.itertuples(index=False):
        x = parse_series(row.x_position) - 500.0
        y = 600.0 - parse_series(row.y_position)
        if (row.blocknr > 2 and not row.swaporder) or (row.blocknr <= 2 and row.swaporder):
            x = -x
        n_valid = int(np.sum(np.isfinite(x)))
        fx = float(x[-1]) if len(x) else math.nan
        fy = float(y[-1]) if len(y) else math.nan
        is_valid = bool(((abs(fx - 500.0) > 50.0) or (abs(600.0 - fy) > 50.0)) and n_valid > 10)
        if spec.study == "face_object":
            chosen = "FACE" if fx > 0 else "OBJECT"
        else:
            chosen = "OBJECT" if fx > 0 else "ANIMAL"
        final_x.append(fx)
        final_y.append(fy)
        valid_trial.append(is_valid)
        selected.append(chosen)
        correct.append(chosen == row.correct_response)
        n_samples.append(n_valid)

    df["final_x_author_px"] = final_x
    df["final_y_author_px"] = final_y
    df["author_valid_trial"] = valid_trial
    df["selected_response"] = selected
    df["correct"] = correct
    df["raw_valid_samples"] = n_samples

    subject_quality = (
        df.groupby("subject", as_index=False)
        .agg(mean_valid=("author_valid_trial", "mean"), mean_correct=("correct", "mean"))
    )
    subject_quality["subject_include"] = (
        (subject_quality["mean_valid"] > 0.70) & (subject_quality["mean_correct"] > 0.70)
    )
    include_map = subject_quality.set_index("subject")["subject_include"].to_dict()
    df["valid_subject"] = df["subject"].map(include_map).fillna(False).astype(bool)
    df["include"] = df["valid_subject"] & df["author_valid_trial"]
    return df


def canonicalize_study(raw: pd.DataFrame, spec: StudySpec, n_time: int) -> tuple[TrajectoryData, pd.DataFrame, dict[str, Any]]:
    trajectories: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    excluded_extra = raw["condition"].eq(spec.excluded_condition)
    analysis_pool = raw.loc[raw["include"] & ~excluded_extra].copy()
    correct_pool = analysis_pool.loc[analysis_pool["correct"]].copy()
    source_row = 0

    for row in correct_pool.itertuples(index=False):
        x = parse_series(row.x_position) - 500.0
        y = 600.0 - parse_series(row.y_position)
        t = parse_series(row.mice_times)
        if (row.blocknr > 2 and not row.swaporder) or (row.blocknr <= 2 and row.swaporder):
            x = -x
        n = min(len(x), len(y), len(t))
        if n < 5:
            continue
        x, y, t = x[:n], y[:n], t[:n]
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
        if finite.sum() < 5:
            continue
        x, y, t = x[finite], y[finite], t[finite]
        order = np.argsort(t)
        x, y, t = x[order], y[order], t[order]
        unique_t, unique_idx = np.unique(t, return_index=True)
        if len(unique_t) < 5 or unique_t[-1] <= unique_t[0]:
            continue
        x, y = x[unique_idx], y[unique_idx]
        timeline = np.linspace(0.0, 800.0, n_time)
        xi = np.interp(timeline, unique_t, x, left=x[0], right=x[-1])
        yi = np.interp(timeline, unique_t, y, left=y[0], right=y[-1])

        target_side = 1.0 if row.correct_response == spec.positive_label_after_alignment else -1.0
        dx = xi - xi[0]
        dy = yi - yi[0]
        hscale = abs(dx[-1])
        vscale = dy[-1]
        if hscale < 50.0 or vscale < 50.0:
            continue
        q = np.column_stack((target_side * dx / hscale, dy / vscale))
        exec_time = first_execution_time(xi, yi)
        source_row += 1
        trajectories.append(q)
        rows.append(
            {
                "source_row": source_row,
                "study": spec.study,
                "subject": int(row.subject),
                "trial": int(row.trialnr),
                "block": int(row.blocknr),
                "condition": str(row.condition),
                "stimulus": str(row.stimulus),
                "stimulus_id": str(row.stimulus_id),
                "sortedstimulusnr": int(row.sortedstimulusnr),
                "correct_response": str(row.correct_response),
                "competitor_response": str(row.competitor_response),
                "selected_response": str(row.selected_response),
                "response_time_ms": exec_time,
                "target_side_author": int(target_side),
                "horizontal_scale_px": float(hscale),
                "vertical_scale_px": float(vscale),
                "raw_valid_samples": int(row.raw_valid_samples),
                "ambiguity_proxy": float(row.ambiguity_proxy),
            }
        )

    if not trajectories:
        raise RuntimeError(f"No canonical trajectories for {spec.study}.")
    metadata = pd.DataFrame.from_records(rows)
    data = TrajectoryData(
        trajectories=np.stack(trajectories),
        metadata=metadata,
        time=np.linspace(0.0, 1.0, n_time),
    )
    report = {
        "analysis_pool_rows_after_subject_trial_and_extra_exclusions": int(len(analysis_pool)),
        "incorrect_analysis_pool_rows_excluded": int((~analysis_pool["correct"]).sum()),
        "canonical_rows": int(len(metadata)),
        "canonical_subjects": int(metadata["subject"].nunique()),
        "canonical_stimuli": int(metadata["stimulus_id"].nunique()),
        "canonical_conditions": metadata["condition"].value_counts().sort_index().to_dict(),
    }
    return data, metadata, report


def analyze_study(
    data: TrajectoryData,
    action_grid: Any,
    spec: StudySpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metadata = data.metadata.reset_index(drop=True).copy()
    rmse_matrix = trajectory_rmse_matrix(data.trajectories, action_grid.target_paths)
    best_idx = np.argmin(rmse_matrix, axis=1)
    fit_df = metadata[["source_row", "study", "subject", "condition", "stimulus_id"]].copy()
    fit_df["rho_hat"] = action_grid.rhos[best_idx]
    fit_df["rmse"] = rmse_matrix[np.arange(len(best_idx)), best_idx]
    fit_df["action_gap"] = action_grid.competitor_actions[best_idx] - action_grid.target_actions[best_idx]

    stim = (
        metadata.merge(fit_df[["source_row", "rho_hat", "action_gap", "rmse"]], on="source_row")
        .groupby(["study", "stimulus_id", "stimulus", "condition", "ambiguity_proxy"], as_index=False)
        .agg(
            mean_rho=("rho_hat", "mean"),
            median_rho=("rho_hat", "median"),
            mean_action_gap=("action_gap", "mean"),
            n_trials=("rho_hat", "size"),
            n_subjects=("subject", "nunique"),
            mean_rmse=("rmse", "mean"),
        )
    )
    condition_tests = condition_difference_tests(
        metadata.merge(fit_df[["source_row", "rho_hat"]], on="source_row"),
        stim,
        spec,
    )
    nll = heldout_nll_comparison(data, action_grid)
    controls = external_controls(stim, metadata, data, action_grid, spec)
    return fit_df, stim, condition_tests, nll, controls


def condition_difference_tests(trials: pd.DataFrame, stim: pd.DataFrame, spec: StudySpec) -> pd.DataFrame:
    ordinary, lookalike = spec.primary_pair
    stim_pair = stim[stim["condition"].isin([ordinary, lookalike])].copy()
    obs = mean_difference(stim_pair, lookalike, ordinary)
    rng = np.random.default_rng(RNG_SEED)
    sub_boot = clustered_bootstrap_difference(trials, lookalike, ordinary, "subject", rng)
    stim_boot = clustered_bootstrap_difference(trials, lookalike, ordinary, "stimulus_id", rng)
    rho_s, p_s = stats.spearmanr(stim_pair["ambiguity_proxy"], stim_pair["mean_rho"])
    try:
        fit = smf.ols("mean_rho ~ ambiguity_proxy", data=stim_pair).fit(cov_type="HC1")
        slope = float(fit.params.get("ambiguity_proxy", math.nan))
        p_ols = float(fit.pvalues.get("ambiguity_proxy", math.nan))
    except Exception:
        slope, p_ols = math.nan, math.nan
    return pd.DataFrame(
        [
            {
                "study": spec.study,
                "test": "lookalike_vs_ordinary_object_stimulus_mean_rho",
                "ordinary_condition": ordinary,
                "lookalike_condition": lookalike,
                "ordinary_mean_rho": float(stim_pair.loc[stim_pair["condition"].eq(ordinary), "mean_rho"].mean()),
                "lookalike_mean_rho": float(stim_pair.loc[stim_pair["condition"].eq(lookalike), "mean_rho"].mean()),
                "difference_lookalike_minus_ordinary": obs,
                "participant_cluster_ci_low": float(np.nanpercentile(sub_boot, 2.5)),
                "participant_cluster_ci_high": float(np.nanpercentile(sub_boot, 97.5)),
                "participant_cluster_p_two_sided": bootstrap_p(sub_boot),
                "stimulus_cluster_ci_low": float(np.nanpercentile(stim_boot, 2.5)),
                "stimulus_cluster_ci_high": float(np.nanpercentile(stim_boot, 97.5)),
                "stimulus_cluster_p_two_sided": bootstrap_p(stim_boot),
                "spearman_ambiguity_proxy_rho": float(rho_s),
                "spearman_ambiguity_proxy_p": float(p_s),
                "ols_ambiguity_proxy_slope": slope,
                "ols_ambiguity_proxy_p_hc1": p_ols,
                "n_stimuli": int(stim_pair["stimulus_id"].nunique()),
            }
        ]
    )


def heldout_nll_comparison(data: TrajectoryData, action_grid: Any) -> pd.DataFrame:
    meta = data.metadata.reset_index(drop=True).copy()
    n_time = data.trajectories.shape[1]
    action_rmse = trajectory_rmse_matrix(data.trajectories, action_grid.target_paths)
    motor = minimum_jerk_path(n_time)
    motor_rmse = path_rmse_vector(data.trajectories, motor)
    strength_grid = np.round(np.arange(0.0, 1.5001, 0.05), 10)
    bezier_paths = bezier_attraction_paths(strength_grid, n_time)
    spline_paths = spline_attraction_paths(strength_grid, n_time)
    bezier_rmse = trajectory_rmse_matrix(data.trajectories, bezier_paths)
    spline_rmse = trajectory_rmse_matrix(data.trajectories, spline_paths)

    rows: list[dict[str, Any]] = []
    groups = meta["stimulus_id"].to_numpy()
    for fold, stim in enumerate(sorted(meta["stimulus_id"].unique()), start=1):
        test_idx = np.where(groups == stim)[0]
        train_idx = np.where(groups != stim)[0]
        append_fixed_nll(rows, "minimum_jerk", fold, meta, test_idx, motor_rmse, fit_tau2(motor_rmse[train_idx]), n_time)
        condition_mean_fold(rows, fold, meta, data.trajectories, train_idx, test_idx, n_time)
        grid_condition_fold(rows, "bezier_condition", fold, meta, train_idx, test_idx, bezier_rmse, strength_grid, n_time)
        grid_condition_fold(rows, "spline_condition", fold, meta, train_idx, test_idx, spline_rmse, strength_grid, n_time)
        action_condition_fold(rows, "condition_only_action_rho", fold, meta, train_idx, test_idx, action_rmse, action_grid.rhos, n_time)
        action_ambiguity_fold(rows, "binary_ambiguity_proxy_action_rho", fold, meta, train_idx, test_idx, action_rmse, action_grid.rhos, n_time)
        trial_fitted_fold(rows, "trial_fitted_action_rho_upper_bound", fold, meta, train_idx, test_idx, action_rmse, action_grid.rhos, n_time)
    trials = pd.DataFrame(rows)
    summary = (
        trials.groupby("model", as_index=False)
        .agg(mean_nll=("nll", "mean"), median_nll=("nll", "median"), mean_rmse=("rmse", "mean"), n_trials=("nll", "size"))
        .sort_values("mean_nll")
    )
    return summary


def append_fixed_nll(
    rows: list[dict[str, Any]],
    model: str,
    fold: int,
    meta: pd.DataFrame,
    test_idx: np.ndarray,
    rmse: np.ndarray,
    tau2: float,
    n_time: int,
) -> None:
    for idx in test_idx:
        rows.append(nll_row(model, fold, meta, idx, float(rmse[idx]), tau2, n_time))


def condition_mean_fold(
    rows: list[dict[str, Any]],
    fold: int,
    meta: pd.DataFrame,
    trajectories: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_time: int,
) -> None:
    means = {}
    train_rmse: list[float] = []
    for condition in meta.loc[train_idx, "condition"].unique():
        idx = train_idx[meta.loc[train_idx, "condition"].to_numpy() == condition]
        means[condition] = trajectories[idx].mean(axis=0)
    for idx in train_idx:
        condition = meta.loc[idx, "condition"]
        train_rmse.append(path_rmse_vector(trajectories[[idx]], means[condition])[0])
    tau2 = fit_tau2(np.asarray(train_rmse))
    for idx in test_idx:
        condition = meta.loc[idx, "condition"]
        if condition not in means:
            continue
        rmse = path_rmse_vector(trajectories[[idx]], means[condition])[0]
        rows.append(nll_row("condition_mean_trajectory", fold, meta, idx, float(rmse), tau2, n_time))


def grid_condition_fold(
    rows: list[dict[str, Any]],
    model: str,
    fold: int,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    rmse_matrix: np.ndarray,
    grid: np.ndarray,
    n_time: int,
) -> None:
    best_by_cond = {}
    for condition in meta.loc[train_idx, "condition"].unique():
        idx = train_idx[meta.loc[train_idx, "condition"].to_numpy() == condition]
        best_by_cond[condition] = int(np.argmin(rmse_matrix[idx].mean(axis=0)))
    train_rmse = [rmse_matrix[idx, best_by_cond[meta.loc[idx, "condition"]]] for idx in train_idx]
    tau2 = fit_tau2(np.asarray(train_rmse))
    for idx in test_idx:
        condition = meta.loc[idx, "condition"]
        if condition not in best_by_cond:
            continue
        best = best_by_cond[condition]
        rows.append(nll_row(model, fold, meta, idx, float(rmse_matrix[idx, best]), tau2, n_time, parameter=float(grid[best])))


def action_condition_fold(
    rows: list[dict[str, Any]],
    model: str,
    fold: int,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    rmse_matrix: np.ndarray,
    rhos: np.ndarray,
    n_time: int,
) -> None:
    best_by_cond = {}
    for condition in meta.loc[train_idx, "condition"].unique():
        idx = train_idx[meta.loc[train_idx, "condition"].to_numpy() == condition]
        best_by_cond[condition] = int(np.argmin(rmse_matrix[idx].mean(axis=0)))
    train_rmse = [rmse_matrix[idx, best_by_cond[meta.loc[idx, "condition"]]] for idx in train_idx]
    tau2 = fit_tau2(np.asarray(train_rmse))
    for idx in test_idx:
        condition = meta.loc[idx, "condition"]
        if condition not in best_by_cond:
            continue
        best = best_by_cond[condition]
        rows.append(nll_row(model, fold, meta, idx, float(rmse_matrix[idx, best]), tau2, n_time, rho=float(rhos[best])))


def action_ambiguity_fold(
    rows: list[dict[str, Any]],
    model: str,
    fold: int,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    rmse_matrix: np.ndarray,
    rhos: np.ndarray,
    n_time: int,
) -> None:
    train = meta.loc[train_idx].copy()
    train["trial_fit_rho"] = rhos[np.argmin(rmse_matrix[train_idx], axis=1)]
    try:
        fit = smf.ols("trial_fit_rho ~ ambiguity_proxy", data=train).fit()
    except Exception:
        return
    train_pred = fit.predict(train).to_numpy()
    train_best = np.array([nearest_index(v, rhos) for v in train_pred])
    tau2 = fit_tau2(rmse_matrix[train_idx, train_best])
    test = meta.loc[test_idx].copy()
    pred = fit.predict(test).to_numpy()
    for idx, rho_pred in zip(test_idx, pred):
        best = nearest_index(rho_pred, rhos)
        rows.append(nll_row(model, fold, meta, idx, float(rmse_matrix[idx, best]), tau2, n_time, rho=float(rhos[best])))


def trial_fitted_fold(
    rows: list[dict[str, Any]],
    model: str,
    fold: int,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    rmse_matrix: np.ndarray,
    rhos: np.ndarray,
    n_time: int,
) -> None:
    train_best = np.min(rmse_matrix[train_idx], axis=1)
    tau2 = fit_tau2(train_best)
    for idx in test_idx:
        best = int(np.argmin(rmse_matrix[idx]))
        rows.append(nll_row(model, fold, meta, idx, float(rmse_matrix[idx, best]), tau2, n_time, rho=float(rhos[best])))


def nll_row(
    model: str,
    fold: int,
    meta: pd.DataFrame,
    idx: int,
    rmse: float,
    tau2: float,
    n_time: int,
    **extra: Any,
) -> dict[str, Any]:
    row = {
        "model": model,
        "fold": fold,
        "study": meta.loc[idx, "study"],
        "source_row": int(meta.loc[idx, "source_row"]),
        "subject": int(meta.loc[idx, "subject"]),
        "stimulus_id": meta.loc[idx, "stimulus_id"],
        "condition": meta.loc[idx, "condition"],
        "rmse": rmse,
        "tau2": tau2,
        "nll": -gaussian_path_loglik(rmse, tau2, n_time),
    }
    row.update(extra)
    return row


def external_controls(
    stim: pd.DataFrame,
    metadata: pd.DataFrame,
    data: TrajectoryData,
    action_grid: Any,
    spec: StudySpec,
) -> dict[str, Any]:
    ordinary, lookalike = spec.primary_pair
    pair = stim[stim["condition"].isin([ordinary, lookalike])].copy()
    observed = mean_difference(pair, lookalike, ordinary)
    rng = np.random.default_rng(RNG_SEED + (1 if spec.study == "animal_object" else 0))
    perm_diffs = []
    values = pair["ambiguity_proxy"].to_numpy()
    for _ in range(N_PERM):
        shuffled = pair.copy()
        shuffled["ambiguity_proxy"] = rng.permutation(values)
        high = shuffled.loc[shuffled["ambiguity_proxy"].eq(1.0), "mean_rho"].mean()
        low = shuffled.loc[shuffled["ambiguity_proxy"].eq(0.0), "mean_rho"].mean()
        perm_diffs.append(float(high - low))
    perm = np.asarray(perm_diffs)

    null_rates = null_simulation_false_positive_rates(
        metadata,
        data,
        action_grid,
        spec,
        observed,
        rng,
    )
    return {
        "observed_difference": float(observed),
        "permutation_p_ge_observed": float((1 + np.sum(perm >= observed)) / (len(perm) + 1)),
        "permutation_null_mean": float(np.mean(perm)),
        "permutation_null_sd": float(np.std(perm, ddof=1)),
        "permutations": int(N_PERM),
        **null_rates,
    }


def null_simulation_false_positive_rates(
    metadata: pd.DataFrame,
    data: TrajectoryData,
    action_grid: Any,
    spec: StudySpec,
    observed: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    ordinary, lookalike = spec.primary_pair
    meta = metadata.reset_index(drop=True)
    pair_mask = meta["condition"].isin([ordinary, lookalike]).to_numpy()
    pooled_pair_mean = data.trajectories[pair_mask].mean(axis=0)
    pair_resid = data.trajectories[pair_mask] - pooled_pair_mean
    sigma = float(np.nanstd(pair_resid))
    n_pair = int(pair_mask.sum())
    pair_meta = meta.loc[pair_mask, ["stimulus_id", "condition"]].reset_index(drop=True)
    exceed = 0
    for _ in range(N_NULL):
        synthetic = pooled_pair_mean[None, :, :] + rng.normal(0.0, sigma, size=(n_pair, *pooled_pair_mean.shape))
        rmse = trajectory_rmse_matrix(synthetic, action_grid.target_paths)
        rho = action_grid.rhos[np.argmin(rmse, axis=1)]
        tmp = pair_meta.copy()
        tmp["rho_hat"] = rho
        stim = tmp.groupby(["stimulus_id", "condition"], as_index=False).agg(mean_rho=("rho_hat", "mean"))
        diff = mean_difference(stim, lookalike, ordinary)
        exceed += int(diff >= observed)
    return {
        "pooled_object_null_false_positive_rate_ge_observed": float(exceed / N_NULL),
        "pooled_object_null_simulations": int(N_NULL),
    }


def build_validation_table(stimulus: pd.DataFrame, diagnostics: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = {
        "face_object": ("Face/Object", "Face-like objects vs ordinary objects"),
        "animal_object": ("Animal/Object", "Animal-like objects vs ordinary objects"),
    }
    for study, (label, test_label) in labels.items():
        cond = diagnostics[study]["condition_tests"][0]
        controls = diagnostics[study]["controls"]
        nll = pd.DataFrame(diagnostics[study]["heldout_nll"])
        action_cond = get_model_value(nll, "condition_only_action_rho", "mean_nll")
        action_amb = get_model_value(nll, "binary_ambiguity_proxy_action_rho", "mean_nll")
        best_model = str(nll.sort_values("mean_nll").iloc[0]["model"])
        rows.append(
            {
                "study": label,
                "test": test_label,
                "ordinary_mean_rho": cond["ordinary_mean_rho"],
                "lookalike_mean_rho": cond["lookalike_mean_rho"],
                "rho_difference": cond["difference_lookalike_minus_ordinary"],
                "stimulus_cluster_ci": f"[{cond['stimulus_cluster_ci_low']:.3f}, {cond['stimulus_cluster_ci_high']:.3f}]",
                "stimulus_cluster_p": cond["stimulus_cluster_p_two_sided"],
                "ambiguity_proxy_spearman": cond["spearman_ambiguity_proxy_rho"],
                "permutation_p": controls["permutation_p_ge_observed"],
                "null_fp_rate": controls["pooled_object_null_false_positive_rate_ge_observed"],
                "condition_only_action_mean_nll": action_cond,
                "ambiguity_proxy_action_mean_nll": action_amb,
                "best_raw_nll_model": best_model,
            }
        )
    return pd.DataFrame(rows)


def write_preprocessing_report(preprocessing: list[dict[str, Any]], diagnostics: dict[str, Any]) -> None:
    lines = [
        "# Koenig-Robert et al. 2024 External Mouse-Tracking Validation",
        "",
        f"Download date: {DOWNLOAD_DATE}",
        f"OSF project: {OSF_PROJECT_URL}",
        f"OSF API listing: {OSF_API_URL}",
        f"Article: {ARTICLE_URL}",
        "",
        "## Downloaded Files",
        "",
        "| File | URL | Bytes | SHA-256 | Expected SHA-256 | Match |",
        "|---|---:|---:|---|---|---:|",
    ]
    for name, meta in FILES.items():
        path = RAW / name
        actual = sha256(path) if path.exists() else "MISSING"
        size = path.stat().st_size if path.exists() else 0
        lines.append(
            f"| {name} | {meta['url']} | {size} | {actual} | {meta['sha256']} | {actual == meta['sha256']} |"
        )
    lines.extend(
        [
            "",
            "## Data Dictionary Used",
            "",
            "The raw Pavlovia/jsPsych CSV files were read from the downloaded zips. Trial rows were selected where `trial_type == mousetracking`. The raw headers are `stimulus`, `x-position`, `y-position`, `mice-times`, and `nRecordings`; the MATLAB preprocessing script renames these as `stimulus`, `x_position`, `y_position`, `mice_times`, and `nRecordings`.",
            "",
            "Participant metadata were read from the `survey-html-form` response JSON. The response-box swap indicator follows the authors' MATLAB script: face study `swaporder = contains(responses, 'cursor')`; animal study `swaporder = contains(responses, 'cursor1')` after normalizing pointer labels.",
            "",
            "Prespecified preprocessing mirrored the public MATLAB script where possible: only complete participant files were used, x positions were centered at 500 pixels, y positions were transformed as `600 - y`, response sides were flipped after block 2 according to the counterbalancing field, subjects were included when mean valid-trial rate and mean accuracy both exceeded .70, extra balancing stimuli were excluded, and only correct trials with canonical endpoint movement of at least 50 px horizontally and vertically were fitted.",
            "",
            "No scalar independent face-likeness or animal-likeness ratings were distributed in the downloaded OSF mouse-tracking archive. The validation therefore treats lookalike-object condition as a prespecified binary ambiguity proxy and reports this limitation explicitly.",
            "",
            "## Preprocessing Counts",
            "",
            "| Study | Raw mouse rows | Full rows | Complete files | Included participants | Analysis pool | Incorrect excluded | Canonical N | Canonical subjects | Canonical stimuli | Conditions |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in preprocessing:
        conditions = "; ".join(f"{k}={v}" for k, v in row["canonical_conditions"].items())
        lines.append(
            f"| {row['study']} | {row['raw_mouse_rows_in_all_csvs']} | {row['full_dataset_rows']} | {row['complete_participant_files']} | {row['participants_after_quality_filter']} | {row['analysis_pool_rows_after_subject_trial_and_extra_exclusions']} | {row['incorrect_analysis_pool_rows_excluded']} | {row['canonical_rows']} | {row['canonical_subjects']} | {row['canonical_stimuli']} | {conditions} |"
        )
    lines.extend(["", "## Model and Control Summary", ""])
    for study, info in diagnostics.items():
        cond = info["condition_tests"][0]
        ctrl = info["controls"]
        nll = pd.DataFrame(info["heldout_nll"]).sort_values("mean_nll")
        lines.extend(
            [
                f"### {study}",
                "",
                f"- Lookalike-minus-ordinary stimulus mean rho difference: {cond['difference_lookalike_minus_ordinary']:.4f}.",
                f"- Stimulus-cluster bootstrap CI: [{cond['stimulus_cluster_ci_low']:.4f}, {cond['stimulus_cluster_ci_high']:.4f}], p={cond['stimulus_cluster_p_two_sided']:.4f}.",
                f"- Permutation p for ambiguity labels: {ctrl['permutation_p_ge_observed']:.4f}; pooled-object null false-positive rate >= observed: {ctrl['pooled_object_null_false_positive_rate_ge_observed']:.4f}.",
                f"- Best raw held-out NLL model: {nll.iloc[0]['model']} (mean NLL={nll.iloc[0]['mean_nll']:.3f}).",
                "",
            ]
        )
    (RESULTS / "external_koenig_robert_preprocessing_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(df: pd.DataFrame, path: Path) -> None:
    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            " & ".join(
                [
                    latex_escape(row.study),
                    latex_escape(row.test),
                    f"{row.ordinary_mean_rho:.3f}",
                    f"{row.lookalike_mean_rho:.3f}",
                    f"{row.rho_difference:.3f}",
                    latex_escape(row.stimulus_cluster_ci),
                    format_p(row.stimulus_cluster_p),
                    format_p(row.permutation_p),
                    f"{row.null_fp_rate:.3f}",
                    latex_escape(row.best_raw_nll_model.replace('_', ' ')),
                ]
            )
            + r" \\"
        )
    text = "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{External Koenig-Robert mouse-tracking validation. Lookalike-object stimuli are tested against ordinary objects. The raw-NLL column is descriptive and is not used to claim mechanistic superiority.}",
            r"\label{tab:external-mouse-tracking-validation}",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{llrrrrrrrl}",
            r"\toprule",
            r"Study & Test & Ordinary $\rho$ & Lookalike $\rho$ & $\Delta\rho$ & Stimulus CI & $p_{\mathrm{cluster}}$ & $p_{\mathrm{perm}}$ & Null FP & Best raw NLL model \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"}%",
            r"\end{table}",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def plot_rho_by_condition(stim: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8.4, 4.8))
    orders = {
        "face_object": ["face", "ordinary_object", "face_like_object"],
        "animal_object": ["animal", "ordinary_object", "animal_like_object"],
    }
    label_map = {
        ("face_object", "face"): "true face",
        ("face_object", "ordinary_object"): "face task\nordinary object",
        ("face_object", "face_like_object"): "face-like\nobject",
        ("animal_object", "animal"): "true animal",
        ("animal_object", "ordinary_object"): "animal task\nordinary object",
        ("animal_object", "animal_like_object"): "animal-like\nobject",
    }
    colors = {
        "face_object": "#3b6ea8",
        "animal_object": "#b45f4d",
    }
    xpos = 0
    labels = []
    for study in ["face_object", "animal_object"]:
        sub = stim[stim["study"].eq(study)]
        for cond in orders[study]:
            vals = sub.loc[sub["condition"].eq(cond), "mean_rho"].dropna()
            if vals.empty:
                continue
            jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
            plt.scatter(np.full(len(vals), xpos) + jitter, vals, s=26, alpha=0.65, color=colors[study])
            mean = vals.mean()
            ci = bootstrap_ci(vals.to_numpy())
            plt.errorbar([xpos], [mean], yerr=[[mean - ci[0]], [ci[1] - mean]], color="black", capsize=4, marker="o")
            labels.append(label_map.get((study, cond), cond.replace("_", " ")))
            xpos += 1
        xpos += 0.7
    plt.ylabel(r"Stimulus mean fitted $\rho$")
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.xlim(-0.7, xpos - 0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_ambiguity_rho(stim: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(6.8, 4.8))
    labels = {"face_object": "Face/Object", "animal_object": "Animal/Object"}
    colors = {"face_object": "#3b6ea8", "animal_object": "#b45f4d"}
    for study in ["face_object", "animal_object"]:
        sub = stim[
            stim["study"].eq(study)
            & stim["condition"].isin(["ordinary_object", "face_like_object", "animal_like_object"])
        ].copy()
        x = sub["ambiguity_proxy"].to_numpy()
        jitter = (np.arange(len(sub)) % 7 - 3) * 0.012
        plt.scatter(x + jitter, sub["mean_rho"], label=labels[study], color=colors[study], alpha=0.75, s=36)
        if sub["ambiguity_proxy"].nunique() == 2:
            means = sub.groupby("ambiguity_proxy")["mean_rho"].mean()
            plt.plot([0, 1], [means.loc[0.0], means.loc[1.0]], color=colors[study], linewidth=2)
    plt.xticks([0, 1], ["ordinary object", "lookalike object"])
    plt.ylabel(r"Stimulus mean fitted $\rho$")
    plt.xlabel("Binary ambiguity proxy")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def parse_series(text: str) -> np.ndarray:
    if text is None:
        return np.array([], dtype=float)
    out: list[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if part == "" or part.lower() in {"null", "nan", "none"}:
            out.append(math.nan)
        else:
            try:
                out.append(float(part))
            except ValueError:
                out.append(math.nan)
    return np.asarray(out, dtype=float)


def as_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def first_execution_time(x: np.ndarray, y: np.ndarray) -> float:
    reached = np.where((np.abs(x) > 270.0) & (np.abs(y) > 380.0))[0]
    if len(reached) == 0:
        return math.nan
    return float(np.linspace(0.0, 800.0, len(x))[reached[0]])


def fit_tau2(rmse: np.ndarray) -> float:
    rmse = np.asarray(rmse, dtype=float)
    rmse = rmse[np.isfinite(rmse)]
    if rmse.size == 0:
        return 1e-8
    return float(max(np.mean(rmse**2) / 2.0, 1e-8))


def nearest_index(value: float, grid: np.ndarray) -> int:
    return int(np.argmin(np.abs(grid - value)))


def mean_difference(df: pd.DataFrame, high_condition: str, low_condition: str) -> float:
    high = df.loc[df["condition"].eq(high_condition), "mean_rho" if "mean_rho" in df.columns else "rho_hat"].mean()
    low = df.loc[df["condition"].eq(low_condition), "mean_rho" if "mean_rho" in df.columns else "rho_hat"].mean()
    return float(high - low)


def clustered_bootstrap_difference(
    trials: pd.DataFrame,
    high_condition: str,
    low_condition: str,
    cluster_col: str,
    rng: np.random.Generator,
) -> np.ndarray:
    grouped = (
        trials[trials["condition"].isin([high_condition, low_condition])]
        .groupby([cluster_col, "condition"], as_index=False)["rho_hat"]
        .mean()
    )
    out = []
    if cluster_col == "subject":
        piv = grouped.pivot(index=cluster_col, columns="condition", values="rho_hat").dropna(
            subset=[high_condition, low_condition]
        )
        high = piv[high_condition].to_numpy(dtype=float)
        low = piv[low_condition].to_numpy(dtype=float)
        n = len(piv)
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, size=n)
            out.append(float(high[idx].mean() - low[idx].mean()))
    else:
        high = grouped.loc[grouped["condition"].eq(high_condition), "rho_hat"].to_numpy(dtype=float)
        low = grouped.loc[grouped["condition"].eq(low_condition), "rho_hat"].to_numpy(dtype=float)
        for _ in range(N_BOOT):
            out.append(
                float(
                    rng.choice(high, size=len(high), replace=True).mean()
                    - rng.choice(low, size=len(low), replace=True).mean()
                )
            )
    return np.asarray(out, dtype=float)


def bootstrap_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    return float(2.0 * min(np.mean(values <= 0.0), np.mean(values >= 0.0)))


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED)
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return float(values[0]), float(values[0])
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(N_BOOT)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_model_value(df: pd.DataFrame, model: str, column: str) -> float:
    hit = df.loc[df["model"].eq(model), column]
    if hit.empty:
        return math.nan
    return float(hit.iloc[0])


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return re.sub(r"[\\&%$#_{}]", lambda m: replacements[m.group(0)], text)


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if value < 0.001:
        return "$<.001$"
    return f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
