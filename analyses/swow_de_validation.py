from __future__ import annotations

import collections
import hashlib
import math
import sys
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import TheilSenRegressor
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
RAW = ROOT / "data" / "external" / "swow_de_2025" / "raw"

DOWNLOAD_DATE = "2026-06-18"
SWOW_URL = "https://smallworldofwords.org/en/project/research"
SWOW_ZIP = RAW / "SWOW-DE_2025.zip"
SWOW_EMB_ZIP = RAW / "SWOW-DE_2025_embeddings.zip"
RNG_SEED = 20260618
N_BOOT = 4000
N_PERM = 5000

ORTHOGRAPHIC_MAP = {
    "Saeugetier": "Säugetier",
    "Chamaeleon": "Chamäleon",
    "Seeloewe": "Seelöwe",
    "Loewe": "Löwe",
}


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    item_summary = pd.read_csv(ROOT / "outputs" / "item_level_action_summary.csv")
    items = build_mapping_table(item_summary)

    swow = load_swow_r55()
    assoc = association_tables(swow)
    graph = build_graph(swow)
    emb = load_embeddings()

    metrics = compute_item_metrics(items, assoc, graph, emb)
    metric_choice = choose_main_metric(metrics)
    results = validation_statistics(metrics, metric_choice)
    mapping = build_output_mapping(metrics, metric_choice)

    mapping.to_csv(RESULTS / "swow_de_mapping_table.csv", index=False)
    results.to_csv(RESULTS / "swow_de_margin_validation.csv", index=False)
    write_sha256(RESULTS / "swow_de_margin_validation.csv", RESULTS / "swow_de_margin_validation.sha256")
    write_latex_table(results, metric_choice, TABLES / "table_swow_margin_validation.tex")
    plot_main_metric(metrics, metric_choice, FIGURES / "swow_margin_rho.png")
    print(f"Main SWOW-DE metric: {metric_choice}")
    print(f"Wrote {RESULTS / 'swow_de_margin_validation.csv'}")
    return 0


def build_mapping_table(item_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in item_summary.itertuples(index=False):
        rows.append(
            {
                "item_original": row.exemplar,
                "item_swow": canonical(row.exemplar),
                "target_category_original": row.category_correct_x,
                "target_category_swow": canonical(row.category_correct_x),
                "competitor_category_original": row.competitor_category_x,
                "competitor_category_swow": canonical(row.competitor_category_x),
                "condition": row.condition,
                "atypical": int(row.condition == "Atypical"),
                "rho_hat": float(row.rho_hat),
                "primary_semantic_margin": float(row.semantic_margin),
            }
        )
    out = pd.DataFrame(rows)
    external = ROOT / "outputs" / "semantic_sources" / "multisource_rankmean_scores.csv"
    if external.exists():
        emb = pd.read_csv(external)[["item", "rankmean_margin"]].rename(columns={"item": "item_original"})
        out = out.merge(emb, on="item_original", how="left")
    else:
        out["rankmean_margin"] = np.nan
    return out


def canonical(value: Any) -> str:
    text = str(value)
    return ORTHOGRAPHIC_MAP.get(text, text)


def norm(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return canonical(value).strip().casefold()


def load_swow_r55() -> pd.DataFrame:
    if not SWOW_ZIP.exists():
        raise FileNotFoundError(f"Missing {SWOW_ZIP}. Download from {SWOW_URL}.")
    usecols = ["cue", "response_corrected_1", "response_corrected_2", "response_corrected_3"]
    with zipfile.ZipFile(SWOW_ZIP) as zf:
        with zf.open("SWOW-DE_2025_R55.csv") as f:
            return pd.read_csv(f, usecols=usecols)


def association_tables(swow: pd.DataFrame) -> dict[str, dict[tuple[str, str], float]]:
    out: dict[str, dict[tuple[str, str], float]] = {}
    cues = swow["cue"].map(norm)
    r1 = swow["response_corrected_1"].map(norm)
    valid_r1 = r1.ne("")
    denom_r1 = swow.loc[valid_r1].assign(_cue=cues[valid_r1]).groupby("_cue").size()
    pair_r1 = (
        pd.DataFrame({"cue": cues[valid_r1], "response": r1[valid_r1]})
        .groupby(["cue", "response"])
        .size()
    )
    out["r1"] = {(cue, resp): float(count / denom_r1.loc[cue]) for (cue, resp), count in pair_r1.items()}

    long_parts = []
    for col in ["response_corrected_1", "response_corrected_2", "response_corrected_3"]:
        tmp = pd.DataFrame({"cue": cues, "response": swow[col].map(norm)})
        tmp = tmp[tmp["response"].ne("")]
        long_parts.append(tmp)
    long = pd.concat(long_parts, ignore_index=True)
    denom = long.groupby("cue").size()
    pair = long.groupby(["cue", "response"]).size()
    out["r123"] = {(cue, resp): float(count / denom.loc[cue]) for (cue, resp), count in pair.items()}
    return out


def build_graph(swow: pd.DataFrame) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = collections.defaultdict(set)
    cues = swow["cue"].map(norm)
    for col in ["response_corrected_1", "response_corrected_2", "response_corrected_3"]:
        responses = swow[col].map(norm)
        for cue, response in zip(cues, responses):
            if cue and response:
                adj[cue].add(response)
                adj[response].add(cue)
    return adj


def load_embeddings() -> dict[str, np.ndarray]:
    if not SWOW_EMB_ZIP.exists():
        return {}
    with zipfile.ZipFile(SWOW_EMB_ZIP) as zf:
        with zf.open("SWOW-DE_2025_embeddings/swow_ppmi_svd.csv") as f:
            df = pd.read_csv(f)
    cue = df.pop("cue").map(norm)
    values = df.to_numpy(dtype=float)
    norms = np.linalg.norm(values, axis=1)
    values = values / np.maximum(norms[:, None], 1e-12)
    return {c: values[i] for i, c in enumerate(cue)}


def compute_item_metrics(
    items: pd.DataFrame,
    assoc: dict[str, dict[tuple[str, str], float]],
    graph: dict[str, set[str]],
    emb: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for row in items.itertuples(index=False):
        item = norm(row.item_swow)
        target = norm(row.target_category_swow)
        competitor = norm(row.competitor_category_swow)
        direct_t = assoc["r123"].get((item, target), 0.0)
        direct_c = assoc["r123"].get((item, competitor), 0.0)
        reverse_t = assoc["r123"].get((target, item), 0.0)
        reverse_c = assoc["r123"].get((competitor, item), 0.0)
        direct_r1_t = assoc["r1"].get((item, target), 0.0)
        direct_r1_c = assoc["r1"].get((item, competitor), 0.0)
        reverse_r1_t = assoc["r1"].get((target, item), 0.0)
        reverse_r1_c = assoc["r1"].get((competitor, item), 0.0)
        item_is_cue = cue_exists(assoc["r123"], item)
        target_is_cue = cue_exists(assoc["r123"], target)
        competitor_is_cue = cue_exists(assoc["r123"], competitor)

        dt = shortest_path(graph, item, target)
        dc = shortest_path(graph, item, competitor)
        network_t = proximity(dt)
        network_c = proximity(dc)
        emb_t = cosine(emb, item, target)
        emb_c = cosine(emb, item, competitor)

        rows.append(
            {
                **row._asdict(),
                "item_norm": item,
                "target_norm": target,
                "competitor_norm": competitor,
                "item_as_cue": item_is_cue,
                "target_as_cue": target_is_cue,
                "competitor_as_cue": competitor_is_cue,
                "direct_r123_target": direct_t,
                "direct_r123_competitor": direct_c,
                "direct_r123_margin": direct_t - direct_c if item_is_cue else np.nan,
                "reverse_r123_target": reverse_t,
                "reverse_r123_competitor": reverse_c,
                "reverse_r123_margin": reverse_t - reverse_c if target_is_cue and competitor_is_cue else np.nan,
                "symmetric_avg_margin": ((direct_t + reverse_t) / 2.0) - ((direct_c + reverse_c) / 2.0),
                "symmetric_max_margin": max(direct_t, reverse_t) - max(direct_c, reverse_c),
                "direct_r1_margin": direct_r1_t - direct_r1_c if item_is_cue else np.nan,
                "reverse_r1_margin": reverse_r1_t - reverse_r1_c if target_is_cue and competitor_is_cue else np.nan,
                "network_distance_target": dt,
                "network_distance_competitor": dc,
                "network_proximity_target": network_t,
                "network_proximity_competitor": network_c,
                "network_margin": network_t - network_c if np.isfinite(dt) and np.isfinite(dc) else np.nan,
                "embedding_cosine_target": emb_t,
                "embedding_cosine_competitor": emb_c,
                "embedding_margin": emb_t - emb_c if np.isfinite(emb_t) and np.isfinite(emb_c) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def cue_exists(assoc: dict[tuple[str, str], float], cue: str) -> bool:
    return any(key[0] == cue for key in assoc)


def shortest_path(graph: dict[str, set[str]], start: str, goal: str, max_depth: int = 6) -> float:
    if not start or not goal or start not in graph or goal not in graph:
        return math.inf
    if start == goal:
        return 0.0
    queue = collections.deque([(start, 0)])
    seen = {start}
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for nxt in graph[node]:
            if nxt == goal:
                return float(depth + 1)
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, depth + 1))
    return math.inf


def proximity(distance: float) -> float:
    if not np.isfinite(distance):
        return np.nan
    return 1.0 / (1.0 + float(distance))


def cosine(emb: dict[str, np.ndarray], a: str, b: str) -> float:
    if a not in emb or b not in emb:
        return math.nan
    return float(np.dot(emb[a], emb[b]))


def choose_main_metric(metrics: pd.DataFrame) -> str:
    direct_cov = int(metrics["direct_r123_margin"].notna().sum())
    network_cov = int(metrics["network_margin"].notna().sum())
    embedding_cov = int(metrics["embedding_margin"].notna().sum())
    if embedding_cov >= 16:
        return "embedding_margin"
    if network_cov >= 16:
        return "network_margin"
    if direct_cov >= 12:
        return "direct_r123_margin"
    return "symmetric_avg_margin"


def validation_statistics(metrics: pd.DataFrame, main_metric: str) -> pd.DataFrame:
    metric_specs = [
        ("direct_r123_margin", "Direct cue-to-response R123"),
        ("reverse_r123_margin", "Reverse category-to-item R123"),
        ("symmetric_avg_margin", "Symmetric average R123"),
        ("symmetric_max_margin", "Symmetric maximum R123"),
        ("direct_r1_margin", "Direct cue-to-response R1"),
        ("reverse_r1_margin", "Reverse category-to-item R1"),
        ("network_margin", "Undirected SWOW shortest-path proximity"),
        ("embedding_margin", "Official SWOW PPMI-SVD embedding cosine"),
        ("primary_semantic_margin", "Manuscript inverse-typicality margin"),
        ("rankmean_margin", "Existing multisource transformer rank margin"),
    ]
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for metric, label in metric_specs:
        if metric not in metrics.columns:
            continue
        sub = metrics[["item_swow", "rho_hat", "condition", "atypical", metric]].dropna()
        if len(sub) < 4:
            rows.append(empty_stat_row(metric, label, len(sub), metric == main_metric))
            continue
        x = sub[metric].to_numpy(dtype=float)
        y = sub["rho_hat"].to_numpy(dtype=float)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        pearson_r, pearson_p = stats.pearsonr(x, y)
        slope, ci_low, ci_high = theil_sen_bootstrap(x, y, rng)
        loo_min, loo_max = leave_one_out_slope_range(x, y)
        perm_p = permutation_p(x, y, rng)
        control_beta = math.nan
        control_p = math.nan
        if len(sub) >= 8:
            fit = smf.ols(f"rho_hat ~ Q('{metric}') + atypical", data=sub).fit(cov_type="HC1")
            control_beta = float(fit.params.get(f"Q('{metric}')", math.nan))
            control_p = float(fit.pvalues.get(f"Q('{metric}')", math.nan))
        rows.append(
            {
                "metric": metric,
                "label": label,
                "prespecified_main": metric == main_metric,
                "coverage": f"{len(sub)}/19",
                "n": int(len(sub)),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "theil_sen_slope": float(slope),
                "bootstrap_ci_low": float(ci_low),
                "bootstrap_ci_high": float(ci_high),
                "loo_slope_min": float(loo_min),
                "loo_slope_max": float(loo_max),
                "atypical_control_slope": control_beta,
                "atypical_control_p": control_p,
                "permutation_p_expected_negative": float(perm_p),
                "expected_direction": bool(spearman_r < 0),
            }
        )
    return pd.DataFrame(rows)


def empty_stat_row(metric: str, label: str, n: int, main: bool) -> dict[str, Any]:
    return {
        "metric": metric,
        "label": label,
        "prespecified_main": main,
        "coverage": f"{n}/19",
        "n": int(n),
        "spearman_r": np.nan,
        "spearman_p": np.nan,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "theil_sen_slope": np.nan,
        "bootstrap_ci_low": np.nan,
        "bootstrap_ci_high": np.nan,
        "loo_slope_min": np.nan,
        "loo_slope_max": np.nan,
        "atypical_control_slope": np.nan,
        "atypical_control_p": np.nan,
        "permutation_p_expected_negative": np.nan,
        "expected_direction": False,
    }


def theil_sen_bootstrap(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    slope = float(stats.theilslopes(y, x).slope)
    boot = []
    n = len(x)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        if np.unique(x[idx]).size < 2:
            continue
        boot.append(float(stats.theilslopes(y[idx], x[idx]).slope))
    if not boot:
        return slope, math.nan, math.nan
    return slope, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def leave_one_out_slope_range(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slopes = []
    for i in range(len(x)):
        xx = np.delete(x, i)
        yy = np.delete(y, i)
        if np.unique(xx).size < 2:
            continue
        slopes.append(float(stats.theilslopes(yy, xx).slope))
    if not slopes:
        return math.nan, math.nan
    return float(np.min(slopes)), float(np.max(slopes))


def permutation_p(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
    obs = stats.spearmanr(x, y).statistic
    null = []
    for _ in range(N_PERM):
        null.append(stats.spearmanr(rng.permutation(x), y).statistic)
    null = np.asarray(null)
    return float((1 + np.sum(null <= obs)) / (len(null) + 1))


def build_output_mapping(metrics: pd.DataFrame, main_metric: str) -> pd.DataFrame:
    out = metrics.copy()
    out["swow_release"] = "SWOW-DE 2025 R55"
    out["download_date"] = DOWNLOAD_DATE
    out["main_metric"] = main_metric
    cols = [
        "item_original",
        "item_swow",
        "target_category_original",
        "target_category_swow",
        "competitor_category_original",
        "competitor_category_swow",
        "condition",
        "rho_hat",
        "primary_semantic_margin",
        "rankmean_margin",
        "direct_r123_target",
        "direct_r123_competitor",
        "direct_r123_margin",
        "reverse_r123_target",
        "reverse_r123_competitor",
        "reverse_r123_margin",
        "network_distance_target",
        "network_distance_competitor",
        "network_margin",
        "embedding_margin",
        "main_metric",
        "swow_release",
        "download_date",
    ]
    return out[cols]


def write_latex_table(results: pd.DataFrame, main_metric: str, path: Path) -> None:
    display = results[
        results["metric"].isin(
            [
                "direct_r123_margin",
                "reverse_r123_margin",
                "symmetric_avg_margin",
                "network_margin",
                "embedding_margin",
                "primary_semantic_margin",
                "rankmean_margin",
            ]
        )
    ].copy()
    rows = []
    for row in display.itertuples(index=False):
        marker = r"$^\ast$" if row.metric == main_metric else ""
        rows.append(
            f"{latex_escape(row.label)}{marker} & {row.coverage} & {fmt(row.spearman_r)} & {format_p(row.spearman_p)} & {fmt(row.theil_sen_slope)} & [{fmt(row.bootstrap_ci_low)}, {fmt(row.bootstrap_ci_high)}] & {format_p(row.permutation_p_expected_negative)} \\\\"
        )
    text = "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{SWOW-DE margin validation for KH2017 items. The asterisk marks the prespecified main SWOW metric selected before inspecting the $\rho$ association: direct association if coverage is adequate, otherwise network or embedding proximity. Lower margins were expected to predict higher fitted $\rho$.}",
            r"\label{tab:swow-margin-validation}",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"Metric & Coverage & Spearman $r_s$ & $p$ & Theil--Sen slope & Bootstrap CI & Permutation $p$ \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"}%",
            r"\end{table}",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def plot_main_metric(metrics: pd.DataFrame, metric: str, path: Path) -> None:
    sub = metrics[["item_swow", "rho_hat", "condition", metric]].dropna()
    plt.figure(figsize=(6.4, 4.8))
    colors = {"Typical": "#3b6ea8", "Atypical": "#b45f4d"}
    for cond, grp in sub.groupby("condition"):
        plt.scatter(grp[metric], grp["rho_hat"], label=cond, color=colors.get(cond, "#555555"), s=54, alpha=0.8)
    if len(sub) >= 2:
        x = sub[metric].to_numpy()
        y = sub["rho_hat"].to_numpy()
        slope, intercept, _, _ = stats.theilslopes(y, x)
        grid = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        plt.plot(grid, intercept + slope * grid, color="black", linewidth=2)
    plt.xlabel(metric.replace("_", " "))
    plt.ylabel(r"Item mean fitted $\rho$")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def write_sha256(path: Path, hash_path: Path) -> None:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    hash_path.write_text(f"{h.hexdigest()}  {path.name}\n", encoding="utf-8")


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
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fmt(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.3f}"


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if value < 0.001:
        return "$<.001$"
    return f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
