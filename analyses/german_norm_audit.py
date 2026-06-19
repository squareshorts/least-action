from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
SOURCE = ROOT / "schroder_2012.xls"

ORTHOGRAPHIC_MAP = {
    "Saeugetier": "Säugetier",
    "Chamaeleon": "Chamäleon",
    "Seeloewe": "Seelöwe",
    "Loewe": "Löwe",
}


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    item_summary = pd.read_csv(ROOT / "outputs" / "item_level_action_summary.csv")
    norms = load_schroder()
    coverage = merge_items(item_summary, norms)
    coverage.to_csv(RESULTS / "german_norm_coverage.csv", index=False)
    summary = summarize_controls(coverage)
    write_latex(summary, TABLES / "table_german_norm_coverage.tex")
    print(f"Wrote {RESULTS / 'german_norm_coverage.csv'}")
    return 0


def load_schroder() -> pd.DataFrame:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing {SOURCE}.")
    df = pd.read_excel(SOURCE, sheet_name="all items_824", header=1)
    out = df.rename(
        columns={
            "GERMAN ": "german",
            "TRANSLATION (British English)": "english_translation",
            "semantic category": "schroder_semantic_category",
            "nb total (n= 20)": "generation_count",
            "% total": "generation_proportion",
            "M ": "typicality_m",
            "SD": "typicality_sd",
            "M .1": "aoa_m",
            "SD ": "aoa_sd",
            "M": "familiarity_m",
            "SD .1": "familiarity_sd",
            "normalized lemma frequency (per million)": "frequency_per_million",
            "normalized log10 lemma frequency": "log10_frequency",
            "nb PHONEMES": "n_phonemes",
            "nb SYLLABLES": "n_syllables",
        }
    )
    out["norm"] = out["german"].astype(str).str.strip().str.casefold()
    return out


def merge_items(item_summary: pd.DataFrame, norms: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by_norm = norms.drop_duplicates("norm").set_index("norm")
    for row in item_summary.itertuples(index=False):
        item = canonical(row.exemplar)
        hit = by_norm.loc[item.casefold()] if item.casefold() in by_norm.index else None
        base = {
            "item_original": row.exemplar,
            "item_german": item,
            "condition": row.condition,
            "target_category_original": row.category_correct_x,
            "target_category_german": canonical(row.category_correct_x),
            "competitor_category_original": row.competitor_category_x,
            "competitor_category_german": canonical(row.competitor_category_x),
            "rho_hat": float(row.rho_hat),
            "primary_semantic_margin": float(row.semantic_margin),
            "schroder_exact_item_match": hit is not None,
            "exact_target_competitor_typicality_available": False,
        }
        if hit is None:
            rows.append(base)
        else:
            for col in [
                "german",
                "english_translation",
                "schroder_semantic_category",
                "generation_count",
                "generation_proportion",
                "typicality_m",
                "typicality_sd",
                "aoa_m",
                "aoa_sd",
                "familiarity_m",
                "familiarity_sd",
                "frequency_per_million",
                "log10_frequency",
                "n_phonemes",
                "n_syllables",
            ]:
                base[col] = hit[col]
            rows.append(base)
    return pd.DataFrame(rows)


def canonical(value: Any) -> str:
    text = str(value)
    return ORTHOGRAPHIC_MAP.get(text, text)


def summarize_controls(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variables = [
        ("primary_semantic_margin", "Original semantic margin"),
        ("typicality_m", "Schröder broad typicality"),
        ("familiarity_m", "Familiarity"),
        ("aoa_m", "Age of acquisition"),
        ("log10_frequency", "DLEXDB log10 frequency"),
        ("n_phonemes", "Word length (phonemes)"),
        ("exact_target_competitor_typicality_available", "Exact KH target-competitor typicality contrast"),
    ]
    for col, label in variables:
        if col == "exact_target_competitor_typicality_available":
            n = int(df[col].fillna(False).sum()) if col in df else 0
            rows.append(
                {
                    "variable": label,
                    "coverage": f"{n}/19",
                    "spearman_rho": np.nan,
                    "spearman_p": np.nan,
                    "semantic_margin_control_slope": np.nan,
                    "semantic_margin_control_p": np.nan,
                    "note": "Unavailable; Schröder categories do not encode the KH target and competitor category pair.",
                }
            )
            continue
        columns = ["rho_hat", "primary_semantic_margin"] if col == "primary_semantic_margin" else ["rho_hat", "primary_semantic_margin", col]
        sub = df[columns].dropna()
        if len(sub) >= 4 and sub[col].nunique() > 1:
            rho, p = stats.spearmanr(sub[col], sub["rho_hat"])
        else:
            rho, p = math.nan, math.nan
        sem_slope, sem_p = math.nan, math.nan
        if col != "primary_semantic_margin" and len(sub) >= 8 and sub[col].nunique() > 1:
            fit = smf.ols(f"rho_hat ~ primary_semantic_margin + Q('{col}')", data=sub).fit(cov_type="HC1")
            sem_slope = float(fit.params.get("primary_semantic_margin", math.nan))
            sem_p = float(fit.pvalues.get("primary_semantic_margin", math.nan))
        rows.append(
            {
                "variable": label,
                "coverage": f"{len(sub)}/19",
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "spearman_p": float(p) if np.isfinite(p) else np.nan,
                "semantic_margin_control_slope": sem_slope,
                "semantic_margin_control_p": sem_p,
                "note": "Lexical-semantic control; not a replacement semantic margin.",
            }
        )
    return pd.DataFrame(rows)


def write_latex(summary: pd.DataFrame, path: Path) -> None:
    rows = []
    for row in summary.itertuples(index=False):
        rows.append(
            f"{latex_escape(row.variable)} & {row.coverage} & {fmt(row.spearman_rho)} & {format_p(row.spearman_p)} & {fmt(row.semantic_margin_control_slope)} & {format_p(row.semantic_margin_control_p)} & {latex_escape(row.note)} \\\\"
        )
    text = "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Schröder et al. German norm coverage and lexical-semantic controls for the KH2017 exemplars. These norms do not provide exact target--competitor category typicality contrasts for the manuscript's semantic margin.}",
            r"\label{tab:german-norm-coverage}",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{lrrrrrp{5.7cm}}",
            r"\toprule",
            r"Variable & Coverage & Spearman $r_s$ with $\rho$ & $p$ & Semantic-margin slope with control & $p$ & Note \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"}%",
            r"\end{table}",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def latex_escape(value: Any) -> str:
    text = str(value)
    for old, new in {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }.items():
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
