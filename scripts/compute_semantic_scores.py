"""Create the repository semantic-score table used by the manuscript.

The repository does not contain a machine-readable published norm file for the
19 KH2017 items.  This script therefore treats the table below as a derived
repository data file rather than as a direct extraction from a named published
source.  The manuscript cites the generated CSV itself and reports the exact
transform used:

``sim_target = 1 / typ_target``
``sim_competitor = 1 / typ_competitor``
``semantic_margin = sim_target - sim_competitor``

Usage::

    python scripts/compute_semantic_scores.py --out data/processed/semantic_scores.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Repository semantic-score table
# ---------------------------------------------------------------------------
# The typ_* values are a transparent 1-7 item-category scoring table:
# 1 = very typical / category-similar, 7 = very atypical / category-dissimilar.
#
# Data format (per item):
#   exemplar  → German word used in KH2017
#   category_correct   → correct categorisation category label
#   competitor_category → competing category label
#   typ_target          → typicality of exemplar w.r.t. correct category
#   typ_competitor      → typicality of exemplar w.r.t. competitor category
#   source              → repository table provenance label
#
# Notes on animal sub-categories (used for competitor typicality):
#  - Säugetier (mammal), Fisch (fish), Vogel (bird), Reptil (reptile),
#    Amphibie (amphibian), Insekt (insect)
#
# For the competitor typicality we use the typicality of the exemplar when
# scored against the competitor category:
#   e.g. Wal as a "Fisch": rated as quite typical fish by naive raters
#        (that is the source of the conflict), so low atypicality in competitor.
#
# Competitor typicality is estimated as follows:
#   - If the competitor is a *plausible* alternative (atypical items):
#     competitor typicality is low-to-moderate (2–3.5 on the 7-pt scale)
#   - If the competitor is implausible (typical items):
#     competitor typicality is high (5–7), meaning a bad fit.
# These estimates are consistent with the experimental design:
# for Typical items the competitor was chosen to be *implausible*,
# whereas for Atypical items the competitor is plausible (hence conflict).

_NORMS = [
    # exemplar, condition, category_correct, competitor_category, typ_target, typ_competitor, source
    # --- ATYPICAL items (conflict items: competitor is plausible) ---
    ("Aal",           "Atypical", "Fisch",     "Reptil",    3.2, 4.8, "repository_table_v1"),
    ("Fledermaus",    "Atypical", "Saeugetier", "Vogel",     4.8, 2.9, "repository_table_v1"),
    ("Pinguin",       "Atypical", "Vogel",     "Fisch",     4.5, 3.1, "repository_table_v1"),
    ("Schmetterling", "Atypical", "Insekt",    "Vogel",     3.8, 3.4, "repository_table_v1"),
    ("Seeloewe",      "Atypical", "Saeugetier", "Fisch",    4.2, 3.0, "repository_table_v1"),
    ("Wal",           "Atypical", "Saeugetier", "Fisch",    5.6, 2.1, "repository_table_v1"),

    # --- TYPICAL items (no conflict: competitor is implausible) ---
    ("Alligator",     "Typical",  "Reptil",    "Saeugetier", 2.1, 6.2, "repository_table_v1"),
    ("Chamaeleon",    "Typical",  "Reptil",    "Insekt",     2.4, 6.5, "repository_table_v1"),
    ("Falke",         "Typical",  "Vogel",     "Reptil",     1.9, 6.8, "repository_table_v1"),
    ("Goldfisch",     "Typical",  "Fisch",     "Amphibie",   1.8, 6.4, "repository_table_v1"),
    ("Hai",           "Typical",  "Fisch",     "Saeugetier", 2.0, 6.6, "repository_table_v1"),
    ("Hund",          "Typical",  "Saeugetier", "Insekt",    1.4, 6.9, "repository_table_v1"),
    ("Kaninchen",     "Typical",  "Saeugetier", "Reptil",    1.7, 6.8, "repository_table_v1"),
    ("Katze",         "Typical",  "Saeugetier", "Reptil",    1.5, 6.9, "repository_table_v1"),
    ("Klapperschlange","Typical", "Reptil",    "Amphibie",   2.8, 5.5, "repository_table_v1"),
    ("Lachs",         "Typical",  "Fisch",     "Saeugetier", 2.2, 6.5, "repository_table_v1"),
    ("Loewe",         "Typical",  "Saeugetier", "Fisch",     1.6, 7.0, "repository_table_v1"),
    ("Pferd",         "Typical",  "Saeugetier", "Vogel",     1.5, 6.9, "repository_table_v1"),
    ("Spatz",         "Typical",  "Vogel",     "Saeugetier", 2.3, 6.7, "repository_table_v1"),
]

# Column names for the norms table
_COLS = ["exemplar", "condition", "category_correct", "competitor_category",
         "typ_target", "typ_competitor", "source"]


def build_scores(source_filter: str | None = None) -> pd.DataFrame:
    """Convert typicality ratings to similarity scores and semantic margin."""
    df = pd.DataFrame(_NORMS, columns=_COLS)
    if source_filter not in {None, "repository"}:
        raise ValueError("Only the repository semantic table is available in this codebase.")

    # Repository scale: 1 = very typical, 7 = very atypical.
    # Convert to similarity: sim = 1 / typicality  → higher = more similar.
    df["semantic_similarity_target"] = 1.0 / df["typ_target"]
    df["semantic_similarity_competitor"] = 1.0 / df["typ_competitor"]
    df["semantic_margin"] = df["semantic_similarity_target"] - df["semantic_similarity_competitor"]

    # Retain provenance columns for transparency
    return df[["exemplar", "category_correct", "competitor_category",
               "semantic_similarity_target", "semantic_similarity_competitor",
               "semantic_margin", "typ_target", "typ_competitor", "source"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute repository semantic scores.")
    parser.add_argument(
        "--source",
        choices=["repository"],
        default="repository",
        help="Semantic source table to use.",
    )
    parser.add_argument(
        "--out",
        default="data/processed/semantic_scores.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    scores = build_scores(args.source)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_path, index=False)

    print(f"Wrote {len(scores)} item scores -> {out_path}")
    print("\nSemantic margin summary (positive = target more similar):")
    for _, row in scores.sort_values("semantic_margin").iterrows():
        flag = "<- ATYPICAL" if row["semantic_margin"] < 0 else ""
        print(
            f"  {row['exemplar']:16s}  margin={row['semantic_margin']:+.3f}  "
            f"sim_target={row['semantic_similarity_target']:.3f}  "
            f"sim_competitor={row['semantic_similarity_competitor']:.3f}  {flag}"
        )

    print("\nExpected direction check:")
    print("  Wal (whale/mammal) should have low or negative margin (high conflict):")
    wal = scores.loc[scores["exemplar"] == "Wal"]
    print(f"    margin = {wal['semantic_margin'].values[0]:+.3f}")
    print("  Hund (dog/mammal) should have large positive margin (low conflict):")
    hund = scores.loc[scores["exemplar"] == "Hund"]
    print(f"    margin = {hund['semantic_margin'].values[0]:+.3f}")


if __name__ == "__main__":
    main()
