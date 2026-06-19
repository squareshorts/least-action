import pandas as pd
import numpy as np
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

def main():
    norming_path = DATA / "norming" / "independent_ratings.csv"

    RESULTS.mkdir(exist_ok=True, parents=True)
    out_path = RESULTS / "independent_semantic_table.csv"
    hash_path = RESULTS / "independent_semantic_table.sha256"

    if not norming_path.exists():
        print(f"Warning: Norming file {norming_path} does not exist. Creating a placeholder table.")
        create_placeholder(out_path, hash_path)
        return

    # In actual execution, this block would process the empirical data
    df = pd.read_csv(norming_path)

    # Filter participants who failed checks
    df = df[(df['language_check'] == 1) & (df['attention_check'] == 1)]

    # Normally we would fit a mixed-effects model here: rating ~ category_role + (1|participant_id) + (1|item)
    # Using statsmodels mixedlm:
    # import statsmodels.formula.api as smf
    # md = smf.mixedlm("rating_1_to_7 ~ category_role", df, groups=df["participant_id"], re_formula="~1")
    # mdf = md.fit()

    # For simplification, compute mean typicality
    agg = df.groupby(["item", "category_role"])["rating_1_to_7"].mean().reset_index()

    # Compute similarity = 1 / mean_typicality
    agg["similarity"] = 1.0 / agg["rating_1_to_7"]

    pivoted = agg.pivot(index="item", columns="category_role", values="similarity").reset_index()
    if 'target' not in pivoted.columns or 'competitor' not in pivoted.columns:
        print("Missing target or competitor data.")
        create_placeholder(out_path, hash_path)
        return

    pivoted["independent_margin"] = pivoted["target"] - pivoted["competitor"]

    pivoted.to_csv(out_path, index=False)
    save_hash(out_path, hash_path)
    print(f"Computed independent semantic margins to {out_path}")

def create_placeholder(out_path, hash_path):
    # Read original items to get the item list
    items_path = DATA / "semantic_items_master.csv"
    df = pd.read_csv(items_path)

    out_df = pd.DataFrame({
        "item": df["item"],
        "sim_target_independent": np.nan,
        "sim_competitor_independent": np.nan,
        "independent_margin": np.nan
    })

    out_df.to_csv(out_path, index=False)
    save_hash(out_path, hash_path)

def save_hash(file_path, hash_path):
    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    with open(hash_path, "w") as f:
        f.write(f"{file_hash}  {file_path.name}\n")

if __name__ == "__main__":
    main()
