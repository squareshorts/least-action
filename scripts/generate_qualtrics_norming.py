import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

def main():
    items_path = DATA / "semantic_items_master.csv"
    if not items_path.exists():
        print(f"Error: {items_path} not found.")
        return

    df = pd.read_csv(items_path)

    # We need to present each item paired with its target category and competitor category
    rows = []
    for _, row in df.iterrows():
        item = row["item"]
        target = row["target_category"]
        competitor = row["competitor_category"]

        # Target
        rows.append({
            "item": item,
            "category": target,
            "category_role": "target"
        })

        # Competitor
        rows.append({
            "item": item,
            "category": competitor,
            "category_role": "competitor"
        })

    out_df = pd.DataFrame(rows)

    # Add optional filler items to disguise the task
    fillers = [
        {"item": "Auto", "category": "Fahrzeug", "category_role": "filler"},
        {"item": "Apfel", "category": "Frucht", "category_role": "filler"},
        {"item": "Hammer", "category": "Werkzeug", "category_role": "filler"},
    ]
    out_df = pd.concat([out_df, pd.DataFrame(fillers)], ignore_index=True)

    # Shuffle the rows for Qualtrics
    out_df = out_df.sample(frac=1, random_state=42).reset_index(drop=True)

    out_dir = DATA / "norming"
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / "qualtrics_items.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Generated Qualtrics item list at {out_path}")

if __name__ == "__main__":
    main()
