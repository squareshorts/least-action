from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "predictor": "independent_secondary_semantic_margin",
                "status": "not_feasible_from_current_repository",
                "reason": (
                    "No independent embedding file, second published norm table, or frozen LLM "
                    "rating table is present in the repository. No secondary semantic numbers "
                    "were invented."
                ),
                "rho_prediction_test": "not_run",
                "loocv_prediction_test": "not_run",
            }
        ]
    ).to_csv(out_path, index=False)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/secondary_semantic_predictor.csv")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
