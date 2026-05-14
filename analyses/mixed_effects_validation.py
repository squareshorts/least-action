from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trial_fits = pd.read_csv(args.trial_fits)
    semantic_scores = pd.read_csv(args.semantic_scores)
    df = trial_fits.merge(
        semantic_scores[["exemplar", "semantic_margin"]],
        on="exemplar",
        how="left",
    ).dropna(subset=["rho_hat", "semantic_margin", "condition", "subject", "exemplar"])
    df["atypical"] = (df["condition"] == "Atypical").astype(float)

    formula = "rho_hat ~ semantic_margin + atypical"
    estimator = "mixedlm_random_subject_intercept_item_variance_component"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = smf.mixedlm(
                formula,
                df,
                groups=df["subject"],
                vc_formula={"item": "0 + C(exemplar)"},
            ).fit(reml=False, method="lbfgs", maxiter=1000, disp=False)
    except Exception as exc:
        estimator = "ols_subject_fixed_effect_clustered_by_item_fallback"
        formula = "rho_hat ~ semantic_margin + atypical + C(subject)"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = smf.ols(formula, df).fit(cov_type="cluster", cov_kwds={"groups": df["exemplar"]})
        fallback_reason = str(exc)
    else:
        fallback_reason = ""

    conf = result.conf_int()
    rows: list[dict[str, object]] = []
    for term in ["semantic_margin", "atypical"]:
        if term not in result.params:
            continue
        rows.append(
            {
                "estimator": estimator,
                "formula": formula,
                "term": term,
                "coefficient": float(result.params[term]),
                "std_error": float(result.bse[term]),
                "ci_lower": float(conf.loc[term, 0]),
                "ci_upper": float(conf.loc[term, 1]),
                "p_value": float(result.pvalues[term]),
                "n_observations": int(result.nobs),
                "n_subjects": int(df["subject"].nunique()),
                "n_items": int(df["exemplar"].nunique()),
                "fallback_reason": fallback_reason,
            }
        )

    pd.DataFrame(rows).to_csv(out_path, index=False)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-fits", default="outputs/trial_fits.csv")
    parser.add_argument("--semantic-scores", default="data/processed/semantic_scores.csv")
    parser.add_argument("--out", default="outputs/mixed_effects_validation.csv")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
