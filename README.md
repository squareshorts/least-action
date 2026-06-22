# Variational Action Models of Response Competition

**Public Repository:** https://github.com/squareshorts/least-action

This repo tests a bounded but ambitious claim:

> Response competition can be formalized as deformation of an action landscape, with observed behavior organized by task-dependent cognitive potentials.

It uses the public `KH2017_raw` dataset from the `mousetrap` R package. The dataset contains human mouse trajectories from a semantic categorization task with typical and atypical exemplars. Atypical items are expected to pull trajectories toward the competing response more strongly.

## What the run does

1. Downloads `KH2017_raw.rda` from the `mousetrap` GitHub repository if needed.
2. Uses `Rscript` to export the R data frame to CSV.
3. Keeps correct trials and canonicalizes them so:
   - start is `(0, 0)`
   - chosen/correct target is `(1, 1)`
   - competitor is `(-1, 1)`
4. Computes standard mouse-tracking measures:
   - AUC-like leftward deviation from a minimum-jerk motor path
   - maximum deviation
   - response time
5. Fits and evaluates:
   - Model 1: minimum-jerk motor baseline
   - Model 2: descriptive condition-mean trajectory baseline
   - Model 3: spline and Bezier competitor-attraction baselines
   - Model 4: nested least-action deformation paths
   - Model 5: physical-action ablations with motor-only, target-only, shared competitor, condition-specific competitor, and trial-level competitor variants
6. Scores subject-wise cross-validated RMSE and Gaussian path log likelihood.
7. Runs robust subject-level inference, cluster bootstrap checks, permutation checks, counterfactual action-gap tests, item-level difficulty tests, and parameter-recovery simulations.

The main predictive comparison is subject-wise cross-validated trajectory RMSE and path likelihood. The main interpretability checks are whether fitted `rho` tracks typicality, item difficulty, RT, curvature, and eventually independent semantic-similarity scores.

## Reproduce manuscript

```powershell
python -m pip install -r requirements.txt
python reproduce_core.py
```

`python reproduce_core.py` regenerates the core manuscript analysis, figures, generated tables, summaries, and `outputs/reproducibility_summary.json`. `python reproduce_all.py` is kept as a compatibility alias for this fast core run.

For the slower secondary validation suite, run:

```powershell
python reproduce_full.py
```

`python reproduce_full.py` runs the core reproduction first, then adds sensitivity analyses, permutation checks, mixed-effects validation, secondary semantic predictors, multisource semantic validation, and supplementary validation tables.

## Submission outputs

- `manuscript.tex`
- `references.bib`
- `highlights.tex`
- `figures/*.png`
- `figures/decisive_four_panel.pdf`
- `figures/rho_subject_paired.pdf`
- `outputs/tables/*.tex`
- `outputs/reproducibility_summary.json`
- `outputs/semantic_scores_19_items.csv`
- `outputs/sensitivity_action_parameters.csv`
- `outputs/permutation_semantic_prior_summary.json`
- `outputs/mixed_effects_validation.csv`

## Model

The implemented action model is deliberately conservative: it nests the minimum-jerk motor baseline. Let `q_M(t)` be the target-directed minimum-jerk motor plan. The action model finds the lowest-cost deformation of that motor plan under an early competitor potential:

```text
S[q] = integral 0.5 * alpha * |q(t) - q_M(t)|^2
             + 0.5 * beta  * |d2(q(t) - q_M(t))/dt2|^2
             + U_C(q, t) dt
```

with fixed start and endpoint. The competitor potential is:

```text
U_C(q,t) = -A * rho * g_C(t) * exp(-||q-r_C||^2 / (2 sigma^2))
```

The main cognitive parameter is `rho`, the competitor-attraction strength relative to the target-directed motor plan. When `rho = 0`, the nested action model reduces to the minimum-jerk baseline.

## Stronger Tests

The upgraded analysis distinguishes three uses of `rho`:

- full-data condition rho: for visualization only
- fold-selected condition rho: used for subject-wise CV prediction
- trial-level rho: used as a latent-variable estimate, not as an out-of-sample prediction

The current Stage-1 result is mixed in a useful way. The action model beats motor baselines, robustly recovers higher competitor attraction for atypical trials, and item-level `rho` tracks error rate and RT. But simple spline/Bezier baselines and the high-parameter condition-mean trajectory baseline beat the current action model in raw RMSE. That is the next modeling target: preserve the action interpretation while matching the stronger curved baselines.

To add independent semantic geometry, fill `resources/semantic_scores_template.csv` with embedding or lexical scores and run:

```powershell
python scripts/run_analysis.py --semantic-scores resources/semantic_scores_template.csv
```
