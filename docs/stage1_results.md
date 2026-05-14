# Stage 1 Results: KH2017

The current KH2017 analysis supports a stronger bounded claim than the initial proof of concept, but it also exposes the next model-building target.

## What Survives

- The nested action model beats the minimum-jerk motor baseline under subject-wise CV.
- Improvement is larger for atypical items than typical items.
- Trial-level competitor attraction `rho` is higher for atypical than typical trials.
- Subject-level paired inference is robust under bootstrap and sign-flip checks.
- `rho` predicts AUC beyond condition and subject effects under cluster-resampling checks.
- Item-level `rho` tracks error rate and item RT.
- Simulated parameter recovery is strong for the current grid model.

## What Does Not Yet Survive

The current action model does not beat the non-Lagrangian Bezier/spline attraction baselines in raw RMSE. It also does not beat the high-parameter condition-mean trajectory baseline, which is expected but important to report.

This means the strongest honest claim is not yet:

> The action model is the best trajectory predictor.

The stronger defensible claim is:

> Action-derived latent variables recover conflict structure and predict behavior, while the current deterministic action path needs a richer time-course or stochastic formulation to match flexible curved baselines.

## Next Model Upgrades

1. Add independent semantic-similarity scores and test whether `rho` tracks semantic competitor similarity or semantic margin.
2. Replace deterministic path scoring with a stochastic action model: `P[q] proportional to exp(-S[q] / tau)`.
3. Add an item-level or semantic-prior model for `rho`, so held-out subjects are predicted from item geometry rather than only condition labels.
4. Use the EEG plus mouse-tracking dataset for neural validation of `rho`, action gap, and action gradients.
5. Use IBL as a later cross-species decision-dynamics test bed, not as the immediate next step.

