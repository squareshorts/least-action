# Updated Rejection Risk Report

Date: 2026-06-18

## Decision

Status: **YELLOW, moving toward GREEN**.

The Stage 2 evidence is materially stronger than Stage 1: both Koenig-Robert lookalike-object studies show the expected increase in fitted `rho`, and SWOW-DE network proximity predicts KH2017 item-level `rho` in the expected direction. I am not assigning full GREEN because the downloaded OSF mouse-tracking archive did not include scalar independent face-likeness or animal-likeness ratings, and the strongest SWOW result uses network proximity rather than direct cue-response association.

## What Improved

- External mouse-tracking validation now supports the latent-parameter claim in a nonsemantic visual categorization setting.
- Face-like objects categorized as objects had higher stimulus-level `rho` than ordinary objects: difference = 0.031, stimulus-cluster CI [0.012, 0.048], permutation p = 0.002.
- Animal-like objects categorized as objects had higher stimulus-level `rho` than ordinary objects: difference = 0.154, stimulus-cluster CI [0.103, 0.202], permutation p < 0.001.
- Pooled-object null simulations did not produce differences as large as the observed effects.
- SWOW-DE network margin covered 18 of 19 KH2017 items and predicted fitted `rho` in the expected direction: Spearman r = -0.590, p = 0.010; Theil-Sen slope = -0.923, bootstrap CI [-1.958, -0.207]; permutation p = 0.005.
- The Schröder norm audit showed 13/19 item coverage and supports using those norms only as lexical-semantic controls, not as substitute category-contrast evidence.

## Main Remaining Risks

1. **Independent perceptual ratings are unavailable in the downloaded OSF archive.** The external mouse-tracking result uses binary lookalike-object status as the ambiguity proxy. This is useful, but weaker than showing that independent continuous face-likeness or animal-likeness ratings predict `rho`.
2. **SWOW direct association coverage is sparse.** Direct cue-to-response coverage is 11/19 and points opposite the expected direction. The prespecified main SWOW result therefore relies on graph proximity.
3. **SWOW network effects partly overlap with the typical/atypical structure.** The main network result is directionally and statistically supportive, but the atypical-control regression is not independently significant for the SWOW network metric.
4. **Raw curve fitting remains a vulnerability if framed incorrectly.** The manuscript must state plainly that flexible descriptive baselines or trial-fitted upper bounds may win raw trajectory NLL. The claim is about interpretable latent competition, not best raw path prediction.

## Required Framing

Use this language family:

- “converged with”
- “was consistent with”
- “provided convergent external support”
- “generalized to an independent visual categorization mouse-tracking setting”

Avoid:

- “replicated”
- “proved semantic causality”
- “universal least-action law”
- “best trajectory curve-fitter”

## Files Added or Updated

- `analyses/external_koenig_robert_validation.py`
- `results/external_koenig_robert_preprocessing_report.md`
- `results/external_koenig_robert_validation.csv`
- `tables/table_external_mouse_tracking_validation.tex`
- `figures/external_rho_by_condition.png`
- `figures/external_ambiguity_rho.png`
- `analyses/swow_de_validation.py`
- `results/swow_de_mapping_table.csv`
- `results/swow_de_margin_validation.csv`
- `results/swow_de_margin_validation.sha256`
- `tables/table_swow_margin_validation.tex`
- `figures/swow_margin_rho.png`
- `analyses/german_norm_audit.py`
- `results/german_norm_coverage.csv`
- `tables/table_german_norm_coverage.tex`
- `manuscript_insert_external_validation_methods.tex`
- `manuscript_insert_external_validation_results.tex`
- `manuscript_insert_external_validation_discussion.tex`

## Bottom Line

The manuscript is now substantially more defensible. It should be framed as a validated KH2017 reanalysis with convergent external support from independent mouse-tracking and German association norms. It is close to GREEN, but the absence of scalar perceptual ratings and sparse direct SWOW association coverage justify a cautious YELLOW classification.
