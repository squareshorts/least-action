# Results Insertions

## End of Section 3.2

At the subject level, the cross-validated action model reduced RMSE relative to the minimum-jerk baseline across all trials (action: 0.461; minimum jerk: 0.481; paired mean difference, minimum jerk minus action = 0.020, 95% bootstrap CI [0.017, 0.022], paired t test p = 5.332e-22, n = 60 subjects). The same direction was observed for typical trials (difference = 0.015, 95% CI [0.014, 0.017]) and atypical trials (difference = 0.029, 95% CI [0.023, 0.035]).

## End of Section 3.3

In the item-level semantic-only regression, fitted rho decreased as semantic margin increased (slope = -0.546, 95% CI [-0.689, -0.404], p = 3.196e-07, n = 19 items). Semantic margin was strongly associated with fitted item-level rho by Pearson correlation (r = -0.891, p = 3.196e-07) and Spearman rank correlation (rho_s = -0.855, p = 3.152e-06). Leave-one-item-out semantic predictions tracked held-out fitted rho (Pearson r = 0.838; Spearman rho_s = 0.847; RMSE = 0.084; MAE = 0.063; calibration intercept = 0.019; calibration slope = 0.969).

## End of Section 3.4

The semantic-margin action model improved held-out NLL without improving raw trajectory RMSE. Across held-out trials, condition-only action rho had mean RMSE 0.461 and mean NLL 38.813, whereas semantic-margin action rho had mean RMSE 0.462 and mean NLL 38.175. The mean fold-level training variance estimates were tau^2 = 0.1248 for condition-only action rho and tau^2 = 0.1231 for semantic-margin action rho. Item-wise RMSE gain and NLL gain were positively associated (Pearson r = 0.858, Spearman rho_s = 0.708), but raw RMSE favored the semantic model for only 4 of 19 items. Thus the NLL result is best interpreted as stochastic residual/noise calibration, not as a general reduction in raw RMSE.

## Section 3.5

In the 5,000-permutation test, the observed item-wise NLL gain was 0.694; 0 null permutations were at least as large, giving the exact empirical p = (0 + 1)/(5000 + 1) = 2.000e-04 (seed 20260514). The mixed-effects validation used rho_hat ~ semantic_margin + atypical in statsmodels MixedLM (item variance component with subject grouping; no subject random-intercept variance estimated; ML, L-BFGS); the semantic-margin coefficient was -0.661 (95% CI [-0.681, -0.641], p < 1e-300).

## Supplement Notes

Add Table S: semantic regression details and Table S: LOOCV semantic prediction metrics to the semantic predictor provenance section.

Add Table S: subject-wise CV paired comparisons to the trajectory prediction section or main-text supporting material.

Add Table S: stochastic variance calibration to the residual/noise robustness section.

Add Table S: mixed-effects model details, Table S: error logistic details, and Table S: reproducibility metadata to the validation/reproducibility supplement sections.
