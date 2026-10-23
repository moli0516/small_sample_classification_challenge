# Experiment List

## Scope and current status

This is a development research record and roadmap, not a claim that model selection
or independent final evaluation is complete.

Completed studies now include E0–E5, the bounded XGBoost comparison (E5b),
a single-setting XGBoost extension (E5b extension 01), and a bounded 30-trial
Optuna search (E5b extension 02).

- Data audit: completed; 250 labelled rows, 19,750 test rows, 150 numeric predictors.
- Identifier: `id`, excluded from predictors; target: `target`.
- ValidationRunner and SubmissionWriter tests: reported passing by the project owner.
  This is not a claim that every subsequent code revision has been independently audited.
- Baseline submission workflow: completed, as reported by the project owner;
  public score, submission identifier, and timestamp remain to be recorded.
- Current highest-mean development candidate: Optuna Trial 20,
  `e5b_optuna_ext02_trial_0020_depth_4_trees_50`.
- Trial 20 mean development CV AUC: 0.881937; repeat SD: 0.013949.
- Trial 20 versus L1 C=1.0: mean paired improvement +0.019826;
  improved / worse / tied repetitions: 10 / 0 / 0.
- Trial 20 versus L1 C=0.1: mean paired improvement +0.029074;
  improved / worse / tied repetitions: 10 / 0 / 0.
- Retained linear reference: all-feature L1 C=1.0,
  mean development CV AUC 0.862112, repeat SD 0.012204.
- Retained sparse alternative: all-feature L1 C=0.1,
  mean development CV AUC 0.852864, repeat SD 0.010803.
- The original six-setting XGBoost grid and depth-4/1000-round extension did not
  replace L1. The later joint Optuna search found a higher-mean development candidate.
- Optional Elastic-net: not run.
- Pruning and early stopping: not used in the completed Optuna search;
  no subsequent study using them has been confirmed or completed.
- Trial 20 parameters have been selected. A freeze procedure and proposed snapshot
  version `e5b_optuna_trial20_v1` have been supplied, but successful snapshot creation
  has not yet been confirmed. Do not mark the physical archive as completed without evidence.
- No final full-data XGBoost refit, saved final XGBoost model, or XGBoost submission
  has been confirmed. Do not confuse parameter selection with these later steps.
- E4/E5/XGBoost submission outcomes: not recorded.
- No verified Kaggle score is recorded here. Unattributed reports above 0.90 are
  not evidence of this project's leaderboard performance. The XGBoost smoke-test
  AUC above 0.90 was a local single-fold diagnostic, not a Kaggle score.
- E6 individual extension: proposed, not yet confirmed.
- All model-selection scores remain development results. No independent final
  evaluation has been completed.

Historical leading-candidate decisions in E0–E5 describe the state at the end of
those studies. The current leader is stated above and in E5b extension 02.

## Shared experimental rules

1. Use the same training data and row order across comparisons; verify saved split indices.
2. Unless studying validation itself, use shuffled stratified five-fold CV with seeds 0–9.
3. Fit scaling and supervised feature selection inside each training fold only.
   The completed XGBoost studies used neither scaling nor explicit feature selection.
4. Keep estimator randomness fixed separately from the split seed; do not tune model seeds.
5. Report mean fold AUC per repetition, then mean and sample SD across repetitions.
6. Compare candidates using paired repeat-level differences on identical splits.
7. Do not interpret repeated splits as independent datasets or their SD as a confidence interval.
8. Preserve configurations, scores, split indices, package versions, data hashes, and code revision.
9. Define the question, settings/search space, budget, validity rules, and stopping rule
   before each new experiment. Identify adaptive follow-ups honestly.
10. Treat scores used to choose models as development results, not independent final evaluation.
11. Do not submit every candidate or tune directly to public leaderboard feedback.
12. Keep course code, data, and generated artifacts private under the assignment rules.
13. Preserve failures, warnings and rejected trials; never average only successful folds
    to create a scored candidate after a fold has failed.
14. New experiments and snapshots must not overwrite historical results or reuse an
    existing output directory silently.
15. Changes to parameters, early-stopping/pruning rules, features or evaluation after
    candidate selection require an explicit new protocol/version.

## Overview

| ID | Investigation | Status | Priority |
|---|---|---|---|
| E0 | All-feature logistic regression: C sweep | Completed | Preserve |
| E1 | End-to-end baseline submission | Completed | Record actual feedback |
| E2 | Correlation-based feature selection | Completed | Preserve |
| E3 | Selected feature count / C interaction | Completed | Preserve |
| E4 | L1 logistic regression; optional Elastic-net | L1 completed; Elastic-net not run | Retain linear references |
| E5 | Bounded RBF SVM comparison | Completed; warning review completed | Preserve negative finding |
| E5b | Bounded shallow XGBoost grid | Completed | Preserve |
| E5b extension 01 | Depth 4, 1000 boosting rounds | Completed | Preserve negative finding |
| E5b extension 02 | Joint XGBoost tuning with Optuna | 30 trials completed; Trial 20 selected | Archive and freeze candidate |
| E6 | Validation protocol extension | Proposed; not confirmed | Required if chosen |
| E7 | Candidate freeze and final evaluation | Candidate identified; snapshot confirmation/evaluation pending | Required |

These IDs apply to this document. Do not renumber E6/E7 or rename existing experiment
IDs/output directories when recording E5b and its extensions.

## E0 — Completed C sweep

Question: How does C affect a fixed all-feature logistic-regression baseline?

Fixed pipeline: `StandardScaler -> LogisticRegression`, solver `lbfgs`,
max_iter=2000, all 150 predictors, no feature selection.

| C | Mean CV AUC | Repeat SD |
|---:|---:|---:|
| 1.0 | 0.751411 | 0.019922 |
| 0.1 | 0.771935 | 0.013106 |
| 0.01 | 0.792885 | 0.011157 |
| 0.001 | 0.793999 | 0.009091 |
| 0.0001 | 0.790600 | 0.008026 |

Decision at E0 completion: retain C=0.001 as the provisional baseline;
C=0.01 remains a close alternative. This was a sequential exploratory study,
not a fully preregistered grid. Stop the sweep. Subsequent investigations are
recorded separately below.

## E1 — Baseline submission

Status: completed, as reported by the project owner.

Question: Can the complete refit/predict/export/upload workflow run successfully?

Configuration:
- `StandardScaler -> LogisticRegression`, all 150 predictors, id excluded.
- C=0.001; refit on all 250 labelled rows.
- Predict class-1 probabilities for all 19,750 test rows.
- Submission columns: `id,target`, aligned to the official template by ID.

Completed:
- [x] Generate the baseline submission.
- [x] Confirm submission format and ID alignment.
- [x] Complete the Kaggle submission workflow.
- [x] Save the submission, fitted pipeline, and configuration.

Results:
- Mean development CV AUC: 0.793999.
- Repeat SD: 0.009091.
- Public leaderboard score: to be recorded.
- Kaggle submission identifier and timestamp: to be recorded.

Decision: retain as the initial external submission reference. Successful upload
is not evidence of model optimality. Do not duplicate an already completed baseline
submission; recover its actual feedback. This was not the final-solution claim.

## E2 — Correlation-based feature selection

Status: completed.

Question: Does removing low-ranked features improve the baseline?

Protocol:
- `CorrelationSelector -> StandardScaler -> LogisticRegression`.
- Absolute Pearson correlation ranking, fitted within each training fold.
- k in {5,10,20,50,150}; C=0.001.
- Stratified five-fold CV repeated with seeds 0–9.
- Retain original column order after selection; save per-fold selected features.

| k | Mean CV AUC | Repeat SD | Mean difference vs k=150 | Improved / worse repeats |
|---:|---:|---:|---:|---|
| 5 | 0.795349 | 0.006414 | +0.001350 | 6 / 4 |
| 10 | 0.808618 | 0.008244 | +0.014619 | 10 / 0 |
| 20 | 0.798767 | 0.008428 | +0.004769 | 6 / 4 |
| 50 | 0.797398 | 0.008861 | +0.003400 | 4 / 6 |
| 150 | 0.793999 | 0.009091 | 0.000000 | Control |

Findings:
- k=10 had the highest mean and improved over all features in all ten repetitions.
- k=5,20,50 did not consistently improve; k=50 had a positive mean difference
  despite six worse repetitions.
- k=150 reproduced baseline repeat scores.
- Fold-selection frequencies are descriptive, not a bootstrap study.

Decision: retain k=10 and the second-highest-mean k=20 for E3; stop the k sweep.

Limitations: only C=0.001 tested; selected identities vary by fold; the evidence
supports a selection procedure, not exactly ten truly relevant predictors.
The winning score is a development model-selection result.

Artifacts: per-configuration settings/scores/splits/selected_features.json,
comparison and paired tables, selection frequencies, AUC and difference figures.

## E3 — Feature-count / C interaction

Status: completed.

Question: Is C=0.001 still appropriate after feature selection?

Protocol:
- `CorrelationSelector -> StandardScaler -> LogisticRegression`.
- k in {10,20}; C in {0.001,0.01,0.1}.
- Reuse two E2 C=0.001 configurations; four new configurations, 200 additional fits.
- Same data, order, splits and other settings; seeds 0–9, five folds.
- Selection and scaling remain fold-local.

| k | C | Mean CV AUC | Repeat SD | Mean difference vs E2 leader | Improved / worse repeats |
|---:|---:|---:|---:|---:|---|
| 10 | 0.001 | 0.808618 | 0.008244 | 0.000000 | Control |
| 10 | 0.01 | 0.812664 | 0.007200 | +0.004046 | 10 / 0 |
| 10 | 0.1 | 0.822982 | 0.008456 | +0.014364 | 10 / 0 |
| 20 | 0.001 | 0.798767 | 0.008428 | -0.009851 | 2 / 8 |
| 20 | 0.01 | 0.799816 | 0.010557 | -0.008801 | 3 / 7 |
| 20 | 0.1 | 0.794235 | 0.013509 | -0.014382 | 2 / 8 |

The reference above is the E2 leader k=10/C=0.001, not the all-feature model.

| k | Change from C=0.001 | Mean paired difference | Improved / worse repeats |
|---:|---|---:|---|
| 10 | C=0.01 | +0.004046 | 10 / 0 |
| 10 | C=0.1 | +0.014364 | 10 / 0 |
| 20 | C=0.01 | +0.001049 | 5 / 5 |
| 20 | C=0.1 | -0.004532 | 3 / 7 |

Findings:
- k=10/C=0.1 had the highest mean; the C effect depended on k.
- At k=10, both increases improved over C=0.001 in all ten repetitions.
- At k=20, C=0.01 was mixed and C=0.1 reduced the mean.
- Highest mean did not imply lowest repeat SD.
- The supplied summary did not establish direct win/loss counts between
  k=10/C=0.1 and k=10/C=0.01.

Leading candidate at E3 completion:
- `e3_correlation_k_10_c_0_1`, mean 0.822982, SD 0.008456.
- Versus E2 leader: +0.014364, ten improvements.
- Versus all-feature C=0.001: +0.028983, ten improvements.

Decision at E3 completion: retain this provisional leader and earlier references;
stop E3 and proceed to E4. Candidate submission outcome must be recorded separately.

Limitations: adaptive follow-up using reused observations; C=0.1 was the largest
value tested; no global k/C optimum or leaderboard guarantee established.

Artifacts: four new and two reused experiment directories, configs, scores,
splits/selected_features.json, e3_comparison.csv, e3_paired_repeat_scores.csv,
and mean-AUC/repeat-trajectory/paired-difference figures.

## E4 — Sparse regularisation

Status: four planned L1 settings completed. Optional Elastic-net not run.

Question: Can model-based sparsity compete with explicit correlation filtering?

Protocol:
- `StandardScaler -> L1 LogisticRegression`; all 150 predictors, no correlation selector.
- C in {0.01,0.1,1.0,10.0}; same ten repetitions of five-fold CV; 200 fits.
- Scaling fold-local; fixed estimator randomness separate from split seed.
- Save exact/effective nonzero counts, identities/frequencies, iterations and warnings.

Configuration provenance:
- Intended settings: solver=saga, model_seed=42, tol=1e-4, max_iter=10000.
- Verify saved settings, library interface and actual estimator construction.
  Result tables alone do not independently verify training parameters.
- Initial notebook validation failed because regularisation-field mapping did not
  match config schema. Resolve the schema/interface, not by disabling checks or relabelling runs.
- Preserve versions, data hashes, splits and code revision.

References: all-feature L2 C=0.001 (mean 0.793999), and E3 k=10/C=0.1 (mean 0.822982).

| L1 C | Mean CV AUC | Repeat SD | Minimum repeat AUC | Maximum repeat AUC |
|---:|---:|---:|---:|---:|
| 0.01 | 0.500000 | 0.000000 | 0.500000 | 0.500000 |
| 0.1 | 0.852864 | 0.010803 | 0.838038 | 0.876227 |
| 1.0 | 0.862112 | 0.012204 | 0.844883 | 0.878112 |
| 10.0 | 0.817571 | 0.015909 | 0.790680 | 0.842070 |

Ranges here and below refer to repetition-level mean fold AUC, not individual folds.

| L1 C | Mean difference vs all-feature L2 | Difference SD | Improved / worse / tied |
|---:|---:|---:|---|
| 0.01 | -0.293999 | 0.009091 | 0 / 10 / 0 |
| 0.1 | +0.058865 | 0.008286 | 10 / 0 / 0 |
| 1.0 | +0.068113 | 0.012379 | 10 / 0 / 0 |
| 10.0 | +0.023573 | 0.016327 | 9 / 1 / 0 |

| L1 C | Mean difference vs E3 leader | Difference SD | Improved / worse / tied |
|---:|---:|---:|---|
| 0.01 | -0.322982 | 0.008456 | 0 / 10 / 0 |
| 0.1 | +0.029882 | 0.010501 | 10 / 0 / 0 |
| 1.0 | +0.039130 | 0.015253 | 10 / 0 / 0 |
| 10.0 | -0.005411 | 0.019133 | 3 / 7 / 0 |

Direct C=1.0 minus C=0.1 comparison:
- Mean difference +0.009248; sample difference SD 0.014681.
- Improved / worse / tied: 8 / 2 / 0; worse at seeds 7 and 8.
- Difference range -0.016263 to +0.028807.

| L1 C | Mean exact nonzero | Mean effective nonzero | Effective range | All-zero folds | Warning folds | Iteration-limit folds | Max n_iter |
|---:|---:|---:|---|---:|---:|---:|---:|
| 0.01 | 0.00 | 0.00 | 0–0 | 50 | 0 | 0 | 3 |
| 0.1 | 15.76 | 15.76 | 11–22 | 0 | 0 | 0 | 127 |
| 1.0 | 74.96 | 74.96 | 64–85 | 0 | 0 | 0 | 1684 |
| 10.0 | 97.06 | 97.06 | 85–107 | 0 | 0 | 0 | 3978 |

Exact/effective counts matched in every supplied fold. Preserve the threshold.
All four numerical_review_required flags were False; recorded absence of warnings
is not an independent audit of the warning-capture implementation or exact optimality.

Feature-frequency diagnostics (nonzero folds / 50):

| Metric | C=0.1 | C=1.0 |
|---|---:|---:|
| Ever nonzero | 61 | 145 |
| Nonzero in every fold | 4 | 11 |
| Nonzero in at least 90% of folds | 7 | 21 |
| Never nonzero | 89 | 5 |

Frequency totals reproduce mean counts: 788/50=15.76; 3748/50=74.96.
- C=0.1, >=90%: 46/80/100/138 at 50/50; 30/67 at 49/50; 108 at 48/50.
- C=1.0, 50/50: 6,17,22,30,46,80,85,100,115,138,141.
- Shared 50/50 core: 46,80,100,138.
- Feature 67 frequencies: 98% at C=0.1, 0% at C=1.0, 92% at C=10.0.
  Selection was not nested across C; frequency alone does not identify the cause.

Findings:
- C=1.0 had highest L1 mean; C=0.1 is the sparse alternative.
- Both exceeded E3 and all-feature L2 in all ten repetitions.
- C=0.01 had all-zero feature coefficients and AUC=0.5 in every fold; preserve it.
- C=10 retained more coefficients but had worse mean AUC than C=0.1 and C=1.
- Average count is not a fixed selected set. Frequency is not coefficient magnitude,
  sign stability, causal relevance or the true number of informative predictors.

Decision at E4 completion:
- Retain `e4_l1_c_1` (mean 0.862112, SD 0.012204) as the then-leader.
- Retain `e4_l1_c_0_1` (mean 0.852864, SD 0.010803) as sparse alternative.
- Preserve earlier references; stop the four-setting L1 sweep.
- Do not convert global frequencies into a selector and reuse old CV scores.
- Verify provenance before refit; record any actual submission separately.

Limitations: adaptive reused-data development study; no independent final estimate;
complete-pipeline comparison, not solver-controlled ablation; no global optimum or
leaderboard guarantee. Any new frequency-based selector needs appropriate fold-local evaluation.

Artifacts: e4_l1_c_0_01/e4_l1_c_0_1/e4_l1_c_1/e4_l1_c_10 directories,
configs/scores/splits/coefficients/versions/hashes/revision, comparison,
feature-frequency and fold diagnostics, e4_paired_repeat_scores.csv,
e4_direct_paired_comparison.csv, and figures. Verify archive completeness.

Optional Elastic-net, not run: saga; C in {0.1,1.0}; l1_ratio=0.5.
Confirm a new question/protocol before execution; not part of completed E4 results.

## E5 — Bounded RBF SVM comparison

Status: completed; warning review completed.

Question: Does an all-feature RBF SVM improve over retained L1 candidates on identical splits?

Implementation:
- Originally executed through standalone run_e5.py, not the src/ ValidationRunner.
- Split-loader fix supported metadata-wrapped splits and valid_indices;
  normalization occurred in memory, without rewriting/regenerating E4 splits.
- Later file reorganization instructions move execution to scripts/run_e5.py,
  warning inspection to scripts/check_e5_warning.py and tests into tests/,
  retaining a root compatibility entry point. This reorganization is not a new numerical experiment;
  verify actual local paths and preserve the exact executed historical source revision.

Protocol:
- `StandardScaler -> SVC(kernel="rbf")`, all 150 predictors, id excluded.
- C={0.1,1,10}, gamma={scale,0.01}; six candidates, 300 fits.
- Same saved ten repetitions of stratified five-fold CV; fold-local scaling.
- decision_function ROC AUC; mean fold AUC per repeat, then mean/sample SD.
- No feature selection, PCA, calibration or simultaneous extra search.
- probability=False, class_weight=None, tol=1e-3, max_iter=10000,
  shrinking=True, cache_size=200 MB, random_state=42.
  SVC ignores random_state with probability=False.

References: E4 L1 C=1.0 (0.862112/0.012204), E4 L1 C=0.1
(0.852864/0.010803), E3 k=10/C=0.1 (0.822982/0.008456).

| C | Gamma | Mean CV AUC | Repeat SD | Min repeat AUC | Max repeat AUC |
|---:|---|---:|---:|---:|---:|
| 0.1 | scale | 0.807268 | 0.011407 | 0.790022 | 0.827447 |
| 1.0 | scale | 0.807959 | 0.012793 | 0.784573 | 0.825831 |
| 10.0 | scale | 0.803024 | 0.013892 | 0.780751 | 0.823229 |
| 0.1 | 0.01 | 0.811179 | 0.012608 | 0.789726 | 0.831021 |
| 1.0 | 0.01 | 0.812318 | 0.013557 | 0.787496 | 0.833254 |
| 10.0 | 0.01 | 0.812385 | 0.013559 | 0.788799 | 0.829382 |

| C | Gamma | Mean difference vs L1 leader | Difference SD | Improved / worse / tied |
|---:|---|---:|---:|---|
| 0.1 | scale | -0.054844 | 0.012358 | 0 / 10 / 0 |
| 1.0 | scale | -0.054153 | 0.013454 | 0 / 10 / 0 |
| 10.0 | scale | -0.059087 | 0.012564 | 0 / 10 / 0 |
| 0.1 | 0.01 | -0.050932 | 0.013182 | 0 / 10 / 0 |
| 1.0 | 0.01 | -0.049794 | 0.013746 | 0 / 10 / 0 |
| 10.0 | 0.01 | -0.049726 | 0.012739 | 0 / 10 / 0 |

| C | Gamma | Mean difference vs sparse L1 | Difference SD | Improved / worse / tied |
|---:|---|---:|---:|---|
| 0.1 | scale | -0.045595 | 0.008274 | 0 / 10 / 0 |
| 1.0 | scale | -0.044905 | 0.007382 | 0 / 10 / 0 |
| 10.0 | scale | -0.049839 | 0.009024 | 0 / 10 / 0 |
| 0.1 | 0.01 | -0.041684 | 0.008385 | 0 / 10 / 0 |
| 1.0 | 0.01 | -0.040545 | 0.008853 | 0 / 10 / 0 |
| 10.0 | 0.01 | -0.040478 | 0.008079 | 0 / 10 / 0 |

| C | Gamma | Mean difference vs E3 leader | Difference SD | Improved / worse / tied |
|---:|---|---:|---:|---|
| 0.1 | scale | -0.015714 | 0.010717 | 1 / 9 / 0 |
| 1.0 | scale | -0.015023 | 0.012967 | 1 / 9 / 0 |
| 10.0 | scale | -0.019957 | 0.014307 | 1 / 9 / 0 |
| 0.1 | 0.01 | -0.011803 | 0.012194 | 1 / 9 / 0 |
| 1.0 | 0.01 | -0.010664 | 0.013668 | 2 / 8 / 0 |
| 10.0 | 0.01 | -0.010597 | 0.014015 | 1 / 9 / 0 |

| C | Gamma | Completed / valid | Failed | General-warning folds | Convergence-warning folds | Iteration-limit folds | Max n_iter | Mean support fraction |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0.1 | scale | 50 / 50 | 0 | 50 | 0 | 0 | 149 | 0.9869 |
| 1.0 | scale | 50 / 50 | 0 | 50 | 0 | 0 | 251 | 0.9851 |
| 10.0 | scale | 50 / 50 | 0 | 50 | 0 | 0 | 396 | 0.9837 |
| 0.1 | 0.01 | 50 / 50 | 0 | 50 | 0 | 0 | 186 | 1.0000 |
| 1.0 | 0.01 | 50 / 50 | 0 | 50 | 0 | 0 | 292 | 1.0000 |
| 10.0 | 0.01 | 50 / 50 | 0 | 50 | 0 | 0 | 373 | 1.0000 |

All six eligible under the E5 rules. Support fraction counts training samples,
not selected features; all 150 predictors remain available.

Warning review:
- 300 occurrences of one FutureWarning: SVC probability parameter deprecated
  in sklearn 1.9, removal scheduled for 1.11.
- No additional captured categories/messages reported; probability=False and
  decision_function used, so calibration was not added.
- No convergence warnings, failed fits or iteration-limit events.
- Preserve warning_folds=50 per candidate. No rerun solely to remove this warning.

Highest-mean E5 candidate: `e5_rbf_svm_c_10_gamma_0_01`, mean 0.812385,
SD 0.013559. Versus L1 leader: -0.049726, 0/10 improved, difference range
-0.075553 to -0.034599. Versus sparse L1: -0.040478, 0/10 improved.
Versus E3: -0.010597, 1/10 improved.

Findings and decision at E5 completion:
- All six lost to both L1 references in all ten repetitions; all had lower mean than E3.
- gamma=0.01 had higher mean at each C. Its C=10 versus C=1 gap was only 0.000067.
- Stop six-setting study, retain L1 references, do not promote SVM or automatically
  expand grid/calibration/kernels/features. High support fraction is not proof of overfitting.
- No E5 submission outcome recorded.

Limitations: six settings only, not all nonlinear models; complete-pipeline comparison;
reused development observations/splits; no independence, confidence-interval or leaderboard claim.

Artifacts: outputs/results/e5_rbf_run_01/, run_manifest.json,
verified_validation_assignments.csv, reference_repeat_scores.csv, e5_comparison.csv,
e5_paired_repeat_scores.csv, e5_fold_diagnostics.csv, e5_decision.json,
six candidate configs/diagnostics/decision-score tables, source/tests and warning review.
Preserve the original execution version and the later organization history separately.

## E5b — Bounded shallow XGBoost comparison

Status: completed; all six configurations eligible.

Question: Can shallow, regularised boosted trees outperform the retained L1 candidates
on the same saved validation splits?

Protocol:
- Integrated CLI: `python -m scripts.run_e5b`.
- All 150 predictors; raw-feature XGBClassifier pipeline; no scaling or selector.
- max_depth in {1,2}; n_estimators in {50,100,200}; six configurations, 300 fits.
- Load saved E4 assignments and verify both L1 references used matching splits.
- Same ten repeats / five folds / mean-fold-AUC aggregation.
- Fixed: booster=gbtree, objective=binary:logistic, eval_metric=auc,
  tree_method=hist, learning_rate=0.05, min_child_weight=5, reg_lambda=10.0,
  reg_alpha=0, gamma=0, subsample=1.0, colsample_bytree=1.0,
  random_state=42, n_jobs=1, verbosity=1.
- No early stopping, pruning, eval_set, calibration or test-data prediction.
- Reuse existing ValidationResult for score aggregation; training-only loader.

Environment and implementation notes:
- Initial macOS import failed because libomp.dylib was unavailable; the subsequent
  real-library smoke test passed after the installation issue was resolved.
- Smoke test: depth 1, 50 rounds, 200 train / 50 validation rows,
  diagnostic fold AUC 0.9106280193236715, 50 fitted rounds, zero captured warnings.
- This was one fold, not repeated-CV mean or leaderboard feedback.
- The frozen six-setting grid was retained; the smoke fold was refitted in the full run.
- A macOS test compared resolved /private/var paths with unresolved /var paths;
  normalizing the fixture root with Path(...).resolve() corrected the test expectation,
  not the model/data/splits.

| Depth | Trees | Mean CV AUC | Repeat SD | Min repeat AUC | Max repeat AUC |
|---:|---:|---:|---:|---:|---:|
| 1 | 50 | 0.787117 | 0.014404 | 0.761198 | 0.809572 |
| 1 | 100 | 0.804358 | 0.015245 | 0.771887 | 0.825582 |
| 1 | 200 | 0.808197 | 0.017836 | 0.772094 | 0.833691 |
| 2 | 50 | 0.833019 | 0.014400 | 0.807568 | 0.857621 |
| 2 | 100 | 0.839271 | 0.010432 | 0.821956 | 0.858252 |
| 2 | 200 | 0.833426 | 0.014514 | 0.811318 | 0.859844 |

| Depth | Trees | Mean difference vs L1 leader | Difference SD | Improved / worse | Mean difference vs sparse L1 | Difference SD | Improved / worse |
|---:|---:|---:|---:|---|---:|---:|---|
| 1 | 50 | -0.074995 | 0.018410 | 0 / 10 | -0.065746 | 0.019223 | 0 / 10 |
| 1 | 100 | -0.057754 | 0.018556 | 0 / 10 | -0.048506 | 0.018745 | 0 / 10 |
| 1 | 200 | -0.053915 | 0.018472 | 0 / 10 | -0.044667 | 0.018581 | 0 / 10 |
| 2 | 50 | -0.029092 | 0.014534 | 0 / 10 | -0.019844 | 0.016663 | 2 / 8 |
| 2 | 100 | -0.022840 | 0.012873 | 1 / 9 | -0.013592 | 0.014159 | 2 / 8 |
| 2 | 200 | -0.028686 | 0.016277 | 0 / 10 | -0.019437 | 0.015063 | 1 / 9 |

No ties recorded. All six completed 50 valid fits each; zero failed folds and
zero captured Python warnings. No candidate required warning review.

Best within original E5b:
- `e5b_xgboost_depth_2_trees_100`.
- Mean 0.839271; SD 0.010432.
- Versus L1 leader: mean -0.022840, range -0.039438 to +0.005208, 1 / 9 wins/losses.
- Versus sparse L1: mean -0.013592, range -0.045381 to +0.001681, 2 / 8.

Findings:
- Depth 2 exceeded depth 1 at each matching tree count.
- Depth 1 improved across tested counts; depth 2 peaked at 100 among tested counts.
- No setting exceeded either L1 reference in mean.

Decision at original E5b completion: stop the six-setting grid, retain L1 as leader,
and preserve the negative finding. Subsequent extensions below were separate,
adaptive decisions, not retrospectively part of a larger preregistered grid.

Limitations: six shallow settings with several parameters fixed; no claim about all
XGBoost configurations. Single-fold smoke AUC is not comparable with repeated-CV mean.
No global optimum, independent final estimate or leaderboard outcome established.

Artifacts: outputs/results/e5b_xgb_run_01/, resolved config, saved split copy,
reference scores, manifest, e5b_comparison.csv, e5b_paired_repeat_scores.csv,
e5b_fold_diagnostics.csv, e5b_decision.json, candidate configs, fitted booster configs,
probability tables and standard fold_scores/repeat_scores/summary files.

## E5b extension 01 — Depth 4, 1000 boosting rounds

Status: completed.

Question: Does jointly increasing tree depth and boosting rounds improve the previous
fixed-parameter XGBoost results?

Protocol:
- Proposed after reviewing the six-setting E5b results.
- max_depth=4; n_estimators=1000; all other original E5b settings unchanged.
- Same saved 50 folds; one additional candidate, 50 fits.
- Fixed rounds, no early stopping/pruning.
- Configuration: configs/e5b_xgboost_ext01.json.

Candidate: `e5b_xgboost_ext01_depth_4_trees_1000`.

| Mean CV AUC | Repeat SD | Min repeat AUC | Max repeat AUC |
|---:|---:|---:|---:|
| 0.828091 | 0.018197 | 0.792248 | 0.853693 |

| Reference | Mean paired difference | Difference SD | Min difference | Max difference | Improved / worse / tied |
|---|---:|---:|---:|---:|---|
| L1 C=1.0 | -0.034020 | 0.019259 | -0.072105 | -0.008868 | 0 / 10 / 0 |
| L1 C=0.1 | -0.024772 | 0.016792 | -0.045790 | +0.002006 | 1 / 9 / 0 |

- All 50 fits valid; zero failed folds and captured warnings.
- Mean AUC was 0.011180 lower than the original E5b leader; repeat SD was larger.
- Direct paired win/loss counts against the original XGBoost leader were not supplied.

Decision at extension 01 completion: do not promote; retain depth 2 / 100 as the
then-best XGBoost setting and L1 as overall leader. Stop this single-setting extension.

Limitations: depth and rounds changed together; the result does not isolate either
parameter's effect or prove overfitting. Adaptive development study, not independent evaluation.

Artifacts: outputs/results/e5b_xgb_ext01_depth4_trees1000/ and its standard E5b outputs.

## E5b extension 02 — Joint XGBoost tuning with Optuna

Status: completed; 30/30 trials COMPLETE and eligible. Trial 20 selected.
Candidate snapshot creation remains to be confirmed separately under E7.

Question: Can joint adjustment of parameters fixed in earlier XGBoost studies improve
mean development ROC AUC over the retained L1 candidates?

Study design:
- Adaptive follow-up after E5b and extension 01, not independent confirmation.
- 30 trial attempts total, including two queued historical reference settings.
- Each completed trial uses all 50 saved folds: 1500 valid fits reported.
- TPESampler(seed=42, n_startup_trials=10), direction=maximize.
- Optuna n_jobs=1; XGBoost n_jobs=1 and model_seed=42.
- No pruning (NopPruner), early stopping, eval_set or automatic budget extension.
- Objective: mean across ten repeat-level means of five fold AUCs.
- Same raw predictors, row order and saved split assignments; no scaling/selector.
- All captured warnings require review; partial/failed candidates are not scored
  by averaging only successful folds. No rejected trials occurred in this run.
- SQLite study and per-trial artifacts retained; no automatic resume was used.

Search space:

| Parameter | Search domain | Sampling |
|---|---|---|
| max_depth | 1–4 | Integer |
| n_estimators | 50,100,200,500,1000 | Categorical |
| learning_rate | 0.01–0.2 | Log scale |
| min_child_weight | 1–20 | Log scale |
| reg_lambda | 0.1–100 | Log scale |
| subsample | 0.6–1.0 | Linear |
| colsample_bytree | 0.5–1.0 | Linear |

Other fixed settings: booster=gbtree, objective=binary:logistic, eval_metric=auc,
tree_method=hist, reg_alpha=0, gamma=0, verbosity=1, and model/randomness settings above.
Sampler seed and model seed were fixed separately; neither was an optimized parameter.

Reference reproduction checks:
- Trial 0 queued depth 2 / 100 with the original fixed parameters:
  mean AUC 0.8392713888366062, reproducing original E5b.
- Trial 1 queued depth 4 / 1000 with original fixed parameters:
  mean AUC 0.8280912123303427, reproducing extension 01.
- Both reference trials were refitted and counted within the 30-trial budget.

All-trial summary:
- All 30 trials COMPLETE; all 1500 fits valid; no failed folds or captured Python warnings.
- 14/30 trials had a mean above the L1 C=1.0 reference.
- 20/30 trials had a mean above the L1 C=0.1 reference.
- These are counts among adaptively selected trials, not independent replications.
- The following table omits five searched parameters for space; depth/rounds alone
  do not identify a model. Preserve full-precision trial parameters separately.

| Trial | Max depth | Trees | Mean CV AUC | Repeat SD | Mean difference vs L1 leader | Improved / worse repeats |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 2 | 100 | 0.839271 | 0.010432 | -0.022840 | 1 / 9 |
| 1 | 4 | 1000 | 0.828091 | 0.018197 | -0.034020 | 0 / 10 |
| 2 | 2 | 50 | 0.784454 | 0.011689 | -0.077658 | 0 / 10 |
| 3 | 4 | 50 | 0.861304 | 0.012874 | -0.000808 | 5 / 5 |
| 4 | 2 | 200 | 0.851715 | 0.014225 | -0.010397 | 2 / 8 |
| 5 | 4 | 50 | 0.853306 | 0.010223 | -0.008805 | 2 / 8 |
| 6 | 2 | 50 | 0.797905 | 0.016864 | -0.064207 | 0 / 10 |
| 7 | 4 | 1000 | 0.776743 | 0.017531 | -0.085368 | 0 / 10 |
| 8 | 1 | 200 | 0.780406 | 0.011870 | -0.081705 | 0 / 10 |
| 9 | 1 | 200 | 0.818531 | 0.012257 | -0.043580 | 0 / 10 |
| 10 | 4 | 50 | 0.871335 | 0.011479 | +0.009223 | 8 / 2 |
| 11 | 4 | 50 | 0.871008 | 0.012838 | +0.008896 | 7 / 3 |
| 12 | 3 | 50 | 0.861066 | 0.014529 | -0.001045 | 6 / 4 |
| 13 | 3 | 50 | 0.855894 | 0.016830 | -0.006218 | 2 / 8 |
| 14 | 4 | 200 | 0.863943 | 0.016778 | +0.001831 | 4 / 6 |
| 15 | 3 | 50 | 0.865334 | 0.008801 | +0.003222 | 7 / 3 |
| 16 | 3 | 50 | 0.863453 | 0.017550 | +0.001341 | 6 / 4 |
| 17 | 4 | 50 | 0.858072 | 0.015525 | -0.004040 | 3 / 7 |
| 18 | 4 | 500 | 0.846201 | 0.016241 | -0.015911 | 1 / 9 |
| 19 | 4 | 500 | 0.852410 | 0.014155 | -0.009701 | 3 / 7 |
| 20 | 4 | 50 | 0.881937 | 0.013949 | +0.019826 | 10 / 0 |
| 21 | 4 | 50 | 0.859896 | 0.016992 | -0.002216 | 4 / 6 |
| 22 | 4 | 50 | 0.872346 | 0.012090 | +0.010234 | 8 / 2 |
| 23 | 3 | 50 | 0.870344 | 0.012153 | +0.008232 | 7 / 3 |
| 24 | 3 | 50 | 0.873313 | 0.012738 | +0.011202 | 8 / 2 |
| 25 | 4 | 50 | 0.878294 | 0.013941 | +0.016182 | 9 / 1 |
| 26 | 3 | 50 | 0.867781 | 0.011307 | +0.005669 | 6 / 4 |
| 27 | 3 | 50 | 0.872332 | 0.012155 | +0.010220 | 7 / 3 |
| 28 | 4 | 100 | 0.878066 | 0.012679 | +0.015955 | 9 / 1 |
| 29 | 4 | 50 | 0.877573 | 0.012541 | +0.015461 | 9 / 1 |

Selected candidate:
- Trial number: 20.
- Experiment ID: `e5b_optuna_ext02_trial_0020_depth_4_trees_50`.
- Mean development CV AUC: 0.8819374908505344.
- Repeat SD: 0.0139490659001846.
- Repeat-level mean AUC range: 0.8545679360896752–0.8984597528075788.

Full-precision searched parameters:

```json
{
  "max_depth": 4,
  "n_estimators": 50,
  "learning_rate": 0.08316404045417083,
  "min_child_weight": 1.229594255114693,
  "reg_lambda": 0.1877052106059947,
  "subsample": 0.9920641940615834,
  "colsample_bytree": 0.7550128116345981
}
```

Preserve these together with the fixed constructor settings, library versions,
feature order, label definition and execution code. Do not round parameters when
reconstructing this candidate or confuse parameter JSON with fitted model weights.

Paired Trial 20 comparisons:

| Reference | Mean paired difference | Difference SD | Min difference | Max difference | Improved / worse / tied |
|---|---:|---:|---:|---:|---|
| L1 C=1.0 | +0.019826 | 0.012237 | +0.000862 | +0.042695 | 10 / 0 / 0 |
| L1 C=0.1 | +0.029074 | 0.013955 | +0.004069 | +0.049155 | 10 / 0 / 0 |

Values above are rounded for presentation; preserve raw precision in artifacts.
Trial 20 mean AUC was 0.042666 above the original E5b leader and 0.003644 above
Optuna Trial 25, the second-highest mean. Direct paired win/loss counts between
these XGBoost trials were not supplied and must not be inferred from their means.

Findings:
- Joint tuning found a higher-mean development candidate than the earlier fixed
  XGBoost settings and both L1 references.
- Trial 20 exceeded both L1 references in every one of the ten matched repetitions.
- Compared with original E5b settings, it used greater maximum depth, fewer rounds,
  higher learning rate, lower child-weight and lambda settings, and column sampling.
- This is a joint configuration result; no individual parameter's causal contribution
  was isolated. The initial fixed settings may have limited the earlier search,
  but a controlled one-parameter attribution was not performed.
- Nine of the ten highest-mean trials used 50 rounds. This is a search observation,
  not proof that 50 is optimal; 50 was also the lower tested round-count boundary.
- Successful joint search does not establish that TPE outperforms random search;
  no equal-budget sampler comparison was conducted.

Decision:
- Stop the completed 30-trial search without automatically adding trials.
- Retain Trial 20 as the current provisional development leader.
- Retain L1 C=1.0 as the main linear reference and L1 C=0.1 as the sparse alternative.
- Preserve all 30 trials, including lower-scoring results and historical reference reproductions.
- Freeze the full Trial 20 configuration and evidence under a separate snapshot version.
  Snapshot execution/verification is pending confirmation; do not state it is already archived.
- No automatic follow-up using pruning/early stopping is approved or completed.
  Either requires a specific new efficiency/training question and frozen protocol.
- Do not automatically generate submissions for all trials.

Limitations:
- The trial maximum was selected using the same development CV objective used to
  judge candidates; its mean is not an independent estimate and may be optimistic.
- Ten paired improvements do not remove model-selection bias or make repeated
  splits independent datasets. No formal significance claim is made.
- Model families have received unequal search budgets. This does not establish
  that boosted trees intrinsically outperform every equally tuned linear model.
- Search-space boundaries remain; no global optimum claimed.
- Starting nested CV after prior exploration does not create a new untouched labelled dataset.
- No actual XGBoost Kaggle score, final refit or independent final evaluation is recorded.

Artifacts:
- Study directory: outputs/results/e5b_optuna_ext02_run_01/.
- study.sqlite3, config.json, base_config.json, run_manifest.json.
- verified_split_indices.json, reference_repeat_scores.csv.
- optuna_trials.csv, optuna_comparison.csv, optuna_paired_repeat_scores.csv.
- best_parameters.json, optuna_decision.json.
- trials/trial_0000/ through trials/trial_0029/: full E5b-style diagnostics,
  probability tables, per-candidate settings, booster configs and standard score files.
- Selected directory: trials/trial_0020/.
- configs/e5b_optuna_ext02.json and exact executed source/dependency versions.

## Current retained candidates

| Role | Candidate | Mean development CV AUC | Repeat SD |
|---|---|---:|---:|
| Provisional overall leader | Optuna Trial 20, full configuration above | 0.881937 | 0.013949 |
| Main linear reference | All-feature L1 C=1.0 | 0.862112 | 0.012204 |
| Sparse alternative | All-feature L1 C=0.1 | 0.852864 | 0.010803 |

Earlier E3, SVM and XGBoost candidates remain historical references, not automatically
additional finalists. Final shortlist and submission choices remain to be documented.

## E6 — Individual extension: validation protocol design

Status: proposed; not confirmed or executed.

Question: How sensitive are validation schemes to row assignment, and how consistently
do they compare two fixed models with an observed development AUC gap?

Protocol to freeze before running:
- Stratified 80/20 hold-out and shuffled stratified 3-,5-,10-fold CV.
- Thirty repetitions, seeds 0–29.
- One fixed pipeline for score-spread comparisons.
- If comparing two candidates, choose both before the study and use matching splits.
  Trial 20 versus L1 C=1.0 is now a possible pair, not an already approved protocol.
- Keep model randomness fixed and preprocessing fold-local.
- State per-scheme training/validation sizes; schemes do not train on equal data amounts.
- Do not rerun Optuna or retune candidates separately for each validation scheme.
- New split seeds are another development stability study, not fresh independent data.

Outputs: distributions, means/sample SD/ranges; paired differences and ordering
consistency; discussion of observed gaps (including approximately 0.01 AUC if relevant).
Do not claim that the true gap is known or treat repeat SD as a confidence interval.

Write-up: Question, Protocol, Results, Conclusion; maximum two pages.
This study is separate from both the original C sweep and Optuna tuning.

## E7 — Candidate freeze and finalise evaluation

Status: candidate selection has begun; physical freeze confirmation, final evaluation,
final full-data XGBoost refit and final submission remain pending.

Current candidate-freeze record:
- Selected candidate: Optuna Trial 20 with full-precision parameters above.
- Proposed snapshot version: e5b_optuna_trial20_v1.
- Proposed local snapshot directory: outputs/frozen/e5b_optuna_trial20_v1/.
- Freeze instructions supplied: copy complete parameters and supporting study/trial
  evidence; create a snapshot SHA-256 manifest; refuse overwriting an existing snapshot.
- Execution outcome: not yet confirmed by the project owner.
- Parameter/evidence freeze is not the same as saving a final trained model.

Checklist:
- [x] Identify the current development leader and retain L1 references.
- [x] Record selected trial and full-precision searched parameters.
- [ ] Confirm snapshot creation, required artifacts and hashes.
- [ ] Freeze the final candidate shortlist and the next evaluation protocol.
- [ ] Decide/document whether a bounded nested-CV evaluation will be used.
- [ ] If nested CV is used, freeze inner candidates/search budget/selection rules
      before outer runs; distinguish a selection procedure from a fixed-model evaluation.
- [ ] Acknowledge that previous exploration used all labelled rows; no new untouched
      labelled dataset is created by starting nested CV later.
- [ ] Choose final submissions according to official competition/assignment rules.
- [ ] Refit the selected pipeline on all labelled rows using frozen settings,
      preserve the fitted model, feature order and full provenance.
- [ ] Record actual submission identifiers/timestamps/public scores only after submission.
- [ ] Complete report, extension, video and individual reflection.

A changed hyperparameter, feature procedure, pruning rule or early-stopping procedure
requires a new experiment/candidate version. Do not overwrite the frozen v1 record.

## Required record for each experiment

- Question/hypothesis and whether exploratory, adaptive or prospectively specified.
- Fixed/changed settings; full-precision parameters and complete pipeline.
- Data/feature order and label definition; configs, code revision and source hashes.
- Saved split assignments, split seeds and model/sampler seeds separately.
- Search space/budget, sampler/pruner/early-stopping rules where applicable.
- Mean fold AUC per repetition, mean/sample SD/range across repetitions.
- Full paired differences on matching splits; wins/losses/ties and tie tolerance.
- Completed/failed/rejected/pruned states; warnings and numerical diagnostics.
- Decisions/reasons, negative findings and explicit stopping rule.
- Snapshot status, later refit/model/submission status as separate milestones.
- Kaggle feedback only when actually submitted; no unattributed score claims.

## Immediate implementation order

1. Archive the completed E0–E5, original E5b, extension 01 and Optuna extension 02
   artifacts; preserve their separate protocols and exact executed code.
2. Verify Optuna manifest completion (30 finished trials), SQLite study and all
   Trial 20 parameters/diagnostics; keep the reproduced reference trials.
3. Execute/confirm the proposed Trial 20 v1 freeze snapshot, then verify its files
   and hash list. Do not mark it complete based only on supplied instructions.
4. Update the active candidate record to Trial 20; retain L1 C=1.0 and C=0.1.
5. Recover E1's missing actual public score/submission identifier/timestamp;
   confirm whether any later candidate was submitted before planning another upload.
6. Confirm whether E6 is the chosen individual extension and freeze its candidates,
   validation schemes, repetition counts and analysis rules before implementation.
7. Decide the E7 evaluation/shortlist/submission protocol before additional runs.
8. Do not automatically reopen searches or add pruning/early stopping. New work
   needs a specific unresolved question and bounded protocol, not just a higher-score target.
9. Complete final refit/submission and report deliverables only when their preceding
   selection/evaluation decisions are documented; record actual outcomes separately.
