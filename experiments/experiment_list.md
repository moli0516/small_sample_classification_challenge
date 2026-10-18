# Experiment List

## Scope and current status

This is an early-stage research roadmap, not a claim that model selection is complete.
Completed development studies include the all-feature L2 C sweep,
correlation-based feature selection, the selected-feature-count / C interaction,
and the planned four-setting L1 comparison.

* Data audit: completed; 250 labelled rows, 19,750 test rows, 150 numeric predictors.
* Identifier: `id`, excluded from predictors; target: `target`.
* ValidationRunner and SubmissionWriter tests: reported passing by the project owner.
* Baseline submission workflow: completed, as reported by the project owner;
  the actual public score, submission identifier, and timestamp remain to be recorded.
* E0–E4: completed for the studies documented below. The optional Elastic-net
  follow-up in E4 has not been run.
* Current highest-mean development candidate: all-feature L1 logistic regression,
  C=1.0; mean CV AUC 0.862112, repeat SD 0.012204.
* Sparse alternative: all-feature L1 logistic regression, C=0.1;
  mean CV AUC 0.852864, repeat SD 0.010803.
* E4 candidate submission status: not recorded.
* No verified Kaggle score is recorded here. The reported scores above 0.90 have not
  been identified as this project's submissions or other teams' results.
* Individual extension: validation protocol study proposed, not yet confirmed.
* All scores used for candidate selection remain development results;
  no independent final evaluation has been completed.

## Shared experimental rules

1. Use the same training data and row order across comparisons; verify saved split indices.
2. Unless studying validation itself, use shuffled stratified 5-fold CV with seeds 0–9.
3. Fit scaling and supervised feature selection inside each training fold only.
4. Keep estimator randomness fixed separately from the split seed.
5. Report mean fold AUC per repetition, then mean and sample SD across repetitions.
6. Compare candidates using paired repeat-level differences on identical splits.
7. Do not interpret repeated splits as independent datasets or their SD as a confidence interval.
8. Preserve configurations, scores, split indices, package versions, data hashes, and code revision.
9. Define the question, candidate settings, and stopping rule before each new experiment.
10. Treat scores used to choose models as development results, not independent final evaluation.
11. Do not run every candidate as a Kaggle submission or tune directly to public scores.
12. Keep course code, data, and generated artifacts private under the assignment rules.

## Overview


| ID | Investigation                                    | Status                            | Priority                            |
| -- | ------------------------------------------------ | --------------------------------- | ----------------------------------- |
| E0 | All-feature logistic regression: C sweep         | Completed                         | Preserve                            |
| E1 | End-to-end baseline submission                   | Completed                         | Record feedback                     |
| E2 | Correlation-based feature selection              | Completed                         | Preserve results                    |
| E3 | Interaction between selected feature count and C | Completed                         | Preserve results                    |
| E4 | L1 logistic regression; optional Elastic-net     | L1 completed; Elastic-net not run | Preserve results; retain candidates |
| E5 | RBF SVM comparison                               | Optional                          | Medium                              |
| E6 | Validation protocol extension                    | Proposed                          | Required if chosen                  |
| E7 | Candidate freeze and final evaluation            | Later                             | Required                            |

These IDs apply to this document. Earlier chat labels used a different numbering;
existing experiment IDs and output directories must not be renamed.

## E0 — Completed C sweep

Question: How does C affect a fixed all-feature logistic-regression baseline?

Fixed pipeline: `StandardScaler -> LogisticRegression`, solver `lbfgs`,
maximum iterations 2,000, all 150 predictors, no feature selection.


| C      | Mean CV AUC | Repeat SD |
| ------ | ----------- | --------- |
| 1.0    | 0.751411    | 0.019922  |
| 0.1    | 0.771935    | 0.013106  |
| 0.01   | 0.792885    | 0.011157  |
| 0.001  | 0.793999    | 0.009091  |
| 0.0001 | 0.790600    | 0.008026  |

Decision: retain C=0.001 as the provisional baseline; C=0.01 remains a close alternative.
This was a sequential exploratory study, not a fully preregistered grid.
Stop this sweep. Other parameters and model families have not yet been compared.

## E1 — Baseline submission

Status: completed, as reported by the project owner.

Question: Can the complete refit/predict/export/upload workflow run successfully?

Configuration:

- Pipeline: StandardScaler -> LogisticRegression.
- All 150 predictors; `id` excluded from modelling.
- C=0.001.
- Refit on all 250 labelled training rows.
- Generate predictions for all 19,750 test rows.
- Submission columns: `id,target`, aligned to the official template by ID.

Completed:

- [X]  Generate the baseline submission.
- [X]  Confirm submission format and ID alignment.
- [X]  Complete the Kaggle submission workflow.
- [X]  Save the submission, fitted pipeline, and configuration.

Results:

- Mean development CV AUC: 0.793999.
- Repeat-level CV standard deviation: 0.009091.
- Public leaderboard score: to be recorded.
- Kaggle submission identifier and timestamp: to be recorded.

Decision:

- Retain this submission as the initial external reference.
- Do not treat successful submission as evidence of model optimality.
- Proceed to E2: correlation-based feature selection.

Question: Can the complete refit/predict/export/upload workflow run successfully?

* Confirm the official metric and permitted prediction format.
* Refit a fresh all-feature C=0.001 pipeline on all 250 labelled rows.
* Predict test class-1 probabilities and align them with the official template by ID.
* Verify 19,750 rows and exactly the template columns `id,target`.
* Save submission, fitted pipeline, and configuration.
* Upload once and record the actual public score and submission identifier.

Stop after one successful baseline submission. Do not call this the final solution.

## E2 — Correlation-based feature selection

Status: completed.

Question: Does removing low-ranked features improve the baseline?

Protocol:

- Pipeline: CorrelationSelector -> StandardScaler -> LogisticRegression.
- Ranking: absolute Pearson correlation, fitted within each training fold.
- k in {5, 10, 20, 50, 150}.
- C fixed at 0.001.
- Stratified five-fold CV, repeated with seeds 0–9.
- Retain original column order after selection.
- Record selected features separately for each fold.

Results:


|   k | Mean CV AUC | Repeat SD | Mean difference vs k=150 | Improved / worse repeats |
| --: | ----------: | --------: | -----------------------: | -----------------------: |
|   5 |    0.795349 |  0.006414 |                +0.001350 |                    6 / 4 |
|  10 |    0.808618 |  0.008244 |                +0.014619 |                   10 / 0 |
|  20 |    0.798767 |  0.008428 |                +0.004769 |                    6 / 4 |
|  50 |    0.797398 |  0.008861 |                +0.003400 |                    4 / 6 |
| 150 |    0.793999 |  0.009091 |                 0.000000 |                  Control |

Findings:

- k=10 achieved the highest mean development AUC and improved over
  the all-feature baseline in all ten paired repetitions.
- k=5, 20, and 50 did not consistently improve across repetitions.
- k=50 had a positive mean difference despite performing worse in
  six of ten repetitions.
- k=150 reproduced the baseline repeat-level scores.
- The fold-selection frequencies are descriptive, not a bootstrap study.

Decision:

- Retain k=10 as the leading E2 candidate.
- Retain k=20 as the second-highest-mean candidate for E3.
- Stop this k sweep without adding nearby values.

Limitations:

- Only C=0.001 was tested in this experiment.
- Different folds may select different features.
- The results support a selection procedure, not the claim that exactly
  ten predictors are truly relevant.
- The highest score remains a development result used for model selection.

Artifacts:

- Per-configuration scores, settings, splits, and selected_features.json.
- E2 comparison and paired-difference tables.
- Feature-selection frequency table.
- AUC and paired-difference figures.

## E3 — Feature-count / C interaction

Status: completed.

Question: Is C=0.001 still appropriate after feature selection?

Protocol:

- Pipeline: CorrelationSelector -> StandardScaler -> LogisticRegression.
- k in {10, 20}, selected from the two highest mean E2 scores.
- C in {0.001, 0.01, 0.1}.
- Reuse the two C=0.001 configurations from E2.
- Run four new configurations: 200 additional CV fits.
- Retain the same data, row order, validation splits, and other settings.
- Stratified five-fold CV, repeated with seeds 0–9.
- Fit feature selection and scaling within each training fold.

Results:


|  k |     C | Mean CV AUC | Repeat SD | Mean difference vs E2 leader | Improved / worse repeats |
| -: | ----: | ----------: | --------: | ---------------------------: | ------------------------ |
| 10 | 0.001 |    0.808618 |  0.008244 |                     0.000000 | Control                  |
| 10 |  0.01 |    0.812664 |  0.007200 |                    +0.004046 | 10 / 0                   |
| 10 |   0.1 |    0.822982 |  0.008456 |                    +0.014364 | 10 / 0                   |
| 20 | 0.001 |    0.798767 |  0.008428 |                    -0.009851 | 2 / 8                    |
| 20 |  0.01 |    0.799816 |  0.010557 |                    -0.008801 | 3 / 7                    |
| 20 |   0.1 |    0.794235 |  0.013509 |                    -0.014382 | 2 / 8                    |

The E2 leader is k=10/C=0.001. Differences and win/loss counts in this
table are paired against that candidate, not against the all-feature model.

Within-k comparisons:


|  k | Change from C=0.001 | Mean paired AUC difference | Improved / worse repeats |
| -: | ------------------- | -------------------------: | ------------------------ |
| 10 | C=0.01              |                  +0.004046 | 10 / 0                   |
| 10 | C=0.1               |                  +0.014364 | 10 / 0                   |
| 20 | C=0.01              |                  +0.001049 | 5 / 5                    |
| 20 | C=0.1               |                  -0.004532 | 3 / 7                    |

Findings:

- k=10/C=0.1 achieved the highest mean development AUC among the
  six candidates.
- At k=10, both tested increases in C improved on C=0.001 in all
  ten matching repetitions.
- At k=20, changing C to 0.01 gave mixed results, while C=0.1
  reduced the mean score.
- The effect of C therefore depended on the selected feature count.
- The highest-mean candidate did not have the lowest repeat SD.
- The supplied summary does not directly establish the win/loss count
  for k=10/C=0.1 versus k=10/C=0.01.

Leading candidate:

- Experiment ID: e3_correlation_k_10_c_0_1.
- Mean development CV AUC: 0.822982.
- Repeat SD: 0.008456.
- Mean paired improvement over the E2 leader: +0.014364;
  improved in 10/10 repetitions.
- Mean paired improvement over the all-feature C=0.001 baseline:
  +0.028983; improved in 10/10 repetitions.

Decision:

- Retain k=10/C=0.1 as the provisional leading pipeline.
- Preserve the E2 leader and all-feature baseline as reference models.
- Stop the planned E3 comparison and proceed to E4.
- Consider one candidate submission if it has not already been made;
  record the actual outcome separately.

Limitations:

- E3 is an adaptive follow-up informed by E2, not independent confirmation.
- The same labelled observations were reused across experiments.
- These scores were used for model selection and remain development results.
- C=0.1 is the largest value tested in E3, so the full performance peak
  has not been located.
- The results do not establish a globally optimal k/C combination or
  guarantee a corresponding public or private leaderboard score.

Artifacts:

- Four new experiment directories with configurations, scores,
  split indices, and selected_features.json.
- Two reused E2 experiment directories.
- e3_comparison.csv and e3_paired_repeat_scores.csv.
- Mean-AUC, repeat-trajectory, and paired-difference figures.

## E4 — Sparse regularisation

Status: completed for the four planned L1 settings.
The optional Elastic-net follow-up has not been run.

Question: Can model-based sparsity compete with explicit correlation filtering?

Protocol:

- Pipeline: StandardScaler -> L1 LogisticRegression.
- All 150 predictors; no correlation-based feature selection.
- C in {0.01, 0.1, 1.0, 10.0}.
- Shuffled stratified five-fold CV, repeated with split seeds 0–9.
- Four configurations, 50 CV fits each: 200 CV fits in total.
- Fit scaling within each training fold.
- Keep estimator randomness fixed separately from the split seed.
- Compare mean fold AUC per repetition on matching splits.
- Record exact and effective nonzero coefficient counts, feature frequencies,
  iteration counts, convergence warnings, and iteration-limit flags.

Configuration provenance:

- The intended numerical settings were solver=saga, model_seed=42,
  tol=1e-4, and max_iter=10000.
- Verify these values, the installed scikit-learn interface, and the effective
  L1 parameterisation against the saved configurations and training code.
- The supplied result tables do not independently verify those settings.
- A notebook validation initially failed because its regularisation-field mapping
  did not match the saved configuration schema. Resolve this by checking the
  actual schema and estimator construction, not by disabling validation or
  relabelling an experiment.
- Preserve package versions, data hashes, split indices, and code revision.

Reference candidates:

- All-feature L2 baseline: C=0.001; mean development AUC 0.793999.
- E3 leader: CorrelationSelector(k=10) -> StandardScaler ->
  L2 LogisticRegression(C=0.1); mean development AUC 0.822982.
- Reference AUC values and paired comparisons below use the supplied
  repeat-level score table.

Results:


| L1 C | Mean CV AUC | Repeat SD | Minimum repeat AUC | Maximum repeat AUC |
| ---: | ----------: | --------: | -----------------: | -----------------: |
| 0.01 |    0.500000 |  0.000000 |           0.500000 |           0.500000 |
|  0.1 |    0.852864 |  0.010803 |           0.838038 |           0.876227 |
|  1.0 |    0.862112 |  0.012204 |           0.844883 |           0.878112 |
| 10.0 |    0.817571 |  0.015909 |           0.790680 |           0.842070 |

The minimum and maximum above refer to repetition-level mean fold AUC,
not individual-fold AUC.

Paired comparisons against the all-feature L2 baseline:


| L1 C | Mean paired AUC difference | Difference SD | Improved / worse / tied repeats |
| ---: | -------------------------: | ------------: | ------------------------------- |
| 0.01 |                  -0.293999 |      0.009091 | 0 / 10 / 0                      |
|  0.1 |                  +0.058865 |      0.008286 | 10 / 0 / 0                      |
|  1.0 |                  +0.068113 |      0.012379 | 10 / 0 / 0                      |
| 10.0 |                  +0.023573 |      0.016327 | 9 / 1 / 0                       |

Paired comparisons against the E3 leader:


| L1 C | Mean paired AUC difference | Difference SD | Improved / worse / tied repeats |
| ---: | -------------------------: | ------------: | ------------------------------- |
| 0.01 |                  -0.322982 |      0.008456 | 0 / 10 / 0                      |
|  0.1 |                  +0.029882 |      0.010501 | 10 / 0 / 0                      |
|  1.0 |                  +0.039130 |      0.015253 | 10 / 0 / 0                      |
| 10.0 |                  -0.005411 |      0.019133 | 3 / 7 / 0                       |

Direct comparison of the two retained L1 candidates:

Difference is defined as AUC(C=1.0) minus AUC(C=0.1) within each
matching repetition.


| Metric                          |     Value |
| ------------------------------- | --------: |
| Mean paired AUC difference      | +0.009248 |
| Sample SD of paired differences |  0.014681 |
| Improved / worse / tied repeats | 8 / 2 / 0 |
| Minimum paired difference       | -0.016263 |
| Maximum paired difference       | +0.028807 |

C=1.0 performed worse than C=0.1 for split seeds 7 and 8.
The ten repetitions reuse the same observations and are not ten
independent datasets.

Sparsity and numerical diagnostics:


| L1 C | Mean exact nonzero count | Mean effective nonzero count | Effective count range | All-zero folds | Warning folds | Iteration-limit folds | Maximum n_iter |
| ---: | -----------------------: | ---------------------------: | --------------------: | -------------: | ------------: | --------------------: | -------------: |
| 0.01 |                     0.00 |                         0.00 |                  0–0 |             50 |             0 |                     0 |              3 |
|  0.1 |                    15.76 |                        15.76 |                11–22 |              0 |             0 |                     0 |            127 |
|  1.0 |                    74.96 |                        74.96 |                64–85 |              0 |             0 |                     0 |           1684 |
| 10.0 |                    97.06 |                        97.06 |               85–107 |              0 |             0 |                     0 |           3978 |

- Exact and effective nonzero counts matched in every supplied fold record.
- The effective-zero threshold must be preserved with the analysis settings.
- No convergence-warning or iteration-limit flags were recorded in any
  of the 200 fits.
- All four configurations had numerical_review_required=False.
- These are recorded diagnostics, not proof that the warning-capture
  implementation or all numerical behaviour has been independently audited.

Feature-selection frequency:

Each frequency is the number of folds with a nonzero coefficient divided
by 50 for that C setting.


| Metric                                    | C=0.1 | C=1.0 |
| ----------------------------------------- | ----: | ----: |
| Features nonzero in at least one fold     |    61 |   145 |
| Features nonzero in every fold            |     4 |    11 |
| Features nonzero in at least 90% of folds |     7 |    21 |
| Features never nonzero                    |    89 |     5 |

The supplied feature-frequency totals reproduce the mean nonzero counts:
788 / 50 = 15.76 for C=0.1, and 3748 / 50 = 74.96 for C=1.0.

For C=0.1, the features with at least 90% nonzero frequency were:

- 46, 80, 100, 138: 50/50 folds.
- 30, 67: 49/50 folds.
- 108: 48/50 folds.

For C=1.0, the features nonzero in all 50 folds were:

- 6, 17, 22, 30, 46, 80, 85, 100, 115, 138, 141.

Features 46, 80, 100, and 138 were nonzero in every fold under both
C=0.1 and C=1.0.

Feature 67 illustrates that selection is not nested across the tested C values:
its nonzero frequencies were 98% at C=0.1, 0% at C=1.0, and 92% at C=10.0.
The frequency table alone does not establish the cause.

Findings:

- C=1.0 achieved the highest mean development AUC among the four L1 settings.
- Both C=0.1 and C=1.0 improved over the all-feature L2 baseline and the
  E3 leader in all ten paired repetitions.
- C=1.0 exceeded C=0.1 by 0.009248 mean AUC, winning eight of ten repetitions.
- C=0.1 retained substantially fewer nonzero coefficients and remains
  a useful sparse alternative.
- C=0.01 produced zero feature coefficients in all 50 folds and AUC=0.5
  throughout; preserve this negative finding.
- C=10.0 retained more coefficients but performed worse on average than
  C=0.1 and C=1.0, and worse than the E3 leader in seven of ten repetitions.
- The number of nonzero coefficients and their identities are distinct:
  an average count of 74.96 does not imply a fixed set of 75 features.
- Nonzero frequency is descriptive. It does not establish coefficient
  magnitude, sign consistency, causal relevance, or the true number
  of informative predictors.

Retained candidates:


| Role                   | Experiment ID | Configuration         | Mean development CV AUC | Repeat SD |
| ---------------------- | ------------- | --------------------- | ----------------------: | --------: |
| Highest-mean candidate | e4_l1_c_1     | All-feature L1, C=1.0 |                0.862112 |  0.012204 |
| Sparse alternative     | e4_l1_c_0_1   | All-feature L1, C=0.1 |                0.852864 |  0.010803 |

Decision:

- Retain all-feature L1 C=1.0 as the current provisional leading pipeline.
- Retain all-feature L1 C=0.1 as the sparse alternative.
- Preserve the E3 leader, E2 leader, and all-feature L2 baseline as references.
- Stop the planned four-setting L1 sweep; do not add nearby C values
  solely to optimise this development score.
- Do not automatically launch Elastic-net or E5.
- Confirm configuration provenance before refitting a retained candidate.
- Consider one candidate submission if no E4 candidate has already been
  submitted; record its actual outcome separately.
- Do not turn the aggregated frequency table into a fixed global selector
  and reuse the existing CV scores as evaluation of that new procedure.

Limitations:

- E4 follows earlier exploration using the same 250 labelled observations.
- The highest score was selected using development CV and is not an
  independent final performance estimate.
- Paired repetitions reuse observations; their SD is not a confidence interval.
- Comparing E4 against L2 reference pipelines is a comparison of complete
  procedures, not a pure solver-controlled penalty ablation.
- Feature-frequency summaries do not measure coefficient magnitude,
  sign stability, or the predictive value of removing individual features.
- Any new frequency-based selector would require fold-local selection
  and an appropriate evaluation protocol.
- No global optimum or public/private leaderboard outcome is established.
- No verified E4 Kaggle submission score is recorded.

Artifacts:

- Experiment directories:
  - e4_l1_c_0_01
  - e4_l1_c_0_1
  - e4_l1_c_1
  - e4_l1_c_10
- Four-configuration comparison summary.
- Per-feature nonzero-frequency table.
- Per-fold sparsity and convergence-diagnostic table.
- e4_paired_repeat_scores.csv.
- e4_direct_paired_comparison.csv, derived from the supplied repeat scores.
- Archive configurations, raw scores, split indices, coefficient records,
  package versions, data hashes, and code revision; verify completeness.

Optional follow-up — not run:

Elastic-net may be considered only if a new question justifies it.
The previously proposed bounded settings were:

- Solver: saga.
- C in {0.1, 1.0}.
- l1_ratio fixed at 0.5.

Confirm or revise that protocol explicitly before execution.
Elastic-net is not part of the completed E4 results.

## E5 — Bounded RBF SVM comparison

Status: proposed; freeze this protocol before running.

Question:
Does an all-feature RBF SVM improve development ROC AUC over the
retained L1 logistic-regression candidates under identical validation splits?

Purpose:

- Compare a bounded nonlinear alternative against the current linear candidates.
- Do not simultaneously search feature subsets, probability-calibration methods,
  class weights, or additional kernels.
- Treat this as another development study, not independent final evaluation.

Primary reference:

- Experiment ID: e4_l1_c_1.
- All-feature L1 logistic regression, C=1.0.
- Mean development CV AUC: 0.862112.
- Repeat SD: 0.012204.

Secondary references:

- e4_l1_c_0_1:
  all-feature L1 logistic regression, C=0.1;
  mean development CV AUC 0.852864, repeat SD 0.010803.
- e3_correlation_k_10_c_0_1:
  correlation selection with k=10 and L2 logistic regression, C=0.1;
  mean development CV AUC 0.822982, repeat SD 0.008456.

Use saved repeat-level scores for paired comparisons.
Do not reconstruct paired differences from rounded summary values.

Pipeline:

- StandardScaler -> SVC.
- All 150 predictors.
- Exclude id; target remains target.
- No CorrelationSelector, PCA, or other feature selection.
- Fit StandardScaler inside each training fold only.

Candidate settings:


| Experiment ID                | Kernel |    C | Gamma |
| ---------------------------- | ------ | ---: | ----- |
| e5_rbf_svm_c_0_1_gamma_scale | rbf    |  0.1 | scale |
| e5_rbf_svm_c_1_gamma_scale   | rbf    |  1.0 | scale |
| e5_rbf_svm_c_10_gamma_scale  | rbf    | 10.0 | scale |
| e5_rbf_svm_c_0_1_gamma_0_01  | rbf    |  0.1 | 0.01  |
| e5_rbf_svm_c_1_gamma_0_01    | rbf    |  1.0 | 0.01  |
| e5_rbf_svm_c_10_gamma_0_01   | rbf    | 10.0 | 0.01  |

Fixed estimator settings:

- probability=False.
- class_weight=None.
- tol=1e-3.
- max_iter=10000.
- shrinking=True.
- cache_size=200 MB.
- random_state=42 for configuration consistency.
- With probability=False, SVC does not use random_state for probability fitting.
- Verify compatibility with the installed scikit-learn version before execution.
- Save all resolved estimator parameters, not only parameters changed in the grid.

Prediction score:

- Use decision_function for validation ROC AUC.
- Verify that classes_ is [0, 1] and that larger scores correspond to target=1.
- Do not compute the primary AUC from predict() labels.
- Do not label decision scores as probabilities.
- Record score_method="decision_function" in configurations and artifacts.
- Preserve the existing predict_proba path for the logistic-regression references;
  AUC comparisons do not require both models to use the same score scale.

Validation protocol:

- Use the same 250 labelled rows, predictor order, and row order as E4.
- Reuse and verify the saved shuffled stratified five-fold split assignments
  for split seeds 0–9.
- Do not assume that matching seed values alone prove identical splits.
- Six configurations × ten repetitions × five folds = 300 planned CV fits.
- Existing reference scores are reused; no reference refitting is required
  if data and split provenance match.
- Compute AUC separately for each validation fold.
- Average the five fold AUC values within each repetition.
- Report the mean and sample SD across the ten repetition-level means.
- Report minimum and maximum repetition-level mean AUC.
- Do not replace this metric with a pooled out-of-fold AUC.

Paired comparisons:
For each candidate and each repetition, compute:

1. Difference versus e4_l1_c_1: primary comparison.
2. Difference versus e4_l1_c_0_1: secondary comparison.
3. Difference versus the E3 leader: historical reference.

For each reference, report:

- Mean paired AUC difference.
- Sample SD of paired differences.
- Minimum and maximum paired differences.
- Improved / worse / tied repetition counts.
- The full repeat-level difference table.

Use a fixed tie tolerance of 1e-12 for win/loss/tie classification.
Do not interpret repetition SD as a confidence interval or treat repeated
splits as independent datasets.

Implementation checks before the full run:

- Extend ValidationRunner to accept an explicit score method if required.
- Fail clearly if the requested scoring method is unavailable;
  do not silently fall back to predict().
- Confirm that decision scores are finite and one-dimensional for this
  binary task, with one score per validation row.
- Verify positive-class orientation.
- Test that scaling is fitted on training-fold rows only.
- Confirm that existing logistic-regression scoring behaviour is unchanged.
- Verify that failed fits cannot silently become zero-AUC or missing-fold results.
- Keep a scoring-path smoke test separate from the six-configuration comparison.
- If the smoke test causes a scientific protocol change, document the
  amendment before launching the full comparison.

Per-fold diagnostics:

- AUC.
- Training and validation row counts.
- Number of support vectors per class and total support-vector count.
- Support-vector count divided by training-row count.
- fit_status_.
- n_iter_, when exposed by the installed version.
- Convergence warnings and other captured warnings.
- Fit duration.
- Number of unique validation decision scores.
- Validation decision-score minimum, maximum, and standard deviation.
- Requested gamma and fitted numeric gamma, if available through a
  documented or explicitly version-sensitive diagnostic.

Do not interpret support-vector count as selected-feature count.
SVC retains all 150 input predictors in this experiment.

Failure and numerical-review rules:

- Preserve all warnings and failed-fold records.
- A candidate is ineligible for leader selection if any fold fails,
  has non-finite scores, or reports non-convergence.
- Do not rank a candidate using only its successful folds.
- Keep partial summaries clearly marked as incomplete.
- A numerical rerun requires a documented amendment.
- Do not silently change max_iter, tol, C, or gamma during the experiment.

Candidate-selection rule:

- Consider only candidates with all 50 valid, converged fold results.
- Rank eligible candidates by mean repetition-level AUC.
- Use a deterministic tie-breaker for means equal within 1e-12:
  lower C first, then gamma="scale".
- This selects the E5 leader, not automatically the overall project leader.
- Promote the E5 leader to the provisional overall leader only if its
  mean AUC exceeds the E4 leader by more than 1e-12.
- Report paired differences and win/loss counts even if they qualify
  or weaken the interpretation of the mean ordering.
- Do not use the public leaderboard to choose among the six settings.
- Do not describe a small positive development difference as statistically
  established superiority.

Stopping rule:

- Complete the six listed configurations, except where a documented
  implementation failure requires pausing.
- Stop after the bounded comparison and its diagnostics are reviewed.
- Do not automatically add nearby C or gamma values, feature selection,
  another kernel, or probability calibration.
- If the best setting lies on the grid boundary, record that limitation;
  it is not automatic permission to expand the search.

Submission policy:

- No automatic submission is part of this CV experiment.
- If an SVM candidate is retained, first confirm the official permitted
  prediction format.
- Raw decision scores must not be exported as class probabilities.
- If probabilities are required, define a separate fold-local calibration
  procedure and evaluate that complete procedure before submission.
- Do not tune a score-to-probability transformation using leaderboard feedback.
- Record any later submission configuration, identifier, timestamp,
  and actual score separately.

Planned artifacts:

- Six experiment directories with configs, scores, and split assignments.
- Per-fold decision scores aligned to validation-row identifiers.
- Per-fold diagnostic and warning records.
- e5_comparison.csv.
- e5_paired_repeat_scores.csv.
- e5_fold_diagnostics.csv.
- Mean-AUC, repeat-trajectory, and paired-difference figures.
- Package versions, data hashes, code revision, and protocol amendments.

Interpretation limits:

- E5 compares complete RBF SVM and logistic-regression procedures;
  it does not isolate nonlinearity as the sole cause of any difference.
- Only six RBF SVM settings are evaluated.
- A negative result does not rule out every nonlinear model.
- A positive result remains a development finding selected on reused data.
- No untouched labelled evaluation set is created by adding another experiment.
- No public or private leaderboard performance is guaranteed.

Question: Does a nonlinear alternative outperform the linear candidates?

* Pipeline: `StandardScaler -> RBF SVM`.
* Initially retain all 150 predictors.
* C in {0.1, 1, 10}; gamma in {"scale", 0.01}.
* Maximum: six configurations; no simultaneous feature-selection search.
* Before running, extend and test the runner if using `decision_function` instead
  of `predict_proba`. Record which prediction score is used.

Do not switch to a new model solely because another team reports a high public score.
Probability generation for submission must be defined separately if this model is retained.

## E6 — Individual extension: validation protocol design

Status: proposed. Confirm this extension before implementation.

Question: How sensitive are validation schemes to row assignment, and how
consistently do they compare two models with a small observed AUC gap?

Protocol to freeze before running:

* Stratified 80/20 hold-out and shuffled stratified 3-, 5-, and 10-fold CV.
* Thirty repetitions using seeds 0–29.
* One fixed pipeline for score-spread comparisons.
* If comparing two candidates, fix both before the study and use matching splits.
* Keep model randomness fixed and preprocessing fold-local.
* State training/validation sample sizes: schemes do not train on equal amounts of data.
* Do not retune candidates separately to make one validation scheme look better.

Outputs:

* Score distributions, means, sample SDs, and ranges.
* Paired model differences and ordering consistency.
* Discussion of whether an observed 0.01 AUC gap is persuasive.
* No claim that the true model-performance gap is known.

Write-up: Question, Protocol, Results, Conclusion; maximum two pages.
This study is separate from the completed C sweep.

## E7 — Freeze candidates and finalise evaluation

* Shortlist a small number of complete pipelines using recorded local evidence.
* Decide and document whether to run a bounded nested-CV evaluation.
* If used, freeze the inner candidate set and selection rule before the outer run.
* Distinguish evaluation of a selection procedure from evaluation of one fixed model.
* Acknowledge that prior exploration used all labelled rows; no new untouched
  labelled dataset is created by starting nested CV later.
* Choose the final submission(s) according to the official competition rules.
* Refit the chosen pipeline on all labelled rows and archive the artifacts.
* Complete the core report, extension, video, and individual reflection.

## Required record for each experiment

* Question / hypothesis.
* Fixed settings and changed settings.
* Configuration and code revision.
* Split assignment and model seed.
* Mean AUC, repeat SD, range, and paired differences.
* Convergence warnings or failed runs.
* Decision and reason, including negative findings.
* Kaggle feedback only when actually submitted.

## Immediate implementation order

1. Archive the completed E0–E4 results and confirm configuration provenance,
   saved split assignments, package versions, data hashes, and code revision.
2. Preserve all four E4 L1 configurations, including the all-zero C=0.01 result.
3. Retain e4_l1_c_1 as the current highest-mean development candidate and
   e4_l1_c_0_1 as the sparse alternative.
4. Record the missing E1 public score, submission identifier, and timestamp;
   do not duplicate an already completed baseline submission.
5. Confirm whether an E4 candidate has already been submitted. If not,
   consider one documented candidate submission rather than submitting
   every tested configuration.
6. Confirm whether E6 is the chosen individual extension and freeze its
   protocol before implementation.
7. Before further model search, document whether E5, the optional Elastic-net
   follow-up, or progression towards E7 serves a specific unresolved question.
