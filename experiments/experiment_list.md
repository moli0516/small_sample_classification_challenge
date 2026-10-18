# Experiment List

## Scope and current status

This is an early-stage research roadmap, not a claim that model selection is complete.
Completed development studies include the all-feature L2 C sweep,
correlation-based feature selection, the selected-feature-count / C interaction,
the four-setting L1 comparison, and the six-setting RBF SVM comparison.

* Data audit: completed; 250 labelled rows, 19,750 test rows, 150 numeric predictors.
* Identifier: `id`, excluded from predictors; target: `target`.
* ValidationRunner and SubmissionWriter tests: reported passing by the project owner.
* Baseline submission workflow: completed, as reported by the project owner;
  the actual public score, submission identifier, and timestamp remain to be recorded.
* E0–E5: completed for the studies documented below.
* The optional Elastic-net follow-up in E4 has not been run.
* Current highest-mean development candidate: all-feature L1 logistic regression,
  C=1.0; mean CV AUC 0.862112, repeat SD 0.012204.
* Sparse alternative: all-feature L1 logistic regression, C=0.1;
  mean CV AUC 0.852864, repeat SD 0.010803.
* E5 did not replace either retained L1 candidate. Every tested SVM setting
  performed worse than both candidates in all ten paired repetitions.
* E5 warning review: completed. The 300 recorded warnings concerned the
  deprecation of SVC's probability parameter; no convergence warnings,
  failed fits, or iteration-limit events were recorded.
* E4 and E5 candidate submission status: not recorded.
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

| ID | Investigation | Status | Priority |
| -- | -- | -- | -- |
| E0 | All-feature logistic regression: C sweep | Completed | Preserve |
| E1 | End-to-end baseline submission | Completed | Record feedback |
| E2 | Correlation-based feature selection | Completed | Preserve results |
| E3 | Interaction between selected feature count and C | Completed | Preserve results |
| E4 | L1 logistic regression; optional Elastic-net | L1 completed; Elastic-net not run | Retain candidates |
| E5 | Bounded RBF SVM comparison | Completed; warning review completed | Preserve negative finding |
| E6 | Validation protocol extension | Proposed | Required if chosen |
| E7 | Candidate freeze and final evaluation | Later | Required |

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

Status: completed; warning review completed.

Question:
Does an all-feature RBF SVM improve development ROC AUC over the
retained L1 logistic-regression candidates under identical validation splits?

Scope:
- Compare a bounded nonlinear alternative against the current linear candidates.
- No simultaneous feature-subset, class-weight, kernel, or calibration search.
- Treat this as a development study, not independent final evaluation.

Implementation:
- Executed using the standalone run_e5.py script.
- This execution did not integrate SVM support into the existing src/ runner.
- The initial split loader required a compatibility correction:
  the project's saved JSON contains a metadata wrapper with a splits list,
  and uses valid_indices rather than val_indices.
- The correction normalised the representation in memory.
- Existing E4 split files were not regenerated or rewritten.

Protocol:
- Pipeline: StandardScaler -> SVC(kernel="rbf").
- All 150 predictors; id excluded.
- No correlation-based feature selection, PCA, or other feature selection.
- Fit StandardScaler within each training fold only.
- C in {0.1, 1.0, 10.0}.
- gamma in {"scale", 0.01}.
- Six configurations.
- Reuse saved shuffled stratified five-fold split assignments for seeds 0–9.
- Fifty CV fits per configuration: 300 fits in total.
- Compute validation ROC AUC from decision_function scores.
- Average the five fold AUC values within each repetition.
- Report the mean and sample SD across the ten repetition-level means.
- Use saved repeat-level reference scores for paired comparisons.

Fixed estimator settings:
- probability=False.
- class_weight=None.
- tol=1e-3.
- max_iter=10000.
- shrinking=True.
- cache_size=200 MB.
- random_state=42, retained for configuration consistency.
- With probability=False, SVC ignores random_state.

Reference candidates:
- Primary: e4_l1_c_1.
  All-feature L1 logistic regression, C=1.0;
  mean development CV AUC 0.862112, repeat SD 0.012204.
- Secondary: e4_l1_c_0_1.
  All-feature L1 logistic regression, C=0.1;
  mean development CV AUC 0.852864, repeat SD 0.010803.
- Historical reference: e3_correlation_k_10_c_0_1.
  Correlation selection with k=10 and L2 logistic regression, C=0.1;
  mean development CV AUC 0.822982, repeat SD 0.008456.

Results:

| C | Gamma | Mean CV AUC | Repeat SD | Minimum repeat AUC | Maximum repeat AUC |
|---:|---|---:|---:|---:|---:|
| 0.1 | scale | 0.807268 | 0.011407 | 0.790022 | 0.827447 |
| 1.0 | scale | 0.807959 | 0.012793 | 0.784573 | 0.825831 |
| 10.0 | scale | 0.803024 | 0.013892 | 0.780751 | 0.823229 |
| 0.1 | 0.01 | 0.811179 | 0.012608 | 0.789726 | 0.831021 |
| 1.0 | 0.01 | 0.812318 | 0.013557 | 0.787496 | 0.833254 |
| 10.0 | 0.01 | 0.812385 | 0.013559 | 0.788799 | 0.829382 |

Minimum and maximum AUC refer to repetition-level mean fold AUC,
not individual-fold AUC.

Paired comparisons against the E4 leader, all-feature L1 C=1.0:

| SVM C | Gamma | Mean paired AUC difference | Difference SD | Improved / worse / tied repeats |
|---:|---|---:|---:|---|
| 0.1 | scale | -0.054844 | 0.012358 | 0 / 10 / 0 |
| 1.0 | scale | -0.054153 | 0.013454 | 0 / 10 / 0 |
| 10.0 | scale | -0.059087 | 0.012564 | 0 / 10 / 0 |
| 0.1 | 0.01 | -0.050932 | 0.013182 | 0 / 10 / 0 |
| 1.0 | 0.01 | -0.049794 | 0.013746 | 0 / 10 / 0 |
| 10.0 | 0.01 | -0.049726 | 0.012739 | 0 / 10 / 0 |

Paired comparisons against the sparse E4 alternative, all-feature L1 C=0.1:

| SVM C | Gamma | Mean paired AUC difference | Difference SD | Improved / worse / tied repeats |
|---:|---|---:|---:|---|
| 0.1 | scale | -0.045595 | 0.008274 | 0 / 10 / 0 |
| 1.0 | scale | -0.044905 | 0.007382 | 0 / 10 / 0 |
| 10.0 | scale | -0.049839 | 0.009024 | 0 / 10 / 0 |
| 0.1 | 0.01 | -0.041684 | 0.008385 | 0 / 10 / 0 |
| 1.0 | 0.01 | -0.040545 | 0.008853 | 0 / 10 / 0 |
| 10.0 | 0.01 | -0.040478 | 0.008079 | 0 / 10 / 0 |

Paired comparisons against the E3 leader:

| SVM C | Gamma | Mean paired AUC difference | Difference SD | Improved / worse / tied repeats |
|---:|---|---:|---:|---|
| 0.1 | scale | -0.015714 | 0.010717 | 1 / 9 / 0 |
| 1.0 | scale | -0.015023 | 0.012967 | 1 / 9 / 0 |
| 10.0 | scale | -0.019957 | 0.014307 | 1 / 9 / 0 |
| 0.1 | 0.01 | -0.011803 | 0.012194 | 1 / 9 / 0 |
| 1.0 | 0.01 | -0.010664 | 0.013668 | 2 / 8 / 0 |
| 10.0 | 0.01 | -0.010597 | 0.014015 | 1 / 9 / 0 |

Numerical diagnostics:

| C | Gamma | Completed / valid folds | Failed folds | General-warning folds | Convergence-warning folds | Iteration-limit folds | Maximum n_iter | Mean support fraction |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0.1 | scale | 50 / 50 | 0 | 50 | 0 | 0 | 149 | 0.9869 |
| 1.0 | scale | 50 / 50 | 0 | 50 | 0 | 0 | 251 | 0.9851 |
| 10.0 | scale | 50 / 50 | 0 | 50 | 0 | 0 | 396 | 0.9837 |
| 0.1 | 0.01 | 50 / 50 | 0 | 50 | 0 | 0 | 186 | 1.0000 |
| 1.0 | 0.01 | 50 / 50 | 0 | 50 | 0 | 0 | 292 | 1.0000 |
| 10.0 | 0.01 | 50 / 50 | 0 | 50 | 0 | 0 | 373 | 1.0000 |

All six candidates were eligible under the predefined validity rules.

Support fraction is the number of support vectors divided by the number
of training rows in a fold. It is not the fraction of selected predictors.
All 150 predictors remained available to each SVM.

Warning review:
- Completed using the saved e5_fold_diagnostics.csv.
- The review reported 300 occurrences of one FutureWarning.
- The warning concerned deprecation of SVC's probability parameter
  in scikit-learn 1.9, with removal scheduled for version 1.11.
- No additional warning categories or messages were reported.
- E5 explicitly used probability=False and evaluated decision_function scores.
- Probability calibration was not part of this experiment.
- No convergence warnings, failed fits, or iteration-limit events were recorded.
- Preserve warning_folds=50 for each candidate; do not replace the original
  warning counts with zero.
- No rerun is required solely to remove this deprecation warning.

Highest-mean E5 candidate:
- Experiment ID: e5_rbf_svm_c_10_gamma_0_01.
- Configuration: all-feature RBF SVM, C=10.0, gamma=0.01.
- Mean development CV AUC: 0.812385.
- Repeat SD: 0.013559.
- Mean paired difference versus the E4 leader: -0.049726;
  improved in 0/10 repetitions.
- Paired difference range versus the E4 leader:
  -0.075553 to -0.034599.
- Mean paired difference versus the sparse E4 alternative: -0.040478;
  improved in 0/10 repetitions.
- Mean paired difference versus the E3 leader: -0.010597;
  improved in 1/10 repetitions.

Findings:
- None of the six RBF SVM settings improved over either retained
  L1 candidate in any of the ten paired repetitions.
- All six SVM settings also had lower mean AUC than the E3 leader.
- gamma=0.01 had a higher mean AUC than gamma="scale" at each tested C.
- Within gamma=0.01, C=10.0 exceeded C=1.0 by approximately 0.000067
  mean AUC; this small difference does not establish a meaningful advantage.
- The bounded comparison provides no evidence for replacing the
  current L1 candidates with these SVM settings.
- High support-vector fractions are descriptive diagnostics, not proof
  of overfitting or an implementation error.

Decision:
- Complete and stop the planned six-setting RBF SVM comparison.
- Retain e4_l1_c_1 as the provisional overall leading pipeline.
- Retain e4_l1_c_0_1 as the sparse alternative.
- Preserve the E3 leader and earlier baselines as historical references.
- Record the E5 leader as the highest-mean candidate within E5 only.
- Do not automatically expand C or gamma because the E5 leader lies
  on the tested C boundary.
- Do not add feature selection, another kernel, or probability calibration
  to this completed experiment.
- Do not submit the six SVM candidates solely to seek a better leaderboard score.
- E5 submission status: not recorded.

Limitations:
- Only six all-feature RBF SVM settings were tested.
- The result does not establish that every nonlinear model is inferior.
- This is a comparison of complete pipelines, not an isolated test
  of nonlinearity alone.
- E5 reused the same labelled observations and development validation
  protocol used in earlier studies.
- Repeated splits are not independent datasets, and their SD is not
  a confidence interval.
- Choosing the highest-mean SVM setting remains development model selection,
  not independent final evaluation.
- No public or private leaderboard outcome is established.

Artifacts:
- Run directory: outputs/results/e5_rbf_run_01/.
- run_manifest.json.
- verified_validation_assignments.csv.
- reference_repeat_scores.csv.
- e5_comparison.csv.
- e5_paired_repeat_scores.csv.
- e5_fold_diagnostics.csv.
- e5_decision.json.
- Six candidate subdirectories containing configurations, fold diagnostics,
  and validation decision scores.
- Standalone execution script and tests.
- Warning-review output from python -m check_e5_warning.

Preserve the exact executed script version, including the split-loader
compatibility correction, together with the original output files.

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

1. Archive the completed E0–E5 results, configurations, saved splits,
   package versions, available data hashes, and code revisions.
2. Preserve the exact E5 execution script, including the split-loader fix,
   and record the completed warning review.
3. Retain e4_l1_c_1 as the provisional overall leader and
   e4_l1_c_0_1 as the sparse alternative.
4. Record the missing E1 public score, submission identifier, and timestamp;
   do not duplicate an already completed baseline submission.
5. Confirm whether an E4 candidate has already been submitted.
   If not, consider one documented candidate submission rather than
   submitting every tested configuration.
6. Confirm whether E6 is the chosen individual extension.
   Freeze its pipelines, validation schemes, repetitions, and comparison
   rules before implementation.
7. Prepare E7: freeze a small candidate set and document whether a bounded
   nested-CV evaluation will be used.
8. Do not automatically reopen the completed E4 or E5 searches.
   Any additional experiment requires a specific unresolved question,
   bounded settings, and a stopping rule defined in advance.