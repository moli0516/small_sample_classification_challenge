# Project Plan

## 1. Objective

Build a binary classifier for the small-sample classification challenge,
using a reproducible validation protocol to guide model selection.

This document records early experiments and next steps, not a final solution.

## 2. Initial data audit

- Training rows: 250; test rows: 19,750.
- Predictors: 150 numeric features.
- Target: `target`; identifier: `id`, excluded from modelling.
- Classes: 137 negative and 113 positive.
- No missing predictor values, exact duplicate predictor rows,
  constant training features, or exact train/test predictor matches detected.
- No observations or predictors removed.

These checks establish structural consistency, not the absence of leakage
or distribution shift.

## 3. Current baseline

Pipeline:

    StandardScaler -> LogisticRegression

Fixed settings:

- All 150 predictors; no feature selection.
- Solver: `lbfgs`; maximum iterations: 2,000.
- Scaling fitted separately inside each CV training fold.
- Stratified five-fold CV, repeated with split seeds 0–9.
- Metric: ROC-AUC using class-1 probabilities.
- Report the mean and sample standard deviation of the ten
  repeat-level mean fold AUCs.

The reported spread measures split sensitivity, not a confidence interval.
The official Kaggle metric and submission format still need confirmation.

## 4. Experiment 1: varying C

Question: How does changing C affect the fixed baseline pipeline?

| C | Mean CV AUC | Repeat SD |
|---:|---:|---:|
| 1.0 | 0.751411 | 0.019922 |
| 0.1 | 0.771935 | 0.013106 |
| 0.01 | 0.792885 | 0.011157 |
| 0.001 | 0.793999 | 0.009091 |
| 0.0001 | 0.790600 | 0.008026 |

Only C was varied. Other model parameters, preprocessing, features,
and the validation scheme were held fixed.

Findings:

- C=0.1 and C=0.01 each improved on the preceding setting in all
  ten paired repetitions.
- C=0.001 improved on C=0.01 in seven repetitions, but its mean
  advantage was only about 0.0011 AUC.
- C=0.0001 performed worse than C=0.001 in nine repetitions.

Decision:

- Retain C=0.001 as the provisional baseline candidate.
- Retain C=0.01 as a close alternative.
- Stop this C sweep; further investigations will be separate experiments.

This was a sequential, exploratory comparison. The selected CV score
is a development result, not an independent final evaluation.

## 5. Reproducibility

Keep each experiment's configuration, fold scores, repeat scores,
summary, and split indices.

- [ ] Verify identical split assignments across configurations.
- [ ] Record data hashes, package versions, and code revision.
- [ ] Check convergence warnings and rerun reproducibility.

## 6. Next steps

- [ ] Confirm Kaggle evaluation and submission requirements.
- [ ] Implement and test the submission workflow.
- [ ] Refit the provisional pipeline on all training rows and submit.
- [ ] Record the public score and compare it with local validation.
- [ ] Plan one controlled feature-selection or alternative-model experiment.
- [ ] Confirm the individual extension.

Proposed extension: compare repeated hold-out with stratified
3-, 5-, and 10-fold CV, keeping model settings fixed.

Further model development remains open. The current study has not compared
other regularisation forms, preprocessing methods, feature sets, or models.