# Small-Sample Binary Classification Challenge

Course project for DSAI4203. The task is binary classification with 250 labelled
observations, 150 numeric predictors and 19,750 unlabelled test observations.
Submissions are scored by ROC AUC. According to the assignment, most predictors
are noise.

## Final Result


| Procedure                                                | Development CV AUC | Public AUC | Role                          |
| -------------------------------------------------------- | ------------------ | ---------- | ----------------------------- |
| Trial 21 standalone (geometry-based, label-informed GMM) | 0.995469           | 0.98859    | CV-selected procedure         |
| 95% Trial 21 + 5% original hybrid                        | 0.995438           | 0.98877    | Highest observed public score |
| 95% Trial 21 + 5% Trial 25                               | 0.995308           | 0.98846    | Additional comparison         |

- Development AUC is the mean of ten repetitions of shuffled stratified five-fold CV.
- Trial 21 was selected by a 50/50 GMM–Trial 25 hybrid objective (hybrid CV AUC 0.983730),
  not by standalone AUC. The standalone result is a post-search follow-up.
- The 95% blends are additional submission comparisons. All eight tested fixed blends
  scored below standalone on mean CV AUC.
- Public scores cover a small subset of the test set. Their ordering is not evidence of
  private leaderboard superiority.
- The same 250 labels were reused across successive tuning and selection steps. All
  development results are adaptive estimates, not independent generalisation evidence.

Final prediction file: `TODO: path to the chosen submission CSV`

## Method Overview

1. **Discriminative baselines.** L1/L2 logistic regression, XGBoost, LightGBM and
   equal-weight ensembles, with training-local feature selection.
2. **Trial 25 reference.** An Elastic-net addition to the L1–XGBoost–LightGBM average,
   found by a 30-trial Optuna search and frozen as the discriminative reference.
3. **Geometry-based GMM.** Within each training portion: standardise, rank features
   by squared correlations with other predictors plus a kurtosis term (labels unused),
   keep the top features, apply PCA, then fit a tied-covariance Gaussian mixture on
   the joint distribution of representation and label. Predictions average seeds
   42, 314 and 2026.
4. **Joint tuning.** A 30-trial Optuna search optimised a 50/50 GMM–Trial 25 hybrid.
   Trial 21 used 20 features, 12 PCA dimensions and 12 components.
5. **Fixed-weight blending.** Zero-fit comparison of eight blends against standalone.
6. **Validation-protocol extension.** Hold-out versus 3-, 5- and 10-fold CV, tested on
   earlier discriminative candidate pairs. It does not evaluate Trial 21.

The full write-up is in the final report (`TODO: report location`).

## Repository Layout


| Path               | Contents                                                            |
| ------------------ | ------------------------------------------------------------------- |
| `src/`             | Library code, pipelines, validation and experiment modules          |
| `scripts/`         | Entry points for each experiment and export                         |
| `configs/`         | Experiment configurations                                           |
| `docs/`            | Per-experiment runbooks                                             |
| `data` | Training and test datasets                                             |
| `tests/`           | Unit and integration tests                                          |
| `outputs/`         | Experiment artifacts (results, models, frozen exports, submissions) |
| `cli.py`           | Command-line entry point                                            |
| `requirements.txt` | Single dependency file                                              |




## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Python version: Python 3.13.7.

## Data

Competition data is not redistributed here unless explicitly stated.
Data should be placed in data/raw, which do not included in this repository

## Tests

```bash
python -m pytest -q tests
```

Local results from the author's machine at the time of writing: a targeted suite
passed 100 tests, and a repository-wide run passed 507 tests with 24 subtests and
59 warnings. The repository-wide run also collected a test file under
`outputs/maintenance/`, so the 507 figure is not a count of files in `tests/`.
Warnings concern scikit-learn (`SVC(probability=True)`) and pandas
(`set_index(..., verify_integrity=True)`) deprecations.

## Reproducing the Trial 21 Results

```bash
python -m scripts.export_generative_trial21_v1
python -m scripts.run_trial21_weights_v1 compare
```

These commands need artifacts from earlier studies under `outputs/` (the Trial 25
exports, the Trial 21 search study and recorded provenance) and strictly verify source
hashes, row order and identifier alignment. A fresh clone without those artifacts will
not reproduce them. See `docs/generative_trial21_export_v1.md`,
`docs/trial21_weights_v1.md` and `docs/generative_blend_export_v1.md` for details.

Some stored manifests record absolute paths from the original machine. Keeping the
directory layout does not resolve these checks elsewhere. `TODO: describe any path override or confirm none is needed.`

## Experiment Branches

Some negative-result experiments referenced in the report live on separate branches
and are not part of `main`:

- `feat/RF`: bounded random-forest ensemble experiments (PR #1, unmerged).
- `feat/pca-l2-experiment-v1`, `feat/pca-three-model-representation-v1`: PCA follow-ups.

`TODO: update this section if these are merged or archived.`

## Scope and Limitations

- Training-local fitting prevents direct estimation from held-out labels, but it does
  not remove optimism from repeated tuning and selection on the same labels.
- Research modes do not fit on competition test features. Test features are used only
  for transformation and prediction.
- No procedure is automatically promoted or uploaded to the competition.
- Private leaderboard results were unavailable at the time of writing.

## Author

TAM King Yeung Oscar (25074264d)
