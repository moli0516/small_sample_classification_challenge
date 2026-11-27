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
- Trial 21 was selected by a 50/50 GMM-Trial 25 hybrid objective (hybrid CV AUC 0.983730),
  not by standalone AUC. The standalone result is a post-search follow-up.
- The 95% blends are additional submission comparisons. All eight tested fixed blends
  scored below standalone on mean CV AUC.
- Public scores cover a small subset of the test set. Their ordering is not evidence of
  private leaderboard superiority. They are values reported by the competition
  platform; no file in this repository reproduces them.
- The same 250 labels were reused across successive tuning and selection steps. All
  development results are adaptive estimates, not independent generalisation evidence.

Final submissions (the two entries selected on the competition platform):

1. Primary, CV-selected: `outputs/submissions/generative_optuna_trial21_v1/standalone/submission.csv`
2. Secondary, highest observed public score: `outputs/submissions/trial21_A095_manual_v1/submission.csv`
   (95% Trial 21 + 5% original hybrid; assembled from frozen probability files
   without refitting; see `lineage.json` and `prediction_manifest.json` in that folder).

## Method Overview

1. **Discriminative baselines.** L1/L2 logistic regression, XGBoost, LightGBM and
   equal-weight ensembles, with training-local feature selection.
2. **Trial 25 reference.** An Elastic-net addition to the L1-XGBoost-LightGBM average,
   found by a 30-trial Optuna search and frozen as the discriminative reference.
3. **Geometry-based GMM.** Within each training portion: standardise, rank features
   by squared correlations with other predictors plus a kurtosis term (labels unused),
   keep the top features, apply PCA, then fit a tied-covariance Gaussian mixture on
   the joint distribution of representation and label. Predictions average seeds
   42, 314 and 2026.
4. **Joint tuning.** A 30-trial Optuna search optimised a 50/50 GMM-Trial 25 hybrid.
   Trial 21 used 20 features, 12 PCA dimensions and 12 components.
5. **Fixed-weight blending.** Zero-fit comparison of eight blends against standalone.
6. **Validation-protocol extension.** Hold-out versus 3-, 5- and 10-fold CV, tested on
   earlier discriminative candidate pairs. It does not evaluate Trial 21.

The full write-up is the final report submitted with this project.

## Repository Layout


| Path                    | Contents                                                              |
| ----------------------- | --------------------------------------------------------------------- |
| `src/`                  | Library code, pipelines, validation and experiment modules            |
| `scripts/`              | Entry points for each experiment and export                           |
| `configs/`              | Experiment configurations                                             |
| `docs/`                 | Per-experiment runbooks                                               |
| `tests/`                | Unit and integration tests                                            |
| `experiments/`          | Experiment record and plans                                           |
| `outputs/`              | Experiment artifacts (results, models, frozen exports, submissions)   |
| `evidence/`             | Selected result files that support the report (see Evidence below)    |
| `archived_experiments/` | Code for negative-result experiments (see Archived Experiments below) |
| `data/`                 | Competition data, not redistributed (see Data below)                  |
| `cli.py`                | Command-line entry point                                              |
| `requirements.txt`      | Single dependency file                                                |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Python version: 3.13.7.

## Data

Competition data is not redistributed here unless explicitly stated. Place the
files in `data/raw/` (by default `data/raw/test.csv` and `data/raw/sample_submission.csv`,
together with the labelled training file).

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

Run from the project root, in the original environment. Start with dry-runs,
which perform no fits and write no output:

```bash
python -m scripts.export_generative_trial21_v1 --dry-run
python -m scripts.run_trial21_weights_v1 compare --dry-run
```

Then the full commands:

```bash
python -m scripts.export_generative_trial21_v1
python -m scripts.run_trial21_weights_v1 compare
```

Requirements:

- Artifacts from earlier studies under `outputs/` (the Trial 25 export, the
  generative Optuna study, and the old hybrid and K8 exports).
- The competition files at the paths in `configs/generative_trial21_export_v1.json`.
- The recorded package versions. The scripts verify source hashes, versions, row
  order and identifier alignment, and stop on any mismatch.

Path handling:

- The two new configs use paths relative to the project root and can be edited.
- Existing output directories are never overwritten. Pass `--output` to write to a
  new directory.
- Some stored manifests from earlier studies record absolute paths from the
  original machine. No override is provided for these, and they must not be edited
  to bypass a failed check. If a check fails on another machine or after the
  source or environment has changed, the historical artifacts cannot be
  re-verified there. The stored results can still be inspected directly in
  `outputs/results/`, `outputs/submissions/` and `evidence/`.

The validation-protocol extension is subject to the same check. Its stored results
are kept in `evidence/validation_protocol_v1_run_01/`. A fresh run of
`python -m scripts.run_validation_protocol_v1` stopped at the source-provenance
check in the current environment and was not forced.

See `docs/generative_trial21_export_v1.md`, `docs/trial21_weights_v1.md` and
`docs/generative_blend_export_v1.md` for details.

### Reproducing the final submission (A095)

A095 is the secondary submission, `outputs/submissions/trial21_A095_manual_v1/submission.csv`:
95% Trial 21 standalone GMM plus 5% of the original 0.5/0.5 hybrid,

```
A095 = 0.95*G_new + 0.05*H_old
```

where `G_new` is the Trial 21 standalone export
(`outputs/submissions/generative_optuna_trial21_v1/standalone/submission.csv`)
and `H_old` is the old K8-GMM/Trial 25 hybrid export
(`outputs/submissions/generative_blend_gmm05_v1/submission.csv`). The file was
assembled manually from frozen probability files; no single committed command
writes that folder. The replay script reproduces the blend only: zero fits, no
model loading, and both input SHA256 values are checked against the recorded
A095 manifest before writing.

```bash
python -m scripts.replay_a095 --output outputs/repro_check/a095/submission.csv
```

Expected SHA256:
`d772108c1815ccd7cef1018995c1d8873a42a87fedf6d47ab751971e22b991e9`.

The script exits non-zero when the output hash differs from `--expected-sha256`
and refuses to overwrite an existing output file. Refitting `G_new` or `H_old`
is subject to the limitations below.

### Known reproducibility limitations

- Strict historical provenance verification only passes at the original absolute
  path (`/Users/moli/Documents/coding/DSAI4203_Project/small_sample_classification`)
  with the `settings.json` and `prediction_manifest.json` versions from the
  submission package. Those versions are what the recorded hashes were computed
  over, so they must not be regenerated.
- The unified CLI was added as `clil.py` (commit `7ddb2de`, 2026-10-24) and renamed to
  `cli.py` (`04b1511`, 2026-11-19). Two commits changed it afterwards, both on
  2026-11-19: `04b1511` (file name in the docstring, the test path, and the printed
  `plan` help text) and `dda0330` (one help-text line, `requirements-lgbm-optuna.txt`
  -> `requirements.txt`). No training or prediction logic changed. Its SHA256
  changed from `d348c983...` to `8caab9a2...`.
- Stored evidence files under `evidence/` record the CLI path as `cil.py` with SHA256
  `d348c983...`, which matches the original `clil.py` content. `cil.py` does not
  appear in this repository's git history. These records are left unmodified. The `settings.json` files under `outputs/results/` (e.g. `lgbm_optuna_v1_run_01`,
  `ensemble_research_suite_v1_run_01/*`) record the path `clil.py` instead.
- `src/experiments/lgbm_optuna_experiment.py` changed its reference from `clil.py` to
  `cli.py` in `04b1511` (SHA256 `944c0854...` -> `8a2aa22b...`). Because of this,
  strict source-hash verification against the recorded values fails, even at the
  original path. Result files (`outputs/`, A095 hash) are unaffected.
- The labelled CSV `data/raw/train.csv` has the recorded SHA256
  (`7ef80ff93ef49c8c784445f3484f81a9ea6f0e30a6b59703ca5a15daf690b61f`). Only the
  absolute-path key in the recorded binding differs, so the dry-runs stop with
  `ValueError: Reference does not bind this exact labelled CSV`.
- The default output directories already exist, so the dry-runs need a fresh
  `--output` path:

  ```bash
  python -m scripts.export_generative_trial21_v1 --dry-run --output outputs/repro_check/<new_dir>
  python -m scripts.run_trial21_weights_v1 compare --dry-run --output outputs/repro_check/<new_dir>
  ```

## Evidence

`evidence/` holds copies of the result files that support the numbers in the
report. The complete result directories remain under `outputs/results/`. Two run
directories are kept whole:

- `gmm_trial21_weights_v1_run_01/`: the eight-blend comparison against Trial 21
  standalone (Section V.B), including paired comparisons.
- `validation_protocol_v1_run_01/`: the validation-protocol extension (Table III and
  Fig. 1).


| File                                          | Supports                                                           |
| --------------------------------------------- | ------------------------------------------------------------------ |
| `l2_c_0_001_summary.json`                     | All-feature L2 logistic regression, 0.793999                       |
| `l2_k10_c_0_1_summary.json`                   | Correlation filter, k = 10, C = 0.1, 0.822982                      |
| `l1_c1_summary.json`                          | All-feature L1, C = 1, 0.862112                                    |
| `rbf_svm_c_10_gamma_0_01_decision.json`       | Best RBF-SVM setting, 0.812385                                     |
| `xgb_init_decision.json`                      | Initial shallow XGBoost grid, 0.839271                             |
| `xgb_optuna_trial20_decision.json`            | XGBoost Optuna Trial 20, 0.881937                                  |
| `xgb_k5_summary.json`, `xgb_k20_summary.json` | Top-5 (0.862745) and top-20 feature selection                      |
| `top20_xgb_decision.json`                     | Top-20 XGBoost, 0.890672 (nine wins over Trial 20)                 |
| `feature_selection_k_comparison.csv`          | Full k comparison: k = 5, 10, 20 and all features, with wins       |
| `linear_nonlinear_ensemble_comparison.csv`    | L1 + XGBoost (0.902878) and the fixed-LightGBM comparison          |
| `tuned_three_model_ensemble_decision.json`    | Tuned three-model ensemble mean, 0.908336                          |
| `trial25_reference_T.json`                    | Trial 25 reference T, 0.909210                                     |
| `original_training-only_gmm_decision.json`    | Original training-only GMM, 0.929640                               |
| `original_5050_hybrid_decision.json`          | Original 50/50 hybrid, 0.938275                                    |
| `trial21_hybrid_objective_decision.json`      | Trial 21 hybrid objective, 0.983730, and its selected parameters   |
| `trial21_standalone_decision.json`            | Trial 21 standalone retained, 0.995469                             |
| `0.95gmm_summary.csv`                         | Means and repeat SDs for the eight blends and reference procedures |

Not covered by files in `evidence/`: the capacity follow-up, the transductive
mode comparison, the Elastic-net grid, the RF and PCA follow-ups (see below), and
the public scores in Table II.

## Archived Experiments

Negative-result experiments referenced in the report were run on separate
development branches and are not part of the final pipeline. Their code is
archived under `archived_experiments/` for traceability only, and their result
directories are retained under `outputs/results/`.


| Folder                                  | Experiment                                                                                       | Report outcome                                                                                                                                                                                                                                                                  |
| --------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `archived_experiments/rf/`              | Bounded random-forest addition to the Trial 14 ensemble (V1, and the`max_features` follow-up V2) | The report covers V1 only: it improved the mean by 0.000031 and was kept as a backup, not adopted. The V2 follow-up is not discussed in the report. It retained the original`sqrt` setting and promoted no new `max_features` value (see `outputs/results/rf_auto_v2_run_01/`). |
| `archived_experiments/pca_l2/`          | PCA with L2 logistic regression                                                                  | Did not improve the fixed discriminative procedures                                                                                                                                                                                                                             |
| `archived_experiments/pca_three_model/` | PCA applied to the three-model ensemble (`pca_three_model_v1`, `pca_ensemble_v1`)                | Did not improve the fixed discriminative procedures; best common-PCA ensemble 0.771577 against 0.908336                                                                                                                                                                         |

These archives are not wired into `cli.py` and are not covered by the test
command above. Their provenance checks expect the original local paths and
environment, so they may not rerun on another machine. Do not edit the stored
manifests to bypass those checks.

## Scope and Limitations

- Training-local fitting prevents direct estimation from held-out labels, but it does
  not remove optimism from repeated tuning and selection on the same labels.
- Research modes do not fit on competition test features. Test features are used only
  for transformation and prediction.
- No procedure is automatically promoted or uploaded to the competition.
- Private leaderboard results were unavailable at the time of writing.

## Author

TAM King Yeung Oscar (25074264d)
