# Elastic-net Trial25 prediction/export

Continue on the current branch. No new branch, GitHub write, registry mutation, search
rerun, incumbent promotion or Kaggle upload is performed by this patch.

## Structure

- src/modeling/elasticnet_trial_model.py: ElasticnetTrialModel (RegisteredModel subclass).
- src/submission/elasticnet_trial_snapshot.py: ElasticnetTrialSnapshot (read-only verifier).
- src/submission/elasticnet_trial_submission.py: ElasticnetTrialSubmission (full-data exporter).
- scripts/predict_elasticnet_trial_25.py: thin function-only CLI.
- tests/test_elasticnet_trial_submission.py: function-only pytest tests.

One newly defined class per source file. New filenames/methods are snake_case; class
names follow the existing PascalCase convention. Existing registered_models.py is not
refactored because it is part of historically hashed sources.

## Apply from the project root on the current branch

    git apply --check elasticnet_trial_25_export.patch
    git apply elasticnet_trial_25_export.patch
    python -m pytest -q tests/test_elasticnet_trial_submission.py
    python -m pytest -q
    python -m scripts.predict_elasticnet_trial_25 --dry-run
    python -m scripts.predict_elasticnet_trial_25
    python clil.py results --run outputs/submissions/elasticnet_optuna_trial_25_run_01

Keep the existing frozen environment; do not upgrade packages. The patch relies on the
already completed study, local historical artifacts and the existing Optuna environment.
No new requirements file is needed. All six files in this patch are additive.
clil.py is intentionally unchanged: LGBMOptunaExperiment._prepare includes its digest in
historical hashes. Adding a route there would invalidate existing study verification.
Use the standalone thin module entry point above.

Options: --run, --config, --output, --test, --template, --dry-run.
Defaults:

    --run outputs/results/elasticnet_optuna_v1_run_01
    --config configs/elasticnet_optuna_v1.json
    --output outputs/submissions/elasticnet_optuna_trial_25_run_01

The default test/template paths and expected row count come from the existing frozen
candidate. Fresh output only; no resume/overwrite. --dry-run validates the real test and
template (test_opened=true) but does zero fits and creates no output directory. Historical
study test_opened=false describes the earlier search and is not rewritten.

## Integrity checks

Requires completed30-trial/1500-fit study with no test access/upload during search.
Verifies every artifact checksum with containment checks; reselects the highest objective
from the saved30 COMPLETE trials (exact ties earliest); requires Trial25; cross-checks
trials.csv/JSON, selected parameters, best/result/decision and control evidence.
Reuses historical prepare ONLY as a read-only compatibility adapter. No legacy execute,
search, CV or refit function is called during preflight. Old scripts stay unchanged.
Compares current config, original code/input hashes and recorded package versions.
Checks cohort and ordered splits. Reconstructs Trial0 and Trial25 blends from saved
NPZ probabilities and recomputes AUC, selected repeat SD, standalone AUC and both paired
screening comparisons before allowing export. Does not rely on a boolean flag alone.
Trial25 identity: mean_auc0.9092096430139908; C0.18702703609689042;
l1_ratio0.9589771634360093; alpha0.06655791394873416.

## Refit and formula

Fit the existing L1Model, XGBTopKModel and LightGBMModel from the verified source context,
plus the new ElasticnetTrialModel. All fit the full labelled training set. XGBoost keeps
its full-data ranker -> top20 selector -> fresh classifier procedure; it does not use
cross-fold feature frequencies. LightGBM keeps original Trial14 parameters. Elastic-net
uses StandardScaler -> saga logistic regression, max_iter20000, tol1e-4, seed42,
and the exact selected C/l1_ratio. Scaler never fits test data.

    p_original = (p_l1 + p_xgboost + p_lightgbm) / 3
    p_final = (1-alpha) * p_original + alpha * p_elasticnet

No CatBoost refit, replacement branch, changed boosting rounds, rank averaging or weight
retuning. Costs: L1=1, XGBoost=2, LightGBM=1, Elastic-net=1, total5 estimator fits.

All four model procedures are saved via existing RegisteredModel.save/load; test
probabilities before/after reload must agree within1e-12. Capture Python warnings and
fail closed on any warning. Native informational log output goes into fit.log.

## Output and manual review

    prediction_manifest.json
    source_decision.json
    source_best_trial.json
    ensemble.json
    models/l1.joblib
    models/xgboost.joblib
    models/lightgbm.joblib
    models/elasticnet.joblib
    component_predictions.csv
    warnings.json
    fit.log
    submission.csv

component_predictions.csv is in original test order. submission.csv is aligned by string
ID to template order (including leading zeros), never by row position. Checks schema,
ID uniqueness/sets, numeric finite predictors, expected row count, probability bounds,
and CSV round-trip. Hashes prediction inputs BEFORE reading and verifies again before
publication. Writes to a pending CSV first; failed validation removes final-named CSV.

Review manifest before manually uploading:

    status: ready_for_manual_submission
    rows:19750
    fits_performed:5
    saved_and_reloaded_models_verified:true
    kaggle_upload_performed:false
    automatic_promotion:false

The manifest records source/input/new-code hashes, model artifacts, submission hash,
versions, fit counts, warnings status and formula. Failed/interrupted output is not a
valid submission candidate. Never bypass historical hash/version mismatches or regenerate
old manifests to force a pass. Keep prior submissions untouched.

## Verification limits

Author sandbox:21 tests passed, including actual sklearn fitting, joblib save/reload,
formula checks, string-ID reordering, scaler training-only mean, and failure paths for
changed artifacts/sources/environment, incorrect selection, invalid data, warnings,
reload mismatches and input edits during refit. One-class-per-file AST checks also pass.

These tests use a synthetic completed-study fixture and injected sklearn substitutes
for the three native base branches. The sandbox uses a source-compatible RegisteredModel
fixture and a strict CSV-helper contract; on the real repository tests import its actual
RegisteredModel/CSV helper. This is not a native XGBoost/LightGBM integration run and does
not establish that all real local historical hashes will pass. Full real-data preflight,
five-fit export and full repository tests must be run locally. No submission was made.

Full-data refit/export is not a fresh performance evaluation. The selected CV result
remains adaptive development evidence, not independent/nested confirmation.
