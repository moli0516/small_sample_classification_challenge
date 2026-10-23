# Registry ensemble patch — quick start

Target: moli0516/small_sample_classification_challenge
Reviewed latest commit: d737af3630541095e53c5dfdcffde9ed30ace313
Commit message: L1_xgboost_ensemble_ver

## Apply: choose ONE method

A. Merge the archive's src/, scripts/, configs/, tests/, docs/ and
requirements-registry.txt into the project root. All runtime files are NEW;
no existing file or historical output is replaced.

B. Alternatively, extract the archive somewhere else and, in the project root, run:

    git apply --check /path/to/extracted/registry_ensembles.patch
    git apply /path/to/extracted/registry_ensembles.patch

Do not apply the patch AFTER merging the same files. The patch is additive and
was tested with git apply --check and git apply. It does not change requirements.txt,
old factories/runners, source snapshots or their historical checksums.
Do not copy temporary test fixtures into the project; none are included.

## Run (existing .venv, project root)

    python -m pip install -r requirements-registry.txt
    python -m pytest -q tests/test_registered_ensembles.py
    python -m pytest -q
    python -m scripts.run_registered_ensembles list
    python -m scripts.run_registered_ensembles cv --dry-run
    python -m scripts.run_registered_ensembles cv

Defaults are in configs/registry_ensembles.json. Existing completed files required:
- outputs/frozen/ext03_top20_v1/
- outputs/results/soft_vote_l1_xgb20_v1_run_01/
- configs/e4_l1_c_1.json and the frozen training file.

The three aliases are l1, xgboost, lightgbm. Seven comparisons are saved:
- Three individual models.
- l1_xgb, lgbm_l1, lgbm_xgb, l1_xgb_lgbm.
All pairs are equal-weight; the triple is exactly 1/3 each, not 1/4,1/4,1/2.
The existing l1_xgb is the reference (reported development AUC 0.9028784219,
reported public AUC 0.91514); neither historical record is overwritten.

## Cost and OOF reuse

By default, verify/reuse the prior L1 and XGBoost out-of-fold probabilities and
fit ONLY 50 LightGBM models on the existing 10x5 saved splits.
Verification covers source completion, artifact hashes, original source/input
hashes, environment, exact fold keys, row IDs, labels, complete coverage, original
50/50 blend and all source fold AUCs. There is NO silent fallback.

If the cache genuinely cannot be used, inspect why. To intentionally refit all
three complete procedures, use:

    python -m scripts.run_registered_ensembles cv --no-cache --dry-run
    python -m scripts.run_registered_ensembles cv --no-cache

That is 200 fits (per fold: L1, XGBoost ranker, selected XGBoost, LightGBM).
Use an unused --output if an earlier full run created the default directory.
Dry runs create no output directory. No competition test file is opened during CV.
No early stopping, weight search, Optuna or new CV splitting is introduced.

## Decision and final export

Read outputs/results/registry_ensembles_v1_run_01/comparison.csv and decision.json.
New ensembles qualify only if mean paired AUC exceeds the existing L1+XGBoost AND
they win >=6/10 repeats. This is a pragmatic rule, NOT a significance test.
Among qualifying new ensembles choose highest mean AUC; exact ties (1e-12) prefer
fewer members then stable name order. Single models are diagnostic controls, not
automatically promoted. If nothing qualifies, auto export stops; keep 0.91514.

    python -m scripts.run_registered_ensembles predict --dry-run
    python -m scripts.run_registered_ensembles predict

Only the selected candidate is exported by default. --candidate NAME can select
another qualifying ensemble, not the unchanged baseline or a failed candidate.
No automatic sweep over public leaderboard submissions occurs.

Output: outputs/submissions/registry_ensemble_v1_run_01/.
- submission.csv: probabilities, template ID order.
- models/ALIAS.joblib: complete fitted adapters (not just bare classifiers).
- ensemble.json: weights, features, model diagnostics/configurations.
- prediction_manifest.json, warnings.json, fit.log.

All required procedures are refitted using ALL 250 labelled rows. XGBoost reselects
its actual top20 using that full training set; test never fits anything. Models
are reloaded and probability identity checked at 1e-12 before CSV export.
Only load joblib artifacts you created/trust: they execute Python deserialization.
Models are environment-bound; preserve the recorded package versions and source.

Upload ONLY submission.csv manually after prediction_manifest.json says
ready_for_manual_submission, rows=19750, saved_and_reloaded_models_verified=true.
Do not upload partial CSVs when status is failed_or_interrupted. Record actual
Kaggle ID/time/score separately. No GitHub write or Kaggle API call is made.

## Small architecture

- src/modeling/model_registry.py: protocol, strict registration, safe soft voting.
- src/modeling/registered_models.py: three complete adapters, lazy native imports.
- src/experiments/registered_ensembles.py: shared CV, verified cache and export.
- scripts/run_registered_ensembles.py: list/cv/predict entry point.

Register another model by adding a concrete RegisteredModel subclass and decorating:

    @ModelRegistry.register("new_model")
    class NewModel(RegisteredModel):
        def __init__(self, *, context=None, parameters=None): ...
        def fit(self, X, y): ...
        def predict_proba(self, X): ...

Store a fitted self.model with classes_=[0,1], or override classes_. predict_proba
must return n-by-2 probabilities. Use _remember(X)/_check(X) to protect column order.
Keep optional dependency imports inside fit. Add the class in registered_models.py,
or explicitly import its module there: simply creating a file does not register it.
Then add its alias/type/parameters and desired ensemble members/weights to the JSON.
No runner modification is needed for another protocol-compatible classifier.
The current historical-cache contract reserves l1/xgboost aliases; do not rename them.
For a custom model requiring multiple estimator fits, update accounting in _cost;
that count is diagnostic, not used to fit or validate the model.

## Warnings and reproducibility

Any Python warning stops the run for review. Native LightGBM logs are retained in
fit.log, including normal no-more-splits messages; they are not automatically
classified as Python errors. Actual round/tree counts and number of unique
validation probabilities are in diagnostics.json. Constant predictions are retained
as evidence (e.g. AUC .5), not silently discarded. Fixed n_estimators is the cap;
LightGBM may finish fewer effective rounds when no valid split remains.

The old source files are intentionally NOT refactored, because historical OOF
manifests hash them. Existing CLIs and frozen candidates remain usable.
CV scores remain adaptive development results, not independent final estimates.

## Build validation / limitations

30 tests discovered: 28 passed, 2 skipped (native LightGBM and XGBoost unavailable).
Tested strict registry, lazy imports, weights/probability/feature/class checks,
cache hashes/fold keys/IDs/labels/AUC, 50-versus-200 planned fits, synthetic 50-fold
seven-candidate CV, promotion gating and joblib reload/template-aligned export.
Synthetic core flows mock model procedures and the existing context loader; no real
250x150 data were fitted and no actual competition submission CSV was generated.
Native LightGBM wrapper and XGBoost tests will run locally if installed.
The full existing repository suite was NOT available/run here. CLI and patch
application were tested. Run both local test commands above before real training.
