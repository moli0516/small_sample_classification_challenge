# Frozen Trial21 final export patch

## Deliverables / current branch

The companion patch generator has since been removed from the repository (commit `2b6fee8`).
It was a self-contained STANDARD-LIBRARY tool that wrote an additive unified diff only: no
source files, git branch, commit, model fits, network calls, submissions or uploads were
performed by it. The attached generated patch remains equivalent and is kept as a historical
record; the steps below apply to that patch.

    git apply --check trial21_export.patch
    git apply trial21_export.patch
    python -m pytest -q tests/test_generative_trial21_export_v1.py
    python -m scripts.export_generative_trial21_v1 --dry-run
    python -m scripts.export_generative_trial21_v1

Continue CURRENT branch. Patch adds6 files, no historical changes. Requires the earlier
local generative geometry, follow-up and generative Optuna patches. Their modules were
previously absent on remote default branch; no claim is made that a remote checkout
contains your local sources. Existing current environment is required, no upgrades.

## Local paths

New config: configs/generative_trial21_export_v1.json
study=outputs/results/generative_optuna_v1_run_01
study_config=configs/generative_optuna_v1.json
trial25_export=outputs/submissions/elasticnet_optuna_trial_25_run_01
test_path=data/raw/test.csv
sample_submission_path=data/raw/sample_submission.csv
expected_rows=19750
output_dir=outputs/submissions/generative_optuna_trial21_v1

Correct paths in this NEW config to match actual local files. --output can select a new
fresh destination. Existing outputs are refused, never resumed or overwritten. Do not
edit old source manifests/settings/code to bypass hash/version failures.

## Source verification before refit

Delegate to the existing generative_optuna snapshot.prepare, preserving its exact250x150
labelled data,50 ordered fold partitions, Trial25/K8/original-hybrid source reproduction,
source-code/input/hash/environment checks. No fits or competition-input reads occur.

Then verify every generative Optuna artifact against manifest; require EXACT completed
30/30 trials, zero failures,4500 attempted/completed/converged composite fits and1500
projector fits, no test access/upload. Pin selected best/decision/parameters/predictions
SHA256 to the manifest supplied by the user. This is a dedicated Trial21 snapshot, NOT
an any-winner or future-study exporter. Pins are not optional CLI switches.

Validate all30 trial records and CSV agreement, exact tie-breaking (21 before24), frozen
parameters, selected result=best_trial, decision screening and original Trial0 control
reproduction. Recompute Trial0 and21 row-level hybrid formulas, CV mean/std and paired
comparisons from cached predictions; verify saved model_summary. Validate50 geometry
records and150 converged seed records for each checked trial, effective dimensions and
selected-feature indices. No individual old models are refitted during verification.
Overall4500 counts are manifest evidence; fit diagnostics are individually inspected for
Trial0 and21, not independent re-execution of all4500 fits.

Trial21 frozen values:
features20, PCA12, K12, tied covariance,
reg_covar=1.4159458366980159e-05,
label_smoothing=1.3711160213584192e-05.
Standalone development AUC0.9954691845996194;
hybrid development AUC0.9837300018821757.
No standalone-winner reselection across30 trials is performed.

Verify prior Trial25 export ready/reload/5-fit accounting, all export artifact hashes,
source input/code hashes, exact same underlying Trial25 study and training file, source
snapshot identity, parameters and full150-column feature order. This uses the actual
schema read from src/submission/elasticnet_trial_submission.py on GitHub. Trial25 model
files are hashed but NEVER loaded or refitted. Their prediction-input hashes are deferred
until actual export, to preserve dry-run no-test access. Same official test/template
content must match those frozen hashes. All recorded original paths must remain valid.

Dry-run checks prediction file EXISTENCE only (filesystem metadata), not test/template
contents. If a purported development-only watched map contains these paths, fail before
hashing them. Full competition input checks are deliberately deferred. Source CSV/NPZ
predictions may be read during dry-run; these are labelled development/export artifacts,
not fitting on competition features. Dry-run writes no output and performs zero fits.

## One refit, two candidates

Fit the exact search projector on all250 labelled rows, then three native mixture models
at seeds42/314/2026. Reuse original source model factory and numerical settings. All3
custom EM fits must converge. This is3 composite fits (each includes density initialization
plus custom EM) and1 projector fit, not6 fits for two submissions. No parameter changes,
seed selection, extra iterations after failure, Optuna search, B mode or test fitting.

Save model_bundle.joblib, freshly reload it, verify TRAIN predictions. Only then open
competition test/template. Hash inputs before reads, require same source Trial25 inputs,
validate numeric finite test features, schema, row count, nonempty unique string IDs and
matching ID sets. Transform by TRAIN-fitted projector, verify original/reloaded TEST
predictions. Never deserialize external/source pickle files: only this run's fresh bundle.

Reconstruct frozen Trial25 test predictions from component_predictions.csv and exact
Elastic-net formula; align by official template IDs, not source row positions. Compute:

    standalone = mean_three_Trial21_GMM_seeds
    hybrid = 0.5*standalone + 0.5*frozen_Trial25

No Trial25 refit. No weight tuning, five-model equal vote or K8 substitution.
Write both17-significant-digit CSVs to pending names, reload/verify IDs/probabilities,
recheck source/input hashes, then publish both. Catchable failure removes both final and
pending CSV names, even if first rename succeeded before second failed. A SIGKILL cannot
execute cleanup: only use candidates when parent manifest is ready_for_manual_submission.

## Outputs

outputs/submissions/generative_optuna_trial21_v1/
    model_bundle.joblib
    standalone/submission.csv
    standalone/candidate_manifest.json
    hybrid/submission.csv
    hybrid/candidate_manifest.json
    aligned_predictions.csv
    geometry_diagnostics.json
    fit_diagnostics.json
    source_decision.json
    source_best_trial.json
    source_model_summary.csv
    settings.json
    prediction_manifest.json

Each candidate manifest has its own CSV SHA256, formula/role, shared model bundle hash and
source Trial25 CSV hash. Parent manifest stores source/input/code hashes and all output
hashes, counters, row count, persistence and roundtrip checks. Its own bytes are not self-
hashed. standalone is explicitly post_search_standalone_followup; hybrid is the candidate
corresponding to the original search objective. Neither is called an independently
validated final winner. Old public0.95267 candidate remains untouched. No upload occurs.

Success: status=ready_for_manual_submission, completed_candidate_fits=3,
completed_projector_fits=1, saved_and_reloaded_models_verified=true,
csv_roundtrip_verified=true, rows=19750, competition_test_used_for_fitting=false,
kaggle_upload_performed=false. No new public/private score is implied by this export.

## Validation limitations

13 tests pass, no skipped tests. Synthetic native tests use real sklearn density-GMM
initialization and custom tied-covariance labelled EM, model save/reload and actual CSV
alignment/blending; source tests exercise full2500-row/50-fold artifacts, selection/ties,
parameter/score/comparison checks, diagnostics, pinned fingerprints, input corruption,
geometry corruption, accounting failures and no-test/no-write dry-run. Publication tests
cover rollback when the second candidate rename fails.

The temporary test checkout reconstructs the prior numerical kernels from earlier supplied
patches. Historical search.snapshot.prepare is explicitly injected in source fixtures,
and frozen production fingerprint/score constants are patched ONLY INSIDE synthetic
fixtures. These temporary legacy modules are NOT included in the patch. Tests do NOT
prove the real local source fingerprint or real0.995469/0.983730 replay; local dry-run
must do that. No Optuna optimizer is run by this exporter or its tests.

Generator was executed and its patch checked/applied in a clean temporary git repository,
then retested. Actual local git-state compatibility, full repository tests, real-data
preflight, real250-row refit, real19750-row outputs and Kaggle upload have NOT run here.
