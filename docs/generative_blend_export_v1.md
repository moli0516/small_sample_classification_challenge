# Fixed blend_gmm0.5 export v1

## Scope

Standalone, additive exporter. Continue the CURRENT branch; do not create a new branch.
Does NOT modify the K8 standalone exporter, Trial25 exporter, registries, old submissions,
models, source manifests, or existing experiment modules. No model fitting, model loading,
pickle/joblib deserialization, hyperparameter search, public-score tuning or network/upload.

Fixed formula ONLY:

    p_new = 0.5*p_A_k8_tied + 0.5*p_elasticnet_optuna_trial_25

Trial25 is the FULL old ensemble plus its tuned Elastic-net addition, not Elastic-net
alone. It is not an equal-weight vote of five individual model branches.

## Apply and run from project root

    git apply --check generative_blend_export_v1.patch
    git apply generative_blend_export_v1.patch
    python -m pytest -q tests/test_generative_blend_export_v1.py
    python -m scripts.export_generative_blend_v1 --dry-run
    python -m scripts.export_generative_blend_v1

Use the downloaded patch location if it is not in project root. Existing numpy/pandas/
sklearn environment is sufficient. No imports from the historical experiment scripts.
No package upgrade needed. --output can specify another fresh destination.
Existing output is refused; no overwrite/resume. Never overwrite either component export.

## Configure actual local locations

configs/generative_blend_export_v1.json defaults:

    gmm_export: outputs/submissions/generative_a_k8_tied_v1
    trial25_export: outputs/submissions/elasticnet_optuna_trial_25_run_01
    blend_study: outputs/results/generative_followup_v1_run_01/blend
    train_path: data/raw/train.csv
    test_path: data/raw/test.csv
    sample_submission_path: data/raw/sample_submission.csv
    expected_rows: 19750
    output_dir: outputs/submissions/generative_blend_gmm05_v1

Correct paths in THIS new config if your local names differ. Do not edit source manifests
or old configs to make hashes pass. A loose CSV with no source export manifest is refused.
Changing the50/50 design is refused. All3 source directories must be distinct. Output
cannot overlap sources, their declared original studies, or official input files.

Required source formats:
- GMM exporter from generative_followup_v1: run_manifest.json, settings.json,
  model_bundle.joblib and submission.csv, plus all other manifest-listed artifacts.
- Original Trial25 exporter from src.submission.elasticnet_trial_submission:
  prediction_manifest.json, submission.csv, component_predictions.csv, ensemble.json,
  source_best_trial.json, source_decision.json, plus all other manifest-listed artifacts.
- Completed fixed-weight blend CV experiment with settings, decision, cohort, splits,
  candidate_specs and predictions.npz.

## Source integrity and selection verification

1. Verify ALL source artifact hashes, including model bytes, without loading models.
2. Verify source watched input/code hashes; reject conflicts, missing/changed files,
   unsafe artifact traversal and hashes not bound to exact source bytes.
3. Require both exports ready_for_manual_submission with saved/reloaded verification,
  19750 rows and no upload. GMM must be train-only,3-seed export; Trial25 must retain
   trial25 identity, exact parameters and its5-estimator-fit accounting.
4. Bind both source exports to the same configured official training CSV hash and the
   same test/template FILE CONTENT hashes. Equivalent paths with identical contents
   can be matched, but all original recorded watched paths must remain valid.
5. Require the GMM export and CV blend study share original GMM study and representation/
   seed design. Verify original GMM selected eligible screening-passed A_k8_tied.
6. Recompute CV probability blends for weights0.25/0.50/0.75,50 fold AUCs,10 repeat means,
   highest-mean selection, fixed0.9382748290356986 selected AUC and both-control screening.
   Verify2500 OOF rows,250 row/ID/label identities, saved partitions and once-per-repeat
   coverage. The CV metrics are development evidence, not independent confirmation.
7. Compare the original K8 and Trial25 source development scores with matched CV controls.
8. Reconstruct Trial25 TEST probabilities from its component_predictions.csv and exact
   alpha formula; verify its standalone submission agrees by ID within1e-12.

Hashes show consistency of recorded artifacts and declared lineage; they are NOT digital
signatures or an independent proof that the original training run used its declared code.
GMM probabilities rely on the existing validated standalone export and its persistence
attestation. GMM model bytes are hashed, never deserialized or independently re-predicted.
Historical model-training preflight is not rerun, because this stage performs no training.
Do not claim this exporter independently reproduced the GMM model's predictions.

## IDs, order and publication

CSV headers must be unique and source submissions exactly id,target. IDs are strings
(to preserve leading zeros), nonempty and unique. Both source sets must match official
test/template IDs exactly. Source row orders need not match: both are explicitly reindexed
by ID to official template order. Record original/template ordered-ID SHA256 and whether
reordering was needed. Never average by unchecked positional row order.

Validate finite probabilities in[0,1]. Trial25 audit values and reconstructed formula
are checked before composition. Test predictor values are not loaded for learning; the
exporter reads official test IDs only, and hashes files for provenance. Template target
placeholder values are ignored. No scaling, fitting, pseudo-labeling or weight changes.

Write17 significant digits to submission.pending.csv, reload and check schema, IDs,
probabilities and exact50/50 result within1e-12. Recheck all source/input/code hashes
immediately before atomic rename to submission.csv. Failure removes both pending and
final CSV names and marks the output failed_or_interrupted. Other diagnostics remain.

Dry-run performs FULL read-only checks, including source predictions, CV data, test IDs
and template. It does zero fits and creates no output directory. Unlike the earlier K8
dry-run, this dry-run truthfully reports competition_test_opened=true for ID/hash checks;
competition_test_used_for_fitting=false. No claim is made that it never opens test.csv.

## Output

outputs/submissions/generative_blend_gmm05_v1/
    submission.csv
    aligned_source_predictions.csv
    lineage.json
    settings.json
    source_cv_decision.json
    prediction_manifest.json

Successful manifest:

    status = ready_for_manual_submission
    candidate = blend_gmm0.5
    fits_performed = 0
    rows = 19750
    csv_roundtrip_verified = true
    source_predictions_verified = true
    competition_test_used_for_fitting = false
    kaggle_upload_performed = false
    automatic_promotion = false

Manifest includes complete watched source/input/code hashes, both source submissions,
source manifest hashes through lineage, official input hashes and new submission SHA256.
Own manifest is not self-hashed. Submission hash also appears in artifact_sha256.
Lineage and aligned_source_predictions permit audit of every final row's calculation.
Nothing is uploaded; manually submit only after reviewing successful manifest.

## Verification scope

20 tests pass, no skips. They exercise the ACTUAL new exporter with synthetic artifacts,
2500-row/50-fold CV records and real CSV files; no legacy implementation stubs are needed.
Coverage includes leading-zero IDs, deliberately different source row orders, template
alignment, exact blend, full hash preservation, source CV reproduction, source corruption,
manifestless CSV rejection, rehashed wrong Trial25 formula, duplicate IDs, different test
features, wrong GMM fit count, CV formula changes, unsafe paths, non50/50 design rejection,
invalid probabilities, no-write CLI dry-run and rollback after input/roundtrip failures.
Model files in fixtures deliberately contain invalid pickle bytes: success confirms that
this exporter does not deserialize them. No model fitting occurs in these tests.

Trial25's actual source export schema was read from the GitHub repository's existing
elasticnet_trial_25_export.patch. GMM export schema follows the previously supplied local
follow-up patch. Full current local Git apply check, production source verification,
real19750-row blend, full repository tests and actual Kaggle upload were NOT executed.
This patch was checked/applied in a clean temporary git checkout and retested there.
