# Frozen generative K8 export and three independent follow-up experiments

Continue the CURRENT branch. This patch adds7 files, does not modify v1 source,
source results, incumbent registry, old submissions or cli.py, and never uploads.
Depends on the preceding generative_geometry_v1 patch in your LOCAL checkout. Its
modules were not found on the remote default branch when preparing this extension.
Do not apply this patch as a replacement for v1. It reuses its actual package APIs.

## Apply / test

    git apply --check generative_followup_v1.patch
    git apply generative_followup_v1.patch
    python -m pytest -q tests/test_generative_followup_v1.py

Thin CLI calls src.experiments.generative_followup_v1; export logic lives in
src.submission.generative_export_v1. Plain PCA control lives in src.features.
No new branch. No package upgrades. Existing joblib/scipy/sklearn environment required.

## Source selection and preflight

Default generative source: outputs/results/generative_geometry_v1_run_01.
If you used the separate AB folder, append this to EVERY command:

    --source-study outputs/results/generative_geometry_ab_v1_run_01

Use the actual path you ran; the file names above are not assertions that both exist.
Preflight delegates the labelled-data/Trial25 checks to the existing v1 prepare API,
then verifies ALL generative manifest artifacts, fit-budget completion, frozen source
settings, source package versions, source input/code hashes, exact cohort/split identity,
reference prediction equality, source selection/eligibility, and recomputed K8 AUC.
The expected K8 mean is0.9296401907271473 (tolerance1e-12).
The original source must have selected eligible, screening-passed A_k8_tied.
Other source studies or changed winners are deliberately refused by this v1 extension.
This freezes the experiment you supplied; it is not a generic any-winner exporter.

Preflight writes no output and does no fitting. Export dry-run checks existence of
configured test/template paths without reading their contents. The default template
path data/raw/sample_submission.csv is configurable; correct it if your local template
is elsewhere. Research modes do not require or read test/template files.

## 1. Frozen standalone submission FIRST

    python -m scripts.run_generative_followup_v1 export --dry-run
    python -m scripts.run_generative_followup_v1 export

Default output: outputs/submissions/generative_a_k8_tied_v1.
Full-data refit of geometry on250 labelled rows, then3 tied-covariance K8 models using
fixed seeds42/314/2026. No test features used for fitting; no new parameter choices.
Exactly3 composite candidate fits plus1 projector fit; each candidate fit includes
sklearn density initialization and custom EM. All3 final EM fits must converge.

Store model_bundle.joblib and check train predictions after fresh reload BEFORE first
opening test. Read/validate test and template, reorder columns by TRAIN feature names,
predict with original and reloaded bundles, average all3 seeds, align rows by template
ID, write id,target CSV, read it back and verify identity/probabilities. Test features
are never passed to fit. Training-only scaling/geometry is retained. Test/template
inputs are hashed before reading and checked after prediction. Source inputs/code
are checked before final CSV export. Failed exports remove submission.csv.

Require19750 unique nonempty test/template IDs with matching sets, exact predictor
schema, numeric finite feature values, finite[0,1] probabilities, saved/reloaded
agreement within1e-12. Template target values are ignored. Local template ID order
is preserved even if test order differs. CSV uses17 significant digits.

Success: run_manifest.json status=ready_for_manual_submission,
saved_and_reloaded_models_verified=true, rows=19750,
competition_test_used_for_fitting=false, completed_candidate_fits=3.
No competition upload occurs. Only load this run's newly generated local joblib file;
never load untrusted pickle/joblib models. Preserve public0.92042 incumbent separately.

## 2. Capacity extension (independent output; does NOT alter export)

    python -m scripts.run_generative_followup_v1 capacity --dry-run
    python -m scripts.run_generative_followup_v1 capacity

Fixed geometry, tied covariance, original numerical settings and fixed3-seed averaging.
K8 is reused from hash-verified source predictions; K12/K16 are newly fitted under the
same50 folds. New fits:2*3*50=300 composite fits;100 projector fits.
No seed selection, weight tuning, early stopping or Optuna. Candidates must converge
on all new folds/seeds to be eligible. Highest eligible mean, exact ties original K8
first then K12/K16. Cached K8 is not refitted merely to enlarge the compute budget.

## 3. Representation ablation (independent output)

    python -m scripts.run_generative_followup_v1 ablation --dry-run
    python -m scripts.run_generative_followup_v1 ablation

Reuse frozen geometry-K8. Compare it with train-only StandardScaler+plain PCA on all
predictors, using the same latent dimension cap8, K8 tied model and3 fixed seeds.
New fits:1*3*50=150 composite fits;50 projector fits. This isolates the representation
pipeline choice as a whole; it does not separately attribute feature-ranking versus
PCA effects. No claim of recovered true informative variables.

## 4. Complementarity and fixed-grid blending (independent output)

    python -m scripts.run_generative_followup_v1 blend --dry-run
    python -m scripts.run_generative_followup_v1 blend

Use ONLY frozen aligned K8 and original Trial25 CV predictions. No new fits or test access.
Try fixed GMM weights0.25/0.50/0.75:

    p_blend = gmm_weight*p_K8 + (1-gmm_weight)*p_Trial25

No new weights selected from public feedback. Report comparisons versus BOTH controls,
per-fold prediction Spearman and threshold0.5 error/disagreement diagnostics. The
threshold diagnostics are not AUC objectives or calibration claims. Primary score
remains mean5 fold AUCs per repetition, then mean10 repetitions, not pooled AUC.
Select highest of the3 fixed blends, ties weight order; screening against both controls
requires positive mean>1e-12 and>=6 repetition wins against EACH. Failure is preserved;
no weight expansion/fallback. Endpoints remain controls, not searched blend candidates.

## Outputs and limitations

Research outputs: outputs/results/generative_followup_v1_run_01/{capacity,ablation,blend}.
Each has settings/cohort/splits/predictions, fold/repeat/model_summary/comparison CSVs,
candidate_specs.json, decision.json, diagnostics and run_manifest with artifact hashes.
All modes refuse existing output folders. --output may specify another fresh location.
No resume/overwrite. Numerical errors stop the run; nonconverged research candidates
are ineligible but preserved. No automatic promotion. Research winners cannot change
the frozen K8 exporter or trigger submissions; a future exporter would need a new
explicit recipe. The original custom-EM covariance-ridge limitations remain unchanged.

These are reused-label adaptive development experiments, NOT independent confirmation,
nested CV or significance tests. Agreement on10 repetitions does not prove future/private
superiority. Do not compare internal0.929640 with public0.92042 as if on the same set.
Stage C semi-supervised fitting on competition test remains unimplemented and permission
unconfirmed. This export uses test ONLY for prediction/format checks, not training.

## Verification scope

14 tests pass. Native synthetic tests exercise24 capacity fits,12 ablation fits and3
full-data-export fits (39 composite fits, each with real sklearn initialization/custom
weighted EM), saved-model reloads, template reordering, nonconvergence rejection, input
schema/ID rejection, blending with no fits/test files, source hashes/winner/cohort/version
checks, and no-write preflight behavior.

Remote default branch did not contain the three local v1 modules. For artifact-generation
testing, their model/projector/scoring logic was reconstructed from the previous supplied
patch. The legacy labelled/Trial25 preflight was explicitly injected in fixture tests;
that is NOT a full production-source integration test. Nothing from these temporary
legacy fixtures is included in this patch. Your LOCAL v1 files remain the actual runtime
dependencies, and their recorded source hashes must pass.

git apply was checked/applied in a clean temporary checkout, not your current local Git
state. Full repository tests, real-data preflight, production450 follow-up fits, and real
19750-row export have NOT been executed here. Actual runtime is not promised.
