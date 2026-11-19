# Trial21 zero-fit weights v1: comparison and selected export

## Additive patch / current branch

Requires the ALREADY APPLIED trial21_export.patch and earlier local experiment modules.
Does not replace, edit or rerun the Trial21 exporter, prior experiments, model registries,
old CSVs or public0.95267 candidate. Continue current branch. No new model fitting, model
loading/pickle deserialization, parameter search, automatic promotion or upload.

The remote default branch did not expose generative_trial21_snapshot_v1.py during the
read attempt. Runtime uses your existing LOCAL helper API and its strict source preflight.
The delivered tests explicitly inject synthetic historical source preparation; they do
not claim access to or verification of current local artifacts.

## Apply and compare

    git apply --check trial21_weights_v1.patch
    git apply trial21_weights_v1.patch
    python -m pytest -q tests/test_trial21_weights_v1.py
    python -m scripts.run_trial21_weights_v1 compare --dry-run
    python -m scripts.run_trial21_weights_v1 compare

Only5 new files: src/experiments/trial21_weights_v1.py (comparison and zero-fit export),
scripts/run_trial21_weights_v1.py (thin CLI), config, tests and this documentation.
No source edits to installed Trial21 snapshot/export files. Existing numpy/pandas/sklearn
environment; no upgrades. --output can select fresh output, never overwrite/resume.

New config: configs/trial21_weights_v1.json
trial21_export_config=configs/generative_trial21_export_v1.json
trial21_export=outputs/submissions/generative_optuna_trial21_v1
old_hybrid_export=outputs/submissions/generative_blend_gmm05_v1
comparison_output=outputs/results/trial21_weights_v1_run_01
export_output=outputs/submissions/trial21_weights_v1_run_01

Correct these NEW config paths before comparison. If using --output for comparison,
export must be configured to read that actual comparison_output BEFORE comparison runs;
do not change config after freezing a comparison, because its hash is bound to artifacts.
Simplest: use configured default directories for both actions. Changing the alpha grid
is refused in v1. Paths to sources and destination cannot overlap.

## Fixed definitions

G_new=the frozen Trial21 standalone mean of seeds42/314/2026.
G_old=old training-only geometry/K8 tied GMM.
T=FULL frozen Elastic-net Trial25 ensemble, not Elastic-net alone.
H_old=.5*G_old+.5*T (the older public0.95267 candidate).
H_new=.5*G_new+.5*T (original generative Optuna objective candidate).

Family A: alpha*G_new+(1-alpha)*H_old.
Family B: alpha*G_new+(1-alpha)*T.
Both alpha={0.50,0.75,0.90,0.95}; standalone G_new is also selectable.

Exactly9 selectable recipes. H_new is an alias of B_a0.5, not a tenth candidate. Save
family/formula/alpha plus effective G_new/G_old/T weights. Keep the original family
arithmetic for both CV and export, rather than changing floating-point association by
summing the three expanded weights. No unrestricted three-way weight search or public-
feedback adjustment. No search over other trials/seeds. Model fits=0 for both actions.

## Comparison verification and selection

Calls the existing Trial21 source verifier for30-trial/4500-fit metadata, original
cohort/splits/IDs/labels/feature order, hashes, environments, Trial0 and Trial21 scores,
row predictions and selection. Reads frozen Trial21 NPZ and older verified CV arrays.
Rechecks H_old formula and H_new/B_a0.5 equality. Does not open competition test/template,
fit any model, require existing new Trial21 test exports or automatically trigger export.

Score remains mean5 fold AUCs per repeat then mean10 repeats; sample SD of repeat means.
No pooled OOF AUC replacement. Save folds/repeats/summaries, paired per-repeat differences
and paired statistics against G_new, H_old and H_new. Self-comparisons are omitted.

Highest mean selects NUMERICAL winner. Exact ties: G_new, then A in alpha order, then B
in alpha order. If G_new wins, selected_is_control=true and screening_passed=null,
not_applicable_control_selected. A winning blend must have positive mean>1e-12 and>=6
wins against G_new to pass. Report H_old screening separately, not as a substitute.

If winning blend fails, retained_candidate=G_new. Do not pick the second blend or extend
grid. selected_candidate and retained_candidate therefore intentionally may differ.
No automatic model/submission promotion. Post-search reused-label evidence, not nested,
independent evaluation, causal attribution or a significance test.

## Comparison artifacts

outputs/results/trial21_weights_v1_run_01/
    settings.json
    candidate_specs.json / candidate_specs.csv
    cohort.csv / split_indices.json
    fold_scores.csv / repeat_scores.csv / model_summary.csv
    paired_comparisons.csv / paired_repeat_differences.csv
    predictions.npz
    decision.json
    run_manifest.json

Settings/source hashes freeze the comparison. Interrupted outputs are diagnostic only;
no final decision is retained. Successful run hashes all artifacts, excluding manifest.

## Export prerequisite: finish original Trial21 export first

Merely applying trial21_export.patch does NOT produce test predictions. If they do not
exist, run the already installed exporter separately:

    python -m scripts.export_generative_trial21_v1 --dry-run
    python -m scripts.export_generative_trial21_v1

That older command performs its own three GMM fits. THIS patch never invokes it, and
its compare/export fits_performed remains0. Keep old files unchanged.

Then explicitly invoke the new exporter:

    python -m scripts.run_trial21_weights_v1 export --dry-run
    python -m scripts.run_trial21_weights_v1 export

Before using test predictions, verify completed comparison artifact hashes, unchanged
config, exact cohort/splits, re-evaluated all9 candidates and exact decision replay,
plus every saved candidate probability formula. No weight reselection from test/public.

Export ONLY retained_candidate as a new standalone CSV in export_output. If standalone
wins or the numerical winning blend fails screening, export the verified standalone
predictions (no refit), not a failing blend. Does not output all8 weights or overwrite
existing standalone/H_new files. Those already remain in the old Trial21 export folder.

## Test prediction sources and audit

Always require ready Trial21 export,3 fits, saved-model reload attestation, exact selected
parameters/seeds/tied covariance, same source best/decision, all artifact/input/code hashes.
Verify its standalone,H_new CSVs and aligned audit with exact formulas. Hash model bytes
but NEVER load model_bundle.joblib. Source attestation is not independent model replay.

For B: require frozen T export, same source, prediction-input hashes, IDs and component
formula; it must match Trial21 audit's T probabilities.
For A: additionally require ready OLD hybrid export. Verify its .5/.5 weights, zero-fit
accounting, old CV mean/candidate, T fingerprint, all artifact/input/code hashes, aligned
oldG/T/formula audit, and original K8 export's ready/reload status, source geometry study,
source design, model/submission fingerprints and exact original K8 probabilities.
Missing sources fail; never fit old models or silently substitute formulas/other blends.

Official test file is hashed and only ID column is loaded, no feature learning. Template
ID order is canonical. Unique nonempty string IDs preserve leading zeros. Sets/row counts
must match; source CSV orders may differ and are explicitly reindexed by ID. Validate
finite probabilities in[0,1], write17 significant digits, reload/check, rehash all sources,
then rename pending CSV. Catchable failure removes final/pending names. SIGKILL requires
manual inspection: only use outputs marked ready_for_manual_submission.

Export artifacts:
    submission.csv
    aligned_predictions.csv
    source_decision.json
    lineage.json
    prediction_manifest.json

Manifest records retained recipe, source/input/code hashes, new submission SHA256,
fits_performed=0, competition_test_used_for_fitting=false and no upload/promotion.
No new public/private performance is implied.

## Dry-run semantics

Both dry-runs do0 fits, write no directory and do not read competition test/template.
Compare validates labelled CV sources. Export additionally verifies/replays the frozen
comparison and checks REQUIRED source export file EXISTENCE. Actual export-artifact
contents and competition prediction-input checks are explicitly deferred to formal
export; dry-run does not certify those deferred checks. Snapshot helper may hash/read
historical prediction artifacts, not raw competition features. The existing Trial21
helper itself verifies the older T export for provenance, even if standalone is selected;
new numeric source loading of T/oldG happens only when retained coefficients require it.

## Test scope

22 tests pass, no skips. Actual new implementation handles synthetic2500-row/50-fold
CV arrays and real CSV fixtures, formulas/expanded weights/alias, standalone/A/B ties,
screening rejection without fallback, full compare->frozen replay->export, leading-zero
IDs/different row orders, source/formula fingerprints, missing A source, B without old
hybrid, tampering, no-write/no-test dry-runs and zero fit/pickle use.

The old installed snapshot prepare is explicitly injected and its I/O/scoring helper
contract reconstructed in a temporary checkout; no temporary helper is delivered or
modifies local v1. Fixture model files are invalid pickle bytes to ensure no loading.
No real source artifacts, real full repository test suite or actual19750-row export were
run here. Clean temporary git apply and byte identity were checked, then tests rerun.
Local source preflight and real comparison results remain your next execution steps.
