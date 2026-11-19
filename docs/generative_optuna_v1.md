# Generative hybrid Optuna v1

## Apply on CURRENT branch

Only7 additive files. No historical code/output/exporter/registry edits, new branch,
remote writes, test fitting, submission export or upload. Requires the local preceding
patches: generative_geometry_v1 and generative_followup_v1. Their modules/artifacts were
not found on remote main or the two checked elasticnet branches. The local sources must
therefore pass the actual preflight; remote inspection was NOT full local integration.

    git apply --check generative_optuna_v1.patch
    git apply generative_optuna_v1.patch
    python -m pytest -q tests/test_generative_optuna_v1.py
    python -m scripts.run_generative_optuna_v1 --dry-run
    python -m scripts.run_generative_optuna_v1

Use the actual downloaded patch location if needed. Keep existing environment with
Optuna installed; do not upgrade frozen packages. No resume/overwrite. --output can name
another fresh directory. Config defaults to configs/generative_optuna_v1.json:
followup_config=configs/generative_followup_v1.json,
blend_study=outputs/results/generative_followup_v1_run_01/blend,
output_dir=outputs/results/generative_optuna_v1_run_01.
Correct only source/destination paths in the NEW config to match local artifacts.
The old followup config identifies the original GMM study (including AB source if used).
Do not alter recorded old settings/manifests to bypass a failed check.

## Source verification

Snapshot calls the existing frozen followup.prepare(action='blend') to verify labelled
training data, Trial25/K8 provenance, versions, IDs/labels/feature order and saved50 folds.
This research-mode adapter never opens test or template. The new snapshot additionally
verifies the completed zero-fit/no-test blending run, all hashed artifacts and watched
source inputs/code, source config/design/lineage/version agreement, exact cohort/splits,
row-level K8/Trial25 arrays and all3 fixed CV blend formulas. It recomputes highest-mean
selection, recorded paired statistics and both-control screening.
Expected source AUCs: GMM0.9296401907271473, hybrid0.9382748290356986, tolerance1e-12.
Conflicting hashes are fatal. Paths cannot overlap sources/watched inputs. New code and
config are hashed. This is consistency/provenance verification, not a digital signature.

## Frozen search

Objective: p=.5*p_GMM_theta+.5*p_frozen_Trial25; mean5 fold AUCs per repetition, then
mean10 repetition scores. Same saved50 splits, tied covariance, original geometry ranking
formula, unwhitened PCA, fixed initialization seeds42/314/2026 averaged equally.
Same original max_iter/tol and other numerical settings. No seed selection or B mode.

- selected_features: categorical10/20/30/50
- latent_dimensions: categorical4/6/8/12/16
- n_components: categorical4/8/12/16
- reg_covar: log1e-5..1e-1
- label_smoothing: log1e-5..1e-2

SearchGeometryProjector subclasses the OLD projector and does not rewrite its arithmetic.
Effective dimensions=min(requested,nonconstant selected feature count,n_train-1).
Record requested/effective feature counts/dimensions, selected columns and explained
variance. Different requests may share an effective dimension; all still count as trials.
Every scaler/ranking/PCA/mixture is fitted on that fold's training portion only.
No competition-test parameter or code path exists. Dry-run imports native Optuna, runs
read-only source checks, performs zero fits and writes no output. Does not read test.

TPE(seed42,n_startup_trials10), n_jobs1, NopPruner. No reporting/pruning or early stopping.
Native Optuna create_study in memory, fresh30 TOTAL optimization attempts including all
failures and Trial0. Failed trials are not replaced. No hidden database/resume budget.
TPE behavior after rejected trials is native Optuna's, not an invented replacement policy.

Control0 queued first: features20, dimensions8, K8, reg_covar0.001, smoothing0.001.
Must converge and reproduce BOTH standalone/hybrid mean AUC and average prediction for
EVERY saved OOF row. Failure stops the whole study immediately. Seeds are averaged;
old source contains averaged arrays, so this check does not claim individual seed-array
reproduction against nonexistent source seed files.

At most30*50*3=4500 composite candidate fits, each including ONE sklearn density-GMM
initialization plus ONE custom labelled EM. At most1500 projector fits. Trial25 is
never retrained. Runtime is not promised. Per-seed records flush immediately, per-fold
progress is printed and manifest updated. Sources are rechecked throughout and before
selection. Parallel threads are limited to1 in native fitting.

## Failures and counters

Custom-EM nonconvergence, numpy LinAlgError, FloatingPointError and the specific sklearn
ValueError containing 'ill-defined empirical covariance' are rejected candidates (FAIL,
no objective value). All other ValueErrors/data/hash/ID/split/API/programming errors are
fatal and stop the study. Trial0 numerical/nonconvergence rejection is always fatal.
Initialization warnings are recorded; they do not alone invalidate converged final EM,
consistent with the frozen v1 procedure. Numerical ridge/EM limitations remain unchanged;
no monotonic-objective guarantee is claimed.

Attempted/completed/rejected/failed trial counts are separate. Failed includes rejected
and fatal FAIL states; rejected counts only explicitly classified numerical/convergence
rejections. Fit counters: attempted_candidate_fits, completed_candidate_fits (fit returned),
converged_candidate_fits; attempted/completed_projector_fits separately. A returned but
nonconverged fit is completed but not converged, and its whole trial has no eligible score.
Partial prediction arrays/errors are diagnostic only. Do not treat them as scored OOF.
Rejected trials terminate early; later folds are not fitted and do not receive scores.
Completed study can contain FAIL trials but must account for all30 attempts.
Fatal/interrupted study has no final decision/best_trial; preserve its diagnostics.

## Selection

Choose highest mean HYBRID AUC among fully completed/converged trials, exact ties earliest.
Then compare against original hybrid (primary), old standalone K8, Trial25, and compare new
standalone GMM against old standalone K8. Primary pass: paired mean>1e-12 AND>=6/10 wins
against original hybrid. Never choose a fallback or append trials after seeing failure.
If Trial0 wins, selected_is_control=true, screening_passed=null,
screening_status=not_applicable_control_selected. automatic_promotion=false always.
No exporter or public-score feedback is used. Keep public0.95267 incumbent untouched.
These are adaptive reused-label development estimates, not independent/nested CV,
statistical significance or a guarantee of private improvement.

## Artifacts

settings.json, run_manifest.json, source_reproduction.csv, cohort.csv, split_indices.json,
trials.csv, trials.json, best_trial.json, decision.json; no native study persistence/resume.
Each trial: parameters.json, geometry_diagnostics.jsonl, fit_diagnostics.jsonl;
complete trial additionally fold_scores.csv, repeat_scores.csv, model_summary.csv,
paired_comparisons.csv, paired_repeat_differences.csv, predictions.npz, result.json.
Failed trial: error.json and partial_predictions.npz; no invented full scores.
Successful study hashes all artifacts recursively (manifest excludes itself).
No final-fitted model or submission exists in this output.

## Validation and known unverified work

19 tests pass;1 native Optuna end-to-end test is skipped in the artifact-generation
environment because Optuna is not installed. The skipped test is provided and should
execute in your existing Optuna environment. Synthetic orchestration uses an explicitly
labelled API test double, NOT native TPE. Tests exercise real sklearn GMM initialization,
weighted custom EM, old/new geometry parity, clipping, seed averaging, full row-level
control reproduction, trial rejection/budget accounting, fatal-data behavior, control
failure, ties, source tampering/conflicts, no-write CLI and 50-fold source replay.

The temporary test checkout reconstructs v1 numerical kernels/scoring from the earlier
supplied patch. Its historical followup preflight is explicitly injected in synthetic
source tests. These TEMPORARY legacy fixtures are not included in the delivered patch.
This is NOT testing against fetched current local v1 modules: the remote paths were absent.
No claim of native TPE execution, real-data Trial0 reproduction,30-attempt/4500-fit run,
full repository test success or actual local git-state compatibility is made.
The patch is checked/applied in a clean temporary Git repository and retested there.
