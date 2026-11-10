# Elastic-net ensemble Optuna v1

Continue on your CURRENT research branch. No new branch is required.
This patch only adds four files and preserves the completed grid study and all incumbent
source code/hashes. Requires run_elasticnet_ensemble_v1.py, the original research suite,
and their completed historical CatBoost/reference artifacts. Use your existing Optuna
environment; do not upgrade frozen sklearn/native libraries or historical source files.

## Apply and run from project root

    git apply --check elasticnet_optuna_v1.patch
    git apply elasticnet_optuna_v1.patch
    python -m pytest -q tests/test_elasticnet_optuna_v1.py
    python -m scripts.run_elasticnet_optuna_v1 --dry-run
    python -m scripts.run_elasticnet_optuna_v1
    python clil.py results --run outputs/results/elasticnet_optuna_v1_run_01

Use the actual downloaded patch location if needed. No git switch command is necessary.
Fresh output only, no overwrite/resume. --output can specify another nonexisting folder.
Dry-run imports Optuna and validates historical sources but does zero fits and no writes.
The new output destination is explicitly passed into the old grid prepare method; the
completed elasticnet_v1 result folder is not overwritten or incorrectly rejected.

## Frozen design

30 total trials, INCLUDING queued Trial0. TPE seed42, n_startup_trials10, sequential n_jobs1,
NopPruner, no pruning and no early stopping. All trials use ALL50 saved CV assignments.
Maximum1500 new logistic-classifier fits. Existing tree/CatBoost predictions are reused.
StandardScaler -> elastic-net logistic regression, saga, max_iter20000, tol1e-4, seed42;
these settings come from the prior module. Scaler fits within training fold only.

C:0.001..1, log sampling.
l1_ratio:0.50..1.00, uniform (1.0 permits pure L1 boundary).
alpha:0.01..0.20, uniform.
Only ADDITION to the original three-model ensemble is searched:

    p_candidate = (1-alpha)*p_original + alpha*p_elasticnet

No replacement mode, no additional CatBoost branch inside these candidate formulas,
no simultaneous changes to other model parameters. Frozen CatBoost blend is SECOND control.
Objective is ensemble mean AUC, not standalone AUC. Standalone is diagnostic only.

Trial0: C0.1, l1_ratio0.75, alpha0.05 must reproduce0.9084647196603719 within1e-12.
Failure stops the WHOLE study, with no candidate selection. It is one of the30 trials,
not an extra31st fit. Trials1..29 are suggested by TPE (startup policy unchanged).
Sampler seed does not eliminate all environment/version effects; original environment
verification and fixed estimator seed remain necessary. No promise about actual runtime.

Score: mean of5 fold AUCs per repeat, then mean of10 repeats. Sample SD over repeat means.
Select max objective among30 COMPLETE trials, exact ties earliest trial. No fallback.
Report selected trial paired deltas/SD/wins/losses/ties versus original AND frozen CatBoost.
Screening requires positive mean>1e-12 and>=6 wins; both-control flag separate.
automatic_promotion always false. No final submission/refit implementation.

## Artifacts

run_manifest.json, settings.json, cohort.csv, split_indices.json
trials.csv, trials.json, best_trial.json, decision.json
trials/trial_0000..0029/parameters.json, warnings.json, result.json, diagnostics.json
Each complete trial also writes fold_scores.csv, repeat_scores.csv, model_summary.csv,
comparison.csv, comparison_vs_catboost.csv, paired_repeat_differences.csv,
paired_vs_catboost.csv, predictions.npz. All hashed recursively at successful completion.
ConvergenceWarning or other exception stops the study; no catch-and-continue on partial
results. Failed/interrupted output is diagnostic only, not eligible for final selection.

The search space was chosen after seeing the earlier grid. This is adaptive development
tuning on reused labelled data, NOT independent confirmation or nested CV. Increasing
search budget cannot by itself establish real generalisation improvement. Do not use
public leaderboard feedback inside the objective or extend the budget after inspecting
results without declaring another study. Existing incumbent remains unchanged.

## Validation status

Tests cover bounds/log suggestions, queued-control contract, real logistic fits through a
three-trial synthetic optimization loop (150 classifier fits), control-failure stop,
no partial selection, earliest-trial ties, output destination handling and frozen config.
In the author environment Optuna is not installed; optimization orchestration is exercised
with a small Optuna API test double and a minimal Elastic-net source contract. This is NOT
native TPE integration. A native Optuna smoke test is included and skipped there. The real
local source preflight, full30-trial study and full repository tests have not been executed.
