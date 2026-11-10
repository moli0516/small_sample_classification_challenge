# Elastic-net logistic ensemble v1

Branch: feat/elasticnet-ensemble-experiment-v1

This additive experiment requires the existing ensemble research suite plus completed
catboost source artifacts. It does NOT modify suite code or retrain CatBoost/trees.
No new dependency: uses the established sklearn environment. Do not upgrade frozen libraries.

    git switch -c feat/elasticnet-ensemble-experiment-v1
    git apply --check elasticnet_ensemble_v1.patch
    git apply elasticnet_ensemble_v1.patch
    python -m pytest -q tests/test_elasticnet_ensemble_v1.py
    python -m scripts.run_elasticnet_ensemble_v1 --dry-run
    python -m scripts.run_elasticnet_ensemble_v1

Use actual patch location if not at project root. Preserve earlier work before branching.
Default source config: configs/ensemble_research_suite_v1.json.
Default output: suite output_root / elasticnet_v1 (must not exist).
--output accepts another fresh directory; no overwrite/resume.

## Frozen design

StandardScaler -> LogisticRegression, elastic-net penalty, saga, seed42, max_iter20000,
tol1e-4. The implementation supports old penalty='elasticnet' and newer sklearn l1_ratio
APIs by inspecting the existing signature. This is LOGISTIC classification, not sklearn's
ElasticNet regression estimator. Scaler fits strictly within each training fold.
C=[0.1,1,10], l1_ratio=[0.25,0.5,0.75]:9 models,50 shared historical folds,450 actual fits.
Each saved raw prediction is evaluated in four ways:

1. Replace original L1: (EN+original XGB+original LGB)/3.
2-4. Add EN to original three-model average at alpha0.05/0.10/0.25:
     (1-alpha)*original_reference + alpha*EN.

36 ensemble candidates. These candidates do NOT additionally contain CatBoost. The frozen
CatBoost blend is a second control; no new five-model search is performed here.
Standalones are diagnostic only, never used to pre-filter/select the9 settings.
Select highest ensemble mean across all36; exact ties: C ascending, ratio ascending,
replacement before alpha ascending. No fallback based on wins after observing results.
Primary reference is original three-model ensemble; secondary control is the selected
cat_10_alpha0.1 blend. Separate screening flags require positive mean delta>1e-12 and
>=6/10 repeat wins versus each control. automatic_promotion is always false.

Score: mean of five fold AUCs per repeat, then mean over10 repeats. SD is sample repeat SD.
Existing suite preparation verifies original source hashes/versions/data/splits/probabilities
and AUCs. suite.load_child verifies CatBoost artifacts/cohort/selected-decision reproduction.
The selected CatBoost score is checked again by this module before any new fits.
ConvergenceWarning stops the run; failed/partial output must not guide candidate selection.

## Artifacts

model_summary.csv:38 rows (36 ensembles + two controls)
comparison.csv:36 paired comparisons vs original
comparison_vs_catboost.csv:36 vs frozen CatBoost blend
fold_scores.csv; repeat_scores.csv; paired_repeat_differences.csv; paired_vs_catboost.csv
standalone_summary.csv:9 raw EN summaries, not selection objective
coefficient_diagnostics.json:per-fold nonzero counts, feature names, solver iteration counts
candidate_specs.json; predictions.npz; cohort.csv; split_indices.json
settings.json; warnings.json; decision.json; run_manifest.json with output hashes

    python clil.py results --run outputs/results/ensemble_research_suite_v1_run_01/elasticnet_v1

No competition test data, submission export, or final full-data refit. This is another
adaptive development study on previously reused labelled observations, not independent
confirmation or a guarantee of leaderboard gain. Coefficient sparsity does not establish
that selected columns are the true relevant features. Training cost is not benchmarked
on the user's real dataset;450 fits are not promised to finish within a particular time.

## Tests

Nine tests include450 REAL sklearn elastic-net logistic fits on synthetic data, exact
ensemble formulas, training-only scaling, warnings/failure handling, no overwrite,
probability/hash guards, tie handling and a mocked zero-training source adapter.
No CatBoost or native tree training is needed. The real historical-source preflight and
full existing repository test suite have not been run in the author environment.
