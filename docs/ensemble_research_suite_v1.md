# Ensemble research suite v1

Branch: feat/ensemble-research-suite-v1

All six experiments are implemented in a NEW standalone module. No clil.py, existing
project implementation, registry, source config, historical output or submission is edited.
Earlier PCA patches are not required. No final full-data refit/export is implemented.
The reference is ALWAYS the original equal-weight L1/XGBoost-top20/Trial14 ensemble,
mean CV AUC 0.9083364180103309. It does not change after a screening success.

## Apply

Save/commit previous work before branching. Place the delivered patch at the root or
substitute its actual location:

    git switch -c feat/ensemble-research-suite-v1
    git apply --check ensemble_research_suite_v1.patch
    git apply ensemble_research_suite_v1.patch
    python -m pytest -q tests/test_ensemble_research_suite_v1.py
    python -m scripts.run_ensemble_research_suite_v1 plan

Use the established project environment. Existing xgboost/lightgbm/numpy/pandas/sklearn
versions must remain unchanged because historical provenance verifies them. The optional
new dependency is CatBoost. Inspect pip's proposal before installing:

    python -m pip install --dry-run -r requirements-ensemble-research-suite.txt
    python -m pip install -r requirements-ensemble-research-suite.txt

Do not use --upgrade. If pip proposes changing frozen libraries, stop and resolve dependency
compatibility rather than bypass historical verification. CatBoost is required for `all`,
not for a weights-only run. XGBoost ES requires constructor callable metrics/callbacks
supported by XGBoost >=1.6. No automatic package installation is performed by the runner.

## Check sources and run

    python -m scripts.run_ensemble_research_suite_v1 preflight
    python -m scripts.run_ensemble_research_suite_v1 run --dry-run
    python -m scripts.run_ensemble_research_suite_v1 run --experiment all

The preflight checks all required native imports and the historical source, with ZERO fits
and no output writes. `all --dry-run` defers combination child checks until those children
exist; it prints this explicitly. The real combination verifies them before running.
For one experiment:

    python -m scripts.run_ensemble_research_suite_v1 run --experiment catboost
    python -m scripts.run_ensemble_research_suite_v1 run --experiment weights
    python -m scripts.run_ensemble_research_suite_v1 run --experiment lgbm_es
    python -m scripts.run_ensemble_research_suite_v1 run --experiment xgb_es
    python -m scripts.run_ensemble_research_suite_v1 run --experiment combine
    python -m scripts.run_ensemble_research_suite_v1 run --experiment stacking

Order for `all`: catboost, weights, lgbm_es, xgb_es, combine, stacking. Each child uses a
fresh directory inside outputs/results/ensemble_research_suite_v1_run_01. Native stdout,
stderr and fold progress are written to that child's fit.log. In another terminal:

    tail -f outputs/results/ensemble_research_suite_v1_run_01/catboost/fit.log

No overwrite/resume. If one task fails, completed tasks remain intact; do not rerun `all`
over them. Investigate the failure, preserve its output, and deliberately use a new run
root for a new study, or run only remaining tasks that have no output yet. A fresh root
requires another config with ONLY output_root changed; the design grid is frozen.

## Historical preflight and reference

Requires configs/validation_protocol_v1.json, its original implementation, the completed
validation study, and all frozen registry/Trial14/XGBoost sources that its _prepare needs.
The suite invokes this ORIGINAL verifier instead of duplicating/relaxing its code/data/
version checks. It then verifies every completed validation artifact hash, exact ordered
50 cv5 splits, row IDs/labels, coverage and original component averaging.

The validation fold_scores.csv contains FOUR procedures, not standalone tree scores.
L1 and ensemble AUCs are checked against that table. Native tree AUCs are checked against
verified Trial14 only when ordered train/valid matches are true. No old broken PCA
reference checker is imported. Every new task records source/code/input hashes, versions,
cohort identity and its own artifact hashes. Competition test CSV is never loaded.

## Frozen experiments

### CatBoost: 800 fits; 48 ensemble candidates

16 settings: depth2/4 x iterations100/300 x learning_rate0.03/0.10 x l2_leaf_reg3/10.
Ordered CPU, seed42, one thread, Logloss objective, AUC metric, use_best_model=False,
allow_writing_files=False; no eval_set or early stopping.
Each model is blended at alpha0.05/0.10/0.25:

    (1-alpha)*reference_probability + alpha*catboost_probability

Select directly on ensemble mean; NEVER filter/rank models by standalone AUC first.
Raw CatBoost predictions and standalone component_summary.csv are diagnostic only.

### Weights: zero fits; four candidates

[L1,XGBoost,LightGBM]: equal thirds, [0.5,0.25,0.25], [0.25,0.5,0.25], [0.25,0.25,0.5].
The equal-weight control reuses exact baseline probabilities, avoiding floating-point
recomputation of tied rankings. All candidates use verified held-out source predictions.

### LightGBM early stopping: 250 estimator fits

For each outer training fold, make ONE stratified inner80/20 split with seed
10000+repeat*5+fold. Fit L1 and the original XGBoost top20 procedure on INNER training
only, predict INNER validation, then train LightGBM capped at1000 rounds with patience30.
Stop exclusively on equal-weight INNER ensemble AUC. Native default metric is disabled
(metric='None'); callback uses first_metric_only=True and explicitly maximizes the custom
ensemble metric. Refit only LightGBM on ALL outer training rows at the selected rounds.
Use verified original outer L1/XGBoost predictions for the final outer ensemble evaluation.
Counts per fold: inner L1=1, inner XGBoost ranker+classifier=2, inner LGB=1, refit LGB=1.

### XGBoost early stopping: 300 estimator fits

Same inner split rule. Fit inner L1 and inner LightGBM. Fit a fixed-original-round XGBoost
ranking model on inner training; select top20 original features using the original
GainTopKSelector. Early-stop ONLY its reduced classifier on INNER ensemble AUC, cap1000,
patience30. XGBoost callback explicitly sets maximize=True, named metric='ensemble_auc',
data_name='validation_0', save_best=True. best_iteration is zero-based and converted to
round count with +1. Refit an original-round ranker on ALL outer training rows, recompute
top20, then refit the reduced classifier at selected rounds. Original ranking rounds are
NEVER changed in this experiment. Counts per fold: inner L1=1, inner LGB=1, ranker=1,
inner reduced stop model=1, outer ranker=1, outer reduced refit=1.

Both early-stopping studies retain all other parameters. They save exact inner indices,
selected rounds, selected-at-cap flags and actual refit rounds. No original scoring fold
is used for stopping. Final deployment/full-data rounds policy is intentionally NOT
implemented because this suite is research-only; no submission can be exported from it.

### Combination: zero fits; one candidate or skip

Requires ALL FOUR earlier studies to be completed, even when their screening fails.
Verifies child hashes, design/source/cohort identity, re-scores saved probabilities and
reproduces each child's selected decision. Uses only screening-passed changes.
If fewer than two changes passed, mark task skipped. Otherwise test EXACTLY ONE combination:
all passed early-stopped replacements, passed original-model weights (else equal thirds),
and passed CatBoost model/alpha if applicable. No subset/weight/model search is added.
Interactions are evaluated against the frozen original reference, not assumed additive.
Stacking is separate and is not included in this combination.

### Stacking: 650 estimator fits; one candidate

Each OUTER training fold gets three stratified INNER folds, seed40000+repeat*5+fold.
Fit the original three complete procedures inside each inner training portion, including
scaling/feature selection, to generate one OOF row per outer-training observation.
Fit L2 logistic meta-model C0.1, lbfgs, max_iter5000, tol1e-4, seed42 on these inner OOF
probabilities. No meta-parameter search. Evaluate it on verified original OUTER validation
component probabilities. Those base models used full outer TRAIN rows, not outer valid.
It NEVER trains the meta-model on global OOF predictions and scores those same rows.
Counts per fold:3 inner folds*(L1 1+XGB2+LGB1)+meta1=13, total650.

Total planned successful estimator fits:2000. This counts XGBoost ranker and classifier
separately. Attempted fit count is a procedure-cost budget on failed wrapper calls; a
failed two-stage wrapper may have stopped before its second estimator was attempted.

## Selection, artifacts and interpretation

AUC = mean of5 fold AUCs per repetition, then mean across10 repeats. SD = sample SD across
10 repeat means. Predictions are not pooled into one different scoring metric.
Select highest ENSEMBLE mean; ties use generation order documented in settings: CatBoost
grid order then ascending alpha; weights equal control then L1/XGB/LGB-heavy. No fallback.
Screening: delta>1e-12 and >=6/10 wins. abs(delta)<=1e-12 is a tie. This is descriptive,
not a significance test or an automatic adoption rule. automatic_promotion is ALWAYS false.

Every child writes run_manifest.json, settings.json, cohort.csv, split_indices.json,
inner_traces.json, source_reproduction.csv, warnings.json, fit.log and decision.json.
Completed non-skipped studies add model_summary.csv, fold_scores.csv, repeat_scores.csv,
comparison.csv, paired_repeat_differences.csv and predictions.npz (no pickle).
Where component predictions exist, component_summary.csv is also saved for diagnosis.
Read results, e.g.:

    python clil.py results --run outputs/results/ensemble_research_suite_v1_run_01/catboost

The incumbent stays frozen even if another study passes. Model/weight selection adapts to
reused labelled data; comparisons have unequal search budgets and remain DEVELOPMENT
research. Inner splits prevent direct stopping/meta-training leakage, but do not undo the
previous parameter selection or make the entire process nested/independent evaluation.
ConvergenceWarning fails the run. Only manifest status='completed' is eligible for analysis;
failed/interrupted folders, even with some output files, must never be used for selection.

## Tests and honest limitations

The delivery tests cover complete six-task orchestration with sklearn SURROGATES, expected
fit budgets, early-stopping callback contracts/maximize flags, inner-split isolation,
stacking OOF coverage, source four-procedure schema, source tampering, saved-decision
reproduction, combination gates, exact equal-weight control, no-fit preflight and failures.
Surrogate tests are NOT proof of native-library compatibility or predictive performance.
Two native smoke tests (CatBoost; LightGBM/XGBoost callbacks) are included and skip if the
libraries are unavailable. Run them locally after installing CatBoost. The author sandbox
has none of these three libraries, so native smoke tests and full historical preflight
have NOT run there. Full existing repository tests and real competition runs are also
not claimed. No external GitHub state is modified by delivering this patch.
