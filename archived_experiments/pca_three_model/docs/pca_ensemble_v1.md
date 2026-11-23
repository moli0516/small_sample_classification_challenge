# PCA ensemble study v1

Suggested branch: feat/pca-ensemble-experiment-v1

This additive patch creates an offline ensemble experiment. It does not import a model
trainer, perform fits, load raw train/test CSVs, export submissions, or alter the incumbent.
It uses completed LOCAL artifacts produced by the validation-protocol and PCA-L2 patches.
No existing source files, clil.py, source settings or registries are edited.

## Apply from project root

Commit/preserve your previous PCA changes first, then create the new branch:

    git switch -c feat/pca-ensemble-experiment-v1
    git apply --check pca_ensemble_v1.patch
    git apply pca_ensemble_v1.patch
    python -m pytest -q tests/test_pca_ensemble_v1.py
    python -m scripts.run_pca_ensemble_v1 --dry-run
    python -m scripts.run_pca_ensemble_v1
    python clil.py results --run outputs/results/pca_ensemble_v1_run_01

Place the delivered patch at the project root, or substitute its actual location.
No additional dependencies beyond numpy, pandas, scikit-learn and pytest are required.
The module entry point is intentional: the historical clil.py is not modified.

## Required source runs

outputs/results/validation_protocol_v1_run_01
outputs/results/pca_l2_v1_run_01

Both must be complete, unchanged and accompanied by their original run_manifest.json
artifact hashes. The runner checks every hashed source artifact. PCA settings must record
the exact current validation source hashes, including its manifest; both source settings
must record the same original training CSV digest. No raw CSV is opened by this experiment.

The same 50 ordered cv5 TRAIN and VALID indices must match, not merely their lengths.
Prediction identity is joined by repeat/fold/row_index; id, target, split_seed and validation
coverage must agree. CSV row order may differ and is explicitly aligned by keys. Missing
rows, duplicate keys, inconsistent row identity, invalid probabilities or source changes
fail closed. IDs are loaded as strings (leading zeros preserved).

The original equal-weight three-component probability average is reproduced. All 50
baseline fold AUCs and its expected mean 0.9083364180103309 must reproduce within 1e-12.
Selected PCA and no-PCA fold scores, repeat scores, means and SDs are also reproduced.
The frozen candidate names must match both the source branch maxima and decision.json.
If a check fails, inspect the cause; do not widen tolerance or regenerate historical files.

## Frozen design

PCA component: pca_50_C0.01
No-PCA L2 component: no_pca_C0.001
Baseline: l1_xgb_lgbm, each original branch retains the same relative weight.
For each new component q and alpha in [0.05, 0.10, 0.25]:

    p_blend = (1-alpha)*p_baseline + alpha*q

Six blends total. alpha=0.25 is four-model equal weighting. This is adding a fourth branch,
NOT applying PCA to the existing three branches or replacing their input features.
No additional component-count or C search is performed in this experiment.

Report six blend-versus-baseline comparisons and three PCA-versus-no-PCA contrasts at
matched alpha. Score is the mean of five fold AUCs per repetition, averaged over ten
repetitions; SD is the sample SD across the ten repeat means. Probability predictions are
not pooled into one AUC. Wins/losses/ties use a 1e-12 tolerance.

Select the highest mean across SIX blends. Exact ties use PCA first, ascending alpha,
then no-PCA ascending alpha. A descriptive screening flag requires mean delta>1e-12 and
at least six repeat wins versus baseline; no fallback selection. Automatic promotion is
always false. A passing no-PCA blend is not evidence that PCA helped. Use matched-alpha
comparisons to describe PCA-specific evidence without selecting a convenient alternate.

The previous standalone PCA choice and these six comparisons adaptively reuse the same
labelled dataset. Results are development evidence, not nested CV, an independent
estimate or a significance test. Negative results apply to these frozen branches/weights.

## Outputs

run_manifest.json, settings.json, split_indices.json, decision.json
model_summary.csv: baseline, two standalone components, six blends (9 rows)
fold_scores.csv: 9 candidates x 50 folds (450 rows)
repeat_scores.csv: 9 candidates x 10 repetitions (90 rows)
comparison.csv: six baseline contrasts plus three matched-alpha contrasts (9 rows)
paired_repeat_differences.csv: 90 rows
source_reproduction.csv: 150 verified source fold comparisons
validation_probabilities.csv: 2,500 held-out validation rows with all nine probabilities

Dry-run performs all source checks but writes nothing and makes no fits.
Existing output directories are rejected, including empty directories. To run another:

    python -m scripts.run_pca_ensemble_v1 --output outputs/results/pca_ensemble_v1_run_02 --dry-run
    python -m scripts.run_pca_ensemble_v1 --output outputs/results/pca_ensemble_v1_run_02

Tests run on synthetic artifacts only. The real local artifacts and full historical project
test suite have not been executed by the patch author. A local preflight remains required.
