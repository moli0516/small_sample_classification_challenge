# PCA representations for three frozen models — v1

Suggested branch: feat/pca-three-model-representation-v1

This experiment trains L1, XGBoost and LightGBM ON PCA SCORES and averages their positive
probabilities. It does NOT add PCA-L2 as a fourth branch and does not depend on the previous
PCA-L2 or PCA-ensemble patch. All existing source files and source artifacts are unchanged.

## Apply and run from project root

Preserve/commit preceding work before creating the new branch.

    git switch -c feat/pca-three-model-representation-v1
    git apply --check pca_three_model_v1.patch
    git apply pca_three_model_v1.patch
    python -m pytest -q tests/test_pca_three_model_v1.py
    python -m scripts.run_pca_three_model_v1 --dry-run
    python -m scripts.run_pca_three_model_v1
    python clil.py results --run outputs/results/pca_three_model_v1_run_01

Place the delivered patch at the root or substitute its real location.
Use the established project environment with xgboost/lightgbm installed. Do not upgrade
packages casually: historical verification requires the original environment and code.
No modification to clil.py is needed. The dry-run constructs unfitted classifiers but
performs no fit and creates no output. Existing output directories are rejected.

## Required original artifacts

configs/validation_protocol_v1.json and the original implementation in
src/extensions/validation_protocol_v1.py must exist. The source run indicated by that
configuration (normally outputs/results/validation_protocol_v1_run_01) must be complete.
All earlier frozen registry/Trial14 sources that the original validation _prepare method
requires must remain available. This patch reuses that verifier rather than duplicating
or weakening its provenance checks.

Preflight invokes the original validation _prepare: it verifies historical source artifacts,
original code/data/config/environment, frozen XGBoost, Trial14 and L1 settings. It then
verifies the completed validation run hashes, compares all 50 ordered cv5 train/valid
assignments, checks prediction row IDs and labels against current labelled rows, and
reproduces original component AUCs and the three-model mean 0.9083364180103309 (atol1e-12).
No competition test CSV is loaded. The original baseline scores are reused only after
those checks; the original models are not unnecessarily retrained in this experiment.

## Frozen comparison

Common k = 5, 10, 20, 50. For each saved split and k:

    Training-only StandardScaler -> PCA(full SVD, whiten=False)
                                    |-> original L1 classifier parameters
                                    |-> original XGBoost Trial20 classifier parameters
                                    |-> original LightGBM Trial14 classifier parameters
                                    -> equal 1/3 probability average

One deterministic scaler/PCA representation is fitted per split/k and shared by the three
classifiers. This is equivalent to independently fitting the same deterministic transform
three times on identical training rows, without redundant work. Each classifier is fitted
afresh. PCA scores are NOT standardized a second time before L1.

L1 classifier parameters are cloned from the original PipelineFactory; only its input
representation changes. XGBoost directly trains on all k PCA scores; there is NO ranking
model or top-20 original-feature selection in this branch. Original baseline XGBoost
remains the historical top-20 procedure. Thus its comparison is between representation
PROCEDURES, not an isolated PCA-only ablation. LightGBM uses its original parameters on
PCA scores. No estimator retuning, whitening, early stopping or weight search is done.

50 splits x 4 representations = 200 PCA/scaler fits and 600 classifier fits.
Four original reference procedures plus four PCA procedures per k = 20 scored procedures.
Standalone component performance does not select k; choose the highest ensemble mean
among the four common k values. Exact ties choose smaller k. No fallback selection.
Positive mean delta >1e-12 and >=6/10 repeat wins is a descriptive screening flag only.
Automatic promotion is always false; no refit-on-all-data or submission command is added.

Mean: average fold AUC within repeat, then average over ten repeats.
Repeat SD: sample SD of those ten repeat means; not fold SD or a confidence interval.
Compare each PCA component with its respective ORIGINAL reference and each ensemble with
the ORIGINAL ensemble. Never pool held-out predictions into a different AUC definition.
The XGBoost representation change must be explicitly disclosed when reporting results.

This is fixed-parameter representation research on adaptively reused labelled data, not
nested CV or an independent evaluation. A failure does not rule out PCA with retuning.

## Outputs

fold_scores.csv: 1,000 rows (20 procedures x 50 folds)
repeat_scores.csv: 200 rows
model_summary.csv: 20 rows
comparison.csv: 16 rows (4 representations x 3 components + ensemble)
paired_repeat_differences.csv: 160 rows
validation_probabilities.csv: 10,000 rows (4 k x 10 repetitions x 250 observations)
source_reproduction.csv: 200 original component/ensemble fold checks
pca_diagnostics.json: training scaler means and per-fold PCA explained variance
settings.json, split_indices.json, warnings.json, fit.log, decision.json, run_manifest.json

ConvergenceWarning stops the experiment. Failed/partial runs must not be used for model
selection. Warnings are recorded; native training stdout/stderr goes to fit.log.
Existing outputs are never overwritten or resumed. For a genuinely new run use --output.

## Validation status of this delivery

Synthetic engine/guard tests include a complete 600-classifier-fit run using sklearn
logistic-regression SURROGATES for the three classifier slots. They test the loop, metrics,
training-only transform, preserved constructor parameters, and failure handling. They are
NOT a native XGBoost/LightGBM benchmark. A separate native smoke test is included and is
skipped if either optional library is absent. The author environment lacks both native
libraries; local native integration and full historical preflight are still required.
No real competition experiment or full repository test suite was run by the patch author.
