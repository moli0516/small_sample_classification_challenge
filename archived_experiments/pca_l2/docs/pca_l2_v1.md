# PCA–L2 development study v1

This additive patch does not modify clil.py, existing source files, frozen registries,
historical study outputs, ensemble weights, or submissions. Use the module entry point.

## Apply and check (from the project root)

    git apply --check pca_l2_v1.patch
    git apply pca_l2_v1.patch
    python -m pytest -q tests/test_pca_l2_v1.py
    python -m scripts.run_pca_l2_v1 --dry-run
    python -m scripts.run_pca_l2_v1
    python clil.py results --run outputs/results/pca_l2_v1_run_01

The patch file can be stored outside the project; substitute its actual location in git apply.
Existing numpy, pandas, scikit-learn, scipy, threadpoolctl and pytest are required.
No new model library is required. Run in the established project environment.

## Preconditions

The completed local outputs/results/validation_protocol_v1_run_01 directory must exist,
with its original run_manifest.json and artifact checksums. The study reuses its 50 saved
cv5 assignments, NOT newly generated splits. It verifies all source artifact checksums
and checks train.csv against the source study training hash. Only the labelled training
CSV is loaded. target and id are explicit config names; if your target column has a
different name, edit the config before running. Missing historical outputs fail closed.

Dry-run validates inputs and saved splits but performs zero fits and creates no output.
A nonexisting output directory is mandatory; no overwrite or resume. For another run:

    python -m scripts.run_pca_l2_v1 --output outputs/results/pca_l2_v1_run_02 --dry-run
    python -m scripts.run_pca_l2_v1 --output outputs/results/pca_l2_v1_run_02

## Design frozen before running

Control: StandardScaler -> L2 logistic regression.
PCA: StandardScaler -> PCA(full SVD, whiten=False) -> L2 logistic regression.
Both use lbfgs, max_iter=5000, tol=1e-4, random_state=42.
C values: 0.001, 0.01, 0.1, 1, 10 for BOTH branches.
PCA component counts: 5, 10, 20, 50. Thus 5 controls + 20 PCA candidates,
50 shared splits, 1,250 complete pipeline fits (classifier fits), including 1,000 PCA fits.
Each scaler and PCA is fitted strictly inside the training fold. Explained variance is
recorded for diagnosis, never used to select candidates or component counts.

Score: mean fold ROC AUC within each repeat, then mean across 10 repeats.
SD: sample SD across repeat means, NOT SD across 50 folds or a confidence interval.
Select highest mean in each branch; exact ties use configuration order (smaller C for
controls; smaller k then C for PCA). No fallback candidate selection.
Report each PCA candidate versus its same-C control AND selected PCA versus selected
control. A descriptive screening flag requires mean delta > 1e-12 and >=6/10 wins.
Tie tolerance: abs(delta)<=1e-12. No automatic promotion or submission is performed.

## Important interpretation

This is adaptive DEVELOPMENT tuning, not nested CV or independent evaluation.
Selecting the best of 20 PCA candidates versus 5 controls has unequal search opportunity;
report the full grid and same-C contrasts, not just a winning score. This experiment
answers whether a PCA/L2 procedure is promising, not whether to replace the incumbent
three-model ensemble. Any later ensemble addition needs its own controlled experiment.
The new matched L2 controls are not claimed to reproduce old historical L2 scores;
solver/iteration settings are explicitly common to both new branches.

## Artifacts

settings.json: resolved design, versions, source/input/code hashes.
split_indices.json: exact ordered saved cv5 splits.
fold_scores.csv: 1,250 rows on successful default run.
repeat_scores.csv: 250 candidate/repeat scores.
model_summary.csv: 25 candidate means and repeat SDs.
comparison.csv: 20 same-C comparisons and 1 selected-branch comparison.
paired_repeat_differences.csv: 210 rows.
validation_probabilities.csv: 62,500 held-out predictions (not competition test predictions).
pca_diagnostics.json: per-fold explained variance for PCA candidates.
warnings.json; decision.json; run_manifest.json with artifact hashes.
ConvergenceWarning stops the run. Failed/partial output must never be used for selection.

Tests include a complete synthetic 200-fit run and leakage/provenance/preflight guards.
They do not certify scores, performance, or compatibility with unshared local changes.
The real 250-row PCA experiment has NOT been run by the patch author.
