# RF stabilizer V1: predeclared bounded development experiment

Status: implementation prepared; real-data execution NOT performed by this patch.
Base repository inspected: 7ddb2de9c8352bd7fc40d0bfb06a865bda082623.
Additive patch: no historical source/config/results changed; no clil.py integration.

## Question
Can a fixed 10% RF contribution improve the selected Trial14 three-model procedure?
"Stabilizer" is a hypothesis, not an established property of adding RF.

## Fixed procedure
- Original 250 labelled rows, 150 predictors, existing verified ten repeats/five folds.
- Baseline: (L1 + XGBoost + selected Trial14 LightGBM)/3.
- Candidate: 0.9*baseline + 0.1*RF. This is NOT four-model equal weighting.
- Four RF configurations in this exact tie-break order: (max_depth,min_samples_leaf)
  = (3,5),(3,10),(5,5),(5,10).
- RF: 500 trees, max_features="sqrt", bootstrap=True, random_state=42, n_jobs=1,
  class_weight=None; remaining sklearn defaults frozen by source environment versions.
- No scaling/selection, seed tuning, weight search, early stopping, Optuna, calibration,
  OOB-based selection or additional candidate families. 200 new forest fits maximum;
  each forest has 500 trees. Never describe this as only 200 individual tree fits.
- Fit RF on each original training fold; predict only its held-out validation rows.
- Reuse baseline OOF only after historical study/artifact/code/environment checks,
  exact ID/target/split matching, probability validity, component-cache checks,
  equal-weight identity and fold/aggregate AUC reproduction at atol=1e-12.

## Selection and descriptive stability
Primary: mean fold AUC per repeat, then mean across ten repeats. Compare with paired
repeat differences. Select highest mean blend; exact ties retain configuration order.
Only that winner may pass the pragmatic AUC gate: positive mean difference >1e-12
and >=6/10 repeat wins. No fallback to another RF configuration or RF alone.

Report SD, minimum/maximum repeat AUC, paired difference SD/range, wins/losses/ties,
probability spread, unique probability count and baseline/RF Spearman correlation.
"stability_observed" means BOTH SD no higher AND minimum repeat AUC no lower than
baseline (1e-12 tolerance). It is descriptive, not a statistical guarantee, and is
separate from the AUC gate. AUC promotion alone does not establish stabilization.
The same reused splits make this adaptive development, not nested/independent CV.
Never use competition public scores to choose the configuration or RF weight.

## Preconditions and safety
Use the original repository/environment and intact local outputs. The completed
LightGBM Optuna study and original fixed-LightGBM/soft-vote/frozen artifacts must
exist. Absolute historical paths and package versions are intentionally checked.
Do NOT edit old hashes, manifests, settings or configs to make a failed check pass.
A different environment/moved outputs requires a separate provenance migration or
new reference study, not tolerance relaxation. Optuna/XGBoost/LightGBM are needed
by the existing provenance preflight even though this new experiment only fits RF.
No new packages beyond the existing study environment and pytest are introduced.

## Apply and run from repository root
Save the attached patch in the repository root, then:

```bash
git apply --check rf_stabilizer_v1.patch
git apply rf_stabilizer_v1.patch
python -m pytest -q tests/test_rf_stabilizer.py
python -m scripts.run_rf_stabilizer --dry-run
python -m scripts.run_rf_stabilizer
```

Dry-run verifies sources without creating output, fitting RF or opening competition
test/template files. Default output: outputs/results/rf_stabilizer_v1_run_01.
Existing output is refused, even after failed/interrupted execution. Preserve it and
use a new --output name; there is no resume. No predict/export/upload command exists.

## Artifacts and failure policy
settings.json, fold_scores.csv (long: candidate/repeat/split_seed/fold/auc),
repeat_scores.csv, comparison.csv, decision.json, validation_probabilities.csv
(configuration/repeat/split_seed/fold/row_index/id/target/baseline/rf/blend),
diagnostics.json, warnings.json, fit.log, run_manifest.json; error.json on failure.
Manifest includes actual attempted/completed forest counts and final artifact hashes.
Any fold failure or captured Python warning stops the entire experiment; partial
results are diagnostics, not accepted scored candidates. Do not average successful
folds only. Review native warnings in fit.log as well; native output is not guaranteed
to appear in Python warning capture.

Retain the 0.91993 submitted candidate regardless of whether this study passes.
Positive results require manual review and a separately approved final-refit/export
protocol. This patch does not create a competition submission or modify experiment_list.md.

## Test coverage and limits
Synthetic tests cover exact weights, config lock, source OOF corruption, checksum
failure/path escape, aggregation, exact ties, no-fallback selection, incomplete folds,
failure/warning manifests, input mutation, no-write dry-run and mocked 200-fit
end-to-end lifecycle. A small real sklearn forest smoke test covers its API.
These tests do NOT run the real labelled dataset or audit the native dependency
stack. Existing historical-runner preflight remains required on the user's machine.
