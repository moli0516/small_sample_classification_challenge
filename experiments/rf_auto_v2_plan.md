# RF Auto V2: bounded automatic max_features search

Status: code prepared; real-data execution pending. No result is claimed by this patch.
Requires the locally applied rf_stabilizer_v1.patch AND pandas compatibility patch,
and completed outputs/results/rf_stabilizer_v1_run_01. RF V1 files were not present
on the remote GitHub repository at preparation time; interfaces are taken from the
previously supplied patches and the user's successful V1 execution.

## Question and fixed protocol
Does changing RF max_features improve the V1 depth3/leaf5 blend?
This is AUTOMATIC EXHAUSTIVE SEARCH over three declared choices, NOT broad Optuna
or an expansion to depth/leaf/weight/seed tuning. Trials run in order:
1. sqrt: reuse the verified V1 depth3/leaf5 control, no new fits.
2. 0.2: fifty new forest fits.
3. 0.5: fifty new forest fits.

Keep 500 trees/forest, max_depth3, min_samples_leaf5, bootstrap=True,
random_state42, n_jobs1, class_weight=None, remaining version-pinned sklearn defaults.
Use all150 predictors and unchanged saved ten repeats/five folds; no scaling or selector.
Candidate = .9 * Trial14 three-model OOF + .1 * RF. Frozen L1/XGB/LGBM branches,
weights and data are unchanged. No competition test file/template reading, OOB-based
selection, calibration, early stopping, pruning, native model refit or upload.
Maximum100 NEW forest fits (50,000 new trees); reuse50 control folds. More features
can increase work per tree; this is not a guarantee of shorter wall-clock runtime.

## Source checks
Call the original RFStabilizer._prepare without changing historical files. It checks
Trial14 and its original provenance using the established local environment. Verify
all V1 manifest artifact hashes, completion counts, versions, input/code hashes,
original fixed settings and reported selected depth3/leaf5 control.
Check cached row/split IDs, targets, finite probability range, exact .9/.1 blend,
Trial14 baseline agreement, all selected control fold AUCs and mean AUC (atol1e-12).
No verify_integrity keyword is introduced; index.is_unique checks remain explicit.

The V1 study is already adaptive development. Reusing its selected configuration is
an explicit follow-up decision, not a claim of preregistration before V1.

## Selection and decision (declare before execution)
Primary objective: mean fold AUC per repeat, then mean over ten repeats, for the
BLENDED candidate. RF standalone scores are descriptive only.
Highest mean of the three blends wins; exact ties keep sqrt, then0.2, then0.5.
No fallback to a lower-scoring candidate after checking its wins/stability.

The winner is flagged promote_new_configuration ONLY if:
- it is a NEW setting (not sqrt),
- positive paired mean improvement and >=6/10 wins versus Trial14,
- paired mean improvement > minimum_mean_gain +1e-12 versus V1 blend AND >=6/10 wins versus V1.

Default minimum_mean_gain =0.0001 AUC, chosen BEFORE executing this follow-up.
It is a pragmatic small-effect screen, not a significance threshold. Do not edit it
after looking at scores; a changed gate requires another declared experiment version.
This gate is stricter than V1 intentionally: merely reproducing V1's approximately
+0.000031 gain versus Trial14 is not sufficient to promote a new RF configuration.

Report both baseline and V1 paired mean/difference SD/wins/losses/ties, repeat SD,
minimum/maximum repeat AUC and baseline-RF rank correlation. stability_observed
remains descriptive: SD no higher AND minimum repeat AUC no lower than Trial14.
It is separate from promotion. Reused splits are not independent final evaluation,
and an increased development score does not guarantee better public/private AUC.
No model replaces the submitted0.91993 candidate automatically.

## Installation and commands (repository root)
This patch only ADDS files. It does not modify V1 or any hashed historical source.
Keep the existing environment; no new dependency is required beyond the V1 setup.

```bash
git apply --check rf_auto_v2.patch
git apply rf_auto_v2.patch
python -m pytest -q -W error tests/test_rf_auto_search.py
python -m scripts.run_rf_auto --dry-run
python -m scripts.run_rf_auto
```

Optional fresh output directory, declared before execution:
```bash
python -m scripts.run_rf_auto --output outputs/results/rf_auto_v2_run_02
```
No --resume or overwrite. Do NOT rerun V1, regenerate its cache, change old sources,
suppress warnings, migrate absolute paths or alter artifact hashes to bypass a failed
preflight. Diagnose incompatibilities separately. Existing native historical dependency
imports remain required even though only RF is newly trained.

## Performance, progress and file safety
Print one live progress line per new forest fit, including setting/repeat/fold/time.
Record attempted/completed counts in run_manifest.json before/after each forest.
Fully hash sources at preflight/start/end. Between fits, check size/mtime before and
after training rather than rereading all artifact contents repeatedly. The final
full-hash check is required before any successful study decision.
Write accumulated OOF/score CSVs after each completed setting, not every forest;
write partial diagnostics on exceptions/interruption. Python warning/fold failure
stops the whole search. Never average only successful folds. Native messages may
need separate fit.log review; Python warning capture is not a native-log audit.

Artifacts: settings.json, fold_scores.csv (350 long-format rows), repeat_scores.csv,
comparison.csv (seven procedures), validation_probabilities.csv (7,500 rows),
diagnostics.json, warnings.json, fit.log, decision.json and final hashed run_manifest.json.
On failure: error.json, partial_fold_scores.csv, partial_validation_probabilities.csv.
A decision promote_new_configuration=true is for manual review only. There is no
prediction, export, Kaggle submission or full-data final training command.

## Verification scope
The supplied tests use synthetic data for reference checks, corruption guards,
paired aggregation, exact ties, practical-gain/no-fallback gates, warning failure,
progress output, checkpoint counts, and a mocked100-fit lifecycle. A small real
sklearn forest validates API use. Isolated generation tests reconstructed the V1
helper interfaces; they did not run the original native historical _prepare or your
real data. Run local pytest and --dry-run before trusting project integration.
The separate delivered validation report records the exact test environment/limits.
