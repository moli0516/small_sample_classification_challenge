# Individual extension: Validation protocol design (V1)

Status: implementation prepared; real labelled-data results NOT yet produced.
Use a dedicated branch from the final-model development baseline. No RF dependency.
This patch only ADDS files and does not change clil.py, historical runners, frozen
snapshots, old configurations, submissions or src/extensions/validation_protocol_study.py.

## Question
How sensitive are validation estimates and model rankings to split assignment when
only250 labelled rows are available? Which validation scheme would we trust for an
approximately0.01 AUC difference, and for our actual final-model decision?

Primary pair: fixed equal-weight L1+XGB versus fixed equal-weight L1+XGB+Trial14 LGBM.
Their previous development means differed by about0.005458, NOT0.01.
Auxiliary pair: original all-feature L1 C0.1 versus C1, previous mean gap about0.009248.
Do not invent an oracle model ordering or a new synthetic0.01 shift in real results.
The four scored procedures share components inside each split, not across splits.

## Protocol
- Original250 labelled rows/150 predictors, unchanged ID/feature order and labels.
- Stratified80/20 holdout; shuffled StratifiedKFold k={3,5,10}.
- Each scheme repeated ten times using split seeds0..9. Model seeds fixed42.
- Same ordered train/validation indices for every candidate within a split.
- Fixed original pipeline/frozen hyperparameters and original feature-selection rule.
- Fit fresh StandardScaler+L1 C0.1 and fresh StandardScaler+L1 C1 on training only.
- Fit fresh XGB ranker on training only, select top20 by total_gain and refit a fresh
  XGB classifier on those columns. Never import the full-data selected20 columns.
- Fit all-feature LGBM using stored selected Trial14 parameters on training only.
- Predict positive-class probabilities on the held-out rows; equal weights2/3 models.
- Existing all250-row fitted submission models are NEVER used to predict held-out
  training rows. Their selected PROCEDURE, not fitted state, is frozen.
- No Optuna, feature/seed/weight tuning, calibration, RF, competition test or upload.

Budget:10*(1+3+5+10)=190 split evaluations. Each uses five estimator fits: sparse L1,
dense L1, XGB ranker, XGB selected classifier, LGBM. Total950 fits. Two ensembles
reuse the same split-local components. All190 splits are refitted, including cv5.
Counter attempted_estimator_fits reserves two units before the compound XGB .fit;
if ranker fails early it is an upper-bound attempted count, not exact failed call count.
Completed fits are counted only after the whole branch completes without warnings.

## Source verification and historical comparison
Read the completed original lgbm_optuna_v1_run_01; require promoted Trial14 and its
reported0.9083364180103309 objective, manifest, resolved configuration and hashes.
Reuse existing LGBMOptunaExperiment._prepare only for original provenance/context
verification, not search or prediction. Require original package/code/input equality.
Inspect sparse/dense L1 configs: differ only in C and administrative CV/output fields;
require model_seed42 and no selector. Require frozen XGB/LGBM seeds42.
Original native packages, Optuna and intact absolute artifact paths are required.
Do NOT alter old files/hashes, suppress warnings, or rewrite paths to force a pass.
Some historical upstream APIs may issue pandas deprecation warnings; new extension
code introduces no verify_integrity keyword and does not edit hashed old APIs.

Generate splits once and save split_indices.json before training. For cv5, compare
ordered train AND validation indices with the saved original study. Only exact
matches require historical fold AUC reproduction for all shared branches/ensembles
at atol1e-12. Nonmatching assignments are marked not comparable, not silently reused.
Historical reproduction is a check, not new independent validation.

## Metrics
One repeat-level estimate = holdout AUC, or UNWEIGHTED mean of its k fold AUCs.
Do not pool OOF probabilities into a different AUC definition. Report10 repeat values
for each scheme/procedure, mean, sample SD(ddof1), minimum, maximum and median.
Record train/validation sample sizes (3-fold sizes differ by one row).

For each pair, compute within-repeat B-minus-A differences, then mean/sample SD/range,
B wins/A wins/ties at1e-12 tolerance and whether either direction wins across splits.
The b_win_fraction is empirical frequency on these splits, NOT probability B is best.
The difference_SD/.01 column is descriptive scale comparison, NOT a test statistic.
Do not count folds as independent repeated experiments or report repeat SD as a CI.
No automatic choice of a validation scheme based on maximum AUC or minimum SD.

## Commands from repository root
Required: original project environment including XGBoost/LightGBM/Optuna, matplotlib
and pytest. Keep historical package versions; do not indiscriminately upgrade.

```bash
git apply --check validation_protocol_v1.patch
git apply validation_protocol_v1.patch
python -m pytest -q -W error tests/test_validation_protocol_v1.py
python -m scripts.run_validation_protocol_v1 --dry-run
python -m scripts.run_validation_protocol_v1
```

The default destination is outputs/results/validation_protocol_v1_run_01.
Use --output with a NEW directory for a separate rerun. Existing destinations are
refused, even interrupted ones; no resume/overwrite. Dry-run opens no competition
test/template and creates no output or model fits. It still performs native source
and environment checks and split generation. Do not execute RF scripts first.
Progress prints after each split with scheme/repeat/fold/completed estimator fits.

## Outputs
- settings.json: fully resolved procedure, config, versions and source hashes.
- split_indices.json: all190 ordered train/validation assignments.
- fold_scores.csv:760 rows, four scored procedures per split, train/valid sizes.
- repeat_scores.csv:160 rows, ten repeats/four schemes/four procedures.
- model_summary.csv:16 rows, mean/SD/range and sample-size diagnostics.
- paired_repeat_differences.csv:80 rows, two pairs/four schemes/ten repeats.
- paired_summary.csv:8 rows, paired differences/spread/wins/ranking reversals.
- validation_probabilities.csv:8,000 validation rows; component and ensemble scores.
  Holdout contributes500 rows; each CV scheme contributes2,500.
- cv5_reproduction.csv:exact-match status and historical/current fold comparisons.
- auc_spread.png and paired_differences.png:boxplots with all10 repeat points.
- diagnostics.json:feature counts and fold-local XGB selections/round diagnostics.
- warnings.json, fit.log, run_manifest.json; error.json and partial tables on failure.
- interpretation.json:analysis reminders, no automatic protocol/model selection.

Any Python fit/prediction warning or invalid split/control stops the entire study;
partial scores are diagnostic only. Review native messages in fit.log separately.
Checkpoint CSVs are written per completed repeat. Source SHA256 is verified at
start/end, with size+mtime checked per split; outputs receive final artifact hashes.
Completion of an estimator followed by a warning is not accepted as a valid fit.

## Two-page report template (include author name)
### Question
State both the approximately0.01 auxiliary comparison and actual ensemble comparison.
### Protocol
Fixed procedures, split/random seeds, stratification, training-only transformations,
repeat score definition, budgets and descriptive metrics.
### Results
Use paired_summary.csv/model_summary.csv and selected figures; show spreads, not
only means. Do not paste all fold rows. Fill only after actual execution.
### Conclusion
Choose/defend a protocol using evidence, ranking consistency and practical cost.
It is acceptable to conclude that0.01 remains difficult to resolve. Discuss that
k changes train size; stratification controls class proportions; repeat datasets
overlap; candidates were selected adaptively earlier. No independent generalisation
truth is available from this study or the small public leaderboard.

## Test scope
Synthetic tests validate design counts, exact split coverage/disjointness, seeds,
repeat aggregation, pairing, ties/reversals, failure checks, plotting, full mocked
190-split lifecycle, identical-cv5 reproduction failure and artifact checksums.
A real sklearn pipeline smoke test checks training-only scaler fitting. No real
project data, original native preflight or full native ensemble training was executed
during patch generation. Local pytest and --dry-run remain required.
