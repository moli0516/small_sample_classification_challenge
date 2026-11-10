# Experiment List

## Scope, evidence and current checkpoint

Last updated: 2026-10-24, approximately 03:53 HKT.

This is the current development research record and roadmap. It distinguishes
completed numerical studies, engineering changes, competition feedback and
unexecuted proposals. It is not a claim of independent final evaluation.

Evidence used for this update:

- Existing repository experiment_list.md (historical Git blob
  `227ec043a8f72a5127443745e59f6b98c1f1b81f`). Earlier detailed tables and diagnostic
  narratives remain recoverable from Git history; this update consolidates them.
- Repository configurations, registry/Optuna implementations and runbooks inspected
  through GitHub. Historical settings are not silently rewritten.
- Owner-supplied soft-voting summary, fixed-LightGBM comparison, completed LightGBM
  Optuna manifest/decision/best-trial/trial table, and subsequent submission reports.
- Actual local artifacts/checksums were not independently recomputed during this
  documentation update. Public scores and current rank are owner-reported.

Current checkpoint:

- 250 labelled rows; 19,750 unlabelled competition test rows; 150 numeric predictors.
- `id` is excluded from predictors; `target` contains 0/1 labels.
- Current retained submitted candidate: equal-weight L1 + top-20 XGBoost + tuned
  LightGBM, using LightGBM Optuna Trial 14.
- Selected development CV AUC: 0.9083364180103309.
- Difference vs existing L1+XGBoost development reference: +0.00545799611017.
- Paired repeat wins/losses/ties vs that reference: 8/2/0.
- Latest reported public AUC: 0.91993; owner-confirmed public rank: fifth.
- Retained earlier submitted candidates: XGBoost top20 (0.91132 public), and fixed
  equal-weight L1+XGBoost (0.91514 public).
- No private score, nested-CV result or untouched final evaluation is recorded.
- Early stopping and Optuna pruning were not used in the completed searches.
- Early stopping is a possible next experiment, not an implemented or completed result.
- Current stage is a valid stopping/checkpoint point; no extra experiment is required
  merely because another technique remains available.

## Shared experimental rules

1. Preserve labelled data, feature/row order and verified saved split assignments.
2. Unless studying validation explicitly, use the established shuffled stratified
   five-fold splits repeated with seeds 0-9 (50 folds).
3. Fit all learned scaling and supervised selection only on the current training
   portion. Later XGBoost top-k selection is fold-local; earlier all-feature XGBoost
   experiments did not use explicit selection or scaling.
4. Keep model randomness fixed separately from split randomness; do not tune seeds.
5. Compute mean fold AUC per repeat, then mean/sample SD across the ten repeats.
6. Compare methods with paired repeat-level differences on identical splits. Do not
   treat overlapping repeats as independent datasets or repeat SD as a confidence interval.
7. Save configurations, versions, input/code/artifact hashes, split indices, scores,
   predictions, diagnostics and warnings. Local manifests are the detailed authority.
8. Define questions, search budgets, validity checks, objective and stopping rules
   before new execution. Mark adaptive follow-ups honestly.
9. Development scores used to select procedures are not independent final estimates.
   Public leaderboard feedback is also not a private/final generalisation estimate.
10. Use class-1 probabilities for the current probability-submission workflow; align
    component predictions and submissions by verified IDs/row/fold keys.
11. Do not submit every setting or choose parameters/weights using public feedback.
12. Do not overwrite historical outputs, candidate snapshots or submission files.
13. Preserve failures/warnings; do not average only successful folds after failures.
14. Changes to features, hyperparameters, weights, early stopping, pruning or validation
    require an explicit new protocol/version. Never bypass hashes or relax control tolerances.
15. Keep course code/data/artifacts private according to assignment rules; this record
    does not grant permission to redistribute them.

## Overview and stable document IDs

Historical E0-E7 numbering is retained. ENS-01, REG-01 and LGBM-OPT-01 below are
additional documentation labels, not renamed directories or new historical claims.


| ID               | Investigation                                           | Current status                                                                                   | Retained decision                                |
| ---------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------ |
| E0               | All-feature L2 logistic-regression C sweep              | Completed                                                                                        | Initial C=0.001 reference                        |
| E1               | Baseline refit/predict/submission workflow              | Owner-reported completed                                                                         | Recover original submission metadata             |
| E2               | Correlation-based feature selection                     | Completed                                                                                        | k=10 provisional leader; k=20 follow-up          |
| E3               | Selected feature count / C interaction                  | Completed                                                                                        | k=10, C=0.1 then led                             |
| E4               | L1 regularisation                                       | Four settings completed                                                                          | C=1 linear reference; C=0.1 sparse alternative   |
| E5               | Bounded RBF SVM comparison                              | Completed; warning review recorded                                                               | Not promoted                                     |
| E5b              | Six-setting shallow XGBoost grid                        | Completed                                                                                        | Did not replace L1                               |
| E5b extension 01 | Depth4 / 1000-round XGBoost                             | Completed                                                                                        | Not promoted                                     |
| E5b extension 02 | 30-trial XGBoost Optuna search                          | Completed                                                                                        | Trial20 selected at that stage                   |
| E5b extension 03 | Fold-local XGBoost total_gain top-k selection           | Completed                                                                                        | Top20 retained; top10 alternative                |
| ENS-01           | Fixed 50/50 L1 + top20 XGBoost                          | Completed; submitted                                                                             | Public 0.91514; retained reference               |
| REG-01           | Fixed-LightGBM single/ensemble comparison               | Completed                                                                                        | No new candidate promoted                        |
| LGBM-OPT-01      | 20-trial LightGBM tuning for fixed three-model ensemble | Completed; Trial14 submitted                                                                     | Current candidate; public 0.91993                |
| E6               | Validation-protocol individual extension                | Proposed; not confirmed executed                                                                 | Separate research question if chosen             |
| E7               | Candidate freezing, refit, export and evaluation        | Submission workflow completed for retained candidates; final independent assessment not recorded | Preserve snapshots and actual submission records |
| ES-01            | Early-stopping comparison                               | Idea/protocol discussion only                                                                    | Not started; no numerical claim                  |

## E0 — All-feature L2 logistic-regression C sweep

Question: how does C affect the fixed baseline?
Pipeline: StandardScaler -> LogisticRegression, all 150 predictors, lbfgs,
max_iter=2000. Sequential exploratory development study, not a retrospectively
preregistered grid.


|      C | Mean CV AUC | Repeat SD |
| -----: | ----------: | --------: |
|    1.0 |    0.751411 |  0.019922 |
|    0.1 |    0.771935 |  0.013106 |
|   0.01 |    0.792885 |  0.011157 |
|  0.001 |    0.793999 |  0.009091 |
| 0.0001 |    0.790600 |  0.008026 |

Historical decision: retain C=0.001; C=0.01 close alternative. Sweep closed.

## E1 — Baseline submission workflow

Owner-reported complete: all-feature scaled L2 C=0.001 refitted on all 250 rows,
class-1 probabilities generated for 19,750 test rows, template-ID alignment,
submission/model/configuration saved and baseline upload completed.
Development AUC 0.793999; repeat SD 0.009091.
Original public score, exact submission ID/time remain unrecorded in available evidence.
Do not label later XGBoost submission as the project's first-ever submission.

## E2 — Correlation-based feature selection

Pipeline: fold-local absolute Pearson-correlation ranking -> scaling -> L2 logistic
regression, C=0.001. Preserve original column order after choosing k.


|   k | Mean CV AUC | Repeat SD | Difference vs k=150 | Wins / losses |
| --: | ----------: | --------: | ------------------: | ------------- |
|   5 |    0.795349 |  0.006414 |           +0.001350 | 6 / 4         |
|  10 |    0.808618 |  0.008244 |           +0.014619 | 10 / 0        |
|  20 |    0.798767 |  0.008428 |           +0.004769 | 6 / 4         |
|  50 |    0.797398 |  0.008861 |           +0.003400 | 4 / 6         |
| 150 |    0.793999 |  0.009091 |                   0 | Control       |

Historical decision: retain k=10 and k=20 for E3; close k sweep.
Selection frequencies are descriptive, not bootstrap results or a fixed validated selector.

## E3 — Feature-count / C interaction

Same fold-local correlation/scaling pipeline, k={10,20}, C={0.001,0.01,0.1}.
Two E2 settings reused; four new settings added 200 fits.


|  k |     C | Mean CV AUC | Repeat SD |
| -: | ----: | ----------: | --------: |
| 10 | 0.001 |    0.808618 |  0.008244 |
| 10 |  0.01 |    0.812664 |  0.007200 |
| 10 |   0.1 |    0.822982 |  0.008456 |
| 20 | 0.001 |    0.798767 |  0.008428 |
| 20 |  0.01 |    0.799816 |  0.010557 |
| 20 |   0.1 |    0.794235 |  0.013509 |

Historical leader: e3_correlation_k_10_c_0_1. Versus E2 k10/C0.001:
+0.014364 and 10/0 repeat wins/losses. C effect depended on k; no global optimum established.

## E4 — L1 regularisation

All-feature StandardScaler -> L1 logistic regression; no correlation selector.
Four settings/200 fits. Intended configuration: saga, model_seed42, tol1e-4,
max_iter10000; saved configuration/factory remain the precise provenance.


|    C | Mean CV AUC | Repeat SD | Mean effective nonzero coefficients |
| ---: | ----------: | --------: | ----------------------------------: |
| 0.01 |    0.500000 |  0.000000 |                                0.00 |
|  0.1 |    0.852864 |  0.010803 |                               15.76 |
|  1.0 |    0.862112 |  0.012204 |                               74.96 |
| 10.0 |    0.817571 |  0.015909 |                               97.06 |

C1 versus C0.1: +0.009248, 8/2 wins/losses. Both beat E3 and the all-feature L2
reference in all ten repeats. C0.01 produced all-zero feature coefficients in all
50 folds; retain this negative/degenerate result. No warning or iteration-limit
folds were reported in the supplied E4 diagnostics.

Retain e4_l1_c_1 as the linear reference and e4_l1_c_0_1 as sparse alternative.
Do not turn global coefficient frequencies into a selector and reuse historical CV.
Optional Elastic-net was not run.

## E5 — Bounded RBF SVM comparison

All-feature StandardScaler -> SVC(rbf), saved splits unchanged; six settings/300 fits.
AUC used decision_function, not calibrated probabilities. probability=False,
max_iter10000, tol1e-3; exact execution settings in historical artifacts.


|   C | Gamma | Mean CV AUC | Repeat SD |
| --: | ----- | ----------: | --------: |
| 0.1 | scale |    0.807268 |  0.011407 |
|   1 | scale |    0.807959 |  0.012793 |
|  10 | scale |    0.803024 |  0.013892 |
| 0.1 | 0.01  |    0.811179 |  0.012608 |
|   1 | 0.01  |    0.812318 |  0.013557 |
|  10 | 0.01  |    0.812385 |  0.013559 |

All six lost to both retained L1 references in all ten repeats. Best E5 mean was
0.812385, below L1 C1 by 0.049726. No convergence/iteration-limit failures reported.
300 SVC probability-parameter FutureWarnings were reviewed and retained, not erased.
Historical standalone implementation and later source reorganisation are not separate
numerical experiments. Close bounded SVM comparison; do not claim all nonlinear methods fail.

## E5b — Original shallow XGBoost grid

All raw predictors, no scaling/selector. Fixed learning_rate0.05, min_child_weight5,
reg_lambda10, full row/column sampling, seed42, hist, n_jobs1. Six settings/300 fits;
no early stopping, pruning or eval_set.


| Depth | Trees | Mean CV AUC | Repeat SD |
| ----: | ----: | ----------: | --------: |
|     1 |    50 |    0.787117 |  0.014404 |
|     1 |   100 |    0.804358 |  0.015245 |
|     1 |   200 |    0.808197 |  0.017836 |
|     2 |    50 |    0.833019 |  0.014400 |
|     2 |   100 |    0.839271 |  0.010432 |
|     2 |   200 |    0.833426 |  0.014514 |

All 300 fits valid, no captured Python warnings reported. Best depth2/100 still
lost to L1 C1 by mean 0.022840 (1/9 repeat wins/losses). Original study closed.
A prior single-fold smoke AUC of 0.910628 was an environment diagnostic, NOT CV mean
or leaderboard score. macOS libomp/path-resolution fixes are engineering events.

### E5b extension 01 — Depth4 / 1000 rounds

One adaptive follow-up, 50 fits, original remaining parameters unchanged.
Mean AUC0.828091, repeat SD0.018197; -0.034020 versus L1 C1, 0/10 wins/losses.
No captured warnings or failed folds reported. Not promoted.
Depth and rounds changed jointly; this does not isolate a causal parameter effect.
Output: outputs/results/e5b_xgb_ext01_depth4_trees1000/.

### E5b extension 02 — XGBoost Optuna

30-trial bounded adaptive search; config configs/e5b_optuna_ext02.json;
output outputs/results/e5b_optuna_ext02_run_01/; sampler seed42, startup trials10.
Search: depth1..4, rounds{50,100,200,500,1000}, learning rate 0.01..0.2 log,
min_child_weight1..20 log, reg_lambda.1..100 log, subsample.6..1,
colsample_bytree.5..1. No early stopping/pruning.

Historical selected candidate: Trial20,
e5b_optuna_ext02_trial_0020_depth_4_trees_50.
Mean development AUC0.881937; repeat SD0.013949.
Versus L1 C1: +0.019826 and 10/0/0 repeat wins/losses/ties.
Versus L1 C0.1: +0.029074 and 10/0/0.
Full-precision constructor settings must come from stored source parameters; the
rounded historical summary is not a substitute for them.

### E5b extension 03 — Fold-local total_gain feature selection

Config: configs/e5b_feature_selection_ext03.json.
Source: Trial20 from e5b_optuna_ext02_run_01. Reuse its exact parameters.
For every training fold, fit the full-feature control/ranker, select total_gain
k={5,10,20}, then refit a fresh classifier on selected columns. Zero-fill unused
gains, tie-break by original predictor order, retain original order after selection.
All-feature control plus three subsets: 200 fits; 50 historical control checks.


| Procedure                   |       Mean development CV AUC |                     Repeat SD |
| --------------------------- | ----------------------------: | ----------------------------: |
| All-feature Trial20 control | 0.881937 (historical rounded) | 0.013949 (historical rounded) |
| Fold-local top10            |   0.888786 (reported rounded) |   Not supplied in this update |
| Fold-local top20            |            0.8906718321935714 |          0.009546924598731034 |

Top20 versus top10: reported six repeat wins/four losses; mean gap about0.001886.
Top20 retained as the then-leading procedure; top10 remains a compact alternative.
Top5 result is not reproduced here because its numerical row was not supplied;
consult the original feature_selection_comparison.csv, rather than inventing a value.

Output: outputs/results/e5b_feature_selection_ext03_run_01/.
Frozen procedure used in subsequent completed workflows: outputs/frozen/ext03_top20_v1/.
Final full-data selection must rerank on all labelled training rows, not use global
cross-fold selection frequencies. First reported submission of this candidate: public0.91132.

## ENS-01 — Fixed L1 + XGBoost soft voting

Question: do the existing linear and tree procedures complement each other?
Fixed branches: original all-feature L1 C1 pipeline; frozen Trial20/top20 XGBoost.
Fixed weights: .5/.5. No new parameter/weight search.
Same 50 folds; 150 fits when rerun (L1, XGB ranker and subset classifier per fold).


| Candidate               |        Mean CV AUC |            Repeat SD |
| ----------------------- | -----------------: | -------------------: |
| L1                      |  0.862111696676914 | 0.012203784380591066 |
| Top20 XGBoost           | 0.8906718321935714 | 0.009546924598731034 |
| Equal-weight L1+XGBoost | 0.9028784219001611 |  0.00883941169930424 |

Mean paired difference versus XGB: +0.01220658970658971; wins/losses/ties9/1/0.
Passed predeclared pragmatic promotion rule (positive mean and >=6/10 wins).
This is not a significance test. Full-data refit/export/submission subsequently reported.
Public AUC0.91514, improving on the top20 XGB submission by0.00382.

Output: outputs/results/soft_vote_l1_xgb20_v1_run_01/.
Submission output: outputs/submissions/soft_vote_l1_xgb20_v1_run_01/.
Retain this as the baseline for every subsequent ensemble comparison.

## REG-01 — Fixed LightGBM and seven-candidate ensemble comparison

Purpose: test the three single models and four equal-weight combinations without
changing the existing L1/XGB controls. The registry integration is engineering;
its numerical seven-candidate comparison is the experiment recorded here.

LightGBM: all150 predictors; objective binary, learning_rate.05, n_estimators100,
max_depth3, num_leaves7, min_child_samples20, reg_lambda10, subsample1,
colsample_bytree1, seed42, n_jobs1, deterministic=True, force_col_wise=True.
No early stopping or pruning. Existing L1/XGB OOF reused only after hash, version,
split, ID/label and probability/AUC checks; default additional budget50 fits.


| Candidate   | Mean CV AUC | Repeat SD | Difference vs L1+XGB | Wins / losses / ties |
| ----------- | ----------: | --------: | -------------------: | -------------------- |
| L1+XGB      |    0.902878 |  0.008839 |                    0 | 0/0/10               |
| L1+XGB+LGBM |    0.900614 |  0.009702 |            -0.002264 | 1/9/0                |
| XGB         |    0.890672 |  0.009547 |            -0.012207 | 1/9/0                |
| LGBM+XGB    |    0.886114 |  0.011460 |            -0.016765 | 0/10/0               |
| LGBM+L1     |    0.884225 |  0.011864 |            -0.018653 | 0/10/0               |
| L1          |    0.862112 |  0.012204 |            -0.040767 | 0/10/0               |
| LGBM        |    0.856224 |  0.015250 |            -0.046655 | 0/10/0               |

Rounded values above follow the owner-supplied comparison. No new blend promoted
or reported submitted from this fixed-setting experiment. Baseline promotable=False
means it is the reference, not that it failed a test against itself.
Conclusion limited to THIS LightGBM configuration and equal weights, not the entire
LightGBM family. This motivated the separately bounded tuning study below.

Config: configs/registry_ensembles.json.
Output: outputs/results/registry_ensembles_v1_run_01/.
Key artifacts: comparison.csv, decision.json, fold_scores.csv, repeat_scores.csv,
validation_probabilities.csv, diagnostics.json, fit.log, warnings.json, run_manifest.json.

## LGBM-OPT-01 — Bounded LightGBM Optuna for the fixed three-model ensemble

### Question, objective and budget

Can tuning only the LightGBM branch improve (L1+XGB+LGBM)/3 over the retained
(L1+XGB)/2 baseline? This is not a search for the best standalone LightGBM.

Config: configs/lgbm_optuna.json; primary objective l1_xgb_lgbm.
Reference: completed REG-01. L1/XGB branches and their verified OOF predictions fixed.
20 total trials including control/failures; maximum1000 additional LightGBM fits.
TPESampler(seed42,n_startup_trials5), sequential, NopPruner; no early stopping,
weight search, test-label use or leaderboard-driven trial selection.

Search space:

- tree_shape (depth:leaves): 2:3,3:3,3:7,4:3,4:7,4:15.
- min_child_samples:5..30.
- learning_rate:0.02..0.15, log.
- n_estimators:{50,100,200}.
- reg_lambda:.1..30, log.
- colsample_bytree:.6..1.
- All other constructor settings retained from fixed reference.

Trial0 enqueued the original LightGBM parameters and reproduced all seven historical
candidate fold AUCs within1e-12. Subsequent trials retained fixed control checks.

### Execution status

Owner-supplied manifest:

- status=completed; planned20, completed20, failed0.
- control_trial_reproduced=true; test_opened=false; kaggle_upload_performed=false
  at search completion. Later manual upload is recorded separately below.
- started UTC2026-10-23T19:44:55.112134+00:00.
- finished UTC2026-10-23T19:45:51.047160+00:00.
- best_trial14; promote=true.

### Trial results

All20 trials COMPLETE. AUC is the fixed three-model primary objective. Win/loss/tie
counts compare each trial with the existing L1+XGB baseline across ten repeats.
Trial13 has one tie (10 minus7 wins minus2 losses).


| Trial | Primary objective AUC | Difference vs L1+XGB | Wins / losses / ties | Passes promotion rule |
| ----: | --------------------: | -------------------: | -------------------- | --------------------- |
|     0 |           0.900614321 |         -0.002264101 | 1 / 9 / 0            | No                    |
|     1 |           0.896479129 |         -0.006399293 | 0 / 10 / 0           | No                    |
|     2 |           0.901639427 |         -0.001238994 | 2 / 8 / 0            | No                    |
|     3 |           0.908013311 |         +0.005134889 | 9 / 1 / 0            | Yes                   |
|     4 |           0.898419861 |         -0.004458561 | 2 / 8 / 0            | No                    |
|     5 |           0.901967240 |         -0.000911182 | 4 / 6 / 0            | No                    |
|     6 |           0.905134419 |         +0.002255997 | 8 / 2 / 0            | Yes                   |
|     7 |           0.901576584 |         -0.001301838 | 3 / 7 / 0            | No                    |
|     8 |           0.904778583 |         +0.001900161 | 7 / 3 / 0            | Yes                   |
|     9 |           0.900836889 |         -0.002041533 | 3 / 7 / 0            | No                    |
|    10 |           0.902458435 |         -0.000419987 | 3 / 7 / 0            | No                    |
|    11 |           0.902160999 |         -0.000717423 | 5 / 5 / 0            | No                    |
|    12 |           0.907426020 |         +0.004547598 | 8 / 2 / 0            | Yes                   |
|    13 |           0.907297980 |         +0.004419558 | 7 / 2 / 1            | Yes                   |
|    14 |           0.908336418 |         +0.005457996 | 8 / 2 / 0            | Yes                   |
|    15 |           0.905975438 |         +0.003097016 | 8 / 2 / 0            | Yes                   |
|    16 |           0.901906749 |         -0.000971673 | 3 / 7 / 0            | No                    |
|    17 |           0.905915469 |         +0.003037047 | 8 / 2 / 0            | Yes                   |
|    18 |           0.902939697 |         +0.000061275 | 6 / 4 / 0            | Yes                   |
|    19 |           0.904322208 |         +0.001443786 | 7 / 3 / 0            | Yes                   |

### Selection decision

Select highest primary-objective AUC, earliest trial for exact ties. Only that winner
can be promoted if its mean paired difference is positive and it wins >=6/10 repeats.
Do not replace it with another trial based on more wins or a secondary ensemble score.

Winner: Trial14.

- Mean objective CV AUC:0.9083364180103309.
- Difference vs L1+XGB:+0.00545799611017.
- Wins/losses/ties:8/2/0.
- Trial3:0.9080133111654851,9/1/0; not selected because objective AUC was lower.
- Best trial repeat SD and standalone LightGBM AUC are not in supplied best_trial.json;
  obtain them from trials/trial_0014/cv/comparison.csv, not from the fixed-LightGBM table.

Selected LightGBM constructor parameters:

```json
{
  "objective": "binary",
  "learning_rate": 0.08060078627579852,
  "n_estimators": 50,
  "max_depth": 4,
  "num_leaves": 7,
  "min_child_samples": 6,
  "reg_lambda": 0.12763701705576885,
  "subsample": 1.0,
  "colsample_bytree": 0.771566198351983,
  "random_state": 42,
  "n_jobs": 1,
  "verbosity": 1,
  "deterministic": true,
  "force_col_wise": true
}
```

Study: outputs/results/lgbm_optuna_v1_run_01/.
Winner: trials/trial_0014/cv/ within that study.
Artifacts: study.sqlite3, settings.json, trials.csv, trial_results.json, best_trial.json,
decision.json, run_manifest.json, per-trial resolved registry configuration and full
CV tables/probabilities/diagnostics/logs/manifests. The supplied manifest contains hashes;
this document intentionally does not duplicate its long checksum list.

### Final refit and reported competition feedback

Selected composition remains (p_L1+p_XGB+p_LGBM)/3, not .5*p_old_ensemble+.5*p_LGBM.
Full-data workflow refits all250 labelled rows: L1 once, XGB ranker/classifier twice,
LightGBM once (four fits). Save/load agreement and CSV/template checks belong in
prediction_manifest.json. Owner subsequently reported completed submission.

Submission output: outputs/submissions/lgbm_optuna_v1_run_01/.
Reported public AUC:0.91993; owner-confirmed rank:fifth at that checkpoint.
Public increment vs L1+XGB:0.00479; increment vs first top20 XGB submission:0.00861.
Exact Kaggle submission ID/time not supplied. No private score recorded.

Conclusion: fixed-setting LightGBM failed to add value, but a bounded parameter
search found a three-model candidate with higher development CV and higher reported
public AUC. This does not isolate which parameter caused the improvement or prove
an independent/private performance gain.

## E6 — Validation-protocol extension

Status: proposed; not confirmed chosen or executed. Do not mark completed.
Possible question: how do hold-out and different fold counts alter model ranking
or apparent differences? A new protocol must specify candidates, repeats, seeds,
training sizes, aggregation and budget. Repeatedly changing splits is not a new
independent test set. Do not reopen completed searches automatically.

## E7 — Candidate lifecycle and evaluation

Reported milestones now supersede the old 'XGBoost submission not confirmed' status:

- Baseline end-to-end submission workflow completed earlier (score still unrecorded).
- Frozen top20 XGBoost procedure used in subsequent verified development workflows.
- Full-data top20 XGB submission reported:0.91132.
- Full-data 50/50 L1+XGB submission reported:0.91514.
- Selected Trial14 three-model submission reported:0.91993; rank fifth reported.

Preserve all candidates and their models, selected features, configurations, inputs,
versions, code provenance and CSV hashes; do not overwrite earlier snapshots.
Repository command support and successful reported workflows are not a claim that
all local metadata files have been independently audited here.

Final evaluation remains distinct: no nested-CV result or private-leaderboard result
has been supplied. Course-specific final deliverables/grading have not been verified.

### Submission ledger (owner-reported feedback)


| Submission/candidate                  | Development CV AUC |   Public AUC | Exact Kaggle ID/time | Status                               |
| ------------------------------------- | -----------------: | -----------: | -------------------- | ------------------------------------ |
| E1 L2 baseline C0.001                 |           0.793999 | Not recorded | Pending              | Baseline workflow reported complete  |
| Frozen Trial20/top20 XGB              | 0.8906718321935714 |      0.91132 | Pending              | Submitted                            |
| ENS-01 L1+XGB, equal weights          | 0.9028784219001611 |      0.91514 | Pending              | Submitted; retained reference        |
| LGBM-OPT-01 Trial14 three-model blend | 0.9083364180103309 |      0.91993 | Pending              | Current retained submitted candidate |

No submission from fixed REG-01 was reported. Public scores are not classification
accuracy; rank is time-specific, not a prediction of final/private placement.
At the time of inspection experiments/log.csv and experiments/submissions.csv had
headers only. Recover actual submission IDs/timestamps before marking those ledgers
complete; documentation here does not silently populate them.

## ES-01 — Early stopping: future experiment only

Status: discussed, not run; no new code execution or score claimed.
Question: with Trial14's other LightGBM parameters and equal weights fixed, does
training-only selection of boosting rounds improve the existing procedure?

Items to define BEFORE any execution:

- Fixed maximum rounds and patience; inner split seed/fraction or inner-CV scheme.
- Whether stopping monitors LightGBM-only AUC or ensemble AUC; these are different objectives.
- Inner training/validation drawn only from each outer training fold.
- Round-selection rule, then refit at selected rounds on the full outer training fold.
- Score the outer validation fold only after round selection, not use it to stop training.
- Final full-labelled-data round-selection/refit rule, diagnostics, budget and promotion rule.
- Fixed50-round Trial14 procedure as reference; new immutable output/version.

Do not add early stopping silently to the completed Optuna study, overwrite its
hashes, or call this unexecuted idea a proven improvement. The existing procedure
already searched coarse round counts; early stopping is a new selection rule.
Optuna pruning is distinct and remains unused; it is not required to close this stage.

## Other proposals not executed or selected for this checkpoint

- Optional Elastic-net.
- XGB-selected-feature transfer to L1.
- Shrinkage LDA.
- Learned stacking, arbitrary ensemble weight optimisation or new model families.

These are ideas, not completed experiments or mandatory next tasks.

## Engineering changes versus numerical experiments

- Registry/wrapper adapters standardise complete train/predict procedures and lazy
  model creation; they do not themselves establish score gains.
- Four ensemble compositions are configuration-driven probability combinations.
- Verified OOF reuse reduces new fits without changing fold membership.
- clil.py (intentional filename) forwards to existing runners and supplies doctor,
  tests, plan and results inspection. It does not automatically submit to Kaggle.
- Package build tests and mocked/synthetic adapters are not real-data model validation.
  Completion of the owner-reported numerical study is recorded separately above.
- Patches do not alter old results merely by being applied; preserve executed revisions.

## Commands for inspecting this checkpoint (not rerunning searches)

```bash
python clil.py results --run outputs/results/lgbm_optuna_v1_run_01
python clil.py results --run outputs/submissions/lgbm_optuna_v1_run_01
```

The raw original two-model and registry reports remain under their respective
outputs/results/ directories. Do not edit hashed configs/artifacts of completed runs.
Updating this research record is not permission to rerun, overwrite, retune or upload.

## Stage-close checklist

- [X]  Preserve historical E0-E5b decisions, including negative results.
- [X]  Record extension03, ENS-01, REG-01 and LGBM-OPT-01 outcomes.
- [X]  Separate development AUC from reported public scores.
- [X]  Record selected Trial14 and actual reported0.91993/fifth-place checkpoint.
- [X]  Mark early stopping, E6 and other unexecuted ideas accurately.
- [ ]  Recover exact Kaggle submission IDs and timestamps for the submission ledger.
- [ ]  Verify local final model/CSV/manifest archives and preserve a code revision.
- [ ]  Complete any required report/course deliverables against the actual rubric.
- [ ]  Record private/final evaluation only when it exists; never infer it from public AUC.

Current decision: retain Trial14's equal-weight three-model submission as the
best reported submitted candidate and preserve the0.91514 two-model fallback.
Stop at this checkpoint; any next experiment requires an explicit new plan.
