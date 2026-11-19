# Geometry-first semi-supervised generative experiment v1

## Scope and permissions

This patch implements stages A and B, NOT stage C of the proposed research plan.
A: training-fold features and labels only.
B: training-fold features/labels plus validation FEATURES; validation labels are used
only for scoring, never fitting or component-to-class mapping. B is explicitly
transductive validation, NOT ordinary untouched inductive CV. It requires the CLI flag.

There is NO competition-test reader, training entry point, or submission export.
Unknown config fields (including test_path or purported permission fields) are rejected
before data loading. This is not a legal/competition eligibility certification. Teacher
requirements still apply to B. No claim is made that test-set transduction is allowed.

## Existing project integration

Continue your CURRENT branch. Only seven new files; no changes to old modules/hashes.
Core model: src/modeling/semi_supervised_mixture_v1.py
Geometry: src/features/generative_geometry_v1.py
Experiment: src/experiments/generative_geometry_v1.py
Thin CLI: scripts/run_generative_geometry_v1.py
Reuses the repository's real src.data.labelled_data_loader.LabelledDataLoader API.
No imports from historical scripts; no new dependencies beyond numpy/scipy/pandas/sklearn
and threadpoolctl. Use the existing environment; do not upgrade frozen packages.

Requires the completed outputs/results/elasticnet_optuna_v1_run_01 study selecting
Trial25, including all manifest-listed artifacts, exact original training CSV,
cohort.csv, saved splits, and trials/trial_0025/predictions.npz.
Preflight verifies all reference artifact hashes, exact binding of training CSV hash
from the reference settings, all50 split partitions, once-per-repeat coverage,
row IDs/indices/labels, and recomputed mean reference AUC. It hashes new source code,
config and loader. The complete historical source-code dependency/version verifier is
not rerun: training binding/artifact/score identity are the reference checks here.

## Apply from project root (no branch creation)

    git apply --check generative_geometry_v1.patch
    git apply generative_geometry_v1.patch
    python -m pytest -q tests/test_generative_geometry_v1.py

Use the actual downloaded patch location if it is not in the project root.

## Default safe A-only experiment

    python -m scripts.run_generative_geometry_v1 --dry-run
    python -m scripts.run_generative_geometry_v1

6 candidate structures x3 fixed seeds x50 folds =900 composite candidate fits.
50 projector fits. Each candidate fit includes ONE sklearn GMM initialization and ONE
custom EM optimization (900 of each); initialization calls are NOT additional candidate
settings. No seed is selected; predictions are averaged across all3 seeds.
Fresh output required, no overwrite/resume.

## Explicit A+B mechanism comparison (use this INSTEAD of A-only for a paired study)

    python -m scripts.run_generative_geometry_v1 --transductive-validation --dry-run
    python -m scripts.run_generative_geometry_v1 --transductive-validation

If an A-only output already exists, preserve it and pass, e.g.:

    python -m scripts.run_generative_geometry_v1 --transductive-validation --output outputs/results/generative_geometry_ab_v1_run_01 --dry-run
    python -m scripts.run_generative_geometry_v1 --transductive-validation --output outputs/results/generative_geometry_ab_v1_run_01

A+B uses12 candidates x3 seeds x50 folds =1800 composite fits;100 projector fits.
Actual runtime is not promised. Progress is printed per fold and detailed fit records
are flushed to fit_diagnostics.jsonl after every fit. An interrupted output is diagnostic
only; final selection is not performed. Retain it rather than resuming/overwriting.

## Representation and model

Inside each declared fit scope, fit StandardScaler. Rank nonconstant features by
sum of squared correlations plus0.05*absolute excess fourth moment. Retain up to20,
then use up to8 unwhitened PCA components. This is an explicit, unvalidated geometry
heuristic, not evidence of recovered informative features or a reconstructed generator.
Diagnostics preserve rankings, selected columns and explained variance per fold/mode.
A fits geometry on training only; B fits geometry on training+validation features.
The method changes both representation and mixture fitting between A/B, so B-vs-A is
an end-to-end transductive-procedure comparison, not attribution to EM alone. There is
no plain-PCA/raw-feature ablation in v1; do not claim the geometry heuristic is superior.

Mixture: p(z,y)=sum_k pi_k Normal(z;mu_k,Sigma_k) Bernoulli(y;q_k).
K={2,4,8}; covariance={tied,diag}; fixed seeds={42,314,2026}.
Joint responsibilities for labelled rows include the observed label likelihood;
unlabelled responsibilities do not. No hard pseudo-labels and no changing true labels.

Weighted objective: mean labelled joint log likelihood + lambda*mean unlabelled
marginal log likelihood + small symmetric log-prior on q. lambda=1 for B;
A has no unlabelled term. Separate means prevent raw U/L count domination.
Covariance ridge=0.001, class pseudocount=0.001, max_iter300, tol0.0001.
Covariance ridge updates are numerical stabilization, NOT exact maximization of a
specified covariance prior. Objective decreases are recorded; monotonic EM improvement
is NOT guaranteed. The convergence rule uses relative absolute objective change.
GMM initialization warnings are recorded. Candidates with any custom-EM nonconvergence
across all folds/seeds are ineligible for selection, but their scores are preserved.
Numerical exceptions fail the full run. Initial density GMM uses one seeded initialization;
its convergence warnings do not automatically invalidate a converged final custom EM.

## Outputs and interpretation

run_manifest.json, settings.json, split_indices.json, cohort.csv
fit_diagnostics.jsonl, geometry_diagnostics.json, seed_stability.csv
fold_scores.csv, repeat_scores.csv, model_summary.csv
comparison.csv (versus Trial25), B_vs_A.csv (matched K/covariance)
candidate_specs.json, predictions.npz, decision.json

Scores: mean5 fold AUCs per repetition, then mean10 repetitions; sample repeat SD.
Select the highest mean among fully converged seed-averaged candidates, exact ties
predeclared generation order A-before-B, K2/4/8, tied-before-diag. No seed tuning.
Selection is exploratory across eligible candidates. Screen against Trial25 with positive
mean>1e-12 and>=6 repetition wins. No blending, automatic promotion, test use or export.

B_vs_A is descriptive: also inspect convergence eligibility and seed stability, rather
than treating a positive single number as proof of the cluster assumption. Candidate
selection, reused labels, unequal fit scopes and research adaptivity limit interpretation.
Do not call these independent evaluation, nested CV, causal attribution, or significance.
Public0.92042 is NOT compared numerically with internal AUC in this experiment.

## Validation status

11 tests pass in the artifact-generation environment. They run real sklearn Gaussian
initializations and custom weighted EM, cover diagonal/tied covariance, label inversion,
finite normalized predictions, deterministic seeds, objective normalization, training-only
geometry transformation, invalid inputs, A/B runner integration and artifact hashes,
reference tampering and identity rejection, preflight no-write behavior and CLI dry-run.
The runner test uses a reduced four-fold/two-seed synthetic fixture (16 composite fits),
NOT the production900/1800 fits. Preflight tests exercise50 synthetic saved fold partitions.
The existing labelled loader was copied from the read GitHub file into the temporary test
checkout; it is not included/modified in this patch. The full repository was not cloned,
and its full test suite, your real-data preflight and production study were NOT executed.
