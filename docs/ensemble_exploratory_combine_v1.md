# Exploratory combination v1

Branch: feat/ensemble-exploratory-combine-v1
Requires the original ensemble research suite code and completed local catboost/lgbm_es/
xgb_es/stacking studies. This ADDITIVE patch does not modify that suite's code or its
screening and combination policy. Its source_signature remains unchanged.

    git switch -c feat/ensemble-exploratory-combine-v1
    git apply --check ensemble_exploratory_combine_v1.patch
    git apply ensemble_exploratory_combine_v1.patch
    python -m pytest -q tests/test_ensemble_exploratory_combine_v1.py
    python -m scripts.run_ensemble_exploratory_combine_v1 --dry-run
    python -m scripts.run_ensemble_exploratory_combine_v1

Freeze CatBoost cat_10_alpha0.1: depth4, iterations100, learning_rate0.1, l2_leaf_reg3,
alpha0.10. Fixed original three-component relative weights: equal thirds.
Four new candidates: CatBoost + LGB early stopping; CatBoost + XGB early stopping;
CatBoost + BOTH early-stopped replacements; CatBoost + fixed stacking output.
The first three are 0.3*L1 + 0.3*XGB + 0.3*LGB + 0.1*CatBoost with specified replacements.
The fourth is 0.9*stacking_probability + 0.1*CatBoost_probability, not a five-model mean.
No weight/parameter search, retraining, competition test access or submission is added.

Each child must pass the suite's artifact, source/cohort and saved-decision reproduction
checks. Frozen CatBoost probabilities AND its selected mean must reproduce. The selected
candidate is the highest-mean exploratory combination, with exact ties in listed order.
Both original ensemble and frozen CatBoost challenger are controls. Candidate screening
against BOTH controls is reported separately; no fallback or automatic promotion.

This is deliberately POST-HOC analysis of combinations, including individually failed
changes, requested after seeing single-change scores. Do not rewrite the original suite
as though these candidates were preregistered or call this independent confirmation.
A small positive result remains adaptive development evidence with extra selection.

Default output: outputs/results/ensemble_research_suite_v1_run_01/exploratory_combine_v1
Must not exist; --output permits another fresh directory. Existing source output is never
overwritten. Inspect model_summary.csv, comparison_vs_original.csv,
comparison_vs_catboost.csv and decision.json. Only completed manifests are usable.

Five new tests exercise exact combination formulas, no input mutation, fixed alpha,
invalid-probability refusal and no .fit call. In the author sandbox these tests use a
minimal suite contract because the full project/native artifacts are unavailable. They
are not a full real-source integration test; local --dry-run is mandatory.
