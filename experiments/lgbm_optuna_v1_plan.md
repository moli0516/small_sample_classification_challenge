# LGBM Optuna V1 — predeclared plan

Base code interface: ccad344b6395101f2407919817400ad670bb3bd7.
Question: can bounded LightGBM tuning improve the fixed equal-weight three-model
ensemble over the existing L1+XGBoost development reference?

- Primary objective: l1_xgb_lgbm mean fold AUC per repeat, then mean of 10 repeats.
- Secondary descriptive scores: three individual models and all four equal blends.
- Models: existing all-feature L1 C=1; frozen Trial20/top20 XGBoost; all-feature LightGBM.
- Splits/data: unchanged 250-labelled-row dataset and existing verified 10x5 splits.
- Cached controls: previously completed L1/XGBoost OOF; no retraining/tuning those.
- Control trial0: reproduce all seven candidates of the completed fixed-LGBM run.
- Budget: 20 trials including control/failures, at most 1000 new fits, sequential.
- Search: tree_shape x child count x learning rate x round count x lambda x column sampling.
- Fixed weights: 1/3 each; no early stopping, pruning or public-score optimization.
- Selection: highest primary objective, earliest trial on exact ties.
- Promotion: that winner must improve mean AUC vs L1+XGBoost and win >=6/10 repeats.
- No fallback to a secondary ensemble or a different trial after inspecting results.
- Reproducibility: exact input/code/version/artifact hashes, immutable output paths.
- Evaluation limitation: adaptive development, NOT independent or nested evaluation.

Reported existing candidate public AUC: 0.91514 (user-reported, retained).
Study execution: pending.
Native-library full integration: pending local verification.
Actual selected parameters and CV result: pending.
Submission ID/time/public/private result: pending.
