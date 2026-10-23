# LightGBM Optuna + clil.py 操作手冊

## 交付及相容性

本 patch 以 GitHub commit ccad344b6395101f2407919817400ad670bb3bd7 的介面為依據。
只新增檔案，不修改現有 runner、registry、模型 wrapper、舊設定、frozen artifacts 或結果。
新增入口確實叫 clil.py，按使用者指定名稱保留，並非 cli.py。
.patch 及 ZIP 中的同名 patch 完全相同，只套用一次；ZIP 不另附需要手動覆寫的程式檔案。

git apply --check 只驗證檔案套用，不會證明本機資料或模型正確。
本機 preflight 還會驗證 frozen data、OOF、既有固定 LightGBM 實驗、版本及來源 hash。
不得透過改 hash 或放寬 tolerance 繞過失敗。

## 0. 套用 patch

在 small_sample_classification 專案根目錄、原本 .venv 中執行。
先保存/備份自己的工作。假設 patch 已放在 Downloads：

    git apply --stat ~/Downloads/lgbm_optuna_clil.patch
    git apply --check ~/Downloads/lgbm_optuna_clil.patch
    git apply ~/Downloads/lgbm_optuna_clil.patch
    git status --short

若檔案已存在或 patch 已套用，停止，不要再 force apply。

## 1. 安裝與環境檢查

    python -m pip install -r requirements-lgbm-optuna.txt
    python clil.py doctor

依賴包含既有 requirements-registry.txt 及 Optuna >=3,<6。
不要另外任意 upgrade numpy/pandas/scikit-learn/XGBoost/LightGBM；版本變動可能使
既有 OOF cache 不能重用。doctor 預設真的 import 原生套件以檢查 native library 載入。

## 2. 測試

    python clil.py test --scope new
    python clil.py test

新測試失敗或既有測試回歸失敗，先停止。
建置環境 34 個測試：32 passed、2 skipped（沒有原生 Optuna/LightGBM）。
合成測試使用假的 Optuna/registry/來源 adapter，不是 TPE 效能或真實 50-fold 訓練驗證。
包含兩個 optional native tests，本機套件存在時會跑 Optuna SQLite/control budget 及 LightGBM fit。
既有完整 repo suite 未在建置環境執行；實際 250x150 資料、XGBoost/LightGBM/Optuna
端到端研究尚未執行，沒有產生真實 Kaggle predictions。

## 3. 檢查註冊名稱與所有指令

    python clil.py models
    python clil.py --help
    python clil.py lgbm-optuna --help
    python clil.py plan

plan 只列印步驟，不會自動訓練/安裝/上傳。
子命令 help：例如 python clil.py registry cv --help。

## 4. 搜尋前 preflight

    python clil.py lgbm-optuna search --dry-run

預期核心欄位：
- status=preflight_passed
- objective=l1_xgb_lgbm
- n_trials=20
- max_new_fits=1000
- cached_models=[l1,xgboost]
- control_trial=0
- early_stopping=false, pruning=false, test_opened=false

preflight 不新增 study 目錄、不做任何 fits。
要求原本兩個來源皆已完成並可驗證：
- outputs/results/soft_vote_l1_xgb20_v1_run_01（L1/XGBoost OOF）
- outputs/results/registry_ensembles_v1_run_01（固定 LightGBM 比較）
以及 outputs/frozen/ext03_top20_v1。

不提供 --no-cache 搜尋：這輪預算建立在固定 L1/XGBoost OOF 可驗證重用之上。

## 5. 開始正式搜尋

    python clil.py lgbm-optuna search

預設設定 configs/lgbm_optuna.json；輸出 outputs/results/lgbm_optuna_v1_run_01。
20 trials 含控制 trial0，最多 1000 次新增 LightGBM fits。每 trial 50 folds。
TPESampler(seed=42,n_startup_trials=5)，單執行緒 study，NopPruner，SQLite 保存 study。
Trial0 透過 enqueue_trial 使用現有完整 LightGBM 設定；其七候選逐 fold AUC
必須重現既有 registry 實驗（1e-12）。若控制失敗，整輪停止。
後續每 trial 仍逐 fold 核對 L1、XGBoost、既有雙模型 ensemble 控制分數。

只最佳化固定等權重的 (L1+XGBoost+LightGBM)/3 平均 repeat AUC。
所有四種 ensemble 仍在每個 trial 的 comparison.csv 中保存，但不得輪流挑最高
組合作為 objective；最高 LightGBM 單模型也不一定是本研究的最佳 trial。

搜尋空間：
- tree_shape（max_depth:num_leaves）：2:3,3:3,3:7,4:3,4:7,4:15。
- min_child_samples：5..30。
- learning_rate：0.02..0.15，log。
- n_estimators：50/100/200。
- reg_lambda：0.1..30，log。
- colsample_bytree：0.6..1。
其餘參數照既有固定設定，包括 random_state/n_jobs/deterministic。
使用 tree_shape 避免每個 trial 改變 categorical distribution，並保留原 3:7 控制。

不採 early stopping、pruning、權重搜尋或 leaderboard feedback。
Python-warning-bearing 非控制 trial 會記為 FAIL，佔用 20 次名額，不額外補跑。
控制 warning、輸入/code 改動、OOF/control/版本錯誤則停止整個 study。
原生 LightGBM logs 保存在各 trial/cv/fit.log；不把所有「No further splits」訊息
自動當成 Python warning，但應檢視 diagnostics 與實際樹數/預測退化。

## 6. 查看結果

    python clil.py results --run outputs/results/lgbm_optuna_v1_run_01

主要輸出：
- study.sqlite3：Optuna 持久化儲存；本版本沒有原地續跑功能。
- settings.json：objective、搜尋空間、版本及輸入/code hashes。
- trials.csv / trial_results.json：trial 結果與拒絕紀錄。
- best_trial.json：最高 objective AUC 的 trial、完整 LightGBM 參數、cv_run。
- decision.json：是否晉升。
- trials/trial_0000/registry_config.json：控制設定。
- trials/trial_XXXX/cv/comparison.csv：該 trial 的全部七個候選。
- trials/trial_XXXX/cv/validation_probabilities.csv：該 trial 的預測。
- run_manifest.json：完成/失敗狀態、預算、來源檢查及產物 checksums。

選最高 objective AUC，完全同分取較早 trial。只有這個 winner 對既有 L1+XGBoost
平均 AUC 有正改善且至少 6/10 repeat wins，才 promote=true。不從其他 trial
或 secondary ensemble 救援選模。這是實務規則，不是顯著性檢定。
所有數字仍是 adaptive development 結果，不是 nested CV 或獨立最終估計。

## 7. 只有 promote=true 才 export

    python clil.py lgbm-optuna predict --dry-run
    python clil.py lgbm-optuna predict

預設選 best trial 的 l1_xgb_lgbm，而非該 trial registry decision 裡可能最高的其他組合。
會先驗證 study artifacts、原始設定、版本、來源 hash，然後委派現有 registry.predict。
全部 250 rows refit 三個分支，實際 4 fits：L1(1)+XGBoost ranker/classifier(2)+LightGBM(1)。
已有 runner 執行 save/load、predict agreement、template ID alignment、CSV roundtrip。

預設輸出 outputs/submissions/lgbm_optuna_v1_run_01：
- submission.csv（唯一手動上傳的檔案）。
- models/*.joblib、ensemble.json、prediction_manifest.json、warnings.json、fit.log。

確認 prediction_manifest.json status=ready_for_manual_submission、rows=19750、
fits_performed=4、saved_and_reloaded_models_verified=true，並檢查警告/日誌。
沒有自動 Kaggle upload。不要上傳 partial/failed CSV。
若未 promote，保留已完成的 0.91514 candidate。

## 改輸出路徑 / 中斷後重跑

既有 output 永不覆寫；也不會偷偷刪掉/恢復半輪 Optuna。

    python clil.py lgbm-optuna search --output outputs/results/lgbm_optuna_v1_run_02 --dry-run
    python clil.py lgbm-optuna search --output outputs/results/lgbm_optuna_v1_run_02
    python clil.py results --run outputs/results/lgbm_optuna_v1_run_02
    python clil.py lgbm-optuna predict --run outputs/results/lgbm_optuna_v1_run_02 --output outputs/submissions/lgbm_optuna_v1_run_02 --dry-run
    python clil.py lgbm-optuna predict --run outputs/results/lgbm_optuna_v1_run_02 --output outputs/submissions/lgbm_optuna_v1_run_02

改參數/程式/設定會破壞已執行研究的 hash 一致性。完成研究後先封存再規劃新版本。

## 既有實驗統一入口（不更改舊參數或預設輸出）

clil.py 只用目前 sys.executable 以 subprocess 呼叫原 module，cwd 固定專案根目錄，
原 arguments 原樣轉送，不 shell=True，不自動重跑已有研究：

    python clil.py baseline --help
    python clil.py xgb --help
    python clil.py xgb-optuna --help
    python clil.py xgb-features --help
    python clil.py frozen-xgb --help
    python clil.py soft-vote --help
    python clil.py registry --help
    python clil.py extension --help
    python clil.py audit --help

其餘已知 scripts：

    python clil.py run make_submission --help
    python clil.py run create_e4_configs --help

轉送後的實際旗標仍由舊 CLI 決定，不假設所有舊 script 都有 dry-run。
這是統一入口，不是重寫既有模型生命週期；目前清單涵蓋現有實驗/配置產生入口。

## 安全範圍

不修改原有 GitHub、不自動 commit/push、也不自動上傳任何資料。
joblib 僅載入本機自己產生/信任的模型。
本 patch 不包含訓練資料、舊輸出、模型二進位檔或 credentials。
