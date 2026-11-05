# Shadow controller：固定 2,000 次評估流程

## 已確認的規模與用途

依使用者決定，以 Ada 隨附資產為準，總共 2,000 次評估，不是 200 個不同物體：

- 89 個不同物體、100 個原生 scene。
- 每 scene 使用 `right_full_752k_v1/step_050000.pth` 的 100 proposals，固定保留 top-k 10，共 1,000 grasps。
- Phase 1（op，無 Ada controller）與 Phase 2（ours，有 Ada controller）各評估 1,000 筆。

所有場景皆來自 `examples/assets/object/DGN_2k`。保留原生 mesh、尺寸與 pose，不補入外部 BODex 物件，不重新縮放。完整原生子集包含訓練與測試物件，因此此流程是 **controller 回歸測試**，不是 DexLearn held-out 泛化效能評估。

| 原生尺度 | 場景數 | 每 phase grasp 數 | 兩 phase 評估次數 |
| --- | ---: | ---: | ---: |
| 0.06 | 38 | 380 | 760 |
| 0.08 | 21 | 210 | 420 |
| 0.10 | 19 | 190 | 380 |
| 0.12 | 22 | 220 | 440 |
| 合計 | 100 | 1,000 | 2,000 |

四尺度都有涵蓋，但不等量。這裡「評估次數」包含初始化被拒絕的 episode；拒絕的輸入不會被換成其他成功樣本。

## 執行指令

在 repo 根目錄執行，或使用 sh 的絕對路徑。主腳本不開 viewer。

```bash
# 日常完整驗證：自動使用唯一的新輸出目錄，共 2,000 次評估
bash script/test_shadow_controller.sh

# 第一次建立完整結果，供你選作比較基準
bash script/test_shadow_controller.sh \
  --output-root output/regression/baseline-v1

# 改動 Ada controller 後，使用完全相同輸入比較
bash script/test_shadow_controller.sh \
  --output-root output/regression/controller-change-v1 \
  --baseline output/regression/baseline-v1
```

後兩個命令是完整執行的範例，本文驗證階段尚未產生 `baseline-v1`。輸出目錄必須不存在；基準只讀，不會被自動覆蓋或升格成新 golden。

先檢查程式是否能跑，可使用：

```bash
# 每尺度一筆，保留它在完整 suite 的原始 sample index／seed，共 8 次評估
bash script/test_shadow_controller.sh --smoke

# 只建立或驗證固定 suite，不跑控制
bash script/test_shadow_controller.sh --prepare-only
```

Smoke 與完整 2,000 次結果不能互作基準；腳本會檢查 suite digest、protocol 及 sample ID 清單，遇到不相容比較直接拒絕。

## 預覽指令獨立

```bash
bash script/preview_shadow_regression.sh \
  --run-root output/regression/controller-change-v1 \
  --port 8083
```

開啟 `http://localhost:8083`，Phase 選單切換兩階段，Sample 選案例，Recorded replay 看實際執行。預覽只載入記錄，不執行模型或控制器。可改 port，避免與現有 8081／8082 服務衝突。

目前已驗證的 smoke 結果可直接預覽：

```bash
bash script/preview_shadow_regression.sh \
  --run-root output/regression/smoke-native-v2 --port 8083
```

## 為何不每次重新生成 grasp

首次準備時會建立完整點雲、一次性模型推論、關節轉換與 IK，再將完整初始化衍生輸入保存。之後主 sh 直接重用固定的 prepared grasps，**不重新抽物體、不重新推論、不重新 IK**。如此改 controller 前後才能比較同一輸入。

預設固定資料目錄：

```text
output/regression/shadow-native-100x10-v1/
```

`scene_manifest.json`、`generation.json` 保存原生 scene、checkpoint hash、metadata hash、推論命令；`suite_manifest.json`／`.sha256` 保存 1,000 筆來源與 prepared hashes；dependencies 保存原物體資產與生成點雲 hashes。

每次執行前驗證固定 manifest、raw prediction、prepared input、scene 和 object dependencies。既有目錄若不完整或內容變動會失敗，不會默默重新生成。需要換模型或測試協定時，應另外建立 suite 版本，不能沿用原基準比較。

本機已完成 suite 準備。日常驗證只需要 Ada 的 CPU 相容環境；首次準備另外需要本機 HUGS-DexLearn GPU 環境、752k checkpoint 與對應訓練 metadata。環境路徑可覆寫：

```bash
PYTHON_BIN=/absolute/path/to/ada/python \
DEXLEARN_ROOT=/absolute/path/to/HUGS-DexLearn \
SHADOW_CHECKPOINT=/absolute/path/to/step_050000.pth \
SHADOW_METADATA=/absolute/path/to/right_full/metadata.json \
SHADOW_SUITE_ROOT=/absolute/path/to/new-suite \
bash script/test_shadow_controller.sh --prepare-only
```

目前固定 suite 保留本機絕對路徑；移動或搬到另一台機器需要重新準備或明確遷移來源，不能聲稱開箱即用的可攜資料包。原始資產沒有被修改或重新宣告授權。

## 固定科學設定

- 摩擦 `[0.8, 0.02]`，物體質量 0.1 kg。
- pregrasp 平移與手指 `grasp + 2*(pregrasp-grasp)`；quaternion 不變。
- 初始穿透門檻 10 mm、停用初始力 gate。
- grasp／squeeze input 不變，無額外 squeeze 外推。Phase 2 仍依 controller 計算實際命令。
- offset 0、seed 12、預設 8 workers（`--workers 1` 可切回串行）、既有 dummy-arm/no_tendon、原生 stage／lift／成功判定。
- 保留原生 solver failure policy，不調鬆數值容差、不修改既有 golden。

這些是本 regression 入口的固定 overrides，沒有變動公開預設。你改動 controller 的 objectives、參數或實作時，其他變動會反映於各次 resolved manifest、code_state 與結果；不應同時改 frozen inputs。

## 結果與退出碼

每次新結果包含：

- `summary.md`：Phase 1／2、每尺度分類，以及基準差異。
- `comparison.json`：每個 sample 的兩階段分類及成功率分母。
- `scale_results.json`：四尺度分組計數。
- `baseline_delta.json`：有指定基準時，逐筆分類變化、失去的成功、新退化、新執行錯誤。相同成功總數但交換了成功／失敗案例也會被指出。
- `regression.json`／`experiment_manifest.json`：suite digest、protocol、sample IDs；比較時必須相容。
- `code_state.json`：Git HEAD 與實際工作樹中的程式、設定、hand XML／shell hashes，涵蓋未提交 controller 修改。
- `phase{1,2}/progress.jsonl`：每筆即時進度及時間。
- 各 phase 的原生 eval／stat manifests、run reports、failures，以及控制 NPY、scene XML、replay、phase_results。

每筆保留 success、failure、invalid_initialization、solver_degraded、execution_error，不把退化算成功。摘要同時給 success/selected 與 success/(success+failure)，不能只看排除退化後的成功率。

| Exit code | 意義 |
| --- | --- |
| 0 | 評估正常完成、無求解退化／執行錯誤；若指定基準，也未出現成功流失等已檢查退步 |
| 1 | 求解退化、執行／統計錯誤、缺失結果，或基準比較發現成功流失／新退化／新執行錯誤 |
| 2 | 輸入／環境／基準不相容或 preflight 失敗 |

有物理抓取 failure 不必然導致非零退出；這沿用 native 應用程式退出語意。反之，**即使退化與基準完全相同，仍會返回 1**，不會為了讓 CI 變綠而吞掉 solver_degraded。需搭配 `baseline_delta.json` 區分「已存在的退化」與「這次新增的退步」。目前比較是分類層級，不是完整軌跡 golden gate。

## 成本與本次驗證

先前 40 筆的原生資產測試，eval manifest 到 report 的時間約 Phase 1 8.5 秒、Phase 2 37.5 秒；記錄約 8 MB 與 30 MB。以比例估算，1,000 筆／phase 約 **20 分鐘、約 1 GB／次**，但 optimizer 的接觸數／收斂速度、序列執行、CPU 負載會使實際時間不同，並非完成時間保證。這不含首次準備時間。

本次完成：

- 固定 100 scenes／89 objects／1,000 grasps 已建立且全部準備成功。
- 全部 3,000 targets 做 FK 稽核，最大 palm 位置誤差約 0.489 mm，沒有超過 1 mm／0.01 rad 的既定檢查範圍；`fk_audit.json` 保存結果。
- 兩次乾淨 smoke（各 8 episodes）均完成；第二次從 `/tmp` 用絕對 sh 路徑執行且指定第一輪基準，分類無差異。
- 8 對重複回放的 qpos／qvel／ctrl／act／object_pose／time 及 segment 通過 rtol=1e-5、atol=1e-6 比對。
- 135 個 unit tests 通過；全專案 Ruff check、compileall 通過；新增檔案 format check 通過。全專案 format check 尚有 8 個既有、無關檔案不符，未順手改動。
- shell 語法與獨立 preview help 入口通過。

**尚未執行完整 2,000 次，也尚未建立完整控制結果基準。** 本次先完成自動化、固定輸入及小規模重複驗證，之後用上方正式指令建立你的完整 baseline。這是額外的 controller 實驗工具，不取代現有 quick release gate。

## CPU workers 更新

主 sh 預設使用 8 workers；Phase 1 完成後才執行 Phase 2。可指定 `--workers 2`、`--workers 4` 或 `--workers 1`。例如：

```bash
bash script/test_shadow_controller.sh --workers 8 --output-root output/regression/baseline-v1
```

worker 數記錄在 experiment manifest 與 resolved config。保留完整 suite 的 sample_index、seed、原始結果排序及既有錯誤回報；不調整 BLAS threads。`progress.jsonl` 依輸入順序寫入，較慢案例完成前可能暫無進度輸出，其他 workers 仍可繼續運算。

本次更新僅做靜態檢查，未啟動模擬或完整回歸。先前的小規模測速請見 [worker benchmark](shadow-controller-worker-benchmark.md)；完整 2,000 次結果仍待實際執行。
