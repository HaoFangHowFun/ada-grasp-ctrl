# Shadow 752k：對齊 DexGraspBench 初始化的 Phase 1 對照

後續決策：使用者已指定此文件的「完整對齊初始化」為後續實驗基準。動態差異與下一輪對照見 [benchmark 動態差異分析](shadow-752k-benchmark-dynamics-gap.md)。

日期：2026-11-01。使用 `right_full_752k_v1/ckpts/step_050000.pth` 已生成的同一批 50 筆預測，沿用 10 個 scene、0.060／0.081／0.106 尺度，不重新推論或挑選樣本。全部為 Ada `op`，沒有 adaptive controller。

## 結果

| Ada 實驗 | pregrasp 外推倍率 | 初始拒絕條件 | 成功 | 失敗 | 初始拒絕 | 成功／全部 50 筆 | 成功／有效執行 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 既有基準 | 1 | 穿透 >5 mm 或接觸 wrench norm >1 | 22 | 9 | 19 | 44% | 22/31 = 71.0% |
| 只對齊拒絕門檻 | 1 | 穿透 >10 mm；不檢查力 | 34 | 16 | 0 | 68% | 68% |
| 只對齊 pregrasp | 2 | 原本 Ada 門檻 | 33 | 15 | 2 | 66% | 33/48 = 68.8% |
| 完整對齊初始化 | 2 | 穿透 >10 mm；不檢查力 | **34** | **16** | **0** | **68%** | **68%** |
| DexGraspBench 歷史設定（前次同批實驗） | 2 | 穿透 >10 mm；不檢查力 | 42 | 8 | 0 | 84% | 84% |

三組新實驗各 50 筆，control_eval、control_stat 均退出 0，沒有 execution error 或 solver degradation。表中「成功／全部」保留被拒絕的輸入；repo 正式成功率分母為 success + failure，兩者不能混用。這是固定小批次診斷，不是整體模型效能估計。

## 實際套用範圍

- 平移與手指：`new_pregrasp = grasp + 2 * (pregrasp - grasp)`。
- pregrasp quaternion 保持原值；使用 palm offset `[0, 0, 0.034]` 轉成 dummy-arm 平移，保留既有 Euler 旋轉。沒有外推 dummy-arm 旋轉。
- grasp、squeeze 與物體初始 pose 逐元素保持不變；squeeze 倍率仍為 1，沒有加深。
- 對齊的拒絕設定為 `max_pene=0.01`、`max_force=inf`。Infinity 只用在門檻設定，表示關閉初始力拒絕，不涉及非有限的控制指令。
- 相同 sample index、seed 12、單 worker、offset 0。保留 Ada 的物理、驅動、各階段時間、抬升距離與成功判準。沒有修改公開預設或既有實驗輸出。

這裡的「相同初始化」指 pregrasp 幾何、物體 pose 與初始拒絕條件；兩套 simulator 的致動器、動態模型與完整內部狀態並未變成相同。

## 原因分析

完整對齊後，原本成功的 22 筆仍成功，原本失敗的 9 筆仍失敗；原本 19 筆初始拒絕變成 12 成功、7 失敗。因此先前的低完成成功比例，有很大部分來自初始化篩選。不能把被拒絕直接解釋成模型一定抓不起來。

只放寬門檻也有 34 成功，但並非逐筆相同：加上 pregrasp 外推後，g0019 從成功變失敗、g0023 從失敗變成功，總數抵銷。外推的主要可見作用是降低初始接觸：只外推時拒絕從 19 降到 2；剩餘 g0018、g0024 的穿透分別約 3.939、0.951 mm，仍因初始 wrench norm 約 184.6、13.7 超過 1 而拒絕。這是 reset 後的瞬時數值，不能視為穩態抓取力。

完整對齊不表示完全沒有穿透：仍有 2 筆輕微手物重疊，最大 3.939 mm，都低於 10 mm 門檻。關閉力拒絕只是允許動態執行，沒有消除接觸。

與 benchmark 歷史設定逐筆對照：34 筆兩邊成功、8 筆兩邊失敗、8 筆 benchmark 成功但 Ada 失敗；沒有 Ada 成功而 benchmark 失敗的樣本。後者為 g0003、g0016、g0018、g0019、g0042、g0043、g0044、g0046。

這 8 筆在 Ada 結束時物體淨高度變化介於約 -0.11 至 +1.69 mm，並非只差一點成功閾值。回放採樣顯示 g0018 在 lift 階段曾升高約 92.4 mm、g0046 約 165.1 mm，最後都回落；另 6 筆採樣到的最大升高只有約 5.1–12.3 mm。這支持繼續調查動態抓持與滑落，但尚不能單獨歸因於摩擦或某個致動器設定。

仍未對齊的因素包括 Ada 獨立關節致動器／dummy arm 與 benchmark mocap／tendon、摩擦 0.6 與 0.8、固定質量 0.1 kg 與隨尺度密度策略，以及每階段 2 秒與 1 秒、抬升 0.2 m 與 0.1 m。成功判準也不同。下一步適合固定這 8 筆差異案例逐項做對照；本次沒有擅自修改這些設定。

## 驗證與產物

- 150 筆新回放的三階段 targets 均與各組 prepared input 完全相同；有效執行各 62 frames，初始拒絕各 2 frames。
- 原始 50 筆 prediction、prepared input 的 SHA256 均保持不變；新組別與基準 sample seed 相同。
- 完整對齊的初始 palm 位置與 benchmark 目標最大誤差約 `3.35e-9 m`；初始手物接觸距離最大差約 `2.52e-9 m`。
- 原生 resolved manifests、eval/stat reports、NPY 控制記錄與 replay sidecars 均保留於各組目錄。

結果根目錄：`output/shadow-phases/dexlearn-752k-three-scales-v1/ada-benchmark-init-v1/`。

- `comparison.json`：四組逐筆分類與計數。
- `analysis.json`：分類轉移、benchmark 差異案例、初始化幾何驗證。
- `validation.json`：derived pregrasp 與 benchmark 對齊、grasp/squeeze 不變的樣本清單。
- `suite/suite_manifest.json`：獨立衍生資料與來源 hash。
- `{gates_only,pregrasp_only,matched}/phase1/`：各組原生結果與回放。

實驗腳本位於 workspace 的 `reviews/ada_shadow_phases_20261101/run_matched_initialization.py`，分析腳本為同目錄 `analyze_matched_initialization.py`。執行環境為 Ada `.conda/bin/python`，清除 `PYTHONPATH`／`PYTHONHOME`。執行腳本要求全新輸出目錄，避免混入舊結果。

8081 預覽已切換至完整對齊初始化的結果：<http://localhost:8081>。選擇 `Phase 1 replay` 再播放可檢查實際動態；`Target poses` 是靜態命令目標。標題會標明 pregrasp x2、10 mm、force gate off、squeeze unchanged。

前次跨環境調查：[shadow-752k-penetration-environment-comparison.md](shadow-752k-penetration-environment-comparison.md)。
