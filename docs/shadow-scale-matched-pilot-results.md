# Shadow：限制為訓練尺度的新一批 50-grasp 結果

日期：2026-11-01。接續[尺度調查](shadow-pilot-object-scale-audit.md)，使用新的 scene 清單、模型預測、準備資料與輸出目錄；沒有覆寫前一批。

## 結果

| 狀態 | Phase 1：op | Phase 2：ours |
| --- | ---: | ---: |
| success | 0 | 1 |
| failure | 3 | 2 |
| invalid_initialization | 47 | 47 |
| solver_degraded／execution_error／preparation_error | 0 | 0 |
| success / selected 50 | 0% | 2% |
| success / (success + failure) | 0/3 = 0% | 1/3 = 33.3% |

配對：47 筆兩者初始化無效、2 筆兩者 failure、1 筆 Phase 1 failure／Phase 2 success。唯一成功為本批 `g0006`，物體為 camera。兩邊能開始執行的只有 `g0002`、`g0005`、`g0006`。

**限制到訓練尺度後，結果沒有改善。** 47 筆無效初始化中，42 筆手／物體穿透超過 5 mm；另 5 筆不屬此條件，仍觸發原生其他初始化檢查。未放寬碰撞／接觸力門檻。

這說明尺度不是唯一問題，但不能把新舊結果差異完全歸因於尺度：換了 scenes 並重新推理，候選與物體擺放也可能改變。這不是只縮放舊預測的單變因實驗。

## 清單如何固定

保留舊批相同的十個 test objects，每物件一個 scene、每 scene 五個預測。依當前訓練資料的 grasp 加權尺度分布，以 largest-remainder 分配十個 scene 名額，再以 seed 42 將尺度隨機分配給十個物件。

| Scale | Scene 數 |
| --- | ---: |
| 0.060 | 2 |
| 0.081 | 2 |
| 0.106 | 2 |
| 0.133 | 2 |
| 0.162 | 1 |
| 0.194 | 1 |
| 0.227 | 0 |

80% scenes 的 scale ≤ 0.133，接近訓練 grasp 分布的 81.68%；僅十個名額，不能精確重現完整分布，最少見的 0.227 分得零個名額。所有物件都有對應尺度與原 pose 檔名 suffix 的 scene，無需 fallback、替換物件或依結果重挑。保留 pose 檔名不保證不同尺度的 settled pose 數值相同。

清單在推理前寫入 manifest。使用原 100k checkpoint、sampling seed 42、每 scene 100 proposals、per-type top-k 5，共 50 筆。Ada seed 12，兩個 phase 共用一次 CPU IK；仍保留原始 pregrasp／grasp／squeeze，額外 squeeze 為零。

## 新尺寸與預覽

尺寸由完整點雲的物體局部 bounding box 計算，單位 cm，與碰撞 mesh 有取樣及近似差異。

| ID 範圍 | 物體 | Scale | 尺寸（cm） |
| --- | --- | ---: | --- |
| g0000–g0004 | bottle | 0.081 | 4.8 × 15.0 × 4.8 |
| g0005–g0009 | camera | 0.060 | 7.4 × 5.5 × 7.9 |
| g0010–g0014 | jar | 0.060 | 7.0 × 6.4 × 7.0 |
| g0015–g0019 | mug | 0.106 | 10.4 × 11.8 × 14.6 |
| g0020–g0024 | mug | 0.081 | 8.2 × 8.2 × 11.6 |
| g0025–g0029 | tetra pak | 0.106 | 10.1 × 10.4 × 15.8 |
| g0030–g0034 | sauce container | 0.194 | 17.1 × 27.9 × 21.8 |
| g0035–g0039 | cereal box | 0.133 | 5.5 × 15.7 × 21.3 |
| g0040–g0044 | mouse | 0.162 | 16.4 × 27.3 × 7.1 |
| g0045–g0049 | table clock | 0.133 | 14.9 × 19.8 × 10.8 |

**符合訓練尺度不代表一定是日常物品大小**，例如 mouse 仍約 27 cm。這批目的是對齊模型資料的尺度分布，沒有按物品的日常大小手動挑選尺度。

- 新批次預覽：[localhost:8081](http://localhost:8081)。
- 舊批次預覽保留在 [localhost:8080](http://localhost:8080)。
- 兩個頁面的 g 編號對應相同物件，但代表各批重新產生的候選，不是同一筆姿態。畫面為目標姿態的靜態預覽，並非 controller 實際軌跡。

## 驗證與檔案

50 筆全部準備成功，兩個 phase 各有 50 筆 eval／stat 結果、合計 100 個 replay sidecars。50 組初始狀態、seed、prepared hashes、三階段 effective targets 一致。

150 個姿態 FK 的最大 palm 位置誤差為 0.834 mm、旋轉誤差 0.00801 rad 以下，named finger joint 誤差為 0。逐筆比對原始 mesh 經 scene scale／pose 的世界 bounds 和 MuJoCo 編譯後 bounds，最大誤差 6.04 × 10⁻⁹ m 以下。

- [推理前固定的 scene manifest](../output/shadow-phases/scale-matched-v1/scene_manifest.json)
- [命令與退出碼](../output/shadow-phases/scale-matched-v1/stage_results.json)
- [共用 suite manifest](../output/shadow-phases/scale-matched-v1/suite/suite_manifest.json)
- [配對結果](../output/shadow-phases/scale-matched-v1/run/comparison.json)
- [Phase 1 診斷](../output/shadow-phases/scale-matched-v1/run/phase1/phase_results.json)、[Phase 2 診斷](../output/shadow-phases/scale-matched-v1/run/phase2/phase_results.json)
- [FK／尺寸驗證](../output/shadow-phases/scale-matched-v1/run/validation.json)
- [抽樣與執行腳本](../../reviews/ada_shadow_phases_20261101/run_scale_matched_pilot.py)

此次未變更控制程式或科學參數，使用既有入口完成實際整合驗證，沒有重跑無關的 unit suite／release gate。下一個應優先隔離的問題仍是：同 scene 的資料集參考 grasp 在 Ada 是否也穿透，以及兩邊手模型的碰撞設定是否一致。
