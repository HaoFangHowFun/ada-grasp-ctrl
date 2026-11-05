# DexLearn 752k：三尺度 Phase 1

日期：2026-11-01。依使用者指定，採用 `right_full_752k_v1` 的 `step_050000.pth`，場景實際 scale 僅允許 **0.060、0.081、0.106**。只執行 Phase 1，沒有 Ada 自適應控制。

後續已對同一批預測執行 DexGraspBench 對照：42/50 成功、0 初始化拒絕。Ada 的 19 筆無效中僅 5 筆超過 5 mm，另外 14 筆由初始接觸力 gate 拒絕；詳見[跨環境穿透調查](shadow-752k-penetration-environment-comparison.md)。

## 結果

| 狀態 | 數量 |
| --- | ---: |
| success | 22 |
| failure | 9 |
| invalid_initialization | 19 |
| solver_degraded／execution_error／preparation_error | 0 |
| success / selected | **22/50 = 44%** |
| success / (success + failure) | 22/31 = 70.97% |
| 至少有一筆成功的 scene | 9/10 |

這批可執行性明顯優於先前 100k 批次，但 checkpoint 與 scene 尺度同時改變，不能將改善全部歸因於其中一項。與之前 BODex 的 48% 也不是同場景的嚴格比較：物件相同，這次部分尺度／scene 不同。沒有重跑 BODex 或 Phase 2。

## 選擇與流程

- 保留原十個 test objects，每個一個 scene、五個預測，共五十筆。
- 三個尺度採近似等額分配：0.060 為四個 scenes，0.081 和 0.106 各三個；seed 42 隨機分配至十個物件。在推理前固定 scene manifest，不依結果重選。
- 優先保留原 pose 檔名 suffix。使用實際存在的 scene，不直接縮放舊預測或舊場景資產。
- 重新載入 **752k checkpoint**，metadata 與 training data path 皆改為 `BimanBODex_shadow_tabletop_777k_v1/shadow/right_full` 對應版本。
- complete PC、1,024 points、每 scene 100 proposals、per-type diffusion top-k 5；sampling seed 42。全部 50 筆轉換與 CPU IK 成功。
- Ada seed 12，原生 `op`，`arm_pregrasp_is_grasp=false`、`pregrasp_t=0`、`squeeze_t=0`。保留模型三階段目標，不額外加深 squeeze；執行原生 0.2 m lift。

## 預覽

**[localhost:8081](http://localhost:8081)** 已切換至本批，頁面標題明確標示 `right_full_752k_v1` 與三個尺度。

- `Target poses`：查看 pregrasp／grasp／squeeze。
- `Play Phase 1`：播放實際執行與 lift 記錄。
- 可先看成功案例 `g0000`、`g0006`、`g0010`，或 `g0001` 的無效初始化。
- 舊 BODex 與 DexLearn 結果均保留在原輸出目錄，未覆寫。

## 驗證與追溯

已驗證 checkpoint hash、全部 scene scale、50 筆 eval／stat 結果與模型姿態轉換。150 個 palm FK 最大位置誤差 0.436 mm 以下、旋轉誤差 0.00795 rad 以下，named finger joint 誤差 0，effective targets 與 prepared targets 完全一致。

- [推理前固定的 scene manifest／checkpoint SHA](../output/shadow-phases/dexlearn-752k-three-scales-v1/scene_manifest.json)
- [完整命令與退出碼](../output/shadow-phases/dexlearn-752k-three-scales-v1/stage_results.json)
- [Phase 1 詳細結果](../output/shadow-phases/dexlearn-752k-three-scales-v1/run/phase1/phase_results.json)
- [結果摘要](../output/shadow-phases/dexlearn-752k-three-scales-v1/run/comparison.json)
- [FK 與完整性驗證](../output/shadow-phases/dexlearn-752k-three-scales-v1/run/validation.json)
- [生成與執行腳本](../../reviews/ada_shadow_phases_20261101/run_752k_three_scales.py)

comparison 的 Phase 2 missing_result 表示此次刻意未執行。沒有改動控制程式、物理參數或 golden；此次以完整推理→準備→Phase 1 流程驗證，未重跑無關的 release gate。
