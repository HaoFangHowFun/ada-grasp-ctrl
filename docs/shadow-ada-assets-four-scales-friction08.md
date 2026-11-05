# Ada 原生資產四尺度：摩擦 0.8 Phase 1／Phase 2

日期：2026-11-01。依使用者指定，四個原生尺度 0.06／0.08／0.10／0.12 全部抽樣，摩擦改為 0.8，兩個 phase 使用相同 752k 新預測及完整對齊初始化。

## 結果

每尺度 2 個不同物件、各 5 筆 grasp，因此每尺度 10 筆、總共 40 筆，兩階段共 80 episodes。Phase 1 無 Ada controller（op）；Phase 2 使用 Ada controller（ours）。

| 尺度 | Phase 1 成功 | 失敗 | 初始拒絕 | Phase 2 成功 | 失敗 | 求解器退化 | 初始拒絕 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.06 | 4/10 | 6 | 0 | 8/10 | 0 | 2 | 0 |
| 0.08 | 10/10 | 0 | 0 | 10/10 | 0 | 0 | 0 |
| 0.10 | 8/10 | 1 | 1 | 8/10 | 0 | 1 | 1 |
| 0.12 | 4/10 | 6 | 0 | 1/10 | 3 | 6 | 0 |
| **合計** | **26/40** | **13** | **1** | **27/40** | **3** | **9** | **1** |

成功／全部選定輸入為 65% 與 67.5%。Repo 的 success/(success+failure) 另為 Phase 1 的 26/39=66.67%、Phase 2 的 27/30=90%；後者排除了 9 退化與 1 初始拒絕，因此不能只用 90% 描述整批表現。兩階段都無 execution_error、preparation_error。

control_eval／control_stat 退出碼：Phase 1 為 0／0；Phase 2 為 1／1，因求解器退化。所有選定樣本都有結果，未挑除或重跑失敗案例。

## 抽樣與固定設定

原先規劃每尺度 3 物件；盤點後 0.06、0.10 在原生 test split 各只有 2 個不同物件，故在推論前改成每尺度 2 物件，以保留均衡樣本數與 test split 限制。

資產取自 `examples/assets/object/DGN_2k`。先按 0.06 → 0.08 → 0.10 → 0.12 的順序，從符合原 DexLearn test split 的物件中，以 seed 42 抽樣；跨尺度不重複物件，每物件取一個原生 scene。清單在推論前凍結。這是有固定篩選順序的小型測試，不代表整個資產庫的均勻隨機抽樣。

保留原生 mesh、scale、pose；從各物件自己的 simplified mesh 抽 16,384 個完整點雲點，不預先縮放。DexLearn loader 按 scene 只套用一次 scale／pose，使用原設定的 1024 點輸入。checkpoint 為 `right_full_752k_v1/ckpts/step_050000.pth`；每場景 100 proposals，保留 per-type top-k 5。

固定條件：pregrasp 平移／手指 ×2、quaternion 不變；初始穿透門檻 10 mm、停用初始力拒絕；grasp／squeeze input 不變且無額外 squeeze 外推；摩擦 `[0.8,0.02]`、物體質量 0.1 kg、offset 0、Ada seed 12、單 worker。沿用原生 dummy-arm/no_tendon、控制器政策、lift 與成功判準，沒有更改公開 defaults。

| 樣本 | 物件 | 尺度 | mesh 局部尺寸 cm | Phase 1 成功 | Phase 2 成功 |
| --- | --- | ---: | --- | ---: | ---: |
| g0000–g0004 | `core_camera_6d036fd1c70e5a5849493d905c02fa86` | 0.06 | 8.1 × 5.4 × 7.3 | 2/5 | 5/5 |
| g0005–g0009 | `core_jar_275cc21fad6aa1b6bfa78a7d067c469c` | 0.10 | 11.0 × 13.1 × 11.0 | 3/5 | 3/5 |
| g0010–g0014 | `core_mug_c0c130c04edabc657c2b66248f91b3d8` | 0.08 | 8.1 × 9.6 × 10.2 | 5/5 | 5/5 |
| g0015–g0019 | `core_pillow_d777679b4dd35c76a4136492f17b9a59` | 0.12 | 16.9 × 8.2 × 15.3 | 1/5 | 0/5 |
| g0020–g0024 | `ddg_gd_tape_poisson_000` | 0.10 | 13.9 × 13.9 × 5.0 | 5/5 | 5/5 |
| g0025–g0029 | `mujoco_Cole_Hardware_Flower_Pot_1025` | 0.12 | 14.7 × 14.5 × 12.8 | 3/5 | 1/5 |
| g0030–g0034 | `mujoco_Lego_Friends_Advent_Calendar` | 0.06 | 9.9 × 2.0 × 6.8 | 2/5 | 3/5 |
| g0035–g0039 | `sem_TapeDispenser_844516b6fe304dafe36f6f37ca7926` | 0.08 | 13.0 × 5.3 × 8.3 | 5/5 | 5/5 |

## 逐筆變化與異常

配對結果為：23 筆兩邊成功、4 筆 Phase 1 失敗 → Phase 2 成功、6 筆失敗 → 求解器退化、3 筆成功 → 求解器退化、3 筆兩邊失敗、1 筆兩邊初始拒絕。Phase 2 嚴格成功數淨增 1 筆。

初始拒絕 **g0007** 是 scale 0.10 的 jar，初態手物穿透 **16.36 mm**，超過既定 10 mm；自碰撞約 0.05 mm。即使已關閉初始力 gate，仍應因深度門檻拒絕。沒有為此改門檻。

Phase 2 退化樣本：g0006、g0016、g0017、g0018、g0025、g0027、g0028、g0030、g0031。共 23 次不被接受的 solve，16 次 `Iteration limit reached`、7 次 `Positive directional derivative for linesearch`；g0030 單筆包含 15 次。沿用原生 apply_candidate policy，記錄中的可套用候選仍由既定策略執行，episode 保留 solver_degraded 分類，不計作成功。完整診斷見各筆 metrics 與 validation.json。

0.12 這組退化較多，值得後續檢查接觸與求解器；但**各尺度用的是不同物件與場景，不能把差異單獨歸因於尺度**，也不能用這 40 筆與上一批摩擦 0.6 的 50 筆直接估計摩擦改善量。這次重新抽樣及推論，g 編號不沿用先前的候選身份。

## 驗證與預覽

40 對初始 qpos／qvel／ctrl／object_pose、sample seed、prepared hash、三階段 targets 通過配對檢查；80 個回放的 qpos／ctrl 均有限且有 terminal frame，編譯後 collision geom 滑動摩擦均為 0.8。兩 phase resolved config 除輸出路徑外只差 method。

原生資產與 scene hashes 通過；120 個 FK targets 最大 palm 位置誤差 0.384 mm、旋轉誤差 0.008 rad 以下；碰撞 mesh 世界 bounds 最大誤差 3.45e-9 m。實驗／驗證腳本通過 Ruff、py_compile 與 diff whitespace 檢查。

預覽 <http://localhost:8081> 已切換本次四尺度資料。透過 Phase 選單切換 phase1／phase2，使用 Recorded replay 看實際執行，Target poses 僅為靜態目標。初始拒絕與退化均保留原分類。

輸出根目錄：`output/shadow-phases/ada-assets-four-scales-friction08-v1/`。

- `scene_manifest.json`、`sampling_command.json`、`sampling_exit.json`：推論前清單、來源 hashes、模型與命令。
- `raw_suite/`、`matched_suite/`：原始準備資料與完整初始化衍生輸入。
- `comparison.json`、`scale_results.json`：配對分類與逐尺度結果。
- `protocol.json`、`exit_codes.json`、`config_comparison.json`：實驗設定與退出狀態。
- `validation.json`：來源、FK、尺寸、配對與退化摘要；其中 per_object 計數是 Phase 1。
- `run/phase{1,2}/`：原生 eval/stat reports、resolved manifests、控制記錄、XML、replay 與逐筆 metrics。

工作區腳本：`reviews/ada_shadow_phases_20261101/run_ada_assets_four_scales.py`、`validate_ada_assets_four_scales.py`。先前實驗結果均保留。
