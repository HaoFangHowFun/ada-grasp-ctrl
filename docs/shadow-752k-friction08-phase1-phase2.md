# Shadow 752k：摩擦 0.8 的 Phase 1／Phase 2 配對測試

日期：2026-11-01。依使用者授權，對既有同一批 50 筆 grasp 執行摩擦 0.8 對照，兩個 phase 都使用完整對齊初始化。模型為 `right_full_752k_v1/ckpts/step_050000.pth`，10 個 scene、尺度 0.060／0.081／0.106，沒有重新推論或挑選樣本。

## 結果

| 設定 | 成功 | 失敗 | 求解器退化 | 初始拒絕／執行錯誤 | 成功／全部 50 筆 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 既有 Phase 1，摩擦 0.6 | 34 | 16 | 0 | 0／0 | 68% |
| 本次 Phase 1，摩擦 0.8，無 Ada controller | **43** | **7** | **0** | **0／0** | **86%** |
| 本次 Phase 2，摩擦 0.8，有 Ada controller | **44** | **4** | **2** | **0／0** | **88%** |

兩階段各完成 50 筆，10 個 scene 都至少有一筆成功。Phase 1 的 control_eval／control_stat 退出碼為 0／0；Phase 2 為 1／1，原因是兩筆 solver_degraded，不是整批沒跑完。

Repo 正式 `success/(success+failure)` 分母會排除退化案例，因此 Phase 2 為 44/48 = 91.67%；本表同時保留全部 50 筆作分母，避免隱藏退化。不能把 Phase 2 說成「只有 4 筆沒成功」。

## 摩擦的影響

Phase 1 只將 `task.miu_coef` 從 `[0.6, 0.02]` 改成 `[0.8, 0.02]`。除輸出路徑外，resolved config 沒有其他變更。原本 34 筆成功全部保留，9 筆失敗變成功，沒有成功變失敗。

恢復成功的樣本：g0003、g0016、g0018、g0019、g0027、g0029、g0043、g0044、g0046。

前次「benchmark 成功、Ada 失敗」的 8 筆中，**7 筆只靠此摩擦變更就恢復成功**，僅 g0042 仍失敗。這提供了單因子證據：在這批固定樣本與 Ada 配置下，摩擦差異是先前差距的重要因素。先前觀察到的 tendon、質量與時序差異仍存在，但不能再把那 8 筆主要失敗直接歸因於 tendon。

Phase 1 的 43/50 與 benchmark 的 42/50 仍是不同動態／成功判準下的結果，不表示已建立全面優於 benchmark 的結論。

## Ada controller 的逐筆影響

| Phase 1 → Phase 2 | 數量 | 樣本 |
| --- | ---: | --- |
| 成功 → 成功 | 39 | 見 comparison.json |
| 失敗 → 成功 | 5 | g0004、g0037、g0040、g0041、g0042 |
| 成功 → 失敗 | 3 | g0016、g0027、g0043 |
| 成功 → 求解器退化 | 1 | g0018 |
| 失敗 → 求解器退化 | 1 | g0025 |
| 失敗 → 失敗 | 1 | g0030 |

因此 controller 在本批救回 5 筆，但有 3 筆成功退步和另外 2 筆求解退化，最終嚴格計入的成功數僅淨增 1 筆。這是零物體位姿擾動的測試，不能據此判斷 controller 在位姿不確定性下的效益。本次沒有摩擦 0.6 的完整初始化 Phase 2 對照，不能估計摩擦對 Phase 2 的獨立貢獻。

## 兩筆求解器退化

- **g0018**：81 筆 solver diagnostics 中有 7 次不被接受，全部是 `Iteration limit reached`（200 iterations）。物體最後回到桌面附近。
- **g0025**：139 筆 diagnostics 中有 3 次不被接受，全部是 `Positive directional derivative for linesearch`。物體最後升高約 205 mm，但不能因抬起來就將退化案例改列成功。

沿用原生 `apply_candidate` failure policy：這 10 次都有有限且可套用的 candidate，原策略仍套用了候選指令並將整個 episode 標為 solver_degraded。沒有調鬆求解接受門檻、改 failure policy、忽略退化或重跑挑選成功結果。逐次殘差、狀態與決策保留於各筆 metrics／原生記錄。

## 固定條件與驗證

- pregrasp 平移與手指外推 ×2，quaternion 保持原值；直接沿用前次衍生 suite，不重複外推。
- 初始穿透門檻 10 mm、關閉初始 force gate。
- grasp／squeeze input 不變，沒有額外 squeeze 外推。Phase 2 仍按原生 adaptive controller 調整執行命令。
- 物體質量 0.1 kg、原生 dummy arm／no_tendon、抬升 0.2 m、原生控制／成功定義保留。
- seed 12、單 worker、offset 0；兩階段 method 分別為 `op`、`ours`。
- 50 對初始 qpos／qvel／ctrl／object_pose、sample seed、prepared hash 與三個 targets 均通過配對檢查。
- 100 筆 replay 的 qpos／ctrl 均有限，均有 terminal frame；100 個編譯後模型的 collision geom 滑動摩擦確認為 0.8。
- 設定稽核確認兩個 phase 除輸出路徑外只差 `task.method`。
- 實驗與回放腳本通過 Ruff、py_compile；未修改 repo controller 實作或公開預設。

## 預覽與產物

預覽：<http://localhost:8081>。使用 **Phase** 選單切換 `phase1`／`phase2`，同一個 sample ID 會載入該 phase 的原生回放。選擇 **Recorded replay**、按 **Play replay**；重新播放時把 progress 拉回 0。`Target poses` 是靜態目標，不能當成已達成的狀態。退化狀態會原樣顯示。

結果根目錄：`output/shadow-phases/dexlearn-752k-three-scales-v1/friction08-phases-v1/`。

- `protocol.json`／`suite_manifest.json`：設定範圍、來源與 hash。
- `comparison.json`：兩階段統計與逐筆 cross table。
- `validation.json`：配對驗證與相對摩擦 0.6 基準的轉移。
- `audit.json`：模型摩擦、回放驗證、設定差異、求解退化摘要。
- `exit_codes.json`：原生任務退出碼。
- `phase{1,2}/phase_results.json`：逐筆完整 metrics。
- 各 phase 的 `log/` 與 `control/`：resolved manifests、run reports、failures、控制 NPY、scene XML、replay NPZ。

工作區腳本：`reviews/ada_shadow_phases_20261101/run_friction08_phases.py`；預覽腳本：同目錄 `preview_paired_phases.py`。既有摩擦 0.6 與 benchmark 結果均保留。
