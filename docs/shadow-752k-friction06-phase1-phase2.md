# Shadow 752k：摩擦 0.6 的 Phase 1／Phase 2 與摩擦對照

日期：2026-11-01。這次補跑完整對齊初始化的摩擦 0.6 Phase 2，使用同一批 50 筆 752k 預測。Phase 1 沿用前次 `ada-benchmark-init-v1/matched/phase1` 已驗證結果，沒有重跑；來源路徑與結果 SHA256 記錄於 `reused_phase1.json`。

## 四組結果

| 摩擦 | 階段 | 成功 | 失敗 | 求解器退化 | 成功／全部 50 筆 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.6 | Phase 1，無 Ada controller | 34 | 16 | 0 | 68% |
| **0.6** | **Phase 2，有 Ada controller** | **39** | **9** | **2** | **78%** |
| 0.8 | Phase 1，無 Ada controller | 43 | 7 | 0 | 86% |
| 0.8 | Phase 2，有 Ada controller | 44 | 4 | 2 | 88% |

四組均無初始拒絕或 execution_error。摩擦 0.6 Phase 2 完成 50 筆；control_eval／control_stat 退出碼均為 1，原因為 solver_degraded。Repo 的有效樣本成功率另為 39/(39+9)=81.25%；不應以此分母隱藏兩筆退化。

## 摩擦 0.6 下 Ada controller 的效果

- 31 筆兩階段皆成功。
- 8 筆失敗 → 成功：g0003、g0004、g0027、g0029、g0037、g0040、g0041、g0046。
- 3 筆成功 → 失敗：g0017、g0023、g0031。
- 2 筆失敗 → 求解器退化：g0018、g0025。
- 6 筆兩階段皆失敗。

成功數淨增 5 筆（34 → 39），比摩擦 0.8 下的淨增 1 筆大。但這只是固定 50 筆、零位姿擾動的單次對照，不能推廣為 controller 在所有低摩擦條件下都較有效。

## Phase 2 的摩擦效應

從 0.6 改為 0.8，Phase 2 有 38 筆維持成功、3 筆維持失敗、2 筆維持求解器退化。

- 6 筆失敗 → 成功：g0017、g0019、g0023、g0031、g0042、g0044。
- 1 筆成功 → 失敗：g0027。

因此 Phase 2 成功數淨增 5 筆（39 → 44）；更高摩擦並不是每筆都單調改善。對照 Phase 1 則是 9 筆恢復成功，沒有成功退步。

## 求解器退化

兩個摩擦值退化的樣本均為 g0018、g0025，不能只把問題歸因於摩擦 0.8。

- g0018：摩擦 0.6 下共 6 次拒絕，4 次 `Iteration limit reached`、2 次 `Positive directional derivative for linesearch`；最終物體回到桌面附近。
- g0025：1 次 `Positive directional derivative for linesearch`；最終物體升高約 204.9 mm，但仍依規則分類為 solver_degraded，不算成功。

沿用原生 `apply_candidate` failure policy，沒有調整求解器接受門檻、修改策略或挑選重跑結果。摩擦 0.8 下對應拒絕次數為 7 與 3。

## 固定條件與驗證

完整對齊初始化為 pregrasp 平移／手指 ×2、quaternion 不變、穿透門檻 10 mm、關閉初始力拒絕。直接沿用已衍生 suite，沒有重複外推。grasp、squeeze input 保持不變，無額外 squeeze 加深。

相同物體、尺度、seed 12、單 worker、offset 0、物體質量 0.1 kg、原生 dummy-arm/no_tendon、lift 與成功判準。50 對初始狀態、seed、來源 hash、三個 targets 通過既有配對驗證；新 50 筆回放均有 terminal frame，qpos／ctrl 有限，編譯後碰撞幾何的滑動摩擦均確認為 0.6。

兩個摩擦值的 Phase 2 resolved config 比對確認，除輸出路徑外只差 `task.miu_coef=[0.6,0.02]` 與 `[0.8,0.02]`。沒有修改 repo 控制器或公開 defaults。新增腳本通過 Ruff 與 py_compile。

## 預覽與產物

- 摩擦 0.6：<http://localhost:8082>
- 摩擦 0.8：<http://localhost:8081>

兩者均可透過 Phase 選單切換 phase1／phase2，選 Recorded replay 查看實際動態。Target poses 為靜態輸入目標。

新結果位於 `output/shadow-phases/dexlearn-752k-three-scales-v1/friction06-phases-v1/`：

- `phase1`：指向已驗證舊 Phase 1 的 symlink；`reused_phase1.json` 明確記錄沿用來源。
- `phase2/`：本次新執行的 reports、resolved manifests、控制記錄、scene XML 與 replay。
- `comparison.json`、`validation.json`：摩擦 0.6 的兩階段配對結果。
- `friction_comparison.json`：四組逐筆分類、設定稽核、求解退化摘要。
- `protocol.json`、`suite_manifest.json`、`exit_codes.json`：實驗範圍、來源與原生退出碼。

工作區腳本：`reviews/ada_shadow_phases_20261101/run_friction06_phase2.py`、`compare_friction_phases.py`。
