# Sharpa／Shadow：完整 Phase 1／Phase 2 評估紀錄

整理日期：2026-11-09。兩種手各完成 1,000 筆輸入 × 2 phases，共 4,000 episodes。Phase 1 是 `op` 固定位置控制；Phase 2 是 `ours` 接觸回饋自適應控制。

本紀錄同時保留原始分類與使用者指定的重算方式：**排除初始化拒絕，並將所有 solver degradation 計入成功**。重算僅用於本表，不修改原始 NPY／JSON、控制器、原生成功判定或退出碼，也沒有更新 golden。

## 執行來源與完整性

| 項目 | Sharpa | Shadow |
| --- | --- | --- |
| 模型 | Sharpa right 100K，`step_100000.pth` | Shadow right_full 752k，`step_050000.pth` |
| 物體／scenes／候選 | 25／100／1,000 | 89／100／1,000 |
| 尺度 | 0.060、0.081、0.106、0.133 | 0.06、0.08、0.10、0.12 |
| CPU workers | 1 | 8 |
| Phase 1 eval／stat 退出碼 | 0／0 | 0／0 |
| Phase 2 eval／stat 退出碼 | 1／1 | 1／1 |
| Execution errors | 0 | 0 |
| 驗證相同初態／targets／seed 的配對數 | 1,000 | 1,000 |

來源目錄（相對 repo）：

```text
output/regression/sharpa-runs/20261109T070435458523Z-1656791/
output/regression/runs/20261109T073609Z-1710221/
```

每個 phase 都有 1,000 個唯一 sample IDs、1,000 個控制輸出與對應 replay，未發現缺失。Phase 2 非零退出碼來自 solver degradation，並非批次未完成。提交前比對兩個 run 的 `code_state.json`：其中記錄的程式／設定檔 hash 均與目前工作樹一致。

兩套均採零物體位姿擾動、摩擦 `[0.8,0.02]`、固定物體質量 0.1 kg、pregrasp 外推 ×2、無額外 squeeze、初始穿透門檻 10 mm，停用初始力拒絕。兩種手的物體、尺度與來源分布不同；本紀錄支持各手內的 Phase 1／2 比較，不是手型優劣的公平比較。Sharpa 15 N 力目標仍是未校正的實驗設定，訓練 split 與 checkpoint 的完整來源關係尚待補證。

## 使用者指定的重算表

計算方式：

```text
分母 = 全部 1,000 筆 − invalid_initialization
Phase 1 分子 = success
Phase 2 分子 = success + solver_degraded
```

| 手型 | 排除初始化拒絕 | 有效分母 | Phase 1：op | Phase 2：ours，含 degradation | 提升 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sharpa | 62 | 938 | 744／938 = **79.32%** | 800／938 = **85.29%** | **+5.97 個百分點** |
| Shadow | 17 | 983 | 799／983 = **81.28%** | 889／983 = **90.44%** | **+9.16 個百分點** |

此處把 Sharpa 的 61 筆、Shadow 的 120 筆退化全部視為成功，尚未逐筆核驗其實際抬升結果。因此這是指定分類規則下的比例，不應標示為所有列入案例皆已驗證物理抓取成功。

## 原始分類

| 手型／phase | Success | Failure | Solver degraded | Invalid initialization | Success／全部 1,000 筆 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sharpa Phase 1 | 744 | 194 | 0 | 62 | 74.4% |
| Sharpa Phase 2 | 739 | 138 | 61 | 62 | 73.9% |
| Shadow Phase 1 | 799 | 184 | 0 | 17 | 79.9% |
| Shadow Phase 2 | 769 | 94 | 120 | 17 | 76.9% |

原生 `success/(success+failure)` 另為 Sharpa 79.32% → 84.26%、Shadow 81.28% → 89.11%；此分母同時排除了 invalid 與 degraded，與上方指定重算表不同。

## 逐筆配對變化

| Phase 1 → Phase 2 | Sharpa | Shadow |
| --- | ---: | ---: |
| 成功 → 成功 | 669 | 719 |
| 失敗 → 成功 | 70 | 50 |
| 成功 → 失敗 | 33 | 21 |
| 成功 → 退化 | 42 | 59 |
| 失敗 → 退化 | 19 | 61 |
| 失敗 → 失敗 | 105 | 73 |
| 初始化拒絕 → 初始化拒絕 | 62 | 17 |

按原始分類，嚴格成功數分別淨減 5、30；按指定重算方式，列入成功的數量分別淨增 56、90。這兩個結論使用不同統計定義，需與各自分母一併保留。

## 退化時實際如何執行

兩次保存的 `phase2/log/control_eval/run_manifest.yaml` 都明確使用 `solver_failure_policy=apply_candidate`：

1. 某個控制求解未通過接受條件時，若 candidate 維度正確且全部為有限值，仍施加 `current_qpos_a + delta_qpos_a` 位置目標。這不保證 candidate 滿足所有最佳化約束。
2. 下一控制步繼續取回饋與求解，不自動切成 `op`；正常走完控制後繼續原生 0.2 m lift。
3. 若 candidate 缺失、維度錯誤或含 NaN／Inf，則中止該 episode 並跳過 lift。
4. 任一求解診斷被拒絕後，episode 的退化標記會保留，不因後續求解恢復或物體抬起而清除。

因此「solver degraded」描述求解過程，而非直接等同 lift outcome。原始分類與 diagnostics 保留，可供後續分析實際抬升和 solver 原因。

初始化拒絕則發生在控制前，本次指手物或手部自碰撞穿透超過 10 mm。兩個 phase 使用相同初態，拒絕數相同。

## 後續分析線索

最大尺度集中了較多退化：Sharpa scale 0.133 有 30／61 筆；Shadow scale 0.12 有 66／120 筆。後續可優先區分未收斂、約束殘差與實際抬升 outcome，保留原始候選不重選。這是描述性分組，尚不能單獨歸因於尺度。

## 來源摘要 SHA-256

以下 hashes 用來識別本次結果；大型原始輸出保留在本機 `output/`，不加入版本控制。

```text
Sharpa comparison.json
870d075f179319864c0f8782c3833b2bc641636316daa6726f67ad383b5876d7
Sharpa experiment_manifest.json
3e7a5c7fc87b42513c92850dd3e5e57415dbd2c37f1d4bbb411ebf555fd83a76
Shadow comparison.json
bce809e2eacbc33b83eb5a9724f8bfd9c8b5bc5c4cc03e8b65f223d44c435f6d
Shadow experiment_manifest.json
d4a4a5658f59143adeb629b5caf506e863244587c657f8b92f0f335e42b0fe52
```

操作說明：[Sharpa controller regression](sharpa-controller-regression.md)、[Shadow controller regression](shadow-controller-regression.md)。既有驗證包含 149 個 unit tests、兩輪 Sharpa smoke、全專案 Ruff check／compileall 與變更檔案格式檢查通過；完整 release gate 未宣稱通過。
