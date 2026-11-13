# P3 實機 native completion 與 lift 銜接失敗報告 — 2026-11-13

## 結論

操作者使用新的 MPC 線性傳送架構執行一次實機夾取，回報已抓住物體但沒有抬升。紀錄確認 **Ours 已達成原始 native completion 條件，receiver 與 sender 均成功完成 Grasp＋Squeeze action 的交接回報**。

沒有抬升的直接原因是：sender 在送出 lift 前，把當下 measured joints 與原預覽的固定 squeeze endpoint 比較；MPC 完成後的手臂姿態與該 endpoint 不同，觸發既有 `check_start()`。因此 **lift goal 尚未送出**，整段流程以 execution_error 結束。

本次沒有出現上次的 `Joint-limit violation`。目前問題在 **MPC 完成姿態 → 原固定 lift 軌跡** 的銜接，不是 native completion 未達成，也不是 sender 沒收到 action result。

使用者觀察到夾住物體，與程式達成 native completion，是兩種不同證據；本次未完成 lift，仍不能宣稱物體抬升成功或完整抓取流程成功。

## 執行與證據路徑

Repository：`/home/hao-fang/hugs-workspace/ada-grasp-ctrl`。

- 本次執行：`output/mpc_p3/active_20261113_111440/`
- 本次重新擷取／選定候選：`output/grasp_pipeline/p3_plan_20261113_111347/`
- 候選：`grouplive_c0003`／`c0003`
- 環境：hardware，ROS domain 87
- [Sender run_report.json](../output/mpc_p3/active_20261113_111440/sender/run_report.json)
- [Sender events.jsonl](../output/mpc_p3/active_20261113_111440/sender/events.jsonl)
- [Receiver 主紀錄](../output/mpc_p3/active_20261113_111440/receiver.log)
- [Grasp action result.json](../output/mpc_p3/active_20261113_111440/receiver/20261113T191508_497475Z_478626/result.json)
- [Grasp proposals／commands／native_complete 紀錄](../output/mpc_p3/active_20261113_111440/receiver/20261113T191508_497475Z_478626/events.jsonl)
- [當次 motion_plan.json](../output/grasp_pipeline/p3_plan_20261113_111347/grouplive/export/motion_plan.json)

當次 receiver action 目錄亦保留 `model.urdf`、`config.json`、`versions.json` 與 `native-worker.log`。程式目前有尚未提交的線性傳送重構；實際執行版本應連同當次 artifacts 與工作目錄內容判讀，不僅依賴 git HEAD。

## 實際流程

| 階段 | 結果 |
| --- | --- |
| above_object | action 成功；沿用原 arm 到位策略 |
| pregrasp | action 成功；手指位置誤差依既有設定不阻擋到位 |
| Grasp＋Squeeze | native_complete，action 成功，sender 已收到結果 |
| 獨立 squeeze action | 依 active 合併 episode 設計跳過 |
| lift | sender 起始姿態檢查失敗，沒有送出 goal |
| hold／release／return | 沒有送出 |

Receiver 回傳文字：

```text
Ours native_complete: grasp+squeeze episode; no automatic next motion
```

其中「no automatic next motion」表示 receiver 本身不自行產生下一個 action；後續動作由 sender 排程。這次不是因為該文字而不抬升，而是 sender 的下一階段檢查失敗。

Sender 最終結果：

```text
status: execution_error
reason: Robot moved since preview: right_J1; rerun selection and preview
```

`task_completed=false`，`object_lifted=null`，`object_held=null`。

## MPC 完成證據

| 欄位 | 值 |
| --- | ---: |
| `mpc_proposal` 紀錄數 | 49（包括 terminal completion 回報，不應全部算成可發布 command） |
| `mpc_command` ticks | 464 組 arm／hand 命令 |
| `native_complete` 事件 | 1 |
| 完成時 index | 48 |
| native reference 長度 | 44 |
| native max_steps | 88 |
| 完成時 native stage | 2 |
| 完成時正常力總和 | 16.37576318949169 N |
| 原始 final_sum_force | 15.0 N |
| `should_stop` | true |

原始條件為 `index >= len(qpos_path)` 且正常力總和 `>15 N`。這次兩者皆滿足。Receiver 的 Grasp `result.json` 為 `final_state=DONE`、`task_completed=true`、`termination=native_complete`，沒有 abort_reason。

這裡 action 的 `task_completed=true` 指 Grasp＋Squeeze episode；與 sender 整段流程的 `task_completed=false` 並不矛盾。

## lift 為何被擋住

目前 sender 在每個 phase 前呼叫：

```python
check_start(start, plan["keyframes"][index], ...)
```

來源：`src/ada_grasp_ctrl/experiments/right_ros_sender.py` 的 `execute()`。
`src/ada_grasp_ctrl/pipeline/plan.py` 的 `check_start()` 對六軸手臂使用固定 **0.01 rad** 容差，按 canonical joint order 檢查；遇到第一個超差就報錯。

對 lift 而言，`plan["keyframes"][index]` 是原預覽中的 squeeze endpoint。MPC 在抓取過程中可以調整手臂姿態，因此完成姿態不保證回到該固定 endpoint。

以下比較使用 **receiver native completion 保存的 final_positions** 與 **原計畫 lift 起點**；它說明此次不匹配的量級。Sender 檢查瞬間的完整 measured vector 沒有另外保存，因此不宣稱這些數字就是 sender 拋錯瞬間的逐軸數值。

| 手臂關節 | 原計畫 lift 起點 rad | MPC 完成量測 rad | 差值 rad | 差值 ° |
| --- | ---: | ---: | ---: | ---: |
| right_J1 | -0.993767500 | -1.043671126 | -0.049903627 | -2.8593 |
| right_J2 | -1.006917822 | -0.993003288 | +0.013914534 | +0.7972 |
| right_J3 | 3.484720335 | 3.469827258 | -0.014893077 | -0.8533 |
| right_J4 | -0.421784323 | -0.346057610 | +0.075726713 | +4.3388 |
| right_J5 | -1.133291794 | -1.075276657 | +0.058015137 | +3.3240 |
| right_J6 | -0.455432540 | -0.465424618 | -0.009992078 | -0.5725 |

J1 是第一個超過 0.01 rad 的軸，所以錯誤顯示 J1；不是只有 J1 不匹配。依完成快照，J1–J5 均超差。

這次設定 `hand_arrival_required=false`，手指起點誤差不作此階段的阻擋條件。直接觸發的是手臂起點檢查。

「Robot moved since preview」是泛用錯誤訊息；此處與 MPC 自身調整姿態相符，**不能據此推論操作者碰動了機器人或物體**。

## 原 lift 仍綁定預覽 squeeze 姿態

當次 plan 的 lift 設定為 world +Z 0.05 m，維持原 squeeze FK 的末端方向，手部使用 `retain_selected_squeeze`。lift 的絕對關節目標已在預覽階段由 IK 算好，並非依 MPC 完成量測重新計算。

因此不應只放寬／刪除 `check_start()` 後直接送舊 lift target：這樣會從新的實際姿態移向舊的絕對目標，且可能把 MPC 建立的手指姿態拉回 selected squeeze。原有 receiver 在接收下一個 reviewed goal 時也保留起點檢查；只改 sender 並不能完整完成銜接。

當次 plan 記錄 `collision_checked=false`、`path_checked=false`；本報告不把既有 lift target 解讀為從任意 MPC endpoint 出發都已驗證的路徑。

## 與前一次失敗比較

| 項目 | 2026-11-12 首次試驗 | 2026-11-13 本次試驗 |
| --- | --- | --- |
| MPC 傳送層 | 舊 bounded／braking stream | 新線性 stream |
| native completion | 未達成 | 已達成，且 sender 收到成功結果 |
| 直接停止點 | Ada streaming joint-limit fault | sender 的 lift 起始姿態檢查 |
| lift goal | 未送出 | 未送出 |
| 可確認的進展 | MPC 已接管並傳送命令 | MPC episode 正常完成；後續 handoff 待處理 |

本次沒有再觸發舊錯誤，不表示所有下游保護或所有軌跡都已驗證；線性重構報告所列的 commissioning 缺口仍須獨立處理。

## 後續修改方向（尚未實作）

需明確定義 **native completion 後的 lift handoff**：

1. 以完成後的新鮮實際姿態為 lift 起點，避免要求 MPC 回到原 squeeze endpoint。
2. 保留完成時的抓取手形；區分最後發布的手指 target 與 measured hand joints，明確選定後續保持策略，不能悄悄改回 selected squeeze。
3. 若維持原本抬升意圖，從該實際姿態的 FK 出發，維持完成時方向，計算 world +Z 5 cm 的 lift 目標。
4. 同步更新 sender／receiver 的 reviewed handoff 資料與檢查範圍；保留 fresh feedback、native success、IK／關節與既有下游保護，不把任意 phase-start mismatch 全部忽略。
5. 先用這次保存的 final_positions 做離線 handoff regression，確認新起點、lift 方向、手形保持及失敗時不 lift；不需重跑 MuJoCo 抓取。

以上是待討論／實作方向，**不是已授權或已完成的現場補發 lift**。本次只讀取紀錄並寫報告，未修改控制程式、未送出動作、未 commit／push，也未查詢目前實機狀態。

相關文件：[首次實機試驗](mpc-p3-first-hardware-trial-2026-11-12.md)、[線性傳送層重構](mpc-guardedstream-linear-results-2026-11-13.md)。
