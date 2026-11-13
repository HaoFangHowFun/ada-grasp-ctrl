# P3 首次實機 MPC 執行紀錄與交接 — 2026-11-12

## 結論

操作者自行啟動這次實機執行，回報「已抓住，但沒有抬升」。紀錄確認 MPC 已接管 Grasp 並送出命令，但在 native completion 前，Ada 的 bounded streaming 層拋出 `Joint-limit violation`，receiver 中止，因此沒有送出 lift。

這次可確認 **實機 MPC 接線與命令傳送已運作**；不能宣稱完成 MPC grasp、穩定持物或成功抬升。使用者觀察到抓住物體與程式判定完成是不同證據。

本次收尾只讀取既有紀錄並寫報告；沒有修改控制程式、重跑測試、啟動硬體或送出機器人命令。先保留現況，待下一次明確指示再處理。

## 執行來源與證據

Repository：`/home/hao-fang/hugs-workspace/ada-grasp-ctrl`。

- 實機執行目錄：`output/mpc_p3/active_20261112_221050/`
- 本次重新擷取／選定候選的 packet：`output/grasp_pipeline/p3_plan_20261112_221009/grouplive/export/targets.json`
- Sender 結果：[run_report.json](../output/mpc_p3/active_20261112_221050/sender/run_report.json)
- Sender 事件：[events.jsonl](../output/mpc_p3/active_20261112_221050/sender/events.jsonl)
- Receiver 主紀錄：[receiver.log](../output/mpc_p3/active_20261112_221050/receiver.log)
- Grasp action 結果：[result.json](../output/mpc_p3/active_20261112_221050/receiver/20261113T061118_607537Z_444514/result.json)
- Grasp 詳細命令與 proposal：[events.jsonl](../output/mpc_p3/active_20261112_221050/receiver/20261113T061118_607537Z_444514/events.jsonl)
- 同一 action 的 `model.urdf`、`config.json`、`versions.json`、`native-worker.log` 保留模型、設定、當次來源雜湊與 worker 記錄。

工作目錄含先前大量未提交修改；本次未 commit／push，也未清理或覆寫原始執行證據。

## 實際階段與數據

| 項目 | 紀錄 |
| --- | --- |
| 環境 | hardware，ROS domain 87 |
| above_object | 成功；沿用原本 arm 到位、手指誤差不阻擋的策略 |
| pregrasp | 成功；同上 |
| grasp | MPC 接管後 fault |
| 有效 MPC proposals | 33／33，index 0–32 |
| 成對 arm／hand command ticks | 313 |
| 第一至最後 command 紀錄間隔 | 3.22525 s |
| 紀錄中的 native stage | 全部為 Stage 2 |
| 量測正常力總和範圍 | 3.7742–15.2942 N |
| 最後一筆正常力總和 | 11.6095 N |
| native reference 長度／budget | 44／88 steps |
| 最後 proposal 的 should_stop | false |
| native completion | 0 次 |
| lift／hold／release／return | 沒有送出 |

原始完成條件仍是 `index >= len(qpos_path)` 且正常力總和 `>15 N`。即使過程中短暫超過 15 N，最後 index 仍只有 32，尚未完成 44-step reference；這次也沒有走到 88-step native timeout。直接中止原因是 joint-limit guard。

## 錯誤由誰發出

錯誤字串來自 **Ada 自己的 MPC streaming 層**：

- `src/ada_grasp_ctrl/mpc/streaming.py`：`GuardedStream.check_limits()` 拋出 `ValueError("Joint-limit violation")`。
- `src/ada_grasp_ctrl/mpc/live_execution.py`：呼叫 guard 檢查 measured feedback、MPC proposal 與產生的 bounded command。
- `src/ada_grasp_ctrl/ros_execution/bridge.py`：接到例外後呼叫既有 `Runtime.abort()`，並以 ROS logger 印出錯誤。

因此，看到 ROS logger 的錯誤前綴不表示警報來自 ROS driver。這次記錄的 33 筆 MPC proposals 全為 valid，沒有 solver rejection。不能從此錯誤推論 CRX／Sharpa 實際關節已越過機械限制。

## 目前最可疑的位置：無名指 MCP_AA 的傳送指令

最後的傳送紀錄指向 `right_ring_MCP_AA`（右無名指 MCP 左右擺動）：

| 數值 | rad |
| --- | ---: |
| 當次 URDF 下限 | -0.3491 |
| MPC 目標 | -0.34909999999999997 |
| 倒數第二筆傳送指令 | -0.34883730710778654 |
| 最後一筆傳送指令 | -0.3490208210570351 |
| 最後一筆指令距下限 | 0.00007917894296494143 |

MPC 目標在合法下限上，最後成功送出的指令也仍合法。指令正向下限逼近，streaming 層又保留速度／加速度歷史，因此**下一個平滑步進超越下限**是目前最合理的待驗證假設。

但現在的例外只有通用字串，沒有記錄「是 measured feedback、proposal 還是新 command」、關節名稱與被拒絕數值。因此尚未完整定位，不能直接把上述假設當作已確認根因，也不能逕自放寬 joint limits 或裁切命令當作修復。

## 中止與錯誤回報狀態

Grasp 的 receiver `result.json`：

- `final_state = FAULT`
- `task_completed = false`
- `abort_reason = Joint-limit violation`
- `command_publishers_created = true`
- `controller_deactivated = true`：既有 abort 已記錄右手臂 controller 停用。
- 手部停止後續命令，保留最後 bounded target／既有 backend timeout policy；**不是關閉手部馬達**。
- `physical_stop_confirmed = false`：紀錄沒有確認實際物理停止狀態。

Sender 只記錄 above_object、pregrasp 兩個成功結果，最終是 `ROS discovery/result timeout`，取消 acknowledgment／result 都是 false。Receiver 另有 `Failed to send result response (the client may have gone away)`。

這代表 receiver 的具體 fault 沒有順利回傳到 sender；兩份紀錄應一起閱讀。回傳問題的確切原因尚未確認，不應把 sender timeout 誤認成 MPC 仍在持續等待完成。

上述是當次停止紀錄，不是本次收尾對目前硬體狀態的即時確認。

## 較早啟動失敗，與本次分開看

`output/mpc_p3/active_20261112_215738/` 曾在第一個動作前出現 `hand: stale/invalid timestamp`：sender 記錄手部 joint-state age 0.65614 s，超過既有 0.5 s 門檻，沒有送出 goal。

之後也曾因 shell 的 `PYTHONPATH` 寫成普通文字而出現 `ModuleNotFoundError`。這些都不是本次執行至 Grasp 後的直接中止原因；不宣稱先前 timestamp 問題已根治。

## 下次接續的最小工作

本次只記錄，不執行以下修改：

1. 在既有 guard 錯誤紀錄補上檢查來源、joint name、值與上下限，讓下一次能精確歸因。
2. 用這次已記錄的 targets／command history 離線重現靠近 `right_ring_MCP_AA` 下限的情況，確認是否為平滑步進越界；不需實機或 MuJoCo 抓取。
3. 若確認是 streaming 問題，只修傳送層的邊界處理，保留關節位置、速度、加速度限制與 Ours 原始數學／完成條件。
4. 另查 receiver fault／action result 為何沒有回到 sender，避免使用者只看到 timeout。
5. 修復後的實機動作由操作者另行啟動；物體或姿態改變時重新擷取並產生 packet。不能直接補發 lift 或把這次視為正常完成。

前期軟體與離線證據見 [P3 results](mpc-p3-results-2026-11-12.md)；當時「physical active not performed」是交付時的歷史狀態，本報告補充後續由操作者執行的首次實機試驗，結果仍為未完成。
