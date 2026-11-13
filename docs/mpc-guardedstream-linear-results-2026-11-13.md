# MPC active 線性傳送層重構 — 2026-11-13

已將 MPC active 的 10 Hz → 100 Hz 傳送改為 100 ms 線性插值，離線與 ROS mock 驗證通過。沒有實機操作、driver 啟動或 MuJoCo 抓取；沒有 commit／push。

**實機 commissioning 尚有 blocker**：目前下游 CRX 軟體的關節位置限制與整個 Ada 程序消失後的停止保障未完整確認。不能將本次測試解讀為完整硬體保護已驗證，也不能將離線插值結果當作實機抓取／lift 成功。

任務：[guardedstream linear refactor](ada_grasp_guardedstream_linear_refactor_agent.md)。起始 HEAD：`6cb993327b5db4a53f58e16b631fc9adaca6e3b5`。開始時只有使用者新增的任務文件未追蹤，已保留。

## 實作與發布語義

`GuardedStream` 不再保存速度狀態、煞車距離、加速度 ramp、位置上下限、速度／單步 delta 限制。Ours 原本的 SLSQP `q_limits`、每 MPC step 的 delta bounds、solver 判定與 native completion／timeout 都沒有修改。

- 首次發布前，使用新鮮 measured 28 軸作為 seed；`q_last_published` 此時為 `None`，不冒稱已送出。
- 接收合法 proposal 時，以最後完整發布的 command 作 `q_start`。接收時間 `segment_started` 使用 monotonic clock，不以 proposal 計算時間倒推補跑。
- 每個 100 Hz 發布機會使用 `alpha = clamp((now - segment_started) / 0.1, 0, 1)`。同號數值從較近端點計算、跨零使用 convex sum，降低浮點誤差；只處理 alpha，不裁切關節角度。
- 在名目 100 ms 端點直接回傳原 target，確保精確到點；無新 target 時保持端點，proposal 年齡超過既有 0.25 s 就 fault。
- 中途換目標，從最後成功發布值重新開始 100 ms。未發布 sample、measured feedback 或舊目標都不作新起點。
- `tick()` 只產生 pending command，**不更新** `q_last_published`。`published()` 必須在 arm 與 hand 的兩個本地 `publish()` 都成功返回後才提交。
- 兩份 ROS message 在任何一次發布前都先建立，使用相同 ROS header timestamp、同一份 28 軸向量的 canonical 6＋22 切片。
- 任一 message 建立或 publish 失敗：記錄已返回成功的 `published_groups`、嘗試值與前一次完整 pair，latch fault，沿用既有 abort，不重試、不繼續下一 tick。
- 兩個 ROS topic 沒有原子送達或物理 ACK：`last_published` 只表示本地 API 返回成功，**不是 last_applied**。arm 發出而 hand 拋錯時，無法撤回 arm；會記錄 partial send 並停止。網路層的不對稱送達仍是限制。

保留 command ownership、canonical joint order、finite／shape 檢查、feedback 時序／freshness、proposal valid／deadline／freshness、50 ms command-rate stall、取消與 fault latching。native completion 使用 `check()` 確認執行狀態，不再為完成判定產生一個未發布 command。沒有失敗後自動切 baseline 或 lift。

## 修改檔案

| 檔案 | 用途 |
| --- | --- |
| `src/ada_grasp_ctrl/mpc/streaming.py` | 新的線性 GuardedStream、pending／published 協定與 operational guards |
| `src/ada_grasp_ctrl/mpc/baseline_stream.py` | 保存原 GuardedStream 的 bounded 行為供非 MPC 路徑使用 |
| `src/ada_grasp_ctrl/ros_execution/bridge.py` | 原 explicit-linear 路徑改 import 保留的 BaselineStream；其餘流程未改 |
| `src/ada_grasp_ctrl/mpc/live_execution.py` | active 使用線性 stream、paired publication，補 segment／publication／stop 記錄 |
| `src/ada_grasp_ctrl/ros_execution/mpc_stream.py` | mock 同步使用線性 stream；共用 paired publication helper |
| `src/ada_grasp_ctrl/mpc/offline.py` | 只有 active grasp/squeeze 使用新 stream；其餘保留 BaselineStream |
| `tests/test_mpc_linear_stream.py` | 常數目標、重設目標、上下限、finite extremes、ack／watchdog regression |
| `tests/test_mpc_phase2.py` | 原 bounded 測試留給 BaselineStream；active 測試改為 interpolation contract |
| `tests/test_mpc_p3.py` | 新 stream constructor 的 explicit hardware opt-in 測試 |
| `tests/test_mpc_p3_receiver.py` | 相同 baseline/passive bytes、paired mapping／timestamp、partial publish、fault 到 sender 後不再送下一 phase |

原 baseline class 的 AST 除類名外逐項相同。Ours controller、native wrapper、原 linear reference、Runtime、Model、joint contract、TactileAdapter 與 compatibility conversion 的起始／結束 SHA-256 相同。完整稽核在 `output/mpc-linear-refactor-20261113/initial.json`、`summary.json`。

## 離線與 mock 結果

| 檢查 | 結果 |
| --- | --- |
| Native Python MPC suite | 44 passed，3 ROS-only skips；新增 sender case 於 ROS Python 另跑 |
| P3 receiver／sender suite | 4 passed（包含多個故障 subtests） |
| 既有 ROS receiver／dual receiver | 9 passed；與修改前的 3 個 P3 receiver tests 合跑時共 12 passed |
| Tactile suite | 45 passed，3 ROS-dependent skips |
| Grasp pipeline suite | 16 passed |
| `test_pipeline*.py` | 28 passed |
| ROS transport contract suite | 11 passed |
| Python compile、變更檔 Ruff／format、git diff whitespace | passed |
| baseline／passive fixture | 相同輸入、相同 clock 下，序列化命令 bytes 完全一致；passive 例外不改動 baseline |
| retarget | 30／50／70 ms 換目標，忽略未送 sample 與不同 measured q，從最後發布值重新開始 |
| 故障 | invalid／missing／stale proposal、stale／reordered feedback、deadline、rate stall、owner collision／loss、cancel、arm／hand publish failure 均停止；無失敗後 lift |

原 native `should_stop`、timeout 與 15 N threshold 測試保留。新增的 sender 測試使用 synthetic action result，確認 receiver fault **一旦送達** sender，會保留錯誤文字並只送出 grasp、不送 squeeze／lift；不是對上次實機 action-result 遺失問題的修復或現場重現。

ROS mock 使用 localhost domain **199**，明確 `crx_mock:=true sharpa_backend:=mock`：

- 30 個 native proposals，288 組 arm／hand commands，左側 commands 為 0。
- MPC period p50/p99：100.354／100.769 ms。
- Publish period p50/p99：10.322／10.873 ms；實際為約 100 Hz 的非即時 Python 排程。
- duplicate owner 被拒絕；注入 60 ms stall 後停止；兩個 command publisher 計數均回到 0。
- 只關閉本次啟動的 mock launch。沒有啟動、停止或連接實機 driver。

## 上次無名指邊界案例

來源：[首次實機試驗報告](mpc-p3-first-hardware-trial-2026-11-12.md)，以及該次 `events.jsonl`／`model.urdf`。

`right_ring_MCP_AA` 的兩筆最後成功 command：`-0.34883730710778654`、`-0.3490208210570351 rad`；URDF 下限 `-0.3491`，合法 MPC target `-0.34909999999999997 rad`。

1. 維護中的 unit regression 使用這些已保存數值，確認從最後 command 到下限的 10 個 10 ms sample 全部留在 segment 區間、最後精確等於 target；對稱上限亦通過。
2. 額外將已記錄的 **33 個 MPC targets** 依事件送達順序，以 10 ms clock 重新取樣，尾端延長 110 ms 讓最後 segment 到點：產生 **334 個模擬完整 pair**，依當次 28 軸 URDF 檢查 **0 個位置越界**，最終 28 軸均精確等於最後 target。Ring 最小／最終值皆為 `-0.34909999999999997 rad`。

這是 counterfactual **target interpolation replay**：沒有重算 solver、沒有改寫 recorded feedback vectors、沒有硬體輸出。Feedback 使用 log event-delivery clock；由 `event.time - feedback_age` 無法精確重建原始 receipt（兩者取樣時點不同），所以沒有宣稱重現原始 source clock。也沒有猜測舊程式當時未保存的 rejected command。

證據：`recorded-target-replay.json`、`recorded-target-commands.jsonl`，可重跑的唯讀腳本為 `output/mpc-linear-refactor-20261113/replay_commands.py`。

## 下游保護：唯讀稽核與 commissioning blockers

稽核的是目前 sourced workspace 實際 import 的檔案；沒有向硬體查詢參數。檔案位置與 SHA-256 在 `downstream-sources.json`。

| 路徑 | 已確認的程式行為 | 未確認／限制 |
| --- | --- | --- |
| CRX `dual_crx_control/interpolation/node.py` | `joint_targets()` 驗證完整 6／12 軸 names、shape、finite；quintic 呼叫既有 derivative-limited planner | 該 validator 明確不檢查 joint position limits；feedback 只初始化，不是 runtime watchdog |
| CRX `interpolation/quintic.py` | 限制速度／加速度／jerk，可延長 horizon；結束後回傳 target | 沒有 joint-position interval 約束；沒有輸入斷流 target timeout，繼續完成／保持目標 |
| FANUC `fanuc_hardware_interface/src/hardware_interface.cpp:write()` | hw inactive 不寫；stream disconnect 回傳 ERROR；target 轉成角度再呼叫 `writeJointTarget()` | 此函式未顯示 position bounds；底層 client、controller-manager enforcement、韌體／實機限位保障未完整驗證 |
| Sharpa `hand_node.py` | canonical 22 軸、finite，MIT 不接受額外 velocity／effort；steady-clock command timeout | 本地 publish 成功不等於 hand backend 接受，subscriber 失敗會記錄 warning |
| Sharpa `sharpa_sdk_hand.py`＋`mit_limits.py` | MIT target 依載入模型限制裁切；measured limit 超過 configured tolerance 後鎖定；非 normal 拒絕一般 command | 保留原有下游行為，沒有在 Ada 插值層加入／模仿裁切；實際裝置保護與標定未現場驗證 |
| Sharpa MIT timeout | 取消 recovery／清除 command session 的既有狀態；不再發 target，保留 SDK session／feedback | 不是 motor disable，也不是保證物理立即停止；後續 takeover 按原 backend measured-pose 條件處理 |
| Ada Runtime execution interface | watchdog 檢查 arm／hand 唯一 publisher、receiver、controller、force limiter／hand state；abort 要求停用 arm controller | 保護依賴程序仍可執行 abort；整個程序突然消失的情境未證明有獨立 CRX stop |

`bimanual_manipulation/right_arm_hand_bridge.py` 是 Ada receiver 的 compatibility wrapper，不能另算一層獨立安全保護。

**Blockers**：CRX 下游 joint-position enforcement 尚未確認；quintic 本身沒有輸入斷流 timeout，整程序遺失後的停止保證未確認；arm／hand 無跨 topic 原子送達。原有 explicit hardware enable／operator confirmation 仍保留，但這些旗標不代表上述缺口已解決。本次不提供已完成實機資格驗證的宣稱。

## 重跑指令（離線／mock）

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl MUJOCO_GL=egl OMP_NUM_THREADS=4 \
ADA_RIGHT_FULL_PREVIEW="$PWD/output/regression/crx-sharpa-right-preview-x080-y070-config-v1" \
ADA_P3_PACKET="$PWD/output/grasp_pipeline/physical_baseline_20261112_194720/grouplive/export/targets.json" \
.conda/bin/python -m unittest discover -s tests -p 'test_mpc*.py' -q

PYTHONPATH=src .conda/bin/python output/mpc-linear-refactor-20261113/replay_commands.py
```

ROS receiver tests 不初始化 ROS context，也不建立 publisher：

```bash
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
/usr/bin/python3 -m unittest discover -s tests -p 'test_mpc_p3_receiver.py' -v
```

隔離 mock：在第一個終端設定 domain 並確認空閒後啟動；第二個終端 source 相同環境並設定相同 domain，再跑 script。

```bash
export ROS_DOMAIN_ID=199 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
/usr/bin/python3 -m ada_grasp_ctrl.pipeline.ros_probe --empty-domain
ros2 launch bimanual_manipulation single_arm_hand.launch.py \
  prefix:=right_ crx_mock:=true sharpa_backend:=mock rviz:=false viewer:=false
```

```bash
export ROS_DOMAIN_ID=199 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
/usr/bin/python3 script/test_mpc_ros_mock.py \
  --output-root "$PWD/output/mpc_linear_mock_$(date +%Y%m%d_%H%M%S)" \
  --model-report "$PWD/output/regression/crx-sharpa-right-preview-x080-y070-config-v1/preview_report.json" \
  --mpc-python "$PWD/.conda/bin/python" --duration 3
```

## 操作者交接

目前 active command 已改成無額外運動規劃的直線插值；原 baseline／passive 保留。新的 `mpc_command`／`mpc_publish` 記錄區分 `q_mpc`、`q_segment_start`、`q_100Hz`、`q_last_published`、`measured_q`、segment／sample／publish 時間；失敗以 `mpc_publish_failed`／`mpc_stop` 記錄。

下一步是確認上述 CRX 下游保護缺口與實際工作站配置。這次依任務範圍只做離線及 mock，沒有物理試驗或自動重啟，也沒有放寬 Ours 或 driver 的限制。
