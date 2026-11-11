# Ada ROS execution：右 CRX＋Sharpa 姿態傳送

本文描述原始三階段 prepared-data/mock 入口。當前的即時相機、七階段循環與明確實機模式，
請使用 [完整抓取入口](grasp-pipeline.md)。底層 receiver 仍不自行決定下一階段。

任務決策在 **ada-grasp-ctrl**。Ada 依序送出自己產生的 pregrasp、grasp、squeeze；
`ada_grasp_ctrl.ros_execution.bridge` 被動接收一個具名 28 軸目標，限速執行，回傳 feedback/result。
`bimanual_manipulation` 只負責組合模型與 driver launch；CRX driver 保留原有底層插值。
接收端不產生抓取姿態、不選擇下一階段、不自行開手或抬升。

```text
Ada 的 prepared.npy
  pregrasp_qpos / grasp_qpos / squeeze_qpos
         ↓ Ada exporter（Python 3.10）
  targets.json：保留完整三組 28 軸數值與來源證據
         ↓ Ada ROS sender（system Python 3.12）
  /bimanual/right_arm_hand/follow_joint_trajectory
         ↓ Ada 被動 action receiver（system Python 3.12）
  既有右 CRX／Sharpa drivers → joint feedback → action result → Ada 下一階段
```

這是 experimental 的固定姿態傳送入口，沒有變更公開 hand/method/converter support matrix，
也沒有修改 MuJoCo controller、階段定義或成功率計算。本文的 legacy exporter 限於 **mock**；
連續 Ada 接觸自適應、實機安裝與物體抓取仍未驗證。

## 使用現有 Ada 輸出

在 Ada checkout 下，用原本 Python 3.10 環境讀取已準備的資料。
以下 preview root 是本次已有的真實資料，並非手工 mock 目標：

```bash
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl
.conda/bin/python -m ada_grasp_ctrl.experiments.right_ros_export \
  --preview-root output/regression/crx-sharpa-right-preview-x080-y070-config-v1 \
  --sample-id g0000 \
  --output-root output/ros_bridge/my-g0000-export
```

每次使用新的 output root；已有目錄會被拒絕。Exporter 驗證 preview manifest、來源 hashes、
28 軸名稱、IK 結果、limits 與靜態 arm/table clearance，再原樣輸出三組 qpos。
任何階段的手型都保留；squeeze 的手臂姿態也保留，不會強制沿用 grasp 的手臂。
不重做 registration、不外推 pregrasp、不裁切角度、不改用固定 closed_hand。

本次明確選用 g0000。未通過 IK 的 g0020、仍有靜態手臂穿桌的 g0010 會被拒絕，
不會自動改選其他樣本。通過靜態檢查不代表 home→target 路徑無碰撞。

## 啟動被動 ROS 系統

每個 ROS shell 都明確使用 domain 88，避開 87，使用 system Python 而不是 Ada Conda：

```bash
source /opt/ros/jazzy/setup.bash
source "$HOME/ws_fanuc/install/setup.bash"
export ROS_DOMAIN_ID=88
```

第一個 shell 啟動右側 mock drivers（沒有任務 sender）：

```bash
ros2 launch bimanual_manipulation single_arm_hand.launch.py \
  prefix:=right_ crx_mock:=true sharpa_backend:=mock rviz:=false viewer:=false
```

第二個已 source 的 shell，在 Ada checkout 啟動被動 receiver：

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
/usr/bin/python3 -m ada_grasp_ctrl.ros_execution.bridge \
  --output-dir /tmp/right_bridge_runs
```

Receiver 的預設限制由 Ada 的 `ros_execution/settings.py` 管理，可用 `--config` 提供 YAML overrides；
只有速度、加速度、回饋 timeout、到位容差等執行限制，沒有 pregrasp/grasp/squeeze 目標。
未指定 `--output-dir` 時寫入 Ada output root 下的 `ros_bridge/receiver`，每個 goal 都建立新子目錄。啟動後在沒有 client goal 時，
不建立 arm/hand 命令 publisher。一次只接受一個 goal。

保留 `/bimanual/right_arm_hand/follow_joint_trajectory` action 名稱，避免現有 client 改線；
名稱不代表程式仍位於 bimanual。原 `ros2 run bimanual_manipulation right_arm_hand_bridge`
入口只是轉接至 Ada，必須同樣設定 Ada 的 `PYTHONPATH`。舊 bimanual bridge YAML
保留為可手動指定的舊設定檔，新的預設啟動不會讀取它。Ada receiver 不 import
bimanual 任務程式，但仍使用 ROS、`dual_crx_control` 與 `dual_sharpa_wave` driver 介面。

## 預設時間與插值

移動後維持原本速度與加速度，沒有降速。三個階段都從當下 feedback 計算時間，
沒有各自固定的秒數。每一組以最大關節位移 `delta`（rad）計算：

```text
T = max(2.0,
        1.875 * arm_delta / 0.15, sqrt(5.774 * arm_delta / 0.3),
        1.875 * hand_delta / 0.4, sqrt(5.774 * hand_delta / 0.8)) * 1.05 + 0.1
```

| 設定 | 預設 |
| --- | --- |
| pregrasp / grasp / squeeze 最短規劃時間 | 各 2.2 秒；位移較大時自動延長 |
| 手臂／手指速度上限 | 0.15 / 0.4 rad/s |
| 手臂／手指加速度上限 | 0.3 / 0.8 rad/s² |
| 實際到位後穩定時間 | 0.3 秒 |
| 規劃時間後額外到位 timeout | 8 秒；不是固定停留時間 |
| 接收端命令頻率 | 100 Hz |
| feedback timeout | 0.5 秒 |
| sender／receiver 啟動等待上限 | 25 / 20 秒 |

Receiver 先檢查指定時間是否滿足設定與模型速度限制，再以五次 smoothstep
`10u³ − 15u⁴ + 6u⁵` 產生具有限速的 arm/hand 命令。底層 CRX 插值保持原有設定。
實際牆鐘時間另外包含 preflight、ROS discovery、回饋延遲與到位穩定時間；
2.2 秒是規劃下限，並非整個階段一定在 2.2 秒完成。


## 從 Ada repo 送出姿態

第三個已 source 的 ROS shell，在 Ada checkout 執行：

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
/usr/bin/python3 -m ada_grasp_ctrl.experiments.right_ros_sender \
  --targets output/ros_bridge/my-g0000-export/targets.json \
  --output-root output/ros_bridge/my-g0000-preview
```

上面預設只預覽，不初始化 ROS、不送 goal。實際 mock 執行：

```bash
/usr/bin/python3 -m ada_grasp_ctrl.experiments.right_ros_sender \
  --targets output/ros_bridge/my-g0000-export/targets.json \
  --output-root output/ros_bridge/my-g0000-execute \
  --execute
```

Ada sender 驗證 live combined model 的關節、axis、limits、world/base/mount/flange/palm
鏈與來源模型一致，從 live feedback 決定合理執行時間。每階段送出完整 28 軸目標，
收到 action 成功結果才送下一階段。任何拒絕、timeout、回饋失效或取消都不會繼續。
最後停在 squeeze；沒有自動 lift、鬆手、返回初始位或下一 cycle。

接收端使用標準 `control_msgs/action/FollowJointTrajectory`，goal 為一個
`JointTrajectoryPoint`，`joint_names` 是右臂六軸與右手 22 軸，`positions` 用 rad，
`time_from_start` 指定執行時間。它會拒絕左側／錯誤名稱、非有限值、多個 trajectory
points、未支援的 velocity/effort 欄位、越界與過短時間，避免悄悄改寫 client 意圖。
`goal_time_tolerance` 可指定到位等待；不指定時使用 receiver 設定。

Ctrl+C 由 Ada client 發出 cancel；receiver 停止後續目標並請求停用右臂 position
controller。這是軟體停止，不是急停；Sharpa 沿用最後有界目標與上游 timeout policy，
沒有 motor-disable 保證。取消後重新啟動 mock stack 再做下一個測試。
如果 client 程序被強制殺死而無法 cancel，已接受的單一目標可能仍會完成，
receiver 不會自行進入下一個目標。成功 result 表示到位，不能代表物體抓穩。

## 測試與紀錄

每個 export/client run 都有新的 `run_manifest.yaml`、`run_report.json`；client 額外保存
傳送目標、實際 ROS model、各階段 action feedback/result。Receiver 每個 goal 另存
實際 command/feedback 軌跡、原因與 stop acknowledgment。

在 domain 88 沒有其他 stack 時，可直接跑完整 audit；它只管理自己啟動的 mock 程序：

```bash
/usr/bin/python3 script/check_right_ros_bridge.py \
  --targets output/ros_bridge/my-g0000-export/targets.json \
  --output-root output/ros_bridge/my-bridge-audit
```

Audit 驗證 idle 零命令、錯誤名稱拒絕、三階段原始數值完全保留、各階段到位、
沒有左側命令，以及取消後不送下一階段。實際結果見 `right-crx-ros-bridge-validation.md`。

先前 `bimanual_manipulation/right_arm_hand_task` 與 `right_arm_hand_task.yaml` 是
獨立 mock 示範，不是 Ada 的傳送鏈。新資料流不呼叫那個 standalone 任務入口。
原 YAML 暫時保留，因既有 Ada preparation 的初始姿態及來源 hashes 正在引用它；
本次沒有改動既有 preparation、資產或評估工作。
