# 實機觸覺精簡錄製

使用 `ada_grasp_ctrl.tactile_preview.record_compact` 錄製右側 baseline 執行資料。
它保留 F6、接觸點、來源時間與識別資訊、關節回授及命令，寫入 MCAP 前省略
`dist_force` 分布圖陣列。錄製器只訂閱資料，不發送機器人動作命令。

## 下次錄製

先照常啟動實機 driver，再在另一個終端執行：

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
export ROS_DOMAIN_ID=87
export PYTHONPATH="$PWD/src:$PYTHONPATH"

/usr/bin/python3 -m ada_grasp_ctrl.tactile_preview.record_compact
```

這個指令取代原本九個 topic 的 `ros2 bag record`。看到 `Recording without dist_force`
後，依[既有 grasp pipeline](grasp-pipeline.md)操作 baseline。錄製器每五秒顯示各 topic
累積筆數；命令 topic 在執行動作前可能仍是零。完成動作後，在錄製終端按 **Ctrl+C**，
等待顯示 `Saved:` 和結果報告。

錄製器不使用 8089。若先前開著 Ada tactile preview，需先關閉它，才能讓 grasp preview
使用同一個 8089 port；錄製器可以繼續執行。

可選參數：

```bash
# 60 秒後自動結束；務必讓錄製時間涵蓋整次動作。
/usr/bin/python3 -m ada_grasp_ctrl.tactile_preview.record_compact --run-seconds 60

# 指定輸出父目錄；每次仍建立新的錄製子目錄。
/usr/bin/python3 -m ada_grasp_ctrl.tactile_preview.record_compact \
  --output-root /home/hao-fang/hugs-workspace/ada-grasp-ctrl/output/tactile_recordings
```

使用 ROS Jazzy 的 `/usr/bin/python3`，不是專案 Python 3.10 conda interpreter。
實機使用 domain **87**；隔離 mock 測試使用 localhost domain **199**。

## 存檔與確認

未指定輸出設定時，預設放在專案的 `output/physical_compact_<UTC時間>_<識別碼>/`。
若有設定 `ADA_GRASP_CTRL_OUTPUT_ROOT`，則改用該父目錄；`--output-root` 優先於環境變數。
相對輸出路徑以專案根目錄為基準。

| 檔案／目錄 | 內容 |
| --- | --- |
| `bag/metadata.yaml`、`bag/*.mcap` | 九個 topic 的 ROS 訊息 |
| `run_manifest.yaml` | domain、topic 清單、欄位省略方式、時間與 QoS 定義 |
| `run_report.json` | 結束狀態、每個 topic 的筆數、未收到資料的 topic、DDS 回報遺失數 |

確認 `status` 是 `completed`，並檢查 `topics_without_messages`。`completed` 只表示錄製程序
正常結束；即使沒有收到資料，也不會因此自動判定夾取或資料驗證成功。
`dds_reported_lost` 不包含所有上游遺失或 SDK cache overwrite，不能用零值宣稱沒有丟資料。

九個 topic 為：

- `/crx5ia/right/joint_states`
- `/sharpa/right_hand/joint_states`
- `/crx5ia/joint_targets`
- `/sharpa/right_hand/joint_command`
- `/sharpa/right_hand/tactile/{thumb,index,middle,ring,pinky}/frame`（五個）

## 保留與省略的語意

ROS topic 名稱與訊息型別不變。關節訊息保留收到的 CDR bytes；觸覺訊息只替換
`dist_force` block，F6、contact_points、header、SDK/session/sequence、來源計數器與錯誤欄位保留。
bag 的訊息時間是錄製器 callback 進入時的 ROS clock，不能當成感測器採樣時間。

被替換的 `dist_force` 為 `MISSING`、空陣列，`error` 中明確記錄
`reason: omitted_by_compact_recorder`，以及原始 status、shape、dtype 等 metadata。
這表示「本份錄製省略」，**不是實測零值，也不是 `EMPTY`**。
`contact_points` 原本的 `MISSING`／`EMPTY`／`VALID`／`INVALID` 完全不變。

driver、SDK、原始 TactileAdapter 與即時 ROS 資料不變，因此減少的是磁碟用量，
不是 driver 到錄製器的 ROS 傳輸量。精簡 bag 無法重建原始分布圖；需要研究分布圖時，
仍需另外錄製完整訊息。

2026-11-12 的完整 bag 有 12,538 筆五指訊息，每筆 `dist_force` 為
`60 × 60 × 3 × 8 = 86,400 bytes`，光陣列就約 1.08 GB。
省略它能大幅減少容量；實際精簡 bag 大小仍取決於錄製時長與訊息數。

## 驗證與目前限制

已通過兩項 ROS Jazzy regression tests：原始訊息與非省略欄位保持一致，以及九個 topic
的 MCAP 寫入／讀回與時間戳保留。另以 localhost domain 199 的合成資料測試三秒錄製，
九個 topic 都收到資料並正常結束存檔；這不是實機精簡錄製的驗證。

在上述 sourced ROS 環境可重跑：

```bash
/usr/bin/python3 -m unittest discover -s tests -p test_tactile_compact_recording.py -v
```

本工具產生 ROS bag，尚未提供 bag → `ada_tactile_adapter_replay` 的完整轉換。
實機 passive MPC runner 仍待實作，`mpc_active` 仍限制於 mock。
錄製成功不代表 force／geometry 已正式校正，也不代表已能啟用實機 MPC。
參見[實機 baseline 錄製檢查](mpc-phase2-validation-2026-11-12.md#operator-baseline-recording-follow-up)。
