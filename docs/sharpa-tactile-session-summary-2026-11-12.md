# Sharpa tactile 唯讀預覽：本次實作與驗證整理

日期：2026-11-12。專案：`/home/hao-fang/hugs-workspace/ada-grasp-ctrl`。

本文件記錄原始 F6 階段。後續已在同一入口加入 contact／normal／F6 顯示，
目前的接觸點顯示條件與幾何驗證狀態請見[擴充說明](sharpa-contact-preview.md)
與[擴充驗證報告](sharpa-contact-preview-validation-2026-11-12.md)。

擴充交接後，操作者補充確認：「力的方向沒問題 單位是牛頓也合理」。
已記錄為本次觀察下的力方向通過、以牛頓（N）解讀合理，通道維持 `0,1,2`。
接觸點位置、藍色表面法向量及幾何對齊仍保留未驗證；本次回報未提供參考負載校正。

本次已完成右手 tactile 預覽的本機實作、軟體測試與顯示問題修正。
操作者後續回報唯讀測試的方向正確，並確認通道維持 `0,1,2`。
這是本次人工觀察結果；逐指、逐方向紀錄與反號開關狀態尚未完整記錄。

## 文件索引

| 文件 | 用途 |
| --- | --- |
| 本文件 | 本次工作總覽、啟動方式、問題處理與交接 |
| [預覽操作與技術說明](sharpa-tactile-preview.md) | 介面、topic、座標、freshness、參數及 SSH forwarding |
| [實作驗證報告](sharpa-tactile-preview-validation-2026-11-12.md) | 實際測試結果、歷史環境稽核與限制 |
| [人工方向驗證紀錄表](sharpa-tactile-validation.md) | 已收到的操作者回報與待補的逐指紀錄 |

原始任務文件：
`/home/hao-fang/hugs-workspace/agent_tasks/ada_sharpa_tactile_live_preview_phase1a_aligned.md`。

## 完成的功能與程式

- 獨立啟動入口：`.conda/bin/python -m ada_grasp_ctrl.tactile_preview`。
- 預設網址：`http://localhost:8089`，只綁定 `127.0.0.1`。
- 顯示官方右手 URDF、五指標籤、elastomer 座標軸及力箭頭。
- 顯示原始六通道 F6、選定的三維向量、力大小、frame ID、時間、更新頻率與連線狀態。
- 支援可調通道、箭頭比例、近零門檻、local/world 顯示及明確標示的診斷反號。
- 對 missing、empty、invalid、stale、斷線與姿態不足狀態提供明確提示。
- 關節資料不足或過期時保留靜態姿態，隱藏世界座標力箭頭；local 模式仍可查看新鮮的局部向量。
- 未加入控制命令、機器人運動、自動校正、MPC、力矩估計或左手預覽。

| 程式 | 責任 |
| --- | --- |
| `src/ada_grasp_ctrl/tactile_preview/__main__.py` | CLI、HTTP 預覽生命週期、ROS 子程序與資料交換 |
| `src/ada_grasp_ctrl/tactile_preview/ros_reader.py` | ROS Python 3.12 唯讀訂閱、QoS 查詢與最新資料快照 |
| `src/ada_grasp_ctrl/tactile_preview/data.py` | F6 正規化、有效性、時效、通道選擇與向量旋轉 |
| `src/ada_grasp_ctrl/tactile_preview/viewer.py` | Python 3.10 Viser、URDF FK、箭頭與操作介面 |
| `tests/test_tactile_preview.py` | 單元測試、模型變換、HTTP 與 DDS 整合測試 |
| `tests/tactile_ros_fixture.py` | 隔離環境中的合成觸覺與關節資料；無硬體命令 |

沿用既有 Viser/yourdfpy 依賴。未變更 upstream、URDF 資產或其他既有修改，未 commit／push。

## 資料流與座標

```text
既有 Sharpa 唯讀驅動
  → 五個 atomic TactileFrame topics + 右手 JointState
  → ROS Python 3.12 訂閱程序
  → 精簡 F6／關節資料快照
  → Ada Python 3.10：URDF FK + Viser
  → localhost:8089
```

實際訊息型別：`dual_sharpa_wave/msg/TactileFrame`。

觸覺 topic：`/sharpa/right_hand/tactile/{pinky,ring,middle,index,thumb}/frame`。
關節 topic：`/sharpa/right_hand/joint_states`。
訂閱 QoS：BEST_EFFORT／VOLATILE／KEEP_LAST depth 1。

模型來源：
`/home/hao-fang/ws_fanuc/install/dual_sharpa_wave/share/dual_sharpa_wave/urdf/right_sharpa_wave_with_flange.urdf`。

箭頭起點為各指 `right_<finger>_elastomer` 原點，方向以 `f_W = R_WE @ f_E` 轉換。
此處 W 是固定的 `right_hand_flange` 預覽世界，不是實驗室或機械臂全域座標。
平移只作用於箭頭起點，不作用於力方向。

預設用本機 monotonic clock 計算到達後時間，SDK timestamp 不作 wall-clock 時效依據。
只有確認驅動與訂閱端 ROS 時鐘同步後，才使用 `--receipt-clock-synchronized` 啟用 ROS receipt age 檢查。

## 操作者回報的驗證結果

操作者表示「唯讀完了方向是對的」，並在通道確認時表示「都維持012」。

| 項目 | 本次可記錄的結果 |
| --- | --- |
| Fx 選擇 | `F6[0]` |
| Fy 選擇 | `F6[1]` |
| Fz 選擇 | `F6[2]` |
| 預覽方向 | 操作者確認本次實際測試方向正確 |
| 每指、每個方向的 channel/sign 數值 | 尚未提供逐項紀錄 |
| Sign flip | 預設為關閉，但本次人工測試的實際開關狀態未另行確認 |
| 實機頻率、延遲與 QoS 數值 | 本次後續人工測試未附量測紀錄 |

後續保留 `0,1,2` 作為本次操作使用的對應。介面仍保留 mapping／alignment 未經正式認證的提示，
因為單次操作者回報不等於完整五指、多方向校正或廠商規格確認。
力以 N、原點在 pad center 為先前操作者提供的測試輸入；其他三通道仍以 raw values 顯示。

## 重新啟動與測試

使用兩個終端，兩邊的 `ROS_DOMAIN_ID` 與 discovery 設定必須一致；本次稽核環境為 domain `87`。
以下指令不覆蓋既有 domain 設定。若已有驅動執行，不要再啟動第二份。

終端 A：啟動既有雙手唯讀驅動。

```bash
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
export PYTHONPATH="/opt/sharpa-wave-sdk/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/sharpa-wave-sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ros2 launch dual_sharpa_wave dual_sharpa.launch.py \
  backend:=sharpa_sdk read_only:=true use_rviz:=false
```

終端 B：檢查 ROS，再執行 Ada 預覽。

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"

ros2 param get /sharpa/right_hand/hand_node read_only
ros2 topic info -v /sharpa/right_hand/tactile/index/frame

.conda/bin/python -m ada_grasp_ctrl.tactile_preview
```

確認 `read_only` 回覆 True、食指 frame 有一個 publisher，再打開 `http://localhost:8089`。
查看五指 F6／frame ID 是否持續更新，再逐指輕觸，記錄 channel、sign 與箭頭方向。
若資料更新但 world 箭頭隱藏，查看姿態與 freshness 提示；可切換 `Local f_E` 檢查局部向量。

在終端 B 按 Ctrl-C 只會停止 Ada 預覽及其訂閱程序。終端 A 的驅動須另外停止。
本次助理啟動的 tmux 預覽已依操作者要求關閉；後續服務由操作者自己的終端管理。
此為交接紀錄，不表示文件被閱讀時 8089 必然開啟或關閉。

## 本次遇到的問題與修正

| 問題 | 原因 | 處理 |
| --- | --- | --- |
| URDF 零件重疊、組裝錯亂 | Viser 建立 joint nodes 後尚未套用初始變換；原本只在有 live joint data 時更新 | 建立模型後立即呼叫 `update_cfg(hand.cfg)`；新增無 ROS 資料的初始變換回歸測試，並以瀏覽器確認 |
| 8089 connection refused | 當時預覽程序已停止，port 無 listener | 改用獨立 tmux 工作階段啟動並確認 HTTP 200；其後依要求關閉，改由操作者手動啟動 |
| `PackageNotFoundError: ros2cli` | 輸入 `PYTHONPATH="PWD/src:PYTHONPATH"`，遺漏 `$` 並覆蓋 ROS 套件路徑 | 清除錯誤值、重新 source，再正確 prepend；不需重裝 ROS |
| 模擬範例 Pinocchio import 失敗 | Python 3.10 繼承到 ROS Python 3.12 套件路徑 | 純模擬檢查使用 `PYTHONPATH=src` 隔離；ROS 預覽則保留 ROS 路徑給其 Python 3.12 子程序 |

若再次誤設 PYTHONPATH，在 **Ada 預覽終端** 執行：

```bash
unset PYTHONPATH
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

`$PWD` 與 `$PYTHONPATH` 的 `$` 必須保留。驅動終端則仍需額外加入 SDK 的 Python/library 路徑。

## 實際軟體驗證

| 檢查 | 結果與時點 |
| --- | --- |
| 專用測試，含隔離 DDS 整合 | 初始組裝修正後 15/15 通過 |
| 全庫 unittest | 修正前一輪共 393 項：329 通過、64 跳過；未聲稱修正後重新跑過全庫 |
| LEAP Tac3D quick example | Exit 0；100 筆輸出，93 completed、7 invalid initialization，無 execution error／solver degradation |
| Lint／compile／diff | 已執行的檢查通過；修正後另驗證變更檔案 |
| Format | 新增／修正 Python 檔案通過；全庫另有 8 個既有檔案格式差異，未改動 |
| 瀏覽器 | 初始錯誤畫面已重現；修正後確認右手正確組裝，8089 回應 HTTP 200 |

合成 DDS 測試的子程序使用 localhost domain `198`，不變更硬體驅動設定。
完整指令、結果路徑與限制請見[驗證報告](sharpa-tactile-preview-validation-2026-11-12.md)。
目前沒有本次人工觸壓的 JSONL／CSV 實機數值紀錄；表格中未提供的細項保留為「未記錄」。
