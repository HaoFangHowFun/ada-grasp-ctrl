# 即時抓取、抬升、鬆手與返回

## Task 1：單命令實機 MPC

在已啟動固定相機與右側 CRX／Sharpa drivers 的工作站執行：

```bash
python script/run_grasp_pipeline.py --controller-mode mpc_active
```

若系統只有 `python3`，用 `python3` 執行同一入口即可；入口會切換到專案的 Ada 環境。
明確的 `--controller-mode mpc_active` 是**本次**實機動作授權，無額外 Enter、ROI 視窗或網頁確認。
8089 在準備開始時啟動並嘗試開啟瀏覽器；Task 2 加上完整 CRX／Sharpa 實測姿態、五指力與階段顯示。
只顯示外觀網格，不顯示碰撞網格。關閉網頁不影響執行；Ctrl+C 仍取消目前 action 並關閉本次啟動的程序。
顯示意義與排錯見 [8089 即時監看操作說明](task2-dashboard-operator.md)。

`grasp_pipeline.yaml` 的 `active_hardware` 區段只適用這個明確的 live active 執行，設定硬體、
ROS domain 87、native Python 與服務路徑；`--ros-domain-id` 優先於 `ROS_DOMAIN_ID`，再使用該區段設定。
每次自動建立 `output/grasp_pipeline/active_<UTC timestamp>_<unique suffix>`，也可指定新的 `--output-root`。
其餘 baseline／linear／passive／mock／replay／preview-only／export-only 流程不取得這個自動實機授權。

自動擷取使用 `fixed_camera.yaml` 的 `automatic_workspace_roi`（預設完整有效 RGB 範圍，亦可設定固定像素框），
沿用既有桌面高度、voxel 與 clustering。保留高度第 5 百分位數距桌面不超過 **5 公分**的 cluster，
再選有效點數最多者；沒有符合者回報 `NO_OBJECT_DETECTED`。設定為 `automatic_table_lower_height_m: 0.05`。
這只是靠近桌面的近似判斷；物體下緣被遮住可能遭排除，物件也可能分裂或合併，並非語意分割。
不在每次執行重做桌面／外參校正。2026-11-13 操作者確認既有兩者均已驗證，
`fixed_camera_acceptance_2026-11-13.json` 保存這次確認及精確檔案雜湊，原始檔案不被改寫。

MPC `native_complete` 後取得新的六軸＋22 指關節實測狀態，只由 sender 規劃一次 world +Z **0.15 m** Lift，
保留當下末端方向與手指角度，並檢查關節限制與已知機器人／桌面路徑。
Receiver 比對受控 sender 的本次計畫、hash、階段、時效與新鮮起點，不重跑 IK／路徑抽樣。
Lift 起點手臂容差仍為 0.01 rad、手指 0.02 rad；後續沿用既有到位政策、5 秒 Hold、開手與返回。
既有軌跡插值／速度、10 Hz MPC、100 Hz streaming、原始 15 N 完成條件均未改動。

排錯：

- 8089 被占用：關閉占用它的舊 viewer，再重新啟動；不連接別次執行的服務。
- 無符合物件或選錯物件：查看本次 `grouplive/object/clusters.json` 與 RGB，整理工作區或修改固定 ROI。
  自動模式不接受 `--roi`／`--cluster`；需要手動除錯時使用明確的 preview／export 工作流程。
- 校正 hash 不符：確認設定是否指向經驗證的檔案；不要直接修改 hash 來掩蓋未驗證的幾何變動。
- 缺少 driver／tactile publisher／服務：依既有部署方式啟動，確認 ROS domain；launcher 不啟動或更改實體 drivers。
- Lift 過期、起點變動或不可行：本次中止；不要放寬容差或套用舊 squeeze 起點。
- 日誌：終端印出 `RUN_OUTPUT`；`timings.jsonl` 是準備階段與 cleanup，`sender/events.jsonl` 是 action 等待、
  native completion acknowledgement、fresh snapshot、planner 完成與 goal submission；
  `receiver/*/events.jsonl` 是實際 native completion、goal received、first command 與 action finished。
  `sender/dynamic-lift-result.json` 的 `geometry_s` 記錄 worker 內的來源載入與規劃時間。

詳見 [Task 1 實作與驗證結果](task1-one-command-real-mpc-results-2026-11-13.md)。

## 既有 baseline 與手動預覽流程

以下說明保留原本的非自動 active 工作流程（包括 5 cm 名目 Lift 與 Enter gate）。

入口：`script/run_grasp_pipeline.py`。預設使用當下相機資料，執行一次完整流程：

**擷取點雲 → 套用 calibration → 選物體 → 推論與候選篩選 → 六格圖片 → 8089 預覽 → 終端 Enter → above_object → pregrasp → grasp → squeeze → lift → hold → release → return。**

## 在新的終端機執行

先由操作者啟動固定相機與右側 CRX／Sharpa drivers，確認它們使用相同 ROS domain。
入口會自行切換 Ada Python 並為 ROS 子程序 source 設定的 Jazzy／workspace 環境。
它不會自動啟動實體機器人 driver、相機 driver，或替操作者啟用控制器。

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl

# Set this to the domain actually used by your camera and robot drivers.
: "${ROS_DOMAIN_ID:?Set ROS_DOMAIN_ID to your camera and robot domain first}"

python3 script/run_grasp_pipeline.py \
  --environment hardware \
  --ros-domain-id "$ROS_DOMAIN_ID" \
  --output-root "output/grasp_pipeline/live_$(date +%Y%m%d_%H%M%S)"
```

`--environment hardware` 必須明確指定；預設 backend 仍是 mock。
實機模式要求明確的 `--ros-domain-id` 或 `ROS_DOMAIN_ID`，不會偷偷使用 mock domain。
每次都需要新的 output root；不覆寫上次的證據。

擷取完成後，在 RGB 視窗框選這次要抓的物體並按 Enter。若框內只有一個有效 cluster，會繼續處理。
也可使用 `--roi x0 y0 x1 y1` 明確指定影像區域；多個 cluster 時需查看輸出的 cluster 圖片，
再以 `--cluster ID` 指定。無桌面環境需要明確的 ROI。
**RGB 視窗的 Enter 只確認物體範圍，不會批准機器人動作。**

推論、篩選與動作目標產生完成後，手動開啟 **http://localhost:8089**。
預覽整段動作，回到執行入口的終端機，按一次 Enter 才會開始送出 ROS 目標。
輸入 `q`、EOF 或 Ctrl+C 會取消；執行中 Ctrl+C 會取消目前目標，不繼續鬆手或返回。

## 動作定義

- above_object：沿用選中 pregrasp 的水平位置與手腕方向，22 個手指關節目標全部設為 `0 rad` 張開，與 release 使用相同全開姿態。依全開手形重新計算最低點，向 world +Z 平移至高於物體觀測點雲最高點至少 10 公分；若已更高則保留高度。移往上方端點時逐漸張開，下降途中逐漸回到原 pregrasp 手形。兩段均沿用關節插值，不保證直線下降。超出關節限制或端點 IK 失敗時拒絕執行，不裁切角度或重選候選。
- lift：從選中的 squeeze 姿態，維持末端方向，沿 world +Z 抬升 5 公分。
- hold：lift 確認到位後，持續送出相同的 28 軸目標 5 秒，保留回饋及取消檢查。
- release：手臂停在抬升位置，22 個 Sharpa 關節回到既有的全零張開姿態。
- return：保持手張開，手臂回到本次候選篩選前量測的六軸起始姿態。

hold 使用獨立 ROS action，不能在 lift 失敗時開始。每個 action 沿用 0.3 秒到位穩定期；
所以實際從 lift 結束到 release 開始還包含 ROS／驗證延遲及穩定期，至少保持 5 秒。
速度與加速度保持原設定：手臂 0.15 rad/s、0.3 rad/s²；手指 0.4 rad/s、0.8 rad/s²。
其餘階段依關節位移計算時間，最少 2.2 秒。

抬升只在候選選定後計算，不影響候選分數、IK／路徑篩選或重新挑選候選。
ROS 接受關節角度，因此仍需一次有界 IK 轉換來產生抬升目標；轉換失敗時保留 grasp 與圖片，
但該組沒有可執行 packet。不得把失敗近似解送出。

依操作者要求，抬升及新增回程不做路徑檢查，回程使用現有關節空間五次插值。
world +Z 是抬升終點位移，不保證中途沿笛卡兒直線。
鬆手發生在抬升位置，沒有放到另一個位置的動作。

## 實機連接與檢查

實機模式會讀取實際 arm plugin／hand backend、模型與完整 28 軸回饋。
右手須為可寫入的 `sharpa_sdk` MIT 模式；右側插值器須為 100 Hz，控制器須啟用，
既有 force limiter 須為 ARMED。不同 backend、模型、關節順序或過期回饋會拒絕執行。

按 Enter 後，入口重新檢查起始姿態與資料雜湊，才啟動 Ada receiver。
receiver 綁定這次確認的 packet，僅接受下一階段的精確關節目標和時間；不能替換順序、目標或速度。
實機的未驗證路徑例外只適用這份完整且明確確認的流程，任意外部關節命令仍受原有路徑保護限制。
實體動作、標定、安裝與環境驗證由操作者進行；程式不把 `hardware_execution_ready` 或
`continuous_path_verified` 標成 true，也不從手臂到位推論物體抓取成功。

## 預覽、匯出與 mock 測試

- `--preview-only`：只開 8089，不接受動作批准。可搭配實機 backend 讀取目前狀態。
- `--export-only`：只產生圖片、selection、plan 和 packet，不開 viewer／receiver。
- `--lift-height-m 0.05`：覆寫抬升高度；mock 使用 0 時停用 lift／hold／release／return。搭配 `--above-object-clearance-m 0` 可恢復舊三階段。
- `--above-object-clearance-m 0.10`：覆寫新增接近階段的手對物體淨空；設為 `0` 可關閉此階段。新端點 IK 失敗時保留抓取結果，但不產生可執行 packet。
- `--above-object-open-hand`：上方階段使用 22 軸全零張開；目前預設設定已啟用。預覽與 ROS packet 驗證相同的全開目標；既有 `hand_arrival_required: false` 仍表示不保證手指實際到位。
- `--no-above-object-open-hand --above-object-hand-factor 1.5`：恢復先前上方階段的手指外插策略；倍數 `1` 沿用原 pregrasp。全開模式下外插倍數不參與手形計算。只改手指，不外插手腕位置或方向。
- 目前設定 `hand_arrival_required: false`：手指位置誤差只記錄，不阻止階段完成或下一階段啟動。仍等待指令完成、手臂到位並穩定 0.3 秒；首次執行前的姿態檢查、力保護、回饋與關節限制保留。必要時依實際手指起點延長指令時間以維持速度／加速度限制；預覽時間為名目值。
- `--hand-tolerance-rad 0.06`：手指誤差報告的判定值，約 3.44°；只有 `hand_arrival_required: true` 時才作為手指到位門檻。手臂到位容差仍為 0.01 rad。

五組舊點雲是明確的 replay 模式；不會在預設 live 模式偷偷回退成舊資料：

```bash
python3 script/run_grasp_pipeline.py \
  --mode replay --environment mock --ros-domain-id 88 \
  --start-mock --group all --reuse-candidates \
  --output-root "output/grasp_pipeline/mock_$(date +%Y%m%d_%H%M%S)"
```

`--reuse-candidates` 只可用於 replay；省略它會對舊點雲重新推論。
`--start-mock` 只允許 mock 且 domain 必須空閒。測試真實相機＋mock 時，先在同 domain 啟動它們，
省略 `--start-mock`，使用預設 live 模式。mock 禁止使用 domain 87。
入口只清理本次擁有的程序；8089 被占用會停止，不會改用其他 port。

## 輸出

若要同時錄製實機關節與觸覺資料，使用[精簡錄製指南](tactile-compact-recording.md)。
它省略 `dist_force` 分布圖、保留 F6 與接觸點，使用 domain 87；在另一個終端啟動，
動作結束後按 Ctrl+C 存檔。錄製器不占用 8089，也不改變 baseline 控制方式。

live 輸出在 `grouplive/`；replay 則在 `groupNN/`：

| 路徑 | 內容 |
| --- | --- |
| `capture/`、`object/`、`asset/`、`inference/` | 本次擷取、物體區域、表面與實際推論證據 |
| `selection/` | 候選分數、可行性與選中結果 |
| `images/candidates_001.png`、`images/index.json` | 每頁六格合格候選與總數 |
| `export/motion_plan.json`、`export/targets.json` | 與預覽綁定的八階段目標、時間和來源雜湊 |
| `lift_failure.json` | 抬升轉換失敗時的診斷；不重新選候選 |
| `integration_summary.json`、`run_manifest.yaml`、`run_report.json` | 本次輸入、設定與流程結果 |
| `sender/`、`receiver/` | Enter 後才建立的執行證據 |

`hold_motion_completed`、`release_motion_completed`、`return_motion_completed` 表示命令完成；
`object_held`／`object_lifted` 仍為未知。批次若部分組別準備失敗，會記錄 `failures.jsonl`
及 `preparation_errors`，即使其他組執行成功也回傳 exit 1。

設定：`src/ada_grasp_ctrl/config/experiments/grasp_pipeline.yaml`。
測試證據：[grasp-pipeline-validation.md](grasp-pipeline-validation.md)。

雙側啟動、自動選擇左手或右手的即時入口見 [dual-live-grasp-pipeline.md](dual-live-grasp-pipeline.md)。
