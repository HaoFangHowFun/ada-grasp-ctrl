1. 我想要在這個project上可以用我們新的crx-sharpa的asset來做抓取的推理
2. 現在的IK求解，先幫我調查怎麼把現在dummy arm的IK 改成fanuc_crx_5ia的，
3. 我知道sharpa手跟shadow手不一樣，但是能不能一樣先做一個pipe-line出來，並且給一些全開全握的指令去抓物體 來當作example
4. 最小effort 不要加過多限制

## Phase 1：先做最小 CRX＋Sharpa 抓取範例

可以沿用這個 repo「物體位姿＋目標手掌位姿＋手指關節目標 → 手臂 IK → 執行抓取」的資料流，但目前不能只把 `hand=shadow` 換成 Sharpa。現有 `dummy_arm_qpos` 算的是三個平移軸、三個旋轉軸的虛擬手臂；CRX 是六個旋轉關節。現有 `control_eval` 也只接受 dummy arm 搭配 Shadow／Allegro／LEAP Tac3D。已匯入的 CRX＋Sharpa URDF 目前只有展示用途：MuJoCo 載入後有 28 個可動關節（CRX 6＋Sharpa 22），但 **0 個 actuator**，因此還無法靠 `data.ctrl` 執行抓取。

這裡的「0 個 actuator」只描述**此 repo 的 MuJoCo 匯入模型**，不是說實體機器人缺少馬達。已核對 `bimanual_manipulation` 的單側 ROS 流程：`dual_crx_control` 對 `left_J1`～`left_J6` 提供 position 命令與關節回饋；`dual_sharpa_wave` 對左手 22 個命名關節提供 `joint_command` 和 `joint_states`，硬體後端把 22 維目標送給 Sharpa SDK。因此，軟體控制介面涵蓋這 28 個關節；在 `read_only:=true`、硬體未連線或發生故障時，不能據此認定當下可執行動作。MuJoCo 要動起來仍需在控制用模型中配置自己的 actuator。

### 1. 把手掌位姿反算成 CRX J1–J6

我建議保留現有 dummy arm 流程，另做 CRX 專用 IK 入口。輸入使用世界座標系中的 `left_hand_C_MC`（手掌基座）目標位姿 `[x, y, z, qw, qx, qy, qz]`，輸出固定順序的 `left_J1`～`left_J6`（弧度）。從目前 `model.urdf` 用 repo 已有的 `pytorch_kinematics.build_serial_chain_from_urdf(..., "left_hand_C_MC")` 實際確認，這條鏈的可動關節**正好是這六軸**；它包含 `left_arm_to_hand` 的固定接合變換。使用目前實際關節角作初始值，逐一求 pregrasp、grasp、squeeze，並檢查關節限制和 FK 回算誤差。這比把 dummy arm 的 XYZ／Euler seed 套到 CRX 上更直接。

目前手部接合位姿 `xyz=[0,0,0]、rpy=[0,1.570796,0]` 是來源專案的展示用假設。模擬範例可先沿用；若之後要把角度送到實機，必須換成實測的法蘭到手掌變換，並確認機器人 base/world 座標轉換。URDF 給本地 `pytorch_kinematics` 解析時需先去掉 XML 宣告行；這只是解析格式問題，不需要改動關節或網格。

### 2. 手指先用固定「全開／全握」姿勢

這個 repo 的上游 grasp 紀錄本來就直接提供手指關節角，**沒有用 fingertip 位置對手指做 IK**。最小範例也照這個接口：建立固定的 22 維 Sharpa 關節向量，按 URDF 的關節名稱排序；全開先取各關節的可用中性值（優先 0），全握先讓 FE、PIP、DIP、IP 等屈曲關節朝閉合方向移動，AA／外展關節先維持中性。數值應由 Sharpa 的關節方向和限制確認，不能直接把 22 軸全部設成上限。三個目標姿勢可先用「開手接近 → 開手到抓取位 → 定位後閉手」；這只是一組測試手勢，不保證任何物體都能抓穩。

### 3. 最小可跑流程

1. 用現有的 CRX＋Sharpa URDF 和隨附的一個物體，在同一個 MuJoCo 場景中擺好物體與機器人；先選一個手能碰到的物體位姿。目標手掌位姿可先手工指定，之後再改成 `世界_T_物體 × 物體_T_抓取手掌`。
2. 對開手接近位與抓取位做 CRX IK，產生三段目標：`[CRX 六軸角度，Sharpa 全開] → [CRX 六軸角度，Sharpa 全開] → [CRX 六軸角度，Sharpa 全握]`。記錄每段的 IK 成功／失敗與 FK 誤差。
3. 為**控制用** MuJoCo 模型補上關節 position actuator，先用關節位置插值執行三段動作；在畫面中顯示手、物體和接觸，並存下關節軌跡。預覽用 URDF、外觀網格與現有 Shadow 範例不用因此改動。
4. 驗收先看：CRX 六軸在限制內、手掌到達指定位置、22 軸能開合、手與物體產生接觸。物體是否被穩定抬起另做一次實測，不能僅憑有接觸就宣稱抓取成功。

這條最小範例先用固定手勢和位置控制跑通 **CRX IK＋實際模型動作＋接觸**，不需要先訓練 DexLearn，也不需要先做 Sharpa fingertip IK。若要讓它真正使用本 repo 的 `ours` 接觸最佳化控制，下一步才把 Sharpa 的精確關節／actuator 順序、接觸 body 與可控自由度接到 `RobotFactory`、MuJoCo 載入器和 `control_eval`。那是額外整合工作，不能把固定手勢範例的結果稱為 `ours` MPC 的結果。

### Phase 1 已實作的執行方式

在 repo 根目錄、啟用 `ada-grasp-ctrl` Conda 環境後執行：

```bash
PYTHONPATH=src python script/run_crx_sharpa_phase1.py
```

程式會在 `output/examples/crx_sharpa/<新 run ID>/` 產生 `control_model.xml`（28 個 MuJoCo position actuator）、`grasp.npy`（同一筆 28 軸 grasp 資料）和 `phase1_summary.json`。加上 `--viewer` 可看模擬；viewer 只顯示 CRX／Sharpa 的外觀網格和瓶子外觀網格，碰撞網格仍參與物理運算但不顯示。初始視角涵蓋手臂與瓶子，動作結束後畫面會停在最後姿態，關閉 viewer 視窗才退出程式並印出摘要。瓶子初始放在桌面上；headless 試跑在閉手過程中最多觀察到 2 個手物接觸，閉手末端沒有穩定保持接觸。物體未通過抬升驗證，因此這是**可執行的抓取嘗試**，尚不是成功抓起物體的證據。

## Phase 2：讓 CRX＋Sharpa 使用本專案的 `ours` 自適應控制器

**目標：**沿用 Phase 1 的同一個物體、CRX IK 結果與開合手目標，讓 `control_eval` 每步根據 MuJoCo 的目前關節位置、手物接觸位置與接觸力，呼叫既有 `GraspController.ctrl_opt()` 計算下一個 28 維關節位置目標。這才是此專案的接觸回饋自適應控制；上游物體位姿仍只用於初始化場景，不需要即時相機串流。先在模擬中驗證，不把模擬命令直接送往 ROS／實機。

預計按下面順序改，維持既有 Shadow／Allegro／LEAP 的控制設定和結果：

1. **控制模型與關節映射。**由 CRX＋Sharpa 資產產生給 MuJoCo 使用的控制 MJCF，給 `left_J1`～`left_J6` 和左手 22 關節各一個位置 actuator；Pinocchio 使用原始 URDF，避免 MJCF 匯入時遺失第一軸。新增 CRX arm、Sharpa hand 與兩者組合的 robot metadata，以及對應的 Hydra hand 設定。固定 28 維順序為 `[CRX J1…J6, Sharpa 22 軸]`，明確列出 `dof_names`、`doa_names`、速度／關節限制與手掌 frame `left_hand_C_MC`。先以一軸對一個 actuator 的映射接入 `RobotAdaptor`，逐項核對 MuJoCo、Pinocchio 與 ROS 的名稱和順序；不從 XML 出現順序猜測。
2. **沿用抓取資料接口。**Phase 1 產生的 `pregrasp_qpos`、`grasp_qpos`、`squeeze_qpos` 是三組 28 維關節目標，並附 `joint_names`、`obj_path`、`obj_scale`、`obj_pose`。CRX 六軸由手掌目標位姿做 IK；Sharpa 22 軸沿用固定開／合手勢，暫不加入 fingertip IK 或訓練模型。實驗腳本直接使用既有 `BaseEval` 與 `ours` 共用執行器讀取資料；公開 `control_eval` CLI 的支援清單仍維持原樣。
3. **接上接觸回饋與 `ours`。**調整 MuJoCo 場景載入與命名，使手指／手掌對物體的接觸能回傳現有控制器需要的接觸點、座標 frame 與六維接觸力。現有 `MjHO` 以 body 編號範圍判斷「手部接觸」，新模型需改為依明確的手部 body 集合過濾，避免把 CRX 手臂碰撞當成手指接觸。保留 `ours` 的 Stage 1／Stage 2、wrench balance、最佳化目標與求解失敗紀錄；只增加機器人適配，不用改控制器的論文參數。
4. **移除 dummy arm 專屬動作假設。**原本抓取後的 lift 用 `lift_qpos_a[2] += 0.2`，那個索引對 dummy arm 是 Z 平移，對 CRX 卻是 J3 角度。CRX 範例改成「手掌在世界座標上移 → CRX IK → 關節插值」的 lift 路徑，既有 dummy arm lift 邏輯保留。現有 `ours` 的 Stage 1 不追蹤手掌世界 X／Y 位移，所以此範例的 Phase 2 從抓取位、開手姿勢開始自適應閉手；Phase 1 仍保留獨立的接近動作。

**如何確認真的用了自適應控制：**同一筆 28 維 grasp 紀錄，先跑 Phase 1 的固定位置命令，再跑 Phase 2 的 `ours`。檢查輸出的每步目標關節位置、接觸／接觸力、Stage、最佳化結果與 solver 狀態；有接觸時，`ours` 的實際命令應能隨回饋改變，而非只是重播開合手軌跡。再檢查物體 lift 結果與統計，並跑一次既有 Shadow quick 範例確認沒有回歸。首次跑通只代表這個 CRX＋Sharpa 測試場景可用，不代表已完成實機控制或公開支援。

### Phase 2 已實作的執行方式

接續 Phase 1 印出的 run 目錄執行；請使用同一個目錄，且每個 run 目錄只執行一次 Phase 2：

```bash
PYTHONPATH=src python script/run_crx_sharpa_phase2.py output/examples/crx_sharpa/<Phase-1-run-ID>
```

程式會輸出 `control/ours_default/grasp_dist_0_pos_0.npy` 和 `phase2_summary.json`。實測一次完整 80 步：16 步有手物接觸，最多 7 個接觸點；接觸時最佳化器產生了不同於固定軌跡的命令，Stage 1／2 都有執行，沒有 solver degraded。物體最終沒有抬起，**因此這驗證了控制器接入與接觸回饋，沒有驗證抓取成功**。`hand.controller.final_sum_force=15 N` 目前沿用 Shadow 的數值作為實驗設定，尚未針對 Sharpa 校正。這個實驗腳本沒有產生公開 `control_eval` 的 batch report 或 `control_stat` 成功率，也沒有連接實體 ROS 命令。
