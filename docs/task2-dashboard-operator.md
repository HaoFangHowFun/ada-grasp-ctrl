# 8089 即時監看

由專案目錄執行原本的一行命令（這會執行實機抓取）：

```bash
python3 script/run_grasp_pipeline.py --controller-mode mpc_active
```

瀏覽器開啟 <http://localhost:8089>。同一頁從準備階段更新到 Return，沒有網頁或 Enter 確認。
成功／失敗後保留最後狀態約 5 秒，再由入口關閉本次 viewer 和讀取程序。

- CRX 六軸與 Sharpa 22 關節使用實際 ROS feedback。只畫外觀網格；碰撞檢查不變。
- 桌面使用既有校正。`Static pre-grasp object cloud` 是本次選取物體的原始點雲，只載入一次；
  沿用 candidate preview 的已錄製表面網格＋彩色點與固定標籤；Lift 後仍停在原位，不表示物體追蹤。感知尚未完成時顯示 `Waiting for perception`。
- 紅點是可解讀的指尖接觸位置；橘色是施加到物體的力，綠色是指尖朝物體的表面法向。
  力箭頭長度與力成正比，滑桿只調整顯示比例。滑鼠可旋轉、平移和縮放場景。
- 左上角獨立顯示 **Live contact ΣFn**（即時感測器估計）和 **Grasp complete ΣFn**
  （`native_complete` 當下的實際總力，之後固定保存；完成前顯示 `—`）。
  兩者都是各指法向力加總，不是三維向量合力。
- 五指 `Fn` 和總和單位為 N。`MONITOR ESTIMATE` 是感測器推算；
  `MPC completion force` 來自既有 native event，並分別標記即時或歷史值。
  **15 N 是 native 完成門檻，不是力上限**；超過 15 N 不會截斷數值。
- Lift、Hold、Release、Return 顯示 `MPC inactive`，仍持續更新感測器。
  這些階段沒有新增 MPC 力控制。接觸幾何／力校正仍沿用既有 provisional 診斷政策。
- 缺資料、多接觸無法解讀、過期或斷線時顯示 `—`／`PARTIAL`／`UNAVAILABLE`，不當成零力；
  對應箭頭會隱藏。關節過期時保留最後姿態並明確標記，不當成即時姿態。

網頁沒有動作控制。重新整理、斷線或 viewer 執行中故障不會取消控制程序；
安全與中止仍由原本 receiver/controller 負責。Ctrl+C 在入口仍按原有程序取消並清理。
`linear`、`mpc_passive` 與手動預覽沿用原流程；本 dashboard 只接到自動實機 `mpc_active`。

遇到 telemetry loss，查看本次 `viewer.log`，確認 drivers、ROS domain 87 及關節／觸覺 topic 正常；
模型不符會拒絕顯示。不要用舊資料或目標姿態代替 feedback。
8089 被占用時入口會明確停止；先關閉舊 viewer，不會偷偷改用 8090。
本次 `dashboard_state.json` 與 `dashboard_metrics.json` 可供離線排錯。

離線展示不連 ROS、不動硬體（每次指定新的輸出目錄）：

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .conda/bin/python \
  -m ada_grasp_ctrl.pipeline.dashboard_replay \
  --state /tmp/ada-pear-check-uz6okdpn/state_live/robot_state.json \
  --object-root /tmp/ada-pear-check-uz6okdpn/grouplive/object \
  --output-root /tmp/ada-dashboard-demo-new --seconds 15
```

上面的本機錄製檔須仍存在。Replay 明確標示 synthetic，關節與觸覺是假資料，
桌面和物體使用既有錄製資產。結果見 [Task 2 驗證紀錄](task2-dashboard-results-2026-11-13.md)。
