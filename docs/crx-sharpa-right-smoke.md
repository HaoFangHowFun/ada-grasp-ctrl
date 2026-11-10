# 右 CRX＋右 Sharpa smoke 測試

使用者已確認代表物體中心 X=0.8、Y=-0.7 m 與 config 初始關節姿態。
此入口使用已確認預覽的場景、prepared targets 和 frozen suite，僅執行 MuJoCo。
目前提供四尺度各一筆 × op／ours；完整 1,000 × 2 模式尚未提供。

## 執行

```bash
bash script/test_crx_sharpa_right_controller.sh --smoke

# 僅驗證來源、模型、控制器建構、IK 與靜態 home 路徑，不推進物理
bash script/test_crx_sharpa_right_controller.sh --prepare-only

# 明確指定已確認預覽與全新輸出目錄
bash script/test_crx_sharpa_right_controller.sh --smoke --workers 1 \
  --preview-root output/regression/crx-sharpa-right-preview-x080-y070-config-v1 \
  --output-root output/regression/my-new-right-smoke
```

shell 可從任意目錄以絕對路徑呼叫；相對輸入／輸出路徑以 repository 為基準。
Python 預設 `.conda/bin/python`，可由 `PYTHON_BIN` 覆寫；清除繼承的 Python module 搜尋路徑。
預設每次建立新的 timestamp 目錄，拒絕覆寫。來源 suite 由已確認的 preview 綁定，
不另提供可悄悄替換資料的 `--suite-root`。僅驗證 serial 執行，因此 workers 限定 1。

退出碼沿用專案語意：0 表示執行完成（可包含 physical failure／invalid），
1 表示有 execution error 或 solver degradation，2 表示前置輸入／環境錯誤。
**退出 0 不代表抓取成功或 smoke 驗收通過**；需查看 `smoke_acceptance_passed` 與逐筆狀態。

## 控制與記錄

- 初始 28 軸直接使用預覽中的 `initial_qpos`；右臂來源為 `right_arm_hand_task.yaml`
  的 `offline_seed`，J1–J6 為 `[-90, -30, 210, 0, -60, 0]` 度所對應弧度。
  不做示教器 J3 換算。手指使用具名 `open_hand`。
- 兩方法共享 config→pregrasp 的關節線性參考，至少 2 秒，依關節速度上限延長。
  先抽查 61 個靜態路徑姿態；這不是連續無碰撞證明。
  動態接近過程也記錄碰撞；保存實際追蹤誤差，未假定已到達參考。
- op 與 ours 分別使用原生 open-loop／coordinated policy，重用 shared episode runner。
  保留既有 approach／squeeze 參考、ours 終止規則及 solver `apply_candidate` 政策。
  拒絕的有限 solver candidate 即使按政策套用，該 episode 仍是 degraded。
- 全域重力關閉，以物體向下 1 N 模擬既有 benchmark 重力；保留 mass 設定 0.1 kg、
  friction `[0.8, 0.02]`、手部 kp=7.5、force target 15 N、offset=0、10 Hz 控制。
  CRX 使用已確認模型的暫定 kp=100、kv=10，沒有為本次結果改 gain。
- 與 dummy arm 的必要差異：實際 CRX 關節／限制／IK、config→pregrasp 過程、
  保留手臂及手部對桌面的接觸、Cartesian lift。原 dummy benchmark 的桌面接觸排除不能套用。
  手物接觸只允許右手 body 集合進入控制器，手臂碰撞另存。
- Lift 從實際 flange 位姿向 world +Z 0.2 m，以 20 個 IK 目標執行 2 秒，手指保持命令。
  每點檢查 IK、手臂穿桌與最大單關節分支步長 0.35 rad；不使用 `q[2] += 0.2`。
- 成功判定仍是 `abs(object_z_at_control_start + 0.2 - terminal_object_z) < 0.1`。
  統計分類互斥：execution error、initialization invalid、solver degraded、IK invalid，
  最後才對完成 lift 的結果分類 success／physical failure。前置 IK invalid 不進物理。
  runtime lift IK error 在原生 `control.npy` 中使用 execution_error 相容狀態，
  實驗 `result.json` 則保留 ik_invalid；分析本實驗須使用 `run_report.json`。

輸出含 resolved `run_manifest.yaml`、`code_state.json`、`preflight.json`、逐筆 `result.json`、
`results.json`、`run_report.json`、`summary.md`、`failures.jsonl`。
實際執行者另存 `control.npy`、`control.scene.xml`、`control.replay.npz`、`collisions.json`，
包含 home、控制、lift、terminal 的實際狀態；invalid 保留 sample ID 與原因，不補抽。
每對執行後核對初始／home／control-start 的 qpos、qvel、ctrl、act、物體位姿及 targets 完全相同。

## 2026-11-09 實際結果

修正前 v1 測試報告：
`output/regression/crx-sharpa-right-smoke/20261109T231719970723Z-2201343/run_report.json`。
本輪 code hashes 已保存；執行時基礎 commit 為 `cd7f094`，runner 當時是尚未 commit 的新增程式。

| 樣本 | scale | op | ours | 原因 |
| --- | --- | --- | --- | --- |
| g0000 | 0.060 | physical_failure | physical_failure | 完成 lift，但無手物接觸、物體留在桌面 |
| g0010 | 0.081 | initialization_invalid | initialization_invalid | frozen targets 有手臂穿桌 |
| g0020 | 0.106 | ik_invalid | ik_invalid | 原 IK 位置 residual 約 29–31 mm |
| g0030 | 0.133 | initialization_invalid | initialization_invalid | home 路徑第 8/60 點手物穿透約 18 mm |

8 個預期 pairs 均有狀態，missing=0；實際物理執行 2 次。
三目標 IK 可達為 3/4 samples，通過本次路徑／初始化 preflight 為 1/4。
兩方法各 success/selected=0/4，success/(success+physical_failure)=0/1。
execution_error=0、solver_degraded=0；g0000 兩方法的完整初始化狀態核對一致。

op 執行 40 個控制步、2,000 個 physics steps；ours 執行 80 個控制步、3,000 個 physics steps。
兩者都有 20 個 lift 目標，沒有記錄到手物接觸或手臂／桌面碰撞。
物體高度差分別約 -2.50e-6 m 與 +3.47e-6 m，未抓起物體。
home 結束最大關節追蹤誤差約 0.517 rad；squeeze 結束 op 約 0.405 rad、ours 約 0.133 rad，
主要落在 J1。這是實際追蹤誤差證據，尚未證明成因；沒有直接調 gain 或改時間以追求通過。

`smoke_acceptance_passed=false`：尚未驗證接觸下的 ours 自適應反應，四尺度也未全部完成控制。
因此尚未啟動完整 2,000 筆；此小樣本不可用來估計完整 suite 成功率。
後續需處理關節追蹤、home 路徑及可達性；影響擺位的修改須重新預覽確認，
控制參數或時序修改須先確認新的 protocol。

驗證：16 項右側 preparation／runner 測試通過；完整單元測試 172 項通過。
全專案 Ruff check、compileall、新 shell 語法檢查及四個新增 Python 檔案 format check 通過。
沒有修改共用 controller 或公開 hand registry；既有 Shadow quick 結果記錄於預覽文件，
本輪不宣稱重新完成完整三手型 release gate。

## 底座／J1 接觸修正後的完整抓取與抬升

後續逐步診斷確認，最初 2 秒 home→pregrasp 的實際手掌會往負 Y 遠離物體，
雖然同時的命令 FK 朝物體靠近。`right_base_link` 與 `right_J1_link` 在軸承介面接觸，
產生抵消 J1 馬達力矩的約束阻力；固定於 world 的父 body 不會獲得一般的父子碰撞過濾。
這是接近階段的碰撞建模問題，不能歸因於何時開始抓取，也不是原始目標的 Y 平移錯誤。

使用者要求套用該修正後跑完整抓取／抬升。實驗 adapter 現在只新增這一對相鄰 body 的
contact exclusion，驗證它們確實為父子關係；保留手臂／桌面／物體的其他碰撞。
已確認的預覽檔案、placement、關節、IK targets、控制增益、摩擦、時序和成功門檻未修改。
新 protocol 名稱為 `crx-sharpa-right-smoke-v2-base-j1-exclusion`；manifest 明列新增 exclusion，
每筆回放 scene 保存實際 exclusion，`result.json` 保存該 scene 的 SHA-256。
舊的 v1 結果保留，不可把兩種碰撞設定混為同一輪。

本輪輸出：`output/regression/crx-sharpa-right-smoke/20261110T001318350251Z-5052/run_report.json`。

| g0000 方法 | 接觸控制步數 | 抬升目標步數 | 物體最大高度增量 | 結果 |
| --- | ---: | ---: | ---: | --- |
| op | 32 / 40 | 20 | 約 0.12 mm | physical_failure |
| ours | 67 / 80 | 20 | 約 0.89 mm | physical_failure |

兩筆均完整執行接近、抓取、Cartesian lift 和 terminal；物體最後仍在桌面。
兩者沒有 execution error 或 solver degradation，沒有記錄到手臂穿透或超標手／桌穿透事件。
ours 的 67 個接觸步記錄到命令偏離固定參考，通過 runner 的自適應命令變化檢查；
這項檢查本身不代表抓取成功。op／ours 初始及 home 狀態一致。
home 結束的最大關節追蹤誤差從舊版約 0.517 rad 降至約 0.084 rad。

其餘固定尺度依舊為 4 筆 initialization_invalid 與 2 筆 ik_invalid；總計 8 筆狀態、
missing=0、實際物理執行 2 筆。完整 2,000 筆未啟動，四尺度 smoke 驗收仍未通過。
本輪 18 項右側測試、全專案 176 項單元測試、Ruff check、變更檔案 format、compileall、
shell 語法檢查均通過；測試涵蓋 exclusion 的冪等性、父子關係、FK 不變及桌面接觸保留。


## 重開回放

```bash
bash script/replay_crx_sharpa_right.sh \
  output/regression/crx-sharpa-right-smoke/20261110T001318350251Z-5052 \
  --sample-id g0000 --port 8086
```

開啟 `http://localhost:8086`，預設 0.5 倍速循環播放 op／ours。
可切換方法、播放／暫停、拖曳進度、重新播放及切換全景／手物近景。
碰撞殼預設關閉；勾選 `Collision geometry` 才顯示，透過同步可見性更新網頁物件。
這只控制畫面，不修改碰撞計算或已保存軌跡。回放不執行新的物理步進。
必須指定同時有 op／ours 回放的樣本；缺失會拒絕啟動。Ctrl+C 關閉 server。
