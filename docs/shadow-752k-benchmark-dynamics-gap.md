# 完整對齊初始化後，Ada 與 benchmark 的動態差異

日期：2026-11-01。

## 後續實驗基準

依使用者決定，後續 Shadow 752k Phase 1／Phase 2 對照採「完整對齊初始化」：pregrasp 平移及手指外推倍率 2、quaternion 不變、初始穿透門檻 10 mm、不使用初始力拒絕。grasp 與 squeeze 保留模型輸出，squeeze 倍率 1。正式基準為 `ada-benchmark-init-v1/matched` 的 34 成功、16 失敗、0 初始拒絕；不是先前的 22/50 配置。這是本系列實驗協定，沒有改動 repo 公開 defaults。

## 本次做了什麼

同一批 50 筆、相同 checkpoint／scene／尺度／targets。額外在原生 DexGraspBench 執行器外加被動記錄，保存每 40 ms 的完整 qpos、qvel、ctrl、act、mocap 與物體 pose；沒有改寫控制指令或物理參數。50 筆分類與既有 historical run 完全相同（42 成功、8 失敗）；最終位置／角度誤差也通過 `rtol=1e-5, atol=1e-6` 比對。

對照既有 Ada 50 筆完整初始化回放，8 筆差異仍是：g0003、g0016、g0018、g0019、g0042、g0043、g0044、g0046。它們都是 benchmark 成功、Ada 失敗。

## 已確認的失敗型態

![八筆差異案例的實際抬升軌跡](../output/shadow-phases/dexlearn-752k-three-scales-v1/matched-dynamics-audit-v3/lift_disagreements.png)

圖橫軸是開始 lift 後的時間，縱軸是物體相對初態的升高。Ada 命令抬高 200 mm／2 秒；benchmark 100 mm／1 秒，平均命令速度同為 100 mm/s。曲線為離散記錄點，最大值不是連續時間精確最大值。

| 樣本 | Ada 最大升高 mm | Benchmark 最終升高 mm | Benchmark 質量 g | Ada 抬升約 100 mm 時套用 benchmark 判準 |
| --- | ---: | ---: | ---: | --- |
| g0003 | 5.8 | 97.7 | 66.0 | 失敗 |
| g0016 | 12.3 | 97.7 | 52.8 | 失敗 |
| g0018 | 92.4 | 99.9 | 52.8 | 失敗：角度誤差 18.12° >15° |
| g0019 | 7.4 | 103.1 | 52.8 | 失敗 |
| g0042 | 6.3 | 95.5 | 118.4 | 失敗 |
| g0043 | 5.1 | 94.5 | 118.4 | 失敗 |
| g0044 | 7.2 | 94.0 | 118.4 | 失敗 |
| g0046 | 165.1 | 95.9 | 324.1 | 通過；繼續抬升後掉落 |

Ada 這些物體質量均為 100 g。六筆在 lift 前段即失去抓持；g0018 能抬起但姿態已偏移，接著掉落；g0046 則是較長 lift 才暴露滑落。故「Ada 抬得更高／更久」能解釋一部分觀察，無法單獨解釋全部 8 筆。

中途判定只是在既有 Ada 軌跡的 t≈5 s 套用「期望初始位置 +100 mm、位置誤差 <50 mm、角度誤差 <15°」；不是實際重跑縮短 lift，也不能將此診斷數字當正式新成功率。整批如此觀察為 33/50，並不會直接變成 benchmark 的 42/50。

## 最值得隔離的差異

### 1. Tendon 改變的是接觸後的運動自由度與力分配

編譯後模型顯示 Ada 有 22 個手指 actuator，加 6 個 dummy-arm actuator；benchmark 有 18 個手指 actuator、4 條 tendon，base 以 mocap 控制。兩邊手指 actuator 都是 kp=5、forcerange=[-3,3]，但這不代表控制行為相同。

對食／中／無名／小指，benchmark 控制 J1+J2 的總和，兩個關節可以在接觸物體後重新分配彎曲角度；Ada 各自控制 J1 與 J2。忽略限幅時，獨立關節的回復力矩分別與 `5e1`、`5e2` 成正比；固定 tendon 的共同驅動則與 `5(e1+e2)` 成正比。相同 squeeze 角度輸入，在接觸約束下會產生不同實際姿態與力分配。

具體例子 g0016：squeeze 結束時，Ada 的有效手物接觸集中在中指與拇指；benchmark 同時出現食指、無名指與小指接觸。手指相對目標的 RMSE，Ada 約 0.078 rad、benchmark 約 0.213 rad，後者反而成功。對 tendon 模型而言，單關節偏離輸入角度並不必然是控制失準，可能是維持總彎曲量下的接觸適應。

這是驅動差異的重要線索，仍不能從觀察直接宣稱 tendon 是 8 筆失敗的唯一原因。

### 2. 摩擦不同，而且不能用總法向力直接推成功

兩邊實際 collision geom 摩擦分別為 Ada `[0.6, 0.02, 0.0001]`、benchmark `[0.8, 0.02, 0.0001]`。相同法向負載下，Ada 的名義滑動摩擦上限是 benchmark 的 75%，容易影響臨界抓持。

但這批並非「Ada 總夾力都比較小」。以 squeeze 末態離線重算手物接觸法向力總和：g0016 約 30.0 vs 33.5 N，g0042 約 25.6 vs 21.8 N，g0046 約 21.7 vs 20.4 N（Ada vs benchmark）。總力相近甚至更高仍可能失敗，因為接觸位置、力方向、包覆程度、摩擦與重力矩一起決定抓持。

這些力是把已記錄的 qpos/qvel/ctrl/act/mocap 與物體外力放回各自模型後 `mj_forward` 重算的邊界值，不是完整原始 solver warm-start 狀態下的高頻力訊號；只作定性診斷，不能視為兩邊夾持力優劣的獨立因果證明。

### 3. 質量有影響，但不是全都對 Ada 不利

g0003、g0016、g0018、g0019 的 benchmark 物體較輕，Ada 固定 100 g 的確增加負載。但 g0042–g0044 的 benchmark 約 118 g、g0046 約 324 g，benchmark 更重仍成功。因此不能只把質量改輕，就預期解決所有差異。慣量也隨模型質量設定改變。

### 4. 時序與 base 控制仍有差異

Ada approach／squeeze 各 2 秒，benchmark 各 1 秒；Ada 高層 waypoints 10 Hz，再以 20 ms 內插致動器命令，benchmark mocap／actuator waypoints 25 Hz。最後 lift 2 秒／200 mm vs 1 秒／100 mm。不同接近路徑時序與 base 約束可在 squeeze 前形成不同接觸狀態。

不過 8 筆在 squeeze 結束時的兩邊 palm 位置差只有約 0.13–0.84 mm，沒有看到厘米級腕部位置錯配。這降低了「末端 wrist 目標大幅錯位」的嫌疑，但不排除過程中的旋轉、追蹤與接觸交互作用。

## 已排除或不能成立的解釋

- **Lift 沒有把 squeeze 指令鬆開。** Ada 的 `get_qpos_a()` 實際回傳 target command cache。50 筆兩邊全部 lift frame 的手指 ctrl 相對最後 squeeze ctrl 最大差都為 0。先看函式名稱而懷疑採用實測角度的猜測已被程式與記錄否定。
- **不是手指 kp 或 force limit 不同。** 編譯後均為 5 與 ±3；關鍵是獨立關節與 tendon 作用方式不同。
- **不是 timestep 或主要 solver 選項不同。** 代表模型兩邊均為 timestep 0.004、implicitfast、Newton、elliptic cone、impratio 10、iterations 100、tolerance 1e-8，停用內建重力並對物體施加等效外力。
- **初始化已對齊**，前次驗證 palm／手物初始距離差在約 1e-9 m 級。這次沒有重新增加 squeeze。

## 建議下一輪因果對照

觀測已縮小範圍，但還沒有量到每個因素對成功數的獨立貢獻。後續全部使用上述完整初始化、固定 50 筆和來源 hashes，每個新條件使用獨立輸出目錄，保留現有 34/50 基準與 8081 回放。

1. **摩擦單因子**：僅把 Ada 滑動摩擦 0.6 → 0.8，其餘不動。成本最低，先檢查 8 筆差異能恢復哪些，同時查看原本 34 成功是否退步。
2. **質量／慣量單因子**：逐 scene 使用 benchmark 已保存的實際質量，不能只統一改成某個重量；物體慣量與等效重力同步保持一致。
3. **Tendon 對照**：在獨立實驗模型用相同四條 tendon 替代相應獨立 actuator，保留 dummy arm、增益、摩擦和時間；需要檢查 DOF→actuator mapping，不能直接套用現有 28 維 actuator 假設。
4. **時序對照**：拆開 approach/squeeze 時間與 lift 距離／時間，避免一次全改無法辨識原因。先保留正式 Ada 成功判準，任何改判準的數字另外標示。

本次只做被動記錄與分析，沒有執行以上物理／控制條件改動。專案 `AGENTS.md` 的 “Changing paper hyperparameters, friction, controller objectives, lift thresholds, stage semantics, or success-rate definitions” 屬 Ask first 範圍；因此需確認具體對照範圍後才執行，不應直接把它們升格為新預設。

## 產物與驗證

正式診斷輸出：`output/shadow-phases/dexlearn-752k-three-scales-v1/matched-dynamics-audit-v3/`。

- `benchmark_report.json`：50 筆原生重跑結果與來源 hash。
- `benchmark_config.yaml`：沿用 historical 設定，僅另存輸出與解析資產絕對路徑。
- `benchmark/*.replay.npz`、`*.scene.xml`：每筆 76 frames，包含 mocap 狀態。
- `dynamics_comparison.json`、`summary.json`：逐筆末態、接觸、100 mm 時點判定與 lift 指令檢查。
- `model_parameters.json`：編譯後物理選項、增益、限幅、tendon 數量。
- `lift_disagreements.png`／`.svg`：可獨立匯出的八案例曲線。

工作區腳本：`reviews/ada_shadow_phases_20261101/{record_benchmark_dynamics,analyze_matched_dynamics,plot_matched_dynamics}.py`。早期 v1 因 dataset root 缺失中止；v2 回放未保存 mocap，不能用於接觸力重建。以上結論一律使用補齊 mocap 且已重新驗證結果的 v3。
