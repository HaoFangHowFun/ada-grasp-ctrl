# Shadow Phase 1／Phase 2 與 50-grasp 小規模測試計畫

日期：2026-11-01。狀態：**已完成首版入口與 50-grasp 配對 pilot；實際命令、結果和驗證限制見[實作結果](shadow-phase1-phase2-pilot-results.md)。** 下方保留原設計內容，擬定介面以實作結果文件為準。

## 1. 目標與建議方案

使用同一個 DexLearn Shadow checkpoint 產生的同一批抓取姿態，建立兩個可獨立執行、觀看及重播的環境：

- **Phase 1：固定位置命令。** 初始化至預測的 pregrasp，依序插值到 grasp、squeeze，不使用 Ada 接觸最佳化。
- **Phase 2：Ada 自適應控制。** 使用相同初始狀態與三個參考姿態，執行既有 `ours` 接觸回饋控制。
- 兩者之後均執行相同規則的 lift 評估，逐筆比較 grasp 結果。
- 第一批 **50 個 grasp × 2 個 phase = 100 個 episodes**。一個 grasp 包含三階段姿態，不是把 50 個 stage 當成 50 次抓取。

**建議優先重用現有 `op`／`ours` 和共用 runner，加一層薄的實驗管理與資料準備功能。** 不必再複製兩套 Shadow 模擬迴圈，也不必先把整個 Sharpa demo 重寫成大型通用框架。

首版以 Shadow＋dummy arm 和 HUGS-DexLearn 的右手預測為目標；共用函式保留 robot-specific 準備／lift 的邊界，後續讓 CRX-Sharpa 接入相同介面。這不代表將 CRX-Sharpa 加入公開支援矩陣。

## 2. 現有功能與 Sharpa commit 可借用的部分

| 來源 | 已有功能 | 本次用途 |
| --- | --- | --- |
| `cd50127` | Sharpa 固定目標 Phase 1 | 參考「準備輸入 → 開手／接近／閉手 → viewer／摘要」的使用方式 |
| `39d0b3a` | Sharpa Phase 2 接上 `ours`，新增 lift hook | 重用控制器適配方式，不重寫最佳化數學 |
| `be6755e`、`73dd28a` | viewer 視角、關閉碰撞網格顯示、結束保留畫面 | 抽出共用視覺化操作；物理碰撞保持啟用 |
| `58d4ba1` | Phase 2 獨立準備輸入與離線重播 | 兩個 phase 可獨立執行，皆讀同一份 prepared inputs |
| `b990d89` | Phase 2 live viewer | 使用者能直接看當次控制，而非必須先跑完重播 |
| 現有 `tabletop_dummy_arm_op.py` | 不依接觸回饋改變目標的位置控制 | 作為 Phase 1 的第一版實作 |
| 現有 `tabletop_dummy_arm_ours.py` | Ada 自適應控制 | 作為 Phase 2 |
| `BaseEval`、`DummyArmEpisodeRunner` | 場景、初始化、插值、取樣、step、記錄與 lift | 直接共用，避免 phase 各自實作而產生差異 |
| `run_ablation_baselines.py` | 各 case 隔離輸出、呼叫 eval/stat、收集 exit code | 優先沿用可用 helper；不足處才做小範圍抽取 |

`op` 目前仍建立 `GraspController` 物件，用它的插值與記錄等共用工具，**但 policy 不呼叫 `ctrl_opt()` 或 `check_wrench_balance()`**。這符合「沒有 Ada 自適應控制」的行為要求；MuJoCo position actuator 的底層 servo 仍存在。首版不為了消除 class 名稱而拆散共用執行流程。

Sharpa 舊 demo 的結果不能直接當作公平 A/B 樣板：Phase 1 從接近位開始，Phase 2 設定 `arm_pregrasp_is_grasp=true`，兩者的起點不同；Phase 1 也沒有相同的 lift／批次統計。新實驗要補齊這些比較契約。

## 3. 固定實驗協定

### 3.1 同一筆輸入、兩次獨立 reset

每個 `sample_id` 綁定原始預測、scene、三個轉換後姿態與一次 IK 結果。Phase 1、Phase 2 分別從相同 qpos、零初始速度、物體姿態、actuator 初始目標開始，不能讓 Phase 2 接續 Phase 1 的末態。

pregrasp 是模擬初始化位置；首版不額外引入 robot home → pregrasp 的動作。初始化後正常推進物理：`pregrasp → grasp → squeeze → lift`，不在每個 stage 直接重設 qpos。

Phase 1 是接觸回饋之外的固定參考軌跡；碰撞、接觸力仍會影響實際運動並被記錄，不是假裝手永遠精確走到目標。

### 3.2 明確保留 DexLearn 的三階段目標

現有預設會把 pregrasp 的手臂改成 grasp，並外推手指 pregrasp。為符合本次「直接執行三階段 grasp 指令」的需求，**新實驗入口的兩個 phase 都顯式設定**：

```yaml
task.arm_pregrasp_is_grasp: false
task.graspdata.pregrasp_t: 0
task.graspdata.squeeze_t: 0
task.offsets: [0.0]
```

這是此計畫提出的實驗 protocol，不修改 repo 的公開 defaults，也不把新結果冒充原 paper／quick baseline。記錄 stored targets 與 effective targets，檢查執行時沒有額外改寫。

**已確認的使用者要求：取消額外 squeeze 加深。** 執行端目前的外推公式是 `squeeze += squeeze_t * (squeeze - grasp)`（僅手指關節）；本批兩個 phase 明確固定 `squeeze_t=0`，不再加額外閉合角度、深度 offset 或倍率。保留模型原本預測的 squeeze，不將它改成 grasp。現在 repo 的 squeeze 預設已是 0，仍需在新入口顯式固定並測試，避免載入其他實驗設定時被覆寫。Phase 2 的 squeeze **參考目標**保持原值，但 Ada 依接觸回饋產生的實際命令仍可不同；取消的是人工外推，不是關閉自適應施力。

需特別追蹤一個已知限制：`ours` 原 Stage 1 的 hand-base X／Y tracking weights 是 0。保留完整 pregrasp 手臂接近動作後，Phase 2 可能不如 Phase 1 追蹤橫向位移。首版保持既有 objective，不偷偷改權重，也不只替 Phase 2 跳到 grasp。

先以 2～5 筆 smoke 檢查手掌參考／實際路徑。若 XY 接近明顯成為主因，另開明確命名的後續協定，例如兩者共用固定 approach 後再比較閉手控制；這屬不同實驗，不能與本批結果混算。

### 3.3 共同物理條件與不同控制策略

| 項目 | 兩個 phase 的約定 |
| --- | --- |
| Robot／asset | 同一份 Shadow＋dummy-arm XML、joint mapping、meshes |
| Object | 同一 scene pose、scale、collision meshes、質量設定 |
| Physics | 沿用共同 timestep、摩擦、重力／外力與初始化檢查 |
| Reference | 同一 pregrasp／grasp／squeeze，接近與閉手各 2 秒的 nominal reference |
| Seed | 固定 manifest 順序，兩者每筆 sample index／derived seed 一致 |
| Lift | 相同 0.2 m dummy-arm Z lift、插值規則與成功定義；起點各自取本次控制末態 |
| Phase 1 | `method=op`，固定位置目標，不跑接觸最佳化 |
| Phase 2 | `method=ours`，保留 Stage 1／Stage 2、力目標和 solver failure policy |

保留兩個方法原本的終止規則：`op` 通常 40 個 control steps；`ours` 最多 80 步並可能提早結束。必須報告實際模擬時間和步數，**首批比較的是原生兩個策略的整體表現，不是固定控制時長下的單一變因消融**。若後續要做等時長比較，再加入清楚定義的 hold／budget protocol，不先改 `ours` 科學設定。

實驗的 **Phase 1／Phase 2** 與 `ours` 內部的 **Stage 1／Stage 2** 分開命名，避免報告混淆。

## 4. 第一批 50 grasp 的來源與選擇

### 使用現有模型，重新產生一批資料

建議第一輪使用已實測能推理的：

```text
HUGS-DexLearn/output/
  shadowMulti_robotMultiHierar_right_full_100k_v1/ckpts/step_050000.pth
```

先固定一個 checkpoint，不把 100k／752k 混進同一個控制器比較。模型與資料設定保持：`robotMultiHierar`、`shadowMulti`、`right_full`、complete PC、1,024 points、WXYZ。

建議取 **10 個 test objects，每個選 1 個 scene，每個 scene 保留 5 個候選**：

1. 使用固定 seed，從 test split 中具備必要資產的物件／scene 建立明確白名單；不依抓取結果選 scene。
2. 將 10 個 scene 路徑固定為 manifest；HUGS 已支援 `test_scene_list_path`。使用清單時 `test_scene_num=0`，兩者互斥。
3. 每個 scene 生成 100 個 candidates，依固定且記錄的 selection policy 保留 5 個 `3_right_full`，共 50 個。
4. 記錄 1,000 個 proposal 的生成預算、50 個保留樣本、seed、checkpoint SHA-256、metadata 與點雲模式。top-k 是模型分數，不是物理成功標籤。
5. 若不足 50 筆或某 scene 生成失敗，明確回報缺額；不能默默換一批較容易的 scene 或補成功樣本。

50 grasp 是 smoke／pilot 規模；同一 scene 的 5 個候選相關，不視為 50 個獨立物件，也不據此宣稱統計顯著的泛化改善。

### 共用的資料轉換與 IK

沿用已驗證的 HUGS→Ada 契約：

```text
58-D HUGS record
  → 取右手 29-D
  → 依訓練 metadata 將手指重排為 Ada 順序
  → p_base = p_palm - R(q) @ [0, 0, 0.034]
  → Ada Learning format
  → CPU dummy-arm IK
  → prepared 28-D grasp record
```

模型輸出已恢復 world position，不再加點雲 centroid。先把既有調查用 [export_shadow_to_ada.py](../../reviews/ada_shadow_dexlearn_20261101/export_shadow_to_ada.py) 整理成具測試的單一格式 adapter，不寫成可猜測任意手型／frame 的 converter。

**只準備與求解一次 IK，兩個 phase 讀相同檔案 hash。** 格式、IK 失敗留在 50 筆選定樣本的 preparation report，作為 `preparation_error`／未評估原因；這是外層實驗欄位，不擅自增加 Ada 原生 `SampleStatus`。不把它們從總樣本表移除或替換。

## 5. 共用功能的實作邊界

先新增一個輕量 experiment module 和一個 CLI，沿用原有控制邏輯。以下名稱為**擬新增介面**：

| 函式／元件 | 責任 | 不應承擔的責任 |
| --- | --- | --- |
| `prepare_grasp_suite(...)` | source adapter、schema 驗證、一次 IK、固定 manifest | 不挑掉物理上失敗的抓取 |
| `build_phase_config(..., phase)` | 建立共同設定，僅選 `op`／`ours` 與 phase 輸出根目錄 | 不改原始 targets 或論文參數 |
| `run_phase_suite(..., phase)` | 呼叫現有 eval/stat、保存當次 reports、處理 exit code | 不複製控制器數學／MuJoCo step loop |
| `compare_phase_reports(...)` | 依 `sample_id` join 兩份當次結果、缺件檢查、配對表 | 不用資料夾掃描猜測本次輸出 |
| `configure_scene_viewer(...)` | 相機、visual/collision 顯示、結束保留畫面 | 不改物理 collision 屬性 |
| `record_episode_state(...)`／`replay_episode(...)` | 保存、播放實際狀態及時間戳 | 不重新最佳化或猜測未記錄的 lift |

預計程式位置：

```text
src/ada_grasp_ctrl/experiments/phase_comparison.py  # new orchestration
src/ada_grasp_ctrl/experiments/hugs_shadow_input.py # explicit input adapter
script/run_grasp_phases.py                         # new CLI
tests/test_phase_comparison.py                    # pairing/contracts
tests/test_hugs_shadow_input.py                   # frames/joint order
```

先使用現成 `METHOD_REGISTRY` 和 `SUPPORTED_HANDS`；不新增一個平行 task registry。`run_ablation_baselines.py` 的 subprocess／report helper 如需抽取，只做最小重構並保留原入口行為。

首版「通用」的範圍是 **準備後的 grasp、執行、報告、viewer**。Shadow 的 58→29 轉換與 dummy-arm IK，CRX 的固定手勢／六旋轉軸 IK，保留各自明確的 preparation function。CRX lift 繼續使用 `_lift_target_qpos()` override；不能套 Shadow 的 `qpos[2] += 0.2` 到 CRX J3。

待 Shadow pilot 完成後，再讓 Sharpa 既有 Phase 1／2 腳本轉呼叫共用 viewer／report／preparation helpers。第一輪不改 Sharpa 舊範例的數值或執行時序，以免同時擴大回歸範圍。

## 6. Viewer 與重播

- `--viewer` 用於單筆 live run；一次看 Phase 1 或 Phase 2，場景相機一致，結束後停在最後姿態。
- 50 筆 batch 預設 headless；結束後可依 sample ID 選看成功、失敗、兩 phase 結果不同的案例。
- 保留 native MuJoCo／既有 mjviser backend 的使用路徑；第一輪不用另外建網頁控制面板。
- 目前 control record 的 `dof` 沒有完整 lift 關節序列，Sharpa replay 也不重播 lift。要符合新的完整 demo，增加**可選的 recording hook**，由共用 step／lift 執行路徑記錄 sidecar。
- Sidecar 保存完整 simulator `qpos`、必要的 `qvel`／actuator `ctrl`、simulation time、joint names、phase／segment、reference target、實際發出的 command，以及場景模型／資產身份。包含 init、approach、squeeze、lift 和 terminal state；取樣節點明確區分 action 前後。
- 用 additive sidecar 避免破壞既有 control schema；未啟用 recorder 時原本路徑應無數值變化。Replay 直接恢復記錄狀態並 forward，不重跑 controller，不用最後一個 control frame 冒充 lift 結果。

## 7. 輸出結構與比較指標

```text
<fresh-run-root>/
  experiment_manifest.json
  preparation_report.json
  scene_manifest.json
  raw_predictions/
  adapted_learning/
  formatted/
  prepared_grasps/
  phase1/
    log/control_eval/run_report.json
    log/control_stat/run_report.json
    control/...
    replay/...
  phase2/
    log/control_eval/run_report.json
    log/control_stat/run_report.json
    control/...
    replay/...
  comparison.json
  comparison.csv
  summary.md
```

Phase 1／2 可各自讀取同一 suite manifest 執行；Phase 2 不依賴 Phase 1 是否已完成。各 phase 的既有輸出不可覆寫，重跑應建立新的 run root。首版不做複雜斷點續跑。

必備比較內容：

1. 每筆 checkpoint、scene、candidate、原始／prepared hash、seed、prep 狀態與兩 phase 的狀態。
2. `success / failure / invalid_initialization / solver_degraded / execution_error` 分別計數，保留缺少結果與 preparation error。
3. 原 Ada 的 `success / (success + failure)`，以及另外清楚標示的 `success / 50 selected grasps`。兩者分母不可混淆。
4. 配對結果：兩者成功、只有 Phase 1 成功、只有 Phase 2 成功、兩者皆未成功；另外交叉列出非正常狀態，不能把 degraded 偷算為普通失敗或成功。
5. 接觸步數／接觸力、手掌與手指 tracking error、squeeze 結束與 lift 後物體高度、實際控制步數／模擬時間／wall time。
6. Phase 2 的 solver acceptance、rejected solve 次數、apply／hold／abort 診斷；沿用現有 `apply_candidate` 預設，但被拒絕的候選即使施加也必須標 degraded。
7. 每個 scene／object 至少一筆成功的比例，作為 pilot 描述，避免只看候選總平均。

兩個 phase 都執行與現有 BaseEval 相同的 lift。若使用者只想觀看三階段到 squeeze，可在 viewer 選擇停止播放該段；不能因此把未做 lift 的軌跡報成已通過抓取成功判定。

## 8. 擬提供的操作方式

以下是**預計實作的命令介面，目前不可直接執行**；不是現有 repo 已有命令。

```bash
# Prepare once from the saved HUGS predictions and explicit metadata.
python script/run_grasp_phases.py prepare \
  --robot dummy_arm_shadow --source hugs-shadow-right-full \
  --input-root <saved-predictions> --metadata <training-metadata.json> \
  --selection-manifest <selected-50.json> --ik-device cpu \
  --output-root <new-suite-root>

# Run both policies against the exact same prepared inputs.
python script/run_grasp_phases.py run \
  --suite <suite-manifest.json> --phase both \
  --protocol predicted-three-pose --workers 1 --seed 12 \
  --output-root <new-run-root>

# Run one sample with a live viewer.
python script/run_grasp_phases.py run \
  --suite <suite-manifest.json> --phase phase2 --sample-id <id> \
  --protocol predicted-three-pose --viewer mujoco \
  --output-root <new-single-run-root>

# Replay an existing recording without rerunning control.
python script/run_grasp_phases.py replay \
  --run-root <finished-run-root> --phase phase1 --sample-id <id>
```

DexLearn sampling 使用既有 `HUGS-DexLearn/.venv` GPU 環境；Ada 準備／模擬使用本 repo `.conda/bin/python`，IK 設 CPU。兩個程序不共用 Python interpreter，也不把 ROS Python path 帶進 Ada。

## 9. 分階段實作與驗收

### A. 先用已有兩筆新預測接通兩個 phase

- 將已產生的兩筆 DexLearn 記錄作為小型 smoke，不再用 bundled 範例替代 learned inputs。
- 包裝現有 `op`／`ours`，兩者採本計畫的共同 raw-target protocol。
- 各產生完整 control/stat report；有接觸與實際 command 的證據。
- 確認 Phase 1 不呼叫 contact optimizer，Phase 2 確實呼叫；無接觸時命令可能接近固定軌跡，不強迫每一步都不同。

### B. 固定 50 筆 suite 與批次比較

- 建立 10-scene whitelist，重新推理並固定 50 個 sample IDs。
- 先跑固定的 5 筆做 tracking／初始化診斷，再對同一份 50 筆 manifest 跑兩個 phase。
- 50 個 selected IDs 在 comparison table 各有一列；準備失敗、未執行或無效初始化都有明確原因。若全部準備成功，應恰有 100 個 episode 結果。
- 批次中的單筆失敗不阻止另一個 phase 或其餘樣本；流程共享資產／設定缺失則 preflight 失敗。

### C. 共用 viewer／完整 replay

- 補齊 sidecar recorder 和 lift state，提供單筆 live view 與離線 replay。
- 確認 recorder 開關前後控制結果／既有欄位不變，replay 不寫回原輸出。
- 之後才將適合的 helper 接回 Sharpa 舊入口；robot-specific IK 和 asset preparation 保持獨立。

### 必要測試

- 右手切片、metadata 重排、旋轉後 34 mm offset；MuJoCo FK 對回原預測 palm。
- 同一 suite 的兩個 phase：初始 qpos／qvel／object pose、prepared hash、sample seed、effective targets 一致。
- Phase 1 將 `ctrl_opt()`／`check_wrench_balance()` 替換為會報錯的 stub 仍可執行；Phase 2 spy 確認 optimizer 與 diagnostics 路徑有被呼叫。
- invalid／degraded／execution error／preparation error／missing result 不被報成 success，兩份 stats 只讀自己當次 eval report。
- viewer lifecycle、replay terminal／lift state、sidecar 時間順序與狀態重播。
- 改到 shared runtime／controller recording 時，跑完整 unit suite、compile、Ruff 和 maintained quick gate；不改 expected inventory 或容差。

## 10. 已知限制與完成標準

目前環境已有 124 個測試通過、bundled 單筆 full 成功；但 Shadow quick 的三筆歷史分類差異尚未定位，兩次本機 quick 分類一致。詳細見 [環境文件](local-environment-2026-11-01.md)。新工作不得把這三筆差異宣稱為已修好，也不得生成新的 golden 蓋過它們；shared-function 重構前後應另作同環境比較，區分既有差異與新回歸。

本次完成標準是：**同一批真實 DexLearn 預測能在無接觸最佳化／有 Ada 接觸最佳化兩種模式下執行，50 筆結果完整可配對，個案能看、能重播，設定與失敗原因可追溯。** 不把「Phase 2 一定比 Phase 1 高成功率」當成工程驗收條件；若沒有改善，報告要能指出是初始穿透、接近 tracking、閉手接觸、solver，還是 lift 階段的問題。

第一版不包含重新訓練、真實相機／ROS／CRX 實機控制、改摩擦／力目標、一次支援所有手型，也不先做廣泛的模型／控制參數搜尋。先把這個 50-grasp paired pilot 做清楚，再決定下一輪要比較 752k checkpoint、位置擾動，或不同 approach protocol。

## 原始碼參考

- [Open-loop policy](../src/ada_grasp_ctrl/tasks/control_eval_func/tabletop_dummy_arm_op.py)
- [Ours policy](../src/ada_grasp_ctrl/tasks/control_eval_func/tabletop_dummy_arm_ours.py)、[optimized runner](../src/ada_grasp_ctrl/tasks/control_eval_func/optimized_runner.py)
- [Shared episode runner](../src/ada_grasp_ctrl/tasks/control_eval_func/episode_runner.py)、[BaseEval](../src/ada_grasp_ctrl/tasks/control_eval_func/base.py)
- [Sharpa Phase 1](../src/ada_grasp_ctrl/crx_sharpa_demo.py)、[Sharpa Phase 2／replay](../src/ada_grasp_ctrl/crx_sharpa_adaptive.py)
- [Batch matrix runner](../script/run_ablation_baselines.py)
- [DexLearn 接入調查](shadow-dummy-arm-dexlearn-investigation-2026-11-01.md)
