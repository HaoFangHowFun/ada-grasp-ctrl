# Shadow Phase 1／Phase 2：實作與 50-grasp pilot 結果

日期：2026-11-01。對應[實作計畫](shadow-phase1-phase2-pilot-plan.md)。

**後續尺度調查更正：此批有 25/50 筆的 scene scale 落在 right_full 訓練尺度集合之外，抽樣分布也偏大。物體尺寸與原始 scene 一致，未發現 Ada／viewer 重複縮放；這批成功率不能視為正常訓練尺度分布下的表現。詳見[物體尺度調查](shadow-pilot-object-scale-audit.md)。**

已完成資料準備、兩個 phase、配對報告、單筆 live viewer 與含 lift 的離線 replay。**這次使用新產生的 DexLearn 預測，不是 bundled grasp，也不是手工指定手勢。** Phase 1 使用原生 `op`；Phase 2 使用原生 `ours` Ada 接觸最佳化。

## 結果

固定 10 個 test objects、每個 1 個 scene、每個保留 5 個候選。所有 50 筆都完成一次 CPU IK，兩個 phase 各評估 50 筆，沒有替換無效或失敗案例。

| 結果 | Phase 1：op | Phase 2：ours |
| --- | ---: | ---: |
| success | 2 | 2 |
| failure | 10 | 8 |
| invalid_initialization | 38 | 38 |
| solver_degraded | 0 | 2 |
| execution_error／preparation_error／missing | 0 | 0 |
| success / (success + failure) | 2/12 = 16.7% | 2/10 = 20.0% |
| success / 50 selected | 4.0% | 4.0% |
| 至少一筆成功的 scene／object | 1/10 | 1/10 |

配對結果：2 筆兩者成功、8 筆兩者 failure、2 筆 Phase 1 failure／Phase 2 degraded、38 筆兩者初始化無效。**沒有只在 Phase 2 成功的案例。** 原生有效案例成功率分母排除 degraded，因此不能把 16.7% → 20% 解讀成改善；兩者實際成功的是同兩筆。

成功樣本為 `g0025`、`g0027`，同屬 `ddg_gd_tetra_pak_poisson_022`。逐物件結果與全部 ID 保存在下方報告，不把 50 個候選視為 50 個獨立物件。

## 固定的控制協定

兩個 phase 都明確設定：

```yaml
task.arm_pregrasp_is_grasp: false
task.graspdata.pregrasp_t: 0
task.graspdata.squeeze_t: 0
task.offsets: [0.0]
```

**沒有額外 squeeze 加深、閉合倍率或深度 offset。** 保留 DexLearn 原本三個不同姿態；不將 squeeze 改成 grasp。Phase 2 仍會根據接觸回饋改變實際命令。

pregrasp 是初始化姿態，之後沿原生 runner 執行 approach → squeeze → 0.2 m lift。Phase 1 有 40 個 control steps、含 lift 共 6 秒模擬時間；Phase 2 此批有效初始化案例為 42、60 或 80 步，含 lift 為 6.2、8 或 10 秒。這不是等時長消融。公開 defaults、接觸條件、摩擦、最佳化目標與成功判定都未改動。

`op` 仍使用 `GraspController` 的插值與記錄工具，但不呼叫 `ctrl_opt()`／`check_wrench_balance()`；已用會拋錯的 stub 配合真實 MuJoCo episode 驗證。`ours` 的兩個方法則用 spy 驗證確實被呼叫。

## 目前瓶頸

- **初始化無效佔 76%。** 38 筆中，37 筆手／物體距離小於 −5 mm，另 1 筆因初始接觸力超過 1 N。這些案例尚未進入 approach／squeeze，就被既有檢查擋下。兩個 phase 的原因和數值一致。不能歸因於 squeeze 加深。
- **部分有效初始化案例完全沒有接觸。** Phase 1 的 12 筆可執行案例中，7 筆在 40 步內沒有手／物體接觸。模型的目標與物體幾何／位置關係值得優先檢查。
- **Phase 2 的 XY 接近 tracking 有差異。** 例如 `g0003`：Phase 1 有 13 個接觸步、Phase 2 沒有；Phase 2 palm X/Y tracking RMSE 約 5.96/5.55 mm。既有 `ours` 的 base X/Y tracking weights 是 0，此輪沒有修改。這是值得進一步檢查的線索，不是已隔離證實的唯一原因。
- **solver degraded 保留為獨立狀態。** `g0022` 與 `g0042` 分別有 39 與 3 次 rejected solves；沿用原本 `apply_candidate` policy，詳細 acceptance／decision 留在報告。兩筆均未達 lift 高度，仍不把 degraded 混入普通 failure。

上述穿透是在 **Ada 目前的碰撞模型** 中量測。已排除明顯的 joint-order／palm-frame 接入錯誤，但尚未逐項確認 DexLearn 訓練環境與 Ada 的 collision mesh、接觸排除和 pregrasp 定義完全相同。因此下一步優先觀看無效初態與無接觸案例、對照兩邊模型，再決定是否需要改資料／模型或另立共用 approach 協定；不先增加 squeeze 或放寬無效初始化門檻。

## 模型與資料追溯

- Checkpoint：`HUGS-DexLearn/output/shadowMulti_robotMultiHierar_right_full_100k_v1/ckpts/step_050000.pth`。
- 推理：`robotMultiHierar`、`shadowMulti`、`right_full`、complete PC、1,024 points；seed 42。
- 10 個 scene 在推理前以 seed 固定；只依 scene／PC／資產可用性選取，未依物理成功結果過濾。
- 每 scene 生成 100 個候選，依 `sample_selection.scope=per_type`、`mode=topk` 的 diffusion log probability 保留 5 個。總預算 1,000 proposals，保留 50。
- Ada seed 12；sample index 由 suite 固定。單筆 viewer 重跑仍保留原 index，不重新從 0 編號。
- HUGS 的 58-D 三階段輸出取右手、按 metadata 重排 22 個手指關節，再做 `p_base = p_palm - R(wxyz) @ [0, 0, 0.034]`。不再次加 centroid。
- 模型 SHA、scene／PC SHA 與選擇規則見 [scene manifest](../output/shadow-phases/pilot-source/scene_manifest.json)。推理完整 Hydra 設定保留在 sampling run 的 `.hydra/config.yaml` 與 `overrides.yaml`。

## 可直接使用的入口

在 `ada-grasp-ctrl` 根目錄執行，使用已補齊的 `.conda`；避免帶入 ROS 的 Python 路徑。首版支援 Shadow right_full，serial worker 固定為 1。

```bash
# 觀看一筆真正執行 Ada 的成功案例；使用新的輸出目錄。
env -u PYTHONPATH -u PYTHONHOME .conda/bin/python script/run_grasp_phases.py run \
  --suite output/shadow-phases/pilot-suite/suite_manifest.json \
  --phase phase2 --sample-id g0025 --viewer mujoco \
  --output-root output/shadow-phases/live-g0025-phase2

# 不重新執行 controller，重播已保存的 Phase 1（包含 lift）。
env -u PYTHONPATH -u PYTHONHOME .conda/bin/python script/run_grasp_phases.py replay \
  --run-root output/shadow-phases/pilot-final \
  --phase phase1 --sample-id g0025

# 同一筆 Phase 2 replay；也可選 g0000 看無效初態、g0003 看無接觸案例。
env -u PYTHONPATH -u PYTHONHOME .conda/bin/python script/run_grasp_phases.py replay \
  --run-root output/shadow-phases/pilot-final \
  --phase phase2 --sample-id g0025

# 全部 50 筆再次比較，讀取同一份 prepared inputs。
env -u PYTHONPATH -u PYTHONHOME .conda/bin/python script/run_grasp_phases.py run \
  --suite output/shadow-phases/pilot-suite/suite_manifest.json \
  --phase both --workers 1 --seed 12 \
  --output-root output/shadow-phases/new-comparison
```

所有 run output root 都必須不存在，避免把舊輸出當成本次結果。`--phase phase1` 或 `phase2` 都可獨立執行；單 phase 的 comparison 將另一個 phase 標成 missing，不假造配對結果。含 degraded／execution error／preparation error 的已完成批次退出碼為 1；無效初始化或一般抓取失敗本身不代表執行錯誤。

Live viewer 限單一 sample、單一 phase；結束會保留末態。Native viewer 關窗結束；`--viewer mjviser` 使用既有 browser backend，末態以 Ctrl-C 結束。Replay 可設 `--speed 0.5` 或 `--no-hold`。

新模型預測的準備方式：

```bash
env -u PYTHONPATH -u PYTHONHOME .conda/bin/python script/run_grasp_phases.py prepare \
  --input-root /absolute/path/to/predictions/tests/step_050000 \
  --metadata /home/hao-fang/hugs-dataset/BimanBODex_shadow_100k_v1/shadow/right_full/metadata.json \
  --data-root /home/hao-fang/hugs-dataset \
  --ik-device cpu --output-root /absolute/path/to/new-suite
```

可加 `--selection-manifest selected.json`，內容為 `{"samples": ["relative/path/one.npy", "relative/path/two.npy"]}`，順序就是固定 sample index。沒有清單時依全部 `.npy` 的排序固定選擇。非 right_full、格式錯誤或缺少來源檔案都保留為 preparation error，不默默丟棄。重新推理仍使用 HUGS-DexLearn 的 GPU `.venv`，不與 Ada 的 CPU IK 環境混用。

## 程式與輸出

- [CLI](../script/run_grasp_phases.py)：`prepare`、`run`、`replay`。
- [實驗管理](../src/ada_grasp_ctrl/experiments/phase_comparison.py)：設定、準備、配對、統計與每筆診斷。
- [Shadow adapter](../src/ada_grasp_ctrl/experiments/hugs_shadow_input.py)：明確的 frame／joint-order 契約。
- [錄製與 replay](../src/ada_grasp_ctrl/experiments/episode_recording.py)：不推進物理的被動記錄與狀態還原。
- [suite manifest](../output/shadow-phases/pilot-suite/suite_manifest.json)：50 筆來源、scene、prepared SHA、準備狀態。
- [最終 comparison](../output/shadow-phases/pilot-final/comparison.json)、[CSV](../output/shadow-phases/pilot-final/comparison.csv)。
- [Phase 1 詳細結果](../output/shadow-phases/pilot-final/phase1/phase_results.json)、[Phase 2 詳細結果](../output/shadow-phases/pilot-final/phase2/phase_results.json)。
- [FK／重跑驗證與逐物件結果](../output/shadow-phases/pilot-final/validation.json)。

各 phase 都保留原生 `log/control_eval/run_report.json`、`log/control_stat/run_report.json` 與 resolved run manifest。統計明確讀取同次 eval report。詳細結果包括接觸步數／力、solver decisions、初始／prelift／final object Z、控制步數、模擬時間、wall time、base／手指 tracking 和 palm tracking。

Replay 使用 `.replay.npz` 加 `.scene.xml`，避免被原生 `.npy` control 掃描誤認。Sidecar 保存實際 qpos、qvel、act、ctrl、時間、reference、command、segment 和 joint names；取樣是 init、每個 0.1 秒 action 之後及 terminal，**不是每個 0.004 秒 physics tick**。Invalid episode 只有 init／terminal，沒有不存在的 lift。XML 參考本機 mesh 路徑，因此不是可任意搬到另一台機器的獨立資產包。

首版共用準備後的執行／報告／viewer；Shadow adapter 與 dummy-arm IK 明確分離。尚未改寫 Sharpa 舊入口，也未新增公開手型支援。

## 驗證與限制

- **130 個 unit／integration tests 全部通過**，包含原有 124 個及新增 6 個。真實 MuJoCo 測試確認 op 不跑接觸最佳化、ours 確實呼叫最佳化，recorder 開／關時舊軌跡完全相同。
- 50 組的 prepared hash、derived seed、初始 qpos／qvel／ctrl／object pose 完全一致；兩個 phase 的三階段 effective targets 都與 stored targets 完全一致。
- 150 個姿態 MuJoCo palm FK：最大位置誤差 **0.818 mm**、旋轉誤差 **0.00779 rad**；named finger joint 誤差 0。
- 最終 NPZ sidecar 版本重跑同一批 100 episodes，與首次完整 pilot 的既有 control 欄位在 `rtol=1e-5, atol=1e-6` 內全部一致，分類一致。
- Replay 測試重新載入場景、還原末態並驗證 viewer attach／close；live/replay 的 GUI 互動尚未由人眼驗收。
- `compileall`、全範圍 `ruff check`、本次修改檔案的 format check、`git diff --check` 通過。
- 全專案 `ruff format --check` 仍指出 8 個本次未修改的既有 Sharpa／viewer／IK 測試檔格式差異，沒有擴大修改範圍。
- Maintained quick gate 在 Shadow 歷史分類比較停止，仍是既有的 68 success／4 failure／21 invalid／7 degraded，與 golden 的 3 筆差異尚未解決。**不能宣稱 release gate 通過。** 修改前後同環境 100 筆 trajectory 嚴格比較全部通過，見[比較報告](../output/shadow-phases/regression-gate/same-environment-comparison.json)；沒有接受新 golden，gate 停止後未跑 Allegro／LEAP。

這輪完成的是可追溯的 paired pilot 與測試環境，不代表已改善模型抓取品質。建議先從 `g0000` 的初始碰撞、`g0003` 的接近／接觸差異，以及 `g0025` 的成功案例並排檢查。
