# Sharpa right 100K：實驗用 controller 一鍵測試

2026-11-07 首版；2026-11-09 更新。已新增 `script/test_sharpa_controller.sh`，重用既有 Sharpa 100K learned predictions，在 dummy arm＋右手上配對執行原生 `op`／`ours`。**完整 2,000 次評估已完成**；原始分類與使用者指定的重算表見[完整評估紀錄](sharpa-shadow-full-evaluation-2026-11-09.md)。

## 小規模執行

在 repo 根目錄執行：

```bash
# 四尺度各一筆、兩種控制器，共八次執行
bash script/test_sharpa_controller.sh --smoke

# 僅準備／驗證同一組四筆輸入，不啟動模擬
bash script/test_sharpa_controller.sh --smoke --prepare-only
```

也可以直接使用 shell 的絕對路徑，與呼叫者工作目錄無關。預設用本 repo 的 `.conda/bin/python`，可用 `PYTHON_BIN` 覆寫。此入口是 CPU preparation／simulation，會清除 `PYTHONHOME`、設置專案 `PYTHONPATH` 並隱藏 CUDA 裝置；不重新訓練、不執行 GPU 推論、不開啟 viewer。

預設 **1 worker**。`--workers 2` 已用相同 smoke 輸入比對；其餘 worker 數可指定，但尚未驗證 Sharpa 的大批次吞吐量。

每次自動建立 `output/regression/sharpa-runs/<UTC timestamp>-<pid>/`，也可指定 `--output-root <尚不存在的目錄>`。已存在的輸出會被拒絕，沒有覆蓋或接續舊結果。

## 固定輸入與完整模式

預設來源為：

```text
~/hugs-dataset/BimanBODex_sharpa_1264k_v1/
  dexlearn_training/sharpaMulti_robotMultiHierar_sharpa_right_full_1m_100k_v1/
    ckpts/step_100000.pth
    tests/step_100000/sharpaMulti_scale_matched/3_right_full/
```

從專用四尺度 manifest 的 `tabletop_ur10e` 場景中，以 seed 42 選 25 個有四尺度的物體，每物體／尺度選一個 scene，再從既有 20 候選抽 10 筆。總共 100 scenes、1,000 grasps；候選選擇不讀取 Bench 成敗，也不依本次執行結果補換樣本。各尺度的 pose 是各自抽選，不保證相同 pose suffix；不能僅憑尺度分組解釋因果。

Smoke 是這份選樣的每尺度第一筆，保留完整 sample index 與 seed。本次四筆恰好來自同一個物體，只用來驗證四尺度流程，不能推廣為多物體成功率。

| 模式 | Suite 目錄 | 評估次數 |
| --- | --- | ---: |
| `--smoke` | `output/regression/sharpa-right-100k-tabletop-smoke-v1` | 4 × 2 = 8 |
| 不加 `--smoke` | `output/regression/sharpa-right-100k-tabletop-100x10-v1` | 1,000 × 2 = 2,000 |

首版僅執行 smoke；後續已由使用者完成完整 suite 與 2,000 次評估。需要再次完整評估時執行：

```bash
bash script/test_sharpa_controller.sh
```

首次只轉換既有候選並建立 suite；後續驗證並重用所有 source／prepared／scene／模型／mesh／checkpoint hashes，不重新推論或 IK。既有 suite 部分完成、內容變動或資產遺失時，會失敗並要求使用新目錄，不能自動修補來掩蓋來源漂移。

來源覆寫變數只在建立新 suite 時使用：

| 變數 | 用途 |
| --- | --- |
| `HUGS_DATASET_ROOT` | 預設 `~/hugs-dataset` |
| `SHARPA_CHECKPOINT` | 明確的 100K checkpoint，保存 hash，不載入權重做推論 |
| `SHARPA_METADATA` | 右手 metadata JSON |
| `SHARPA_PREDICTIONS_ROOT` | 對應 checkpoint 的 `3_right_full` 原始候選目錄 |
| `SHARPA_SCENE_MANIFEST` | 四尺度 scene 清單 |
| `SHARPA_SOURCE_XML` | Bench 右手控制模型 |
| `SHARPA_SOURCE_URDF` | HUGS-Sharpa-Assets 右手運動學模型 |
| `SHARPA_SUITE_ROOT` | 固定 suite 目錄，也可用 `--suite-root` |

替換 checkpoint 必須同時提供與它相符的候選並建立新 suite；本工具不會從 checkpoint 自動產生預測，也不能僅靠 hash 證明任意外部候選確實出自該 checkpoint。

## 已實作的手型與 protocol

- 使用右手 `right_hand_flange` world pose，解析 58D 中的右手 29D，再轉成 6 dummy joints＋22 finger joints。依 metadata 關節名稱排列，採 WXYZ；不套用 Shadow 34 mm offset。
- Dummy arm 使用固定 X／Y／Z slides 與 intrinsic XYZ hinges；pregrasp 平移和手指 ×2，旋轉保持原值，grasp／squeeze 不額外外推。準備與 preflight 都核對三個 targets 的 flange、所有右手 body frames 和 Jacobian。
- 由現有 Bench XML 建立 suite-local MJCF，使用既有右手碰撞幾何與 22 個 position actuators；另外產生保留 frame／joint axes 的運動學 URDF。原始資產不改動，衍生模型、來源與 mesh 均保存 hashes。
- 手指 kp=7.5、kv=0、被動 damping 沿用來源 XML，速度限值 2 rad/s；dummy arm 增益／速度沿用既有 metadata。`final_sum_force=15 N` 是未校正的實驗起點。
- 摩擦 `[0.8,0.02]`、固定物體質量 0.1 kg、offset 0、初始化穿透門檻 10 mm、關閉初始力拒絕、seed 12。控制器摩擦錐參數與原 objectives 保留。
- 原生 `op`／`ours` 共用初始化、接觸、控制與 0.2 m lift；使用原生 `control_stat` 成功判定與 solver failure policy。手與桌面的接觸排除仍沿用 Ada 原生場景設定，並非另外驗證了 collision-free approach。
- 右手 body prefix 用 `right_`，包含手掌及手指；機器人不接 CRX，不輸出實機命令。

公開 `control_eval` 支援清單沒有擴大；這是獨立實驗入口，有自己的 preflight。唯一共用程式修改是 `control_stat` 尊重 hand config 的 `robot_prefix` 與 `kinematics_file_type`，與 BaseEval 已有行為一致；原手型預設仍是原 prefix／MJCF。

## 報告與重播

每個 run 保留原生 eval/stat reports、resolved manifests、分類與 solver 診斷、控制 NPY、scene XML，以及含 lift／terminal state 的 replay。頂層提供 `summary.md`、`comparison.json`、`scale_results.json`、`object_results.json`、`validation.json`、`regression.json`、`code_state.json`、`exit_codes.json`。

摘要同時保留 success/selected 與 success/(success+failure)，獨立列出 invalid initialization、solver degradation 和 execution error。Smoke 的摘要明確標記為可行性測試，不作效能估計。

```bash
# 用同一 frozen suite 的先前 run 作分類基準
bash script/test_sharpa_controller.sh --smoke \
  --baseline output/regression/sharpa-feasibility-v1

# 只開啟已保存的回放，不重新模擬
bash script/preview_sharpa_regression.sh \
  --run-root output/regression/sharpa-feasibility-v2 --port 8083
```

Baseline 必須有相同 suite digest、protocol 與 sample IDs；smoke 與完整模式不互相比較。退出碼 0 表示正常完成且無已檢出的退化／錯誤／分類退步；1 表示退化、執行／統計錯誤或基準退步；2 表示前置條件失敗。物理 grasp failure 本身不必然使退出碼非零。

## 首版小規模驗證與限制（2026-11-07）

- 兩輪四筆×兩階段 smoke，合計 **16 episodes**；第一輪 1 worker，第二輪從 `/tmp` 用絕對 shell 路徑、2 workers 和第一輪 baseline 執行。兩輪皆正常完成，未觀察到 solver degradation 或 execution error。
- 四對輸入的初始 qpos／qvel／ctrl／物體 pose、seed 與三個 targets 一致。
- 8 組跨輪 episode 的完整 control record、contacts、solver diagnostics、replay frames／targets／metadata 比對通過 `rtol=1e-5, atol=1e-6`；只排除 `t_*` 效能計時欄位。這是本小樣本的 serial／parallel 一致性，並非所有 solver 失敗案例的保證。
- 四個 `ours` episodes 都有接觸，接觸時的最佳化命令有偏離固定參考軌跡，驗證確實走到 adaptive controller。
- 149 個 unit tests 全數通過，其中 12 個新增契約測試涵蓋 frame、左右手、關節排列、錯誤輸入、來源漂移、baseline、公開支援邊界及統計相容性。
- 全專案 Ruff check、compileall 與新增 shell 語法通過。全專案 format check 仍有 8 個既有無關檔案未符合格式，未改動它們；新增／修改檔案格式檢查通過。
- 依本次「只做小規模可行性測試」範圍，未執行完整 2,000 次、未跑 3×100 release gate、未產生或更新 golden。完整回歸、更多失敗案例與 controller 性能尚待後續驗證。
- 訓練 split 來源關係尚未完整還原，不能把本實驗直接宣稱為已核驗的 unseen-object 泛化結果；Bench 的 68.73% 也不是本 runner 的對照值。

小測證據：`output/regression/sharpa-feasibility-v1/`、`sharpa-feasibility-v2/`。第二輪的 `feasibility_checks.json` 保存跨輪控制／replay 核對摘要。

設計背景：[Sharpa 100K 評估與實作建議](../../agent_tasks/Sharpa_100K_Adaptive_Controller_評估與實作建議.md)。
