# ada-grasp-ctrl：演算法、訊號 I/O 與上游專案配置

調查日期：2026-10-14。本文的「另外兩個 project」指 README 所列的 **BODex** 與 **DexLearn**。本 repo 的接口結論依據 README、設定、程式與隨附範例資料；DexLearn 預訓練模型資訊另依其官方 README 與 Hugging Face 檔案頁核對。

## 先看全貌

```text
BODex（產生抓取資料） ─┐
                     ├─ 原始 .npy → format → 共同 grasp .npy → dummy_arm_qpos
DexLearn（產生抓取資料）┘                                    ↓
                                              dummy-arm grasp .npy
                                                       ↓
                                     control_eval（MuJoCo + 控制器）
                                                       ↓
                                      control .npy + run_report.json
                                                       ↓
                                         control_stat → 統計 YAML
```

這個 repo 維護的是 `format → dummy_arm_qpos → control_eval → control_stat` 四階段；BODex 與 DexLearn 是上游資料來源，並非執行控制器所需的 Python import。支援的手是 Shadow、Allegro、LEAP Tac3D；控制方法是 `ours`、`op`、`bs1`、`bs2`、`bs3`。入口是 `python src/main.py`，Hydra 根設定在 `src/ada_grasp_ctrl/config/base.yaml`，任務 dispatch 在 `src/ada_grasp_ctrl/tasks/__init__.py`。

**DexLearn 不一定要自己 train。** 若只是試跑這個 repo，隨附的 quick grasp 資料可直接評估；若要從新點雲生成 grasp，DexLearn 官方 README 提供下載預訓練 checkpoint 後執行 `dexlearn.sample` 的流程。只有你要建立或微調自己的生成模型時，才需要考慮訓練。官方文件：[DexLearn README](https://github.com/JYChen18/DexLearn)、[包含 `dexonomy_learning_ckpt.tar.gz` 的 Dexonomy 檔案頁](https://huggingface.co/datasets/JiayiChenPKU/Dexonomy/tree/main)。這裡的 **Dexonomy Shadow** 指「以 Dexonomy 抓取資料／任務為基礎、對應 Shadow 靈巧手的資料或模型」；`Dexonomy_GRASP_shadow.tar.gz` 是該手型的抓取資料檔，DexLearn README 的 `dexonomy_shadow_nflow_cond` 是其預訓練模型範例。它不是第四種手，也不表示該 checkpoint 可直接用於 Allegro 或 LEAP Tac3D；其他手型或自訂資料是否有相容 checkpoint，應逐一核對模型檔和 DexLearn 設定。

## 最小化跑通這個 repo 的完整範例

**不需要安裝 BODex、DexLearn，不需要下載 Dexonomy checkpoint。** `shadow full` 使用 repo 自帶的一筆 raw Learning grasp、scene 與物件資料，依序跑完 `format → dummy_arm_qpos → control_eval → control_stat`。README 用 Conda 建立維護中的環境；程式與範例腳本本身沒有強制檢查 Conda。使用 venv、Mamba 等方式也可以，但必須自行備齊等效的 Python 3.10、Pinocchio、PyTorch/CUDA 與其餘依賴。以下是在 Linux x86-64、NVIDIA GPU 可供 CUDA 12.1 使用時，README 所支援的安裝路線：

```bash
cd /mnt/c/Users/Howard/ada-grasp-ctrl
git submodule sync --recursive
git submodule update --init --recursive
conda env create -f environment.yml
conda activate ada-grasp-ctrl
python src/main.py --help
python -c 'import torch; assert torch.cuda.is_available(), "CUDA not available"'
bash script/run_example.sh shadow full
```

腳本會建立不重用的 `output/examples/shadow/<run_id>/`，最後印出實際輸出路徑與一筆抓取的狀態摘要。重點檢查該目錄的 `formatted/`、`dummy_arm/`、`control/`，以及 `log/{format,dummy_arm_qpos,control_eval,control_stat}/run_report.json`。若只想先確認控制與統計兩階段，可用 `bash script/run_example.sh shadow quick`；它使用預先算好的 100 筆 dummy-arm grasp，無須 IK／GPU，但仍須裝好專案環境與 submodule。

目前這台 WSL 的檢查結果（2026-10-14）：內建範例資料存在；三個 submodule 已按 repo 的 commit 初始化；Linux Miniforge 安裝於 `/home/howard/miniforge3`，`ada-grasp-ctrl` Conda 環境已由 `environment.yml` 建立，內含 Python 3.10.20、PyTorch 2.2.2、MuJoCo 3.6.0 與 Pinocchio 3.0.0。新 shell 可用 `source /home/howard/miniforge3/etc/profile.d/conda.sh`，再執行 `conda activate ada-grasp-ctrl`。`python src/main.py --help` 已通過。在一般 WSL 環境中，`/dev/dxg` 存在，`nvidia-smi` 顯示 NVIDIA GeForce RTX 3050 Laptop GPU，Conda 環境的 `torch.cuda.is_available()` 為 `True`。先前的 GPU 不可用結果來自程式代理的隔離環境，它未開放 GPU 裝置；並非 WSL 的 GPU 驅動故障。第一次 `shadow full` 的 IK 因隨機初始值未收斂而失敗；改用 dummy arm 對應目標姿態的關節值作 IK 初始值後，重跑四階段成功，輸出在 `output/examples/shadow/20261014T181320Z-6078/`，統計為成功 1 筆、失敗 0 筆。若 full 範例回傳 1，應檢查本次 report 是否有 `solver_degraded`／`execution_error`；回傳 2 表示設定、資產或環境前置檢查失敗。

## 主要演算法：`ours`

這是**模擬中的閉迴路接觸控制**。每個控制步從 MuJoCo 取目前關節、物體與接觸狀態，與預先插值的抓取路徑比較，求下一個 actuator 位置目標；目前 repo 沒有 ROS topic、socket 或真實機器手臂的即時訊號接口。

1. **建立模擬場景。** `control_eval` 讀取 dummy-arm grasp 紀錄，載入物件碰撞網格、質量資訊與手的 MJCF。每個物體偏移量 `0` 產生一個 episode，非零偏移量產生八個平面方向。初始穿透或接觸力超過設定門檻時，標成 `invalid_initialization`。程式位置：`tasks/control_eval_func/base.py`、`utils/hand_util.py`。
2. **建立目標路徑。** 從目前姿態插值到 grasp，再插值到 squeeze；預設控制頻率為 10 Hz，兩段各 2 秒。共用步進與記錄邏輯在 `tasks/control_eval_func/episode_runner.py`。
3. **估計可否進入穩定抓取。** 依每個接觸點的位置及座標框架組成 grasp matrix `G`，求受摩擦錐與總法向力限制的接觸力，評估合成 wrench 的平衡程度。小於 `balance_thres` 時進入 Stage 2；走完路徑仍未達標也有切換機制。實作在 `utils/grasp_controller.py::check_wrench_balance` 與 `tasks/control_eval_func/optimized_runner.py`。
4. **Stage 1：接近與建立接觸。** 最佳化 actuator 增量 `Δq_a` 與下一步接觸力 `f`。目標包含手基座／手指跟隨路徑、動作平滑及切向接觸移動懲罰；預設以完整三維等式約束接觸力變化與預測模型一致。`ours` 在此階段允許 dummy arm 移動。
5. **Stage 2：協調施力。** 目標總法向力逐步增加至手型設定值；最佳化偏重減少物體淨 wrench、引導未接觸手指、平滑動作及限制關節負載。預設只把接觸力模型的**法向**分量當硬等式，切向誤差改成成本；同時保留摩擦錐、力界限、關節與速度限制。詳見 `docs/practical-modifications-in-implementation.md` 與 `utils/grasp_controller.py::_build_control_problem`。
6. **求解與施加命令。** 使用 SciPy SLSQP；決策變數大致為 `[Δq_a, 每個接觸的三維力]`。求解後檢查成功旗標、有限值、維度、界限與約束。預設 `solver_failure_policy=apply_candidate`：被拒絕但形狀正確且有限值的候選命令可以施加，但 episode 仍標 `solver_degraded`；另有 `hold_current`、`fail_episode`。施加的是 `q_a,next = q_a,current + Δq_a` 的**actuator 位置目標**，經插值送入 MuJoCo，並非直接輸出關節扭矩。程式位置：`optimization.py`、`utils/grasp_controller.py::_solve_control_problem`、`utils/hand_util.py::ctrl_qpos_a_with_interp`。
7. **抬升測試。** 控制階段後，dummy arm 嘗試將物體抬高 0.2 m；`control_stat` 用最後物體高度判斷成功，規則是與目標高度之差小於 `lift_height / 2`。統計的 `task.lift_height` 預設同為 0.2 m；評估端的抬升高度目前寫在程式中，若單改統計設定會造成兩端不一致。

其他方法共用 episode runner，但策略不同：`op` 只跟隨位置路徑；`bs1` 對手指獨立做位置／接觸力回授；`bs2` 與 `ours` 共用最佳化路徑，差別是 Stage 1 不移動 dummy arm；`bs3` 使用每接觸點相同目標力的最佳化。對應程式在 `tasks/control_eval_func/tabletop_dummy_arm_*.py`。

### 控制迴圈的訊號接口

| 方向 | 訊號 | 表示方式／來源 |
| --- | --- | --- |
| 離線輸入 | `pregrasp_qpos`、`grasp_qpos`、`squeeze_qpos`，物體姿態、尺度與資產路徑 | dummy-arm grasp `.npy`；`schema.py::validate_grasp_record` 驗證 |
| 每步量測 | `current_qpos_f`（全部關節）、`current_qpos_a`（actuator 目標）、`obj_pose`、hand-object contacts | MuJoCo `MjHO`；`episode_runner.py::sample_step` |
| 每步接觸 | 位置 `contact_pos (3,)`、局部 wrench `contact_force (6,)`、世界座標框架 `contact_frame (3,3)`、接觸距離等 | `utils/hand_util.py::get_curr_contact_info`；法向是框架第 0 軸，wrench 方向為手作用於物體 |
| 每步參考 | `target_qpos_f`、上一個 `Δq_a`、目標總法向力、控制 stage | 插值路徑與控制策略 |
| 最佳化輸出 | `q_a`、`dq_a`、預測接觸力及 solver 診斷 | `utils/grasp_controller.py::ctrl_opt`；`optimized_runner.py` |
| 模擬命令 | `target_qpos_a` | 透過 `MjHO.ctrl_qpos_a_with_interp` 寫到 MuJoCo actuator position control |

`qpos_f` 是模型關節自由度（DOF）向量；`qpos_a` 是 actuator（DOA）向量。兩者的次序和映射由 `utils/robots/` 與 `utils/robot_adaptor.py` 決定，不能把 `.npy` 中的欄位依字典順序直接當 actuator 命令。

## 四階段的檔案 I/O 契約

所有 `.npy` 紀錄是 NumPy 儲存的 Python mapping，讀取時使用 `allow_pickle=True`。因此只應載入可信任的資料檔。新紀錄帶 `schema_version: 1`；缺少版本欄位的舊紀錄仍可讀，但依然會被驗證。姿態統一使用 `[x, y, z, qw, qx, qy, qz]`，四元數是 **WXYZ**。

| 階段 | 輸入 | 主要輸出 |
| --- | --- | --- |
| `format` | `task.data_path` 下的原始 `.npy`，以及 `scene_path` 指向的 scene `.npy` 和物件 mesh | `output_root/<exp_name>_<hand>/graspdata/**/*.npy`；共同 grasp record |
| `dummy_arm_qpos` | 上一步 grasp record，`7 + 手部 DOF` 的三個 qpos | `output_root/<exp_name>_dummy_arm_<hand>/graspdata/**/*.npy`；將 7 值自由基座姿態改成 6 個 dummy-arm 關節，成為 `6 + 手部 DOF` |
| `control_eval` | dummy-arm grasp record、hand MJCF、物件網格／metadata、`task.method`、`task.offsets` | `control/**/*.npy` 軌跡，以及 `run_manifest.yaml`、`run_report.json`、`failures.jsonl` |
| `control_stat` | 指向本次 `control_eval/run_report.json` 的 `task.input_report`，或舊式目錄掃描 | 成功率及誤差等統計 YAML，另有本階段的 report |

### BODex 和 DexLearn 的原始輸入

`format` 的 converter 定義在 `tasks/convert_format.py`，形狀驗證在 `schema.py::validate_raw_record`：

| `task.data_name` | 原始 `.npy` 必要欄位 | 形狀與轉換 |
| --- | --- | --- |
| `BODex` | `scene_path`、`robot_pose`、`joint_names` | `robot_pose`: `(1, N, T, 7+H)`，`H` 為手部 DOF；每個 grasp 拆成獨立紀錄。所有目前支援手的 `mocap=true` 設定預期至少 3 個時間點，依序取 pregrasp、grasp、squeeze。檔案搜尋樣式是 `*_grasp.npy`；非 mocap 相容路徑搜尋 `*_mogen.npy`。Shadow 與 Allegro 另有明確的關節／座標修正。 |
| `Learning` | `scene_path`、`pregrasp_qpos`、`grasp_qpos`、`squeeze_qpos` | 每個 qpos 是 `(7+H,)`，一個輸入對應一個輸出；這是 README 所示的單抓取 DexLearn 接口。 |
| `Batched` | 上述三個 qpos、`scene_path`、`scene_scale` | qpos 各為 `(N,7+H)`，`scene_scale` 為 `(N,)`；拆成 N 個 grasp 輸出。 |

`scene_path` 可以是絕對路徑，或相對於原始 `.npy` 所在目錄／`data_root` 的路徑。scene `.npy` 需要 `task.obj_name`，以及 `scene[物件名].file_path`、`pose`、`scale`；`file_path` 必須指到存在的 mesh。轉換器從 scene 取得物件位置、尺度與物件目錄。**`task.data_path` 應指向只放原始抓取 `.npy` 的目錄**：`Learning`／`Batched` 會遞迴讀取其中所有 `.npy`，若把 scene `.npy` 一起放在該目錄，scene 也會被誤當抓取輸入。

共同 grasp record 至少包含 `obj_path`、`obj_pose (7,)`、正值 `obj_scale`、三個等長的一維 qpos；BODex 紀錄另保留 `joint_names`、`scene_path`，Learning 的 `joint_names` 可缺省。`dummy_arm_qpos` 會補上完整且有序的 `joint_names`，供 `control_eval` 嚴格檢查。物件目錄還須有 `info/simplified.json` 與 `urdf/meshes/convex_piece_*.obj`。

隨附範例驗證了具體長度：Shadow 原始／formatted qpos 為 29（`7+22`），dummy-arm qpos 為 28（`6+22`）；Allegro 和 LEAP Tac3D 分別為 23（`7+16`）與 22（`6+16`）。這些是 `examples/data/<hand>/` 中的實際檔案形狀，不代表可以忽略 joint-name 次序。

`control_eval` 對每個 offset 生成獨立 `.npy`；`0.0` 對應一個 `_dist_0_pos_0.npy`，非零距離對應八個方向。`ours` 的方法目錄名稱是 `ours_<ablation_name>`，預設為 `ours_default`。control record 包含 `obj_pose` 軌跡、`dof`、`doa`、`contacts`、`planned_dof`、`stage`、`balance_metric`、`desired_sum_force`、`solver_diagnostics`、`episode_status` 等。`control_stat` 只把完成且有效的 `success + failure` 算進成功率分母；`invalid_initialization`、`solver_degraded`、`execution_error` 各自計數。CLI exit code：0 正常、1 有執行錯誤／solver degradation、2 前置檢查失敗。

## BODex、DexLearn 應放在哪裡？

**建議作為平行的獨立 repo，不要複製進 `ada-grasp-ctrl/src` 或 `third_party`。** 例如在 Windows：

```text
C:\Users\Howard\
├── ada-grasp-ctrl\       # 這個評估 repo
├── BODex\                # 上游：合成抓取資料
├── DexLearn\             # 上游：訓練／採樣抓取姿態
├── grasp-data\           # 大型資料與兩個上游的匯出結果
│   ├── raw\bodex\shadow\
│   ├── raw\dexlearn\shadow\
│   └── assets\object\DGN_2k\   # 對應紀錄中的 assets/object/DGN_2k/... 相對路徑
│       ├── processed_data\
│       ├── scene_cfg\
│       └── valid_split\
└── ada-grasp-runs\       # 本 repo 的每次輸出
```

在 WSL 中，這些位置分別是 `/mnt/c/Users/Howard/ada-grasp-ctrl`、`/mnt/c/Users/Howard/BODex`、`/mnt/c/Users/Howard/DexLearn` 等。這只是建議的放置方式：本程式靠**檔案路徑**連接兩個上游，不依賴它們與本 repo 的相對位置，也沒有會自動尋找 BODex／DexLearn repo 的設定。`.gitmodules` 的三個套件是 `mujoco_menagerie`、`pytorch_kinematics`、`utils_python`，與這兩個上游 project 不同；現有 checkout 的三個 submodule 已初始化。

對外部資料，三種 root 分工如下：

| 設定 | 應指向 | 何時設定 |
| --- | --- | --- |
| `asset_root` | 手的 MJCF 與 meshes，預設為本 repo 的 `assets/` | 手資產搬離 repo 時 |
| `data_root` | 原始／scene 紀錄內相對路徑的共同基準；預設為本 repo 根目錄 | 上游 scene 或物件資料搬到 `grasp-data/` 時 |
| `output_root` | 本次轉換、模擬、報告輸出的根目錄；預設為本 repo 的 `output/` | 建議每次實驗使用新的輸出目錄 |
| `task.data_path` | **該次 format 要讀的原始抓取檔資料夾** | 每次執行 `format` 都要指定 |

`DGN_2k_processed.zip` 是 README 指向的**物件資料**，不是 BODex 或 DexLearn 程式碼；若使用外部 BODex 資料，需有與 scene 紀錄相符的 `processed_data`／`scene_cfg` 佈局。上圖的 `assets/object/` 前綴適用於紀錄以 `assets/object/DGN_2k/...` 為相對路徑的情況；若上游紀錄使用其他路徑，必須讓實際資料樹與紀錄一致，或提供可解析的絕對路徑。repo 自帶 quick 範例所需的小型物件集，`bash script/run_example.sh shadow quick` 不需要另下載整包 DGN 2k。

### 串接範例（在 WSL、已啟用專案 Python 3.10 環境）

以下路徑是**配置範例**，請換成實際存在的上游輸出；`demo-001` 應是新的輸出目錄。先以 DexLearn 單抓取輸出為例：

```bash
cd /mnt/c/Users/Howard/ada-grasp-ctrl

python src/main.py setting=tabletop hand=shadow task=format exp_name=demo \
  data_root=/mnt/c/Users/Howard/grasp-data \
  output_root=/mnt/c/Users/Howard/ada-grasp-runs/demo-001 \
  task.data_name=Learning \
  task.data_path=/mnt/c/Users/Howard/grasp-data/raw/dexlearn/shadow \
  task.max_num=-1
```

若讀 BODex 輸出，改用 `task.data_name=BODex`，並將 `task.data_path` 指到包含 `*_grasp.npy` 的 BODex raw 目錄；若 DexLearn 匯出一檔多筆，則用 `task.data_name=Batched`。**converter 名稱由資料紀錄形狀決定**，不是由上游資料夾名稱決定。

接著以**同一組** `exp_name`、`data_root`、`output_root` 執行：

```bash
python src/main.py setting=tabletop hand=shadow task=dummy_arm_qpos exp_name=demo \
  data_root=/mnt/c/Users/Howard/grasp-data \
  output_root=/mnt/c/Users/Howard/ada-grasp-runs/demo-001 \
  task.device=cuda:0 task.max_num=-1

python src/main.py setting=tabletop hand=dummy_arm_shadow task=control_eval exp_name=demo \
  data_root=/mnt/c/Users/Howard/grasp-data \
  output_root=/mnt/c/Users/Howard/ada-grasp-runs/demo-001 \
  log_dir=demo_dummy_arm_shadow/log/control_eval \
  task.method=ours task.offsets='[0.0]' task.max_num=-1

python src/main.py setting=tabletop hand=dummy_arm_shadow task=control_stat exp_name=demo \
  data_root=/mnt/c/Users/Howard/grasp-data \
  output_root=/mnt/c/Users/Howard/ada-grasp-runs/demo-001 \
  task.method=ours task.setting_name=dist_0 \
  task.input_report=demo_dummy_arm_shadow/log/control_eval/run_report.json
```

`dummy_arm_qpos` 的預設 `cuda:0` 需要相容 GPU／CUDA 環境；已備好的 dummy-arm quick 範例可直接從 `control_eval` 開始，CPU 即可。正式結果請檢查**本次** `run_manifest.yaml`、`run_report.json`、`failures.jsonl` 與統計 YAML，勿從舊的 `output/` 目錄推斷本次執行成功。

## 調查範圍與索引

- 公開流程、安裝與資料配置：`README.md`、`environment.yml`、`src/ada_grasp_ctrl/config/`。
- Raw／grasp／control 欄位與形狀：`src/ada_grasp_ctrl/schema.py`、`src/ada_grasp_ctrl/tasks/convert_format.py`。
- IK 與模型關節轉換：`src/ada_grasp_ctrl/tasks/dummy_arm_qpos.py`、`src/ada_grasp_ctrl/utils/robots/`。
- 接觸控制：`src/ada_grasp_ctrl/tasks/control_eval_func/`、`src/ada_grasp_ctrl/utils/grasp_controller.py`、`src/ada_grasp_ctrl/optimization.py`、`src/ada_grasp_ctrl/utils/hand_util.py`。
- 統計與報告：`src/ada_grasp_ctrl/tasks/control_stat.py`、`src/ada_grasp_ctrl/batch.py`。

本次讀取了隨附的 raw、formatted、dummy-arm `.npy` 範例以核對欄位和形狀，並建立了符合 `environment.yml` 的 Conda 環境。`shadow full` 已在一般 WSL 環境中完整跑通；程式代理的隔離環境仍無法存取 GPU。
