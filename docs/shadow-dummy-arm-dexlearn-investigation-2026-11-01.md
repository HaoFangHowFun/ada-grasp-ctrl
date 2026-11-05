# Shadow＋dummy arm 與 DexLearn → Ada 接入調查

調查日期：2026-11-01。範圍是本機 `ada-grasp-ctrl`、`HUGS-DexLearn`、`HUGS-DexGraspBench` 與已有 Shadow checkpoint；沒有重新訓練模型，也沒有修改這三個 repo 的控制器、訓練程式或套件安裝。

> **後續更新（同日）：** 依使用者要求，已成功執行 `git submodule update --init --recursive`，三個 submodules 都已 checkout 到 repo 指定 commit。Shadow 與 dummy-arm Shadow 的原始 XML 已直接載入成功；使用 `PYTHONPATH=src:third_party/pytorch_kinematics/src` 和既有 Bench Python 環境，原 CPU IK 測試通過，不再需要隔離資產目錄。下文「submodules 未初始化／缺少 meshes」是初次調查時的歷史狀態，現在已解決。Pinocchio 環境問題尚未處理，尚未重驗完整 `ours` 控制。

> **最新環境更新（同日）：** 已建立 repo 內獨立 `.conda` 環境，補齊 Pinocchio 3.0.0 與 `environment.yml` 指定依賴。124 個測試通過，bundled Shadow 四階段使用 CPU IK 實跑成功並抬起物體。100 筆 quick 完成且無 execution error，但有 3 筆分類與 expected inventory 不一致，嚴格 gate 尚未通過；沒有變更基準。使用命令、GPU 限制和驗證結果見 [本機相容環境文件](local-environment-2026-11-01.md)。以下保留初次調查過程，不代表這些環境阻礙仍然存在。

## 1. 結論

**Shadow＋dummy arm 仍是 Ada 正式維護的配置，可以繼續作為 baseline；但目前這份 checkout／Python 環境不能直接跑完 `ours`。** 問題目前定位在未初始化的 submodules 和缺少 Python 3.10 可用的 Pinocchio，不能據此認定 Shadow 控制方法已失效。

**不需要先重新訓練 DexLearn。** 本機有兩組 Shadow `RobotHierarchicalModel` 的 50,000-step checkpoint。本次已用 100k 版本完成新的 GPU 推理，取得兩筆包含 `pregrasp_qpos / grasp_qpos / squeeze_qpos` 的樣本，並完成 Ada `format → dummy_arm_qpos` 與 MuJoCo FK 驗證。

**不能把目前 HUGS-DexLearn 輸出直接丟給 Ada 的 `Learning` converter。** 有三個實際差異：

1. HUGS 每階段是 58 維雙手表示；Ada Shadow 輸入是 29 維單手表示。
2. 本機訓練 metadata 的關節順序是拇指先；Ada 是食指先、拇指最後。
3. HUGS 的世界姿態對應 `rh_palm`；Ada dummy-arm IK 對應手掌下方的終端 frame，必須減去旋轉後的局部 `[0, 0, 0.034]` 公尺。

已提供這個特定契約的 [轉換工具](../../reviews/ada_shadow_dexlearn_20261101/export_shadow_to_ada.py)，並以六個真實推理 stage 驗證。這些結果證明資料接入與 IK 可用，**尚未證明 learned grasp 在 Ada 中能抓成功，也沒有新的 Ada 成功率數字**。

Ada 在此負責抓取執行與接觸控制；模型訓練和姿態生成在 DexLearn。若 CRX／Sharpa 表現不佳，應分別檢查上游姿態品質、robot/frame 適配和下游控制，不能只憑一段模擬歸因於訓練。

## 2. 本次實際驗證

| 項目 | 結果 | 限制／解讀 |
| --- | --- | --- |
| Ada CLI `--help` | 通過 | 任務與 Shadow 配置仍存在 |
| 直接執行 bundled `shadow full` | 失敗 | `format` 成功，IK 因缺少 submodule 的 Shadow meshes 失敗 |
| 借用 Bench 新版 IK 依賴與 meshes | 失敗 | PK 0.9.1 出現 CPU dtype／CUDA device 不相容；不能替代 Ada 的 pinned source |
| 抽出 Ada 精確 pinned IK source／Shadow assets 後，原 CPU IK 測試 | 通過，1 test | 原有 bundled fixture 的三階段 IK／FK 測試 |
| 相同隔離資產下再次執行 bundled `shadow full` | `format`、CUDA IK 通過；控制失敗 | `control_eval` 回報 `ModuleNotFoundError: No module named 'pinocchio'`，無有效控制軌跡 |
| 100k Shadow checkpoint 新推理 | 通過 | 1 個 test scene、4 個 candidates、保留 2 筆；只作接口 smoke |
| 新推理資料的橋接與 Ada `format` | 通過，2 筆 | 原生 `Learning` converter，非改過的 formatter |
| 新推理資料的 Ada CUDA IK | 通過，2 筆 × 3 stages | 使用 pinned PK source |
| MuJoCo FK 回算至原始 HUGS `rh_palm` | 通過 | 最大位置誤差 **0.000809396 m**；旋轉誤差 **0.00753909 rad**；命名手指關節誤差 **0 rad** |

這次沒有跑 3×100 的完整 release gate，也沒有宣稱與既有 golden trajectories 數值相同。沒有有效 `control_eval` 輸出時，後續統計檔即使被建立也不代表抓取通過。

證據位置：

- [FK 數值報告](../../reviews/ada_shadow_dexlearn_20261101/handoff_fk_validation.json)
- [版本記錄](../../reviews/ada_shadow_dexlearn_20261101/versions.json)
- [完整驗證日誌目錄](../../reviews/ada_shadow_dexlearn_20261101/logs/)
- [新資料 format report](../../reviews/ada_shadow_dexlearn_20261101/ada_pipeline/log/format/run_report.json)
- [新資料 IK report](../../reviews/ada_shadow_dexlearn_20261101/ada_pipeline/log/dummy_arm_qpos/run_report.json)
- [控制失敗的詳細紀錄](../output/examples/shadow/shadow-dexlearn-audit-20261101-pinned/log/control_eval/failures.jsonl)

## 3. 原本 Shadow 配置的狀態與環境準備

### 程式與資產契約仍在

四階段維持：

```text
raw grasp → format → dummy_arm_qpos → control_eval → control_stat
```

- `format`、IK 使用 `hand=shadow`。
- 控制與統計使用 `hand=dummy_arm_shadow`。
- 原始共同姿態：`xyz(3) + quaternion WXYZ(4) + fingers(22) = 29`。
- IK 後：`dummy arm(6) + fingers(22) = 28`。
- dummy arm 是三個平移軸＋三個旋轉軸，不是有工作空間／奇異點限制的 CRX。
- `ours` 仍沿用原本 Stage 1／Stage 2、15 N 目標總力與 0.2 m lift；本次沒有更動這些參數。
- 最近 CRX 改動加入可覆寫的 lift hook、URDF kinematics 選項和 contact prefix；Shadow 仍走既有預設路徑。

### 目前機器的實際阻礙

`git submodule status` 顯示以下三個全部未初始化：

```text
mujoco_menagerie     0601a3160d773533b7c425d5484bf29bca7399ff
pytorch_kinematics   74de5a01bff2a3222b8174e6f8eb91db8756bfb3
utils_python        2d8dc1a5abf5899069f9ec73c13de73674f4c897
```

本次借用 `HUGS-DexGraspBench/.venv/bin/python`：Python 3.10.22、PyTorch 2.8.0+cu128、MuJoCo 3.6.0、RTX 5090。這**不是** Ada `environment.yml` 的完整鎖定環境。

該環境沒有可用的 Python 3.10 Pinocchio；一般 shell 的 `PYTHONPATH` 另外帶入 ROS Jazzy 的 Python 3.12 Pinocchio 路徑，但匯入其 extension 失敗。只用 `find_spec('pinocchio')` 判斷安裝成功會誤判，必須實際 import。

為確認 IK，本次從 sibling repo 的本機 Git objects 解包 Ada 指定 commit 到 [pinned_runtime](../../reviews/ada_shadow_dexlearn_20261101/pinned_runtime/)，沒有更動 submodule 或安裝套件。未初始化 checkout 與隔離驗證結果應分開看。

### 正式恢復環境的建議

以下是後續準備步驟，**本次沒有執行安裝或 submodule 初始化**：

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
git submodule update --init --recursive
conda env create -f environment.yml
conda activate ada-grasp-ctrl

# Avoid mixing ROS Python 3.12 packages into this Python 3.10 process.
env -u PYTHONPATH python -c \
  'import mujoco, pinocchio, torch, pytorch_kinematics; print(pinocchio.__version__, torch.__version__)'
env -u PYTHONPATH python src/main.py --help
env -u PYTHONPATH bash script/run_example.sh shadow quick
```

需要先有 Conda 才能使用此安裝方式。本機 RTX 5090 與文件中的舊 PyTorch 2.2.2／CUDA 12.1 stack 不能假定 GPU 相容；Ada 控制與 quick 範例可用 CPU，IK 也能明確設 `task.device=cpu`。完整四階段腳本沒有提供額外 IK override 參數，CPU IK 請使用第 7 節的逐階段命令。

若要保留目前較新的 CUDA stack，應建立獨立的 Ada 環境，核對 Pinocchio 的 MJCF API 與 pinned PK source；不要直接在既有 DexLearn 訓練環境降版 Torch。新環境的 `ours` smoke 和 quick classifications 仍需再驗證。

## 4. 已有模型與應選的路線

### 優先使用本機 HUGS Shadow checkpoint

兩個檔案都存在，已讀取 checkpoint metadata，皆包含 `model / optimizer / iter`，`iter=50000`：

```text
HUGS-DexLearn/output/shadowMulti_robotMultiHierar_right_full_100k_v1/ckpts/step_050000.pth
HUGS-DexLearn/output/shadowMulti_robotMultiHierar_right_full_752k_v1/ckpts/step_050000.pth
```

相容設定：

- `algo=robotMultiHierar`、`data=shadowMulti`、`test_data=shadowMulti`。
- `RobotHierarchicalModel`、`WrappedMinkUNet`、`DiffusionBiRT_MLPRTJ`。
- `data.model.joint_num=44`、`traj_length=3`：這是雙手內部表示；即使只做右手，不能改成 22 再載入這份 checkpoint。
- `data.robot.metadata_group=right_full`、`test_data.metadata_group=right_full`。
- `test_data.grasp_type_lst=["3_right_full"]`，明確限制右手完整抓取。
- 這兩個實驗使用 complete point clouds；採樣先保持 `test_data.pc_source=complete`。

100k／752k 是實驗名稱，不能直接當作有效訓練成功 grasp 數。例如既有 100k audit 記錄實際收集 50,726 筆成功 grasp，train split 為 40,814 筆。本次沒有重新清點 752k 資料，也沒有比較兩模型品質。

同一個 checkpoint 會生成三階段，不需要三個模型分別生成 pregrasp、grasp、squeeze。此架構以 diffusion 建模 squeeze wrist RT，配合 MLP 預測其餘階段 RT 與三階段關節；`task/sample.py` 最後以 stage index 0／1／2 寫出三個欄位。

### 不要混用 Human prior 或舊 DexLearn checkpoint

`humanMultiHierar` 的 score／pose prior 不包含完整 Shadow 22 軸三階段關節，不能直接送 Ada。這個需求要用 **robot grasp model**。

repo 雖保留 `bodex_tabletop_shadow.yaml`、`dexonomy_shadow.yaml`、`nflow.yaml` 等舊設定，它們不是本次已驗證的 HUGS robot checkpoint 路線。舊配置的 dataset schema、model class 和訓練 frame 需搭配對應版本，不能把外部 Dexonomy nflow checkpoint 改名後載入 `RobotHierarchicalModel`。本次沒有下載或驗證外部預訓練模型。

### 如果之後要重新訓練

先準備經 Bench 評估、collect／concatenate 的成功 robot records，而非把 Ada 控制軌跡直接當 training records。現有 robot dataset 讀取：

```text
pregrasp_global_pose / grasp_global_pose / squeeze_global_pose
pregrasp_joint_pos  / grasp_joint_pos  / squeeze_joint_pos
grasp_type + scene metadata + named-joint metadata
```

模型還需要 object split、scene pose／scale 和點雲。可沿用現有資料作新的獨立實驗：

```bash
cd /home/hao-fang/hugs-workspace/HUGS-DexLearn
export HUGS_DATASET_ROOT=/home/hao-fang/hugs-dataset
export HUGS_ASSET_ROOT=/home/hao-fang/hugs-workspace/HUGS-DexLearn/assets

.venv/bin/python -m dexlearn.main task=train \
  algo=robotMultiHierar data=shadowMulti \
  exp_name=shadow_right_full_new resume=false \
  data.paths.grasp_path=/home/hao-fang/hugs-dataset/BimanBODex_shadow_100k_v1/shadow \
  data.robot.metadata_group=right_full \
  data.point_cloud.source=complete data.point_cloud.preload=false
```

這是依目前程式整理的訓練配方，本次未啟動訓練。採用新的 experiment name，並另存 resolved config／metadata；既有實驗的 `.hydra/config.yaml` 會被後續 sample／visualize 呼叫覆寫，不能當作不可變的原始訓練設定紀錄。

## 5. 可重跑的 Shadow 推理

下列命令等同本次已通過的小型採樣；`audit_root` 使用新的目錄以保留舊結果。checkpoint 使用絕對路徑，避免新 experiment 下 `ckpt=050000` 找不到舊權重。

```bash
cd /home/hao-fang/hugs-workspace/HUGS-DexLearn
export HUGS_DATASET_ROOT=/home/hao-fang/hugs-dataset
export HUGS_ASSET_ROOT=/home/hao-fang/hugs-workspace/HUGS-DexLearn/assets
audit_root="/home/hao-fang/hugs-workspace/reviews/shadow_handoff_$(date -u +%Y%m%dT%H%M%S)"

OMP_NUM_THREADS=2 .venv/bin/python -m dexlearn.main \
  task=sample algo=robotMultiHierar data=shadowMulti test_data=shadowMulti \
  exp_name=ada_shadow output_folder="$audit_root/sampling" \
  ckpt=/home/hao-fang/hugs-workspace/HUGS-DexLearn/output/shadowMulti_robotMultiHierar_right_full_100k_v1/ckpts/step_050000.pth \
  data.paths.grasp_path=/home/hao-fang/hugs-dataset/BimanBODex_shadow_100k_v1/shadow \
  data.robot.metadata_group=right_full \
  data.point_cloud.source=complete data.point_cloud.preload=false \
  test_data.pc_source=complete test_data.preload_point_clouds=false \
  test_data.metadata_group=right_full \
  'test_data.grasp_type_lst=["3_right_full"]' \
  test_data.test_scene_num=1 test_data.test_subset_seed=42 \
  algo.test_grasp_num=4 algo.test_topk=2 \
  num_workers=0 persistent_workers=false seed=42
```

輸出根目錄：

```text
$audit_root/sampling/shadowMulti_robotMultiHierar_ada_shadow/tests/step_050000/
```

Logger 每個候選寫一份 `.npy`，每份包含三個 `(58,)` 姿態、scene path、predicted grasp type 等。不是一份 `(N,29)` 的 Batched record。擴大到正式比較時，可使用原設定每 scene 100 candidates、保留 10，但應保持新輸出目錄並記錄 selection／seed；top-k 分數不等於物理抓取成功機率。

點雲／scene 必須相符：

- Complete cloud 原始檔位於 `processed_data/<object>/complete_point_cloud.npy`；loader 會依 scene 的 object scale／rotation／translation 轉到世界座標。
- 模型輸入為抽樣的 1,024 點，voxel size 0.005 m，且會中心化。
- `task_sample` 已把 centroid 加回有效右手姿態；橋接時不能再加 centroid 或 object translation。
- 想從真實相機的新 partial point cloud 抓取，還需要相機→scene/world 的標定、對應資料入口，以及 partial-PC 的模型品質驗證。不能把 complete-PC checkpoint 的 smoke 當作實機 partial-PC 泛化證據。

## 6. HUGS → Ada 的精確轉換

### 維度與關節順序

HUGS 每個 stage 的 layout：

```text
[right xyz, right wxyz, right 22 joints, left xyz, left wxyz, left 22 joints]
             0:7            7:29            29:36           36:58
```

right_full 只取右手 29 維。關節的來源順序必須讀**該 checkpoint 訓練資料的** `metadata.json`，不能依 URDF 的列舉順序猜。

本機 100k metadata 為 `TH → FF → MF → RF → LF`；Ada 的順序如下：

```text
rh_FFJ4 rh_FFJ3 rh_FFJ2 rh_FFJ1
rh_MFJ4 rh_MFJ3 rh_MFJ2 rh_MFJ1
rh_RFJ4 rh_RFJ3 rh_RFJ2 rh_RFJ1
rh_LFJ5 rh_LFJ4 rh_LFJ3 rh_LFJ2 rh_LFJ1
rh_THJ5 rh_THJ4 rh_THJ3 rh_THJ2 rh_THJ1
```

### 原點轉換

HUGS 的成功訓練資料由 Bench `_extract_robot_poses()` 取得 `rh_palm` 的 world pose；metadata 也明確記錄 `wrist_body_names=["rh_palm"]`。

Ada IK 求的是 `rh_dummy_RotZB`，而 `right_no_tendon.xml` 中 `rh_palm` 相對它平移 `[0,0,0.034]`，沒有額外旋轉。因此三個 stage 都要做：

```text
p_ada = p_hugs_palm - R(q_wxyz) @ [0, 0, 0.034]
q_ada = q_hugs
joints_ada = reorder_by_joint_names(joints_hugs)
```

這是 34 mm 的固定 frame offset，不是世界 Z 軸固定減 34 mm；手旋轉時偏移方向也旋轉。不能只截取 `[:29]`。

Ada 的 `Learning` converter 只複製 qpos，**不會**執行以上重排與 frame correction，也不會根據 raw `joint_names` 自動重排。`BODex` converter 有類似修正，但其 raw schema 完全不同，不能改 `task.data_name=BODex` 來讀 HUGS Learning 檔案。

### 執行橋接

接續第 5 節同一個 shell：

```bash
/home/hao-fang/hugs-workspace/HUGS-DexGraspBench/.venv/bin/python \
  /home/hao-fang/hugs-workspace/reviews/ada_shadow_dexlearn_20261101/export_shadow_to_ada.py \
  --input "$audit_root/sampling/shadowMulti_robotMultiHierar_ada_shadow/tests/step_050000" \
  --output "$audit_root/ada_raw" \
  --metadata /home/hao-fang/hugs-dataset/BimanBODex_shadow_100k_v1/shadow/right_full/metadata.json \
  --data-root /home/hao-fang/hugs-dataset
```

工具檢查 58 維、有限數值、單位四元數、22 個命名關節、`rh_palm` 契約與有效 scene path；僅輸出 `pred_grasp_type_id=3`，其他類型記錄為 skipped。輸出目錄必須不存在，保留原始檔名階層與 `export_manifest.json`。

此工具適用本次確認的 HUGS Shadow 契約。舊單手 DexLearn 輸出可能已是 Ada base frame，不能再套一次 34 mm 修正。

## 7. 將橋接資料送進 Ada

先完成第 3 節環境準備並 activate Ada 環境。下列 `$audit_root` 沿用第 5 節；換 shell 時需重新設為那次的絕對路徑。所有階段保持相同 `exp_name`、`output_root`、`data_root`。

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
ada_out="$audit_root/ada_pipeline"

env -u PYTHONPATH python src/main.py \
  setting=tabletop hand=shadow task=format exp_name=handoff n_worker=1 \
  output_root="$ada_out" data_root=/home/hao-fang/hugs-dataset \
  log_dir="$ada_out/log/format" \
  task.data_name=Learning task.data_path="$audit_root/ada_raw" task.max_num=-1

env -u PYTHONPATH python src/main.py \
  setting=tabletop hand=shadow task=dummy_arm_qpos exp_name=handoff n_worker=1 \
  output_root="$ada_out" data_root=/home/hao-fang/hugs-dataset \
  log_dir="$ada_out/log/dummy_arm_qpos" task.device=cpu task.max_num=-1

env -u PYTHONPATH python src/main.py \
  setting=tabletop hand=dummy_arm_shadow task=control_eval exp_name=handoff n_worker=1 \
  output_root="$ada_out" data_root=/home/hao-fang/hugs-dataset \
  log_dir="$ada_out/log/control_eval" \
  task.method=ours task.offsets='[0.0]' task.max_num=-1 \
  task.debug_viewer=false

env -u PYTHONPATH python src/main.py \
  setting=tabletop hand=dummy_arm_shadow task=control_stat exp_name=handoff n_worker=1 \
  output_root="$ada_out" data_root=/home/hao-fang/hugs-dataset \
  log_dir="$ada_out/log/control_stat" \
  task.method=ours task.setting_name=dist_0 \
  task.input_report="$ada_out/log/control_eval/run_report.json"
```

逐條檢查 exit code 和 report 再執行下一階段；不可只靠最後一條命令判斷整條 pipeline 成功。`0` 為乾淨完成，`1` 為 execution error／solver degradation，`2` 為前置檢查失敗。

本次實跑前兩階段時，用 Bench 的 Python、隔離 `asset_root` 與 pinned PK `PYTHONPATH`，IK 使用 `cuda:0`。CPU pinned IK 另由原測試通過。上面後兩條為**環境補齊後的既有 CLI 配方，本次尚未通過控制驗證**。

raw 目錄只放 grasp `.npy`；scene `.npy` 保持在 dataset 的 `object/DGN_2k/scene_cfg/`，不要混進 raw 搜尋範圍。橋接已將 scene path 轉成絕對路徑。物件仍須有 scene 引用的 mesh、`info/simplified.json` 與碰撞 mesh；只有點雲和 checkpoint 不足以執行 Ada MuJoCo。

## 8. 如何評估「回到 Shadow 是否改善」

建議先分三層建立 baseline：

1. **控制器 baseline**：補齊環境，跑 bundled Shadow quick，檢查當次 reports 與既有 expected classifications，確認 Ada 自身可重現。
2. **資料接入 baseline**：沿用本報告的命名關節／34 mm frame 轉換；先少量檢查 MuJoCo FK 與場景位置，再擴大採樣。
3. **模型品質比較**：100k 和 752k 在相同 test scenes、點雲種類、proposal budget、selection、seed 與 Ada 設定下比較。不要同時改 controller force 或 pregrasp 外推來掩蓋上游差異。

既有 HUGS Bench 調查中，100k Shadow 模型的 seen 組為 21/260（8.08%），unseen 組為 331/6,720（4.93%）；來源是 [seen-scene 調查](../../HUGS-DexLearn/docs/dexlearn_seen_training_scene_evaluation.md)。這是歷史 Bench 結果，兩組場景不同，且不是 Ada `ours` 的測試。它提醒我們：**換回 Shadow 或換控制器，不會自動保證 learned grasp 品質良好**。

Ada 的主要成功率分母只含 `success + failure`；Bench 上述結果保留初始穿透樣本在總分母，因此不能直接比較百分比。比較時至少同時列出總候選數、invalid initialization、solver degraded、execution error、有效 rollout 成功率，以及以全部候選為分母的成功數比例。

## 9. 原始碼索引

| 議題 | 來源 |
| --- | --- |
| Ada 公開流程／支援 | [README](../README.md)、[run_example.sh](../script/run_example.sh) |
| Ada Shadow 關節順序 | [shadow.py](../src/ada_grasp_ctrl/utils/robots/shadow.py) |
| Ada Learning／BODex 差異 | [convert_format.py](../src/ada_grasp_ctrl/tasks/convert_format.py) |
| Ada IK target／seed | [dummy_arm_qpos.py](../src/ada_grasp_ctrl/tasks/dummy_arm_qpos.py)、[pk_helper.py](../src/ada_grasp_ctrl/utils/pk_helper.py) |
| 34 mm offset | [right_no_tendon.xml](../assets/hand/dummy_arm_shadow/right_no_tendon.xml) |
| HUGS 三階段採樣／decentering | [sample.py](../../HUGS-DexLearn/dexlearn/task/sample.py) |
| HUGS 每候選存檔 | [logger.py](../../HUGS-DexLearn/dexlearn/utils/logger.py) |
| HUGS right-only 的雙手表示 | [robot_multidex.py](../../HUGS-DexLearn/dexlearn/dataset/robot_multidex.py) |
| HUGS 模型設定 | [robotMultiHierar.yaml](../../HUGS-DexLearn/dexlearn/config/algo/robotMultiHierar.yaml) |
| 訓練姿態來源 frame | [Bench BaseEval](../../HUGS-DexGraspBench/src/task/eval_func/base.py) 的 `_extract_robot_poses()` |
| 過往資料座標／loss 調查 | [DataLoader audit](../../HUGS-DexLearn/docs/dexlearn_dataloader_coordinate_audit.md)、[loss audit](../../HUGS-DexLearn/docs/dexlearn_right_full_loss_supervision_audit.md) |

版本：Ada `a89c632`；HUGS-DexLearn `347e45e`；HUGS-DexGraspBench `1da1b5f`。本次所附工具與輸出位於 workspace 的 `reviews/ada_shadow_dexlearn_20261101/`，是調查證據，不是已納入 Ada 公開支援的通用 converter。
