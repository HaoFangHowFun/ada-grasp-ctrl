# 本機 Ada 相容環境

建立日期：2026-11-01。依 `environment.yml`，以 micromamba 建立獨立環境：

```text
/home/hao-fang/hugs-workspace/ada-grasp-ctrl/.conda
```

已初始化 repo 鎖定的三個 submodules，並 editable 安裝 Ada、`pytorch_kinematics` 和 `utils_python`。沒有修改既有 HUGS-DexLearn／HUGS-DexGraspBench 的環境，也沒有修改控制器或套件版本宣告。

## 已安裝版本

| 套件 | 版本 |
| --- | --- |
| Python | 3.10.20 |
| Pinocchio | 3.0.0，Conda build |
| PyTorch | 2.2.2，CUDA 12.1 build |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| MuJoCo | 3.6.0 |
| pytorch_kinematics | 0.7.5，repo pinned source |

PyPI 沒有 Pinocchio 3.0.0 的 wheel，因此最終使用專案原本的 Conda 路線。`pip check` 通過，Pinocchio 能載入原始 Shadow MJCF 並得到 28 個自由度。

## 使用方式

不需要先安裝系統 Conda 或使用 `conda activate`；可直接指定環境中的 Python。清除 shell 的 `PYTHONPATH`／`PYTHONHOME`，避免 ROS Jazzy 的 Python 3.12 packages 混入 Python 3.10：

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
env -u PYTHONPATH -u PYTHONHOME .conda/bin/python src/main.py --help

env -u PYTHONPATH -u PYTHONHOME \
  PYTHON_BIN="$PWD/.conda/bin/python" \
  bash script/run_example.sh shadow quick
```

**這個環境的 Ada IK 請使用 CPU：`task.device=cpu`。** 本機 RTX 5090 是 `sm_120`，而 repo 鎖定的 PyTorch 2.2.2 只列出至 `sm_90` 的支援。`torch.cuda.is_available()` 不代表這個舊版 Torch 支援該 GPU。MuJoCo 和接觸最佳化本來就可在 CPU 執行；既有 DexLearn 的新版本 GPU 環境保持不變。

原 `script/run_example.sh shadow full` 會使用預設 CUDA IK，不能在這台機器直接當作 CPU full 命令。可使用本次驗證工具跑相同四階段、僅將 IK device 改為 CPU：

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
run_root="$PWD/output/examples/shadow/full-cpu-$(date -u +%Y%m%dT%H%M%S)"
env -u PYTHONPATH -u PYTHONHOME .conda/bin/python \
  ../reviews/ada_environment_20261101/run_shadow_full_cpu.py "$run_root"
```

輸出目錄必須不存在。工具沿用 bundled raw Learning fixture、原始 `ours` 參數與每次的確切 control report，不改手型／科學指標。若要接自己的 DexLearn 資料，參考 [接入報告第 7 節](shadow-dummy-arm-dexlearn-investigation-2026-11-01.md#7-將橋接資料送進-ada)，將其中 `python` 換成 `.conda/bin/python`，保留 `task.device=cpu`。

## 實際驗證結果

- 核心套件匯入、Pinocchio MJCF loader、CPU Torch 運算：通過。
- `python -m pip check`：`No broken requirements found.`
- 完整既有測試：**124 tests，全部通過**。
- bundled 單筆 Shadow full：四階段 exit code 都是 0；**成功抬升 1 筆、失敗 0、invalid 0、solver degraded 0、execution error 0**。
- Shadow 100 筆 quick：完成 100 筆，**68 success、4 failure、21 invalid initialization、7 solver degraded、0 execution error**。

**quick 的嚴格分類比對未通過，不能宣稱 release gate 通過。** 預期 inventory 是 66 success／4 failure／21 invalid／9 degraded，有以下三筆差異：

| 物件／樣本 | 既有預期 | 新環境結果 |
| --- | --- | --- |
| `core_mug_73b8b6456221f4ea20d3c05c08e26f`，`scale012_pose000_0/partial_pc_00_6` | failure | success |
| `ddg_gd_tape_poisson_000`，`scale010_pose004_0/partial_pc_01_4` | solver_degraded | success |
| `sem_Pizza_99dadb83b9bcf2498af30108ea9ccb6c`，`scale008_pose000_0/partial_pc_01_3` | solver_degraded | failure |

這些是科學分類／求解狀態差異，沒有 execution error。沒有改 golden、放寬容差或接受基準變更。初次 fixture／manifest／報告一致性檢查通過，但目前只有 expected classifications，未取得建立該 inventory 時的完整數值軌跡與 Conda build lock，尚未確定差異根因。環境已能執行原流程，不等同於完全重現發布基準。

以相同環境、seed、worker 數與輸入另開新目錄重跑一次，**100 筆分類逐筆與第一次一致**；仍是上述三筆與歷史基準不同。這確認了兩次測試的分類穩定性，沒有進一步宣稱所有浮點軌跡逐位相同。

## 證據

- [環境與 GPU 支援範圍](../../reviews/ada_environment_20261101/environment.json)
- [Conda 套件完整清單](../../reviews/ada_environment_20261101/conda-explicit.txt)、[pip freeze](../../reviews/ada_environment_20261101/pip-freeze.txt)
- [NumPy build／BLAS 資訊](../../reviews/ada_environment_20261101/numpy-build.txt)
- [安裝日誌](../../reviews/ada_environment_20261101/installation.log)、[124 個測試結果](../../reviews/ada_environment_20261101/tests.log)
- [full 摘要](../../reviews/ada_environment_20261101/shadow-full.log)、[full 當次統計](../output/examples/shadow/environment-20261101-full-cpu/control_stat_res/dist_0_ours_default.yaml)
- [quick 當次統計](../output/examples/shadow/environment-20261101-quick/control_stat_res/dist_0_ours_default.yaml)、[逐筆分類差異](../../reviews/ada_environment_20261101/shadow-quick-comparison.json)
- [兩次 quick 分類一致性](../../reviews/ada_environment_20261101/shadow-quick-repeat-comparison.json)、[首次 quick 日誌](../../reviews/ada_environment_20261101/shadow-quick.log)、[重跑日誌](../../reviews/ada_environment_20261101/shadow-quick-repeat.log)
