# Ada 原生資產：Shadow 752k Phase 1 小規模測試

日期：2026-11-01。保持摩擦 0.6、完整對齊初始化、原始 squeeze，使用 Ada repo 隨附的物體資產重新產生 grasp，僅執行 Phase 1（op，無 Ada controller）。

## 結果

**50 筆全部完成：30 success、20 failure，成功率 60%；無 invalid_initialization、solver_degraded、execution_error 或 preparation_error。** control_eval／control_stat 退出碼均為 0。10 個物體各至少有一筆成功。

模型仍為 `right_full_752k_v1/ckpts/step_050000.pth`。每 scene 產生 100 proposals，再依原生 per-type top-k 選 5 筆；共 10 scenes、50 grasps。這些是本次重新推論的 pregrasp／grasp／squeeze，沒有使用 quick fixture 裡保存的舊 grasp。

## 這次的資產到底來自哪裡

實際物體 mesh、碰撞分解、質量資訊以及原場景來自 repo 的：

`examples/assets/object/DGN_2k/{processed_data,scene_cfg}`

這是 Ada 隨附的 DGN 子集，本身不是 Ada 新造的獨立物件資料庫。已將每個實際使用的 object dependency 對回 `examples/quick_manifest.json` 中的 SHA256；prepared input 的 obj_path 均解析到 repo 的上述原生目錄，沒有使用前一批外部 BODex processed objects 代替。

原生場景保留其 scale、pose、mesh；本次不重縮放、不重新落桌、不修改原檔。從同時符合以下條件的所有十個不同物體選取：

1. 存在於 Ada 隨附資產與 scene。
2. object ID 屬於原 DexLearn `valid_split/test.json`。
3. 原生 scene scale 為 0.06、0.08 或 0.10。

其中一個 camera 有兩個可用 scene，使用 seed 42 選一個；其餘各一個。選擇在推論前寫入 manifest，不依 grasp 結果篩選。最終分布 0.06 ×2、0.08 ×6、0.10 ×2。

**原生 Ada 場景的精確尺度是 0.06／0.08／0.10，不是前一批 BODex 的 0.060／0.081／0.106。** 此處優先保留原生資產測試條件；沒有把場景檔名當作精確尺度，也沒有擅自把 native mesh 改成前一批的尺寸。

## 點雲與尺寸一致性

隨附資產沒有本模型入口需要的 complete point cloud，因此從每個原生 `mesh/simplified.obj` 以表面積加權抽取 16,384 點，保留原 mesh 座標，存成 float32。取樣 seed=42+排序後的 object index。

點雲不預先乘 scene scale、不中心化或重新 normalize；DexLearn 原 loader 按 scene 執行一次 scale／rotation／translation，再按原模型設定抽 1024 點及中心化。每個物體的 scene、點雲、mesh dependencies 與 checkpoint hashes 都保存在 manifest。資料橋接只建立在新 output/dataset 下，mesh／info／urdf 指回原生資產，原始資產完全未改動。

下表尺寸為原生 simplified mesh 的物體局部 bounding box（cm），世界軸尺寸會隨 pose 改變。

| 樣本 | 物體 | Scale | 尺寸 cm | 成功 |
| --- | --- | ---: | --- | ---: |
| g0000–g0004 | `core_camera_6d036fd1c70e5a5849493d905c02fa86` | 0.06 | 8.1 × 5.4 × 7.3 | 3/5 |
| g0005–g0009 | `core_camera_7077395b60bf4aeb3cb44973ec1ffcf8` | 0.08 | 7.7 × 11.8 × 7.8 | 2/5 |
| g0010–g0014 | `core_jar_275cc21fad6aa1b6bfa78a7d067c469c` | 0.10 | 11.0 × 13.1 × 11.0 | 1/5 |
| g0015–g0019 | `core_mug_24651c3767aa5089e19f4cee87249aca` | 0.08 | 8.2 × 8.6 × 11.0 | 5/5 |
| g0020–g0024 | `core_mug_c0c130c04edabc657c2b66248f91b3d8` | 0.08 | 8.1 × 9.6 × 10.2 | 5/5 |
| g0025–g0029 | `ddg_gd_tape_poisson_000` | 0.10 | 13.9 × 13.9 × 5.0 | 2/5 |
| g0030–g0034 | `ddg_gd_tetra_pak_poisson_001` | 0.08 | 5.9 × 6.2 × 13.6 | 3/5 |
| g0035–g0039 | `mujoco_Lego_Friends_Advent_Calendar` | 0.06 | 9.9 × 2.0 × 6.8 | 5/5 |
| g0040–g0044 | `sem_Detergent_547fa0085800c5c3846564a8a219239b` | 0.08 | 8.9 × 4.4 × 12.8 | 1/5 |
| g0045–g0049 | `sem_TapeDispenser_844516b6fe304dafe36f6f37ca7926` | 0.08 | 13.0 × 5.3 × 8.3 | 3/5 |

## 控制與初始化

- 使用 Shadow dummy arm、原生 op Phase 1。
- 摩擦 `[0.6, 0.02]`、物體質量 0.1 kg、offset 0、seed 12、單 worker。
- pregrasp 平移／手指依 `grasp + 2*(pregrasp-grasp)` 外推；quaternion 不變。
- 使用 palm offset 34 mm 轉換 dummy-arm 平移；不外推 Euler 旋轉。
- 初始穿透門檻 10 mm，停用初始 force gate。
- grasp／squeeze 與原 prepared 輸入逐元素相同，沒有增加 squeeze 深度。
- 沿用原生 approach／squeeze／lift、20 cm lift 與成功判準，沒有跑 Phase 2。

## 驗證

- 50 筆來源 hash、原生資產 hashes、scene byte-identical copy 均通過檢查。
- 150 個有效 target 與 replay target 完全一致；每個完整 episode 有 62 frames。
- 150 個 FK pose 最大 palm 位置誤差 **0.418 mm**，旋轉誤差 **0.00798 rad** 以下；named finger joint 對應通過檢查。
- 原生碰撞 vertices 經 scene scale／pose 變換後，與 MuJoCo 編譯後世界 bounds 最大差 **3.90e-9 m** 以下；沒有發現單位或重複縮放問題。
- 新增實驗／驗證腳本通過 Ruff 與 py_compile；沒有修改 controller 或公開 defaults。

這批 60% 與前一批外部場景摩擦 0.6 Phase 1 的 68%，不是只替換 asset 的單因子比較：物件集合、scene pose、精確尺度與點雲取樣也不同，不能把差異單獨歸因於 Ada 資產。

## 預覽與產物

<http://localhost:8081> 已切換本次 Ada 原生資產。選 `Phase 1 replay` 再播放可看實際動態；`Target poses` 是靜態目標。g 編號是本批新候選，不能與前一批同號當成同一 grasp。

結果根目錄：`output/shadow-phases/ada-assets-752k-phase1-v1/`。

- `scene_manifest.json`：推論前選定場景、資產／checkpoint／點雲 hashes、尺寸與來源。
- `sampling_command.json`／`sampling_exit.json`／`sampling.log`：推論命令、退出碼及日誌。
- `sampling/`：新產生的 50 筆模型預測。
- `raw_suite/`：未外推的原生格式準備與 IK 結果。
- `matched_suite/`：完整對齊初始化後的輸入，保留 parent prepared hashes。
- `run/phase1/`：eval／stat reports、resolved configs、逐筆 metrics、控制記錄與回放。
- `summary.json`／`validation.json`：結果、FK／尺寸驗證與逐物件計數。

工作區腳本：`reviews/ada_shadow_phases_20261101/run_ada_assets_phase1.py`、`validate_ada_assets_phase1.py`。所有先前試驗結果均保留。
