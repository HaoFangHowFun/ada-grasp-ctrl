# Shadow pilot 物體尺度調查

日期：2026-11-01。針對 localhost 預覽中 `g0035`、`g0045` 過大，以及其他物件過小的問題。

後續已完成限制為訓練尺度的新一批 50 grasps，見[新批次結果與預覽](shadow-scale-matched-pilot-results.md)。

## 結論

物體大小差異確實存在，來自抽中的原始 scene，沒有發現 Ada 或 viewer 額外縮放。**本次 pilot 的 scene 抽樣沒有套用 Shadow right_full 的訓練尺度限制，50 筆中有 25 筆落在資料集訓練尺度範圍之外。** 這是實驗抽樣設計的疏漏，不能把此批 4% 成功率當成模型在正常尺度分布下的表現。

但尺度問題並未解釋全部失敗：範圍內的另外 25 筆仍有 22 筆初始化無效、1 筆 failure、2 筆 success。此處是事後分組診斷，不是換一個分母重新宣稱實驗成功。

## 實際尺寸

下列是 collision mesh 在物體自身座標軸的 bounding-box 尺寸，單位 cm。物體旋轉後的世界軸尺寸會不同。每列包含連續五個候選，例如 g0035–g0039 共用同一物體與 scene。

| 第一筆 ID | 物體 | Scene scale | 實際尺寸（cm） | 屬於訓練尺度 |
| --- | --- | ---: | --- | --- |
| g0000 | bottle | 0.028 | 1.7 × 5.2 × 1.7 | 否 |
| g0005 | camera | 0.227 | 28.1 × 20.7 × 29.9 | 是 |
| g0010 | jar | 0.162 | 19.3 × 17.3 × 19.1 | 是 |
| g0015 | mug | 0.227 | 22.2 × 25.2 × 31.3 | 是 |
| g0020 | mug | 0.300 | 30.3 × 30.3 × 43.0 | 否 |
| g0025 | tetra pak | 0.060 | 5.7 × 5.8 × 9.0 | 是 |
| g0030 | sauce container | 0.194 | 17.1 × 27.9 × 21.8 | 是 |
| g0035 | cereal box | 0.300 | 12.2 × 35.4 × 47.9 | 否 |
| g0040 | mouse | 0.020 | 2.0 × 3.4 × 0.9 | 否 |
| g0045 | table clock | 0.300 | 33.5 × 44.6 × 24.2 | 否 |

`scale` 是乘在 normalized mesh 上的倍率，不等同於「物體最長邊的公尺數」。例如 g0035 原 mesh 的 Z extent 約 1.597，乘 0.3 後就是 0.479 m。資料中的物體是合成尺度變體，不保證符合日常同類物品的大小。

## 尺度追蹤與交叉驗證

1. **Scene**：從 `.npy` 內 `scene[task.obj_name].scale` 讀取實值，不從檔名猜測。`scale003` 的實值可能是 0.028。
2. **DexLearn**：`robot_multidex.py::_transform_complete_pc` 執行 `(points * scale_xyz) @ R.T + translation`；這次設定沒有額外 `pc_runtime_scale` override，預設為 1。
3. **Ada converter**：`convert_format.py::_object_fields` 將 scene scale[0] 存成 obj_scale；此批十個 scene 全部為等向縮放。
4. **Ada MuJoCo**：`hand_util.py::_add_object` 對 collision mesh 套一次 `[obj_scale] * 3`。
5. **Viewer**：mjviser 讀取 MuJoCo 已編譯的 `model.mesh_vert`，並使用 geom pose，沒有再次乘 scene scale。

已對十個物體逐一計算原始 collision vertices 經 scene scale／pose 後的世界 bounds，再和最終 replay XML 載入的 MuJoCo vertices 比較。最大 bounds 誤差 **8.91 × 10⁻⁹ m**。所有 prepared scale 與 scene scale 一致。

完整點雲與 collision mesh 的局部尺寸也接近，差異來自取樣、float16 點雲和 collision approximation；沒有看到倍數級尺寸不一致。例如 g0035 的點雲約 12.30 × 35.48 × 48.02 cm，collision mesh 約 12.20 × 35.37 × 47.91 cm。DexLearn 條件輸入本身就看到了這個大物體。

## 為甚麼抽到這些尺度

原抽樣先固定 test objects，再從各物體全部可用的 tabletop scenes 隨機取一個。它只檢查資產與 PC 可用性，沒有檢查該 scene 是否落在 right_full 使用的尺度集合。

實際掃描 `BimanBODex_shadow_100k_v1/shadow/right_full`，共 4,142 個收集後的 scene files；其中 train split 為 3,334 scenes、40,814 筆抓取紀錄。其尺度集合為：

```text
0.060, 0.081, 0.106, 0.133, 0.162, 0.194, 0.227
```

這與 `HUGS-BODex/src/curobo/content/configs/baselines/heur_multi_shadow.yml` 的 right_full 尺度標籤 `[0.06, 0.08, 0.11, 0.13, 0.16, 0.19, 0.23]` 相符。BODex 以四捨五入的檔名標籤篩選；精確尺度仍應讀 scene 內容。

原始場景庫另有小型尺度與 0.300 等大型尺度，並非所有場景尺度都適用於同一 grasp type。例如 heuristic 的 0.300 出現在 both_full 尺度集合，而不在 right_full 集合。

即使只看範圍內的五個 pilot scenes，分布也偏大：train split 中 **81.68% 的抓取紀錄**來自 scale ≤ 0.133，但這五個 scenes 只有一個符合。這裡的抓取加權比例與 scene 比例權重不同，僅用於指出分布不具代表性，不能作為嚴格配平比較。

## 對成功率的影響

| 分組 | 筆數 | Phase 1 | Phase 2 |
| --- | ---: | --- | --- |
| 訓練尺度集合內 | 25 | 22 invalid、1 failure、2 success | 22 invalid、1 failure、2 success |
| 訓練尺度集合外 | 25 | 16 invalid、9 failure | 16 invalid、7 failure、2 degraded |

範圍外案例沒有成功；但範圍內也大量初始化無效，因此不能斷言只要調整尺寸就能修好。還需要在合理、事先固定的尺度分布下，對照資料集抓取和模型預測，再檢查手模型／碰撞相容性。

## 建議修正方式

- 保留本次原始結果與 localhost 尺寸，不直接把物體任意放大／縮小。現有預測位置是以原尺寸點雲產生，單獨改 mesh 會破壞對應關係。
- 下一批從 right_full 的七個實際尺度選 scenes，在推理前凍結名單；優先讓尺度分布接近訓練資料，或明確設計逐尺度比較。
- 使用新的 scene 尺度重新生成 DexLearn 預測，再做共用 IK 和兩個 phase。新 suite 與舊結果分開記錄。
- 若目標是日常大小的實物，應另依物體的實際公分尺寸定義測試集合，不能只憑 scale 數字判斷大小。

本次只調查，未更動原始資產、scene、預測、controller 參數或結果。

證據：[尺度量測與資料分布 JSON](../output/shadow-phases/object_scale_audit.json)；[可重跑的調查腳本](../../reviews/ada_shadow_phases_20261101/audit_object_scales.py)。
