# 752k 同一批預測：Ada 與 DexGraspBench 初始化差異

日期：2026-11-01。使用目前 8081 預覽那一批 **50 個原始 752k 預測**，沒有重新抽樣、重新推理或替換樣本。DexLearn 是生成器，這裡所稱「原 DexLearn 評估環境」是 HUGS-DexGraspBench。

## 實際結果

| 評估配置 | success | failure | 初始化拒絕 | 總數 |
| --- | ---: | ---: | ---: | ---: |
| Ada Phase 1，保留原始三姿態 | 22 | 9 | 19 | 50 |
| DexGraspBench，保留原始三姿態 | **42** | 8 | **0** | 50 |
| DexGraspBench，歷史 pregrasp ratio=2 | **42** | 8 | **0** | 50 |

兩個 benchmark 實驗都保留原始 squeeze（ratio=1），並使用 native 10 mm 初始穿透門檻。歷史 pregrasp ratio=2 來自既有 752k 評估的保存設定及 `process_learning_grasp_types.py`；benchmark task YAML 本身的 pregrasp 預設為 1。

兩組 benchmark 雖同為 42 成功，逐筆分類並不完全相同：`g0000` 在 ratio 1 時 failure、ratio 2 時 success；`g0040` 相反。

## 先更正「19 筆 penetration」的解讀

Ada 的 `invalid_initialization` 不等同於「穿透超過 5 mm」：原生判定是手／物體穿透超過 5 mm，或手自碰撞超過 5 mm，**或初始接觸力檢查超過 max_force=1**。

這批的 19 筆無效初始化中：

- 5 筆手／物體穿透超過 5 mm，這些也超過接觸力門檻。
- 另外 14 筆手／物體與自碰撞深度均未超過 5 mm，是由接觸力條件拒絕。
- 沒有額外的自碰撞深度單獨拒絕案例。

例如 `g0039` 的手／物體穿透只有約 **0.041 mm**，但初始化接觸力檢查值約 **64.9**，仍被拒絕。這是 `mj_resetDataKeyframe` 後 `mj_forward` 計算出的接觸約束反力，不是手完成接近後的穩定抓握力。小幅幾何重疊在這個動力模型下也可能產生大的初始反力。

19 筆被 Ada 擋下的案例中，**17 筆在保留原姿態的 benchmark 中最後成功**，2 筆失敗。因此不能直接把 Ada 的 19 筆初始化拒絕當成模型生成了 19 個嚴重不可用姿態。

## 兩邊是否真的有不同的幾何穿透

對每筆同一原始 pregrasp，分別建立兩邊模型並直接量測初態接觸，沒有推進物理：

| Benchmark 初態 | 原始 pregrasp | 歷史 ratio=2 |
| --- | ---: | ---: |
| 有手／物體幾何重疊（距離 < 0） | 21 | 2 |
| 超過 5 mm | 5 | 0 |
| 超過 10 mm | 0 | 0 |
| 最大穿透 | 7.204 mm | 3.939 mm |

**在原始 pregrasp 下，Ada 與 benchmark 的逐筆手／物體最小距離幾乎相同，最大差約 0.0000131 mm。** 這批沒有顯示「同一初始姿態只在 Ada 才多穿入幾公分」的情形。

所以答案是：原環境也存在一些小幅重疊；但它接受最多 10 mm 的初始穿透，沒有 Ada 的初始 max_force 拒絕條件，因此這批沒有初始化拒絕。歷史入口還會先擴大 pregrasp，讓重疊案例從 21 減為 2。

這個幾何結論只針對本批 752k 三尺度初態，不回推成先前 100k 異常姿態也一定如此，也不代表兩套動力模型完全一致。

## 為何成功率不能直接對比成控制器優劣

| 項目 | Ada Phase 1 | DexGraspBench |
| --- | --- | --- |
| 初始穿透門檻 | 5 mm | 10 mm |
| 初始接觸力拒絕 | max_force=1 | 無此 gate |
| Hand base | 6-DOF dummy arm | mocap base |
| 手指致動 | no_tendon，各關節 actuator | tendon 耦合 actuator |
| 物體質量 | 固定 0.1 kg | scale_power density，最大 0.5 kg |
| 摩擦 | [0.6, 0.02] | [0.8, 0.02] |
| 各移動階段 | approach／squeeze／lift 各 2 s | 各 1 s |
| Lift | 0.2 m | 0.1 m |
| 成功判定 | lift Z 誤差 < 0.1 m | 位置誤差 < 0.05 m 且角度誤差 < 15° |

上述差異保留原樣，沒有為了提高 Ada 成功率而改門檻。兩邊格式轉換後的 palm pose 和每個 named finger joint 都已逐項對回同一 raw prediction。來源檔案的 hash 與 suite manifest 一致。

下一個有用的實驗是明確命名、獨立保存的初始化協定對照，先隔離 pregrasp 調整與拒絕條件，再逐項對齊動力設定；不能直接修改公開 defaults、取消碰撞或把新數字混入目前 Phase 1 結果。

## 證據與目前預覽

所有新結果存於 `output/shadow-phases/dexlearn-752k-three-scales-v1/benchmark-environment-audit/`：

- [逐筆跨環境比較與初態深度](../output/shadow-phases/dexlearn-752k-three-scales-v1/benchmark-environment-audit/comparison.json)
- [Benchmark 原始／外推初態量測](../output/shadow-phases/dexlearn-752k-three-scales-v1/benchmark-environment-audit/initial_contacts.json)
- [命令與退出碼](../output/shadow-phases/dexlearn-752k-three-scales-v1/benchmark-environment-audit/commands.json)
- [執行腳本](../../reviews/ada_shadow_phases_20261101/compare_benchmark_environment.py)
- [初態量測腳本](../../reviews/ada_shadow_phases_20261101/benchmark_initial_contacts.py)

兩組 benchmark 都完成 50 筆評估；沒有輸入缺件。僅新增診斷輸出，沒有修改評估器或 controller 原始碼。**8081 仍顯示先前 Ada Phase 1 的 22/50 結果**，沒有偷偷改成 benchmark 的軌跡。
