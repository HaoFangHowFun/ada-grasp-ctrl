# Shadow Phase 1 / Phase 2 CPU workers 小規模測速

本機測試中，8 workers 最快，兩階段控制批次合計較現有串行方式快 **1.81 倍**。2、4、8 workers 的分類、軌跡及 solver 診斷均與串行基準一致。建議後續將 8 workers 作為本機正式回歸的候選設定；本次未修改主測試 sh 的串行預設，也未執行完整 2,000 次回歸。

## 固定條件與測試方法

- CPU：AMD Ryzen 7 9800X3D，8 核心、16 logical CPUs；使用 native MuJoCo CPU 模擬及 Ada CPU solver。
- 使用已凍結的 `output/regression/shadow-native-100x10-v1/suite_manifest.json`，模型來源為 `right_full_752k_v1/ckpts/step_050000.pth`；測速不包含重新推論或 IK。
- Ada 資產尺度 0.06、0.08、0.10、0.12；selection seed 42，各尺度抽 2 個物體，各取第一個 scene 的前 2 個 grasp，共 16 個輸入。完整選擇保存在 `selection.json`。
- Phase 1：`op`；Phase 2：`ours`。摩擦 `[0.8, 0.02]`、完整對齊初始化、pregrasp extrapolation 2、無額外 squeeze；最大初始 penetration 0.01 m，無初始力上限。
- 評估 seed 12，保留完整 suite 的 sample_index，避免子集重新編號導致 seed 改變。
- 先跑一次現有直接呼叫的串行基準，再跑 Pool workers 1/2/4/8，第二輪逆序 8/4/2/1。各設定獨立 subprocess，彼此不並行。
- 共 9 組 × 16 grasp × 2 phases = **288 次執行**。
- 保留既有 thread 環境，未設 OMP/MKL/OpenBLAS thread 上限。Pool worker 完成單筆時觀測到 8 threads；這不代表 8 threads 都在持續運算，亦不足以判定速度瓶頸。

## 速度

秒數為兩輪中位數；串行基準只有一次。計時包含 pool 建立/關閉、控制模擬及 XML/NPZ 輸出，排除 import、preflight、輸入 hash 驗證及 control_stat。

| 設定 | Phase 1 | Phase 2 | 合計 | 相對串行加速 |
| --- | ---: | ---: | ---: | ---: |
| 現有串行 | 3.59 s | 13.62 s | 17.22 s | 1.00× |
| Pool 1 worker | 3.57 s | 13.74 s | 17.31 s | 0.99× |
| Pool 2 workers | 2.27 s | 9.79 s | 12.06 s | 1.43× |
| Pool 4 workers | 1.60 s | 9.15 s | 10.75 s | 1.60× |
| Pool 8 workers | 1.24 s | 8.26 s | 9.49 s | 1.81× |

8 workers 的 Phase 2 兩輪為 7.96 / 8.55 秒。包含 subprocess 啟動、檢查與統計的整體時間為 11.20 / 11.91 秒；上述表格不是整條 sh 的端到端時間。

只有 16 個輸入，工作量差異、啟動成本與排程都會影響加速比；不能直接拿此表線性預測完整 2,000 次執行時間。尚未測試限制每個 worker 的 BLAS threads。

## 結果一致性

8 組 Pool 執行各與串行比對 32 筆，共 **256 組 episode 配對**：

- 每筆 sample ID、完整 sample_index、sample_seed、輸入 hash、分類及 evaluation 訊息完全一致。
- 所有設定 Phase 1 均為 **14 success / 2 failure**；Phase 2 均為 **15 success / 1 failure**。本次沒有 invalid、solver_degraded 或 execution_error，因此尚未涵蓋這些案例的平行一致性。
- 完整 replay 的 targets、metadata、初始狀態及所有 frame 的時間、qpos、qvel、ctrl、act、object_pose、reference、command 通過原生容差 `rtol=1e-5, atol=1e-6`。
- 原生 control record 的 contacts 順序、階段、優化結果及 solver_diagnostics 一併比對；含 accepted、decision、message、action_applied 等欄位。僅排除 `t_` 開頭的效能計時欄位，未放寬數值容差。

**XML 檔案並非全部逐 byte 相同。** 4/8 workers 共 20 個 XML 配對有差異，全部僅為 `.obj` mesh 是否明寫 `content_type="model/obj"`；在比對副本移除這個可由副檔名推得的屬性後，其餘 XML 完全相同。原始 XML 未修改，差異清單保存在 JSON 的 `xml_byte_differences`。上述軌跡與 solver 一致性不依賴此 XML 正規化。此現象的序列化成因尚未追查，不能宣稱輸出檔案 bitwise reproducible。

## 產物與後續

- [完整數值與一致性報告](../output/regression/worker-benchmark-v1/comparison.json)
- [固定選樣與執行環境](../output/regression/worker-benchmark-v1/selection.json)
- 原始輸出：`output/regression/worker-benchmark-v1/<case>/phase{1,2}/`；每個 case 附 `timing.json`、獨立 log 與 process 計時。
- 測速工具：workspace 的 `reviews/ada_shadow_phases_20261101/benchmark_workers.py`。
- 比對工具：workspace 的 `reviews/ada_shadow_phases_20261101/analyze_workers.py`。

後續接入主 sh 時應保留可調 workers、完整 sample_index、每筆 seed 與既有錯誤回報規則，且以 sample ID 整理結果，避免完成順序影響報告。8 workers 是本次速度最佳的候選；完整回歸仍需確認包含 solver_degraded 案例時的一致性。
