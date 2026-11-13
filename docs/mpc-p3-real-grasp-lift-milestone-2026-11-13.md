# Milestone：P3 實機 MPC 抓取與 15 公分抬升完整成功

日期：2026-11-13（America/Los_Angeles）

平台：右側 FANUC CRX-5iA + Sharpa Wave

模式：`mpc_active`，ROS domain 87

執行：`active_20261113_132624`

候選：`grouplive_c0004`

## Milestone 結論

**操作者完成一次實機 MPC 抓取，並確認夾取成功；執行紀錄確認從接近、MPC Grasp＋Squeeze，到動態 Lift、Hold、Release、Return 的完整流程全部成功。**

本次將原先已成功的 Ours MPC 抓取，銜接到以完成後實測姿態重新規劃的 **世界 +Z 15 公分 Lift**。實測關節經既有 FK 換算，末端實際上升 **14.994394 公分**，與目標相差約 **0.056 mm**。

這是 P3 實機整合的重要 milestone：MPC 能正常達成原始完成條件，並把完成後的實際姿態交給後續傳統軌跡執行，不再因舊預覽 squeeze 起點不符而卡在 Lift 前。

## 成功證據

兩種證據分別保留：

- **現場觀察**：操作者回報「夾取成功了」。
- **程式與量測紀錄**：sender 為 `status=completed`、`task_completed=true`；七個 receiver actions 全部為 `DONE`，沒有 abort reason。Lift、Hold、Release、Return 的 motion-completed flags 全部為 true。

`object_lifted` 與 `object_held` 在原始 JSON 中仍為 `null`，因為目前流程沒有自動追蹤物體位姿或判定物體保持狀態。這不等同失敗；物體成功夾取採用操作者觀察，手臂運動完成則由 action 與關節量測佐證。本紀錄不改寫原始結果欄位。

## 本次完整流程

| 階段 | 結果 | 關鍵紀錄 |
| --- | --- | --- |
| Above object | 成功 | 手臂到位 |
| Pregrasp | 成功 | 手臂到位，沿用既有手指到位政策 |
| MPC Grasp＋Squeeze | 成功 | `native_complete`，正常力總和 15.091320 N |
| Dynamic Lift | 成功 | 以新鮮實測姿態建立世界 +Z 0.15 m 目標 |
| Hold | 成功 | 設定保持時間 5.0 s |
| Release | 成功 | 執行既有零角度開手政策 |
| Return | 成功 | 返回原計畫的手臂起點 |

獨立的 Squeeze action 依既有 active 設計跳過，已包含於同一個 Ours episode。

Sender 從首個紀錄事件到最後結果約 **53.96 秒**；這不是單獨 MPC 計算時間。

## MPC 抓取數據

| 指標 | 結果 |
| --- | ---: |
| 有效 MPC proposals | 68 |
| `mpc_proposal` 紀錄總數 | 69，含最後的 completion 回報 |
| arm／hand 成對 command ticks | 658 |
| 首末 MPC command 時間跨度 | 6.795893 s |
| native completion index | 68 |
| reference 長度 | 44 |
| native max_steps | 88 |
| 完成時 native stage | 2 |
| 完成時正常力總和 | 15.091320191 N |
| 原始完成力門檻 | 15.0 N |
| 完成時採用的接觸點數 | 3 |

原始 reference 與正常力完成條件均已滿足。最後一筆 proposal 不帶動作目標，是 `native_complete` 結束訊號，不能算成一次 solver rejection 或待發布命令。

本次未出現先前的 joint-limit fault，也未出現 Lift 起點不匹配中止。正常力數值沿用目前系統的量測與轉換慣例；本次成功不代表另外完成了力感測標定。

## 動態 Lift 驗證

| 指標 | 結果 |
| --- | ---: |
| 要求的世界 Z 位移 | +0.150000 m |
| 實測關節 FK 的世界 Z 位移 | +0.149943941 m |
| 相對目標高度誤差 | −0.000056059 m，約 −0.056 mm |
| 實測關節 FK 的世界 X 位移 | +0.000000715 m |
| 實測關節 FK 的世界 Y 位移 | +0.000001994 m |
| 末端方向變化 | 0.003254° |
| Receiver Lift 起點最大手臂差值 | 0.000006525 rad |
| Receiver Lift 起點最大手指差值 | 0.006367552 rad |
| 保留的手臂起點容差 | 0.01 rad |
| Lift 規劃時長 | 4.819652 s |
| 已知幾何路徑抽樣 | 181 點通過 |
| 計畫抽樣的最小手／桌面間隙 | 0.027168261 m |

FK 位移比較使用 sender 動態 handoff 的實測起點與 receiver Lift action 的 `final_positions`，不是比較兩個 IK 目標。數值是關節回授與目前模型推算的末端位移，並非獨立外部量測的精度認證。

起點檢查維持原容差，本次沒有放寬、略過檢查或要求 MPC 回到舊預覽姿態。Lift 目標保持完成後實測的 22 個手指角度；Hold 與 Release 銜接新的 Lift 手臂位置。

Dynamic review ID：

```text
3e6d715f157a2f8808533a9195747e071d6140119745042f4da47ad36e4f5239
```

## 觀察到的非阻擋項目

- 既有設定為 `hand_arrival_required=false`、手指到位容差 0.06 rad。
- Pregrasp 的最大手指誤差約 0.108638 rad；Release 約 0.066216 rad。兩者記錄 `hand_within_tolerance=false`，依原有政策沒有阻擋流程。
- Lift 與 Hold 的最大手指誤差分別約 0.030116 rad、0.043428 rad，均在此次 0.06 rad 到位容差內。
- 路徑檢查覆蓋既有右側機器人／桌面模型的抽樣幾何；完整物體／未知環境碰撞與連續路徑認證仍未宣稱完成。

這些項目保留於 milestone 紀錄，不影響「本次操作者確認成功、完整 action 流程成功」的結論，也不把單次成功延伸成成功率或通用硬體認證。

## 原始紀錄與版本脈絡

- [整次執行目錄](../output/mpc_p3/active_20261113_132624/)
- [Sender 結果](../output/mpc_p3/active_20261113_132624/sender/run_report.json)
- [Sender 事件](../output/mpc_p3/active_20261113_132624/sender/events.jsonl)
- [動態 Lift 計畫與驗證](../output/mpc_p3/active_20261113_132624/sender/dynamic-lift-result.json)
- [本次 manifest](../output/mpc_p3/active_20261113_132624/run_manifest.yaml)
- [本次重新擷取與選定的 targets](../output/grasp_pipeline/p3_plan_20261113_132559/grouplive/export/targets.json)
- [MPC Grasp action](../output/mpc_p3/active_20261113_132624/receiver/20261113T212651_354799Z_491041/result.json)
- [Lift action](../output/mpc_p3/active_20261113_132624/receiver/20261113T212701_062948Z_491041/result.json)
- [Hold action](../output/mpc_p3/active_20261113_132624/receiver/20261113T212708_019864Z_491041/result.json)
- [Release action](../output/mpc_p3/active_20261113_132624/receiver/20261113T212713_393613Z_491041/result.json)
- [Return action](../output/mpc_p3/active_20261113_132624/receiver/20261113T212720_136679Z_491041/result.json)

本次使用工作目錄中的線性 MPC streamer 與 dynamic Lift handoff 實作；文件整理時這些變更尚未全部提交，不能僅以當前 Git HEAD 代表實機執行版本。各 action 目錄保留的 model、config、versions 與 run manifest 應與工作目錄變更一起保存。

進展脈絡：

1. [首次實機 MPC 試驗](mpc-p3-first-hardware-trial-2026-11-12.md)：進入 MPC，遭 streaming joint-limit fault 中止。
2. [線性傳送層重構](mpc-guardedstream-linear-results-2026-11-13.md)：完成新的 10 Hz → 100 Hz 線性傳送接線。
3. [Native completion 後 Lift 銜接失敗](mpc-p3-native-complete-lift-handoff-2026-11-13.md)：MPC 完成，但舊預覽 Lift 起點阻擋後續動作。
4. [Dynamic Lift 實作與離線驗證](mpc-p3-dynamic-lift-results-2026-11-13.md)：從實測完成姿態規劃 15 公分 Lift。
5. **本次 milestone**：實機完成抓取 → 15 公分抬升 → 保持 → 放開 → 返回，操作者確認夾取成功。

本文件僅整理已完成的實機紀錄；撰寫期間沒有修改控制程式、操作硬體或執行新的抓取。
