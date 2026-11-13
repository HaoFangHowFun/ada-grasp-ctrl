# P3：操作到實機前的交接

> 2026-11-13 更新：MPC active 傳送層已改為線性插值；離線／mock 通過，但 CRX 下游位置限制與斷流停止保障尚有 commissioning blocker。實機使用前請先閱讀 [重構結果與保護稽核](mpc-guardedstream-linear-results-2026-11-13.md)。下列舊操作指令不代表新版本已完成實機資格驗證。

軟體已接線並完成離線／mock 驗證；本次沒有啟動、連接或命令實機。實機 read-only 與 active 試驗由操作者執行。
原本 `script/run_grasp_pipeline.py` 的預設成功流程保持不變。

> Dynamic Lift 更新：實機 `mpc_active` 完成後以新鮮實測起點建立世界 +Z **15 公分** Lift，保留實測手形與原起點容差。實作、測試與直接操作指令見 [動態 Lift 結果](mpc-p3-dynamic-lift-results-2026-11-13.md)。

## 模式與入口

新入口：`/usr/bin/python3 -m ada_grasp_ctrl.mpc.live`，須 source 原本 ROS 環境。
省略 `--execute` 時只產生 `readiness.json`，不建立機器人 command publisher。

| 模式 | 指令來源 |
| --- | --- |
| `linear`（預設） | 原本 receiver 的完整流程，保留既有 quintic smoothing |
| `mpc_passive` | 同一個 baseline 指令流程；背景 Ours 計算只寫診斷 |
| `mpc_active` | 接近／pregrasp 沿用 baseline；Grasp＋Squeeze 共用一個 Ours episode；native completion 才能進入後續 lift |

這裡的 `linear` 是原本成功的非 MPC 流程名稱；不會切換原本 receiver 的插值方法。
P2 的 `--controller-mode` mock-only pipeline 入口仍維持原限制；實機 P3 請用上述新入口。

## 操作者執行順序

1. 依既有方式啟動相機、右側 CRX／Sharpa drivers。P3 不會啟動 driver，也不會自動恢復停用的 controller。
2. 產生**當次**選定候選的 packet；不要直接執行歷史實驗的 packet。

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
export ROS_DOMAIN_ID=87  # Use the domain of your existing real driver stack.
P3_PLAN="output/grasp_pipeline/p3_plan_$(date +%Y%m%d_%H%M%S)"
python3 script/run_grasp_pipeline.py \
  --environment hardware --ros-domain-id "$ROS_DOMAIN_ID" \
  --export-only --output-root "$P3_PLAN"
export P3_TARGETS="$PWD/$P3_PLAN/grouplive/export/targets.json"
```

查看這次輸出的 `grouplive/images/`、selection 與 motion plan。
`--export-only` 不啟動 viewer 或送出動作。若使用原本 8089 preview，先在 replay 終端 Ctrl+C 關閉舊 replay server；8089 只留一個 server。P3 執行入口本身不占用 8089。

3. 同一終端 source ROS，先做 read-only readiness：

```bash
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
/usr/bin/python3 -m ada_grasp_ctrl.mpc.live \
  --targets "$P3_TARGETS" \
  --output-root "output/mpc_p3/readiness_$(date +%Y%m%d_%H%M%S)"
```

檢視 `readiness.json` 的 status、block_reasons、28 軸、資料年齡、command ownership 與 proposal。
這是當下姿態的 measured-hold 計算探針；不是已驗證實體抓取。真正 active 接管時，在 baseline pregrasp 後會重新取得新資料並求解。

4. 先試 passive（**這會執行原本實機 baseline 動作**）：

```bash
/usr/bin/python3 -m ada_grasp_ctrl.mpc.live \
  --targets "$P3_TARGETS" --controller-mode mpc_passive --execute \
  --output-root "output/mpc_p3/passive_$(date +%Y%m%d_%H%M%S)"
```

終端要求輸入 `mpc_passive` 才開始。診斷在 `receiver/mpc_passive/passive.jsonl`。
每次實體執行後，重新確認場景與起點並重新產生當次 packet。

5. 由你在場執行 active：

```bash
/usr/bin/python3 -m ada_grasp_ctrl.mpc.live \
  --targets "$P3_TARGETS" --controller-mode mpc_active \
  --execute --hardware-enable \
  --output-root "output/mpc_p3/active_$(date +%Y%m%d_%H%M%S)"
```

終端要求輸入 `mpc_active`。它會先執行原本 above_object／pregrasp，再由 Ours 控制 Grasp＋Squeeze。
原始條件仍是走完 native reference 且正常力總和 **>15 N**；budget 仍是 reference 長度的兩倍。
成功後才跳過獨立 squeeze action，依新鮮實測姿態重建世界 +Z 15 公分 Lift，並同步銜接 hold／release／return。Sender 與 receiver 驗證同一份動態計畫，保留原本起點容差；若規劃後的實測姿態仍超差或驗證失敗，停止並記錄差值。

Ctrl+C 取消目前 action；timeout、solver rejection、過期資料或 ownership 問題都停止，不會自動 lift 或重新開始。
沿用既有 Runtime abort：停用 arm controller，手部保留既有 backend timeout policy；這不是新的急停機制。
中止後按原本操作流程復原 controller。之後直接使用原 pipeline，或新入口的 `--controller-mode linear --execute`，即可使用預設 baseline。

## P3 接觸政策與已知限制

- 新鮮且 F6／transport 健康的 MISSING／EMPTY 是 NO_CONTACT；保留原始逐指 status 與來源。
- INVALID、過期、斷線、來源異常是 unavailable；沒有五指必須全部接觸的條件。
- 一個 F6 對應多接觸點時標為 AMBIGUOUS，略過該指；保留其他有效接觸。全部接觸都歧義時不偽裝成 valid-empty。
- `f_object = -f_Sharpa`，normal 不翻轉；座標轉換沿用 adapter／FK／geometry。
- P3 將**量測力**的摩擦錐超限記為診斷；負正常力仍拒絕。原始 solver 的力決策變數仍使用原本 `mu=0.3` 約束。舊 compatibility 預設 strict 保留。
- 檢查原始 policy 後確認量測正常力用於力目標／stiffness 計算，solver 再約束最佳化力。因此量測超錐不等於資料格式失效；原始 solver failure 與輸出限制仍可停止執行。
- 使用同機 receipt-time freshness（0.2 s）及 skew（0.15 s）；不是 sensor acquisition synchronization。
- geometry／force calibration 仍是 provisional，`hardware_eligible:false` 是未完成標定的來源註記，不代表離線資料已被認證。實機操作仍需明確 enable 與當次人工確認。
- reference 來自 selected packet；frozen report 只提供同一機器人的模型／控制設定，不替換成 g0000 grasp。

證據與測試見 [P3 結果](mpc-p3-results-2026-11-12.md)。
