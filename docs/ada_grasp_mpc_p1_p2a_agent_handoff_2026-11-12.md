# Ada-Grasp-CTRL — MPC Phase 2 P1 + P2a Implementation Handoff

**日期：** 2026-11-12  
**交接對象：** 新 Codex / coding agent  
**專案：** `/home/hao-fang/hugs-workspace/ada-grasp-ctrl` (`HaoFangHowFun/ada-grasp-ctrl`)  
**任務範圍：** **P1：Ours MPC ↔ ROS stage execution contract**；**P2a：已錄製的真實 MCAP → Ours MPC offline replay**。  
**執行限制：** 本次只允許離線讀檔、MuJoCo 模型作為 *kinematics* 依賴（若現有程式需要）、ROS mock / isolated DDS；**不可啟動、連接或命令真實 CRX / Sharpa**。不需要重做先前已成功的 MuJoCo grasp demonstration。

---

## 0. 執行目標與不可妥協的規則

這次**不是重新開發 MPC**，也不是重新驗證 MuJoCo 能不能抓起物體。A/B/C 的離線基礎建設已存在；我們要讓新程式正確繼承**原始 Ours Controller 對 stage、force trajectory、termination、timeout 與 solver-failure 的權限**，並讓真實錄製的 tactile/joint 資料能**離線**產生（或明確拒絕）10 Hz 的 MPC control proposals。

### 絕對保留原有成功的 Non-MPC Pipeline

- `script/run_grasp_pipeline.py` 的 **non-MPC baseline** 必須維持現有成功的 camera → DexLearn → selector → pregrasp → grasp → squeeze → lift → hold → release → return 路徑；預設仍為 `linear` / 原本的安全 baseline 行為。
- `mpc_passive` 是**旁路觀察**：既有 Linear Controller 是**唯一**實機命令發布者；MPC 只能讀 joint/tactile、計算和寫日誌，**絕不可發布 robot commands、改寫 Linear target、延遲/阻塞 Linear stage、或以 MPC 的 failure/timeout 導致 baseline 中斷**。
- `mpc_active` 目前僅允許**離線/mock**；不得為通過測試而啟用真機開關。
- 相同 replay input 和測試條件下，`linear` 與 `mpc_passive` 必須具有**位元相同的 baseline command trace**；如真機資料比較只可比較 commands，不能宣稱兩次實際物理 trajectories 位元相同。
- 不修改原始 Ours MPC 數學、optimal control objective、force threshold、stage-switch 及 termination semantics；不修改 `dual_crx_control`、`dual_sharpa_wave_ros2`、`bimanual_manipulation` 驅動、安全限制、500 Hz interpolation、Sharpa SDK、或既有 tactile adapter。**Ada reference** 使用 linear；**CRX lower-level quintic** 保留。
- 先檢查 HEAD、dirty/untracked、各檔初始 hash；保留其他同事/任務的所有未提交變更；**不要擅自 commit/push**。

### MPC authority contract

| 決策 | Owner |
|---|---|
| Stage 1 / Stage 2 switching、desired force trajectory、optimization、solver-failure policy | **原始 Ours** |
| Grasp/control episode normal completion | **原始 Ours `should_stop()`** |
| 原始控制 episode 最大 steps / timeout | **原始 Ours `max_steps()`** |
| 10 Hz proposal → 100 Hz guarded streaming、feedback validity、watchdog、joint limits、velocity/position safety、publisher exclusivity | **ROS Executor** |
| 非 MPC pregrasp、lift、hold、release、return 與所有 `linear` / `mpc_passive` 真實動作 | **既有 Linear Pipeline** |

注意：**Ours 正常完成權限不等於覆蓋 safety stop**。ROS Executor 仍可因 safety/stale feedback/transport fault 安全中止。`mpc_passive` 只能記錄這些 MPC-side fault，不得影響 Linear 執行。

---

## 1. 來源、現況與已經完成的驗證

先閱讀下列文件，不要依過去口頭摘要臆測當前 code：

1. `docs/mpc-phase2-validation-2026-11-12.md`（本次需求依據；若實際路徑不同先搜尋 repo）
2. `docs/tactile-compact-recording.md`
3. `docs/right_full_mpc_integration.md`
4. `docs/grasp-pipeline.md`、`docs/grasp-pipeline-validation.md`
5. `src/ada_grasp_ctrl/tasks/control_eval_func/optimized_runner.py`
6. `src/ada_grasp_ctrl/experiments/right_full_control.py`
7. 新的 Phase 2 A/B/C 模組、runtime、`ContactObservation` compatibility、replay、stage runner、ROS mocks：從 repo 搜尋與測試來定位，**不要假設檔案名稱**。

### 已知現況（2026-11-12，均需與當前 workspace 比對）

- 當時 HEAD：`7907b4925cc4c84d76e2898c5f39016f66716eff`，Agent 未 commit/push；原有 dirty/untracked 必須保留。
- A/B/C 已實作與離線驗證：explicit modes、Linear reference、Passive/Active stage boundary、native Ours wrapper、10 Hz proposals / 100 Hz guarded ROS mock streaming、exclusive publisher、tactile point-contact compatibility、synthetic replay、Jacobian/Ks/optimizer equivalence。
- Baseline + passive known cycle：各自完整執行 7 階段，各 3,537 mock baseline commands，**byte-identical**。MuJoCo contact fixture passive 有 51 個有效 proposals、**零** MPC commands / ROS publishers；合成 tactile fixture 可正確拒絕不新鮮/不一致的 observations。
- ROS Mock timing：MPC request period p50 ~100.377 ms；guarded publish p50 ~10.315 ms；30 proposals、289 paired command publishes；沒有左側指令；mock quintic retiming 已檢查。
- **Active known-cycle fixture 未通過**：pregrasp 完成，grasp 期間有 88 accepted proposals、880 guarded mock commands，最終 native stop/timeout 與固定 joint arrival gate 不一致；最後 `right_J1≈−0.011121 rad`、`right_J4≈+0.012267 rad`，現有 arrival tolerance 0.01 rad；**contact count = 0**；未執行 squeeze/lift。這不表示 MPC 物理抓取失敗，也不能人為宣稱此無接觸 case 應成功。
- 既有全套 unit tests：修改後 444 total、387 passed、57 skipped、0 failed；24 個 focused MPC tests passed；原 `right_full_control.py` tests 20 passed。原 release gate 仍非全綠，Shadow/LEAP golden deviations 在 reconstructed original workspace 可重現，**不要誤歸因於本次工作**。
- 真實 baseline **已執行且完整錄製**，不是只存在 synthetic bag：`/home/hao-fang/ws_fanuc/output/physical_mpc_observe_20261112_194701/`；約 92.385 s、75,626 ROS messages、~1.1 GB、九個 topics 均有資料。原 baseline report 在 `output/grasp_pipeline/physical_baseline_20261112_194720/run_report.json`；初步 bag decode report：`output/physical_bag_inspection_20261112_194701/report.json`。**這是 full bag，包含 `dist_force`，不是後來加的 compact recorder bag。**
- 真實 bag 有 5 指 `TactileFrame` 12,538 筆；contact `VALID`/`MISSING` 混合，沒有 `EMPTY`；少數 multi-point frames，不能把整指 resultant force 隨意分配到多接觸點；錄製終端曾報 53 transport-lost messages；bag callback receipt time ≠ sensor acquisition time。
- `record_compact` 已新增並通過 mock/round-trip 測試，但 **bag → complete synchronized MPC replay 尚未實作**；請勿再擴充 recording/visualization 作為此任務主要工作。

---

## 2. P1：繼承原始 Ours Stage / Termination Contract（Offline）

### 2.1 先 source audit、不要自行創造完成條件

原始實作在 `src/ada_grasp_ctrl/tasks/control_eval_func/optimized_runner.py`：

```python
def max_steps(self, runner):
    return len(runner.qpos_path) * 2

def should_stop(self, runner, step):
    return (
        step.index >= len(runner.qpos_path)
        and step.current_sum_force > self.final_sum_force
    )
```

`src/ada_grasp_ctrl/experiments/right_full_control.py` 的 `Controller` 已 expose `should_stop()` 與 `max_steps()`，並由 `_CoordinatedWrenchPolicy` 負責 stage 1/2 switch、desired normal force、optimization、solver policy。以**目前 checkout 的真實程式碼為準**，確認上述邏輯仍相同。

**設計要求：**

1. 在 MPC active **mock** 執行中，normal completion 必須來自 Ours `should_stop()`；native max-step timeout 來自 `max_steps()`。不得另外用 Linear fixed-target joint arrival 作為 **Ours 的 normal completion 判定**。
2. 不要把 grasp / squeeze 的 Linear pipeline stage 標籤誤當成 Ours stage 1/2；查明現有 stage sequencing 怎麼映射，避免兩層 state machine 重複擁有決策權。
3. 若 Ours `should_stop()` 尚未達標而 `max_steps()` 用盡，回報明確 **native termination not achieved / native timeout**，safe abort；**不能**因固定 joint arrival 接近、最末關節誤差很小就偽造成功。
4. 對 `contact_count=0` 或 valid-empty contact，`current_sum_force=0`，不得憑空宣稱 force-based completion；無 contact case 預期可以 timeout 並 safe-abort。
5. Ours stage termination 之外，安全監控（feedback freshness、finite values、limits、single publisher、stream/scheduler failure、explicit stop）仍在 executor，並保留 existing guard semantics。
6. 既有 fixed-target Linear arrival gate 在 **linear** 路徑維持原狀；MPC active 模式若移除其 completion authority，需保留必要的 command settling / safe transition checks，但不得悄悄新增另一種 grasp success metric。
7. Passive Mode 的 Ours `should_stop()`/timeout 只能寫 diagnostic；不會切換、提前結束、延遲或 abort 實際 Linear stages。

### 2.2 釐清既有失敗，不能只修飾 error message

讀取 `output/mpc-phase2-20261113T021739Z/` 下的 `active-mujoco-cycle-v2/`、`ros-mock/trace.json`、`run_report.json`、`proposals.jsonl`、`commands.jsonl`（如可取得）。逐 tick 回答：

- Ours policy 的 `stage`、`step.index`、`len(qpos_path)`、`current_sum_force`、`final_sum_force`、`should_stop()`、`max_steps()` 與 solver accepted/rejected 是什麼？
- 首個中止事件究竟是 **native max_steps**、**Linear arrival threshold**、**feedback/stream fault**，還是其他 gate？哪個 component 擁有該 gate？
- 最後 J1/J4 誤差究竟是終止 *原因*、附帶診斷，還是進一步執行前的安全門檻？不得從數值單獨推斷根因。
- 修正後重新跑相同 no-contact case：允許且預期以清楚的 **Ours native timeout** 安全結束；**不要要求 Grasp → Squeeze → Lift 自動成功**。

### 2.3 P1 必須新增/保留的測試

- 以 fake/native policy fixture 模擬：`should_stop=True` 且所有 safety checks 通過 → adapter 回報 Ours normal completion；**不依賴 Linear joint arrival**。
- `should_stop=False`、force 不足直至 `max_steps` → precise native timeout + no further stage progression + safe shutdown。
- Solver rejected、stale feedback、position/speed violation、command ownership conflict、shutdown 等仍 fail closed。
- `linear` 和 `mpc_passive` 在相同 replay packet 的 baseline commands **byte-identical**；MPC computation error/invalid contact 不影響 baseline。
- 10 Hz/100 Hz 時序與 28 個具名 absolute joint position outputs 不回歸；mock 不向任何左側 topic 發送命令。

---

## 3. P2a：使用真正已錄製的 MCAP 做 MPC Offline Replay

### 3.1 指定資料來源與操作規範

**優先使用既存、已成功 decode 的真實 full bag：**

```
/home/hao-fang/ws_fanuc/output/physical_mpc_observe_20261112_194701/
```

相關結果：

```
output/physical_bag_inspection_20261112_194701/report.json
output/grasp_pipeline/physical_baseline_20261112_194720/run_report.json
```

錄製來源的 9 個 topics：

```
/crx5ia/right/joint_states
/sharpa/right_hand/joint_states
/crx5ia/joint_targets
/sharpa/right_hand/joint_command
/sharpa/right_hand/tactile/thumb/frame
/sharpa/right_hand/tactile/index/frame
/sharpa/right_hand/tactile/middle/frame
/sharpa/right_hand/tactile/ring/frame
/sharpa/right_hand/tactile/pinky/frame
```

- **只讀 MCAP**；不 publish、不可 launch 真實 ROS drivers，也不可在 physical domain 87 發送訊息。建議 pure offline reader / deterministic event scheduler；若因工具要求 ROS mock，必須隔離 localhost domain 199 且不發 robot commands。
- 以錄製中的 reference/commands、joint feedback 和 source timestamps 建立時序，**不得把 bag callback timestamp 假裝成 tactile acquisition timestamp**；維持原始 clock domain、sequence、session id / frame id、接收時間、原始 status。
- CRX 6 + Sharpa 22 = 28 joints：使用現有 canonical names/order 映射，不可默認 bag order 與 MPC order 相同；拒絕重複/missing/非有限 joint positions。
- 限定 **Grasp + Squeeze** 的 MPC proposal；Pregrasp/Lift/Hold/Release/Return 只做 baseline replay 或 annotation。
- 若 recorded Linear path 與 frozen native Ours reference frame/scene 不一致，不能憑空宣稱其為一致；標記 reference/target provenance、無法匹配的情況，並依 contract 拒絕需要該資訊的 proposal。不要硬把現有 `g0000` 的場景當作實機錄製的真實物體。

### 3.2 Contact compatibility 要尊重現有 contract

使用既有 `TactileContactObservation`、MPC-compatible `ContactObservation` 和原始 Ours solver，不改動其數學或 SDK adapter：

- Tactile surface normal：**指腹朝外，Hand → Object**。
- Sharpa F6 前 3 分量的 provisional convention：**Object → Finger**（按壓時向手指內部）；保留 `provisional` quality，不宣稱已校正。
- Ours canonical：normal 與力均指向 **Hand → Object**；因此在 **MPC compatibility layer** 轉換 `f_object = -f_object_on_finger`，再 `f_local = R_contact.T @ f_object`；`R_contact` 第一軸是 normal，右手正交，正常壓縮要有 `f_local[0] >= 0`，friction cone orientation 與原始 solver 一致。**不要在原始 tactile adapter 反轉 normal 或 force**。
- 原始 Ours grasp matrix 以 **contact centroid** 為 wrench reference，力臂 **m → cm ×100**；保留 `G @ f` 和原本的 scale。局部接觸點 torque 沒被實際校正，不是 point-contact 模型的必然 blocker；**不得把未量測到的 torque 當成 measured zero**。若 legacy 6-vector schema 需要 point-contact modeled zero，必須用明確 model provenance，先 audit 所有 downstream consumers。
- 嚴格建立 `contact_pos_local`、`contact_frame_local`、關聯指腹 URDF link/body identity、FK 和 Jacobian/Ks，與原先 MuJoCo point-contact geometry 在合成等價測試中一致。
- 目前 F6 是**每指 resultant**，多個 contact points 不可將同一 resultant 任意分配到各點。對來源狀態區分 `VALID`、`EMPTY`（確實有效無接觸）、`MISSING`、`INVALID`、過期、時間不一致、multi-contact 不可歸因。**`MISSING` ≠ `EMPTY`**；除非 source semantics 被證實，不可將 widespread MISSING 當成 no contact。
- Joint/tactile 對齊依真實可用 clock 與 uncertainty 執行；不以 future samples 偷看，若資料 lag/freshness 不合格，回報 `unavailable` + reason，不能補上虛構 observation / force / frame / timestamp。

### 3.3 Replay 結果必須同時輸出 proposals 與拒絕原因

每個 10 Hz tick 記錄至少：

- 時間軸（bag receipt / source / replay / MPC tick 的具名 clock domain）、joint age、tactile age、clock skew、source/session/sequence IDs。
- 28 named measured joint positions、baseline commanded positions、MPC 28 named **absolute** proposed targets（若可用）。
- 所有 per-finger contact 的 status、contact count、frame、position、normal、force、force sign conversion / unit provenance、是否被 solver 接受。
- Native Ours `stage`、reference index、`current_sum_force`、`final_sum_force`、`should_stop`、`max_steps`/remaining steps、solver status、duration、guard diagnostics。
- proposal status：`accepted` / `rejected` / `unavailable`，精確 reason code。**不將 rejected/unavailable 偽裝成有效 MPC output**。
- aggregated summary：ticks、eligible observations、invalid categories、valid proposals、solver rejections、p50/p95/p99 timing、最常見 blocker、哪一段 source data 可用。

**重要：這是 Counterfactual Replay。** 真實硬體當時跟隨 Linear commands，不會因 MPC proposed output 改變 recorded feedback，因此不得宣稱 replay 是 MPC 物理 closed loop，更不能宣稱抓取成功率、接觸力改善或擾動恢復能力。

### 3.4 P2a acceptance

1. 用**真實 full bag** 跑完具時間同步的 offline parser/adapter/replay，產生 manifests、per-tick trace、summary/report；保存可重現 command，非只寫 tests 或文檔。
2. 嚴格的 no-hardware evidence：檢查 process、domain、topic owner；沒有 robot command publishers，沒有碰真實 driver。
3. 若實機資料在 Grasp/Squeeze 找不到適合的有效 single-contact snapshots，**正確結果是具體分類的 rejection/unavailable 統計**；不要合成 contact 讓測試看似通過。可另外用 synthetic fixture 驗證 solver 可接收正確 contact，但必須與 real-bag evidence 明確分開。
4. 測試完成下列邊界：contact VALID/EMPTY/MISSING/INVALID、multi-contact resultant、反向 force、zero-normal compression、stale、clock skew、out-of-order、SDK sequence gap、joint missing、model/geometry provenance 缺失、source timestamp 不可比。
5. 確認 baseline 和 passive 在 identical offline packet 下 commands byte-identical，並且 replay **完全不發布** MPC targets。
6. 如有 blocking external calibration / reference alignment，**報告真正 blocker 與可復現 tick/example**，不要修改 safety 或假設測量有效。

---

## 4. 執行順序（先 P1，再 P2a；不要分心擴展其他功能）

### Step 0 — Audit

- `git status --short`、`git rev-parse HEAD`、要編輯檔案的初始 SHA256，列出現存變更與已支援的 modes。
- 核對上述來源文件和目前 workspace，指出不同之處；不覆蓋他人 dirty changes。

### Step 1 — P1 minimum change + focused regression

- 重現 active mock known-cycle timeout；提供 exact component/gate evidence。
- 用 native `should_stop` / `max_steps` 做 normal completion/timeout **唯一 authority**，保留 safety error gates；不改 Ours 內部邏輯。
- 實測 native-complete、native-timeout、ROS-safety-stop；verify `linear`/`mpc_passive` trace parity。

### Step 2 — P2a minimal real-MCAP reader/replay

- 優先重用 A/B/C 已有 adapter、worker、runtime、tactile replay schema；只補缺的 full-MCAP synchronized input 與 time-mapping，不再寫新的 recorder。
- 先做 raw read + synchronized frame/joint timeline 與 status audit，再逐 100 ms 向 native Ours propose。
- 保存 read-only replay traces / reports；必要時產生 audit-only records，清楚標記 replay unavailable ticks。

### Step 3 — Full offline regression and handoff

- 重跑 420→444 類的現有 unit/regression（實際 numbers 以 current code 為準）；`right_full_control` parity、new focused tests、ROS mock ownership/timing/retiming、`ruff check`、`compileall`、`git diff --check`。
- 如果 release gate 非綠，區分既有 baseline mismatch 與本次 regression，不修改 golden/tolerances 掩蓋問題。
- 附 exact commands、environment、output paths 和可重跑 evidence；不 commit/push，除非另有明確要求。

---

## 5. 預期最終交付（必須是真實實作，不是只有設計文件）

1. **P1 code changes**：原始 Ours control/termination contract 被正確呼叫與回報；safe-fail 區分清楚；Linear baseline unchanged。
2. **P2a code changes**：真實 full MCAP 解析、joint/tactile synchronized replay、`TactileContactObservation → ContactObservation → Ours proposal`（符合資料可用條件才出 proposal）。
3. **One reproducible command**：從既有真實 MCAP 只讀跑整段 Grasp/Squeeze replay，輸出完整 per-tick logs 和 summary；CLI 實際 flags 由 agent 依現有程式碼決定並在報告中列出。
4. **Regression tests**：P1 native completion/timeout/safety、P2a provenance/time/status/contact math 和 passive baseline parity。
5. **Status report**：包括 implemented/validated/blocked、命令與環境、所有實際統計、仍需實機校正的問題、沒有硬體 I/O 的證據。

**不要宣稱 P2b Real Passive 已完成**。P2b 是下一階段：操作員在真機既有 Linear baseline 上開啟 read-only、10 Hz Ours 計算與 log，**MPC 絕不發 ROS command**。只有 P1/P2a 足夠的實機資料品質與失效隔離驗證完成後，才能另行安排 P2b，之後才考慮經安全審查的 Active 模式。

---

## 6. 必須報告的科學限制與風險

- Ours 原本完成條件依 contact force；**沒有有效 contact 的 mock/replay 不能自然達到 force-based completion**，這是物理/資料限制，不是自動需要放寬 threshold 的軟體 bug。
- 原始 10 Hz solver 在 valid-empty contact 下的 timing 不代表真實多接觸、stale/ambiguous data 也可穩定 10 Hz，請分開報告。
- 真實 bag 中 `MISSING` contact 很多、偶有 multi-point；檢查其根因與可用率，不得用 synthetic force 補齊 real-bag evidence。
- 原 bag 記錄了 53 transport-lost 的報告，不能將零 observed gaps 等同於無遺失，也不能把 SDK forward gaps 與 DDS losses 混為一談。
- 真實錄製中所有 baseline motion phases 完成，但 `object_lifted`/`object_held` 未獨立驗證；不要把 motion completion 當抓取物理成功。

**核心驗收口號：** `linear` 永遠可獨立執行；`mpc_passive` 零 robot publishes、零 baseline command changes；`mpc_active` 只在 mock 測試且遵從 Ours 自有 termination；真實 MCAP replay 有證據才提 proposal，否則精確拒絕，絕不偽造 contact。
