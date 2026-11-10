# 右 CRX＋右 Sharpa：評估前擺位預覽

本入口完成 Task 01 的第一個里程碑：使用 frozen suite 的真實物體、右手模型和抓取目標，
建立右 CRX 的共用場景與 IK preparation，提供靜態預覽。**不執行物理步進、控制器或 lift**。
使用者確認擺位後，已新增獨立右側 op／ours runner 並執行四尺度 smoke；
結果與重跑方式見 [右側 smoke 測試](crx-sharpa-right-smoke.md)。完整 2,000 個 pairs 尚未執行。

## 執行與重開

```bash
# 新輸出目錄；準備四尺度、寫入報告及十二張截圖，接著開啟預覽
bash script/preview_crx_sharpa_right.sh --port 8086

# 只準備與截圖，不啟動 server
bash script/preview_crx_sharpa_right.sh --prepare-only

# 重開已產生的同一份預覽；不重新選樣或求解
bash script/preview_crx_sharpa_right.sh \
  --run-root output/regression/crx-sharpa-right-preview-x080-y070-config-v1 --port 8086

# 另外納入指定樣本，仍保留四尺度固定案例
bash script/preview_crx_sharpa_right.sh --sample-id g0100
```

可從任意工作目錄用 shell 絕對路徑執行；相對路徑以 repository 為基準。
預設 Python 為 `.conda/bin/python`，可由 `PYTHON_BIN` 覆寫。
可用 `--suite-root` 指定同一契約的完整 frozen suite，以 `--output-root` 指定新目錄；
存在的目錄會被拒絕。`--run-root` 僅用來重開，不可與另兩個 root 參數混用。
錯誤的來源、hash、資產或環境會退出 2；正常準備退出 0。IK invalid 會記錄於預覽，
不會因為要展示而丟棄樣本。EGL 截圖使用已安裝的 MuJoCo renderer。

預覽網址為 `http://localhost:8086`，僅綁定 loopback；遠端使用可轉送此 port。
預設顯示 `Initial (config)`，使用 `bimanual_manipulation/config/right_arm_hand_task.yaml` 的
`offline_seed`：J1～J6 = `[-90, -30, 210, 0, -60, 0]` 度，右手採同檔案具名 `open_hand`。
程式直接讀取弧度，並與 `dual_crx_control/config/initial_pose.yaml` 的 URDF 度數交叉核對；
不做 FANUC 示教器 J3 角度換算。此姿態也是第一個 IK seed，後續階段使用連續 seed。
`initial_qpos`、初始 FK／碰撞、config 來源與 hash 都保存於本輪輸出，另提供 `initial_overview.png`。

介面可切換 `g0000`／`g0010`／`g0020`／`g0030`（四尺度）、pregrasp／grasp／squeeze、
單一或三組 target ghost、碰撞幾何、座標軸，以及全景／手物近景。
初始模式的實色機器人顯示 config 姿態；其他模式顯示求解所得 FK 姿態。
半透明手是原始 registered target，不能將其視為已達成姿態。
IK invalid 時仍顯示最佳候選與 residual，沒有將它當作可執行姿態。

## 座標與模型契約

- 28 維資料順序固定為 `right_J1`～`right_J6`，再接 canonical metadata 的 22 軸右手；
  MuJoCo qpos、DOF、actuator 及 Pinocchio 都以名稱建立映射。每筆 prepared NPY 保留 joint_names。
- 位置為 m，關節與角誤差為 rad，pose quaternion 為 WXYZ；URDF RPY 是 extrinsic xyz。
- DexLearn `robot_multidex.py` 從 global pose 取 hand targets，並將物體點雲放到 world；
  `sample.py::_decenter_robot_pose` 在輸出前恢復 pc_centroid。
  frozen adapter 保留這些 source-world flange poses，已做 pregrasp 外推 ×2；本入口不重做外推。
- 使用 `eval_T_hand = eval_T_source @ source_T_hand`，物體與桌面也乘同一 registration。
  不再乘 object pose；三階段均檢查 registration 前後 `inverse(world_T_object) @ world_T_hand`。
- 依使用者最新要求，將代表樣本 g0000 的物體中心放到世界 X=0.8、Y=-0.7 m，Z 保持來源值。
  對全部樣本套用相同平移 `[0.8439450985964946, -0.7016988483249088, 0]` m。
  其他樣本保留原場景間的位置差異，並非各自重新置中。沒有使用控制結果挑選 registration。
- mesh scale 僅作用於物體 mesh vertices；不縮放 world translation，不額外移動 mesh 原點。
  原始桌面必須是 source-world z=0 的 plane。平面物理上無限，預覽顯示有限大小供辨識。
- 右臂由真正的 FANUC macro 加上 reference 右 base origin 展開，右手來自 frozen suite；
  不是將左手改名。右 ee_mount→hand flange 採 reference 的 placeholder mount，尚非實機校正。
  來源、hash、revision 與限制見 `assets/robot/crx5ia_sharpa_wave_right/SOURCE.md`。
- 保留原右手幾何、軸向、limits、damping、kp=7.5，以及既有 object density 計算方式與摩擦。
  CRX position gain kp=100、kv=10 仍為待控制驗證的暫定值。
  桌面接觸在新場景中顯式保留；與原 dummy runner 的排除行為有差異，後續 protocol 必須記錄。

## 數值檢查與輸出

全套 frozen suite 為 25 objects、100 scenes、1,000 grasps，四尺度各 250 筆。
manifest SHA-256：`f167bb725747b9366cbc3f06504a54705338d3e5188303092a906b23f7ec8ec8`。
每次準備和重開都核對整套 input/dependency hashes；不重新推論或補抽樣。

`preview_report.json` 保存 source/evaluation poses、固定變換、尺度、包圍盒、物體底面間隙、
三目標 IK、每次 seed 嘗試、獨立 MuJoCo/Pinocchio FK/Jacobian、零姿勢與三目標的靜態接觸。
同目錄有 `run_manifest.yaml`、模型、每筆 `prepared.npy`／`scene.xml`／`overview.png`／`closeup.png`。
report、模型、來源及共用準備程式均保存 hash；影響坐標或擺位的改動須產生新預覽重新確認。

容差：IK 位置 1 mm／角度 0.01 rad；獨立模型 FK/Jacobian 最大誤差 1e-6；相對位姿不變性 1e-10。
IK 使用連續 seed 與固定 fallback seeds，並優先採用無手臂穿桌的可達解；
沒有無碰撞解時保留 failure diagnostics，不改變抓取目標。IK 可達與初始化碰撞分開記錄。
抓取／squeeze 的靜態重疊不是動態接觸結果，不能據此宣稱控制成功。

目前代表樣本中心 X=0.8、Y=-0.7 m，四筆 config 初始姿態皆無手臂穿桌。
在固定 seeds 與求解設定下，g0000／g0030 的三目標通過 IK 且無手臂穿桌；
g0010 通過 IK，但三目標均有手臂穿桌；g0020 未通過 IK，位置誤差約 29–31 mm。
這不是完整工作空間證明。四筆物體底面保留來源約 0.15–0.33 mm 的桌面微小穿透，未自動抬高。
相對抓取位姿不變性檢查通過；後續 home→pregrasp 路徑與動態測試結果見上述 smoke 文件。

單元測試涵蓋 config 的角度單位與具名手部關節、剛體搬移不變性、尺度只套用一次、
錯誤 pose/frame、名稱映射、零 physics steps 與 hash 漂移。
全專案 Ruff check、compileall、新 shell 語法及新增檔案 format check 通過；
全專案 format check 有八個既有檔案不符格式，未修改它們。

獨立的原 Shadow quick regression 已產生 100 筆結果：success 68、failure 4、
invalid initialization 21、solver degraded 7、execution error 0；退化保留為非零退出狀態。
第一次嘗試受到繼承的 ROS Python 3.12 路徑影響而失敗；清除該路徑後在新目錄重新執行。
這項檢查不計入右 CRX 評估次數，也不代表已通過完整三手型 release gate。
