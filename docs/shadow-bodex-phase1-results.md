# BODex → Ada Shadow Phase 1

日期：2026-11-01。依使用者要求，改用 BODex 生成姿態，只執行 Phase 1，預覽服務切換至 [localhost:8081](http://localhost:8081)。

## 結果

同一批十個尺度修正後的 scenes，每個由 BODex 新生成五個抓取，共五十個；保留全部候選，未挑選成功案例。

| 結果 | BODex＋Phase 1 | 同場景 DexLearn＋Phase 1 |
| --- | ---: | ---: |
| success | 24 | 0 |
| failure | 25 | 3 |
| invalid_initialization | 1 | 47 |
| solver_degraded／execution_error | 0 | 0 |
| success / selected | 48% | 0% |
| success / (success + failure) | 24/49 = 49.0% | 0/3 = 0% |

BODex 在八個 scenes 至少成功一筆。可先觀看 `g0000` 的成功抓取、`g0003` 的失敗抓取與 `g0022` 的無效初始化。

**BODex 在相同 Ada 場景與 Phase 1 設定下明顯減少初始化問題。** 這表示 Ada 並非普遍無法執行這些場景的抓取，後續應優先調查 DexLearn 輸出或其資料語意。仍不能據此斷言兩套手部模型完全一致，或已隔離模型哪一部分有問題。

這是相同場景與 evaluator 的對照，不是相同資訊／算力預算的生成器比較：BODex 使用完整 mesh 與最佳化；DexLearn 使用點雲與學習模型。兩邊的 candidate index 也不代表相同姿態。

## 生成與執行設定

- 使用 `HUGS-BODex` 的 `sim_shadow.yml`／`3_right_full`／`surface_sample`，seed 123。
- 明確使用上一批 `scale-matched-v1` 的十個 scene 路徑，沒有修改物體尺度、位置或資產。每 scene `seed_num=5`；dry plan 與實際輸出均為 10 scenes、50 candidates，沒有 height／scale／success filtering。
- 使用 Ada 既有 `BODex` converter：以 raw joint names 驗證 Shadow 順序，轉換 palm 到 dummy-arm base 的 34 mm offset，再做一次 CPU IK。
- Phase 1 使用既有 `op`；Ada seed 12，`arm_pregrasp_is_grasp=false`、`pregrasp_t=0`、`squeeze_t=0`、offset=0。未執行 Phase 2。
- **Ada 沒有額外 squeeze 加深。** BODex 自身的 native 三階段輸出是：最佳化 pregrasp／grasp，再以 grasp palm 和 `2*grasp_fingers - pregrasp_fingers` 構成 squeeze。此次直接保留該原生輸出，不再外推一次；這和 DexLearn 預測三個姿態的生成語意不同。

150 個姿態 FK 已核對回 raw BODex palm：最大位置誤差 0.619 mm 以下、旋轉誤差 0.00799 rad 以下，named finger joint 誤差為 0。三階段 stored／effective targets 完全一致。

## 8081 預覽

- **Sample**：切換全部 50 筆。
- **View → Target poses**：觀看原始 pregrasp／grasp／squeeze 目標。
- **Play Phase 1**：播放已執行的實際狀態，包含 approach、squeeze 與 lift。
- **Replay progress (%)**：拖曳查看執行過程或末態；重播後可拉回 0 再播放。
- invalid 初始化案例只有 init／terminal，沒有虛構控制或 lift 動作。

此頁顯示 BODex 資料與已保存的 Phase 1 replay，不會在瀏覽時重新最佳化。之前 8081 的 DexLearn 資料仍完整保留於磁碟；8080 的原批預覽未更動。

## 證據

- [來源與場景清單](../output/shadow-phases/bodex-phase1-v1/source_manifest.json)
- [BODex 命令與退出碼](../output/shadow-phases/bodex-phase1-v1/commands.json)
- [固定候選清單](../output/shadow-phases/bodex-phase1-v1/selection.json)
- [準備後 suite](../output/shadow-phases/bodex-phase1-v1/suite_manifest.json)
- [Phase 1 每筆結果](../output/shadow-phases/bodex-phase1-v1/run/phase1/phase_results.json)
- [FK 與完整性驗證](../output/shadow-phases/bodex-phase1-v1/run/validation.json)
- [生成／執行腳本](../../reviews/ada_shadow_phases_20261101/run_bodex_phase1.py)
- [預覽腳本](../../reviews/ada_shadow_phases_20261101/preview_phase1.py)

共 50 筆準備結果、50 筆 eval／stat 結果及 50 份完整 sidecar，沒有 execution error。comparison 中 Phase 2 的 missing_result 表示此次刻意未執行，不是 Phase 2 測試失敗。此次沿用控制程式，未改 solver、物理參數或 golden baseline。
