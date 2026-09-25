# NG-0076：歷史重放失敗與無 Observer 對照

2026-09-13 UTC。[原計畫](ng0076-endpoint-union-trajectory-plan.zh.md) 的 `historical-trajectory-v1` 在第80步失敗，完整失敗現場保留。這不是新的 loss 訓練，也不是放寬失敗 attempt 的 gate。

## 已知事實與未知原因

原始80步／320次曝光的 query ID、input token 計數及 objective metadata 一致；loss、scores、score gradients 和 update norms 仍在原 `rtol=2e-5, atol=2e-7` 內。然而第80步第0個 example 的 document NNZ 為17,779，歷史值17,780，故 exact readout-count gate 正確地阻止後續執行。

| 最早可見差異 | Step | 最大絕對差異，前80步 |
| --- | ---: | ---: |
| 實際參數位移 L2 norm | 69 | 2.89519013e-10 |
| Scores | 74 | 6.67572021e-6 |
| Score gradient | 74 | 6.89178705e-8 |
| Loss | 74 | 3.42726707e-7 |
| Gradient norm before clipping | 74 | 2.02655792e-6 |
| Document NNZ | 80 | 1 |

step0、16、32、48、64的 observer 前後 model／moments／gradients／模式／RNG 指紋一致；不能由此推導 CUDA algorithm、allocator 或 workspace 狀態也完全不變。小模型 CPU 測試同樣不能代替 GPU 歷史軌跡重現。update norm 最早差異不是第一個參數元素差異的證明：歷史沒有保存每一步完整參數。

實際環境為 Torch `2.9.1+cu128`、CUDA12.8、Transformers5.15.1；預設 deterministic algorithms 為false，未設定 `CUBLAS_WORKSPACE_CONFIG`。這只是數值重現待查因素，**尚未證明 GPU nondeterminism 或 observer 是原因**。ReLU 支持集合在零附近可不連續，但沒有差異前後的該 activation，不能把這次NNZ差異斷言為近零量化問題。也沒有證據顯示資料損壞、OOM或新模型品質崩潰。

CPU審核工具 [audit_ng76_replay.py](../../../../scripts/audit_ng76_replay.py) 獨立比較原始JSON trace，區分 identity、readout、numeric tolerance 和bit-exact差異，保留全部差異位置；不執行encoder、不重新計算loss。沒有改動原 frozen source 或原observer gate。

## 唯一下一個對照

新 unit `NG-0076/no-observer-control-v1`，由 [ng76_replay_control.py](../../../../scripts/ng76_replay_control.py) 凍結獨立manifest。使用失敗attempt已驗證的同一批source與所有539個parent dependencies，再綁定完整失敗inventory。保持NG71 D原base、384個TRAIN、A0 witness、原順序、FP32、RMS、optimizer、學習率、token長度、seed和backend設定，僅明確 `observer=None`。

只執行原始前96個optimizer updates、保存並reload實際checkpoint和moments。這是96次可丟棄的診斷更新，不是0次optimizer操作，不啟動D192、不評DEV或LOCKED_TEST。測量整段trace與原D96完整model／optimizer fingerprint；不在每步增加GPU hashing或額外sentinel inference，避免把觀察副作用重新加回對照。

本對照的「程序完成」與「歷史重放合格」分離。即使96步完成並成功保存，任一原identity/readout/tolerance或endpoint gate不符，都記為 `historical_replay_qualified=false`，不得把它用作原NG71歷史中段的證據。保留不符值正是對照目的，不能把ClearML程序成功或exit0當成歷史一致性通過。

單GPU，啟動前檢查空閒且不移除別人的工作。CPU4、RSS16GiB、GPU20GiB、host available>24GiB、disk free>40GiB、完整phase1800秒，包含input驗證、載入、保存、reload、tracking close和checksum。依原354.84秒D96及這次376.08秒失敗程序，估計完整對照含傳輸審核10–20分鐘，5分鐘輪詢後按實際剩餘時間調整。禁止失敗自動重試或調大時限。

## 決策

1. 若無observer仍無法匹配歷史，至少排除「必須存在週期observer才會出現差異」；尚不能單憑一次否定observer的額外影響，也不能斷定是哪個kernel。先解決計算重現契約，不新增loss。
2. 若無observer完全匹配、原有observer不匹配，observer相關執行路徑是更強嫌疑，但仍需有界配對重現。先定位backend／記憶體工作區或採獨立快照觀察，不再擴大資料規模。
3. 若需改deterministic backend，另立新計算契約及匹配對照；不能追認新軌跡就是原NG71。不因只有1個NNZ差異就刪掉gate，更不因此宣稱已超越dense。

完整混合品質、全部正例、NQ最低線、未曝光holdout與native total cost目標不變。本輪優先排除量測系統自身的歧義，才有資格用時間定位決定retention或候選覆蓋訓練。
