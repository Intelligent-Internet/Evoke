# NG-0086 執行紀錄

更新：2026-09-18 16:52 UTC。依據[凍結協議](ng0086-std-gradient-intervention.zh.md)啟動，協議原件未在啟動後修改。**後續 evaluation-v2 已補齊配對分析，見[完整結果](ng0086-evaluation-results.zh.md)**。下列原 run 的缺口與啟動快照保留歷史時點，不代表評估仍未完成或仍在執行。

## 原 run 終態：穩定性改善，當時品質對照未完成

本次只讀核對遠端現場，未重啟、改 loss、放寬預算或恢復 polling。所有 NG86 worker 已結束，相關 tmux session 已退出；GPU 0 已有其他工作，不是可供本次隨意重用的空閒卡。

| Phase | 終態 | UTC 完成時間 | 耗時 |
|---|---|---|---|
| train-control | 完成至 2846；2847 捕捉預期支持失效 | 08:06:21 | 1968.65 秒 |
| train-fullstd | 完成至 3072，全部 1024 次新更新，無 failure | 08:14:54 | 2469.10 秒 |
| profile-control | ETA gate 停止，只有 16 個 base/fit/FEVER row | 08:07:23 | 51.05 秒 |
| profile-fullstd | 三個 checkpoint，各 640 題，共 1920 row | 08:21:17 | 372.80 秒 |
| review | 偵測 profile-control failed 後退出，未產生正式配對結果 | 不適用 | 不適用 |

原始 `inputs.json` SHA 與 16 個來源 payload 仍吻合；四個 phase 共 78 個 payload 的 exact inventory/SHA 全部核對通過，包含 failed profile 的原始證據。所有 owned process group 已關閉、四個 ClearML offline task 已 close，沒有 server-sync 聲明。`train-control` 的 passed receipt 只代表成功捕捉預定失效，不代表完成訓練。大型新輸出目前仍在 Lambda2，本次未聲稱已新增完整本地副本。

### 穩定性與支持漂移

Control 的第 2847 步出現有限、非空的 query：255 個活化全部落在 zero-RMS 維度，分母為零；參數仍有限。它與 NG85 的第 2840 步不是 bit-exact 同一步重現，但屬於相同支持失效機制。Full-std 則完成至 3072，三次 640 題評估皆無 invalid query。

對兩臂共同的第 2753--2816 步，核對相同 64 updates、256 queries 及順序後，直接統計原始訓練紀錄：

| 診斷量 | 原 loss | Full-std |
|---|---|---|
| zero-RMS 維度上的 query raw mass 比例均值 | 81.54% | 16.40% |
| query NNZ 均值 | 335.80 | 74.32 |
| 候選 document NNZ 均值 | 2485.97 | 801.80 |
| query 分母最小值 | 0.001050 | 0.702178 |
| score std 中位數 | 49.33 | 3.44 |
| score std 最大值 | 13779.37 | 8.22 |

這是 std backward 介入抑制支持/尺度漂移的局部對照證據，不是所有訓練失敗的唯一根因證明，也不是 posting bytes、DF 或 native latency 的測量。

### 已完成的 512 題 validation

以下均是 full-std、四域等權、原 bank candidate-only 的已曝光 validation；不與舊 64 題均值直接混比。

| Checkpoint | Hybrid nDCG@10 | Recall@10 | Document NNZ 均值 | Zero-RMS raw mass |
|---|---|---|---|---|
| 2560 | 0.657995 | 0.752400 | 1008.28 | 21.23% |
| 2816（預定主要點） | 0.656405 | 0.755557 | 798.08 | 15.99% |
| 3072 | 0.660089 | 0.757739 | 752.74 | 12.67% |

同一候選集的既存 PPLX dense 參考 nDCG 為 0.745786，BM25 為 0.461216。因此尚未達到 hybrid 超越 dense；這些候選集數字也不能替代完整語料上的產品 baseline。

另外只用既有結果作歷史重疊分析：重新驗證 NG85 profile 的封存 inventory/SHA，並按 query identity、split、domain、完整候選 ID 集逐題核對原有 64 題 validation。没有新增 inference，沒有從大 panel 中按表現挑題。

| 原有同一 64 題 | nDCG@10 | Recall@10 | Document NNZ 均值 |
|---|---|---|---|
| 原底座 | 0.725384 | 0.794164 | 290.58 |
| 介入前 B/2048 | 0.663508 | 0.785395 | 2529.91 |
| Full-std/2560 | 0.721606 | 0.815343 | 1004.20 |
| Full-std/2816 | 0.704844 | 0.776280 | 789.42 |
| Full-std/3072 | 0.696876 | 0.797262 | 745.14 |

此為事後整理的歷史重疊描述，不替代預定 512 題 paired endpoint，不據此挑選 2560 為最佳模型。它顯示部分排名恢復，但持續訓練並非單調提升，NNZ 仍高於原底座。

### 缺口與下一步

對照 profile 的前 16 題投影為 1859.42 秒，比 1800 秒 cap 高 59.42 秒，因此按原協議提前停下；實際執行僅 51 秒，不是已耗盡 30 分鐘，也不是新的模型數值失效。現行估時從 model load 之前計時，並將前 16 題耗時外推到全部 2560 row，再乘 1.5；固定初始化開銷也被放大。Full-std 的實際 1920 row 在 372.80 秒完成，可作估時計量診斷，但不能替缺失的對照評估背書。

Control 的 2560/2816 checkpoint 已成功保存，缺的是其評估及同一 512 題的 base/start anchors，而不是模型或必須重新訓練。下一步建議另立不可覆寫的 bounded evaluation continuation，分開初始化與 steady-state 估時，保留硬 timeout、同一 panel、同一 checkpoint 和主要比較點；先補齊配對結果再決定是否完整重訓。此次僅匯報，未執行該延續或改寫舊 gate。不能把 standalone summary 中的 paired-unavailable 誤解為沒有 checkpoint：2560/2816 存在，是 profile 缺失。

## 啟動與准入

### 隨後授權的 evaluation-v2

使用者同意推進後，另建[唯讀評估延續](ng0086-evaluation-continuation.zh.md)，不覆寫上述失敗。新 manifest SHA256 為 `10ae379dc050a3783861d0ae9fc4d80c5e09d5174af2f7655c0210dd4e2eb33b`；18 個凍結 source/protocol payload 與 manifest 已在 Betty 獨立副本逐檔驗證。Mac 相關回歸 `40 passed, 5 skipped`；Lambda2 CPU/Torch 回歸 `20 passed`。首次 Linux 測試因測試 bundle 遺漏既有 `review_ng85_diagnose.py` helper 有一項 import failure，補齊測試依賴後全部通過，未改評分或忽略測試。

評估 worker 使用 GPU 3，PID `2411286`，start Unix `1789749693.229243`，wall cap 維持 1800 秒。其他三卡已有工作，未介入。四域 fit/validation 共 8 個原 full-std/2816 anchor 與 16 個舊 base anchor 的 raw-score 最大絕對差均為 `0.0`，指標一致，模型 hash 未變。完整 2560 row 在 504.79 秒內完成、exit 0，配對 review 於 16:50:11 UTC 完成。主要點 full-std/control nDCG 為 0.656405/0.599128；相對底座 0.647473 的小幅改善區間仍跨零。新 profile/review 與原兩個 profile 均已完整鏡像及 SHA 核對，詳見[結果與後續方向](ng0086-evaluation-results.zh.md)。沒有自動新增訓練。

以下為 07:39 UTC 的歷史啟動快照。

兩個獨立單卡工作已在 Lambda2 實際執行，不是排隊或僅建立設定：

| Arm | GPU | Worker PID | Start Unix | 最新已記錄更新 |
|---|---|---|---|---|
| 原 stop-gradient std 對照 | 0 | 2307189 | 1789716812.4082832 | 2176 |
| Full-std 唯一介入 | 3 | 2307416 | 1789716825.3665292 | 2176 |

兩臂均從 B/2048 恢復，故這是各完成 128 次新更新，不是從零訓練 2176 步。GPU 1/2 已被其他工作使用，未介入。GPU 0/3 是兩個各自單卡工作，沒有 DDP 或跨卡通信。

- 凍結 inputs SHA256：`859dbcdc6aed659f74936551bdd6436e2018723f91b61ea71dec583972135783`。
- 16 個來源/協議 payload 及 manifest 已有本地大檔區的 SHA 驗證副本；正在生成的 checkpoint/log 不冒稱已完成備份。
- 兩臂實際載入 model SHA：`55b8e831bd59628a6aba28544650f734bc017db5158c6aa4c6f901a15de0ef8c`。
- 兩臂實際載入 optimizer SHA：`2ec275ed9e3fdac57bfb486b3dd98fedd0ae2390662a7fd237200259a045b212`。
- 兩臂均通過四域 full-graph/VJP 檢查，每域 relative gradient L2 為 `0.0`；兩種 loss 的 FP32 forward 完全一致。
- Mac 擴展相關回歸：`36 passed, 5 skipped`；Torch-only 測試未在 Mac 執行。Lambda2 新介入與既有診斷測試：`16 passed`，包含真實 Torch 梯度檢查。新增來源的 Ruff 與文件連結/空白檢查通過。
- ClearML 使用 offline actual-start/close 記錄，未同步到遠端追蹤伺服器。

NG85 observational v2 的全部 23 個診斷 payload（1,193,979,742 bytes）亦已完成獨立本地鏡像，exact inventory/SHA 全部吻合，包括首次漂移與失敗時的 model/optimizer。遠端原件保留；此驗證是完整副本，不宣稱已做恢復演練。

## 有界執行與估時

第 16 次新更新的保守 train ETA 為對照 `4565.95` 秒、full-std `4232.95` 秒，均小於原定 `5400` 秒上限，不需放寬。第 128 次更新的訓練內計時為 `315.98` / `306.75` 秒，線性推估訓練窗口約 41--42 分鐘，尚不包括 checkpoint、前置驗證或評估。

兩臂平行執行，整體暫估 50--70 分鐘；profile 進入後仍需自己的前 16 row ETA gate，不將此估時當成完成保證。每臂 train/profile 外部硬上限分別為 90/30 分鐘，另有記憶體、顯存、磁碟保護。原 recurring polling 沒有恢復。

兩臂各自完成固定 train 窗口後，自動執行已凍結的 640 題 profile。另有一次性的 bounded join/review 工作等待四個 phase 的終態，驗證 inventory/SHA，再產生配對分析；最多等待 9000 秒，非無限監控。任何未預期 execution failure 均保留證據並停止，不重試或擴大研究。

## 本輪要回答的問題

主要品質比較固定在 **2816**，較早觀測為 2560；3072 僅在雙臂皆到達時作配對比較。評估同時列出原底座和 B/2048 起點，防止把「比已退化對照稍好」稱為恢復底座能力。

1. 標準差梯度是否改變支持外 activation mass、分母、NNZ 及 score scale 的漂移？
2. 能否在相同更新進度改善完整 BM25+SAE 的 nDCG@10 / Recall@10，而不只是避免例外？
3. 能否跨過舊第 2840 步失效區間？當輪 control 若未重現，必須如實保留反例，不能替它套用歷史結局。

這是從已漂移 checkpoint 介入的機制實驗。512 題 validation 仍是曝光研究資料，candidate-only 品質不等於 full-corpus recall、DF 成本或泛化。早期 NNZ 變小不構成排序或速度改善證據。尚無新品質結果；不重校 RMS、不新增 epsilon、不改 lexical 權重、監督、seed 或訓練預算。

正向結果才進入從成熟底座開始的完整 matched A/B 設計；只改善穩定性而不改善排名，則不擴大；若仍有支持失效，下一個單因素候選才是 TRAIN-only 正支持校準。這些均不是本次 runner 可自行啟動的工作。
