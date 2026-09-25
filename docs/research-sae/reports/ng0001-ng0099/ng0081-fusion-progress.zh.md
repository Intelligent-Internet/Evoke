# NG-0081A：固定融合探索執行進度

最新遠端核對：2026-09-14 20:27 UTC。**八個 phase 已於17:38:34 UTC（多倫多13:38:34）全部完成並自動停止**，controller 實際經過1734.49秒，約28.9分鐘。本報告不替代 [設計](ng0081-overall-hybrid-quality-cost-plan.zh.md) 或已凍結的 [執行契約](ng0081-fusion-execution.zh.md)。這是固定 endpoint 的已曝光診斷結果，不是新訓練、獨立泛化或產品資格。

## 完成結果

原模型、S、P 的 serving/z-score 校準全部選 alpha0.1。Serving 校準因此完全重現歷史 hybrid，沒有改比例的收益；z-score 只有小幅品質改善。

| 方法 | Macro nDCG@10 | All-positive Recall@100 |
|---|---:|---:|
| 原 hybrid / serving 校準 | 0.770716 | 0.948846 |
| 原 hybrid / z-score 校準 | 0.771412 | 0.948412 |
| S hybrid / serving 校準 | 0.786079 | 0.951925 |
| S hybrid / z-score 校準 | 0.787832 | 0.952657 |
| P hybrid / serving 校準 | 0.775382 | 0.946605 |
| P hybrid / z-score 校準 | 0.780330 | 0.951438 |
| PPLX dense | 0.822175 | 0.965235 |
| TRAIN 校準 dense+BM25 | 0.820317 | 0.971423 |

S 的 z-score 相對自身歷史配置，nDCG 增量0.001753，95%配對區間[-0.001221, 0.004813]；Recall 增量0.000732，區間[-0.000895, 0.002360]，兩者均跨零。P 的對應增量為 nDCG0.004948 [0.001285, 0.008636]、Recall0.004833 [0.001750, 0.008299]，在這組診斷上有較明確改善，但其點估計仍不及 S，更未超越 dense。

即使看 S 的 z-score 對照，與 dense 的 nDCG 差仍為-0.034343，區間[-0.043053, -0.025449]；Recall 差-0.012578，區間[-0.018498, -0.006768]。校準只收回 S 原有 nDCG 差距約4.86%。結論限於此次固定 grid 和兩種 normalization：**不支持把主要差距歸因於融合比例，或繼續細掃權重；不是證明所有可能融合算法都無效。**

分域披露：S z-score 的 FEVER/HotpotQA/NQ nDCG 分別為0.851111/0.825857/0.686528，dense 為0.910471/0.827457/0.728598。此處不是以單域 gate 否決它，overall 本身仍有明確差距。

## 成本機制的實測

下表是完整233,009文件、1,536題上未平移的非負原始通道 support，並非 native 引擎實際掃描量或延遲。

| 模型 | 語意文件NNZ / 原模型 | 總literal DF-work / 原模型 | 平均語意touch比例 | 平均雙通道union touch比例 | 最大DF比例 | DF至少90%的feature數 |
|---|---:|---:|---:|---:|---:|---:|
| 原模型 | 1.00x | 1.00x | 82.0870% | 97.5097% | 26.4526% | 0 |
| S | 4.02x | 7.57x | 99.9992% | 99.9996% | 97.2289% | 13 |
| P | 4.63x | 1.95x | 98.3079% | 99.5995% | 81.1140% | 0 |

三個模型皆沒有 DF恰為100%的 feature。S 卻平均觸及233,007.22個文件，双通道聯集233,008.03個；**沒有 universal feature 並不等於具有選擇性**。高DF集合與 query 激活的聯合分布才是需要檢查的對象，不能把刪除單一 universal feature 當作解法。

P 的文件NNZ比 S 更多，但總DF-work增幅遠小於 S，說明文件稀疏度本身也不能代替查詢工作量。原 hybrid 的 union 已達97.51%，因此不能只盯「是否觸及文件」一個指標；重複 posting 貢獻數、真實 pruning/decoding 和品質必須分開量測。7.57x是literal工作代理，不是7.57x線上延遲。

## 閉合、校驗與當前狀態

- 八個 phase 全部 exit0、無 error、owned process group closed，ClearML offline actual-start/closed/passed；未放寬任何資源或科學 gate。
- 完整新 run 已複製至 Mac 外部 NG-0081 單元。28份 source/config SHA、八個 phase 共91份受清單約束的檔案 SHA、exact inventory、退出及 tracking receipts 全部通過。另在遠端重跑 input/source/dependency verification，exit0；這不宣稱所有父依賴此次重新做了本機備份或 restore test。
- Mac 使用標準庫从三個模型逐題正例 ranks 重新計算九組 profile 的分域與macro品質，與 review 結果在1e-12內一致，沒有在 Mac 執行 Torch 或 inference。
- 獨立 CSR coordinate head 分數最大誤差1.4211e-14；historical 三組的完整 top100 IDs/scores 和正例排名均精確重現 NG80。Review 是獨立 metric/coordinate 驗證，不冒充另一套獨立全庫排序。
- `review/results.json` SHA：`4bb1e6783776fc79e7a2a6ab41d6fce0ea5eb8922c3b20f0ffb56d0cfeb47a84`。
- 整輪最大 process-tree RSS約7.43GiB，低於16GiB限制。20:27 UTC四張GPU均為1MiB/0%，主機available RAM約117GiB，研究磁碟647GiB可用；NG81 tmux已退出，沒有後續訓練自動啟動。

| Phase | 耗時秒 | 峰值process-tree RSS GiB |
|---|---:|---:|
| encode-calibration | 75.58 | 1.67 |
| calibrate-initial | 199.79 | 1.91 |
| calibrate-S | 271.35 | 6.56 |
| calibrate-P | 230.85 | 7.43 |
| rank-initial | 210.74 | 2.30 |
| rank-S | 380.40 | 6.58 |
| rank-P | 240.03 | 7.43 |
| review | 125.20 | 5.69 |

## 後续決策

這輪關閉「只因沒有公平校準才落後」的主要解釋，不重開更細alpha/RRF/gate搜索。下一個有界問題是：S增加的posting是否為可去除的低排名效用冗餘，還是品質提升需要的分布改變。依既定設計先做TRAIN上的joint mask效用與同work預算DF對照，整組mask後再完整排序；不能把個別feature效用相加當成成功。

成本診斷與後續G/G+B訓練需各自凍結來源、數值和停止契約；沒有因本報告自動啟動它們。G+C/G+B+C仍須先證明能改变真實support，不直接再試DF-FLOPS係數。研究保持overall品質目標，未用於選型的獨立資料和native成本驗證仍然缺失；無模型promotion、部署、locked-test評分或自動輪詢恢復。

以下保留啟動當時的紀錄，其「進行中」描述不代表最新狀態。

## 歷史 review 後的修正

- M550 已有固定 BM25 融合相對 dense 的 macro 正例，M1951 也有相對其 OpenSearch parent 的 macro 改善；不能把融合互補當作尚未試過的新發現，也不能混淆其基線、成本與當前 NG80 模型。
- M160A/M1329 的 RRF、rank interpolation、gate/residual 探索不重開。本輪只補原/S/P 新 endpoint 缺少的對稱 TRAIN 校準，固定七個 alpha 和兩種有明確用途的 normalization，完成後不追加細網格。
- M1518 連 budget-matched DF-FLOPS 都未解決全庫 touch；成本臂不直接重跑此配方。先量測新模型的真實 support、DF 和聯集，再決定是否有可證偽的新成本機制。
- 下一步以完整 hybrid 的預先固定任務權重 overall 結果為主，不要求逐域勝出。保留分域損益、all-positive Recall 和獨立泛化限制，不更改歷史 gate。

文獻與各歷史報告的逐項連結、適用範圍及數學設計見主設計。NG81-A 是一個缺失對照，不宣稱新融合算法。

## 已驗證的準備與啟動

執行單元是 Lambda2 的 `NG-0081/fusion-v1`，透過 Spark 跳板訪問。啟動前重新確認四張 TITAN RTX 閒置；只在 query encoding 使用 GPU 0，其餘 phase 使用共享 CPU lane lock，最多四個計算執行緒。沒有多 GPU 作業，也不改其他人的作業。

- `preflight-v1`：36 passed、1 failed。平面 staging 缺少舊 freeze 測試所需的 repository 相對 tests/docs 佈局；保留此失敗，不當作測試通過。
- `preflight-v2`：修正 staging 佈局並新增契約測試後，Lambda2 CPU 測試 **40 passed、0 failed、0 skipped，41.90 秒**。涵蓋歷史權重精確重現、兩通道端點、全庫勝出文件可不在單路 top100 聯集內、全部正例分母、TRAIN 選型和 freeze 後輸入不可變。JUnit/stdout 已複製到 Mac 外部研究單元。
- 遠端 freeze 驗證 **816 項 dependency entries、28 份 source/config**；Mac 已取得 input manifest 及28份 immutable source/config，逐項 SHA 一致。這不是所有新 phase 結果已完成備份的宣稱。
- 已閉合的 `encode-calibration` 和 `calibrate-initial` 也已複製至 Mac，分別16/11個受 manifest 約束的檔案逐一通過 SHA、exact inventory、exit0、owned-group-closed 和 offline tracking closed/passed 核對；仍在執行的 phase 不宣稱完成備份。
- tmux `ii42_ng81_fusion_v1` 於 **17:09:13 UTC** 派送；controller actual start 為 **17:09:39 UTC**，不是以 tmux 建立時間代替 worker 啟動。
- 本輪零 optimizer 更新，零文件重編，不讀 locked test，不改產品索引或部署，不恢復已暫停的自動輪詢。

| 凍結項目 | SHA-256 |
|---|---|
| `inputs.json` | `ba73d53e688c49ec17c40a41cafabdd17eab2a900b9ea54e9f9f38e1aa831bca` |
| `ng81_fusion.py` | `8bb56b69778b92f35831d7766df3ad01b50220f4b077492d7b6c19d3d4f15825` |
| `test_ng81_fusion.py` | `9274f21a48be69fab01546edbc0fdbefeb0cec1ff6ea18803b2735349535b0a2` |
| 執行契約 | `1df75aa0a9b1ddd38c10c266129b788e5f278e63a62134cf2c1328032d20ad01` |

程式與測試保存在主工作樹，但本輪尚未提交；freeze SHA 才是實際執行版本，不能用之後可能變動的工作樹代替。

## 啟動時階段與估時（17:18 UTC）

| Phase | 已驗證狀態 | 實測 |
|---|---|---|
| `encode-calibration` | exit0、owned group closed、ClearML offline closed/passed；controller 已進入下一階段 | 75.58 秒，峰值 process-tree RSS 約1.67GiB |
| `calibrate-initial` | exit0、owned group closed、complete receipt；controller 已進入下一階段 | 199.79 秒，峰值 process-tree RSS 約1.91GiB |
| `calibrate-S` | 正在執行，16題成本 canary 通過 | 16題8.07秒，含1.5x係數及300秒餘量的階段估時590.55秒 |
| 其餘5個 phase | 待執行 | 不宣稱已有結果 |

三個模型的384題 query encoding 都通過前後參數 hash 核對；GPU 隨階段完成已釋放。模型載入輸出包含 `lm_head.decoder.bias` 的 missing-key 訊息，但載入後完整 state dict 的 hash 與封存版本精確一致，沒有以新隨機權重替代原模型。

原模型 TRAIN 校準的 serving/z-score 皆選 alpha0.1；這只說明固定384題選型結果，尚不是1536題診斷收益，更不是獨立泛化結論。排名階段必須精確重現 NG80 historical profile，最後 review 再獨立重算選型、正例品質與 head coordinate 分數。

initial canary 的保守階段估時499.74秒，實際閉合199.79秒。S 的590.55秒同樣是保守 phase 估時，不是剩餘整輪時間。排名與 review 尚未起跑，待其實際 canary/吞吐後再更新整輪預期，不用校準的小樣本耗時冒充完整 ETA。

17:15 UTC 主機約113GiB available RAM，S worker RSS 約3.59GiB；没有看到本輪引發的記憶體壓力。ClearML 為 offline，未宣稱已同步遠端服務。

本機只做標準庫檢查，沒有執行 Torch 或訓練：兩份新 Python 檔通過 AST/行長檢查，相關文件156個本地連結目標存在，NG階段文件數56與索引一致，`git diff --check` 通過。這不等於產品全套回歸或新研究品質通過。

## 終止與後續

固定八個 phase 完成即停止；任何資源/內容/數值/閉合 gate 失敗亦停止並保留原嘗試。未取得 `review/results.json` 與所有成功閉合證據前，不宣稱融合有效或模型已達標。完成後再區分「比例失配」、「normalization 效果」、「真正表達缺口」及「support 成本」來決定訓練；不自動追加四臂或恢復 polling。
