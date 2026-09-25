# NG-0066：訓練充分性、順序重複與資料廣度決策

日期：2026-09-12。研究版本：`NG-0066/attempt-v2`。這是下一代模型的實驗結果，不是 Beta 1 模型變更或 production 驗收。

## 結論與完成邊界

三個訓練順序種子全部完成四輪訓練及第 1、2、4 輪全背景評估。較長訓練能明顯改善固定 TRAIN 上的 BM25+SAE 混合排序，但不能把這個改善等同於泛化提升：已曝光 DEV 只有較小收益，條件式 query bootstrap 區間跨零，且三個最終種子仍低於同背景的 PPLX dense。

結果支持下一階段獨立測試「不同監督樣本數」而非只對原有 1,536 題增加輪數。它不證明資料廣度是唯一原因，也不證明 backbone、readout、候選池或 loss 已無瓶頸。本輪沒有啟動第 5--8 輪，沒有改動 production；後續另以 [NG68](ng0068-supervision-and-ranking-review.zh.md) 與 [NG69](ng0069-matched-exposure-breadth-plan.zh.md) 記錄新研究。

執行於 2026-09-12 05:51:45 UTC 完成；12 個 train 和 9 個 eval 階段均正常退出、資源界線未觸發。GPU 已釋放。遠端 243 個 frozen source／階段產物 SHA 核對通過。本機已獨立重算 15,552 份 query 評估記錄的指標及正例／top-100 一致性，並核對連續 optimizer 恢復鏈。**完整 348 檔、8,096,357,527 bytes 已回收，rsync exit0；全量本機 SHA、243 個 frozen 來源／階段檔，以及九組 posting 的 NNZ、CSR bytes、DF 代理重算全部通過。** 遠端來源保留；沒有以副本校驗冒充新的模型 restore 或完整 inference。

## 固定協議

| 維度 | 本輪設定 |
|---|---|
| 初始化 | NG59 使用的同一 NG3 checkpoint-096；不是三個獨立初始化 |
| 訓練順序種子 | 59059、66061、66067，不挑最佳種子 |
| 訓練資料 | 每輪固定 1,536 題，Fever／HotpotQA／NQ 各 512 |
| 曝光量 | 每種子 4 epochs、6,144 題次、1,536 optimizer updates |
| 合計 | 4,608 updates；梯度累積每步 4 題 |
| 模型／目標 | NG59 encoder、雙側梯度、完整非負 readout、原有 teacher／positive 混合 target 不變 |
| 混合權重 | lexical 0.1、semantic 0.9；表中不是 SAE 單獨成績 |
| 輸入／數值 | query64／document256 tokens，原 RMS、AdamW、constant LR、clip=1、FP32、noTF32 |
| 全背景評估 | 59,111 文件、1,728 query；其中 192 舊 DEV 已曝光 |
| 資源限制 | 單 GPU、CPU4、host tree 16 GiB、GPU allocated 20 GiB、每 phase 1,800 秒、controller 六小時 |

觀測最大 host process-tree RSS 約 2.847 GiB，訓練峰值 GPU allocated 約 1.749 GiB。allocated 並非整張卡的總使用量。每階段 ClearML 都從實際開始記錄並正常 close，但 21 份 task 是 offline，尚未同步追蹤伺服器；不能稱服務端已有完整結果。

v1 是保留的失敗嘗試：跨次重訓位元相等 guard 過嚴，且最終 checkpoint 在 guard 前未保存。v2 在改動前凍結數值容忍度和先保存策略，不修改模型、loss 或資料，也不把 v1 當作成功。v2 種子59059第一輪參數與 NG59 參考一致；跨次編碼最大分數差為 `3.592626e-6`，六個 domain/split 的 nDCG10／Recall100 差均為零，符合預先設定的數值契約。當次兩種獨立累加路徑的分數差則為零，兩者不能混為一談。

## 全部種子結果

| 順序種子 | Epoch | TRAIN nDCG10 | DEV nDCG10 | DEV R100 | 文件 NNZ |
|---|---:|---:|---:|---:|---:|
| 59059 | 1 | 0.873099 | 0.893538 | 0.981424 | 15,589,919 |
| 59059 | 2 | 0.894908 | 0.893893 | 0.981424 | 18,547,386 |
| 59059 | 4 | 0.922063 | 0.894454 | 0.981424 | 18,124,486 |
| 66061 | 1 | 0.874037 | 0.890731 | 0.981424 | 14,552,323 |
| 66061 | 2 | 0.893753 | 0.897517 | 0.981424 | 18,532,538 |
| 66061 | 4 | 0.922655 | 0.897394 | 0.982986 | 18,393,000 |
| 66067 | 1 | 0.874545 | 0.893037 | 0.981424 | 16,505,824 |
| 66067 | 2 | 0.895092 | 0.897991 | 0.981424 | 17,553,381 |
| 66067 | 4 | 0.924200 | 0.900582 | 0.985330 | 18,509,103 |

| Epoch | TRAIN nDCG10 均值 ± 順序種子 SD | DEV nDCG10 均值 ± 順序種子 SD | DEV R100 均值 |
|---|---:|---:|---:|
| 1 | 0.873894 ± 0.000734 | 0.892436 ± 0.001497 | 0.981424 |
| 2 | 0.894584 ± 0.000726 | 0.896467 ± 0.002242 | 0.981424 |
| 4 | 0.922973 ± 0.001104 | 0.897476 ± 0.003065 | 0.983247 |

相同背景、相同 query 的 PPLX dense nDCG10 是 TRAIN `0.883058`、DEV `0.920745`。最終混合模型 TRAIN 平均高 `0.039915`，DEV 平均仍低 `0.023268`。已訓練資料上的超越不是整體目標通過。沒有在更大的 NG67 文件背景上沿用這些 dense 數字。

新增本機分析先對每個 query 的三個種子差值取平均，再按 domain 等權分層 bootstrap 10,000 次，seed66066。不是把相同 192 題重複當成 576 個獨立樣本；區間條件於這三個順序種子，不能代表初始化、上游污染或新資料集的不確定性。

| 比較 | TRAIN nDCG10 差及 95% 區間 | 已曝光 DEV nDCG10 差及 95% 區間 |
|---|---|---|
| Epoch 1 → 2 | +0.020691 [0.017431, 0.024089] | +0.004032 [-0.001263, 0.010422] |
| Epoch 1 → 4 | +0.049079 [0.043366, 0.054952] | +0.005041 [-0.006236, 0.015771] |
| Epoch 2 → 4 | +0.028389 [0.024399, 0.032429] | +0.001009 [-0.007794, 0.009420] |

三個種子 epoch1→4 的 DEV 點估計都為正，但區間仍不足以宣稱穩健泛化增益。epoch2→4 的新增收益尤其有限；不能在此事後挑 epoch2 或最好種子，再把它當成預先選定的泛化結果。

## 領域與全部正例損害

| Domain | TRAIN nDCG10：epoch1 → 4 | DEV nDCG10：epoch1 → 4 |
|---|---|---|
| Fever | 0.931303 → 0.947528 | 0.923370 → 0.939853 |
| HotpotQA | 0.894439 → 0.934009 | 0.915316 → 0.918858 |
| NQ | 0.795938 → 0.887381 | 0.838621 → 0.833718 |

NQ 的 TRAIN 明顯上升而 DEV 平均下降，是需要擴展獨立監督並檢查目標泛化的訊號，不是模型沒有訓練的證據。NQ DEV 的逐種子變化為 `-0.013535`、`+0.003403`、`-0.004575`，並非三次都退步。

下表比較各自 epoch1→4；「正例排名改善／退步」包含所有已知正例，沒有只選每題最好的一個正例。每個種子的 DEV 有 192 題、330 個 query-positive 配對，不是 330 個獨立 query。

| 種子 | 集合 | nDCG 改善／退步 query | 正例排名改善／退步 | 進入／跌出 top100 |
|---|---|---|---|---|
| 59059 | TRAIN | 400／39 | 766／214 | 38／2 |
| 66061 | TRAIN | 415／46 | 775／213 | 38／2 |
| 66067 | TRAIN | 417／41 | 761／229 | 30／2 |
| 59059 | DEV | 25／21 | 59／47 | 0／0 |
| 66061 | DEV | 27／20 | 55／49 | 1／1 |
| 66067 | DEV | 32／19 | 61／45 | 2／1 |

具體 DEV 損害：種子66061的 Fever 正例 rank59→101；種子66067的 NQ 正例 rank53→112。HotpotQA 的一些正例同時進入 top100，因此 macro Recall100 上升與局部正例丟失可以同時發生。逐域計數、query ID 和正例 ID 保存在獨立分析 JSON 中。

## 成本界線

| 三種子均值 | Epoch 1 | Epoch 2 | Epoch 4 |
|---|---:|---:|---:|
| 文件 semantic NNZ | 15,549,355 | 18,211,102 | 18,342,196 |
| 文件 CSR bytes | 124,631,291 | 145,925,261 | 146,974,019 |
| 全部 query semantic NNZ | 60,274 | 57,987 | 58,185 |
| 每 query 的 semantic DF 工作代理 | 126,561 | 126,500 | 128,806 |

Epoch1→4 文件 NNZ 約增加18%，query NNZ 減少約3.47%，但 DF 工作代理反而增加約1.77%。更少 query atoms 不等於更少掃描工作：還取決於保留 atom 的文件頻率。完整 readout 的採用是沿用診斷分支，不表示 NG59 證明它優於截斷。

此處工作代理是每題活躍 semantic atom 的文件頻率總和；沒有計算 block-max 跳過、cache、完整 lexical 路徑或引擎排程。CSR bytes 不是 II42 原生磁碟索引大小，compressed NPZ 傳輸大小也不是 CSR resident bytes。沒有原生查詢延遲、吞吐或總成本改善的結論。

## 下一步：歷史提案與後續協議

NG-0067 已準備 13,824 TRAIN（1,536 舊題加12,288新題）、1,536 新 DEV、1,536 LOCKED_TEST，三域均衡，共同背景233,009文件。資料準備與來源／ID／文字／兩種 tokenizer／全部正例審計通過；近重複檢查有 preparer fixture，但沒有獨立重做相同演算法。保留 transductive 文件重疊和無法排除上游訓練曝光的限制。新 DEV／TEST 尚未評分。

下一階段應先凍結窄資料與寬資料的配對協議：相同初始化、模型、loss、候選池生成策略、共同文件背景與新 DEV。匹配更新量可比較舊1,536題×9輪與新13,824題×1輪，兩者都是13,824題次；這是新實驗提案，不是本輪已執行的 epoch5--9。另保留寬資料較充分訓練條件，避免只給寬資料一輪就判定它無效。具體種子、LR schedule、teacher生成及全背景評估預算需在下一輪開始前決定。

後續已選擇成本較低、可直接重用本輪 checkpoints 的 NG69 首個對照：舊 1,536 題四輪與 6,144 不同題目一輪。上述九輪對照保留為歷史提案，沒有執行，也不是目前自動化任務。以 NG69 凍結協議為準。

先區分資料廣度和訓練曝光，再考慮修改表示或成本 loss；同時改動會無法解釋收益來源。繼續報告 per-domain quality、全部正例損害與 DF 工作代理。只有在明確可行的候選上做原生引擎成本測量，不能以代理值完成成本驗收。LOCKED_TEST 必須等候選選擇協議凍結後才允許使用。

## 證據與重現

外部產物單元為 `NG-0066/attempt-v2`、`NG-0066/review-v2` 與不可變的 `NG-0059` 參考資料；不要改寫 frozen v1、NG59--65 或 NG67。source-frozen manifest SHA256：`2119b161855b044f18b94082bcf6e1dabb7e446829deef076feb012e7f2d987b`。

遠端348個檔案的完整回收 manifest SHA256：`440d626ca9d539db9670c2d4c7cac3d4d2fcb38b22514945f19e4cab939ca7b3`。本機完成全量驗證的 `full-review.json` SHA256：`e3f5b5fff9443804ae1907626d5e7ca8150860b810ca6c9708a14f5bbba6a6ab`。先前本機逐題及 optimizer lifecycle review SHA256：`0dde3a9b43516e91de43b46fc3c8c12f31f3221b0fd2f50b236adc451948385a`；兩份證據均保留。

本機分析工具不 import frozen trainer、不連線 DB、不啟動 encoder，也不讀 NG67 的 test scoring；原始九次 full-score 獨立累加和全體正例 rank audit 在遠端執行，本機重新驗證儲存記錄、來源／成果 hash 和恢復鏈，不冒充第二次本機完整推論。

```bash
python -B scripts/analyze_ng66_learning_curves.py \
    --run "$RESEARCH_ROOT/NG-0066/attempt-v2" \
    --reference "$RESEARCH_ROOT/NG-0059" \
    --manifest "$RESEARCH_ROOT/NG-0066/review-v2/remote-manifest.json" \
    --output "$RESEARCH_ROOT/NG-0066/review-v2/full-review.json"
python -B -m unittest discover -s tests -p test_ng66_learning_curves.py -v
```

依賴 Python3.12、NumPy；完整 artifact 驗證另使用 SciPy 重算儲存 posting 的 NNZ、CSR bytes 和 DF 工作代理。輸出採排他建立，不能覆蓋已凍結證據；若全量副本尚缺檔，`--manifest` 必須失敗而不是跳過。fixture 目前7項通過。本輪不聲稱已做 PostgreSQL、GPU訓練或整個產品測試套件的重跑。
