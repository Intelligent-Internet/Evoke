# NG-0079：固定排序目標下的等曝光 query 監督廣度試驗

日期：2026-09-13。這是產生新訓練結果之前凍結的單一比較，不是 NG77 失敗後的自動擴訓。此階段先準備 TRAIN witness；訓練控制器必須另行完成實作、測試和來源凍結，不能把資料準備通過當作模型品質通過。

## 1. 問題與依據

最終目標仍是 **完整 BM25 + learned semantic hybrid** 在獨立未見資料上的前排 nDCG 和 all-positive Recall 超過 competitive PPLX dense，同時降低實際 native 總成本。不是只讓 SAE 更稀疏，也不是只提高 TRAIN 擬合。

[NG69](ng0069-final-breadth-review.zh.md) 的三種 query order 在等曝光下，擴大 query 廣度取得 new-DEV nDCG +0.012425，95% CI [0.007169, 0.017956]，但仍低於 dense 0.039682，document NNZ 達初始的 2.34–2.52 倍。因此只繼承「比較監督覆蓋」的方法，不繼承該輪 CE loss 或無條件放大。

[NG71](ng0071-final-pilot-review.zh.md) 的 D 排序目標優於 A，但 NQ 泛化下限失敗；[NG72](ng0072-retention-diagnosis-review.zh.md) 指向存活 support 的權重漂移，而不只是 posting 丟失。[NG73](ng0073-gradient-replay-review.zh.md)、[NG74](ng0074-uniform-control-review.zh.md)、[NG75](ng0075-parameter-update-diagnosis-review.zh.md) 和 [NG76](ng0076-authentic-midpoint-review.zh.md) 不支持把問題簡化為單個 query 自己的錯誤梯度、某組 metric 權重或只在後半程才出現的變壞。

[NG77](ng0077-final-retention-review.zh.md) 的 trusted-margin retention 沒有通過原定品質門檻。[NG78](ng0078-boundary-coverage-review.zh.md) 發現：NQ Pilot 受損 gold 的 40 個持續 top10 reversal 中，34 個已在可信監督池內，只有 1 個在池外；Sentinel 的原始 gold 也有明顯損失。這不是 causal attribution，但使「繼續加大候選 K」缺乏優先性。下一輪檢驗不同 query 提供的約束是否比重複相同 query 更能保留前排判別能力。

## 2. 唯一變量與比較

| 項目 | R：少量 query 重複 | B：更多 query 一次 |
| --- | --- | --- |
| 不同 TRAIN queries | 384，沿用原 NG71 Pilot | 1,536，原 384 + 新增 1,152 |
| 每個 domain | 128 | 512 |
| 每 query 曝光 | 4 次 | 1 次 |
| 總 query 曝光 | 1,536 | 1,536 |
| AdamW updates | 384 | 384 |
| objective | 原 D balanced soft pair | 相同 |
| witness reference | 共同 NG3 初始 snapshot，始終固定 | 相同 |

兩臂均**重新從同一成熟 NG3 checkpoint096 初始化**，不從 K96 繼續，也不物理復用舊 optimizer prefix。原 NG77 Z96 僅是共同首 96 update 的可重現性檢查，不是新品質證據。不重啟或修改已封存的 NG71/77 工作。

第一個 384 exposure 完全沿用 NG71 第一輪的順序。其後 R 用 seed 79001 對原 Pilot 產生三個 permutation；B 按 R 每個位置的 domain，放入該輪尚未用過的同 domain query。因此每一個 update 的 domain 序列相同。不能宣稱 token、document 或 FLOP 曝光相同；必須記錄實際數量和時間。

新增 query 只在原 6,144 TRAIN 中選取：已解析的來源 lineage、固定 `SHA256("NG-0079/breadth-v1/" + query_id)` 順序、每 domain 新增 384。先排除原 Pilot、Sentinel 的 ID 和 NFC/空白正規化 query text，再全域去重。保留所有 resolved positives；不按 teacher agreement、舊分數、受損案例、長度或「容易學」篩 query。已知 teacher-opposed/unobserved pair 繼續依原 D 契約處理，不能偽造負例或人類標籤。

## 3. 不變的數學與執行契約

固定 BM25 權重 0.1、semantic 權重 0.9、既有 RMS，不重估 calibration。NG3 shared trunk + MLM vocabulary head 都可訓練；這不是重新訓練 PPLX，也不是新增 latent SAE。query64/document256 是輸入 token 上限，不是輸出 posting cap。FP32、禁用 TF32，所有 tokenizer/模型/資料/teacher 皆由既有 manifest SHA 綁定。

每 query 的 loss 沿用 D：對全部 gold 平均，再對可監督 pair 做帶前排 nDCG/Recall metric weight 的 soft-target logistic loss。權重與 pair 池由共同初始完整 corpus 排名決定、stop gradient；教師溫度 0.04、student 溫度 1、pair floor 0.05。不增 retention 項、不 sweep lambda、不換 uniform、不凍結單一 encoder role。

原始池與全部 gold 完整保留，最多新增 64 個 witness，最多 256 documents；超限就停止，不能截掉 gold。witness 使用原 hybrid head/boundary、PPLX head、BM25 head、固定 hash sampling 配額。**不做 A96 refresh 或各臂 self-mining**，避免同時改變 query breadth 與候選刷新。每次更新仍用當前 encoder 的 fresh query/document forward 與原 exact VJP，不把初始向量當作可訓練表示。

原 AdamW：trunk LR 5e-6，head LR 2e-5，weight decay 0.01，betas 0.9/0.999，eps 1e-8，clip norm 1；每 update 4 queries，query microbatch1 / document microbatch4。以 96-update 子階段封存，最終只選 384 checkpoint，不能按觀察到的分數提前挑 checkpoint。

## 4. 資料準備與界限

先重用 SHA 綁定的 NG71 全 6,144 TRAIN provenance/text/teacher audit，重新產生明確 selection/order/composition manifest。這是重用已核驗的來源文字證據，不是重新取得 human judgments。新增 query 的 teacher 分數可由既有完整 PPLX codes 計算；這仍是新的 TRAIN 監督觀察，但沒有新 encoder inference。

對 1,536 queries 使用完整 233,009 documents 的 cached initial/PPLX/BM25 codes 準備 witness。兩種 float64 reduction 必須逐分數相符至 1e-12，獨立 head selection 和全部 gold/selected ranks 必須一致，tie 固定 corpus integer ID。前 384 的 witness、D loss/derivative、teacher、all-positive lineage 和 token visibility 必須重現旧 step0。不得靠放寬 tolerance 接受 ranking 差异。

每 domain 的可監督 positive fraction 必須 >=0.8；不能刪掉無監督 query 來湊 gate，但整批全無監督時停止。保留每 domain original/added positives、truncation 與未知 supporting span；沒有 span judgment 不能宣稱證據不存在。

NG71 384-query snapshot 主迴圈約 467.8 秒、整階段 506.7 秒、peak tree RSS 約 5.38 GB。線性估計本輪迴圈約 31 分鐘，加 1.5 倍保守係數約 47 分鐘，採 5,400 秒硬上限而不是盲目使用舊小批次 ETA。前 16 query 的 1.5 倍預測必須 <=5,100 秒。沿用 CPU4、treeRSS16 GiB、host available >24 GiB、disk free >40 GiB；獨立 lambda2 CPU 階段，不占 GPU。actual-start ClearML 明確 offline，必須成功關閉，不能冒充已同步遠端。

## 5. 終點、門檻與退出

主要比較使用已多次觀察但始終不參與梯度的 **384 TRAIN_SENTINEL**，每 domain128。它不是 independent holdout。原 Pilot 和新增 TRAIN 的擬合結果分開列出，不與泛化混算。DEV/LOCKED_TEST 不在本輪資料使用或控制圖中。

兩臂都對完整 corpus 評估完整 hybrid，包含全部 gold，並列初始、PPLX dense、BM25。主終點為固定 terminal384；paired domain-stratified bootstrap 10,000 次，seed79079。門檻全部凍結：B-R macro nDCG >=0.005 且95% lower >0；Recall95% lower >=-0.002；每 domain 的 nDCG **及** Recall 相對 R 和初始都 >=-0.005；B 的完整 document NNZ 和 Sentinel query-DF mean 均不得 >1.25 倍初始。

最後兩項只是 semantic cost proxies，不能宣稱 native total-cost 勝利。actual token/document exposure、NNZ、DF、訓練時間、RSS/GPU峰值都要列出。若通過，只支持進入另行凍結的 replication/未見評估/native-cost qualification；不等於已超越 dense。任一品質/成本 gate 失敗，不自動增加步數、資料、K、lambda 或放寬下限。

此規模是原 384-query 規模的 4 倍 distinct breadth，但只有 NG69 的 1/4 總 exposure；足以做有方向性的第一個診斷，仍不足以估計 seed variance。每 96-update 子階段的預計 GPU 時間由新的 canary 決定；原 D96 約6分鐘只是歷史參照。啟動前重新確認 GPU/鎖，單卡，不改網路，不碰 production。

全部產物保留 exact inventory、SHA、退出碼、process-group closure 和雙端 mirror；行數達100%不是完成。獨立 reviewer 核對 selection/order、scores/ranks/metrics 和原定 gate。只在真正結果或風險時通知，不把準備活動當成科研突破。
