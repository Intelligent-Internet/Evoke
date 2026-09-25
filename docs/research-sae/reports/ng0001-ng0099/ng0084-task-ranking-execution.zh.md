# NG-0084：有期限的任務排序監督驗證

日期：2026-09-14。使用者已批准按[路線總審查](../designs/ng-research-route-review-20260914.zh.md)推進。**本文件凍結第一階段 teacher 准入；第二階段的資料 manifest 與訓練排程要在准入後單獨凍結。** 不改歷史 gate，不部署、不讀 locked test，不恢復舊輪詢。

終點更新：第一階段已完整執行與 replay，原 gate 為 `inconclusive_do_not_auto_launch`，故第二階段未啟動。見[結果與裁決](ng0084-progress.zh.md)。下文保留事前設定；執行時的原協議副本在 frozen run 中，沒有依結果回寫或放寬。

## 1. 本輪真正改變什麼

主問題：在成熟 sparse 基座、資料、曝光量、完整 hybrid scoring 和成本處理相同時，更有任務資訊的 teacher 能否帶來可泛化的 overall 品質／成本前沿改善？

- A 使用 PPLX 排序分布，B 使用通過准入的任務排序 teacher 分布。可靠 human relevance 訊號兩臂共用；teacher 不得覆蓋明確的人類相反判斷。
- 不再做 utility mask、線性 probe、alpha／DF penalty 掃描，亦不重新包裝 NG75–78 的 competitor coverage audit。
- M1701 已測 RankT5，但正式 fit 只有四個 updates；其局部結果不能裁決充分訓練的效果。本輪不宣稱首創強 teacher 或 listwise distillation。
- 主產品指標是 **BM25 + semantic overall nDCG@10 超過 pure dense，並降低實際成本**。dense+BM25 是額外強對照；不要求每域都贏，但完整披露分域损失與 all-positive recall。

## 2. 第一階段：凍結的 teacher 准入

只回答「這個現成 teacher 在目前 TRAIN 上是否提供可用增量資訊」，不是模型訓練成果、全庫品質、泛化或延遲結論。

| 項目 | 固定設定 |
|---|---|
| Query | NG80 的 12,288 fit 中每域 hash 抽 128，FEVER／HotpotQA／NQ 共 384；域間交錯 |
| 隔離 | 不使用舊 1,536 diagnostic 做 teacher 選型；不讀 locked cohort |
| Candidates | 復用已封存 NG83 operator bank，保留全部 known positives，不新增 mining、不依 teacher 成敗挑題 |
| PPLX 對照 | 同一 bank 已驗證的 float64 teacher scores |
| 強 teacher | `Soyoung97/RankT5-3b`，revision `40e2b98fcbc6d60457c88508bd775fcb8395a5b0` |
| 來源驗證 | 與 M1701 Betty 歸檔七個檔案 SHA256 逐一相同；不是 NAVER encoder-only RankT5 reproduction |
| 推理 | T5 conditional generation 第一個 decoder step 的 `<extra_id_10>` logit，與原 generate-first-step 對照 |
| 精度／上下文 | FP32，batch 2，`Query: ... Document: ...`，最多 512 tokens；TITAN RTX 不用 BF16 |
| 指標 | 所有 known-positive 的 panel nDCG@10、Recall@10、MRR、完整 positive ranks；同分按整數文件 ID |
| 決策 | 每域等權 macro nDCG 增量至少 0.005，分層 paired bootstrap 95% lower > 0 才准入；不設每域都贏門檻 |
| 不確定性 | point gain 但區間跨零，或小於實用門檻：inconclusive，不自動擴樣或啟動訓練；upper <= 0：本 surface 無有效訊號 |

該 bootstrap 重抽 query，不能消除共享文件與過往曝光的相依性；不是正式顯著性資格測試。候選由既有模型形成，不是所有文件，亦非 RankT5 全庫排序。未標正例只是 unjudged，不據此宣稱人類負例準確度。

RankT5 和歷史 PPLX 的 tokenizer／上下文不完全一致；記錄截斷長度，正向結果仍須在第二階段使用 student-visible text 的配對 target 准備中驗證，不能將增加可見證據混稱為純粹 loss 改進。第一階段不根據任意逐題分數篩掉 teacher 錯誤。

資源：Lambda2 當前僅 GPU3 空閒，單卡；最多 4 CPU threads、32 GiB owned RSS、20 GiB GPU total memory，主機 available >24 GiB、disk free >40 GiB。3 小時硬限；16 queries canary 估算全量時加 1.5 倍時間餘量及 600 秒，超過 10,200 秒則停並保留產物。新 teacher 的 RSS 預算獨立事前設定，不修改 NG71 舊的 16 GiB gate。

其他 GPU／Spark 記憶體正被使用，不能因 GPU utilization 低而佔用。使用共用 GPU lock、開跑前空閒檢查和只終止 owned process group 的 watchdog。ClearML 實際 start/close receipt 必須存在；線上或明示 offline 皆不得冒充已同步。模型載入、數值、資源失败保留原 attempt，不把工程失敗當科學結論。

啟動修正（v2，首個 attempt 保留）：NG9 既有 runtime 不含此 teacher 所需的 SentencePiece／Protobuf。隔離 overlay 固定 `sentencepiece==0.2.1`、`protobuf==6.33.6`，載入時 fail-fast 檢查依賴，並對照原 SentencePiece token IDs，避免 fallback parser 改變編碼。此修正不改樣本、teacher、分數定義、精度、資源或科學 gate。

## 3. 准入後的兩臂訓練設計

以下是下一個 manifest 的約束，不是本次已啟動的梯度訓練。

1. **初始化**：NG3 `seed-3003/coverage-off/checkpoint-096`，state SHA `b61b0e54e060a49bc6fd5bbab120871fdb4aa67c31483849943c42d1d14c1e9c`，共享 trunk + MLM head；不以增密的 NG80 S 作新 default，不從零學新字典。
2. **資料**：目標先建立約 16k distinct queries，再以有效曲線決定是否擴至 64k。兩臂使用完全相同多來源 TRAIN。現有三域 reservoir 只算同域擴量；SciFact TRAIN 等新任務訊號須核對原始 split、授權、舊曝光與近重複。舊 diagnostic 不改名為新 holdout。MS MARCO／用途未清的資料單獨隔離，不混入宣称可公開發布的權重。
3. **可靠監督**：known-positive、明確 judged-negative、unjudged 三類分開。known-positive 與 unjudged 不做強制 hard-negative 全排序；明確 positive/negative 才用可靠 relevance loss。teacher 分布是軟監督，不是人類 labels。84 個人工 review cards 未有人判斷，不算已有新人工資料。
4. **完整 hybrid loss**：$s_h=0.9s_{semantic}+0.1s_{BM25}$，A/B 共用 TRAIN-only 尺度。對同一候選集，$p_T=\operatorname{softmax}(t/\tau_T)$、$p_h=\operatorname{softmax}(s_h/\tau_h)$；主要排序項 $D_{KL}(p_T\Vert p_h)$ 加共用可靠 relevance 項。兩 teacher 的 entropy/尺度校準使用相同 TRAIN 規則，避免把 logit 大小差誤認為 supervision 差。
5. **不是又一次 pair 權重實驗**：不只保留 teacher 原本同意的正向 margin；可靠人類判斷能提供糾正方向。若只有 positive-only labels，明確報告人類 supervision 仍不完整，不能靠新命名假裝補齊。
6. **深度與公平性**：共用 AdamW trunk LR 5e-6、head LR 2e-5、weight decay 0.01、clip 1.0 作起始配方；正式步數按資料/吞吐預先凍結為 2 個完整 epochs、終點選擇，不根據 diagnostic 临時延長或挑尖峰。第二 training-order seed 為有收益候選的必要確認。這次不以四個 updates 裁決機制。
7. **成本**：NNZ、feature DF、query literal work 全程跟踪。品質探索可暫時變貴，但文件 NNZ 或 literal work > parent 1.5x 的 endpoint 不自動擴規模／生產化；這是擴規模資源限制，不等於否定其品質訊號。shared cost 控制和對應梯度 parity 要在正式 manifest 中凍結；不臨時掃係數救結論。
8. **評估**：TRAIN 曲線、舊曝光診斷、獨立來源/任務族 validation 各自報告。full-corpus hybrid nDCG 為主，all-positive Recall、pure sparse、pure dense、dense+BM25、逐域結果全部保留；一個 domain 的收益不能經改任務權重擴大。真正候選再測 native bytes、decoded postings、warm p50/p95 與維護代價，不用 DF proxy 宣稱實際加速。

只有 teacher 信號、multi-source exposure manifest、text/context 可學性和資料使用邊界核實後才凍結正式訓練排程，不用無限診斷代替這些具體準備。若現有資料不足，報出具體缺口，而不是把複製／改寫舊題算成新來源。

## 4. 裁決與停止

- 第一階段 teacher 無增量：停止這個 teacher 配對，不宣判所有任務監督無解；不自動換一串 teacher 搜榜。
- B 在充分曝光、兩 seed、獨立來源下仍無可重現收益：停止同一家族的局部 teacher/loss 微調。
- 可重現品質改善只靠明显增密、無 joint frontier 改善：不推為低成本產品候選，討論下一種結構前先保留該結論。
- TRAIN 也學不會且實作／數值排錯完成：才考慮一次更大成熟 base 匹配對照。TRAIN 學會但新來源不行，不直接以放大模型繞過 transfer 問題。
- 第一階段完成不代表正式訓練或產品目標完成；所有結果經原始分數 replay，正式 endpoint 仍需獨立 review。

## 5. 文獻與成果邊界

[SPLADE-v3](https://arxiv.org/html/2403.06789v1) 支持成熟 warm-start、強 teacher 與蒸餾配方的組合，不能保證某個 RankT5 在三域有效。[RankT5](https://arxiv.org/abs/2210.10634) 是 ranking-loss teacher 的方法來源。當前復用的 [Soyoung97 權重](https://huggingface.co/Soyoung97/RankT5-3b) 沒有完整 model card；這是研究-only teacher admission，並未完成可公開發布衍生權重的授權清算，不與其他同名 reproduction 混淆。

小型可重現 source、測試、協議留主 Git 庫。大模型和完整 raw scores 置 Betty NG84 資料單元並保留原 archive；本輪不重建 PostgreSQL 索引、不碰生產代碼或服務。
