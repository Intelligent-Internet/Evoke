# NG-0085：成熟 sparse student 的固定預算 teacher A/B

日期：2026-09-16。使用者在 NG84 完成後批准向前推進。本輪是新的研究授權，不修改 NG84 `inconclusive_do_not_auto_launch` 的歷史裁決，不宣稱 RankT5 已通過原 gate。

## 問題與停止條件

在相同成熟 NG3 基座、相同可見文本、同一候選集合、相同 hybrid 分數及充分曝光下，PPLX 與 RankT5 的任務排序訊號，究竟能否讓 BM25+SAE 的整體前排品質出現可重現差異？本輪不重開 teacher 選型、權重掃描、mask、routing 或新字典實驗。即使 B 不贏，也不宣判所有 sparse 路線無解。

這是一次訓練機制對照，不是直接產品晉級。NG84 的 FEVER 負向結果及不完整 qrels 均保留；不刪不利問題，不用 teacher 事後分數改標籤。

## 凍結的資料

- TRAIN：NG80 原 fit 的 FEVER / HotpotQA / NQ 各 4,096，另從已溯源的 canonical FiQA TRAIN 按固定 hash 選 4,096，共 16,384 個問題。
- Validation：原 NG80 diagnostic 每域 512，加 FiQA TRAIN 的不重疊 512，共 2,048。這是 gradient-excluded research validation，**不是項目未曝光 holdout**。不讀 NG67 locked-test 或 FiQA test qrels，也不宣称已完成全球歷史近重複稽核。
- 背景：NG67 的 233,009 篇固定文件，加 FiQA canonical corpus 中有文字的 57,600 篇，合計 290,609。這是完整的本輪固定背景，**不是完整 BEIR 三域 corpus**，不能跨背景直接比較歷史絕對數值。
- 資料完整性 v2：首個 preparation attempt 在遇到空白文件時停止，沒有任何 teacher inference 或 student update。逐筆核實原 FiQA corpus 有 38 篇空白文件，35 個 TRAIN positive memberships、34 個問題涉及無文字正例。新 attempt 在任何模型評分前隔離這些文件及全部受影響問題，保留原 qrels 不動；從其餘 5,466 個 TRAIN 問題按原 hash 順序補足 fit/validation。排除清單逐項保存，不把沒有證據的正例重新標為負例。這是研究資料完整性子集，不冒稱完整 canonical FiQA 評估。
- 不合併不同 source ID，即使文本近似；保留全部 known-positive membership。FiQA 使用原始 TRAIN qrels，舊三域保留 NG67/RLHN 來源標註，不將它們重命名為新人工審核。
- 兩臂及 student 使用完全相同的 literal source prefix，不經 tokenizer decode 重寫。Query 同時滿足 student/PPLX 64 tokens、RankT5 96；document 同時滿足 student/PPLX 256、RankT5 384。RankT5 每個 combined prompt 再確認不超過 512，禁止靜默二次截斷。
- 不回用舊長上下文 PPLX/RankT5 分數。所有 teacher targets 由上述文字重新編碼。
- 舊三域 fit 保留 NG83 bank IDs；FiQA 與 validation 使用本輪背景的 dense top32、BM25 top32、全部 known positives 與 16 個固定背景文件的 union。ID 同分排序固定，兩臂完全一致。沒有 RankT5 專屬 mining。
- BM25 重新在本輪相同可見 corpus 建立 vocabulary/DF/RMS，使用已驗證的 II42 native tokenizer 和原 NG8 BM25 公式；不讓 FiQA 因沿用舊 vocabulary 而丟失 OOV。

## 模型、Loss、成本取捨

初始化固定 NG3 `seed-3003/coverage-off/checkpoint-096`，共享 trunk + MLM head，完整 nonnegative sparse readout；state SHA `b61b0e54e060a49bc6fd5bbab120871fdb4aa67c31483849943c42d1d14c1e9c`。不以 NG80 增密 endpoint 作新 default，不從零训练。

A 使用已釘住的 PPLX normalized mean-pooled dot scores；B 使用 `Soyoung97/RankT5-3b` revision `40e2b98fcbc6d60457c88508bd775fcb8395a5b0` 的 `<extra_id_10>` 第一 decoder step logit。B 是研究-only，衍生權重的公開發布權利仍待釐清。

同一 query 候選上的目標為：

$$p_T=\operatorname{softmax}\left(\frac{t-\operatorname{mean}(t)}{\max(\operatorname{std}(t),10^{-6})}\right).$$

兩臂使用相同 affine calibration 規則，不根據 validation 調 entropy。Student 完整分數 $s_h=0.9s_{SAE}+0.1s_{BM25}$，保留 NG71 的固定 semantic RMS 歸一化，lexical 使用本輪 corpus RMS；loss 為上述 $p_T$ 對 $s_h/\operatorname{stopgrad}(\max(\operatorname{std}(s_h),10^{-6}))$ 的 cross entropy，與 KL 的梯度相同。截斷比例／尺度都要留 receipt。

目前資料沒有經確認的 explicit judged-negative，因此可靠人類相反判斷 loss 的實際權重為 0；不把 unjudged 全部設為人工負例。這是明確的監督限制，而不是宣稱完整多元 human supervision 已解決。Known positives 仍必須保留在兩臂候選與評估分母。

為分離品質訊號與既有 cost penalty 的抑制，本輪兩臂 cost regularizer 同為 0，不硬裁 posting、不掃 FLOPS 係數。全程記錄 NNZ，正式 endpoint 再算 DF/literal work。相較 parent，document NNZ 或 literal work >1.5x 不自動擴規模或產品化，即使品質上升。這個品質探索不構成低成本成功；真正候選仍要通過 native bytes、decoded postings、warm p50/p95 及維護代價實測。

## 曝光、測試與執行

- A/B 各兩個 training-order seeds：85001、85002。每個 update 各域一題，累積 4 題；兩完整 epochs，每臂每 seed 8,192 updates、32,768 query exposures。取固定終點，不選 validation 尖峰。
- FP32，TF32 關閉；AdamW trunk LR 5e-6、head LR 2e-5、weight decay .01、clip 1。Query microbatch 1，document microbatch 4，以現有 exact VJP replay 控制記憶體。
- 正式 fit 前：四域真實資料的 full graph / VJP gradient parity、16 個 disposable updates、實際 model/optimizer 保存與重載一致性。Canary 一律丟棄，正式 fit 重新由 NG3 初始化；不能用 canary 的 loss 裁決路線。
- 每 1,024 updates 保存並重載精確 optimizer moments、model state、兩種 role outputs。失敗不覆盖，保留原 attempt。兩 seed/兩臂 order 及 targets 有 SHA manifest。
- Spark 可用記憶體不足時用 Lambda2 空卡，每工作單卡，明確 CUDA 綁定及共享 lock，不使用跨 NVLink 對或四卡 DDP，不驅逐其他工作。
- 每 preparation/dense/bank/RankT5 shard 最長 8 小時；每 student run 最長 12 小時。CPU owned RSS 48 GiB、GPU job owned RSS 24 GiB、GPU allocation 22 GiB；host available memory >20 GiB、disk free >40 GiB。開跑要求更高餘量。超限只終止 owned process group，保留產物。
- ClearML 實際 start/close；目前採明示 offline tracking，不能說已上傳。新閉環可依已封存輸入串接固定 phase，但不恢復舊自動輪詢、不自動開下一輪研究。

## 終點評估

Trainer 的 complete 只代表訓練完成。固定 endpoint 必須在本輪 290,609 文件背景重編碼／排序，報告 parent hybrid、A/B hybrid、pure sparse、pure dense、dense+BM25，逐域及 macro nDCG@10、all-positive Recall@100。用同 query paired bootstrap，同時披露兩 seed 差異與 validation 既有曝光；不能拿 panel loss、pair count 或 Recall 提升代替前排品質。

真正的「超過 dense 並降成本」結論還需要独立來源/任務族確認與 native query path 成本；本輪不能直接證明產品目標。A/B 都沒有可重現差異時關閉此固定 teacher 配對的局部微調，不換另一組權重延長。

## 延續性

方法來自 [NG84 協議](ng0084-task-ranking-execution.zh.md)、[NG84 終點](ng0084-progress.zh.md)及[全路線回顧](../designs/ng-research-route-review-20260914.zh.md)。强 teacher、成熟 warm-start 的依據與界線已在 NG84 的 SPLADE-v3 / RankT5 引用中整理；本輪不宣稱新論文已證明本 loss 或 teacher 必定有效。

小型 source、測試及報告留主庫，大文件留 NG-0085 資料單元。此協議不授權重建或修改 production PostgreSQL。
