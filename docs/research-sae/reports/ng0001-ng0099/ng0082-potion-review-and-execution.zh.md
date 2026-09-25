# NG-0082: Potion review and two bounded causal diagnostics

日期：2026-09-14。這是執行前協議；凍結後不得修改 run 中的副本。成果另寫 progress/report。

## 決策背景

目標仍是 **BM25+SAE 的 overall quality 超過 PPLX dense，且成本可接受**；不要求 SAE 單路或每個 domain 都贏。NG81 已完成，並不是還在訓練：dense nDCG@10 0.822175、S historical hybrid 0.786079、S z-score hybrid 0.787832。後兩者差值的 paired CI 跨零；S 相對初始模型的 document NNZ 為 4.0232 倍、semantic DF work 為 11.121 倍，hybrid total DF work 為 7.571 倍。這批 1,536 題是已曝光、三域均衡的 diagnostic，不能代表 BEIR15 或獨立泛化。

因此停止擴大 scalar fusion 搜尋，也不立即放大目前的 S 訓練。下一步分開判斷：(A) 新增的高 DF support 是否具有排序價值；(B) 在不引入 sparse encoder 誤差之前，有限維度能否保存 teacher 的排序幾何。

## Potion: 有效機制與不可直接移植的部分

[potion-multilingual-128M 模型卡](https://huggingface.co/minishlab/potion-multilingual-128M)描述 BGE-M3 蒸餾、2M C4 passages、101 語言與溫度平滑採樣，輸出 256 維 static embedding。它是 token lookup 與平均，不是 128M transformer，也不是稀疏倒排模型。卡中的 retrieval 分數 37.86 高於該表 LaBSE 的 33.17，但低於 static-similarity-mrl 的 41.21；不能從整體平均分宣稱檢索全面領先，更不能外推超過我們的 PPLX teacher。

[Model2Vec 說明](https://minish.ai/packages/model2vec/introduction/)中的初始化是逐 token 取得 teacher 表示，再以 PCA 與可選詞頻權重處理，推論只查表並平均。這消除 contextual attention 的成本，也丟失順序、否定作用域與多義詞依賴上下文的交互。新增詞彙可吸收常見片語，但不能一般性替代 compositional context。encoder 加速不等於倒排掃描加速，dense 維度也不等於 posting NNZ。

[Tokenlearn 2.0](https://minish.ai/blog/2025-05-31-tokenlearn-release/)採用無標註文本的 teacher mean targets、PCA 及 MSE，並分開參數化 token 向量和權重；它不再依賴舊版 post-training PCA/SIF。這啟發的是「先保真，再做檢索適配」和 token 權重的可學習性，不是直接抄一個 loss 或 batch size。權重與方向的因子化本身沒有造成稀疏性，也沒有使 DF 成本自動變低。

版本核對不能省略：2025-05-28 的 [train.py](https://github.com/MinishLab/tokenlearn/blob/14dd7d50158a98b822727812e7669503c0dde2e0/tokenlearn/train.py) 和 [pretrain.py](https://github.com/MinishLab/tokenlearn/blob/14dd7d50158a98b822727812e7669503c0dde2e0/tokenlearn/pretrain.py)確實是 MSE。其 target PCA 在 validation split 前擬合；`state_dict()` 沒有 clone 為不可變 best checkpoint；訓練含 affine output head，匯出卻只匯出加權 embedding。後者不天然保存訓練 head 的內積幾何。這些是該 source snapshot 的可重現性風險，不是已證明發布權重失敗。現在 Tokenlearn `fcb6ed7` 呼叫 Model2Vec similarity trainer；[Model2Vec `0ef8c28` 的實作](https://github.com/MinishLab/model2vec/blob/0ef8c28ecb7e30cacd821e65b4f71e36d3ad5928/model2vec/train/similarity.py)是 cosine loss，不能混稱為 2025 年模型的實際訓練配方。

[potion-retrieval-32M](https://huggingface.co/minishlab/potion-retrieval-32M)是另一個英文模型：在成熟 static 底座上，以 13 個資料集、MultipleNegativesRankingLoss 與 MatryoshkaLoss 做檢索訓練。這支持分階段、多來源 supervision 的方向，但其卡片 retrieval 35.06 仍低於同表 MiniLM 的 42.92。它用 NanoBEIR 選 checkpoint，該集合不能再被當成 untouched final test；我們也不能把 in-batch 未標註項直接視為可靠負例。Matryoshka 的 dense prefix 不等於可選擇性的 sparse support。

## 舊實驗反證與本輪新增資訊

- [M401](../m0400-m0499/ii42-m401-structural-dense-tail-distillation-report.md)：joint-PCA tail 的確定性表示較強，學習映射卻退化；本輪不再測 deeper MLP 或重跑同一 reconstruction loss，而是直接比較原 teacher 幾何的不同有限秩近似，先移除 encoder 混雜因素。
- [M409](../m0400-m0499/ii42-m409-raw-text-posting-distillation-report.md)：hash word/bigram/char 特徵的 raw-text 蒸餾失敗。這不否定語義初始化的 Potion，但不支持重開相同 hash 詞袋模型。
- NG53 的 learned mask：TRAIN 單獨刪除 loss delta 合計 -0.012885，真正 joint delta +0.008373，非加性殘差 +0.021258；所以 **individual utility 不是可相加的質量保證**。
- NG80 的 PPLX teacher 已是 masked mean + L2；不能再把 mean pooling 當新方法。NG80 supervision audit 量測 query spectrum，但未測以下 query-weighted asymmetric score approximation 的全 corpus 排序。
- [NG81 結果](ng0081-fusion-progress.zh.md)：校準未消除 S 的主要 quality gap，也未消除高 DF。不能再將 scalar calibration 當主突破口。

## A: 排序 utility 與高 DF 的聯合 mask 診斷

固定 S 與 BM25，不更新 encoder、不改 RMS denominator、不重選 alpha。只在 **384 個既有 TRAIN calibration query** 設計 mask。對每題 teacher 全 corpus 排序，固定六組 0-based 位置 `(0,9), (9,10), (9,49), (49,99), (99,100), (99,199)`；這些是 teacher 偏好，不是假裝人類標註的 relevance。所有排序使用 score 降序、document ID 升序 tie break。

設現有 hybrid margin 為 `m`，特徵 j 的 semantic margin 貢獻為 `c_j`。TRAIN 上固定 `tau = max(median(abs(m)), 1e-8)`，單特徵移除 utility：

$$
u_j = E[\operatorname{softplus}(-(m-c_j)/\tau)
        -\operatorname{softplus}(-m/\tau)].
$$

精確 logical work 為 `w_j = DF_j * Pr_TRAIN(q_j != 0)`，而非 weight 大小。比較按 `u_j / w_j` 由低至高刪除，與按 DF 由高至低刪除；相同 TRAIN semantic work target 分別為 S 的 50%、初始模型的 semantic work。刪除至低於 target，報告實際 undershoot，**不假稱離散特徵可完全等成本**。TRAIN 沒有出現的特徵保留並單獨報告其 diagnostic work。mask 同時定義 query/document 的同一 feature support；本輪以 query mask 等價計分，不改實體索引。

另外兩個無 fitting 的 causal control：只去掉 `DF/N >= 0.90` 與 `>=0.50` 的 semantic features。所有六個 masks 和未變動 S 都完整執行 1,536 diagnostic queries、233,009 documents；BM25 不變。對所有 mask，重算真正 joint TRAIN pair loss 與 additive prediction 的差異，不用它繼續 retune。量測 nDCG@10、Recall@100、per-domain、paired bootstrap、query/document NNZ、literal semantic/lexical DF work、被觸達 union。成本是 traversal proxy，不是 native p95。

預先判定：相對未改 S，nDCG CI 下界不低於 -0.005、Recall CI 下界不低於 -0.005，且 diagnostic total DF work <= 0.70 倍，才值得下一輪做原生索引與未曝光驗證；不是本輪直接推廣。若 utility 不勝 DF，停止稱其為 learned-selectivity 進展。若高 DF ablation 顯著傷害排序，將工作移往 **保留貢獻但改變表示方式**，不是再提高刪除力度。

## B: 表示方差與 query-weighted score 幾何

使用 cached PPLX query/document 向量，不再載入模型。TRAIN 的 12,288 題拟合 query second moment；document covariance 使用 seed 82082 固定抽出的 65,536 個 corpus documents（unlabeled transductive）。diagnostic 題不參與 fitting。比較固定 k=128、256，不做維度搜尋：

1. ordinary centered document PCA，query 不中心化、不重新 L2；document centering 只產生每題常數，不改完整維度排名。
2. query-weighted asymmetric projection。令 `Cq=E[qq^T]+eps*I`，eps 固定為 `1e-4*trace(E[qq^T])/dimension`，`A=Cq^(1/2)`。取 `A Cd A` 的前 k 個 eigenvectors U，編碼：

$$
z_d=(d-\mu)AU,\qquad z_q=qA^{-1}U,
\qquad \hat{s}(q,d)=z_qz_d^T.
$$

這最小化 regularized query 分佈下、中心化 document score 的平方誤差，不保證 nDCG 最優。full rank 時 `A U U^T A^{-1}=I`，所以有可測的精確退化控制。亦以獨立 dense operator 重算 sampled scores；不得把高 explained variance 當排序保真。

每個 profile 對完整 corpus 計分，同時報 pure teacher approximation 與固定 `0.9*zscore(semantic)+0.1*zscore(BM25)` hybrid；後者沿用 NG81 的 **昂貴離線診斷**，不是 serving 提案。對照完整 teacher 的排名、頭部 overlap、margin error 與 human qrels metrics。若 relative teacher nDCG/Recall CI 下界均 >= -0.005，才形成低秩 target 候選；若 query-weighted 不勝 PCA，不升級該方向。若仍失真，下一轮應測 ranking-boundary supervision，而不是僅擴大 MSE 語料。

## 執行與安全

兩條 CPU lane 在 Lambda2 並行，資料已在該機，避免模型重載和跨機大搬運。每 lane 4 threads / RSS <=16 GiB / <=5,400 秒，整機 available RAM >24 GiB、disk free >40 GiB；第一批完整排名作 canary，超出原限額即保留失敗並停，不放寬。持有舊 `.ng80-cpu.lock` 的 shared lock 與各 lane 的 exclusive lock；與既有 exclusive CPU 任務互斥，兩條新 lane 可共存。GPU 不用、不預占、不啟動多 GPU。

source、實際依賴、選題與本協議在新 `NG-0082/diagnostics-v1` freeze；先跑 unit tests，執行前後驗 SHA。每 phase 使用既有 process-group guard，保存 start/exit/complete、ClearML offline actual-start/closed receipts。不得宣稱已同步 ClearML server。診斷 ranking 保存所有已標註 positive ranks、top100，另算 direct rank 交叉驗證。

完成後兩路一起 review：先檢查證據與 numeric contracts，再比較 quality/cost，不對 diagnostic winner 直接擴大。不碰 locked test、不自動恢復輪詢、不部署、不推送模型。若要 scale，下一協議先凍結 source/topic-disjoint queries 與等更新量的 two-stage vs direct-ranking 對照。

## 有條件的下一輪訓練，不在本輪偷跑

- B 有保真 target：在成熟 NG3 底座上增加 **可匯出並保持同一 serving score 的 teacher geometry auxiliary target**；同更新量比較 representation warm-up→ranking 與直接 ranking，不複製 Potion 被丟棄的 output head。
- A 顯示高 DF 仍有用：不把其頻率直接當垃圾，研究可保留 margin 的共用/殘差分解與 block pruning；所有效能結論必須走 native query path。
- A 能低成本保住排序：才把支援度 gate 帶進訓練，loss 必須評估 joint mask 而非個別 utility 的和；教師排序與 human supervision 分開記帳。
- 任一路失敗：保留反證，不能以更多 epoch、synthetic variants 或更大 query batch 覆蓋問題。大量、多來源人類 query supervision 仍重要，但在固定底座的 head/boundary teacher agreement 先有改善後才擴大。
