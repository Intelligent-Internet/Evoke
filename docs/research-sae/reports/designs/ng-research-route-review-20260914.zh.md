# NG 研究路線審查：停止重複診斷，驗證可用的排序學習路線

日期：2026-09-14。狀態：方向審查與待討論建議，不是新實驗執行協議。

後續狀態：使用者已批准按本思路推進；新的分階段執行權威是
[NG84 task-ranking protocol](../ng0001-ng0099/ng0084-task-ranking-execution.zh.md)。
本文保留作為當時的方向審查，不能把下文「尚未啟動」解讀為 NG84 的即時狀態。

本次重新閱讀結構性總復盤、成熟 sparse 對照、強 teacher 訓練、融合／成本閉合報告，以及 NG66 至 NG83 的關鍵分支，並核對 NG80 的實際 loss 和 NG71／NG69 的監督契約。沒有重新執行全部歷史實驗，沒有啟動訓練、讀取 locked test、改產品代碼或部署。原始失敗門檻、封存產物和曝光身分不變。

## 1. 判斷

**目前的局部探索循環應當停止；「成熟 learned-sparse + BM25 在 overall 品質與成本上勝過 dense」尚未被證明無解。**

最不值得繼續的循環是：同一小範圍監督上改 pair 權重、做局部保留、剪 posting、再用另一種描述性診斷解釋全庫排名為何沒有跟上。新命名不等於新機制。

但也沒有證據保證下一輪只要擴資料便會成功。合理的下一步不是無限續跑，而是一次有明確資源上限、足夠訓練深度、真正改變監督資訊的決策性訓練。若它仍不能推進品質／成本前沿，就停止當前模型家族的微調，不再以另一個 proxy 延長研究。

## 2. 歷史證據到底支持什麼

以下數值屬於各自報告的比較面，不能拼成跨實驗排行榜。

| 證據 | 已得到的結論 | 不能推出的結論 |
|---|---|---|
| [結構總復盤](../ii42-unified-posting-structural-diagnosis-and-reset-report.md)：signed coordinates、dense tail、balanced routing、自由 score factorization 等 | 高保真的 dense 幾何可以保留排序，但常失去倒排選擇性；自由 oracle 可表達不等於文本 encoder 可學會 | 所有 sparse 檢索都不可能勝過 dense |
| [M1904/M1905](../m1900-m1999/ii42-m1900-m1920-learned-sparse-success-failure-factor-report.md)：相同 10k rows／teacher／schedule，改用預訓練 MLM 詞彙頭 | Disjoint candidate pair accuracy 從 0.798584 到 0.869629，文件 NNZ 從 101.36 到 37.38；成熟輸出座標有強正面證據 | 這是全庫勝過 dense；或新 latent basis 永遠不可能成功 |
| [M1701](../m1700-m1799/ii42-m1701-listwise-hard-negative-curriculum-report.md)：RankT5、listwise、溫度校準、成本梯度投影 | 強排序 teacher 曾有局部增益；實際訓練只有四個 updates，選出的 step3 原生 macro nDCG/MAP/Recall/MRR 均為正，但 CUB 失敗 | 已充分訓練並否定強排序監督；或可以忽略舊 gate 直接晉級 |
| [M1951](../m1900-m1999/ii42-m1950-m1951-opensearch-one-index-closure-report.md) | 四行比較面 macro nDCG 0.455414 到 0.464823，但 FiQA 回退使當時規則不通過 | 已超過 dense，或逐域代價可以隱藏 |
| [NG66](../ng0001-ng0099/ng0066-training-duration-review.zh.md)／[NG69](../ng0001-ng0099/ng0069-final-breadth-review.zh.md) | 重複舊題的 TRAIN 提升未可靠轉移；同曝光量下，6144 個不同 query 優於 1536 個 query 重複四次，macro nDCG 差 +0.012425，三 order seeds 的報告區間為 [0.007169, 0.017956] | 多來源泛化已成立；品質改善没有增密代價 |
| [NG77](../ng0001-ng0099/ng0077-final-retention-review.zh.md)／[NG78](../ng0001-ng0099/ng0078-boundary-coverage-review.zh.md) | 所檢查 NQ TRAIN 受損 gold 的 40 個前排反超 pair 中，34 個已是可信 anchors，只有 1 個在 pool 外 | 所有 query 都不需要 mining；或 85% 是總誤差的因果歸因 |
| [NG80 S/P](../ng0001-ng0099/ng0080-sparse-endpoint-review.zh.md) | S hybrid nDCG 0.770716 到 0.786079，仍低於 dense 0.822175；文件 NNZ 4.02x、總 literal DF-work 7.57x | 沒有學到任何東西；或線上延遲真的增加 7.57 倍 |
| [NG81](../ng0001-ng0099/ng0081-fusion-progress.zh.md) | 固定校準最多只收回 S 與 dense nDCG 差距約 4.86%，不能再把主因歸結為 alpha | 一切融合形式都已被否定 |
| [NG82](../ng0001-ng0099/ng0082-progress.zh.md)／[NG83](../ng0001-ng0099/ng0083-progress.zh.md) | Utility-mask 在自己的 TRAIN 上也使排名大跌；額外 boundary weighting 未可靠優於 field；global MSE 更差時 pure nDCG 反而改善 | 只缺更多相同 utility 監督；或線性 rank256 結果就是 sparse encoder 的容量上限 |

### 已經做過，不能再包裝成新方向

- 候選覆蓋、前排反超文件是否進入 pool：NG75 至 NG78 已逐步做過。NG83 的新 operator bank 並非同一產物，若將來決策確實需要，可補差額核對；它不應再成為獨立主線或訓練前的無限關卡。
- 更強 teacher、listwise、hard negatives、BM25 residual、DF-FLOPS：歷史均有先例。新試驗必須說明改變了什麼，以及為何舊結果不能回答這個問題。
- 局部 posting 效用相加後剪整組：NG53 與 NG82/83 已有反例。動態 mask 可以是索引機制，但不能假設它能無害修復模型準確度。
- 再跑線性 probe／自由 oracle 證明底座足夠：不能回答可訓練的 text-to-posting 模型是否足夠。NG80 F/D 的弱 ridge 初始化也不是強預訓練 dense sibling 的容量對照。

## 3. 更根本的問題：壓縮教師與超過教師不是同一任務

設 teacher 分數為 $t(q,d)$，student 分數為 $t(q,d)+e(q,d)$。對 teacher 認為正向的一組比較，保留順序要求：

$$
e(q,d^{-})-e(q,d^{+}) < t(q,d^{+})-t(q,d^{-}).
$$

平均重建誤差很小，不會自動約束最接近排名邊界的誤差差值。NG82/83 的實測正好反對把平均 MSE 或少數 pair loss 當成最終排序的替代品。這個公式解釋可能的失配，並未唯一辨識當前模型的根因。

更重要的是，**即使完整複製 PPLX 分數，也只得到 PPLX，而不是新的 relevance 訊號。** BM25 的互補或 student 的歸納偏置可能讓 hybrid 超過 teacher，但並不保證。本次 NG80/81 比較面上，TRAIN 校準 dense+BM25 的 nDCG 是 0.820317，pure dense 是 0.822175；因此「完美模仿 dense，再加既定 BM25」在該配置下也不是自動達標的方法。這不是所有融合的數學上界。

實作上的區別也很具體：

- [NG80 `protective_loss`](../../../../scripts/ng80_training.py) 只對 `teacher_positive_margin > 0` 的關係給權重。它是 teacher 同意時的正例保護，不是充分驗證過的、可依可靠標註糾正 teacher 錯誤的任務監督。
- 不能據此聲稱歷史完全沒用 gold：[NG69](../ng0001-ng0099/ng0069-matched-exposure-breadth-plan.zh.md) 已使用 `0.5 gold + 0.5 teacher` CE；[NG71](../ng0001-ng0099/ng0071-global-boundary-ranking-plan.zh.md) 也設計了 judged-negative 分支。問題是監督可信程度、來源範圍與實際覆蓋，不能用另一個混合係數代替新的資訊。
- [NG83 來源盤點](../ng0001-ng0099/ng0083-supervision-readiness.zh.md) 仍是 FEVER／HotpotQA／NQ 三域。準備的 84 題盲化人工複核尚無新人工判斷。未標為正例不等於人工判定不相關；更多 teacher 分數不是更多人類監督。

**最值得驗證的突破口是：讓成熟 sparse 模型學到比單一 PPLX 幾何更有用的任務排序資訊，而不是把保真、稀疏化、少數 pair 保留不斷互相補救。** 這是優先假說，不是已證明的解法；也不是本輪才首次提出的觀念。欠缺的是完成一次足以裁決它的訓練。

### 輸出基座與 DF 的取捨

先保留成熟詞彙稀疏頭，不強求重新發明 latent SAE 字典。M1904/M1905 提供了比「隨機字典更大應該更好」強得多的對照證據。名稱不是產品能力，現在也不是把 trunk 和輸出字典一起從零訓練的好時機。

DF 不能不管，但也不能成為無條件刪除規則。Literal posting 工作量的期望為：

$$
C_{\mathrm{literal}} = \sum_j \Pr[z_j(q)\ne 0]\,\mathrm{DF}_j.
$$

它不是文件 NNZ，更不是 WAND 的實際成本。S 平均幾乎觸及全部文件，原 hybrid 的 union touch 也已超過 97%；必須同時看重複 posting 貢獻、解碼／pruning 和整體原生延遲。保留一個品質有效的模型後，才能判斷哪些長 posting 值得付費，不能先假設高 DF 都是噪聲。

## 4. 建議下一步：一次決策性訓練，而不是下一串診斷

這是待討論的研究設計，尚未凍結資料、步數、資源或啟動命令。

### 唯一主要問題

**在相同成熟 sparse 基座、資料、輸出／成本策略和訓練預算下，加入經核實更有用的任務排序 teacher，能否比 PPLX 主導的蒸餾產生穩定、可泛化的完整 hybrid 改善？**

建議固定兩臂，不做十幾種 loss 搜索：

| 項目 | A：幾何 teacher 控制 | B：任務排序 teacher |
|---|---|---|
| 初始化 | 同一成熟 parent；不以增密 S 默認取代 | 相同 |
| 資料 | 同一批有來源紀錄的多來源 TRAIN，可靠人工訊號兩臂共用 | 相同 |
| 未充分判定關係的 teacher | PPLX score distribution | 通過同資料預檢的強排序 teacher distribution |
| 排名與成本實作 | 共用、事前固定；直接作用於完整 hybrid 分數 | 相同 |
| 真正改變 | 對照 | teacher 提供的排序資訊，而非僅多算幾組 pair |

強 teacher 不自動等於正確標籤。先復用歷史 RankT5 等候選作預檢，只在 TRAIN 上確認資訊增益及不確定性；不能把 unknown 當硬負例，不能讓 teacher 覆蓋明確可靠的相反判斷。若此預檢沒有增量資訊，就不應啟動 B。這是訓練的必要准入，不再報成另一輪突破。

兩臂都擴來源，故 A/B 可辨識 teacher 差異，不能另外聲稱隔離證明「多來源本身」的因果效果。相對舊模型的改變包含資料和訓練過程，必須如實標示。

### 規模、訓練與評估

1. 可先以約 16k 到 64k 條真正不同的監督建立階段曲線，數字只是預算提案；先核對可用來源、發布用途、去重和吞吐。不是把 16k 舊 query 改寫成 64k，也不是直接承諾十萬級訓練。歷史已有 100k 級其他配方，不能說項目從未做過大資料。
2. 固定 optimizer、尺度校準、資料與評估排程，留出足夠重複曝光和收斂觀察。不要再用三四個 updates 或單一新 head 的一次遍歷，推論整個訓練家族無效。也不在評估後臨時延長失敗 checkpoint 或挑最佳短暫尖峰。
3. 主要指標是預先固定任務權重的 full-corpus hybrid nDCG@10，對照 pure dense、dense+BM25 和原 parent；all-positive Recall、分域損失及原生成本獨立披露。小規模階段可以尚未超過 dense，但必須看到可重現的前沿改善，才擴資料。
4. 新來源／topic／query-family 的 validation 用於研究選型；最終未曝光 test 另保留。舊 1536 題不能改名為新 holdout。第二 seed 及獨立驗證確認後才談一般化，不因每域必須勝出而否決 overall 收益。
5. 成本從起點記錄並有事前上限。允許品質診斷暫時變貴，不允許靠 4x NNZ／7x literal work 自動過關。只有被預先允許的候選才做實際索引 bytes、warm p50/p95、解碼和維護測量；不以 proxy 直接聲稱更快。

不在這輪同時更換模型家族、做新字典、引入路由、多向量 rerank、重設動態 mask。模型真正顯示 joint quality/cost 潛力後，才值得投入下一種結構。

### 何時停止，何時換結構

- 若 B 在有足夠訓練、獨立 query/source 和重複 seed 後仍無穩定收益：停止「再換 teacher／pair 權重」的局部延伸。
- 若有品質收益卻始終只能用明顯增密換取，且事前指定的成本比較沒有前沿改善：停止將當前小模型／單稀疏內積家族視為主要突破候選。
- 若可靠標註和可見文本下 TRAIN 都學不好，且优化排錯完成：才做一次更大成熟 base 的匹配對照，而不是再跑弱 probe 宣稱容量不足。
- 若 TRAIN 可以學會但獨立來源仍無收益：不能以換大模型跳過 supervision／transfer 問題。
- 停止某個家族是研究資源決策，不是證明所有 sparse 方法無解。若最後要改為 posting 召回加小型 dense 修正，需單獨討論產品邊界；[M1972/M1973](../m1900-m1999/ii42-m1972-m1973-native-compact-dense-closure-report.md) 已測過類似方案，不能再稱為新發現或偷換成純 posting 成果。

## 5. 文獻校核與研究規則修正

[Granite report §6](https://arxiv.org/html/2502.20204#S6) 使用 retrieval-oriented pretraining 和 dense-to-sparse score-distribution 蒸餾，並明確報告單純 FLOPS 不足。它支持成熟底座與整體訓練配方，不保證我們的模型可以超過 PPLX。

[SPLADE-v3](https://arxiv.org/html/2403.06789v1) 的改善來自成熟 warm-start、較強排序 teacher 與蒸餾設計；增加 negatives 的收益主要在 in-domain，不能當作泛化的充分條件。其 aggregate 評估也容許部分資料集回退。這些方法並非本輪發明，M1701 已有縮小版先例。

[DF-FLOPS](https://arxiv.org/html/2505.15070v1) 研究的主要設定是 SPLADE-Doc、binary lexical query，不可把速度數字移植到雙端 learned query/document 模型。歷史已有 DF-FLOPS 失敗紀錄，不再直接重掃係數。

最後需要修正的是研究判決，而不只是 loss：

- **區分數值／安全失敗、探索階段未過 gate、真正沒有科學收益、沒有產品價值。** 它們不是同義詞。
- M1701 的 CUB／Spearman 或 M1951 的逐域條件有當時的保護目的，但不應成為「排序監督在科學上已無效」的證據。原始門檻和失敗結果保留，新的產品目標只影響未來規則。
- 未跨過小診斷上的所有 proxy，不能無限阻止一個資訊不同、預算可控的正式訓練；反過來，增加報告、測量和模型編號也不是研究進展。
- 本次撤回把 NG83 後的候選覆蓋核對當作下一條主線。後續只保留能直接改變訓練／停止決策的必要測量。

結論：**值得繼續的是一次有期限的任務排序學習驗證，不值得繼續的是当前這種微調與診斷循環。沒有證據承諾成功，也沒有證據宣判整條 sparse-hybrid 路線無解。**
