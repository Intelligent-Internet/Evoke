# 從局部排序微調到 Teacher 關係蒸餾

## 決策摘要

本備忘錄重新審視 NG79 之後的訓練方向。它是研究提案，不是新 NG 執行協議，不授權啟動訓練、使用 locked test、放寬歷史門檻或更新生產模型。證據截點為 2026-09-13；NG79 最終數值取自本次會話直接核對的 lambda2 已完成 endpoint review，完整歸檔收尾與科學結果分開處理。

最值得驗證的主線是：**以成熟 sparse 模型為 student，把廣泛、任務多樣的 query 對真實文件的 teacher 分數關係當作能力保留監督，再用可追溯的 relevance judgments 修正 teacher，最後在品質邊界內優化 posting 成本。** 這不是同時重訓語言模型、字典、路由器和索引。

「更多問題能探測更多排序幾何」有明確數學動機，但尚未被本地實驗證明為主因。NG69/79 測試了同一來源分布內的 distinct-query 數量，沒有充分測試任務多樣性、跨來源資料、完整 teacher 關係監督或新的人類 graded judgments。失敗不能外推成這些路線都無效；反過來，也不能先宣布資料量就是唯一瓶頸。

## 1. 我們究竟在訓練什麼

目前 NG 主線以成熟 NG3 checkpoint096 為初始化，使用約 49.65M 參數的 Transformer 與 MLM vocabulary sparse head，PPLX 為凍結 teacher。它不是從零學習的新 latent SAE。基座 fine-tuning 與 teacher distillation 是兩個獨立選擇，不是二選一。[基座及計分契約](../ng0001-ng0099/ng0071-global-boundary-ranking-plan.zh.md)

最近的訓練來源主要是 FEVER、HotpotQA、NQ 的 BGE/RLHN 衍生資料。NG69 選取 6,144 個 TRAIN query；NG79 在其中比較 384 個 query 各四次與 1,536 個不同 query 各一次。背景為同一 233,009 文件集合。這不是完整 BEIR15 訓練，也不是十五個 benchmark 全部得到同等監督。[NG69](../ng0001-ng0099/ng0069-matched-exposure-breadth-plan.zh.md)、[NG79](../ng0001-ng0099/ng0079-query-breadth-plan.zh.md)

歷史 BEIR15/MTEB10 是另一組凍結評估面，不應與近輪三域結果拼成同一排行榜。Benchmark 是量尺，不應成為產生題目、教師標註和反覆調參的唯一來源。來源 corpus 相同、query 不同的評估，與新任務、新文件或新領域泛化亦不同。

既有正例混合原始來源和後續新增標註。NG70 查得 NQ 約 43.5% 的正例相對原始資料為新增，但這不是誤標率；原始正例也不是本輪重新人工認證的判斷。NG68 準備了 84 題盲化人工複核，尚未取得新人工標註。[NG68](../ng0001-ng0099/ng0068-supervision-and-ranking-review.zh.md)、[NG70](../ng0001-ng0099/ng0070-supervision-provenance-review.zh.md)

## 2. 歷史證據排除了什麼，又留下什麼

| 證據 | 支持的結論 | 不支持的延伸 |
| --- | --- | --- |
| M360–M601、M1541/M1630 的 dense-faithful 表示 | 相近 dense 排名可由 posting 代數表達；問題包括選擇性和訪問成本 | dense 幾何不能保留，或保留後一定低成本 |
| M1904/M1905 同條件輸出 basis 對照 | 成熟 MLM basis 比當時新 SAE basis 更有利於品質與頻率分布 | 所有 latent SAE 都不可能成功 |
| NG43/47 activation、support、impact 對照 | 模仿輸出 activation、恢復 support 或權重之一，都不等於保住搜尋能力 | 再加一個 activation MSE 就能解決泛化 |
| NG60/61 全庫 oracle 診斷 | 局部可分、局部 margin 修復，不保證完整語料上沒有新競爭者 | 在 candidate pool 學好就是全庫學好 |
| NG66 的三個順序種子 | 重複同一批 query 的 TRAIN 收益遠大於 exposed DEV 收益 | 模型完全學不動，或單憑此可證明容量足夠 |
| NG69 三種順序的廣度試驗 | 同來源增加 distinct queries 有正向 new-DEV 訊號，伴隨成本膨脹 | 任務多樣性已測過，或更多資料保證超 dense |
| NG72–78 | 前排損害不只來自 support 消失；保留 anchor、改 uniform weights、加 K 都不能當通用修復 | 已定位唯一有害梯度／domain，或再做同類微調就會好 |
| NG79 | 固定工作量下，多 query 的點估計較好，仍有 NQ 傷害與 NNZ 膨脹 | 多樣監督路線已被否證，或新模型已通過門檻 |

來源：[早期結構總結](../ii42-unified-posting-structural-diagnosis-and-reset-report.md)、[M1900–1920 對照](../m1900-m1999/ii42-m1900-m1920-learned-sparse-success-failure-factor-report.md)、[NG71 歷史交叉核對](../ng0001-ng0099/ng0071-global-boundary-ranking-plan.zh.md)、[NG66](../ng0001-ng0099/ng0066-training-duration-review.zh.md)、[NG69 最終結果](../ng0001-ng0099/ng0069-final-breadth-review.zh.md)、[NG72](../ng0001-ng0099/ng0072-retention-diagnosis-review.zh.md)、[NG78](../ng0001-ng0099/ng0078-boundary-coverage-review.zh.md)。另直接重讀外部凍結 NG43、NG47、NG48 history-overlap、NG61 原報告，未重新執行其實驗。

### NG79 的最新觀測

訓練與九階段 endpoint graph 正常結束，最終 reviewer 的 execution audit 通過，scientific gate 不通過。以下是 384 個已反覆觀察、但未用於梯度的 TRAIN_SENTINEL；不是獨立 holdout。數值均為完整 BM25 + semantic hybrid，dense 為同背景基準。

| 模型 | Macro nDCG@10 | Macro Recall@100 | 語義 document NNZ / initial | Sentinel query-DF / initial |
| --- | ---: | ---: | ---: | ---: |
| Initial hybrid | 0.765559 | 0.965647 | 1.000000 | 1.000000 |
| R384：384 query 重複四次 | 0.768857 | 0.972895 | 0.887458 | 0.290751 |
| B384：1,536 query 各一次 | 0.777067 | 0.970616 | 1.584837 | 0.567964 |
| PPLX dense | 0.828470 | 0.978906 | 不適用 | 不適用 |

B-R nDCG 的 paired 95% 區間為 [-0.001322, 0.017854]；B-dense 為 [-0.071139, -0.031704]。B 的 NQ nDCG 從 initial 0.651603 降至 0.602289，而 Recall 從 0.972331 至 0.973958。R 在直接訓練的 Pilot 上 nDCG 0.842807，高於同組 dense 0.810350，但 Sentinel 沒有保住這個優勢。

這支持「能擬合部分訓練排序，卻未可靠保留跨 query 的前排判別」的診斷方向；不是對 overfitting、資料不足或容量的唯一因果證明。B 的文件 posting 增加 58.48%，查詢 DF 代理下降 43.20%，也說明兩種成本維度不能混為一談。沒有新的 native latency 結果。

最新數值來源為外部 `NG-0079/terminal384-endpoints-v1/review/results.json`，SHA256 `e766206ddec12af42257071329e76cfcb1b8c44a67e98dede3a91be942750a57`；endpoint input SHA256 `b95b74c0e7cedd4343c7b870c2b92e8ff32a39157fba6e6df590172e84158a94`。完整本機鏡像狀態不能由這份數值核對推定。

## 3. 「更多監督」至少有四種不同含義

| 軸 | 真正增加的資訊 | 不能用什麼冒充 |
| --- | --- | --- |
| 題目數量 | 更多不同 query 的 teacher 觀察 | 相同問題反覆曝光 |
| 任務／語義方向 | 實體、條件、關係、證據角色等不同判別需求 | 換字、同義改寫或單純換 domain 名稱 |
| 每題的關係覆蓋 | 多份文件的相對分數、近鄰／邊界／背景、跨 query 共享文件 | 只有一個正例和幾個容易負例 |
| 判斷可靠度 | 可追溯 graded relevance、支持片段、爭議裁定 | 把 teacher 分數或生成文件當成人類 gold |

NG79 主要改變第一軸；現在的提議實際上涉及第二、第三軸，第四軸則是超越 teacher 的重要補充。這幾者不應一起改完後，只根據新模型變好就宣稱某一種有效。

### 當前 loss 的資訊通道

實際 [`loss_and_gradient`](../../../../scripts/ng71_ranking.py) 只遍歷已知 positive 與 nonpositive；對未判負例，teacher 若不偏好 positive 就標為 unresolved、不產生該 pair 梯度。positive-positive、nonpositive-nonpositive 的 teacher 關係沒有直接 loss。

這是合法的 qrel-protective ranking objective，不是程式錯誤，也不是完整 teacher distillation。部分未直接監督的關係可能由共同 anchor margin 間接推得，但在缺邊、soft target 飽和、共享參數受限和有限更新下，不能假定已完整傳遞。

尤其不能把「没有顯式 nonpositive-nonpositive loss」單獨當成證明：若對同一 anchor 的全部 teacher margins 都被精確保留，其餘 pair margins 可由相減恢复。真正要量測的是有效連通性、被屏蔽的關係、margin 精度和不同 query 的覆蓋，而不是只比較 pair 數量。新方案不能用全配對數量誇大新增資訊。

更早的 NG66/69 曾在小候選池蒸餾混合 teacher distribution，不能說「我們從未使用 soft scores」。差別應具體落在採樣分布、共同競爭背景、保留哪些關係以及評估方式，而不是把 MSE/KL 換個名字。

## 4. 為什麼不同問題能探測排序幾何

以下為說明可觀測性的線性推導，不是對現有非線性 student 已完成的因果測量。令 teacher 的角色化 query/document embedding 為 $q_T,d_T$，分數為 inner product：

$$
t(q,d)=q_T^T d_T,
\qquad
t(q,d_i)-t(q,d_j)=q_T^T(d_{T,i}-d_{T,j}).
$$

不同問題是在詢問：哪些文件差異方向有意義？若把已觀察 query 堆成矩陣 $Q$，任何滿足 $Qv=0$ 的方向，在這批 query 上都不可見。把文件 embedding 沿 $v$ 移動，不改變這些已觀察分數，卻可能改變新問題的排序。若 $Q$ 已滿秩，仍可能有接近零的奇異值，使某些方向受到很弱約束。

重複問題不會增加線性 rank；大量近義題可能主要改善相同方向的估計，而沒有補上欠覆蓋方向。可先量測 teacher query second moment 的譜、正則化 leverage，以及對共同文件集合的 score-profile 差異。但有效 rank 只是覆蓋代理，不是泛化保證；不能為提高 rank 而加入與使用場景無關的噪聲。

例如對相同文件集合詢問「方法如何提升準確度」「其計算代價」「在哪種資料條件失效」「相對另一方法的差別」，比只把第一題改寫四次更可能提供不同的排序約束。是否真的不同，要由 teacher 分數和人工支持證據核對，不能依問題文字外觀看起來多樣就認定。

若題目變化沒有讓 teacher 分清條件／關係，這不是 student 蒸餾能憑空補回的資訊。此時需要新的 evidence-grounded judgments，或另一個已驗證更能辨別該需求的 teacher。

## 5. 直接蒸餾與學習排序不是互斥路線

### 5.1 不必只靠 qrels 猜測可直接取得的 teacher 能力

凍結 dense teacher 可對合法 TRAIN 文字產生 embedding，也可對任意 query-document 配對提供分數，不必先有人類 qrel。這使大規模能力保留監督與少量高可靠 relevance 監督能分工。

但 teacher 輸出是「teacher 的行為」，不是「真實相關性」。若在所有問題上精確模仿其排名，就只能得到相同排名；純蒸餾並不提供系統性糾正 teacher 的新訊息。不能因此宣稱 student 絕不可能超過 teacher：有限模型的正則化或 BM25 互補可能使測試指標更好，只是並無這種保證。最終應同時比較 pure dense 與校準好的 dense+BM25，區分保留能力與 lexical complement。

### 5.2 向量重建不等於稀疏點積保真

直接把同維 student embedding 擬合成 dense embedding，是合理診斷控制。在 teacher 向量範數不超過 1、query/document 各重建誤差不超過 $\epsilon$ 時：

$$
|q_T^Td_T-\hat q^T\hat d|\le 2\epsilon+\epsilon^2.
$$

這是 worst-case 條件界，不是平均 MSE 的保證。即使逐分數誤差至多 $\eta$，也只保證 teacher margin 大於 $2\eta$ 的 pair 不反轉；前排近同分文件仍可能換位。

真正 SAE 多了一層：忽略 decoder bias，若 $q_T\approx Wz_q,d_T\approx Wz_d$，則

$$
q_T^Td_T\approx z_q^T W^T Wz_d,
\qquad
s_{\mathrm{posting}}=z_q^Tz_d.
$$

兩者不是同一函數。重建 decoder 若需要跨 latent 交互才能保留分數，普通倒排 scalar posting 未必實現得了。過完備字典也不能全域滿足 $W^TW=I$；至多考慮實際 active supports 上的限制條件。[Single-Stage Sparse Coding](https://arxiv.org/html/2605.30120v2) 的相關誤差界也依賴重建及 restricted decoder geometry，且其目標為 multi-vector，不能直接套成 II-42 保證。

所以主 loss 必須觀察**實際輸出的 sparse scoring function**，不能只看 decoder 重建。早期 dense-faithful 但高訪問量路線，與 M1911 可用 latent vocabulary 卻高 DF 的本地結果，正是這個區別的工程提醒。

### 5.3 簡單可測的候選：中心化 teacher 分數關係

先作 geometry-only 機制對照。令 $s_\theta(q,d)$ 為實際語義 sparse score，包含既有 RMS；$a>0$ 是只由 TRAIN 決定並凍結的 teacher 尺度。對每個共享文件 panel $C$，定義

$$
e_j=s_\theta(q,d_j)-a\,t(q,d_j),\qquad
\bar e=\frac{1}{m}\sum_{j=1}^{m}e_j,\qquad
L_{\mathrm{field}}=\frac{1}{m}\sum_{j=1}^{m}(e_j-\bar e)^2.
$$

它忽略不影響單題排名的共同分數 offset，卻保留文件之間的分數差。等權平方形式具有恆等式：

$$
L_{\mathrm{field}}=\frac{1}{2m^2}
\sum_{i=1}^{m}\sum_{j=1}^{m}
\left[(s_i-s_j)-a(t_i-t_j)\right]^2.
$$

因此算 $m$ 個分數就可約束整個 panel 的 pair margins，不需要顯式建立 $m^2$ 個 pair。這是 MarginMSE 類關係蒸餾的簡單變體，不宣稱新發明，也不表示得到 $m^2$ 份獨立資訊。若加入 mask、不等權或 robust loss，上述等價必須重新推導。

只做每個 query 的 row centering；不要順手去掉 document column mean，因為文件間的共同偏置仍會影響搜尋排名。Panel 包含固定背景及 head/boundary/相似文件群，按組別明示權重，避免大量易分背景主宰 MSE。它是 panel-distribution surrogate，不是假裝全庫無偏 loss。

這個機制對照不是直接把 dense score 減 BM25，再強迫非負 sparse 表示擬合 residual。初步 geometry loss 只約束 semantic 分量；評估仍看完整 hybrid。後續用可信 relevance 對完整

$$
h_\theta(q,d)=0.1b(q,d)+0.9s_\theta(q,d)
$$

作排序優化，讓 BM25 已解決的部分和 semantic 所需的修正體現在實際 margin 中。不宣稱上述 geometry loss 自動去除 lexical 重複；它本身沒有這種性質。先前 dense-minus-BM25 路線的失敗仍有效。

## 6. 重新架構訓練，而不是再堆 loss

```text
Approved TRAIN sources + task taxonomy + real source documents
                              |
                 Diverse query probes / shared document panels
                              |
             +----------------+----------------+
             |                                 |
    Frozen dense teacher                 Evidence judgments
    scores / relations                   graded / provenance
             |                                 |
    A. Preserve teacher function         B. Improve relevance
             +----------------+----------------+
                              |
                  Mature sparse trunk + head
                              |
                  BM25 + semantic posting score
                              |
             Full-corpus quality and unseen-task evaluation
                              |
                 C. Bounded cost-aware optimization
```

**A：能力保留。** 從多種合法 TRAIN 來源建立問題探針與共同文件 panels，對真實 text-to-sparse scoring 蒸餾 teacher 關係。沒有 qrel 的 query 仍可使用，不把跨 query 文件一律當硬負例。共同 panels 讓同一文件在不同意圖下的角色受到交叉約束；這是採樣設計，不是額外一套部署架構。固定背景可快取 teacher document embedding；student 參與更新的文件必須 fresh forward。

**B：相關性提升。** 在獨立記錄的可信標註上對完整 hybrid 訓練，並以另一批 TRAIN geometry anchors 監測 teacher 能力是否流失。人類／證據支持的修正可有意偏離 teacher；兩種目標不能悄悄混成模糊的 soft gold。Teacher 與人工判斷衝突時要明示例外和代價，不能同時要求精確模仿與反向排序。先比較分階段與固定混合中的一個簡單方案，不直接引入多 teacher 投票、梯度手術、動態 router。

**C：成本控制。** 品質學習從一開始記錄 NNZ、DF 和原生開銷，但第一個機制對照不改動既有 sparsity policy。建立可信品質 parent 後，另凍結受約束的成本對照，避免過度稀疏讓資料／幾何問題不可辨識。這不是成本可以永遠不管：若解法只能持續增密，不能直接擴大或晉級。

服務時仍是一次 query encoder 和既有 posting index，不把 teacher、標註器、共享 panels 或 dense 輔助 head 帶到前台。研究中的 dense 控制模型不是偷加第二路 serving reranker。

## 7. 問題應該如何變多

第一批優先在已授權真實文件上產生不同檢索意圖：實體／術語定位、機制／原因、條件／例外、方法比較、支持／反駁某主張、多跳中單份支持文件的角色。初期維持英語和既有 student 可見文本長度，避免同時增加語言和 context 的變量。

同義改寫作為對照，而不是多樣性成果。特別有資訊的是「最小但有意義的問題變化」：主題基本相同，但條件、關係或證據需求改變，正確文件次序也應有依據地改變。需要有可見 evidence；不能隨機加否定詞就假定正負一定互換。比較／多跳任務也不能假定單份文件必須獨自回答整題。

建議以 task family、source/topic cluster、teacher score-profile 分布三者共同記錄覆蓋；採樣保留代表性底座，只用有限配額補幾何欠覆蓋與 teacher/student disagreement。純 uncertainty sampling 可能挑到 teacher 錯誤和怪異題目，不能取代穩定分布。

LLM generator 只負責提出問題，不自動宣告其來源文件是唯一正例。檢索候選經 teacher 軟分數和必要的 evidence 複核，unclear 保留不確定性。生成模型、prompt、來源、token 可見範圍、重複家族均要可追溯；不把生產私有文字送到未批准的外部服務。

文獻有相關正向證據，但不能直接借用其收益：E5 的工作先產生任務再產生例子，區分不同長度／對稱性需求；這支持 task-first synthesis 的可行性。它使用大型 LLM 和不同訓練規模，不是我們約 50M sparse 模型的增益保證。[Wang et al., ACL 2024](https://arxiv.org/html/2401.00368v3)

GPL 將生成 query 和 cross-encoder 的連續 pseudo-label 結合，而非把生成配對當絕對正例；其每文件 query 數的最適值在資料集之間不同。這支持資料品質與問題多樣性要一起觀察，不支持「每份文件問越多越好」。[Wang et al., NAACL 2022](https://aclanthology.org/2022.naacl-main.168.pdf)

## 8. 最小而有辨識力的實驗順序

這些是下一協議的設計要求，不是已授權執行的參數。先凍結新資料來源、可用算力與試驗預算，再決定題目數、步數、模型或停止門檻；不套用 NG79 的失敗分數重設過關線。

1. **監督資訊預檢。** 只用 TRAIN teacher cache 或新合法 TRAIN 探針，比較同義改寫、同來源隨機新題、task-diverse 新題的 score-profile 覆蓋、近鄰／邊界 margin、teacher 能否辨識條件變化及輸入可見性。沒有有效差異就不把生成量擴大。覆蓋診斷一次收斂，不再開長串微小梯度診斷。
2. **先分離目標，再分離資料。** 第一個訓練對照固定相同已有 query、文件 panels、基座和工作預算，比較現行 qrel-conditioned 監督與 teacher-field 蒸餾；它比較的是目標，不是宣稱單一係數的效果。第二個對照固定選定的簡單 teacher-field 目標及 query 數，比較同來源／同類型新題與 task-diverse 新題。共同保留完全相同的可信 relevance batches，單列其曝光；不能把新增 query 的 teacher 頂位自動當人類標籤。
3. **必要時才做表示瓶頸控制。** 若 teacher-field 在 TRAIN 或新任務仍無法學好，才給同一成熟 trunk 加 training-only dense projection，對同樣文本／teacher 關係做控制。Dense head 明顯可學、sparse head 不可學，才支持在該設定下優先研究 sparse interaction；兩者都不行則先查資料、token 可見性及訓練充分性。它不是無條件容量證明，也不要求上線 dense correction。
4. **有方向後做足規模。** 同時報等曝光與實際 GPU/token/文件工作量；廣度組只看一遍不足以推論其收斂上限。先用 TRAIN learning curve 判斷是否還在明顯學習，再另凍結更充分訓練的必要對照，不能根據 holdout 挑步數。具實質訊號後做三個順序種子與更大來源覆蓋，不停在幾百題做永久微調，也不一次放大所有維度。

### 結果如何決定下一步

| 結果形狀 | 應採取的方向 |
| --- | --- |
| Teacher-field 保真提升，task-diverse 又在新任務上提升 | 擴展有價值探針與訓練充分性，保持模型架構簡單 |
| Teacher-field 保真提升，但 relevance 不升／下降 | 查 teacher 任務能力與爭議 judgments；不要只加更多同 teacher 標籤 |
| TRAIN 保真好，新任務保真差 | 優先分布覆蓋與泛化，不宣稱模型已學到通用幾何 |
| Dense 控制可學，sparse 控制明顯失敗 | 在相同資料與成本下研究 readout／interaction，而不是盲目生成更多題 |
| 品質提升但必須大幅增密 | 封存品質 parent，做獨立 rate-distortion／native 成本對照；不直接擴大 |
| 均無明確方向 | 停下來重新討論，不能以更多細小診斷迴避架構判斷 |

## 9. 驗收和 DF 必須與能力分開量測

Teacher fidelity 至少包括未參與梯度問題上的 top-k overlap、near-tie 分層 margin error 和排名相關；relevance 另看全正例 nDCG@10、Recall@100、逐域／任務 harm。兩種分數不互相替代。既有 exposed Sentinel/DEV 是 regression 面，不再冒稱新 holdout。

新驗證集先按來源文件群、topic、問題生成家族和 task family 拆分，而不只是 query string 去重；至少區分同語料新問法、新 topic、新任務與新文件。使用共有 corpus 的 transductive 蒸餾需明示；corpus overlap 不自動等於 qrel 洩漏，也不能冒稱跨新文件泛化。NG67 locked test 不作生成種子、不作 mining 或中途診斷；污染核對優先使用既有隔離的 hash manifest，必要的新確認集由獨立流程建立。

在成本上，對 literal postings 的未剪枝遍歷工作有：

$$
C_{\mathrm{scan}}=\mathbb E_q\left[\sum_{j:z_{q,j}>0}\mathrm{df}_j\right].
$$

它比單看 document NNZ 更接近查詢工作，但仍不是 block-max/WAND 或實際 native latency。要另外記錄 bytes、block/posting touches、候選量、fresh query encoding、P50/P95/P99 及背景維護。已有 DF-FLOPS 文獻支持關注高 DF term 的工程動機，不證明任意高 DF posting 都可刪。[Porco et al., 2025](https://arxiv.org/abs/2505.15070)

關鍵 utility 應是 posting 對 $s(q,d_i)-s(q,d_j)$ 的差異貢獻與聯合移除後的排序變化，而不是 impact 或 DF 單值。共同激活的部分可能無區分力，也可能在某類稀有 query 上不可替代；單項 marginal utility 不能相加成全 mask 的保證。NG53/54、M1911 的反證不因採用新訓練方向而失效。

## 10. 文獻與新方案的適用邊界

| 來源 | 本方案借用的原則 | 不做的外推 |
| --- | --- | --- |
| [Cross-Architecture KD, 2020](https://arxiv.org/abs/2010.02666) | 蒸餾 score margins，讓不同架構不必逐 activation 模仿 | 任意 teacher 尺度或候選池都適用 |
| [SPLADE-v3, 2024](https://arxiv.org/html/2403.06789v1) | 成熟 warm start、教師品質與分數尺度、蒸餾方法都重要 | 其 KL/MSE 配重就是我們的最優值；更多 negatives 保證跨域提升 |
| [Relational KD, 2019](https://arxiv.org/abs/1904.05068) | 可蒸餾樣本間關係，而非只複製單點表示 | 視覺 metric-learning 結果直接保證文字倒排品質 |
| [Anisotropic VQ, 2020](https://proceedings.mlr.press/v119/guo20h.html) | 壓縮應關注 inner-product 重要方向，不是等向重建 | 向量量化的成功等於 scalar posting 低成本成功 |
| [Interpret and Control Dense Retrieval, 2025](https://arxiv.org/html/2411.00786v2) | SAE 可加入檢索關係保留；重建與 latent retrieval 分開評估 | 所有稀疏度、所有表面都無損；表中低 K 仍有明顯差距 |
| [Latent Terms, 2026](https://arxiv.org/html/2605.29384v1) | Retrieval-native backbone 能提供可用 latent vocabulary | 本地 M1911 的高 DF／成本問題因此消失 |

本方案的新處不在發明一個叫 geometry 的 loss，而在可驗證地改變 **teacher 行為被採樣、保存、修正和驗收的方式**。如果這些改變沒有在新任務的全庫排名上取得利益，就應否證這個方案，而不是把它改名後再做一輪。
