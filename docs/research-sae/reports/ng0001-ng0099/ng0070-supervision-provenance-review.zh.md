# NG-0070：監督來源與訓練目標質量分配

日期：2026-09-12。依 [TRAIN-only 協議](ng0070-supervision-provenance-plan.zh.md)，完成原始來源核對與保存 target 的再計算。沒有重新推論、沒有 DEV／LOCKED_TEST 分數，也沒有修改進行中的 NG69 模型、loss 或 qrels。

## 核對覆蓋

固定原始 BGE revision `78f5c99b534a52824ab26bd24edda592eaed4c7a`，掃描 Fever 29,096、HotpotQA 84,516、NQ 58,568 列的 query／positive 欄位。只保留與 NG69 6,144 個 TRAIN query IDs 對應的來源列。12 個 parquet shards 未完整下載，不聲稱上游完整檔案 SHA 已驗證；原始匹配列、shard／row offset、metadata 與所有派生產物 SHA 均保存。

6,133／6,144 題可按嚴格 NFC／空白正規化完成連接。11 個 Fever 問題在原資料存在不同正例集合，保留 ambiguous，33 個現有正例不強制歸類。沒有 query-not-found 或 original-positive-missing 案例。重複原始列只在正例集合完全一致時合併，保留所有 document identities。

| TRAIN 域 | 題數 | 可解析原始來源正例 | 相對原資料新增正例 | 未解析正例 | 含新增正例的題數 |
|---|---:|---:|---:|---:|---:|
| Fever | 2,048 | 2,597 | 350 | 33 | 248／2,037 可解析題 |
| HotpotQA | 2,048 | 4,096 | 183 | 0 | 124／2,048 |
| NQ | 2,048 | 2,048 | 1,579 | 0 | 650／2,048 |

NQ 的新增正例占該域全部正例約 **43.5%**，但只有 **31.7% 的問題**含新增正例；兩個分母不同，不能混稱。HotpotQA 新增正例約占 4.3%。`added_vs_original` 是來源差異，**不是誤標**；`original_positive` 也不是本輪新取得或逐 pair 重新認證的人類標註。RLHN 的公開方法說明了 LLM 重標機制，但本次配對不能證明每個新增正例使用了哪位 judge。[RLHN §3](https://arxiv.org/html/2505.16967v1)

## 保存的訓練目標

獨立將 6,133 題所有正例鍵對齊到 frozen document index，重新計算 teacher softmax 與目前 target，要求絕對誤差不超過 `1e-12`。11 題歧義來源不做來源分類分析，不影響它們仍保留在原訓練協議中。

$$
y(d\mid q)=\frac{1}{2}\frac{\mathbf{1}[d\in P_q]}{|P_q|}
            +\frac{1}{2}p_T(d\mid q),
\qquad
p_T(d\mid q)=\operatorname{softmax}(t_{qd}/0.04).
$$

令原始來源正例集合為 $O_q\subseteq P_q$，它從 label 部分取得的總量為：

$$
b_O(q)=\frac{1}{2}\frac{|O_q|}{|P_q|}.
$$

因此「標籤占 target 一半」並不等於「原始來源正例占一半」。新增正例越多，原始集合取得的 label 份額越小；這可能是合理的多正例覆蓋，也可能把不可靠標註放大，單憑公式不能判定利弊。

| 可解析 TRAIN 域 | 原始集合 label 份額均值 | 原始集合 teacher 機率均值 | 原始集合最終 target 均值 | 新增集合最終 target 均值 |
|---|---:|---:|---:|---:|
| Fever | 0.4686 | 0.9765 | 0.9568 | 0.0344 |
| HotpotQA | 0.4881 | 0.9308 | 0.9535 | 0.0133 |
| NQ | 0.3965 | 0.5527 | 0.6729 | 0.1435 |

表中是每題內集合機率加總再跨題平均，不是按正例數加權。原始與新增 target 的剩餘量分配给來源負例；來源負例仍可能包含未標註的相關文件。

NQ 有 559／2,048 題，教師把至少一個來源負例排在全部原始正例之前；225 題的某個新增正例高於全部原始正例。混合 target 後，仍有 53 題的來源負例 target 高於某個原始正例。對應 Fever 為 4／5／1 題，HotpotQA 為 48／3／13 題。這不是因果歸因或 label error rate，也不能把 NG68 的較小問題集合計數直接當作前後改善比較。

## 策略修正

現在有理由優先把 NQ 的「原始來源／新增正例／高排名未標註文件」分開診斷，而不是對所有域一律增加資料或訓練輪數。更多人類監督值得嘗試，但優先需求是**對影響前排的爭議配對給出可靠的 graded relevance 與 evidence span**，而非增加同類 soft labels 的總數。

先完成 NG69 的相同曝光量 A/B；不要根據本次 TRAIN 診斷改它的門檻或 target。若廣度仍無穩健收益，下一個單變量對照應優先核對這 53 個 target 反向案例與部分正常控制，再比較原文與 query64/document256 的可見證據。原始正例並非一定唯一正確答案，新增正例也不得自動移除。若可信複核支持原排序，才考慮以來源可信度／graded relevance 決定 pair 方向與權重；若新增正例更好，反而應保留其訊號。

暫不採用任意「原始權重乘二」或全刪 LLM 正例，也不立即把 dense teacher 全換成 CE。需要以固定資料、固定模型的對照分開測試 teacher discrimination、label ambiguity、token visibility 和 student capacity。最終仍是完整 BM25＋semantic hybrid 的 held-out 排名、全正例召回與實際總成本，不能以 TRAIN 目標更符合原始標籤代替。

## 執行與重現

來源核對核心約 77.9 秒，exit 0，實際 offline ClearML task 已關閉；未 online sync。先前一次 preflight 在輸出／tracking／網路下載之前發現 NG67 已包含相同 NG59 TRAIN prefix；失敗 source 已保留，修正只允許內容完全一致的重複 ID，任何衝突仍停止。沒有更改 match 規則來追認結果。

`provenance-v1` 保存來源與逐 pair 分類；`target-mass-v2.json` 保存 target 分配再計算，另核對 document identity map 與 lexical frozen inputs 的 SHA。早先相同統計的 `target-mass-v1.json` 保留，不覆寫。完成回執驗證不等於上游完整 shard 校驗或模型 restore test。新人工標註仍為 **零**。

NG60 系列與 NG70 的研究 fixtures 合計 **56 項通過**；未重跑與本次研究無關的產品完整回歸。

```bash
python scripts/analyze_ng70_targets.py --research-root "$RESEARCH_ROOT" \
    --provenance "$NG70_PROVENANCE_V1" --output "$NG70_NEW_TARGET_AUDIT"
python -m unittest discover -s tests -p 'test_ng70*.py' -v
```
