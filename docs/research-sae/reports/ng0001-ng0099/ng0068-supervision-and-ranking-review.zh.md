# NG-0068：監督來源、排序目標與人工複核

日期：2026-09-12。研究診斷，不修改產品、既有模型、歷史 qrels 或 frozen NG run。

## 決策背景

[NG-0066](ng0066-training-duration-review.zh.md) 在相同的 1,536 TRAIN 題目上，以三個順序 seed 訓練四輪。TRAIN nDCG@10 平均從 0.873894 升至 0.922973，但 exposed DEV 只從 0.892436 升至 0.897476；逐題平均跨 seed 後的分域 bootstrap 差值區間包含零。NQ DEV 平均從 0.838621 降到 0.833718。這支持停止以反覆訓練同一小批題目為主線，但不能單獨證明模型容量不足或資料標註錯誤。

目前目標仍是 BM25 + learned semantic posting 的完整方案超過 PPLX dense，同時改善實際總體成本。不能把單獨 SAE、已曝光 TRAIN、posting 數量或壓縮檔案大小的收益當成達標。

## 本輪實測

`scripts/analyze_ng68_supervision.py` 只解析 NG59 TRAIN、其固定教師 targets，以及 NG66 epoch 1/4 的 TRAIN ranking prefix。既有文件背景 59,111；沒有重新 inference，也沒有讀取 NG67 DEV/LOCKED_TEST 的查詢或評估結果。完整來源檔案 SHA 校驗與解析 held-out 記錄是不同操作。

| TRAIN 域 | 題數 | 教師把來源負例排在所有正例之前 | 至少一個正例被教師排到來源負例後 | 混合 target 仍有正負次序衝突 | 教師分配給正例的平均機率 |
|---|---:|---:|---:|---:|---:|
| Fever | 512 | 0 | 51 | 0 | 0.982986 |
| HotpotQA | 512 | 8 | 163 | 2 | 0.943852 |
| NQ | 512 | 118 | 222 | 16 | 0.635113 |
| 合計 | 1,536 | 126 | 436 | 18 | 0.853983 |

這裡的「來源負例」不等於人類確定不相關。因此表中是教師與現有 qrels 的衝突，不是已證明的教師錯誤率。所有 40,530 個來源正負配對中，教師有 3,189 個反向排序；混合 target 後只剩 69 個。不能以 18 題解釋全部泛化差距。

固定訓練池平均 16.759 個文件。教師分佈的 effective support `exp(H(p))` 中位數為 1.381；混合 target 對正例的總機率中位數為 0.992059。這是分佈集中度的描述，不是 loss 錯誤或資訊完全消失的證明。

另一個診斷只計算真正能改變 nDCG@10 的前排競爭文件：位於 top-10、不是已知正例、且排在至少一個正例之前。epoch 4 的三個 seed，各有 412 / 400 / 401 個 TRAIN 問題包含這類文件。先在每個受影響問題內計算池外比例再平均，為 65.68% / 66.39% / 65.80%；不是把所有 top-10 非正例都算成錯誤。

Fever 同一比例為 88.91–90.18%，HotpotQA 為 69.47–71.83%，NQ 為 48.59–50.55%。NQ 同時存在池內排序與池外競爭問題，不能只增加 hard negatives 或只替換教師。這些數值是 epoch 4 當前候選診斷，與 NG60 不同 checkpoint、不同 witness 定義的數字不能直接相減。

## 數學上的取捨

歷史 target 與 loss 維持不變：

$$
p_T(d\mid q)=\operatorname{softmax}_{d\in C_q}(t_{qd}/0.04),
\qquad
y_{qd}=\frac{1}{2}\frac{\mathbf{1}[d\in P_q]}{|P_q|}
       +\frac{1}{2}p_T(d\mid q).
$$

$$
\mathcal{L}_q=-\sum_{d\in C_q}y_{qd}
\log\operatorname{softmax}_{d\in C_q}(s_{qd}),
\qquad s_{qd}=s_{\mathrm{BM25}}(q,d)+s_{\mathrm{semantic}}(q,d).
$$

分數包含既有固定 lexical scaling 與 semantic RMS normalization，不能把此式理解成原始分數直接相加。若每題 logits 可以自由設定，最優分數差為 `log(y_positive/y_negative)`；實際共享 encoder 不一定能同時達到每題的自由最優解。

單正例時，50% 的標籤保底使任何單一負例的 target 不會嚴格超過正例。多正例時，保底被分成 `0.5/|P|`，個別來源負例仍可能超過弱正例。本輪量化這件事，而不是憑直覺把 soft labels 全部換掉。對 teacher/gold 有爭議的 pair，先保留來源與不確定性，不讓教師自動推翻可信的人類標註。

## 文獻與監督來源

- **RLHN 不是純人類標註。** 其方法保留原有正例，使用兩級 LLM 判定、重標 hard negatives；研究另以 670 個 pair 做人工驗證。這不等於每個新正例已由人類確認。NG59/67 現有本地記錄保留來源 document ID，但沒有逐 pair 區分原始正例與 LLM 新增正例；需要與原始版本對照，不可直接把當前所有正例稱作 human gold。[論文 §3、§7–8](https://arxiv.org/html/2505.16967v1)、[公開資料卡](https://huggingface.co/datasets/rlhn/rlhn-400K)。
- **更好的排序監督值得控制實驗。** SPLADE-v3 同時研究 cross-encoder 教師、分數尺度與 KL/MarginMSE；其結果不能證明我們直接照搬 loss 權重就有效。論文區分 100 個 mined negatives 的來源池與實際每次取 8 個 negatives。增加負例主要改善 in-domain，不保證跨域。[論文 §2–3](https://arxiv.org/html/2403.06789v1)。
- **池外競爭者不能自動視為真負例。** NV-Retriever 的 positive-aware mining 以正例分數作參考來排除疑似 false negatives。本文借用「相對正例辨識不確定性」的原則，不直接移植其 95% 閾值到不同 score/多正例契約。[論文 §3.1](https://arxiv.org/html/2407.15831v1)。
- **新增原始人類監督是獨立選項。** MS MARCO 的人類標註來源可供研究，但 passage qrels 是稀疏判定，unjudged 仍不等於已判負例；原始 passage 與轉移到 document 的標籤也要區分。先做 TRAIN-only、小量、可追溯樣本與曝光排除，不直接混入所有資料或宣稱已完成授權審查。[官方資料說明](https://github.com/microsoft/MSMARCO-Passage-Ranking)、[原始 QA 資料說明](https://microsoft.github.io/MSMARCO-Question-Answering/)。

## 人工複核已準備，尚未標註

已按「教師衝突／非衝突但學生退步／其餘控制」與三個域做互斥分層。每格最多依固定 SHA 順序取 10 題，不足不補其他格，實際 **84 題、696 個文件卡片**。這是偏重診斷的分層樣本，不可用未加權總平均估計全資料的標註錯誤率。

每題包含所有既有正例，以及代表 seed 59059 epoch 4 hybrid、PPLX 和 BM25 各前三個非已知正例的聯集；以固定 hash 打亂文件順序。給評審的 `annotation-blind.json` 不包含模型名稱、分數、原始標籤、stratum、domain 或 corpus index；`annotation-private-key.json` 必須分開保管。模型 top 候選只是待判斷文件，不是預先標記的負例。

建議至少兩人獨立判定 `directly_supports / partially_supports / not_supporting / unclear`，標出原文 evidence span 與理由；另外記錄「完整原文支持，但 query64/document256 截斷後看不到」的案例。不同證據粒度、不完整標註與 encoder 的 token 可見範圍必須分開。Hotpot 的單篇 supporting document 不要求單篇獨自回答完整多跳問題。分歧需獨立裁定，不因某模型分數較高就採用它。

**目前收集到的人類新標註數為零。** 未聯絡／付費委託標註者，也沒有把 LLM 判定冒充人工結果。人工複核不是其餘研究繼續前進的阻塞條件。

## 下一步

1. [NG-0069](ng0069-matched-exposure-breadth-plan.zh.md) 比較相同 6,144 query exposures：舊 1,536 題重複四輪，與 6,144 個不同題目各一輪。先固定模型、原 target、來源、RMS、詞表和 optimizer，以隔離題目廣度。擴大資料不等於人類監督分支。
2. 在 TRAIN-only 上恢復原始與 RLHN 正例的 pair-level provenance，檢查人工樣本中的原文／截斷與標籤歧義。接著才決定值得增加的是新題目、可信 graded labels，還是更強 cross-encoder 的排序 supervision。
3. hard-negative 後續候選採取有界「當前模型會排到前面、且會改變正例 rank」的 witness 聯集，再作可信標籤優先及不確定性屏蔽。這是有待驗證的研究假說，不是已達成的創新效果，也不是單按 DF 刪 posting。
4. 所有方案都以完整 hybrid 的前排品質、全正例 recall/rank harm、逐域泛化和實際成本驗收。若廣度無改善，不再盲目延長同一輪；先根據 CE／人類監督診斷決定表示能力或 token truncation 對照。

## 重現與產物

```bash
python scripts/analyze_ng68_supervision.py \
    --reference "$RESEARCH_ROOT/NG-0059" \
    --run "$RESEARCH_ROOT/NG-0066/attempt-v2" \
    --output "$RESEARCH_ROOT/NG-0068/supervision-audit-v1"
python -m unittest discover -s tests -p 'test_ng68_supervision.py' -v
```

外部產物包含 `results.json`、逐 TRAIN 題診斷、盲化樣本、私有 mapping 與 SHA manifest；原文卡片不進 Git。來源和輸出不可覆寫，修正另開 attempt。測試包含 target 公式、單／多正例、排名 witness 邊界、部分 strata、held-out parse 防護與盲化／完整正例保留。
