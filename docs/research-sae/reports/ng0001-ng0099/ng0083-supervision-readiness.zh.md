# NG-0083-C: 監督準備與曝光邊界

日期：2026-09-14。metadata audit 已完成；沒有下載新大資料、重新編碼或開啟 locked-test。執行契約見 [NG83 protocol](ng0083-ranking-causality-protocol.zh.md)，狀態見 [NG83 progress](ng0083-progress.zh.md)。

## 已核對的現有資料

NG80 使用的是 NG67 的 `data/train.jsonl`，SHA `01fbe89cbb8c9ba30330033fb4711dcaa780547c0fa394baba045d74d06ea775`；不是 NG67 的 `dev.jsonl` 或 `locked-test.jsonl`。NG67 原 13,824 TRAIN 再拆成 12,288 fit / 1,536 exposed diagnostic。原 NG67 的新資料 cohort、NG80 的新 split 不是同一件事，不可混稱「新 holdout」。本輪只查看 NG67 provenance metadata／preparer source，沒有對其 locked cohort 評分。

| Domain | TRAIN | Diagnostic | TRAIN query words p50 / p90 | 已驗證含義 |
|---|---:|---:|---:|---|
| FEVER | 4,096 | 512 | 8 / 12 | 較多 claim 形狀，不能全當自然搜索問題 |
| HotpotQA | 4,096 | 512 | 15 / 29 | 多 hop / 比較問題来源，不代表已自動驗證所有 intent |
| NQ | 4,096 | 512 | 9 / 11 | 首詞 who 1,612、when 860；詞面分布不是人工意圖標籤 |

- 13,824 條在 NFC + whitespace + casefold 下全數 unique，TRAIN/diagnostic normalized duplicate groups=0。
- **共享正例文件 547 個**。這本身不證明 qrel 洩漏；它證明不能宣稱 document/source-disjoint generalization。
- 本輪沒有新的 semantic near-duplicate 或 topic-family audit。NG67 preparer 曾做近重複排除，其歷史獨立 audit 明示 near-duplicate 檢查由 preparer 完成，不能升格成本輪獨立證據。
- 現有 frozen qrels 提供 positives；source 裡叫 `negative_passages` 的候選不自動等於人工確認不相關。不能由未命中 gold 集合估計 teacher-human 真負例錯誤率。
- NG67 metadata 記錄上游 TRAIN reservoir 為 NQ 51,801、HotpotQA 75,962、FEVER 25,488 條。這是歷史來源條目計數，不是新增可用／互不重複／授權已審核的 153,251 條 supervision，也不是本輪已載入訓練。

## 下一批資料的准入優先級

| Source | 可提供的不同訊號 | 來源／授權核對 | 決策 |
|---|---|---|---|
| 現有 NQ / HotpotQA / FEVER 原 TRAIN reservoir | 更多同域 query 與 teacher 排序關係 | 依既有 NG67 來源 offsets、SHA 和曝光 registry 重新篩；[NQ 官方資料描述](https://github.com/google-research-datasets/natural-questions) 的人類 QA 標註不是完整 retrieval judgments | 適合 exposure learning curve；不包裝成多來源突破 |
| HotpotQA 原 TRAIN | 多 hop、比較及 supporting facts | [官方頁](https://hotpotqa.github.io/) 明示資料與 processed Wikipedia 為 CC BY-SA4.0 | 保留來源與條款；不能拿新問法但同文檔當新來源 heldout |
| FEVER 原 TRAIN | claim/evidence、supported/refuted distinction | [官方頁](https://fever.ai/dataset/fever.html) 與 [資料授權](https://fever.ai/download/fever/license.html)：Wikipedia 相應條款，缺時採 CC BY-SA3.0 | refuted evidence 仍可能是相關文件；不得當 retrieval negative |
| SciFact TRAIN | 科學主張、證據支持／反駁，可補 Wikipedia 問題之外的機制 | [資料 repo](https://github.com/allenai/scifact)；[授權](https://github.com/allenai/scifact/blob/master/LICENSE.md) 區分 claims/evidence CC BY4.0、abstracts ODC-By1.0、code Apache2.0 | 小型 human-signal control；既往 SciFact 多次曝光，不能當全新最終資格測試，也不能支撐十萬題規模 |
| MS MARCO | 真實 web query / answer relevance，與目前三域有較大來源差異 | [官方條款](https://microsoft.github.io/msmarco/) 限 non-commercial research，沒有自動授予產品用途權利 | **先隔離為研究候選，不混入可發布模型訓練資料**；用途/條款另審，不接受條款或下載 |

上述是資料來源與用途盤點，不是模型權重的法律清算。NQ repo 的 code license 不應拿來推定所有 Wikipedia 文本的權利；每批 data revision、原文條款及加工來源要單獨記錄。

## 如何增加真正有效的 supervision

1. **先修目標，再擴相同目標的資料。** NG83-A 同 TRAIN 排名已顯著惡化，不能把原 teacher-pair utility loss 直接放大到十萬題；更多相同 surrogate 不會自動解除 objective mismatch。
2. **保留兩種 supervision 身分。** 完整 dense scores/margins 是 teacher geometry；human relevance 是 task truth。對 teacher 高分但未判定、teacher 低分但 known-positive、雙方已判定衝突分别標籤，不將 unknown 負例化。
3. **每批记录有效新增量。** raw rows、unique queries、query family、source document cluster、task/intent、teacher ranking diversity、document pairs 分開統計。近重複改寫和大量 pair 不能冒充更多獨立問題。
4. **評估先拆來源再凍結。** 使用新來源/topic/family；共享 corpus 的 transductive 設定單列。優先用既有隔離 hash manifest 排除 locked exposure，不用 locked-text 生成問題或挖 hard negatives。近重複／topic 檢查通過後才申請下一個未曝光評估。
5. **下一個 encoder 對照保留算力公平。** mature base direct ranking 與 passage-alignment warm-up -> ranking，分開標註總 query/doc/token exposure 和 compute。只在 B 的可輸出 scoring surface 顯示有效方向之後凍結，不能把 warm-up 多訓練的收益全歸方法。

本輪未完成全新來源的清洗、近重複驗證、人工 conflict review 或大規模 teacher encoding；以上為明確剩餘項，不報成完成。
