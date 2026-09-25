# NG-0070：監督來源核對與後續分支

日期：2026-09-12。僅 TRAIN 資料來源診斷，與 [NG69 評估續跑](ng0069-evaluation-continuation.zh.md) 獨立。沒有新訓練、模型推論、DEV／LOCKED_TEST 排名或 qrels 修改。目的不是證明所有誤差來自標籤，而是補上原始正例與後加正例的來源資訊，防止後续把 LLM 重標錯稱為新增人類監督。

## 為何先做這件事

[NG68](ng0068-supervision-and-ranking-review.zh.md) 已發現教師／來源標籤衝突和池外前排競爭，但這不證明哪一方錯。RLHN 的方法是保留原正例，再用兩級 LLM 重標部分 hard negatives；其人工驗證只有 670 個配對，並非全部資料經人工確認。更強 reranker 也未辨識所有被 RLHN 重標的文件。因此教師分數、人類來源、相關性真值必須分開。[RLHN §3、§6–7](https://arxiv.org/html/2505.16967v1)

NV-Retriever 支持「相對正例排除疑似 false negatives」而不是無差別挖最高分負例；其 95% 門檻是特定模型、資料與分數尺度的實驗結果，不能直接套到多正例 hybrid。SPLADE-v3 支持檢查教師排序、分數尺度和 KL／MarginMSE，但同時更換多個因素會讓我們失去因果對照。[NV-Retriever §3](https://arxiv.org/html/2407.15831v1)、[SPLADE-v3 §2](https://arxiv.org/html/2403.06789v1)

## 有界資料核對

只處理 NG69 teacher targets 指定的 6,144 TRAIN query IDs，在 NG59／NG67 的 TRAIN 檔案尋找完整候選。`source_split=heldout` 是早期來源切分名稱，不代表這些被 NG67 明確分配到 TRAIN 的 query 是新的 locked test；不打開 NG67 DEV／LOCKED_TEST query 檔案。

上游比較來源為 `cfli/bge-full-data` revision `78f5c99b534a52824ab26bd24edda592eaed4c7a`，只取 Fever、HotpotQA、NQ 共 12 個 parquet shards 的 `query`、`pos` 欄位。以 column projection 避免下載龐大的 `neg` 欄位。保存上游 metadata／LFS SHA、取回的匹配原始列與列位置、派生輸出 SHA；**沒有下載完整 shard，因此不能宣稱完整上游檔案 SHA 已驗證**。本比較也不構成對上游資料逐例人工來源／授權的完整審核。

以 domain 加 NFC／空白正規化後的 query 連接，不用大小寫摺疊、模糊配對或語意近似。原 query 重複且正例集合不同則標記 ambiguous；找不到 query、原正例缺失或空集合都保持 unresolved。只有原正例完整保留時，當前正例才分為 `original_positive` 或 `added_vs_original`。保留每一個 document identity，不因文字正規化相同丟失配對。

`added_vs_original` 表示相對這個原始資料版本新增的正例，與 RLHN 描述的流程一致，但不獨立證明其 LLM judge 身分、正確性或誤標；`original_positive` 同樣不是逐 pair 人類真值認證。輸出 lineage 不能覆寫既有 qrels。

CPU-only、RSS 2 GiB、核心核對 wall 900 秒。實際啟動 offline ClearML，記錄來源、設定、結果與關閉回執；不得宣稱已 online sync。失敗保留 attempt，不默默改 matching 規則直到得到想要的統計。

## 結果如何改變下一輪

1. **廣度有效且來源可分清：** 優先增加去重、跨題型且来源可追溯的 TRAIN，而非增加相同問題的 epochs。固定模型／loss 對照，仍須控制 token、配對和時間成本。
2. **廣度不足且爭議配對集中：** 先對既有盲化 84 題／696 卡片做人工或獨立 CE 診斷。人工數據與模型判斷各自保存，模糊案例不當作確定負例；未付費或聯絡標註者。
3. **可信標籤仍分不開前排：** 在原文與模型可見 token 同時保留的配對上，區分 evidence truncation、teacher discrimination 和 student 表示容量。若原文可分而 encoder 看不見，單加 loss 或更大負例池無法補回遺失內容。
4. **池外競爭為主：** 在 TRAIN 上選真正改變正例名次的 witness，再做不確定性處理；不是把所有高分未標註文件当成負例，也不是再次直接按 DF 刪 posting。

新的設計候選是可信 graded pair 的 rank-impact supervision：以完整 hybrid 分數差訓練，權重依交換名次對 DCG 的影響，而不是單個 posting 的 DF。只有已判斷的 pair 才施加方向性梯度；未判斷 pair 保留為待複核或 soft teacher 目標。這是待驗證假說，不是已證明提升。

未來的低成本約束必須依實際訪問頻率和 query-atom 共現來測量。先取得可靠排序對照，再研究對低區分度且高訪問成本的共同 posting 降權；不把「較少 posting」等同更高品質或更低 native latency。

## 重現

```bash
python scripts/audit_ng70_provenance.py --research-root "$RESEARCH_ROOT" \
    --output "$NG70_NEW_PROVENANCE_DIR"
python -m unittest discover -s tests -p 'test_ng70*.py' -v
```

原文、上游匹配列和逐 pair lineage 保存在外部 research run；程式、測試、協議和精簡結論在主倉庫。新的人工標註數仍為零，本文件不宣稱品質或成本目標已達成。
