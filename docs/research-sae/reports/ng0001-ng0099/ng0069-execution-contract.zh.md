# NG-0069：監督廣度實驗的執行契約

日期：2026-09-12，執行版本 v1。在新的 DEV 排名產生之前凍結。本文件補充 [研究協議](ng0069-matched-exposure-breadth-plan.zh.md)，不改其假說、模型、scoring、資料分割或晉級門檻，也不改寫已凍結的教師準備協議。

## 已完成的準備

- 教師準備 exit 0，實際耗時 1,362.18 秒，process-tree peak RSS 4,208,099,328 bytes。實際開始的 offline ClearML task 已關閉；未同步 online。
- 6,144 個 TRAIN targets 經 Mac 和訓練節點分別獨立重算通過。最大 score 差 `7.78e-16` 以下，最大 target 差 `1.89e-15` 以下；舊 embeddings、targets 重用完全一致。這不是再次執行完整模型推論。
- 固定 lexical 資料覆蓋 233,009 文件、6,144 TRAIN、1,536 新 DEV、192 已曝光 DEV。舊 59,111 文件 prefix、舊 query 和訓練池分數保持一致。95,475 詞表、IDF 和 RMS 不重新擬合；平均文件長度與 IDF 仍由原先 14,539 文件擬合。
- 新文件 16,626,243 個 token 中有 640,049 個 OOV，約 3.85%。文件長度正規化仍包含 OOV 的完整 token 長度。沒有空 lexical query。這是固定詞表的覆蓋限制，不在看到 DEV 成績之後偷偷擴詞表。
- 新人工判斷數仍為零；[NG68](ng0068-supervision-and-ranking-review.zh.md) 的 84 題、696 卡片只是一份盲化待複核材料。未將來源負例當成人工確定負例。

## 執行順序與完整性

`scripts/prepare_ng69_lexical.py` 產生固定 lexical extension。`scripts/ng69_pipeline.py` 在可信輸入上凍結 source、protocol、資料、模型依賴與 A checkpoints 的 SHA，核對教師退出／追蹤回執、獨立 audit，以及 query、pool 順序、所有正例和分數的對齊。遠端使用相同凍結副本，不修改 NG59/66 的常數或歷史程式。

有限序列共 29 個階段：

```text
verified TRAIN targets + fixed lexical data
                    |
        B seed 59059: four quarters
        B seed 66061: four quarters
        B seed 66067: four quarters
                    |
         all training checkpoints sealed
                    |
         BM25 full-background evaluation
                    |
    PPLX dense -> initial hybrid -> A/B each seed
       encode then evaluate each fixed model
                    |
       independent paired review + cost audit
```

每個 quarter 是同一 seed 的 6,144 題 permutation 中不重疊的 1,536 題，384 個 updates；四段合計 1,536 updates。checkpoint 與 AdamW moments 保存後重新載入，參數／optimizer SHA、累計步數和 query/document probe 輸出必須一致。每題保留 loss、NNZ、候選數、實際截斷後 token 數與 VJP replay 結果；每次更新核對非零有限梯度及參數改變。

這只匹配 query exposures 與更新數，不匹配精確 FLOPs。A 的歷史記錄若缺 token 統計，不能把 B 的 token 統計冒充 A 的測量；需明列已測與未知成本。

## 評估邊界

全部模型在同一 233,009 文件背景上排名。PPLX 可原樣重用已驗證的 117,541 文件與 6,144 TRAIN embeddings，只補剩餘文件與 1,728 DEV queries。重用索引明確映射，未編碼行不能以零值代替。未重用的模型輸出重新編碼；query64/document256 契約不變。

所有 TRAIN、新 DEV、已曝光 DEV 分開保存。訓練全部完成才進行 DEV 評估，不能用中途結果選 checkpoint。1,536 LOCKED_TEST queries 不編碼、不評分。語意模型評估的是既有固定 BM25 0.1 + semantic 0.9/RMS 的完整 hybrid；PPLX dense 與 BM25-only 另列。不能拿舊 59,111 背景結果直接比較。

每題以兩種累加方式核對全文件分數，絕對誤差上限 `1e-12`，再独立驗證 top-100 tie-break 與全部正例 rank。全正例 nDCG@10、Recall@100、逐題 harm 保留，三個 seed 先逐題平均再作分域 paired bootstrap。晉級門檻仍由研究協議決定，單 seed、TRAIN 或 loss 下降均不代表勝過 dense。

CSR NNZ、bytes 與 query-DF work 只是成本代理。此 pipeline 不測 native postings visited、真實 index bytes 或 warm p50/p95 latency，不能據此宣布低成本目標完成。

## 資源與失敗處理

使用 lambda2 的空閒 GPU0，持有 cooperative GPU lock 與 run lock，CPU4、process-tree RSS 16 GiB、GPU allocated 20 GiB。每個 train quarter 上限 30 分鐘，encode/rank 階段上限 90 分鐘；先用 TRAIN/文件前綴做成本 canary，不用 DEV 分數調整資源界線。啟動前確認 host 可用記憶體大於 24 GiB、run filesystem 可用空間大於 40 GiB，以及 GPU0 空閒。不占用同事的 GPU1/2。

控制器記錄實際 PID、峰值 RSS、退出碼，關閉自己所屬 process group，要求 `ESRCH` 才算清理完成。每階段只有 exit 0、closed tracking 和完整輸出 SHA 同時成立才可前進。失敗／中断 attempt 保留，不原地自動重試。成功階段可經校驗跳過；失敗後先診斷再決定新 attempt，不能為跑完而放寬門檻。

```bash
python scripts/prepare_ng69_lexical.py \
    --research-root "$RESEARCH_ROOT" --output "$NG69_LEXICAL_RUN"
python scripts/ng69_pipeline.py \
    --research-root "$RESEARCH_ROOT" --run "$NG69_PIPELINE_RUN" --mode freeze
python scripts/ng69_pipeline.py \
    --research-root "$RESEARCH_ROOT" --run "$NG69_PIPELINE_RUN" --mode supervise
python -m unittest discover -s tests -p 'test_ng6*.py' -v
```

先將本文件保存為新 pipeline run 的 `protocol.md`，並將研究協議保存為 `research-protocol.md`，再 freeze，連同 source 和 inputs manifest 複製至計算節點。兩份協議均綁定 SHA。大型輸出與完整逐題資料存外部研究目錄，程式、測試與本協議保存在主倉庫。實際進度另記錄於 run 回執，不修改本協議追認結果。

## 結果如何改變策略

若廣度帶來穩健新 DEV 收益，下一階段優先擴充有來源、去重且曝光可控的監督；若無改善，先用 NG68 的人工複核／原始標籤 provenance、CE pair discrimination 和可見 token 截斷診斷定位，而不是重複舊輪數。池外競爭者以當前模型的 rank-impact witness 挑選，再處理 false-negative 不確定性；不把單純高 DF 或高模型分數當成刪 posting／判負例的充分條件。
