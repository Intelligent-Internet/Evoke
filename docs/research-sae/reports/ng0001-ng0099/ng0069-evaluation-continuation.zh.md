# NG-0069：僅評估續跑與策略調整

日期：2026-09-12。使用者重新授權繼續研究後，建立 `evaluation-v2`，不修改已凍結的 `pipeline-v1`。本協議在新 attempt 的 DEV 排名產生之前保存；沿用 [廣度實驗](ng0069-matched-exposure-breadth-plan.zh.md) 及 [執行契約](ng0069-execution-contract.zh.md)。

## 不重訓的繼續方式

原停止原因是 [dense 評估器的獨立累加成本](ng0069-dense-evaluation-cost-review.zh.md)，不是模型品質結果。新 attempt 只把第二條核對路徑的分塊乘积再 sum 改為 `np.einsum('ij,j->i', documents, vector, optimize=False)`。主分數仍是逐 query float64 matvec。保持絕對誤差 `1e-12`、top-100 和每一個正例 rank 核對、5,100 秒 TRAIN canary、5,400 秒 phase 上限及全部原資源界線。

12 個成功 training quarters、BM25 排名和 dense embeddings 共 14 個階段，逐一核對完整輸出 SHA、exit 0、closed tracking 和 process-group 清理，再以相對連結明確引用。這不是新產物或獨立備份；不複製大型 checkpoint，也不重跑訓練。新 inputs manifest 綁定原 source、inputs、兩份協議、失敗回執和所有重用階段的 complete/exit/tracking SHA。驗證時必須核對連結的真實目的地，worker 禁止執行繼承階段。原失敗 `rank-dense` 不被引用為成功；新目錄獨立執行剩下 15 階段。

原失敗 attempt 的 Mac 副本已通過 231 個一般檔案、8,168,895,409 bytes 的全量 size/SHA，65 個 frozen inputs 與 14 個成功階段校驗。失敗階段保留，遠端沒有刪除。這不是重新模型推論的 restore test。續跑完成後仍需新產物鏡像與独立 reviewer，不以舊副本替代。

## 實驗與決策邊界

模型、三個順序 seed、teacher、loss、BM25 權重、RMS、詞表、query64/document256、資料分割和所有晉級門檻完全不變。三個順序 seed 不是三個獨立 initialization。新 DEV 用於選方案，已曝光 DEV 分開列示；LOCKED_TEST 不編碼、不排名。新增的人類判斷仍是零。原 84 題／696 卡片為待標註材料，不稱作人工資料集。

結果分開回答三件事：B（6,144 個不同問題）是否優於 A（1,536 題重複四輪）；完整 hybrid 是否優於同背景 PPLX dense；實際總成本是否較低。第一個門檻通過不等於後兩項達成。NNZ／CSR bytes／query-DF 是診斷代理，不代替 native 索引大小、posting 訪問、encoder 成本和 warm latency。

若廣度門檻通過，下一輪才考慮更廣且有逐 pair 來源的监督。若未通過，不再以延長相同 epochs 為主線：先檢查 TRAIN 的原始人類來源與 LLM 新增標籤、池外 rank-impact 競爭者，以及原文證據是否超出 tokenizer 的可見窗口。人工／CE 的判斷不能覆寫 frozen qrels，unjudged 不能直接當真負例。任何新 loss、候選池或資料量更改需另開對照，不在本次 A/B 中混入。

## 重現

保留原 `protocol.md`、`research-protocol.md`，另將本文件保存為 `continuation-protocol.md`。在新的空目錄凍結目前 source：

```bash
python scripts/ng69_pipeline.py --research-root "$RESEARCH_ROOT" \
    --run "$NG69_EVALUATION_V2" --reuse-run "$NG69_PIPELINE_V1" --mode freeze
python "$NG69_EVALUATION_V2/ng69_pipeline.py" \
    --research-root "$RESEARCH_ROOT" --run "$NG69_EVALUATION_V2" --mode supervise
python scripts/analyze_ng69_breadth.py --research-root "$RESEARCH_ROOT" \
    --run "$NG69_EVALUATION_V2" --output "$NG69_NEW_REVIEW_JSON"
```

reviewer 必須使用與 attempt 相同的 pipeline source，並在結果中列明 inherited provenance。失敗保留、不自動重試或放寬成本門檻；先定位新證據，再決定下一個 attempt。這份文件不宣稱全量評估或模型目標已完成。
