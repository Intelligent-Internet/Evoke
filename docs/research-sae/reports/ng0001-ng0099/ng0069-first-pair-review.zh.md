# NG-0069：第一組廣度對照的初步品質與成本權衡

觀察時間：2026-09-12 16:52 UTC。**僅順序 seed 59059 的 A/B 已配對完成，不是三個 seed 的最終結論。** 第二個 A 已完成，第二個 B 正在評估；沒有改變模型、訓練或晉級門檻。背景、計分與限制見 [廣度協議](ng0069-matched-exposure-breadth-plan.zh.md)、[執行契約](ng0069-execution-contract.zh.md) 和 [評估續跑](ng0069-evaluation-continuation.zh.md)。

## 已核對的品質

所有結果使用同一 233,009 文件背景、新 DEV 1,536 題，每域 512 題。三域 macro 等權。Hybrid 是固定 BM25 0.1 + semantic 0.9/RMS，不是 SAE-only。新 DEV 已曝光，用於方案選擇；LOCKED_TEST 未編碼或評分。

| 模型 | nDCG@10 | 全正例 Recall@100 |
|---|---:|---:|
| Initial hybrid | 77.7044% | 96.2175% |
| A-59059：舊 1,536 題重複四輪 | 77.9292% | 95.8991% |
| B-59059：6,144 個不同 TRAIN queries 一輪 | 79.0783% | 96.4867% |
| PPLX dense | 83.0229% | 97.7456% |

B 相比 A 的 macro nDCG 增加 1.1491 個百分點，Recall 增加 0.5876 個百分點。但相對 dense 仍分別低 3.9446 和 1.2588 個百分點，不能稱為整體目標達成。

| 領域 | B-A nDCG 變化，百分點 | B-A Recall 變化，百分點 |
|---|---:|---:|
| Fever | +3.1520 | +0.0895 |
| HotpotQA | -0.3813 | +0.5534 |
| NQ | +0.6765 | +1.1199 |

廣度的初步排序改善主要來自 Fever。NQ 的 B nDCG 仍低於 initial（66.9863% 對 68.2021%）；不能把 B 優於 A 解讀為所有領域都比訓練前更好。這是描述性結果，尚無三個順序 seed 的配對 bootstrap 結論，也不代表獨立初始化的可重現性。

## 成本代理顯著增加

| 語意表示量 | A-59059 | B-59059 | B/A |
|---|---:|---:|---:|
| 文件非零權重總數 | 69,252,920 | 148,155,857 | 2.1393 |
| 每文件平均非零權重 | 297.21 | 635.84 | 2.1393 |
| 文件 CSR bytes | 554,955,400 | 1,186,178,896 | 2.1374 |
| 新 DEV query-DF 平均代理 | 481,478.93 | 1,238,066.40 | 2.5714 |

兩組使用相同 full nonnegative readout，沒有 top-k posting 裁切。協議中的 query64/document256 是 **tokenizer 輸入長度上限**，不是每 query/document 的 posting 上限。凍結的 `model59.Encoder(False, ...)` 會保留所有正 readout 值，因此上述 NNZ 差異沒有違反輸出上限；本輪原本就未設該上限。

模型品質改善伴隨較稠密的語意表示，這是已觀察到的關聯，不是「多出的 posting 造成全部收益」的因果證明。兩組 BM25 部分固定；表格不是完整混合索引的實體尺寸，也不是原生引擎實際訪問的 postings。Query-DF 代理沒有模擬剪枝、cache 或維護成本，不能直接換算為 2.57 倍查詢延遲。

評估 process-tree 峰值 RSS：A 約 3.58 GB、B 約 6.10 GB，均低於原有 16 GiB 上限。這是離線評估程序記憶體，不是 PostgreSQL resident index 記憶體。原生 index bytes、postings visited、warm latency 和完整總成本仍未測量。

## 證據與後續判斷

`evaluation-v2/rank-A-59059` 的 complete SHA 為 `1c17685fae7d0d12383646b36aa0f1ec517809c67389148f2cfa2b310fcfa8f2`；`rank-B-59059` 為 `d39fc482a2f25eccb8db3002b0d53d50b71f0646df4c51630a999798d98b49f0`。兩個階段的 exit 0、owned group closed、actual-start offline ClearML closed，以及 complete 中全部輸出 SHA 均通過。每階段 7,872 題的分數、top-100 和全部正例 rank 已由 worker 的兩條路徑核對；整輪 70,848 筆的獨立 reviewer 尚未執行，不能混稱為已通過。完整觀察回執為 `NG-0069/evaluation-monitor-20260912T1652Z.json`，新輸出的 Mac 全量鏡像仍待完成。

下一步保持原三個順序 seed 的實驗，先完成凍結的配對品質門檻，再同時審閱成本代理的變化。即使 B-A 品質門檻通過，仍不表示 B 超過 dense 或成本更低。這個初步結果不足以授權直接擴大訓練，也不應在本輪看過 DEV 後加 DF penalty、mask、換 checkpoint 或追調混合權重。若成本增長跨 seed 重現，需要另立 TRAIN-only 的表示密度／排序貢獻診斷，不能只按高 DF 刪除 posting。

沒有產品或生產部署變更，沒有新增人工標註，沒有刪除任一 attempt。
