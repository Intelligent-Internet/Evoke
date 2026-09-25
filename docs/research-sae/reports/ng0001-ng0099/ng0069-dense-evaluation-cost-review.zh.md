# NG-0069：dense 評估成本保護的停止與診斷

日期：2026-09-12。這是離線評估器的工程診斷，不是模型品質結論，也不是 PostgreSQL／ANN 服務查詢的效能測試。[原研究協議](ng0069-matched-exposure-breadth-plan.zh.md) 與 [執行契約](ng0069-execution-contract.zh.md) 保持不變。

## 停止點

三個 B seed 的 12 個訓練 quarter 已完成，模型／optimizer 延續及輸出 SHA 通過。每組 6,144 個不同 TRAIN 問題、1,536 次更新；BM25 全背景評估與 PPLX dense 全文件／非 locked-test query embeddings 也完成。

`pipeline-v1/rank-dense` 在前 16 個 TRAIN queries 後，成本 canary 預估 5,185.87 秒，超過凍結的 5,100 秒門檻，依協議退出。不是執行 5,100 秒後逾時：該 phase 實際 33.61 秒即停止，peak process-tree RSS 約 3.53 GiB，沒有達到 16 GiB 記憶體上限。前 16 題均已通過兩種累加分數、top-100 及全部正例排名核對。

failed phase 的實際 offline ClearML task 關閉，owned process group 清理完成，沒有自動重試。完成的 dense encode phase 為 exit 0：重用 117,541 文件／6,144 TRAIN query embeddings，新增 115,468 文件／1,728 DEV query embeddings。沒有編碼或評分 LOCKED_TEST。

## 有界 profiling

`scripts/diagnose_ng69_dense_rank.py` 只使用已保存的 embeddings，不重新模型推論。取原前 16 個 TRAIN queries，以及在 TRAIN index 128--6143 中固定等距選取的 16 題。三個模式交錯順序測量三輪，共 **32 個不同 TRAIN queries**；不是把重複測量當成 288 個獨立問題。CPU4、RSS 16 GiB、核心診斷 wall budget 240 秒，不用 GPU、不讀取 DEV/LOCKED_TEST query rows。

所有模式保留雙路 float64 分數核對，絕對誤差上限 `1e-12`，top-100 tie-break、每一個正例 rank 完全一致。原停止點 16 題另外與保存的排名逐題比對。輸入／輸出 SHA、實際開始且關閉的 offline ClearML task 均保留；Mac 小型診斷輸出副本已通過完整 SHA。

| 模式 | 每 16 題平均總秒數 | 獨立累加平均秒數 | 最慢測量的全量保守預估 |
|---|---:|---:|---:|
| 原方法：逐 query matvec + 分塊乘積再 sum | 7.317 | 6.213 | 5,447 s |
| 相同主分數 + 直接獨立 reduction | 3.856 | 2.750 | 2,855 s |
| batch16 主矩陣乘法 + 直接獨立 reduction | 3.280 | 2.749 | 2,428 s |

診斷共約 87.8 秒，peak RSS 約 2.44 GiB。原方法約 84.9% 時間用於第二次累加。避免乘積暫存後，同一診斷 harness 的平均總時間下降約 **47.3%**；batch16 相對原方法下降約 55.2%。表中包含全排名與獨立核對，不是純 GEMM throughput。

診斷把 16 個主要分數存成一個結果陣列，與原 pipeline 即時計算一題再核對的排程略有不同。因此診斷原方法的 5,447 秒與原停止時的 5,186 秒不是同一個計時樣本，不將差值歸因為模型退化。各優化模式使用同一診斷流程，且順序輪換，降低快取／順序造成的偏差；仍不是已完成的全量執行結果。

## 原因與最小修正方向

原獨立累加會對每題、每個文件塊建立 `documents_block * query` 的 float64 暫存，再沿維度相加。對 233,009 × 1,024 的文件矩陣，每題累計產生約 1.91 GB 的乘積暫存資料流；不是同時占有 1.91 GB 額外 RSS。分塊限制 peak RSS，但沒有消除重複配置、寫入和再次讀取。

以下操作仍計算相同數學內積，但不建立完整乘積塊：

```python
alternate = np.einsum('ij,j->i', documents, vector, optimize=False)
```

建議優先採用 **只替換獨立驗證累加** 的方案，保留原 `documents @ vector` 主分數、float64、每題全部排名檢查及 5,100 秒門檻。這比連主要分數也改成 batch GEMM 更小；樣本中 direct reduction 最大分數差約 `7.78e-16`，batch16 最大約 `1.00e-15`，兩者皆通過嚴格 rank 核對，但本輪無需為額外約 15% 的時間節省增加主路徑變量。

這不是新增模型架構或解決泛化瓶頸的證據。沒有重新訓練、換 teacher、調 mixing／RMS、改 qrels 或放寬驗證門檻。

## 保存與繼續條件

原 `pipeline-v1` 保持不可變，包含成功訓練、成功基準與失敗階段。停止後的完整 inventory 為 231 個一般檔案、8,168,895,409 bytes；Mac 鏡像 exit 0，全量 size/SHA、65 個 frozen inputs 與 14 個成功階段核對通過。`pipeline-v1-copy-verification.json` 保留校驗回執；失敗階段完整保存，遠端來源保留。沒有重新執行完整模型推論的 restore test。

依原協議「診斷後另行授權新 attempt」的界線，本輪只完成診斷，未重啟完整 pipeline。若確認續跑，可建立獨立 evaluation attempt，明確綁定原成功階段與 checkpoint SHA，只從失敗的 dense ranking 繼續；不要重跑三組訓練或已成功的 embeddings。新 attempt 要以相同 TRAIN-only canary、原成本上限和完整排名核對重新驗收，再跑未完成的 A/B 評估。

額外監督／人工複核、模型泛化與 native posting 成本仍須等待正式 A/B 結果判定，不因這次評估器成本失敗改變研究方向。

後續使用者已授權繼續研究；新 [evaluation-v2 協議](ng0069-evaluation-continuation.zh.md) 明確重用成功階段，只續跑評估，不改寫這次失敗證據。

## 重現與已知界線

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python scripts/diagnose_ng69_dense_rank.py \
    --run "$NG69_PIPELINE_V1" --output "$NG69_NEW_DIAGNOSTIC_DIR"
python -m unittest discover -s tests -p 'test_ng6*.py' -q
```

研究 fixtures 為 41 項通過，另有 32 題實際全背景數值／排名診斷；沒有重跑產品完整回歸。第一次診斷預檢因缺少可選 `threadpoolctl` 結束，尚未計算或建立 tracking；已保留來源與失敗記錄，改成只記錄既有 NumPy 與 thread environment，不安裝或改動遠端 runtime。成功診斷 source 另存版本，不覆蓋失敗預檢 source。
