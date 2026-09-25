# NG-0086 完整配對評估：修復尺度梯度，尚未完成品質／成本突破

日期：2026-09-18。狀態：原定 candidate-only 配對分析已補齊；未新增訓練、未驗收產品模型。

依據[原介入協議](ng0086-std-gradient-intervention.zh.md)及[另行授權的評估延續](ng0086-evaluation-continuation.zh.md)，本輪從原 checkpoint 補齊 control/base/start 的 2560 個評估 row，與原 full-std 的 1920 row 配對。512 題 validation 每域 128 題，128 題 fit 每域 32 題；split、候選、readout 和主要 checkpoint 2816 均未變。此前只有 64 題歷史重疊比較，不能將其均值與本次 512 題混比。

## 1. 主要結果

全部為同一候選集、完整 BM25+SAE hybrid 分數、四域等權的 512 題 validation。文件 NNZ 是候選文件的 query-weighted 均值，不是全庫去重後的索引大小或 DF。

| 狀態 | nDCG@10 | Recall@10 | Document NNZ | Zero-RMS query mass |
|---|---|---|---|---|
| 原成熟底座 | 0.647473 | 0.757484 | 290.03 | 0.82% |
| 介入前 B/2048 | 0.637742 | 0.747074 | 2564.25 | 54.02% |
| Control/2560 | 0.631546 | 0.726450 | 2753.67 | 74.68% |
| Full-std/2560 | 0.657995 | 0.752400 | 1008.28 | 21.23% |
| Control/2816 | 0.599128 | 0.707677 | 2397.07 | 83.60% |
| **Full-std/2816，主要點** | **0.656405** | **0.755557** | **798.08** | **15.99%** |
| Full-std/3072，無成對 control | 0.660089 | 0.757739 | 752.74 | 12.67% |

同候選集 PPLX dense 參考 nDCG 為 **0.745786**，BM25 為 **0.461216**。它們是既存 bank 分數，不是新增正常全文／全語料的產品 baseline。

預定主要比較 Full-std/2816 - Control/2816：**+0.057277**，相對約 **+9.56%**，固定 5000 次域內配對 bootstrap 的描述性 95% 區間為 **[+0.037804, +0.076601]**。2560 的次要成對差為 +0.026448，區間 [+0.009803, +0.043617]。不選 3072 當最佳成對 endpoint，因 control 根本沒有該 checkpoint。

對主要點另作已指定 anchors 的描述性比較，沿用相同 resampling 方式：

| Full-std/2816 減去 | nDCG 差 | 描述性 95% 配對區間 |
|---|---|---|
| 原底座 | +0.008933 | [-0.005765, +0.023612] |
| 介入前 B/2048 | +0.018664 | [+0.004185, +0.032987] |
| 同候選集 PPLX | -0.089380 | [-0.108975, -0.068568] |

**結論不是單純「沒崩潰」：修正後相對原 loss 同時改善了排名、召回和 NNZ。** 但相對原底座的小幅 nDCG 優勢仍不確定，Recall 略低，NNZ 約為底座 **2.75 倍**。因此尚不能說超過成熟底座的品質／成本前沿，更沒有超過 dense。

512 題上「略高於底座」與先前 64 題上「仍低於底座」並不矛盾：兩者是不同大小的固定 panel，本次逐題一致性已驗證，不能據此改挑有利子集。這也支持不用很小 panel 的單一均值裁決整條訓練路線。

## 2. 得失與邊界

Control 在第 2847 步重現同類有限、非空但完全 zero-RMS-support 的失效；Full-std 完成至 3072，所有已評 checkpoint 都沒有 invalid query。主要品質差不是刪除無效 query 或將其排除分母造成。

主要點相對 control 的分域 nDCG 差：FEVER -0.007105、HotpotQA +0.139410、NQ +0.050805、FiQA +0.045998。相對原底座則為 FEVER +0.050681、HotpotQA -0.008603、NQ -0.042554、FiQA +0.036207。按既定 overall 目標，不要求每域都勝出，但完整披露 NQ 等損失，不做事後 teacher routing。

128 題 fit panel 上，原底座、起點、control/2816、full-std/2816 的 nDCG 分別為 0.664072、0.660156、0.614708、0.673771。訓練題和 validation 的方向都支持修復尺度／支持漂移，不是僅依賴一個 validation 峰值。

仍保留以下界線：

- 這是單一 order seed、從已漂移 B/2048 出發的晚期介入。不能保證從底座開始的完整兩 epochs 或其他 seed 都穩定。
- 全部 validation 已曝光，bootstrap 僅描述當前 query 集，不是獨立泛化證明。沒有讀取 locked test。
- Full-std 並未在數學上消除 fixed RMS 的 zero-support 邊界；本窗口不失效不代表此風險永久不存在。
- NNZ 相對 control 下降約 66.7%，不等於 posting bytes、DF-work 或原生延遲下降 66.7%。本輪沒有測試那些成本。
- 原 NG85 並未完成原本 PPLX/RankT5 teacher A/B，所以仍不能拿該失敗宣判任務排序 teacher 無效。

## 3. 執行與重現證據

新 evaluator 為 [`ng86_evaluation.py`](../../../../scripts/ng86_evaluation.py)，新測試為 [`test_ng86_evaluation.py`](../../../../tests/test_ng86_evaluation.py)。沿用原 frozen summarizer，未改主要步數、bootstrap seed 或 invalid-row 處理。

`evaluation-v2` 的 manifest SHA256：`10ae379dc050a3783861d0ae9fc4d80c5e09d5174af2f7655c0210dd4e2eb33b`。評估 worker 在 GPU 3 使用 **504.79 秒（8.41 分鐘）**，低於不變的 1800 秒硬上限，exit 0、owned group closed；2026-09-18 16:50:11 UTC 完成配對 review。ClearML offline task 已關閉，沒有 server-sync 聲明，也沒有後續 GPU 工作或 polling 自動啟動。

- 四域 fit/validation 共 8 個 full-std anchor，及原失敗 profile 的 16 個 base anchor，原始分數最大差均為 **0.0**，nDCG/Recall 一致。
- 對 4480 個原始逐題結果，用 Python `sorted((-score, document_id))`、binary relevance、`math.log2` 另行重算 nDCG/Recall，最大絕對誤差為 **2.220446049250313e-16**。未使用原 NumPy metric helper 重複冒充獨立實作。
- 新 control profile 的 18 個 payload、review 的 3 個 payload、18 個 frozen source/protocol payload 與 manifest 已鏡像到本地大檔區並逐檔驗證 SHA。原 full-std profile 13 個 payload 與 failed control profile 10 個 payload 也已完整鏡像驗證，failed 狀態未更改。舊大型訓練 checkpoint 尚未在此輪宣稱已全部本地備份。
- 新 profile receipt SHA：`41bcea7f48b887aeb2b7aa28c00b2a467c8174e3c7e4b2493053ce0e0454e0ac`；review receipt SHA：`e581b9a1133d892a081f72996785074e284cfaf844303c3930ba029a00b7c418`。
- Mac 相關回歸 **40 passed, 5 skipped**；Lambda2 含 Torch 測試 **20 passed**；Ruff、文件連結及 whitespace 檢查通過。Mac 沒有執行模型推論。

估時說明：parity 的初始化／首次 CUDA 成本包含在已耗 wall time；之後每個 label 的逐題樣本才用於剩餘 row 外推。也就是成本計一次，而非把固定成本乘上整批題數；凍結協議中「留在樣本內」應依此執行明細理解，不是將 parity warm-up 重複計入每題成本。

## 4. 下一階段建議，尚未啟動

不再延長 B/2048 的救援窗口，也不繼續掃 alpha、增加 cost penalty 或做另一輪同類支持診斷。這次足以將正確 std derivative 列為後續訓練的共同預設候選，但還不足以發布新模型。

下一個需要裁決的問題是：**從乾淨的成熟底座開始，排除這個已辨識的梯度問題後，原本的多域排序監督能否真正推進完整 hybrid？** 合理的後續是重新凍結同資料、同預算、共同 full-std 的 PPLX／RankT5 比較，恢復 NG85 尚未完成的科學問題，而不是再把已知不穩定的 detach loss 當主要研究主線。

新階段必須另定完整訓練／評估預算及第二 seed 規則，持續監測 fixed-RMS 支持但不偷偷加 epsilon 或改 RMS；若再次失效，直接保存失敗，不無限續修。小 panel 只作訓練監測，晉級判斷需同一 full-corpus 面上的原底座、pure dense、dense+BM25 以及真正 posting/DF/native cost。這是下一輪設計方向，不是本次已啟動或已證明成功的工作。
