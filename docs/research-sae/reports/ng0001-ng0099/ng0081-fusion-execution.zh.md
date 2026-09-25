# NG-0081A：固定 endpoint 的對稱融合執行契約

2026-09-14。使用者已要求 review 歷史後開始探索。本契約只啟動 [NG81 設計](ng0081-overall-hybrid-quality-cost-plan.zh.md) 的固定模型對照，不啟動梯度訓練或自動輪詢，不使用 locked test。

## 新資訊與不重複範圍

M550/M1951 已有固定融合的 macro 正例；M160A/M1329 已有 RRF、rank interpolation、gate 與 residual 的限制。M1518 的 DF-FLOPS/Budget-matched 失敗亦保留。這一輪不是重新尋找所有融合方法，只補 NG80 新 S/P endpoint 未曾做過的對稱 TRAIN 校準，並量測其精確 support touch。完成後停止，review 決定是否訓練，不做更細 alpha sweep。

## 固定輸入與計分

- Parent：NG80 `sparse-endpoints-v3`，input SHA `aa98a6bf917c2e5e0ea901d0fd20223f2a33ecc974a3120aba6f0a6b6f123642`，review results SHA `365aad14fe4738786d7d38f699333a026e7932a51e51db8910ed50f0133814f7`。
- 重用其已閉合 initial/S/P 文件編碼及 1,536 題診斷 query code。重新編碼的只有相同 384 個 TRAIN 校準 query，各模型一次；不重建文件或產品索引。
- 原模型與 S/P 權重、tokenizer、query64/document256、RMS、FP32 eval、TF32 off 全部不變。零 optimizer 更新。
- 同一完整 233,009 文件背景；FEVER/HotpotQA/NQ 各128題 TRAIN 校準、各512題歷史曝光診斷。前者允許 TRAIN 曝光，與後者不重疊；不是独立泛化測試。
- 每個模型、每個 normalization 從 alpha `[0, .1, .25, .5, .75, .9, 1]` 中以 TRAIN macro nDCG、Recall、較小 lexical 權重選一次；沒有逐域或逐題選型。
- 原 cached semantic/lexical 分數已含 0.9/0.1。Serving 對照必須使用 `((1-alpha)/0.9)*semantic + (alpha/0.1)*lexical`，而非再次直接乘 0.9/0.1。alpha=.1 必須精確重現 NG80 hybrid。
- Z-score 對照在完整背景分別正規化後混合；它是離線品質對照，不宣稱是可部署的低成本 normalization。
- Dense/BM25/dense-hybrid 重用 NG80 同一份已驗證的全背景排名與 dense TRAIN 校準，不重跑相同 baseline。

## 八個 phase

1. `encode-calibration`：一張空閒 GPU，依次編碼 initial/S/P 的384題。
2. `calibrate-initial`：固定 grid 的兩種 normalization。
3. `calibrate-S`：相同流程。
4. `calibrate-P`：相同流程。
5. `rank-initial`：原配置、serving 校準、z-score 校準三組全背景排名。
6. `rank-S`：相同流程。
7. `rank-P`：相同流程。
8. `review`：獨立 TRAIN 選擇重算、全部正例 metric/head 對齊與 CSR coordinate 計分核對、macro/分域及配對區間。

Raw semantic/lexical 分數使用 NG80 的兩條獨立全背景 reduction 檢查。融合公式另有 CPU contract tests；正例 rank 再用直接比較計數檢查，不只是從同一排序陣列讀值。Review 為獨立 metric/coordinate 重算，不宣稱第二次獨立全庫排序。最終 historical profile 的正例 ranks、top100 IDs/scores 必須精確等於 NG80，否則保留失敗、停止。

報告 query/document NNZ、literal DF-sum、semantic/lexical/union 精確 touch、universal/max DF。Touch 使用未做 z-score 的非負通道 support；不是把均值平移後的非零分數算成 posting。該 work 表描述已編碼的雙通道基礎，不對 alpha=0/1 宣稱兩路仍為必要成本；實際單通道可跳過的工作另記，不把離線全掃時間當 native 延遲。

## 執行安全與停止

每階段沿用 5,400 秒、16GiB process-tree RSS、24GiB host available、40GiB disk free、GPU總用量20GiB限制。16題 ranking/calibration canary 用 `1.5 * elapsed * total / 16 + 300` 必須小於5,100秒，不通過就保留階段，不擅自放寬限制。GPU編碼按64題作估時，每個模型計時獨立，整個phase仍受總時限約束。

GPU僅在編碼phase鎖定，之後釋放；CPU phase 使用既有共享 CPU lane lock，每次最多4個計算執行緒。ClearML actual-start/closed 記錄採 offline，不能宣稱 server 已同步。所有 phase 須 exit0、無 error、owned group closed、精確檔案 inventory/SHA 與 tracking closed/passed。輸入與凍結 source 每階段前後驗證。

配對區間沿用 NG80 分域 bootstrap，只描述這組已曝光診斷 query，不作產品 promotion 或多 seed 泛化宣稱。評估以 overall hybrid 為主，分域披露但不要求逐域勝出。完成後先區分校準收益、剩餘表達差距和 cost 機制，再決定下游實驗。
