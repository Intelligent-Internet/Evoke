# NG-0080：Teacher 可觀測性與監督連通性審計

2026-09-13，新增審計結果產生前固定。這是 [NG80 聯合診斷](ng0080-bottleneck-campaign.zh.md) 的獨立 CPU 分支。只用已閉合的 TRAIN query、完整無標籤 corpus embedding 和 token 長度；不改模型、不讀 DEV/LOCKED query、不補人工標籤。

## 固定比較

1. Teacher query 的未中心化 second moment：舊 1,536 題、新 TRAIN 中每域 hash 選 512 題、全部 12,288 fit 題。新 1,536 題的 hash salt 固定為 `NG80/geometry-fresh-v1/`。報 eigenvalues、participation/entropy rank 和固定維度累計能量。舊題集可能包含本輪 diagnostic 身分，但不參與新模型梯度；本分支僅描述歷史覆蓋，不用于挑參數。
2. 對舊題 top128 方向的剩餘能量：比較同樣三個集合。這是 score 可觀測性的代理，不代表 task taxonomy，也不是 generalization 證明。由固定問題推得的 rank 不能替代新任務評估。
3. 正例保護圖：每題固定 panel 的 positive/nonpositive，僅當 teacher margin>0 時有邊。非正例未被標為硬負例。報連通分量、邊數、target>=.99 比例。若圖有 c 個分量、n 份文件，可識別的獨立精確 margin 約束為 n-c；完整分數場為 n-1。因此「全配對很多」不代表新增同等多的獨立資訊。此處忽略原 D metric weights 及實際模型梯度，不能冒充舊訓練回放。
4. 每域 query64 截斷數及 positive document256 超長曝光數。沒有 evidence span，僅稱輸入窗口風險，不判定正例信息一定在窗口外。

## 執行

`scripts/ng80_supervision.py` 先驗證 `readout-v1` input SHA 及 prepare/teacher/hidden 三個 phase 的退出、tracking、內容 SHA，再以新目錄凍結輸入與來源。只在新單元內執行 `audit`。沿用 CPU4、RSS16GiB、host available>24GiB、free>40GiB、5400秒的 process-group guard。實際開始的 ClearML offline 記錄必須閉合；不宣稱 online 同步。

本分支不依結果重選問題或 eigenvalue cutoff。結果與 F/D/S/P 梯度對照、更多合法 TRAIN 來源、真人標註及 native 成本分開判讀。支持或反對某一數學代理，都不等於找到了唯一因果。
