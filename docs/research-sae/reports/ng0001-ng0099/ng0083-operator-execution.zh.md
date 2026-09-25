# NG-0083-B: Matched Score Operator 執行協議

日期：2026-09-14。隸屬 [NG83 因果拆解](ng0083-ranking-causality-protocol.zh.md)。凍結後不因結果改寫。

## 固定資料與容量

沿用完整 PPLX 1024 維 teacher cached vectors，233,009 documents，NG80 的 12,288 TRAIN queries。不重新編碼文本、不更新 NG3/S/P encoder。僅更新 rank-256 的 query/document 兩個 linear maps，各 1024x256，共 524,288 個參數，初始化都為 NG82 PCA256 maps。保留 document centering mean，query 不中心化，無 post-projection L2 normalization。

共同 candidate bank：每 TRAIN query 在完整 corpus teacher 排名上取 0-based positions `0..15,19,29,39,49,59,69,79,89,99,100,127,199`，合併既有 NG80 panel（含固定背景）及全部 human positives，去重；每 query 上限 256。沒有用 diagnostic 決定 bank、scale、訓練 order。teacher target 為原始 dot product，独立 reduction parity 1e-12。

## 唯一處理差異

共同 field loss 為 candidate 內 centered score-error MSE。共同 boundary loss 為六組 teacher pair `(0,9),(9,10),(9,49),(49,99),(99,100),(99,199)` 的 margin-error MSE 乘 0.5。两者除以同一 TRAIN candidate teacher variance 的平均值，該 scale 凍結。

$$
L_\lambda = [(1-\lambda)L_{field}+\lambda L_{boundary}]/\sigma^2_{TRAIN}.
$$

- field arm：lambda=0。
- boundary arm：lambda=0.5，保留一半 field anchor。

這是 bounded weighting 的 margin regression；沒有 sigmoid saturation、unbounded inverse-margin、teacher preference 充當 human negative。兩臂即使 lambda=0 也計算相同 components，僅 scalar weighting 不同。有限近 tie 不額外放大。本輪不聲稱六組 pairs 已足夠覆蓋所有排名關係。

## 訓練、驗證與停止

相同 seed83083、同一 TRAIN permutation、每 update 4 queries、單 epoch 3,072 updates、AdamW lr1e-4、weight_decay0、clip1、FP32、TF32 off、deterministic algorithms。GPU0/3 各一獨立單卡任務，**不是 NVLink distributed job**。無多 seed、lr grid 或中途根據排名 early stop；固定最後 endpoint，結果差也保留。

首 16 updates 是固定訓練的一部分，僅用於 finite-gradient/resource canary，不用於選方法或重置重跑。推算超過原 5,400 秒 bound 即停，保留 attempt。每 phase RSS<=16GiB、GPU total<=20GiB、available RAM>24GiB、disk free>40GiB，各4CPUthreads。prepare、fit、rank 分別有 source-bound supervisor/exit/closure。

保存兩 maps、完整 optimizer state、每 update loss/components/gradient norm、exact query exposure、bank/init SHA。匯出使用相同 query map 與 document map，不能丟 decoder；最末實際 batch GPU FP32 vs 匯出 FP64 score tolerance1e-5，序列化 map 位元及 optimizer step 必須一致。完整 corpus 排名用 FP64，並保留原 initial/dense 重播對照；數學等價性與梯度另有 CPU tests。

排名使用 384 TRAIN calibration 與 1,536 已曝光 diagnostic，報 pure 與固定0.9/0.1 full-corpus z-score hybrid、所有正例 ranks、nDCG@10、Recall@100、global centered error、六組 head/boundary margin error及order retention。z-score 是離線診斷而非線上建議。macro 固定三域等權，2,000 次 query paired stratified bootstrap。

預定 primary comparison：diagnostic `boundary_hybrid - field_hybrid` 的 nDCG paired CI 下界>0，Recall CI 下界>=-0.005；還需 `boundary_hybrid - initial_hybrid` 的 nDCG 下界>0、Recall下界>=-0.005。通過僅支持往 SAE transfer 做下一個小對照；未通過不掃更多 lambda。dense差距、TRAIN/diagnostic差異、teacher order需一起報告。沒有 native成本、DF改善、獨立holdout或overall BEIR15資格。
