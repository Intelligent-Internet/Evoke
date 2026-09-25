# NG-0088 第一階段執行契約：全背景與固定預算

日期：2026-09-20。研究範圍依 [品質保留／成本計畫](ng0088-quality-retention-cost-plan.zh.md)。此文件和 source、資料清單在首個 phase 前一同凍結；執行結果另記，不覆寫原始契約。

## 執行與資料

- 使用 Lambda2 **實體 GPU 0**，`CUDA_VISIBLE_DEVICES=0`，核對已登記 GPU UUID、空閒狀態與 cooperative lock。這是使用者最新資源限制，覆蓋前一版 plan 的多卡並行建議。GPU 1/2/3 不使用、不 fallback、不搶占他人工作。
- Base、NG87-A/8192、B/8192 順序執行。每模型 290,609 documents 分兩個固定連續 shard；另編碼 128 TRAIN calibration 和全部 2,048 exposed-validation queries。既有 prefix、microbatch、FP32 readout、RMS、BM25、0.1/0.9 weighting 不變，沒有 optimizer update。
- 依賴 NG87 `clean-base-ab-v1` manifest `27d5965fc24524ca6ab90c7bcf60d10b931d029aaa9f8357990e646296f7b361`。父任務與各 phase 必須關閉並完整 hash 驗證，所有消費的模型／資料／receipt 另加入新 manifest。
- 不讀 locked test，不重新編碼 PPLX，不新增 supervision。此背景是 NG85 固定 corpus，不是全 BEIR15；validation 已多次曝光，不宣稱獨立泛化。

## 23 個固定 Phase

每個 arm 先做兩段 documents 與一段 queries encoding。Base 只評估 full；A/B 各評估 full、Q96/D768、Q64/D384、document-only D384、query-only Q64。最後為 TRAIN dense+BM25 calibration、validation baselines 和 terminal review。

TopK 按 raw 正值大小、token ID tie-break；先 query mask 再固定 RMS normalize，不重估 RMS，不補零成 posting。無法正規化即失敗保留，不跳過 query。首個 encode batch 必須與原 frozen Encoder 完全一致；full profile 的既有候選 raw scores 另以 `rtol=atol=1e-5` 核對，不能只比較名次。

全背景使用 float64 sparse dot、全 universe deterministic top-100，全部 known-positive ranks 以完整 corpus 計算。所有正例留在 Recall 分母；記錄完整 top-100 IDs/scores。每題對 top-100 與正例獨立做 row-dot audit，review 再檢查排名／metric receipt。

固定 baseline 是純 BM25、同 prefix PPLX，以及 TRAIN-only 校準的 dense+BM25。只在已凍結的 128 TRAIN queries 上從 `alpha=[0,.1,.25,.5,.75,.9,1]` 選 macro nDCG@10，其次 Recall@100，再以較小 alpha tie-break。這個 baseline 使用全背景 z-score，是離線品質診斷，不冒充可部署 serving route。正常原文 dense 另待獨立協議，不能拿 prefix dense 當完整產品上限。

## 成本與決策

輸出 semantic／lexical postings、document/query NNZ、DF、matching union、combined literal DF-work mean/p95。這些不等於 native index bytes、WAND decoded postings 或實際延遲。

A/full 對 base/full 的宏平均 nDCG@10 增益至少 0.005、5,000 次 domain-stratified paired bootstrap 描述區間下界大於 0，且 Recall@100 點差不低於 -0.002，才建議凍結後續 R/P 預算內訓練。此階段不自動启动尚未凍結／測試的訓練程式。所有 profile 與 domain 結果均披露，不能事後挑中間 checkpoint 或調 budget。

## 資源與停止

- 每 phase 上限 90 分鐘，完整 campaign 上限 24 小時；前 512 篇文件或前 16 題排名作保守 ETA admission。超出即停，不偷偷延長。
- 每個 worker process tree RSS 上限 16 GiB，GPU 0 總顯存上限 20 GiB；主機 available RAM 必須大於 24 GiB、磁碟餘量大於 40 GiB。phase admission 另保留更嚴格安全餘量。
- 只有一個 phase 執行，無重試、無未註冊 phase、無資源換卡；失敗保留原始 log、exit 與 ClearML receipt，只終止自己建立的 process group。
- 沿用可審計的 ClearML offline tracking。`remote_synced=false`，不聲稱 dashboard 已同步。舊 polling 不恢復；執行器自身負責資源監控與 terminal review。

重現入口是 `scripts/ng88_eval.py`、數值實作 `scripts/ng88_math.py`、測試 `tests/test_ng88.py`。遠端凍結時將依賴 source 與此契約平鋪到新 run，由 `freeze` 建 manifest，`supervise --gpu 0` 執行。原 NG87 attempt 不修改。大型檔案保留於 NG-0088 外部 artifact 單位，Git 保留方法、測試與精簡結果。
