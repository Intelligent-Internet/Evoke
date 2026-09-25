# NG-0083: 排序目標與監督瓶頸的因果拆解

日期：2026-09-14。接續 [NG82 完整結果](ng0082-progress.zh.md)。本文件隨 A/C 首次執行凍結；後續進度與結論另記，不回寫協議。B 的數值設定與 source 另凍結，沒有默認啟動大規模 encoder 訓練。

## 問題與不再重複的路線

總目標仍是 **BM25+SAE overall 超過 dense 並控制檢索成本**。本輪不是生產晉級測試。NG82 已排除「TRAIN pair surrogate 改善必定改善排名」及「保留大部分二階能量就保住頭部排序」；不再細掃 alpha、PCA 維度或 individual-utility pruning。

NG53 的 feature utility 非加性、M401 的 learned tail decoder 失敗、M409 的 random-hash text-to-tail 失敗仍是反證。NG80 S/P 同時改了 score surface、loss 與 temperature，不能作為單因素 ranking-weighting 實驗。12,288 個三域問題與 1,536 個已曝光 diagnostics 不代表 BEIR15 泛化。

## A: 固定 mask 的同 TRAIN 全 corpus 重播

- 父產物 `NG-0082/diagnostics-v1`，inputs SHA `5f591075a122242f85b126abe3d4d12a62da974424978dcaf70dc360e2605804`。
- 獨立 review SHA `d566c47b45cad82704bff5341aa7b042e5ecd1bf4090b418c70705291fe4fee6`。
- 僅 384 個原 calibration TRAIN queries、233,009 documents；S、utility-half、DF-half 原 mask 不重新設計。
- query cache 已含 semantic 0.9 / lexical 0.1，直接相加。不再次加權、不重算 RMS、不校準 alpha。
- 每 query 保存所有 human-positive ranks、top100、nDCG@10、Recall@100、literal DF work、原六組 teacher pair 的 joint loss。原 pair ID、temperature、joint delta 必須重播一致至 1e-12。
- 額外原始 full teacher 排名是定位參考，不參與 mask 選擇。沒有新增梯度或 feature 篩選。
- 預定比較僅 utility-half 與 DF-half 各自對 S；使用三域分層 paired bootstrap，2,000 次、seed 82082，分別報 TRAIN 與既有 diagnostic。
- 若 TRAIN human ranking 已退化，proxy-to-ranking mismatch 已在樣本內發生；若 TRAIN 改善而 diagnostic 退化，支持泛化問題。若 CI 不明確或兩者混合，如實標為未分清，不能強行二分。teacher preference 不是人類負例，未判定文件不得改叫 negative。

## B: 相同容量、只改關係 weighting

先在 teacher cached vectors 的可審計 rank-256 score operator 上實驗，固定相同 TRAIN queries、全 corpus teacher 定位的 candidate bank、兩個角色的初始化、optimizer、更新次數與計算流程。不是宣稱 rank-256 等同 256 sparse NNZ。teacher 真值始終保留完整 1024 維。

令 error 為 `e_i = s(q,d_i) - t(q,d_i)`，有精確恆等式：

$$
\frac{1}{m}\sum_i(e_i-\bar e)^2
=\frac{1}{2m^2}\sum_{i,j}(e_i-e_j)^2.
$$

所以 centered score-field 本身已經是均勻 pair-margin regression。新路線不是把 MSE 換個名字，而是 **在相同 candidate bank 上，把有限權重從平均 pair 分配給 teacher 頭部與 top-100 邊界**，保留背景 anchor。兩臂走同一程式路徑，只改一個 mixture coefficient。禁止無界 inverse-margin、測試集調參、discarded training decoder。

若 teacher margin 為正，`abs(student_margin - teacher_margin) < teacher_margin` 是維持順序的充分條件；全局 MSE 小並不提供每一關鍵 pair 的保證。這是數學動機，不是實驗成效。參考 [Margin-MSE](https://arxiv.org/abs/2010.02666) 的跨架構 margin 蒸餾及 [RankDistil](https://proceedings.mlr.press/v130/reddi21a.html) 的 top-ranking 對齊；本輪不是其原演算法重現。

同時報 TRAIN 與 diagnostic 的 full-corpus pure/hybrid quality、teacher head/boundary order、global score error、匯出 parity、資源和 exposure。不以 panel quality 或 TRAIN loss 宣告成功；B 不直接解決 DF、encoder 或 sparse-support 容量，通過也只能准入下一個 SAE transfer 小實驗。

## C: 監督準備，不冒充已取得十萬條新資料

盤點同一 TRAIN/diagnostic 的來源、NFC/空白/casefold exact duplicates、正例 document overlap、長度與 query 首詞分布。首詞不是經人工確認的 intent/topic；exact 去重不代表 semantic 去重；query split 不代表 source/document/topic split。

另列候選人類監督来源、原始授權、可用 TRAIN split、已曝光重疊、teacher-human disagreement 的標註需求和 source-disjoint 計畫。只有 positive qrels 時，不估計真負例錯誤率。資料下載、near-duplicate 模型、十萬級編碼與 locked-test 評分均不包含在這次 metadata audit 中。原始碼、方法與小報告留主倉庫。

## 執行限制

A/C 在已有資料的 Lambda2 獨立 CPU lane 執行，各 4 threads、RSS 16 GiB、5,400 秒，host available RAM >24 GiB、disk free >40 GiB；沿用 process-group closure。首 32 題推算 `1.5 * 全程估計 + 300 < 5100` 才繼續。新 source、input、mask、selection、closure 均 SHA 綁定，失敗保留另開 attempt，不放寬限制。

ClearML 使用 actual-start/closed offline receipts，不宣稱 online sync。沒有部署、重建索引、模型晉級、資料刪除或恢復週期 polling。後續 GPU lane 必須獨立凍結、測試、檢查空閒並顯式綁定單 GPU；不與已有任務搶 GPU。
