# NG-0082 execution progress

更新：2026-09-14 21:21 UTC。兩條 CPU lane 與獨立 review 全部閉合，新產物已在 Mac 完整鏡像並核對 SHA。**沒有方案同時通過預定品質與成本門檻，不晉級、不擴大這兩套配方。**

## Frozen experiment

- [文獻分析、數學與執行協議](ng0082-potion-review-and-execution.zh.md)。
- 原始碼：`scripts/ng82_diagnostics.py`；測試：`tests/test_ng82_diagnostics.py`。
- 執行主機 Lambda2，經 Spark1 ProxyJump；資料已在該機，避免大檔跨機傳輸。
- Run：`NG-0082/diagnostics-v1`。
- Input SHA256：`5f591075a122242f85b126abe3d4d12a62da974424978dcaf70dc360e2605804`。
- 新測試與 NG81 regression：`42 passed`；加入獨立 reviewer 測試後 `47 passed`。第一次組合測試的 1 個失敗是隔離測試樹缺 NG81 frozen protocol fixture；補齊 fixture 後全通過，未改舊測試或數值容差。
- 啟動時 available RAM 117 GiB、disk free 647 GiB，四 GPU 均空閒且本輪不使用。
- 每 lane 4 threads、16 GiB RSS、5,400 秒；沿用 process-group supervisor，不放寬停止條件。
- ClearML offline actual-start：mask `offline-17e5f30a114c4b2ebe5e152cca1c8bee`；geometry `offline-7a1a7a358d54439ebff7f90ea2d15a6a`。兩者已 closed/passed，未宣稱 server 同步。

## 完成與核驗

A 使用 384 TRAIN queries 設計 mask；B 使用 12,288 TRAIN queries 與 65,536 個無標註 corpus documents 擬合變換。兩者在相同的 1,536 題、233,009 documents 上完成全量排名。這是已曝光三域 diagnostic，不是獨立 test 或 BEIR15 overall。

| Phase | Wall time | Peak RSS | Status |
|---|---:|---:|---|
| geometry | 543.49 秒 | 6.08 GiB | exit0、owned group closed、controller complete |
| mask | 1,378.32 秒 | 9.69 GiB | exit0、owned group closed、controller complete |
| independent review | 另一次有界只讀分析 | 未獨立測量峰值 | exit0、sealed |

`scripts/review_ng82_diagnostics.py` 核對 closure、所有 positive ranks/top100、macro quality、paired CI、門檻與 teacher head/boundary margins。Mac 以 stdlib 另重算 26,112 組 nDCG/Recall，容差 1e-12；31 份 source、mask 15 份 payload、geometry 19 份 payload、review 3 份 payload 的 exact inventory/SHA 均通過。新產物約 40.7 MB，沒有重複搬運歷史模型與 corpus。

| Artifact | SHA256 |
|---|---|
| mask/results.json | `315f8e4918e01bab1a3f6858c8a95bee869199df94373167203f02717737a24e` |
| geometry/results.json | `244cc275fc4d4e42758ff323062c52e9c63eaa1c581881df6ebcf7a61e8b6835` |
| review-v1/results.json | `d566c47b45cad82704bff5341aa7b042e5ecd1bf4090b418c70705291fe4fee6` |
| review-v1/complete.json | `b7a728be2829da491c65658a4a551376a1e7271013059bb3d51aa5fdc251b0b0` |

## A: 縮小掃描不等於保住排序

成本是 semantic+BM25 的 literal DF traversal，相對未改 S；不是秒數或物理 I/O。BM25、融合係數、serving RMS 不變。

| Profile | nDCG@10 | Recall@100 | Total DF work / S |
|---|---:|---:|---:|
| 未改 S hybrid | 0.786079 | 0.951925 | 1.0000 |
| 刪除 DF>=90% 的 13 個 features | 0.785336 | 0.952467 | 0.8991 |
| 刪除 DF>=50% 的 251 個 features | 0.770191 | 0.946689 | 0.4499 |
| DF 排序，半數 semantic work target | 0.776869 | 0.948583 | 0.5188 |
| utility 排序，同一半數 target | 0.744317 | 0.936851 | 0.5231 |
| DF 排序，初始模型 work target | 0.731942 | 0.920475 | 0.1303 |
| utility 排序，同一初始 target | 0.686663 | 0.908426 | 0.1329 |

1. **13 個極高 DF features 有小幅清理價值。** nDCG 差 -0.000743，CI [-0.002859, +0.001265]；logical work 減少約 10.1%。但未達 30% 成本改善門檻，union 仍幾乎全 corpus。這不是已修好高 DF，也不直接部署。
2. **高 DF 不能一律視為無用。** DF-half 的 nDCG 差 -0.009210，CI [-0.014009, -0.004636]。回到初始 work 時，S 被刪到 0.731942，低於原模型 hybrid 的 0.770716。不能把更密的新模型裁回舊成本就當進步。
3. **utility proxy 比同成本 DF 更差。** utility-half 的 TRAIN joint teacher-pair loss 下降 0.095372，但 diagnostic nDCG 下降 0.041762、Recall 下降 0.015073。這不只是 additive approximation 錯；連真實 joint surrogate 改善也未轉成最終排名。
4. **不能唯一歸因於過擬合。** TRAIN pair surrogate、teacher preference、人類 qrels、完整 corpus ranking、新 diagnostic queries 是不同表面。需要補同一 TRAIN query 的 full-corpus qrels ranking，才能區分目標失配與跨 query 泛化失敗。不能斷言增加資料必定修好。

所有 diagnostic profiles 都有相同約 69,645 semantic DF work 來自 TRAIN 未出現的 features；utility 與 matched-work DF 保留了這些 features。離散刪除使實際成本略有差異，表中如實報告，不宣稱完美等成本。

## B: 二階目標保留仍會丟失排序

這是 teacher 向量的直接變換，**還沒有 text-to-SAE encoder 的誤差**。hybrid 使用固定 0.9/0.1 的 full-corpus z-score，僅是昂貴離線診斷。

| Profile | Pure nDCG@10 | Hybrid nDCG@10 | Hybrid Recall@100 |
|---|---:|---:|---:|
| 原始 PPLX 1024 維 | 0.822175 | 0.820317 | 0.971423 |
| PCA 128 | 0.658335 | 0.754494 | 0.950656 |
| query-weighted 128 | 0.660186 | 0.757352 | 0.950216 |
| PCA 256 | 0.782377 | 0.801238 | 0.962153 |
| query-weighted 256 | 0.781213 | 0.803392 | 0.960905 |

query-weighted 256 保留其 **regularized query-weighted centered-score 二階目標** 的 94.213%，不是 94.213% 的檢索能力；pure nDCG 仍少 0.040962。6,144 組固定 teacher head/boundary pairs 只有 80.143% 保持順序；普通 PCA256 為 79.997%。

同維度 weighted-vs-PCA 的 nDCG/Recall paired CI 全部跨零。weighted256 hybrid 相對 dense 的 nDCG 差 -0.018784，CI [-0.025376, -0.012399]，不符合保真門檻。停止把 global query covariance weighting 當成已有效解法。PCA 的 67.691% document variance 與 weighted 的 94.213% 是不同目標，不能比較成多保存了多少 dense 資訊。

full-rank transform 誤差 8.40e-16，獨立 operator score 重算誤差 3.61e-16，teacher 原排名重播一致。沒有發現矩陣方向、中心化或 reduction bug。**這不代表 256 NNZ 不足**：overcomplete sparse code 的非零數不等於 rank-256 線性子空間。本輪否定的是這兩種低秩 target，不是所有稀疏表示。

## 重新安排下一輪

仍值得研究，但停止擴大 scalar fusion、PCA/whitening grid、individual-utility pruning。原 NG3 底座保留；没有證據支持直接換成 Potion/BGE-M3 或從零重建 encoder。

### NG83-A: 分清 proxy 與泛化問題

在同一 384 TRAIN query 上跑完整 corpus 的 S、utility-half、DF-half，不重新選 mask、不加梯度。若 TRAIN qrels ranking 也下降，先修正 proxy；若 TRAIN 排名改善而 diagnostic 下降，才有依據將重點移往 query 多樣性與 regularization。這是目前最便宜、最能阻止誤判的分支。

### NG83-B: 相同容量的排序條件對照

保留完整 teacher 1024 維作真值，不把 PCA target 當無損 teacher。小型 frozen-input 對照只改關係 weighting：centered score-field versus 預定 head/100-boundary margin；相同 TRAIN query、candidate bank、表示容量、初始化與更新預算。先在可審計 score operator 驗證，再決定是否帶入 SAE，不同時改 encoder、dictionary support 和 loss。

排序穩定取決於 `abs(delta_score_error) < abs(teacher_margin)`，不是全局平均 MSE 小。未條件化的 `Cq` 與 `Cd` 忽略「這個 query 下哪些文件競爭頭部」的相依性。可測 bounded、未飽和的 margin regression；近乎平手的 pair 要限制權重，不用無界 inverse-margin，也不將 teacher-human disagreement 冒充可靠人工負例。這是可檢驗假說，不是已證明修復。

與 S/P 的不同是 **同一 scoring surface 的單因素 weighting 對照**；P 的 protective loss、temperature、hybrid surface 同時不同，不能當作已完成這個因果對照。M401 的反證保留；匯出必須重現訓練 score，不允許訓練靠 decoder 補償而匯出丟掉 decoder。

### NG83-C: 並行準備有效監督

建立來源、授權、query 去重、topic/intent、teacher-human disagreement 與 source-disjoint 評估清單。12,288 題只有三域，不等於多任務泛化；優先真人 relevance 與真正不同問題，不以改寫數或 pair 數冒充獨立 supervision。NG80 已證明本批 NQ 沒有 student 窗口截斷，不能再全怪 context；FEVER 長正例另做 context control。

下一個 encoder 階段再比較「成熟底座直接 ranking」與「matched-context passage-level teacher alignment warm-up，再 ranking」。借鑑 Potion 的 **分階段與更廣無標註曝光**，不抄 256 維、MSE 或 token-mean 架構。保持完整 teacher target／可匯出 score 契約，按 logical query/document/token exposure 與 compute 記帳；不能因 warm-up 多訓練了就稱方法勝利。full-corpus hybrid 與 DF 成本未共同改善，不擴大十萬級訓練。

NG83 三分支需另凍結執行協議，本輪沒有暗中追加。沒有 gradient training、locked-test scoring、部署、模型晉級、資料刪除或恢復 polling。主要程式、方法和結論在本倉庫；大型產物在本機 `II42/development/research/NG-0082`，歷史依賴按原凍結路徑保留。
