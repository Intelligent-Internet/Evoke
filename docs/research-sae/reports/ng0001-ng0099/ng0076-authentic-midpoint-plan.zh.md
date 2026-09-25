# NG-0076：原始 D96 中點，重用文件快取

2026-09-13 UTC。接續[無observer對照](ng0076-replay-control-plan.zh.md)及[執行狀態](ng0076-endpoint-union-trajectory-status.zh.md)。query-only controller、CPU全庫排序及獨立review已實作，凍結／執行狀態另見狀態頁；不改原失敗attempt。

## 為什麼不繼續重放

無observer的96步已逐項匹配原NG71 trace、model和AdamW moments；原observer attempt卻在80步失敗。單次對照把observer相關執行路徑提升為主要嫌疑，尚未識別具體kernel／allocator原因，也不證明每次無observer都能bit-exact重現。繼續先解決所有GPU歷史重現細节，會延遲真正要回答的排序問題。

原NG71早已保留D96 checkpoint及完整 `encode-D-96` 文件CSR。中點診斷不需要產生一個「近似原歷史」的新模型，更不必執行任何optimizer。直接讀原始狀態即可先把問題分為前96步和後96步。

## 已確認的可重用材料

- 原D96 model SHA256：`92e574071092f08f6288a980ac27a1f0e17b82b0274acac40d3cf8cc228abd49`。
- `NG-0071/pilot-v1/encode-D-96/results.json` 綁定同一model SHA與原train-D-96封存回執。
- 文件CSR：233,009 rows，50,554,209 NNZ，405,365,712 bytes的CSR陣列大小；這不是本次新增索引或native總成本測量。
- 舊D96 query CSR只有384個TRAIN_PILOT queries，固定24題TRAIN_SENTINEL的覆蓋是 **0／24**；不能把pilot query向量當成sentinel，也不能聲稱可直接從舊排名讀出這24題中點品質。
- 原initial／D192的24題scores、完整top100／gold ranks和固定BM25分數已在NG76準備階段核對。
- 現有control manifest **尚未包含encode-D-96產物**。新unit必須增加該phase完整inventory、exit／ClearML／model關聯驗證，不能因文件在磁碟上就視為已凍結輸入。

## 最小新工作

1. 新manifest綁定同24題hash-selected TRAIN sentinel、原D96 checkpoint、完整D96文件編碼phase、既有lexical caches／identity順序及initial／D192端點證據。保留所有gold、所有5,226個固定聯集pairs及控制組，不按結果換題。
2. 只用原D96模型新增24題query forward。另固定使用舊D96 query-ids的前2個TRAIN_PILOT query作編碼parity controls，不依結果選取。先驗完整model SHA、eval模式、FP32、RMS／tokenizer契約，再核对control的sparse support必須完全一致，正值且有限的FP32 readout沿用 `rtol=2e-5, atol=2e-7`；遇差異不即場放寬。這個前置gate在任何sentinel forward之前執行。
3. 不重編任何文件。query和cached document必須綁同一原D96 model，而非新no-observer control產物，即使後者SHA相同也不改來源鏈。無backward、無optimizer更新、無DEV或LOCKED_TEST推論。
4. CPU用新24題query CSR與已封存233,009份D96 document CSR，以及同一份已加權BM25，計算完整混合全庫top100和全部gold ranks；score高到低、global document ID處理tie。另抽取原固定聯集分數作中點margin。中點新增head不回寫原pool，另外記錄中段top100有多少不在端點聯集內。
5. 對initial→D96→D192比較query-balanced nDCG@10／all-positive Recall@100，並按域保留8題全部結果；對5,226 pairs分別報前半／後半margin變化、lost／regained／gained／retained。驗證兩個增量相加等於端點變化，不以某域出現在batch就認定因果責任。
6. 獨立CPU reduction逐項核對saved sparse dot、BM25只加一次、完整排名／gold cutoff、固定池margin和區間分類。不得只重跑同一個helper就稱為獨立實作。終端資源回執、ClearML actual-start／closed、完整mirror／SHA與source驗證完成才宣稱本階段完成。

這次D96若完成完整233,009文件排序，可明確報「這24題TRAIN的全庫中點排名」，不再把固定端點聯集排名冒充全庫。然而24題仍是已曝光診斷集合，不是新held-out品質評估，也不足以取代1536題DEV或未曝光holdout的准入。兩個區間仍只提供時間關聯。

## 資源與決策

新query forward單GPU，先查空閒、不移除其他人工作，CPU4／RSS16GiB／GPU20GiB／host available>24GiB／disk>40GiB／1800秒外層guard。只編碼26題，不作資料或算力擴大。CPU全庫reduction另設4threads／RSS8GiB／disk40GiB／900秒，包括I/O與SHA；remote controller保留較嚴格host floor24GiB，Mac獨立review用12GiB。`ng71_pilot.phase`只新增可縮緊RSS的optional參數，預設16GiB不變，拒絕大於16GiB；原凍結run不修改。先以最小query canary估時，包含已消耗的載入／SHA時間，不任意加大同一attempt上限。沒有產生新耗時資料之前不把估計當作實測。

## 實作與獨立核對

[ng76_authentic_midpoint.py](../../../../scripts/ng76_authentic_midpoint.py) 只呼叫原 `Encoder.encode(..., 'query')`，沒有document encoder／backward／optimizer呼叫。每個phase完成後重新驗證frozen inputs，ClearML實際啟動及關閉，保存完整資源回執。原D96 encoding phase的封存SHA固定為 `cd9ecb1f0349e9d0f3b3744dcda82c56da50048b5403e3c9d8003ba52b4df4ea`。

[ng76_midpoint_math.py](../../../../scripts/ng76_midpoint_math.py) 以CSR乘積和全排序產生scores、gold ranks及三時間點pairs；[review_ng76_midpoint.py](../../../../scripts/review_ng76_midpoint.py) 則逐個CSC vocabulary column散加，再用heap top100與逐gold全庫count比較排名，不重用primary乘法／排序／margin分類helper。完整scores的獨立FP64比較預先固定 `rtol=1e-12, atol=1e-12`；top100和gold ranks要求相同，metrics絕對差小於1e-12，telescoping誤差小於1e-10。全部24題和5,226對均核對，不抽樣。

若中點已失去主要關係，下一步聚焦前半程的baseline保持與監督覆蓋；若後半程才流失，先查A96共同witness與D96當前模型的候選／權重失配。若大量middle-only rivals出現，先改善觀察範圍而不是據窄pool推因果。只有這些證據指向必要的更細時間解析時，才另做可重現且配對的新軌跡對照。不要以這份計畫直接安裝retention loss、PCGrad或權重網格。
