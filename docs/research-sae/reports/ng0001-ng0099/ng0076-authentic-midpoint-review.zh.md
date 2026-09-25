# NG-0076：NQ 的前排退化在前半程已出現

2026-09-13 UTC。原始D96 query-only診斷、CPU全庫排名、完整Mac mirror及另一套獨立CSC/count review均已完成。這是定位進展，不是新的模型品質勝利。對照[方案](ng0076-authentic-midpoint-plan.zh.md)與[執行歷史](ng0076-endpoint-union-trajectory-status.zh.md)。

## 主要結論

1. **不能把問題只歸給第96步以後的witness refresh。** 同8題NQ的混合nDCG@10從0.753529降到D96的0.695181，D192為0.702839。前半程已下降0.058349，後半程平均回升0.007658，但沒有回到起點。這排除「只在後半程才開始退化」的解釋，不識別某batch或域的因果責任。
2. **這8題的Recall@100在三個時間點均為1。** 文檔沒有漏出top100，前十位的相對次序仍明顯退化。不能拿召回率、平均pair margin或posting更少替代前排品質；本輪不量測native latency。
3. **訓練並非單調改善或單調退化。** FEVER的nDCG中點最好，HotpotQA先降後恢復；個別NQ關係持續惡化，同時其他題改善。只看全域平均或最後一個checkpoint，會掩蓋這些不同路徑。
4. **端點聯集不是完整的中途競爭集合。** 中點top100共307個query-document位置不在兩端聯集內，占24題合計2,400位置的12.79%。它們不一定都越過gold，不能把這個數字直接叫作307個關鍵錯誤；但不能再用窄pool當作中途全庫排名。

## 實際做了什麼

直接載入原NG71 D96 checkpoint，不使用新的no-observer control checkpoint。兩個固定舊TRAIN_PILOT query先通過exact support和FP32值parity，最大差0，再只補24題固定TRAIN_SENTINEL query。模型SHA、eval模式、無gradient和RNG前後一致。

**共26個query forwards，0個document forwards，0次optimizer更新。** 重用同一原模型的233,009份D96 document CSR，加入一次既有加權BM25。三時點都使用同樣24题／每域8题、全部gold，不按結果換題。這24題未用於本輪或原pilot的梯度，但已被研究觀察，不是新holdout。原13點歷史observer軌跡仍未完成，也沒有將其失敗gate移除。

下表為query-balanced混合nDCG@10及all-positive Recall@100；各域只有8題，不作顯著性或泛化推論。

| 域 | nDCG initial | nDCG D96 | nDCG D192 | Recall initial | Recall D96 | Recall D192 |
|---|---:|---:|---:|---:|---:|---:|
| FEVER | 0.757803 | 0.850139 | 0.804005 | 0.937500 | 0.895833 | 1.000000 |
| HotpotQA | 0.893252 | 0.826643 | 0.887939 | 0.937500 | 0.875000 | 0.875000 |
| NQ | 0.753529 | 0.695181 | 0.702839 | 1.000000 | 1.000000 | 1.000000 |

兩個具體NQ例子幫助解釋平均數：Alma-Ata declaration題的其中一個gold由第2名變為10、再變為19，另一個gold一直第1。首座metal roller coaster題的gold由2→4→5。另一方面，Stormy Weather題在後半程改善，使NQ後半程平均數上升；**平均恢復不代表每個曾退化的gold都恢復**。這些例子是完成後的說明，不用於下一輪anchor選取或gate調整。

## 全部固定 Pair 的時間區間

相同完整端點聯集共5,226個gold/rival pairs，所有pair均核對兩段margin增量之和等於端點變化；positive分母和query分母不混用。non-gold不自動視為人工判無關。

| 域 | 全部pairs | 端點lost | 中點已lost且終點仍lost | 中點仍ahead、後半才lost | 先lost後regained | 中途top100聯集外位置 |
|---|---:|---:|---:|---:|---:|---:|
| FEVER | 1,434 | 35 | 34 | 1 | 51 | 67 |
| HotpotQA | 2,058 | 75 | 69 | 6 | 10 | 163 |
| NQ | 1,734 | 24 | 10 | 14 | 6 | 77 |

NQ的24個端點lost有14個在後半才出現，卻不能據此否定前半程nDCG下降：**pair數量沒有rank位置的重要性權重**。同一gold下降多名會產生多個pair翻轉，但不同名次對nDCG影響不等。這也不是NG72原有限說明集合的60個lost分母；擴充聯集後總共134個端點lost，不能混用兩份集合的計數。

## 對下一輪的修正

沿用較好的既有基座，不重新隨機訓練大型SAE。從step0就測試「學習新排序，同時只懲罰可信舊排序的退化」，不是等D96後才補救。這是待驗證干預，不是已證明原因。候選方向見[NG77可信margin保留準備方案](ng0077-trusted-margin-retention-plan.zh.md)。

- 優先最小、單變量的原D loss versus D加單側baseline-margin retention；相同TRAIN來源、曝光、encoder、optimizer及完整hybrid口徑。anchor取自原TRAIN資料與可信判斷，不取這24題或事後失敗例子。
- NG73的score-gradient、NG75的實際位移與本輪時間分布共同說明：僅更換pair weights或只限制NNZ不夠；是否能透過保留可信關係改善參數共享下的轉移，仍須真正配對訓練回答。
- 先審核TRAIN anchor的頭部／cutoff覆蓋和教師unknown分布，不用unknown假裝negative。若已有材料沒有足夠覆蓋，先修正資料方案，不啟動注定無法測試假說的loss sweep。
- 新訓練只在checkpoint停止後做全庫觀察，不再把定期GPU observer插進同一條要求exact replay的更新路徑。成本統計包括encoder、全庫編碼、tracking、hash、mirror，不僅update秒數。

完整hybrid超過dense並降低實際總成本的目標仍未達成。NG71／74已曝光DEV的NQ floor失敗與dense差距不因本輪診斷而改寫；沒有生產部署、LOCKED_TEST推論、native-cost准入或新模型發布。

## 執行與獨立核對

source commit `608af42acdb967026daa0d72dd81052cc59d43a1`。query phase83.497945秒，實際forward約0.986057秒，peak tree RSS1,781,071,872 bytes／GPU allocated223,909,376 bytes。CPU rank phase102.514604秒，24題全庫score／rank迴圈19.393935秒，peak RSS2,380,410,880 bytes。其餘時間包含載入、兩次依賴SHA核驗和追蹤收尾，不把它們算作純推論吞吐。

两phase均exit0／error null／owned group closed，ClearML actual-start／closed／passed；仍是offline且未remote-sync。controller於07:10:45 UTC終端，07:12查GPU0回到1MiB／0%，同事工作未干預。部分ProxyCommand查詢在banner階段逾時，切換使用者提供的 `ssh -A spark-1` 再nested SSH後恢復；沒有重啟健康工作。

Mac完整run加fixtures為73個檔案／51,434,330 bytes，另保留remote inventory本身。增傳37檔案／48,079,836 bytes，rsync退出0／0刪除／不覆寫既有檔案，传输與核驗41.433043秒。全部678個依賴與36個source/payload再驗SHA，來源保留，沒有聲稱災難還原測試。

獨立review讀frozen `review_ng76_midpoint.py`，逐CSC column散加，而非重用producer的CSR乘法；用heap top100與逐gold全庫count，而非重用lexsort／inverse-rank。全部5,592,216個混合分數最大差 **0**；24題完整排名、全部gold ranks、5,226對margin／分類、per-query及domain彙總逐項通過。Mac12.859026秒、peak RSS1,198,522,368 bytes，exit0／error null／owned group closed，沒有Torch或模型推論。

| 證據 | SHA256 |
|---|---|
| frozen `inputs.json` | `eb62acf84b654536523e7953473d28326549eca4347abeaf31c0f7971c70ac28` |
| `encode-query/complete.json` | `cd8358a919300af3de82b259c1db23972e26a968556de6ce0cf25657893a7dc1` |
| `rank/complete.json` | `44440b5c287f223967382799801a6fd6f7d8b31dfb6641d9b6151e73194b3383` |
| remote final inventory | `665ebe5a913c82875c240b2f56625542b90ac7e1232310a94cb9e4fbc2398013` |
| Mac full mirror `verified.json` | `f77d916c214fa433027b7fa576a3ae223dc85088097a4a2095741e825fdfefe7` |
| independent review `results.json` | `ff3a2a9cb74b401efbcb5496630487c97d99ccda6b22a1d0ec499d8e00f49083` |
| independent procedure `complete.json` | `a39d2b0a3133095b09bec4c23663150d16c39cf148a2b132dd9369f99dded977` |

完整artifact位於catalog的 `ng0076-authentic-midpoint-v1`，主repo保留實作、測試、方案與結論。凍結前local研究／文件測試358 passed，remote frozen fixtures66 passed；加入完成報告後同一local套件358 passed／3.59秒，`git diff --check`通過。獨立審核不是再跑同一個producer helper。
