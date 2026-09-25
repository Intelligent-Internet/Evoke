# NG-0080：Score-field 改善部分排序，但增密成本不合格

2026-09-14。這是 [S/P 訓練](ng0080-sparse-training.zh.md) 與 [全庫終點協議](ng0080-sparse-endpoints.zh.md) 的完成報告。最後審核於06:11:54 UTC正常退出；11個邏輯phase全部閉合，其中4個明示重用V2已驗證的審計／編碼，7個在V3新執行。這不是整個研究計畫或產品准入完成。

## 結論與比較邊界

S的完整hybrid在本批診斷題上有正向macro nDCG增益，但仍低於PPLX及TRAIN校準的dense+BM25，且增益集中在FEVER，另兩域均有點估計回退。P未取得區間排除零的macro增益。兩組document NNZ及literal posting-work皆增加，沒有達成「品質超越dense且降低總成本」。兩組均保留作研究產物，不替換產品模型。

兩臂從同一成熟原模型開始，各3,072更新、12,288個TRAIN query一次曝光，固定資料順序、候選panel、全部已知正例、query64/document256、FP32 eval與原RMS。S學習teacher centered score-field；P採用hybrid正例保護監督。這是配方比較，不把多個loss/scoring差異稱為單因素因果。

評估包括233,009個canonical文件、1,536個本輪gradient-excluded但具有歷史曝光的TRAIN_DIAGNOSTIC query，FEVER/HotpotQA/NQ各512。不是獨立holdout或多seed泛化證据。Student使用固定BM25/semantic權重0.1/0.9，不用診斷結果調融合；dense hybrid使用預先固定384個TRAIN題校準、full-corpus per-query z-score，選定alpha=0.1。它不是窮盡所有dense融合方案的上界。

## 全庫品質

| 方法 | Macro nDCG@10 | All-positive Recall@100 | Teacher top10 overlap |
|---|---:|---:|---:|
| BM25 | 0.532324 | 0.849309 | 不適用 |
| PPLX dense | 0.822175 | 0.965235 | 1.000000 |
| TRAIN-calibrated dense+BM25 | 0.820317 | 0.971423 | 未記錄 |
| 原模型 semantic | 0.762465 | 0.944371 | 0.516602 |
| 原模型 hybrid | 0.770716 | 0.948846 | 0.522917 |
| S semantic | 0.782791 | 0.947218 | 0.547266 |
| S hybrid | **0.786079** | **0.951925** | 0.554753 |
| P semantic | 0.767815 | 0.942698 | 0.486589 |
| P hybrid | 0.775382 | 0.946605 | 0.500977 |

| 配對比較 | nDCG差 | 95% bootstrap區間 | Recall差 | 95% bootstrap區間 |
|---|---:|---|---:|---|
| S hybrid - 原hybrid | +0.015363 | [0.008176, 0.022589] | +0.003079 | [-0.002107, 0.008249] |
| P hybrid - 原hybrid | +0.004666 | [-0.004419, 0.013684] | -0.002241 | [-0.008275, 0.003842] |
| S hybrid - PPLX | -0.036096 | [-0.045225, -0.027069] | -0.013310 | [-0.019341, -0.007497] |
| S hybrid - dense hybrid | -0.034238 | [-0.042479, -0.025977] | -0.019498 | [-0.025635, -0.013745] |
| S hybrid - P hybrid | +0.010697 | [0.003440, 0.018010] | +0.005320 | [0.000317, 0.010502] |

區間為預先固定seed80080、分域配對bootstrap10,000次的macro區間，僅涵蓋本次初始化與order下的query uncertainty；沒有多seed、獨立資料或事後多重比較校正的宣稱。S的排序增益相對原模型約1.99%，收回原hybrid到dense之nDCG差距的約29.9%；不能以此宣稱Recall已可靠提升。

| 領域 | 原hybrid nDCG | S hybrid | S - 原 | P hybrid | P - 原 |
|---|---:|---:|---:|---:|---:|
| FEVER | 0.792176 | 0.853772 | +0.061596 | 0.854627 | +0.062451 |
| HotpotQA | 0.826577 | 0.817338 | -0.009239 | 0.799319 | -0.027258 |
| NQ | 0.693395 | 0.687128 | -0.006266 | 0.672200 | -0.021195 |

兩臂都不是全域改善。S比P更能保留HotpotQA/NQ，但仍未回到這兩域的原模型表現。這是分域點估計，不能把上述macro CI誤用為各域顯著性。

## 幾何與成本

| 指標 | 原模型 | S | P |
|---|---:|---:|---:|
| 固定panel centered semantic MSE | 0.00636518 | 0.00235804 | 0.03498718 |
| Document NNZ | 63,312,968 | 254,721,463 | 292,984,531 |
| Document NNZ / 原模型 | 1.00x | 4.02x | 4.63x |
| 1,536題 Query NNZ | 79,009 | 221,474 | 95,000 |
| 每題semantic literal DF-sum | 1,016,902 | 11,309,055 | 2,512,204 |
| semantic DF-sum / 原模型 | 1.00x | 11.12x | 2.47x |
| lexical+semantic DF-sum / 原hybrid | 1.00x | 7.57x | 1.95x |
| Document CSR bytes | 507,435,784 | 2,038,703,744 | 2,344,808,288 |

三組lexical DF-sum相同，均為每題549,385.713。Total DF-sum是兩通道posting長度之和，重複文件仍會計入，既不是unique candidate數，也不是WAND實際訪問或P95/P99。CSR bytes只描述semantic矩陣，不含完整產品索引。沒有native latency、總索引空間、維護或端到端成本測試，不能說線上慢了7.57倍。

S的panel score-field誤差下降約63.0%，teacher top10 overlap和macro nDCG也有改善，但工作量代理大幅增長。這支持「直接對齊檢索分數能學到部分有效訊號」，同時反對「只要降低field誤差，便能自動得到跨域高效posting」。更低MSE既不能保證所有領域都好，也沒有約束query會觸發多少條長posting。

P的field誤差和teacher head overlap均較差，但P不是以S的同一field MSE為目標；不能僅凭該MSE判定實作錯誤。完整hybrid的區間及分域結果已足以說明這一配方沒有提供可靠的整體升級。

## 對整體假說的更新

| 問題 | 當前判斷 | 仍缺少的識別 |
|---|---|---|
| 成熟底座是否值得繼續適配 | F/D已支持同預算下解凍trunk有效；S亦有有限hybrid收益 | 不證明現有底座達上限、必須從零訓練或先預訓練一定最好 |
| F/D的巨大提升是否已超過原產品能力 | 否；D的dense讀出nDCG0.666245，不能把它與其弱initial投影的增益當作原hybrid0.770716的升級 | 兩種head與目標不同，不能據此孤立判斷trunk誰較優 |
| 直接teacher field能否轉移到sparse/hybrid | 支持部分保真與FEVER收益；不支持所有領域改善或超越dense | 前排競爭者覆蓋、跨域保留及label/teacher衝突的匹配對照 |
| 正例保護是否足以同時保住跨域品質 | 本次P配方未支持；HotpotQA/NQ點估計下滑 | 不能把配方比較當作所有正例監督方法無效 |
| 問題是否只是成本壓力過強 | 不能如此解釋；本次沒有新增強成本loss，增密仍未追上dense | 表達、優化、資料與score interaction仍混合 |
| 高DF能否被無害mask | 尚不能識別；目前只完成NNZ及DF-work觀測 | posting對margin/rank的條件貢獻、joint mask與native執行 |
| 是否需要更長context或更多多元監督 | 仍待新對照；既有監督圖連通、target飽和與token-window審計只提供線索 | source/topic/task-family隔離、真人判斷、固定context control |

下一步不應直接擴大相同S loss去追MSE，也不應以增密的S作為默認新底座。先保留原模型與S作matched research parents，研究如何在保留有效排序訊號時限制實際query-weighted posting工作量；高DF的聯合mask與訓練期成本配方需分開識別。這是後續研究優先順序，不是這一輪已啟動的新實驗或已證明的方法。獨立泛化與native總成本不能由本批曝光診斷取代。

## 完成與核對證據

- V3 input SHA：`aa98a6bf917c2e5e0ea901d0fd20223f2a33ecc974a3120aba6f0a6b6f123642`。
- Frozen runner SHA：`5faed2779920b164892c050a3d90f8cbe2cbbbff68d450aadae73f4bca224b72`。
- Review results SHA：`365aad14fe4738786d7d38f699333a026e7932a51e51db8910ed50f0133814f7`。
- Review complete SHA：`737c7c6fe9125fecb249974f0a2230b48ab08d813fe0bb95857ea54e4c1d7e62`。
- 27份source和698項dependencies、全部11phase精確inventory／檔案SHA／exit0／error=null／owned-group closed／actual-start closed-passed tracking再次核對。四組逐題已知正例rank重新計算nDCG與Recall，各域／macro均與review一致。
- Reviewer獨立head-coordinate最大誤差`1.0658141036401503e-14`，低於事先固定`1e-10`。不是第二次完全獨立全庫排序；完整背景計分另有雙reduction檢查。
- Review耗時4,657.58秒，peak process-tree RSS9,218,383,872 bytes，未放寬5,400秒／16GiB限制。其逐題CSC row讀取成本不代表產品query延遲。
- ClearML實際開始並正常closed/passed，但仍是offline，未宣稱server同步。未重新啟動訓練、自動輪詢或生產部署。
- 外部V3閉合root為118檔、41,784,192 bytes；Mac鏡像另見[執行紀錄](ng0080-execution-ledger.zh.md)。V2失敗與其重用產物保持原樣，不以恢復成功覆蓋失敗歷史。
