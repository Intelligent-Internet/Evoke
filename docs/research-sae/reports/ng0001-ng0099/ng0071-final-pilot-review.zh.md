# NG-0071：全庫候選與排序目標四組終局審核

日期：2026-09-13。依[凍結協議](ng0071-global-boundary-ranking-plan.zh.md)完成 `pilot-v1`，不是新的產品模型發布。準備與逐段執行證據見[前置／執行報告](ng0071-preparation-review.zh.md)。本報告的品質是 **BM25 0.1 + learned semantic 0.9 的完整 hybrid**，不是 semantic-only。

## 1. 結論與決策

全29個phase正常完成。D（全庫witness＋balanced soft pair）相對A（原pool＋CE）的DEV宏平均 nDCG@10增加0.008494、Recall@100增加0.006994，兩個paired區間均高於零。然而NQ nDCG下降0.007544，越過預先規定的單域最低差值 -0.005，因此 **P1品質門檻未通過，不進入P2擴大訓練**。

D仍比dense低0.045361 nDCG、0.012668 Recall；本輪沒有達成整體目標。成本代理沒有膨脹到阻擋線，但尚未測native索引、總CPU、記憶體、encoder或端到端延遲。不能把query-DF的改善寫成63%的實際加速。

最有價值的發現不是「新的loss已經成功」，而是 **loss與真正競爭候選有交互作用；減少高頻工作量和保留跨query排序能力仍是兩個問題**。接續先做TRAIN-only雙端編碼／margin保留診斷，不改本輪模型、不補訓、不放寬NQ門檻，也不丟棄NQ或新增正例。

## 2. 配對範圍

共同成熟NG3初始化、49,648,089參數、同一seed71001與曝光順序。每arm384題TRAIN（每域128）、兩遍、192次optimizer更新；不是更多獨立樣本。四組皆使用相同BM25比例、RMS、tokenizer、query64/document256輸入長度、FP32及完整233,009文件背景；没有輸出posting cap。A/B沿用原pool，C/D使用相同A0/A96全庫witness；不是各自self-mining。

所有八段訓練封存後才評估terminal2,304題：pilot TRAIN384、未參與梯度的TRAIN sentinel384、已曝光DEV_NEW1536（每域512）。中途step96只觀察pilot TRAIN384。共審核17,664列rank records；不是17,664個獨立query。Sentinel仍是TRAIN，DEV_NEW名稱也不代表未看過的holdout；LOCKED_TEST完全未編碼／評分。

## 3. 終局品質

| 固定模型／方案 | DEV nDCG@10 | DEV Recall@100 | 文件NNZ／初始 | DEV query-DF／初始 |
| --- | ---: | ---: | ---: | ---: |
| 共同初始hybrid | 0.777044 | 0.962175 | 1.0000 | 1.0000 |
| A 原pool＋CE | 0.776374 | 0.957794 | 0.8435 | 0.3838 |
| B 原pool＋pair | 0.767935 | 0.951651 | 0.7449 | 0.2293 |
| C witness＋CE | 0.779700 | 0.958304 | 1.0609 | 0.3518 |
| D witness＋pair | 0.784868 | 0.964788 | 0.9604 | 0.3706 |
| PPLX dense | 0.830229 | 0.977456 | 不適用 | 不適用 |
| BM25-only | 0.544635 | 0.855900 | 不適用 | 不適用 |

以下使用固定paired query bootstrap10,000次、seed71071、每域等權。區間只對這一次初始化／TRAIN順序及已曝光評估面條件成立，不是多seed確認，也未校正整組探索性比較的多重檢定。

| DEV對比 | nDCG差值 [95%區間] | Recall差值 [95%區間] |
| --- | --- | --- |
| B-A | -0.008439 [-0.012281,-0.004584] | -0.006143 [-0.010072,-0.002409] |
| C-A | +0.003326 [-0.001906,+0.008643] | +0.000510 [-0.002610,+0.003577] |
| D-C | +0.005168 [+0.000552,+0.009811] | +0.006484 [+0.003557,+0.009761] |
| (D-C)-(B-A) | +0.013606 [+0.007419,+0.019748] | +0.012627 [+0.008311,+0.017412] |
| D-A | +0.008494 [+0.004117,+0.012889] | +0.006994 [+0.003557,+0.010642] |
| D-initial | +0.007824 [+0.002086,+0.013747] | +0.002613 [-0.001255,+0.006639] |
| D-dense | -0.045361 [-0.055163,-0.035727] | -0.012668 [-0.017877,-0.007757] |

單換pair在原pool上變差；witness＋CE在FEVER大升、NQ明顯變差；兩者組合D的平均收益較好。這支持「候選與目標必須共同匹配」的假說，不能將收益單獨歸給metric weights：CE到pair同時改變loss形狀、positive balancing和uncertainty處理。若後續仍保留rank-weighted pair，必須做同pairs／targets的 `w=1` 消融，再主張複雜權重的價值；本輪不因宏平均上升直接啟動擴大。

| DEV域 | 初始nDCG | A nDCG | C nDCG | D nDCG | Dense nDCG | D-A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FEVER | 0.801461 | 0.822893 | 0.873939 | 0.844053 | 0.912343 | +0.021160 |
| HotpotQA | 0.847650 | 0.838511 | 0.827147 | 0.850378 | 0.844397 | +0.011867 |
| NQ | 0.682021 | 0.667716 | 0.638014 | 0.660173 | 0.733946 | **-0.007544** |

NQ不是只有新pair才退步，A/B/C/D在DEV都低於共同初始。D的NQ pilot-TRAIN nDCG由0.621384升至0.714451，但sentinel由0.651603降至0.627006、DEV由0.682021降至0.660173。這與[NG66](ng0066-training-duration-review.zh.md)的訓練／遷移分離相符；**不是沒有執行有效更新的證據，也未證明容量不足、標籤錯誤或單一原因**。不能僅據此加epoch。

DEV的2,725個正例pair中，D相對A有33個進入top100、6個掉出；相對初始是33進／21出，相對dense是19進／69出。D對A有222題nDCG改善、141題變差。宏平均增益並未消除尾部傷害；正例pair計數和每題等權Recall差值不是相同分母。

## 4. 成本與行為

D文件NNZ由初始63,312,968降至60,805,150（-3.96%）；文件CSR由507,435,784降至487,373,240 bytes。DEV平均query-DF由1,008,178.89降至373,678.45（-62.94%），P95由2,123,082降至792,742.75。DF>0.1的座標由412減為175，其postings由13,697,859降至5,533,311。

所以收益主要不是「整個索引大幅縮小」，而是查詢活躍support與文件DF分布改變。Query-DF是各active query座標對應DF之和，含同一文件多次出現，不是unique候選数、真正block/page touches或剪枝後成本。表中只算semantic CSR/support，沒有把BM25、native metadata／accelerator與索引維護成本算進去。B的代理成本最低但品質最差，再次否定「一味少posting就更好」。

訓練實際成本也不可省略：A/B各1,787,294 document tokens／12,840 candidate entries，C/D各7,961,577／61,074，witness增加約4.45x文本及4.76x候選。四組均有有限非零參數更新與完整optimizer續訓；clipped updates A/B/C/D為192/92/192/94。TRAIN loss與score-gradient全量重算通過既定容差，不用gradient大小直接推导學習率倍數。

## 5. 執行與證據

Source execution commit `16646238`。`inputs.json` SHA `5c44c86e13d688972100f1eb1a85b562ba8a631bdf31a2b2e4073ef9b304fb2a`，31份source/config/test/protocol與162個dependencies未改動。控制器在2026-09-13 00:44 UTC寫出 `all_phases_complete`，29個phase均exit0／error null／owned group closed／actual-start offline ClearML closed。GPU0已釋放，沒有重啟、重訓或變更production。

| 證據 | SHA256 |
| --- | --- |
| `encode-D-192/complete.json` | `81084a138dfc9857ccf1aac39413ca2c5a9b0def35c2aea413905e500e823363` |
| `rank-D-192/complete.json` | `58be7c7614d4fc5f363c9aa9ba5c08f3c8c9d1e69da93845f09c4a6d34a7c7a3` |
| `review/complete.json` | `f99d5a33670d3a9cb018eba86ec5780256f4c51d14fcfeb14345ff51c7e6fda3` |
| 外部 `pilot-v1-final-source.json` | `f52b7285f5cf706b634be96ee5714f195519bf72691ae728978be86f42bf1e60` |
| 外部 `pilot-v1-final-copy-review.json` | `501417bdde862809acc466eb9963e438a818d58dc23eaed5bc78d445d511255d` |

終局reviewer耗時107.65秒，最後D192編碼677.83秒、排序427.41秒；這些包含研究環境的雙路全分數驗證，不是產品query latency。部分早先phase有約300秒tracking收尾，正常自行封存；不追認為已證實的deadlock。ClearML為offline，未宣稱online同步。

遠端封存後另用唯讀程序核對source/dependencies及全部phase SHA，並建立445檔案、7,236,077,890 bytes的全run清單。00:57 UTC，Mac回傳exit0且完整inventory／全部SHA通過；31份source與162個dependencies、29份phase回執及controller終局均再核對。Mac用[凍結reviewer](../../../../scripts/review_ng71_pilot.py)在30.74秒內重算全部17,664列rank metrics、TRAIN loss／optimizer／exposures、CSR／獨立CSC DF、paired intervals、harm和gates，所有欄位一致，浮點最大差 `1.1102230246251565e-16`。審核前後run檔案inventory與SHA不變，沒有新模型推論。

完整mirror回執為外部 `pilot-v1-final-copy-review.json`；原remote資料保留。這不是獨立備份restore test，同一Betty上的其他分類也不是獨立災難備援。本輪134項文檔測試通過，另逐一檢查新終局報告／歷史報告的links與fragments；未重跑與文檔更新無關的產品全套回歸。

## 6. 下一個最小診斷

先回答「成熟基座哪一端的哪些margin沒有保住」，再選loss或更多監督。沿用[NG62–65的雙端變化問題](ng0071-global-boundary-ranking-plan.zh.md)，使用初始與D192已封存的同一詞彙座標／校準encoded artifacts，在pilot與sentinel兩個 **TRAIN** surface上做唯讀counterfactual；不產生新DEV或LOCKED分數，不更新模型。這是待執行診斷，不是已得因果結論。

令已包含凍結尺度的query/document codes為 $q_0,d_0$ 和 $q_1,d_1$，固定lexical項為 $b$：

$$
s_{00}=b+q_0^T d_0,\quad s_{10}=b+q_1^T d_0,\quad
s_{01}=b+q_0^T d_1,\quad s_{11}=b+q_1^T d_1.
$$

$$
s_{11}-s_{00}=(q_1-q_0)^T d_0+q_0^T(d_1-d_0)
               +(q_1-q_0)^T(d_1-d_0).
$$

對固定正例與競爭者的margin做相同恆等分解，分開query變化、document變化與交互項；不要將非線性nDCG也當成可直接相加。完整top-k仍對233,009文件算；候選小池只用於解釋margin，不能代替全庫排名。先重現00與11既有TRAIN ranks，再信任兩個交叉組合。共同校準／座標相容必須先核對SHA；交叉編碼只是診斷，不是新的可部署雙encoder方案。

同時利用[NG70來源表](ng0070-supervision-provenance-review.zh.md)與保存的teacher/pair manifest，按original/added正例、teacher同向／反向、截斷可見性unknown分層記錄，而不預判新增正例錯誤。對lost margin再區分support消失與共同support的weight改變，避免只算Jaccard就推斷品質。分析涵蓋固定TRAIN全集及改善／不變的控制組，不按DEV失敗案例選樣。

若主要損失集中query端，才設計固定文件基線／query retention的最小對照；若集中document端，才測文件margin／support retention；若以交互或teacher爭議為主，優先處理共同幾何／可信pair監督。上述是條件式路線，不同時疊加新loss、teacher與資料。後續保留rank weights仍須uniform-pair消融；擴大、獨立holdout和native成本驗證均未被本輪自動授權。
