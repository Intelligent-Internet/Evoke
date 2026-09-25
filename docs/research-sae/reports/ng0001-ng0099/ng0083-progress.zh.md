# NG-0083 進度與階段結論

更新：2026-09-14。A/B/C、完整 corpus 排名與獨立 review 全部閉合，新產物已完整鏡像並驗證。**B 未通過預定的額外 boundary-weighting 准入門檻，不擴大此配方、不晉級 SAE／生產模型。**

後續方向已由 [2026-09-14 歷史路線審查](../designs/ng-research-route-review-20260914.zh.md) 更新：NG75-NG78 已檢查前排競爭者覆蓋，不再把本文原先提出的 coverage audit 當成新主線。下文實驗結果、原始門檻與當時的下一步提案保留作歷史紀錄；沒有因此啟動新訓練。

## A: 瓶頸已不是單純 query 泛化不足

[A/C frozen protocol](ng0083-ranking-causality-protocol.zh.md) 固定原 384 TRAIN queries、233,009 documents、NG82 masks、融合/RMS、六組 teacher pairs，沒有重新挑 mask 或訓練。

| 固定 profile | TRAIN nDCG@10 | TRAIN Recall@100 | TRAIN joint pair loss |
|---|---:|---:|---:|
| S | 0.762049 | 0.957663 | 0.574883 |
| utility-half | 0.683540 | 0.936085 | 0.479511 |
| DF-half | 0.751311 | 0.960918 | 0.574909 |
| 原始 dense | 0.797628 | 0.973303 | 不適用 |

utility-half 對 S 的 TRAIN nDCG 差 **-0.078509**，paired CI **[-0.097713,-0.059230]**；Recall 差 -0.021577，CI [-0.036988,-0.007375]。其 joint surrogate 真實下降0.095372，與 NG82 重播誤差<=1e-12。DF-half nDCG 差-0.010738，CI[-0.021329,-0.000898]。

**結論：原 utility surrogate 對完整 human-qrel 排名的失配，在它自己的 TRAIN 題目上已存在。** 不能僅歸因 query 太少／新題泛化；增添更多同配方資料不是当前首選。這沒有否定所有 pruning 或蒸餾，也未唯一分離 sparse-support 刪除、有限 pair 覆蓋、teacher-human 目標差異各自的責任。

補充描述性檢查（不是新增選型門檻）：同一批已保存 top lists 與 dense 的 O@10 / O@100，S 為0.569271 / 0.450286，utility-half 為0.480208 / 0.357083，DF-half 為0.549740 / 0.428411。原少數 pair loss 改善也沒有保住 dense 整體候選列表，不能只以「teacher 和 human 不一致」解釋這次失敗；仍未建立唯一因果分解。

## B: 條件化訓練有小幅收益，額外 boundary weighting 未被證明有效

[Operator protocol](ng0083-operator-execution.zh.md) 只改 relation weighting。共同12,288 TRAIN、rank256、PCA initialization、candidate bank、3,072 updates、optimizer與資源；field 和 boundary 各用獨立單GPU。原始1024 teacher是真值，不把壓縮表示當成無損target。

其 changed mechanism 是回歸有限的 teacher margin，而非讓少數 pair margin 無限制變大；保留 field anchor，並以 whole-corpus hybrid ranking 判定。不是 MSE 改名，也不是直接重做 M401 的 coordinate/tail reconstruction。仍是檢驗假說，不代表已超過dense，更不代表DF成本已改善。

prepare 337.82秒完成：12,288 queries、685,044 candidate exposures、197,515 unique documents，每題44至73個候選。共用 TRAIN scale 為0.030202658116153223。兩臂各3,072 updates及後續全排名均已閉合，沒有更改預算／checkpoint選擇。

同一1,536題exposed diagnostic；hybrid是固定0.9/0.1的**離線full-corpus z-score**，不是新的線上索引方案：

| Score operator | Pure nDCG@10 | Hybrid nDCG@10 | Hybrid Recall@100 |
|---|---:|---:|---:|
| 原始 dense1024 | 0.822175 | 0.820317 | 0.971423 |
| 未訓練 PCA256 | 0.782377 | 0.801238 | 0.962153 |
| field256 | 0.793661 | 0.804562 | 0.960803 |
| boundary256 | 0.794514 | 0.805811 | 0.962715 |

- boundary hybrid對field hybrid：nDCG差+0.001249，CI[-0.001154,+0.003644]；Recall差+0.001912，CI[-0.000122,+0.004164]。**沒有足夠證據表明額外boundary weighting更好。**
- boundary hybrid對initial hybrid：nDCG差+0.004572，CI[+0.000489,+0.008545]；Recall CI[-0.003145,+0.004438]。有小幅相對初始化收益，但不能把共同TRAIN/candidate條件化的收益全歸新權重。
- boundary hybrid對pure dense仍差-0.016364，CI[-0.022684,-0.010366]。沒有超過dense，也不是完整BM25+SAE模型的結果。
- pure field對initial nDCG差+0.011284，CI[+0.006572,+0.016557]；pure boundary差+0.012137，CI[+0.006917,+0.017580]。共同的query-conditioned candidate訓練有可測收益，但field hybrid的改善CI仍跨零。

### Loss 與泛化的更精確結論

| 測量面 | field | boundary |
|---|---:|---:|
| TRAIN六組boundary margin MSE | 0.00108725 | 0.00066789 |
| Diagnostic六組boundary margin MSE | 0.00143927 | 0.00149712 |
| TRAIN六組teacher order retention | 77.82% | 78.95% |
| Diagnostic六組teacher order retention | 76.49% | 75.81% |

額外權重降低了TRAIN邊界誤差，但對新query的相同測量面沒有收益。這支持目前**有限pair weighting的跨query轉移不足**，不等於所有 ranking-aware 蒸餾都失敗，也不足以證明僅增加query數能解決。此處六組pairs和NG82早期四組head-pair統計不同，不可直接比較百分比。

另一個重要對照：diagnostic全corpus centered-score MSE由initial的0.00021522升為field的0.00048539／boundary的0.00049906，pure nDCG反而改善。field訓練最小化的是query-conditioned **candidate bank** 的誤差，不是全corpus均勻誤差。這直接說明「平均dense分數重建更好」與「最重要的競爭排序更好」不是同一目標，不能再用單一global MSE作模型選型。

review確認兩臂相同bank/init/每題exposure、完整3072-step traces、全部positive IDs及ranks；initial／dense排名與NG82歷史全量控制完全重播。checkpoint與optimizer狀態保存，實際末batch GPU FP32到匯出FP64的最大score誤差為6.43e-8／5.69e-8，低於1e-5預定容差。

## 下一步只保留可判別的問題

1. 停止原utility-mask擴大、lambda sweep與直接SAE transfer；本輪沒有通過後兩者的預定准入門檻。
2. 保留已凍結field/boundary operators與共同candidate訓練作對照。下一個最便宜檢查是利用已有full-rank lists和bank，量化「真正造成top10錯誤的競爭文件是否在監督bank內」，分開候選覆蓋不足與同候選關係權重不足。先讀現有產物，不重跑encoder。
3. 若關鍵競爭文件未被覆蓋，再設計coverage改變、其餘資料/算力匹配的對照；若已覆蓋，優先分析teacher-human disagreement與query/source-family轉移。沒有這個判別前，不直接擴十萬題或換大底座。
4. B直接使用teacher vectors，只更新linear maps，尚未測試text-to-SAE表示能力或posting成本。不能從這一輪宣稱NG3已到容量上限，也不能把rank256的表現當成256NNZ稀疏表示上限。

以上是根據本輪閉合結果的新條件計畫，沒有在本輪暗中啟動下一輪。

## C: 監督邊界

[詳細盤點](ng0083-supervision-readiness.zh.md) 記錄13,824個exact-normalized unique queries、零跨split exact duplicates、547個跨split共享正例文件。因此保留 exposed diagnostic 標記，不宣稱 source-disjoint。沒有新資料下載、locked-test scoring或near-duplicate模型。

## 執行與重現

- `scripts/ng83_audit.py`、`scripts/ng83_operator.py`、`scripts/review_ng83.py` 及相應 tests 保存在主倉庫；大型產物為 `NG-0083/audit-v1`、`audit-review-v1/v2`、`operator-v1`、`operator-review-v1`。
- A/C inputs SHA `931b08b5f472a32bd8b260e8f9312df2a44213695ff9b14adbaab0fbe50a45ee`。
- B inputs SHA `8210873f711a7f74b519cd7322fa792e2bd21e11a09559c717058a658cff08fa`。
- A wall188.04秒，peak RSS6.11GiB；A/C均exit0、owned process group closed、ClearML offline closed/passed。A 獨立 review 重算1,536組rank records、paired CI與原joint loss。
- 新測試與 NG81/NG82 regression：69 passed；`git diff --check` 通過。Mac 不執行模型／Torch測試。
- Mac 已鏡像A/C全部新產物並核對34份source、exact phase inventory/SHA及1,536組independent rank metrics。`audit-review-v1`保留；`audit-review-v2`增加gold ID核對和保存metric validator source，結論相同。v2 results SHA `b448960d9e18c487166b0421ffa3774113a0126c92e32738712e2d151782d762`。
- B的prepare / field fit / boundary fit / field rank / boundary rank分別337.82 / 138.72 / 143.78 / 468.47 / 468.30秒；全部exit0、owned group closed、ClearML offline closed/passed。fit peak RSS約2.40/2.38GiB，rank約5.75/5.92GiB；無超限。review未另測RSS峰值。
- `operator-review-v1/results.json` SHA `565f56560a46b38e673f22047177b8f7fc14af55cc14d6d5a0e8300ed35fc9ca`；complete SHA `6e534b3dc59f4fccd36bbf19a7f84827995bdaada7ad1277433d3c60dfefe9e7`。predeclared `eligible_for_small_SAE_transfer=false`。
- Mac最終核對A/B各34份source、7個phase和3個review的exact inventory/SHA，以stdlib重算共24,576組rank metrics。新增payload約61.97MB，沒有重複複製大型歷史dataset/model。source snapshot與本倉庫實際執行的A/B程式及frozen protocol一致。
- 最終現場無NG83 worker/controller；四GPU均1MiB/0%，available RAM117GiB、disk free643GiB。不是持續自動訓練中，未恢復週期polling。
- 執行Lambda2，資料已在該機，經Spark1 ProxyJump。新產物放本地大檔規則指定研究根目錄；不搬動舊資料、不改生產、不晉級，不恢復週期 polling。
