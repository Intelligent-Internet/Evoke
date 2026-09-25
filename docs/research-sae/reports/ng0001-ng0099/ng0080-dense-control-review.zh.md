# NG-0080：底座適配有收益，但尚未保留完整檢索能力

2026-09-13。這是 [F/D 配對訓練](ng0080-dense-control.zh.md) 與 [全背景終點協議](ng0080-dense-endpoints.zh.md) 的一次性終點結論，不是新產品、全研究完成或部署批准。兩組各3,072更新，全部訓練、獨立曝光／optimizer審計、fresh encoding、四組全背景ranking與review已閉合。沒有中途選checkpoint或改loss。

## 比較範圍

F凍結成熟NG3 trunk，只訓練共同dense讀出；D更新同一trunk及同一讀出。共同初始化為TRAIN-only mean ridge，使用相同12,288題、固定候選panel、teacher、順序、query64/document256及optimizer配置。兩組query/token/candidate曝光相同，不代表計算量或參數量相同。

評估是1,536個本輪gradient-excluded `TRAIN_DIAGNOSTIC` query，FEVER/HotpotQA/NQ各512題，全部233,009文件參與競爭，保留所有已知正例。題目具有歷史曝光，不是新的獨立holdout。**Initial是未適配的mean dense投影，不是成熟模型原來的BM25+sparse baseline。** 因此不能把下表增益說成已超越既有產品或早期不同題集的里程碑。

## 全背景結果

| 讀出 | Macro nDCG@10 | All-positive Recall@100 | Teacher全庫top10 overlap |
| --- | ---: | ---: | ---: |
| Initial mean projection | 0.333550 | 0.646566 | 0.195182 |
| F：只訓練dense讀出 | 0.500623 | 0.778086 | 0.236523 |
| D：trunk與dense讀出一起訓練 | 0.666245 | 0.840540 | 0.348763 |
| PPLX teacher | 0.822175 | 0.965235 | 1.000000 |

| 領域 | F nDCG@10 | D nDCG@10 | PPLX nDCG@10 | D-F |
| --- | ---: | ---: | ---: | ---: |
| FEVER | 0.699941 | 0.851457 | 0.910471 | +0.151516 |
| HotpotQA | 0.409762 | 0.606172 | 0.827457 | +0.196410 |
| NQ | 0.392167 | 0.541106 | 0.728598 | +0.148939 |

D-F的macro nDCG差為 **+0.165622**，預先固定的分域paired bootstrap95%區間 `[0.150875, 0.180781]`；Recall差為+0.062454，區間 `[0.048914, 0.076551]`。D-PPLX仍為-0.155930 nDCG，區間 `[-0.169364, -0.142500]`；Recall差-0.124696。每個區間只反映同一初始化和資料順序下的query差異，不包含多seed訓練不確定性。

## 幾何與相關性分開看

| 讀出 | 固定panel centered score MSE | 固定panel nDCG@10 | 全背景nDCG@10 |
| --- | ---: | ---: | ---: |
| Initial | 0.01605670 | 0.679622 | 0.333550 |
| F | 0.00530414 | 0.771466 | 0.500623 |
| D | 0.00403839 | 0.834945 | 0.666245 |
| PPLX | 約0 | 0.917731 | 0.822175 |

D相對F同時改善固定panel幾何、全庫teacher head overlap及全庫relevance，不是只有training loss或局部候選分數改善。但D的teacher top10 overlap仍僅0.3488；小panel的MSE與nDCG不能替代全庫前排保真。不同候選宇宙本來就有不同難度，不能單憑panel/full差距把原因判為過擬合或負例採樣不足。

本對照沒有posting剪枝或成本懲罰，也沒有BM25。它的剩餘差距不能用「本輪壓縮太狠」解釋，但這不證明過往sparse問題與成本無關。模型容量、輸入可見性、訓練充分性、資料／任務覆蓋、teacher與標註衝突仍有混合影響，尚不能宣稱唯一主因。

## 對下一步的約束

- **支持：** 在這個共同初始化、固定資料與更新預算下，繼續適配trunk比只訓練dense讀出有效，而且三域都有收益。不能以第一波linear probe失敗宣布底座沒有可利用的信息。
- **未證明：** 大量無監督／teacher預訓練一定比直接排序訓練好；從零訓練一定必要；現有底座已達容量上限。這次没有direct-versus-staged或訓練充分性對照。
- **下一個已固定對照：** 保持 [S/P實際sparse准入](ng0080-sparse-preflight.zh.md) 不變，再測實際sparse score-field與完整hybrid保護監督。不因F/D勝負改尺度、loss、初始化或題目。GPU暫被其他工作占用，只等待空閒，不搶占。
- **仍待閉合：** 多來源任務監督、可追溯human judgments、domain-specific context、DF聯合mask、完整hybrid對dense與dense+BM25，以及native bytes／遍歷／尾延遲和獨立泛化。不能由這個dense診斷批准產品模型晉級。

## 重現與驗證

外部單元 `NG-0080/dense-endpoints-v1` input SHA256：`da62c44c522cdb3b5a0573ff0d4a731e7c528b661aa2bd07ef62174b69c32473`。最後ranking controller於2026-09-13 21:23:20 UTC完成，全部9個phase逐檔SHA／精確inventory／exit0／error=null／closed process group／actual-start closed tracking在本次核對通過。ClearML是offline，不是已同步到server。

原始review結果 SHA256：`8ed5c1cb7c4d32934d7fcdecc968d7201e927a2acf68cf82f0af2d18154b785b`。全庫dot/einsum核對及head-coordinate獨立重算通過，review最大座標誤差 `7.77e-16`。四個rank phase各415–432秒，peak RSS約3.35GB；review60.02秒、2.04GB。此計算成本是離線FP64診斷成本，不是服務native查詢延遲。

Frozen runner SHA `c0e63234a5664481d717dd82752a83e5774be42019dd4ed84857fa5245d4baed`；獨立訓練reviewer SHA `f1e35b551ccb12c6870fbf19a3f7740cc8d36401267e6dad2ef75d745f1127d0`。當前Git HEAD不包含新增程式，必須依逐檔source manifest重現，不能只checkout該HEAD。

完整endpoint為148份文件、2,902,226,572 bytes；remote inventory SHA `a8ab7aa710df8611250a8df3b6a92cb37fe1bb921c42d7990cb044fb81859fbd`。Mac鏡像狀態另見 [執行紀錄](ng0080-execution-ledger.zh.md)，不能把remote驗證通過當成mirror完成。來源及所有失敗嘗試均保留。
