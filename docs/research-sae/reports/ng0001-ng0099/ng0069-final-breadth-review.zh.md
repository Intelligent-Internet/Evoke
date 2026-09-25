# NG-0069：三個順序 seed 的廣度實驗終局審核

日期：2026-09-12。**廣度研究門檻通過；完整 hybrid 仍未超越 dense，且表示明顯增密。** 不晉級產品，不直接放大同一訓練配方。

## 完整性與比較口徑

沿用 [預先凍結的研究協議](ng0069-matched-exposure-breadth-plan.zh.md)、[執行契約](ng0069-execution-contract.zh.md) 及 [僅評估續跑](ng0069-evaluation-continuation.zh.md)。A 是1,536題重複四遍，B是6,144不同TRAIN題目一遍；相同6,144 exposures、1,536 updates、初始化、teacher、CE、固定BM25/semantic混合及輸入長度。不是equal-token或equal-FLOPs比較。

29個繼承／新階段均通過complete、exit0、process-group closed、offline ClearML closed和輸出SHA。凍結在 `evaluation-v2` 內的獨立 reviewer 正常退出，核對70,848條逐題排名、所有正例、三組訓練／optimizer chain、NNZ／CSR及DF代理。最後階段用時2,197.34秒，峰值treeRSS約5.92GiB；controller與所屬tmux已退出。沒有重跑健康訓練或修改失敗attempt。

結果檔 `NG-0069/paired-review-v1.json` 的遠端與Mac SHA均為 `ae0604ec64a0abd517f8f392f7effc71587500e9d2d23ab14d7e88cb469d8ae2`。pipeline SHA為 `7bb0d84e40b26586c0316f6eb561508ac5c72321d11f69a80fabf72545aba31e`，inputs SHA為 `488a1d7c8d24e7bf3b52e27a7cb2f27027657bd1fca867bf56f920f9c1e49457`。19:29UTC完整鏡像已終局校驗：188份一般檔案／4,221,176,598 bytes逐檔SHA相符，14個繼承相對連結相符，Mac再次核對全部29階段與inputs通過。回執為 `NG-0069/evaluation-v2-copy-verification.json`；遠端原件保留，未重新執行模型inference restore test。

## 新 DEV 結果

同233,009文件背景、1,536 DEV queries，三域各512。三個順序seeds為59059、66061、66067，不是獨立初始化。A/B數字為三seed平均；bootstrap先按query平均，再分域等權重抽樣10,000次，區間條件於這三個順序seed。DEV已曝光，不叫locked test。

| 完整方案 | macro nDCG@10 | macro Recall@100 |
| --- | ---: | ---: |
| BM25 only | 0.544635 | 0.855900 |
| 初始成熟 hybrid | 0.777044 | 0.962175 |
| A：重複小集合 | 0.778122 | 0.960977 |
| B：增加不同題目 | 0.790547 | 0.964423 |
| PPLX dense | 0.830229 | 0.977456 |

B-A nDCG提升 **1.2425個百分點**，paired95%區間 **[0.7169, 1.7956]個百分點**；Recall提升0.3446個百分點。预定四項廣度門檻全部通過。

| 域 | B-A nDCG差 | B-A Recall差 | B-dense nDCG差 |
| --- | ---: | ---: | ---: |
| FEVER | +0.037157 | +0.001351 | -0.047351 |
| HotpotQA | -0.004236 | -0.000760 | -0.004927 |
| NQ | +0.004353 | +0.009747 | -0.066769 |

收益主要來自FEVER；HotpotQA有小幅回退，不能寫成所有域都改善。B的NQ nDCG為0.667178，仍低於初始0.682021。B-dense macro nDCG差為-0.039682，95%區間[-0.048503,-0.030795]；Recall差-0.013033。廣度收益可信，但沒有接近到可以宣布超過dense。

## 成本：增密不是單 seed 偶然現象

| 順序 seed | B/A document NNZ | B/init document NNZ | B/A DEV query-DF | B/init DEV query-DF |
| --- | ---: | ---: | ---: | ---: |
| 59059 | 2.1393x | 2.3401x | 2.5714x | 1.2280x |
| 66061 | 2.2266x | 2.4779x | 2.8713x | 1.4472x |
| 66067 | 2.2749x | 2.5211x | 3.4992x | 1.7859x |

B的semantic document NNZ為148,155,857至159,619,937，初始為63,312,968；原始CSR約1.19至1.28GB。這些是完整文件表示／未剪枝query-DF工作代理，**不是原生索引bytes或warm latency倍數**。也沒有證明每個新增posting都造成收益，不能直接據此任意mask。A的query-DF比初始低，使B/A看起來更大；因此同時列B/init，避免選擇性基線。

## 對下一輪的決策

1. 不再以重複同一小集合更多epochs為主線。廣度確實有用，但不把目前B直接升為新基座，也不立即放大相同CE配方。
2. 推進已預先設計的 [NG71](ng0071-global-boundary-ranking-plan.zh.md)：共同成熟NG3初始化，拆開全庫witness與排序目標的四組對照，保留正例、teacher不確定性與成本晉級門檻。
3. 在訓練前先查96題TRAIN的真實head／rank100邊界：大量easy-pair監督是否掩蓋了對真正競爭者的teacher衝突。token截斷和支持證據可見性分開記錄，沒有span標註就保留unknown。
4. 不因廣度門檻通過便繞過NG71的1.25x對init成本代理門檻、未實作的執行層或CUDA預演。LOCKED_TEST的1,536題仍未編碼／未評分；native總成本仍待後續獨立驗證。

本輪收斂的是「更多不同監督比重複擬合好，但仍以增密換來有限且跨域不均衡的收益」。對全庫排序邊界和可信偏好的診斷，比再做一輪無歸因的大訓練更有價值；這是下一步投資判斷，不是已證明的唯一瓶頸。
