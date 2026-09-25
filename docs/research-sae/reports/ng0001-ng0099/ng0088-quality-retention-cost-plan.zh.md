# NG-0088：保留排序收益與控制 Posting 成本

日期：2026-09-20。狀態：下一階段設計；第一階段的具體凍結範圍見 [執行契約](ng0088-execution.zh.md)。下列預算與門檻不是已獲得的結果；執行前須一次凍結，不能看完 validation 再修改。第二階段訓練仍以第一階段結果為前提。

## 1. 唯一主要問題

**能否在明顯較低的可部署 posting 成本下，保留 NG87-A 的大部分 overall hybrid 排序收益？**

[NG87 terminal review](ng0087-terminal-review.zh.md) 支持「成熟底座可以進一步學到排序」，還不支持「收益已轉移到全背景」或「增密可無損移除」。這輪先建立可重現的品質／成本 operating point，再追求更高品質，不同時改教師、資料、字典、融合和索引引擎。

A/8192 是固定品質參考，不是完美 relevance oracle。B/8192 保留為較低密度參考，兩者尚未做等成本比較。原成熟底座和所有失敗紀錄不覆寫，不用短暫較好的中間 checkpoint 替代既定 endpoint。

## 2. 與之前探索的差別

| 既有證據 | 本輪必須避開的重複 |
|---|---|
| [M91/M92 sparse-teacher compression](../sae-m91-m92-support-compression-results-report.md) 曾改善早期排名但傷害 coverage | 壓縮蒸餾並非新發明；不能只用 teacher KL 或 nDCG，必須獨立看全背景 Recall。舊 latent SAE 與本次成熟詞彙頭／修正後目標不同，舊的 k384 失敗不是跨模型容量定理 |
| [M1518 DF-FLOPS](../m1500-m1599/ii42-m1518-df-flops-paired-smoke-report.md) 降低 head concentration，仍全庫觸及 | 不能只改善 DF 直方圖或改 penalty 係數，必須量 query 分佈下的整體工作量 |
| [M1933](../m1900-m1999/evoke-m1933-semantic-budget-frontier-report.md)／[M1934](../m1900-m1999/evoke-m1934-fixed-budget-unseen-transfer-report.md) 的固定容量改善有可轉移但有限的收益 | 保留 deterministic pruning 作必要對照；神經訓練必須勝過同成本、不訓練的 publisher，否則不增加模型複雜度 |
| [NG82](ng0082-progress.zh.md)／[NG83](ng0083-progress.zh.md) 的 utility mask 使真實排名下降 | 不重做逐 posting utility 相加或全局 mask 選擇；直接在最終 masked hybrid 分數上訓練、評估 |
| [NG87](ng0087-clean-base-teacher-ab.zh.md) 的 full-std quality-only 訓練 | 現在有完成訓練、可校驗的較好 endpoint；測試它的排序能否在預算內重學，而不是在原退化 checkpoint 上增加多個互相拉扯的 loss |

這是有新起點與可識別對照的舊假說再驗證，不宣稱首次提出 self-distillation、TopK 或 DF-aware training。

## 3. 第一階段：全背景與固定成本剖面

先完成 evaluation-only 單位，不以更多 probe 代替最終排序。背景沿用 NG85 的 290,609 篇可搜索文件；這是本輪完整固定背景，**不是 BEIR15 或各來源的全部原始 corpus**。

1. Base、A/8192、B/8192 各自編碼相同背景和全部 2,048 exposed-validation queries；保持共同 prefix、BM25、固定 RMS 和融合權重。先重現已保存候選分數，再做真正的 global top-k。
2. 同時報告 hybrid nDCG@10、Recall@10／100、known-positive ranks，pure sparse、BM25、PPLX dense、TRAIN 校準 dense+BM25。正常原文輸入的 dense 另列，不以 student-prefix 限制後的 dense 冒充產品上限。缺少正常輸入對照時清楚標 pending。
3. 去重計算全庫 postings、每文件 NNZ 分佈、query NNZ、DF、query posting union；報告 BM25、semantic、combined literal DF-work 的 mean/p95。不拿重複候選文件的均值當全庫大小。
4. 在已編碼資料上，對 A/B 各做兩個事前固定的 magnitude TopK profiles：較寬的 Q96/D768，以及主要壓縮目標 Q64/D384。Q/D 指最多保留的非零維度，不是 top-k 文件數。兩個點只界定損失曲線，不掃網格、不事後換主要點。
5. 為分清 query/document 裁剪損失，主要點另報 document-only 與 query-only 消融。所有配置完整披露，不挑最好的 domain-specific mask。相同 query support／doc support 下，不重估 RMS、不重新調 alpha。

Q64/D384 的用意是先限制可發出的 posting 數，D768 是較寬的診斷參照。384 並非聲稱已證明的最佳容量；以候選均值 290.29 推算的 1.32x 也不能代替全庫實測。若原始 A 在全背景不再改善，就先關閉「保住 A」的主張，不直接開始一輪昂貴壓縮訓練。

這階段只做無梯度編碼／排名，不新增教師、不改產品、不碰 locked test。宏平均 domain 權重、空文件與正例分母沿用凍結資料；不以過濾難題製造提升。

## 4. 第二階段：兩個等預算訓練臂

只有 A 的全背景改善得到支持，才進入有界訓練。兩臂同從 A/8192 載入參數，fresh AdamW，相同 TRAIN bank、order、曝光量、masked scoring path 和最終 Q64/D384。保持 lexical 部分與 0.1/0.9 融合不變。

| 臂 | 唯一不同之處 | 要回答的問題 |
|---|---|---|
| R：保留排序 | frozen A/8192 的完整 hybrid soft targets | 同一模型能否用較少 posting 重新實現已有排序，而不是只刪去排序貢獻？ |
| P：繼續改善 | 沿用原 PPLX soft targets | 相同成本、起點與訓練量下，直接向原 teacher 學是否比自我保留更好？ |

凍結 A 本身、A 的零更新 TopK profile，以及同預算的 B profile 是必要對照。第一個 pilot 不把 A、PPLX、RankT5 和 gold 同時加權混合，避免無法辨識收益來源。這是壓縮目標對照，不是聲稱 self-distillation 自己會創造新 relevance 資訊。

建議正式 pilot 為每臂一個完整 epoch：16,384 TRAIN queries、4,096 updates。先用 16-step disposable canary 檢查實際步速與 parity；沿用已驗證 optimizer 配方，不按 validation 掃 LR。直接在固定預算下訓練，不先偷偷練較大的表示再於 endpoint 裁剪。終點固定，失敗保留、不臨時延長；pilot 失敗只能否定此配方與預算，不能宣稱稀疏模型容量無解。

### 訓練與服務必須是同一個分數

令 $T_K$ 保留正權重最大的 K 個座標，零值不強行補成非零；ties 用固定 token ID 排序。文件編碼和 query 正規化為：

$$
\widetilde d=T_{K_d}(d_\theta),\qquad
v=T_{K_q}(q_\theta),\qquad
\widetilde q=0.9\,v/(v^\top r).
$$

其中 $r$ 是既有固定 RMS 向量，$q_\theta$ 指尚未正規化的 raw query output。先 mask、再用剩餘 support 計算分母；不能訓練一套，匯出時才用另一套。分母非正或非有限即記錄並停止，不加隱藏 epsilon/fallback。TopK 可能使僅餘的 support 全落在零 RMS 座標，這是必做的 canary 邊界，而非已解決問題。

$$
s_\theta(q,d)=b(q,d)+\widetilde q^\top\widetilde d,
\qquad
L=\mathrm{KL}\left(p_T\,\Vert\,
\operatorname{softmax}(s_\theta/\max(\sigma(s_\theta),10^{-6}))\right).
$$

$b(q,d)$ 是既有已加權 lexical score；$p_T$ 分別來自 frozen A hybrid 或原 PPLX，並按同一既有 teacher std 規則準備。Student std 的導數保留完整；teacher stop-gradient。這個 KL 與 fixed-target CE 梯度相同，但不保證最終 relevance 或 Recall，所以必須用全背景 qrels 評估。

Hard TopK 在 support 不變區域對保留的值可微，未入選項不直接取得該 score 的梯度；不能聲稱有精確可微的 support selection。本輪不用未驗證 straight-through estimator。檢查 masked full-graph/VJP parity、near-tie 選擇穩定性、checkpoint reload、零 RMS support、以及離線／匯出分數一致性。

## 5. 成本不等於 NNZ

$$
C_{\mathrm{literal}}(q)=
\sum_{j:\widetilde q_j\ne0}\mathrm{DF}_j,
\qquad
\mathbb E_q C_{\mathrm{literal}}=
\sum_j\Pr_q(\widetilde q_j\ne0)\mathrm{DF}_j.
$$

TopK 限制文件大小與 query 長度，但仍可能保留少數覆蓋全庫的座標，因此**不保證 DF-work 或 warm latency 降低**。上述是 literal traversal，不是 WAND 實際解碼量；query encoder 時間也必須另計。

若訓練後 NNZ 大減、DF-work 卻沒降，不能宣布成本成功或立即再縮 K。先記錄高 DF 路線仍然失敗；只有在同一輸出預算下有新機制假說時，才另立跨 query/document occupancy 的成本處理協議。禁止以 NG88 名義自動開啟 FLOPS 係數掃描或動態索引 mask。

## 6. 事前決策與停止條件

令 $Q_0,Q_A,Q_C$ 為相同全背景上的 base、A 和壓縮候選 nDCG@10。若 $G=Q_A-Q_0>0$，第一個研究目標為保留至少 80% 的已驗證增益：

$$
Q_C-Q_0\ge0.8G.
$$

這不是保留「80% 的絕對 nDCG」。以候選集舊數值示意，門檻約為 0.6891；真正裁決必須用新全背景數值，不能混用比較面。80% 是明確的研究取捨，不是保真定理。

建議的 pilot joint screen：上述增益保留、相對 base 的 paired nDCG 區間下界 >0、Recall@100 點差不低於 base -0.002；全庫 semantic postings 不高於 base 1.5x，combined literal DF-work mean/p95 均不高於 base 1.5x 且不高於 unmasked A 的一半。各指標及區間完整披露。這些是新 pilot 的提案，不追溯更改 NG87 的已失敗 gate，也不是 native latency 驗收。

- 若零更新 TopK 已達相同品質／成本，而訓練沒有可信額外收益：保留簡單 publisher，不為了訓練而訓練。
- 若 R 保住品質而 P 較差：支持「先保存已學幾何，再分階段改善」，但不是自蒸餾能超過 dense 的證據。
- 若 P 等成本品質更好：不用 R 的複雜保留路徑，優先原教師的預算內續訓。
- 若兩臂都不能兼顧品質和成本：關閉此 pilot，不重掃混合權重。D768 的靜態結果只幫助區分過度壓縮與普遍脆弱，不自動授權下一個更寬訓練。
- 若 joint screen 通過：先第二 seed（同一父模型，驗證壓縮可重現性），再獨立來源、正常輸入 dense 對照、原生 index bytes／warm p50/p95／decoded postings。要聲稱整條訓練流程可重現，仍需另一個 clean-base NG87 seed，不能用第二個壓縮 seed 代替。

整體品質優先，不要求所有域都勝 dense；FEVER/FiQA 增益和 HotpotQA/NQ 損失仍分開披露。不以同一 exposed validation 上反覆調參稱獨立泛化。

## 7. 後續強化與資源

在成本 operating point 成立之前，不再擴大教師、資料和模型。其後才在相同 serving 預算下引入新的可靠人工 relevance、topic/source-disjoint 問題和 teacher-human disagreement；unknown 不當成 judged negative。A 的錯誤不應被永久凍結成真值，PPLX imitation 也不保證超過 PPLX。新 supervision 的價值必須在固定成本下比較，不靠回到增密來宣稱突破。

使用者已將 II-42 暫時限制為 Lambda2 **GPU 0 only**。無梯度編碼、後續 R/P 臂全部順序執行，CPU 統計也採單 phase，避免 RAM 疊加。檢查實體 UUID、即時占用及 cooperative lock；不使用 GPU 1/2/3，不 fallback 到其他卡。舊 frozen campaign 的歷史配置不改寫。

NG87 每臂約 5.7 小時／8,192 updates；本 pilot 同吞吐下每臂約 2.9 小時，僅是初估，未含新 teacher cache、全背景編碼／排序和 preflight。全背景耗時先用有限批次外推，不承諾整輪 ETA；硬時間／RAM／磁碟上限與所有 inputs hashes 必須在啟動前凍結。ClearML 能在線則在線，否則明記 offline；不自動恢復已暫停 polling。

本方案不授權更改產品／引擎、存取 locked test 或部署。執行進度另記，不把計畫當結果；重要方法與結論留 Git，產物依本地 storage policy 登記。

## 8. 文獻依據與邊界

- [SPLADE-mask](https://arxiv.org/html/2112.09628)：提供 training-time TopK 和 masked/unmasked 排序蒸餾先例。本文採 frozen endpoint、forward KL、完整 hybrid 的受控變體，不冒充論文原配方。原文沒有建立 query latency 結論，不能以它證明高 DF 已解決。
- [SPLADE-v3](https://arxiv.org/html/2403.06789v1)：支持成熟 sparse warm-start、較強 supervision 和蒸餾設計的重要性，但其多 loss 與更多 negatives 的成果不能直接外推為本項目下一步必然有效。
- [DF-FLOPS](https://arxiv.org/html/2505.15070v1)：說明 per-vector sparsity 不足以控制 posting list 長度；其主要設定是 binary lexical query 的 SPLADE-Doc，不能把速度倍數套用到我們的雙端 learned output。採納成本測量的理由，不重啟已失敗的 M1518 配方。
