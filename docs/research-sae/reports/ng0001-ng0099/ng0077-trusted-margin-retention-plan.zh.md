# NG-0077：从 Step 0 保留可信排序，而非限制所有變化

2026-09-13 UTC。接續[NG76原始中點結果](ng0076-authentic-midpoint-review.zh.md)。[CPU準備與獨立審核已完成](ng0077-trusted-margin-preparation-review.zh.md)：原384 TRAIN題／660 positives有45,175個可信anchors，初始top10可信比較全部在既有pool，穩定loss與fixtures通過。第一個canary的跨batch比較失敗仍保留；[上下文診斷](ng0077-batch-context-diagnostic.zh.md)之後，獨立凍結的[上下文正確CUDA gate](ng0077-context-qualified-canary.zh.md)已完成12題score/noise與三域shared VJP准入、closure／mirror及CPU核對。下一步是[固定lambda0／1的96步配對實驗](ng0077-matched-retention-pilot.zh.md)，不是宣稱已有品質提升。

## 假說與範圍

NG72顯示失去的關係主要受存活support的權重漂移影響；NG73發現多數已暴露lost對在score-space反而被要求擴margin；NG74均勻權重沒有解除NQ回退；NG75確認局部真實參數位移可由一階量測解釋，但窄rival集合不足。NG76現在確認同8題NQ在D96已失去前排品質，Recall@100卻仍完整。這些結果支持測試**可信舊關係保留不足**，不等於已經證明所有退化是catastrophic forgetting。

保持NG3 seed3003已訓練基座、shared trunk＋MLM head、PPLX教師、BM25 0.1＋semantic 0.9與原RMS。不新增DF刪除／posting cap／原始dense重建loss，不換base、資料域、optimizer或truncation。先96 updates，避免一開始同時改pool-refresh與retention。

這沿用[NG71完整歷史綜合](ng0071-global-boundary-ranking-plan.zh.md#1-從完整歷史導出的取捨)，不是只基於最近兩輪：早期dense幾何近乎保留不等於低成本倒排；成熟MLM比新latent字典更有依據；NG53–59的posting刪除不可加，NG60/61的局部可分不保證全庫不冒出新競爭者，NG66–70亦未支持盲目重複小資料或把teacher不確定性當成硬標籤。NG77要測的是在既有排序訓練中補一個可否證的保留機制，不重新包裝「換teacher＋KL」或更大K。

## 最小 Loss 候選

令完整hybrid分數為 $s_\theta(q,d)$，原基座為 $\theta_0$。可信gold/rival對的當前與參照margin為：

$$
m_\theta=s_\theta(q,p)-s_\theta(q,n),\qquad
a=m_\theta/T,\qquad a_0=m_{\theta_0}/T>0.
$$

只在當前margin小於可信基線時加入單側Bernoulli distillation gap：

$$
p_0=\sigma(a_0),\qquad
h(a;a_0)=
\begin{cases}
\operatorname{softplus}(a)-\operatorname{softplus}(a_0)-p_0(a-a_0),&a<a_0,\\
0,&a\ge a_0.
\end{cases}
$$

它在精確算術下非負，邊界連續且一階導數為0；違反基線時導數為 $\sigma(a)-p_0\le0$，因此該pair的直接score pressure不會要求繼續縮margin。改善基線不受罰，不強制embedding／全部logits不變，也沒有向無限大margin推動的固定hard-positive目標。**共享參數更新仍可干擾別的pair，這個式子不是全庫品質保證。**

準備階段比較此gap的標準庫／NumPy／Torch值及gradient，特別是a接近a0、極大正負值、空anchor、梯度累積與原D零係數parity。不能用不穩定的兩個大型BCE數值相減；要以FP64或經核對的穩定等價式實作，並記錄cast與求導契約。原baseline CPU CSR與當前FP32 forward的no-update誤差，須在不做optimizer的canary先量測／凍結處理規則，不能拿品質結果調deadband。

研究候選為 $L=L_D+\lambda L_{\mathrm{keep}}$，其中 $L_D$ 完全沿用NG71，keep按query／全部已知positive平均，未覆蓋positive仍留在分母；使用凍結的可信度及head/cutoff重要性，不用大量容易pair淹沒前排關係。第一個比較只允許零係數control和一個固定非零係數；**尚無實驗證明最佳lambda**，先查anchor數與loss／gradient尺度，再在新training manifest中選定一次，不以DEV／sentinel結果掃參數。

## Anchor 來源與先行 Gate

1. 從原NG71的384個TRAIN_PILOT queries及initial witness取資料；128題／域，沿用既有身份、來源解析、query順序及全部gold。不使用384個TRAIN_SENTINEL、其中24題NG75／76 probes、DEV或LOCKED_TEST選anchor。
2. 初始hybrid margin必須正，且支持關係來自人工judged-negative或PPLX對該gold/rival的明確正margin。teacher未觀察、tie或反對，均保留為unresolved統計而非自動negative；不憑基座分數把任意非gold認定無關。
3. 先用已有scores／CSR／witness作CPU audit：逐域、query、全部positive、rank1–10／11–100／100以外，統計可用anchors、missing／unknown、原D已見／未見。檢查是否真的覆盖頭部及cutoff，不能以總pair量大宣称充分。
4. 優先重用現有forward pool，避免同時改候選與loss。若必要anchors不在原pool，必须明確建立共同augmented-pool control，兩臂相同document曝光／計算；不得仍把它叫作完全原D。沒有足夠TRAIN可信coverage時先修正資料準備，不启动loss sweep或用sentinel補訓練。
5. 新unit登記catalog，凍結來源、全部parents、anchor identity／baseline margins／權重／scalar配置。CPU fixtures和無更新parity完成前，training-enabled保持false。

## 有界的配對訓練

兩臂從同一原base開始，相同384題順序／每題一次、96次AdamW更新、FP32／TF32 false、microquery1／microdocument4、原learning rates／weight decay／clip。兩臂執行相同anchor bookkeeping，只有非零keep項不同。分別保存初始與終端完整model／moments指紋；lambda0應匹配原D的score-space計算，但新CUDA執行不預先承諾歷史bit-exact。若要重用原D96文檔cache，必須實際確認終端model SHA一致，否則不能混用舊codes。

不在training內插入GPU觀察器。封存checkpoint並退出training process後，另進程編碼完整233,009文件與既定TRAIN評估query，計算完整hybrid top100和全部positive ranks。不只評24題；至少保留既有384個互斥TRAIN_SENTINEL的完整query-balanced分母。它們已曝光，仍只是探索性准入，不稱新held-out成功。原1536題DEV的用途／讀取時機和原floor須在execution manifest預先固定，這個準備階段不讀取它們作選擇。

報告retain成功與失敗、原來錯誤關係是否可改善、trusted／unresolved群、各域nDCG與Recall、query-paired不確定性。不能因TRAIN loss／anchors改善就自動延長到192步。延長、完整DEV檢查及獨立未曝光評估各設明確gate；NQ不刪、原非退化floor不放寬、不看LOCKED_TEST。若keep改善anchor却不改善廣泛TRAIN前排，先判斷代表性與shared-parameter transfer，不能連續換lambda掩蓋失敗。

估時以既有原D96證據為起點：96更新完整程序354.84秒，233,009份編碼662.93秒；不是只用不到1秒的NG76 query forward估算新訓練。新keep成本尚未量測，先最小canary，納入兩臂、全庫rank、closure／hash／mirror／independent review，預估40–70分鐘級而非承諾。各phase獨立guard、不放寬失敗attempt；備妥後依實測在5分鐘準備輪詢與10–15分鐘計算輪詢間調整。GPU新查空閒、單卡、不移除同事工作。

## 文獻依據與不適用的保證

[RankDistil，AISTATS 2021，§2–3，特別§3.3](https://proceedings.mlr.press/v130/reddi21a/reddi21a.pdf) 區分保留頭部內部次序與限制頭部之外的競爭者，並提供pairwise distillation形式。其一致性結論依賴教師／抽樣support等假設，不能直接套在固定小pool與不完整judgments上；本方案也不是其演算法的原樣實作。

[GEM，NeurIPS 2017，§3式6–8](https://papers.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf) 用不增加既有任務loss的單側約束允許正向transfer；其gradient角度近似需要局部性與記憶代表性。此處借用單側保留的設計動機，不引入QP／投影optimizer，也不把AdamW的raw-gradient內積当作實際位移。上述Bernoulli gap的導數與非負性來自softplus凸性，屬本輪待測工程選擇，不是文獻已證明能超dense的結論。

成本線先保持不變，以免同時解兩個問題；待排序干預真正改善，才在同一qualified模型上回到DF／posting成本和native總成本量測。最終成功仍是完整BM25＋semantic超dense且總成本較低，不是保住某幾個margin就完成。
