# NG-0072：TRAIN 排序能力保留診斷終局審核

更新：2026-09-13。狀態：診斷完成，Mac 獨立 margin／指標審核與完整產物 mirror 通過。這是 TRAIN counterfactual 的定位結果，不是新模型通過泛化／dense／native-cost 門檻。

## 固定範圍

按[預先固定協議](ng0072-retention-diagnosis-plan.zh.md)，比較初始與 NG71 D192 的四種 query/document code 組合。只計算固定 pilot384 與 sentinel384 TRAIN query，完整 233,009 文件背景；沒有 optimizer 更新、模型或 teacher 推論、DEV／LOCKED_TEST 評分。NG71 的 NQ 品質門檻失敗保持不變，不自動擴大訓練或修改生產服務。

Source／測試／協議 commit：`9c35375f`。新 run `NG-0072/cross-codes-v1` 的 `inputs.json` SHA256：`8ca39510880622470e7423f389a34d5dd5b06c68b504d0c47b6a7f5ec467b794`。Mac 與 lambda2 啟動前均核對 272 項依賴、三個 source／test／protocol SHA；原 NG71 封存資料未改動。272 項引用內容共 7,255,793,277 bytes，不代表本輪新生成資料量。

## 主要發現

**不是單一端全面壞掉，也不主要是 posting 被刪掉。** HotpotQA 的 query 端遷移抵消 document 端收益；NQ 的兩端各自都出現 sentinel 前排退步，document-only 的 nDCG 跌幅更大，但失去的 pair margin 同時受到兩端及交互項影響。不能用「全部凍結文件」或「只約束 query」概括。

以下均為完整固定 BM25+semantic hybrid 的 nDCG@10，每域每個 surface128題。00為初始雙端，10只換query，01只換document，11為D192雙端；並非四種新訓練結果。

| Surface／域 | 00 | 10 | 01 | 11 |
|---|---:|---:|---:|---:|
| pilot FEVER | 0.790954 | 0.829475 | 0.835539 | 0.867742 |
| pilot HotpotQA | 0.821257 | 0.830944 | 0.860409 | 0.867894 |
| pilot NQ | 0.621384 | 0.670135 | 0.677522 | 0.714451 |
| sentinel FEVER | 0.798729 | 0.816066 | 0.843125 | 0.856583 |
| sentinel HotpotQA | 0.846345 | 0.828277 | 0.864730 | 0.844501 |
| sentinel NQ | 0.651603 | 0.642118 | 0.629321 | 0.627006 |

HotpotQA sentinel 的 query-only 是 -0.018068，document-only 是 +0.018385，而雙端 -0.001844。NQ sentinel 分別為 -0.009485、-0.022282、-0.024597。nDCG非線性，這些變化不能相加，也不能將document-only跌幅除以雙端跌幅當成因果責任百分比。FEVER 的雙端都有遷移收益，因此一刀切凍結某端會捨棄已有改善。

NQ sentinel 的11 Recall@100反而由0.972331升至0.979818，nDCG卻下降。相對00，23題nDCG改善、36題變差，全部252個正例有54個rank改善、90個變差；top100正例進入2個、退出2個。Recall是先按每題正例數正規化再平均，因此進／出pair數相同不代表Recall差為零。這再次說明只看Recall或平均margin會漏掉前排損傷。

## Margin 與支援集合

按[協議](ng0072-retention-diagnosis-plan.zh.md)的有限說明集合，在原本正例勝過競爭者、最後反被超過的 `lost` pairs 上，使用相同全體競爭者分母，再正例／query平衡。百分比為該集合**負margin變化總量的代數貢獻**，不是nDCG責任占比，也不是文件posting刪除率。

| Sentinel | lost pairs | query項 | document項 | 交互項 | 共同support權重 | support失去 | support新增 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 226 | 79.83% | 11.67% | 8.50% | 72.45% | 25.60% | 1.95% |
| NQ | 276 | 39.67% | 41.31% | 19.02% | 69.95% | 19.71% | 10.34% |

兩組各自的三項相加為100%；modal分解和support分解是兩種座標，不能把六個百分比一起加。這些lost集合的各項均為負，所以此處能直接取其與負總量的比例。Support新增也可能令margin下降，例如競爭者新增的匹配得分大於正例。

NQ sentinel 對**全部**說明pairs的query平衡margin平均仍增加1.261174，HotpotQA增加1.287811，但前排已有退步。大量原本就贏的pairs的權重增長掩蓋了少數重要競爭關係損失。約70%的lost負margin来自**仍存在的matching coordinate權重變化**；只做support保留、Jaccard或調posting數量，不能直接對準這個問題。此處是query/document有效乘積support，不單是document的非零座標。

## 監督覆蓋的界線

NQ sentinel276個lost pairs中，81個涉及original positive、195個涉及added positive；HotpotQA226個中224個涉及original positive。不能將退步全部歸咎新增標籤。這些是pair次數，不是独立query數，也不是兩類正例的等分母錯誤率。

逐一對已審核的 `margins.jsonl` 按 `baseline_ahead=true && final_ahead=false` 分組、計數凍結teacher欄位：NQ sentinel276個lost中，original teacher對214個未觀察，46個支持正例、16個反對；HotpotQA為182／37／7。Sentinel沒有A0/A96 witness，不能據此判定高比例teacher衝突或標籤錯誤，更不能把unobserved當負例。

反過來，NQ pilot119個lost中，A96 teacher對77個支持正例、31個反對、11個未觀察；其中original-positive的25個為17／6／2。表示「teacher支持」並不自動保證整體共享參數更新保留那個pair的優勢，但也沒有證明是該pair自己的gradient直接令其退步。Token/evidence span仍未人工判定，不能把截斷欄位解讀成證據缺失。

## 下一個最小驗證

下一步先重放**既有**NG71 TRAIN更新中的pair score／gradient方向與覆蓋，不急著疊加新的loss、域權重或更多epochs。[目前soft-pair公式](../../../../scripts/ng71_ranking.py)對margin的導數是

$$
\frac{\partial L}{\partial m}
=\frac{c}{T_s}\left[\sigma(m/T_s)-t\right].
$$

即使teacher支持正例，只要student margin已高於 $T_s\operatorname{logit}(t)$，這個項仍可能要求縮小margin。這是程式可直接驗證的性質，**尚不是本次實際掉榜的原因證明**。應先按既有baseline前排／boundary對照，量化直接縮margin的更新、原本無監督的pairs，以及共享參數下的間接漂移；不能由平均score gradient推導參數LR倍數。

若上述證據支持，最小干預優先考慮只對可信、既有有效排序margin採取單側保留／滿足後不再壓縮的目標，而不是重建整個dense embedding或保留全部posting。是否採teacher floor、baseline floor或僅限制direct shrink，必須另做固定配置對照；不在本報告憑直覺選係數。要包含uniform-pair對照，分離rank權重與retention作用。維持共同初始化、候選來源、資料／曝光量、校準及成本門檻；pilot以外的sentinel不反過來加入gradient資料。

此外目前query/document共用trunk與head。僅在document forward上`detach`不等於文件encoder被凍結，後續query梯度仍會改變同一組參數。不得把這種detach對照冒充真實固定document模型。跨端codes診斷也不授權上線混用模型。

本輪足以把下一步從盲目控NNZ、全局凍結單端，收窄到**前排既有margin的保留與soft目標的方向性**；尚未證明新loss能超過dense，也不授權直接放大或部署。

## 實際執行與完整性

lambda2 worker於01:30:34 UTC啟動、01:38:40結束，CPU threads4、`CUDA_VISIBLE_DEVICES=''`；Spark-1僅作跳板，沒有占用GPU。ClearML `offline-d0f0ea397a9e4d36930ee83724042394` actual-start與自然close均已核對；仍未線上同步。

768題全部完成，loop433.38秒，worker總wall485.81秒，peak tree RSS4,463,812,608 bytes（4.16GiB）。初始canary的801.88秒是含1.5x計算餘裕与180秒收尾的投影，不是實測耗時；本次沒有NG71曾出現的約五分鐘額外close。不可把研究排序wall time當作native serving latency。

Terminal exit0、error=null、owned group已關閉，原tmux正常消失。完整13項phase產物及seal通過；整個run含controls／supervisor.log共19 files、59,699,539 bytes，rsync exit0後Mac完整inventory／SHA核對通過。遠端來源保留；mirror不是獨立backup restore。

Mac 準備時 available RAM 約 20.7 GiB，未達此協議的 24 GiB 下限，因此沒有在 Mac 啟動診斷 worker。遠端執行前 available RAM 約 112.8 GiB、磁碟可用約 717 GiB，CPU 低負載；同事 GPU1／2 工作保持不變。

## 獨立審核與測試

新診斷與既有 NG71 相關測試、navigation 測試共 `168 passed`；三組主要文檔測試另為 `134 passed`，兩個集合包含重複項，不能相加成唯一測試總數。新協議三條本地連結均可解析，`git diff --check` 通過；本輪沒有產品 source 修改，也未重跑產品全套測試。

[独立reviewer](../../../../scripts/review_ng72_diagnosis.py)，commit `cab2342f`，在Mac以較小的read-only payload完成：23.33秒、peak RSS1.57GiB，採4GiB RSS／8GiB host available／900秒界線，不是將原診斷worker的24GiB門檻放寬。它從凍結code以四組coordinate product另算全部stored top100分數及所有正例／競爭者margin與分解，最大head score差7.1054e-15；逐一驗證3,072 ranking rows、1,371 positive rows、原始00/11排名完全一致、gold／tie契約、teacher／lineage／未知證據欄位，以及獨立分母與summary。審核前後272依賴及sealed output均保持SHA不變。

審核沒有重新跑四次全庫排序；cross-code全庫rank來自producer的兩條完整score累加及rank檢查，reviewer獨立檢查保存的rank契約／metrics並另算所有head和margin分數。這個界線保留在JSON，不稱為另一輪independent full-corpus rerank。沒有新模型推論或DEV／LOCKED query評分。

新增review數學、空support、teacher缺失、query/positive分母與recursive比較測試，連同原診斷測試為 `18 passed`；原有168與文檔134的紀錄保留如上，不將重複集合相加。
最後將NG71／NG72相關測試與三組文檔測試合併執行，唯一收集結果為 `236 passed`（2.72秒）；本終局報告四條本地連結可解析，`git diff --check`通過。

主要證據SHA256（外部NG72單元，重要code／方法／本結論留Git）：

| 證據 | SHA256 |
|---|---|
| `diagnosis/complete.json` | `74f69b38d84456976810a89748b7ea7b2a9f50819471abb8dc527ef86c264671` |
| `cross-codes-v1-independent-review.json` | `f145c6ec65305d753e44dc79e0c8173a56bf79bd21a40226648e390d7d92929c` |
| `cross-codes-v1-copy-review.json` | `86a974ff4d35f9707ea1468715c2ba2282f78b83f0bd8915fd6c7f89d3c01103` |
| `cross-codes-v1-remote-final.json` | `cbb043d48a509442b4c88c867e88c57f3e1c459add26903a230b987e1ba8d61d` |
