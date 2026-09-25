# NG-0081：以整體 hybrid 品質與 posting 成本為目標

2026-09-14。狀態：經歷史回顧修訂的研究設計。使用者另已授權開始新探索；先執行下述 NG81-A 固定模型對照，四臂訓練仍待前置結果與配置凍結。這份文件不授權恢復輪詢、使用 locked test、替換模型或部署。

## 1. 目標與已知證據

使用者本次明確設定：完整 BM25+SAE 方案整體超越 dense 即可，不要求 SAE 單獨超越 dense，也不要求每個領域都超越 dense。這是後續研究判準，不回寫歷史實驗的既定 gate。

主要品質指標是預先固定任務權重的 overall nDCG@10；all-positive Recall@100 是另一個必報品質維度，不能以排序平均分掩蓋召回損失。現有三域診斷沿用等權 macro；擴展任務時先固定任務清單和權重，再看結果，不按題數或結果調權重。分域勝負、逐題損益和最差域保留披露，但不再具有「單域未勝就否決整體方案」的規則。

主要對照為 PPLX dense；TRAIN 校準的 dense+BM25 是強參考，不是已證明的品質上界。報告必须分別說明「超越 dense」、「超越 dense hybrid」、「相對原 hybrid 改善」，不能混稱。獨立測試的排名優越性以配對區間判定；Recall 若存在下降或區間仍不能排除下降，明列其不確定性，不自行新增容許損失來宣稱完整達標。

依據 [NG80 完成報告](ng0080-sparse-endpoint-review.zh.md)：

| 方法 | Macro nDCG@10 | Recall@100 | 總 literal DF-work / 原 hybrid |
|---|---:|---:|---:|
| 原 hybrid | 0.770716 | 0.948846 | 1.00x |
| S hybrid | 0.786079 | 0.951925 | 7.57x |
| P hybrid | 0.775382 | 0.946605 | 1.95x |
| PPLX dense | 0.822175 | 0.965235 | 不適用 |
| TRAIN dense+BM25 | 0.820317 | 0.971423 | 不適用 |

S 已收回約 29.9% 的原模型 nDCG 差距，支持在成熟底座上學習 teacher score-field。但 document NNZ 增至 4.02x，且全局品質仍落後 dense。因此即使取消逐域勝出的要求，NG80 仍未達標。這些品質是歷史曝光 TRAIN_DIAGNOSTIC 結果，DF-work 是代理，不是產品延遲。

目前成熟底座是 NG3 的 Transformer+MLM vocabulary sparse head。研究中稱它為 semantic/SAE 路線，不表示它已換成一個從零訓練的新 latent SAE。NG81 不先更換底座、詞表、context、tokenizer 或工程索引格式。

## 2. 歷史回顧：哪些不再重做

| 歷史證據 | 已回答的問題 | 本輪處置 |
|---|---|---|
| [M160A C6 post-hoc](../m0100-m0199/ii42-m160a-c6-posthoc-fusion-diagnostic-report.md) | 在 886 題混合表面，RRF、additive/residual/excess、多權重/候選配額及 runtime gate 都試過；沒有一種策略全面改善所有指標 | 不重開 RRF、配額或 adaptive gate 大網格；其選參表面不是當前獨立結果 |
| [M550 trained fusion](../m0500-m0599/ii42-m550-bm25-aware-trained-fusion-report.md) | dense-equivalent M549 加小權重 BM25，broad10 heldout nDCG 相對 exact dense +0.01515、Recall +0.00624；固定混合優於 learned gate，自由 residual 大幅失敗 | 已知融合可以提高整體品質，不再把這件事當新假說；但 dense-equivalent surface 不是當前低成本 sparse 模型，heldout 用於當時選型也不是今日的新 test |
| [M1327-M1329 soft-rank](../m1300-m1399/ii42-m1327-m1329-soft-rank-fusion-report.md) | risk7 的小幅改進沒有在 shared15 保持；更細權重搜索不值得繼續 | 不重做 rank-source 插值與逐題 oracle gate |
| [M1930A](../m1900-m1999/ii42-m1930a-bm25-semantic-complementarity-report.md) | BM25/semantic 候選有互補，但 union capacity 不是固定 top-k 排名成果 | 明確區分 union touch/capacity 與最終 nDCG/Recall |
| [M1951 fixed-query calibration](../m1900-m1999/ii42-m1951-opensearch-fixed-query-calibration-report.md) | RMS 校準相對 OpenSearch parent 的四項 macro 品質均升，nDCG +0.009409；因 FiQA row floor 等歷史 gate 未通過而停止 | 保留歷史判定，但不把「不逐域安全」誤讀為「不可能 overall 有益」；也不是相對 PPLX dense 的勝出證據 |
| [M1518 DF-FLOPS](../m1500-m1599/ii42-m1518-df-flops-paired-smoke-report.md) | 同強度與 budget-matched 正則都能降低 head concentration，卻仍触及全部 sampled documents；出現向中低 DF 維度擴散 | 不直接重跑 DF-FLOPS；先檢查新 source 是否仍有 universal support/union 飽和，成本臂須先證明能改變真實 support |
| [NG71/NG78](ng0078-boundary-coverage-review.zh.md) 與 [NG80 P](ng0080-sparse-endpoint-review.zh.md) | boundary/retention/完整 hybrid 監督不是新方法，也不能憑局部 pair 成功推全庫品質 | B 仍是待證假說，只有保留 G 的匹配增量對照通過後才擴展 |

另外，[NG71 D](ng0071-final-pilot-review.zh.md) 在自己的已曝光 DEV 上已做到相對原模型 nDCG +0.007824、document NNZ 0.9604x、semantic query-DF 0.3706x；它不是「品質改善必須增密」的例子，但仍低於該表面的 dense 0.045361 nDCG。這個現成的低成本方向應保留作後續同表面對照，不先斷言必須加入新 DF loss。它與 NG80 的評估 query 不同，不能直接拿兩個 macro 分數判斷 D 與 S 孰優；NG81-A 首批只使用已具備相同 query/document code 的原/S/P，沒有把 NG71 D 混入比較或宣稱它已通過新 gate。

所以 NG81-A 是一次有停止條件的 **新 endpoint 公平比較**，不是新的融合發明。不同的是 NG80 產生了新 S/P 分數分布，而且此前只有 dense 一側做過 TRAIN 校準。若它不能消除主要差距，停止 scalar/normalization 搜索，把結果交回表達與監督設計，不再細化 alpha 網格。

## 3. 先補公平的融合對照，不先增加訓練

NG80 student 使用固定原 RMS 配置及 BM25/semantic 0.1/0.9；dense hybrid 則在 384 個 TRAIN 題上校準，使用 full-corpus per-query z-score。故目前差距同時含有模型能力與融合配置差異，不能全部歸因於 sparse 表達損失。

NG81-A 固定原模型、S、P、dense 四個 endpoint，不更新模型參數，做以下對照：

| 對照 | 目的 |
|---|---|
| 原 serving normalization + 原權重 | 重現 NG80 的歷史錨點 |
| 原 serving normalization + TRAIN 校準權重 | student 能否只改一個全局融合係數受益 |
| 所有 endpoint 同用 per-query z-score + TRAIN 校準權重 | 移除不對稱融合校準，做公平離線品質比較 |

校準沿用事先固定的 384 個 TRAIN query、同一 alpha grid `[0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]`，按 TRAIN macro nDCG、Recall、較小 lexical 權重打破平手。每個 endpoint 可選一個全局 alpha，不能按診斷領域或逐題 oracle 選通道。384 題屬 TRAIN，不宣稱是獨立驗證；其與參數更新題的重疊須列入 exposure manifest。

在相同 normalization 下，融合公式為：

$$
h(q,d)=\alpha\,\bar b(q,d)+(1-\alpha)\,\bar s(q,d).
$$

其中 alpha 是分數權重，不是兩路候選配額。每組須重新以完整背景排序，不能只在兩路 top-100 的聯集上重排後當作全库融合。已封存 document code 可重用；若 S/P 的 384 題 query code 未存在，補做 TRAIN query encoding，不能假定所有輸入均已缓存。

full-corpus z-score 是品質診斷，不是低成本線上算法。若收益只在該 normalization 出現，須另驗證可部署的 TRAIN 固定尺度或模型/語料 moment 方法，對排名等價或品質差距實測；不能讓線上每題先掃全庫以取得尺度。這個工程可行性檢查在追加模型訓練前完成。

輸出為每種 normalization 的完整 macro/分域/配對品質表、alpha、原始分數尺度及校準成本。既有 1,536 題只用於解釋歷史差距，不據此挑配置再稱獨立成功。

這一對照的依據是可校準凸組合的融合研究，而不是假定任何固定 alpha 或 normalization 普遍最佳：[An Analysis of Fusion Functions for Hybrid Retrieval](https://arxiv.org/html/2210.11934v2)。其附錄亦包含 SPLADE+BM25，而不只 dense；其實驗主要在 top-1000 聯集上做補分融合，不替代本輪完整背景驗證。線性 normalization 在單題上可轉換為 rank-equivalent 權重，不代表一個全局 alpha 在每題都等價。因此只保留兩個有不同可部署含義的 normalization 對照，不另加 min-max/RRF 多路搜索。

## 4. 條件性四臂訓練：幾何保留、hybrid 排序與成本分開識別

NG81-A 與資料凍結通過後，才執行 NG81-B。四臂从同一原 NG3 parent 開始，使用同一 TRAIN 題、候選、全部已知正例、資料順序、optimizer、context、步數與 serving-compatible scoring。S endpoint 是外部診斷參考，不直接拿增密 4x 的 S 當所有新訓練的初始化。

| 臂 | Score-field G | 完整 hybrid 邊界監督 B | DF-aware 成本 C | 要回答的問題 |
|---|---|---|---|---|
| G | 有 | 無 | 無 | 同條件下重現 S 方向 |
| G+B | 有 | 有 | 無 | 直接優化最終排序是否超出純幾何對齊 |
| G+C | 有 | 無 | 有 | 在幾何學習時控制 DF，是否避免增密 |
| G+B+C | 有 | 有 | 有 | 互補排序是否讓有限 posting 預算更有效 |

不是把新的 loss、資料、底座和 normalization 同時換掉後比較兩個結果。共同配方寫成：

$$
L=L_{\mathrm{field}}+\beta L_{\mathrm{hybrid\ boundary}}+\lambda L_{\mathrm{cost}}.
$$

G 保留 NG80 的 centered semantic score-field 目標與 TRAIN 校準尺度。β、λ 只在 TRAIN preflight 決定並封存，禁止用診斷或新 DEV 反覆掃係數。四臂使用同一個可部署 normalization 及訓練期 alpha；終點另外給所有臂同等 TRAIN alpha 再校準機會，並保留固定 alpha 表，避免融合重校準遮蔽訓練效果。

### 4.1 B：讓總分為有效排名負責，不再追所有分差

BM25 保持固定，但放在可微目標的完整總分 h 內。監督優先覆蓋 top-10 排序及 top-100 召回邊界，保留背景對照；不是讓 SAE 必須獨自解決每個 query。

若 lexical 分差已令某對文件在完整 hybrid 中正確分開，該對不應持續要求 semantic 通道重複製造更大的同向分差。採用候選形式：

$$
L_{\mathrm{hybrid\ boundary}}
=\mathbb E_q\sum_{(i,j)\in P_q}
w_{qij}\,[\rho_{qij}-y_{qij}(h(q,d_i)-h(q,d_j))]_+^2.
$$

y 是有來源與信心標記的相對次序，rho 是事先固定且有上限的足夠分差，w 在每題內正規化。已满足分差時，此 boundary 項為零；**G 仍可能要求額外 semantic 保真，因此是否真的降低冗餘，必須由四臂交互效果驗證，不能從公式直接宣稱。**

可靠 relevance judgment 可糾正 teacher；未判定文件不能自動標為不相關。只有 teacher 次序的 pair 應標為弱排序蒸餾，依 TRAIN 上的 teacher margin 信心加權，不冒充人工負例。原始與 LLM 補充 labels 分開追蹤；teacher/label 衝突不以 teacher 強行覆蓋。沒有足夠判斷的資料只能支持 teacher 保真結論，不能證明已挖掘出優於 teacher 的 relevance。

候選由 dense、BM25、原 hybrid 的 TRAIN head/boundary 聯集、全部已知正例及固定背景樣本形成。同題各臂初始候選完全相同。第一輪不再擴大 K 作為主要變量；只在有證據顯示新模型的主要失分競爭者不在池內時，另做各臂一致的固定時點候選刷新對照。

rho 的尺度、head/boundary/background 配額、judgment 信任規則及 β 必須先用 TRAIN 數值/梯度檢查固定。不能直接套舊溫度，把幾乎全部 margin 壓成接近 1 的分類目標。已知 NG80 的 teacher target 飽和不等於 student BCE 梯度必然消失，兩者不能混說。

這不是宣稱發明新的 ranking loss，也不是直接重跑 P。P 已在完整 hybrid 上做正例保護而未得到可靠整體增益；本輪的明確差異是保留 G、使用有上限且聚焦排名邊界的約束，並有 G/G+B 成對對照。也不重試已失敗的 `dense - BM25` 分數殘差編碼：lexical 只進入總分，不把相減後的數值當非負 sparse code 的必達目標。

SPLADE-v3 的研究支持在成熟 sparse 底座上結合分差與前排蒸餾，但不證明本配方有效，也不支持單靠增加 negatives 解決跨域泛化。本輪不照搬其 loss 係數或一次引入多個新 teacher：[SPLADE-v3](https://arxiv.org/html/2403.06789v1)。

### 4.2 C：先跨過已有失敗機制，再引入訓練

固定語料和 TRAIN query 分布下，語意 literal posting 工作量為：

$$
C_{\mathrm{post}}=\mathbb E_q\sum_j
\mathbf 1[z_{qj}>0]\operatorname{DF}_j.
$$

另記總 NNZ/bytes，以及 query 實際 support 聯集觸及的文件比例；不能用獨立 feature 假設的機率乘積代替精確聯集。短 posting 很多與少量極長 posting 是不同問題；DF-sum、unique touch、WAND 實際讀取又是三個不同量。BM25 的固定成本也計入完整 hybrid 報告，不能只展示 semantic 省下的比例。

C 使用 TRAIN 語料抽樣的週期性 DF 統計，並考慮 query 激活頻率，作為訓練期成本權重。DF 估計樣本须獨立於 mined candidate 的頻率偏差；在一個訓練區間內固定，按事先固定時點更新，不每個 minibatch 重建全索引。試驗可採固定 8,192 文件樣本、每 256 更新重估，這些是待 canary 核定的建議，不是已驗證配置。

DF-FLOPS 提供 document-frequency 加權稀疏正則的直接依據；其 SPLADE-Doc query 是 lexical 形式，不能原樣視為本項目 learned query/document 雙端的成本解。[DF-FLOPS](https://arxiv.org/html/2505.15070v1) 與 M1518 的差異已指出：只改正則分布或匹配 loss 幅度可能仍全庫觸及。故 C 暫不指定為「DF-FLOPS 再加一個係數」；先從固定模型診斷識別 universal feature、query 活躍率及冗餘 support，只有 TRAIN 有界 canary 能降低精確 work/聯集且保留排序時，才凍結具體 C 並執行四臂。這個前置 gate 不通過就不啟動 G+C/G+B+C。

連續激活正則不等於 support 計數，可能只縮小數值而未縮短 posting，RMS 也可能抵消縮放。因此 C 必須先通過幅值縮放、零點、shared query/document head 梯度及實際序列化 NNZ/DF 對照；若代理下降但 support/DF 未下降，判為成本配方無效，不加大訓練掩蓋。

第一個成本預算錨點建議為原 hybrid 的 literal work，而非增密 S 的 work。這是學習曲線參考，不把代理 budget 當產品延遲 SLO。可保留較高成本但更高品質的 Pareto 候選作後續研究，不以它宣稱低成本目標已達成。

## 5. 與訓練解耦的低成本 mask 診斷

使用固定原/S code，在 TRAIN 上比較共同 feature mask。先測語意通道，BM25 不刪；兩通道一起剪會讓目前的識別問題再次混合。這不否定後續統一 posting，而是先確認有無可安全回收的冗餘。

- 對照一：只依 DF 移除 feature。
- 對照二：依完整 hybrid 的前排/召回邊界損失，選擇效用相對 posting 工作量較低的 feature 組合。
- 兩者用相同目標 work 預算比較，最終把整組 mask 一次套用後完整背景排序；不累加單 feature 消融增益當 joint 結果。

初次使用 binary shared mask，同步作用於 query 和 document，固定原 scoring denominator 以隔離 support 效果。若改用 soft mask，需明確定義乘一次或 query/document 各乘平方根，避免不小心變成 mask 平方。這不是 query-specific label oracle，也不移除文件或跳過全部已知正例檢查。

mask 是索引/推理側診斷，不能直接減少 encoder FLOPS。高 DF 不必然無用，某個高 DF feature 可能恰好處理重要區分；多個看似次要 feature 也可能共同必要。若 S 在原成本附近仍保有大部分品質增益，才有理由把此 mask 或其偏好蒸餾回訓練。若無法保留，就需要學更好的 support 分布，而不是更激進地刪。

索引隨 CRUD 週期性更新 mask 是另一個工程階段，NG81 不實作生產生命週期變更，也不把語料自適應 mask 當跨語料泛化已解決。

## 6. 規模、泛化與成本的晉級順序

1. 先完成 NG81-A 及固定模型 mask 診斷；取得完整對照，避免尚未理解融合就追加訓練。
2. 四臂從小 canary 通過後，使用 NG80 同量級的 12,288 個獨立 TRAIN query、每臂 3,072 更新作首次配方比較。初期 canary 只驗數值與成本，不以十幾題品質選模型。額外 B/C 開銷、document/token work 均報告；匹配更新與曝光不冒充 FLOPs 完全相同。
3. 用事先劃定、未參與本輪選參的新 DEV 確認方向。現有 1,536 題因歷史曝光只作診斷。先建立 source/topic/query-family 去重與曝光清單；若找不到真正新 DEV，先準備資料，不把已看過題目換名成 holdout。
4. 有穩定品質/成本收益後，只擴展勝出配方到約 50,000 個有來源的獨立 TRAIN query，再考慮 100,000 以上。採同題數同曝光的「原任務分布」與「更多任務/意圖」對照來識別多元性；不能用更多 pair 或 paraphrase 數冒充更多獨立排序資訊。新資料的存在、授權與標註品質需先確認。
5. 最終候選用三個 training-order seed、預先固定的較廣任務集合與未用於選型的獨立測試評估。僅反覆查看同一 DEV 不能支持最終整體勝出；若使用 NG67 locked test，須另行授權和凍結測試計畫，本提案不讀取它。

不再把「另一域點估計下降」本身當阻止擴展的理由；也不以單一強域重複加題推高 overall 權重。任務集合及權重變更要給同版本所有 baseline 一起重跑，不能把不同集合的分數接成一条進步曲線。

品質候選與產品升級分開。進入實際索引後，測完整 query encoding、BM25+semantic 檢索、候選/metadata 開銷、warm P50/P95/P99、cold 行為、總磁碟/RSS、構建和增量維護。固定硬體、query 集及 engine revision，先驗結果等價。dense 成本對照須區分 exact quality baseline 與同品質的 ANN 服務路徑，不能用 offline dense 全掃作便宜勝出的比較對象。

## 7. 用結果決定下一步，而非一直增加模型或 loss

| 觀察 | 對應行動 |
|---|---|
| 公平校準已顯著縮小差距 | 先處理可部署 normalization 和成本，不立即重訓 |
| G+B 勝 G，且 G+B+C 勝 G+C | 支持 hybrid 邊界監督有增量效果；再擴監督多元性 |
| G+C 保留 G 的品质並降 work | 幾何保真與成本可共存，優先簡單配方，未必需要 B |
| 成本代理降而真實 support/work 不降 | 修正成本代理的失效機制，不把數值變小當成功 |
| TRAIN 改善而新 DEV 無改善 | 優先監督來源、任務多元性及過擬合，不重複原樣加步数 |
| 新 TRAIN 與 DEV 均無品質改善 | 回查輸入可見性、teacher/label 衝突、梯度尺度與表達，不能只說資料不夠 |
| 品質提高但 native 成本不降 | 保留研究候選，定位實際 traversal/encoding 成本，不宣稱已完成總目標 |
| 整體獨立品質與成本達標，但個別域未勝 | 可以達成本次整體目標；完整披露域損益，不重新要求逐域冠軍 |

## 8. 執行與追蹤邊界

重要方法、程式、結論與精簡重現資料留在主代碼庫；大 artifact 沿本機研究儲存規則使用 NG-0081 單元並註冊，不改既有凍結 NG80 產物。本輪已新增有測試的固定融合評估程式，實際 source/input 身分與階段狀態另見 [執行進度](ng0081-fusion-progress.zh.md)；梯度訓練配方尚未實作或啟動。

當前自動輪詢保持暫停。未檢查即不宣稱 GPU 空閒。正式執行依當時可用資源分配一 GPU 一實驗；若用 Lambda2，遵守 GPU 0+3/1+2 的 NVLink 配對限制，不占用別人的作業。不為了湊四臂同時執行而超賣資源。

ETA 用實際 canary 的 TRAIN、全庫 encoding、ranking、independent review 分段推算，再決定長任務輪詢間隔；不先承諾總小時數或為取得品質結果放寬既定資源限制。沿用 phase exit、exact inventory/SHA、owned process closure 和 ClearML actual-start/closed 狀態審核；offline tracking 不冒充服務端已同步。

## 參考脈絡

- [Teacher geometry 與 supervision reset](../designs/ng-teacher-geometry-supervision-reset.zh.md)：區分表達、teacher 保真、真正 relevance、query 多元性及成本。
- [NG69 三 seed breadth 結果](ng0069-final-breadth-review.zh.md)：更多獨立 query 的收益與增密代價，支持有條件擴展而非永遠停在小實驗。
- [NG71 global-boundary 設計](ng0071-global-boundary-ranking-plan.zh.md)、[NG78 coverage 結果](ng0078-boundary-coverage-review.zh.md)：邊界/rival/retention 不是全新想法，也不是把候選 K 增大就能解決。
- [NG80 supervision audit](ng0080-supervision-audit.zh.md)、[S/P 完成報告](ng0080-sparse-endpoint-review.zh.md)：本提案的直接依據，不用本提案覆寫其失敗與不確定性。
