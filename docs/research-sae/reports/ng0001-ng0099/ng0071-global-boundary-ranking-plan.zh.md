# NG-0071：成熟基座上的全庫邊界排序強化

日期：2026-09-12。配置 v2。狀態：**有界四組pilot已在lambda2啟動；A0..96訓練已封存，正進行A96全庫編碼。尚無終局品質結論**。啟動及SHA證據見[準備／執行報告](ng0071-preparation-review.zh.md)。

本輪「強化」指有監督的排序 fine-tuning，不指 PPO、工具 agent RL 或更換生產模型。目標仍是 **BM25 + learned sparse 的完整混合方案，在前排品質、全正例召回及實際成本上勝過有競爭力的 dense 方案**。不要求語意單路取代所有 lexical 能力，也不以候選池 oracle 冒充可部署模型。

配套：[機器可讀配置](ng0071-ranking-config.json)、[數學參考實作](../../../../scripts/ng71_ranking.py)、[單元測試](../../../../tests/test_ng71_ranking.py)。配置是待驗證的先驗選擇，不是經實驗證實的最佳超參數；它不是可直接啟動訓練的 pipeline。NG69 原協議、已封存的評估和所有凍結 artifacts 不改動。

執行準備、來源選樣修正、監督飽和度與真實模型驗證見 [NG71 準備報告](ng0071-preparation-review.zh.md)。一個 disposable 工程更新不算正式 P0/P1，也不構成品質提升證據。

## 1. 從完整歷史導出的取捨

本節重新對照原始報告與跨階段總結，而不是重新執行所有歷史實驗。不同 corpus、曝光面、模型與引擎的數字不拼成同一 leaderboard。舊 NG 原報告仍在外部研究檔案中，本節保留足以決定新方案的主要結論。

| 路線與證據 | 已知的結果／限制 | NG71 的選擇 |
|---|---|---|
| M9–M353：重建型 SAE、atom retrieval | 重建可學；候選上界、dense 重建誤差與可部署排序／遍歷成本不是同一問題 | 不再把重建誤差當主目標 |
| M360–M601：signed geometry、M549U | dense 排序可以近乎保留，但完整 support 的低成本倒排沒有因此成立 | 不宣稱「dense 根本無法保留」，改問如何在可索引表示中保留決策 |
| M733–M1710：加 atom、routing、residual code | oracle 有益不代表 encoder 可泛化；高 DF、全 block 掃描和跨 query harm 反覆出現 | 不再從零做隨機／平衡字典、無監督路由或直接 dense-minus-BM25 |
| M1701、M1720–M1730 | 更強 CE、listwise、residual、候選擴張已做過；局部改善沒有共同解決完整排序與成本 | 不把「換 teacher + KL」或「更大 K」包裝成新機制 |
| M1904/M1905 同訓練條件對照；M1911 | 成熟 MLM 的排序／DF 明顯優於新 SAE；latent terms 可學不等於低成本 | 用成熟預訓練幾何，不先壓成新小字典 |
| M1918–M1951 | candidate heldout 改善可在 full-native 失敗；BM25 互補真實，但某些校準的 macro 改善遮住 FiQA 損失 | 對完整混合分數訓練，跨域與 full-corpus 驗收 |
| M1963、M1972/M1973 | raw M190 的 15-row 品質回退；compact dense correction 有用但仍有 row harm，且不是單層 posting 模型 | 保留為架構備選，不偷加 reranker 完成當前目標 |
| NG3、NG31–38、NG43–47 | 多正例 coverage、成熟 output basis 有價值；r64、active-union／residual、固定 support 和小集續訓不是未探索路線 | 保留成熟 trunk/head，暫不做梯度手術或 query-only 架構變更 |
| NG48/49 | 34K 文件上 dense core 可更快、更小；但固定 CPU 的完整 fresh query，hybrid p95 11.204 ms、PPLX 125.040 ms | 不再先犧牲品質壓小 encoder；也不外推為所有 dense 的成本下限 |
| NG53–59 | 單 posting 價值不具刪除可加性；前排改善仍可丟 recall；取消截斷不穩定勝出 | 不加入獨立 DF mask、不把 uncapped 當新品質突破 |
| NG60/61 | 59,111 文件上，hybrid top100 經 PPLX 重排 nDCG .922751，原 hybrid .893538；344/357 局部 LP 可分，但 103 題全庫退步，102 題出現聯集外新前排 | 主要假說是**有用的區分訊號未在真實競爭背景下學好**，不是已有容量／泛化保證 |
| NG62–65 | 共享梯度、AdamW 與有限更新不普遍失效；已在 pool 裡的正例仍能由 rank94 掉到102，雖 CE 分數梯度要求它升分 | 不能只修採樣；同時觀察每個正例的邊界、query/doc 兩端變化與其他 query |
| NG66/68/70 | 重複訓練 TRAIN 大升、DEV 證據弱；舊 pool 平均約16.8；teacher 機率集中；來源負例不是人工確定負例；NQ added-positive 比例高 | 不盲目加 epoch、加 CE 或把 unjudged 硬判成負例 |
| NG69 第一組 | 同233,009背景，B nDCG .790783 / R100 .964867，dense .830229 / .977456；B 比 A 的 NNZ 2.1393x、query-DF 2.5714x | 廣度值得繼續驗證，但不是已超過 dense；不得以更密的 posting 當免費收益 |

主要倉庫入口：[M9–M1710 結構總結](../ii42-unified-posting-structural-diagnosis-and-reset-report.md)、[M1900–M1920 成敗因素](../m1900-m1999/ii42-m1900-m1920-learned-sparse-success-failure-factor-report.md)、[M1900–M1951 里程碑](../m1900-m1999/ii42-m1900-m1951-learned-sparse-one-index-milestone.md)、[M1963](../m1900-m1999/ii42-m1963-m190-bmp-full15-report.md)、[M1972/73](../m1900-m1999/ii42-m1972-m1973-native-compact-dense-closure-report.md)、[NG66](ng0066-training-duration-review.zh.md)、[NG68](ng0068-supervision-and-ranking-review.zh.md)、[NG69 第一組](ng0069-first-pair-review.zh.md)、[NG70](ng0070-supervision-provenance-review.zh.md)。

外部 NG 證據定位：`NG-0048/history-overlap.zh.md`、`NG-0049/milestone-01.zh.md`、`NG-0053` 至 `NG-0065` 的 `milestone-01.zh.md`，相對於既有研究 artifact 根目錄。執行前把本輪使用的原報告 SHA 放入 source manifest，不修改原件。

**推論而非定論：**下一個最值得測的，不是更強重建器，而是「成熟語意 basis + 全庫競爭者見證 + 每個正例的排序邊界」的聯合訓練程序。teacher、資料可見性和表示容量仍可能限制結果；此方案專門設計為可以否證這個推論。

## 2. 基座、教師和混合計分

第一輪四組共同從既有 NG3 `seed-3003/coverage-off/checkpoint-096` 起步，延續 NG59/66/69 的成熟 Granite sparse 路線；不按已看過的 DEV 成績選 NG69 最佳 seed。這是共享 Transformer + MLM vocab sparse head，不是新訓練的 latent SAE。實際參數約49.65M，不能只按上游「30M」名稱核算資源。

- 沿用完整 trunk/head fine-tuning；PPLX 凍結為 teacher。**既有基座 fine-tune 與 PPLX 蒸餾並不互斥。**
- 固定 tokenizer、lexical vocabulary/IDF、RMS、BM25權重0.1、語意權重0.9；不在看到 DEV 後重新校準。
- 沿用 query64/document256 **輸入 token**，第一輪 full nonnegative output，沒有 Q64/D256 posting cap。這只為控制變量，不推翻 NG59 的截斷比較。
- 同一 FP32、no-TF32、query microbatch1/doc microbatch4、每4 queries更新一次、原 VJP replay 路徑。AdamW trunk `5e-6` / head `2e-5`、weight decay `.01`、累積後global gradient clip norm1；先不掃 LR。
- 不修改產品模型包、查詢函數、index format 或在線模型；未授權 production rebuild。

令原始非負語意表示為 $u_q,v_d$，凍結 RMS 向量為 $r$，固定 lexical score 為 $b$：

$$
s(q,d)=0.1b(q,d)+0.9\frac{u_q^T v_d}{u_q^T r}.
$$

所有訓練排序比較使用這個完整 $s$。BM25 沒有可訓練參數，但會改變競爭者、margin 和梯度需求：lexical 已經分得很開的 pair，不應要求語意通道重複花 posting；這是混合 margin 的作用，**不是已證明能自動去重或降低 DF**。正規化 denominator 必須有限且大於0；不為零 RMS 座標偷偷加 epsilon。

## 3. 核心目標：排序同時覆蓋前排和召回邊界

對全庫 $D$、全部已知正例 $P_q$，確定性 tie-break 後的排名為 $r_q(d)$：

$$
R@K(q)=\frac{1}{|P_q|}\sum_{p\in P_q}\mathbf{1}[r_q(p)\le K].
$$

因此 recall 確實是全庫排名的截斷函數。但只排好 candidate pool 或 nDCG@10，不能保證 R@100；即使所有正例排在負例之前，$|P_q|>K$ 時 recall 也不能是1。

以目前 binary qrels 為準，定义 $D_{10}(r)=\mathbf{1}[r\le10]/\log_2(1+r)$，$Z_q=\sum_{i=1}^{\min(10,|P_q|)}D_{10}(i)$。對正例 $p$ 和非正例 witness $n$，凍結全庫排名後使用：

$$
w_{pn}=\frac{|D_{10}(r_p)-D_{10}(r_n)|}{Z_q}
+\beta\frac{|\mathbf{1}[r_p\le100]-\mathbf{1}[r_n\le100]|}{|P_q|}
+\frac{\gamma}{|P_q|}.
$$

第一版 $β=1,γ=0.05$。前兩項是 binary qrel 下互換 positive/nonpositive 的指標變化；對 unjudged 只是 **qrel-conditioned priority proxy**，不是已知真實相關性。小 floor 避免兩文件都在100名外時零梯度，不能替代 rank100 附近 witness。這兩個常數是明示先驗，不是數學推導出的最優值。

令 $z_{pn}=(s_p-s_n)/\tau_s$，$\tau_s=1$。採用 soft pair BCE：

$$
\ell_{pn}=\log(1+e^{z_{pn}})-t_{pn}z_{pn},\qquad
\frac{\partial\ell_{pn}}{\partial(s_p-s_n)}=
\frac{\sigma(z_{pn})-t_{pn}}{\tau_s}.
$$

不要求 student 重建 teacher 的絕對 embedding 或每個 dot product。教師提供可信程度受限的**相對偏好**，以降低把 dense 全部背景／共性訊號照搬成高 DF posting 的壓力。這是待測假說，不保證僅靠改 loss 即可稀疏化。

### 標籤和 teacher 不確定性

1. 所有既有正例保留；原始／added lineage分開報告，不能因 added 就刪除。非正例 witness 若同時是任一既有正例，禁止作負例。
2. 人工明確判不相關的 witness：$t=1,c=1$。不能把 `source_negative` 升格為這一類。
3. Unjudged/source-negative：只在凍結 PPLX 偏好正例時使用 soft target $t=\sigma((T_p-T_n)/0.04)>0.5$，信心 $c=2t-1$。相同分數、相反偏好或不完整 provenance 標記為 unresolved，不給硬負例梯度。這不是已校準的人類相關機率。
4. 人工標籤衝突先隔離，不能用 teacher 多數決覆蓋。已知正例間沒有假造的 binary preference；未來真的有 graded labels 時需另版公式與測試。

每個 query 先對每個正例的 witness 做加權平均，再對**全部正例**等權平均：

$$
L_q=\frac{1}{|P_q|}\sum_{p\in P_q}
\frac{\sum_n c_{pn}w_{pn}\ell_{pn}}{\sum_n w_{pn}}.
$$

求和只含eligible pairs；無 eligible pair 的正例貢獻0，仍保留在外層分母並記錄「未受監督」，不得靜默丟掉。分母不含信心c，避免只有少數低信心witness時將小梯度重新放大。`w,c,t` 均 stop-gradient。這是受 LambdaRank/LambdaLoss 啟發的有限 witness surrogate，**不是論文中某個原式的直接重現，也不是全庫指標的無偏估計／單調改善保證**。平均梯度仍可能透過共享參數傷害某些 query，所以 NG64 類尾部 harm 必須直接量測。

## 4. 全庫見證，不是無限制擴大訓練 pool

用同233,009文件背景生成確定性全庫排名；只對 TRAIN query 採樣或調整。每輪保留完整原 pool 和全部正例，包括原本在top100外的正例，再增加最多64個去重 witness：

| 來源 | 新增上限 | 用途 |
|---|---:|---|
| Reference hybrid top32 | 24 | 真正影響前排的競爭者 |
| Reference hybrid rank80–120 | 16 | 距rank100最近優先；讓尾部正例看見召回邊界 |
| 凍結PPLX top32 | 8 | teacher鄰域保留／teacher-student分歧 |
| BM25 top32 | 8 | lexical獨有的排序競爭者 |
| 全庫其餘文件的固定hash樣本 | 8 | 避免只學最難幾份文件；不宣稱無偏估計 |

依表順序去重，先移除全部正例和既有pool；同優先級以穩定ID/hash定序，不用 DEV 分數。某來源不足不跨來源補滿。總候選超过256時 preflight停止，不能裁掉正例或跳過難題湊預算。保留原始ID、文本SHA、來源、teacher分數、完整positive mask、reference rank、checkpoint SHA、corpus SHA、refresh步數。

**索引快照只選ID與排序權重；訓練分數必須用当前 query/document encoder 重新 forward。** 不拿舊 document embedding 當作本輪可求梯度的輸出，也不每一步重編整庫。第一個機制對照用同一 reference 的 step0/96 快照；每個快照的 head/boundary/所有正例 rank 均由第二條全庫累加路徑核對。排名不能由80個訓練候選的相對位置冒充。

## 5. 可歸因的四組對照

第一輪384個TRAIN query（每域128，按固定hash選），兩遍共768 exposures、192 updates；同init、順序seed71001、optimizer、scoring、token長度、全positive集合。全部 query/文件 split 身份沿用 NG69，禁止把 locked test 用於 mining、teacher新增推論或early stopping。

配置 v2 在固定 hash 前只隔離 NG70 已知 provenance unresolved 的11條 FEVER query，不按模型得分／teacher信心／難度挑題。Pilot384與canary96不變，只有sentinel兩題按同hash序列遞補。來源困難查詢仍留在 NG69 原始比較；下一次擴大到6,144個來源合格query前必須先處理這11條缺口，不能重複補數。

| Arm | 候選策略 | 目標 |
|---|---|---|
| A | 完整原 pool | 舊 `.5 gold + .5 teacher` CE |
| B | 完整原 pool | 本文 balanced soft pair objective |
| C | 原 pool + 有界全庫 witness | 舊 CE，按擴充 pool 重算同公式 target |
| D | 與C完全相同的 witness manifest | 本文 balanced soft pair objective |

A在step96產生共同reference快照，C/D第二遍使用它；第一遍使用共同init快照。B/D的排名權重亦使用相應共同reference全庫rank。A/B第二遍仍是原pool。這避免 loss 組各自挖出的不同文件混淆第一次比較，代價是快照不一定最適合C/D當前模型。實際 next-stage self-mining 另列，不能把此輪稱為已驗證完全自適應。

看 `B-A`、`C-A`、`D-C` 和 `(D-C)-(B-A)`，而不是只報D最高。CE與pair的差異同時包含損失形式、positive balancing及不確定性處理；**不能僅凭四組把收益歸因於指標權重本身**。若D有優勢，必做同manifest、同target的 uniform-pair消融（只令 `w=1`）；再決定保留複雜權重與否。

候選數、token/FLOPs、teacher額外推論、全庫快照成本均不完全匹配，逐組報告。以相同updates作機制比較，再以實際GPU時間和編碼文件量作成本比較，不能聲稱equal-compute。

## 6. 執行順序、配置和晉級

### P0：零更新的 TRAIN-only preflight

- 等待 NG69 凍結評估和獨立 reviewer 完成；不用其當前單seed選基座或改協議。先核對新增資料的provenance、token截斷可見性、existing state SHA及完整 no-op輸出。
- 96個TRAIN canary（每域32，從384中固定hash選），生成witness和方向診斷；不把這96題當泛化結果。
- 檢查所有正例是否在模型256-token窗口內有支持證據；未知就記unknown，不因 teacher 用較長文本就偽造 student 可見性。單列原始／added正例、有無截斷、BM25-only可見詞，先不一起改長度。
- 數學參考實作通過後，仍需在真實Torch/VJP路徑核對雙端梯度、normalization、有限更新、checkpoint reload、snapshot SHA與rank scope。參考實作不是完整trainer驗證。
- 每域至少80%正例有一個eligible pair，否則停在標註／teacher診斷，不靠降低標籤安全要求補滿。此80%是執行覆蓋門檻，不是品質結論。
- 按head／rank80–120／PPLX／BM25／hash witness分層記錄正例監督與teacher分歧，另報跨top10／100邊界中被遮掉的metric-priority mass。原pool的99%「至少一個比較」覆蓋不能代替對真實排名對手的有效監督。
- 在每份完整384題manifest上另外檢查固定4-query batches。個別query零eligible可保留零loss和原曝光計數；整個batch無eligible則停止並記failure，不能只做weight decay當成有效update，也不悄悄換題／補步。不能直接沿用舊trainer「每題梯度必非零」的assert來處理新objective。
- 記錄soft target分位數、接近0.5／超過0.99的比例和實際confidence-weighted梯度量。沿用teacher溫度0.04不等於已知適合pair蒸餾；若soft targets實際全飽和，必須在獨立TRAIN-only溫度／校準對照中驗證，而非宣稱已保留更多dense幾何。
- 正式啟動前封存 source、config、模型、teacher、RMS、lexical、query/doc/qrel、provenance、witness manifest及依賴SHA。出現缺項時不自動下載新版本頂替。

### P1：384題四組機制試驗

每組固定terminal step192，不選最佳checkpoint。step0/96/192記錄全庫行為；step1/24只作有限canary診斷，不能拿子庫rank冒充global rank。

未參與梯度的TRAIN sentinel384題（每域128）與已曝光DEV1536題，所有組訓練封存後才做terminal比較。Sentinel仍是TRAIN，不冒充holdout；DEV已被NG69曝光，不叫fresh test。現有LOCKED_TEST1536題保持未編碼、未評分。

觀察不只兩個均值：全部正例rank、出入top10/top100、池外新entrant、其他query的harm、reference排名過時程度、teacher分歧、每域正例監督覆蓋；同時記query/doc NNZ、DF直方圖、高DF尾部、query-DF、score margin、梯度及參數變動、每輪重編文件量、RSS/VRAM/磁盤、訓練與快照時間。

以同query配對、分域等權bootstrap：D對A的macro nDCG提升至少0.005且95%下界大於0；Recall差95%下界不低於-0.002；各域nDCG與Recall點差均不低於-0.005。B/C可解釋機制但不替D補考；no-effect可能是power不足，不等於模型不可能。這些是**研究晉級門檻，不是生產無回退承諾**。

若相對共同init的document NNZ或query-DF平均超過1.25x，完成可安全完成的科學記錄但禁止直接擴大；轉成本歸因。document NNZ使用完整233,009文件；query-DF gate使用terminal已曝光DEV1536題同一query集合，TRAIN另列。這不是延遲倍數估計，也不靠中途修改loss修飾結果。記錄D/C、D/A成本比；不能用A自身膨脹掩蓋對init的增加。

### P2：條件式放大，不永遠停在小實驗

通過P1後，先以uniform-pair消融判斷metric weights是否值得，再凍結最簡有效方案。用三個TRAIN順序seeds與6,144不同queries的一遍訓練驗證；不把順序seeds說成獨立初始化。這時每1,536exposures才以**各自當前模型**刷新全庫witness，最多step0/384/768/1152四份快照；新增階段另凍結協議，不默認已被P1授權。

更大corpus、更多域與有來源的人類監督是下一個放大維度；不是同時增大模型、字典、context和epochs。單域或高度熟悉DEV的成功不足以晉級。先選定新域和終局確認集，再做一次性frozen確認；舊已曝光的NQ/FiQA只能作regression，不能冒充新holdout。

### P3：把成本納入訓練，而不是訓練完亂刪

只有排序機制通過，才做**獨立的 constrained-ranking 對照**：

$$
\min_\theta L_{rank}(\theta)\quad
\text{subject to}\quad
C_{work}(\theta)\le B_{work},\quad C_{bytes}(\theta)\le B_{bytes}.
$$

先固定相對合格parent的預算，再用TRAIN-only平滑代理作Lagrangian／dual更新；不先拍一個大FLOPS係數。零更新counterfactual需要直接移除**整組**posting後重新算hybrid pair margins，因為 $0.9u_{q,j}(v_{p,j}-v_{n,j})/(u_q^Tr)$ 只表示固定表示與正規化下的單語意座標區分貢獻，並不保證聯合mask或重新正規化後可相加。涉及lexical mask時還必須重新計入BM25差值。

查詢分布下的 $\sum_j P(q_j>0)\,df_j$ 是未剪枝遍歷工作代理，比單純document NNZ貼近問題，但不是native latency。可借鉴DF-FLOPS只壓問題尾部；必須同時保留低DF但重要的證據和高DF但有區分力的證據。平滑activation/DF代理的公式、溫度、dual步長及原生預算必須在該階段另立配置；本輪不把未實作的代理假稱為已解決。

用native同一混合索引核對quantization後的品質、實體bytes、postings/block visits、完整fresh query warm p50/p95/p99、encoder成本、build/maintenance重寫量；dense對手也包含encoder，並從TRAIN選合格的exact/ANN/量化profile。NG48/49已證明，單比cached core或故意選慢ANN都不公平。達成品質但尚無原生成本證據只能稱quality candidate。

部署期的自適應mask留作後續工程：canonical posting保留、mask與query view同generation、背景發布、舊查詢快照可繼續服務。它不能替代模型品質驗證，也不授權本輪修改production lifecycle。

## 7. 結果如何改變下一步

| 觀察 | 下一個有界動作 |
|---|---|
| C改善、B/D無額外收益 | 保留舊CE，採用有效的witness策略；不強留新loss |
| B改善、C無收益 | 比較uniform-pair與metric權重；檢查新witness標籤品質和刷新成本 |
| D改善且recall安全，但NNZ/DF大增 | 不直接擴大；轉P3成本/區分margin診斷 |
| 局部margin改善，全庫／sentinel退步 | 量化新entrant和reference staleness；只增加一次self-refresh對照，不先換模型 |
| 原始標籤、added標籤或teacher衝突集中在失敗處 | 在TRAIN按分歧×邊界×長文本分層作盲化人工pair標註，加獨立no-new-label對照；先用現有84題卡片，不把LLM票數算人類標註 |
| 長文本證據不可見 | 單獨做longer-window對照並計算encoder成本；不以更多epochs處理輸入缺失 |
| 局部也不可分，且teacher可信、文本可見、訓練梯度正確 | 才比較成熟基座解凍範圍／更大成熟基座或token-level latent basis；不是直接PPLX從零重建 |
| 無穩定收益 | 保存negative result，檢查效應區間；不自動掃更多beta、LR、K和teacher |

## 8. 資源、安全與交付

依使用者最新指定，NG71在lambda2執行，spark-1僅作已授權的SSH跳板；僅在空閒、依賴與numerical no-op通過後啟動。每實驗一GPU，明確綁定、cooperative lock及實際utilization雙檢查；不要占同事GPU。未規劃多卡訓練；真需要時只允許0+3或1+2。

沿用CPU4、treeRSS16GiB、GPU allocated20GiB的起始上限；host available大於24GiB、artifact mount free大於40GiB。P0先估實際成本，90分鐘stage上限內無法完成就重新設計分段，不能為完成靜默放寬。新輸出只有獨立NG71路徑；Mac持久artifact依本地storage policy，暫存／備份不混入研究正文。

ClearML優先實際開始時online追蹤，先測服務可達性；不可用時記錄actual-start offline、closed、未同步，不能事後聲稱online成功。保存PID、owned-process-group退出、source/input/output SHA、optimizer chain、失敗attempt；有真實完成回執才算完成。

目前已有數學參考、Torch雙端梯度／CE與pair訓練元件、完整分數向量上的有界witness選擇器和有界準備控制器。NG69 [全組審核](ng0069-final-breadth-review.zh.md)已通過；[step0全庫預演](ng0071-global-preflight-contract.zh.md)已核對完整384題TRAIN witness、score-gradient及192固定batches，lambda2 CUDA工程fixture也已完成。96-update訓練chunk、A96全庫編碼／引用元件、[四arm調度器](../../../../scripts/ng71_pilot.py)、[全庫觀察](../../../../scripts/ng71_observation.py)及[終局reviewer](../../../../scripts/review_ng71_pilot.py)已完成本地測試。**實際執行需另行核對phase回執；checked-in準備配置保持training disabled，新run才在完整封存後啟用。Native cost trial仍未執行。** 不把零更新診斷或disposable工程更新當成正式訓練。

最新143項NG66–71／reviewer測試及134項文檔測試通過。真實49.65M基座的一個TRAIN fixture已分別在CPU和lambda2 CUDA上做前向、107組梯度VJP、一次disposable有限更新、checkpoint和optimizer重載。各裝置fixture驗證通過，不代表跨裝置逐位相同，也不替代完整四組科學實驗。沒有讀取locked test；實際推論、工程更新及封存證據詳列在[準備報告](ng0071-preparation-review.zh.md)。

## 9. 文獻依據與適用邊界

- [LambdaLoss, CIKM 2018](https://research.google/pubs/the-lambdaloss-framework-for-ranking-metric-optimization/)：metric-aware pair weighting的依據；本文balanced/soft/witness組合是自訂surrogate，沒有繼承全庫收斂保證。
- [SmoothI](https://arxiv.org/abs/2105.00942)：直接平滑rank indicators可優化IR指標；不解決候選缺失及false negatives。先保留較簡單pair reference，無需一開始導入完整differentiable sorting。
- [SPLADE-v3](https://arxiv.org/html/2403.06789v1)：成熟warm-start、蒸餾與negative設計的有效組合；其資料量、teacher與in-domain收益不能移植成我們的預期增益。
- [ANCE](https://arxiv.org/abs/2007.00808)：隨模型更新的corpus-level hard negatives；本輪借用refresh原則，不引入ANN作產品召回，也不每步重建。
- [NV-Retriever](https://arxiv.org/html/2407.15831v1)：positive-aware false-negative處理；其教師閾值不能直接視為我們標籤正確率。
- [DF-FLOPS](https://arxiv.org/abs/2505.15070)：頻繁term對production sparse成本的重要性；論文引擎速度不能當作II42的新結果。
- [Anisotropic Vector Quantization](https://proceedings.mlr.press/v119/guo20h.html)：保存inner-product決策而非等向重建的動機；量化方法不直接提供稀疏可索引的encoder。
- [Single-Stage Sparse Coding](https://arxiv.org/html/2605.30120v2)：sparse coding與retrieval訓練結合的参考；multi-vector設定不同，重建與latent dot等價需要額外幾何條件。
- [Latent Terms](https://arxiv.org/html/2605.29384v1)、[Interpret and Control Dense Retrieval](https://arxiv.org/html/2411.00786v2)：預訓練dense的latent特徵可以轉為檢索表示；M1911等本地實驗提醒不能跳過DF／原生品質驗證。
- [MetaRAG](https://arxiv.org/abs/2608.24214)、[FastContext](https://arxiv.org/abs/2606.14066)、[Chroma Context-1](https://www.trychroma.com/research/context-1)：保留供後續SEARCH/READ/FINISH agent研究。工具策略與evidence packing不替代本輪單次retrieval排序／成本驗收。
