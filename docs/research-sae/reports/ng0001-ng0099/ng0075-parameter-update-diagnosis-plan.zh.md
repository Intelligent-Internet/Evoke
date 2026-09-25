# NG-0075：從 Score 梯度到實際參數更新的診斷設計

2026-09-13 UTC。**診斷實作／預演階段，執行 manifest 和實際狀態另見 [本輪 status](ng0075-parameter-update-diagnosis-status.zh.md)；未因本方案啟動新科學模型訓練。**前置 [NG74](ng0074-uniform-control-review.zh.md) 的完整 mirror／獨立審核已通過。不是擴大未通過品質門檻的 pilot。

## 為什麼不是再換權重

NG69 題目廣度改善泛化但令 posting 膨脹；NG71 witness 和 ranking objective 的組合降低成本代理並改善 macro，但 NQ 門檻失敗。NG72 看到 surviving support 上的權重漂移與雙端影響；NG73 看到多數 TRAIN lost pairs 的 own score-gradient 要求正確方向。NG74 移除 rank weights 仍未修復 NQ，且參訓題的更大收益没有轉移。

這些發現還沒有把共享參數、AdamW 位移、跨題干擾、非線性和監督缺口分開。現在最有價值的問題是：**實際 optimizer 更新之後，哪些既有 margin 被改動？score-space 的正確方向在哪一步沒有保留下來？**

## 量測而非假定

固定完整 hybrid pair margin：

$$
m_i(\theta)=s_\theta(q_i,p_i)-s_\theta(q_i,n_i),\qquad
\Delta\theta=\theta_{t+1}-\theta_t.
$$

比較實際 finite change 與局部一階預測：

$$
\Delta m_i=m_i(\theta_t+\Delta\theta)-m_i(\theta_t),\qquad
\widehat{\Delta m_i}=\nabla_\theta m_i(\theta_t)^T\Delta\theta.
$$

不把 `-learning_rate * gradient` 當成實際 AdamW 位移：兩個 parameter group 的 LR、clip、歷史 moments、adaptive scaling 和 decoupled decay 都參與實際更新。量測 float32 參數更新前後的差，再以 float64 reduction 計算內積。記錄 Taylor residual、no-update 數值底噪和符號一致性；ReLU/max-pooling 的非光滑處尤其不能聲稱一階式精確。

query/document 共用 trunk 和 MLM head，因此同一 margin 的 Jacobian 可分為 query 路徑與 document 路徑，**兩者都作用到同一組參數**。分別停止另一條輸出路徑的梯度，只用於 Jacobian 分解，並核對兩項之和等於完整 derivative；不是部署混版 encoder，更不是 `document.detach()` 就凍結了 document encoder。再按不重複的 named parameters 分成 trunk/head，tied weight 只計一次。

`own pair derivative`、同題總 score-gradient、完整 minibatch parameter update 對 probe margin 的作用是三種不同量。本輪沿用 NG73 的前兩種記錄，再加入第三種；沒有實際反事實干預時，不把負內積的比例稱為192步最終回退的因果比例。

## 有界執行提案

1. 重用 D/U 的初始化及96-update checkpoint/optimizer，僅 replay 每段第一個實際 batch：step1與97，共四個 disposable one-step cases。從已封存 order/witness/progress 取得四題，不改 pool、target、LR、seed、clip 或曝光。不接續成新模型、不覆寫任何來源，也不拿 probe 作 optimizer loss。
2. replay 先對已記錄的 score/loss/score-gradient、update L2、初始模型和 optimizer 指紋做核對。四題 VJP 和實際完整梯度都必須正確；若實際 CUDA replay 不符合凍結容忍度，先停止查復現，不把不一致結果拿來分析。既有紀錄沒有每個 step 的完整 parameter SHA，不能宣稱從其 update norm 唯一重建了整條訓練軌跡。
3. 固定24題未參訓 TRAIN sentinel，每域8題，以 `SHA256('NG75-probes-v1' + NUL + query_id)` 排序選取，與 query ID 作 tie-break；不按 D/U 輸贏選題。它們已在早期報告曝光，不稱盲測。保留每題所有 gold/provenance；每題固定初始 top100中的最高分 non-gold 及最接近rank100的 non-gold 作最多兩個 rival，去重。`non-gold` 不等於已判無關，教師未知／衝突單獨標記。先凍結 query/pair 身份和文本 SHA，再做新的參數 probe。
4. 每個 case 同時觀察當前四題及相同 sentinel probes，報告改善／不變／惡化全部分母，原始與 added-positive 分開。只量有限 pair margins，不重編全库、不新算 DEV，不把 pair 結果當作 Recall/nDCG 評估。24題不是泛化樣本量，兩個局部 step 不是完整學習曲線。
5. 單張空閒 GPU，預計沿用 lambda2 via spark-1；啟動前再核對 occupancy。CPU4、RSS16GiB、GPU20GiB、host available24GiB、disk free40GiB，單 case1800秒。先一個 probe、再八個 probes 做 wall/RSS canary，包含載入、replay、Jacobian、核對和tracking關閉；若全 case 預測不能留出安全餘裕則停下，不縮小分母或放寬界線以過關。每個 case 最多128個 sentinel gold/rival pairs；準備若超出，先根據身份／數量修訂資源協議並重新凍結，不根據輸贏刪 pair。

執行前仍需 small-model chain-rule／實際 optimizer displacement／no-update／tied-parameter／frozen-input／TRAIN-only fixtures。正式 source/config/probe manifest 和依賴 SHA 必須先入 Git/外部 catalog；每個 phase保留 exit、資源、owned group closure、ClearML actual-start/closed 和獨立 reviewer。預演通過不等於NG75科學診斷完成。

實作入口為 `scripts/ng75_update_diagnosis.py`，數學量測為 `scripts/ng75_update_math.py`。預先固定 replay 的 score／loss／gradient／update norm 容忍度為 `rtol=2e-5, atol=2e-7`，重新計算的位移 L2 對本次 optimizer 記錄則為 `rtol=0, atol=1e-12`。雙路 Jacobian 逐參數之和與完整 derivative、雙路投影之和與直接投影，同樣採前述 `2e-5/2e-7` 容忍度；記錄最大絕對差和 relative L2，不以 tolerance 隱藏實際 residual。

同一參數狀態的重複 forward 必須 bit-exact；margin 方向的數值底線預先固定為 `max(1e-7, 10 * no_update_error)`，不依結果調整。canary 的全 phase 估時為「自 worker 啟動至完成載入/replay 的時間 + 最慢已測 pair 時間 × 完整 pair 數 × 1.5 + 360 秒關閉／hash reserve」，必須小於1740秒，外層1800秒仍獨立執行。輸出同時保留 pair-weighted 和 query-balanced 的 mean change，防止多 gold 題的數量淹沒其他題；這仍不是 nDCG／Recall 指標。

step1的D/U有共同初始化；step97的兩組各自累積了不同參數和optimizer history。後者是各自真實軌跡上的局部診斷，不是相同state下只替換weight的因果對照；不得把它的組間差值全歸因於當步weight。

設計時只讀既有 TRAIN sentinel 的 initial ranking 做身份／數量核對：上述 hash 規則選出 FEVER／HotpotQA／NQ 各8題，gold 數12／16／14，pair 數24／32／28，共84 pairs，未超過128上限。沒有根據訓練後輸贏選樣，也沒有新的 encoder/Jacobian 計算；完整 query/pair manifest 尚待正式凍結。

## 結果如何決定下一步

- 若 own score 梯度支持正確方向，但實際位移對同題或 sentinel 持續造成可復現的負 margin change，而且一階預測能解釋，才有依據設計 **D versus D + trusted baseline-margin retention** 的單變量對照。保留較好基座已支持、來源可信的相對排序，不強迫完整 embedding 重建，也不禁止所有新排序。
- 若一階預測失效、非線性 residual 主導，先改善對 update 大小與讀出非光滑性的量測；不能直接套 gradient projection 就聲稱安全。
- 若兩個局部 case 不出現足夠證據，明確記為診斷覆蓋不足；擴大預先固定的 trajectory checkpoints/步數或回到 witness coverage/監督廣度，不由陰性结果證明「沒有干擾」。

Retention 係數、anchor 選擇和任何新訓練均未在本設計中獲得效果認定。後續 pilot 還需固定的三域品質 floors、完整 hybrid/all-positive ranks、未曝光 holdout，以及 native total cost；不因本輪因果診斷而放寬 NG71/74 的失敗門檻。

## 文獻的作用與限制

[PCGrad](https://arxiv.org/html/2001.06782v4) 的§2.2–2.4把 gradient conflict、幅度失衡、曲率一起分析，並非「有負cosine就一定有害」。它支持量測共享更新，但其多任務結果和特定假設下的理論不證明本檢索模型能改善；本輪不直接實作PCGrad。

[GEM](https://proceedings.neurips.cc/paper_files/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf) 提供保留既有任務表現同時允許正向transfer的研究方向。把這個方向轉成 retrieval margin anchors 是待測設計，不是照搬其記憶庫loss就能保證全庫前排排序。

[AdamW](https://arxiv.org/html/1711.05101v3) 區分 adaptive optimizer中的 L2 regularization 和 decoupled weight decay。本輪因此使用真實參數位移做 margin預測，而不是把 raw-gradient內積當成實際更新效果。上述一階式和雙路 Jacobian 分解是鏈式法則的診斷應用，不是新演算法或單調品質保證。
