# NG-0078：前排失位與監督覆蓋診斷協議

2026-09-13 UTC。**在新 reduction 執行之前凍結；只有唯讀診斷，不是新訓練或品質晉級。**

## 問題與依據

[NG77](ng0077-final-retention-review.zh.md)的完整 BM25＋semantic 在參訓 Pilot 上改善，Sentinel 沒有可靠轉移。加上可信 margin 保留後，原有 anchors 的非正 margin 從217淨減至201，但28個 top10 明顯失位未減。需要核對：具體修復和新增的關係是什麼？與真正受損的前排正例是否相遇？

這不是重新執行[NG72](ng0072-retention-diagnosis-review.zh.md)的 query/document 交叉或 support/weight 分解，也不是[NG75](ng0075-parameter-update-diagnosis-review.zh.md)的局部 parameter attribution。兩者已顯示存續權重與跨 query 轉移重要，少數初始 rivals 不足以代表終端品質。[NG69](ng0069-final-breadth-review.zh.md)的 matched-exposure 廣度改善已成立，但 CE 增密且 NQ 未修好，不能不加控制地照搬。

## 固定資料與計分

- 重用 NG77 initial／Z96／K96 的768個已曝光 TRAIN query，Pilot／Sentinel 各384、三域各128；完整233,009文件背景，全部1,371正例，原始／added來源均保留。
- 每題唯讀集合為三個完整 top100、全部 gold、原 D 訓練 pool 的聯集，不截短，不另抽容易的題目。重用 cached CSR，保持完整固定 BM25＋semantic 計分，FP64兩路核對，絕對誤差上限1e-12，ID解同分。
- 聯集能核對這三個終端的完整 top100；不能表示其他 checkpoint 的完整排序，也不枚舉所有低於100名的 rivals。正例的 rank 直接使用已核實的全庫 rank，絕不用聯集中的 rank 替代。
- Pilot 優先使用 step0 已封存 teacher 分數，另保存原 teacher 分數供追蹤；Sentinel 只有原 teacher，其未參訓狀態不可當成新發現。未觀察／teacher相反／同分均不假造為負例。保存正例來源與既有 token visibility；沒有 supporting-span judgments 就明確維持 unknown。
- 依賴 NG77完整 remote inventory、Mac獨立核對回執與原 NG72 lineage manifest 的既定 SHA，不能對已變動資料重新算 SHA 就冒充舊來源。不得新 encoder/teacher inference、DEV／LOCKED_TEST評分、更新模型或更動 frozen run。

## 可核算的定義

令 `s_m(q,d)` 是完整 hybrid 分數；`ahead_m(p,n)` 使用分數遞減、document ID遞增的全序。對模型 `m` 及 cutoff `c ∈ {10,100}`，定義初始正確而終端逆轉的前排 rival 集：

```text
L_m(q,p,c) = {n in top_c(m), n not in gold(q):
              ahead_initial(p,n) and not ahead_m(p,n)}

persistent = L_Z intersect L_K
repaired   = L_Z minus L_K
new        = L_K minus L_Z
```

「repaired」只表示不再符合這個終端 boundary-loss 定義，也可能是 rival 離開 cutoff，並不必然表示 margin 恢復正值；另列全部原可信 anchors 的嚴格 `margin <= 0` 集合交差，分清同分與 rank 規則。此交差必須還原NG77逐域217／201舊計數；不得把淨差16叫作恰好修好16個。

對每個 gold `p`，保存真實三個 rank，以及：

```text
dcg_part_m(p) = 1[rank_m(p) <= 10] / log2(rank_m(p) + 1) / IDCG10(q)
recall_part_m(p) = 1[rank_m(p) <= 100] / number_of_all_gold(q)
```

先求同題全部正例的和，再以每域128題計算平均，必須逐域重現NG77的三個差值。只對相對initial的 gross positive DCG loss 分割互斥覆蓋類別：not-trained、沒有nongold crossing、至少一個outside-pool、全在pool且全trusted、全在pool但有untrusted；再分正例來源。K−Z只報品質差，不套用initial-relative分類。正例之間換位可能產生一正一負的 contribution，gross loss 不等於 net query loss。

各 cutoff 的 repaired／new／persistent pair 另依 pool／anchor 與 teacher 狀態計數，並列在受損正例上的子集。受損參照為initial；repaired使用Z的損害，其他使用K。**pair計數不是DCG的可加因果責任，覆蓋分割也不是證明該類別造成損害。** 兩路cached score驗證不是重新做全庫排序；本輪沒有bootstrap新試驗或parameter VJP。

## 執行與出口

實作為[`ng78_boundaries.py`](../../../../scripts/ng78_boundaries.py)，fixtures涵蓋同分、全部正例分母、完整聯集、gold互換、unknown teacher、修復與新增並存、CPU-only生命週期及資料邊界。

另以不import主reducer的[`review_ng78_boundaries.py`](../../../../scripts/review_ng78_boundaries.py)逐scalar重算全部positive分類、pair交差及DCG／Recall contribution，核對實際輸出。它重用相同frozen raw witnesses，不是假稱第二套模型、teacher或全庫ranking驗證。

使用獨立CPU phase，4threads、process-tree RSS不超16GiB、host available維持24GiB以上、磁碟free超40GiB、硬上限1,800秒。前16題耗時外推全768題乘1.5，須小於1,200秒，留600秒給hash、tracking與收尾；失敗保留，不放寬限制重跑。Mac低於free-memory floor時只做輕量準備，實際CPU診斷優先使用有餘裕的lambda2，不佔GPU。ClearML以actual-start offline任務記錄，須自然close、exit0、owned process group消失、完整SHA鏡像核對才能宣告完成。

本輪不自動啟動training。若損害主要伴隨原pool外的新競爭者，再設計一次current-model witness refresh最小對照；若pool覆蓋良好、可信关系多保住而跨query轉移仍壞，优先設計matched-exposure監督廣度／跨query功能保留對照；若既有teacher／來源疑義突出，使用NG68盲化人工審查，不以LLM充當人類。這些是診斷後的分支，不是用任意比例門檻推斷唯一原因。

新訓練需另凍結資料、比較、曝光量、逐域nDCG／all-positive Recall與成本gates；不掃lambda、不擴大失敗K、不丟NQ、不降floor、不碰production。semantic NNZ／query-DF仍只是代理，最終目標依然是完整hybrid超越dense，且actual total native cost更低。
