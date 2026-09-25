# NG-0080：成熟底座、監督與稀疏交互的聯合診斷

日期：2026-09-13。使用者已授權推進及在資源允許時並行小實驗；本協議在新結果產生前固定。目標仍為完整 BM25 + semantic hybrid 的品質及實際總成本，不是只模仿 dense、取得較小 NNZ 或通過 TRAIN。禁止生產變更、LOCKED_TEST 存取、刪除舊產物和失敗後原地重試。

## 本輪要回答什麼

| 問題 | 對照／證據 | 不能宣稱的結論 |
| --- | --- | --- |
| 成熟 trunk 是否含可讀出的 teacher 能力 | 凍結 trunk 的 mean/CLS hidden，TRAIN-only ridge 讀出；以 query/document 共同投影保留 teacher 座標 | 線性讀出失敗證明不存在資訊，或成功等於 sparse 可用 |
| 繼續適配 trunk 是否有益 | 同初始化、資料、候選 panels 的 frozen/full dense 讀出訓練 | dense head 的收益一定能移交 sparse |
| sparse scoring 是否丟失判別能力 | 原輸出與新 teacher-field 目標；固定 RMS 與實際 query/document 計分 | decoder 重建改善等於倒排分數改善 |
| 監督資訊是否不足 | qrel-protective positive/nonpositive 與全 panel teacher 關係、近鄰／背景覆蓋 | 缺少顯式 pair 就一定缺少獨立資訊；teacher 即真實相關性 |
| 輸入是否看不到必要資訊 | 兩種 tokenizer 的 token 長度、query 截斷，source-positive 文件可見比例；後續固定子集長上下文對照 | 沒有 evidence span 就把截斷當成已證實的錯誤原因 |
| 高 DF 是否可無害消除 | DF、query DF-work、分數差貢獻與聯合 mask 對照；後續 native 測量 | 高 DF 即無用，或 proxy 改善等於低延遲 |

## 資料與觀察邊界

第一波使用完整已封存 NG67 TRAIN：13,824 個問題，FEVER/HotpotQA/NQ 各 4,608；233,009 文件背景。每域按 `SHA256(NG80/split-v1/ + query_id)` 選 512 題為 gradient-excluded diagnostic，其餘 4,096 題訓練，共 12,288/1,536。正規化相同 query 不得跨新 split；若出現衝突，停止而不靜默改數量。這些題目有歷史曝光，名稱為 TRAIN_DIAGNOSTIC，**不是新的獨立泛化證據**。只讀 NG67 TRAIN 和 documents；不讀 DEV/LOCKED_TEST query、qrel 或 pool。

Corpus 為繼承的 transductive 背景，可能包含舊評估文件；只蒸餾無標籤 corpus 不等於使用 test query，但不得宣稱新文件泛化。完整來源 provenance 沿用 NG67，原始／LLM 新增 label 的可靠度不假裝重新人工驗證。

本輪先建立可擴展的流程，不要求 12k 診斷先超過 dense 才能研究更大資料。同樣不能把 12k 三域結果當成十萬級多意圖監督的答案。第二波必須新增合法多來源 TRAIN、source/topic/task-family 隔離評估；先做來源、授權、去重與曝光盤點，不從 locked benchmark 生成題目。84 題人工複核仍未取得真人新標註。

## 第一波固定圖

1. `prepare`：核對模型、corpus、TRAIN、pool/labels SHA；固定 split、source pool 加共用 hash 背景 16 份的 panel，所有原正例保留；產生不含原文的來源／規模審計。
2. `encode-teacher`：GPU 0，PPLX FP32、query64/document256 的既有 mean-pool + L2 contract，重用完整 document.npy，重新產生全部 TRAIN query vectors；對既有 query 做容差核對。這不是新 teacher 訓練。
3. `encode-hidden`：GPU 3，成熟 NG3 checkpoint096，eval-mode FP32，輸出全部 TRAIN query 與 corpus document 的 masked mean 和 CLS hidden；不修改參數、不推論 locked query。
4. `diagnose`：前兩個 GPU 階段閉合後，CPU4。TRAIN-only fit 共同 query/document ridge projection，query/document role 各占一半的均方目標；document 訓練集合僅取 TRAIN source pools。固定 ridge=1e-3（對平均二階矩），intercept 不懲罰。mean/CLS 是兩個預先聲明的診斷，不依結果自動挑 winner。在固定 source+background panels 分別報 TRAIN 與 TRAIN_DIAGNOSTIC 的 teacher margin error、teacher top10 overlap、all-positive nDCG，以及輸入可見性。Panel 指標不是全庫 recall。

第一波只回答讀出與資訊預檢，不能冒稱完成 full-trunk/sparse 梯度對照。後續訓練必須有單獨的 frozen config、實際 loss/梯度測試、chunk ETA、source manifest。首個訓練矩陣為 frozen-dense/full-dense/full-sparse field，加現有 sparse 監督對照；固定資料／panel，保留初始模型。是否增加大規模底座適配由同總預算的 direct vs staged 對照決定。所有分支結果齊備後再反思，不中途換 loss/seed/資料去追分。

## 數學與判讀

Ridge 只是有界的讀出下界。使用共同映射 `h -> [h,1] W` 近似 1024 維 teacher 向量；推論後 L2 normalize，維持共享 query/document 座標。frozen trunk 的 mean/CLS 不是原 sparse pooling 的替代產品；兩者均失敗時不能直接推論模型容量不足。

Teacher score-field 比較只移除單題共同 offset，保留相對分數。不做 document-column centering，不把 BM25 score 從 teacher 中相減。Teacher fidelity 與 relevance 各自列出，teacher-unjudged conflict 不改成硬負例。取得完整 ranking head 後才可討論外部 rival、Recall@100 與 DF utility；不重演局部候選池的 oracle 誤判。

解凍／稀疏訓練保留實際 RMS scoring 與 query/document freshness。成本從第一次更新記錄，先不增加新的強 sparsity pressure；其後提高成本約束仍混入能力保留與相關性監督。若品質只能靠增密，封存為 quality parent，不部署；後續 Pareto 對照不能改寫舊門檻。階段式課程不是全局最優保證。

## 執行與停止條件

只用一張空閒 GPU/worker。先檢查 GPU 使用及 `.ng-gpu-N.lock`；本次 lambda2 0/3 可用，1/2 屬其他工作。Spark 可用記憶體不足，本輪不使用。CPU 每個 worker4 threads；RSS16GiB、host available>24GiB、GPU total<=20GiB、artifact free>40GiB；每階段 <=5,400 秒。前64項 canary 外推加1.5倍裕量與300秒必須落在階段預算內；超限保留 attempt，重新規劃較小 chunk，不放寬上限。

重用既有 process-group supervisor 與 GPU cooperative locks，為 parallel child phases 分別保留 controller/worker PID、退出、資源峰值及 hash。採實際開始的 ClearML tracking；既有環境尚未驗證 online authentication，明確 offline、closed，不能宣稱同步成功。成功必須有 exit0、error=null、owned_group_closed、closed tracking 和輸出 SHA；資料準備不是訓練，phase 閉合不是科學成功。

一次性 terminal review 分列每個問題：支持／反對／尚不能識別／需新資料。不得把尚未執行的項目寫成已排除。原生成本、標註可靠性與多來源泛化在取得相應證據前始終 pending。

## 依據

- [先前綜合備忘錄](../designs/ng-teacher-geometry-supervision-reset.zh.md)：NG66–79、teacher/label 與資料覆蓋的區別。
- [M1904/M1905、M1911](../m1900-m1999/ii42-m1900-m1920-learned-sparse-success-failure-factor-report.md)：成熟 basis 與高 DF 反證。
- [Granite §6](https://arxiv.org/html/2502.20204)：retrieval pretraining、dense-to-sparse distillation；不是本輪新預訓練的充分必要證明。
- [E5](https://arxiv.org/html/2401.00368v3)、[GPL](https://aclanthology.org/2022.naacl-main.168/)：任務多樣性與生成／判斷分工。
- [DF-FLOPS](https://arxiv.org/abs/2505.15070)：頻率感知成本，不是無條件刪除高 DF。
