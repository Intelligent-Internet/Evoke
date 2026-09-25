# Evoke：PostgreSQL 中的統一稀疏檢索與收斂式索引

## 系統技術報告 (Beta 1)

**版本日期：** 2026-09-03。**實作基線：** `9a658b63`、extension `0.2.5`、page-native v3、隨套件發布的 P2.2 模型。

[English version](technical-report-evoke-system.md)

### 摘要

Evoke 是 PostgreSQL 檢索引擎，在 `psql_bm25s` 的詞法檢索基礎上加入模型產生的稀疏語意證據。它不是把 BM25、向量資料庫與融合服務放在各自獨立的更新管線後面，而是將詞法與語意 atom 表示在同一個由資料庫 relation 擁有的倒排索引中。PostgreSQL 繼續負責交易、資料列可見性、故障恢復與索引生命週期；模型將文字編譯為稀疏貢獻值，索引則讓這些貢獻值可被查詢及持續維護。

系統的核心問題是：當文件、語料統計與衍生查詢結構不斷變化時，如何維持高效讀取。Evoke 使用不可變 posting 物件、寫入時複製（copy-on-write，COW）中繼資料、經驗證的根節點發布、有界的變更前沿、可重用的 term fold，以及獨立更新的語意加速器。背景 worker 處理新證據時，相容且已發布的加速器仍可繼續服務。這讓前台服務的連續性與背景收斂分離，同時明確區分近似排名的新鮮度與當前資料列的可見性。

本報告將架構、算分模型、儲存協定、執行路徑與實驗證據整理為完整系統敘事。凍結的 P2.1 品質評估在 BEIR15 與 MTEB10 上分別達到 0.666885 與 0.703125 的 macro Recall@100，接近對應 dense 參考值 0.670880 與 0.707213。歷史詞法回歸研究將 5,183 篇文件的 SciFact 平均查詢時間恢復至 0.448 ms，原始 `psql_bm25s` 記錄為 0.454 ms。這些是版本與口徑分開的實驗，不是對目前 P2.2 套件所有功能重新執行的一次綜合基準測試。

**關鍵詞：** PostgreSQL、BM25、稀疏語意檢索、倒排索引、寫入時複製、MVCC、背景收斂、查詢加速。

## 1. 範圍、沿革與主要貢獻

原有的[詞法技術報告](technical-report-psql_bm25s.md)記錄 BM25 基礎與早期可變索引工程；獨立的[模型技術報告](technical-report-evoke-model-zh.md)說明模型編譯、校準與詳細品質結果。本報告將兩者連接到目前的 Evoke 系統，不覆蓋其中任何一份文件。

BM25 基礎承接 eager sparse scoring 的思路：預先準備 term 貢獻值，再使用稀疏累加，減少重複的查詢時計算。[BM25S](https://arxiv.org/abs/2407.03618) 將此方法用於 Python 稀疏矩陣。交易型 PostgreSQL 引擎還必須處理語料統計變化、tuple 版本消失、讀寫競爭與崩潰恢復。因此，Evoke 將預計算視為可重用的證據表示，而不是永遠不再變動的整體語料矩陣。

目前實作的主要貢獻包括：

1. **單一檢索權威來源。** 詞法與語意證據共用 posting 命名空間、實體索引與發布生命週期；語意檢索不需要獨立維護另一個 ANN 索引。
2. **增量式不可變儲存。** COW 目錄只替換受影響的路徑，共享未變動物件；term fold 減少重複遍歷，同時保留尚未涵蓋的證據。
3. **非同步模型整合。** 共用 runtime worker 擁有模型 session；一般語意索引寫入只發布詞法證據與待處理身分，不在前台執行文件推理。
4. **服務與收斂解耦。** 小幅變更與純中繼資料根節點更新不會直接廢除相容的 baseline 加速器，背景則按維護欠帳調度並發布替代版本。
5. **明確的證據邊界。** 對 posting 的精確計算、近似候選執行、舊 baseline 服務，以及對人工相關性標註的品質，分開評估。

本報告介紹 Beta 1，`0.2.5` 與 P2.2 保留為工程識別。本報告綜合引用的原始碼與歷史實驗，不表示任何特定伺服器目前已部署此 revision。操作契約以 [Architecture](architecture-and-design.md)、[Query Semantics](query-semantics.md) 與 [Maintenance Lifecycle](maintenance-lifecycle.md) 為準。

## 2. 系統架構與產品介面

Evoke 提供單一 PostgreSQL access method：`USING evoke`。預設 `sae = false` 模式提供精確 BM25；設為 `sae = true` 時，合格模型將語意 atom 加入同一索引。本文以 Sparse Semantic Retrieval（SSR）指稱產品層的稀疏語意檢索路徑；`sae` reloption 保留為目前 SQL/catalog 名稱，SAE 則保留給 encoder／vocabulary 機制與歷史實驗名稱。

```text
                         PostgreSQL application
                                   |
                     SQL / evoke_query / predicates
                                   |
                    +--------------+--------------+
                    |                             |
             exact posting path          derived candidate path
                    |                             |
                    +----------+------------------+
                               |
                  tuple identity + MVCC recheck
                               |
                         ranked table rows

  +------------------- one Evoke index relation -------------------+
  | checked root -> COW manifest / term / document / lexicon trees |
  | lexical + semantic postings | linked L0 | folds | accelerators |
  +--------------------------------------------------------------+
              ^                                ^
              |                                |
      foreground lexical DML           background maintenance
      + pending semantic work          + shared model runtime
```

`evoke_query` 的 explicit-hit overload 同時服務兩種模式。SSR 另支援 planner-native scalar marker：planner 將符合條件的排名查詢轉為 custom scan，而不是逐列呼叫模型。BM25 也保留原生 operator 與有序 index scan 介面。`ctid` 和索引內部 `doc_id` 是執行身分，不是可長期保存的應用主鍵。

對於資料表 `docs(id, title, body)`，安裝 extension 並設定共用 runtime 與合格模型 checkout 後：

```sql
CREATE INDEX docs_retrieval_idx
    ON docs USING evoke (title, body)
    WITH (sae = true, field_aware = true);

SELECT d.id, hit.score
FROM evoke_query(
    'docs_retrieval_idx'::regclass,
    'transaction-safe semantic retrieval',
    ARRAY['title', 'body']::text[],
    ARRAY[2.0, 1.0]::real[],
    20::int4
) AS hit
JOIN docs AS d ON d.ctid = hit.ctid
ORDER BY hit.score DESC, d.id;
```

這是一個保留欄位身分的索引，不是先取得兩份獨立 top-k 再做 late fusion。欄位命名空間保留其身分，並允許查詢時指定權重。目前 field-aware 契約並非 BM25F 式的獨立逐欄長度正規化。不啟用 `field_aware` 時，多欄輸入會合為一個邏輯文件。詳見 [Field-Aware Indexes](field-aware-indexes.md) 與 [Getting Started](getting-started.md)。

## 3. 檢索的數學模型

### 3.1 詞法證據與統計

令 $N$ 為文件數， $df_t$ 為 term 文件頻率， $tf_{t,d}$ 為詞頻， $|d|$ 為文件長度。以下採用 Lucene-style 變體：

```math
\mathrm{idf}_t = \log\left(1+\frac{N-df_t+0.5}{df_t+0.5}\right),
\qquad
L_t(d)=\mathrm{idf}_t\,
\frac{tf_{t,d}}
{tf_{t,d}+k_1\left(1-b+b\frac{|d|}{\overline{|d|}}\right)}.
```

省略的全域 $(k_1+1)$ 乘數不會改變固定參數下的詞法排名，但在校準詞法與語意的相對尺度時仍有意義，因此模型／索引契約必須固定此慣例。歷史模型評估採用 $k_1=1.5$ 與 $b=0.75$；其他支援的 BM25 變體見原始報告。

Evoke 區分詞頻等**中性證據**與**依統計特化的 impact**。中性 fold 可以跨越語料統計變化繼續使用；特化的 impact fold 則必須匹配其 statistics epoch。如此保留 eager scoring 的收益，又不必在每次 $N$、 $df_t$ 或平均長度改變時重寫全部 posting。

### 3.2 語意 Atom 與統一算分

對 token 位置 $i$ 與語意座標 $j$ 的 encoder logit $z_{ij}$，稀疏基礎模型產生非負的序列 impact：

```math
w_j(x)=\max_{i\in x}\log\left(1+\mathrm{ReLU}(z_{ij})\right).
```

模型編譯器再套用合格的 support、校準與文字分窗策略，產生可發布的 query 與 document impact。詞法座標 $\mathcal{V}_L$ 與語意座標 $\mathcal{V}_S$ 互不重疊。概念上的單欄位分數為：

```math
S(q,d)=
\sum_{t\in\mathcal{V}_L}q^L_t L_t(d)
+c(q)\sum_{j\in\mathcal{V}_S}\hat q_j\hat d_j.
```

查詢尺度 $c(q)$ 屬於模型校準契約，不是應用端的 reciprocal-rank fusion 係數。兩個加總共同寫入同一文件的累加器，因此文件可以因中等強度的詞法與語意證據相加而進榜，不必先進入任一路獨立 top-k。

對選定欄位 $f$ 及其權重 $a_f$，field-aware 合成為：

```math
S_{fields}(q,d)=\sum_f a_f S_f(q,d).
```

此式表達欄位範圍內的加法算分，不表示引擎實作了所有 BM25F 正規化方案。

### 3.3 精度、保留比例與精確性的分層

三項選擇作用於不同層次：

| 選擇 | 層次 | 意義 |
| --- | --- | --- |
| Runtime precision，預設 `fp16` | ONNX 推理 | 文字編譯的執行精度與 provider |
| Semantic impact storage，預設 `f32` | 權威 posting | 編譯後語意數值的儲存表示；`u8` 是明確選擇的量化設定 |
| `semantic_alpha_mass`，預設 `1.0` | 逐欄位發布 | 正常模型 budget 之後保留的正語意 impact 質量比例 |

將正常 budget 後的 impact 依質量由大到小排序，保留策略可表示為選取滿足下式的最短前綴 $J_\alpha$：

```math
\sum_{j\in J_\alpha}\hat d_j
\geq \alpha\sum_j\hat d_j,
\qquad 0<\alpha\leq 1.
```

對非負 query impact，在另外的量化誤差之外，被捨棄的質量可用來解釋尚未乘上尺度的語意內積損失上界：

```math
0\leq\Delta S_{sem}(q,d)
\leq\|\hat q\|_\infty
\sum_{j\notin J_\alpha}\hat d_j.
```

這是保留策略的數學解釋，不是執行器已實作的 top-k 剪枝證明。總質量損失小，不保證相近分數的排序不變。`f32`／alpha `1.0` 表示這兩層沒有額外的儲存／保留近似，不會因此讓候選加速器變成窮舉，也不會讓待處理語意工作即時可見。精確執行指的是對適用的已發布表示與可見性契約精確計算，不代表等同未剪枝神經模型或人工相關性判斷。

## 4. 模型編譯與推理流程

### 4.1 有版本契約的編譯器

隨套件發布的模型由[模型鎖定檔](../packaging/milestone-model.json)綁定：

| 身分 | 數值 |
| --- | --- |
| Bundle | `evoke-p2.2-nfcorpus-v2` |
| Model ID | `evoke_p2_p22_nfcorpus_v2_smoke` |
| Runtime ABI | `evoke_p2_unified_text_atoms_v2` |
| Manifest SHA-256 | `760f57eb9230b3d20f04315f2a4cd29a4c9967123dbc29f85656c27f91cbbf86` |
| ONNX Runtime | `1.29.0` |

完全相同的凍結 checkout 已發布為 [Evoke Model (Beta 1)](https://huggingface.co/Intelligent-Internet/Evoke-Model-Beta-1)。[下載指南](examples/semantic-model-checkout.md#download-the-default-model)固定 revision 與 archive checksum；此次分發不改變模型或歷史評估身分。

上游 checkpoint 為 `ibm-granite/granite-embedding-30m-sparse`，revision 為 `ad82b1fd09541c998c8d45045d601c51fdb8a9b7`。約 30.3M 參數提供了緊湊的稀疏檢索基礎；上游模型家族與訓練方法見 [Granite Embedding Models](https://arxiv.org/abs/2502.20204)。Evoke 的本地工作集中在編譯、校準、發布與系統整合，不應與從零訓練基礎模型混為一談。

P2.2 採用確定性的 ABI-v2 分窗，編譯完整 query 與 document 文字。套件中的詞法詞彙表與校準以 NFCorpus 凍結。這使工程文字路徑超出歷史 P2.1 單序列評估的範圍，但本身不等於已取得新的長文件或跨領域基準結果。

```text
                        qualified model checkout
                      tokenizer + ONNX + calibration
                                | signature
             +------------------+------------------+
             |                                     |
       document input                         query input
             |                                     |
   deterministic text windows            deterministic text windows
             |                                     |
      shared runtime encode                 local query runtime lane
             |                                     |
   aggregate / calibrate / budget          aggregate / calibrate
             |                                     |
      lexical + semantic atoms              sparse query atoms
             |                                     |
      canonical index postings  <-------- matching + accumulation
                                                   |
                                            candidate/row recheck
```

編譯器／runtime 契約綁定 atom ID 與數值的解釋方式；只有檔名或模型維度相同並不足夠。不相容的 checkout 身分會安全拒絕，而不是靜默混用某模型的 posting 與另一模型的 query。

### 4.2 Runtime 所有權與遠端編碼

模型 session 由共用 runtime worker 各自擁有，不會為每個 SQL backend 各載入一份模型，也不是 arena 中共享的 ONNX session。Backend 提交有界請求並取得稀疏結果。SSR 必須使用已設定的共用 runtime，不會靜默退回 backend 私有模型。啟用 query-lane reservation 且至少有兩個健康的本地 worker 時，保留的 lane 使文件批次不會耗盡查詢編碼的准入名額。

可選的遠端 runtime service 為構建與維護增加**文件編碼能力**。它們不擁有 PostgreSQL page、不執行資料庫查詢，也不取代本地 query lane。Dispatcher 會考慮未完成請求數、服務 batch 上限、已觀測延遲與退避；批次可以亂序完成並釋放 runtime slot，而發布仍維持文件順序。

這與**持久化查詢加速器**不同：後者是索引擁有、由既有 posting 建成的衍生物件。批量匯入結束後移除遠端編碼資源，並不會因此移除已發布的查詢加速器。兩者詳見 [Shared Runtime and Residency](shared-runtime-and-residency.md)。

### 4.3 模型實驗的具體貢獻

凍結的 P2.1 實驗將表示學習與輕量檢索校準分開：固定上游 encoder，只學習控制 query power、document power 與 score scale 的三個 scalar。訓練使用 10,000 筆 MS MARCO teacher row，以及 1,000 筆 query-disjoint validation row；每筆包含一個 positive 與八個 negative。得到的單調變換為：

```math
\hat q_j=0.696368\,q_j^{1.851864},
\qquad
\hat d_j=d_j^{0.562796}.
```

發布策略之前，support membership 固定為 query top-50、document top-192。歷史 b1.125 策略按詞法 posting 數配置語意 posting，並非目前逐欄位 alpha-mass 選項。Query-local RMS 校準接著對齊詞法與語意 activation 尺度，推理時不需要 qrels。這提供了小而可檢查的適配面，而不是在服務架構中再加入一個大型模型。

上游基礎模型使用公開與非公開訓練材料；此處的可重現性涵蓋固定 checkpoint 及其下游編譯／評估管線。當前 P2.2 身分與歷史 P2.1 量測保持分開；完整訓練目標、選擇過程的注意事項與 artifact 參照，見[模型報告](technical-report-evoke-model-zh.md)。

## 5. 實體索引與 COW 實作

### 5.1 Page-Native 物件

權威索引資料全部位於 PostgreSQL index relation 中。經驗證的根節點指向不可變 manifest，以及 active／pending linked-L0 前沿；下層包含 term 目錄、document／version 中繼資料、詞法查找結構、權威 posting extent、持久化 term fold，以及可選的加速器參照。

```text
  checked root
      |
      +-- immutable manifest
      |      +-- term COW tree ------> extents / neutral / impact folds
      |      +-- document COW tree --> doc slots / versions / TIDs / lengths
      |      +-- lexical lookup ----> full term bytes / stable term IDs
      |      +-- accelerator dir ---> seeds / forward rows / scope data
      |
      +-- pending L0 frontier ------> immutable interval being processed
      +-- active L0 frontier -------> transaction-aware incoming records
```

權威 run 區分詞法中性證據與直接語意 impact。穩定的 term ID 讓詞彙增加時不必為整個語料重新編號。雜湊詞法查找仍比較完整 term 位元組，因此雜湊碰撞不會變成錯誤詞法命中；有序 prefix 結構支援有界前綴展開。文件中繼資料保留 slot incarnation 與 tuple 身分，避免回收的儲存位置被誤認為舊文件。

### 5.2 路徑複製與完整參照

Term 目錄使用持久化 64-way radix tree，每個 leaf 包含 16 個 term。變更建立新的 leaf 與受影響祖先，未變動分支繼續共享。其他 COW 結構使用各自的版面，但遵守相同的不可變參照原則。

```text
  reader pinned at R0                   newly staged R1
          |                                    |
         A0                                   A1
        /  \                                 /  \
       B    C0                                B    C1
           /  \                                /  \
          D    E0                             D    E1
               |                                   |
          old term record                     changed record

  R0 remains readable. Publish R1 only after its new closure is valid.
  Shared B and D are not rewritten merely to change their owner.
```

每個外部參照都包含實體位置、物件身分、所有權、長度與驗證資訊，而不只是邏輯序號。新物件由下而上準備並寫入，取得真實 page 位置之後，才序列化其 parent。這一點很重要：並發 L0 allocation 可能穿插在 COW allocation 之間，若假設實體 page 必然連續，就會破壞原本邏輯正確的樹。

Descendant 共享物件時，實作保留其 ancestor ownership，不會只是為了換 owner 而遞迴複製整個歷史 closure。若樹高為 $h$、有 $m$ 個 leaf 改變，新寫入的中繼資料沿受影響路徑成長；在去除重複路徑之前，可概念性寫成 $O(mh)$，而不是必然重寫 $O(|\mathcal{V}|)$ 的完整詞彙表。這是中繼資料界限，不是變更 posting payload 或完整加速器重建的工作量上界。

實作入口包括 [term COW](../src/evoke_term_cow.c)、[document COW](../src/evoke_document_cow.c)、[lexicon COW](../src/evoke_lexicon_cow.c)，以及 [segment pages](../src/evoke_segment_pages.c) 中由下而上的 writer。完整儲存設計見 [Convergent Segmented Index](convergent-segmented-index.md)。

### 5.3 經驗證的發布與回收

COW 構建是準備，不是發布。Publisher 先驗證 staged closure 與來源／前沿假設，再以短暫、受 WAL 保護的根節點切換安裝 successor。Reader 使用一致的權威來源，不會讀到一半舊 manifest、一半新 manifest。引擎依 PostgreSQL 的 [index access-method](https://www.postgresql.org/docs/18/indexam.html) 與 [extension WAL](https://www.postgresql.org/docs/18/wal-for-extensions.html) 介面整合。

```text
  capture source -> prepare new objects -> validate closure/frontiers
                                               |
                                  short checked-root publication
                                               |
                     +-------------------------+----------------+
                     |                                          |
                new readers use R1                    old readers finish R0
                                                                |
                                                  retire / reclaim safely
```

回收是同一 relation 生命週期中獨立的階段。只要 reader 還可能參照，已被取代的物件就不能立刻重用。符合退休條件且 reader fence 允許後，有界工作才把 page 交回可重用儲存。條件式 fence 取得會讓既有 reader 優先，但成功持鎖後仍可能短暫延遲新 reader；直接重用 page 的寫入須受保護直到發布完成。失敗的準備工作也需要清理。Page 或 slot 重用時，仍需要物件身分與文件 incarnation 檢查。

## 6. 變更、Sealing 與持續收斂

### 6.1 初始構建與發布

初始構建遵循 PostgreSQL 的 `table_index_build_scan` 可見性協定、指派文件身分，並編譯詞法證據；SSR 構建另透過有界 runtime batch 編碼文件。Builder 產生權威 posting 物件與 COW 中繼資料，驗證其 closure 後發布 checked root。`REINDEX` 也以相同表示為目標；健康的當前格式索引，不必只因更新相容衍生加速器就重建整個語料。

```text
  PostgreSQL build protocol -> heap scan -> lexical compilation
                         |
                         +-> SSR document batches -> model runtime
                         |                              |
                         +--------- unified atoms <-----+
                                         |
                          posting objects + COW metadata
                                         |
                           validation + checked root
                                         |
                            query-ready exact storage
                                         |
                      background accelerator / warmup
```

`CREATE INDEX` 與 `REINDEX` 建立權威 root；可選語意加速器由後續背景工作建構。因此 heap scan 完成不等於索引已發布，索引已發布也不等於所有衍生性能產物都完成。構建吞吐、取得 query readiness 的時間，以及進入暖查加速服務的時間，是三種不同量測。

### 6.2 詞法優先的寫入

對 SSR 索引，前台 DML 記錄詞法變化與待完成語意身分，不在寫入交易內執行文件推理。背景完成時會檢查 document version、來源文字與模型契約仍相符；過期結果會捨棄，不會被接到替代資料列上。重複的單列失敗會被呈現並隔離，而不是靜默阻擋整個佇列。

```text
  INSERT / UPDATE transaction
             |
     lexical evidence + pending identity
             |
     transaction-aware linked L0
             |
             +--> commit / abort / savepoint / prepared transaction
             |
             +--> rotate / seal lexical evidence without waiting for model
             |
             `--> bounded semantic batch -> encode changed fields
                           |
                 revalidate version + contract
                           |
                 append completion to linked L0 -> later seal

  checked sealed state -> selected fold / compact / accelerator refresh
```

Heap MVCC 決定可以返回哪些資料列。舊語意分數不能使已刪除或被取代的 tuple version 重新可見；另一方面，相容的近似 baseline 可能在更新前漏掉新插入或分數提升的文件。精確的列可見性與立即完整的語意排名是不同保證。`REPEATABLE READ` 保留 PostgreSQL snapshot，但不會為所有 statement 固定同一個歷史排名根節點。

### 6.3 前沿與 Term Fold

Active L0 接收新記錄，pending 區間獨立處理。Seal 某個區間時，必須保留後來寫入 active frontier 的內容。權威 segment 保持可重用證據；compaction 整理已持久化 posting，不會重新編碼未變動文件。

對 term $t$，令 $F_t^{\leq c}$ 為涵蓋 sequence 邊界 $c$ 的 fold， $E_{t,r}$ 為其後的 extent。精確邏輯 term stream 可表示為：

```math
P_t = F_t^{\leq c}\;\oplus\!
\bigoplus_{r:\,\mathrm{sequence}(r)>c} E_{t,r}.
```

此處 $\oplus$ 表示理解可見性與版本的合併，不是將重複 posting 直接相加。經驗證的 coverage watermark 避免 minor fold 升級為更大 fold 時遺漏或重複計數。Neutral fold 跨 statistics epoch 重用證據；specialized fold 需要匹配的 epoch。兩者共同提供虛擬 sparse column，不要求重寫一份連續的完整語料矩陣。

### 6.4 定期維護，而非服務 TTL

Scheduler 區分緊急性與可執行時機。L0 高水位欠帳、pending 工作、缺少必要產物或產物不相容，都需要處理；小額欠帳在週期到期後也可被處理，因此安靜索引不必等到跨過大閾值才能收斂。最初的低欠帳時間錨點不會被每次後續寫入向後推延。

此基線的 L0 高水位條件包括 65,536 筆記錄或 512 個 page；加速器更新另考慮 sealed record 與 byte 欠帳，低欠帳週期預設為一小時。這些是部署控制，不是稀疏檢索的數學常數。Serving baseline **不會只因存在 delta 就有最大使用年齡**：相容產物可以持續有效，直到替代版本發布。

可選加速器準備使用獨立的 per-index build lock，不會在整體語料掃描期間持有緊急維護／discovery lock。查詢、寫入與安全的 frontier 工作仍可前進；互相衝突的不可變發布可能延後或觸發重試。當 frontier 已無法安全容納更多工作，緊急 sealing 優先。這是在發布邊界上的刻意序列化，不是宣稱整個系統無鎖。

## 7. 查詢執行與 Filter 語義

### 7.1 精確與衍生路徑

符合條件的查詢先綁定索引／模型契約，一次編譯 query atom，再選取服務表示、計算候選分數，並驗證返回的 tuple 身分。Exact posting 路徑是已發布表示的參考；優先衍生路徑則在准入成功時使用相容的候選與緊湊算分產物。

```text
  SQL text + field weights + optional predicate
                       |
            bind index / compile query once
                       |
        compatible accelerator and memory admission?
                    /     \
                  yes      no
                   |        |
           derived candidates    exact posting traversal
           + compact scores      + fold / uncovered evidence
                    \       /
                     \     /
               current-row MVCC / qual recheck
                            |
                       ranked results
```

使用相容的舊 baseline 時，加速路徑以有界 overfetch 補償已變更或失效的資料列，不會在每次查詢中合併無界的 exact post-baseline posting tail。後一種耦合會把背景延遲轉成前台工作量放大。新證據透過後續發布取得；退回 exact 執行仍可能比較昂貴，並不是延遲保證。

### 7.2 Filter 屬於檢索，不只是結果驗證

對允許集合 $A$，過濾短小的未過濾排名前綴，通常不等同在 $A$ 內取得 top-k：

```math
\mathrm{TopK}\{S(q,d):d\in A\}
\neq
A\cap\mathrm{TopK}\{S(q,d):d\in D\}.
```

即使 overfetch，差異仍可能存在，尤其當 scorer 或候選策略依賴選定 scope。只檢查每個返回結果都符合 predicate，並沒有驗證過濾後的排名品質。

Evoke 由合格的 `INCLUDE` 欄位建構 same-root scope metadata。支援的 planner predicate 包括直接 AND 組合的相等、overlap、range 與可准入的 `ILIKE` 形狀。Structured JSON filter 也可重用相容且已發布的 scope baseline。兩者均重新檢查當前資料列 membership，並可依近似契約暫時漏掉 baseline 之後的新命中。

Fallback 契約明確區分不同路線：planner-native 在 scope 路徑不可用或不合適時，可以使用完整的 visible TID 集；完全由 scope 支援的 structured 請求，在收斂期間可以返回少於 k 筆，而不自動建立完整 matching universe。其他 structured 請求先執行上限 65,536 筆匹配加一筆溢出見證的 SQL membership probe；若完整取得匹配集合，便使用該 TID 集。若溢出，可嘗試有界 global rank-prefix 准入，再回退完整 SQL resolution。匹配數上限並不限制掃描列數或執行時間。因此，「所有 filter 都必須 exact-current 列舉」與「所有 filter 都不需列舉」都不能代表目前產品。完整 overload 規則見 [Query Semantics](query-semantics.md)。

## 8. 持久化語意查詢加速器

### 8.1 既有證據的衍生表示

查詢加速器從不可變 posting baseline 建立，組合逐 term 選取的 seed document、跨 term 的 residual 候選累加、可直接算分的緊湊 forward row，以及適用時的 scope metadata。Forward 與 inverted 視圖服務不同存取模式：inverted 視圖找出與 query atom 關聯的文件；forward 視圖計算選定文件，避免重複遍歷每條長 posting list。

```text
  immutable lexical + completed semantic postings
                       |
             stream one term at a time
                       |
         seeds + residual data + temporary transpose
                       |
          compact forward rows + INCLUDE scope
                       |
           validate source/policy + publish directory
                       |
  query atoms -> candidates -> forward scores -> current-row recheck
```

目前實作使用 policy 7、accelerator directory v10、forward／transpose format v6，以及 scope format v6；這些元件版本不等於 extension 版本。候選設計包括每個選定 term 最多 64 個 seed document，以及額外跨 term 工作；「64 seeds」不表示整個查詢只能檢查 64 份文件。衍生緊湊 row 使用逐列尺度、delta-varint term ID 與 signed-int8 貢獻值，與權威語意 posting 可選的 unsigned `u8` codec 不同。

### 8.2 記憶體有界，不代表總工作量固定

Builder 串流讀取 posting，並以暫存檔支援轉置。准入估算包含文件中繼資料、詞彙狀態、最大解碼 term、sort workspace 與發布暫存，避免把全部語料 posting 同時載入 RAM。Scope extraction 以有界 MVCC snapshot batch 讀取 heap 值，包括外部 TOAST 資料；若整個長時間構建都固定同一個舊 snapshot，會不必要地延後 dead tuple 回收。

完整 refresh 仍可能掃過 baseline 的全部 posting，並收集其 scope 值。它**不是**增量 O(delta) 的模型推理工作，也不保證在單次 scheduler tick 內完成。將其鎖與緊急維護分離，再配合重試冷卻，可避免成為序列化的前台依賴。CPU、磁碟頻寬、page cache、WAL 與暫存空間仍是共享實體資源，必須在負載下量測。

### 8.3 新鮮度與預熱使用不同身分

服務中的索引有四個可獨立觀測的面向：

| 面向 | 回答的問題 |
| --- | --- |
| 可見性 | 返回的資料列版本是否符合此 SQL snapshot？ |
| 語意完成度 | 待處理文件版本是否已取得模型證據？ |
| 加速器新鮮度 | 衍生執行器代表哪個已完成 baseline？ |
| 預熱狀態 | 有用的中繼資料與 page 是否位於預期的 cache／shared-memory 層？ |

`ready_baseline_delta` 是可服務狀態，不等於「沒有加速器」；`baseline_current=false` 表示仍有收斂工作。Warm marker 綁定 serving accelerator／baseline 身分，不會在每次無關的 manifest 推進時被丟棄，因此 COW 發布與小額新寫入可以和穩定的暖查路徑共存。

Shared runtime 容量、page prewarm I/O budget 與 exact resident-fold 准入是不同預算。將 page 讀入 cache 不等於永久保留它，放大預熱預算也不會消除 posting 工作或磁碟放大。模型、共享表示、backend 暫存與 OS cache 必須分開計算；將所有 process RSS 相加可能重複計入共享 page。詳見 [Connection Memory](connection-memory.md) 與 [Shared Runtime and Residency](shared-runtime-and-residency.md)。

## 9. 實驗證據

### 9.1 方法與歸屬

證據分為詞法效率、模型相關性、衍生執行器效率，以及增量儲存行為，各自回答不同問題。下列數字取自引用的已提交記錄，不是在撰寫本報告時重新執行得到的結果。

延遲比較必須說明語料／tokenizer 身分、SQL 形狀、k、cache 狀態、模型／codec、executor policy、硬體與並發工作。Server execution time 不包含應用網路及後續 RAG 階段；直接 SRF microbenchmark 也不自動等同完整業務查詢的 `EXPLAIN ANALYZE`。標示 macro 的品質指標按資料集等權平均，不是將所有 query 混合後計算。

### 9.2 詞法基礎與回歸修復

[2026-08-18 BM25 回歸研究](performance/reports/bm25-page-native-regression-2026-08-18.md)使用 SciFact：5,183 篇文件、1,109 個 query、26,559 個 term、top-1000、固定 tokenization 與 Lucene BM25 參數。修復版本先預熱兩輪，再量測三輪；[機器可讀記錄](performance/data/diagnostics/bm25-page-native-regression-2026-08-18.json)保留 fixture 與時間結果。

| 記錄中的實作 | Mean ms | p50 ms | p95 ms |
| --- | ---: | ---: | ---: |
| 原始官方 `psql_bm25s` 基線 | 0.454 | 0.451 | 0.525 |
| 修復前的 page-native 路徑 | 63.430 | 63.370 | 65.896 |
| 修復後的 page-native 路徑 | 0.448 | 0.435 | 0.558 |

這個 fixture 只有 BM25，因此回歸不是語意推理的必然成本。Page 路徑主要耗時於逐 hit 重複的 document-COW projection；按 block 合併 projection、有界 resident scoring 與緊湊的 tie-selection 狀態，恢復了記錄中的平均性能。修復後 mean 比舊記錄約低 1.4%，p95 則約高 6.3%；不能將它寫成「所有百分位都沒有回退」。在抽樣的 101 個 query 中，resident 與強制 page 路徑的身分／順序不一致數為零，最大分數差也為零。

這建立了一個有用的詞法回歸 fixture，不是目前所有 PostgreSQL BM25 產品的跨引擎排名。完整研究中的歷史比較與同 cluster 控制組必須分開解讀。

### 9.3 模型階段成果：P2.1 統一檢索

凍結於 2026-07-15 的 P2.1／b1.125 評估涵蓋 BEIR15 的 46,417 個 query、33,860,494 篇文件，以及 MTEB10 的 8,815 個 query、1,096,451 篇文件。下表均為資料集／task 等權 macro 平均。CUB@1000 表示候選數最多 1,000 時，記錄中的 positive-document candidate coverage，不是實際 reranker 的結果。

| 評估組 | Retriever | NDCG@10 | Recall@100 | CUB@1000 |
| --- | --- | ---: | ---: | ---: |
| BEIR15 | BM25 | 0.374297 | 0.562964 | 0.735619 |
| BEIR15 | PPLX dense / VectorChord | 0.544873 | 0.670880 | 0.810525 |
| BEIR15 | Evoke P2.1 | 0.490809 | 0.666885 | 0.839658 |
| MTEB10 | BM25 | 0.383554 | 0.595367 | 0.777387 |
| MTEB10 | PPLX dense / VectorChord | 0.544127 | 0.707213 | 0.841063 |
| MTEB10 | Evoke P2.1 | 0.503440 | 0.703125 | 0.875910 |

相對於 BM25 到 dense 的 Recall@100 提升，P2.1 在兩組評估上都取得約 96.3% 的增益：

```math
G_R=\frac{R_{P2.1}-R_{BM25}}{R_{dense}-R_{BM25}}.
```

主要階段成果是在單一稀疏 posting 索引內取得接近 dense 參考的語意候選召回，並在這些 macro 統計上得到更高的候選上界。更好的前段排名是清楚的改進方向：即使 Recall@100 接近，NDCG@10 仍低於 dense 參考。

這些結果使用歷史 P2.1 support、校準、發布 budget 與文字長度限制，不是 P2.2 全文／U8／alpha-0.50 基準。BEIR dense 的十三列沿用先前 VectorChord 結果，兩列重新收集；dense query encoder 延遲未作為可比的端到端成本包含在內。部分校準／策略選擇使用過 BEIR 證據，因此不能稱為完全未接觸評估集的 zero-shot 評估。[模型報告](technical-report-evoke-model-zh.md)提供逐資料集數字、方法與 artifact 連結。

### 9.4 衍生執行器成果與被否決的捷徑

[加速器執行記錄](performance/reports/semantic-accelerator-bounded-execution.md)區分原型與已安裝 binary 的證據，其中 policy-3 Shadow ArXiv 部署比較為：

| 固定測試面 | Exact p50/p95 ms | Derived p50/p95 ms | Mean/minimum O@100 |
| --- | ---: | ---: | ---: |
| 跨領域 150 個 query | 215.22 / 272.56 | 138.83 / 163.71 | 0.9826 / 0.92 |
| 獨立 TREC 50 個 query | 218.88 / 272.28 | 134.50 / 153.27 | 0.9834 / 0.92 |

O@100 衡量與 exact top-100 的重疊，不是 qrels recall。對應 artifact 從不可變 posting 重建耗時 2 分 21 秒，沒有執行文件推理。這些結果展示緊湊 forward scoring 與重用 posting 的價值；它們屬於歷史 policy 3，而目前設計是 policy 7。對目前 policy 的准入，仍須將新量測綁定到實際安裝的 artifact 與 binary。

同一份記錄也否決了表面上很有吸引力的捷徑。在 523 萬篇文件的 Hotpot 測試面上，約 11 ms 的幾何搜尋提供的候選不夠完整；補回 exact residual coverage 後，整體執行反而慢於 exact 路徑。增加每個 term 的 seed cap，也未得到良好的全域品質／成本取捨。經驗是優化**符合排名品質要求的完整檢索工作量**，而不只是孤立且快速的候選 kernel。

### 9.5 COW 更新成本與儲存收斂

[COW 設計記錄](convergent-segmented-index.md)在移除遞迴 full-root reuse inventory 後，記錄了固定變更量的中繼資料實驗：

| 詞彙 term 數 | Before median ms | Changed-path median ms |
| ---: | ---: | ---: |
| 2,000 | 5.93 | 1.29 |
| 20,000 | 64.81 | 1.51 |
| 100,000 | 272.93 | 1.93 |

這是歷史中繼資料 microbenchmark，不是端到端索引構建吞吐。它支持一個具體設計主張：小幅更新不應走遍全部未變動詞彙中繼資料。另外，[工程開發記錄](archive/engineering/convergent-segmented-index-development-record.md)記錄固定 live set 的變更測試：加入回收與 incarnation-safe reuse 後，page 與 document slot 高水位能夠進入平台期。兩類測試分別處理 CPU 放大與不可變儲存的磁碟代價。

## 10. 工程准入與可重現性

核心證明義務不只是一個 scorer 單元測試：

| 性質 | 必須觀測的內容 |
| --- | --- |
| 排名 | 結果身分、順序、分數容差、qrels 指標，以及完整 filtered reference 比較 |
| 交易 | Commit／abort、savepoint、prepared transaction、HOT／non-HOT update、delete 與 TID reuse |
| 發布 | 不混讀不同 root；準備失敗或取消時保留舊的可讀 root |
| 收斂 | Pending 工作排空；停止寫入後，可准入的加速器最終推進 |
| 服務連續性 | 維護前／中／後的 warm baseline、延遲尾部與 lock wait，而不只 readiness flag |
| 資源 | Private memory／PSS、shared memory、暫存磁碟、WAL、page 成長與排空後的平台期 |
| 恢復 | 冷啟動、crash replay、`REINDEX`、reader-safe 回收與實體複寫 |
| 套件身分 | Extension SQL／binary、PostgreSQL major、ORT ABI、model digest 與 accelerator policy |

[驗證指南](testing-and-validation.md)將這些義務對應到隔離 PostgreSQL lifecycle、writer concurrency、semantic fairness、memory 與 replication harness。Python unit／contract test 是另一層，不能用來證明部署現場的延遲成果。

可重現的混合負載實驗應記錄暖查唯讀基線、平行 reader 控制組、持續寫入加維護，以及寫入停止後的排空階段。需一起取樣 query p50／p95／最大值、結果及分數品質、source／serving generation、L0／semantic debt、worker action、已處理 byte、I/O、記憶體與磁碟增長。健康系統可以有受控抖動及短暫較舊的排名；只有 readiness 無法證明 worker 持續前進，也不能證明查詢工作量穩定。

PostgreSQL 負責 `DROP INDEX` 清理與索引 page 的實體複寫；standby 仍需配置匹配的外部模型／runtime artifact。Logical replication 傳遞資料列，不傳遞實體 Evoke 索引。目前支援範圍不含 RLS-backed search 與全域排名的 partitioned-parent index；parallel heap build、parallel AM scan 與 parallel VACUUM discovery 也是後續工程方向。當前邊界與安裝細節見 [README](../README.md) 及 [Migration](upgrading.md)。

## 11. 改進方向

下一階段是強化同一架構，而非增加相互競爭的索引權威來源：

- **更廣的模型證據與更好的前段排名。** 在套件 P2.2 文字路徑上重跑完整品質矩陣，擴大獨立 holdout 與長文件評估，改善校準，讓高候選召回更充分地轉成前段精度。
- **高 document frequency 的執行效率。** 在驗證完整 top-k 品質的前提下減少 posting 與 residual 總工作量，特別是寬 filter 和長 query；評估端到端成本，而非只有幾何搜尋速度。
- **可預測的並發行為。** 跨語料規模與硬體擴展混合負載量測，改善背景資源准入與公平性，讓新鮮度／延遲取捨更容易觀察。
- **緊湊的權威與衍生表示。** 持續將每 posting byte 數、COW 中繼資料成本、構建暫存空間及 cache residency，與檢索品質、恢復行為一起量測。
- **更廣部署與可重現性。** 擴充合格輸入與工作負載覆蓋，並發布嚴格綁定 binary／model／data manifest 的新量測。

具體提案、實驗矩陣與 promotion gate 應放在 [Product Roadmap](product-roadmap.md) 和 [Model Planning](model-planning.md)，不代表本報告已實作這些未來功能。

## 12. 結論與審閱索引

Evoke 的工程主張是：詞法與學習式稀疏檢索不只可以共用分數累加器，也可以共用交易型儲存與維護設計。COW 讓不可變證據得以重用，有界前沿局部化一般變更工作；worker 獨立編譯語意證據及準備衍生讀取結構；經驗證的發布讓新結構上線，而不必先拆除舊的服務路徑。

既有實驗建立了具體階段成果：受控回歸 fixture 上的詞法效率、良好的稀疏語意候選召回、衍生執行器的延遲改善，以及降低的中繼資料放大。下一個評估重點是目前套件模型在真實混合負載下的整體行為。適當的成功標準，是收斂過程中持續提供有用排名與可預測的資源使用，而不只是一個快速的唯讀快照，或所有狀態欄位都顯示正常。

進行程式碼導向審閱時，可從下列對照開始：

| 領域 | 主要參考 |
| --- | --- |
| 沿革與模型成果 | [詞法報告](technical-report-psql_bm25s.md)、[模型報告](technical-report-evoke-model-zh.md) |
| 索引版面與 COW | [儲存設計](convergent-segmented-index.md)、[term COW header](../src/evoke_term_cow.h)、[segment pages](../src/evoke_segment_pages.c) |
| 查詢與 filter | [查詢契約](query-semantics.md)、[page query](../src/evoke_page_query.c)、[scope](../src/evoke_scope.c)、[filter](../src/evoke_filter.c) |
| 模型執行 | P2 runtime 實作、[runtime 契約](shared-runtime-and-residency.md)、[模型 lock](../packaging/milestone-model.json) |
| 加速器 | [Builder](../src/evoke_am_accelerator.c)、[directory](../src/evoke_semantic_accelerator_directory.c)、[forward format](../src/evoke_semantic_forward.c)、[執行證據](performance/reports/semantic-accelerator-bounded-execution.md) |
| 並發與准入 | [Lifecycle](maintenance-lifecycle.md)、[scheduler](../src/evoke_am_scheduler.c)、[validation](testing-and-validation.md) |

外部基礎文獻：[BM25S](https://arxiv.org/abs/2407.03618)、[Granite Embedding Models](https://arxiv.org/abs/2502.20204)，以及 PostgreSQL 的 [index access-method](https://www.postgresql.org/docs/18/indexam.html) 和 [extension WAL](https://www.postgresql.org/docs/18/wal-for-extensions.html) 文件。外部工作的貢獻各自歸屬於原作者；上述 Evoke 性能數字來自連結的專案實驗證據，而不是這些論文。
