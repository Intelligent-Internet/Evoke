# Evoke 模型技術報告 (Beta 1)

- 報告修訂日期：2026-09-03
- 評估快照日期：2026-07-15
- 狀態：Beta 1
- 當前套件契約：Evoke v0.2.5 / P2.2 / ABI-v2
- 評估形態：原生單一 unified posting index
- 對照：BM25、PPLX dense（VectorChord）
- 固定評估版本：P2.1 / b1.125
- 語言：繁體中文 | [English](technical-report-evoke-model.md)

## 摘要

Evoke 模型 Beta 1 建立在 P2.1 learned sparse retrieval 路線及其
套件化後繼版本 P2.2 之上。它不是把 BM25
與 ANN 結果做 RRF 或後期 fusion，而是由一個約 30.3M 參數的 sparse
encoder 生成語義 posting，並把 lexical posting 與 semantic posting
放入互不衝突的 namespace，最終由同一物理倒排索引、同一次 sparse dot
product 完成候選召回。

本報告保留使用固定 P2.1 路線測得的兩個原生工程評估面：

1. 完整 BEIR15：15/15 datasets、46,417 queries、33,860,494 documents。
2. 固定 MTEB10 retrieval 面：10/10 tasks、8,815 queries、1,096,451
   documents。

主要結果如下：

| 評估面 | 方法 | NDCG@10 | MAP@100 | Recall@100 | MRR@20 | CUB@1000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BEIR15 | BM25 | 0.374297 | 0.274610 | 0.562964 | 0.475641 | 0.735619 |
| BEIR15 | PPLX dense / VectorChord | **0.544873** | **0.412036** | **0.670880** | **0.650711** | 0.810525 |
| BEIR15 | P2.1 | 0.490809 | 0.371455 | 0.666885 | 0.595950 | **0.839658** |
| MTEB10 | BM25 | 0.383554 | 0.268718 | 0.595367 | 0.461805 | 0.777387 |
| MTEB10 | PPLX dense / VectorChord | **0.544127** | **0.413201** | **0.707213** | **0.629992** | 0.841063 |
| MTEB10 | P2.1 | 0.503440 | 0.374100 | 0.703125 | 0.578901 | **0.875910** |

P2.1 在兩個面上都恢復了約 96.3% 的「dense 相對 BM25 的
Recall@100 增益」，Recall@100 與 dense 的絕對差距分別只有 0.0040
與 0.0041；CUB@1000 則分別高出 dense 0.0291 與 0.0348。這證明
P2.1 已具備適用於 RAG first-stage retrieval 的強候選生成能力。以此召回
基礎為起點，下一步品質方向是提升 NDCG、MAP 與 MRR，向 dense 級頭部
排序品質前進。

本技術報告介紹 Evoke 模型 Beta 1，記錄模型設計、
已測得的檢索品質與評估範圍。P2.1 矩陣提供模型路線固定的品質與延遲基線；
當前 P2.2 的全矩陣評估及面向工作負載的效能目標是後續方向。具體工作另存於
[Model Planning](model-planning.md)。

## 1. 報告範圍與評估口徑

### 1.1 三種工程實體

| 名稱 | 工程實體 | 主要用途 |
| --- | --- | --- |
| BM25 | lexical inverted index | 精確詞項與 lexical baseline |
| PPLX dense | `perplexity-ai/pplx-embed-v1-0.6B` 的 1024 維 stored embedding，加 VectorChord | dense baseline |
| P2.1 | Granite sparse compiler、lexical/semantic unified postings、單一 native inverted index | 本次發布的固定評估基線 |

Dense baseline 使用 PPLX 1024 維向量和 VectorChord 搜尋；矩陣測量的是已
寫入資料庫的 embedding 與 VectorChord index 品質，不包含 PPLX encoder
端到端推理時間。因此，本報告不把 VectorChord lookup latency 與 P2.1
「tokenizer + ONNX + calibration + merge + lookup」延遲直接比較。

### 1.2 指標定義

- `NDCG@10`：使用 graded relevance 的前 10 名排序品質。
- `MAP@100`：前 100 名 average precision，分母為該 query 的全部正例。
- `Recall@100`：全部正例中進入前 100 名的比例。
- `MRR@20`：前 20 名第一個正例的 reciprocal rank。
- `CUB@1000`：candidate upper bound；全部正例中進入最多 1,000 個候選的比例。
- `Macro`：dataset/task 等權平均，而不是 query 加權平均。

品質表格顯示到小數點後六位；可重現 JSON 保留完整浮點精度。

### 1.3 發布身份與證據邊界

Beta 1 保留 P2.2 工程識別與 extension 版本 v0.2.5。
當前套件由[模型鎖定檔](../packaging/milestone-model.json)綁定：

| 項目 | Beta 1 套件契約 |
| --- | --- |
| Bundle | `evoke-p2.2-nfcorpus-v2` |
| Model ID | `evoke_p2_p22_nfcorpus_v2_smoke` |
| Manifest SHA-256 | `b61060a3958ee56209de47a34ee5cbe08351bfeb3fcbbfdcbf477403210764f7` |
| Runtime ABI | `evoke_p2_unified_text_atoms_v2` |
| ONNX Runtime | 1.29.0，由[依賴鎖定檔](../packaging/onnxruntime.version)指定 |

完全相同的凍結 checkout 已以
[Evoke Model (Beta 1)](https://huggingface.co/Intelligent-Internet/Evoke-Model-Beta-1)
發布於 Hugging Face；取得方式見[固定版本下載與校驗說明](examples/semantic-model-checkout.md#download-the-default-model)。
此次分發不改變模型身份或實驗證據邊界。

P2.2 保留 Granite sparse 母體，透過 ABI-v2 的確定性視窗處理完整 query 與
document 文字。套件 checkout 的 lexical vocabulary 與 calibration 固定來自
NFCorpus，更廣泛的跨領域校準是後續改進方向。部署自行驗證的 checkout 仍遵守
同一[模型契約](examples/semantic-model-checkout.md)。

第 2 至 8 節描述固定 P2.1 評估流程，包括 b1.125 發布預算、ABI-v1 artifact
與歷史 runtime；當前套件契約則由上表單獨標識。
當前引擎預設使用 `f32` semantic impacts 與 `semantic_alpha_mass = 1.0`；
`u8` 和 alpha 0.50 是明確選用的近似索引配置，各自與固定基準配置分開評估。
目前儲存、查詢與生命週期語義以[架構文件](architecture-and-design.md)及
[查詢語義](query-semantics.md)為準。

## 2. P2.1 架構

```text
raw text
   |
   v
RoBERTa byte-level BPE
   |
   v
frozen Granite 30M sparse encoder (ONNX, CPU)
   |
   +--> fixed-support semantic impacts
   |       query top-50 / document top-192
   |       M1914 monotonic power calibration
   |       RMS query-local scale
   |
   +--> lexical TF/BM25-equivalent impacts
           |
           v
disjoint lexical + semantic atom namespaces
           |
           v
one unified posting map / one physical inverted index
           |
           v
one additive sparse dot-product lookup
```

P2.1 的產品邊界有三個重要限制：

1. 沒有外部 ANN index。
2. 沒有把 BM25 與 semantic ranking 做 RRF 或 post-hoc score fusion。
3. lexical 與 semantic atoms 雖屬不同 namespace，但共同寫入一個 posting
   index，並在一次查詢中累加。

## 3. 模型與訓練設計

### 3.1 成熟 sparse 母體

P2.1 不從頭重新學習語言與 retrieval geometry，而是使用
`ibm-granite/granite-embedding-30m-sparse`：

- revision：`ad82b1fd09541c998c8d45045d601c51fdb8a9b7`
- 參數量：約 30.3M
- Transformer layers：6
- hidden / embedding size：384
- intermediate size：1,536
- vocabulary：50,265
- maximum sequence length：512

Granite sparse 模型採 retrieval-oriented pretraining、dense teacher
distillation 與 sparse regularization。其設計與訓練細節可見
[Granite Embedding Models](https://arxiv.org/abs/2502.20204)。上游 checkpoint
採 Apache-2.0；但其完整資料混合包含非公開資料，因此 P2.1 的本地可重現性
是「固定上游 checkpoint 後的 compiler 與 index pipeline」，不是宣稱可從
所有公開資料完整重訓上游母體。

### 3.2 Sparse impact 生成

對 token 位置 $i$ 與 vocabulary/latent coordinate $j$ 的 logit $z_{ij}$，
sequence-level sparse impact 為：

```math
w_j(x)=\max_{i\in x}\log\left(1+\mathrm{ReLU}(z_{ij})\right).
```

語義分數為：

```math
S_{sem}(q,d)=\sum_j w_j(q)w_j(d).
```

max pooling 讓每個 coordinate 可以直接發布為倒排 posting，但也表示 corpus
wide document frequency 和 posting traversal 是產品成本的重要控制面。

### 3.3 上游訓練目標

令 teacher 與 student 在同一 candidate set $D_q$ 上的 score distribution 為：

```math
P_T(d\mid q)=\frac{\exp(s_T(q,d)/\tau_T)}
{\sum_{d'\in D_q}\exp(s_T(q,d')/\tau_T)},
```

```math
P_S(d\mid q)=\frac{\exp(s_S(q,d)/\tau_S)}
{\sum_{d'\in D_q}\exp(s_S(q,d')/\tau_S)}.
```

其 score-distribution distillation loss 為：

```math
\mathcal{L}_{KD}=-\sum_{d\in D_q}P_T(d\mid q)\log P_S(d\mid q).
```

標準 sparse regularization 可寫為：

```math
\mathcal{L}_{FLOPS}=
\sum_j\left(\frac{1}{B}\sum_{i=1}^{B}w_j(x_i)\right)^2,
```

```math
\mathcal{L}_{NORM}=\sum_j w_j(x).
```

上游總目標的抽象形式為：

```math
\mathcal{L}=
\mathcal{L}_{KD}
+\lambda_q\mathcal{L}^{q}_{FLOPS}
+\lambda_d\mathcal{L}^{d}_{FLOPS}
+\sigma_q\mathcal{L}^{q}_{NORM}
+\sigma_d\mathcal{L}^{d}_{NORM}.
```

KD 保留 retrieval score geometry；FLOPS 約束 batch-level coordinate usage；
NORM 約束單樣本 impact 總量。三者共同控制品質、posting sparsity 與倒排成本。

### 3.4 M1914 固定 support 校準

P2.1 的本地訓練不改 sparse support membership，只校準 impact geometry：

```math
\hat q_j=0.696368\,q_j^{1.851864},
\qquad
\hat d_j=d_j^{0.562796}.
```

query 固定保留 top-50，document 固定保留 top-192。訓練資料是固定的
MS MARCO teacher surface：10,000 train rows、1,000 query-disjoint validation
rows，每列一個正例與八個負例。僅學習全局 query power、document power
與 score scale 三個 scalar；沒有 dataset-specific branch。

令正負 margin 向量為 $m$，本地校準 loss 為：

```math
\mathcal{L}_{cal}=
D_{KL}(P_T\Vert P_\theta)
+0.05\,\mathrm{MSE}(m_\theta,m_T)
+10^{-3}\left[(\gamma_q-1)^2+(\gamma_d-1)^2\right].
```

訓練使用 AdamW、1,000 steps、batch 128、learning rate `1e-2`、seed
1914；power 被限制在 `[0.5, 2.0]`。此設計的重點是 monotonic
calibration：改善 score distribution，但不重新選擇 atoms。

### 3.5 Lexical impact 與單索引算分

document lexical impact 使用 BM25-equivalent term contribution：

```math
\mathrm{idf}_t=
\log\left(1+\frac{N-df_t+0.5}{df_t+0.5}\right),
```

```math
d^{L}_t=\mathrm{idf}_t
\frac{tf_{t,d}}
{tf_{t,d}+k_1\left(1-b+b\frac{|d|}{\overline{|d|}}\right)},
```

其中 $k_1=1.5$ 與 $b=0.75$，query lexical impact 為 query term
frequency。這與標準 BM25 只差一個不影響排序的全局 $(k_1+1)$ 因子。

lexical atoms 和 semantic atoms 位於不相交 namespace，故沒有交叉項：

```math
S_{P2}(q,d)=
\sum_{t\in\mathcal{V}_L}q^L_td^L_t
+c(q)\sum_{j\in\mathcal{V}_S}\hat q_j\hat d_j.
```

### 3.6 Query-local RMS calibration

對每個 atom 的 corpus RMS：

```math
r_j=\sqrt{\frac{1}{N}\sum_d d_j^2}.
```

query 的 lexical 與 semantic activation proxy 為：

```math
A_L(q)=\sum_t q^L_t r_t,
\qquad
A_S(q)=\sum_j \hat q_j r_j.
```

固定 global scale $g=6.281606583836263$，P2.1 使用：

```math
c(q)=\mathrm{clip}
\left(4\frac{A_L(q)}{A_S(q)},\,0.5g,\,4g\right).
```

若 proxy 無效則回退到 $g$。推理不需要 qrels；但 multiplier 與 policy
family 曾在四個 BEIR rows 上做全局 LODO 選擇，因此不能把整條 P2.1
路線宣稱為完全 untouched BEIR zero-shot。

### 3.7 b1.125 發布預算

`b1.125` 不是另一個 neural checkpoint，而是固定的 deterministic
publisher budget：

```math
|P_{semantic}|=1.125\,|P_{lexical}|,
```

因此在兩個 disjoint namespaces 下：

```math
|P_{total}|=2.125\,|P_{lexical}|.
```

它是此前 budget frontier 上最小、且能在選擇面保持全部宏觀指標改善的
point；四個 LODO fold 均選到 b1.125，之後在 SciDocs、Quora、
TREC-COVID 三個未見 rows 上通過 transfer 驗證。

## 4. 模型體積與工程規模

| 項目 | P2.1 |
| --- | ---: |
| Sparse encoder 參數 | 約 30.3M |
| PPLX dense 對照參數 | 名義約 0.6B |
| 參數量比例 | P2.1 約為 PPLX 的 1/19.8 |
| ONNX artifact | 198,727,261 bytes（189.52 MiB） |
| ONNX SHA-256 | `12daec0053759f4bc2a4106d52ffb40d624372e7580af19c1ec7985f394c4999` |
| 評估 runtime | ONNX Runtime 1.27.1 / CPU EP（歷史環境） |
| Query support | top-50 semantic atoms |
| Document support | top-192 semantic atoms |
| Candidate depth | 1,000 |
| Native ABI | `evoke_p2_unified_text_atoms_v1` |

P2.1 原生 runtime parity 已在 1,623 queries 上驗證：atom IDs 全部精確
一致，impact 使用 absolute tolerance `2e-4`、relative tolerance `2e-2`。
模型 artifact、tokenizer、compiler 與 generation 都有固定 hash/manifest。

小型產品 smoke surface 包含 34,473 documents、1,623 queries：

- 8,022,679 postings，平均約 232.7 postings/document。
- posting payload 122.72 MB。
- index build 25.01 seconds。
- crash recovery 與 immutable artifact 驗證已通過。

## 5. 宏觀結果

### 5.1 P2.1 相對 BM25

| 評估面 | ΔNDCG@10 | ΔMAP@100 | ΔRecall@100 | ΔMRR@20 | ΔCUB@1000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BEIR15 | +0.116511 | +0.096845 | +0.103922 | +0.120309 | +0.104039 |
| MTEB10 | +0.119885 | +0.105382 | +0.107757 | +0.117095 | +0.098524 |

P2.1 對 BM25 的 row wins：

- BEIR15：五個指標均為 14/15。
- MTEB10：NDCG 9/10，其餘四個指標均為 10/10。

這表示改善不是由單一大型 dataset 支撐，而是具有較廣泛的 row-level
一致性。

### 5.2 P2.1 相對 PPLX dense / VectorChord

| 評估面 | ΔNDCG@10 | ΔMAP@100 | ΔRecall@100 | ΔMRR@20 | ΔCUB@1000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BEIR15 | -0.054065 | -0.040581 | -0.003995 | -0.054761 | **+0.029133** |
| MTEB10 | -0.040688 | -0.039101 | -0.004088 | -0.051091 | **+0.034847** |

P2.1 對 dense 的 row wins：

| 評估面 | NDCG | MAP | Recall | MRR | CUB |
| --- | ---: | ---: | ---: | ---: | ---: |
| BEIR15 | 3/15 | 3/15 | 11/15 | 2/15 | 12/15 |
| MTEB10 | 2/10 | 3/10 | 7/10 | 3/10 | 7/10 |

若以 dense 相對 BM25 的增益為分母，P2.1 的增益恢復率為：

| 評估面 | NDCG | MAP | Recall | MRR |
| --- | ---: | ---: | ---: | ---: |
| BEIR15 | 68.30% | 70.47% | **96.30%** | 68.72% |
| MTEB10 | 74.66% | 72.94% | **96.34%** | 69.62% |

這是 P2.1 產品定位的數學依據：candidate membership 幾乎達到 dense，
但 top-rank geometry 尚未完全恢復。

## 6. BEIR15 完整矩陣

評估規模：15 datasets、46,417 queries、33,860,494 documents、
161,708 qrels。

| Dataset | Method | NDCG@10 | MAP@100 | Recall@100 | MRR@20 | CUB@1000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ArguAna | BM25 | 0.344115 | 0.237186 | 0.952891 | 0.233893 | 0.990007 |
|  | PPLX dense | 0.429182 | 0.300000 | 0.972877 | 0.298787 | 0.972877 |
|  | **P2.1** | 0.411456 | 0.286486 | **0.987866** | 0.284977 | **1.000000** |
| Climate-FEVER | BM25 | 0.127530 | 0.097173 | 0.341694 | 0.181684 | 0.567427 |
|  | **PPLX dense** | **0.394597** | **0.314161** | **0.703507** | **0.513265** | **0.836721** |
|  | P2.1 | 0.228576 | 0.175821 | 0.530945 | 0.315448 | 0.743779 |
| CQADupStack | BM25 | 0.291987 | 0.265961 | 0.524878 | 0.295992 | 0.697639 |
|  | **PPLX dense** | **0.431176** | **0.389596** | **0.747247** | **0.425859** | 0.870515 |
|  | P2.1 | 0.418136 | 0.380379 | 0.730500 | 0.416496 | **0.888289** |
| DBPedia | BM25 | 0.239804 | 0.173662 | 0.390971 | 0.504366 | 0.580338 |
|  | PPLX dense | **0.387310** | 0.250729 | 0.458798 | **0.733080** | 0.629907 |
|  | **P2.1** | 0.353671 | **0.261121** | **0.528869** | 0.689000 | **0.733407** |
| FEVER | BM25 | 0.449661 | 0.395967 | 0.838526 | 0.410674 | 0.933235 |
|  | **PPLX dense** | **0.846450** | **0.820341** | 0.905383 | **0.862403** | 0.912997 |
|  | P2.1 | 0.766722 | 0.713127 | **0.949535** | 0.756337 | **0.967121** |
| FiQA | BM25 | 0.231073 | 0.183892 | 0.510764 | 0.292597 | 0.725380 |
|  | **PPLX dense** | **0.510599** | **0.451725** | **0.804102** | **0.599713** | **0.912474** |
|  | P2.1 | 0.358516 | 0.297809 | 0.662176 | 0.439020 | 0.855207 |
| HotpotQA | BM25 | 0.511627 | 0.430958 | 0.724646 | 0.662780 | 0.847738 |
|  | **PPLX dense** | **0.688827** | **0.619042** | 0.798447 | **0.834553** | 0.856516 |
|  | P2.1 | 0.659351 | 0.575048 | **0.816205** | 0.824656 | **0.901283** |
| MS MARCO | BM25 | 0.366717 | 0.264019 | 0.408121 | 0.743909 | 0.672591 |
|  | **PPLX dense** | **0.647114** | **0.383274** | 0.472836 | **0.961240** | 0.700965 |
|  | P2.1 | 0.524892 | 0.346804 | **0.487246** | 0.884367 | **0.784065** |
| NFCorpus | BM25 | 0.306793 | 0.137331 | 0.233236 | 0.515219 | 0.424066 |
|  | PPLX dense | 0.322832 | 0.139780 | 0.269111 | 0.548789 | 0.541961 |
|  | **P2.1** | **0.347391** | **0.169076** | **0.299101** | **0.567236** | **0.588587** |
| NQ | BM25 | 0.242799 | 0.203360 | 0.678568 | 0.215921 | 0.860926 |
|  | **PPLX dense** | **0.584749** | **0.517974** | 0.912732 | **0.541040** | 0.935762 |
|  | P2.1 | 0.481829 | 0.415296 | **0.919298** | 0.435048 | **0.980229** |
| Quora | BM25 | 0.738212 | 0.695173 | 0.947661 | 0.735647 | 0.986610 |
|  | **PPLX dense** | **0.880790** | **0.851136** | 0.984258 | **0.874570** | 0.987865 |
|  | P2.1 | 0.844995 | 0.807626 | **0.988816** | 0.838923 | **0.997925** |
| SciDocs | BM25 | 0.150534 | 0.103194 | 0.348567 | 0.277201 | 0.561417 |
|  | **PPLX dense** | **0.225314** | **0.159333** | **0.484483** | **0.385142** | **0.753817** |
|  | P2.1 | 0.200236 | 0.142215 | 0.467100 | 0.348686 | 0.734433 |
| SciFact | BM25 | 0.663931 | 0.626833 | 0.882556 | 0.634838 | 0.965000 |
|  | PPLX dense | 0.714060 | **0.685142** | 0.898222 | **0.694454** | 0.918222 |
|  | **P2.1** | **0.723722** | 0.679675 | **0.956000** | 0.688295 | **0.993333** |
| TREC-COVID | BM25 | 0.571959 | 0.066611 | 0.100074 | 0.817222 | 0.362988 |
|  | **PPLX dense** | **0.833468** | **0.131466** | 0.157343 | **0.970000** | 0.498362 |
|  | P2.1 | 0.752852 | 0.131082 | **0.161646** | 0.930000 | **0.573787** |
| Webis-Touche2020 | **BM25** | **0.377718** | **0.237831** | **0.561303** | **0.612666** | **0.858916** |
|  | PPLX dense | 0.276631 | 0.166837 | 0.493851 | 0.517772 | 0.828912 |
|  | P2.1 | 0.289785 | 0.190253 | 0.517975 | 0.520760 | 0.853424 |
| **Macro** | BM25 | 0.374297 | 0.274610 | 0.562964 | 0.475641 | 0.735619 |
|  | **PPLX dense** | **0.544873** | **0.412036** | **0.670880** | **0.650711** | 0.810525 |
|  | **P2.1** | 0.490809 | 0.371455 | 0.666885 | 0.595950 | **0.839658** |

## 7. MTEB10 Retrieval 完整矩陣

這裡的 MTEB10 是專案固定的十個 English retrieval tasks，不代表完整
MTEB 全 task suite。評估規模為 8,815 queries、1,096,451 documents、
44,595 qrels。

| Task | Method | NDCG@10 | MAP@100 | Recall@100 | MRR@20 | CUB@1000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ArguAna | BM25 | 0.344115 | 0.237186 | 0.952891 | 0.233893 | 0.990007 |
|  | **PPLX dense** | **0.450704** | **0.319743** | 0.982156 | **0.318966** | 0.982156 |
|  | P2.1 | 0.415546 | 0.290486 | **0.989293** | 0.289107 | **1.000000** |
| CQADupStack Gaming | BM25 | 0.465550 | 0.430617 | 0.746695 | 0.457166 | 0.869162 |
|  | PPLX dense | 0.554296 | 0.513284 | 0.770804 | 0.545836 | 0.813355 |
|  | **P2.1** | **0.579577** | **0.535178** | **0.866450** | **0.561797** | **0.955603** |
| CQADupStack Unix | BM25 | 0.282784 | 0.259707 | 0.543694 | 0.284534 | 0.754300 |
|  | **PPLX dense** | **0.449146** | **0.410350** | 0.767735 | **0.440482** | 0.870142 |
|  | P2.1 | 0.433789 | 0.394026 | **0.777384** | 0.428277 | **0.936138** |
| ClimateFEVER Hard Negatives | BM25 | 0.141582 | 0.107780 | 0.411267 | 0.201400 | 0.712000 |
|  | **PPLX dense** | **0.377672** | **0.298098** | **0.675817** | **0.504430** | 0.801333 |
|  | P2.1 | 0.247970 | 0.194502 | 0.585350 | 0.338304 | **0.831250** |
| FEVER Hard Negatives | BM25 | 0.501070 | 0.452277 | 0.895438 | 0.466675 | 0.975288 |
|  | **PPLX dense** | **0.854417** | **0.829734** | 0.914605 | **0.864621** | 0.921224 |
|  | P2.1 | 0.785359 | 0.737200 | **0.966279** | 0.774984 | **0.988602** |
| FiQA2018 | BM25 | 0.231073 | 0.183888 | 0.510764 | 0.292591 | 0.725380 |
|  | **PPLX dense** | **0.518076** | **0.458882** | **0.833468** | **0.603800** | **0.954942** |
|  | P2.1 | 0.354177 | 0.295080 | 0.658711 | 0.434631 | 0.859093 |
| HotpotQA Hard Negatives | BM25 | 0.549783 | 0.466817 | 0.811000 | 0.719432 | 0.933000 |
|  | PPLX dense | **0.684454** | **0.615105** | 0.832000 | 0.816141 | 0.894000 |
|  | **P2.1** | 0.676280 | 0.591675 | **0.882000** | **0.832442** | **0.962000** |
| SCIDOCS | BM25 | 0.150694 | 0.103316 | 0.348367 | 0.277795 | 0.561617 |
|  | **PPLX dense** | **0.222704** | **0.157222** | **0.483100** | **0.378192** | **0.750733** |
|  | P2.1 | 0.200611 | 0.142496 | 0.467100 | 0.349516 | 0.734633 |
| TREC-COVID | BM25 | 0.572088 | 0.066617 | 0.100137 | 0.817222 | 0.363134 |
|  | PPLX dense | 0.724736 | 0.116822 | 0.151098 | 0.871667 | 0.497221 |
|  | **P2.1** | **0.758159** | **0.129599** | **0.160195** | **0.907500** | **0.568840** |
| Touche2020 v3 | BM25 | 0.596805 | 0.378977 | 0.633422 | 0.867347 | 0.889978 |
|  | **PPLX dense** | **0.605068** | 0.412771 | 0.661348 | **0.955782** | **0.925524** |
|  | P2.1 | 0.582929 | **0.430758** | **0.678484** | 0.872449 | 0.922945 |
| **Macro** | BM25 | 0.383554 | 0.268718 | 0.595367 | 0.461805 | 0.777387 |
|  | **PPLX dense** | **0.544127** | **0.413201** | **0.707213** | **0.629992** | 0.841063 |
|  | **P2.1** | 0.503440 | 0.374100 | 0.703125 | 0.578901 | **0.875910** |

## 8. 已測得的推理與查詢延遲

### 8.1 小型端到端原生 smoke

以下時間包含 tokenizer、ONNX inference、M1914 calibration、RMS scale、
lexical/semantic merge 與 native index lookup；不包含一次性 model load 與
artifact SHA 驗證。每個資料集先做 10 次 warmup。

| Dataset | Queries | Mean | P95 |
| --- | ---: | ---: | ---: |
| NFCorpus | 323 | 7.38 ms | 8.82 ms |
| SciFact | 300 | 10.22 ms | 12.44 ms |
| SciDocs | 1,000 | 15.48 ms | 17.50 ms |
| **合計** | **1,623** | **12.90 ms** | **17.16 ms** |

這些固定的端到端 CPU 數據顯示，小型評估面可達低十毫秒級執行時間；
不能直接外推到大型 corpus lookup，或當前套件在混合流量下的表現。

### 8.2 完整矩陣 query path

| 評估面 | Compile mean | Lookup mean | Total mean | Dataset-P95 mean |
| --- | ---: | ---: | ---: | ---: |
| BEIR15，dataset 等權 | 9.598 ms | 125.205 ms | 134.803 ms | 196.486 ms |
| BEIR15，query 加權 | 13.058 ms | 121.169 ms | 134.227 ms | 不適用 |
| MTEB10，task 等權 | 8.543 ms | 28.034 ms | 36.577 ms | 44.173 ms |
| MTEB10，query 加權 | 9.755 ms | 17.548 ms | 27.304 ms | 不適用 |

`Dataset-P95 mean` 是各 dataset/task P95 的平均，不是把所有 query 合併
後的 global P95。BEIR15 最快 NFCorpus 為 mean 9.304 ms、P95 11.152 ms；
最慢 MS MARCO 為 mean 397.130 ms、P95 471.326 ms。MTEB10 最快
SCIDOCS 為 mean 20.272 ms、P95 23.699 ms；最慢 Touche 為 mean
104.653 ms、P95 117.685 ms。

BEIR15 P2.1 共 131 shards，logical posting bytes 46,664,936,827，stored
bytes 34,320,714,805（31.96 GiB）。MTEB10 共 18 shards，stored bytes
2,912,242,585（2.71 GiB）。完整面上 lookup 而非 encoder compile 是主要
延遲來源。即使 query support 很小，高 document frequency（high-DF）語義
atoms 仍可能觸及大量 postings，因此 traversal 成本是擴展效能的改進重點。
這些 single-worker 測量值提供目前引擎按工作負載進行 profiling 的基線。

## 9. 發布定位

### 9.1 證據摘要

1. **候選召回接近 dense。** 兩個面 Recall@100 只落後 dense 約
   0.004，恢復 dense 相對 BM25 增益約 96.3%。
2. **候選池上限超過 dense。** CUB@1000 在 BEIR15 與 MTEB10 分別高
   0.0291 與 0.0348，表示 unified lexical/semantic postings 有實際互補性。
3. **不是局部單 row 信號。** P2.1 在 Recall 上分別勝 dense 11/15 與
   7/10；相對 BM25 則幾乎全面 row-level 改善。
4. **工程產品形態成立。** 單一 CPU encoder、單一 posting map、單一
   physical inverted index，沒有外部 ANN/RRF dependency。
5. **artifact 可稽核。** 模型 revision、ONNX SHA、ABI、atom parity、
   generation manifest 與 crash recovery 均已有固定證據。
6. **模型體積精簡。** 30.3M 參數明顯小於名義 0.6B PPLX dense
   encoder，可作為不依賴大型 dense encoder 的 CPU retrieval compiler。

### 9.2 適用角色

本次發布定位為：

- RAG first-stage candidate retriever；
- 同一 PostgreSQL 索引內 BM25 的語義擴展；
- 不希望部署大 dense encoder 和獨立 ANN 的單索引方案。

若下游需要最佳 top-10/head ranking，可在候選之後使用獨立的 reranker；
但應把 reranker 效果與 P2.1 的單索引 first-stage claim 分開報告。

## 10. 評估範圍與資料來源

報告矩陣主要是 single-worker 品質與延遲評估。其配置和測量值保持固定，
作為後續模型與引擎改進的參考。證據範圍如下：

1. BEIR15 為 15/15 完整本地 corpus；MTEB10 僅代表本專案固定 retrieval
   面，不代表全部 MTEB tasks。
2. BEIR dense 矩陣有 13/15 rows 沿用此前 serial VectorChord 結果，
   ArguAna 與 Climate-FEVER 為當前面直接生成；來源在最終 JSON 中逐 row
   標記。品質比較保留逐 row 來源；補齊統一的 PPLX revision、embedding hash
   與 VectorChord build manifest 是後續規劃中的證據改進。Stored-vector
   計時測量 lookup；端到端比較另需納入 PPLX encoder 推理。
3. MTEB10 的 VectorChord 缺失 rows 已修復，10/10 均為該次評估快照的
   single-worker 結果。
4. ArguAna 因 corpus 缺文檔，五條 qrels 被一致排除；不是 P2.1 特有處理。
5. BEIR NFCorpus 有 10/323 queries 回傳少於 1,000 candidates，最少 467；
   這是 candidate exhaustion，不是缺失 query。
6. MTEB prepare path 在 6,000 characters 截斷文字；被評估的 P2.1 模型
   token limit 為 512。這些結果未測量 P2.2 的完整文字視窗策略；長文檔
   評估是後續覆蓋範圍的改進方向。
7. b1.125 與 RMS policy 曾使用有限 BEIR rows 做全局選擇；現有 unseen
   transfer 與 MTEB 面支持這些評估面上的泛化。獨立 holdout 和更廣泛的
   校準將擴展目前 NFCorpus-bound vocabulary 與 calibration 的驗證範圍。

## 11. 未來方向

後續工作以現有候選召回成果為基礎，沿三個方向改進：

- **排序品質：** 提升 NDCG、MAP 與 MRR，向 dense 級頭部排序前進，以已測得
  約 0.04 至 0.055 的宏觀差距指引評估，同時保留候選召回能力。
- **規模與查詢效率：** 降低高 DF posting traversal 成本，改善讀寫併發及
  後台維護下的預熱查詢穩定性，建立面向具體工作負載的延遲 SLO 與資源基準。
- **泛化與證據：** 擴展跨領域校準、長文檔評估和當前 P2.2 的全矩陣覆蓋，
  完善對照資料的來源追蹤。

這些是後續迭代的改進目標。單索引設計仍是基礎；具體實驗與驗收條件統一
放在 [Model Planning](model-planning.md)。

## 12. 結論

Evoke 模型 Beta 1 提供原生單索引的 first-stage retriever。
在固定 P2.1 評估中，約 30.3M 參數 CPU sparse encoder 在兩個完整工程
評估面上幾乎保留 PPLX dense 的 Recall@100，並取得更高 CUB@1000；
相對 BM25 的品質改善也廣泛且一致。套件化後繼版本 P2.2 提供當前模型
契約，並保留 P2.1 結果作為有明確來源的評估基線。

本次發布在 PostgreSQL 內結合 lexical 與 semantic retrieval，不需要獨立
ANN 索引，為 RAG 檢索及上述排序品質、規模效率與評估覆蓋的持續改進，
提供精簡且可稽核的基礎。

## 13. 可重現資料

- [當前套件模型鎖定檔](../packaging/milestone-model.json)
- [P2 b1.125 產品化報告](research-sae/reports/m1900-m1999/ii42-p2-b1125-productization-report.md)
- [P2.1 BEIR15 完整矩陣](research-sae/reports/m1900-m1999/ii42-p2.1-beir15-native-full-matrix-report.md)
- [P2.1 MTEB10 完整矩陣](research-sae/reports/m1900-m1999/ii42-p2.1-mteb10-native-matrix-report.md)
- [MTEB10 VectorChord 修復報告](research-sae/reports/m1900-m1999/ii42-p2.1-mteb10-vectorchord-repair-report.md)
- [M1914 固定 support 校準報告](research-sae/reports/m1900-m1999/ii42-m1914-granite-fixed-support-calibration-report.md)
- [M1930b 單索引 additive closure](research-sae/reports/m1900-m1999/ii42-m1930b-one-index-additive-closure-report.md)
- [M1931 query-local calibration](research-sae/reports/m1900-m1999/ii42-m1931-query-local-source-calibration-report.md)
- [M1933 semantic budget frontier](research-sae/reports/m1900-m1999/ii42-m1933-semantic-budget-frontier-report.md)
- [M1934 unseen transfer](research-sae/reports/m1900-m1999/ii42-m1934-fixed-budget-unseen-transfer-report.md)
- BEIR15 JSON：`runs/evoke-p2-beir15-native-full-v1/evoke_p2_1_beir15_native_full_matrix.json`
- MTEB10 JSON：`runs/evoke-p2-mteb10-native-v1/evoke_p2_1_mteb10_native_matrix.json`
- ClearML M1914 task：`72d822802e104fa09a0e555c1a93533a`
