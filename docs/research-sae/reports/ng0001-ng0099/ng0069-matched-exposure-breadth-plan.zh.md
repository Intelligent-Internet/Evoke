# NG-0069：相同題次，區分訓練廣度與重複擬合

協議版本 v1，2026-09-12。在查看 NG67 新 DEV 模型表現之前凍結。這是研究協議，不改產品，不宣稱新模型已勝過 dense。

## 假說與唯一主要變量

[NG66](ng0066-training-duration-review.zh.md) 的 TRAIN 收益與 DEV 弱改善，以及 [NG68](ng0068-supervision-and-ranking-review.zh.md) 的監督診斷，支持先比較「更多不同 TRAIN 題目」而不是第五到第九輪相同小集合訓練。

| 路線 | 不同 TRAIN 題目 | 每題曝光 | 總 query exposures | AdamW updates |
|---|---:|---:|---:|---:|
| A：reuse NG66 epoch 4 | 1,536 | 4 | 6,144 | 1,536 |
| B：breadth | 6,144 | 1 | 6,144 | 1,536 |

B 保留原有 1,536 題，從 NG67 已凍結的新 TRAIN 按 `SHA256('NG69-breadth-v1:' + query_id)` 每域選 1,536 題，共新增 4,608 題。三域總數各 2,048。選題不用分數、不讀 DEV/LOCKED_TEST query，不加入尚未釐清來源的額外標籤。

A 與 B 的更新數、query exposures 相同，不代表 token/pair 數或精確 FLOPs 相同；記錄实际 token 長度、候選數、GPU/host 時間與 RSS，報告這些殘留差異。A 已有三個順序 seed，不重跑健康完成的 A。

## 固定訓練契約

沿用 NG59/66 的 49,648,089 parameter 初始模型、full nonnegative readout、BM25 0.1 + semantic 0.9/RMS 的同一 scoring 實作、固定 lexical vocabulary/normalization、PPLX 教師、`0.5 gold + 0.5 softmax(teacher/0.04)` target、cross entropy、query64/document256、FP32/noTF32、AdamW、學習率、weight decay、梯度裁切及每次更新四題。query/doc 都透過既有 replay/VJP 更新共享 encoder。

B 順序 seeds 為 59059、66061、66067；它們是同一初始化的順序變動，不冒稱三個獨立初始化。每個 seed 在 384、768、1,152、1,536 updates 保存模型／optimizer 與復原測試，不用中途 DEV 選擇最好 checkpoint。主要比較最後一步，所有結果都報告。

不在本輪同時加 DF penalty、mask、換 teacher、改 loss、混入 MS MARCO 或新人工標註。這些需要另外編號與對照，否則無法歸因「廣度」的效果。

## 教師準備階段

`scripts/prepare_ng69_breadth.py` 是此階段可執行 source。它只準備 selected TRAIN 的固定 pool targets，不是 SGD 訓練，也不產生新 DEV/LOCKED_TEST query embedding。

重用 NG59 已審計的 59,111 document embeddings 和 1,536 TRAIN query embeddings。只編碼 B 新選題池中尚未編碼的文檔及新增 4,608 個 TRAIN queries。`document-ids.npy` 明確對照 NG67 corpus index；沒有編碼的文檔不能補零冒充有效 embedding。此階段不是完整 233,009 corpus 檢索評估。

凍結 source/protocol/input SHA。PPLX 載入必須無 missing/unexpected/mismatched keys，執行的 custom modeling source SHA 必須等於已驗證模型檔。參數 SHA 等於 NG59 教師；重算前 64 個舊文檔及四個舊 query，事前容差 `rtol=1e-4, atol=2e-5`。舊 targets 原樣保留。新 query、pool 與全部正例 ID 必須一致。

執行前取得 GPU0 cooperative lock、確認 GPU0 空閒，不碰別人的 GPUs。CPU4、host process-tree 16 GiB、GPU allocated 20 GiB、單階段 90 分鐘；以新文檔分散樣本和新 query canary 預估成本，不通過就保存失敗而不是放寬門檻重跑。ClearML 如目前 endpoint 無可用 authentication，明確記錄 actual-start offline task，不冒稱 online synced。成功需 exit0、完整 SHA、closed tracking 和後续独立 target 重算。

## 完整評估與防止污染

正式 A/B 評估前，實作並測試以下契約；教師準備完成本身不能被當成 A/B 訓練完成：

- 全部模型與 dense/BM25 基準在同一 NG67 **233,009 文檔**背景重新評估；不能拿舊 59,111 背景的 dense 數字直接比較。
- 使用 NG67 新 DEV 共 1,536 題，三域各 512。此 cohort 是模型選擇用 DEV，不再稱 unseen test；旧 DEV 與新 DEV 分開列示。LOCKED_TEST 的 1,536 題仍不編碼、不評分、不挑參數。
- 不因看到某域結果就調 lexical mixing、RMS、vocabulary 或長度再重算同一預註冊結果。若 OOV 或 token 可見範圍確有問題，先報告並另開實驗。
- 全正例 nDCG@10、Recall@100、掉出／進入 top-100、逐題 rank harm 全部保留。跨 seed 先按 query 平均，再做分域 paired bootstrap；同一 query 不算三個獨立樣本。
- 主要門檻為 B-A 的新 DEV macro nDCG@10 改善且 95% paired interval 不含零，同時 macro Recall@100 不下降超過 0.005、任何單域 nDCG@10 不下降超過 0.01。單 seed 只供進度，不選赢家。門檻是下一階段投資判準，不是統計保證。
- 如 B 有穩健改善，再安排更大資料／額外監督分支；如沒有，回到 NG68 的可信標籤、CE pair discrimination 與截斷診斷，不把「再訓練久一點」作為默認答案。
- 品質與成本分開判定；CSR bytes、NNZ、DF work proxy 先報，最終還需 native postings visited、index bytes、warm p50/p95 latency。超過 dense 的品質不能被成本好看取代，反之亦然。

## 執行順序

1. NG68 診斷已完成，人工複核只準備樣本，尚無新人工 labels。
2. 執行 B 的資料選取與教師準備，先獨立驗證 targets 與舊模型重用契約。
3. 在主倉庫完成 lexical 擴展、B 訓練、完整背景 A/B evaluation 的可重現實作與測試，再啟動；不能用臨時 monkey patch 改 NG59/66 的 frozen 常數。
4. 與此同時可做資料來源調查，恢復 original-vs-RLHN pair provenance；不得接觸 locked-test ranking，或自動聯絡／付費委託人工標註。

```bash
python scripts/prepare_ng69_breadth.py \
    --research-root "$RESEARCH_ROOT" --run "$NG69_TEACHER_RUN" --mode freeze
python scripts/prepare_ng69_breadth.py \
    --research-root "$RESEARCH_ROOT" --run "$NG69_TEACHER_RUN" --mode supervise
```

將本協議的固定副本放在 run 的 `protocol.md`。`freeze` 在可信本地輸入上產生 manifest，連同固定 script/protocol 複製到計算節點；`supervise` 重新核對所有輸入才啟動。失敗、部分輸出與來源均保留，不自動覆寫重試。執行狀態記錄在外部 run，不改寫本協議內容以追認結果。
