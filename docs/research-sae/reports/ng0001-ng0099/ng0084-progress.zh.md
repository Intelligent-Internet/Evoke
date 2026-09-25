# NG-0084：執行進度

日期：2026-09-14（EDT）；首次凍結與啟動為 2026-09-15 01:49 UTC 附近。
契約：[task-ranking execution](ng0084-task-ranking-execution.zh.md)。

## 本階段裁決：完成，但不准入正式兩臂

v3 在 **2026-09-15 02:29:53 UTC** 完整結束；384/384 題、21,463 pairs。exit0、error=null、owned process group closed、actual offline ClearML start/close 均通過。**執行成功不等於 teacher 品質准入成功。**

| 固定 TRAIN panel | PPLX nDCG@10 | RankT5 nDCG@10 | 差值 | PPLX / RankT5 all-positive Recall@10 |
|---|---:|---:|---:|---:|
| FEVER，128 題 | 0.900724 | 0.846553 | -0.054171 | 0.932292 / 0.941927 |
| HotpotQA，128 題 | 0.847274 | 0.885694 | +0.038420 | 0.870722 / 0.930990 |
| NQ，128 題 | 0.707094 | 0.745631 | +0.038537 | 0.851042 / 0.904253 |
| 等權 macro | **0.818364** | **0.825959** | **+0.007595** | **0.884685 / 0.925723** |

事前分層 paired query bootstrap 95% interval：**[-0.011516, +0.026883]**。點估計超过 0.005，但 lower 不大於零，故原 gate 判為 **`inconclusive_do_not_auto_launch`**。不是因為要求每域都贏而拒絕，也不是 negative/null 結果；沒有足夠把握認定該 teacher 在這個比較面提供 overall 增量。未挑有利領域、加樣到顯著、改 gate 或自動換 teacher。

這一輪尚無 optimizer updates、獨立 holdout 或原生成本測量，不是新 sparse 模型，不能拿 panel nDCG 與 NG80 的 full-corpus nDCG 相減。較長上下文與 qrel 完整性限制仍成立。

### 驗證與可重現證據

- Frozen worker 的 `review` 通過模型七檔 SHA、source/input SHA、終端 manifest、逐題 identity 與所有正例指標 replay。
- 另外以 [獨立實作 reviewer](../../../../scripts/review_ng84_teacher.py) 的 NumPy lexsort／向量化 DCG 重算全部分數，不 import worker 的 metrics/summarize；macro、bootstrap 與 gate 在 `1e-12` 內一致。這是第二種指標實作，不是第二名獨立研究者或獨立資料驗證。
- 8 個 frozen payload 和 **16 個終端產物**（含完整 scores、logs、ClearML offline bundle）在 Lambda2 與 Betty 鏡像完整驗證。`complete.json` SHA：`14aadf4a7f6abb7f18fac03cca61b62e6a38e62d120bd3377edb204e340c56a2`。
- Raw scores SHA：`f0e1a50462ad7440233a653658fe8dd1ef042fbd1717f5d0a48c36fe0c1d2233`；`NG-0084/review-v1/results.json` SHA：`d9f1e20683c60056fd24c7c8177be28c0aaf3ea79c9f44c28711cf398d47bce1`。
- 評分 1,723.51 秒；完整 worker／supervisor 1,764.34 秒，峰值 owned RSS **10.28 GiB**，Torch peak allocated VRAM 12.17 GiB。結束後四張 GPU 均 1 MiB／0%，NG84 tmux 已不存在，主機 available 117 GiB、disk free 609 GiB。沒有殘留本輪訓練或評分任務。
- 本地 CPU 相關測試 **47 passed**（包括新 reviewer 8 tests）；遠端 teacher tests **13 passed**。本輪未改 PostgreSQL 產品代碼，不把這些測試說成全產品回歸。

### 對下一階段的具體影響

1. **撤回「現成 RankT5 全面替代 PPLX」的自動訓練資格。** 保留成熟基座、完整 hybrid 目標、足夠 epochs 的設計；不因准入不確定而偷偷退回 alpha、mask、pair 權重循環。
2. **區分排序與覆蓋。** 三域 Recall@10 都上升，但 FEVER MRR 從 0.952474 降到 0.857259；增加 top-10 正例覆蓋不保證最前排更好。這不支持只以 Recall 或 teacher 規模決定蒸餾來源。
3. **不要誤判標註／文件身分差異。** 終點後人工閱讀四個最大 FEVER 降分案例：`55a15e8e091101ee7a85fa3a7a552bc4` 的指定正例是 United Kingdom（doc41938），RankT5 首位是未標註的 Monarchy of the United Kingdom（doc77234）；`10f50462e91beb722d286a6f2e5c0d8f` 中另一版 Lake Ontario（doc112617）超過已標註版本（doc141497）。這些是可追蹤的疑似替代證據／近重複例子，**不是新增人類 qrels、錯標率或 teacher 更正確的證明**。NG70/NG83 已記錄過此風險，不能再包裝成全新原因；本輪不重算「修正後分數」救 gate。
4. **新增 supervision 必須改變可學資訊，不只是 teacher 名稱。** 已實際準備 canonical FiQA TRAIN（下節），但正式開訓仍需 matched student-visible targets、舊曝光排除與新 validation manifest。若要採用互補 teacher 或任務感知監督，應另凍結對照，而不是從這三個分域結果事後選擇 teacher。缺少可信爭議判斷時，就明說尚未取得，不用模型自評代替人類真值。

本次在這個真正完成的階段裁決處停下匯報。第二階段未啟動；原定時輪詢保持停用。這不是 sparse-hybrid 路線無解的證明，也不是已完成充分訓練後的失敗結論。

## 已完成

- 建立兩臂共同底座／資料／完整 hybrid 目標，唯一主變量為 PPLX 與任務排序 teacher 的資訊。正式梯度訓練尚未啟動，不把 teacher 准入當作模型進步。
- 凍結 NG80 fit-only 384 題，每域 128；NG83 bank 共 21,463 pairs、14,502 個 distinct documents、696 個 known-positive memberships。無 teacher-correct-only 篩選，無 locked-test access。
- RankT5 的七檔 SHA 與 M1701 歸檔逐一相同；其確切上游是 `Soyoung97/RankT5-3b`，不是 NAVER encoder-only reproduction。固定 revision 見協議與 `scripts/ng84-rankt5-sha256.json`。
- 初始 Mac -> Lambda2 複製吞吐低，已停止本輪擁有的 rsync，改為 Lambda2 直接下載同一固定 revision 並通過完整 SHA 驗證。原歸檔未動；中止複製不是模型／實驗失敗。
- 最初本地 CPU 相關測試 **32 passed**；Lambda2 **12 passed**。後續最終測試數見上節。`git diff --check`、Python AST 與新協議本地連結檢查通過。
- v1 凍結 input SHA：`4edc627319caac8f1bcb806aa3a69d2e8916f3b8958ee0e82087469b48387c99`。七個 frozen source/data/config payload 已鏡像 Betty 並逐一驗證，零 mismatch；此為初始 attempt，不是當前結果 manifest。

## 執行現場

- 主機 Lambda2，經 Spark1 SSH jump；原 tmux `ng84_teacher_v3`，僅使用 GPU3，現已結束釋放。
- 已閉合 run：`NG-0084/teacher-admission-v3`；stage `teacher`。
- Spark1／Spark2 當時 available RAM 僅 9.4／3.8 GiB，沒有在其上載入 teacher。Lambda2 原 GPU0–2 的其他任務後來自行結束；本輪沒有終止或修改它們，也沒有因此擴大到多 GPU。
- 主 Python：既有 `research-runtime/ng9-py312`，Torch `2.9.1+cu128`、Transformers `5.15.1`、ClearML `2.0.2`。SentencePiece `0.2.1`、Protobuf `6.33.6` 裝在本輪隔離 overlay，未改共享 Python 套件。
- ClearML API ping 可達 HTTP200，但實際 SDK online Task.init 回傳 `MissingConfigError`。本輪明示 offline tracking；不宣稱在線 dashboard 已同步。
- 三小時 hard cap、16-query 時間 canary、RAM／VRAM／disk floor、GPU lock 和 owned-process-group 清理均在 supervisor 內。不恢復先前已停的定時輪詢。

## 環境修正與保留的 attempt

v1 在 tokenizer 載入時發現缺 Protobuf，錯誤回退到 TikToken extractor 並失敗，尚無任何 query score 或 optimizer update。主動停止該 controller，`owned_group_closed=true`，8 個終端產物已完整鏡像並驗證，`complete.passed=false` 保留，不修改成通過。

v2 補齊固定依賴並增加 fail-fast 與原 SentencePiece token ID parity。資料、模型、384 題抽樣、分數與資源/scientific config 均與 v1 相同，未改准入 gate。v2 input SHA：`e1c4fb4f406573f73806f1ed4f78f4fe12b88b84dcdfcb94c8cf55bf3a5a6e8c`；8 個 frozen payload 已鏡像並驗證，零 mismatch。本地相關 32 tests 與遠端 12 tests 再次通過。v2 tokenizer parity 兩個固定 probes 已通過。

v2 完成模型載入後，記錄 Transformers loading report 時遇到 `set` 無法 JSON 序列化。尚無 query score；terminal `owned_group_closed=true`，9 個終端檔案已鏡像並驗證，失敗原樣保留。v3 只將 loading issue sets 轉為經零錯誤檢查的計數，補充集合序列化／非空錯誤拒絕測試，不改實驗樣本或科學設定。最新本地相關測試 **33 passed**，遠端本輪 **13 passed**。

v3 input SHA：`8b4d9d147cfdf5ca20636249c837f65dbb2de4bbad2e0efaba4723c4b905328e`。這三個編號是工程執行 attempts，不是三個不同科學 arms，也不是三輪模型失敗。

v3 的 8 個 frozen payload 已鏡像驗證，與 v1 的資料／模型／科學設定完全相同。已通過 tokenizer parity、checkpoint 完整載入檢查及第一批 forward-vs-generate score parity（max absolute error 0）。早期現場樣本為 9/384 queries、511 pairs／42 秒，推算全量約 30 分鐘；此為吞吐估計，不是終點或品質結論。GPU3 為 13,144 MiB、98% utilization；主機 available 116 GiB。模型實際輸入依然固定截斷至 512，log 的長序列 warning 來自另外統計未截斷 token 長度，不是把長序列送進模型。

16-query 時間 canary 已通過：含 1.5x 餘量與固定緩衝的推算為 3,333.51 秒，低於固定 10,200 秒門檻。後續 104/384 queries、5,854 pairs／469.46 秒仍維持約 29 分鐘全量推算。沒有用任何途中品質值修改篩選或 gate。

進一步核對 frozen `NG-0069/evaluation-v2/ng69_pipeline.py` 的 dense `perform`：query max length 64、document 256。RankT5 的 512 是 query+document 合併後長度，tokenizer 也不同。故本次准入只衡量整個現成 teacher 管線是否有增量；它不能隔離 cross-encoder 能力與較長可見文本的影響。此限制已在原協議保留：正式 A/B 必須以相同 student-visible text 準備兩套 targets，不能直接把當前分数當作已完成 matched-context distillation。

## 同時完成的多來源資料準備

不是重開 competitor coverage 診斷，而是落實下一輪新訓練資料來源。

實際讀取本地 RLHN 的 11 個 parquet shards 後，648,766 rows 包含 FiQA 5,496、FEVER 28,869、HotpotQA 84,395、NQ 57,539、MS MARCO passage 455,748、ArguAna 4,065、SciDocsRR 12,654。原 dataset card 的近似數字不是本次 manifest 計數，也不能把所有來源自動准入訓練。

**5,496 條 RLHN FiQA 全部在 NFC／空白／casefold 後唯一對上原始 FiQA TRAIN 問題**。這證明這批問題的 public TRAIN 來源，不等於原 RLHN relabel 是人類真值，也不證明全部從未在本項目曝光。因而新 preparer 直接使用 canonical qrels，不沿用 relabel positives/negatives。

- [BeIR FiQA qrels](https://huggingface.co/datasets/BeIR/fiqa-qrels) revision `252958f2d646e22cab6d0c72dd3f0d5de6d0655a`，只下載 `train.tsv` 與 README，沒有下載 dev/test qrels。
- [BeIR FiQA corpus/queries](https://huggingface.co/datasets/BeIR/fiqa) revision `979c07a7cb5ccc6ca009792241fa1250b98055dd`，query 表先按 TRAIN IDs 在 Arrow 過濾，才轉成 Python rows。
- 產生 `NG-0084/fiqa-train-v1`：**5,500 canonical TRAIN queries、57,638 documents、14,166 known-positive memberships**；全部正例都能找到原 corpus 文件，沒有捏造 judged negatives。
- 原 qrels／query／corpus SHA 固定於 `scripts/ng84_prepare_fiqa.py`；manifest 保存來源 revision、11 個 RLHN shard SHA、輸出 SHA 與曝光限制，原始大檔保存在 Betty，不進 Git。
- FiQA manifest SHA `6d1da51fbc148930b97907abdb79c3f8324b5a07e1eb679cec15e3c7a73dc499`；兩個 materialized payload 已在 Betty 與 Lambda2 各自 SHA 驗證。
- 人類正例、relabel、teacher scores 三者分開。dataset card 宣告 CC-BY-SA4.0，但衍生發布仍需保留來源／授權與 teacher 使用邊界，不把這份準備升格為法律清算。
- 本地相關測試已增至 **39 passed**。FiQA preparer 不載入 Torch，不做 query inference，不啟動模型訓練。

這是現成的非 Wikipedia 訓練來源，不是另一個 sparse 模型成果。正式兩臂尚須凍結 FiQA 抽樣、全局舊曝光／近重複排除、student-visible teacher targets 與完整 candidate/corpus 比較；還沒有新的 independent holdout，不能將 FiQA 或舊 1,536 題直接改名成「未曝光泛化測試」。

Teacher parity／時間 canary／384 題終點／完整原始分數 replay 現均完成，結果與停止決策見本文首節。多來源正式 manifest、student-visible target、兩臂訓練、第二 seed、獨立來源評估與真實成本驗證仍未完成；不把資料準備或這次 teacher 准入當作模型成果。
