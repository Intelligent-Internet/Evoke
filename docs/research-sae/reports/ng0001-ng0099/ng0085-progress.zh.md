# NG-0085 進度與執行紀錄

更新：2026-09-17。科學設定見 [固定 student A/B 協議](ng0085-student-ab.zh.md)。`student-ab-v4` 教師評分已完成，student 因數值異常停止；新的 [有界 loss 診斷](ng0085-loss-diagnosis.zh.md) 保留原 attempt，先定位原因。沒有完整 student 品質結果、production 部署或 locked-test inference，舊自動輪詢沒有恢復。下文 v2/v3/v4 啟動狀態均為歷史快照。

## 2026-09-17：教師完成，Student 數值失穩

四個 RankT5 shards 的完整 inventory、SHA、exit 及 complete receipt 已核對通過：合計 16,384 fit queries、993,661 pairs，各 shard 實際評分約 3.78--3.85 小時，均在原八小時預算內。GPU 3 / 0 自動分配各兩片，沒有驅逐其他任務。先前的排队及 teacher padding 瓶頸已跨過，不把這個執行成果當作模型品質改善。

B/85001 最後成功落盤 step 2,839，下一次 update 在 query encode 觸發 `invalid fixed RMS normalization`，於 2026-09-17 08:51:12 UTC exit 1。A/85001 最後落盤 step 2,832，於 08:52:28 UTC 被 matched-stage stop request 關閉。兩個 owned groups 均已退出；seed 85002 與 full-corpus endpoint 尚未開始。A/B 的 checkpoint 1,024 / 2,048 和原始失敗證據仍保留。

診斷先區分 raw query 全零、RMS support 脫節、非有限值及 loss 的尺度梯度。原 loss / optimizer / order 不變，不直接改 clamp 或重啟整轮。

固定五 checkpoint 的 candidate profile 已完成：B/2048 在 64 題曝光 validation 的 nDCG@10 從底座 0.72538 降至 0.66351，不能宣稱訓練成功。嚴格重放前 740 updates 的記錄統計完全一致，但第 2789 步因小幅 gradient-norm 差異被 trace gate 停止；原失敗尚未精確重現。v1 原樣保留，v2 改為有界觀測 trace 差異並保存失效狀態，loss 仍不變。具體數據與介入准入限制見 [診斷報告](ng0085-loss-diagnosis.zh.md)。

## v2 停止與 v3 資源重分配

2026-09-16 21:43 UTC 現場核對：v2 在 08:52:51 UTC 因 `GPU occupied; never evict another job` 停止，並非持續訓練中。RankT5 shard 1 沒有建立 phase，其他三個 shard 被 matched-stage fail-fast 終止；四個 student 目錄均不存在。當時沒有保存 GPU 准入快照，所以不能事後斷言是瞬時活動還是真實資源競爭。

已完成的三個 phase 重新檢查全部 payload inventory、SHA、exit 及 complete receipt：

| 階段 | 已驗證結果 |
|---|---|
| prepare | 16,384 fit、2,048 validation、290,609 文件；381.39 秒 |
| dense | 共同文字 query/document 的 1,024 維 PPLX 編碼；6,302.92 秒 |
| bank | 18,432 queries、1,141,279 pairs，保留 known positives；597.13 秒 |

使用者批准重新分配資源後，新建 `student-ab-v3`，保留 v2 不變。三個 completed phase 以獨立副本沿用並再次逐項驗證，不重算 dense 或 candidates，不複製失敗的 RankT5 phase。資料、targets、trainer、protocol、模型依賴和所有科學設定均直接繼承 v2；只替換排程控制器並新增 queue。`continuation.json` 保存父 inputs SHA、各 reused complete SHA 與本次排程限制。

新排程只處理這一個固定 campaign：

- 優先順序 GPU 3、0、1、2，最多同時兩個獨立單卡工作；不驅逐其他任務，不使用 DDP。啟動時 0/1/2 正在執行其他 AIIM 工作，GPU 3 為 1 MiB / 0% 且沒有 compute process。
- 每 60 秒檢查隊列；GPU 准入同時要求原 memory/utilization 條件、無 compute PID 及 cooperative lock。實際准入另存 GPU UUID/PID/記憶體/利用率和 host/disk 快照。
- 僅「worker 尚未啟動、phase 目錄未建立」的暫忙可重新排隊。任何已開始的計算失敗、資料或數值不一致仍停止並保留 attempt，不能用 retry 掩蓋。
- 原每 teacher shard 8 小時、每 student 12 小時和記憶體/磁碟上限不變。每 stage 完全無 active 工作的等待上限 12 小時，整個串行化 campaign 上限 96 小時；不自動擴研究範圍。Student 排程順序改為同 seed 的 A/B 相鄰，兩臂各自的凍結 training order 不變。
- Mac CPU 回歸 `20 passed`；Lambda2 含實際 PyTorch score-gradient 的 NG85 回歸 `23 passed`，`ruff check`、`git diff --check` 通過。這些不代表 full-size training canary 或 endpoint 已通過。

v3 第一個 `rankt5-0` 已於 22:10 UTC 通過 GPU 3 准入並啟動，其他三個 shard 排隊；tmux 為 `ng85_student_ab_v3`。全部 teacher shard 封存通過後，會依相同排程執行固定四個 student run，再停在 full-background evaluation pending。沒有啟用舊 heartbeat。以下「當前執行鏈」段落保留的是 v2 啟動時紀錄，不代表最新狀態。

### v3 的實際吞吐停止與有界批次對照

v3 啟動後通過 tokenizer / forward-generate parity（誤差 0），但在第 32 題觸發 `RankT5 shard cannot fit predeclared eight-hour budget`。Phase 於 212.59 秒退出，owned group 已關閉，沒有 student updates。這是第二個獨立執行限制，不是原先 GPU 准入問題，也不是模型品質失敗。排程與計算失敗仍分開處理。

`teacher-batch-probe-v1` 在同一 GPU、FP32、同一前 32 個 fit queries / 1,849 pairs 對照原 batch 2 和 batch 8，外部 watchdog 上限 900 秒。兩者分數在 `rtol=1e-5, atol=1e-5` 內一致，最大絕對差 `1.04904e-5`；但 142.05 秒變 148.95 秒，batch 8 反而慢約 4.9%。加上原 1.5 倍 safety factor 與 600 秒固定餘量後，投影分別為 29,209.59 / 30,597.63 秒，均超過 28,800 秒，所以沒有採用或放寬 gate。

隨後針對 padding 做 CPU 計數：相同原始有效 tokens 342,942，batch 2 padded tokens 407,797、未分組 batch 8 為 507,413；同 query 內按實際 RankT5 token 長度分組的 batch 8 為 373,563。這是計算工作量診斷，不冒充實測速度。`teacher-batch-probe-v2` 因此只驗證 length-grouped batch 8，沒有掃其他 batch/precision，也沒有改資料或 supervision。分數必須還原原 candidate 順序並重新通過 parity、原時間 gate 才允許建立 v4；v3 和未分組 batch 8 的負結果完整保留。

## v4：有證據的正式續跑

第二次 probe 已以 exit 0 完成：同一 32 題 / 1,849 pairs，batch 2 用 142.10 秒，length-grouped batch 8 用 110.01 秒，耗時下降 22.58%。最大絕對分數差仍為 `1.04904e-5`，通過原 `rtol=1e-5, atol=1e-5`；不是宣稱 bitwise identical。峰值 CUDA allocation 約 12.70 GiB。Batch 8 的原始 shard 耗時投影為 14,771.18 秒（約 4.10 小時），按原公式加入安全餘量為 22,756.77 秒（約 6.32 小時），小於原 28,800 秒上限。這是固定 TRAIN canary 的執行測量，不是排序或 retrieval latency 收益。

因此建立獨立 `student-ab-v4`，沒有修改 v2/v3。只在 teacher 執行層將 query 內的候選按 RankT5 token 長度穩定分組，以 batch 8 計算並回填到原 candidate 位置。模型、精度、文字、候選、targets 定義、student loss、兩 epochs、兩 seeds 和 wall-time/memory gates 不變；沒有 truncation、混合精度、刪除候選或放寬預算。Teacher 正式 phase 的前 32 題仍重新接受原時間 gate，新增 `canary.json` 記錄，不依 probe 直接跳過。

建立 v4 前，controller 重新驗證 probe 原始分數及來源 SHA；probe input/results、兩份 score arrays、測試的 teacher source 都加入 frozen dependencies。排程保持 GPU 3 優先、最多兩個單卡工作、每 60 秒檢查，不驅逐其他工作；重複的 pre-start resource-busy 也計入 12 小時等待上限。完成四個 teacher shard 後才依 seed 相鄰 A/B 順序執行四個 student，固定終點後停止等待評估。

最後回歸：Mac NG85/NG84 相關 CPU 測試 `49 passed`，Lambda2 NG85（包含真實 PyTorch score-gradient）`25 passed`，`ruff check` 及 `git diff --check` 通過。RankT5 批次變更的 parity 和候選位置還原都有獨立檢查。這仍不代表 full-size student canary、兩 epochs 或 endpoint 已完成。

正式 tmux：`ng85_student_ab_v4`。大檔延續三份已驗證 prerequisite，不重算 PPLX/bank。按單卡 canary 粗估 teacher 工作約 16 小時，空閒第二卡可縮短排隊時間；長文本和共享主機負載可能改變速度，不作完成時間保證。Student 用時須等其自身 canary，不能把 teacher ETA 當作整輪完成時間。

實際正式 `rankt5-0` 已通過前 32 題 canary：1,849 pairs / 110.22 秒，`guarded_seconds=22798.63 < 28800`，`passed=true`；forward/generate parity 誤差仍為 0。這驗證了正式 pipeline 已越過 v3 的停止點，不只是獨立 probe 較快。當前在 GPU 3 持續 teacher scoring，其餘 shard 排隊；尚未進入 student fit。

## 已完成

- 新增獨立 NG85 的資料、targets、訓練及 bounded controller，沿用 NG71 的成熟 sparse readout、exact VJP replay、完整 optimizer moments 保存與重載。
- PPLX / RankT5 的主要差異僅為同一候選上的 teacher 分布；共同文字、hybrid score、訓練 order、優化器、曝光及成本處理均凍結。
- Lambda2 可用四張 TITAN RTX；啟動檢查時均為 1 MiB / 0%。Spark-1 / Spark-2 可用主機記憶體約 9.5 / 4.9 GiB，沒有為這次工作驅逐其既有程序，故改用 Lambda2。
- Mac 相關資料／契約回歸 `35 passed`；Linux 上 NG85 真實 PyTorch score gradient、共同 BM25 代數、資料隔離、固定 exposure 與檔案完整性測試 `11 passed`。`ruff check` 及 `git diff --check` 通過。這不代表真實大型 student VJP canary 已通過，它會在 fit 開始前另外執行。

## 第一個 Attempt 的資料問題

`student-ab-v1` 在 286 秒時因 canonical FiQA 空白 document 被 fail-fast 攔截，owned process group 已關閉，peak owned RSS 約 1.30 GiB，尚無 teacher inference 或 student update。原輸入、已生成資料、exit 及 ClearML offline bundle 均保留。

核對完整 FiQA TRAIN 來源後確認 38 篇空白文件，35 個 positive memberships 分屬 34 個問題。這不是由 teacher 得分或 validation 輸贏決定的排除，也不是把原始 qrels 改標成負例。

`student-ab-v2` 在推理前隔離這些不可用證據及受影響問題，從其餘 5,466 題按原 hash 順序補足 4,096 fit + 512 validation。全部排除 ID 記在 `prepare/fiqa-integrity.json`，原 canonical 資料不修改。新固定背景是 290,609 篇有文字文件；這個研究子集的結果不能冒充完整 FiQA 或完整 BEIR 三域。

## v2 啟動時執行鏈（歷史快照）

Lambda2 tmux：`ng85_student_ab_v2`。順序為共同文字準備、PPLX 重新編碼、共同 bank，然後四個獨立單卡 RankT5 target shards；全部封存通過後才會啟動 A/B 各兩個種子的 student 訓練。

每個 student 的正式預算為 16,384 個唯一 fit query、兩完整 epochs、8,192 updates。Canary 的 16 updates 會丟棄並重新由 NG3 初始化，不能把 canary 混算成正式曝光。單 phase 失敗停止整個 matched stage，沒有無限重試、權重掃描或自动新增下一輪。

v2 preparation 已完整通過：16,384 個唯一 fit、2,048 個唯一 validation、跨 split 的可見文字正規化重複為 0，文件 290,609 篇；用時 381.39 秒，exit 0，owned group 已關閉，峰值 RSS 約 1.30 GiB。全部 preparation payload 已鏡像到 Betty 並按 complete manifest 驗證。

GPU0 准入檢查曾因瞬時活動攔截，沒有開始 dense worker、沒有驅逐程序。再次核對四卡均 1 MiB / 0%、沒有 GPU compute process 後，人工恢復同一 frozen campaign，跳過已封存 preparation，不重算資料、不放寬 guard。現在已確認 GPU0 的 dense worker 啟動，進行同可見文字的 PPLX 重新編碼；尚未有 student 品質結果。

ClearML 使用實際 offline task start/close，沒有宣稱已同步伺服器。Trainer 終點之後仍須完整固定背景的 parent / hybrid / dense / dense+BM25 排名、兩 seed 比較和成本檢驗；目前不能宣稱超過 dense 或成本下降。

## 評估補充：不能削弱 Dense 對照

在任何 teacher targets 或 student 終點結果產生前，另行明確凍結以下評估界線，不修改正在執行的 v2 training inputs：共同文字的 A/B 是 teacher 可學性因果對照，不足以證明超過正常 dense。除了 matched-context 表，還必須保留原始未經共同前綴裁切的文本，按各原始模型正常 token budget 執行 parent/student/PPLX，BM25 使用完整原始文本；同一固定 corpus/qrels 上比較 hybrid、pure dense、dense+BM25 的正常輸入路徑。若只贏過共同裁切版本、沒有贏過未弱化的正常 dense，不列為產品品質成功。兩種輸入情境分表，不混用分數或成本。

## 指紋與追溯

| 產物 | SHA256 |
|---|---|
| v1 frozen inputs | `f0044f55d0d3d8b8bb65f7c1c6bc37bdc425a72e239155ecb3e8683d2741afbd` |
| v2 frozen inputs | `70009935fca50ac7df3d311af23fc61d9578d0662816e05e0a7b9bc0f2a8327d` |
| v2 completed preparation | `d5d5a540e2f5bd81dbdb3315433c42aaae4840836b1446354899955de851262d` |
| v2 completed dense | `ffdf6392c276944297ef2d2ffdf4dd955861727e9a1d70503c07f97fd6f6a02e` |
| v2 completed bank | `275da8b5791ddf5aa8b77a4f15b817ef1b2c491e08e926ab3b003dbcd8f8fdf8` |
| v3 frozen inputs | `3b22af5f2bad7e3ac423053fb4ac97026525bcfb17fa78c319c04717cd3554e1` |
| v3 continuation receipt | `a7a452de8849e439a4fe1e537d24b8b7cad40b87ff86608c83d79f09863e0fcb` |
| v4 frozen inputs | `5b5f7fa82c7e449e6e01f5309cf833919341fa20a69e3d9a74bd9c0c5ce50387` |
| v4 continuation receipt | `abe45d95e189ab07a9273e9fcc533af9627493230827cea888a387378ae6b1fe` |
| length-grouped batch probe result | `723f05992ac8626ff35ba041e53bc000becc2f2460c14190d553833920226471` |

兩份 frozen source/dependency manifest 均已在 Mac 與 Lambda2 校驗。大型產物位於機器本地 NG-0085 資料單元，路徑和保留規則記在本地 catalog，不寫進產品部署設定。NG84 inconclusive 裁決、原 corpus 與其他研究工作樹均未覆蓋。

v3 的 frozen source、inputs/continuation 和 Linux 測試回執已鏡像到 Betty；inputs/continuation SHA 與遠端一致。沿用的大型 dense/bank 副本目前已在 Lambda2 驗證，不宣稱已完成 Betty 大型產物備份。

v4 的 17 份 frozen source/metadata 及六份 batch-probe dependency 已在 Betty 逐項 SHA 驗證。兩次 batch probe、v3 的 11 份失敗 phase payload 及 v4 Linux 測試回執也已保存；v4 scoring 是進行中產物，只有狀態/canary 快照，不把它列成完整 teacher 備份。
