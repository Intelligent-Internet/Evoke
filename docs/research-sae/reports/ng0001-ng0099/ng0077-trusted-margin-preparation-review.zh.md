# NG-0077：可信 Margin 準備完成，頭部覆蓋不是這一輪的缺口

2026-09-13 UTC。接續[設計方案](ng0077-trusted-margin-retention-plan.zh.md)與[NG76 原始中點](ng0076-authentic-midpoint-review.zh.md)。本輪完成原始 TRAIN anchor 審核、穩定 loss 實作、獨立數值／來源核對及 lambda2 完整副本校驗。**尚未訓練新模型，不能宣稱排序、召回或成本已有改善。**

## 1. 新證據與取捨

使用原 NG71 的 384 個 TRAIN_PILOT queries，三域各 128 題，保留全部 660 個 query-positive 關係和原始 D forward pool。沒有用 TRAIN_SENTINEL、24 題歷史 probes、DEV 或 LOCKED_TEST 選 anchor。初始模型、PPLX cache、完整語料排名及文檔身份均核對原封存來源；對 pool 外的初始 top100，只讀既有 TRAIN query/document 教師向量做 CPU 點積，不新增 encoder inference，也不加入訓練。

| TRAIN 域 | 全部 positive／有 anchor | 可信 anchors | 初始 top10 可信比較覆蓋 | rank11–100 可信比較覆蓋 |
| --- | ---: | ---: | ---: | ---: |
| FEVER | 165／165 | 11,625 | 1,076／1,076 | 4,901／13,198，37.1% |
| HotpotQA | 260／260 | 18,242 | 1,614／1,614 | 8,592／20,976，41.0% |
| NQ | 235／235 | 15,308 | 994／994 | 8,129／19,451，41.8% |

共審核 85,417 個 gold/rival 關係，得到 45,175 個可直接重用的 anchors，384 題均有 anchor。**所有初始 top10 中可按既定規則保留的可信比較，已在原 D pool 內**。因此第一個干預不需要擴候選、增加文檔曝光或換 pool，只改 loss 即可測試。這不保證初始 top10 外的文檔日後不會升入頭部；rank11–100 明確有缺口，outside100 的審核也只是原 witness 子集，不能把該子集的 100% 當成全庫覆蓋。

全部 anchors 原來亦已通過 D 的 supervision eligibility，不是這次才找到的未見訓練對。另一個值得注意的現象：初始 D 的直接 score-space pressure 會縮小 FEVER 424、HotpotQA 272、NQ 892 個可信正 margin；其中 rival 在 top10 的分別是 22、20、40 個。原因可以是基座的 student preference 比 soft teacher target 更強，原 objective 會讓它回到 teacher target，而不保留基座已有的 margin。**這些是目標函數的局部方向，不等於實際 AdamW 更新必然縮小，也未證明它們造成歷史 NQ 全部回退。** 共享參數干擾仍須配對訓練驗證。

Positive 來源與 truncation 沒有被掩蓋：FEVER 原始／補入為 145／20、100 個截斷；HotpotQA 256／4、3 個截斷；NQ 128／107、0 個截斷。原始標註沒有 supporting spans，因此「evidence 是否可見」仍是 unknown，不能從截斷與否推論證據一定存在或缺失。

## 2. 已實作與數值驗證

[`ng77_retention.py`](../../../../scripts/ng77_retention.py) 實作單側 Bernoulli gap、可信 anchor 係數及 `L_D + lambda * L_keep` 組合；尚未接上新的實驗 controller。使用原 head/cutoff priority、teacher confidence，逐 positive 正規化後除以**全部** positive 數，未覆蓋者仍在分母。unknown／tie／opposition 不當成負例。

數值上先 cast FP64 再做 margin 差分，以反向 Bernoulli 事件避免大正 logit 相減。當下降量不超過 `1e-3`，使用五階 cumulant 展開避免兩個一階小量相消；這是經驗證的浮點近似，不是忽略小變化的 deadband。Decimal 100 位參照涵蓋邊界、`1e-12` 級下降與極端 logits，另測 empty anchors、FP32 gradient cast、shared query/document VJP、四 query 梯度累積及 lambda0 的原始 FP32 scalar/gradient **bit-exact** parity。改善 margin 的直接 penalty 與 gradient 為零。

原始準備程式的 95 項 frozen fixtures 通過；新增獨立 reviewer 的 8 項 fixtures 亦通過。独立 reviewer 不 import generator math 或 Torch，以 `math.fsum` 重算 cached teacher 點積、全部 anchor 身份／margin／confidence／權重／係數、頭部覆蓋及假想下降量的 loss／score gradient。85,417 對、45,175 anchors、660 positive 分母全部一致；對原 witness 的教師分數最大差 `4.44e-16`。**這仍不是實際 CUDA encoder 的 no-update 或新訓練 VJP 資格。**

## 3. 下一個固定干預

準備階段預先固定檢查 `.1/.5/1.0` 三種「所有 positive 分數下降」的合成情境，僅了解量綱，沒有拿訓練或 DEV 結果掃係數。`lambda=1` 時，下降 0.5 的 keep score-gradient L1／原始初始 D L1，三域約為 0.413／0.472／0.501；下降 1.0 時約為 0.962／1.060／1.138。它不是一開始就壓倒 D 的項，也不是只在 margin 穿零後才反應。

因此下一個 execution protocol **預先只採 lambda0 對 lambda1、T=1**。這是同量綱、單一非零對照選擇，不是宣稱最佳係數。凍結的 preparation manifest 仍如實保留當時 `retention_coefficient=null`、`training_enabled=false`，不回寫歷史。

接下來先用原首輪順序中每域最早 4 題，共 12 個 TRAIN queries，做無 optimizer 的 CUDA canary。固定 CPU CSR baseline，不動模型／RMS calibration、不根據品質結果設 deadband。分別量測 FP32 forward 與 CPU baseline 分數差、零係數與原 D 的相容性、keep 的數值噪聲尺度以及 VJP。分數／code parity 延用預先的 `rtol=2e-5, atol=2e-7` 邊界；不能失敗後放寬或忽略 support 差異。

baseline 是 CPU FP64、forward 是 FP32，所以初始 keep 不必字面等於 0。若單文檔分數最大誤差為 e，由 sigmoid 的 1/4 Lipschitz 界與每 query 係數和不大於 1，可得該 query 的 keep score-gradient L1 不超過 `e / T^2`（另計可量測的浮點捨入）。canary 同時檢查實測初始 keep gradient 不超過原 D query-mean L1 的 `1e-3`；不通過就停在數值診斷，不做 optimizer。

canary 通過後才另凍結兩臂各 96 updates、同一批 384 queries 一次曝光的流程。中途不插入 GPU observer；training 關閉並封存模型／moments 後，另進程編碼全 233,009 文檔，至少評完整 384 TRAIN_PILOT 與 384 互斥 TRAIN_SENTINEL、全部 gold ranks，而非只看 24 題。lambda0 終端模型只有實際 SHA 與舊 D96 完全一致，才允許重用其文檔 cache。

在看任何新 terminal 結果前，execution manifest 必須固定 paired bootstrap／各域 non-regression／前排與 Recall 的 gate，保留既有 `-0.005` 各域 floor，不刪 NQ、不靠 loss 下降自動加長至 192 步。既有 1,536 DEV 的用途與讀取 barrier 也必須先固定；本輪不進入 DEV。若保住訓練 anchor 卻無法改善廣泛 TRAIN_SENTINEL，應分析 transfer／pool 外競爭者，而不是繼續換 lambda。

兩臂加全庫編碼、排名、hash、closure、mirror 和獨立 review 仍先按 40–70 分鐘量級規劃，須由 canary 更新；本輪 16 秒的 CPU audit 不能拿來估計新訓練。單卡新查空閒，CPU4、RSS16GiB／GPU20GiB／host24GiB／disk40GiB，優先每 phase 1800 秒有界，不放寬失敗邊界。

## 4. 可重現性與完成範圍

程式：[`prepare_ng77_retention.py`](../../../../scripts/prepare_ng77_retention.py)、[`review_ng77_preparation.py`](../../../../scripts/review_ng77_preparation.py)。source commits `240107dc23892f8ef37770ae552de5a791ec464f`、`185b4b91`。耐久單元為 `NG-0077/preparation-v1`，程序與獨立審核分別在同級 `preparation-v1-procedure-v1`、`preparation-v1-review-v1`。

| 證據 | SHA256 |
| --- | --- |
| Frozen inputs，143 dependencies／8 source entries | `8e9861f011bcbff1f1f43cee450748fee88256379a9fc2370f1c1857d16c9ac4` |
| Anchor audit results | `5d46abe3cc3b102b99862cf89eb9495e4a60005ad2005dde028131d0a97bc5d4` |
| 準備程序 complete | `6f4ee6940642389cce99c3fb2989c58810f710d6578d92faf6f3900f39cd6ff5` |
| 獨立審核 results | `0450d60914add252b386928c2b33001e2002d7cb9571b01add1f1eeb3a00777c` |
| 獨立審核 complete | `faff1a4c1ee6f7060b4b9fd9ae11c767a87112853be5625e202be3f32531ad5b` |
| 32 檔 relative-path→SHA inventory 的 sorted-JSON digest | `d21bc18370b3accee6f5cdb6473ee0c860e5a31e092578f8307127b324d2c7fe` |

準備程序 exit0／error null／owned group closed，16.615 秒、peak tree RSS 995,491,840 bytes；ClearML 在實際開始建立並已 offline closed/passed，尚未 remote-synced。獨立審核亦 exit0／error null／group closed，8.874 秒、986,546,176 bytes。兩者均未觸發或放寬 900 秒／8GiB 的 CPU bound。

主工作樹的 NG70–77 相關回歸及架構／導航／模型發布文檔測試共 `461 passed`，3.62 秒；`git diff --check` 通過。這是本輪相關 suite，不冒充全產品 PostgreSQL 回歸或尚未執行的 CUDA canary。

完整 32 檔、72,828,023 bytes 已複製到 lambda2 並重新核對 exact inventories、全部 143 dependencies 和 source SHA，沒有覆蓋、刪除或聲稱 disaster restore。07:58 UTC 現場 GPU0／3 空閒，GPU1／2 已佔用；**沒有新 GPU job 啟動，之後 launch 必須重新查詢而非依賴此快照。**

整體 hybrid 超 dense、獨立 held-out 泛化和 native 總成本降低仍全部未達成。本輪的實質進展是把下一個試驗收斂為：不改資料、不擴 pool，只補可信舊排序的單側保留，且能獨立判斷它是否真的有效。
