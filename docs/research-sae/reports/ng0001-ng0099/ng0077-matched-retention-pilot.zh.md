# NG-0077：可信排序保留的配對 96 步實驗

2026-09-13 UTC。接續[retention設計及歷史依據](ng0077-trusted-margin-retention-plan.zh.md)與[已完成的CUDA准入](ng0077-context-qualified-canary.zh.md)。本輪只比較原D排序訓練與加入單側保留項，完整目標仍是BM25＋semantic超過dense且native總成本更低，並非只提高anchor上的一致性。

終局狀態：兩臂及獨立TRAIN終端執行均已完成，品質門檻失敗；[最終審核](ng0077-final-retention-review.zh.md)記錄全部比較、限制及下一個有界問題。本文件保留事前選擇，不因結果改動gates。

## 固定比較

- Z：原D objective＋lambda0；K：原D objective＋lambda1，T=1。兩臂均計算相同45,175 anchors的keep bookkeeping，只有其loss係數不同，不掃lambda。
- 同一NG3 seed3003 checkpoint096、shared trunk＋MLM head、PPLX teacher、BM25 0.1＋semantic 0.9／RMS。模型、tokenizer、全部parents及source在新unit逐檔綁定SHA。
- 每臂384原TRAIN_PILOT queries各一次，全部660 query-positive關係保留，原D候選pool／順序不變；96次AdamW、batch4、microquery1／microdocument4，query64／document256是輸入token上限，不是posting cap。
- trunk LR5e-6、head LR2e-5、weight decay0.01、betas0.9／0.999、eps1e-8、clip1、FP32／TF32 false、eval/dropout off保持原契約。共享encoder的document VJP完整replay，不以detach冒充凍結文件encoder。
- 沒有GPU observer插在optimizer updates之間，沒有pool refresh、DF刪除、模型重置、第二epoch、DEV或LOCKED_TEST。新增execution hook只注入已綁定query ID的loss callback並讀取其scalar receipt；lambda0保留原FP32 loss物件。

## 分階段執行與證據

新`matched-retention-96-v1`先凍結**整個研究的比較、768題TRAIN終端身份及以下判定規則**，但其執行圖只包含`train-Z-96`、`train-K-96`。兩臂在同一空閒GPU依序執行，各自從原base開始；Z的完整closure是K啟動前提。這不是把兩臂串成192步，也不在Z品質出來後選擇K的參數。

每臂保存96次真實finite updates、384個query身份與score／score-gradient／D及keep components、query/document tokens與曝光量、完整model和AdamW moments、重載bit-exact證明、actual-start且closed的ClearML，以及exit／error／owned group closure。共享`ng71_pilot`的有界supervisor，不另造一套資源管理。CPU4、GPU20GiB、tree RSS16GiB、host free24GiB、disk free40GiB、每phase1800秒；不因失敗放寬或重跑。

兩個training process退出並封存後，才能進入另行凍結的encoding／ranking執行單元。預定編碼全部233,009份文件及384 TRAIN_PILOT＋384互斥TRAIN_SENTINEL，共768題，計算完整hybrid top100和全部gold ranks。使用明確`include_dev=false`的選擇器，不使用舊terminal預設附带1536 DEV的路徑。

舊D96文檔cache只在Z終端model SHA精確等於`92e574071092f08f6288a980ac27a1f0e17b82b0274acac40d3cf8cc228abd49`時可重用；否則兩臂均重新編碼。即使doc cache匹配，768題的終端query仍須按新凍結身份完整產生，不混用舊384題cache。新encoding controller必須在任何新終端品質計算前凍結，不以training loss挑checkpoint或變更以下gate。

## 事前判定

主要探索准入surface是完整384題TRAIN_SENTINEL，TRAIN_PILOT另列，不把已看過的sentinel稱獨立held-out。固定query-paired、逐域stratified bootstrap，seed71071、10,000次；比較K−Z、K−initial、K−dense，同時呈現Z−initial、Z−dense及BM25基線。

1. K−Z macro nDCG@10至少+0.005，95%下界嚴格大於0；Recall@100的95%下界至少-0.002。
2. 三域各自nDCG@10及all-positive Recall@100相對Z與initial都不得低於-0.005；NQ不得刪除，控制臂自身退化亦完整顯示。
3. K相對initial的document NNZ及sentinel query-DF工作量proxy皆不超過1.25倍。這些是風險上限，**不是native延遲／bytes勝出**。
4. 即使上述通過，也不自動延長192步、不自動進DEV、不宣告超dense。先完成逐query／domain、原可信pair保留／新競爭者／錯誤transfer分析，再決定下一個有界實驗。既有1536 DEV已曝光；最終宣稱仍需要未曝光的獨立驗證及完整native總成本。

若keep讓anchor變好而廣泛TRAIN前排沒有變好，優先分析有限pool之外的競爭與shared-parameter transfer，不反覆換lambda，也不把未知／teacher反對的pair改為negative。

## 時間與限制

舊96步程序354.84秒、全庫編碼662.93秒僅作起點。這次CUDA無更新body24.74秒，但完整phase373.40秒，其中ClearML repository自動偵測出現超時警告後仍正常offline closure；此tracking開銷必須納入預估，不能只看GPU計算。先估兩臂training與closure約15–25分鐘，完整endpoint、mirror及audit約60–90分鐘，隨實際phase估時修正輪詢。沒有可宣稱的新模型品質提升。

## 09:31 UTC：已開始真實配對訓練

程式／協議commit `f60d7a91ad35e4fa79187ab49b457fd4e7c82a95`；新inputs SHA `16f9bf709176a88d46627ea56e24466820430d352a1008d080b488ab65691e97`，365 dependencies／25 source entries，全部在Mac及lambda2驗證。主工作樹相關NG70–77／文檔測試513 passed；遠端只用新frozen sources的35個fixtures全部通過。沒有push或production部署。

lambda2 GPU0的`train-Z-96`於09:29 UTC啟動。09:31快照已有33/96次真實更新；每次VJP replay及lambda0 components檢查通過。第1／24次估時636.63／689.03秒，低於新1800秒phase對應的1500秒預估門檻；最新一次update L2為0.00776435、gradient norm0.433761、2.862秒。GPU0為2,584MiB／85%，worker RSS2,127,507,456 bytes；GPU1／2同事工作未動。這是資源及執行進度，不是模型品質結論。

controller PID3492304，Z worker PID3492321，tmux `ii42_ng77_matched_96_v1`。actual-start ClearML `offline-514cfebdd6e946be9807c4e9f328b6e8`目前仍開啟；Z尚未closure，K依序等待，兩臂都沒有新終端品質評估。控制器健康，不重啟。凍結manifest的`actual_training_started=false`表示凍結時的狀態，後續真實狀態以`started.json`／`progress.jsonl`／`exit.json`為準，不修改immutable inputs。

原始資料單元：Mac `development/research/NG-0077/matched-retention-96-v1`，lambda2 `/home/huoju/leask/research/NG-0077/matched-retention-96-v1`。完成後仍需兩臂closure、完整mirror及實際loss／gradient／曝光審核，再以已凍結的768題身份和gates建下一個endpoint執行單元，不能把此啟動紀錄當成研究完成。

## 09:41 UTC：兩臂訓練完成

Z、K均完成96/96次更新，controller狀態`all_phases_complete`。每臂384題、6,314 query tokens、3,953,027 document tokens、30,493候選文件曝光完全一致。Z／K phase為361.056／362.066秒，峰值tree RSS為3,045,031,936／3,063,808,000bytes；最大GPU allocation兩臂均1,896,107,008bytes。兩phaseexit0／error=null／owned group closed，兩個ClearML offline task已closed（K：`offline-2fcce9232f694185a220988199788928`）。

Z終端parameter SHA與歷史D96相同；K已產生不同parameter SHA。這是控制一致性和真實訓練完成，不是排序提升。後續使用[獨立TRAIN endpoint協議](ng0077-endpoint-protocol.zh.md)完成全量訓練receipt審核、768題終端觀測、全庫排名、可信關係transfer及成本proxy驗證，不啟動192步或DEV。
