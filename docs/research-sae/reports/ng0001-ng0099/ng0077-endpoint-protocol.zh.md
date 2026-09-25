# NG-0077：配對訓練後的完整 TRAIN 終端驗證

2026-09-13 UTC。延續[事前凍結的96步研究協議](ng0077-matched-retention-pilot.zh.md)。本文件只落實執行／核對細節，不重新選query、lambda、checkpoint、品質floor或cost ceiling。目標仍是BM25＋semantic的整體排名／召回與總native成本；目前只是TRAIN探索，不是產品升級或獨立held-out證據。

終局狀態：11:03 UTC九個phase全部完成，但事前品質門檻失敗；見[最終結果與下一步](ng0077-final-retention-review.zh.md)。以下保留事前協議及當時的時間戳快照，不代表程序仍在執行。

## 訓練已退出，品質尚未計算

lambda2上的Z與K各完成96次AdamW更新、384個不同TRAIN query曝光，全部660 positives及45,175 anchors不變。每臂實際曝光6,314 query tokens、3,953,027 document tokens、30,493 candidate documents。Z用361.06秒，K用362.07秒；兩臂exit0／error=null／owned group closed，actual-start ClearML均已closed，沒有GPU observer。controller於09:41 UTC完成，GPU0已釋放。

Z的model SHA `92e574071092f08f6288a980ac27a1f0e17b82b0274acac40d3cf8cc228abd49`與歷史D96相同；K的model SHA `b60541948b13a10fc84f06836238ac10e9bf0a18010306cb39362b37ef701056`不同。這證明本輪保留項實際改變了訓練產物，但**不是**改善排序的證據。

來源training inputs SHA `16f9bf709176a88d46627ea56e24466820430d352a1008d080b488ab65691e97`。新remote完整inventory SHA `53649fef49af83ee1a0def472ec719f8940a0ad0c49022d5b001c5e1108c8c8b`覆蓋62files／1,195,418,073bytes。Mac完整mirror及source／parent／內容核對必須完成才能freeze下一個endpoint單元；不覆寫原訓練manifest或既有失敗canary。

## 執行圖與算術審核

```text
closed Z96 + closed K96 + complete Mac mirror
                    |
         freeze train-endpoints-v1
                    |
      audit-training (CPU, no inference)
                    |
       encode-Z-96 -> encode-K-96
                    |
   rank initial / dense / BM25 / Z96 / K96
                    |
        metric + transfer + cost review
                    |
      process closure + mirror + review
```

第一phase的獨立CPU審核逐條重算兩臂768個實際query exposures，而不是只相信最終loss下降：原D項用獨立NumPy reference；keep項用50位Decimal凸函數gap及独立sigmoid導數，不調用訓練的Taylor近似。對原D與total loss沿用rtol2e-6／atol2e-7，四query accumulation後的score gradients沿用rtol2e-5／atol2e-7；keep gap另用rtol2e-10／atol1e-13。這些驗證界線在讀取新scalar審核結果之前固定，數值fixture包含lambda0／1、小margin變化、錯誤正例分母、遺漏`/4`及破損receipts。

另外核對query順序、全部正例、finite實際update、VJP replay receipt、tokens／候選曝光、完整model／AdamW檔案hash、重載receipt、兩phase不重疊、actual-start ClearML及安全closure。此CPU階段**不獨立重算整個模型parameter VJP**，後者的直接／replay准入已由封存CUDA gate提供；不能把scalar審核誇大成重新驗證所有shared-parameter導數。

## 編碼與觀測邊界

只用既定233,009文件與768題TRAIN：384pilot＋384sentinel。所有768個終端query codes都重新產生。Z匹配歷史D96完整parameter SHA，因此允許只重用其已封存、逐檔綁定的document CSR；同時綁定原corpus、tokenizer、RMS與source provenance。舊D96 complete SHA為`cd9ecb1f0349e9d0f3b3744dcda82c56da50048b5403e3c9d8003ba52b4df4ea`。不重用舊384題query cache；K完整重編全部文件。若model不符，改為兩臂均完整編碼，不放寬cache契約。

兩個encoding都成功退出後，才開始任何新排名。沿用既有full-background scorer、兩條獨立FP64累加、完整top100與全部gold ranks，document ID作deterministic tie-break。明確指定`train_only=true`及外部training root／兩臂barrier，避免舊terminal預設引入1536 DEV。沒有新增encoder語義、posting刪除或候選截短。

review按兩個TRAIN surface分開列出initial／dense／BM25／Z／K品質；固定比較K−Z、K−initial、K−dense、Z−initial、Z−dense。逐域query-paired bootstrap seed71071／10,000次，沿用事前凍結的全部gates，尤其三域nDCG／all-positive Recall相對Z及initial的-0.005 floor。執行成功不等於品質准入成功，也不授權192步或DEV。

額外只做描述性分析：重算原45,175可信pair的終端margin、原正確關係變錯／正例出top10而競爭者進top10，並列出新top10中不在原pool的競爭者。strict margin微變仍可能含浮點context噪聲；不以這些微變判定品質改善。這能區分「有沒有保住原關係」與「全庫排名有沒有改善」，但不能單靠相關性證明shared-parameter update的因果來源。

NNZ、CSR bytes與query-DF分佈只是成本proxy；K相對initial不得超過1.25倍的事前風險上限，不能把通過上限稱為成本下降。要宣稱勝過dense仍需要未曝光query驗證與BM25＋semantic總native latency／memory／physical bytes。

## 執行資源與重現

所有phase使用既有supervisor，CPU4、GPU上限20GiB、tree RSS16GiB、host available至少24GiB、disk至少40GiB、每phase1800秒。encoding預估門檻1200秒保留600秒收尾，ranking預估門檻1500秒保留300秒收尾，沒有放寬科學契約。ClearML仍為明確offline actual-start記錄；每個phase的load、hash、tracking close與mirror全部計入時間，不能只看GPU kernel時間。

預估endpoint含全庫K編碼、五個ranking、closure及mirror約30–60分鐘，按實際stage canary調整輪詢。任何失敗保留attempt，不自動重跑／加時／調lambda。單元在大型資料根目錄的`NG-0077/train-endpoints-v1`；重要程式、tests及本協議留主倉庫。

```bash
python -B scripts/ng77_endpoints.py freeze --research-root "$RESEARCH_ROOT" --run "$RUN"
python -B "$RUN/ng77_endpoints.py" verify --research-root "$RESEARCH_ROOT" --run "$RUN"
python -B "$RUN/ng77_endpoints.py" supervise --research-root "$RESEARCH_ROOT" --run "$RUN" --gpu 0
```

啟動前必須確認GPU0真的空閒，維持lambda2經spark-1跳轉的既有SSH路徑，不動其他GPU工作。直到endpoint／review／完整mirrors均完成，這輪研究都仍未完成。

## 10:09 UTC：獨立終端執行已啟動

程式／tests／協議commit `4fff32dc28138ed7d688fbe194637f796f0bc1f7`。Mac的訓練全鏡像62files／1,195,418,073bytes已完成exact inventory與逐檔SHA核對，原365 dependencies／25 source entries重驗通過。傳輸本身約580秒，這項成本已納入後續完成時間，沒有刪除或覆蓋remote產物。

新`train-endpoints-v1` inputs SHA `46fed531851ee1b9d7a49f7ef9ed32436aec1e55fe52d293e4d7759cdc1400d7`，460 dependencies／31 source entries，Mac及lambda2驗證通過。相關NG70–77與architecture／navigation／published-model文檔回歸544 passed（3.29秒），remote凍結sources的39個fixtures通過（7.79秒）；不是新增PostgreSQL產品回歸。代碼只本地提交，沒有push或production變更。

啟動前舊controller及Z／K worker均已不存在，GPU0為1MiB／0%，host available121,483,956,224bytes、disk free756,796,506,112bytes。新tmux `ii42_ng77_train_endpoints_v1`的controller PID3498598於10:08:46 UTC啟動，第一phase是CPU `audit-training`，worker PID3498644。10:09快照仍在input核對，尚未有ClearML-start／exit，也尚未進入encoding或新品質計算；程序存活，不重啟。下一個需觀察的證據是actual-loss audit成功封存，然後768題新query及全庫K編碼／排名。

## 10:12 UTC：實際訓練算術審核通過

`audit-training`已於10:11 UTC成功退出並封存：135.134秒，peak RSS1,056,993,280bytes，exit0／error=null／owned group closed；actual-start ClearML `offline-512c6babe8634d3e8c753f51fd76ad56`正常關閉。两臂全部768個真實exposures的score gradients與独立reference通過：Z最大誤差1.22525e-8，K為9.53593e-9；keep scalar與Decimal最大誤差分別1.32273e-17／1.38778e-17。全部positive／順序／曝光／重載與closure checks通過；Z完整parameter及optimizer fingerprint和歷史D96相同。

這只能說明本輪真的按已凍結的D＋keep objective執行。Z／K的mean keep loss為0.000328012／0.000308749，mean keep score-gradient L1為0.00322935／0.00311423，mean D L1為0.121793／0.122022；不同訓練trajectory上的平均值不是獨立品質A/B，也不是調lambda的許可。

控制器已進入`encode-Z-96`（GPU0，worker3498892，ClearML `offline-72359407191443a38b251a9f72f49637`），新768題query的canary預估9.88秒且通過。仍未有任何新品質排名結果；後續保持原K全庫編碼與五模型完整TRAIN endpoint，不在看到training scalar後變更gates。
