# NG-0077：上下文正確的 CUDA 准入對照

2026-09-13 UTC。[已完成的上下文診斷](ng0077-batch-context-diagnostic.zh.md)定位首個canary的跨batch cache比對前提不成立。新`context-qualified-canary-v2`是明確重設**對照上下文**的獨立嘗試，不修改原失敗單元、数值閾值、模型、loss、baseline或training pool，也不補發v1通過證明。

沿用原首輪順序每域最早4題共12個TRAIN_PILOT queries、全部原候選池和45,175個CPU anchors。原全庫按document ID每4份分组，凍結每題所需原batch IDs；companions只供無optimizer的cache一致性對照，絕不加入loss或後续科學訓練。

- 每題的query仍按原microquery1執行；cache support及`rtol=2e-5, atol=2e-7`不變。
- 文檔cache parity在原全庫四文件上下文執行；檢查全部target及companion rows的support和值，不只挑目標文件。
- 實際loss仍使用原D candidate順序／microdocument4，其forward與原NG59公式逐值bit-exact。實際pool與cache的跨上下文code差異另存診斷，不冒充通過逐座標等價gate，也不刪除或mask微小posting。
- 實際pool的hybrid分數仍對比原CPU baseline，`rtol=2e-5, atol=2e-7`不變。原lambda0 scalar／gradient bit-exact、每query keep L1不超過score error加`1e-12`、mean keep不超過mean D的`1e-3`、獨立sigmoid gradient及三域兩文件shared-VJP的原`1e-5` relative L2邊界，全部保留。
- 原[canary協議](ng0077-cuda-canary-protocol.zh.md)的人工active-penalty fixture、1/4 accumulation、model／RNG不變、零optimizer及完整closure要求沿用。沒有新的品質評估、sentinel、DEV或LOCKED_TEST inference。

重用原canary controller，只新增原context controls和新manifest協議；舊frozen v1及context diagnostic來源不變。先凍結source／tests／本協議、已封存的context proof與全部parents，再單卡執行。CPU4、RSS16GiB、GPU20GiB、host free24GiB、disk free40GiB、phase1800秒與原1740秒估時門檻不變。只有完整通過、mirror及核對後才可另凍結lambda0／lambda1各96更新的科學訓練。

## 已完成：12 題與 Shared VJP 准入

lambda2 GPU0的新v2已完成並封存。12題的945個原候選曝光、3,456個原corpus-context文件曝光全部核對；同上下文document cache逐值誤差0，query最大誤差5.9604645e-8。12題跨上下文document value gate仍全部不通過，support則均一致；這些診斷保留，不改稱跨上下文code等價。

實際候選池hybrid對原CPU baseline的最大score error為2.0983550e-6，通過原分數容差。mean keep score-gradient L1為4.9566775e-8，mean D為0.1686346242，比值2.9393000e-7，遠低於原1e-3上限；每題Lipschitz與獨立sigmoid梯度核對亦通過。沒有因CPU/CUDA初始微差而產生相近於原D的保留壓力。

三域人工兩文件fixtures中，lambda0／1的direct與replayed shared-parameter gradients涵蓋全部107個tensors，relative L2與最大誤差均0。lambda0另外對未加adapter的原D共享梯度bit-exact；初始／終端model SHA及CPU/CUDA RNG不變，grad清空，零optimizer updates。

這解除的是實作與實驗對照准入問題，**沒有證明排序提升，也沒有證明歷史NQ回退來自batch rounding**。後續進入[固定lambda0／1配對96步研究](ng0077-matched-retention-pilot.zh.md)，不是繼續重跑已完成診斷。

### 封存與獨立核對

| 證據 | SHA256 |
| --- | --- |
| inputs，251 dependencies／21 source entries | `362a94b730e511bdb652e31c0bcae27feaaadc6b42c11ef0bed3f08d6cfe31ca` |
| CUDA complete | `7fd7c380d1e9a75c4e8275ed5f2c219b76833ec7fce0337e947e496f81abbb9c` |
| CUDA results | `4218c1ba74d889b72560652caaca58e83a6e63ec80e5ecf61a78bdf3ece3a4b6` |
| remote full inventory | `c24d1b4f4c941369688d22d570eaa12530f74645c2befff021a98846fc09748e` |
| Mac independent mirror／raw-code／scalar-score review | `89449f53e778adb3c13925612709c815e2ed6e03f7af9ffe42412c71372146d5` |
| Mac review procedure | `94ffa81a07228821e654b412ba878e1d22fb852484cd1c5eaa4cb186d3fcb54a` |

完整111檔／7,309,388 bytes已在Mac與lambda2核對exact inventory與SHA，另存launcher／inventory／review receipts，無覆蓋或刪除舊attempt。CUDA phase373.396秒、peak tree RSS3,236,577,280 bytes、GPU allocated957,789,184 bytes，exit0／error null／owned group closed，兩個owned PIDs均退出，GPU0回到1MiB／0%。actual-start ClearML `offline-8ad4165028b741808b9d6aeb8476a406` closed passed；repository自動偵測有超時警告，但offline session及閉合紀錄完整，尚未remote-synced。

Mac reviewer[`review_ng77_cuda_canary.py`](../../../../scripts/review_ng77_cuda_canary.py)不載入Torch或模型，獨立逐座標核對所有原context arrays，以`math.fsum`重算945個hybrid分數，與CUDA的最大差1.1856289e-6；再獨立重算keep score-gradient及noise上限。耗時5.993秒、RSS712,097,792 bytes、exit0／error null／group closed。完整參數VJP沒有在Mac重算，只有逐檔驗證sealed CUDA fixtures與其receipts，兩種證據不混稱。
