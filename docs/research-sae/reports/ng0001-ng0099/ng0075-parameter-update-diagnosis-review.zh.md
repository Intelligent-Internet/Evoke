# NG-0075：真實位移可預測，但原診斷漏掉關鍵競爭者

2026-09-13 UTC。**四個disposable replay、終端審核、完整Mac mirror及另一套獨立數值reduction均完成。** 本輪是定位證據，不是新模型、泛化通過或native-cost成果。前置與逐時狀態見 [方案](ng0075-parameter-update-diagnosis-plan.zh.md)／[狀態](ng0075-parameter-update-diagnosis-status.zh.md)。

## 結論

1. **局部更新不是不可控的非線性跳變。** 真實AdamW參數位移對388個固定pair的margin增减方向，全部與一階Jacobian投影一致；各surface/domain的 residual L1／actual-change L1為0.28%–3.98%。這不是每個pair的相對誤差都小於4%，更不是192步線性外推保證。
2. **score-space要求正確方向不保證共享参数真的照做。** 可觀察同batch總score pressure要求擴margin、實際位移卻縮margin的情況：44個此類pair-exposures中2個，來自同一pair在D1/U1兩個case，不能當作兩個獨立樣本。兩端／trunk／head都可能造成間接影響，沒有證據支持一刀切凍結某端。
3. **但這沒有解釋終局NQ回退。** 四個case的NQ sentinel平均margin均改善，388個局部對比沒有觀察到預定數值底線外的正負margin翻轉。不能把「有縮margin」直接叫做已證明的掉榜原因。
4. **真正的新發現是量測覆蓋不足。** 同一hash選定24題，在NG72既有完整說明集合中有60個端點lost comparisons；NG75每題兩個initial rivals只覆蓋其中1個。應改善觀察對象和時間跨度，而不是憑這個窄量測再調loss權重。

## 固定實驗與結果

原NG71 D／NG74 U，在step1與97各重放原首batch，四次實際AdamW更新、零新增科學模型訓練。共同base或各自D96/U96的model+optimizer保持原軌跡；每步scores／loss／score-gradients／update L2及初始model/moments核對通過。Step97的D/U不是相同state下只換權重的反事實對照。舊log沒有各step完整parameter SHA，不宣稱norm匹配唯一證明整條歷史軌跡。

24題未參訓但已曝光TRAIN sentinel，每域8題，42個gold及84個固定gold/rival pairs全部保留。加原四題batch對比，case列數102／102／92／92，共388次pair observation。沒有新DEV／LOCKED_TEST評分、全庫重編碼或模型推論到其他query。

下表是 **query-balanced mean finite hybrid margin change**，不是nDCG或Recall；分數尺度已含固定BM250.1、semantic0.9與RMS。括號為縮小pair數／全部pair數，分母是pair，不是query。

| Case | FEVER sentinel | HotpotQA sentinel | NQ sentinel |
|---|---:|---:|---:|
| D1 | -0.002214 (14/24) | +0.001797 (13/32) | +0.006910 (8/28) |
| U1 | -0.004424 (12/24) | +0.000287 (17/32) | +0.003432 (10/28) |
| D97 | +0.010144 (6/24) | +0.005445 (6/32) | +0.006912 (5/28) |
| U97 | +0.011325 (7/24) | +0.005571 (6/32) | +0.007210 (5/28) |

一階direction matches為388/388，共273次擴大、115次縮小，沒有stationary。這是同一小組pair在四個預定局部step上的數字，不是泛化成功率。Forward重複bit-exact；query/document Jacobian分解在同一共享參數上相加並核對full derivative，沒有把detach誤作凍結encoder。

各case／域original與added-positive分母、教師agree／oppose／tie／unobserved、pre-update正margin、同batchscore pressure與query/document/trunk/head投影均保存在獨立reduction。84個sentinel pairs中只有16個得到原teacher明確支持；多數不在已保存teacher pool，不能當作教師反對或已判無關。局部縮margin並不自動意味gold排序或標籤受損。

## 為什麼要修正下一輪量測

終端後的**描述性覆蓋審核**沒有新inference，沒有重新選NG75題目或改其結果。方法：以 `query_id/positive_id/rival_id` 將凍結NG75 pairs連接NG72 `margins.jsonl`，後者的lost為 `baseline_ahead && !final_ahead`；同時連接NG71初始／D192既有rankings計算相同query集合的端點變化。重現實作在 `scripts/prepare_ng76_trajectory.py:coverage`；它只報覆蓋，不用lost篩選新pool。

| 同8題／域 | NG72說明pairs | 其中lost | NG75固定pairs | 涵蓋lost | 初始→D192 nDCG平均差 |
|---|---:|---:|---:|---:|---:|
| FEVER | 752 | 8 | 24 | 0 | +0.046202 |
| HotpotQA | 992 | 34 | 32 | 0 | -0.005313 |
| NQ | 814 | 18 | 28 | 1 | -0.050690 |

NQ這8題中2題改善、3題持平、3題變差；原28個固定pair的端點平均margin卻增加1.008574。這與NG72「平均margin改善、前排品質下降」一致，現在進一步確認NG75的簡化rival選擇漏掉了大多數已知的關鍵對比。表中的60個lost是NG72有限說明集合，不是宣稱已枚舉全庫所有反超關係；24題也不是新的代表性holdout。

因此下一輪不是把更多參數調到這84個pair好看，而是 [NG76完整端點top100聯集／軌跡](ng0076-endpoint-union-trajectory-plan.zh.md)：同24題、不只挑lost，保留初始與D192全部top100及所有gold，在原D192步歷史replay上定期觀察。先檢查端點排序可重現，再定位何時哪些關係開始流失；不先增加新loss／epoch或宣稱干擾已解決。

## 執行與審核

source `66d2ad663488cf385f5d9617ebc1332c19ae9301`；manifest `93087ed7480f320b762d296f9f1e36cf7fa891546cab8b36a6181c14f87aa822`。lambda2 GPU0控制器於04:49:39 UTC全部完成；五個phase均exit0／error null／owned group closed／actual-start offline ClearML closed。總worker時間1090.06秒，個別D1/U1/D97/U97/review為395.20／100.27／100.50／400.39／93.70秒。追蹤收尾不是每個phase固定300秒，之前25–35分鐘估時偏保守，沒有重啟或放寬上限。

最大tree RSS2,218,627,072 bytes，遠低於16GiB；GPU0已回到1MiB/0%，沒有NG75訓練worker残留。同事GPU1/2仍有工作，未干預。這些是研究作業資源，不是native serving latency。

Mac完整mirror共101 files／3,545,534 bytes；本輪增傳73 files／824,890 bytes，rsync退出0、0刪除。遠端終局inventory SHA、每個phase seal和全部407依賴都重新核對；遠端来源仍保留，沒有做災難恢復測試。

獨立reviewer `scripts/review_ng75_displacement.py`，commit `8d47f149`，不import producer數學helpers、不importTorch、不做模型／Jacobian推論。以標準庫另算全部388列finite delta、role/group投影和、Taylor residual、direction、teacher分類、same-batch pressure及pair/query-balanced分母，對照producer與終局review，最大差`2.7755575615628914e-17`。前後全部輸入／输出SHA保持不變；Mac工作11.91秒、peak RSS33,013,760 bytes，exit0／group closed。這是獨立reduction實作，不冒充raw Jacobian獨立重算。

| 證據 | SHA256 |
|---|---|
| `review/complete.json` | `df2b9e2d2e0030020fe53aa7c511fb59e29aefc10dd5639badd5c8dc13d00667` |
| remote全run inventory | `488c70dbe85742768f63dd223f6e0cf72189d35c6badf2f956fc2e096f9ada6d` |
| Mac independent `verification.json` | `fa456b784ef0e73e8137259893527f1b478557f7d34b6e2028a5ab8108941c0c` |
| Mac independent `complete.json` | `0c37c05b083c81037939254f7b3e66fc814893873ac261757f8907fee4e2cf5f` |

NG70–75、獨立review與三組主要文檔測試為284 passed（1.99秒），不是新增284個測試。新方法與結論在主repo；原始結果、來源及全部成功／失敗證據在外部catalog，不清理或改寫凍結實驗。完整hybrid仍未超過dense，NG71/74失敗的NQ門檻及native total-cost待驗證狀態保持不變。
