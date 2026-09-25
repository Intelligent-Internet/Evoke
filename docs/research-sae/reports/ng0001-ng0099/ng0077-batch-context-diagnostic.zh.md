# NG-0077：首個 CUDA Canary 失敗與 Batch 上下文診斷

2026-09-13 UTC。無optimizer canary在第一題的document cache value parity退出，沒有進入新loss或VJP測試，更沒有科學訓練。原失敗unit、閾值、baseline和partial outputs均保留，不原地重跑或把失敗改成通過。

## 已有證據

第一題是原training order的首題NQ，共78份文件。query的53個posting與document的23,753個posting support完全相同；query最大值誤差5.96e-8通過，document最大值誤差3.337e-6未通過預設`rtol=2e-5, atol=2e-7`。CPU逐座標檢查發現558個超界值，涉及34份文件；其中10個座標出現在本題query的support。

舊cache按全庫document ID順序每4份編碼；本次forward按TRAIN candidate pool每4份編碼。兩者皆使用動態padding，而非固定補到256 tokens。對這78份文件，27份padding未變，全部通過；51份padding改變，其中34份未通過、17份通過。這是上下文關聯，不是已證實的CUDA kernel因果。CPU使用已保存FP32 codes重算FP64 semantic分數，最大差1.381e-6、仍在原分數誤差內；它不是未完成的CUDA score／loss／VJP gate，也不能代表品質不受影響。

## 新的診斷，不是放寬准入

新`batch-context-diagnostic-v1`只使用失敗首題的固定78份文件及它們在原全庫四文件batch中的companions。來源、failed attempt完整inventory、原NG59公式、NG69編碼順序、同一模型／tokenizer／RMS及兩種group IDs在新unit凍結。沒有新query選擇、梯度、optimizer、quality ranking、sentinel、DEV或LOCKED_TEST inference。

1. 按原candidate pool上下文重算，對比保存的失敗輸出，判斷是否可重現。
2. 同一模型按原全庫四文件上下文重算，對比舊cache。仍使用完全相同support和值閾值，不刪除微小posting、不重新定義baseline。
3. 在每個上下文內，現有encoder與從原NG59逐步核對的直接forward公式要求bit-exact，排除research wrapper公式改寫；記錄全部group IDs、padding長度與逐文件超界數。
4. 前後model SHA、CPU/CUDA RNG與空gradient不變；actual-start/closed ClearML、full file hashes、exit/error與owned process group closure完整保存。

若只有恢復原batch上下文後cache parity恢復，才支持「跨上下文的數值比對不等於同上下文實作一致性」這個解釋。若仍未恢復，保留結果並檢查runtime/來源；不自動加大容差或重試。此診斷即使完成，**也不授權96-step訓練、不補發原canary通過證書**。後續需以明確的同上下文實作parity，加上未變的實際score／keep-noise／VJP gates設計新的准入，不能用一句「浮點誤差」跳過驗證。

只用一張重新確認空閒的lambda2 GPU，CPU4、GPU20GiB、tree RSS16GiB、host free24GiB、disk free40GiB、phase1800秒。預計全流程包含模型載入、父檔hash、ClearML closure和mirror約數分鐘，不將少量forward時間冒充總耗時。

## 08:43 UTC：對照完成

兩種上下文的實際CUDA對照已完成，未做optimizer update：

- 恢復原全庫四文件batch後，78份目標文件與舊cache的support和值**完全一致，最大誤差0**。Mac再直接核對全部228份重新編碼的target／companion documents，亦逐值一致。
- 按候選池batch重新計算的78份文件，與失敗canary保存的輸出**逐值完全一致**；對舊cache的558個超界值／34份文件亦重現。
- 在每一種上下文內，新encoder與原NG59直接forward公式均bit-exact；初始／終端model SHA一致，CPU/CUDA RNG不變，沒有gradient或參數更新。

因此本例能定位到**batch執行上下文造成的輸出數值差異**，而不是新loss、模型更新或wrapper公式變更。動態padding長度是可觀測的差異，尚未進一步隔離到特定CUDA kernel；不能據此宣稱歷史NQ排序回退也是padding造成的。首個canary把「不同batch上下文」當成了逐座標cache等價對照，這個比較前提不成立。

這是測試設計診斷，不是品質進步。原canary仍FAILED，原12題的實際CUDA score／keep-noise與三域VJP還沒有完成。下一個新凍結canary須分開兩件事：在同一原cache上下文驗證strict support/value parity；在**不改原D candidate pool及microbatch**的實際路徑驗證原公式一致性、原CPU baseline分數差、lambda0和共享VJP。跨上下文的逐座標差異繼續保存，不刪除、不靠放寬閾值偽造舊gate通過。實際分數仍用`rtol=2e-5, atol=2e-7`，keep noise／Lipschitz／VJP邊界不變，45,175個baseline anchors不重設。這一新對照設計需要另行凍結，不能直接啟動96步訓練。

### 執行證據

實作與協議commit `576c8f5dc7dfb7a48efcebac07c181a77c9df1ae`；程式[`ng77_context_diagnostic.py`](../../../../scripts/ng77_context_diagnostic.py)，測試[`test_ng77_context.py`](../../../../tests/test_ng77_context.py)。原canary程式commit `3513fb4572a6415fdb293ede5eb7735d468d72de`，沒有在失敗單元中修改檔案。

| 證據 | SHA256 |
| --- | --- |
| 失敗canary inputs | `38c37764dd99f6193ace406e46c64f0f211ff3486d87551c7d2d3bcf61f75560` |
| 失敗canary complete，passed=false | `9ba905faf1df8298260cbe2f30bbe4b03baac5f83c449b23a8f1425c74d94b73` |
| 新上下文診斷inputs，211 dependencies／22 source entries | `be201ffa3ae5d7a6c6694bdd734a1b4580fd32414645ba0671b3322afb999237` |
| 新診斷complete | `acaf688396078a64519f3f1a4f5949c98309231607a3d799a1e785c5930963bb` |
| 新診斷results | `15fa5c006afd29edfc966aa1cfecb53dd43f62761b177537f0df5e2e19c0e315` |
| 兩個完整單元的遠端inventory | `0be30ca2f9578d1b610f8684b39cdece7b19a36724160498b0f242e7c20d76c9` |
| Mac mirror／獨立逐值及fsum核對 | `94d30e4169a1823ba968d698ecb7a4278d666d655bf756f058f311265787b719` |

失敗canary exit1／error null／group closed，38.341秒、peak RSS2,371,678,208 bytes；診斷exit0／error null／group closed，74.287秒、peak RSS2,212,327,424 bytes、GPU allocated624,919,552 bytes。兩者均有actual-start並closed的ClearML offline紀錄，未remote-synced。08:41 UTC遠端未見owned workers，GPU0回到1MiB／0%；後續啟動仍需重新檢查。

兩個完整單元共75檔／1,211,947 bytes已在Mac與lambda2核對exact inventories、SHA及全部父資料／來源，無覆蓋、刪除或disaster-restore宣稱。Mac未做inference，以逐座標直接差分及`math.fsum`重算，CPU hybrid最大分數差1.3805725723e-6，仍在原score容差；這不補發CUDA score/VJP准入。主工作樹相關研究／文檔回歸487 passed；遠端frozen context＋CUDA fixtures 26 passed。沒有新訓練、DEV或LOCKED_TEST評估，也沒有新的native-cost或超dense成果。
