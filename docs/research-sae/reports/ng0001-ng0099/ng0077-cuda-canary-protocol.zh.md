# NG-0077：無 Optimizer CUDA Canary 執行契約

2026-09-13 UTC。在[CPU準備與獨立審核](ng0077-trusted-margin-preparation-review.zh.md)完成後，先驗證真實基座、FP32 forward與新loss的梯度鏈。**本協議沒有optimizer update，不是新模型訓練、品質評估或係數搜尋。**

執行結果：首題document cache的support相同，但值未通過以下原定閾值，因此attempt保持失敗，後續loss/VJP未執行。見[batch上下文診斷](ng0077-batch-context-diagnostic.zh.md)；下列凍結協議不因結果而放寬。

## 固定輸入與測試

使用原始首輪384 TRAIN_PILOT query順序，每域取最早4題，共12題；不按分數或NG76案例選取。所有canonical text、原始D pool、initial模型、CPU CSR、PPLX supervision、RMS、BM25 .1＋semantic .9及input limits保留並SHA綁定。沒有sentinel、DEV或LOCKED_TEST inference。

1. 對12題的原始完整forward pool，先保存實際query/document CSR，再比對initial cache。support必須完全相同；值與完整hybrid分數使用已預定`rtol=2e-5, atol=2e-7`。錯誤會保存差異及退出，不能改成忽略很小posting。
2. 以原始FP32 D objective計算loss及score gradient；lambda0做相同keep bookkeeping，仍須返回原scalar object且梯度bit-exact。lambda1仍是唯一非零候選，T=1。
3. 初始keep score gradient與獨立FP64 sigmoid差分核對；每query L1不得超過觀測最大單文件分數誤差e，加`1e-12`浮點捨入界。12題的平均keep gradient亦不得超過平均D gradient的`1e-3`。固定CPU baseline，不為通過而rebaseline或加deadband。
4. 每域第一題取第一個已凍結anchor的兩份TRAIN原文，另做**明確標記的人工兩文件VJP fixture**。其baseline設為該人工fixture當前margin＋1，只為確保keep分支有非零梯度；使用固定人工pair target .8、coefficient .6，不冒充training qrel或改寫45,175個真實anchors。這不是給科學訓練重設baseline。
5. 人工fixture分別在lambda0與1比對direct autograd與分document replay，兩條路共享同一encoder。relative gradient L2不超過既有NG71的`1e-5`；lambda0另與未加transform的原D路徑逐tensor bit-exact比較。一次按四query累積的1/4縮放；完整四query合成累積另由CPU fixtures涵蓋。只保留必要梯度摘要／SHA，不產生多份大型梯度dump。
6. 結束時model state與CPU/CUDA RNG不變、gradients已清空、無optimizer建立或step。測試器新增的loss-transform入口僅在研究訓練primitive中，預設None路徑不改舊行為；不改PostgreSQL產品代碼。

## 執行與下一階段

新unit獨立凍結source／tests／本協議／父資料和12題身份；禁止原地改CPU準備單元。單張現場重新確認空閒的lambda2 GPU，沿用GPU鎖與監督器：CPU4、tree RSS16GiB、GPU總使用20GiB、host可用24GiB、disk可用40GiB、phase最長1800秒。第一題後以完整load時間＋12題預估＋360秒VJP/closure餘量檢查1740秒內可完成。

ClearML實際開始、offline結束狀態、stdout、partial outputs、exit/error、owned process group、complete inventory與遠端／Mac鏡像都保留。只有canary通過且完整核對後，才能另凍結科學訓練controller；此controller本身不能啟動96-step study。

後續仍固定lambda0對lambda1、各96次更新、同384題一次曝光與相同pool。另進程編碼全233,009文件和768個TRAIN pilot/sentinel queries；不要誤用會包含DEV的NG71 terminal surface helper。先固定新終端comparisons／bootstrap／各域floor與後續DEV barrier，再開始訓練。NG77的CPU結果不是已贏dense，canary通過也不是。
