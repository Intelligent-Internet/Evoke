# NG-0072：凍結編碼的雙端排序能力保留診斷

日期：2026-09-13。狀態：預先固定診斷，尚未執行。延續[NG71終局審核](ng0071-final-pilot-review.zh.md)第6節；不把失敗的NQ門檻放寬，也不擴大訓練。

## 目標與界線

NG71的D組相對A有平均收益，但NQ的TRAIN提升沒有遷移到sentinel／DEV。先在現有編碼中定位query端、document端及其交互項，而不是直接疊加loss或更多資料。這是代數counterfactual，不是訓練干預的因果證明；不代表可以直接部署混用版本的encoder。

固定NG71 pilot384與sentinel384，兩者均為TRAIN，每surface每域128。不按新結果選題，不觀察新DEV／LOCKED分數。每種組合以完整233,009文件作背景，保存所有正例rank與top100，不以候選池代替全庫。沿用同一詞彙座標、RMS、BM25/semantic比例與已封存FP32編碼；累加使用FP64，沒有重新校準、模型推論、teacher推論或optimizer更新。

## 四種組合與可驗證分解

00：初始query／初始document；10：D192 query／初始document；01：初始query／D192 document；11：D192 query／D192 document。每個query先重現00／11的既有TRAIN top100、全部gold ranks、scores與metrics；兩條全分數累加路徑誤差不得超過1e-12，並保持numeric-ID tie ordering。交叉結果只有在這兩個identity control通過後才有意義。

已包含凍結尺度的code，固定lexical項相消，令 $\Delta q=q_1-q_0$、$\Delta d=d_1-d_0$：

$$
\Delta s=\Delta q^T d_0+q_0^T\Delta d+\Delta q^T\Delta d.
$$

對正例與同一競爭文件的margin做同樣分解。排序／nDCG非線性，不能把三個score項當作三個可相加的nDCG改善。另對每個文件的有效matching product $q_jd_j$ 分成「support失去」「support新生」「共同support權重改變」，三項之和也必須等於score差；這個support同時受query和document影響，不能直接叫作document posting刪除。

## 競爭集合與監督分層

每題margin說明集合預先固定為00／11的top10與rank80–100聯集，加原始pool與可用的A0/A96 witness；去除全部gold、依numeric ID排序。所有gold各自對完整說明集合保留margin，不只挑掉榜案例。固定TRAIN全集包括改善、不變和變差的控制組。全庫排序仍獨立於這個說明集合。

按NG70原始／新增positive lineage保存來源，不將added標成錯誤。Teacher方向只讀已存在original／A0／A96分數；說明集合中新出現但未評過的pair標為unobserved，不補推論、不當成負例。Pilot有既存token截斷欄位；sentinel沒有則保存unknown。沒有evidence span judgments時，一律不推斷語義證據已可見或缺失。

Margin先對競爭者平均，再正例平均，再query平均，避免多正例／大pool支配。按baseline/final同一pair排序分為lost、retained、gained、stayed-behind，零分數差採numeric-ID tie rule。各組記錄pair數及對同一全體分母的平均貢獻，不能把各組不同分母的均值相加。原始／新增正例的子群均值另行標明其query與positive分母。

## 工程與下一步

[實作](../../../../scripts/ng72_diagnose.py)重用凍結NG71的完整score/rank校驗與NG66全正例harm審核；[測試](../../../../tests/test_ng72_diagnose.py)涵蓋雙端／support恆等式、單端控制、全部gold／tie parity、TRAIN-only與input不可變性。NG62–65的舊rank/review已做全庫差異與雙路累加，不重新打造框架；本輪新增的是四種cross-code組合與margin解剖。

新unit `NG-0072/cross-codes-v1`，source／protocol／tests及依賴SHA啟動前凍結，原NG71不改動。CPU4、treeRSS16GiB、host available>24GiB、free>40GiB、5400秒，初始16題canary推估含1.5x餘裕與180秒收尾不得超過5100秒。超界保留失敗，不重試／放寬；資源監控每0.5秒，擁有的process group在任何退出路徑關閉。ClearML actual-start offline並記錄真實close，無GPU分配。

Mac準備時可用RAM低於24GiB，故不在Mac啟動此受限worker；優先lambda2 CPU執行，Spark-1只作跳板，不占同事GPU。重要source／方法／結論進repo，大型逐題產物進外部NG單元；封存後回傳與全SHA核對。實際耗時以canary調整agent輪詢；研究排序耗時不能充當native serving latency。

如果query-only替换主要造成既有margin損失，再設計query retention／固定文件對照；若document-only主導，優先document margin/support retention；若交互與teacher衝突主導，先驗證共享幾何／可信pair監督。任何loss干預另凍結最小對照，不由本診斷自動授權擴大或發布。Rank weights若繼續使用，uniform-pair消融仍必要。這些分支是待驗證假說，不是本協議預先宣布的答案。
