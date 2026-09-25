# NG-0080：F/D 終點的全背景診斷協議

2026-09-13，在新的終點品質分數產生前凍結。依據 [配對讀出訓練](ng0080-dense-control.zh.md)，不選中途 checkpoint，不修改已完成的訓練；先獨立審計兩組全部 3,072 更新，再比較 terminal3072。

## 固定比較

四個 normalized dense dot score：共同未訓練 mean projection（initial）、凍結 trunk 的 F、解凍 trunk 的 D，以及既有 PPLX teacher。不是舊 sparse initial，不是 BM25+SAE 產品結果，不評估 native cost。三個 student 都 fresh encode 全 233,009 canonical 文件，以及已固定 1,536 TRAIN_DIAGNOSTIC query；teacher 使用已 SHA 綁定的相同語料／角色 cache。不因結果選擇 mean/CLS 或輸入窗口。

TRAIN_DIAGNOSTIC 各域 512 題，對本次 optimizer 無曝光，但有歷史曝光，不能稱為獨立 holdout。保留全部已知正例；整個 233,009 文件背景，不做域內或候選 panel 限制。查詢身份與文件順序都從 readout-v1 綁定。沒有 DEV 或 LOCKED_TEST query 讀取，也沒有新增訓練。

## 先審計、再推論

獨立 CPU reviewer 重算每題 centered MSE 與 score derivative（含四題累積）、teacher score、正例 mask、完整 query 順序、實際 token/candidate exposure。逐 chunk 檢查同組 checkpoint chain，直接載入 serialized optimizer 的全部 moments／step、重算 name-bound fingerprint。這不是獨立重播全部參數 VJP 或 AdamW 更新；後兩項僅有原 worker 的 real-base canary、fresh replay 與 exact reload 證據，不混淆證明範圍。

每個 phase 必須 exit0、error=null、process group closed、actual-start tracking closed、精確檔案清單與 SHA 一致。Failed attempt 保留，不原地重試。模型輸出 fresh、FP32、TF32 false、eval mode、query64/document256、batch4；以參數 hash 和 normalization 驗證，推論無 optimizer 更新。

## 度量與判讀

全背景 FP64 dot 與獨立 einsum 兩種 reduction 每題核對，絕對誤差 <=1e-12。固定 document ordinal 打破同分；保存完整 top100 ID/score 與每個正例的 global rank。獨立 reviewer 重算 nDCG@10、all-positive Recall@100、top100 座標分數。另報 fixed panel 的 centered score MSE／nDCG，以及全庫 teacher top10 overlap：fidelity 改善不等於 relevance 改善。

比較 F-initial、D-initial、D-F、F-teacher、D-teacher；三域 macro 與分域都報。預先固定分域 paired bootstrap 10,000 次、seed80080，95% CI 僅反映同一初始化／順序下的 query variability，不涵蓋 training seed uncertainty。不新增晉級閾值，不以本次 TRAIN 診斷批准部署、擴模或接觸 locked test。

若 D 相對 F 的幾何及 relevance 同時改善，支持「此預算下底座適配有收益」；若僅 panel 提升，需檢查全庫分佈／負例覆盖；若幾何改善而 relevance 不變，檢查 teacher-label 衝突與 retrieval metric 對齊。任何結果都不是底座容量上界。S/P 真正 sparse 訓練、多源 supervision、FEVER context control、DF 聯合 mask 與完整 hybrid/native 成本仍需分別閉合。

每 phase <=5,400 秒、RSS <=16 GiB、主機 available >24 GiB、disk free >40 GiB；GPU 單卡最多20 GiB。前64份編碼／前16題排名外推1.5倍+300秒需 <5,100秒。CPU 四線程。兩條 encoding lane 分別使用空閒 GPU0/3，cooperative locks，不驅逐其他工作；排名只在兩條完整閉合後進行。Mac 不執行模型或 Torch 測試；完整大型輸出與 mirror receipt 分開驗證。
