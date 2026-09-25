# NG-0076：完整端點 Top-100 聯集與訓練軌跡

2026-09-13 UTC。接續 [NG75終局](ng0075-parameter-update-diagnosis-review.zh.md)。CPU輸入準備與端點驗證已完成，但第一個有界歷史replay在第80步readout-count gate失敗，見[執行狀態](ng0076-endpoint-union-trajectory-status.zh.md)及[獨立無observer對照](ng0076-replay-control-plan.zh.md)；**歷史一致性與整條軌跡尚未完成，不是新loss訓練或模型發布。** 凍結 artifact 中的原始計畫和失敗現場保留不改；下面是原方案，不是繞過失敗繼續啟動第二段的授權。

## 改變的不是 Loss，而是觀察範圍

NG75的388個局部pair方向皆可由實際AdamW位移的一階投影預測，但原本兩個initial rivals僅涵蓋同24題NG72已知60個lost comparisons中的1個。兩個時間點也不能解釋192步累積變化。繼續放大同一個狹窄量測，或者據此挑retention係數，都不是合理下一步。

保持NG75的hash選定24題TRAIN sentinel，不因D/U結果換題、刪gold或只保留lost pairs。每題改為：

$$
C_q = H_{100}(s_0,q)\cup H_{100}(s_{192},q)\cup G_q.
$$

其中兩個head來自**已完成且已審核**的NG71初始／D192完整233,009文件排名，$G_q$是全部gold。所有gold對所有non-gold競爭者的margin均保留，包含retained、lost、gained和stayed-behind控制組。這是已曝光TRAIN上的回溯解釋，**不是盲測，non-gold也不是已判負例**。不得把終端聯集作為新模型的候選索引來宣稱召回品質。

身份統計：24題、2,974個不同文件、3,038個query-document組合、5,226個gold/rival pairs，最大單題pool144文件。不是只從2558個NG72說明pairs挑60個lost繼續研究；它擴為全部兩端top100和所有gold，不過濾勝負。

## 為什麼可以保留端點排名

在任一已知端點，若gold在top100內，所有排在它前面的文件必然存在於該端點的top100，因而都在 $C_q$ 中。如果gold原本在100以外，原top100的100個文件也全部在聯集中，仍會排在它之前。採相同score與global document-ID tie-break時，聯集可精確重現兩個端點的top100，以及全部gold的top100內rank／在100外的狀態，所以同時保留端點nDCG@10與all-positive Recall@100。

這個保證**不適用於未知的中間checkpoint**：某個文件可能只在訓練中段進榜，卻不在兩端head。中段只報這個固定聯集中的score、margin、crossing和變化，不稱為全庫rank或泛化品質。端點準備須用已保存CSR重算scores，核對原全庫top100和gold cutoffs；若不符，先停止資料／尺度診斷，不啟動GPU。

## 有界軌跡方案

1. 只重放NG71 D原始192個optimizer updates／768個TRAIN exposures，仍是同384題兩遍，原config、order、A0/A96 witnesses、teacher targets、FP32、RMS、trunk/head LR、clip和AdamW moments均不改。這是192次**歷史更新replay**，不虛稱「沒有執行optimizer」，但零新增科學模型訓練，不接續部署。先不重放U整條曲線，以免將已失敗的uniform對照擴大。
2. 分兩段0→96、96→192，每段開始使用原始base或已封存D96 model+optimizer。第一段終端必須與原D96的完整model SHA和optimizer fingerprint一致，才開始第二段；第二段同樣匹配原D192。每個實際step都核對舊scores／loss／score gradients／update norm，容忍度沿NG75。任何不一致保存attempt、停止解釋，不自行放寬。
3. 在0、16、32、…、192共13個固定checkpoint觀察同24題、相同聯集；96重複邊界亦須一致。observer只做no-grad編碼和固定score計算，assert沒有改變model、optimizer、RNG狀態，也不得將sentinel加入loss。query/document皆用當時**同一共享模型**，不混版。
4. 保存所選query/document sparse codes、所有fixed-pool scores／gold margins、各步TRAIN組成與原始loss trace、數值一致性／模型指紋和資源回執。對每個pair以 $m_{16j}-m_{16(j-1)}$ 檢查telescoping到端點的總變化，找出何時、哪一類batch附近開始失去優勢。這是時間定位，不把某個16-step區間或域標籤當作無反事實干預的因果責任。
5. 先不在每個checkpoint計算5226個完整Jacobian。NG75已給出局部方向可量測的證據；本輪先用廉價forward觀察，只有軌跡指出必要位置才另設有界參數診斷，不一開始堆疊所有工具。

原D的兩段worker分別354.84／359.50秒，包含既有收尾。新observer每次只編碼2974文件和24query，不重編全庫。單GPU，CPU4、RSS16GiB、GPU20GiB、host available24GiB、disk free40GiB，每段1800秒。啟動前查空閒GPU；先測step0 observation與首16步，再以完整剩餘觀察次數、訓練時間、hash／tracking closure預留360秒及1.5倍計算裕量估時；須低於1740秒，外層1800秒獨立guard不變。若超出則保留測試並重新設計批次化／實作，不刪難題、不調大同一attempt上限。

CPU準備／獨立reduction另以RSS8GiB、host floor12GiB、disk40GiB、900秒規範，不是放寬GPU replay門檻。凍結source、prepared payload、parent依賴和catalog後，完成small-model observer immutability／原update parity／端點rank與telescoping tests，才允許新的獨立執行manifest啟動GPU。`prepared.json`的training/replay標誌保持false，不原地改為true。

## 決策界線

### 執行實作

[ng76_trajectory.py](../../../../scripts/ng76_trajectory.py) 以獨立 execution manifest 承接準備產物。[ng76_trajectory_observer.py](../../../../scripts/ng76_trajectory_observer.py) 使用原 `ng71_execution.train_chunk` 新增的 optional observer hook；預設 `None` 不改變既有訓練。原始 frozen NG71 source 不修改。只增加初始化後與每次原 update 完成後的觀察呼叫，不複製 loss／optimizer 迴圈，也不使用 monkeypatch 替換訓練。

每個觀察點保存 sparse codes、固定池分數、model／完整 optimizer state／梯度／training mode／requires-grad／Python、NumPy、Torch RNG 的前後一致指紋。step0 或段首觀察後先估算整段觀察與歷史訓練時間；首16步及後續觀察以實際最慢 update／observation 重估，包含360秒收尾保留。段末保存並 reload 實際 replay checkpoint+moments，再對原 D96／D192 全狀態核對；96邊界另驗 codes 和分數一致。失敗前的原始 update trace 先落盤，避免觀察器中止時丟失現場。

這些是觀測隔離和 replay 一致性的檢查，不是數值誤差必然不會發生的承諾；任何不符仍保留獨立 attempt。新實作不得因舊訓練函數保留自己的較長 canary 而放寬外層每段1800秒或觀察器1740秒預測門檻。

### 科學決策

- 若正確相對次序在特定階段持續流失，再考慮獨立凍結的D versus D+trusted baseline-margin retention。anchor從允許參訓的TRAIN另選，不從sentinel挑loss；兩組匹配資料與計算曝光，不能把更多監督的收益混作retention效果。
- 若主要是新競爭者滲入／固定witness陳舊而不是既有對比持續受壓，優先修監督候選覆蓋；若觀察不足則明確補足範圍。不要因本輪可測線性就直接安裝PCGrad，或因沒有看到單步crossing就否定長程干擾。
- 後續品質仍以完整BM25+semantic、三域最低線、全部正例、未曝光holdout和native total cost為準。NG75／76均不是已超越dense的新模型，沒有新retention係數得到效果認定。

本方案的端點聯集與telescoping是排序／代數恆等式的工程應用，不宣称新檢索演算法，也沒有引用文獻作未測效果保證。既有NG71–75文獻脈絡仍有效；若後續選擇新loss或gradient干預，另讀原始論文並獨立設計對照。
