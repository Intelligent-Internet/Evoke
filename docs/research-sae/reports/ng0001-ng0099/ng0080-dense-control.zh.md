# NG-0080：凍結與解凍底座的配對讀出訓練

2026-09-13，兩組訓練結果產生前固定。遵守 [主協議](ng0080-bottleneck-campaign.zh.md) 及 [執行紀錄](ng0080-execution-ledger.zh.md)。這是可觀測性診斷，不是替換 sparse 的產品模型，不宣稱兩組中的任何一組是底座能力上界。

## 唯一主要變因

F：凍結成熟 NG3 trunk，只更新共同 dense readout。D：更新同一 trunk 和同一 readout。兩者都從 TRAIN-only mean ridge 初始化，不因 CLS 的診斷分數較高而改選初始化。共同線性讀出失敗不足以證明容量不足；只有配對訓練才回答在這個固定預算下解凍是否有益。

使用同一 12,288 TRAIN、原 panel 加16份共用背景、PPLX mean-pool L2、同一完整文件背景及同一 query/document 角色。以 `NG80/fit-order-v1/` 的 query_id SHA 在每域排序，再按 FEVER/Hotpot/NQ 交錯；各題只曝光一次。每次更新4題，合計3,072更新。query64/document256 為輸入 token 窗口，不是 posting cap。

目標為單題 student 與 teacher 的 centered score-field MSE：只移除共同 query offset，保留相對 margin。兩組都只訓練 semantic dense readout，沒有 BM25 subtraction，沒有成本懲罰；不能把這兩組的品質當成完整 hybrid 效果。其後 S/P 仍須以實際稀疏計分與完整 hybrid 另行驗證。

沿用 FP32、TF32 false、eval-mode、trunk LR5e-6/readout LR2e-5、AdamW及 gradient clip1；fresh document VJP，不重用舊模型的訓練 document codes。固定三個1,024-update chunk，各自<=5400秒；前16更新外推1.5倍+300秒不得超過5100秒。不依中途指標改 checkpoint、loss 或 seed。

## 開始前與繼續條件

每組先做獨立 `train-canary`：實際底座4文件 full graph/VJP 的 global gradient relative L2<=1e-5；一次完整4題 optimizer update；checkpoint、projection、每個 trainable parameter 的 optimizer moments、step以及輸出 reload 完全一致。F 的原模型必須未變，D 的 trunk 必須有變。Canary 更新不被帶入正式訓練；第一個 chunk 重新從共同 parent 初始化。

每個後續 chunk 僅續接同組上一個已閉合 checkpoint 和全部 optimizer moments，記錄所有 query identity、score gradient、實際 token/candidate work。不能把 exposure 相同說成 FLOPs 相同。每個 worker 保留 exit/process-group/tracking/resource/hash；failed attempt 不原地重試，不提高界限。

此單元只訓練，不做中途 quality selection。兩組終止之後，另行凍結同一 TRAIN_DIAGNOSTIC 的 fresh encoding / full-background ranking；先檢查幾何及 loss，再看保留下來的搜尋能力。TRAIN_DIAGNOSTIC 有歷史曝光，不是新的獨立 holdout；全套 mixed-hybrid、其他領域及 native 成本仍待後續分支。
