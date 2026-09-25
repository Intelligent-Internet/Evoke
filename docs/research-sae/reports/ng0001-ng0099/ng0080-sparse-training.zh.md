# NG-0080：固定預算的 S/P 真實稀疏訓練

2026-09-14，新完整訓練開始前固定。依據[聯合診斷](ng0080-bottleneck-campaign.zh.md)及[實際准入](ng0080-sparse-preflight.zh.md)，不因F/D結果或canary loss改配方。這是訓練協議，不是品質結果或部署批准。

## 對照與不變項

S直接蒸餾實際semantic score-field：`mean(center(s/a-t)^2)`、prefactor1。P對已含既定lexical的完整hybrid採既有uniform-per-positive protective loss、teacher temperature0.04；不把未標文件當硬負例。兩者是scoring/loss配方對照，不是單因素因果，也不是NG79原D重播。

共同起點為成熟NG3 checkpoint096，參數SHA `b61b0e54e060a49bc6fd5bbab120871fdb4aa67c31483849943c42d1d14c1e9c`。不使用任何F/D或canary權重。12,288 TRAIN各一次、每更新4題、3,072更新；順序、panels、所有正例完全繼承preflight。三域equal exposure，無diagnostic梯度或DEV/LOCKED query讀取。BM250.1/semantic0.9、原RMS、query64/document256、MLM head、FP32/eval-mode/TF32false、AdamW trunk5e-6/head2e-5、clip1不變。沒有新成本loss或posting cap。

只有凍結的TRAIN scale `a=9.667940425864051`，不隨模型或diagnostic重估。舊文件CSR只用於已閉合的準備，實際每次更新均使用fresh文件forward/VJP。原 `ng80_training.py`、Encoder及optimizer primitive必須與成功preflight來源SHA相同。Token/panel教師/正例/lexical來源一律綁定，不重跑詞法校準。

## 執行與驗證

preflight-v4三phase已閉合，兩組各16更新的三域真實VJP relative L2=0、初始score及完整reload通過，所有canary更新丟棄。原預算公式選出共同1,024-update chunk；新圖為train-1024、train-2048、train-3072。兩條單GPU lane僅在0/3空閒並取得cooperative lock時啟動，不改用1/2或驅逐其他工作。

首chunk從原模型與空optimizer開始；其後只接同臂上一個已閉合checkpoint及全部name-bound AdamW moments/step。每chunk核對完整參數hash、optimizer fingerprint、query與document角色輸出exact reload。禁止用F/D專屬projection holder。首次完整未更新batch保留原score容差 `rtol=2e-5, atol=2e-7`；每次fresh VJP仍需同context逐值完全相同。

每更新保存query ID/index、pool、全部teacher/positive/lexical/scores/score derivatives、gradient norm/clip、parameter update L2、query/document NNZ、候選和logical token exposures。Token計數不重複計算VJP replay，不能稱為完整FLOPs。分開記錄optimizer迴圈、checkpoint/reload、tracking close、完整worker/phase wall time；P canary全phase大於compute的現象不靠猜測歸因。

每chunk最初16更新以原1.5倍裕量加300秒外推，必須小於5,100秒。phase上限5,400秒、CPU4threads、RSS16GiB、host available>24GiB、GPU total<=20GiB、disk free>40GiB不變。超限保留failed attempt，不原地重試、不提高界限。Actual-start offline ClearML必須closed；完整exit/process group/精確inventory/SHA另行核對，不將offline記錄稱為伺服器同步。

本單元不評估品質、不挑中間checkpoint、不根據loss改seed/步數/權重。兩臂全部閉合後另行審計全量loss/identity/exposure/optimizer chain，再凍結fresh全233,009文件的initial/S/P完整hybrid、PPLX dense及dense+BM25終點比較。TRAIN_DIAGNOSTIC並非獨立holdout；原生成本、多來源泛化、多seed、context與DF utility分支仍需各自結論。
