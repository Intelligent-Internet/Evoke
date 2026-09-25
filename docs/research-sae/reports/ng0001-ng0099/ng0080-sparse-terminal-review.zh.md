# NG-0080：S/P 終點訓練審計與評估界線

2026-09-14，在兩臂第二個訓練chunk運行、尚未讀取終點品質之前固定。這是[稀疏訓練](ng0080-sparse-training.zh.md)的獨立後置驗證，不修改任何frozen loss、步數、資料順序或checkpoint。

## 訓練審計

`review_ng80_sparse_fit.py`只准入S/P兩條完整3,072-update圖全部成功閉合的單元。逐phase核對exact inventory/SHA、exit0/error=null/group closed、actual-start closed/passed offline tracking和原資源界限。檢查query identity/order、panel、全部positive、teacher和lexical alignment、4題累積、clip/nonzero update、逐題和chunk的candidate/token/NNZ exposure；token數來自已凍結長度資料，不將VJP replay計成新增監督。

獨立NumPy實作檢查每題loss及對score的導數。S為`e=center(s/a-t)`、`L=mean(e^2)`，四題累積後導數`e/(2*N*a)`。P重新構造teacher-positive margin、temperature0.04、confidence、eligible count及uniform-per-positive weighting，以穩定logaddexp/sigmoid計算BCE和所有score的導數；不調用訓練loss/autograd。教師eligibility保持實際FP32比較，審計運算主要用FP64；採與F/D審計相同`rtol=3e-5, atol=2e-8`，失敗必須保留並診斷，不可依實際誤差偷偷放寬。

### V2 數值契約修正

首次實際audit-v1已失敗封存，不能重寫為成功。原因是原審計把訓練的FP32 CUDA sigmoid target改算成FP64；在接近1的target及大的student margin下，`(1-p)*margin`會把target差異放大到超過既定小loss的驗證界限。12,288筆P記錄中舊門檻1,491筆不符；CPU FP32重放剩2筆，後者的CPU/CUDA target相差1.1920928955078125e-7。三筆固定診斷case用CUDA重放loss及score gradient都與訓練記錄完全相同，詳見execution ledger及`ng80_sparse_numeric_probe.py`。

V2只讓teacher target使用與frozen訓練相同的FP32 CUDA elementwise運算；取得target後，loss與score derivatives仍由獨立NumPy FP64公式計算，不調用訓練BCE/autograd。原rtol/atol、S公式、模型、資料與步數全部不變。這是數值契約修正，不是純FP64 target的獨立驗證，也不是重演完整梯度。科學audit因此需預約一張空閒0/3卡及cooperative lock，只執行小型scalar tensor運算；checkpoint/moment仍在CPU驗證。全量V2審計通過前，不以三筆probe代替准入。

另在CPU讀取完整checkpoint參數及全部AdamW moments/step，重新計算name-bound fingerprint，驗證trunk/head完整覆蓋、learning rates/betas/eps/weight decay及形狀/precision。原encoder註冊順序不是F/D的sorted ParameterDict，不能沿用錯誤排序。檢查同臂checkpoint chain，worker原有exact query/document reload與canary真實VJP證據仍各自保留。

這不是獨立重演全部parameter VJP或AdamW更新，亦不是模型推論。Scalar正確、moment序列化一致，不等於證明品質有效。禁止只審核成功行、忽略某域或用中途loss挑checkpoint。

## 後續評估

兩臂審計通過後，另行凍結fresh full-corpus endpoint單元及通過測試的runner，不能直接把本審計當成其已實作或通過。使用原mature initial、S-3072、P-3072，全部233,009文件和固定1,536 TRAIN_DIAGNOSTIC query，原query64/document256、RMS、BM250.1/semantic0.9均不改。

同時報semantic與完整hybrid，另列BM25、PPLX dense及TRAIN-only校準的competitive dense+BM25。Dense hybrid的校準集合、score normalization、候選權重和tie contract必須在任何diagnostic融合結果前固定；不得用diagnostic選blend或把固定而未校準的弱baseline說成competitive。若實際全庫校準超過budget，拆分有界phase而不是退回小候選panel冒充全庫。

所有標註正例、全文件背景與穩定tie-break保留；geometry、relevance、DF-work與NNZ分開報告。這一組仍屬歷史曝光TRAIN_DIAGNOSTIC，不能推廣成独立holdout、更多監督的泛化收益或完整native成本准入。多來源資料、context、jointmask utility與原生成本仍依[联合診斷](ng0080-bottleneck-campaign.zh.md)分別閉合。
