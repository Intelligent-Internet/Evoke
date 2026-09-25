# NG-0071：全庫 step0 見證預演契約

日期：2026-09-12。這是 [NG71](ng0071-global-boundary-ranking-plan.zh.md) 的零更新TRAIN診斷，不是四arm訓練，也不替代CUDA驗證。

前置 [NG69獨立終局審核](ng0069-final-breadth-review.zh.md) 已通過。只重用NG69完整233,009文件的共同初始模型及PPLX編碼，不以DEV選擇最好checkpoint。綁定review SHA、原始pipeline／inputs、model state、frozen dependencies、completed/exit/tracking及實際輸出SHA；啟動worker前再次逐檔核對。新attempt保留source/config/dependencies完整清單，原NG69結果與失敗嘗試不修改。

固定96 TRAIN canary由配置v2的provenance-only hash選樣取得。保留全部原pool／正例，加最多64個witness；逐query在完整背景同時產生BM25、完整hybrid和PPLX分數。稀疏雙路使用query×轉置倒排與document×dense-query累加；dense雙路使用float64 matvec及獨立einsum reduction。每個完整分數向量誤差不超過1e-12，top120獨立partition核對，每個最終候選／正例rank再以全庫count核對。tie-break沿用凍結corpus整數document ID，轉成固定寬度字串供selector使用；不改成不同的document-key排序。

PPLX在原pool的分數還必須等於凍結TRAIN targets。模型輸入採canonical corpus文本，記錄每個query/document文字SHA。這一步只重用既有embedding，未執行新的模型推論或optimizer更新；後續訓練仍必須對當前query/doc encoder重新forward。

對original pool、hybrid head、rank80–120、PPLX head、BM25 head、hash樣本各自保存：eligible／masked pair數、每個正例的監督覆蓋、teacher soft target分布、confidence總量、qrel-conditioned priority mass、跨top10／100且被遮掉的priority mass。source-mined或新witness都保持unjudged，不假造人工負例。每域80%正例有eligible比較只是執行門檻，不是改善保證；完整384題與192 batches仍需後續驗證。

正例另記來源、完整token數、256-token實際可見前綴、是否截斷。無支持span judgments時，證據可見性是unknown；「文本被截斷」不能當成「相關證據一定不可見」。此階段不自動改窗口、teacher溫度、label或loss。

使用lambda2 CPU4、treeRSS16GiB、90分鐘上限，host available大於24GiB、磁碟free大於40GiB，不占GPU。16題成本canary預測加1.5倍餘量不得超過5,100秒；控制器每0.5秒核對RSS／deadline。這是預先已有的科學資源界線，不放寬先前CPU單fixture的8GiB／600秒attempt，也不重啟舊attempt。ClearML在實際開始記錄offline／closed及未同步狀態。

```bash
python scripts/ng71_preflight.py --mode witness \
    --research-root "$RESEARCH_ROOT" --output "$NG71_NEW_P0_RUN" \
    --config docs/research-sae/reports/ng0001-ng0099/ng0071-ranking-config.json
```

`scripts/ng71_snapshot.py`是全庫診斷實作，`tests/test_ng71_snapshot.py`覆蓋完整向量、獨立tie/head/tail ranks及easy-pair掩蓋head衝突的fixture。配置仍為training disabled；P0完成也不代表step96快照、CUDA預演、四arm執行器或終局native成本驗證已完成。

## 完整384題 step0 manifest 與 score-gradient 診斷

在96題P0與独立CUDA工程fixture完成後，另開 `pilot-witness` attempt；不覆寫上述 `global-preflight-v1`。選用相同配置v2固定的384題TRAIN、共同初始checkpoint和233,009文件背景，沿用同一分數／rank雙路核對、CPU4／RSS16GiB／90分鐘限制。這是零模型更新、零新encoder inference的共享初始manifest生成，不使用sentinel／DEV／LOCKED作候選採樣。

新manifest額外記錄最終候選的實際初始hybrid分數、固定768曝光順序，以及A/B原pool和C/D擴充pool四個目標的score-space梯度。A/C重算 `.5 * uniform-positive + .5 * softmax(teacher/.04)`，A必須與NG69既存target逐項相符。B/D沿用既有per-positive分母、confidence及rank weight，不因新witness改temperature／label／目標。正例在所有arm完整保留；B/D所有192個四query batches均需存在有效pair，個別零監督query仍保留。

NumPy解析導數與Torch float64 autograd逐項比較，loss與score-gradient絕對差皆不超過1e-12；pair係數加總回候選gradient也需獨立相符。對原pool、hybrid head、rank100附近、PPLX、BM25、hash witness記錄歸一化coefficient mass、絕對pair導數、跨top10／100導數、推升正例與縮小正例margin的pair數。Soft target即使偏好正例，也可能縮小student已經過大的margin；不能把所有eligible pair都稱作正例提升。

這些量是在既有snapshot hybrid scores上的**score-space**導數，不是新encoder forward、parameter梯度、跨query參數抵消、梯度下降後的真實global rank變化或效果預測。Pair導數先按source記錄，再聚合到candidate，兩種L1不可混淆。正式訓練仍要fresh forward及真實optimizer觀察。完整384題manifest通過不等於step96生成、四組控制器或科學訓練已完成。

```bash
python scripts/ng71_preflight.py --mode pilot-witness \
    --research-root "$RESEARCH_ROOT" --output "$NG71_NEW_FULL_STEP0_RUN" \
    --config docs/research-sae/reports/ng0001-ng0099/ng0071-ranking-config.json
```

實作為 `scripts/ng71_diagnostics.py`，相關fixture為 `tests/test_ng71_diagnostics.py`。本階段先以96題實測見證耗時外推384題約8–12分鐘加校驗，16題canary再確認；不使用CUDA單query工程fixture來預測四組真實訓練工期。
