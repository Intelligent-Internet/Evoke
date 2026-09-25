# NG-0080：真實 Sparse 計分的尺度與訓練准入

2026-09-13，在 S/P 新結果產生前固定。延續 [聯合診斷](ng0080-bottleneck-campaign.zh.md)；與 F/D 終點排名並行準備，不依其分數選擇本協議參數。這個單元只有準備及丟棄更新的 canary，不是完整科學訓練或新模型晋級。

## 同一實際輸出，不偷換讀出

S/P 都從成熟 NG3 checkpoint096 開始，沿用 MLM sparse head、query64/document256、原 RMS 與 BM250.1/semantic0.9。用同一12,288 TRAIN、同一固定順序／panels／全部正例；無 diagnostic 梯度，無 DEV/LOCKED query讀取，無重建字典／成本懲罰／posting cap。S約束實際semantic sparse score，P對完整hybrid採uniform-per-positive保護監督。它們是不同scoring/loss配方，不稱為單因素因果，也不把P稱為NG79原D重播。

先 fresh encode12,288 TRAIN query。既有全233,009文件 initial sparse CSR只作固定準備參考：逐檔SHA綁定，同一原全庫四文件batch的64個固定TRAIN來源文件做support/value bit-exact重現。不能拿不同candidate batch的逐座標誤差冒充實作不一致；[NG77上下文反證](ng0077-batch-context-diagnostic.zh.md)仍有效。實際訓練文件永遠fresh forward/VJP，不拿此cache提供更新後文件梯度。

## 尺度的明確選擇

只用12,288 TRAIN panels，在共同未更新模型上計算

$$
a=\sqrt{\frac{\mathbb{E}_q\operatorname{Var}_{d\in C_q}[s_0(q,d)]}
{\mathbb{E}_q\operatorname{Var}_{d\in C_q}[t(q,d)]}}.
$$

各query等權，三域人數相同。分母／分子必須有限且正；不按relevance、diagnostic或下游勝負選a。這只是把初始分數差的均方尺度對齊，不是回歸擬合teacher方向；另報每域variance和cross-covariance，不能把variance相等當成幾何保真。凍結後不重估a、不改RMS或serving權重。

S的實際loss明定為 `mean(center(s/a - t)^2)`，prefactor=1。它是先前備忘錄 `mean(center(s - a*t)^2)` 的 `1/a^2` 倍，**梯度大小不是等價的**。選normalized形式是讓teacher單位明確，不保證與P有相同參數更新強度；必须記錄真實gradient norm／更新與clip行為。F/D恰好a=1，不受此差異影響。

P沿用 `ng80_training.protective_loss`：已知positive對nonpositive，僅teacher正margin，temperature0.04，soft target及confidence，不把unjudged改硬負例。其score已包含既定weighted lexical，不能重乘0.1或0.9。

## 實際准入與預算

每臂先從同一parent，對每域固定第一題，以第一個positive加前三個nonpositive作小型full-graph/VJP fixture；只用于數值准入，不改science panels。Global gradient relative L2<=1e-5。然後從未更新parent跑同一前64題、16次AdamW更新，batch4/querymicro1/docmicro4、FP32/TF32false/eval-mode、LR5e-6/2e-5、clip1。保存所有teacher/positive/lexical/scores/gradient、query/document NNZ、token與候選曝光、更新量及完整checkpoint/optimizer reload。全部canary更新丟棄，不能當training checkpoint續用。

首個完整未更新batch的fresh semantic分數對固定CPU CSR參考用原NG77 score容差 `rtol=2e-5, atol=2e-7`；同一context內fresh VJP需逐值相同。不得因失敗放寬比較界限。16update耗時只用于估算下一個固定3,072-update科學對照的資源：從1024/512/256/128/64中選兩組都能滿足 `1.5 * chunk * seconds_per_update + 300 < 5100` 的最大chunk；不根據品質調步數。若最小chunk亦不安全，保留結果，重新規劃而非啟動長任務。

原四線程、16GiB RSS、hostavailable>24GiB、GPU<=20GiB、diskfree>40GiB、每phase5400秒不變。新controller只在空閒GPU0或3執行，cooperative lock，CPU排名可並行。Actual-start offline ClearML／exit／process group／精確SHA閉合必需。只有全部准入通過後，才另凍結完整S/P訓練；沒有training loss改善或canary通過即可部署的捷徑。

## 執行失敗與平台／Locale 修復

`sparse-preflight-v1` 的12,288 TRAIN reference encode成功，產生631,815個query非零值，64個固定文件的原batch context重現通過，沒有optimizer更新。隨後calibrate在載入原Mac arm64 `libtext.dylib` 時於Linux失敗（`invalid ELF header`）；exit=1、owned process group已關閉、actual-start offline ClearML標記failed/closed。這是runtime可執行格式錯誤，不是尺度校準失敗或模型結論。先前197個合成／契約測試沒有覆蓋真實native載入，不再把這類測試當平台准入證據。

v1保持不可變。v2只排程calibrate及S/P canary，透過SHA綁定並直接使用v1已閉合的reference encode，不重新編碼，也不複製它冒充v2新產物。config、TRAIN順序、loss、模型、尺度公式、數值容差和資源界限不變。

獨立 `lexical-runtime-v2` 使用與NG8凍結記錄SHA完全相同的八個C/H檔案，建立Linux原生函式庫。runtime-v1在編譯前因缺少ICU開發標頭/pc檔停止，保留；v2將與主機執行庫相同版本的ICU開發及執行套件解壓到獨立toolchain，使用相對RPATH，不安裝或更改系統套件。Mac只用標準庫/ctypes及原凍結函式庫，按固定TRAIN order選取12,288 query輸出golden token序列；讀取既有TRAIN metadata容器，但不對其中TRAIN_DIAGNOSTIC題做分詞/推論，不載入Torch或模型、不讀DEV/LOCKED query。Linux必須逐題驗證query ID、文本SHA與完整token序列完全相同，並記錄compiler、ICU、build命令和library SHA，才允許calibrate。這證明本輪訓練輸入的跨平台等價，不宣稱所有Unicode輸入永遠等價。原詞彙表、RMS、count/scale代數與文件BM25快取均不重新擬合；不替換歷史Mach-O檔案。

runtime-v2在獨立SSH程序通過全部12,288題，但 `sparse-preflight-v2` 的tmux worker再次被parity gate擋住，仍未開始optimizer。原C tokenizer用 `ubrk_open(..., NULL, ...)` 繼承ICU default locale；已重現tmux的 `en_CA` 將 `F.C.` 產生 `f.c`，而原Mac及獨立SSH的 `en_US_POSIX` 產生 `f`,`c`。因此「同一binary SHA」不足以鎖定詞法特徵。

v3 source validation增加顯式ICU78 locale=`en_US_POSIX`，但未啟動新preflight：新增的歷史BM25分數bit-exact檢查過嚴。獨立診斷中，5,485個共同TRAIN query在正確locale下最大差僅4.44e-16，全部差異低於1e-12；錯誤en_CA則有60題超過1e-12，最大差0.586820。245個單元測試通過但實際整合檢查失敗，此validation保留failed，不能標為全部通過。

v4是明確記錄的驗證契約修正，不是改loss或偷偷放寬模型門檻：完整token序列仍bit-exact；歷史BM25分數的跨平台FP64正數加總、RMS scaling及dot，以query非零項數決定運算數上界 `m=4*nnz+8`、`u=2^-53`、`gamma=m*u/(1-m*u)`，檢查兩條浮點路徑差不超過 `2*gamma*max(actual,reference)`。不從觀測誤差反推閾值；非有限、負值與超界分數一律失敗，reference為零時不容許任意正數漏入。三個ambient locale都要回到同一固定token序列，並核對上述5,485題的原candidate pools；不讀舊DEV query/score容器。

v4仍使用相同runtime-v2函式庫與v1已閉合的reference query，不重新build或encode；研究用wrapper顯式設定並核對ICU78 locale，不依賴shell環境。模型的原fresh score容差、VJP容差及全部science設定不變。這是研究前處理的可重現性修復，沒有修改產品C代碼或部署；失敗attempt和validation均保留。
