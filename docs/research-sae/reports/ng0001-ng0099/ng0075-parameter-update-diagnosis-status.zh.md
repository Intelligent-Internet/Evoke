# NG-0075：實際參數位移診斷狀態

2026-09-13 UTC。對應 [固定方案](ng0075-parameter-update-diagnosis-plan.zh.md)。

**終局：五個phase於04:49:39 UTC全部完成，完整Mac mirror及另一套獨立reduction通過。** 最新結論見 [NG75終局審核](ng0075-parameter-update-diagnosis-review.zh.md)：388/388局部方向匹配，但兩個initial rivals只涵蓋同24題NG72已知60個lost comparisons中的1個。下面保留啟動／canary歷史，不再代表pending狀態。下一步為 [NG76完整端點聯集／軌跡](ng0076-endpoint-union-trajectory-plan.zh.md)，不是增加loss或宣稱NQ已修好。

## 執行前檢查

- NG74 的四臂參照／uniform 對照、完整 mirror 和獨立終端 reduction 已完成；本輪不重跑 NG74，也不擴大失敗的訓練設定。
- 新實作 `scripts/ng75_update_diagnosis.py` 與 `scripts/ng75_update_math.py` 僅執行四次 disposable optimizer replay 及 TRAIN margin/Jacobian 診斷，零新科學模型訓練。產品和部署不變。
- 24 題未參訓 TRAIN sentinel，每域8題，初始 ranking/ID hash 選樣，全部 gold 保留；84個 sentinel pairs。加上實際首 batch probes，D1/U1 各102個、D97/U97 各92個。32個不同 TRAIN query、712個既有 canonical documents 的必要文本，沒有 DEV/LOCKED_TEST inference。
- CPU fixtures 驗證實際 AdamW 位移與 raw-gradient step 不相同、雙路鏈式法則、中心有限差分、tied parameter 去重、no-update、資料身份/原始檔不變、正例分母、錯誤追蹤關閉及有界 phase。最初一個測試用例嘗試覆寫 exclusive-write fixture，被正確拒絕；修正測試本身後通過。
- NG70–75 加 documentation navigation／architecture／published-model 回歸為278 passed（3.20秒）；`git diff --check` 通過。
- 四個 phase 的固定1800秒／CPU4／RSS16GiB／GPU20GiB限制不變。Mac 身份準備沒有 encoder inference；遠端實際 replay 啟動前另查 host available24GiB、disk40GiB和GPU占用。

## 當前界線

04:34 UTC，正式 run `NG-0075/actual-displacement-v1` 已在 lambda2 GPU0 啟動，tmux `ii42_ng75_actual_displacement_v1`。GPU1當時有其他工作，本輪未占用或干預它；啟動前GPU0為1MiB/0%，主機可用111.76GiB、磁碟可用712.55GiB。

- source commit：`66d2ad663488cf385f5d9617ebc1332c19ae9301`。
- manifest：`93087ed7480f320b762d296f9f1e36cf7fa891546cab8b36a6181c14f87aa822`，407項依賴、27項source/payload；本機和遠端SHA核對通過。
- 本地 CPU-only 準備正常退出，16.59秒／peak RSS590,544,896 bytes，owned process group已關閉。其專用限制為CPU4／RSS8GiB／host floor12GiB／disk floor40GiB／900秒，沒有降低遠端實際模型phase的安全限制。
- 另從既有 initial ranking／全部 gold labels 重算 pair身份和標量 BM25 dot，全部吻合，最大lexical margin差`2.22e-16`。input audit SHA：`3cdee3581b2205adadef486517b2fc26697fcb30e1fa966c77867a0c4fefbfad`。
- 28個凍結檔案（2,720,644 bytes）rsync退出0，沒有刪除；遠端NG75 CPU fixtures再跑16 passed（6.61秒）。這只是初始輸入同步，不是終端結果備份已完成。
- 第一個D1實際更新已核對原始trace：gradient norm`2.08439040184021`、實際update L2`0.02021542711196222`。初始化SHA和既有NG3完全一致，更新後另記SHA；不宣稱舊紀錄曾保存該步的完整參數SHA。
- D1的102個probes已輸出，forward重複bit-exact，雙路Jacobian核對通過，最大chain absolute差`4.47e-7`、relative L2`2.83e-8`；GPU allocated peak2,320,772,608 bytes。數值驗證是按預定absolute+relative容忍度，不是只拿absolute差和atol比較。
- 第1/8 pair canary均通過，whole-phase保守估時438.85秒，含52.14秒setup、1.5倍最慢pair時間外推與360秒closure/hash預留。pair計算約16秒，不能用這16秒代替完整phase時間。

**D1的results已寫出，但此快照尚未產生完整phase seal，ClearML關閉／terminal process狀態仍待核對；其餘三個case和終端review未完成。** 尚不報告跨題干擾的科學結論，不以單個case的局部margin取代終端四case比較。完整流程預計25–35分鐘，輪詢改為10分鐘；獨立資源guard維持0.5秒，健康工作不重啟。

本地只讀身份準備沒有挑選 final D/U 表現，也沒有改寫既有凍結目錄。完成後還需全部exit/seal/SHA核對、Mac完整mirror與獨立terminal reduction，才可稱本輪診斷完成。ClearML為真實啟動的offline task，並未聲稱已同步線上server。

04:38 UTC補記：D1現已正常退出並封存，exit0／error null／owned group closed，完整phase395.20秒、peak tree RSS2,201,198,592 bytes；controller已自動進入U1。前述04:34「D1尚待關閉」是歷史快照，不再代表最新狀態。仍須等餘下三個case及review完成，不能提前判定科學結論。

若 canary 或 replay 不符預註冊界線，保存 attempt 並分析原因，不縮小分母、不調大 timeout、不自動重試同一目錄。即使本輪所有診斷完成，也不能把局部 margin 變化宣稱為全庫 Recall/nDCG 或192步最終回退的因果比例。
