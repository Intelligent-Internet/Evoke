# NG-0080：S/P 全庫 Hybrid 終點協議

2026-09-14，在S/P新終點排名產生前固定。沿用[訓練協議](ng0080-sparse-training.zh.md)及[訓練審計](ng0080-sparse-terminal-review.zh.md)，不根據中途loss重選模型。這是歷史曝光的TRAIN_DIAGNOSTIC診斷，不是生產准入或新的獨立holdout。

## 先審計、再全庫推論

兩臂三個chunk及controller均須成功閉合；新單元綁定全部source、inputs和closed outputs的精確SHA。第一個phase執行獨立全量scalar/identity/exposure/checkpoint/optimizer審計，成功後才准入其餘phase。沒有再次訓練或採用canary權重。V1因審計改算FP64 target失敗並保留；V2依[數值契約修正](ng0080-sparse-terminal-review.zh.md#v2-數值契約修正)預約一張空閒0/3卡重現FP32 CUDA target，其餘scalar公式和checkpoint審計仍在CPU，原數值門檻不變。

原成熟initial、S-3072、P-3072分別fresh encode全部233,009 canonical文件及固定1,536 diagnostic query。Query64/document256、FP32/eval/TF32false、query microbatch1/document microbatch4、原RMS及BM250.1/semantic0.9不變。文件outer batch96保持原全庫四文件padding context；沒有posting cap，沒有重新擬合詞彙表或RMS。Query與文件身份、canonical CSR、正值/非空、參數hash必須通過。

GPU lane first依次initial/S，second為P；每條單GPU lane只用當時空閒的0/3並持有cooperative lock，其他卡／任務不碰。CPU preparation可與encoding並行，但排名必須等三個encoding全部閉合。每phase5400秒、CPU4、RSS16GiB、hostavailable>24GiB、GPUtotal<=20GiB、free>40GiB不變；前8outer encoding batches／16 ranking或calibration題以1.5倍+300秒外推，須<5100秒。不為超限放寬界限，失敗單元保留。

## 比較集合

固定比較BM25、PPLX dense、TRAIN-calibrated dense+BM25，及initial/S/P各自semantic、完整hybrid。Teacher cache沿用相同corpus/角色的SHA綁定結果。Lexical query用已驗證Linux原library及ICU en_US_POSIX，原vocabulary/count/RMS代數不變；校準query逐值核對已閉合TRAIN lexical cache。禁止讀取含旧DEV的query.npz。

Dense融合只在既定12,288 TRAIN順序中每32題取一題，共384題、三域各128。這是受資源約束的預先固定TRAIN calibration，不冒稱用了全12k或涵蓋所有融合方法。與1536 diagnostic完全不重合，不用diagnostic選參數。兩路全庫分數各作單題z-score；constant分數路徑為全零。融合為`(1-alpha)*dense_z + alpha*bm25_z`，alpha固定候選`[0,.1,.25,.5,.75,.9,1]`。在完整233,009背景上按TRAIN macro nDCG@10、其次all-positive Recall@100、最後較小alpha決定。包含純dense與純BM25端點，但不保證diagnostic上一定勝出；稱為TRAIN-calibrated比較，不宣稱窮盡competitive融合上界。

Student依原固定產品權重，不參與此weight fitting。這避免用強校準補丁掩蓋S/P配方的實際結果；若有必要比較student calibration，那是另立的對照，不改本結果。

## 排名與驗證

全背景FP64計分、document ordinal tie-break，保存每個positive的全庫rank、top100 ID/score及各指標。Sparse使用CSC active columns，而非逐題掃完整CSR；另一條顯式posting accumulator獨立核對全score vector，絕對誤差<=1e-12。這是數值檢查，不是native WAND性能測試。Dense dot與einsum同樣<=1e-12。

Terminal reviewer重算nDCG與all-positive Recall、gold/head一致性，另以原座標重算top100；融合用已保存的全庫normalization statistics，head誤差<=1e-10（涵蓋z-score amplification，不從實際誤差倒推）。完整背景排名由雙reduction及既有stable-ranking測試支撐，reviewer不重新sort每個完整score vector，不把證據說成雙份完全獨立檢索引擎。

分列semantic teacher-field MSE、teacher top10 overlap、hybrid relevance、分域harm；S/P hybrid各與initial hybrid、dense、dense hybrid比較，另列S-P。沿用分域paired bootstrap10,000、seed80080、95%CI，只涵蓋同初始化和order下的query uncertainty，不能代表多seed穩定性。

記錄完整query/document NNZ、CSR及compressed bytes、literal DF-sum work、encoding walltime與資源峰值。它們只是成本代理：不宣稱native bytes、block visits、P95/P99、含encoding端到端成本已通過。更大多源監督、context、jointmask utility和獨立泛化仍需各自結論；沒有因本輪好看就部署的捷徑。

## V3 Tracking 退出修正與閉合產物重用

V2的實際全量審計和initial/S/P三個全庫encoding已成功閉合；但`prepare-lexical`在results與offline tracking已寫出後，於Python interpreter shutdown因daemon thread競爭stderr鎖而abort，exit=-6。Complete仍為failed，不能因為有results或tracking closed就改判成功。沒有執行calibration/ranking。

已安裝ClearML的`Task.init`即使收到`auto_connect_frameworks=False`仍設定`detect_repo=True`；只有dictionary中的`detect_repository=False`才關閉該分支。現場有repository autodetection失敗訊息與finalizing abort，支持此退出路徑診斷，但不宣稱已重現每一次thread interleaving。V3使用明列所有framework和repository皆false的dictionary，不升級套件、不patch第三方原碼，也不改`task.close()`或process exit門檻。Source/依賴仍由原SHA manifest保全，不仰賴自動repo discovery。參考[ClearML Task原碼](https://github.com/clearml/clearml/blob/master/clearml/task.py)及[自動記錄設定](https://clear.ml/docs/latest/docs/clearml_sdk/task_sdk/)。測試需覆蓋已安裝版本的全部framework keys和三次真實offline subprocess退出，不能只mock `Task.init`。

新V3單元可用`freeze --reuse-v2`，只讀取固定input SHA的V2 `audit-training`、`encode-initial`、`encode-S`、`encode-P`；要求三條對應controller完整閉合、每phase精確inventory/SHA/exit/tracking通過。V2失敗phase和全部source/output亦納入新dependencies，不覆寫或刪除。`reused_phases`明列既有實體路徑，沒有複製大矩陣、新增symlink或假造新phase execution receipt；拒絕路徑逃逸、failed輸出准入、同名新output遮蔽或再執行reused lane。這是跨單元artifact reuse，不是整個V2成功。

V3重新執行lexical preparation、TRAIN calibration及原五個ranking/review phases，數值方法、資料、model、RMS、K、權重、容差、資源／時間門檻全部不變。既有encoding已fresh產生，不為tracking失敗重跑模型。只有新七phase和重用四phase均通過後才解釋完整比較；同時核對原V2內容保持不變。NNZ的增密觀測先保留，不能在排名未完成時推定品質或成本收益。
