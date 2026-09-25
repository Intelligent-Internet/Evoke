# NG-0074：固定 witness 的 Uniform Pair 對照

2026-09-13 UTC，pre-execution protocol。接續 [NG73](ng0073-gradient-replay-review.zh.md)，不把 one-sided soft-target loss 當作已證明的修復。先完成 [NG71](ng0071-global-boundary-ranking-plan.zh.md) 原先要求的 rank-weight 消融。

## 唯一科學變量

新 U 組和已封存 NG71 D 相同：成熟 NG3 初始化、固定 384 TRAIN、128／域、兩遍、768 題次、192 次更新、四題累積、相同順序、相同 A0/A96 全庫 witness、teacher 溫度／soft targets／confidence、全部正例及 per-positive normalization、AdamW／LR／clip、FP32、RMS、BM250.1+semantic0.9、query64/doc256 input tokens、uncapped posting。

唯一變量：每個 eligible pair 的 rank-dependent `w` 改為1，再按既有規則重算 per-positive 分母。不是令最終 coefficient 都一樣：teacher confidence 和正例平衡仍然存在。候選身份、pair eligibility、targets 必須逐項相同；不得因不受監督跳題或增加曝光。

不新增 retention、改 LR、改 teacher 或重新選 seed。D 對照重用已有 frozen 模型／結果，不以新的條件重訓或選擇較好的 D。這是單 initialization/order 的條件比較；若要確認廣泛泛化，仍需新受控重複和獨立 held-out 評估，不能把反覆使用的 DEV 稱為盲測。

## 執行與驗收

1. 凍結相關來源、測試、本協議、原 NG71 config/selection/order、A0/A96 witness、來源依賴與終局 review 指紋。CPU 逐題比較 metric/uniform 的 pool、indices、targets、監督覆蓋以及 NumPy/Torch 導數；每個四題 batch 必須有監督。改動只涉及 research helpers 的可選 uniform objective、明確的訓練 barrier／reference 參數，NG71 default 行為保留。
2. 先在一題既有 TRAIN engineering fixture 跑 CUDA direct/VJP 梯度、一次 disposable AdamW 更新、checkpoint/optimizer reload。它不是科學 checkpoint，不得拿來作 U 初始化。reuse 既有 NG71 model-check；uniform 改變係數準備與記錄名稱，不改 forward/backward kernel。
3. 在 lambda2 空閒的一張 GPU 上按96+96更新執行。每個 phase 執行前再檢查GPU，取得既有 cooperative lock，不占用同事GPU1/2。持續維持 CPU4／RSS16GiB／GPU20GiB／host available24GiB／disk free40GiB，phase5400秒；訓練1／24更新 canary 仍用5100秒門檻。
4. 所有 U 訓練與 optimizer continuation 已封存後才編碼完整233,009文件與固定2304題：384 pilot TRAIN、384未參訓 sentinel TRAIN、1536已曝光 DEV_NEW。不編碼、計分或選擇 LOCKED_TEST query。不做 U96中途DEV或checkpoint selection。
5. 沿用完整背景 float64 雙路分數核對、stable-ID top100與全部正例rank；獨立重算指標、全部記錄梯度、optimizer鏈、NNZ／CSC-DF與逐題harm。比較 U-D、U-A、U-initial、U-dense。1536 DEV的三域 paired bootstrap 固定10000次／seed71071。

原 NG71 對 A 的品質門檻保持：不因 U 而放寬 NQ 或任何其他域的 floor。U-D 用來判斷 rank weighting 是否有必要；不以宏平均掩蓋域 harm。成本仍區分 document NNZ、query-DF、CSR bytes 與尚未測得的 complete native cost。不以較低NNZ或Recall單項判定勝過dense。

所有 phase 有 actual-start/closed ClearML offline 回執、exit0、group closure和完整SHA才算完成。若失敗，保留原attempt，停止而不覆寫／自动重試；若品質門檻失敗，不自動擴大。預估科學訓練約20分鐘，完整encoding/ranking/review另約20–40分鐘，加上tracking關閉／hash／回傳；使用實際canary更新預估及輪詢，不把optimizer loop當整輪時間。

## 事先定義的解讀

- U改善NQ且不損其他域：複雜 rank weights 未必值得保留，再以更多獨立訓練題／重複驗證，不直接宣布解決泛化。
- U與D同樣失去前排能力：權重不是足夠修復，優先設計 actual shared-parameter update／trusted baseline-margin retention 配對；不再無根據掃 weight 超參數。
- U變差或成本大增：rank weighting有受控支持，保留D方向但不忘D的未過關NQ；進一步隔離表示漂移與監督不足，不回退到純控NNZ。

這一輪不聲稱直接辨識 parameter interference 的因果比例；它只隔離權重這一個可移除因素。
