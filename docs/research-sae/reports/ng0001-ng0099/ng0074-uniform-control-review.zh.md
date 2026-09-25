# NG-0074：Uniform Pair 對照結果

2026-09-13 UTC。按[事先凍結協議](ng0074-uniform-control-plan.zh.md)，lambda2 完成六個 phase、192 次 U 更新及完整背景排名，03:29:50 UTC controller 正常結束。**移除 rank-dependent weights 沒有修好泛化；原 NQ 品質門檻仍未通過。**

## 品質與判斷

評分是固定 BM25 0.1 + learned semantic 0.9 的完整 hybrid，不是只評 semantic。相同 233,009 文件；下表為已曝光的 DEV_NEW，三域各512題。U 和既有 D 使用相同成熟 NG3 基座、384 TRAIN 題兩遍、順序、A0/A96 witness、teacher target/confidence 和 optimizer；只改 rank weight 為1，仍保持 per-positive normalization。沒有重新挑 seed 或重新訓練 D。

| 對照 | macro nDCG@10 | macro all-positive Recall@100 | NQ nDCG@10 |
|---|---:|---:|---:|
| 成熟初始化 | 0.777044 | 0.962175 | 0.682021 |
| NG71 A，舊 pool + CE | 0.776374 | 0.957794 | 0.667716 |
| NG71 D，witness + rank weights | 0.784868 | 0.964788 | 0.660173 |
| NG74 U，相同 witness + uniform weights | 0.785282 | 0.965276 | 0.652854 |
| PPLX dense | 0.830229 | 0.977456 | 0.733946 |

U-D macro nDCG 差 **+0.000414**，paired95% interval **[-0.002499, 0.003420]**；Recall 差 +0.000488，interval [-0.001628, 0.002604]。這不是優勢證據，也不是正式等效性檢定。三域的 nDCG 差分別為 FEVER +0.011310、HotpotQA -0.002748、NQ -0.007319：平均數掩蓋了取捨。

U-A macro nDCG +0.008909、Recall +0.007483，但 **NQ nDCG -0.014862**，違反原定 -0.005 domain floor；不能因 macro 有增益而過關。U 仍落後 dense 0.044947 nDCG、0.012179 Recall。Intervals 固定10000次、seed71071，條件於單一 initialization/order；反覆使用的 DEV 不是新盲測。

| U-D nDCG 差 | TRAIN pilot | 未參訓 TRAIN sentinel | 已曝光 DEV_NEW |
|---|---:|---:|---:|
| macro | +0.012468 | +0.000863 | +0.000414 |
| NQ | +0.008010 | -0.007968 | -0.007319 |

這是本輪最有用的發現：uniform 在參訓題上更好，但 NQ sentinel/DEV 更差。它不支持「換成較簡單權重即可保留基座泛化」，也不能單憑一輪證明 rank weighting 普遍必要。兩種權重都未解決成熟模型前排能力流失，因此停止這條權重掃描，不自動擴大失敗配置。

## 成本與證據界線

U 的 document NNZ 為60,649,898，semantic CSR 為486,131,224 bytes；相對初始化分別是 NNZ 0.957938x、DEV query-DF 0.350651x。D 的對應比值為0.960390x／0.370647x。U 比 D 更低的成本代理仍不能補償 NQ 退步，也不是實測原生查詢速度或完整 BM25+semantic 總成本。

U 實際訓練768題次、192更新，query tokens12,628、document tokens7,961,577、candidate-document evaluations61,074。兩段 worker 共1354.803秒，包含封存／tracking 關閉，不能當成純 optimizer wall time。132次更新觸發 clip；此數字本身不能證明 LR 太大。沒有新增人類監督、LOCKED_TEST 編碼／評分或 production 變更。

NG72 的雙端表示分解、NG73 的實際 score-gradient 重放和本輪 single-factor 控制共同指出：繼續只改 loss 權重的資訊收益很低。**shared-parameter interference 仍是假說，不是已證明根因。**下一步按 [NG75 診斷設計](ng0075-parameter-update-diagnosis-plan.zh.md)，量測真實參數位移對同題和未參訓 TRAIN margin 的作用，再決定是否需要最小 retention 干預。

## 執行與審核

研究來源 commit `bc77843b`。所有六個 phase 的 exit0、error=null、自有 group 關閉、actual-start／closed tracking、complete 及逐檔 SHA 已在遠端核對；source21項、dependencies298項保持。ClearML 使用 offline 回執，不宣稱已同步服務器。結束後 GPU0/3 各1MiB，無本次研究 worker 或 tmux 遺留；未干預同事的 GPU1/2。

遠端 terminal reviewer 核對了全部192更新的 exposure/loss/score-gradient/optimizer continuation、11,520條比較記錄的 query/label/rank metric、U 編碼模型身份、雙路全庫分數校驗回執和獨立 CSC-DF。它不是重新執行模型訓練，也不把 reduced metrics audit 說成新完整模型 inference。

截至03:55 UTC，完整109檔／2,138,731,280 bytes的 Mac mirror 與獨立 terminal reduction 均通過。rsync exit0，這次增量傳輸74檔／1,542,521,262 bytes，未刪除任何檔案。Mac 在 reduction 前後核對全量 inventory、全部 SHA、21 source／298 dependencies與六個 phase回執；沒有多餘的 run 檔案。

Mac 從凍結來源呼叫 `review_ng74_uniform_control.review(base, run)`，重算所有訓練記錄、11,520條 rank metrics、paired intervals和 CSC-DF，再逐字段對照 remote results。數值最大差 `1.3877787807814457e-17`，在預設 `rtol=atol=1e-12` 內；這是獨立於訓練產生器的 reduction 重跑，不是兩套獨立 reviewer 實作。Wall21.257秒，peak process-tree RSS1,819,328,512bytes；exit0、error=null、owned group closed。唯讀 QA 使用 CPU4／RSS8GiB／host available12GiB／disk free40GiB／900秒 envelope，沒有放寬科學訓練界線。

驗證產物保存在 frozen run 外的 `NG-0074/uniform-v1-mac-review-v1`。未在Mac重跑全庫ranking或model inference restore；同一Betty上的分類不是獨立災難備援。Remote source保留，LOCKED_TEST和production不變。本次相關research+documentation回歸 `258 passed / 3.29s`，其中三套文檔檢查134項；另核對82個本地文檔連結及 `git diff --check`，不是產品完整回歸。

| 證據 | SHA256 |
|---|---|
| inputs.json | `7465eca9d3d4decced2ea58d7a483bd0507d8f0f3c8d51ed5b5aed4c616706f2` |
| review/complete.json | `456592dd5fc1f8c86a16ff3fce80b6e07eb0b5776f734939508a57f03083cf59` |
| review/results.json | `59398706799a4fb5cf037028e8409765502e6a1558b0408be9435211f615e702` |
| 外部 uniform-v1-remote-final.json | `c9c3519ab3d96f633e0d78bb9ab3d9be40f17fda0928ac13bee7dc133fcf2334` |
| Mac review/verification.json | `7d93af665a8780db4962ae801c5d5bf4f586d5917588569693022a7d07995a4e` |
| Mac review/complete.json | `7d9824833a12cf0ff751af9388124ec790e621e9c1051df9dec12eb023fdb1d2` |
