# NG-0087：乾淨底座上的 Full-std 教師對照

日期：2026-09-18。狀態：使用者授權的新研究協議；執行狀態另見 [progress](ng0087-progress.zh.md)。不是修改 NG85/86 的凍結嘗試，也不是模型發布。

## 問題與證據

[NG86 完整配對評估](ng0086-evaluation-results.zh.md) 已表明 std derivative 不是無關細節：主要點相對舊 loss 的 nDCG 增益 +0.057277，支持集漂移與候選文件 NNZ 同時減少。但它从已退化的 B/2048 出發，相對乾淨底座只有不確定的 +0.008933，NNZ 仍為 2.75 倍。繼續救援該 checkpoint 不能回答「正確訓練能否改善成熟底座」。

NG87 回到 NG3 成熟 MLM vocabulary 底座，用共同 full-std 比較 PPLX 與 RankT5。這是恢復 NG85 被數值問題打斷的科學對照，不把新教師當成已證明的突破，也不再盲掃 alpha、DF penalty 或候選覆蓋。

## 數學與可否證預期

完整 hybrid 分數仍為既存 readout 的 lexical 項加上 sparse semantic 項。對每個 query 的候選分數向量 $s$，教師分佈 $p_T$ 由各自教師既存分數按 query std 標準化取得。

$$
L(s)=-\sum_j p_{T,j}\log\operatorname{softmax}\left(\frac{s}{\max(\sigma(s),10^{-6})}\right)_j.
$$

共同修正只讓 $\sigma(s)$ 保留梯度；FP32、population std、既有 floor 與 forward 均不變。當 floor 不啟動、$c=s-\bar{s}$ 時，若原 detached 梯度為 $g_0$，完整導數為：

$$
g=g_0-\frac{c(c^\top g_0)}{n\sigma(s)^2},\qquad c^\top g=0.
$$

這消除了該目標不應具有的尺度方向梯度，不代表消除了 query fixed-RMS 的零支持邊界。Loss 的 scale invariance 也不等於 AdamW 下參數尺度不變或保證稀疏。仍記錄支持、分數 std、NNZ 和真實排名；不偷偷加 RMS epsilon、重估 RMS 或遮罩 unsupported coordinates。

[Logit Standardization KD 的作者實作](https://github.com/sunshangquan/logit-standardization-KD/blob/eb569ac3783f2eed16fbc2b021101175af4fb2f2/mdistiller/distillers/KD.py) 同樣保留 student std 的梯度。該工作是分類 KD，使用的 std/floor 細節也不同；只支持導數設計先例，不是檢索效果保證。本輪主要實證依據仍是 NG86 的受控介入。

可區分的結果：若 A 改善而 B 沒有，優先檢查任務教師轉移而非擴大 RankT5；若 B 改善且 A 沒有，支持新的排序監督；若兩者皆改善，舊數值路徑至少是共同障礙之一；若兩者皆無收益或皆膨脹，不再歸因於資料量不夠，先看已記錄的 TRAIN/validation 與品質／密度差異。這些是後續解釋，不授權自動換目標或無限新實驗。

## 凍結對照

- 新單位 `NG-0087/clean-base-ab-v1`。不從 NG85/86 訓練 checkpoint 接續；兩臂的模型均從成熟底座載入、AdamW state 均為空。
- 底座 `NG-0003/seed-3003/coverage-off/checkpoint-096`，參數 SHA `b61b0e54e060a49bc6fd5bbab120871fdb4aa67c31483849943c42d1d14c1e9c`。
- 重用 NG85 `student-ab-v4` 封存 prepare/bank/四組 RankT5 targets；manifest SHA `5b5f7fa82c7e449e6e01f5309cf833919341fa20a69e3d9a74bd9c0c5ce50387`。不新增 teacher inference，不重新採樣候選。
- A=PPLX normalized mean-pooled dot soft supervision；B=RankT5 first-decoder `<extra_id_10>` logit soft supervision。兩者只在 teacher distribution 不同。
- TRAIN 共 16,384 題，FEVER/HotpotQA/NQ/FiQA 每域 4,096 題；每步各一題、兩個完整 epochs、32,768 exposures、8,192 updates。保持 NG85 order seed 85001，以排除新 order 的額外混淆。
- 原有共同可見 text prefix、候選與所有 known positives、FiQA 空文件隔離均不變。無人工 explicit judged negatives，不偽造負例；cost regularizer=0。
- 完整 BM25+SAE hybrid 才是優化／判斷對象；lexical/semantic 權重 0.1/0.9、原固定 RMS 不變。FP32、eval mode、TF32 off；microbatch query=1、document=4，exact VJP replay。
- AdamW trunk LR 5e-6、head LR 2e-5、weight decay 0.01、clip norm 1。只帶入已驗證的 base/optimizer 配置，不繼承 NG71 過期的 disabled-training/arm/gate 宣告。
- 第二 seed 本輪不自動啟動。第一 seed 先回答整個完整訓練方向是否有希望，第二 seed 是後續可重現性必要條件，不可用來挑有利結果。

## 階段與預算

固定順序：A preflight、B preflight、base profile、A train/profile、B train/profile、CPU paired review。單卡順序執行，任一實際計算失敗停止後續，不自動 retry。GPU 忙碌時不搶占、不終止其他模型；啟動前重新檢查 cooperative lock、程序和利用率。Lambda2 的 inputs/runtime 已就緒，Spark 優先資源當前被其他程序佔用，因此使用 Lambda2 GPU 3。

每臂 preflight 最多 1,800 秒：四域 full-graph/VJP gradient parity、16 個 disposable updates、精確 model/optimizer/query/document reload；預估 $1.5\times t_{16}\times8192/16+1800$ 必須小於 12 小時。preflight checkpoint 不作正式初始化。每臂正式訓練最多 12 小時，每個 profile 最多 1 小時。總執行上限為七階段上限之和 28 小時，另加有限的 hash/CPU review 開銷；實際估時由 canary 回報，不把上限當預計耗時。

主機 admission available >28 GiB、disk >60 GiB；運行期 available >20 GiB、disk >40 GiB、owned RSS <24 GiB、GPU used <22 GiB。每 5 秒 watchdog。每 16 步記錄進度，保存每步四域 query IDs、梯度 norm、raw RMS 支持、NNZ、score geometry；ClearML 明確 offline actual-start/close，不宣稱已同步 server。原 recurring polling 不恢復。

更新失敗保存模型、optimizer、raw query 和失敗階段，不改寫舊 attempt。固定 checkpoints 2048/4096/8192 均做精確 reload；8192 是唯一主要 endpoint，前兩者僅作趨勢解釋，不能事後選峰值。

## 評估與決策

2048/4096 沿用 NG86 640 題 panel（128 fit、512 exposed validation）。base/8192 使用同一 128 fit 加上全部 2,048 exposed validation（每域 512），共 2,176 題。base 與 NG86 640 個原始分數 anchors 做 parity。不得將不同 panel 的均值當成同一條訓練曲線；主要 base/A/B 比較只在共同的全部 validation 上進行。

報告完整 hybrid candidate nDCG@10/Recall@10、pure semantic、PPLX、BM25、各域結果及 NNZ/zero-RMS mass。每個失效 query 保留在品質分母，記為 0，不靜默刪除。比較 A-base、B-base、B-A、各臂-PPLX；5000 次域內 paired bootstrap，固定 seed 87001/87002 分別對應 nDCG/Recall。只作單 seed、已曝光 validation 的描述性區間，不稱獨立統計泛化。

候選品質信號的預定篩選：零 invalid、相對 base 的 nDCG 點差至少 +0.005、區間下界 >0、Recall 差區間下界 >=-0.002。候選文件 NNZ 超過 base 1.5 倍標記成本障礙，不因品質改善就忽略。此 proxy 不是去重全庫 NNZ、DF-work、posting bytes 或原生延遲，不能據此發布。

沒有「所有域都必須勝 dense」的門檻，目標仍為整體 hybrid。即使篩選通過，也只能提出 full-background 與第二 seed 的下一關：同一 290,609 background 的全量重新編碼、global top-k/Recall@100、DF/posting work，加上原底座、pure dense 和 TRAIN 校準 dense+BM25。正式對外 dense 對照還需正常原文輸入，不能用共同裁切 prefix 將 dense 降級後宣稱超越。這些不是本轮已啟動或已完成的評估。

本輪不碰 locked test，不部署、重建產品索引或宣稱 native cost 改善；RankT5 衍生物仍受既有 release-rights 審查限制。

## 實作

[orchestration](../../../../scripts/ng87_run.py)、[training/profile](../../../../scripts/ng87_train.py)、[fixed review](../../../../scripts/review_ng87.py)、[tests](../../../../tests/test_ng87.py)。重要方法／結論留在主代碼庫；大型 artifacts 用既定 NG-0087 外部路徑及 catalog 登記。
