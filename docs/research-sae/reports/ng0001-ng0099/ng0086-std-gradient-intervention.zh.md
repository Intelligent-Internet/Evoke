# NG-0086：尺度梯度的前瞻性成對介入

日期：2026-09-18。使用者授權策劃並啟動。研究目標仍是整體 BM25+SAE 品質超越 dense 且降低成本；本輪只處理已定位的訓練機制，不宣稱完成該產品目標。

## 為何是這個實驗

[NG85 診斷](ng0085-loss-diagnosis.zh.md) 的 observational v2 已在第 2840 步捕捉非空 query 的零分母：180 個活化全在 zero-RMS 維度，參數及 raw activation 有限。23 個封存 payload 的 inventory/SHA 通過。原控制自第 2789 步有數值漂移，故不能通過原 `loss-std-gradient-v1` 的 bit-exact 准入。本輪不修改那個協議或重標舊結果，而是建立新的前瞻性對照。

NG71/75 已檢查共享 encoder 的雙端梯度與 VJP；NG77 排除過 batch-context 假比較；NG80 使用凍結的 TRAIN 全局 scale，並非本輪 dynamic stop-gradient std；NG81 對稱融合校準未消除 sparse 缺口。因此不再掃融合權重，也不把這次問題直接歸因於監督不足、模型容量或 high-DF 成本。

對中心化分數 $c=s-\operatorname{mean}(s)$，在 std floor 以上，完整標準化 CE 的梯度滿足 $c^Tg=0$；stop-gradient std 則不保證此性質。[Logit Standardization in Knowledge Distillation 的官方實作](https://github.com/sunshangquan/logit-standardization-KD/blob/eb569ac3783f2eed16fbc2b021101175af4fb2f2/mdistiller/distillers/KD.py#L6) 保留 student std 的 autograd。這是有依據的介入候選，不是 sparse retrieval 成功證據；不移植其不同 epsilon、std estimator 或 temperature 配方。

## 冻結因素與時序

| 項目 | Control | Full-std |
|---|---|---|
| 初始狀態 | NG85 B/85001 checkpoint 2048，包括 AdamW moments | 完全相同 |
| Teacher / order / candidates | 原 RankT5 及原四域 order | 完全相同 |
| Student loss forward | `CE(target, softmax(score / max(std(score), 1e-6)))` | FP32 forward 相同 |
| Student std backward | detach | 保留梯度，唯一科學變更 |
| Readout / lexical | 原固定 RMS；完整 BM25 .1 + SAE .9 | 完全相同 |
| 其他設定 | 原 FP32/eval/TF32-off、trunk/head LR、clip、decay、microbatch | 完全相同 |
| 更新窗口 | 2049--3072，最多 1024 updates | 完全相同 |

這個起點已經漂移，故只測「晚期介入能否改善這個窗口」，不能用失敗否定從底座開始修正，也不能用成功宣稱完整兩 epochs 可行。先做這個有界實驗的理由是保留相同 optimizer/trajectory 起點，直接測到已知失效區間，而非再投幾小時大訓練。

固定保存 2560、2816、3072。**主要同進度品質比較是 2816**，2560 為預先指定的較早觀測，3072 僅在雙臂皆到達時作成對比較。不因中途分數改 checkpoint、停止點或超參數。若某臂較早失效，該主要比較記為不可得，不能事後以最後共同點替代。另一臂可完成自己的固定窗口，不能把不等步數的結果稱為配對成功。

每臂啟動重驗 model/optimizer SHA、四域真實 full-graph/VJP 梯度一致性、兩種 loss 的 FP32 forward 相等。checkpoint 必須保存與重載同一參數和 optimizer；實際 query/document 輸出重載檢查若失敗，不假造 checkpoint 成功。

## 評估與可觀測性

沿用 NG85 salted query-ID hash，預先選各域 32 fit +128 validation，共 640 題，包含原小 panel。Validation 共 512 題但仍是曝光研究 split，不稱獨立泛化；不讀 locked test。這是原 bank 的 candidate-only 評估，不能替代 full-corpus recall、DF 或 native latency。

每臂評估所有成功封存的固定 checkpoint；control lane 額外評估相同 panel 的 NG3 底座及起點 B/2048。沒有模型更新、沒有重選題。每題保留完整候選 ID/scores、hybrid 與 semantic 的 nDCG@10、all-known-positive Recall@10、document NNZ、query 支持內外 mass、分母、兩通道分數 std。Invalid query 以零品質納入總分母並單列，不丟掉困難樣本。文件 NNZ 不是實際索引 DF 或 native 搜尋成本。

逐 update 記錄 gradient norm、loss、score std、兩端 NNZ、raw 支持及 score-space 徑向導數。與 NG85 trace 的差異僅作觀測，不要求跨 GPU/執行上下文 bit-exact。遇到 readout failure 保存當時 raw vector、model、optimizer 和 step/stage。只有「非空有限 query、零分母、全落在 zero-RMS 維度、參數有限」才算本協議預期的支持失效；其餘 NaN/Inf、VJP、IO 或資源錯誤是 execution failure，不升格為科學結果。

主要分析按域等權，paired bootstrap 只描述這 512 題，不因它看似顯著就聲稱獨立泛化。除了 full-std/control 差，必須列相對起點及原底座的絕對差。Teacher、BM25 分數沿原 bank 作參考，不重新校準融合或重新評 RankT5 validation。

## 預算與停止

- 兩個獨立單卡 lane，优先 Lambda2 0/3；每卡啟動前無 compute PID，持 cooperative lock，不驅逐他人、不用 DDP。
- 每臂 train 上限 5400 秒、profile 上限 1800 秒；前 16 updates / profile rows 作保守 ETA gate，不夠就停止保留，不能放寬預算。兩臂並行預估 50--70 分鐘，依實測調整估時。
- 沿用 host available >28 GiB、disk >60 GiB admission；執行中 owned RSS <24 GiB、host available >20 GiB、disk >40 GiB、GPU usage <22 GiB。外部 watchdog 僅處理 owned group。
- ClearML offline actual-start/close 及來源、輸入、每 phase exact inventory/SHA 必須完整。不冒稱已同步到 server。
- 預期支持失效是已捕捉的科學結果，不要求停止另一臂；非預期 execution failure 不自動重試。Phase 結束後只做已凍結的 profile/review，不啟動第二 seed、全長訓練或部署。舊 recurring polling 保持暫停。

## 判讀與後續，不自動執行

1. Full-std 跨過原失敗區間且 2816 品質改善、無新增 invalid：有局部正向證據，但還要從成熟底座重新作完整 matched A/B，至少另一 seed，之後 full-corpus/native 驗收。
2. 不崩潰但 ranking 或 NNZ/支持繼續惡化：只解決了穩定性的一部分，不擴大訓練。
3. 兩臂仍有支持失效：不能否定 loss 導數差異，但證明它不足以修复 fixed-RMS 邊界。下一個單因素才是 TRAIN-only、具有明確正支持契約的校準；需同時評估它改變 hybrid 相對尺度的副作用，不偷偷加 epsilon 或刪掉新維度。
4. 現控制未重現原失效：直接報此反例和 trajectory 差異，不能把歷史失敗當成當輪 control 的結局。

只有品質方向清楚後才重新引入 posting/DF 成本目標。這不是忘記成本，而是避免用成本懲罰遮住數值/梯度問題，再把更小但更差的索引誤判為突破。
