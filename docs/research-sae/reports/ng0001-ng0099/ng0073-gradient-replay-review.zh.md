# NG-0073：實際 TRAIN 梯度重放終局審核

日期：2026-09-13 UTC。按[凍結協議](ng0073-gradient-replay-plan.zh.md)，完成 NG71 D 的全部 192 次更新、768 題次、384 題和 1,320 個正例題次的唯讀重放。沒有新模型推論、teacher 推論、optimizer 更新或 DEV／LOCKED_TEST 評分。

## 結果改變了下一步選擇

「soft teacher target 會壓縮已經比 teacher 更大的 student margin」這個數學性質成立，但它**不是已觀察 TRAIN lost pairs 的主要直接梯度方向**。

| TRAIN pilot 域 | lost pair 題次 | pool 未包含 | 不符合 pair 監督條件 | 直接擴大 margin | 直接縮小 margin |
|---|---:|---:|---:|---:|---:|
| FEVER | 50 | 22 | 17 | 11 | 0 |
| HotpotQA | 84 | 27 | 31 | 25 | 1 |
| NQ | 238 | 30 | 61 | 143 | 4 |

這是每個最終 lost pair 的兩次实际訓練曝光，不是獨立 query 數，也不是新增失敗樣本。NQ 與 HotpotQA lost 集合中的 shrink 只占有監督 pair 絕對導數量的 **0.00915%** 與 **0.00159%**。相反，NQ retained 集合有 3,784 次 shrink；所有 NQ pair 題次有 3,821 次 shrink。只看 loss 能縮小 margin 的靜態性質，會錯誤地把大量仍保留正確排序的情況當成主要失敗證據。

把同題其他 pairs 的導數相加後，NQ lost 集合有 192 次 aggregate score-space pressure 要求擴大、16 次要求縮小、30 次因 rival 不在 pool 而未知。即使某 pair 自己和同題總梯度支持正確方向，最終共享模型仍可能失去它的排序優勢。

**界線：**上面的數字來自 TRAIN pilot，不是 NG72 未參訓 sentinel 的梯度。它不能證明 shared-parameter interference、AdamW moments 或跨題更新是根因；也不能排除「其他 retained pairs 上的 shrink」間接改變共同表示。它足以否定「lost pair 被其 own loss 直接壓縮是主要已證明原因」，不足以斷言所有 soft targets 都沒有問題。

## 策略調整

不立即把 one-sided teacher floor 當作主修復，不疊加新的 retention 係數。先完成 NG71 原協議要求、至今尚未隔離的 **uniform pair `w=1` 對照**：同成熟基座、全庫 witness、teacher confidence/soft targets、正例平衡、LR、資料和更新量，只移除 rank-dependent weight。這是在判斷複雜權重是否必要，不是假定 uniform 必定更好。

下一個有界 pilot 仍比較完整 BM25+semantic hybrid 的 front-rank quality、all-positive recall、逐題 harm 和成本代理。對照重用已封存 D，不按結果重挑 seed。NG71 的 NQ 門檻失敗仍有效；不自動擴大或部署。若 uniform 仍呈現相同泛化損失，再以實際 parameter-update／margin 保留的配對實驗分離間接漂移，而不是繼續盲加 loss。

## 執行與審核

- Mac CPU4，wall 6.257 秒，peak process-tree RSS 511,475,712 bytes；exit0、error=null、自有 process group 已關閉。
- Offline ClearML `offline-97f8d0a5a4c34042a90834dd55dc9000` 有 actual-start／closed／passed 回執，未宣稱 online sync。
- 既有 NumPy 與 Torch score-gradient 重放一致；與實際四題累積後記錄梯度最大差 `1.2252482153862765e-8`，在凍結容忍度內。
- [獨立 reviewer](../../../../scripts/review_ng73_gradient_replay.py) 另外從實際 progress 和 witness 重建每個 pair target、係數、signed derivative、coverage、outcome 及全 rival／正例／query 分母；1,320 列全部通過，pair derivative 差 0。輸入和產物 SHA 在前後保持不變。没有重算全庫排名或 parameter Jacobian。
- 新增相關 fixtures 與既有 NG71／72 套件為 `114 passed`。這不是產品完整回歸，也不包含新的 CUDA 訓練驗收。

外部產物：`NG-0073/actual-gradients-v1`，研究來源 commit `8c4f44bb`。重要程式、協議和結論在本倉庫；未刪除任何既有研究。

完成後的控制器review另補上「started receipt寫入失敗也必須關閉已啟動worker」的finally保護和失敗fixture。這是後續來源的生命週期修正；沒有改動已封存v1來源、重跑實驗或改寫上述結果。

| 證據 | SHA256 |
|---|---|
| inputs.json | `9890fc6a951d61868f83a096bcb94fe54852d3a07cc1432e963ceb9d3e784098` |
| audit/complete.json | `18c61eba0d0cffe1c71cf9204950194acd4bf7552d848aed293fd4c17b58bbbe` |
| audit/results.json | `f5fb1fbee165b0d9b112b56bcf983f5fe4baef8694ed52150e9386ca930f9b8b` |
| 外部 independent-review.json | `89af4cb2afbc63540568db06e8c5d07b9fa85212525bfe81bcdba8b5479796a7` |
