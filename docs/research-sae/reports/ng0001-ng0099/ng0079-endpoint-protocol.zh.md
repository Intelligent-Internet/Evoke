# NG-0079：terminal384 的 TRAIN-only 評估與独立核對契約

日期：2026-09-13。在任何本輪 terminal 品質結果之前固定執行方法，從屬於 [NG79 唯一比較與門檻](ng0079-query-breadth-plan.zh.md)。[八段訓練](ng0079-training-execution.zh.md) 使用不可變 source snapshot，這份 endpoint 實作不改動正在訓練的程式或資料，也不是增加訓練步數。

## 評估前的硬邊界

只有 `NG-0079/matched-breadth-384-v1`，input `fab3122d21c270dda659bef84e9e35bcaebe5d5a2ada1b8bc448437f97ef7068` 的全部八段成功關閉後，才能 freeze 新 endpoint unit。每段須 exit0、error=null、owned process group closed、actual-start ClearML closed、完整 output SHA inventory；整張圖須 controller 完成、完整無刪除／無覆蓋 Mac mirror 核對。模型和 optimizer 均保留，不能只用 progress 或 results 代替 closure。

新圖的第一段是獨立 CPU 實際訓練 review，通過之前不允許任何新 encoder inference。它用獨立標量 soft-target logistic 實作重算全部3,072 query exposure的loss／4-query accumulation score gradient，核對全部gold分母、固定witness、順序、token/document實測計數、每段同臂model／optimizer fingerprint及predecessor complete SHA。既有 FP32 review容差不變。它不重新計算parameter VJP，也不獨立載入optimizer tensor；這兩者的實際save／restore／query-document replay仍由封存的worker證據負責，不能混稱全面獨立重訓。

## 固定終點與計分

```text
all8 training phases closed + exact verified Mac mirror
                         |
              independent training audit
                         |
          encode R384 -> encode B384 (fresh)
                         |
       both complete before any new quality outcome
                         |
      rank initial / dense / BM25 / R384 / B384
                         |
       independent metrics + head scores + DF review
                         |
          original fixed gate, no automatic expansion
```

完整233,009個document背景；query共1,920個，全為TRAIN。前384為原Pilot，接著384為已曝光但不進梯度的Sentinel，最後1,152為本輪新增TRAIN。保留原768-query inference batch prefix，新增資料只附在後面，避免把不同padding／batch context誤當模型變化。三組分開報告，每domain分別128／128／384題；不得把TRAIN_ADDED擬合混入Sentinel。

只採R384／B384，不評分或挑選中途96／192／288checkpoint。fresh learned query及document encoding沿用原batch96 documents、batch8 queries、tokenizer／RMS／FP32 no-TF32及全部輸入上限。不得借用舊D96的document cache充當384結果。舊 `encode_checkpoint` 的96／192防線原樣保留；新384入口驗證固定TRAIN surface後共用同一encoding engine。ranking亦共用既有雙路float64 full-corpus scorer，沒有候選截斷或另一套metric。

初始、PPLX dense、BM25重用SHA綁定的原始codes，但對這1,920個TRAIN query做完整corpus ranking。兩種分數reduction、全部gold ranks及head/tie-ID一致性門檻仍為1e-12；新增query不會被假裝成未曾曝光的holdout。原768個baseline rows須重現NG77對應結果，防止比較口徑悄悄漂移。所有query／label的lexical index、corpus order和模型role由manifest綁定。

獨立review重新核對全部gold的nDCG@10／Recall@100、query/domain/surface身份和初始baseline；額外用逐coordinate乘積核對每個模型192,000個head scores。這不是第二次獨立全corpus rerank，完整雙路排名驗證仍由scorer封存。另以稀疏coordinate計數獨立核對document NNZ／CSR bytes和每個surface query-DF工作量。它們不是native index總bytes或實際latency。

## 統計、時間與退出

主比較只有384個exposed TRAIN_SENTINEL。paired domain-stratified bootstrap固定10,000次、seed79079，domain macro而非按各domain題數加權。R/B各只有單一初始化與order，bootstrap不估計訓練seed variance。原Pilot／新增TRAIN的擬合和positive harm分開列出，不參與checkpoint選擇。

門檻不變：B-R macro nDCG >=0.005且95%lower>0；Recall95%lower >=-0.002；各domain的nDCG及Recall，相對R及初始都 >=-0.005；B的document NNZ與Sentinel query-DF mean均 <=1.25倍初始。execution review `passed=true` 與科學gate通過是兩個不同欄位。沒有DEV／LOCKED_TEST、lambda/K sweep、production或自動擴訓分支。

每段仍1800秒，CPU4、treeRSS16GiB、GPU20GiB、hostavailable>24GiB、diskfree>40GiB。歷史NG77完整K96 encoding為751秒，768-query dense rank為325秒；本輪背景document數不變，query增加2.5倍，dense rank暫估約813秒。這只是啟動前估計，不放寬門檻。encoding前8batch canary須低於1200秒；ranking前16query canary須低於1500秒。估計整圖約60–90分鐘，依實際完整phase與鏡像時間修正輪詢。

實作、測試和source commit須先於本輪endpoint inference。任何phase、品質或成本gate失敗保留原attempt並停止，不重試掩蓋錯誤、不減corpus、不換分母、不放寬NQ或容差。全部通過也只支持另行凍結replication／獨立未見評估／native總成本驗證，不能宣稱已完成dense超越或直接升級產品。
