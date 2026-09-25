# NG-0079：固定初始 witness 的 384-update 執行契約

日期：2026-09-13。本文件在新訓練結果之前固定控制圖，沿用 [query-breadth protocol](ng0079-query-breadth-plan.zh.md) 的唯一比較和全部品質／成本門檻，沒有追加新 loss 或允許失敗後自動加碼。

## 準備已核對，尚不代表品質提升

`NG-0079/static-witnesses-v1` 的 1,536-query CPU 階段已完成，主迴圈 1,848.76 秒，整階段 1,913.25 秒，peak tree RSS 5,467,893,760 bytes；exit0、error=null、owned group closed，actual-start offline ClearML 已關閉，但沒有宣稱遠端同步。控制器於 13:29:40 UTC 完成。48 個檔案、56,099,100 bytes 已原樣鏡像，沒有刪除或覆蓋舊資料。

初始完整 corpus 的雙路 float64 分數最大差異 1.22e-15；head、selected/all-positive rank 檢查通過，原 384-query witness 的 D 資料完全重現。這不是新 encoder inference，也不是訓練或 held-out 評估。

獨立 reviewer 沒有匯入 generator 的 selection/loss 函數或 Torch。它重建 ID/text 排除、hash selection、R/B orders、domain 配額、全部 positive 與 canonical-text/provenance、原 teacher-pool 分數，以及標量 D loss、pair targets/coefficients/gradient。1,536 queries / 2,680 positives 通過，scalar 最大差異 3.27e-16，歷史 384 個 witness 的 canonical JSON 完全一致。沒有重新做第二次全 corpus 排序；完整 rank 證據仍來自封存的雙路準備階段。

加入固定 global witnesses 後，FEVER 693 / HotpotQA 1,066 / NQ 921 個 positives 都至少有一個可監督 pair，eligible pairs 分別 51,882 / 78,488 / 63,550。這是**監督可用性**而非 retrieval 效果；不能據此宣稱泛化已解決。原始池無監督的 NQ queries 沒有被刪除。

Mac 的有界獨立 review 因 available RAM 觸及既有 24 GiB 下限，在 1.15 秒後安全停止，保留原程序回執；沒有改低下限。改用 lambda2 CPU 執行同一已提交 reviewer，22.40 秒、peak RSS 125,693,952 bytes，exit0/error-null/group-closed，6 個 frozen fixtures 通過。Mac 負責逐檔 SHA 鏡像核對，不冒充已在 Mac 完成相同的數學重跑。review 是獨立 CPU 驗證，不新增訓練 tracking task。

## 訓練控制圖

```text
sealed preparation + independent review + verified mirror
                          |
                       R step96
                          |
                       B step96
                          |
             both exact common-prefix gates
                          |
                 R192 -> B192
                          |
                 R288 -> B288
                          |
                 R384 -> B384
                          |
             close and mirror all eight phases
                          |
        separately frozen terminal encoding/ranking/review
```

每一臂都從 NG3 checkpoint096 fresh 初始化；並非從 K96 或共用 optimizer 開始。前96updates的模型 hash 必須為 `92e574071092f08f6288a980ac27a1f0e17b82b0274acac40d3cf8cc228abd49`，完整 named optimizer-state fingerprint 必須為 `f7726f77dcec15d17c888a6b48ff03168d43fae1fd8d428fc888580d9e820540`。後者是 **state fingerprint，不是 serialization 檔案 SHA**。query tokens 6,314、document tokens 3,953,027、candidate query-document exposures 30,493 同時必須完全一致。這些是已知共同 prefix 的重現 gate，不是新品質結果。

`ng71_execution.train_chunk` 保留原 NG71/77 的 768 exposures 與 midpoint-reference hard guards；新增的 `train_fixed_reference_chunk` 只接受 384×4 或 1,536×1、合計1,536 exposures /384updates，96-update 分段、TRAIN/all-positive IDs、static reference0、正確 optimizer predecessor。兩入口共用原本的 forward、exact VJP、更新、trace、checkpoint/optimizer 和 readout reload 實作，不複製一份新的 optimizer 邏輯。

R/B 採相同 D loss，沒有 loss transform 或 GPU observer。每段 worker 都固定相同 RNG seed71001、eval-mode encoder、FP32/no-TF32；序列仍由 frozen seed79001 manifest 決定。每個後續段恢復相同 arm 的上一個完整 optimizer state，不能只載模型而重設 moments。第一個192段之前，兩個96段都必須安全關閉並通過 exact prefix gate。任一 phase 失敗就停止整張圖並保留 evidence，不跳過失敗臂。

## 上限與後续決策

每 phase1800秒，沿用 `worst observed update × 96 × 1.5 + 180 seconds` 的 canary 預測，必須低於1500秒，預留300秒封存。原 D96 約6分鐘，只用作初始 ETA；此圖8段預計約50–70分鐘，按 actual canary、追蹤、hash 和 mirror 修正輪詢。CPU4、treeRSS16GiB、GPU20GiB、hostavailable>24GiB、diskfree>40GiB 不變。啟動前檢查空閒 GPU 和 cooperative lock，單卡，不驅逐別人的 job。

所有 segment 保留 fresh query/document forward 的 token 和 candidate counts、loss/gradient/clip/update trace、模型及完整 optimizer。每臂是等 query exposure，不是假裝等 token、文檔或 FLOPs。只比較 terminal384，不依中途分數挑 checkpoint。

當前控制器只做訓練，沒有 endpoint encoding/ranking、DEV、LOCKED_TEST 或 production 分支。訓練完成後還須凍結對應384-step的 endpoint實作及獨立 review；不能把既有96/192特定的 evaluator 悄悄當作384版本。沿用原計劃的 exposed TRAIN_SENTINEL、all-positive metrics、bootstrap/domain/Recall/cost門檻；無論本輪結果如何，都不把已曝光資料稱為獨立未見驗證。

## 關鍵指紋

- Preparation input：`f6e0430ea6de939f48c76c153f2f45362d4b7a07761a33a1550e2a647d2fe87c`。
- Preparation inventory：`8cc056c9c5d8f70dd045b1e191292d53abb05e0c31db2c3dbecaf9dd083c49e8`。
- Independent review input：`17cd6f12413cce893a8509196d28677836834ed958a7773dc898e060eff3a9ca`。
- Independent review inventory：`e2bb37e1adadb3cbbc52e411f6e17df654a7c0fcd79c2e55b0f6ea7888df264b`。
- Reviewer source：`de57c031d533eb489147487c8213cd5be8bf1e790208bf3ce96b9f6f7c44e253`。

## 實際啟動與測試回執

執行來源 commit 為 `a497f2793977c65d2bb1835b641f738d219a9e98`。新的 `NG-0079/matched-breadth-384-v1` input SHA 為 `fab3122d21c270dda659bef84e9e35bcaebe5d5a2ada1b8bc448437f97ef7068`，714 個 dependencies、38 個 frozen source/config 檔案在兩端通過 SHA 驗證。沒有修改已封存的 preparation、NG71/77 或正在執行的 source snapshot。

lambda2 對已提交 source archive 執行 NG70–79 與主要文件回歸：**616 passed，11.80 秒**；另在實際 frozen unit 中執行 **68 passed，8.52 秒**。全部有界 subprocess 都 exit0、error=null、owned group closed。source-validation 的 8 個根層 archive／回執檔案已完整鏡像核對，inventory SHA 為 `db776379cc30884b7c81d398a38e65ac226aef9d8441ad242a9fb0b65d46d8c5`；遠端解壓 source tree 可由同一 SHA-pinned archive 重建，不混稱為額外完整鏡像。

2026-09-13 14:08:06 UTC，控制器在 lambda2 **GPU0** 的 `ii42_ng79_breadth_v1` 啟動；controller3553574、首段 worker3553622。啟動前 GPU0 空閒且 cooperative lock 可取得，GPU1 的其他任務沒有被更動。14:11:11 UTC 的首段 R 完成47/96updates，compute137.91秒，reference0；first/24-update canary 均預測652.71秒，低於1500秒門檻。tree RSS 約2.43GB、host available 約110.6GiB、GPU0 約2.52GiB，未跨資源門檻。

actual-start offline ClearML `offline-30e0372ad53c481bb0c64526259d8be3` 尚開啟，**沒有宣稱遠端同步或完整關閉**。啟動觀察的 SHA 為 `7371372693e8bd28e4691efc9a0cf12c8da2f67dbbd1b5ed5638b354b1b6222c`。47個update不是96段完成，更不是通過共同prefix或terminal384品質門檻；以各段最終回執為準。

按目前每update約2.93秒，整張圖加上初始化、序列化、完整SHA檢查和tracking封存，暫估約50–70分鐘。首次後續檢查提前到約15分鐘，以覆蓋兩個新鮮96段的 exact-prefix barrier；若兩段均通過，再按剩餘完整階段的actual時間放長輪詢。所有terminal inference、full-corpus ranking和獨立review仍需另外凍結，當前沒有新的retrieval品質或native總成本結果。

## 後續封存：2026-09-13 15:21 UTC

以上啟動回執是歷史觀察；目前八個 training phases 已全部安全關閉，controller 於 **15:14:17 UTC** 寫入 `all_phases_complete`，原 tmux/controller 已退出。每段 exit0、error=null、owned group closed、actual-start offline ClearML closed，沒有遠端 tracking 同步聲明。兩臂 fresh96 的完整 prefix gate 都已通過，後續依序恢復各自的模型及 optimizer moments。

| Phase | 完整 worker 秒數 |
| --- | ---: |
| R96 | 665.9894 |
| B96 | 387.0662 |
| R192 | 383.3668 |
| B192 | 379.6438 |
| R288 | 380.8937 |
| B288 | 692.2766 |
| R384 | 692.9499 |
| B384 | 379.1576 |

完整 graph 約66分鐘，符合原50–70分鐘粗估；這包含訓練、初始化、checkpoint/optimizer reload、hash 和 tracking 封存，不把這些秒數冒充純 GPU compute。WAN 鏡像另計，並以已關閉 phase 為單位提前搬運，沒有複製仍在寫入的 trace，也沒有因保存較慢而重啟健康工作。最大 phase tree RSS 為3,175,862,272 bytes，未跨越原資源限制。

全量 **179檔、4,779,845,422 bytes** 已經 no-delete/no-overwrite 傳回 Mac，兩次有回執的後續 rsync 均 exit0，逐檔 SHA 和 exact inventory 相符。遠端 inventory SHA：`e60d294e5f36ea1443eabc4edb1b6bf0d2ae895d654d43cc6eaa6373c0daccff`；Mac mirror receipt SHA：`660fda75c656cd413bdb37b6b7b368768357758e54dcee06194d5e84d481018b`。Mac 做的是檔案核驗，不是參數 VJP 或數學重跑。

terminal384 模型 state fingerprint：R `d16c0ec840c70289b254ee7af71c3601f3d53d83264290f7ded3ae261302204c`；B `a0b174bf3b664f0bccdcd90343e8e4386d6b07aa8279d5a0ba4ae1764252084a`。模型已產生不代表品質較好；目前仍沒有新的 terminal nDCG、Recall 或實際 native 總成本結果。

下一階段按另行凍結的 [endpoint contract](ng0079-endpoint-protocol.zh.md) 執行。來源 `fd885dd9536b7214963c3cd737382762417891b7` 的 NG70–79／三組文件回歸為 **659 passed，12.47秒**，完整 source-validation 根層 archive／回執也已 SHA 鏡像。實際 flat frozen endpoint unit 的 **111 fixtures passed，9.21秒**，CPU procedure10.89秒、peak RSS690,393,088 bytes、exit0/error-null/group-closed，未使用 GPU。endpoint input `b95b74c0e7cedd4343c7b870c2b92e8ff32a39157fba6e6df590172e84158a94` 綁定895dependencies／45source及config檔案。先獨立稽核全部實際 loss、曝光和 optimizer chain，再進行新 encoding；原定品質、domain、Recall、NNZ、DF gate 一律不变。

## 實際訓練稽核與 endpoint 啟動

endpoint controller3574825 於15:25:57 UTC在 `ii42_ng79_endpoints_v1` 啟動。首段獨立 CPU audit 於15:30:02 UTC安全關閉，245.47秒、peak tree RSS1,069,420,544 bytes，exit0/error-null/group-closed，actual-start offline ClearML已關閉。全3,072次 query exposure、原 D scalar loss、score gradient、all-positive supervision、固定順序和同臂 optimizer chain 通過。最大 score-gradient 誤差為2.10e-8，使用原 tolerance，沒有放寬。

| 實際工作 | R：重複 | B：廣度 |
| --- | ---: | ---: |
| 不同 TRAIN queries | 384 | 1,536 |
| Query exposures / updates | 1,536 / 384 | 1,536 / 384 |
| Query tokens | 25,256 | 25,774 |
| Document tokens | 15,812,108 | 15,736,972 |
| Query-document exposures | 121,972 | 121,916 |
| 更新迴圈 compute 秒數 | 1,185.38 | 1,189.34 |

兩臂實際 document work 和更新時間相近，query tokens並不完全相同；這支持本輪比較的曝光控制，**不構成排序品質優勢**。獨立 audit 沒有重新計算 parameter VJP 或獨立載入 optimizer tensor；這兩項仍分別依赖原 frozen worker 的 exact VJP/readout 與 optimizer reload 證據，不能混稱。

15:33:57 UTC，R384 正在 GPU0 新編碼完整233,009文件，已觀察到33,888行；document canary預測1,073.21秒，小於原1,200秒門檻。tree RSS約2.10GB，host available約100.5GiB，GPU0約1.05GiB。actual-start offline tracking仍開啟，編碼尚未封存；不能把行數或 canary當作 endpoint完成。觀察SHA為 `e41948d5d34720cc6de9b0cf40523cf9eed19e70aaa39c3d6d7a1675942b2a20`，audit complete SHA為 `8d306e1ec75ec8b431646bb5fdd709d6e322df0e1f2d0ecd3c850375002c671c`。

剩餘兩臂 encoding、五種 full-corpus排名與独立review，暫估還需約60–90分鐘，最終output mirror另計；維持30分鐘檢查，在看到實際後續canary/完整退出碼後再調整。沒有開啟DEV或LOCKED_TEST，沒有中途選checkpoint，也没有追加訓練或修改生產服務。
