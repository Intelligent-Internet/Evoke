# NG-0088 執行紀錄

日期：2026-09-20。協議：[全背景與固定預算](ng0088-execution.zh.md)，研究設計：[品質保留／成本](ng0088-quality-retention-cost-plan.zh.md)。

## 已啟動範圍

使用者指定 II-42 暫時只使用 Lambda2 **實體 GPU 0**。第一階段 evaluation-only campaign 已在 2026-09-20 07:40:13 UTC 啟動，其他 GPU 不使用、不 fallback。並未啟動壓縮訓練、重開 polling 或更改產品／生產服務。

- Run：`NG-0088/full-background-v1`；tmux：`ii42_ng88_full_background_v1_gpu0`。
- Inputs SHA256：`ba9bd659611f666ef4e0b641051cc911947ab9d6fa2983e75d7cc1aa16dabe60`。
- Controller PID `3692294`；首個 worker PID `3694409`。GPU UUID `GPU-a5759f0b-0e22-e3b7-8243-7945961ba2e8`。
- Base、A/8192、B/8192 全部順序執行，完整固定背景 290,609 篇、全部 2,048 exposed-validation queries。23 個 phase 和 stop rules 已凍結。
- 第一階段 terminal review 只裁決全背景轉移是否支持後續 R/P 預算內訓練；沒有自動啟動未凍結訓練的分支。

## 檢查證據

Mac 數值／控制契約測試 **21 passed、1 skipped**；該 skipped 為只在 Lambda2 執行的 Torch FP32 readout 測試。Lambda2 staged source 和最終 frozen source 均 **22 passed、0 skipped**。Ruff、`git diff --check` 通過。

測試覆蓋正值 TopK、tie-break、不改原矩陣、query 先 mask 再 RMS、零／負／非有限 RMS 拒絕、獨立 sparse dot、literal DF-work 不等於 union、完整正例分母、全排序 oracle、錯誤排名 receipt、TRAIN-only calibration、locked-test inventory 拒絕、manifest 篡改和 GPU 0-only admission。實際首批 document encode 與原 frozen Encoder 的輸出 **bit-exact**；載入後完整參數 SHA 符合既有 base。

27 個 frozen source/config/protocol payload 與 manifest 已鏡像到 Mac 持久化 NG-0088 單位，逐檔 SHA256 通過，catalog 已登記。大型編碼矩陣和後續排名輸出尚未鏡像，不能稱完整 run 已備份。

## 啟動後實測

07:41:41 UTC 現場快照：首個 `encode-base-0` 已完成 **26,688 / 145,304** 篇；編碼計時 70.28 秒，仍在運行。首批 512 篇的保守 phase ETA 為 1,018 秒，低於 5,400 秒硬上限。這是單 shard admission，不是整輪完成時間；A/B 較密矩陣及排名時間仍待實測。

GPU 0 使用 1,574 MiB、利用率 67%；唯一 compute PID 對應本 worker。GPU 1/2/3 各為 1 MiB、0%，没有其他卡上的 II-42 任務。主機 available RAM 116 GiB、磁碟餘量 367 GiB；既有 swap 使用量 1.9 GiB，不把快取造成的低 free RAM 誤判為壓力。執行器持續監測 process-tree RSS、主機 available RAM、GPU 顯存和磁碟安全線。

ClearML 使用 offline task 並保存完整本地 session；`remote_synced=false`。目前沒有新品質結論，不能把前述 readout parity／速度／啟動成功當成 NG88 成果。原始候選分數與新完整背景分數的 anchor 檢查將在 full rank phase 執行。

## 接手續跑

08:17 UTC 直接連線 Lambda2 檢查時，原 controller 已保留失敗退出：
`encode-base-0`、`encode-base-1` 均 closed，下一個
`encode-base-queries` 未啟動。`controller-exit.json` 顯示
`ResourceBusy: resources busy; no worker started`；`admission.jsonl` 中 GPU0
記憶體 1 MiB、無 compute PID，但 utilization 瞬時為 8%，因此觸發保守
resource gate。這是 pre-start admission stop，不是 phase 計算失敗。

08:19 UTC 啟動 `continuation-v1` 控制器，保留原 `controller-start/exit`
不覆寫，從已 closed phase 之後繼續。續跑仍使用直接 `lambda2`、實體 GPU0、
同一 frozen manifest、同一 90 分鐘 phase cap 與原 24 小時 campaign
deadline；沒有換卡、沒有自動訓練、沒有 locked-test access。
`encode-base-queries` 已在 36.14 秒內完成，首批 readout parity 通過。
08:21 UTC `rank-base-full` 已開始，16-query admission ETA 為 1,055.68 秒，
低於 5,400 秒上限，並已寫出 128 / 2,048 validation queries。

## 後續裁決

先觀察原 A 的 hybrid 排序增益在完整背景是否成立，再比較固定 Q96/D768、Q64/D384 和 query-only／document-only 消融。既要看 nDCG，也要看 Recall@100 與 combined DF-work，不能只靠 document NNZ 宣稱成本降低。

若 A 不再改善，停止「保住 A」路線並分析轉移失敗；若改善得到支持，才按已提出計畫凍結、測試 R/P 兩臂，仍只在 GPU 0 順序執行。不事後挑更好的 NG87 中途 checkpoint，也不使用 exposed validation 重新掃 alpha／budget。
