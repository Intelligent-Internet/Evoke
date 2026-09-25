# NG-0087 執行紀錄

日期：2026-09-18。協議：[乾淨底座教師 A/B](ng0087-clean-base-teacher-ab.zh.md)。

更新 2026-09-20：本輪已於 9 月 19 日 05:23:36 UTC 全部完成。兩臂均完成 8,192 updates，profiles 與 sealed review 通過；詳見 [terminal review](ng0087-terminal-review.zh.md)。下文是保留的啟動時紀錄，不是目前步數。下一階段 [NG88](ng0088-progress.zh.md) 已啟動 GPU 0-only 全背景評估；舊 GPU 3 指派僅是歷史紀錄。

新 immutable runner 已凍結並提交 Lambda2 GPU 3 的有界順序 campaign，先執行真實 preflight，尚未宣稱任何 NG87 品質結果。Spark-1、Spark-2 有其他 GPU 程序；Lambda2 GPU 0/1/2 也有其他訓練，GPU 3 通過啟動前資源選擇。使用同一 Lambda2 runtime 與 sealed inputs，可避免遷移資料／軟體再引入變量。

## 凍結與檢查

- 單位 `NG-0087/clean-base-ab-v1`，tmux `ii42_ng87_clean_base_ab_v1`。
- Manifest SHA256：`27d5965fc24524ca6ab90c7bcf60d10b931d029aaa9f8357990e646296f7b361`。
- Mac 相關回歸 45 passed、6 skipped；6 個 Torch 測試按規則在 Linux 執行。Lambda2 的新舊 loss／evaluator／NG87 相關測試 29 passed，無 skipped。
- Ruff 通過；179 個主計劃／導航／NG87 文件內部連結檢查通過；`git diff --check` 通過。
- 22 個 frozen source/config/protocol payload 與 manifest 已鏡像至本地持久化 NG-0087 單位，逐檔 SHA256 驗證。Catalog 已登記；沒有宣稱未來大型 training checkpoint 已備份。
- 配置明確只繼承相同 base/optimizer，排除旧 NG71 的 disabled-training／arm／gate 宣告。兩臂共同 full-std 的入口經測試驗證，異常時會恢復 helper 綁定；舊 frozen helper 不修改。

本轮 controller 只執行七個固定 GPU 階段及末尾 CPU review。任一計算失敗即停止；不自行新增 seed、teacher、loss、全庫推論或延長預算。正式訓練前先通過四域梯度一致性、兩臂各 16 步 disposable canary 和 checkpoint/optimizer 精確 reload。

## 真實 Preflight

兩臂已完成並封存，exit 0、owned process group closed，ClearML offline task 關閉。四域 full-graph/VJP relative gradient L2 均為 0.0；兩臂初始參數 SHA 均精確匹配成熟底座、optimizer state entries=0。每臂 16 步 disposable 更新後的 model/optimizer 和 query/document 輸出精確 reload 通過，不用這些更新初始化正式訓練。

| Arm | Preflight 總秒數 | Owned peak RSS | 保守正式訓練外推 | 硬上限 |
|---|---|---|---|---|
| A / PPLX | 96.91 | 3.28 GiB | 9.70 小時 | 12 小時 |
| B / RankT5 | 91.81 | 3.25 GiB | 8.72 小時 | 12 小時 |

保守外推含 1.5 倍 canary 時間及 1,800 秒餘量，不是預期實際工時。原始步速估計兩臂約 11.6 小時，再加 profiles／封存成本，初估整輪約 12–14 小時；不能把這個估時當成完成承諾或放宽 cap 的理由。

Loader 的 `lm_head.decoder.bias` missing-alias 訊息已核對：凍結 `checkpoint_io.py` 將其重新綁定到已載入的 canonical `lm_head.bias`，拒絕其他 missing/unexpected/mismatched keys。初始完整 state hash 和實际 checkpoint exact reload 均通過，不是任由缺失權重隨機初始化。

## 正式執行

Base 的 2,176 題 profile 已於 2026-09-18 17:33:07 UTC 完成：418.31 秒、exit 0、owned group closed；重疊的 640 個 NG86 anchors 原始分數最大差 **0.0**。這確認本輪擴大 panel 前的底座／資料／readout 一致性，不是新模型品質結果。

A 正式訓練於 17:33:12 UTC 在 Lambda2 GPU 3 啟動，worker PID `2423240`，已實際完成至少 32/8192 updates。正式 initial receipt 再次確認底座 SHA 精確一致、optimizer state entries=0；不是把 disposable preflight 接著練。B 正式訓練尚待固定順序啟動。兩臂共用的 source/config/protocol 與本地倉庫新檔逐檔 hash 一致。

兩臂 preflight 各有 13/21 個小型證據 payload 已鏡像並依封存 receipt 驗 SHA；未鏡像的包含大型 checkpoint／完整 tracking payload，不能稱為完整 run 備份。基線小型證據也已拉回。尚無正式訓練 endpoint 結論；沒有把 canary loss、基線或 NNZ 當成模型改善。

本輪明確區分 preflight、正式訓練、候選評估、全庫評估和產品驗收。原 polling 暫停狀態不變。
