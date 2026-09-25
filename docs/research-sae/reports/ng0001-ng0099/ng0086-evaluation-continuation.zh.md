# NG-0086：固定 checkpoint 評估延續

日期：2026-09-18。使用者在終態匯報後授權推進。這是新的不可覆寫單元 `NG-0086/evaluation-v2`，不是新訓練，也不修改 `paired-std-v1` 的失敗回執或協議。

## 範圍

[原協議](ng0086-std-gradient-intervention.zh.md)與[終態紀錄](ng0086-progress.zh.md)確定 control 已保存 2560/2816，full-std 已保存並評估 2560/2816/3072。Control profile 僅因初始 ETA 1859.42 秒略超 1800 秒而停止，不是 checkpoint 丟失或需要重訓。

本次只計算原 640 題的 `base`、`start`、`control-2560`、`control-2816`，共 2560 row；原 full-std 1920 row 在驗證 exact inventory/SHA 後直接重用。題目、候選、split、指標、權重、readout、FP32 與 microbatch 不變，不訪問 locked test，不新增語料或選題。唯一評估執行修正是初始化與逐題推論分開估時。

## 准入與估時

原 run manifest SHA256 固定為 `859dbcdc6aed659f74936551bdd6436e2018723f91b61ea71dec583972135783`。新 manifest 封存所有原 helper、新 runner、本協議與四個原 phase 回執。訓練 phase、已完成 full-std profile 及失敗 control profile 均核对原檔與 inventory/SHA，不將 failed receipt 改成 passed。

先使用 full-std/2816，在原 panel 的四域、fit/validation 各第一題共 8 題驗證新評估程式。原始候選與 query identity 必須一致，raw score `rtol=1e-5, atol=1e-5`，nDCG/Recall 必須在 `1e-12` 內一致。另核對原失敗 profile 已完成的 16 題 base 原始分數與指標。任一不符停止，不能擴容或放寬 parity 門檻。

每個 label 的第 16 題作固定 ETA 檢查：

```text
projected = elapsed_wall
          + 1.5 * (observed_seconds_per_query * remaining_queries
                   + max_observed_model_load * remaining_models)
          + 120 seconds
```

逐題計時從該模型載入及狀態 hash 完成後開始；初始化在已耗時內計一次，未載入模型按已觀測最大 load 估計。最初 CUDA 推論开銷不隱藏，仍留在樣本內。這不是取消時間限制：外部 worker wall cap 仍為 1800 秒，包含輸入驗證、parity、模型載入及評估；每 label ETA 不通過便停，無自動重試。

單張空閒 GPU，預期 Lambda2 GPU 3；再次 admission 與 cooperative lock 後才启动，不觸碰其他工作。沿用 admission host available >28 GiB、disk >60 GiB；執行中 owned RSS <24 GiB、host available >20 GiB、disk >40 GiB、GPU used <22 GiB。ClearML offline actual-start/close，不能冒稱 server-synced。

## 結果與停止

只在完整新 control profile 封存之後，與原 full-std 固定結果配對，逐題驗證 split/domain/候選 ID，一律保留 invalid query 為零品質。使用原 summarizer 的域等權指標、固定 5000 次 bootstrap 與 seed，主要 checkpoint 仍是 2816，2560 次要。Control 沒有 3072，因此該點不產生假的成對比較。

必須同時列出 base、start、兩臂相同進度的品質與支持/NNZ，不能只宣傳相對已退化 control 的改善。512 題 validation 仍已曝光，candidate-only 成果不等於 full-corpus、獨立泛化、DF 或 native latency。

完成後停止並匯報，再依結果决定是否從成熟底座開始完整 matched A/B。本 runner 沒有訓練、最佳 checkpoint 挑選、成本懲罰、模型發布、production deploy 或擴展 seed 的路徑。失敗則保留所有 partial artifacts；不原地重跑。
