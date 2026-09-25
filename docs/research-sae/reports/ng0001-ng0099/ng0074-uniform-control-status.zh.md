# NG-0074：Uniform Pair 對照執行狀態

03:55 UTC結案更新：六個phase已在lambda2封存，controller於03:29:50正常結束；沒有殘留本次worker。完整109檔／2,138,731,280bytes已mirror至Mac，全部SHA與獨立terminal reduction通過。U未通過原NQ門檻，詳見[結果與審核](ng0074-uniform-control-review.zh.md)，不擴大或部署。以下保留啟動與首段觀察，不代表仍在訓練。

2026-09-13 02:51 UTC。**CUDA engineering fixture已正式封存；U科學訓練已在lambda2 GPU0執行至少85／192次更新，首24步成本canary通過。完整訓練、全庫編碼和品質比較尚未完成。**

## 凍結與預檢

遵循[單變量協議](ng0074-uniform-control-plan.zh.md)，只把 NG71 D 的 rank-dependent pair weight 改為1。程式 commit `bc77843b`；重要code、tests和protocol在Git，外部檔案不代替研究結論。

NG71 frozen config、selection、order、A0/A96 witnesses 和已封存比較基準保持。兩個快照各384TRAIN query，共47,593／47,859 eligible pairs。逐項確認候選對身份、soft targets、teacher conflict排除及正例覆蓋一致；全部192個四題batch保持有效監督。所有384題在每個快照的coefficient有改變，符合正在隔離的rank-weight因素。NumPy與Torch導數最大差 `3.885780586188048e-16`。

凍結前research+documentation相關套件 `257 passed`；後續加入NG73 started-receipt失敗清理fixture後，最新為 `258 passed / 3.19s`，不是產品完整回歸。沿用forward、direct/VJP和optimizer/checkpoint kernels；改動為可選uniform pair objective、明確指定終局training barrier/reference，以及沿用bounded supervisor時指定簽定worker來源。NG71預設四組行為保持。

Mac凍結 controls 經rsync exit0傳至lambda2，遠端逐檔source/dependency verify exit0。未把controls副本當作完整訓練產物備份。

## 現場與界線

- Remote run：`/home/huoju/leask/research/NG-0074/uniform-v1`。
- Mac controls：`/Volumes/Betty/II42/development/research/NG-0074/uniform-v1`。
- tmux：`ii42_ng74_uniform_v1`；controller PID3437518，最初preflight worker PID3437649。這些是本次觀察，不是未來可直接kill的永久標識。
- Controller/worker start：02:40:08 UTC；physical GPU0。啟動前GPU0／3各1MiB且0%使用，GPU1／2有同事工作，未使用或干預。
- Preflight ClearML：`offline-b9c58bca44644fa2ac3933f80a8481f8`，actual-start／closed／passed；exit0、error=null、owned group已關閉。Wall366.153秒、peak RSS2,264,891,392bytes，VJP gradient relative L2和max abs皆0，checkpoint/optimizer reload bit-exact，基座state SHA一致；GPU allocation peak1,664,449,536bytes。Offline，不宣稱online同步。
- U0..96 worker PID3439505，02:46:15 UTC啟動；ClearML `offline-58a89fc17d744b939806e73ff92cdc62` actual-start，仍執行中。首24更新cost canary預估700.344秒／96step，通過5100秒門檻；這是帶餘裕預估，不是已完成phase wall time。
- 02:51觀察已85更新，最新單步3.110秒、clip前gradient norm1.3234、實際parameter update L2為0.0087774。Host available118,267,482,112bytes、free disk768,146,374,656bytes；沒有碰到資源界線。
- 原資源與時間界線保持：CPU4、RSS16GiB、GPU20GiB、host available24GiB、disk free40GiB、每phase5400秒，獨立持續監控。

首1／24步canary已通過；96步checkpoint+optimizer restore／封存、192步終局、完整233,009文件encoding/ranking和獨立review尚待完成。不因preflight或loss下降宣稱模型改善。LOCKED_TEST不使用，無production變更。Preflight完整產物正在回傳Mac，尚不能宣稱整輪已mirror。其餘訓練／encoding／ranking／review與tracking關閉暫估還需約40–60分鐘，回傳另计；agent輪詢改為30分鐘，獨立資源／timeout檢查保持原頻率。

本次觀察後preflight回傳已完成：rsync exit0，Mac核對全部13檔／595,909,431bytes、21份source及已知complete SHA，sealed lifecycle通過。外部回執 `NG-0074/uniform-v1-preflight-copy-review.json`。遠端來源保留，未在Mac重新inference restore；整輪NG74仍未完成或全量mirror。

| 凍結產物 | SHA256 |
|---|---|
| inputs.json | `7465eca9d3d4decced2ea58d7a483bd0507d8f0f3c8d51ed5b5aed4c616706f2` |
| math-preflight.json | `28a02049f6ce37f67551da52a2c75c4edcd5040c580b7b37f3c5057d7c9df6bd` |
| train-preflight/complete.json | `10629ccbf52ccdb4cd89d89beb10d98ee51a75c1693c56f2a8433d254b141e41` |
