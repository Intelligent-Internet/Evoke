# NG-0085：失穩的有界因果診斷

日期：2026-09-17。使用者批准先診斷、再作最小修改。此輪不改 [原 A/B 協議](ng0085-student-ab.zh.md)，不啟動下一個 seed，也不覆蓋 `student-ab-v4`。

## 已確認現象

四個 RankT5 shards 已全部以 exit 0 封存，16,384 個 fit queries、993,661 pairs。完整 inventory/SHA 校驗通過。B/85001 在成功記錄 2,839 updates 後，於 query encode 拋出 `invalid fixed RMS normalization`；A/85001 最後落盤 2,832 updates，被 matched-stage fail-fast 停止。這不是完整 student endpoint，也不是 RankT5 排序無效的證據。

最後 64 個落盤 updates：B 的 candidate score std 中位數 73.054、最高 12,124.079，候選文件平均 NNZ 2,435.68；A 的對應值為 0.23768、1.39003、682.00。兩者窗口終點相差 7 updates，這些是訓練現場統計，不冒稱完全配對的品質或 native cost 比較。

固定 RMS 共 50,265 維，其中 23,888 維為零；其餘均有限且非負。這是校準 support 與後續可訓練 readout 可能脫節的風險，不直接證明這次 failure 是落在 zero-support 還是整個 query 歸零／非有限值。

## 假說與區分方法

1. **歸一化失效**：記錄 raw query NNZ、RMS 支持內外 NNZ/mass、實際分母及參數有限性，區分全零輸出、zero-support 活化與 NaN/Inf。
2. **尺度梯度與前向目標不同**：現有 loss 每次按當前 std 標準化，但 std stop-gradient。分數整體正比例放大時，前向 CE 不變，實際 score gradient 卻可能有非零徑向分量。以解析導數、有限差分及真實 Torch autograd 三方校驗；非零分量不自動證明是 parameter-space 崩潰的原因。
3. **checkpoint / VJP 問題**：驗證保存的 model/optimizer SHA 和 reload state；在 2,048 checkpoint 重新比對 full graph / exact VJP，再要求原始訓練 trace 的分數 std、loss、NNZ、gradient norm 重放一致。
4. **品質與穩定性分開**：用相同候選與固定研究樣本評估各 checkpoint，避免因數值異常就把教師訊號或整條 sparse 路線判死。

## 固定診斷預算

新產物單元 `NG-0085/loss-diagnosis-v1`；程式為 `scripts/ng85_diagnose.py`。沿用 parent source 的 immutable copies，輸入包含 parent manifest、phase receipts、checkpoint 全部 payload SHA 和原 training order。失敗亦保留，不靜默重試。

- **replay**：單卡從 B/85001 第 2,048 步 model + optimizer 精確恢復，只按原 order 執行到第 2,864 步或第一個異常。最多 816 updates、90 分鐘。觀測 hook 不改 forward 或梯度；第一次數值失效時保存 raw query、model、optimizer 和 failing IDs。只有 trace 匹配且 failure point 一致，才能稱為原失敗的重現。
- **profile**：底座、A/1024、A/2048、B/1024、B/2048，共五個預先指定 checkpoint。按 query ID 的固定 salted hash，從每域選 16 fit + 16 validation，共 128 題；其中 validation 64 題不參與梯度，但已屬曝光研究資料，不稱 independent holdout。每題只評原 bank 的完整候選，保存 candidate nDCG@10、all-known-positive Recall@10、raw query 支持、NNZ 和 score-space 梯度，最多 30 分鐘。
- 不評 locked test，不重編全部 290,609 文件，不把 candidate 指標當 full-corpus recall。所有 bank known positives 必須保留；invalid query 明列，不能刪除或只對成功子集報總體改善。
- Lambda2 GPU 0 / 3 有空才啟動，最多兩個獨立單卡；不碰其他研究工作、不用 DDP。每卡共用 lock，啟動需無 compute PID，host free >28 GiB、disk >60 GiB；執行中 owned RSS <24 GiB、host available >20 GiB、disk >40 GiB、GPU memory <22 GiB。外部 watchdog 只結束 owned process group。
- ClearML 明示 offline task，保存 start/close，不宣稱已同步。

## 後續決策

本輪沒有授權自動換 teacher、增加資料、掃權重或加 epsilon 掩蓋錯誤。先取得重現與品質方向，再凍結單因素候選修正。修正至少需通過梯度測試、同資料對照與跨過原 failure 區間；短 canary 不足以宣稱兩 epochs 穩定。若 replay 不一致，先調查重現性，不能拿另一條 trajectory 的成功當修復。

方法與負結果留主庫；大型權重、原始逐題結果及完整 logs 留外部 NG-0085 單元。此輪沒有 production PostgreSQL 或產品 readout 變更。

## 梯度的靜態證據

令候選分數為 $s$、中心化向量 $c=s-\operatorname{mean}(s)$、維數為 $n$、$\sigma=\operatorname{std}(s)>10^{-6}$，$p=\operatorname{softmax}(s/\sigma)$，目標分布為 $t$。目前 stop-gradient 的實際導數是：

$$
g_{\mathrm{stop}}=\frac{p-t}{\sigma}.
$$

若對完整 std-normalized CE 求導，則為：

$$
g_{\mathrm{full}}=g_{\mathrm{stop}}-
\frac{c\,(c^{\mathsf T}g_{\mathrm{stop}})}{n\sigma^2},
\qquad c^{\mathsf T}g_{\mathrm{full}}=0.
$$

前向 CE 對 $s\mapsto a s$、$a>0$ 不變，但 $c^{\mathsf T}g_{\mathrm{stop}}$ 一般不為零。若為負，score-space gradient descent 具有增加尺度的分量；若為正則具有縮小尺度的分量。這只證明 surrogate gradient 和完整標準化目標的尺度行為不同，不證明 AdamW 作用到共享 trunk/head 後一定膨脹，更不能以此單獨解釋 fixed RMS failure。

CPU 有限差分與 Lambda2 真實 Torch autograd 已對照通過；另驗證 query hook 不改 forward/gradient，且能分辨活化全落在 zero-RMS 支持外的情況。最新診斷測試在 Mac 為 7 passed、3 skipped（三項 Torch 檢查只在 Linux 執行），Lambda2 為 10 passed，包含前向相同而 std 梯度不同，以及失敗 reference 不得改標為成功的檢查。第一個 Linux 測試命令因未加入 frozen parent module path 而缺少 `ng85_data`，補正測試環境後通過，沒有修改科學實作或掩蓋測試。

診斷 manifest SHA：`a961fe58392303f919702b752ebeb36c72360913ac844c8e10724fa75612ad5d`。13 個 frozen source payload 已鏡像到 Betty 並逐項校驗。原失敗訓練的全部大型 payload 不在此宣稱已鏡像完成。

### Fixed RMS 的支持邊界

原 query readout 是 $q(v)=0.9v/(r^{\mathsf T}v)$，其中 $v\geq0$ 為 pooled activation，$r$ 為固定 RMS。如果 $r_i>0$，則 $q_i\leq0.9/r_i$；若 $r_i=0$，該維沒有這個上界。取支持內向量 $u$ 與完全位於 zero-RMS 維度的非零 $w$，令 $v=\epsilon u+w$，則分母為 $\epsilon r^{\mathsf T}u$，而 $w$ 部分的 query weight 隨 $1/\epsilon$ 增長。到 $\epsilon=0$ 時，即使 raw query 非空，分母也可為零。

這是 current readout 的結構性風險，不以這個代數反例代替實際失敗重現。單純修正 std 導數也不能數學上消除這個支持邊界；反過來，直接把 zero-RMS 維度 mask 掉可能刪除新學到的特徵，加 epsilon 或重估 RMS 都會改變校準契約。因此兩個問題要分開做單因素驗證，不能一次合併多個補丁後宣稱原因已找準。

## 固定 Candidate Panel 結果

Profile 已在 168.39 秒內以 exit 0 完成，owned group closed、ClearML offline closed，640 筆（128 題乘五個 checkpoint）均有效，全部 profile payload 已鏡像到 Betty 並按 complete SHA 驗證。`review_ng85_diagnose.py` 作封存後分析，不重新選 query 或 checkpoint。

下表僅為同一 bank 的 **64 題研究 validation**，每域 16 題，並非 full-corpus endpoint 或獨立泛化測試。NNZ 是各題候選文件平均值，不是完整索引 DF 或實際搜尋耗時。

| Checkpoint | Candidate nDCG@10 | All-positive Recall@10 | 文件平均 NNZ | Query 活化 mass 落在 zero-RMS 維度 |
|---|---:|---:|---:|---:|
| NG3 底座 | 0.72538 | 0.79416 | 290.58 | 1.063% |
| A/1024 | 0.59097 | 0.73797 | 818.78 | 0.0117% |
| A/2048 | 0.58183 | 0.71922 | 730.19 | 0.0158% |
| B/1024 | 0.71663 | 0.79413 | 1,184.10 | 16.606% |
| B/2048 | 0.66351 | 0.78539 | 2,529.91 | 52.521% |

同 query、按域分層的 10,000 次 paired bootstrap 僅作此小 panel 的描述：B/2048 對底座的 nDCG 差 -0.06188，95% 區間 [-0.10698, -0.01896]；A/2048 差 -0.14356，區間 [-0.20764, -0.08199]。既有曝光、小題數及固定候選限制不會因 bootstrap 消失。B/1024 比較接近底座，但不能事後挑它冒充本輪固定終點成功。

在獨立的 64 題 fit panel、同一個未更新底座分數上：A target 的 score 徑向導數平均 +0.25749（只有 1.56% 題為負）；B target 平均 -0.24578（71.88% 題為負）。在 B/2048 上，B target 仍為 -0.13874、59.38% 題為負。方向與 A 的分數尺度縮小、B 的尺度增大一致，但這仍是支持機制的觀測，不等於完成 parameter-space 單因素因果實驗。

因此不能把「A 沒有先崩潰」解讀成 A 已學好，也不能把「B 一度較接近底座」解讀成 RankT5 已勝過 dense。下一個修正需同時觀察穩定性、candidate 品質和支持/NNZ 漂移，不能只看 loss 或程序是否跑完。

## 歷史與文獻核對

[NG80 sparse preflight](ng0080-sparse-preflight.zh.md) 使用凍結的 TRAIN 全局尺度，並非 NG85 的逐題 dynamic stop-gradient std；[NG80 數值審計](ng0080-sparse-terminal-review.zh.md) 的已知問題則是 FP32 CUDA target 與 FP64 審計重算不同。這次的支持漂移和梯度尺度交互不能直接歸入已解決的 CPU/CUDA 小誤差，也不是重開一次權重掃描。

Sun 等人的 [Logit Standardization in Knowledge Distillation, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Logit_Standardization_in_Knowledge_Distillation_CVPR_2024_paper.html) 提供以標準化蒸餾 logit 關係、而不強迫幅度相等的相關依據。[作者實作 KD.py](https://github.com/sunshangquan/logit-standardization-KD/blob/eb569ac3783f2eed16fbc2b021101175af4fb2f2/mdistiller/distillers/KD.py#L6) 的 student std 保留 autograd，沒有 detach；teacher forward 則另用 no-grad。不能把本輪 surrogate gradient 當成該實作的等價照搬。

該研究的主要證據是影像分類，且其 std estimator、epsilon 與 temperature 配方和本輪不同，不能據此保證 sparse retrieval 的品質或穩定性。它只支持把「保留 std 導數」列為有依據的單因素候選；若做對照，其他 NG85 設定仍需保持不變，不能順便複製整套不同配方。

## 單因素介入的准入協議

在控制重放尚未完成時預先固定，尚不能稱為已啟動或已修復。新的 `loss-std-gradient-v1` 必須先驗證 v1 profile / replay 全部封存產物、原 trajectory 所有已重放統計誤差均為零、同一步失敗且 model parameters 有限，否則拒絕 freeze/launch。

唯一變更是移除 student std 的 detach，保留原 FP32 forward 數值、population std、1e-6 floor、teacher target、RMS、0.9/0.1 hybrid、optimizer moments、learning rates、clip 及 training order。從同一 B/85001 checkpoint-2048 開始，固定做到 3072，最多 1024 updates / 90 分鐘；原 failure 約在 2840，因此至少跨過該區間。這是局部機制對照，不是从 NG3 重新完成兩 epochs 的 A/B。

介入仍需 full-graph/VJP parity，不能只靠 loss.forward 相同。任何 invalid readout 或資源異常都停止並保留，沒有自動重試、調 floor 或變更 RMS 支持。成功到 3072 才保存／重載完整 checkpoint+optimizer，並按原 128 題 panel 做一次不超過 30 分鐘的 candidate profile；不另選題、不延長步數、不挑 checkpoint。

解讀界線：若能跨過原 failure，只能稱「這個 checkpoint/order 窗口的穩定性改善」；若 candidate 品質或 NNZ 仍退化，不能宣稱目標已達成。若仍失敗，保留這個反證，再區分 fixed-RMS support 與 loss 導數的獨立作用。即使成功，也不自動啟動第二 seed、正式全長訓練或部署。

## Teacher 參考核對

對同一已凍結 panel 的既有 teacher 分數作 CPU 排序，沒有新模型推論，也沒有新增 RankT5 validation targets：fit 64 題的 candidate nDCG@10 為 PPLX 0.72545、RankT5 0.71535、原底座 hybrid 0.65134；validation 64 題為 PPLX 0.79015、底座 hybrid 0.72538。這些數值僅比較同一有限候選集合，不能當作正常全文 dense 或 full-corpus 指標。

至少此 panel 顯示監督分數包含可利用的前排訊號，而 A/2048 學習後反而下降；因此「只是 teacher 不夠強」不足以解釋這輪現象。這也不排除不完整 qrels、candidate selection 或 student 表達能力問題，且沒有給 RankT5 全域優於 PPLX 的結論。

## 嚴格重放停止與觀測協議修正

`loss-diagnosis-v1/replay` 在 1,830.95 秒後以 exit 1 結束，owned group 已關閉。這是診斷的 trace gate 失敗，不是再次出現原來的 RMS exception。第 2049--2788 步共 740 updates 的五組記錄統計均與原 trace 完全相同；第 2789 步首次不同：loss 最大絕對差 2.38419e-7、score std 差 1.75006e-5、gradient norm 差 9.63211e-5（相對約 9.26e-6），query/document NNZ 不變。不能由這些摘要相等推論所有參數 bit-exact，也不能把浮點級前向差異的來源直接歸因於 CUDA。

停止時第一題 raw query 仍有限且非空，415 個活化中只有 8 個位於正 RMS 支持，分母 0.0064943；約 98.99% raw mass 位於 zero-RMS 維度。這強化了支持漂移的證據，但仍沒有觀察到分母歸零。此失敗 attempt 的全部 13 個 replay payload 已鏡像到 Betty 並校驗，`passed=false` 原樣保留。

因此不啟動 `loss-std-gradient-v1`，也不偷偷放寬原 replay 的門檻。另建 `loss-diagnosis-v2` 作**未改 loss 的觀測重放**：驗證並引用 v1 的 failed receipt/SHA，仍從相同 2048 model/optimizer、同 order 執行到 2864 或 readout 失效，90 分鐘 cap 不變。唯一執行層變更是把 trace 差異記錄為觀測，不因差異提前停止；首次超出原 tolerance 時另存更新後 model/optimizer，失效時再保存 raw query 和完整狀態。數值合法性、原 readout、資源與來源完整性檢查不放寬。

v2 若觀察到同類 failure，可以診斷其實際機制；只有完整 trace 無誤差且原步數相同才能標記 exact original reproduction。若軌跡有偏差，即使同一步失敗也不得這樣稱呼。loss 介入仍未獲此控制的准入；若需要對照，須另行凍結前瞻性的 matched control / treatment 協議，而不是把觀測 run 事後升格為 bit-exact 控制。固定五 checkpoint 的 profile 已完成，不在 v2 重複執行。

v2 已在 Lambda2 GPU 0 啟動，首次現場確認到第 2064 步，各記錄誤差為零；這只是啟動快照。Manifest SHA 為 `bcf88bf364b6275a4903239a81fb77661538871220528cb34d69408edfbeec49`，13 份 frozen sources 已鏡像並驗證。最新 Mac NG85 相關測試為 29 passed、3 skipped；Linux 新診斷測試為 10 passed。預估約半小時、90 分鐘硬上限，結束後沒有自動 loss 介入或擴大訓練。

## v2 終局與新介入邊界

v2 於 2026-09-17 21:44:28 UTC 結束，耗時 1937.59 秒（32.29 分鐘），exit 0、owned group closed、ClearML offline closed。`passed=true` 表示診斷正常保存失效證據，不表示模型訓練成功。2026-09-18 重新校驗全部 23 個 phase payload、13 個 source 與原 manifest 通過。

最後成功 update 為 2839；2840 的第一題為 FEVER fit query。原始向量有 180 個非零活化，全部位於 zero-RMS 維度；raw/parameters 均有限，正 RMS 支持活化數為 0，分母為 0。獨立 NumPy FP64 對原始向量與 RMS 重算，分母仍為 0。直接失效機制是可訓練表示移出固定校準支持，不是 NaN、全零輸出或 OOM。

第 2789 步起有 trace 漂移，所以只是同一步、同類錯誤的機制重現，不稱 bit-exact 原 trajectory。原嚴格 loss 介入 gate 沒有放寬。使用者之後另行授權 [NG86 前瞻性成對介入](ng0086-std-gradient-intervention.zh.md)：重新凍結相同起點下的 control/full-std，分開驗證 std 導數的作用；不把本次直接失效機制當作 loss 唯一根因已證明。
