# NG-0071：排序訓練準備、監督可用性與真實基座預演

日期：2026-09-12，終局入口更新2026-09-13。本文按時序保存準備與執行歷史；**四組科學訓練及29個phase現已完成，最新品質與成本判斷以[終局審核](ng0071-final-pilot-review.zh.md)為準。D的平均收益未通過NQ單域門檻，不擴大或發布模型。** 下文各時間點的「尚未啟動／待評估」保留其歷史含義，不是當前狀態。

本報告延續 [NG71 計劃](ng0071-global-boundary-ranking-plan.zh.md)。不修改 NG69 凍結比較、不選其最佳 seed、不推進生產模型，也沒有讀取或推論 LOCKED_TEST。新增結果的用途是判斷訓練程序能否被正確執行、監督是否可用，而非證明已超越 dense。

## 1. 已完成的實作與驗證

- [Torch 訓練元件](../../../../scripts/ng71_training.py)：balanced soft-pair loss、固定係數與 teacher target、全部正例分母、舊 CE 路徑、共享 trunk/head 的雙端梯度與 document VJP 重播、4-query accumulation、有限 AdamW 更新。無有效監督的整批不允許只做 weight decay。
- [全庫 witness 選擇器](../../../../scripts/ng71_witness.py)：輸入必須是完整 corpus 分數向量；全庫 stable-ID tie-break、原 pool／全部正例保留、前排與 rank100 邊界採樣、來源順序去重、不跨來源補額度、溢出停止。這只是選擇器；獨立全庫分數累加核對與 snapshot manifest 綁定仍屬下一步 execution layer。
- [資料審核](../../../../scripts/ng71_data.py)：校驗 NG59/67 TRAIN、NG69 teacher/corpus 及 NG70 provenance 的 SHA；逐項核對 pool、positive mask、document identity、文本與來源。
- [有界準備程序](../../../../scripts/ng71_preflight.py)：source/config/dependency freeze、actual-start offline ClearML、子程序群組退出、RSS／wall limit、checkpoint/optimizer 重載與 output SHA。這不是四組 pipeline 的完成宣告。

本輪 99 項 NG66–71 測試通過，包括獨立 NumPy score-space 導數、Torch autograd、CE 對照、shared-encoder VJP、全庫排序與候選邊界、正例不作負例、Unicode 來源等價及 fail-closed 檢查。沒有運行與本輪无關的完整 PostgreSQL 產品測試。

## 2. TRAIN 審核：覆蓋充足不代表有用的排序比較充足

固定 hash 選出的 384 題 pilot，各域 128 題；以下統計基於**原始候選池**，尚非新增全庫 witness。

| 域 | 全部正例 | 有 eligible pair 的正例 | 正例覆蓋 | 完全無 eligible pair 的 query | 有效 pair 中 target > .99 |
|---|---:|---:|---:|---:|---:|
| FEVER | 165 | 163 | 98.79% | 0 | 87.95% |
| HotpotQA | 260 | 258 | 99.23% | 0 | 71.19% |
| NQ | 235 | 233 | 99.15% | 2 | 42.27% |

全 6,144 條 TRAIN 中，三域正例 eligible 覆蓋亦為 99% 左右；NQ 有 11 題在原 pool 完全沒有 eligible pair。這些 query 沒有被按難度刪掉。Pilot 固定兩遍、768 exposures、192 個 4-query batches 中，没有整批完全無監督；個別無監督 query 保留零 loss 和原曝光計數。

這不能被解讀為 99% 標籤已驗證正確，也不能推論 99% 召回問題可以解決。`eligible` 僅表示目前規則下至少存在 teacher 偏好正例的一個比較，可能只是容易區分的遠端負例。NQ 235 個 pilot 正例中，107 個屬於 `added_vs_original`；這是來源分類，不是 107 個錯誤標籤。

**對計劃的實質修正：**P0 不能只驗收「每個正例至少有一個比較」。必須按 hybrid head、rank80–120、PPLX head、BM25 head 和 hash witness 分別報告：

1. 全部正例的 eligible coverage、teacher margin、soft target／confidence 分布。
2. 真正跨 top10／top100 邊界的 pair，有多少被 teacher 反向偏好或 provenance 問題遮掉。
3. 被遮掉的 qrel-conditioned metric-priority mass；這是診斷代理，不是真實錯誤率。
4. 零監督／弱監督集中在哪些原始／added 正例、長文本與原本全庫漏召回 query。

目前不調 `.04` teacher temperature：三域飽和度不同，而且全庫 witness 的 margin 分布尚未觀察。若困難邊界也普遍飽和或反向，才在 TRAIN 上另凍結校準／可信 pair 標註對照；不能看到結果後只改 D 組溫度，也不能把 source-negative 或 LLM 判斷冒充人工負例。

## 3. 選樣修正與失敗證據

第一版未先隔離 provenance 歧義，導致固定 sentinel 中出現 2 題來源不明的 FEVER query。審核程序正常完成，但 `eligibility_gate_passed=false`，沒有拿 exit0 冒充科學准入。

配置 v2 將 **NG70 已標記來源 unresolved 的 11 條 FEVER query 在固定 hash 選樣前隔離**，不按 teacher confidence、模型得分或 DEV 成績篩選。結果：384 題 pilot 與 96 題 canary 完全不變；只有 sentinel 中兩題由同一 hash 序列後續的合格 query 遞補。Pilot/sentinel 的 query ID 與正規化問題文字均不重疊。

原 NG69 的 6,144 TRAIN surface 完全不改。若 P2 最終仍要求 6,144 個來源合格、分域平衡的不同 query，需要先解決／有來源地補足這 11 題並重凍結資料協議；不能把 6,133 說成 6,144，也不能用重複題湊數。

另一個資料預演失敗來自兩份來源文本與 canonical corpus 文本的 NFC/NFD Unicode 表示不同。兩份上游檔案各自符合既有 SHA，對應 document key 相同，但個別文本 byte SHA 不同。目前只接受 NFC 等價，不接受 case-fold、空白折疊、fuzzy match 或內容變動。**模型依然讀取未改動的 canonical corpus 文本**，沒有為通過審核重寫任何資料。

## 4. 真實模型，不只 Toy Test

使用相同 NG3 checkpoint、tokenizer、RMS 與 NG59 dependency manifest，49,648,089 個參數，CPU FP32、query64/document256 輸入 token、完整非負輸出、不做 posting cap。

`model-preflight-v3` 使用一個既有 TRAIN query 的兩個文本作工程 fixture，固定人工構造的 pair target `.8`，另核對 CE score gradient。它做 4 次相同 fixture exposure 和 **1 次 disposable optimizer update**；這不是人類監督資料、正式 canary 訓練、品質或泛化實驗。

| 驗證 | 結果 |
|---|---|
| 原始模型 state SHA | `b61b0e54e060a49bc6fd5bbab120871fdb4aa67c31483849943c42d1d14c1e9c` |
| query/doc 前向相對 NG59 公式 | bit-exact |
| 直接 autograd 與 VJP replay | 107 組梯度 tensor，相對 L2 誤差 0，max-abs 0 |
| 舊 CE score gradient | max-abs 誤差 0 |
| 有限 optimizer 更新 | 非零、有限；disposable state `742fa8535d100c784e81aebc5815dfc2c3d9fb7d51d590bb759518d8b6895e03` |
| checkpoint reload | state SHA 和 query 輸出 bit-exact |
| optimizer reload | group 與全部 tensor bit-exact |
| 執行完成 | exit0、owned process group closed、ClearML offline closed |
| 資源 | 7.30 秒、peak process-tree RSS 2,603,892,736 bytes |

環境：Python3.12.14、Torch2.9.1、NumPy2.2.6、Transformers5.15.1、macOS arm64。這不替代實際訓練節點的 CUDA no-op／數值檢查，也不證明所有 input shape 皆 bit-exact。

實作過程修正了 `.9 * query / denominator` 的浮點運算順序；不能把它改成 `.9 * (query / denominator)` 後僅憑代數等價宣稱一致。第一個 model attempt 還遇到 ClearML import hook 在 Torch 初始化中途介入；目前先完成 Torch import，再建立 actual-start tracking，未改動既有 runtime 套件。失敗 attempt 均保留。

CPU 準備使用單獨的 4 threads／8 GiB RSS／600 秒上限，無 GPU。科學訓練原本的 host available >24 GiB、RSS16 GiB、GPU20 GiB、90 分鐘 stage 上限未放寬。

## 5. 證據位置與下一步

Generated evidence 位於本地 storage policy 下的 `development/research/NG-0071/`，已登錄 catalog，不進 Git。主要 source/tests/本報告均留在主倉庫。

| Evidence | SHA256 |
|---|---|
| `data-preflight-v4/complete.json` | `cc774780aef0c93e09549723a10d938923a6dbc89cbd9769dd63ed4f29a39ecf` |
| `data-preflight-v4/source-frozen.json` | `8799fe4ec917f480af016d5c39bdeeb6be3103a8aaaa8547d4a40f0fe5aa27ec` |
| `model-preflight-v3/complete.json` | `65c605f6d56905f4eeecf7e634760670674f15eb3e2680ce15fa39bc68c666b9` |
| `model-preflight-v3/source-frozen.json` | `7d6cb8728221e480256ad0c9e14b7176b58fb154cb1c1c399eda5b2219a496fc` |

各 attempt 只為其封存的 source/config 提供證據；不能把當前工作樹或後續 pipeline 一併當成已驗證。Data v4 與 model v3 的共同選樣／數學配置一致；model v3 另加既有 dependency-manifest anchor，沒有改變 loss 或訓練資料。

獨立重新核對 data v4 的18份、model v3 的23份檔案 SHA、exit0、process-group closed 及 offline tracking closed 通過。另有71項主要文檔 navigation tests及兩份NG71報告的連結檢查通過。

截至 18:13 UTC，NG69 已進入第三個 order seed 的 `encode-B-66067`，其後仍需 `rank-B-66067`、完整退出與獨立三-seed reviewer。不能把這個階段當成全部完成。Direct SSH 當時不可達，經 Spark-1 跳板恢復只讀監測；未重啟健康程序。兩台 Spark 的可用 RAM 不足科學訓練門檻，沒有搶占其他工作。

後續按順序推進：完成 NG69 reviewer 與 matched-background 品質／密度總結；凍結 NG71 全庫 snapshot／score 獨立核對和四 arm 有界控制器；執行96題正式 P0 witness/可見性診斷；條件通過後再做384題四組比較。成本膨脹、全正例邊界傷害及 uniform-pair 消融仍是原定准入要求。**目前沒有 NG71 超過 dense、降低原生延遲或完成整體目標的證據。**

## 6. 19:26 UTC 更新：全庫 step0 見證診斷完成

NG69的 [三seed獨立審核](ng0069-final-breadth-review.zh.md)完成後，在使用者指定lambda2執行 [全庫P0契約](ng0071-global-preflight-contract.zh.md)。source commit `da27650c`，啟動前10份source/config/protocol雙端SHA相符；`global-preflight-v1` 使用96題TRAIN、完整233,009文件背景，重用已驗證初始模型及PPLX編碼，**零optimizer更新、不占GPU**。

| 驗證 | 結果 |
| --- | --- |
| 完整分數雙路、top120、每個候選／正例global rank | 通過；max score error `7.771561172376096e-16` |
| FEVER／HotpotQA／NQ eligible positive coverage | 41/41、67/67、58/58 |
| 已保留原pool與正例的最終pool大小 | FEVER80–84、HotpotQA75–82、NQ69–83 |
| 封存與退出 | exit0、owned group closed、actual-start offline ClearML closed |
| worker wall／treeRSS | 156.88秒／5,329,989,632 bytes |
| 其中96題見證與診斷 | 116.58秒；初始化／輸入校驗等時間另計 |
| Mac回傳 | 22份檔案、1,855,200 bytes；全部SHA重核通過，遠端不刪除 |
| complete SHA | `e5a2051afce353e816c742cb7338cc6f939122c93de115375a398043db4cf2ac` |
| source manifest SHA | `619f62e030a738e95c879092bf68bf1119a3470d20471eddbe772cbb438e4055` |

這次的全局排名是真實全庫排名，不是80個候選的local rank。但它仍是TRAIN可行性診斷，不能推論DEV品質會增加。

**覆蓋和關鍵衝突必須分開。** FEVER hybrid-head中跨top100的未歸一化metric-priority mass約61.43%被teacher反向／平手遮掉；HotpotQA約55.86%。這分別只涉及2／4題具有此跨界比較的query，不能外推為整個域一半的召回都無監督。NQ這32題的58個正例全部已在top100內，沒有head跨top100對；零mass不是證明NQ沒有邊界問題。這些mass也不是經per-positive normalization及confidence後的實際參數梯度。

**截斷並不均勻。** 166個正例配對中27個文本超過256tokens，26個來自FEVER、1個HotpotQA、NQ為0。沒有supporting-span judgments，全部166個仍將「相關證據是否可見」保留unknown，不能把27次截斷說成27個不可學正例，也不能把無截斷當作標籤一定正確。

原定80%eligible-positive門檻通過，仍不直接改teacher溫度／label／窗口。下一步先補完整384題共享manifest及固定batch检查，記錄實際初始hybrid margins／confidence-weighted score梯度，完成lambda2 CUDA雙端VJP／reload及四組有界execution layer，再啟動機制對照。step96刷新、終局DEV和native成本尚未執行；不能把此zero-update成功當成完整P0/P1已完成。

## 7. 19:45 UTC 更新：lambda2 CUDA 工程預演與回傳完成

`cuda-model-preflight-v1` 在 lambda2 的獨立 TITAN RTX GPU0 執行，source commit `6e0fd824`。啟動前核對8份程式／配置 SHA，取得共同 GPU lock，檢查實際占用；沒有使用同事正在運行的 GPU1／2。這與第4節同樣是一個 TRAIN query、兩個文本、固定構造 target `.8`、4次 fixture exposure 和1次 disposable optimizer update，**不是四組訓練，也不是新的人類標註**。

| 驗證 | CUDA 結果 |
| --- | --- |
| 原始 state SHA | `b61b0e54e060a49bc6fd5bbab120871fdb4aa67c31483849943c42d1d14c1e9c`，未改變共同初始化 |
| legacy forward／VJP replay forward | 此 fixture 內 bit-exact |
| 直接 autograd 與 VJP | 107組 gradient tensors；relative L2 0、max-abs 0 |
| 舊 CE score gradient | max-abs `5.960464477539063e-08`，不是 bit-exact |
| checkpoint／optimizer reload | bit-exact |
| worker wall／peak treeRSS | 26.83秒／2,317,668,352 bytes |
| peak GPU allocated | 1,664,449,536 bytes；不是整張 GPU 的全部占用 |
| 結束狀態 | exit0、error null、owned group closed、actual-start offline ClearML closed |
| Mac 回傳 | rsync exit0，23份檔案共596,006,236 bytes；完整 inventory／全部 SHA 與遠端 complete 核對通過 |
| complete SHA | `8688c029c056cba220d40f8d034674c601076fd9b5d3ee260d7d17ac7dbf9b50` |
| source manifest SHA | `88f9300794f399eb0380c4802ad67a865918832b86ea39abfb47792a74f7d75e` |
| copy review SHA | `4f7ae4528d850c4fc46b8b46b0aa48f22564327dfa189368dd15f520041a35b1` |

環境為 Python3.12.12、Torch2.9.1+cu128、NumPy2.2.6、Transformers5.15.1，FP32、TF32關閉。這裡的 exact 是同一 CUDA fixture 各路徑／重載之間，不是宣稱 Mac CPU 與 CUDA 跨裝置逐位相同，也不涵蓋所有候選池大小。Disposable checkpoint 不得作正式訓練初始化；遠端證據保留。

CUDA 控制器保持4 CPU threads、treeRSS16 GiB、600秒上限；另檢查指定 GPU 總占用20 GiB上限，STOP訊號進入自有 process-group 清理。Offline ClearML 已正常封存但未同步 online；這次通過不能當作線上追蹤已恢復。相關研究及文檔測試合計 **244 passed**，`git diff --check` 通過；沒有擴大到無關的 PostgreSQL 產品測試。

接下來仍需完整384題共享 manifest、實際初始 score-gradient／固定 batch 檢查、step96 reference 快照生成及四組有界控制器。96題見證迴圈實測116.58秒，因此同等工作下384題暫估8–12分鐘加驗證；這不是整個訓練流程的工期。工程 fixture 的26.83秒也不能外推為真實四組訓練速度。後續以實際 canary／進度更新估算，按下一個需要介入的階段調整 agent 觸發間隔，控制器的資源／超時監測不隨之放慢。

## 8. 20:14 UTC 更新：完整384題 manifest 與獨立導數審核

`pilot-step0-v1` 使用 source commit `203ee935`，啟動前11份source/config/protocol跨主機SHA相符。lambda2保持CPU4、RSS16GiB、90分鐘界線，不占GPU；worker實測506.71秒，見證／梯度診斷主迴圈467.83秒，peak treeRSS5,383,458,816 bytes。這與預估8–12分鐘相符。25份輸出共17,524,704 bytes完整回傳Mac，遠端不刪除。

完整233,009文件分數／排名核對最大誤差 `1.2212453270876722e-15`；192個固定4-query batches全部有監督，保留768 exposures及全部660個正例。獨立[唯讀 reviewer](../../../../scripts/review_ng71_snapshot.py)重核全部依賴／輸出SHA、固定選樣與曝光順序，重算384題的四arm score-space loss／導數；跨Mac與lambda2最大差 `1.7763568394002505e-15`。先前96題的候選、正例、teacher分數、global rank、source及文本SHA完全相同，沒有看到結果後重抽canary。

| TRAIN 域 | 全部／有eligible比較正例 | 初始rank>100正例 | 正例文本截斷 | 最終pool大小 |
| --- | ---: | ---: | ---: | ---: |
| FEVER | 165／165 | 15 | 100 | 78–84 |
| HotpotQA | 260／260 | 16 | 3 | 75–82 |
| NQ | 235／235 | 4 | 0 | 65–84 |

NQ擴大到128題後出現4個top100外正例；先前32題canary的零漏召回不能外推。660個正例的證據可見性仍全部unknown，截斷不是判定標籤錯誤或證據缺失的依據。

**新增witness並非只帶來容易的比較。** D的hybrid-head加rank80–120 witness分別承接FEVER68.58%、HotpotQA54.57%、NQ57.90%的絕對pair導數量。這是confidence與per-positive normalization之後、candidate相互抵消之前的score-space導數；不是參數梯度占比，也不代表同等比例的效能或召回收益。另一方面，head跨top100的未歸一化priority mass仍有64.86%、57.21%、60.84%被teacher反向／平手遮掉。監督覆蓋100%與困難對手被有效監督是不同結論。

**CE與pair的更新訊號不能直接按loss數值比較。** D/C的candidate score-gradient L1為FEVER0.0834x、HotpotQA0.1574x、NQ0.1280x；D/B則為1.6147x、0.9550x、0.8087x。增加候選會重新分配per-positive分母，並不保證讓pair總梯度變大。D在score-space推升／降低正例分數的數量為161／4、260／0、233／2；soft target有時應縮小student過大的margin，不能把每個eligible pair都叫作正例提升。AdamW、自適應moment、gradient clipping及共享參數會改變實際更新，**不因此把D的LR放大10倍，也不提前宣稱它更好或更弱**。

下一步保留預先凍結的四arm／溫度／LR／資料；step1／24必須看實際參數更新、clipping及fresh-forward分數，之後才做terminal品質與成本比較。這輪已補上[96-update訓練chunk及A96全庫編碼元件](../../../../scripts/ng71_execution.py)、name-bound optimizer完整moment／step／重載校驗，以及A96引用的固定TRAIN identity／全庫rank路徑；**這些元件尚未執行科學訓練，完整有界調度器、終局評估及獨立四組reviewer仍需完成後才能啟動。**

| 證據 | SHA256 |
| --- | --- |
| `pilot-step0-v1/complete.json` | `5a736aedeff4270bec4e070640725d67236db52cdd7f82fc7684dcdc7ae5fc96` |
| `pilot-step0-v1/source-frozen.json` | `405916fd2f7968f4b585642e80dd45467e3d59eb7c929843d21487dead92388a` |
| `pilot-step0-v1-copy-review.json` | `c3b847295d6f32596c5863e122c6ee2691bd82d6c0f40f6eaf631d8cee445629` |
| 唯讀reviewer source | `b69ce3182a0d164522ce07c750f2536217dc84e2615d48467adef6d90077234e` |

初次本地複核因import在回傳目錄生成5份bytecode，被exact-inventory檢查阻止；僅將本輪新生成cache移到指定Tmp，原25份封存檔案未改動。正式reviewer在import前禁用bytecode、拒絕將輸出寫進frozen attempt，再完成上述校驗。這是審核工具的唯讀性修正，不是訓練或資料失敗。ClearML offline task正常關閉，未同步online；LOCKED_TEST仍未編碼或評分。

## 9. 有界四組執行層與終局 reviewer

20:59 UTC：補齊[單GPU有界調度器](../../../../scripts/ng71_pilot.py)、[全庫觀察](../../../../scripts/ng71_observation.py)及[獨立終局reviewer](../../../../scripts/review_ng71_pilot.py)。本節先凍結執行契約；只有後續真正的啟動／退出回執才能證明科學訓練已運行。原配置v2仍保持`training_enabled=false`，新run只在封存已驗證P0／CUDA證據、source、tests、protocol與全部依賴之後，產生唯一差異為啟用執行／status的新配置，不修改凍結舊attempt。

```text
common NG3 init + accepted step0 witnesses
  -> A 0..96 -> encode A96 (all documents, pilot TRAIN only)
  -> shared A96 witnesses -> A 96..192
  -> B 0..96 -> B 96..192
  -> C 0..96 -> C 96..192
  -> D 0..96 -> D 96..192
  -> ALL EIGHT TRAIN CHUNKS SEALED
  -> initial / dense / BM25 observations from frozen codes
  -> A/B/C/D step96 pilot-TRAIN observations
  -> A/B/C/D step192 pilot + sentinel + exposed-DEV observations
  -> independent paired review; no automatic scale or production promotion
```

全部29個phase獨立保存source/input校驗、actual-start offline ClearML、process-group退出、stdout和完整檔案SHA。GPU採用同一cooperative lock；每次GPU階段前重新檢查占用，保留CPU4、treeRSS16GiB、GPU20GiB、host available大於24GiB、free大於40GiB與每phase5400秒界線，約0.5秒檢查資源；`nvidia-smi`另有5秒timeout。STOP或建立started receipt失敗也必須關閉自有process group。不能重跑／覆蓋已存在phase，也不重啟健康工作。

Terminal固定2,304題：pilot TRAIN384、未參與梯度的TRAIN sentinel384、已曝光DEV1536；step96只編碼pilot TRAIN384。所有查詢共用完整233,009文件背景；不編碼LOCKED、不重建teacher、不用目前成績調參。記錄全正例rank、top100、top10/100 entrants、相對A96 reference的pool ranks／池外新前排，及CSR、DF分布、高DF尾部、query-DF。這些仍是研究工作量代理，不是native latency。

Reviewer在固定query上比較B-A、C-A、D-C、interaction、D-A、D-dense、D-init；每域等權、paired query bootstrap10,000次、seed71071。區間只對此初始化／順序有效，不冒充多seed確認。逐步重核768exposures／192updates、optimizer chain、實際token／candidate counters與參數更新。對每個fresh-forward score重新計算float64 loss／導數；float32訓練值的預設核對界線為loss `rtol=2e-6, atol=2e-7`、accumulated score gradient `rtol=2e-5, atol=2e-7`，不能看結果後再放寬。完整排序兩條累加路徑仍使用`1e-12`絕對界線。

本地277項研究／文檔測試通過，包括真實subprocess group終止、receipt寫入失敗清理、terminal barrier、LOCKED拒絕、完整ties／多正例、bootstrap與成本門檻。完整step0的47,593個eligible pair target轉float32後沒有變成0.5（最小約0.50000599），不是用人工fixture替代實際manifest的精度检查。這些是啟動前檢查，尚不構成四組品質結論。

## 10. 21:09 UTC：正式 pilot 已啟動，首段訓練已封存

`pilot-v1` 使用 source commit `16646238`；Mac封存後，以exit0傳輸到lambda2，31份source/config/test/protocol及162個依賴逐項SHA核對通過，`inputs.json` SHA為 `5c44c86e13d688972100f1eb1a85b562ba8a631bdf31a2b2e4073ef9b304fb2a`。啟動前GPU0只有1MiB／0%使用、host available110.24GiB、artifact free726.17GiB。GPU1／2另有同事工作，沒有占用。Tmux為`ii42_ng71_pilot_v1`，controller PID3370572，remote run為`NG-0071/pilot-v1`。

**A0..96已真實完成**，不是新增一個準備階段：384 query exposures、6,420 candidate pairs、query tokens6,314、document tokens893,647。Worker129.31秒，exit0／error null／owned group closed，peak treeRSS2,948,599,808 bytes、GPU allocated1,869,964,288 bytes；checkpoint與named optimizer完整重载核對通過。其complete SHA為 `5051dcf34459d72ef370c1cc26725ac1934467568c4da58dfadaee2d0e9abc15`。共同初始化仍是`b61b0e54...`，A96 state為 `81d4e6dbc8aea75915fb073dc1f4c35e6d0be06c0be491cebe9e30dce11c0c20`。

Step1／24成本canary均通過。另唯讀審核step1／24／96的12個實際fresh-forward TRAIN exposures，float64參考與accumulated float32 score gradient最大差 `1.649776781853518e-08`，沒有重做更新或查看DEV品質。Step80的preclip gradient norm約6.43、update L2約0.00870，顯示clipping確實生效；這不是pair組或品質結論。

控制器已自動進入A96全庫編碼。初始768文件canary的含1.5x餘裕預估1,067.82秒，通過原4,800秒門檻；兩次後續觀察為12,384文件／33.22秒與21,600文件／58.65秒，約368–373文件/秒。仍需完成全部文件、query編碼、封存，再建立共用A96 witness快照；不能將canary當成全庫完成。後續四組品質、成本代理及獨立review均待終局，LOCKED未使用。Mac目前只有verified frozen controls，**不是完整輸出備份**；遠端所有產物保留，完成後再做全量回傳與完整SHA審核。

## 11. 21:42 UTC：共用 reference 交接與 A/B 續訓通過

同一`pilot-v1`、source `16646238`繼續運行，沒有重新啟動、改動凍結程式或重跑已完成更新。六個phase的完整inventory／全部SHA在lambda2重新核對通過，均為exit0、error null、owned group closed及actual-start offline ClearML closed。

| 新完成 phase | Worker wall | Complete SHA256 |
| --- | ---: | --- |
| `encode-A-96` | 673.81秒 | `a2bddb6e765a8207a02dedd4d95a4c0b8a31adb0aa5121cd69c5018111a2d4df` |
| `snapshot-96` | 508.26秒 | `5007bb89105fec2feab3dd09d48f856f2655706718dae67fff8d77a6af5220df` |
| `train-A-192` | 129.80秒 | `8847c9b85f860bf96e727ffadccd3c2329503ecec2382313c693bc9e9348cbd5` |
| `train-B-96` | 424.85秒 | `f9e80a6f033dec890d311addf8381704ffa51168305ef065bd99ef82046c1fa7` |
| `train-B-192` | 134.43秒 | `6869916b5c5b16541fbf30d4fc1512cda6206dccc67ae3bc433bbf816dfee0ab` |

A96編碼涵蓋全部233,009文件及384題pilot TRAIN，文件NNZ49,794,307、CSR399,286,496 bytes，query NNZ12,758。共用快照用同一A96模型重算全庫排名，最大雙路score誤差`1.2212453270876722e-15`；384題／192個固定batch的eligibility gate通過。這首次實際驗證了「新模型編碼全庫 → 共用見證刷新 → 同一arm optimizer續訓」交接；不是只驗證零更新的初始快照。

A、B各完成768 exposures／192 updates。A192 state SHA為`bbf64769f266d6c2369023814fec4da977170bb24cad062d045b4a669e606e89`；B96為`401a68a75e1b9d6b6c768ef6e8168c28c020f9944190d008cdee3ca40a560470`，B192為`f03ff4c0a92238a235760e358600bcedf4c3955451e366793e6859d30bb2632a`。B192明確引用上述B96 complete及optimizer fingerprint `3b628b984e21dcde81860fb96d0280cce7d8eb9956d52cb77fdba2d5cb381fac`，重載通過；沒有由A或disposable fixture接續。

**訓練計算完成與tracking封存分開計時。** B0..96的更新迴圈只用69.73秒，但在已寫出checkpoint／optimizer reload結果後，ClearML收尾暫未返回。當時沒有`clearml.json`／`exit.json`，因此未宣稱phase完成。它其後自行正常退出，whole worker424.85秒；B96..192只用134.43秒。已安裝SDK的`Task.close()`會先等待repo detection，明確timeout為300秒，與額外約5分鐘吻合；**未取得當時Python stack，不能把這個對應當作已證實的內部根因或deadlock**。未中止程序、改SDK或放寬界線；後续工期須包含這種追蹤開銷，不能只用optimizer-loop秒數。

截至這次觀察，C0..96已完成66／96更新，前1／24次成本canary預估716.51秒，通過原5,100秒門檻；C／D尚未全部封存，不能開始sentinel／DEV觀察。四組品質、成本代理和terminal獨立review仍待後續，LOCKED未使用。最先四個已封存phase正在回傳Mac，**傳輸中的檔案不是已校驗的備份，整個run尚未完成mirror**；遠端保留。

排程依下一個需要介入的事件而非整個run總長度調整：A96交接已完成，C擴池canary通過，先確認D的首次canary與當前partial-copy封存；之後健康的長編碼／排序段可採30分鐘agent觸發。0.5秒資源檢查、GPU占用檢查及stage timeout保持原樣。按實測約11.2分鐘一次全庫編碼、C每update約3秒及既有同背景rank回執分開估算，剩餘流程暫約2–3.5小時，仍有D組、密度變化與tracking收尾的不確定性；不是完成或模型品質承諾。

21:47 UTC補記：首批`train-A-96`、`encode-A-96`、`snapshot-96`、`train-A-192`回傳已exit0，61份檔案共1,489,994,400 bytes；Mac逐項inventory、全部SHA、退出／tracking回執及31份凍結source再核對通過。外部`pilot-v1-first-four-copy-review.json` SHA為`8944b91900edaf36143ea5864b4590269b3d0c8bb3403f5c0eec4c8ab207bdcc`。只有這四個phase完成mirror，不能推及尚未回傳的B/C/D或整個run。C首段的96次更新及checkpoint reload已返回結果，退出／tracking封存仍待確認；只讀debugger無法attach，沒有變更主機ptrace設定或聲稱取得堆疊證據。

## 12. 22:09 UTC：全部四組訓練封存，進入完整觀察

八個訓練chunk已全部exit0／error null／owned group closed／actual-start offline tracking closed。A/B/C/D各為384題固定TRAIN、兩輪、768 exposures及192次真實optimizer更新，合計3,072 exposures／768 updates，**不是3,072個不同query**。共同A96快照及全部source／依賴維持凍結；未重訓、調LR或更換checkpoint。

在lambda2用凍結[reviewer的TRAIN審核](../../../../scripts/review_ng71_pilot.py)獨立重新校驗：八份完整inventory及全部SHA、初始／續訓model與optimizer鏈、768次曝光順序／arm、全部fresh-forward loss及四query累積後的score gradient、非零有限參數更新、token／candidate計數。35.26秒完成，沒有讀terminal品質或執行模型推論。全部loss／導數通過第9節預先固定的容差。

| Arm | Clipped updates／192 | 每次parameter update L2範圍 | 最大score-gradient重算誤差 |
| --- | ---: | ---: | ---: |
| A 原pool＋CE | 192 | 0.007348–0.019424 | `6.8565e-08` |
| B 原pool＋pair | 92 | 0.006364–0.019353 | `1.0647e-08` |
| C witness＋CE | 192 | 0.007782–0.020312 | `6.9102e-08` |
| D witness＋pair | 94 | 0.006642–0.020215 | `1.2252e-08` |

這證明pair路線有實際、有限的參數更新；不能只憑step0的score-space導數較小，就說pair「沒有訓練到」或直接倍增LR。表中L2僅反映更新大小，沒有證明其方向、泛化或品質更好。A/B各處理1,787,294 document tokens／12,840 candidate entries；C/D各7,961,577／61,074，額外witness確有約4.45x文本與4.76x候選工作量，不可隱藏這項訓練成本，也不能把它當成serving成本。Worker wall A/B/C/D各259.11／559.28／994.57／714.33秒包含初始化、重載及tracking，B96/C96的額外約300秒收尾會混淆直接速度比較。

C192、D96、D192的step24含餘裕預估各700.77／693.97／757.01秒，均通過原5,100秒canary門檻；實際worker各349.73／354.84／359.50秒，沒有GPU或host-memory上限放寬。最後D192於Unix`1789250795.339222`退出，首個`rank-initial`於`1789250796.5188575`才開始，終局觀察屏障時序已核對。全部八段完成不等於29個phase或最終品質審核完成。

| 新封存證據 | SHA256 |
| --- | --- |
| `train-C-192/complete.json` | `54f2aa21071f1f3f0b127b90b2e0558f978ce5bda03a7f8e791d422a194a2449` |
| `train-D-96/complete.json` | `cbb8e9a0c77985359ce30f36914b8aeaf3fc9fe8608c4471c78ddcd6779419c0` |
| `train-D-192/complete.json` | `da2c84e5fe59e2987537f2fe95f02a48ee62737a0b191ab76a7435b5fce6fa0d` |
| 外部`pilot-v1-training-audit.json` | `6c2fb1ccc44e8fb662450cda042b1b6eaa4495b8cf3dcd599c4fdf0acb0b3668` |

現在按凍結順序完成initial／dense／BM25對照、step96 TRAIN及step192全surface觀察，再做完整paired reviewer。不提前比較不完整品質、不改loss，也不推論已超越dense。B/C/D六段checkpoint另起唯讀回傳，首批四段mirror仍有效，但新的傳輸尚未完成；遠端保留。首D canary已通過，agent輪詢延長至30分鐘，獨立資源／超時監測不變；剩餘編碼、排序和review暂估約2–3小時，回傳約35分鐘並行，不把傳輸進度當作SHA驗證完成。
