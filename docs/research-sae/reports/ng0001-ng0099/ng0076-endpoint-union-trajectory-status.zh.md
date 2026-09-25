# NG-0076：端點聯集與歷史軌跡執行狀態

2026-09-13 UTC。對應 [NG76 計畫](ng0076-endpoint-union-trajectory-plan.zh.md) 和 [NG75 終局](ng0075-parameter-update-diagnosis-review.zh.md)。

## 最新狀態：原始 D96 中點、全庫排名與獨立審核完成

07:17 UTC，兩個remote phase均已封存，完整Mac mirror與獨立CSC/count review通過。26 query forwards、0 document forwards／optimizer updates；24題全庫5,592,216個混合分數獨立重算最大差0，全部排名／gold／5,226對margin／per-query及domain彙總通過。參見[完整中點結論](ng0076-authentic-midpoint-review.zh.md)。

同8題NQ的nDCG為0.753529→0.695181→0.702839，Recall@100始終1；前排退化在前96步已存在，不能只歸給後半程witness，但沒有識別因果batch／域。原13點歷史observer仍失敗且保留。NG77已整理為[可信margin保留準備方案](ng0077-trusted-margin-retention-plan.zh.md)，不是已啟動新訓練。

Remote query／rank elapsed83.497945／102.514604秒，均exit0／error null／group closed／actual-start offline ClearML closed passed。controller於07:10:45終端，GPU0重新空閒。完整run加fixtures73檔案／51,434,330 bytes；全678 dependencies／36 source再驗，來源未刪。Mac獨立review12.859026秒、RSS1,198,522,368 bytes，exit0／group closed，proof SHA `a39d2b0a3133095b09bec4c23663150d16c39cf148a2b132dd9369f99dded977`。下方提交／載入描述均為歷史快照。

### 中點提交：07:07 UTC

07:07 UTC，新 `authentic-midpoint-v1` 已在lambda2 GPU0提交，tmux `ii42_ng76_authentic_midpoint_v1`。它只新增26個query forwards、重用233,009份原D96文件CSR；沒有document forward、optimizer更新、DEV或LOCKED_TEST推論。完整24題TRAIN全庫排名與獨立CSC/count review尚未取得，不宣稱新科學結論。原失敗observer和已完成control均不重啟。

- source commit `608af42acdb967026daa0d72dd81052cc59d43a1`，local研究／文件測試358 passed；remote frozen fixtures66 passed／3.02秒。
- manifest `eb62acf84b654536523e7953473d28326549eca4347abeaf31c0f7971c70ac28`，678 dependencies／36 source-payload entries，37個初始檔案／3,367,850 bytes。remote依賴審核0缺少、0不符；只傳新增unit，rsync exit0／0刪除／不覆寫既有檔案。
- CPU freeze14.439750秒、peak RSS244,252,672 bytes，exit0／error null／group closed；complete SHA `299cf6c8b9d2303f7da64391fee34da8c8ad478028f82c0d551fbba4b22e2d5f`。
- remote fixture程序含兩次全依賴核驗49.464921秒、peak RSS680,656,896 bytes，exit0／error null／group closed；complete SHA `a62482619e8b560b6ac16fa556f2d32d5dfdfebd2987002e0c0659cd5371d9a6`。
- 新查GPU0為1MiB／0%，host available120,206,692,352 bytes、free disk761,826,574,336 bytes；GPU1／2同事工作未干預。query段1800秒／RSS16GiB，CPU排名段900秒／RSS8GiB，其他安全底線不變。按載入／SHA／tracking收尾／mirror／independent review整體估時，維持5分鐘輪詢。

## 歷史重放第80步失敗，現場已保存

05:51:54 UTC，`replay-D-96` 因 exact readout-count 不符退出1，controller停止；沒有啟動第二段。第80步第0個example的document NNZ是17,779，歷史值17,780。GPU0已釋放，沒有OOM、host/disk或時間guard觸發。前80步numeric tolerance均通過，但這不能替代exact readout與終端model/moments gate；**NG76整條歷史軌跡未完成**。

只讀CPU trace審核顯示最早update norm微差在69步、scores／loss／score gradient差異在74步；身份與token數無差異。尚未識別因果機制，不能直接歸因模型品質、GPU nondeterminism或observer。詳見[失敗診斷與無observer對照計畫](ng0076-replay-control-plan.zh.md)。

失敗worker完整elapsed376.079546秒、peak tree RSS2,413,801,472 bytes，owned group closed；ClearML actual-start／closed／failed，offline未remote-sync。failed `complete.json` 的37個檔案SHA已核對。完整run在remote與Mac的69檔案／24,852,998 bytes exact inventory一致，539個parent dependencies和26個frozen source/payload重新核對通過；來源未刪除，不是災難還原測試。remote failed inventory SHA256：`b3f3a3cf288833b624a1e387912ec2b819636c9249fbaa72217b807515ef08cb`。

`historical-trajectory-v1` 保持immutable失敗。下一步只移除週期觀察的96步獨立對照，區分量測副作用與歷史數值重現問題，不改loss、不放寬gate、不重跑同一attempt。以下05:47／05:50紀錄是歷史快照，不是目前仍在執行。

## 獨立無 Observer 對照：06:26 UTC

- 新 source commit `36a3f081c5a8d2c43a303fe5b4e99c47f813026a`；local研究／文件測試333 passed，remote frozen fixtures41 passed。失敗attempt和原gate未修改，沒有放大資料或新loss。
- `NG-0076/no-observer-control-v1` inputs SHA256 `5831092424a3f68c8bb309322ef0bb94d04729209aaf9d8d3fd9d9bb4f79671d`；609 dependencies、30 source/payload entries，31個初始檔案、3,308,513 bytes。609包含原539依賴與完整失敗現場／inventory，不是擴大訓練集合。
- CPU凍結程序exit0／error null／group closed，13.985690秒，peak RSS231,424,000 bytes；complete SHA256 `27f088ca0b26b4fd07ae2679a1a1fdf19552ab81282b4c847f686633e8d21629`。
- remote fixtures41 passed／5.85秒；完整程序7.773285秒、peak RSS681,254,912 bytes，exit0／error null／group closed；complete SHA256 `4ee5213d76583fce1a78b490fd7cc58e66221d36a4120c3c86b9e832ad0f891b`。前後frozen inputs核對，fixture完整回執與launch快照已mirror並驗SHA。這不是完整control結果備份。
- `ii42_ng76_no_observer_control_v1` 的單一 `control-D-96` worker於06:23:36 UTC啟動。controller PID3472930、worker3473018只作時刻證據；後續需查新identity。GPU0原1MiB／0%，host available約112GiB、free約711GiB，無移除他人GPU工作。
- **實際完成27／96更新**，累計update84.28秒；首步／第24步canary通過，舊函數保守段估計704.01秒，外層1800秒guard仍獨立有效。ClearML已actual-start，offline，尚未closed。不能從步數推斷終端parity或程序完成。
- Torch2.9.1+cu128／Transformers5.15.1與NG71原preflight記錄一致，NG71原D96同樣使用GPU0；這排除明顯版本號／GPU型號差異，並不證明相同CUDA kernel選擇。

輪詢縮為5分鐘，按剩餘updates、保存／reload、tracking closure、checksum、mirror及CPU audit重估。若無observer仍不匹配，不盲目放寬或反覆重放：**優先直接讀原已封存D96 checkpoint**，用原initial／D96／D192三個真實狀態在完整端點聯集定位前半／後半變化。這只需新增一次D96的有界forward，不需要optimizer replay；若需進一步細分時間，再設新且可重現的計算契約。這是下一步設計方向，D96新觀察尚未執行。

### 對照終端：06:30 UTC

無observer程序已完成96步／384 exposures，**全部trace欄位除耗時外逐值一致**，不是僅落在tolerance內；原model SHA與AdamW fingerprint亦完全一致。`historical_replay_qualified=true`，但`overall_goal_qualified=false`。前80步原observer的微差因此不能當成「相同配置必然無法重現」；observer相關執行路徑是主要嫌疑，但單次對照未定位kernel或證明普遍決定性。

- 完整worker392.037576秒；實際update304.733902秒；peak tree RSS2,807,754,752 bytes，peak GPU allocated1,896,107,008 bytes。exit0／error null／owned group closed，ClearML actual-start／closed／passed，仍為offline且未remote-sync。
- 原／control model SHA256 `92e574071092f08f6288a980ac27a1f0e17b82b0274acac40d3cf8cc228abd49`；optimizer fingerprint `f7726f77dcec15d17c888a6b48ff03168d43fae1fd8d428fc888580d9e820540`。
- phase complete SHA256 `836fecdee2ed0a0044dde2e4d6b7104554164520b0886b7f9fbbc895bff158ad`；remote完整inventory54files／600,682,490 bytes，SHA256 `df0bb53d56102dd526bcbcf5675b5616099c0f8aa998d042ac58bd5894f21d94`。controller已終端且GPU0重新1MiB／0%，無後续phase。
- Mac完整checkpoint／optimizer／raw-trace mirror於06:39 UTC完成：54個檔案／600,682,490 bytes exact inventory一致，609 dependencies／30 source entries重新核對。Mac另以原始JSON除`seconds`外的整體相等檢查确认96步完全一致，並對照原model與optimizer指紋；不把重跑同一numeric helper稱為第二套獨立數學實作。
- 傳輸加核驗308.564784秒；`no-observer-control-v1-final-mirror/verified.json`保存結果，`control-versus-observer-first80.json`保存與失敗observer的逐項差異。來源未刪除、沒有執行災難還原測試。最终研究／文件測試333 passed／3.06秒，`git diff --check`通過。

下一步收斂為[原始D96中點、重用文件快取](ng0076-authentic-midpoint-plan.zh.md)。新查確認D96已有233,009份文件編碼，但query cache不含這24題sentinel。因此只需補24題query forward及2個固定parity controls，再用舊完整document CSR做CPU全庫排序，比重編2,974份文件或反覆重放更小。它仍是待凍結新診斷，未執行；原完整checkpoint與runtime重現問題保留，不因本次對照成功而刪除失敗現場。

## 已完成：CPU 準備與端點驗證

本輪已實際完成 CPU 準備，不只是計畫。後續 controller／observer 亦已實作、凍結並提交 lambda2，見下面 05:47 UTC 執行紀錄；沒有新的科學模型訓練、部署、LOCKED_TEST 推論或效果認定。

固定 NG75 原本 hash 選定的 24 題 TRAIN sentinel。每題包含 initial top100、D192 top100 和全部 gold 的完整聯集，不依終局勝負篩選。準備結果：

| 項目 | 結果 |
| --- | ---: |
| TRAIN queries | 24 |
| 不重複文件 | 2,974 |
| Query-document 組合 | 3,038 |
| 全部 gold/rival pairs | 5,226 |
| 最大單題聯集 | 144 |
| 兩端 top100 和全部 gold cutoff rank | 均與原全庫結果一致 |

這個擴充針對 NG75 的觀察盲點：原固定 rivals 只涵蓋同 24 題 NG72 已知 60 個 lost comparisons 中的 1 個。這是 NG72 說明集合的覆蓋統計，不是全庫所有丟失關係的分母。NG76 的池子不是只挑這 60 個失敗比較。

端點驗證只使用已保存的 CSR／排名與固定 BM25 分數，不重新執行 encoder。兩端排名正確不代表中段聯集排名等於中段全庫排名；只在中段暫時出現的競爭者仍可能不在池內。

## 凍結與回執

- 準備來源 commit：`346c8ba6b53db6a5a65d9ac21b28fe2b5c6e2b02`。
- 準備程式：[prepare_ng76_trajectory.py](../../../../scripts/prepare_ng76_trajectory.py)。
- 耐久 artifact unit：`NG-0076/preparation-v1`；5 個檔案、3,010,295 bytes；manifest 包含 530 個 parent dependencies 和 4 個 source/payload entries。
- `inputs.json` SHA256：`28f676cb3f858094a097f74c772b7ee509c10fcdb1aa3e79a113f907f197d80d`。
- 有界程序回執：`NG-0076/preparation-v1-procedure/complete.json`，SHA256 `78e8b143024dc2aab54d69c305374bff6e65537154992581e18097491a51e94e`。
- 終端結果：exit 0、error null、owned process group closed；15.651686 秒，peak tree RSS 1,746,173,952 bytes。
- CPU 準備限制：4 threads、RSS 8 GiB、host available floor 12 GiB、disk free floor 40 GiB、900 秒。這不是放寬歷史 GPU replay 的限制。
- 所有 frozen dependencies 和 source/payload 在準備前後驗證；NG75 全部終端輸出、獨立 Mac reduction、NG72 diagnosis 的封存證據亦驗證。未刪除來源。沒有聲稱災難還原驗證或獨立第二份 NG76 備份已完成。

## 已實作與啟動：05:47 UTC 快照

- 實作 commit：`700244efc67a46023a085d8dcd8c521398c06668`。optional observer hook 預設不啟用，原 loss／optimizer／update 路徑不複製；只讀觀察器會核對 model、完整 optimizer state、梯度、模式、requires-grad 與 RNG 前後一致。
- 研究與文件測試 **318 passed / 3.01 秒**；remote frozen CPU fixtures **26 passed / 5.76 秒**。小模型連續8步有／無 observer 的實際 updates、model SHA 和 AdamW moments 完全一致；參數／moment／梯度／模式／RNG 故意污染均被拒絕。這不代替大型模型的實際歷史 replay 驗證。
- 新 frozen execution unit：`NG-0076/historical-trajectory-v1`，539 dependencies、26 source/payload entries，初始27檔案、3,274,575 bytes；`inputs.json` SHA256 `b53a8a6fea1710b260855626e9baced73ed7ddef81dfb15802a33c47b3135822`。
- 凍結程序 exit 0、error null、owned group closed；14.623268秒、peak tree RSS223,674,368 bytes。回執 `historical-trajectory-v1-preparation/complete.json` SHA256 `fdb8a43018276f0d51e49de5704f8e4c1fc754cb5812e55187cde4fff2ad67f6`。
- 遠端先驗539個依賴，16個缺少、0個內容不符；只補齊缺少依賴並新增 execution unit。rsync exit0，43個檔案／6,396,909 bytes、0刪除、不覆寫既有目的檔案。remote 和 Mac 再驗 frozen hashes 通過。這是輸入同步，不是已完成 trajectory 結果備份。
- remote fixtures 獨立程序 exit0、error null、owned group closed，7.626513秒、peak RSS680,841,216 bytes；前後 frozen inputs 皆驗證。fixtures complete SHA256 `b8d03a7b928be4b3a8289e537731bcb5d2cdd65f3b73b94acada8077974539bc`，該回執已複製回 Mac 並核對。
- lambda2 的 `ii42_ng76_historical_trajectory_v1` tmux 已提交；controller log 已進入 `replay-D-96`。GPU0在啟動核驗時1MiB／0%，沒有移除他人工作；host available 約112GiB、free disk 約711GiB。單GPU與每段1800秒獨立資源guard不變。
- **此快照尚未取得 step0／首16步 canary，不宣稱歷史192步已完成或新模型品質有提升。** 下一步先讀實際phase回執、原始step trace與observer canary，健康載入／checksum／tracking closure不重啟。

## 實機 Canary：05:50 UTC 更新

首段 worker 已實際執行，不只停在 tmux／manifest：controller PID3468205、worker PID3468299，段開始於05:45:38 UTC。以下 PID 僅作該時刻證據，後續操作仍需重新核對身份。

- step0 完整24題／2,974文件觀察9.6700秒，初始端點scores／top100和gold cutoff parity通過；model、optimizer、梯度、模式和RNG前後一致。
- step16、32觀察分別12.6615／12.6358秒；均完成只讀狀態檢查。`NG71_TRAIN D 40` 已輸出，表示前40個原歷史更新逐步通過loss／scores／score-gradient／update norm／identity及readout計數核對。
- step0／16／32的完整首段預測為1,046.75／1,005.59／964.01秒，包含360秒收尾保留，均低於1,740秒canary門檻；每段1,800秒外層guard不變。這是有裕量的預測，不是承諾完成時間。
- 採樣worker RSS1,867,153,408 bytes，host available約110GiB；沒有觀察到資源門檻觸發。GPU0仍由既有controller單獨協調，不重啟健康重放。
- **尚未取得D96／D192終端model+moments精確匹配，也沒有完成軌跡reduction或新的模型品質結論。** 初始輸入與canary回執的Mac副本不是全程結果備份。

依實測observer與update時間，整個後續流程暫估25–35分鐘量級，包含兩段重放、checksum、可能較慢的tracking closure、review和Mac核驗；輪詢改為10分鐘，剩餘工作縮短後再調整。

## 後續驗證條件

1. 實作只讀 observer，於原 D 歷史步數 0、16、…、192 保存同版 query/document codes 和固定池分數；small-model 測試必須證明 observer 不改變 model、optimizer、RNG 和原始 updates。
2. 建立獨立的 execution manifest，凍結實作與測試。不要修改目前 `prepared.json` 中仍為 false 的 scientific-training／trajectory-replay flags。
3. 在 lambda2 新查空閒 GPU 和 host 資源後，以原始 D 配置重放兩段 96 updates。先做 step0 與首 16 步 canary，沿計畫估算完整時間及收尾；每段 1800 秒上限不變。
4. 每步匹配舊數值 trace，段末匹配原 D96／D192 model SHA 與 optimizer fingerprint；任何差異保留 attempt 並停止解釋。只有回執、封存、獨立 reduction 全部完成，才宣稱軌跡完成。

以上實作、凍結及首段canary已完成，歷史終端 parity、封存與獨立 reduction 仍待完成。不得將舊計畫的「未啟動」文字用作重複派發理由。沒有改動產品程式。
