# NG-0080：執行紀錄與問題閉合表

日期：2026-09-13。科學協議見 [NG80 聯合診斷](ng0080-bottleneck-campaign.zh.md)，背景見 [NG79 後綜合備忘錄](../designs/ng-teacher-geometry-supervision-reset.zh.md)。本文件為可更新狀態，不覆寫外部 frozen protocol。沒有生產部署或模型晉級。

## 已執行與未執行

最新runtime狀態（2026-09-14 04:11 UTC）：S/P各3,072-update訓練、V2全量審計和initial/S/P三個全庫編碼已成功閉合；S/P training完整Mac鏡像已核對。V2 lexical準備在Python shutdown因tracking thread錯誤abort，原exit=-6保留，尚未准入calibration/ranking。V3只修tracking設定並重用已閉合審計／編碼，重新執行準備與排名。下方舊「active」時間點是歷史快照，不代表那些attempt仍運行；目前沒有新訓練。

`NG-0080/readout-v1` 的 input SHA256 為 `d8dca9c36ff49047c0fc0acfe0ee9cbcd607537a65596d9013300537f2e8dff1`。來源由逐檔 SHA 鎖定；起點 Git HEAD `7e8b3054` 尚不包含本輪新增檔案，不能把該 HEAD 當作可單獨重現新程式的版本。

- `prepare` 已閉合：exit0、closed process group、actual-start offline ClearML closed、輸出 SHA 校驗。共 13,824 題，三域各 4,608；12,288 TRAIN、1,536 TRAIN_DIAGNOSTIC；233,009 文件。TRAIN source pools 涉及 158,190 文件；panel 最大 51 份，包含共用背景 16 份，原始正例全保留。
- lambda2 GPU 0 的 `encode-teacher` 與 GPU 3 的 `encode-hidden` 在 18:54:53 UTC 啟動，現已完整閉合。Teacher query 計算61.48秒，對舊1,536題的最大差4.612e-6，通過預先容差；hidden query/document計算368.26秒，參數 hash 未變。
- `diagnose` 在兩條 encoding 完整閉合後執行，19:03:52 UTC 完成，exit0/error=null/group closed。全 phase50.18秒，peak RSS5.10GB。輸出及完整 Mac mirror 分開驗證。
- 第一波無 optimizer update；第二波 [F/D 讀出訓練](ng0080-dense-control.zh.md) 的兩組固定訓練已閉合，下方另列獨立審計／終點與 mirror 狀態，不能由訓練完成推定品質。
- ClearML 為真實開始的 offline tracking，未聲稱已同步遠端服務。NG79 endpoint完整Mac鏡像現已核對；closure與鏡像證據見下方更新，不改其scientific failed結論。

NG80 第一波與監督審計已完成 Mac 鏡像驗證：readout58份 phase payload、838,307,433 bytes，receipt SHA `25754a79a45cdcd2593c4dd330b043c59f5b0e82e353054cb439d01cbf1c4e0b`；監督 v2為10份、4,277,109 bytes，receipt SHA `d8457b175dfbf669be282a16892949aff655ea45dfe67690635cbfd3710cd44f`。源檔、input及精確 phase inventory全部核對。失敗v1亦鏡像保留，沒有刪除。

## 完整問題表

| 問題 | 本輪判斷方法 | 當前邊界／下一步 |
| --- | --- | --- |
| 底座是否含 teacher 可讀出的訊息 | 固定 mean/CLS 的 TRAIN-only shared ridge | 簡單線性讀出仍有大差距；負結果只能否定此讀出，不是模型容量上界 |
| 解凍底座是否優於只訓練讀出 | F frozen dense / D full dense，固定初始化及更新預算 | 全背景終點支持D優於F：nDCG+0.165622，三域皆改善；仍低於PPLX，不是sparse/hybrid產品結論 |
| 稀疏讀出是否限制分數幾何 | S actual sparse field / P protective hybrid | 正式訓練、全量審計和全庫編碼已閉合，排名尚待tracking修正後執行。P為uniform-per-positive對照，不宣稱重播NG79原D |
| 更多 query 與更豐富關係哪個有益 | 固定 TRAIN split/panels，比較 teacher field 與正例保護監督 | 12k 三域不能代表十萬級多任務；panel pair 數不是獨立 supervision 數 |
| teacher 與標註是否衝突 | 原／新增正例分開；unknown 不設硬負例 | 人工複核未取得真人答案；soft teacher 不是 relevance ground truth |
| 輸入窗口是否缺少關鍵證據 | 長度統計與預先固定子集的 context 對照 | 截斷統計不等於因果；未取得 evidence span 不宣称看不到正解 |
| 高 DF 是否實際缺乏判別作用 | 分數差貢獻、query DF-work、聯合 mask 後排名 | 不按單個 posting utility 可加假設刪除；不能將成本 proxy 等同 native latency |
| 多來源泛化與總成本是否過關 | source/topic/task-family 隔離評估、多 seed、full corpus hybrid、native | 尚未執行；不可接觸 LOCKED_TEST，不得提早宣稱超越 dense |

## 第一波觀測

以下僅是固定51份以內 panel 的 all-positive nDCG@10，不是全庫 top-k，也不是完整 hybrid。

| Shared ridge | TRAIN nDCG | TRAIN_DIAGNOSTIC nDCG | Diagnostic teacher top10 overlap |
| --- | ---: | ---: | ---: |
| mean hidden | 0.680360 | 0.679622 | 0.690365 |
| CLS hidden | 0.731606 | 0.726368 | 0.694987 |
| PPLX teacher | 0.914576 | 0.917731 | 1.000000 |

TRAIN 與診斷同時有明顯差距，支持「這個共同線性讀出不足」，不證明 trunk 容量不足、原 sparse 必然相同差距或需要從零預訓練。按既定協議，F/D 仍從 mean 投影開始，不用 CLS 較高的結果臨時換初始化。

完整13,824題中 teacher query64超長112題、student query64超長106題；233,009文件中 student document256超長13,388份。超長不是錯誤率，還需要正例曝光及真實證據窗口審計。[獨立 cached audit](ng0080-supervision-audit.zh.md) 同時量測 query second moment 與正例保護圖的可識別 margin 自由度。

監督審計 `cached-supervision-v1` 因 NumPy `int64` 輸出 JSON 失敗，exit1、owned group 已關閉，原 attempt 保留。修正僅把飽和 pair 計数轉為 Python int，並新增 JSON roundtrip 回歸；另凍結 v2，不修改 v1 或科學選題／閾值。

### 已閉合的監督審計 v2

Input SHA `b05ca935499f5dbfe69e056690649c183fb1b2813c61e940d3d4933bfa18f3a9`，19:18:55 UTC 終止、controller `all_phases_complete`。這不是訓練成效。

- 12,288題中11,157題（90.8%）的 teacher-positive-margin 圖已連通；平均可識別31.476個精確 margin 自由度，完整 field 為31.771。因此不能把缺少顯式 nonpositive/nonpositive loss 等同缺失大量獨立約束。原 D metric weights 和實際模型梯度不在此審計中。
- 平均52.851個有效 pair 中44.423個 target>=.99，約84.1%。這表示 soft targets 對較大正 margin 的區別被壓縮；**不等於學生 BCE 梯度消失**，也不是已證明的唯一失敗原因。Field 目標因此應測精度／條件數，而不是宣稱增加了平方倍的獨立資訊。
- NQ 的連通率僅3,074/4,096（75.05%），平均缺0.816個相對自由度；與 FEVER/Hotpot 不同，需保留分域判讀及 teacher/label 衝突研究。
- 本批 NQ canonical TRAIN query 與正例文件都未觸發 student64/256截斷，故不能把這次 student 窗口截斷當成 NQ 回退的主解釋。這不排除上游 canonical 文本本身缺少資訊或 teacher tokenizer 的不同窗口。FEVER 則有3,578/5,916次正例文件曝光超256，需分開做 context control，不能全域一刀切。
- 等量舊／新1,536題的 teacher entropy rank 約404.05/404.89；全12,288題約490.23。Sample size 本身會影響譜估計，不能據此證明 task diversity。新題在舊 top128 子空間外的平均能量0.5648，高於舊題0.4392，但舊題是 in-sample projection，仍不能直接當成新任務因果證據。

### 已啟動的 F/D 訓練

| 單元 | GPU | Input SHA256 |
| --- | ---: | --- |
| `NG-0080/dense-F-v1` | 0 | `c6f6e0bd612dde796bb34cabb8aade176b3dffd2b9998fa68058676783f0a942` |
| `NG-0080/dense-D-v1` | 3 | `b8fd35a599bbad9080317907b9131333d44473e778c1e3f5f43abcc88940c656` |

兩組的 `train-canary` 已閉合：real-base full-graph / VJP global gradient relative L2 都為0，完整 checkpoint/projection/optimizer/output reload 全同。共同原始 projection SHA `dc976b4592c0169d8ccaa453b14cd92d28919154fb7df7be80bad5487defaa31`，共同 fixture loss0.011868188；F保留原 trunk hash，D trunk確實更新。兩組都丟棄 canary 更新，重新從 parent 進入正式 `train-1024`。

前16更新外推單個1,024-update chunk：F1,052.73秒、D1,535.02秒（含1.5倍裕量和300秒），均通過5100秒門檻。按三個 chunk 估計訓練約50–80分鐘，完整 endpoint ranking 另計。這是 ETA，不是完成時間承諾或品質結果。監控按約30分鐘的有效邊界檢查，健康時不重啟。

### F/D 訓練閉合與獨立終點

F controller 於19:55:05 UTC、D於20:04:42 UTC記錄 `all_phases_complete`，兩組各四個 phase 已逐檔 SHA／精確 inventory／exit／tracking／resource 驗證。這僅證明固定訓練完成，不表示搜尋品質改善。兩組 remote inventory 分別92檔、845,679,237 bytes與92檔、1,810,834,219 bytes；Mac完整鏡像傳輸exit0，已另完成 exact file set／全部SHA／input／source／phase closure核對。F/D mirror receipt SHA分別為 `b8379c04f1fa0e580beee1109e64fbdc375d4804847dadd8941f0ce008001822`、`1a8898e2f69d41d702b4fd1188ccc38b274e840cc00c5cabab16b9feb84c1098`。沒有刪除遠端來源或在Mac跑模型restore。

[全背景終點協議](ng0080-dense-endpoints.zh.md) 在產生新的品質分數前另行凍結為 `NG-0080/dense-endpoints-v1`，input SHA `da62c44c522cdb3b5a0573ff0d4a731e7c528b661aa2bd07ef62174b69c32473`。先獨立重算12,288題／組的 scalar loss、score derivative、正例／teacher／曝光身份與完整 serialized optimizer fingerprint，再啟動 fresh initial/F/D encoding及固定1,536題的全233,009文件背景排名。沒有DEV或locked query，也不把這個 dense readout 當成 sparse/hybrid 產品。

新增獨立審計與終點合成測試後，lambda2 CPU `184 passed`，pytest5.62秒、程序7.41秒、peak RSS792,800KiB，exit0。`endpoint-source-validation-v1/validation.json` SHA `b5f1222de12c2f562137050eea13c5f2680c66b44f87cb527893115a9e348091`，source archive SHA `f61725193a90d588257e487d8668d56c6613d22467053b81481d4c14c95bc3e5`。本段與導覽更新在此測試後，沒有回寫 frozen source。

獨立 `audit-training` 已於20:20:26 UTC閉合：exit0/error=null/group closed、79.69秒、peak RSS2.11GB。每組3,072更新／12,288個唯一query、402,691次候選文件曝光、206,563 query tokens與53,510,807 document tokens，逐題身份和全量累積一致。Scalar score derivative最大差 F7.98e-10、D7.45e-10；F每份optimizer含2個參數tensor，D含103個，全部serialized moments／step fingerprint獨立重算通過。同組checkpoint chain及worker exact reload通過；沒有聲稱重新獨立執行全部參數VJP／AdamW更新。

通過審計後兩條fresh encoding lane已閉合：initial於20:28:17 UTC、D於20:28:23 UTC、F於20:34:46 UTC完成。三phase分別392.39、397.82、386.78秒，peak RSS約2.96GB；exit／tracking／精確輸出SHA和兩個controller均核對通過。全背景ranking已於20:53:44 UTC以CPU4順序啟動，沒有重啟任一encoding。品質僅在四組ranking及獨立review全部閉合後判讀。

編碼前64文件的保守外推（1.5倍+300秒）initial849.14秒、D952.95秒，皆通過5100秒門檻；現場每卡約438MiB，無其他任務被驅逐。這是單個encode phase的早期ETA，仍需完整閉合。此短期邊界按15分鐘檢查，排名階段取得自身canary後再調整，不輪詢秒級進度。

`endpoint-source-validation-v1` 已完成Mac鏡像與全部2,421份archive檔案hash核對，傳輸exit0；沒有在Mac跑Torch。F/D checkpoint鏡像已另外核對如上；終點輸出完整鏡像仍待graph閉合。Ranking前16題canary保守外推844.58秒／phase，通過5100秒門檻；後續按30分鐘有效邊界監控。

### F/D 終點結果

F/D全背景終點已於21:23:20 UTC全部閉合，[獨立結果報告](ng0080-dense-control-review.zh.md)記錄完整對照及限制。Initial mean/F/D/PPLX的macro nDCG分別0.333550/0.500623/0.666245/0.822175；D-F的95%區間 `[0.150875,0.180781]`，但D-PPLX仍-0.155930。這是TRAIN_DIAGNOSTIC dense-readout結果，不是原hybrid提升或獨立泛化證據。

本次核對9個phase及4個lane controller均閉合，review SHA `8ed5c1cb7c4d32934d7fcdecc968d7201e927a2acf68cf82f0af2d18154b785b`。Endpoint remote inventory148檔、2,902,226,572 bytes，SHA `a8ab7aa710df8611250a8df3b6a92cb37fe1bb921c42d7990cb044fb81859fbd`。完整Mac鏡像傳輸exit0，22:40 UTC本輪已核對exact file set、全部SHA、input/source與9phase/4controller閉合；新mirror receipt SHA `b9e07c964a76ee227b55aa90f3dd93c92af6affc68a13f4550fe86fba5792f15`。没有在Mac執行模型，也没有刪除來源。

### 實際 sparse 准入準備

[S/P preflight](ng0080-sparse-preflight.zh.md) 在新結果前固定，明確原RMS／hybrid權重、TRAIN-only variance scale及 `mean(center(s/a-t)^2)` 的prefactor，不把它與未正規化loss說成optimizer等價。64個固定TRAIN來源文件須重現原全庫四文件batch的bit-exact cache；實際訓練一直fresh文件VJP。每臂只有丟棄的16次更新、per-domain真實VJP與完整optimizer reload，不能將canary當正式3072-update訓練。

新增程式與合成測試在lambda2 CPU驗證：`197 passed`，6.43秒，程序8.71秒，peak RSS792,640KiB，exit0；沒有模型推論。包含尺度梯度prefactor、query等權而非candidate等權、S/P fresh replay、lexical只加一次、token曝光和不可放寬的時間預算。新source validation unit獨立於既有已封存版本。

`sparse-source-validation-v1` validation SHA `50bd4a72ada9e2b577cfca6357c056ce86324ca549a57a00a91fa8db560e1cf2`；source archive SHA `68acae6772a1c900f4005f669cff2265343da97728b256bb0cc110864152a8e0`。已傳輸exit0，Mac以串流逐一核對2,424份archive文件和全部receipt SHA；新mirror receipt `e478a633e730d9c0ad3dfec76f7108774b051eff6a2b2a2c324d05d1753578cd`。最新狀態／導覽文字在此source snapshot之後，另以stdlib核對8份文件134個本地連結；沒有在Mac跑pytest或模型。

`sparse-preflight-v1` input SHA `253de2432f9d4f2b05970eaf744505dc7b1fa9fe3f8dc897db1aae94a77c47cc`，runner SHA `791e12f68bba3dd004e75469fb7815c1b62f55f7f4ccb056b7b43197c8ecf24f`，新測試SHA `ccb290a358598542e66b3d370eada0df9a60bac996aa23f3b36fdcb656ca9119`。Frozen source／config／order的Mac副本及本地程式相符；22:40 UTC重新確認GPU空閒並驗證全部dependencies後，於22:42:25 UTC在GPU0取得cooperative lock啟動。Controller PID3698873、首個worker PID3699087，actual-start offline ClearML已記錄。首階段為TRAIN query reference encoding，不是正式訓練或canary已通過；其後依次calibrate/S-canary/P-canary。沒有改動已凍結參數。

21:07 UTC現場GPU0–3都已被其他任務使用（0約10.7GiB、3約9.0GiB），故這一輪只做CPU準備／驗證及凍結，不爭用0/3、不改用1/2、不驅逐其他工作。GPU canary必須等重新確認空閒及取得cooperative lock後啟動。沒有把排隊準備說成正在訓練。

22:44 UTC首階段正常推進，GPU0約1.1GiB，其他卡未被占用；主機available約122GB、disk free684GiB。所有resource／exit／tracking／SHA門檻維持原值，未複製active phase output。短期准入邊界按15分鐘監控，取得真實optimizer canary時間後再調整科學訓練chunk及輪詢間隔。

### Sparse Runtime 修復與獨立證據

preflight-v1於22:49:29 UTC失敗閉合：reference encode成功（12,288 query／631,815 NNZ，原batch context parity通過），calibrate載入Mach-O library失敗。兩個phase的exact inventory／SHA／exit／tracking重新核對，S/P均未啟動；44檔3,935,366 bytes已完整Mac鏡像，receipt `f25c0ab63e5af826a0df30f81f37ea503a3008a5e4b51a39ac74cec2adb5828f`。

Linux native runtime-v1在編譯前發現缺少ICU開發檔，保留；runtime-v2從NG8相同八份C/H source，使用獨立解壓的ICU78.2套件建立，不安裝系統套件。Library SHA `976c408efd8588df5bd437bc317986716a8cb5b7bfbfb3eb5a2a9ad7c521c1c9`；complete SHA `e25b311236ac1113eaa77926ec943492621758b2ec0e052d2b51207a30b90f75`。ICU相依以相對RPATH限定；263檔鏡像SHA全部通過，receipt `825e9aa3786428819e6f9f357d4b2f066c322c40fc37686137040b85f4a4255b`。Standalone 12,288題token parity通過，但當時尚未涵蓋tmux locale差異。

preflight-v2於23:22:51 UTC被token gate擋住，exit1／error=null／owned group closed／actual-start offline tracking failed/closed，沒有optimizer更新。固定query `Liverpool F.C.` 在tmux ICU `en_CA` 變成token `f.c`，在Mac及獨立SSH的 `en_US_POSIX` 是 `f`,`c`；三個具體錯例與環境保存在 `lexical-process-diagnostic-v1`。這是可重現的ambient locale依賴，不是模型loss反證。v2的32檔完整鏡像receipt `cc9f20f9367bdfe8b8d358281d2700b8ed2dc1f0565873adab1996fe8eaa84d7`。

source-validation-v2是242個CPU測試通過，但沒有捕獲上述tmux差異。v3加入explicit locale與歷史分數檢查，245個單元測試通過、實際整合失敗；validation SHA `9737840b96c7a4162a25720bf17740439b9cbc675275410a1491daf7aca15065`，完整鏡像receipt `f4a231eba267a74c92da1e38635a2a7b5ffee9b2103ead03256d382e6d8d5249`。不能把245 passed當成此版本全部准入通過。

原因再分成兩類：5,485共同TRAIN query在正確locale下，最大old/new BM25差4.440892098500626e-16，是跨平台FP64求和／scaling路徑的數值誤差；en_CA則有60題差異超1e-12、最大0.5868202083886918。v3新增的score bit-exact gate過嚴，故保留failed validation，另以[協議](ng0080-sparse-preflight.zh.md)明示的正數FP64 forward-error界限修正它。完整token序列仍bit-exact；原模型score／VJP門檻不變。這是驗證契約修正，不是隱藏失敗或調寬模型品質門檻。

source-validation-v4已完整閉合：246 passed／pytest7.52秒，全程序14.99秒，peak RSS803,104KiB，CPU-only、無模型推論。從en_US_POSIX/en_CA/en_US三種ambient ICU locale都回到明定en_US_POSIX，全部12,288題token完全相同；5,485歷史TRAIN分數通過forward-error界限，max allowed bound5.6577e-14，實際max error4.4409e-16。無DEV/LOCKED score讀取。Validation SHA `d7cd47b80186db00e675047263fd1f0fb26fe3a43f6c127e288f9c9c4988c12b`；archive SHA `c3f2d955398233879a6722ca925e129c85765b7254a82e90a1d0fd4b59c378a8`。Mac以stdlib串流驗證完整2,419檔archive和全部receipt，mirror SHA `e1ae295c627ebf887013a966057e032bc5d259e3783ace922c87e470fb2a4d90`。

preflight-v4 input SHA `4ac4fca9342bf12fc25218f31b5bdc6b64da5e6ce193b9b4c410b6d5c268087b`，runner SHA `bf31c770f94e861b2e5050d790a1fed17a5b0bc27b2d16ba977cf12db5342c94`，native wrapper SHA `44f87491469aa2493157241b62b5df5d6bbee5e09ee186900f03ea08e452d2d0`。直接綁定v1成功reference，不重新build、encode或採用failed/canary模型；config/order與v1完全相同。GPU0派發新controller，phase為calibrate、S-canary、P-canary；原cooperative lock、資源和閉合檢查不變。

### S/P 真實准入閉合

v4 controller於23:55:44 UTC記錄 `all_phases_complete`。另行驗證全部dependencies/source、三phase的exact inventory／SHA、exit0／error=null／owned group closed，以及actual-start offline ClearML的closed/passed狀態；不能把這個當成完整科學訓練或遠端ClearML同步完成。

- Calibrate：12,288 TRAIN等權尺度 `a=9.667940425864051`，原RMS不變、全部正例保留；獨立sparse score重算最大差7.1054e-15。全部token序列與5,485共同歷史TRAIN lexical分數的既定檢查通過；沒有使用diagnostic/DEV分數調尺度。
- S/P各16次真實更新、64 query、2,085 candidate exposures，query/document logical tokens均為1,146/282,109。三域full graph對fresh VJP的gradient relative L2均為0，初始完整batch score及完整checkpoint／optimizer／forward reload均通過。這些canary權重僅保留作驗證證據，不能用作正式訓練parent。
- 更新迴圈計算S25.954秒、P25.219秒；以既定1.5倍裕量加300秒估計，1,024-update chunk分別2,791.58／2,721.01秒，兩者共同安全chunk為1,024。這只是chunk排程准入，尚未授權任何新runner自動跳過凍結、測試或終點驗證。
- 實際整個phase elapsed為calibrate56.42秒、S77.34秒、P377.56秒，最高RSS3,040,636,928 bytes。P的全phase明顯長於optimizer迴圈，因此不能把loop時間當端到端時間；後續保留完整wall time／tracking close／checkpoint計時，不推測額外時間來源。全部phase仍低於原5,400秒界限。
- 遠端根目錄精確inventory為67檔、1,224,903,025 bytes，inventory SHA `8444647b5cb431bc25ea43dfbbde2c730daa3a367b0941ba051be00a7900f0b6`。完整Mac傳輸exit0後，2026-09-14另以stdlib核對精確root/phase inventories、全部SHA、input/source及controller/phase closure；mirror receipt SHA `c7414d16d641adb679b0ba68d431c0e1fe4fbe3cbf9b4386752cdbd80646ea92`。此單元已無active transfer，沒有刪除來源或在Mac載入模型。

正式訓練、獨立review及full-corpus hybrid評估未完成前，稀疏表達瓶頸仍屬未識別。

### 完整 S/P 訓練已派發

[固定預算協議](ng0080-sparse-training.zh.md)保留原loss／RMS／資料／順序／所有正例與3,072更新，使用共同1,024-update chunks及fresh document VJP。新增runner不使用F/D專屬projection holder；保存並重載全部encoder參數、AdamW moments/step與兩角色輸出，另記checkpoint/reload、tracking close、完整wall time。沒有新增品質選擇或放寬原score、VJP、資源界限。

Source validation-v1為259 passed/2 failed：新隨機toy fixture的ReLU全部為零，被既定nonzero-gradient gate阻擋，沒有研究模型訓練。修正只有測試用已知active、非constant權重，同時增加dead-feature必須失敗的負對照；validation-v2為263 passed。進一步靜態review發現freeze先複製config後以exclusive-create重寫會衝突，在實際凍結前改為只新建完整訓練config，保留原canary config不可變；新增freeze/re-freeze測試。

最终source-validation-v3：lambda2 CPU **264 passed**，pytest8.71秒、程序10.72秒、peak RSS808,432KiB，exit0，無研究模型推論。Validation SHA `d6a7be63be3792fba7382504441366c2f5c127069f06dddb1bc1146b26bb22e1`；source archive SHA `0c1996768f18582e3255a3a03d12ece846c1989786dfdd2b89afb64a794f7f60`。三個validation的source archive、逐檔inventory及頂層receipts已在Mac核對，v3全部2,422份source內容一致，mirror receipt `51de81386654dd9ae35d5b3be5b1b5d494482d8b58f28b0d83538cee3760ce04`。遠端extract workspace及pytest temporary files未鏡像，也未刪除；不稱為整個遠端validation目錄的完整副本。

| 單元 | GPU | Input SHA256 |
| --- | ---: | --- |
| `NG-0080/sparse-S-v1` | 0 | `35d02db27758f7dc9409639efa051882d54fc20b0d7e6c38ac46e00d6d04d109` |
| `NG-0080/sparse-P-v1` | 3 | `23c6d49d74cd83ba34f70e193379225a74cdcd3a0fcdfc02e7b2e306e6a794f7` |

共同runner SHA `dd939b9c551c325c122ba997ac43c26c7952c683b2aa62b2557efc6f71460f33`。各22份frozen source/config/order及input的Mac副本、主倉庫runner/test與remote所有dependencies均在派發前核對。沒有複製active phase output。

00:43:57 UTC兩個controller各自取得GPU cooperative lock，S PID3779120/worker3779177；P PID3779122/worker3779175。Actual-start offline ClearML分別 `offline-5baf492a552a4e0d91ae39a4696cc04d`、`offline-c83c9a79d5af4cceabd908ecdd021cbf`，尚未closed或server-synced。第一個完整未更新batch通過原score核對；兩臂前15更新有非零梯度及參數變化，這不是終點品質成果。

現場GPU0/3各約2,726MiB、1/2維持空閒；host available118,626MiB、disk free683GiB，沒有驅逐其他工作。第一chunk各自前16更新ETA gate已通過：S21.8369秒／預估2,396.34秒，P22.6762秒／預估2,476.91秒，均包含既定1.5倍裕量與300秒。兩臂已記錄64/1,024更新，尚未phase閉合。監控調整為約30分鐘有效邊界；中途loss僅供執行診斷、不用於選checkpoint。訓練全部閉合後仍需獨立audit及fresh全庫initial/S/P hybrid、PPLX與dense+BM25比較。

最後只做Mac stdlib静態檢查：17份NG80 Python AST、10份文件的147個本地連結，以及tracked/untracked變更的whitespace檢查通過；未在Mac執行Torch或模型。最新ledger/導覽文字更新晚於已凍結source archive，沒有回寫frozen run。

### 首個完整 Sparse Chunk 已核對

2026-09-14 01:25 UTC另行讀取frozen runner，重新驗證兩臂inputs/source/dependencies及train-1024的exact inventory／全部SHA、exit0/error=null/owned group closed和actual-start closed/passed offline ClearML。S於01:08:02 UTC閉合，P於01:12:47 UTC閉合；各1,024更新、4,096 query、134,254 candidate exposures、query67,658/document17,797,680 logical tokens。全部encoder/optimizer/query及document reload均exact，兩臂都從共同原模型開始。

| 臂 | 完整phase秒 | compute秒 | checkpoint/reload秒 | tracking close秒 | Peak RSS bytes | Complete SHA256 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| S | 1,444.734 | 1,386.294 | 5.751 | 8.822 | 3,047,018,496 | `a3052a1880c842f4777d33cf376c53be67c0c85406f6443e3d0674669052c98e` |
| P | 1,730.137 | 1,370.553 | 5.963 | 309.970 | 3,134,922,752 | `66af960b398ddd5d408f24ef80dffe5590e89668f58fb7566ccbab9d260b0438` |

P的額外約300秒現在由實際計時定位到ClearML `task.close()`，不是optimizer迴圈；stdout另有repository autodetection的SystemExit warning，但offline session已成功保存。不把這個當成模型失敗，不修改或重啟健康的frozen graph；warning與close內部等待的完整因果仍未證實。這也不授權改掉5400秒phase界限。

兩臂已自然進入train-2048，自己的16-update ETA均通過（S預估2639.30秒、P2356.36秒）。01:19 UTC stdout分別達1472/2048、1280/2048，GPU0/3約2,850MiB、GPU1/2未使用，host available118,624MiB、disk free681GiB。這些是進度不是終點quality；沒有copy active phase、沒有用中途loss作選擇。

等待期間新增[獨立S/P訓練審計](ng0080-sparse-terminal-review.zh.md)source/tests/protocol，與active source分離。它重算S與P的全部scalar loss/score derivatives，並以原encoder registration order驗證完整optimizer state，不誤用F/D的sorted ParameterDict。此審計尚未對完整科學訓練執行；full-corpus endpoint runner尚未凍結或運行。

新source-validation-v1在lambda2 CPU通過 **275 tests／9.43秒**，程序11.58秒、peak RSS806,448KiB、exit0；source前後2,425檔完全相同。包括兩種loss、不同positive mask／無eligible邊、score derivative mutation、non-sorted encoder registration order與corrupt optimizer負對照，沒有研究模型推論。Validation SHA `c6ed7a678fbff37d4ad1983dded8a7e622883099c601e8cd220dd6668eaddcb9`，source archive `d62e34cea6429607cc2d5b64a70442a471c67c30ffbb74af649fb61af4682140`。

Mac傳輸exit0後，以stdlib重新核對四份頂層檔案、完整archive的2,425個source member及三份新source/test/protocol與主倉庫一致；mirror receipt `722a1774ba870623c2d82878f05bf92ca7ad3d3c3c0cb9112d87b0d391c9f078`。Remote extracted workspace及pytest temporary files未copy或刪除。這不是完整training output mirror，亦不是actual training audit已通過。新reviewer SHA `fcb01a3cee021f78bccf3dce6571187d8b73010cfcfe3bf0e0ab58091663755f`、test SHA `95fbfb02efdb769303d58b478e2765f1c6fff88490c12023ecd6e2d417d029e3`；active runner SHA保持不變。Mac只做19份NG80 Python AST、11份文件的152個本地連結、catalog JSON、line-length及tracked/untracked whitespace檢查，不載入Torch或模型。

### 兩臂訓練完整閉合、準備全庫終點

2026-09-14重新現場核對：S controller於01:56:46 UTC、P於02:06:32 UTC成功結束三phase。六phase的exact inventory／SHA、exit0/error=null/group closed、actual-start closed/passed offline tracking全部通過。各3,072更新／12,288 query／402,691 candidate exposures／query206,563與document53,510,807 logical tokens，same-arm optimizer/checkpoint continuation和worker exact reload均完成。這是完整訓練執行閉合，不是終點retrieval已通過。

| 臂 | Terminal model SHA256 | train-3072 complete SHA256 | 全目錄檔案／bytes | Remote inventory SHA256 |
| --- | --- | --- | --- | --- |
| S | `d9c94a6eb42c831fcaf0e49218d557a7192a01ba1b7d077941ea901ffabade73` | `ed10d032db953836d179c336fa326123a4ad3a415c229b87fec72fbdf3adc231` | 71／1,832,547,152 | `972a899dd62da3e94845cff3e6f8e7cb1ca30218ceef3ba05e32419906036043` |
| P | `cccf59f95b8bb138305dd7305684fec15ab2aa43eced31ce46b9422fde8fbfe8` | `46b8ab7142745bab549c545f7a6e3e56fc582a1d1f60b8a8b299caa9ae6ce749` | 71／1,832,117,703 | `31227147225d9da1d2ddd9255b711f0021449660336abd2396839cd7b40e0988` |

S後兩chunk完整wall1471.24／1448.80秒，P1455.88／1765.00秒；所有phase peak RSS<=3,173,908,480 bytes。P最後tracking close309.29秒、compute1403.82秒，與首chunk的close等待相同類型，保存成功。HF的decoder.bias MISSING提示由原strict checkpoint loader恢復canonical bias alias，原exact model hash／reload門檻未移除，不把已覆核的alias提示說成隨機丟權重。

開始closed S/P完整Mac鏡像；完成與SHA回執另記，不能以rsync已啟動當成鏡像完成。02:05 UTC GPU1被其他工作使用（約9086MiB），未碰觸；0空閒、3當時仍在P close，新的GPU任務必須重查資源及鎖。

新增[全庫終點協議](ng0080-sparse-endpoints.zh.md)及独立runner，在新diagnostic排名之前固定。先審計、再三模型fresh全庫編碼，同時準備TRAIN-only dense融合校準（384固定TRAIN題、三域各128、完整背景、固定七個權重候選），不以diagnostic選blend。Sparse排名使用active CSC columns及獨立posting accumulator核對，並不宣稱已測native WAND。

新endpoint source-validation-v1：lambda2 CPU **281 passed／9.82秒**，程序11.95秒、peak RSS808,604KiB、exit0；2,428 source檔前後一致，無研究模型推論。Validation SHA `f3a0a3239c66e6d2ef325da8f101ca993fbb2e14e87f32affe9fabf690afd382`；archive SHA `b9966c901482dbbe48992ba1c68b2800cd84b62a86fce368a87533706f97aa26`。Runner SHA `b074cfb3f632ee73029096f3f3f64896b402019f8fe61454a3159b08575dac44`，test SHA `2c85e533ed17bd0c9232f7d4b8607633182513069a3fe39760b9c7d37e6c72d3`。測試包括全背景／empty query、雙reduction、normalization、TRAIN校準tie、split以及fresh freeze／unfinished parent／tamper拒絕；不是actual training audit或endpoint quality已通過。

### 全庫 V1 審計失敗與數值定位

`sparse-endpoints-v1` input `ea4d25f2ba81e7a941447fc0d7d0c6b405fd8a2536ba22afb0b4d9dae9edaf78` 綁定已閉合S/P；source validation-v1全部2,428 archive members、四份receipt及28份frozen source已在Mac核對。02:24:04 UTC啟動CPU audit，02:25:37 UTC失敗閉合，沒有啟動任何endpoint inference。原phase93.39秒、peak RSS2,833,178,624 bytes，exit1/error=null/group closed、actual-start offline tracking closed/failed。失敗complete SHA `a09550546770faf9296522fcbfe111adf72f52077f71f0503e24c3b8c1eb39b5`。39檔／457,482 bytes完整Mac鏡像已逐檔核對，inventory SHA `9b3bceb884d423377948fd23351e2823b810cfa4fc4293e605349293d5c10b48`，不會把它改成成功。

已走過S三chunk審計，P第一chunk第7更新query index11716的loss校验失敗：record0.0003524725616443902，理想FP64 target重算差9.0688e-8。審計把訓練的FP32 sigmoid target改算為FP64，而BCE中`(1-p)*student_margin`會放大接近1處的target誤差；不是僅以相對誤差小就放行。

對全部12,288個P記錄做有界CPU scalar replay（沒有模型或optimizer），舊門檻1,491筆不符；使用FP32 CPU運算剩2筆，最大loss差1.1920928955078125e-7、最大gradient差1.4901161193847656e-8。剩餘是train-3072的update2051/query13664和update2723/query12571，loss差各4.5751e-8／4.5897e-8，gradient差<1e-9。不能因為接近就直接放寬門檻。

新增固定三case的`scripts/ng80_sparse_numeric_probe.py`（SHA `3cc4fe3b772a7c9321ae6b9d779a5eedf07c9caef1957800b44899caa753da83`）。GPU0已被其他任務占用，第一次readiness檢查拒絕，未載入CUDA；重新查看後，GPU3空閒，在cooperative lock下執行scalar-only probe1.41秒。三case的CUDA loss與score gradient對原記錄均為exact zero error；兩個殘餘case的CPU/CUDA target差均1.1920928955078125e-7，使用CUDA target後獨立FP64穩定loss與記錄差分別約1.50e-11／5.11e-12。Torch2.9.1+cu128/CUDA12.8；沒有模型推論、新optimizer更新或完整audit准入宣稱。

V2只修審計數值契約：teacher probability重放同一FP32 CUDA算子，loss/score derivative仍由獨立NumPy FP64公式計算。`rtol=3e-5, atol=2e-8`、S/P frozen training及所有資料／RMS／步數不變。審計lane須占用一張空閒0/3卡，checkpoint/optimizer仍CPU核對。增加飽和target和拒絕錯誤scientific backend測試，lambda2 CPU **285 passed／10.01秒**，程序12.25秒，peak RSS703,385,600 bytes；全部2,429 source前後一致。Validation SHA `e9c01001f33a7fd3cfdb5cd625e3eab2b77052bf25ccf40e1b76f9c37681e681`，archive SHA `38b4e7c53353ddac47924dfc20e05507f24aecfa2d200924cde3a4864068a983`。必須新建v2，不能重啟或覆寫v1。

V2新input `44c8feeae1c437652cb4baff685bcf635f32244834bd8cf8c078ad02238b52f2`、runner `e35885f5c12be9f22d6712ce3f254ac87ca9d5feebcb101a38452a5bdfa2b5a2`、reviewer `41e227aa08bc6758ec515dc9675e54dfe70385b75d33b1f8f8eb96f4834d0015`。Freeze與全部dependencies驗證完成後，Mac核對28份frozen source/input。首次copy比freeze完成早，endpoint路徑不存在而exit23；沒有endpoint partial output被複製，另一次完整copy exit0後才認定source mirror完成。Validation-v2四份receipt/全部2,429 archive members及scalar probe四檔的remote/Mac SHA另行一致，未刪除任何來源。

V2 dispatch嘗試後沒有controller-start或phase-start、tmux也未存活；再看GPU0/1/3均已被其他工作使用，GPU2不是本campaign允許的替代卡。因此V2目前是**已準備、等待空閒0/3，不是正在審計**；不驅逐其他工作，也不放寬guard。Full S/P Mac鏡像仍在傳输，不以部分已到達當成完成。22份NG80 Python AST/行寬/whitespace、13份文件171個本地連結、catalog JSON及git diff --check通過；最新ledger/navigation更新晚於source snapshot。

### V2 全量審計閉合與鏡像收尾

2026-09-14 03:20 UTC重新查看lambda2，GPU0/1/2/3均空閒，host available122,375MiB、disk free678GiB；沒有沿用先前忙碌或空閒的假設。V2 audit在GPU0 cooperative lock下於03:22:04 UTC開始，03:24:06 UTC成功閉合。獨立重驗全部frozen dependencies/source、phase精確inventory/SHA、exit0/error=null/group closed、actual-start closed/passed offline tracking及controller，完整phase121.72秒，peak RSS2,999,623,680 bytes。

兩臂各12,288題全部通過loss/score-derivative及identity/exposure檢查；S最大gradient差1.7281122745901184e-10，P9.022648342341317e-9，原門檻不變。各臂三checkpoint均重新hash107個name-bound optimizer parameter states與完整model，總query/document logical tokens206,563／53,510,807、candidate exposures402,691一致。這完成了V2的全量scalar／序列化審計，不是完整parameter-VJP/AdamW獨立重演或終點retrieval品質結論。

Audit complete SHA `fe120598a3f81a5e345da0280d5438063b6c383c50c52da3046f10cb9b91bf12`，results SHA `7c5ce4855d3f9bb2cfed609e6b47f41ae71102b984d6c5bb76cd1fe7a220e2e0`。已閉合audit/controller/log共13檔46,614 bytes，inventory `5d94354a43863c458c2b47bd67bc2f18fa5ed76f66d99373151fa2810216bcf5`；在其他lane派發之前完成Mac傳輸exit0與全部source／audit SHA、closure核對。這只是audit mirror，不把尚未執行的完整endpoint算作鏡像完成。

先前S/P完整training傳輸已exit0，Mac以stdlib重驗兩個exact root、六phase、全部source／SHA／controller／tracking closure。S71檔1,832,547,152 bytes，mirror receipt `90f95d477ee4254a464a9dc58f8cb1b45b5cf64b5da18ab1d1aeed3e19ce2dd8`；P71檔1,832,117,703 bytes，receipt `fe09ebc69812f98321917c6bb621b1995424c9fca23b1c340fc9208a0e0d84e5`。兩臂training鏡像都不再pending。

等待期間亦收尾NG79既有closed endpoint鏡像，不重跑研究：remote/Mac完整159檔910,005,411 bytes、九phase closure及source全部一致；inventory SHA `8d4651dd8560f510d4872ec71918c9b0f7311a005c504b06b00a49ec5258601a`。原review SHA及scientific failed結論不變，沒有把存檔完成解讀成模型改進。沒有讀取locked query、進行Mac模型推論或刪除來源。

03:30:53 UTC，audit barrier核對及已閉合資料鏡像完成後，三條lane各自派發：prepare controller3911752/worker3911910（CPU），first3911761/worker3911922（GPU0，initial之後S），second3911769/worker3911923（GPU3，P）。均取得各自cooperative lock與actual-start offline ClearML；first/second首階段不是新訓練，也沒有借用canary checkpoint。

Initial/P前768文件canary各2.3734／2.6748秒，保守全phase預估1,380.13／1,517.29秒，原5,100秒准入門檻均通過。讀取stdout時兩者均已超過21,600／233,009文件，仍在正常編碼；這不是全phase或publication完成。GPU0/3各878MiB，host available117,193MiB，GPU1出現其他程序400MiB、GPU2空閒，均未碰觸。沒有copy active phase output。

全部S/P training、已閉合V2 audit與舊NG79 endpoint的Mac鏡像均已完成，沒有遺留active transfer。新V2的encoding/calibration/ranking仍須按各自closure、全背景計分及terminal reviewer通過後才解釋結果。既有30分鐘節奏接近首lane兩模型的正常完成邊界，不因健康進度反覆重啟或發通知。

### V2 編碼閉合與 Tracking Abort 的有界修正

04:11 UTC重新現場檢查：V2 first/second controller已成功閉合，三個全庫encoding皆exit0/error=null/group closed/actual-start closed-passed offline tracking。逐檔重驗全部source、dependencies及五個已結束phase的精確inventory/SHA，沒有新訓練。

| 模型 | Document NNZ | Query NNZ | 完整phase秒 | Peak RSS bytes | Complete SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| initial | 63,312,968 | 79,009 | 706.95 | 3,075,022,848 | `9366cc5b86946fe31875ed008eeecd33cf47f1486075a111e9b806de9ac4a208` |
| S | 254,721,463 | 221,474 | 757.18 | 6,236,004,352 | `afcd13d26e71cbfd0234aca9a475c59b6850851e675c4d9223917db529996d26` |
| P | 292,984,531 | 95,000 | 812.04 | 7,039,868,928 | `185c9e8228257a36e0f7c95956b1a428b46131488d265a170013a0c97d44fe45` |

各233,009文件／1,536診斷題與原模型SHA一致。Document NNZ明顯增加，只能記作增密觀測；還不知道完整排名是否受益、DF-work是否上升，不能由此宣稱native成本或新模型品質。現場四GPU空閒、hostavailable122,615MiB、diskfree675GiB；swap舊占用1,014MiB不等同當前記憶體壓力。

同一V2的`prepare-lexical`已產生兩個矩陣、reference replay exact及closed/passed offline tracking，但程序在359.07秒以SIGABRT（exit=-6）結束。Stdout先有`Failed auto-detecting task repository`，再有`_enter_buffered_busy`／interpreter finalizing／stderr lock的fatal錯誤。Tracking close本身309.88秒；controller正確停在failed_preserve_attempt、completed=[]，未啟動calibration或ranking。保留complete `c293bb19f3cb4a2a46323bb3e78a5bb9bc2a2daeb8463b38a0b4082d8031ec9f`，不把written results升格為成功。

已核對已安裝ClearML原碼：`auto_connect_frameworks=False`仍走`detect_repo=True`，dictionary的`detect_repository=False`才關閉該分支。修正明列全部framework false；保留正常task.close、actual-start記錄、process exit與原資源門檻。不使用os._exit、不吞abort、不升級套件。三次獨立真實offline subprocess確認repository thread未啟動且正常exit0；它不保證所有可能的SDK race皆被排除。

Source validation-v3：**291 passed／55.53秒**，程序58.06秒、peak RSS1,636,958,208 bytes、exit0；沒有研究模型推論。Validation SHA `4f0ea06dda5c90b46a9224178005456dda5aca44f66f7e7aa450e19d91cd1b1b`，source archive `82377241eaaa11df2868c437bd779f309814f812d2ab501478d0d3194c672ce4`。Extract inventory4,858檔包括2,429份source與2,429份Mac tar AppleDouble metadata，前後全部不變；不把metadata算成新增source。Mac另核對四份頂層檔案和所有archive member SHA，三份修改source/test/protocol與主倉庫一致。Remote workspace／pytest臨時資料未copy或刪除。

V3使用[明確的閉合phase重用契約](ng0080-sparse-endpoints.zh.md#v3-tracking-退出修正與閉合產物重用)，只重用V2的audit及三個encoding；失敗lexical必須fresh執行。沒有複製大矩陣、symlink或假造新execution receipt。AST另外核對selection、encoding、lexical準備、校準、兩種score reduction與normalization等十個scientific函式，與V2完全一致；rank/review僅改已封存encoding路徑解析，計分公式不改。

V2所有lane已終止，完整目錄109檔3,270,331,961 bytes的remote inventory SHA `ce2962b54bb706f9b9ce56d9c7bbfd76b39dacdb767feeaffeb743f615f1845d`。已開始完整Mac鏡像，包括失敗phase；尚未以傳輸啟動冒稱完成。

V3 input SHA `aa98a6bf917c2e5e0ea901d0fd20223f2a33ecc974a3120aba6f0a6b6f123642`，runner `5faed2779920b164892c050a3d90f8cbe2cbbbff68d450aadae73f4bca224b72`；28份frozen source/input在派發前完成Mac exact inventory及SHA核對。CPU prepare controller3948531／worker3949172於04:24:33 UTC開始，actual-start offline task `offline-fecf3de170e442fead98b67b53a2e135`。

新的lexical準備於04:25:51 UTC以exit0、error=null、owned group closed正常閉合，完整phase77.71秒／peak RSS959,508,480 bytes；tracking close7.84秒，沒有原repository／fatal shutdown訊息。兩份矩陣、query IDs和results均與V2舊產物SHA一致，沒有把tracking修正變成新科學方法。`calibrate-dense`已actual-start：`offline-fd6f16aabe0640c5887001798b4eebdc`；完整calibration及ranking尚未閉合。

現場runtime為Python3.12.12、ClearML2.0.2、NumPy2.2.6、SciPy1.17.1、Torch2.9.1、Transformers5.15.1，沒有套件升級。已安裝ClearML task.py SHA `454bdecc16e60ae8f85742cf7774eba4de1be05ccf3b36f916c377e424f381fd`另記外部sibling observation，不回寫frozen unit，也不冒稱依賴lockfile。全部active V3 phase仍不copy。

TRAIN校準於04:29:39 UTC完整閉合：227.70秒／peak RSS3,096,477,696 bytes／tracking close9.13秒，complete SHA `431a7c9c4a85922c56472d4fda9767887574e3a47c0562ba34cac1b6e53819ac`。先前16題canary6.21秒，保守預估523.67秒通過原界限。獨立重新核對全部384 query identity、七組metric平均及既定tie規則，選定alpha=0.1；TRAIN nDCG0.804201／Recall0.980475，純dense為0.797628／0.973303。這只是固定TRAIN校準，不是新的diagnostic或泛化收益。

在六個先決phase（四個V2重用、兩個新V3）精確inventory/SHA／exit／tracking和全部source/dependencies重新核對後，派發CPU ranking lane `ii42_ng80_sparse_endpoint_ranking_v3`，tmux pane PID3955338。它依原順序執行baselines、initial、S、P及review；實際worker開始與終點結果仍須另驗，不把tmux存在當成計分完成。準備階段短且排名有多個有效邊界，後續輪詢調整為15分鐘；完整V2 Mac傳輸仍單獨追蹤，沒有刪除或生產變更。

後續實際開始已確認：ranking controller3955338／worker3955500於04:32:26 UTC啟動，`rank-baselines` actual-start offline task為`offline-d8d59b11e1e7452a9897db1da7ebf90e`。前16題4.64秒，預估968.39秒通過原5,100秒canary gate；host available119,767MiB，沒有資源壓力。這仍是進度，非五phase或terminal review完成。最終本機檢查22份NG80 Python AST／行寬／whitespace、13份文檔172個local links、catalog JSON與git diff --check通過；未在Mac載入模型，未提交或推送。

### 2026-09-14 05:07 UTC：四組排名閉合，終點審核仍運行

V3 baselines、initial、S、P 各完成1,536題全庫排名，獨立核對四個phase精確inventory／所有SHA、exit0／error=null／owned group closed及actual-start closed-passed offline tracking；frozen source/input SHA仍一致。四組完整phase耗時分別532.64、182.27、367.13、227.89秒，peak process-tree RSS分別3,147,657,216、2,664,456,192、7,258,734,592、8,174,264,320 bytes。這包含audit、ranking和tracking等工作，不是線上query latency。

| Phase | Complete SHA256 |
|---|---|
| rank-baselines | `b5d42621a6f486840043900513592a227d02707209c9c97c8f7091bd7fd4fbec` |
| rank-initial | `267ddb0ccfb4be149b760411767212069e8831a1bc01ba5e956b3178f9d63049` |
| rank-S | `4332b73e5f58d863fe84c15f46410512a623c772a7e950747eccd32254cb68db` |
| rank-P | `f27b7b69599c3205ba8fd3cfa364706fbc7e1c1655f761fcc3e6db78cbf78e6e` |

review於04:54:16 UTC actual-start，worker3968815／controller3955338，offline task `offline-84b502ad934a4583877f41d5830db6bc`。05:06末仍在CPU計算，elapsed約12分15秒、主worker RSS2,532,200KiB，host available120,834MiB、disk free675G；沒有review results／exit或ranking controller terminal receipt。空stdout不是失敗證據，review本身沒有逐題progress輸出；不重啟或改寫健康worker，不在terminal audit閉合前公布品質判斷。原5,400秒phase限制保持不變。

V2完整Mac鏡像此輪已傳輸exit0，109檔／3,270,331,961 bytes，精確root清單、每檔size/SHA、input、五個phase及四個controller閉合再次核對通過；新證據寫在外部sibling `sparse-endpoints-v2-full-mirror-review.json`。原audit-only回執保留。這包括原lexical exit-6失敗，不把完整保存誤記為整個V2科學成功；沒有V2 calibration/ranking，沒有刪遠端來源，也不把同一Betty上的副本說成独立災難備援。V3 active review輸出尚未copy，完整study仍為七個fresh phase加四個明示重用phase。

### 2026-09-14 06:11 UTC終點閉合；15:54 UTC起重新核對

V3最後review於06:11:54.500 UTC正常完成，exit0／error=null／owned group closed，actual-start offline tracking closed/passed。完整phase4,657.58秒、peak process-tree RSS9,218,383,872 bytes，沒有放寬5,400秒／16GiB限制。Ranking controller為all_phases_complete，五phase皆列入completed；新現場檢查已無NG80 endpoint程序。使用者已將自動輪詢設為PAUSED，本次未恢復。

再次核對27份source與698項dependencies的SHA、全部11個邏輯phase精確inventory／所有SHA／exit／tracking閉合。四組ranking各1,536題的gold-rank指標用stdlib獨立重算，各域及macro一致；此步不是重新模型推論。Review results SHA `365aad14fe4738786d7d38f699333a026e7932a51e51db8910ed50f0133814f7`，complete SHA `737c7c6fe9125fecb249974f0a2230b48ab08d813fe0bb95857ea54e4c1d7e62`；原review head-coordinate最大誤差1.0658141036401503e-14。

[完整S/P終點結論](ng0080-sparse-endpoint-review.zh.md)：原hybrid／S／P／dense／dense hybrid的macro nDCG依次0.770716／0.786079／0.775382／0.822175／0.820317，Recall依次0.948846／0.951925／0.946605／0.965235／0.971423。S相對原hybrid的nDCG差+0.015363，95% paired bootstrap CI[0.008176,0.022589]；Recall差區間包含0。S與P的FEVER皆上升，但HotpotQA/NQ均有點估計回退。P的macro nDCG增益區間亦包含0。這不是獨立holdout或多seed結論。

S的panel field MSE下降約63.0%，document NNZ卻為原模型4.02倍，semantic DF-work11.12倍，加lexical後literal總DF-work7.57倍；P分別為4.63／2.47／1.95倍。DF-work不是unique candidates、native實際訪問或查詢延遲，不把代理倍數說成线上慢多少倍。品質／低總成本雙重目標未達成；原模型保留，兩個新模型不晉級。其他監督、context、joint mask、native成本與獨立泛化分支尚未全做完。

V3完整已閉合root已Mac傳輸exit0：118檔／41,784,192 bytes，精確file list、每檔size/SHA、source/input、七個fresh phase及兩個controller再次驗證。Inventory SHA `d8caf10a46434a4537fc7e92c83720df85cf953b25e5882585086671fd3c53f2`，新外部sibling `sparse-endpoints-v3-full-mirror-review.json`記錄覆蓋範圍。四個重用phase仍在已完整鏡像的V2；未覆写凍結單元、未刪遠端來源、未啟動新訓練或生產操作。

## 下一波固定原則

先用 F/D 對照測 continued adaptation 是否有增益，再用原 sparse 輸出直接約束 teacher score-field，與正例保護路徑比較。F/D 只改 trunk 是否更新；S/P 比較不同監督配方，不把多個 scoring/loss 差異說成單因素因果。語義 fidelity 與完整 hybrid relevance 各自報告。

每組保留共同初始模型、資料順序、所有正例、query/document token 限制、RMS 與累計 query exposure。短 canary 決定安全 chunk 大小，不依中途 nDCG 挑選 checkpoint 或改 loss。訓練要保存 optimizer chain、fresh document VJP、一致 reload 與資源峰值；全部對照閉合之後才決定是否追加 pretraining 或擴大多元 supervision。

若品質只能以增密取得，保留為研究 quality parent，不部署。成本約束安排為能力診斷後的單獨 matched control；不要求它在當前資訊尚未充分學會時把模型一併壓垮。反之，也不能無限推遲總成本與完整 hybrid 的驗證。

## 驗證紀錄

- 第一波 selection/panel/ridge/score-field/metric 的合成測試：lambda2 `11 passed`。未讀實際 locked query。
- 加上新訓練 primitive、監督審計及三組文檔回歸：lambda2 `170 passed`，5.41秒；程序7.21秒、peak RSS781,756KiB。F/D orchestrator 的後加測試另記，不沿用此數冒充已測。
- F/D orchestrator加入後：lambda2 `173 passed`，5.89秒；程序7.81秒、peak RSS792,668KiB。序列化修正後監督審計6個測試另行通過；新增 JSON 測試不是沿用舊測試計數。
- 全部修正後完整重跑：`174 passed`，5.85秒；程序7.81秒、peak RSS793,244KiB，exit0。`source-validation-v1/validation.json` SHA `938ddafa9180fffd97a8c5eb9641a36ab2bc11f0be75e21e96811efd31aed2f9`，封存2,409份 source files；source archive SHA `fd8827b88ce24c98a1aab5d4e9ed9a2eb66a2f924cb0efa76ddcf184dad4bb32`。本行及鏡像狀態屬測試後紀錄更新，不回寫 frozen source。
- 最終結論必須逐項標為支持、反對、尚不能識別或需新增資料；任一主要分支未閉合時，不發布「全部問題已解決」的結論。
