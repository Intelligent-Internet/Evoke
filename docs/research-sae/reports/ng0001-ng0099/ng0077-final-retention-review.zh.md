# NG-0077：局部排序保留沒有修復跨 query 品質

2026-09-13 UTC。**Z／K兩臂、九個終端phase、完整Mac鏡像與本地核對已完成；事前品質門檻失敗，不擴大或晉級。** 本輪不是未完成的訓練，也不是執行錯誤後的重試。參見[配對協議](ng0077-matched-retention-pilot.zh.md)與[終端執行契約](ng0077-endpoint-protocol.zh.md)。下面區分科學結果與交付驗證。

## 1. 完整混合品質

比較同一成熟NG3基座、384題各一次／96次更新。Z是原D objective，K只加lambda1單側可信margin保留；全部660正例、45,175 anchors、順序、teacher、完整BM25＋semantic計分不變。兩臂不是SAE單路，也不是重新訓練latent字典。

全庫233,009文件。主要surface為384題TRAIN_SENTINEL（三域各128），另報實際參訓的384題TRAIN_PILOT。Sentinel已在前輪被觀察，不是獨立held-out；沒有DEV／LOCKED_TEST評分。

| 完整方案 | Sentinel nDCG@10 | Sentinel all-positive R@100 | Pilot nDCG@10 | Pilot R@100 |
| --- | ---: | ---: | ---: | ---: |
| 初始hybrid | 0.765559 | 0.965647 | 0.744532 | 0.958550 |
| Z96：原D | 0.763061 | 0.962934 | 0.786903 | 0.965712 |
| K96：D＋保留 | 0.764426 | 0.962934 | 0.787384 | 0.968316 |
| PPLX dense | 0.828470 | 0.978906 | 0.810350 | 0.963715 |
| BM25 only | 0.545273 | 0.858092 | 0.510476 | 0.798822 |

以下差值是原始0–1尺度，非相對百分比。固定逐query配對、分域bootstrap seed71071／10,000次；區間條件於這次初始化、順序與已曝光TRAIN面。

| Sentinel比較 | nDCG差 | 95%區間 | R@100差 | 95%區間 |
| --- | ---: | --- | ---: | --- |
| K−Z | +0.001364 | [-0.000187, +0.003210] | 0 | [0, 0] |
| K−initial | -0.001133 | [-0.012851, +0.010425] | -0.002713 | [-0.009983, +0.004340] |
| K−dense | -0.064044 | [-0.084555, -0.044037] | -0.015972 | [-0.028126, -0.004686] |
| Z−initial | -0.002498 | [-0.014273, +0.009047] | -0.002713 | [-0.009983, +0.004340] |
| Z−dense | -0.065408 | [-0.086048, -0.045439] | -0.015972 | [-0.028126, -0.004686] |

K−Z點差未達+0.005，區間下界也未大於0。這不是證明兩方案等價，也不否定所有保留方法；它否定本次固定lambda1、同pool的干預已達預定有效幅度。K仍低於dense約6.40個nDCG百分點及1.60個Recall百分點，不能宣稱接近完成總目標。

## 2. 宏觀平均掩蓋了什麼

| Sentinel域 | initial nDCG | Z nDCG | K nDCG | K−initial nDCG | K−initial R@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FEVER | 0.798729 | 0.847959 | 0.847568 | +0.048839 | +0.003906 |
| HotpotQA | 0.846345 | 0.829719 | 0.832301 | -0.014044 | -0.011719 |
| NQ | 0.651603 | 0.611506 | 0.613408 | -0.038196 | -0.000326 |

HotpotQA／NQ nDCG均違反相對initial的-0.005 floor，HotpotQA Recall也違反floor。FEVER增益不能抵銷這些失敗。K的三域Recall分別為0.959766／0.957031／0.972005；dense分別為0.988021／0.967448／0.981250。

真正參訓的Pilot，K−initial nDCG為+0.042852，95%區間[+0.028776, +0.057437]；Recall為+0.009766，區間[+0.003038, +0.017578]。尤其NQ Pilot從0.621384升至0.691484，**增加7.01個百分點**；同域Sentinel卻下降3.82個百分點。

因此不能再籠統說「模型完全學不動」或「optimizer沒有作用」。本輪明顯擬合參訓題目，但有用排序沒有穩定轉移到其他query。這是觀察，不是已經把原因唯一定位到過擬合、資料不足、某個domain batch或teacher錯誤。

K對Z的Sentinel有11題nDCG改善、3題變差、370題不變；all-positive rank有64項改善／17項變差，但沒有任何正例跨越top100。相比initial則66題改善／65題變差，4個正例進top100、7個離開。不能把局部rank改善計數當作macro品質或Recall改善。

## 3. 保住anchors為何還不夠

獨立CPU核對兩臂全部768次真實query曝光的D／keep loss與score gradients通過。Z parameter及optimizer fingerprint精確回到歷史D96；K的parameter SHA不同。這排除了「新項根本沒有接上」作為本輪結果的解釋，不等於重算了整個parameter VJP。

| 原可信TRAIN關係 | anchors | Z終端margin不再嚴格正 | K終端margin不再嚴格正 | Z／K：rival進top10且positive在10外 |
| --- | ---: | ---: | ---: | ---: |
| FEVER | 11,625 | 8 | 8 | 0／0 |
| HotpotQA | 18,242 | 90 | 82 | 14／14 |
| NQ | 15,308 | 119 | 111 | 14／14 |
| 合計 | 45,175 | 217 | 201 | 28／28 |

原D本來就讓約99.52%的這些可信關係保持正margin。K令失去正margin的數量**淨減少16**，不是已驗證恰好修好16個而沒有新增損失。兩臂平均anchor margin均大幅增加，但最明顯的top10失位計數沒有減少。

這把主要疑問從「能否保住大量已知局部關係」轉向「保留的關係是否覆蓋真正決定全庫前排和跨query轉移的約束」。小pair平均loss不能保證每個重要pair，也不能保證其他query；shared parameters會同時移動query與document表示。原pool之外競爭者、未受監督／teacher相反的關係仍不能假造負例。本輪沒有證明這三者各占多少因果份額。

## 4. 成本結果與非結果

| 相對initial | Z96 | K96 |
| --- | ---: | ---: |
| semantic document NNZ | 0.798481x | 0.803210x |
| Sentinel semantic query-DF工作量 | 0.292093x | 0.298127x |

K有50,853,614個semantic document nonzeros，比initial少19.68%；未剪枝query-DF代理少70.19%。但它略高於Z，且品質未過門檻。原始document CSR為407,760,952bytes，不是PostgreSQL索引的physical bytes。沒有本輪native latency、block visits、完整hybrid總成本或維護成本驗收；不能把代理減少叫作已達降本目標。

## 5. 下一步：不再反覆改係數

先做一個有明確問題的TRAIN-only唯讀分析：把**Z→K真正修復／新增／仍持續的前排失位**，對照原pool、可信anchor、既有teacher及positive lineage。以全部gold的真實全庫rank和終端top10／100競爭者為準，不再只看平均margin或少數初始rival。逐positive的DCG變化可以直接核算；pair數量不可當作nDCG的可加因果分解。

這與[NG72](ng0072-retention-diagnosis-review.zh.md)不同：NG72已做過query/doc交叉與支持集合分解，顯示約70%的負margin變化來自存續support的權重。下一步**不重跑該角色定位**，而是判斷這次干預改善了哪些與品質有關的限制、還漏了哪些。也不重放NG76失敗的GPU observer。

按新證據選擇下一個最小對照，而非同時加入多種改動：

1. 若主要損害集中在更新後才進榜、原pool未覆蓋的競爭者，才準備一次current-model witness refresh對照；不把擴大K當成一般解答。
2. 若同題pool已涵蓋、原可信關係也多數保住，但不同query仍退步，優先設計**匹配曝光量的排序監督廣度／跨query功能保留對照**。NG69已證明廣度比重複訓練有效，但其CE增密且NQ未修好；不能直接照搬該配方或自動擴大失敗K。
3. 若失位集中在既有teacher未覆蓋／反對且來源可疑的關係，優先使用NG68既有盲化人工審查材料，分清原始／added正例與student文字可見性。缺人工判定就保留unknown，不以LLM共識代替人類監督。

診斷只重用已封存TRAIN排名／codes／supervision，零新encoder或teacher推論、零更新、不碰DEV／LOCKED_TEST。其結論是選擇實驗的依據，不是新的held-out結果。任何新訓練均需先凍結比較、資料、步數、成本與逐域准入；本輪失敗的gates保持不變，不掃lambda、不自動192步、不調NQ floor，也不先更換整個基座。

## 6. 執行與可重現證據

九個phase於11:03:44 UTC完成，全部exit0／error=null／owned group closed／actual-start offline ClearML closed。phase耗時135.13／424.61／750.98／555.22／325.31／474.58／250.08／250.50／129.98秒；包括hash與tracking收尾，不是serving latency。所有自有controller／worker PID已消失。11:07資源快照GPU0／3閒置、GPU1／2同事工作未動，host available約121.20GB、disk free約755.16GB。

完整remote單元143files／585,062,277bytes，Mac傳輸exit0，沒有刪除／覆蓋remote原件。`review.passed=true`只表示執行和驗證成功；`decision.exploratory_gate_passed=false`才是科學晉級結論。新增[鏡像review入口](../../../../scripts/review_ng77_endpoints.py)對Mac完整inventory、460 dependencies／31 frozen sources、實際loss、指標／bootstrap／cost重新核对，另用coordinate-product累加核對五模型全部384,000個已存top100分數。這不是第二次全庫排序，也沒有模型inference。

| 證據 | SHA256 |
| --- | --- |
| endpoint inputs | `46fed531851ee1b9d7a49f7ef9ed32436aec1e55fe52d293e4d7759cdc1400d7` |
| remote inventory | `dd8e23dc862f57ef0e54779c4e89df170973a928525ec7c258469b08c2f19454` |
| review complete | `ab4ead031c36d890379d01a8f46bf3e4f8debde8db012675fff86cbd8aa4ae3a` |
| remote review results | `bfcb167a07d2f3933075d2a33c0a70363f0ca24c787c0e8bd5e52f20ac07bb36` |
| Mac local-review | `ffc57b0da6b74230a445cb2571976d78ebc384309c546607d237f5d775d2229e` |
| local reviewer source | `d0bc44dea06fe8e9aca91075346dcc701793e61abb9bc635514433fd795156cb` |

Mac唯讀review於11:19:50 UTC完成，exit0，36.710秒、peak RSS2,579,546,112bytes。全部143files／585,062,277bytes及460dependencies／31sources前後驗證一致。重播既定CPU reviewer的actual-loss reduction最大差2.78e-17，endpoint指標／bootstrap／cost／anchor reduction與remote精確一致；這是相同reviewer跨主機重播，不是假稱兩套獨立bootstrap實作。另一路coordinate-product驗證全部384,000個已存head分數，最大差7.11e-15，低於既有1e-12界線。沒有新推論、parameter VJP重算或全庫重排。

本地回執位於`NG-0077/train-endpoints-v1-local-review.json`，在immutable run之外，以exclusive create寫入，連同全部imported reviewer source SHA保留。原始remote與Mac訓練／終端單元都保留，不做刪除或模型晉級。

相關NG70–77／architecture／navigation／published-model測試共**550 passed，3.85秒**，包含per-domain Recall floor、非法query split、錯inventory、篡改head scores等fixtures；`git diff --check`通過。終端執行來源為`4fff32dc28138ed7d688fbe194637f796f0bc1f7`；本輪後加的Mac reviewer只做唯讀審核，沒有回寫凍結sources。沒有PostgreSQL產品改動、production部署或push。
