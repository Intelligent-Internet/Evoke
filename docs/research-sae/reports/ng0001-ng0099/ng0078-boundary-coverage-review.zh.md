# NG-0078：不是只缺更多負例，而是局部保留沒有轉成全局排序保留

2026-09-13 UTC。**預先凍結的唯讀診斷已完成，lambda2執行、獨立scalar核對、Mac完整鏡像均通過。沒有新模型訓練或品質晉級。** [協議](ng0078-boundary-coverage-plan.zh.md)固定全部768個已曝光TRAIN query、1,371個正例、233,009文件背景。使用NG77 initial／Z96／K96已封存的完整hybrid終端結果；不是SAE單路或新held-out試驗。

## 1. 原217→201究竟改了哪些關係

將45,175個原可信anchors的終端 `margin <= 0` 集合逐一交差，而不是只比較總數：

| TRAIN Pilot | Z／K都非正 | Z非正、K恢復正值 | Z正、K新變非正 | Z總非正 | K總非正 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FEVER | 8 | 0 | 0 | 8 | 8 |
| HotpotQA | 81 | 9 | 1 | 90 | 82 |
| NQ | 111 | 8 | 0 | 119 | 111 |
| 合計 | 200 | 17 | 1 | 217 | 201 |

因此是**17個修復、1個新增、200個持續**，淨少16。這些是strict-margin關係，不等於17個文件或query被修好。先前28個「原anchor rival在top10而gold在外」兩臂仍相同；本輪沒有把這個舊結果改成成功。

## 2. NQ主要的已知前排損害，不在pool外

以下只針對實際參訓的128題NQ Pilot，且只列K相對initial DCG contribution下降的gold。不是拿Sentinel天生未參訓當成coverage發現。

這些gold有40個「初始gold在前、K時nongold反超且進top10」pair：

| 實際參訓NQ受損gold上的前排反超 | pair數 | 比例 |
| --- | ---: | ---: |
| 原pool內、原可信anchor、teacher同意 | 34 | 85.0% |
| 原pool內、teacher相反而非anchor | 5 | 12.5% |
| 原pool外、teacher未觀察 | 1 | 2.5% |

以每個gold的真實全庫rank直接計算gross positive DCG loss，再按該gold的完整失位集合做互斥分組：

- **78.53%**的gross loss落在「所有前排反超rivals都已是可信anchors」的gold，18個gold，固定128題分母的損失量0.01988783。
- 只有**1.65%**落在「至少一個前排反超rival在pool外」的gold，1個gold，損失量0.00041812。
- 總gross loss為0.02532483，並不等於net query nDCG變化。其他gold改善後，整體NQ Pilot仍淨升0.07009960；gold之間換位也可能產生一負一正contribution。

因此，對這個受檢查的TRAIN前排失敗群，**「只要再擴大candidate pool／多挖負例就能解決」沒有得到支持**。Hard-negative coverage依然可能影響別的query／cutoff；不能將此局部結果外推為全局無效。尤其top100的pool外pair很多，但不少沒有令任何gold跨出100，不能用大量pair數冒充Recall損失。

K的Pilot top10 boundary-loss集合比Z消失11個pair，沒有新增；NQ只消失3個，67個持續。Sentinel則消失13個、新增4個。這裡「消失」也可能是rival離開cutoff，故不能等同strict margin恢復。兩種定義已分開核算。

## 3. 跨query退步也不是只發生在added正例

| K相對initial，TRAIN Sentinel | 原始gold受損個數 | added gold受損個數 | 原始gold占gross DCG loss |
| --- | ---: | ---: | ---: |
| HotpotQA | 32 | 1 | 97.16% |
| NQ | 27 | 22 | 80.66% |

NQ原始／added gold的gross損失量分別為0.04641204／0.01113170。原始gold只有128／252個，卻承擔大部分這種前排損失；不能把失敗簡化成「評測中added正例本身較差，所以刪掉就會好」。這**不排除added訓練標籤可能干擾其他gold**，也沒有驗證每個原始label都正確。刪標籤或重權重仍需要另外的對照／人工判斷。

受損NQ Sentinel gold上的100個持續top10反超pair，原teacher未觀察71個、同意gold在前21個、反對8個。未知佔多數，但不能自動當作hard negatives，更不能用teacher缺失直接證明label錯誤。HotpotQA對應為64／15／2。原始teacher和step0 teacher分數、正例lineage、既有token visibility都保留在raw witnesses；沒有span judgments的部分依然unknown。

## 4. 下一個受控方向

把這些結果與[NG69廣度](ng0069-final-breadth-review.zh.md)、[NG72角色／權重](ng0072-retention-diagnosis-review.zh.md)、[NG73真實loss方向](ng0073-gradient-replay-review.zh.md)及[NG74均勻權重](ng0074-uniform-control-review.zh.md)合在一起：

1. 已知pair不一定缺失，自己的loss多半也不是要求它反向，但共享parameter更新後仍可能壞掉。**大量easy anchors的平均表現，不等於少數前排關係的保證，更不等於對其他query的函數保留。** 本輪只定位這個缺口，沒有完成parameter層的唯一因果歸因。
2. 下一個優先受控分支選擇**在同一成熟基座、同一ranking objective下，匹配總曝光量的監督query廣度**，而不是先擴candidate K、再掃retention lambda。兩臂均由同一NG3 initialization出發，使用原D ranking objective，不延續失敗K checkpoint，也不沿用NG69的CE objective。NG69已提供廣度有效的三seed證據，但其CE增密與NQ問題尚存，下一次不能把舊CE配方直接放大。先核對可用TRAIN provenance和初始witness，凍結「更多不同query」對「較少query重複」的公平比較、實際token／文件曝光及domain組成；不是立即加長失敗K。
3. 本輪已曝光Sentinel不加入訓練，也不重新命名為獨立holdout。NQ及全部gold保留，逐域nDCG與all-positive Recall floor不放寬。若廣度仍不能保住可信前排少數關係，再單獨測試跨query函數保留／尾部風險約束；不在同一輪同時改pool、loss、基座和監督來源。

這是下一個protocol的優先次序，**NG78沒有啟動新training，也沒有讓NG77失敗的門檻變成通過**。最終目標不變：完整BM25＋semantic超越competitive dense，並且actual total native cost更低。沒有新的native成本結果或產品部署。

## 5. 驗證與可重現性

執行來源commit `92e49ef7687ff7b89bc04f18aae4b76a1c7b975a`，本地相關測試573 passed／2.95秒；lambda2凍結fixtures23 passed／2.90秒。`git diff --check`通過。CPU phase使用4threads，沒有GPU、模型inference、training、DEV／LOCKED_TEST評分。

phase於12:06:05 UTC閉合，57.394秒，peak process-tree RSS2,609,856,512bytes，exit0／error=null／owned group closed；controller於12:06:20結束，兩個自有PID均已消失。16題canary的保守全程主reduction外推8.77秒，原1,800秒硬上限未變。ClearML `offline-14a81c369a184c8fbe782174e1535b4b` actual-start／closed／passed，offline證據已保存，**未宣稱已同步遠端ClearML服務**。

兩路cached分數最大差7.11e-15，低於1e-12；完整三模型top100與全部gold ranks／逐域NG77品質差值核對通過。獨立stdlib reviewer不import主reducer，逐scalar重新產生全部1,371個positive的分類、pair交差、DCG／Recall contribution並核對；它重用相同raw witnesses，不是另一套模型或全庫ranking。

外部單元`NG-0078/boundary-coverage-v1`完整44files／14,222,592bytes，no-delete／no-overwrite傳輸exit0。Mac核對完整inventory、606 dependencies、26 frozen sources，重播獨立scalar reviewer並與remote結果一致；6.732秒、peak RSS39,796,736bytes。沒有回寫sealed run；本地回執位於其同層`boundary-coverage-v1-local-review.json`。

| 證據 | SHA256 |
| --- | --- |
| frozen inputs | `adf0862176bb81e40f7bbd994c3ae4d17771c1d4ede86f51dc9f8351fd1c6778` |
| remote complete inventory | `a6e35948e372529219c306a7c28f5bd929c2f58e9efbb40145585d2846a9d42c` |
| Mac local review | `79b9051174932d456d92faf123dfd313df083990993d0019d2f55dc98980f665` |

本輪不是另一個單純看margin平均的重複audit：它核實了17修復／1新增，並將真正的gold前排損失與pool、可信關係和來源相連。但覆蓋類別、pair計數和gross損失分割均為描述性證據，不能報作「某機制造成78.53%總誤差」或新的泛化成果。
