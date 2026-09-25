# NG-0087：完成結果與下一階段判斷

日期：2026-09-20。狀態：候選集訓練／評估閉合；不是全背景、泛化或產品驗收。

## 已核實的完成範圍

[凍結協議](ng0087-clean-base-teacher-ab.zh.md) 的兩臂均完成 8,192 updates、32,768 exposures。A=PPLX、B=RankT5；共同使用成熟底座、fresh AdamW 與 full-std derivative。兩臂 preflight、base profile、正式訓練、endpoint profiles、末尾 review 均通過。

Campaign 於 2026-09-19 05:23:36 UTC 結束，耗時 43,250.36 秒。9 月 19 日已核對 source hashes、base/A/B profiles 和 review 的 sealed payloads，相關程序退出；9 月 20 日重新讀取 terminal JSON 與以下 SHA。沒有延長訓練或事後挑選 2,048／4,096 checkpoints。

| 證據 | SHA256 |
|---|---|
| `inputs.json`（原凍結 manifest） | `27d5965fc24524ca6ab90c7bcf60d10b931d029aaa9f8357990e646296f7b361` |
| `campaign-exit.json` | `be36e406fb48523b80f2d0db9c1a5c7f6988420fbda67c6903028f5b6f2e6ccf` |
| `review/results.json` | `f77525c46f3fd4d52d8f94569db736fc5a2a931b7c55736315b64f83af879241` |
| A/8192 model state | `e87b675e90af54773ab9ca1e4102867a3b828700e22f1fdbc6d7e4f3dfcfacd9` |
| B/8192 model state | `125cbe68bf8b7bc4929776655b9fd2ecb886ff675465ec970521d3dee57953bb` |

產物單位是 Lambda2 的 `NG-0087/clean-base-ab-v1`。本次沒有重新執行 GPU 評估、同步大型 checkpoint 或宣稱本地已有完整 run 備份；只讀核驗與文檔收斂不能替代資料鏡像驗收。

## 主要結果

以下是全部 2,048 個已曝光 validation queries、相同候選 bank 的完整 BM25+semantic hybrid。文件 NNZ 是 query-weighted candidate proxy，不是全庫去重 posting 數或延遲。

| 模型 | nDCG@10 | Recall@10 | 文件 NNZ | Query NNZ | 文件 NNZ / base |
|---|---:|---:|---:|---:|---:|
| Base | 0.670314 | 0.769094 | 290.29 | 57.99 | 1.00 |
| A/PPLX | 0.693741 | 0.777042 | 1579.74 | 105.57 | 5.44 |
| B/RankT5 | 0.684901 | 0.774435 | 810.09 | 64.88 | 2.79 |
| PPLX dense reference | 0.751470 | 0.826430 | 不適用 | 不適用 | 不適用 |

| 配對差 | nDCG@10 差 | 描述性 95% 區間 | Recall@10 差及區間 |
|---|---:|---|---|
| A - base | +0.023427 | [+0.015650, +0.030956] | +0.007949 [-0.001717, +0.017216] |
| B - base | +0.014586 | [+0.007281, +0.021855] | +0.005341 [-0.003112, +0.013973] |
| B - A | -0.008840 | [-0.016553, -0.000803] | -0.002607 [-0.012247, +0.007260] |

所有 endpoint queries 均無 invalid。A 通過預先定義的候選品質篩選；B 的 Recall 差區間下界低於 -0.002，因此沒有通過該篩選，**不等於 B 沒有 nDCG 改善**。兩臂均超過原有 1.5x 文件 NNZ 成本篩選，不能因本輪訓練成功而取消這個失敗紀錄。

## 不能忽略的取捨

| Domain | Base nDCG | A nDCG | B nDCG |
|---|---:|---:|---:|
| FEVER | 0.786470 | 0.900653 | 0.831909 |
| FiQA | 0.360803 | 0.408628 | 0.398924 |
| HotpotQA | 0.836330 | 0.803281 | 0.836401 |
| NQ | 0.697654 | 0.662401 | 0.672368 |

- A 的整體收益由 FEVER／FiQA 推動，不能描述成所有問題都改善。未來仍以固定權重 overall hybrid 為主要目標，不恢復「每域必勝」規則，但必須披露兩個回退域。
- A 品質較高，B 較稀疏，這兩點互不支配。不能把不同成本下的 A/B 品質差當作公平的等成本 teacher 勝負，也不能宣布 RankT5 家族無效。
- 正確 std derivative 解除了本次訓練的數值障礙，但沒有自行保住稀疏性。NG87 沒有舊導數的 clean-base 對照，因果證據應連同 NG86 的受控介入解讀，不能只憑本輪前後比較獨立歸因。
- 本輪 cost regularizer 明確為 0。增密與品質改善同時發生，不足以判定新增 posting 全是浪費，或品質必須依賴全部增密。
- 2,048 queries 曾用於研究，paired bootstrap 是單 seed 的描述性證據。仍缺全固定背景排名、第二 seed、新來源驗證、原生成本及正常輸入 dense 對照。

## 下一步

保留 A/8192 為品質參考，B/8192 為較低密度參考，原成熟底座仍保留。不換產品模型、不重建索引、不重啟自動輪詢。

下一個工作設計是 [NG88 全背景驗證與預算內排序恢復](ng0088-quality-retention-cost-plan.zh.md)：先證實全背景收益，再讓實際有輸出預算的 student 學習，而不是將另一輪剪枝或 loss 權重掃描包裝成新突破。
