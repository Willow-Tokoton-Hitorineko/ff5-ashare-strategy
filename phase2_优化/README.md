# Phase 2: 汇报后优化版本

针对期末汇报暴露的问题，由组长（本人）推进，进行了系统性修正和深入分析。

## 修正内容

| 问题 | 期末版本 | 优化版本 |
|------|---------|---------|
| 个股收益 | 市值变化 pct_change | Mretwd (CSMAR真实收益) |
| INV构造 | financial_indicators (54.7%缺失, ROE填充) | FS_Combas自算 (19.3%缺失, 不填充) |
| 财报滞后 | 前视偏差 (7-12月用当年年报) | 严格6个月滞后 |
| 冲击成本 | 文献查表 (3-80bp离散值) | 参与率模型 (连续, 数据驱动) |
| 稳健性 | 无因子剔除对比 | 三场景 (纯INV/去INV/ROE填) |
| 反向策略 | 未检验 | 14/14窗口全正Alpha |

## 脚本运行顺序

```bash
# 1. 检查FS_Combas字段映射
python step1_check_fields.py

# 2. 构造因子面板 (核心步骤，输出 factors_panel.csv, ~50MB)
python step2_build_factors.py

# 3. 策略回测 + FF5归因 + 三场景对比
python step3_backtest.py

# 4. 冲击成本 + Code A对比
python step4_final.py

# 5. 反向策略检验
python step5_reverse.py

# 6. 全量检验汇总 + 滚动窗口
python step6_save_all.py
```

## 各脚本说明

### step1_check_fields.py
检查 `FS_Combas.csv` 的字段映射，确认 A001000000=资产总计（用于INV自算）。
运行后打印字段列表，验证关键列的存在。

### step2_build_factors.py
**最核心的步骤**。完成：
1. 读取所有原始CSV
2. 股票池筛选 (A股主板, 非金融, IPO≥12月, 有效收益)
3. Size = log(Msmvttl)
4. BM/OP 来自 financial_indicators.csv
5. INV = (TA_t - TA_{t-1}) / TA_{t-1} (从FS_Combas自算, 6个月财报滞后)
6. 月度横截面1%/99%缩尾
输出: `factors_panel.csv` (约50MB, 226,436条记录)

### step3_backtest.py
策略回测 + FF5归因：
- 三场景: 纯INV / 去INV(三因子) / ROE填INV
- z-score标准化 → 等权打分 → top 20%持仓
- 等权组合月收益 (Mretwd)
- FF5 OLS + Newey-West (12期滞后)
- 子区间归因
输出: `regression_compare.csv`, `subperiod_compare.csv`, `corr_raw.csv`, `corr_winsorized.csv`

### step4_final.py
冲击成本 (参与率模型) + Code A对比：
- 参与率模型: Impact(bp) = sigma * sqrt(participation) * 10000
- 按市值分组输出冲击成本
- 与Code A期末版本的全维度对比
输出: `final_comparison.csv`, `impact_cost.csv`

### step5_reverse.py
反向策略快速检验：持有得分最低20%的股票。

### step6_save_all.py
全量检验汇总：
- 六场景全量FF5归因
- 滚动窗口 (5年/1年步长)
- 子区间 (正反向)
- 冲击成本
输出: `all_scenarios_regression.csv`, `rolling_window.csv`, `rolling_window_reverse.csv`, `subperiod_all.csv`

## 结果文件

`results/` 目录下所有CSV：
- `all_scenarios_regression.csv` — 六场景归因总表（最核心）
- `rolling_window.csv` — 正向14窗口Alpha
- `rolling_window_reverse.csv` — 反向14窗口Alpha
- `subperiod_all.csv` — 子区间正反向
- `impact_cost.csv` — 参与率模型成本
- `corr_raw.csv` / `corr_winsorized.csv` — 相关性矩阵
- `final_comparison.csv` — Code A对比
