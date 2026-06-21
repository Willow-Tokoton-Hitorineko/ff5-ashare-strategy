# FF5多因子A股策略研究

基于Fama-French五因子框架的A股多因子选股策略实证研究，课程期末项目。

## 项目结构

```
ff5-ashare-strategy/
├── phase1_期末/               # 期末版本（原始组员交付）
│   ├── codeA/                  # 多因子打分+FF5归因
│   ├── codeB/                  # 交易可行性分析
│   └── PPT_content.md          # 期末汇报PPT解析版
├── phase2_优化/               # 汇报后优化版本
│   ├── step1~6_*.py            # 逐步分析脚本
│   └── results/                # 输出结果CSV
├── data/                       # 数据说明
├── reports/                    # 期末报告+进度报告
└── requirements.txt            # Python依赖
```

## 两个版本的关系

| 维度 | 期末版本 (Phase 1) | 优化版本 (Phase 2) |
|------|------------------|-------------------|
| 个股收益 | 市值变化 pct_change | Mretwd 真实收益 |
| INV因子 | 54.7%缺失用ROE替代 | 纯总资产增速, 19.3%缺失不填充 |
| 财报滞后 | 含前视偏差 | 修正为严格6个月滞后 |
| 冲击成本 | 文献查表(3-80bp) | 参与率模型(数据驱动) |
| 稳健性检验 | 未做 | 三场景对比(纯INV/去INV/ROE填) |
| 反向策略 | 未检验 | 14/14窗口全部正Alpha |

**结论一致**：四个因子在A股为反向有效，得分最低20%年化收益35-41%，Alpha +22-26%。

## 快速复现

### 1. 环境准备
```bash
pip install -r requirements.txt
```

### 2. 数据准备
从CSMAR数据库下载以下文件，放入 `data/` 目录（详见 `data/README.md`）：
- TRD_Mnth.csv
- FS_Combas.csv (含A001000000=资产总计字段)
- financial_indicators.csv
- STK_MKT_FIVEFACMONTH.csv
- STK_LISTEDCOINFOANL.csv
- STK_INDUSTRYCLASS.csv
- TRD_Nrrate.csv
- market_cap_monthly.csv
- turnover_monthly.csv
- ff5_factors.csv
- sample_universe.csv

### 3. 运行优化版分析
```bash
cd phase2_优化

# 步骤1: 检查FS_Combas字段
python step1_check_fields.py

# 步骤2: 构造因子面板 (核心步骤, 约2-3分钟)
python step2_build_factors.py

# 步骤3: 策略回测 + FF5归因 + 三场景对比
python step3_backtest.py

# 步骤4: 冲击成本 + Code A对比
python step4_final.py

# 步骤5: 反向策略检验
python step5_reverse.py

# 步骤6: 全量检验汇总
python step6_save_all.py
```

### 4. 结果文件说明

`phase2_优化/results/` 目录：

| 文件 | 内容 |
|------|------|
| all_scenarios_regression.csv | 六场景全量FF5归因 |
| regression_compare.csv | 三场景归因对比 |
| rolling_window.csv | 正向策略14个滚动窗口Alpha |
| rolling_window_reverse.csv | 反向策略14个滚动窗口Alpha |
| subperiod_all.csv | 4段子区间×正反向Alpha |
| impact_cost.csv | 参与率模型冲击成本 |
| corr_raw.csv / corr_winsorized.csv | 缩尾前后因子相关性 |
| final_comparison.csv | Code A vs 优化版对比 |

## 主要发现

1. 正向策略年化Alpha -17%~-22%，14/14滚动窗口全负
2. 反向策略年化Alpha +22~+26%，14/14滚动窗口全正，夏普1.1-1.3
3. Size/BM/OP/INV四因子在A股定价方向与美股相反
4. INV与ROE相关性仅0.031，不可混用
5. Size-BM相关性0.04，是A股真实现象（非数据错误）

## 技术栈

- Python 3.9+
- pandas, numpy, scipy, statsmodels, matplotlib, openpyxl
- Newey-West HAC (12期滞后)
- 参与率冲击成本模型 (Almgren 2005框架)
