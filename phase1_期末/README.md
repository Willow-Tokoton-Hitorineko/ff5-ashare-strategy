# Phase 1: 期末版本

技术组组员分工交付的原始版本，包含期末汇报的文本解析内容。

## 文件说明

### codeA/ — 多因子打分策略 + FF5归因分析
- **负责人**: 技术产出组
- **功能**: 四因子(Size/BM/OP/INV)等权打分 → 前20%持仓 → FF5归因(OLS+NW) → 滚动窗口 → 子区间
- **已知问题**:
  1. 个股收益用市值变化(pct_change)近似，非Mretwd
  2. INV缺失54.7%时以ROE替代
  3. 财报滞后存在前视偏差

### codeB/ — 交易可行性分析
- **负责人**: 技术产出组
- **功能**: 市值/流动性分组分析、冲击成本估算(文献查表)、多场景约束测试
- **输出**: size_group_results.csv, liquidity_results.csv, impact_cost.csv, tradeability_summary.csv

### PPT_content.md
- 期末汇报PPT的文字解析版

## 运行

Code A需要以下输入文件（原始代码中硬编码路径，运行前需修改）：
- market_cap_monthly.csv
- financial_indicators.csv
- turnover_monthly.csv
- ff5_factors.csv
- sample_universe.csv
- 无风险利率.xlsx

运行前需修改代码中的文件路径。

## 注意

此版本结果与优化版存在定量差异，但主结论一致。
优化版详见 `../phase2_优化/`。
