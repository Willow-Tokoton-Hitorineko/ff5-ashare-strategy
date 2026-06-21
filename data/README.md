# 数据说明

## 数据来源

全部原始数据来自 **CSMAR (国泰安) 数据库**。

## 所需文件清单

以下文件需从CSMAR下载后放入本目录。因文件较大（部分>300MB），未纳入Git仓库。

### Phase 1 (期末版本) 需要的文件

| 文件名 | CSMAR表名 | 关键字段 | 用途 |
|--------|----------|---------|------|
| market_cap_monthly.csv | 个股月市值 | Stkcd, Trdmnt, Msmvttl, Markettype | Size因子 |
| financial_indicators.csv | 财务指标(预计算) | Stkcd, Trdmnt, BM, OP, INV, ROE | BM/OP/INV/ROE因子 |
| turnover_monthly.csv | 个股月换手率 | Stkcd, Trdmnt, MonthlyTurnover | 流动性分组 |
| ff5_factors.csv | FF5因子(预计算) | Trdmnt, MKT, SMB, HML, RMW, CMA, Rf | FF5归因 |
| sample_universe.csv | 筛选后股票池 | Stkcd, Trdmnt | 基准universe |
| 无风险利率.xlsx | 无风险利率表 | Clsdt, Nrrmtdt | 无风险利率 |

### Phase 2 (优化版本) 额外需要的文件

| 文件名 | CSMAR表名 | 关键字段 | 用途 |
|--------|----------|---------|------|
| TRD_Mnth.csv | 月个股回报率 | Stkcd, Trdmnt, Mretwd, Msmvttl, Mnvaltrd, Markettype | 真实收益+成交额 |
| FS_Combas.csv | 合并资产负债表 | Stkcd, Accper, Typrep, A001000000 | INV自算(总资产) |
| STK_LISTEDCOINFOANL.csv | 上市信息 | Symbol, LISTINGDATE, IndustryCode | IPO过滤+金融行业剔除 |
| STK_INDUSTRYCLASS.csv | 行业分类 | Symbol, IndustryCode, ImplementDate | 行业分类 |
| STK_MKT_FIVEFACMONTH.csv | FF5因子(官方) | TradingMonth, RiskPremium2, SMB2, HML2, RMW2, CMA2 | 官方FF5因子(总市值加权) |
| TRD_Nrrate.csv | 无风险利率 | Clsdt, Nrrmtdt | 无风险利率(月度化) |

## CSMAR下载指引

1. TRD_Mnth: 股票市场交易 → 月个股回报率 → 选择所有A股, 2000-2025
2. FS_Combas: 财务报表 → 合并资产负债表 → 选择所有字段, 2000-2025
3. STK_LISTEDCOINFOANL: 公司信息 → 上市信息 → 所有A股
4. STK_INDUSTRYCLASS: 公司信息 → 行业分类 → 证监会2012版
5. STK_MKT_FIVEFACMONTH: 因子研究 → Fama-French五因子 → 月度, 总市值加权
6. TRD_Nrrate: 市场利率 → 无风险利率 → 日度, 月度化后使用

## 数据预处理说明

- CSMAR导出的CSV通常包含3行英文列名（列名/中文描述/单位），`read_csmar()` 函数自动处理
- encoding: utf-8-sig 或 gbk
- Stkcd: 字符串，前补零至6位
- Trdmnt: YYYY-MM 格式
