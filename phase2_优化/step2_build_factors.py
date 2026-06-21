"""
Step 2: 股票筛选 + 因子构造
============================================================
方法论说明
------------------------------------------------------------
【Size】log(总市值 Msmvttl)，来自 market_cap_monthly.csv
【BM】账面市值比，直接使用 financial_indicators.csv（数据组预计算）
      来源: 股东权益 / 总市值，已在 financial_indicators 中按财报滞后规则对齐
【OP】盈利能力，直接使用 financial_indicators.csv（数据组预计算）
      来源: 营业利润 / 股东权益
【INV】总资产同比增长率，从 FS_Combas.csv 底层自算：
      1. 取 CSMAR 合并资产负债表 (Typrep='A') 的 A001000000 (资产总计)
      2. 只保留 12-31 年报数据
      3. INV(t) = (TA_t - TA_{t-1}) / TA_{t-1}，t 为年报年度
      4. 财报滞后规则 (Fama-French 标准 6 个月滞后):
         - 7月至次年6月的月度调仓 → 使用上一年年报
         - 即: month∈[7,12] → report_year = year-1
                month∈[1,6]  → report_year = year-2
         - 例: 2020年7月调仓, 使用 2019年年报 (2019年12月截止, 2020年4月底前公布)
      5. 缺失不填充 (不同于 Code A 用 ROE 替代的做法)
【缩尾】月度横截面 1%/99% Winsorize，每个因子保留 _raw 列用于缩尾前后对比
【筛选】A股主板 (Markettype 1,4), 非金融 (IndustryCode 非 J 开头),
        上市 ≥ 12 个月, 月收益在 [-95%, +500%] 区间内
============================================================
输出: factors_panel.csv
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUT  = os.path.dirname(os.path.abspath(__file__))

def read_csmar(path, **kw):
    """读取 CSMAR CSV (含3行header: 英文列名/中文描述/单位)"""
    for enc in ['utf-8-sig', 'gbk', 'utf-8']:
        try:
            return pd.read_csv(path, encoding=enc, skiprows=[1,2], low_memory=False, **kw)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"decode fail: {path}")

print("=" * 60)
print("Step 2: 股票筛选 + 因子构造")
print("=" * 60)

# ─── 加载 ───
print("\n[2.1] 加载...")
trd = read_csmar(f"{DATA}/TRD_Mnth.csv", dtype={'Stkcd': str})
trd['Stkcd'] = trd['Stkcd'].str.zfill(6)
for c in ['Mretwd','Msmvttl','Mnvaltrd','Markettype']:
    trd[c] = pd.to_numeric(trd[c], errors='coerce')
print(f"  TRD_Mnth: {len(trd):,}")

fs = read_csmar(f"{DATA}/FS_Combas.csv", usecols=['Stkcd','Accper','Typrep','A001000000'],
                dtype={'Stkcd': str})
fs['Stkcd'] = fs['Stkcd'].str.zfill(6)
fs = fs[fs['Typrep']=='A'].drop(columns='Typrep')
fs['A001000000'] = pd.to_numeric(fs['A001000000'], errors='coerce')
fs['Accper'] = pd.to_datetime(fs['Accper'])
print(f"  FS_Combas(合并): {len(fs):,}")

fi = pd.read_csv(f"{DATA}/financial_indicators.csv", dtype={'Stkcd': str})
fi['Stkcd'] = fi['Stkcd'].str.zfill(6)
for c in ['BM','OP','ROE']:
    fi[c] = pd.to_numeric(fi[c], errors='coerce')
print(f"  financial_indicators: {len(fi):,}")

li = read_csmar(f"{DATA}/STK_LISTEDCOINFOANL.csv", dtype={'Symbol': str})
li = li.rename(columns={'Symbol':'Stkcd'})
li['Stkcd'] = li['Stkcd'].str.zfill(6)
li['LISTINGDATE'] = pd.to_datetime(li['LISTINGDATE'], errors='coerce')
# 每只股票只取一条上市信息（最小LISTINGDATE）
li = li.dropna(subset=['LISTINGDATE']).sort_values('LISTINGDATE').groupby('Stkcd').first().reset_index()
print(f"  listing: {len(li):,}")

ic = read_csmar(f"{DATA}/STK_INDUSTRYCLASS.csv", dtype={'Symbol': str})
ic = ic.rename(columns={'Symbol':'Stkcd'})
ic['Stkcd'] = ic['Stkcd'].str.zfill(6)
ic_latest = ic.sort_values('ImplementDate').groupby('Stkcd').last().reset_index()
print(f"  industry(raw): {len(ic_latest):,}")

ic_latest['IndustryCode'] = ic_latest['IndustryCode'].fillna('').astype(str)
ic_latest['fin'] = ic_latest['IndustryCode'].str.startswith('J')
# 一只股票可能换过行业，取最新的（last）
ic_latest = ic_latest[['Stkcd','IndustryCode','fin']].drop_duplicates(subset='Stkcd', keep='last')
print(f"  industry(dedup): {len(ic_latest):,}")

# ─── 筛选 ───
print("\n[2.2] 筛选...")
df = trd[['Stkcd','Trdmnt','Mretwd','Msmvttl','Mnvaltrd','Markettype']].copy()
df = df.merge(ic_latest[['Stkcd','fin']], on='Stkcd', how='left')
df = df.merge(li[['Stkcd','LISTINGDATE']], on='Stkcd', how='left')
df['month_dt'] = pd.to_datetime(df['Trdmnt'].astype(str)+'-01', errors='coerce')
df['months_since_ipo'] = (df['month_dt'] - df['LISTINGDATE']).dt.days / 30.44

before = len(df)
m = (
    df['Markettype'].isin([1, 4]) &
    (~df['fin'].fillna(False)) &
    (df['months_since_ipo'] >= 12) &
    df['Mretwd'].notna() & df['Mretwd'].between(-0.95, 5.0)
)
df = df[m].copy()
print(f"  筛选后: {len(df):,} 条 (移除 {before - len(df):,}), "
      f"{df['Stkcd'].nunique()} 只, 月均 {df.groupby('Trdmnt').size().mean():.0f} 只")

# ─── 合并 financial_indicators ───
df = df.merge(fi[['Stkcd','Trdmnt','BM','OP','ROE']], on=['Stkcd','Trdmnt'], how='left')

# Size
df['Size'] = np.log(df['Msmvttl'].clip(lower=1))

# ─── INV: 总资产同比增长率（向量化）───
print("\n[2.3] 计算 INV (向量化)...")
# 年报数据：fs_annual[stk, year] → total assets
fs['year'] = fs['Accper'].dt.year
fs_a = fs[fs['Accper'].dt.month == 12][['Stkcd','year','A001000000']].drop_duplicates(
    subset=['Stkcd','year'], keep='last')

# t-1 年数据
fs_a_prev = fs_a.copy()
fs_a_prev['year'] = fs_a_prev['year'] + 1
fs_a_prev = fs_a_prev.rename(columns={'A001000000': 'TA_prev'})

fs_a = fs_a.merge(fs_a_prev, on=['Stkcd','year'], how='left')
fs_a['INV'] = (fs_a['A001000000'] - fs_a['TA_prev']) / fs_a['TA_prev']
fs_a.loc[fs_a['TA_prev'].isna() | (fs_a['TA_prev'] <= 0), 'INV'] = np.nan

# 映射到月度: 标准6个月财报滞后
# 7月~次年6月 → 使用上一年年报 (年报最晚4月底发布, 6月底前可获取)
# 即: month∈[7,12]→year-1, month∈[1,6]→year-2
df['report_year'] = df['month_dt'].dt.year - 2 + (df['month_dt'].dt.month >= 7).astype(int)

df = df.merge(fs_a[['Stkcd','year','INV']],
              left_on=['Stkcd','report_year'], right_on=['Stkcd','year'], how='left')
df = df.drop(columns=['year','report_year'])

inv_miss = df['INV'].isna().mean()
monthly_miss = df.groupby('Trdmnt')['INV'].apply(lambda x: x.isna().mean())
inv_roe_corr = df[['INV','ROE']].dropna().corr().iloc[0,1]

print(f"  INV 缺失率: {inv_miss:.1%}")
print(f"  月度缺失率: 均值 {monthly_miss.mean():.1%}, min {monthly_miss.min():.1%}, max {monthly_miss.max():.1%}")
print(f"  INV-ROE 相关性: {inv_roe_corr:.4f}  "
      f"({'完全不同质→不应混用' if abs(inv_roe_corr) < 0.3 else '弱相关→不应混用' if abs(inv_roe_corr) < 0.5 else '有相关但不等价'})")

# ─── 缩尾 ───
print("\n[2.4] 1%/99% 月度缩尾...")
factor_cols = ['Size','BM','OP','INV','ROE']
for c in factor_cols:
    df[c + '_raw'] = df[c]
    qlo = df.groupby('Trdmnt')[c].transform(lambda x: x.quantile(0.01))
    qhi = df.groupby('Trdmnt')[c].transform(lambda x: x.quantile(0.99))
    df[c] = df[c].clip(lower=qlo, upper=qhi)

# 清理
df = df.dropna(subset=['Size','BM','OP']).copy()
print(f"  最终: {len(df):,} 条, {df['Stkcd'].nunique()} 只, {df['Trdmnt'].nunique()} 个月")

# ─── 保存 ───
cols_save = ['Stkcd','Trdmnt','Mretwd','Msmvttl','Mnvaltrd','Size','BM','OP','INV','ROE',
             'Size_raw','BM_raw','OP_raw','INV_raw','ROE_raw']
out_path = os.path.join(OUT, 'factors_panel.csv')
df[cols_save].to_csv(out_path, index=False)
print(f"\n[DONE] {out_path}")
print(f"  Size: {os.path.getsize(out_path)/1024/1024:.1f} MB")
print("=" * 60)
print("Complete.")
