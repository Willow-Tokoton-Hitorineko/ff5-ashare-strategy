"""
Step 3: 策略回测 + FF5 归因 + 三场景对比 (4-factor-INV_self / 3-factor / 4-factor-INV_use模拟)
"""
import pandas as pd, numpy as np, os, warnings
from scipy import stats
import statsmodels.api as sm
warnings.filterwarnings('ignore')

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUT  = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("Step 3: 策略回测 + 归因")
print("=" * 60)

# ─── 3.1 加载 ───
print("\n[3.1] 加载因子面板...")
df = pd.read_csv(os.path.join(OUT, 'factors_panel.csv'), dtype={'Stkcd': str})
print(f"  {len(df):,} 条, {df['Stkcd'].nunique()} 只, {df['Trdmnt'].nunique()} 个月")

ff5 = pd.read_csv(f"{DATA}/ff5_factors.csv")
print(f"  FF5: {len(ff5)} 个月")

# ─── 3.2 相关性诊断：缩尾前 vs 缩尾后 ───
print("\n[3.2] 因子相关性: 缩尾前 vs 缩尾后")
names = {'Size':'Size','BM':'BM','OP':'OP','INV':'INV'}
print("\n--- 缩尾后 (winsorized) ---")
corr_w = df[['Size','BM','OP','INV']].corr().round(4)
print(corr_w.to_string())
print(f"  max|corr| = {corr_w.values[np.triu_indices(4,1)].max():.4f}")

print("\n--- 缩尾前 (raw) ---")
raw_cols = [c+'_raw' for c in ['Size','BM','OP','INV']]
raw_df = df[raw_cols].dropna()
raw_df.columns = ['Size','BM','OP','INV']
corr_r = raw_df.corr().round(4)
print(corr_r.to_string())
print(f"  max|corr| = {corr_r.values[np.triu_indices(4,1)].max():.4f}")

print("\n--- 差异 (raw - winsorized) ---")
diff = (corr_r - corr_w).abs()
print(diff[['Size','BM','OP','INV']].round(4).to_string())
print(f"  最大差异: {diff.values[np.triu_indices(4,1)].max():.4f}")

# ─── 3.3 策略回测（辅助函数）───
def run_strategy(df, factor_list):
    """给定因子列表，跑 z-score→等权打分→top20%→等权组合月收益"""
    d = df.dropna(subset=factor_list).copy()
    # z-score (Size 取负)
    for f in factor_list:
        mean = d.groupby('Trdmnt')[f].transform('mean')
        std  = d.groupby('Trdmnt')[f].transform('std').replace(0, np.nan)
        val  = (d[f] - mean) / std
        if f == 'Size':
            val = -val
        d[f'{f}_z'] = val

    z_cols = [f'{f}_z' for f in factor_list]
    d['score'] = d[z_cols].mean(axis=1, skipna=True)
    d['rank']  = d.groupby('Trdmnt')['score'].rank(pct=True)
    d['in_port'] = (d['rank'] >= 0.80).astype(int)

    port = d[d['in_port']==1].groupby('Trdmnt')['Mretwd'].mean().reset_index()
    port.columns = ['Trdmnt', 'port_ret']
    return port, d.groupby('Trdmnt')['in_port'].sum()

# ─── 3.4 三场景回测 ───
print("\n[3.4] 三场景回测...")

scenarios = {
    'A_4factor_INVpure': ['Size','BM','OP','INV'],       # 纯 INV (缺失直接丢)
    'B_3factor_noINV':   ['Size','BM','OP'],               # 三因子
    'C_4factor_INVuse':  ['Size','BM','OP'],               # 暂时和 B 一样, 后面单独做 INV_use
}
# 注: C 场景 (INV_use=INV填ROE) 需要单独构造因子值，下面单独处理

results = {}
for name, factors in scenarios.items():
    port_ret, holdings = run_strategy(df, factors)
    results[name] = port_ret
    print(f"  {name}: {len(port_ret)} 月, 月均持仓 {holdings.mean():.0f} 只, "
          f"月均收益 {port_ret['port_ret'].mean():.4f}")

# C 场景: INV用ROE填充
print("  构造 C 场景 (INV_use = INV fillna ROE)...")
df['INV_use'] = df['INV'].fillna(df['ROE'])
# 缩尾 INV_use
for c in ['INV_use']:
    qlo = df.groupby('Trdmnt')[c].transform(lambda x: x.quantile(0.01))
    qhi = df.groupby('Trdmnt')[c].transform(lambda x: x.quantile(0.99))
    df[c] = df[c].clip(lower=qlo, upper=qhi)
port_c, h_c = run_strategy(df, ['Size','BM','OP','INV_use'])
results['C_4factor_INVuse'] = port_c
print(f"  C_4factor_INVuse: {len(port_c)} 月, 月均持仓 {h_c.mean():.0f} 只, "
      f"月均收益 {port_c['port_ret'].mean():.4f}")

# ─── 3.5 合并 FF5 因子 → 超额收益 ───
print("\n[3.5] 合并 FF5...")
scenario_returns = {}
for name, port_ret in results.items():
    m = port_ret.merge(ff5, on='Trdmnt', how='inner')
    m['excess_ret'] = m['port_ret'] - m['Rf']
    scenario_returns[name] = m.dropna(subset=['excess_ret','MKT','SMB','HML','RMW','CMA'])
    print(f"  {name}: {len(scenario_returns[name])} 月可用于回归")

# ─── 3.6 FF5 OLS + Newey-West ───
print("\n[3.6] FF5 归因 (NW 12 期滞后)...")
from statsmodels.stats.sandwich_covariance import cov_hac

reg_results = []
for name, data in scenario_returns.items():
    y = data['excess_ret']
    X = sm.add_constant(data[['MKT','SMB','HML','RMW','CMA']])
    model = sm.OLS(y, X).fit()
    nw_cov = cov_hac(model, nlags=12)
    nw_se  = np.sqrt(np.diag(nw_cov))
    nw_t   = model.params.values / nw_se
    nw_p   = 2 * (1 - stats.t.cdf(np.abs(nw_t), df=model.df_resid))

    reg_results.append({
        '场景': name,
        'N个月': len(data),
        'Alpha(月)': model.params['const'],
        'Alpha(NW t)': nw_t[0],
        'Alpha_显著': '***' if nw_p[0]<0.01 else ('**' if nw_p[0]<0.05 else ('*' if nw_p[0]<0.1 else '')),
        'Alpha(年化%)': model.params['const']*12*100,
        'Beta_MKT': model.params['MKT'],
        'Beta_SMB': model.params['SMB'],
        'Beta_HML': model.params['HML'],
        'Beta_RMW': model.params['RMW'],
        'Beta_CMA': model.params['CMA'],
        'Adj_R2': model.rsquared_adj,
        '夏普(年化)': data['excess_ret'].mean()/data['excess_ret'].std()*np.sqrt(12),
        '月均收益': data['port_ret'].mean(),
        '月均超额': data['excess_ret'].mean(),
    })

reg_df = pd.DataFrame(reg_results)
print("\n" + reg_df.to_string(index=False))

# ─── 3.7 子区间归因 ───
print("\n[3.7] 子区间归因...")
sub_periods = [('2005-01','2009-12','05-09'), ('2010-01','2014-12','10-14'),
               ('2015-01','2019-12','15-19'), ('2020-01','2025-12','20-25')]

sub_rows = []
for name, data in scenario_returns.items():
    if name not in ['A_4factor_INVpure', 'B_3factor_noINV']:
        continue
    for s, e, lbl in sub_periods:
        chunk = data[(data['Trdmnt']>=s) & (data['Trdmnt']<=e)]
        if len(chunk) < 12: continue
        y = chunk['excess_ret']
        X = sm.add_constant(chunk[['MKT','SMB','HML','RMW','CMA']])
        m_s = sm.OLS(y, X).fit()
        nw_s = cov_hac(m_s, nlags=min(12, len(chunk)//4))
        se_s = np.sqrt(np.diag(nw_s))
        t_s  = m_s.params['const'] / se_s[0]
        sub_rows.append({'场景':name, '区间':lbl, 'N月':len(chunk),
                         'Alpha(月)':m_s.params['const'], 'Alpha(t)':t_s,
                         'Adj_R2':m_s.rsquared_adj,
                         'Alpha(年化%)':m_s.params['const']*12*100})

sub_df = pd.DataFrame(sub_rows)
print(sub_df.to_string(index=False))

# ─── 保存 CSV ───
reg_df.to_csv(os.path.join(OUT, 'regression_compare.csv'), index=False)
sub_df.to_csv(os.path.join(OUT, 'subperiod_compare.csv'), index=False)
corr_w.to_csv(os.path.join(OUT, 'corr_winsorized.csv'))
corr_r.to_csv(os.path.join(OUT, 'corr_raw.csv'))

print(f"\n[DONE] CSV files saved to {OUT}")
print("=" * 60)
