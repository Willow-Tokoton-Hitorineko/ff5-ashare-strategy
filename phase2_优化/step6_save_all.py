"""Step 6: 全量检验 + 反向策略 + 滚动窗口 + 统一保存"""
import pandas as pd, numpy as np, os, warnings
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
warnings.filterwarnings('ignore')

OUT = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT, 'factors_panel.csv'), dtype={'Stkcd': str})
ff5 = pd.read_csv(os.path.join(os.path.dirname(OUT), 'data', 'ff5_factors.csv'))

def backtest(df, factor_cols, top=True):
    d = df.dropna(subset=factor_cols).copy()
    for f in factor_cols:
        mean = d.groupby('Trdmnt')[f].transform('mean')
        std  = d.groupby('Trdmnt')[f].transform('std').replace(0, np.nan)
        val  = (d[f] - mean) / std
        if f == 'Size': val = -val
        d[f'{f}_z'] = val
    d['score'] = d[[f'{f}_z' for f in factor_cols]].mean(axis=1, skipna=True)
    d['rank']  = d.groupby('Trdmnt')['score'].rank(pct=True)
    d['in_port'] = (d['rank'] >= 0.80).astype(int) if top else (d['rank'] < 0.20).astype(int)
    port = d[d['in_port']==1].groupby('Trdmnt')['Mretwd'].mean().reset_index()
    port.columns = ['Trdmnt', 'port_ret']
    return port, d.groupby('Trdmnt')['in_port'].sum()

def ff5_reg(chunk, nlags=12):
    y = chunk['excess_ret']
    X = sm.add_constant(chunk[['MKT','SMB','HML','RMW','CMA']])
    m = sm.OLS(y, X).fit()
    nw = cov_hac(m, nlags=nlags)
    se = np.sqrt(np.diag(nw))
    t  = m.params.values / se
    p  = 2 * (1 - stats.t.cdf(np.abs(t), df=m.df_resid))
    return m, t, p

print("=" * 60)
print("Step 6: 全量检验汇总")
print("=" * 60)

# -------- 1. 六场景全量回归 --------
print("\n[1] 六场景 FF5 归因...")
configs = [
    ('正向_top20%_纯INV', ['Size','BM','OP','INV'], True),
    ('反向_bottom20%_纯INV', ['Size','BM','OP','INV'], False),
    ('正向_top20%_去INV', ['Size','BM','OP'], True),
    ('反向_bottom20%_去INV', ['Size','BM','OP'], False),
    ('正向_top20%_ROE填INV', ['Size','BM','OP'], True),  # will use INV_use
    ('反向_bottom20%_ROE填INV', ['Size','BM','OP'], False),
]

# 构造 INV_use
df['INV_use'] = df['INV'].fillna(df['ROE'])
for c in ['INV_use']:
    qlo = df.groupby('Trdmnt')[c].transform(lambda x: x.quantile(0.01))
    qhi = df.groupby('Trdmnt')[c].transform(lambda x: x.quantile(0.99))
    df[c] = df[c].clip(lower=qlo, upper=qhi)

all_reg = []
for name, factors, top in configs:
    use_factors = ['Size','BM','OP','INV_use'] if 'ROE填INV' in name else factors
    port, hld = backtest(df, use_factors, top=top)
    m = port.merge(ff5, on='Trdmnt', how='inner')
    m['excess_ret'] = m['port_ret'] - m['Rf']
    m = m.dropna(subset=['excess_ret','MKT','SMB','HML','RMW','CMA'])

    model, nw_t, nw_p = ff5_reg(m)
    sig = '***' if nw_p[0]<0.01 else ('**' if nw_p[0]<0.05 else ('*' if nw_p[0]<0.1 else ''))
    ann_sharpe = (m['excess_ret'].mean()/m['excess_ret'].std())*np.sqrt(12) if m['excess_ret'].std()>0 else 0

    all_reg.append({
        '场景': name, 'N月': len(m), '月均持仓': hld.mean(),
        '月均收益': m['port_ret'].mean(), '年化收益%': m['port_ret'].mean()*12*100,
        'Alpha(月)': model.params['const'], 'Alpha(NW_t)': nw_t[0],
        'Alpha显著': sig, '年化Alpha%': model.params['const']*12*100,
        'MKT': model.params['MKT'], 'SMB': model.params['SMB'],
        'HML': model.params['HML'], 'RMW': model.params['RMW'],
        'CMA': model.params['CMA'], 'AdjR2': model.rsquared_adj,
        '夏普(年化)': ann_sharpe
    })
    print(f"  {name}: Alpha={model.params['const']*12*100:.1f}%, t={nw_t[0]:.2f}{sig}, "
          f"年化={m['port_ret'].mean()*12*100:.1f}%, 夏普={ann_sharpe:.2f}")

reg_df = pd.DataFrame(all_reg)
reg_df.to_csv(os.path.join(OUT, 'all_scenarios_regression.csv'), index=False, encoding='utf-8-sig')

# -------- 2. 滚动窗口 (5年) --------
print("\n[2] 滚动窗口归因 (5年/步长1年)...")
# 用正向纯INV场景
port, _ = backtest(df, ['Size','BM','OP','INV'], top=True)
m = port.merge(ff5, on='Trdmnt', how='inner')
m['excess_ret'] = m['port_ret'] - m['Rf']
m = m.dropna(subset=['excess_ret','MKT','SMB','HML','RMW','CMA'])
m = m.sort_values('Trdmnt').reset_index(drop=True)

window, step = 60, 12
roll_rows = []
for start in range(0, len(m)-window+1, step):
    end = start + window
    chunk = m.iloc[start:end]
    model, nw_t, nw_p = ff5_reg(chunk, nlags=12)
    sig = '***' if nw_p[0]<0.01 else ('**' if nw_p[0]<0.05 else '*')
    roll_rows.append({
        'start': chunk['Trdmnt'].iloc[0], 'end': chunk['Trdmnt'].iloc[-1],
        'mid': chunk['Trdmnt'].iloc[window//2], 'n': len(chunk),
        'Alpha(月)': model.params['const'], 'Alpha_t': nw_t[0], 'Alpha_sig': sig,
        '年化Alpha%': model.params['const']*12*100,
        'MKT': model.params['MKT'], 'SMB': model.params['SMB'],
        'HML': model.params['HML'], 'RMW': model.params['RMW'],
        'CMA': model.params['CMA'], 'AdjR2': model.rsquared_adj
    })

roll_df = pd.DataFrame(roll_rows)
pos_alphas = (roll_df['Alpha(月)'] > 0).sum()
print(f"  {len(roll_df)} 个滚动窗口, Alpha为正: {pos_alphas}/{len(roll_df)}")
roll_df.to_csv(os.path.join(OUT, 'rolling_window.csv'), index=False, encoding='utf-8-sig')

# 反向策略的滚动窗口
port_r, _ = backtest(df, ['Size','BM','OP','INV'], top=False)
m_r = port_r.merge(ff5, on='Trdmnt', how='inner')
m_r['excess_ret'] = m_r['port_ret'] - m_r['Rf']
m_r = m_r.dropna(subset=['excess_ret','MKT','SMB','HML','RMW','CMA'])
m_r = m_r.sort_values('Trdmnt').reset_index(drop=True)

roll_rev_rows = []
for start in range(0, len(m_r)-window+1, step):
    end = start + window
    chunk = m_r.iloc[start:end]
    model, nw_t, nw_p = ff5_reg(chunk, nlags=12)
    sig = '***' if nw_p[0]<0.01 else ('**' if nw_p[0]<0.05 else '*')
    roll_rev_rows.append({
        'start': chunk['Trdmnt'].iloc[0], 'end': chunk['Trdmnt'].iloc[-1],
        'mid': chunk['Trdmnt'].iloc[window//2],
        'Alpha(月)': model.params['const'], 'Alpha_t': nw_t[0], 'Alpha_sig': sig,
        '年化Alpha%': model.params['const']*12*100,
        'AdjR2': model.rsquared_adj
    })
roll_rev_df = pd.DataFrame(roll_rev_rows)
pos_alphas_r = (roll_rev_df['Alpha(月)'] > 0).sum()
print(f"  反向策略: {len(roll_rev_df)} 个滚动窗口, Alpha为正: {pos_alphas_r}/{len(roll_rev_df)}")
roll_rev_df.to_csv(os.path.join(OUT, 'rolling_window_reverse.csv'), index=False, encoding='utf-8-sig')

# -------- 3. 子区间 --------
print("\n[3] 子区间对比...")
subs = [('2005-01','2009-12','05-09'), ('2010-01','2014-12','10-14'),
        ('2015-01','2019-12','15-19'), ('2020-01','2025-12','20-25')]
sub_rows = []
for name, factors, top in [('正向_纯INV',['Size','BM','OP','INV'],True),
                             ('反向_纯INV',['Size','BM','OP','INV'],False)]:
    port, _ = backtest(df, factors, top=top)
    pm = port.merge(ff5, on='Trdmnt', how='inner')
    pm['excess_ret'] = pm['port_ret'] - pm['Rf']
    pm = pm.dropna(subset=['excess_ret','MKT','SMB','HML','RMW','CMA'])
    for s,e,lbl in subs:
        chunk = pm[(pm['Trdmnt']>=s)&(pm['Trdmnt']<=e)]
        if len(chunk)<12: continue
        model, nw_t, nw_p = ff5_reg(chunk, nlags=min(12,len(chunk)//4))
        sub_rows.append({
            '场景': name, '区间': lbl, 'N月': len(chunk),
            'Alpha(月)': model.params['const'], 'Alpha_t': nw_t[0],
            '年化Alpha%': model.params['const']*12*100,
            'AdjR2': model.rsquared_adj,
            'SMB': model.params['SMB']
        })

sub_df = pd.DataFrame(sub_rows)
sub_df.to_csv(os.path.join(OUT, 'subperiod_all.csv'), index=False, encoding='utf-8-sig')

# -------- 4. 冲击成本 --------
print("\n[4] 冲击成本 (参与率模型)...")
d = df.dropna(subset=['Size','BM','OP','INV']).copy()
for f in ['Size','BM','OP','INV']:
    mean = d.groupby('Trdmnt')[f].transform('mean')
    std  = d.groupby('Trdmnt')[f].transform('std').replace(0, np.nan)
    val  = (d[f] - mean) / std
    if f == 'Size': val = -val
    d[f'{f}_z'] = val
d['score'] = d[[f'{f}_z' for f in ['Size','BM','OP','INV']]].mean(axis=1, skipna=True)
d['rank']  = d.groupby('Trdmnt')['score'].rank(pct=True)

# Mnvaltrd 已包含在 factors_panel.csv 中（step2 保存），直接使用

d['adv'] = d['Mnvaltrd'].clip(lower=1e4) / 22
AUM = 100e6
N_pm = d.groupby('Trdmnt')['Stkcd'].transform('count')
d['trade_size'] = AUM / N_pm
d['participation'] = (d['trade_size'] / d['adv']).clip(upper=0.1)
d['sigma'] = np.abs(d['Mretwd']).clip(lower=0.005, upper=0.3)
d['impact_bp'] = d['sigma'] * np.sqrt(d['participation']) * 10000

# 正向和反向的冲击成本
cost_rows = []
for tag, top in [('正向top20%',True), ('反向bottom20%',False)]:
    d['in_port'] = (d['rank'] >= 0.80).astype(int) if top else (d['rank'] < 0.20).astype(int)
    pc = d[d['in_port']==1].groupby('Trdmnt')['impact_bp'].mean()
    d['size_grp'] = d.groupby('Trdmnt')['Size'].transform(
        lambda x: pd.qcut(x, 3, labels=['Small','Mid','Large'], duplicates='drop'))
    bp_sz = d[d['in_port']==1].groupby('size_grp')['impact_bp'].median()
    cost_rows.append({
        '场景': tag, '均值bp': pc.mean(), '中位数bp': pc.median(),
        'P95bp': pc.quantile(0.95), '大盘bp': bp_sz.get('Large', np.nan),
        '中盘bp': bp_sz.get('Mid', np.nan), '小盘bp': bp_sz.get('Small', np.nan)
    })

cost_df = pd.DataFrame(cost_rows)
cost_df.to_csv(os.path.join(OUT, 'impact_cost.csv'), index=False, encoding='utf-8-sig')

# -------- 5. 汇总 --------
print("\n" + "=" * 60)
print("全部检验完成。输出文件:")
for f in sorted(os.listdir(OUT)):
    if f.endswith('.csv'):
        sz = os.path.getsize(os.path.join(OUT, f))/1024
        tag = ' [NEW]' if f in ['all_scenarios_regression.csv','rolling_window.csv',
                                 'rolling_window_reverse.csv','subperiod_all.csv',
                                 'impact_cost.csv'] else ''
        print(f"  {f:35s} {sz:7.1f} KB{tag}")

print(f"\n关键结论:")
pos = reg_df[reg_df['Alpha(月)']>0]['场景'].tolist()
neg = reg_df[reg_df['Alpha(月)']<0]['场景'].tolist()
print(f"  Alpha为正: {pos}")
print(f"  Alpha为负: {neg}")
print(f"  滚动窗口(正向): {pos_alphas}/{len(roll_df)} 为正")
print(f"  滚动窗口(反向): {pos_alphas_r}/{len(roll_rev_df)} 为正")
