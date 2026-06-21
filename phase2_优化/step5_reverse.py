"""Step 5: 反向策略 (做多得分最低的 20%)"""
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

print("=" * 60)
print("正向(top20%) vs 反向(bottom20%) 策略对比")
print("=" * 60)

for label, factors in [('纯INV(4因子)', ['Size','BM','OP','INV']),
                        ('去INV(3因子)', ['Size','BM','OP'])]:
    for direction, tag in [(True, '正向 top20%'), (False, '反向 bottom20%')]:
        port_ret, holdings = backtest(df, factors, top=direction)
        m = port_ret.merge(ff5, on='Trdmnt', how='inner')
        m['excess_ret'] = m['port_ret'] - m['Rf']
        m = m.dropna(subset=['excess_ret','MKT','SMB','HML','RMW','CMA'])

        y = m['excess_ret']
        X = sm.add_constant(m[['MKT','SMB','HML','RMW','CMA']])
        model = sm.OLS(y, X).fit()
        nw_cov = cov_hac(model, nlags=12)
        nw_se  = np.sqrt(np.diag(nw_cov))
        nw_t   = model.params.values / nw_se
        nw_p   = 2 * (1 - stats.t.cdf(np.abs(nw_t), df=model.df_resid))
        sig    = '***' if nw_p[0]<0.01 else ('**' if nw_p[0]<0.05 else '')

        print(f"\n{label} {tag}:")
        print(f"  月均持仓: {holdings.mean():.0f}只, 月均收益: {m['port_ret'].mean():.4f} "
              f"(年化 {m['port_ret'].mean()*12*100:.1f}%)")
        print(f"  Alpha(月)={model.params['const']:.5f}, t(NW)={nw_t[0]:.2f}{sig}, "
              f"AdjR2={model.rsquared_adj:.3f}, 夏普(年化)={m['excess_ret'].mean()/m['excess_ret'].std()*np.sqrt(12):.2f}")

print("\n结论: ", end="")
