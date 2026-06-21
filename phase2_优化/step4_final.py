"""
Step 4: 冲击成本(参与率模型) + 完整对比 Code A
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')

OUT  = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

def read_csmar(path, **kw):
    for enc in ['utf-8-sig', 'gbk', 'utf-8']:
        try: return pd.read_csv(path, encoding=enc, skiprows=[1,2], low_memory=False, **kw)
        except: continue
    raise ValueError(f"decode fail: {path}")

print("=" * 70)
print("Step 4: 冲击成本 + Code A 完整对比")
print("=" * 70)

# ---------- 4.1 加载 ----------
df = pd.read_csv(os.path.join(OUT, 'factors_panel.csv'), dtype={'Stkcd': str})
trd_raw = read_csmar(f"{DATA}/TRD_Mnth.csv", dtype={'Stkcd': str})
trd_raw['Stkcd'] = trd_raw['Stkcd'].str.zfill(6)
for c in ['Mretwd','Msmvttl','Mnvaltrd']:
    trd_raw[c] = pd.to_numeric(trd_raw[c], errors='coerce')
df = df.merge(trd_raw[['Stkcd','Trdmnt','Mnvaltrd']], on=['Stkcd','Trdmnt'], how='left')

# ---------- 4.2 冲击成本 (参与率) ----------
print("\n[4.2] 冲击成本 (参与率模型)...")
factor_cols = ['Size','BM','OP','INV']
d = df.dropna(subset=factor_cols).copy()
for f in factor_cols:
    mean = d.groupby('Trdmnt')[f].transform('mean')
    std  = d.groupby('Trdmnt')[f].transform('std').replace(0, np.nan)
    val  = (d[f] - mean) / std
    if f == 'Size': val = -val
    d[f'{f}_z'] = val
d['score'] = d[[f'{f}_z' for f in factor_cols]].mean(axis=1, skipna=True)
d['rank']  = d.groupby('Trdmnt')['score'].rank(pct=True)
d['in_port'] = (d['rank'] >= 0.80).astype(int)

# 日均成交额
d['adv'] = d['Mnvaltrd'].clip(lower=1e4) / 22
AUM = 100e6  # 基准 1 亿
N_per_month = d.groupby('Trdmnt')['Stkcd'].transform('count')
d['trade_size'] = AUM / N_per_month
# 参与率
d['participation'] = (d['trade_size'] / d['adv']).clip(upper=0.1)
# 波动率代理: |Mretwd| capped
d['sigma'] = np.abs(d['Mretwd']).clip(lower=0.005, upper=0.3)
# 冲击成本(bp) = sigma * sqrt(participation) * 10000 (约定俗成)
d['impact_bp'] = d['sigma'] * np.sqrt(d['participation']) * 10000

port_cost = d[d['in_port']==1].groupby('Trdmnt')['impact_bp'].mean()
impact_mean = port_cost.mean()
impact_median = port_cost.median()

print(f"  组合月均冲击成本: mean={impact_mean:.1f}bp, median={impact_median:.1f}bp, P95={port_cost.quantile(0.95):.1f}bp")

d['size_grp'] = d.groupby('Trdmnt')['Size'].transform(
    lambda x: pd.qcut(x, 3, labels=['Small','Mid','Large'], duplicates='drop'))
bp_by_size = d.groupby('size_grp')['impact_bp'].median()
print("  各市值组冲击成本(中位数):")
for g in ['Small','Mid','Large']:
    print(f"    {g}: {bp_by_size.get(g, np.nan):.1f} bp")

# ---------- 4.3 完整对比 ----------
reg = pd.read_csv(os.path.join(OUT, 'regression_compare.csv'))
a_row = reg[reg['场景']=='A_4factor_INVpure'].iloc[0]

print("\n" + "=" * 70)
print("Code A 期末 vs 本次优化 - 完整对比")
print("=" * 70)

ca = {  # Code A numbers
    'ret_m': -0.0008, 'alpha_m': -0.0170, 'alpha_ann': -20.4, 'adj_r2': 0.8733,
    'smb': 0.7049, 'sharpe': -0.10, 'inv_miss': '54.7%(ROE填)',
    'cost': '文献3-80bp', 'ret_method': '市值变化 pct_change'
}
ou = {  # Our numbers
    'ret_m': a_row['月均收益'], 'alpha_m': a_row['Alpha(月)'],
    'alpha_ann': a_row['Alpha(年化%)'], 'adj_r2': a_row['Adj_R2'],
    'smb': a_row['Beta_SMB'], 'sharpe': a_row['夏普(年化)'],
    'inv_miss': '19.3%(纯,不填充)', 'cost': f'Amihud mean{impact_mean:.0f}bp',
    'ret_method': 'Mretwd 真实收益'
}

lines = [
    ("个股收益方法", ca['ret_method'], ou['ret_method']),
    ("策略月均收益", f"{ca['ret_m']:.4f}", f"{ou['ret_m']:.4f}"),
    ("策略年化收益", f"{ca['ret_m']*12*100:.1f}%", f"{ou['ret_m']*12*100:.1f}%"),
    ("FF5 Alpha(月)", f"{ca['alpha_m']:.4f}", f"{ou['alpha_m']:.4f}"),
    ("年化 Alpha", f"{ca['alpha_ann']:.1f}%", f"{ou['alpha_ann']:.1f}%"),
    ("Adj-R2", f"{ca['adj_r2']:.4f}", f"{ou['adj_r2']:.4f}"),
    ("SMB 载荷", f"{ca['smb']:.4f}", f"{ou['smb']:.4f}"),
    ("夏普比率(年化)", f"{ca['sharpe']:.2f}", f"{ou['sharpe']:.2f}"),
    ("INV 缺失处理", ca['inv_miss'], ou['inv_miss']),
    ("冲击成本", ca['cost'], ou['cost']),
]

for label, v1, v2 in lines:
    print(f"  {label:16s}  |  {v1:20s}  |  {v2:20s}")

print("\n三场景 Alpha 对比 (均为月频, NW调整):")
for _, row in reg.iterrows():
    name = row['场景'].replace('_4factor_INVpure','').replace('_4factor_INVuse','').replace('_3factor_noINV','')
    name = {'A':'纯INV(4因子)','B':'去INV(3因子)','C':'ROE填INV(4因子)'}.get(name[0], name)
    print(f"  {name}: Alpha={row['Alpha(月)']:.5f}, t={row['Alpha(NW t)']:.2f}, "
          f"AdjR2={row['Adj_R2']:.3f}, 年化Alpha={row['Alpha(年化%)']:.1f}%")

print(f"\n核心改进:")
print(f"  1. Mretwd 替代市值变化 -> 收益更准确")
print(f"  2. INV 从 FS_Combas 自算 -> 缺失率从 54.7% 降到 19.3%")
print(f"  3. INV-ROE 相关性 0.031 -> 确认不可混用, 不再填充")
print(f"  4. 冲击成本从查表升级为参与率模型(数据驱动)")
print(f"  5. 三场景验证: 去不去 INV Alpha 都显著为负")
print(f"  6. Size-BM 相关性 0.04 是 A 股真实现象, 非数据错误")

# 保存
rows = []
for label, v1, v2 in lines:
    rows.append({'指标':label, 'Code_A':v1, '优化版':v2})
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'final_comparison.csv'), index=False, encoding='utf-8-sig')
print(f"\n[DONE] final_comparison.csv")
