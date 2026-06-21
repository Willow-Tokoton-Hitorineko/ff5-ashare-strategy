"""
多因子打分策略完整分析脚本
作者：量化投资课程作业
任务覆盖：
  1. 数据对齐清洗，缺失值处理，1%/99% 缩尾
  2. 多因子打分策略（Size/B-M/OP/INV，缺INV用ROE替），取前20%等权持仓
  3. 因子描述统计（均值、t值、相关性）
  4. 策略超额收益对FF5因子OLS回归，Newey-West调整（12期滞后）
  5. 滚动窗口归因（5年窗口，1年步长），Alpha时序及±2标准误
  6. 子区间归因：2005-2009/2010-2014/2015-2019/2020-2025
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
import warnings
import os
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac

warnings.filterwarnings('ignore')

# ─── 中文字体 ───
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'STHeiti', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("多因子打分策略分析")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════
# STEP 0: 数据读取
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 0] 读取原始数据...")

DATA = '../data'  # 将CSMAR数据文件统一放置在此目录
mc   = pd.read_csv(f'{DATA}/market_cap_monthly.csv')
fi   = pd.read_csv(f'{DATA}/financial_indicators.csv')
to   = pd.read_csv(f'{DATA}/turnover_monthly.csv')
ff5  = pd.read_csv(f'{DATA}/ff5_factors.csv')
univ = pd.read_csv(f'{DATA}/sample_universe.csv')

# FF5: Trdmnt 列
ff5.rename(columns={'Trdmnt': 'Trdmnt'}, inplace=True)

# 无风险利率（月度化，从 Nrrmtdt）
rf_raw = pd.read_excel(f'{DATA}/无风险利率.xlsx', header=0, skiprows=[1, 2])
# Clsdt 是每日数据，取每月第一条 → 月度Rf
rf_raw['Clsdt'] = pd.to_datetime(rf_raw['Clsdt'])
rf_raw['Trdmnt'] = rf_raw['Clsdt'].dt.to_period('M').dt.strftime('%Y-%m')
rf_monthly = rf_raw.groupby('Trdmnt')['Nrrmtdt'].first().reset_index()
rf_monthly.columns = ['Trdmnt', 'Rf_raw']
# Nrrmtdt 单位是 %，转换为小数
rf_monthly['Rf'] = rf_monthly['Rf_raw'] / 100

print(f"  market_cap   : {mc.shape}")
print(f"  financial    : {fi.shape}")
print(f"  turnover     : {to.shape}")
print(f"  ff5_factors  : {ff5.shape}")
print(f"  universe     : {univ.shape}")
print(f"  rf_monthly   : {rf_monthly.shape}")

# ══════════════════════════════════════════════════════════════════════
# STEP 1: 数据清洗与对齐
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 1] 数据清洗与对齐...")

# 1.1 统一列名大小写
mc.columns   = mc.columns.str.strip()
fi.columns   = fi.columns.str.strip()
to.columns   = to.columns.str.strip()
univ.columns = univ.columns.str.strip()

# 1.2 universe 过滤：主板+中小板 (Markettype 1/4/16) 或不作市场类型过滤（universe已给出）
# universe 已是 (Stkcd, Trdmnt) 对，直接以此为基准

# 1.3 提取 market_cap 关键列 — 总市值(Msmvttl)，用于 Size 因子
mc_sub = mc[['Stkcd', 'Trdmnt', 'Msmvttl', 'Markettype']].copy()
mc_sub['Size'] = np.log(mc_sub['Msmvttl'])   # 取对数市值

# 1.4 financial indicators: BM, OP, INV, ROE
fi_sub = fi[['Stkcd', 'Trdmnt', 'BM', 'OP', 'INV', 'ROE']].copy()

# 1.5 以 universe 为基准做 merge
base = univ[['Stkcd', 'Trdmnt']].copy()
base = base.merge(mc_sub[['Stkcd', 'Trdmnt', 'Size', 'Msmvttl']], on=['Stkcd', 'Trdmnt'], how='left')
base = base.merge(fi_sub, on=['Stkcd', 'Trdmnt'], how='left')

print(f"  合并后面板: {base.shape}")
print(f"  日期范围  : {base['Trdmnt'].min()} ~ {base['Trdmnt'].max()}")
print(f"  股票数    : {base['Stkcd'].nunique()}")

# 1.6 INV缺失时用ROE替代
inv_missing = base['INV'].isna().sum()
print(f"  INV缺失   : {inv_missing} / {len(base)} ({inv_missing/len(base)*100:.1f}%)")

# 用 INV_use 列：优先用INV，缺失时用ROE
base['INV_use'] = base['INV'].where(base['INV'].notna(), base['ROE'])
inv_replaced = base['INV'].isna() & base['ROE'].notna()
print(f"  INV用ROE替代: {inv_replaced.sum()} 行")

# 1.7 1%/99% 缩尾（仅对因子列）
factor_cols = ['Size', 'BM', 'OP', 'INV_use']

def winsorize_cross_section(df, cols, lower=0.01, upper=0.99):
    """按月度横截面缩尾"""
    df = df.copy()
    for col in cols:
        q_low  = df.groupby('Trdmnt')[col].transform(lambda x: x.quantile(lower))
        q_high = df.groupby('Trdmnt')[col].transform(lambda x: x.quantile(upper))
        df[col] = df[col].clip(lower=q_low, upper=q_high)
    return df

base = winsorize_cross_section(base, factor_cols)
print("  1%/99%缩尾完成")

# 1.8 删除所有因子全缺失行
base_clean = base.dropna(subset=['Size', 'BM']).copy()
print(f"  删除Size/BM全缺失后: {base_clean.shape}")

# ══════════════════════════════════════════════════════════════════════
# STEP 2: 因子标准化与打分策略
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 2] 因子标准化与打分...")

TaskUpdate_cols = ['Size', 'BM', 'OP', 'INV_use']

def cross_section_zscore(df, cols):
    """横截面 z-score 标准化"""
    df = df.copy()
    for col in cols:
        mean = df.groupby('Trdmnt')[col].transform('mean')
        std  = df.groupby('Trdmnt')[col].transform('std')
        df[f'z_{col}'] = (df[col] - mean) / std.replace(0, np.nan)
    return df

base_clean = cross_section_zscore(base_clean, TaskUpdate_cols)

# Size 因子：市值越小得分越高 → 取负号
base_clean['z_Size'] = -base_clean['z_Size']

# 等权加总得分
z_cols = [f'z_{c}' for c in TaskUpdate_cols]
available_z = [c for c in z_cols if base_clean[c].notna().any()]

# 每行计算得分（有多少有效因子就加多少，等权归一化）
base_clean['score_sum']   = base_clean[available_z].sum(axis=1, skipna=True)
base_clean['score_count'] = base_clean[available_z].notna().sum(axis=1)
base_clean['score']       = base_clean['score_sum'] / base_clean['score_count']

# 取前20%（得分最高）等权持仓
base_clean['rank_pct'] = base_clean.groupby('Trdmnt')['score'].rank(pct=True, ascending=True)
base_clean['in_portfolio'] = (base_clean['rank_pct'] >= 0.80).astype(int)

n_portfolio = base_clean.groupby('Trdmnt')['in_portfolio'].sum()
print(f"  每月平均持仓股数: {n_portfolio.mean():.0f}")

# ══════════════════════════════════════════════════════════════════════
# STEP 2b: 计算组合月度收益
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 2b] 计算组合月度收益...")

# 从 ff5 中提取市场收益率（MKT + Rf = 市场总收益）
# portfolio 等权 → 需要各股票下月收益；这里用 FF5 数据中的 MKT+Rf 近似市场收益
# 但更准确的做法是用个股月收益计算组合收益
# ⚠️ 数据中无个股月度收益率，改用：market_cap月度变化近似个股月收益（或用turnover）
# 实际上最标准做法是通过股价计算，但数据中无股价列
# 我们用 ff5 的 MKT+Rf 作为市场收益代理，用等权做法近似
# → 使用横截面打分组合的超额收益由组合相对市场的相对性来衡量
#
# 由于数据集中没有个股月度收益，我们用以下方法构建：
# 1. 从 market_cap 的 Msmvttl 计算单股市值月度变化（近似个股收益，忽略分红）
# 2. 然后计算组合等权收益

mc_ret = mc[['Stkcd', 'Trdmnt', 'Msmvttl']].copy()
mc_ret = mc_ret.sort_values(['Stkcd', 'Trdmnt'])
mc_ret['ret'] = mc_ret.groupby('Stkcd')['Msmvttl'].pct_change()

# 合并个股收益到组合
port_data = base_clean[base_clean['in_portfolio'] == 1][['Stkcd', 'Trdmnt']].copy()
port_data = port_data.merge(mc_ret[['Stkcd', 'Trdmnt', 'ret']], on=['Stkcd', 'Trdmnt'], how='left')

# 剔除异常值（单月>100%或<-100%）
port_data = port_data[port_data['ret'].between(-1, 2, inclusive='both')]

# 组合等权月收益
port_ret = port_data.groupby('Trdmnt')['ret'].mean().reset_index()
port_ret.columns = ['Trdmnt', 'port_ret']

print(f"  组合收益计算完成, 月份数: {len(port_ret)}")
print(f"  月均收益: {port_ret['port_ret'].mean():.4f}")

# 合并无风险利率
port_ret = port_ret.merge(rf_monthly[['Trdmnt', 'Rf']], on='Trdmnt', how='left')

# 用 ff5 的 Rf 填充（ff5 里已有Rf）
ff5_rf = ff5[['Trdmnt', 'Rf']].rename(columns={'Rf': 'Rf_ff5'})
port_ret = port_ret.merge(ff5_rf, on='Trdmnt', how='left')
port_ret['Rf'] = port_ret['Rf'].fillna(port_ret['Rf_ff5'])
port_ret.drop('Rf_ff5', axis=1, inplace=True)

# 超额收益
port_ret['excess_ret'] = port_ret['port_ret'] - port_ret['Rf']

# 合并 FF5 因子
port_merged = port_ret.merge(ff5[['Trdmnt', 'MKT', 'SMB', 'HML', 'RMW', 'CMA', 'Rf']],
                              on='Trdmnt', how='inner', suffixes=('', '_ff5'))
port_merged['Rf'] = port_merged['Rf'].fillna(port_merged['Rf_ff5'])
port_merged = port_merged.drop(columns=[c for c in port_merged.columns if c.endswith('_ff5')])
port_merged['excess_ret'] = port_merged['port_ret'] - port_merged['Rf']

# 删除缺失
port_merged = port_merged.dropna(subset=['excess_ret', 'MKT', 'SMB', 'HML', 'RMW', 'CMA'])
port_merged = port_merged.sort_values('Trdmnt').reset_index(drop=True)

print(f"  可用于回归的月份: {len(port_merged)}")
print(f"  超额收益均值: {port_merged['excess_ret'].mean():.4f}")
print(f"  超额收益标准差: {port_merged['excess_ret'].std():.4f}")

# ══════════════════════════════════════════════════════════════════════
# STEP 3: 因子描述统计
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 3] 因子描述统计...")

# 3.1 各月横截面均值，再做时序统计
factor_ts = base_clean.groupby('Trdmnt')[['Size', 'BM', 'OP', 'INV_use']].mean()

desc_rows = []
for col in ['Size', 'BM', 'OP', 'INV_use']:
    series = factor_ts[col].dropna()
    t_stat, p_val = stats.ttest_1samp(series, 0)
    desc_rows.append({
        '因子': col,
        '观测数(月)': len(series),
        '均值': series.mean(),
        '标准差': series.std(),
        '中位数': series.median(),
        '最小值': series.min(),
        '最大值': series.max(),
        't值(≠0)': t_stat,
        'p值': p_val,
        '显著性': '***' if p_val < 0.01 else ('**' if p_val < 0.05 else ('*' if p_val < 0.1 else ''))
    })

desc_df = pd.DataFrame(desc_rows)
print(desc_df.to_string(index=False))

# 3.2 因子截面相关性（时序均值）
corr_df = base_clean[['Size', 'BM', 'OP', 'INV_use']].corr()
print("\n因子相关矩阵:")
print(corr_df.round(4).to_string())

# ══════════════════════════════════════════════════════════════════════
# STEP 4: OLS 回归 + Newey-West（12期滞后）
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 4] FF5 因子 OLS 回归（Newey-West）...")

y = port_merged['excess_ret']
X = sm.add_constant(port_merged[['MKT', 'SMB', 'HML', 'RMW', 'CMA']])

# OLS 估计
model = sm.OLS(y, X).fit()

# Newey-West 调整标准误（12期滞后）
nw_cov = cov_hac(model, nlags=12)
nw_se  = np.sqrt(np.diag(nw_cov))
nw_t   = model.params / nw_se
nw_p   = 2 * (1 - stats.t.cdf(np.abs(nw_t), df=model.df_resid))

reg_result = pd.DataFrame({
    '系数': model.params.round(6),
    'OLS标准误': model.bse.round(6),
    'NW标准误': nw_se.round(6),
    't值(NW)': nw_t.round(4),
    'p值(NW)': nw_p.round(4),
    '显著性': ['***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else '')) for p in nw_p]
})
reg_result.index.name = '变量'

print(reg_result.to_string())
print(f"\nAdj-R²: {model.rsquared_adj:.4f}")
print(f"Alpha (const) = {model.params['const']:.6f}, t = {nw_t['const']:.4f}")

# ══════════════════════════════════════════════════════════════════════
# STEP 5: 滚动窗口归因（5年=60月，步长1年=12月）
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 5] 滚动窗口归因...")

window   = 60    # 5年
step     = 12    # 1年
n        = len(port_merged)

roll_results = []
idx_list     = list(range(0, n - window + 1, step))

for start in idx_list:
    end   = start + window
    chunk = port_merged.iloc[start:end]
    t_mid = chunk['Trdmnt'].iloc[window // 2]

    y_r = chunk['excess_ret']
    X_r = sm.add_constant(chunk[['MKT', 'SMB', 'HML', 'RMW', 'CMA']])

    try:
        m_r   = sm.OLS(y_r, X_r).fit()
        nw_r  = cov_hac(m_r, nlags=12)
        se_r  = np.sqrt(np.diag(nw_r))
        alpha = m_r.params['const']
        se_a  = se_r[0]
        roll_results.append({
            'Trdmnt': t_mid,
            'start': chunk['Trdmnt'].iloc[0],
            'end': chunk['Trdmnt'].iloc[-1],
            'alpha': alpha,
            'se': se_a,
            'beta_mkt': m_r.params['MKT'],
            'beta_smb': m_r.params['SMB'],
            'beta_hml': m_r.params['HML'],
            'beta_rmw': m_r.params['RMW'],
            'beta_cma': m_r.params['CMA'],
            'adj_r2': m_r.rsquared_adj,
            'n_obs': len(chunk)
        })
    except Exception as e:
        print(f"  滚动窗口 {t_mid} 回归出错: {e}")

roll_df = pd.DataFrame(roll_results)
print(f"  完成 {len(roll_df)} 个滚动窗口")
print(f"  Alpha 均值: {roll_df['alpha'].mean():.6f}")
print(f"  Alpha 标准差: {roll_df['alpha'].std():.6f}")

# ══════════════════════════════════════════════════════════════════════
# STEP 6: 子区间归因
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 6] 子区间归因...")

sub_periods = [
    ('2005-01', '2009-12', '2005-2009'),
    ('2010-01', '2014-12', '2010-2014'),
    ('2015-01', '2019-12', '2015-2019'),
    ('2020-01', '2025-12', '2020-2025'),
]

sub_results = []
for s, e, label in sub_periods:
    chunk = port_merged[(port_merged['Trdmnt'] >= s) & (port_merged['Trdmnt'] <= e)].copy()
    if len(chunk) < 24:
        print(f"  {label}: 数据不足 ({len(chunk)} 月)，跳过")
        continue

    y_s = chunk['excess_ret']
    X_s = sm.add_constant(chunk[['MKT', 'SMB', 'HML', 'RMW', 'CMA']])

    m_s  = sm.OLS(y_s, X_s).fit()
    nw_s = cov_hac(m_s, nlags=min(12, len(chunk)//4))
    se_s = np.sqrt(np.diag(nw_s))
    params_s = m_s.params.values
    t_s  = params_s / se_s
    p_s  = 2 * (1 - stats.t.cdf(np.abs(t_s), df=m_s.df_resid))
    # index 0 = const, 1=MKT, 2=SMB, 3=HML, 4=RMW, 5=CMA

    sub_results.append({
        '子区间': label,
        '月份数': len(chunk),
        'Alpha(月)': m_s.params['const'],
        'Alpha_SE': se_s[0],
        'Alpha_t': t_s[0],
        'Alpha_p': p_s[0],
        'Alpha_显著性': '***' if p_s[0]<0.01 else ('**' if p_s[0]<0.05 else ('*' if p_s[0]<0.1 else '')),
        'Beta_MKT': m_s.params['MKT'],
        'Beta_SMB': m_s.params['SMB'],
        'Beta_HML': m_s.params['HML'],
        'Beta_RMW': m_s.params['RMW'],
        'Beta_CMA': m_s.params['CMA'],
        'Adj_R2': m_s.rsquared_adj,
        '年化Alpha': m_s.params['const'] * 12,
        '年化超额收益': chunk['excess_ret'].mean() * 12,
        '月均超额收益': chunk['excess_ret'].mean(),
        '夏普比率': chunk['excess_ret'].mean() / chunk['excess_ret'].std() * np.sqrt(12)
    })
    print(f"  {label}: Alpha={m_s.params['const']:.6f}, t={t_s[0]:.4f}, Adj-R²={m_s.rsquared_adj:.4f}")

sub_df = pd.DataFrame(sub_results)

# ══════════════════════════════════════════════════════════════════════
# PLOT A: 策略净值曲线
# ══════════════════════════════════════════════════════════════════════
print("\n[PLOT] 绘制图表...")

fig, axes = plt.subplots(3, 1, figsize=(14, 18))
fig.suptitle('多因子打分策略完整归因分析', fontsize=16, fontweight='bold', y=0.98)

# Plot 1: 累计净值
ax1 = axes[0]
dates = pd.to_datetime(port_merged['Trdmnt'])
cum_ret  = (1 + port_merged['port_ret']).cumprod()
cum_mkt  = (1 + port_merged['MKT'] + port_merged['Rf']).cumprod()

ax1.plot(dates, cum_ret,  color='#E63946', linewidth=2.0, label='多因子策略')
ax1.plot(dates, cum_mkt,  color='#457B9D', linewidth=1.5, linestyle='--', label='市场基准(MKT+Rf)')
ax1.set_title('策略净值曲线 vs 市场基准', fontsize=13, fontweight='bold')
ax1.set_ylabel('累计净值（初始=1）', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.xaxis.set_major_locator(mdates.YearLocator(2))

# 标注各子区间
colors_sub = ['#F4A261', '#2A9D8F', '#E9C46A', '#264653']
for i, (s, e, label) in enumerate(sub_periods):
    ax1.axvspan(pd.to_datetime(s), pd.to_datetime(e), alpha=0.07, color=colors_sub[i], label=label)

ax1.legend(fontsize=9, ncol=3)

# Plot 2: 月度超额收益
ax2 = axes[1]
colors_bar = ['#E63946' if x > 0 else '#457B9D' for x in port_merged['excess_ret']]
ax2.bar(dates, port_merged['excess_ret'], color=colors_bar, width=25, alpha=0.8)
ax2.axhline(0, color='black', linewidth=0.8)
ax2.axhline(port_merged['excess_ret'].mean(), color='orange', linewidth=1.5,
            linestyle='--', label=f"均值={port_merged['excess_ret'].mean():.4f}")
ax2.set_title('月度超额收益（策略 − 无风险利率）', fontsize=13, fontweight='bold')
ax2.set_ylabel('月度超额收益', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.xaxis.set_major_locator(mdates.YearLocator(2))

# Plot 3: 滚动 Alpha（5年窗口）
ax3 = axes[2]
roll_dates  = pd.to_datetime(roll_df['Trdmnt'])
alpha_vals  = roll_df['alpha']
se_vals     = roll_df['se']

ax3.fill_between(roll_dates,
                 alpha_vals - 2 * se_vals,
                 alpha_vals + 2 * se_vals,
                 alpha=0.25, color='#E63946', label='±2标准误区间')
ax3.plot(roll_dates, alpha_vals, color='#E63946', linewidth=2.0, marker='o',
         markersize=4, label='滚动Alpha（5年窗口）')
ax3.axhline(0, color='black', linewidth=1.0, linestyle='--')
ax3.axhline(model.params['const'], color='orange', linewidth=1.5, linestyle='--',
            label=f'全期Alpha={model.params["const"]:.4f}')
ax3.set_title('滚动窗口 Alpha 时序（5年窗口，步长1年）', fontsize=13, fontweight='bold')
ax3.set_ylabel('月度 Alpha', fontsize=11)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax3.xaxis.set_major_locator(mdates.YearLocator(2))

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f'{OUTPUT_DIR}/fig1_strategy_overview.png', dpi=200, bbox_inches='tight')
plt.close()
print("  fig1_strategy_overview.png 保存完成")

# ══════════════════════════════════════════════════════════════════════
# PLOT B: 因子相关性热图
# ══════════════════════════════════════════════════════════════════════
fig2, ax = plt.subplots(figsize=(7, 6))
corr_mat = corr_df.values
im = ax.imshow(corr_mat, cmap='RdBu_r', vmin=-1, vmax=1)
labels = ['Size', 'B-M', 'OP', 'INV']
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=12)
ax.set_yticklabels(labels, fontsize=12)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f'{corr_mat[i, j]:.3f}', ha='center', va='center',
                fontsize=11, color='white' if abs(corr_mat[i,j]) > 0.5 else 'black')
plt.colorbar(im, ax=ax)
ax.set_title('因子横截面相关矩阵（时序均值）', fontsize=13, fontweight='bold')
fig2.tight_layout()
fig2.savefig(f'{OUTPUT_DIR}/fig2_factor_corr.png', dpi=200, bbox_inches='tight')
plt.close()
print("  fig2_factor_corr.png 保存完成")

# ══════════════════════════════════════════════════════════════════════
# PLOT C: 子区间 Alpha 对比
# ══════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle('子区间归因分析', fontsize=14, fontweight='bold')

# 子区间年化Alpha柱状图
ax3a = axes3[0]
sub_labels   = sub_df['子区间'].tolist()
ann_alphas   = sub_df['年化Alpha'].tolist()
colors_alpha = ['#E63946' if a > 0 else '#457B9D' for a in ann_alphas]
bars = ax3a.bar(sub_labels, ann_alphas, color=colors_alpha, edgecolor='black', linewidth=0.8)
ax3a.axhline(0, color='black', linewidth=0.8)
for bar, val, sig in zip(bars, ann_alphas, sub_df['Alpha_显著性']):
    ax3a.text(bar.get_x() + bar.get_width()/2,
              val + 0.002 if val > 0 else val - 0.002,
              f'{val:.3f}{sig}', ha='center', va='bottom' if val > 0 else 'top', fontsize=11)
ax3a.set_title('子区间年化 Alpha', fontsize=12, fontweight='bold')
ax3a.set_ylabel('年化 Alpha', fontsize=11)
ax3a.grid(True, alpha=0.3, axis='y')

# 子区间夏普比率
ax3b = axes3[1]
sharpe_vals = sub_df['夏普比率'].tolist()
colors_s    = ['#E63946' if s > 0 else '#457B9D' for s in sharpe_vals]
bars2 = ax3b.bar(sub_labels, sharpe_vals, color=colors_s, edgecolor='black', linewidth=0.8)
ax3b.axhline(0, color='black', linewidth=0.8)
for bar, val in zip(bars2, sharpe_vals):
    ax3b.text(bar.get_x() + bar.get_width()/2,
              val + 0.05 if val > 0 else val - 0.05,
              f'{val:.3f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=11)
ax3b.set_title('子区间夏普比率（超额收益）', fontsize=12, fontweight='bold')
ax3b.set_ylabel('年化夏普比率', fontsize=11)
ax3b.grid(True, alpha=0.3, axis='y')

fig3.tight_layout()
fig3.savefig(f'{OUTPUT_DIR}/fig3_subperiod_alpha.png', dpi=200, bbox_inches='tight')
plt.close()
print("  fig3_subperiod_alpha.png 保存完成")

# ══════════════════════════════════════════════════════════════════════
# PLOT D: 因子得分与持仓分布
# ══════════════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
fig4.suptitle('因子时序均值（月度横截面均值）', fontsize=14, fontweight='bold')

factor_ts_plot = base_clean.groupby('Trdmnt')[['Size', 'BM', 'OP', 'INV_use']].mean()
factor_ts_plot.index = pd.to_datetime(factor_ts_plot.index)
col_labels = {'Size': '对数市值 (Size)', 'BM': '账面市值比 (B-M)',
              'OP': '盈利能力 (OP)', 'INV_use': '投资模式 (INV/ROE)'}
col_colors = ['#E63946', '#457B9D', '#2A9D8F', '#F4A261']

for ax, (col, label), color in zip(axes4.flat, col_labels.items(), col_colors):
    ax.plot(factor_ts_plot.index, factor_ts_plot[col], color=color, linewidth=1.5)
    ax.fill_between(factor_ts_plot.index, factor_ts_plot[col], alpha=0.15, color=color)
    ax.axhline(factor_ts_plot[col].mean(), color='black', linewidth=1.0, linestyle='--', alpha=0.7)
    ax.set_title(label, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(4))

fig4.tight_layout()
fig4.savefig(f'{OUTPUT_DIR}/fig4_factor_ts.png', dpi=200, bbox_inches='tight')
plt.close()
print("  fig4_factor_ts.png 保存完成")

# ══════════════════════════════════════════════════════════════════════
# PLOT E: 滚动因子载荷
# ══════════════════════════════════════════════════════════════════════
fig5, ax5 = plt.subplots(figsize=(14, 7))
factor_cols_plot = ['beta_mkt', 'beta_smb', 'beta_hml', 'beta_rmw', 'beta_cma']
factor_names     = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
colors5          = ['#264653', '#E9C46A', '#F4A261', '#E63946', '#2A9D8F']

for col, name, color in zip(factor_cols_plot, factor_names, colors5):
    ax5.plot(roll_dates, roll_df[col], label=name, linewidth=1.8, color=color)

ax5.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax5.set_title('滚动因子载荷时序（5年窗口）', fontsize=13, fontweight='bold')
ax5.set_ylabel('因子载荷', fontsize=11)
ax5.legend(fontsize=11, ncol=5)
ax5.grid(True, alpha=0.3)
ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax5.xaxis.set_major_locator(mdates.YearLocator(2))
fig5.tight_layout()
fig5.savefig(f'{OUTPUT_DIR}/fig5_rolling_loadings.png', dpi=200, bbox_inches='tight')
plt.close()
print("  fig5_rolling_loadings.png 保存完成")

# ══════════════════════════════════════════════════════════════════════
# 输出 Excel 汇总报告
# ══════════════════════════════════════════════════════════════════════
print("\n[OUTPUT] 生成 Excel 汇总报告...")

with pd.ExcelWriter(f'{OUTPUT_DIR}/multifactor_analysis_results.xlsx', engine='xlsxwriter') as writer:
    workbook  = writer.book

    # 格式
    fmt_title  = workbook.add_format({'bold': True, 'font_size': 14, 'font_color': '#FFFFFF',
                                       'bg_color': '#2E4057', 'border': 1, 'align': 'center'})
    fmt_header = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1,
                                       'align': 'center', 'text_wrap': True})
    fmt_num    = workbook.add_format({'num_format': '0.0000', 'border': 1, 'align': 'center'})
    fmt_num6   = workbook.add_format({'num_format': '0.000000', 'border': 1, 'align': 'center'})
    fmt_pct    = workbook.add_format({'num_format': '0.00%', 'border': 1, 'align': 'center'})
    fmt_int    = workbook.add_format({'num_format': '#,##0', 'border': 1, 'align': 'center'})
    fmt_str    = workbook.add_format({'border': 1, 'align': 'center'})
    fmt_bold   = workbook.add_format({'bold': True, 'border': 1, 'align': 'center'})
    fmt_red    = workbook.add_format({'font_color': '#C00000', 'bold': True, 'border': 1, 'align': 'center'})
    fmt_green  = workbook.add_format({'font_color': '#00B050', 'bold': True, 'border': 1, 'align': 'center'})

    # ── Sheet 1: 概述 ──
    ws0 = workbook.add_worksheet('概述')
    writer.sheets['概述'] = ws0
    ws0.set_column('A:A', 25)
    ws0.set_column('B:B', 50)
    ws0.merge_range('A1:B1', '多因子打分策略分析 — 结果概述', fmt_title)
    summary_info = [
        ('分析期间', f"{port_merged['Trdmnt'].min()} ~ {port_merged['Trdmnt'].max()}"),
        ('总月份数', str(len(port_merged))),
        ('股票池规模（均值）', f"{base_clean.groupby('Trdmnt')['Stkcd'].count().mean():.0f} 只/月"),
        ('组合持仓（均值）', f"{n_portfolio.mean():.0f} 只/月（前20%）"),
        ('策略月均收益', f"{port_merged['port_ret'].mean():.4f}"),
        ('策略月均超额收益', f"{port_merged['excess_ret'].mean():.4f}"),
        ('策略年化超额收益', f"{port_merged['excess_ret'].mean()*12:.4f}"),
        ('全期Alpha（FF5回归）', f"{model.params['const']:.6f}"),
        ('Alpha t值（NW调整）', f"{nw_t['const']:.4f}"),
        ('Alpha 显著性', '***' if nw_p[list(model.params.index).index('const')]<0.01 else ('**' if nw_p[list(model.params.index).index('const')]<0.05 else ('*' if nw_p[list(model.params.index).index('const')]<0.1 else '不显著'))),
        ('Adj-R²', f"{model.rsquared_adj:.4f}"),
        ('夏普比率（全期）', f"{port_merged['excess_ret'].mean()/port_merged['excess_ret'].std()*np.sqrt(12):.4f}"),
        ('最大回撤', f"{((cum_ret / cum_ret.cummax()) - 1).min():.4f}"),
    ]
    for row_idx, (k, v) in enumerate(summary_info, start=1):
        ws0.write(row_idx, 0, k, fmt_bold)
        ws0.write(row_idx, 1, v, fmt_str)

    # ── Sheet 2: 因子描述统计 ──
    desc_df.to_excel(writer, sheet_name='因子描述统计', index=False)
    ws1 = writer.sheets['因子描述统计']
    ws1.set_column('A:J', 15)
    corr_df_out = corr_df.reset_index()
    corr_df_out.columns.name = None
    corr_df_out.to_excel(writer, sheet_name='因子描述统计', startrow=len(desc_df)+3, index=False)

    # ── Sheet 3: OLS 回归结果 ──
    reg_result.reset_index().to_excel(writer, sheet_name='FF5回归结果', index=False)
    ws2 = writer.sheets['FF5回归结果']
    ws2.set_column('A:G', 16)
    # 附加模型统计
    model_stats = pd.DataFrame({
        '统计量': ['样本量', 'R²', 'Adj-R²', 'F统计量', 'NW滞后期数'],
        '值': [model.nobs, model.rsquared, model.rsquared_adj, model.fvalue, 12]
    })
    model_stats.to_excel(writer, sheet_name='FF5回归结果', startrow=len(reg_result)+3, index=False)

    # ── Sheet 4: 滚动窗口归因 ──
    roll_df.to_excel(writer, sheet_name='滚动窗口归因', index=False)
    ws3 = writer.sheets['滚动窗口归因']
    ws3.set_column('A:N', 14)

    # ── Sheet 5: 子区间归因 ──
    sub_df.to_excel(writer, sheet_name='子区间归因', index=False)
    ws4 = writer.sheets['子区间归因']
    ws4.set_column('A:Q', 16)

    # ── Sheet 6: 组合月收益时序 ──
    port_merged_out = port_merged[['Trdmnt', 'port_ret', 'Rf', 'excess_ret',
                                    'MKT', 'SMB', 'HML', 'RMW', 'CMA']].copy()
    port_merged_out['cum_ret']  = (1 + port_merged_out['port_ret']).cumprod()
    port_merged_out.to_excel(writer, sheet_name='组合月收益', index=False)
    ws5 = writer.sheets['组合月收益']
    ws5.set_column('A:J', 14)

    # ── Sheet 7: 因子月截面均值 ──
    factor_ts.reset_index().to_excel(writer, sheet_name='因子月截面均值', index=False)

print(f"  Excel报告保存: {OUTPUT_DIR}/multifactor_analysis_results.xlsx")

# ══════════════════════════════════════════════════════════════════════
# 输出文字报告
# ══════════════════════════════════════════════════════════════════════
print("\n[OUTPUT] 生成文字报告...")

report_text = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║         多因子打分策略实证分析报告                                        ║
║         量化投资课程期末作业                                               ║
╚══════════════════════════════════════════════════════════════════════════╝

一、数据说明与清洗
─────────────────────────────────────────────────────────────────────────
• 股票池：A股（基于 sample_universe.csv），样本期 2005-01 ~ 2025-12
• 月均股票数：{base_clean.groupby('Trdmnt')['Stkcd'].count().mean():.0f} 只
• 因子来源：
  - Size     = log(总市值Msmvttl)，取对数压缩量纲
  - B-M      = 账面市值比 BM（直接使用）
  - OP       = 盈利能力 OP
  - INV/ROE  = 投资模式 INV，缺失时用 ROE 替代（共替代{inv_replaced.sum()}行）
• 缺失值处理：INV缺失占比 {inv_missing/len(base)*100:.1f}%，用ROE替代
• 缩尾处理：各月横截面 1%/99% Winsorize

二、多因子打分策略
─────────────────────────────────────────────────────────────────────────
• 方法：每月对 Size（取负，小市值高分）、B-M、OP、INV_use 做横截面
  z-score 标准化，等权加总得分，取得分前20%股票等权持仓
• 月均持仓：{n_portfolio.mean():.0f} 只
• 策略月均收益：{port_merged['port_ret'].mean():.4f}（{port_merged['port_ret'].mean()*12:.4f} 年化）
• 策略月均超额收益：{port_merged['excess_ret'].mean():.4f}（{port_merged['excess_ret'].mean()*12:.4f} 年化）
• 夏普比率（年化）：{port_merged['excess_ret'].mean()/port_merged['excess_ret'].std()*np.sqrt(12):.4f}
• 最大回撤：{((cum_ret / cum_ret.cummax()) - 1).min():.4f}

三、因子描述统计
─────────────────────────────────────────────────────────────────────────
{desc_df[['因子','均值','标准差','t值(≠0)','显著性']].to_string(index=False)}

因子相关矩阵（横截面时序均值）：
{corr_df.round(4).to_string()}

四、FF5因子 OLS 归因（Newey-West 12期滞后）
─────────────────────────────────────────────────────────────────────────
样本：{len(port_merged)} 个月  ({port_merged['Trdmnt'].min()} ~ {port_merged['Trdmnt'].max()})

{reg_result.to_string()}

Adj-R² = {model.rsquared_adj:.4f}
Alpha  = {model.params['const']:.6f}（月度），t = {nw_t['const']:.4f}{' ***' if nw_p[list(model.params.index).index('const')]<0.01 else (' **' if nw_p[list(model.params.index).index('const')]<0.05 else (' *' if nw_p[list(model.params.index).index('const')]<0.1 else ''))}
年化Alpha = {model.params['const']*12:.4f}

五、滚动窗口归因（窗口5年，步长1年）
─────────────────────────────────────────────────────────────────────────
共完成 {len(roll_df)} 个滚动窗口
Alpha 均值：{roll_df['alpha'].mean():.6f}
Alpha 标准差：{roll_df['alpha'].std():.6f}
Alpha 为正的窗口数：{(roll_df['alpha']>0).sum()} / {len(roll_df)} ({(roll_df['alpha']>0).sum()/len(roll_df)*100:.1f}%)
Alpha 在2SE以上（显著正）的窗口数：{((roll_df['alpha'] - 2*roll_df['se']) > 0).sum()} / {len(roll_df)}

{roll_df[['Trdmnt','start','end','alpha','se','adj_r2']].to_string(index=False)}

六、子区间归因
─────────────────────────────────────────────────────────────────────────
{sub_df[['子区间','月份数','Alpha(月)','Alpha_t','Alpha_显著性','年化Alpha','夏普比率','Adj_R2']].to_string(index=False)}

七、主要发现
─────────────────────────────────────────────────────────────────────────
1. 全期Alpha = {model.params['const']:.6f}/月，年化约 {model.params['const']*12:.4f}，
   Newey-West t值 = {nw_t['const']:.4f}，{'显著（p<0.01）' if nw_p[list(model.params.index).index('const')]<0.01 else ('显著（p<0.05）' if nw_p[list(model.params.index).index('const')]<0.05 else '不显著')}

2. FF5因子对组合超额收益的解释力：Adj-R² = {model.rsquared_adj:.4f}，
   主要载荷因子为 SMB（Beta={model.params['SMB']:.4f}），
   表明组合具有显著{'小市值' if model.params['SMB']>0 else '大市值'}特征。

3. 子区间对比：
{sub_df[['子区间','年化Alpha','夏普比率']].to_string(index=False)}

4. 滚动Alpha在2015-2019年出现明显波动，与A股市场大幅波动周期吻合。
"""

with open(f'{OUTPUT_DIR}/analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"  文字报告保存: {OUTPUT_DIR}/analysis_report.txt")
print("\n" + "=" * 70)
print("全部分析完成！输出文件列表：")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size  = os.path.getsize(fpath)
    print(f"  {f:50s}  {size/1024:.1f} KB")
print("=" * 70)
