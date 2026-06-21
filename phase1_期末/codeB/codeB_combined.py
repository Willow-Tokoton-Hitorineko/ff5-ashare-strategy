"""
================================================================================
代码B - 多因子策略约束分析整合代码
================================================================================
功能：
  1. 数据清洗
  2. 市值分组分析
  3. 流动性分组分析
  4. 冲击成本估算
  5. 约束场景分析
  6. 可视化

输出文件：
  - market_cap_monthly_cleaned.csv
  - turnover_monthly_cleaned.csv
  - sample_universe_cleaned.csv
  - portfolio_holdings.csv
  - size_group_results.csv
  - liquidity_results.csv
  - impact_cost.csv
  - tradeability_summary.csv
  - turnover_tradeoff.png

数据期间：2005-01 至 2025-12
================================================================================
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

input_dir = r'c:\Users\Administrator\Desktop\量化小组作业\期末任务'
parent_dir = r'c:\Users\Administrator\Desktop\量化小组作业'
output_dir = input_dir

# ================================================================================
# 第一部分：数据清洗
# ================================================================================

def process_market_cap():
    """处理市值数据"""
    cap_path = os.path.join(input_dir, 'market_cap_monthly.csv')
    if os.path.exists(cap_path):
        df = pd.read_csv(cap_path)
        print(f"market_cap_monthly.csv 原始列: {df.columns.tolist()}")

        df = df.rename(columns={'Stkcd': 'stock_code', 'Trdmnt': 'date'})
        df = df[['date', 'stock_code', 'Msmvttl']]

        df['date'] = pd.to_datetime(df['date'], format='%Y-%m')
        df['date'] = df['date'].dt.to_period('M').dt.to_timestamp()
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')

        mask = (df['date'] >= '2005-01-01') & (df['date'] <= '2025-12-31')
        df = df[mask]

        output_path = os.path.join(output_dir, 'market_cap_monthly_cleaned.csv')
        df.to_csv(output_path, index=False)
        print(f"处理完成: {output_path}")
        print(f"记录数: {len(df)}")
        print(f"日期范围: {df['date'].min()} 至 {df['date'].max()}")
        print()
    else:
        print(f"文件不存在: {cap_path}")

def process_turnover():
    """处理换手率数据"""
    turnover_path = os.path.join(input_dir, 'turnover_monthly.csv')
    if os.path.exists(turnover_path):
        df = pd.read_csv(turnover_path)
        print(f"turnover_monthly.csv 原始列: {df.columns.tolist()}")

        df = df.rename(columns={'Stkcd': 'stock_code', 'Trdmnt': 'date'})

        if 'turnover' not in df.columns:
            if 'MonthlyTurnover' in df.columns:
                df = df.rename(columns={'MonthlyTurnover': 'turnover'})
            elif 'Mnvaltrd' in df.columns and 'Msmvosd' in df.columns:
                df['turnover'] = df['Mnvaltrd'] / df['Msmvosd']

        df = df[['date', 'stock_code', 'turnover']]

        df['date'] = pd.to_datetime(df['date'], format='%Y-%m')
        df['date'] = df['date'].dt.to_period('M').dt.to_timestamp()
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')

        mask = (df['date'] >= '2005-01-01') & (df['date'] <= '2025-12-31')
        df = df[mask]

        output_path = os.path.join(output_dir, 'turnover_monthly_cleaned.csv')
        df.to_csv(output_path, index=False)
        print(f"处理完成: {output_path}")
        print(f"记录数: {len(df)}")
        print(f"日期范围: {df['date'].min()} 至 {df['date'].max()}")
        print()
    else:
        print(f"文件不存在: {turnover_path}")

def process_sample_universe():
    """处理样本空间数据"""
    universe_path = os.path.join(input_dir, 'sample_universe.csv')
    if os.path.exists(universe_path):
        df = pd.read_csv(universe_path)
        print(f"sample_universe.csv 原始列: {df.columns.tolist()}")

        df = df.rename(columns={'Stkcd': 'stock_code', 'Trdmnt': 'date'})

        if 'tradable' not in df.columns:
            df['tradable'] = 1

        df = df[['date', 'stock_code', 'tradable']]

        df['date'] = pd.to_datetime(df['date'], format='%Y-%m')
        df['date'] = df['date'].dt.to_period('M').dt.to_timestamp()
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')

        mask = (df['date'] >= '2005-01-01') & (df['date'] <= '2025-12-31')
        df = df[mask]

        output_path = os.path.join(output_dir, 'sample_universe_cleaned.csv')
        df.to_csv(output_path, index=False)
        print(f"处理完成: {output_path}")
        print(f"记录数: {len(df)}")
        print(f"日期范围: {df['date'].min()} 至 {df['date'].max()}")
        print()
    else:
        print(f"文件不存在: {universe_path}")

def create_simulated_portfolio():
    """基于sample_universe创建模拟投资组合持仓（等权）"""
    universe_path = os.path.join(input_dir, 'sample_universe_cleaned.csv')
    if os.path.exists(universe_path):
        df = pd.read_csv(universe_path)
        print(f"基于 sample_universe_cleaned.csv 创建模拟投资组合...")

        stock_count_per_month = df.groupby('date')['stock_code'].transform('count')
        df['weight'] = 1.0 / stock_count_per_month

        df = df[['date', 'stock_code', 'weight']]

        output_path = os.path.join(output_dir, 'portfolio_holdings.csv')
        df.to_csv(output_path, index=False)
        print(f"处理完成: {output_path}")
        print(f"记录数: {len(df)}")
        print(f"日期范围: {df['date'].min()} 至 {df['date'].max()}")
        print()
    else:
        print(f"文件不存在，请先运行 sample_universe 处理: {universe_path}")

def clean_data_main():
    """数据清洗主函数"""
    print("=" * 60)
    print("第一部分：数据清洗")
    print("=" * 60)
    print()

    process_market_cap()
    process_turnover()
    process_sample_universe()
    create_simulated_portfolio()

    print("数据清洗完成!")
    print()

# ================================================================================
# 第二部分：市值分组分析
# ================================================================================

def analyze_size_groups():
    """按市值分组分析策略表现"""
    print("=" * 60)
    print("第二部分：市值分组分析")
    print("=" * 60)
    print()

    cap = pd.read_csv(f'{input_dir}/market_cap_monthly.csv')
    portfolio = pd.read_csv(f'{input_dir}/portfolio_holdings.csv')
    universe = pd.read_csv(f'{input_dir}/sample_universe_cleaned.csv')
    returns = pd.read_csv(f'{parent_dir}/TRD_Mnth.csv')

    cap = cap.rename(columns={'Stkcd': 'stock_code', 'Trdmnt': 'date'})
    cap['date'] = pd.to_datetime(cap['date'], format='%Y-%m')
    cap['date'] = cap['date'].dt.to_period('M').dt.to_timestamp()
    cap['date'] = cap['date'].dt.strftime('%Y-%m-%d')

    portfolio['date'] = pd.to_datetime(portfolio['date'])
    portfolio['date'] = portfolio['date'].dt.to_period('M').dt.to_timestamp()
    portfolio['date'] = portfolio['date'].dt.strftime('%Y-%m-%d')

    universe['date'] = pd.to_datetime(universe['date'])
    universe['date'] = universe['date'].dt.to_period('M').dt.to_timestamp()
    universe['date'] = universe['date'].dt.strftime('%Y-%m-%d')

    returns = returns.rename(columns={'Stkcd': 'stock_code', 'Trdmnt': 'date'})
    returns['date'] = pd.to_datetime(returns['date'], format='%Y-%m')
    returns['date'] = returns['date'].dt.to_period('M').dt.to_timestamp()
    returns['date'] = returns['date'].dt.strftime('%Y-%m-%d')

    mask = (cap['date'] >= '2005-01-01') & (cap['date'] <= '2025-12-31')
    cap = cap[mask]
    portfolio = portfolio[portfolio['date'] >= '2005-01-01']
    universe = universe[(universe['date'] >= '2005-01-01') & (universe['date'] <= '2025-12-31')]
    returns = returns[(returns['date'] >= '2005-01-01') & (returns['date'] <= '2025-12-31')]

    portfolio['in_portfolio'] = 1
    merged = cap.merge(portfolio, on=['date', 'stock_code'], how='left')
    merged['in_portfolio'] = merged['in_portfolio'].fillna(0)

    tradable = universe[universe['tradable'] == 1][['date', 'stock_code']].copy()
    merged = merged.merge(tradable, on=['date', 'stock_code'], how='inner')

    def assign_size_group(df):
        df = df.copy()
        df['size_group'] = None
        for date in df['date'].unique():
            mask = df['date'] == date
            month_data = df.loc[mask, 'Msmvttl']
            if month_data.notna().sum() > 0:
                p33 = month_data.quantile(0.333)
                p67 = month_data.quantile(0.667)
                cond = month_data.notna()
                df.loc[mask & cond, 'size_group'] = pd.cut(
                    month_data[cond],
                    bins=[-np.inf, p33, p67, np.inf],
                    labels=['Small', 'Mid', 'Large']
                )
        return df

    merged = assign_size_group(merged)

    returns_merge = returns[['date', 'stock_code', 'Mretwd']].rename(columns={'Mretwd': 'return'})
    merged = merged.merge(returns_merge, on=['date', 'stock_code'], how='left')

    def max_drawdown(cum_returns):
        if len(cum_returns) == 0 or cum_returns.isna().all():
            return np.nan
        cummax = cum_returns.cummax()
        drawdown = (cum_returns - cummax) / cummax
        return drawdown.min()

    results = []
    for group in ['Small', 'Mid', 'Large']:
        group_data = merged[merged['size_group'] == group].copy()
        portfolio_returns = group_data[group_data['in_portfolio'] == 1].groupby('date')['return'].mean()

        if len(portfolio_returns) == 0:
            continue

        annual_return = portfolio_returns.mean() * 12
        annual_vol = portfolio_returns.std() * np.sqrt(12)
        sharpe = (annual_return - 0.02) / annual_vol if annual_vol > 0 else np.nan
        cum_ret = (1 + portfolio_returns).cumprod()
        max_dd = max_drawdown(cum_ret)

        turnover_list = []
        dates = sorted(group_data['date'].unique())
        for i in range(1, len(dates)):
            prev_date = dates[i - 1]
            curr_date = dates[i]
            prev_holdings = set(group_data[(group_data['date'] == prev_date) & (group_data['in_portfolio'] == 1)]['stock_code'])
            curr_holdings = set(group_data[(group_data['date'] == curr_date) & (group_data['in_portfolio'] == 1)]['stock_code'])
            if len(prev_holdings) > 0:
                turnover = len(curr_holdings - prev_holdings) / len(prev_holdings)
                turnover_list.append(turnover)

        avg_turnover = np.mean(turnover_list) if len(turnover_list) > 0 else np.nan

        results.append({
            'group': group,
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'turnover': avg_turnover
        })
        print(f"  {group}: 年化收益={annual_return:.4f}, 夏普={sharpe:.4f}, 最大回撤={max_dd:.4f}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(f'{input_dir}/size_group_results.csv', index=False)
    print(f"\n市值分组结果已保存到: {input_dir}/size_group_results.csv")
    print()

# ================================================================================
# 第三部分：流动性分组分析
# ================================================================================

def analyze_liquidity():
    """按流动性分组分析策略表现"""
    print("=" * 60)
    print("第三部分：流动性分组分析")
    print("=" * 60)
    print()

    turnover = pd.read_csv(f'{input_dir}/turnover_monthly.csv')
    portfolio = pd.read_csv(f'{input_dir}/portfolio_holdings.csv')
    universe = pd.read_csv(f'{input_dir}/sample_universe_cleaned.csv')
    returns = pd.read_csv(f'{parent_dir}/TRD_Mnth.csv')

    turnover = turnover.rename(columns={'Stkcd': 'stock_code', 'Trdmnt': 'date'})
    turnover['date'] = pd.to_datetime(turnover['date'], format='%Y-%m').dt.strftime('%Y-%m-%d')

    if 'turnover' not in turnover.columns:
        if 'MonthlyTurnover' in turnover.columns:
            turnover['turnover'] = turnover['MonthlyTurnover']

    portfolio['date'] = pd.to_datetime(portfolio['date']).dt.strftime('%Y-%m-%d')
    portfolio['in_portfolio'] = 1

    universe['date'] = pd.to_datetime(universe['date']).dt.strftime('%Y-%m-%d')

    returns = returns.rename(columns={'Stkcd': 'stock_code', 'Trdmnt': 'date'})
    returns['date'] = pd.to_datetime(returns['date'], format='%Y-%m').dt.strftime('%Y-%m-%d')

    turnover = turnover[(turnover['date'] >= '2005-01-01') & (turnover['date'] <= '2025-12-31')]
    portfolio = portfolio[portfolio['date'] >= '2005-01-01']
    universe = universe[(universe['date'] >= '2005-01-01') & (universe['date'] <= '2025-12-31')]
    returns = returns[(returns['date'] >= '2005-01-01') & (returns['date'] <= '2025-12-31')]

    merged = turnover[['date', 'stock_code', 'turnover']].merge(
        portfolio[['date', 'stock_code', 'in_portfolio']],
        on=['date', 'stock_code'], how='left'
    )
    merged['in_portfolio'] = merged['in_portfolio'].fillna(0)

    tradable = universe[universe['tradable'] == 1][['date', 'stock_code']]
    merged = merged.merge(tradable, on=['date', 'stock_code'], how='inner')

    merged = merged.dropna(subset=['turnover'])

    merged['liquidity_group'] = merged.groupby('date')['turnover'].transform(
        lambda x: pd.qcut(x, 4, labels=['Q1(lowest)', 'Q2', 'Q3', 'Q4(highest)'], duplicates='drop')
    )

    returns_merge = returns[['date', 'stock_code', 'Mretwd']].rename(columns={'Mretwd': 'return'})
    merged = merged.merge(returns_merge, on=['date', 'stock_code'], how='left')

    def max_drawdown(cum_returns):
        cummax = cum_returns.cummax()
        drawdown = (cum_returns - cummax) / cummax
        return drawdown.min()

    results = []
    for group in ['Q1(lowest)', 'Q2', 'Q3', 'Q4(highest)']:
        group_data = merged[merged['liquidity_group'] == group]
        portfolio_returns = group_data[group_data['in_portfolio'] == 1].groupby('date')['return'].mean()

        if len(portfolio_returns) == 0:
            continue

        annual_return = portfolio_returns.mean() * 12
        annual_vol = portfolio_returns.std() * np.sqrt(12)
        sharpe = (annual_return - 0.02) / annual_vol if annual_vol > 0 else np.nan
        cum_ret = (1 + portfolio_returns).cumprod()
        max_dd = max_drawdown(cum_ret)

        turnover_list = []
        dates = sorted(group_data['date'].unique())
        for i in range(1, len(dates)):
            prev_date = dates[i - 1]
            curr_date = dates[i]
            prev_holdings = set(group_data[(group_data['date'] == prev_date) & (group_data['in_portfolio'] == 1)]['stock_code'])
            curr_holdings = set(group_data[(group_data['date'] == curr_date) & (group_data['in_portfolio'] == 1)]['stock_code'])
            if len(prev_holdings) > 0:
                turnover_rate = len(curr_holdings - prev_holdings) / len(prev_holdings)
                turnover_list.append(turnover_rate)

        avg_turnover = np.mean(turnover_list) if len(turnover_list) > 0 else np.nan

        results.append({
            'liquidity_group': group,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'turnover': avg_turnover
        })
        print(f"  {group}: 年化收益={annual_return:.4f}, 夏普={sharpe:.4f}, 最大回撤={max_dd:.4f}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(f'{input_dir}/liquidity_results.csv', index=False)
    print(f"\n流动性分组结果已保存到: {input_dir}/liquidity_results.csv")
    print()

# ================================================================================
# 第四部分：冲击成本估算
# ================================================================================

def calculate_impact_cost():
    """
    冲击成本估算模型
    说明：由于缺少日频数据，使用简化的基于市值和流动性分组的冲击成本模型
    冲击成本为文献参考值，用于后续约束总表的比较
    """
    print("=" * 60)
    print("第四部分：冲击成本估算")
    print("=" * 60)
    print()

    cap = pd.read_csv(f'{input_dir}/market_cap_monthly.csv')
    turnover = pd.read_csv(f'{input_dir}/turnover_monthly.csv')
    universe = pd.read_csv(f'{input_dir}/sample_universe_cleaned.csv')

    cap = cap.rename(columns={'Stkcd': 'stock_code', 'Trdmnt': 'date'})
    cap['date'] = pd.to_datetime(cap['date'], format='%Y-%m').dt.strftime('%Y-%m-%d')

    turnover = turnover.rename(columns={'Stkcd': 'stock_code', 'Trdmnt': 'date'})
    turnover['date'] = pd.to_datetime(turnover['date'], format='%Y-%m').dt.strftime('%Y-%m-%d')

    if 'turnover' not in turnover.columns:
        if 'MonthlyTurnover' in turnover.columns:
            turnover['turnover'] = turnover['MonthlyTurnover']

    universe['date'] = pd.to_datetime(universe['date']).dt.strftime('%Y-%m-%d')

    cap = cap[(cap['date'] >= '2005-01-01') & (cap['date'] <= '2025-12-31')]
    turnover = turnover[(turnover['date'] >= '2005-01-01') & (turnover['date'] <= '2025-12-31')]
    universe = universe[(universe['date'] >= '2005-01-01') & (universe['date'] <= '2025-12-31')]

    merged = cap[['date', 'stock_code', 'Msmvttl']].merge(
        turnover[['date', 'stock_code', 'turnover']],
        on=['date', 'stock_code'], how='inner'
    )

    tradable = universe[universe['tradable'] == 1][['date', 'stock_code']]
    merged = merged.merge(tradable, on=['date', 'stock_code'], how='inner')

    merged['size_group'] = merged.groupby('date')['Msmvttl'].transform(
        lambda x: pd.qcut(x, 3, labels=['Small', 'Mid', 'Large'], duplicates='drop')
    )

    merged['liquidity_group'] = merged.groupby('date')['turnover'].transform(
        lambda x: pd.qcut(x, 3, labels=['Low', 'Medium', 'High'], duplicates='drop')
    )

    # 冲击成本映射表（单位：bps，基于A股流动性研究文献参考值）
    impact_cost_map = {
        ('Small', 'Low'): 80,
        ('Small', 'Medium'): 50,
        ('Small', 'High'): 20,
        ('Mid', 'Low'): 30,
        ('Mid', 'Medium'): 15,
        ('Mid', 'High'): 8,
        ('Large', 'Low'): 12,
        ('Large', 'Medium'): 6,
        ('Large', 'High'): 3
    }

    merged['impact_cost_bps'] = merged.apply(
        lambda row: impact_cost_map.get((row['size_group'], row['liquidity_group']), np.nan),
        axis=1
    )

    group_results = merged.groupby(['size_group', 'liquidity_group']).agg({
        'impact_cost_bps': 'first'
    }).reset_index()

    group_results = group_results[['size_group', 'liquidity_group', 'impact_cost_bps']]

    group_results.to_csv(f'{input_dir}/impact_cost.csv', index=False)
    print("冲击成本表:")
    print(group_results.to_string(index=False))
    print(f"\n冲击成本结果已保存到: {input_dir}/impact_cost.csv")
    print()

# ================================================================================
# 第五部分：约束场景分析
# ================================================================================

def analyze_tradeability():
    """不同约束场景下的策略表现分析"""
    print("=" * 60)
    print("第五部分：约束场景分析")
    print("=" * 60)
    print()

    portfolio = pd.read_csv(f'{input_dir}/portfolio_holdings.csv')
    cap = pd.read_csv(f'{input_dir}/market_cap_monthly.csv')
    turnover = pd.read_csv(f'{input_dir}/turnover_monthly.csv')
    returns = pd.read_csv(f'{parent_dir}/TRD_Mnth.csv')
    impact_cost = pd.read_csv(f'{input_dir}/impact_cost.csv')

    cap = cap.rename(columns={'Stkcd': 'stock_code', 'Trdmnt': 'date'})
    cap['date'] = pd.to_datetime(cap['date'], format='%Y-%m').dt.strftime('%Y-%m-%d')

    turnover = turnover.rename(columns={'Stkcd': 'stock_code', 'Trdmnt': 'date'})
    turnover['date'] = pd.to_datetime(turnover['date'], format='%Y-%m').dt.strftime('%Y-%m-%d')
    if 'turnover' not in turnover.columns:
        turnover['turnover'] = turnover['MonthlyTurnover']

    portfolio['date'] = pd.to_datetime(portfolio['date']).dt.strftime('%Y-%m-%d')
    portfolio['in_portfolio'] = 1

    returns = returns.rename(columns={'Stkcd': 'stock_code', 'Trdmnt': 'date'})
    returns['date'] = pd.to_datetime(returns['date'], format='%Y-%m').dt.strftime('%Y-%m-%d')

    def assign_groups(df):
        df['size_group'] = df.groupby('date')['Msmvttl'].transform(
            lambda x: pd.qcut(x, 3, labels=['Small', 'Mid', 'Large'], duplicates='drop')
        )
        df['liquidity_group'] = df.groupby('date')['turnover'].transform(
            lambda x: pd.qcut(x, 3, labels=['Low', 'Medium', 'High'], duplicates='drop')
        )
        return df

    stock_info = cap[['date', 'stock_code', 'Msmvttl']].merge(
        turnover[['date', 'stock_code', 'turnover']], on=['date', 'stock_code'], how='inner'
    )

    stock_info = assign_groups(stock_info)
    stock_info = stock_info.merge(impact_cost, on=['size_group', 'liquidity_group'], how='left')

    merged = portfolio[['date', 'stock_code', 'in_portfolio']].merge(
        stock_info, on=['date', 'stock_code'], how='left'
    ).merge(
        returns[['date', 'stock_code', 'Mretwd']].rename(columns={'Mretwd': 'return'}),
        on=['date', 'stock_code'], how='left'
    )

    def calc_metrics(df, keep_col):
        monthly_returns = df[df[keep_col] == 1].groupby('date')['return'].mean()

        if len(monthly_returns) == 0:
            return None

        annual_return = monthly_returns.mean() * 12
        annual_vol = monthly_returns.std() * np.sqrt(12)
        sharpe = (annual_return - 0.02) / annual_vol if annual_vol > 0 else np.nan

        cum_ret = (1 + monthly_returns).cumprod()
        cummax = cum_ret.cummax()
        drawdown = (cum_ret - cummax) / cummax
        max_dd = drawdown.min()

        dates = sorted(df['date'].unique())
        turnover_list = []
        for i in range(1, len(dates)):
            prev = set(df[(df['date'] == dates[i - 1]) & (df[keep_col] == 1)]['stock_code'])
            curr = set(df[(df['date'] == dates[i]) & (df[keep_col] == 1)]['stock_code'])
            if len(prev) > 0:
                turnover_list.append(len(curr - prev) / len(prev))

        avg_turnover = np.mean(turnover_list) if turnover_list else np.nan
        avg_impact = df[df[keep_col] == 1].groupby('date')['impact_cost_bps'].mean().mean()

        return {
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'turnover': avg_turnover,
            'avg_impact_cost_bps': avg_impact
        }

    results = []

    df = merged.copy()

    df['keep'] = df['in_portfolio']
    m = calc_metrics(df, 'keep')
    if m:
        m['scenario'] = '无约束'
        results.append(m)

    df['keep'] = df['in_portfolio'] & (df['Msmvttl'] >= 50)
    m = calc_metrics(df, 'keep')
    if m:
        m['scenario'] = '市值门槛50亿'
        results.append(m)

    df['keep'] = df['in_portfolio'] & (df['Msmvttl'] >= 100)
    m = calc_metrics(df, 'keep')
    if m:
        m['scenario'] = '市值门槛100亿'
        results.append(m)

    df['liquidity_rank'] = df.groupby('date')['turnover'].rank(pct=True)
    df['keep'] = df['in_portfolio'] & (df['liquidity_rank'] > 0.25)
    m = calc_metrics(df, 'keep')
    if m:
        m['scenario'] = '排除最低流动性25%'
        results.append(m)

    df['keep'] = df['in_portfolio'] & (df['Msmvttl'] >= 50) & (df['liquidity_rank'] > 0.25)
    m = calc_metrics(df, 'keep')
    if m:
        m['scenario'] = '联合约束(市值≥50亿且非最低流动性25%)'
        results.append(m)

    result_df = pd.DataFrame(results)
    cols = ['scenario', 'annual_return', 'sharpe', 'max_drawdown', 'turnover', 'avg_impact_cost_bps']
    result_df = result_df[cols]

    result_df.to_csv(f'{input_dir}/tradeability_summary.csv', index=False)
    print("约束场景分析结果:")
    print(result_df.to_string(index=False))
    print(f"\n约束场景结果已保存到: {input_dir}/tradeability_summary.csv")
    print()

# ================================================================================
# 第六部分：可视化
# ================================================================================

def plot_tradeoff():
    """绘制双Y轴折线图"""
    print("=" * 60)
    print("第六部分：可视化")
    print("=" * 60)
    print()

    df = pd.read_csv(f'{input_dir}/tradeability_summary.csv')

    scenario_mapping = {
        '无约束': '无约束',
        '市值门槛50亿': '市值≥50亿',
        '市值门槛100亿': '市值≥100亿',
        '排除最低流动性25%': '排最低流动性组',
        '联合约束(市值≥50亿且非最低流动性25%)': '联合约束'
    }

    df['scenario_mapped'] = df['scenario'].map(scenario_mapping)

    desired_order = ['无约束', '市值≥50亿', '排最低流动性组', '联合约束', '市值≥100亿']
    df['scenario_mapped'] = pd.Categorical(df['scenario_mapped'], categories=desired_order, ordered=True)
    df = df.sort_values('scenario_mapped')

    fig, ax1 = plt.subplots(figsize=(12, 7))

    x = range(len(df))
    annual_return_pct = df['annual_return'] * 100
    turnover_pct = df['turnover'] * 100
    impact_cost = df['avg_impact_cost_bps']

    color1 = 'blue'
    color2 = 'red'

    ax1.set_xlabel('约束严格程度 →', fontsize=12)
    ax1.set_ylabel('年化收益率 (%)', color=color1, fontsize=12)
    ax1.plot(x, annual_return_pct, color=color1, marker='o', linestyle='-',
             linewidth=2, markersize=10, label='年化收益率')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['scenario_mapped'], rotation=30, ha='right', fontsize=10)

    ax2 = ax1.twinx()
    ax2.set_ylabel('换手率 (%)', color=color2, fontsize=12)
    ax2.plot(x, turnover_pct, color=color2, marker='s', linestyle='-',
             linewidth=2, markersize=10, label='换手率')
    ax2.tick_params(axis='y', labelcolor=color2)

    for i, (ret, turn, ic) in enumerate(zip(annual_return_pct, turnover_pct, impact_cost)):
        ax1.annotate(f'{ret:.1f}%', (i, ret), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9, color=color1)
        ax2.annotate(f'{turn:.1f}%', (i, turn), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9, color=color2)
        ax1.annotate(f'({ic:.1f}bp)', (i, ret), textcoords="offset points",
                     xytext=(0, -15), ha='center', fontsize=8, color='gray')

    plt.title('不同约束下的收益与换手权衡', fontsize=14, fontweight='bold', pad=20)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{input_dir}/turnover_tradeoff.png', dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {input_dir}/turnover_tradeoff.png")
    print()

# ================================================================================
# 主函数
# ================================================================================

if __name__ == '__main__':
    print()
    print("=" * 60)
    print("代码B - 多因子策略约束分析")
    print("数据期间：2005-01 至 2025-12")
    print("=" * 60)
    print()

    clean_data_main()
    analyze_size_groups()
    analyze_liquidity()
    calculate_impact_cost()
    analyze_tradeability()
    plot_tradeoff()

    print("=" * 60)
    print("所有分析完成!")
    print("=" * 60)
    print()
    print("输出文件清单:")
    print("  1. market_cap_monthly_cleaned.csv - 市值数据")
    print("  2. turnover_monthly_cleaned.csv - 换手率数据")
    print("  3. sample_universe_cleaned.csv - 样本空间")
    print("  4. portfolio_holdings.csv - 模拟持仓")
    print("  5. size_group_results.csv - 市值分组结果")
    print("  6. liquidity_results.csv - 流动性分组结果")
    print("  7. impact_cost.csv - 冲击成本估算")
    print("  8. tradeability_summary.csv - 约束场景分析")
    print("  9. turnover_tradeoff.png - 可视化图表")