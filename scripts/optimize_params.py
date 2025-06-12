import itertools
import pandas as pd
import os
from src.agents.ml_strategy_agent import MLStrategyAgent
from src.utils.data_loader import DataLoader
from src.utils.config_loader import load_config

# 1. 加载配置和数据
config = load_config()
data_loader = DataLoader(config)
symbol = config['data']['symbol']
interval = config['data']['interval']
market_data = data_loader.load_data(symbol, interval, force_update=False)
market_data = market_data.iloc[50:].copy()  # 跳过缺失指标的前N行

# 2. 定义参数搜索空间
param_grid = {
    # 只保留核心参数
    'rsi_long': [40, 45, 50, 55],              # 多头RSI阈值
    'rsi_short': [50, 55, 60, 65],             # 空头RSI阈值
    'volume_ratio': [1.05, 1.10, 1.15, 1.20],  # 成交量比率
    'stop_loss_pct': [0.01, 0.02, 0.03],       # 止损比例
    'take_profit_pct': [0.03, 0.05, 0.07, 0.08], # 止盈比例
    'min_entry_conditions': [2, 3],            # 最少满足条件数
}
# 其它参数固定
FIXED_PARAMS = {
    'rsi_exit_long': 70,
    'rsi_exit_short': 25,
    'macd_trend': 0,
    'ma5_offset': 0.99,
    'ma5_offset_short': 1.01,
    'bb_upper_offset': 0.97,
    'bb_lower_offset': 1.01,
}

# 创建params目录
os.makedirs('params', exist_ok=True)

# 3. 遍历参数组合
keys, values = zip(*param_grid.items())
results = []
total_combinations = len(list(itertools.product(*values)))
print(f"总共需要测试 {total_combinations} 种参数组合")

# 尝试加载已有的结果
results_file = 'params/param_opt_results.csv'
if os.path.exists(results_file):
    results_df = pd.read_csv(results_file)
    results = results_df.to_dict('records')
    print(f"已加载 {len(results)} 条已有结果")

# 获取已测试的参数组合
tested_params = set(tuple(sorted(d.items())) for d in results)

# 遍历参数组合
for i, v in enumerate(itertools.product(*values)):
    params = dict(zip(keys, v))
    params.update(FIXED_PARAMS)
    # 检查是否已经测试过这个参数组合
    param_tuple = tuple(sorted(params.items()))
    if param_tuple in tested_params:
        print(f"跳过已测试的参数组合 {i+1}/{total_combinations}")
        continue
    print(f"测试参数组合 {i+1}/{total_combinations}")
    print(f"当前参数: {params}")
    try:
        agent = MLStrategyAgent(config, data_loader, strategy_params=params)
        signals = agent.generate_signals(market_data)
        backtest = agent._backtest_strategy(market_data, signals)
        perf = agent._calculate_performance_metrics(backtest)
        perf['max_drawdown'] = agent._calculate_strategy_risk_metrics(backtest)['max_drawdown']
        result = {**params, **perf}
        results.append(result)
        # 实时保存结果
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_file, index=False)
        # 打印当前最优结果
        print("\n当前最优结果（按夏普比率排序）:")
        print(results_df.sort_values('sharpe_ratio', ascending=False).head())
        print("\n" + "="*50 + "\n")
    except Exception as e:
        print(f"测试参数组合时出错: {str(e)}")
        continue
# 4. 输出最终结果
results_df = pd.DataFrame(results)
results_df.to_csv(results_file, index=False)
print("\n最终结果（按夏普比率排序）:")
print(results_df.sort_values('sharpe_ratio', ascending=False).head()) 