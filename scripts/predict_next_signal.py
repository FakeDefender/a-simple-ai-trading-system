import pandas as pd
from src.agents.ml_strategy_agent import MLStrategyAgent
from src.utils.data_loader import DataLoader
from src.utils.config_loader import load_config

# 1. 加载配置和数据
config = load_config()
data_loader = DataLoader(config)
symbol = config['data']['symbol']
interval = config['data']['interval']
market_data = data_loader.load_data(symbol, interval, force_update=False)
market_data = market_data.iloc[50:].copy()

# 2. 自动读取最优参数（多重排序：夏普比率、总收益率、最大回撤）
results_file = 'params/param_opt_results.csv'
results_df = pd.read_csv(results_file)
# 多重排序：先夏普，再收益，再回撤
best_row = results_df.sort_values(
    ['sharpe_ratio', 'total_return', 'max_drawdown'],
    ascending=[False, False, True]
).iloc[0]
# 只保留参数列（去掉统计指标）
stat_cols = ['total_return', 'sharpe_ratio', 'win_rate', 'profit_factor', 'max_drawdown', 'var_95', 'volatility']
param_cols = [col for col in results_df.columns if col not in stat_cols]
best_params = best_row[param_cols].to_dict()

# 3. 实例化策略
agent = MLStrategyAgent(config, data_loader, strategy_params=best_params)

# 4. 只用最新一行数据预测
latest_data = market_data.tail(1)
signals = agent.generate_signals(latest_data)
latest_signal = signals.iloc[-1]['signal']
if latest_signal == 1:
    print("建议：买入/做多")
elif latest_signal == -1:
    print("建议：卖出/做空")
else:
    print("建议：观望") 