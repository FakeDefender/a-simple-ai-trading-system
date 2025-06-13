import os
import pandas as pd
import numpy as np
from src.agents.ml_strategy_agent import MLStrategyAgent

# 假设market_data已加载为DataFrame，且已包含所有技术指标
# agent建议通过MLStrategyAgent实例获取

def generate_ml_training_data(market_data: pd.DataFrame, agent: MLStrategyAgent, output_csv: str = None):
    """
    遍历market_data,按信号判断函数生成决策点和信号,在决策点获取agent建议,合成训练集。
    如果output_csv已存在则直接读取并返回。
    """
    if output_csv and os.path.exists(output_csv):
        print(f"{output_csv} 已存在，直接读取。")
        return pd.read_csv(output_csv)
    features = []
    position = 0
    entry_price = None
    for i in range(len(market_data)):
        if i % 3 == 0:
            print(f"正在处理第{i}行，共{len(market_data)}行")
        current_data = market_data.iloc[i]
        current_price = float(current_data['close'])
        # 1. 开仓点
        is_long_entry = False
        is_short_entry = False
        is_long_exit = False
        is_short_exit = False
        market_indicators = market_data
        if position == 0:
            if agent._check_long_entry_conditions(current_data, market_indicators, {}):
                is_long_entry = True
            elif agent._check_short_entry_conditions(current_data, market_indicators, {}):
                is_short_entry = True
        elif position == 1:
            if agent._check_long_exit_conditions(current_data, market_indicators, {}):
                is_long_exit = True
        elif position == -1:
            if agent._check_short_exit_conditions(current_data, market_indicators, {}):
                is_short_exit = True
        # 只在决策点采集样本
        if is_long_entry or is_short_entry or is_long_exit or is_short_exit:
            # 获取agent建议
            advice = agent._get_agent_advice(market_data.iloc[:i+1])
            # 合成特征
            row = dict(current_data)
            # strategy_developer参数
            sd_params = advice.get('strategy_developer', {}).get('parameters', {})
            for k, v in sd_params.items():
                row[f'sd_param_{k}'] = v
            # risk_analyst
            risk_level = advice.get('risk_analyst', {}).get('risk_level', None)
            row['risk_level'] = risk_level
            # trading_advisor
            ta_signal = advice.get('trading_advisor', {}).get('current_signal', None)
            row['ta_signal'] = ta_signal
            # 目标变量
            if is_long_entry:
                signal = 1
                position = 1
                entry_price = current_price
            elif is_short_entry:
                signal = -1
                position = -1
                entry_price = current_price
            elif is_long_exit or is_short_exit:
                signal = 0
                position = 0
                entry_price = None
            row['ml_signal'] = signal
            # 下根K线收益率
            if i < len(market_data) - 1:
                next_close = market_data.iloc[i+1]['close']
                row['next_return'] = (next_close - current_price) / current_price
            else:
                row['next_return'] = np.nan
            features.append(row)
    df = pd.DataFrame(features)
    if output_csv:
        df.to_csv(output_csv, index=False)
    return df

# 用法示例：
# from src.agents.ml_strategy_agent import MLStrategyAgent
# agent = MLStrategyAgent(...)
# market_data = pd.read_csv('your_market_data.csv')
# df = generate_ml_training_data(market_data, agent, output_csv='ml_training_data.csv')

if __name__ == "__main__":
    from src.utils.config_loader import load_config
    from src.utils.data_loader import DataLoader
    config = load_config()
    data_loader = DataLoader(config)
    agent = MLStrategyAgent(config, data_loader)
    market_data = pd.read_csv('你的market_data文件.csv')
    df = generate_ml_training_data(market_data, agent, output_csv='ml_training_data.csv')
    print(df.head()) 