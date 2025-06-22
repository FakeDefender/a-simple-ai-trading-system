import os
import pandas as pd
import numpy as np
from src.agents.ml_strategy_agent import MLStrategyAgent

# 假设market_data已加载为DataFrame，且已包含所有技术指标
# agent建议通过MLStrategyAgent实例获取

def generate_ml_training_data(market_data: pd.DataFrame, agent: MLStrategyAgent, output_csv: str = None):
    """
    先用agent.generate_signals生成信号序列，然后遍历market_data采集特征，ml_signal直接用signals['signal']，保证训练数据标签与实盘一致。
    只采集信号不为0的行。
    """
    if output_csv and os.path.exists(output_csv):
        print(f"{output_csv} 已存在，直接读取。")
        return pd.read_csv(output_csv)
    features = []
    signals = agent.generate_signals(market_data)
    for i in range(len(market_data)):
        signal = signals.iloc[i]['signal'] if 'signal' in signals.columns else signals.iloc[i]
        if signal == 0:
            continue  # 只采集信号不为0的行
        current_data = market_data.iloc[i]
        row = dict(current_data)
        # 可选：采集更多特征，如历史窗口、技术指标等
        # 采集agent建议参数
        advice = agent._get_agent_advice(market_data.iloc[:i+1])
        sd_params = advice.get('strategy_developer', {}).get('parameters', {})
        for k, v in sd_params.items():
            row[f'sd_param_{k}'] = v
        risk_level = advice.get('risk_analyst', {}).get('risk_level', None)
        row['risk_level'] = risk_level
        ta_signal = advice.get('trading_advisor', {}).get('current_signal', None)
        row['ta_signal'] = ta_signal
        row['ml_signal'] = signal
        # 下根K线收益率
        if i < len(market_data) - 1:
            next_close = market_data.iloc[i+1]['close']
            row['next_return'] = (next_close - current_data['close']) / current_data['close']
        else:
            row['next_return'] = np.nan
        features.append(row)
    df = pd.DataFrame(features)
    if output_csv:
        df.to_csv(output_csv, index=False)
    # 特征相关性分析
    if len(df) > 1:
        corr = df.corr(numeric_only=True)
        print('特征相关性矩阵:')
        print(corr)
        corr.to_csv('feature_correlation.csv')
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