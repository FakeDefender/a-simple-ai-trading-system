import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.utils.config_loader import load_config
from src.utils.data_loader import DataLoader
from src.agents.ml_strategy_agent import MLStrategyAgent

if __name__ == "__main__":
    # 加载配置和初始化
    config = load_config()
    data_loader = DataLoader(config)
    agent = MLStrategyAgent(config, data_loader)

    # 获取基准数据
    benchmark_data = data_loader.get_benchmark_data()
    print("benchmark_data index range:", benchmark_data.index.min(), benchmark_data.index.max())
    print("benchmark_data close 非空数量:", benchmark_data['close'].notnull().sum())

    # 构造模拟equity_curve，索引与benchmark_data一致，模拟收益曲线
    if not benchmark_data.empty:
        dates = benchmark_data.index
        # 随机生成一个与benchmark_data长度一致的权益曲线
        np.random.seed(42)
        equity_curve = pd.Series(100000 + np.cumsum(np.random.randn(len(dates))*10), index=dates)
        print("equity_curve index range:", equity_curve.index.min(), equity_curve.index.max())
        print("equity_curve 非空数量:", equity_curve.notnull().sum())

        # 计算beta和correlation
        beta = agent._calculate_beta(equity_curve, benchmark_data)
        correlation = agent._calculate_correlation(equity_curve, benchmark_data)
        print(f"测试结果：Beta={beta}, Correlation={correlation}")
    else:
        print("benchmark_data 为空，无法测试。") 