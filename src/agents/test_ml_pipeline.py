import pandas as pd
import os
from src.agents.ml_strategy_agent import MLStrategyAgent
from src.agents.ml_feature_engineering import generate_ml_training_data
import src.agents.ml_xgboost_train as ml_xgb
from src.utils.config_loader import load_config
from src.utils.data_loader import DataLoader

# 配置文件路径
MARKET_DATA_PATH = 'data/market_data/processed/tsla.us_d_technical_indicators_20250614.csv'
TRAIN_CSV = 'ml_training_data.csv'

# 初始化agent（请根据你的实际参数补充）
def get_agent():
    config = load_config()
    # 创建数据加载器
    data_loader = DataLoader(config)
    return MLStrategyAgent(config, data_loader)

def main():
    print('1. 加载market_data...')
    market_data = pd.read_csv(MARKET_DATA_PATH)
    print('数据行数:', len(market_data))
    print('2. 初始化agent...')
    agent = get_agent()
    print('3. 生成训练数据...')
    df = generate_ml_training_data(market_data, agent, output_csv=TRAIN_CSV)
    print('训练数据样例:')
    print(df.head())
    print('4. 训练XGBoost模型...')
    ml_xgb.train_xgboost_models(TRAIN_CSV)
    print('5. 检查模型文件...')
    assert os.path.exists('xgb_signal_classifier.model'), '信号分类模型未生成!'
    assert os.path.exists('xgb_return_regressor.model'), '收益率回归模型未生成!'
    print('全部流程测试通过！')

if __name__ == '__main__':
    main() 