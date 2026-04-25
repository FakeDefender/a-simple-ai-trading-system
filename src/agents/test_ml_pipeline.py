"""Manual smoke test for the optional XGBoost training pipeline."""

import os
import sys

if __package__ in {None, ""}:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

from src.agents.ml_feature_engineering import generate_ml_training_data
from src.agents.ml_strategy_agent import MLStrategyAgent
import src.agents.ml_xgboost_train as ml_xgb
from src.utils.config_loader import load_config
from src.utils.data_loader import DataLoader
from src.utils.data_processor import DataProcessor


TRAIN_CSV = "ml_training_data.csv"


def main():
    print("1. 加载配置与市场数据...")
    config = load_config()
    data_loader = DataLoader(config)
    data_processor = DataProcessor()
    data_config = config.get("data", {})
    symbol = data_config.get("symbol", "aapl.us")
    interval = data_config.get("interval", "d")
    market_data = data_loader.load_data(symbol=symbol, interval=interval, force_update=False)
    if market_data is None or market_data.empty:
        raise RuntimeError(f"未获取到 {symbol} 的有效市场数据")
    complete_data = data_processor.get_complete_data(
        market_data,
        min_periods=max(50, int(config.get("strategy", {}).get("slow_ma", 20))),
    )

    print("2. 生成训练数据...")
    agent = MLStrategyAgent(config, data_loader)
    training_data = generate_ml_training_data(complete_data, agent, output_csv=TRAIN_CSV)
    print(training_data.head())

    print("3. 训练 XGBoost 模型...")
    ml_xgb.train_xgboost_models(TRAIN_CSV)
    assert os.path.exists(ml_xgb.CLASSIFIER_MODEL_PATH), "信号分类模型未生成"
    assert os.path.exists(ml_xgb.REGRESSOR_MODEL_PATH), "收益率回归模型未生成"
    print("全部流程测试通过")


if __name__ == "__main__":
    main()
