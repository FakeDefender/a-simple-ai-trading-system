import unittest
import logging
from data.yahoo_finance_data import StooqDataFetcher
from strategies.ml_trading_strategy import MLTradingStrategy

class TestMLTradingStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        
        # 配置参数
        cls.config = {
            'symbol': 'aapl.us',  # 使用苹果股票作为测试
            'interval': '1d',  # 使用日线数据
            'lookback_period': 20,  # 回看20个周期
            'prediction_period': 5,  # 预测未来5个周期
            'train_size': 0.8,  # 80%的数据用于训练
            'test_size': 0.2,  # 20%的数据用于测试
            'random_state': 42  # 固定随机种子
        }
        
        # 创建数据获取器
        cls.data_fetcher = StooqDataFetcher()
        
        # 创建策略实例
        cls.strategy = MLTradingStrategy(cls.config)
        
        # 获取训练数据
        cls.training_data = cls.data_fetcher.get_historical_data(
            symbol=cls.config['symbol'],
            interval=cls.config['interval']
        )

    def test_data_fetching(self):
        """测试数据获取功能"""
        self.assertIsNotNone(self.training_data)
        self.assertGreater(len(self.training_data), 0)
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        self.assertTrue(all(col in self.training_data.columns for col in required_columns))

    def test_model_training(self):
        """测试模型训练功能"""
        if self.training_data is None or len(self.training_data) == 0:
            self.skipTest("没有训练数据")
        
        # 训练模型
        model = self.strategy.train_model(self.training_data)
        self.assertIsNotNone(model)

    def test_signal_generation(self):
        """测试信号生成功能"""
        if self.training_data is None or len(self.training_data) == 0:
            self.skipTest("没有训练数据")
        
        # 生成信号
        signals = self.strategy.generate_signals(self.training_data)
        self.assertIsNotNone(signals)
        self.assertGreater(len(signals), 0)

    def test_strategy_execution(self):
        """测试策略执行功能"""
        if self.training_data is None or len(self.training_data) == 0:
            self.skipTest("没有训练数据")
        
        # 执行策略
        results = self.strategy.run(self.training_data)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)

if __name__ == '__main__':
    unittest.main() 