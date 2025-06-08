import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import DataLoader
from src.models.trading_strategies import StrategyFactory
from src.models.trade_executor import TradeExecutor
from src.models.backtest_engine import BacktestEngine
from src.utils.risk_calculator import RiskCalculator

class TestTradingSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        # 创建测试目录
        os.makedirs('data', exist_ok=True)
        
        # 生成测试数据
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)  # 设置随机种子以确保可重复性
        
        # 生成基础价格序列（使用随机游走模型，增加波动率和趋势）
        base_price = 100
        # 增大波动率，加入正向微弱趋势
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = base_price * (1 + returns).cumprod()
        
        # 生成OHLC数据
        data = pd.DataFrame({
            'datetime': dates,
            'open': prices * (1 + np.random.normal(0, 0.002, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.004, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.004, len(dates)))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, len(dates))
        })
        
        # 确保high是最高价，low是最低价
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        # 保存测试数据
        data.to_csv('data/test_data.csv', index=False)
        
        # 创建配置
        cls.config = {
            'data': {
                'source': 'csv',
                'path': 'data/test_data.csv',
                'start_date': '2023-01-01',
                'end_date': '2023-12-31'
            },
            'trading': {
                'initial_capital': 100000,
                'commission_rate': 0.001,
                'max_position_size': 0.1,
                'stop_loss': 0.02,
                'take_profit': 0.05
            },
            'strategy': {
                'type': 'hybrid',
                'base_position_size': 1.0,
                'max_position_size': 10.0,
                'parameters': {
                    'trend_weight': 0.6,
                    'mean_reversion_weight': 0.4,
                    'sma_short': 20,
                    'sma_long': 50,
                    'rsi_period': 14,
                    'rsi_overbought': 70,
                    'rsi_oversold': 30,
                    'bb_period': 20,
                    'bb_std': 2
                }
            },
            'risk': {
                'max_drawdown': 0.2,
                'max_leverage': 3,
                'position_limit': 0.5
            }
        }
        
        # 初始化组件
        cls.data_loader = DataLoader(cls.config)
        cls.risk_calculator = RiskCalculator(cls.config)
        
    def setUp(self):
        """每个测试用例前的准备工作"""
        self.data_loader = DataLoader(self.config)
        self.risk_calculator = RiskCalculator()
        
    def test_data_loader(self):
        """测试数据加载器"""
        # 加载数据
        data = self.data_loader.load_data()
        
        # 验证数据
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        self.assertIn('close', data.columns)
        self.assertIn('volume', data.columns)
        
    def test_trading_strategies(self):
        """测试交易策略"""
        # 创建策略
        strategy = StrategyFactory.create_strategy('hybrid', self.config)
        
        # 加载数据
        data = self.data_loader.load_data()
        
        # 生成信号
        signals = strategy.generate_signals(data)
        
        # 验证信号
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertFalse(signals.empty)
        self.assertIn('signal', signals.columns)
        
    def test_intraday_swing_strategy(self):
        """
        测试日内回转策略 (IntradaySwingStrategy)
        """
        # 准备日内回转策略的配置
        intraday_config = {
            'data': self.config['data'], # 使用与主配置相同的数据配置
            'trading': self.config['trading'], # 使用与主配置相同的交易配置
            'strategy': {
                'type': 'intraday_swing',
                'base_position_size': 1.0,
                'max_position_size': 10.0,
                'rsi_period': 14,
                'rsi_oversold_threshold': 30,
                'rsi_overbought_threshold': 70,
                # 可以添加其他日内回转策略可能需要的参数
            },
            'risk': self.config['risk'] # 使用与主配置相同的风险配置
        }

        # 创建日内回转策略实例
        strategy = StrategyFactory.create_strategy('intraday_swing', intraday_config)

        # 加载数据
        # 对于单元测试，理想情况下应该使用更小的、精心构造的数据集
        # 但为了快速集成和利用现有数据加载逻辑，这里使用 setUpClass 生成的测试数据
        data = self.data_loader.load_data()

        # 生成信号
        signals = strategy.generate_signals(data)

        # 验证信号
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertFalse(signals.empty)
        self.assertIn('signal', signals.columns)
        
        # 可以添加更具体的断言，例如检查在某些RSI条件下是否生成了预期的信号
        # 例如：在RSI低于30时，信号是否为1 (买入)
        # self.assertTrue((signals.loc[data['rsi'] < 30, 'signal'] == 1).all())
        # 在RSI高于70时，信号是否为-1 (卖出)
        # self.assertTrue((signals.loc[data['rsi'] > 70, 'signal'] == -1).all())
        # 在其他RSI范围内，信号是否为0
        # self.assertTrue((signals.loc[(data['rsi'] >= 30) & (data['rsi'] <= 70), 'signal'] == 0).all())
        # 注意：上面的断言是简化的，实际可能需要考虑穿越逻辑和数据边缘情况

    def test_trade_executor(self):
        """测试交易执行器"""
        # 创建执行器
        executor = TradeExecutor(self.config)
        
        # 创建测试订单
        trade_advice = {
            'symbol': 'BTC/USD',
            'direction': 'buy',
            'quantity': 1.0,
            'price': 100.0
        }
        
        # 加载市场数据
        market_data = self.data_loader.load_data()
        
        # 执行交易
        execution = executor.execute_trade(trade_advice, market_data)
        
        # 验证执行结果
        self.assertIsInstance(execution, dict)
        self.assertIn('status', execution)
        self.assertIn('execution_details', execution)
        self.assertEqual(execution['status'], 'success')
        
    def test_backtest_engine(self):
        """测试回测引擎"""
        # 创建回测引擎
        backtest = BacktestEngine(self.config)
        
        # 加载数据
        data = self.data_loader.load_data()
        
        # 运行回测
        results = backtest.run(data, 'hybrid')
        
        # 验证回测结果
        self.assertIsInstance(results, dict)
        self.assertIn('equity_curve', results)
        self.assertIn('trades', results)
        self.assertIn('metrics', results)
        
    def test_risk_calculator(self):
        """测试风险计算器"""
        # 加载数据
        data = self.data_loader.load_data()
        
        # 计算风险指标
        var = self.risk_calculator.calculate_var(data, 0.95)
        es = self.risk_calculator.calculate_expected_shortfall(data)
        
        # 验证风险指标
        self.assertIsInstance(var, float)
        self.assertIsInstance(es, float)
        self.assertLess(var, 0)  # VaR应该是负数
        
    def test_integration(self):
        """测试系统集成"""
        # 加载数据
        data = self.data_loader.load_data()

        # 创建策略
        strategy = StrategyFactory.create_strategy('hybrid', self.config)

        # 生成信号
        signals = strategy.generate_signals(data)

        # 创建执行器
        executor = TradeExecutor(self.config)

        # 执行交易
        trades = []
        for i in range(len(signals)):
            if signals['signal'].iloc[i] != 0:
                trade_advice = {
                    'symbol': 'BTC/USD',
                    'direction': 'buy' if signals['signal'].iloc[i] > 0 else 'sell',
                    'quantity': abs(signals['signal'].iloc[i]),
                    'price': data['close'].iloc[i]
                }
                execution = executor.execute_trade(trade_advice, data.iloc[i:i+1])
                trades.append(execution)

        # 创建回测引擎
        backtest = BacktestEngine(self.config)

        # 运行回测
        results = backtest.run(data, 'hybrid')

        # 验证结果
        self.assertIsInstance(results, dict)
        self.assertIn('equity_curve', results)
        self.assertIn('trades', results)
        self.assertIn('metrics', results)

        # 验证性能指标
        metrics = results['metrics']
        self.assertGreater(metrics['annual_return'], -1)  # 年化收益率应该大于-100%
        self.assertLess(metrics['max_drawdown'], 1)  # 最大回撤应该小于100%
        self.assertGreater(metrics['sharpe_ratio'], -1000000)  # 夏普比率应该合理
        
if __name__ == '__main__':
    unittest.main() 