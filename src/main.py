import logging
import os
from src.agents.ml_strategy_agent import MLStrategyAgent
from src.utils.config_loader import load_config
from src.utils.data_loader import DataLoader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    try:
        # 加载配置
        logger.info("开始加载配置")
        config = load_config()
        logger.info("配置加载成功")
        
        # 创建数据加载器
        data_loader = DataLoader(config)
        logger.info("数据加载器创建成功")
        
        # 获取市场数据
        symbol = config['data']['symbol']
        interval = config['data']['interval']
        logger.info(f"开始获取数据 - 交易品种: {symbol}, 时间间隔: {interval}")
        
        market_data = data_loader.load_data(symbol, interval, force_update=True)
        if market_data is None:
            logger.error("获取市场数据失败")
            return
            
        logger.info(f"成功获取市场数据，数据形状: {market_data.shape}")
        logger.info(f"数据列: {list(market_data.columns)}")
        
        # 检查数据完整性并使用完整数据
        logger.info(f"原始数据: {len(market_data)}行")
        complete_data = market_data.iloc[20:].copy()  # 跳过前20行，确保主要指标完整
        logger.info(f"使用完整数据: {len(complete_data)}行 (跳过前20行)")
        
        # 创建策略代理
        logger.info("创建策略代理")
        strategy_agent = MLStrategyAgent(config, data_loader)
        logger.info("策略代理创建成功")
        
        # 准备市场数据摘要（使用完整数据的最新值）
        market_summary = {
            'symbol': symbol,
            'interval': interval,
            'latest_price': complete_data['close'].iloc[-1],
            'moving_averages': {
                'ma5': complete_data['ma5'].iloc[-1],
                'ma10': complete_data['ma10'].iloc[-1],
                'ma20': complete_data['ma20'].iloc[-1]
            },
            'rsi': complete_data['rsi'].iloc[-1],
            'config': config
        }
        
        logger.info("市场数据准备完成")
        logger.info(f"最新价格: {market_summary['latest_price']}")
        logger.info(f"移动平均线: {market_summary['moving_averages']}")
        logger.info(f"RSI: {market_summary['rsi']}")
        
        # 开始市场分析
        logger.info("开始市场分析")
        market_indicators = data_loader.data_processor.calculate_market_indicators(market_summary)
        logger.info(f"市场分析完成 - 趋势强度: {market_indicators['trend_strength']:.4f}, 市场强度: {market_indicators['market_strength']:.4f}, 波动性: {market_indicators['volatility']:.4f}")
        
        # 生成交易信号（传入完整数据）
        signals = strategy_agent.generate_signals(complete_data)
        logger.info("交易信号生成完成")
        
        # 输出信号统计
        logger.info(f"信号数据形状: {signals.shape}")
        signal_counts = signals['signal'].value_counts()
        logger.info(f"信号统计: {signal_counts.to_dict()}")
        
        # 运行回测分析
        logger.info("开始运行回测分析")
        backtest_results = strategy_agent._backtest_strategy(complete_data, signals)
        logger.info("回测分析完成")
        
        # 计算绩效指标
        performance_metrics = strategy_agent._calculate_performance_metrics(backtest_results)
        logger.info(f"绩效指标: {performance_metrics}")
        
        # 计算风险指标
        risk_metrics = strategy_agent._calculate_strategy_risk_metrics(backtest_results)
        logger.info(f"风险指标: {risk_metrics}")
        
        # 生成优化建议
        recommendations = strategy_agent._generate_recommendations(backtest_results)
        logger.info(f"优化建议: {recommendations}")
        
        # 保存回测结果
        logger.info("保存回测结果到文件")
        strategy_agent._save_backtest_results(
            backtest_results, 
            performance_metrics, 
            risk_metrics, 
            recommendations
        )
        
        # 输出最终总结
        logger.info("=" * 60)
        logger.info("回测分析完成！")
        logger.info(f"总收益率: {performance_metrics.get('total_return', 0):.2%}")
        logger.info(f"夏普比率: {performance_metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"胜率: {performance_metrics.get('win_rate', 0):.2%}")
        logger.info(f"最大回撤: {risk_metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"结果已保存到: {strategy_agent.results_dir}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        import traceback
        logger.error(f"错误堆栈: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main() 