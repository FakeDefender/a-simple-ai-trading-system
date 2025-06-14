import logging
import os
import pandas as pd
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
        
        market_data = data_loader.load_data(symbol, interval)
        if market_data is None:
            logger.error("获取市场数据失败")
            return
            
        logger.info(f"成功获取市场数据，数据形状: {market_data.shape}")
        logger.info(f"数据列: {list(market_data.columns)}")
        
        # 检查数据完整性并使用完整数据
        logger.info(f"原始数据: {len(market_data)}行")
        complete_data = market_data.iloc[50:].copy()
        logger.info(f"使用完整数据: {len(complete_data)}行 (跳过前50行)")
        
        # 创建策略代理
        logger.info("创建策略代理")
        strategy_agent = MLStrategyAgent(config, data_loader)
        logger.info("策略代理创建成功")
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
        
        # 获取基准数据并对比分析 
        logger.info("获取纳斯达克100基准数据")
        benchmark_data = data_loader.get_benchmark_data()
        if benchmark_data is not None and not benchmark_data.empty:
            # 生成对齐的equity_curve
            equity_curve = strategy_agent._calculate_equity_curve(
                backtest_results["trades"],
                initial_capital=100000,  # 或从config读取
                all_dates=benchmark_data.index
            )
            beta = strategy_agent._calculate_beta(equity_curve, benchmark_data)
            correlation = strategy_agent._calculate_correlation(equity_curve, benchmark_data)
            logger.info(f"策略与纳斯达克100的Beta: {beta}")
            logger.info(f"策略与纳斯达克100的相关性: {correlation}")
            # 计算基准指数收益率和超额收益
            benchmark_return = benchmark_data['close'].iloc[-1] / benchmark_data['close'].iloc[0] - 1
            strategy_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
            alpha = strategy_return - benchmark_return
            logger.info(f"策略总收益率: {strategy_return:.2%}")
            logger.info(f"基准指数收益率: {benchmark_return:.2%}")
            logger.info(f"超额收益（Alpha）: {alpha:.2%}")
            # 可选：将基准对比结果加入回测结果保存
            extra_metrics = {"beta": beta, "correlation": correlation, "strategy_return": strategy_return, "benchmark_return": benchmark_return, "alpha": alpha}
        else:
            logger.warning("未能获取到纳斯达克100基准数据，无法进行对比分析")
            extra_metrics = {"beta": None, "correlation": None}
        
        # 保存回测结果
        logger.info("保存回测结果到文件")
        # 合并extra_metrics到performance_metrics
        performance_metrics.update(extra_metrics)
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