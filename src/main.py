import copy
import json
import logging
import os
import sys
from datetime import datetime

if __package__ in {None, ''}:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
from typing import Any, Dict, List

from src.agents.ml_strategy_agent import MLStrategyAgent
from src.execution.paper_trading_engine import PaperTradingEngine
from src.execution.portfolio_paper_trading_engine import PortfolioPaperTradingEngine
from src.utils.config_loader import load_config
from src.utils.data_loader import DataLoader
from src.utils.data_processor import DataProcessor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace("\\", "_").replace(":", "_")


def _build_symbol_config(config: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    symbol_config = copy.deepcopy(config)
    symbol_config.setdefault("data", {})["symbol"] = symbol
    return symbol_config


def _run_symbol_research(config: Dict[str, Any], symbol: str, output_dir: str) -> Dict[str, Any]:
    symbol_config = _build_symbol_config(config, symbol)
    data_loader = DataLoader(symbol_config)
    data_processor = DataProcessor()

    interval = symbol_config.get("data", {}).get("interval", "d")
    market_data = data_loader.load_data(symbol=symbol, interval=interval)
    if market_data is None or market_data.empty:
        raise RuntimeError(f"未获取到 {symbol} 的有效市场数据")

    min_periods = max(50, int(symbol_config.get("strategy", {}).get("slow_ma", 20)))
    complete_data = data_processor.get_complete_data(market_data, min_periods=min_periods)

    strategy_agent = MLStrategyAgent(symbol_config, data_loader)
    strategy_agent.results_dir = output_dir
    signals = strategy_agent.generate_signals(complete_data)
    backtest_results = strategy_agent._backtest_strategy(complete_data, signals)
    performance_metrics = strategy_agent._calculate_performance_metrics(backtest_results)
    risk_metrics = strategy_agent._calculate_strategy_risk_metrics(backtest_results)
    recommendations = strategy_agent._generate_recommendations(backtest_results)

    benchmark_data = data_loader.get_benchmark_data()
    if benchmark_data is not None and not benchmark_data.empty:
        equity_curve = backtest_results["equity_curve"]
        performance_metrics["beta"] = strategy_agent._calculate_beta(equity_curve, benchmark_data)
        performance_metrics["correlation"] = strategy_agent._calculate_correlation(equity_curve, benchmark_data)
        performance_metrics["benchmark_return"] = float(
            benchmark_data["close"].iloc[-1] / benchmark_data["close"].iloc[0] - 1
        )
    else:
        performance_metrics["beta"] = None
        performance_metrics["correlation"] = None
        performance_metrics["benchmark_return"] = None

    strategy_agent._save_backtest_results(
        backtest_results,
        performance_metrics,
        risk_metrics,
        recommendations,
    )

    return {
        "symbol": symbol,
        "agent": strategy_agent,
        "market_data": complete_data,
        "signals": signals,
        "backtest_results": backtest_results,
        "performance_metrics": performance_metrics,
        "risk_metrics": risk_metrics,
        "recommendations": recommendations,
    }


def _run_single_symbol(config: Dict[str, Any], output_root: str):
    symbol = config.get("data", {}).get("symbol", "aapl.us")
    result = _run_symbol_research(config, symbol, output_root)

    paper_results = None
    if config.get("paper_trading", {}).get("enabled", False):
        paper_engine = PaperTradingEngine(config)
        paper_results = paper_engine.run(result["market_data"], result["signals"], symbol=symbol)
        paper_engine.save_results(paper_results, output_root)
        logger.info(
            "paper trading 完成: final_equity=%.2f, total_return=%.2f%%, max_drawdown=%.2f%%, fees_paid=%.2f, rebalances=%s",
            paper_results["summary"]["final_equity"],
            paper_results["summary"]["total_return"] * 100,
            paper_results["summary"]["max_drawdown"] * 100,
            paper_results["summary"]["fees_paid"],
            paper_results["summary"]["rebalances"],
        )

    logger.info("=" * 60)
    logger.info("v0.4 研究与单标的 paper trading 流程完成")
    logger.info(f"回测总收益率: {result['performance_metrics']['total_return']:.2%}")
    logger.info(f"回测年化收益率: {result['performance_metrics']['annual_return']:.2%}")
    logger.info(f"回测夏普比率: {result['performance_metrics']['sharpe_ratio']:.2f}")
    logger.info(f"回测最大回撤: {result['risk_metrics']['max_drawdown']:.2%}")
    if paper_results is not None:
        logger.info(f"Paper Trading 最终权益: {paper_results['summary']['final_equity']:.2f}")
        logger.info(f"Paper Trading 总收益率: {paper_results['summary']['total_return']:.2%}")
        logger.info(f"Paper Trading 最大回撤: {paper_results['summary']['max_drawdown']:.2%}")
        logger.info(f"Paper Trading 累计费用: {paper_results['summary']['fees_paid']:.2f}")
        logger.info(f"Paper Trading 调仓次数: {paper_results['summary']['rebalances']}")
    logger.info(f"结果目录: {output_root}")
    logger.info("=" * 60)


def _run_portfolio(config: Dict[str, Any], symbols: List[str], output_root: str):
    market_data_by_symbol = {}
    signals_by_symbol = {}
    summary_payload = []

    for symbol in symbols:
        symbol_output_dir = os.path.join(output_root, "symbols", _safe_symbol(symbol))
        logger.info(f"开始处理标的: {symbol}")
        result = _run_symbol_research(config, symbol, symbol_output_dir)
        market_data_by_symbol[symbol] = result["market_data"]
        signals_by_symbol[symbol] = result["signals"]
        summary_payload.append(
            {
                "symbol": symbol,
                "performance": result["performance_metrics"],
                "risk": result["risk_metrics"],
                "recommendations": result["recommendations"],
            }
        )

    with open(os.path.join(output_root, "portfolio_research_summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary_payload, file, ensure_ascii=False, indent=2, default=str)

    portfolio_results = None
    if config.get("paper_trading", {}).get("enabled", False) and config.get("portfolio", {}).get("enabled", False):
        portfolio_engine = PortfolioPaperTradingEngine(config)
        portfolio_results = portfolio_engine.run(market_data_by_symbol, signals_by_symbol)
        portfolio_engine.save_results(portfolio_results, output_root)
        logger.info(
            "portfolio paper trading 完成: final_equity=%.2f, total_return=%.2f%%, max_drawdown=%.2f%%, fees_paid=%.2f, rebalances=%s",
            portfolio_results["summary"]["final_equity"],
            portfolio_results["summary"]["total_return"] * 100,
            portfolio_results["summary"]["max_drawdown"] * 100,
            portfolio_results["summary"]["fees_paid"],
            portfolio_results["summary"]["rebalances"],
        )

    logger.info("=" * 60)
    logger.info("v0.4 组合研究与组合 paper trading 流程完成")
    logger.info(f"组合标的数: {len(symbols)}")
    if portfolio_results is not None:
        logger.info(f"组合最终权益: {portfolio_results['summary']['final_equity']:.2f}")
        logger.info(f"组合总收益率: {portfolio_results['summary']['total_return']:.2%}")
        logger.info(f"组合最大回撤: {portfolio_results['summary']['max_drawdown']:.2%}")
        logger.info(f"组合累计费用: {portfolio_results['summary']['fees_paid']:.2f}")
        logger.info(f"组合调仓次数: {portfolio_results['summary']['rebalances']}")
    logger.info(f"结果目录: {output_root}")
    logger.info("=" * 60)


def main():
    try:
        logger.info("开始加载配置")
        config = load_config()
        logger.info("配置加载完成")

        output_root = os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(output_root, exist_ok=True)

        symbols = list(dict.fromkeys(config.get("data", {}).get("symbols") or [config.get("data", {}).get("symbol", "aapl.us")]))
        portfolio_enabled = bool(config.get("portfolio", {}).get("enabled", False))

        if len(symbols) > 1 and portfolio_enabled:
            _run_portfolio(config, symbols, output_root)
        else:
            if len(symbols) > 1 and not portfolio_enabled:
                logger.warning("检测到多个 symbols，但 portfolio.enabled=false，按单标的模式运行第一个标的")
            config.setdefault("data", {})["symbol"] = symbols[0]
            _run_single_symbol(config, output_root)
    except Exception as exc:
        logger.exception(f"程序执行失败: {exc}")
        raise


if __name__ == "__main__":
    main()
