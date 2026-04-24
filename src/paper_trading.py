import copy
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


def _build_symbol_config(config: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    symbol_config = copy.deepcopy(config)
    symbol_config.setdefault("data", {})["symbol"] = symbol
    return symbol_config


def _load_symbol_signal(config: Dict[str, Any], symbol: str):
    symbol_config = _build_symbol_config(config, symbol)
    loader = DataLoader(symbol_config)
    processor = DataProcessor()
    interval = symbol_config.get("data", {}).get("interval", "d")
    market_data = loader.load_data(symbol=symbol, interval=interval)
    if market_data is None or market_data.empty:
        raise RuntimeError(f"未获取到 {symbol} 的有效市场数据")
    complete_data = processor.get_complete_data(market_data, min_periods=max(50, int(symbol_config.get("strategy", {}).get("slow_ma", 20))))
    agent = MLStrategyAgent(symbol_config, loader)
    signals = agent.generate_signals(complete_data)
    return complete_data, signals


def main():
    config = load_config()
    output_root = os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_root, exist_ok=True)

    symbols: List[str] = list(dict.fromkeys(config.get("data", {}).get("symbols") or [config.get("data", {}).get("symbol", "aapl.us")]))
    if len(symbols) > 1 and config.get("portfolio", {}).get("enabled", False):
        market_data_by_symbol = {}
        signals_by_symbol = {}
        for symbol in symbols:
            market_data_by_symbol[symbol], signals_by_symbol[symbol] = _load_symbol_signal(config, symbol)
        engine = PortfolioPaperTradingEngine(config)
        results = engine.run(market_data_by_symbol, signals_by_symbol)
        engine.save_results(results, output_root)
        logger.info(
            "组合 paper trading 完成: final_equity=%.2f, total_return=%.2f%%, max_drawdown=%.2f%%, fees_paid=%.2f, rebalances=%s, output=%s",
            results["summary"]["final_equity"],
            results["summary"]["total_return"] * 100,
            results["summary"]["max_drawdown"] * 100,
            results["summary"]["fees_paid"],
            results["summary"]["rebalances"],
            output_root,
        )
    else:
        symbol = symbols[0]
        market_data, signals = _load_symbol_signal(config, symbol)
        engine = PaperTradingEngine(config)
        results = engine.run(market_data, signals, symbol=symbol)
        engine.save_results(results, output_root)
        logger.info(
            "单标的 paper trading 完成: final_equity=%.2f, total_return=%.2f%%, max_drawdown=%.2f%%, fees_paid=%.2f, rebalances=%s, output=%s",
            results["summary"]["final_equity"],
            results["summary"]["total_return"] * 100,
            results["summary"]["max_drawdown"] * 100,
            results["summary"]["fees_paid"],
            results["summary"]["rebalances"],
            output_root,
        )


if __name__ == "__main__":
    main()
