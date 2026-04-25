"""Generate the latest signal using current config and optional optimized params."""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

if __package__ in {None, ""}:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

from src.agents.ml_strategy_agent import MLStrategyAgent
from src.utils.config_loader import load_config
from src.utils.data_loader import DataLoader
from src.utils.data_processor import DataProcessor


PARAM_COLUMNS = {
    "fast_ma",
    "slow_ma",
    "rsi_long_threshold",
    "rsi_short_threshold",
    "rsi_exit_long",
    "rsi_exit_short",
    "min_volume_ratio",
    "stop_loss_pct",
    "take_profit_pct",
}


def load_best_params(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    sort_columns = [column for column in ["sharpe_ratio", "total_return", "max_drawdown"] if column in frame.columns]
    if sort_columns:
        ascending = [False if column != "max_drawdown" else True for column in sort_columns]
        best_row = frame.sort_values(sort_columns, ascending=ascending).iloc[0]
    else:
        best_row = frame.iloc[0]
    return {column: best_row[column] for column in PARAM_COLUMNS if column in frame.columns and pd.notna(best_row[column])}


def load_complete_market_data(config: Dict[str, Any]) -> pd.DataFrame:
    data_loader = DataLoader(config)
    data_processor = DataProcessor()
    data_config = config.get("data", {})
    strategy_config = config.get("strategy", {})
    symbol = data_config.get("symbol", "aapl.us")
    interval = data_config.get("interval", "d")
    market_data = data_loader.load_data(symbol=symbol, interval=interval, force_update=False)
    if market_data is None or market_data.empty:
        raise RuntimeError(f"未获取到 {symbol} 的有效市场数据")
    min_periods = max(50, int(strategy_config.get("slow_ma", 20)))
    return data_processor.get_complete_data(market_data, min_periods=min_periods)


def main():
    parser = argparse.ArgumentParser(description="输出当前标的最新一根 K 线对应的策略信号")
    parser.add_argument("--params", default="params/param_opt_results.csv", help="可选参数搜索结果 CSV")
    args = parser.parse_args()

    config = load_config()
    data_loader = DataLoader(config)
    market_data = load_complete_market_data(config)
    best_params = load_best_params(Path(args.params))

    agent = MLStrategyAgent(config, data_loader, strategy_params=best_params or None)
    signals = agent.generate_signals(market_data)
    latest_timestamp = signals.index[-1]
    latest_signal = int(signals.iloc[-1]["signal"])

    label = {1: "买入/做多", -1: "卖出/做空", 0: "观望"}.get(latest_signal, "未知")
    print(f"最新时间: {latest_timestamp}")
    print(f"最新信号: {latest_signal} ({label})")
    if best_params:
        print(f"使用优化参数: {best_params}")


if __name__ == "__main__":
    main()
