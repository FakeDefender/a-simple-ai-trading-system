"""Grid-search baseline strategy parameters.

The script intentionally optimizes only parameters consumed by
MLStrategyAgent.strategy_params. Results are resumable in params/param_opt_results.csv.
"""

import argparse
import itertools
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

if __package__ in {None, ""}:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

from src.agents.ml_strategy_agent import MLStrategyAgent
from src.utils.config_loader import load_config
from src.utils.data_loader import DataLoader
from src.utils.data_processor import DataProcessor


PARAM_GRID: Dict[str, List[Any]] = {
    "fast_ma": [5, 10],
    "slow_ma": [20, 50],
    "rsi_long_threshold": [50, 55, 60],
    "rsi_short_threshold": [40, 45, 50],
    "rsi_exit_long": [45, 48, 50],
    "rsi_exit_short": [50, 52, 55],
    "min_volume_ratio": [1.0, 1.1, 1.2],
    "stop_loss_pct": [0.02, 0.03],
    "take_profit_pct": [0.05, 0.08],
}

def _normalize_param_value(value: Any) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return f"{numeric_value:.12g}"


def _param_fingerprint(params: Dict[str, Any]) -> tuple:
    return tuple((key, _normalize_param_value(params.get(key))) for key in sorted(PARAM_GRID))


def _load_existing_results(output_path: Path) -> tuple[List[Dict[str, Any]], set]:
    if not output_path.exists():
        return [], set()

    frame = pd.read_csv(output_path)
    results = frame.to_dict("records")
    tested = {
        _param_fingerprint({key: row.get(key) for key in PARAM_GRID})
        for row in results
        if all(key in row and not pd.isna(row.get(key)) for key in PARAM_GRID)
    }
    return results, tested


def _iter_param_grid() -> Iterable[Dict[str, Any]]:
    keys = list(PARAM_GRID)
    for values in itertools.product(*(PARAM_GRID[key] for key in keys)):
        yield dict(zip(keys, values))


def _load_market_data(config: Dict[str, Any]) -> pd.DataFrame:
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


def run_optimization(output_path: Path) -> pd.DataFrame:
    config = load_config()
    data_loader = DataLoader(config)
    market_data = _load_market_data(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results, tested = _load_existing_results(output_path)
    combinations = list(_iter_param_grid())
    print(f"总共需要测试 {len(combinations)} 种参数组合，已完成 {len(tested)} 种")

    for index, params in enumerate(combinations, start=1):
        fingerprint = _param_fingerprint(params)
        if fingerprint in tested:
            print(f"跳过已测试参数组合 {index}/{len(combinations)}")
            continue

        print(f"测试参数组合 {index}/{len(combinations)}: {params}")
        try:
            agent = MLStrategyAgent(config, data_loader, strategy_params=params)
            signals = agent.generate_signals(market_data)
            backtest = agent._backtest_strategy(market_data, signals)
            performance = agent._calculate_performance_metrics(backtest)
            risk = agent._calculate_strategy_risk_metrics(backtest)
            result = {**params, **performance, "max_drawdown": risk["max_drawdown"]}
            results.append(result)
            pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8")
            tested.add(fingerprint)
        except Exception as exc:
            print(f"参数组合失败，已跳过: {exc}")

    result_frame = pd.DataFrame(results)
    if not result_frame.empty:
        result_frame.to_csv(output_path, index=False, encoding="utf-8")
        print("最终结果（按夏普比率排序）:")
        print(result_frame.sort_values(["sharpe_ratio", "total_return"], ascending=[False, False]).head())
    return result_frame


def main():
    parser = argparse.ArgumentParser(description="优化基线策略参数")
    parser.add_argument("--output", default="params/param_opt_results.csv", help="参数搜索结果 CSV 路径")
    args = parser.parse_args()
    run_optimization(Path(args.output))


if __name__ == "__main__":
    main()
