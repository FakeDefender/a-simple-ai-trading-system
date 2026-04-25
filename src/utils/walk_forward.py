"""Walk-forward evaluation helpers for single-symbol and multi-symbol validation."""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.agents.ml_strategy_agent import MLStrategyAgent
from src.execution.paper_trading_engine import PaperTradingEngine
from src.utils.data_loader import DataLoader
from src.utils.data_processor import DataProcessor


def build_walk_forward_folds(
    index: pd.Index,
    train_size: int,
    test_size: int,
    step_size: Optional[int] = None,
) -> List[Dict[str, int]]:
    train_size = int(train_size)
    test_size = int(test_size)
    step_size = int(step_size or test_size)
    if train_size <= 0 or test_size <= 0 or step_size <= 0:
        raise ValueError("train_size、test_size、step_size 必须大于 0")
    if len(index) < train_size + test_size:
        raise ValueError("数据不足以构造一个 walk-forward 窗口")

    folds: List[Dict[str, int]] = []
    fold_number = 1
    train_end = train_size
    while train_end + test_size <= len(index):
        train_start = train_end - train_size
        test_start = train_end
        test_end = test_start + test_size
        folds.append(
            {
                "fold": fold_number,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        train_end += step_size
        fold_number += 1
    return folds


def _build_symbol_config(config: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    symbol_config = copy.deepcopy(config)
    symbol_config.setdefault("data", {})["symbol"] = symbol
    return symbol_config


def _safe_symbol(symbol: str) -> str:
    return str(symbol).replace("/", "_").replace("\\", "_").replace(":", "_")


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_complete_data(config: Dict[str, Any], symbol: str):
    symbol_config = _build_symbol_config(config, symbol)
    data_loader = DataLoader(symbol_config)
    data_processor = DataProcessor()
    interval = symbol_config.get("data", {}).get("interval", "d")
    market_data = data_loader.load_data(symbol=symbol, interval=interval)
    if market_data is None or market_data.empty:
        raise RuntimeError(f"未获取到 {symbol} 的有效市场数据")
    min_periods = max(50, int(symbol_config.get("strategy", {}).get("slow_ma", 20)))
    complete_data = data_processor.get_complete_data(market_data, min_periods=min_periods)
    if complete_data is None or complete_data.empty:
        raise RuntimeError(f"{symbol} 的技术指标数据为空")
    return symbol_config, data_loader, complete_data


def _resolve_symbols(config: Dict[str, Any], symbols: Optional[List[str]] = None) -> List[str]:
    if symbols:
        return list(dict.fromkeys([str(symbol) for symbol in symbols]))
    data_config = config.get("data", {})
    configured_symbols = data_config.get("symbols") or [data_config.get("symbol", "aapl.us")]
    return list(dict.fromkeys([str(symbol) for symbol in configured_symbols if symbol]))


def _evaluate_slice(
    symbol_config: Dict[str, Any],
    data_loader: DataLoader,
    market_data: pd.DataFrame,
    symbol: str,
    include_paper: bool,
) -> Dict[str, Any]:
    agent = MLStrategyAgent(symbol_config, data_loader)
    signals = agent.generate_signals(market_data)
    backtest_results = agent._backtest_strategy(market_data, signals)
    performance_metrics = agent._calculate_performance_metrics(backtest_results)
    risk_metrics = agent._calculate_strategy_risk_metrics(backtest_results)

    paper_summary = None
    if include_paper and bool(symbol_config.get("paper_trading", {}).get("enabled", False)):
        engine = PaperTradingEngine(symbol_config)
        paper_results = engine.run(market_data, signals, symbol=symbol)
        paper_summary = paper_results["summary"]

    return {
        "rows": len(market_data),
        "start": str(market_data.index[0]),
        "end": str(market_data.index[-1]),
        "performance": performance_metrics,
        "risk": risk_metrics,
        "paper_summary": paper_summary,
    }


def _aggregate_walk_forward(folds: List[Dict[str, Any]]) -> Dict[str, Any]:
    test_returns = [_safe_float(fold["test"]["performance"].get("total_return")) for fold in folds]
    test_returns = [value for value in test_returns if value is not None]
    test_sharpes = [_safe_float(fold["test"]["performance"].get("sharpe_ratio")) for fold in folds]
    test_sharpes = [value for value in test_sharpes if value is not None]
    test_drawdowns = [_safe_float(fold["test"]["risk"].get("max_drawdown")) for fold in folds]
    test_drawdowns = [value for value in test_drawdowns if value is not None]
    paper_returns = [
        _safe_float((fold["test"].get("paper_summary") or {}).get("total_return"))
        for fold in folds
    ]
    paper_returns = [value for value in paper_returns if value is not None]

    if not folds:
        return {"fold_count": 0}

    test_return_series = pd.Series(test_returns, dtype=float) if test_returns else pd.Series(dtype=float)
    test_sharpe_series = pd.Series(test_sharpes, dtype=float) if test_sharpes else pd.Series(dtype=float)
    test_drawdown_series = pd.Series(test_drawdowns, dtype=float) if test_drawdowns else pd.Series(dtype=float)
    paper_return_series = pd.Series(paper_returns, dtype=float) if paper_returns else pd.Series(dtype=float)

    return {
        "fold_count": len(folds),
        "positive_test_fold_rate": float((test_return_series > 0).mean()) if not test_return_series.empty else 0.0,
        "avg_test_return": float(test_return_series.mean()) if not test_return_series.empty else 0.0,
        "median_test_return": float(test_return_series.median()) if not test_return_series.empty else 0.0,
        "avg_test_sharpe": float(test_sharpe_series.mean()) if not test_sharpe_series.empty else 0.0,
        "worst_test_drawdown": float(test_drawdown_series.min()) if not test_drawdown_series.empty else 0.0,
        "avg_paper_return": float(paper_return_series.mean()) if not paper_return_series.empty else None,
    }


def _run_single_symbol_walk_forward(
    runtime_config: Dict[str, Any],
    symbol: str,
    output_root: str,
    train_size: int,
    test_size: int,
    step_size: Optional[int],
    include_paper: bool,
) -> Dict[str, Any]:
    symbol_config, data_loader, complete_data = _load_complete_data(runtime_config, symbol)
    folds = build_walk_forward_folds(
        complete_data.index,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
    )

    os.makedirs(output_root, exist_ok=True)
    fold_payloads: List[Dict[str, Any]] = []
    flat_rows: List[Dict[str, Any]] = []
    for fold in folds:
        train_data = complete_data.iloc[fold["train_start"] : fold["train_end"]].copy()
        test_data = complete_data.iloc[fold["test_start"] : fold["test_end"]].copy()

        train_result = _evaluate_slice(symbol_config, data_loader, train_data, symbol, include_paper=False)
        test_result = _evaluate_slice(symbol_config, data_loader, test_data, symbol, include_paper=include_paper)
        payload = {
            "fold": fold["fold"],
            "train": train_result,
            "test": test_result,
        }
        fold_payloads.append(payload)
        flat_rows.append(
            {
                "symbol": symbol,
                "fold": fold["fold"],
                "train_start": train_result["start"],
                "train_end": train_result["end"],
                "test_start": test_result["start"],
                "test_end": test_result["end"],
                "train_total_return": _safe_float(train_result["performance"].get("total_return")),
                "train_sharpe_ratio": _safe_float(train_result["performance"].get("sharpe_ratio")),
                "test_total_return": _safe_float(test_result["performance"].get("total_return")),
                "test_sharpe_ratio": _safe_float(test_result["performance"].get("sharpe_ratio")),
                "test_max_drawdown": _safe_float(test_result["risk"].get("max_drawdown")),
                "paper_total_return": _safe_float((test_result.get("paper_summary") or {}).get("total_return")),
                "paper_max_drawdown": _safe_float((test_result.get("paper_summary") or {}).get("max_drawdown")),
                "paper_rebalances": (test_result.get("paper_summary") or {}).get("rebalances"),
            }
        )

    summary = _aggregate_walk_forward(fold_payloads)
    summary.update(
        {
            "symbol": symbol,
            "available_rows": int(len(complete_data)),
            "train_size": int(train_size),
            "test_size": int(test_size),
            "step_size": int(step_size or test_size),
            "include_paper": bool(include_paper),
        }
    )

    with open(os.path.join(output_root, "walk_forward_summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(output_root, "walk_forward_folds.json"), "w", encoding="utf-8") as file:
        json.dump(fold_payloads, file, ensure_ascii=False, indent=2, default=str)
    pd.DataFrame(flat_rows).to_csv(
        os.path.join(output_root, "walk_forward_folds.csv"),
        index=False,
        encoding="utf-8",
    )

    return {
        "output_dir": output_root,
        "symbol": symbol,
        "summary": summary,
        "folds": fold_payloads,
        "rows": flat_rows,
    }


def _aggregate_symbol_summaries(symbol_payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not symbol_payloads:
        return {"symbol_count": 0}

    positive_rates = [
        _safe_float(payload["summary"].get("positive_test_fold_rate"))
        for payload in symbol_payloads
    ]
    positive_rates = [value for value in positive_rates if value is not None]
    avg_test_returns = [
        _safe_float(payload["summary"].get("avg_test_return"))
        for payload in symbol_payloads
    ]
    avg_test_returns = [value for value in avg_test_returns if value is not None]
    avg_paper_returns = [
        _safe_float(payload["summary"].get("avg_paper_return"))
        for payload in symbol_payloads
    ]
    avg_paper_returns = [value for value in avg_paper_returns if value is not None]

    best_symbol = max(
        symbol_payloads,
        key=lambda item: _safe_float(item["summary"].get("avg_test_return")) or float("-inf"),
    )["symbol"]
    worst_symbol = min(
        symbol_payloads,
        key=lambda item: _safe_float(item["summary"].get("avg_test_return")) or float("inf"),
    )["symbol"]

    return {
        "symbol_count": len(symbol_payloads),
        "evaluated_symbols": [payload["symbol"] for payload in symbol_payloads],
        "avg_positive_test_fold_rate": float(pd.Series(positive_rates, dtype=float).mean()) if positive_rates else 0.0,
        "avg_symbol_test_return": float(pd.Series(avg_test_returns, dtype=float).mean()) if avg_test_returns else 0.0,
        "avg_symbol_paper_return": float(pd.Series(avg_paper_returns, dtype=float).mean()) if avg_paper_returns else None,
        "best_symbol": best_symbol,
        "worst_symbol": worst_symbol,
    }


def run_walk_forward(
    config: Dict[str, Any],
    output_dir: Optional[str] = None,
    train_size: int = 84,
    test_size: int = 21,
    step_size: Optional[int] = None,
    include_paper: bool = True,
    symbols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    runtime_config = copy.deepcopy(config or {})
    output_root = output_dir or os.path.join(
        "results",
        "walk_forward",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(output_root, exist_ok=True)
    resolved_symbols = _resolve_symbols(runtime_config, symbols=symbols)

    if len(resolved_symbols) == 1:
        return _run_single_symbol_walk_forward(
            runtime_config,
            symbol=resolved_symbols[0],
            output_root=output_root,
            train_size=train_size,
            test_size=test_size,
            step_size=step_size,
            include_paper=include_paper,
        )

    symbol_payloads: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []
    for symbol in resolved_symbols:
        symbol_output_dir = os.path.join(output_root, "symbols", _safe_symbol(symbol))
        payload = _run_single_symbol_walk_forward(
            runtime_config,
            symbol=symbol,
            output_root=symbol_output_dir,
            train_size=train_size,
            test_size=test_size,
            step_size=step_size,
            include_paper=include_paper,
        )
        symbol_payloads.append(payload)
        aggregate_rows.extend(payload["rows"])

    summary = _aggregate_symbol_summaries(symbol_payloads)
    summary.update(
        {
            "train_size": int(train_size),
            "test_size": int(test_size),
            "step_size": int(step_size or test_size),
            "include_paper": bool(include_paper),
        }
    )
    symbol_summaries = [payload["summary"] for payload in symbol_payloads]
    with open(os.path.join(output_root, "walk_forward_summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(output_root, "walk_forward_symbol_summary.json"), "w", encoding="utf-8") as file:
        json.dump(symbol_summaries, file, ensure_ascii=False, indent=2, default=str)
    pd.DataFrame(symbol_summaries).to_csv(
        os.path.join(output_root, "walk_forward_symbol_summary.csv"),
        index=False,
        encoding="utf-8",
    )
    pd.DataFrame(aggregate_rows).to_csv(
        os.path.join(output_root, "walk_forward_folds.csv"),
        index=False,
        encoding="utf-8",
    )

    return {
        "output_dir": output_root,
        "symbols": resolved_symbols,
        "summary": summary,
        "symbol_summaries": symbol_summaries,
    }
