"""应用服务层：统一封装研究、paper trading、live trading 与结果查询。"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import quote

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from src.agents.ml_strategy_agent import MLStrategyAgent
from src.execution.live_trading_service import LiveTradingService
from src.execution.paper_trading_engine import PaperTradingEngine
from src.execution.portfolio_paper_trading_engine import PortfolioPaperTradingEngine
from src.utils.ai_reviewer import generate_ai_review
from src.utils.config_loader import build_runtime_config, load_config
from src.utils.data_loader import DataLoader
from src.utils.data_processor import DataProcessor


logger = logging.getLogger(__name__)

CHART_FILE_SPECS = {
    "equity_curve.csv": {
        "title": "研究权益曲线",
        "preferred_columns": ["equity"],
        "labels": {"equity": "权益"},
    },
    "paper_account_history.csv": {
        "title": "Paper 账户轨迹",
        "preferred_columns": ["equity", "cash", "drawdown", "realized_pnl", "unrealized_pnl"],
        "labels": {
            "equity": "权益",
            "cash": "现金",
            "drawdown": "回撤",
            "realized_pnl": "已实现盈亏",
            "unrealized_pnl": "浮动盈亏",
        },
    },
    "portfolio_account_history.csv": {
        "title": "组合账户轨迹",
        "preferred_columns": ["equity", "cash", "drawdown", "realized_pnl", "unrealized_pnl"],
        "labels": {
            "equity": "组合权益",
            "cash": "现金",
            "drawdown": "回撤",
            "realized_pnl": "已实现盈亏",
            "unrealized_pnl": "浮动盈亏",
        },
    },
    "live_account_history.csv": {
        "title": "Live 账户轨迹",
        "preferred_columns": ["equity", "cash", "daily_drawdown", "realized_pnl", "unrealized_pnl"],
        "labels": {
            "equity": "权益",
            "cash": "现金",
            "daily_drawdown": "日内回撤",
            "realized_pnl": "已实现盈亏",
            "unrealized_pnl": "浮动盈亏",
        },
    },
}

TEXT_PREVIEW_SUFFIXES = {
    ".txt",
    ".log",
    ".md",
    ".yaml",
    ".yml",
    ".py",
    ".js",
    ".css",
    ".html",
    ".xml",
}

IMAGE_PREVIEW_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_config_path() -> str:
    return os.path.join(get_project_root(), "src", "config", "config.yaml")


def get_results_root() -> str:
    return os.path.join(get_project_root(), "results")


def _project_relative_path(path: str) -> str:
    try:
        path = os.path.relpath(path, get_project_root())
    except ValueError:
        path = os.path.abspath(path)
    return path.replace("\\", "/")


def _resolve_project_path(relative_path: str) -> str:
    project_root = get_project_root()
    normalized = os.path.normpath(os.path.join(project_root, relative_path))
    try:
        common = os.path.commonpath([os.path.normcase(project_root), os.path.normcase(normalized)])
    except ValueError as exc:
        raise ValueError("非法文件路径") from exc
    if common != os.path.normcase(project_root):
        raise ValueError("非法文件路径")
    return normalized


def load_runtime_config() -> Dict[str, Any]:
    return copy.deepcopy(load_config())


def load_config_text() -> str:
    config_path = get_config_path()
    with open(config_path, "r", encoding="utf-8") as file:
        return file.read()


def parse_config_text(text: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("未安装 PyYAML，无法解析配置文本")
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError("配置内容必须是 YAML 字典结构")
    return data


def parse_runtime_config_text(text: str) -> Dict[str, Any]:
    return build_runtime_config(parse_config_text(text))


def render_config_text(config: Dict[str, Any]) -> str:
    if yaml is None:
        raise RuntimeError("未安装 PyYAML，无法序列化配置文本")
    return yaml.safe_dump(config, allow_unicode=True, sort_keys=False)


def save_config_text(text: str) -> Dict[str, Any]:
    config = parse_config_text(text)
    config_path = get_config_path()
    with open(config_path, "w", encoding="utf-8") as file:
        file.write(render_config_text(config))
    return config


def patch_config_text(text: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    config = parse_config_text(text)
    _deep_update(config, patch or {})
    rendered = render_config_text(config)
    return {"config": config, "text": rendered}


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _safe_symbol(symbol: str) -> str:
    return str(symbol).replace("/", "_").replace("\\", "_").replace(":", "_")


def _runtime_fingerprint(config: Dict[str, Any]) -> str:
    tracked = {
        key: config.get(key, {})
        for key in [
            "data",
            "market",
            "strategy",
            "risk",
            "execution_costs",
            "backtest",
            "paper_trading",
            "portfolio",
            "live_trading",
            "live_risk",
        ]
    }
    payload = json.dumps(_sanitize_for_json(tracked), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _build_run_context(run_type: str, config: Dict[str, Any], symbols: List[str]) -> Dict[str, Any]:
    data = config.get("data", {})
    strategy = config.get("strategy", {})
    backtest = config.get("backtest", {})
    llm = config.get("llm", {})
    force_update = bool(data.get("force_update", False))
    return {
        "run_type": run_type,
        "symbols": symbols,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_fingerprint": _runtime_fingerprint(config),
        "data_source": data.get("source", "api"),
        "data_interval": data.get("interval", "d"),
        "start_date": data.get("start_date"),
        "end_date": data.get("end_date"),
        "force_update": force_update,
        "data_mode": "强制刷新行情" if force_update else "本地缓存优先",
        "repeat_note": "同一标的、区间、策略参数和缓存数据重复运行时，回测结果会保持一致。",
        "benchmark_source": backtest.get("benchmark_source", "market"),
        "benchmark_symbol": backtest.get("benchmark_symbol"),
        "benchmark_fallback_to_target": bool(backtest.get("benchmark_fallback_to_target", True)),
        "strategy_name": strategy.get("name", "baseline_trend_following"),
        "allow_short": bool(strategy.get("allow_short", False)),
        "paper_enabled": bool(config.get("paper_trading", {}).get("enabled", False)),
        "portfolio_enabled": bool(config.get("portfolio", {}).get("enabled", False)),
        "llm_enabled": bool(llm.get("enabled", False)),
        "llm_review_timeout": llm.get("review_timeout", llm.get("timeout")),
    }


def _write_run_context(output_dir: str, run_type: str, config: Dict[str, Any], symbols: List[str]) -> Dict[str, Any]:
    context = _build_run_context(run_type, config, symbols)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run_context.json"), "w", encoding="utf-8") as file:
        json.dump(context, file, ensure_ascii=False, indent=2, default=str)
    return context


def _make_output_dir(base_dir: Optional[str] = None) -> str:
    output_root = base_dir or get_results_root()
    output_dir = os.path.join(output_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _build_symbol_config(config: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    symbol_config = copy.deepcopy(config)
    symbol_config.setdefault("data", {})["symbol"] = symbol
    return symbol_config


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover
            return str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover
            return str(value)
    return value


def _collect_artifacts(output_dir: str) -> List[Dict[str, Any]]:
    artifacts: List[Dict[str, Any]] = []
    if not os.path.isdir(output_dir):
        return artifacts

    for name in sorted(os.listdir(output_dir)):
        path = os.path.join(output_dir, name)
        if not os.path.isfile(path):
            continue
        stat = os.stat(path)
        artifacts.append(
            {
                "name": name,
                "relative_path": _project_relative_path(path),
                "size_bytes": int(stat.st_size),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            }
        )
    return artifacts


def _load_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _load_csv_preview_if_exists(path: str, limit: int = 5) -> Optional[List[Dict[str, Any]]]:
    if not os.path.exists(path):
        return None
    import pandas as pd

    frame = pd.read_csv(path)
    if frame.empty:
        return []
    return _sanitize_for_json(frame.head(limit).to_dict(orient="records"))


def _format_chart_x(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover
            return str(value)
    return str(value)


def _select_chart_x(frame) -> List[str]:
    import pandas as pd

    if frame.empty:
        return []

    columns = list(frame.columns)
    for candidate in ["timestamp", "datetime", "date"]:
        if candidate in columns:
            x_values = [_format_chart_x(value) for value in frame[candidate]]
            frame.drop(columns=[candidate], inplace=True)
            return x_values

    first_column = columns[0]
    parsed_first = pd.to_datetime(frame[first_column], errors="coerce")
    if parsed_first.notna().sum() >= max(2, min(len(frame), 3)):
        x_values = [_format_chart_x(value) for value in parsed_first]
        frame.drop(columns=[first_column], inplace=True)
        return x_values

    return [str(index) for index in range(len(frame))]


def _pick_chart_columns(frame, preferred_columns: List[str]) -> List[str]:
    import pandas as pd

    numeric_columns: List[str] = []
    for column in frame.columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any():
            numeric_columns.append(column)

    chosen = [column for column in preferred_columns if column in numeric_columns]
    if chosen:
        return chosen[:4]
    return numeric_columns[:4]


def _build_chart_payload(file_path: str, relative_path: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    import pandas as pd

    frame = pd.read_csv(file_path)
    if frame.empty:
        return {
            "file_name": os.path.basename(file_path),
            "relative_path": relative_path,
            "title": spec["title"],
            "series": [],
            "point_count": 0,
        }

    frame = frame.copy()
    x_values = _select_chart_x(frame)
    labels = spec.get("labels", {})
    chosen_columns = _pick_chart_columns(frame, spec.get("preferred_columns", []))

    series_payload = []
    for column in chosen_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        points = []
        for index, value in enumerate(numeric):
            if pd.isna(value):
                continue
            points.append({"x": x_values[index], "y": float(value)})
        if points:
            series_payload.append(
                {
                    "key": column,
                    "label": labels.get(column, column),
                    "points": points,
                }
            )

    point_count = max((len(item["points"]) for item in series_payload), default=0)
    return {
        "file_name": os.path.basename(file_path),
        "relative_path": relative_path,
        "title": spec["title"],
        "series": series_payload,
        "point_count": point_count,
        "x_start": x_values[0] if x_values else None,
        "x_end": x_values[-1] if x_values else None,
    }


def _read_text_preview(file_path: str, limit: int) -> Dict[str, Any]:
    lines: List[str] = []
    truncated = False
    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        for index, line in enumerate(file):
            if index >= limit:
                truncated = True
                break
            lines.append(line.rstrip("\n"))
    return {
        "kind": "text",
        "text": "\n".join(lines),
        "line_count": len(lines),
        "truncated": truncated,
    }


def _build_run_payload(
    run_type: str,
    output_dir: str,
    symbols: List[str],
    summary: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "run_type": run_type,
        "output_dir": output_dir,
        "relative_output_dir": _project_relative_path(output_dir),
        "symbols": symbols,
        "summary": _sanitize_for_json(summary),
        "artifacts": _collect_artifacts(output_dir),
    }
    if extra:
        payload.update(_sanitize_for_json(extra))
    return payload


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

    benchmark_data = data_loader.get_benchmark_data(reference_data=complete_data, reference_symbol=symbol)
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

    ai_review = generate_ai_review(
        config=symbol_config,
        symbol=symbol,
        performance=performance_metrics,
        risk_metrics=risk_metrics,
        recommendations=recommendations,
        backtest_results=backtest_results,
        strategy_params=strategy_agent.strategy_params,
        llm_client=strategy_agent.llm,
        create_llm_client=False,
    )

    strategy_agent._save_backtest_results(
        backtest_results,
        performance_metrics,
        risk_metrics,
        recommendations,
    )
    with open(os.path.join(output_dir, "ai_review.json"), "w", encoding="utf-8") as file:
        json.dump(ai_review, file, ensure_ascii=False, indent=2, default=str)

    return {
        "symbol": symbol,
        "market_data": complete_data,
        "signals": signals,
        "backtest_results": backtest_results,
        "performance_metrics": performance_metrics,
        "risk_metrics": risk_metrics,
        "recommendations": recommendations,
        "ai_review": ai_review,
    }


def run_main_pipeline(config: Optional[Dict[str, Any]] = None, output_dir: Optional[str] = None) -> Dict[str, Any]:
    runtime_config = copy.deepcopy(config or load_runtime_config())
    run_output_dir = output_dir or _make_output_dir()
    symbols = list(
        dict.fromkeys(
            runtime_config.get("data", {}).get("symbols")
            or [runtime_config.get("data", {}).get("symbol", "aapl.us")]
        )
    )
    paper_enabled = bool(runtime_config.get("paper_trading", {}).get("enabled", False))
    portfolio_enabled = bool(runtime_config.get("portfolio", {}).get("enabled", False))

    if len(symbols) > 1 and portfolio_enabled:
        market_data_by_symbol = {}
        signals_by_symbol = {}
        summary_payload = []
        for symbol in symbols:
            symbol_output_dir = os.path.join(run_output_dir, "symbols", _safe_symbol(symbol))
            result = _run_symbol_research(runtime_config, symbol, symbol_output_dir)
            market_data_by_symbol[symbol] = result["market_data"]
            signals_by_symbol[symbol] = result["signals"]
            summary_payload.append(
                {
                    "symbol": symbol,
                    "performance": result["performance_metrics"],
                    "risk": result["risk_metrics"],
                    "recommendations": result["recommendations"],
                    "ai_review": result.get("ai_review"),
                }
            )

        with open(os.path.join(run_output_dir, "portfolio_research_summary.json"), "w", encoding="utf-8") as file:
            json.dump(summary_payload, file, ensure_ascii=False, indent=2, default=str)

        run_type = "portfolio_research"
        summary = {
            "symbol_count": len(symbols),
            "portfolio_enabled": True,
            "completed_symbols": len(summary_payload),
        }
        extra = {
            "research_summary": _sanitize_for_json(summary_payload),
        }

        if paper_enabled:
            engine = PortfolioPaperTradingEngine(runtime_config)
            paper_results = engine.run(market_data_by_symbol, signals_by_symbol)
            engine.save_results(paper_results, run_output_dir)
            run_type = "portfolio_main"
            summary = paper_results["summary"]
            extra.update(
                {
                    "portfolio_result_preview": _load_json_if_exists(os.path.join(run_output_dir, "portfolio_summary.json")),
                    "symbol_summary": _load_json_if_exists(os.path.join(run_output_dir, "portfolio_symbol_summary.json")),
                }
            )

        run_context = _write_run_context(run_output_dir, run_type, runtime_config, symbols)
        extra["run_context"] = run_context
        return _build_run_payload(
            run_type=run_type,
            output_dir=run_output_dir,
            symbols=symbols,
            summary=summary,
            extra=extra,
        )

    symbol = symbols[0]
    runtime_config.setdefault("data", {})["symbol"] = symbol
    result = _run_symbol_research(runtime_config, symbol, run_output_dir)

    run_type = "research"
    summary = {
        "total_return": result["performance_metrics"]["total_return"],
        "annual_return": result["performance_metrics"]["annual_return"],
        "sharpe_ratio": result["performance_metrics"]["sharpe_ratio"],
        "max_drawdown": result["risk_metrics"]["max_drawdown"],
        "recommendation_count": len(result["recommendations"]),
    }
    extra = {
        "performance_metrics": result["performance_metrics"],
        "risk_metrics": result["risk_metrics"],
        "recommendations": result["recommendations"],
        "ai_review": result.get("ai_review"),
    }

    if paper_enabled:
        engine = PaperTradingEngine(runtime_config)
        paper_results = engine.run(result["market_data"], result["signals"], symbol=symbol)
        engine.save_results(paper_results, run_output_dir)
        run_type = "main"
        summary = paper_results["summary"]
        extra.update(
            {
                "paper_result_preview": _load_json_if_exists(os.path.join(run_output_dir, "paper_summary.json")),
                "account_preview": _load_csv_preview_if_exists(os.path.join(run_output_dir, "paper_account_history.csv")),
            }
        )

    run_context = _write_run_context(run_output_dir, run_type, runtime_config, [symbol])
    extra["run_context"] = run_context
    return _build_run_payload(
        run_type=run_type,
        output_dir=run_output_dir,
        symbols=[symbol],
        summary=summary,
        extra=extra,
    )


def run_paper_pipeline(config: Optional[Dict[str, Any]] = None, output_dir: Optional[str] = None) -> Dict[str, Any]:
    runtime_config = copy.deepcopy(config or load_runtime_config())
    run_output_dir = output_dir or _make_output_dir()
    symbols = list(
        dict.fromkeys(
            runtime_config.get("data", {}).get("symbols")
            or [runtime_config.get("data", {}).get("symbol", "aapl.us")]
        )
    )

    if len(symbols) > 1 and runtime_config.get("portfolio", {}).get("enabled", False):
        market_data_by_symbol = {}
        signals_by_symbol = {}
        for symbol in symbols:
            symbol_output_dir = os.path.join(run_output_dir, "symbols", _safe_symbol(symbol))
            result = _run_symbol_research(runtime_config, symbol, symbol_output_dir)
            market_data_by_symbol[symbol] = result["market_data"]
            signals_by_symbol[symbol] = result["signals"]

        engine = PortfolioPaperTradingEngine(runtime_config)
        results = engine.run(market_data_by_symbol, signals_by_symbol)
        engine.save_results(results, run_output_dir)
        run_context = _write_run_context(run_output_dir, "portfolio_paper", runtime_config, symbols)
        return _build_run_payload(
            run_type="portfolio_paper",
            output_dir=run_output_dir,
            symbols=symbols,
            summary=results["summary"],
            extra={
                "result_preview": _load_json_if_exists(os.path.join(run_output_dir, "portfolio_summary.json")),
                "symbol_summary": _load_json_if_exists(os.path.join(run_output_dir, "portfolio_symbol_summary.json")),
                "run_context": run_context,
            },
        )

    symbol = symbols[0]
    runtime_config.setdefault("data", {})["symbol"] = symbol
    result = _run_symbol_research(runtime_config, symbol, run_output_dir)
    engine = PaperTradingEngine(runtime_config)
    paper_results = engine.run(result["market_data"], result["signals"], symbol=symbol)
    engine.save_results(paper_results, run_output_dir)
    run_context = _write_run_context(run_output_dir, "paper", runtime_config, [symbol])
    return _build_run_payload(
        run_type="paper",
        output_dir=run_output_dir,
        symbols=[symbol],
        summary=paper_results["summary"],
        extra={
            "result_preview": _load_json_if_exists(os.path.join(run_output_dir, "paper_summary.json")),
            "account_preview": _load_csv_preview_if_exists(os.path.join(run_output_dir, "paper_account_history.csv")),
            "run_context": run_context,
        },
    )


def run_live_pipeline(
    config: Optional[Dict[str, Any]] = None,
    results_root: Optional[str] = None,
    force_update: Optional[bool] = None,
    save_results: bool = True,
) -> Dict[str, Any]:
    runtime_config = copy.deepcopy(config or load_runtime_config())
    runtime_config.setdefault("live_service", {})
    runtime_config["live_service"]["results_root"] = results_root or os.path.join(get_results_root(), "live_service")
    runtime_config["live_service"]["max_cycles"] = 1
    service = LiveTradingService(runtime_config)
    payload = service.run_once(force_update=force_update, save_results=save_results)

    live_output_dir = payload.get("results_dir") or runtime_config["live_service"]["results_root"]
    live_symbols = payload.get("symbols") or [payload.get("symbol", runtime_config.get("data", {}).get("symbol", "unknown"))]
    run_context = _write_run_context(live_output_dir, "live", runtime_config, live_symbols)
    return _build_run_payload(
        run_type="live",
        output_dir=live_output_dir,
        symbols=live_symbols,
        summary=payload["summary"],
        extra={
            "latest_timestamp": payload.get("latest_timestamp"),
            "processed_rows": payload.get("processed_rows"),
            "results_dir": payload.get("results_dir"),
            "run_context": run_context,
        },
    )


def _detect_run_type_from_files(file_names: List[str]) -> str:
    if "live_summary.json" in file_names:
        return "live"
    if "portfolio_summary.json" in file_names:
        return "portfolio_paper"
    if "paper_summary.json" in file_names:
        return "paper"
    if "metrics.json" in file_names or "portfolio_research_summary.json" in file_names:
        return "research"
    return "unknown"


def _find_result_directories(root_dir: str) -> List[str]:
    directories: List[str] = []
    if not os.path.isdir(root_dir):
        return directories

    for current_root, dir_names, file_names in os.walk(root_dir):
        if any(name.endswith("_summary.json") or name in {"metrics.json", "portfolio_research_summary.json"} for name in file_names):
            directories.append(current_root)
            dir_names[:] = []
    return directories


def get_run_chart_data(relative_path: str) -> Dict[str, Any]:
    normalized = _resolve_project_path(relative_path)
    if not os.path.isdir(normalized):
        raise FileNotFoundError("结果目录不存在")

    charts = []
    for file_name, spec in CHART_FILE_SPECS.items():
        file_path = os.path.join(normalized, file_name)
        if not os.path.exists(file_path):
            continue
        relative_file_path = _project_relative_path(file_path)
        charts.append(_build_chart_payload(file_path, relative_file_path, spec))

    return {
        "relative_output_dir": _project_relative_path(normalized),
        "charts": charts,
    }


def get_file_preview(relative_path: str, limit: int = 50) -> Dict[str, Any]:
    normalized = _resolve_project_path(relative_path)
    if not os.path.isfile(normalized):
        raise FileNotFoundError("文件不存在")

    relative_file_path = _project_relative_path(normalized)
    suffix = os.path.splitext(normalized)[1].lower()
    payload = {
        "path": relative_file_path,
        "name": os.path.basename(normalized),
        "size_bytes": int(os.path.getsize(normalized)),
    }

    if suffix == ".json":
        with open(normalized, "r", encoding="utf-8") as file:
            content = json.load(file)
        payload.update(
            {
                "kind": "json",
                "content": _sanitize_for_json(content),
                "text": json.dumps(content, ensure_ascii=False, indent=2, default=str),
            }
        )
        return payload

    if suffix == ".csv":
        import pandas as pd

        frame = pd.read_csv(normalized)
        preview = frame.head(limit)
        payload.update(
            {
                "kind": "table",
                "columns": [str(column) for column in frame.columns],
                "rows": _sanitize_for_json(preview.to_dict(orient="records")),
                "row_count": int(len(frame)),
                "truncated": len(frame) > limit,
            }
        )
        return payload

    if suffix in IMAGE_PREVIEW_SUFFIXES:
        payload.update(
            {
                "kind": "image",
                "file_url": f"/files/{quote(relative_file_path)}",
            }
        )
        return payload

    if suffix in TEXT_PREVIEW_SUFFIXES:
        payload.update(_read_text_preview(normalized, limit))
        return payload

    payload.update(
        {
            "kind": "binary",
            "message": "当前文件类型不支持直接预览，可点击右侧链接单独打开。",
        }
    )
    return payload


def get_run_details(relative_path: str) -> Dict[str, Any]:
    project_root = get_project_root()
    normalized = _resolve_project_path(relative_path)
    if not os.path.isdir(normalized):
        raise FileNotFoundError("结果目录不存在")

    file_names = sorted(
        [name for name in os.listdir(normalized) if os.path.isfile(os.path.join(normalized, name))]
    )
    details = {
        "run_type": _detect_run_type_from_files(file_names),
        "output_dir": normalized,
        "relative_output_dir": _project_relative_path(normalized),
        "artifacts": _collect_artifacts(normalized),
        "summary": None,
        "metrics": _load_json_if_exists(os.path.join(normalized, "metrics.json")),
        "ai_review": _load_json_if_exists(os.path.join(normalized, "ai_review.json")),
        "run_context": _load_json_if_exists(os.path.join(normalized, "run_context.json")),
        "paper_summary": _load_json_if_exists(os.path.join(normalized, "paper_summary.json")),
        "portfolio_summary": _load_json_if_exists(os.path.join(normalized, "portfolio_summary.json")),
        "live_summary": _load_json_if_exists(os.path.join(normalized, "live_summary.json")),
        "portfolio_research_summary": _load_json_if_exists(os.path.join(normalized, "portfolio_research_summary.json")),
    }

    for key in ["live_summary", "portfolio_summary", "paper_summary", "metrics"]:
        if details.get(key):
            details["summary"] = details[key]
            break
    return _sanitize_for_json(details)


def list_result_runs(limit: int = 20) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for directory in _find_result_directories(get_results_root()):
        stat = os.stat(directory)
        relative_output_dir = _project_relative_path(directory)
        details = get_run_details(relative_output_dir)
        runs.append(
            {
                "run_type": details["run_type"],
                "relative_output_dir": relative_output_dir,
                "output_dir": directory,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "summary": details.get("summary"),
                "artifact_count": len(details.get("artifacts", [])),
                "ai_review": details.get("ai_review"),
                "metrics": details.get("metrics"),
                "paper_summary": details.get("paper_summary"),
                "run_context": details.get("run_context"),
            }
        )

    runs.sort(key=lambda item: item["modified_at"], reverse=True)
    return runs[: max(int(limit), 1)]
