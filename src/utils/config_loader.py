import copy
import logging
import os
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


logger = logging.getLogger(__name__)


DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "source": "api",
        "symbol": "aapl.us",
        "symbols": ["aapl.us"],
        "interval": "d",
        "timeframe": "1d",
        "start_date": "2024-01-01",
        "end_date": "2025-01-01",
        "force_update": False,
        "path": "",
        "paths": {},
        "storage": {
            "base_dir": "data/market_data",
            "raw_data_dir": "raw",
            "processed_data_dir": "processed",
        },
    },
    "market": {
        "profile": "auto",
        "symbol_profiles": {},
        "symbol_overrides": {},
    },
    "strategy": {
        "name": "baseline_trend_following",
        "fast_ma": 10,
        "slow_ma": 20,
        "rsi_period": 14,
        "rsi_long_threshold": 55,
        "rsi_short_threshold": 45,
        "rsi_exit_long": 48,
        "rsi_exit_short": 52,
        "min_volume_ratio": 1.0,
        "allow_short": True,
    },
    "risk": {
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.06,
        "max_drawdown": 0.2,
        "daily_loss_limit": 0.05,
        "position_size": 1.0,
        "commission": 0.001,
        "slippage": 0.0005,
    },
    "execution_costs": {
        "fixed_commission": 0.0,
        "min_commission": 0.0,
        "sell_tax_rate": 0.0,
    },
    "backtest": {
        "initial_capital": 100000.0,
        "benchmark_source": "market",
        "benchmark_symbol": "^ndx",
        "benchmark_enabled": True,
        "benchmark_fallback_to_target": True,
        "benchmark_yfinance_fallback": False,
    },
    "paper_trading": {
        "enabled": False,
        "initial_cash": 100000.0,
        "allocation_pct": 0.95,
        "allow_fractional": False,
        "allow_short": True,
        "lot_size": 1.0,
        "close_positions_on_finish": True,
        "price_field": "close",
        "quantity_precision": 6,
        "rebalance_frequency": "daily",
        "rebalance_weekday": 0,
        "rebalance_day_of_month": 1,
        "turnover_buffer": 0.0,
        "max_account_drawdown": 0.2,
        "adapter": "paper",
    },
    "portfolio": {
        "enabled": False,
        "target_gross_allocation": 0.95,
        "max_positions": 3,
        "max_gross_exposure": 1.0,
        "max_symbol_allocation": 0.35,
        "max_portfolio_drawdown": 0.2,
        "close_positions_on_finish": True,
        "price_field": "close",
        "selection_metric": "market_strength",
        "rebalance_frequency": "daily",
        "rebalance_weekday": 0,
        "rebalance_day_of_month": 1,
        "turnover_buffer": 0.0,
        "adapter": "paper",
    },
    "live_trading": {
        "enabled": False,
        "initial_cash": 100000.0,
        "allocation_pct": 0.95,
        "allow_fractional": False,
        "allow_short": True,
        "lot_size": 1.0,
        "quantity_precision": 6,
        "price_field": "close",
        "adapter": "paper_live",
        "timezone": "Asia/Shanghai",
        "trading_days": [0, 1, 2, 3, 4],
        "sessions": [{"start": "09:30", "end": "15:00"}],
        "session_start": "09:30",
        "session_end": "15:00",
        "exit_only_start": "14:55",
        "cancel_after_seconds": 300,
        "max_order_retries": 2,
        "flatten_outside_trading_hours": False,
        "min_signal_strength": 0.0,
        "fill_delay_seconds": 0,
        "reject_first_n_orders": 0,
        "close_positions_on_finish": False,
    },
    "live_risk": {
        "max_order_notional": 0.0,
        "max_position_notional": 0.0,
        "max_daily_drawdown": 0.0,
        "max_open_orders": 10,
        "max_orders_per_day": 0,
        "max_consecutive_failures": 0,
    },
    "broker": {
        "provider": "generic_rest",
        "paper": True,
        "base_url": "",
        "account_id": "",
        "timeout_seconds": 10.0,
    },
    "live_service": {
        "poll_interval_seconds": 60,
        "force_update_each_cycle": True,
        "save_results_each_cycle": True,
        "max_cycles": 1,
        "stop_on_error": True,
        "results_root": "results/live_service",
    },
    "llm": {
        "enabled": False,
        "provider": "deepseek",
        "model": "deepseek-chat",
        "temperature": 0.2,
        "max_tokens": 1200,
        "timeout": 120.0,
        "max_retries": 0,
        "review_max_tokens": 1200,
        "review_timeout": 120.0,
        "base_url": "https://api.deepseek.com/v1",
    },
    "api_keys": {},
    "model_type": "deepseek",
}



def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged



def _load_yaml_file(path: str) -> Dict[str, Any]:
    if yaml is None:
        logger.warning("PyYAML 未安装，无法读取配置文件，改用默认配置。")
        return {}

    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"配置文件格式错误: {path}")

    return data



def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _config_path() -> str:
    return os.path.join(_project_root(), "src", "config", "config.yaml")


def _api_keys_path() -> str:
    return os.path.join(_project_root(), "src", "config", "api_keys.yaml")


def _apply_runtime_secrets(config: Dict[str, Any]) -> Dict[str, Any]:
    llm_section = config.setdefault("llm", {})
    model_type = llm_section.get("provider") or config.get("model_type", "deepseek")
    config["model_type"] = model_type

    api_keys = config.setdefault("api_keys", {})
    if not isinstance(api_keys, dict):
        api_keys = {}
        config["api_keys"] = api_keys

    provider_config = api_keys.get(model_type, {})
    if not isinstance(provider_config, dict):
        provider_config = {}

    env_api_key = os.getenv(f"{model_type.upper()}_API_KEY")
    if env_api_key:
        provider_config["api_key"] = env_api_key
        api_keys[model_type] = provider_config

    llm_enabled = bool(llm_section.get("enabled", False))
    has_api_key = bool(api_keys.get(model_type, {}).get("api_key"))
    if llm_enabled and not has_api_key:
        logger.warning("LLM 已启用但未找到 API key，自动降级为离线模式。")
        config["llm"]["enabled"] = False

    return config


def build_runtime_config(config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """合成运行时配置。

    Web 页面只提交 config.yaml 文本，不会也不应该把 api_keys.yaml 发到浏览器。
    任务启动前必须在后端重新合并本地密钥和环境变量。
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_overrides:
        config = _deep_merge(config, config_overrides)

    api_keys = _load_yaml_file(_api_keys_path())
    if api_keys:
        existing_api_keys = config.get("api_keys", {})
        if not isinstance(existing_api_keys, dict):
            existing_api_keys = {}
        config["api_keys"] = _deep_merge(existing_api_keys, api_keys)

    return _apply_runtime_secrets(config)



def load_config() -> Dict[str, Any]:
    """加载配置文件。

    1. 主配置文件可覆盖默认值。
    2. api_keys.yaml 为可选文件，不存在时自动回退。
    3. 环境变量优先级最高。
    """
    try:
        return build_runtime_config(_load_yaml_file(_config_path()))
    except Exception as exc:  # pragma: no cover
        logger.error(f"加载配置失败，使用默认配置: {exc}")
        return copy.deepcopy(DEFAULT_CONFIG)
