"""Live trading 轮询服务。"""

import copy
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

from src.agents.ml_strategy_agent import MLStrategyAgent
from src.utils.data_loader import DataLoader
from src.utils.data_processor import DataProcessor

from .live_trading_engine import LiveTradingEngine


DEFAULT_LIVE_SERVICE_CONFIG = {
    "poll_interval_seconds": 60,
    "force_update_each_cycle": True,
    "save_results_each_cycle": True,
    "max_cycles": 1,
    "stop_on_error": True,
    "results_root": "results/live_service",
}


class LiveTradingService:
    def __init__(
        self,
        config: Dict[str, Any],
        data_loader: Optional[DataLoader] = None,
        data_processor: Optional[DataProcessor] = None,
        strategy_agent: Optional[MLStrategyAgent] = None,
        engine: Optional[LiveTradingEngine] = None,
        sleep_fn=None,
    ):
        self.config = config or {}
        self.live_service_config = dict(DEFAULT_LIVE_SERVICE_CONFIG)
        self.live_service_config.update(self.config.get("live_service", {}))
        self.symbol = self.config.get("data", {}).get("symbol", "aapl.us")
        self.symbol_config = self._build_symbol_config(self.config, self.symbol)

        self.logger = logging.getLogger(__name__)
        self.data_loader = data_loader or DataLoader(self.symbol_config)
        self.data_processor = data_processor or DataProcessor()
        self.strategy_agent = strategy_agent or MLStrategyAgent(self.symbol_config, self.data_loader)
        self.engine = engine or LiveTradingEngine(self.symbol_config)
        self.sleep_fn = sleep_fn or time.sleep

        self.results_dir = os.path.join(
            self.live_service_config.get("results_root", "results/live_service"),
            datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        self.last_results = None

    def _build_symbol_config(self, config: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        symbol_config = copy.deepcopy(config)
        symbol_config.setdefault("data", {})["symbol"] = symbol
        return symbol_config

    def _load_market_snapshot(self, force_update: Optional[bool] = None):
        data_config = self.symbol_config.get("data", {})
        interval = data_config.get("interval", "d")
        market_data = self.data_loader.load_data(
            symbol=self.symbol,
            interval=interval,
            force_update=force_update,
        )
        if market_data is None or market_data.empty:
            raise RuntimeError(f"未获取到 {self.symbol} 的有效市场数据")

        min_periods = max(50, int(self.symbol_config.get("strategy", {}).get("slow_ma", 20)))
        complete_data = self.data_processor.get_complete_data(market_data, min_periods=min_periods)
        if complete_data is None or complete_data.empty:
            raise RuntimeError(f"{self.symbol} 的技术指标数据为空")
        return complete_data

    def run_once(self, force_update: Optional[bool] = None, save_results: Optional[bool] = None) -> Dict[str, Any]:
        force_update = self.live_service_config.get("force_update_each_cycle", True) if force_update is None else force_update
        save_results = self.live_service_config.get("save_results_each_cycle", True) if save_results is None else save_results

        previous_count = self.engine.processed_count
        market_data = self._load_market_snapshot(force_update=force_update)
        signals = self.strategy_agent.generate_signals(market_data)
        results = self.engine.run_incremental(market_data, signals, symbol=self.symbol)
        processed_rows = self.engine.processed_count - previous_count

        if save_results:
            os.makedirs(self.results_dir, exist_ok=True)
            self.engine.save_results(results, self.results_dir)

        latest_timestamp = market_data.index[-1]
        payload = {
            "symbol": self.symbol,
            "latest_timestamp": latest_timestamp,
            "processed_rows": int(processed_rows),
            "results_dir": self.results_dir if save_results else None,
            "summary": results["summary"],
            "results": results,
        }
        self.last_results = payload
        return payload

    def run_forever(self, max_cycles: Optional[int] = None, sleep_seconds: Optional[float] = None):
        max_cycles = self.live_service_config.get("max_cycles", 1) if max_cycles is None else max_cycles
        sleep_seconds = self.live_service_config.get("poll_interval_seconds", 60) if sleep_seconds is None else sleep_seconds
        stop_on_error = bool(self.live_service_config.get("stop_on_error", True))

        cycle = 0
        last_payload = None
        while max_cycles in (None, 0) or cycle < int(max_cycles):
            cycle += 1
            try:
                last_payload = self.run_once()
            except Exception as exc:
                self.logger.exception(f"live service 第 {cycle} 个周期失败: {exc}")
                if stop_on_error:
                    raise
            if max_cycles not in (None, 0) and cycle >= int(max_cycles):
                break
            if sleep_seconds and sleep_seconds > 0:
                self.sleep_fn(float(sleep_seconds))
        return last_payload
