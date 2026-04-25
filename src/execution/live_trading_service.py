"""Live trading 轮询服务。"""

import copy
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

from src.agents.ml_strategy_agent import MLStrategyAgent
from src.utils.data_loader import DataLoader
from src.utils.data_processor import DataProcessor

from .portfolio_live_trading_engine import PortfolioLiveTradingEngine
from .live_trading_engine import LiveTradingEngine
from .observability import EventBus, JsonlEventJournal, RuleBasedAlertSink, build_event
from .reconciliation import build_operator_report, build_reconciliation_report


DEFAULT_LIVE_SERVICE_CONFIG = {
    "poll_interval_seconds": 60,
    "force_update_each_cycle": True,
    "save_results_each_cycle": True,
    "max_cycles": 1,
    "stop_on_error": True,
    "results_root": "results/live_service",
}

DEFAULT_MONITORING_CONFIG = {
    "enabled": True,
    "max_recent_events": 2000,
    "event_journal_filename": "live_events.jsonl",
    "alert_log_filename": "live_alerts.jsonl",
    "drawdown_alert_threshold": 0.02,
    "reconciliation_report_filename": "live_reconciliation.json",
    "operator_report_filename": "live_operator_report.json",
}


class LiveTradingService:
    def __init__(
        self,
        config: Dict[str, Any],
        data_loader: Optional[Any] = None,
        data_processor: Optional[DataProcessor] = None,
        strategy_agent: Optional[Any] = None,
        engine: Optional[Any] = None,
        event_bus: Optional[EventBus] = None,
        sleep_fn=None,
    ):
        self.config = config or {}
        self.live_service_config = dict(DEFAULT_LIVE_SERVICE_CONFIG)
        self.live_service_config.update(self.config.get("live_service", {}))
        self.monitoring_config = dict(DEFAULT_MONITORING_CONFIG)
        self.monitoring_config.update(self.config.get("monitoring", {}))
        self.symbols = list(
            dict.fromkeys(
                self.config.get("data", {}).get("symbols")
                or [self.config.get("data", {}).get("symbol", "aapl.us")]
            )
        )
        self.symbol = self.symbols[0]
        self.multi_symbol_mode = len(self.symbols) > 1 and bool(self.config.get("portfolio", {}).get("enabled", False))
        self.symbol_config = self._build_symbol_config(self.config, self.symbol)

        self.logger = logging.getLogger(__name__)
        self.data_processor = data_processor or DataProcessor()
        self.sleep_fn = sleep_fn or time.sleep

        self.results_dir = os.path.join(
            self.live_service_config.get("results_root", "results/live_service"),
            datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        self.event_bus = event_bus or EventBus(max_recent_events=int(self.monitoring_config.get("max_recent_events", 2000)))
        self._configure_observability()
        if self.multi_symbol_mode:
            self.data_loader = None
            if isinstance(data_loader, dict):
                self.data_loaders = dict(data_loader)
            elif data_loader is not None:
                self.data_loaders = {symbol: data_loader for symbol in self.symbols}
            else:
                self.data_loaders = {
                    symbol: DataLoader(self._build_symbol_config(self.config, symbol))
                    for symbol in self.symbols
                }
            if isinstance(strategy_agent, dict):
                self.strategy_agents = dict(strategy_agent)
            elif strategy_agent is not None:
                self.strategy_agents = {symbol: strategy_agent for symbol in self.symbols}
            else:
                self.strategy_agents = {
                    symbol: MLStrategyAgent(
                        self._build_symbol_config(self.config, symbol),
                        self.data_loaders[symbol],
                    )
                    for symbol in self.symbols
                }
        else:
            self.data_loader = data_loader or DataLoader(self.symbol_config)
            self.strategy_agent = strategy_agent or MLStrategyAgent(self.symbol_config, self.data_loader)
        if engine is not None:
            self.engine = engine
        elif self.multi_symbol_mode:
            self.engine = PortfolioLiveTradingEngine(self.config, event_bus=self.event_bus)
        else:
            self.engine = LiveTradingEngine(self.symbol_config, event_bus=self.event_bus)
        if getattr(self.engine, "event_bus", None) is None:
            self.engine.event_bus = self.event_bus
        self.last_results = None

    def _configure_observability(self):
        if not bool(self.monitoring_config.get("enabled", True)):
            return
        event_path = os.path.join(
            self.results_dir,
            str(self.monitoring_config.get("event_journal_filename", "live_events.jsonl")),
        )
        alert_path = os.path.join(
            self.results_dir,
            str(self.monitoring_config.get("alert_log_filename", "live_alerts.jsonl")),
        )
        self.event_bus.subscribe(JsonlEventJournal(event_path))
        self.event_bus.subscribe(
            RuleBasedAlertSink(
                alert_path,
                drawdown_alert_threshold=float(self.monitoring_config.get("drawdown_alert_threshold", 0.02)),
            )
        )

    def _build_symbol_config(self, config: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        symbol_config = copy.deepcopy(config)
        symbol_config.setdefault("data", {})["symbol"] = symbol
        return symbol_config

    def _load_symbol_market_snapshot(self, symbol: str, force_update: Optional[bool] = None):
        symbol_config = self._build_symbol_config(self.config, symbol)
        data_config = symbol_config.get("data", {})
        interval = data_config.get("interval", "d")
        loader = self.data_loaders[symbol] if self.multi_symbol_mode else self.data_loader
        market_data = loader.load_data(
            symbol=symbol,
            interval=interval,
            force_update=force_update,
        )
        if market_data is None or market_data.empty:
            raise RuntimeError(f"未获取到 {symbol} 的有效市场数据")

        min_periods = max(50, int(symbol_config.get("strategy", {}).get("slow_ma", 20)))
        complete_data = self.data_processor.get_complete_data(market_data, min_periods=min_periods)
        if complete_data is None or complete_data.empty:
            raise RuntimeError(f"{symbol} 的技术指标数据为空")
        return complete_data

    def _load_market_snapshot(self, force_update: Optional[bool] = None):
        if not self.multi_symbol_mode:
            return self._load_symbol_market_snapshot(self.symbol, force_update=force_update)
        return {
            symbol: self._load_symbol_market_snapshot(symbol, force_update=force_update)
            for symbol in self.symbols
        }

    def run_once(self, force_update: Optional[bool] = None, save_results: Optional[bool] = None) -> Dict[str, Any]:
        force_update = self.live_service_config.get("force_update_each_cycle", True) if force_update is None else force_update
        save_results = self.live_service_config.get("save_results_each_cycle", True) if save_results is None else save_results

        previous_count = self.engine.processed_count
        self.event_bus.publish(
            build_event(
                "service_cycle_started",
                source="live_service",
                timestamp=datetime.now(),
                symbol=None if self.multi_symbol_mode else self.symbol,
                payload={
                    "force_update": bool(force_update),
                    "save_results": bool(save_results),
                    "symbol_count": len(self.symbols),
                },
            )
        )
        market_data = self._load_market_snapshot(force_update=force_update)
        if self.multi_symbol_mode:
            latest_timestamp = max(frame.index[-1] for frame in market_data.values())
            market_rows = {symbol: int(len(frame)) for symbol, frame in market_data.items()}
            self.event_bus.publish(
                build_event(
                    "market_snapshot_loaded",
                    source="live_service",
                    timestamp=latest_timestamp,
                    symbol=None,
                    payload={"rows_by_symbol": market_rows, "symbol_count": len(market_data)},
                )
            )
            signals = {
                symbol: self.strategy_agents[symbol].generate_signals(frame)
                for symbol, frame in market_data.items()
            }
            self.event_bus.publish(
                build_event(
                    "signals_generated",
                    source="live_service",
                    timestamp=latest_timestamp,
                    symbol=None,
                    payload={
                        "rows_by_symbol": {symbol: int(len(frame)) for symbol, frame in signals.items()},
                        "non_flat_signals_by_symbol": {
                            symbol: int((frame["signal"] != 0).sum()) if "signal" in frame.columns else 0
                            for symbol, frame in signals.items()
                        },
                    },
                )
            )
            results = self.engine.run_incremental(market_data, signals)
        else:
            latest_timestamp = market_data.index[-1]
            self.event_bus.publish(
                build_event(
                    "market_snapshot_loaded",
                    source="live_service",
                    timestamp=latest_timestamp,
                    symbol=self.symbol,
                    payload={"rows": int(len(market_data)), "start": str(market_data.index[0]), "end": str(market_data.index[-1])},
                )
            )
            signals = self.strategy_agent.generate_signals(market_data)
            self.event_bus.publish(
                build_event(
                    "signals_generated",
                    source="live_service",
                    timestamp=latest_timestamp,
                    symbol=self.symbol,
                    payload={
                        "rows": int(len(signals)),
                        "non_flat_signals": int((signals["signal"] != 0).sum()) if "signal" in signals.columns else 0,
                    },
                )
            )
            results = self.engine.run_incremental(market_data, signals, symbol=self.symbol)
        processed_rows = self.engine.processed_count - previous_count

        if save_results:
            os.makedirs(self.results_dir, exist_ok=True)
            self.engine.save_results(results, self.results_dir)

        payload = {
            "symbol": self.symbol,
            "symbols": list(self.symbols),
            "latest_timestamp": latest_timestamp,
            "processed_rows": int(processed_rows),
            "results_dir": self.results_dir if save_results else None,
            "summary": results["summary"],
            "results": results,
            "recent_events": self.event_bus.snapshot(limit=50),
        }
        reconciliation = build_reconciliation_report(results, self.engine.adapter, latest_timestamp)
        payload["reconciliation"] = reconciliation
        if reconciliation.get("status") != "ok":
            self.event_bus.publish(
                build_event(
                    "reconciliation_mismatch",
                    source="live_service",
                    timestamp=latest_timestamp,
                    symbol=None if self.multi_symbol_mode else self.symbol,
                    severity="critical",
                    payload={"mismatches": reconciliation.get("mismatches", [])},
                )
            )
        payload["operator_report"] = build_operator_report(payload, reconciliation=reconciliation)
        self.event_bus.publish(
            build_event(
                "service_cycle_completed",
                source="live_service",
                timestamp=latest_timestamp,
                symbol=None if self.multi_symbol_mode else self.symbol,
                payload={
                    "processed_rows": int(processed_rows),
                    "results_dir": self.results_dir if save_results else None,
                    "summary": dict(results["summary"]),
                    "symbol_count": len(self.symbols),
                },
            )
        )
        if save_results:
            with open(
                os.path.join(
                    self.results_dir,
                    str(self.monitoring_config.get("reconciliation_report_filename", "live_reconciliation.json")),
                ),
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(reconciliation, file, ensure_ascii=False, indent=2, default=str)
            with open(
                os.path.join(
                    self.results_dir,
                    str(self.monitoring_config.get("operator_report_filename", "live_operator_report.json")),
                ),
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(payload["operator_report"], file, ensure_ascii=False, indent=2, default=str)
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
                self.event_bus.publish(
                    build_event(
                        "service_cycle_failed",
                        source="live_service",
                        timestamp=datetime.now(),
                        symbol=None if self.multi_symbol_mode else self.symbol,
                        severity="critical",
                        message=str(exc),
                    )
                )
                self.logger.exception(f"live service 第 {cycle} 个周期失败: {exc}")
                if stop_on_error:
                    raise
            if max_cycles not in (None, 0) and cycle >= int(max_cycles):
                break
            if sleep_seconds and sleep_seconds > 0:
                self.sleep_fn(float(sleep_seconds))
        return last_payload
