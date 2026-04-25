"""Multi-symbol live dry-run execution skeleton."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .broker_adapter import BrokerAdapter, PaperLiveBrokerAdapter, build_live_adapter
from .live_broker import PaperLiveBroker
from .live_risk_manager import LiveRiskManager
from .live_trading_engine import DEFAULT_LIVE_TRADING_CONFIG
from .market_profile import MarketProfileResolver
from .market_session import MarketSession
from .observability import EventBus, build_event
from .trading_calendar import TradingCalendar


DEFAULT_PORTFOLIO_LIVE_CONFIG = {
    "enabled": False,
    "target_gross_allocation": 0.95,
    "max_positions": 3,
    "max_gross_exposure": 1.0,
    "max_symbol_allocation": 0.35,
    "max_portfolio_drawdown": 0.2,
    "price_field": "close",
    "selection_metric": "market_strength",
    "rebalance_frequency": "daily",
    "rebalance_weekday": 0,
    "rebalance_day_of_month": 1,
    "turnover_buffer": 0.0,
}


class PortfolioLiveTradingEngine:
    def __init__(
        self,
        config: Dict[str, Any],
        adapter: Optional[BrokerAdapter] = None,
        broker: Optional[PaperLiveBroker] = None,
        risk_manager: Optional[LiveRiskManager] = None,
        event_bus: Optional[EventBus] = None,
    ):
        self.config = config or {}
        self.market_resolver = MarketProfileResolver(self.config)
        self.symbols = list(
            dict.fromkeys(
                self.config.get("data", {}).get("symbols")
                or [self.config.get("data", {}).get("symbol", "aapl.us")]
            )
        )
        self.portfolio_config = dict(DEFAULT_PORTFOLIO_LIVE_CONFIG)
        self.portfolio_config.update(self.config.get("portfolio", {}))
        self.live_base_config = dict(DEFAULT_LIVE_TRADING_CONFIG)
        self.live_base_config.update(self.config.get("live_trading", {}))

        self._owned_adapter = adapter is None and broker is None
        if adapter is not None:
            self.adapter = adapter
        elif broker is not None:
            self.adapter = PaperLiveBrokerAdapter(broker)
        else:
            self.adapter = build_live_adapter(self.config)

        self.risk_manager = risk_manager or LiveRiskManager(self.config.get("live_risk", {}))
        self.event_bus = event_bus
        self.calendar = TradingCalendar(
            rebalance_frequency=self.portfolio_config.get("rebalance_frequency", "daily"),
            rebalance_weekday=int(self.portfolio_config.get("rebalance_weekday", 0)),
            rebalance_day_of_month=int(self.portfolio_config.get("rebalance_day_of_month", 1)),
        )
        self._sessions: Dict[str, MarketSession] = {}
        self._peak_equity = float(self.adapter.initial_cash)
        self._failed_attempts: Dict[str, int] = {}
        self._account_rows: List[Dict[str, Any]] = []
        self._symbol_rows: List[Dict[str, Any]] = []
        self._last_processed_at = None
        self._last_rebalance_at = None
        self._paused = False

    @property
    def processed_count(self) -> int:
        return len(self._account_rows)

    def reset_state(self):
        if self._owned_adapter:
            self.adapter = build_live_adapter(self.config)
        self.risk_manager = LiveRiskManager(self.config.get("live_risk", {}))
        self._sessions = {}
        self._peak_equity = float(self.adapter.initial_cash)
        self._failed_attempts = {}
        self._account_rows = []
        self._symbol_rows = []
        self._last_processed_at = None
        self._last_rebalance_at = None
        self._paused = False

    def _emit_event(
        self,
        event_type: str,
        timestamp: Any,
        severity: str = "info",
        symbol: Optional[str] = None,
        message: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ):
        if self.event_bus is None:
            return
        self.event_bus.publish(
            build_event(
                event_type=event_type,
                source="portfolio_live_engine",
                timestamp=timestamp,
                severity=severity,
                symbol=symbol,
                message=message,
                payload=payload,
            )
        )

    def _live_config_for(self, symbol: str) -> Dict[str, Any]:
        config = dict(self.live_base_config)
        config.update(self.market_resolver.section_settings(symbol=symbol, section_name="live_trading"))
        return config

    def _session_for(self, symbol: str) -> MarketSession:
        if symbol not in self._sessions:
            live_config = self._live_config_for(symbol)
            self._sessions[symbol] = MarketSession(
                timezone=live_config.get("timezone", "Asia/Shanghai"),
                trading_days=live_config.get("trading_days", [0, 1, 2, 3, 4]),
                session_start=live_config.get("session_start", "09:30"),
                session_end=live_config.get("session_end", "15:00"),
                exit_only_start=live_config.get("exit_only_start", "14:55"),
                sessions=live_config.get("sessions"),
            )
        return self._sessions[symbol]

    def _common_index(
        self,
        market_data_by_symbol: Dict[str, pd.DataFrame],
        signals_by_symbol: Dict[str, pd.DataFrame],
    ) -> pd.Index:
        common_index = None
        for symbol, market_data in market_data_by_symbol.items():
            if symbol not in signals_by_symbol:
                raise ValueError(f"缺少 {symbol} 的信号数据")
            symbol_index = market_data.index.intersection(signals_by_symbol[symbol].index)
            common_index = symbol_index if common_index is None else common_index.intersection(symbol_index)
        if common_index is None or len(common_index) == 0:
            raise ValueError("多标的市场数据与信号数据没有共同时间索引")
        return common_index.sort_values()

    def _signal_strength(self, signal_row: pd.Series, signal: int) -> float:
        if "signal_strength" in signal_row:
            return abs(float(signal_row.get("signal_strength", 0.0)))
        if "market_strength" in signal_row:
            return abs(float(signal_row.get("market_strength", 0.0)))
        return float(abs(signal))

    def _score_symbol(self, signal_row: pd.Series) -> float:
        metric = self.portfolio_config.get("selection_metric", "market_strength")
        if metric == "market_strength":
            market_strength = float(signal_row.get("market_strength", 0.5))
            return abs(market_strength - 0.5) * 2
        return abs(float(signal_row.get("signal", 0)))

    def _drawdown_state(self, timestamp: Any):
        snapshot = self.adapter.get_account_snapshot(timestamp)
        self._peak_equity = max(self._peak_equity, float(snapshot.equity))
        drawdown = 0.0 if self._peak_equity <= 0 else 1 - float(snapshot.equity) / self._peak_equity
        if drawdown >= float(self.portfolio_config.get("max_portfolio_drawdown", 0.2)):
            self._paused = True
        return snapshot, float(drawdown)

    def _select_symbols(
        self,
        timestamp: Any,
        market_data_by_symbol: Dict[str, pd.DataFrame],
        signals_by_symbol: Dict[str, pd.DataFrame],
    ) -> List[Tuple[str, float, int, float, float]]:
        candidates: List[Tuple[str, float, int, float, float]] = []
        min_signal_strength = float(self.live_base_config.get("min_signal_strength", 0.0))
        for symbol, market_data in market_data_by_symbol.items():
            signal_row = signals_by_symbol[symbol].loc[timestamp]
            signal = int(signal_row.get("signal", 0))
            live_profile = self.market_resolver.resolve(symbol=symbol, section_name="live_trading")
            if signal == 0 or (signal < 0 and not live_profile.allow_short):
                continue
            signal_strength = self._signal_strength(signal_row, signal)
            if signal_strength < min_signal_strength:
                continue
            price = float(market_data.loc[timestamp, self.portfolio_config.get("price_field", "close")])
            if price <= 0:
                continue
            score = self._score_symbol(signal_row)
            candidates.append((symbol, score, signal, price, signal_strength))

        candidates.sort(key=lambda item: item[1], reverse=True)
        max_positions = int(self.portfolio_config.get("max_positions", 3))
        return candidates[:max_positions]

    def _target_quantities(
        self,
        timestamp: Any,
        selected_symbols: List[Tuple[str, float, int, float, float]],
    ) -> Dict[str, float]:
        snapshot = self.adapter.get_account_snapshot(timestamp)
        equity = max(float(snapshot.equity), 0.0)
        gross_target = min(
            float(self.portfolio_config.get("target_gross_allocation", 0.95)),
            float(self.portfolio_config.get("max_gross_exposure", 1.0)),
        )

        targets: Dict[str, float] = {}
        if not selected_symbols or equity <= 0:
            return targets

        equal_weight = gross_target / len(selected_symbols)
        per_symbol_weight = min(equal_weight, float(self.portfolio_config.get("max_symbol_allocation", 0.35)))

        for symbol, _score, signal, price, _strength in selected_symbols:
            profile = self.market_resolver.resolve(symbol=symbol, section_name="live_trading")
            raw_quantity = equity * per_symbol_weight / price
            if profile.allow_fractional:
                raw_quantity = round(raw_quantity, profile.quantity_precision)
            else:
                raw_quantity = math.floor(raw_quantity / profile.lot_size) * profile.lot_size
            if signal < 0 and not profile.allow_short:
                continue
            targets[symbol] = float(raw_quantity * signal)
        return targets

    def _should_skip_trade(self, current_quantity: float, target_quantity: float, price: float, equity: float) -> bool:
        if equity <= 0:
            return True
        delta_notional = abs(target_quantity - current_quantity) * price
        threshold = float(self.portfolio_config.get("turnover_buffer", 0.0))
        return delta_notional / equity < threshold

    def _constrain_target_by_session(self, session_state: str, current_quantity: float, desired_target: float):
        current_quantity = float(current_quantity)
        desired_target = float(desired_target)

        if session_state == "open":
            return desired_target, "session_open"
        if session_state == "closed":
            if abs(current_quantity) > 1e-12 and bool(self.live_base_config.get("flatten_outside_trading_hours", False)):
                return 0.0, "session_closed_flatten"
            return current_quantity, "session_closed"
        if abs(current_quantity) <= 1e-12:
            return 0.0, "exit_only_no_entry"
        if abs(desired_target) <= 1e-12 or current_quantity * desired_target <= 0:
            return 0.0, "exit_only_flatten"
        if abs(desired_target) < abs(current_quantity):
            return desired_target, "exit_only_reduce"
        return current_quantity, "exit_only_hold"

    def _cancel_stale_orders(self, timestamp: Any) -> int:
        cancel_after_seconds = int(self.live_base_config.get("cancel_after_seconds", 300) or 0)
        canceled_count = 0
        for open_order in list(self.adapter.get_open_orders()):
            if cancel_after_seconds <= 0:
                continue
            age_seconds = (pd.Timestamp(timestamp) - pd.Timestamp(open_order.submitted_at)).total_seconds()
            if age_seconds < cancel_after_seconds:
                continue
            canceled_order = self.adapter.cancel_order(open_order.order_id, timestamp, reason="stale_timeout")
            if canceled_order is not None:
                canceled_count += 1
                self._failed_attempts[open_order.symbol] = self._failed_attempts.get(open_order.symbol, 0) + 1
                self._emit_event(
                    "order_canceled",
                    timestamp,
                    severity="warning",
                    symbol=open_order.symbol,
                    message="撤销超时订单",
                    payload={"order_id": open_order.order_id},
                )
        return canceled_count

    def _build_account_row(
        self,
        timestamp: Any,
        selected_symbols: List[str],
        rebalanced: bool,
        rebalance_reason: str,
        drawdown: float,
        canceled_count: int,
    ) -> Dict[str, Any]:
        snapshot = self.adapter.get_account_snapshot(timestamp).to_dict()
        snapshot.update(
            {
                "selected_symbols": ",".join(selected_symbols),
                "selected_count": len(selected_symbols),
                "paused": bool(self._paused),
                "peak_equity": float(self._peak_equity),
                "drawdown": float(drawdown),
                "rebalanced": bool(rebalanced),
                "rebalance_reason": rebalance_reason,
                "canceled_orders": int(canceled_count),
                "daily_orders": int(self.risk_manager.daily_orders),
                "consecutive_failures": int(self.risk_manager.consecutive_failures),
            }
        )
        return snapshot

    def _build_symbol_rows(
        self,
        timestamp: Any,
        market_data_by_symbol: Dict[str, pd.DataFrame],
        signals_by_symbol: Dict[str, pd.DataFrame],
        target_quantities: Dict[str, float],
        selected_lookup: Dict[str, float],
        session_states: Dict[str, str],
        decision_reasons: Dict[str, str],
        rebalanced: bool,
        rebalance_reason: str,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for symbol, market_data in market_data_by_symbol.items():
            signal_row = signals_by_symbol[symbol].loc[timestamp]
            signal = int(signal_row.get("signal", 0))
            price = float(market_data.loc[timestamp, self.portfolio_config.get("price_field", "close")])
            position = self.adapter.get_position(symbol)
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "market_profile": self.market_resolver.resolve(symbol=symbol, section_name="live_trading").name,
                    "price": price,
                    "signal": signal,
                    "signal_strength": self._signal_strength(signal_row, signal),
                    "market_strength": float(signal_row.get("market_strength", 0.5)),
                    "risk_level": signal_row.get("risk_level", "unknown"),
                    "selected": symbol in selected_lookup,
                    "selection_score": float(selected_lookup.get(symbol, 0.0)),
                    "session_state": session_states.get(symbol, "closed"),
                    "decision_reason": decision_reasons.get(symbol, rebalance_reason),
                    "target_quantity": float(target_quantities.get(symbol, float(position.quantity))),
                    "position_quantity": float(position.quantity),
                    "position_side": position.side,
                    "avg_entry_price": float(position.avg_price),
                    "market_value": float(position.market_value),
                    "unrealized_pnl": float(position.unrealized_pnl),
                    "rebalanced": bool(rebalanced),
                    "rebalance_reason": rebalance_reason,
                    "open_orders": len(self.adapter.get_open_orders(symbol=symbol)),
                }
            )
        return rows

    def _summary(self, account_history: pd.DataFrame, symbol_history: pd.DataFrame) -> Dict[str, Any]:
        if account_history.empty:
            return {
                "initial_equity": float(self.adapter.initial_cash),
                "final_equity": float(self.adapter.initial_cash),
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "orders": 0,
                "fills": 0,
                "submitted_orders": 0,
                "rejected_orders": 0,
                "canceled_orders": 0,
                "open_orders": 0,
                "active_symbols": 0,
                "session_blocks": 0,
                "risk_blocks": 0,
                "fees_paid": 0.0,
                "paused": False,
            }

        equity = account_history["equity"].astype(float)
        baseline = pd.Series([float(self.adapter.initial_cash)])
        drawdown = pd.concat([baseline, equity.reset_index(drop=True)], ignore_index=True)
        drawdown = drawdown / drawdown.cummax() - 1
        orders_df = pd.DataFrame([order.to_dict() for order in self.adapter.orders])
        if orders_df.empty:
            submitted_orders = 0
            rejected_orders = 0
            canceled_orders = 0
            open_orders = 0
        else:
            submitted_orders = int(orders_df["status"].isin(["submitted", "filled", "canceled"]).sum())
            rejected_orders = int((orders_df["status"] == "rejected").sum())
            canceled_orders = int((orders_df["status"] == "canceled").sum())
            open_orders = int((orders_df["status"] == "submitted").sum())

        session_blocks = int(symbol_history["decision_reason"].astype(str).str.startswith("session_").sum()) + int(
            symbol_history["decision_reason"].astype(str).str.startswith("exit_only").sum()
        )
        risk_blocks = int(symbol_history["decision_reason"].astype(str).str.startswith("risk_").sum())
        return {
            "initial_equity": float(self.adapter.initial_cash),
            "final_equity": float(equity.iloc[-1]),
            "total_return": float(equity.iloc[-1] / float(self.adapter.initial_cash) - 1),
            "max_drawdown": float(abs(drawdown.min())),
            "orders": len(self.adapter.orders),
            "fills": len(self.adapter.fills),
            "submitted_orders": submitted_orders,
            "rejected_orders": rejected_orders,
            "canceled_orders": canceled_orders,
            "open_orders": open_orders,
            "active_symbols": int(symbol_history.loc[symbol_history["position_quantity"].abs() > 1e-12, "symbol"].nunique()) if not symbol_history.empty else 0,
            "session_blocks": session_blocks,
            "risk_blocks": risk_blocks,
            "fees_paid": float(account_history["fees_paid"].iloc[-1]),
            "realized_pnl": float(account_history["realized_pnl"].iloc[-1]),
            "unrealized_pnl": float(account_history["unrealized_pnl"].iloc[-1]),
            "paused": bool(self._paused),
        }

    def _build_results(self) -> Dict[str, Any]:
        account_history = pd.DataFrame(self._account_rows)
        if not account_history.empty:
            account_history["timestamp"] = pd.to_datetime(account_history["timestamp"])
            account_history = account_history.set_index("timestamp")

        symbol_history = pd.DataFrame(self._symbol_rows)
        if not symbol_history.empty:
            symbol_history["timestamp"] = pd.to_datetime(symbol_history["timestamp"])
            symbol_history = symbol_history.sort_values(["timestamp", "symbol"])

        return {
            "orders": [order.to_dict() for order in self.adapter.orders],
            "fills": [fill.to_dict() for fill in self.adapter.fills],
            "account_history": account_history,
            "symbol_history": symbol_history,
            "positions": self.adapter.get_positions(),
            "summary": self._summary(account_history, symbol_history),
        }

    def _process_step(
        self,
        timestamp: Any,
        market_data_by_symbol: Dict[str, pd.DataFrame],
        signals_by_symbol: Dict[str, pd.DataFrame],
    ):
        market_prices = {
            symbol: float(market_data.loc[timestamp, self.portfolio_config.get("price_field", "close")])
            for symbol, market_data in market_data_by_symbol.items()
        }
        for symbol, price in market_prices.items():
            self.adapter.mark_to_market(symbol, price, timestamp)

        cycle_fills = self.adapter.process_pending_orders(market_prices, timestamp)
        if cycle_fills:
            self.risk_manager.record_success()
            for fill in cycle_fills:
                self._failed_attempts[fill.symbol] = 0
                self._emit_event(
                    "order_filled",
                    fill.filled_at,
                    symbol=fill.symbol,
                    payload={
                        "order_id": fill.order_id,
                        "side": fill.side,
                        "quantity": float(fill.quantity),
                        "fill_price": float(fill.fill_price),
                    },
                )

        snapshot, drawdown = self._drawdown_state(timestamp)
        risk_state = self.risk_manager.evaluate(snapshot, timestamp)
        self._emit_event(
            "account_snapshot",
            timestamp,
            payload={
                "equity": float(snapshot.equity),
                "cash": float(snapshot.cash),
                "daily_drawdown": float(risk_state.get("daily_drawdown", 0.0)),
                "active_positions": int(snapshot.active_positions),
            },
        )

        selected: List[Tuple[str, float, int, float, float]] = []
        target_quantities: Dict[str, float] = {}
        selected_lookup: Dict[str, float] = {}
        rebalance_reason = "hold"
        scheduled = self.calendar.should_rebalance(timestamp, self._last_rebalance_at)

        if risk_state.get("hard_halt", False):
            self._paused = True
            rebalance_reason = "risk_pause"
            self._emit_event(
                "risk_halt",
                timestamp,
                severity="critical",
                message="组合级风控暂停",
                payload={
                    "reason": risk_state.get("reason", "blocked"),
                    "daily_drawdown": float(risk_state.get("daily_drawdown", 0.0)),
                },
            )
        elif self._paused:
            rebalance_reason = "paused"
        elif scheduled:
            rebalance_reason = self.calendar.rebalance_reason(timestamp, self._last_rebalance_at)
            selected = self._select_symbols(timestamp, market_data_by_symbol, signals_by_symbol)
            target_quantities = self._target_quantities(timestamp, selected)
            selected_lookup = {symbol: score for symbol, score, _signal, _price, _strength in selected}

        session_states: Dict[str, str] = {}
        decision_reasons: Dict[str, str] = {}
        rebalanced = False
        canceled_count = self._cancel_stale_orders(timestamp)
        open_orders_total = len(self.adapter.get_open_orders())
        for symbol, market_data in market_data_by_symbol.items():
            price = float(market_data.loc[timestamp, self.portfolio_config.get("price_field", "close")])
            signal_row = signals_by_symbol[symbol].loc[timestamp]
            signal = int(signal_row.get("signal", 0))
            current_position = self.adapter.get_position(symbol)
            session_state = self._session_for(symbol).get_state(timestamp)
            session_states[symbol] = session_state

            desired_target = float(current_position.quantity)
            if rebalance_reason == "risk_pause":
                desired_target = 0.0
            elif rebalance_reason == "paused":
                desired_target = float(current_position.quantity)
            elif rebalance_reason != "hold":
                desired_target = float(target_quantities.get(symbol, 0.0))

            target_quantity, decision_reason = self._constrain_target_by_session(
                session_state,
                current_position.quantity,
                desired_target,
            )
            decision_reasons[symbol] = decision_reason
            if decision_reason != "session_open":
                self._emit_event(
                    "session_blocked",
                    timestamp,
                    symbol=symbol,
                    message=decision_reason,
                    payload={"session_state": session_state, "desired_target": float(desired_target)},
                )

            if rebalance_reason in {"hold", "paused"}:
                target_quantities.setdefault(symbol, float(target_quantity))
                continue

            if self._should_skip_trade(float(current_position.quantity), float(target_quantity), price, snapshot.equity):
                target_quantities[symbol] = float(target_quantity)
                continue

            if self.adapter.get_open_orders(symbol=symbol):
                target_quantities[symbol] = float(target_quantity)
                continue

            risk_reason = self.risk_manager.validate_order(
                current_quantity=float(current_position.quantity),
                target_quantity=float(target_quantity),
                price=price,
                open_order_count=open_orders_total,
            )
            if risk_reason is not None:
                decision_reasons[symbol] = f"risk_{risk_reason}"
                target_quantities[symbol] = float(current_position.quantity)
                continue

            order = self.adapter.create_target_order(
                symbol=symbol,
                target_quantity=target_quantity,
                timestamp=timestamp,
                requested_price=price,
                signal=signal,
                metadata={
                    "market_strength": float(signal_row.get("market_strength", 0.0)),
                    "risk_level": signal_row.get("risk_level", "unknown"),
                    "selection_score": float(selected_lookup.get(symbol, 0.0)),
                    "rebalance_reason": rebalance_reason,
                    "market_profile": self.market_resolver.resolve(symbol=symbol, section_name="live_trading").name,
                },
            )
            if order is None:
                target_quantities[symbol] = float(current_position.quantity)
                continue

            order = self.adapter.submit_order(order, price, timestamp)
            self.risk_manager.record_submission(timestamp)
            open_orders_total = len(self.adapter.get_open_orders())
            rebalanced = True
            target_quantities[symbol] = float(target_quantity)
            if order.status == "rejected":
                self.risk_manager.record_failure()
                self._failed_attempts[symbol] = self._failed_attempts.get(symbol, 0) + 1
                decision_reasons[symbol] = "order_rejected"
                self._emit_event(
                    "order_rejected",
                    timestamp,
                    severity="warning",
                    symbol=symbol,
                    message="订单提交被拒绝",
                    payload={"order_id": order.order_id, "side": order.side, "quantity": float(order.quantity)},
                )
            else:
                self._emit_event(
                    "order_submitted",
                    timestamp,
                    symbol=symbol,
                    message="组合订单已提交",
                    payload={"order_id": order.order_id, "side": order.side, "quantity": float(order.quantity)},
                )
                same_cycle_fills = self.adapter.process_pending_orders(market_prices, timestamp)
                if same_cycle_fills:
                    self.risk_manager.record_success()
                    for fill in same_cycle_fills:
                        self._failed_attempts[fill.symbol] = 0
                        self._emit_event(
                            "order_filled",
                            fill.filled_at,
                            symbol=fill.symbol,
                            payload={
                                "order_id": fill.order_id,
                                "side": fill.side,
                                "quantity": float(fill.quantity),
                                "fill_price": float(fill.fill_price),
                            },
                        )

        if rebalance_reason not in {"hold", "paused"}:
            self._last_rebalance_at = timestamp

        for symbol, price in market_prices.items():
            self.adapter.mark_to_market(symbol, price, timestamp)
        self._account_rows.append(
            self._build_account_row(
                timestamp,
                [item[0] for item in selected],
                rebalanced,
                rebalance_reason,
                drawdown,
                canceled_count,
            )
        )
        self._symbol_rows.extend(
            self._build_symbol_rows(
                timestamp,
                market_data_by_symbol,
                signals_by_symbol,
                target_quantities,
                selected_lookup,
                session_states,
                decision_reasons,
                rebalanced,
                rebalance_reason,
            )
        )
        self._last_processed_at = pd.Timestamp(timestamp)

    def run_incremental(
        self,
        market_data_by_symbol: Dict[str, pd.DataFrame],
        signals_by_symbol: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        common_index = self._common_index(market_data_by_symbol, signals_by_symbol)
        if self._last_processed_at is not None:
            common_index = common_index[common_index > pd.Timestamp(self._last_processed_at)]

        for timestamp in common_index:
            self._process_step(timestamp, market_data_by_symbol, signals_by_symbol)
        return self._build_results()

    def run(
        self,
        market_data_by_symbol: Dict[str, pd.DataFrame],
        signals_by_symbol: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        self.reset_state()
        return self.run_incremental(market_data_by_symbol, signals_by_symbol)

    def save_results(self, results: Dict[str, Any], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        pd.DataFrame(results.get("orders", [])).to_csv(
            os.path.join(output_dir, "live_orders.csv"), index=False, encoding="utf-8"
        )
        pd.DataFrame(results.get("fills", [])).to_csv(
            os.path.join(output_dir, "live_fills.csv"), index=False, encoding="utf-8"
        )

        account_history = results.get("account_history")
        if isinstance(account_history, pd.DataFrame) and not account_history.empty:
            account_history.to_csv(os.path.join(output_dir, "live_account_history.csv"), encoding="utf-8")

        symbol_history = results.get("symbol_history")
        if isinstance(symbol_history, pd.DataFrame) and not symbol_history.empty:
            symbol_history.to_csv(os.path.join(output_dir, "live_symbol_history.csv"), index=False, encoding="utf-8")

        with open(os.path.join(output_dir, "live_positions.json"), "w", encoding="utf-8") as file:
            json.dump(results.get("positions", {}), file, ensure_ascii=False, indent=2, default=str)

        with open(os.path.join(output_dir, "live_summary.json"), "w", encoding="utf-8") as file:
            json.dump(results.get("summary", {}), file, ensure_ascii=False, indent=2, default=str)
