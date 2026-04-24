"""Live trading 执行骨架。"""

import json
import math
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from .broker_adapter import PaperLiveBrokerAdapter, build_live_adapter
from .live_broker import PaperLiveBroker
from .live_risk_manager import LiveRiskManager
from .market_profile import MarketProfileResolver
from .market_session import MarketSession


DEFAULT_LIVE_TRADING_CONFIG = {
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
}


class LiveTradingEngine:
    def __init__(
        self,
        config: Dict[str, Any],
        adapter: Optional[PaperLiveBrokerAdapter] = None,
        broker: Optional[PaperLiveBroker] = None,
        session: Optional[MarketSession] = None,
        risk_manager: Optional[LiveRiskManager] = None,
    ):
        self.config = config or {}
        self.market_resolver = MarketProfileResolver(self.config)
        self.symbol = self.config.get("data", {}).get("symbol", "unknown")
        self.live_config = self._resolve_live_config(self.symbol)

        self._owned_adapter = adapter is None and broker is None
        if adapter is not None:
            self.adapter = adapter
        elif broker is not None:
            self.adapter = PaperLiveBrokerAdapter(broker)
        else:
            self.adapter = build_live_adapter(self.config)

        self._external_session = session
        self._external_risk_manager = risk_manager
        self.session = session or self._build_session()
        self.risk_manager = risk_manager or self._build_risk_manager()

        self._peak_equity = float(self.adapter.initial_cash)
        self._failed_attempts: Dict[str, int] = {}
        self._history_rows: List[Dict[str, Any]] = []
        self._last_processed_at = None

    def _resolve_live_config(self, symbol: Optional[str]) -> Dict[str, Any]:
        live_config = dict(DEFAULT_LIVE_TRADING_CONFIG)
        live_config.update(self.config.get("live_trading", {}))
        live_config.update(self.market_resolver.section_settings(symbol=symbol, section_name="live_trading"))
        return live_config

    def _refresh_symbol_context(self, symbol: Optional[str]):
        resolved_symbol = symbol or self.symbol
        self.symbol = resolved_symbol
        self.live_config = self._resolve_live_config(self.symbol)
        if self._external_session is None:
            self.session = self._build_session()

    def _build_session(self) -> MarketSession:
        return MarketSession(
            timezone=self.live_config.get("timezone", "Asia/Shanghai"),
            trading_days=self.live_config.get("trading_days", [0, 1, 2, 3, 4]),
            session_start=self.live_config.get("session_start", "09:30"),
            session_end=self.live_config.get("session_end", "15:00"),
            exit_only_start=self.live_config.get("exit_only_start", "14:55"),
            sessions=self.live_config.get("sessions"),
        )

    def _build_risk_manager(self) -> LiveRiskManager:
        return LiveRiskManager(self.config.get("live_risk", {}))

    @property
    def processed_count(self) -> int:
        return len(self._history_rows)

    @property
    def last_processed_at(self):
        return self._last_processed_at

    def reset_state(self):
        if self._owned_adapter:
            self.adapter = build_live_adapter(self.config)
        self._refresh_symbol_context(self.config.get("data", {}).get("symbol", self.symbol))
        self.risk_manager = self._external_risk_manager or self._build_risk_manager()
        self._peak_equity = float(self.adapter.initial_cash)
        self._failed_attempts = {}
        self._history_rows = []
        self._last_processed_at = None

    def _calculate_target_quantity(self, signal: int, equity: float, price: float) -> float:
        if signal == 0 or price <= 0:
            return 0.0
        if signal < 0 and not bool(self.live_config.get("allow_short", True)):
            return 0.0

        allocation_pct = float(self.live_config.get("allocation_pct", 0.95))
        raw_quantity = max(equity, 0.0) * allocation_pct / price

        if not self.live_config.get("allow_fractional", False):
            lot_size = float(self.live_config.get("lot_size", 1.0)) or 1.0
            raw_quantity = math.floor(raw_quantity / lot_size) * lot_size
        else:
            precision = int(self.live_config.get("quantity_precision", 6))
            raw_quantity = round(raw_quantity, precision)

        return float(raw_quantity * signal)

    def _infer_signal_strength(self, signal_row: pd.Series, signal: int) -> float:
        if "signal_strength" in signal_row:
            return abs(float(signal_row.get("signal_strength", 0.0)))
        if "market_strength" in signal_row:
            return abs(float(signal_row.get("market_strength", 0.0)))
        return float(abs(signal))

    def _constrain_target_by_session(self, session_state: str, current_quantity: float, desired_target: float):
        current_quantity = float(current_quantity)
        desired_target = float(desired_target)

        if session_state == "open":
            return desired_target, "session_open"

        if session_state == "closed":
            if abs(current_quantity) > 1e-12 and bool(self.live_config.get("flatten_outside_trading_hours", False)):
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
        cancel_after_seconds = int(self.live_config.get("cancel_after_seconds", 300) or 0)
        canceled_count = 0
        for open_order in list(self.adapter.get_open_orders(symbol=self.symbol)):
            if cancel_after_seconds <= 0:
                continue
            age_seconds = (pd.Timestamp(timestamp) - pd.Timestamp(open_order.submitted_at)).total_seconds()
            if age_seconds < cancel_after_seconds:
                continue
            canceled_order = self.adapter.cancel_order(open_order.order_id, timestamp, reason="stale_timeout")
            if canceled_order is not None:
                canceled_count += 1
                self._failed_attempts[self.symbol] = self._failed_attempts.get(self.symbol, 0) + 1
        return canceled_count

    def _build_history_row(
        self,
        timestamp: Any,
        price: float,
        signal: int,
        signal_strength: float,
        session_state: str,
        risk_state: Dict[str, Any],
        target_quantity: float,
        decision_reason: str,
        execution_event: str,
        order,
        latest_fill,
        canceled_count: int,
    ) -> Dict[str, Any]:
        snapshot = self.adapter.get_account_snapshot(timestamp).to_dict()
        position = self.adapter.get_position(self.symbol)
        snapshot.update(
            {
                "market_profile": self.live_config.get("market_profile", "default"),
                "price": float(price),
                "signal": int(signal),
                "signal_strength": float(signal_strength),
                "target_quantity": float(target_quantity),
                "position_quantity": float(position.quantity),
                "position_side": position.side,
                "avg_entry_price": float(position.avg_price),
                "session_state": session_state,
                "risk_halt": bool(risk_state.get("hard_halt", False)),
                "risk_reason": risk_state.get("reason", "ok"),
                "daily_drawdown": float(risk_state.get("daily_drawdown", 0.0)),
                "daily_orders": int(self.risk_manager.daily_orders),
                "consecutive_failures": int(self.risk_manager.consecutive_failures),
                "decision_reason": decision_reason,
                "execution_event": execution_event,
                "order_id": order.order_id if order else None,
                "order_status": order.status if order else None,
                "order_attempt_count": int(getattr(order, "attempt_count", 0) or 0) if order else 0,
                "fill_id": latest_fill.fill_id if latest_fill else None,
                "fill_price": float(latest_fill.fill_price) if latest_fill else None,
                "open_orders": len(self.adapter.get_open_orders(symbol=self.symbol)),
                "canceled_orders": int(canceled_count),
                "retry_count": int(self._failed_attempts.get(self.symbol, 0)),
                "peak_equity": float(self._peak_equity),
            }
        )
        return snapshot

    def _summary_from_history(self, account_history: pd.DataFrame) -> Dict[str, Any]:
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
                "session_blocks": 0,
                "risk_blocks": 0,
                "fees_paid": 0.0,
            }

        equity = account_history["equity"].astype(float)
        baseline = pd.Series([float(self.adapter.initial_cash)])
        equity_with_baseline = pd.concat([baseline, equity.reset_index(drop=True)], ignore_index=True)
        drawdown = equity_with_baseline / equity_with_baseline.cummax() - 1
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

        session_blocks = int(account_history["decision_reason"].astype(str).str.startswith("session_").sum()) + int(
            account_history["decision_reason"].astype(str).str.startswith("exit_only").sum()
        )
        risk_blocks = int(account_history["decision_reason"].astype(str).str.startswith("risk_").sum())
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
            "session_blocks": session_blocks,
            "risk_blocks": risk_blocks,
            "fees_paid": float(account_history["fees_paid"].iloc[-1]),
            "realized_pnl": float(account_history["realized_pnl"].iloc[-1]),
            "unrealized_pnl": float(account_history["unrealized_pnl"].iloc[-1]),
        }

    def _build_results(self) -> Dict[str, Any]:
        account_history = pd.DataFrame(self._history_rows)
        if not account_history.empty:
            account_history["timestamp"] = pd.to_datetime(account_history["timestamp"])
            account_history = account_history.set_index("timestamp")

        return {
            "orders": [order.to_dict() for order in self.adapter.orders],
            "fills": [fill.to_dict() for fill in self.adapter.fills],
            "account_history": account_history,
            "positions": self.adapter.get_positions(),
            "summary": self._summary_from_history(account_history),
        }

    def _process_step(self, timestamp: Any, price: float, signal_row: pd.Series):
        signal = int(signal_row.get("signal", 0))
        signal_strength = self._infer_signal_strength(signal_row, signal)

        self.adapter.mark_to_market(self.symbol, price, timestamp)
        cycle_fills = self.adapter.process_pending_orders({self.symbol: price}, timestamp)
        if cycle_fills:
            self.risk_manager.record_success()
            self._failed_attempts[self.symbol] = 0

        snapshot = self.adapter.get_account_snapshot(timestamp)
        self._peak_equity = max(self._peak_equity, float(snapshot.equity))
        risk_state = self.risk_manager.evaluate(snapshot, timestamp)
        current_position = self.adapter.get_position(self.symbol)
        session_state = self.session.get_state(timestamp)

        desired_target = float(current_position.quantity)
        decision_reason = "hold"
        if risk_state.get("hard_halt", False):
            if abs(current_position.quantity) > 1e-12:
                desired_target = 0.0
                decision_reason = "risk_flatten"
            else:
                decision_reason = f"risk_{risk_state.get('reason', 'blocked')}"
        else:
            if signal_strength < float(self.live_config.get("min_signal_strength", 0.0)):
                desired_target = 0.0
                decision_reason = "weak_signal"
            else:
                desired_target = self._calculate_target_quantity(signal, snapshot.equity, price)
                decision_reason = "signal_target"

        target_quantity, session_reason = self._constrain_target_by_session(session_state, current_position.quantity, desired_target)
        if session_reason != "session_open":
            decision_reason = session_reason

        execution_event = "hold"
        canceled_count = self._cancel_stale_orders(timestamp)
        if canceled_count > 0:
            execution_event = "cancel_stale_order"
            if decision_reason == "hold":
                decision_reason = "stale_order_canceled"

        open_orders = self.adapter.get_open_orders(symbol=self.symbol)
        order = None
        latest_fill = next((fill for fill in reversed(cycle_fills) if fill.symbol == self.symbol), None)
        if open_orders:
            execution_event = "open_order_wait"
        elif abs(target_quantity - float(current_position.quantity)) > 1e-9:
            failed_attempts = self._failed_attempts.get(self.symbol, 0)
            max_retries = int(self.live_config.get("max_order_retries", 2))
            if failed_attempts > max_retries:
                decision_reason = "retry_limit_exceeded"
            else:
                risk_reason = self.risk_manager.validate_order(
                    current_quantity=float(current_position.quantity),
                    target_quantity=float(target_quantity),
                    price=price,
                    open_order_count=0,
                )
                if risk_reason is not None:
                    decision_reason = f"risk_{risk_reason}"
                else:
                    order = self.adapter.create_target_order(
                        symbol=self.symbol,
                        target_quantity=target_quantity,
                        timestamp=timestamp,
                        requested_price=price,
                        signal=signal,
                        metadata={
                            "market_strength": float(signal_row.get("market_strength", 0.0)),
                            "risk_level": signal_row.get("risk_level", "unknown"),
                            "signal_strength": float(signal_strength),
                            "market_profile": self.live_config.get("market_profile", "default"),
                        },
                    )
                    if order is not None:
                        order = self.adapter.submit_order(order, price, timestamp)
                        self.risk_manager.record_submission(timestamp)
                        if order.status == "rejected":
                            execution_event = "order_rejected"
                            self.risk_manager.record_failure()
                            self._failed_attempts[self.symbol] = self._failed_attempts.get(self.symbol, 0) + 1
                            if decision_reason == "signal_target":
                                decision_reason = "order_rejected"
                        else:
                            execution_event = "order_submitted"
                            same_cycle_fills = self.adapter.process_pending_orders({self.symbol: price}, timestamp)
                            if same_cycle_fills:
                                latest_fill = next(
                                    (fill for fill in reversed(same_cycle_fills) if fill.symbol == self.symbol),
                                    latest_fill,
                                )
                                self.risk_manager.record_success()
                                self._failed_attempts[self.symbol] = 0

        self.adapter.mark_to_market(self.symbol, price, timestamp)
        final_snapshot = self.adapter.get_account_snapshot(timestamp)
        self._peak_equity = max(self._peak_equity, float(final_snapshot.equity))
        self._history_rows.append(
            self._build_history_row(
                timestamp,
                price,
                signal,
                signal_strength,
                session_state,
                risk_state,
                target_quantity,
                decision_reason,
                execution_event,
                order,
                latest_fill,
                canceled_count,
            )
        )
        self._last_processed_at = pd.Timestamp(timestamp)

    def _close_positions(self, timestamp: Any, price: float):
        current_position = self.adapter.get_position(self.symbol)
        if abs(current_position.quantity) <= 1e-12:
            return
        closing_order = self.adapter.create_target_order(
            symbol=self.symbol,
            target_quantity=0.0,
            timestamp=timestamp,
            requested_price=price,
            signal=0,
            metadata={"reason": "close_positions_on_finish"},
        )
        if closing_order is None:
            return

        closing_order = self.adapter.submit_order(closing_order, price, timestamp)
        self.risk_manager.record_submission(timestamp)
        same_cycle_fills = self.adapter.process_pending_orders({self.symbol: price}, timestamp)
        closing_fill = next((fill for fill in reversed(same_cycle_fills) if fill.symbol == self.symbol), None)
        if closing_fill is not None:
            self.risk_manager.record_success()
        self.adapter.mark_to_market(self.symbol, price, timestamp)
        final_risk_state = self.risk_manager.evaluate(self.adapter.get_account_snapshot(timestamp), timestamp)
        self._history_rows.append(
            self._build_history_row(
                timestamp,
                price,
                0,
                0.0,
                self.session.get_state(timestamp),
                final_risk_state,
                0.0,
                "finish_close",
                "order_submitted",
                closing_order,
                closing_fill,
                0,
            )
        )
        self._last_processed_at = pd.Timestamp(timestamp)

    def _validate_inputs(self, market_data: pd.DataFrame, signals: pd.DataFrame):
        if market_data is None or market_data.empty:
            raise ValueError("市场数据为空，无法运行 live trading")
        if signals is None or signals.empty:
            raise ValueError("信号数据为空，无法运行 live trading")

        price_field = self.live_config.get("price_field", "close")
        if price_field not in market_data.columns:
            raise ValueError(f"市场数据中缺少价格列: {price_field}")

    def run_incremental(self, market_data: pd.DataFrame, signals: pd.DataFrame, symbol: Optional[str] = None) -> Dict[str, Any]:
        resolved_symbol = symbol or self.symbol
        if resolved_symbol != self.symbol and self._last_processed_at is not None:
            self.reset_state()
        self._refresh_symbol_context(resolved_symbol)
        self._validate_inputs(market_data, signals)

        price_field = self.live_config.get("price_field", "close")
        aligned_index = market_data.index.intersection(signals.index).sort_values()
        if len(aligned_index) == 0:
            raise ValueError("市场数据与信号数据没有共同时间索引")

        if self._last_processed_at is not None:
            aligned_index = aligned_index[aligned_index > pd.Timestamp(self._last_processed_at)]

        aligned_market = market_data.loc[:, :]
        aligned_signals = signals.loc[:, :]
        for timestamp in aligned_index:
            price = float(aligned_market.loc[timestamp, price_field])
            signal_row = aligned_signals.loc[timestamp]
            self._process_step(timestamp, price, signal_row)

        return self._build_results()

    def run(self, market_data: pd.DataFrame, signals: pd.DataFrame, symbol: Optional[str] = None) -> Dict[str, Any]:
        self.reset_state()
        self._refresh_symbol_context(symbol or self.symbol)
        results = self.run_incremental(market_data, signals, symbol=self.symbol)
        if self.live_config.get("close_positions_on_finish", False):
            aligned_index = market_data.index.intersection(signals.index).sort_values()
            if len(aligned_index) > 0:
                last_timestamp = aligned_index[-1]
                last_price = float(market_data.loc[last_timestamp, self.live_config.get("price_field", "close")])
                self._close_positions(last_timestamp, last_price)
                results = self._build_results()
        return results

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

        with open(os.path.join(output_dir, "live_positions.json"), "w", encoding="utf-8") as file:
            json.dump(results.get("positions", {}), file, ensure_ascii=False, indent=2, default=str)

        with open(os.path.join(output_dir, "live_summary.json"), "w", encoding="utf-8") as file:
            json.dump(results.get("summary", {}), file, ensure_ascii=False, indent=2, default=str)
