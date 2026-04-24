"""Paper trading engine: 将策略信号转换为订单、成交和账户状态。"""

import json
import math
import os
from typing import Any, Dict, Optional

import pandas as pd

from .broker_adapter import PaperBrokerAdapter, build_paper_adapter
from .market_profile import MarketProfileResolver
from .paper_broker import PaperBroker
from .trading_calendar import TradingCalendar


DEFAULT_PAPER_TRADING_CONFIG = {
    "enabled": False,
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
}


class PaperTradingEngine:
    def __init__(self, config: Dict[str, Any], adapter: Optional[PaperBrokerAdapter] = None, broker: Optional[PaperBroker] = None):
        self.config = config or {}
        self.market_resolver = MarketProfileResolver(self.config)
        self.symbol = self.config.get("data", {}).get("symbol", "unknown")
        self.paper_config = self._resolve_paper_config(self.symbol)

        if adapter is not None:
            self.adapter = adapter
        elif broker is not None:
            self.adapter = PaperBrokerAdapter(broker)
        else:
            self.adapter = build_paper_adapter(self.config)

        self.calendar = TradingCalendar(
            rebalance_frequency=self.paper_config.get("rebalance_frequency", "daily"),
            rebalance_weekday=int(self.paper_config.get("rebalance_weekday", 0)),
            rebalance_day_of_month=int(self.paper_config.get("rebalance_day_of_month", 1)),
        )
        self._paused = False
        self._peak_equity = float(self.adapter.initial_cash)
        self._last_rebalance_at = None

    def _resolve_paper_config(self, symbol: Optional[str]) -> Dict[str, Any]:
        paper_config = dict(DEFAULT_PAPER_TRADING_CONFIG)
        paper_config.update(self.config.get("paper_trading", {}))
        paper_config.update(self.market_resolver.section_settings(symbol=symbol, section_name="paper_trading"))
        return paper_config

    def _refresh_symbol_context(self, symbol: Optional[str]):
        self.symbol = symbol or self.symbol
        self.paper_config = self._resolve_paper_config(self.symbol)
        self.calendar = TradingCalendar(
            rebalance_frequency=self.paper_config.get("rebalance_frequency", "daily"),
            rebalance_weekday=int(self.paper_config.get("rebalance_weekday", 0)),
            rebalance_day_of_month=int(self.paper_config.get("rebalance_day_of_month", 1)),
        )

    def _calculate_target_quantity(self, signal: int, equity: float, price: float) -> float:
        if signal == 0 or price <= 0:
            return 0.0
        if signal < 0 and not bool(self.paper_config.get("allow_short", True)):
            return 0.0

        allocation_pct = float(self.paper_config.get("allocation_pct", 0.95))
        raw_quantity = max(equity, 0.0) * allocation_pct / price

        if not self.paper_config.get("allow_fractional", False):
            lot_size = float(self.paper_config.get("lot_size", 1.0)) or 1.0
            raw_quantity = math.floor(raw_quantity / lot_size) * lot_size
        else:
            precision = int(self.paper_config.get("quantity_precision", 6))
            raw_quantity = round(raw_quantity, precision)

        return float(raw_quantity * signal)

    def _should_skip_trade(self, current_quantity: float, target_quantity: float, price: float, equity: float) -> bool:
        if equity <= 0:
            return True
        delta_notional = abs(target_quantity - current_quantity) * price
        threshold = float(self.paper_config.get("turnover_buffer", 0.0))
        return delta_notional / equity < threshold

    def _drawdown_state(self, timestamp: Any):
        snapshot = self.adapter.get_account_snapshot(timestamp)
        self._peak_equity = max(self._peak_equity, float(snapshot.equity))
        drawdown = 0.0 if self._peak_equity <= 0 else 1 - float(snapshot.equity) / self._peak_equity
        if drawdown >= float(self.paper_config.get("max_account_drawdown", 0.2)):
            self._paused = True
        return snapshot, float(drawdown)

    def _build_history_row(
        self,
        timestamp: Any,
        price: float,
        signal: int,
        target_quantity: float,
        order,
        fill,
        rebalanced: bool,
        rebalance_reason: str,
        drawdown: float,
    ) -> Dict[str, Any]:
        snapshot = self.adapter.get_account_snapshot(timestamp).to_dict()
        position = self.adapter.get_position(self.symbol)
        snapshot.update(
            {
                "market_profile": self.paper_config.get("market_profile", "default"),
                "price": float(price),
                "signal": int(signal),
                "target_quantity": float(target_quantity),
                "position_quantity": float(position.quantity),
                "position_side": position.side,
                "avg_entry_price": float(position.avg_price),
                "order_id": order.order_id if order else None,
                "fill_id": fill.fill_id if fill else None,
                "fill_price": float(fill.fill_price) if fill else None,
                "fill_quantity": float(fill.quantity) if fill else 0.0,
                "rebalanced": bool(rebalanced),
                "rebalance_reason": rebalance_reason,
                "paused": bool(self._paused),
                "peak_equity": float(self._peak_equity),
                "drawdown": float(drawdown),
            }
        )
        return snapshot

    def _summarize(self, account_history: pd.DataFrame) -> Dict[str, Any]:
        if account_history.empty:
            return {
                "initial_equity": float(self.adapter.initial_cash),
                "final_equity": float(self.adapter.initial_cash),
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "orders": 0,
                "fills": 0,
                "closed_trades": 0,
                "win_rate": 0.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "fees_paid": 0.0,
                "rebalances": 0,
                "paused": bool(self._paused),
            }

        equity = account_history["equity"].astype(float)
        baseline = pd.Series([float(self.adapter.initial_cash)])
        equity_with_baseline = pd.concat([baseline, equity.reset_index(drop=True)], ignore_index=True)
        running_max = equity_with_baseline.cummax()
        drawdown = equity_with_baseline / running_max - 1
        closed_fills = [fill for fill in self.adapter.fills if abs(getattr(fill, "gross_realized_pnl", 0.0)) > 1e-12]
        wins = [fill for fill in closed_fills if fill.realized_pnl > 0]
        return {
            "initial_equity": float(self.adapter.initial_cash),
            "final_equity": float(equity.iloc[-1]),
            "total_return": float(equity.iloc[-1] / float(self.adapter.initial_cash) - 1),
            "max_drawdown": float(abs(drawdown.min())),
            "orders": len(self.adapter.orders),
            "fills": len(self.adapter.fills),
            "closed_trades": len(closed_fills),
            "win_rate": float(len(wins) / len(closed_fills)) if closed_fills else 0.0,
            "realized_pnl": float(account_history["realized_pnl"].iloc[-1]),
            "unrealized_pnl": float(account_history["unrealized_pnl"].iloc[-1]),
            "fees_paid": float(account_history["fees_paid"].iloc[-1]),
            "rebalances": int(account_history["rebalanced"].sum()),
            "paused": bool(self._paused),
        }

    def run(self, market_data: pd.DataFrame, signals: pd.DataFrame, symbol: Optional[str] = None) -> Dict[str, Any]:
        if market_data is None or market_data.empty:
            raise ValueError("市场数据为空，无法运行 paper trading")
        if signals is None or signals.empty:
            raise ValueError("信号数据为空，无法运行 paper trading")

        self._refresh_symbol_context(symbol or self.symbol)
        price_field = self.paper_config.get("price_field", "close")
        if price_field not in market_data.columns:
            raise ValueError(f"市场数据中缺少价格列: {price_field}")

        aligned_index = market_data.index.intersection(signals.index)
        if len(aligned_index) == 0:
            raise ValueError("市场数据与信号数据没有共同时间索引")

        history_rows = []
        aligned_market = market_data.loc[aligned_index]
        aligned_signals = signals.loc[aligned_index]

        for timestamp in aligned_index:
            price = float(aligned_market.loc[timestamp, price_field])
            signal = int(aligned_signals.loc[timestamp, "signal"])

            self.adapter.mark_to_market(self.symbol, price, timestamp)
            snapshot, drawdown = self._drawdown_state(timestamp)
            current_position = self.adapter.get_position(self.symbol)
            rebalanced = False
            rebalance_reason = "hold"
            target_quantity = float(current_position.quantity)
            order = None
            fill = None

            scheduled = self.calendar.should_rebalance(timestamp, self._last_rebalance_at)
            if self._paused and abs(current_position.quantity) > 1e-12:
                rebalance_reason = "risk_pause"
                target_quantity = 0.0
            elif self._paused:
                rebalance_reason = "paused"
            elif scheduled:
                rebalance_reason = self.calendar.rebalance_reason(timestamp, self._last_rebalance_at)
                target_quantity = self._calculate_target_quantity(signal, snapshot.equity, price)

            if rebalance_reason not in {"hold", "paused"}:
                if not self._should_skip_trade(float(current_position.quantity), target_quantity, price, snapshot.equity):
                    order = self.adapter.create_target_order(
                        symbol=self.symbol,
                        target_quantity=target_quantity,
                        timestamp=timestamp,
                        requested_price=price,
                        signal=signal,
                        metadata={
                            "market_strength": float(aligned_signals.loc[timestamp].get("market_strength", 0.0)),
                            "risk_level": aligned_signals.loc[timestamp].get("risk_level", "unknown"),
                            "rebalance_reason": rebalance_reason,
                            "market_profile": self.paper_config.get("market_profile", "default"),
                        },
                    )
                    fill = self.adapter.execute_order(order, price, timestamp) if order else None
                    rebalanced = True if order else False
                else:
                    rebalance_reason = "turnover_skip"
                self._last_rebalance_at = timestamp

            self.adapter.mark_to_market(self.symbol, price, timestamp)
            history_rows.append(
                self._build_history_row(
                    timestamp,
                    price,
                    signal,
                    target_quantity,
                    order,
                    fill,
                    rebalanced,
                    rebalance_reason,
                    drawdown,
                )
            )

        if self.paper_config.get("close_positions_on_finish", True):
            last_timestamp = aligned_index[-1]
            last_price = float(aligned_market.loc[last_timestamp, price_field])
            closing_order = self.adapter.create_target_order(
                symbol=self.symbol,
                target_quantity=0.0,
                timestamp=last_timestamp,
                requested_price=last_price,
                signal=0,
                metadata={"reason": "close_positions_on_finish"},
            )
            if closing_order:
                closing_fill = self.adapter.execute_order(closing_order, last_price, last_timestamp)
                self.adapter.mark_to_market(self.symbol, last_price, last_timestamp)
                _closing_snapshot, closing_drawdown = self._drawdown_state(last_timestamp)
                history_rows.append(
                    self._build_history_row(
                        last_timestamp,
                        last_price,
                        0,
                        0.0,
                        closing_order,
                        closing_fill,
                        True,
                        "finish_close",
                        closing_drawdown,
                    )
                )

        account_history = pd.DataFrame(history_rows)
        if not account_history.empty:
            account_history["timestamp"] = pd.to_datetime(account_history["timestamp"])
            account_history = account_history.set_index("timestamp")

        return {
            "orders": [order.to_dict() for order in self.adapter.orders],
            "fills": [fill.to_dict() for fill in self.adapter.fills],
            "account_history": account_history,
            "positions": self.adapter.get_positions(),
            "summary": self._summarize(account_history),
        }

    def save_results(self, results: Dict[str, Any], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        pd.DataFrame(results.get("orders", [])).to_csv(
            os.path.join(output_dir, "paper_orders.csv"), index=False, encoding="utf-8"
        )
        pd.DataFrame(results.get("fills", [])).to_csv(
            os.path.join(output_dir, "paper_fills.csv"), index=False, encoding="utf-8"
        )

        account_history = results.get("account_history")
        if isinstance(account_history, pd.DataFrame) and not account_history.empty:
            account_history.to_csv(os.path.join(output_dir, "paper_account_history.csv"), encoding="utf-8")

        with open(os.path.join(output_dir, "paper_positions.json"), "w", encoding="utf-8") as file:
            json.dump(results.get("positions", {}), file, ensure_ascii=False, indent=2, default=str)

        with open(os.path.join(output_dir, "paper_summary.json"), "w", encoding="utf-8") as file:
            json.dump(results.get("summary", {}), file, ensure_ascii=False, indent=2, default=str)
