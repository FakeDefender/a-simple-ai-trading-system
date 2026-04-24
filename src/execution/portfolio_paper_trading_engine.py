"""多标的组合 paper trading engine。"""

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .broker_adapter import PaperBrokerAdapter, build_paper_adapter
from .market_profile import MarketProfileResolver
from .paper_broker import PaperBroker
from .trading_calendar import TradingCalendar


DEFAULT_PORTFOLIO_CONFIG = {
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
}


class PortfolioPaperTradingEngine:
    def __init__(self, config: Dict[str, Any], adapter: Optional[PaperBrokerAdapter] = None, broker: Optional[PaperBroker] = None):
        self.config = config or {}
        self.market_resolver = MarketProfileResolver(self.config)
        self.portfolio_config = dict(DEFAULT_PORTFOLIO_CONFIG)
        self.portfolio_config.update(self.config.get("portfolio", {}))

        if adapter is not None:
            self.adapter = adapter
        elif broker is not None:
            self.adapter = PaperBrokerAdapter(broker)
        else:
            self.adapter = build_paper_adapter(self.config)

        self.calendar = TradingCalendar(
            rebalance_frequency=self.portfolio_config.get("rebalance_frequency", "daily"),
            rebalance_weekday=int(self.portfolio_config.get("rebalance_weekday", 0)),
            rebalance_day_of_month=int(self.portfolio_config.get("rebalance_day_of_month", 1)),
        )
        self._paused = False
        self._peak_equity = float(self.adapter.initial_cash)
        self._last_rebalance_at = None

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
    ) -> List[Tuple[str, float, int, float]]:
        candidates: List[Tuple[str, float, int, float]] = []
        for symbol, market_data in market_data_by_symbol.items():
            signal_row = signals_by_symbol[symbol].loc[timestamp]
            signal = int(signal_row.get("signal", 0))
            market_profile = self.market_resolver.resolve(symbol=symbol, section_name="paper_trading")
            if signal == 0 or (signal < 0 and not market_profile.allow_short):
                continue
            price = float(market_data.loc[timestamp, self.portfolio_config.get("price_field", "close")])
            if price <= 0:
                continue
            score = self._score_symbol(signal_row)
            candidates.append((symbol, score, signal, price))

        candidates.sort(key=lambda item: item[1], reverse=True)
        max_positions = int(self.portfolio_config.get("max_positions", 3))
        return candidates[:max_positions]

    def _target_quantities(
        self,
        timestamp: Any,
        selected_symbols: List[Tuple[str, float, int, float]],
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

        for symbol, _score, signal, price in selected_symbols:
            market_profile = self.market_resolver.resolve(symbol=symbol, section_name="paper_trading")
            raw_quantity = equity * per_symbol_weight / price
            if market_profile.allow_fractional:
                raw_quantity = round(raw_quantity, market_profile.quantity_precision)
            else:
                raw_quantity = math.floor(raw_quantity / market_profile.lot_size) * market_profile.lot_size
            if signal < 0 and not market_profile.allow_short:
                continue
            targets[symbol] = float(raw_quantity * signal)
        return targets

    def _should_skip_trade(self, current_quantity: float, target_quantity: float, price: float, equity: float) -> bool:
        if equity <= 0:
            return True
        delta_notional = abs(target_quantity - current_quantity) * price
        threshold = float(self.portfolio_config.get("turnover_buffer", 0.0))
        return delta_notional / equity < threshold

    def _build_account_row(
        self,
        timestamp: Any,
        selected_symbols: List[str],
        rebalanced: bool,
        rebalance_reason: str,
        drawdown: float,
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
            }
        )
        return snapshot

    def _build_symbol_rows(
        self,
        timestamp: Any,
        targets: Dict[str, float],
        market_data_by_symbol: Dict[str, pd.DataFrame],
        signals_by_symbol: Dict[str, pd.DataFrame],
        selected_lookup: Dict[str, float],
        rebalanced: bool,
        rebalance_reason: str,
    ) -> List[Dict[str, Any]]:
        rows = []
        for symbol, market_data in market_data_by_symbol.items():
            signal_row = signals_by_symbol[symbol].loc[timestamp]
            signal = int(signal_row.get("signal", 0))
            price = float(market_data.loc[timestamp, self.portfolio_config.get("price_field", "close")])
            position = self.adapter.get_position(symbol)
            market_profile = self.market_resolver.resolve(symbol=symbol, section_name="paper_trading")
            if rebalance_reason in {"hold", "paused"}:
                target_quantity = float(targets.get(symbol, float(position.quantity)))
            elif rebalance_reason == "risk_pause":
                target_quantity = 0.0
            else:
                target_quantity = float(targets.get(symbol, 0.0))
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "market_profile": market_profile.name,
                    "price": price,
                    "signal": signal,
                    "market_strength": float(signal_row.get("market_strength", 0.5)),
                    "risk_level": signal_row.get("risk_level", "unknown"),
                    "selected": symbol in selected_lookup,
                    "selection_score": float(selected_lookup.get(symbol, 0.0)),
                    "target_quantity": target_quantity,
                    "position_quantity": float(position.quantity),
                    "position_side": position.side,
                    "avg_entry_price": float(position.avg_price),
                    "market_value": float(position.market_value),
                    "unrealized_pnl": float(position.unrealized_pnl),
                    "rebalanced": bool(rebalanced),
                    "rebalance_reason": rebalance_reason,
                }
            )
        return rows

    def _summarize(self, account_history: pd.DataFrame, symbol_history: pd.DataFrame) -> Dict[str, Any]:
        if account_history.empty:
            return {
                "initial_equity": float(self.adapter.initial_cash),
                "final_equity": float(self.adapter.initial_cash),
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "orders": 0,
                "fills": 0,
                "active_symbols": 0,
                "paused": bool(self._paused),
                "fees_paid": 0.0,
                "rebalances": 0,
            }

        equity = account_history["equity"].astype(float)
        baseline = pd.Series([float(self.adapter.initial_cash)])
        equity_with_baseline = pd.concat([baseline, equity.reset_index(drop=True)], ignore_index=True)
        drawdown = equity_with_baseline / equity_with_baseline.cummax() - 1
        return {
            "initial_equity": float(self.adapter.initial_cash),
            "final_equity": float(equity.iloc[-1]),
            "total_return": float(equity.iloc[-1] / float(self.adapter.initial_cash) - 1),
            "max_drawdown": float(abs(drawdown.min())),
            "orders": len(self.adapter.orders),
            "fills": len(self.adapter.fills),
            "active_symbols": int(symbol_history["symbol"].nunique()) if not symbol_history.empty else 0,
            "paused": bool(self._paused),
            "realized_pnl": float(account_history["realized_pnl"].iloc[-1]),
            "unrealized_pnl": float(account_history["unrealized_pnl"].iloc[-1]),
            "fees_paid": float(account_history["fees_paid"].iloc[-1]),
            "rebalances": int(account_history["rebalanced"].sum()),
        }

    def _symbol_summary(self) -> List[Dict[str, Any]]:
        results = []
        grouped_fills: Dict[str, List[float]] = {}
        for fill in self.adapter.fills:
            grouped_fills.setdefault(fill.symbol, []).append(float(fill.realized_pnl))

        positions = self.adapter.get_positions()
        for symbol, position in positions.items():
            pnl_series = grouped_fills.get(symbol, [])
            market_profile = self.market_resolver.resolve(symbol=symbol, section_name="paper_trading")
            results.append(
                {
                    "symbol": symbol,
                    "market_profile": market_profile.name,
                    "final_quantity": float(position.get("quantity", 0.0)),
                    "avg_price": float(position.get("avg_price", 0.0)),
                    "market_value": float(position.get("market_value", 0.0)),
                    "realized_pnl": float(position.get("realized_pnl", 0.0)),
                    "unrealized_pnl": float(position.get("unrealized_pnl", 0.0)),
                    "fills": int(sum(1 for fill in self.adapter.fills if fill.symbol == symbol)),
                    "win_rate": float(sum(1 for pnl in pnl_series if pnl > 0) / len(pnl_series)) if pnl_series else 0.0,
                }
            )
        return results

    def run(
        self,
        market_data_by_symbol: Dict[str, pd.DataFrame],
        signals_by_symbol: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        if not market_data_by_symbol:
            raise ValueError("市场数据为空，无法运行组合 paper trading")
        if not signals_by_symbol:
            raise ValueError("信号数据为空，无法运行组合 paper trading")

        common_index = self._common_index(market_data_by_symbol, signals_by_symbol)
        account_rows: List[Dict[str, Any]] = []
        symbol_rows: List[Dict[str, Any]] = []

        for timestamp in common_index:
            for symbol, market_data in market_data_by_symbol.items():
                price = float(market_data.loc[timestamp, self.portfolio_config.get("price_field", "close")])
                self.adapter.mark_to_market(symbol, price, timestamp)

            snapshot, drawdown = self._drawdown_state(timestamp)
            selected: List[Tuple[str, float, int, float]] = []
            targets: Dict[str, float] = {}
            selected_lookup: Dict[str, float] = {}
            rebalanced = False
            rebalance_reason = "hold"

            scheduled = self.calendar.should_rebalance(timestamp, self._last_rebalance_at)
            active_positions = bool(self.adapter.get_positions())
            if self._paused and active_positions:
                rebalance_reason = "risk_pause"
            elif self._paused:
                rebalance_reason = "paused"
            elif scheduled:
                rebalance_reason = self.calendar.rebalance_reason(timestamp, self._last_rebalance_at)
                selected = self._select_symbols(timestamp, market_data_by_symbol, signals_by_symbol)
                targets = self._target_quantities(timestamp, selected)
                selected_lookup = {symbol: score for symbol, score, _signal, _price in selected}

            if rebalance_reason not in {"hold", "paused"}:
                for symbol, market_data in market_data_by_symbol.items():
                    price = float(market_data.loc[timestamp, self.portfolio_config.get("price_field", "close")])
                    signal = int(signals_by_symbol[symbol].loc[timestamp].get("signal", 0))
                    current_position = self.adapter.get_position(symbol)
                    target_quantity = 0.0 if rebalance_reason == "risk_pause" else targets.get(symbol, 0.0)
                    if self._should_skip_trade(float(current_position.quantity), target_quantity, price, snapshot.equity):
                        continue
                    order = self.adapter.create_target_order(
                        symbol=symbol,
                        target_quantity=target_quantity,
                        timestamp=timestamp,
                        requested_price=price,
                        signal=signal,
                        metadata={
                            "selection_score": float(selected_lookup.get(symbol, 0.0)),
                            "market_strength": float(signals_by_symbol[symbol].loc[timestamp].get("market_strength", 0.0)),
                            "risk_level": signals_by_symbol[symbol].loc[timestamp].get("risk_level", "unknown"),
                            "rebalance_reason": rebalance_reason,
                            "market_profile": self.market_resolver.resolve(symbol=symbol, section_name="paper_trading").name,
                        },
                    )
                    if order is not None:
                        self.adapter.execute_order(order, price, timestamp)
                        rebalanced = True
                    self.adapter.mark_to_market(symbol, price, timestamp)
                self._last_rebalance_at = timestamp

            account_rows.append(self._build_account_row(timestamp, [item[0] for item in selected], rebalanced, rebalance_reason, drawdown))
            symbol_rows.extend(
                self._build_symbol_rows(
                    timestamp,
                    targets,
                    market_data_by_symbol,
                    signals_by_symbol,
                    selected_lookup,
                    rebalanced,
                    rebalance_reason,
                )
            )

        if self.portfolio_config.get("close_positions_on_finish", True) and len(common_index) > 0:
            last_timestamp = common_index[-1]
            for symbol, market_data in market_data_by_symbol.items():
                last_price = float(market_data.loc[last_timestamp, self.portfolio_config.get("price_field", "close")])
                closing_order = self.adapter.create_target_order(
                    symbol=symbol,
                    target_quantity=0.0,
                    timestamp=last_timestamp,
                    requested_price=last_price,
                    signal=0,
                    metadata={"reason": "portfolio_close_positions_on_finish"},
                )
                if closing_order is not None:
                    self.adapter.execute_order(closing_order, last_price, last_timestamp)
                    self.adapter.mark_to_market(symbol, last_price, last_timestamp)
            if account_rows:
                _final_snapshot, final_drawdown = self._drawdown_state(last_timestamp)
                account_rows.append(self._build_account_row(last_timestamp, [], True, "finish_close", final_drawdown))
                symbol_rows.extend(
                    self._build_symbol_rows(
                        last_timestamp,
                        {},
                        market_data_by_symbol,
                        signals_by_symbol,
                        {},
                        True,
                        "finish_close",
                    )
                )

        account_history = pd.DataFrame(account_rows)
        if not account_history.empty:
            account_history["timestamp"] = pd.to_datetime(account_history["timestamp"])
            account_history = account_history.set_index("timestamp")

        symbol_history = pd.DataFrame(symbol_rows)
        if not symbol_history.empty:
            symbol_history["timestamp"] = pd.to_datetime(symbol_history["timestamp"])
            symbol_history = symbol_history.sort_values(["timestamp", "symbol"])

        return {
            "orders": [order.to_dict() for order in self.adapter.orders],
            "fills": [fill.to_dict() for fill in self.adapter.fills],
            "account_history": account_history,
            "symbol_history": symbol_history,
            "positions": self.adapter.get_positions(),
            "summary": self._summarize(account_history, symbol_history),
            "symbol_summary": self._symbol_summary(),
        }

    def save_results(self, results: Dict[str, Any], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        pd.DataFrame(results.get("orders", [])).to_csv(
            os.path.join(output_dir, "portfolio_orders.csv"), index=False, encoding="utf-8"
        )
        pd.DataFrame(results.get("fills", [])).to_csv(
            os.path.join(output_dir, "portfolio_fills.csv"), index=False, encoding="utf-8"
        )

        account_history = results.get("account_history")
        if isinstance(account_history, pd.DataFrame) and not account_history.empty:
            account_history.to_csv(os.path.join(output_dir, "portfolio_account_history.csv"), encoding="utf-8")

        symbol_history = results.get("symbol_history")
        if isinstance(symbol_history, pd.DataFrame) and not symbol_history.empty:
            symbol_history.to_csv(os.path.join(output_dir, "portfolio_symbol_history.csv"), index=False, encoding="utf-8")

        with open(os.path.join(output_dir, "portfolio_positions.json"), "w", encoding="utf-8") as file:
            json.dump(results.get("positions", {}), file, ensure_ascii=False, indent=2, default=str)

        with open(os.path.join(output_dir, "portfolio_summary.json"), "w", encoding="utf-8") as file:
            json.dump(results.get("summary", {}), file, ensure_ascii=False, indent=2, default=str)

        with open(os.path.join(output_dir, "portfolio_symbol_summary.json"), "w", encoding="utf-8") as file:
            json.dump(results.get("symbol_summary", []), file, ensure_ascii=False, indent=2, default=str)
