"""Paper broker: 负责订单成交、持仓和账户权益状态。"""

import math
from itertools import count
from typing import Any, Dict, Optional

from .cost_model import ExecutionCostModel
from .market_profile import MarketProfileResolver
from .models import AccountSnapshot, Fill, Order, Position


class PaperBroker:
    def __init__(
        self,
        initial_cash: float,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        allow_short: bool = True,
        allow_fractional: bool = False,
        lot_size: float = 1.0,
        cost_model: Optional[ExecutionCostModel] = None,
        market_resolver: Optional[MarketProfileResolver] = None,
        market_section: str = "paper_trading",
    ):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.allow_short = bool(allow_short)
        self.allow_fractional = bool(allow_fractional)
        self.lot_size = float(lot_size) if lot_size else 1.0
        self.cost_model = cost_model or ExecutionCostModel(
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
        )
        self.market_resolver = market_resolver
        self.market_section = market_section

        self.positions: Dict[str, Position] = {}
        self.orders = []
        self.fills = []
        self.realized_pnl = 0.0
        self.fees_paid = 0.0

        self._order_counter = count(1)
        self._fill_counter = count(1)

    def _rules_for(self, symbol: str) -> Dict[str, Any]:
        if self.market_resolver is None:
            return {
                "allow_short": bool(self.allow_short),
                "allow_fractional": bool(self.allow_fractional),
                "lot_size": float(self.lot_size),
            }
        return self.market_resolver.section_settings(symbol=symbol, section_name=self.market_section)

    def _normalize_quantity(self, quantity: float, symbol: Optional[str] = None) -> float:
        quantity = abs(float(quantity))
        if quantity == 0:
            return 0.0

        rules = self._rules_for(symbol or "")
        step = float(rules.get("lot_size", self.lot_size) or 1.0)
        if bool(rules.get("allow_fractional", self.allow_fractional)):
            normalized = round(quantity / step) * step
        else:
            normalized = math.floor(quantity / step) * step
        return float(normalized if normalized > 0 else 0.0)

    def _get_or_create_position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def get_position(self, symbol: str) -> Position:
        return self._get_or_create_position(symbol)

    def create_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        timestamp: Any,
        requested_price: Optional[float] = None,
        signal: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Order]:
        normalized_quantity = self._normalize_quantity(quantity, symbol=symbol)
        if normalized_quantity <= 0:
            return None

        order = Order(
            order_id=f"ORD-{next(self._order_counter):06d}",
            symbol=symbol,
            side=side,
            quantity=normalized_quantity,
            submitted_at=timestamp,
            requested_price=requested_price,
            signal=signal,
            updated_at=timestamp,
            metadata=metadata or {},
        )
        self.orders.append(order)
        return order

    def create_target_order(
        self,
        symbol: str,
        target_quantity: float,
        timestamp: Any,
        requested_price: Optional[float] = None,
        signal: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Order]:
        current_position = self.get_position(symbol)
        current_quantity = float(current_position.quantity)
        target_quantity = float(target_quantity)
        rules = self._rules_for(symbol)

        if not bool(rules.get("allow_short", self.allow_short)) and target_quantity < 0:
            target_quantity = 0.0

        delta = target_quantity - current_quantity
        if abs(delta) < 1e-9:
            return None

        side = "buy" if delta > 0 else "sell"
        return self.create_order(
            symbol=symbol,
            side=side,
            quantity=abs(delta),
            timestamp=timestamp,
            requested_price=requested_price,
            signal=signal,
            metadata=metadata,
        )

    def mark_to_market(self, symbol: str, price: float, timestamp: Any) -> Position:
        position = self.get_position(symbol)
        price = float(price)
        position.last_price = price
        position.updated_at = timestamp

        if abs(position.quantity) < 1e-12:
            position.market_value = 0.0
            position.unrealized_pnl = 0.0
            return position

        position.market_value = float(position.quantity * price)
        if position.quantity > 0:
            position.unrealized_pnl = float((price - position.avg_price) * position.quantity)
        else:
            position.unrealized_pnl = float((position.avg_price - price) * abs(position.quantity))
        return position

    def execute_order(self, order: Order, market_price: float, timestamp: Any) -> Fill:
        market_price = float(market_price)
        quantity = float(order.quantity)
        direction = 1 if order.side == "buy" else -1
        signed_quantity = quantity * direction
        rules = self._rules_for(order.symbol)

        if direction < 0 and not bool(rules.get("allow_short", self.allow_short)) and self.get_position(order.symbol).quantity <= 0:
            raise ValueError("当前 broker 未启用做空")

        costs = self.cost_model.estimate(order.side, market_price, quantity)
        fill_price = float(costs.fill_price)
        notional = float(costs.notional)
        total_fees = float(costs.total_fees)

        position = self.get_position(order.symbol)
        prev_quantity = float(position.quantity)
        prev_avg_price = float(position.avg_price)
        new_quantity = prev_quantity + signed_quantity
        gross_realized_pnl = 0.0

        if abs(prev_quantity) < 1e-12:
            new_avg_price = fill_price
            position.opened_at = timestamp
        elif prev_quantity * signed_quantity > 0:
            total_quantity = abs(prev_quantity) + abs(signed_quantity)
            weighted_cost = abs(prev_quantity) * prev_avg_price + abs(signed_quantity) * fill_price
            new_avg_price = weighted_cost / total_quantity
        else:
            closing_quantity = min(abs(prev_quantity), abs(signed_quantity))
            if prev_quantity > 0:
                gross_realized_pnl = (fill_price - prev_avg_price) * closing_quantity
            else:
                gross_realized_pnl = (prev_avg_price - fill_price) * closing_quantity

            if abs(new_quantity) < 1e-12:
                new_avg_price = 0.0
                position.opened_at = None
                new_quantity = 0.0
            elif prev_quantity * new_quantity > 0:
                new_avg_price = prev_avg_price
            else:
                new_avg_price = fill_price
                position.opened_at = timestamp

        net_realized_pnl = float(gross_realized_pnl - total_fees)
        if direction > 0:
            self.cash -= notional + total_fees
        else:
            self.cash += notional - total_fees

        self.realized_pnl += net_realized_pnl
        self.fees_paid += total_fees
        position.quantity = float(new_quantity)
        position.avg_price = float(new_avg_price)
        position.realized_pnl += net_realized_pnl
        position.updated_at = timestamp
        self.mark_to_market(order.symbol, fill_price, timestamp)

        order.status = "filled"
        order.filled_quantity = float(quantity)
        order.avg_fill_price = float(fill_price)
        order.updated_at = timestamp
        order.status_reason = "filled"
        order.last_error = None
        fill = Fill(
            fill_id=f"FIL-{next(self._fill_counter):06d}",
            order_id=order.order_id,
            broker_order_id=order.broker_order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            fill_price=fill_price,
            commission=float(costs.commission),
            slippage=float(costs.slippage),
            tax=float(costs.tax),
            total_fees=total_fees,
            notional=notional,
            filled_at=timestamp,
            realized_pnl=net_realized_pnl,
            gross_realized_pnl=float(gross_realized_pnl),
            metadata=dict(order.metadata),
        )
        self.fills.append(fill)
        return fill

    def get_account_snapshot(self, timestamp: Any) -> AccountSnapshot:
        market_value = float(sum(position.market_value for position in self.positions.values()))
        unrealized_pnl = float(sum(position.unrealized_pnl for position in self.positions.values()))
        gross_exposure = float(sum(abs(position.market_value) for position in self.positions.values()))
        active_positions = int(sum(1 for position in self.positions.values() if abs(position.quantity) > 1e-12))
        equity = float(self.cash + market_value)
        return AccountSnapshot(
            timestamp=timestamp,
            cash=float(self.cash),
            market_value=market_value,
            equity=equity,
            realized_pnl=float(self.realized_pnl),
            unrealized_pnl=unrealized_pnl,
            gross_exposure=gross_exposure,
            net_exposure=market_value,
            active_positions=active_positions,
            fees_paid=float(self.fees_paid),
        )

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        return {
            symbol: position.to_dict()
            for symbol, position in self.positions.items()
            if abs(position.quantity) > 1e-12 or abs(position.realized_pnl) > 1e-12
        }
