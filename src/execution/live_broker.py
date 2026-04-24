"""面向实盘骨架的 paper live broker。"""

from itertools import count
from typing import Any, Dict, List, Optional

import pandas as pd

from .cost_model import ExecutionCostModel
from .market_profile import MarketProfileResolver
from .paper_broker import PaperBroker
from .models import AccountSnapshot, Fill, Order, Position


class PaperLiveBroker:
    def __init__(
        self,
        initial_cash: float,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        allow_short: bool = True,
        allow_fractional: bool = False,
        lot_size: float = 1.0,
        cost_model: Optional[ExecutionCostModel] = None,
        fill_delay_seconds: int = 0,
        reject_first_n_orders: int = 0,
        broker: Optional[PaperBroker] = None,
        market_resolver: Optional[MarketProfileResolver] = None,
        market_section: str = "live_trading",
    ):
        self.paper_broker = broker or PaperBroker(
            initial_cash=initial_cash,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            allow_short=allow_short,
            allow_fractional=allow_fractional,
            lot_size=lot_size,
            cost_model=cost_model,
            market_resolver=market_resolver,
            market_section=market_section,
        )
        self.fill_delay_seconds = max(int(fill_delay_seconds), 0)
        self.reject_first_n_orders = max(int(reject_first_n_orders), 0)

        self._open_orders: Dict[str, Order] = {}
        self._broker_order_counter = count(1)
        self._rejected_submissions = 0

    @property
    def initial_cash(self) -> float:
        return float(self.paper_broker.initial_cash)

    @property
    def orders(self):
        return self.paper_broker.orders

    @property
    def fills(self):
        return self.paper_broker.fills

    def create_target_order(self, *args, **kwargs):
        return self.paper_broker.create_target_order(*args, **kwargs)

    def mark_to_market(self, *args, **kwargs) -> Position:
        return self.paper_broker.mark_to_market(*args, **kwargs)

    def get_account_snapshot(self, timestamp: Any) -> AccountSnapshot:
        snapshot = self.paper_broker.get_account_snapshot(timestamp)
        snapshot.open_orders = len(self._open_orders)
        return snapshot

    def get_position(self, *args, **kwargs) -> Position:
        return self.paper_broker.get_position(*args, **kwargs)

    def get_positions(self, *args, **kwargs):
        return self.paper_broker.get_positions(*args, **kwargs)

    def get_order(self, order_id: str) -> Optional[Order]:
        for order in self.paper_broker.orders:
            if order.order_id == order_id:
                return order
        return None

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        orders = list(self._open_orders.values())
        if symbol is not None:
            orders = [order for order in orders if order.symbol == symbol]
        return orders

    def submit_order(self, order: Order, market_price: float, timestamp: Any) -> Order:
        order.updated_at = timestamp
        order.attempt_count = int(order.attempt_count) + 1
        if not order.broker_order_id:
            order.broker_order_id = f"BRO-{next(self._broker_order_counter):06d}"

        if self._rejected_submissions < self.reject_first_n_orders:
            self._rejected_submissions += 1
            order.status = "rejected"
            order.rejected_at = timestamp
            order.status_reason = "simulated_reject"
            order.last_error = "simulated_reject"
            return order

        order.status = "submitted"
        order.status_reason = "submitted"
        order.last_error = None
        self._open_orders[order.order_id] = order
        return order

    def cancel_order(self, order_id: str, timestamp: Any, reason: str = "canceled") -> Optional[Order]:
        order = self._open_orders.pop(order_id, None)
        if order is None:
            return None

        order.status = "canceled"
        order.canceled_at = timestamp
        order.updated_at = timestamp
        order.status_reason = reason
        return order

    def process_pending_orders(self, market_prices: Dict[str, float], timestamp: Any) -> List[Fill]:
        fills: List[Fill] = []
        current_timestamp = pd.Timestamp(timestamp)
        for order in list(self._open_orders.values()):
            order_timestamp = pd.Timestamp(order.submitted_at)
            if (current_timestamp - order_timestamp).total_seconds() < self.fill_delay_seconds:
                continue

            market_price = market_prices.get(order.symbol)
            if market_price is None:
                continue

            fill = self.paper_broker.execute_order(order, float(market_price), timestamp)
            self._open_orders.pop(order.order_id, None)
            fills.append(fill)
        return fills
