"""Live trading 风控。"""

from typing import Any, Dict, Optional

import pandas as pd


class LiveRiskManager:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        settings = dict(config or {})
        self.max_order_notional = float(settings.get("max_order_notional", 0.0) or 0.0)
        self.max_position_notional = float(settings.get("max_position_notional", 0.0) or 0.0)
        self.max_daily_drawdown = float(settings.get("max_daily_drawdown", 0.0) or 0.0)
        self.max_open_orders = int(settings.get("max_open_orders", 10) or 10)
        self.max_orders_per_day = int(settings.get("max_orders_per_day", 0) or 0)
        self.max_consecutive_failures = int(settings.get("max_consecutive_failures", 0) or 0)

        self._current_day = None
        self._daily_start_equity = None
        self._daily_orders = 0
        self._consecutive_failures = 0

    @property
    def daily_orders(self) -> int:
        return int(self._daily_orders)

    @property
    def consecutive_failures(self) -> int:
        return int(self._consecutive_failures)

    def _roll_day(self, timestamp: Any, equity: float):
        current_day = pd.Timestamp(timestamp).normalize()
        if self._current_day is None or current_day != self._current_day:
            self._current_day = current_day
            self._daily_start_equity = float(equity)
            self._daily_orders = 0

    def evaluate(self, snapshot, timestamp: Any) -> Dict[str, Any]:
        equity = float(snapshot.equity)
        self._roll_day(timestamp, equity)

        start_equity = float(self._daily_start_equity or equity or 0.0)
        daily_drawdown = 0.0 if start_equity <= 0 else max(0.0, 1 - equity / start_equity)
        hard_halt = False
        reason = "ok"

        if self.max_daily_drawdown > 0 and daily_drawdown >= self.max_daily_drawdown:
            hard_halt = True
            reason = "daily_drawdown_limit"
        elif self.max_consecutive_failures > 0 and self._consecutive_failures >= self.max_consecutive_failures:
            hard_halt = True
            reason = "consecutive_failure_limit"

        return {
            "hard_halt": bool(hard_halt),
            "reason": reason,
            "daily_drawdown": float(daily_drawdown),
            "daily_orders": int(self._daily_orders),
            "consecutive_failures": int(self._consecutive_failures),
        }

    def validate_order(
        self,
        current_quantity: float,
        target_quantity: float,
        price: float,
        open_order_count: int,
    ) -> Optional[str]:
        price = float(price)
        current_quantity = float(current_quantity)
        target_quantity = float(target_quantity)
        if price <= 0:
            return "invalid_price"

        if self.max_open_orders > 0 and int(open_order_count) >= self.max_open_orders:
            return "too_many_open_orders"

        if self.max_orders_per_day > 0 and self._daily_orders >= self.max_orders_per_day:
            return "daily_order_limit"

        current_notional = abs(current_quantity) * price
        target_notional = abs(target_quantity) * price
        delta_notional = abs(target_quantity - current_quantity) * price
        increases_exposure = target_notional > current_notional + 1e-12 or (current_quantity * target_quantity < 0 and target_notional > 0)

        if increases_exposure and self.max_order_notional > 0 and delta_notional > self.max_order_notional:
            return "order_notional_limit"

        if increases_exposure and self.max_position_notional > 0 and target_notional > self.max_position_notional:
            return "position_notional_limit"

        return None

    def record_submission(self, timestamp: Any):
        if self._current_day is None:
            self._current_day = pd.Timestamp(timestamp).normalize()
        self._daily_orders += 1

    def record_failure(self):
        self._consecutive_failures += 1

    def record_success(self):
        self._consecutive_failures = 0
