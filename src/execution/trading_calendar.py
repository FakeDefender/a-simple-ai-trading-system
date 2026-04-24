"""调仓日历与调仓频率判断。"""

from typing import Any, Optional

import pandas as pd


class TradingCalendar:
    def __init__(
        self,
        rebalance_frequency: str = "daily",
        rebalance_weekday: int = 0,
        rebalance_day_of_month: int = 1,
    ):
        self.rebalance_frequency = str(rebalance_frequency or "daily").lower()
        self.rebalance_weekday = int(rebalance_weekday)
        self.rebalance_day_of_month = int(rebalance_day_of_month)

    def is_trading_day(self, timestamp: Any) -> bool:
        ts = pd.Timestamp(timestamp)
        return ts.weekday() < 5

    def should_rebalance(self, timestamp: Any, last_rebalance_at: Optional[Any] = None) -> bool:
        ts = pd.Timestamp(timestamp)
        if last_rebalance_at is None:
            return True

        last_ts = pd.Timestamp(last_rebalance_at)
        frequency = self.rebalance_frequency
        if frequency == "daily":
            return ts.normalize() > last_ts.normalize()

        if frequency == "weekly":
            same_week = ts.isocalendar().year == last_ts.isocalendar().year and ts.isocalendar().week == last_ts.isocalendar().week
            if same_week:
                return False
            return ts.weekday() >= self.rebalance_weekday

        if frequency == "monthly":
            same_month = ts.year == last_ts.year and ts.month == last_ts.month
            if same_month:
                return False
            return ts.day >= self.rebalance_day_of_month

        return ts.normalize() > last_ts.normalize()

    def rebalance_reason(self, timestamp: Any, last_rebalance_at: Optional[Any] = None) -> str:
        if not self.should_rebalance(timestamp, last_rebalance_at):
            return "hold"
        frequency = self.rebalance_frequency
        if frequency == "daily":
            return "scheduled_daily"
        if frequency == "weekly":
            return "scheduled_weekly"
        if frequency == "monthly":
            return "scheduled_monthly"
        return "scheduled"
