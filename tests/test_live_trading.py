import unittest

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover
    np = None
    pd = None


@unittest.skipUnless(pd is not None and np is not None, "需要安装 pandas 和 numpy 才能运行 live trading 测试")
class TestLiveTrading(unittest.TestCase):
    def _build_market_frame(self, timestamps, prices):
        close = np.asarray(prices, dtype=float)
        index = pd.to_datetime(list(timestamps))
        return pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.linspace(1000, 2000, len(close)),
            },
            index=index,
        )

    def _build_signal_frame(self, index, signals, strengths=None):
        if strengths is None:
            strengths = [1.0] * len(index)
        return pd.DataFrame(
            {
                "signal": list(signals),
                "market_strength": [float(value) for value in strengths],
                "risk_level": ["low"] * len(index),
            },
            index=index,
        )

    def _build_config(self, **overrides):
        config = {
            "data": {"symbol": "live.us"},
            "strategy": {"allow_short": True},
            "risk": {"commission": 0.0, "slippage": 0.0},
            "execution_costs": {"fixed_commission": 0.0, "min_commission": 0.0, "sell_tax_rate": 0.0},
            "live_trading": {
                "enabled": True,
                "initial_cash": 100000.0,
                "allocation_pct": 0.5,
                "allow_fractional": False,
                "lot_size": 1.0,
                "price_field": "close",
                "session_start": "09:30",
                "session_end": "15:00",
                "exit_only_start": "14:55",
                "cancel_after_seconds": 300,
                "max_order_retries": 2,
                "fill_delay_seconds": 0,
                "reject_first_n_orders": 0,
                "close_positions_on_finish": False,
            },
            "live_risk": {
                "max_order_notional": 0.0,
                "max_position_notional": 0.0,
                "max_daily_drawdown": 0.0,
                "max_open_orders": 10,
                "max_orders_per_day": 0,
                "max_consecutive_failures": 0,
            },
        }
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                merged = dict(config[key])
                merged.update(value)
                config[key] = merged
            else:
                config[key] = value
        return config

    def test_live_trading_respects_session_boundaries(self):
        from src.execution.live_trading_engine import LiveTradingEngine

        timestamps = [
            "2024-01-02 08:50:00",
            "2024-01-02 09:35:00",
            "2024-01-02 15:10:00",
        ]
        market_data = self._build_market_frame(timestamps, [100.0, 100.0, 100.0])
        signals = self._build_signal_frame(market_data.index, [1, 1, 1])
        engine = LiveTradingEngine(self._build_config())

        results = engine.run(market_data, signals, symbol="live.us")
        history = results["account_history"]

        self.assertEqual(history.iloc[0]["session_state"], "closed")
        self.assertEqual(history.iloc[0]["decision_reason"], "session_closed")
        self.assertEqual(history.iloc[1]["session_state"], "open")
        self.assertEqual(history.iloc[1]["order_status"], "filled")
        self.assertEqual(history.iloc[2]["session_state"], "closed")
        self.assertEqual(len(results["orders"]), 1)

    def test_live_trading_exit_only_flattens_without_reversing(self):
        from src.execution.live_trading_engine import LiveTradingEngine

        timestamps = [
            "2024-01-02 10:00:00",
            "2024-01-02 14:56:00",
            "2024-01-02 14:57:00",
        ]
        market_data = self._build_market_frame(timestamps, [100.0, 100.0, 100.0])
        signals = self._build_signal_frame(market_data.index, [1, -1, -1])
        engine = LiveTradingEngine(self._build_config())

        results = engine.run(market_data, signals, symbol="live.us")
        history = results["account_history"]

        self.assertEqual(history.iloc[1]["decision_reason"], "exit_only_flatten")
        self.assertEqual(history.iloc[1]["order_status"], "filled")
        self.assertEqual(history.iloc[2]["decision_reason"], "exit_only_no_entry")
        self.assertEqual(results["summary"]["fills"], 2)
        self.assertFalse(results["positions"])

    def test_live_trading_retries_after_rejection(self):
        from src.execution.live_trading_engine import LiveTradingEngine

        timestamps = [
            "2024-01-02 10:00:00",
            "2024-01-02 10:01:00",
            "2024-01-02 10:02:00",
        ]
        market_data = self._build_market_frame(timestamps, [100.0, 100.0, 100.0])
        signals = self._build_signal_frame(market_data.index, [1, 1, 1])
        engine = LiveTradingEngine(
            self._build_config(
                live_trading={
                    "reject_first_n_orders": 1,
                    "max_order_retries": 1,
                }
            )
        )

        results = engine.run(market_data, signals, symbol="live.us")
        history = results["account_history"]

        self.assertEqual(history.iloc[0]["decision_reason"], "order_rejected")
        self.assertEqual(history.iloc[1]["order_status"], "filled")
        self.assertEqual(results["summary"]["rejected_orders"], 1)
        self.assertEqual(results["summary"]["fills"], 1)
        self.assertEqual(len(results["orders"]), 2)

    def test_live_trading_cancels_stale_order_and_resubmits(self):
        from src.execution.live_trading_engine import LiveTradingEngine

        timestamps = [
            "2024-01-02 10:00:00",
            "2024-01-02 10:10:00",
            "2024-01-02 10:30:00",
        ]
        market_data = self._build_market_frame(timestamps, [100.0, 100.0, 100.0])
        signals = self._build_signal_frame(market_data.index, [1, 1, 1])
        engine = LiveTradingEngine(
            self._build_config(
                live_trading={
                    "fill_delay_seconds": 900,
                    "cancel_after_seconds": 300,
                }
            )
        )

        results = engine.run(market_data, signals, symbol="live.us")
        history = results["account_history"]
        orders = pd.DataFrame(results["orders"])

        self.assertEqual(int(history.iloc[1]["canceled_orders"]), 1)
        self.assertEqual(results["summary"]["canceled_orders"], 1)
        self.assertEqual(results["summary"]["fills"], 1)
        self.assertEqual(len(orders), 2)
        self.assertEqual(int((orders["status"] == "canceled").sum()), 1)
        self.assertEqual(int((orders["status"] == "filled").sum()), 1)

    def test_live_trading_flattens_on_daily_drawdown_limit(self):
        from src.execution.live_trading_engine import LiveTradingEngine

        timestamps = [
            "2024-01-02 10:00:00",
            "2024-01-02 10:05:00",
        ]
        market_data = self._build_market_frame(timestamps, [100.0, 90.0])
        signals = self._build_signal_frame(market_data.index, [1, 1])
        engine = LiveTradingEngine(
            self._build_config(
                live_risk={
                    "max_daily_drawdown": 0.03,
                }
            )
        )

        results = engine.run(market_data, signals, symbol="live.us")
        history = results["account_history"]

        self.assertEqual(history.iloc[1]["decision_reason"], "risk_flatten")
        self.assertEqual(history.iloc[1]["order_status"], "filled")
        self.assertEqual(results["positions"]["live.us"]["quantity"], 0.0)


if __name__ == "__main__":
    unittest.main()
