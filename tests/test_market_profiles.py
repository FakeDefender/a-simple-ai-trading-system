import unittest

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover
    np = None
    pd = None


@unittest.skipUnless(pd is not None and np is not None, "需要安装 pandas 和 numpy 才能运行市场抽象测试")
class TestMarketProfiles(unittest.TestCase):
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

    def _build_signal_frame(self, index, signal=1, strength=0.95):
        return pd.DataFrame(
            {
                "signal": [signal] * len(index),
                "market_strength": [float(strength)] * len(index),
                "risk_level": ["low"] * len(index),
            },
            index=index,
        )

    def test_cn_market_profile_supports_split_sessions(self):
        from src.execution.market_profile import MarketProfileResolver
        from src.execution.market_session import MarketSession

        config = {
            "data": {"symbol": "600000.sh"},
            "market": {"profile": "cn_equity"},
            "strategy": {"allow_short": False},
        }
        resolver = MarketProfileResolver(config)
        profile = resolver.resolve(symbol="600000.sh", section_name="live_trading")
        session = MarketSession(
            timezone=profile.timezone,
            trading_days=profile.trading_days,
            sessions=profile.to_dict().get("sessions"),
            exit_only_start=profile.exit_only_start,
        )

        self.assertEqual(session.get_state("2024-01-02 10:00:00"), "open")
        self.assertEqual(session.get_state("2024-01-02 11:45:00"), "closed")
        self.assertEqual(session.get_state("2024-01-02 13:05:00"), "open")
        self.assertEqual(session.get_state("2024-01-02 14:58:00"), "exit_only")

    def test_build_paper_adapter_applies_symbol_level_market_rules(self):
        from src.execution.broker_adapter import build_paper_adapter

        config = {
            "data": {"symbol": "aapl.us"},
            "market": {
                "profile": "us_equity",
                "symbol_profiles": {
                    "aapl.us": "us_equity",
                    "600000.sh": "cn_equity",
                },
            },
            "strategy": {"allow_short": True},
            "risk": {"commission": 0.0, "slippage": 0.0},
            "execution_costs": {"fixed_commission": 0.0, "min_commission": 0.0, "sell_tax_rate": 0.0},
            "paper_trading": {
                "initial_cash": 100000.0,
                "allow_fractional": False,
                "lot_size": 1.0,
            },
        }

        adapter = build_paper_adapter(config)
        timestamp = pd.Timestamp("2024-01-02 10:00:00")

        cn_order = adapter.create_target_order(
            symbol="600000.sh",
            target_quantity=155.0,
            timestamp=timestamp,
            requested_price=10.0,
            signal=1,
        )
        us_order = adapter.create_target_order(
            symbol="aapl.us",
            target_quantity=15.0,
            timestamp=timestamp,
            requested_price=100.0,
            signal=1,
        )
        cn_short = adapter.create_target_order(
            symbol="600000.sh",
            target_quantity=-100.0,
            timestamp=timestamp,
            requested_price=10.0,
            signal=-1,
        )

        self.assertIsNotNone(cn_order)
        self.assertEqual(cn_order.quantity, 100.0)
        self.assertIsNotNone(us_order)
        self.assertEqual(us_order.quantity, 15.0)
        self.assertIsNone(cn_short)

    def test_portfolio_engine_supports_mixed_market_lot_sizes(self):
        from src.execution.portfolio_paper_trading_engine import PortfolioPaperTradingEngine

        market_data_by_symbol = {
            "600000.sh": self._build_market_frame(["2024-01-02", "2024-01-03"], [13.0, 13.0]),
            "aapl.us": self._build_market_frame(["2024-01-02", "2024-01-03"], [137.0, 137.0]),
        }
        signals_by_symbol = {
            symbol: self._build_signal_frame(frame.index)
            for symbol, frame in market_data_by_symbol.items()
        }
        config = {
            "data": {"symbols": ["600000.sh", "aapl.us"]},
            "market": {
                "profile": "us_equity",
                "symbol_profiles": {
                    "aapl.us": "us_equity",
                    "600000.sh": "cn_equity",
                },
            },
            "strategy": {"allow_short": True},
            "risk": {"commission": 0.0, "slippage": 0.0},
            "execution_costs": {"fixed_commission": 0.0, "min_commission": 0.0, "sell_tax_rate": 0.0},
            "paper_trading": {
                "initial_cash": 100000.0,
                "allow_fractional": False,
                "lot_size": 1.0,
                "close_positions_on_finish": False,
            },
            "portfolio": {
                "enabled": True,
                "target_gross_allocation": 0.70,
                "max_positions": 2,
                "max_symbol_allocation": 0.35,
                "close_positions_on_finish": False,
            },
        }

        engine = PortfolioPaperTradingEngine(config)
        results = engine.run(market_data_by_symbol, signals_by_symbol)
        positions = results["positions"]

        self.assertEqual(positions["600000.sh"]["quantity"], 2600.0)
        self.assertEqual(positions["aapl.us"]["quantity"], 255.0)
        self.assertEqual(results["summary"]["fills"], 2)


if __name__ == "__main__":
    unittest.main()
