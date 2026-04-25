import unittest

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover
    np = None
    pd = None


class FixedSignalAgent:
    def __init__(self, signal: int = 1, market_strength: float = 0.95):
        self.signal = int(signal)
        self.market_strength = float(market_strength)

    def generate_signals(self, market_data):
        signals = pd.DataFrame(index=market_data.index)
        signals["signal"] = self.signal
        signals["market_strength"] = self.market_strength
        signals["risk_level"] = "low"
        return signals


class FakeMultiDataLoader:
    def __init__(self, datasets):
        self.datasets = {str(symbol): frame.copy() for symbol, frame in datasets.items()}

    def load_data(self, symbol=None, interval=None, force_update=None):
        return self.datasets[str(symbol)].copy()


@unittest.skipUnless(pd is not None and np is not None, "需要安装 pandas 和 numpy 才能运行组合 live 测试")
class TestPortfolioLiveTrading(unittest.TestCase):
    def _build_market_frame(self, prices, start: str = "2024-01-02 10:00:00", freq: str = "1D"):
        close = np.asarray(prices, dtype=float)
        index = pd.date_range(start, periods=len(close), freq=freq)
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

    def _build_signal_frame(self, index, signal_values, strength_values):
        return pd.DataFrame(
            {
                "signal": list(signal_values),
                "market_strength": [float(value) for value in strength_values],
                "risk_level": ["low"] * len(index),
            },
            index=index,
        )

    def test_portfolio_live_engine_respects_selection_and_limits(self):
        from src.execution.portfolio_live_trading_engine import PortfolioLiveTradingEngine

        alpha_market = self._build_market_frame([100.0], start="2024-01-02 10:00:00")
        beta_market = self._build_market_frame([100.0], start="2024-01-02 10:00:00")
        gamma_market = self._build_market_frame([100.0], start="2024-01-02 10:00:00")
        market_data_by_symbol = {
            "alpha.us": alpha_market,
            "beta.us": beta_market,
            "gamma.us": gamma_market,
        }
        signals_by_symbol = {
            "alpha.us": self._build_signal_frame(alpha_market.index, [1], [0.95]),
            "beta.us": self._build_signal_frame(beta_market.index, [1], [0.85]),
            "gamma.us": self._build_signal_frame(gamma_market.index, [1], [0.65]),
        }
        config = {
            "data": {"symbol": "alpha.us", "symbols": ["alpha.us", "beta.us", "gamma.us"]},
            "strategy": {"allow_short": False},
            "risk": {"commission": 0.0, "slippage": 0.0},
            "live_trading": {
                "enabled": True,
                "initial_cash": 100000.0,
                "allocation_pct": 0.5,
                "close_positions_on_finish": False,
                "session_start": "09:30",
                "session_end": "15:00",
            },
            "live_risk": {
                "max_order_notional": 0.0,
                "max_position_notional": 0.0,
                "max_daily_drawdown": 0.0,
                "max_open_orders": 10,
                "max_orders_per_day": 0,
                "max_consecutive_failures": 0,
            },
            "portfolio": {
                "enabled": True,
                "target_gross_allocation": 0.8,
                "max_positions": 2,
                "max_gross_exposure": 0.8,
                "max_symbol_allocation": 0.4,
                "rebalance_frequency": "daily",
                "turnover_buffer": 0.0,
            },
        }

        engine = PortfolioLiveTradingEngine(config)
        results = engine.run(market_data_by_symbol, signals_by_symbol)

        self.assertEqual(len(results["orders"]), 2)
        self.assertEqual(results["summary"]["active_symbols"], 2)
        symbol_history = results["symbol_history"]
        self.assertFalse(symbol_history.loc[symbol_history["symbol"] == "gamma.us", "selected"].any())
        self.assertLessEqual(int(results["account_history"]["selected_count"].max()), 2)

    def test_live_service_supports_multiple_symbols(self):
        from src.execution.live_trading_service import LiveTradingService

        alpha_market = self._build_market_frame([100.0, 101.0], start="2024-01-02 10:00:00")
        beta_market = self._build_market_frame([100.0, 100.5], start="2024-01-02 10:00:00")
        config = {
            "data": {"symbol": "alpha.us", "symbols": ["alpha.us", "beta.us"], "interval": "d"},
            "strategy": {"allow_short": False, "slow_ma": 2},
            "risk": {"commission": 0.0, "slippage": 0.0},
            "live_trading": {
                "enabled": True,
                "initial_cash": 100000.0,
                "allocation_pct": 0.5,
                "close_positions_on_finish": False,
                "session_start": "09:30",
                "session_end": "15:00",
            },
            "portfolio": {
                "enabled": True,
                "target_gross_allocation": 0.8,
                "max_positions": 1,
                "max_gross_exposure": 0.8,
                "max_symbol_allocation": 0.8,
                "rebalance_frequency": "daily",
            },
            "live_service": {
                "save_results_each_cycle": False,
                "force_update_each_cycle": False,
                "max_cycles": 1,
            },
        }
        service = LiveTradingService(
            config,
            data_loader=FakeMultiDataLoader({"alpha.us": alpha_market, "beta.us": beta_market}),
            strategy_agent={
                "alpha.us": FixedSignalAgent(signal=1, market_strength=0.95),
                "beta.us": FixedSignalAgent(signal=1, market_strength=0.75),
            },
            sleep_fn=lambda seconds: None,
        )

        payload = service.run_once()

        self.assertEqual(payload["symbols"], ["alpha.us", "beta.us"])
        self.assertEqual(payload["summary"]["active_symbols"], 1)
        self.assertTrue(any(event["event_type"] == "service_cycle_started" for event in payload["recent_events"]))


if __name__ == "__main__":
    unittest.main()
