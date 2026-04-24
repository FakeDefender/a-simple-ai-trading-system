import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover
    np = None
    pd = None


@unittest.skipUnless(pd is not None and np is not None, "需要安装 pandas 和 numpy 才能运行组合交易测试")
class TestPortfolioPaperTrading(unittest.TestCase):
    def _build_symbol_csv(self, directory: str, symbol: str, trend: float) -> str:
        dates = pd.date_range("2024-01-01", periods=160, freq="D")
        base = np.linspace(100, 100 + trend, len(dates))
        wave = np.sin(np.linspace(0, 12, len(dates))) * 2
        close = base + wave
        data = pd.DataFrame(
            {
                "date": dates,
                "open": close * 0.995,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.linspace(1000, 6000, len(dates)),
            }
        )
        path = Path(directory) / f"{symbol}.csv"
        data.to_csv(path, index=False, encoding="utf-8")
        return str(path)

    def _build_drawdown_csv(self, directory: str, symbol: str) -> str:
        dates = pd.date_range("2024-01-01", periods=90, freq="D")
        up_leg = np.linspace(100, 120, 45)
        down_leg = np.linspace(120, 70, 45)
        close = np.concatenate([up_leg, down_leg])
        data = pd.DataFrame(
            {
                "date": dates,
                "open": close * 0.995,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.linspace(1000, 4000, len(dates)),
            }
        )
        path = Path(directory) / f"{symbol}_dd.csv"
        data.to_csv(path, index=False, encoding="utf-8")
        return str(path)

    def _build_market_frame(self, prices, start: str = "2024-01-01", freq: str = "B"):
        close = np.asarray(prices, dtype=float)
        index = pd.date_range(start, periods=len(close), freq=freq)
        return pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.linspace(1000, 2500, len(close)),
            },
            index=index,
        )

    def _build_signal_frame(self, index, signal_values, market_strength_values):
        return pd.DataFrame(
            {
                "signal": list(signal_values),
                "market_strength": [float(value) for value in market_strength_values],
                "risk_level": ["low"] * len(index),
            },
            index=index,
        )

    def test_portfolio_engine_respects_selection_and_exposure_limits(self):
        from src.execution.portfolio_paper_trading_engine import PortfolioPaperTradingEngine
        from src.utils.data_loader import DataLoader
        from src.utils.data_processor import DataProcessor

        with tempfile.TemporaryDirectory() as tmpdir:
            symbol_paths = {
                "alpha.us": self._build_symbol_csv(tmpdir, "alpha", 40),
                "beta.us": self._build_symbol_csv(tmpdir, "beta", 20),
                "gamma.us": self._build_symbol_csv(tmpdir, "gamma", -10),
            }
            config = {
                "data": {
                    "source": "csv",
                    "symbol": "alpha.us",
                    "symbols": ["alpha.us", "beta.us", "gamma.us"],
                    "paths": symbol_paths,
                    "interval": "d",
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                    "storage": {
                        "base_dir": str(Path(tmpdir) / "market_data"),
                        "raw_data_dir": "raw",
                        "processed_data_dir": "processed",
                    },
                },
                "strategy": {"allow_short": True},
                "risk": {"commission": 0.0, "slippage": 0.0},
                "paper_trading": {
                    "allow_fractional": False,
                    "lot_size": 1.0,
                },
                "portfolio": {
                    "enabled": True,
                    "target_gross_allocation": 0.8,
                    "max_positions": 2,
                    "max_gross_exposure": 0.8,
                    "max_symbol_allocation": 0.4,
                    "max_portfolio_drawdown": 0.5,
                    "close_positions_on_finish": True,
                },
            }

            loader = DataLoader(config)
            processor = DataProcessor()
            market_data_by_symbol = {}
            signals_by_symbol = {}
            strengths = {"alpha.us": 0.95, "beta.us": 0.85, "gamma.us": 0.65}
            directions = {"alpha.us": 1, "beta.us": 1, "gamma.us": -1}

            for symbol in config["data"]["symbols"]:
                market_data = loader.load_data(symbol=symbol)
                complete_data = processor.get_complete_data(market_data, min_periods=50)
                market_data_by_symbol[symbol] = complete_data
                signals = pd.DataFrame(index=complete_data.index)
                signals["signal"] = directions[symbol]
                signals["market_strength"] = strengths[symbol]
                signals["risk_level"] = "low"
                signals_by_symbol[symbol] = signals

            engine = PortfolioPaperTradingEngine(config)
            results = engine.run(market_data_by_symbol, signals_by_symbol)
            output_dir = Path(tmpdir) / "portfolio_results"
            engine.save_results(results, str(output_dir))

            self.assertFalse(results["account_history"].empty)
            self.assertFalse(results["symbol_history"].empty)
            self.assertLessEqual(results["account_history"]["selected_count"].max(), 2)
            self.assertFalse(results["symbol_history"].loc[results["symbol_history"]["symbol"] == "gamma.us", "selected"].any())
            exposure_ratio = results["account_history"]["gross_exposure"] / results["account_history"]["equity"]
            self.assertTrue((exposure_ratio.fillna(0) <= 0.800001).all())
            self.assertTrue((output_dir / "portfolio_orders.csv").exists())
            self.assertTrue((output_dir / "portfolio_fills.csv").exists())
            self.assertTrue((output_dir / "portfolio_account_history.csv").exists())
            self.assertTrue((output_dir / "portfolio_symbol_history.csv").exists())
            self.assertTrue((output_dir / "portfolio_summary.json").exists())
            self.assertTrue((output_dir / "portfolio_symbol_summary.json").exists())

    def test_portfolio_engine_pauses_after_drawdown_limit(self):
        from src.execution.portfolio_paper_trading_engine import PortfolioPaperTradingEngine
        from src.utils.data_loader import DataLoader
        from src.utils.data_processor import DataProcessor

        with tempfile.TemporaryDirectory() as tmpdir:
            symbol_paths = {
                "crash.us": self._build_drawdown_csv(tmpdir, "crash"),
            }
            config = {
                "data": {
                    "source": "csv",
                    "symbol": "crash.us",
                    "symbols": ["crash.us"],
                    "paths": symbol_paths,
                    "interval": "d",
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                    "storage": {
                        "base_dir": str(Path(tmpdir) / "market_data"),
                        "raw_data_dir": "raw",
                        "processed_data_dir": "processed",
                    },
                },
                "strategy": {"allow_short": False},
                "risk": {"commission": 0.0, "slippage": 0.0},
                "paper_trading": {
                    "allow_fractional": False,
                    "lot_size": 1.0,
                },
                "portfolio": {
                    "enabled": True,
                    "target_gross_allocation": 0.9,
                    "max_positions": 1,
                    "max_gross_exposure": 0.9,
                    "max_symbol_allocation": 0.9,
                    "max_portfolio_drawdown": 0.1,
                    "close_positions_on_finish": False,
                },
            }

            loader = DataLoader(config)
            processor = DataProcessor()
            market_data = loader.load_data(symbol="crash.us")
            complete_data = processor.get_complete_data(market_data, min_periods=20)
            signals = pd.DataFrame(index=complete_data.index)
            signals["signal"] = 1
            signals["market_strength"] = 0.95
            signals["risk_level"] = "low"

            engine = PortfolioPaperTradingEngine(config)
            results = engine.run({"crash.us": complete_data}, {"crash.us": signals})

            self.assertTrue(results["summary"]["paused"])
            self.assertTrue(results["account_history"]["paused"].any())
            paused_rows = results["account_history"][results["account_history"]["paused"]]
            self.assertTrue((paused_rows["selected_count"] == 0).all())

    def test_portfolio_weekly_rebalance_only_on_schedule(self):
        from src.execution.portfolio_paper_trading_engine import PortfolioPaperTradingEngine

        alpha_prices = [100.0] * 15
        beta_prices = [100.0] * 15
        alpha_market = self._build_market_frame(alpha_prices, start="2024-01-01", freq="B")
        beta_market = self._build_market_frame(beta_prices, start="2024-01-01", freq="B")
        market_data_by_symbol = {
            "alpha.us": alpha_market,
            "beta.us": beta_market,
        }

        alpha_strength = [0.95] * 5 + [0.55] * 5 + [0.96] * 5
        beta_strength = [0.60] * 5 + [0.99] * 5 + [0.50] * 5
        signals_by_symbol = {
            "alpha.us": self._build_signal_frame(alpha_market.index, [1] * 15, alpha_strength),
            "beta.us": self._build_signal_frame(beta_market.index, [1] * 15, beta_strength),
        }
        config = {
            "data": {
                "symbol": "alpha.us",
                "symbols": ["alpha.us", "beta.us"],
            },
            "strategy": {"allow_short": False},
            "risk": {"commission": 0.0, "slippage": 0.0},
            "paper_trading": {
                "initial_cash": 100000.0,
                "allow_fractional": False,
                "lot_size": 1.0,
            },
            "portfolio": {
                "enabled": True,
                "target_gross_allocation": 0.5,
                "max_positions": 1,
                "max_gross_exposure": 0.5,
                "max_symbol_allocation": 0.5,
                "max_portfolio_drawdown": 0.8,
                "close_positions_on_finish": False,
                "rebalance_frequency": "weekly",
                "rebalance_weekday": 0,
                "turnover_buffer": 0.0,
            },
        }

        engine = PortfolioPaperTradingEngine(config)
        results = engine.run(market_data_by_symbol, signals_by_symbol)
        rebalanced_rows = results["account_history"][results["account_history"]["rebalanced"]]
        symbol_history = results["symbol_history"]
        second_monday = pd.Timestamp("2024-01-08")
        alpha_second_week = symbol_history[
            (symbol_history["timestamp"] == second_monday) & (symbol_history["symbol"] == "alpha.us")
        ].iloc[0]
        beta_second_week = symbol_history[
            (symbol_history["timestamp"] == second_monday) & (symbol_history["symbol"] == "beta.us")
        ].iloc[0]

        self.assertEqual(len(rebalanced_rows), 3)
        self.assertTrue((rebalanced_rows.index.weekday == 0).all())
        self.assertTrue((rebalanced_rows["rebalance_reason"] == "scheduled_weekly").all())
        self.assertEqual(alpha_second_week["target_quantity"], 0.0)
        self.assertTrue(bool(beta_second_week["selected"]))
        self.assertGreater(beta_second_week["target_quantity"], 0.0)

    def test_portfolio_turnover_buffer_skips_small_rebalance(self):
        from src.execution.portfolio_paper_trading_engine import PortfolioPaperTradingEngine

        market_data = self._build_market_frame([100.0] * 5 + [100.2] * 5, start="2024-01-01", freq="B")
        signals = self._build_signal_frame(market_data.index, [1] * len(market_data), [0.95] * len(market_data))
        config = {
            "data": {
                "symbol": "alpha.us",
                "symbols": ["alpha.us"],
            },
            "strategy": {"allow_short": False},
            "risk": {"commission": 0.0, "slippage": 0.0},
            "paper_trading": {
                "initial_cash": 100000.0,
                "allow_fractional": True,
                "lot_size": 0.0001,
                "quantity_precision": 4,
            },
            "portfolio": {
                "enabled": True,
                "target_gross_allocation": 0.5,
                "max_positions": 1,
                "max_gross_exposure": 0.5,
                "max_symbol_allocation": 0.5,
                "max_portfolio_drawdown": 0.8,
                "close_positions_on_finish": False,
                "rebalance_frequency": "weekly",
                "rebalance_weekday": 0,
                "turnover_buffer": 0.01,
            },
        }

        engine = PortfolioPaperTradingEngine(config)
        results = engine.run({"alpha.us": market_data}, {"alpha.us": signals})
        second_monday = pd.Timestamp("2024-01-08")

        self.assertEqual(len(results["orders"]), 1)
        self.assertEqual(int(results["account_history"]["rebalanced"].sum()), 1)
        self.assertFalse(bool(results["account_history"].loc[second_monday, "rebalanced"]))
        self.assertEqual(results["account_history"].loc[second_monday, "rebalance_reason"], "scheduled_weekly")


if __name__ == "__main__":
    unittest.main()
