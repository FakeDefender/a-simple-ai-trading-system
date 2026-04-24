import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover
    np = None
    pd = None


@unittest.skipUnless(pd is not None and np is not None, "需要安装 pandas 和 numpy 才能运行 paper trading 测试")
class TestPaperTrading(unittest.TestCase):
    def _build_sample_csv(self, directory: str) -> str:
        dates = pd.date_range("2024-01-01", periods=180, freq="D")
        base = np.linspace(100, 140, len(dates))
        wave = np.sin(np.linspace(0, 12, len(dates))) * 4
        close = base + wave
        data = pd.DataFrame(
            {
                "date": dates,
                "open": close * 0.995,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.linspace(1000, 5000, len(dates)),
            }
        )
        path = Path(directory) / "sample_prices.csv"
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
                "volume": np.linspace(1000, 2000, len(close)),
            },
            index=index,
        )

    def _build_signal_frame(self, index, signal_values, market_strength=0.9):
        if np.isscalar(market_strength):
            strength_values = [float(market_strength)] * len(index)
        else:
            strength_values = [float(value) for value in market_strength]
        return pd.DataFrame(
            {
                "signal": list(signal_values),
                "market_strength": strength_values,
                "risk_level": ["low"] * len(index),
            },
            index=index,
        )

    def test_paper_broker_round_trip(self):
        from src.execution.paper_broker import PaperBroker

        broker = PaperBroker(
            initial_cash=100000.0,
            commission_rate=0.0,
            slippage_rate=0.0,
            allow_short=True,
            allow_fractional=False,
            lot_size=1.0,
        )
        order_open = broker.create_order("sample.us", "buy", 10, pd.Timestamp("2024-01-01"), 100.0)
        fill_open = broker.execute_order(order_open, 100.0, pd.Timestamp("2024-01-01"))
        broker.mark_to_market("sample.us", 105.0, pd.Timestamp("2024-01-02"))
        order_close = broker.create_target_order("sample.us", 0.0, pd.Timestamp("2024-01-02"), 105.0)
        fill_close = broker.execute_order(order_close, 105.0, pd.Timestamp("2024-01-02"))
        broker.mark_to_market("sample.us", 105.0, pd.Timestamp("2024-01-02"))

        snapshot = broker.get_account_snapshot(pd.Timestamp("2024-01-02"))
        self.assertEqual(fill_open.fill_price, 100.0)
        self.assertEqual(fill_close.realized_pnl, 50.0)
        self.assertEqual(broker.get_position("sample.us").quantity, 0.0)
        self.assertEqual(snapshot.cash, 100050.0)
        self.assertEqual(snapshot.equity, 100050.0)

    def test_execution_cost_model_applies_min_commission_and_sell_tax(self):
        from src.execution.cost_model import ExecutionCostModel
        from src.execution.paper_broker import PaperBroker

        broker = PaperBroker(
            initial_cash=100000.0,
            allow_short=False,
            cost_model=ExecutionCostModel(
                commission_rate=0.0,
                slippage_rate=0.0,
                fixed_commission=0.0,
                min_commission=2.0,
                sell_tax_rate=0.001,
            ),
        )

        buy_order = broker.create_order("cost.us", "buy", 10, pd.Timestamp("2024-01-01"), 100.0)
        buy_fill = broker.execute_order(buy_order, 100.0, pd.Timestamp("2024-01-01"))
        sell_order = broker.create_target_order("cost.us", 0.0, pd.Timestamp("2024-01-02"), 110.0)
        sell_fill = broker.execute_order(sell_order, 110.0, pd.Timestamp("2024-01-02"))
        broker.mark_to_market("cost.us", 110.0, pd.Timestamp("2024-01-02"))

        snapshot = broker.get_account_snapshot(pd.Timestamp("2024-01-02"))
        self.assertAlmostEqual(buy_fill.total_fees, 2.0, places=6)
        self.assertAlmostEqual(sell_fill.tax, 1.1, places=6)
        self.assertAlmostEqual(sell_fill.total_fees, 3.1, places=6)
        self.assertAlmostEqual(snapshot.cash, 100094.9, places=6)
        self.assertAlmostEqual(snapshot.fees_paid, 5.1, places=6)
        self.assertAlmostEqual(sell_fill.gross_realized_pnl, 100.0, places=6)
        self.assertAlmostEqual(sell_fill.realized_pnl, 96.9, places=6)

    def test_build_paper_adapter_uses_execution_cost_config(self):
        from src.execution.broker_adapter import PaperBrokerAdapter, build_paper_adapter

        config = {
            "strategy": {"allow_short": False},
            "risk": {"commission": 0.0012, "slippage": 0.0003},
            "execution_costs": {
                "fixed_commission": 1.5,
                "min_commission": 3.0,
                "sell_tax_rate": 0.001,
            },
            "paper_trading": {
                "initial_cash": 123456.0,
                "allow_fractional": True,
                "lot_size": 0.01,
            },
        }

        adapter = build_paper_adapter(config)
        self.assertIsInstance(adapter, PaperBrokerAdapter)
        self.assertAlmostEqual(adapter.initial_cash, 123456.0, places=6)
        self.assertFalse(adapter.broker.allow_short)
        self.assertTrue(adapter.broker.allow_fractional)
        self.assertAlmostEqual(adapter.broker.cost_model.commission_rate, 0.0012, places=9)
        self.assertAlmostEqual(adapter.broker.cost_model.slippage_rate, 0.0003, places=9)
        self.assertAlmostEqual(adapter.broker.cost_model.fixed_commission, 1.5, places=9)
        self.assertAlmostEqual(adapter.broker.cost_model.min_commission, 3.0, places=9)
        self.assertAlmostEqual(adapter.broker.cost_model.sell_tax_rate, 0.001, places=9)

    def test_weekly_rebalance_only_trades_on_schedule(self):
        from src.execution.paper_trading_engine import PaperTradingEngine

        market_data = self._build_market_frame([100.0] * 15, start="2024-01-01", freq="B")
        signal_values = [1 if idx % 2 == 0 else -1 for idx in range(len(market_data))]
        signals = self._build_signal_frame(market_data.index, signal_values)
        config = {
            "data": {"symbol": "weekly.us"},
            "strategy": {"allow_short": True},
            "risk": {"commission": 0.0, "slippage": 0.0},
            "paper_trading": {
                "enabled": True,
                "initial_cash": 100000.0,
                "allocation_pct": 0.5,
                "allow_fractional": False,
                "lot_size": 1.0,
                "close_positions_on_finish": False,
                "price_field": "close",
                "rebalance_frequency": "weekly",
                "rebalance_weekday": 0,
                "turnover_buffer": 0.0,
                "max_account_drawdown": 0.9,
            },
        }

        engine = PaperTradingEngine(config)
        results = engine.run(market_data, signals, symbol="weekly.us")
        rebalanced_rows = results["account_history"][results["account_history"]["rebalanced"]]

        self.assertEqual(len(rebalanced_rows), 3)
        self.assertTrue((rebalanced_rows.index.weekday == 0).all())
        self.assertTrue((rebalanced_rows["rebalance_reason"] == "scheduled_weekly").all())
        self.assertLess(len(rebalanced_rows), len(results["account_history"]))

    def test_paper_trading_engine_runs_and_saves_results(self):
        from src.agents.ml_strategy_agent import MLStrategyAgent
        from src.execution.paper_trading_engine import PaperTradingEngine
        from src.utils.data_loader import DataLoader
        from src.utils.data_processor import DataProcessor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self._build_sample_csv(tmpdir)
            config = {
                "data": {
                    "source": "csv",
                    "path": csv_path,
                    "symbol": "sample.us",
                    "interval": "d",
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                    "storage": {
                        "base_dir": str(Path(tmpdir) / "market_data"),
                        "raw_data_dir": "raw",
                        "processed_data_dir": "processed",
                    },
                },
                "strategy": {
                    "fast_ma": 10,
                    "slow_ma": 20,
                    "rsi_long_threshold": 52,
                    "rsi_short_threshold": 48,
                    "allow_short": True,
                },
                "risk": {
                    "stop_loss_pct": 0.03,
                    "take_profit_pct": 0.08,
                    "position_size": 1.0,
                    "commission": 0.001,
                    "slippage": 0.0005,
                },
                "backtest": {"initial_capital": 100000.0},
                "paper_trading": {
                    "enabled": True,
                    "initial_cash": 100000.0,
                    "allocation_pct": 0.95,
                    "allow_fractional": False,
                    "lot_size": 1.0,
                    "close_positions_on_finish": True,
                    "price_field": "close",
                },
                "llm": {"enabled": False},
            }
            loader = DataLoader(config)
            processor = DataProcessor()
            market_data = loader.load_data()
            complete_data = processor.get_complete_data(market_data, min_periods=50)

            agent = MLStrategyAgent(config, loader)
            signals = agent.generate_signals(complete_data)

            engine = PaperTradingEngine(config)
            results = engine.run(complete_data, signals, symbol="sample.us")
            output_dir = Path(tmpdir) / "paper_results"
            engine.save_results(results, str(output_dir))

            self.assertIn("account_history", results)
            self.assertIn("summary", results)
            self.assertFalse(results["account_history"].empty)
            self.assertIn("equity", results["account_history"].columns)
            self.assertIn("fees_paid", results["account_history"].columns)
            self.assertIn("rebalanced", results["account_history"].columns)
            self.assertTrue((output_dir / "paper_orders.csv").exists())
            self.assertTrue((output_dir / "paper_fills.csv").exists())
            self.assertTrue((output_dir / "paper_account_history.csv").exists())
            self.assertTrue((output_dir / "paper_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
