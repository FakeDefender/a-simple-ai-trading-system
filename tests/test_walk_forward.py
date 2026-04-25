import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover
    np = None
    pd = None


@unittest.skipUnless(pd is not None and np is not None, "需要安装 pandas 和 numpy 才能运行 walk-forward 测试")
class TestWalkForward(unittest.TestCase):
    def _build_sample_csv(self, directory: str) -> str:
        dates = pd.date_range("2024-01-01", periods=280, freq="D")
        base = np.linspace(100, 150, len(dates))
        wave = np.sin(np.linspace(0, 16, len(dates))) * 4
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

    def test_build_walk_forward_folds(self):
        from src.utils.walk_forward import build_walk_forward_folds

        index = pd.date_range("2024-01-01", periods=10, freq="D")
        folds = build_walk_forward_folds(index, train_size=4, test_size=2, step_size=2)

        self.assertEqual(len(folds), 3)
        self.assertEqual(folds[0]["train_start"], 0)
        self.assertEqual(folds[0]["train_end"], 4)
        self.assertEqual(folds[0]["test_start"], 4)
        self.assertEqual(folds[0]["test_end"], 6)
        self.assertEqual(folds[-1]["test_end"], 10)

    def test_run_walk_forward_writes_summary_files(self):
        from src.utils.walk_forward import run_walk_forward

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self._build_sample_csv(tmpdir)
            output_dir = Path(tmpdir) / "walk_forward_results"
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
                    "allow_short": False,
                },
                "risk": {
                    "stop_loss_pct": 0.03,
                    "take_profit_pct": 0.08,
                    "position_size": 1.0,
                    "commission": 0.001,
                    "slippage": 0.0005,
                },
                "paper_trading": {
                    "enabled": True,
                    "initial_cash": 100000.0,
                    "allocation_pct": 0.6,
                    "allow_short": False,
                    "rebalance_frequency": "weekly",
                    "turnover_buffer": 0.01,
                },
                "llm": {"enabled": False},
            }

            payload = run_walk_forward(
                config=config,
                output_dir=str(output_dir),
                train_size=100,
                test_size=40,
                step_size=40,
                include_paper=True,
            )

            self.assertEqual(payload["summary"]["fold_count"], 3)
            self.assertTrue((output_dir / "walk_forward_summary.json").exists())
            self.assertTrue((output_dir / "walk_forward_folds.json").exists())
            self.assertTrue((output_dir / "walk_forward_folds.csv").exists())

    def test_run_walk_forward_supports_multiple_symbols(self):
        from src.utils.walk_forward import run_walk_forward

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path_a = self._build_sample_csv(tmpdir)
            csv_path_b = self._build_sample_csv(tmpdir)
            output_dir = Path(tmpdir) / "walk_forward_multi"
            config = {
                "data": {
                    "source": "csv",
                    "symbol": "alpha.us",
                    "symbols": ["alpha.us", "beta.us"],
                    "paths": {
                        "alpha.us": csv_path_a,
                        "beta.us": csv_path_b,
                    },
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
                    "allow_short": False,
                },
                "risk": {
                    "stop_loss_pct": 0.03,
                    "take_profit_pct": 0.08,
                    "position_size": 1.0,
                    "commission": 0.001,
                    "slippage": 0.0005,
                },
                "paper_trading": {
                    "enabled": False,
                },
                "llm": {"enabled": False},
            }

            payload = run_walk_forward(
                config=config,
                output_dir=str(output_dir),
                train_size=100,
                test_size=40,
                step_size=40,
                include_paper=False,
            )

            self.assertEqual(payload["summary"]["symbol_count"], 2)
            self.assertEqual(len(payload["symbol_summaries"]), 2)
            self.assertTrue((output_dir / "walk_forward_summary.json").exists())
            self.assertTrue((output_dir / "walk_forward_symbol_summary.csv").exists())
