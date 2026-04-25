import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover
    np = None
    pd = None


@unittest.skipUnless(pd is not None and np is not None, "需要安装 pandas 和 numpy 才能运行 app_service 测试")
class TestAppService(unittest.TestCase):
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

    def test_run_main_pipeline_writes_ai_review_and_run_context(self):
        from src.app_service import run_main_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self._build_sample_csv(tmpdir)
            output_dir = Path(tmpdir) / "results"
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
                "backtest": {"initial_capital": 100000.0, "benchmark_enabled": False},
                "paper_trading": {
                    "enabled": True,
                    "initial_cash": 100000.0,
                    "allocation_pct": 0.6,
                    "allow_fractional": False,
                    "allow_short": False,
                    "rebalance_frequency": "weekly",
                    "turnover_buffer": 0.01,
                    "close_positions_on_finish": True,
                },
                "llm": {"enabled": False},
            }

            payload = run_main_pipeline(config=config, output_dir=str(output_dir))

            self.assertEqual(payload["run_type"], "main")
            self.assertTrue((output_dir / "ai_review.json").exists())
            self.assertTrue((output_dir / "run_context.json").exists())
            self.assertTrue((output_dir / "paper_summary.json").exists())
            artifact_paths = {item["relative_path"] for item in payload["artifacts"]}
            self.assertTrue(any(path.endswith("/ai_review.json") or path.endswith("\\ai_review.json") for path in artifact_paths))
