import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover
    np = None
    pd = None


@unittest.skipUnless(pd is not None and np is not None, "需要安装 pandas 和 numpy 才能运行集成测试")
class TestTradingSystem(unittest.TestCase):
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

    def _build_api_config(self, directory: str, symbol: str = "aapl.us", interval: str = "d") -> dict:
        return {
            "data": {
                "source": "api",
                "symbol": symbol,
                "interval": interval,
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "storage": {
                    "base_dir": str(Path(directory) / "market_data"),
                    "raw_data_dir": "raw",
                    "processed_data_dir": "processed",
                },
            },
            "strategy": {"fast_ma": 10, "slow_ma": 20},
            "risk": {"stop_loss_pct": 0.03, "take_profit_pct": 0.06, "position_size": 1.0},
            "backtest": {"initial_capital": 100000.0, "benchmark_symbol": "^ndx"},
            "llm": {"enabled": False},
        }

    def test_csv_data_loader(self):
        from src.utils.data_loader import DataLoader

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
                "strategy": {"fast_ma": 10, "slow_ma": 20},
                "risk": {"stop_loss_pct": 0.03, "take_profit_pct": 0.06, "position_size": 1.0},
                "backtest": {"initial_capital": 100000.0},
                "llm": {"enabled": False},
            }
            loader = DataLoader(config)
            market_data = loader.load_data()

            self.assertIsNotNone(market_data)
            self.assertFalse(market_data.empty)
            self.assertIn("ma20", market_data.columns)
            self.assertIn("rsi", market_data.columns)
            self.assertIn("atr", market_data.columns)

    def test_api_data_loader_falls_back_to_yahoo_chart_when_stooq_is_unavailable(self):
        from src.utils.data_loader import DataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_api_config(tmpdir)
            dates = pd.date_range("2024-01-01", periods=180, freq="D")
            timestamps = [int(ts.timestamp()) for ts in dates]
            closes = np.linspace(100, 140, len(dates)).tolist()
            payload = {
                "chart": {
                    "result": [
                        {
                            "timestamp": timestamps,
                            "indicators": {
                                "quote": [
                                    {
                                        "open": closes,
                                        "high": [value + 1 for value in closes],
                                        "low": [value - 1 for value in closes],
                                        "close": closes,
                                        "volume": list(np.linspace(1000, 5000, len(dates))),
                                    }
                                ]
                            },
                        }
                    ],
                    "error": None,
                }
            }
            request_calls = []

            def fake_get(url, headers=None, params=None, timeout=None):
                request_calls.append({"url": url, "headers": headers, "params": params, "timeout": timeout})
                return SimpleNamespace(status_code=200, text="ok", raise_for_status=lambda: None, json=lambda: payload)

            fake_requests = SimpleNamespace(get=fake_get)
            loader = DataLoader(config)
            with mock.patch.object(loader, "_fetch_from_stooq", return_value=None) as stooq_mock, mock.patch.dict(
                sys.modules, {"requests": fake_requests}
            ):
                market_data = loader.load_data(force_update=True)

            self.assertIsNotNone(market_data)
            self.assertFalse(market_data.empty)
            self.assertIn("ma20", market_data.columns)
            self.assertTrue(request_calls[0]["url"].endswith("/AAPL"))
            self.assertEqual(request_calls[0]["params"]["interval"], "1d")
            stooq_mock.assert_not_called()

    def test_fetch_from_yahoo_chart_url_encodes_benchmark_symbol(self):
        from src.utils.data_loader import DataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_api_config(tmpdir)
            loader = DataLoader(config)
            request_calls = []
            payload = {
                "chart": {
                    "result": [
                        {
                            "timestamp": [1704067200, 1704153600],
                            "indicators": {
                                "quote": [
                                    {
                                        "open": [100.0, 101.0],
                                        "high": [101.0, 102.0],
                                        "low": [99.0, 100.0],
                                        "close": [100.5, 101.5],
                                        "volume": [1000.0, 1200.0],
                                    }
                                ]
                            },
                        }
                    ],
                    "error": None,
                }
            }

            def fake_get(url, headers=None, params=None, timeout=None):
                request_calls.append({"url": url, "params": params, "timeout": timeout})
                return SimpleNamespace(status_code=200, text="ok", raise_for_status=lambda: None, json=lambda: payload)

            fake_requests = SimpleNamespace(get=fake_get)
            with mock.patch.dict(sys.modules, {"requests": fake_requests}):
                data = loader._fetch_from_yahoo_chart("^ndx", "1d")

            self.assertIsNotNone(data)
            self.assertFalse(data.empty)
            self.assertTrue(request_calls[0]["url"].endswith("/%5ENDX"))

    def test_api_data_loader_falls_back_to_yfinance_when_yahoo_chart_is_unavailable(self):
        from src.utils.data_loader import DataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_api_config(tmpdir)
            dates = pd.date_range("2024-01-01", periods=180, freq="D")
            raw_data = pd.DataFrame(
                {
                    "Open": np.linspace(100, 140, len(dates)),
                    "High": np.linspace(101, 141, len(dates)),
                    "Low": np.linspace(99, 139, len(dates)),
                    "Close": np.linspace(100, 140, len(dates)),
                    "Volume": np.linspace(1000, 5000, len(dates)),
                },
                index=dates,
            )
            download_calls = []

            def fake_download(symbol, start=None, end=None, interval=None, progress=None, auto_adjust=None):
                download_calls.append(
                    {
                        "symbol": symbol,
                        "start": start,
                        "end": end,
                        "interval": interval,
                        "progress": progress,
                        "auto_adjust": auto_adjust,
                    }
                )
                return raw_data.copy()

            fake_yfinance = SimpleNamespace(download=fake_download)
            loader = DataLoader(config)
            with mock.patch.object(loader, "_fetch_from_yahoo_chart", return_value=None), mock.patch.dict(
                sys.modules, {"yfinance": fake_yfinance}
            ):
                market_data = loader.load_data(force_update=True)

            self.assertIsNotNone(market_data)
            self.assertFalse(market_data.empty)
            self.assertIn("ma20", market_data.columns)
            self.assertEqual(download_calls[0]["symbol"], "AAPL")
            self.assertEqual(download_calls[0]["interval"], "1d")

    def test_benchmark_data_skips_yfinance_fallback_by_default(self):
        from src.utils.data_loader import DataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_api_config(tmpdir)
            loader = DataLoader(config)
            fallback_data = pd.DataFrame(
                {
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.5],
                    "volume": [1000.0],
                },
                index=pd.date_range("2024-01-01", periods=1, freq="D"),
            )

            with mock.patch.object(loader, "_fetch_from_yahoo_chart", return_value=None) as yahoo_mock, mock.patch.object(
                loader, "_fetch_from_yfinance", return_value=fallback_data
            ) as yfinance_mock:
                benchmark = loader.get_benchmark_data()

            self.assertTrue(benchmark.empty)
            yahoo_mock.assert_called_once()
            yfinance_mock.assert_not_called()

    def test_benchmark_data_uses_external_source_before_target_fallback_by_default(self):
        from src.utils.data_loader import DataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_api_config(tmpdir)
            loader = DataLoader(config)
            reference = pd.DataFrame(
                {
                    "open": [100.0, 101.0, 102.0],
                    "high": [101.0, 102.0, 103.0],
                    "low": [99.0, 100.0, 101.0],
                    "close": [100.5, 101.5, 102.5],
                    "volume": [1000.0, 1100.0, 1200.0],
                },
                index=pd.date_range("2024-01-01", periods=3, freq="D"),
            )

            with mock.patch.object(loader, "_fetch_from_yahoo_chart", return_value=None) as yahoo_mock, mock.patch.object(
                loader, "_fetch_from_yfinance"
            ) as yfinance_mock:
                benchmark = loader.get_benchmark_data(reference_data=reference, reference_symbol="aapl.us")

            self.assertFalse(benchmark.empty)
            self.assertEqual(list(benchmark["close"]), [100.5, 101.5, 102.5])
            yahoo_mock.assert_called_once()
            yfinance_mock.assert_not_called()

    def test_market_benchmark_uses_external_data_when_available(self):
        from src.utils.data_loader import DataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_api_config(tmpdir)
            loader = DataLoader(config)
            reference = pd.DataFrame(
                {"close": [100.0, 101.0], "volume": [1000.0, 1200.0]},
                index=pd.date_range("2024-01-01", periods=2, freq="D"),
            )
            external = pd.DataFrame(
                {
                    "open": [200.0, 202.0],
                    "high": [201.0, 203.0],
                    "low": [199.0, 201.0],
                    "close": [200.5, 202.5],
                    "volume": [5000.0, 5100.0],
                },
                index=pd.date_range("2024-01-01", periods=2, freq="D"),
            )

            with mock.patch.object(loader, "_fetch_from_yahoo_chart", return_value=external) as yahoo_mock:
                benchmark = loader.get_benchmark_data(reference_data=reference, reference_symbol="aapl.us")

            self.assertEqual(list(benchmark["close"]), [200.5, 202.5])
            yahoo_mock.assert_called_once()

    def test_market_benchmark_falls_back_to_target_reference(self):
        from src.utils.data_loader import DataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_api_config(tmpdir)
            config["backtest"]["benchmark_source"] = "market"
            config["backtest"]["benchmark_fallback_to_target"] = True
            loader = DataLoader(config)
            reference = pd.DataFrame(
                {"close": [100.0, 103.0], "volume": [1000.0, 1200.0]},
                index=pd.date_range("2024-01-01", periods=2, freq="D"),
            )

            with mock.patch.object(loader, "_fetch_from_yahoo_chart", return_value=None), mock.patch.object(
                loader, "_fetch_from_yfinance"
            ) as yfinance_mock:
                benchmark = loader.get_benchmark_data(reference_data=reference, reference_symbol="aapl.us")

            self.assertFalse(benchmark.empty)
            self.assertEqual(list(benchmark["close"]), [100.0, 103.0])
            yfinance_mock.assert_not_called()

    def test_fetch_from_stooq_ignores_instruction_text(self):
        from src.utils.data_loader import DataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_api_config(tmpdir)
            config["data"]["stooq_api_key"] = "demo-key"
            loader = DataLoader(config)
            response = mock.MagicMock()
            response.__enter__.return_value = response
            response.read.return_value = (
                "Get your apikey:\n\n1. Open https://stooq.com/q/d/?s=aapl.us&get_apikey".encode("utf-8")
            )

            with mock.patch("src.utils.data_loader.urlopen", return_value=response):
                data = loader._fetch_from_stooq("aapl.us", "d")

            self.assertIsNone(data)

    def test_signal_generation_and_backtest(self):
        from src.agents.ml_strategy_agent import MLStrategyAgent
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
                "llm": {"enabled": False},
            }
            loader = DataLoader(config)
            processor = DataProcessor()
            market_data = loader.load_data()
            complete_data = processor.get_complete_data(market_data, min_periods=50)

            agent = MLStrategyAgent(config, loader)
            signals = agent.generate_signals(complete_data)
            results = agent._backtest_strategy(complete_data, signals)
            performance = agent._calculate_performance_metrics(results)
            risk = agent._calculate_strategy_risk_metrics(results)

            self.assertEqual(len(signals), len(complete_data))
            self.assertIn("signal", signals.columns)
            self.assertIn("equity_curve", results)
            self.assertIn("trades", results)
            self.assertIn("total_return", performance)
            self.assertIn("max_drawdown", risk)

    def test_backtest_plot_uses_non_gui_backend_and_writes_png(self):
        import matplotlib

        from src.agents.ml_strategy_agent import MLStrategyAgent
        from src.utils.data_loader import DataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_api_config(tmpdir)
            loader = DataLoader(config)
            agent = MLStrategyAgent(config, loader)
            agent.results_dir = tmpdir
            equity_curve = pd.Series(
                [100000.0, 101000.0, 100500.0, 102000.0],
                index=pd.date_range("2024-01-01", periods=4, freq="D"),
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                agent._plot_backtest_results({"equity_curve": equity_curve})

            warning_messages = [str(item.message) for item in caught]
            self.assertEqual(matplotlib.get_backend().lower(), "agg")
            self.assertTrue((Path(tmpdir) / "backtest_results.png").exists())
            self.assertFalse(any("Starting a Matplotlib GUI" in message for message in warning_messages))
            self.assertFalse(any("Glyph" in message for message in warning_messages))


if __name__ == "__main__":
    unittest.main()
