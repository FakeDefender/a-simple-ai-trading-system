import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover
    np = None
    pd = None


class FakeDataLoader:
    def __init__(self, datasets):
        self.datasets = list(datasets)
        self.calls = 0

    def load_data(self, symbol=None, interval=None, force_update=None):
        index = min(self.calls, len(self.datasets) - 1)
        self.calls += 1
        return self.datasets[index].copy()


class PassthroughProcessor:
    def get_complete_data(self, market_data, min_periods=50):
        return market_data.copy()


class FixedSignalAgent:
    def generate_signals(self, market_data):
        signals = pd.DataFrame(index=market_data.index)
        signals["signal"] = 1
        signals["market_strength"] = 0.95
        signals["risk_level"] = "low"
        return signals


class FakeBrokerClient:
    def __init__(self):
        self.next_order_id = 1
        self.open_orders = []
        self.fill_payloads = []
        self.account = {
            "cash": 100000.0,
            "equity": 100000.0,
            "market_value": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "gross_exposure": 0.0,
            "net_exposure": 0.0,
            "fees_paid": 0.0,
        }
        self.positions = []

    def submit_order(self, payload):
        broker_order_id = f"BRK-{self.next_order_id:06d}"
        self.next_order_id += 1
        order = {
            "id": broker_order_id,
            "broker_order_id": broker_order_id,
            "client_order_id": payload.get("client_order_id"),
            "symbol": payload.get("symbol"),
            "side": payload.get("side"),
            "quantity": payload.get("quantity"),
            "requested_price": payload.get("requested_price"),
            "status": "submitted",
        }
        self.open_orders.append(order)
        return order

    def cancel_order(self, broker_order_id):
        self.open_orders = [order for order in self.open_orders if order["broker_order_id"] != broker_order_id]
        return {"broker_order_id": broker_order_id, "status": "canceled"}

    def get_order(self, broker_order_id):
        for order in self.open_orders:
            if order["broker_order_id"] == broker_order_id:
                return dict(order)
        return {"broker_order_id": broker_order_id, "status": "canceled"}

    def list_open_orders(self, symbol=None):
        if symbol is None:
            return list(self.open_orders)
        return [order for order in self.open_orders if order["symbol"] == symbol]

    def list_fills(self, symbol=None):
        if symbol is None:
            return list(self.fill_payloads)
        return [fill for fill in self.fill_payloads if fill.get("symbol") == symbol]

    def get_account(self):
        return dict(self.account)

    def list_positions(self):
        return list(self.positions)


@unittest.skipUnless(pd is not None and np is not None, "需要安装 pandas 和 numpy 才能运行 live service 测试")
class TestLiveService(unittest.TestCase):
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

    def test_live_engine_incremental_processing_only_handles_new_rows(self):
        from src.execution.live_trading_engine import LiveTradingEngine

        first_market = self._build_market_frame(["2024-01-02 10:00:00"], [100.0])
        second_market = self._build_market_frame(["2024-01-02 10:00:00", "2024-01-02 10:01:00"], [100.0, 100.0])
        first_signals = FixedSignalAgent().generate_signals(first_market)
        second_signals = FixedSignalAgent().generate_signals(second_market)
        config = {
            "data": {"symbol": "inc.us"},
            "strategy": {"allow_short": False},
            "risk": {"commission": 0.0, "slippage": 0.0},
            "live_trading": {
                "enabled": True,
                "initial_cash": 100000.0,
                "allocation_pct": 0.5,
                "session_start": "09:30",
                "session_end": "15:00",
                "exit_only_start": "14:55",
                "close_positions_on_finish": False,
            },
        }

        engine = LiveTradingEngine(config)
        first_results = engine.run_incremental(first_market, first_signals, symbol="inc.us")
        second_results = engine.run_incremental(second_market, second_signals, symbol="inc.us")

        self.assertEqual(len(first_results["account_history"]), 1)
        self.assertEqual(len(second_results["account_history"]), 2)
        self.assertEqual(engine.processed_count, 2)
        self.assertEqual(len(second_results["orders"]), 1)

    def test_live_service_run_once_skips_duplicate_bars(self):
        from src.execution.live_trading_engine import LiveTradingEngine
        from src.execution.live_trading_service import LiveTradingService

        dataset_one = self._build_market_frame(["2024-01-02 10:00:00"], [100.0])
        dataset_two = self._build_market_frame(["2024-01-02 10:00:00", "2024-01-02 10:01:00"], [100.0, 100.0])
        loader = FakeDataLoader([dataset_one, dataset_two, dataset_two])
        config = {
            "data": {"symbol": "svc.us", "interval": "1m"},
            "strategy": {"allow_short": False, "slow_ma": 2},
            "risk": {"commission": 0.0, "slippage": 0.0},
            "live_trading": {
                "enabled": True,
                "initial_cash": 100000.0,
                "allocation_pct": 0.5,
                "close_positions_on_finish": False,
            },
            "live_service": {
                "save_results_each_cycle": False,
                "force_update_each_cycle": False,
                "max_cycles": 1,
            },
        }

        service = LiveTradingService(
            config,
            data_loader=loader,
            data_processor=PassthroughProcessor(),
            strategy_agent=FixedSignalAgent(),
            engine=LiveTradingEngine(config),
            sleep_fn=lambda seconds: None,
        )

        first_payload = service.run_once()
        second_payload = service.run_once()
        third_payload = service.run_once()

        self.assertEqual(first_payload["processed_rows"], 1)
        self.assertEqual(second_payload["processed_rows"], 1)
        self.assertEqual(third_payload["processed_rows"], 0)
        self.assertEqual(service.engine.processed_count, 2)
        self.assertEqual(len(second_payload["results"]["orders"]), 1)

    def test_build_live_adapter_real_uses_client_protocol(self):
        from src.execution.broker_adapter import RealBrokerAdapter, build_live_adapter

        client = FakeBrokerClient()
        config = {
            "data": {"symbol": "real.us"},
            "strategy": {"allow_short": False},
            "risk": {"commission": 0.0, "slippage": 0.0},
            "live_trading": {
                "enabled": True,
                "adapter": "real",
                "initial_cash": 100000.0,
                "allow_fractional": False,
                "lot_size": 1.0,
            },
            "broker": {
                "provider": "fake",
                "paper": True,
            },
        }

        adapter = build_live_adapter(config, client=client)
        self.assertIsInstance(adapter, RealBrokerAdapter)

        order = adapter.create_target_order(
            symbol="real.us",
            target_quantity=10.0,
            timestamp=pd.Timestamp("2024-01-02 10:00:00"),
            requested_price=100.0,
            signal=1,
        )
        self.assertIsNotNone(order)
        submitted = adapter.submit_order(order, 100.0, pd.Timestamp("2024-01-02 10:00:00"))
        self.assertEqual(submitted.status, "submitted")
        self.assertEqual(len(adapter.get_open_orders("real.us")), 1)

        canceled = adapter.cancel_order(order.order_id, pd.Timestamp("2024-01-02 10:01:00"), reason="manual")
        self.assertEqual(canceled.status, "canceled")
        self.assertEqual(len(adapter.get_open_orders("real.us")), 0)

    def test_live_service_writes_event_and_alert_journals(self):
        from src.execution.live_trading_service import LiveTradingService

        dataset = self._build_market_frame(["2024-01-02 10:00:00"], [100.0])
        loader = FakeDataLoader([dataset])
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "data": {"symbol": "svc.us", "interval": "1m"},
                "strategy": {"allow_short": False, "slow_ma": 2},
                "risk": {"commission": 0.0, "slippage": 0.0},
                "live_trading": {
                    "enabled": True,
                    "initial_cash": 100000.0,
                    "allocation_pct": 0.5,
                    "reject_first_n_orders": 1,
                    "close_positions_on_finish": False,
                },
                "live_service": {
                    "save_results_each_cycle": False,
                    "force_update_each_cycle": False,
                    "results_root": str(Path(tmpdir) / "live_service"),
                    "max_cycles": 1,
                },
            }

            service = LiveTradingService(
                config,
                data_loader=loader,
                data_processor=PassthroughProcessor(),
                strategy_agent=FixedSignalAgent(),
                sleep_fn=lambda seconds: None,
            )
            payload = service.run_once()

            event_log = Path(service.results_dir) / "live_events.jsonl"
            alert_log = Path(service.results_dir) / "live_alerts.jsonl"
            self.assertTrue(event_log.exists())
            self.assertTrue(alert_log.exists())
            self.assertTrue(any(event["event_type"] == "service_cycle_started" for event in payload["recent_events"]))

            event_lines = [json.loads(line) for line in event_log.read_text(encoding="utf-8").splitlines() if line.strip()]
            alert_lines = [json.loads(line) for line in alert_log.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertTrue(any(line["event_type"] == "order_rejected" for line in event_lines))
            self.assertTrue(any(line["alert_type"] == "order_rejected" for line in alert_lines))

    def test_live_service_writes_reconciliation_and_operator_reports(self):
        from src.execution.live_trading_service import LiveTradingService

        dataset = self._build_market_frame(["2024-01-02 10:00:00"], [100.0])
        loader = FakeDataLoader([dataset])
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "data": {"symbol": "svc.us", "interval": "1m"},
                "strategy": {"allow_short": False, "slow_ma": 2},
                "risk": {"commission": 0.0, "slippage": 0.0},
                "live_trading": {
                    "enabled": True,
                    "initial_cash": 100000.0,
                    "allocation_pct": 0.5,
                    "close_positions_on_finish": False,
                },
                "live_service": {
                    "save_results_each_cycle": True,
                    "force_update_each_cycle": False,
                    "results_root": str(Path(tmpdir) / "live_service"),
                    "max_cycles": 1,
                },
            }

            service = LiveTradingService(
                config,
                data_loader=loader,
                data_processor=PassthroughProcessor(),
                strategy_agent=FixedSignalAgent(),
                sleep_fn=lambda seconds: None,
            )
            payload = service.run_once()

            reconciliation_file = Path(service.results_dir) / "live_reconciliation.json"
            operator_file = Path(service.results_dir) / "live_operator_report.json"
            self.assertTrue(reconciliation_file.exists())
            self.assertTrue(operator_file.exists())
            self.assertEqual(payload["reconciliation"]["status"], "ok")
            self.assertIn("operator_report", payload)


if __name__ == "__main__":
    unittest.main()
