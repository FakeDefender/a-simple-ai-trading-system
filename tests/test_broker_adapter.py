import unittest


class FakeRESTResponse:
    def __init__(self, payload=None, content=True):
        self.payload = payload if payload is not None else {}
        self.content = b"{}" if content else b""

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeRESTSession:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def request(self, method, url, params=None, json=None, headers=None, timeout=None):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "params": params or {},
                "json": json,
                "headers": headers or {},
                "timeout": timeout,
            }
        )
        payload = self.payloads.pop(0) if self.payloads else {}
        return FakeRESTResponse(payload)


class TestConfigurableRESTBrokerClient(unittest.TestCase):
    def _build_config(self):
        return {
            "broker": {
                "provider": "unit_rest",
                "base_url": "https://broker.example/api",
                "account_id": "ACC-1",
                "timeout_seconds": 3.5,
                "auth": {"type": "bearer", "token": "unit-token"},
                "endpoints": {
                    "submit_order": {"method": "POST", "path": "/v1/orders", "response_path": "data"},
                    "list_open_orders": {"method": "GET", "path": "/v1/orders/open", "response_path": "data"},
                    "cancel_order": {
                        "method": "DELETE",
                        "path": "/v1/orders/{broker_order_id}",
                        "response_path": "data",
                    },
                    "get_account": {"method": "GET", "path": "/v1/account", "response_path": "data"},
                    "list_fills": {"method": "GET", "path": "/v1/fills", "response_path": "data"},
                    "list_positions": {"method": "GET", "path": "/v1/positions", "response_path": "data"},
                },
            }
        }

    def test_configurable_rest_client_maps_protocol_calls_to_http(self):
        from src.execution.broker_adapter import ConfigurableRESTBrokerClient

        session = FakeRESTSession(
            [
                {"data": {"broker_order_id": "BRK-1", "status": "submitted"}},
                {"data": [{"broker_order_id": "BRK-1", "symbol": "rest.us", "status": "submitted"}]},
                {"data": {"cash": 100000.0, "equity": 100500.0}},
                {"data": [{"broker_fill_id": "FIL-1", "symbol": "rest.us"}]},
                {"data": [{"symbol": "rest.us", "quantity": 10.0}]},
                {"data": {"broker_order_id": "BRK-1", "status": "canceled"}},
            ]
        )
        client = ConfigurableRESTBrokerClient(self._build_config(), session=session)

        submitted = client.submit_order({"symbol": "rest.us", "side": "buy", "quantity": 10})
        open_orders = list(client.list_open_orders("rest.us"))
        account = client.get_account()
        fills = list(client.list_fills("rest.us"))
        positions = list(client.list_positions())
        canceled = client.cancel_order("BRK-1")

        self.assertEqual(submitted["status"], "submitted")
        self.assertEqual(open_orders[0]["broker_order_id"], "BRK-1")
        self.assertEqual(account["equity"], 100500.0)
        self.assertEqual(fills[0]["broker_fill_id"], "FIL-1")
        self.assertEqual(positions[0]["quantity"], 10.0)
        self.assertEqual(canceled["status"], "canceled")

        first_call = session.calls[0]
        self.assertEqual(first_call["method"], "POST")
        self.assertEqual(first_call["url"], "https://broker.example/api/v1/orders")
        self.assertEqual(first_call["json"]["symbol"], "rest.us")
        self.assertEqual(first_call["headers"]["Authorization"], "Bearer unit-token")
        self.assertEqual(first_call["timeout"], 3.5)

        self.assertEqual(session.calls[1]["params"]["symbol"], "rest.us")
        self.assertEqual(session.calls[1]["params"]["account_id"], "ACC-1")
        self.assertEqual(session.calls[-1]["method"], "DELETE")
        self.assertEqual(session.calls[-1]["url"], "https://broker.example/api/v1/orders/BRK-1")

    def test_configurable_rest_client_requires_base_url_before_requests(self):
        from src.execution.broker_adapter import ConfigurableRESTBrokerClient

        client = ConfigurableRESTBrokerClient({"broker": {"base_url": ""}}, session=FakeRESTSession([]))
        with self.assertRaises(ValueError):
            client.get_account()


if __name__ == "__main__":
    unittest.main()
