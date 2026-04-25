"""Broker adapter 抽象层。"""

import math
from abc import ABC, abstractmethod
from itertools import count
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote

from .cost_model import ExecutionCostModel
from .live_broker import PaperLiveBroker
from .market_profile import MarketProfileResolver
from .models import AccountSnapshot, Fill, Order, Position
from .paper_broker import PaperBroker


class BrokerAdapter(ABC):
    @property
    @abstractmethod
    def initial_cash(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def orders(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def fills(self):
        raise NotImplementedError

    @abstractmethod
    def create_target_order(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def execute_order(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def mark_to_market(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_account_snapshot(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_position(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_positions(self, *args, **kwargs):
        raise NotImplementedError

    def submit_order(self, *args, **kwargs):
        raise NotImplementedError("当前 adapter 不支持异步 submit_order")

    def cancel_order(self, *args, **kwargs):
        raise NotImplementedError("当前 adapter 不支持 cancel_order")

    def get_order(self, *args, **kwargs):
        raise NotImplementedError("当前 adapter 不支持 get_order")

    def get_open_orders(self, *args, **kwargs):
        return []

    def process_pending_orders(self, *args, **kwargs):
        return []


class BrokerAPIClient(ABC):
    @abstractmethod
    def submit_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_order(self, broker_order_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_open_orders(self, symbol: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def list_fills(self, symbol: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_account(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_positions(self) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError


class ConfigurableRESTBrokerClient(BrokerAPIClient):
    DEFAULT_ENDPOINTS = {
        "submit_order": {"method": "POST", "path": "/orders"},
        "cancel_order": {"method": "POST", "path": "/orders/{broker_order_id}/cancel"},
        "get_order": {"method": "GET", "path": "/orders/{broker_order_id}"},
        "list_open_orders": {"method": "GET", "path": "/orders/open"},
        "list_fills": {"method": "GET", "path": "/fills"},
        "get_account": {"method": "GET", "path": "/account"},
        "list_positions": {"method": "GET", "path": "/positions"},
    }

    def __init__(self, config: Dict[str, Any], session: Optional[Any] = None):
        broker_config = config.get("broker", {})
        self.provider = str(broker_config.get("provider", "generic_rest"))
        provider_secrets = config.get("api_keys", {}).get(self.provider, {})
        if not isinstance(provider_secrets, dict):
            provider_secrets = {}

        self.base_url = str(broker_config.get("base_url") or provider_secrets.get("base_url") or "").rstrip("/")
        self.account_id = str(broker_config.get("account_id", ""))
        self.timeout_seconds = float(broker_config.get("timeout_seconds", 10.0))
        self.paper = bool(broker_config.get("paper", True))
        self.headers = dict(broker_config.get("headers", {}) or {})
        self.auth_config = dict(broker_config.get("auth", {}) or {})
        self.api_key = (
            broker_config.get("api_key")
            or self.auth_config.get("token")
            or provider_secrets.get("api_key")
            or provider_secrets.get("token")
        )
        self.endpoints = dict(self.DEFAULT_ENDPOINTS)
        self.endpoints.update(dict(broker_config.get("endpoints", {}) or {}))
        self.response_paths = dict(broker_config.get("response_paths", {}) or {})
        if session is None:
            try:
                import requests
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("未安装 requests，无法使用 ConfigurableRESTBrokerClient") from exc
            session = requests.Session()
        self.session = session

    def _endpoint_config(self, action: str) -> Dict[str, Any]:
        endpoint = self.endpoints.get(action)
        if endpoint is None:
            raise NotImplementedError(f"broker.provider={self.provider} 未配置 endpoint: {action}")
        default_endpoint = dict(self.DEFAULT_ENDPOINTS.get(action, {}))
        if isinstance(endpoint, str):
            default_endpoint["path"] = endpoint
            return default_endpoint
        if not isinstance(endpoint, dict):
            raise ValueError(f"broker.endpoints.{action} 必须是字符串或字典")
        default_endpoint.update(endpoint)
        return default_endpoint

    def _format_path(self, path: str, path_params: Optional[Dict[str, Any]] = None) -> str:
        params = {"account_id": self.account_id}
        params.update(path_params or {})
        formatted = str(path or "")
        for key, value in params.items():
            formatted = formatted.replace("{" + key + "}", quote(str(value), safe=""))
        return formatted.lstrip("/")

    def _build_headers(self, has_body: bool = False) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if has_body:
            headers["Content-Type"] = "application/json"
        headers.update({str(key): str(value) for key, value in self.headers.items()})

        auth_headers = self.auth_config.get("headers", {})
        if isinstance(auth_headers, dict):
            headers.update({str(key): str(value) for key, value in auth_headers.items()})

        token = self.api_key
        auth_type = str(self.auth_config.get("type", "bearer" if token else "none")).lower()
        if token and auth_type not in {"none", "disabled"}:
            header_name = str(self.auth_config.get("header_name", "Authorization"))
            if auth_type == "bearer":
                prefix = str(self.auth_config.get("prefix", "Bearer "))
                headers[header_name] = f"{prefix}{token}"
            else:
                headers[header_name] = str(token)
        return headers

    def _compact_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in params.items() if value is not None and value != ""}

    def _select_response_path(self, payload: Any, response_path: Optional[str]) -> Any:
        if not response_path:
            return payload
        current = payload
        for part in str(response_path).split("."):
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                current = current[int(part)]
            else:
                return None
        return current

    def _request(
        self,
        action: str,
        path_params: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        json_payload: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if not self.base_url:
            raise ValueError("broker.base_url 不能为空；使用 live_trading.adapter=real 前请配置真实券商 REST 地址")

        endpoint = self._endpoint_config(action)
        method = str(endpoint.get("method", "GET")).upper()
        path = self._format_path(str(endpoint.get("path", "")), path_params=path_params)
        url = f"{self.base_url}/{path}"
        query_params = dict(endpoint.get("params", {}) or {})
        query_params.update(params or {})
        if self.account_id and bool(endpoint.get("include_account_id", True)):
            query_params.setdefault("account_id", self.account_id)

        response = self.session.request(
            method,
            url,
            params=self._compact_params(query_params),
            json=json_payload,
            headers=self._build_headers(has_body=json_payload is not None),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        if not getattr(response, "content", b""):
            payload = {}
        else:
            payload = response.json()
        response_path = endpoint.get("response_path") or self.response_paths.get(action)
        return self._select_response_path(payload, response_path)

    def _as_list(self, payload: Any, candidate_keys: List[str]) -> Iterable[Dict[str, Any]]:
        if payload is None:
            return []
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in candidate_keys + ["data", "items", "results"]:
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        return []

    def submit_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self._request("submit_order", json_payload=payload)
        return response if isinstance(response, dict) else {}

    def cancel_order(self, broker_order_id: str) -> Dict[str, Any]:
        response = self._request("cancel_order", path_params={"broker_order_id": broker_order_id})
        return response if isinstance(response, dict) else {}

    def get_order(self, broker_order_id: str) -> Dict[str, Any]:
        response = self._request("get_order", path_params={"broker_order_id": broker_order_id})
        return response if isinstance(response, dict) else {}

    def list_open_orders(self, symbol: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        params = {"symbol": symbol} if symbol else {}
        payload = self._request("list_open_orders", params=params)
        return self._as_list(payload, ["open_orders", "orders"])

    def list_fills(self, symbol: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        params = {"symbol": symbol} if symbol else {}
        payload = self._request("list_fills", params=params)
        return self._as_list(payload, ["fills", "executions"])

    def get_account(self) -> Dict[str, Any]:
        response = self._request("get_account")
        return response if isinstance(response, dict) else {}

    def list_positions(self) -> Iterable[Dict[str, Any]]:
        payload = self._request("list_positions")
        return self._as_list(payload, ["positions"])


class PaperBrokerAdapter(BrokerAdapter):
    def __init__(self, broker: PaperBroker):
        self.broker = broker

    @property
    def initial_cash(self) -> float:
        return float(self.broker.initial_cash)

    @property
    def orders(self):
        return self.broker.orders

    @property
    def fills(self):
        return self.broker.fills

    def create_target_order(self, *args, **kwargs):
        return self.broker.create_target_order(*args, **kwargs)

    def execute_order(self, *args, **kwargs):
        return self.broker.execute_order(*args, **kwargs)

    def mark_to_market(self, *args, **kwargs):
        return self.broker.mark_to_market(*args, **kwargs)

    def get_account_snapshot(self, *args, **kwargs):
        return self.broker.get_account_snapshot(*args, **kwargs)

    def get_position(self, *args, **kwargs):
        return self.broker.get_position(*args, **kwargs)

    def get_positions(self, *args, **kwargs):
        return self.broker.get_positions(*args, **kwargs)


class PaperLiveBrokerAdapter(BrokerAdapter):
    def __init__(self, broker: PaperLiveBroker):
        self.broker = broker

    @property
    def initial_cash(self) -> float:
        return float(self.broker.initial_cash)

    @property
    def orders(self):
        return self.broker.orders

    @property
    def fills(self):
        return self.broker.fills

    def create_target_order(self, *args, **kwargs):
        return self.broker.create_target_order(*args, **kwargs)

    def execute_order(self, order, market_price: float, timestamp: Any):
        return self.broker.paper_broker.execute_order(order, market_price, timestamp)

    def submit_order(self, *args, **kwargs):
        return self.broker.submit_order(*args, **kwargs)

    def cancel_order(self, *args, **kwargs):
        return self.broker.cancel_order(*args, **kwargs)

    def get_order(self, *args, **kwargs):
        return self.broker.get_order(*args, **kwargs)

    def get_open_orders(self, *args, **kwargs):
        return self.broker.get_open_orders(*args, **kwargs)

    def process_pending_orders(self, *args, **kwargs):
        return self.broker.process_pending_orders(*args, **kwargs)

    def mark_to_market(self, *args, **kwargs):
        return self.broker.mark_to_market(*args, **kwargs)

    def get_account_snapshot(self, *args, **kwargs):
        return self.broker.get_account_snapshot(*args, **kwargs)

    def get_position(self, *args, **kwargs):
        return self.broker.get_position(*args, **kwargs)

    def get_positions(self, *args, **kwargs):
        return self.broker.get_positions(*args, **kwargs)


class RealBrokerAdapter(BrokerAdapter):
    def __init__(
        self,
        client: BrokerAPIClient,
        initial_cash: float,
        allow_short: bool = True,
        allow_fractional: bool = False,
        lot_size: float = 1.0,
        market_resolver: Optional[MarketProfileResolver] = None,
        market_section: str = "live_trading",
    ):
        self.client = client
        self._initial_cash = float(initial_cash)
        self.allow_short = bool(allow_short)
        self.allow_fractional = bool(allow_fractional)
        self.lot_size = float(lot_size) if lot_size else 1.0
        self.market_resolver = market_resolver
        self.market_section = market_section

        self._orders: List[Order] = []
        self._fills: List[Fill] = []
        self._positions: Dict[str, Position] = {}
        self._account_state: Dict[str, Any] = {
            "cash": float(initial_cash),
            "equity": float(initial_cash),
            "market_value": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "gross_exposure": 0.0,
            "net_exposure": 0.0,
            "fees_paid": 0.0,
        }
        self._order_counter = count(1)
        self._fill_counter = count(1)
        self._fill_ids_by_broker_fill_id = set()

    @property
    def initial_cash(self) -> float:
        return float(self._initial_cash)

    @property
    def orders(self):
        return self._orders

    @property
    def fills(self):
        return self._fills

    def _rules_for(self, symbol: str) -> Dict[str, Any]:
        if self.market_resolver is None:
            return {
                "allow_short": bool(self.allow_short),
                "allow_fractional": bool(self.allow_fractional),
                "lot_size": float(self.lot_size),
            }
        return self.market_resolver.section_settings(symbol=symbol, section_name=self.market_section)

    def _normalize_quantity(self, quantity: float, symbol: Optional[str] = None) -> float:
        quantity = abs(float(quantity))
        if quantity == 0:
            return 0.0
        rules = self._rules_for(symbol or "")
        step = float(rules.get("lot_size", self.lot_size) or 1.0)
        if bool(rules.get("allow_fractional", self.allow_fractional)):
            normalized = round(quantity / step) * step
        else:
            normalized = math.floor(quantity / step) * step
        return float(normalized if normalized > 0 else 0.0)

    def _order_by_id(self, order_id: str) -> Optional[Order]:
        for order in self._orders:
            if order.order_id == order_id:
                return order
        return None

    def _order_by_broker_id(self, broker_order_id: Optional[str]) -> Optional[Order]:
        if not broker_order_id:
            return None
        for order in self._orders:
            if order.broker_order_id == broker_order_id:
                return order
        return None

    def _get_or_create_position(self, symbol: str) -> Position:
        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol)
        return self._positions[symbol]

    def _normalize_order_status(self, status: Optional[str]) -> str:
        normalized = str(status or "submitted").lower()
        mapping = {
            "new": "submitted",
            "accepted": "submitted",
            "submitted": "submitted",
            "pending": "pending",
            "partially_filled": "submitted",
            "filled": "filled",
            "done": "filled",
            "canceled": "canceled",
            "cancelled": "canceled",
            "rejected": "rejected",
        }
        return mapping.get(normalized, normalized)

    def _apply_order_payload(self, order: Order, payload: Dict[str, Any], timestamp: Any):
        broker_order_id = payload.get("broker_order_id") or payload.get("id") or order.broker_order_id
        order.broker_order_id = str(broker_order_id) if broker_order_id is not None else order.broker_order_id
        order.status = self._normalize_order_status(payload.get("status", order.status))
        order.updated_at = payload.get("updated_at", timestamp)
        order.status_reason = payload.get("status_reason") or payload.get("message") or order.status_reason
        order.filled_quantity = float(payload.get("filled_quantity", order.filled_quantity or 0.0) or 0.0)
        order.avg_fill_price = float(payload.get("avg_fill_price", order.avg_fill_price or 0.0) or 0.0)
        if order.status == "canceled":
            order.canceled_at = payload.get("canceled_at", timestamp)
        if order.status == "rejected":
            order.rejected_at = payload.get("rejected_at", timestamp)
            order.last_error = payload.get("error") or payload.get("message") or order.last_error

    def _sync_account_state(self, timestamp: Any):
        try:
            account_payload = self.client.get_account() or {}
        except NotImplementedError:
            account_payload = {}

        try:
            positions_payload = list(self.client.list_positions() or [])
        except NotImplementedError:
            positions_payload = []

        if account_payload:
            self._account_state["cash"] = float(account_payload.get("cash", self._account_state["cash"]))
            self._account_state["equity"] = float(account_payload.get("equity", self._account_state["equity"]))
            self._account_state["market_value"] = float(account_payload.get("market_value", self._account_state["market_value"]))
            self._account_state["realized_pnl"] = float(account_payload.get("realized_pnl", self._account_state["realized_pnl"]))
            self._account_state["unrealized_pnl"] = float(account_payload.get("unrealized_pnl", self._account_state["unrealized_pnl"]))
            self._account_state["gross_exposure"] = float(account_payload.get("gross_exposure", self._account_state["gross_exposure"]))
            self._account_state["net_exposure"] = float(account_payload.get("net_exposure", self._account_state["net_exposure"]))
            self._account_state["fees_paid"] = float(account_payload.get("fees_paid", self._account_state["fees_paid"]))

        if positions_payload:
            self._positions = {}
            total_market_value = 0.0
            total_gross_exposure = 0.0
            total_unrealized_pnl = 0.0
            for payload in positions_payload:
                symbol = str(payload.get("symbol", ""))
                if not symbol:
                    continue
                position = Position(
                    symbol=symbol,
                    quantity=float(payload.get("quantity", 0.0) or 0.0),
                    avg_price=float(payload.get("avg_price", 0.0) or 0.0),
                    last_price=float(payload.get("last_price", payload.get("mark_price", 0.0)) or 0.0),
                    market_value=float(payload.get("market_value", 0.0) or 0.0),
                    unrealized_pnl=float(payload.get("unrealized_pnl", 0.0) or 0.0),
                    realized_pnl=float(payload.get("realized_pnl", 0.0) or 0.0),
                    opened_at=payload.get("opened_at"),
                    updated_at=payload.get("updated_at", timestamp),
                )
                self._positions[symbol] = position
                total_market_value += float(position.market_value)
                total_gross_exposure += abs(float(position.market_value))
                total_unrealized_pnl += float(position.unrealized_pnl)

            self._account_state["market_value"] = total_market_value
            self._account_state["gross_exposure"] = total_gross_exposure
            self._account_state["net_exposure"] = total_market_value
            self._account_state["unrealized_pnl"] = total_unrealized_pnl
            if not account_payload:
                self._account_state["equity"] = float(self._account_state["cash"] + total_market_value)

    def _register_fill_payload(self, payload: Dict[str, Any], timestamp: Any) -> Optional[Fill]:
        broker_fill_id = payload.get("broker_fill_id") or payload.get("id")
        if broker_fill_id is not None and str(broker_fill_id) in self._fill_ids_by_broker_fill_id:
            return None

        broker_order_id = payload.get("broker_order_id") or payload.get("order_id")
        local_order = self._order_by_broker_id(str(broker_order_id) if broker_order_id is not None else None)
        if local_order is not None:
            fill_quantity = float(payload.get("quantity", 0.0) or 0.0)
            filled_total = float(local_order.filled_quantity) + fill_quantity
            local_order.filled_quantity = min(float(local_order.quantity), filled_total)
            local_order.avg_fill_price = float(payload.get("fill_price", local_order.avg_fill_price or 0.0) or 0.0)
            local_order.status = self._normalize_order_status(payload.get("order_status", local_order.status))
            local_order.updated_at = payload.get("filled_at", payload.get("updated_at", timestamp))

        fill = Fill(
            fill_id=f"FIL-{next(self._fill_counter):06d}",
            order_id=local_order.order_id if local_order is not None else str(payload.get("local_order_id", "remote")),
            broker_order_id=str(broker_order_id) if broker_order_id is not None else None,
            symbol=str(payload.get("symbol", local_order.symbol if local_order is not None else "unknown")),
            side=str(payload.get("side", local_order.side if local_order is not None else "unknown")),
            quantity=float(payload.get("quantity", 0.0) or 0.0),
            fill_price=float(payload.get("fill_price", 0.0) or 0.0),
            commission=float(payload.get("commission", 0.0) or 0.0),
            slippage=float(payload.get("slippage", 0.0) or 0.0),
            tax=float(payload.get("tax", 0.0) or 0.0),
            total_fees=float(payload.get("total_fees", 0.0) or 0.0),
            notional=float(payload.get("notional", 0.0) or 0.0),
            filled_at=payload.get("filled_at", timestamp),
            realized_pnl=float(payload.get("realized_pnl", 0.0) or 0.0),
            gross_realized_pnl=float(payload.get("gross_realized_pnl", 0.0) or 0.0),
            metadata=dict(payload.get("metadata", {}) or {}),
        )
        if broker_fill_id is not None:
            self._fill_ids_by_broker_fill_id.add(str(broker_fill_id))
        self._fills.append(fill)
        return fill

    def create_target_order(
        self,
        symbol: str,
        target_quantity: float,
        timestamp: Any,
        requested_price: Optional[float] = None,
        signal: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Order]:
        current_quantity = float(self.get_position(symbol).quantity)
        target_quantity = float(target_quantity)
        rules = self._rules_for(symbol)
        if not bool(rules.get("allow_short", self.allow_short)) and target_quantity < 0:
            target_quantity = 0.0

        delta = target_quantity - current_quantity
        normalized_quantity = self._normalize_quantity(delta, symbol=symbol)
        if normalized_quantity <= 0:
            return None

        side = "buy" if delta > 0 else "sell"
        order = Order(
            order_id=f"ORD-{next(self._order_counter):06d}",
            symbol=symbol,
            side=side,
            quantity=normalized_quantity,
            submitted_at=timestamp,
            requested_price=requested_price,
            signal=signal,
            updated_at=timestamp,
            metadata=metadata or {},
        )
        self._orders.append(order)
        return order

    def execute_order(self, *args, **kwargs):
        raise NotImplementedError("RealBrokerAdapter 不支持本地 execute_order，请使用 submit_order + process_pending_orders")

    def submit_order(self, order: Order, market_price: float, timestamp: Any):
        payload = {
            "symbol": order.symbol,
            "side": order.side,
            "quantity": float(order.quantity),
            "order_type": order.order_type,
            "requested_price": float(order.requested_price or market_price or 0.0),
            "client_order_id": order.order_id,
            "metadata": dict(order.metadata),
        }
        response = self.client.submit_order(payload) or {}
        order.attempt_count = int(order.attempt_count) + 1
        order.updated_at = timestamp
        self._apply_order_payload(order, response, timestamp)
        if order.status == "pending":
            order.status = "submitted"
        return order

    def cancel_order(self, order_id: str, timestamp: Any, reason: str = "canceled") -> Optional[Order]:
        order = self._order_by_id(order_id)
        if order is None or not order.broker_order_id:
            return order

        response = self.client.cancel_order(order.broker_order_id) or {}
        response.setdefault("status", "canceled")
        response.setdefault("status_reason", reason)
        self._apply_order_payload(order, response, timestamp)
        order.status = "canceled"
        order.canceled_at = timestamp
        return order

    def get_order(self, order_id: str):
        order = self._order_by_id(order_id)
        if order is None or not order.broker_order_id:
            return order
        response = self.client.get_order(order.broker_order_id) or {}
        self._apply_order_payload(order, response, response.get("updated_at", order.updated_at))
        return order

    def get_open_orders(self, symbol: Optional[str] = None):
        try:
            payloads = list(self.client.list_open_orders(symbol=symbol) or [])
        except NotImplementedError:
            payloads = []

        for payload in payloads:
            broker_order_id = payload.get("broker_order_id") or payload.get("id")
            order = self._order_by_broker_id(str(broker_order_id) if broker_order_id is not None else None)
            if order is None:
                order = Order(
                    order_id=str(payload.get("client_order_id") or f"ORD-{next(self._order_counter):06d}"),
                    symbol=str(payload.get("symbol", symbol or "unknown")),
                    side=str(payload.get("side", "unknown")),
                    quantity=float(payload.get("quantity", 0.0) or 0.0),
                    submitted_at=payload.get("submitted_at"),
                    requested_price=payload.get("requested_price"),
                    status="submitted",
                    metadata=dict(payload.get("metadata", {}) or {}),
                )
                self._orders.append(order)
            self._apply_order_payload(order, payload, payload.get("updated_at"))

        open_orders = [order for order in self._orders if order.status in {"pending", "submitted"}]
        if symbol is not None:
            open_orders = [order for order in open_orders if order.symbol == symbol]
        return open_orders

    def process_pending_orders(self, market_prices: Dict[str, float], timestamp: Any):
        try:
            fills_payload = list(self.client.list_fills(symbol=list(market_prices.keys())[0] if len(market_prices) == 1 else None) or [])
        except NotImplementedError:
            fills_payload = []

        new_fills = []
        for payload in fills_payload:
            fill = self._register_fill_payload(payload, timestamp)
            if fill is not None:
                new_fills.append(fill)

        self._sync_account_state(timestamp)
        return new_fills

    def mark_to_market(self, symbol: str, price: float, timestamp: Any) -> Position:
        position = self._get_or_create_position(symbol)
        position.last_price = float(price)
        position.updated_at = timestamp
        if abs(position.quantity) > 1e-12 and abs(position.market_value) < 1e-12:
            position.market_value = float(position.quantity * price)
            if position.quantity > 0:
                position.unrealized_pnl = float((price - position.avg_price) * position.quantity)
            else:
                position.unrealized_pnl = float((position.avg_price - price) * abs(position.quantity))
        return position

    def get_account_snapshot(self, timestamp: Any):
        self._sync_account_state(timestamp)
        active_positions = int(sum(1 for position in self._positions.values() if abs(position.quantity) > 1e-12))
        return AccountSnapshot(
            timestamp=timestamp,
            cash=float(self._account_state.get("cash", self._initial_cash)),
            market_value=float(self._account_state.get("market_value", 0.0)),
            equity=float(self._account_state.get("equity", self._initial_cash)),
            realized_pnl=float(self._account_state.get("realized_pnl", 0.0)),
            unrealized_pnl=float(self._account_state.get("unrealized_pnl", 0.0)),
            gross_exposure=float(self._account_state.get("gross_exposure", 0.0)),
            net_exposure=float(self._account_state.get("net_exposure", 0.0)),
            active_positions=active_positions,
            fees_paid=float(self._account_state.get("fees_paid", 0.0)),
            open_orders=len(self.get_open_orders()),
        )

    def get_position(self, symbol: str):
        return self._get_or_create_position(symbol)

    def get_positions(self):
        return {
            symbol: position.to_dict()
            for symbol, position in self._positions.items()
            if abs(position.quantity) > 1e-12 or abs(position.realized_pnl) > 1e-12
        }



def _build_cost_model(config):
    risk_config = config.get("risk", {})
    cost_config = config.get("execution_costs", {})
    return ExecutionCostModel(
        commission_rate=float(risk_config.get("commission", 0.001)),
        slippage_rate=float(risk_config.get("slippage", 0.0005)),
        fixed_commission=float(cost_config.get("fixed_commission", 0.0)),
        min_commission=float(cost_config.get("min_commission", 0.0)),
        sell_tax_rate=float(cost_config.get("sell_tax_rate", 0.0)),
    )



def build_paper_adapter(config, broker: Optional[PaperBroker] = None) -> PaperBrokerAdapter:
    paper_config = config.get("paper_trading", {})
    symbol = config.get("data", {}).get("symbol")
    market_resolver = MarketProfileResolver(config)
    market_settings = market_resolver.section_settings(symbol=symbol, section_name="paper_trading")

    if broker is None:
        broker = PaperBroker(
            initial_cash=float(
                paper_config.get(
                    "initial_cash",
                    config.get("backtest", {}).get("initial_capital", 100000.0),
                )
            ),
            allow_short=bool(market_settings.get("allow_short", True)),
            allow_fractional=bool(market_settings.get("allow_fractional", False)),
            lot_size=float(market_settings.get("lot_size", 1.0)),
            cost_model=_build_cost_model(config),
            market_resolver=market_resolver,
            market_section="paper_trading",
        )
    return PaperBrokerAdapter(broker)



def build_live_adapter(
    config,
    broker: Optional[PaperLiveBroker] = None,
    client: Optional[BrokerAPIClient] = None,
):
    paper_config = config.get("paper_trading", {})
    live_config = config.get("live_trading", {})
    adapter_type = str(live_config.get("adapter", "paper_live")).lower()
    symbol = config.get("data", {}).get("symbol")
    market_resolver = MarketProfileResolver(config)
    market_settings = market_resolver.section_settings(symbol=symbol, section_name="live_trading")

    if adapter_type == "real":
        if client is None:
            client = ConfigurableRESTBrokerClient(config)
        return RealBrokerAdapter(
            client=client,
            initial_cash=float(
                live_config.get(
                    "initial_cash",
                    paper_config.get(
                        "initial_cash",
                        config.get("backtest", {}).get("initial_capital", 100000.0),
                    ),
                )
            ),
            allow_short=bool(market_settings.get("allow_short", True)),
            allow_fractional=bool(market_settings.get("allow_fractional", False)),
            lot_size=float(market_settings.get("lot_size", 1.0)),
            market_resolver=market_resolver,
            market_section="live_trading",
        )

    if broker is None:
        broker = PaperLiveBroker(
            initial_cash=float(
                live_config.get(
                    "initial_cash",
                    paper_config.get(
                        "initial_cash",
                        config.get("backtest", {}).get("initial_capital", 100000.0),
                    ),
                )
            ),
            allow_short=bool(market_settings.get("allow_short", True)),
            allow_fractional=bool(market_settings.get("allow_fractional", False)),
            lot_size=float(market_settings.get("lot_size", 1.0)),
            cost_model=_build_cost_model(config),
            fill_delay_seconds=int(live_config.get("fill_delay_seconds", 0)),
            reject_first_n_orders=int(live_config.get("reject_first_n_orders", 0)),
            market_resolver=market_resolver,
            market_section="live_trading",
        )
    return PaperLiveBrokerAdapter(broker)
