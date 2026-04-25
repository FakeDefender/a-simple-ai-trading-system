"""Operational observability primitives for live execution."""

from __future__ import annotations

import json
import os
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return value


def _normalize_timestamp(value: Any) -> str:
    if value is None:
        return datetime.now().isoformat(timespec="seconds")
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return str(value)


@dataclass(frozen=True)
class TradingEvent:
    event_type: str
    timestamp: str
    source: str
    severity: str = "info"
    symbol: Optional[str] = None
    message: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": str(self.event_type),
            "timestamp": str(self.timestamp),
            "source": str(self.source),
            "severity": str(self.severity),
            "symbol": self.symbol,
            "message": self.message,
            "payload": _to_json_safe(self.payload),
        }


class EventBus:
    def __init__(self, max_recent_events: int = 2000):
        self._subscribers: List[Callable[[TradingEvent], None]] = []
        self._recent_events: Deque[TradingEvent] = deque(maxlen=max(int(max_recent_events), 1))

    def subscribe(self, handler: Callable[[TradingEvent], None]):
        self._subscribers.append(handler)

    def publish(self, event: TradingEvent):
        self._recent_events.append(event)
        for subscriber in list(self._subscribers):
            subscriber(event)

    def snapshot(self, limit: int = 100) -> List[Dict[str, Any]]:
        limit = max(int(limit), 1)
        return [event.to_dict() for event in list(self._recent_events)[-limit:]]


class JsonlEventJournal:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def __call__(self, event: TradingEvent):
        payload = json.dumps(event.to_dict(), ensure_ascii=False, default=str)
        with self._lock:
            with open(self.file_path, "a", encoding="utf-8") as file:
                file.write(payload + "\n")


class RuleBasedAlertSink:
    def __init__(self, file_path: str, drawdown_alert_threshold: float = 0.02):
        self.file_path = file_path
        self.drawdown_alert_threshold = float(drawdown_alert_threshold)
        self._lock = threading.Lock()
        self._drawdown_alert_active: Dict[str, bool] = {}
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def _write_alert(self, event: TradingEvent, alert_type: str, severity: str, message: str):
        payload = {
            "alert_type": alert_type,
            "severity": severity,
            "timestamp": event.timestamp,
            "source": event.source,
            "symbol": event.symbol,
            "message": message,
            "event_type": event.event_type,
            "payload": _to_json_safe(event.payload),
        }
        with self._lock:
            with open(self.file_path, "a", encoding="utf-8") as file:
                file.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def __call__(self, event: TradingEvent):
        symbol_key = str(event.symbol or "_global")
        if event.event_type == "account_snapshot":
            drawdown = float(event.payload.get("daily_drawdown", 0.0) or 0.0)
            if drawdown >= self.drawdown_alert_threshold:
                if not self._drawdown_alert_active.get(symbol_key, False):
                    self._drawdown_alert_active[symbol_key] = True
                    self._write_alert(
                        event,
                        alert_type="drawdown_threshold",
                        severity="warning",
                        message=f"日内回撤达到 {drawdown:.2%}",
                    )
            else:
                self._drawdown_alert_active[symbol_key] = False
            return

        alert_mapping = {
            "order_rejected": ("order_rejected", "warning", "订单被拒绝"),
            "order_canceled": ("order_canceled", "warning", "订单被撤销"),
            "risk_halt": ("risk_halt", "critical", "触发风控暂停"),
            "service_cycle_failed": ("service_cycle_failed", "critical", "live service 周期失败"),
            "reconciliation_mismatch": ("reconciliation_mismatch", "critical", "账户或持仓对账不一致"),
        }
        if event.event_type in alert_mapping:
            alert_type, severity, message = alert_mapping[event.event_type]
            self._write_alert(event, alert_type=alert_type, severity=severity, message=message)


def build_event(
    event_type: str,
    source: str,
    timestamp: Any = None,
    severity: str = "info",
    symbol: Optional[str] = None,
    message: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> TradingEvent:
    return TradingEvent(
        event_type=event_type,
        timestamp=_normalize_timestamp(timestamp),
        source=source,
        severity=severity,
        symbol=symbol,
        message=message,
        payload=dict(payload or {}),
    )
