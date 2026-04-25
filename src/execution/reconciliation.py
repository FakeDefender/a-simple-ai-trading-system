"""Reconciliation and operator reporting helpers for live execution."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


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


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _latest_account_snapshot(results: Dict[str, Any]) -> Dict[str, Any]:
    account_history = results.get("account_history")
    if isinstance(account_history, pd.DataFrame) and not account_history.empty:
        row = account_history.iloc[-1].to_dict()
        row["timestamp"] = account_history.index[-1]
        return _to_json_safe(row)
    summary = dict(results.get("summary", {}) or {})
    if summary:
        summary["timestamp"] = datetime.now().isoformat(timespec="seconds")
    return _to_json_safe(summary)


def _normalize_positions(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    normalized: Dict[str, Dict[str, Any]] = {}
    for symbol, data in dict(payload or {}).items():
        if not isinstance(data, dict):
            continue
        normalized[str(symbol)] = {
            "quantity": _safe_float(data.get("quantity")),
            "avg_price": _safe_float(data.get("avg_price")),
            "market_value": _safe_float(data.get("market_value")),
            "realized_pnl": _safe_float(data.get("realized_pnl")),
            "unrealized_pnl": _safe_float(data.get("unrealized_pnl")),
        }
    return normalized


def build_reconciliation_report(
    results: Dict[str, Any],
    adapter: Any,
    timestamp: Any,
    quantity_tolerance: float = 1e-9,
    account_tolerance: float = 1e-6,
) -> Dict[str, Any]:
    reported_account = _latest_account_snapshot(results)
    broker_account = _to_json_safe(adapter.get_account_snapshot(timestamp).to_dict())
    reported_positions = _normalize_positions(results.get("positions", {}))
    broker_positions = _normalize_positions(adapter.get_positions())

    mismatches: List[Dict[str, Any]] = []
    for key in ["cash", "equity", "realized_pnl", "unrealized_pnl", "fees_paid", "open_orders"]:
        reported_value = _safe_float(reported_account.get(key))
        broker_value = _safe_float(broker_account.get(key))
        if abs(reported_value - broker_value) > account_tolerance:
            mismatches.append(
                {
                    "scope": "account",
                    "field": key,
                    "reported": reported_value,
                    "broker": broker_value,
                }
            )

    all_symbols = sorted(set(reported_positions) | set(broker_positions))
    for symbol in all_symbols:
        reported_position = reported_positions.get(symbol, {})
        broker_position = broker_positions.get(symbol, {})
        reported_quantity = _safe_float(reported_position.get("quantity"))
        broker_quantity = _safe_float(broker_position.get("quantity"))
        if abs(reported_quantity - broker_quantity) > quantity_tolerance:
            mismatches.append(
                {
                    "scope": "position",
                    "symbol": symbol,
                    "field": "quantity",
                    "reported": reported_quantity,
                    "broker": broker_quantity,
                }
            )

    return {
        "timestamp": _to_json_safe(timestamp),
        "status": "ok" if not mismatches else "mismatch",
        "account_tolerance": float(account_tolerance),
        "quantity_tolerance": float(quantity_tolerance),
        "reported_account": reported_account,
        "broker_account": broker_account,
        "reported_positions": reported_positions,
        "broker_positions": broker_positions,
        "mismatches": mismatches,
    }


def build_operator_report(
    payload: Dict[str, Any],
    reconciliation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summary = dict(payload.get("summary", {}) or {})
    recent_events = list(payload.get("recent_events", []) or [])
    positions = dict(payload.get("results", {}).get("positions", {}) or {})

    active_positions = [
        {
            "symbol": symbol,
            "quantity": _safe_float(data.get("quantity")),
            "market_value": _safe_float(data.get("market_value")),
            "unrealized_pnl": _safe_float(data.get("unrealized_pnl")),
        }
        for symbol, data in positions.items()
        if abs(_safe_float(data.get("quantity"))) > 1e-12
    ]
    active_positions.sort(key=lambda item: abs(item["market_value"]), reverse=True)
    alert_events = [event for event in recent_events if str(event.get("severity", "info")) in {"warning", "critical"}]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "symbols": list(payload.get("symbols") or [payload.get("symbol", "unknown")]),
        "latest_timestamp": _to_json_safe(payload.get("latest_timestamp")),
        "processed_rows": int(payload.get("processed_rows", 0) or 0),
        "final_equity": _safe_float(summary.get("final_equity")),
        "total_return": _safe_float(summary.get("total_return")),
        "max_drawdown": _safe_float(summary.get("max_drawdown")),
        "orders": int(summary.get("orders", 0) or 0),
        "fills": int(summary.get("fills", 0) or 0),
        "rejected_orders": int(summary.get("rejected_orders", 0) or 0),
        "canceled_orders": int(summary.get("canceled_orders", 0) or 0),
        "session_blocks": int(summary.get("session_blocks", 0) or 0),
        "risk_blocks": int(summary.get("risk_blocks", 0) or 0),
        "reconciliation_status": (reconciliation or {}).get("status", "unknown"),
        "active_positions": active_positions[:10],
        "recent_event_count": len(recent_events),
        "recent_alert_count": len(alert_events),
        "recent_alerts": alert_events[-10:],
    }
