"""LLM-backed structured event factor extraction.

This module keeps LLM usage on the research side: it produces structured features
that can be merged into the existing signal pipeline, but it never emits orders.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


DEFAULT_EVENT_FACTOR = {
    "headline": "",
    "sentiment_score": 0.0,
    "event_risk_score": 0.0,
    "confidence_score": 0.0,
    "direction_hint": "neutral",
    "source": "disabled",
    "llm_used": False,
}


def _safe_float(value: Any, minimum: float = -1.0, maximum: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return float(max(min(numeric, maximum), minimum))


def _normalize_direction(value: Any) -> str:
    normalized = str(value or "neutral").strip().lower()
    if normalized in {"bullish", "positive", "buy", "long"}:
        return "bullish"
    if normalized in {"bearish", "negative", "sell", "short"}:
        return "bearish"
    return "neutral"


def _build_fallback_factor(reason: str) -> Dict[str, Any]:
    payload = dict(DEFAULT_EVENT_FACTOR)
    payload["source"] = reason
    return payload


def _build_prompt(symbol: str, latest_snapshot: Dict[str, Any]) -> str:
    payload = {
        "symbol": symbol,
        "latest_snapshot": latest_snapshot,
    }
    return (
        "输出严格 json，不要 markdown。"
        "字段: headline, sentiment_score, event_risk_score, confidence_score, direction_hint。"
        "sentiment_score 范围 [-1,1]，event_risk_score 范围 [0,1]，confidence_score 范围 [0,1]。"
        "direction_hint 只能是 bullish、bearish、neutral。"
        "如果没有外部事件信息，就基于当前行情快照给出保守判断。数据如下:\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'), default=str)}"
    )


def _parse_factor(raw_text: str) -> Optional[Dict[str, Any]]:
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    factor = dict(DEFAULT_EVENT_FACTOR)
    factor.update(
        {
            "headline": str(parsed.get("headline", "") or ""),
            "sentiment_score": _safe_float(parsed.get("sentiment_score"), minimum=-1.0, maximum=1.0),
            "event_risk_score": _safe_float(parsed.get("event_risk_score"), minimum=0.0, maximum=1.0),
            "confidence_score": _safe_float(parsed.get("confidence_score"), minimum=0.0, maximum=1.0),
            "direction_hint": _normalize_direction(parsed.get("direction_hint")),
            "source": "llm_event_factor",
            "llm_used": True,
        }
    )
    return factor


def generate_llm_event_factor(
    config: Dict[str, Any],
    symbol: str,
    latest_snapshot: Dict[str, Any],
    llm_client: Optional[Any] = None,
) -> Dict[str, Any]:
    llm_config = dict(config.get("llm", {}) or {})
    if not bool(llm_config.get("enabled", False)):
        return _build_fallback_factor("disabled")
    if not bool(llm_config.get("event_factor_enabled", False)):
        return _build_fallback_factor("event_factor_disabled")
    if llm_client is None or not getattr(llm_client, "is_available", lambda: False)():
        return _build_fallback_factor("llm_unavailable")

    try:
        raw_text = llm_client.chat(
            system_prompt=(
                "你是谨慎的金融事件分析助手。"
                "只输出结构化因子，不直接给出下单指令。"
            ),
            user_prompt=_build_prompt(symbol, latest_snapshot),
            max_tokens=int(llm_config.get("event_factor_max_tokens", 300)),
            timeout=float(llm_config.get("event_factor_timeout", llm_config.get("timeout", 30.0))),
            model=llm_config.get("event_factor_model", llm_config.get("model")),
            response_format={"type": "json_object"},
            thinking=str(llm_config.get("event_factor_thinking", llm_config.get("thinking", "disabled"))),
            reasoning_effort=llm_config.get("event_factor_reasoning_effort", llm_config.get("reasoning_effort")),
        )
        parsed = _parse_factor(raw_text or "")
        if not parsed:
            fallback = _build_fallback_factor("llm_parse_error")
            fallback["llm_raw_text"] = raw_text
            return fallback
        return parsed
    except Exception as exc:  # pragma: no cover
        fallback = _build_fallback_factor("llm_error")
        fallback["llm_error"] = str(exc)
        return fallback
