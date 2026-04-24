import json
import logging
from typing import Any, Dict, List, Optional

from src.utils.openai_client import OpenAIClient


logger = logging.getLogger(__name__)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _percent(value: Any) -> str:
    return f"{_to_float(value) * 100:.2f}%"


def _round_metric(value: Any) -> float:
    return round(_to_float(value), 6)


def _build_local_review(
    symbol: str,
    performance: Dict[str, Any],
    risk_metrics: Dict[str, Any],
    recommendations: List[str],
    trade_statistics: Dict[str, Any],
    strategy_params: Dict[str, Any],
) -> Dict[str, Any]:
    total_return = _to_float(performance.get("total_return"))
    sharpe_ratio = _to_float(performance.get("sharpe_ratio"))
    max_drawdown = _to_float(risk_metrics.get("max_drawdown"))
    benchmark_return = performance.get("benchmark_return")
    win_rate = _to_float(performance.get("win_rate"))
    total_trades = int(_to_float(performance.get("total_trades")))
    profit_factor = _to_float(performance.get("profit_factor"))

    headline = "策略表现偏弱，需要先降低风险暴露"
    if total_return > 0 and sharpe_ratio >= 1.0 and max_drawdown > -0.15:
        headline = "策略表现可继续观察，但仍需扩大样本验证"
    elif total_return > 0:
        headline = "策略有正收益，但风险收益比还不稳定"

    key_findings = [
        f"{symbol} 本轮收益为 {_percent(total_return)}，夏普比率为 {sharpe_ratio:.2f}。",
        f"最大回撤为 {_percent(max_drawdown)}，胜率为 {_percent(win_rate)}，交易次数为 {total_trades}。",
    ]
    if benchmark_return is not None:
        active_return = total_return - _to_float(benchmark_return)
        key_findings.append(
            f"同期基准收益为 {_percent(benchmark_return)}，策略相对收益为 {_percent(active_return)}。"
        )
    if profit_factor:
        key_findings.append(f"盈亏比为 {profit_factor:.2f}，低于 1 代表亏损交易仍占主导。")

    risk_notes = []
    if max_drawdown <= -0.2:
        risk_notes.append("最大回撤已经超过 20%，当前参数不适合直接进入实盘。")
    if sharpe_ratio < 0:
        risk_notes.append("夏普比率为负，说明承担波动后没有获得正向补偿。")
    if win_rate < 0.4 and total_trades >= 10:
        risk_notes.append("胜率偏低且交易次数不低，信号过滤条件需要收紧。")
    if not risk_notes:
        risk_notes.append("当前风险指标没有触发严重警戒，但仍需用更长区间回测。")

    parameter_suggestions = [
        {
            "name": "position_size",
            "current": _round_metric(strategy_params.get("position_size")),
            "suggested": round(min(_to_float(strategy_params.get("position_size"), 1.0), 0.5), 4),
            "reason": "先降低单次信号对账户权益的影响，避免回撤继续扩大。",
        },
        {
            "name": "allow_short",
            "current": bool(strategy_params.get("allow_short", False)),
            "suggested": False,
            "reason": "AAPL 在样本期处于强基准环境，当前做空交易拖累明显，建议先关闭做空重新评估。",
        },
        {
            "name": "rsi_long_threshold",
            "current": _round_metric(strategy_params.get("rsi_long_threshold")),
            "suggested": max(_to_float(strategy_params.get("rsi_long_threshold"), 55.0), 58.0),
            "reason": "提高做多门槛，减少震荡行情里的过早进场。",
        },
        {
            "name": "stop_loss_pct",
            "current": _round_metric(strategy_params.get("stop_loss_pct")),
            "suggested": min(_to_float(strategy_params.get("stop_loss_pct"), 0.03), 0.025),
            "reason": "亏损交易较多时先缩短容错距离，控制单笔亏损。",
        },
    ]

    next_steps = [
        "先用以上参数建议跑一轮 research，对比 total_return、sharpe_ratio 和 max_drawdown。",
        "再跑 paper 模式检查订单、成交和资金曲线是否同步改善。",
        "在至少两个不同时间区间验证前，不建议接入真实券商下单。",
    ]

    return {
        "source": "local_rules",
        "llm_used": False,
        "headline": headline,
        "key_findings": key_findings,
        "risk_notes": risk_notes,
        "parameter_suggestions": parameter_suggestions,
        "next_steps": next_steps,
        "base_recommendations": recommendations,
        "trade_statistics": trade_statistics,
    }


def _build_llm_prompt(
    symbol: str,
    performance: Dict[str, Any],
    risk_metrics: Dict[str, Any],
    recommendations: List[str],
    trade_statistics: Dict[str, Any],
    strategy_params: Dict[str, Any],
) -> str:
    payload = {
        "symbol": symbol,
        "performance": {key: _round_metric(value) for key, value in performance.items()},
        "risk_metrics": {key: _round_metric(value) for key, value in risk_metrics.items()},
        "trade_statistics": trade_statistics,
        "strategy_params": strategy_params,
        "base_recommendations": recommendations[:3],
    }
    return (
        "输出严格 JSON，不要 Markdown。字段: headline, key_findings, risk_notes, "
        "parameter_suggestions, next_steps。每个数组最多 3 项，中文短句。"
        "parameter_suggestions 每项包含 name,current,suggested,reason。"
        "禁止承诺收益，禁止建议直接实盘下单。数据如下:\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'), default=str)}"
    )


def _parse_llm_review(raw_text: str) -> Optional[Dict[str, Any]]:
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def generate_ai_review(
    config: Dict[str, Any],
    symbol: str,
    performance: Dict[str, Any],
    risk_metrics: Dict[str, Any],
    recommendations: List[str],
    backtest_results: Dict[str, Any],
    strategy_params: Dict[str, Any],
    llm_client: Optional[Any] = None,
    create_llm_client: bool = True,
) -> Dict[str, Any]:
    trade_statistics = dict(backtest_results.get("trade_statistics", {}) or {})
    local_review = _build_local_review(
        symbol=symbol,
        performance=performance,
        risk_metrics=risk_metrics,
        recommendations=recommendations,
        trade_statistics=trade_statistics,
        strategy_params=strategy_params,
    )

    client = llm_client
    if client is None and create_llm_client and config.get("llm", {}).get("enabled", False):
        client = OpenAIClient(config)

    if client is None or not getattr(client, "is_available", lambda: False)():
        return local_review

    try:
        llm_config = config.get("llm", {})
        raw_text = client.chat(
            system_prompt=(
                "你是谨慎的量化交易复盘助手，只做研究复盘、风险解释和参数建议。"
                "输出必须简洁、可执行、无收益承诺。"
            ),
            user_prompt=_build_llm_prompt(
                symbol=symbol,
                performance=performance,
                risk_metrics=risk_metrics,
                recommendations=recommendations,
                trade_statistics=trade_statistics,
                strategy_params=strategy_params,
            ),
            max_tokens=int(llm_config.get("review_max_tokens", 600)),
            timeout=float(llm_config.get("review_timeout", llm_config.get("timeout", 12.0))),
            model=llm_config.get("review_model"),
        )
        parsed = _parse_llm_review(raw_text or "")
        if not parsed:
            local_review.update(
                {
                    "source": "local_rules_after_llm_parse_error",
                    "llm_used": False,
                    "llm_error": "LLM 返回内容不是有效 JSON",
                    "llm_raw_text": raw_text,
                }
            )
            return local_review

        parsed.setdefault("headline", local_review["headline"])
        parsed.setdefault("key_findings", local_review["key_findings"])
        parsed.setdefault("risk_notes", local_review["risk_notes"])
        parsed.setdefault("parameter_suggestions", local_review["parameter_suggestions"])
        parsed.setdefault("next_steps", local_review["next_steps"])
        parsed.update(
            {
                "source": "llm",
                "llm_used": True,
                "base_recommendations": recommendations,
                "trade_statistics": trade_statistics,
            }
        )
        return parsed
    except Exception as exc:  # pragma: no cover
        logger.warning("LLM 复盘失败，已降级为本地规则复盘: %s", exc)
        local_review.update(
            {
                "source": "local_rules_after_llm_error",
                "llm_used": False,
                "llm_error": str(exc),
            }
        )
        return local_review
