import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None
import numpy as np
import pandas as pd

from src.utils.data_loader import DataLoader
from src.utils.data_processor import DataProcessor
from src.utils.risk_calculator import RiskCalculator


logger = logging.getLogger(__name__)


class MLStrategyAgent:
    """v0.1 基线策略代理。

    当前版本默认使用离线、可解释、可回测的规则策略。
    LLM 仅保留为未来增强入口，不再作为主链路前置条件。
    """

    def __init__(self, config: Dict[str, Any], data_loader: DataLoader, strategy_params: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.data_loader = data_loader
        self.logger = logging.getLogger(__name__)
        self.data_processor = DataProcessor()
        self.risk_calculator = RiskCalculator(self.config.get("risk", {}))
        self.results_dir = os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S"))

        strategy_config = self.config.get("strategy", {})
        risk_config = self.config.get("risk", {})
        self.strategy_params = {
            "fast_ma": int(strategy_config.get("fast_ma", 10)),
            "slow_ma": int(strategy_config.get("slow_ma", 20)),
            "rsi_long_threshold": float(strategy_config.get("rsi_long_threshold", 55)),
            "rsi_short_threshold": float(strategy_config.get("rsi_short_threshold", 45)),
            "rsi_exit_long": float(strategy_config.get("rsi_exit_long", 48)),
            "rsi_exit_short": float(strategy_config.get("rsi_exit_short", 52)),
            "min_volume_ratio": float(strategy_config.get("min_volume_ratio", 1.0)),
            "allow_short": bool(strategy_config.get("allow_short", True)),
            "stop_loss_pct": float(risk_config.get("stop_loss_pct", 0.03)),
            "take_profit_pct": float(risk_config.get("take_profit_pct", 0.06)),
            "position_size": float(risk_config.get("position_size", 1.0)),
            "commission": float(risk_config.get("commission", 0.001)),
            "slippage": float(risk_config.get("slippage", 0.0005)),
        }
        if strategy_params:
            self.strategy_params.update(strategy_params)

        self.llm = self._initialize_llm()

    def set_params(self, **kwargs):
        self.strategy_params.update(kwargs)

    def _initialize_llm(self):
        if not self.config.get("llm", {}).get("enabled", False):
            return None

        try:
            from src.utils.openai_client import OpenAIClient

            client = OpenAIClient(self.config)
            if client.is_available():
                self.logger.info("LLM 客户端已启用")
                return client
            self.logger.warning("LLM 已配置但当前不可用，继续使用离线模式")
            return None
        except Exception as exc:  # pragma: no cover
            self.logger.warning(f"初始化 LLM 失败，继续使用离线模式: {exc}")
            return None

    def _get_agent_advice(self, data: pd.DataFrame) -> Dict[str, Any]:
        if data is None or data.empty:
            return {
                "strategy_developer": {"parameters": {}},
                "risk_analyst": {"risk_level": "unknown"},
                "trading_advisor": {"current_signal": "neutral"},
            }

        market_indicators = self.data_processor.calculate_market_indicators(data)
        latest = data.iloc[-1]
        trend_strength = float(market_indicators.get("trend_strength", 0.0))
        market_strength = float(market_indicators.get("market_strength", 0.5))
        volatility = float(market_indicators.get("volatility", 0.0))
        volume_ratio = float(latest.get("volume_ratio", 1.0))

        risk_level = "low"
        if volatility > 0.35 or abs(trend_strength) < 0.003:
            risk_level = "high"
        elif volatility > 0.2:
            risk_level = "moderate"

        position_size = self.strategy_params["position_size"]
        if risk_level == "high":
            position_size *= 0.4
        elif risk_level == "moderate":
            position_size *= 0.7

        current_signal = "neutral"
        if trend_strength > 0.01 and market_strength > 0.55:
            current_signal = "buy"
        elif self.strategy_params["allow_short"] and trend_strength < -0.01 and market_strength < 0.45:
            current_signal = "sell"

        current_price = float(latest["close"])
        atr = float(latest.get("atr", current_price * 0.02))

        return {
            "strategy_developer": {
                "strategy_name": "baseline_trend_following",
                "strategy_description": "基于均线、RSI、MACD 和成交量过滤的离线基线策略",
                "parameters": {
                    "trend_threshold": round(abs(trend_strength), 4),
                    "market_strength_threshold": round(market_strength, 4),
                    "volatility_threshold": round(volatility, 4),
                    "position_size": round(position_size, 4),
                    "stop_loss_atr": round(max(1.0, self.strategy_params["stop_loss_pct"] * current_price / max(atr, 1e-6)), 4),
                    "take_profit_atr": round(max(1.0, self.strategy_params["take_profit_pct"] * current_price / max(atr, 1e-6)), 4),
                    "volume_ratio": round(volume_ratio, 4),
                },
            },
            "risk_analyst": {
                "market_state": {
                    "trend_strength": trend_strength,
                    "market_strength": market_strength,
                    "volatility": volatility,
                },
                "risk_level": risk_level,
                "risk_management": {
                    "max_drawdown": float(self.config.get("risk", {}).get("max_drawdown", 0.2)),
                    "daily_loss_limit": float(self.config.get("risk", {}).get("daily_loss_limit", 0.05)),
                },
                "position_suggestions": {
                    "position_size": round(position_size, 4),
                    "stop_loss": round(current_price * (1 - self.strategy_params["stop_loss_pct"]), 4),
                    "take_profit": round(current_price * (1 + self.strategy_params["take_profit_pct"]), 4),
                },
            },
            "trading_advisor": {
                "current_signal": current_signal,
                "position_size": round(position_size, 4),
                "stop_loss": round(current_price * (1 - self.strategy_params["stop_loss_pct"]), 4),
                "take_profit": round(current_price * (1 + self.strategy_params["take_profit_pct"]), 4),
                "entry_points": [round(current_price, 4)],
                "exit_points": [
                    round(current_price * (1 - self.strategy_params["stop_loss_pct"]), 4),
                    round(current_price * (1 + self.strategy_params["take_profit_pct"]), 4),
                ],
                "time_horizon": "swing",
            },
        }

    def _check_long_entry_conditions(self, current_data: pd.Series, market_indicators: Dict[str, Any]) -> bool:
        return all(
            [
                float(current_data.get("ma5", 0.0)) >= float(current_data.get("ma20", 0.0)),
                float(current_data.get("rsi", 50.0)) >= self.strategy_params["rsi_long_threshold"],
                float(current_data.get("macd", 0.0)) >= float(current_data.get("signal", 0.0)),
                float(current_data.get("volume_ratio", 1.0)) >= self.strategy_params["min_volume_ratio"],
                float(market_indicators.get("trend_strength", 0.0)) > 0,
            ]
        )

    def _check_short_entry_conditions(self, current_data: pd.Series, market_indicators: Dict[str, Any]) -> bool:
        return self.strategy_params["allow_short"] and all(
            [
                float(current_data.get("ma5", 0.0)) <= float(current_data.get("ma20", 0.0)),
                float(current_data.get("rsi", 50.0)) <= self.strategy_params["rsi_short_threshold"],
                float(current_data.get("macd", 0.0)) <= float(current_data.get("signal", 0.0)),
                float(current_data.get("volume_ratio", 1.0)) >= self.strategy_params["min_volume_ratio"],
                float(market_indicators.get("trend_strength", 0.0)) < 0,
            ]
        )

    def _check_long_exit_conditions(self, current_data: pd.Series) -> bool:
        return any(
            [
                float(current_data.get("ma5", 0.0)) < float(current_data.get("ma20", 0.0)),
                float(current_data.get("rsi", 50.0)) <= self.strategy_params["rsi_exit_long"],
                float(current_data.get("macd", 0.0)) < float(current_data.get("signal", 0.0)),
            ]
        )

    def _check_short_exit_conditions(self, current_data: pd.Series) -> bool:
        return any(
            [
                float(current_data.get("ma5", 0.0)) > float(current_data.get("ma20", 0.0)),
                float(current_data.get("rsi", 50.0)) >= self.strategy_params["rsi_exit_short"],
                float(current_data.get("macd", 0.0)) > float(current_data.get("signal", 0.0)),
            ]
        )

    def generate_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        if market_data is None or market_data.empty:
            raise ValueError("市场数据为空，无法生成交易信号")

        self.logger.info(f"开始生成交易信号，数据行数: {len(market_data)}")
        signals = pd.DataFrame(index=market_data.index)
        signals["signal"] = 0
        signals["market_strength"] = 0.5
        signals["risk_level"] = "unknown"
        signals["advisor_signal"] = "neutral"

        current_position = 0
        for i in range(len(market_data)):
            history = market_data.iloc[: i + 1]
            current_data = history.iloc[-1]
            indicators = self.data_processor.calculate_market_indicators(history)
            advice = self._get_agent_advice(history)
            advisor_signal = advice["trading_advisor"].get("current_signal", "neutral")
            risk_level = advice["risk_analyst"].get("risk_level", "unknown")
            market_strength = float(indicators.get("market_strength", 0.5))

            target_position = current_position
            if current_position == 0:
                if self._check_long_entry_conditions(current_data, indicators):
                    target_position = 1
                elif self._check_short_entry_conditions(current_data, indicators):
                    target_position = -1
            elif current_position == 1 and self._check_long_exit_conditions(current_data):
                target_position = 0
            elif current_position == -1 and self._check_short_exit_conditions(current_data):
                target_position = 0

            current_position = target_position
            signals.iloc[i, signals.columns.get_loc("signal")] = target_position
            signals.iloc[i, signals.columns.get_loc("market_strength")] = market_strength
            signals.iloc[i, signals.columns.get_loc("risk_level")] = risk_level
            signals.iloc[i, signals.columns.get_loc("advisor_signal")] = advisor_signal

        self.logger.info(
            "信号统计 - 多头持仓: %s, 空头持仓: %s, 空仓: %s",
            int((signals["signal"] == 1).sum()),
            int((signals["signal"] == -1).sum()),
            int((signals["signal"] == 0).sum()),
        )
        return signals

    def _close_position(
        self,
        capital: float,
        position: int,
        entry_price: float,
        exit_price: float,
        entry_time,
        exit_time,
        exit_reason: str,
    ) -> (float, Dict[str, Any]):
        gross_profit_pct = (exit_price - entry_price) / entry_price if position == 1 else (entry_price - exit_price) / entry_price
        cost_pct = 2 * (self.strategy_params["commission"] + self.strategy_params["slippage"])
        net_profit_pct = gross_profit_pct - cost_pct
        new_capital = capital * (1 + net_profit_pct * self.strategy_params["position_size"])
        trade = {
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "gross_profit_pct": float(gross_profit_pct),
            "profit_pct": float(net_profit_pct),
            "holding_days": max(1, int((exit_time - entry_time).days) if hasattr(exit_time - entry_time, "days") else 1),
            "position_type": "long" if position == 1 else "short",
            "exit_reason": exit_reason,
        }
        return new_capital, trade

    def _mark_to_market(self, capital: float, position: int, entry_price: Optional[float], current_price: float) -> float:
        if position == 0 or entry_price is None:
            return capital
        unrealized_pct = (current_price - entry_price) / entry_price if position == 1 else (entry_price - current_price) / entry_price
        return capital * (1 + unrealized_pct * self.strategy_params["position_size"])

    def _backtest_strategy(self, historical_data: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, Any]:
        if historical_data is None or historical_data.empty:
            raise ValueError("历史数据为空，无法回测")
        if signals is None or signals.empty:
            raise ValueError("信号数据为空，无法回测")

        self.logger.info("开始回测策略")
        initial_capital = float(self.config.get("backtest", {}).get("initial_capital", 100000.0))
        capital = initial_capital
        position = 0
        entry_price = None
        entry_time = None
        trades: List[Dict[str, Any]] = []
        equity_values: List[float] = []

        aligned_index = historical_data.index.intersection(signals.index)
        historical = historical_data.loc[aligned_index]
        signal_frame = signals.loc[aligned_index]

        for timestamp in aligned_index:
            row = historical.loc[timestamp]
            target_position = int(signal_frame.loc[timestamp, "signal"])
            current_price = float(row["close"])

            if position != 0 and entry_price is not None:
                current_profit_pct = (current_price - entry_price) / entry_price if position == 1 else (entry_price - current_price) / entry_price
                if (
                    current_profit_pct <= -self.strategy_params["stop_loss_pct"]
                    or current_profit_pct >= self.strategy_params["take_profit_pct"]
                ):
                    capital, trade = self._close_position(
                        capital,
                        position,
                        entry_price,
                        current_price,
                        entry_time,
                        timestamp,
                        "risk_rule",
                    )
                    trades.append(trade)
                    position = 0
                    entry_price = None
                    entry_time = None

            if target_position != position:
                if position != 0 and entry_price is not None:
                    capital, trade = self._close_position(
                        capital,
                        position,
                        entry_price,
                        current_price,
                        entry_time,
                        timestamp,
                        "signal_change",
                    )
                    trades.append(trade)
                    position = 0
                    entry_price = None
                    entry_time = None

                if target_position != 0:
                    position = target_position
                    entry_price = current_price
                    entry_time = timestamp

            equity_values.append(self._mark_to_market(capital, position, entry_price, current_price))

        if position != 0 and entry_price is not None:
            last_time = historical.index[-1]
            last_price = float(historical.iloc[-1]["close"])
            capital, trade = self._close_position(
                capital,
                position,
                entry_price,
                last_price,
                entry_time,
                last_time,
                "end_of_backtest",
            )
            trades.append(trade)
            equity_values[-1] = capital

        equity_curve = pd.Series(equity_values, index=historical.index, name="equity")
        trade_statistics = self._calculate_trade_statistics(trades)
        return {
            "trades": trades,
            "equity_curve": equity_curve,
            "signals": signal_frame,
            "trade_statistics": trade_statistics,
        }

    def _calculate_equity_curve(self, trades, initial_capital, all_dates=None):
        if all_dates is None:
            if not trades:
                return pd.Series([initial_capital])
            start_date = trades[0]["entry_time"]
            end_date = trades[-1]["exit_time"]
            all_dates = pd.date_range(start=start_date, end=end_date, freq="D")

        equity_curve = pd.Series(initial_capital, index=all_dates, dtype=float)
        capital = initial_capital
        last_time = all_dates[0]
        for trade in trades:
            equity_curve.loc[last_time:trade["exit_time"]] = capital
            capital *= 1 + trade["profit_pct"] * self.strategy_params["position_size"]
            last_time = trade["exit_time"]
        equity_curve.loc[last_time:] = capital
        return equity_curve

    def _calculate_trade_statistics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_profit": 0.0,
                "long_trades": 0,
                "short_trades": 0,
                "long_win_rate": 0.0,
                "short_win_rate": 0.0,
            }

        profits = [trade["profit_pct"] for trade in trades]
        long_trades = [trade for trade in trades if trade["position_type"] == "long"]
        short_trades = [trade for trade in trades if trade["position_type"] == "short"]

        def _win_rate(items: List[Dict[str, Any]]) -> float:
            if not items:
                return 0.0
            return sum(1 for item in items if item["profit_pct"] > 0) / len(items)

        return {
            "total_trades": len(trades),
            "win_rate": _win_rate(trades),
            "avg_profit": float(np.mean(profits)),
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "long_win_rate": _win_rate(long_trades),
            "short_win_rate": _win_rate(short_trades),
        }

    def _calculate_sharpe_ratio(self, equity_curve: pd.Series) -> float:
        returns = equity_curve.pct_change().dropna()
        if returns.empty or returns.std() == 0:
            return 0.0
        return float((returns.mean() / returns.std()) * np.sqrt(252))

    def _calculate_sortino_ratio(self, equity_curve: pd.Series) -> float:
        returns = equity_curve.pct_change().dropna()
        downside = returns[returns < 0]
        if returns.empty or downside.empty or downside.std() == 0:
            return 0.0
        return float((returns.mean() / downside.std()) * np.sqrt(252))

    def _calculate_profit_factor(self, trades: List[Dict[str, Any]]) -> float:
        gross_profit = sum(trade["profit_pct"] for trade in trades if trade["profit_pct"] > 0)
        gross_loss = abs(sum(trade["profit_pct"] for trade in trades if trade["profit_pct"] < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return float(gross_profit / gross_loss)

    def _calculate_performance_metrics(self, backtest_results):
        trades = backtest_results["trades"]
        equity_curve = backtest_results["equity_curve"]
        stats = backtest_results["trade_statistics"]
        if equity_curve.empty:
            return {
                "total_return": 0.0,
                "annual_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
            }

        total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
        num_days = max(1, (equity_curve.index[-1] - equity_curve.index[0]).days)
        annual_return = float((1 + total_return) ** (365 / num_days) - 1) if num_days > 0 else total_return
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": self._calculate_sharpe_ratio(equity_curve),
            "sortino_ratio": self._calculate_sortino_ratio(equity_curve),
            "win_rate": float(stats["win_rate"]),
            "profit_factor": self._calculate_profit_factor(trades),
            "total_trades": int(stats["total_trades"]),
        }

    def _calculate_strategy_risk_metrics(self, backtest_results):
        equity_curve = backtest_results["equity_curve"]
        return self.risk_calculator.calculate_portfolio_risk(equity_curve)

    def _calculate_beta(self, equity_curve, benchmark_data):
        benchmark_series = benchmark_data[["close"]].copy()
        benchmark_series.columns = ["close"]
        strategy_frame = equity_curve.to_frame(name="close")
        return self.risk_calculator.calculate_beta(strategy_frame, benchmark_series)

    def _calculate_correlation(self, equity_curve, benchmark_data):
        benchmark_series = benchmark_data[["close"]].copy()
        benchmark_series.columns = ["close"]
        strategy_frame = equity_curve.to_frame(name="close")
        return self.risk_calculator.calculate_correlation(strategy_frame, benchmark_series)

    def _generate_recommendations(self, backtest_results):
        performance = self._calculate_performance_metrics(backtest_results)
        risk = self._calculate_strategy_risk_metrics(backtest_results)
        recommendations = []
        if performance["sharpe_ratio"] < 1.0:
            recommendations.append("建议继续优化信号过滤条件，优先减少震荡区间的无效交易。")
        if risk["max_drawdown"] < -0.2:
            recommendations.append("建议收紧止损比例或降低仓位上限，先把最大回撤压到 20% 以内。")
        if performance["total_trades"] < 5:
            recommendations.append("交易次数偏少，建议扩大样本区间后再评估策略稳定性。")
        if not recommendations:
            recommendations.append("当前基线策略已形成可重复研究闭环，下一步可接入 paper trading。")
        return recommendations

    def _plot_backtest_results(self, backtest_results):
        os.makedirs(self.results_dir, exist_ok=True)
        equity_curve = backtest_results["equity_curve"]
        drawdown = equity_curve / equity_curve.cummax() - 1

        if plt is None:
            self.logger.warning("未安装 matplotlib，跳过图表输出")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        ax1.plot(equity_curve.index, equity_curve.values, color="#005f73", label="Equity Curve")
        ax1.set_title("Strategy Equity Curve")
        ax1.set_ylabel("Equity")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.fill_between(drawdown.index, drawdown.values, 0, color="#bb3e03", alpha=0.35, label="Drawdown")
        ax2.set_title("Drawdown")
        ax2.set_ylabel("Drawdown")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(self.results_dir, "backtest_results.png"))
        plt.close(fig)

    def _save_backtest_results(self, backtest_results: Dict, performance: Dict, risk_metrics: Dict, recommendations: List[str]):
        os.makedirs(self.results_dir, exist_ok=True)

        pd.DataFrame(backtest_results["trades"]).to_csv(
            os.path.join(self.results_dir, "trades.csv"), index=False, encoding="utf-8"
        )
        backtest_results["equity_curve"].to_csv(
            os.path.join(self.results_dir, "equity_curve.csv"), encoding="utf-8"
        )
        if "signals" in backtest_results:
            backtest_results["signals"].to_csv(
                os.path.join(self.results_dir, "signals.csv"), encoding="utf-8"
            )

        metrics = {
            "performance": performance,
            "risk_metrics": risk_metrics,
            "recommendations": recommendations,
        }
        with open(os.path.join(self.results_dir, "metrics.json"), "w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2, default=str)

        self._plot_backtest_results(backtest_results)
        self.logger.info(f"回测结果已保存到目录: {self.results_dir}")
