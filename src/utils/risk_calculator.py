from typing import Dict

import numpy as np
import pandas as pd


class RiskCalculator:
    def __init__(self, config=None):
        self.config = config or {}

    def _returns(self, data: pd.DataFrame) -> pd.Series:
        return data["close"].pct_change().dropna()

    def calculate_var(self, data: pd.DataFrame, confidence_level: float = 0.95) -> float:
        returns = self._returns(data)
        if returns.empty:
            return 0.0
        return float(np.percentile(returns, (1 - confidence_level) * 100))

    def calculate_expected_shortfall(self, data: pd.DataFrame, confidence_level: float = 0.95) -> float:
        returns = self._returns(data)
        if returns.empty:
            return 0.0
        var = self.calculate_var(data, confidence_level)
        tail = returns[returns <= var]
        if tail.empty:
            return 0.0
        return float(tail.mean())

    def calculate_beta(self, data: pd.DataFrame, market_data: pd.DataFrame) -> float:
        returns = self._returns(data)
        market_returns = self._returns(market_data)
        common_index = returns.index.intersection(market_returns.index)
        if len(common_index) < 2:
            return 0.0
        returns = returns.loc[common_index]
        market_returns = market_returns.loc[common_index]
        market_variance = market_returns.var()
        if market_variance == 0:
            return 0.0
        covariance = returns.cov(market_returns)
        return float(covariance / market_variance)

    def calculate_correlation(self, data: pd.DataFrame, market_data: pd.DataFrame) -> float:
        returns = self._returns(data)
        market_returns = self._returns(market_data)
        common_index = returns.index.intersection(market_returns.index)
        if len(common_index) < 2:
            return 0.0
        return float(returns.loc[common_index].corr(market_returns.loc[common_index]))

    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        if equity_curve.empty:
            return 0.0
        running_max = equity_curve.cummax()
        drawdown = equity_curve / running_max - 1
        return float(drawdown.min())

    def calculate_portfolio_risk(self, equity_curve: pd.Series) -> Dict[str, float]:
        if equity_curve.empty or len(equity_curve) < 2:
            return {
                "max_drawdown": 0.0,
                "volatility": 0.0,
                "var_95": 0.0,
                "expected_shortfall": 0.0,
            }

        returns = equity_curve.pct_change().dropna()
        if returns.empty:
            return {
                "max_drawdown": 0.0,
                "volatility": 0.0,
                "var_95": 0.0,
                "expected_shortfall": 0.0,
            }

        var_95 = float(np.percentile(returns, 5))
        tail = returns[returns <= var_95]
        return {
            "max_drawdown": self.calculate_max_drawdown(equity_curve),
            "volatility": float(returns.std() * np.sqrt(252)),
            "var_95": var_95,
            "expected_shortfall": float(tail.mean()) if not tail.empty else 0.0,
        }
