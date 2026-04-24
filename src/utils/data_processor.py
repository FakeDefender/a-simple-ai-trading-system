import logging
from typing import Dict, Union

import numpy as np
import pandas as pd


class DataProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def get_historical_data(self, symbol: str, timeframe: str = "1d") -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("未安装 yfinance，无法在线拉取历史数据") from exc

        data = yf.download(symbol, period="1y", interval=timeframe, progress=False)
        if data.empty:
            return pd.DataFrame()
        data.columns = [str(col).lower() for col in data.columns]
        return self.calculate_technical_indicators(data)

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            raise ValueError("输入数据为空，无法计算技术指标")

        df = data.copy()
        df = df.sort_index()
        df.columns = [str(col).lower() for col in df.columns]

        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必要列: {missing_columns}")

        df[required_columns] = df[required_columns].astype(float)
        df = self._calculate_moving_averages(df)
        df = self._calculate_rsi(df)
        df = self._calculate_macd(df)
        df = self._calculate_bollinger_bands(df)
        df = self._calculate_volume_indicators(df)
        df = self._calculate_atr(df)
        df["returns"] = df["close"].pct_change()

        missing_stats = self._analyze_missing_values(df)
        self.logger.info(f"技术指标计算完成，缺失值统计: {missing_stats}")
        return df

    def _calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        data["ma5"] = data["close"].rolling(window=5).mean()
        data["ma10"] = data["close"].rolling(window=10).mean()
        data["ma20"] = data["close"].rolling(window=20).mean()
        data["ma50"] = data["close"].rolling(window=50).mean()
        data["ma5_slope"] = data["ma5"].pct_change(periods=3)
        data["ma10_slope"] = data["ma10"].pct_change(periods=3)
        data["ma20_slope"] = data["ma20"].pct_change(periods=5)
        data["ma5_10_cross"] = np.where(data["ma5"] >= data["ma10"], 1, -1)
        data["ma10_20_cross"] = np.where(data["ma10"] >= data["ma20"], 1, -1)
        return data

    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        delta = data["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        data["rsi"] = rsi.fillna(50.0)
        return data

    def _calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        exp12 = data["close"].ewm(span=12, adjust=False).mean()
        exp26 = data["close"].ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        data["macd"] = macd
        data["signal"] = signal
        data["macd_hist"] = macd - signal
        return data

    def _calculate_bollinger_bands(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        middle = data["close"].rolling(window=window).mean()
        std = data["close"].rolling(window=window).std()
        data["bb_middle"] = middle
        data["bb_upper"] = middle + 2 * std
        data["bb_lower"] = middle - 2 * std
        return data

    def _calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        data["volume_sma"] = data["volume"].rolling(window=20).mean()
        data["volume_ratio"] = data["volume"] / data["volume_sma"].replace(0, np.nan)
        data["volume_ratio"] = data["volume_ratio"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
        return data

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        high_low = data["high"] - data["low"]
        high_close = (data["high"] - data["close"].shift(1)).abs()
        low_close = (data["low"] - data["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data["atr"] = true_range.rolling(window=period).mean().bfill()
        return data

    def calculate_market_indicators(self, data: Union[Dict, pd.DataFrame]) -> Dict:
        if isinstance(data, pd.DataFrame):
            if data.empty:
                return {
                    "trend_strength": 0.0,
                    "market_strength": 0.5,
                    "volatility": 0.0,
                    "volume_trend": 0.0,
                    "moving_averages": {},
                }
            latest = data.iloc[-1]
            return {
                "trend_strength": self._calculate_trend_strength_from_df(data),
                "market_strength": self._calculate_market_strength(data),
                "volatility": self._calculate_volatility_from_df(data),
                "volume_trend": self._calculate_volume_trend(data),
                "moving_averages": {
                    "ma5": float(latest.get("ma5", 0.0)),
                    "ma10": float(latest.get("ma10", 0.0)),
                    "ma20": float(latest.get("ma20", 0.0)),
                    "ma50": float(latest.get("ma50", 0.0)),
                },
            }

        if isinstance(data, dict):
            return {
                "trend_strength": self._calculate_trend_strength(data),
                "market_strength": self._calculate_market_strength(data),
                "volatility": self._calculate_volatility(data),
                "volume_trend": float(data.get("volume_trend", 0.0)),
                "moving_averages": data.get("moving_averages", {}),
            }

        raise ValueError(f"不支持的数据类型: {type(data)}")

    def calculate_risk_metrics(self, data: Union[pd.DataFrame, Dict]) -> Dict:
        if isinstance(data, pd.DataFrame):
            returns = data["close"].pct_change().dropna()
            if returns.empty:
                return {
                    "volatility": 0.0,
                    "max_drawdown": 0.0,
                    "sharpe_ratio": 0.0,
                    "var_95": 0.0,
                    "expected_shortfall": 0.0,
                }
            volatility = float(returns.std() * np.sqrt(252))
            cumulative = (1 + returns).cumprod()
            drawdown = cumulative / cumulative.cummax() - 1
            sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if returns.std() > 0 else 0.0
            var_95 = float(returns.quantile(0.05))
            expected_shortfall = float(returns[returns <= var_95].mean()) if (returns <= var_95).any() else 0.0
            return {
                "volatility": volatility,
                "max_drawdown": float(drawdown.min()),
                "sharpe_ratio": sharpe,
                "var_95": var_95,
                "expected_shortfall": expected_shortfall,
            }

        indicators = self.calculate_market_indicators(data)
        return {
            "volatility": float(indicators.get("volatility", 0.0)),
            "trend_risk": abs(float(indicators.get("trend_strength", 0.0))),
            "market_risk": abs(float(indicators.get("market_strength", 0.5)) - 0.5) * 2,
        }

    def _calculate_trend_strength(self, data: Dict) -> float:
        ma_data = data.get("moving_averages", {})
        ma5 = float(ma_data.get("ma5", 0.0))
        ma20 = float(ma_data.get("ma20", 0.0))
        return (ma5 - ma20) / ma20 if ma20 else 0.0

    def _calculate_market_strength(self, data: Union[Dict, pd.DataFrame]) -> float:
        if isinstance(data, pd.DataFrame):
            latest = data.iloc[-1]
            rsi = float(latest.get("rsi", 50.0))
            macd = float(latest.get("macd", 0.0))
            signal = float(latest.get("signal", 0.0))
            volume_ratio = float(latest.get("volume_ratio", 1.0))
        else:
            rsi = float(data.get("rsi", 50.0))
            macd = float(data.get("macd", 0.0))
            signal = float(data.get("signal", 0.0))
            volume_ratio = float(data.get("volume_ratio", 1.0))

        rsi_component = np.clip((rsi - 30) / 40, 0, 1)
        macd_gap = macd - signal
        macd_component = np.clip(0.5 + np.tanh(macd_gap) * 0.5, 0, 1)
        volume_component = np.clip(volume_ratio / 2.0, 0, 1)
        return float(0.4 * rsi_component + 0.4 * macd_component + 0.2 * volume_component)

    def _calculate_volatility(self, data: Dict) -> float:
        ma_data = data.get("moving_averages", {})
        if not ma_data:
            return 0.0
        values = [float(value) for value in ma_data.values() if value]
        if not values:
            return 0.0
        return float(np.std(values) / np.mean(values))

    def _calculate_volatility_from_df(self, data: pd.DataFrame) -> float:
        returns = data["close"].pct_change().dropna()
        if returns.empty:
            return 0.0
        return float(returns.std() * np.sqrt(252))

    def _calculate_trend_strength_from_df(self, data: pd.DataFrame) -> float:
        latest = data.iloc[-1]
        ma20 = float(latest.get("ma20", 0.0))
        if not ma20:
            return 0.0
        ma5 = float(latest.get("ma5", 0.0))
        return float((ma5 - ma20) / ma20)

    def _calculate_volume_trend(self, data: pd.DataFrame) -> float:
        volume_change = data["volume"].pct_change().dropna()
        if volume_change.empty:
            return 0.0
        return float(volume_change.tail(20).mean())

    def _analyze_missing_values(self, data: pd.DataFrame) -> Dict:
        stats = {}
        for column in [
            "ma5",
            "ma10",
            "ma20",
            "ma50",
            "rsi",
            "macd",
            "signal",
            "bb_middle",
            "bb_upper",
            "bb_lower",
            "volume_sma",
            "volume_ratio",
            "atr",
        ]:
            if column not in data.columns:
                continue
            missing_count = int(data[column].isna().sum())
            stats[column] = {
                "missing_count": missing_count,
                "missing_percentage": round(missing_count / len(data) * 100, 2),
                "first_valid_index": data[column].first_valid_index(),
            }
        return stats

    def get_complete_data(self, data: pd.DataFrame, min_periods: int = 50) -> pd.DataFrame:
        if data is None or data.empty:
            raise ValueError("输入数据为空")

        complete = data.iloc[min_periods:].dropna().copy() if len(data) > min_periods else data.dropna().copy()
        if complete.empty:
            raise ValueError("没有可用于回测的完整数据，请增加历史样本或检查指标计算")
        return complete

    def get_data_summary(self, data: pd.DataFrame) -> Dict:
        if data is None or data.empty:
            return {
                "total_rows": 0,
                "complete_rows": 0,
                "missing_percentage": 0.0,
                "date_range": {"start": None, "end": None},
                "columns": [],
                "missing_by_column": {},
            }

        return {
            "total_rows": int(len(data)),
            "complete_rows": int(len(data.dropna())),
            "missing_percentage": round((len(data) - len(data.dropna())) / len(data) * 100, 2),
            "date_range": {
                "start": str(data.index.min()),
                "end": str(data.index.max()),
            },
            "columns": list(data.columns),
            "missing_by_column": {
                column: {
                    "missing_count": int(data[column].isna().sum()),
                    "missing_percentage": round(data[column].isna().sum() / len(data) * 100, 2),
                }
                for column in data.columns
            },
        }
