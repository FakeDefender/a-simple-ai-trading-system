import logging
import os
from io import StringIO
from typing import Dict, Optional
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd

from .data_processor import DataProcessor
from .data_storage import DataStorage


logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, config: Dict):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.data_storage = DataStorage(self.config)
        self.data_processor = DataProcessor()

    def load_data(
        self,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
        force_update: Optional[bool] = None,
    ) -> Optional[pd.DataFrame]:
        data_config = self.config.get("data", {})
        symbol = symbol or data_config.get("symbol", "aapl.us")
        interval = interval or data_config.get("interval", "d")
        force_update = data_config.get("force_update", False) if force_update is None else force_update
        source = data_config.get("source", "api")

        self.logger.info(
            f"开始加载数据 - source={source}, symbol={symbol}, interval={interval}, force_update={force_update}"
        )

        if source == "csv":
            data = self._load_from_csv(symbol=symbol)
            return self.data_processor.calculate_technical_indicators(data)

        if not force_update:
            cached = self.data_storage.load_processed_data(symbol, interval, "technical_indicators")
            if cached is not None and not cached.empty:
                self.logger.info(f"命中本地缓存: {symbol} {interval}")
                return cached

        raw_data = self._fetch_market_data(symbol, interval)
        if raw_data is None or raw_data.empty:
            self.logger.error("获取市场数据失败")
            return None

        self.data_storage.save_raw_data(raw_data, symbol, interval)
        processed = self.data_processor.calculate_technical_indicators(raw_data)
        self.data_storage.save_processed_data(processed, symbol, interval, "technical_indicators")
        return processed

    def _fetch_market_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        data_config = self.config.get("data", {})
        source = data_config.get("source", "api")
        if source != "api":
            raise ValueError(f"暂不支持的数据源类型: {source}")

        normalized_interval = str(interval or "d").lower()
        timeframe = self._resolve_remote_timeframe(normalized_interval)
        if normalized_interval in {"d", "w", "m"} and self._stooq_api_key():
            stooq_data = self._fetch_from_stooq(symbol, normalized_interval)
            if stooq_data is not None and not stooq_data.empty:
                return stooq_data
            self.logger.warning("Stooq 数据不可用，改用 Yahoo Chart - symbol=%s, interval=%s", symbol, normalized_interval)

        yahoo_data = self._fetch_from_yahoo_chart(symbol, timeframe)
        if yahoo_data is not None and not yahoo_data.empty:
            return yahoo_data

        self.logger.warning("Yahoo Chart 数据不可用，改用 yfinance - symbol=%s, interval=%s", symbol, normalized_interval)
        return self._fetch_from_yfinance(symbol, timeframe)

    def _log_fetch_issue(self, optional: bool, message: str, *args):
        log = self.logger.info if optional else self.logger.error
        log(message, *args)

    def _resolve_remote_timeframe(self, interval: str) -> str:
        normalized_interval = str(interval or "").lower()
        mapping = {
            "d": "1d",
            "w": "1wk",
            "m": "1mo",
        }
        if normalized_interval in mapping:
            return mapping[normalized_interval]
        return str(self.config.get("data", {}).get("timeframe", "1d"))

    def _stooq_api_key(self) -> str:
        return str(self.config.get("data", {}).get("stooq_api_key", "")).strip()

    def _format_stooq_date(self, value: Optional[str]) -> str:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return ""
        return parsed.strftime("%Y%m%d")

    def _to_unix_timestamp(self, value: Optional[str], default_offset_days: int) -> int:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            timestamp = pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=default_offset_days)
        else:
            timestamp = pd.Timestamp(parsed)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return int(timestamp.timestamp())

    def _fetch_from_stooq(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        api_key = self._stooq_api_key()
        if not api_key:
            self.logger.info("未配置 Stooq apikey，跳过 Stooq 数据源 - symbol=%s", symbol)
            return None

        data_config = self.config.get("data", {})
        start_token = self._format_stooq_date(start_date or data_config.get("start_date"))
        end_token = self._format_stooq_date(end_date or data_config.get("end_date"))
        url = (
            "https://stooq.com/q/d/l/"
            f"?s={symbol}&d1={start_token}&d2={end_token}&i={interval}&apikey={api_key}"
        )

        try:
            with urlopen(url, timeout=20) as response:
                payload = response.read().decode("utf-8", errors="replace").strip()
        except Exception as exc:
            self.logger.error(f"从 Stooq 获取数据失败: {exc}")
            return None

        if not payload:
            self.logger.error("Stooq 返回空响应")
            return None

        first_line = payload.splitlines()[0].strip().lstrip("\ufeff")
        if first_line.lower() != "date,open,high,low,close,volume":
            self.logger.warning("Stooq 返回了非标准 CSV 内容，已跳过 - symbol=%s, first_line=%s", symbol, first_line[:120])
            return None

        try:
            data = pd.read_csv(StringIO(payload))
        except Exception as exc:
            self.logger.error(f"解析 Stooq 数据失败: {exc}")
            return None

        if data.empty or len(data.columns) < 6:
            self.logger.error("Stooq 返回空数据或字段不完整")
            return None

        data.columns = ["date", "open", "high", "low", "close", "volume"]
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date"])
        data[["open", "high", "low", "close", "volume"]] = data[
            ["open", "high", "low", "close", "volume"]
        ].astype(float)
        data = data.set_index("date").sort_index()
        return data

    def _normalize_symbol_for_yfinance(self, symbol: str) -> str:
        raw_symbol = str(symbol or "").strip()
        normalized = raw_symbol.lower()
        if normalized.endswith(".us"):
            return raw_symbol[:-3].upper()
        if normalized.endswith(".hk"):
            return f"{raw_symbol[:-3].upper()}.HK"
        if normalized.endswith(".sh"):
            return f"{raw_symbol[:-3]}.SS"
        if normalized.endswith(".sz"):
            return f"{raw_symbol[:-3]}.SZ"
        if normalized.endswith(".bj"):
            return f"{raw_symbol[:-3]}.BJ"
        if "/" in raw_symbol:
            base, quote = raw_symbol.split("/", 1)
            quote = "USD" if quote.lower() in {"usd", "usdt", "usdc"} else quote.upper()
            return f"{base.upper()}-{quote}"
        if raw_symbol.startswith("^"):
            return raw_symbol.upper()
        return raw_symbol

    def _fetch_from_yahoo_chart(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        optional: bool = False,
    ) -> Optional[pd.DataFrame]:
        try:
            import requests
        except ImportError:
            self._log_fetch_issue(optional, "未安装 requests，无法通过 Yahoo Chart 拉取数据")
            return None

        data_config = self.config.get("data", {})
        ticker = self._normalize_symbol_for_yfinance(symbol)
        period1 = self._to_unix_timestamp(start_date or data_config.get("start_date"), default_offset_days=-365)
        period2 = self._to_unix_timestamp(end_date or data_config.get("end_date"), default_offset_days=1)
        if period2 <= period1:
            period2 = period1 + 86400

        encoded_ticker = quote(ticker, safe="")
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_ticker}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Origin": "https://finance.yahoo.com",
            "Referer": "https://finance.yahoo.com/",
        }
        params = {
            "period1": period1,
            "period2": period2,
            "interval": timeframe,
            "includeAdjustedClose": "true",
            "events": "div,splits",
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            self._log_fetch_issue(optional, "Yahoo Chart 请求失败 - symbol=%s, ticker=%s, error=%s", symbol, ticker, exc)
            return None

        chart = payload.get("chart", {}) if isinstance(payload, dict) else {}
        result = chart.get("result") or []
        if not result:
            error_payload = chart.get("error") or {}
            error_text = error_payload.get("description") or error_payload.get("code") or "empty result"
            self._log_fetch_issue(optional, "Yahoo Chart 未返回有效数据 - symbol=%s, ticker=%s, error=%s", symbol, ticker, error_text)
            return None

        item = result[0]
        timestamps = item.get("timestamp") or []
        indicators = item.get("indicators") or {}
        quote_list = indicators.get("quote") or []
        if not timestamps or not quote_list:
            self._log_fetch_issue(optional, "Yahoo Chart 返回字段不完整 - symbol=%s, ticker=%s", symbol, ticker)
            return None

        quote_payload = quote_list[0]
        index = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None)
        frame = pd.DataFrame(
            {
                "open": quote_payload.get("open", []),
                "high": quote_payload.get("high", []),
                "low": quote_payload.get("low", []),
                "close": quote_payload.get("close", []),
                "volume": quote_payload.get("volume", []),
            },
            index=index,
        )
        adjclose_list = (indicators.get("adjclose") or [{}])[0].get("adjclose") or []
        if len(adjclose_list) == len(frame):
            adjusted_close = pd.Series(adjclose_list, index=frame.index)
            frame["close"] = pd.to_numeric(frame["close"], errors="coerce").fillna(adjusted_close)

        for column in ["open", "high", "low", "close", "volume"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["open", "high", "low", "close"])
        frame["volume"] = frame["volume"].fillna(0.0)
        frame.index.name = "date"
        if frame.empty:
            self._log_fetch_issue(optional, "Yahoo Chart 解析后无有效行 - symbol=%s, ticker=%s", symbol, ticker)
            return None
        return frame.sort_index()

    def _fetch_from_yfinance(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        optional: bool = False,
    ) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
        except ImportError:
            self._log_fetch_issue(optional, "未安装 yfinance，无法拉取该类标的")
            return None

        data_config = self.config.get("data", {})
        start_date = start_date or data_config.get("start_date")
        end_date = end_date or data_config.get("end_date")
        ticker = self._normalize_symbol_for_yfinance(symbol)
        self.logger.info("通过 yfinance 拉取数据 - symbol=%s, ticker=%s, timeframe=%s", symbol, ticker, timeframe)
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=timeframe,
            progress=False,
            auto_adjust=False,
        )
        if data.empty:
            self._log_fetch_issue(optional, "yfinance 未返回有效数据 - symbol=%s, ticker=%s", symbol, ticker)
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [str(column[0]).lower() for column in data.columns]
        else:
            data.columns = [str(col).lower() for col in data.columns]
        if "adj close" in data.columns and "close" not in data.columns:
            data = data.rename(columns={"adj close": "close"})
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self._log_fetch_issue(optional, "yfinance 返回字段不完整: %s", missing_columns)
            return None
        return data[required_columns]

    def _resolve_csv_path(self, symbol: Optional[str] = None) -> str:
        data_config = self.config.get("data", {})
        path_mapping = data_config.get("paths", {}) or {}
        if symbol and symbol in path_mapping:
            return path_mapping[symbol]
        return data_config.get("path", "")

    def _load_from_csv(self, symbol: Optional[str] = None) -> pd.DataFrame:
        file_path = self._resolve_csv_path(symbol)
        if not file_path:
            raise ValueError("source=csv 时必须配置 data.path 或 data.paths[symbol]")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        data = pd.read_csv(file_path)
        data.columns = [str(col).lower() for col in data.columns]
        date_col = "datetime" if "datetime" in data.columns else "date"
        if date_col not in data.columns:
            raise ValueError("CSV 中缺少 date 或 datetime 列")

        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
        data = data.dropna(subset=[date_col]).set_index(date_col).sort_index()

        start_date = self.config.get("data", {}).get("start_date")
        end_date = self.config.get("data", {}).get("end_date")
        if start_date:
            data = data[data.index >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data.index <= pd.to_datetime(end_date)]

        return self.preprocess_data(data)

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df = self._handle_missing_values(df)
        df = self._handle_outliers(df)
        return df

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.ffill().bfill()

    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(data) < 20:
            return data

        df = data.copy()
        returns = df["close"].pct_change()
        rolling_std = returns.rolling(window=20).std()
        rolling_median = df["close"].rolling(window=20).median()
        outlier_mask = returns.abs() > rolling_std * 6
        df.loc[outlier_mask, "close"] = rolling_median.loc[outlier_mask]
        return df

    def save_data(self, data: pd.DataFrame, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data.to_csv(file_path, encoding="utf-8")
        self.logger.info(f"数据已保存到: {file_path}")

    def _build_target_benchmark(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        reference_symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        data_config = self.config.get("data", {})
        symbol = reference_symbol or data_config.get("symbol", "target")
        benchmark = reference_data.copy() if reference_data is not None else pd.DataFrame()

        if benchmark.empty:
            interval = data_config.get("interval", "d")
            cached = self.data_storage.load_latest_data(symbol, interval, "raw")
            if cached is None or cached.empty:
                cached = self.data_storage.load_processed_data(symbol, interval, "technical_indicators")
            benchmark = cached.copy() if cached is not None else pd.DataFrame()

        if benchmark.empty or "close" not in benchmark.columns:
            self.logger.warning("无法构造标的买入持有基准 - symbol=%s", symbol)
            return pd.DataFrame()

        benchmark = benchmark.sort_index().copy()
        for column in ["open", "high", "low"]:
            if column not in benchmark.columns:
                benchmark[column] = benchmark["close"]
        if "volume" not in benchmark.columns:
            benchmark["volume"] = 0.0

        benchmark = benchmark[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        self.logger.info("使用标的买入持有作为基准: %s", symbol)
        return benchmark

    def get_benchmark_data(
        self,
        start_date: str = None,
        end_date: str = None,
        reference_data: Optional[pd.DataFrame] = None,
        reference_symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        backtest_config = self.config.get("backtest", {})
        if not backtest_config.get("benchmark_enabled", True):
            self.logger.info("基准对比已关闭")
            return pd.DataFrame()

        benchmark_source = str(backtest_config.get("benchmark_source", "market")).lower()
        if benchmark_source in {"target", "target_buy_hold", "buy_hold", "self"}:
            return self._build_target_benchmark(reference_data, reference_symbol)

        benchmark_symbol = backtest_config.get("benchmark_symbol", "^ndx")
        if not benchmark_symbol:
            self.logger.info("未配置 benchmark_symbol，跳过基准对比")
            return self._build_target_benchmark(reference_data, reference_symbol)

        cache_interval = "d_benchmark"
        force_update = bool(self.config.get("data", {}).get("force_update", False))
        if not force_update:
            cached = self.data_storage.load_latest_data(benchmark_symbol, cache_interval, "raw")
            if cached is not None and not cached.empty:
                self.logger.info("命中本地基准缓存: %s", benchmark_symbol)
                return cached

        def persist(benchmark: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if benchmark is not None and not benchmark.empty:
                self.data_storage.save_raw_data(benchmark, benchmark_symbol, cache_interval)
                return benchmark
            return None

        if self._stooq_api_key():
            benchmark = self._fetch_from_stooq(benchmark_symbol, "d", start_date=start_date, end_date=end_date)
            cached_benchmark = persist(benchmark)
            if cached_benchmark is not None:
                return cached_benchmark

        timeframe = self._resolve_remote_timeframe("d")
        benchmark = self._fetch_from_yahoo_chart(
            benchmark_symbol,
            timeframe,
            start_date=start_date,
            end_date=end_date,
            optional=True,
        )
        cached_benchmark = persist(benchmark)
        if cached_benchmark is not None:
            return cached_benchmark

        if backtest_config.get("benchmark_yfinance_fallback", False):
            fallback = self._fetch_from_yfinance(
                benchmark_symbol,
                timeframe,
                start_date=start_date,
                end_date=end_date,
                optional=True,
            )
            cached_benchmark = persist(fallback)
            if cached_benchmark is not None:
                return cached_benchmark

        if backtest_config.get("benchmark_fallback_to_target", True):
            self.logger.info("外部基准不可用，改用标的买入持有基准 - benchmark=%s", benchmark_symbol)
            return self._build_target_benchmark(reference_data, reference_symbol)

        self.logger.warning("基准数据不可用，已跳过基准对比 - symbol=%s", benchmark_symbol)
        return pd.DataFrame()
