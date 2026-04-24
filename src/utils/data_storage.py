import json
import logging
import os
from datetime import datetime
from typing import Optional

import pandas as pd


logger = logging.getLogger(__name__)


class DataStorage:
    """负责原始数据和处理后数据的本地缓存。"""

    def __init__(self, config):
        self.config = config or {}
        storage_config = self.config.get("data", {}).get("storage", {})
        base_dir = storage_config.get("base_dir", os.path.join("data", "market_data"))
        self.data_dir = base_dir
        self.raw_data_dir = os.path.join(base_dir, storage_config.get("raw_data_dir", "raw"))
        self.processed_data_dir = os.path.join(base_dir, storage_config.get("processed_data_dir", "processed"))
        self._create_directories()

    def _create_directories(self):
        for directory in [self.data_dir, self.raw_data_dir, self.processed_data_dir]:
            os.makedirs(directory, exist_ok=True)

    def _safe_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "_").replace("\\", "_").replace(":", "_")

    def save_raw_data(self, data: pd.DataFrame, symbol: str, interval: str):
        filename = f"{self._safe_symbol(symbol)}_{interval}_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = os.path.join(self.raw_data_dir, filename)
        data.to_csv(filepath, index=True, encoding="utf-8")
        self._update_metadata(symbol, interval, filepath)
        logger.info(f"原始数据已保存到: {filepath}")

    def save_processed_data(self, data: pd.DataFrame, symbol: str, interval: str, process_type: str):
        filename = (
            f"{self._safe_symbol(symbol)}_{interval}_{process_type}_{datetime.now().strftime('%Y%m%d')}.csv"
        )
        filepath = os.path.join(self.processed_data_dir, filename)
        data.to_csv(filepath, index=True, encoding="utf-8")
        logger.info(f"处理后数据已保存到: {filepath}")

    def load_latest_data(self, symbol: str, interval: str, data_type: str = "raw") -> Optional[pd.DataFrame]:
        directory = self.raw_data_dir if data_type == "raw" else self.processed_data_dir
        prefix = f"{self._safe_symbol(symbol)}_{interval}_"
        if not os.path.exists(directory):
            return None

        files = [name for name in os.listdir(directory) if name.startswith(prefix) and name.endswith(".csv")]
        if not files:
            return None

        latest_file = max(files)
        filepath = os.path.join(directory, latest_file)
        return pd.read_csv(filepath, index_col=0, parse_dates=True)

    def load_processed_data(self, symbol: str, interval: str, process_type: str) -> Optional[pd.DataFrame]:
        prefix = f"{self._safe_symbol(symbol)}_{interval}_{process_type}_"
        if not os.path.exists(self.processed_data_dir):
            return None

        files = [
            name
            for name in os.listdir(self.processed_data_dir)
            if name.startswith(prefix) and name.endswith(".csv")
        ]
        if not files:
            logger.info(f"未找到 {symbol} {interval} {process_type} 的本地缓存")
            return None

        latest_file = max(files)
        filepath = os.path.join(self.processed_data_dir, latest_file)
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        data.index.name = "date"
        logger.info(f"已加载处理后数据: {filepath}")
        return data

    def _update_metadata(self, symbol: str, interval: str, filepath: str):
        metadata_file = os.path.join(self.data_dir, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r", encoding="utf-8") as file:
                metadata = json.load(file)
        else:
            metadata = {}

        metadata.setdefault(symbol, {})[interval] = {
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filepath": filepath,
        }

        with open(metadata_file, "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

    def get_data_info(self, symbol: Optional[str] = None):
        metadata_file = os.path.join(self.data_dir, "metadata.json")
        if not os.path.exists(metadata_file):
            return {}

        with open(metadata_file, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        if symbol is None:
            return metadata
        return metadata.get(symbol, {})
