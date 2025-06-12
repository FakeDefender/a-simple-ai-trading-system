from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import yfinance as yf
import ccxt
import requests
import json
from .data_storage import DataStorage
from .data_processor import DataProcessor
import akshare as ak
import tushare as ts

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config: Dict):
        """
        初始化数据加载器
        
        Args:
            config: 配置字典，包含数据加载参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_storage = DataStorage(config)
        self.data_processor = DataProcessor()
        
    def load_data(self, symbol: str, interval: str, force_update: bool = True) -> pd.DataFrame:
        """加载市场数据"""
        try:
            self.logger.info(f"开始加载数据 - 交易品种: {symbol}, 时间间隔: {interval}, 强制更新: {force_update}")
            
            # 尝试从本地加载已处理数据
            processed_data = self.data_storage.load_processed_data(
                symbol, 
                interval, 
                process_type='technical_indicators'
            )
            if processed_data is not None and not force_update:
                self.logger.info(f"从本地加载已处理数据成功: {symbol} {interval}")
                self.logger.info(f"本地已处理数据列: {processed_data.columns}")
                self.logger.info(f"本地已处理数据形状: {processed_data.shape}")
                return processed_data
            
            # 如果没有本地数据或需要强制更新，则从API获取数据
            raw_data = self._fetch_market_data(symbol, interval)
            if raw_data is None:
                self.logger.error("获取市场数据失败")
                return None
                
            # 计算技术指标
            processed_data = self.data_processor.calculate_technical_indicators(raw_data)
            
            # 保存处理后的数据
            self.data_storage.save_processed_data(
                processed_data, 
                symbol, 
                interval, 
                process_type='technical_indicators'
            )
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"加载数据时出错: {str(e)}")
            return None
        
    def _fetch_market_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """从API获取市场数据"""
        try:
            self.logger.info(f"从API获取数据: {symbol} {interval}")
            return self._fetch_from_api(symbol, interval)
        except Exception as e:
            self.logger.error(f"从API获取数据时出错: {str(e)}")
            return None
        
    def _fetch_from_api(self, symbol: str, interval: str) -> pd.DataFrame:
        """
        从Stooq API获取数据
        
        Args:
            symbol (str): 交易品种代码，例如 'aapl.us'
            interval (str): 时间间隔，支持 'd'（日线）, 'w'（周线）, 'm'（月线）
            
        Returns:
            pd.DataFrame: 获取的原始OHLCV数据
        """
        try:
            # 构建Stooq数据URL
            # 使用配置文件中的时间范围
            start_date = pd.to_datetime(self.config['data']['start_date'])
            end_date = pd.to_datetime(self.config['data']['end_date'])
            url = f"https://stooq.com/q/d/l/?s={symbol}&d1={start_date.strftime('%Y%m%d')}&d2={end_date.strftime('%Y%m%d')}&i={interval}"
            # 使用pandas直接读取CSV数据
            df = pd.read_csv(url)
            logger.info(f"原始数据列名: {df.columns}")
            # 重命名列以匹配标准格式
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            logger.info(f"重命名后的列名: {df.columns}")
            # 转换日期列
            df['date'] = pd.to_datetime(df['date'])
            # 确保数值列为float类型
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            df.set_index('date', inplace=True)
            # 按日期排序
            df.sort_index(inplace=True)
            
            logger.info(f"成功从Stooq获取原始数据: {symbol} {interval}")
            return df
            
        except Exception as e:
            logger.error(f"从Stooq获取数据失败: {str(e)}")
            return None
        
    def _load_from_csv(self) -> pd.DataFrame:
        """
        从CSV文件加载数据
        """
        file_path = self.config['data']['path']
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
            
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 转换日期列
        date_col = 'datetime' if 'datetime' in df.columns else 'date'
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        
        # 过滤时间范围
        start_date = pd.to_datetime(self.config['data']['start_date'])
        end_date = pd.to_datetime(self.config['data']['end_date'])
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        # 预处理数据
        df = self.preprocess_data(df)
        
        return df
        
    def _load_from_database(self) -> pd.DataFrame:
        """
        从数据库加载数据
        """
        # TODO: 实现数据库连接和数据加载
        raise NotImplementedError("数据库加载功能尚未实现")
        
            
    def _load_crypto_data(self, symbols: List[str], timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载加密货币数据
        """
        # 初始化交易所
        exchange = ccxt.binance({
            'enableRateLimit': True
        })
        
        # 转换时间周期
        timeframe_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        
        # 加载数据
        dfs = []
        for symbol in symbols:
            try:
                # 获取OHLCV数据
                ohlcv = exchange.fetch_ohlcv(
                    symbol,
                    timeframe_map[timeframe],
                    exchange.parse8601(start_date),
                    exchange.parse8601(end_date)
                )
                
                # 转换为DataFrame
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                
                # 转换时间戳
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # 添加交易品种列
                df['symbol'] = symbol
                
                dfs.append(df)
                
            except Exception as e:
                logger.error(f"加载{symbol}数据错误: {str(e)}")
                continue
                
        # 合并数据
        if not dfs:
            raise ValueError("没有成功加载任何数据")
            
        return pd.concat(dfs)
        
    def _load_stock_data(self, symbols: List[str], timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载股票数据
        """
        # 转换时间周期
        interval_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        
        # 加载数据
        dfs = []
        for symbol in symbols:
            try:
                # 获取股票数据
                stock = yf.Ticker(symbol)
                df = stock.history(
                    start=start_date,
                    end=end_date,
                    interval=interval_map[timeframe]
                )
                
                # 添加交易品种列
                df['symbol'] = symbol
                
                dfs.append(df)
                
            except Exception as e:
                logger.error(f"加载{symbol}数据错误: {str(e)}")
                continue
                
        # 合并数据
        if not dfs:
            raise ValueError("没有成功加载任何数据")
            
        return pd.concat(dfs)
        
    def save_data(self, data: pd.DataFrame, file_path: str):
        """
        保存数据到CSV文件
        """
        try:
            # 创建目录
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 保存数据
            data.to_csv(file_path)
            
            logger.info(f"数据已保存到: {file_path}")
            
        except Exception as e:
            logger.error(f"保存数据错误: {str(e)}")
            raise
            
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理
        """
        df = data.copy()
        
        # 处理缺失值
        df = self._handle_missing_values(df)
        
        # 处理异常值
        df = self._handle_outliers(df)
        
        # 计算技术指标
        df = self.data_processor.calculate_technical_indicators(df)
        
        return df
        
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        处理缺失值
        """
        df = data.copy()
        
        # 前向填充
        df.ffill(inplace=True)
        
        # 后向填充
        df.bfill(inplace=True)
        
        return df
        
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        处理异常值
        """
        df = data.copy()
        
        # 计算价格变化率
        df['returns'] = df['close'].pct_change()
        
        # 计算移动平均和标准差
        df['ma'] = df['close'].rolling(window=20).mean()
        df['std'] = df['close'].rolling(window=20).std()
        
        # 定义异常值阈值
        threshold = 3
        
        # 替换异常值
        df.loc[abs(df['returns']) > threshold * df['std'], 'close'] = df['ma']
        
        # 删除临时列
        df.drop(['returns', 'ma', 'std'], axis=1, inplace=True)
        
        return df
        
    def get_benchmark_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        获取基准指数数据（使用标普500指数）
        
        Args:
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            
        Returns:
            pd.DataFrame: 基准指数数据，包含OHLCV数据
        """
        try:
            # 如果没有指定日期范围，使用配置文件中的日期范围
            if start_date is None:
                start_date = self.config.get('data', {}).get('start_date')
            if end_date is None:
                end_date = self.config.get('data', {}).get('end_date')
                
            # 构建Stooq数据URL
            url = f"https://stooq.com/q/d/l/?s=^spx&d1={start_date.replace('-', '')}&d2={end_date.replace('-', '')}&i=d"
            
            # 使用pandas直接读取CSV数据
            benchmark = pd.read_csv(url)
            
            # 重命名列以匹配标准格式
            benchmark.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            
            # 转换日期列
            benchmark['date'] = pd.to_datetime(benchmark['date'])
            benchmark.set_index('date', inplace=True)
            
            return benchmark
            
        except Exception as e:
            self.logger.error(f"获取基准数据失败: {str(e)}")
            # 如果获取失败，返回一个空的DataFrame
            return pd.DataFrame() 