import pandas as pd
import os
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataStorage:
    def __init__(self, config):
        """
        初始化数据存储工具
        
        Args:
            config (dict): 配置信息，包含数据存储路径等
        """
        self.config = config
        self.data_dir = os.path.join(config.get('data_dir', 'data'), 'market_data')
        self.raw_data_dir = os.path.join(self.data_dir, 'raw')
        self.processed_data_dir = os.path.join(self.data_dir, 'processed')
        
        # 创建必要的目录
        self._create_directories()
        
    def _create_directories(self):
        """创建数据存储目录"""
        for directory in [self.data_dir, self.raw_data_dir, self.processed_data_dir]:
            os.makedirs(directory, exist_ok=True)
            
    def save_raw_data(self, data: pd.DataFrame, symbol: str, interval: str):
        """
        保存原始数据
        
        Args:
            data (pd.DataFrame): 要保存的数据
            symbol (str): 交易品种代码
            interval (str): 时间间隔
        """
        try:
            # 生成文件名
            filename = f"{symbol}_{interval}_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join(self.raw_data_dir, filename)
            
            # 保存数据
            data.to_csv(filepath, index=True)
            logger.info(f"原始数据已保存到: {filepath}")
            
            # 更新元数据
            self._update_metadata(symbol, interval, filepath)
            
        except Exception as e:
            logger.error(f"保存原始数据失败: {str(e)}")
            raise
            
    def save_processed_data(self, data: pd.DataFrame, symbol: str, interval: str, process_type: str):
        """
        保存处理后的数据
        
        Args:
            data (pd.DataFrame): 要保存的数据
            symbol (str): 交易品种代码
            interval (str): 时间间隔
            process_type (str): 处理类型（如 'technical_indicators', 'ml_features' 等）
        """
        try:
            # 生成文件名
            filename = f"{symbol}_{interval}_{process_type}_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join(self.processed_data_dir, filename)
            
            # 保存数据
            data.to_csv(filepath, index=True)
            logger.info(f"处理后的数据已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"保存处理后的数据失败: {str(e)}")
            raise
            
    def load_latest_data(self, symbol: str, interval: str, data_type: str = 'raw'):
        """
        加载最新的数据
        
        Args:
            symbol (str): 交易品种代码
            interval (str): 时间间隔
            data_type (str): 数据类型（'raw' 或 'processed'）
            
        Returns:
            pd.DataFrame: 加载的数据
        """
        try:
            # 确定目录
            directory = self.raw_data_dir if data_type == 'raw' else self.processed_data_dir
            
            # 查找最新的数据文件
            pattern = f"{symbol}_{interval}_"
            files = [f for f in os.listdir(directory) if f.startswith(pattern)]
            
            if not files:
                logger.warning(f"未找到 {symbol} {interval} 的数据文件")
                return None
                
            # 获取最新的文件
            latest_file = max(files)
            filepath = os.path.join(directory, latest_file)
            
            # 加载数据
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logger.info(f"已加载数据: {filepath}")
            
            return data
            
        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            raise
            
    def _update_metadata(self, symbol: str, interval: str, filepath: str):
        """
        更新元数据
        
        Args:
            symbol (str): 交易品种代码
            interval (str): 时间间隔
            filepath (str): 数据文件路径
        """
        try:
            metadata_file = os.path.join(self.data_dir, 'metadata.json')
            
            # 加载现有元数据
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
                
            # 更新元数据
            if symbol not in metadata:
                metadata[symbol] = {}
                
            metadata[symbol][interval] = {
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'filepath': filepath
            }
            
            # 保存元数据
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
                
        except Exception as e:
            logger.error(f"更新元数据失败: {str(e)}")
            raise
            
    def get_data_info(self, symbol: str = None):
        """
        获取数据信息
        
        Args:
            symbol (str, optional): 交易品种代码，如果为None则返回所有品种的信息
            
        Returns:
            dict: 数据信息
        """
        try:
            metadata_file = os.path.join(self.data_dir, 'metadata.json')
            
            if not os.path.exists(metadata_file):
                return {}
                
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            if symbol:
                return metadata.get(symbol, {})
            else:
                return metadata
                
        except Exception as e:
            logger.error(f"获取数据信息失败: {str(e)}")
            raise

    def load_processed_data(self, symbol, interval, process_type):
        """
        加载本地已处理（带技术指标）数据
        Args:
            symbol (str): 交易品种代码
            interval (str): 时间间隔
            process_type (str): 处理类型（如 'technical_indicators'）
        Returns:
            pd.DataFrame or None
        """
        try:
            # 查找最新的已处理数据文件
            pattern = f"{symbol}_{interval}_{process_type}_"
            files = [f for f in os.listdir(self.processed_data_dir) if f.startswith(pattern) and f.endswith('.csv')]
            if not files:
                logger.warning(f"未找到 {symbol} {interval} {process_type} 的已处理数据文件")
                return None
            latest_file = max(files)
            filepath = os.path.join(self.processed_data_dir, latest_file)
            df = pd.read_csv(filepath)
            # 自动识别日期列并设为索引
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            elif 'Unnamed: 0' in df.columns:
                # 兼容index_col=0的情况
                df = df.rename(columns={'Unnamed: 0': 'date'})
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            logger.info(f"已加载已处理数据: {filepath}")
            return df
        except Exception as e:
            logger.error(f"加载已处理数据失败: {str(e)}")
            return None 