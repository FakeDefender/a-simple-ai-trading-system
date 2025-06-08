import pandas as pd
import numpy as np
import yfinance as yf
import os
from typing import Dict, List, Optional, Union
import logging

class DataProcessor:
    def __init__(self):
        """
        初始化数据处理器
        """
        self.logger = logging.getLogger(__name__)
        
    async def get_historical_data(self, symbol: str, timeframe: str = "1d") -> pd.DataFrame:
        """
        获取历史数据
        
        Args:
            symbol (str): 交易品种代码
            timeframe (str): 时间周期，默认"1d"（日线）
            
        Returns:
            pd.DataFrame: 历史数据
        """
        try:
            # 使用yfinance获取数据
            data = yf.download(symbol, period="1y", interval=timeframe)
            
            # 重命名列
            data.columns = [col.lower() for col in data.columns]
            
            # 计算技术指标
            data = self.calculate_technical_indicators(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"获取历史数据时出错: {str(e)}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            data (pd.DataFrame): 原始数据
            
        Returns:
            pd.DataFrame: 添加了技术指标的数据
        """
        try:
            self.logger.debug(f"开始计算技术指标，数据行数: {len(data)}")
            
            # 确保数据按日期排序
            data = data.sort_index()
            
            # 检查数据量是否足够
            if len(data) < 50:
                self.logger.warning(f"数据量不足({len(data)}行)，某些长期指标可能不准确")
            
            # 计算移动平均线
            data = self._calculate_moving_averages(data)
            
            # 计算RSI
            data = self._calculate_rsi(data)
            
            # 计算MACD
            data = self._calculate_macd(data)
            
            # 计算布林带
            data = self._calculate_bollinger_bands(data)
            
            # 计算成交量指标
            data = self._calculate_volume_indicators(data)
            
            # 统计缺失值
            missing_stats = self._analyze_missing_values(data)
            self.logger.info(f"技术指标计算完成，缺失值统计: {missing_stats}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"计算技术指标时出错: {str(e)}")
            raise
    
    def _calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算移动平均线"""
        try:
            # 计算5日、10日、20日、50日移动平均线
            data['ma5'] = data['close'].rolling(window=5).mean()
            data['ma10'] = data['close'].rolling(window=10).mean()
            data['ma20'] = data['close'].rolling(window=20).mean()
            data['ma50'] = data['close'].rolling(window=50).mean()
            
            # 计算移动平均线的斜率
            data['ma5_slope'] = data['ma5'].pct_change(periods=5)
            data['ma10_slope'] = data['ma10'].pct_change(periods=10)
            data['ma20_slope'] = data['ma20'].pct_change(periods=20)
            
            # 计算移动平均线的交叉信号
            data['ma5_10_cross'] = np.where(data['ma5'] > data['ma10'], 1, -1)
            data['ma10_20_cross'] = np.where(data['ma10'] > data['ma20'], 1, -1)
            
            self.logger.debug("移动平均线计算完成")
            return data
            
        except Exception as e:
            self.logger.error(f"计算移动平均线时出错: {str(e)}")
            raise
    
    def _calculate_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算RSI指标 - 使用Wilder's方法
        
        Args:
            data (pd.DataFrame): 原始数据
            
        Returns:
            pd.DataFrame: 添加了RSI指标的数据
        """
        try:
            self.logger.debug("开始计算RSI...")
            
            # 计算价格变化
            delta = data['close'].diff()
            
            # 分离上涨和下跌
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # 使用Wilder's平滑方法（指数移动平均）
            alpha = 1.0 / 14  # Wilder's平滑因子
            avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
            avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
            
            # 计算相对强度
            rs = avg_gain / avg_loss.replace(0, np.inf)
            
            # 计算RSI
            rsi = 100 - (100 / (1 + rs))
            
            # 处理无效值 - 前14个值设为NaN，而不是50
            rsi.iloc[:14] = np.nan
            
            # 处理无穷值
            rsi = rsi.replace([np.inf, -np.inf], np.nan)
            
            data['rsi'] = rsi
            self.logger.debug(f"RSI计算完成，有效值从第{rsi.first_valid_index()}开始")
            return data
            
        except Exception as e:
            self.logger.error(f"计算RSI时出错: {str(e)}")
            raise
    
    def _calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算MACD指标
        
        Args:
            data (pd.DataFrame): 原始数据
            
        Returns:
            pd.DataFrame: 添加了MACD指标的数据
        """
        try:
            self.logger.debug("开始计算MACD...")
            
            exp1 = data['close'].ewm(span=12, adjust=False).mean()
            exp2 = data['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            data['macd'] = macd
            data['signal'] = signal
            
            self.logger.debug("MACD计算完成")
            return data
            
        except Exception as e:
            self.logger.error(f"计算MACD时出错: {str(e)}")
            raise
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算布林带
        
        Args:
            data (pd.DataFrame): 原始数据
            
        Returns:
            pd.DataFrame: 添加了布林带的数据
        """
        try:
            self.logger.debug("开始计算布林带...")
            
            # 计算移动平均线
            middle_band = data['close'].rolling(window=20).mean()
            
            # 计算标准差
            std = data['close'].rolling(window=20).std()
            
            # 计算上轨和下轨
            upper_band = middle_band + (std * 2)
            lower_band = middle_band - (std * 2)
            
            data['bb_middle'] = middle_band
            data['bb_upper'] = upper_band
            data['bb_lower'] = lower_band
            
            self.logger.debug("布林带计算完成")
            return data
            
        except Exception as e:
            self.logger.error(f"计算布林带时出错: {str(e)}")
            raise
    
    def _calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量指标
        
        Args:
            data (pd.DataFrame): 原始数据
            
        Returns:
            pd.DataFrame: 添加了成交量指标的数据
        """
        try:
            self.logger.debug("开始计算成交量指标...")
            
            # 计算成交量移动平均线
            data['volume_sma'] = data['volume'].rolling(window=20).mean()
            self.logger.debug(f"成交量移动平均线计算完成，成交量移动平均线值: {data['volume_sma'].iloc[-1]:.2f}")
            
            # 计算成交量比率
            data['volume_ratio'] = data['volume'] / data['volume_sma']
            self.logger.debug(f"成交量比率计算完成，成交量比率值: {data['volume_ratio'].iloc[-1]:.2f}")
            
            self.logger.debug("成交量指标计算完成")
            return data
            
        except Exception as e:
            self.logger.error(f"计算成交量指标时出错: {str(e)}")
            raise
    
    def calculate_market_indicators(self, market_summary: Dict) -> Dict:
        """计算市场指标"""
        self.logger.info("从汇总数据计算市场指标，数据键: %s", list(market_summary.keys()))
        
        # 计算趋势强度
        trend_strength = self._calculate_trend_strength(market_summary)
        
        # 计算市场强度
        market_strength = self._calculate_market_strength(market_summary)
        
        # 计算波动性
        volatility = self._calculate_volatility(market_summary)
        
        return {
            'trend_strength': trend_strength,
            'market_strength': market_strength,
            'volatility': volatility
        }
    
    def calculate_risk_metrics(self, data: Union[pd.DataFrame, Dict]) -> Dict:
        """
        计算风险指标，支持DataFrame和汇总数据两种格式
        
        Args:
            data: 市场数据，可以是DataFrame或汇总数据字典
            
        Returns:
            Dict: 风险指标
        """
        try:
            self.logger.info(f"开始计算风险指标，输入数据类型: {type(data)}")
            
            if isinstance(data, pd.DataFrame):
                self.logger.info(f"DataFrame列名: {data.columns}")
                self.logger.info(f"DataFrame前几行数据:\n{data.head()}")
                return self._calculate_risk_metrics_from_df(data)
            elif isinstance(data, dict):
                self.logger.info(f"汇总数据键: {data.keys()}")
                return self._calculate_risk_metrics_from_summary(data)
            else:
                raise ValueError(f"不支持的数据类型: {type(data)}")
                
        except Exception as e:
            self.logger.error(f"计算风险指标时出错: {str(e)}")
            self.logger.error(f"错误详情: {type(e).__name__}")
            import traceback
            self.logger.error(f"错误堆栈: {traceback.format_exc()}")
            raise
    
    def _calculate_risk_metrics_from_df(self, data: pd.DataFrame) -> Dict:
        """
        从DataFrame计算风险指标
        
        Args:
            data: 包含原始价格数据的DataFrame
            
        Returns:
            Dict: 风险指标
        """
        try:
            self.logger.info(f"从DataFrame计算风险指标，数据列: {data.columns}")
            
            # 计算收益率
            returns = data['close'].pct_change()
            
            # 计算波动率
            volatility = returns.std() * np.sqrt(252)
            self.logger.info(f"计算得到波动率: {volatility}")
            
            # 计算最大回撤
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            max_drawdown = drawdowns.min()
            self.logger.info(f"计算得到最大回撤: {max_drawdown}")
            
            # 计算夏普比率
            risk_free_rate = 0.02  # 假设无风险利率为2%
            excess_returns = returns.mean() * 252 - risk_free_rate
            sharpe_ratio = excess_returns / volatility if volatility != 0 else 0
            self.logger.info(f"计算得到夏普比率: {sharpe_ratio}")
            
            # 计算VaR和期望尾部损失
            var_95 = returns.quantile(0.05)
            expected_shortfall = returns[returns <= var_95].mean()
            self.logger.info(f"计算得到VaR(95%): {var_95}")
            self.logger.info(f"计算得到期望尾部损失: {expected_shortfall}")
            
            return {
                'volatility': float(volatility),
                'max_drawdown': float(max_drawdown),
                'sharpe_ratio': float(sharpe_ratio),
                'var_95': float(var_95),
                'expected_shortfall': float(expected_shortfall)
            }
            
        except Exception as e:
            self.logger.error(f"从DataFrame计算风险指标时出错: {str(e)}")
            raise
    
    def _calculate_risk_metrics_from_summary(self, data: Dict) -> Dict:
        """
        从汇总数据计算风险指标
        
        Args:
            data: 包含汇总市场数据的字典
            
        Returns:
            Dict: 风险指标
        """
        try:
            self.logger.info(f"从汇总数据计算风险指标，数据键: {data.keys()}")
            
            # 从移动平均线计算波动性
            ma_data = data.get('moving_averages', {})
            if ma_data:
                # 确保所有值为Python原生类型
                ma_values = [float(v) for v in ma_data.values()]
                volatility = np.std(ma_values) / np.mean(ma_values) if ma_values else 0
            else:
                volatility = 0
            self.logger.info(f"计算得到波动性: {volatility}")
            
            # 从RSI计算市场风险
            rsi = data.get('rsi')
            if rsi is None:
                self.logger.warning("RSI值为None，使用默认值50")
                rsi = 50
            else:
                rsi = float(rsi)  # 确保RSI为Python原生类型
            market_risk = abs(rsi - 50) / 50  # 归一化RSI偏离度
            self.logger.info(f"计算得到市场风险: {market_risk}")
            
            # 从移动平均线计算趋势风险
            if ma_data:
                ma5 = float(ma_data.get('ma5', 0))
                ma20 = float(ma_data.get('ma20', 0))
                trend_risk = abs((ma5 - ma20) / ma20) if ma20 != 0 else 0
            else:
                trend_risk = 0
            self.logger.info(f"计算得到趋势风险: {trend_risk}")
            
            return {
                'volatility': float(volatility),
                'market_risk': float(market_risk),
                'trend_risk': float(trend_risk)
            }
            
        except Exception as e:
            self.logger.error(f"从汇总数据计算风险指标时出错: {str(e)}")
            raise
    
    def calculate_correlation_matrix(self, symbols: List[str]) -> pd.DataFrame:
        """
        计算相关性矩阵
        
        Args:
            symbols (List[str]): 交易品种列表
            
        Returns:
            pd.DataFrame: 相关性矩阵
        """
        try:
            data = {}
            for symbol in symbols:
                data[symbol] = yf.download(symbol, period="1y")['Close']
                
            df = pd.DataFrame(data)
            return df.corr()
            
        except Exception as e:
            self.logger.error(f"计算相关性矩阵时出错: {str(e)}")
            raise 

    def _calculate_trend_strength(self, data: Dict) -> float:
        """计算趋势强度"""
        try:
            ma_data = data.get('moving_averages', {})
            if not ma_data:
                return 0.0
                
            ma5 = float(ma_data.get('ma5', 0))
            ma20 = float(ma_data.get('ma20', 0))
            return (ma5 - ma20) / ma20 if ma20 != 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"计算趋势强度时出错: {str(e)}")
            return 0.0
            
    def _calculate_market_strength(self, data: Union[Dict, pd.DataFrame]) -> float:
        """计算市场强度
        
        Args:
            data: 市场数据，可以是字典或DataFrame
            
        Returns:
            float: 市场强度值 (0-1)
        """
        try:
            # 如果是DataFrame，获取最新一行数据
            if isinstance(data, pd.DataFrame):
                if data.empty:
                    self.logger.warning("市场数据为空，返回中性市场强度")
                    return 0.5
                latest_data = data.iloc[-1]
                rsi = float(latest_data['rsi'])
                macd = float(latest_data['macd'])
                signal = float(latest_data['signal'])
                volume = float(latest_data['volume'])
                volume_sma = float(latest_data['volume_sma'])
            else:
                # 如果是字典，直接获取值
                rsi = float(data.get('rsi', 50))
                macd = float(data.get('macd', 0))
                signal = float(data.get('signal', 0))
                volume = float(data.get('volume', 0))
                volume_sma = float(data.get('volume_sma', 0))
            
            # 计算RSI的贡献
            rsi_contribution = (rsi - 30) / 40  # 归一化到0-1范围
            rsi_contribution = max(0, min(1, rsi_contribution))  # 限制在0-1范围内
            
            # 计算MACD的贡献
            macd_contribution = 0.5 + (macd - signal) / (2 * abs(signal)) if signal != 0 else 0.5
            macd_contribution = max(0, min(1, macd_contribution))  # 限制在0-1范围内
            
            # 计算成交量的贡献
            volume_contribution = min(volume / volume_sma, 2) / 2 if volume_sma != 0 else 0.5  # 归一化到0-1范围
            
            # 计算加权平均
            market_strength = (
                0.4 * rsi_contribution +
                0.4 * macd_contribution +
                0.2 * volume_contribution
            )
            
            return float(market_strength)  # 确保返回float类型
            
        except Exception as e:
            self.logger.error(f"计算市场强度时出错: {str(e)}")
            return 0.5  # 发生错误时返回中性值
    
    def _calculate_volatility(self, data: Dict) -> float:
        """计算波动性"""
        try:
            ma_data = data.get('moving_averages', {})
            if not ma_data:
                return 0.0
                
            ma_values = [float(v) for v in ma_data.values()]
            return np.std(ma_values) / np.mean(ma_values) if ma_values else 0.0
            
        except Exception as e:
            self.logger.error(f"计算波动性时出错: {str(e)}")
            return 0.0
            
    def _calculate_volatility_from_df(self, data: pd.DataFrame) -> float:
        """从DataFrame计算波动率"""
        try:
            return data['close'].pct_change().std() * np.sqrt(252)
        except Exception as e:
            self.logger.error(f"从DataFrame计算波动率时出错: {str(e)}")
            return 0.0
            
    def _calculate_trend_strength_from_df(self, data: pd.DataFrame) -> float:
        """从DataFrame计算趋势强度"""
        try:
            return abs(data['close'].pct_change().mean() * 252)
        except Exception as e:
            self.logger.error(f"从DataFrame计算趋势强度时出错: {str(e)}")
            return 0.0
            
    def _calculate_volume_trend(self, data: pd.DataFrame) -> float:
        """计算成交量趋势"""
        try:
            return data['volume'].pct_change().mean() * 252
        except Exception as e:
            self.logger.error(f"计算成交量趋势时出错: {str(e)}")
            return 0.0
    
    def _analyze_missing_values(self, data: pd.DataFrame) -> Dict:
        """分析缺失值情况"""
        missing_stats = {}
        
        # 检查各个指标的缺失情况
        indicators = ['ma5', 'ma10', 'ma20', 'ma50', 'rsi', 'macd', 'signal', 
                     'bb_middle', 'bb_upper', 'bb_lower', 'volume_sma', 'volume_ratio']
        
        for indicator in indicators:
            if indicator in data.columns:
                missing_count = data[indicator].isna().sum()
                total_count = len(data)
                missing_stats[indicator] = {
                    'missing_count': missing_count,
                    'missing_percentage': round(missing_count / total_count * 100, 2),
                    'first_valid_index': data[indicator].first_valid_index()
                } 
        
        return missing_stats
    
    def get_complete_data(self, data: pd.DataFrame, min_periods: int = 50) -> pd.DataFrame:
        """
        获取没有缺失值的完整数据
        
        Args:
            data: 包含技术指标的数据
            min_periods: 最小数据期数，默认50（确保MA50等长期指标有效）
            
        Returns:
            pd.DataFrame: 没有缺失值的完整数据
        """
        try:
            self.logger.info(f"原始数据行数: {len(data)}")
            
            # 方法1: 删除所有包含NaN的行
            complete_data = data.dropna()
            self.logger.info(f"删除缺失值后行数: {len(complete_data)}")
            
            # 方法2: 从指定期数开始（确保长期指标有效）
            if len(data) > min_periods:
                data_from_period = data.iloc[min_periods:]
                complete_from_period = data_from_period.dropna()
                self.logger.info(f"从第{min_periods}期开始的完整数据行数: {len(complete_from_period)}")
                
                # 选择数据更完整的方案
                if len(complete_from_period) > len(complete_data):
                    complete_data = complete_from_period
                    self.logger.info(f"使用从第{min_periods}期开始的数据")
            
            # 验证数据质量
            if len(complete_data) == 0:
                raise ValueError("没有找到完整的数据行")
            
            # 检查关键指标是否存在
            required_indicators = ['ma5', 'ma10', 'ma20', 'rsi', 'macd']
            missing_indicators = [ind for ind in required_indicators if ind not in complete_data.columns]
            if missing_indicators:
                self.logger.warning(f"缺少关键指标: {missing_indicators}")
            
            self.logger.info(f"返回完整数据: {len(complete_data)}行，时间范围: {complete_data.index[0]} 到 {complete_data.index[-1]}")
            return complete_data
            
        except Exception as e:
            self.logger.error(f"获取完整数据时出错: {str(e)}")
            raise
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict:
        """
        获取数据摘要信息
        
        Args:
            data: 数据DataFrame
            
        Returns:
            Dict: 数据摘要
        """
        try:
            summary = {
                'total_rows': len(data),
                'complete_rows': len(data.dropna()),
                'missing_percentage': round((len(data) - len(data.dropna())) / len(data) * 100, 2),
                'date_range': {
                    'start': str(data.index[0]) if len(data) > 0 else None,
                    'end': str(data.index[-1]) if len(data) > 0 else None
                },
                'columns': list(data.columns),
                'missing_by_column': {}
            }
            
            # 统计每列的缺失值
            for col in data.columns:
                missing_count = data[col].isna().sum()
                summary['missing_by_column'][col] = {
                    'missing_count': missing_count,
                    'missing_percentage': round(missing_count / len(data) * 100, 2)
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"生成数据摘要时出错: {str(e)}")
            return {} 