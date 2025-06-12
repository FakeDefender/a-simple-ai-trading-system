import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
import yaml
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from crewai import Agent, Task, Crew
from crewai.tools import tool
from src.utils.data_loader import DataLoader
from src.utils.data_processor import DataProcessor
from src.utils.risk_calculator import RiskCalculator
from src.utils.openai_client import OpenAIClient
import re
import ast

logger = logging.getLogger(__name__)

class MLStrategyAgent:
    """机器学习策略代理"""
    
    # Agent输出格式定义
    AGENT_OUTPUT_FORMAT = {
        "strategy_developer": {
            "strategy_name": "string",
            "strategy_description": "string",
            "parameters": {
                "trend_threshold": "float",
                "market_strength_threshold": "float",
                "volatility_threshold": "float",
                "position_size": "float",
                "stop_loss_atr": "float",
                "take_profit_atr": "float"
            },
            "entry_conditions": ["string"],
            "exit_conditions": ["string"],
            "position_management": {"key": "value"},
            "risk_management": {"key": "value"}
        },
        "risk_analyst": {
            "market_state": {"key": "value"},
            "risk_metrics": {"key": "value"},
            "risk_level": "string",
            "risk_control_suggestions": ["string"],
            "position_suggestions": {"key": "value"}
        },
        "trading_advisor": {
            "current_signal": "string",
            "entry_points": ["float"],
            "exit_points": ["float"],
            "position_size": "float",
            "stop_loss": "float",
            "take_profit": "float",
            "time_horizon": "string"
        }
    }

    def __init__(self, config: Dict[str, Any], data_loader: DataLoader, strategy_params=None):
        """
        初始化机器学习策略代理
        
        Args:
            config: 配置字典
            data_loader: 数据加载器
            strategy_params: 策略参数
        """
        # 保存配置
        self.config = config
        self.data_loader = data_loader
        self.logger = logging.getLogger(__name__);
        
        # 创建结果保存目录
        self.results_dir = os.path.join('results', datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        # 初始化LLM
        self.llm = self._initialize_llm()
        
        # 初始化数据处理器
        self.data_processor = DataProcessor()
        
        # 初始化AI代理
        self.strategy_developer = Agent(
            role='策略开发专家',
            goal='基于技术指标开发最优交易策略',
            backstory='我是一位专业的量化交易策略专家，擅长基于技术指标开发交易策略。我会仔细分析市场数据和技术指标，设计出稳健的交易策略。',
            tools=[],
            llm=self.llm,
            verbose=True
        )
        
        self.risk_analyst = Agent(
            role='风险分析师',
            goal='评估和管理交易风险',
            backstory='我是一位专业的风险分析师，擅长评估和管理交易风险。我会基于技术指标和市场数据，提供详细的风险评估。',
            tools=[],
            llm=self.llm,
            verbose=True
        )
        
        self.trade_advisor = Agent(
            role='交易顾问',
            goal='提供交易建议',
            backstory='我是一位专业的交易顾问，擅长基于技术分析提供交易建议。我会结合多个技术指标，给出清晰的交易建议。',
            tools=[],
            llm=self.llm,
            verbose=True
        )
        
        # 创建任务
        self.strategy_task = Task(
            description="基于技术指标开发交易策略",
            expected_output="输出一个基于技术指标的完整交易策略，包括策略逻辑、参数设置和信号生成规则",
            agent=self.strategy_developer
        )
        
        self.risk_task = Task(
            description="基于技术指标评估交易风险",
            expected_output="输出基于技术指标的风险评估报告，包括风险指标、最大回撤分析和风险控制建议",
            agent=self.risk_analyst
        )
        
        self.trade_task = Task(
            description="基于技术指标提供交易建议",
            expected_output="输出基于技术指标的具体交易建议，包括入场点、止损位和目标价位",
            agent=self.trade_advisor
        )
        
        # 创建团队
        self.crew = Crew(
            agents=[self.strategy_developer, self.risk_analyst, self.trade_advisor],
            tasks=[self.strategy_task, self.risk_task, self.trade_task],
            verbose=True
        )

        # 策略参数集中管理
        self.strategy_params = strategy_params or {
            'rsi_long': 45,              # 多头RSI阈值
            'rsi_short': 55,             # 空头RSI阈值
            'macd_trend': 0,             # MACD阈值
            'volume_ratio': 1.05,        # 成交量比率
            'ma5_offset': 0.98,          # MA5偏离容忍度（多头）
            'ma5_offset_short': 1.02,    # MA5偏离容忍度（空头）
            'bb_upper_offset': 0.99,     # 布林带上轨偏移（空头）
            'bb_lower_offset': 1.01,     # 布林带下轨偏移（多头）
            'stop_loss_pct': 0.02,       # 止损比例
            'take_profit_pct': 0.05,     # 止盈比例
            'min_entry_conditions': 1,   # 最少满足条件数
            'rsi_exit_long': 75,         # 多头RSI出场
            'rsi_exit_short': 25,        # 空头RSI出场
        }

    def set_params(self, **kwargs):
        self.strategy_params.update(kwargs)

    def _initialize_strategy_params(self) -> Dict:
        """初始化策略参数"""
        return {
            'trend_threshold': 0.01,
            'market_strength_threshold': 0.1,
            'volatility_threshold': 0.01,
            'position_size': 0.2,
            'stop_loss_atr': 2.0,
            'take_profit_atr': 1.5
        }

    def _save_backtest_results(self, backtest_results: Dict, performance: Dict, risk_metrics: Dict, recommendations: List[str]):
        """
        保存回测结果到文件
        """
        try:
            os.makedirs(self.results_dir, exist_ok=True)
            # 保存交易记录
            trades_df = pd.DataFrame(backtest_results['trades'])
            trades_df.to_csv(os.path.join(self.results_dir, 'trades.csv'), index=False)
            
            # 保存权益曲线
            equity_curve = backtest_results['equity_curve']
            equity_curve.to_csv(os.path.join(self.results_dir, 'equity_curve.csv'))
            
            # 保存性能指标
            metrics = {
                'performance': performance,
                'risk_metrics': risk_metrics,
                'recommendations': recommendations
            }
            with open(os.path.join(self.results_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=4)
            
            # 生成并保存图表
            self._plot_backtest_results(backtest_results)
            
            self.logger.info(f"回测结果已保存到目录: {self.results_dir}")
            
        except Exception as e:
            self.logger.error(f"保存回测结果失败: {str(e)}")
    
    def _plot_backtest_results(self, backtest_results):
        """
        绘制回测结果图表
        """
        try:
            # 使用matplotlib的默认样式
            plt.style.use('default')
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # 绘制权益曲线
            equity_curve = backtest_results["equity_curve"]
            ax1.plot(equity_curve.index, equity_curve.values, label='权益曲线', color='blue')
            ax1.set_title('策略表现')
            ax1.set_xlabel('日期')
            ax1.set_ylabel('权益')
            ax1.legend()
            ax1.grid(True)
            
            # 绘制回撤
            returns = equity_curve.pct_change().dropna()
            cumulative_returns = (1 + returns).cumprod()
            drawdown = (cumulative_returns / cumulative_returns.expanding().max() - 1)
            ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3, label='回撤')
            ax2.set_title('回撤分析')
            ax2.set_xlabel('日期')
            ax2.set_ylabel('回撤')
            ax2.legend()
            ax2.grid(True)
            
            # 调整布局并保存
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'backtest_results.png'))
            plt.close()
            
        except Exception as e:
            self.logger.error(f"绘制回测结果图表时出错: {str(e)}")
    
    def _calculate_drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """
        计算回撤序列
        
        Args:
            equity_curve: 权益曲线
            
        Returns:
            pd.Series: 回撤序列
        """
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown

    def _initialize_llm(self):
        """
        初始化LLM
        """
        try:
            # 创建OpenAI客户端
            client = OpenAIClient(self.config)
            
            # 获取LLM实例
            return client.get_llm()
            
        except Exception as e:
            self.logger.error(f"初始化LLM失败: {str(e)}")
            raise
            
    def _get_agent_advice(self, data: pd.DataFrame) -> Dict:
        """
        获取智能体建议
        
        Args:
            data (pd.DataFrame): 包含历史数据的DataFrame
            
        Returns:
            Dict: 包含各个智能体建议的字典
        """
        try:
            # 计算技术指标
            # logger.info("开始计算技术指标...")
            # data = self.data_processor.calculate_technical_indicators(data)
            # logger.info("技术指标计算完成")
            
            # 准备市场数据摘要
            # 使用最近20个交易日的数据进行分析
            recent_data = data.tail(20)
            
            market_summary = {
                "current": {
                    "price": float(data['close'].iloc[-1]),
                    "ma5": float(data['ma5'].iloc[-1]),
                    "ma10": float(data['ma10'].iloc[-1]),
                    "ma20": float(data['ma20'].iloc[-1]),
                    "rsi": float(data['rsi'].iloc[-1]),
                    "macd": float(data['macd'].iloc[-1]),
                    "volume": float(data['volume'].iloc[-1]),
                    "volume_ratio": float(data['volume_ratio'].iloc[-1])
                },
                "historical": {
                    "prices": recent_data['close'].tolist(),
                    "volumes": recent_data['volume'].tolist(),
                    "rsi_values": recent_data['rsi'].tolist(),
                    "macd_values": recent_data['macd'].tolist(),
                    "ma5_values": recent_data['ma5'].tolist(),
                    "ma10_values": recent_data['ma10'].tolist(),
                    "ma20_values": recent_data['ma20'].tolist()
                },
                "statistics": {
                    "price_change": float(data['close'].iloc[-1] / data['close'].iloc[-2] - 1),  # 日涨跌幅
                    "volume_change": float(data['volume'].iloc[-1] / data['volume'].iloc[-2] - 1),  # 成交量变化
                    "volatility": float(recent_data['close'].pct_change().std()),  # 波动率
                    "trend": self._calculate_trend(recent_data)  # 趋势方向
                }
            }
            
            logger.info(f"市场数据摘要准备完成")
            
            # 获取策略开发者建议
            strategy_advice = self._get_strategy_developer_advice(market_summary)
            logger.info(f"策略开发者建议: {json.dumps(strategy_advice, indent=2)}")
            
            # 获取风险分析师建议
            risk_advice = self._get_risk_analyst_advice(market_summary)
            logger.info(f"风险分析师建议: {json.dumps(risk_advice, indent=2)}")
            
            # 获取交易顾问建议
            trading_advice = self._get_trading_advisor_advice(market_summary)
            logger.info(f"交易顾问建议: {json.dumps(trading_advice, indent=2)}")
            
            # 合并所有建议
            advice = {
                "strategy_developer": strategy_advice,
                "risk_analyst": risk_advice,
                "trading_advisor": trading_advice
            }
            
            logger.info("完成获取智能体建议")
            return advice
            
        except Exception as e:
            self.logger.error(f"获取智能体建议失败: {str(e)}")
            return {
                "strategy_developer": {},
                "risk_analyst": {},
                "trading_advisor": {}
            }
            
    def _calculate_trend(self, data: pd.DataFrame) -> str:
        """
        计算趋势方向
        
        Args:
            data (pd.DataFrame): 历史数据
            
        Returns:
            str: 趋势方向 ('up', 'down', 'sideways')
        """
        # 使用20日移动平均线判断趋势
        ma20 = data['ma20']
        current_price = data['close'].iloc[-1]
        
        if current_price > ma20.iloc[-1] * 1.02:  # 价格高于MA20 2%以上
            return 'up'
        elif current_price < ma20.iloc[-1] * 0.98:  # 价格低于MA20 2%以上
            return 'down'
        else:
            return 'sideways'

    def _extract_json_from_llm_response(self, text: str) -> dict:
        """
        从LLM返回内容中提取JSON部分并解析,支持标准JSON和Python dict格式
        """
        try:
            # 优先提取 ```json ... ``` 代码块
            match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', text)
            if match:
                json_str = match.group(1)
            else:
                match = re.search(r'(\{[\s\S]*\})', text)
                if match:
                    json_str = match.group(1)
                else:
                    self.logger.error(f"未找到有效JSON: {text}")
                    return {}
            # 先尝试标准JSON解析
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # 再尝试Python字典解析
                return ast.literal_eval(json_str)
        except Exception as e:
            self.logger.error(f"解析LLM建议JSON失败: {e}\n原始内容: {text}")
            return {}

    def _get_strategy_developer_advice(self, market_summary: Dict) -> Dict:
        """
        获取策略开发者建议
        
        Args:
            market_summary (Dict): 包含当前和历史市场数据的摘要
            
        Returns:
            Dict: 策略建议
        """
        try:
            # 准备提示词
            prompt = f"""
            作为策略开发者，请分析以下市场数据并提供交易策略建议：
            
            当前市场数据：
            {json.dumps(market_summary['current'], indent=2)}
            
            历史数据统计：
            {json.dumps(market_summary['statistics'], indent=2)}
            
            请严格按照以下格式返回标准 JSON，不要加任何注释、解释或 markdown 格式。
            
            策略要求：
            1. 入场条件适中，建议2-3个条件，优先趋势跟随策略
            2. RSI条件：买入时用 "rsi > 45" 且 "rsi < 65"，避免极端值
            3. MACD条件：买入时用 "macd > signal" 或 "macd > -1.0"
            4. 价格趋势：买入时用 "price > ma5" 或 "ma5 > ma20"
            5. 出场条件：用 "rsi > 70" 或 "price < ma10"
            6. 确保是趋势跟随策略，不要逆势交易
            7. 考虑历史波动率和趋势方向
            
            {self.AGENT_OUTPUT_FORMAT['strategy_developer']}
            """
            # 发送提示词到LLM
            response = self.llm.call(prompt)
            # 用增强的解析逻辑
            advice = self._extract_json_from_llm_response(response)
            return advice
        except Exception as e:
            self.logger.error(f"获取策略开发者建议失败: {str(e)}")
            return {}

    def _get_risk_analyst_advice(self, market_summary: Dict) -> Dict:
        """
        获取风险分析师建议
        
        Args:
            market_summary (Dict): 包含当前和历史市场数据的摘要
            
        Returns:
            Dict: 风险分析建议
        """
        try:
            prompt = f"""
            作为风险分析师，请评估以下市场数据的风险水平：
            
            当前市场数据：
            {json.dumps(market_summary['current'], indent=2)}
            
            历史数据统计：
            {json.dumps(market_summary['statistics'], indent=2)}
            
            请考虑以下因素：
            1. 当前波动率水平
            2. 价格趋势方向
            3. 成交量变化
            4. 技术指标状态
            
            请严格按照以下格式返回标准 JSON，不要加任何注释、解释或 markdown 格式：
            {self.AGENT_OUTPUT_FORMAT['risk_analyst']}
            """
            response = self.llm.call(prompt)
            advice = self._extract_json_from_llm_response(response)
            return advice
        except Exception as e:
            self.logger.error(f"获取风险分析师建议失败: {str(e)}")
            return {}

    def _get_trading_advisor_advice(self, market_summary: Dict) -> Dict:
        """
        获取交易顾问建议
        
        Args:
            market_summary (Dict): 包含当前和历史市场数据的摘要
            
        Returns:
            Dict: 交易建议
        """
        try:
            prompt = f"""
            作为交易顾问，请根据以下市场数据提供具体的交易建议：
            
            当前市场数据：
            {json.dumps(market_summary['current'], indent=2)}
            
            历史数据统计：
            {json.dumps(market_summary['statistics'], indent=2)}
            
            请考虑以下因素：
            1. 当前市场趋势
            2. 技术指标状态
            3. 波动率水平
            4. 成交量变化
            5. 价格动量
            
            请严格按照以下格式返回标准 JSON，不要加任何注释、解释或 markdown 格式：
            {self.AGENT_OUTPUT_FORMAT['trading_advisor']}
            """
            response = self.llm.call(prompt)
            advice = self._extract_json_from_llm_response(response)
            return advice
        except Exception as e:
            self.logger.error(f"获取交易顾问建议失败: {str(e)}")
            return {}

    def _check_long_entry_conditions(self, current_data: pd.Series, market_indicators: Dict, strategy_advice: Dict) -> bool:
        """检查做多入场条件 - 更灵活的条件判断
        修正：做多信号应为 rsi > rsi_long
        """
        params = self.strategy_params
        rsi = current_data.get('rsi', 50)
        macd = current_data.get('macd', 0)
        signal = current_data.get('signal', 0)
        close_price = current_data.get('close', 0)
        ma5 = current_data.get('ma5', close_price)
        volume_ratio = current_data.get('volume_ratio', 1)
        macd_prev = current_data.get('macd_prev', 0)
        bb_lower = current_data.get('bb_lower', 0)
        
        conditions_met = 0
        total_conditions = 2  # 只统计主条件
        
        # 1. RSI条件 - 放宽条件
        if rsi > params['rsi_long'] * 0.8:
            conditions_met += 1
        else:
            self.logger.info(f"做多失败: RSI={rsi} <= {params['rsi_long']*0.8}")
        
        # 2. MACD条件 - 放宽条件
        if macd > signal or (macd > macd_prev):
            conditions_met += 1
        else:
            self.logger.info(f"做多失败: MACD={macd}, signal={signal}, macd_prev={macd_prev}")
        
        # 3. 价格条件 - 放宽条件
        if close_price > ma5 * 0.98:
            conditions_met += 1
        else:
            self.logger.info(f"做多失败: close={close_price} <= ma5*0.98={ma5*0.98}")
        
        # 4. 成交量条件 - 更宽松
        if volume_ratio > 0.7:
            conditions_met += 1
            total_conditions += 1
        else:
            self.logger.info(f"做多失败: volume_ratio={volume_ratio} <= 0.5")
        
        # 5. 布林带下轨辅助 - 完全可选，不计入总条件
        if bb_lower and close_price < bb_lower * 1.02:
            self.logger.info(f"布林带辅助满足: close={close_price} < bb_lower*1.02={bb_lower*1.02}")
        else:
            if bb_lower:
                self.logger.info(f"布林带辅助未满足: close={close_price} >= bb_lower*1.02={bb_lower*1.02}")
        return conditions_met >= params['min_entry_conditions']

    def _check_short_entry_conditions(self, current_data: pd.Series, market_indicators: Dict, strategy_advice: Dict) -> bool:
        """检查做空入场条件 - 更灵活的条件判断
        修正：做空信号应为 rsi < rsi_short
        """
        params = self.strategy_params
        rsi = current_data.get('rsi', 50)
        macd = current_data.get('macd', 0)
        signal = current_data.get('signal', 0)
        close_price = current_data.get('close', 0)
        ma5 = current_data.get('ma5', close_price)
        volume_ratio = current_data.get('volume_ratio', 1)
        macd_prev = current_data.get('macd_prev', 0)
        bb_upper = current_data.get('bb_upper', float('inf'))
        
        conditions_met = 0
        total_conditions = 4
        # 修正方向：做空信号应为 rsi < rsi_short
        if rsi < params['rsi_short']:
            conditions_met += 1
        if macd < signal - params['macd_trend'] or (macd < macd_prev):
            conditions_met += 1
        if close_price < ma5 * params['ma5_offset_short']:
            conditions_met += 1
        if volume_ratio > params['volume_ratio']:
            conditions_met += 1
        # 可选：布林带上轨辅助
        if bb_upper and close_price > bb_upper * params['bb_upper_offset']:
            conditions_met += 1
            total_conditions += 1
        return conditions_met >= params['min_entry_conditions']

    def _check_long_exit_conditions(self, current_data: pd.Series, market_indicators: Dict, strategy_advice: Dict) -> bool:
        """检查做多出场条件"""
        params = self.strategy_params
        rsi = current_data.get('rsi', 50)
        macd = current_data.get('macd', 0)
        signal = current_data.get('signal', 0)
        close_price = current_data.get('close', 0)
        ma5 = current_data.get('ma5', close_price)
        # RSI过热
        if rsi > params.get('rsi_exit_long', 75):
            return True
        # MACD死叉
        if macd < signal:
            return True
        # 跌破MA5
        if close_price < ma5 * 0.98:
            return True
        return False

    def _check_short_exit_conditions(self, current_data: pd.Series, market_indicators: Dict, strategy_advice: Dict) -> bool:
        """检查做空出场条件"""
        params = self.strategy_params
        rsi = current_data.get('rsi', 50)
        macd = current_data.get('macd', 0)
        signal = current_data.get('signal', 0)
        close_price = current_data.get('close', 0)
        ma5 = current_data.get('ma5', close_price)
        bb_lower = current_data.get('bb_lower', 0)
        # RSI超卖
        if rsi < params.get('rsi_exit_short', 25):
            return True
        # MACD金叉
        if macd > signal:
            return True
        # 突破MA5
        if close_price > ma5 * 1.02:
            return True
        # 接近布林带下轨
        if bb_lower and close_price < bb_lower * params['bb_lower_offset']:
            return True
        return False

    def _backtest_strategy(self, historical_data: pd.DataFrame, signals: pd.DataFrame) -> Dict:
        """回测策略 - 支持完整的多空策略"""
        try:
            self.logger.info("开始回测策略（支持多空）")
            
            trades = []
            position = 0  # 0=无仓位, 1=多头, -1=空头
            entry_price = 0
            entry_time = None
            entry_atr = 0  # 记录开仓时的ATR
            initial_capital = 100000
            
            # 计算真实的ATR
            high_low = historical_data['high'] - historical_data['low']
            high_close = abs(historical_data['high'] - historical_data['close'].shift(1))
            low_close = abs(historical_data['low'] - historical_data['close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            
            # 确保数据长度一致，取较小长度
            min_length = min(len(signals), len(historical_data))
            
            for i in range(min_length):
                # 验证信号有效性
                if i >= len(signals) or i >= len(historical_data):
                    break
                    
                signal = signals.iloc[i]['signal']
                price = historical_data.iloc[i]['close']
                current_time = signals.index[i]
                current_atr = atr.iloc[i] if i < len(atr) and not pd.isna(atr.iloc[i]) else price * 0.02  # 默认2%
                
                # 基本数据验证
                if pd.isna(price) or price <= 0:
                    continue
                
                # 处理交易信号
                if signal == 1:  # 做多信号
                    if position == 0:  # 无仓位 -> 开多仓
                        position = 1
                        entry_price = price
                        entry_time = current_time
                        entry_atr = current_atr
                        self.logger.debug(f"开多仓: {current_time}, 价格: {price}")
                        
                    elif position == -1:  # 有空仓 -> 先平空仓，再开多仓
                        # 平空仓
                        exit_price = price
                        exit_time = current_time
                        # 空头利润计算：入场价格 - 出场价格
                        profit_pct = (entry_price - exit_price) / entry_price
                        
                        trade = {
                            'entry_time': entry_time,
                            'entry_price': entry_price,
                            'exit_time': exit_time,
                            'exit_price': exit_price,
                            'profit_pct': profit_pct,
                            'holding_days': (exit_time - entry_time).days,
                            'atr_at_entry': entry_atr,
                            'position_type': 'short'
                        }
                        trades.append(trade)
                        self.logger.debug(f"平空仓: {current_time}, 价格: {price}, 收益: {profit_pct:.2%}")
                        
                        # 立即开多仓
                        position = 1
                        entry_price = price
                        entry_time = current_time
                        entry_atr = current_atr
                        self.logger.debug(f"开多仓: {current_time}, 价格: {price}")
                        
                elif signal == -1:  # 做空信号
                    if position == 0:  # 无仓位 -> 开空仓
                        position = -1
                        entry_price = price
                        entry_time = current_time
                        entry_atr = current_atr
                        self.logger.debug(f"开空仓: {current_time}, 价格: {price}")
                        
                    elif position == 1:  # 有多仓 -> 先平多仓，再开空仓
                        # 平多仓
                        exit_price = price
                        exit_time = current_time
                        # 多头利润计算：出场价格 - 入场价格
                        profit_pct = (exit_price - entry_price) / entry_price
                        
                        trade = {
                            'entry_time': entry_time,
                            'entry_price': entry_price,
                            'exit_time': exit_time,
                            'exit_price': exit_price,
                            'profit_pct': profit_pct,
                            'holding_days': (exit_time - entry_time).days,
                            'atr_at_entry': entry_atr,
                            'position_type': 'long'
                        }
                        trades.append(trade)
                        self.logger.debug(f"平多仓: {current_time}, 价格: {price}, 收益: {profit_pct:.2%}")
                        
                        # 立即开空仓
                        position = -1
                        entry_price = price
                        entry_time = current_time
                        entry_atr = current_atr
                        self.logger.debug(f"开空仓: {current_time}, 价格: {price}")
                
                elif signal == 0:  # 平仓信号
                    if position == 1:  # 平多仓
                        exit_price = price
                        exit_time = current_time
                        profit_pct = (exit_price - entry_price) / entry_price
                        
                        trade = {
                            'entry_time': entry_time,
                            'entry_price': entry_price,
                            'exit_time': exit_time,
                            'exit_price': exit_price,
                            'profit_pct': profit_pct,
                            'holding_days': (exit_time - entry_time).days,
                            'atr_at_entry': entry_atr,
                            'position_type': 'long'
                        }
                        trades.append(trade)
                        position = 0
                        self.logger.debug(f"平多仓: {current_time}, 价格: {price}, 收益: {profit_pct:.2%}")
                        
                    elif position == -1:  # 平空仓
                        exit_price = price
                        exit_time = current_time
                        profit_pct = (entry_price - exit_price) / entry_price
                        
                        trade = {
                            'entry_time': entry_time,
                            'entry_price': entry_price,
                            'exit_time': exit_time,
                            'exit_price': exit_price,
                            'profit_pct': profit_pct,
                            'holding_days': (exit_time - entry_time).days,
                            'atr_at_entry': entry_atr,
                            'position_type': 'short'
                        }
                        trades.append(trade)
                        position = 0
                        self.logger.debug(f"平空仓: {current_time}, 价格: {price}, 收益: {profit_pct:.2%}")
            
            # 处理最后未平仓的交易
            if position != 0 and len(historical_data) > 0:
                exit_price = historical_data.iloc[-1]['close']
                exit_time = signals.index[-1] if len(signals) > 0 else historical_data.index[-1]
                
                if position == 1:  # 强制平多仓
                    profit_pct = (exit_price - entry_price) / entry_price
                    position_type = 'long'
                elif position == -1:  # 强制平空仓
                    profit_pct = (entry_price - exit_price) / entry_price
                    position_type = 'short'
                
                trade = {
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'profit_pct': profit_pct,
                    'holding_days': (exit_time - entry_time).days,
                    'atr_at_entry': entry_atr,
                    'position_type': position_type
                }
                trades.append(trade)
                self.logger.debug(f"强制平{position_type}仓: {exit_time}, 价格: {exit_price}, 收益: {profit_pct:.2%}")
            
            # 统计多空交易
            long_trades = [t for t in trades if t['position_type'] == 'long']
            short_trades = [t for t in trades if t['position_type'] == 'short']
            
            self.logger.info(f"回测完成，共 {len(trades)} 笔交易（多头: {len(long_trades)}, 空头: {len(short_trades)}）")
            
            return {
                "trades": trades,
                "equity_curve": self._calculate_equity_curve(trades, initial_capital),
                "trade_statistics": self._calculate_trade_statistics(trades)
            }
            
        except Exception as e:
            self.logger.error(f"回测策略失败: {str(e)}")
            return {
                "trades": [],
                "equity_curve": pd.Series([initial_capital]),
                "trade_statistics": {"total_trades": 0, "win_rate": 0, "avg_profit": 0, "max_drawdown": 0}
            }

    def _calculate_equity_curve(self, trades, initial_capital):
        """
        计算权益曲线
        """
        try:
            if not trades:
                return pd.Series([initial_capital])
                
            equity = [initial_capital]
            equity_times = [trades[0]['entry_time']]  # 从第一笔交易开始
            current_equity = initial_capital
            
            for trade in trades:
                # 每笔交易完成后更新权益
                profit_pct = trade['profit_pct']
                current_equity *= (1 + profit_pct)
                equity.append(current_equity)
                equity_times.append(trade['exit_time'])
            
            return pd.Series(equity, index=equity_times)
            
        except Exception as e:
            self.logger.error(f"计算权益曲线失败: {str(e)}")
            return pd.Series([initial_capital])
    
    def _calculate_trade_statistics(self, trades):
        """
        计算交易统计（支持多空分析）
        """
        try:
            if not trades:
                return {
                    "total_trades": 0,
                    "win_rate": 0,
                    "avg_profit": 0,
                    "max_drawdown": 0,
                    "long_trades": 0,
                    "short_trades": 0,
                    "long_win_rate": 0,
                    "short_win_rate": 0,
                    "long_avg_profit": 0,
                    "short_avg_profit": 0
                }
            
            # 分离多空交易
            long_trades = [t for t in trades if t.get('position_type') == 'long']
            short_trades = [t for t in trades if t.get('position_type') == 'short']
            
            # 总体统计
            profits = [trade['profit_pct'] for trade in trades]
            
            # 多头统计
            long_profits = [t['profit_pct'] for t in long_trades] if long_trades else []
            long_win_rate = sum(1 for p in long_profits if p > 0) / len(long_profits) if long_profits else 0
            long_avg_profit = sum(long_profits) / len(long_profits) if long_profits else 0
            
            # 空头统计
            short_profits = [t['profit_pct'] for t in short_trades] if short_trades else []
            short_win_rate = sum(1 for p in short_profits if p > 0) / len(short_profits) if short_profits else 0
            short_avg_profit = sum(short_profits) / len(short_profits) if short_profits else 0
            
            return {
                "total_trades": len(trades),
                "win_rate": sum(1 for p in profits if p > 0) / len(profits) if profits else 0,
                "avg_profit": sum(profits) / len(profits) if profits else 0,
                "max_drawdown": self._calculate_max_drawdown(profits) if profits else 0,
                "long_trades": len(long_trades),
                "short_trades": len(short_trades),
                "long_win_rate": long_win_rate,
                "short_win_rate": short_win_rate,
                "long_avg_profit": long_avg_profit,
                "short_avg_profit": short_avg_profit
            }
            
        except Exception as e:
            self.logger.error(f"计算交易统计失败: {str(e)}")
            return {
                "total_trades": 0,
                "win_rate": 0,
                "avg_profit": 0,
                "max_drawdown": 0,
                "long_trades": 0,
                "short_trades": 0,
                "long_win_rate": 0,
                "short_win_rate": 0,
                "long_avg_profit": 0,
                "short_avg_profit": 0
            }
    
    def _calculate_max_drawdown(self, profits: Union[List[float], pd.Series]) -> float:
        """计算最大回撤"""
        if isinstance(profits, list):
            if not profits:  # 检查列表是否为空
                return 0
            profits = pd.Series(profits)
        elif isinstance(profits, pd.Series):
            if profits.empty:  # 检查Series是否为空
                return 0
        else:
            return 0
            
        cumulative = (1 + profits).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    def _calculate_sharpe_ratio(self, equity_curve):
        """
        计算夏普比率
        """
        if len(equity_curve) < 2:
            return 0
            
        returns = equity_curve.pct_change().dropna()
        if len(returns) == 0 or returns.std() == 0:
            return 0
            
        return np.sqrt(252) * returns.mean() / returns.std()
    
    def _calculate_performance_metrics(self, backtest_results):
        """
        计算性能指标
        """
        trades = backtest_results["trades"]
        equity_curve = backtest_results["equity_curve"]
        stats = backtest_results["trade_statistics"]
        
        # 验证数据
        if len(equity_curve) < 2:
            return {
                "total_return": 0,
                "sharpe_ratio": 0,
                "win_rate": 0,
                "profit_factor": 0
            }
            
        # 计算总收益率
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) > 0 else 0
        
        # 计算夏普比率
        sharpe_ratio = self._calculate_sharpe_ratio(equity_curve)
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "win_rate": stats["win_rate"],
            "profit_factor": self._calculate_profit_factor(trades)
        }
    
    def _calculate_profit_factor(self, trades):
        """
        计算盈亏比 (Profit Factor)
        """
        try:
            if not trades:
                return 0
                
            profits = [trade['profit_pct'] for trade in trades]
            
            if not profits:
                return 0
                
            gross_profit = sum(p for p in profits if p > 0)
            gross_loss = abs(sum(p for p in profits if p < 0))
            
            return gross_profit / gross_loss if gross_loss != 0 else float('inf')
            
        except Exception as e:
            self.logger.error(f"计算盈亏比失败: {str(e)}")
            return 0
    
    def _calculate_strategy_risk_metrics(self, backtest_results):
        """
        计算策略风险指标
        """
        try:
            trades = backtest_results["trades"]
            equity_curve = backtest_results["equity_curve"]
            
            if len(equity_curve) < 2:
                return {
                    "max_drawdown": 0,
                    "var_95": 0,
                    "volatility": 0
                }
            
            returns = equity_curve.pct_change().dropna()
            
            return {
                "max_drawdown": self._calculate_max_drawdown(returns),
                "var_95": self._calculate_var(returns, 0.95),
                "volatility": returns.std() * np.sqrt(252) if len(returns) > 0 else 0  # 年化波动率
            }
            
        except Exception as e:
            self.logger.error(f"计算策略风险指标失败: {str(e)}")
            return {
                "max_drawdown": 0,
                "var_95": 0,
                "volatility": 0
            }
    
    def _calculate_var(self, returns, confidence_level):
        """
        计算风险价值(VaR)
        """
        if len(returns) == 0:
            return 0
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def _calculate_beta(self, equity_curve, benchmark_data):
        """
        计算贝塔系数
        
        Args:
            equity_curve (pd.Series): 策略权益曲线
            benchmark_data (pd.DataFrame): 基准数据
            
        Returns:
            float: 贝塔系数
        """
        try:
            # 确保数据对齐
            strategy_returns = equity_curve.pct_change().dropna()
            benchmark_returns = benchmark_data['close'].pct_change().dropna()
            
            # 对齐数据
            common_index = strategy_returns.index.intersection(benchmark_returns.index)
            strategy_returns = strategy_returns[common_index]
            benchmark_returns = benchmark_returns[common_index]
            
            # 计算协方差和方差
            covariance = strategy_returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            
            # 计算贝塔系数
            beta = covariance / benchmark_variance
            
            return beta
            
        except Exception as e:
            self.logger.error(f"计算贝塔系数失败: {str(e)}")
            return 0
    
    def _calculate_correlation(self, equity_curve, benchmark_data):
        """
        计算相关性
        Args:
            equity_curve (pd.Series): 策略权益曲线
            benchmark_data (pd.DataFrame): 基准数据         
        Returns:
            float: 相关系数
        """
        try:
            # 确保数据对齐
            strategy_returns = equity_curve.pct_change().dropna()
            benchmark_returns = benchmark_data['close'].pct_change().dropna()
            
            # 对齐数据
            common_index = strategy_returns.index.intersection(benchmark_returns.index)
            strategy_returns = strategy_returns[common_index]
            benchmark_returns = benchmark_returns[common_index]
            
            # 计算相关系数
            correlation = strategy_returns.corr(benchmark_returns)
            
            return correlation
            
        except Exception as e:
            self.logger.error(f"计算相关性失败: {str(e)}")
            return 0
    
    def _generate_recommendations(self, backtest_results):
        """
        生成策略建议
        """
        recommendations = []
        # 分析策略表现
        performance = self._calculate_performance_metrics(backtest_results)
        risk = self._calculate_strategy_risk_metrics(backtest_results)
        # 生成建议
        if performance["sharpe_ratio"] < 1:
            recommendations.append("建议优化模型参数以提高风险调整后收益")
        if risk["max_drawdown"] > 0.2:
            recommendations.append("建议加强风险控制措施")
        if performance["win_rate"] < 0.4:
            recommendations.append("建议改进特征工程以提高预测准确率")
        return recommendations
    def _calculate_market_strength(self, data: pd.DataFrame) -> float:
        """计算市场强度
        
        Args:
            data: 市场数据
            
        Returns:
            float: 市场强度值 (0-1)
        """
        try:
            # 确保数据不为空
            if data.empty:
                self.logger.warning("市场数据为空，返回中性市场强度")
                return 0.5
                
            # 获取最新数据
            latest_data = data.iloc[-1]
            
            # 计算RSI的贡献 - 修正归一化范围
            rsi = float(latest_data['rsi'])
            rsi_contribution = (rsi - 30) / (70 - 30)  # 归一化到0-1范围，30-70对应0-1
            rsi_contribution = max(0, min(1, rsi_contribution))  # 限制在0-1范围内
            
            # 计算MACD的贡献 - 使用更稳定的计算方式
            macd = float(latest_data['macd'])
            signal = float(latest_data['signal'])
            if abs(signal) > 0.0001:  # 避免除以接近0的值
                macd_contribution = 0.5 + (macd - signal) / (2 * abs(signal))
            else:
                macd_contribution = 0.5 + np.sign(macd) * 0.5  # 当signal接近0时，直接用macd的符号
            macd_contribution = max(0, min(1, macd_contribution))  # 限制在0-1范围内
            
            # 计算成交量的贡献 - 使用更敏感的计算方式
            volume = float(latest_data['volume'])
            volume_sma = float(latest_data['volume_sma'])
            if volume_sma > 0:
                volume_ratio = volume / volume_sma
                volume_contribution = 0.5 + (volume_ratio - 1) / 4  # 成交量在0.5-1.5倍之间时，贡献在0.375-0.625之间
            else:
                volume_contribution = 0.5
            volume_contribution = max(0, min(1, volume_contribution))  # 限制在0-1范围内
            
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

    def _map_expert_params(self, expert_params, current_atr=None, current_price=None):
        """
        将专家参数映射为策略参数，采用温和映射，避免过拟合。
        """
        mapped = {}
        # 1. RSI阈值映射（温和处理，避免极端）
        if 'trend_threshold' in expert_params:
            mapped['rsi_long'] = max(35, min(50, 50 - expert_params['trend_threshold'] * 5))
            mapped['rsi_short'] = min(70, max(50, 50 + expert_params['trend_threshold'] * 5))
        # 2. 成交量比率
        if 'market_strength_threshold' in expert_params:
            mapped['volume_ratio'] = 1.0 + min(0.5, max(0.0, expert_params['market_strength_threshold'] * 0.3))
        # 3. 均线偏移
        if 'volatility_threshold' in expert_params:
            mapped['ma5_offset'] = 1.0 - min(0.05, max(0.0, expert_params['volatility_threshold']))
            mapped['ma5_offset_short'] = 1.0 + min(0.05, max(0.0, expert_params['volatility_threshold']))
        # 4. 动态止损止盈（温和处理，避免过拟合）
        if current_atr and current_price:
            if 'stop_loss_atr' in expert_params:
                mapped['stop_loss_pct'] = min(0.05, max(0.005, expert_params['stop_loss_atr'] * current_atr / current_price))
            if 'take_profit_atr' in expert_params:
                mapped['take_profit_pct'] = min(0.15, max(0.01, expert_params['take_profit_atr'] * current_atr / current_price))
        # 5. 仓位
        if 'position_size' in expert_params:
            mapped['position_size'] = min(1.0, max(0.01, expert_params['position_size']))
        return mapped

    def generate_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号（支持多空策略，融合多智能体建议）
        Args:
            market_data: 市场数据（已经是完整数据，无缺失值）
        Returns:
            pd.DataFrame: 包含交易信号的数据框
        """
        try:
            self.logger.info(f"开始生成交易信号（支持多空），数据行数: {len(market_data)}")
            # 只初始化signals和持仓等变量，不再获取agent建议
            signals = pd.DataFrame(index=market_data.index)
            signals['signal'] = 0
            position = 0
            entry_price = None
            equity = 1.0
            peak_equity = 1.0
            daily_loss = 0.0
            current_day = None
            trading_paused = False
            for i in range(len(market_data)):
                trading_paused = False
                current_data = market_data.iloc[i]
                current_price = float(current_data['close'])  # 确保转换为float类型
                signal_value = 0
                # 记录日期
                if 'date' in current_data:
                    this_day = current_data['date']
                else:
                    this_day = market_data.index[i]
                # === 新增：计算市场强度 ===
                market_strength = self._calculate_market_strength(market_data.iloc[:i+1])
                self.logger.info(f"market_strength: {market_strength}, position: {position}")  # 打印市场强度
                # === 每根K线都获取agent建议 ===
                advice = self._get_agent_advice(market_data.iloc[:i+1])
                # === 同步专家参数 ===
                expert_params = advice.get('strategy_developer', {}).get('parameters', {})
                if expert_params:
                    if 'atr' in market_data.columns:
                        current_atr = market_data['atr'].iloc[i]
                    else:
                        current_atr = None
                    mapped_params = self._map_expert_params(expert_params, current_atr, current_price)
                    self.set_params(**mapped_params)
                    self.logger.info(f"已自动同步并映射专家参数: {mapped_params}")
                # === 获取风险分析师和交易顾问建议 ===
                risk_advice = advice.get('risk_analyst', {})
                trading_advice = advice.get('trading_advisor', {})
                risk_mgmt = risk_advice.get('risk_management', {})
                max_drawdown = risk_mgmt.get('max_drawdown', 0.2)
                daily_loss_limit = risk_mgmt.get('daily_loss_limit', 0.05)
                risk_level = risk_advice.get('risk_level', 'moderate')
                position_suggestions = risk_advice.get('position_suggestions', {})
                current_signal_advisor = trading_advice.get('current_signal', 'neutral')
                advisor_position_size = trading_advice.get('position_size', None)
                advisor_stop_loss = trading_advice.get('stop_loss', None)
                advisor_take_profit = trading_advice.get('take_profit', None)
                # === 计算市场指标 ===
                # 注意：如果指标依赖于全量数据，仍然用 market_data，否则可用 market_data.iloc[:i+1]
                market_indicators = self.data_processor.calculate_market_indicators(market_data)
                # 2. 自身信号
                strategy_signal = 0
                if position == 0:
                    if market_strength > 0.4:
                        if self._check_long_entry_conditions(current_data, market_indicators, advice.get('strategy_developer', {})):
                            strategy_signal = 1
                    elif market_strength < 0.4:
                        if self._check_short_entry_conditions(current_data, market_indicators, advice.get('strategy_developer', {})):
                            strategy_signal = -1
                # 3. 信号融合（可恢复为多智能体融合）
                if current_signal_advisor == 'buy':
                    advisor_signal = 1
                elif current_signal_advisor == 'sell':
                    advisor_signal = -1
                elif current_signal_advisor == 'hold':
                    advisor_signal = 0
                else:
                    advisor_signal = 0
                if advisor_signal == 0:
                    signal_value = strategy_signal if abs(strategy_signal) == 1 else 0
                elif advisor_signal == strategy_signal and advisor_signal != 0:
                    signal_value = advisor_signal
                elif advisor_signal != 0 and strategy_signal == 0:
                    signal_value = advisor_signal if risk_level != 'high' else 0
                else:
                    signal_value = 0
                # 4. 仓位/止损/止盈融合
                position_size = self.strategy_params.get('position_size', 0.1)
                if advisor_position_size:
                    position_size = advisor_position_size
                elif risk_level == 'high':
                    position_size = min(position_size, 0.05)
                stop_loss_pct = self.strategy_params.get('stop_loss_pct', 0.02)
                take_profit_pct = self.strategy_params.get('take_profit_pct', 0.05)
                if advisor_stop_loss and advisor_stop_loss > 0:
                    stop_loss_pct = abs(advisor_stop_loss / current_price - 1)
                if advisor_take_profit and advisor_take_profit > 0:
                    take_profit_pct = abs(advisor_take_profit / current_price - 1)
                # 5. 全局风控
                if signal_value == 0 and entry_price is not None:
                    profit_pct = float((current_price - entry_price) / entry_price if position == 1 else (entry_price - current_price) / entry_price)
                    equity *= (1 + profit_pct * position_size)
                    if equity > peak_equity:
                        peak_equity = equity
                    drawdown = (peak_equity - equity) / peak_equity
                    if drawdown > max_drawdown:
                        self.logger.warning(f"触发最大回撤风控，强制平仓并暂停交易，当前回撤: {drawdown:.2%}")
                        trading_paused = True
                    if this_day != current_day:
                        daily_loss = 0.0
                        current_day = this_day
                    daily_loss += min(0, profit_pct)
                    if abs(daily_loss) > daily_loss_limit:
                        self.logger.warning(f"触发单日亏损风控，今日不再开新仓，亏损: {daily_loss:.2%}")
                        trading_paused = True
                if trading_paused and position == 0:
                    signals.loc[signals.index[i], 'signal'] = 0
                    continue
                if position == 0:
                    if signal_value == 1:
                        position = 1
                        entry_price = current_price
                    elif signal_value == -1:
                        position = -1
                        entry_price = current_price
                elif position == 1 and entry_price is not None:
                    loss_pct = (current_price - entry_price) / entry_price
                    if loss_pct < -stop_loss_pct or loss_pct > take_profit_pct:
                        signal_value = 0
                        position = 0
                        entry_price = None
                    elif self._check_long_exit_conditions(current_data, market_indicators, advice.get('strategy_developer', {})):
                        signal_value = 0
                        position = 0
                        entry_price = None
                elif position == -1 and entry_price is not None:
                    loss_pct = (entry_price - current_price) / entry_price
                    if loss_pct < -stop_loss_pct or loss_pct > take_profit_pct:
                        signal_value = 0
                        position = 0
                        entry_price = None
                    elif self._check_short_exit_conditions(current_data, market_indicators, advice.get('strategy_developer', {})):
                        signal_value = 0
                        position = 0
                        entry_price = None
                signals.loc[signals.index[i], 'signal'] = signal_value
            # 统计信号
            long_signals = (signals['signal'] == 1).sum()
            short_signals = (signals['signal'] == -1).sum()
            close_signals = (signals['signal'] == 0).sum()
            self.logger.info(f"信号统计 - 做多: {long_signals}, 做空: {short_signals}, 平仓: {close_signals}")
            return signals
        except Exception as e:
            self.logger.error(f"生成交易信号时出错: {str(e)}")
            raise