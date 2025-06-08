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

    def __init__(self, config: Dict[str, Any], data_loader: DataLoader):
        """
        初始化机器学习策略代理
        
        Args:
            config: 配置字典
            data_loader: 数据加载器
        """
        # 保存配置
        self.config = config
        self.data_loader = data_loader
        self.logger = logging.getLogger(__name__);
        
        # 创建结果保存目录
        self.results_dir = os.path.join('results', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.results_dir, exist_ok=True)
        
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

        # 策略参数
        self.strategy_params = self._initialize_strategy_params()

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
        
        Args:
            backtest_results: 回测结果
            performance: 性能指标
            risk_metrics: 风险指标
            recommendations: 优化建议
        """
        try:
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
        """
        try:
            # 计算技术指标
            logger.info("开始计算技术指标...")
            data = self.data_processor.calculate_technical_indicators(data)
            logger.info("技术指标计算完成")
            
            # 准备市场数据摘要
            market_summary = {
                "latest_price": float(data['close'].iloc[-1]),
                "ma5": float(data['ma5'].iloc[-1]),
                "ma10": float(data['ma10'].iloc[-1]),
                "ma20": float(data['ma20'].iloc[-1]),
                "rsi": float(data['rsi'].iloc[-1]),
                "macd": float(data['macd'].iloc[-1]),
                "volume": float(data['volume'].iloc[-1]),
                "volume_ratio": float(data['volume_ratio'].iloc[-1])
            }
            
            logger.info(f"市场数据摘要: {json.dumps(market_summary, indent=2)}")
            
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
            
    def _extract_json_from_llm_response(self, text: str) -> dict:
        """
        从LLM返回内容中提取JSON部分并解析，支持标准JSON和Python dict格式
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
        """
        try:
            # 准备提示词
            prompt = f"""
            作为策略开发者，请分析以下市场数据并提供交易策略建议：
            
            市场数据：
            {json.dumps(market_summary, indent=2)}
            
            请严格按照以下格式返回标准 JSON，不要加任何注释、解释或 markdown 格式。
            
            策略要求：
            1. 入场条件适中，建议2-3个条件，优先趋势跟随策略
            2. RSI条件：买入时用 "rsi > 45" 且 "rsi < 65"，避免极端值
            3. MACD条件：买入时用 "macd > signal" 或 "macd > -1.0"
            4. 价格趋势：买入时用 "latest_price > ma5" 或 "ma5 > ma20"
            5. 出场条件：用 "rsi > 70" 或 "latest_price < ma10"
            6. 确保是趋势跟随策略，不要逆势交易
            
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
        """
        try:
            prompt = f"""
            作为风险分析师，请评估以下市场数据的风险水平：
            
            市场数据：
            {json.dumps(market_summary, indent=2)}
            
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
        """
        try:
            prompt = f"""
            作为交易顾问，请根据以下市场数据提供具体的交易建议：
            
            市场数据：
            {json.dumps(market_summary, indent=2)}
            
            请严格按照以下格式返回标准 JSON，不要加任何注释、解释或 markdown 格式：
            {self.AGENT_OUTPUT_FORMAT['trading_advisor']}
            """
            response = self.llm.call(prompt)
            advice = self._extract_json_from_llm_response(response)
            return advice
        except Exception as e:
            self.logger.error(f"获取交易顾问建议失败: {str(e)}")
            return {}

    def _check_entry_conditions(self, current_data: pd.Series, market_indicators: Dict, strategy_advice: Dict) -> bool:
        """检查入场条件"""
        if not strategy_advice or 'entry_conditions' not in strategy_advice:
            self.logger.warning("策略建议为空或缺少入场条件")
            return False
        
        # 添加适中的趋势过滤：确保基本上升趋势
        ma5 = float(current_data.get('ma5', 0))
        ma10 = float(current_data.get('ma10', 0))
        ma20 = float(current_data.get('ma20', 0))
        
        # 对于高波动股票(如TSLA)使用更宽松的条件
        current_price = float(current_data.get('close', 0))
        if 'tsla' in str(current_data.name).lower() or current_price > 200:  # 高价股票更宽松
            if ma5 <= ma20:  # 只要求MA5 > MA20
                self.logger.debug("趋势过滤：MA5 <= MA20，跳过买入信号")
                return False
        else:  # 低价股票更严格
            if not (ma5 > ma10 > ma20):  # 完整多头排列
                self.logger.debug("趋势过滤：不满足MA5 > MA10 > MA20，跳过买入信号")
                return False
        
        entry_conditions = strategy_advice['entry_conditions']
        self.logger.info(f"检查入场条件，共 {len(entry_conditions)} 个条件")
        
        satisfied_conditions = 0
        for i, condition in enumerate(entry_conditions):
            result = self._evaluate_condition(condition, current_data, market_indicators)
            if result:
                satisfied_conditions += 1
            self.logger.info(f"入场条件 {i+1}/{len(entry_conditions)}: '{condition}' = {result}")
        
        all_satisfied = satisfied_conditions == len(entry_conditions)
        self.logger.info(f"入场条件满足情况: {satisfied_conditions}/{len(entry_conditions)}, 全部满足: {all_satisfied}")
        
        return all_satisfied

    def _check_exit_conditions(self, current_data: pd.Series, market_indicators: Dict, strategy_advice: Dict) -> bool:
        """检查出场条件"""
        if not strategy_advice:
            return False
            
        for condition in strategy_advice['exit_conditions']:
            if self._evaluate_condition(condition, current_data, market_indicators):
                return True
        return False

    def _check_risk_conditions(self, current_data: pd.Series, market_indicators: Dict, risk_advice: Dict) -> bool:
        """检查风险条件"""
        if not risk_advice:
            return False
            
        return risk_advice['risk_level'] == 'high'

    def _check_trading_conditions(self, current_data: pd.Series, market_indicators: Dict, trading_advice: Dict) -> bool:
        """检查交易条件"""
        if not trading_advice:
            return False
            
        return trading_advice['current_signal'] != 'neutral'

    def _get_trading_signal(self, trading_advice: Dict) -> int:
        """获取交易信号"""
        if not trading_advice:
            return 0
            
        signal_map = {
            'buy': 1,
            'sell': -1,
            'neutral': 0
        }
        return signal_map.get(trading_advice['current_signal'], 0)

    def _evaluate_condition(self, condition: str, current_data: pd.Series, market_indicators: Dict) -> bool:
        """评估条件"""
        try:
            self.logger.info(f"开始评估条件: {condition}")
            
            # 创建安全的评估环境，直接使用已计算的技术指标
            safe_dict = {
                'latest_price': float(current_data['close']),
                'price': float(current_data['close']),
                'open': float(current_data['open']),
                'high': float(current_data['high']),
                'low': float(current_data['low']),
                'volume': float(current_data['volume']),
                'ma5': float(current_data['ma5']) if 'ma5' in current_data else 0.0,
                'MA5': float(current_data['ma5']) if 'ma5' in current_data else 0.0,  # 大写版本
                'ma10': float(current_data['ma10']) if 'ma10' in current_data else 0.0,
                'MA10': float(current_data['ma10']) if 'ma10' in current_data else 0.0,  # 大写版本
                'ma20': float(current_data['ma20']) if 'ma20' in current_data else 0.0,
                'MA20': float(current_data['ma20']) if 'ma20' in current_data else 0.0,  # 大写版本
                'ma50': float(current_data['ma50']) if 'ma50' in current_data else 0.0,
                'MA50': float(current_data['ma50']) if 'ma50' in current_data else 0.0,  # 大写版本
                'rsi': float(current_data['rsi']) if 'rsi' in current_data else 50.0,
                'RSI': float(current_data['rsi']) if 'rsi' in current_data else 50.0,  # 大写版本
                'macd': float(current_data['macd']) if 'macd' in current_data else 0.0,
                'MACD': float(current_data['macd']) if 'macd' in current_data else 0.0,  # 大写版本
                'signal': float(current_data['signal']) if 'signal' in current_data else 0.0,
                'SIGNAL': float(current_data['signal']) if 'signal' in current_data else 0.0,  # 大写版本
                'volume_ratio': float(current_data['volume_ratio']) if 'volume_ratio' in current_data else 1.0,
                'trend_strength': float(market_indicators.get('trend_strength', 0.0)),
                'market_strength': float(market_indicators.get('market_strength', 0.5)),
                'market_strength_threshold': 40.0,  # 调整阈值从30.0到40.0，增加信号生成概率
                'volatility': float(market_indicators.get('volatility', 0.0))
            }
            
            self.logger.info(f"评估环境变量: {json.dumps({k: v for k, v in safe_dict.items()}, ensure_ascii=False, indent=2)}")
            
            # 使用简单的条件评估
            result = self._simple_eval(condition, safe_dict)
            self.logger.info(f"条件评估结果: {result}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"条件评估失败: {str(e)}")
            return False

    def _simple_eval(self, condition: str, safe_dict: Dict) -> bool:
        """简单的条件评估方法，支持基本的比较表达式"""
        try:
            self.logger.info(f"开始简单评估条件: {condition}")
            
            # 替换变量名为实际值，按变量名长度降序排列，避免部分替换冲突
            eval_condition = condition
            # 按变量名长度降序排序，确保长变量名优先替换
            sorted_vars = sorted(safe_dict.items(), key=lambda x: len(x[0]), reverse=True)
            for var_name, var_value in sorted_vars:
                eval_condition = eval_condition.replace(var_name, str(var_value))
            
            self.logger.info(f"替换后的条件: {eval_condition}")
            
            # 使用安全的 eval，只允许基本的数学和比较运算
            allowed_names = {"__builtins__": {}}
            result = eval(eval_condition, allowed_names)
            
            return bool(result)
            
        except Exception as e:
            self.logger.error(f"简单评估失败: {str(e)}")
            return False

    def _backtest_strategy(self, historical_data: pd.DataFrame, signals: pd.DataFrame) -> Dict:
        """回测策略 - 简化版本，只处理多头策略"""
        try:
            self.logger.info("开始回测策略")
            
            trades = []
            position = 0  # 0=无仓位, 1=有仓位
            entry_price = 0
            entry_time = None
            initial_capital = 100000
            
            # 计算真实的ATR
            high_low = historical_data['high'] - historical_data['low']
            high_close = abs(historical_data['high'] - historical_data['close'].shift(1))
            low_close = abs(historical_data['low'] - historical_data['close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            
            for i in range(len(signals)):
                signal = signals.iloc[i]['signal']
                price = historical_data.iloc[i]['close']
                current_time = signals.index[i]
                current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else price * 0.02  # 默认2%
                
                if signal == 1 and position == 0:  # 买入信号且无仓位
                    position = 1
                    entry_price = price
                    entry_time = current_time
                    self.logger.debug(f"开仓: {current_time}, 价格: {price}")
                    
                elif signal == -1 and position == 1:  # 卖出信号且有仓位
                    # 平仓
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
                        'atr_at_entry': current_atr
                    }
                    trades.append(trade)
                    
                    position = 0
                    self.logger.debug(f"平仓: {current_time}, 价格: {price}, 收益: {profit_pct:.2%}")
            
            # 处理最后未平仓的交易
            if position == 1:
                exit_price = historical_data.iloc[-1]['close']
                exit_time = signals.index[-1]
                profit_pct = (exit_price - entry_price) / entry_price
                
                trade = {
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'profit_pct': profit_pct,
                    'holding_days': (exit_time - entry_time).days,
                    'atr_at_entry': atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else exit_price * 0.02
                }
                trades.append(trade)
                self.logger.debug(f"强制平仓: {exit_time}, 价格: {exit_price}, 收益: {profit_pct:.2%}")
            
            self.logger.info(f"回测完成，共 {len(trades)} 笔交易")
            
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
        计算交易统计
        """
        try:
            if not trades:
                return {
                    "total_trades": 0,
                    "win_rate": 0,
                    "avg_profit": 0,
                    "max_drawdown": 0
                }
                
            profits = [trade['profit_pct'] for trade in trades]
            
            return {
                "total_trades": len(profits),
                "win_rate": sum(1 for p in profits if p > 0) / len(profits) if profits else 0,
                "avg_profit": sum(profits) / len(profits) if profits else 0,
                "max_drawdown": self._calculate_max_drawdown(profits) if profits else 0
            }
            
        except Exception as e:
            self.logger.error(f"计算交易统计失败: {str(e)}")
            return {
                "total_trades": 0,
                "win_rate": 0,
                "avg_profit": 0,
                "max_drawdown": 0
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
            
            # 计算RSI的贡献
            rsi = float(latest_data['rsi'])
            rsi_contribution = (rsi - 30) / 40  # 归一化到0-1范围
            rsi_contribution = max(0, min(1, rsi_contribution))  # 限制在0-1范围内
            
            # 计算MACD的贡献
            macd = float(latest_data['macd'])
            signal = float(latest_data['signal'])
            macd_contribution = 0.5 + (macd - signal) / (2 * abs(signal)) if signal != 0 else 0.5
            macd_contribution = max(0, min(1, macd_contribution))  # 限制在0-1范围内
            
            # 计算成交量的贡献
            volume = float(latest_data['volume'])
            volume_sma = float(latest_data['volume_sma'])
            volume_contribution = min(volume / volume_sma, 2) / 2  # 归一化到0-1范围
            
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

    def generate_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            market_data: 市场数据（已经是完整数据，无缺失值）
            
        Returns:
            pd.DataFrame: 包含交易信号的数据框
        """
        try:
            self.logger.info(f"开始生成交易信号，数据行数: {len(market_data)}")
            
            # 1. 获取智能体建议
            advice = self._get_agent_advice(market_data)
            
            # 2. 计算市场指标
            market_indicators = self.data_processor.calculate_market_indicators(market_data)
            
            # 3. 生成信号
            signals = pd.DataFrame(index=market_data.index)
            signals['signal'] = 0  # 0表示无信号，1表示买入，-1表示卖出
            
            # 4. 持仓状态管理
            position = 0  # 0=无仓位, 1=多头
            entry_price = None  # 入场价格，None表示无仓位
            
            for i in range(len(market_data)):
                current_data = market_data.iloc[i]
                current_price = current_data['close']
                signal_value = 0
                
                if position == 0:  # 无仓位时检查入场条件
                    if self._check_entry_conditions(current_data, market_indicators, advice['strategy_developer']):
                        signal_value = 1  # 买入信号
                        position = 1  # 更新持仓状态
                        entry_price = current_price  # 记录入场价格
                        self.logger.debug(f"日期 {current_data.name}: 生成买入信号，价格: {entry_price}")
                        
                elif position == 1 and entry_price is not None:  # 有多头仓位时检查出场条件
                    # 检查止损：亏损超过2%（更严格的止损）
                    loss_pct = (current_price - entry_price) / entry_price
                    if loss_pct < -0.02:
                        signal_value = -1  # 止损卖出
                        position = 0
                        entry_price = None  # 清除入场价格
                        self.logger.debug(f"日期 {current_data.name}: 触发止损，价格: {current_price}, 亏损: {loss_pct:.2%}")
                    # 检查RSI过热出场：RSI > 75 立即出场
                    elif current_data.get('rsi', 50) > 75:
                        signal_value = -1  # RSI过热卖出
                        position = 0
                        profit_pct = (current_price - entry_price) / entry_price
                        entry_price = None
                        self.logger.debug(f"日期 {current_data.name}: RSI过热出场，价格: {current_price}, 收益: {profit_pct:.2%}")
                    # 检查正常出场条件
                    elif self._check_exit_conditions(current_data, market_indicators, advice['strategy_developer']):
                        signal_value = -1  # 卖出信号
                        position = 0  # 更新持仓状态
                        profit_pct = (current_price - entry_price) / entry_price
                        entry_price = None  # 清除入场价格
                        self.logger.debug(f"日期 {current_data.name}: 生成卖出信号，价格: {current_price}, 收益: {profit_pct:.2%}")
                
                signals.iloc[i]['signal'] = signal_value
            
            # 5. 统计信号
            buy_signals = (signals['signal'] == 1).sum()
            sell_signals = (signals['signal'] == -1).sum()
            hold_signals = (signals['signal'] == 0).sum()
            
            self.logger.info(f"信号统计 - 买入: {buy_signals}, 卖出: {sell_signals}, 持有: {hold_signals}")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"生成交易信号时出错: {str(e)}")
            raise