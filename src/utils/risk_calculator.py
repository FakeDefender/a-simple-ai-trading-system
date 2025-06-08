import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats

class RiskCalculator:
    def __init__(self, config=None):
        """
        初始化风险计算器
        
        Args:
            config (dict, optional): 配置参数
        """
        self.config = config or {}
        
    def calculate_var(self, data: pd.DataFrame, confidence_level: float = 0.95) -> float:
        """
        计算VaR (Value at Risk)
        """
        returns = data['close'].pct_change().dropna()
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def calculate_expected_shortfall(self, data: pd.DataFrame, confidence_level: float = 0.95) -> float:
        """
        计算期望损失 (Expected Shortfall)
        """
        returns = data['close'].pct_change().dropna()
        var = self.calculate_var(data, confidence_level)
        return returns[returns <= var].mean()
    
    def calculate_beta(self, data: pd.DataFrame, market_data: pd.DataFrame) -> float:
        """
        计算Beta系数
        """
        returns = data['close'].pct_change().dropna()
        market_returns = market_data['close'].pct_change().dropna()
        
        covariance = np.cov(returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        
        return covariance / market_variance
    
    def calculate_correlation(self, data: pd.DataFrame, market_data: pd.DataFrame) -> float:
        """
        计算相关性
        """
        returns = data['close'].pct_change().dropna()
        market_returns = market_data['close'].pct_change().dropna()
        
        return np.corrcoef(returns, market_returns)[0][1]
    
    def calculate_counterparty_risk(self, trade_advice: Dict) -> float:
        """
        计算交易对手风险
        """
        # 基于交易规模、对手方评级等计算
        position_size = trade_advice.get('position_size', 0)
        counterparty_rating = trade_advice.get('counterparty_rating', 'BBB')
        
        rating_risk = {
            'AAA': 0.001,
            'AA': 0.002,
            'A': 0.003,
            'BBB': 0.005,
            'BB': 0.01,
            'B': 0.02,
            'CCC': 0.05
        }
        
        return position_size * rating_risk.get(counterparty_rating, 0.01)
    
    def calculate_settlement_risk(self, trade_advice: Dict) -> float:
        """
        计算结算风险
        """
        # 基于结算周期、交易规模等计算
        settlement_period = trade_advice.get('settlement_period', 2)
        position_size = trade_advice.get('position_size', 0)
        
        return position_size * (settlement_period / 252) * 0.01
    
    def calculate_concentration_risk(self, trade_advice: Dict) -> float:
        """
        计算集中度风险
        """
        # 基于持仓集中度计算
        position_size = trade_advice.get('position_size', 0)
        portfolio_size = trade_advice.get('portfolio_size', 1)
        
        concentration_ratio = position_size / portfolio_size
        return concentration_ratio ** 2  # 使用平方来惩罚高集中度
    
    def calculate_bid_ask_spread(self, market_conditions: Dict) -> float:
        """
        计算买卖价差
        """
        return market_conditions.get('bid_ask_spread', 0.001)
    
    def calculate_market_impact(self, market_conditions: Dict) -> float:
        """
        计算市场冲击
        """
        volume = market_conditions.get('volume', 0)
        trade_size = market_conditions.get('trade_size', 0)
        
        if volume == 0:
            return 0
            
        return (trade_size / volume) ** 0.5 * 0.01
    
    def calculate_liquidity_ratio(self, market_conditions: Dict) -> float:
        """
        计算流动性比率
        """
        volume = market_conditions.get('volume', 0)
        price = market_conditions.get('price', 1)
        trade_size = market_conditions.get('trade_size', 0)
        
        if price == 0:
            return 0
            
        return (volume * price) / (trade_size * price)
    
    def calculate_execution_risk(self, trade_advice: Dict) -> float:
        """
        计算执行风险
        """
        # 基于执行策略、市场条件等计算
        execution_strategy = trade_advice.get('execution_strategy', {})
        market_volatility = trade_advice.get('market_volatility', 0.01)
        
        strategy_risk = {
            'market': 1.0,
            'limit': 0.8,
            'iceberg': 0.6,
            'twap': 0.4,
            'vwap': 0.3
        }
        
        return market_volatility * strategy_risk.get(execution_strategy.get('order_type', 'market'), 1.0)
    
    def calculate_system_risk(self, trade_advice: Dict) -> float:
        """
        计算系统风险
        """
        # 基于系统稳定性、备份机制等计算
        system_stability = trade_advice.get('system_stability', 0.99)
        backup_systems = trade_advice.get('backup_systems', 1)
        
        return (1 - system_stability) * (1 / backup_systems)
    
    def calculate_compliance_risk(self, trade_advice: Dict) -> float:
        """
        计算合规风险
        """
        # 基于合规要求、监管环境等计算
        compliance_score = trade_advice.get('compliance_score', 0.95)
        regulatory_environment = trade_advice.get('regulatory_environment', 'strict')
        
        environment_risk = {
            'strict': 0.8,
            'moderate': 0.5,
            'lenient': 0.2
        }
        
        return (1 - compliance_score) * environment_risk.get(regulatory_environment, 0.5)
    
    def calculate_diversification_risk(self, trade_advice: Dict, portfolio_state: Dict) -> float:
        """
        计算分散化风险
        """
        # 基于投资组合分散度计算
        portfolio_correlation = portfolio_state.get('correlation_matrix', pd.DataFrame())
        if portfolio_correlation.empty:
            return 0
            
        return 1 - portfolio_correlation.mean().mean()
    
    def calculate_portfolio_concentration_risk(self, trade_advice: Dict, portfolio_state: Dict) -> float:
        """
        计算组合集中度风险
        """
        # 基于持仓集中度计算
        position_weights = portfolio_state.get('position_weights', {})
        if not position_weights:
            return 0
            
        return sum(weight ** 2 for weight in position_weights.values())
    
    def calculate_leverage_risk(self, trade_advice: Dict, portfolio_state: Dict) -> float:
        """
        计算杠杆风险
        """
        # 基于杠杆率计算
        leverage_ratio = portfolio_state.get('leverage_ratio', 1.0)
        return max(0, leverage_ratio - 1)
    
    def calculate_total_risk(self, market_risk: Dict, credit_risk: Dict,
                           liquidity_risk: Dict, operational_risk: Dict,
                           portfolio_risk: Dict) -> float:
        """
        计算总风险
        """
        # 使用风险加权方法计算总风险
        weights = {
            'market_risk': 0.4,
            'credit_risk': 0.2,
            'liquidity_risk': 0.2,
            'operational_risk': 0.1,
            'portfolio_risk': 0.1
        }
        
        total_risk = (
            weights['market_risk'] * market_risk['var_95'] +
            weights['credit_risk'] * credit_risk['counterparty_risk'] +
            weights['liquidity_risk'] * liquidity_risk['liquidity_ratio'] +
            weights['operational_risk'] * operational_risk['execution_risk'] +
            weights['portfolio_risk'] * portfolio_risk['concentration_risk']
        )
        
        return total_risk 