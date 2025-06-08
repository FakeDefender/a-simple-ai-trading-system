"""
AI量化交易系统
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导出主要模块
from src.agents.ml_strategy_agent import MLStrategyAgent
from src.utils.config_loader import load_config
from src.utils.data_loader import DataLoader
from src.utils.data_processor import DataProcessor
from src.utils.risk_calculator import RiskCalculator

__version__ = "0.1.0"

# 空文件，用于标记目录为Python包 