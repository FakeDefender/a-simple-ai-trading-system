import os
import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """加载配置文件
    
    Returns:
        dict: 配置参数字典
    """
    try:
        # 获取项目根目录
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 构建配置文件路径
        api_keys_path = os.path.join(root_dir, 'src', 'config', 'api_keys.yaml')
        config_path = os.path.join(root_dir, 'src', 'config', 'config.yaml')
        
        # 读取API密钥配置
        with open(api_keys_path, 'r', encoding='utf-8') as f:
            api_keys = yaml.safe_load(f)
            
        # 读取主配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 获取模型类型
        model_type = api_keys.get('model_type', 'deepseek')
        config['model_type'] = model_type
        
        # 从环境变量读取API密钥
        api_key = os.getenv(f'{model_type.upper()}_API_KEY')
        if api_key:
            # 使用配置文件中的模型设置
            model_config = api_keys.get(model_type, {})
            model_config['api_key'] = api_key
            config['api_keys'] = {model_type: model_config}
            config['model'] = model_config.get('model')
        
        return config
        
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        # 返回默认配置
        return {
            'data': {
                'symbol': 'aapl.us',
                'interval': 'd'
            },
            'strategy': {
                'rsi_period': 14,
                'ma_periods': [5, 10, 20]
            },
            'risk': {
                'max_drawdown': 0.15,
                'var_confidence': 0.95
            },
            'api_keys': {},
            'model_type': 'deepseek',
            'model': 'deepseek-chat'
        } 