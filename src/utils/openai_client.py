import os
import logging
from openai import OpenAI
from typing import Dict, Any

logger = logging.getLogger(__name__)

class OpenAIClient:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化客户端
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.model_type = config.get('model_type', 'openai')
        
        # 从api_keys中获取模型配置
        self.model_config = config.get('api_keys', {}).get(self.model_type, {})
        
        # 优先使用环境变量中的API密钥
        env_api_key = os.environ.get(f"{self.model_type.upper()}_API_KEY")
        self.api_key = env_api_key if env_api_key else self.model_config.get('api_key')
        
        self.model = self.model_config.get('model')
        self.temperature = self.model_config.get('temperature', 0.7)
        self.max_tokens = self.model_config.get('max_tokens', 2000)
        
        # 检查API密钥
        if not self.api_key:
            raise ValueError(f"未找到{self.model_type}的API密钥配置，请设置环境变量{self.model_type.upper()}_API_KEY或在配置文件中提供")
            
        logger.info(f"环境变量 {self.model_type.upper()}_API_KEY: {'已设置' if env_api_key else '未设置'}")
        
        # 创建客户端
        self.client = OpenAI(
            api_key=self.api_key,
            timeout=30.0
        )
        
        logger.info(f"使用的API密钥: {'已设置' if self.api_key else '未设置'}")
        
        # 记录创建LLM实例的参数值
        logger.info("创建 LLM 实例的参数值：")
        logger.info(f"model: {self.model}")
        logger.info(f"temperature: {self.temperature}")
        logger.info(f"max_tokens: {self.max_tokens}")
        logger.info(f"api_key: {'已设置' if self.api_key else '未设置'}")
        
    def get_llm(self):
        """
        获取LLM实例
        """
        try:
            from crewai.llm import LLM
            
            # 创建LLM实例
            llm = LLM(
                model='openai/'+self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.api_key,
                base_url="https://api.deepseek.com/v1"
            )
            
            return llm
        except Exception as e:
            logger.error(f"创建LLM实例失败: {str(e)}")
            raise 