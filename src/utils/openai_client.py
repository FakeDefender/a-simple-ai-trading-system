import logging
import os
from typing import Any, Dict, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


logger = logging.getLogger(__name__)


class OpenAIClient:
    """可选的 LLM 客户端封装。

    v0.1 默认离线运行，因此这里不再在缺失依赖或缺失密钥时直接抛异常。
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.llm_config = self.config.get("llm", {})
        self.model_type = self.llm_config.get("provider", self.config.get("model_type", "deepseek"))
        self.model_config = self.config.get("api_keys", {}).get(self.model_type, {})
        self.api_key = os.getenv(f"{self.model_type.upper()}_API_KEY") or self.model_config.get("api_key")
        self.model = self.llm_config.get("model") or self.model_config.get("model")
        self.base_url = self.llm_config.get("base_url") or self.model_config.get("base_url")
        self.timeout = float(self.llm_config.get("timeout", 120.0))
        self.max_retries = int(self.llm_config.get("max_retries", 0))
        self.default_thinking = str(self.llm_config.get("thinking", "disabled")).lower()
        self.reasoning_effort = self.llm_config.get("reasoning_effort")
        self._client: Optional[OpenAI] = None

        if not self.llm_config.get("enabled", False):
            logger.info("LLM 已关闭，使用离线策略模式")
            return

        if OpenAI is None:
            logger.warning("未安装 openai 包，LLM 功能不可用")
            return

        if not self.api_key:
            logger.warning("未配置 API key，LLM 功能不可用")
            return

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def is_available(self) -> bool:
        return self._client is not None and bool(self.model)

    def get_llm(self) -> Optional[OpenAI]:
        return self._client

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        model: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        thinking: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Optional[str]:
        if not self.is_available():
            return None

        request_options: Dict[str, Any] = {}
        if timeout is not None:
            request_options["timeout"] = float(timeout)
        if response_format is not None:
            request_options["response_format"] = response_format

        thinking_mode = str(thinking or self.default_thinking or "disabled").lower()
        if self.model_type == "deepseek" and thinking_mode in {"enabled", "disabled"}:
            request_options["extra_body"] = {"thinking": {"type": thinking_mode}}
            selected_effort = reasoning_effort or self.reasoning_effort
            if thinking_mode == "enabled" and selected_effort:
                request_options["reasoning_effort"] = selected_effort

        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": int(max_tokens if max_tokens is not None else self.llm_config.get("max_tokens", 1200)),
            **request_options,
        }
        if not (self.model_type == "deepseek" and thinking_mode == "enabled"):
            payload["temperature"] = float(self.llm_config.get("temperature", 0.2))

        response = self._client.chat.completions.create(**payload)
        return response.choices[0].message.content if response.choices else None
