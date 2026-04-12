from .base import LLMClientABC
from .gemini_client import GeminiClient
from .vllm_client import VLLMClient
from .vllm_single_client import VLLMSingleClient
from src.core.config import ModelConfig
from src.core.exceptions import ConfigurationError

def create_client_instance(config: ModelConfig) -> LLMClientABC:
    if config.type == "gemini":
        return GeminiClient(
            api_key=config.api_key,
            model=config.model_name,
            enable_thinking=config.thinking_enabled,
            timeout_seconds=config.timeout_seconds
        )
    elif config.type == "vllm":
        return VLLMClient(
            model=config.model_name,
            api_base=config.api_base,
            enable_thinking=config.thinking_enabled,
            timeout_seconds=config.timeout_seconds
        )
    elif config.type == "vllm_single":
        return VLLMSingleClient(
            model=config.model_name,
            api_base=config.api_base,
            enable_thinking=config.thinking_enabled,
            timeout_seconds=config.timeout_seconds
        )
    else:
        raise ConfigurationError(f"Unknown client type '{config.type}'. Supported types: gemini, vllm, vllm_single")
