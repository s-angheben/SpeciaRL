import os
import yaml
from typing import List, Dict, Any, Union
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    type: str
    model_name: str
    api_key: str = None
    api_base: str = None
    thinking_enabled: bool = True
    dtype: str = None
    timeout_seconds: int = 7200
    
    def __post_init__(self):
        if self.type == "gemini" and not self.api_key:
            raise ValueError(f"Model '{self.name}' of type 'gemini' requires 'api_key'")
        elif self.type in ["vllm", "vllm_single"] and not self.api_base:
            raise ValueError(f"Model '{self.name}' of type '{self.type}' requires 'api_base'")
        elif self.type not in ["gemini", "vllm", "vllm_single"]:
            raise ValueError(f"Model '{self.name}' has unsupported type '{self.type}'. Supported types: gemini, vllm, vllm_single")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        return cls(
            name=config_dict["name"],
            type=config_dict["type"],
            model_name=config_dict.get("model_name") or config_dict.get("model_path"),
            api_key=config_dict.get("api_key"),
            api_base=config_dict.get("api_base"),
            thinking_enabled=config_dict.get("thinking_enabled", True),
            dtype=config_dict.get("dtype")
        )

class Settings:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.getenv("CONFIG_PATH", "config.yml")
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file '{config_path}' not found. "
                "Please provide a config.yml (see scripts/eval/config.apptainer.yml for a working example)."
            )
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file '{config_path}': {e}")

        self.REDIS_URL: str = os.getenv("REDIS_URL") or config_data.get("redis_url", "redis://localhost:6379")
        self.DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL") or config_data.get("default_llm_model")
        self.DEFAULT_VERIFIER_PROMPT: str = os.getenv("DEFAULT_VERIFIER_PROMPT") or config_data.get("default_verifier_prompt")

        logging_config = config_data.get("logging", {})
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL") or logging_config.get("level", "INFO")
        self.LOG_FORMAT: str = os.getenv("LOG_FORMAT") or logging_config.get("format", "json")
        self.LOG_INCLUDE_STDLIB: bool = os.getenv("LOG_INCLUDE_STDLIB", "").lower() == "true" or logging_config.get("include_stdlib", True)
        
        file_logging_config = logging_config.get("file_logging", {})
        self.LOG_FILE_ENABLED: bool = file_logging_config.get("enabled", False)
        self.LOG_ERROR_FILE: str = file_logging_config.get("error_file", "logs/errors.log")
        self.LOG_MAX_SIZE: str = file_logging_config.get("max_size", "50MB")
        self.LOG_ROTATE_COUNT: int = file_logging_config.get("rotate_count", 3)
        
        batching_config = config_data.get("batching", {})
        self.MAX_BATCH_PROMPT_SIZE: int = batching_config.get("max_batch_prompt_size", 16)
        self.MAX_CONCURRENT_GEMINI: int = batching_config.get("max_concurrent_gemini", 1)
        
        mongodb_config = config_data.get("mongodb", {})
        self.MONGODB_ENABLED: bool = os.getenv("MONGODB_ENABLED", "").lower() == "true" or mongodb_config.get("enabled", False)
        self.MONGODB_URL: str = os.getenv("MONGODB_URL") or mongodb_config.get("url", "mongodb://localhost:27017")
        self.MONGODB_DATABASE: str = mongodb_config.get("database", "llm_caching")
        self.MONGODB_COLLECTION: str = mongodb_config.get("collection", "classification_logs")
        self.MONGODB_MAX_OCCURRENCES: int = mongodb_config.get("max_occurrences", 1000)
        self.MONGODB_WORKER_COUNT: int = mongodb_config.get("worker_count", 2)
        
        models_data = config_data.get("models", [])
        self.MODELS_CONFIG: List[ModelConfig] = [ModelConfig.from_dict(model) for model in models_data]

        if not self.DEFAULT_LLM_MODEL or not self.MODELS_CONFIG:
            raise ValueError("Missing critical configuration. 'default_llm_model' and 'models' must be set in config.yml")
        
        available_models = [model.name for model in self.MODELS_CONFIG]
        if self.DEFAULT_LLM_MODEL not in available_models:
            raise ValueError(f"Default model '{self.DEFAULT_LLM_MODEL}' not found in models list. Available models: {available_models}")
        
        if not self.DEFAULT_VERIFIER_PROMPT:
            raise ValueError("Missing 'default_verifier_prompt' in config.yml")

settings = Settings()
