import orjson
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from src.core.exceptions import LLMResponseParseError, IncompleteLLMResponseError

class LLMClientABC(ABC):
    @abstractmethod
    def get_client_type(self) -> str:
        """Return client type: 'gemini', 'vllm', or 'vllm_single'"""
        pass

    @abstractmethod
    async def initialize(self):
        pass

    @abstractmethod
    async def get_classifications(self, ground_truths: List[str], predictions: List[str], prompt_template: Dict[str, str], categories: List[str]) -> Dict[Tuple[str, str], str]:
        pass

    @abstractmethod
    async def close_client(self):
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    async def check_health(self) -> bool:
        pass

    @abstractmethod
    def get_thinking_enabled(self) -> bool:
        pass

    def build_user_content(self, ground_truths: List[str], predictions: List[str], prompt_template: Dict[str, str]) -> str:
        ndjson_input = "\n".join([
            orjson.dumps({"ground_truth": gt, "prediction": pred}).decode('utf-8')
            for gt, pred in zip(ground_truths, predictions)
        ])
        
        return prompt_template["prompt"] % ndjson_input


    