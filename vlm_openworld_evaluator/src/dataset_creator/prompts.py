import orjson
import random
import logging
from typing import List, Optional, Dict

from src.schemas import PromptConfig

logger = logging.getLogger(__name__)

class PromptManager:
    """Loads VLM prompts from NDJSON files: a global pool plus per-source pools loaded on demand."""

    def __init__(self, global_prompts_path: str):
        self._global_prompt_pool = self._load_prompts_from_file(global_prompts_path)
        self._source_prompt_cache: Dict[str, List[PromptConfig]] = {}
        logger.info(f"Loaded {len(self._global_prompt_pool)} prompts into the global pool from {global_prompts_path}")

    def _load_prompts_from_file(self, path: Optional[str]) -> List[PromptConfig]:
        if not path:
            return []
        
        prompts = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = orjson.loads(line)
                    prompts.append(PromptConfig(**data))
        return prompts

    def get_random_prompt(self, dataset_name: str, source_prompts_path: Optional[str] = None) -> PromptConfig:
        prompt_pool = self._global_prompt_pool

        if source_prompts_path:
            if source_prompts_path not in self._source_prompt_cache:
                logger.info(f"Loading source-specific prompts from: {source_prompts_path}")
                self._source_prompt_cache[source_prompts_path] = self._load_prompts_from_file(source_prompts_path)
            prompt_pool = self._source_prompt_cache[source_prompts_path]

        if not prompt_pool:
            raise ValueError(
                f"The prompt pool is empty. Check the path: "
                f"'{source_prompts_path or 'global_prompts_path'}'"
            )

        valid_prompts = [
            p for p in prompt_pool
            if p.dataset_specific is None or p.dataset_specific == dataset_name
        ]

        if not valid_prompts:
            raise ValueError(
                f"No valid prompts found for dataset '{dataset_name}' in the "
                f"selected prompt pool. Check 'dataset_specific' fields in your prompt file."
            )
            
        return random.choice(valid_prompts)
