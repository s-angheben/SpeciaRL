import asyncio
import time
import uuid
from typing import List, Dict, Tuple
from openai import AsyncOpenAI, APIError

from .base import LLMClientABC
from src.core.exceptions import ServiceUnavailableError, LLMResponseParseError
from src.core.logging import get_logger

class VLLMSingleClient(LLMClientABC):
    """
    vLLM client that makes individual API calls for each ground_truth/prediction pair.
    
    Uses vLLM's 'guided_choice' for efficient, direct classification, eliminating
    the need for JSON parsing. It also supports toggling reasoning features
    via the 'enable_thinking' parameter for compatible models.
    """
    
    
    def __init__(self, model: str, api_base: str, enable_thinking: bool = True, timeout_seconds: int = 600):
        """
        Initializes the VLLMSingleClient.

        Args:
            model (str): The name of the model to use (e.g., 'meta-llama/Llama-3.1-8B-Instruct').
            api_base (str): The base URL of the vLLM server (e.g., 'http://localhost:8000').
            enable_thinking (bool): Flag to enable reasoning features in compatible models.
                                    For IBM Granite, set to True to enable.
                                    For Qwen3, it's on by default; set to False to disable.
            timeout_seconds (int): The timeout for API calls in seconds.
        """
        self.model_name = model
        self.api_base = api_base.rstrip('/') + '/v1'
        self.enable_thinking = enable_thinking
        self.timeout_seconds = timeout_seconds
        self.logger = get_logger(__name__)
        
        self.client = AsyncOpenAI(
            base_url=self.api_base,
            api_key="dummy",  # vLLM server does not require an API key
            timeout=self.timeout_seconds
        )

    async def initialize(self):
        """
        Initializes and validates the connection to the vLLM server by fetching available models.
        
        Raises:
            ServiceUnavailableError: If the client cannot connect to the vLLM server.
        """
        init_start = time.time()
        try:
            await self.client.models.list()
            init_duration = (time.time() - init_start) * 1000
            self.logger.info(
                "VLLMSingleClient initialized successfully",
                model=self.model_name,
                api_base=self.api_base,
                initialization_time_ms=round(init_duration, 2)
            )
        except Exception as e:
            init_duration = (time.time() - init_start) * 1000
            self.logger.error(
                "Failed to connect to vLLM server",
                model=self.model_name,
                api_base=self.api_base,
                error=str(e),
                initialization_time_ms=round(init_duration, 2)
            )
            raise ServiceUnavailableError(f"Cannot connect to vLLM server at {self.api_base}: {e}") from e

    def get_name(self) -> str:
        return self.model_name

    def get_client_type(self) -> str:
        return "vllm_single"

    def get_thinking_enabled(self) -> bool:
        return self.enable_thinking

    async def get_classifications(
        self, 
        ground_truths: List[str], 
        predictions: List[str], 
        prompt_template: Dict[str, str],
        categories: List[str]
    ) -> Dict[Tuple[str, str], str]:
        """
        Processes each ground_truth/prediction pair as an individual, concurrent API call.

        Args:
            ground_truths (List[str]): A list of ground truth strings.
            predictions (List[str]): A list of prediction strings.
            prompt_template (Dict[str, str]): A dictionary containing the 'system_prompt'.

        Returns:
            A dictionary mapping each (ground_truth, prediction) pair to its classification string.
            Failed classifications are marked as 'error'.
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        system_prompt = prompt_template.get("system_prompt", "")
        pairs = list(zip(ground_truths, predictions))
        
        self.logger.debug("Starting individual classifications",
                        request_id=request_id,
                        item_count=len(ground_truths),
                        sample_ground_truths=ground_truths[:2] if len(ground_truths) > 2 else ground_truths,
                        sample_predictions=predictions[:2] if len(predictions) > 2 else predictions,
                        thinking_enabled=self.enable_thinking)
        
        tasks = [self._classify_single_pair(gt, pred, system_prompt, prompt_template, request_id, categories) for gt, pred in pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        self.logger.debug("Processing individual results",
                        request_id=request_id,
                        total_tasks=len(tasks))
        
        result_map = {}
        success_count = 0
        for i, ((gt, pred), result) in enumerate(zip(pairs, results)):
            if isinstance(result, Exception):
                self.logger.error(
                    "Single pair classification failed with an exception",
                    request_id=request_id,
                    item_index=i,
                    ground_truth=gt,
                    prediction=pred, 
                    error=str(result),
                    error_type=type(result).__name__
                )
                raise LLMResponseParseError(f"Classification failed for pair ('{gt}', '{pred}'): {str(result)}")
            else:
                result_map[(gt, pred)] = result
                success_count += 1
                    
                self.logger.debug("Processed single pair",
                                request_id=request_id,
                                item_index=i,
                                ground_truth=gt,
                                prediction=pred,
                                classification=result)
        
        duration_ms = round((time.time() - start_time) * 1000, 2)
        success_rate = round((success_count / len(pairs) * 100), 2) if pairs else 0
        
        self.logger.info(
            "Individual classifications completed",
            request_id=request_id,
            model=self.model_name,
            total_processing_time_ms=duration_ms,
            result_count=len(result_map),
            success_rate=success_rate,
            thinking_enabled=self.enable_thinking
        )
        
        return result_map

    async def _classify_single_pair(self, ground_truth: str, prediction: str, system_prompt: str, prompt_template: Dict[str, str], request_id: str, categories: List[str]) -> str:
        """
        Classifies a single ground_truth/prediction pair using vLLM's 'guided_choice'.

        Args:
            ground_truth (str): The ground truth string.
            prediction (str): The prediction string.
            system_prompt (str): The system prompt to guide the model.
            prompt_template (Dict[str, str]): The prompt template containing formatting.
            request_id (str): The request ID for logging tracking.
            categories (List[str]): The list of valid classification categories.

        Returns:
            The classification string from categories.
        
        Raises:
            APIError: Propagates exceptions from the OpenAI client for the caller to handle.
        """
        user_content = self.build_user_content([ground_truth], [prediction], prompt_template)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        
        try:
            self.logger.debug("Starting single API call",
                            request_id=request_id,
                            ground_truth=ground_truth,
                            prediction=prediction,
                            thinking_enabled=self.enable_thinking)
            
            api_start_time = time.time()
            completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                extra_body={
                    "guided_choice": categories,
                    # Pass kwargs to the chat template to control model features
                    "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
                }
            )
            api_duration = time.time() - api_start_time
            api_duration_ms = round(api_duration * 1000, 2)
            
            classification = completion.choices[0].message.content.strip()
            
            self.logger.debug("Single API call completed",
                            request_id=request_id,
                            ground_truth=ground_truth,
                            prediction=prediction,
                            api_call_duration_ms=api_duration_ms,
                            classification=classification)

            if classification in categories:
                return classification
            
            # This case is unlikely with guided_choice but serves as a safeguard.
            self.logger.warning(
                "Invalid classification received from model despite guided choice",
                request_id=request_id,
                response=classification,
                expected_choices=categories,
                ground_truth=ground_truth,
                prediction=prediction
            )
            raise LLMResponseParseError(f"Invalid classification '{classification}' received despite guided choice. Expected one of: {categories}")
                
        except APIError as e:
            api_duration = time.time() - api_start_time if 'api_start_time' in locals() else 0
            api_duration_ms = round(api_duration * 1000, 2)
            
            self.logger.error(
                "API call failed for a single pair",
                request_id=request_id,
                ground_truth=ground_truth,
                prediction=prediction,
                api_call_duration_ms=api_duration_ms,
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def check_health(self) -> bool:
        """
        Checks if the vLLM server is healthy and responsive.

        Returns:
            True if the server responds successfully, False otherwise.
        """
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False

    async def close_client(self):
        await self.client.close()