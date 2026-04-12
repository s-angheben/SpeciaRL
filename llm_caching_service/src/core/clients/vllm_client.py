import orjson
import time
import uuid
from typing import List, Dict, Tuple
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from .base import LLMClientABC
from src.core.exceptions import RateLimitError, ServiceUnavailableError, LLMResponseParseError
from src.core.logging import get_logger

class VLLMClient(LLMClientABC):
    """vLLM client using structured outputs with guided JSON for reliable classification processing.
    
    Supports reasoning models like Qwen3 with thinking capabilities. By default, thinking is enabled
    for reasoning models. Set enable_thinking=False to disable reasoning for Qwen3 models.
    """
    
    
    def __init__(self, model: str, api_base: str, enable_thinking: bool = True, timeout_seconds: int = 600):
        self.model_name = model
        self.api_base = api_base
        self.enable_thinking = enable_thinking
        self.timeout_seconds = timeout_seconds
        self.logger = get_logger(__name__)
        
        base_url = self.api_base.rstrip('/') + '/v1'
        self.client = AsyncOpenAI(base_url=base_url, api_key="dummy", timeout=timeout_seconds)

    @retry(
        wait=wait_fixed(5),
        stop=stop_after_attempt(4),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def initialize(self):
        init_start = time.time()
        try:
            base_url = self.api_base.rstrip('/') + '/v1'
            self.logger.debug("Initializing VLLMClient", 
                            api_base=self.api_base, 
                            base_url=base_url,
                            model=self.model_name)
            models = await self.client.models.list()
            
            available_model_ids = [model.id for model in models.data]
            self.logger.debug("Model validation",
                            available_models=available_model_ids,
                            requested_model=self.model_name,
                            model_match=self.model_name in available_model_ids)
            
            init_duration = (time.time() - init_start) * 1000
            self.logger.info("VLLMClient initialized successfully",
                           model=self.model_name,
                           api_base=self.api_base,
                           initialization_time_ms=round(init_duration, 2))
        except Exception as e:
            init_duration = (time.time() - init_start) * 1000
            self.logger.error("Failed to connect to vLLM server",
                            api_base=self.api_base,
                            model=self.model_name,
                            error=str(e),
                            initialization_time_ms=round(init_duration, 2))
            raise ServiceUnavailableError(f"vLLM server not available: {e}") from e

    def get_name(self) -> str:
        return self.model_name
    
    def get_client_type(self) -> str:
        return "vllm"

    def get_thinking_enabled(self) -> bool:
        return self.enable_thinking


    async def get_classifications(self, ground_truths: List[str], predictions: List[str], prompt_template: Dict[str, str], categories: List[str]) -> Dict[Tuple[str, str], str]:
        """Process all ground_truth/prediction pairs in a single batch request using guided JSON."""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        self.logger.debug("Starting classification request",
                        request_id=request_id,
                        item_count=len(ground_truths),
                        sample_ground_truths=ground_truths[:2] if len(ground_truths) > 2 else ground_truths,
                        sample_predictions=predictions[:2] if len(predictions) > 2 else predictions)
        
        system_prompt = prompt_template.get("system_prompt", "")
        self.logger.debug("Prompt preparation",
                        request_id=request_id,
                        system_prompt_length=len(system_prompt))
        
        user_content = self.build_user_content(ground_truths, predictions, prompt_template)
        self.logger.debug("User content prepared",
                        request_id=request_id,
                        user_content_length=len(user_content))
        
        item_schemas = []
        for gt, pred in zip(ground_truths, predictions):
            item_schemas.append({
                "type": "object",
                "properties": {
                    "ground_truth": {"const": gt},
                    "prediction": {"const": pred},
                    "classification": {
                        "type": "string",
                        "enum": categories
                    }
                },
                "required": ["ground_truth", "prediction", "classification"],
                "additionalProperties": False
            })
        
        dynamic_schema = {
            "type": "array",
            "items": item_schemas
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        self.logger.debug("Messages prepared",
                        request_id=request_id,
                        total_messages=len(messages))
        
        try:
            self.logger.debug("API preparation",
                            request_id=request_id,
                            model=self.model_name,
                            thinking_enabled=self.enable_thinking)
            
            completion_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "classification_array",
                        "schema": dynamic_schema
                    }
                }
            }
            
            completion_params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
            self.logger.debug("Thinking configuration set for request", 
                            request_id=request_id, 
                            thinking_enabled=self.enable_thinking)
            
            self.logger.debug("Starting API call", request_id=request_id)
            api_start_time = time.time()
            completion = await self.client.chat.completions.create(**completion_params)
            api_duration = time.time() - api_start_time
            api_duration_ms = round(api_duration * 1000, 2)
            self.logger.debug("API call completed",
                            request_id=request_id,
                            api_call_duration_ms=api_duration_ms)
            
            reasoning_content = getattr(completion.choices[0].message, 'reasoning_content', None)
            if reasoning_content:
                self.logger.debug("Reasoning content received",
                                request_id=request_id,
                                reasoning_length=len(reasoning_content),
                                reasoning_preview=reasoning_content[:200])
            else:
                self.logger.debug("No reasoning content", request_id=request_id)
            
            response_text = completion.choices[0].message.content.strip()
            self.logger.debug("Response content received",
                            request_id=request_id,
                            content_length=len(response_text),
                            content_preview=response_text[:200])
            
            self.logger.debug("Starting JSON parsing", request_id=request_id)
            try:
                classification_list = orjson.loads(response_text)
                parsed_count = len(classification_list) if isinstance(classification_list, list) else "non-list"
                self.logger.debug("JSON parsing successful",
                                request_id=request_id,
                                parsed_items=parsed_count)
            except orjson.JSONDecodeError as e:
                self.logger.error("JSON parsing failed",
                                request_id=request_id,
                                parse_error=str(e),
                                raw_response=response_text)
                raise LLMResponseParseError(f"Failed to parse JSON response: {e}. Response: {response_text}") from e
            
            self.logger.debug("Starting result processing", request_id=request_id)
            result_map = {}
            
            if isinstance(classification_list, list):
                self.logger.debug("Processing classification items",
                                request_id=request_id,
                                item_count=len(classification_list))
                for i, item in enumerate(classification_list):
                    if isinstance(item, dict) and all(key in item for key in ["ground_truth", "prediction", "classification"]):
                        gt = item["ground_truth"]
                        pred = item["prediction"]
                        classification = item["classification"]
                        
                        self.logger.debug("Processing classification item",
                                        request_id=request_id,
                                        item_index=i,
                                        ground_truth=gt,
                                        prediction=pred,
                                        classification=classification)
                        
                        if classification in categories:
                            result_map[(gt, pred)] = classification
                        else:
                            self.logger.warning("Invalid classification value",
                                              request_id=request_id,
                                              item_index=i,
                                              invalid_classification=classification,
                                              valid_choices=categories)
                            raise LLMResponseParseError(f"Invalid classification '{classification}'. Valid categories: {categories}")
                    else:
                        self.logger.warning("Malformed classification item",
                                          request_id=request_id,
                                          item_index=i,
                                          item=item)
                        if isinstance(item, dict) and "ground_truth" in item and "prediction" in item:
                            raise LLMResponseParseError(f"Malformed classification item: {item}")
                        else:
                            raise LLMResponseParseError(f"Completely invalid classification item: {item}")
            else:
                self.logger.error("Response is not a list",
                                request_id=request_id,
                                response_type=type(classification_list).__name__)
            
            missing_count = 0
            missing_pairs = []
            for gt, pred in zip(ground_truths, predictions):
                if (gt, pred) not in result_map:
                    missing_pairs.append((gt, pred))
                    missing_count += 1
            
            if missing_pairs:
                missing_str = ", ".join([f"('{gt}', '{pred}')" for gt, pred in missing_pairs[:3]])
                if len(missing_pairs) > 3:
                    missing_str += f" and {len(missing_pairs) - 3} more"
                raise LLMResponseParseError(f"Missing classification results for {missing_count} pairs: {missing_str}")
            
            
            total_duration = time.time() - start_time
            total_duration_ms = round(total_duration * 1000, 2)
            self.logger.info("Classification request completed",
                           request_id=request_id,
                           model=self.model_name,
                           api_base=self.api_base,
                           total_processing_time_ms=total_duration_ms,
                           result_count=len(result_map),
                           success_rate=100.0 if result_map else 0)
            return result_map

        except Exception as e:
            error_duration = time.time() - start_time
            error_duration_ms = round(error_duration * 1000, 2)
            error_msg = str(e)
            error_type = type(e).__name__
            
            self.logger.error("Classification request failed",
                            request_id=request_id,
                            model=self.model_name,
                            error_type=error_type,
                            error_message=error_msg,
                            processing_time_ms=error_duration_ms,
                            exception_repr=repr(e))
            
            if "429" in error_msg or "rate limit" in error_msg.lower():
                self.logger.error("Rate limiting detected", request_id=request_id)
                raise RateLimitError("vLLM server is overloaded.") from e
            elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                self.logger.error("Connection/timeout error", request_id=request_id)
                raise ServiceUnavailableError(f"Error communicating with vLLM API: {e}") from e
            else:
                self.logger.error("Parse/other error", request_id=request_id)
                raise LLMResponseParseError(f"Error parsing vLLM guided batch response: {e}") from e

    async def close_client(self):
        self.logger.info("VLLMClient closed", model=self.model_name)

    async def check_health(self) -> bool:
        try:
            models = await self.client.models.list()
            return True
        except Exception:
            return False