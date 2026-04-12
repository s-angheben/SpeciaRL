import time
import uuid
from typing import List, Dict, Tuple, Literal
from enum import Enum
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from pydantic import BaseModel, Field

from src.core.exceptions import LLMResponseParseError, ServiceUnavailableError
from .base import LLMClientABC
from src.core.logging import get_logger


class ClassificationItem(BaseModel):
    ground_truth: str = Field(description="The ground truth label")
    prediction: str = Field(description="The predicted label")
    classification: str = Field(description="The classification result")


class GeminiClient(LLMClientABC):
    """A client for making async API calls to Google's Gemini models with structured output."""
    
    def __init__(self, api_key: str, model: str, enable_thinking: bool = True, timeout_seconds: int = 600):
        # Initialize client without custom http_options that might cause issues
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.enable_thinking = enable_thinking
        self.timeout_seconds = timeout_seconds
        self.logger = get_logger(__name__)

    async def initialize(self):
        try:
            self.logger.info("GeminiClient initialized", 
                           model=self.model,
                           thinking_enabled=self.enable_thinking)
        except Exception as e:
            self.logger.error("Failed to initialize Gemini client", error=str(e))
            raise ServiceUnavailableError(f"Gemini client initialization failed: {e}") from e
    
    def get_name(self) -> str:
        return self.model
    
    def get_client_type(self) -> str:
        return "gemini"

    def get_thinking_enabled(self) -> bool:
        return self.enable_thinking

    async def get_classifications(
        self, 
        ground_truths: List[str], 
        predictions: List[str], 
        prompt_template: Dict[str, str], 
        categories: List[str]
    ) -> Dict[Tuple[str, str], str]:
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        self.logger.debug("Starting Gemini classification request",
                         request_id=request_id,
                         model=self.model,
                         item_count=len(ground_truths),
                         thinking_enabled=self.enable_thinking,
                         categories=categories,
                         sample_ground_truths=ground_truths[:2] if len(ground_truths) > 2 else ground_truths,
                         sample_predictions=predictions[:2] if len(predictions) > 2 else predictions)
        
        system_instruction = prompt_template.get("system_prompt", "")
        user_content = self.build_user_content(ground_truths, predictions, prompt_template)

        # Using dictionary instead of Pydantic model to avoid SDK conversion issues
        config_params = {
            "response_mime_type": "application/json",
            "response_schema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ground_truth": {"type": "string"},
                        "prediction": {"type": "string"},
                        "classification": {"type": "string", "enum": categories}
                    },
                    "required": ["ground_truth", "prediction", "classification"]
                }
            },
            "temperature": 0.0,
        }
        
        if system_instruction:
            config_params["system_instruction"] = system_instruction

        if self.enable_thinking:
            # Use dynamic thinking budget (-1) for better reasoning
            config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=-1)
            self.logger.debug("Enabled dynamic thinking", request_id=request_id)
        else:
            config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            self.logger.debug("Disabled thinking", request_id=request_id)

        try:
            self.logger.debug("Starting Gemini API call", request_id=request_id)
            api_start_time = time.time()
            
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=user_content,
                config=types.GenerateContentConfig(**config_params)
            )
            
            api_duration = time.time() - api_start_time
            api_duration_ms = round(api_duration * 1000, 2)
            self.logger.debug("Gemini API call completed",
                             request_id=request_id,
                             api_call_duration_ms=api_duration_ms)
            
            # Parse JSON response manually since using dictionary schema
            import orjson
            response_text = response.text.strip() if response.text else ""
            
            if not response_text:
                self.logger.warning("No response text received from Gemini", 
                                   request_id=request_id)
                raise LLMResponseParseError("No classification results returned from Gemini API")
            
            try:
                classification_list = orjson.loads(response_text)
            except orjson.JSONDecodeError as e:
                self.logger.error("JSON parsing failed", 
                                 request_id=request_id,
                                 parse_error=str(e),
                                 raw_response=response_text[:500])
                raise LLMResponseParseError(f"Failed to parse JSON response: {e}") from e
            
            if not isinstance(classification_list, list):
                self.logger.error("Response is not a list",
                                 request_id=request_id,
                                 response_type=type(classification_list).__name__)
                raise LLMResponseParseError(f"Expected array response, got {type(classification_list).__name__}")
            
            self.logger.debug("Processing Gemini classification results",
                             request_id=request_id,
                             parsed_items=len(classification_list))
            
            result_map = {}
            for i, item in enumerate(classification_list):
                if not isinstance(item, dict) or not all(key in item for key in ["ground_truth", "prediction", "classification"]):
                    self.logger.error("Malformed classification item",
                                     request_id=request_id,
                                     item_index=i,
                                     item=item)
                    raise LLMResponseParseError(f"Malformed classification item at index {i}: {item}")
                
                gt = item["ground_truth"]
                pred = item["prediction"]
                classification = item["classification"]
                
                self.logger.debug("Processing classification item",
                                 request_id=request_id,
                                 item_index=i,
                                 ground_truth=gt,
                                 prediction=pred,
                                 classification=classification)
                
                if classification not in categories:
                    self.logger.warning("Invalid classification value from Gemini",
                                       request_id=request_id,
                                       item_index=i,
                                       invalid_classification=classification,
                                       valid_categories=categories)
                    raise LLMResponseParseError(f"Invalid classification '{classification}'. Valid categories: {categories}")
                
                result_map[(gt, pred)] = classification
            
            missing_pairs = []
            for gt, pred in zip(ground_truths, predictions):
                if (gt, pred) not in result_map:
                    missing_pairs.append((gt, pred))
            
            if missing_pairs:
                missing_str = ", ".join([f"('{gt}', '{pred}')" for gt, pred in missing_pairs[:3]])
                if len(missing_pairs) > 3:
                    missing_str += f" and {len(missing_pairs) - 3} more"
                self.logger.error("Missing classification results from Gemini",
                                 request_id=request_id,
                                 missing_count=len(missing_pairs))
                raise LLMResponseParseError(f"Missing classification results for {len(missing_pairs)} pairs: {missing_str}")

            total_duration = time.time() - start_time
            total_duration_ms = round(total_duration * 1000, 2)
            
            self.logger.info("Gemini classification completed successfully",
                           request_id=request_id,
                           model=self.model,
                           total_processing_time_ms=total_duration_ms,
                           result_count=len(result_map),
                           success_rate=100.0,
                           thinking_enabled=self.enable_thinking)
            
            return result_map

        except ClientError as e:
            error_duration = time.time() - start_time
            error_duration_ms = round(error_duration * 1000, 2)
            
            self.logger.error("Gemini API client error",
                             request_id=request_id,
                             model=self.model,
                             error_code=getattr(e, 'status_code', 'unknown'),
                             error_message=str(e),
                             processing_time_ms=error_duration_ms)
            
            # Check if it's the recurring API issue mentioned in forums
            if "INVALID_ARGUMENT" in str(e) and "schema" in str(e).lower():
                raise ServiceUnavailableError(
                    f"Gemini API schema validation error (this may be a temporary API issue): {e}"
                ) from e
            else:
                raise ServiceUnavailableError(f"Gemini API failed: {e}") from e
                
        except Exception as e:
            error_duration = time.time() - start_time
            error_duration_ms = round(error_duration * 1000, 2)
            error_type = type(e).__name__
            
            self.logger.error("Gemini classification request failed",
                             request_id=request_id,
                             model=self.model,
                             error_type=error_type,
                             error_message=str(e),
                             processing_time_ms=error_duration_ms,
                             exception_repr=repr(e))
            
            if isinstance(e, (LLMResponseParseError, ServiceUnavailableError)):
                raise
            else:
                raise ServiceUnavailableError(f"Unexpected Gemini API error: {e}") from e

    async def close_client(self):
        # Gemini SDK doesn't require explicit cleanup
        self.logger.debug("Gemini client closed", model=self.model)

    async def check_health(self) -> bool:
        try:
            test_response = await self.client.aio.models.generate_content(
                model=self.model,
                contents="Test"
            )
            return test_response is not None
        except Exception as e:
            self.logger.warning("Gemini health check failed", error=str(e))
            return False