from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Optional, Dict, Any
from enum import Enum

class StandardClassificationCategory(str, Enum):
    WRONG = "wrong"
    ABSTAIN = "abstain"
    GENERIC = "generic"
    SPECIFIC = "specific"
    LESS_SPECIFIC = "less specific"
    MORE_SPECIFIC = "more specific"

class BinaryClassificationCategory(str, Enum):
    CORRECT = "correct"
    WRONG = "wrong"

class ClassificationRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ground_truths": ["wild pansy", "common daisy"],
                "predictions": ["wild pansy", "viola tricolor"],
                "model": "gemini-flash", 
                "verifier_prompt": "ver_base_json",
                "extra_info": [
                    {"source_dataset": "flowers_v1", "source_index": 42, "index": 0},
                    {"source_dataset": "flowers_v1", "source_index": 43, "index": 1}
                ]
            }
        }
    )
    
    ground_truths: List[str] = Field(
        ...,
        description="The normalized ground truth labels for the classification task."
    )
    predictions: List[str] = Field(
        ...,
        description="The normalized predicted labels for the classification task."
    )

    model: Optional[str] = Field(
        default=None,
        description="The name of the LLM model to use (e.g., 'gpt-3.5-turbo', 'gpt-4'). Uses a system default if not provided."
    )

    verifier_prompt: Optional[str] = Field(
        default=None,
        description="The verifier prompt to use for classification. If not provided, the system default will be used."
    )

    run_name: Optional[str] = Field(
        default="default",
        description="Optional run name for tracking purposes."
    )

    extra_info: Optional[List[Optional[Dict[str, Any]]]] = Field(
        default=None,
        description="Optional metadata for each ground_truth/prediction pair. Must match the length of ground_truths/predictions if provided."
    )

    @field_validator('extra_info')
    @classmethod
    def validate_extra_info_length(cls, v, info):
        """Validate that extra_info length matches ground_truths/predictions if provided."""
        if v is not None:
            ground_truths = info.data.get('ground_truths', [])
            if len(v) != len(ground_truths):
                raise ValueError(f"extra_info length ({len(v)}) must match ground_truths length ({len(ground_truths)})")
        return v

class StandardClassificationResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "classifications": ["specific", "wrong"]
            }
        }
    )
    
    classifications: List[StandardClassificationCategory] = Field(
        ...,
        description="A list of standard classification results."
    )

class BinaryClassificationResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "classifications": ["correct", "wrong"]
            }
        }
    )
    
    classifications: List[BinaryClassificationCategory] = Field(
        ...,
        description="A list of binary classification results ('correct' or 'wrong')."
    )

CATEGORIES_STANDARD = [e.value for e in StandardClassificationCategory]
CATEGORIES_BINARY = [e.value for e in BinaryClassificationCategory]