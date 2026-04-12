from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Any
from enum import Enum

from .prediction_record import PredictionRecord
from .base_record import DataContext
from ..verification.classifications import VerificationStatus, VerificationErrorType

class StandardClassificationCategory(str, Enum):
    WRONG = "wrong"
    ABSTAIN = "abstain"
    GENERIC = "generic"
    LESS_SPECIFIC = "less specific"
    SPECIFIC = "specific"
    MORE_SPECIFIC = "more specific"

class PreparedPrediction(BaseModel):
    """A prediction that passed parsing/normalization and is ready to be sent to the verifier API."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    prediction_id: str
    original_record: PredictionRecord
    parsed_prediction: str
    normalized_prediction: str
    normalized_ground_truth: str


class VerificationConfigSnapshot(BaseModel):
    """Key verification settings embedded in each record for traceability."""
    model: Optional[str]
    verifier_prompt: Optional[str]
    api_base_url: str

class VerificationRecord(BaseModel):
    """Canonical schema for a single verification result, written one-per-line to the output NDJSON."""
    prediction_id: str = Field(description="The unique ID of the prediction that was verified.")
    sample_id: str = Field(description="The ID of the original data sample.")
    prediction_group_id: str = Field(description="The ID grouping multiple predictions for the same sample.")
    prediction_index: int = Field(description="The index of the prediction within its group.")
    model_name: str = Field(description="The identifier of the model that generated the prediction.")
    vlm_prediction: Any = Field(description="The raw, original output from the VLM.")
    parsed_prediction: str = Field(description="The extracted answer from the VLM's output.")
    normalized_prediction: str = Field(description="The normalized version of the parsed prediction.")
    normalized_ground_truth: str = Field(description="The normalized version of the ground truth label.")
    
    status: VerificationStatus = Field(description="The overall outcome of the verification process.")
    
    classification: Optional[StandardClassificationCategory] = Field(
        default=None,
        description="The API classification result (only if status is 'success')."
    )
    
    error_type: Optional[VerificationErrorType] = Field(None, description="The specific type of error (only on failure).")
    error_detail: Optional[str] = Field(None, description="A detailed error message (only on failure).")
    
    verification_config: VerificationConfigSnapshot = Field(description="A snapshot of the verification settings used.")
    data_context: DataContext = Field(description="Complete data context from the original sample for rich analysis.")
    
    
    def is_success(self) -> bool:
        return self.status == VerificationStatus.SUCCESS

    def is_preparation_failure(self) -> bool:
        return self.status == VerificationStatus.PREPARATION_FAILURE

    def is_api_failure(self) -> bool:
        return self.status == VerificationStatus.API_FAILURE

    def is_failure(self) -> bool:
        return not self.is_success()
