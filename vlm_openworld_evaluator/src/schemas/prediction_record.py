from pydantic import BaseModel, Field
from typing import Any, Optional

from src.schemas.base_record import DataContext
from src.schemas.prompt_config import AnswerFormat
from src.predictions.config import PredictionConfig

class PredictionRecord(BaseModel):
    """Canonical schema for a single VLM prediction result."""
    sample_id: str = Field(description="The unique ID of the sample this prediction corresponds to.")
    prediction_group_id: str = Field(description="An ID grouping multiple predictions for the same sample (often the same as sample_id).")
    model_name: str = Field(description="The identifier of the model that generated the prediction.")
    vlm_prediction: Any = Field(description="The raw output from the VLM.")
    prediction_index: int = Field(description="The index of this prediction if multiple are generated for the same sample (e.g., for temperature sampling).")
    prediction_config: PredictionConfig = Field(description="The configuration used for this specific prediction run.")
    data_context: DataContext = Field(description="A snapshot of the original DataRecord (excluding large fields like images) for context.")

    def get_ground_truth(self) -> Optional[str]:
        try:
            return self.data_context.reward_model.ground_truth
        except AttributeError:
            return None

    def get_answer_format(self) -> Optional[AnswerFormat]:
        try:
            return self.data_context.prompt_info.answer_format
        except AttributeError:
            return None

    def get_data_source(self) -> str:
        try:
            return self.data_context.extra_info.data_source
        except AttributeError:
            return 'unknown'

    def get_split(self) -> str:
        try:
            return self.data_context.split
        except AttributeError:
            return 'unknown'

    def get_prompt_name(self) -> str:
        try:
            return self.data_context.prompt_info.name
        except AttributeError:
            return 'unknown'
