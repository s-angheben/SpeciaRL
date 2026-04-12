from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from .prompt_config import PromptConfig

class RewardModel(BaseModel):
    style: str
    ground_truth: str

class ExtraInfo(BaseModel):
    data_source: str
    index_orig: int
    source_split: str
    source_index: int
    loader_args: Dict[str, Any] = Field(default_factory=dict)
    best_prediction: Optional[str] = Field(default=None, description="Best prediction from PassN analysis")


class BaseDataRecord(BaseModel):
    """Core record fields without image bytes; used as both DataRecord's base and the data_context attached to predictions."""
    data_source: str = Field(description="For verl")
    sample_id: str = Field(description="Unique and deterministic identifier for the sample.")
    index: int = Field(description="Sequential index of the sample in the final dataset.")
    prompt: List[Dict[str, Any]] = Field(description="The full VLM prompt, including roles and content.")
    prompt_info: PromptConfig = Field(description="The full configuration of the prompt used.")
    split: str = Field(description="The target split this sample belongs to (e.g., 'train', 'test').")
    ability: str = Field(description="The primary AI ability being tested by this sample.")
    reward_model: RewardModel = Field(description="Information needed for reward modeling and verification.")
    extra_info: ExtraInfo = Field(description="Additional metadata about the sample's origin and prompting.")

DataContext = BaseDataRecord
