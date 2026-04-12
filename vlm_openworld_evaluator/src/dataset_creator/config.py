from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from src.schemas.prompt_config import PromptConfig

class SourceConfig(BaseModel):
    """Defines the configuration for a single data source."""
    name: str
    source_split: str = Field(
        description="The split to load from the original source dataset (e.g., 'train', 'test')."
    )
    target_split: str = Field(
        description="The name to assign to this split in the output dataset (e.g., 'train', 'curriculum_easy')."
    )
    indices: Optional[str] = None
    subset_size: Optional[str] = Field(
        default=None,
        description="Number of samples to randomly select from available indices. Can be absolute number (e.g., '100') or percentage (e.g., '10%'). If not specified, all available samples are used."
    )
    include: Optional[List[str]] = Field(
        default=None,
        description="List of class names to include in the dataset. After subset_size sampling, only samples with labels in this list are kept. If not specified, all classes are included."
    )
    exclude: Optional[List[str]] = Field(
        default=None,
        description="List of class names to exclude from the dataset. After subset_size sampling, samples with labels in this list are discarded. If not specified, no classes are excluded."
    )
    source_prompts_path: Optional[str] = Field(
        default=None,
        description="Path to a source-specific NDJSON prompt file, overriding the global pool."
    )
    loader_args: Dict[str, Any] = Field(default_factory=dict)
    ability: Optional[str] = "OpenWorld classification"
    style: Optional[str] = "AI"
    cache_dir: str = Field(default="data/datasets", description="Local directory path for cached datasets")
    
class BuildConfig(BaseModel):
    """Defines the main build configuration."""
    output_dir: str
    global_prompts_path: str = Field(
        description="Path to the default NDJSON file containing the global pool of prompts."
    )
    sources: List[SourceConfig]
    seed: int = Field(default=42, description="Random seed for reproducibility.")
