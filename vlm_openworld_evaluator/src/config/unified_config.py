from pydantic import BaseModel, Field, model_validator
from typing import Optional, Dict, Any
import tempfile
import os
from pathlib import Path

from src.dataset_creator.config import BuildConfig
from src.predictions.config import PredictionConfig
from src.verification.config import VerificationConfig


class UnifiedConfig(BaseModel):
    """Unified configuration for the complete VLM evaluation pipeline."""
    
    # Experiment metadata
    experiment_name: str = Field(description="Unique name for this experiment")
    experiment_group: str = Field(default="no_group", description="Group name for organizing experiments")
    description: Optional[str] = Field(default=None, description="Human-readable description")
    
    # Pipeline configuration sections - using original classes directly
    dataset: BuildConfig = Field(description="Dataset creation configuration")
    prediction: PredictionConfig = Field(description="Prediction configuration")
    verification: VerificationConfig = Field(description="Verification configuration")
    
    # Global settings
    output_dir: str = Field(default="./output/", description="Global output directory for all stages")
    results_dir: str = Field(default="./results/", description="Global results directory for reports")
    seed: int = Field(default=42, description="Global random seed")
    force_rebuild: bool = Field(default=False, description="Force rebuild of existing artifacts")
    
    # Special fields for handling inline prompts
    prompts_content: Optional[str] = Field(default=None, description="Inline prompt content (alternative to prompts_file)")
    prompts_file: Optional[str] = Field(default=None, description="Path to prompt file")
    temp_prompt_file: Optional[str] = Field(default=None, exclude=True, description="Internal: temp file path for cleanup")
    
    @model_validator(mode='before')
    @classmethod
    def set_global_defaults_and_handle_prompts(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(values, dict):
            if values.get('prompts_content') and not values.get('prompts_file'):
                # Inline prompt content: spill to a temp NDJSON so the dataset stage can read it as a file like normal.
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.ndjson', delete=False)
                temp_file.write(values['prompts_content'])
                temp_file.flush()
                temp_file.close()

                if 'dataset' not in values:
                    values['dataset'] = {}
                values['dataset']['global_prompts_path'] = temp_file.name

                values['temp_prompt_file'] = temp_file.name

            elif values.get('prompts_file'):
                if 'dataset' not in values:
                    values['dataset'] = {}
                values['dataset']['global_prompts_path'] = values['prompts_file']

            global_output_dir = values.get('output_dir', './output/')
            global_results_dir = values.get('results_dir', './results/')

            if 'dataset' in values and isinstance(values['dataset'], dict):
                if 'output_dir' not in values['dataset']:
                    values['dataset']['output_dir'] = global_output_dir
                if 'seed' not in values['dataset']:
                    values['dataset']['seed'] = values.get('seed', 42)

            if 'prediction' in values and isinstance(values['prediction'], dict):
                if 'output_dir' not in values['prediction']:
                    values['prediction']['output_dir'] = global_output_dir
                if 'seed' not in values['prediction']:
                    values['prediction']['seed'] = values.get('seed', 42)

            if 'verification' in values and isinstance(values['verification'], dict):
                if 'output_dir' not in values['verification']:
                    values['verification']['output_dir'] = global_output_dir
                if 'results_dir' not in values['verification']:
                    values['verification']['results_dir'] = global_results_dir

        return values

    def validate_prompts(self) -> None:
        if not self.prompts_content and not self.prompts_file:
            raise ValueError("Either prompts_content or prompts_file must be specified")
        if self.prompts_content and self.prompts_file:
            raise ValueError("Only one of prompts_content or prompts_file should be specified")

    def cleanup_temp_files(self) -> None:
        if hasattr(self, 'temp_prompt_file') and self.temp_prompt_file:
            try:
                if os.path.exists(self.temp_prompt_file):
                    os.unlink(self.temp_prompt_file)
            except OSError:
                pass