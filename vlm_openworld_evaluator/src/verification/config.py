from pydantic import BaseModel, Field
from typing import Optional

class VerificationConfig(BaseModel):
    """Configuration for the verification pipeline."""
    
    # Input/Output
    output_dir: str = Field(default="./output", description="Base output directory for raw verification data")
    results_dir: str = Field(default="./results", description="Results directory for final analysis reports")
    
    # API Configuration
    api_base_url: str = Field(description="Base URL for verification API")
    api_timeout: int = Field(default=60, description="API timeout in seconds")
    batch_size: int = Field(default=10, description="Batch size for API calls")
    
    # Processing Configuration
    model: Optional[str] = Field(default=None, description="LLM model for verification (e.g., 'gpt-4')")
    verifier_prompt: Optional[str] = Field(default=None, description="Verifier prompt to use for classification")
    
    # Resumability
    resume: bool = Field(default=True, description="Resume from existing results")
    
    # Length Filtering
    max_label_words: Optional[int] = Field(default=None, description="Maximum number of words allowed in parsed predictions. If exceeded, classify as TOO_LONG")
    
    # Results Saving
    save_verifier_result: bool = Field(default=True, description="Wait for API processing completion and download all results")