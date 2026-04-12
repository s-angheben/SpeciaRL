"""
Pydantic schemas for run management API endpoints.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

from .logging import ClassificationLogEntry


class RunStatus(str, Enum):
    """Status of a classification run."""
    PROCESSING = "processing"
    COMPLETED = "completed" 
    IDLE = "idle"


class RunStatusResponse(BaseModel):
    """Response schema for run status endpoint."""
    run_name: str = Field(description="Name of the run")
    status: RunStatus = Field(description="Current status of the run")
    queue_pending_count: int = Field(description="Number of pending events in queue")
    total_processed: int = Field(description="Total events processed for this run")
    last_activity: Optional[datetime] = Field(description="Last activity timestamp")
    
    model_config = {
        "json_encoders": {
            datetime: lambda dt: dt.isoformat()
        }
    }


class ClassificationStats(BaseModel):
    """Classification statistics for a run."""
    classification_distribution: Dict[str, int] = Field(
        description="Count of each classification result type"
    )
    total_classifications: int = Field(description="Total number of classifications")
    success_rate: float = Field(description="Percentage of successful classifications")
    models_used: List[str] = Field(description="List of models used in this run")
    time_range: Dict[str, Optional[datetime]] = Field(
        description="Time range of classifications"
    )
    
    model_config = {
        "json_encoders": {
            datetime: lambda dt: dt.isoformat()
        }
    }


class RunResultsResponse(BaseModel):
    """Response schema for run results endpoint."""
    run_name: str = Field(description="Name of the run")
    total_count: int = Field(description="Total number of classification entries")
    returned_count: int = Field(description="Number of entries in this response")
    offset: int = Field(description="Current offset")
    limit: int = Field(description="Maximum entries per response")
    has_more: bool = Field(description="Whether more results are available")
    stats: Optional[ClassificationStats] = Field(description="Aggregated statistics")
    results: List[ClassificationLogEntry] = Field(description="Classification log entries")
    
    model_config = {
        "json_encoders": {
            datetime: lambda dt: dt.isoformat()
        }
    }