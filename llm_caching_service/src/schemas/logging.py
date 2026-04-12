"""
Pydantic schemas for MongoDB classification logging.
Defines the structure of classification log documents.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from bson import ObjectId


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic v2."""
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.str_schema(),
            serialization=core_schema.to_string_ser_schema(),
        )

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        if isinstance(v, str) and ObjectId.is_valid(v):
            return ObjectId(v)
        raise ValueError("Invalid ObjectId")


class OccurrenceEntry(BaseModel):
    """Individual occurrence of a classification event."""
    timestamp: datetime = Field(description="When this classification occurred (Rome timezone)")
    extra_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for this occurrence"
    )

    model_config = {
        "json_encoders": {
            datetime: lambda dt: dt.isoformat()
        }
    }


class ClassificationLogEntry(BaseModel):
    """
    MongoDB document schema for classification logging.
    Tracks unique combinations of ground_truth, prediction, run_name, model, and prompt.
    """
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    ground_truth: str = Field(description="The ground truth label")
    prediction: str = Field(description="The predicted label")
    run_name: str = Field(description="Name of the run/experiment")
    model: str = Field(description="Model used for classification")
    prompt: str = Field(description="Prompt template used")
    classification_result: str = Field(description="The classification result")
    count: int = Field(description="Number of times this combination was seen")
    first_seen: datetime = Field(description="First occurrence timestamp (Rome timezone)")
    last_seen: datetime = Field(description="Most recent occurrence timestamp (Rome timezone)")
    occurrences: List[OccurrenceEntry] = Field(
        default_factory=list,
        description="List of individual occurrence timestamps"
    )

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {
            ObjectId: str,
            datetime: lambda dt: dt.isoformat()
        }
    }


class ClassificationLogEvent(BaseModel):
    """
    Event schema for queuing classification events.
    Used to pass data from classification service to logging workers.
    """
    ground_truth: str
    prediction: str
    run_name: str
    model: str
    prompt: str
    classification_result: str
    timestamp: datetime
    request_id: Optional[str] = None
    extra_info: Optional[Dict[str, Any]] = None

    model_config = {
        "json_encoders": {
            datetime: lambda dt: dt.isoformat()
        }
    }


class ClassificationLogBatch(BaseModel):
    """Batch of classification log events for efficient processing."""
    events: List[ClassificationLogEvent]
    
    model_config = {
        "json_encoders": {
            datetime: lambda dt: dt.isoformat()
        }
    }