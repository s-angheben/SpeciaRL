from pydantic import Field
from typing import List

from .base_record import BaseDataRecord

class DataRecord(BaseDataRecord):
    """Canonical dataset record: BaseDataRecord plus serialized image bytes."""
    images: List[bytes] = Field(description="List of images associated with the sample, as bytes.")
