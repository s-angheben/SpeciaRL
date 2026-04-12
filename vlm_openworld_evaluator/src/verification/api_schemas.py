from pydantic import BaseModel
from typing import List
from src.schemas.verification import StandardClassificationCategory

class StandardClassificationResponse(BaseModel):
    classifications: List[StandardClassificationCategory]
