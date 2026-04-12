from dataclasses import dataclass
from typing import List, Dict, Type
from pydantic import BaseModel

from src.schemas.verification import StandardClassificationCategory

@dataclass
class VerifierPromptConfig:
    endpoint: str
    categories: List[str]
    category_ranks: Dict[str, int]
    response_type: str
    response_model: Type[BaseModel]
    scoring_map: Dict[StandardClassificationCategory, float]
    min_best: str
    max_best: str

from .api_schemas import StandardClassificationResponse

STANDARD_VERIFIER = VerifierPromptConfig(
    endpoint="/api/v1/classify/standard",
    response_model=StandardClassificationResponse,
    categories=[cat.value for cat in StandardClassificationCategory],
    category_ranks={
        "more specific": 1, "specific": 2, "less specific": 3, "generic": 4,
        "abstain": 5, "wrong": 6,
    },
    response_type="standard",
    scoring_map={
        StandardClassificationCategory.MORE_SPECIFIC: 1.0,
        StandardClassificationCategory.SPECIFIC: 1.0,
        StandardClassificationCategory.LESS_SPECIFIC: 0.6,
        StandardClassificationCategory.GENERIC: 0.2,
        StandardClassificationCategory.ABSTAIN: 0.0,
        StandardClassificationCategory.WRONG: -1.0,
    },
    min_best="abstain",
    max_best="specific"
)

VERIFIER_MANIFEST: Dict[str, VerifierPromptConfig] = {
    "ver_base_json": STANDARD_VERIFIER,
    "ver_base_soft_json": STANDARD_VERIFIER,
    "ver_base_single": STANDARD_VERIFIER,
    "ver_base_soft_single": STANDARD_VERIFIER,
    "ver_decision_tree_single": STANDARD_VERIFIER,
    "ver_normalize_strict_single": STANDARD_VERIFIER,
    "ver_checklist_single": STANDARD_VERIFIER,
}
