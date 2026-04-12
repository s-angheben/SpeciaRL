from typing import List, Dict
from enum import Enum
from src.schemas.verification import VerificationRecord, StandardClassificationCategory
from .classifications import VerificationStatus, VerificationErrorType


FAILURE_SCORING_MAPS: Dict[str, Dict[VerificationErrorType, float]] = {
    "format_score": {
        VerificationErrorType.PARSING_VLM_OUTPUT: -1,
        VerificationErrorType.PREDICTION_TOO_LONG: -0.5,
        VerificationErrorType.API_NETWORK_ERROR: 0.0,
        VerificationErrorType.API_RESPONSE_ERROR: 0.0,
        VerificationErrorType.API_UNKNOWN_ERROR: 0.0,
        VerificationErrorType.PARSING_GROUND_TRUTH: -0.0,
        VerificationErrorType.PARSING_ANSWER_FORMAT: -0.0,
    },
    "ignore_all": {
        VerificationErrorType.PARSING_VLM_OUTPUT: 0.0,
        VerificationErrorType.PREDICTION_TOO_LONG: 0.0,
        VerificationErrorType.API_NETWORK_ERROR: 0.0,
        VerificationErrorType.API_RESPONSE_ERROR: 0.0,
        VerificationErrorType.API_UNKNOWN_ERROR: 0.0,
        VerificationErrorType.PARSING_GROUND_TRUTH: 0.0,
        VerificationErrorType.PARSING_ANSWER_FORMAT: 0.0,
    }
}


class ScoreCalculator:
    """Scores verification records via a success-category map and a named failure strategy."""
    def __init__(
        self,
        success_map: Dict[StandardClassificationCategory, float],
        failure_strategy: str = "format_score"
    ):
        if failure_strategy not in FAILURE_SCORING_MAPS:
            raise ValueError(f"Unknown failure strategy: {failure_strategy}")

        self.success_map = success_map
        self.failure_map = FAILURE_SCORING_MAPS[failure_strategy]

    def get_score(self, record: VerificationRecord) -> float:
        if record.is_success() and record.classification:
            return self.success_map.get(record.classification, 0.0)
        elif record.is_failure():
            return self.failure_map.get(record.error_type, 0.0)
        return 0.0

    def calculate_average_score(self, records: List[VerificationRecord]) -> float:
        if not records:
            return 0.0

        total_score = sum(self.get_score(record) for record in records)
        return total_score / len(records)

class VerificationMetrics:
    """Strategy-independent metrics over a list of VerificationRecords."""

    @staticmethod
    def count_format_errors(records: List[VerificationRecord]) -> int:
        return sum(1 for record in records if record.error_type == VerificationErrorType.PARSING_VLM_OUTPUT)

    @staticmethod
    def calculate_format_error_rate(records: List[VerificationRecord]) -> float:
        if not records:
            return 0.0
        format_errors = VerificationMetrics.count_format_errors(records)
        return (format_errors / len(records)) * 100