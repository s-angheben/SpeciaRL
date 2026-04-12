import logging
from typing import List, Any, Dict
from collections import defaultdict

from src.schemas.verification import VerificationRecord
from ..classifications import VerificationStatus, VerificationErrorType
from .utils import generate_summary_table_from_records

logger = logging.getLogger(__name__)


class PassNAnalysisGenerator:
    """Pass@N analysis: pick the best prediction per sample (by classification rank) for exploration runs."""

    def __init__(self, records: List[VerificationRecord], category_ranks: Dict[str, int], ordered_report_categories: List[str]):
        self.records = records
        self.category_ranks = category_ranks
        self.ordered_report_categories = ordered_report_categories

    def is_exploration_scenario(self) -> bool:
        if not self.records:
            return False

        samples = defaultdict(list)
        for record in self.records:
            samples[record.sample_id].append(record)

        return any(len(predictions) > 1 for predictions in samples.values())

    def get_record_rank(self, record: VerificationRecord) -> int:
        if record.is_success():
            classification_value = record.classification
            return self.category_ranks.get(classification_value, 99)
        else:
            error_type = record.error_type
            ERROR_RANK = {
                VerificationErrorType.PARSING_VLM_OUTPUT: 100,
                VerificationErrorType.PREDICTION_TOO_LONG: 101,
                VerificationErrorType.PARSING_GROUND_TRUTH: 102,
                VerificationErrorType.PARSING_ANSWER_FORMAT: 103,
                VerificationErrorType.API_NETWORK_ERROR: 104,
                VerificationErrorType.API_RESPONSE_ERROR: 105,
                VerificationErrorType.API_UNKNOWN_ERROR: 106,
            }
            return ERROR_RANK.get(error_type, 999)

    def select_best_per_sample(self) -> List[VerificationRecord]:
        samples = defaultdict(list)
        for record in self.records:
            samples[record.sample_id].append(record)

        best_predictions = []
        for sample_id, predictions in samples.items():
            best_prediction = min(predictions,
                                key=lambda p: (self.get_record_rank(p),
                                             p.prediction_index))
            best_predictions.append(best_prediction)

        return best_predictions

    def generate_passn_summary_table(self, best_records: List[VerificationRecord]) -> str:
        record_dicts = [record.model_dump() for record in best_records]
        return generate_summary_table_from_records(record_dicts, self.ordered_report_categories)

    def generate_passn_analysis_section(self, score_generator) -> List[str]:
        if not self.is_exploration_scenario():
            return []

        content = []
        best_records = self.select_best_per_sample()

        content.append("### 4.1. PassN Performance Summary")
        passn_table = self.generate_passn_summary_table(best_records)
        content.append(passn_table)
        content.append("")

        content.append("### 4.2. PassN Score Summary")
        passn_score_table = score_generator.generate_score_summary_table(best_records)
        content.append(passn_score_table)
        content.append("")

        return content