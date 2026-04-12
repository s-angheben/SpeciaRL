from typing import List
from tabulate import tabulate

from src.schemas.verification import VerificationRecord
from ..scoring import ScoreCalculator


class ScoreAnalysisGenerator:
    """Renders the score-summary table for a single configured ScoreCalculator strategy."""

    def __init__(self, records: List[VerificationRecord], score_calculator: ScoreCalculator):
        self.records = records
        self.score_calculator = score_calculator

    def generate_score_summary_table(self, records: List[VerificationRecord] = None) -> str:
        if records is None:
            records = self.records

        if not records:
            return "No records to score."

        total_score = sum(self.score_calculator.get_score(record) for record in records)
        avg_score = total_score / len(records)

        table_data = [
            ["Total Samples", len(records)],
            ["Total Score", f"{total_score:.2f}"],
            ["Average Score", f"{avg_score:.3f}"],
        ]
        
        return tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid")