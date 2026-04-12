import logging
from typing import List, Any
from collections import Counter, defaultdict
from tabulate import tabulate

from src.schemas.verification import VerificationRecord
from ..classifications import VerificationStatus, VerificationErrorType
from .utils import get_detailed_error_categories, clean_category_header, extract_data_source, extract_split

logger = logging.getLogger(__name__)


class ErrorAnalysisGenerator:
    """Renders breakdown tables for preparation and API failures."""

    def __init__(self, records: List[VerificationRecord]):
        self.records = records

    def get_failure_records(self) -> List[VerificationRecord]:
        return [r for r in self.records if r.is_failure()]

    def get_preparation_failure_records(self) -> List[VerificationRecord]:
        return [r for r in self.records if r.is_preparation_failure()]

    def get_api_failure_records(self) -> List[VerificationRecord]:
        return [r for r in self.records if r.is_api_failure()]

    def generate_preparation_failure_table(self) -> str:
        prep_records = self.get_preparation_failure_records()

        error_counts = Counter()
        for record in prep_records:
            error_type = record.error_type or 'unknown'
            error_counts[error_type] += 1

        total_prep_errors = len(prep_records)

        prep_error_types = [
            VerificationErrorType.PARSING_VLM_OUTPUT,
            VerificationErrorType.PREDICTION_TOO_LONG,
            VerificationErrorType.PARSING_GROUND_TRUTH,
            VerificationErrorType.PARSING_ANSWER_FORMAT
        ]

        headers = [clean_category_header(error_type.value) for error_type in prep_error_types] + ["Total"]

        count_row = []
        percentage_row = []

        for error_type in prep_error_types:
            count = error_counts.get(error_type.value, 0)
            percentage = (count / total_prep_errors) * 100 if total_prep_errors > 0 else 0
            count_row.append(count)
            percentage_row.append(f"{percentage:.1f}%")

        count_row.append(total_prep_errors)
        percentage_row.append("100.0%" if total_prep_errors > 0 else "0.0%")

        table_data = [count_row, percentage_row]
        row_labels = ["Count", "Percentage"]

        headers_with_label = ["Metric"] + headers
        table_data_with_labels = []
        for i, row in enumerate(table_data):
            table_data_with_labels.append([row_labels[i]] + row)

        return tabulate(table_data_with_labels, headers=headers_with_label, tablefmt="grid")

    def generate_api_failure_table(self) -> str:
        api_records = self.get_api_failure_records()

        error_counts = Counter()
        for record in api_records:
            error_type = record.error_type or 'unknown'
            error_counts[error_type] += 1

        total_api_errors = len(api_records)

        api_error_types = [
            VerificationErrorType.API_NETWORK_ERROR,
            VerificationErrorType.API_RESPONSE_ERROR,
            VerificationErrorType.API_UNKNOWN_ERROR
        ]

        headers = [clean_category_header(error_type.value) for error_type in api_error_types] + ["Total"]

        count_row = []
        percentage_row = []

        for error_type in api_error_types:
            count = error_counts.get(error_type.value, 0)
            percentage = (count / total_api_errors) * 100 if total_api_errors > 0 else 0
            count_row.append(count)
            percentage_row.append(f"{percentage:.1f}%")

        count_row.append(total_api_errors)
        percentage_row.append("100.0%" if total_api_errors > 0 else "0.0%")

        table_data = [count_row, percentage_row]
        row_labels = ["Count", "Percentage"]

        headers_with_label = ["Metric"] + headers
        table_data_with_labels = []
        for i, row in enumerate(table_data):
            table_data_with_labels.append([row_labels[i]] + row)

        return tabulate(table_data_with_labels, headers=headers_with_label, tablefmt="grid")

    def generate_error_analysis_section(self) -> List[str]:
        content = []

        failure_records = self.get_failure_records()
        if not failure_records:
            content.append("No verification errors occurred.")
            return content

        prep_failures = self.get_preparation_failure_records()
        api_failures = self.get_api_failure_records()

        content.append(f"Total errors: {len(failure_records)} out of {len(self.records)} records ({(len(failure_records)/len(self.records)*100):.1f}%)")
        content.append(f"- Preparation failures: {len(prep_failures)} ({(len(prep_failures)/len(self.records)*100):.1f}%)")
        content.append(f"- API failures: {len(api_failures)} ({(len(api_failures)/len(self.records)*100):.1f}%)")
        content.append("")

        content.append("### Preparation Failures Breakdown")
        if len(prep_failures) > 0:
            prep_table = self.generate_preparation_failure_table()
            content.append(prep_table)
        else:
            content.append("No preparation failures occurred.")
        content.append("")

        content.append("### API Failures Breakdown")
        if len(api_failures) > 0:
            api_table = self.generate_api_failure_table()
            content.append(api_table)
        else:
            content.append("No API failures occurred.")
        
        return content