import logging
from typing import Dict, List, Any
from collections import Counter, defaultdict
from tabulate import tabulate

from src.schemas.verification import VerificationRecord
from .utils import categorize_record, clean_category_header, extract_data_source, extract_split, extract_prompt_name

logger = logging.getLogger(__name__)


class BreakdownAnalysisGenerator:
    """Renders count/percentage breakdown tables grouped by data source, split, or prompt name."""

    def __init__(self, records: List[VerificationRecord], ordered_report_categories: List[str]):
        self.records = records
        self.ordered_report_categories = ordered_report_categories

    def _generate_breakdown_table(self, grouping_func, group_name: str) -> str:
        grouped_records = defaultdict(list)
        for record in self.records:
            group_value = grouping_func(record.model_dump())
            grouped_records[group_value].append(record)

        group_names = sorted(grouped_records.keys())
        category_headers = [clean_category_header(cat) for cat in self.ordered_report_categories]

        headers = ["Metric"] + category_headers + ["Total"]

        table_data = []

        for group_name_value in group_names:
            records = grouped_records[group_name_value]
            categories = [categorize_record(r.model_dump()) for r in records]
            total = len(categories)
            counts = Counter(categories)

            count_row = [f"{group_name_value} Count"]
            percentage_row = [f"{group_name_value} %"]

            for cat in self.ordered_report_categories:
                count = counts.get(cat, 0)
                percentage = (count / total) * 100 if total > 0 else 0
                count_row.append(count)
                percentage_row.append(f"{percentage:.1f}%")

            count_row.append(total)
            percentage_row.append("100.0%")

            table_data.append(count_row)
            table_data.append(percentage_row)

        return tabulate(table_data, headers=headers, tablefmt="grid")

    def generate_dataset_breakdown_table(self) -> str:
        return self._generate_breakdown_table(extract_data_source, "Dataset Source")

    def generate_split_breakdown_table(self) -> str:
        return self._generate_breakdown_table(extract_split, "Split")

    def generate_prompt_breakdown_table(self) -> str:
        return self._generate_breakdown_table(extract_prompt_name, "Prompt Name")

    def generate_breakdown_analysis_section(self) -> List[str]:
        content = []

        dataset_breakdown_table = self.generate_dataset_breakdown_table()
        content.append("### 3.1. Breakdown by Dataset Source")
        content.append(dataset_breakdown_table)
        content.append("")

        split_breakdown_table = self.generate_split_breakdown_table()
        content.append("### 3.2. Breakdown by Split")
        content.append(split_breakdown_table)
        content.append("")

        prompt_breakdown_table = self.generate_prompt_breakdown_table()
        content.append("### 3.3. Breakdown by Prompt Name")
        content.append(prompt_breakdown_table)

        return content