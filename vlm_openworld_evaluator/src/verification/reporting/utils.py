from typing import List, Dict, Any
from collections import Counter
from tabulate import tabulate
from ..classifications import VerificationStatus, VerificationErrorType


def get_error_types_for_report() -> List[str]:
    return [err.value for err in VerificationErrorType]


def clean_category_header(category_value: str) -> str:
    header = category_value.replace('_', ' ').title()
    if header == "Parsing Vlm Output":
        return "Parsing_VLM_Output"
    elif header == "Prediction Too Long":
        return "Prediction_Too_Long"
    elif header == "Api Network Error":
        return "API_Network_Error"
    elif header == "Api Response Error":
        return "API_Response_Error"
    elif header == "Api Unknown Error":
        return "API_Unknown_Error"
    elif header == "Preparation Failure":
        return "Preparation_Failure"
    elif header == "Api Failure":
        return "API_Failure"
    return header


def extract_data_source(record: dict) -> str:
    """Return the record's data_source, suffixed with sorted supercategories from loader_args when present."""
    if 'data_context' not in record or not record['data_context']:
        return 'unknown'
    
    try:
        data_context = record['data_context']
        if not isinstance(data_context, dict):
            return 'unknown'
            
        extra_info = data_context.get('extra_info', {})
        if not isinstance(extra_info, dict):
            return 'unknown'
            
        base_source = extra_info.get('data_source', 'unknown')
        loader_args = extra_info.get('loader_args', {})
        
        if isinstance(loader_args, dict) and 'supercategories' in loader_args:
            supercats = loader_args['supercategories']
            if isinstance(supercats, list) and supercats:
                sorted_cats = sorted(s for s in supercats if s)
                if sorted_cats:
                    return f"{base_source} ({', '.join(sorted_cats)})"
                    
        return base_source
        
    except (AttributeError, TypeError, KeyError):
        return 'unknown'


def extract_split(record: dict) -> str:
    if 'data_context' in record and record['data_context']:
        try:
            data_context = record['data_context']
            if isinstance(data_context, dict):
                return data_context.get('split', 'unknown')
        except (AttributeError, TypeError, KeyError):
            pass
    return 'unknown'


def extract_prompt_name(record: dict) -> str:
    if 'data_context' in record and record['data_context']:
        try:
            data_context = record['data_context']
            if isinstance(data_context, dict):
                prompt_info = data_context.get('prompt_info', {})
                if isinstance(prompt_info, dict):
                    return prompt_info.get('name', 'unknown')
        except (AttributeError, TypeError, KeyError):
            pass
    return 'unknown'


def categorize_record(record: dict) -> str:
    """Bucket a record into a fine-grained category: success classification, error_type for failures, or raw status."""
    status = record.get('status')

    if status == VerificationStatus.SUCCESS.value:
        return record.get('classification', 'unknown')
    elif status in [VerificationStatus.PREPARATION_FAILURE.value, VerificationStatus.API_FAILURE.value]:
        return record.get('error_type', status)
    else:
        return status or 'unknown'


def generate_summary_table_from_records(records: List[Dict[str, Any]], ordered_categories: List[str]) -> str:
    if not records:
        return "No records to generate a summary table from."

    total = len(records)
    categories = [categorize_record(r) for r in records]
    counts = Counter(categories)

    headers = [clean_category_header(cat) for cat in ordered_categories] + ["Total"]

    count_row = []
    percentage_row = []

    for cat in ordered_categories:
        count = counts.get(cat, 0)
        percentage = (count / total) * 100 if total > 0 else 0
        count_row.append(count)
        percentage_row.append(f"{percentage:.1f}%")

    count_row.append(total)
    percentage_row.append("100.0%")

    table_data = [count_row, percentage_row]
    row_labels = ["Count", "Percentage"]

    headers_with_label = ["Metric"] + headers
    table_data_with_labels = []
    for i, row in enumerate(table_data):
        table_data_with_labels.append([row_labels[i]] + row)

    return tabulate(table_data_with_labels, headers=headers_with_label, tablefmt="grid")