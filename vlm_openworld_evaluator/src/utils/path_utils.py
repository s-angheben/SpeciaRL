"""Pipeline artifact path helpers and hash extraction."""

import re
from pathlib import Path
from typing import Optional


def extract_hash_from_filename(filepath: Path, expected_prefix: str) -> str:
    """Extract the hash from a filename of the form `{prefix}_{hash}.{ext}`."""
    filename = filepath.stem

    pattern = f"^{re.escape(expected_prefix)}_([a-zA-Z0-9]+)$"
    match = re.match(pattern, filename)

    if not match:
        raise ValueError(
            f"Filename '{filename}' doesn't match expected pattern '{expected_prefix}_<hash>'. "
            f"Expected format: {expected_prefix}_{{hash}}.{{ext}}"
        )

    return match.group(1)


def extract_dataset_hash(dataset_path: Path) -> str:
    return extract_hash_from_filename(dataset_path, "dataset")


def extract_predictions_hash(predictions_path: Path) -> str:
    filename = predictions_path.stem

    for prefix in ["predictions", "exploration"]:
        try:
            return extract_hash_from_filename(predictions_path, prefix)
        except ValueError:
            continue

    raise ValueError(
        f"Filename '{filename}' doesn't match expected pattern. "
        f"Expected format: predictions_{{hash}}.ndjson or exploration_{{hash}}.ndjson"
    )


def extract_verification_hash(verification_path: Path) -> str:
    return extract_hash_from_filename(verification_path, "verification")


def extract_api_results_hash(api_results_path: Path) -> str:
    return extract_hash_from_filename(api_results_path, "api_results")


def validate_file_exists(filepath: Path, stage_name: str) -> None:
    if not filepath.exists():
        raise FileNotFoundError(
            f"{stage_name} input file not found at {filepath}. "
            f"Make sure the previous stage completed successfully."
        )