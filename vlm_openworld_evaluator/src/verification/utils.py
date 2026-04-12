import yaml
from pathlib import Path
from typing import List, Generator, Dict, Any

from src.utils.hash_utils import generate_config_hash
from .config import VerificationConfig

def batch_list(data: List, batch_size: int) -> Generator[List, None, None]:
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def generate_verification_hash(config: VerificationConfig, predictions_hash: str) -> str:
    verification_config_dict = config.model_dump()
    verification_config_dict["predictions_hash"] = predictions_hash
    return generate_config_hash(verification_config_dict)

def load_verification_config(config_path: Path) -> VerificationConfig:
    try:
        with config_path.open('r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        return VerificationConfig(**config_data)
    except Exception as e:
        raise ValueError(f"Failed to load verification config from {config_path}: {e}")


def ensure_output_directory(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
