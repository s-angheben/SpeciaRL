import orjson
import hashlib
from typing import Dict, Any
from pathlib import Path


def generate_config_hash(config: Dict[str, Any]) -> str:
    """Deterministic SHA256 of a config dict; keys ending in `_path`/`_file` are replaced by file-content hashes."""
    config_for_hash = _prepare_config_for_hashing(config)
    config_json = orjson.dumps(config_for_hash, option=orjson.OPT_SORT_KEYS).decode('utf-8')
    return hashlib.sha256(config_json.encode()).hexdigest()


def _prepare_config_for_hashing(config: Dict[str, Any]) -> Dict[str, Any]:
    config_copy = {}

    for key, value in config.items():
        if isinstance(value, dict):
            config_copy[key] = _prepare_config_for_hashing(value)
        elif isinstance(value, list):
            config_copy[key] = [
                _prepare_config_for_hashing(item) if isinstance(item, dict) else item
                for item in value
            ]
        elif _is_file_path_key(key) and isinstance(value, str):
            config_copy[key] = _hash_file_content(value)
        else:
            config_copy[key] = value

    return config_copy


def _is_file_path_key(key: str) -> bool:
    return key.endswith('_path') or key.endswith('_file') or key in ['global_prompts_path', 'source_prompts_path']


def _hash_file_content(file_path: str) -> str:
    try:
        path = Path(file_path)
        if path.exists() and path.is_file():
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            return hashlib.sha256(content.encode('utf-8')).hexdigest()
        else:
            # Fall back to hashing the raw path string so older artifacts (built before file-content hashing) stay reachable.
            return file_path
    except Exception:
        return file_path


def calculate_dataset_hash(build_config) -> str:
    return generate_config_hash(build_config.model_dump())


def calculate_prediction_hash(prediction_config, dataset_hash: str) -> str:
    """Prediction hash with backward-compat: drop batch-size knobs and force batch_size=1 so older artifacts still match."""
    config_dict = prediction_config.model_dump()
    config_dict["dataset_hash"] = dataset_hash

    config_dict.pop("inference_batch_size", None)
    config_dict.pop("prediction_chunk_size", None)
    config_dict["batch_size"] = 1

    if config_dict.get("prompts_override_file") is None:
        config_dict.pop("prompts_override_file", None)

    return generate_config_hash(config_dict)


def calculate_verification_hash(verification_config, predictions_hash: str) -> str:
    config_dict = verification_config.model_dump()
    config_dict["predictions_hash"] = predictions_hash
    return generate_config_hash(config_dict)


def sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "_").replace("-", "_")