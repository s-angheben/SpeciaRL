import hashlib
import orjson


def create_item_cache_key(ground_truth: str, prediction: str, model: str, prompt_name: str, model_type: str, thinking_enabled: bool = False, dtype: str = None) -> str:
    key_data = orjson.dumps({
        "gt": ground_truth,
        "pred": prediction,
        "model": model,
        "prompt_name": prompt_name,
        "model_type": model_type,
        "thinking_enabled": thinking_enabled,
        "dtype": dtype
    }, option=orjson.OPT_SORT_KEYS).decode('utf-8')
    
    hash_value = hashlib.sha256(key_data.encode('utf-8')).hexdigest()
    return f"item:{model}:{model_type}:{prompt_name}:{hash_value}"