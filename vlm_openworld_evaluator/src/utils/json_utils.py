import orjson
import json
import numpy as np
from pydantic import BaseModel

def custom_json_serializer(obj):
    """orjson `default=` hook for Pydantic models and numpy scalars/arrays."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)

    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

class CustomEncoder:
    """orjson wrapper that mimics `json.dumps(cls=...)` for drop-in compatibility."""
    @staticmethod
    def encode(obj, **kwargs):
        options = orjson.OPT_INDENT_2 if kwargs.get('indent') else 0
        ensure_ascii = kwargs.get('ensure_ascii', True)
        if not ensure_ascii:
            options |= orjson.OPT_NON_STR_KEYS

        result = orjson.dumps(obj, default=custom_json_serializer, option=options)
        return result.decode('utf-8')
