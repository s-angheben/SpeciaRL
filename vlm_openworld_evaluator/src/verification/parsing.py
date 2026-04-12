import re
import orjson
from typing import Optional, Dict, Any
import regex as re

from src.schemas.prompt_config import AnswerFormat


def normalize_text(text: str) -> str:
    """Lowercase, drop punctuation (keeping letters/digits/Unicode dashes/whitespace), and collapse whitespace."""
    if not text:
        return ""
    normalized = re.sub(r'[^\p{L}\p{N}\p{Pd}\s]', ' ', str(text))
    normalized = normalized.lower()
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized.strip()

def extract_answer(output: str, fmt: AnswerFormat) -> str:
    if not isinstance(output, str):
        return ""
        
    try:
        if fmt.type == "last_line":
            return output.strip().splitlines()[-1].strip()

        elif fmt.type == "prefix" and fmt.value:
            for line in output.splitlines():
                if line.strip().startswith(fmt.value):
                    return line.strip()[len(fmt.value):].strip()
            return ""

        elif fmt.type == "suffix" and fmt.value:
            for line in output.splitlines():
                stripped = line.strip()
                if stripped.endswith(fmt.value):
                    return stripped[: -len(fmt.value)].strip()
            return ""

        elif fmt.type == "tag" and fmt.value:
            pattern = f"<{fmt.value}>(.*?)</{fmt.value}>"
            match = re.search(pattern, output, re.DOTALL)
            return match.group(1).strip() if match else ""

        elif fmt.type == "regex" and fmt.value:
            match = re.search(fmt.value, output, re.DOTALL)
            # Convention: first capturing group holds the answer.
            return match.group(1).strip() if match and match.groups() else ""

        elif fmt.type == "json" and fmt.key:
            match = re.search(r'\{.*\}', output, re.DOTALL)
            if match:
                try:
                    parsed = orjson.loads(match.group(0))
                    return str(parsed.get(fmt.key, ""))
                except ValueError:
                    return ""
            return ""

    except Exception:
        return ""

    return output.strip()