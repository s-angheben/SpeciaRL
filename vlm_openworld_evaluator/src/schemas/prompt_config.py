from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal

def _format_image_prompt(template: str) -> List[Dict[str, Any]]:
    """Split a template on `<image>` markers into a list of multimodal text/image content blocks."""
    content = []
    if "<image>" in template:
        parts = template.split("<image>")
        for i, part in enumerate(parts):
            if stripped_part := part.strip():
                content.append({"type": "text", "text": stripped_part})
            if i < len(parts) - 1:
                content.append({"type": "image"})
    else:
        # No marker: default to image-first, then the text body.
        content.append({"type": "image"})
        if stripped_text := template.strip():
            content.append({"type": "text", "text": stripped_text})
    return content

class AnswerFormat(BaseModel):
    """How to extract the answer from LLM output."""
    type: Literal[
        "last_line",
        "prefix",
        "suffix",
        "tag",
        "regex",
        "json"
    ]
    value: Optional[str] = None
    key: Optional[str] = None


class PromptConfig(BaseModel):
    name: str
    system_prompt: str = ""
    user_prompt_template: str
    requires_reasoning: bool = False
    dataset_specific: Optional[str] = None
    answer_format: AnswerFormat

    def get_vlm_prompt(self) -> List[Dict[str, Any]]:
        user_content = _format_image_prompt(self.user_prompt_template)
        final_prompt = []
        if self.system_prompt:
            final_prompt.append({
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}]
            })
        final_prompt.append({"role": "user", "content": user_content})
        return final_prompt
