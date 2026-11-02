from pydantic import BaseModel, Field
from typing import Literal

class RunRequest(BaseModel):
    model_key: str = "qwen_0_5b"
    precision: Literal["FP16"] = "FP16"
    prompt: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=64, ge=1, le=256)
