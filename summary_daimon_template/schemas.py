from typing import Optional
from pydantic import BaseModel


class InputSchema(BaseModel):
    url: str
    output_path: Optional[str] = None
    ollama_model: Optional[str] = "mistral:latest"