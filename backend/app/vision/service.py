from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from app.core.config import VLM_PROVIDER


class VisionJsonClient(Protocol):
    def extract_json(self, image_path: Path, prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        ...


def get_vision_client() -> VisionJsonClient:
    if VLM_PROVIDER == "llama_cpp":
        from app.vision.llama_cpp import LlamaCppVisionClient

        return LlamaCppVisionClient()
    from app.vision.ollama import OllamaVisionClient

    return OllamaVisionClient()
