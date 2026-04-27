from __future__ import annotations

from pathlib import Path
import base64
import json
from typing import Any

import requests

from app.core.config import OLLAMA_BASE_URL, OLLAMA_MODEL


class OllamaVisionClient:
    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def extract_json(self, image_path: Path, prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": f"{prompt}\n\nReturn only JSON.\n\nPayload:\n{json.dumps(payload, ensure_ascii=False)}",
                "images": [image_b64],
                "stream": False,
                "format": "json",
            },
            timeout=120,
        )
        response.raise_for_status()
        text = response.json().get("response", "{}")
        return json.loads(text)
