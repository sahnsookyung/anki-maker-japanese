from __future__ import annotations

from pathlib import Path
import base64
import json
from typing import Any

import requests

from app.core.config import LLAMA_CPP_BASE_URL, LLAMA_CPP_MODEL


class LlamaCppVisionClient:
    """OpenAI-compatible llama.cpp server client for multimodal local models."""

    def __init__(self, model: str = LLAMA_CPP_MODEL, base_url: str = LLAMA_CPP_BASE_URL) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def extract_json(self, image_path: Path, prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{prompt}\nReturn only JSON.\nPayload:\n{json.dumps(payload, ensure_ascii=False)}",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                            },
                        ],
                    }
                ],
                "response_format": {"type": "json_object"},
            },
            timeout=120,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)
