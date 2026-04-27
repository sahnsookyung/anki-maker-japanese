from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.core.ids import new_id
from app.core.script import classify_script
from app.models.schemas import OcrToken


class OcrProvider(Protocol):
    name: str

    def recognize(self, image_path: Path, page_id: str) -> list[OcrToken]:
        ...


def make_token(page_id: str, text: str, bbox: list[float], confidence: float, source: str) -> OcrToken:
    return OcrToken(
        id=new_id("tok"),
        page_id=page_id,
        text=text.strip(),
        bbox=bbox,
        confidence=max(0.0, min(1.0, confidence)),
        script_class=classify_script(text),
        source=source,
    )
