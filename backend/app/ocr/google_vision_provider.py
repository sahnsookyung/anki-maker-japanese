from __future__ import annotations

from pathlib import Path

from app.models.schemas import OcrToken
from app.ocr.providers import make_token


class GoogleVisionOcrProvider:
    name = "google_vision"

    def __init__(self) -> None:
        try:
            from google.cloud import vision
        except Exception as exc:
            raise RuntimeError("google-cloud-vision is not installed. Run `uv sync --extra cloud` in backend/.") from exc
        self._vision = vision
        self._client = vision.ImageAnnotatorClient()

    def recognize(self, image_path: Path, page_id: str) -> list[OcrToken]:
        image = self._vision.Image(content=image_path.read_bytes())
        response = self._client.document_text_detection(image=image)
        if response.error.message:
            raise RuntimeError(response.error.message)

        tokens: list[OcrToken] = []
        annotation = response.full_text_annotation
        for page in annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        text = "".join(symbol.text for symbol in word.symbols).strip()
                        if not text:
                            continue
                        vertices = word.bounding_box.vertices
                        xs = [float(vertex.x or 0) for vertex in vertices]
                        ys = [float(vertex.y or 0) for vertex in vertices]
                        confidence = float(getattr(word, "confidence", 0.0) or 0.0)
                        tokens.append(make_token(page_id, text, [min(xs), min(ys), max(xs), max(ys)], confidence, self.name))
        return tokens
