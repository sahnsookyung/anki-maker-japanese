from __future__ import annotations

from pathlib import Path
import os

from app.core.config import (
    PADDLE_OCR_KOREAN_TEXT_DETECTION_MODEL_NAME,
    PADDLE_OCR_KOREAN_TEXT_RECOGNITION_MODEL_NAME,
    PADDLE_OCR_MAX_SIDE_LEN,
    PADDLE_OCR_TEXT_DETECTION_MODEL_NAME,
    PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME,
    PADDLE_OCR_USE_DOC_ORIENTATION_CLASSIFY,
    PADDLE_OCR_USE_DOC_UNWARPING,
    PADDLE_OCR_USE_TEXTLINE_ORIENTATION,
)
from app.ocr.providers import make_token
from app.models.schemas import OcrToken


class PaddleOcrProvider:
    name = "paddleocr"

    def __init__(
        self,
        *,
        name: str = "paddleocr",
        text_detection_model_name: str = PADDLE_OCR_TEXT_DETECTION_MODEL_NAME,
        text_recognition_model_name: str = PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME,
    ) -> None:
        self.name = name
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        self._predict_kwargs = {
            "use_doc_orientation_classify": PADDLE_OCR_USE_DOC_ORIENTATION_CLASSIFY,
            "use_doc_unwarping": PADDLE_OCR_USE_DOC_UNWARPING,
            "use_textline_orientation": PADDLE_OCR_USE_TEXTLINE_ORIENTATION,
            "text_det_limit_side_len": PADDLE_OCR_MAX_SIDE_LEN,
            "return_word_box": False,
        }
        self._ocr = PaddleOCR(
            text_detection_model_name=text_detection_model_name,
            text_recognition_model_name=text_recognition_model_name,
            use_doc_orientation_classify=PADDLE_OCR_USE_DOC_ORIENTATION_CLASSIFY,
            use_doc_unwarping=PADDLE_OCR_USE_DOC_UNWARPING,
            use_textline_orientation=PADDLE_OCR_USE_TEXTLINE_ORIENTATION,
            text_det_limit_side_len=PADDLE_OCR_MAX_SIDE_LEN,
        )

    def recognize(self, image_path: Path, page_id: str) -> list[OcrToken]:
        image, scale_x, scale_y = self._load_image(image_path)
        result = self._ocr.predict(image, **self._predict_kwargs)
        tokens: list[OcrToken] = []
        for page in result or []:
            texts = list(page.get("rec_texts") or [])
            scores = list(page.get("rec_scores") or [])
            boxes = page.get("rec_boxes")
            if boxes is None or len(boxes) == 0:
                boxes = page.get("rec_polys") or []
            if hasattr(boxes, "tolist"):
                boxes = boxes.tolist()
            for text, confidence, box in zip(texts, scores, boxes):
                bbox = self._box_to_bbox(box, scale_x=scale_x, scale_y=scale_y)
                if bbox is None:
                    continue
                token = make_token(page_id, str(text), bbox, float(confidence), self.name)
                if token.text:
                    tokens.append(token)
        return tokens

    def _load_image(self, image_path: Path):
        import numpy as np
        from PIL import Image, ImageOps

        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        original_width, original_height = image.size
        longest_side = max(image.width, image.height)
        if PADDLE_OCR_MAX_SIDE_LEN > 0 and longest_side > PADDLE_OCR_MAX_SIDE_LEN:
            scale = PADDLE_OCR_MAX_SIDE_LEN / float(longest_side)
            new_size = (
                max(1, int(round(image.width * scale))),
                max(1, int(round(image.height * scale))),
            )
            resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
            image = image.resize(new_size, resample=resampling)
        scale_x = original_width / float(image.width)
        scale_y = original_height / float(image.height)
        return np.array(image), scale_x, scale_y

    def _box_to_bbox(self, box, *, scale_x: float = 1.0, scale_y: float = 1.0) -> list[float] | None:
        if box is None:
            return None
        if len(box) == 4 and not isinstance(box[0], (list, tuple)):
            return [
                float(box[0]) * scale_x,
                float(box[1]) * scale_y,
                float(box[2]) * scale_x,
                float(box[3]) * scale_y,
            ]

        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        if not xs or not ys:
            return None
        return [min(xs) * scale_x, min(ys) * scale_y, max(xs) * scale_x, max(ys) * scale_y]


class PaddleKoreanOcrProvider(PaddleOcrProvider):
    def __init__(self) -> None:
        super().__init__(
            name="paddleocr_korean",
            text_detection_model_name=PADDLE_OCR_KOREAN_TEXT_DETECTION_MODEL_NAME,
            text_recognition_model_name=PADDLE_OCR_KOREAN_TEXT_RECOGNITION_MODEL_NAME,
        )
