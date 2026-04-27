from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import (
    PADDLE_OCR_VL_API_KEY,
    PADDLE_OCR_VL_API_MODEL_NAME,
    PADDLE_OCR_VL_BACKEND,
    PADDLE_OCR_VL_MAX_NEW_TOKENS,
    PADDLE_OCR_VL_MAX_PIXELS,
    PADDLE_OCR_VL_SERVER_URL,
    PADDLE_OCR_VL_USE_LAYOUT_DETECTION,
)
from app.models.schemas import DocumentParseBlock, DocumentParseResult


class PaddleOcrVlDocumentParser:
    provider = "paddleocr_vl"

    def __init__(self) -> None:
        try:
            from paddleocr import PaddleOCRVL
        except Exception as exc:
            raise RuntimeError("PaddleOCR-VL is unavailable. Run `uv sync --extra ocr-vl` in backend/.") from exc

        kwargs: dict[str, Any] = {
            "vl_rec_backend": PADDLE_OCR_VL_BACKEND or None,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_layout_detection": PADDLE_OCR_VL_USE_LAYOUT_DETECTION,
            "use_chart_recognition": False,
            "use_seal_recognition": False,
            "use_ocr_for_image_block": False,
            "format_block_content": False,
            "merge_layout_blocks": True,
            "use_queues": False,
        }
        if PADDLE_OCR_VL_SERVER_URL:
            kwargs["vl_rec_server_url"] = PADDLE_OCR_VL_SERVER_URL
        if PADDLE_OCR_VL_API_MODEL_NAME:
            kwargs["vl_rec_api_model_name"] = PADDLE_OCR_VL_API_MODEL_NAME
        if PADDLE_OCR_VL_API_KEY:
            kwargs["vl_rec_api_key"] = PADDLE_OCR_VL_API_KEY

        self._parser = PaddleOCRVL(**kwargs)

    def parse(self, image_path: Path, page_id: str) -> DocumentParseResult:
        results = self._parser.predict(
            str(image_path),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_layout_detection=PADDLE_OCR_VL_USE_LAYOUT_DETECTION,
            use_chart_recognition=False,
            use_seal_recognition=False,
            use_ocr_for_image_block=False,
            format_block_content=False,
            max_pixels=PADDLE_OCR_VL_MAX_PIXELS,
            max_new_tokens=PADDLE_OCR_VL_MAX_NEW_TOKENS,
        )
        blocks: list[DocumentParseBlock] = []
        markdown_texts: list[str] = []
        warnings: list[str] = []

        for result in results:
            payload = _result_payload(result)
            blocks.extend(_blocks_from_payload(payload))
            markdown_text = _markdown_text(result)
            if markdown_text:
                markdown_texts.append(markdown_text)

        if not blocks and not markdown_texts:
            warnings.append("PaddleOCR-VL returned no document blocks or markdown.")

        return DocumentParseResult(
            page_id=page_id,
            provider=self.provider,
            source_image_path=str(image_path),
            backend=PADDLE_OCR_VL_BACKEND,
            block_count=len(blocks),
            blocks=blocks,
            markdown_text="\n\n".join(markdown_texts),
            warnings=warnings,
        )


@lru_cache(maxsize=1)
def get_paddle_ocr_vl_parser() -> PaddleOcrVlDocumentParser:
    return PaddleOcrVlDocumentParser()


def _result_payload(result: Any) -> dict[str, Any]:
    data = getattr(result, "json", None)
    if isinstance(data, dict):
        nested = data.get("res")
        return nested if isinstance(nested, dict) else data
    if isinstance(result, dict):
        return result
    return {}


def _blocks_from_payload(payload: dict[str, Any]) -> list[DocumentParseBlock]:
    blocks: list[DocumentParseBlock] = []
    for item in payload.get("parsing_res_list") or []:
        if not isinstance(item, dict):
            continue
        blocks.append(
            DocumentParseBlock(
                label=str(item.get("block_label") or "unknown"),
                content=str(item.get("block_content") or ""),
                bbox=item.get("block_bbox"),
                order=item.get("block_order"),
            )
        )
    return blocks


def _markdown_text(result: Any) -> str:
    data = getattr(result, "markdown", None)
    if isinstance(data, dict):
        return str(data.get("markdown_texts") or "").strip()
    return ""
