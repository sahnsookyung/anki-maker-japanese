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


LOCAL_BACKEND_ALIASES = {"", "local", "native", "paddle", "paddlepaddle"}
SERVER_BACKENDS = {"vllm-server", "sglang-server", "fastdeploy-server", "mlx-vlm-server", "llama-cpp-server"}


class PaddleOcrVlDocumentParser:
    provider = "paddleocr_vl"

    def __init__(self) -> None:
        try:
            from paddleocr import PaddleOCRVL
        except Exception as exc:
            raise RuntimeError("PaddleOCR-VL is unavailable. Run `uv sync --extra ocr-vl` in backend/.") from exc

        kwargs: dict[str, Any] = {
            "vl_rec_backend": _normalized_vl_backend(PADDLE_OCR_VL_BACKEND),
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

        try:
            self._parser = PaddleOCRVL(**kwargs)
        except Exception as exc:
            raise RuntimeError(_pipeline_creation_help(PADDLE_OCR_VL_BACKEND, exc)) from exc

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
            backend=_display_vl_backend(PADDLE_OCR_VL_BACKEND),
            block_count=len(blocks),
            blocks=blocks,
            markdown_text="\n\n".join(markdown_texts),
            warnings=warnings,
        )


@lru_cache(maxsize=1)
def get_paddle_ocr_vl_parser() -> PaddleOcrVlDocumentParser:
    return PaddleOcrVlDocumentParser()


def _normalized_vl_backend(value: str | None) -> str | None:
    normalized = (value or "").strip().lower()
    if normalized in LOCAL_BACKEND_ALIASES:
        return None
    if normalized in SERVER_BACKENDS:
        return normalized
    raise RuntimeError(
        "Unsupported PADDLE_OCR_VL_BACKEND value "
        f"{value!r}. Leave it empty for local PaddlePaddle inference, or use one of: {', '.join(sorted(SERVER_BACKENDS))}."
    )


def _pipeline_creation_help(configured_backend: str, exc: Exception) -> str:
    backend = _normalized_vl_backend(configured_backend)
    if backend is None:
        return (
            "PaddleOCR-VL local pipeline creation failed. Run `uv sync --group dev --extra ocr-vl` from backend/, "
            "then leave PADDLE_OCR_VL_BACKEND empty for local PaddlePaddle inference. "
            f"Original error: {exc}"
        )
    return (
        f"PaddleOCR-VL pipeline creation failed for backend {backend!r}. "
        "Verify PADDLE_OCR_VL_SERVER_URL points to a running OpenAI-compatible /v1 service and that optional "
        f"dependencies are installed with `uv sync --group dev --extra ocr-vl`. Original error: {exc}"
    )


def _display_vl_backend(value: str | None) -> str:
    return _normalized_vl_backend(value) or "local-paddlepaddle"


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
