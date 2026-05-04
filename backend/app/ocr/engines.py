from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.models.schemas import DocumentParseResult, OcrToken
from app.ocr.service import recognize_with_warnings
from app.vision.paddle_ocr_vl import get_paddle_ocr_vl_parser


PADDLEOCR_ENGINE = "paddleocr"
PADDLEOCR_VL_ENGINE = "paddleocr_vl"
SUPPORTED_OCR_ENGINES = {PADDLEOCR_ENGINE, PADDLEOCR_VL_ENGINE}


@dataclass(frozen=True)
class OcrEngineResult:
    engine: str
    tokens: list[OcrToken]
    warnings: list[str]
    document_parse: DocumentParseResult | None = None
    evidence_tokens: list[OcrToken] | None = None


def run_ocr_engine(image_path: Path, page_id: str, engine: str = PADDLEOCR_ENGINE) -> OcrEngineResult:
    normalized = normalize_ocr_engine(engine)
    if normalized == PADDLEOCR_ENGINE:
        tokens, warnings = recognize_with_warnings(image_path, page_id)
        return OcrEngineResult(engine=normalized, tokens=tokens, warnings=warnings)
    parsed = get_paddle_ocr_vl_parser().parse(image_path, page_id)
    warnings = [
        *parsed.warnings,
        (
            "PaddleOCR-VL returned document blocks. Candidate extraction and visual evidence use "
            "block-level geometry instead of PaddleOCR word boxes."
        ),
    ]
    if not parsed.blocks and not parsed.markdown_text.strip():
        warnings.append("PaddleOCR-VL produced no document text; review page manually.")
    return OcrEngineResult(
        engine=normalized,
        tokens=[],
        warnings=warnings,
        document_parse=parsed,
        evidence_tokens=[],
    )


def normalize_ocr_engine(engine: str | None) -> str:
    normalized = (engine or PADDLEOCR_ENGINE).strip().lower().replace("-", "_")
    if normalized in {"base", "local", "vanilla"}:
        normalized = PADDLEOCR_ENGINE
    if normalized in {"vl", "paddle_vl", "paddleocrvl"}:
        normalized = PADDLEOCR_VL_ENGINE
    if normalized not in SUPPORTED_OCR_ENGINES:
        raise ValueError(f"Unsupported OCR engine {engine!r}. Use one of: {', '.join(sorted(SUPPORTED_OCR_ENGINES))}.")
    return normalized
