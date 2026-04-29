from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from app.core.script import script_summary
from app.models.schemas import DocumentParseResult, OcrToken
from app.ocr.providers import make_token
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


def run_ocr_engine(image_path: Path, page_id: str, engine: str = PADDLEOCR_ENGINE) -> OcrEngineResult:
    normalized = normalize_ocr_engine(engine)
    if normalized == PADDLEOCR_ENGINE:
        tokens, warnings = recognize_with_warnings(image_path, page_id)
        return OcrEngineResult(engine=normalized, tokens=tokens, warnings=warnings)
    parsed = get_paddle_ocr_vl_parser().parse(image_path, page_id)
    vl_tokens, conversion_warnings = tokens_from_document_parse(parsed)
    geometry_tokens, geometry_warnings = recognize_with_warnings(image_path, page_id)
    tokens = _align_tokens_to_geometry(vl_tokens, geometry_tokens) if geometry_tokens else vl_tokens
    warnings = [
        *parsed.warnings,
        *conversion_warnings,
        *geometry_warnings,
        (
            "PaddleOCR-VL text was parsed, but visual evidence uses PaddleOCR word boxes "
            "because VL boxes are document-block geometry."
        ),
    ]
    if geometry_tokens and vl_tokens:
        warnings.append("PaddleOCR-VL tokens keep VL text with PaddleOCR-aligned evidence boxes when text matches.")
    elif not geometry_tokens and vl_tokens:
        warnings.append("PaddleOCR word geometry was unavailable; using PaddleOCR-VL block-derived boxes for review only.")
    if not tokens:
        warnings.append("PaddleOCR-VL produced no normalized OCR tokens; review page manually.")
    return OcrEngineResult(engine=normalized, tokens=tokens, warnings=warnings, document_parse=parsed)


def _align_tokens_to_geometry(vl_tokens: list[OcrToken], geometry_tokens: list[OcrToken]) -> list[OcrToken]:
    if not vl_tokens or not geometry_tokens:
        return vl_tokens
    used_geometry_ids: set[str] = set()
    aligned: list[OcrToken] = []
    for token in vl_tokens:
        match = _best_geometry_match(token, geometry_tokens, used_geometry_ids)
        if match is None:
            aligned.append(token)
            continue
        used_geometry_ids.add(match.id)
        aligned.append(
            token.model_copy(
                update={
                    "bbox": match.bbox,
                    "confidence": min(token.confidence, match.confidence),
                }
            )
        )
    return aligned


def _best_geometry_match(token: OcrToken, geometry_tokens: list[OcrToken], used_geometry_ids: set[str]) -> OcrToken | None:
    normalized_token = _normalize_match_text(token.text)
    if not normalized_token:
        return None
    candidates = [
        candidate
        for candidate in geometry_tokens
        if candidate.id not in used_geometry_ids and _normalize_match_text(candidate.text) == normalized_token
    ]
    if not candidates:
        return None
    token_center_y = (token.bbox[1] + token.bbox[3]) / 2
    return min(candidates, key=lambda candidate: (abs(((candidate.bbox[1] + candidate.bbox[3]) / 2) - token_center_y), -candidate.confidence))


def _normalize_match_text(text: str) -> str:
    normalized = re.sub(r"\\(?:underline|text)\{([^{}]*)\}", r"\1", text)
    normalized = re.sub(r"\\[A-Za-z]+", "", normalized)
    normalized = normalized.replace("{", "").replace("}", "")
    return re.sub(r"\s+", "", normalized)


def normalize_ocr_engine(engine: str | None) -> str:
    normalized = (engine or PADDLEOCR_ENGINE).strip().lower().replace("-", "_")
    if normalized in {"base", "local", "vanilla"}:
        normalized = PADDLEOCR_ENGINE
    if normalized in {"vl", "paddle_vl", "paddleocrvl"}:
        normalized = PADDLEOCR_VL_ENGINE
    if normalized not in SUPPORTED_OCR_ENGINES:
        raise ValueError(f"Unsupported OCR engine {engine!r}. Use one of: {', '.join(sorted(SUPPORTED_OCR_ENGINES))}.")
    return normalized


def tokens_from_document_parse(result: DocumentParseResult) -> tuple[list[OcrToken], list[str]]:
    warnings: list[str] = []
    tokens: list[OcrToken] = []
    ordered_blocks = sorted(result.blocks, key=lambda block: block.order if block.order is not None else 10_000)
    synthetic_block_count = 0
    y_cursor = 20.0
    for block_index, block in enumerate(ordered_blocks):
        content = block.content.strip()
        if not content:
            continue
        block_tokens, used_synthetic_bbox = _tokens_from_text_block(
            page_id=result.page_id,
            text=content,
            block_bbox=block.bbox,
            block_index=block_index,
            y_cursor=y_cursor,
        )
        if used_synthetic_bbox:
            synthetic_block_count += 1
        if block_tokens:
            y_cursor = max(token.bbox[3] for token in block_tokens) + 18
            tokens.extend(block_tokens)

    if not tokens and result.markdown_text.strip():
        markdown_tokens, _ = _tokens_from_text_block(
            page_id=result.page_id,
            text=result.markdown_text,
            block_bbox=None,
            block_index=0,
            y_cursor=20.0,
        )
        tokens.extend(markdown_tokens)
        synthetic_block_count += 1

    if synthetic_block_count:
        warnings.append(
            "PaddleOCR-VL output was converted to synthetic OCR boxes; use it for text quality comparison, not precise evidence geometry."
        )
    return tokens, warnings


def _tokens_from_text_block(
    *,
    page_id: str,
    text: str,
    block_bbox: list[float] | None,
    block_index: int,
    y_cursor: float,
) -> tuple[list[OcrToken], bool]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        lines = [text.strip()]
    bbox = _valid_bbox(block_bbox)
    used_synthetic_bbox = bbox is None
    if bbox is None:
        width = max(320.0, max(len(line) for line in lines) * 18.0)
        height = max(28.0 * len(lines), 36.0)
        bbox = [20.0, y_cursor + block_index * 2.0, 20.0 + width, y_cursor + block_index * 2.0 + height]

    x1, y1, x2, y2 = bbox
    line_height = max(18.0, (y2 - y1) / max(1, len(lines)))
    tokens: list[OcrToken] = []
    for line_index, line in enumerate(lines):
        parts = _line_parts(line)
        if not parts:
            continue
        total_chars = max(1, sum(max(1, len(part)) for part in parts))
        available_width = max(24.0, x2 - x1)
        cursor = x1
        line_top = y1 + line_index * line_height
        line_bottom = min(y2, line_top + line_height * 0.82)
        for part in parts:
            part_width = max(12.0, available_width * (max(1, len(part)) / total_chars))
            token = make_token(
                page_id=page_id,
                text=part,
                bbox=[cursor, line_top, min(x2, cursor + part_width), max(line_top + 1.0, line_bottom)],
                confidence=0.68,
                source=PADDLEOCR_VL_ENGINE,
            )
            if token.text:
                tokens.append(token)
            cursor += part_width
    return tokens, used_synthetic_bbox


def _valid_bbox(value: list[float] | None) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    if not all(isinstance(item, (int, float)) for item in value):
        return None
    x1, y1, x2, y2 = [float(item) for item in value]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _line_parts(line: str) -> list[str]:
    spaced = re.split(r"\s+", line.strip())
    if len(spaced) > 1:
        return [part for part in spaced if part]
    return [line.strip()]


def engine_script_summary(tokens: list[OcrToken]) -> dict[str, int]:
    return script_summary([token.text for token in tokens])
