from __future__ import annotations

from statistics import median
from typing import Any

from app.extraction.geometry import union_bbox
from app.models.schemas import DocumentParseBlock, OcrToken


def token_evidence(tokens: list[OcrToken], text: str = "", provenance: str = "ocr") -> dict[str, Any]:
    present = [token for token in tokens if token]
    if not present:
        return {"text": text, "provenance": provenance}
    return {
        "bbox": union_bbox([token.bbox for token in present]),
        "token_ids": [token.id for token in present],
        "text": text or "".join(token.text for token in present),
        "confidence": round(float(median([token.confidence for token in present])), 3),
        "provenance": provenance,
    }


def static_evidence(text: str, provenance: str = "inferred") -> dict[str, Any]:
    return {"text": text, "provenance": provenance}


def block_evidence(block: DocumentParseBlock | None, text: str, provenance: str = "paddleocr_vl_block") -> dict[str, Any]:
    evidence: dict[str, Any] = {"text": text, "provenance": provenance}
    if block:
        if block.bbox:
            evidence["bbox"] = block.bbox
        if block.id:
            evidence["block_ids"] = [block.id]
        if block.confidence is not None:
            evidence["confidence"] = round(float(block.confidence), 3)
    return evidence


def block_list_evidence(
    blocks: list[DocumentParseBlock] | tuple[DocumentParseBlock, ...],
    text: str,
    provenance: str = "paddleocr_vl_block",
) -> dict[str, Any]:
    evidence: dict[str, Any] = {"text": text, "provenance": provenance}
    present = [block for block in blocks if block]
    bboxes = [block.bbox for block in present if block.bbox]
    if bboxes:
        evidence["bbox"] = union_bbox(bboxes)
    block_ids = [block.id for block in present if block.id]
    if block_ids:
        evidence["block_ids"] = list(dict.fromkeys(block_ids))
    confidences = [float(block.confidence) for block in present if block.confidence is not None]
    if confidences:
        evidence["confidence"] = round(float(median(confidences)), 3)
    return evidence


def put_evidence(source: dict[str, Any], field: str, evidence: dict[str, Any]) -> None:
    current = source.get("field_evidence")
    field_evidence = dict(current) if isinstance(current, dict) else {}
    field_evidence[field] = evidence
    source["field_evidence"] = field_evidence
