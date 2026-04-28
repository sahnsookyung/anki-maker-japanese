from __future__ import annotations

from statistics import median
from typing import Any

from app.extraction.geometry import union_bbox
from app.models.schemas import OcrToken


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


def put_evidence(source: dict[str, Any], field: str, evidence: dict[str, Any]) -> None:
    current = source.get("field_evidence")
    field_evidence = dict(current) if isinstance(current, dict) else {}
    field_evidence[field] = evidence
    source["field_evidence"] = field_evidence
