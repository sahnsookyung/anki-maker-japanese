from __future__ import annotations

from pathlib import Path
from typing import Any

from app.extraction.crops import crop_bbox
from app.models.schemas import OcrToken
from app.validation.dictionary import DictionaryValidator
from app.vision.service import get_vision_client


VOCAB_PROMPT = (
    "You extract Japanese study vocabulary from a cropped workbook row. "
    "Return only JSON matching the schema. Do not invent missing data. "
    "Every field must be supported by visible text or OCR tokens. "
    "If uncertain, set needs_review=true and add a warning."
)

MCQ_PROMPT = (
    "You extract a Japanese multiple-choice vocabulary question from a cropped workbook region. "
    "Return only JSON. Do not guess the correct answer unless it is present in the supplied answer map. "
    "If the target underline is unclear, set needs_review=true."
)


def cleanup_vocab_items(
    image_path: Path,
    items: list[dict[str, Any]],
    tokens: list[OcrToken],
    validator: DictionaryValidator,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    token_map = {token.id: token for token in tokens}
    client = _client_or_warning(warnings)
    if not client:
        return items, warnings

    cleaned: list[dict[str, Any]] = []
    for item in items:
        row_tokens = [token_map[token_id] for token_id in item.get("evidence_tokens", []) if token_id in token_map]
        crop_path = crop_bbox(image_path, item["bbox"], item["id"])
        payload = {
            "task": "extract_vocab_row",
            "ocr_tokens": [_token_payload(token) for token in row_tokens],
            "expected_fields": ["surface", "reading", "meaning_ko"],
        }
        response = _call_json(client, crop_path, VOCAB_PROMPT, payload, warnings, item["id"])
        cleaned.append(_merge_vocab_response(item, response, row_tokens, validator))
    return cleaned, warnings


def cleanup_mcq_items(
    image_path: Path,
    items: list[dict[str, Any]],
    tokens: list[OcrToken],
    answer_map: dict[int, int],
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    client = _client_or_warning(warnings)
    if not client:
        return items, warnings

    cleaned: list[dict[str, Any]] = []
    for item in items:
        row_tokens = _tokens_in_bbox(tokens, item["bbox"])
        crop_path = crop_bbox(image_path, item["bbox"], item["id"])
        payload = {
            "task": "extract_mcq_question",
            "ocr_tokens": [_token_payload(token) for token in row_tokens],
            "answer_map": {str(k): v for k, v in answer_map.items()},
            "expected_fields": ["sentence", "target", "choices", "correct_choice_no"],
        }
        response = _call_json(client, crop_path, MCQ_PROMPT, payload, warnings, item["id"])
        cleaned.append(_merge_mcq_response(item, response, row_tokens))
    return cleaned, warnings


def _client_or_warning(warnings: list[str]):
    try:
        return get_vision_client()
    except Exception as exc:
        warnings.append(f"VLM cleanup unavailable: {exc}")
        return None


def _call_json(client, crop_path: Path, prompt: str, payload: dict[str, Any], warnings: list[str], item_id: str) -> dict[str, Any]:
    try:
        response = client.extract_json(crop_path, prompt, payload)
        if not isinstance(response, dict):
            warnings.append(f"VLM response for {item_id} was not a JSON object.")
            return {}
        return response
    except Exception as exc:
        warnings.append(f"VLM cleanup failed for {item_id}: {exc}")
        return {}


def _merge_vocab_response(
    item: dict[str, Any],
    response: dict[str, Any],
    row_tokens: list[OcrToken],
    validator: DictionaryValidator,
) -> dict[str, Any]:
    if not response:
        return item
    merged = dict(item)
    item_warnings = list(merged.get("warnings", []))
    evidence = response.get("evidence_tokens") or []
    row_token_ids = {token.id for token in row_tokens}
    if not evidence or not set(evidence).issubset(row_token_ids):
        item_warnings.append("VLM output did not cite only row OCR evidence; kept OCR extraction.")
        merged["warnings"] = _unique(item_warnings)
        merged["needs_review"] = True
        return merged

    for key in ("surface", "reading", "meaning_ko"):
        value = response.get(key)
        if isinstance(value, str) and value.strip() and value.strip() != merged.get(key):
            merged[key] = value.strip()
            item_warnings.append("VLM corrected OCR grouping; verify against source crop.")
    _, validation_warnings = validator.validate_vocab(merged.get("surface", ""), merged.get("reading", ""))
    item_warnings.extend(validation_warnings)
    item_warnings.extend(response.get("warnings") or [])
    merged["evidence_tokens"] = evidence
    merged["needs_review"] = bool(response.get("needs_review")) or bool(item_warnings)
    merged["warnings"] = _unique(item_warnings)
    merged["confidence"] = min(float(response.get("confidence", merged.get("confidence", 0.5))), float(merged.get("confidence", 0.5)))
    return merged


def _merge_mcq_response(item: dict[str, Any], response: dict[str, Any], row_tokens: list[OcrToken]) -> dict[str, Any]:
    if not response:
        return item
    merged = dict(item)
    item_warnings = list(merged.get("warnings", []))
    evidence = response.get("evidence_tokens") or []
    row_token_ids = {token.id for token in row_tokens}
    if evidence and not set(evidence).issubset(row_token_ids):
        item_warnings.append("VLM output cited OCR tokens outside the question crop; kept OCR extraction.")
        merged["warnings"] = _unique(item_warnings)
        merged["needs_review"] = True
        return merged

    for key in ("sentence", "target", "choices", "correct_choice_no"):
        value = response.get(key)
        if value and value != merged.get(key):
            if key == "correct_choice_no" and merged.get("answer_source") == "answer_strip":
                continue
            merged[key] = value
            item_warnings.append("VLM corrected OCR grouping; verify against source crop.")
    if merged.get("correct_choice_no") and merged.get("choices") and not merged.get("correct_answer"):
        choice_no = int(merged["correct_choice_no"])
        choices = list(merged["choices"])
        if 1 <= choice_no <= len(choices):
            merged["correct_answer"] = choices[choice_no - 1]
    item_warnings.extend(response.get("warnings") or [])
    merged["needs_review"] = bool(response.get("needs_review")) or bool(item_warnings)
    merged["warnings"] = _unique(item_warnings)
    merged["confidence"] = min(float(response.get("confidence", merged.get("confidence", 0.5))), float(merged.get("confidence", 0.5)))
    return merged


def _tokens_in_bbox(tokens: list[OcrToken], bbox: list[float]) -> list[OcrToken]:
    x1, y1, x2, y2 = bbox
    return [
        token
        for token in tokens
        if ((token.bbox[0] + token.bbox[2]) / 2) >= x1
        and ((token.bbox[0] + token.bbox[2]) / 2) <= x2
        and ((token.bbox[1] + token.bbox[3]) / 2) >= y1
        and ((token.bbox[1] + token.bbox[3]) / 2) <= y2
    ]


def _token_payload(token: OcrToken) -> dict[str, Any]:
    return {
        "id": token.id,
        "text": token.text,
        "bbox": token.bbox,
        "confidence": token.confidence,
        "script_class": token.script_class,
    }


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
