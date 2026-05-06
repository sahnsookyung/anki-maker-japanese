from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.script import classify_script
from app.evaluation.golden import GoldenPage, GoldenVocabRow, meaning_matches, normalize_text
from app.models.schemas import CardCandidate, ProcessResult


@dataclass(frozen=True)
class VocabEvalResult:
    page_id: str
    image_path: str
    expected_page_type: str
    actual_page_type: str
    expected_rows: int
    extracted_items: int
    layout_matched_rows: int
    ocr_supported_items: int
    glossary_supported_items: int
    matched_rows: int
    surface_matches: int
    reading_matches: int
    surface_reading_matches: int
    meaning_matches: int
    generated_notes: int
    korean_field_missing_hangul: int
    japanese_field_has_hangul: int
    missing_row_ids: list[str]

    @property
    def row_accuracy(self) -> float:
        return self.matched_rows / self.expected_rows if self.expected_rows else 0.0

    @property
    def layout_recall(self) -> float:
        return self.layout_matched_rows / self.expected_rows if self.expected_rows else 0.0

    @property
    def surface_accuracy(self) -> float:
        return self.surface_matches / self.expected_rows if self.expected_rows else 0.0

    @property
    def reading_accuracy(self) -> float:
        return self.reading_matches / self.expected_rows if self.expected_rows else 0.0

    @property
    def meaning_accuracy(self) -> float:
        return self.meaning_matches / self.expected_rows if self.expected_rows else 0.0


def evaluate_vocab_page(golden: GoldenPage, process_result: ProcessResult) -> VocabEvalResult:
    items = _items_from_cards(process_result.cards)
    live_token_ids = {token.id for token in process_result.tokens}
    live_block_ids = {
        block.id for block in (process_result.document_parse.blocks if process_result.document_parse else []) if block.id
    }
    ocr_supported_items = [item for item in items if _item_has_ocr_evidence(item, live_token_ids, live_block_ids)]
    layout_matched_rows = _layout_matched_rows(golden.expected_rows, items)
    matched: set[str] = set()
    matched_item_indexes: set[int] = set()
    surface_item_indexes: set[int] = set()
    reading_item_indexes: set[int] = set()
    meaning_item_indexes: set[int] = set()
    surface_reading_item_indexes: set[int] = set()
    surface_matches = 0
    reading_matches = 0
    surface_reading_matches = 0
    meaning_match_count = 0

    for row in golden.expected_rows:
        surface_index = _first_unmatched_field_index(items, surface_item_indexes, "surface", row.surface, live_token_ids, live_block_ids)
        if surface_index is not None:
            surface_matches += 1
            surface_item_indexes.add(surface_index)
        reading_index = _first_unmatched_field_index(items, reading_item_indexes, "reading", row.reading, live_token_ids, live_block_ids)
        if reading_index is not None:
            reading_matches += 1
            reading_item_indexes.add(reading_index)
        meaning_index = _first_unmatched_field_index(items, meaning_item_indexes, "meaning_ko", row.meaning_ko, live_token_ids, live_block_ids)
        if meaning_index is not None:
            meaning_match_count += 1
            meaning_item_indexes.add(meaning_index)

        surface_reading_index = _first_unmatched_row_index(
            items,
            surface_reading_item_indexes,
            row,
            ("surface", "reading"),
            live_token_ids,
            live_block_ids,
        )
        if surface_reading_index is not None:
            surface_reading_matches += 1
            surface_reading_item_indexes.add(surface_reading_index)

        candidate_index = _first_unmatched_row_index(
            items,
            matched_item_indexes,
            row,
            ("surface", "reading", "meaning_ko"),
            live_token_ids,
            live_block_ids,
        )
        if candidate_index is not None:
            matched_item_indexes.add(candidate_index)
            matched.add(row.row_id)

    return VocabEvalResult(
        page_id=golden.page_id,
        image_path=str(golden.image_path),
        expected_page_type=golden.expected_page_type,
        actual_page_type=process_result.page.page_type,
        expected_rows=len(golden.expected_rows),
        extracted_items=len(items),
        layout_matched_rows=layout_matched_rows,
        ocr_supported_items=len(ocr_supported_items),
        glossary_supported_items=sum(1 for item in items if _item_has_glossary_evidence(item)),
        matched_rows=len(matched),
        surface_matches=surface_matches,
        reading_matches=reading_matches,
        surface_reading_matches=surface_reading_matches,
        meaning_matches=meaning_match_count,
        generated_notes=len(items),
        korean_field_missing_hangul=sum(1 for item in items if not _has_script(str(item.get("meaning_ko", "")), "hangul")),
        japanese_field_has_hangul=sum(
            1
            for item in items
            if _has_script(str(item.get("surface", "")), "hangul") or _has_script(str(item.get("reading", "")), "hangul")
        ),
        missing_row_ids=[row.row_id for row in golden.expected_rows if row.row_id not in matched],
    )


def _items_from_cards(cards: list[CardCandidate]) -> list[dict[str, Any]]:
    by_source: dict[str, dict[str, Any]] = {}
    for card in cards:
        if card.source_type != "vocab_item":
            continue
        by_source.setdefault(card.source_id, card.source)
    return list(by_source.values())


def _layout_matched_rows(rows: list[GoldenVocabRow], items: list[dict[str, Any]]) -> int:
    matched = 0
    used_indexes: set[int] = set()
    for row in rows:
        for index, item in enumerate(items):
            if index in used_indexes:
                continue
            if _row_has_any_matching_field(item, row):
                used_indexes.add(index)
                matched += 1
                break
    return matched


def _row_has_any_matching_field(item: dict[str, Any], row: GoldenVocabRow) -> bool:
    if normalize_text(str(item.get("surface", ""))) == normalize_text(row.surface):
        return True
    if normalize_text(str(item.get("reading", ""))) == normalize_text(row.reading):
        return True
    return meaning_matches(str(item.get("meaning_ko", "")), row.meaning_ko)


def _first_unmatched_field_index(
    items: list[dict[str, Any]],
    matched_item_indexes: set[int],
    field: str,
    expected: str,
    live_token_ids: set[str],
    live_block_ids: set[str],
) -> int | None:
    for index, item in enumerate(items):
        if index in matched_item_indexes:
            continue
        if _item_field_matches_with_ocr(item, field, expected, live_token_ids, live_block_ids):
            return index
    return None


def _first_unmatched_row_index(
    items: list[dict[str, Any]],
    matched_item_indexes: set[int],
    row: GoldenVocabRow,
    fields: tuple[str, ...],
    live_token_ids: set[str],
    live_block_ids: set[str],
) -> int | None:
    for index, item in enumerate(items):
        if index in matched_item_indexes:
            continue
        if all(
            _item_field_matches_with_ocr(item, field, _expected_field_value(row, field), live_token_ids, live_block_ids)
            for field in fields
        ):
            return index
    return None


def _item_has_ocr_evidence(item: dict[str, Any], live_token_ids: set[str], live_block_ids: set[str]) -> bool:
    evidence = item.get("field_evidence")
    if not isinstance(evidence, dict):
        return False
    return all(_field_has_ocr_evidence(evidence.get(field), item.get(field), live_token_ids, live_block_ids) for field in ("surface", "reading", "meaning_ko"))


def _field_has_ocr_evidence(value: Any, expected_value: Any, live_token_ids: set[str], live_block_ids: set[str]) -> bool:
    if not isinstance(value, dict):
        return False
    provenance = value.get("provenance")
    token_ids = value.get("token_ids")
    block_ids = value.get("block_ids")
    bbox = value.get("bbox")
    evidence_text = str(value.get("text") or "")
    expected_text = str(expected_value or "")
    has_bbox = isinstance(bbox, list) and len(bbox) == 4
    if not has_bbox or not _evidence_supports_value(evidence_text, expected_text):
        return False
    if provenance in {"ocr", "crop_ocr", "google_vision"}:
        if not isinstance(token_ids, list) or not token_ids:
            return False
        return all(isinstance(token_id, str) and token_id in live_token_ids for token_id in token_ids)
    if provenance == "paddleocr_vl_block":
        if not isinstance(block_ids, list) or not block_ids:
            return False
        return all(isinstance(block_id, str) and block_id in live_block_ids for block_id in block_ids)
    return False


def _item_field_matches_with_ocr(
    item: dict[str, Any],
    field: str,
    expected: str,
    live_token_ids: set[str],
    live_block_ids: set[str],
) -> bool:
    if not _field_value_matches(field, item.get(field), expected):
        return False
    evidence = item.get("field_evidence")
    if not isinstance(evidence, dict):
        return False
    return _field_has_ocr_evidence(evidence.get(field), item.get(field), live_token_ids, live_block_ids)


def _field_value_matches(field: str, actual: Any, expected: str) -> bool:
    if field == "meaning_ko":
        return meaning_matches(str(actual or ""), expected)
    return normalize_text(str(actual or "")) == normalize_text(expected)


def _expected_field_value(row: GoldenVocabRow, field: str) -> str:
    if field == "surface":
        return row.surface
    if field == "reading":
        return row.reading
    if field == "meaning_ko":
        return row.meaning_ko
    return ""


def _evidence_supports_value(evidence_text: str, expected_text: str) -> bool:
    if not expected_text:
        return False
    if _has_script(expected_text, "hangul"):
        return meaning_matches(evidence_text, expected_text)
    evidence_norm = normalize_text(evidence_text)
    expected_norm = normalize_text(expected_text)
    return bool(expected_norm and (expected_norm in evidence_norm or evidence_norm in expected_norm))


def _item_has_glossary_evidence(item: dict[str, Any]) -> bool:
    evidence = item.get("field_evidence")
    if not isinstance(evidence, dict):
        return False
    return any(isinstance(value, dict) and value.get("provenance") == "glossary" for value in evidence.values())


def _has_script(text: str, script: str) -> bool:
    return any(classify_script(char) == script for char in text)
