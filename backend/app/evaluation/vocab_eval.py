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
    matched_rows: int
    surface_reading_matches: int
    meaning_matches: int
    generated_cards: int
    korean_field_missing_hangul: int
    japanese_field_has_hangul: int
    missing_row_ids: list[str]

    @property
    def row_accuracy(self) -> float:
        return self.matched_rows / self.expected_rows if self.expected_rows else 0.0


def evaluate_vocab_page(golden: GoldenPage, process_result: ProcessResult) -> VocabEvalResult:
    items = _items_from_cards(process_result.cards)
    matched: set[str] = set()
    surface_reading_matches = 0
    meaning_match_count = 0

    for row in golden.expected_rows:
        candidates = [
            item
            for item in items
            if normalize_text(str(item.get("surface", ""))) == normalize_text(row.surface)
            and normalize_text(str(item.get("reading", ""))) == normalize_text(row.reading)
        ]
        if candidates:
            surface_reading_matches += 1
            if any(meaning_matches(str(item.get("meaning_ko", "")), row.meaning_ko) for item in candidates):
                matched.add(row.row_id)
                meaning_match_count += 1

    return VocabEvalResult(
        page_id=golden.page_id,
        image_path=str(golden.image_path),
        expected_page_type=golden.expected_page_type,
        actual_page_type=process_result.page.page_type,
        expected_rows=len(golden.expected_rows),
        extracted_items=len(items),
        matched_rows=len(matched),
        surface_reading_matches=surface_reading_matches,
        meaning_matches=meaning_match_count,
        generated_cards=len(process_result.cards),
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


def _has_script(text: str, script: str) -> bool:
    return any(classify_script(char) == script for char in text)
