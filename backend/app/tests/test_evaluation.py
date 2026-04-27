from __future__ import annotations

from pathlib import Path

from app.evaluation.golden import load_golden_pages, meaning_matches
from app.extraction.vocab import extract_vocab_items_dual_ocr
from app.models.schemas import OcrToken


def test_load_golden_pages_accepts_nested_expected_rows() -> None:
    pages = load_golden_pages(
        Path("../data/evaluation/golden_pages.example.json").resolve(),
        Path("..").resolve(),
    )

    assert len(pages) == 4
    assert pages[0].category == "vocab_table"
    assert len(pages[0].expected_rows) >= 30
    assert pages[0].expected_rows[0].surface == "間"
    assert pages[1].category == "reading_mcq"
    assert len(pages[1].expected_questions) == 10


def test_meaning_matches_normalized_korean_parts() -> None:
    assert meaning_matches("발 다리", "발, 다리")
    assert meaning_matches("북쪽출입구", "북쪽 출입구")


def test_dual_ocr_uses_glossary_and_filters_reading_noise() -> None:
    japanese_tokens = [
        _token("jp1", "回会う", [450, 400, 505, 424], "mixed"),
        _token("jp2", "あう日", [570, 404, 650, 428], "mixed"),
        _token("jp3", "去とこのひと世", [575, 445, 720, 470], "mixed"),
        _token("jp4", "おんなのひと", [580, 480, 720, 505], "hiragana"),
    ]
    korean_tokens = [_token("ko1", "만니다", [565, 402, 650, 430], "hangul", source="paddleocr_korean")]

    items = extract_vocab_items_dual_ocr(japanese_tokens, korean_tokens)

    assert [(item["surface"], item["reading"], item["meaning_ko"]) for item in items] == [
        ("会う", "あう", "만나다")
    ]


def _token(
    token_id: str,
    text: str,
    bbox: list[float],
    script_class: str,
    *,
    source: str = "paddleocr",
) -> OcrToken:
    return OcrToken(
        id=token_id,
        page_id="page",
        text=text,
        bbox=bbox,
        confidence=0.9,
        script_class=script_class,
        source=source,
    )
