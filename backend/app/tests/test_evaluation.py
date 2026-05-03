from __future__ import annotations

from pathlib import Path

from app.evaluation.golden import GoldenQuestion, GoldenPage, load_golden_pages, meaning_matches
from app.extraction.vocab import extract_vocab_items_dual_ocr
from app.models.schemas import Page, ProcessResult, OcrToken
from scripts import benchmark_ocr_modes, evaluate_golden


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


def test_benchmark_token_text_coverage_exposes_source_mismatches(tmp_path) -> None:
    golden = GoldenPage(
        page_id="mcq-page",
        image_path=tmp_path / "page.jpg",
        category="spelling_mcq",
        expected_page_type="spelling_mcq",
        expected_questions=[
            GoldenQuestion(
                question_id="q1",
                question_no=1,
                sentence="にわに しろい はなが さきました。",
                target="はな",
                choices=["木", "花", "犬", "山"],
                correct_choice_no=2,
                correct_answer="花",
                answer_source="answer_strip",
            )
        ],
    )
    process_result = ProcessResult(
        page=_page(tmp_path),
        tokens=[
            _token("tok-1", "さきました。", [10, 10, 70, 20], "hiragana"),
            _token("tok-2", "はなが", [80, 10, 120, 20], "hiragana"),
            _token("tok-3", "にわにしろい", [130, 10, 210, 20], "hiragana"),
            _token("tok-4", "はな", [10, 30, 40, 40], "hiragana"),
            _token("tok-5", "花", [10, 50, 40, 60], "kanji"),
        ],
        cards=[],
        script_summary={},
        answer_map={},
    )

    coverage = benchmark_ocr_modes._token_text_coverage(golden, process_result, "paddleocr")

    assert coverage.mode == "paddleocr_normalized_token_text"
    assert coverage.fields_expected == 7
    assert coverage.fields_matched == 3
    assert coverage.item_accuracy == 0.0


def test_from_db_evaluator_labels_persisted_vl_state(tmp_path) -> None:
    result = ProcessResult(
        page=_page(tmp_path, warnings=["Processed with PaddleOCR-VL; verify output."]),
        tokens=[_token("tok-vl", "学校", [10, 10, 60, 20], "kanji", source="paddleocr_vl")],
        cards=[],
        script_summary={},
        answer_map={},
    )

    assert evaluate_golden._persisted_engine_label(result) == "persisted_paddleocr_vl"


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


def _page(tmp_path: Path, warnings: list[str] | None = None) -> Page:
    return Page(
        id="page",
        original_image_path=str(tmp_path / "page.jpg"),
        upload_name="page.jpg",
        display_name="page",
        page_type="uploaded",
        page_type_confidence=0.0,
        warnings=warnings or [],
        created_at="2026-05-03T00:00:00+00:00",
    )
