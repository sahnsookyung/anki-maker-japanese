from __future__ import annotations

from pathlib import Path

import pytest

from app.db import database
from app.evaluation.golden import GoldenQuestion, GoldenPage, load_golden_pages, meaning_matches
from app.extraction import pipeline
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
    assert coverage.item_accuracy == pytest.approx(0.0)


def test_benchmark_google_vision_path_reports_text_coverage_and_resources(tmp_path, monkeypatch) -> None:
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
    process_result = ProcessResult(page=_page(tmp_path), tokens=[], cards=[], script_summary={}, answer_map={})
    monkeypatch.setattr(
        benchmark_ocr_modes,
        "recognize_with_provider",
        lambda image_path, page_id, provider: (
            [
                _token("google-1", "にわにしろいはながさきました。", [1, 1, 10, 10], "hiragana", source="google_vision"),
                _token("google-2", "はな", [1, 20, 10, 30], "hiragana", source="google_vision"),
                _token("google-3", "木 花 犬 山", [1, 40, 80, 50], "mixed", source="google_vision"),
            ],
            [],
        ),
    )

    result = benchmark_ocr_modes._run_google_vision_evaluation(golden, process_result, [])

    assert result["mode"] == "google_vision_ocr_text"
    assert result["token_count"] == 3
    assert result["ocr_text_coverage"]["field_accuracy"] == pytest.approx(1.0)
    assert result["resource_metrics"]["rss_samples"]


def test_from_db_evaluator_labels_persisted_vl_state(tmp_path) -> None:
    result = ProcessResult(
        page=_page(tmp_path, warnings=["Processed with PaddleOCR-VL; verify output."]),
        tokens=[_token("tok-vl", "学校", [10, 10, 60, 20], "kanji", source="paddleocr_vl")],
        cards=[],
        script_summary={},
        answer_map={},
    )

    assert evaluate_golden._persisted_engine_label(result) == "persisted_paddleocr_vl"


def test_from_db_run_id_evaluation_requires_matching_golden_page(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "run-id-eval.db")
    database.init_db()
    page = _page(tmp_path)
    database.upsert_page(page)
    run = database.start_ocr_run(page.id, "paddleocr_vl")
    database.replace_tokens(page.id, [_token("tok-vl", "学校", [10, 10, 60, 20], "kanji", source="paddleocr_vl")], run.id)
    database.complete_ocr_run(run.id)
    matching = GoldenPage(
        page_id="match",
        image_path=tmp_path / "page.jpg",
        category="vocab_table",
        expected_page_type="vocab_table",
    )
    mismatch = GoldenPage(
        page_id="mismatch",
        image_path=tmp_path / "different.jpg",
        category="vocab_table",
        expected_page_type="vocab_table",
    )

    assert evaluate_golden._process_result_from_db(matching, run.id) is not None
    assert evaluate_golden._process_result_from_db(mismatch, run.id) is None


def test_evaluate_runtime_uses_isolated_state_and_restores_globals(tmp_path) -> None:
    original_db_path = database.DB_PATH
    original_processed_dir = pipeline.PROCESSED_DIR
    work_dir = tmp_path / "eval"

    with evaluate_golden._evaluation_runtime(str(work_dir), keep_work_dir=True) as runtime_dir:
        assert runtime_dir == work_dir.resolve()
        assert database.DB_PATH == work_dir.resolve() / "evaluation.db"
        assert pipeline.PROCESSED_DIR == work_dir.resolve() / "processed"
        assert pipeline.PROCESSED_DIR.exists()

    assert database.DB_PATH == original_db_path
    assert pipeline.PROCESSED_DIR == original_processed_dir
    assert work_dir.exists()


def test_evaluate_runtime_cleans_default_temp_state() -> None:
    with evaluate_golden._evaluation_runtime("", keep_work_dir=False) as runtime_dir:
        assert runtime_dir.exists()
        temp_path = runtime_dir

    assert not temp_path.exists()


def test_evaluate_vl_processing_uses_bounded_worker(tmp_path, monkeypatch) -> None:
    page = _page(tmp_path)
    captured: dict[str, object] = {}

    def fake_worker(page_id: str, engine: str, **kwargs) -> ProcessResult:
        captured["page_id"] = page_id
        captured["engine"] = engine
        captured["max_rss_mb"] = kwargs["max_rss_mb"]
        captured["env_overrides"] = kwargs["env_overrides"]
        return ProcessResult(page=page, tokens=[], cards=[], script_summary={}, answer_map={})

    monkeypatch.setattr(evaluate_golden, "run_page_process_worker", fake_worker)

    result = evaluate_golden._process_page_for_evaluation(page, "paddleocr_vl")

    assert result.page.id == page.id
    assert captured == {
        "page_id": page.id,
        "engine": "paddleocr_vl",
        "max_rss_mb": evaluate_golden.OCR_VL_PAGE_WORKER_MAX_RSS_MB,
        "env_overrides": {
            "ANKI_MAKER_DB": str(database.DB_PATH),
            "ANKI_MAKER_PROCESSED_DIR": str(pipeline.PROCESSED_DIR),
        },
    }


def test_evaluate_base_processing_also_uses_isolated_worker(tmp_path, monkeypatch) -> None:
    page = _page(tmp_path)
    captured: dict[str, object] = {}

    def fake_worker(page_id: str, engine: str, **kwargs) -> ProcessResult:
        captured["page_id"] = page_id
        captured["engine"] = engine
        captured["max_rss_mb"] = kwargs["max_rss_mb"]
        captured["env_overrides"] = kwargs["env_overrides"]
        return ProcessResult(page=page, tokens=[], cards=[], script_summary={}, answer_map={})

    monkeypatch.setattr(evaluate_golden, "run_page_process_worker", fake_worker)

    result = evaluate_golden._process_page_for_evaluation(page, "paddleocr")

    assert result.page.id == page.id
    assert captured == {
        "page_id": page.id,
        "engine": "paddleocr",
        "max_rss_mb": evaluate_golden.OCR_PAGE_WORKER_MAX_RSS_MB,
        "env_overrides": {
            "ANKI_MAKER_DB": str(database.DB_PATH),
            "ANKI_MAKER_PROCESSED_DIR": str(pipeline.PROCESSED_DIR),
        },
    }


def test_evaluate_json_mode_redirects_ocr_chatter_to_stderr(tmp_path, monkeypatch, capsys) -> None:
    page = _page(tmp_path)

    def noisy_worker(page_id: str, engine: str, **kwargs) -> ProcessResult:
        print("paddle model loading chatter")
        return ProcessResult(page=page, tokens=[], cards=[], script_summary={}, answer_map={})

    monkeypatch.setattr(evaluate_golden, "run_page_process_worker", noisy_worker)

    evaluate_golden._process_page_for_evaluation(page, "paddleocr", redirect_logs=True)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "paddle model loading chatter" in captured.err


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
