from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.db import database
from app.evaluation.golden import GoldenQuestion, GoldenPage, GoldenVocabRow, load_golden_pages, meaning_matches
from app.evaluation.vocab_eval import evaluate_vocab_page
from app.extraction import pipeline
from app.extraction.document_graph import graph_from_tokens, graph_with_card_hypotheses
from app.extraction.vocab import extract_vocab_items_dual_ocr
from app.models.schemas import (
    CardCandidate,
    DocumentParseBlock,
    DocumentParseResult,
    FieldOcrPreviewResponse,
    OcrComparison,
    OcrRun,
    OcrToken,
    Page,
    ProcessResult,
)
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


def test_dual_ocr_extracts_vocab_entry_from_ocr_evidence_without_glossary_fill() -> None:
    japanese_tokens = [
        _token("jp1", "回会う", [450, 400, 505, 424], "mixed"),
        _token("jp2", "あう日", [570, 404, 650, 428], "mixed"),
        _token("jp3", "去とこのひと世", [575, 445, 720, 470], "mixed"),
        _token("jp4", "おんなのひと", [580, 480, 720, 505], "hiragana"),
    ]
    korean_tokens = [_token("ko1", "만나다", [565, 402, 650, 430], "hangul", source="paddleocr_korean")]

    items = extract_vocab_items_dual_ocr(japanese_tokens, korean_tokens)

    assert ("会う", "あう", "만나다") in [(item["surface"], item["reading"], item["meaning_ko"]) for item in items]
    complete_item = next(item for item in items if item["surface"] == "会う")
    assert all(complete_item["field_evidence"][field]["provenance"] == "ocr" for field in ("surface", "reading", "meaning_ko"))
    assert any("MISSING_KOREAN_MEANING" in item.get("warning_codes", []) for item in items)


def test_dual_ocr_keeps_kana_heavy_combined_surface_tokens() -> None:
    japanese_tokens = [_token("jp1", "あたらしい新しい", [450, 410, 590, 435], "mixed")]
    korean_tokens = [_token("ko1", "새롭다", [450, 410, 520, 435], "hangul", source="paddleocr_korean")]

    items = extract_vocab_items_dual_ocr(japanese_tokens, korean_tokens)

    assert [(item["surface"], item["reading"], item["meaning_ko"]) for item in items] == [
        ("新しい", "あたらしい", "새롭다")
    ]


def test_dual_ocr_preserves_duplicate_vocab_rows_for_benchmark_visibility() -> None:
    japanese_tokens = [
        _token("jp1", "学校", [100, 100, 140, 124], "kanji"),
        _token("jp2", "がっこう", [160, 100, 230, 124], "hiragana"),
        _token("jp3", "学校", [100, 150, 140, 174], "kanji"),
        _token("jp4", "がっこう", [160, 150, 230, 174], "hiragana"),
    ]
    korean_tokens = [
        _token("ko1", "학교", [250, 100, 300, 124], "hangul", source="paddleocr_korean"),
        _token("ko2", "학교", [250, 150, 300, 174], "hangul", source="paddleocr_korean"),
    ]

    items = extract_vocab_items_dual_ocr(japanese_tokens, korean_tokens)

    assert [(item["surface"], item["reading"], item["meaning_ko"]) for item in items] == [
        ("学校", "がっこう", "학교"),
        ("学校", "がっこう", "학교"),
    ]


def test_vocab_evaluation_requires_ocr_supported_fields_for_accuracy(tmp_path) -> None:
    golden = GoldenPage(
        page_id="vocab-page",
        image_path=tmp_path / "page.jpg",
        category="vocab_table",
        expected_page_type="vocab_table",
        expected_rows=[
            GoldenVocabRow(
                row_id="row-1",
                section="",
                column="left",
                surface="学校",
                reading="がっこう",
                meaning_ko="학교",
            )
        ],
    )
    card = CardCandidate(
        id="card-1",
        page_id="page",
        source_type="vocab_item",
        source_id="row-1",
        note_type="jp_vocab_entry",
        front="がっこう",
        back="学校",
        source={
            "surface": "学校",
            "reading": "がっこう",
            "meaning_ko": "학교",
            "field_evidence": {
                "surface": {"text": "学校", "provenance": "glossary"},
                "reading": {"text": "がっこう", "provenance": "ocr", "token_ids": ["reading"], "bbox": [1, 1, 2, 2]},
                "meaning_ko": {"text": "학교", "provenance": "ocr", "token_ids": ["meaning"], "bbox": [3, 3, 4, 4]},
            },
        },
        confidence=0.9,
        review_state="green",
    )
    result = evaluate_vocab_page(
        golden,
        ProcessResult(page=_page(tmp_path), tokens=[], cards=[card], script_summary={}, answer_map={}),
    )

    assert result.extracted_items == 1
    assert result.ocr_supported_items == 0
    assert result.glossary_supported_items == 1
    assert result.row_accuracy == pytest.approx(0.0)
    assert result.missing_row_ids == ["row-1"]


def test_vocab_evaluation_accepts_vl_block_evidence_without_glossary(tmp_path) -> None:
    golden = GoldenPage(
        page_id="vocab-page",
        image_path=tmp_path / "page.jpg",
        category="vocab_table",
        expected_page_type="vocab_table",
        expected_rows=[
            GoldenVocabRow(
                row_id="row-1",
                section="",
                column="left",
                surface="学校",
                reading="がっこう",
                meaning_ko="학교",
            )
        ],
    )
    block_evidence = {"provenance": "paddleocr_vl_block", "block_ids": ["block-1"], "bbox": [1, 1, 100, 20]}
    card = CardCandidate(
        id="card-1",
        page_id="page",
        source_type="vocab_item",
        source_id="row-1",
        note_type="jp_vocab_entry",
        front="がっこう",
        back="学校",
        source={
            "surface": "学校",
            "reading": "がっこう",
            "meaning_ko": "학교",
            "field_evidence": {
                "surface": {**block_evidence, "text": "学校"},
                "reading": {**block_evidence, "text": "がっこう"},
                "meaning_ko": {**block_evidence, "text": "학교"},
            },
        },
        confidence=0.78,
        review_state="green",
    )
    result = evaluate_vocab_page(
        golden,
        ProcessResult(
            page=_page(tmp_path),
            tokens=[],
            cards=[card],
            script_summary={},
            answer_map={},
            document_parse=DocumentParseResult(
                page_id="page",
                provider="paddleocr_vl",
                source_image_path=str(tmp_path / "page.jpg"),
                backend="fake",
                block_count=1,
                blocks=[DocumentParseBlock(id="block-1", label="text", content="学校 がっこう 학교", bbox=[1, 1, 100, 20])],
            ),
        ),
    )

    assert result.ocr_supported_items == 1
    assert result.glossary_supported_items == 0
    assert result.row_accuracy == pytest.approx(1.0)


def test_vocab_evaluation_accepts_crop_ocr_only_with_live_supporting_tokens(tmp_path) -> None:
    golden = GoldenPage(
        page_id="vocab-page",
        image_path=tmp_path / "page.jpg",
        category="vocab_table",
        expected_page_type="vocab_table",
        expected_rows=[
            GoldenVocabRow(row_id="row-1", section="", column="left", surface="学校", reading="がっこう", meaning_ko="학교")
        ],
    )
    tokens = [
        _token("surface", "学校", [1, 1, 20, 10], "kanji", source="crop_ocr"),
        _token("reading", "がっこう", [22, 1, 48, 10], "hiragana", source="crop_ocr"),
        _token("meaning", "학교", [50, 1, 70, 10], "hangul", source="crop_ocr"),
    ]
    card = _vocab_eval_card(
        {
            "surface": "学校",
            "reading": "がっこう",
            "meaning_ko": "학교",
            "field_evidence": {
                "surface": {"text": "学校", "provenance": "crop_ocr", "token_ids": ["surface"], "bbox": [1, 1, 20, 10]},
                "reading": {"text": "がっこう", "provenance": "crop_ocr", "token_ids": ["reading"], "bbox": [22, 1, 48, 10]},
                "meaning_ko": {"text": "학교", "provenance": "crop_ocr", "token_ids": ["meaning"], "bbox": [50, 1, 70, 10]},
            },
        }
    )

    result = evaluate_vocab_page(golden, ProcessResult(page=_page(tmp_path), tokens=tokens, cards=[card], script_summary={}))

    assert result.ocr_supported_items == 1
    assert result.row_accuracy == pytest.approx(1.0)


def test_vocab_evaluation_rejects_stale_or_unsupported_evidence(tmp_path) -> None:
    golden = GoldenPage(
        page_id="vocab-page",
        image_path=tmp_path / "page.jpg",
        category="vocab_table",
        expected_page_type="vocab_table",
        expected_rows=[
            GoldenVocabRow(row_id="row-1", section="", column="left", surface="学校", reading="がっこう", meaning_ko="학교")
        ],
    )
    card = _vocab_eval_card(
        {
            "surface": "学校",
            "reading": "がっこう",
            "meaning_ko": "학교",
            "field_evidence": {
                "surface": {"text": "先生", "provenance": "ocr", "token_ids": ["surface"], "bbox": [1, 1, 20, 10]},
                "reading": {"text": "がっこう", "provenance": "ocr", "token_ids": ["missing-token"], "bbox": [22, 1, 48, 10]},
                "meaning_ko": {"text": "학교", "provenance": "manual", "token_ids": ["meaning"], "bbox": [50, 1, 70, 10]},
            },
        }
    )
    tokens = [
        _token("surface", "学校", [1, 1, 20, 10], "kanji"),
        _token("meaning", "학교", [50, 1, 70, 10], "hangul"),
    ]

    result = evaluate_vocab_page(golden, ProcessResult(page=_page(tmp_path), tokens=tokens, cards=[card], script_summary={}))

    assert result.ocr_supported_items == 0
    assert result.row_accuracy == pytest.approx(0.0)


def test_vocab_evaluation_field_metrics_are_independent_of_full_row_support(tmp_path) -> None:
    golden = GoldenPage(
        page_id="vocab-page",
        image_path=tmp_path / "page.jpg",
        category="vocab_table",
        expected_page_type="vocab_table",
        expected_rows=[
            GoldenVocabRow(row_id="row-1", section="", column="left", surface="学校", reading="がっこう", meaning_ko="학교"),
            GoldenVocabRow(row_id="row-2", section="", column="left", surface="先生", reading="せんせい", meaning_ko="선생님"),
        ],
    )
    tokens = [
        _token("row1-surface", "学校", [1, 1, 20, 10], "kanji"),
        _token("row1-reading", "がっこう", [22, 1, 60, 10], "hiragana"),
        _token("row2-surface", "学生", [1, 20, 20, 30], "kanji"),
        _token("row2-reading", "せん", [22, 20, 60, 30], "hiragana"),
        _token("row2-meaning", "선생님", [62, 20, 90, 30], "hangul"),
    ]
    surface_reading_only = _vocab_eval_card(
        {
            "surface": "学校",
            "reading": "がっこう",
            "meaning_ko": "학교",
            "field_evidence": {
                "surface": {"text": "学校", "provenance": "ocr", "token_ids": ["row1-surface"], "bbox": [1, 1, 20, 10]},
                "reading": {"text": "がっこう", "provenance": "ocr", "token_ids": ["row1-reading"], "bbox": [22, 1, 60, 10]},
                "meaning_ko": {"text": "학교", "provenance": "glossary", "bbox": [62, 1, 90, 10]},
            },
        }
    )
    meaning_only = _vocab_eval_card(
        {
            "surface": "学生",
            "reading": "せん",
            "meaning_ko": "선생님",
            "field_evidence": {
                "surface": {"text": "学生", "provenance": "ocr", "token_ids": ["row2-surface"], "bbox": [1, 20, 20, 30]},
                "reading": {"text": "せん", "provenance": "ocr", "token_ids": ["row2-reading"], "bbox": [22, 20, 60, 30]},
                "meaning_ko": {"text": "선생님", "provenance": "ocr", "token_ids": ["row2-meaning"], "bbox": [62, 20, 90, 30]},
            },
        }
    ).model_copy(update={"id": "card-2", "source_id": "row-2"})

    result = evaluate_vocab_page(
        golden,
        ProcessResult(
            page=_page(tmp_path),
            tokens=tokens,
            cards=[surface_reading_only, meaning_only],
            script_summary={},
            answer_map={},
        ),
    )

    assert result.ocr_supported_items == 1
    assert result.surface_matches == 1
    assert result.reading_matches == 1
    assert result.surface_reading_matches == 1
    assert result.meaning_matches == 1
    assert result.matched_rows == 0
    assert result.surface_accuracy == pytest.approx(0.5)
    assert result.reading_accuracy == pytest.approx(0.5)
    assert result.meaning_accuracy == pytest.approx(0.5)
    assert result.row_accuracy == pytest.approx(0.0)


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


def test_benchmark_uses_vl_document_text_for_text_coverage(tmp_path) -> None:
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
        tokens=[],
        cards=[],
        script_summary={},
        answer_map={},
        document_parse=DocumentParseResult(
            page_id="page",
            provider="paddleocr_vl",
            source_image_path="page.png",
            backend="fake",
            block_count=1,
            blocks=[
                DocumentParseBlock(
                    id="block-1",
                    label="text",
                    content="1 にわに しろい はなが さきました。 1 木 2 花 3 犬 4 山",
                    bbox=[0, 0, 100, 100],
                )
            ],
        ),
    )

    coverage = benchmark_ocr_modes._token_text_coverage(golden, process_result, "paddleocr_vl")

    assert coverage.mode == "paddleocr_vl_document_text"
    assert coverage.field_accuracy == pytest.approx(1.0)
    assert coverage.item_accuracy == pytest.approx(1.0)


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


def test_document_graph_records_processed_coordinate_text_nodes() -> None:
    tokens = [
        _token("surface", "学校", [10, 20, 50, 40], "kanji"),
        _token("reading", "がっこう", [55, 20, 110, 40], "hiragana"),
    ]

    graph = graph_from_tokens("page-graph", tokens).model_dump()

    assert graph["schema_version"] == 1
    assert graph["transform"]["coordinate_space"] == "processed_image"
    assert graph["text_nodes"][0]["bbox"] == [10, 20, 50, 40]
    assert graph["line_nodes"][0]["token_ids"] == ["surface", "reading"]
    assert graph["metrics"]["bbox_coverage"] == pytest.approx(1.0)


def test_document_graph_records_cells_marks_and_card_hypotheses() -> None:
    tokens = [
        _token("mark", "□", [1, 1, 8, 8], "punctuation"),
        _token("surface", "学校", [12, 1, 32, 10], "kanji"),
        _token("reading", "がっこう", [40, 1, 80, 10], "hiragana"),
    ]
    card = _vocab_eval_card(
        {
            "surface": "学校",
            "reading": "がっこう",
            "meaning_ko": "학교",
            "field_evidence": {
                "surface": {"text": "学校", "provenance": "ocr", "token_ids": ["surface"], "bbox": [12, 1, 32, 10]},
                "reading": {"text": "がっこう", "provenance": "ocr", "token_ids": ["reading"], "bbox": [40, 1, 80, 10]},
            },
            "bbox": [1, 1, 90, 14],
        }
    )

    graph = graph_with_card_hypotheses(graph_from_tokens("page", tokens), [card]).model_dump()

    assert graph["table_cells"]
    assert graph["selection_marks"][0]["token_id"] == "mark"
    assert {field["field"] for field in graph["field_hypotheses"]} == {"surface", "reading"}
    assert graph["row_hypotheses"][0]["source_id"] == "row-1"
    assert graph["metrics"]["field_hypothesis_count"] == 2
    assert graph["metrics"]["evidence_alignment_score"] > 0


def test_benchmark_payload_includes_candidate_and_profile_metadata(tmp_path) -> None:
    page = _page(tmp_path)
    card = _vocab_eval_card(
        {
            "surface": "学校",
            "reading": "がっこう",
            "meaning_ko": "학교",
            "field_evidence": {},
        }
    )
    process_result = ProcessResult(page=page, tokens=[], cards=[card], script_summary={}, ocr_run=None)
    args = type("Args", (), {"model_profile": "jp_v5_mobile_general", "extraction_variant": "table_graph_v1", "engine": "paddleocr", "benchmark_mode": "fresh_cli"})()

    quality = benchmark_ocr_modes._quality_payload(process_result)
    manifest = benchmark_ocr_modes._benchmark_manifest(args, process_result)

    assert quality["candidate_recall_count"] == 1
    assert quality["failure_taxonomy"]["stale_or_missing_evidence"] == 1
    assert manifest["model_profile"] == "jp_v5_mobile_general"
    assert manifest["promotion_status"] == "experimental"
    assert "cache" in manifest


def test_profile_matrix_skips_heavy_profiles_unless_requested() -> None:
    safe_args = type("Args", (), {"profile_matrix": True, "include_heavy_profiles": False, "model_profile": "jp_v3_mobile_current"})()
    heavy_args = type("Args", (), {"profile_matrix": True, "include_heavy_profiles": True, "model_profile": "jp_v3_mobile_current"})()

    assert "jp_v5_server_general" not in benchmark_ocr_modes._profile_ids_for_run(safe_args)
    assert "jp_lang_auto" not in benchmark_ocr_modes._profile_ids_for_run(safe_args)
    assert "jp_v5_server_general" in benchmark_ocr_modes._profile_ids_for_run(heavy_args)
    assert "jp_lang_auto" in benchmark_ocr_modes._profile_ids_for_run(heavy_args)


def test_staged_benchmark_variants_are_pre_registered() -> None:
    stage_two = type(
        "Args",
        (),
        {
            "experiment_stage": "2",
            "stage_profiles": "",
            "include_heavy_profiles": False,
            "variant_matrix": False,
            "extraction_variant": "baseline_current",
            "model_profile": "jp_v3_mobile_current",
        },
    )()
    stage_three = type(
        "Args",
        (),
        {
            "experiment_stage": "3",
            "stage_profiles": "jp_v3_mobile_current,jp_v5_server_general",
            "include_heavy_profiles": False,
            "variant_matrix": False,
            "extraction_variant": "baseline_current",
            "model_profile": "jp_v3_mobile_current",
        },
    )()

    assert benchmark_ocr_modes._variant_ids_for_run(stage_two) == ["line_graph_v1", "table_graph_v1", "ranked_rows_v1"]
    assert benchmark_ocr_modes._profile_ids_for_run(stage_two) == ["jp_v3_mobile_current", "jp_v5_mobile_general"]
    assert benchmark_ocr_modes._variant_ids_for_run(stage_three) == ["crop_confirm_v1"]
    assert benchmark_ocr_modes._profile_ids_for_run(stage_three) == ["jp_v3_mobile_current"]


def test_staged_benchmark_skips_invalid_profile_ids() -> None:
    args = type(
        "Args",
        (),
        {
            "experiment_stage": "2",
            "stage_profiles": "missing_profile,jp_v3_mobile_current",
            "include_heavy_profiles": False,
            "variant_matrix": False,
            "extraction_variant": "baseline_current",
            "model_profile": "jp_v3_mobile_current",
        },
    )()

    assert benchmark_ocr_modes._profile_ids_for_run(args) == ["jp_v3_mobile_current"]


def test_extraction_variant_metrics_are_diagnostic_only() -> None:
    card = _vocab_eval_card(
        {
            "surface": "学校",
            "reading": "がっこう",
            "meaning_ko": "학교",
            "field_evidence": {
                "surface": {"text": "学校", "provenance": "ocr", "token_ids": ["surface"], "bbox": [1, 1, 20, 10]}
            },
        }
    ).model_copy(update={"warnings": ["Weak OCR evidence; verify this row manually."], "review_state": "yellow"})
    graph = graph_with_card_hypotheses(graph_from_tokens("page", [_token("surface", "学校", [1, 1, 20, 10], "kanji")]), [card])

    ranked = pipeline._extraction_variant_metrics("ranked_rows_v1", [card], graph.model_dump())
    crop = pipeline._extraction_variant_metrics("crop_confirm_v1", [card], graph.model_dump())
    agreement = pipeline._extraction_variant_metrics("provider_agreement_v1", [card], graph.model_dump())

    assert ranked["candidate_mutation"] is False
    assert ranked["ranked_rows"][0]["source_id"] == "row-1"
    assert crop["uncertain_fields"][0]["field"] == "surface"
    assert agreement["diagnostic_only"] is True
    assert agreement["automatic_extraction_decision"] is False


def test_crop_confirm_variant_runs_limited_crop_ocr_without_mutating_candidates(tmp_path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeCropWorker:
        def preview(self, **kwargs):
            calls.append(kwargs)
            return FieldOcrPreviewResponse(
                card_id=str(kwargs["card_id"]),
                page_id=str(kwargs["page_id"]),
                field=str(kwargs["field"]),
                bbox=list(kwargs["bbox"]),
                provider="paddle",
                text="学校",
                confidence=0.91,
                tokens=[_token("crop-token", "学校", [1, 1, 20, 10], "kanji")],
                field_evidence={"text": "学校", "bbox": list(kwargs["bbox"]), "provenance": "crop_ocr"},
            )

    monkeypatch.setattr(pipeline, "crop_ocr_worker", FakeCropWorker())
    monkeypatch.setattr(pipeline, "OCR_CROP_CONFIRM_MAX_FIELDS", 1)
    card = _vocab_eval_card(
        {
            "surface": "学校",
            "reading": "がっこう",
            "meaning_ko": "학교",
            "field_evidence": {
                "surface": {"text": "学校", "bbox": [1, 1, 40, 20], "token_ids": ["surface"], "provenance": "ocr"},
                "reading": {"text": "がっこう", "bbox": [45, 1, 90, 20], "token_ids": ["reading"], "provenance": "ocr"},
            },
        }
    ).model_copy(update={"warnings": ["Weak OCR evidence; verify this row manually."], "review_state": "yellow"})
    source_before = card.source.model_copy() if hasattr(card.source, "model_copy") else dict(card.source)

    diagnostics = pipeline._extraction_variant_diagnostics(
        "crop_confirm_v1",
        [card],
        tmp_path / "page.png",
        "page",
        200,
        100,
        [],
    )
    metrics = pipeline._extraction_variant_metrics(
        "crop_confirm_v1",
        [card],
        graph_with_card_hypotheses(graph_from_tokens("page", [_token("surface", "学校", [1, 1, 40, 20], "kanji")]), [card]).model_dump(),
        variant_diagnostics=diagnostics,
    )

    assert len(calls) == 1
    assert calls[0]["field"] == "surface"
    assert card.source == source_before
    confirmation = metrics["crop_confirmation"]
    assert confirmation["attempted"] == 1
    assert confirmation["candidate_mutation"] is False
    assert confirmation["results"][0]["text"] == "学校"
    assert confirmation["results"][0]["agrees_with_candidate"] is True


def test_provider_agreement_variant_records_review_only_signal(tmp_path, monkeypatch) -> None:
    def fake_compare(image_path, page_id, primary_tokens, compare_provider):
        return OcrComparison(
            primary_provider="paddleocr",
            compare_provider=compare_provider,
            primary_token_count=len(primary_tokens),
            compare_token_count=1,
            agreement=0.5,
            missing_from_primary=["학교"],
            missing_from_comparison=["学校"],
            warnings=["diagnostic only"],
        )

    monkeypatch.setattr(pipeline, "compare_ocr_tokens", fake_compare)
    monkeypatch.setattr(pipeline, "OCR_COMPARE_PROVIDER", "google_vision")
    card = _vocab_eval_card({"field_evidence": {}})
    diagnostics = pipeline._extraction_variant_diagnostics(
        "provider_agreement_v1",
        [card],
        tmp_path / "page.png",
        "page",
        200,
        100,
        [_token("surface", "学校", [1, 1, 40, 20], "kanji")],
    )
    metrics = pipeline._extraction_variant_metrics(
        "provider_agreement_v1",
        [card],
        graph_from_tokens("page", []).model_dump(),
        variant_diagnostics=diagnostics,
    )

    signal = metrics["provider_agreement"]
    assert signal["status"] == "ok"
    assert signal["agreement"] == pytest.approx(0.5)
    assert signal["automatic_extraction_decision"] is False
    assert signal["candidate_mutation"] is False


def test_benchmark_modes_reload_persisted_db_and_api_state(tmp_path, monkeypatch) -> None:
    from app.api import routes

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "parity.db")
    database.init_db()
    image_path = tmp_path / "page.jpg"
    image_path.write_bytes(b"fake")
    golden = GoldenPage(
        page_id="bench-page",
        image_path=image_path,
        category="vocab_table",
        expected_page_type="vocab_table",
        expected_rows=[
            GoldenVocabRow(
                row_id="row-1",
                section="",
                column="left",
                surface="学校",
                reading="がっこう",
                meaning_ko="학교",
            )
        ],
    )

    process_calls = 0

    def fake_process(page: Page, engine: str = "paddleocr", **_kwargs) -> ProcessResult:
        nonlocal process_calls
        process_calls += 1
        token_suffix = f"{page.id}-{process_calls}"
        tokens = [
            _token(f"surface-{token_suffix}", "学校", [1, 1, 20, 12], "kanji"),
            _token(f"reading-{token_suffix}", "がっこう", [24, 1, 55, 12], "hiragana"),
            _token(f"meaning-{token_suffix}", "학교", [60, 1, 90, 12], "hangul"),
        ]
        source = {
            "surface": "学校",
            "reading": "がっこう",
            "meaning_ko": "학교",
            "bbox": [1, 1, 90, 20],
            "field_evidence": {
                "surface": {"text": "学校", "bbox": [1, 1, 20, 12], "token_ids": [tokens[0].id], "provenance": "ocr"},
                "reading": {"text": "がっこう", "bbox": [24, 1, 55, 12], "token_ids": [tokens[1].id], "provenance": "ocr"},
                "meaning_ko": {"text": "학교", "bbox": [60, 1, 90, 12], "token_ids": [tokens[2].id], "provenance": "ocr"},
            },
        }
        card = _vocab_eval_card(source).model_copy(update={"id": f"card-{token_suffix}", "source_id": f"row-{page.id}"})
        run = database.start_ocr_run(page.id, engine, image_sha256="sha", provider_config={"cache_key": "cache"})
        processed = page.model_copy(update={"page_type": "vocab_table", "processed_image_path": str(image_path), "active_ocr_run_id": run.id})
        database.upsert_page(processed)
        database.replace_tokens(page.id, tokens, run.id)
        database.replace_cards(page.id, [card], run.id)
        completed = database.complete_ocr_run(run.id, metrics={"script_summary": {"kanji": 1}, "page_type": "vocab_table"})
        return ProcessResult(
            page=processed,
            tokens=tokens,
            cards=[card.model_copy(update={"page_id": page.id, "run_id": run.id})],
            script_summary={},
            ocr_run=completed,
        )

    def fake_base(golden_page: GoldenPage, _samples: list, engine: str, _profile: str, _variant: str) -> ProcessResult:
        page = Page(
            id="persisted-page",
            original_image_path=str(golden_page.image_path),
            upload_name=golden_page.image_path.name,
            display_name=golden_page.image_path.stem,
            page_type="uploaded",
            page_type_confidence=0,
            created_at="2026-05-03T00:00:00+00:00",
        )
        database.upsert_page(page)
        return fake_process(page, engine)

    monkeypatch.setattr(benchmark_ocr_modes, "_run_base_pipeline", fake_base)
    monkeypatch.setattr(routes, "process_page", fake_process)
    common = {"model_profile": "jp_v3_mobile_current", "extraction_variant": "baseline_current"}
    fresh_args = type("Args", (), {"benchmark_mode": "fresh_cli", **common})()
    persisted_args = type("Args", (), {"benchmark_mode": "persisted_db", **common})()
    api_args = type("Args", (), {"benchmark_mode": "ui_api", **common})()

    fresh_result = benchmark_ocr_modes._run_benchmark_pipeline(golden, [], "paddleocr", fresh_args)
    persisted_result = benchmark_ocr_modes._run_benchmark_pipeline(golden, [], "paddleocr", persisted_args)
    api_result = benchmark_ocr_modes._run_benchmark_pipeline(golden, [], "paddleocr", api_args)
    scores = [
        benchmark_ocr_modes._result_dict(benchmark_ocr_modes._evaluate_base(golden, result))["strict_ocr_score"]
        for result in (fresh_result, persisted_result, api_result)
    ]

    assert persisted_result.cards[0].source["surface"] == "学校"
    assert api_result.cards[0].source["surface"] == "学校"
    assert scores == pytest.approx([1.0, 1.0, 1.0])
    assert persisted_result.ocr_run is not None
    assert api_result.ocr_run is not None


def test_result_cache_lookup_is_keyed_by_image_and_profile_not_benchmark_page_id(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "cache.db")
    database.init_db()
    for page_id in ("page-a", "page-b"):
        database.upsert_page(
            Page(
                id=page_id,
                original_image_path=str(tmp_path / f"{page_id}.jpg"),
                upload_name=f"{page_id}.jpg",
                display_name=page_id,
                page_type="uploaded",
                page_type_confidence=0,
                created_at="2026-05-03T00:00:00+00:00",
            )
        )
    run = database.start_ocr_run("page-a", "paddleocr", image_sha256="same-image", provider_config={"cache_key": "profile-key"})
    database.complete_ocr_run(run.id)

    cached = database.find_succeeded_run_by_cache_key(None, "paddleocr", "same-image", "profile-key")

    assert cached is not None
    assert cached.id == run.id
    assert database.find_succeeded_run_by_cache_key("page-b", "paddleocr", "same-image", "profile-key") is None


def test_benchmark_audit_artifacts_include_overlay_json_and_png(tmp_path) -> None:
    from PIL import Image

    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 60), "white").save(image_path)
    page = _page(tmp_path).model_copy(update={"original_image_path": str(image_path), "processed_image_path": str(image_path)})
    result = ProcessResult(
        page=page,
        tokens=[_token("tok-1", "学校", [10, 10, 40, 20], "kanji")],
        cards=[_vocab_eval_card({"surface": "学校", "reading": "がっこう", "meaning_ko": "학교"})],
        script_summary={},
        ocr_run=OcrRun(
            id="run-audit",
            page_id=page.id,
            engine="paddleocr",
            status="succeeded",
            started_at="2026-05-03T00:00:00+00:00",
            metrics={
                "document_graph": {
                    "transform": {"coordinate_space": "processed_image", "processed_width": 100, "processed_height": 60},
                    "metrics": {"evidence_alignment_score": 0.5},
                }
            },
        ),
    )
    args = type(
        "Args",
        (),
        {
            "work_dir": str(tmp_path),
            "benchmark_mode": "fresh_cli",
            "engine": "paddleocr",
            "model_profile": "jp_v3_mobile_current",
            "extraction_variant": "baseline_current",
        },
    )()
    golden = GoldenPage(page_id="page", image_path=image_path, category="vocab_table", expected_page_type="vocab_table")

    artifacts = benchmark_ocr_modes._write_audit_artifacts(golden, result, args)

    assert Path(artifacts["overlay_json"]).exists()
    assert Path(artifacts["overlay_png"]).exists()
    overlay = json.loads(Path(artifacts["overlay_json"]).read_text(encoding="utf-8"))
    assert overlay["coordinate_space"] == "processed_image"
    assert overlay["transform"]["processed_width"] == 100
    assert overlay["document_graph_metrics"]["evidence_alignment_score"] == pytest.approx(0.5)


def test_benchmark_dashboard_markdown_summarizes_comparison_rows(tmp_path) -> None:
    dashboard = tmp_path / "dashboard.md"
    result = benchmark_ocr_modes.PageBenchmark(
        page_id="page-1",
        image_path=str(tmp_path / "page.jpg"),
        base={
            "matched": 1,
            "expected": 1,
            "accuracy": 1.0,
            "benchmark": {
                "mode": "fresh_cli",
                "model_profile": "jp_v3_mobile_current",
                "extraction_variant": "baseline_current",
                "document_graph_metrics": {"evidence_alignment_score": 0.75},
            },
        },
        vl=None,
        memory_samples=[],
        resource_metrics={
            "wall_seconds": 1.2,
            "peak_rss_mb": 42,
            "cache": {"result_cache_hit": False, "cache_phase": "cold_or_uncached"},
        },
        errors=[],
    )

    benchmark_ocr_modes._write_dashboard_markdown([result], dashboard)

    text = dashboard.read_text(encoding="utf-8")
    assert "OCR Benchmark Dashboard" in text
    assert "jp_v3_mobile_current" in text
    assert "75.0%" in text
    assert "cold_or_uncached" in text


def test_benchmark_artifacts_are_schema_versioned_and_label_cache_phase(tmp_path) -> None:
    result = benchmark_ocr_modes.PageBenchmark(
        page_id="page-1",
        image_path=str(tmp_path / "page.jpg"),
        base={"matched": 0, "expected": 0, "accuracy": 0.0},
        vl=None,
        memory_samples=[],
        resource_metrics={},
        errors=[],
    )
    summary = benchmark_ocr_modes._cache_summary(
        {"benchmark": {"cache": {"hit": False, "model_cache_hit": True, "key": "cache-key"}}}
    )

    assert result.schema_version == 1
    assert summary["cache_phase"] == "cold_or_uncached"
    assert summary["timing_bucket"] == "cold_or_uncached"
    assert summary["model_cache_hit"] is True


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
    assert captured["page_id"] == page.id
    assert captured["engine"] == "paddleocr_vl"
    assert captured["max_rss_mb"] == evaluate_golden.OCR_VL_PAGE_WORKER_MAX_RSS_MB
    assert captured["env_overrides"]["ANKI_MAKER_DB"] == str(database.DB_PATH)
    assert captured["env_overrides"]["ANKI_MAKER_PROCESSED_DIR"] == str(pipeline.PROCESSED_DIR)
    assert captured["env_overrides"]["PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME"] == "japan_PP-OCRv3_mobile_rec"


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
    assert captured["page_id"] == page.id
    assert captured["engine"] == "paddleocr"
    assert captured["max_rss_mb"] == evaluate_golden.OCR_PAGE_WORKER_MAX_RSS_MB
    assert captured["env_overrides"]["ANKI_MAKER_DB"] == str(database.DB_PATH)
    assert captured["env_overrides"]["ANKI_MAKER_PROCESSED_DIR"] == str(pipeline.PROCESSED_DIR)
    assert captured["env_overrides"]["PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME"] == "japan_PP-OCRv3_mobile_rec"


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


def _vocab_eval_card(source: dict) -> CardCandidate:
    return CardCandidate(
        id="card-1",
        page_id="page",
        source_type="vocab_item",
        source_id="row-1",
        note_type="jp_vocab_entry",
        front="front",
        back="back",
        source=source,
        confidence=0.9,
        review_state="green",
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
