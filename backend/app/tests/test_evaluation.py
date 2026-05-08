from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.db import database
from app.evaluation.golden import GoldenQuestion, GoldenPage, GoldenVocabRow, load_golden_pages, meaning_matches
from app.evaluation.mcq_eval import evaluate_mcq_page
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
from app.validation.dictionary import DictionaryValidator
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


def test_v5_token_split_variant_derives_field_bboxes_from_raw_token() -> None:
    japanese_tokens = [_token("jp1", "あたらしい新しい", [100, 100, 240, 124], "mixed")]
    korean_tokens = [_token("ko1", "새롭다", [260, 100, 330, 124], "hangul", source="paddleocr_korean")]

    [item] = extract_vocab_items_dual_ocr(japanese_tokens, korean_tokens, extraction_variant="v5_token_split_v1")

    surface_evidence = item["field_evidence"]["surface"]
    reading_evidence = item["field_evidence"]["reading"]
    assert item["surface"] == "新しい"
    assert item["reading"] == "あたらしい"
    assert reading_evidence["bbox"][2] < surface_evidence["bbox"][2]
    assert reading_evidence["derived_from_token_ids"] == ["jp1"]
    assert surface_evidence["bbox_strategy"] == "split_merged_vocab_token"


def test_row_and_korean_alignment_variants_are_guarded_from_candidate_mutation() -> None:
    japanese_tokens = [
        _token("jp1", "学校", [100, 100, 140, 124], "kanji"),
        _token("jp2", "がっこう", [150, 100, 220, 124], "hiragana"),
        _token("jp3", "先生", [100, 112, 140, 136], "kanji"),
        _token("jp4", "せんせい", [150, 112, 220, 136], "hiragana"),
    ]
    korean_tokens = [
        _token("ko1", "학교", [250, 100, 300, 124], "hangul", source="paddleocr_korean"),
        _token("ko2", "오답", [250, 142, 300, 166], "hangul", source="paddleocr_korean"),
    ]

    baseline = extract_vocab_items_dual_ocr(japanese_tokens, korean_tokens)
    row_variant = extract_vocab_items_dual_ocr(japanese_tokens, korean_tokens, extraction_variant="v5_vocab_rows_v1")
    korean_variant = extract_vocab_items_dual_ocr(japanese_tokens, korean_tokens, extraction_variant="ko_alignment_v1")

    assert [(item["surface"], item["reading"], item["meaning_ko"]) for item in row_variant] == [
        (item["surface"], item["reading"], item["meaning_ko"]) for item in baseline
    ]
    assert [(item["surface"], item["reading"], item["meaning_ko"]) for item in korean_variant] == [
        (item["surface"], item["reading"], item["meaning_ko"]) for item in baseline
    ]


def test_v5_token_split_does_not_split_single_kana_honorific_surface() -> None:
    japanese_tokens = [
        _token("jp1", "お父さん", [100, 100, 180, 124], "mixed"),
        _token("jp2", "おとうさん", [200, 100, 300, 124], "hiragana"),
    ]
    korean_tokens = [_token("ko1", "아버지", [320, 100, 390, 124], "hangul", source="paddleocr_korean")]

    items = extract_vocab_items_dual_ocr(japanese_tokens, korean_tokens, extraction_variant="v5_token_split_v1")

    assert [(item["surface"], item["reading"], item["meaning_ko"]) for item in items] == [
        ("お父さん", "おとうさん", "아버지")
    ]
    assert items[0]["field_evidence"]["surface"]["text"] == "お父さん"


def test_dual_ocr_allows_pure_katakana_vocab_surfaces() -> None:
    japanese_tokens = [
        _token("jp1", "エアコン", [100, 100, 180, 124], "katakana"),
        _token("jp2", "えあこん", [200, 100, 300, 124], "hiragana"),
    ]
    korean_tokens = [_token("ko1", "에어컨", [320, 100, 390, 124], "hangul", source="paddleocr_korean")]

    items = extract_vocab_items_dual_ocr(japanese_tokens, korean_tokens, extraction_variant="v5_token_split_v1")

    assert [(item["surface"], item["reading"], item["meaning_ko"]) for item in items] == [
        ("エアコン", "えあこん", "에어컨")
    ]


def test_v5_reading_match_penalizes_cross_column_boundary_tokens() -> None:
    japanese_tokens = [
        _token("jp1", "書く丛C", [282, 100, 374, 124], "mixed"),
        _token("jp2", "かく", [71, 100, 149, 124], "hiragana"),
        _token("jp3", "がくせい", [496, 100, 620, 124], "hiragana"),
        _token("jp4", "学生", [650, 100, 730, 124], "kanji"),
    ]
    korean_tokens = [
        _token("ko1", "<쓰다", [275, 100, 380, 124], "hangul", source="paddleocr_korean"),
        _token("ko2", "학생", [650, 100, 730, 124], "hangul", source="paddleocr_korean"),
    ]

    items = extract_vocab_items_dual_ocr(japanese_tokens, korean_tokens, extraction_variant="v5_token_split_v1")

    assert ("書く丛", "かく", "쓰다") in [(item["surface"], item["reading"], item["meaning_ko"]) for item in items]


def test_v5_reading_extraction_uses_trailing_hiragana_from_noisy_token() -> None:
    japanese_tokens = [
        _token("jp1", "0ちさロえあこん", [323, 100, 612, 124], "mixed"),
        _token("jp2", "エアコンI0", [671, 100, 821, 124], "mixed"),
    ]
    korean_tokens = [_token("ko1", "L그√에어컨", [670, 100, 824, 124], "mixed", source="paddleocr_korean")]

    items = extract_vocab_items_dual_ocr(japanese_tokens, korean_tokens, extraction_variant="v5_token_split_v1")

    assert [(item["surface"], item["reading"], item["meaning_ko"]) for item in items] == [
        ("エアコン", "えあこん", "그에어컨")
    ]


def test_v5_token_split_ignores_previous_row_reading_before_delimited_surface(tmp_path: Path) -> None:
    dictionary_path = tmp_path / "dict.json"
    dictionary_path.write_text(json.dumps([{"surface": "金よう日", "reading": "きんようび"}]), encoding="utf-8")
    japanese_tokens = [
        _token("jp1", "きゅうせんえん口金よう日", [300, 100, 620, 124], "mixed"),
        _token("jp2", "きんようび$9", [660, 100, 820, 124], "mixed"),
    ]
    korean_tokens = [_token("ko1", "금요일", [660, 100, 800, 124], "hangul", source="paddleocr_korean")]

    items = extract_vocab_items_dual_ocr(
        japanese_tokens,
        korean_tokens,
        DictionaryValidator(dictionary_path),
        extraction_variant="v5_token_split_v1",
    )

    item = next(item for item in items if item["surface"] == "金よう日")
    assert item["reading"] == "きんようび"
    assert item["meaning_ko"] == "금요일"
    assert item["field_evidence"]["reading"]["token_ids"] == ["jp2"]


def test_v5_korean_alignment_can_use_much_closer_cross_column_merged_meaning() -> None:
    japanese_tokens = [
        _token("jp-left-surface", "日北がわ", [100, 100, 180, 124], "mixed"),
        _token("jp-left-reading", "きたがわ", [240, 101, 340, 125], "hiragana"),
        _token("jp-right-surface", "日北口", [500, 100, 560, 124], "kanji"),
        _token("jp-right-reading", "きたぐち", [620, 101, 720, 125], "hiragana"),
    ]
    korean_tokens = [
        _token("ko-same-column-previous-row", "화요일", [245, 130, 320, 154], "hangul", source="paddleocr_korean"),
        _token("ko-cross-column-previous-row", "강", [620, 96, 680, 120], "hangul", source="paddleocr_korean"),
        _token("ko-cross-column-merged", "북쪽출입구", [620, 108, 780, 132], "hangul", source="paddleocr_korean"),
    ]

    baseline = extract_vocab_items_dual_ocr(japanese_tokens, korean_tokens)
    adapted = extract_vocab_items_dual_ocr(japanese_tokens, korean_tokens, extraction_variant="v5_token_split_v1")

    baseline_left = next(item for item in baseline if item["surface"] == "北がわ")
    adapted_left = next(item for item in adapted if item["surface"] == "北がわ")
    assert baseline_left["meaning_ko"] == "화요일"
    assert adapted_left["meaning_ko"] == "북쪽출입구"


def test_v5_dictionary_refinement_trims_surface_noise_and_normalizes_yoon_reading(tmp_path: Path) -> None:
    dictionary_path = tmp_path / "dict.json"
    dictionary_path.write_text(
        json.dumps([{"surface": "今週", "reading": "こんしゅう"}, {"surface": "午後", "reading": "ごご"}]),
        encoding="utf-8",
    )
    japanese_tokens = [
        _token("jp1", "こんしゆう今週", [100, 100, 260, 124], "mixed"),
        _token("jp2", "午後 章", [100, 150, 220, 174], "kanji"),
        _token("jp3", "ごご", [20, 150, 80, 174], "hiragana"),
    ]
    korean_tokens = [
        _token("ko1", "이번주", [280, 100, 360, 124], "hangul", source="paddleocr_korean"),
        _token("ko2", "오후", [280, 150, 360, 174], "hangul", source="paddleocr_korean"),
    ]

    items = extract_vocab_items_dual_ocr(
        japanese_tokens,
        korean_tokens,
        DictionaryValidator(dictionary_path),
        extraction_variant="v5_token_split_v1",
    )

    assert ("今週", "こんしゅう", "이번주") in [(item["surface"], item["reading"], item["meaning_ko"]) for item in items]
    assert ("午後", "ごご", "오후") in [(item["surface"], item["reading"], item["meaning_ko"]) for item in items]
    refined_item = next(item for item in items if item["surface"] == "今週")
    assert refined_item["field_evidence"]["reading"]["normalization_strategy"] == "v5_ocr_dictionary_refinement"


def test_v5_reading_cleanup_strips_katakana_box_marker(tmp_path: Path) -> None:
    dictionary_path = tmp_path / "dict.json"
    dictionary_path.write_text(json.dumps([{"surface": "午前", "reading": "ごぜん"}]), encoding="utf-8")
    japanese_tokens = [
        _token("jp1", "ロごぜん", [20, 100, 100, 124], "mixed"),
        _token("jp2", "午前", [140, 100, 200, 124], "kanji"),
    ]
    korean_tokens = [_token("ko1", "오전", [240, 100, 300, 124], "hangul", source="paddleocr_korean")]

    items = extract_vocab_items_dual_ocr(
        japanese_tokens,
        korean_tokens,
        DictionaryValidator(dictionary_path),
        extraction_variant="v5_token_split_v1",
    )

    assert [(item["surface"], item["reading"], item["meaning_ko"]) for item in items] == [("午前", "ごぜん", "오전")]


def test_v5_reading_match_prefers_hiragana_over_katakana_noise(tmp_path: Path) -> None:
    dictionary_path = tmp_path / "dict.json"
    dictionary_path.write_text(json.dumps([{"surface": "多い", "reading": "おおい"}]), encoding="utf-8")
    japanese_tokens = [
        _token("jp1", "口おおい", [70, 100, 150, 124], "mixed"),
        _token("jp2", "多い", [280, 100, 340, 124], "kanji"),
        _token("jp3", "シリイ", [380, 100, 450, 124], "katakana"),
        _token("jp4", "会社", [700, 100, 760, 124], "kanji"),
        _token("jp5", "学校", [780, 140, 840, 164], "kanji"),
    ]
    korean_tokens = [_token("ko1", "많다", [300, 100, 360, 124], "hangul", source="paddleocr_korean")]

    items = extract_vocab_items_dual_ocr(
        japanese_tokens,
        korean_tokens,
        DictionaryValidator(dictionary_path),
        extraction_variant="v5_token_split_v1",
    )

    assert ("多い", "おおい", "많다") in [(item["surface"], item["reading"], item["meaning_ko"]) for item in items]


def test_v5_reading_match_prefers_non_crossing_same_column_reading(tmp_path: Path) -> None:
    dictionary_path = tmp_path / "dict.json"
    dictionary_path.write_text(json.dumps([{"surface": "五日", "reading": "いつか"}]), encoding="utf-8")
    japanese_tokens = [
        _token("jp1", "口いつか", [70, 100, 150, 124], "mixed"),
        _token("jp2", "五日", [280, 100, 340, 124], "kanji"),
        _token("jp3", "三ホコロいま", [380, 100, 620, 124], "mixed"),
        _token("jp4", "会社", [700, 100, 760, 124], "kanji"),
        _token("jp5", "学校", [780, 140, 840, 164], "kanji"),
    ]
    korean_tokens = [_token("ko1", "5일", [300, 100, 360, 124], "hangul", source="paddleocr_korean")]

    items = extract_vocab_items_dual_ocr(
        japanese_tokens,
        korean_tokens,
        DictionaryValidator(dictionary_path),
        extraction_variant="v5_token_split_v1",
    )

    assert ("五日", "いつか", "5일") in [(item["surface"], item["reading"], item["meaning_ko"]) for item in items]


def test_v5_dictionary_refinement_trims_reading_suffix_noise(tmp_path: Path) -> None:
    dictionary_path = tmp_path / "dict.json"
    dictionary_path.write_text(json.dumps([{"surface": "お国", "reading": "おくに"}]), encoding="utf-8")
    japanese_tokens = [
        _token("jp1", "お国", [100, 100, 180, 124], "mixed"),
        _token("jp2", "おくには誌", [200, 100, 310, 124], "mixed"),
    ]
    korean_tokens = [_token("ko1", "(남의)고향", [330, 100, 430, 124], "hangul", source="paddleocr_korean")]

    items = extract_vocab_items_dual_ocr(
        japanese_tokens,
        korean_tokens,
        DictionaryValidator(dictionary_path),
        extraction_variant="v5_token_split_v1",
    )

    assert ("お国", "おくに", "(남의)고향") in [(item["surface"], item["reading"], item["meaning_ko"]) for item in items]


def test_v5_reading_cleanup_uses_internal_hiragana_run_when_dictionary_validates(tmp_path: Path) -> None:
    dictionary_path = tmp_path / "dict.json"
    dictionary_path.write_text(json.dumps([{"surface": "男の人", "reading": "おとこのひと"}]), encoding="utf-8")
    japanese_tokens = [
        _token("jp1", "日男の人", [500, 100, 600, 124], "mixed"),
        _token("jp2", "去とこのひと世", [640, 100, 800, 124], "mixed"),
    ]
    korean_tokens = [_token("ko1", "Z이남자", [640, 100, 800, 124], "mixed", source="paddleocr_korean")]

    items = extract_vocab_items_dual_ocr(
        japanese_tokens,
        korean_tokens,
        DictionaryValidator(dictionary_path),
        extraction_variant="v5_token_split_v1",
    )

    assert ("男の人", "おとこのひと", "이남자") in [
        (item["surface"], item["reading"], item["meaning_ko"]) for item in items
    ]
    item = next(item for item in items if item["surface"] == "男の人")
    assert item["field_evidence"]["reading"]["normalization_strategy"] == "v5_ocr_dictionary_refinement"


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


def test_benchmark_vocab_miss_analysis_classifies_failure_causes(tmp_path) -> None:
    golden = GoldenPage(
        page_id="vocab-page",
        image_path=tmp_path / "page.jpg",
        category="vocab_table",
        expected_page_type="vocab_table",
        expected_rows=[
            GoldenVocabRow(row_id="row-1", section="", column="left", surface="五つ", reading="いつつ", meaning_ko="5개"),
            GoldenVocabRow(row_id="row-2", section="", column="left", surface="英語", reading="えいご", meaning_ko="영어"),
            GoldenVocabRow(row_id="row-3", section="", column="right", surface="学校", reading="がっこう", meaning_ko="학교"),
        ],
    )
    tokens = [
        _token("surface-1", "五つ", [1, 1, 20, 10], "kanji"),
        _token("reading-1", "いつつ", [22, 1, 60, 10], "hiragana"),
        _token("meaning-1", "5", [62, 1, 75, 10], "number", source="paddleocr_korean"),
        _token("reading-2", "口えいご", [22, 20, 70, 30], "mixed"),
        _token("meaning-2", "영어", [72, 20, 95, 30], "hangul", source="paddleocr_korean"),
        _token("surface-3", "学校", [1, 40, 20, 50], "kanji"),
        _token("reading-3", "がっこう", [22, 40, 60, 50], "hiragana"),
        _token("meaning-3-expected", "학교", [62, 40, 90, 50], "hangul", source="paddleocr_korean"),
        _token("meaning-3-paired", "선생", [92, 40, 120, 50], "hangul", source="paddleocr_korean"),
    ]
    cards = [
        _vocab_eval_card(
            {
                "surface": "五つ",
                "reading": "いつつ",
                "meaning_ko": "5",
                "field_evidence": {
                    "surface": {"text": "五つ", "provenance": "ocr", "token_ids": ["surface-1"], "bbox": [1, 1, 20, 10]},
                    "reading": {"text": "いつつ", "provenance": "ocr", "token_ids": ["reading-1"], "bbox": [22, 1, 60, 10]},
                    "meaning_ko": {"text": "5", "provenance": "ocr", "token_ids": ["meaning-1"], "bbox": [62, 1, 75, 10]},
                },
            }
        ),
        _vocab_eval_card(
            {
                "surface": "学校",
                "reading": "がっこう",
                "meaning_ko": "선생",
                "field_evidence": {
                    "surface": {"text": "学校", "provenance": "ocr", "token_ids": ["surface-3"], "bbox": [1, 40, 20, 50]},
                    "reading": {"text": "がっこう", "provenance": "ocr", "token_ids": ["reading-3"], "bbox": [22, 40, 60, 50]},
                    "meaning_ko": {"text": "선생", "provenance": "ocr", "token_ids": ["meaning-3-paired"], "bbox": [92, 40, 120, 50]},
                },
            }
        ).model_copy(update={"id": "card-2", "source_id": "row-3"}),
    ]
    process_result = ProcessResult(page=_page(tmp_path), tokens=tokens, cards=cards, script_summary={})
    eval_result = evaluate_vocab_page(golden, process_result)

    analysis = benchmark_ocr_modes._miss_analysis(golden, process_result, eval_result)

    assert analysis["counts"] == {
        "wrong_pairing": 1,
        "surface_ocr_error": 1,
        "korean_ocr_error": 1,
    }
    reasons = {row["row_id"]: row["reason"] for row in analysis["rows"]}
    assert reasons == {
        "row-1": "korean_ocr_error",
        "row-2": "surface_ocr_error",
        "row-3": "wrong_pairing",
    }
    assert analysis["rows"][0]["best_candidate"]["field_matches"]["surface"] is True


def test_benchmark_mcq_miss_analysis_lists_source_field_errors(tmp_path) -> None:
    golden = GoldenPage(
        page_id="mcq-page",
        image_path=tmp_path / "page.jpg",
        category="reading_mcq",
        expected_page_type="reading_mcq",
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
    card = CardCandidate(
        id="card-q1",
        page_id="page",
        source_type="question_item",
        source_id="q1",
        note_type="jp_reading_mcq_recall",
        front="front",
        back="back",
        source={
            "question_no": 1,
            "sentence": "にわに しろい はな が",
            "target": "はな",
            "choices": ["木", "化", "犬", "山"],
            "correct_choice_no": 2,
            "correct_answer": "花",
        },
    )
    process_result = ProcessResult(page=_page(tmp_path), tokens=[], cards=[card], script_summary={})
    eval_result = evaluate_mcq_page(golden, process_result)

    analysis = benchmark_ocr_modes._miss_analysis(golden, process_result, eval_result)

    assert analysis["counts"] == {"source_field_ocr_error": 2, "source_question_mismatch": 1}
    assert analysis["field_error_counts"] == {"sentence": 1, "choices": 1}
    assert analysis["rows"] == [
        {
            "question_id": "q1",
            "question_no": 1,
            "reason": "source_field_ocr_error",
            "field_errors": ["sentence", "choices"],
            "field_matches": {
                "sentence": False,
                "target": True,
                "choices": False,
                "correct_answer": True,
                "correct_choice_no": True,
            },
            "expected": {
                "sentence": "にわに しろい はなが さきました。",
                "target": "はな",
                "choices": ["木", "花", "犬", "山"],
                "correct_answer": "花",
                "correct_choice_no": 2,
            },
            "actual": {
                "sentence": "にわに しろい はな が",
                "target": "はな",
                "choices": ["木", "化", "犬", "山"],
                "correct_answer": "花",
                "correct_choice_no": 2,
            },
        }
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
    process_result = ProcessResult(
        page=page,
        tokens=[],
        cards=[card],
        script_summary={},
        ocr_run=OcrRun(
            id="run-benchmark",
            page_id=page.id,
            engine="paddleocr",
            status="succeeded",
            started_at="2026-05-03T00:00:00+00:00",
            metrics={
                "extraction_variant_metrics": {
                    "variant": "table_graph_v1",
                    "candidate_mutation": False,
                }
            },
        ),
    )
    args = type(
        "Args",
        (),
        {
            "model_profile": "jp_v5_mobile_general",
            "korean_profile": "ko_v5_current",
            "extraction_variant": "table_graph_v1",
            "engine": "paddleocr",
            "benchmark_mode": "fresh_cli",
        },
    )()

    quality = benchmark_ocr_modes._quality_payload(process_result)
    manifest = benchmark_ocr_modes._benchmark_manifest(args, process_result)

    assert quality["candidate_recall_count"] == 1
    assert quality["failure_taxonomy"]["stale_or_missing_evidence"] == 1
    assert manifest["model_profile"] == "jp_v5_mobile_general"
    assert manifest["korean_profile"] == "ko_v5_current"
    assert manifest["extraction_variant_metrics"]["variant"] == "table_graph_v1"
    assert manifest["promotion_status"] == "experimental"
    assert "cache" in manifest


def test_profile_matrix_skips_heavy_profiles_unless_requested() -> None:
    safe_args = type("Args", (), {"profile_matrix": True, "include_heavy_profiles": False, "model_profile": "jp_v3_mobile_current", "experiment_stage": ""})()
    heavy_args = type("Args", (), {"profile_matrix": True, "include_heavy_profiles": True, "model_profile": "jp_v3_mobile_current", "experiment_stage": ""})()

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

    assert benchmark_ocr_modes._variant_ids_for_run(stage_two) == [
        "v5_token_split_v1",
        "v5_vocab_rows_v1",
        "ko_alignment_v1",
        "v5_mcq_v1",
        "ko_crop_confirm_v1",
        "ko_region_columns_v1",
        "mcq_source_rebuild_v1",
        "mcq_choice_band_ocr_v1",
        "jp_region_columns_v1",
        "ko_residual_glyph_v1",
        "mcq_prompt_line_ocr_v1",
        "mcq_choice_glyph_v1",
    ]
    assert benchmark_ocr_modes._profile_ids_for_run(stage_two) == ["jp_v3_det_v3_rec", "jp_v3_det_v5_rec", "jp_v5_det_v3_rec", "jp_v5_det_v5_rec"]
    assert benchmark_ocr_modes._variant_ids_for_run(stage_three) == [
        "v5_token_split_plus_vocab_rows_v1",
        "v5_vocab_rows_plus_ko_alignment_v1",
        "v5_token_split_plus_mcq_v1",
        "v5_full_adapted_v1",
        "ko_consensus_v1",
        "accuracy_recovery_v1",
        "accuracy_recovery_v2",
    ]
    assert benchmark_ocr_modes._profile_ids_for_run(stage_three) == ["jp_v3_mobile_current"]


def test_accuracy_recovery_v2_metrics_schema_aggregates_components_and_cache() -> None:
    recovery = pipeline._append_recovery_diagnostics(
        {},
        "japanese_vocab_recovery",
        {
            "schema_version": 1,
            "kind": "japanese_vocab_recovery",
            "attempted": 2,
            "accepted": 1,
            "counts": {"jp_recovered_missing_row": 1},
            "attempts": [{"cache": {"hit": False}}],
        },
    )
    recovery = pipeline._append_recovery_diagnostics(
        recovery,
        "mcq_prompt_line_recovery",
        {
            "schema_version": 1,
            "kind": "mcq_prompt_line_recovery",
            "attempted": 1,
            "accepted": 1,
            "counts": {"prompt_line_accepted": 1, "prompt_line_resource_cap": 1},
            "attempts": [{"candidates": [{"cache": {"hit": True}}]}],
        },
    )
    recovery = pipeline._append_recovery_diagnostics(
        recovery,
        "v2_mcq_source_recovery",
        {
            "schema_version": 2,
            "kind": "accuracy_recovery_v2",
            "components": {
                "mcq_choice_glyph_recovery": {
                    "attempted": 4,
                    "accepted": 2,
                    "counts": {"choice_glyph_accepted": 2},
                    "attempts": [{"cache": {"hit": True}}],
                }
            },
        },
    )
    recovery = pipeline._append_recovery_diagnostics(
        recovery,
        "korean_residual_glyph_recovery",
        {
            "schema_version": 1,
            "kind": "korean_residual_glyph_recovery",
            "attempted": 1,
            "accepted": 1,
            "counts": {"ko_glyph_accepted": 1},
            "attempts": [{"cache": {"hit": False}}],
        },
    )

    assert recovery["schema_version"] == 2
    assert recovery["kind"] == "accuracy_recovery_v2"
    assert recovery["attempted"] == 8
    assert recovery["accepted"] == 5
    assert recovery["counts"]["jp_recovered_missing_row"] == 1
    assert recovery["counts"]["prompt_line_accepted"] == 1
    assert recovery["counts"]["choice_glyph_accepted"] == 2
    assert recovery["counts"]["ko_glyph_accepted"] == 1
    assert recovery["resource_caps"] == {"prompt_line_resource_cap": 1}
    assert recovery["cache"] == {"hits": 2, "misses": 2}
    assert set(recovery["components"]) == {"japanese_vocab_recovery", "korean_recovery", "mcq_prompt_line_recovery", "mcq_choice_glyph_recovery"}


def test_japanese_recovery_does_not_replace_valid_low_confidence_fields() -> None:
    item = {
        "surface": "英語",
        "reading": "えいご",
        "field_evidence": {
            "surface": {"confidence": 0.31, "bbox": [10, 10, 30, 30]},
            "reading": {"confidence": 0.31, "bbox": [32, 10, 70, 30]},
        },
    }

    assert pipeline._jp_field_needs_recovery(item, "surface") is False
    assert pipeline._jp_field_needs_recovery(item, "reading") is False
    assert pipeline._jp_field_needs_recovery({**item, "surface": "□"}, "surface") is True
    assert pipeline._jp_field_needs_recovery({**item, "reading": "英語"}, "reading") is True


def test_missing_vocab_row_recovery_requires_complete_live_region_evidence(monkeypatch, tmp_path) -> None:
    existing_items = [
        {
            "id": "row-left-existing",
            "column": "left",
            "row_bbox": [100, 96, 270, 124],
            "field_evidence": {
                "surface": {"bbox": [100, 100, 140, 120], "token_ids": ["l-s"]},
                "reading": {"bbox": [150, 100, 210, 120], "token_ids": ["l-r"]},
                "meaning_ko": {"bbox": [220, 100, 270, 120], "token_ids": ["l-m"]},
            },
            "evidence_tokens": ["l-s", "l-r", "l-m"],
        },
        {
            "id": "row-right-existing",
            "column": "right",
            "row_bbox": [500, 96, 690, 124],
            "field_evidence": {
                "surface": {"bbox": [500, 100, 540, 120], "token_ids": ["r-s"]},
                "reading": {"bbox": [550, 100, 610, 120], "token_ids": ["r-r"]},
                "meaning_ko": {"bbox": [620, 100, 690, 120], "token_ids": ["r-m"]},
            },
            "evidence_tokens": ["r-s", "r-r", "r-m"],
        },
    ]
    all_tokens = [
        _token("l-s", "学校", [100, 100, 140, 120], "kanji"),
        _token("l-r", "がっこう", [150, 100, 210, 120], "hiragana"),
        _token("r-s", "先生", [500, 100, 540, 120], "kanji"),
        _token("r-r", "せんせい", [550, 100, 610, 120], "hiragana"),
        _token("missing-left-reading", "えいご", [150, 150, 210, 170], "hiragana"),
        _token("missing-left-meaning", "영어", [220, 150, 270, 170], "hangul", source="paddleocr_korean"),
        _token("missing-right-reading", "えれべえたあ", [550, 150, 630, 170], "hiragana"),
        _token("missing-right-meaning", "엘리베이터", [620, 150, 700, 170], "hangul", source="paddleocr_korean"),
    ]

    def fake_recognize_region(**kwargs):
        bbox = kwargs["bbox"]
        center_x = (bbox[0] + bbox[2]) / 2
        column = "left" if center_x < 350 else "right"
        field = kwargs["field"]
        if field == "surface":
            text = "英語" if column == "left" else "エレベ-ター"
            token_id = f"{column}-surface-region"
            return SimpleNamespace(
                text=text,
                confidence=0.88,
                tokens=[_token(token_id, text, [bbox[0] + 1, bbox[1] + 1, bbox[2] - 1, bbox[3] - 1], "mixed", source="jp_region_ocr", confidence=0.88)],
                bbox=bbox,
                cache={"hit": False},
                warnings=[],
            )
        if field == "meaning_ko":
            text = "영어" if column == "left" else "엘리베이터"
            token_id = f"{column}-meaning-region"
            return SimpleNamespace(
                text=text,
                confidence=0.86,
                tokens=[_token(token_id, text, [bbox[0] + 1, bbox[1] + 1, bbox[2] - 1, bbox[3] - 1], "hangul", source="ko_glyph_ocr", confidence=0.86)],
                bbox=bbox,
                cache={"hit": False},
                warnings=[],
            )
        return None

    monkeypatch.setattr(pipeline, "_safe_recognize_region", fake_recognize_region)

    new_items, recovered_tokens, diagnostics = pipeline._recover_missing_vocab_rows_from_unpaired_tokens(
        existing_items,
        all_tokens,
        {"l-s", "l-r", "l-m", "r-s", "r-r", "r-m"},
        "page",
        DictionaryValidator(),
        image_path=tmp_path / "page.jpg",
        page_width=800,
        page_height=1000,
        preprocessing_hash="hash",
        profile_id="jp_v3_det_v3_rec",
        korean_profile_id="ko_v5_current",
    )
    ordered = sorted([*existing_items, *new_items], key=pipeline._vocab_item_workbook_order_key)

    assert [(item["surface"], item["reading"], item["meaning_ko"]) for item in new_items] == [
        ("英語", "えいご", "영어"),
        ("エレベーター", "えれべえたあ", "엘리베이터"),
    ]
    assert [item["id"] for item in ordered[:2]] == ["row-left-existing", "row-right-existing"]
    assert ordered[2]["column"] == "left"
    assert ordered[3]["column"] == "right"
    assert all(item["field_evidence"]["surface"]["provenance"] == "jp_region_ocr" for item in new_items)
    assert all(item["field_evidence"]["meaning_ko"]["provenance"] == "ko_glyph_ocr" for item in new_items)
    assert len(recovered_tokens) == 4
    assert diagnostics["attempted"] == 2
    assert diagnostics["rejected"] == 0


def test_korean_residual_glyph_repairs_are_guarded() -> None:
    numeric_item = {"surface": "九つ", "reading": "ここのつ", "meaning_ko": "400"}
    wrong_month_item = {"surface": "九つ", "reading": "ここのつ", "meaning_ko": "9월"}

    assert pipeline._residual_korean_shape_repair_text(numeric_item) == "9개"
    assert pipeline._residual_korean_shape_repair_text(wrong_month_item) is None
    assert pipeline._recovery_shortens_existing_hangul("그에어컨", "에어") is True


def test_korean_residual_glyph_skips_when_region_ocr_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "_safe_recognize_region", lambda **_kwargs: None)
    item = {
        "id": "row-1",
        "surface": "九つ",
        "reading": "ここのつ",
        "meaning_ko": "400",
        "field_evidence": {"meaning_ko": {"bbox": [60, 20, 96, 42], "token_ids": ["meaning-token"], "confidence": 0.7}},
    }

    items, recovered_tokens, diagnostics = pipeline._recover_korean_residual_glyph_items(
        [item],
        [],
        Path("unused.jpg"),
        "page",
        800,
        1000,
        "hash",
        "jp_v3_det_v3_rec",
        "ko_v5_current",
    )

    assert recovered_tokens == []
    assert items[0]["meaning_ko"] == "400"
    assert diagnostics["counts"]["ko_glyph_rejected_no_region"] == 1


def test_prompt_line_cleanup_preserves_leading_digits_and_removes_noise() -> None:
    assert pipeline._clean_mcq_prompt_line_text("-50の半分は25です日") == "50の半分は25です。"
    assert pipeline._clean_mcq_prompt_line_text("①きょうは天気です②あした") == "きょうは天気です。"
    assert pipeline._repair_mcq_prompt_sentence_v2("きょうは土よう目です", "") == "きょうは土よう日です。"
    assert pipeline._repair_mcq_prompt_sentence_v2("傘をよみます", "本") == "本をよみます。"


def test_mcq_choice_glyph_recovery_updates_source_fields_only(monkeypatch: pytest.MonkeyPatch) -> None:
    item = {
        "id": "question-1",
        "question_type": "spelling_mcq",
        "target": "てんき",
        "choices": ["semantic-a", "semantic-b", "semantic-c", "semantic-d"],
        "correct_answer": "semantic-b",
        "source_fields": {
            "target": "てんき",
            "choices": ["天気", "天気", "夫気", "夫気"],
            "correct_choice_no": 2,
            "correct_answer": "天気",
        },
        "field_evidence": {"choices": {"bbox": [10, 10, 90, 40], "token_ids": ["choice-band"]}},
    }
    crop_text = ["天気", "天気", "夫気", "夫気"]

    def fake_recognize_region(**kwargs: object) -> SimpleNamespace:
        field = str(kwargs["field"])
        choice_no = int(field.rsplit("_", 1)[-1])
        bbox = list(kwargs["bbox"])
        token = OcrToken(
            id=f"choice-crop-{choice_no}",
            page_id="page",
            text=crop_text[choice_no - 1],
            bbox=bbox,
            confidence=0.88,
            script_class="mixed",
            source="choice_glyph_ocr",
        )
        return SimpleNamespace(text=token.text, confidence=0.88, tokens=[token], bbox=bbox, cache={"hit": False})

    monkeypatch.setattr(pipeline, "_safe_recognize_region", fake_recognize_region)

    items, recovered_tokens, diagnostics = pipeline._recover_mcq_choice_glyphs(
        [item],
        Path("unused.jpg"),
        "page",
        800,
        1000,
        "hash",
        "jp_v3_det_v3_rec",
        "ko_v5_current",
    )

    assert len(recovered_tokens) == 8
    assert diagnostics["counts"]["choice_glyph_accepted"] == 1
    assert items[0]["choices"] == ["semantic-a", "semantic-b", "semantic-c", "semantic-d"]
    assert items[0]["correct_answer"] == "semantic-b"
    assert items[0]["source_fields"]["choices"] == ["天気", "天气", "夫気", "夫气"]
    assert items[0]["source_fields"]["correct_answer"] == "天气"
    assert items[0]["field_evidence"]["choices"]["provenance"] == "choice_glyph_ocr"
    assert set(items[0]["field_evidence"]["choices"]["token_ids"]).issubset({token.id for token in recovered_tokens})
    assert items[0]["field_evidence"]["choice_2"]["region_strategy"] == "mcq_choice_glyph_2"


def test_mcq_choice_glyph_rejects_missing_crop_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    item = {
        "id": "question-1",
        "question_type": "spelling_mcq",
        "choices": ["semantic-a", "semantic-b", "semantic-c", "semantic-d"],
        "correct_answer": "semantic-b",
        "source_fields": {
            "choices": ["天気", "天気", "夫気", "夫気"],
            "correct_choice_no": 2,
            "correct_answer": "天気",
        },
        "field_evidence": {"choices": {"bbox": [10, 10, 90, 40], "token_ids": ["choice-band"]}},
    }
    monkeypatch.setattr(pipeline, "_safe_recognize_region", lambda **kwargs: SimpleNamespace(text="", confidence=0.0, tokens=[], bbox=kwargs["bbox"], cache={"hit": False}))

    items, recovered_tokens, diagnostics = pipeline._recover_mcq_choice_glyphs(
        [item],
        Path("unused.jpg"),
        "page",
        800,
        1000,
        "hash",
        "jp_v3_det_v3_rec",
        "ko_v5_current",
    )

    assert recovered_tokens == []
    assert diagnostics["counts"]["choice_glyph_rejected_no_crop"] == 1
    assert items[0]["source_fields"]["choices"] == ["天気", "天気", "夫気", "夫気"]


def test_accuracy_recovery_v2_gate_uses_safe_local_peak_rss_not_process_tree_peak() -> None:
    result = benchmark_ocr_modes.PageBenchmark(
        page_id="gate",
        image_path="",
        base={
            "matched": 80,
            "expected": 80,
            "source_field_matches": 160,
            "source_field_expected": 160,
            "meaning_matches": 60,
            "surface_matches": 60,
            "reading_matches": 60,
            "benchmark": {
                "mode": "fresh_cli",
                "model_profile": "jp_v3_det_v3_rec",
                "korean_profile": "ko_v5_current",
                "extraction_variant": "accuracy_recovery_v2",
                "document_graph_metrics": {"evidence_alignment_score": 0.95},
            },
        },
        vl=None,
        google_vision=None,
        memory_samples=[],
        resource_metrics={"peak_rss_mb": 3100, "process_tree_peak_rss_mb": 9000},
        errors=[],
        audit_artifacts=None,
    )

    [gate] = benchmark_ocr_modes._success_gate_payload([result])

    assert gate["passed"] is True
    assert gate["peak_rss"] == "3100.0 MB"


def test_staged_benchmark_skips_invalid_profile_ids() -> None:
    args = type(
        "Args",
        (),
        {
            "experiment_stage": "2",
            "stage_profiles": "missing_profile,jp_v3_mobile_current",
            "stage_variants": "",
            "include_heavy_profiles": False,
            "variant_matrix": False,
            "extraction_variant": "baseline_current",
            "model_profile": "jp_v3_mobile_current",
        },
    )()

    assert benchmark_ocr_modes._profile_ids_for_run(args) == ["jp_v3_mobile_current"]


def test_variant_matrix_reuses_page_profile_work_dir_for_ocr_cache(tmp_path) -> None:
    golden = GoldenPage(
        page_id="page-1",
        image_path=tmp_path / "page.jpg",
        category="vocab_table",
        expected_page_type="vocab_table",
    )
    stage_two = type(
        "Args",
        (),
        {
            "experiment_stage": "2",
            "profile_matrix": False,
            "variant_matrix": True,
            "korean_profile": "ko_v5_current",
        },
    )()
    baseline_path = benchmark_ocr_modes._page_work_dir_for_run(
        tmp_path,
        golden,
        "jp_v3_mobile_current",
        stage_two,
    )
    table_path = benchmark_ocr_modes._page_work_dir_for_run(
        tmp_path,
        golden,
        "jp_v3_mobile_current",
        stage_two,
    )
    v5_path = benchmark_ocr_modes._page_work_dir_for_run(
        tmp_path,
        golden,
        "jp_v5_mobile_general",
        stage_two,
    )
    v5_alias_path = benchmark_ocr_modes._page_work_dir_for_run(
        tmp_path,
        golden,
        "jp_v5_det_v5_rec",
        stage_two,
    )
    korean_alias_args = type(
        "Args",
        (),
        {
            "experiment_stage": "2",
            "profile_matrix": False,
            "variant_matrix": True,
            "korean_profile": "ko_v5_det_v5_rec",
        },
    )()
    korean_alias_path = benchmark_ocr_modes._page_work_dir_for_run(
        tmp_path,
        golden,
        "jp_v3_mobile_current",
        korean_alias_args,
    )

    assert baseline_path == table_path
    assert baseline_path == tmp_path / "page-1.jp_v3_mobile_current.ko_v5_current"
    assert v5_path == tmp_path / "page-1.jp_v5_mobile_general.ko_v5_current"
    assert v5_alias_path == v5_path
    assert korean_alias_path == baseline_path


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
    rows_only = pipeline._extraction_variant_metrics("v5_vocab_rows_v1", [card], graph.model_dump())
    adapted = pipeline._extraction_variant_metrics("v5_full_adapted_v1", [card], graph.model_dump())

    assert ranked["candidate_mutation"] is False
    assert ranked["ranked_rows"][0]["source_id"] == "row-1"
    assert crop["uncertain_fields"][0]["field"] == "surface"
    assert agreement["diagnostic_only"] is True
    assert agreement["automatic_extraction_decision"] is False
    assert rows_only["diagnostic_only"] is True
    assert rows_only["candidate_mutation"] is False
    assert rows_only["row_alignment_candidate_replacement"] == "guarded_off"
    assert adapted["candidate_mutation"] is True
    assert adapted["components"] == ["ko_alignment_v1", "v5_mcq_v1", "v5_token_split_v1", "v5_vocab_rows_v1"]
    assert adapted["vocab_rows_candidate_replacement"] == "guarded_off"


def test_row_alignment_diagnostics_report_shadow_rows_without_mutation(tmp_path) -> None:
    card = _vocab_eval_card(
        {
            "surface": "学校",
            "reading": "がっこう",
            "meaning_ko": "학교",
            "field_evidence": {
                "surface": {"text": "学校", "provenance": "ocr", "token_ids": ["surface"], "bbox": [100, 100, 140, 124]},
                "reading": {"text": "がっこう", "provenance": "ocr", "token_ids": ["reading"], "bbox": [150, 100, 220, 124]},
                "meaning_ko": {"text": "학교", "provenance": "ocr", "token_ids": ["ko1"], "bbox": [250, 100, 300, 124]},
            },
        }
    )
    tokens = [
        _token("surface", "学校", [100, 100, 140, 124], "kanji"),
        _token("reading", "がっこう", [150, 100, 220, 124], "hiragana"),
        _token("surface-2", "先生", [100, 142, 140, 166], "kanji"),
        _token("reading-2", "せんせい", [150, 142, 220, 166], "hiragana"),
        _token("ko1", "학교", [250, 100, 300, 124], "hangul", source="paddleocr_korean"),
        _token("ko2", "선생", [250, 250, 300, 274], "hangul", source="paddleocr_korean"),
    ]
    graph = graph_with_card_hypotheses(graph_from_tokens("page", tokens), [card])

    diagnostics = pipeline._extraction_variant_diagnostics(
        "v5_vocab_rows_plus_ko_alignment_v1",
        [card],
        tmp_path / "page.png",
        "page",
        400,
        300,
        tokens,
    )
    metrics = pipeline._extraction_variant_metrics(
        "v5_vocab_rows_plus_ko_alignment_v1",
        [card],
        graph.model_dump(),
        variant_diagnostics=diagnostics,
    )

    alignment = metrics["vocab_alignment"]
    assert alignment["candidate_replacement"] == "guarded_off"
    assert alignment["shadow_row_count"] == 2
    assert alignment["shadow_complete_row_count"] == 1
    assert alignment["warning_counts"]["MISSING_KOREAN_MEANING"] == 1
    assert metrics["candidate_mutation"] is False
    assert metrics["korean_alignment"]["paired_korean_token_count"] == 1


def test_row_alignment_diagnostics_skip_non_vocab_candidates(tmp_path) -> None:
    card = CardCandidate(
        id="card-q",
        page_id="page",
        source_type="question_item",
        source_id="q-1",
        note_type="jp_reading_mcq_recall",
        front="front",
        back="back",
        source={"question_no": 1},
    )
    diagnostics = pipeline._extraction_variant_diagnostics(
        "v5_full_adapted_v1",
        [card],
        tmp_path / "page.png",
        "page",
        400,
        300,
        [_token("sentence", "学校へ行きます", [100, 100, 220, 124], "mixed")],
    )

    assert "vocab_alignment" not in diagnostics


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


def test_korean_crop_recovery_persists_live_recovery_tokens(tmp_path, monkeypatch) -> None:
    recovered = _token("ko-recovered", "5개", [80, 10, 105, 24], "hangul", source="paddleocr_korean")

    class FakeCropWorker:
        def recognize_region(self, **kwargs):
            return SimpleNamespace(
                text="5개",
                confidence=0.93,
                tokens=[recovered],
                bbox=list(kwargs["bbox"]),
                field_evidence={
                    "bbox": list(kwargs["bbox"]),
                    "token_ids": [recovered.id],
                    "text": "5개",
                    "confidence": 0.93,
                    "provenance": kwargs["provenance"],
                    "provider": "paddle_korean",
                },
                cache={"hit": False, "key": "region-key"},
                warnings=[],
            )

    monkeypatch.setattr(pipeline, "crop_ocr_worker", FakeCropWorker())
    item = {
        "id": "vocab-1",
        "surface": "五つ",
        "reading": "いつつ",
        "meaning_ko": "5",
        "bbox": [10, 10, 105, 24],
        "row_bbox": [10, 8, 116, 28],
        "reading_bbox": [45, 10, 75, 24],
        "column": "left",
        "field_evidence": {
            "surface": {"text": "五つ", "bbox": [10, 10, 35, 24], "token_ids": ["surface"], "provenance": "ocr"},
            "reading": {"text": "いつつ", "bbox": [45, 10, 75, 24], "token_ids": ["reading"], "provenance": "ocr"},
            "meaning_ko": {"text": "5", "bbox": [80, 10, 94, 24], "token_ids": ["meaning"], "provenance": "ocr"},
        },
        "warnings": ["OCR script classification does not match the expected field."],
        "confidence": 0.6,
    }

    items, tokens, diagnostics = pipeline._recover_korean_vocab_items(
        [item],
        [],
        tmp_path / "page.png",
        "page",
        200,
        100,
        "pre-hash",
        "jp_v3_det_v3_rec",
        "ko_v5_current",
        frozenset({"ko_crop_confirm_v1"}),
    )

    assert items[0]["meaning_ko"] == "5개"
    assert items[0]["field_evidence"]["meaning_ko"]["provenance"] == "crop_ocr"
    assert items[0]["field_evidence"]["meaning_ko"]["token_ids"] == ["ko-recovered"]
    assert tokens == [recovered]
    assert diagnostics["accepted"] == 1
    assert diagnostics["counts"]["recovered_by_crop"] == 1


def test_korean_region_recovery_accepts_high_confidence_token_from_noisy_region(tmp_path, monkeypatch) -> None:
    noisy_number = _token("noise-number", "2", [80, 10, 88, 18], "number", confidence=0.09, source="paddleocr_korean")
    noisy_slash = _token("noise-slash", "/", [90, 10, 98, 18], "punctuation", confidence=0.23, source="paddleocr_korean")
    recovered = _token("ko-recovered", "5개", [100, 12, 124, 28], "mixed", confidence=0.99, source="paddleocr_korean")

    class FakeCropWorker:
        def recognize_region(self, **kwargs):
            return SimpleNamespace(
                text="2 / 5개",
                confidence=0.43,
                tokens=[noisy_number, noisy_slash, recovered],
                bbox=list(kwargs["bbox"]),
                field_evidence={
                    "bbox": list(kwargs["bbox"]),
                    "token_ids": [noisy_number.id, noisy_slash.id, recovered.id],
                    "text": "2 / 5개",
                    "confidence": 0.43,
                    "provenance": kwargs["provenance"],
                    "provider": "paddle_korean",
                },
                cache={"hit": True, "key": "region-key"},
                warnings=[],
            )

    monkeypatch.setattr(pipeline, "crop_ocr_worker", FakeCropWorker())
    item = {
        "id": "vocab-1",
        "surface": "五つ",
        "reading": "いつつ",
        "meaning_ko": "5",
        "bbox": [10, 10, 124, 28],
        "row_bbox": [10, 8, 130, 32],
        "reading_bbox": [45, 10, 75, 24],
        "column": "left",
        "field_evidence": {
            "meaning_ko": {"text": "5", "bbox": [80, 10, 94, 24], "token_ids": ["meaning"], "provenance": "ocr"},
        },
        "warnings": ["OCR script classification does not match the expected field."],
        "confidence": 0.6,
    }

    items, tokens, diagnostics = pipeline._recover_korean_vocab_items(
        [item],
        [],
        tmp_path / "page.png",
        "page",
        200,
        100,
        "pre-hash",
        "jp_v3_det_v3_rec",
        "ko_v5_current",
        frozenset({"ko_region_columns_v1"}),
    )

    assert items[0]["meaning_ko"] == "5개"
    assert items[0]["field_evidence"]["meaning_ko"]["token_ids"] == ["ko-recovered"]
    assert items[0]["field_evidence"]["meaning_ko"]["confidence"] == pytest.approx(0.99)
    assert tokens == [noisy_number, noisy_slash, recovered]
    assert diagnostics["accepted"] == 1


def test_korean_recovery_prioritizes_bare_numeric_before_low_confidence_hangul_under_cap(tmp_path, monkeypatch) -> None:
    recovered = _token("ko-recovered", "5개", [100, 12, 124, 28], "mixed", confidence=0.99, source="paddleocr_korean")
    attempted_ids: list[str] = []

    class FakeCropWorker:
        def recognize_region(self, **kwargs):
            attempted_ids.append(str(kwargs["region_id"]))
            return SimpleNamespace(
                text="5개",
                confidence=0.99,
                tokens=[recovered],
                bbox=list(kwargs["bbox"]),
                field_evidence={
                    "bbox": list(kwargs["bbox"]),
                    "token_ids": [recovered.id],
                    "text": "5개",
                    "confidence": 0.99,
                    "provenance": kwargs["provenance"],
                    "provider": "paddle_korean",
                },
                cache={"hit": True, "key": "region-key"},
                warnings=[],
            )

    monkeypatch.setattr(pipeline, "OCR_RECOVERY_MAX_FIELDS", 1)
    monkeypatch.setattr(pipeline, "crop_ocr_worker", FakeCropWorker())
    low_confidence_good = {
        "id": "already-good",
        "surface": "雨",
        "reading": "あめ",
        "meaning_ko": "비",
        "bbox": [10, 10, 120, 28],
        "row_bbox": [10, 8, 130, 32],
        "reading_bbox": [45, 10, 75, 24],
        "column": "left",
        "field_evidence": {
            "meaning_ko": {
                "text": "비",
                "bbox": [100, 10, 114, 24],
                "token_ids": ["meaning-good"],
                "provenance": "ocr",
                "confidence": 0.5,
            },
        },
        "warnings": [],
        "confidence": 0.5,
    }
    bare_numeric = {
        "id": "bare-numeric",
        "surface": "五つ",
        "reading": "いつつ",
        "meaning_ko": "5",
        "bbox": [10, 40, 120, 58],
        "row_bbox": [10, 38, 130, 62],
        "reading_bbox": [45, 40, 75, 54],
        "column": "left",
        "field_evidence": {
            "meaning_ko": {
                "text": "5",
                "bbox": [100, 40, 114, 54],
                "token_ids": ["meaning-numeric"],
                "provenance": "ocr",
                "confidence": 0.8,
            },
        },
        "warnings": ["OCR script classification does not match the expected field."],
        "confidence": 0.6,
    }

    items, _tokens, diagnostics = pipeline._recover_korean_vocab_items(
        [low_confidence_good, bare_numeric],
        [],
        tmp_path / "page.png",
        "page",
        200,
        100,
        "pre-hash",
        "jp_v3_det_v3_rec",
        "ko_v5_current",
        frozenset({"ko_region_columns_v1"}),
    )

    assert attempted_ids == ["bare-numeric"]
    assert items[0]["meaning_ko"] == "비"
    assert items[1]["meaning_ko"] == "5개"
    assert diagnostics["attempted"] == 1
    assert diagnostics["counts"]["recovery_resource_cap"] == 1


def test_korean_recovery_completes_numeric_unit_from_ocr_backed_japanese_row(tmp_path, monkeypatch) -> None:
    class FakeCropWorker:
        def recognize_region(self, **_kwargs):
            raise AssertionError("OCR-backed numeric unit completion should not require region OCR")

    monkeypatch.setattr(pipeline, "crop_ocr_worker", FakeCropWorker())
    item = {
        "id": "minute-row",
        "surface": "五分",
        "reading": "ごふん",
        "meaning_ko": "5",
        "bbox": [10, 10, 140, 28],
        "row_bbox": [10, 8, 150, 32],
        "column": "left",
        "field_evidence": {
            "surface": {
                "text": "五分",
                "bbox": [10, 10, 42, 24],
                "token_ids": ["surface"],
                "provenance": "ocr",
                "confidence": 1.0,
            },
            "reading": {
                "text": "ごふん",
                "bbox": [50, 10, 88, 24],
                "token_ids": ["reading"],
                "provenance": "ocr",
                "confidence": 0.9,
            },
            "meaning_ko": {
                "text": "5",
                "bbox": [96, 10, 110, 24],
                "token_ids": ["meaning"],
                "provenance": "ocr",
                "confidence": 0.82,
            },
        },
        "warnings": ["OCR script classification does not match the expected field."],
        "confidence": 0.82,
    }

    items, tokens, diagnostics = pipeline._recover_korean_vocab_items(
        [item],
        [_token("meaning", "5", [96, 10, 110, 24], "number", confidence=0.82, source="paddleocr_korean")],
        tmp_path / "page.png",
        "page",
        200,
        100,
        "pre-hash",
        "jp_v3_det_v3_rec",
        "ko_v5_current",
        frozenset({"ko_region_columns_v1"}),
    )

    assert items[0]["meaning_ko"] == "5분"
    assert items[0]["field_evidence"]["meaning_ko"]["token_ids"] == ["meaning"]
    assert items[0]["field_evidence"]["meaning_ko"]["normalization_strategy"] == "numeric_unit_completion_v1"
    assert tokens == []
    assert diagnostics["accepted"] == 1
    assert diagnostics["counts"]["recovered_by_numeric_unit"] == 1


def test_korean_recovery_rejects_numeric_unit_when_digits_disagree(tmp_path, monkeypatch) -> None:
    class FakeCropWorker:
        def recognize_region(self, **_kwargs):
            return None

    monkeypatch.setattr(pipeline, "crop_ocr_worker", FakeCropWorker())
    item = {
        "id": "counter-row",
        "surface": "九つ",
        "reading": "ここのつ",
        "meaning_ko": "400",
        "bbox": [10, 10, 140, 28],
        "row_bbox": [10, 8, 150, 32],
        "column": "left",
        "field_evidence": {
            "surface": {
                "text": "九つ",
                "bbox": [10, 10, 42, 24],
                "token_ids": ["surface"],
                "provenance": "ocr",
                "confidence": 0.98,
            },
            "reading": {
                "text": "ここのつ",
                "bbox": [50, 10, 88, 24],
                "token_ids": ["reading"],
                "provenance": "ocr",
                "confidence": 0.95,
            },
            "meaning_ko": {
                "text": "400",
                "bbox": [96, 10, 120, 24],
                "token_ids": ["meaning"],
                "provenance": "ocr",
                "confidence": 0.9,
            },
        },
        "warnings": ["OCR script classification does not match the expected field."],
        "confidence": 0.9,
    }

    items, tokens, diagnostics = pipeline._recover_korean_vocab_items(
        [item],
        [_token("meaning", "400", [96, 10, 120, 24], "number", confidence=0.9, source="paddleocr_korean")],
        tmp_path / "page.png",
        "page",
        200,
        100,
        "pre-hash",
        "jp_v3_det_v3_rec",
        "ko_v5_current",
        frozenset({"ko_region_columns_v1"}),
    )

    assert items[0]["meaning_ko"] == "400"
    assert tokens == []
    assert diagnostics["accepted"] == 0


def test_korean_region_recovery_rejects_unrelated_token_for_existing_hangul(tmp_path, monkeypatch) -> None:
    unrelated = _token("ko-unrelated", "말하다", [100, 12, 150, 28], "hangul", confidence=0.99, source="paddleocr_korean")

    class FakeCropWorker:
        def recognize_region(self, **kwargs):
            return SimpleNamespace(
                text="l 말하다 2 2",
                confidence=0.45,
                tokens=[unrelated],
                bbox=list(kwargs["bbox"]),
                field_evidence={
                    "bbox": list(kwargs["bbox"]),
                    "token_ids": [unrelated.id],
                    "text": "l 말하다 2 2",
                    "confidence": 0.45,
                    "provenance": kwargs["provenance"],
                    "provider": "paddle_korean",
                },
                cache={"hit": True, "key": "region-key"},
                warnings=[],
            )

    monkeypatch.setattr(pipeline, "crop_ocr_worker", FakeCropWorker())
    item = {
        "id": "vocab-1",
        "surface": "今",
        "reading": "いま",
        "meaning_ko": "수 지금",
        "bbox": [10, 10, 150, 28],
        "row_bbox": [10, 8, 160, 32],
        "reading_bbox": [45, 10, 75, 24],
        "column": "left",
        "field_evidence": {
            "meaning_ko": {
                "text": "수 지금",
                "bbox": [100, 10, 144, 24],
                "token_ids": ["meaning"],
                "provenance": "ocr",
                "confidence": 0.5,
            },
        },
        "warnings": ["Weak OCR evidence; verify this row manually."],
        "confidence": 0.5,
    }

    items, tokens, diagnostics = pipeline._recover_korean_vocab_items(
        [item],
        [],
        tmp_path / "page.png",
        "page",
        200,
        100,
        "pre-hash",
        "jp_v3_det_v3_rec",
        "ko_v5_current",
        frozenset({"ko_region_columns_v1"}),
    )

    assert items[0]["meaning_ko"] == "수 지금"
    assert tokens == []
    assert diagnostics["accepted"] == 0
    assert diagnostics["counts"]["rejected_by_consensus"] == 1


def test_korean_region_recovery_rejects_neighbor_expanded_existing_meaning(tmp_path, monkeypatch) -> None:
    recovered = _token("ko-expanded", "5일 개", [80, 10, 120, 24], "hangul", source="paddleocr_korean")

    class FakeCropWorker:
        def recognize_region(self, **kwargs):
            return SimpleNamespace(
                text="5일 개",
                confidence=0.99,
                tokens=[recovered],
                bbox=list(kwargs["bbox"]),
                field_evidence={
                    "bbox": list(kwargs["bbox"]),
                    "token_ids": [recovered.id],
                    "text": "5일 개",
                    "confidence": 0.99,
                    "provenance": kwargs["provenance"],
                    "provider": "paddle_korean",
                },
                cache={"hit": False, "key": "region-key"},
                warnings=[],
            )

    monkeypatch.setattr(pipeline, "crop_ocr_worker", FakeCropWorker())
    item = {
        "id": "vocab-1",
        "surface": "犬",
        "reading": "いぬ",
        "meaning_ko": "개",
        "bbox": [10, 10, 120, 24],
        "row_bbox": [10, 8, 130, 28],
        "reading_bbox": [45, 10, 75, 24],
        "column": "left",
        "field_evidence": {
            "meaning_ko": {"text": "개", "bbox": [80, 10, 94, 24], "token_ids": ["meaning"], "provenance": "ocr", "confidence": 0.6},
        },
        "warnings": ["Weak OCR evidence; verify this row manually."],
        "confidence": 0.6,
    }

    items, tokens, diagnostics = pipeline._recover_korean_vocab_items(
        [item],
        [],
        tmp_path / "page.png",
        "page",
        200,
        100,
        "pre-hash",
        "jp_v3_det_v3_rec",
        "ko_v5_current",
        frozenset({"ko_region_columns_v1"}),
    )

    assert items[0]["meaning_ko"] == "개"
    assert tokens == []
    assert diagnostics["accepted"] == 0
    assert diagnostics["counts"]["rejected_by_consensus"] == 1


def test_korean_recovery_skips_header_like_rows_without_reading(tmp_path, monkeypatch) -> None:
    class FakeCropWorker:
        def recognize_region(self, **_kwargs):
            raise AssertionError("header-like rows must not invoke recovery OCR")

    monkeypatch.setattr(pipeline, "crop_ocr_worker", FakeCropWorker())
    item = {
        "id": "header-row",
        "surface": "式幸",
        "reading": "",
        "meaning_ko": "한자읽기 기출어휘",
        "bbox": [10, 10, 160, 24],
        "row_bbox": [10, 8, 170, 28],
        "column": "left",
        "field_evidence": {
            "meaning_ko": {"text": "한자읽기 기출어휘", "bbox": [80, 10, 150, 24], "token_ids": ["meaning"], "provenance": "ocr", "confidence": 0.6},
        },
        "warnings": ["OCR script classification does not match the expected field."],
        "confidence": 0.6,
    }

    items, tokens, diagnostics = pipeline._recover_korean_vocab_items(
        [item],
        [],
        tmp_path / "page.png",
        "page",
        200,
        100,
        "pre-hash",
        "jp_v3_det_v3_rec",
        "ko_v5_current",
        frozenset({"ko_crop_confirm_v1", "ko_region_columns_v1"}),
    )

    assert items[0]["meaning_ko"] == "한자읽기 기출어휘"
    assert tokens == []
    assert diagnostics["attempted"] == 0


def test_mcq_evaluator_scores_source_fields_separately_from_semantics(tmp_path) -> None:
    golden = GoldenPage(
        page_id="mcq-page",
        image_path=tmp_path / "page.jpg",
        category="reading_mcq",
        expected_page_type="reading_mcq",
        expected_questions=[
            GoldenQuestion(
                question_id="q1",
                question_no=1,
                sentence="学校へ行きます。",
                target="学校",
                choices=["がっこう", "せんせい", "でんしゃ", "きょうしつ"],
                correct_answer="がっこう",
                correct_choice_no=1,
                answer_source="answer_strip",
            )
        ],
    )
    card = CardCandidate(
        id="card",
        page_id="page",
        source_type="question_item",
        source_id="q1",
        note_type="jp_reading_mcq_recall",
        front="front",
        back="back",
        source={
            "question_no": 1,
            "sentence": "学校へ行きます。",
            "target": "学校",
            "choices": ["がっこう", "せんせい", "でんしゃ", "きょうしつ"],
            "correct_answer": "がっこう",
            "correct_choice_no": 1,
            "source_fields": {
                "sentence": "学校",
                "target": "学校",
                "choices": ["がっこう", "せんせい", "でんしゃ", "きょうしつ"],
                "correct_answer": "がっこう",
                "correct_choice_no": 1,
            },
        },
    )

    result = evaluate_mcq_page(golden, ProcessResult(page=_page(tmp_path), tokens=[], cards=[card], script_summary={}))

    assert result.matched_questions == 1
    assert result.source_field_matches == 4
    assert result.source_field_expected == 5


def test_mcq_source_recovery_does_not_copy_glossary_answer_into_strict_fields(tmp_path) -> None:
    item = {
        "id": "q1",
        "question_no": 1,
        "sentence": "会社は休みです。",
        "target": "会社",
        "choices": ["かいしゃ", "でんしゃ", "がっこう", "びょういん"],
        "correct_answer": "かいしゃ",
        "correct_choice_no": 1,
        "answer_source": "local_glossary",
        "bbox": [10, 10, 120, 60],
    }

    items, tokens, diagnostics = pipeline._recover_mcq_source_items(
        [item],
        tmp_path / "page.png",
        "page",
        200,
        100,
        "pre-hash",
        "jp_v3_det_v3_rec",
        "ko_v5_current",
        frozenset({"mcq_source_rebuild_v1"}),
    )

    assert tokens == []
    assert diagnostics["attempted"] == 0
    assert items[0]["semantic_fields"]["correct_answer"] == "かいしゃ"
    assert items[0]["source_fields"]["correct_answer"] == ""
    assert items[0]["source_fields"]["correct_choice_no"] is None


def test_mcq_choice_band_ocr_applies_answer_strip_to_strict_fields(tmp_path, monkeypatch) -> None:
    strip_token = _token("answer-strip", "1②", [10, 80, 40, 96], "mixed")
    calls: list[dict] = []

    class FakeCropWorker:
        def recognize_region(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                text="1②",
                confidence=0.91,
                tokens=[strip_token],
                bbox=list(kwargs["bbox"]),
                provider="paddle",
                field_evidence={
                    "bbox": list(kwargs["bbox"]),
                    "token_ids": [strip_token.id],
                    "text": "1②",
                    "confidence": 0.91,
                    "provenance": kwargs["provenance"],
                    "provider": "paddle",
                },
                cache={"hit": False, "key": "answer-strip-key"},
                warnings=[],
            )

    monkeypatch.setattr(pipeline, "crop_ocr_worker", FakeCropWorker())
    item = {
        "id": "q1",
        "question_no": 1,
        "sentence": "会社は休みです。",
        "target": "会社",
        "choices": ["とようび", "どようび", "かようび", "がようび"],
        "correct_answer": "かいしゃ",
        "correct_choice_no": 1,
        "answer_source": "local_glossary",
        "bbox": [10, 10, 120, 60],
    }

    items, tokens, diagnostics = pipeline._recover_mcq_source_items(
        [item],
        tmp_path / "page.png",
        "page",
        200,
        100,
        "pre-hash",
        "jp_v3_det_v3_rec",
        "ko_v5_current",
        frozenset({"mcq_choice_band_ocr_v1"}),
    )

    assert tokens == [strip_token]
    assert diagnostics["attempted"] == 1
    assert diagnostics["accepted"] == 1
    assert items[0]["semantic_fields"]["correct_answer"] == "かいしゃ"
    assert items[0]["source_fields"]["correct_choice_no"] == 2
    assert items[0]["source_fields"]["correct_answer"] == "どようび"
    assert items[0]["field_evidence"]["correct_choice_no"]["provenance"] == "answer_strip_ocr"
    assert calls[0]["bbox"] == [20.0, 76.0, 192.0, 100.0]


def test_mcq_answer_strip_image_parser_creates_live_answer_tokens(tmp_path) -> None:
    Image = pytest.importorskip("PIL.Image")
    ImageDraw = pytest.importorskip("PIL.ImageDraw")
    ImageFont = pytest.importorskip("PIL.ImageFont")

    font_path = pipeline._answer_strip_template_font_path(ImageFont)
    if not font_path:
        pytest.skip("No circled-digit font available for answer-strip parser test.")
    image_path = tmp_path / "answer-strip.png"
    image = Image.new("RGB", (540, 180), (245, 242, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 70, 520, 120], fill=(178, 178, 172))
    font = ImageFont.truetype(font_path, 22)
    draw.text((50, 82), "1② 2① 3④ 4④ 5① 6② 7③ 8④ 9④ 10④", fill=(0, 0, 0), font=font)
    image.save(image_path)

    answer_map, tokens, warnings = pipeline._parse_answer_strip_image(
        image_path=image_path,
        page_id="page",
        bbox=[20, 70, 520, 130],
    )

    assert answer_map == {1: 2, 2: 1, 3: 4, 4: 4, 5: 1, 6: 2, 7: 3, 8: 4, 9: 4, 10: 4}
    assert [token.text for token in tokens] == ["2", "1", "4", "4", "1", "2", "3", "4", "4", "4"]
    assert all(token.page_id == "page" and token.source == "answer_strip_template_ocr" for token in tokens)
    assert warnings == []


def test_mcq_source_rebuild_repairs_ocr_choice_noise_without_semantic_copy() -> None:
    items = [
        {
            "question_type": "reading_mcq",
            "source_fields": {
                "sentence": "四月は花がきれいです。",
                "target": "花",
                "choices": ["そら", "はなス：もり", "みどり"],
                "correct_answer": "はなス：もり",
                "correct_choice_no": 2,
            },
            "semantic_fields": {"correct_answer": "semantic-only"},
            "field_evidence": {},
        },
        {
            "question_type": "spelling_mcq",
            "source_fields": {
                "sentence": "たかいやまのうえからがっこうがみえます。",
                "target": "やま",
                "choices": ["川", "士", "山", "田"],
                "correct_answer": "山",
                "correct_choice_no": 3,
            },
            "field_evidence": {},
        },
        {
            "question_type": "spelling_mcq",
            "source_fields": {
                "sentence": "あしたのてんきははれるでしょう",
                "target": "てんき",
                "choices": ["天気", "天気", "夫気", "夫気"],
                "correct_answer": "天気",
                "correct_choice_no": 1,
            },
            "field_evidence": {},
        },
    ]

    accepted = pipeline._repair_mcq_source_fields(items)

    assert accepted == 3
    assert items[0]["source_fields"]["choices"] == ["そら", "はな", "もり", "みどり"]
    assert items[0]["source_fields"]["correct_answer"] == "はな"
    assert items[0]["semantic_fields"]["correct_answer"] == "semantic-only"
    assert items[1]["source_fields"]["choices"] == ["川", "土", "山", "田"]
    assert items[2]["source_fields"]["sentence"] == "あしたのてんきははれるでしょう。"
    assert items[0]["field_evidence"]["choices"]["provenance"] == "source_rebuild"
    assert items[2]["field_evidence"]["sentence"]["normalization_strategy"] == "mcq_source_rebuild_v1"


def test_miss_inventory_is_diagnostic_only() -> None:
    page = benchmark_ocr_modes.PageBenchmark(
        page_id="page-1",
        image_path="page.jpg",
        base={
            "benchmark": {
                "mode": "fresh_cli",
                "model_profile": "jp_v3_det_v3_rec",
                "korean_profile": "ko_v5_current",
                "extraction_variant": "v5_full_adapted_v1",
            },
            "miss_analysis": {
                "kind": "vocab",
                "rows": [
                    {
                        "row_id": "row-1",
                        "reason": "korean_ocr_error",
                        "raw_presence": {"surface": True, "reading": True, "meaning_ko": False},
                        "expected": {"surface": "五つ", "reading": "いつつ", "meaning_ko": "5개"},
                        "best_candidate": {
                            "surface": "五つ",
                            "reading": "いつつ",
                            "meaning_ko": "5",
                            "field_matches": {"surface": True, "reading": True, "meaning_ko": False},
                        },
                    }
                ],
            },
        },
        vl=None,
        google_vision=None,
        memory_samples=[],
        resource_metrics={},
        errors=[],
        audit_artifacts={},
    )

    inventory = benchmark_ocr_modes._miss_inventory_payload([page], focus_source="previous.json")

    assert inventory["diagnostic_only"] is True
    assert inventory["oracle_use_allowed"] is False
    assert inventory["vocab_entry_count"] == 1
    assert inventory["entries"][0]["failed_fields"] == ["meaning_ko"]


def test_residual_diagnostics_exposes_required_schema_fields(tmp_path: Path) -> None:
    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_text(
        json.dumps(
            {
                "processed_image_path": str(tmp_path / "processed.png"),
                "cards": [
                    {
                        "id": "card-1",
                        "source_id": "row-1",
                        "source_bbox": [10, 20, 130, 44],
                        "confidence": 0.66,
                        "field_evidence": {
                            "meaning_ko": {
                                "text": "5",
                                "provenance": "ocr",
                                "token_ids": ["meaning-token"],
                                "bbox": [62, 22, 75, 39],
                                "confidence": 0.61,
                            }
                        },
                    }
                ],
                "extraction_variant_metrics": {
                    "recovery": {
                        "schema_version": 2,
                        "kind": "accuracy_recovery_v2",
                        "components": {
                            "korean_recovery": {
                                "attempts": [
                                    {
                                        "source_id": "row-1",
                                        "strategy": "korean_residual_glyph",
                                        "bbox": [60, 20, 96, 42],
                                        "text": "5개",
                                        "confidence": 0.73,
                                        "accepted": False,
                                        "reason": "ko_glyph_rejected_low_confidence",
                                        "cache": {"hit": False, "strategy": "korean_residual_glyph"},
                                        "candidates": [
                                            {
                                                "text": "5개",
                                                "confidence": 0.73,
                                                "accepted": False,
                                                "reason": "below_threshold",
                                                "cache": {"hit": True, "strategy": "korean_residual_glyph"},
                                            }
                                        ],
                                    }
                                ]
                            }
                        },
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    page = benchmark_ocr_modes.PageBenchmark(
        page_id="page-1",
        image_path="page.jpg",
        base={
            "benchmark": {
                "mode": "fresh_cli",
                "model_profile": "jp_v3_det_v3_rec",
                "korean_profile": "ko_v5_current",
                "extraction_variant": "accuracy_recovery_v2",
            },
            "miss_analysis": {
                "kind": "vocab",
                "rows": [
                    {
                        "row_id": "row-1",
                        "reason": "korean_ocr_error",
                        "raw_presence": {"surface": True, "reading": True, "meaning_ko": False},
                        "expected": {"surface": "五つ", "reading": "いつつ", "meaning_ko": "5개"},
                        "best_candidate": {
                            "surface": "五つ",
                            "reading": "いつつ",
                            "meaning_ko": "5",
                            "field_matches": {"surface": True, "reading": True, "meaning_ko": False},
                        },
                    }
                ],
            },
        },
        vl=None,
        google_vision=None,
        memory_samples=[],
        resource_metrics={"peak_rss_mb": 42.0},
        errors=[],
        audit_artifacts={"overlay_json": str(overlay_path)},
    )

    diagnostics_dir = tmp_path / "diagnostics"
    benchmark_ocr_modes._write_residual_diagnostics([page], diagnostics_dir, focus_source="previous.json")

    diagnostics = json.loads((diagnostics_dir / "residual-diagnostics.json").read_text(encoding="utf-8"))
    [entry] = diagnostics["entries"]
    assert diagnostics["diagnostic_only"] is True
    assert diagnostics["oracle_use_allowed"] is False
    assert entry["diagnostic_only"] is True
    assert entry["oracle_use_allowed"] is False
    assert entry["miss_kind"] == "vocab"
    assert entry["expected_value"]["meaning_ko"] == "5개"
    assert entry["actual_value"]["meaning_ko"] == "5"
    assert entry["field_evidence"]["meaning_ko"]["provenance"] == "ocr"
    assert entry["token_ids"] == ["meaning-token"]
    assert entry["crop_bbox"] == [60.0, 20.0, 96.0, 42.0]
    assert entry["region_strategy"] == "korean_residual_glyph"
    assert entry["ocr_candidates"][0]["text"] == "5개"
    assert entry["rejected_candidates"][0]["text"] == "5개"
    assert "ko_glyph_rejected_low_confidence" in entry["rejection_reasons"]
    assert entry["cache"]["hits"] == 1
    assert entry["cache"]["misses"] == 1
    assert entry["confidence"] == 0.73
    assert entry["resource_metrics"]["peak_rss_mb"] == 42.0
    assert (diagnostics_dir / "page-1.residuals.png").exists()
    assert (diagnostics_dir / "README.md").exists()


def test_focus_miss_inventory_is_diagnostic_only_and_does_not_filter_gates(tmp_path: Path) -> None:
    focus_path = tmp_path / "misses.json"
    focus_path.write_text(
        json.dumps(
            {
                "entries": [
                    {"page_id": "page-1", "failed_fields": ["meaning_ko", "surface"]},
                    {"page_id": "page-2", "failed_fields": ["choices"]},
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = benchmark_ocr_modes._focus_miss_summary(str(focus_path), "page-1")

    assert summary == {
        "source": str(focus_path),
        "status": "loaded",
        "diagnostic_only": True,
        "oracle_use_allowed": False,
        "page_entry_count": 1,
        "failed_fields": ["meaning_ko", "surface"],
    }
    assert benchmark_ocr_modes._focus_miss_summary(str(tmp_path / "missing.json"), "page-1") == {
        "source": str(tmp_path / "missing.json"),
        "status": "missing",
        "diagnostic_only": True,
    }

    def page_result(page_id: str) -> benchmark_ocr_modes.PageBenchmark:
        return benchmark_ocr_modes.PageBenchmark(
            page_id=page_id,
            image_path=f"{page_id}.jpg",
            base={
                "matched": 40,
                "expected": 40,
                "surface_matches": 40,
                "reading_matches": 40,
                "meaning_matches": 40,
                "benchmark": {
                    "mode": "fresh_cli",
                    "model_profile": "jp_v3_det_v3_rec",
                    "korean_profile": "ko_v5_current",
                    "extraction_variant": "accuracy_recovery_v2",
                    "document_graph_metrics": {"evidence_alignment_score": 1.0},
                    "focus_misses": benchmark_ocr_modes._focus_miss_summary(str(focus_path), page_id),
                },
            },
            vl=None,
            google_vision=None,
            memory_samples=[],
            resource_metrics={"peak_rss_mb": 10.0},
            errors=[],
            audit_artifacts={},
        )

    [gate] = benchmark_ocr_modes._success_gate_payload([page_result("page-1"), page_result("page-2")])

    assert gate["overall"] == "80/80"
    assert gate["surface"] == "80/80"
    assert gate["meaning"] == "80/80"


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

    def fake_base(
        golden_page: GoldenPage,
        _samples: list,
        engine: str,
        _profile: str,
        _korean_profile: str,
        _variant: str,
    ) -> ProcessResult:
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
    common = {"model_profile": "jp_v3_mobile_current", "korean_profile": "ko_v5_current", "extraction_variant": "baseline_current"}
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
    run = database.start_ocr_run(
        "page-a",
        "paddleocr",
        image_sha256="same-image",
        provider_config={"cache_key": "profile-key", "extraction_variant": "v5_full_adapted_v1"},
    )
    database.complete_ocr_run(run.id)
    recovery_run = database.start_ocr_run(
        "page-a",
        "paddleocr",
        image_sha256="same-image",
        provider_config={"cache_key": "profile-key", "extraction_variant": "accuracy_recovery_v1", "full_page_cache_write": False},
    )
    database.complete_ocr_run(recovery_run.id)
    legacy_recovery_run = database.start_ocr_run(
        "page-a",
        "paddleocr",
        image_sha256="same-image",
        provider_config={"cache_key": "profile-key", "extraction_variant": "ko_consensus_v1"},
    )
    database.complete_ocr_run(legacy_recovery_run.id)
    legacy_v2_recovery_run = database.start_ocr_run(
        "page-a",
        "paddleocr",
        image_sha256="same-image",
        provider_config={"cache_key": "profile-key", "extraction_variant": "accuracy_recovery_v2"},
    )
    database.complete_ocr_run(legacy_v2_recovery_run.id)

    cached = database.find_succeeded_run_by_cache_key(None, "paddleocr", "same-image", "profile-key")

    assert cached is not None
    assert cached.id == run.id
    assert database.find_succeeded_run_by_cache_key("page-b", "paddleocr", "same-image", "profile-key") is None


def test_v2_recovery_variants_do_not_seed_full_page_ocr_cache() -> None:
    manifest = {
        "cache": {"key": "cache-key"},
        "profile_id": "jp_v3_det_v3_rec",
    }

    assert pipeline._provider_config("paddleocr", manifest, "v5_full_adapted_v1")["full_page_cache_write"] is True
    for variant in (
        "jp_region_columns_v1",
        "ko_residual_glyph_v1",
        "mcq_prompt_line_ocr_v1",
        "mcq_choice_glyph_v1",
        "accuracy_recovery_v2",
    ):
        assert pipeline._provider_config("paddleocr", manifest, variant)["full_page_cache_write"] is False


def test_ocr_cache_key_reuses_payload_across_extraction_variants() -> None:
    manifest = {
        "profile_id": "jp_v3_det_v3_rec",
        "provider": "paddle",
        "env_fingerprint": "env-a",
        "model_config": {"japanese_detection_model": "PP-OCRv3_mobile_det"},
        "language_config": {"lang": None},
        "package_versions": {"paddleocr": "x"},
        "model_cache_paths": {"det": "/models/det"},
        "korean_profile": "ko_v5_current",
        "preprocessing_config": {
            "processed_width": 100,
            "processed_height": 200,
            "transform": {
                "original_image_path": "/tmp/a.png",
                "processed_image_path": "/tmp/a.processed.png",
                "original_to_processed": {"pipeline": ["resize"]},
            },
        },
    }

    baseline_key = pipeline._ocr_cache_key(
        image_sha="image-a",
        preprocessing_hash="pre-a",
        profile_manifest=manifest,
        engine="paddleocr",
        extraction_variant="baseline_current",
    )
    path_changed_manifest = {
        **manifest,
        "preprocessing_config": {
            **manifest["preprocessing_config"],
            "transform": {
                **manifest["preprocessing_config"]["transform"],
                "processed_image_path": "/tmp/b.processed.png",
            },
        },
    }
    variant_key = pipeline._ocr_cache_key(
        image_sha="image-a",
        preprocessing_hash="pre-a",
        profile_manifest=path_changed_manifest,
        engine="paddleocr",
        extraction_variant="v5_full_adapted_v1",
    )
    korean_key = pipeline._ocr_cache_key(
        image_sha="image-a",
        preprocessing_hash="pre-a",
        profile_manifest={**manifest, "korean_profile": "ko_lang_auto"},
        engine="paddleocr",
        extraction_variant="baseline_current",
    )

    assert variant_key == baseline_key
    assert korean_key != baseline_key


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
                },
                "extraction_variant_metrics": {"variant": "baseline_current", "candidate_mutation": False},
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
    assert overlay["extraction_variant_metrics"]["variant"] == "baseline_current"


def test_benchmark_dashboard_markdown_summarizes_comparison_rows(tmp_path) -> None:
    dashboard = tmp_path / "dashboard.md"
    results = [
        benchmark_ocr_modes.PageBenchmark(
            page_id="page-1",
            image_path=str(tmp_path / "page.jpg"),
            base={
                "matched": 1,
                "expected": 1,
                "accuracy": 1.0,
                "strict_ocr_score": 1.0,
                "surface_accuracy": 1.0,
                "reading_accuracy": 0.0,
                "meaning_accuracy": 0.0,
                "surface_matches": 1,
                "reading_matches": 0,
                "meaning_matches": 0,
                "manual_review_count": 2,
                "red_candidate_count": 1,
                "benchmark": {
                    "mode": "fresh_cli",
                    "model_profile": "jp_v3_mobile_current",
                    "korean_profile": "ko_v5_current",
                    "extraction_variant": "baseline_current",
                    "document_graph_metrics": {"evidence_alignment_score": 0.75},
                    "extraction_variant_metrics": {
                        "recovery": {
                            "schema_version": 2,
                            "kind": "accuracy_recovery_v2",
                            "attempted": 2,
                            "accepted": 1,
                            "counts": {"jp_recovered_missing_row": 1, "jp_resource_cap": 1},
                            "components": {
                                "japanese_vocab_recovery": {
                                    "attempted": 2,
                                    "accepted": 1,
                                    "counts": {"jp_recovered_missing_row": 1, "jp_resource_cap": 1},
                                    "attempts": [{"cache": {"hit": True}}, {"cache": {"hit": False}}],
                                }
                            },
                            "resource_caps": {"jp_resource_cap": 1},
                            "cache": {"hits": 1, "misses": 1},
                        }
                    },
                },
                "raw_field_coverage": {
                    "korean_raw_recall": {"matched": 1, "expected": 2, "accuracy": 0.5},
                },
                "miss_analysis": {"schema_version": 1, "kind": "vocab", "counts": {"wrong_pairing": 1}, "rows": []},
            },
            vl=None,
            memory_samples=[],
            resource_metrics={
                "wall_seconds": 1.2,
                "peak_rss_mb": 42,
                "cache": {"result_cache_hit": False, "cache_phase": "cold_or_uncached"},
            },
            errors=[],
        ),
        benchmark_ocr_modes.PageBenchmark(
            page_id="page-2",
            image_path=str(tmp_path / "page-2.jpg"),
            base={
                "matched": 0,
                "expected": 1,
                "accuracy": 0.0,
                "source_field_matches": 3,
                "source_field_expected": 5,
                "strict_ocr_score": 0.6,
                "manual_review_count": 1,
                "red_candidate_count": 0,
                "benchmark": {
                    "mode": "fresh_cli",
                    "model_profile": "jp_v3_mobile_current",
                    "korean_profile": "ko_v5_current",
                    "extraction_variant": "baseline_current",
                    "document_graph_metrics": {"evidence_alignment_score": 0.25},
                },
                "raw_field_coverage": {
                    "korean_raw_recall": {"matched": 0, "expected": 0, "accuracy": None},
                },
            },
            vl=None,
            memory_samples=[],
            resource_metrics={
                "wall_seconds": 2.0,
                "peak_rss_mb": 48,
                "cache": {"result_cache_hit": True, "cache_phase": "warm_ocr_cache"},
            },
            errors=[],
        ),
    ]

    benchmark_ocr_modes._write_dashboard_markdown(results, dashboard)

    text = dashboard.read_text(encoding="utf-8")
    assert "OCR Benchmark Dashboard" in text
    assert "## Summary" in text
    assert "Strict OCR" in text
    assert "Surface" in text
    assert "Reading" in text
    assert "Meaning" in text
    assert "Miss Causes" in text
    assert "Field Errors" in text
    assert "## Recovery Details" in text
    assert "| fresh_cli | jp_v3_mobile_current | ko_v5_current | baseline_current | japanese_vocab_recovery | 1 | 2 | 1 | 1 | 1/2 | jp_resource_cap:1 | jp_recovered_missing_row:1, jp_resource_cap:1 |" in text
    assert "Blocked" in text
    assert "| fresh_cli | jp_v3_mobile_current | ko_v5_current | baseline_current | 2 | 1/2 | 50.0% | 66.7% | 50.0% | 50.0% | 100.0% | 0.0% | 0.0% | wrong_pairing:1 | none |" in text
    assert "1/2" in text
    assert "50.0%" in text
    assert "66.7%" in text
    assert "| page-2 |" in text
    assert "60.0%" in text
    assert "1/2" in text
    assert "jp_v3_mobile_current" in text
    assert "75.0%" in text
    assert "50.0%" in text
    assert "cold_or_uncached" in text


def test_raw_field_coverage_marks_non_applicable_fields() -> None:
    golden = GoldenPage(page_id="mcq", image_path=Path("page.jpg"), category="reading_mcq", expected_page_type="reading_mcq")
    result = ProcessResult(
        page=Page(id="mcq", original_image_path="page.jpg", created_at="2026-05-06T00:00:00+09:00"),
        tokens=[],
        cards=[],
        script_summary={},
    )

    coverage = benchmark_ocr_modes._raw_field_coverage(golden, result)

    assert coverage["meaning_ko"]["expected"] == 0
    assert coverage["meaning_ko"]["accuracy"] is None
    assert coverage["korean_raw_recall"]["accuracy"] is None


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
    confidence: float = 0.9,
) -> OcrToken:
    return OcrToken(
        id=token_id,
        page_id="page",
        text=text,
        bbox=bbox,
        confidence=confidence,
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
