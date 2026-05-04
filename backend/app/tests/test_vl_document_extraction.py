from __future__ import annotations

from app.core.config import DICTIONARY_PATH
from app.extraction.vl_document import extract_from_document_parse
from app.models.schemas import DocumentParseBlock, DocumentParseResult
from app.validation.dictionary import DictionaryValidator


def _parse(blocks: list[DocumentParseBlock], markdown: str = "") -> DocumentParseResult:
    return DocumentParseResult(
        page_id="page-vl",
        provider="paddleocr_vl",
        source_image_path="processed.png",
        backend="fake",
        block_count=len(blocks),
        blocks=blocks,
        markdown_text=markdown,
    )


def test_vl_document_vocab_extraction_creates_one_entry_with_block_evidence() -> None:
    result = extract_from_document_parse(
        _parse(
            [
                DocumentParseBlock(
                    id="block-vocab-1",
                    label="text",
                    content="あたらしい 新しい 새롭다",
                    bbox=[10, 20, 360, 52],
                    order=1,
                )
            ]
        ),
        DictionaryValidator(DICTIONARY_PATH),
    )

    assert result.page_type == "vocab_table"
    assert len(result.items) == 1
    item = result.items[0]
    assert item["surface"] == "新しい"
    assert item["reading"] == "あたらしい"
    assert item["meaning_ko"] == "새롭다"
    assert item["evidence_blocks"] == ["block-vocab-1"]
    assert item["field_evidence"]["surface"]["block_ids"] == ["block-vocab-1"]


def test_vl_document_mcq_extraction_preserves_sentence_order_and_choices() -> None:
    result = extract_from_document_parse(
        _parse(
            [
                DocumentParseBlock(
                    id="block-q10",
                    label="text",
                    content="10 にわに しろい \\underline{\\text{はな}}が さきました。 1 木 2 花 3 木 4 花",
                    bbox=[10, 100, 720, 150],
                    order=1,
                ),
                DocumentParseBlock(
                    id="block-answer",
                    label="text",
                    content="答 10 4",
                    bbox=[10, 760, 420, 785],
                    order=2,
                ),
            ]
        ),
        DictionaryValidator(DICTIONARY_PATH),
    )

    assert result.page_type == "spelling_mcq"
    assert result.answer_map == {10: 4}
    assert len(result.items) == 1
    item = result.items[0]
    assert item["sentence"] == "にわに しろい はなが さきました。"
    assert item["target"] == "はな"
    assert item["choices"] == ["木", "花", "木", "花"]
    assert item["correct_choice_no"] == 4
    assert item["correct_answer"] == "花"
    assert item["evidence_blocks"] == ["block-q10"]


def test_vl_document_mcq_keeps_separate_choice_blocks_with_question() -> None:
    result = extract_from_document_parse(
        _parse(
            [
                DocumentParseBlock(
                    id="block-q10",
                    label="text",
                    content="10 にわに しろい \\underline{\\text{はな}}が さきました。",
                    bbox=[10, 100, 720, 125],
                    order=1,
                ),
                DocumentParseBlock(id="block-c1", label="text", content="1 木", bbox=[10, 130, 70, 150], order=2),
                DocumentParseBlock(id="block-c2", label="text", content="2 花", bbox=[120, 130, 180, 150], order=3),
                DocumentParseBlock(id="block-c3", label="text", content="3 犬", bbox=[230, 130, 290, 150], order=4),
                DocumentParseBlock(id="block-c4", label="text", content="4 山", bbox=[340, 130, 400, 150], order=5),
                DocumentParseBlock(id="block-answer", label="text", content="答 10 2", bbox=[10, 760, 420, 785], order=6),
            ]
        ),
        DictionaryValidator(DICTIONARY_PATH),
    )

    assert len(result.items) == 1
    item = result.items[0]
    assert item["question_no"] == 10
    assert item["choices"] == ["木", "花", "犬", "山"]
    assert item["correct_choice_no"] == 2
    assert item["correct_answer"] == "花"
    assert item["evidence_blocks"] == ["block-q10", "block-c1", "block-c2", "block-c3", "block-c4"]
    assert item["field_evidence"]["choice_2"]["block_ids"] == ["block-q10", "block-c1", "block-c2", "block-c3", "block-c4"]


def test_vl_document_mcq_keeps_flattened_questions_before_answer_fragment() -> None:
    result = extract_from_document_parse(
        _parse(
            [
                DocumentParseBlock(
                    id="flat-page",
                    label="ocr",
                    content=(
                        "1 にわに しろい \\underline{\\text{はな}}が さきました。 1 木 2 花 3 犬 4 山 "
                        "2 その まちには \\underline{\\text{がっこう}}が あります。 1 学校 2 学校 3 学校 4 学校 "
                        "答 12 24"
                    ),
                    bbox=[0, 0, 1013, 1800],
                    order=1,
                )
            ]
        ),
        DictionaryValidator(DICTIONARY_PATH),
    )

    assert result.answer_map == {1: 2, 2: 4}
    assert [item["question_no"] for item in result.items] == [1, 2]
    assert result.items[0]["correct_answer"] == "花"
    assert result.items[1]["correct_answer"] == "学校"


def test_vl_document_splits_flattened_questions_before_latex_answer_marker() -> None:
    result = extract_from_document_parse(
        _parse(
            [
                DocumentParseBlock(
                    id="flat-latex-page",
                    label="ocr",
                    content=(
                        "1 わたしは \\underline{\\text{ほん}}を よみます。 1 本 2 木 3 水 4 火 "
                        "2 まいにち \\underline{\\text{かんじ}}を おぼえます。 1 新しい 2 新しい 3 新い 4 新い "
                        "\\text{日 }12 24"
                    ),
                    bbox=[0, 0, 1013, 1800],
                    order=1,
                )
            ]
        ),
        DictionaryValidator(DICTIONARY_PATH),
    )

    assert result.answer_map == {1: 2, 2: 4}
    assert [item["question_no"] for item in result.items] == [1, 2]
    assert result.items[0]["choices"] == ["本", "木", "水", "火"]
    assert result.items[1]["choices"] == ["新しい", "新しい", "新い", "新い"]


def test_vl_document_mcq_handles_circled_questions_above_ten() -> None:
    result = extract_from_document_parse(
        _parse(
            [
                DocumentParseBlock(
                    id="block-q11",
                    label="ocr",
                    content="⑪ きのう ともだちに \\underline{\\text{あいました}}。 1 会いました 2 合いました 3 買いました 4 開いました\n答 ⑪ 2",
                    bbox=[10, 100, 720, 150],
                    order=1,
                )
            ]
        ),
        DictionaryValidator(DICTIONARY_PATH),
    )

    assert result.answer_map == {11: 2}
    assert len(result.items) == 1
    assert result.items[0]["question_no"] == 11
    assert result.items[0]["correct_answer"] == "合いました"


def test_vl_document_does_not_treat_japanese_day_as_answer_marker() -> None:
    result = extract_from_document_parse(
        _parse(
            [
                DocumentParseBlock(
                    id="block-day",
                    label="ocr",
                    content="1 つぎの 日 1 にち 2 ひ 3 じつ 4 び\n12",
                    bbox=[10, 100, 720, 150],
                    order=1,
                )
            ]
        ),
        DictionaryValidator(DICTIONARY_PATH),
    )

    assert result.answer_map == {1: 2}
    assert len(result.items) == 1
    assert result.items[0]["choices"] == ["にち", "ひ", "じつ", "び"]
    assert result.items[0]["correct_answer"] == "ひ"


def test_vl_document_reading_mcq_infers_unmarked_targets_from_answer_reading() -> None:
    result = extract_from_document_parse(
        _parse(
            [
                DocumentParseBlock(
                    id="block-reading",
                    label="ocr",
                    content=(
                        "1 わたしの 会社は 土よう日と 日よう日が 休みです。\n"
                        "1 とようび 2 どようび 3 かようび 4 がようび\n"
                        "12"
                    ),
                    bbox=[0, 0, 1013, 1800],
                    order=1,
                )
            ]
        ),
        DictionaryValidator(DICTIONARY_PATH),
    )

    assert result.answer_map == {1: 2}
    assert len(result.items) == 1
    item = result.items[0]
    assert item["target"] == "土よう日"
    assert item["correct_choice_no"] == 2
    assert item["correct_answer"] == "どようび"


def test_vl_document_warns_when_only_page_level_geometry_is_available() -> None:
    result = extract_from_document_parse(
        _parse(
            [
                DocumentParseBlock(
                    id="page-block",
                    label="ocr",
                    content="あたらしい 新しい 새롭다",
                    bbox=[0, 0, 1013, 1800],
                    order=1,
                )
            ]
        )
    )

    assert "PaddleOCR-VL returned page-level block geometry only; visual evidence is semantic but not field-precise." in result.warnings


def test_vl_document_mcq_missing_choices_sets_review_warning() -> None:
    result = extract_from_document_parse(
        _parse(
            [
                DocumentParseBlock(
                    id="block-q2",
                    label="text",
                    content="2 まいにち \\underline{\\text{あたらしい}} かんじを いつつ おぼえます。 1 新しい 2 新しい",
                    bbox=[10, 100, 720, 150],
                    order=1,
                )
            ]
        )
    )

    assert len(result.items) == 1
    item = result.items[0]
    assert item["choices"] == ["新しい", "新しい", "", ""]
    assert "Could not extract exactly four choices." in item["warnings"]


def test_vl_document_missing_choice_keeps_correct_answer_numbered_slot() -> None:
    result = extract_from_document_parse(
        _parse(
            [
                DocumentParseBlock(
                    id="block-q2",
                    label="text",
                    content="2 まいにち \\underline{\\text{あたらしい}} かんじを おぼえます。 2 正しい 3 間違い 4 違う\n答 2 2",
                    bbox=[10, 100, 720, 150],
                    order=1,
                )
            ]
        )
    )

    assert len(result.items) == 1
    item = result.items[0]
    assert item["choices"] == ["", "正しい", "間違い", "違う"]
    assert item["correct_choice_no"] == 2
    assert item["correct_answer"] == "正しい"
    assert "Could not extract exactly four choices." in item["warnings"]
