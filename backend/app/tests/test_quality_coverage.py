from __future__ import annotations

import csv
import json
from io import StringIO

import pytest

from app.core.script import char_script, classify_script, script_summary
from app.evaluation.golden import load_golden_pages, meaning_matches
from app.export.anki_csv import VOCAB_FILE_HEADERS, cards_to_csv, vocab_key, write_csv, write_export_csvs
from app.models.schemas import CardCandidate
from app.validation.dictionary import DictionaryValidator


def test_script_detection_covers_all_supported_buckets() -> None:
    assert char_script("あ") == "hiragana"
    assert char_script("ア") == "katakana"
    assert char_script("学") == "kanji"
    assert char_script("한") == "hangul"
    assert char_script("４") == "number"
    assert char_script("A") == "latin"
    assert char_script("。") == "punctuation"
    assert char_script("\u0378") == "other"

    assert classify_script("!? ") == "punctuation"
    assert classify_script("１２3") == "number"
    assert classify_script("学校がっこう") == "mixed"
    assert script_summary(["学校", "がっこう", "학교", "123"]) == {
        "kanji": 1,
        "hiragana": 1,
        "hangul": 1,
        "number": 1,
    }


def test_cards_to_csv_writes_anki_headers_and_round_trips_fields(tmp_path) -> None:
    card = CardCandidate(
        id="card-1",
        page_id="page-1",
        source_type="vocab_item",
        source_id="vocab-1",
        source={"surface": '学校, "がっこう"\t校', "reading": "がっこう", "meaning_ko": "학교"},
        note_type="jp_vocab_entry",
        front='学校, "がっこう"\t校',
        back="school\n학교<br>学校",
        tags=["jlpt", "needs-review"],
        confidence=0.87654,
        source_bbox=[1, 2, 3, 4],
        warnings=[],
    )

    csv_text = cards_to_csv([card])

    assert csv_text.splitlines()[: len(VOCAB_FILE_HEADERS)] == VOCAB_FILE_HEADERS
    data_lines = [line for line in csv_text.splitlines() if not line.startswith("#")]
    rows = list(csv.reader(StringIO("\n".join(data_lines))))
    assert rows == [
        [
            vocab_key('学校, "がっこう"\t校', "がっこう", "학교"),
            '学校, "がっこう"\t校',
            "がっこう",
            "학교",
            "1",
            "",
            "",
            "",
            "page-1",
            "[1,2,3,4]",
            "0.877",
            "",
            "jlpt needs-review",
        ]
    ]

    output = tmp_path / "exports" / "cards.csv"
    write_csv(output, [card])
    assert output.read_text(encoding="utf-8") == csv_text


def test_cards_to_csv_rejects_mixed_or_empty_schemas() -> None:
    vocab_card = CardCandidate(
        id="vocab-card",
        page_id="page-1",
        source_type="vocab_item",
        source_id="vocab-1",
        source={"surface": "学校", "reading": "がっこう", "meaning_ko": "학교"},
        note_type="jp_vocab_entry",
        front="front",
        back="back",
    )
    mcq_card = CardCandidate(
        id="mcq-card",
        page_id="page-1",
        source_type="question_item",
        source_id="q-1",
        source={"question_no": 1},
        note_type="jp_reading_mcq_recall",
        front="front",
        back="back",
    )

    with pytest.raises(ValueError, match="at least one"):
        cards_to_csv([])
    with pytest.raises(ValueError, match="one export schema"):
        cards_to_csv([vocab_card, mcq_card])


def test_vocab_csv_omits_notes_with_no_enabled_study_direction() -> None:
    card = CardCandidate(
        id="vocab-card",
        page_id="page-1",
        source_type="vocab_item",
        source_id="vocab-1",
        source={
            "surface": "学校",
            "reading": "がっこう",
            "meaning_ko": "학교",
            "study_writing": False,
            "study_reading": False,
            "study_meaning": False,
        },
        note_type="jp_vocab_entry",
        front="front",
        back="back",
    )

    csv_text = cards_to_csv([card])

    assert csv_text.splitlines() == VOCAB_FILE_HEADERS


def test_vocab_csv_preserves_enabled_study_direction_fields() -> None:
    card = CardCandidate(
        id="vocab-card",
        page_id="page-1",
        source_type="vocab_item",
        source_id="vocab-1",
        source={
            "surface": "学校",
            "reading": "がっこう",
            "meaning_ko": "학교",
            "study_writing": False,
            "study_reading": True,
            "study_meaning": False,
            "study_japanese_to_korean": False,
        },
        note_type="jp_vocab_entry",
        front="がっこう",
        back="学校",
    )

    csv_text = cards_to_csv([card])
    data_lines = [line for line in csv_text.splitlines() if not line.startswith("#")]
    [row] = list(csv.reader(StringIO("\n".join(data_lines))))

    assert row[4:8] == ["", "1", "", ""]


def test_vocab_csv_allows_meaning_only_vocab_without_reading() -> None:
    card = CardCandidate(
        id="vocab-card",
        page_id="page-1",
        source_type="vocab_item",
        source_id="vocab-1",
        source={
            "vocab_type": "jp_ko_meaning",
            "surface": "学校",
            "reading": "",
            "meaning_ko": "학교",
            "study_writing": False,
            "study_reading": False,
            "study_meaning": False,
            "study_japanese_to_korean": True,
        },
        note_type="jp_vocab_entry",
        front="学校",
        back="학교",
    )

    csv_text = cards_to_csv([card])
    data_lines = [line for line in csv_text.splitlines() if not line.startswith("#")]
    [row] = list(csv.reader(StringIO("\n".join(data_lines))))

    assert row[:4] == [vocab_key("学校", "", "학교"), "学校", "", "학교"]
    assert row[4:8] == ["", "", "", "1"]


def test_vocab_csv_migrates_legacy_meaning_only_study_flag() -> None:
    card = CardCandidate(
        id="vocab-card",
        page_id="page-1",
        source_type="vocab_item",
        source_id="vocab-1",
        source={
            "vocab_type": "jp_ko_meaning",
            "surface": "学校",
            "reading": "",
            "meaning_ko": "학교",
            "study_writing": False,
            "study_reading": False,
            "study_meaning": True,
        },
        note_type="jp_vocab_entry",
        front="学校",
        back="학교",
    )

    csv_text = cards_to_csv([card])
    data_lines = [line for line in csv_text.splitlines() if not line.startswith("#")]
    [row] = list(csv.reader(StringIO("\n".join(data_lines))))

    assert row[4:8] == ["", "", "", "1"]


def test_vocab_csv_deduplicates_notes_by_surface_and_reading() -> None:
    weaker = CardCandidate(
        id="vocab-card-weak",
        page_id="page-1",
        source_type="vocab_item",
        source_id="vocab-1",
        source={
            "surface": "学校",
            "reading": "がっこう",
            "meaning_ko": "학",
            "study_writing": True,
            "study_reading": False,
            "study_meaning": False,
        },
        note_type="jp_vocab_entry",
        front="がっこう",
        back="学校",
        confidence=0.62,
        warnings=["Weak OCR evidence; verify this row manually."],
    )
    stronger = CardCandidate(
        id="vocab-card-strong",
        page_id="page-2",
        source_type="vocab_item",
        source_id="vocab-2",
        source={
            "surface": "学校",
            "reading": "がっこう",
            "meaning_ko": "학교",
            "study_writing": False,
            "study_reading": True,
            "study_meaning": False,
        },
        note_type="jp_vocab_entry",
        front="学校",
        back="がっこう",
        confidence=0.96,
        warnings=[],
    )

    csv_text = cards_to_csv([weaker, stronger])
    data_lines = [line for line in csv_text.splitlines() if not line.startswith("#")]
    [row] = list(csv.reader(StringIO("\n".join(data_lines))))

    assert vocab_key("学校", "がっこう", "학") == vocab_key("学校", "がっこう", "학교")
    assert row[:4] == [vocab_key("学校", "がっこう"), "学校", "がっこう", "학교"]
    assert row[4:8] == ["1", "1", "", ""]


def test_vocab_export_counts_deduplicated_notes_and_generated_cards(tmp_path) -> None:
    first = CardCandidate(
        id="vocab-card-first",
        page_id="page-1",
        source_type="vocab_item",
        source_id="vocab-1",
        source={
            "surface": "学校",
            "reading": "がっこう",
            "meaning_ko": "학교",
            "study_writing": True,
            "study_reading": False,
            "study_meaning": False,
        },
        note_type="jp_vocab_entry",
        front="がっこう",
        back="学校",
        confidence=0.94,
    )
    duplicate_with_extra_direction = CardCandidate(
        id="vocab-card-duplicate",
        page_id="page-2",
        source_type="vocab_item",
        source_id="vocab-2",
        source={
            "surface": " 学校 ",
            "reading": "がっこう",
            "meaning_ko": "학교",
            "study_writing": False,
            "study_reading": True,
            "study_meaning": False,
        },
        note_type="jp_vocab_entry",
        front="学校",
        back="がっこう",
        confidence=0.82,
    )

    files, note_count, generated_card_count = write_export_csvs(
        tmp_path,
        "export-test",
        [first, duplicate_with_extra_direction],
    )
    csv_text = (tmp_path / files[0]["filename"]).read_text(encoding="utf-8")
    data_lines = [line for line in csv_text.splitlines() if not line.startswith("#")]
    [row] = list(csv.reader(StringIO("\n".join(data_lines))))

    assert files[0]["row_count"] == 1
    assert note_count == 1
    assert generated_card_count == 2
    assert row[:4] == [vocab_key("学校", "がっこう"), "学校", "がっこう", "학교"]
    assert row[4:8] == ["1", "1", "", ""]


def test_dictionary_validator_handles_missing_invalid_and_unknown_pairs(tmp_path) -> None:
    dictionary_path = tmp_path / "dictionary.json"
    dictionary_path.write_text(json.dumps([{"surface": "学校", "reading": "がっこう"}]), encoding="utf-8")
    validator = DictionaryValidator(dictionary_path)

    assert validator.validate_vocab("学校", "がっこう") == ("valid", [])

    status, warnings = validator.validate_vocab("", "gakkou")
    assert status == "review"
    assert "Missing written form." in warnings
    assert "Reading is not kana-only." in warnings

    status, warnings = validator.validate_vocab("校", "こう")
    assert status == "review"
    assert warnings == ["Surface-reading pair was not found in the local dictionary."]


def test_golden_loader_skips_stubs_and_flattens_nested_rows(tmp_path) -> None:
    image_path = tmp_path / "page.jpg"
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(
        json.dumps(
            {
                "pages": [
                    {"page_id": "stub", "category": "vocab_table", "stub": True},
                    {"page_id": "ignored", "category": "notes", "image_path": "notes.jpg"},
                    {
                        "page_id": "page-1",
                        "category": "reading_mcq",
                        "expected_page_type": "reading_mcq",
                        "image_path": str(image_path),
                        "expected_entries": [[{"row_id": "r1", "surface": "学校", "reading": "がっこう", "meaning_ko": "학교"}]],
                        "expected_questions": [
                            [
                                {
                                    "question_id": "q1",
                                    "question_no": "1",
                                    "target": "学校",
                                    "choices": ["がっこう", "せんせい"],
                                    "correct_choice_no": "1",
                                    "correct_answer": "がっこう",
                                    "answer_source": "answer_strip",
                                }
                            ]
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    pages = load_golden_pages(golden_path, tmp_path)

    assert len(pages) == 1
    assert pages[0].expected_rows[0].surface == "学校"
    assert pages[0].expected_questions[0].correct_choice_no == 1
    assert meaning_matches("북쪽 출입구", "남쪽/북쪽")
    assert not meaning_matches("", "학교")
