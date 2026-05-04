from __future__ import annotations

import csv
import json
from io import StringIO

from app.core.script import char_script, classify_script, script_summary
from app.evaluation.golden import load_golden_pages, meaning_matches
from app.export.anki_csv import ANKI_FILE_HEADERS, cards_to_csv, write_csv
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
        note_type="jp_vocab_entry",
        front='学校, "がっこう"\t校',
        back="school\n학교<br>学校",
        tags=["jlpt", "needs-review"],
        confidence=0.87654,
        source_bbox=[1, 2, 3, 4],
        warnings=[],
    )

    csv_text = cards_to_csv([card])

    assert csv_text.splitlines()[: len(ANKI_FILE_HEADERS)] == ANKI_FILE_HEADERS
    data_lines = [line for line in csv_text.splitlines() if not line.startswith("#")]
    rows = list(csv.reader(StringIO("\n".join(data_lines))))
    assert rows == [
        [
            "jp_vocab_entry",
            '学校, "がっこう"\t校',
            "school<br>학교<br>学校",
            "page-1",
            "[1.0, 2.0, 3.0, 4.0]",
            "0.877",
            "jlpt needs-review",
        ]
    ]

    output = tmp_path / "exports" / "cards.csv"
    write_csv(output, [card])
    assert output.read_text(encoding="utf-8") == csv_text


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
