from __future__ import annotations

import os

import pytest

from app.export.anki_csv import cards_to_csv
from app.models.schemas import CardCandidate
from scripts.verify_anki_csv_import import parse_anki_csv, verify_with_anki


def test_verify_anki_csv_parser_accepts_export_contract(tmp_path) -> None:
    export_path = tmp_path / "cards.csv"
    export_path.write_text(cards_to_csv([_card()]), encoding="utf-8")

    rows = parse_anki_csv(export_path)

    assert len(rows) == 1
    assert rows[0].surface == "学校"
    assert rows[0].reading == "がっこう"
    assert rows[0].meaning_ko == "학교"
    assert rows[0].study_writing == "1"
    assert rows[0].study_reading == ""
    assert rows[0].study_meaning == ""
    assert rows[0].tags == "jlpt"


def test_verify_anki_csv_parser_rejects_missing_headers(tmp_path) -> None:
    export_path = tmp_path / "cards.csv"
    export_path.write_text("jp_vocab_entry,front,back,page,,,tag\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing Anki CSV header"):
        parse_anki_csv(export_path)


@pytest.mark.skipif(os.environ.get("RUN_ANKI_IMPORT_TESTS") != "1", reason="Real Anki import test is optional.")
def test_verify_anki_csv_imports_into_temporary_anki_collection(tmp_path) -> None:
    export_path = tmp_path / "cards.csv"
    export_path.write_text(cards_to_csv([_card()]), encoding="utf-8")

    assert verify_with_anki(export_path) == 1


def _card() -> CardCandidate:
    return CardCandidate(
        id="card-1",
        page_id="page-1",
        source_type="vocab_item",
        source_id="source-1",
        source={
            "surface": "学校",
            "reading": "がっこう",
            "meaning_ko": "학교",
            "study_meaning": False,
        },
        note_type="jp_vocab_entry",
        front="学校",
        back="がっこう",
        tags=["jlpt"],
        confidence=0.9,
        status="approved",
        review_state="green",
        warnings=[],
    )
