from __future__ import annotations

import json
import sqlite3

from fastapi.testclient import TestClient

from app.main import app
from app.core.script import classify_script
from app.api import routes
from app.db import database
from app.extraction.answer_strip import parse_answer_strip_text
from app.extraction.cards import mcq_cards
from app.extraction.mcq import extract_mcq_items
from app.export.tsv import clean_tsv_field
from app.models.schemas import CardCandidate, OcrToken, Page
from fastapi import HTTPException


def test_script_classifier() -> None:
    assert classify_script("がっこう") == "hiragana"
    assert classify_script("学校") == "kanji"
    assert classify_script("학교") == "hangul"
    assert classify_script("学校が") == "mixed"


def test_answer_strip_parser() -> None:
    assert parse_answer_strip_text("1 2 2 3 3 1 10 4") == {1: 2, 2: 3, 3: 1, 10: 4}
    assert parse_answer_strip_text("① 2 ② 3") == {1: 2, 2: 3}


def test_mcq_extraction_keeps_printed_question_numbers_when_previous_blocks_are_absent() -> None:
    tokens = [
        _token("q5-no", "5", 10, 100, "number"),
        _token("q5-sentence", "あしたのてんきははれるでしょう。", 30, 100, "hiragana"),
        _token("q5-c1", "1 天気", 30, 130, "mixed"),
        _token("q5-c2", "2 天気", 120, 130, "mixed"),
        _token("q5-c3", "3 天気", 210, 130, "mixed"),
        _token("q5-c4", "4 天気", 300, 130, "mixed"),
        _token("q8-no", "8", 10, 220, "number"),
        _token("q8-sentence", "そのまちにはがっこうがいつつあります。", 30, 220, "hiragana"),
        _token("q8-c1", "1 学校", 30, 250, "mixed"),
        _token("q8-c2", "2 学校", 120, 250, "mixed"),
        _token("q8-c3", "3 学校", 210, 250, "mixed"),
        _token("q8-c4", "4 学校", 300, 250, "mixed"),
    ]

    items = extract_mcq_items(tokens, {5: 1, 8: 4}, "spelling_mcq")

    assert [item["question_no"] for item in items] == [5, 8]


def test_mcq_extraction_keeps_number_prefixed_question_sentences() -> None:
    tokens = [
        _token("q1-sentence", "1そのほんはうえのたなにあるよ。", 10, 100, "hiragana"),
        _token("q1-c1", "1上", 30, 130, "mixed"),
        _token("q1-c2", "2下", 120, 130, "mixed"),
        _token("q1-c3", "3止", 210, 130, "mixed"),
        _token("q1-c4", "4午", 300, 130, "mixed"),
        _token("q2-sentence-a", "まいにち", 10, 180, "hiragana"),
        _token("q2-noise", "-", 80, 180, "punctuation", confidence=0.1),
        _token("q2-sentence-b", "あたらしいかんじをいつつおぼえます。", 120, 180, "hiragana"),
        _token("q2-c1", "1新しい", 30, 210, "mixed"),
        _token("q2-c234", "２新しい駅３新い龍４新い", 120, 210, "mixed"),
        _token("q3-sentence", "3なつやすみにがいこくりょこうをするひとがおおくなっている。", 10, 300, "hiragana"),
        _token("q3-c1", "1大く", 30, 330, "mixed"),
        _token("q3-stray", "1", 105, 330, "number"),
        _token("q3-c2", "2太く", 120, 330, "mixed"),
        _token("q3-c3", "3広く", 210, 330, "mixed"),
        _token("q3-c4", "4多く", 300, 330, "mixed"),
        _token("q4-sentence", "だれかがきょうしつのそとにたっています。", 10, 420, "hiragana"),
        _token("q4-c1", "1赤って", 30, 450, "mixed"),
        _token("q4-c2", "2並って", 120, 450, "mixed"),
        _token("q4-c3", "3丘って", 210, 450, "mixed"),
        _token("q4-c4", "4立って", 300, 450, "mixed"),
    ]

    items = extract_mcq_items(tokens, {1: 1, 2: 1, 3: 4, 4: 4}, "spelling_mcq")

    assert [item["question_no"] for item in items] == [1, 2, 3, 4]
    assert items[0]["target"] == "うえ"
    assert items[1]["choices"] == ["新しい", "新しい駅", "新い龍", "新い"]
    assert items[1]["confidence"] > 0.9
    assert items[2]["choices"] == ["大く", "太く", "広く", "多く"]
    assert items[3]["correct_answer"] == "立って"


def test_mcq_cards_deduplicate_structural_warnings() -> None:
    [card, _] = mcq_cards(
        "page-1",
        {
            "id": "q-1",
            "question_type": "spelling_mcq",
            "sentence": "あしたのてんきははれるでしょう。",
            "target": "てんき",
            "choices": ["天気", "夫気", "夫气"],
            "correct_choice_no": 1,
            "correct_answer": "天気",
            "bbox": [1, 2, 3, 4],
            "confidence": 1.0,
            "warnings": ["Expected exactly four choices."],
            "answer_source": "answer_strip",
        },
    )

    assert card.review_state == "red"
    assert card.warnings == ["Expected exactly four choices."]


def test_tsv_cleaning() -> None:
    assert clean_tsv_field("a\tb\nc") == "a b<br>c"


def test_page_display_name_migrates_and_persists(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "legacy.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE pages (
                id TEXT PRIMARY KEY,
                original_image_path TEXT NOT NULL,
                processed_image_path TEXT,
                page_type TEXT NOT NULL,
                page_type_confidence REAL NOT NULL,
                image_width INTEGER,
                image_height INTEGER,
                warnings_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO pages (
                id, original_image_path, processed_image_path, page_type,
                page_type_confidence, image_width, image_height, warnings_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-page",
                str(tmp_path / "new upload (category 1 page).jpg"),
                None,
                "unprocessed",
                0.0,
                None,
                None,
                json.dumps([]),
                "2026-04-27T00:00:00+00:00",
            ),
        )

    database.init_db()
    assert database.get_page("legacy-page").display_name == "new upload (category 1 page)"

    updated = database.update_page_display_name("legacy-page", "Chapter 1 review")
    assert updated is not None
    assert updated.display_name == "Chapter 1 review"

    database.upsert_page(
        Page(
            id="legacy-page",
            original_image_path=str(tmp_path / "new upload (category 1 page).jpg"),
            display_name=None,
            processed_image_path=str(tmp_path / "processed.jpg"),
            page_type="vocab",
            page_type_confidence=0.99,
            image_width=1200,
            image_height=1600,
            warnings=[],
            created_at="2026-04-27T00:00:00+00:00",
        )
    )
    assert database.get_page("legacy-page").display_name == "Chapter 1 review"

    cleared = database.update_page_display_name("legacy-page", None)
    assert cleared is not None
    assert cleared.display_name == "new upload (category 1 page)"


def test_export_download_rejects_path_traversal(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(routes, "EXPORT_DIR", tmp_path)

    safe_file = tmp_path / "export_123.tsv"
    safe_file.write_text("note_type\tfront\n", encoding="utf-8")

    response = routes.download_export("export_123.tsv")
    assert response.path == safe_file

    for filename in ("../secret.tsv", "nested/export.tsv", "export_123.txt"):
        try:
            routes.download_export(filename)
        except HTTPException as exc:
            assert exc.status_code == 404
        else:
            raise AssertionError(f"{filename} should not be downloadable")


def test_list_pages_includes_card_counts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "counts.db")
    database.init_db()
    database.upsert_page(
        Page(
            id="counted-page",
            original_image_path=str(tmp_path / "page.jpg"),
            display_name="Counted page",
            page_type="vocab_table",
            page_type_confidence=1.0,
            warnings=[],
            created_at="2026-04-27T00:00:00+00:00",
        )
    )
    database.replace_cards(
        "counted-page",
        [
            _card("card-1", status="approved", review_state="green"),
            _card("card-2", status="pending_review", review_state="red"),
        ],
    )

    page = database.list_pages()[0]

    assert page.card_count == 2
    assert page.approved_card_count == 1
    assert page.red_card_count == 1


def test_upload_page_writes_file_and_defaults_display_name(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "upload.db")
    monkeypatch.setattr(routes, "UPLOAD_DIR", tmp_path / "uploads")
    routes.UPLOAD_DIR.mkdir()
    database.init_db()
    client = TestClient(app)

    response = client.post(
        "/api/pages/upload",
        files={"file": ("Chapter 1 review.jpg", b"fake image bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    page_id = response.json()["page_id"]
    page = database.get_page(page_id)
    assert page is not None
    assert page.display_name == "Chapter 1 review"
    assert page.original_image_path.endswith(".jpg")
    assert (tmp_path / "uploads" / f"{page_id}.jpg").read_bytes() == b"fake image bytes"


def test_delete_page_removes_database_rows_and_runtime_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "delete.db")
    upload_dir = tmp_path / "uploads"
    processed_dir = tmp_path / "processed"
    crop_dir = tmp_path / "crops"
    upload_dir.mkdir()
    processed_dir.mkdir()
    crop_dir.mkdir()
    monkeypatch.setattr(routes, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(routes, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(routes, "CROP_DIR", crop_dir)
    database.init_db()

    original_path = upload_dir / "page-delete.jpg"
    processed_path = processed_dir / "page-delete.png"
    original_path.write_bytes(b"original")
    processed_path.write_bytes(b"processed")
    page = Page(
        id="page-delete",
        original_image_path=str(original_path),
        display_name="Delete me",
        processed_image_path=str(processed_path),
        page_type="vocab_table",
        page_type_confidence=1.0,
        warnings=[],
        created_at="2026-04-28T00:00:00+00:00",
    )
    database.upsert_page(page)
    database.replace_cards("page-delete", [_card("card-delete", status="approved", review_state="green")])
    client = TestClient(app)

    response = client.delete("/api/pages/page-delete")

    assert response.status_code == 200
    assert response.json() == {"page_id": "page-delete", "status": "deleted"}
    assert database.get_page("page-delete") is None
    assert database.get_cards("page-delete") == []
    assert not original_path.exists()
    assert not processed_path.exists()


def _card(card_id: str, *, status: str, review_state: str) -> CardCandidate:
    return CardCandidate(
        id=card_id,
        page_id="counted-page",
        source_type="vocab_item",
        source_id="source-1",
        source={},
        note_type="jp_vocab_reading",
        front="front",
        back="back",
        confidence=0.9,
        status=status,
        review_state=review_state,
        warnings=[],
    )


def _token(token_id: str, text: str, x: float, y: float, script_class: str, confidence: float = 0.95) -> OcrToken:
    return OcrToken(
        id=token_id,
        page_id="page-mcq",
        text=text,
        bbox=[x, y, x + 70, y + 20],
        confidence=confidence,
        script_class=script_class,
        source="test",
    )
