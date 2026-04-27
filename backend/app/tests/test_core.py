from __future__ import annotations

import json
import sqlite3

from app.core.script import classify_script
from app.db import database
from app.extraction.answer_strip import parse_answer_strip_text
from app.export.tsv import clean_tsv_field
from app.models.schemas import Page


def test_script_classifier() -> None:
    assert classify_script("がっこう") == "hiragana"
    assert classify_script("学校") == "kanji"
    assert classify_script("학교") == "hangul"
    assert classify_script("学校が") == "mixed"


def test_answer_strip_parser() -> None:
    assert parse_answer_strip_text("1 2 2 3 3 1 10 4") == {1: 2, 2: 3, 3: 1, 10: 4}
    assert parse_answer_strip_text("① 2 ② 3") == {1: 2, 2: 3}


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
                "/tmp/new upload (category 1 page).jpg",
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
            original_image_path="/tmp/new upload (category 1 page).jpg",
            display_name=None,
            processed_image_path="/tmp/processed.jpg",
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
