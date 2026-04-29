from __future__ import annotations

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.script import classify_script
from app.api import routes
from app.db import database
from app.extraction.answer_strip import parse_answer_strip_text
from app.extraction.cards import mcq_cards
from app.extraction.mcq import extract_mcq_items
from app.extraction.sentence_order import repair_predicate_first_sentence
from app.export.tsv import clean_tsv_field
from app.models.schemas import CardCandidate, DocumentParseResult, FieldOcrPreviewResponse, OcrToken, Page, ProcessResult
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
    assert items[1]["field_evidence"]["sentence"]["bbox"]
    assert items[1]["field_evidence"]["choice_2"]["text"] == "新しい駅"
    assert items[1]["field_evidence"]["correct_answer"]["text"] == "新しい"


def test_mcq_sentence_pass_excludes_choices_and_bleedthrough_noise() -> None:
    tokens = [
        _token("q2-no", "2", 10, 100, "number"),
        _token("q2-s1", "まいにち", 40, 100, "hiragana"),
        _token("q2-separator", "-", 112, 100, "punctuation", confidence=0.2),
        _token("q2-target", "あたらしい", 130, 100, "hiragana"),
        _token("q2-s2", "かんじを", 230, 100, "hiragana"),
        _token("q2-s3", "いつつ", 320, 100, "hiragana"),
        _token("q2-s4", "おぼえます。", 390, 100, "hiragana"),
        _token("q2-bleed", "しよさい、な -", 520, 100, "hiragana", confidence=0.3),
        _token("q2-c1-no", "1", 40, 145, "number"),
        _token("q2-c1", "新しい", 80, 145, "kanji"),
        _token("q2-c2-no", "2", 210, 145, "number"),
        _token("q2-c2", "新しい", 250, 145, "kanji"),
        _token("q2-c3-no", "3", 380, 145, "number"),
        _token("q2-c3", "新い", 420, 145, "kanji"),
        _token("q2-c4-no", "4", 550, 145, "number"),
        _token("q2-c4", "新い", 590, 145, "kanji"),
    ]

    [item] = extract_mcq_items(tokens, {2: 1}, "spelling_mcq")

    assert item["sentence"] == "まいにちあたらしいかんじをいつつおぼえます。"
    assert item["choices"] == ["新しい", "新しい", "新い", "新い"]
    assert item["field_evidence"]["sentence"]["bbox"][3] == 120
    assert "q2-bleed" not in item["field_evidence"]["sentence"]["token_ids"]
    assert "q2-c1" not in item["field_evidence"]["sentence"]["token_ids"]


def test_mcq_extraction_repairs_predicate_first_sentence_order() -> None:
    tokens = [
        _token("q10-no", "10", 10, 100, "number"),
        _token("q10-reversed", "さきました。はながしろいにわに", 40, 100, "hiragana"),
        _token("q10-c1", "1木", 40, 135, "mixed"),
        _token("q10-c2", "2花", 130, 135, "mixed"),
        _token("q10-c3", "3木", 220, 135, "mixed"),
        _token("q10-c4", "4花", 310, 135, "mixed"),
    ]

    [item] = extract_mcq_items(tokens, {10: 4}, "spelling_mcq")

    assert item["sentence"] == "にわにしろいはながさきました。"
    assert item["choices"] == ["木", "花", "木", "花"]


def test_predicate_first_repair_uses_general_locative_after_subject() -> None:
    repaired, changed = repair_predicate_first_sentence("あそびました。こどもがこうえんでたのしく")

    assert changed is True
    assert repaired == "こうえんでたのしくこどもがあそびました。"


def test_mcq_extraction_recovers_split_choice_lines_and_avoids_low_confidence_overwrite() -> None:
    tokens = [
        _token("q7-noise", "子", 10, 100, "kanji", confidence=0.4),
        _token("q7-sentence", "ともだちが外国からきました。", 30, 100, "mixed"),
        _token("q7-c1", "1がいごく", 30, 130, "mixed"),
        _token("q7-c2", "２かいこく", 120, 130, "mixed"),
        _token("q7-c34", "3がいこくて４かいごく", 210, 130, "mixed"),
        _token("q10-no", "10", 10, 240, "number"),
        _token("q10-sentence", "このしろいさかなは高いです。", 30, 240, "mixed"),
        _token("q10-c3", "３ふとい", 210, 245, "mixed"),
        _token("q10-c4", "4たかい", 300, 245, "mixed"),
        _token("q10-c2", "２ひくい", 120, 255, "mixed"),
        _token("q10-c1", "1ほそい", 30, 270, "mixed"),
        _token("q6-sentence", "じぶんのものにはなまえをかいてください。", 30, 360, "hiragana"),
        _token("q6-c1-good", "1各前", 30, 390, "mixed", confidence=0.99),
        _token("q6-c2", "２名前", 120, 390, "mixed"),
        _token("q6-c3", "３各前", 210, 390, "mixed"),
        _token("q6-c4", "1４名前", 300, 390, "mixed"),
        _token("q6-c1-noise", "1山", 390, 400, "mixed", confidence=0.45),
    ]

    reading_items = extract_mcq_items(tokens[:11], {}, "reading_mcq")
    spelling_items = extract_mcq_items(tokens[11:], {}, "spelling_mcq")
    missing_marker_items = extract_mcq_items(
        [
            _token("q5-sentence", "あしたのてんきははれるでしょう", 30, 20, "hiragana"),
            _token("q5-c1", "天気", 30, 55, "kanji"),
            _token("q5-c2", "２天気", 120, 55, "mixed"),
            _token("q5-c3", "３夫気", 210, 55, "mixed"),
            _token("q5-c4", "４夫気", 300, 55, "mixed"),
        ],
        {},
        "spelling_mcq",
    )

    assert reading_items[0]["choices"] == ["がいごく", "かいこく", "がいこくて", "かいごく"]
    assert reading_items[1]["choices"] == ["ほそい", "ひくい", "ふとい", "たかい"]
    assert reading_items[1]["correct_choice_no"] == 4
    assert spelling_items[0]["choices"] == ["各前", "名前", "各前", "名前"]
    assert spelling_items[0]["correct_choice_no"] == 2
    assert missing_marker_items[0]["choices"] == ["天気", "天気", "夫気", "夫気"]
    assert missing_marker_items[0]["correct_choice_no"] == 1


def test_mcq_extraction_recovers_missing_choice_marker_by_position() -> None:
    tokens = [
        _token("q2-no", "2", 10, 100, "number"),
        _token("q2-sentence", "あとでおふろに入ります。", 40, 100, "mixed"),
        _token("q2-c1", "1はいります", 40, 135, "mixed"),
        _token("q2-c2", "2しま叩ます", 160, 135, "mixed"),
        _token("q2-c3-missing-marker", "：い叩ます", 300, 135, "mixed"),
        _token("q2-c4", "4おります", 430, 135, "mixed"),
    ]

    [item] = extract_mcq_items(tokens, {2: 1}, "reading_mcq")

    assert item["choices"] == ["はいります", "しまります", "いります", "おります"]
    assert item["warnings"] == []


def test_mcq_cards_deduplicate_structural_warnings() -> None:
    [card] = mcq_cards(
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
    assert card.note_type == "jp_spelling_mcq_recall"


def test_mcq_cards_generate_one_semantic_card_per_question() -> None:
    cards = mcq_cards(
        "page-1",
        {
            "id": "q-2",
            "question_type": "reading_mcq",
            "sentence": "その本は上のたなにあるよ。",
            "target": "上",
            "choices": ["うえ", "した", "とまる", "うま"],
            "correct_choice_no": 1,
            "correct_answer": "うえ",
            "bbox": [1, 2, 3, 4],
            "confidence": 0.95,
            "warnings": [],
            "answer_source": "answer_strip",
        },
    )

    assert len(cards) == 1
    assert cards[0].back == "うえ"
    assert not cards[0].note_type.endswith("_exam")


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
    assert page.upload_name == "Chapter 1 review.jpg"
    assert page.original_image_path.endswith(".jpg")
    assert (tmp_path / "uploads" / f"{page_id}.jpg").read_bytes() == b"fake image bytes"


def test_upload_page_replaces_same_filename_state_and_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "upload-replace.db")
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

    old_upload = upload_dir / "page-existing.jpg"
    old_processed = processed_dir / "page-existing.png"
    old_crop = crop_dir / "page-existing_card_sentence.png"
    old_upload.write_bytes(b"old upload")
    old_processed.write_bytes(b"old processed")
    old_crop.write_bytes(b"old crop")
    database.upsert_page(
        Page(
            id="page-existing",
            original_image_path=str(old_upload),
            upload_name="Chapter 1 review.jpg",
            display_name="Renamed chapter page",
            processed_image_path=str(old_processed),
            page_type="reading_mcq",
            page_type_confidence=0.9,
            warnings=["old warning"],
            created_at="2026-04-28T00:00:00+00:00",
        )
    )
    database.replace_tokens("page-existing", [_token("old-token", "古い", 1, 1, "kanji")])
    database.replace_cards(
        "page-existing",
        [
            CardCandidate(
                id="old-card",
                page_id="page-existing",
                source_type="question_item",
                source_id="old-source",
                source={},
                note_type="jp_reading_mcq_recall",
                front="old",
                back="old",
                confidence=0.5,
                review_state="yellow",
                warnings=[],
            )
        ],
    )
    client = TestClient(app)

    response = client.post(
        "/api/pages/upload",
        files={"file": ("Chapter 1 review.jpg", b"new image bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json() == {"page_id": "page-existing", "status": "replaced"}
    page = database.get_page("page-existing")
    assert page is not None
    assert page.display_name == "Renamed chapter page"
    assert page.upload_name == "Chapter 1 review.jpg"
    assert page.page_type == "uploaded"
    assert page.processed_image_path is None
    assert page.warnings == []
    assert old_upload.read_bytes() == b"new image bytes"
    assert not old_processed.exists()
    assert not old_crop.exists()
    assert database.get_tokens("page-existing") == []
    assert database.get_cards("page-existing") == []


def test_upload_page_does_not_replace_by_renamed_display_name(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "upload-display-name.db")
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(routes, "UPLOAD_DIR", upload_dir)
    database.init_db()
    existing_upload = upload_dir / "page-renamed.jpg"
    existing_upload.write_bytes(b"existing")
    database.upsert_page(
        Page(
            id="page-renamed",
            original_image_path=str(existing_upload),
            upload_name="original filename.jpg",
            display_name="Chapter 1 review",
            page_type="reading_mcq",
            page_type_confidence=0.9,
            warnings=[],
            created_at="2026-04-28T00:00:00+00:00",
        )
    )
    client = TestClient(app)

    response = client.post(
        "/api/pages/upload",
        files={"file": ("Chapter 1 review.jpg", b"new image bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "uploaded"
    assert response.json()["page_id"] != "page-renamed"
    assert database.get_page("page-renamed") is not None
    assert existing_upload.read_bytes() == b"existing"


def test_process_page_accepts_explicit_ocr_engine(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "process-engine.db")
    database.init_db()
    page = Page(
        id="page-engine",
        original_image_path=str(tmp_path / "page.jpg"),
        display_name="Engine page",
        page_type="uploaded",
        page_type_confidence=0.0,
        warnings=[],
        created_at="2026-04-28T00:00:00+00:00",
    )
    database.upsert_page(page)
    captured: dict[str, str | float] = {}

    def fake_process_page(page_arg: Page, engine: str = "paddleocr") -> ProcessResult:
        captured["page_id"] = page_arg.id
        captured["engine"] = engine
        return ProcessResult(page=page_arg, tokens=[], cards=[], script_summary={}, answer_map={})

    def fake_worker(page_id: str, engine: str, **kwargs) -> ProcessResult:
        captured["page_id"] = page_id
        captured["engine"] = engine
        captured["max_rss_mb"] = kwargs["max_rss_mb"]
        return ProcessResult(page=page, tokens=[], cards=[], script_summary={}, answer_map={})

    monkeypatch.setattr(routes, "process_page", fake_process_page)
    monkeypatch.setattr(routes, "run_page_process_worker", fake_worker)
    client = TestClient(app)

    response = client.post("/api/pages/page-engine/process?engine=paddleocr_vl")

    assert response.status_code == 200
    assert captured == {
        "page_id": "page-engine",
        "engine": "paddleocr_vl",
        "max_rss_mb": routes.OCR_VL_PAGE_WORKER_MAX_RSS_MB,
    }


def test_document_parse_route_uses_bounded_worker(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "document-parse.db")
    database.init_db()
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"image bytes")
    database.upsert_page(
        Page(
            id="page-document",
            original_image_path=str(image_path),
            display_name="Document page",
            page_type="uploaded",
            page_type_confidence=0.0,
            warnings=[],
            created_at="2026-04-28T00:00:00+00:00",
        )
    )
    captured: dict[str, str | float] = {}

    def fake_document_worker(path, page_id: str, **kwargs) -> DocumentParseResult:
        captured["path"] = str(path)
        captured["page_id"] = page_id
        captured["max_rss_mb"] = kwargs["max_rss_mb"]
        return DocumentParseResult(
            page_id=page_id,
            provider="paddleocr_vl",
            source_image_path=str(path),
            backend="fake",
            block_count=1,
            markdown_text="text",
            warnings=[],
        )

    monkeypatch.setattr(routes, "run_document_parse_worker", fake_document_worker)
    client = TestClient(app)

    response = client.post("/api/pages/page-document/document/parse")

    assert response.status_code == 200
    assert captured == {
        "path": str(image_path),
        "page_id": "page-document",
        "max_rss_mb": routes.OCR_VL_PAGE_WORKER_MAX_RSS_MB,
    }
    assert response.json()["block_count"] == 1


def test_dedupe_pages_keeps_newest_upload_and_removes_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "dedupe.db")
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
    old_upload = upload_dir / "old.jpg"
    old_processed = processed_dir / "old.png"
    old_crop = crop_dir / "page-old_card_sentence.png"
    old_upload.write_bytes(b"old")
    old_processed.write_bytes(b"processed")
    old_crop.write_bytes(b"crop")
    database.upsert_page(
        Page(
            id="page-old",
            original_image_path=str(old_upload),
            upload_name="same.jpg",
            display_name="Same",
            processed_image_path=str(old_processed),
            page_type="uploaded",
            page_type_confidence=0.0,
            warnings=[],
            created_at="2026-04-27T00:00:00+00:00",
        )
    )
    database.upsert_page(
        Page(
            id="page-new",
            original_image_path=str(upload_dir / "new.jpg"),
            upload_name="same.jpg",
            display_name="Same",
            page_type="uploaded",
            page_type_confidence=0.0,
            warnings=[],
            created_at="2026-04-28T00:00:00+00:00",
        )
    )
    client = TestClient(app)

    response = client.post("/api/pages/dedupe")

    assert response.status_code == 200
    assert response.json()["removed_count"] == 1
    assert database.get_page("page-new") is not None
    assert database.get_page("page-old") is None
    assert not old_upload.exists()
    assert not old_processed.exists()
    assert not old_crop.exists()


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


def test_card_patch_persists_field_evidence_and_review_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "patch.db")
    database.init_db()
    database.replace_cards(
        "counted-page",
        [_card("card-patch", status="pending_review", review_state="yellow")],
    )
    client = TestClient(app)

    response = client.patch(
        "/api/cards/card-patch",
        json={
            "confidence": 0.64,
            "review_state": "red",
            "source_bbox": [1, 2, 3, 4],
            "warnings": ["manual check"],
            "source": {"field_evidence": {"target": {"bbox": [1, 2, 3, 4], "text": "上"}}},
        },
    )

    assert response.status_code == 200
    updated = database.get_card("card-patch")
    assert updated is not None
    assert updated.confidence == pytest.approx(0.64)
    assert updated.review_state == "red"
    assert updated.source_bbox == [1, 2, 3, 4]
    assert updated.warnings == ["manual check"]
    assert updated.source["field_evidence"]["target"]["text"] == "上"


def test_field_ocr_preview_does_not_mutate_card(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "field-preview.db")
    image_path = tmp_path / "page.png"
    from PIL import Image

    Image.new("RGB", (120, 80), "white").save(image_path)
    database.init_db()
    database.upsert_page(
        Page(
            id="page-field",
            original_image_path=str(image_path),
            display_name="Field page",
            page_type="reading_mcq",
            page_type_confidence=0.9,
            image_width=120,
            image_height=80,
            warnings=[],
            created_at="2026-04-28T00:00:00+00:00",
        )
    )
    original = CardCandidate(
        id="card-field",
        page_id="page-field",
        source_type="question_item",
        source_id="q-field",
        source={"target": "上"},
        note_type="jp_reading_mcq_recall",
        front="front",
        back="back",
        confidence=0.9,
        review_state="green",
        warnings=[],
    )
    database.replace_cards("page-field", [original])

    class FakeWorker:
        def preview(self, **kwargs):
            return FieldOcrPreviewResponse(
                card_id=kwargs["card_id"],
                page_id=kwargs["page_id"],
                field=kwargs["field"],
                bbox=kwargs["bbox"],
                provider="paddle",
                text="うえ",
                confidence=0.99,
                suggested_source={"target": "うえ"},
                field_evidence={"bbox": kwargs["bbox"], "text": "うえ"},
                worker={"state": "running"},
            )

    monkeypatch.setattr(routes, "crop_ocr_worker", FakeWorker())
    client = TestClient(app)

    response = client.post("/api/cards/card-field/field-ocr/preview", json={"field": "target", "bbox": [5, 5, 80, 30]})

    assert response.status_code == 200
    assert response.json()["suggested_source"] == {"target": "うえ"}
    assert database.get_card("card-field").source == original.source


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
