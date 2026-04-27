from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Any

from app.core.config import DB_PATH
from app.models.schemas import CardCandidate, OcrToken, Page


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS pages (
                id TEXT PRIMARY KEY,
                original_image_path TEXT NOT NULL,
                processed_image_path TEXT,
                page_type TEXT NOT NULL,
                page_type_confidence REAL NOT NULL,
                image_width INTEGER,
                image_height INTEGER,
                warnings_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ocr_tokens (
                id TEXT PRIMARY KEY,
                page_id TEXT NOT NULL,
                text TEXT NOT NULL,
                x1 REAL NOT NULL,
                y1 REAL NOT NULL,
                x2 REAL NOT NULL,
                y2 REAL NOT NULL,
                confidence REAL NOT NULL,
                script_class TEXT NOT NULL,
                source TEXT NOT NULL,
                FOREIGN KEY(page_id) REFERENCES pages(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS cards (
                id TEXT PRIMARY KEY,
                page_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                source_json TEXT NOT NULL,
                note_type TEXT NOT NULL,
                front TEXT NOT NULL,
                back TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                review_state TEXT NOT NULL,
                source_bbox_json TEXT,
                warnings_json TEXT NOT NULL,
                FOREIGN KEY(page_id) REFERENCES pages(id) ON DELETE CASCADE
            );
            """
        )


def upsert_page(page: Page) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO pages (
                id, original_image_path, processed_image_path, page_type,
                page_type_confidence, image_width, image_height, warnings_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                processed_image_path=excluded.processed_image_path,
                page_type=excluded.page_type,
                page_type_confidence=excluded.page_type_confidence,
                image_width=excluded.image_width,
                image_height=excluded.image_height,
                warnings_json=excluded.warnings_json
            """,
            (
                page.id,
                page.original_image_path,
                page.processed_image_path,
                page.page_type,
                page.page_type_confidence,
                page.image_width,
                page.image_height,
                json.dumps(page.warnings, ensure_ascii=False),
                page.created_at,
            ),
        )


def list_pages() -> list[Page]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM pages ORDER BY created_at DESC").fetchall()
    return [_page_from_row(row) for row in rows]


def get_page(page_id: str) -> Page | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM pages WHERE id = ?", (page_id,)).fetchone()
    return _page_from_row(row) if row else None


def replace_tokens(page_id: str, tokens: list[OcrToken]) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM ocr_tokens WHERE page_id = ?", (page_id,))
        conn.executemany(
            """
            INSERT INTO ocr_tokens
            (id, page_id, text, x1, y1, x2, y2, confidence, script_class, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    token.id,
                    token.page_id,
                    token.text,
                    token.bbox[0],
                    token.bbox[1],
                    token.bbox[2],
                    token.bbox[3],
                    token.confidence,
                    token.script_class,
                    token.source,
                )
                for token in tokens
            ],
        )


def get_tokens(page_id: str) -> list[OcrToken]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM ocr_tokens WHERE page_id = ?", (page_id,)).fetchall()
    return [_token_from_row(row) for row in rows]


def replace_cards(page_id: str, cards: list[CardCandidate]) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM cards WHERE page_id = ?", (page_id,))
        _insert_cards(conn, cards)


def upsert_card(card: CardCandidate) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM cards WHERE id = ?", (card.id,))
        _insert_cards(conn, [card])


def get_card(card_id: str) -> CardCandidate | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM cards WHERE id = ?", (card_id,)).fetchone()
    return _card_from_row(row) if row else None


def get_cards(page_id: str | None = None) -> list[CardCandidate]:
    with connect() as conn:
        if page_id:
            rows = conn.execute("SELECT * FROM cards WHERE page_id = ? ORDER BY id", (page_id,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM cards ORDER BY page_id, id").fetchall()
    return [_card_from_row(row) for row in rows]


def _insert_cards(conn: sqlite3.Connection, cards: list[CardCandidate]) -> None:
    conn.executemany(
        """
        INSERT INTO cards (
            id, page_id, source_type, source_id, source_json, note_type, front, back,
            tags_json, confidence, status, review_state, source_bbox_json, warnings_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                card.id,
                card.page_id,
                card.source_type,
                card.source_id,
                json.dumps(card.source, ensure_ascii=False),
                card.note_type,
                card.front,
                card.back,
                json.dumps(card.tags, ensure_ascii=False),
                card.confidence,
                card.status,
                card.review_state,
                json.dumps(card.source_bbox) if card.source_bbox else None,
                json.dumps(card.warnings, ensure_ascii=False),
            )
            for card in cards
        ],
    )


def _page_from_row(row: sqlite3.Row) -> Page:
    return Page(
        id=row["id"],
        original_image_path=row["original_image_path"],
        processed_image_path=row["processed_image_path"],
        page_type=row["page_type"],
        page_type_confidence=row["page_type_confidence"],
        image_width=row["image_width"],
        image_height=row["image_height"],
        warnings=json.loads(row["warnings_json"]),
        created_at=row["created_at"],
    )


def _token_from_row(row: sqlite3.Row) -> OcrToken:
    return OcrToken(
        id=row["id"],
        page_id=row["page_id"],
        text=row["text"],
        bbox=[row["x1"], row["y1"], row["x2"], row["y2"]],
        confidence=row["confidence"],
        script_class=row["script_class"],
        source=row["source"],
    )


def _card_from_row(row: sqlite3.Row) -> CardCandidate:
    source_bbox: Any = json.loads(row["source_bbox_json"]) if row["source_bbox_json"] else None
    return CardCandidate(
        id=row["id"],
        page_id=row["page_id"],
        source_type=row["source_type"],
        source_id=row["source_id"],
        source=json.loads(row["source_json"]),
        note_type=row["note_type"],
        front=row["front"],
        back=row["back"],
        tags=json.loads(row["tags_json"]),
        confidence=row["confidence"],
        status=row["status"],
        review_state=row["review_state"],
        source_bbox=source_bbox,
        warnings=json.loads(row["warnings_json"]),
    )
