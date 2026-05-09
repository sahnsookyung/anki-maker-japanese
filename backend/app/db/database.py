from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Any

from app.core.ids import new_id
from app.core.config import DB_PATH
from app.models.schemas import CardCandidate, DocumentParseResult, OcrRun, OcrToken, Page


_PAGE_WITH_ACTIVE_RUN_SQL = """
    SELECT
        pages.*,
        active_run.engine AS active_ocr_engine,
        active_run.completed_at AS active_ocr_completed_at,
        active_run.duration_ms AS active_ocr_duration_ms
    FROM pages
    LEFT JOIN ocr_runs AS active_run
        ON active_run.id = pages.active_ocr_run_id
"""
_OCR_RUN_BY_ID_SQL = "SELECT * FROM ocr_runs WHERE id = ?"

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
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
                upload_name TEXT,
                display_name TEXT,
                processed_image_path TEXT,
                active_ocr_run_id TEXT,
                page_type TEXT NOT NULL,
                page_type_confidence REAL NOT NULL,
                image_width INTEGER,
                image_height INTEGER,
                warnings_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ocr_runs (
                id TEXT PRIMARY KEY,
                page_id TEXT NOT NULL,
                engine TEXT NOT NULL,
                status TEXT NOT NULL,
                image_sha256 TEXT,
                processed_image_path TEXT,
                image_width INTEGER,
                image_height INTEGER,
                preprocessing_json TEXT NOT NULL,
                model_config_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                warnings_json TEXT NOT NULL,
                error TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                duration_ms INTEGER,
                FOREIGN KEY(page_id) REFERENCES pages(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS ocr_tokens (
                id TEXT PRIMARY KEY,
                page_id TEXT NOT NULL,
                run_id TEXT,
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
                run_id TEXT,
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
        _ensure_page_columns(conn)
        _ensure_run_columns(conn)
        _backfill_legacy_runs(conn)
        _delete_unsupported_vocab_cards(conn)
        _ensure_indexes(conn)


def upsert_page(page: Page) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO pages (
                id, original_image_path, upload_name, display_name, processed_image_path, active_ocr_run_id,
                page_type, page_type_confidence, image_width, image_height, warnings_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                original_image_path=excluded.original_image_path,
                upload_name=COALESCE(excluded.upload_name, pages.upload_name),
                display_name=COALESCE(excluded.display_name, pages.display_name),
                processed_image_path=excluded.processed_image_path,
                active_ocr_run_id=COALESCE(excluded.active_ocr_run_id, pages.active_ocr_run_id),
                page_type=excluded.page_type,
                page_type_confidence=excluded.page_type_confidence,
                image_width=excluded.image_width,
                image_height=excluded.image_height,
                warnings_json=excluded.warnings_json
            """,
            (
                page.id,
                page.original_image_path,
                page.upload_name,
                page.display_name,
                page.processed_image_path,
                page.active_ocr_run_id,
                page.page_type,
                page.page_type_confidence,
                page.image_width,
                page.image_height,
                json.dumps(page.warnings, ensure_ascii=False),
                page.created_at,
            ),
        )


def update_page_display_name(page_id: str, display_name: str | None) -> Page | None:
    with connect() as conn:
        conn.execute("UPDATE pages SET display_name = ? WHERE id = ?", (display_name, page_id))
        row = conn.execute(_PAGE_WITH_ACTIVE_RUN_SQL + " WHERE pages.id = ?", (page_id,)).fetchone()
    return _page_from_row(row) if row else None


def list_pages() -> list[Page]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT
                pages.*,
                active_run.engine AS active_ocr_engine,
                active_run.completed_at AS active_ocr_completed_at,
                active_run.duration_ms AS active_ocr_duration_ms,
                COALESCE(card_summary.card_count, 0) AS card_count,
                COALESCE(card_summary.approved_card_count, 0) AS approved_card_count,
                COALESCE(card_summary.red_card_count, 0) AS red_card_count
            FROM pages
            LEFT JOIN ocr_runs AS active_run
                ON active_run.id = pages.active_ocr_run_id
            LEFT JOIN (
                SELECT
                    page_id,
                    COUNT(*) AS card_count,
                    SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END) AS approved_card_count,
                    SUM(CASE WHEN review_state = 'red' THEN 1 ELSE 0 END) AS red_card_count
                FROM cards
                WHERE run_id IS NULL
                   OR run_id = (SELECT active_ocr_run_id FROM pages WHERE pages.id = cards.page_id)
                GROUP BY page_id
            ) AS card_summary ON card_summary.page_id = pages.id
            ORDER BY pages.created_at DESC
            """
        ).fetchall()
    return [_page_from_row(row) for row in rows]


def get_page(page_id: str) -> Page | None:
    with connect() as conn:
        row = conn.execute(_PAGE_WITH_ACTIVE_RUN_SQL + " WHERE pages.id = ?", (page_id,)).fetchone()
    return _page_from_row(row) if row else None


def get_page_by_upload_name(upload_name: str) -> Page | None:
    with connect() as conn:
        row = conn.execute(
            _PAGE_WITH_ACTIVE_RUN_SQL + " WHERE pages.upload_name = ? ORDER BY pages.created_at DESC LIMIT 1",
            (upload_name,),
        ).fetchone()
    return _page_from_row(row) if row else None


def delete_page(page_id: str) -> bool:
    with connect() as conn:
        conn.execute("DELETE FROM ocr_tokens WHERE page_id = ?", (page_id,))
        conn.execute("DELETE FROM cards WHERE page_id = ?", (page_id,))
        conn.execute("DELETE FROM ocr_runs WHERE page_id = ?", (page_id,))
        cursor = conn.execute("DELETE FROM pages WHERE id = ?", (page_id,))
        return cursor.rowcount > 0


def clear_page_runs(page_id: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM ocr_tokens WHERE page_id = ?", (page_id,))
        conn.execute("DELETE FROM cards WHERE page_id = ?", (page_id,))
        conn.execute("DELETE FROM ocr_runs WHERE page_id = ?", (page_id,))
        conn.execute("UPDATE pages SET active_ocr_run_id = NULL WHERE id = ?", (page_id,))


def start_ocr_run(
    page_id: str,
    engine: str,
    *,
    image_sha256: str | None = None,
    processed_image_path: str | None = None,
    preprocessing: dict[str, Any] | None = None,
    provider_config: dict[str, Any] | None = None,
) -> OcrRun:
    run = OcrRun(
        id=new_id("run"),
        page_id=page_id,
        engine=engine,
        status="running",
        image_sha256=image_sha256,
        processed_image_path=processed_image_path,
        preprocessing=preprocessing or {},
        provider_config=provider_config or {},
        metrics={},
        warnings=[],
        started_at=utc_now(),
    )
    with connect() as conn:
        _insert_ocr_run(conn, run)
    return run


def complete_ocr_run(
    run_id: str,
    *,
    warnings: list[str] | None = None,
    metrics: dict[str, Any] | None = None,
    processed_image_path: str | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
    activate: bool = True,
) -> OcrRun | None:
    completed_at = utc_now()
    with connect() as conn:
        row = conn.execute(_OCR_RUN_BY_ID_SQL, (run_id,)).fetchone()
        if not row:
            return None
        started_at = row["started_at"]
        duration_ms = _duration_ms(started_at, completed_at)
        conn.execute(
            """
            UPDATE ocr_runs
            SET status = 'succeeded',
                warnings_json = ?,
                metrics_json = ?,
                processed_image_path = COALESCE(?, processed_image_path),
                image_width = COALESCE(?, image_width),
                image_height = COALESCE(?, image_height),
                completed_at = ?,
                duration_ms = ?,
                error = NULL
            WHERE id = ?
            """,
            (
                json.dumps(warnings or [], ensure_ascii=False),
                json.dumps(metrics or {}, ensure_ascii=False),
                processed_image_path,
                image_width,
                image_height,
                completed_at,
                duration_ms,
                run_id,
            ),
        )
        updated = conn.execute(_OCR_RUN_BY_ID_SQL, (run_id,)).fetchone()
        if activate:
            conn.execute("UPDATE pages SET active_ocr_run_id = ? WHERE id = ?", (run_id, row["page_id"]))
    return _ocr_run_from_row(updated) if updated else None


def fail_ocr_run(run_id: str, error: str, *, warnings: list[str] | None = None) -> OcrRun | None:
    completed_at = utc_now()
    with connect() as conn:
        row = conn.execute(_OCR_RUN_BY_ID_SQL, (run_id,)).fetchone()
        if not row:
            return None
        conn.execute(
            """
            UPDATE ocr_runs
            SET status = 'failed',
                warnings_json = ?,
                error = ?,
                completed_at = ?,
                duration_ms = ?
            WHERE id = ?
            """,
            (
                json.dumps(warnings or [], ensure_ascii=False),
                error,
                completed_at,
                _duration_ms(row["started_at"], completed_at),
                run_id,
            ),
        )
        updated = conn.execute(_OCR_RUN_BY_ID_SQL, (run_id,)).fetchone()
    return _ocr_run_from_row(updated) if updated else None


def get_ocr_run(run_id: str) -> OcrRun | None:
    with connect() as conn:
        row = conn.execute(_OCR_RUN_BY_ID_SQL, (run_id,)).fetchone()
    return _ocr_run_from_row(row) if row else None


def get_active_ocr_run(page_id: str) -> OcrRun | None:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT ocr_runs.*
            FROM pages
            JOIN ocr_runs ON ocr_runs.id = pages.active_ocr_run_id
            WHERE pages.id = ?
            """,
            (page_id,),
        ).fetchone()
    return _ocr_run_from_row(row) if row else None


def find_succeeded_run_by_cache_key(
    page_id: str | None,
    engine: str,
    image_sha256: str | None,
    cache_key: str | None,
) -> OcrRun | None:
    if not cache_key:
        return None
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM ocr_runs
            WHERE (? IS NULL OR page_id = ?)
              AND engine = ?
              AND status = 'succeeded'
              AND (? IS NULL OR image_sha256 = ?)
            ORDER BY completed_at DESC, started_at DESC
            """,
            (page_id, page_id, engine, image_sha256, image_sha256),
        ).fetchall()
    for row in rows:
        run = _ocr_run_from_row(row)
        if run.provider_config.get("cache_key") == cache_key and _can_seed_full_page_ocr_cache(run):
            return run
    return None


def _can_seed_full_page_ocr_cache(run: OcrRun) -> bool:
    if "full_page_cache_write" in run.provider_config:
        return run.provider_config.get("full_page_cache_write") is not False
    variant = str(run.provider_config.get("extraction_variant") or "")
    recovery_variants = {
        "ko_crop_confirm_v1",
        "ko_region_columns_v1",
        "ko_consensus_v1",
        "mcq_source_rebuild_v1",
        "mcq_choice_band_ocr_v1",
        "accuracy_recovery_v1",
        "jp_region_columns_v1",
        "ko_residual_glyph_v1",
        "mcq_prompt_line_ocr_v1",
        "mcq_choice_glyph_v1",
        "accuracy_recovery_v2",
    }
    return variant not in recovery_variants


def get_active_document_parse(page_id: str) -> DocumentParseResult | None:
    run = get_active_ocr_run(page_id)
    return _document_parse_from_run(run) if run else None


def get_document_parse_for_run(run_id: str) -> DocumentParseResult | None:
    run = get_ocr_run(run_id)
    return _document_parse_from_run(run) if run else None


def list_ocr_runs(page_id: str) -> list[OcrRun]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM ocr_runs WHERE page_id = ? ORDER BY started_at DESC, id DESC",
            (page_id,),
        ).fetchall()
    return [_ocr_run_from_row(row) for row in rows]


def activate_ocr_run(page_id: str, run_id: str) -> OcrRun | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM ocr_runs WHERE id = ? AND page_id = ? AND status = 'succeeded'",
            (run_id, page_id),
        ).fetchone()
        if not row:
            return None
        run = _ocr_run_from_row(row)
        page_type = run.metrics.get("page_type")
        page_type_confidence = run.metrics.get("page_type_confidence")
        if isinstance(page_type, str) and isinstance(page_type_confidence, (int, float)):
            conn.execute(
                """
                UPDATE pages
                SET active_ocr_run_id = ?,
                    processed_image_path = COALESCE(?, processed_image_path),
                    image_width = COALESCE(?, image_width),
                    image_height = COALESCE(?, image_height),
                    page_type = ?,
                    page_type_confidence = ?,
                    warnings_json = ?
                WHERE id = ?
                """,
                (
                    run_id,
                    run.processed_image_path,
                    run.image_width,
                    run.image_height,
                    page_type,
                    float(page_type_confidence),
                    json.dumps(run.warnings, ensure_ascii=False),
                    page_id,
                ),
            )
        else:
            conn.execute(
                """
                UPDATE pages
                SET active_ocr_run_id = ?,
                    processed_image_path = COALESCE(?, processed_image_path),
                    image_width = COALESCE(?, image_width),
                    image_height = COALESCE(?, image_height),
                    warnings_json = ?
                WHERE id = ?
                """,
                (
                    run_id,
                    run.processed_image_path,
                    run.image_width,
                    run.image_height,
                    json.dumps(run.warnings, ensure_ascii=False),
                    page_id,
                ),
            )
    return run


def replace_tokens(page_id: str, tokens: list[OcrToken], run_id: str | None = None) -> None:
    target_run_id = run_id or _active_or_legacy_run_id(page_id)
    page_tokens = [token.model_copy(update={"page_id": page_id}) for token in tokens]
    with connect() as conn:
        if target_run_id:
            conn.execute("DELETE FROM ocr_tokens WHERE page_id = ? AND run_id = ?", (page_id, target_run_id))
        else:
            conn.execute("DELETE FROM ocr_tokens WHERE page_id = ? AND run_id IS NULL", (page_id,))
        conn.executemany(
            """
            INSERT INTO ocr_tokens
            (id, page_id, run_id, text, x1, y1, x2, y2, confidence, script_class, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    token.id,
                    token.page_id,
                    target_run_id,
                    token.text,
                    token.bbox[0],
                    token.bbox[1],
                    token.bbox[2],
                    token.bbox[3],
                    token.confidence,
                    token.script_class,
                    token.source,
                )
                for token in page_tokens
            ],
        )


def get_tokens(page_id: str, run_id: str | None = None) -> list[OcrToken]:
    target_run_id = run_id or _active_run_id(page_id)
    with connect() as conn:
        if target_run_id:
            rows = conn.execute(
                "SELECT * FROM ocr_tokens WHERE page_id = ? AND run_id = ? ORDER BY y1, x1, id",
                (page_id, target_run_id),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM ocr_tokens WHERE page_id = ? AND run_id IS NULL ORDER BY y1, x1, id",
                (page_id,),
            ).fetchall()
    return [_token_from_row(row) for row in rows]


def replace_cards(page_id: str, cards: list[CardCandidate], run_id: str | None = None) -> None:
    target_run_id = run_id or _active_or_legacy_run_id(page_id)
    page_cards = [card.model_copy(update={"page_id": page_id, "run_id": target_run_id}) for card in cards]
    with connect() as conn:
        if target_run_id:
            conn.execute("DELETE FROM cards WHERE page_id = ? AND run_id = ?", (page_id, target_run_id))
        else:
            conn.execute("DELETE FROM cards WHERE page_id = ? AND run_id IS NULL", (page_id,))
        _insert_cards(conn, page_cards)


def upsert_card(card: CardCandidate) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM cards WHERE id = ?", (card.id,))
        _insert_cards(conn, [card])


def get_card(card_id: str) -> CardCandidate | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM cards WHERE id = ?", (card_id,)).fetchone()
    return _card_from_row(row) if row else None


def get_cards(page_id: str | None = None, run_id: str | None = None) -> list[CardCandidate]:
    with connect() as conn:
        target_run_id = run_id or (_active_run_id(page_id) if page_id else None)
        if page_id and target_run_id:
            rows = conn.execute(
                "SELECT * FROM cards WHERE page_id = ? AND run_id = ?",
                (page_id, target_run_id),
            ).fetchall()
        elif page_id:
            rows = conn.execute(
                "SELECT * FROM cards WHERE page_id = ? AND run_id IS NULL",
                (page_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT cards.*
                FROM cards
                LEFT JOIN pages ON pages.id = cards.page_id
                WHERE cards.run_id IS NULL OR cards.run_id = pages.active_ocr_run_id
                """
            ).fetchall()
    return sorted((_card_from_row(row) for row in rows), key=_card_sort_key)


def _card_sort_key(card: CardCandidate) -> tuple[Any, ...]:
    bbox = _sort_bbox(card)
    question_no = _numeric_sort_value(card.source.get("question_no"))
    source_order = _numeric_sort_value(card.source.get("order"))
    position = question_no
    if position is None:
        position = source_order
    if position is None:
        position = 1_000_000
    source_rank = 0 if card.source_type == "question_item" else 1
    note_rank = {
        "jp_vocab_entry": 0,
    }.get(card.note_type, 9)
    return (
        card.page_id,
        source_rank,
        position,
        bbox[1],
        bbox[0],
        card.source_id,
        note_rank,
        card.id,
    )


def _sort_bbox(card: CardCandidate) -> tuple[float, float]:
    value = card.source_bbox or card.source.get("bbox")
    if not isinstance(value, list) or len(value) != 4:
        return (1_000_000.0, 1_000_000.0)
    try:
        x1, y1, _x2, _y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return (1_000_000.0, 1_000_000.0)
    return (x1, y1)


def _numeric_sort_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _insert_cards(conn: sqlite3.Connection, cards: list[CardCandidate]) -> None:
    conn.executemany(
        """
        INSERT INTO cards (
            id, page_id, source_type, source_id, source_json, note_type, front, back,
            run_id, tags_json, confidence, status, review_state, source_bbox_json, warnings_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                card.run_id,
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
    keys = set(row.keys())
    display_name = row["display_name"] if "display_name" in keys else None
    return Page(
        id=row["id"],
        original_image_path=row["original_image_path"],
        upload_name=row["upload_name"] if "upload_name" in keys else None,
        display_name=display_name or _default_display_name(row["original_image_path"], row["id"]),
        processed_image_path=row["processed_image_path"],
        active_ocr_run_id=row["active_ocr_run_id"] if "active_ocr_run_id" in keys else None,
        active_ocr_engine=row["active_ocr_engine"] if "active_ocr_engine" in keys else None,
        active_ocr_completed_at=row["active_ocr_completed_at"] if "active_ocr_completed_at" in keys else None,
        active_ocr_duration_ms=row["active_ocr_duration_ms"] if "active_ocr_duration_ms" in keys else None,
        page_type=row["page_type"],
        page_type_confidence=row["page_type_confidence"],
        image_width=row["image_width"],
        image_height=row["image_height"],
        warnings=json.loads(row["warnings_json"]),
        created_at=row["created_at"],
        card_count=int(row["card_count"]) if "card_count" in keys else 0,
        approved_card_count=int(row["approved_card_count"]) if "approved_card_count" in keys else 0,
        red_card_count=int(row["red_card_count"]) if "red_card_count" in keys else 0,
    )


def _ensure_page_columns(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(pages)").fetchall()}
    if "upload_name" not in columns:
        conn.execute("ALTER TABLE pages ADD COLUMN upload_name TEXT")
    if "display_name" not in columns:
        conn.execute("ALTER TABLE pages ADD COLUMN display_name TEXT")
    if "active_ocr_run_id" not in columns:
        conn.execute("ALTER TABLE pages ADD COLUMN active_ocr_run_id TEXT")


def _ensure_run_columns(conn: sqlite3.Connection) -> None:
    token_columns = {row["name"] for row in conn.execute("PRAGMA table_info(ocr_tokens)").fetchall()}
    if "run_id" not in token_columns:
        conn.execute("ALTER TABLE ocr_tokens ADD COLUMN run_id TEXT")
    card_columns = {row["name"] for row in conn.execute("PRAGMA table_info(cards)").fetchall()}
    if "run_id" not in card_columns:
        conn.execute("ALTER TABLE cards ADD COLUMN run_id TEXT")


def _backfill_legacy_runs(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT
            pages.id,
            pages.processed_image_path,
            pages.image_width,
            pages.image_height,
            pages.page_type,
            pages.page_type_confidence,
            pages.warnings_json,
            pages.created_at
        FROM pages
        WHERE pages.active_ocr_run_id IS NULL
          AND (
            EXISTS (SELECT 1 FROM ocr_tokens WHERE ocr_tokens.page_id = pages.id)
            OR EXISTS (SELECT 1 FROM cards WHERE cards.page_id = pages.id)
          )
        """
    ).fetchall()
    for row in rows:
        run_id = f"run_legacy_{row['id']}"
        conn.execute(
            """
            INSERT OR IGNORE INTO ocr_runs (
                id, page_id, engine, status, image_sha256, processed_image_path, image_width, image_height,
                preprocessing_json, model_config_json, metrics_json, warnings_json, error,
                started_at, completed_at, duration_ms
            )
            VALUES (?, ?, 'legacy', 'succeeded', NULL, ?, ?, ?, '{}', '{}', ?, ?, NULL, ?, ?, NULL)
            """,
            (
                run_id,
                row["id"],
                row["processed_image_path"],
                row["image_width"],
                row["image_height"],
                json.dumps(
                    {
                        "page_type": row["page_type"],
                        "page_type_confidence": row["page_type_confidence"],
                    },
                    ensure_ascii=False,
                ),
                row["warnings_json"] or "[]",
                row["created_at"],
                row["created_at"],
            ),
        )
        conn.execute("UPDATE ocr_tokens SET run_id = ? WHERE page_id = ? AND run_id IS NULL", (run_id, row["id"]))
        conn.execute("UPDATE cards SET run_id = ? WHERE page_id = ? AND run_id IS NULL", (run_id, row["id"]))
        conn.execute("UPDATE pages SET active_ocr_run_id = ? WHERE id = ?", (run_id, row["id"]))


def _delete_unsupported_vocab_cards(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM cards WHERE source_type = 'vocab_item' AND note_type != 'jp_vocab_entry'")


def _ensure_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_pages_upload_name_created_at
            ON pages(upload_name, created_at DESC)
            WHERE upload_name IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_ocr_runs_page_status
            ON ocr_runs(page_id, status, started_at DESC);
        CREATE INDEX IF NOT EXISTS idx_ocr_tokens_page_id
            ON ocr_tokens(page_id);
        CREATE INDEX IF NOT EXISTS idx_ocr_tokens_run_id
            ON ocr_tokens(run_id);
        CREATE INDEX IF NOT EXISTS idx_cards_page_id
            ON cards(page_id);
        CREATE INDEX IF NOT EXISTS idx_cards_run_id
            ON cards(run_id);
        CREATE INDEX IF NOT EXISTS idx_cards_page_status_review
            ON cards(page_id, status, review_state);
        CREATE INDEX IF NOT EXISTS idx_cards_source
            ON cards(source_type, source_id);
        """
    )


def _default_display_name(original_image_path: str, page_id: str) -> str:
    stem = Path(original_image_path).stem.strip()
    return stem or page_id


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
        run_id=row["run_id"] if "run_id" in set(row.keys()) else None,
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


def _insert_ocr_run(conn: sqlite3.Connection, run: OcrRun) -> None:
    conn.execute(
        """
        INSERT INTO ocr_runs (
            id, page_id, engine, status, image_sha256, processed_image_path, image_width, image_height,
            preprocessing_json, model_config_json, metrics_json, warnings_json, error,
            started_at, completed_at, duration_ms
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run.id,
            run.page_id,
            run.engine,
            run.status,
            run.image_sha256,
            run.processed_image_path,
            run.image_width,
            run.image_height,
            json.dumps(run.preprocessing, ensure_ascii=False),
            json.dumps(run.provider_config, ensure_ascii=False),
            json.dumps(run.metrics, ensure_ascii=False),
            json.dumps(run.warnings, ensure_ascii=False),
            run.error,
            run.started_at,
            run.completed_at,
            run.duration_ms,
        ),
    )


def _ocr_run_from_row(row: sqlite3.Row) -> OcrRun:
    return OcrRun(
        id=row["id"],
        page_id=row["page_id"],
        engine=row["engine"],
        status=row["status"],
        image_sha256=row["image_sha256"],
        processed_image_path=row["processed_image_path"],
        image_width=row["image_width"],
        image_height=row["image_height"],
        preprocessing=json.loads(row["preprocessing_json"]),
        provider_config=json.loads(row["model_config_json"]),
        metrics=json.loads(row["metrics_json"]),
        warnings=json.loads(row["warnings_json"]),
        error=row["error"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        duration_ms=row["duration_ms"],
    )


def _document_parse_from_run(run: OcrRun | None) -> DocumentParseResult | None:
    payload = run.metrics.get("document_parse") if run else None
    if not isinstance(payload, dict):
        return None
    try:
        return DocumentParseResult(**payload)
    except ValueError:
        return None


def _active_run_id(page_id: str | None) -> str | None:
    if not page_id:
        return None
    with connect() as conn:
        row = conn.execute("SELECT active_ocr_run_id FROM pages WHERE id = ?", (page_id,)).fetchone()
    return row["active_ocr_run_id"] if row else None


def _active_or_legacy_run_id(page_id: str) -> str | None:
    return _active_run_id(page_id)


def _duration_ms(started_at: str, completed_at: str) -> int | None:
    try:
        started = datetime.fromisoformat(started_at)
        completed = datetime.fromisoformat(completed_at)
    except ValueError:
        return None
    return max(0, int((completed - started).total_seconds() * 1000))
