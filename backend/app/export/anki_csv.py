from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from io import StringIO
from pathlib import Path
from typing import Any

from app.models.schemas import CardCandidate


MCQ_HEADER = ["notetype", "front", "back", "source_page", "source_bbox", "confidence", "tags"]
VOCAB_HEADER = [
    "VocabKey",
    "Surface",
    "Reading",
    "MeaningKo",
    "StudyWriting",
    "StudyReading",
    "StudyMeaning",
    "SourcePage",
    "SourceBBox",
    "Confidence",
    "Warnings",
    "tags",
]
ANKI_FILE_HEADERS = [
    "#separator:Comma",
    "#html:true",
    f"#columns:{','.join(MCQ_HEADER)}",
    "#notetype column:1",
    "#tags column:7",
]
VOCAB_FILE_HEADERS = [
    "#separator:Comma",
    "#html:true",
    "#notetype:jp_vocab_entry",
    f"#columns:{','.join(VOCAB_HEADER)}",
    "#tags column:12",
]
VOCAB_NOTE_TYPE = "jp_vocab_entry"
VOCAB_STUDY_FIELDS = ("study_writing", "study_reading", "study_meaning")


def clean_csv_field(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def cards_to_csv(cards: list[CardCandidate]) -> str:
    if not cards:
        raise ValueError("cards_to_csv requires at least one card")
    if all(is_vocab_note(card) for card in cards):
        return vocab_notes_to_csv(cards)
    if all(is_mcq_card(card) for card in cards):
        return mcq_cards_to_csv(cards)
    raise ValueError("cards_to_csv supports one export schema at a time")


def mcq_cards_to_csv(cards: list[CardCandidate]) -> str:
    output = StringIO()
    output.write("\n".join(ANKI_FILE_HEADERS))
    output.write("\n")
    writer = csv.writer(output, lineterminator="\n")
    for card in cards:
        writer.writerow(
            [
                clean_csv_field(card.note_type),
                clean_csv_field(card.front),
                clean_csv_field(card.back),
                clean_csv_field(card.page_id),
                compact_bbox(card.source_bbox),
                f"{card.confidence:.3f}",
                clean_csv_field(" ".join(card.tags)),
            ]
        )
    return output.getvalue()


def vocab_notes_to_csv(cards: list[CardCandidate]) -> str:
    output = StringIO()
    output.write("\n".join(VOCAB_FILE_HEADERS))
    output.write("\n")
    writer = csv.writer(output, lineterminator="\n")
    for card in cards:
        row = vocab_note_row(card)
        if row:
            writer.writerow(row)
    return output.getvalue()


def write_csv(path: Path, cards: list[CardCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cards_to_csv(cards), encoding="utf-8")


def write_export_csvs(export_dir: Path, export_id: str, cards: list[CardCandidate]) -> tuple[list[dict[str, Any]], int, int]:
    export_dir.mkdir(parents=True, exist_ok=True)
    vocab_cards, mcq_cards = split_cards_for_export(cards)
    files: list[dict[str, Any]] = []
    note_count = 0
    generated_card_count = 0

    if vocab_cards:
        filename = f"{export_id}_vocab_notes.csv"
        path = export_dir / filename
        path.write_text(vocab_notes_to_csv(vocab_cards), encoding="utf-8")
        row_count = len(vocab_cards)
        files.append(_file_payload("vocab", filename, path, row_count))
        note_count += row_count
        generated_card_count += sum(vocab_generated_card_count(card) for card in vocab_cards)

    if mcq_cards:
        filename = f"{export_id}_mcq_cards.csv"
        path = export_dir / filename
        path.write_text(mcq_cards_to_csv(mcq_cards), encoding="utf-8")
        files.append(_file_payload("mcq", filename, path, len(mcq_cards)))
        note_count += len(mcq_cards)
        generated_card_count += len(mcq_cards)

    return files, note_count, generated_card_count


def split_cards_for_export(cards: list[CardCandidate]) -> tuple[list[CardCandidate], list[CardCandidate]]:
    vocab_cards = [card for card in cards if vocab_note_row(card)]
    mcq_cards = [card for card in cards if is_mcq_card(card)]
    return vocab_cards, mcq_cards


def is_vocab_note(card: CardCandidate) -> bool:
    return card.source_type == "vocab_item" and card.note_type == VOCAB_NOTE_TYPE


def is_mcq_card(card: CardCandidate) -> bool:
    return card.source_type == "question_item"


def vocab_note_row(card: CardCandidate) -> list[str] | None:
    if not is_vocab_note(card):
        return None
    source = card.source
    surface = _source_text(source, "surface")
    reading = _source_text(source, "reading")
    meaning = _source_text(source, "meaning_ko")
    has_vocab_card = study_direction_count(source)
    if not (surface and reading and meaning and has_vocab_card):
        return None
    return [
        vocab_key(surface, reading, meaning),
        clean_csv_field(surface),
        clean_csv_field(reading),
        clean_csv_field(meaning),
        "1",
        "",
        "",
        clean_csv_field(card.page_id),
        compact_bbox(card.source_bbox or source.get("bbox")),
        f"{card.confidence:.3f}",
        clean_csv_field(" | ".join(dict.fromkeys(card.warnings))),
        clean_csv_field(" ".join(card.tags)),
    ]


def vocab_key(surface: str, reading: str, meaning_ko: str) -> str:
    normalized = "|".join(_normalize_key_part(part) for part in (surface, reading, meaning_ko))
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"vocab_{digest}"


def vocab_generated_card_count(card: CardCandidate) -> int:
    return study_direction_count(card.source)


def study_direction_count(source: dict[str, Any]) -> int:
    return 1 if any(_study_field(source, field) for field in VOCAB_STUDY_FIELDS) else 0


def compact_bbox(value: object) -> str:
    if not isinstance(value, list) or len(value) != 4:
        return ""
    try:
        bbox = [_compact_number(float(item)) for item in value]
    except (TypeError, ValueError):
        return ""
    return json.dumps(bbox, separators=(",", ":"))


def _file_payload(kind: str, filename: str, path: Path, row_count: int) -> dict[str, Any]:
    return {
        "kind": kind,
        "filename": filename,
        "path": str(path),
        "download_url": f"/api/exports/{filename}",
        "row_count": row_count,
    }


def _source_text(source: dict[str, Any], field: str) -> str:
    value = source.get(field)
    return "" if value is None else str(value).strip()


def _study_field(source: dict[str, Any], field: str) -> str:
    value = source.get(field, True)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else ""
    if isinstance(value, (int, float)) and value == 0:
        return ""
    if str(value).strip().lower() in {"", "0", "false", "no"}:
        return ""
    return "1"


def _normalize_key_part(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _compact_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value
