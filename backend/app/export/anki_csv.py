from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

from app.models.schemas import CardCandidate


HEADER = ["notetype", "front", "back", "source_page", "source_bbox", "confidence", "tags"]
ANKI_FILE_HEADERS = [
    "#separator:Comma",
    "#html:true",
    f"#columns:{','.join(HEADER)}",
    "#notetype column:1",
    "#tags column:7",
]


def clean_csv_field(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def cards_to_csv(cards: list[CardCandidate]) -> str:
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
                clean_csv_field(card.source_bbox or ""),
                f"{card.confidence:.3f}",
                clean_csv_field(" ".join(card.tags)),
            ]
        )
    return output.getvalue()


def write_csv(path: Path, cards: list[CardCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cards_to_csv(cards), encoding="utf-8")
