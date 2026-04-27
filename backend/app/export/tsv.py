from __future__ import annotations

from pathlib import Path

from app.models.schemas import CardCandidate


HEADER = ["note_type", "front", "back", "source_page", "source_bbox", "confidence", "tags"]


def clean_tsv_field(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("\t", " ").replace("\r", " ").replace("\n", "<br>")


def cards_to_tsv(cards: list[CardCandidate]) -> str:
    lines = ["\t".join(HEADER)]
    for card in cards:
        row = [
            card.note_type,
            card.front,
            card.back,
            card.page_id,
            card.source_bbox or "",
            f"{card.confidence:.3f}",
            " ".join(card.tags),
        ]
        lines.append("\t".join(clean_tsv_field(value) for value in row))
    return "\n".join(lines) + "\n"


def write_tsv(path: Path, cards: list[CardCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cards_to_tsv(cards), encoding="utf-8")
