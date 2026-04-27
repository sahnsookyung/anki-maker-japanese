from __future__ import annotations

import re

from app.models.schemas import OcrToken


_CIRCLED = {
    "①": 1,
    "②": 2,
    "③": 3,
    "④": 4,
    "⑤": 5,
    "⑥": 6,
    "⑦": 7,
    "⑧": 8,
    "⑨": 9,
    "⑩": 10,
}


def parse_answer_strip_text(text: str) -> dict[int, int]:
    normalized = text.translate(str.maketrans({k: str(v) for k, v in _CIRCLED.items()}))
    pairs: dict[int, int] = {}
    for question, answer in re.findall(r"\b(10|[1-9])\s*[:.\-]?\s*([1-4])\b", normalized):
        pairs[int(question)] = int(answer)
    return pairs


def parse_answer_strip(tokens: list[OcrToken], image_height: int | None) -> dict[int, int]:
    if not image_height:
        return parse_answer_strip_text(" ".join(token.text for token in tokens))
    bottom = [token for token in tokens if token.bbox[1] >= image_height * 0.82]
    text = " ".join(token.text for token in sorted(bottom, key=lambda t: (t.bbox[1], t.bbox[0])))
    return parse_answer_strip_text(text)
