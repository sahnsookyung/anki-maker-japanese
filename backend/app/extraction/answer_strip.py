from __future__ import annotations

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
    "⑪": 11,
    "⑫": 12,
    "⑬": 13,
    "⑭": 14,
    "⑮": 15,
    "⑯": 16,
    "⑰": 17,
    "⑱": 18,
    "⑲": 19,
    "⑳": 20,
}


def parse_answer_strip_text(text: str) -> dict[int, int]:
    normalized = text.translate(str.maketrans({k: str(v) for k, v in _CIRCLED.items()}))
    normalized = normalized.translate(str.maketrans({":": " ", ".": " ", "-": " "}))
    numbers = [int(part) for part in normalized.split() if part.isdigit()]
    pairs: dict[int, int] = {}
    for question, answer in zip(numbers[::2], numbers[1::2]):
        if question >= 1 and 1 <= answer <= 4:
            pairs[question] = answer
    return pairs


def parse_answer_strip(tokens: list[OcrToken], image_height: int | None) -> dict[int, int]:
    if not image_height:
        return parse_answer_strip_text(" ".join(token.text for token in tokens))
    bottom = [token for token in tokens if token.bbox[1] >= image_height * 0.82]
    text = " ".join(token.text for token in sorted(bottom, key=lambda t: (t.bbox[1], t.bbox[0])))
    return parse_answer_strip_text(text)
