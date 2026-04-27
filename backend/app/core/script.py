from __future__ import annotations

from collections import Counter
from unicodedata import category


def char_script(ch: str) -> str:
    code = ord(ch)
    if 0x3040 <= code <= 0x309F:
        return "hiragana"
    if 0x30A0 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF:
        return "katakana"
    if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
        return "kanji"
    if 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F:
        return "hangul"
    if ch.isdigit() or ("０" <= ch <= "９"):
        return "number"
    if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
        return "latin"
    if category(ch).startswith("P") or ch.isspace() or category(ch).startswith("S"):
        return "punctuation"
    return "other"


def classify_script(text: str) -> str:
    scripts = [char_script(ch) for ch in text if not ch.isspace()]
    meaningful = [s for s in scripts if s != "punctuation"]
    if not meaningful:
        return "punctuation"
    counts = Counter(meaningful)
    if len(counts) == 1:
        return next(iter(counts))
    if counts["number"] and sum(counts.values()) == counts["number"]:
        return "number"
    return "mixed"


def script_summary(texts: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts[classify_script(text)] += 1
    return dict(counts)
