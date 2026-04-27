from __future__ import annotations

from collections import Counter
import re

from app.models.schemas import OcrToken


CHOICE_PREFIX_RE = re.compile(r"^[^\d１２３４①②③④]{0,2}([1-4１２３４①②③④])\s*[\-:：.．・]?\s*(.+)$")
QUESTION_PREFIX_RE = re.compile(r"^[^\d]{0,2}(10|[1-9]|１０|[１-９])\D+")


def classify_page(tokens: list[OcrToken], image_height: int | None = None) -> tuple[str, float, dict[str, int]]:
    scripts = Counter(token.script_class for token in tokens)
    texts = [token.text for token in tokens]
    checkbox_count = sum(1 for text in texts if text in {"□", "☐", "▢", "口", "ロ"} or any(mark in text for mark in ("□", "☐", "▢", "口", "ロ")))
    question_numbers = sum(1 for text in texts if re.fullmatch(r"(?:[1-9]|10|①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩)", text))
    choice_markers = sum(1 for text in texts if re.fullmatch(r"[1-4①②③④]", text))
    prefixed_choices = [_choice_text(text) for text in texts if _choice_text(text)]
    prefixed_questions = sum(1 for text in texts if QUESTION_PREFIX_RE.match(_normalize_digits(text)))
    bottom_tokens = 0
    if image_height:
        bottom_tokens = sum(1 for token in tokens if token.bbox[1] > image_height * 0.82)

    has_japanese = scripts["hiragana"] + scripts["katakana"] + scripts["kanji"] + scripts["mixed"] > 3
    has_hangul = sum(1 for text in texts if _has_hangul(text)) > 2
    features = {
        "checkbox_count": checkbox_count,
        "question_numbers": question_numbers,
        "choice_markers": choice_markers,
        "prefixed_choices": len(prefixed_choices),
        "prefixed_questions": prefixed_questions,
        "bottom_tokens": bottom_tokens,
        **{f"script_{key}": value for key, value in scripts.items()},
    }

    if question_numbers >= 8 or choice_markers >= 16 or len(prefixed_choices) >= 12 or prefixed_questions >= 6:
        kana_choices = sum(1 for text in prefixed_choices if _kana_count(text) >= _kanji_count(text))
        kanji_choices = sum(1 for text in prefixed_choices if _kanji_count(text) > _kana_count(text))
        if kana_choices >= kanji_choices:
            return "reading_mcq", 0.72, features
        return "spelling_mcq", 0.72, features

    if has_japanese and ((has_hangul and (checkbox_count >= 4 or len(tokens) >= 20)) or checkbox_count >= 8):
        return "vocab_table", 0.68, features

    if has_japanese and len(tokens) >= 40:
        return "vocab_table", 0.62, features

    if len(tokens) < 8:
        return "unknown_review_required", 0.2, features
    return "unknown_review_required", 0.4, features


def _choice_text(text: str) -> str:
    match = CHOICE_PREFIX_RE.match(_normalize_digits(text.strip()))
    if not match:
        return ""
    return match.group(2).strip()


def _normalize_digits(text: str) -> str:
    return text.translate(str.maketrans("１２３４５６７８９０", "1234567890"))


def _has_hangul(text: str) -> bool:
    return any(0xAC00 <= ord(char) <= 0xD7AF for char in text)


def _kana_count(text: str) -> int:
    return sum(1 for char in text if 0x3040 <= ord(char) <= 0x30FF)


def _kanji_count(text: str) -> int:
    return sum(1 for char in text if 0x4E00 <= ord(char) <= 0x9FFF)
