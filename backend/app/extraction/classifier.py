from __future__ import annotations

from collections import Counter

from app.models.schemas import OcrToken


DIGIT_MARKERS = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"}
CHOICE_MARKERS = {"1", "2", "3", "4", "①", "②", "③", "④"}


def classify_page(tokens: list[OcrToken], image_height: int | None = None) -> tuple[str, float, dict[str, int]]:
    scripts = Counter(token.script_class for token in tokens)
    texts = [token.text for token in tokens]
    checkbox_count = sum(1 for text in texts if text in {"□", "☐", "▢", "口", "ロ"} or any(mark in text for mark in ("□", "☐", "▢", "口", "ロ")))
    question_numbers = sum(1 for text in texts if _normalize_digits(text.strip()) in DIGIT_MARKERS)
    choice_markers = sum(1 for text in texts if _normalize_digits(text.strip()) in CHOICE_MARKERS)
    prefixed_choices = [_choice_text(text) for text in texts if _choice_text(text)]
    prefixed_questions = sum(1 for text in texts if _has_question_prefix(text))
    bottom_tokens = 0
    if image_height:
        bottom_tokens = sum(1 for token in tokens if token.bbox[1] > image_height * 0.82)

    japanese_count = scripts["hiragana"] + scripts["katakana"] + scripts["kanji"] + scripts["mixed"]
    hangul_count = sum(1 for text in texts if _has_hangul(text))
    has_japanese = japanese_count > 3
    has_hangul = hangul_count > 2
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

    if has_japanese and has_hangul and _looks_like_meaning_only_vocab(scripts, hangul_count, checkbox_count, len(tokens)):
        return "jp_ko_meaning_vocab", 0.66, features

    if has_japanese and ((has_hangul and (checkbox_count >= 4 or len(tokens) >= 20)) or checkbox_count >= 8):
        return "vocab_table", 0.68, features

    if has_japanese and len(tokens) >= 40:
        return "vocab_table", 0.62, features

    if len(tokens) < 8:
        return "unknown_review_required", 0.2, features
    return "unknown_review_required", 0.4, features


def _choice_text(text: str) -> str:
    normalized = _normalize_digits(text.strip())
    marker_index = _first_marker_index(normalized, CHOICE_MARKERS)
    if marker_index is None or marker_index > 2:
        return ""
    after_marker = normalized[marker_index + 1 :].lstrip(" \t-:：.．・")
    return after_marker.strip()


def _has_question_prefix(text: str) -> bool:
    normalized = _normalize_digits(text.strip())
    if normalized.startswith("10"):
        return len(normalized) > 2 and not normalized[2].isdigit()
    return bool(normalized and normalized[0] in "123456789" and len(normalized) > 1 and not normalized[1].isdigit())


def _first_marker_index(text: str, markers: set[str]) -> int | None:
    for index, char in enumerate(text[:3]):
        if char in markers:
            return index
    return None


def _normalize_digits(text: str) -> str:
    return text.translate(str.maketrans("１２３４５６７８９０", "1234567890"))


def _has_hangul(text: str) -> bool:
    return any(0xAC00 <= ord(char) <= 0xD7AF for char in text)


def _looks_like_meaning_only_vocab(
    scripts: Counter[str],
    hangul_count: int,
    checkbox_count: int,
    token_count: int,
) -> bool:
    if checkbox_count < 4 and token_count < 12:
        return False
    hiragana_count = scripts["hiragana"]
    katakana_count = scripts["katakana"]
    kanji_like_count = scripts["kanji"] + scripts["mixed"]
    if hiragana_count <= max(2, kanji_like_count // 2):
        return True
    return katakana_count >= hiragana_count and hiragana_count <= max(3, hangul_count // 2)


def _kana_count(text: str) -> int:
    return sum(1 for char in text if 0x3040 <= ord(char) <= 0x30FF)


def _kanji_count(text: str) -> int:
    return sum(1 for char in text if 0x4E00 <= ord(char) <= 0x9FFF)
