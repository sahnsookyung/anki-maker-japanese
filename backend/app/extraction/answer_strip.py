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
    spaced_pairs = _spaced_answer_pairs(numbers)
    compact_pairs = _compact_answer_pairs(normalized, spaced_pairs)
    if compact_pairs:
        return compact_pairs
    return spaced_pairs


def _compact_answer_pairs(text: str, spaced_pairs: dict[int, int] | None = None) -> dict[int, int]:
    digit_parts = [part for part in text.split() if part.isdigit()]
    if not digit_parts:
        return {}
    parsed_pairs = []
    for part in digit_parts:
        parsed = _parse_compact_pair(part)
        if not parsed:
            continue
        question, answer = parsed
        if question >= 1 and 1 <= answer <= 4:
            parsed_pairs.append((question, answer))
    if not parsed_pairs:
        return {}
    if len(parsed_pairs) == 1:
        if spaced_pairs:
            return {}
        question, answer = parsed_pairs[0]
        return {question: answer}
    return _longest_consecutive_run(parsed_pairs)


def _longest_consecutive_run(parsed_pairs: list[tuple[int, int]]) -> dict[int, int]:
    best: dict[int, int] = {}
    for start_index, (start_question, start_answer) in enumerate(parsed_pairs):
        current = {start_question: start_answer}
        expected_question = start_question + 1
        for question, answer in parsed_pairs[start_index + 1 :]:
            if question < expected_question:
                continue
            if question > expected_question:
                break
            current[question] = answer
            expected_question += 1
        if len(current) > len(best):
            best = current
    return best if len(best) >= 2 else {}


def _spaced_answer_pairs(numbers: list[int]) -> dict[int, int]:
    candidates: list[list[tuple[int, int]]] = []
    for offset in (0, 1):
        pairs = [
            (question, answer)
            for question, answer in zip(numbers[offset::2], numbers[offset + 1 :: 2])
            if 1 <= question <= 20 and 1 <= answer <= 4
        ]
        if pairs:
            candidates.append(pairs)
    if not candidates:
        return {}
    best_pairs = max(candidates, key=lambda pairs: (_consecutive_prefix_length(pairs), len(pairs)))
    return dict(best_pairs)


def _consecutive_prefix_length(pairs: list[tuple[int, int]]) -> int:
    if not pairs:
        return 0
    expected = pairs[0][0]
    count = 0
    for question, _answer in pairs:
        if question != expected:
            break
        count += 1
        expected += 1
    return count


def _parse_compact_pair(part: str) -> tuple[int, int] | None:
    if len(part) == 2:
        return int(part[0]), int(part[1])
    if len(part) == 3:
        question = int(part[:-1])
        answer = int(part[-1])
        if 1 <= question <= 20:
            return question, answer
    return None


def parse_answer_strip(tokens: list[OcrToken], image_height: int | None) -> dict[int, int]:
    if not image_height:
        return parse_answer_strip_text(" ".join(token.text for token in tokens))
    bottom = [token for token in tokens if token.bbox[1] >= image_height * 0.82]
    text = " ".join(token.text for token in sorted(bottom, key=lambda t: (t.bbox[1], t.bbox[0])))
    return parse_answer_strip_text(text)
