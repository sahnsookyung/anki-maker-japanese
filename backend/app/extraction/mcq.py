from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re

from app.core.config import KOREAN_GLOSSARY_PATH
from app.core.ids import new_id
from app.extraction.geometry import group_tokens_by_line, text_of, union_bbox
from app.models.schemas import OcrToken


QUESTION_NO_RE = re.compile(r"^(10|[1-9]|①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩)$")
CHOICE_NO_RE = re.compile(r"^[1-4①-④]$")
CHOICE_CHUNK_RE = re.compile(r"([1-4①-④])([^1-4①-④]+)")
CIRCLED = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5, "⑥": 6, "⑦": 7, "⑧": 8, "⑨": 9, "⑩": 10}
FULLWIDTH_DIGITS = str.maketrans("１２３４５６７８９０", "1234567890")


def extract_mcq_items(tokens: list[OcrToken], answer_map: dict[int, int], page_type: str) -> list[dict]:
    lines = group_tokens_by_line(tokens, tolerance=22)
    blocks = _question_blocks(lines)
    if not blocks:
        blocks = []
        current: list[OcrToken] = []
        for line in lines:
            starts_question = line and QUESTION_NO_RE.match(_normalize_digits(line[0].text))
            if starts_question and current:
                blocks.append(current)
                current = []
            current.extend(line)
        if current:
            blocks.append(current)

    items: list[dict] = []
    filtered_blocks: list[list[OcrToken]] = []
    for block in blocks:
        sentence_like = [token for token in block if not _is_choice_token(token.text)]
        sentence_text = text_of(sentence_like, "")
        if _has_hiragana(sentence_text) and not _is_header_text(sentence_text):
            filtered_blocks.append(block)

    for sequence_no, block in enumerate(filtered_blocks, start=1):
        choices = _extract_choices(block)
        question_no = _question_no(block) or sequence_no
        sentence_tokens = [
            token
            for token in block
            if not _is_choice_token(token.text) and not QUESTION_NO_RE.match(_normalize_digits(token.text))
        ]
        sentence = _clean_sentence(text_of(sentence_tokens, " "))
        target = _guess_target(sentence_tokens, page_type)
        resolved_target, resolved_answer, resolved_choice_no = _resolve_from_glossary(sentence, choices, page_type)
        if resolved_target:
            target = resolved_target
        correct_choice_no = answer_map.get(question_no) or resolved_choice_no
        correct_answer = ""
        if correct_choice_no and 1 <= correct_choice_no <= len(choices):
            correct_answer = choices[correct_choice_no - 1]
        if resolved_answer:
            correct_answer = resolved_answer
        warnings: list[str] = []
        if not target:
            warnings.append("Could not confidently identify the underlined target.")
        if not correct_choice_no:
            warnings.append("Correct choice is missing from the answer strip or local glossary.")
        if len(choices) != 4:
            warnings.append("Could not extract exactly four choices.")
        bbox = union_bbox([token.bbox for token in block])
        token_roles = _token_roles(block, target, correct_answer)
        target_tokens = [token for token in block if _token_role(token, target, correct_answer) == "target"]
        target_bbox = union_bbox([token.bbox for token in target_tokens]) if target_tokens else None
        confidence = min((token.confidence for token in block), default=0.5)
        items.append(
            {
                "id": new_id("q"),
                "type": "question_item",
                "question_type": "spelling_mcq" if page_type == "spelling_mcq" else "reading_mcq",
                "question_no": question_no,
                "sentence": sentence,
                "target": target,
                "target_bbox": target_bbox,
                "choices": choices,
                "correct_choice_no": correct_choice_no,
                "correct_answer": correct_answer,
                "answer_source": _answer_source(question_no, answer_map, resolved_choice_no),
                "bbox": bbox,
                "evidence_tokens": [token.id for token in block],
                "token_roles": token_roles,
                "confidence": round(confidence, 3),
                "needs_review": bool(warnings),
                "warnings": warnings,
            }
        )
    return items


def _question_blocks(lines: list[list[OcrToken]]) -> list[list[OcrToken]]:
    blocks: list[list[OcrToken]] = []
    pending: list[OcrToken] = []
    for line in lines:
        if _is_header_text(text_of(line, "")):
            pending = []
            continue
        choices = _extract_choices(line)
        if len(choices) >= 2 and pending:
            blocks.append([*pending, *line])
            pending = []
            continue
        if len(choices) >= 2 and _looks_like_question_with_inline_choices(line):
            blocks.append(line)
            pending = []
            continue
        if _looks_like_question_line(line):
            if pending and _extract_choices(pending):
                pending = []
            pending.extend(line)
        elif pending and not choices:
            pending.extend(line)
    return blocks


def _question_no(tokens: list[OcrToken]) -> int | None:
    for token in tokens[:5]:
        text = _normalize_digits(token.text)
        if text in CIRCLED:
            return CIRCLED[text]
        if text.isdigit() and 1 <= int(text) <= 10:
            return int(text)
        match = re.match(r"^(10|[1-9])\D+", text)
        if match:
            return int(match.group(1))
    return None


def _token_roles(tokens: list[OcrToken], target: str, correct_answer: str) -> list[dict[str, str]]:
    return [{"token_id": token.id, "role": _token_role(token, target, correct_answer)} for token in tokens]


def _token_role(token: OcrToken, target: str, correct_answer: str) -> str:
    normalized = _normalize_for_match(token.text)
    if target and _normalize_for_match(target) in normalized:
        return "target"
    if correct_answer and _normalize_for_match(correct_answer) in normalized:
        return "answer"
    if _is_choice_token(token.text):
        return "choice"
    return "sentence"


def _extract_choices(tokens: list[OcrToken]) -> list[str]:
    ordered = sorted(tokens, key=lambda t: (t.bbox[1], t.bbox[0]))
    found: dict[int, tuple[int, float, str]] = {}
    idx = 0
    while idx < len(ordered):
        token = ordered[idx]
        text = _normalize_digits(token.text)
        marker_num = _choice_marker_no(text)
        chunks = _choice_chunks(text)
        if chunks:
            for number, choice in chunks:
                _set_choice(found, number, _clean_choice(choice), priority=0, y=token.bbox[1])
        elif marker_num and _choice_text_after_marker(text):
            _set_choice(found, marker_num, _clean_choice(_choice_text_after_marker(text)), priority=0, y=token.bbox[1])
        elif marker_num and idx + 1 < len(ordered) and not _is_choice_token(ordered[idx + 1].text):
            _set_choice(found, marker_num, _clean_choice(ordered[idx + 1].text), priority=1, y=token.bbox[1])
            idx += 1
        idx += 1
    return [found[number][2] for number in range(1, 5) if found.get(number)]


def _answer_source(question_no: int, answer_map: dict[int, int], resolved_choice_no: int | None) -> str:
    if question_no in answer_map:
        return "answer_strip"
    if resolved_choice_no:
        return "local_glossary"
    return "unknown"


def _guess_target(tokens: list[OcrToken], page_type: str) -> str:
    if page_type == "spelling_mcq":
        preferred = {"hiragana", "katakana"}
    else:
        preferred = {"kanji", "mixed"}
    for token in tokens:
        if token.script_class in preferred and len(token.text) >= 1:
            return _strip_question_prefix(token.text)
    return tokens[0].text if tokens else ""


def _choice_marker_no(text: str) -> int | None:
    stripped = text.strip().translate(FULLWIDTH_DIGITS).lstrip("「『【[(（ \t")
    if _looks_like_prefixed_question_text(stripped):
        return None
    if stripped.startswith("10"):
        return None
    if stripped in CIRCLED:
        return CIRCLED[stripped]
    if CHOICE_NO_RE.match(stripped):
        return int(stripped)
    if not stripped:
        return None
    marker = stripped[0]
    if marker not in CIRCLED and marker not in {"1", "2", "3", "4"}:
        return None
    return CIRCLED[marker] if marker in CIRCLED else int(marker)


def _choice_text_after_marker(text: str) -> str:
    stripped = text.strip().translate(FULLWIDTH_DIGITS).lstrip("「『【[(（ \t")
    if _looks_like_prefixed_question_text(stripped):
        return ""
    if not stripped or stripped[0] not in {"1", "2", "3", "4", "①", "②", "③", "④"}:
        return ""
    return stripped[1:].lstrip(" \t.．:：・-").strip()


def _choice_chunks(text: str) -> list[tuple[int, str]]:
    normalized = text.translate(FULLWIDTH_DIGITS)
    if _looks_like_prefixed_question_text(normalized):
        return []
    if normalized.strip().startswith("10"):
        return []
    placeholder = _choices_from_question_mark_placeholder(normalized)
    if placeholder:
        return placeholder
    chunks: list[tuple[int, str]] = []
    for match in CHOICE_CHUNK_RE.finditer(normalized):
        marker = match.group(1)
        number = CIRCLED[marker] if marker in CIRCLED else int(marker)
        choice = match.group(2)
        if choice and not choice[0].isdigit():
            chunks.append((number, choice))
    return chunks


def _is_choice_token(text: str) -> bool:
    normalized = _normalize_digits(text)
    return bool(_choice_marker_no(normalized) or _choice_chunks(normalized))


def _looks_like_prefixed_question_text(text: str) -> bool:
    stripped = text.strip()
    match = re.match(r"^(10|[1-9])(.+)", stripped)
    if not match:
        return False
    body = match.group(2).strip()
    if not _has_hiragana(body):
        return False
    normalized_body = _normalize_for_match(body)
    return "。" in body or len(normalized_body) >= 8


def _looks_like_question_line(line: list[OcrToken]) -> bool:
    if not line:
        return False
    text = text_of(line, "")
    if len(_extract_choices(line)) >= 2:
        return False
    return _has_japanese(text) and not text.startswith(("もんだい", "のことば", "いちばん"))


def _looks_like_question_with_inline_choices(line: list[OcrToken]) -> bool:
    text = text_of(line, "")
    return bool(re.match(r"^(10|[1-9])", _normalize_digits(text))) and _has_japanese(text)


def _set_choice(found: dict[int, tuple[int, float, str]], number: int, text: str, priority: int, y: float) -> None:
    if not text:
        return
    previous = found.get(number)
    if (
        previous is None
        or priority < previous[0]
        or (priority == previous[0] and y > previous[1] + 8)
        or (priority == previous[0] and abs(y - previous[1]) <= 8 and len(text) < len(previous[2]))
    ):
        found[number] = (priority, y, text)


def _clean_choice(text: str) -> str:
    text = _normalize_digits(text)
    text = text.strip("「『【[(（ \t\r\n:：・-？?")
    text = text.strip("」』】])） \t\r\n:：・-？?")
    return text.strip()


def _clean_sentence(text: str) -> str:
    text = _normalize_digits(text)
    text = re.sub(r"\\s+", "", text)
    return _strip_question_prefix(text)


def _strip_question_prefix(text: str) -> str:
    return re.sub(r"^(10|[1-9])", "", _normalize_digits(text).strip())


def _resolve_from_glossary(sentence: str, choices: list[str], page_type: str) -> tuple[str, str, int | None]:
    entries = _load_glossary_entries()
    normalized_sentence = _normalize_for_match(sentence)
    normalized_choices = [_normalize_for_match(choice) for choice in choices]
    best: tuple[str, str, int | None] = ("", "", None)
    for entry in entries:
        surface = entry["surface"]
        reading = entry["reading"]
        if page_type == "spelling_mcq":
            if _normalize_for_match(reading) not in normalized_sentence:
                continue
            choice_no = _choice_no_for_answer(surface, normalized_choices)
            if choice_no:
                return reading, surface, choice_no
        else:
            if _normalize_for_match(surface) not in normalized_sentence:
                continue
            choice_no = _choice_no_for_answer(reading, normalized_choices)
            if choice_no:
                return surface, reading, choice_no
            if not best[0]:
                best = (surface, reading, None)
    return best


def _choice_no_for_answer(answer: str, normalized_choices: list[str]) -> int | None:
    normalized_answer = _normalize_for_match(answer)
    exact_matches = [idx for idx, choice in enumerate(normalized_choices, start=1) if choice == normalized_answer]
    if exact_matches:
        return exact_matches[-1]
    for idx, choice in enumerate(normalized_choices, start=1):
        if normalized_answer and normalized_answer in choice:
            return idx
    return None


@lru_cache(maxsize=1)
def _load_glossary_entries() -> list[dict[str, str]]:
    path = KOREAN_GLOSSARY_PATH
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = []
    for item in data:
        surface = str(item.get("surface") or "")
        reading = str(item.get("reading") or "")
        if surface and reading:
            entries.append({"surface": surface, "reading": reading})
    return sorted(entries, key=lambda item: max(len(item["surface"]), len(item["reading"])), reverse=True)


def _normalize_digits(text: str) -> str:
    return text.translate(FULLWIDTH_DIGITS)


def _normalize_for_match(text: str) -> str:
    normalized = re.sub(r"\\s+", "", _normalize_digits(text))
    return normalized.replace("目", "日").replace("囲", "国")


def _has_japanese(text: str) -> bool:
    return any(0x3040 <= ord(char) <= 0x30FF or 0x4E00 <= ord(char) <= 0x9FFF for char in text)


def _has_hiragana(text: str) -> bool:
    return any(0x3040 <= ord(char) <= 0x309F for char in text)


def _is_header_text(text: str) -> bool:
    normalized = _normalize_for_match(text)
    return any(marker in normalized for marker in ("もんだい", "もんしだい", "ことば", "かきますか", "えらんで", "いちばんいいもの"))


def _choices_from_question_mark_placeholder(text: str) -> list[tuple[int, str]]:
    if "？" not in text and "?" not in text:
        return []
    marker = "？" if "？" in text else "?"
    if text.startswith(marker):
        return [(2, text[1:])]
    if not text or text[0] not in {"1", "2", "3", "4"}:
        return []
    split_at = text.find(marker)
    if split_at <= 1:
        return []
    number = int(text[0])
    return [(number, text[1:split_at]), (min(4, number + 1), text[split_at + 1 :])]
