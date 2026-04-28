from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
from statistics import median

from app.core.config import KOREAN_GLOSSARY_PATH
from app.core.ids import new_id
from app.extraction.field_evidence import static_evidence, token_evidence
from app.extraction.geometry import group_tokens_by_line, text_of, union_bbox
from app.extraction.sentence_order import repair_predicate_first_sentence
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
        sentence_like = _sentence_tokens_from_block(block, _question_no(block))
        sentence_text = text_of(sentence_like, "")
        if _has_hiragana(sentence_text) and not _is_header_text(sentence_text):
            filtered_blocks.append(block)

    for sequence_no, block in enumerate(filtered_blocks, start=1):
        block = _augment_block_with_nearby_choices(block, lines)
        choice_records = _extract_choice_records(block)
        choices = [record["text"] for number in range(1, 5) if (record := choice_records.get(number))]
        question_no = _question_no(block) or sequence_no
        sentence_tokens = _sentence_tokens_from_block(block, question_no)
        sentence = _clean_sentence(text_of(sentence_tokens, ""))
        target = _guess_target(sentence_tokens, page_type)
        resolved_target, resolved_answer, resolved_choice_no = _resolve_from_glossary(sentence, choices, page_type)
        if resolved_target:
            target = resolved_target
        if resolved_answer:
            resolved_choice_no = _choice_no_for_answer_from_records(resolved_answer, choice_records) or resolved_choice_no
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
        field_evidence = _mcq_field_evidence(
            block=block,
            sentence_tokens=sentence_tokens,
            target_tokens=target_tokens,
            choice_records=choice_records,
            question_no=question_no,
            sentence=sentence,
            target=target,
            correct_choice_no=correct_choice_no,
            correct_answer=correct_answer,
            answer_source=_answer_source(question_no, answer_map, resolved_choice_no),
        )
        confidence = _block_confidence(block)
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
                "field_evidence": field_evidence,
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


def _augment_block_with_nearby_choices(block: list[OcrToken], lines: list[list[OcrToken]]) -> list[OcrToken]:
    if not block:
        return block
    block_ids = {token.id for token in block}
    min_y = min(token.bbox[1] for token in block)
    max_y = max(token.bbox[3] for token in block)
    initial_max_y = max_y
    augmented = list(block)
    for line in lines:
        if all(token.id in block_ids for token in line):
            continue
        choices = _extract_choices(line)
        if not choices:
            continue
        if _looks_like_question_line(line) and not all(_is_choice_token(token.text) for token in line):
            continue
        center_y = sum((token.bbox[1] + token.bbox[3]) / 2 for token in line) / len(line)
        if min_y - 12 <= center_y <= min(max_y + 38, initial_max_y + 70):
            augmented.extend(token for token in line if token.id not in block_ids)
            block_ids.update(token.id for token in line)
            max_y = max(max_y, *(token.bbox[3] for token in line))
    return augmented


def _question_no(tokens: list[OcrToken]) -> int | None:
    for token in tokens[:5]:
        text = _normalize_digits(token.text)
        if _choice_text_after_marker(text) or _choice_chunks(text):
            continue
        if text in CIRCLED:
            return CIRCLED[text]
        if text.isdigit() and 1 <= int(text) <= 10:
            return int(text)
        match = re.match(r"^(10|[1-9])\D+", text)
        if match:
            return int(match.group(1))
    return None


def _sentence_tokens_from_block(block: list[OcrToken], question_no: int | None) -> list[OcrToken]:
    lines = group_tokens_by_line(block, tolerance=22)
    selected: list[OcrToken] = []
    for line in lines:
        if _is_choice_line(line):
            continue
        sentence_tokens = [_clean_sentence_token(token, question_no) for token in line]
        sentence_tokens = [token for token in sentence_tokens if token is not None]
        if not sentence_tokens:
            continue
        selected.extend(sentence_tokens)
        if any("。" in token.text for token in sentence_tokens):
            break
    if not selected:
        selected = [
            token
            for token in block
            if _clean_sentence_token(token, question_no) is not None
        ]
    return _trim_sentence_tokens(selected)


def _is_choice_line(line: list[OcrToken]) -> bool:
    choices = _extract_choices(line)
    if not choices:
        return False
    if _looks_like_question_with_inline_choices(line):
        return False
    non_choice_text = "".join(
        token.text
        for token in line
        if not _is_choice_token(token.text) and not QUESTION_NO_RE.match(_normalize_digits(token.text))
    )
    return len(choices) >= 2 or not _has_hiragana(non_choice_text)


def _clean_sentence_token(token: OcrToken, question_no: int | None) -> OcrToken | None:
    text = _normalize_digits(token.text).strip()
    if not text or _is_separator_noise(text):
        return None
    if _is_choice_token(text):
        return None
    if _is_question_marker(text, question_no):
        return None
    if not _has_japanese(text):
        return None
    return token


def _is_question_marker(text: str, question_no: int | None) -> bool:
    normalized = _normalize_digits(text).strip()
    if not QUESTION_NO_RE.match(normalized):
        return False
    if question_no is None:
        return True
    return normalized == str(question_no) or CIRCLED.get(normalized) == question_no


def _is_separator_noise(text: str) -> bool:
    return bool(re.fullmatch(r"[-‐‑‒–—―ー・、。,.．\s]+", text))


def _trim_sentence_tokens(tokens: list[OcrToken]) -> list[OcrToken]:
    trimmed: list[OcrToken] = []
    for token in tokens:
        trimmed.append(token)
        if "。" in token.text:
            break
    return trimmed


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


def _mcq_field_evidence(
    *,
    block: list[OcrToken],
    sentence_tokens: list[OcrToken],
    target_tokens: list[OcrToken],
    choice_records: dict[int, dict[str, object]],
    question_no: int,
    sentence: str,
    target: str,
    correct_choice_no: int | None,
    correct_answer: str,
    answer_source: str,
) -> dict[str, dict[str, object]]:
    question_tokens = [
        token
        for token in block[:5]
        if _normalize_digits(token.text) == str(question_no) or _normalize_digits(token.text).startswith(str(question_no))
    ]
    field_evidence: dict[str, dict[str, object]] = {
        "question_no": token_evidence(question_tokens, str(question_no)) if question_tokens else static_evidence(str(question_no)),
        "sentence": token_evidence(sentence_tokens, sentence),
        "target": token_evidence(target_tokens, target) if target_tokens else static_evidence(target),
        "answer_source": static_evidence(answer_source, "metadata"),
    }
    for number in range(1, 5):
        record = choice_records.get(number)
        if not record:
            continue
        choice_tokens = [token for token in record.get("tokens", []) if isinstance(token, OcrToken)]
        field_evidence[f"choice_{number}"] = token_evidence(choice_tokens, str(record.get("text", "")))
    if correct_choice_no and correct_choice_no in choice_records:
        record = choice_records[correct_choice_no]
        choice_tokens = [token for token in record.get("tokens", []) if isinstance(token, OcrToken)]
        field_evidence["correct_answer"] = token_evidence(choice_tokens, correct_answer)
    else:
        field_evidence["correct_answer"] = static_evidence(correct_answer, answer_source or "unknown")
    return field_evidence


def _extract_choices(tokens: list[OcrToken]) -> list[str]:
    records = _extract_choice_records(tokens)
    return [record["text"] for number in range(1, 5) if (record := records.get(number))]


def _extract_choice_records(tokens: list[OcrToken]) -> dict[int, dict[str, object]]:
    ordered = sorted(tokens, key=lambda t: (t.bbox[1], t.bbox[0]))
    found: dict[int, dict[str, object]] = {}
    idx = 0
    while idx < len(ordered):
        token = ordered[idx]
        text = _normalize_digits(token.text)
        marker_num = _choice_marker_no(text)
        chunks = _choice_chunks(text)
        if chunks:
            for number, choice in chunks:
                _set_choice(
                    found,
                    number,
                    _clean_choice(choice),
                    priority=0,
                    y=token.bbox[1],
                    confidence=token.confidence,
                    tokens=[token],
                )
        elif marker_num and _choice_text_after_marker(text):
            _set_choice(
                found,
                marker_num,
                _clean_choice(_choice_text_after_marker(text)),
                priority=0,
                y=token.bbox[1],
                confidence=token.confidence,
                tokens=[token],
            )
        elif marker_num and idx + 1 < len(ordered) and not _is_choice_token(ordered[idx + 1].text):
            confidence = min(token.confidence, ordered[idx + 1].confidence)
            _set_choice(
                found,
                marker_num,
                _clean_choice(ordered[idx + 1].text),
                priority=1,
                y=token.bbox[1],
                confidence=confidence,
                tokens=[token, ordered[idx + 1]],
            )
            idx += 1
        idx += 1
    _fill_missing_leading_choice(found, ordered)
    _fill_missing_choices_by_position(found, ordered)
    return found


def _fill_missing_leading_choice(found: dict[int, dict[str, object]], ordered: list[OcrToken]) -> None:
    if 1 in found or 2 not in found:
        return
    choice_two_tokens = [token for token in found[2].get("tokens", []) if isinstance(token, OcrToken)]
    if not choice_two_tokens:
        return
    choice_two_left = min(token.bbox[0] for token in choice_two_tokens)
    choice_two_y = float(found[2]["y"])
    candidates = [
        token
        for token in ordered
        if not _is_choice_token(token.text)
        and _has_japanese(token.text)
        and abs(token.bbox[1] - choice_two_y) <= 14
        and token.bbox[2] <= choice_two_left
    ]
    if not candidates:
        return
    candidates.sort(key=lambda token: (abs(token.bbox[2] - choice_two_left), -token.confidence))
    token = candidates[0]
    _set_choice(
        found,
        1,
        _clean_choice(token.text),
        priority=2,
        y=token.bbox[1],
        confidence=token.confidence,
        tokens=[token],
        inferred=True,
    )


def _fill_missing_choices_by_position(found: dict[int, dict[str, object]], ordered: list[OcrToken]) -> None:
    used_ids = {
        token.id
        for record in found.values()
        for token in record.get("tokens", [])
        if isinstance(token, OcrToken)
    }
    for number in range(1, 5):
        if number in found:
            continue
        left_numbers = [candidate for candidate in range(1, number) if candidate in found]
        right_numbers = [candidate for candidate in range(number + 1, 5) if candidate in found]
        if not left_numbers or not right_numbers:
            continue
        left = found[max(left_numbers)]
        right = found[min(right_numbers)]
        left_tokens = [token for token in left.get("tokens", []) if isinstance(token, OcrToken)]
        right_tokens = [token for token in right.get("tokens", []) if isinstance(token, OcrToken)]
        if not left_tokens or not right_tokens:
            continue
        left_x = max(token.bbox[2] for token in left_tokens)
        right_x = min(token.bbox[0] for token in right_tokens)
        if right_x <= left_x:
            continue
        expected_y = (float(left["y"]) + float(right["y"])) / 2
        expected_x = (left_x + right_x) / 2
        candidates = [
            token
            for token in ordered
            if token.id not in used_ids
            and not _is_choice_token(token.text)
            and _has_japanese(token.text)
            and abs(token.bbox[1] - expected_y) <= 22
            and left_x <= (token.bbox[0] + token.bbox[2]) / 2 <= right_x
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda token, expected_x=expected_x: (abs(((token.bbox[0] + token.bbox[2]) / 2) - expected_x), -token.confidence)
        )
        token = candidates[0]
        _set_choice(
            found,
            number,
            _clean_choice(token.text),
            priority=3,
            y=token.bbox[1],
            confidence=token.confidence,
            tokens=[token],
            inferred=True,
        )
        used_ids.add(token.id)


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
    if re.search(r"[1-4①-④]", _normalize_digits(body)):
        return False
    if not _has_hiragana(body):
        return False
    if not body or not (0x3040 <= ord(body[0]) <= 0x309F):
        return False
    normalized_body = _normalize_for_match(body)
    return "。" in body or len(normalized_body) >= 8


def _block_confidence(tokens: list[OcrToken]) -> float:
    content_confidences = [
        token.confidence
        for token in tokens
        if token.script_class != "punctuation" and _has_japanese_or_number(token.text)
    ]
    if not content_confidences:
        content_confidences = [token.confidence for token in tokens]
    return round(float(median(content_confidences)), 3) if content_confidences else 0.5


def _has_japanese_or_number(text: str) -> bool:
    return any(
        char.isdigit() or 0x3040 <= ord(char) <= 0x30FF or 0x4E00 <= ord(char) <= 0x9FFF
        for char in _normalize_digits(text)
    )


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


def _set_choice(
    found: dict[int, dict[str, object]],
    number: int,
    text: str,
    priority: int,
    y: float,
    confidence: float,
    tokens: list[OcrToken],
    inferred: bool = False,
) -> None:
    if not text:
        return
    previous = found.get(number)
    if (
        previous is None
        or priority < int(previous["priority"])
        or (
            priority == int(previous["priority"])
            and y > float(previous["y"]) + 8
            and confidence >= float(previous.get("confidence", 0.0)) - 0.1
        )
        or (priority == int(previous["priority"]) and abs(y - float(previous["y"])) <= 8 and len(text) < len(str(previous["text"])))
    ):
        found[number] = {
            "priority": priority,
            "y": y,
            "text": text,
            "confidence": confidence,
            "tokens": tokens,
            "inferred": inferred,
        }


def _clean_choice(text: str) -> str:
    text = _normalize_digits(text)
    text = text.replace("叩", "り").replace("おほえ", "おぼえ").replace("ほu", "ほん")
    text = text.strip("「『【[(（ \t\r\n:：・-？?")
    text = text.strip("」』】])） \t\r\n:：・-？?")
    return text.strip()


def _clean_sentence(text: str) -> str:
    text = _normalize_digits(text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[-‐‑‒–—―]+", "", text)
    text = _strip_question_prefix(text)
    text, _repaired = repair_predicate_first_sentence(text)
    stop_index = text.find("。")
    if stop_index >= 0:
        text = text[: stop_index + 1]
    return text


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
        return exact_matches[0]
    for idx, choice in enumerate(normalized_choices, start=1):
        if normalized_answer and normalized_answer in choice:
            return idx
    return None


def _choice_no_for_answer_from_records(answer: str, choice_records: dict[int, dict[str, object]]) -> int | None:
    normalized_answer = _normalize_for_match(answer)
    exact_matches = [
        number
        for number, record in choice_records.items()
        if _normalize_for_match(str(record.get("text", ""))) == normalized_answer
    ]
    if not exact_matches:
        return None
    clean_matches = [number for number in exact_matches if not _choice_record_has_conflicting_marker(number, choice_records[number])]
    inferred_matches = [number for number in clean_matches if choice_records[number].get("inferred")]
    if inferred_matches:
        return min(inferred_matches)
    return max(clean_matches or exact_matches)


def _choice_record_has_conflicting_marker(number: int, record: dict[str, object]) -> bool:
    tokens = [token for token in record.get("tokens", []) if isinstance(token, OcrToken)]
    text = _normalize_digits("".join(token.text for token in tokens))
    markers = re.findall(r"[1-4①-④]", text)
    if len(markers) <= 1:
        return False
    expected = str(number)
    return not text.strip().startswith(expected)


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
    normalized = re.sub(r"\s+", "", _normalize_digits(text))
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
