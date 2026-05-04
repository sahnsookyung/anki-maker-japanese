from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from app.core.ids import new_id
from app.extraction.answer_strip import parse_answer_strip_text
from app.extraction.field_evidence import block_evidence, block_list_evidence, static_evidence
from app.extraction.geometry import union_bbox
from app.extraction.sentence_order import repair_predicate_first_sentence
from app.models.schemas import DocumentParseBlock, DocumentParseResult
from app.validation.dictionary import DictionaryValidator


_UNDERLINE_RE = re.compile(r"\\underline\{(?:\\text\{)?([^{}]+)\}?\}")
_TEXT_RE = re.compile(r"\\text\{([^{}]+)\}")
_ANSWER_LABEL_RE = re.compile(r"[答답]\s*(?=[①-⑳0-9])")
_LATEX_DAY_ANSWER_RE = re.compile(r"\\text\{\s*日\s*\}\s*(?=[①-⑳0-9])")
_DAY_ANSWER_PREFIX_RE = re.compile(r"^\s*日\s*(?=[①-⑳0-9])")
_FLAT_QUESTION_BREAK_RE = re.compile(r"(\s[4④]\s*[^0-9①-④答답日]{1,24})(?=\s+(?:[1-9]\d?|[①-⑳])\s+)")
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
_FULLWIDTH_DIGITS = str.maketrans("１２３４５６７８９０", "1234567890")
_CIRCLED_DIGITS = str.maketrans({key: str(value) for key, value in _CIRCLED.items()})


@dataclass(frozen=True)
class DocumentExtraction:
    page_type: str
    page_type_confidence: float
    answer_map: dict[int, int]
    items: list[dict[str, Any]]
    warnings: list[str]


@dataclass(frozen=True)
class _Line:
    text: str
    block: DocumentParseBlock | None
    bbox: list[float] | None
    order: int
    blocks: tuple[DocumentParseBlock, ...] = ()


def extract_from_document_parse(
    result: DocumentParseResult,
    validator: DictionaryValidator | None = None,
) -> DocumentExtraction:
    validator = validator or DictionaryValidator()
    lines = _document_lines(result)
    full_text = "\n".join(line.text for line in lines)
    page_type, confidence = classify_document_parse(full_text)
    answer_map = _answer_map_from_lines(lines)
    warnings: list[str] = []
    if not lines:
        return DocumentExtraction(
            page_type="unknown_review_required",
            page_type_confidence=0.0,
            answer_map={},
            items=[],
            warnings=["PaddleOCR-VL returned no parseable document lines."],
        )

    if page_type == "vocab_table":
        items = extract_vl_vocab_items(lines, validator)
    elif page_type in {"reading_mcq", "spelling_mcq"}:
        items = extract_vl_mcq_items(lines, answer_map, page_type, validator)
    else:
        items = []
        warnings.append("PaddleOCR-VL page type was unclear; review document blocks manually.")

    if result.blocks and all(not block.bbox for block in result.blocks):
        warnings.append("PaddleOCR-VL returned text without usable block bboxes; focused evidence is unavailable.")
    if _has_only_page_level_blocks(result):
        warnings.append("PaddleOCR-VL returned page-level block geometry only; visual evidence is semantic but not field-precise.")
    return DocumentExtraction(
        page_type=page_type,
        page_type_confidence=confidence,
        answer_map=answer_map,
        items=items,
        warnings=warnings,
    )


def classify_document_parse(text: str) -> tuple[str, float]:
    normalized = _clean_document_text(text)
    question_count = _document_question_count(normalized)
    has_choice_markers = len(re.findall(r"\s[1-4①-④]\s*[\u3040-\u30ff\u3400-\u9fff]", normalized)) >= 2
    has_answer_strip = _find_answer_marker(normalized) is not None
    has_vocab_header = "기출어휘" in normalized or "語彙" in normalized or "어휘" in normalized
    has_hangul = _has_hangul(normalized)
    has_many_vocab_rows = len(_vocab_triples(normalized)) >= 3
    if has_vocab_header and (has_hangul or has_many_vocab_rows) and question_count < 3:
        return "vocab_table", 0.68
    if question_count >= 2 or has_answer_strip or (question_count >= 1 and has_choice_markers):
        return ("spelling_mcq" if _looks_like_spelling_mcq(normalized) else "reading_mcq", 0.72)
    if has_many_vocab_rows or (has_hangul and (has_vocab_header or question_count == 0)):
        return "vocab_table", 0.58
    return "unknown_review_required", 0.2


def extract_vl_vocab_items(lines: list[_Line], validator: DictionaryValidator) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for line in lines:
        for surface, reading, meaning in _vocab_triples(line.text):
            key = (surface, reading, meaning)
            if key in seen:
                continue
            seen.add(key)
            _, validation_warnings = validator.validate_vocab(surface, reading)
            confidence = 0.78 if line.bbox else 0.62
            warnings = list(validation_warnings)
            if not line.bbox:
                warnings.append("PaddleOCR-VL row has no usable visual bbox.")
            item_id = new_id("vocab")
            field_evidence = {
                "surface": block_evidence(line.block, surface),
                "reading": block_evidence(line.block, reading),
                "meaning_ko": block_evidence(line.block, meaning),
            }
            items.append(
                {
                    "id": item_id,
                    "type": "vocab_item",
                    "surface": surface,
                    "reading": reading,
                    "meaning_ko": meaning,
                    "field_evidence": field_evidence,
                    "evidence_blocks": _block_ids([line.block]),
                    "bbox": line.bbox,
                    "confidence": confidence,
                    "needs_review": bool(warnings),
                    "warnings": warnings,
                    "provider": "paddleocr_vl",
                }
            )
    return items


def extract_vl_mcq_items(
    lines: list[_Line],
    answer_map: dict[int, int],
    page_type: str,
    validator: DictionaryValidator,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for question in _question_chunks(lines):
        question_parts = _split_question_prefix(question.text)
        if not question_parts:
            continue
        question_token, body = question_parts
        question_no = _question_number(question_token)
        if not question_no:
            continue
        choices_by_no = _choices_from_text(body)
        choices = [choices_by_no.get(index, "") for index in range(1, 5)]
        present_choices = [choice for choice in choices if choice]
        sentence_part = _text_before_choices(body)
        sentence = _clean_sentence(sentence_part)
        target = _target_from_text(sentence_part)
        correct_choice_no = answer_map.get(question_no)
        correct_answer = choices_by_no.get(correct_choice_no, "") if correct_choice_no else ""
        answer_source = "answer_strip" if correct_choice_no else "unknown"
        canonical_answer = ""
        if not target and correct_answer:
            target, canonical_answer = _infer_target_from_dictionary(sentence, correct_answer, present_choices, validator)
        if page_type == "reading_mcq" and correct_choice_no and target and canonical_answer and 1 <= correct_choice_no <= len(choices):
            choices[correct_choice_no - 1] = canonical_answer
            correct_answer = canonical_answer
        warnings: list[str] = []
        if len(present_choices) != 4:
            warnings.append("Could not extract exactly four choices.")
        if not target:
            warnings.append("Could not confidently identify the underlined target.")
        if not correct_choice_no:
            warnings.append("Correct choice is missing from the answer strip.")
        if not question.bbox:
            warnings.append("PaddleOCR-VL question block has no usable visual bbox.")
        confidence = 0.78
        if warnings:
            confidence = 0.68 if sentence and choices else 0.52
        field_evidence = {
            "question_no": block_list_evidence(question.blocks, str(question_no)),
            "sentence": block_list_evidence(question.blocks, sentence),
            "target": block_list_evidence(question.blocks, target),
            "correct_answer": block_list_evidence(question.blocks, correct_answer),
            "answer_source": static_evidence(answer_source, "answer_strip" if correct_choice_no else "unknown"),
        }
        for index, choice in enumerate(choices, start=1):
            field_evidence[f"choice_{index}"] = block_list_evidence(question.blocks, choice)
        items.append(
            {
                "id": new_id("q"),
                "type": "question_item",
                "question_type": "spelling_mcq" if page_type == "spelling_mcq" else "reading_mcq",
                "question_no": question_no,
                "sentence": sentence,
                "target": target,
                "choices": choices,
                "correct_choice_no": correct_choice_no,
                "correct_answer": correct_answer,
                "answer_source": answer_source,
                "field_evidence": field_evidence,
                "bbox": question.bbox,
                "evidence_blocks": _block_ids(question.blocks),
                "token_roles": {},
                "confidence": round(confidence, 3),
                "needs_review": bool(warnings),
                "warnings": warnings,
                "provider": "paddleocr_vl",
            }
        )
    return sorted(items, key=lambda item: int(item.get("question_no") or 10_000))


def _document_lines(result: DocumentParseResult) -> list[_Line]:
    lines: list[_Line] = []
    ordered_blocks = sorted(result.blocks, key=lambda block: (block.order if block.order is not None else 10_000, block.id or ""))
    for index, block in enumerate(ordered_blocks):
        for text in _split_block_text(block.content):
            lines.append(
                _Line(
                    text=text,
                    block=block,
                    bbox=block.bbox,
                    order=block.order if block.order is not None else index,
                    blocks=(block,),
                )
            )
    if not lines and result.markdown_text.strip():
        for index, text in enumerate(_split_block_text(result.markdown_text)):
            lines.append(_Line(text=text, block=None, bbox=None, order=index, blocks=()))
    return lines


def _has_only_page_level_blocks(result: DocumentParseResult) -> bool:
    blocks_with_boxes = [block for block in result.blocks if block.bbox]
    if len(blocks_with_boxes) != 1:
        return False
    x1, y1, x2, y2 = blocks_with_boxes[0].bbox or [0, 0, 0, 0]
    return x1 <= 1 and y1 <= 1 and x2 > 300 and y2 > 300


def _split_block_text(text: str) -> list[str]:
    candidates: list[str] = []
    for raw_line in text.replace("\r", "\n").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if "|" in stripped:
            stripped = " ".join(part.strip() for part in stripped.split("|") if part.strip())
        if "\\text{" in stripped and "\\underline" not in stripped:
            question_text, answer_text = _split_answer_fragment(stripped)
            candidates.extend(_split_flattened_question_text(question_text))
            if answer_text:
                candidates.append(answer_text)
        else:
            candidates.extend(_split_flattened_question_text(stripped))
    if not candidates and text.strip():
        candidates.append(text.strip())
    return candidates


def _split_flattened_question_text(text: str) -> list[str]:
    split_text = _FLAT_QUESTION_BREAK_RE.sub(r"\1\n", text)
    return [part.strip() for part in split_text.splitlines() if part.strip()]


def _document_question_count(text: str) -> int:
    count = 0
    for line in text.splitlines():
        count += sum(1 for part in _split_flattened_question_text(line) if _split_question_prefix(part))
    return count


def _split_question_prefix(text: str) -> tuple[str, str] | None:
    stripped = text.strip()
    if stripped and stripped[0] in {"□", "☐", "▢"}:
        stripped = stripped[1:].lstrip()
    if not stripped:
        return None
    for circled in _CIRCLED:
        if stripped.startswith(circled):
            return circled, stripped[len(circled) :].lstrip()
    normalized = stripped.translate(_FULLWIDTH_DIGITS)
    digit_count = 0
    while digit_count < min(3, len(normalized)) and normalized[digit_count].isdigit():
        digit_count += 1
    if digit_count == 0:
        return None
    question = normalized[:digit_count]
    if question.startswith("0"):
        return None
    return question, stripped[digit_count:].lstrip()


def _find_answer_marker(text: str) -> re.Match[str] | None:
    matches = [
        match
        for pattern in (_ANSWER_LABEL_RE, _LATEX_DAY_ANSWER_RE, _DAY_ANSWER_PREFIX_RE)
        if (match := pattern.search(text))
    ]
    return min(matches, key=lambda match: match.start()) if matches else None


def _choice_markers(text: str) -> list[tuple[int, int]]:
    normalized = text.translate(_FULLWIDTH_DIGITS)
    markers: list[tuple[int, int]] = []
    for index, char in enumerate(normalized):
        number = _choice_number(char)
        if not number:
            continue
        if index > 0 and not normalized[index - 1].isspace():
            continue
        if index + 1 < len(normalized) and normalized[index + 1] in "0123456789①②③④":
            continue
        markers.append((index, number))
    return markers


def _question_chunks(lines: list[_Line]) -> list[_Line]:
    chunks: list[_Line] = []
    current: list[_Line] = []
    for line in lines:
        question_text, _answer_text = _split_answer_fragment(line.text)
        line = (
            line
            if question_text == line.text
            else _Line(text=question_text, block=line.block, bbox=line.bbox, order=line.order, blocks=line.blocks)
        )
        if not line.text.strip() or _is_answer_line(line.text) or _is_header_line(line.text):
            continue
        if current and _looks_like_choice_line(line.text):
            current.append(line)
            continue
        if current and _looks_like_standalone_choice_line(line.text):
            current.append(line)
            continue
        if _split_question_prefix(line.text):
            if current:
                chunks.append(_merge_lines(current))
            current = [line]
        elif current and not _split_question_prefix(line.text):
            current.append(line)
    if current:
        chunks.append(_merge_lines(current))
    return [chunk for chunk in chunks if _has_japanese(chunk.text)]


def _merge_lines(lines: list[_Line]) -> _Line:
    text = " ".join(line.text for line in lines)
    bboxes = [line.bbox for line in lines if line.bbox]
    blocks: list[DocumentParseBlock] = []
    seen_blocks: set[str] = set()
    for line in lines:
        for block in line.blocks:
            key = block.id or f"object:{id(block)}"
            if key in seen_blocks:
                continue
            seen_blocks.add(key)
            blocks.append(block)
    return _Line(
        text=text,
        block=lines[0].block,
        bbox=union_bbox(bboxes) if bboxes else None,
        order=lines[0].order,
        blocks=tuple(blocks),
    )


def _vocab_triples(text: str) -> list[tuple[str, str, str]]:
    segments = [_clean_segment(segment) for segment in re.split(r"[\s,，;；]+", _clean_document_text(text))]
    segments = [segment for segment in segments if segment and not _is_noise_segment(segment)]
    triples: list[tuple[str, str, str]] = []
    for meaning_index, raw_meaning in enumerate(segments):
        meaning = _clean_meaning(raw_meaning)
        if not meaning:
            continue
        lookback = segments[max(0, meaning_index - 4) : meaning_index]
        reading = next((_clean_reading(segment) for segment in reversed(lookback) if _clean_reading(segment)), "")
        surface = next(
            (_clean_surface(segment) for segment in reversed(lookback) if _clean_surface(segment) and _clean_surface(segment) != reading),
            "",
        )
        if not surface or not reading:
            lookahead = segments[meaning_index + 1 : meaning_index + 5]
            reading = reading or next((_clean_reading(segment) for segment in lookahead if _clean_reading(segment)), "")
            surface = surface or next((_clean_surface(segment) for segment in lookahead if _clean_surface(segment)), "")
        if surface and reading and meaning:
            triples.append((surface, reading, meaning))
    return triples


def _answer_map_from_lines(lines: list[_Line]) -> dict[int, int]:
    answer_parts: list[str] = []
    for line in lines:
        _question_text, answer_text = _split_answer_fragment(line.text)
        if answer_text:
            answer_parts.append(answer_text)
        elif _is_answer_line(line.text):
            answer_parts.append(line.text)
    answer_text = " ".join(answer_parts)
    return parse_answer_strip_text(answer_text) if answer_text else {}


def _split_answer_fragment(text: str) -> tuple[str, str]:
    match = _find_answer_marker(text)
    if not match:
        return text, ""
    return text[: match.start()].strip(), text[match.end() :].strip()


def _choices_from_text(text: str) -> dict[int, str]:
    normalized = text.translate(_FULLWIDTH_DIGITS)
    choices: dict[int, str] = {}
    markers = _choice_markers(normalized)
    for marker_index, (start, number) in enumerate(markers):
        end = markers[marker_index + 1][0] if marker_index + 1 < len(markers) else len(normalized)
        marker_width = 1
        choice = _clean_choice(normalized[start + marker_width : end])
        if number and choice:
            choices[number] = choice
    return choices


def _text_before_choices(text: str) -> str:
    markers = _choice_markers(text.translate(_FULLWIDTH_DIGITS))
    return text[: markers[0][0]].strip() if markers else text.strip()


def _target_from_text(text: str) -> str:
    match = _UNDERLINE_RE.search(text)
    if not match:
        return ""
    return _clean_target(match.group(1))


def _clean_sentence(text: str) -> str:
    cleaned = _clean_document_text(text)
    question_parts = _split_question_prefix(cleaned)
    if question_parts:
        _question_token, cleaned = question_parts
    cleaned = _clean_latex(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    repaired, _changed = repair_predicate_first_sentence(cleaned)
    return repaired or cleaned


def _infer_target_from_dictionary(
    sentence: str,
    reading: str,
    choices: list[str],
    validator: DictionaryValidator,
) -> tuple[str, str]:
    compact_sentence = re.sub(r"\s+", "", sentence)
    candidates = sorted(
        [(surface, candidate_reading) for surface, candidate_reading in validator.entries if surface in compact_sentence],
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for surface, candidate_reading in candidates:
        if candidate_reading == reading:
            return surface, candidate_reading
    for surface, candidate_reading in candidates:
        if any(_reading_distance(candidate_reading, choice) <= 1 for choice in [reading, *choices] if choice):
            return surface, candidate_reading
    return "", ""


def _reading_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if abs(len(left) - len(right)) > 1:
        return 2
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current.append(min(previous[right_index] + 1, current[-1] + 1, previous[right_index - 1] + cost))
        previous = current
    return previous[-1]


def _clean_choice(text: str) -> str:
    cleaned = _clean_latex(_clean_document_text(text))
    cleaned = re.sub(r"[^\u3040-\u30ff\u3400-\u9fff々〆〤]", "", cleaned)
    return cleaned.strip()


def _clean_target(text: str) -> str:
    target = _clean_choice(text)
    if len(target) > 3 and target.endswith("から"):
        return target[:-2]
    if len(target) > 2 and target[-1] in {"が", "を", "に", "は", "へ", "で"}:
        return target[:-1]
    return target


def _clean_surface(text: str) -> str:
    cleaned = _clean_choice(text)
    if not cleaned:
        return ""
    return cleaned


def _clean_reading(text: str) -> str:
    cleaned = re.sub(r"[^\u3040-\u30ff]", "", _clean_latex(text))
    japanese_chars = [ch for ch in _clean_latex(text) if _is_kana(ch) or _is_cjk(ch)]
    if japanese_chars and any(_is_cjk(ch) for ch in japanese_chars):
        return ""
    return cleaned.strip()


def _clean_meaning(text: str) -> str:
    cleaned = re.sub(r"^[□☐▢①-⑳0-9.\-:：\s]+", "", _clean_latex(text))
    cleaned = "".join(ch for ch in cleaned if _is_hangul(ch) or ch.isdigit())
    return cleaned.strip()


def _clean_segment(text: str) -> str:
    return _clean_latex(text).strip("[]()（）{}<>「」『』")


def _clean_latex(text: str) -> str:
    cleaned = _UNDERLINE_RE.sub(r"\1", text)
    cleaned = _TEXT_RE.sub(r"\1", cleaned)
    cleaned = re.sub(r"\\[A-Za-z]+", "", cleaned)
    return cleaned.replace("{", "").replace("}", "").replace("$", "")


def _clean_document_text(text: str) -> str:
    return text.translate(_FULLWIDTH_DIGITS).replace("\u3000", " ")


def _question_number(text: str) -> int | None:
    normalized = text.translate(_FULLWIDTH_DIGITS)
    if normalized in _CIRCLED:
        return _CIRCLED[normalized]
    return int(normalized) if normalized.isdigit() else None


def _choice_number(text: str) -> int | None:
    value = _question_number(text)
    return value if value and 1 <= value <= 4 else None


def _is_noise_segment(text: str) -> bool:
    return text in {"□", "☐", "▢"} or bool(re.fullmatch(r"[0-9①-⑳/]+", text))


def _is_answer_line(text: str) -> bool:
    if _is_header_line(text):
        return False
    if _find_answer_marker(text):
        return True
    compact_count = _compact_answer_pair_count(text)
    compact_text = re.sub(r"\s+", "", text.translate(_CIRCLED_DIGITS))
    if not _has_japanese(text) and (compact_count >= 2 or (compact_count == 1 and compact_text.isdigit())):
        return True
    normalized = text.translate(_CIRCLED_DIGITS)
    return len(re.findall(r"\b(?:[1-9]\d{0,2}|[1-4])\b", normalized)) >= 8


def _is_header_line(text: str) -> bool:
    normalized = _clean_document_text(text)
    return "もんだい" in normalized or "문제" in normalized or "기출어휘" in normalized


def _looks_like_choice_line(text: str) -> bool:
    return bool(re.match(r"^\s*[1①]\s+", text)) and len(_choices_from_text(text)) >= 2


def _looks_like_standalone_choice_line(text: str) -> bool:
    normalized = text.translate(_FULLWIDTH_DIGITS).strip()
    if not re.match(r"^[1-4①-④]\s+", normalized):
        return False
    choices = _choices_from_text(normalized)
    if len(choices) != 1:
        return False
    choice = next(iter(choices.values()))
    return len(choice) <= 12 and not re.search(r"[。！？?]|です|ます", choice)


def _compact_answer_pair_count(text: str) -> int:
    normalized = text.translate(_CIRCLED_DIGITS)
    digit_parts = [part for part in normalized.split() if part.isdigit()]
    if not digit_parts:
        return 0
    compact_parts = [
        part
        for part in digit_parts
        if len(part) == 2 or (len(part) == 3 and 1 <= int(part[:-1]) <= 20)
    ]
    return len(compact_parts) if len(compact_parts) == len(digit_parts) else 0


def _looks_like_spelling_mcq(text: str) -> bool:
    return "表記" in text or "표기" in text or len(re.findall(r"[1-4]\s*[\u4e00-\u9fff]", text)) >= 4


def _has_japanese(text: str) -> bool:
    return any(_is_kana(ch) or _is_cjk(ch) for ch in text)


def _has_hangul(text: str) -> bool:
    return any(_is_hangul(ch) for ch in text)


def _is_hangul(ch: str) -> bool:
    return "\uac00" <= ch <= "\ud7af"


def _is_kana(ch: str) -> bool:
    return "\u3040" <= ch <= "\u30ff" or ch == "ー"


def _is_cjk(ch: str) -> bool:
    return "\u3400" <= ch <= "\u9fff"


def _block_ids(blocks: tuple[DocumentParseBlock | None, ...] | list[DocumentParseBlock | None]) -> list[str]:
    return list(dict.fromkeys(block.id for block in blocks if block and block.id))
