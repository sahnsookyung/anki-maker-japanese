from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from app.evaluation.golden import GoldenPage, normalize_text
from app.models.schemas import CardCandidate, ProcessResult


@dataclass(frozen=True)
class McqEvalResult:
    page_id: str
    image_path: str
    expected_page_type: str
    actual_page_type: str
    expected_questions: int
    extracted_questions: int
    matched_questions: int
    source_matched_questions: int
    sentence_matches: int
    target_matches: int
    choices_matches: int
    correct_answer_matches: int
    correct_choice_matches: int
    source_field_matches: int
    source_field_expected: int
    generated_cards: int
    missing_question_ids: list[str]
    source_mismatch_question_ids: list[str]

    @property
    def question_accuracy(self) -> float:
        return self.matched_questions / self.expected_questions if self.expected_questions else 0.0

    @property
    def source_field_accuracy(self) -> float:
        return self.source_field_matches / self.source_field_expected if self.source_field_expected else 0.0


def evaluate_mcq_page(golden: GoldenPage, process_result: ProcessResult) -> McqEvalResult:
    items = _question_items_from_cards(process_result.cards)
    live_tokens = {token.id: token for token in process_result.tokens}
    live_block_ids = {
        block.id for block in (process_result.document_parse.blocks if process_result.document_parse else []) if block.id
    }
    by_no: dict[int, dict[str, Any]] = {}
    for item in items:
        question_no = item.get("question_no")
        if isinstance(question_no, int):
            by_no.setdefault(question_no, item)

    matched: set[str] = set()
    source_matched: set[str] = set()
    sentence_matches = 0
    target_matches = 0
    choices_matches = 0
    answer_matches = 0
    choice_matches = 0
    for question in golden.expected_questions:
        item = by_no.get(question.question_no)
        if not item:
            continue
        source_item = _strict_source_fields(item)
        sentence_ok = _sentence_matches(str(source_item.get("sentence", "")), question.sentence) and _source_field_has_ocr_evidence(
            item,
            "sentence",
            source_item.get("sentence"),
            live_tokens,
            live_block_ids,
        )
        target_ok = normalize_text(str(source_item.get("target", ""))) == normalize_text(question.target) and _target_has_ocr_evidence(
            item,
            source_item.get("target"),
            live_tokens,
            live_block_ids,
        )
        choices_ok = _choices_match(source_item.get("choices"), question.choices) and _choices_have_ocr_evidence(
            item,
            source_item.get("choices"),
            live_tokens,
            live_block_ids,
        )
        choice_ok = source_item.get("correct_choice_no") == question.correct_choice_no and _correct_choice_no_has_ocr_evidence(
            item,
            source_item.get("correct_choice_no"),
            live_tokens,
            live_block_ids,
        )
        answer_ok = normalize_text(str(source_item.get("correct_answer", ""))) == normalize_text(question.correct_answer) and _correct_answer_has_ocr_evidence(
            item,
            source_item,
            live_tokens,
            live_block_ids,
        )
        semantic_target_ok = normalize_text(str(item.get("target", ""))) == normalize_text(question.target)
        semantic_answer_ok = normalize_text(str(item.get("correct_answer", ""))) == normalize_text(question.correct_answer)
        semantic_choice_ok = item.get("correct_choice_no") == question.correct_choice_no
        if sentence_ok:
            sentence_matches += 1
        if target_ok:
            target_matches += 1
        if choices_ok:
            choices_matches += 1
        if answer_ok:
            answer_matches += 1
        if choice_ok:
            choice_matches += 1
        if semantic_target_ok and semantic_answer_ok and semantic_choice_ok:
            matched.add(question.question_id)
        if sentence_ok and target_ok and choices_ok and answer_ok and choice_ok:
            source_matched.add(question.question_id)

    source_field_matches = sentence_matches + target_matches + choices_matches + answer_matches + choice_matches
    source_field_expected = len(golden.expected_questions) * 5

    return McqEvalResult(
        page_id=golden.page_id,
        image_path=str(golden.image_path),
        expected_page_type=golden.expected_page_type,
        actual_page_type=process_result.page.page_type,
        expected_questions=len(golden.expected_questions),
        extracted_questions=len(items),
        matched_questions=len(matched),
        source_matched_questions=len(source_matched),
        sentence_matches=sentence_matches,
        target_matches=target_matches,
        choices_matches=choices_matches,
        correct_answer_matches=answer_matches,
        correct_choice_matches=choice_matches,
        source_field_matches=source_field_matches,
        source_field_expected=source_field_expected,
        generated_cards=len(process_result.cards),
        missing_question_ids=[
            question.question_id for question in golden.expected_questions if question.question_id not in matched
        ],
        source_mismatch_question_ids=[
            question.question_id for question in golden.expected_questions if question.question_id not in source_matched
        ],
    )


def _question_items_from_cards(cards: list[CardCandidate]) -> list[dict[str, Any]]:
    by_source: dict[str, dict[str, Any]] = {}
    for card in cards:
        if card.source_type != "question_item":
            continue
        by_source.setdefault(card.source_id, card.source)
    return list(by_source.values())


def _strict_source_fields(item: dict[str, Any]) -> dict[str, Any]:
    source_fields = item.get("source_fields")
    return source_fields if isinstance(source_fields, dict) else item


OCR_TOKEN_PROVENANCES = {"ocr", "crop_ocr", "region_ocr", "answer_strip_ocr", "prompt_line_ocr", "choice_glyph_ocr", "google_vision"}


def _source_field_has_ocr_evidence(
    item: dict[str, Any],
    field: str,
    expected_value: Any,
    live_tokens: dict[str, Any],
    live_block_ids: set[str],
) -> bool:
    evidence = item.get("field_evidence")
    field_evidence = evidence.get(field) if isinstance(evidence, dict) else None
    return _evidence_has_live_support(field_evidence, expected_value, live_tokens, live_block_ids)


def _target_has_ocr_evidence(item: dict[str, Any], target: Any, live_tokens: dict[str, Any], live_block_ids: set[str]) -> bool:
    return _source_field_has_ocr_evidence(item, "target", target, live_tokens, live_block_ids) or _source_field_has_ocr_evidence(
        item,
        "sentence",
        target,
        live_tokens,
        live_block_ids,
    )


def _choices_have_ocr_evidence(item: dict[str, Any], choices: Any, live_tokens: dict[str, Any], live_block_ids: set[str]) -> bool:
    if not isinstance(choices, list) or len(choices) != 4:
        return False
    evidence = item.get("field_evidence")
    if not isinstance(evidence, dict):
        return False
    combined = evidence.get("choices")
    if _evidence_has_live_support(combined, json.dumps(choices, ensure_ascii=False), live_tokens, live_block_ids):
        return True
    return all(
        _evidence_has_live_support(evidence.get(f"choice_{index}"), str(choice), live_tokens, live_block_ids)
        for index, choice in enumerate(choices, start=1)
    )


def _correct_choice_no_has_ocr_evidence(
    item: dict[str, Any],
    choice_no: Any,
    live_tokens: dict[str, Any],
    live_block_ids: set[str],
) -> bool:
    if not isinstance(choice_no, int):
        return False
    return _source_field_has_ocr_evidence(item, "correct_choice_no", str(choice_no), live_tokens, live_block_ids)


def _correct_answer_has_ocr_evidence(
    item: dict[str, Any],
    source_item: dict[str, Any],
    live_tokens: dict[str, Any],
    live_block_ids: set[str],
) -> bool:
    correct_answer = source_item.get("correct_answer")
    if _source_field_has_ocr_evidence(item, "correct_answer", correct_answer, live_tokens, live_block_ids):
        return True
    choices = source_item.get("choices")
    choice_no = source_item.get("correct_choice_no")
    if not isinstance(choices, list) or not isinstance(choice_no, int) or not (1 <= choice_no <= len(choices)):
        return False
    if normalize_text(str(choices[choice_no - 1])) != normalize_text(str(correct_answer)):
        return False
    return _correct_choice_no_has_ocr_evidence(item, choice_no, live_tokens, live_block_ids) and _source_field_has_ocr_evidence(
        item,
        f"choice_{choice_no}",
        correct_answer,
        live_tokens,
        live_block_ids,
    )


def _evidence_has_live_support(
    value: Any,
    expected_value: Any,
    live_tokens: dict[str, Any],
    live_block_ids: set[str],
) -> bool:
    if not isinstance(value, dict):
        return False
    bbox = value.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    evidence_text = str(value.get("text") or "")
    expected_text = str(expected_value or "")
    if not _text_supports_value(evidence_text, expected_text):
        return False
    provenance = value.get("provenance")
    if provenance in OCR_TOKEN_PROVENANCES:
        token_ids = value.get("token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            return False
        tokens = [live_tokens.get(token_id) for token_id in token_ids if isinstance(token_id, str)]
        if len(tokens) != len(token_ids) or any(token is None for token in tokens):
            return False
        token_text = "".join(str(getattr(token, "text", "") or "") for token in tokens if token)
        return _text_supports_value(token_text, expected_text)
    if provenance == "paddleocr_vl_block":
        block_ids = value.get("block_ids")
        if not isinstance(block_ids, list) or not block_ids:
            return False
        return all(isinstance(block_id, str) and block_id in live_block_ids for block_id in block_ids)
    return False


def _text_supports_value(actual: str, expected: str) -> bool:
    actual_norm = normalize_text(actual)
    expected_norm = normalize_text(expected)
    if not expected_norm:
        return False
    if expected_norm in actual_norm or actual_norm in expected_norm:
        return True
    actual_compact = _strip_loose_punctuation(actual_norm)
    expected_compact = _strip_loose_punctuation(expected_norm)
    if not expected_compact:
        return False
    return expected_compact in actual_compact or actual_compact in expected_compact


def _strip_loose_punctuation(value: str) -> str:
    return re.sub(r"[。、，,.．!?？！\"'「」『』（）()\[\]{}:：;；\\s]+", "", value)


def _sentence_matches(actual: str, expected: str) -> bool:
    actual_norm = normalize_text(actual)
    expected_norm = normalize_text(expected)
    return bool(expected_norm and (actual_norm == expected_norm or expected_norm in actual_norm))


def _choices_match(actual: Any, expected: list[str]) -> bool:
    if not isinstance(actual, list):
        return False
    normalized_actual = [normalize_text(str(choice)) for choice in actual]
    normalized_expected = [normalize_text(choice) for choice in expected]
    return normalized_actual == normalized_expected
