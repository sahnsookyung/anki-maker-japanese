from __future__ import annotations

from dataclasses import dataclass
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
        sentence_ok = _sentence_matches(str(item.get("sentence", "")), question.sentence)
        target_ok = normalize_text(str(item.get("target", ""))) == normalize_text(question.target)
        choices_ok = _choices_match(item.get("choices"), question.choices)
        answer_ok = normalize_text(str(item.get("correct_answer", ""))) == normalize_text(question.correct_answer)
        choice_ok = item.get("correct_choice_no") == question.correct_choice_no
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
        if target_ok and answer_ok and choice_ok:
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
