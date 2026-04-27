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
    target_matches: int
    correct_answer_matches: int
    correct_choice_matches: int
    generated_cards: int
    missing_question_ids: list[str]

    @property
    def question_accuracy(self) -> float:
        return self.matched_questions / self.expected_questions if self.expected_questions else 0.0


def evaluate_mcq_page(golden: GoldenPage, process_result: ProcessResult) -> McqEvalResult:
    items = _question_items_from_cards(process_result.cards)
    by_no: dict[int, dict[str, Any]] = {}
    for item in items:
        question_no = item.get("question_no")
        if isinstance(question_no, int):
            by_no.setdefault(question_no, item)

    matched: set[str] = set()
    target_matches = 0
    answer_matches = 0
    choice_matches = 0
    for question in golden.expected_questions:
        item = by_no.get(question.question_no)
        if not item:
            continue
        target_ok = normalize_text(str(item.get("target", ""))) == normalize_text(question.target)
        answer_ok = normalize_text(str(item.get("correct_answer", ""))) == normalize_text(question.correct_answer)
        choice_ok = item.get("correct_choice_no") == question.correct_choice_no
        if target_ok:
            target_matches += 1
        if answer_ok:
            answer_matches += 1
        if choice_ok:
            choice_matches += 1
        if target_ok and answer_ok and choice_ok:
            matched.add(question.question_id)

    return McqEvalResult(
        page_id=golden.page_id,
        image_path=str(golden.image_path),
        expected_page_type=golden.expected_page_type,
        actual_page_type=process_result.page.page_type,
        expected_questions=len(golden.expected_questions),
        extracted_questions=len(items),
        matched_questions=len(matched),
        target_matches=target_matches,
        correct_answer_matches=answer_matches,
        correct_choice_matches=choice_matches,
        generated_cards=len(process_result.cards),
        missing_question_ids=[
            question.question_id for question in golden.expected_questions if question.question_id not in matched
        ],
    )


def _question_items_from_cards(cards: list[CardCandidate]) -> list[dict[str, Any]]:
    by_source: dict[str, dict[str, Any]] = {}
    for card in cards:
        if card.source_type != "question_item":
            continue
        by_source.setdefault(card.source_id, card.source)
    return list(by_source.values())
