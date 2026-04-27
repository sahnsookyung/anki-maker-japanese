from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GoldenVocabRow:
    row_id: str
    section: str
    column: str
    surface: str
    reading: str
    meaning_ko: str


@dataclass(frozen=True)
class GoldenQuestion:
    question_id: str
    question_no: int
    sentence: str
    target: str
    choices: list[str]
    correct_choice_no: int | None
    correct_answer: str
    answer_source: str


@dataclass(frozen=True)
class GoldenPage:
    page_id: str
    image_path: Path
    category: str
    expected_page_type: str
    expected_rows: list[GoldenVocabRow] = field(default_factory=list)
    expected_questions: list[GoldenQuestion] = field(default_factory=list)


def load_golden_pages(path: Path, repo_root: Path) -> list[GoldenPage]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pages: list[GoldenPage] = []
    for page_data in data.get("pages", []):
        if page_data.get("stub"):
            continue
        category = str(page_data.get("category") or "")
        if category not in {"vocab_table", "spelling_vocab_table", "reading_mcq", "spelling_mcq"}:
            continue
        image_path = Path(page_data.get("image_path") or "")
        if not image_path.is_absolute():
            image_path = repo_root / image_path
        expected_rows_data = page_data.get("expected_rows", page_data.get("expected_entries", []))
        pages.append(
            GoldenPage(
                page_id=str(page_data["page_id"]),
                image_path=image_path,
                category=category,
                expected_page_type=str(page_data.get("expected_page_type") or category),
                expected_rows=[_row_from_dict(row) for row in _flatten_rows(expected_rows_data)],
                expected_questions=[
                    _question_from_dict(question) for question in _flatten_rows(page_data.get("expected_questions", []))
                ],
            )
        )
    return pages


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value).strip().lower()


def meaning_matches(actual: str, expected: str) -> bool:
    actual_norm = normalize_text(actual)
    expected_parts = [part for part in re.split(r"[,，、/()（）\s]+", expected) if part]
    if not actual_norm:
        return False
    return any(normalize_text(part) in actual_norm for part in expected_parts)


def _flatten_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        rows.extend(_flatten_rows(item))
    return rows


def _row_from_dict(data: dict[str, Any]) -> GoldenVocabRow:
    return GoldenVocabRow(
        row_id=str(data["row_id"]),
        section=str(data.get("section") or ""),
        column=str(data.get("column") or ""),
        surface=str(data["surface"]),
        reading=str(data["reading"]),
        meaning_ko=str(data["meaning_ko"]),
    )


def _question_from_dict(data: dict[str, Any]) -> GoldenQuestion:
    correct_choice = data.get("correct_choice_no")
    return GoldenQuestion(
        question_id=str(data["question_id"]),
        question_no=int(data["question_no"]),
        sentence=str(data.get("sentence") or ""),
        target=str(data["target"]),
        choices=[str(choice) for choice in data.get("choices", [])],
        correct_choice_no=int(correct_choice) if correct_choice is not None else None,
        correct_answer=str(data.get("correct_answer") or ""),
        answer_source=str(data.get("answer_source") or ""),
    )
