from __future__ import annotations

from html import escape
from typing import Any

from app.core.ids import new_id
from app.models.schemas import CardCandidate


def review_state(confidence: float, warnings: list[str], blocked: bool = False) -> str:
    if blocked:
        return "red"
    if warnings or confidence < 0.75:
        return "yellow"
    return "green"


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def vocab_cards(page_id: str, item: dict[str, Any]) -> list[CardCandidate]:
    surface = item.get("surface", "")
    reading = item.get("reading", "")
    meaning = item.get("meaning_ko", "")
    bbox = item.get("bbox")
    confidence = float(item.get("confidence", 0.5))
    warnings = list(item.get("warnings", []))
    blocked = not (surface and reading and meaning and bbox)
    state = review_state(confidence, warnings, blocked)
    source_id = item.get("id", new_id("vocab"))
    source = {
        **item,
        "id": source_id,
        "study_writing": item.get("study_writing", True),
        "study_reading": item.get("study_reading", False),
        "study_meaning": item.get("study_meaning", False),
    }
    return [
        CardCandidate(
            id=new_id("card"),
            page_id=page_id,
            source_type="vocab_item",
            source_id=source_id,
            source=source,
            note_type="jp_vocab_entry",
            front=escape(reading),
            back=escape(surface),
            tags=["jlpt", "vocab"],
            confidence=confidence,
            review_state=state,
            source_bbox=bbox,
            warnings=warnings,
        ),
    ]


def mcq_cards(page_id: str, item: dict[str, Any]) -> list[CardCandidate]:
    question_type = item.get("question_type", "reading_mcq")
    sentence = item.get("sentence", "")
    target = item.get("target", "")
    choices = item.get("choices", [])
    correct_answer = item.get("correct_answer", "")
    bbox = item.get("bbox")
    confidence = float(item.get("confidence", 0.5))
    warnings = list(item.get("warnings", []))
    blocked = not (sentence and target and correct_answer and bbox)
    if len(choices) != 4 or not all(choices):
        blocked = True
        warnings.append("Expected exactly four choices.")
    if item.get("answer_source") in {"unknown", "model_inferred"}:
        blocked = True
        warnings.append("Answer source is not export-safe.")
    warnings = _unique(warnings)
    state = review_state(confidence, warnings, blocked)
    source_id = item.get("id", new_id("q"))
    source = {**item, "id": source_id}

    prompt = "읽는 법?" if question_type == "reading_mcq" else "올바른 표기는?"
    tags = ["jlpt", "mcq", "reading" if question_type == "reading_mcq" else "writing"]
    return [
        CardCandidate(
            id=new_id("card"),
            page_id=page_id,
            source_type="question_item",
            source_id=source_id,
            source=source,
            note_type=f"jp_{question_type}_recall",
            front=f"{escape(sentence)}<br><br>밑줄: {escape(target)}<br>{prompt}",
            back=escape(correct_answer),
            tags=tags,
            confidence=confidence,
            review_state=state,
            source_bbox=bbox,
            warnings=warnings,
        )
    ]
