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
    source = {**item, "id": source_id}
    return [
        CardCandidate(
            id=new_id("card"),
            page_id=page_id,
            source_type="vocab_item",
            source_id=source_id,
            source=source,
            note_type="jp_vocab_reading",
            front=f"{escape(surface)}<br>뜻: {escape(meaning)}<br><br>읽는 법?",
            back=escape(reading),
            tags=["jlpt", "vocab", "reading"],
            confidence=confidence,
            review_state=state,
            source_bbox=bbox,
            warnings=warnings,
        ),
        CardCandidate(
            id=new_id("card"),
            page_id=page_id,
            source_type="vocab_item",
            source_id=source_id,
            source=source,
            note_type="jp_vocab_meaning",
            front=f"{escape(surface)}<br>{escape(reading)}<br><br>뜻?",
            back=escape(meaning),
            tags=["jlpt", "vocab", "meaning"],
            confidence=confidence,
            review_state=state,
            source_bbox=bbox,
            warnings=warnings,
        ),
        CardCandidate(
            id=new_id("card"),
            page_id=page_id,
            source_type="vocab_item",
            source_id=source_id,
            source=source,
            note_type="jp_vocab_writing",
            front=f"{escape(reading)}<br>{escape(meaning)}<br><br>올바른 표기는?",
            back=escape(surface),
            tags=["jlpt", "vocab", "writing"],
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
    correct_choice_no = item.get("correct_choice_no")
    correct_answer = item.get("correct_answer", "")
    bbox = item.get("bbox")
    confidence = float(item.get("confidence", 0.5))
    warnings = list(item.get("warnings", []))
    blocked = not (sentence and target and correct_answer and bbox)
    if len(choices) != 4:
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
    active = CardCandidate(
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
    choice_html = "<br>".join(f"{idx}. {escape(str(choice))}" for idx, choice in enumerate(choices, start=1))
    exam = CardCandidate(
        id=new_id("card"),
        page_id=page_id,
        source_type="question_item",
        source_id=source_id,
        source=source,
        note_type=f"jp_{question_type}_exam",
        front=f"{escape(sentence)}<br><br>{choice_html}",
        back=f"정답: {correct_choice_no}. {escape(correct_answer)}",
        tags=[*tags, "exam-style"],
        confidence=confidence,
        review_state=state,
        source_bbox=bbox,
        warnings=warnings,
    )
    return [active, exam]
