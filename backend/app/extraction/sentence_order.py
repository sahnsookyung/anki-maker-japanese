from __future__ import annotations

import re


TERMINAL_PREDICATE_RE = re.compile(r"(ました|ます|でした|です|でしょう|あります|います|いますか)$")


def repair_predicate_first_sentence(text: str) -> tuple[str, bool]:
    compact = re.sub(r"\s+", "", text).strip()
    if "。" not in compact or compact.endswith("。"):
        return text, False
    predicate, tail = compact.split("。", 1)
    if not predicate or not tail or not TERMINAL_PREDICATE_RE.search(predicate):
        return text, False
    repaired_tail = _repair_tail_order(tail)
    if repaired_tail == tail and len(tail) < 4:
        return text, False
    return f"{repaired_tail}{predicate}。", True


def _repair_tail_order(tail: str) -> str:
    # Common OCR failure on horizontal Japanese workbook lines: the predicate is read first,
    # then the preceding phrases are emitted right-to-left. Reorder a simple place/adjective/subject
    # prefix without changing normal left-to-right OCR output.
    subject = _extract_subject_phrase(tail)
    locative = _extract_locative_phrase(tail, subject)
    remainder = tail
    for phrase in (locative, subject):
        if phrase:
            remainder = remainder.replace(phrase, "", 1)
    if locative and subject and remainder:
        return f"{locative}{remainder}{subject}"
    if locative and subject:
        return f"{locative}{subject}"
    return tail


def _extract_locative_phrase(text: str, subject: str) -> str:
    search_from = 0
    if subject:
        subject_index = text.find(subject)
        if subject_index >= 0:
            search_from = subject_index + len(subject)
    suffix = text[search_from:]
    for start in range(len(suffix)):
        match = re.match(r"([ぁ-んァ-ン一-龯]{1,8}?(?:に|で|へ))", suffix[start:])
        if not match:
            continue
        phrase = match.group(1)
        # Avoid treating adjective endings such as しろい as locatives.
        particle = phrase[-1]
        base = phrase[:-1] if particle in {"に", "で", "へ"} else phrase
        if base.endswith("い") and len(base) <= 4:
            continue
        return phrase
    return ""


def _extract_subject_phrase(text: str) -> str:
    match = re.search(r"([ぁ-んァ-ン一-龯]+?が)", text)
    return match.group(1) if match else ""
