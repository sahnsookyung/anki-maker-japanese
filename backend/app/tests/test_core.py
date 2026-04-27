from __future__ import annotations

from app.core.script import classify_script
from app.extraction.answer_strip import parse_answer_strip_text
from app.export.tsv import clean_tsv_field


def test_script_classifier() -> None:
    assert classify_script("がっこう") == "hiragana"
    assert classify_script("学校") == "kanji"
    assert classify_script("학교") == "hangul"
    assert classify_script("学校が") == "mixed"


def test_answer_strip_parser() -> None:
    assert parse_answer_strip_text("1 2 2 3 3 1 10 4") == {1: 2, 2: 3, 3: 1, 10: 4}
    assert parse_answer_strip_text("① 2 ② 3") == {1: 2, 2: 3}


def test_tsv_cleaning() -> None:
    assert clean_tsv_field("a\tb\nc") == "a b<br>c"
