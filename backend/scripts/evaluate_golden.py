from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.ids import new_id
from app.db import database
from app.evaluation.golden import load_golden_pages
from app.evaluation.mcq_eval import McqEvalResult, evaluate_mcq_page
from app.evaluation.vocab_eval import VocabEvalResult, evaluate_vocab_page
from app.extraction.pipeline import process_page
from app.models.schemas import Page


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate local extraction against golden workbook annotations.")
    parser.add_argument(
        "--golden",
        default="../data/evaluation/golden_pages.example.json",
        help="Path to a golden_pages JSON file.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a human-readable summary.")
    args = parser.parse_args()

    backend_dir = BACKEND_DIR
    repo_root = backend_dir.parent
    golden_path = Path(args.golden)
    if not golden_path.is_absolute():
        golden_path = backend_dir / golden_path
    pages = load_golden_pages(golden_path.resolve(), repo_root)
    if not pages:
        print("No supported golden pages found. This evaluator currently supports category=vocab_table.", file=sys.stderr)
        return 2

    database.init_db()
    results: list[VocabEvalResult | McqEvalResult] = []
    for golden in pages:
        if not golden.image_path.exists():
            print(f"Missing image: {golden.image_path}", file=sys.stderr)
            return 2
        page = Page(
            id=new_id("eval"),
            original_image_path=str(golden.image_path),
            processed_image_path=None,
            page_type="uploaded",
            page_type_confidence=0.0,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        database.upsert_page(page)
        process_result = process_page(page)
        if golden.expected_rows:
            results.append(evaluate_vocab_page(golden, process_result))
        elif golden.expected_questions:
            results.append(evaluate_mcq_page(golden, process_result))

    if args.json:
        print(json.dumps([_result_dict(result) for result in results], ensure_ascii=False, indent=2))
    else:
        for result in results:
            print(_format_result(result))
    return 0


def _result_dict(result: VocabEvalResult | McqEvalResult) -> dict:
    if isinstance(result, McqEvalResult):
        return {
            "page_id": result.page_id,
            "image_path": result.image_path,
            "expected_page_type": result.expected_page_type,
            "actual_page_type": result.actual_page_type,
            "expected_questions": result.expected_questions,
            "extracted_questions": result.extracted_questions,
            "matched_questions": result.matched_questions,
            "question_accuracy": round(result.question_accuracy, 4),
            "target_matches": result.target_matches,
            "correct_answer_matches": result.correct_answer_matches,
            "correct_choice_matches": result.correct_choice_matches,
            "generated_cards": result.generated_cards,
            "missing_question_ids": result.missing_question_ids,
        }
    return {
        "page_id": result.page_id,
        "image_path": result.image_path,
        "expected_page_type": result.expected_page_type,
        "actual_page_type": result.actual_page_type,
        "expected_rows": result.expected_rows,
        "extracted_items": result.extracted_items,
        "matched_rows": result.matched_rows,
        "row_accuracy": round(result.row_accuracy, 4),
        "surface_reading_matches": result.surface_reading_matches,
        "meaning_matches": result.meaning_matches,
        "generated_cards": result.generated_cards,
        "korean_field_missing_hangul": result.korean_field_missing_hangul,
        "japanese_field_has_hangul": result.japanese_field_has_hangul,
        "missing_row_ids": result.missing_row_ids,
    }


def _format_result(result: VocabEvalResult | McqEvalResult) -> str:
    if isinstance(result, McqEvalResult):
        lines = [
            f"Page: {result.page_id}",
            f"  type: expected={result.expected_page_type} actual={result.actual_page_type}",
            f"  questions: matched={result.matched_questions}/{result.expected_questions} accuracy={result.question_accuracy:.1%}",
            f"  extracted_questions={result.extracted_questions} generated_cards={result.generated_cards}",
            f"  target matches={result.target_matches} correct_answer matches={result.correct_answer_matches} correct_choice matches={result.correct_choice_matches}",
        ]
        if result.missing_question_ids:
            lines.append(f"  missing_question_ids: {', '.join(result.missing_question_ids[:20])}")
        return "\n".join(lines)
    lines = [
        f"Page: {result.page_id}",
        f"  type: expected={result.expected_page_type} actual={result.actual_page_type}",
        f"  rows: matched={result.matched_rows}/{result.expected_rows} accuracy={result.row_accuracy:.1%}",
        f"  extracted_items={result.extracted_items} generated_cards={result.generated_cards}",
        f"  surface+reading matches={result.surface_reading_matches} meaning matches={result.meaning_matches}",
        f"  script confusion: korean_field_missing_hangul={result.korean_field_missing_hangul}, japanese_field_has_hangul={result.japanese_field_has_hangul}",
    ]
    if result.missing_row_ids:
        lines.append(f"  missing_row_ids: {', '.join(result.missing_row_ids[:20])}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
