from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterator

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import OCR_PAGE_WORKER_MAX_RSS_MB, OCR_VL_PAGE_WORKER_MAX_RSS_MB
from app.core.ids import new_id
from app.db import database
from app.evaluation.golden import load_golden_pages
from app.evaluation.mcq_eval import McqEvalResult, evaluate_mcq_page
from app.evaluation.vocab_eval import VocabEvalResult, evaluate_vocab_page
from app.extraction import pipeline
from app.ocr.engines import PADDLEOCR_ENGINE, PADDLEOCR_VL_ENGINE, normalize_ocr_engine
from app.ocr.page_worker import run_page_process_worker
from app.ocr.profiles import (
    BASELINE_MODEL_PROFILE,
    DEFAULT_EXTRACTION_VARIANT,
    normalize_extraction_variant,
    profile_env_overrides,
    resolve_ocr_model_profile,
)
from app.models.schemas import Page, ProcessResult


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate local extraction against golden workbook annotations.")
    parser.add_argument(
        "--golden",
        default="../data/evaluation/golden_pages.example.json",
        help="Path to a golden_pages JSON file.",
    )
    parser.add_argument(
        "--engine",
        default=PADDLEOCR_ENGINE,
        help="Candidate-generation OCR engine: paddleocr, paddleocr_vl, or all.",
    )
    parser.add_argument("--model-profile", default=BASELINE_MODEL_PROFILE, help="OCR model profile for fresh processing.")
    parser.add_argument("--extraction-variant", default=DEFAULT_EXTRACTION_VARIANT, help="Extraction variant for fresh processing.")
    parser.add_argument("--from-db", action="store_true", help="Evaluate already persisted cards instead of processing images.")
    parser.add_argument("--db-path", default="", help="SQLite DB path to use with --from-db.")
    parser.add_argument("--run-id", default="", help="Evaluate a specific persisted OCR run id with --from-db.")
    parser.add_argument("--work-dir", default="", help="Optional isolated runtime directory for fresh processing.")
    parser.add_argument("--keep-work-dir", action="store_true", help="Keep isolated runtime files for debugging.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a human-readable summary.")
    args = parser.parse_args()
    args.model_profile = resolve_ocr_model_profile(args.model_profile).id
    args.extraction_variant = normalize_extraction_variant(args.extraction_variant)

    backend_dir = BACKEND_DIR
    repo_root = backend_dir.parent
    golden_path = Path(args.golden)
    if not golden_path.is_absolute():
        golden_path = backend_dir / golden_path
    pages = load_golden_pages(golden_path.resolve(), repo_root)
    if not pages:
        print("No supported golden pages found.", file=sys.stderr)
        return 2

    results: list[tuple[str, VocabEvalResult | McqEvalResult]] = []
    if args.from_db:
        if args.db_path:
            database.DB_PATH = Path(args.db_path).resolve()
        database.init_db()
        if args.run_id:
            run = database.get_ocr_run(args.run_id)
            if not run:
                print(f"No persisted OCR run matched {args.run_id}.", file=sys.stderr)
                return 2
            run_page = database.get_page(run.page_id)
            if not run_page:
                print(f"No persisted page matched OCR run {args.run_id}.", file=sys.stderr)
                return 2
            pages = [golden for golden in pages if _page_matches_golden(run_page, golden)]
            if not pages:
                print(f"OCR run {args.run_id} does not match any page in {golden_path.name}.", file=sys.stderr)
                return 2
        for golden in pages:
            if not golden.image_path.exists():
                print(f"Missing image: {golden.image_path}", file=sys.stderr)
                return 2
            process_result = _process_result_from_db(golden, args.run_id or None)
            if process_result is None:
                print(f"No persisted DB page matched {golden.image_path.name}.", file=sys.stderr)
                return 2
            engine_label = _persisted_engine_label(process_result)
            if golden.expected_rows:
                results.append((engine_label, evaluate_vocab_page(golden, process_result)))
            elif golden.expected_questions:
                results.append((engine_label, evaluate_mcq_page(golden, process_result)))
    else:
        with _evaluation_runtime(args.work_dir, args.keep_work_dir):
            database.init_db()
            engines = [PADDLEOCR_ENGINE, PADDLEOCR_VL_ENGINE] if args.engine == "all" else [normalize_ocr_engine(args.engine)]
            for golden in pages:
                if not golden.image_path.exists():
                    print(f"Missing image: {golden.image_path}", file=sys.stderr)
                    return 2
                for engine in engines:
                    page = Page(
                        id=new_id("eval"),
                        original_image_path=str(golden.image_path),
                        upload_name=golden.image_path.name,
                        display_name=golden.image_path.stem,
                        processed_image_path=None,
                        page_type="uploaded",
                        page_type_confidence=0.0,
                        created_at=datetime.now(timezone.utc).isoformat(),
                    )
                    database.upsert_page(page)
                    process_result = _process_page_for_evaluation(
                        page,
                        engine,
                        model_profile=args.model_profile,
                        extraction_variant=args.extraction_variant,
                        redirect_logs=args.json,
                    )
                    if golden.expected_rows:
                        results.append((engine, evaluate_vocab_page(golden, process_result)))
                    elif golden.expected_questions:
                        results.append((engine, evaluate_mcq_page(golden, process_result)))

    if args.json:
        print(json.dumps([_result_dict(result, engine) for engine, result in results], ensure_ascii=False, indent=2))
    else:
        for engine, result in results:
            print(_format_result(result, engine))
    return 0


@contextmanager
def _evaluation_runtime(work_dir_arg: str, keep_work_dir: bool) -> Iterator[Path]:
    previous_db_path = database.DB_PATH
    previous_processed_dir = pipeline.PROCESSED_DIR
    created_temp = False
    if work_dir_arg:
        work_dir = Path(work_dir_arg).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="anki_eval_")).resolve()
        created_temp = True
    processed_dir = work_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    database.DB_PATH = work_dir / "evaluation.db"
    pipeline.PROCESSED_DIR = processed_dir
    try:
        yield work_dir
    finally:
        database.DB_PATH = previous_db_path
        pipeline.PROCESSED_DIR = previous_processed_dir
        if created_temp and not keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


def _process_page_for_evaluation(
    page: Page,
    engine: str,
    *,
    model_profile: str = BASELINE_MODEL_PROFILE,
    extraction_variant: str = DEFAULT_EXTRACTION_VARIANT,
    redirect_logs: bool = False,
) -> ProcessResult:
    if redirect_logs:
        with redirect_stdout(sys.stderr):
            return _process_page_for_evaluation(page, engine, model_profile=model_profile, extraction_variant=extraction_variant)
    max_rss_mb = OCR_VL_PAGE_WORKER_MAX_RSS_MB if engine == PADDLEOCR_VL_ENGINE else OCR_PAGE_WORKER_MAX_RSS_MB
    return run_page_process_worker(
        page.id,
        engine,
        max_rss_mb=max_rss_mb,
        env_overrides={
            "ANKI_MAKER_DB": str(database.DB_PATH),
            "ANKI_MAKER_PROCESSED_DIR": str(pipeline.PROCESSED_DIR),
            **profile_env_overrides(model_profile),
        },
        model_profile=model_profile,
        extraction_variant=extraction_variant,
    )


def _process_result_from_db(golden, run_id: str | None = None) -> ProcessResult | None:
    if run_id:
        run = database.get_ocr_run(run_id)
        if not run:
            return None
        page = database.get_page(run.page_id)
        if not page:
            return None
        if not _page_matches_golden(page, golden):
            return None
        return ProcessResult(
            page=page,
            tokens=database.get_tokens(page.id, run.id),
            cards=database.get_cards(page.id, run.id),
            script_summary={},
            answer_map={},
            ocr_run=run,
        )
    candidates = database.list_pages()
    expected_name = golden.image_path.name
    expected_stem = golden.image_path.stem
    page = next(
        (
            item
            for item in candidates
            if item.upload_name == expected_name
            or item.display_name == expected_stem
            or Path(item.original_image_path).name == expected_name
        ),
        None,
    )
    if not page:
        return None
    return ProcessResult(
        page=page,
        tokens=database.get_tokens(page.id),
        cards=database.get_cards(page.id),
        script_summary={},
        answer_map={},
        ocr_run=database.get_active_ocr_run(page.id),
    )


def _page_matches_golden(page: Page, golden) -> bool:
    expected_name = golden.image_path.name
    expected_stem = golden.image_path.stem
    return (
        page.upload_name == expected_name
        or page.display_name == expected_stem
        or Path(page.original_image_path).name == expected_name
    )


def _persisted_engine_label(process_result: ProcessResult) -> str:
    if process_result.ocr_run:
        return f"persisted_{process_result.ocr_run.engine}"
    sources = {token.source for token in process_result.tokens}
    warnings = " ".join(process_result.page.warnings)
    if PADDLEOCR_VL_ENGINE in sources or "PaddleOCR-VL" in warnings:
        return f"persisted_{PADDLEOCR_VL_ENGINE}"
    if PADDLEOCR_ENGINE in sources:
        return f"persisted_{PADDLEOCR_ENGINE}"
    return "persisted_unknown"


def _result_dict(result: VocabEvalResult | McqEvalResult, engine: str) -> dict:
    if isinstance(result, McqEvalResult):
        return {
            "engine": engine,
            "page_id": result.page_id,
            "image_path": result.image_path,
            "expected_page_type": result.expected_page_type,
            "actual_page_type": result.actual_page_type,
            "expected_questions": result.expected_questions,
            "extracted_questions": result.extracted_questions,
            "matched_questions": result.matched_questions,
            "semantic_accuracy": round(result.question_accuracy, 4),
            "source_matched_questions": result.source_matched_questions,
            "source_field_accuracy": round(result.source_field_accuracy, 4),
            "sentence_matches": result.sentence_matches,
            "target_matches": result.target_matches,
            "choices_matches": result.choices_matches,
            "correct_answer_matches": result.correct_answer_matches,
            "correct_choice_matches": result.correct_choice_matches,
            "source_field_matches": result.source_field_matches,
            "source_field_expected": result.source_field_expected,
            "generated_cards": result.generated_cards,
            "missing_question_ids": result.missing_question_ids,
            "source_mismatch_question_ids": result.source_mismatch_question_ids,
        }
    return {
        "engine": engine,
        "page_id": result.page_id,
        "image_path": result.image_path,
        "expected_page_type": result.expected_page_type,
        "actual_page_type": result.actual_page_type,
        "expected_rows": result.expected_rows,
        "extracted_items": result.extracted_items,
        "layout_matched_rows": result.layout_matched_rows,
        "ocr_supported_items": result.ocr_supported_items,
        "glossary_supported_items": result.glossary_supported_items,
        "matched_rows": result.matched_rows,
        "row_accuracy": round(result.row_accuracy, 4),
        "layout_recall": round(result.layout_recall, 4),
        "surface_accuracy": round(result.surface_accuracy, 4),
        "reading_accuracy": round(result.reading_accuracy, 4),
        "meaning_accuracy": round(result.meaning_accuracy, 4),
        "surface_matches": result.surface_matches,
        "reading_matches": result.reading_matches,
        "surface_reading_matches": result.surface_reading_matches,
        "meaning_matches": result.meaning_matches,
        "generated_notes": result.generated_notes,
        "korean_field_missing_hangul": result.korean_field_missing_hangul,
        "japanese_field_has_hangul": result.japanese_field_has_hangul,
        "missing_row_ids": result.missing_row_ids,
    }


def _format_result(result: VocabEvalResult | McqEvalResult, engine: str) -> str:
    if isinstance(result, McqEvalResult):
        lines = [
            f"Page: {result.page_id} ({engine})",
            f"  type: expected={result.expected_page_type} actual={result.actual_page_type}",
            f"  semantic questions: matched={result.matched_questions}/{result.expected_questions} accuracy={result.question_accuracy:.1%}",
            f"  source fields: matched={result.source_field_matches}/{result.source_field_expected} accuracy={result.source_field_accuracy:.1%}",
            f"  extracted_questions={result.extracted_questions} generated_cards={result.generated_cards}",
            "  field matches="
            f"sentence {result.sentence_matches}, target {result.target_matches}, choices {result.choices_matches}, "
            f"correct_answer {result.correct_answer_matches}, correct_choice {result.correct_choice_matches}",
        ]
        if result.missing_question_ids:
            lines.append(f"  missing_question_ids: {', '.join(result.missing_question_ids[:20])}")
        return "\n".join(lines)
    lines = [
        f"Page: {result.page_id} ({engine})",
        f"  type: expected={result.expected_page_type} actual={result.actual_page_type}",
        f"  rows: matched={result.matched_rows}/{result.expected_rows} accuracy={result.row_accuracy:.1%}",
        (
            f"  extracted_items={result.extracted_items} ocr_supported_items={result.ocr_supported_items} "
            f"glossary_supported_items={result.glossary_supported_items} generated_notes={result.generated_notes}"
        ),
        (
            f"  layout_recall={result.layout_recall:.1%} "
            f"surface={result.surface_matches}/{result.expected_rows} reading={result.reading_matches}/{result.expected_rows} "
            f"meaning={result.meaning_matches}/{result.expected_rows}"
        ),
        f"  surface+reading matches={result.surface_reading_matches} meaning matches={result.meaning_matches}",
        f"  script confusion: korean_field_missing_hangul={result.korean_field_missing_hangul}, japanese_field_has_hangul={result.japanese_field_has_hangul}",
    ]
    if result.missing_row_ids:
        lines.append(f"  missing_row_ids: {', '.join(result.missing_row_ids[:20])}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
