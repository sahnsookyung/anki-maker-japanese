from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.ids import new_id
from app.db import database
from app.evaluation.golden import GoldenPage, load_golden_pages, meaning_matches, normalize_text
from app.evaluation.mcq_eval import McqEvalResult, evaluate_mcq_page
from app.evaluation.vocab_eval import VocabEvalResult, evaluate_vocab_page
from app.extraction import pipeline
from app.ocr.engines import PADDLEOCR_ENGINE, PADDLEOCR_VL_ENGINE, normalize_ocr_engine
from app.ocr.service import recognize_with_provider
from app.models.schemas import Page, ProcessResult


@dataclass(frozen=True)
class TextCoverageResult:
    mode: str
    page_id: str
    fields_matched: int
    fields_expected: int
    items_fully_matched: int
    items_expected: int
    warnings: list[str]

    @property
    def field_accuracy(self) -> float:
        return self.fields_matched / self.fields_expected if self.fields_expected else 0.0

    @property
    def item_accuracy(self) -> float:
        return self.items_fully_matched / self.items_expected if self.items_expected else 0.0


@dataclass(frozen=True)
class PageBenchmark:
    page_id: str
    image_path: str
    base: dict[str, Any]
    vl: dict[str, Any] | None
    memory_samples: list[dict[str, Any]]
    resource_metrics: dict[str, Any]
    errors: list[str]
    google_vision: dict[str, Any] | None = None


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare PaddleOCR and PaddleOCR-VL extraction accuracy and resource usage.")
    parser.add_argument("--golden", default="../data/evaluation/golden_pages.example.json")
    parser.add_argument("--engine", default=PADDLEOCR_ENGINE, help="Primary extraction engine: paddleocr, paddleocr_vl, or all.")
    parser.add_argument("--include-vl", action="store_true", help="Run PaddleOCR-VL extraction sequentially after base OCR.")
    parser.add_argument(
        "--include-google-vision",
        action="store_true",
        help="Run Google Vision OCR text coverage after local processing. Uncached calls require GOOGLE_VISION_ALLOW_CLOUD=true.",
    )
    parser.add_argument("--vl-limit", type=int, default=1, help="Maximum pages to send through PaddleOCR-VL in one run.")
    parser.add_argument("--work-dir", default="", help="Optional benchmark runtime directory. Defaults to a temp dir.")
    parser.add_argument("--keep-work-dir", action="store_true", help="Keep benchmark DB and processed images for debugging.")
    parser.add_argument("--in-process", action="store_true", help="Run all pages in this process instead of per-page subprocesses.")
    parser.add_argument("--worker-timeout-seconds", type=float, default=300, help="Kill a page worker after this many seconds.")
    parser.add_argument("--worker-max-rss-mb", type=float, default=8192, help="Kill a page worker if its RSS exceeds this limit.")
    parser.add_argument("--worker-page-id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--output-json", default="", help=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args()

    repo_root = BACKEND_DIR.parent
    golden_path = Path(args.golden)
    if not golden_path.is_absolute():
        golden_path = BACKEND_DIR / golden_path
    pages = load_golden_pages(golden_path.resolve(), repo_root)

    if args.worker_page_id:
        result = _run_worker_page(args, pages)
        _emit_results([result], args)
        return 0

    if not args.in_process:
        results = _run_pages_in_subprocesses(args, golden_path.resolve(), pages)
        _emit_results(results, args)
        return 0

    with _benchmark_runtime(args.work_dir, args.keep_work_dir):
        database.init_db()
        results: list[PageBenchmark] = []
        vl_pages_run = 0
        primary_engine = _primary_engine(args)
        for golden in pages:
            run_start = _resource_snapshot()
            memory_samples = [_memory_sample("worker_start")]
            process_result = _run_base_pipeline(golden, memory_samples, primary_engine)
            base_eval = _evaluate_base(golden, process_result)
            base_payload = _result_dict(base_eval, primary_engine)
            base_payload["ocr_text_coverage"] = _coverage_dict(_token_text_coverage(golden, process_result, primary_engine))
            memory_samples.append(_memory_sample("base_evaluated"))
            vl_eval: dict[str, Any] | None = None
            if _should_run_vl(args) and vl_pages_run < args.vl_limit and primary_engine != PADDLEOCR_VL_ENGINE:
                memory_samples.append(_memory_sample("before_vl"))
                vl_eval = _run_engine_evaluation(golden, memory_samples, PADDLEOCR_VL_ENGINE)
                memory_samples.append(_memory_sample("after_vl"))
                vl_pages_run += 1
            google_eval = (
                _run_google_vision_evaluation(golden, process_result, memory_samples)
                if getattr(args, "include_google_vision", False)
                else None
            )
            results.append(
                PageBenchmark(
                    page_id=golden.page_id,
                    image_path=str(golden.image_path),
                    base=base_payload,
                    vl=vl_eval,
                    google_vision=google_eval,
                    memory_samples=memory_samples,
                    resource_metrics=_resource_metrics(run_start, _resource_snapshot(), memory_samples),
                    errors=[],
                )
            )

        _emit_results(results, args)
    return 0


def _run_pages_in_subprocesses(args: argparse.Namespace, golden_path: Path, pages: list[GoldenPage]) -> list[PageBenchmark]:
    keep_work_dir = bool(args.keep_work_dir or args.work_dir)
    root_work_dir = Path(args.work_dir).resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix="anki-maker-ocr-bench-"))
    root_work_dir.mkdir(parents=True, exist_ok=True)
    results: list[PageBenchmark] = []
    vl_pages_run = 0
    try:
        for golden in pages:
            page_work_dir = root_work_dir / golden.page_id
            output_json = root_work_dir / f"{golden.page_id}.json"
            primary_engine = _primary_engine(args)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--golden",
                str(golden_path),
                "--worker-page-id",
                golden.page_id,
                "--work-dir",
                str(page_work_dir),
                "--output-json",
                str(output_json),
                "--json",
                "--in-process",
                "--engine",
                primary_engine,
            ]
            if getattr(args, "include_google_vision", False):
                cmd.append("--include-google-vision")
            completed = _run_worker_command(cmd, args)
            if completed.returncode != 0 or not output_json.exists():
                results.append(_failed_page_result(golden, completed))
                continue
            data = json.loads(output_json.read_text(encoding="utf-8"))
            page_result = PageBenchmark(**data[0])
            if _should_run_vl(args) and vl_pages_run < args.vl_limit and primary_engine != PADDLEOCR_VL_ENGINE:
                vl_pages_run += 1
                page_result = _with_vl_worker_result(args, golden_path, golden, root_work_dir, page_result)
            results.append(page_result)
    finally:
        if not keep_work_dir:
            shutil.rmtree(root_work_dir, ignore_errors=True)
    return results


def _with_vl_worker_result(
    args: argparse.Namespace,
    golden_path: Path,
    golden: GoldenPage,
    root_work_dir: Path,
    base_result: PageBenchmark,
) -> PageBenchmark:
    vl_work_dir = root_work_dir / f"{golden.page_id}-vl"
    vl_output_json = root_work_dir / f"{golden.page_id}.vl.json"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--golden",
        str(golden_path),
        "--worker-page-id",
        golden.page_id,
        "--work-dir",
        str(vl_work_dir),
        "--output-json",
        str(vl_output_json),
        "--json",
        "--in-process",
        "--engine",
        PADDLEOCR_VL_ENGINE,
    ]
    completed = _run_worker_command(cmd, args)
    if completed.returncode != 0 or not vl_output_json.exists():
        error = _failed_page_result(golden, completed).errors[0]
        vl_payload = {
            "mode": "paddleocr_vl_extraction",
            "actual_page_type": "worker_failed",
            "matched": 0,
            "expected": _expected_item_count(golden),
            "accuracy": 0.0,
            "generated_cards": 0,
            "missing_ids": [],
            "warnings": [error],
        }
        return PageBenchmark(
            page_id=base_result.page_id,
            image_path=base_result.image_path,
            base=base_result.base,
            vl=vl_payload,
            google_vision=base_result.google_vision,
            memory_samples=base_result.memory_samples,
            resource_metrics=base_result.resource_metrics,
            errors=base_result.errors,
        )
    data = json.loads(vl_output_json.read_text(encoding="utf-8"))
    vl_result = PageBenchmark(**data[0])
    vl_payload = {
        **vl_result.base,
        "resource_metrics": vl_result.resource_metrics,
        "memory_samples": vl_result.memory_samples,
    }
    return PageBenchmark(
        page_id=base_result.page_id,
        image_path=base_result.image_path,
        base=base_result.base,
        vl=vl_payload,
        google_vision=base_result.google_vision,
        memory_samples=base_result.memory_samples,
        resource_metrics=base_result.resource_metrics,
        errors=base_result.errors,
    )


def _run_worker_command(cmd: list[str], args: argparse.Namespace) -> subprocess.CompletedProcess[str]:
    timeout_seconds = float(getattr(args, "worker_timeout_seconds", 300))
    max_rss_mb = float(getattr(args, "worker_max_rss_mb", 8192))
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    start = time.monotonic()
    failure_reason = ""
    while process.poll() is None:
        elapsed = time.monotonic() - start
        rss_mb = _process_tree_rss_mb(process.pid)
        if timeout_seconds > 0 and elapsed > timeout_seconds:
            failure_reason = f"Worker exceeded timeout of {timeout_seconds:.0f}s."
            _terminate_process(process)
            break
        if max_rss_mb > 0 and rss_mb is not None and rss_mb > max_rss_mb:
            failure_reason = f"Worker exceeded RSS limit of {max_rss_mb:.0f} MB (observed {rss_mb:.0f} MB)."
            _terminate_process(process)
            break
        time.sleep(1)
    try:
        stdout, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
        stdout, stderr = process.communicate(timeout=5)
    if failure_reason:
        stderr = "\n".join(part for part in (stderr, failure_reason) if part)
        return subprocess.CompletedProcess(cmd, process.returncode or 137, stdout, stderr)
    return subprocess.CompletedProcess(cmd, process.returncode or 0, stdout, stderr)


def _terminate_process(process: subprocess.Popen[str]) -> None:
    _signal_process_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
        process.wait(timeout=5)


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    _signal_process_group(process, signal.SIGKILL)


def _signal_process_group(process: subprocess.Popen[str], sig: int) -> None:
    try:
        os.killpg(os.getpgid(process.pid), sig)
    except ProcessLookupError:
        return
    except OSError:
        if sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()


def _process_tree_rss_mb(pid: int) -> float | None:
    pids = [pid, *_child_pids(pid)]
    total = 0
    found = False
    for item in pids:
        try:
            output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(item)], text=True).strip()
        except Exception:
            continue
        if not output:
            continue
        found = True
        total += int(output)
    return round(total / 1024.0, 2) if found else None


def _child_pids(pid: int) -> list[int]:
    try:
        output = subprocess.check_output(["pgrep", "-P", str(pid)], text=True).strip()
    except Exception:
        return []
    children = [int(value) for value in output.splitlines() if value.strip().isdigit()]
    descendants: list[int] = []
    for child in children:
        descendants.append(child)
        descendants.extend(_child_pids(child))
    return descendants


def _run_worker_page(args: argparse.Namespace, pages: list[GoldenPage]) -> PageBenchmark:
    selected = next((page for page in pages if page.page_id == args.worker_page_id), None)
    if not selected:
        raise SystemExit(f"Unknown worker page id: {args.worker_page_id}")
    with _benchmark_runtime(args.work_dir, args.keep_work_dir):
        database.init_db()
        run_start = _resource_snapshot()
        memory_samples = [_memory_sample("worker_start")]
        primary_engine = _primary_engine(args)
        process_result = _run_base_pipeline(selected, memory_samples, primary_engine)
        base_eval = _evaluate_base(selected, process_result)
        base_payload = _result_dict(base_eval, primary_engine)
        base_payload["ocr_text_coverage"] = _coverage_dict(_token_text_coverage(selected, process_result, primary_engine))
        memory_samples.append(_memory_sample("base_evaluated"))
        vl_eval = None
        if _should_run_vl(args) and primary_engine != PADDLEOCR_VL_ENGINE:
            memory_samples.append(_memory_sample("before_vl"))
            vl_eval = _run_engine_evaluation(selected, memory_samples, PADDLEOCR_VL_ENGINE)
            memory_samples.append(_memory_sample("after_vl"))
        google_eval = (
            _run_google_vision_evaluation(selected, process_result, memory_samples)
            if getattr(args, "include_google_vision", False)
            else None
        )
        return PageBenchmark(
            page_id=selected.page_id,
            image_path=str(selected.image_path),
            base=base_payload,
            vl=vl_eval,
            google_vision=google_eval,
            memory_samples=memory_samples,
            resource_metrics=_resource_metrics(run_start, _resource_snapshot(), memory_samples),
            errors=[],
        )


def _emit_results(results: list[PageBenchmark], args: argparse.Namespace) -> None:
    payload = [asdict(result) for result in results]
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    for result in results:
        print(_format_page(result))


@contextmanager
def _benchmark_runtime(work_dir_arg: str, keep_work_dir: bool):
    previous_db_path = database.DB_PATH
    previous_processed_dir = pipeline.PROCESSED_DIR
    if work_dir_arg:
        work_dir = Path(work_dir_arg).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        should_cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="anki-maker-ocr-bench-"))
        should_cleanup = not keep_work_dir
    processed_dir = work_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    database.DB_PATH = work_dir / "benchmark.db"
    pipeline.PROCESSED_DIR = processed_dir
    try:
        yield work_dir
    finally:
        database.DB_PATH = previous_db_path
        pipeline.PROCESSED_DIR = previous_processed_dir
        if should_cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)


def _primary_engine(args: argparse.Namespace) -> str:
    engine = getattr(args, "engine", PADDLEOCR_ENGINE)
    return PADDLEOCR_ENGINE if engine == "all" else normalize_ocr_engine(engine)


def _should_run_vl(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "include_vl", False) or getattr(args, "engine", PADDLEOCR_ENGINE) == "all")


def _run_base_pipeline(golden: GoldenPage, memory_samples: list[dict[str, Any]], engine: str = PADDLEOCR_ENGINE) -> ProcessResult:
    page = Page(
        id=new_id("bench"),
        original_image_path=str(golden.image_path),
        upload_name=golden.image_path.name,
        display_name=golden.image_path.stem,
        processed_image_path=None,
        page_type="uploaded",
        page_type_confidence=0.0,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    database.upsert_page(page)
    memory_samples.append(_memory_sample(f"before_{engine}_pipeline"))
    result = pipeline.process_page(page, engine=engine)
    memory_samples.append(_memory_sample(f"after_{engine}_pipeline"))
    return result


def _evaluate_base(golden: GoldenPage, process_result: ProcessResult) -> VocabEvalResult | McqEvalResult:
    if golden.expected_rows:
        return evaluate_vocab_page(golden, process_result)
    return evaluate_mcq_page(golden, process_result)


def _run_engine_evaluation(golden: GoldenPage, memory_samples: list[dict[str, Any]], engine: str) -> dict[str, Any]:
    try:
        result = _run_base_pipeline(golden, memory_samples, engine)
        payload = _result_dict(_evaluate_base(golden, result), engine)
        payload["ocr_text_coverage"] = _coverage_dict(_token_text_coverage(golden, result, engine))
        return payload
    except Exception as exc:
        return {
            "mode": f"{engine}_extraction",
            "actual_page_type": "engine_failed",
            "matched": 0,
            "expected": _expected_item_count(golden),
            "accuracy": 0.0,
            "generated_cards": 0,
            "missing_ids": [],
            "warnings": [str(exc)],
            "ocr_text_coverage": None,
        }


def _run_google_vision_evaluation(
    golden: GoldenPage,
    process_result: ProcessResult,
    memory_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    run_start = _resource_snapshot()
    local_samples = [_memory_sample("before_google_vision")]
    memory_samples.extend(local_samples)
    image_path = Path(process_result.page.processed_image_path or golden.image_path)
    tokens, warnings = recognize_with_provider(image_path, process_result.page.id, "google_vision")
    google_result = ProcessResult(page=process_result.page, tokens=tokens, cards=[], script_summary={}, answer_map={})
    coverage = _token_text_coverage(golden, google_result, "google_vision")
    local_samples.append(_memory_sample("after_google_vision"))
    memory_samples.append(local_samples[-1])
    return {
        "mode": "google_vision_ocr_text",
        "token_count": len(tokens),
        "ocr_text_coverage": _coverage_dict(coverage),
        "warnings": warnings,
        "resource_metrics": _resource_metrics(run_start, _resource_snapshot(), local_samples),
        "memory_samples": local_samples,
    }


def _token_text_coverage(golden: GoldenPage, process_result: ProcessResult, engine: str) -> TextCoverageResult:
    if engine == PADDLEOCR_VL_ENGINE and process_result.document_parse:
        document_parse = process_result.document_parse
        text = "\n".join([document_parse.markdown_text, *[block.content for block in document_parse.blocks]])
        return _text_coverage(
            golden,
            text,
            document_parse.warnings,
            mode=f"{engine}_document_text",
        )
    ordered_tokens = sorted(process_result.tokens, key=lambda token: (token.bbox[1], token.bbox[0], token.id))
    text = "\n".join(token.text for token in ordered_tokens)
    return _text_coverage(golden, text, [], mode=f"{engine}_normalized_token_text")


def _text_coverage(
    golden: GoldenPage,
    text: str,
    warnings: list[str],
    *,
    mode: str = "paddleocr_vl",
) -> TextCoverageResult:
    fields_expected = 0
    fields_matched = 0
    items_expected = 0
    items_fully_matched = 0
    if golden.expected_rows:
        for row in golden.expected_rows:
            checks = [
                _contains(text, row.surface),
                _contains(text, row.reading),
                meaning_matches(text, row.meaning_ko),
            ]
            fields_expected += len(checks)
            fields_matched += sum(1 for matched in checks if matched)
            items_expected += 1
            items_fully_matched += int(all(checks))
    for question in golden.expected_questions:
        checks = [
            _contains(text, question.sentence),
            _contains(text, question.target),
            _contains(text, question.correct_answer),
            *[_contains(text, choice) for choice in question.choices],
        ]
        fields_expected += len(checks)
        fields_matched += sum(1 for matched in checks if matched)
        items_expected += 1
        items_fully_matched += int(all(checks))
    return TextCoverageResult(
        mode=mode,
        page_id=golden.page_id,
        fields_matched=fields_matched,
        fields_expected=fields_expected,
        items_fully_matched=items_fully_matched,
        items_expected=items_expected,
        warnings=warnings,
    )


def _contains(text: str, expected: str) -> bool:
    expected_norm = normalize_text(expected)
    return bool(expected_norm and expected_norm in normalize_text(text))


def _expected_field_count(golden: GoldenPage) -> int:
    return len(golden.expected_rows) * 3 + sum(3 + len(question.choices) for question in golden.expected_questions)


def _expected_item_count(golden: GoldenPage) -> int:
    return len(golden.expected_rows) + len(golden.expected_questions)


def _result_dict(result: VocabEvalResult | McqEvalResult, engine: str = PADDLEOCR_ENGINE) -> dict[str, Any]:
    if isinstance(result, McqEvalResult):
        return {
            "mode": f"{engine}_extraction",
            "actual_page_type": result.actual_page_type,
            "matched": result.matched_questions,
            "expected": result.expected_questions,
            "accuracy": result.question_accuracy,
            "source_matched": result.source_matched_questions,
            "source_field_accuracy": result.source_field_accuracy,
            "sentence_matches": result.sentence_matches,
            "target_matches": result.target_matches,
            "choices_matches": result.choices_matches,
            "correct_answer_matches": result.correct_answer_matches,
            "correct_choice_matches": result.correct_choice_matches,
            "source_field_matches": result.source_field_matches,
            "source_field_expected": result.source_field_expected,
            "generated_cards": result.generated_cards,
            "missing_ids": result.missing_question_ids,
            "source_mismatch_ids": result.source_mismatch_question_ids,
        }
    return {
        "mode": f"{engine}_extraction",
        "actual_page_type": result.actual_page_type,
        "matched": result.matched_rows,
        "expected": result.expected_rows,
        "accuracy": result.row_accuracy,
        "ocr_supported_items": result.ocr_supported_items,
        "glossary_supported_items": result.glossary_supported_items,
        "surface_reading_matches": result.surface_reading_matches,
        "meaning_matches": result.meaning_matches,
        "generated_cards": result.generated_cards,
        "missing_ids": result.missing_row_ids,
    }


def _coverage_dict(result: TextCoverageResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "mode": result.mode,
        "fields_matched": result.fields_matched,
        "fields_expected": result.fields_expected,
        "field_accuracy": result.field_accuracy,
        "items_fully_matched": result.items_fully_matched,
        "items_expected": result.items_expected,
        "item_accuracy": result.item_accuracy,
        "warnings": result.warnings,
    }


def _memory_sample(stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "rss_mb": _rss_mb(),
        "peak_rss_mb": _peak_rss_mb(),
    }


def _resource_snapshot() -> dict[str, float]:
    user_cpu = system_cpu = 0.0
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        user_cpu = float(usage.ru_utime + child_usage.ru_utime)
        system_cpu = float(usage.ru_stime + child_usage.ru_stime)
    except Exception:
        pass
    return {
        "wall_seconds": time.perf_counter(),
        "user_cpu_seconds": user_cpu,
        "system_cpu_seconds": system_cpu,
    }


def _resource_metrics(
    start: dict[str, float],
    end: dict[str, float],
    memory_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    wall_seconds = max(0.0, end["wall_seconds"] - start["wall_seconds"])
    user_cpu_seconds = max(0.0, end["user_cpu_seconds"] - start["user_cpu_seconds"])
    system_cpu_seconds = max(0.0, end["system_cpu_seconds"] - start["system_cpu_seconds"])
    cpu_seconds = user_cpu_seconds + system_cpu_seconds
    peak_rss_mb = max((sample.get("peak_rss_mb") or 0 for sample in memory_samples), default=0)
    return {
        "wall_seconds": round(wall_seconds, 3),
        "user_cpu_seconds": round(user_cpu_seconds, 3),
        "system_cpu_seconds": round(system_cpu_seconds, 3),
        "cpu_seconds": round(cpu_seconds, 3),
        "cpu_percent_of_one_core": round((cpu_seconds / wall_seconds) * 100, 2) if wall_seconds else None,
        "peak_rss_mb": round(float(peak_rss_mb), 2),
        "rss_samples": memory_samples,
        "npu": {
            "available": False,
            "utilization_percent": None,
            "memory_mb": None,
            "note": "PaddleOCR local CPU/PaddlePaddle path does not expose NPU counters here.",
        },
        "gpu": {
            "available": False,
            "utilization_percent": None,
            "memory_mb": None,
            "note": "No GPU metrics collector is configured for this local benchmark.",
        },
    }


def _rss_mb() -> float | None:
    try:
        output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip()
        if not output:
            return None
        return round(int(output) / 1024.0, 2)
    except Exception:
        return None


def _peak_rss_mb() -> float | None:
    try:
        import resource

        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    # macOS reports bytes; Linux reports KiB.
    if peak > 1024 * 1024 * 10:
        return round(peak / (1024 * 1024), 2)
    return round(peak / 1024, 2)


def _failed_page_result(golden: GoldenPage, completed: subprocess.CompletedProcess[str]) -> PageBenchmark:
    message = (completed.stderr or completed.stdout or f"Worker exited with {completed.returncode}").strip()
    return PageBenchmark(
        page_id=golden.page_id,
        image_path=str(golden.image_path),
        base={
            "mode": "base_paddleocr_extraction",
            "actual_page_type": "worker_failed",
            "matched": 0,
            "expected": _expected_item_count(golden),
            "accuracy": 0.0,
            "generated_cards": 0,
            "missing_ids": [],
        },
        vl=None,
        google_vision=None,
        memory_samples=[],
        resource_metrics={
            "wall_seconds": 0.0,
            "user_cpu_seconds": 0.0,
            "system_cpu_seconds": 0.0,
            "cpu_seconds": 0.0,
            "cpu_percent_of_one_core": None,
            "peak_rss_mb": 0.0,
            "rss_samples": [],
            "npu": {"available": False, "utilization_percent": None, "memory_mb": None, "note": "Worker failed."},
            "gpu": {"available": False, "utilization_percent": None, "memory_mb": None, "note": "Worker failed."},
        },
        errors=[message[-2000:]],
    )


def _format_page(result: PageBenchmark) -> str:
    base = result.base
    lines = [
        f"Page: {result.page_id}",
        f"  {base.get('mode', 'base')}: {base['matched']}/{base['expected']} accuracy={base['accuracy']:.1%} cards={base['generated_cards']}",
    ]
    if "source_field_accuracy" in base:
        lines.append(
            "  source fields: "
            f"{base.get('source_field_matches', 0)}/{base.get('source_field_expected', 0)} "
            f"accuracy={base.get('source_field_accuracy', 0):.1%}"
        )
    if result.memory_samples:
        metrics = result.resource_metrics
        lines.append(
            "  resources: "
            f"wall={metrics.get('wall_seconds', 0):.3f}s "
            f"cpu={metrics.get('cpu_seconds', 0):.3f}s "
            f"cpu%={metrics.get('cpu_percent_of_one_core')} "
            f"peak_rss={metrics.get('peak_rss_mb', 0):.2f} MB "
            f"npu={metrics.get('npu', {}).get('note', 'not reported')}"
        )
    if result.vl:
        lines.append(
            "  paddleocr_vl_extraction: "
            f"{result.vl.get('matched', 0)}/{result.vl.get('expected', 0)} "
            f"accuracy={result.vl.get('accuracy', 0):.1%} cards={result.vl.get('generated_cards', 0)}"
        )
        if result.vl.get("warnings"):
            lines.append(f"  vl warnings: {'; '.join(result.vl['warnings'])}")
    if result.google_vision:
        coverage = result.google_vision.get("ocr_text_coverage") or {}
        lines.append(
            "  google_vision_ocr_text: "
            f"fields={coverage.get('fields_matched', 0)}/{coverage.get('fields_expected', 0)} "
            f"accuracy={coverage.get('field_accuracy', 0):.1%} "
            f"tokens={result.google_vision.get('token_count', 0)}"
        )
        if result.google_vision.get("warnings"):
            lines.append(f"  google warnings: {'; '.join(result.google_vision['warnings'])}")
    if result.errors:
        lines.append(f"  errors: {'; '.join(result.errors)}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
