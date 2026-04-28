from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from app.core.config import BACKEND_DIR, OCR_PAGE_JOB_TIMEOUT_SECONDS, OCR_PAGE_WORKER_MAX_RSS_MB
from app.db import database
from app.extraction.pipeline import process_page
from app.models.schemas import DocumentParseResult, ProcessResult
from app.ocr.engines import normalize_ocr_engine
from app.vision.paddle_ocr_vl import get_paddle_ocr_vl_parser


def run_page_process_worker(
    page_id: str,
    engine: str,
    *,
    timeout_seconds: float = OCR_PAGE_JOB_TIMEOUT_SECONDS,
    max_rss_mb: float = OCR_PAGE_WORKER_MAX_RSS_MB,
) -> ProcessResult:
    normalized_engine = normalize_ocr_engine(engine)
    output_fd, output_name = tempfile.mkstemp(prefix=f"anki-page-{page_id}-", suffix=".json")
    os.close(output_fd)
    output_path = Path(output_name)
    try:
        cmd = [
            sys.executable,
            "-m",
            "app.ocr.page_worker",
            "--page-id",
            page_id,
            "--engine",
            normalized_engine,
            "--output-json",
            str(output_path),
        ]
        completed = _run_worker_command(cmd, timeout_seconds=timeout_seconds, max_rss_mb=max_rss_mb)
        if completed.returncode != 0 or not output_path.exists():
            detail = completed.stderr.strip() or completed.stdout.strip() or f"Page OCR worker exited with {completed.returncode}."
            raise RuntimeError(detail)
        return ProcessResult.model_validate_json(output_path.read_text(encoding="utf-8"))
    finally:
        output_path.unlink(missing_ok=True)


def run_document_parse_worker(
    image_path: Path,
    page_id: str,
    *,
    timeout_seconds: float = OCR_PAGE_JOB_TIMEOUT_SECONDS,
    max_rss_mb: float = OCR_PAGE_WORKER_MAX_RSS_MB,
) -> DocumentParseResult:
    output_fd, output_name = tempfile.mkstemp(prefix=f"anki-document-{page_id}-", suffix=".json")
    os.close(output_fd)
    output_path = Path(output_name)
    try:
        cmd = [
            sys.executable,
            "-m",
            "app.ocr.page_worker",
            "--document-parse",
            "--page-id",
            page_id,
            "--image-path",
            str(image_path),
            "--output-json",
            str(output_path),
        ]
        completed = _run_worker_command(cmd, timeout_seconds=timeout_seconds, max_rss_mb=max_rss_mb)
        if completed.returncode != 0 or not output_path.exists():
            detail = completed.stderr.strip() or completed.stdout.strip() or f"Document OCR worker exited with {completed.returncode}."
            raise RuntimeError(detail)
        return DocumentParseResult.model_validate_json(output_path.read_text(encoding="utf-8"))
    finally:
        output_path.unlink(missing_ok=True)


def _run_worker_command(cmd: list[str], *, timeout_seconds: float, max_rss_mb: float) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(  # NOSONAR
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(BACKEND_DIR),
        start_new_session=True,
    )
    start = time.monotonic()
    failure_reason = ""
    while process.poll() is None:
        elapsed = time.monotonic() - start
        rss_mb = _process_tree_rss_mb(process.pid)
        if timeout_seconds > 0 and elapsed > timeout_seconds:
            failure_reason = f"Page OCR worker exceeded timeout of {timeout_seconds:.0f}s."
            _terminate_process(process)
            break
        if max_rss_mb > 0 and rss_mb is not None and rss_mb > max_rss_mb:
            failure_reason = f"Page OCR worker exceeded RSS limit of {max_rss_mb:.0f} MB (observed {rss_mb:.0f} MB)."
            _terminate_process(process)
            break
        time.sleep(0.5)
    try:
        stdout, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        _kill_process(process)
        stdout, stderr = process.communicate(timeout=5)
    if failure_reason:
        stderr = "\n".join(part for part in (stderr, failure_reason) if part)
        return subprocess.CompletedProcess(cmd, process.returncode or 137, stdout, stderr)
    return subprocess.CompletedProcess(cmd, process.returncode or 0, stdout, stderr)


def _terminate_process(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, 15)  # NOSONAR
    except Exception:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _kill_process(process)


def _kill_process(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, 9)  # NOSONAR
    except Exception:
        process.kill()
    process.wait(timeout=5)


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one page through a bounded OCR processing worker.")
    parser.add_argument("--page-id", required=True)
    parser.add_argument("--engine", default="paddleocr")
    parser.add_argument("--document-parse", action="store_true")
    parser.add_argument("--image-path")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    try:
        if args.document_parse:
            if not args.image_path:
                print("--image-path is required for document parsing.", file=sys.stderr)
                return 2
            result = get_paddle_ocr_vl_parser().parse(Path(args.image_path), args.page_id)
        else:
            database.init_db()
            page = database.get_page(args.page_id)
            if not page:
                print(f"Page {args.page_id!r} was not found.", file=sys.stderr)
                return 2
            result = process_page(page, engine=normalize_ocr_engine(args.engine))
        Path(args.output_json).write_text(json.dumps(result.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
