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
from PIL import Image, ImageDraw

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
from app.ocr.profiles import (
    BASELINE_MODEL_PROFILE,
    DEFAULT_KOREAN_PROFILE,
    DEFAULT_EXTRACTION_VARIANT,
    EXTRACTION_VARIANT_ORDER,
    LOCAL_MODEL_PROFILES,
    MODEL_PAIR_PROFILES,
    cache_korean_profile_id,
    cache_model_profile_id,
    extraction_variant_components,
    normalize_korean_profile,
    normalize_extraction_variant,
    profile_env_overrides,
    resolve_korean_ocr_profile,
    resolve_ocr_model_profile,
)
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
    audit_artifacts: dict[str, str] | None = None
    schema_version: int = 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare PaddleOCR and PaddleOCR-VL extraction accuracy and resource usage.")
    parser.add_argument("--golden", default="../data/evaluation/golden_pages.example.json")
    parser.add_argument("--engine", default=PADDLEOCR_ENGINE, help="Primary extraction engine: paddleocr, paddleocr_vl, or all.")
    parser.add_argument("--model-profile", default=BASELINE_MODEL_PROFILE, help="OCR model profile to benchmark.")
    parser.add_argument("--korean-profile", default=DEFAULT_KOREAN_PROFILE, help="Korean OCR sub-profile for the two-pass vocab pipeline.")
    parser.add_argument("--extraction-variant", default=DEFAULT_EXTRACTION_VARIANT, help="Extraction variant to benchmark.")
    parser.add_argument("--profile-matrix", action="store_true", help="Run baseline extraction against every local candidate model profile.")
    parser.add_argument("--variant-matrix", action="store_true", help="Run every registered extraction variant for the selected profile set.")
    parser.add_argument(
        "--experiment-stage",
        choices=["0", "1", "2", "3", "4", "5"],
        default="",
        help="Run the staged accuracy protocol: 0=raw OCR coverage, 1=model pairs, 2=atomic variants, 3=combined variants, 4=parity, 5=heavy/optional diagnostics.",
    )
    parser.add_argument(
        "--stage-profiles",
        default="",
        help="Comma-separated model profiles for staged benchmarks. Defaults follow the staged accuracy matrix.",
    )
    parser.add_argument(
        "--stage-variants",
        default="",
        help="Comma-separated extraction variants for staged/parity runs. Supports '+' aliases such as 'v5_token_split_v1 + v5_vocab_rows_v1'.",
    )
    parser.add_argument("--include-heavy-profiles", action="store_true", help="Include heavy local server models in --profile-matrix.")
    parser.add_argument("--benchmark-mode-matrix", action="store_true", help="Run fresh_cli, persisted_db, and ui_api for each selected profile/variant.")
    parser.add_argument(
        "--benchmark-mode",
        choices=["fresh_cli", "persisted_db", "ui_api"],
        default="fresh_cli",
        help="Label how results were produced so CLI/API/DB parity runs are comparable.",
    )
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
    parser.add_argument("--dashboard-markdown", default="", help="Optional path for a compact Markdown benchmark comparison table.")
    parser.add_argument("--miss-inventory-json", default="", help="Optional path for a machine-readable miss inventory generated from the run.")
    parser.add_argument("--focus-misses-from", default="", help="Optional miss inventory/benchmark JSON used only for diagnostic focus metadata; full pages are still scored.")
    parser.add_argument("--residual-diagnostics-dir", default="", help="Optional directory for diagnostic-only residual miss JSON and contact-sheet artifacts.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args()
    _apply_experiment_stage(args)
    args.model_profile = resolve_ocr_model_profile(args.model_profile).id
    args.korean_profile = resolve_korean_ocr_profile(args.korean_profile).id
    args.extraction_variant = normalize_extraction_variant(args.extraction_variant)
    if not resolve_ocr_model_profile(args.model_profile).creates_candidates and args.engine != "all":
        print(f"Model profile {args.model_profile!r} is diagnostic-only; use dedicated comparison flags instead.", file=sys.stderr)
        return 2
    if (
        args.in_process
        and (args.model_profile != BASELINE_MODEL_PROFILE or args.korean_profile != DEFAULT_KOREAN_PROFILE)
        and not _profile_env_is_active(args.model_profile, args.korean_profile)
    ):
        print(
            "Non-default OCR model profiles must be run through the subprocess benchmark wrapper so PaddleOCR imports fresh profile env.",
            file=sys.stderr,
        )
        return 2

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
            process_result = _run_benchmark_pipeline(golden, memory_samples, primary_engine, args)
            base_payload = _evaluation_payload(golden, process_result, primary_engine)
            base_payload["benchmark"] = _benchmark_manifest(args, process_result)
            base_payload["ocr_text_coverage"] = _coverage_dict(_token_text_coverage(golden, process_result, primary_engine))
            base_payload["raw_field_coverage"] = _raw_field_coverage(golden, process_result)
            memory_samples.append(_memory_sample("base_evaluated"))
            audit_artifacts = _write_audit_artifacts(golden, process_result, args, base_payload)
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
                    resource_metrics=_resource_metrics_with_cache(run_start, _resource_snapshot(), memory_samples, base_payload),
                    errors=[],
                    audit_artifacts=audit_artifacts,
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
        for profile_id in _profile_ids_for_run(args):
            for variant_id in _variant_ids_for_run(args):
                for benchmark_mode in _benchmark_modes_for_run(args):
                    run_args = argparse.Namespace(
                        **{
                            **vars(args),
                            "model_profile": profile_id,
                            "extraction_variant": variant_id,
                            "benchmark_mode": benchmark_mode,
                        }
                    )
                    for golden in pages:
                        page_result = _run_single_page_subprocess(run_args, golden_path, golden, root_work_dir)
                        if _should_run_vl(run_args) and vl_pages_run < run_args.vl_limit and _primary_engine(run_args) != PADDLEOCR_VL_ENGINE:
                            vl_pages_run += 1
                            page_result = _with_vl_worker_result(run_args, golden_path, golden, root_work_dir, page_result)
                        results.append(page_result)
    finally:
        if not keep_work_dir:
            shutil.rmtree(root_work_dir, ignore_errors=True)
    return results


def _run_single_page_subprocess(
    args: argparse.Namespace,
    golden_path: Path,
    golden: GoldenPage,
    root_work_dir: Path,
) -> PageBenchmark:
    model_profile = getattr(args, "model_profile", BASELINE_MODEL_PROFILE)
    korean_profile = getattr(args, "korean_profile", DEFAULT_KOREAN_PROFILE)
    extraction_variant = getattr(args, "extraction_variant", DEFAULT_EXTRACTION_VARIANT)
    benchmark_mode = getattr(args, "benchmark_mode", "fresh_cli")
    page_work_dir = _page_work_dir_for_run(root_work_dir, golden, model_profile, args)
    output_json = root_work_dir / f"{golden.page_id}.json"
    if _uses_matrix_output_names(args):
        stage = getattr(args, "experiment_stage", "") or "run"
        output_json = root_work_dir / f"{stage}.{golden.page_id}.{model_profile}.{korean_profile}.{extraction_variant}.{benchmark_mode}.json"
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
        "--model-profile",
        model_profile,
        "--korean-profile",
        korean_profile,
        "--extraction-variant",
        extraction_variant,
        "--benchmark-mode",
        benchmark_mode,
    ]
    if getattr(args, "include_google_vision", False):
        cmd.append("--include-google-vision")
    completed = _run_worker_command(cmd, args)
    if completed.returncode != 0 or not output_json.exists():
        return _failed_page_result(golden, completed)
    data = json.loads(output_json.read_text(encoding="utf-8"))
    return PageBenchmark(**data[0])


def _with_vl_worker_result(
    args: argparse.Namespace,
    golden_path: Path,
    golden: GoldenPage,
    root_work_dir: Path,
    base_result: PageBenchmark,
) -> PageBenchmark:
    extraction_variant = getattr(args, "extraction_variant", DEFAULT_EXTRACTION_VARIANT)
    korean_profile = getattr(args, "korean_profile", DEFAULT_KOREAN_PROFILE)
    benchmark_mode = getattr(args, "benchmark_mode", "fresh_cli")
    vl_work_dir = root_work_dir / f"{golden.page_id}.{korean_profile}-vl"
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
        "--model-profile",
        BASELINE_MODEL_PROFILE,
        "--korean-profile",
        korean_profile,
        "--extraction-variant",
        extraction_variant,
        "--benchmark-mode",
        benchmark_mode,
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
    env = os.environ.copy()
    env.update(profile_env_overrides(getattr(args, "model_profile", BASELINE_MODEL_PROFILE), getattr(args, "korean_profile", DEFAULT_KOREAN_PROFILE)))
    if 0 < max_rss_mb <= 3200:
        env.setdefault("OCR_RECOVERY_REGION_CACHE_ONLY", "true")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True, env=env)
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
        process_result = _run_benchmark_pipeline(selected, memory_samples, primary_engine, args)
        base_payload = _evaluation_payload(selected, process_result, primary_engine)
        base_payload["benchmark"] = _benchmark_manifest(args, process_result)
        base_payload["ocr_text_coverage"] = _coverage_dict(_token_text_coverage(selected, process_result, primary_engine))
        base_payload["raw_field_coverage"] = _raw_field_coverage(selected, process_result)
        memory_samples.append(_memory_sample("base_evaluated"))
        audit_artifacts = _write_audit_artifacts(selected, process_result, args, base_payload)
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
            resource_metrics=_resource_metrics_with_cache(run_start, _resource_snapshot(), memory_samples, base_payload),
            errors=[],
            audit_artifacts=audit_artifacts,
        )


def _emit_results(results: list[PageBenchmark], args: argparse.Namespace) -> None:
    payload = [asdict(result) for result in results]
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if getattr(args, "miss_inventory_json", ""):
        _write_miss_inventory(results, Path(args.miss_inventory_json), focus_source=getattr(args, "focus_misses_from", ""))
    if getattr(args, "residual_diagnostics_dir", ""):
        _write_residual_diagnostics(results, Path(args.residual_diagnostics_dir), focus_source=getattr(args, "focus_misses_from", ""))
    if getattr(args, "dashboard_markdown", ""):
        _write_dashboard_markdown(results, Path(args.dashboard_markdown))
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    for result in results:
        print(_format_page(result))


def _write_miss_inventory(results: list[PageBenchmark], path: Path, *, focus_source: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    inventory = _miss_inventory_payload(results, focus_source=focus_source)
    path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_residual_diagnostics(results: list[PageBenchmark], path: Path, *, focus_source: str = "") -> None:
    path.mkdir(parents=True, exist_ok=True)
    crop_dir = path / "residual-crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    inventory = _miss_inventory_payload(results, focus_source=focus_source)
    entries: list[dict[str, Any]] = []
    overlays = _overlay_payloads_by_page(results)
    resource_metrics_by_page = {result.page_id: result.resource_metrics for result in results}
    for entry in inventory["entries"]:
        if not isinstance(entry, dict):
            continue
        page_id = str(entry.get("page_id") or "")
        overlay = overlays.get(page_id, {})
        candidate_evidence = _diagnostic_candidate_evidence(entry, overlay)
        recovery_attempts = _diagnostic_recovery_attempts(entry, overlay)
        enriched = {
            **entry,
            "miss_kind": entry.get("kind"),
            "expected_value": entry.get("expected", {}),
            "actual_value": entry.get("current_candidate"),
            "diagnostic_only": True,
            "oracle_use_allowed": False,
            "overlay_json": overlay.get("overlay_json"),
            "processed_image_path": overlay.get("processed_image_path"),
            "candidate_evidence": candidate_evidence,
            "field_evidence": _diagnostic_field_evidence(candidate_evidence, entry),
            "token_ids": _diagnostic_token_ids(candidate_evidence, entry),
            "crop_bbox": _diagnostic_crop_bbox(candidate_evidence, recovery_attempts, entry),
            "region_strategy": _diagnostic_region_strategy(candidate_evidence, recovery_attempts),
            "ocr_candidates": _diagnostic_ocr_candidates(recovery_attempts),
            "rejected_candidates": _diagnostic_rejected_candidates(recovery_attempts),
            "rejection_reasons": _diagnostic_rejection_reasons(entry, recovery_attempts),
            "cache": _diagnostic_cache_summary(recovery_attempts),
            "confidence": _diagnostic_confidence(candidate_evidence, recovery_attempts),
            "resource_metrics": resource_metrics_by_page.get(page_id, {}),
            "recovery_attempts": recovery_attempts,
        }
        entries.append(enriched)
    payload = {
        "schema_version": 1,
        "source": focus_source or "benchmark_results",
        "diagnostic_only": True,
        "oracle_use_allowed": False,
        "entry_count": len(entries),
        "counts": inventory.get("counts", {}),
        "entries": entries,
    }
    (path / "residual-diagnostics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for result in results:
        _write_residual_contact_sheet(path / f"{result.page_id}.residuals.png", result, [entry for entry in entries if entry.get("page_id") == result.page_id])
    (path / "README.md").write_text(
        "# Residual Diagnostics\n\nThese artifacts are diagnostic-only benchmark outputs. They are not extraction inputs and must not be used as an oracle.\n",
        encoding="utf-8",
    )


def _overlay_payloads_by_page(results: list[PageBenchmark]) -> dict[str, dict[str, Any]]:
    overlays: dict[str, dict[str, Any]] = {}
    for result in results:
        artifacts = result.audit_artifacts or {}
        overlay_path = artifacts.get("overlay_json")
        overlay: dict[str, Any] = {}
        if overlay_path and Path(overlay_path).exists():
            try:
                overlay = json.loads(Path(overlay_path).read_text(encoding="utf-8"))
            except Exception:
                overlay = {}
        overlay["overlay_json"] = overlay_path
        overlays[result.page_id] = overlay
    return overlays


def _diagnostic_candidate_evidence(entry: dict[str, Any], overlay: dict[str, Any]) -> list[dict[str, Any]]:
    current = entry.get("current_candidate") if isinstance(entry.get("current_candidate"), dict) else {}
    cards = overlay.get("cards") if isinstance(overlay.get("cards"), list) else []
    source_id = entry.get("row_id") or entry.get("question_id")
    matches = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        if source_id and card.get("source_id") != source_id:
            continue
        matches.append(
            {
                "card_id": card.get("id"),
                "source_id": card.get("source_id"),
                "source_bbox": card.get("source_bbox"),
                "field_evidence": card.get("field_evidence"),
                "confidence": card.get("confidence"),
                "review_state": card.get("review_state"),
                "warnings": card.get("warnings"),
            }
        )
    return matches or ([{"current_candidate": current}] if current else [])


def _diagnostic_recovery_attempts(entry: dict[str, Any], overlay: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = overlay.get("extraction_variant_metrics") if isinstance(overlay.get("extraction_variant_metrics"), dict) else {}
    recovery = metrics.get("recovery") if isinstance(metrics.get("recovery"), dict) else {}
    attempts: list[dict[str, Any]] = []
    for payload in _iter_recovery_payloads(recovery):
        for attempt in payload.get("attempts", []) if isinstance(payload.get("attempts"), list) else []:
            if not isinstance(attempt, dict):
                continue
            if _attempt_matches_inventory_entry(attempt, entry):
                attempts.append(attempt)
    return attempts


def _diagnostic_field_evidence(candidate_evidence: list[dict[str, Any]], entry: dict[str, Any]) -> dict[str, Any]:
    failed_fields = {str(field) for field in entry.get("failed_fields", []) if isinstance(field, str)}
    for candidate in candidate_evidence:
        evidence = candidate.get("field_evidence") if isinstance(candidate, dict) else None
        if not isinstance(evidence, dict):
            continue
        if not failed_fields:
            return evidence
        narrowed = {field: evidence[field] for field in failed_fields if field in evidence}
        if narrowed:
            return narrowed
    return {}


def _diagnostic_token_ids(candidate_evidence: list[dict[str, Any]], entry: dict[str, Any]) -> list[str]:
    token_ids: list[str] = []
    for field_evidence in _diagnostic_field_evidence(candidate_evidence, entry).values():
        if not isinstance(field_evidence, dict):
            continue
        for token_id in field_evidence.get("token_ids", []) or field_evidence.get("derived_from_token_ids", []) or []:
            if isinstance(token_id, str) and token_id not in token_ids:
                token_ids.append(token_id)
    return token_ids


def _diagnostic_crop_bbox(
    candidate_evidence: list[dict[str, Any]],
    recovery_attempts: list[dict[str, Any]],
    entry: dict[str, Any],
) -> list[float] | None:
    for attempt in recovery_attempts:
        bbox = _diagnostic_bbox_value(attempt.get("bbox") or attempt.get("crop_bbox"))
        if bbox is not None:
            return bbox
    for field_evidence in _diagnostic_field_evidence(candidate_evidence, entry).values():
        if not isinstance(field_evidence, dict):
            continue
        bbox = _diagnostic_bbox_value(field_evidence.get("bbox") or field_evidence.get("crop_bbox"))
        if bbox is not None:
            return bbox
    for candidate in candidate_evidence:
        if isinstance(candidate, dict):
            bbox = _diagnostic_bbox_value(candidate.get("source_bbox"))
            if bbox is not None:
                return bbox
    return None


def _diagnostic_bbox_value(value: object) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    bbox: list[float] = []
    for coordinate in value:
        if not isinstance(coordinate, (int, float)):
            return None
        bbox.append(float(coordinate))
    return bbox


def _diagnostic_region_strategy(
    candidate_evidence: list[dict[str, Any]],
    recovery_attempts: list[dict[str, Any]],
) -> str | None:
    for attempt in recovery_attempts:
        strategy = attempt.get("strategy") or attempt.get("region_strategy")
        if isinstance(strategy, str) and strategy:
            return strategy
    for candidate in candidate_evidence:
        evidence = candidate.get("field_evidence") if isinstance(candidate, dict) else None
        if not isinstance(evidence, dict):
            continue
        for field_evidence in evidence.values():
            if not isinstance(field_evidence, dict):
                continue
            strategy = field_evidence.get("region_strategy") or field_evidence.get("normalization_strategy") or field_evidence.get("bbox_strategy")
            if isinstance(strategy, str) and strategy:
                return strategy
    return None


def _diagnostic_ocr_candidates(recovery_attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for attempt in recovery_attempts:
        nested = attempt.get("candidates") if isinstance(attempt.get("candidates"), list) else []
        for candidate in nested:
            if isinstance(candidate, dict):
                candidates.append(candidate)
        text = attempt.get("text") or attempt.get("candidate")
        if isinstance(text, str) and text and not any(candidate.get("text") == text for candidate in candidates if isinstance(candidate, dict)):
            candidates.append(
                {
                    "text": text,
                    "confidence": attempt.get("confidence"),
                    "strategy": attempt.get("strategy"),
                    "accepted": attempt.get("accepted"),
                    "cache": attempt.get("cache"),
                }
            )
    return candidates


def _diagnostic_rejected_candidates(recovery_attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rejected: list[dict[str, Any]] = []
    for candidate in _diagnostic_ocr_candidates(recovery_attempts):
        if candidate.get("accepted") is not True:
            rejected.append(candidate)
    for attempt in recovery_attempts:
        if attempt.get("accepted") is False:
            rejected.append(
                {
                    "text": attempt.get("text") or attempt.get("candidate"),
                    "confidence": attempt.get("confidence"),
                    "strategy": attempt.get("strategy"),
                    "reason": attempt.get("reason") or attempt.get("bucket") or attempt.get("diagnostic_bucket"),
                }
            )
    return rejected


def _diagnostic_rejection_reasons(entry: dict[str, Any], recovery_attempts: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for value in (entry.get("reason"), entry.get("diagnostic_bucket")):
        if isinstance(value, str) and value and value not in reasons:
            reasons.append(value)
    for attempt in recovery_attempts:
        for key in ("reason", "bucket", "diagnostic_bucket", "rejection_reason"):
            value = attempt.get(key)
            if isinstance(value, str) and value and value not in reasons:
                reasons.append(value)
        warnings = attempt.get("warnings") if isinstance(attempt.get("warnings"), list) else []
        for warning in warnings:
            if isinstance(warning, str) and warning and warning not in reasons:
                reasons.append(warning)
        for candidate in attempt.get("candidates", []) if isinstance(attempt.get("candidates"), list) else []:
            if not isinstance(candidate, dict):
                continue
            value = candidate.get("reason") or candidate.get("rejection_reason")
            if isinstance(value, str) and value and value not in reasons:
                reasons.append(value)
    return reasons


def _diagnostic_cache_summary(recovery_attempts: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"hits": 0, "misses": 0, "entries": []}
    for attempt in recovery_attempts:
        _add_diagnostic_cache_entry(summary, attempt)
        for candidate in attempt.get("candidates", []) if isinstance(attempt.get("candidates"), list) else []:
            if isinstance(candidate, dict):
                _add_diagnostic_cache_entry(summary, candidate)
    return summary


def _add_diagnostic_cache_entry(summary: dict[str, Any], payload: dict[str, Any]) -> None:
    cache = payload.get("cache") if isinstance(payload.get("cache"), dict) else {}
    if not cache:
        return
    entries = summary.setdefault("entries", [])
    if isinstance(entries, list):
        entries.append(cache)
    if cache.get("hit") is True:
        summary["hits"] = int(summary.get("hits", 0)) + 1
    elif cache.get("hit") is False:
        summary["misses"] = int(summary.get("misses", 0)) + 1


def _diagnostic_confidence(
    candidate_evidence: list[dict[str, Any]],
    recovery_attempts: list[dict[str, Any]],
) -> float | None:
    values: list[float] = []
    for candidate in candidate_evidence:
        confidence = candidate.get("confidence") if isinstance(candidate, dict) else None
        if isinstance(confidence, (int, float)):
            values.append(float(confidence))
        evidence = candidate.get("field_evidence") if isinstance(candidate, dict) else None
        if isinstance(evidence, dict):
            for field_evidence in evidence.values():
                if isinstance(field_evidence, dict) and isinstance(field_evidence.get("confidence"), (int, float)):
                    values.append(float(field_evidence["confidence"]))
    for attempt in recovery_attempts:
        if isinstance(attempt.get("confidence"), (int, float)):
            values.append(float(attempt["confidence"]))
        for candidate in attempt.get("candidates", []) if isinstance(attempt.get("candidates"), list) else []:
            if isinstance(candidate, dict) and isinstance(candidate.get("confidence"), (int, float)):
                values.append(float(candidate["confidence"]))
    return round(max(values), 4) if values else None


def _iter_recovery_payloads(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not payload:
        return []
    values = [payload]
    components = payload.get("components") if isinstance(payload.get("components"), dict) else {}
    for value in components.values():
        if isinstance(value, dict):
            values.extend(_iter_recovery_payloads(value))
    return values


def _attempt_matches_inventory_entry(attempt: dict[str, Any], entry: dict[str, Any]) -> bool:
    row_id = entry.get("row_id")
    question_no = entry.get("question_no")
    question_id = entry.get("question_id")
    return bool(
        (row_id and attempt.get("source_id") == row_id)
        or (question_id and attempt.get("source_id") == question_id)
        or (isinstance(question_no, int) and attempt.get("question_no") == question_no)
    )


def _write_residual_contact_sheet(path: Path, result: PageBenchmark, entries: list[dict[str, Any]]) -> None:
    width = 1200
    row_height = 92
    height = max(160, 60 + row_height * max(1, len(entries)))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((12, 12), f"{result.page_id} residual diagnostics", fill=(20, 20, 20))
    if not entries:
        draw.text((12, 48), "No residual entries.", fill=(90, 90, 90))
    for index, entry in enumerate(entries):
        y = 48 + index * row_height
        failed = ",".join(str(field) for field in entry.get("failed_fields", []))
        expected = json.dumps(entry.get("expected", {}), ensure_ascii=False)
        current = json.dumps(entry.get("current_candidate", {}), ensure_ascii=False)
        draw.rectangle([8, y - 4, width - 8, y + row_height - 10], outline=(220, 220, 220))
        draw.text((18, y), f"{entry.get('kind')} {entry.get('row_id') or entry.get('question_id') or entry.get('question_no')} {failed}", fill=(20, 20, 20))
        draw.text((18, y + 24), f"expected: {expected[:150]}", fill=(20, 90, 20))
        draw.text((18, y + 48), f"current: {current[:150]}", fill=(140, 50, 20))
    image.save(path)


def _miss_inventory_payload(results: list[PageBenchmark], *, focus_source: str = "") -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    field_error_total = 0
    for result in results:
        base = result.base
        benchmark = base.get("benchmark") if isinstance(base.get("benchmark"), dict) else {}
        miss_analysis = base.get("miss_analysis") if isinstance(base.get("miss_analysis"), dict) else {}
        kind = str(miss_analysis.get("kind") or "")
        for row in miss_analysis.get("rows", []):
            if not isinstance(row, dict):
                continue
            if kind == "vocab":
                entries.append(
                    {
                        "page_id": result.page_id,
                        "kind": "vocab",
                        "row_id": row.get("row_id"),
                        "reason": row.get("reason"),
                        "failed_fields": _failed_vocab_fields(row),
                        "raw_presence": row.get("raw_presence", {}),
                        "expected": row.get("expected", {}),
                        "current_candidate": row.get("best_candidate"),
                        "benchmark": _inventory_benchmark_key(benchmark),
                    }
                )
            elif kind == "mcq":
                fields = [str(field) for field in row.get("field_errors", []) if isinstance(field, str)]
                field_error_total += len(fields)
                entries.append(
                    {
                        "page_id": result.page_id,
                        "kind": "mcq",
                        "question_id": row.get("question_id"),
                        "question_no": row.get("question_no"),
                        "reason": row.get("reason"),
                        "failed_fields": fields,
                        "expected": _expected_mcq_inventory(row),
                        "current_candidate": row.get("actual"),
                        "benchmark": _inventory_benchmark_key(benchmark),
                    }
                )
    counts: dict[str, int] = {}
    for entry in entries:
        key = f"{entry.get('kind')}:{entry.get('reason')}"
        counts[key] = counts.get(key, 0) + 1
    return {
        "schema_version": 1,
        "source": focus_source or "benchmark_results",
        "entry_count": len(entries),
        "vocab_entry_count": sum(1 for entry in entries if entry.get("kind") == "vocab"),
        "mcq_entry_count": sum(1 for entry in entries if entry.get("kind") == "mcq"),
        "mcq_field_error_count": field_error_total,
        "counts": dict(sorted(counts.items())),
        "diagnostic_only": True,
        "oracle_use_allowed": False,
        "entries": entries,
    }


def _failed_vocab_fields(row: dict[str, Any]) -> list[str]:
    candidate = row.get("best_candidate") if isinstance(row.get("best_candidate"), dict) else {}
    matches = candidate.get("field_matches") if isinstance(candidate.get("field_matches"), dict) else {}
    return [field for field in ("surface", "reading", "meaning_ko") if matches.get(field) is not True]


def _expected_mcq_inventory(row: dict[str, Any]) -> dict[str, Any]:
    expected = row.get("expected")
    return expected if isinstance(expected, dict) else {}


def _inventory_benchmark_key(benchmark: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": benchmark.get("mode"),
        "model_profile": benchmark.get("model_profile"),
        "korean_profile": benchmark.get("korean_profile"),
        "extraction_variant": benchmark.get("extraction_variant"),
    }


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


def _profile_ids_for_run(args: argparse.Namespace) -> list[str]:
    stage = getattr(args, "experiment_stage", "")
    if stage in {"0", "1", "2", "3", "4", "5"}:
        return _stage_profile_ids(args)
    if not getattr(args, "profile_matrix", False):
        return [getattr(args, "model_profile", BASELINE_MODEL_PROFILE)]
    profiles = []
    for profile_id in sorted(LOCAL_MODEL_PROFILES):
        profile = resolve_ocr_model_profile(profile_id)
        if not profile.creates_candidates:
            continue
        if profile.budget == "heavy_local" and not getattr(args, "include_heavy_profiles", False):
            continue
        profiles.append(profile_id)
    return profiles


def _variant_ids_for_run(args: argparse.Namespace) -> list[str]:
    stage = getattr(args, "experiment_stage", "")
    configured_variants = _configured_stage_variants(args)
    if configured_variants:
        return configured_variants
    if stage in {"0", "1"}:
        return [DEFAULT_EXTRACTION_VARIANT]
    if stage == "2":
        return [
            "v5_token_split_v1",
            "v5_vocab_rows_v1",
            "ko_alignment_v1",
            "v5_mcq_v1",
            "ko_crop_confirm_v1",
            "ko_region_columns_v1",
            "mcq_source_rebuild_v1",
            "mcq_choice_band_ocr_v1",
            "jp_region_columns_v1",
            "ko_residual_glyph_v1",
            "mcq_prompt_line_ocr_v1",
            "mcq_choice_glyph_v1",
        ]
    if stage == "3":
        return [
            "v5_token_split_plus_vocab_rows_v1",
            "v5_vocab_rows_plus_ko_alignment_v1",
            "v5_token_split_plus_mcq_v1",
            "v5_full_adapted_v1",
            "ko_consensus_v1",
            "accuracy_recovery_v1",
            "accuracy_recovery_v2",
        ]
    if stage == "4":
        return [
            DEFAULT_EXTRACTION_VARIANT,
            "v5_full_adapted_v1",
            "ko_consensus_v1",
            "mcq_source_rebuild_v1",
            "accuracy_recovery_v1",
            "accuracy_recovery_v2",
        ]
    if stage == "5":
        return [DEFAULT_EXTRACTION_VARIANT, "v5_full_adapted_v1", "accuracy_recovery_v1", "accuracy_recovery_v2"]
    if getattr(args, "variant_matrix", False):
        return [variant for variant in EXTRACTION_VARIANT_ORDER if variant != "provider_agreement_v1"]
    return [normalize_extraction_variant(getattr(args, "extraction_variant", DEFAULT_EXTRACTION_VARIANT))]


def _benchmark_modes_for_run(args: argparse.Namespace) -> list[str]:
    if getattr(args, "benchmark_mode_matrix", False) or getattr(args, "experiment_stage", "") == "4":
        return ["fresh_cli", "persisted_db", "ui_api"]
    return [getattr(args, "benchmark_mode", "fresh_cli")]


def _configured_stage_variants(args: argparse.Namespace) -> list[str]:
    values = [
        value.strip()
        for value in str(getattr(args, "stage_variants", "") or "").split(",")
        if value.strip()
    ]
    variants: list[str] = []
    for value in values:
        try:
            variants.append(normalize_extraction_variant(value))
        except ValueError:
            print(f"Skipping unknown extraction variant for staged benchmark: {value}", file=sys.stderr)
    return list(dict.fromkeys(variants))


def _stage_profile_ids(args: argparse.Namespace) -> list[str]:
    configured = [
        value.strip()
        for value in str(getattr(args, "stage_profiles", "") or "").split(",")
        if value.strip()
    ]
    stage = getattr(args, "experiment_stage", "")
    if configured:
        candidates = configured
    elif stage in {"0", "1", "2"}:
        candidates = MODEL_PAIR_PROFILES
    elif stage == "3":
        candidates = ["jp_v3_det_v3_rec", "jp_v3_det_v5_rec", "jp_v5_det_v5_rec"]
    elif stage == "4":
        candidates = [BASELINE_MODEL_PROFILE, "jp_v3_det_v5_rec", "jp_v5_det_v5_rec"]
    elif stage == "5":
        candidates = ["jp_v5_server_general", "jp_lang_auto"]
    else:
        candidates = [BASELINE_MODEL_PROFILE, "jp_v5_mobile_general"]
    profiles: list[str] = []
    for profile_id in candidates:
        try:
            profile = resolve_ocr_model_profile(profile_id)
        except ValueError:
            print(f"Skipping unknown OCR profile for staged benchmark: {profile_id}", file=sys.stderr)
            continue
        if profile.budget == "heavy_local" and not getattr(args, "include_heavy_profiles", False):
            continue
        if profile.creates_candidates:
            profiles.append(profile.id)
    return profiles


def _apply_experiment_stage(args: argparse.Namespace) -> None:
    stage = getattr(args, "experiment_stage", "")
    if stage in {"0", "1"}:
        args.profile_matrix = True
        args.extraction_variant = DEFAULT_EXTRACTION_VARIANT
        args.include_vl = False
        args.include_google_vision = False
    elif stage == "2":
        args.variant_matrix = True
        args.include_vl = False
        args.include_google_vision = False
    elif stage == "3":
        args.variant_matrix = False
        args.include_vl = False
        args.include_google_vision = False
    elif stage == "4":
        args.include_vl = False
        args.include_google_vision = False
        args.benchmark_mode_matrix = True
    elif stage == "5":
        args.profile_matrix = True
        args.include_heavy_profiles = True
        args.include_vl = True
        args.include_google_vision = True


def _uses_matrix_output_names(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "profile_matrix", False)
        or getattr(args, "variant_matrix", False)
        or getattr(args, "benchmark_mode_matrix", False)
        or getattr(args, "experiment_stage", "")
    )


def _page_work_dir_for_run(root_work_dir: Path, golden: GoldenPage, model_profile: str, args: argparse.Namespace) -> Path:
    if not _uses_matrix_output_names(args):
        return root_work_dir / golden.page_id
    # Share one benchmark DB per page/profile so extraction-variant ablations can
    # reuse the same OCR payload and only rerun candidate generation/diagnostics.
    korean_profile = getattr(args, "korean_profile", DEFAULT_KOREAN_PROFILE)
    return root_work_dir / f"{golden.page_id}.{cache_model_profile_id(model_profile)}.{cache_korean_profile_id(korean_profile)}"


def _profile_env_is_active(profile_id: str, korean_profile_id: str = DEFAULT_KOREAN_PROFILE) -> bool:
    return all(
        os.environ.get(key) == value
        for key, value in profile_env_overrides(profile_id, korean_profile_id).items()
    )


def _run_base_pipeline(
    golden: GoldenPage,
    memory_samples: list[dict[str, Any]],
    engine: str = PADDLEOCR_ENGINE,
    model_profile: str = BASELINE_MODEL_PROFILE,
    korean_profile: str = DEFAULT_KOREAN_PROFILE,
    extraction_variant: str = DEFAULT_EXTRACTION_VARIANT,
) -> ProcessResult:
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
    result = pipeline.process_page(
        page,
        engine=engine,
        model_profile=model_profile,
        korean_profile=korean_profile,
        extraction_variant=extraction_variant,
    )
    memory_samples.append(_memory_sample(f"after_{engine}_pipeline"))
    return result


def _run_benchmark_pipeline(
    golden: GoldenPage,
    memory_samples: list[dict[str, Any]],
    engine: str,
    args: argparse.Namespace,
) -> ProcessResult:
    mode = getattr(args, "benchmark_mode", "fresh_cli")
    model_profile = getattr(args, "model_profile", BASELINE_MODEL_PROFILE)
    korean_profile = getattr(args, "korean_profile", DEFAULT_KOREAN_PROFILE)
    extraction_variant = getattr(args, "extraction_variant", DEFAULT_EXTRACTION_VARIANT)
    if mode == "ui_api":
        return _run_api_pipeline(golden, memory_samples, engine, model_profile, korean_profile, extraction_variant)
    result = _run_base_pipeline(golden, memory_samples, engine, model_profile, korean_profile, extraction_variant)
    if mode == "persisted_db":
        return _persisted_process_result(result.page.id)
    return result


def _run_api_pipeline(
    golden: GoldenPage,
    memory_samples: list[dict[str, Any]],
    engine: str,
    model_profile: str,
    korean_profile: str,
    extraction_variant: str,
) -> ProcessResult:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.routes import router

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
    app = FastAPI()
    app.include_router(router)
    memory_samples.append(_memory_sample(f"before_{engine}_api"))
    with TestClient(app) as client:
        response = client.post(
            f"/api/pages/{page.id}/process",
            params={
                "engine": engine,
                "model_profile": model_profile,
                "korean_profile": korean_profile,
                "extraction_variant": extraction_variant,
            },
        )
    memory_samples.append(_memory_sample(f"after_{engine}_api"))
    if response.status_code >= 400:
        raise RuntimeError(response.text)
    return _persisted_process_result(page.id)


def _persisted_process_result(page_id: str) -> ProcessResult:
    page = database.get_page(page_id)
    if not page:
        raise RuntimeError(f"Page {page_id!r} was not found after processing.")
    run = database.get_active_ocr_run(page_id)
    return ProcessResult(
        page=page,
        tokens=database.get_tokens(page_id),
        cards=database.get_cards(page_id),
        script_summary=run.metrics.get("script_summary", {}) if run else {},
        answer_map={},
        ocr_run=run,
        document_parse=database.get_active_document_parse(page_id),
    )


def _quality_payload(process_result: ProcessResult) -> dict[str, Any]:
    cards = process_result.cards
    exportable = [card for card in cards if card.review_state != "red"]
    return {
        "candidate_recall_count": len(cards),
        "exportable_candidate_count": len(exportable),
        "manual_review_count": sum(1 for card in cards if card.review_state == "yellow" or card.warnings),
        "red_candidate_count": sum(1 for card in cards if card.review_state == "red"),
        "unscored_candidate_count": sum(1 for card in cards if not card.source_bbox),
        "failure_taxonomy": _failure_taxonomy(process_result),
    }


def _evaluation_payload(golden: GoldenPage, process_result: ProcessResult, engine: str) -> dict[str, Any]:
    result = _evaluate_base(golden, process_result)
    payload = _result_dict(result, engine)
    payload.update(_quality_payload(process_result))
    payload["miss_analysis"] = _miss_analysis(golden, process_result, result)
    return payload


def _failure_taxonomy(process_result: ProcessResult) -> dict[str, int]:
    taxonomy = {
        "missing_row": 0,
        "wrong_pairing": 0,
        "surface_ocr_error": 0,
        "reading_ocr_error": 0,
        "korean_ocr_error": 0,
        "script_confusion": 0,
        "stale_or_missing_evidence": 0,
        "bbox_misalignment": 0,
    }
    for card in process_result.cards:
        warnings = " ".join(card.warnings).lower()
        if "missing surface" in warnings:
            taxonomy["surface_ocr_error"] += 1
        if "missing reading" in warnings:
            taxonomy["reading_ocr_error"] += 1
        if "missing korean meaning" in warnings:
            taxonomy["korean_ocr_error"] += 1
        if "script" in warnings:
            taxonomy["script_confusion"] += 1
        if "evidence" in warnings or not card.source.get("field_evidence"):
            taxonomy["stale_or_missing_evidence"] += 1
        if not card.source_bbox:
            taxonomy["bbox_misalignment"] += 1
    return taxonomy


def _miss_analysis(
    golden: GoldenPage,
    process_result: ProcessResult,
    result: VocabEvalResult | McqEvalResult,
) -> dict[str, Any]:
    if isinstance(result, VocabEvalResult):
        return _vocab_miss_analysis(golden, process_result, result.missing_row_ids)
    return _mcq_miss_analysis(golden, process_result, result)


def _vocab_miss_analysis(
    golden: GoldenPage,
    process_result: ProcessResult,
    missing_row_ids: list[str],
) -> dict[str, Any]:
    ordered_tokens = sorted(process_result.tokens, key=lambda token: (token.bbox[1], token.bbox[0], token.id))
    all_text = "\n".join(token.text for token in ordered_tokens)
    korean_text = "\n".join(
        token.text
        for token in ordered_tokens
        if token.source == "paddleocr_korean" or any(0xAC00 <= ord(char) <= 0xD7AF for char in token.text)
    )
    vocab_cards = [card for card in process_result.cards if card.source_type == "vocab_item" and isinstance(card.source, dict)]
    rows: list[dict[str, Any]] = []
    counts = {
        "missing_row": 0,
        "wrong_pairing": 0,
        "surface_ocr_error": 0,
        "reading_ocr_error": 0,
        "korean_ocr_error": 0,
    }
    missing_ids = set(missing_row_ids)
    for row in golden.expected_rows:
        if row.row_id not in missing_ids:
            continue
        raw_presence = {
            "surface": _contains(all_text, row.surface),
            "reading": _contains(all_text, row.reading),
            "meaning_ko": meaning_matches(korean_text or all_text, row.meaning_ko),
        }
        candidate = _best_vocab_miss_candidate(row, vocab_cards)
        reason = _classify_vocab_miss(row, raw_presence, candidate)
        counts[reason] = counts.get(reason, 0) + 1
        rows.append(
            {
                "row_id": row.row_id,
                "reason": reason,
                "expected": {
                    "surface": row.surface,
                    "reading": row.reading,
                    "meaning_ko": row.meaning_ko,
                },
                "raw_presence": raw_presence,
                "best_candidate": _vocab_candidate_summary(row, candidate),
            }
        )
    return {
        "schema_version": 1,
        "kind": "vocab",
        "counts": _nonzero_counts(counts),
        "rows": rows,
    }


def _best_vocab_miss_candidate(row: Any, cards: list[Any]) -> Any | None:
    best_card = None
    best_score = 0
    for card in cards:
        source = card.source
        field_matches = _vocab_candidate_field_matches(row, source)
        score = (
            (6 if field_matches["surface"] and field_matches["reading"] else 0)
            + (3 if field_matches["surface"] else 0)
            + (3 if field_matches["reading"] else 0)
            + (2 if field_matches["meaning_ko"] else 0)
        )
        if score > best_score:
            best_score = score
            best_card = card
    return best_card


def _classify_vocab_miss(row: Any, raw_presence: dict[str, bool], candidate: Any | None) -> str:
    if candidate is None:
        if not any(raw_presence.values()):
            return "missing_row"
        if not raw_presence["surface"]:
            return "surface_ocr_error"
        if not raw_presence["reading"]:
            return "reading_ocr_error"
        if not raw_presence["meaning_ko"]:
            return "korean_ocr_error"
        return "wrong_pairing"

    field_matches = _vocab_candidate_field_matches(row, candidate.source)
    if field_matches["surface"] and field_matches["reading"] and not field_matches["meaning_ko"]:
        return "wrong_pairing" if raw_presence["meaning_ko"] else "korean_ocr_error"
    if not field_matches["surface"] and not raw_presence["surface"]:
        return "surface_ocr_error"
    if not field_matches["reading"] and not raw_presence["reading"]:
        return "reading_ocr_error"
    if not field_matches["meaning_ko"] and not raw_presence["meaning_ko"]:
        return "korean_ocr_error"
    return "wrong_pairing"


def _vocab_candidate_summary(row: Any, card: Any | None) -> dict[str, Any] | None:
    if card is None:
        return None
    source = card.source
    return {
        "surface": str(source.get("surface", "")),
        "reading": str(source.get("reading", "")),
        "meaning_ko": str(source.get("meaning_ko", "")),
        "field_matches": _vocab_candidate_field_matches(row, source),
        "warnings": list(card.warnings[:4]),
        "review_state": card.review_state,
    }


def _vocab_candidate_field_matches(row: Any, source: dict[str, Any]) -> dict[str, bool]:
    return {
        "surface": normalize_text(str(source.get("surface", ""))) == normalize_text(row.surface),
        "reading": normalize_text(str(source.get("reading", ""))) == normalize_text(row.reading),
        "meaning_ko": meaning_matches(str(source.get("meaning_ko", "")), row.meaning_ko),
    }


def _mcq_miss_analysis(
    golden: GoldenPage,
    process_result: ProcessResult,
    result: McqEvalResult,
) -> dict[str, Any]:
    by_no: dict[int, dict[str, Any]] = {}
    for card in process_result.cards:
        if card.source_type != "question_item" or not isinstance(card.source, dict):
            continue
        question_no = card.source.get("question_no")
        if isinstance(question_no, int):
            by_no.setdefault(question_no, card.source)

    rows: list[dict[str, Any]] = []
    field_error_counts = {
        "sentence": 0,
        "target": 0,
        "choices": 0,
        "correct_answer": 0,
        "correct_choice_no": 0,
    }
    for question in golden.expected_questions:
        item = by_no.get(question.question_no)
        if not item:
            rows.append(
                {
                    "question_id": question.question_id,
                    "question_no": question.question_no,
                    "reason": "missing_question",
                    "field_errors": ["question"],
                    "expected": {
                        "sentence": question.sentence,
                        "target": question.target,
                        "choices": question.choices,
                        "correct_answer": question.correct_answer,
                        "correct_choice_no": question.correct_choice_no,
                    },
                    "actual": None,
                }
            )
            continue
        source_item = _strict_mcq_source_fields(item)
        field_matches = _mcq_candidate_field_matches(question, item)
        field_errors = [field for field, matched in field_matches.items() if not matched]
        if not field_errors:
            continue
        for field in field_errors:
            field_error_counts[field] += 1
        rows.append(
            {
                "question_id": question.question_id,
                "question_no": question.question_no,
                "reason": "source_field_ocr_error",
                "field_errors": field_errors,
                "field_matches": field_matches,
                "expected": {
                    "sentence": question.sentence,
                    "target": question.target,
                    "choices": question.choices,
                    "correct_answer": question.correct_answer,
                    "correct_choice_no": question.correct_choice_no,
                },
                "actual": {
                    "sentence": str(source_item.get("sentence", "")),
                    "target": str(source_item.get("target", "")),
                    "choices": [str(choice) for choice in source_item.get("choices", [])] if isinstance(source_item.get("choices"), list) else [],
                    "correct_answer": str(source_item.get("correct_answer", "")),
                    "correct_choice_no": source_item.get("correct_choice_no"),
                },
            }
        )

    return {
        "schema_version": 1,
        "kind": "mcq",
        "counts": _nonzero_counts(
            {
                "missing_question": len(result.missing_question_ids),
                "source_field_ocr_error": max(0, result.source_field_expected - result.source_field_matches),
                "source_question_mismatch": len(result.source_mismatch_question_ids),
            }
        ),
        "field_error_counts": _nonzero_counts(field_error_counts),
        "rows": rows,
    }


def _mcq_candidate_field_matches(question: Any, source: dict[str, Any]) -> dict[str, bool]:
    source = _strict_mcq_source_fields(source)
    return {
        "sentence": _sentence_matches_for_analysis(str(source.get("sentence", "")), question.sentence),
        "target": normalize_text(str(source.get("target", ""))) == normalize_text(question.target),
        "choices": _choices_match_for_analysis(source.get("choices"), question.choices),
        "correct_answer": normalize_text(str(source.get("correct_answer", ""))) == normalize_text(question.correct_answer),
        "correct_choice_no": source.get("correct_choice_no") == question.correct_choice_no,
    }


def _strict_mcq_source_fields(source: dict[str, Any]) -> dict[str, Any]:
    source_fields = source.get("source_fields")
    return source_fields if isinstance(source_fields, dict) else source


def _sentence_matches_for_analysis(actual: str, expected: str) -> bool:
    actual_norm = normalize_text(actual)
    expected_norm = normalize_text(expected)
    return bool(expected_norm and (actual_norm == expected_norm or expected_norm in actual_norm))


def _choices_match_for_analysis(actual: object, expected: list[str]) -> bool:
    if not isinstance(actual, list):
        return False
    normalized_actual = [normalize_text(str(choice)) for choice in actual]
    normalized_expected = [normalize_text(choice) for choice in expected]
    return normalized_actual == normalized_expected


def _nonzero_counts(counts: dict[str, int]) -> dict[str, int]:
    return {key: value for key, value in counts.items() if value}


def _benchmark_manifest(args: argparse.Namespace, process_result: ProcessResult) -> dict[str, Any]:
    profile = resolve_ocr_model_profile(getattr(args, "model_profile", BASELINE_MODEL_PROFILE))
    korean_profile = resolve_korean_ocr_profile(getattr(args, "korean_profile", DEFAULT_KOREAN_PROFILE))
    extraction_variant = getattr(args, "extraction_variant", DEFAULT_EXTRACTION_VARIANT)
    run = process_result.ocr_run
    metrics = run.metrics if run else {}
    profile_manifest = metrics.get("model_profile") if isinstance(metrics.get("model_profile"), dict) else {}
    graph = metrics.get("document_graph") if isinstance(metrics.get("document_graph"), dict) else {}
    extraction_variant_metrics = (
        metrics.get("extraction_variant_metrics") if isinstance(metrics.get("extraction_variant_metrics"), dict) else {}
    )
    return {
        "schema_version": 1,
        "mode": getattr(args, "benchmark_mode", "fresh_cli"),
        "experiment_stage": getattr(args, "experiment_stage", "") or None,
        "engine": _primary_engine(args),
        "model_profile": profile.id,
        "model_profile_label": profile.label,
        "korean_profile": korean_profile.id,
        "korean_profile_label": korean_profile.label,
        "budget": profile.budget,
        "extraction_variant": extraction_variant,
        "extraction_variant_components": sorted(extraction_variant_components(extraction_variant)),
        "promotion_status": "experimental",
        "promotion_reason": "Changing the production default requires a holdout set and pre-registered gates.",
        "promotion_gates": {
            "holdout_required": True,
            "min_vocab_row_accuracy_gain_points": 15,
            "max_page_regression_points": 3,
            "mcq_regression_allowed": False,
            "requires_acceptable_evidence_alignment": True,
        },
        "profile_manifest": profile_manifest,
        "document_graph_metrics": graph.get("metrics", {}),
        "extraction_variant_metrics": extraction_variant_metrics,
        "cache": profile_manifest.get("cache", {}),
        "focus_misses": _focus_miss_summary(getattr(args, "focus_misses_from", ""), process_result.page.id),
    }


def _focus_miss_summary(path_value: str, page_id: str) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return {"source": path_value, "status": "missing", "diagnostic_only": True}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"source": path_value, "status": "invalid", "error": str(exc), "diagnostic_only": True}
    entries = payload.get("entries") if isinstance(payload, dict) else []
    page_entries = [entry for entry in entries if isinstance(entry, dict) and entry.get("page_id") == page_id]
    return {
        "source": path_value,
        "status": "loaded",
        "diagnostic_only": True,
        "oracle_use_allowed": False,
        "page_entry_count": len(page_entries),
        "failed_fields": sorted(
            {
                str(field)
                for entry in page_entries
                for field in (entry.get("failed_fields") if isinstance(entry.get("failed_fields"), list) else [])
            }
        ),
    }


def _write_audit_artifacts(
    golden: GoldenPage,
    process_result: ProcessResult,
    args: argparse.Namespace,
    evaluation_payload: dict[str, Any] | None = None,
) -> dict[str, str]:
    work_dir = Path(getattr(args, "work_dir", "") or "")
    if not work_dir:
        return {}
    audit_dir = work_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    profile_id = getattr(args, "model_profile", BASELINE_MODEL_PROFILE)
    korean_profile = getattr(args, "korean_profile", DEFAULT_KOREAN_PROFILE)
    variant = getattr(args, "extraction_variant", DEFAULT_EXTRACTION_VARIANT)
    stem = f"{golden.page_id}.{profile_id}.{korean_profile}.{variant}.{_primary_engine(args)}"
    json_path = audit_dir / f"{stem}.overlay.json"
    png_path = audit_dir / f"{stem}.overlay.png"
    run_metrics = process_result.ocr_run.metrics if process_result.ocr_run else {}
    document_graph = run_metrics.get("document_graph") if isinstance(run_metrics.get("document_graph"), dict) else {}
    extraction_variant_metrics = (
        run_metrics.get("extraction_variant_metrics") if isinstance(run_metrics.get("extraction_variant_metrics"), dict) else {}
    )
    transform = document_graph.get("transform") if isinstance(document_graph.get("transform"), dict) else {}
    payload = {
        "schema_version": 1,
        "page_id": golden.page_id,
        "benchmark_mode": getattr(args, "benchmark_mode", "fresh_cli"),
        "engine": _primary_engine(args),
        "model_profile": profile_id,
        "korean_profile": korean_profile,
        "extraction_variant": variant,
        "image_path": str(golden.image_path),
        "processed_image_path": process_result.page.processed_image_path,
        "coordinate_space": transform.get("coordinate_space", "processed_image"),
        "transform": transform,
        "document_graph_metrics": document_graph.get("metrics", {}),
        "extraction_variant_metrics": extraction_variant_metrics,
        "tokens": [token.model_dump() for token in process_result.tokens],
        "cards": [
            {
                "id": card.id,
                "source_type": card.source_type,
                "source_id": card.source_id,
                "note_type": card.note_type,
                "source_bbox": card.source_bbox,
                "confidence": card.confidence,
                "review_state": card.review_state,
                "warnings": card.warnings,
                "field_evidence": card.source.get("field_evidence"),
            }
            for card in process_result.cards
        ],
        "document_blocks": [
            block.model_dump(mode="json")
            for block in (process_result.document_parse.blocks if process_result.document_parse else [])
        ],
        "major_failures": {
            "missing_ids": (evaluation_payload or {}).get("missing_ids", []),
            "source_mismatch_ids": (evaluation_payload or {}).get("source_mismatch_ids", []),
            "failure_taxonomy": (evaluation_payload or {}).get("failure_taxonomy", {}),
        },
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_overlay_png(png_path, process_result)
    return {"overlay_json": str(json_path), "overlay_png": str(png_path)}


def _write_overlay_png(path: Path, process_result: ProcessResult) -> None:
    image_path = Path(process_result.page.processed_image_path or process_result.page.original_image_path)
    if not image_path.exists():
        return
    with Image.open(image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    for token in process_result.tokens:
        _draw_bbox(draw, token.bbox, outline=(128, 128, 128, 150), width=1)
    document_blocks = process_result.document_parse.blocks if process_result.document_parse else []
    for block in document_blocks:
        if block.bbox:
            _draw_bbox(draw, block.bbox, outline=(110, 70, 170, 180), width=3)
    for card in process_result.cards:
        if card.source_bbox:
            color = {"green": (34, 139, 82, 220), "yellow": (245, 170, 28, 220), "red": (190, 45, 45, 220)}.get(
                card.review_state,
                (40, 120, 160, 220),
            )
            _draw_bbox(draw, card.source_bbox, outline=color, width=3)
    canvas.save(path)


def _draw_bbox(draw: ImageDraw.ImageDraw, bbox: list[float], *, outline: tuple[int, int, int, int], width: int) -> None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return
    x1, y1, x2, y2 = [float(value) for value in bbox]
    if x2 <= x1 or y2 <= y1:
        return
    draw.rectangle([x1, y1, x2, y2], outline=outline, width=width)


def _evaluate_base(golden: GoldenPage, process_result: ProcessResult) -> VocabEvalResult | McqEvalResult:
    if golden.expected_rows:
        return evaluate_vocab_page(golden, process_result)
    return evaluate_mcq_page(golden, process_result)


def _run_engine_evaluation(golden: GoldenPage, memory_samples: list[dict[str, Any]], engine: str) -> dict[str, Any]:
    try:
        result = _run_base_pipeline(golden, memory_samples, engine)
        payload = _evaluation_payload(golden, result, engine)
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


def _raw_field_coverage(golden: GoldenPage, process_result: ProcessResult) -> dict[str, Any]:
    ordered_tokens = sorted(process_result.tokens, key=lambda token: (token.bbox[1], token.bbox[0], token.id))
    all_text = "\n".join(token.text for token in ordered_tokens)
    korean_text = "\n".join(
        token.text
        for token in ordered_tokens
        if token.source == "paddleocr_korean" or any(0xAC00 <= ord(char) <= 0xD7AF for char in token.text)
    )
    field_totals = {
        "surface": [0, 0],
        "reading": [0, 0],
        "meaning_ko": [0, 0],
        "sentence": [0, 0],
        "target": [0, 0],
        "choices": [0, 0],
        "correct_answer": [0, 0],
    }
    for row in golden.expected_rows:
        _count_field_match(field_totals["surface"], _contains(all_text, row.surface))
        _count_field_match(field_totals["reading"], _contains(all_text, row.reading))
        _count_field_match(field_totals["meaning_ko"], meaning_matches(korean_text or all_text, row.meaning_ko))
    for question in golden.expected_questions:
        _count_field_match(field_totals["sentence"], _contains(all_text, question.sentence))
        _count_field_match(field_totals["target"], _contains(all_text, question.target))
        _count_field_match(field_totals["correct_answer"], _contains(all_text, question.correct_answer))
        for choice in question.choices:
            _count_field_match(field_totals["choices"], _contains(all_text, choice))
    return {
        "schema_version": 1,
        "surface": _field_rate(field_totals["surface"]),
        "reading": _field_rate(field_totals["reading"]),
        "meaning_ko": _field_rate(field_totals["meaning_ko"]),
        "sentence": _field_rate(field_totals["sentence"]),
        "target": _field_rate(field_totals["target"]),
        "choices": _field_rate(field_totals["choices"]),
        "correct_answer": _field_rate(field_totals["correct_answer"]),
        "korean_raw_recall": _field_rate(field_totals["meaning_ko"]),
        "token_count": len(ordered_tokens),
        "korean_token_count": sum(1 for token in ordered_tokens if token.source == "paddleocr_korean"),
        "bbox_count": sum(1 for token in ordered_tokens if isinstance(token.bbox, list) and len(token.bbox) == 4),
    }


def _count_field_match(counter: list[int], matched: bool) -> None:
    counter[1] += 1
    if matched:
        counter[0] += 1


def _field_rate(counter: list[int]) -> dict[str, Any]:
    matched, expected = counter
    return {
        "matched": matched,
        "expected": expected,
        "accuracy": matched / expected if expected else None,
    }


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
            "strict_ocr_score": result.source_field_accuracy,
            "strict_ocr_score_kind": "mcq_source_field_accuracy",
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
        "strict_ocr_score": result.row_accuracy,
        "strict_ocr_score_kind": "vocab_row_accuracy",
        "layout_matched_rows": result.layout_matched_rows,
        "layout_recall": result.layout_recall,
        "surface_accuracy": result.surface_accuracy,
        "reading_accuracy": result.reading_accuracy,
        "meaning_accuracy": result.meaning_accuracy,
        "ocr_supported_items": result.ocr_supported_items,
        "glossary_supported_items": result.glossary_supported_items,
        "surface_matches": result.surface_matches,
        "reading_matches": result.reading_matches,
        "surface_reading_matches": result.surface_reading_matches,
        "meaning_matches": result.meaning_matches,
        "generated_notes": result.generated_notes,
        "korean_field_missing_hangul": result.korean_field_missing_hangul,
        "japanese_field_has_hangul": result.japanese_field_has_hangul,
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
        "process_tree_rss_mb": _process_tree_rss_mb(os.getpid()),
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
    process_tree_peak_rss_mb = max((sample.get("process_tree_rss_mb") or 0 for sample in memory_samples), default=0)
    return {
        "wall_seconds": round(wall_seconds, 3),
        "user_cpu_seconds": round(user_cpu_seconds, 3),
        "system_cpu_seconds": round(system_cpu_seconds, 3),
        "cpu_seconds": round(cpu_seconds, 3),
        "cpu_percent_of_one_core": round((cpu_seconds / wall_seconds) * 100, 2) if wall_seconds else None,
        "peak_rss_mb": round(float(peak_rss_mb), 2),
        "process_tree_peak_rss_mb": round(float(process_tree_peak_rss_mb), 2),
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


def _resource_metrics_with_cache(
    start: dict[str, float],
    end: dict[str, float],
    memory_samples: list[dict[str, Any]],
    base_payload: dict[str, Any],
) -> dict[str, Any]:
    metrics = _resource_metrics(start, end, memory_samples)
    metrics["cache"] = _cache_summary(base_payload)
    return metrics


def _cache_summary(base_payload: dict[str, Any]) -> dict[str, Any]:
    benchmark = base_payload.get("benchmark") if isinstance(base_payload.get("benchmark"), dict) else {}
    cache = benchmark.get("cache") if isinstance(benchmark.get("cache"), dict) else {}
    result_cache_hit = cache.get("hit")
    cache_phase = "unknown"
    if result_cache_hit is True:
        cache_phase = "warm_ocr_cache"
    elif result_cache_hit is False:
        cache_phase = "cold_or_uncached"
    return {
        "result_cache_hit": result_cache_hit,
        "model_cache_hit": cache.get("model_cache_hit"),
        "cache_phase": cache_phase,
        "timing_bucket": cache_phase,
        "cache_key": cache.get("key"),
        "timing_note": "Benchmark wall time is measured per run; compare a cold run and a repeated warm run with the same cache key.",
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
        f"  {base.get('mode', 'base')}: {base['matched']}/{base['expected']} accuracy={base['accuracy']:.1%} {_generated_count(base)}",
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
            f"accuracy={result.vl.get('accuracy', 0):.1%} {_generated_count(result.vl)}"
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


def _write_dashboard_markdown(results: list[PageBenchmark], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# OCR Benchmark Dashboard",
        "",
        "## Summary",
        "",
        "| Mode | Profile | Korean | Variant | Pages | Matched | Accuracy | Strict OCR | Evidence | Raw KO | Surface | Reading | Meaning | Miss Causes | Field Errors | Recovery | Review | Blocked | Shadow Rows | Risk | Wall | Peak RSS | Cache Hits | Errors |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    lines.extend(_dashboard_summary_rows(results))
    recovery_rows = _dashboard_recovery_detail_rows(results)
    if recovery_rows:
        lines.extend(
            [
                "",
                "## Recovery Details",
                "",
                "| Mode | Profile | Korean | Variant | Component | Pages | Attempted | Accepted | Rejected | Cache | Resource Caps | Counts |",
                "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        lines.extend(recovery_rows)
    lines.extend(
        [
            "",
            "## Pages",
            "",
        ]
    )
    lines.extend(
        [
            "| Page | Mode | Profile | Korean | Variant | Accuracy | Strict OCR | Evidence | Raw KO | Surface | Reading | Meaning | Miss Causes | Field Errors | Recovery | Review | Blocked | Shadow Rows | Risk | Wall | Peak RSS | Cache | Cache Phase | Errors |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- |",
        ]
    )
    for result in results:
        base = result.base
        benchmark = base.get("benchmark") if isinstance(base.get("benchmark"), dict) else {}
        graph_metrics = benchmark.get("document_graph_metrics") if isinstance(benchmark.get("document_graph_metrics"), dict) else {}
        vocab_alignment = _vocab_alignment_payload(benchmark)
        raw_coverage = base.get("raw_field_coverage") if isinstance(base.get("raw_field_coverage"), dict) else {}
        korean_recall = raw_coverage.get("korean_raw_recall") if isinstance(raw_coverage.get("korean_raw_recall"), dict) else {}
        cache = result.resource_metrics.get("cache") if isinstance(result.resource_metrics.get("cache"), dict) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    result.page_id,
                    str(benchmark.get("mode", "unknown")),
                    str(benchmark.get("model_profile", "unknown")),
                    str(benchmark.get("korean_profile", "unknown")),
                    str(benchmark.get("extraction_variant", "unknown")),
                    _percent(base.get("accuracy")),
                    _percent(base.get("strict_ocr_score")),
                    _percent(graph_metrics.get("evidence_alignment_score")),
                    _percent(korean_recall.get("accuracy")),
                    _percent(base.get("surface_accuracy")),
                    _percent(base.get("reading_accuracy")),
                    _percent(base.get("meaning_accuracy")),
                    _miss_counts_label(base.get("miss_analysis")),
                    _field_error_counts_label(base.get("miss_analysis")),
                    _recovery_label(_recovery_payload(benchmark)),
                    str(int(_numeric(base.get("manual_review_count")))),
                    str(int(_numeric(base.get("red_candidate_count")))),
                    _shadow_rows_label(vocab_alignment),
                    _risk_label(vocab_alignment),
                    f"{result.resource_metrics.get('wall_seconds', 0)}s",
                    f"{result.resource_metrics.get('peak_rss_mb', 0)} MB",
                    str(cache.get("result_cache_hit")),
                    str(cache.get("cache_phase", "unknown")),
                    "; ".join(result.errors)[:80],
                ]
            )
            + " |"
        )
    lines.extend(["", "## Gates", ""])
    lines.extend(_dashboard_gate_rows(results))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dashboard_summary_rows(results: list[PageBenchmark]) -> list[str]:
    groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for result in results:
        base = result.base
        benchmark = base.get("benchmark") if isinstance(base.get("benchmark"), dict) else {}
        graph_metrics = benchmark.get("document_graph_metrics") if isinstance(benchmark.get("document_graph_metrics"), dict) else {}
        raw_coverage = base.get("raw_field_coverage") if isinstance(base.get("raw_field_coverage"), dict) else {}
        korean_recall = raw_coverage.get("korean_raw_recall") if isinstance(raw_coverage.get("korean_raw_recall"), dict) else {}
        key = (
            str(benchmark.get("mode", "unknown")),
            str(benchmark.get("model_profile", "unknown")),
            str(benchmark.get("korean_profile", "unknown")),
            str(benchmark.get("extraction_variant", "unknown")),
        )
        group = groups.setdefault(
            key,
            {
                "pages": 0,
                "matched": 0.0,
                "expected": 0.0,
                "strict_matched": 0.0,
                "strict_expected": 0.0,
                "evidence_sum": 0.0,
                "evidence_count": 0,
                "korean_recall_matched": 0.0,
                "korean_recall_expected": 0.0,
                "surface_matched": 0.0,
                "surface_expected": 0.0,
                "reading_matched": 0.0,
                "reading_expected": 0.0,
                "meaning_matched": 0.0,
                "meaning_expected": 0.0,
                "miss_counts": {},
                "field_error_counts": {},
                "recovery_counts": {},
                "recovery_attempted": 0,
                "recovery_accepted": 0,
                "manual_review_count": 0,
                "red_candidate_count": 0,
                "shadow_complete_rows": 0.0,
                "shadow_total_rows": 0.0,
                "risk_counts": {},
                "wall_seconds": 0.0,
                "peak_rss_mb": 0.0,
                "cache_hits": 0,
                "errors": [],
            },
        )
        group["pages"] += 1
        group["matched"] += _numeric(base.get("matched"))
        group["expected"] += _numeric(base.get("expected"))
        strict_matched, strict_expected = _strict_score_counts(base)
        group["strict_matched"] += strict_matched
        group["strict_expected"] += strict_expected
        group["manual_review_count"] += int(_numeric(base.get("manual_review_count")))
        group["red_candidate_count"] += int(_numeric(base.get("red_candidate_count")))
        evidence = graph_metrics.get("evidence_alignment_score")
        if isinstance(evidence, (int, float)):
            group["evidence_sum"] += float(evidence)
            group["evidence_count"] += 1
        korean_expected = korean_recall.get("expected")
        if isinstance(korean_expected, (int, float)) and korean_expected:
            group["korean_recall_matched"] += _numeric(korean_recall.get("matched"))
            group["korean_recall_expected"] += float(korean_expected)
        for field in ("surface", "reading", "meaning"):
            if isinstance(base.get(f"{field}_accuracy"), (int, float)):
                group[f"{field}_matched"] += _numeric(base.get(f"{field}_matches"))
                group[f"{field}_expected"] += _numeric(base.get("expected"))
        miss_counts = group["miss_counts"] if isinstance(group["miss_counts"], dict) else {}
        for reason, count in _miss_counts(base.get("miss_analysis")).items():
            miss_counts[reason] = int(miss_counts.get(reason, 0)) + count
        group["miss_counts"] = miss_counts
        field_error_counts = group["field_error_counts"] if isinstance(group["field_error_counts"], dict) else {}
        for field, count in _field_error_counts(base.get("miss_analysis")).items():
            field_error_counts[field] = int(field_error_counts.get(field, 0)) + count
        group["field_error_counts"] = field_error_counts
        recovery = _recovery_payload(benchmark)
        group["recovery_attempted"] += int(_numeric(recovery.get("attempted")))
        group["recovery_accepted"] += int(_numeric(recovery.get("accepted")))
        recovery_counts = group["recovery_counts"] if isinstance(group["recovery_counts"], dict) else {}
        for reason, count in (recovery.get("counts") if isinstance(recovery.get("counts"), dict) else {}).items():
            if isinstance(count, int):
                recovery_counts[str(reason)] = int(recovery_counts.get(str(reason), 0)) + count
        group["recovery_counts"] = recovery_counts
        vocab_alignment = _vocab_alignment_payload(benchmark)
        shadow_total = vocab_alignment.get("shadow_row_count")
        if isinstance(shadow_total, (int, float)):
            group["shadow_complete_rows"] += _numeric(vocab_alignment.get("shadow_complete_row_count"))
            group["shadow_total_rows"] += float(shadow_total)
        risk = vocab_alignment.get("risk_level")
        if isinstance(risk, str) and risk:
            risk_counts = group["risk_counts"] if isinstance(group["risk_counts"], dict) else {}
            risk_counts[risk] = int(risk_counts.get(risk, 0)) + 1
            group["risk_counts"] = risk_counts
        group["wall_seconds"] += _numeric(result.resource_metrics.get("wall_seconds"))
        group["peak_rss_mb"] = max(float(group["peak_rss_mb"]), _numeric(result.resource_metrics.get("peak_rss_mb")))
        cache = result.resource_metrics.get("cache") if isinstance(result.resource_metrics.get("cache"), dict) else {}
        if cache.get("result_cache_hit") is True:
            group["cache_hits"] += 1
        group["errors"].extend(result.errors)
    rows: list[tuple[float, float, str]] = []
    for (mode, profile, korean_profile, variant), group in sorted(groups.items()):
        expected = float(group["expected"])
        accuracy = float(group["matched"]) / expected if expected else 0.0
        strict_expected = float(group["strict_expected"])
        strict_accuracy = float(group["strict_matched"]) / strict_expected if strict_expected else None
        evidence_count = int(group["evidence_count"])
        evidence = float(group["evidence_sum"]) / evidence_count if evidence_count else None
        korean_expected = float(group["korean_recall_expected"])
        korean_recall = float(group["korean_recall_matched"]) / korean_expected if korean_expected else None
        surface_accuracy = _field_summary_accuracy(group, "surface")
        reading_accuracy = _field_summary_accuracy(group, "reading")
        meaning_accuracy = _field_summary_accuracy(group, "meaning")
        shadow_total = float(group["shadow_total_rows"])
        shadow_rows = f"{int(group['shadow_complete_rows'])}/{int(shadow_total)}" if shadow_total else "n/a"
        row = (
            "| "
            + " | ".join(
                [
                    mode,
                    profile,
                    korean_profile,
                    variant,
                    str(group["pages"]),
                    f"{int(group['matched'])}/{int(expected)}",
                    _percent(accuracy),
                    _percent(strict_accuracy),
                    _percent(evidence),
                    _percent(korean_recall),
                    _percent(surface_accuracy),
                    _percent(reading_accuracy),
                    _percent(meaning_accuracy),
                    _miss_counts_label({"counts": group.get("miss_counts")}),
                    _field_error_counts_label({"field_error_counts": group.get("field_error_counts")}),
                    _recovery_label(
                        {
                            "attempted": group.get("recovery_attempted"),
                            "accepted": group.get("recovery_accepted"),
                            "counts": group.get("recovery_counts"),
                        }
                    ),
                    str(group["manual_review_count"]),
                    str(group["red_candidate_count"]),
                    shadow_rows,
                    _risk_counts_label(group.get("risk_counts")),
                    f"{round(float(group['wall_seconds']), 3)}s",
                    f"{round(float(group['peak_rss_mb']), 2)} MB",
                    f"{group['cache_hits']}/{group['pages']}",
                    "; ".join(group["errors"])[:80],
                ]
            )
            + " |"
        )
        rows.append((accuracy, float(group["matched"]), row))
    return [row for _accuracy, _matched, row in sorted(rows, key=lambda item: (-item[0], -item[1], item[2]))]


def _dashboard_recovery_detail_rows(results: list[PageBenchmark]) -> list[str]:
    groups: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for result in results:
        base = result.base
        benchmark = base.get("benchmark") if isinstance(base.get("benchmark"), dict) else {}
        recovery = _recovery_payload(benchmark)
        if not recovery:
            continue
        group_prefix = (
            str(benchmark.get("mode", "unknown")),
            str(benchmark.get("model_profile", "unknown")),
            str(benchmark.get("korean_profile", "unknown")),
            str(benchmark.get("extraction_variant", "unknown")),
        )
        for component_name, component in _recovery_dashboard_components(recovery):
            key = (*group_prefix, component_name)
            group = groups.setdefault(
                key,
                {
                    "pages": 0,
                    "attempted": 0,
                    "accepted": 0,
                    "counts": {},
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "resource_caps": {},
                },
            )
            group["pages"] += 1
            group["attempted"] += int(_numeric(component.get("attempted")))
            group["accepted"] += int(_numeric(component.get("accepted")))
            counts = group["counts"] if isinstance(group["counts"], dict) else {}
            for reason, count in (component.get("counts") if isinstance(component.get("counts"), dict) else {}).items():
                if isinstance(count, int) and count:
                    counts[str(reason)] = int(counts.get(str(reason), 0)) + count
            group["counts"] = counts
            cache_counts = _recovery_cache_counts(component)
            group["cache_hits"] += cache_counts["hits"]
            group["cache_misses"] += cache_counts["misses"]
            caps = group["resource_caps"] if isinstance(group["resource_caps"], dict) else {}
            for reason, count in _resource_cap_counts(component).items():
                caps[reason] = int(caps.get(reason, 0)) + count
            group["resource_caps"] = caps
    rows: list[str] = []
    for (mode, profile, korean, variant, component), group in sorted(groups.items()):
        attempted = int(group["attempted"])
        accepted = int(group["accepted"])
        rows.append(
            "| "
            + " | ".join(
                [
                    mode,
                    profile,
                    korean,
                    variant,
                    component,
                    str(group["pages"]),
                    str(attempted),
                    str(accepted),
                    str(max(0, attempted - accepted)),
                    f"{group['cache_hits']}/{group['cache_hits'] + group['cache_misses']}",
                    _counts_summary_label(group.get("resource_caps")),
                    _counts_summary_label(group.get("counts")),
                ]
            )
            + " |"
        )
    return rows


def _recovery_dashboard_components(recovery: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    components = recovery.get("components") if isinstance(recovery.get("components"), dict) else {}
    if not components:
        return [(str(recovery.get("kind") or "recovery"), recovery)]
    rows: list[tuple[str, dict[str, Any]]] = []
    for name, value in sorted(components.items()):
        if not isinstance(value, dict):
            continue
        nested_components = value.get("components") if isinstance(value.get("components"), dict) else {}
        if nested_components:
            rows.extend((f"{name}/{child_name}", child) for child_name, child in _recovery_dashboard_components(value))
        else:
            rows.append((str(name), value))
    return rows


def _recovery_cache_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts = {"hits": 0, "misses": 0}
    cache = payload.get("cache") if isinstance(payload.get("cache"), dict) else {}
    if isinstance(cache.get("hits"), int):
        counts["hits"] += int(cache.get("hits") or 0)
    if isinstance(cache.get("misses"), int):
        counts["misses"] += int(cache.get("misses") or 0)
    attempts = payload.get("attempts") if isinstance(payload.get("attempts"), list) else []
    for attempt in attempts:
        if isinstance(attempt, dict):
            _add_attempt_cache_counts(attempt, counts)
            candidates = attempt.get("candidates") if isinstance(attempt.get("candidates"), list) else []
            for candidate in candidates:
                if isinstance(candidate, dict):
                    _add_attempt_cache_counts(candidate, counts)
    return counts


def _add_attempt_cache_counts(payload: dict[str, Any], counts: dict[str, int]) -> None:
    cache = payload.get("cache") if isinstance(payload.get("cache"), dict) else {}
    if cache.get("hit") is True:
        counts["hits"] += 1
    elif cache.get("hit") is False:
        counts["misses"] += 1


def _resource_cap_counts(payload: dict[str, Any]) -> dict[str, int]:
    resource_caps = payload.get("resource_caps") if isinstance(payload.get("resource_caps"), dict) else {}
    counts = {str(key): int(value) for key, value in resource_caps.items() if isinstance(value, int) and value}
    for key, value in (payload.get("counts") if isinstance(payload.get("counts"), dict) else {}).items():
        if isinstance(value, int) and value and "resource_cap" in str(key):
            counts[str(key)] = counts.get(str(key), 0) + value
    return counts


def _counts_summary_label(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    parts = [f"{key}:{value[key]}" for key in sorted(value) if isinstance(value[key], int) and value[key]]
    return ", ".join(parts) if parts else "none"


def _dashboard_gate_rows(results: list[PageBenchmark]) -> list[str]:
    rows = [
        "| Mode | Profile | Korean | Variant | Overall | Strict | Meaning | Surface | Reading | MCQ Semantic | MCQ Source | Evidence | Peak RSS | Pass |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for gate in _success_gate_payload(results):
        rows.append(
            "| "
            + " | ".join(
                [
                    str(gate["mode"]),
                    str(gate["model_profile"]),
                    str(gate["korean_profile"]),
                    str(gate["extraction_variant"]),
                    gate["overall"],
                    gate["strict"],
                    gate["meaning"],
                    gate["surface"],
                    gate["reading"],
                    gate["mcq_semantic"],
                    gate["mcq_source"],
                    gate["evidence"],
                    gate["peak_rss"],
                    "yes" if gate["passed"] else "no",
                ]
            )
            + " |"
        )
    return rows


def _success_gate_payload(results: list[PageBenchmark]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for result in results:
        base = result.base
        benchmark = base.get("benchmark") if isinstance(base.get("benchmark"), dict) else {}
        key = (
            str(benchmark.get("mode", "unknown")),
            str(benchmark.get("model_profile", "unknown")),
            str(benchmark.get("korean_profile", "unknown")),
            str(benchmark.get("extraction_variant", "unknown")),
        )
        group = groups.setdefault(
            key,
            {
                "overall": [0.0, 0.0],
                "strict": [0.0, 0.0],
                "meaning": [0.0, 0.0],
                "surface": [0.0, 0.0],
                "reading": [0.0, 0.0],
                "mcq_semantic": [0.0, 0.0],
                "mcq_source": [0.0, 0.0],
                "evidence_sum": 0.0,
                "evidence_count": 0,
                "peak_rss": 0.0,
            },
        )
        group["overall"][0] += _numeric(base.get("matched"))
        group["overall"][1] += _numeric(base.get("expected"))
        strict_matched, strict_expected = _strict_score_counts(base)
        group["strict"][0] += strict_matched
        group["strict"][1] += strict_expected
        for field in ("meaning", "surface", "reading"):
            if isinstance(base.get(f"{field}_matches"), (int, float)):
                group[field][0] += _numeric(base.get(f"{field}_matches"))
                group[field][1] += _numeric(base.get("expected"))
        if isinstance(base.get("source_field_matches"), (int, float)):
            group["mcq_semantic"][0] += _numeric(base.get("matched"))
            group["mcq_semantic"][1] += _numeric(base.get("expected"))
            group["mcq_source"][0] += _numeric(base.get("source_field_matches"))
            group["mcq_source"][1] += _numeric(base.get("source_field_expected"))
        graph = benchmark.get("document_graph_metrics") if isinstance(benchmark.get("document_graph_metrics"), dict) else {}
        evidence = graph.get("evidence_alignment_score")
        if isinstance(evidence, (int, float)):
            group["evidence_sum"] += float(evidence)
            group["evidence_count"] += 1
        group["peak_rss"] = max(float(group["peak_rss"]), _numeric(result.resource_metrics.get("peak_rss_mb")))
    payload: list[dict[str, Any]] = []
    for (mode, profile, korean, variant), group in sorted(groups.items()):
        evidence = group["evidence_sum"] / group["evidence_count"] if group["evidence_count"] else 0.0
        targets = _gate_targets_for_variant(variant)
        passed = (
            group["overall"][0] >= targets["overall"]
            and group["overall"][1] >= 80
            and group["strict"][0] >= targets["strict"]
            and group["meaning"][0] >= targets["meaning"]
            and group["surface"][0] >= targets["surface"]
            and group["reading"][0] >= targets["reading"]
            and group["mcq_semantic"][0] >= 20
            and group["mcq_source"][0] >= targets["mcq_source"]
            and evidence >= targets["evidence"]
            and group["peak_rss"] < 3276.8
        )
        payload.append(
            {
                "mode": mode,
                "model_profile": profile,
                "korean_profile": korean,
                "extraction_variant": variant,
                "overall": _gate_count(group["overall"]),
                "strict": _gate_count(group["strict"]),
                "meaning": _gate_count(group["meaning"]),
                "surface": _gate_count(group["surface"]),
                "reading": _gate_count(group["reading"]),
                "mcq_semantic": _gate_count(group["mcq_semantic"]),
                "mcq_source": _gate_count(group["mcq_source"]),
                "evidence": _percent(evidence),
                "peak_rss": f"{round(float(group['peak_rss']), 2)} MB",
                "passed": passed,
            }
        )
    return payload


def _gate_targets_for_variant(variant: str) -> dict[str, float]:
    if variant == "accuracy_recovery_v2":
        return {
            "overall": 75,
            "strict": 150,
            "meaning": 55,
            "surface": 59,
            "reading": 58,
            "mcq_source": 95,
            "evidence": 0.92,
        }
    return {
        "overall": 72,
        "strict": 142,
        "meaning": 52,
        "surface": 58,
        "reading": 58,
        "mcq_source": 90,
        "evidence": 0.885,
    }


def _gate_count(value: list[float]) -> str:
    matched, expected = value
    return f"{int(matched)}/{int(expected)}" if expected else "n/a"


def _vocab_alignment_payload(benchmark: dict[str, Any]) -> dict[str, Any]:
    metrics = benchmark.get("extraction_variant_metrics") if isinstance(benchmark.get("extraction_variant_metrics"), dict) else {}
    alignment = metrics.get("vocab_alignment") if isinstance(metrics.get("vocab_alignment"), dict) else {}
    return alignment


def _recovery_payload(benchmark: dict[str, Any]) -> dict[str, Any]:
    metrics = benchmark.get("extraction_variant_metrics") if isinstance(benchmark.get("extraction_variant_metrics"), dict) else {}
    recovery = metrics.get("recovery") if isinstance(metrics.get("recovery"), dict) else {}
    return recovery


def _recovery_label(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "n/a"
    attempted = int(_numeric(value.get("attempted")))
    accepted = int(_numeric(value.get("accepted")))
    counts = value.get("counts") if isinstance(value.get("counts"), dict) else {}
    rejected = counts.get("rejected_by_consensus") if isinstance(counts, dict) else None
    suffix = f", rejected:{rejected}" if isinstance(rejected, int) and rejected else ""
    return f"{accepted}/{attempted}{suffix}" if attempted or accepted else "n/a"


def _shadow_rows_label(alignment: dict[str, Any]) -> str:
    total = alignment.get("shadow_row_count")
    if not isinstance(total, (int, float)):
        return "n/a"
    return f"{int(_numeric(alignment.get('shadow_complete_row_count')))}/{int(total)}"


def _risk_label(alignment: dict[str, Any]) -> str:
    risk = alignment.get("risk_level")
    return risk if isinstance(risk, str) and risk else "n/a"


def _risk_counts_label(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "n/a"
    order = ["high", "medium", "low", "not_applicable"]
    parts = [f"{risk}:{value[risk]}" for risk in order if risk in value]
    parts.extend(f"{risk}:{count}" for risk, count in sorted(value.items()) if risk not in order)
    return ", ".join(parts)


def _miss_counts(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or not isinstance(value.get("counts"), dict):
        return {}
    return {str(reason): int(count) for reason, count in value["counts"].items() if isinstance(count, int) and count}


def _miss_counts_label(value: object) -> str:
    counts = _miss_counts(value)
    if not counts:
        return "none"
    order = [
        "korean_ocr_error",
        "surface_ocr_error",
        "reading_ocr_error",
        "wrong_pairing",
        "missing_row",
        "source_field_ocr_error",
        "source_question_mismatch",
        "missing_question",
    ]
    parts = [f"{reason}:{counts[reason]}" for reason in order if reason in counts]
    parts.extend(f"{reason}:{count}" for reason, count in sorted(counts.items()) if reason not in order)
    return ", ".join(parts)


def _field_error_counts(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or not isinstance(value.get("field_error_counts"), dict):
        return {}
    return {str(field): int(count) for field, count in value["field_error_counts"].items() if isinstance(count, int) and count}


def _field_error_counts_label(value: object) -> str:
    counts = _field_error_counts(value)
    if not counts:
        return "none"
    order = ["sentence", "target", "choices", "correct_answer", "correct_choice_no"]
    parts = [f"{field}:{counts[field]}" for field in order if field in counts]
    parts.extend(f"{field}:{count}" for field, count in sorted(counts.items()) if field not in order)
    return ", ".join(parts)


def _strict_score_counts(base: dict[str, Any]) -> tuple[float, float]:
    source_expected = _numeric(base.get("source_field_expected"))
    if source_expected:
        return _numeric(base.get("source_field_matches")), source_expected
    return _numeric(base.get("matched")), _numeric(base.get("expected"))


def _field_summary_accuracy(group: dict[str, Any], field: str) -> float | None:
    expected = float(group.get(f"{field}_expected", 0.0))
    if not expected:
        return None
    return float(group.get(f"{field}_matched", 0.0)) / expected


def _numeric(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _percent(value: object) -> str:
    return f"{float(value) * 100:.1f}%" if isinstance(value, (int, float)) else "n/a"


def _generated_count(payload: dict[str, Any]) -> str:
    if "generated_notes" in payload:
        return f"notes={payload.get('generated_notes', 0)}"
    return f"cards={payload.get('generated_cards', 0)}"


if __name__ == "__main__":
    raise SystemExit(main())
