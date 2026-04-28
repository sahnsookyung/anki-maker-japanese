from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from queue import Queue
import os
import re
import subprocess
import sys
import time
from threading import Lock, Thread, Timer
from typing import Any
from uuid import uuid4

from PIL import Image, ImageOps

from app.core.config import (
    BACKEND_DIR,
    CROP_DIR,
    OCR_CROP_JOB_TIMEOUT_SECONDS,
    OCR_CROP_MAX_SIDE,
    OCR_CROP_MIN_SIDE,
    OCR_CROP_WORKER_IDLE_SECONDS,
    OCR_CROP_WORKER_MAX_RSS_MB,
    OCR_PROVIDER,
)
from app.extraction.sentence_order import repair_predicate_first_sentence
from app.extraction.geometry import group_tokens_by_line
from app.models.schemas import BBox, FieldOcrPreviewResponse, OcrRuntimeStatus, OcrToken


class CropOcrError(RuntimeError):
    pass


class CropOcrMemoryError(CropOcrError):
    pass


@dataclass
class CropPreview:
    response: FieldOcrPreviewResponse


class CropOcrWorkerManager:
    def __init__(
        self,
        *,
        idle_seconds: float = OCR_CROP_WORKER_IDLE_SECONDS,
        max_rss_mb: float = OCR_CROP_WORKER_MAX_RSS_MB,
        job_timeout_seconds: float = OCR_CROP_JOB_TIMEOUT_SECONDS,
    ) -> None:
        self.idle_seconds = idle_seconds
        self.max_rss_mb = max_rss_mb
        self.job_timeout_seconds = job_timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._lock = Lock()
        self._idle_timer: Timer | None = None
        self._idle_deadline: datetime | None = None
        self._loaded_provider: str | None = None
        self._busy = False
        self._jobs_handled = 0
        self._last_error: str | None = None

    def preview(
        self,
        *,
        image_path: Path,
        page_id: str,
        card_id: str,
        source: dict[str, Any],
        field: str,
        bbox: BBox,
        page_width: int,
        page_height: int,
    ) -> FieldOcrPreviewResponse:
        normalized_bbox = normalize_bbox(bbox, page_width, page_height)
        crop_path, crop_offset = crop_image(image_path, normalized_bbox, page_id, card_id, field)
        provider = provider_for_field(field)
        warnings: list[str] = []
        try:
            payload = self._dispatch({"image_path": str(crop_path), "page_id": page_id, "provider": provider})
        finally:
            crop_path.unlink(missing_ok=True)
        if not payload.get("ok"):
            raise CropOcrError(str(payload.get("error") or "Crop OCR worker failed."))
        warnings.extend(str(warning) for warning in payload.get("warnings", []))
        crop_tokens = [OcrToken(**token) for token in payload.get("tokens", [])]
        mapped_tokens = [_map_token_to_page(token, page_id, crop_offset) for token in crop_tokens]
        raw_text = recognized_text(field, mapped_tokens)
        text, text_warnings = select_preview_text(field, raw_text, source)
        warnings.extend(text_warnings)
        confidence = _mean_confidence(mapped_tokens)
        field_evidence = {
            "bbox": normalized_bbox,
            "token_ids": [token.id for token in mapped_tokens],
            "text": text,
            "confidence": confidence,
            "provenance": "crop_ocr",
            "provider": provider,
        }
        if raw_text and raw_text != text:
            field_evidence["raw_text"] = raw_text
        if not text:
            warnings.append("Crop OCR returned no text for this field.")
        worker_status = self.status().model_dump()
        return FieldOcrPreviewResponse(
            card_id=card_id,
            page_id=page_id,
            field=field,
            bbox=normalized_bbox,
            provider=provider,
            text=text,
            confidence=confidence,
            tokens=mapped_tokens,
            suggested_source=suggested_source_patch(source, field, text),
            field_evidence=field_evidence,
            worker=worker_status,
            warnings=warnings,
        )

    def status(self) -> OcrRuntimeStatus:
        process = self._process
        if process is None or process.poll() is not None:
            state = "stopped"
            pid = None
        else:
            state = "busy" if self._busy else "running"
            pid = process.pid
        return OcrRuntimeStatus(
            state=state,
            pid=pid,
            loaded_provider=self._loaded_provider,
            idle_deadline=self._idle_deadline.isoformat() if self._idle_deadline else None,
            current_rss_mb=_process_rss_mb(pid) if pid else None,
            jobs_handled=self._jobs_handled,
            last_error=self._last_error,
        )

    def offload_if_idle(self) -> bool:
        with self._lock:
            if self._busy or not self._process:
                return False
            if self._idle_deadline and datetime.now(timezone.utc) < self._idle_deadline:
                return False
            self._terminate_locked("idle timeout")
            return True

    def force_offload(self) -> None:
        with self._lock:
            self._terminate_locked("manual offload")

    def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._busy = True
            try:
                return self._dispatch_locked(request)
            finally:
                self._busy = False
                self._schedule_idle_offload_locked()

    def _dispatch_locked(self, request: dict[str, Any]) -> dict[str, Any]:
        process = self._ensure_process_locked()
        provider = str(request.get("provider") or "")
        try:
            assert process.stdin is not None
            process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            process.stdin.flush()
            payload = _read_json_line(process, self.job_timeout_seconds, self.max_rss_mb)
        except CropOcrMemoryError as exc:
            self._last_error = str(exc)
            self._terminate_locked("RSS limit exceeded")
            raise
        except Exception as exc:
            self._last_error = str(exc)
            self._terminate_locked("worker communication failure")
            process = self._ensure_process_locked()
            assert process.stdin is not None
            process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            process.stdin.flush()
            payload = _read_json_line(process, self.job_timeout_seconds, self.max_rss_mb)
        self._jobs_handled += 1
        self._loaded_provider = provider
        rss_mb = _process_rss_mb(process.pid)
        if rss_mb is not None and rss_mb > self.max_rss_mb:
            payload.setdefault("warnings", []).append(
                f"Crop OCR worker exceeded {self.max_rss_mb:.0f} MB RSS and was offloaded after this preview."
            )
            self._terminate_locked("RSS limit exceeded")
        return payload

    def _ensure_process_locked(self) -> subprocess.Popen[str]:
        if self._process and self._process.poll() is None:
            return self._process
        self._process = self._start_process()
        self._loaded_provider = None
        self._last_error = None
        return self._process

    def _start_process(self) -> subprocess.Popen[str]:
        env = os.environ.copy()
        pythonpath = str(BACKEND_DIR)
        if env.get("PYTHONPATH"):
            pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
        env["PYTHONPATH"] = pythonpath
        return subprocess.Popen(
            [sys.executable, "-m", "app.ocr.crop_worker_process"],
            cwd=str(BACKEND_DIR),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def _schedule_idle_offload_locked(self) -> None:
        if self._idle_timer:
            self._idle_timer.cancel()
        self._idle_deadline = datetime.now(timezone.utc) + timedelta(seconds=self.idle_seconds)
        self._idle_timer = Timer(self.idle_seconds, self.offload_if_idle)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _terminate_locked(self, reason: str) -> None:
        if self._idle_timer:
            self._idle_timer.cancel()
            self._idle_timer = None
        self._idle_deadline = None
        process = self._process
        self._process = None
        self._loaded_provider = None
        if not process or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        self._last_error = reason


def provider_for_field(field: str) -> str:
    return "paddle_korean" if field == "meaning_ko" else OCR_PROVIDER


def normalize_bbox(bbox: BBox, page_width: int, page_height: int) -> BBox:
    if len(bbox) != 4 or not all(isinstance(value, (int, float)) for value in bbox):
        raise ValueError("bbox must contain four numeric values.")
    x1, y1, x2, y2 = [float(value) for value in bbox]
    left, right = sorted((max(0.0, min(float(page_width), x1)), max(0.0, min(float(page_width), x2))))
    top, bottom = sorted((max(0.0, min(float(page_height), y1)), max(0.0, min(float(page_height), y2))))
    width = right - left
    height = bottom - top
    if width < OCR_CROP_MIN_SIDE or height < OCR_CROP_MIN_SIDE:
        raise ValueError(f"bbox must be at least {OCR_CROP_MIN_SIDE}px wide and tall.")
    if max(width, height) > OCR_CROP_MAX_SIDE:
        raise ValueError(f"bbox is too large for field OCR; keep the longest side under {OCR_CROP_MAX_SIDE}px.")
    return [left, top, right, bottom]


def crop_image(image_path: Path, bbox: BBox, page_id: str, card_id: str, field: str) -> tuple[Path, tuple[float, float]]:
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        x1, y1, x2, y2 = bbox
        crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
        crop_path = CROP_DIR / f"{page_id}_{card_id}_{field}_{uuid4().hex}.png"
        crop.save(crop_path)
    return crop_path, (float(x1), float(y1))


def recognized_text(field: str, tokens: list[OcrToken]) -> str:
    ordered = _tokens_in_reading_order(tokens)
    separator = " " if field == "meaning_ko" else ""
    return separator.join(token.text for token in ordered if token.text).strip()


def select_preview_text(field: str, raw_text: str, source: dict[str, Any]) -> tuple[str, list[str]]:
    if field != "sentence" or not raw_text:
        return raw_text, []
    repaired_text, repaired = repair_predicate_first_sentence(raw_text)
    if repaired:
        return repaired_text, ["Crop OCR returned predicate-first sentence fragments; reordered them into Japanese reading order."]
    existing_text = _existing_source_text(source, field)
    if not existing_text:
        return raw_text, []
    existing_normalized = _normalize_preview_text(existing_text)
    raw_normalized = _normalize_preview_text(raw_text)
    if not existing_normalized or not raw_normalized or existing_normalized == raw_normalized:
        return raw_text, []
    if _same_sentence_characters(existing_normalized, raw_normalized):
        return existing_text, ["Crop OCR returned sentence fragments in a different order; kept page-level sentence order."]
    return raw_text, []


def suggested_source_patch(source: dict[str, Any], field: str, text: str) -> dict[str, Any]:
    if not text:
        return {}
    if field.startswith("choice_"):
        try:
            choice_index = int(field.split("_", 1)[1]) - 1
        except ValueError:
            return {}
        choices = list(source.get("choices") if isinstance(source.get("choices"), list) else [])
        while len(choices) < 4:
            choices.append("")
        choices[choice_index] = text
        patch: dict[str, Any] = {"choices": choices}
        if source.get("correct_choice_no") == choice_index + 1:
            patch["correct_answer"] = text
        return patch
    if field == "question_no":
        return {field: int(text)} if text.isdigit() else {field: text}
    if field in {"sentence", "target", "correct_answer", "surface", "reading", "meaning_ko", "answer_source"}:
        return {field: text}
    return {}


def _map_token_to_page(token: OcrToken, page_id: str, offset: tuple[float, float]) -> OcrToken:
    x_offset, y_offset = offset
    x1, y1, x2, y2 = token.bbox
    return token.model_copy(
        update={
            "page_id": page_id,
            "bbox": [x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset],
        }
    )


def _tokens_in_reading_order(tokens: list[OcrToken]) -> list[OcrToken]:
    if len(tokens) < 2:
        return tokens
    heights = [max(1.0, abs(token.bbox[3] - token.bbox[1])) for token in tokens]
    tolerance = max(12.0, _median(heights) * 0.85)
    return [token for line in group_tokens_by_line(tokens, tolerance=tolerance) for token in line]


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _existing_source_text(source: dict[str, Any], field: str) -> str:
    direct = source.get(field)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    evidence = source.get("field_evidence")
    if not isinstance(evidence, dict):
        return ""
    field_evidence = evidence.get(field)
    if not isinstance(field_evidence, dict):
        return ""
    text = field_evidence.get("text")
    return text.strip() if isinstance(text, str) else ""


def _normalize_preview_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()


def _same_sentence_characters(expected: str, actual: str) -> bool:
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    matched = sum(min(count, actual_counts[character]) for character, count in expected_counts.items())
    if matched == 0:
        return False
    expected_coverage = matched / len(expected)
    actual_coverage = matched / len(actual)
    return expected_coverage >= 0.9 and actual_coverage >= 0.85


def _mean_confidence(tokens: list[OcrToken]) -> float:
    if not tokens:
        return 0.0
    return round(sum(token.confidence for token in tokens) / len(tokens), 3)


def _read_json_line(process: subprocess.Popen[str], timeout: float, max_rss_mb: float) -> dict[str, Any]:
    if process.stdout is None:
        raise CropOcrError("Crop OCR worker stdout is unavailable.")
    queue: Queue[dict[str, Any] | Exception] = Queue()

    def read_payload() -> None:
        try:
            for line in process.stdout:
                try:
                    queue.put(json.loads(line))
                    return
                except json.JSONDecodeError:
                    continue
            queue.put(CropOcrError("Crop OCR worker exited without a response."))
        except Exception as exc:
            queue.put(exc)

    thread = Thread(target=read_payload, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    while thread.is_alive():
        rss_mb = _process_rss_mb(process.pid)
        if max_rss_mb > 0 and rss_mb is not None and rss_mb > max_rss_mb:
            process.terminate()
            raise CropOcrMemoryError(f"Crop OCR worker exceeded {max_rss_mb:.0f} MB RSS during this preview.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process.terminate()
            raise TimeoutError(f"Crop OCR worker timed out after {timeout:.0f}s.")
        thread.join(timeout=min(0.25, remaining))
    payload = queue.get_nowait()
    if isinstance(payload, Exception):
        raise payload
    return payload


def _process_rss_mb(pid: int | None) -> float | None:
    if not pid:
        return None
    try:
        output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
    except Exception:
        return None
    if not output:
        return None
    return round(int(output) / 1024.0, 2)


crop_ocr_worker = CropOcrWorkerManager()
