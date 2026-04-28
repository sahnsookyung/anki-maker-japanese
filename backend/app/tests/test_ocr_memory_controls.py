from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
import json
import subprocess
import time

import pytest
from PIL import Image

from app.db import database
from app.evaluation.golden import GoldenPage
from app.extraction import pipeline
from app.models.schemas import OcrToken, Page
from app.ocr import service
from app.ocr import crop_worker
from app.ocr import page_worker
from app.ocr.crop_worker import CropOcrWorkerManager
from app.ocr.engines import OcrEngineResult
from scripts import benchmark_ocr_modes


class DummyProvider:
    name = "dummy"

    def recognize(self, image_path: Path, page_id: str):  # pragma: no cover - not needed for cache tests
        return []


def test_ocr_provider_cache_can_be_disabled(monkeypatch) -> None:
    calls = 0

    def fake_builder(provider_name: str):
        nonlocal calls
        calls += 1
        return DummyProvider()

    service._build_ocr_provider_cached.cache_clear()
    monkeypatch.setattr(service, "_build_ocr_provider_uncached", fake_builder)
    monkeypatch.setattr(service, "OCR_PROVIDER_CACHE_ENABLED", True)

    first = service.get_ocr_provider("dummy")
    second = service.get_ocr_provider("dummy")

    assert first is second
    assert calls == 1

    monkeypatch.setattr(service, "OCR_PROVIDER_CACHE_ENABLED", False)

    third = service.get_ocr_provider("dummy")
    fourth = service.get_ocr_provider("dummy")

    assert third is not fourth
    assert calls == 3


def test_benchmark_runtime_uses_isolated_state_and_restores_globals(tmp_path) -> None:
    original_db_path = database.DB_PATH
    original_processed_dir = pipeline.PROCESSED_DIR
    work_dir = tmp_path / "bench"

    with benchmark_ocr_modes._benchmark_runtime(str(work_dir), keep_work_dir=True) as runtime_dir:
        assert runtime_dir == work_dir.resolve()
        assert database.DB_PATH == work_dir.resolve() / "benchmark.db"
        assert pipeline.PROCESSED_DIR == work_dir.resolve() / "processed"
        assert pipeline.PROCESSED_DIR.exists()

    assert database.DB_PATH == original_db_path
    assert pipeline.PROCESSED_DIR == original_processed_dir
    assert work_dir.exists()


def test_benchmark_runtime_cleans_default_temp_state() -> None:
    with benchmark_ocr_modes._benchmark_runtime("", keep_work_dir=False) as runtime_dir:
        assert runtime_dir.exists()
        temp_path = runtime_dir

    assert not temp_path.exists()


def test_process_page_preserves_upload_name_in_result(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "pipeline.db")
    monkeypatch.setattr(pipeline, "PROCESSED_DIR", tmp_path / "processed")
    database.init_db()
    page = Page(
        id="page-upload-name",
        original_image_path=str(tmp_path / "uploaded.jpg"),
        upload_name="Original upload.jpg",
        display_name="Renamed page",
        page_type="uploaded",
        page_type_confidence=0.0,
        warnings=[],
        created_at="2026-04-28T00:00:00+00:00",
    )

    monkeypatch.setattr(
        pipeline,
        "preprocess_image",
        lambda original, processed: SimpleNamespace(width=100, height=100, warnings=[]),
    )
    monkeypatch.setattr(pipeline, "run_ocr_engine", lambda image_path, page_id, engine: OcrEngineResult(engine=engine, tokens=[], warnings=[]))
    monkeypatch.setattr(pipeline, "classify_page", lambda tokens, height: ("unknown_review_required", 0.0, {}))
    monkeypatch.setattr(pipeline, "parse_answer_strip", lambda tokens, height: {})

    result = pipeline.process_page(page)

    assert result.page.upload_name == "Original upload.jpg"
    assert database.get_page("page-upload-name").upload_name == "Original upload.jpg"


def test_benchmark_default_runner_uses_per_page_subprocesses(tmp_path, monkeypatch) -> None:
    commands: list[list[str]] = []
    golden = GoldenPage(
        page_id="page-one",
        image_path=tmp_path / "page.jpg",
        category="vocab_table",
        expected_page_type="vocab_table",
    )
    args = SimpleNamespace(
        work_dir=str(tmp_path / "work"),
        keep_work_dir=False,
        include_vl=True,
        vl_limit=1,
    )

    def fake_run(cmd, args):
        commands.append(list(cmd))
        output_path = Path(cmd[cmd.index("--output-json") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                [
                    {
                        "page_id": "page-one",
                        "image_path": str(golden.image_path),
                        "base": {
                            "mode": "base_paddleocr_extraction",
                            "actual_page_type": "vocab_table",
                            "matched": 0,
                            "expected": 0,
                            "accuracy": 0.0,
                            "generated_cards": 0,
                            "missing_ids": [],
                        },
                        "vl": None,
                        "memory_samples": [{"stage": "worker_start", "rss_mb": 10.0, "peak_rss_mb": 10.0}],
                        "resource_metrics": {
                            "wall_seconds": 1.0,
                            "user_cpu_seconds": 0.5,
                            "system_cpu_seconds": 0.25,
                            "cpu_seconds": 0.75,
                            "cpu_percent_of_one_core": 75.0,
                            "peak_rss_mb": 10.0,
                            "rss_samples": [{"stage": "worker_start", "rss_mb": 10.0, "peak_rss_mb": 10.0}],
                            "npu": {
                                "available": False,
                                "utilization_percent": None,
                                "memory_mb": None,
                                "note": "not reported",
                            },
                            "gpu": {
                                "available": False,
                                "utilization_percent": None,
                                "memory_mb": None,
                                "note": "not reported",
                            },
                        },
                        "errors": [],
                    }
                ]
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(benchmark_ocr_modes, "_run_worker_command", fake_run)

    results = benchmark_ocr_modes._run_pages_in_subprocesses(args, tmp_path / "golden.json", [golden])

    assert results[0].page_id == "page-one"
    assert "--worker-page-id" in commands[0]
    assert "--in-process" in commands[0]
    assert commands[0][commands[0].index("--engine") + 1] == "paddleocr"
    assert commands[1][commands[1].index("--engine") + 1] == "paddleocr_vl"
    assert results[0].vl is not None
    assert results[0].resource_metrics["cpu_seconds"] == pytest.approx(0.75)


def test_page_worker_command_enforces_rss_limit(monkeypatch) -> None:
    process = FakeProcess()
    monkeypatch.setattr(page_worker.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(page_worker, "_process_tree_rss_mb", lambda pid: 10.0)

    completed = page_worker._run_worker_command(["python", "-m", "worker"], timeout_seconds=30, max_rss_mb=5)

    assert completed.returncode != 0
    assert process.terminated is True
    assert "RSS limit" in completed.stderr


def test_resource_metrics_include_cpu_ram_and_device_notes() -> None:
    metrics = benchmark_ocr_modes._resource_metrics(
        {"wall_seconds": 1.0, "user_cpu_seconds": 2.0, "system_cpu_seconds": 0.5},
        {"wall_seconds": 3.0, "user_cpu_seconds": 3.0, "system_cpu_seconds": 1.0},
        [{"stage": "after_base", "rss_mb": 20.0, "peak_rss_mb": 30.0}],
    )

    assert metrics["wall_seconds"] == pytest.approx(2.0)
    assert metrics["cpu_seconds"] == pytest.approx(1.5)
    assert metrics["cpu_percent_of_one_core"] == pytest.approx(75.0)
    assert metrics["peak_rss_mb"] == pytest.approx(30.0)
    assert metrics["npu"]["available"] is False


def test_crop_ocr_helpers_map_fields_and_tokens() -> None:
    token = OcrToken(
        id="tok-crop",
        page_id="crop",
        text="うえ",
        bbox=[2, 3, 20, 10],
        confidence=0.91,
        script_class="hiragana",
        source="paddleocr",
    )

    mapped = crop_worker._map_token_to_page(token, "page", (100, 200))

    assert crop_worker.provider_for_field("meaning_ko") == "paddle_korean"
    assert mapped.page_id == "page"
    assert mapped.bbox == [102, 203, 120, 210]
    assert crop_worker.recognized_text("target", [mapped]) == "うえ"
    assert crop_worker.suggested_source_patch({"choices": ["a"], "correct_choice_no": 2}, "choice_2", "b") == {
        "choices": ["a", "b", "", ""],
        "correct_answer": "b",
    }


def test_crop_ocr_orders_slanted_japanese_sentence_left_to_right() -> None:
    tokens = [
        OcrToken(
            id="right",
            page_id="page",
            text="さきました。",
            bbox=[190, 6, 280, 24],
            confidence=0.92,
            script_class="hiragana",
            source="paddleocr",
        ),
        OcrToken(
            id="middle",
            page_id="page",
            text="はなが",
            bbox=[120, 12, 180, 30],
            confidence=0.93,
            script_class="hiragana",
            source="paddleocr",
        ),
        OcrToken(
            id="left",
            page_id="page",
            text="にわにしろい",
            bbox=[20, 18, 110, 36],
            confidence=0.94,
            script_class="hiragana",
            source="paddleocr",
        ),
    ]

    assert crop_worker.recognized_text("sentence", tokens) == "にわにしろいはながさきました。"


def test_sentence_preview_keeps_page_level_order_for_reversed_crop_text() -> None:
    expected = "にわにしろいはながさきました。"

    text, warnings = crop_worker.select_preview_text(
        "sentence",
        "さきました。はながにわにしろい",
        {"sentence": expected},
    )
    target_text, target_warnings = crop_worker.select_preview_text(
        "target",
        "さきました。はながにわにしろい",
        {"target": "はな"},
    )

    assert text == expected
    assert warnings == ["Crop OCR returned predicate-first sentence fragments; reordered them into Japanese reading order."]
    assert target_text == "さきました。はながにわにしろい"
    assert target_warnings == []


def test_crop_preview_response_uses_corrected_sentence_order(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (640, 1600), "white").save(image_path)
    expected = "にわにしろいはながさきました。"
    manager = CropOcrWorkerManager(idle_seconds=999, max_rss_mb=5000, job_timeout_seconds=1)
    monkeypatch.setattr(crop_worker, "CROP_DIR", tmp_path / "crops")
    monkeypatch.setattr(crop_worker, "_process_rss_mb", lambda pid: 10.0)

    def fake_dispatch(request):
        return {
            "ok": True,
            "provider": "auto",
            "tokens": [
                {
                    "id": "reversed-line",
                    "page_id": "crop",
                    "text": "さきました。はながにわにしろい",
                    "bbox": [0, 0, 350, 20],
                    "confidence": 0.74,
                    "script_class": "hiragana",
                    "source": "paddleocr",
                },
            ],
            "warnings": [],
        }

    monkeypatch.setattr(manager, "_dispatch", fake_dispatch)

    preview = manager.preview(
        image_path=image_path,
        page_id="page-cat4",
        card_id="card-q10",
        source={"sentence": expected},
        field="sentence",
        bbox=[105, 1419, 508, 1454],
        page_width=640,
        page_height=1600,
    )

    assert preview.text == expected
    assert preview.suggested_source == {"sentence": expected}
    assert preview.field_evidence["text"] == expected
    assert preview.field_evidence["raw_text"] == "さきました。はながにわにしろい"
    assert preview.warnings == ["Crop OCR returned predicate-first sentence fragments; reordered them into Japanese reading order."]
    assert preview.tokens[0].bbox == [105.0, 1419.0, 455.0, 1439.0]
    assert list((tmp_path / "crops").glob("*.png")) == []


def test_crop_ocr_bbox_validation_rejects_unsafe_regions() -> None:
    assert crop_worker.normalize_bbox([30, 20, 10, 5], 100, 80) == [10.0, 5.0, 30.0, 20.0]

    for bbox in ([1, 1, 2, 2], [1, 2, 3]):
        try:
            crop_worker.normalize_bbox(bbox, 100, 80)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bbox} should be rejected")
    try:
        crop_worker.normalize_bbox([0, 0, 3000, 20], 4000, 80)
    except ValueError:
        pass
    else:
        raise AssertionError("oversized crop should be rejected")


def test_crop_worker_reuses_offloads_and_reloads(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "page.png"
    crop_dir = tmp_path / "crops"
    Image.new("RGB", (100, 80), "white").save(image_path)
    started: list[FakeProcess] = []

    manager = CropOcrWorkerManager(idle_seconds=999, max_rss_mb=5000, job_timeout_seconds=1)
    monkeypatch.setattr(crop_worker, "CROP_DIR", crop_dir)
    monkeypatch.setattr(crop_worker, "_process_rss_mb", lambda pid: 10.0)

    def start_process():
        process = FakeProcess()
        started.append(process)
        return process

    monkeypatch.setattr(manager, "_start_process", start_process)

    first = manager.preview(
        image_path=image_path,
        page_id="page",
        card_id="card",
        source={"target": "上"},
        field="target",
        bbox=[5, 5, 60, 25],
        page_width=100,
        page_height=80,
    )
    second = manager.preview(
        image_path=image_path,
        page_id="page",
        card_id="card",
        source={"target": "上"},
        field="target",
        bbox=[5, 5, 60, 25],
        page_width=100,
        page_height=80,
    )

    assert len(started) == 1
    assert first.text == "うえ"
    assert second.text == "うえ"
    manager._idle_deadline = datetime.now(timezone.utc) - timedelta(seconds=1)
    assert manager.offload_if_idle() is True
    assert started[0].terminated is True

    manager.preview(
        image_path=image_path,
        page_id="page",
        card_id="card",
        source={"target": "上"},
        field="target",
        bbox=[5, 5, 60, 25],
        page_width=100,
        page_height=80,
    )

    assert len(started) == 2


def test_crop_worker_memory_limit_offloads_after_response(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    manager = CropOcrWorkerManager(idle_seconds=999, max_rss_mb=5, job_timeout_seconds=1)
    process = FakeProcess()
    monkeypatch.setattr(crop_worker, "CROP_DIR", tmp_path / "crops")
    monkeypatch.setattr(crop_worker, "_process_rss_mb", lambda pid: 10.0)
    monkeypatch.setattr(manager, "_start_process", lambda: process)

    preview = manager.preview(
        image_path=image_path,
        page_id="page",
        card_id="card",
        source={"target": "上"},
        field="target",
        bbox=[5, 5, 60, 25],
        page_width=100,
        page_height=80,
    )

    assert process.terminated is True
    assert "offloaded" in " ".join(preview.warnings)


def test_crop_worker_memory_limit_kills_before_response(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    manager = CropOcrWorkerManager(idle_seconds=999, max_rss_mb=5, job_timeout_seconds=5)
    process = FakeProcess(stdout=SlowStdout())
    monkeypatch.setattr(crop_worker, "CROP_DIR", tmp_path / "crops")
    monkeypatch.setattr(crop_worker, "_process_rss_mb", lambda pid: 10.0)
    monkeypatch.setattr(manager, "_start_process", lambda: process)

    try:
        manager.preview(
            image_path=image_path,
            page_id="page",
            card_id="card",
            source={"target": "上"},
            field="target",
            bbox=[5, 5, 60, 25],
            page_width=100,
            page_height=80,
        )
    except crop_worker.CropOcrMemoryError as exc:
        assert "exceeded" in str(exc)
    else:
        raise AssertionError("crop worker should be killed while waiting for an oversized job")
    assert process.terminated is True


class FakeStdin:
    def __init__(self) -> None:
        self.writes: list[str] = []

    def write(self, value: str) -> None:
        self.writes.append(value)

    def flush(self) -> None:
        return None


class FakeStdout:
    def __iter__(self):
        return self

    def __next__(self) -> str:
        return json.dumps(
            {
                "ok": True,
                "provider": "auto",
                "tokens": [
                    {
                        "id": "tok-worker",
                        "page_id": "page",
                        "text": "うえ",
                        "bbox": [1, 2, 20, 12],
                        "confidence": 0.95,
                        "script_class": "hiragana",
                        "source": "paddleocr",
                    }
                ],
                "warnings": [],
            }
        ) + "\n"


class SlowStdout:
    def __iter__(self):
        return self

    def __next__(self) -> str:
        time.sleep(10)
        return ""


class FakeProcess:
    pid = 4242

    def __init__(self, stdout=None) -> None:
        self.stdin = FakeStdin()
        self.stdout = stdout or FakeStdout()
        self.terminated = False
        self.returncode = None

    def poll(self):
        return 0 if self.terminated else None

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.terminated = True
        self.returncode = -9

    def wait(self, timeout=None):
        return 0

    def communicate(self, timeout=None):
        return "", ""
