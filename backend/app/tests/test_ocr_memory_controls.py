from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import subprocess

from app.db import database
from app.evaluation.golden import GoldenPage
from app.extraction import pipeline
from app.ocr import service
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

    def fake_run(cmd, check, capture_output, text):
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

    monkeypatch.setattr(benchmark_ocr_modes.subprocess, "run", fake_run)

    results = benchmark_ocr_modes._run_pages_in_subprocesses(args, tmp_path / "golden.json", [golden])

    assert results[0].page_id == "page-one"
    assert "--worker-page-id" in commands[0]
    assert "--in-process" in commands[0]
    assert "--include-vl" in commands[0]
    assert results[0].resource_metrics["cpu_seconds"] == 0.75


def test_resource_metrics_include_cpu_ram_and_device_notes() -> None:
    metrics = benchmark_ocr_modes._resource_metrics(
        {"wall_seconds": 1.0, "user_cpu_seconds": 2.0, "system_cpu_seconds": 0.5},
        {"wall_seconds": 3.0, "user_cpu_seconds": 3.0, "system_cpu_seconds": 1.0},
        [{"stage": "after_base", "rss_mb": 20.0, "peak_rss_mb": 30.0}],
    )

    assert metrics["wall_seconds"] == 2.0
    assert metrics["cpu_seconds"] == 1.5
    assert metrics["cpu_percent_of_one_core"] == 75.0
    assert metrics["peak_rss_mb"] == 30.0
    assert metrics["npu"]["available"] is False
