from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from app.vision import paddle_ocr_vl
from app.ocr import engines
from app.vision.paddle_ocr_vl import (
    PaddleOcrVlDocumentParser,
    _blocks_from_payload,
    _pipeline_creation_help,
    _normalized_vl_backend,
)
from app.models.schemas import DocumentParseBlock, DocumentParseResult
from app.ocr.engines import normalize_ocr_engine, run_ocr_engine


def test_blocks_from_paddle_ocr_vl_payload() -> None:
    payload = {
        "parsing_res_list": [
            {
                "block_label": "text",
                "block_content": "学校 がっこう 학교",
                "block_bbox": [10, 20, 200, 60],
                "block_order": 1,
            }
        ]
    }

    blocks = _blocks_from_payload(payload)

    assert len(blocks) == 1
    assert blocks[0].label == "text"
    assert blocks[0].content == "学校 がっこう 학교"
    assert blocks[0].bbox == [10, 20, 200, 60]
    assert blocks[0].order == 1


def test_paddle_ocr_vl_processing_returns_document_parse_without_base_geometry(monkeypatch, tmp_path) -> None:
    parsed = DocumentParseResult(
        page_id="page-vl",
        provider="paddleocr_vl",
        source_image_path=str(tmp_path / "page.png"),
        backend="fake",
        block_count=1,
        blocks=[
            DocumentParseBlock(
                label="text",
                content="2 \\underline{\\text{あたらしい}}",
                bbox=[20, 1100, 900, 1150],
                order=1,
            )
        ],
    )
    class FakeParser:
        def parse(self, image_path, page_id):
            return parsed

    monkeypatch.setattr(engines, "get_paddle_ocr_vl_parser", lambda: FakeParser())
    monkeypatch.setattr(
        engines,
        "recognize_with_warnings",
        lambda image_path, page_id: pytest.fail("PaddleOCR-VL processing must not borrow PaddleOCR word geometry."),
    )

    result = run_ocr_engine(tmp_path / "page.png", "page-vl", "paddleocr_vl")

    assert result.tokens == []
    assert result.evidence_tokens == []
    assert result.document_parse == parsed
    assert any("document blocks" in warning for warning in result.warnings)
    assert not any("visual evidence uses PaddleOCR word boxes" in warning for warning in result.warnings)


def test_paddle_ocr_vl_processing_warns_when_parse_has_no_text(monkeypatch, tmp_path) -> None:
    parsed = DocumentParseResult(
        page_id="page-vl",
        provider="paddleocr_vl",
        source_image_path=str(tmp_path / "page.png"),
        backend="fake",
        block_count=0,
        blocks=[],
    )

    class FakeParser:
        def parse(self, image_path, page_id):
            return parsed

    monkeypatch.setattr(engines, "get_paddle_ocr_vl_parser", lambda: FakeParser())

    result = run_ocr_engine(tmp_path / "page.png", "page-vl", "paddleocr_vl")

    assert result.tokens == []
    assert result.document_parse == parsed
    assert any("produced no document text" in warning for warning in result.warnings)


def test_normalize_ocr_engine_aliases() -> None:
    assert normalize_ocr_engine("base") == "paddleocr"
    assert normalize_ocr_engine("vl") == "paddleocr_vl"
    with pytest.raises(ValueError, match="Unsupported OCR engine"):
        normalize_ocr_engine("mystery")


def test_native_backend_alias_uses_local_paddlepaddle_default(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakePaddleOCRVL:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCRVL=FakePaddleOCRVL))
    monkeypatch.setattr(paddle_ocr_vl, "PADDLE_OCR_VL_BACKEND", "native")
    monkeypatch.setattr(paddle_ocr_vl, "PADDLE_OCR_VL_SERVER_URL", "")
    monkeypatch.setattr(paddle_ocr_vl, "PADDLE_OCR_VL_API_MODEL_NAME", "")
    monkeypatch.setattr(paddle_ocr_vl, "PADDLE_OCR_VL_API_KEY", "")

    PaddleOcrVlDocumentParser()

    assert captured["vl_rec_backend"] is None
    assert captured["merge_layout_blocks"] is False


def test_server_backend_passes_openai_compatible_service_settings(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakePaddleOCRVL:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCRVL=FakePaddleOCRVL))
    monkeypatch.setattr(paddle_ocr_vl, "PADDLE_OCR_VL_BACKEND", "llama-cpp-server")
    monkeypatch.setattr(paddle_ocr_vl, "PADDLE_OCR_VL_SERVER_URL", "http://localhost:8080/v1")
    monkeypatch.setattr(paddle_ocr_vl, "PADDLE_OCR_VL_API_MODEL_NAME", "PaddlePaddle/PaddleOCR-VL-1.5")
    monkeypatch.setattr(paddle_ocr_vl, "PADDLE_OCR_VL_API_KEY", "secret")

    PaddleOcrVlDocumentParser()

    assert captured["vl_rec_backend"] == "llama-cpp-server"
    assert captured["vl_rec_server_url"] == "http://localhost:8080/v1"
    assert captured["vl_rec_api_model_name"] == "PaddlePaddle/PaddleOCR-VL-1.5"
    assert captured["vl_rec_api_key"] == "secret"


def test_pipeline_creation_dependency_error_is_actionable(monkeypatch) -> None:
    class BrokenPaddleOCRVL:
        def __init__(self, **_kwargs) -> None:
            raise RuntimeError("A dependency error occurred during pipeline creation.")

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCRVL=BrokenPaddleOCRVL))
    monkeypatch.setattr(paddle_ocr_vl, "PADDLE_OCR_VL_BACKEND", "")
    monkeypatch.setattr(paddle_ocr_vl, "PADDLE_OCR_VL_SERVER_URL", "")

    with pytest.raises(RuntimeError, match="uv sync --python 3.12 --group dev --extra ocr --extra ocr-vl"):
        PaddleOcrVlDocumentParser()


def test_pipeline_creation_error_lists_missing_paddlex_ocr_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(paddle_ocr_vl, "_missing_paddlex_ocr_dependencies", lambda: ["einops", "tokenizers"])
    error = RuntimeError("A dependency error occurred during pipeline creation.")
    error.__cause__ = RuntimeError("PaddleX OCR extras are missing")

    message = _pipeline_creation_help("", error)

    assert "Missing PaddleX OCR dependencies: einops, tokenizers." in message
    assert "Original error: PaddleX OCR extras are missing" in message


def test_dev_backend_script_keeps_ocr_vl_dependencies_installed() -> None:
    script = Path(__file__).resolve().parents[3] / "scripts" / "dev-backend.sh"
    text = script.read_text(encoding="utf-8")

    assert "--no-build" not in text
    assert '--python "$backend_python"' in text
    assert "--extra ocr" in text
    assert "--extra ocr-vl" in text
    assert "ANKI_MAKER_REQUIRE_LOCAL_OCR" in text
    assert "BACKEND_HOST" in text
    assert "BACKEND_PORT" in text


def test_backend_python_version_is_pinned_for_paddle_wheels() -> None:
    python_version = Path(__file__).resolve().parents[2] / ".python-version"

    assert python_version.read_text(encoding="utf-8").strip() == "3.12"


def test_dev_backend_script_falls_back_to_base_ocr_when_vl_install_is_unavailable(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "dev-backend.sh"
    calls_path = tmp_path / "uv-calls.txt"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "$FAKE_UV_CALLS"
if [[ "$1" == "sync" && "$*" == *"--extra ocr-vl"* ]]; then
  echo 'error: Distribution `paddlepaddle==3.3.1` cannot be installed because it has no binary distribution' >&2
  exit 1
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    env = {
        **os.environ,
        "FAKE_UV_CALLS": str(calls_path),
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
    }

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = calls_path.read_text(encoding="utf-8")
    assert "sync --python 3.12 --group dev --extra ocr --extra ocr-vl" in calls
    assert "sync --python 3.12 --group dev --extra ocr\n" in calls
    assert "run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000" in calls
    assert "Full local OCR dependencies could not be installed" in result.stdout
    assert "Started with base PaddleOCR support only" in result.stdout


def test_dev_backend_script_can_start_without_ocr_when_base_paddle_is_unavailable(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "dev-backend.sh"
    calls_path = tmp_path / "uv-calls.txt"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "$FAKE_UV_CALLS"
if [[ "$1" == "sync" && "$*" == *"--extra ocr"* ]]; then
  echo "error: Distribution paddlepaddle has no binary distribution" >&2
  exit 1
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    env = {
        **os.environ,
        "FAKE_UV_CALLS": str(calls_path),
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
    }

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = calls_path.read_text(encoding="utf-8")
    assert "sync --python 3.12 --group dev --extra ocr --extra ocr-vl" in calls
    assert "sync --python 3.12 --group dev --extra ocr\n" in calls
    assert "sync --python 3.12 --group dev\n" in calls
    assert "Base PaddleOCR dependencies also could not be installed" in result.stdout


def test_dev_backend_script_can_require_local_ocr_install(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "dev-backend.sh"
    calls_path = tmp_path / "uv-calls.txt"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "$FAKE_UV_CALLS"
echo "error: Distribution paddlepaddle has no binary distribution" >&2
exit 1
""",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    env = {
        **os.environ,
        "ANKI_MAKER_REQUIRE_LOCAL_OCR": "1",
        "FAKE_UV_CALLS": str(calls_path),
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
    }

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 1
    calls = calls_path.read_text(encoding="utf-8")
    assert "sync --python 3.12 --group dev --extra ocr --extra ocr-vl" in calls
    assert "sync --python 3.12 --group dev\n" not in calls


def test_rejects_unsupported_vl_backend_before_pipeline_creation() -> None:
    with pytest.raises(RuntimeError, match="Unsupported PADDLE_OCR_VL_BACKEND"):
        _normalized_vl_backend("native-but-not-really")
